"""This module provides an interface for the application to get
information about prices and dividends for securities.
"""
from datetime import datetime, timedelta, time
import logging
from typing import Iterable, Sequence, Union

import pandas as pd

from portdash.extensions import db
from portdash.models import Distribution, Quote, Security
from portdash.io import symbol_lookup, UnknownSymbol
from portdash.io.quotes import fetch

log = logging.getLogger(__name__)


def get_price(symbol: str,
              index: Union[pd.DatetimeIndex,
                           Sequence[pd.datetime]]) -> pd.Series:
    """Fetch prices for a given symbol in the provided date range"""
    this_quote = Quote.get_price(symbol, index)

    if this_quote is None:
        log.error(f"Couldn't find a price for {symbol}. "
                  f"Assuming its value is zero.")
        price = pd.Series(0., index=index)
    else:
        price = this_quote
    return price


def get_dividend(symbol: str,
                 index: pd.DatetimeIndex=None) -> pd.Series:
    """Fetch dividends from a given symbol in the provided date range"""
    qu = Distribution.get_amount(symbol, index)
    qu.index.name = ""
    return qu


def get_max_date() -> datetime:
    """Return the datetime of the most recent quote"""
    return datetime.combine(Quote.most_recent().date, time.min)


def refresh_quotes(all_symbols: Iterable[str]=None,
                   skip_downloads: Iterable[str]=None):
    """Update all quotes and distributions

    If given a symbol which is not already in the database, download
    metadata on it, if possible, and add it to the database.

    Parameters
    ----------
    all_symbols:
        Update quotes and dividends for these securities if they're already
        in the database, or download all available historical data if the
        securities aren't in the database. Default to updating everything
        currently in the database.
    skip_downloads:
        If provided, don't try to update these securities. Most useful
        when updating all securities in the database.
    """
    skip_downloads = skip_downloads or []
    if all_symbols is None:
        log.info('Reading all quotes in database.')
        query = Quote.query.with_entities(Quote.symbol).distinct()
        all_symbols = [row.symbol for row in query.all()]
    if skip_downloads:
        log.info(f"Don't try to download quotes for {skip_downloads}")
    symbols_to_download = set(all_symbols) - set(skip_downloads)

    for symbol in symbols_to_download:
        # Determine if we need to update, based on the most recent data.
        # We won't get quotes more often than daily.
        sec = Security.query.get(symbol)
        if not sec:
            sec = _new_security(symbol)
        if sec.last_updated:
            quotes_are_stale = (datetime.today() - sec.last_updated >
                                timedelta(days=1))
        else:
            quotes_are_stale = True

        # Retrieve new quotes and insert into the database.
        if quotes_are_stale:
            last_quote = getattr(Quote.most_recent(symbol), 'date', None)
            if last_quote:
                last_quote = datetime.combine(last_quote, time.max)
            _start = (None if last_quote is None else
                      last_quote + timedelta(days=1))
            log.debug(f"Fetching new quotes for {symbol}. "
                      f"Last quote: {last_quote}.")

            new_quotes = fetch(sec.quote_source, symbol, start_time=_start)
            Quote.insert(new_quotes, symbol=symbol)
            Distribution.insert(new_quotes, symbol=symbol)
            _reset_last_updated(sec)
        else:
            log.debug(f'{symbol} quotes are up to date.')


def _reset_last_updated(sec: Security):
    """Mark a given Security as last updated now."""
    try:
        sec.last_updated = datetime.today()
        db.session.commit()
        log.debug(f'Set last_updated to {sec.last_updated} for {sec.symbol}')
    except Exception:
        log.exception(f"Unable to update last_updated field for {sec.symbol}.")
        db.session.rollback()


def _new_security(symbol: str) -> Security:
    """Insert a new Security in the database, using type and name data
    retrieved from the web API.
    """
    try:
        sec_data = symbol_lookup(symbol)
    except UnknownSymbol:
        log.warning(f"Unable to find {symbol} data.")
        sec_data = {'type': None, 'name': symbol}
    sec = Security(symbol=symbol, type=sec_data['type'], name=sec_data['name'])
    try:
        db.session.add(sec)
        db.session.commit()
        log.debug(f'Added {sec.symbol} to the database.')
    except Exception:
        log.exception(f"Unable to add new security {symbol} to the database.")
        db.session.rollback()
    return sec
