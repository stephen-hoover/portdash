"""Fetch quotes from web APIs
"""
from datetime import datetime, timedelta, time
import logging
from typing import Iterable, Sequence, Union

import pandas as pd

from portdash.models import Distribution, Quote
from portdash.apis._alphavantage import AlphaVantageClient
from config import conf

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
    return datetime.combine(Quote.most_recent().date, time.min)


def refresh_quotes(all_symbols: Iterable[str]=None,
                   skip_downloads: Iterable[str]=None):
    skip_downloads = skip_downloads or []
    if all_symbols is None:
        log.info('Reading all quotes in database.')
        query = Quote.query.with_entities(Quote.symbol).distinct()
        all_symbols = [row.symbol for row in query.all()]
    if skip_downloads:
        log.info(f"Don't try to download quotes for {skip_downloads}")
    symbols_to_download = set(all_symbols) - set(skip_downloads)

    for symbol in symbols_to_download:
        last_quote = getattr(Quote.most_recent(symbol), 'date', None)
        if last_quote:
            last_quote = datetime.combine(last_quote, time.min)
            quotes_are_stale = (datetime.today() - last_quote >
                                timedelta(days=1))
        else:
            quotes_are_stale = True

        if quotes_are_stale:
            _start = (None if last_quote is None else
                      last_quote + timedelta(days=1))
            log.debug(f"Fetching new quotes for {symbol}. "
                      f"Last quote: {last_quote}.")
            new_quotes = (_fetch_from_web(symbol, start_time=_start)
                          .rename(columns={'close': 'price',
                                           'dividend_amount': 'amount'}))
            Quote.insert(new_quotes, symbol=symbol)
            Distribution.insert(new_quotes, symbol=symbol)


def _fetch_from_web(symbol: str, start_time: datetime) -> pd.DataFrame:
    log.info(f'Reading {symbol} data from Alpha Vantage.')
    client = AlphaVantageClient(conf('av_api_key'))
    new_quotes = client.historical_quotes(symbol, start_time=start_time)

    return new_quotes
