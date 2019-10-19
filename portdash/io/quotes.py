"""Retrieve information about assets from external sources
"""
from datetime import datetime
import logging
import os
from typing import Tuple, Union

import pandas as pd

from portdash.io._alphavantage import fetch_from_web, InvalidAPICall

DEFAULT_QUOTE_SOURCE = 'alphavantage'
VALID_QUOTE_SOURCES = ['alphavantage', 'csv', 'const']

log = logging.getLogger(__name__)


def _split_source_str(source: str) -> Tuple[str, Union[str, None]]:
    if not source:
        return DEFAULT_QUOTE_SOURCE, None
    tokens = source.split(':')
    if len(tokens) == 1:
        return tokens[0], None
    elif len(tokens) == 2:
        return tokens[0], tokens[1]
    else:
        raise ValueError(f"Unrecognized source string: {source}")


def fetch(source: str, symbol: str, start_time: datetime=None) -> pd.DataFrame:
    """Return a table of historical security valuations

    Use the `source` string to dispatch the lookup to the appropriate source.

    Parameters
    ----------
    source : str
        The source of the quotes, formatted as "[source]:[args", e.g.
        "csv:/path/to/file".
    symbol : str
        The stock ticker symbol
    start_time : datetime, optional
        If supplied, start the output table at this date.
        The default will return all available historical quotes.

    Returns
    -------
    pd.DataFrame
        A table of historical quotes, indexed by the datetime of the quote.
        The index name will be "date", and the table will have at least
        columns named "price", "dividend_amount", and "volume".
    """
    source_name, source_arg = _split_source_str(source)
    if source_name == 'alphavantage':
        try:
            return (fetch_from_web(symbol, start_time=start_time)
                    .rename(columns={'close': 'price'}))
        except InvalidAPICall:
            log.exception(f'Unable to fetch quotes for {symbol}')
        return fetch_from_web(symbol=symbol, start_time=start_time)
    elif source_name == 'const':
        log.debug(f"Filling {symbol} with constant "
                  f"quotes values of {source_arg}.")
        return _fetch_from_const(float(source_arg), start_time=start_time)
    elif source_name == 'csv':
        log.debug(f"Reading {symbol} values from {source_arg}.")
        return _fetch_from_csv(filename=source_arg, start_time=start_time)
    else:
        raise ValueError(f"Unknown source: {source}. Source name must be "
                         f"one of {VALID_QUOTE_SOURCES}.")


def _fetch_from_const(value: float, start_time: datetime) -> pd.DataFrame:
    """Return a table of historical security valuations, all with
    the same constant quote value and no distributions.

    Parameters
    ----------
    value : float
        Assign this value to the security at every date.
    start_time : datetime, optional
        If supplied, start the output table at this date.
        The default will return all available historical quotes.

    Returns
    -------
    pd.DataFrame
        A table of historical quotes, indexed by the datetime of the quote
    """
    index = pd.date_range(start_time or '2015-01-01', datetime.today(),
                          freq='D').rename('date')
    return pd.DataFrame({'price': value, 'volume': 0.,
                         'dividend_amount': 0.}, index=index)


def _fetch_from_csv(filename: str, start_time: datetime) -> pd.DataFrame:
    """Return a table of historical security valuations read from a CSV.

    Parameters
    ----------
    filename : str
        Path to the csv with the historical quote data.
    start_time : datetime, optional
        If supplied, start the output table at this date.
        The default will return all available historical quotes.

    Returns
    -------
    pd.DataFrame
        A table of historical quotes, indexed by the datetime of the quote
    """
    df = pd.read_csv(os.path.expanduser(filename), index_col=0,
                     parse_dates=True, infer_datetime_format=True)
    df = df.rename(columns={'close': 'price'})
    df.index.name = 'date'
    if start_time:
        df = df.loc[df.index >= start_time]
    return df
