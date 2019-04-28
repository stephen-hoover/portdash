from datetime import datetime, timedelta
from functools import lru_cache
from glob import glob
import logging
import os
import time
from typing import Dict, Iterable, Tuple, Union

import pandas as pd

from portdash.config import conf

log = logging.getLogger(__name__)

CSV_READER_KWARGS = {'index_col': 0, 'parse_dates': True,
                     'infer_datetime_format': True}

# Get historical prices from Alpha Vantage's API
AV_API = ('https://www.alphavantage.co/query?'
          'function=TIME_SERIES_DAILY_ADJUSTED'
          '&symbol={symbol}'
          '&outputsize=full'
          '&datatype=csv'
          '&apikey={api_key}')


@lru_cache(1024)
def _memo_from_cache(fname: str, mod_tm: float) \
      -> Tuple[pd.DataFrame, bool]:
    """Memoize reads from disk. Use time since file modification to
    invalidate previous reads"""
    log.debug(f'Reading from cache {fname}.')
    quotes = pd.read_csv(fname, **CSV_READER_KWARGS)
    if _is_error(quotes):
        current = False
    else:
        # If these quotes are current, then we don't need to re-download
        current = (datetime.today() - quotes.index[0] < timedelta(days=1))
    return quotes, current


def _fetch_from_cache(fname: str) -> Tuple[pd.DataFrame, bool]:
    """Read the cached value of a security over time.

    If the file hasn't changed since the last time we read it,
    we can use the in-memory cache and skip re-reading from disk.
    """
    return _memo_from_cache(fname, os.path.getmtime(fname))


def _fetch_from_web(symbol: str, api_delay: float=0) -> pd.DataFrame:
    log.info(f'Reading {symbol} data from Alpha Vantage.')
    new_quotes = pd.read_csv(AV_API.format(symbol=symbol,
                                           api_key=conf('av_api_key')),
                             **CSV_READER_KWARGS)
    if api_delay:
        time.sleep(api_delay)  # Don't exceed API rate limit
    if _is_error(new_quotes):
        _raise_quote_error(new_quotes, symbol)

    return new_quotes


def _update_cache(quotes: Union[pd.DataFrame, None],
                  new_quotes: pd.DataFrame, fname: str) \
      -> pd.DataFrame:
    if quotes is not None and not _is_error(quotes):
        new_rows = new_quotes[new_quotes.index > quotes.index.max()]
        quotes = (quotes
                  .append(new_rows, verify_integrity=True)
                  .sort_index(ascending=False))
    else:
        quotes = new_quotes
    os.makedirs(conf('cache_dir'), exist_ok=True)
    quotes.to_csv(fname, index=True, header=True)
    return quotes


def _is_error(quotes: pd.DataFrame) -> bool:
    """Check for errors in the quote download"""
    return quotes.index.name == '{'


def _raise_quote_error(quotes: pd.DataFrame, symbol: str):
    raise ValueError(f'Error fetching "{symbol}": '
                     f'{quotes.index[0].strip()}')


def fetch_quotes(symbol: str,
                 refresh_cache: bool=False,
                 retry_errored_cache: bool=False,
                 api_delay: float=0) -> pd.DataFrame:
    fname = os.path.join(conf('cache_dir'), f'{symbol}.csv')
    quotes = None
    if os.path.exists(fname):
        quotes, quotes_are_current = _fetch_from_cache(fname)
        # If the quotes are current, we don't need to re-download.
        refresh_cache = refresh_cache and (not quotes_are_current)
        if retry_errored_cache and _is_error(quotes):
            quotes = None

    if refresh_cache or quotes is None:
        new_quotes = _fetch_from_web(symbol, api_delay=api_delay)
        quotes = _update_cache(quotes, new_quotes, fname)

    if _is_error(quotes):
        _raise_quote_error(quotes, symbol)

    return quotes


def fetch_all_quotes(symbols: Iterable[str],
                     refresh_cache: bool=False,
                     retry_errored_cache: bool=False) -> Dict[str, pd.DataFrame]:
    failed = []
    quotes = {}
    for symbol in symbols:
        try:
            quotes[symbol] = fetch_quotes(
                symbol, refresh_cache=refresh_cache,
                retry_errored_cache=retry_errored_cache,
                api_delay=15)
        except ValueError as err:
            if str(err).startswith('Error fetching'):
                failed.append(symbol)
                log.error(str(err))
            else:
                raise
    log.info(f'Read {len(quotes)} historical price time series.')
    if failed:
        log.info(f'Failed to fetch data for the following {len(failed)} '
                 f'symbols: {failed}')
    return quotes


def get_price(symbol: str,
              index: pd.DatetimeIndex,
              quotes: Dict[str, pd.DataFrame]=None) -> pd.Series:
    """Fetch prices for a given symbol in the provided date range"""
    if quotes is not None:
        this_quote = quotes[symbol]
    else:
        this_quote = fetch_quotes(symbol)
    price = (this_quote['close']
             .reindex(index, method='ffill')
             .fillna(method='ffill')
             .fillna(method='bfill'))
    return price


def get_dividend(symbol: str,
                 index: pd.DatetimeIndex=None,
                 start: pd.Timestamp=None,
                 end: pd.Timestamp=None,
                 quotes: Dict[str, pd.DataFrame]=None) -> pd.Series:
    """Fetch dividends from a given symbol in the provided date range"""
    qu = quotes[symbol] if quotes else fetch_quotes(symbol)
    if index is not None:
        start = index.min()
        end = index.max()
    if start is None or end is None:
        raise TypeError('Provide either an index or start and end times.')
    if 'dividend_amount' not in qu:
        raise ValueError('Provide a quote dictionary with a '
                         '"dividend_amount" column.')
    qu = qu[(qu.index >= start) & (qu.index <= end)]
    qu = qu[qu['dividend_amount'] != 0]['dividend_amount']
    qu.index.name = ""
    return qu


def _cached_symbols(cache_dir: str=None):
    if not cache_dir:
        cache_dir = conf('cache_dir')
    return map(lambda x: os.path.splitext(x)[0],
               map(os.path.basename, glob(os.path.join(cache_dir, '*.csv'))))


def read_all_quotes(all_symbols: Iterable[str]=None,
                    skip_downloads: Iterable[str]=None,
                    refresh_cache: bool=False) -> Dict[str, pd.DataFrame]:
    skip_downloads = skip_downloads or []
    if all_symbols is None:
        log.info('Reading all quotes in cache directory.')
        all_symbols = _cached_symbols()
    if skip_downloads:
        log.info(f"Don't try to download quotes for {skip_downloads}")
    symbols_to_download = set(all_symbols) - set(skip_downloads)

    quotes = fetch_all_quotes(symbols_to_download, refresh_cache=refresh_cache,
                              retry_errored_cache=True)
    cache_quotes = fetch_all_quotes(skip_downloads, refresh_cache=False)
    return {**quotes, **cache_quotes}

