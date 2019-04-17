from datetime import datetime, timedelta
from glob import glob
import logging
import os
import time
from typing import Dict, Iterable

import pandas as pd

from portdash.config import conf

log = logging.getLogger(__name__)


# Get historical prices from Alpha Vantage's API
AV_API = ('https://www.alphavantage.co/query?'
          'function=TIME_SERIES_DAILY_ADJUSTED'
          '&symbol={symbol}'
          '&outputsize=full'
          '&datatype=csv'
          '&apikey={api_key}')


def fetch_quotes(symbol: str,
                 refresh_cache: bool=False,
                 retry_errored_cache: bool=False,
                 api_delay: float=0) -> pd.DataFrame:
    fname = os.path.join(conf('cache_dir'), f'{symbol}.csv')
    os.makedirs(conf('cache_dir'), exist_ok=True)
    reader_kwargs = {'index_col': 0, 'parse_dates': True,
                     'infer_datetime_format': True}
    quotes = None
    if os.path.exists(fname):
        log.info(f'Reading {symbol} data from cache.')
        quotes = pd.read_csv(fname, **reader_kwargs)
        if quotes.index.name == '{' and retry_errored_cache:
            quotes = None
        elif datetime.today() - quotes.index[0] < timedelta(days=1):
            # If these quotes are current, then we don't need to re-download
            return quotes

    if refresh_cache or quotes is None:
        log.info(f'Reading {symbol} data from Alpha Vantage.')
        new_quotes = pd.read_csv(AV_API.format(symbol=symbol,
                                               api_key=conf('av_api_key')),
                                 **reader_kwargs)
        if api_delay:
            time.sleep(api_delay)  # Don't exceed API rate limit
        if new_quotes.index.name == '{':
            # This is an error message.
            raise ValueError(f'Error fetching "{symbol}": '
                             f'{new_quotes.index[0].strip()}')
        if quotes is not None:
            new_rows = new_quotes[new_quotes.index > quotes.index.max()]
            quotes = (quotes
                      .append(new_rows, verify_integrity=True)
                      .sort_index(ascending=False))
        else:
            quotes = new_quotes
        quotes.to_csv(fname, index=True, header=True)
    if quotes.index.name == '{':
        # This is an error message.
        raise ValueError(f'Error fetching "{symbol}": '
                         f'{quotes.index[0].strip()}')
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

