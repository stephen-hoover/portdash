from datetime import datetime, timedelta
from glob import glob
import logging
import os
import time
from typing import Iterable

import pandas as pd

from portdash import config

log = logging.getLogger(__name__)


# Get historical prices from Alpha Vantage's API
AV_API = ('https://www.alphavantage.co/query?'
          'function=TIME_SERIES_DAILY_ADJUSTED'
          '&symbol={symbol}'
          '&outputsize=full'
          '&datatype=csv'
          '&apikey={api_key}')


def fetch_quotes(symbol, refresh_cache=False, retry_errored_cache=False):
    fname = os.path.join(config.CACHE_DIR, f'{symbol}.csv')
    reader_kwargs = {'index_col': 0, 'parse_dates': True,
                     'infer_datetime_format': True}
    quotes = None
    if os.path.exists(fname):
        log.info('Reading %s data from cache.', symbol)
        quotes = pd.read_csv(fname, **reader_kwargs)
        if quotes.index.name == '{' and retry_errored_cache:
            quotes = None
        elif datetime.today() - quotes.index[0] < timedelta(days=1):
            # If these quotes are current, then we don't need to re-download
            return quotes

    if refresh_cache or quotes is None:
        log.info('Reading %s data from Alpha Vantage.', symbol)
        new_quotes = pd.read_csv(AV_API.format(symbol=symbol,
                                               api_key=config.AV_API_KEY),
                                 **reader_kwargs)
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


def fetch_all_quotes(symbols, refresh_cache=False, retry_errored_cache=False):
    failed = []
    quotes = {}
    for symbol in symbols:
        try:
            quotes[symbol] = fetch_quotes(
                symbol, refresh_cache=refresh_cache,
                retry_errored_cache=retry_errored_cache)
            time.sleep(15)  # Don't exceed API rate limit
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


def get_price(symbol, quotes, index):
    price = (quotes[symbol]['close']
             .reindex(index, method='ffill')
             .fillna(method='ffill')
             .fillna(method='bfill'))
    return price


def cached_symbols(cache_dir: str=config.CACHE_DIR):
    return map(lambda x: os.path.splitext(x)[0],
               map(os.path.basename, glob(os.path.join(cache_dir, '*.csv'))))


def read_all_quotes(all_symbols: Iterable[str]=None,
                    skip_downloads: Iterable[str]=config.SKIP_SYMBOL_DOWNLOADS,
                    refresh_cache: bool=False):
    if all_symbols is None:
        log.info('Reading all quotes in cache directory.')
        all_symbols = cached_symbols()
    log.info(f"Don't try to download quotes for {skip_downloads}")
    symbols_to_download = set(all_symbols) - set(skip_downloads)

    quotes = fetch_all_quotes(symbols_to_download, refresh_cache=refresh_cache,
                              retry_errored_cache=True)
    cache_quotes = fetch_all_quotes(skip_downloads, refresh_cache=False)
    return {**quotes, **cache_quotes}

