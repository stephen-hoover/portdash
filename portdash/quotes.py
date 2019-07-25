from datetime import datetime, timedelta
from functools import lru_cache
from glob import glob
import io
import logging
import os
import time
from typing import Dict, Iterable, Sequence, Tuple, Union

import pandas as pd
import requests

from config import conf

log = logging.getLogger(__name__)

QUOTE_COLS = ['timestamp', 'open', 'high', 'low', 'close', 'adjusted_close',
              'volume', 'dividend_amount', 'split_coefficient']
CSV_READER_KWARGS = {'index_col': 0, 'parse_dates': True,
                     'infer_datetime_format': True}
MIN_TIME = datetime(year=1970, month=1, day=1)


class APICallsExceeded(RuntimeError):
    pass


class InvalidAPICall(RuntimeError):
    pass


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class AlphaVantageClient(metaclass=Singleton):
    """Fetch quotes from Alpha Vantage

    Respect the Alpha Vantage rate limit on API requests.
    Check the response for error messages and raise if found.

    This API client implementation only covers the API functions which
    are needed in this library.

    This object is a singleton so that it can maintain a record of
    the next allowed time to query the API. Note that since there's only
    one AlphaVantageClient object, multiple clients with different
    API keys aren't supported.

    Parameters
    ----------
    api_key : str
        To use this object, supply an API key from Alpha Vantage:
        https://www.alphavantage.co/support/#api-key
    """
    # Get historical prices from Alpha Vantage's API
    AV_API = ('https://www.alphavantage.co/query?'
              'function=TIME_SERIES_DAILY_ADJUSTED'
              '&symbol={symbol}'
              '&outputsize={size}'
              '&datatype=csv'
              '&apikey={api_key}')

    # Set limits for use of Alpha Vantage's free API
    max_per_day = 500  # Maximum number of API queries per day
    min_interval = 12  # Min number of seconds between queries

    # Errors in API calls will return a 200, but have the following text
    # in place of the anticipated contents. (This is a partial string.)
    error_msg = b'Error Message'

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._last_query = MIN_TIME
        self._n_queries = 0

    def _wait_until_available(self):
        if self._n_queries >= AlphaVantageClient.max_per_day:
            raise APICallsExceeded(f"You've reached the maximum of "
                                   f"{AlphaVantageClient.max_per_day} "
                                   f"daily API calls.")

        while ((datetime.now() - self._last_query).total_seconds() <
                AlphaVantageClient.min_interval):
            time.sleep(0.2)

    @staticmethod
    def _check_and_raise(response: requests.Response, symbol: str):
        response.raise_for_status()
        if AlphaVantageClient.error_msg in response.content[:200]:
            msg = response.json()['Error Message']
            raise InvalidAPICall(f'Error fetching {symbol}: {msg}')

    def historical_quotes(self, symbol: str,
                          start_time: datetime=None) -> pd.DataFrame:
        """Return a table of historical security valuations

        Parameters
        ----------
        symbol : str
            The stock ticker symbol
        start_time : datetime, optional
            If supplied, start the output table at this date.
            Supplying a recent date will allow us to request fewer
            rows returned from the Alpha Vantage service.
            The default will return all available historical quotes.

        Returns
        -------
        pd.DataFrame
            A table of historical quotes, indexed by the datetime of the quote
        """
        if start_time is not None and start_time >= datetime.today():
            log.debug('No update needed; the requested start '
                      'date is in the future.')
            pd.DataFrame(columns=QUOTE_COLS)
        self._wait_until_available()
        if start_time is None:
            start_time = MIN_TIME

        if (datetime.now() - start_time).total_seconds() / 86400 < 100:
            size = 'compact'  # Return latest 100 data points
            log.debug("Making compact query")
        else:
            size = 'full'  # Return full time series
            log.debug("Making full query")
        web_addr = AlphaVantageClient.AV_API.format(
            symbol=symbol, size=size, api_key=conf('av_api_key'))
        self._last_query = datetime.now()
        self._n_queries += 1
        response = requests.get(web_addr)
        self._check_and_raise(response, symbol)

        quotes = pd.read_csv(io.BytesIO(response.content), **CSV_READER_KWARGS)
        return quotes[quotes.index >= start_time]


@lru_cache(1024)
def _memo_from_cache(fname: str, mod_tm: float) \
      -> Tuple[pd.DataFrame, datetime]:
    """Memoize reads from disk. Use time since file modification to
    invalidate previous reads"""
    log.debug(f'Reading from cache {fname}.')
    quotes = pd.read_csv(fname, **CSV_READER_KWARGS)
    if _is_error(quotes):
        log.debug(f'{fname} is an error message.')
        last_quote_time = None
    else:
        last_quote_time = quotes.index[0]
    return quotes, last_quote_time


def _fetch_from_cache(fname: str) -> Tuple[pd.DataFrame, datetime]:
    """Read the cached value of a security over time.

    If the file hasn't changed since the last time we read it,
    we can use the in-memory cache and skip re-reading from disk.
    """
    return _memo_from_cache(fname, os.path.getmtime(fname))


def _fetch_from_web(symbol: str, start_time: datetime) -> pd.DataFrame:
    log.info(f'Reading {symbol} data from Alpha Vantage.')
    client = AlphaVantageClient(conf('av_api_key'))
    new_quotes = client.historical_quotes(symbol, start_time=start_time)

    return new_quotes


def _update_cache(quotes: Union[pd.DataFrame, None],
                  new_quotes: pd.DataFrame,
                  fname: str) \
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
                 retry_errored_cache: bool=False) -> pd.DataFrame:
    fname = os.path.join(conf('cache_dir'), f'{symbol}.csv')
    quotes, last_quote = None, None
    if os.path.exists(fname):
        quotes, last_quote = _fetch_from_cache(fname)
        # If the quotes are current, we don't need to re-download.
        quotes_are_stale = (datetime.today() - last_quote > timedelta(days=1))
        refresh_cache = refresh_cache and quotes_are_stale
        if retry_errored_cache and _is_error(quotes):
            quotes = None

    if refresh_cache or quotes is None:
        _start = None if last_quote is None else last_quote + timedelta(days=1)
        new_quotes = _fetch_from_web(symbol, start_time=_start)
        quotes = _update_cache(quotes, new_quotes, fname)

    if _is_error(quotes):
        _raise_quote_error(quotes, symbol)

    return quotes


def fetch_all_quotes(symbols: Iterable[str],
                     refresh_cache: bool=False,
                     retry_errored_cache: bool=False) -> \
      Dict[str, pd.DataFrame]:
    failed = []
    quotes = {}
    for symbol in symbols:
        try:
            quotes[symbol] = fetch_quotes(
                symbol, refresh_cache=refresh_cache,
                retry_errored_cache=retry_errored_cache)
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
              index: Union[pd.DatetimeIndex, Sequence[pd.datetime]],
              quotes: Dict[str, pd.DataFrame]=None) -> pd.Series:
    """Fetch prices for a given symbol in the provided date range"""
    if quotes is not None:
        this_quote = quotes.get(symbol)
    else:
        try:
            this_quote = fetch_quotes(symbol)
        except ValueError:
            log.exception(f"Unable to get price for {symbol}. "
                          f"Assuming price is zero.")
            this_quote = None

    if this_quote is None:
        log.error(f"Couldn't find a price for {symbol}. "
                  f"Assuming its value is zero.")
        price = pd.Series(0., index=index)
    else:
        # Use "ffill" to take us through e.g. weekends. Use bfill to
        # make sure that we have valid prices at the beginning of the series.
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


def make_dummy_prices(symbol: str, prototype_symbol: str, value: float = 1):
    """Read the quotes for the prototype symbol, and replace all
    prices with the `value`.

    I can run it by hand once, then be done with it.
    """
    fname_prototype = os.path.join(conf('cache_dir'), f'{prototype_symbol}.csv')
    fname_new = os.path.join(conf('cache_dir'), f'{symbol}.csv')
    reader_kwargs = {'index_col': 0, 'parse_dates': True,
                     'infer_datetime_format': True}
    prototype_quotes = pd.read_csv(fname_prototype, **reader_kwargs)

    set_to_value = ['open', 'high', 'low', 'close', 'adjusted_close']
    set_to_zero = ['volume', 'dividend_amount', 'split_coefficient']
    remainder = set(prototype_quotes) - set(set_to_value) - set(set_to_zero)
    if remainder:
        raise RuntimeError(f"Unable to reset columns {remainder}")
    for col in set_to_value:
        prototype_quotes[col] = value
    for col in set_to_zero:
        prototype_quotes[col] = 0.

    log.info("Wrote file to %s", fname_new)
    prototype_quotes.to_csv(fname_new, index=True, header=True)
