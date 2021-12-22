"""Interface to the AlphaVantage API
"""
from datetime import datetime
import io
import logging
import re
import time
from typing import Dict

import pandas as pd
import requests

from config import conf

__all__ = [
    "AlphaVantageClient",
    "fetch_from_web",
    "APICallsExceeded",
    "InvalidAPICall",
    "UnknownSymbol",
    "symbol_lookup",
]
log = logging.getLogger(__name__)

# These are the columns guaranteed to be returned by requests for quotes.
QUOTE_INDEX = "timestamp"
QUOTE_COLS = [
    "close",
    "volume",
    "dividend_amount",
]
CSV_READER_KWARGS = {"index_col": 0, "parse_dates": True, "infer_datetime_format": True}
MIN_TIME = datetime(year=1970, month=1, day=1)


class APICallsExceeded(RuntimeError):
    pass


class InvalidAPICall(RuntimeError):
    pass


class UnknownSymbol(RuntimeError):
    def __init__(self, symbol, matches):

        # Each key in the blobs is prefixed by a number and a space,
        # e.g. "2. type". Strip the number.
        self.matches = [
            {" ".join(k.split()[1:]): v for k, v in obj.items()} for obj in matches
        ]
        self.symbol = symbol
        msg = f"Unable to find match for {symbol}. " f"Best matches: \n{self.matches}"
        super().__init__(msg)


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
    # The "weekly adjusted" endpoint includes dividend information
    # and adjusts share prices for split events, but only gives
    # prices at the end of each week.
    _AV_API = {
        "prices": (
            "https://www.alphavantage.co/query?"
            "function=TIME_SERIES_DAILY"
            "&symbol={symbol}"
            "&outputsize={size}"
            "&datatype=csv"
            "&apikey={api_key}"
        ),
        "dividends": (
            "https://www.alphavantage.co/query?"
            "function=TIME_SERIES_WEEKLY_ADJUSTED"
            "&symbol={symbol}"
            "&datatype=csv"
            "&apikey={api_key}"
        ),
    }

    # Set limits for use of Alpha Vantage's free API
    max_per_day = 500  # Maximum number of API queries per day
    min_interval = 12  # Min number of seconds between queries

    # Errors in API calls will return a 200, but have the following text
    # in place of the anticipated contents. (This is a partial string.)
    error_msg = b"Error Message"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._last_query = MIN_TIME
        self._n_queries = 0

    def _wait_until_available(self):
        if self._n_queries >= AlphaVantageClient.max_per_day:
            raise APICallsExceeded(
                f"You've reached the maximum of "
                f"{AlphaVantageClient.max_per_day} "
                f"daily API calls."
            )

        while (
            datetime.now() - self._last_query
        ).total_seconds() < AlphaVantageClient.min_interval:
            time.sleep(0.2)

    @staticmethod
    def _check_and_raise(response: requests.Response, symbol: str):
        response.raise_for_status()
        if AlphaVantageClient.error_msg in response.content[:200]:
            msg = response.json()["Error Message"]
            # Include the web API call in the error msg, but mask the API key
            addr = re.sub(r"&apikey=[^&]+", r"&apikey=****", response.url)
            raise InvalidAPICall(f"Error fetching {symbol} from {addr}: {msg}")

    def _query(self, web_addr, symbol):
        """Handle all API interactions through here so we can keep track of
        how often we're hitting the API.
        """
        self._wait_until_available()
        self._last_query = datetime.now()
        self._n_queries += 1
        response = requests.get(web_addr)
        self._check_and_raise(response, symbol)
        return response

    def _fetch_data(self, symbol: str, size: str, type: str) -> pd.DataFrame:
        web_addr = AlphaVantageClient._AV_API[type].format(
            symbol=symbol, size=size, api_key=self._api_key
        )
        response = self._query(web_addr, symbol)

        df = pd.read_csv(io.BytesIO(response.content), **CSV_READER_KWARGS)
        df.columns = [c.replace(" ", "_") for c in df.columns]
        return df

    def historical_quotes(
        self,
        symbol: str,
        start_time: datetime = None,
        all_columns: bool = False,
        return_dividends: bool = False,
    ) -> pd.DataFrame:
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
        all_columns : bool, optional
            If True, return all columns provided by the AlphaVantage API.
            Otherwise, return only guaranteed columns.
        return_dividends : bool, optional
            If True, include the "dividend_amount" column in the output.
            If False, do not. Skipping the dividend column will save one API call
            if it's not needed, and will be considerably faster on compact mode.

        Returns
        -------
        pd.DataFrame
            A table of historical quotes, indexed by the datetime of the quote
        """
        if start_time is not None and start_time >= datetime.today():
            log.debug("No update needed; the requested start date is in the future.")
            return pd.DataFrame(
                columns=QUOTE_COLS, index=pd.Index([], name=QUOTE_INDEX)
            )
        if start_time is None:
            start_time = MIN_TIME

        if (datetime.now() - start_time).total_seconds() / 86400 < 100:
            size = "compact"  # Return latest 100 data points
            log.debug("Making compact query")
        else:
            size = "full"  # Return full time series
            log.debug("Making full query")

        quotes = self._fetch_data(symbol=symbol, size=size, type="prices")
        if return_dividends:
            dividends = self._fetch_data(symbol=symbol, size=size, type="dividends")

            # Merge weekly dividend data into daily quotes
            quotes["dividend_amount"] = dividends["dividend_amount"]
            quotes["dividend_amount"] = quotes["dividend_amount"].fillna(0)
        else:
            quotes["dividend_amount"] = pd.NA

        if not all_columns:
            quotes = quotes[QUOTE_COLS]

        quotes.index.name = QUOTE_INDEX
        return quotes[quotes.index >= start_time]

    def symbol_lookup(self, symbol: str) -> Dict[str, str]:
        """Look up information about a symbol from AlphaVantage.

        See https://www.alphavantage.co/documentation/#symbolsearch

        Parameters
        ----------
        symbol:
            The symbol to retrieve information on.

        Returns
        -------
        dict
            A dictionary which includes keys "symbol", "name", "type",
            "region", and "currency".
        """
        lookup_api = (
            f"https://www.alphavantage.co/query?"
            f"function=SYMBOL_SEARCH"
            f"&keywords={symbol}&apikey={self._api_key}"
        )
        resp = self._query(lookup_api, symbol)

        # The API response is JSON with a "bestMatches" key holding a
        # list of blobs. We're looking for an exact match on the input symbol.
        data = [obj for obj in resp.json()["bestMatches"] if obj["1. symbol"] == symbol]
        if not data:
            raise UnknownSymbol(symbol, resp.json()["bestMatches"])

        # Each key in the response is prefixed by a number and a space,
        # e.g. "2. type". Strip the number.
        return {" ".join(k.split()[1:]): v for k, v in data[0].items()}


def fetch_from_web(symbol: str, start_time: datetime = None) -> pd.DataFrame:
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
    log.info(f"Reading {symbol} data from Alpha Vantage.")
    client = AlphaVantageClient(conf("av_api_key"))
    new_quotes = client.historical_quotes(
        symbol, start_time=start_time, return_dividends=True
    )
    new_quotes.index.name = "date"

    return new_quotes


def symbol_lookup(symbol: str) -> Dict[str, str]:
    """Look up information about a symbol from AlphaVantage.

    See https://www.alphavantage.co/documentation/#symbolsearch

    Parameters
    ----------
    symbol:
        The symbol to retrieve information on.

    Returns
    -------
    dict
        A dictionary which includes keys "symbol", "name", "type",
        "region", and "currency".
    """
    log.info(f"Reading {symbol} data from Alpha Vantage.")
    client = AlphaVantageClient(conf("av_api_key"))

    return client.symbol_lookup(symbol)
