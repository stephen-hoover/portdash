"""Read transaction logs from AceMoney
"""
import argparse
from glob import glob
import logging
import os
import pickle
import time
from typing import Dict, Iterable

import pandas as pd
import numpy as np

from portdash import config

log = logging.getLogger(__name__)

# Get historical prices from Alpha Vantage's API
av_api = ('https://www.alphavantage.co/query?'
          'function=TIME_SERIES_DAILY_ADJUSTED'
          '&symbol={symbol}'
          '&outputsize=full'
          '&datatype=csv'
          '&apikey={api_key}')


def init_portfolio(index):
    """A `portfolio` DataFrame has two columns for each investment,
    one for total number of shares at any particular time and
    another for total distributions from each investment.
    There's also several global / aggregate columns, initialized here."""
    return pd.DataFrame({'_total': 0., '_total_dist': 0., 'cash': 0.,
                         'contributions': 0., 'withdrawals': 0.}, index=index)


def record_action(portfolio, action, quotes):
    """Mark the results of an investment transaction in a portfolio

    `portfolio`: pd.DataFrame
        Initialized as per `init_portfolio`
    `action`: pd.Series
        A row from the all-investment-transactions AceMoney export
    """
    if action.Symbol not in portfolio:
        portfolio[action['Symbol']] = 0.0
        portfolio[f"{action['Symbol']}_value"] = 0.0
        portfolio[f"{action['Symbol']}_dist"] = 0.0

    if not pd.isnull(action['Dividend']) and action['Dividend'] != 0:
        portfolio.loc[action.Date:, f"{action.Symbol}_dist"] += action['Dividend']
        portfolio.loc[action.Date:, f"_total_dist"] += action.Dividend
    if not pd.isnull(action['Quantity']) and action['Quantity'] != 0:
        sign = -1 if action['Action'] in ['Sell', 'Remove Shares'] else 1
        portfolio.loc[action.Date:, f"{action.Symbol}"] += sign * action['Quantity']
    if not pd.isnull(action['Total']) and action['Total'] != 0:
        sign = -1 if action['Action'] == 'Buy' else 1
        portfolio.loc[action['Date']:, "cash"] += sign * action['Total']
    if action['Action'] == 'Add Shares' and '__contribution__' in action['Comment']:
        price = get_price(action.Symbol, quotes, portfolio.index)[action.Date]
        log.debug(f"Contribution of {action.Quantity} shares of "
                  f"{action.Symbol} @ {price} per share.")
        portfolio.loc[action.Date:, 'contributions'] += price * action['Quantity']
    if action['Action'] == 'Add Shares' and '__dividend__' in action['Comment']:
        value = get_price(action.Symbol, quotes, portfolio.index)[
                    action.Date] * action['Quantity']
        log.debug(f"Dividend of {action.Quantity} shares of {action.Symbol} "
                  f"is ${value}.")
        portfolio.loc[action.Date:, f"{action.Symbol}_dist"] += value
        portfolio.loc[action.Date:, f"_total_dist"] += value

    return portfolio


def convert_action(action, quotes, to_symbol, index, ignore_commission=False):
    """For simulated portfolios

    Convert an action on one security to an action on another security.
    Filter out any reinvestment actions.
    If the action is a distribution, then instead sell an equivalent
    quantity of the new security -- the original portfolio has actions
    (such as purchases or withdrawals) which assume that cash is present.
    """
    action = action.copy()
    if ignore_commission:
        raise NotImplementedError()

    if action['Action'] in ['Reinvest Dividend', 'Reinvest L-Term CG Dist',
                            'Reinvest S-Term CG Dist']:
        if action['Total']:
            print(action)
        return None
    skip_comment_flags = ['__dividend__', '__split__', '__merger__']
    if not pd.isnull(action['Comment']) and any(
          [f in action['Comment'] for f in skip_comment_flags]):
        if action['Total']:
            print(action)
        return None

    from_symbol = action['Symbol']
    from_price = get_price(from_symbol, quotes, index)[action.Date]
    to_price = get_price(to_symbol, quotes, index)[action.Date]

    action['Symbol'] = to_symbol
    if action['Action'] in ['Buy', 'Sell']:
        action['Quantity'] = (
                             action['Total'] - action['Commission']) / to_price
        action['Price'] = to_price
    elif action['Action'] in ['Add Shares', 'Remove Shares']:
        new_qty = action['Quantity'] * from_price / to_price
        action['Quantity'] = new_qty
    elif action['Action'] in ['Dividend']:
        # If I didn't auto-reinvest a dividend, "sell" simulated shares at
        # zero commission. I'll do something else with that money soon.
        action['Action'] = 'Sell'
        action['Price'] = to_price
        action['Quantity'] = action['Total'] / to_price
        action['Dividend'] = np.nan
        action['Commission'] = 0.
    else:
        raise RuntimeError(f"Encountered action {action['Action']}")

    if from_symbol in []:
        print(action)
    return action


def add_sim_dividend(portfolio, quotes, sim_symbol):
    """Look at a portfolio and add the dividends that you would have gotten
    from a particular security. For simulated portfolios.
    """
    qu = quotes[sim_symbol]
    div = qu[qu.dividend_amount != 0].sort_index(ascending=True)

    for date in div.index:
        if date in portfolio.index:
            price = get_price(sim_symbol, quotes, portfolio.index)[
                date]
            value = portfolio.loc[date, sim_symbol] * qu.loc[
                [date], 'dividend_amount']
            value = value.reindex(portfolio.index, method='ffill').fillna(0)
            qty = value / price
            portfolio[f"{sim_symbol}_dist"] += value
            portfolio[sim_symbol] += qty
            portfolio['_total_dist'] += value

    return portfolio


def get_price(symbol, quotes, index):
    price = (quotes[symbol]['close']
             .reindex(index, method='ffill')
             .fillna(method='ffill')
             .fillna(method='bfill'))
    return price


def total_portfolio(portfolio, quotes):
    symbols = [c for c in portfolio.columns if c == c.upper()]
    portfolio['_total'] = 0.
    for symbol in symbols:
        if symbol not in quotes:
            log.error(f"Couldn't find a price for {symbol}. "
                      f"Assuming its value is zero.")
            continue
        # Use "ffill" to take us through e.g. weekends. Use bfill to
        # make sure that we have valid prices at the beginning of the series.
        price = (quotes[symbol]['close']
                 .reindex(portfolio.index, method='ffill')
                 .fillna(method='ffill')
                 .fillna(method='bfill'))
        portfolio[f"{symbol}_value"] = portfolio[symbol] * price
        portfolio['_total'] += portfolio[f"{symbol}_value"]
    return portfolio


def record_trans(portfolio, transactions):
    # A "Category" with an "@" should be accounted for by
    # investment transactions
    transactions = (transactions[~transactions['Category']
                    .str.contains('@').fillna(False)])
    # A "Category" which starts with "Dividend" is an investment transaction
    transactions = (transactions[~transactions['Category']
                    .str.startswith('Dividend').fillna(False)])
    for i, row in transactions.iterrows():
        dep, wd = row['Deposit'], row['Withdrawal']  # Alias for convenience
        is_internal = ((row['Category'] in ["Other Income:Interest", "Mail and Paper Work"]) or
                       (pd.isnull(row['Category']) and pd.isnull(row['Payee'])))
        if not pd.isnull(wd) and wd != 0:
            portfolio.loc[row['Date']:, "cash"] -= wd
            if not is_internal:
                portfolio.loc[row['Date']:, "withdrawals"] += wd
        if not pd.isnull(dep) and dep != 0:
            portfolio.loc[row['Date']:, "cash"] += dep
            if not is_internal:
                portfolio.loc[row['Date']:, "contributions"] += dep
        if row['Category'] == "Other Income:Interest":
            portfolio.loc[row['Date']:, "_total_dist"] += dep
    return portfolio


def make_dummy_prices(symbol, prototype_symbol, value: float = 1):
    """Read the quotes for the prototype symbol, and replace all
    prices with the `value`.

    I can run it by hand once, then be done with it.
    """
    fname_prototype = os.path.join(config.CACHE_DIR, f'{prototype_symbol}.csv')
    fname_new = os.path.join(config.CACHE_DIR, f'{symbol}.csv')
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

    if refresh_cache or quotes is None:
        log.info('Reading %s data from Alpha Vantage.', symbol)
        time.sleep(15)
        new_quotes = pd.read_csv(av_api.format(symbol=symbol,
                                               api_key=config.AV_API_KEY),
                                 **reader_kwargs)
        if new_quotes.index.name == '{':
            # This is an error message.
            raise ValueError(f'Error fetching "{symbol}": '
                             f'{quotes.index[0].strip()}')
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


def create_simulated_portfolio(investment_tranactions: pd.DataFrame,
                               port_trans: Dict[str, pd.DataFrame],
                               quotes: Dict[str, pd.DataFrame],
                               sim_symbol: str='SWPPX') -> pd.DataFrame:
    inv = investment_tranactions
    max_date = max((q.index.max() for q in quotes.values()))
    index = pd.date_range(start=inv['Date'].min(), end=max_date, freq='D')
    sim_acct = init_portfolio(index)
    for idx, row in inv.iterrows():
        if sim_symbol and row['Symbol'] not in ['SWXXX']:
            # SWXXX is a money market fund; used like cash
            row = convert_action(row, quotes, sim_symbol, sim_acct.index)
        if row is None:
            continue
        record_action(sim_acct, row, quotes)
    sim_acct = add_sim_dividend(sim_acct, quotes, sim_symbol)
    total_portfolio(sim_acct, quotes)

    for acct_name, trans in port_trans.items():
        record_trans(sim_acct, trans)
    sim_acct['_total'] += sim_acct['cash']

    return sim_acct


def read_investment_transactions(fname=config.INVESTMENT_TRANS_FNAME):
    inv = pd.read_csv(fname,
                      dtype={'Dividend': float, 'Price': float, 'Total': float,
                             'Commission': float, 'Quantity': float},
                      parse_dates=[0],
                      thousands=',')

    log.info(f'Ignore transactions in {config.SKIP_ACCOUNTS} accounts.')
    inv = inv.drop(inv[inv.Account.isin(config.SKIP_ACCOUNTS)].index)
    return inv


def read_portfolio_transactions(all_fnames=config.PORT_TRANSACTIONS):
    return {acct_name: pd.read_csv(fname, parse_dates=[0], thousands=',')
            for acct_name, fname in all_fnames.items()
            if fname and acct_name not in config.SKIP_ACCOUNTS}


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


def refresh_portfolio(refresh_cache=False):
    """This is the "main" function; it runs everything."""
    logging.basicConfig(level='INFO')

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    inv = read_investment_transactions(config.INVESTMENT_TRANS_FNAME)
    quotes = read_all_quotes(inv.Symbol.unique(), config.SKIP_SYMBOL_DOWNLOADS,
                             refresh_cache)
    max_date = max((q.index.max() for q in quotes.values()))
    log.info(f"The most recent quote is from {max_date}")

    index = pd.date_range(start=inv['Date'].min(),  end=max_date, freq='D')
    accounts = {}
    all_accounts = init_portfolio(index)
    for idx, row in inv.iterrows():
        if row['Account'] not in accounts:
            accounts[row['Account']] = init_portfolio(index)
        record_action(all_accounts, row, quotes)
        record_action(accounts[row['Account']], row, quotes)
    total_portfolio(all_accounts, quotes)
    for acct in accounts.values():
        total_portfolio(acct, quotes)

    max_trans_date = None
    portfolio_transactions = read_portfolio_transactions(
        config.PORT_TRANSACTIONS)
    for acct_name, trans in portfolio_transactions.items():
        log.info('Logging portfolio transactions for %s.', acct_name)
        record_trans(accounts[acct_name], trans)
        record_trans(all_accounts, trans)

        accounts[acct_name]['_total'] += accounts[acct_name]['cash']
        accounts[acct_name].loc[accounts[acct_name]['_total'].abs() < 0.01,
                                '_total'] = 0
        if not max_trans_date:
            max_trans_date = trans['Date'].max()
        else:
            max_trans_date = max((max_trans_date, trans['Date'].max()))
    all_accounts['_total'] += all_accounts['cash']

    for symbol in config.SYMBOLS_TO_SIMULATE:
        log.info('Simulating all transactions as %s.', symbol)
        name = f'Simulated {symbol} All-Account'
        accounts[name] = create_simulated_portfolio(
            inv, portfolio_transactions, quotes, symbol)

    log.info("The most recent transaction was on %s", max_trans_date)

    with open(config.ETL_ACCTS, 'wb') as _fout:
        pickle.dump((accounts, max_trans_date.date()), _fout)

    return accounts, max_trans_date.date()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process portfolio data exported from AceMoney so that it "
                    "can be used by the Portfolio Tracker Dash app.")
    parser.add_argument('--quotes', action='store_true', default=False,
                        help="Refresh historical quotes from Alpha Vantage")
    args = parser.parse_args()

    _ = refresh_portfolio(refresh_cache=args.quotes)
