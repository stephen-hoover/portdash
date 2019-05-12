"""Read transaction logs from AceMoney

An investment Action from AceMoney has fields
"Date", "Action", "Symbol", "Account", "Dividend", "Price",
"Quantity", "Commission", "Total", "Comment"
"""
import argparse
from datetime import datetime
import logging
import os
import pickle
from typing import Dict

import pandas as pd
import numpy as np

from portdash import config, quotes
from portdash.config import conf

log = logging.getLogger(__name__)


def init_portfolio(index: pd.DatetimeIndex) -> pd.DataFrame:
    """A `portfolio` DataFrame has two columns for each investment,
    one for total number of shares at any particular time and
    another for total distributions from each investment.
    There's also several global / aggregate columns, initialized here."""
    return pd.DataFrame({'_total': 0., '_total_dist': 0., 'cash': 0.,
                         'contributions': 0., 'withdrawals': 0.}, index=index)


def record_action(portfolio: pd.DataFrame, action: pd.Series) -> pd.DataFrame:
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
        price = quotes.get_price(action.Symbol, [action.Date]).values[0]
        log.debug(f"Contribution of {action.Quantity} shares of "
                  f"{action.Symbol} @ {price} per share.")
        portfolio.loc[action.Date:, 'contributions'] += price * action['Quantity']
    if action['Action'] == 'Add Shares' and '__dividend__' in action['Comment']:
        value = (quotes.get_price(action.Symbol, [action.Date]).values[0] *
                 action['Quantity'])
        log.debug(f"Dividend of {action.Quantity} shares of {action.Symbol} "
                  f"is ${value}.")
        portfolio.loc[action.Date:, f"{action.Symbol}_dist"] += value
        portfolio.loc[action.Date:, f"_total_dist"] += value

    return portfolio


def add_sim_dividend(portfolio: pd.DataFrame, sim_symbol: str) -> pd.DataFrame:
    """Look at a portfolio and add the dividends that you would have gotten
    from a particular security. For simulated portfolios.
    """
    div = quotes.get_dividend(sim_symbol, portfolio.index)
    for date, amount in div.iteritems():
        if date in portfolio.index:
            price = quotes.get_price(sim_symbol, [date]).values[0]

            # Create a Series with 0s before the date, and the value of
            # the distribution at and after the date.
            value = portfolio.loc[[date], sim_symbol] * amount
            value = value.reindex(portfolio.index, method='ffill').fillna(0)

            qty = value / price
            portfolio[f"{sim_symbol}_dist"] += value
            portfolio[sim_symbol] += qty
            portfolio['_total_dist'] += value

    return portfolio


def total_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    symbols = [c for c in portfolio.columns if c == c.upper()]
    portfolio['_total'] = 0.
    for symbol in symbols:
        price = quotes.get_price(symbol, portfolio.index)
        portfolio[f"{symbol}_value"] = portfolio[symbol] * price
        portfolio['_total'] += portfolio[f"{symbol}_value"]
    return portfolio


def get_deposits(portfolio: pd.DataFrame) -> pd.Series:
    """Return a Series indicating when money was deposited into or
    withdrawn from this account. This uses the cumulative contributions
    and cumulative withdrawals columns in the portfolio.
    """
    deposits = (portfolio['contributions'].diff() -
                portfolio['withdrawals'].diff())
    deposits.loc[portfolio.index.min()] = portfolio['_total'].iloc[0]
    return deposits


def record_trans(portfolio: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    # A "Category" with an "@" should be accounted for by
    # investment transactions
    transactions = (transactions[~transactions['Category']
                    .str.contains('@').fillna(False)])
    # A "Category" which starts with "Dividend" is an investment transaction
    transactions = (transactions[~transactions['Category']
                    .str.startswith('Dividend').fillna(False)])
    max_date = portfolio.index.max()
    for i, row in transactions.iterrows():
        if row['Date'] > max_date:
            # Ignore transactions which might come after the end of the
            # period under consideration.
            # Assume that the transaction dates are always greater than
            # the minimum date (this should be handled in initialization).
            continue
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


def create_simulated_portfolio(portfolio: pd.DataFrame,
                               sim_symbol: str = 'SWPPX') -> pd.DataFrame:
    """Given a portfolio, create a parallel portfolio which has
    all of the same contributions and withdrawals. In the parallel
    simulated portfolio, all contributions are immediately used to
    purchase the security indicated by the `sim_symbol`, and all
    withdrawals are financed by selling this security.
    """
    sim_port = init_portfolio(portfolio.index)
    prices = quotes.get_price(sim_symbol, sim_port.index)
    for column in ['contributions', 'withdrawals']:
        sim_port[column] = portfolio[column].copy()

    deposits = get_deposits(portfolio)
    for date, amt in deposits[deposits != 0].iteritems():
        action = pd.Series(
            {"Date": date, "Action": ("Buy" if amt > 0 else "Sell"),
             "Symbol": sim_symbol, "Account": None, "Dividend": 0.,
             "Price": prices.loc[date],
             "Quantity": np.abs(amt) / prices.loc[date], "Commission": 0,
             "Total": 0, "Comment": ""})
        record_action(sim_port, action)

    # We've recorded simulated purchases, but the simulated security
    # would also have generated dividends.
    sim_port = add_sim_dividend(sim_port, sim_symbol)
    total_portfolio(sim_port)

    return sim_port


def read_investment_transactions(fname: str=None) -> pd.DataFrame:
    if not fname:
        fname = conf('investment_transactions')
    inv = pd.read_csv(fname,
                      dtype={'Dividend': float, 'Price': float, 'Total': float,
                             'Commission': float, 'Quantity': float},
                      parse_dates=[0],
                      thousands=',')

    log.info(f'Ignore transactions in {conf("skip_accounts")} accounts.')
    inv = inv.drop(inv[inv.Account.isin(conf('skip_accounts'))].index)
    return inv


def read_portfolio_transactions(acct_fnames: Dict[str, str]=None,
                                ignore_future: bool=True) \
      -> Dict[str, pd.DataFrame]:
    if not acct_fnames:
        acct_fnames = conf('account_transactions')
    trans = {acct_name: pd.read_csv(fname, parse_dates=[0], thousands=',')
             for acct_name, fname in acct_fnames.items()
             if fname and acct_name not in conf('skip_accounts')}
    if ignore_future:
        # Filter out any transactions marked as being in the future.
        # This can happen with scheduled transactions.
        today = datetime.today()
        trans = {k: v[v['Date'] <= today] for k, v in trans.items()}
    return trans


def refresh_portfolio(refresh_cache: bool=False):
    """This is the "main" function; it runs everything."""
    os.makedirs(conf('cache_dir'), exist_ok=True)
    inv = read_investment_transactions(conf('investment_transactions'))
    portfolio_transactions = read_portfolio_transactions(
        conf('account_transactions'))
    # Read all the quotes either from disk or from the web.
    # We won't use the quote_dict except to get a most recent available date,
    # but this call will cache the data and read from the web if requested.
    quote_dict = quotes.read_all_quotes(inv.Symbol.unique(),
                                        conf('skip_symbol_downloads'),
                                        refresh_cache)
    max_date = max((q.index.max() for q in quote_dict.values()))
    log.info(f"The most recent quote is from {max_date}")

    min_port = min(p['Date'].min() for p in portfolio_transactions.values())
    index = pd.date_range(start=min(inv['Date'].min(), min_port),
                          end=max_date, freq='D')
    accounts = {}
    all_accounts = init_portfolio(index)
    log.info('Logging investment transactions for all portfolios.')
    for idx, row in inv.iterrows():
        if row['Account'] not in accounts:
            accounts[row['Account']] = init_portfolio(index)
        record_action(all_accounts, row)
        record_action(accounts[row['Account']], row)
    total_portfolio(all_accounts)
    for acct in accounts.values():
        total_portfolio(acct)

    max_trans_date = None
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

    for symbol in conf('symbols_to_simulate'):
        log.info(f'Simulating all transactions as {symbol}.')
        name = f'Simulated {symbol} All-Account'
        accounts[name] = create_simulated_portfolio(all_accounts, symbol)

    log.info("The most recent transaction was on %s", max_trans_date)

    with open(conf('etl_accts'), 'wb') as _fout:
        pickle.dump((accounts, max_trans_date.date()), _fout)

    return accounts, max_trans_date.date()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process portfolio data exported from AceMoney so that it "
                    "can be used by the Portfolio Tracker Dash app.")
    parser.add_argument('--quotes', action='store_true', default=False,
                        help="Refresh historical quotes from Alpha Vantage")
    parser.add_argument('-c', '--conf', required=True,
                        help="Configuration file in YAML format")
    args = parser.parse_args()

    logging.basicConfig(level='INFO')
    config.load_config(args.conf)
    _ = refresh_portfolio(refresh_cache=args.quotes)
