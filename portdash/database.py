"""Access stored data on transactions and values

Each account is stored as a pandas DataFrame.
The DataFrame is indexed by date, with one row per day, starting at
the first day shown on the dashboard and continuing to the last day
covered by the dashboard.

Each account DataFrame has the following columns:
 _total : Total dollar value of the account on that day
 _total_dist : Cumulative dollar value of all distributions up to that day
 cash : Total cash holdings in the account
 contributions : Cumulative dollar value of all contributions to the account
 withdrawals :  Cumulative dollar value of all withdrawals from the account

In addition, each account has three columns per security held in the account.
(These columns exist for each security which has been present in the account
at any point during its history.) These columns are prefixed by the
stock exchange trading symbol.
 [SYMBOL] : Total units of that security held on that day
 [SYMBOL]_value : Total dollar value of that security on that day
 [SYMBOL]_dist : Cumulative distributions from that security in the account
"""
from datetime import date
from functools import lru_cache
import os
import pickle
from typing import Dict, Tuple

import pandas as pd

from config import conf
from portdash import acemoney as processing


@lru_cache(5)
def _get_accounts(refresh: bool=False,
                  download_quotes: bool=False) -> Dict[str, pd.DataFrame]:
    """Return dictionary of all accounts, keyed by account name"""
    if refresh or not os.path.exists(conf('etl_accts')):
        accts = processing.refresh_portfolio(refresh_cache=download_quotes)[0]
    else:
        accts = pickle.load(open(conf('etl_accts'), 'rb'))[0]
    return accts


def get_account_names(simulated: bool=True) -> Tuple[str]:
    """Return names of all accounts in the database

    If `simulated` is False, return only names of real accounts.
    """
    if simulated:
        account_names = _get_accounts().keys()
    else:
        account_names = [k for k in _get_accounts().keys()
                         if not k.startswith('Simulated')]
    return tuple(sorted(account_names))


def get_account(account_name: str) -> pd.DataFrame:
    """Return the requested account from the database"""
    if account_name not in _get_accounts():
        raise ValueError(f'Account "{account_name}" is not in the database.')
    return _get_accounts()[account_name]


@lru_cache(20)
def sum_accounts(account_names: Tuple[str]) -> pd.DataFrame:
    """Return a new account which is the sum of the input accounts

    Parameters
    ----------
    account_names : Tuple[str]
        Tuple of account names to sum together. Note that this must be a
        tuple rather than list to enable caching.
        If 'All Accounts' (case sensitive) occurs anywhere in the input list,
        then return the sum of all non-simulated accounts.

    Returns
    -------
    pd.DataFrame :
        A new "account" which is the sum of the named accounts in the input
    """
    if 'All Accounts' in account_names:
        account_names = get_account_names(simulated=False)
    return sum(get_account(name) for name in account_names)


def get_last_quote_date() -> date:
    """Return the date of the last quote in the database"""
    return max((a.index.max() for a in _get_accounts().values())).date()


def get_max_trans_date() -> date:
    """Return the date of the last transaction in the database"""
    return pickle.load(open(conf('etl_accts'), 'rb'))[1]
