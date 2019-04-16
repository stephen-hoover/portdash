"""Create sample transaction files in the AceMoney format

We need two kinds of CSV files. One file contains all investment
transactions over all accounts.
The other kind of file holds all transactions
(including non-investment transactions) in each account, and there's
one per account.
"""
from datetime import datetime
from typing import Dict, Tuple, Union

import pandas as pd

from portdash import quotes

TRANS_HEADER = ["Date", "Action", "Symbol", "Account", "Dividend",
                "Price", "Quantity", "Commission", "Total", "Comment"]
ACCT_HEADER = ["Date", "Payee", "Category", "S", "Withdrawal",
               "Deposit", "Total", "Comment"]


def _create_single_buy_transactions(dates: pd.DatetimeIndex,
                                    contribution: float,
                                    symbol: str,
                                    account: str) -> pd.DataFrame:
    """Create buy transactions in a format matching
    AceMoney's investment transactions
    """
    prices = quotes.get_price(symbol, dates)
    trans = pd.DataFrame({"Date": dates, "Action": "Buy",
                          "Symbol": symbol, "Account": account,
                          "Dividend": 0., "Price": prices,
                          "Quantity": contribution / prices,
                          "Commission": 0.,
                          "Total": contribution,
                          "Comment": ""},
                         columns=TRANS_HEADER)
    return trans.reset_index(drop=True)


def _include_reinvest_dividend(buy_transactions: pd.DataFrame,
                               dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Include "Reinvest Dividend" transactions in a buy transactions DataFrame

    This assumes that the `buy_transactions` input is a
    single security in a single account.
    """
    trans = buy_transactions.set_index('Date').sort_index(ascending=True)
    assert len(buy_transactions['Symbol'].unique()) == 1
    assert len(buy_transactions['Account'].unique()) == 1
    symbol = buy_transactions['Symbol'].unique()[0]
    div = quotes.get_dividend(symbol, dates).sort_index()
    prices = quotes.get_price(symbol, div.index)

    div_trans = pd.DataFrame(columns=TRANS_HEADER)  # Empty DF for appending
    for date, div_value in div.iteritems():
        n_shares = trans.loc[:date, "Quantity"].sum() + div_trans[
            "Quantity"].sum()
        div_trans_row = pd.DataFrame(
            {"Date": date, "Action": "Reinvest Dividend", "Symbol": symbol,
             "Commission": 0., "Total": 0., "Comment": "",
             "Account": trans['Account'].unique()[0],
             "Dividend": n_shares * div_value,
             "Price": prices.loc[date],
             "Quantity": (n_shares * div_value) / prices.loc[date],
             },
            columns=TRANS_HEADER, index=[len(div_trans)])
        div_trans = div_trans.append(div_trans_row)
    assert len(div_trans) == len(div) == len(prices)
    return (buy_transactions.append(div_trans)
            .sort_values('Date', ascending=True)
            .reset_index(drop=True))


def _create_sample_acct_trans(dates: pd.DatetimeIndex,
                              contribution: float,
                              invest_trans: pd.DataFrame) -> pd.DataFrame:
    """Make investment transactions
    """
    acct_trans = (pd.concat([
        pd.DataFrame(
            {'Date': dates, 'Payee': '', 'Category': 'Contributions', 'S': 'R',
             'Withdrawal': 0.0, 'Deposit': contribution, 'Total': 0.0,
             'Comment': ""},
            columns=ACCT_HEADER
        ),
        pd.DataFrame(
            {'Date': invest_trans['Date'], 'Payee':invest_trans['Symbol'],
             'Category':invest_trans.apply(_acct_cat_str, axis='columns'),
             'S': 'R', 'Withdrawal': invest_trans['Total'],
             'Deposit': 0.0, 'Total': 0.0,
             'Comment': ""},
            columns=ACCT_HEADER)
    ])
                  .sort_values('Date', ascending=True)
                  .reset_index(drop=True))
    return acct_trans


def _acct_cat_str(row: pd.Series) -> str:
    """In AceMoney investment account CSVs, the "Category" column for
    investment transactions instead describes the transaction.
    """
    if row["Action"].startswith("Reinvest"):
        return (f"{row['Action']} {row['Dividend']} {row['Quantity']} "
                f"{row['Symbol']} @ {row['Price']}")
    elif row["Action"] in ["Buy", "Sell"]:
        return (f"{row['Action']} {row['Quantity']} "
                f"{row['Symbol']} @ {row['Price']}")
    else:
        raise ValueError(f"Not able to handle {row['Action']} actions.")


def _get_total(df: pd.DataFrame) -> pd.Series:
    """Create a correct running total of cash account value in
    an account transaction table
    """
    totals = []
    total = 0.
    for _, row in df.iterrows():
        total = total + row['Deposit'] - row['Withdrawal']
        totals.append(total)
    return pd.Series(totals, index=df.index)


def create_transactions(monthly_contribution: float,
                        security_mix: Dict[str, float],
                        start_date: Union[str, datetime],
                        stop_date: Union[str, datetime],
                        account: str) \
      -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create investment and account transaction tables

    Given a monthly contribution and a mix of securities to purchase
    with that contribution, generate AceMoney-formatted tables of
    investment transactions and account transactions.
    """
    date_range = pd.bdate_range(start=start_date, end=stop_date, freq='BM')
    transact_list = []
    for symbol, frac in security_mix.items():
        this_contribution = monthly_contribution * frac
        buy = _create_single_buy_transactions(date_range, this_contribution,
                                              symbol, account)
        transact_list.append(_include_reinvest_dividend(buy, date_range))
    invest_trans = (pd.concat(transact_list)
                    .sort_values(by='Date', ascending=True)
                    .reset_index(drop=True))
    acct_trans = _create_sample_acct_trans(date_range, monthly_contribution,
                                           invest_trans)
    acct_trans['Total'] = _get_total(acct_trans)

    return invest_trans, acct_trans
