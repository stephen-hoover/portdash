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

import config
from config import conf
from portdash import portfolio as port
from portdash import create_app
from portdash.apis import quotes

log = logging.getLogger(__name__)


def record_inv_action(portfolio: pd.DataFrame, action: pd.Series) -> pd.DataFrame:
    """Mark the results of an AceMoney investment transaction in a portfolio

    `portfolio`: pd.DataFrame
        Initialized as per `init_portfolio`
    `action`: pd.Series
        A row from the all-investment-transactions AceMoney export
    """
    port.init_symbol(portfolio, symbol=action["Symbol"])

    if not pd.isnull(action["Dividend"]) and action["Dividend"] != 0:
        portfolio.loc[action.Date :, f"{action.Symbol}_dist"] += action["Dividend"]
        portfolio.loc[action.Date :, f"_total_dist"] += action.Dividend
    if not pd.isnull(action["Quantity"]) and action["Quantity"] != 0:
        sign = -1 if action["Action"] in ["Sell", "Remove Shares"] else 1
        portfolio.loc[action.Date :, f"{action.Symbol}"] += sign * action["Quantity"]
    if not pd.isnull(action["Total"]) and action["Total"] != 0:
        sign = -1 if action["Action"] == "Buy" else 1
        portfolio.loc[action["Date"] :, "cash"] += sign * action["Total"]
    if action["Action"] == "Add Shares" and "__contribution__" in action["Comment"]:
        price = quotes.get_price(action.Symbol, [action.Date]).values[0]
        log.debug(
            f"Contribution of {action.Quantity} shares of "
            f"{action.Symbol} @ {price} per share."
        )
        portfolio.loc[action.Date :, "contributions"] += price * action["Quantity"]
    if action["Action"] == "Add Shares" and "__dividend__" in action["Comment"]:
        value = (
            quotes.get_price(action.Symbol, [action.Date]).values[0]
            * action["Quantity"]
        )
        log.debug(
            f"Dividend of {action.Quantity} shares of {action.Symbol} "
            f"of {action.Date} is ${value}."
        )
        portfolio.loc[action.Date :, f"{action.Symbol}_dist"] += value
        portfolio.loc[action.Date :, f"_total_dist"] += value

    return portfolio


def record_trans(portfolio: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Insert the contents of an AceMoney account export into a portfolio"""
    # A "Category" with an "@" should be accounted for by
    # investment transactions
    transactions = transactions[
        ~transactions["Category"].str.contains("@").fillna(False)
    ]
    # A "Category" which starts with "Dividend" is an investment transaction
    transactions = transactions[
        ~transactions["Category"].str.startswith("Dividend").fillna(False)
    ]
    for i, row in transactions.iterrows():
        record_acct_action(portfolio, row)
    return portfolio


def record_acct_action(portfolio: pd.DataFrame, action: pd.DataFrame) -> pd.DataFrame:
    """Mark the results of an AceMoney account transaction in a portfolio

    `portfolio`: pd.DataFrame
        Initialized as per `init_portfolio`
    `action`: pd.Series
        A row from the single account AceMoney export
    """
    if action["Date"] <= portfolio.index.max():
        # Ignore transactions which might come after the end of the
        # period under consideration.
        # Assume that the transaction dates are always greater than
        # the minimum date (this should be handled in initialization).
        dep, wd = action["Deposit"], action["Withdrawal"]  # Alias for convenience
        is_internal = (
            action["Category"] in ["Other Income:Interest", "Mail and Paper Work"]
        ) or (pd.isnull(action["Category"]) and pd.isnull(action["Payee"]))
        if not pd.isnull(wd) and wd != 0:
            portfolio.loc[action["Date"] :, "cash"] -= wd
            if not is_internal:
                portfolio.loc[action["Date"] :, "withdrawals"] += wd
        if not pd.isnull(dep) and dep != 0:
            portfolio.loc[action["Date"] :, "cash"] += dep
            if not is_internal:
                portfolio.loc[action["Date"] :, "contributions"] += dep
        if action["Category"] == "Other Income:Interest":
            portfolio.loc[action["Date"] :, "_total_dist"] += dep
    return portfolio


def read_investment_transactions(fname: str = None) -> pd.DataFrame:
    """Read a CSV of investment transactions written by AceMoney"""
    if not fname:
        fname = conf("investment_transactions")
    inv = pd.read_csv(
        fname,
        dtype={
            "Dividend": float,
            "Price": float,
            "Total": float,
            "Commission": float,
            "Quantity": float,
        },
        parse_dates=[0],
        thousands=",",
    )

    log.info(f'Ignore transactions in {conf("skip_accounts")} accounts.')
    inv = inv.drop(inv[inv.Account.isin(conf("skip_accounts"))].index)
    return inv


def read_portfolio_transactions(
    acct_fnames: Dict[str, str] = None, ignore_future: bool = True
) -> Dict[str, pd.DataFrame]:
    """Read a CSV of account transactions written by AceMoney"""
    if not acct_fnames:
        acct_fnames = conf("account_transactions")
    trans = {
        acct_name: pd.read_csv(fname, parse_dates=[0], thousands=",")
        for acct_name, fname in acct_fnames.items()
        if fname and acct_name not in conf("skip_accounts")
    }
    if ignore_future:
        # Filter out any transactions marked as being in the future.
        # This can happen with scheduled transactions.
        today = datetime.today()
        trans = {k: v[v["Date"] <= today] for k, v in trans.items()}
    return trans


def refresh_portfolio(refresh_cache: bool = False):
    """This is the "main" function; it runs everything."""
    os.makedirs(conf("cache_dir"), exist_ok=True)
    inv = read_investment_transactions(conf("investment_transactions"))
    portfolio_transactions = read_portfolio_transactions(conf("account_transactions"))
    # Read all the quotes either from disk or from the web.
    if refresh_cache:
        quotes.refresh_quotes(inv.Symbol.unique())
    max_date = quotes.get_max_date()
    log.info(f"The most recent quote is from {max_date}")

    min_port = min(p["Date"].min() for p in portfolio_transactions.values())
    index = pd.date_range(
        start=min(inv["Date"].min(), min_port), end=max_date, freq="D"
    )
    accounts = {}
    all_accounts = port.init_portfolio(index)
    log.info("Logging investment transactions for all portfolios.")
    for idx, row in inv.iterrows():
        if row["Account"] not in accounts:
            accounts[row["Account"]] = port.init_portfolio(index)
        record_inv_action(all_accounts, row)
        record_inv_action(accounts[row["Account"]], row)
    port.total_portfolio(all_accounts)
    for acct in accounts.values():
        port.total_portfolio(acct)

    max_trans_date = None
    for acct_name, trans in portfolio_transactions.items():
        log.info("Logging portfolio transactions for %s.", acct_name)
        record_trans(accounts[acct_name], trans)
        record_trans(all_accounts, trans)

        accounts[acct_name]["_total"] += accounts[acct_name]["cash"]
        accounts[acct_name].loc[
            accounts[acct_name]["_total"].abs() < 0.01, "_total"
        ] = 0
        if not max_trans_date:
            max_trans_date = trans["Date"].max()
        else:
            max_trans_date = max((max_trans_date, trans["Date"].max()))
    all_accounts["_total"] += all_accounts["cash"]

    log.info("The most recent transaction was on %s", max_trans_date)
    _check_account_dict(accounts)

    with open(conf("etl_accts"), "wb") as _fout:
        pickle.dump((accounts, max_trans_date.date()), _fout)

    return accounts, max_trans_date.date()


def _check_account_dict(accts: Dict[str, pd.DataFrame]):
    """Raise a RuntimeError if any account in the input dictionary has nulls"""
    err_msgs = [(acct_name, _check_account(acct)) for acct_name, acct in accts.items()]
    msg = "\n".join(
        f"Missing values found in {acct_name}:\n{msg}"
        for acct_name, msg in err_msgs
        if msg
    )
    if msg:
        raise RuntimeError(msg)


def _check_account(acct: pd.DataFrame) -> str:
    """Return an error message if the input account has any nulls."""
    msg = []
    for col in acct:
        n_null = pd.isnull(acct[col]).sum()
        if n_null:
            msg.append(f"Column {col} has {n_null} missing values.")
    return "\n".join(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process portfolio data exported from AceMoney so that it "
        "can be used by the Portfolio Tracker Dash app."
    )
    parser.add_argument(
        "--quotes",
        action="store_true",
        default=False,
        help="Refresh historical quotes from Alpha Vantage",
    )
    parser.add_argument(
        "-c", "--conf", required=True, help="Configuration file in YAML format"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Output verbose logs.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=("DEBUG" if args.verbose else "INFO"))
    config.load_config(args.conf)

    server = create_app()

    with server.app_context():
        from portdash.extensions import db

        db.create_all()
        _ = refresh_portfolio(refresh_cache=args.quotes)
