import pandas as pd

from portdash.apis import quotes


def init_portfolio(index: pd.DatetimeIndex) -> pd.DataFrame:
    """A `portfolio` DataFrame has two columns for each investment,
    one for total number of shares at any particular time and
    another for total distributions from each investment.
    There's also several global / aggregate columns, initialized here."""
    return pd.DataFrame(
        {
            "_total": 0.0,
            "_total_dist": 0.0,
            "cash": 0.0,
            "contributions": 0.0,
            "withdrawals": 0.0,
        },
        index=index,
    )


def init_symbol(portfolio: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if symbol not in portfolio:
        portfolio[symbol] = 0.0
        portfolio[f"{symbol}_value"] = 0.0
        portfolio[f"{symbol}_dist"] = 0.0
    return portfolio


def get_deposits(portfolio: pd.DataFrame) -> pd.Series:
    """Return a Series indicating when money was deposited into or
    withdrawn from this account. This uses the cumulative contributions
    and cumulative withdrawals columns in the portfolio.
    """
    deposits = portfolio["contributions"].diff() - portfolio["withdrawals"].diff()
    deposits.loc[portfolio.index.min()] = portfolio["_total"].iloc[0]
    return deposits


def total_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    symbols = [c for c in portfolio.columns if c == c.upper()]
    portfolio["_total"] = 0.0
    for symbol in symbols:
        price = quotes.get_price(symbol, portfolio.index)
        portfolio[f"{symbol}_value"] = portfolio[symbol] * price
        portfolio["_total"] += portfolio[f"{symbol}_value"]
    return portfolio
