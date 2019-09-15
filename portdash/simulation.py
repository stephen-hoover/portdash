"""Allow for simulating counterfactual portfolios

If you'd done something different, what would your portfolio look like?
"""
import pandas as pd
import numpy as np

from portdash.apis import quotes
from portdash import portfolio as port


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


def create_simulated_portfolio(portfolio: pd.DataFrame,
                               sim_symbol: str = 'SWPPX') -> pd.DataFrame:
    """Given a portfolio, create a parallel portfolio which has
    all of the same contributions and withdrawals. In the parallel
    simulated portfolio, all contributions are immediately used to
    purchase the security indicated by the `sim_symbol`, and all
    withdrawals are financed by selling this security.
    """
    sim_port = port.init_portfolio(portfolio.index)
    sim_port = port.init_symbol(sim_port, sim_symbol)
    prices = quotes.get_price(sim_symbol, sim_port.index)
    for column in ['contributions', 'withdrawals']:
        sim_port[column] = portfolio[column].copy()

    deposits = port.get_deposits(portfolio)
    for date, amt in deposits[deposits != 0].iteritems():
        qty = np.abs(amt) / prices.loc[date]
        sign = -1 if amt < 0 else 1
        sim_port.loc[date:, f"{sim_symbol}"] += sign * qty

    # We've recorded simulated purchases, but the simulated security
    # would also have generated dividends.
    sim_port = add_sim_dividend(sim_port, sim_symbol)
    port.total_portfolio(sim_port)

    return sim_port
