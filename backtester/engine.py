"""The core simulation loop.

Every strategy sees the same money arriving on the same days (the
`contributions` schedule) and only decides what to do with its cash. That keeps
comparisons fair: lump sum, DCA and dip buying all get identical deposits.

Timing convention: each trading day, the strategy sees prices up to and
including that day's close and trades at that close. It can never see a future
price.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_TRADE = 1.0  # ignore orders smaller than $1


@dataclass
class Order:
    ticker: str
    amount: float  # dollars; positive buys, negative sells


@dataclass
class Portfolio:
    cash: float = 0.0
    shares: dict = field(default_factory=dict)

    def value(self, prices: dict) -> float:
        return self.cash + sum(n * prices[t] for t, n in self.shares.items())


class Context:
    """Everything a strategy may look at on a given day."""

    def __init__(self, date, day, flow, portfolio, price_array, columns, row):
        self.date = date
        self.day = day            # 0 on the first day of the backtest
        self.flow = flow          # money deposited today
        self.portfolio = portfolio
        self._prices = price_array
        self._columns = columns
        self._row = row           # today's row in the full price history

    def price(self, ticker) -> float:
        return self._prices[self._row, self._columns[ticker]]

    def history(self, ticker, lookback=None) -> np.ndarray:
        """Prices up to and including today, optionally only the last `lookback` days."""
        first = 0 if lookback is None else max(0, self._row + 1 - lookback)
        return self._prices[first:self._row + 1, self._columns[ticker]]


class Strategy:
    name = "strategy"

    def on_day(self, ctx: Context) -> list:
        return []


@dataclass
class BacktestResult:
    name: str
    daily: pd.DataFrame  # columns: value, cash, flow, rf (daily cash return)

    def metrics(self) -> dict:
        from .metrics import summarize
        return summarize(self.daily)


def align_to_trading_days(contributions: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Move each deposit to the first trading day on or after its date."""
    if contributions.empty:
        return pd.Series(0.0, index=dates)
    positions = dates.searchsorted(contributions.index)
    keep = positions < len(dates)
    aligned = pd.Series(contributions.to_numpy()[keep], index=dates[positions[keep]])
    return aligned.groupby(level=0).sum().reindex(dates, fill_value=0.0)


def daily_cash_returns(cash_rate, dates: pd.DatetimeIndex) -> np.ndarray:
    """Convert annual yields to a per-trading-day return for each date."""
    if cash_rate is None:
        return np.zeros(len(dates))
    if np.isscalar(cash_rate):
        annual = np.full(len(dates), float(cash_rate))
    else:
        annual = cash_rate.reindex(dates.union(cash_rate.index)).ffill().reindex(dates).fillna(0).to_numpy()
    return (1 + annual) ** (1 / TRADING_DAYS) - 1


def run_backtest(strategy: Strategy, prices: pd.DataFrame, contributions: pd.Series,
                 cash_rate=None, fee_rate=0.0, start=None, end=None) -> BacktestResult:
    """Simulate `strategy` from `start` to `end`.

    prices: daily prices, one column per ticker. May include history before
        `start`; strategies can look back into it.
    contributions: deposit amounts indexed by date.
    cash_rate: annual yield on uninvested cash, either a number or a Series by date.
    fee_rate: fraction of each trade lost to costs (0.001 = 0.1%).
    """
    prices = prices.sort_index()
    dates = prices.loc[start:end].index
    if len(dates) == 0:
        raise ValueError("No trading days in the requested range")

    first_row = prices.index.get_loc(dates[0])
    price_array = prices.to_numpy(dtype=float)
    columns = {t: i for i, t in enumerate(prices.columns)}
    flows = align_to_trading_days(contributions, dates).to_numpy()
    rf = daily_cash_returns(cash_rate, dates)

    portfolio = Portfolio()
    values, cash = np.empty(len(dates)), np.empty(len(dates))

    for day, date in enumerate(dates):
        row = first_row + day
        if day > 0:
            portfolio.cash *= 1 + rf[day]
        portfolio.cash += flows[day]

        ctx = Context(date, day, flows[day], portfolio, price_array, columns, row)
        for order in strategy.on_day(ctx) or []:
            _execute(order, portfolio, ctx.price(order.ticker), fee_rate)

        today = {t: price_array[row, i] for t, i in columns.items() if t in portfolio.shares}
        values[day] = portfolio.value(today)
        cash[day] = portfolio.cash

    daily = pd.DataFrame({"value": values, "cash": cash, "flow": flows, "rf": rf}, index=dates)
    return BacktestResult(strategy.name, daily)


def _execute(order: Order, portfolio: Portfolio, price: float, fee_rate: float):
    if order.amount > 0:
        spend = min(order.amount, portfolio.cash)
        if spend < MIN_TRADE:
            return
        portfolio.cash -= spend
        bought = spend * (1 - fee_rate) / price
        portfolio.shares[order.ticker] = portfolio.shares.get(order.ticker, 0.0) + bought
    elif order.amount < 0:
        held = portfolio.shares.get(order.ticker, 0.0)
        sold = min(-order.amount / price, held)
        if sold * price < MIN_TRADE:
            return
        portfolio.shares[order.ticker] = held - sold
        portfolio.cash += sold * price * (1 - fee_rate)
