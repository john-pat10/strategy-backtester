"""Performance metrics.

Deposits distort simple return math (a portfolio that grows because you added
money didn't "earn" that growth), so this module uses two standard measures:

- Time-weighted returns strip out deposits, so volatility, drawdown and Sharpe
  describe the strategy itself.
- IRR (money-weighted return) accounts for when money went in, which is the
  fair way to compare lump sum against spreading it out.
"""

import numpy as np
import pandas as pd

from .engine import TRADING_DAYS


def time_weighted_returns(daily: pd.DataFrame) -> pd.Series:
    """Daily returns with each day's deposit removed."""
    prev = daily["value"].shift(1)
    returns = (daily["value"] - daily["flow"]) / prev - 1
    return returns[prev > 0]


def max_drawdown(returns: pd.Series) -> float:
    growth = (1 + returns).cumprod()
    return float((growth / growth.cummax() - 1).min()) if len(growth) else 0.0


def xirr(dates, cash_flows) -> float:
    """Annual rate r where the cash flows' present value is zero.

    Money you put in is negative, money you get back is positive.
    """
    years = np.array([(d - dates[0]).days / 365.25 for d in dates])
    flows = np.asarray(cash_flows, dtype=float)

    def npv(rate):
        return np.sum(flows / (1 + rate) ** years)

    low, high = -0.99, 10.0
    if npv(low) * npv(high) > 0:
        return float("nan")
    for _ in range(200):
        mid = (low + high) / 2
        if npv(low) * npv(mid) <= 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def summarize(daily: pd.DataFrame) -> dict:
    returns = time_weighted_returns(daily)
    years = len(daily) / TRADING_DAYS
    growth = float((1 + returns).prod())
    excess = returns - daily["rf"].reindex(returns.index)
    vol = float(returns.std() * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else 0.0

    deposits = daily.loc[daily["flow"] != 0, "flow"]
    irr_dates = list(deposits.index) + [daily.index[-1]]
    irr_flows = list(-deposits.to_numpy()) + [daily["value"].iloc[-1]]

    return {
        "final_value": float(daily["value"].iloc[-1]),
        "contributed": float(daily["flow"].sum()),
        "irr": xirr(irr_dates, irr_flows) if len(deposits) else float("nan"),
        "twr_annual": growth ** (1 / years) - 1 if years > 0 else 0.0,
        "volatility": vol,
        "sharpe": float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) if vol > 0 else float("nan"),
        "max_drawdown": max_drawdown(returns),
        "avg_cash_share": float((daily["cash"] / daily["value"]).where(daily["value"] > 0).mean()),
    }
