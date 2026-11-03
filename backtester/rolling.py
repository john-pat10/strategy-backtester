"""Run strategies over many start dates instead of one lucky (or unlucky) period."""

import pandas as pd

from .engine import run_backtest


def rolling_backtests(strategies: dict, prices: pd.DataFrame, deposits, horizon_years: int,
                      cash_rate=None, fee_rate=0.0, step="MS", first_start=None) -> pd.DataFrame:
    """Backtest every strategy from every start date, each for `horizon_years`.

    strategies: name -> function returning a fresh strategy (strategies keep state).
    deposits: function (start, end) -> deposit Series, so every window gets the
        same schedule shifted to its own dates.
    Returns one row per (start date, strategy) with its metrics.
    """
    first_start = pd.Timestamp(first_start or prices.index[0])
    last_start = prices.index[-1] - pd.DateOffset(years=horizon_years)
    rows = []
    for start in pd.date_range(first_start, last_start, freq=step):
        end = start + pd.DateOffset(years=horizon_years)
        schedule = deposits(start, end)
        for name, make in strategies.items():
            result = run_backtest(make(), prices, schedule, cash_rate, fee_rate, start, end)
            rows.append({"start": start, "strategy": name, **result.metrics()})
    return pd.DataFrame(rows)


def head_to_head(results: pd.DataFrame, a: str, b: str, metric="final_value") -> dict:
    """How often strategy `a` beat strategy `b`, and by how much."""
    wide = results.pivot(index="start", columns="strategy", values=metric)
    diff = wide[a] / wide[b] - 1
    return {
        "periods": len(diff),
        "a_win_rate": float((diff > 0).mean()),
        "median_edge": float(diff.median()),
        "worst_edge": float(diff.min()),
        "best_edge": float(diff.max()),
    }
