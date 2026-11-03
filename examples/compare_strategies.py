"""Two studies on real S&P 500 data (SPY), each over every 10-year window since 1994.

1. Windfall: you get $12,000 at once. Invest it all now, or spread it out?
2. Paycheck: you have $500 every month. Plain DCA, buy the dip, or DCA with a dip reserve?

Run from the project root:  python -m examples.compare_strategies
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtester import (BuyTheDip, DCAWithDipReserve, InvestImmediately, SpreadOut,
                        head_to_head, lump_sum, monthly, rolling_backtests)
from backtester.data import load_cash_rate, load_prices

TICKER = "SPY"
HORIZON = 10
OUTPUT = Path(__file__).resolve().parent.parent / "output"


def windfall_study(prices, cash_rate):
    strategies = {
        "Lump sum": lambda: InvestImmediately(TICKER),
        "Spread 3 mo": lambda: SpreadOut(TICKER, 3),
        "Spread 6 mo": lambda: SpreadOut(TICKER, 6),
        "Spread 12 mo": lambda: SpreadOut(TICKER, 12),
    }
    results = rolling_backtests(strategies, prices, lambda start, end: lump_sum(12_000, start),
                                HORIZON, cash_rate, first_start="1994-01-01")
    report("Windfall: $12,000 at once", results, baseline="Lump sum")
    return results


def paycheck_study(prices, cash_rate):
    strategies = {
        "DCA": lambda: InvestImmediately(TICKER),
        "Buy the dip 10%": lambda: BuyTheDip(TICKER, 0.10),
        "DCA 80% + dip reserve": lambda: DCAWithDipReserve(TICKER, 0.8),
    }
    results = rolling_backtests(strategies, prices, lambda start, end: monthly(500, start, end),
                                HORIZON, cash_rate, first_start="1994-01-01")
    report("Paycheck: $500 every month", results, baseline="DCA")
    return results


def report(title, results, baseline):
    print(f"\n=== {title} | {HORIZON}-year windows, {results['start'].nunique()} start months ===")
    summary = results.groupby("strategy", sort=False).agg(
        median_final=("final_value", "median"),
        worst_final=("final_value", "min"),
        median_irr=("irr", "median"),
        median_max_dd=("max_drawdown", "median"),
        avg_cash=("avg_cash_share", "mean"),
    )
    print(summary.to_string(formatters={
        "median_final": "${:,.0f}".format, "worst_final": "${:,.0f}".format,
        "median_irr": "{:.2%}".format, "median_max_dd": "{:.1%}".format, "avg_cash": "{:.1%}".format,
    }))
    for name in summary.index.drop(baseline):
        h = head_to_head(results, name, baseline)
        print(f"  {name} beat {baseline} in {h['a_win_rate']:.0%} of windows "
              f"(median edge {h['median_edge']:+.1%}, range {h['worst_edge']:+.1%} to {h['best_edge']:+.1%})")


def plot(results, title, path):
    wide = results.pivot(index="start", columns="strategy", values="final_value")
    ax = wide.plot(figsize=(10, 5), title=title)
    ax.set_xlabel(f"Start month (each line point = a {HORIZON}-year backtest)")
    ax.set_ylabel("Ending value ($)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    prices = load_prices(TICKER, start="1993-01-01")
    cash_rate = load_cash_rate()
    print(f"{TICKER}: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}")

    OUTPUT.mkdir(exist_ok=True)
    windfall = windfall_study(prices, cash_rate)
    paycheck = paycheck_study(prices, cash_rate)
    plot(windfall, "Windfall: lump sum vs spreading it out", OUTPUT / "windfall.png")
    plot(paycheck, "Monthly investing: DCA vs dip strategies", OUTPUT / "paycheck.png")
    pd.concat([windfall.assign(study="windfall"), paycheck.assign(study="paycheck")]) \
        .to_csv(OUTPUT / "results.csv", index=False)
    print(f"\nCharts and full results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
