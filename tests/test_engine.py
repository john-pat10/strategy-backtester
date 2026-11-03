import numpy as np
import pandas as pd
import pytest

from backtester import InvestImmediately, Order, Strategy, lump_sum, run_backtest
from backtester.engine import align_to_trading_days


def make_prices(values, start="2020-01-01"):
    dates = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({"X": values}, index=dates, dtype=float)


def test_lump_sum_tracks_price():
    prices = make_prices([10, 12, 15, 20])
    result = run_backtest(InvestImmediately("X"), prices, lump_sum(1000, prices.index[0]))
    assert result.daily["value"].tolist() == pytest.approx([1000, 1200, 1500, 2000])


def test_dca_buys_more_shares_when_cheap():
    prices = make_prices([10, 20, 20])
    deposits = pd.Series(100.0, index=prices.index[:2])
    result = run_backtest(InvestImmediately("X"), prices, deposits)
    # 10 shares at $10 + 5 shares at $20 = 15 shares, worth $300 at $20
    assert result.daily["value"].iloc[-1] == pytest.approx(300)


def test_cash_earns_interest_daily():
    prices = make_prices([10] * 253)
    hold_cash = Strategy()
    result = run_backtest(hold_cash, prices, lump_sum(1000, prices.index[0]), cash_rate=0.05)
    # 252 days of interest after the deposit day = one year at 5%
    assert result.daily["value"].iloc[-1] == pytest.approx(1050)


def test_fees_reduce_shares_bought():
    prices = make_prices([10, 10])
    result = run_backtest(InvestImmediately("X"), prices, lump_sum(1000, prices.index[0]), fee_rate=0.01)
    assert result.daily["value"].iloc[-1] == pytest.approx(990)


def test_cannot_spend_more_than_cash():
    class Greedy(Strategy):
        def on_day(self, ctx):
            return [Order("X", 1_000_000)]

    prices = make_prices([10, 10])
    result = run_backtest(Greedy(), prices, lump_sum(500, prices.index[0]))
    assert result.daily["value"].iloc[-1] == pytest.approx(500)
    assert result.daily["cash"].iloc[-1] == pytest.approx(0)


def test_sell_order_returns_cash():
    class BuyThenSellHalf(Strategy):
        def on_day(self, ctx):
            if ctx.day == 0:
                return [Order("X", ctx.portfolio.cash)]
            if ctx.day == 1:
                return [Order("X", -ctx.portfolio.shares["X"] * ctx.price("X") / 2)]
            return []

    prices = make_prices([10, 20, 40])
    result = run_backtest(BuyThenSellHalf(), prices, lump_sum(100, prices.index[0]))
    # 10 shares; sell 5 at $20 -> $100 cash; 5 shares at $40 = $200
    assert result.daily["cash"].iloc[-1] == pytest.approx(100)
    assert result.daily["value"].iloc[-1] == pytest.approx(300)


def test_strategy_never_sees_future_prices():
    seen = []

    class Spy(Strategy):
        def on_day(self, ctx):
            seen.append((ctx.history("X")[-1], ctx.price("X")))
            return []

    prices = make_prices([1, 2, 3, 4, 5])
    run_backtest(Spy(), prices, pd.Series(dtype=float), start=prices.index[1], end=prices.index[3])
    assert seen == [(2, 2), (3, 3), (4, 4)]


def test_history_includes_data_before_start():
    lengths = []

    class Look(Strategy):
        def on_day(self, ctx):
            lengths.append(len(ctx.history("X")))
            return []

    prices = make_prices([1, 2, 3, 4])
    run_backtest(Look(), prices, pd.Series(dtype=float), start=prices.index[2])
    assert lengths == [3, 4]


def test_weekend_deposit_moves_to_next_trading_day():
    dates = pd.bdate_range("2024-01-05", periods=3)  # Fri, Mon, Tue
    deposits = pd.Series([100.0, 50.0], index=pd.to_datetime(["2024-01-06", "2024-01-07"]))
    aligned = align_to_trading_days(deposits, dates)
    assert aligned.tolist() == [0, 150, 0]


def test_deposits_after_end_are_ignored():
    dates = pd.bdate_range("2024-01-01", periods=2)
    deposits = pd.Series([100.0], index=pd.to_datetime(["2030-01-01"]))
    assert align_to_trading_days(deposits, dates).sum() == 0


def test_cash_rate_series_is_forward_filled():
    prices = make_prices([10] * 4, start="2024-01-01")
    # 0% until Jan 3, then 100%/yr; the Dec 1 value carries forward into January
    rate = pd.Series([0.0, 1.0], index=pd.to_datetime(["2023-12-01", "2024-01-03"]))
    result = run_backtest(Strategy(), prices, lump_sum(100, prices.index[0]), cash_rate=rate)
    assert result.daily["value"].tolist() == pytest.approx(
        [100, 100, 100 * 2 ** (1 / 252), 100 * 2 ** (2 / 252)])
