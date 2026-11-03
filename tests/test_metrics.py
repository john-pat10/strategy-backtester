import pandas as pd
import pytest

from backtester import InvestImmediately, lump_sum, monthly, rolling_backtests, run_backtest
from backtester.metrics import max_drawdown, time_weighted_returns, xirr
from backtester.rolling import head_to_head


def test_xirr_one_year_ten_percent():
    dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")]
    assert xirr(dates, [-100, 110]) == pytest.approx(0.10, abs=1e-3)


def test_xirr_zero_return():
    dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-07-01"), pd.Timestamp("2021-01-01")]
    assert xirr(dates, [-100, -100, 200]) == pytest.approx(0.0, abs=1e-6)


def test_deposits_do_not_count_as_returns():
    daily = pd.DataFrame({
        "value": [100.0, 200.0, 220.0],
        "flow": [100.0, 100.0, 0.0],
    }, index=pd.bdate_range("2020-01-01", periods=3))
    assert time_weighted_returns(daily).tolist() == pytest.approx([0.0, 0.10])


def test_max_drawdown():
    returns = pd.Series([0.10, -0.50, 0.20])
    assert max_drawdown(returns) == pytest.approx(-0.50)


def test_summary_for_steady_growth():
    dates = pd.bdate_range("2020-01-01", periods=253)
    prices = pd.DataFrame({"X": [100 * 1.10 ** (i / 252) for i in range(253)]}, index=dates)
    metrics = run_backtest(InvestImmediately("X"), prices, lump_sum(1000, dates[0])).metrics()
    assert metrics["twr_annual"] == pytest.approx(0.10, abs=2e-3)
    assert metrics["max_drawdown"] == pytest.approx(0)
    assert metrics["contributed"] == pytest.approx(1000)


def test_monthly_schedule_starts_next_month_if_mid_month():
    schedule = monthly(100, "2020-01-15", "2020-04-01")
    assert list(schedule.index.month) == [2, 3, 4]


def test_rolling_runs_every_window_and_strategy():
    dates = pd.bdate_range("2010-01-01", "2014-12-31")
    prices = pd.DataFrame({"X": range(100, 100 + len(dates))}, index=dates, dtype=float)
    results = rolling_backtests(
        {"a": lambda: InvestImmediately("X"), "b": lambda: InvestImmediately("X")},
        prices, lambda s, e: lump_sum(1000, s), horizon_years=3, step="YS",
    )
    assert len(results) == 2 * 2  # 2010 and 2011 starts fit a 3-year window
    edge = head_to_head(results, "a", "b")
    assert edge["median_edge"] == pytest.approx(0)
