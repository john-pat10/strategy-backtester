import pandas as pd
import pytest

from backtester import BuyTheDip, DCAWithDipReserve, SpreadOut, lump_sum, run_backtest


def make_prices(values, dates=None):
    dates = dates if dates is not None else pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({"X": values}, index=dates, dtype=float)


def month_starts(n, year=2020):
    return pd.DatetimeIndex([pd.Timestamp(year, 1, 1) + pd.DateOffset(months=i) for i in range(n)])


def test_spread_out_invests_equal_monthly_pieces():
    prices = make_prices([10, 20, 40, 40], dates=month_starts(4))
    result = run_backtest(SpreadOut("X", months=3), prices, lump_sum(300, prices.index[0]))
    # $100 each at $10, $20, $40 -> 10 + 5 + 2.5 = 17.5 shares at $40
    assert result.daily["value"].iloc[-1] == pytest.approx(700)
    assert result.daily["cash"].iloc[-1] == pytest.approx(0)


def test_spread_out_only_trades_once_per_month():
    dates = pd.bdate_range("2020-01-01", "2020-02-28")
    prices = make_prices([10] * len(dates), dates=dates)
    result = run_backtest(SpreadOut("X", months=12), prices, lump_sum(1200, dates[0]))
    assert result.daily["cash"].iloc[-1] == pytest.approx(1000)


def test_buy_the_dip_waits_for_threshold():
    prices = make_prices([100, 95, 91, 89, 100])
    result = run_backtest(BuyTheDip("X", threshold=0.10), prices, lump_sum(890, prices.index[0]))
    cash = result.daily["cash"].tolist()
    assert cash[:3] == pytest.approx([890, 890, 890])  # -5%, -9%: still waiting
    assert cash[3] == pytest.approx(0)                 # -11%: buys 10 shares at $89
    assert result.daily["value"].iloc[-1] == pytest.approx(1000)


def test_buy_the_dip_stays_in_cash_if_no_dip():
    prices = make_prices([100, 101, 102, 103])
    result = run_backtest(BuyTheDip("X", threshold=0.10), prices, lump_sum(500, prices.index[0]))
    assert result.daily["cash"].iloc[-1] == pytest.approx(500)


def test_dip_reserve_splits_deposits():
    prices = make_prices([100, 100])
    strategy = DCAWithDipReserve("X", invest_fraction=0.8, levels=(0.10, 0.20))
    result = run_backtest(strategy, prices, lump_sum(1000, prices.index[0]))
    assert result.daily["cash"].iloc[-1] == pytest.approx(200)


def test_dip_reserve_deploys_one_share_per_level():
    prices = make_prices([100, 89, 88, 79, 70])
    strategy = DCAWithDipReserve("X", invest_fraction=0.0, levels=(0.10, 0.20))
    result = run_backtest(strategy, prices, lump_sum(1000, prices.index[0]))
    cash = result.daily["cash"].tolist()
    assert cash[1] == pytest.approx(500)  # -11%: half the reserve
    assert cash[2] == pytest.approx(500)  # same level, nothing new
    assert cash[3] == pytest.approx(0)    # -21%: the rest
    assert cash[4] == pytest.approx(0)


def test_dip_reserve_levels_reset_after_new_high():
    deposit_days = [0, 3]
    prices = make_prices([100, 89, 101, 101, 90])
    deposits = pd.Series(1000.0, index=prices.index[deposit_days])
    strategy = DCAWithDipReserve("X", invest_fraction=0.0, levels=(0.10,))
    result = run_backtest(strategy, prices, deposits)
    # first dip uses the first $1000; after the new high the level re-arms
    assert result.daily["cash"].tolist()[1] == pytest.approx(0)
    assert result.daily["cash"].tolist()[3] == pytest.approx(1000)
    assert result.daily["cash"].tolist()[4] == pytest.approx(0)
