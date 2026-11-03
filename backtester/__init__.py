from .engine import BacktestResult, Context, Order, Portfolio, Strategy, run_backtest
from .rolling import head_to_head, rolling_backtests
from .schedules import lump_sum, monthly
from .strategies import BuyTheDip, DCAWithDipReserve, InvestImmediately, SpreadOut
