"""Deposit schedules: when money arrives and how much."""

import pandas as pd


def lump_sum(amount, date) -> pd.Series:
    return pd.Series([float(amount)], index=pd.DatetimeIndex([pd.Timestamp(date)]))


def monthly(amount, start, end) -> pd.Series:
    """`amount` on the first of every month from `start` through `end`."""
    start = pd.Timestamp(start)
    first = start if start.day == 1 else start + pd.offsets.MonthBegin(1)
    dates = pd.date_range(first, end, freq="MS")
    return pd.Series(float(amount), index=dates)
