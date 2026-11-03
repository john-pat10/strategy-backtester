"""Download and cache market data.

Prices come from Yahoo Finance (via yfinance) and are dividend-adjusted, so
returns are total returns. Cash yields come from FRED's 3-month Treasury bill
series (DTB3), a stand-in for what a high-yield savings account paid.
"""

import warnings
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def load_prices(tickers, start="1990-01-01", end=None, refresh=False) -> pd.DataFrame:
    """Daily total-return prices, one column per ticker."""
    if isinstance(tickers, str):
        tickers = [tickers]
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"prices_{'_'.join(sorted(tickers))}.csv"

    if cache.exists() and not refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # yfinance's own pandas deprecation noise
            raw = yf.download(tickers, start="1970-01-01", auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError(f"No price data returned for {tickers}")
        prices = raw["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])
        prices = prices[tickers].dropna(how="all")
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices.to_csv(cache)

    return prices.loc[start:end].dropna()


def load_cash_rate(series="DTB3", refresh=False) -> pd.Series:
    """Annual cash yield as a decimal (0.05 = 5%), indexed by date."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"fred_{series}.csv"

    if cache.exists() and not refresh:
        raw = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        raw = pd.read_csv(FRED_CSV_URL.format(series=series), index_col=0,
                          parse_dates=True, na_values=".")
        raw.to_csv(cache)

    return (raw.iloc[:, 0] / 100).ffill().dropna().rename("cash_rate")
