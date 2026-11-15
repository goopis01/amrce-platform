"""Daily bars from Yahoo Finance (no key, long history, split/dividend adjusted). Needs: pip install yfinance."""
from __future__ import annotations
import pandas as pd
from .base import DataProvider
from .validators import DataError, validate_bars


def _repair_ohlc(df: pd.DataFrame, tol: float = 0.002) -> pd.DataFrame:
    """Fix rounding-size OHLC inconsistencies (adjusted prices). Anything above `tol` of price still fails."""
    cols = ["open", "high", "low", "close"]
    hi, lo = df[cols].max(axis=1), df[cols].min(axis=1)
    bad_hi = (hi - df["high"]) / df["close"]
    bad_lo = (df["low"] - lo) / df["close"]
    worst = max(bad_hi.max(), bad_lo.max())
    if worst > tol:
        raise DataError(f"OHLC inconsistency too large to be rounding ({worst:.4%} of price)")
    n = int(((bad_hi > 0) | (bad_lo > 0)).sum())
    if n:
        print(f"[yahoo] repaired {n} bars with rounding-size OHLC inconsistencies")
    df = df.copy()
    df["high"], df["low"] = hi, lo
    return df


class YahooData(DataProvider):
    def get_bars(self, symbol, timeframe, start=None, end=None):
        if timeframe != "1D":
            raise ValueError("YahooData supports 1D only")
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.index = pd.DatetimeIndex(df.index).tz_localize("America/New_York")
        return validate_bars(_repair_ohlc(df))