"""Fail-closed data checks. Bad data must stop trading, not be silently 'fixed'."""
from __future__ import annotations
import pandas as pd
from .base import OHLCV


class DataError(Exception):
    pass


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise DataError(f"missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataError("index must be DatetimeIndex")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise DataError("index must be sorted and unique")
    if df[OHLCV].isna().any().any():
        raise DataError("NaN in OHLCV")
    bad = (df.high < df[["open", "close", "low"]].max(axis=1)) | (df.low > df[["open", "close", "high"]].min(axis=1))
    if bad.any():
        raise DataError(f"{int(bad.sum())} bars with inconsistent OHLC")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataError("non-positive prices")
    return df


def is_fresh(last_ts: pd.Timestamp, now: pd.Timestamp, max_age: pd.Timedelta) -> bool:
    return (now - last_ts) <= max_age
