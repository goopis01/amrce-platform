from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from .base import DataProvider
from .validators import validate_bars


class CSVData(DataProvider):
    """Reads <dir>/<SYMBOL>_<timeframe>.csv with columns: timestamp,open,high,low,close,volume."""

    def __init__(self, directory: str, tz: str = "America/New_York"):
        self.dir, self.tz = Path(directory), tz

    def get_bars(self, symbol, timeframe, start=None, end=None):
        df = pd.read_csv(self.dir / f"{symbol}_{timeframe}.csv", index_col="timestamp")
        df.columns = [c.lower() for c in df.columns]
        raw = df.index.astype(str)
        has_offset = raw.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$", regex=True).any()
        idx = pd.to_datetime(raw, utc=True) if has_offset else pd.to_datetime(raw)
        df.index = idx.tz_convert(self.tz) if has_offset else idx.tz_localize(self.tz)
        if start:
            df = df[df.index >= pd.Timestamp(start, tz=self.tz)]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz=self.tz)]
        return validate_bars(df.sort_index())
