"""Data provider interface. Bars: tz-aware DatetimeIndex, columns open/high/low/close/volume."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


class DataProvider(ABC):
    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str, start: Optional[str] = None,
                 end: Optional[str] = None) -> pd.DataFrame: ...
