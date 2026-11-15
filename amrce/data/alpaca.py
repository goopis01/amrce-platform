"""US stocks via Alpaca (free IEX feed). Needs: pip install alpaca-py ; env ALPACA_KEY / ALPACA_SECRET."""
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
from .base import DataProvider
from .validators import validate_bars

_TF = {"1D": (1, "Day"), "1h": (1, "Hour"), "15m": (15, "Minute"), "30m": (30, "Minute"), "5m": (5, "Minute")}


class AlpacaData(DataProvider):
    def get_bars(self, symbol, timeframe, start=None, end=None):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import Adjustment, DataFeed
        n, unit = _TF[timeframe]
        c = StockHistoricalDataClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"])
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame(n, getattr(TimeFrameUnit, unit)),
            start=pd.Timestamp(start, tz="UTC").to_pydatetime() if start else datetime.now(timezone.utc) - timedelta(days=365),
            end=pd.Timestamp(end, tz="UTC").to_pydatetime() if end else None, feed=DataFeed.IEX, adjustment=Adjustment.ALL)
        df = c.get_stock_bars(req).df.reset_index(level=0, drop=True)
        df.index = df.index.tz_convert("America/New_York")
        return validate_bars(df[["open", "high", "low", "close", "volume"]])
