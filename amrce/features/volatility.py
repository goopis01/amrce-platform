from __future__ import annotations
import pandas as pd
from .base import FeatureModule
from . import indicators as ind


class VolatilityModule(FeatureModule):
    """MODULE 5. Reference implementation: BB width percentile, ATR ratio/percentile, NR7, range spikes."""
    name, family = "volatility", "volatility"

    def compute(self, bars):
        p = self.p
        a = ind.atr(bars, p["atr_n"])
        bb = ind.bollinger(bars["close"], p["bb_n"], p["bb_k"])
        return pd.DataFrame({
            "atr_slow": ind.atr(bars, p["atr_slow_n"]),
            "atr_ratio": a / ind.atr(bars, p["atr_slow_n"]),
            "atr_pct": ind.rolling_pct_rank(a, p["atr_pct_lookback"]),
            "bbw": bb["bbw"],
            "bbw_pct": ind.rolling_pct_rank(bb["bbw"], p["bbw_lookback"]),
            "nr7": ind.narrow_range(bars, 7),
            "range_atr": (bars["high"] - bars["low"]) / a,
        }, index=bars.index)
