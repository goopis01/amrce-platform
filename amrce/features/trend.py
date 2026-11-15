from __future__ import annotations
import pandas as pd
from .base import FeatureModule
from . import indicators as ind


class TrendModule(FeatureModule):
    """MODULE 1. Reference implementation: MA stack, slope, separation (all in ATR units)."""
    name, family = "trend", "trend"

    def compute(self, bars):
        p = self.p
        c = bars["close"]
        e_f, e_s, e_t = ind.ema(c, p["ema_fast"]), ind.ema(c, p["ema_slow"]), ind.ema(c, p["ema_trend"])
        a = ind.atr(bars, p["atr_n"])
        k = p["slope_lookback"]
        return pd.DataFrame({
            "ema_fast": e_f, "ema_slow": e_s, "ema_trend": e_t, "atr": a,
            "slope_slow_atr": (e_s - e_s.shift(k)) / (k * a),
            "sep_atr": (e_f - e_s) / a,
            "price_vs_trend_atr": (c - e_t) / a,
            "ma_bull": (e_f > e_s) & (e_s > e_t),
            "ma_bear": (e_f < e_s) & (e_s < e_t),
        }, index=bars.index)
