"""Rule-based regime classifier (spec section 7). Priority: G > F > D > E > A/B > C.
Structure (HH/HL) confirmation is NOT included yet; it arrives with the structure module (M3)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import RegimeClassifier
from ..types import Regime

NEEDED = ["slope_slow_atr", "sep_atr", "price_vs_trend_atr", "bbw_pct", "atr_ratio", "atr_pct", "range_atr"]


class RuleRegimeClassifier(RegimeClassifier):
    def __init__(self, cfg):
        self.c = cfg.regime
        self.reset()

    def reset(self):
        self._current, self._pending, self._count = Regime.NO_TRADE, None, 0

    def _raw(self, f: pd.DataFrame) -> Regime:
        c, r = self.c, f.iloc[-1]
        if r[NEEDED].isna().any():
            return Regime.NO_TRADE
        if r["atr_pct"] >= c.unstable_atr_pct or r["range_atr"] > c.unstable_range_atr:
            return Regime.UNSTABLE
                # NR7 is deliberately NOT part of the regime test (it occurs on ~14% of bars by chance).
        if r["bbw_pct"] <= c.squeeze_pct_max and r["atr_ratio"] < c.atr_ratio_squeeze:
            return Regime.COMPRESSION
        recent = f["bbw_pct"].iloc[-11:-1]
        if (len(recent) and recent.min() <= c.expansion_from_pct
                and r["bbw_pct"] >= c.expansion_to_pct and r["atr_ratio"] > c.atr_ratio_expansion):
            return Regime.EXPANSION
        if (bool(r["ma_bull"]) and r["price_vs_trend_atr"] > 0
                and r["slope_slow_atr"] > c.trend_slope_min and r["sep_atr"] > c.trend_sep_min):
            return Regime.STRONG_BULL
        if (bool(r["ma_bear"]) and r["price_vs_trend_atr"] < 0
                and r["slope_slow_atr"] < -c.trend_slope_min and r["sep_atr"] < -c.trend_sep_min):
            return Regime.STRONG_BEAR
        side = np.sign((f["close"] - f["ema_slow"]).iloc[-31:])
        crossings = int((side.diff().abs() > 0).sum())
        if abs(r["slope_slow_atr"]) < c.range_slope_max and crossings >= c.range_crossings_min:
            return Regime.RANGE
        return Regime.NO_TRADE

    def classify(self, features: pd.DataFrame) -> Regime:
        raw = self._raw(features)
        if raw == self._current:
            self._pending, self._count = None, 0
        elif raw == self._pending:
            self._count += 1
            if self._count >= self.c.hysteresis_bars:
                self._current, self._pending, self._count = raw, None, 0
        else:
            self._pending, self._count = raw, 1
            if self.c.hysteresis_bars <= 1:
                self._current, self._pending = raw, None
        return self._current
