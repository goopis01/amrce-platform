from __future__ import annotations
import pandas as pd
from .base import FeatureModule
from . import indicators as ind


class MomentumModule(FeatureModule):
    """MODULE 2. RSI(14), RSI(2), Stochastic(14,7,3). Values only; interpretation lives in setups."""
    name, family = "momentum", "momentum"

    def compute(self, bars):
        g = self.p.get
        st = ind.stochastic(bars, g("stoch_k", 14), g("stoch_smooth", 7), g("stoch_d", 3))
        out = pd.DataFrame({"rsi": ind.rsi(bars["close"], g("rsi_n", 14)),
                            "rsi2": ind.rsi(bars["close"], g("rsi2_n", 2))}, index=bars.index)
        return out.join(st)
