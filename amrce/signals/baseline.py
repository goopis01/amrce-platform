"""BASELINE strategy (benchmark everything else must beat): in a strong bull regime, buy RSI(2) pullbacks.
Confirmations: trend (regime) + momentum (RSI2 oversold). Stop = N x ATR. Exits/targets come from the pipeline."""
from __future__ import annotations
from typing import List
from .base import SetupDetector
from .confluence import ScoreComponent
from ..types import Candidate, MarketContext, Regime, Side


class BaselinePullback(SetupDetector):
    name = "baseline_pullback"
    regimes = {Regime.STRONG_BULL, Regime.STRONG_BEAR}

    def __init__(self, params: dict):
        self.p = {"rsi2_entry": 10, "stop_atr": 2.0, "shorts": False, **(params or {})}

    def detect(self, ctx: MarketContext) -> List[Candidate]:
        r, p = ctx.last, self.p
        if r["rsi2"] != r["rsi2"] or r["atr"] != r["atr"]:
            return []
        c, a = float(r["close"]), float(r["atr"])
        if ctx.regime is Regime.STRONG_BULL and r["rsi2"] < p["rsi2_entry"]:
            return [Candidate(ctx.symbol, Side.LONG, self.name, c, c - p["stop_atr"] * a,
                              confirmations={"trend": True, "momentum": True})]
        if p["shorts"] and ctx.regime is Regime.STRONG_BEAR and r["rsi2"] > 100 - p["rsi2_entry"]:
            return [Candidate(ctx.symbol, Side.SHORT, self.name, c, c + p["stop_atr"] * a,
                              confirmations={"trend": True, "momentum": True})]
        return []


class TrendAlignmentScore(ScoreComponent):
    name, max_points = "trend_alignment", 20.0

    def score(self, ctx, cand):
        r, up = ctx.last, cand.side is Side.LONG
        aligned = bool(r["ma_bull"] if up else r["ma_bear"])
        slope = r["slope_slow_atr"] * (1 if up else -1)
        return 8 * aligned + 8 * min(max(slope / 0.3, 0), 1) + 4 * (r["price_vs_trend_atr"] * (1 if up else -1) > 0)


class MomentumScore(ScoreComponent):
    name, max_points = "momentum", 15.0

    def score(self, ctx, cand):
        x = ctx.last["rsi2"] if cand.side is Side.LONG else 100 - ctx.last["rsi2"]
        return 15 * min(max((20 - x) / 20, 0), 1)       # deeper pullback -> more points
