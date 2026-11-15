"""Null-hypothesis detector for permutation-style tests: same regime gate, same stop/exit logic, RANDOM timing.
If the real setup cannot beat this distribution, its 'edge' is just exposure to the trend."""
from __future__ import annotations
import numpy as np
from .base import SetupDetector
from ..types import Candidate, Regime, Side


class RandomEntry(SetupDetector):
    name = "random_entry"
    regimes = {Regime.STRONG_BULL}

    def __init__(self, prob: float, stop_atr: float, seed: int):
        self.prob, self.k, self.rng = prob, stop_atr, np.random.default_rng(seed)

    def detect(self, ctx):
        if self.rng.random() >= self.prob:
            return []
        c, a = float(ctx.last["close"]), float(ctx.last["atr"])
        return [Candidate(ctx.symbol, Side.LONG, self.name, c, c - self.k * a,
                          confirmations={"trend": True, "momentum": True})]
