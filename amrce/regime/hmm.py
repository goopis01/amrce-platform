"""HMM regime context. STUB (roadmap M4). Rules for the real implementation (spec section 9):
 - forward-FILTERED posteriors only (no smoothing/Viterbi over a window containing the future)
 - monthly rolling refit, frozen between refits; label states by sorted mean/vol, never by raw index
 - output is context (size multiplier / veto / score nudge), never a standalone signal
 - disabled automatically if history < ~1,250 bars
"""
from __future__ import annotations
from typing import Dict
import pandas as pd


class HMMContext:
    enabled = False

    def posterior(self, features: pd.DataFrame) -> Dict[str, float]:
        raise NotImplementedError("HMM layer not built yet (roadmap M4)")
