"""Feature-module contract + pipeline. Modules MUST be causal (enforced by tests/test_causality)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Type
import pandas as pd


class FeatureModule(ABC):
    name: str = ""
    family: str = ""      # trend | momentum | price_action | structure | volatility | breakout

    def __init__(self, params: dict):
        self.p = params

    @abstractmethod
    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with the same index as `bars`. Row t may use rows <= t only."""


class NotBuilt(FeatureModule):
    """Placeholder for modules scheduled for the model-building stage."""
    def compute(self, bars):
        raise NotImplementedError(f"module '{self.name}' is not built yet (roadmap M3)")


class FeaturePipeline:
    def __init__(self, modules: List[FeatureModule]):
        self.modules = modules

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = bars.copy()
        for m in self.modules:
            f = m.compute(bars)
            clash = set(f.columns) & set(out.columns)
            if clash:
                raise ValueError(f"feature name clash in {m.name}: {clash}")
            out = out.join(f)
        return out
