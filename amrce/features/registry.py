from __future__ import annotations
from typing import Dict, Type
from .base import FeatureModule, FeaturePipeline, NotBuilt
from .trend import TrendModule
from .volatility import VolatilityModule
from .momentum import MomentumModule


def _stub(name, family):
    return type(f"{name.title().replace('_','')}Module", (NotBuilt,), {"name": name, "family": family})


REGISTRY: Dict[str, Type[FeatureModule]] = {
    "trend": TrendModule,
    "volatility": VolatilityModule,
    "momentum": MomentumModule,
    "price_action": _stub("price_action", "price_action"),  # engulfing, pin bar, Heikin-Ashi runs
    "structure": _stub("structure", "structure"),          # pivots, zones, HH/HL, role reversal
    "breakout": _stub("breakout", "breakout"),             # squeeze / NR7 / correction breakout
}


def build_feature_pipeline(cfg) -> FeaturePipeline:
    mods = []
    for n in cfg.features.enabled:
        if n not in REGISTRY:
            raise KeyError(f"unknown feature module '{n}'")
        mods.append(REGISTRY[n](cfg.features.params))
    return FeaturePipeline(mods)
