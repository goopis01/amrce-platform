from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Set
from ..types import Candidate, MarketContext, Regime


class SetupDetector(ABC):
    name: str = ""
    regimes: Set[Regime] = set()      # regimes in which this setup is allowed to fire

    @abstractmethod
    def detect(self, ctx: MarketContext) -> List[Candidate]:
        """Closed-bar logic only. Return [] when nothing fires."""


class NotBuiltDetector(SetupDetector):
    def detect(self, ctx):
        raise NotImplementedError(f"setup '{self.name}' not built yet (roadmap M3)")


def _stub(name, regimes):
    return type(f"Setup_{name}", (NotBuiltDetector,), {"name": name, "regimes": set(regimes)})


R = Regime
SETUP_REGISTRY = {
    "s1_pullback_role_reversal": _stub("s1_pullback_role_reversal", [R.STRONG_BULL, R.STRONG_BEAR]),
    "s2_correction_breakout": _stub("s2_correction_breakout", [R.STRONG_BULL, R.STRONG_BEAR]),
    "s3_zone_fade": _stub("s3_zone_fade", [R.RANGE]),
    "s4_squeeze_breakout": _stub("s4_squeeze_breakout", [R.COMPRESSION, R.EXPANSION]),
    "s5_retest_continuation": _stub("s5_retest_continuation", [R.EXPANSION]),
}
