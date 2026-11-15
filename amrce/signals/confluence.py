"""Confluence score. Components return points in [0, max_points]; total is normalised to 0-100 over the
components that are ACTIVE, so disabling a component (ablation) never silently lowers every score."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List
from ..types import Candidate, MarketContext


class ScoreComponent(ABC):
    name: str = ""
    max_points: float = 10.0

    @abstractmethod
    def score(self, ctx: MarketContext, cand: Candidate) -> float: ...


@dataclass
class ScoreBreakdown:
    total: float
    parts: Dict[str, float] = field(default_factory=dict)


class ConfluenceScorer:
    def __init__(self, components: List[ScoreComponent]):
        self.components = components

    def score(self, ctx: MarketContext, cand: Candidate) -> ScoreBreakdown:
        if not self.components:
            return ScoreBreakdown(0.0, {})
        parts = {c.name: min(max(float(c.score(ctx, cand)), 0.0), c.max_points) for c in self.components}
        denom = sum(c.max_points for c in self.components)
        return ScoreBreakdown(100.0 * sum(parts.values()) / denom, parts)
