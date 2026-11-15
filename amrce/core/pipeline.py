"""StrategyPipeline: wires every component in the order of the spec. Pure decision logic, no I/O.
Stages (first failure stops and is recorded in Decision.stage):
 data -> regime -> setup -> confirmations -> score -> plan -> review -> risk -> approved"""
from __future__ import annotations
from typing import Callable, List, Optional
import pandas as pd
from ..ai.reviewers import NullReviewer, Reviewer, build_payload
from ..features.registry import build_feature_pipeline
from ..regime.base import RegimeClassifier
from ..regime.rules import RuleRegimeClassifier
from ..risk.engine import RiskEngine
from ..risk.sizing import compute_qty
from ..signals.base import SetupDetector
from ..signals.confluence import ConfluenceScorer
from ..types import Decision, MarketContext, PortfolioState, Regime, TradePlan


class StrategyPipeline:
    def __init__(self, cfg, detectors: Optional[List[SetupDetector]] = None,
                 scorer: Optional[ConfluenceScorer] = None, regime: Optional[RegimeClassifier] = None,
                 reviewer: Optional[Reviewer] = None, risk: Optional[RiskEngine] = None):
        self.cfg = cfg
        self.features = build_feature_pipeline(cfg)
        self.detectors = detectors or []
        self.scorer = scorer or ConfluenceScorer([])
        self.regime = regime or RuleRegimeClassifier(cfg)
        self.reviewer = reviewer or NullReviewer()
        self.risk = risk or RiskEngine(cfg)

    def reset(self) -> None:
        self.regime.reset()

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return self.features.compute(bars)

    # ---- helpers ----
    def _build_plan(self, cand, regime, score, atr) -> Optional[TradePlan]:
        s, sg = self.cfg.signals, cand.side.sign
        risk = abs(cand.entry - cand.stop)
        if risk <= 0 or (cand.entry - cand.stop) * sg <= 0:
            return None
        tp1 = cand.tp1 if cand.tp1 is not None else cand.entry + sg * s.tp1_r * risk
        tp2 = cand.tp2 if cand.tp2 is not None else cand.entry + sg * s.tp2_r * risk
        cost = cand.entry * 2 * (self.cfg.costs.commission_bps + self.cfg.costs.slippage_bps) / 1e4
        rr_net = (abs(tp2 - cand.entry) - cost) / (risk + cost)
        return TradePlan(cand.symbol, cand.side, cand.setup, regime, cand.entry, cand.stop, tp1, tp2,
                         score, rr_net, atr)

    def evaluate(self, symbol: str, features: pd.DataFrame, pf: PortfolioState) -> Decision:
        s = self.cfg.signals
        if len(features) < 2 or features.iloc[-1].isna().all():
            return Decision("data", reason="insufficient data")
        regime = self.regime.classify(features)
        if regime is Regime.NO_TRADE or (regime is Regime.UNSTABLE and not self.cfg.regime.allow_unstable):
            return Decision("regime", reason=f"regime {regime.value} not tradable", regime=regime)
        ctx = MarketContext(symbol, features, regime)

        cands = [c for d in self.detectors if regime in d.regimes for c in d.detect(ctx)]
        if not cands:
            return Decision("setup", reason="no setup fired", regime=regime)
        scored = []
        for c in cands:
            if sum(bool(v) for v in c.confirmations.values()) < s.min_confirmations:
                continue
            scored.append((self.scorer.score(ctx, c), c))
        if not scored:
            return Decision("confirmations", reason="fewer than required independent confirmations", regime=regime)
        sb, cand = max(scored, key=lambda t: t[0].total)
        if sb.total < s.score_min:
            return Decision("score", reason=f"score {sb.total:.1f} < {s.score_min}", regime=regime, score=sb.total)

        atr = float(ctx.last.get("atr", float("nan")))
        plan = self._build_plan(cand, regime, sb.total, atr)
        if plan is None:
            return Decision("plan", reason="invalid entry/stop geometry", regime=regime, score=sb.total)
        if plan.rr_net < s.min_rr:
            return Decision("plan", reason=f"net R:R {plan.rr_net:.2f} < {s.min_rr}", regime=regime, plan=plan, score=sb.total)
        if atr == atr and not (s.stop_atr_min <= plan.risk_per_unit / atr <= s.stop_atr_max):
            return Decision("plan", reason="stop distance outside ATR bounds", regime=regime, plan=plan, score=sb.total)

        verdict = self.reviewer.review(build_payload(ctx, plan, sb, pf), plan)
        if verdict.veto or verdict.action == "HOLD":
            return Decision("review", reason=f"reviewer veto: {verdict.reason}", regime=regime, plan=plan,
                            score=sb.total, verdict=verdict)

        cost_pu = plan.entry * (self.cfg.costs.commission_bps + self.cfg.costs.slippage_bps) / 1e4
        qty = compute_qty(plan, pf.equity, pf.drawdown, self.cfg.sizing, cost_per_unit=cost_pu,
                          extra_mult=verdict.size_multiplier)
        rd = self.risk.approve(plan, qty, pf)
        if not rd.approved:
            return Decision("risk", reason=rd.reason, regime=regime, plan=plan, qty=qty, score=sb.total, verdict=verdict)
        return Decision("approved", approved=True, reason="ok", regime=regime, plan=plan, qty=qty,
                        score=sb.total, verdict=verdict)
