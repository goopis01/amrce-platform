"""Deterministic risk engine: final authority over whether a trade may be opened (spec section 6)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
from ..types import PortfolioState, TradePlan
from .killswitch import KillSwitch


@dataclass
class RiskDecision:
    approved: bool
    reason: str = "ok"


class RiskEngine:
    def __init__(self, cfg, killswitch: Optional[KillSwitch] = None,
                 bucket_of: Optional[Callable[[str], str]] = None):
        self.c, self.cfg = cfg.risk, cfg
        self.kill = killswitch or KillSwitch()
        self.bucket_of = bucket_of or (lambda sym: sym)   # default: every symbol its own bucket

    def approve(self, plan: TradePlan, qty: float, pf: PortfolioState) -> RiskDecision:
        c = self.c
        if self.kill.active():
            return RiskDecision(False, f"kill switch: {self.kill.reason or 'flag'}")
        if qty <= 0:
            return RiskDecision(False, "qty<=0 after sizing")
        if pf.daily_loss_pct >= c.max_daily_loss:
            self.kill.trip("daily loss limit")
            return RiskDecision(False, "daily loss limit hit (HARD STOP)")
        if pf.weekly_loss_pct >= c.max_weekly_loss:
            return RiskDecision(False, "weekly loss limit hit")
        if pf.drawdown >= c.max_drawdown:
            return RiskDecision(False, "max drawdown halt (revalidation required)")
        if len(pf.positions) >= c.max_positions:
            return RiskDecision(False, "max simultaneous positions")
        if pf.cooldown_until and pf.today and pf.today <= pf.cooldown_until:
            return RiskDecision(False, "loss-streak cooldown active")
        if pf.consecutive_losses >= c.loss_streak_cooldown:
            return RiskDecision(False, "loss-streak cooldown")
        if pf.stops_today.get(plan.symbol, 0) >= c.max_stops_per_symbol_day:
            return RiskDecision(False, "symbol locked for today (repeated stops)")
        new_risk = qty * plan.risk_per_unit
        bucket = self.bucket_of(plan.symbol)
        bucket_risk = sum(p.risk_usd for p in pf.positions if self.bucket_of(p.symbol) == bucket) + new_risk
        cap = c.bucket_risk_cap_mult * self.cfg.sizing.base_risk * pf.equity
        if bucket_risk > cap + 1e-9:
            return RiskDecision(False, f"correlation bucket '{bucket}' risk cap")
        gross = (sum(p.notional for p in pf.positions) + qty * plan.entry) / pf.equity
        if gross > c.gross_exposure_max + 1e-9:
            return RiskDecision(False, "gross exposure cap")
        return RiskDecision(True)
