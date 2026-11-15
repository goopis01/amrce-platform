"""Reviewer layer. The reviewer can confirm, downgrade or veto. It can NEVER enlarge a trade.
Modes: off -> NullReviewer, shadow -> ShadowReviewer (logs verdict, always confirms), active -> ClaudeReviewer."""
from __future__ import annotations
import json, re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from ..types import Side, TradePlan, Verdict
from .schema import validate_verdict

SYSTEM_PROMPT = (
    "You are a risk-aware trade reviewer for a rules-based system. You do not predict prices. "
    "You receive a candidate trade as JSON and must return ONLY a JSON object matching the schema "
    "(no prose, no markdown). Prefer HOLD with veto=true when evidence conflicts. You may lower "
    "size_multiplier (0-1) but never raise risk. List concrete risk_flags."
)


class Reviewer(ABC):
    @abstractmethod
    def review(self, payload: Dict[str, Any], plan: TradePlan) -> Verdict: ...


class NullReviewer(Reviewer):
    def review(self, payload, plan):
        return Verdict.confirm(plan.side, "reviewer off")


class ShadowReviewer(Reviewer):
    """Runs the real reviewer but ignores its answer; stores both for later forward-validation."""
    def __init__(self, inner: Reviewer):
        self.inner, self.log = inner, []

    def review(self, payload, plan):
        shadow = self.inner.review(payload, plan)
        self.log.append({"ts": payload.get("ts"), "symbol": plan.symbol, "shadow": shadow})
        return Verdict.confirm(plan.side, "shadow mode")


class ClaudeReviewer(Reviewer):
    """Calls the Anthropic API. UNTESTED against the live API in this scaffold; `client_fn` is injectable."""
    def __init__(self, model: str, timeout_s: float = 10, client_fn: Optional[Callable[[str, str], str]] = None):
        self.model, self.timeout_s = model, timeout_s
        self.client_fn = client_fn or self._anthropic_call

    def _anthropic_call(self, system: str, user: str) -> str:
        import anthropic
        c = anthropic.Anthropic(timeout=self.timeout_s)
        msg = c.messages.create(model=self.model, max_tokens=800, temperature=0, system=system,
                                messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    def review(self, payload, plan):
        try:
            text = self.client_fn(SYSTEM_PROMPT, json.dumps(payload, default=str))
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            v = validate_verdict(json.loads(text))
            if v.action != ("BUY" if plan.side is Side.LONG else "SELL"):
                v.veto = True                      # disagreement with the engine's side = veto
            return v
        except Exception as e:                      # timeout, bad JSON, schema failure, API error
            return Verdict.fail_closed(type(e).__name__)


def build_payload(ctx, plan: TradePlan, score, portfolio) -> Dict[str, Any]:
    """Structured, backward-looking context only (no future data)."""
    last = ctx.last
    keep = ["close", "atr", "ema_fast", "ema_slow", "ema_trend", "slope_slow_atr", "sep_atr",
            "bbw_pct", "atr_ratio", "atr_pct", "volume"]
    return {
        "ts": str(ctx.ts), "symbol": ctx.symbol, "regime": ctx.regime.value, "setup": plan.setup,
        "indicators": {k: (None if k not in last or last[k] != last[k] else float(last[k])) for k in keep},
        "plan": {"side": plan.side.value, "entry": plan.entry, "stop": plan.stop, "tp1": plan.tp1,
                 "tp2": plan.tp2, "rr_net": plan.rr_net},
        "score": {"total": score.total, "parts": score.parts},
        "portfolio": {"equity": portfolio.equity, "drawdown": portfolio.drawdown,
                      "open_positions": len(portfolio.positions), "consecutive_losses": portfolio.consecutive_losses},
    }
