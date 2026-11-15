"""Position sizing (spec section 5). Size NEVER grows with score or reviewer confidence."""
from __future__ import annotations
import math
from ..types import Regime, TradePlan


def drawdown_mult(drawdown: float, dd_halt: float, floor: float) -> float:
    return max(floor, 1.0 - drawdown / dd_halt) if dd_halt > 0 else 1.0


def compute_qty(plan: TradePlan, equity: float, drawdown: float, cfg_sizing, cost_per_unit: float = 0.0,
                point_value: float = 1.0, lot: float = 1.0, extra_mult: float = 1.0) -> float:
    reg_mult = cfg_sizing.regime_mult.get(plan.regime.value, 0.0)
    risk_pct = cfg_sizing.base_risk * reg_mult * drawdown_mult(drawdown, cfg_sizing.dd_halt, cfg_sizing.min_dd_mult)
    risk_pct *= min(max(extra_mult, 0.0), 1.0)           # reviewer can only lower
    stop_dist = plan.risk_per_unit + cost_per_unit
    if stop_dist <= 0 or equity <= 0 or risk_pct <= 0:
        return 0.0
    qty_risk = equity * risk_pct / (stop_dist * point_value)
    qty_notional = equity * cfg_sizing.max_position_pct / plan.entry
    return math.floor(min(qty_risk, qty_notional) / lot) * lot
