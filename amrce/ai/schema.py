"""Strict reviewer output schema + validation. Anything invalid becomes a fail-closed veto."""
from __future__ import annotations
from typing import Any, Dict
from ..types import Verdict

REQUIRED = ["schema_version", "action", "confidence", "regime", "setup", "entry", "stop_loss",
            "take_profit_1", "take_profit_2", "risk_reward", "position_size", "signal_score",
            "size_multiplier", "veto", "invalidation", "reason", "risk_flags"]

CLAUDE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": REQUIRED,
    "properties": {
        "schema_version": {"const": "1.0"}, "action": {"enum": ["BUY", "SELL", "HOLD"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 100},
        "regime": {"type": "string"}, "setup": {"type": "string"},
        "entry": {"type": "number"}, "stop_loss": {"type": "number"},
        "take_profit_1": {"type": "number"}, "take_profit_2": {"type": "number"},
        "risk_reward": {"type": "number", "minimum": 0}, "position_size": {"type": "number", "minimum": 0},
        "signal_score": {"type": "number", "minimum": 0, "maximum": 100},
        "size_multiplier": {"type": "number", "minimum": 0, "maximum": 1}, "veto": {"type": "boolean"},
        "invalidation": {"type": "string"}, "reason": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
    },
}


def validate_verdict(d: Dict[str, Any]) -> Verdict:
    if not isinstance(d, dict):
        raise ValueError("verdict must be an object")
    extra, missing = set(d) - set(REQUIRED), [k for k in REQUIRED if k not in d]
    if extra or missing:
        raise ValueError(f"schema mismatch extra={sorted(extra)} missing={missing}")
    if d["schema_version"] != "1.0" or d["action"] not in ("BUY", "SELL", "HOLD"):
        raise ValueError("bad schema_version/action")
    for k in ("confidence", "signal_score"):
        if not isinstance(d[k], (int, float)) or not 0 <= d[k] <= 100:
            raise ValueError(f"{k} out of range")
    if not isinstance(d["size_multiplier"], (int, float)) or not 0 <= d["size_multiplier"] <= 1:
        raise ValueError("size_multiplier out of range")
    if not isinstance(d["veto"], bool) or not isinstance(d["risk_flags"], list):
        raise ValueError("bad veto/risk_flags")
    return Verdict(action=d["action"], veto=d["veto"], size_multiplier=float(d["size_multiplier"]),
                   confidence=float(d["confidence"]), reason=str(d["reason"]),
                   risk_flags=[str(x) for x in d["risk_flags"]], raw=d)
