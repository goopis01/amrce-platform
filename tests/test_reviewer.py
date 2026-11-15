import json, pytest
from amrce.ai.reviewers import ClaudeReviewer, NullReviewer, ShadowReviewer
from amrce.ai.schema import validate_verdict
from amrce.types import Regime, Side, TradePlan

PLAN = TradePlan("AAA", Side.LONG, "t", Regime.STRONG_BULL, 100, 98, 102, 104, 90, 2.0, 1.5)
GOOD = {"schema_version": "1.0", "action": "BUY", "confidence": 70, "regime": "A", "setup": "t", "entry": 100,
        "stop_loss": 98, "take_profit_1": 102, "take_profit_2": 104, "risk_reward": 2, "position_size": 10,
        "signal_score": 80, "size_multiplier": 1.0, "veto": False, "invalidation": "x", "reason": "y", "risk_flags": []}


def test_schema_accepts_good_and_rejects_bad():
    assert validate_verdict(GOOD).action == "BUY"
    for bad in ({**GOOD, "action": "MAYBE"}, {**GOOD, "size_multiplier": 3}, {**GOOD, "extra": 1}):
        with pytest.raises(ValueError):
            validate_verdict(bad)
    with pytest.raises(ValueError):
        validate_verdict({k: v for k, v in GOOD.items() if k != "veto"})


def test_claude_reviewer_fails_closed_on_any_error():
    def boom(*_): raise TimeoutError()
    assert ClaudeReviewer("m", client_fn=boom).review({}, PLAN).veto
    assert ClaudeReviewer("m", client_fn=lambda *_: "not json").review({}, PLAN).veto
    assert ClaudeReviewer("m", client_fn=lambda *_: json.dumps({**GOOD, "extra": 1})).review({}, PLAN).veto


def test_claude_reviewer_accepts_fenced_json_and_vetoes_side_disagreement():
    ok = ClaudeReviewer("m", client_fn=lambda *_: "```json\n" + json.dumps(GOOD) + "\n```").review({}, PLAN)
    assert not ok.veto
    sell = {**GOOD, "action": "SELL"}
    assert ClaudeReviewer("m", client_fn=lambda *_: json.dumps(sell)).review({}, PLAN).veto


def test_shadow_mode_never_blocks_but_logs():
    inner = ClaudeReviewer("m", client_fn=lambda *_: "garbage")
    sh = ShadowReviewer(inner)
    assert not sh.review({"ts": 1}, PLAN).veto
    assert len(sh.log) == 1 and sh.log[0]["shadow"].veto
    assert not NullReviewer().review({}, PLAN).veto
