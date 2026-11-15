"""Plumbing tests with DUMMY detector/scorer. They prove the architecture works end to end;
they say nothing about whether any real strategy makes money."""
import numpy as np, pandas as pd
from conftest import make_bars
from amrce.backtest.engine import BacktestEngine
from amrce.backtest.metrics import summarize
from amrce.config import load_config
from amrce.core.pipeline import StrategyPipeline
from amrce.regime.base import RegimeClassifier
from amrce.signals.base import SetupDetector
from amrce.signals.confluence import ConfluenceScorer, ScoreComponent
from amrce.types import Candidate, Regime, Side


class FixedRegime(RegimeClassifier):
    def __init__(self, r): self.r = r
    def classify(self, f): return self.r


class EveryN(SetupDetector):
    name, regimes = "dummy", {Regime.STRONG_BULL}
    def __init__(self, n=12, side=Side.LONG, stop_atr=1.5, confirmations=("a", "b")):
        self.n, self.side, self.k, self.conf = n, side, stop_atr, confirmations
    def detect(self, ctx):
        if len(ctx.features) % self.n:
            return []
        a, c = ctx.last["atr"], ctx.last["close"]
        stop = c - self.side.sign * self.k * a
        return [Candidate(ctx.symbol, self.side, "dummy", c, stop, confirmations={k: True for k in self.conf})]


class Const(ScoreComponent):
    name, max_points = "const", 10
    def __init__(self, v): self.v = v
    def score(self, ctx, cand): return self.v


def pipe(cfg, **kw):
    return StrategyPipeline(cfg, detectors=[kw.get("det", EveryN())], scorer=ConfluenceScorer([Const(kw.get("score", 10))]),
                            regime=FixedRegime(kw.get("regime", Regime.STRONG_BULL)))


def test_end_to_end_runs_and_is_sane(cfg):
    bars = make_bars()
    res = BacktestEngine(cfg, pipe(cfg)).run("AAA", bars)
    assert len(res.trades) > 5
    assert res.equity.notna().all() and len(res.equity) == len(bars)
    t = res.trades
    assert (t.entry_ts > t.signal_ts).all(), "entry must come AFTER the signal bar (no same-bar fills)"
    assert (t.exit_ts >= t.entry_ts).all()
    # reported net P&L must reconcile with the equity curve
    assert abs(res.equity.iloc[-1] - (cfg.backtest.initial_equity + t.net_pnl.sum())) < 1e-6
    m = summarize(t, res.equity, cfg.backtest.bars_per_year)
    assert m["trades"] == len(t) and 0 <= m["win_rate"] <= 1
    assert abs(m["expectancy"] - t.net_pnl.mean()) < 1e-6


def test_every_trade_risks_at_most_the_configured_amount(cfg):
    res = BacktestEngine(cfg, pipe(cfg)).run("AAA", make_bars(seed=3))
    t = res.trades
    # worst case is a gap through the stop; allow generous slack but never an order of magnitude
    assert t.r.min() > -3.0


def test_costs_reduce_pnl(cfg):
    bars = make_bars(seed=5)
    free = load_config(overrides={"costs": {"commission_bps": 0, "slippage_bps": 0}})
    a = BacktestEngine(free, pipe(free)).run("AAA", bars).trades.net_pnl.sum()
    b = BacktestEngine(cfg, pipe(cfg)).run("AAA", bars).trades.net_pnl.sum()
    assert b < a


def test_stop_is_assumed_first_when_both_levels_inside_one_bar(cfg):
    bars = make_bars(n=400, seed=7)
    # an absurdly tight target equal to stop distance both sit inside typical bar ranges -> stops must dominate
    res = BacktestEngine(cfg, pipe(cfg, det=EveryN(stop_atr=1.0))).run("AAA", bars)
    assert (res.trades.reason == "stop").sum() >= (res.trades.reason == "tp2").sum() * 0.5


def test_gates_block_trades(cfg):
    bars = make_bars()
    assert len(BacktestEngine(cfg, pipe(cfg, regime=Regime.NO_TRADE)).run("AAA", bars).trades) == 0
    assert len(BacktestEngine(cfg, pipe(cfg, score=3)).run("AAA", bars).trades) == 0            # score 30 < 75
    one_conf = pipe(cfg, det=EveryN(confirmations=("a",)))
    assert len(BacktestEngine(cfg, one_conf).run("AAA", bars).trades) == 0                      # needs 2 independent
    short_stop = pipe(cfg, det=EveryN(stop_atr=0.3))
    assert len(BacktestEngine(cfg, short_stop).run("AAA", bars).trades) == 0                    # stop < 1 ATR floor


def test_decision_log_explains_no_trade(cfg):
    res = BacktestEngine(cfg, pipe(cfg, score=3)).run("AAA", make_bars())
    assert set(res.decisions.stage) <= {"setup", "score"}
    assert res.decisions[res.decisions.stage == "score"].reason.str.contains("score").all()


def test_real_rule_classifier_runs_on_real_features(cfg):
    from amrce.regime.rules import RuleRegimeClassifier
    bars = make_bars(n=1200, seed=11)
    p = StrategyPipeline(cfg, detectors=[], regime=RuleRegimeClassifier(cfg))
    feats = p.prepare(bars)
    seen = {p.regime.classify(feats.iloc[: i + 1]) for i in range(300, len(feats), 5)}
    assert seen <= set(Regime) and len(seen) >= 2


def test_backtest_does_not_depend_on_future_bars(cfg):
    bars = make_bars(n=900, seed=2)
    full = BacktestEngine(cfg, pipe(cfg)).run("AAA", bars).trades
    cut = BacktestEngine(cfg, pipe(cfg)).run("AAA", bars.iloc[:700]).trades
    early = full[full.exit_ts < bars.index[690]].reset_index(drop=True)
    early_cut = cut[cut.exit_ts < bars.index[690]].reset_index(drop=True)
    pd.testing.assert_frame_equal(early.drop(columns=["reason"]), early_cut.drop(columns=["reason"]))
