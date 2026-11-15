import math
from amrce.config import load_config
from amrce.risk.engine import RiskEngine
from amrce.risk.sizing import compute_qty
from amrce.types import PortfolioState, Position, Regime, Side, TradePlan


def plan(entry=100.0, stop=98.0, regime=Regime.STRONG_BULL, sym="AAA"):
    return TradePlan(sym, Side.LONG, "t", regime, entry, stop, 102, 104, 90, 2.0, 1.5)


def pf(eq=100_000, **kw):
    d = dict(equity=eq, equity_sod=eq, equity_sow=eq, peak_equity=eq)
    d.update(kw)
    return PortfolioState(**d)


def test_sizing_formula(cfg):
    q = compute_qty(plan(stop=96.0), 100_000, 0.0, cfg.sizing)
    assert q == math.floor(100_000 * 0.005 / 4.0)             # 125 shares -> $500 risk (0.5% of equity)
    assert q * 100 <= 100_000 * cfg.sizing.max_position_pct
    assert compute_qty(plan(), 100_000, 0.0, cfg.sizing) == 200   # 250 by risk, capped to 200 by 20% notional


def test_notional_cap_binds(cfg):
    q = compute_qty(plan(stop=99.9), 100_000, 0.0, cfg.sizing)
    assert q == 200   # 20% of 100k / $100


def test_size_never_increases_with_reviewer_multiplier(cfg):
    base = compute_qty(plan(), 100_000, 0.0, cfg.sizing)
    assert compute_qty(plan(), 100_000, 0.0, cfg.sizing, extra_mult=5.0) == base
    assert compute_qty(plan(), 100_000, 0.0, cfg.sizing, extra_mult=0.5) < base


def test_regime_and_drawdown_scale_down(cfg):
    a = compute_qty(plan(), 100_000, 0.0, cfg.sizing)
    assert compute_qty(plan(regime=Regime.COMPRESSION), 100_000, 0.0, cfg.sizing) < a
    assert compute_qty(plan(), 100_000, 0.05, cfg.sizing) < a
    assert compute_qty(plan(regime=Regime.NO_TRADE), 100_000, 0.0, cfg.sizing) == 0


def test_daily_loss_is_a_hard_stop_and_trips_killswitch(cfg):
    r = RiskEngine(cfg)
    d = r.approve(plan(), 100, pf(eq=97_900, equity_sod=100_000))
    assert not d.approved and "daily loss" in d.reason
    assert r.kill.active()
    assert not r.approve(plan(), 100, pf()).approved           # stays off afterwards


def test_limits(cfg):
    r = RiskEngine(cfg)
    assert r.approve(plan(), 100, pf()).approved
    assert not r.approve(plan(), 0, pf()).approved
    assert not RiskEngine(cfg).approve(plan(), 100, pf(consecutive_losses=3)).approved
    assert not RiskEngine(cfg).approve(plan(), 100, pf(stops_today={"AAA": 2})).approved
    assert not RiskEngine(cfg).approve(plan(), 100, pf(eq=89_000, peak_equity=100_000, equity_sod=89_000, equity_sow=89_000)).approved


def test_correlation_bucket_cap(cfg):
    bucket = lambda s: "semis" if s in ("NVDA", "AMD") else s
    r = RiskEngine(cfg, bucket_of=bucket)
    held = Position("NVDA", Side.LONG, 250, 100, 98, 102, 104)          # $500 risk already
    ok = r.approve(plan(sym="MSFT"), 250, pf(positions=[held]))
    assert ok.approved                                                    # different bucket
    bad = r.approve(plan(sym="AMD"), 250, pf(positions=[held]))
    assert not bad.approved and "bucket" in bad.reason                    # $1000 > 1.5 x $500 cap


def test_cooldown_blocks_then_expires(cfg):
    import datetime as dt
    r = RiskEngine(cfg)
    day = dt.date(2024, 3, 4)
    blocked = r.approve(plan(), 100, pf(today=day, cooldown_until=day + dt.timedelta(days=1)))
    assert not blocked.approved and "cooldown" in blocked.reason
    assert r.approve(plan(), 100, pf(today=day + dt.timedelta(days=2), cooldown_until=day + dt.timedelta(days=1))).approved
