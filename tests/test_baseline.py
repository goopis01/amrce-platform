import numpy as np
from conftest import make_bars
from amrce.backtest.report import baseline_pipeline, format_report, make_config, run_report
from amrce.signals.baseline import BaselinePullback
from amrce.signals.null import RandomEntry
from amrce.types import MarketContext, Regime, Side


def _ctx(cfg, regime, rsi2):
    bars = make_bars(n=400, seed=4)
    f = baseline_pipeline(cfg).prepare(bars)
    f.iloc[-1, f.columns.get_loc("rsi2")] = rsi2
    return MarketContext("AAA", f, regime)


def test_baseline_fires_only_in_bull_regime_on_oversold_rsi2():
    cfg = make_config("1D")
    d = BaselinePullback(cfg.signals.params["baseline"])
    hit = d.detect(_ctx(cfg, Regime.STRONG_BULL, 4.0))
    assert len(hit) == 1 and hit[0].side is Side.LONG and hit[0].stop < hit[0].entry
    assert sum(hit[0].confirmations.values()) >= 2
    assert d.detect(_ctx(cfg, Regime.STRONG_BULL, 55.0)) == []          # not oversold
    assert d.detect(_ctx(cfg, Regime.RANGE, 4.0)) == [] or Regime.RANGE not in d.regimes
    assert d.detect(_ctx(cfg, Regime.STRONG_BEAR, 4.0)) == []           # shorts disabled by default


def test_null_detector_is_reproducible_and_probabilistic():
    cfg = make_config("1D")
    ctx = _ctx(cfg, Regime.STRONG_BULL, 50)
    a = [bool(RandomEntry(0.3, 2.0, seed=1).detect(ctx)) for _ in range(1)]
    r1, r2 = RandomEntry(0.3, 2.0, 7), RandomEntry(0.3, 2.0, 7)
    s1 = [bool(r1.detect(ctx)) for _ in range(200)]
    s2 = [bool(r2.detect(ctx)) for _ in range(200)]
    assert s1 == s2 and 0.15 < np.mean(s1) < 0.45


def test_report_runs_end_to_end_on_synthetic_uptrend():
    bars = make_bars(n=1500, seed=9, freq="1D", drift=0.0008)
    rep = run_report(bars, "AAA", "1D", null_runs=5)
    assert rep["buy_hold"]["net_return"] != 0
    txt = format_report("AAA", "1D", rep)
    assert "BASELINE" in txt and "Stability" in txt
    t = rep["result"].trades
    assert (t.entry_ts > t.signal_ts).all()


def test_csv_loader_roundtrip_with_dst_offsets(tmp_path):
    from amrce.data.csv_loader import CSVData
    b = make_bars(n=600, seed=2, freq="1D")
    b.index.name = "timestamp"
    b.to_csv(tmp_path / "AAA_1D.csv")
    out = CSVData(str(tmp_path)).get_bars("AAA", "1D")
    assert len(out) == len(b) and str(out.index.tz) == "America/New_York"
    assert np.allclose(out.close.values, b.close.values)
