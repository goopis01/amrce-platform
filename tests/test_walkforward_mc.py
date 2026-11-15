import numpy as np, pandas as pd
from amrce.backtest.walkforward import make_windows
from amrce.backtest.montecarlo import block_bootstrap


def test_windows_are_ordered_and_embargoed():
    ws = make_windows(pd.Timestamp("2015-01-01"), pd.Timestamp("2025-01-01"))
    assert len(ws) >= 10
    for w in ws:
        assert w.train[1] < w.val[0] < w.val[1] < w.test[0] < w.test[1]
        assert (w.val[0] - w.train[1]).days == 5 and (w.test[0] - w.val[1]).days == 5
    assert ws[1].test[0] > ws[0].test[0]
    assert ws[1].test[0] >= ws[0].test[1] - pd.Timedelta(days=11)     # test windows tile forward


def test_montecarlo_runs_and_orders_percentiles():
    r = np.r_[np.full(30, 2.0), np.full(70, -1.0)] * 1.0
    out = block_bootstrap(r, 0.005, n_paths=500)
    assert out["terminal_p5"] <= out["terminal_median"] <= out["terminal_p95"]
    assert out["maxdd_p95"] <= 0
