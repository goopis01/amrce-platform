"""Block-bootstrap of per-trade R multiples (spec section 15). Preserves loss streaks via block resampling."""
from __future__ import annotations
import numpy as np


def block_bootstrap(r, risk_pct: float, n_paths: int = 5000, block: int = 4, seed: int = 0) -> dict:
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < block * 2:
        raise ValueError("too few trades for a meaningful bootstrap")
    rng = np.random.default_rng(seed)
    term, mdd = np.empty(n_paths), np.empty(n_paths)
    for k in range(n_paths):
        idx = np.concatenate([(s + np.arange(block)) % n for s in rng.integers(0, n, size=int(np.ceil(n / block)))])[:n]
        eq = np.cumprod(1 + risk_pct * r[idx])
        term[k], mdd[k] = eq[-1], (eq / np.maximum.accumulate(eq) - 1).min()
    return {"terminal_p5": np.percentile(term, 5), "terminal_median": np.median(term),
            "terminal_p95": np.percentile(term, 95), "maxdd_p95": np.percentile(mdd, 5),
            "prob_loss": float((term < 1).mean()), "prob_dd_gt_20": float((mdd < -0.20).mean())}
