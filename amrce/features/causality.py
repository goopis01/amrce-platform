"""Look-ahead guard: features computed on a truncated history must equal the full-history values."""
from __future__ import annotations
import numpy as np
import pandas as pd


def assert_causal(compute_fn, bars: pd.DataFrame, cut: int, atol: float = 1e-9) -> None:
    full = compute_fn(bars).iloc[:cut]
    part = compute_fn(bars.iloc[:cut])
    for col in full.columns:
        a, b = full[col], part[col]
        if a.dtype == bool or b.dtype == bool:
            ok = (a.astype(float).fillna(-1).values == b.astype(float).fillna(-1).values).all()
        else:
            ok = np.allclose(a.values.astype(float), b.values.astype(float), atol=atol, equal_nan=True)
        if not ok:
            raise AssertionError(f"LOOK-AHEAD in feature '{col}'")
