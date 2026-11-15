"""Rolling train/validation/test windows with an embargo gap (spec section 14)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

Span = Tuple[pd.Timestamp, pd.Timestamp]   # [start, end) half-open


@dataclass
class Window:
    train: Span
    val: Span
    test: Span


def make_windows(start: pd.Timestamp, end: pd.Timestamp, train_m: int = 36, val_m: int = 6,
                 test_m: int = 6, embargo_days: int = 5, step_m: int = None) -> List[Window]:
    step_m = step_m or test_m
    gap = pd.Timedelta(days=embargo_days)
    out, t0 = [], start
    while True:
        tr_end = t0 + pd.DateOffset(months=train_m)
        va0, va_end = tr_end + gap, tr_end + gap + pd.DateOffset(months=val_m)
        te0, te_end = va_end + gap, va_end + gap + pd.DateOffset(months=test_m)
        if te_end > end:
            break
        out.append(Window((t0, tr_end), (va0, va_end), (te0, te_end)))
        t0 = t0 + pd.DateOffset(months=step_m)
    return out
