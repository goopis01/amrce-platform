"""Drift monitors. STUB (roadmap M4). Spec section 10: PSI/KS input drift, HMM likelihood drift,
CUSUM/Page-Hinkley on per-setup R. Action ladder: 1 warn (size x0.5, score +5), 2 disable setup, 3 halt."""
from __future__ import annotations


class DriftMonitor:
    def level(self) -> int:
        return 0   # 0 = healthy. Real monitors replace this in M4.
