"""Live/paper runner. STUB (roadmap M6). Planned loop (spec section 18):
  on each closed bar: fetch bars -> validate -> pipeline.evaluate -> ExecutionGuard.submit -> log.
Phases: P0 replay -> P1 shadow (no orders) -> P2 paper orders -> P3 review -> P4 tiny live."""
from __future__ import annotations


class LiveRunner:
    def run(self):
        raise NotImplementedError("M6: build after backtest validation passes")
