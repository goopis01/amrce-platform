"""Shared data types. Every component talks to the others through these."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from datetime import date
from typing import Any, Dict, List, Optional
import pandas as pd


class Regime(str, Enum):
    STRONG_BULL = "A_strong_bull"
    STRONG_BEAR = "B_strong_bear"
    RANGE = "C_range"
    COMPRESSION = "D_compression"
    EXPANSION = "E_expansion"
    UNSTABLE = "F_unstable"
    NO_TRADE = "G_no_trade"


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1


@dataclass
class Candidate:
    """Raw output of a setup detector, before scoring and risk."""
    symbol: str
    side: Side
    setup: str
    entry: float
    stop: float
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    confirmations: Dict[str, bool] = field(default_factory=dict)  # family -> confirmed
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradePlan:
    symbol: str
    side: Side
    setup: str
    regime: Regime
    entry: float
    stop: float
    tp1: float
    tp2: float
    score: float
    rr_net: float
    atr: float

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)


@dataclass
class Verdict:
    """Reviewer (Claude) output after schema validation."""
    action: str = "HOLD"            # BUY | SELL | HOLD
    veto: bool = False
    size_multiplier: float = 1.0    # may only LOWER size
    confidence: float = 0.0
    reason: str = ""
    risk_flags: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fail_closed(cls, why: str) -> "Verdict":
        return cls(action="HOLD", veto=True, size_multiplier=0.0, reason=f"fail-closed: {why}")

    @classmethod
    def confirm(cls, side: Side, why: str = "no reviewer") -> "Verdict":
        return cls(action="BUY" if side is Side.LONG else "SELL", reason=why)


@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    order_type: str = "market"
    stop: Optional[float] = None
    take_profit: Optional[float] = None
    idempotency_key: str = ""
    tag: str = ""


@dataclass
class Position:
    symbol: str
    side: Side
    qty: float
    entry: float
    stop: float
    tp1: float
    tp2: float
    opened_at: Optional[pd.Timestamp] = None
    bucket: str = ""

    @property
    def risk_usd(self) -> float:
        return abs(self.entry - self.stop) * self.qty

    @property
    def notional(self) -> float:
        return self.entry * self.qty


@dataclass
class PortfolioState:
    equity: float
    equity_sod: float
    equity_sow: float
    peak_equity: float
    positions: List[Position] = field(default_factory=list)
    consecutive_losses: int = 0
    stops_today: Dict[str, int] = field(default_factory=dict)
    today: Optional[date] = None
    cooldown_until: Optional[date] = None   # no new entries through this date (inclusive)

    @property
    def drawdown(self) -> float:
        return max(0.0, 1.0 - self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0

    @property
    def daily_loss_pct(self) -> float:
        return max(0.0, 1.0 - self.equity / self.equity_sod) if self.equity_sod > 0 else 0.0

    @property
    def weekly_loss_pct(self) -> float:
        return max(0.0, 1.0 - self.equity / self.equity_sow) if self.equity_sow > 0 else 0.0

    @property
    def gross_exposure(self) -> float:
        return sum(p.notional for p in self.positions) / self.equity if self.equity > 0 else 0.0


@dataclass
class MarketContext:
    """Everything a detector/scorer may look at. `features` ends at the current closed bar."""
    symbol: str
    features: pd.DataFrame
    regime: Regime
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def ts(self) -> pd.Timestamp:
        return self.features.index[-1]

    @property
    def last(self) -> pd.Series:
        return self.features.iloc[-1]


@dataclass
class Decision:
    """Why the pipeline did or did not trade at this bar (for debugging 'no trade')."""
    stage: str                     # last stage reached
    approved: bool = False
    reason: str = ""
    regime: Optional[Regime] = None
    plan: Optional[TradePlan] = None
    qty: float = 0.0
    score: float = 0.0
    verdict: Optional[Verdict] = None
