"""Typed config loaded from YAML. One file controls every threshold."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional
import copy
import yaml


def _build(cls, d: Optional[dict]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (d or {}).items() if k in names})


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class FeaturesCfg:
    enabled: list = field(default_factory=lambda: ["trend", "volatility"])
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeCfg:
    hysteresis_bars: int = 2
    trend_slope_min: float = 0.10
    trend_sep_min: float = 0.5
    range_slope_max: float = 0.05
    range_crossings_min: int = 3
    squeeze_pct_max: float = 20
    expansion_from_pct: float = 25
    expansion_to_pct: float = 50
    atr_ratio_squeeze: float = 0.85
    atr_ratio_expansion: float = 1.1
    unstable_atr_pct: float = 95
    unstable_range_atr: float = 4.0
    allow_unstable: bool = False


@dataclass
class SignalsCfg:
    enabled_setups: list = field(default_factory=list)
    score_min: float = 75
    min_rr: float = 1.5
    min_confirmations: int = 2
    stop_atr_min: float = 1.0
    stop_atr_max: float = 3.0
    tp1_r: float = 1.0
    tp2_r: float = 2.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SizingCfg:
    base_risk: float = 0.005
    max_position_pct: float = 0.20
    dd_halt: float = 0.10
    min_dd_mult: float = 0.25
    regime_mult: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskCfg:
    max_daily_loss: float = 0.02
    max_weekly_loss: float = 0.04
    max_drawdown: float = 0.10
    max_positions: int = 3
    bucket_risk_cap_mult: float = 1.5
    loss_streak_cooldown: int = 3
    max_stops_per_symbol_day: int = 2
    gross_exposure_max: float = 1.0


@dataclass
class CostsCfg:
    commission_bps: float = 1.0
    slippage_bps: float = 1.0


@dataclass
class AICfg:
    mode: str = "off"
    model: str = "claude-sonnet-5-5"
    timeout_s: float = 10


@dataclass
class BacktestCfg:
    initial_equity: float = 100_000
    warmup_bars: int = 300
    tp1_fraction: float = 0.0
    move_stop_to_be_after_tp1: bool = False
    time_stop_bars: int = 40
    bars_per_year: int = 1638


@dataclass
class Config:
    market: str = "us_equities"
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    regime: RegimeCfg = field(default_factory=RegimeCfg)
    signals: SignalsCfg = field(default_factory=SignalsCfg)
    sizing: SizingCfg = field(default_factory=SizingCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    costs: CostsCfg = field(default_factory=CostsCfg)
    ai: AICfg = field(default_factory=AICfg)
    backtest: BacktestCfg = field(default_factory=BacktestCfg)
    raw: Dict[str, Any] = field(default_factory=dict)


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    path = Path(path) if path else Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    raw = yaml.safe_load(path.read_text())
    raw = deep_merge(raw, overrides or {})
    return Config(
        market=raw.get("market", "us_equities"),
        features=_build(FeaturesCfg, raw.get("features")),
        regime=_build(RegimeCfg, raw.get("regime")),
        signals=_build(SignalsCfg, raw.get("signals")),
        sizing=_build(SizingCfg, raw.get("sizing")),
        risk=_build(RiskCfg, raw.get("risk")),
        costs=_build(CostsCfg, raw.get("costs")),
        ai=_build(AICfg, raw.get("ai")),
        backtest=_build(BacktestCfg, raw.get("backtest")),
        raw=raw,
    )
