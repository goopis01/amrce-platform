"""Extra diagnostics: regime occupancy and simple trend-following benchmarks (hold-in-uptrend / hold-above-200d)."""
from __future__ import annotations
import pandas as pd
from ..regime.rules import RuleRegimeClassifier
from .metrics import summarize
from .report import baseline_pipeline


def regime_occupancy(feats: pd.DataFrame, cfg, start_i: int) -> pd.Series:
    clf, out = RuleRegimeClassifier(cfg), []
    for i in range(len(feats)):
        r = clf.classify(feats.iloc[: i + 1])
        if i >= start_i:
            out.append(r.value)
    return pd.Series(out).value_counts(normalize=True).sort_index()


def _hold_when(sig: pd.Series, bars: pd.DataFrame, cfg, start) -> tuple[dict, float]:
    pos = sig.shift(1).astype(float).fillna(0.0)                  # decided at the close, held next bar
    cost = (cfg.costs.commission_bps + cfg.costs.slippage_bps) / 1e4
    ret = pos * bars["close"].pct_change().fillna(0) - pos.diff().abs().fillna(0) * cost
    ret = ret[ret.index >= start]
    eq = (1 + ret).cumprod() * cfg.backtest.initial_equity
    empty = pd.DataFrame(columns=["net_pnl", "r", "bars"])
    return summarize(empty, eq, cfg.backtest.bars_per_year), float(pos[pos.index >= start].mean())


def report_extras(bars: pd.DataFrame, cfg) -> str:
    feats = baseline_pipeline(cfg).prepare(bars)
    warm = cfg.backtest.warmup_bars
    start = feats.index[warm]
    occ = regime_occupancy(feats, cfg, warm)
    trend_up = feats["ma_bull"] & (feats["close"] > feats["ema_trend"])
    above200 = bars["close"] > bars["close"].rolling(200).mean()
    rows = {"Hold in uptrend (20>60>100)": _hold_when(trend_up, bars, cfg, start),
            "Hold above 200-day avg": _hold_when(above200, bars, cfg, start)}
    L = ["", "Regime occupancy (share of bars after warmup):"]
    L += [f"  {k:16}{v*100:5.1f}%" for k, v in occ.items()]
    L += ["", "Simple trend benchmarks (no entries/exits logic, costs included):",
          f"  {'':30}{'Return':>9}{'CAGR':>8}{'MaxDD':>9}{'Sharpe':>8}{'In mkt':>8}"]
    for name, (m, tim) in rows.items():
        L.append(f"  {name:30}{m['net_return']*100:8.1f}%{m['cagr']*100:7.1f}%{m['max_drawdown']*100:8.1f}%"
                 f"{m['sharpe']:8.2f}{tim*100:7.0f}%")
    return "\n".join(L)