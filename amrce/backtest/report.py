"""One-call benchmark report: strategy vs buy&hold, first vs second half, and a random-entry null test."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
from ..backtest.engine import BacktestEngine
from ..backtest.metrics import summarize
from ..config import Config, load_config
from ..core.pipeline import StrategyPipeline
from ..signals.baseline import BaselinePullback, MomentumScore, TrendAlignmentScore
from ..signals.confluence import ConfluenceScorer
from ..signals.null import RandomEntry

TF_PRESETS = {  # bars_per_year, time_stop_bars, warmup_bars
    "1D": (252, 10, 250), "1h": (1638, 40, 300), "15m": (6552, 130, 600),
}


def make_config(tf: str, shorts: bool = False, extra: Optional[dict] = None) -> Config:
    bpy, ts, wu = TF_PRESETS[tf]
    over = {"backtest": {"bars_per_year": bpy, "time_stop_bars": ts, "warmup_bars": wu},
            "signals": {"score_min": 0, "params": {"baseline": {"shorts": shorts}}}}
    if extra:
        from ..config import deep_merge
        over = deep_merge(over, extra)
    return load_config(overrides=over)


def baseline_pipeline(cfg: Config) -> StrategyPipeline:
    return StrategyPipeline(cfg, detectors=[BaselinePullback(cfg.signals.params.get("baseline", {}))],
                            scorer=ConfluenceScorer([TrendAlignmentScore(), MomentumScore()]))


def _slice_metrics(trades, equity, bpy, start, end):
    eq = equity[(equity.index >= start) & (equity.index < end)]
    tr = trades[(trades.entry_ts >= start) & (trades.entry_ts < end)] if len(trades) else trades
    return summarize(tr, eq / eq.iloc[0] * 100_000, bpy)


def run_report(bars: pd.DataFrame, symbol: str, tf: str, null_runs: int = 100, shorts: bool = False) -> Dict:
    cfg = make_config(tf, shorts)
    pipe = baseline_pipeline(cfg)
    feats = pipe.prepare(bars)
    res = BacktestEngine(cfg, pipe).run(symbol, bars, feats=feats)
    bpy, warm = cfg.backtest.bars_per_year, cfg.backtest.warmup_bars
    start = feats.index[warm]
    eq = res.equity[res.equity.index >= start]

    bh_eq = bars["close"][bars.index >= start]
    bh_eq = bh_eq / bh_eq.iloc[0] * cfg.backtest.initial_equity
    empty = pd.DataFrame(columns=res.trades.columns)
    out = {"cfg": cfg, "result": res,
           "strategy": summarize(res.trades, eq, bpy), "buy_hold": summarize(empty, bh_eq, bpy)}
    out["time_in_market"] = float((res.trades.bars.sum()) / max(len(eq), 1)) if len(res.trades) else 0.0
    mid = eq.index[len(eq) // 2]
    out["first_half"] = _slice_metrics(res.trades, res.equity, bpy, start, mid)
    out["second_half"] = _slice_metrics(res.trades, res.equity, bpy, mid, eq.index[-1] + pd.Timedelta(days=1))
    out["split_date"] = mid

    if null_runs and len(res.trades) >= 5:
        # same regime gate + exits + costs, random timing; entry probability set to match baseline frequency
        bull_bars = (res.decisions["regime"] == "A_strong_bull").sum() if len(res.decisions) else 0
        p = min(1.0, len(res.trades) / max(bull_bars, 1) * 1.5)
        net, er = [], []
        for k in range(null_runs):
            det = RandomEntry(p, cfg.signals.params["baseline"]["stop_atr"], seed=k)
            pn = StrategyPipeline(cfg, detectors=[det], scorer=ConfluenceScorer([TrendAlignmentScore(), MomentumScore()]))
            r = BacktestEngine(cfg, pn).run(symbol, bars, feats=feats)
            net.append(r.equity.iloc[-1] / r.equity.iloc[0] - 1)
            er.append(r.trades.r.mean() if len(r.trades) else np.nan)
        net, er = np.array(net), np.array(er)
        out["null"] = {
            "runs": null_runs, "median_net_return": float(np.nanmedian(net)),
            "p95_net_return": float(np.nanpercentile(net, 95)),
            "pctile_return": float((net < out["strategy"]["net_return"]).mean() * 100),
            "median_exp_r": float(np.nanmedian(er)), "p95_exp_r": float(np.nanpercentile(er, 95)),
            "pctile_exp_r": float((er[~np.isnan(er)] < out["strategy"].get("expectancy_r", -9)).mean() * 100),
        }
    return out


def format_report(symbol: str, tf: str, rep: Dict) -> str:
    f = lambda d, k, pct=False, dec=2: ("n/a" if k not in d else (f"{d[k]*100:.1f}%" if pct else f"{d[k]:.{dec}f}"))
    s, b = rep["strategy"], rep["buy_hold"]
    L = [f"\n=== BASELINE (trend + RSI2 pullback) on {symbol} {tf} ===", ""]
    L.append(f"{'':22}{'Strategy':>12}{'Buy&Hold':>12}")
    for lab, k, pct in [("Net return", "net_return", 1), ("CAGR", "cagr", 1), ("Max drawdown", "max_drawdown", 1),
                        ("Sharpe", "sharpe", 0), ("Sortino", "sortino", 0), ("Calmar", "calmar", 0)]:
        L.append(f"{lab:22}{f(s,k,pct):>12}{f(b,k,pct):>12}")
    L += ["", f"Trades {s['trades']}   win rate {f(s,'win_rate',1)}   profit factor {f(s,'profit_factor')}   "
          f"expectancy {f(s,'expectancy_r')}R   avg bars held {f(s,'avg_bars_held',dec=1)}",
          f"Time in market ~{rep['time_in_market']*100:.0f}%   max consecutive losses {s.get('max_consec_losses','n/a')}",
          "", f"Stability (split at {str(rep['split_date'])[:10]}; baseline has no tuned parameters, so both halves are out-of-sample):"]
    for lab, d in (("first half", rep["first_half"]), ("second half", rep["second_half"])):
        L.append(f"  {lab:12} trades {d['trades']:>3}  return {f(d,'net_return',1):>7}  Sharpe {f(d,'sharpe'):>6}  "
                 f"maxDD {f(d,'max_drawdown',1):>7}  expectancy {f(d,'expectancy_r')}R")
    if "null" in rep:
        n = rep["null"]
        L += ["", f"Random-entry null test ({n['runs']} runs; same regime gate, stops, exits, costs; random timing):",
              f"  null median return {n['median_net_return']*100:.1f}%   95th pct {n['p95_net_return']*100:.1f}%   strategy return beats {n['pctile_return']:.0f}% of null runs",
              f"  null median expectancy {n['median_exp_r']:.2f}R  95th pct {n['p95_exp_r']:.2f}R   strategy expectancy beats {n['pctile_exp_r']:.0f}% of null runs",
              "  (a real timing edge should beat >=95% of null runs)"]
    return "\n".join(L)
