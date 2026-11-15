"""Trend + volatility-targeted EXPOSURE across a basket (the regime idea used as a dial, not an entry trigger).

Pre-registered variants (only 3, to limit multiple-testing):
  BH         buy & hold
  SMA200     hold an asset only while close > its 200-day average
  TREND_VOL  SMA200 gate x volatility target:  weight = min(max_w, target_vol / realized_vol)
Weights are decided at the close of day t and earn the return of day t+1 (no look-ahead). Costs on turnover.
Each asset is an equal-weight sleeve; unused sleeve capital sits in cash earning 0 (conservative)."""
from __future__ import annotations
import numpy as np
import pandas as pd

MODES = ("BH", "SMA200", "TREND_VOL")


def asset_weights(close: pd.Series, mode: str, sma_n: int = 200, vol_n: int = 20,
                  target_vol: float = 0.15, max_w: float = 1.0, band: float = 0.10) -> pd.Series:
    if mode == "BH":
        return pd.Series(1.0, index=close.index)
    sma = close.rolling(sma_n).mean()
    raw = ((close > sma) & sma.notna()).astype(float)
    if mode == "TREND_VOL":
        vol = close.pct_change().rolling(vol_n).std() * np.sqrt(252)
        raw = raw * (target_vol / vol).clip(upper=max_w).fillna(0.0)
    w, prev = np.empty(len(raw)), 0.0
    for i, x in enumerate(raw.values):            # banding: ignore small weight changes (cuts turnover)
        if (x == 0) != (prev == 0) or abs(x - prev) > band:
            prev = x
        w[i] = prev
    return pd.Series(w, index=close.index)


def sleeve_returns(close: pd.Series, w: pd.Series, cost_bps: float) -> pd.Series:
    pos = w.shift(1).fillna(0.0)
    return pos * close.pct_change().fillna(0.0) - pos.diff().abs().fillna(0.0) * cost_bps / 1e4


def stats(ret: pd.Series) -> dict:
    eq = (1 + ret).cumprod()
    yrs = max(len(ret) / 252, 1e-9)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sd = ret.std()
    return {"ret": eq.iloc[-1] - 1, "cagr": cagr, "vol": sd * np.sqrt(252), "dd": dd,
            "sharpe": ret.mean() / sd * np.sqrt(252) if sd > 0 else 0.0,
            "calmar": cagr / abs(dd) if dd < 0 else 0.0}


def run_exposure(closes: pd.DataFrame, cost_bps: float = 2.0, **kw) -> dict:
    start = 200 + 20                                   # same evaluation window for every variant
    res = {}
    for m in MODES:
        W = pd.DataFrame({s: asset_weights(closes[s], m, **kw) for s in closes})
        R = pd.DataFrame({s: sleeve_returns(closes[s], W[s], cost_bps) for s in closes})
        res[m] = {"w": W.iloc[start:], "assets": R.iloc[start:], "port": R.mean(axis=1).iloc[start:]}
    return res


def format_exposure(closes: pd.DataFrame, res: dict) -> str:
    L = [f"\n=== EXPOSURE TEST: {', '.join(closes.columns)}  ({res['BH']['port'].index[0].date()} -> {res['BH']['port'].index[-1].date()}) ===",
         "Equal-weight basket, costs included, weights decided at close and applied next day.\n",
         f"{'Portfolio':12}{'Return':>10}{'CAGR':>8}{'Vol':>7}{'MaxDD':>8}{'Sharpe':>8}{'Calmar':>8}{'AvgExp':>8}"]
    st = {m: stats(res[m]["port"]) for m in MODES}
    for m in MODES:
        s = st[m]
        L.append(f"{m:12}{s['ret']*100:9.0f}%{s['cagr']*100:7.1f}%{s['vol']*100:6.1f}%{s['dd']*100:7.1f}%"
                 f"{s['sharpe']:8.2f}{s['calmar']:8.2f}{res[m]['w'].mean().mean()*100:7.0f}%")
    idx = res["BH"]["port"].index
    mid = idx[len(idx) // 2]
    halves = {}
    for name, sl in (("first half", idx < mid), ("second half", idx >= mid)):
        halves[name] = {m: stats(res[m]["port"][sl]) for m in MODES}
        L.append(f"\n{name} ({'<' if name[0]=='f' else '>='} {mid.date()})")
        for m in MODES:
            s = halves[name][m]
            L.append(f"  {m:12} CAGR {s['cagr']*100:6.1f}%  maxDD {s['dd']*100:6.1f}%  Sharpe {s['sharpe']:5.2f}  Calmar {s['calmar']:5.2f}")
    L += ["", "Per-asset Calmar (TREND_VOL vs SMA200 vs BH):"]
    wins_sma = wins_bh = 0
    for s in closes.columns:
        c = {m: stats(res[m]["assets"][s])["calmar"] for m in MODES}
        wins_sma += c["TREND_VOL"] > c["SMA200"]
        wins_bh += c["TREND_VOL"] > c["BH"]
        L.append(f"  {s:6} BH {c['BH']:5.2f}   SMA200 {c['SMA200']:5.2f}   TREND_VOL {c['TREND_VOL']:5.2f}")
    n = len(closes.columns)
    tv, sm = "TREND_VOL", "SMA200"
    checks = [
        ("Calmar > SMA200 in BOTH halves", all(halves[h][tv]["calmar"] > halves[h][sm]["calmar"] for h in halves)),
        ("Sharpe > SMA200 in BOTH halves", all(halves[h][tv]["sharpe"] > halves[h][sm]["sharpe"] for h in halves)),
        (f"Calmar > SMA200 on >= {int(np.ceil(n*2/3))} of {n} assets", wins_sma >= np.ceil(n * 2 / 3)),
    ]
    L += ["", "PRE-REGISTERED PASS/FAIL (TREND_VOL must justify its extra complexity over plain SMA200):"]
    L += [f"  [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in checks]
    L.append(f"  VERDICT: {'keep TREND_VOL' if all(ok for _, ok in checks) else 'do not adopt TREND_VOL; stay with the simpler rule'}")
    L += ["", f"Context: vs buy&hold, TREND_VOL CAGR {st[tv]['cagr']*100:.1f}% vs {st['BH']['cagr']*100:.1f}%, "
          f"maxDD {st[tv]['dd']*100:.1f}% vs {st['BH']['dd']*100:.1f}%, Calmar {st[tv]['calmar']:.2f} vs {st['BH']['calmar']:.2f} "
          f"(TREND_VOL beats BH on Calmar in {wins_bh}/{n} assets).",
          "Caveat: one 15-year sample dominated by a strong US equity bull market; results are not a forecast."]
    return "\n".join(L)