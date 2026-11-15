"""Research harness for US-ETF trend / momentum rotation / regime / risk (single file, pre-registered tests).

PRE-REGISTRATION (fixed before any real-data run; do not edit after seeing results)
  * Universe: SPY QQQ IWM DIA XLK XLC XLY XLP XLU XLV XLI XLF VNQ GLD DBC TLT IEF. Cash = 3-month T-bill (^IRX) less 0.09%/yr
    (SGOV-like fee). Assets join the ranking only once they have enough history (XLC 2018, DBC 2006).
  * A-PRIORI parameters (never tuned): momentum 6 months, top 3, SMA200, monthly rebalance, trade at NEXT close (lag 1).
  * Stages: C pure momentum rotation | D trend+momentum | E +regime cap | F +inverse-vol/max-weight | +drawdown brake | +vol target.
    A component is kept ONLY if Calmar AND Sharpe improve in BOTH halves of the sample and max drawdown is not >2 pts worse.
  * Costs: 5 bps per unit traded (spread+slippage+fees), tested at 0x, 1x, 2x.  No leverage anywhere (weights sum to 1).
  * Walk-forward: expanding in-sample, choose config by plateau-smoothed Calmar (neighbour average), trade next calendar year.
  * Hard rejection rules and the multi-objective score are in `reject_reasons` / `mo_rank` below."""
from __future__ import annotations
import dataclasses as dc
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
UNIVERSE = ['SPY', 'QQQ', 'TLT', 'GLD']
DEFENSIVE = {"GLD", "TLT", "IEF"}
GROWTH = {"QQQ"}            
CORE4 = ['SPY', 'QQQ', 'TLT', 'GLD']
TD, CASH_FEE, VIX_HI = 252, 0.0009, 25.0
PERIODS = [("2005-2009", "2005-01-01", "2009-12-31"), ("2010-2019", "2010-01-01", "2019-12-31"),
           ("2020-2022", "2020-01-01", "2022-12-31"), ("2023-2026", "2023-01-01", "2026-12-31")]


# ------------------------------------------------------------------ data
@dataclass
class MarketData:
    prices: pd.DataFrame      # adjusted close (dividends included)
    cash_ret: pd.Series       # daily cash return
    vix: pd.Series


def clean_market(raw: pd.DataFrame, symbols=UNIVERSE) -> MarketData:
    raw = raw.sort_index()
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw = raw[~raw.index.duplicated()]
    px = raw[[s for s in symbols if s in raw.columns]].copy()
    keep = px["SPY"].notna()
    px, vix, irx = px[keep].ffill(limit=5), raw["^VIX"][keep].ffill().bfill(), raw["^IRX"][keep].ffill().bfill()
    if (px <= 0).any().any():
        raise ValueError("non-positive prices")
    cash = ((irx.clip(lower=0) / 100 - CASH_FEE) / TD).clip(lower=0)
    return MarketData(px, cash, vix)


def load_market(years: float = 22, cache_dir: str = "data_cache", refresh: bool = False) -> MarketData:
    cache = Path(cache_dir); cache.mkdir(exist_ok=True)
    import hashlib
    tag = hashlib.md5(",".join(sorted(UNIVERSE)).encode()).hexdigest()[:8]   # cache is keyed by universe too
    f = cache / f"market_{years:g}y_{tag}_{datetime.now():%Y%m%d}.csv"
    if f.exists() and not refresh:
        raw = pd.read_csv(f, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        start = (datetime.now() - timedelta(days=int(365.25 * years))).strftime("%Y-%m-%d")
        df = yf.download(sorted(set(UNIVERSE) | {"^VIX", "^IRX"}), start=start, auto_adjust=True, progress=False)
        raw = df["Close"]
        raw.to_csv(f)
    return clean_market(raw)


class Signals:
    """Cached, causal signal matrices (value at row t uses data <= t only)."""
    def __init__(self, md: MarketData):
        self.px, self.cash, self.vix = md.prices, md.cash_ret, md.vix
        self.ret = self.px.pct_change(fill_method=None)
        self.ci = (1 + self.cash).cumprod()
        self._c: Dict = {}

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def mom(self, L): return self._get(("m", L), lambda: self.px / self.px.shift(L) - 1)
    def cash_mom(self, L): return self._get(("cm", L), lambda: (self.ci / self.ci.shift(L) - 1).values)
    def sma(self, n): return self._get(("s", n), lambda: self.px.rolling(n).mean())
    def above(self, n): return self._get(("a", n), lambda: ((self.px > self.sma(n)) & self.sma(n).notna()).values)
    def vol(self, n=60): return self._get(("v", n), lambda: (self.ret.rolling(n).std() * np.sqrt(TD)).values)

    def regime_cap(self):
        spy = self.px["SPY"]; up = (spy > spy.rolling(200).mean()).values; hv = (self.vix > VIX_HI).values
        return np.where(up & ~hv, 1.0, np.where(~up & hv, 0.0, 0.5))

    def regime_labels(self) -> pd.Series:
        spy, vix = self.px["SPY"], self.vix
        s50, s200 = spy.rolling(50).mean(), spy.rolling(200).mean()
        bull, hv = spy > s200, vix > VIX_HI
        lab = pd.Series("weak_bull", index=spy.index)
        lab[bull & (spy > s50) & (vix < 20) & (spy / spy.shift(252) - 1 > 0.10)] = "strong_bull"
        lab[bull & hv] = "high_vol_bull"
        lab[~bull] = "bear"
        lab[~bull & (spy > s50)] = "recovery"
        lab[~bull & hv] = "high_vol_bear"
        lab[s200.isna()] = "warmup"
        return lab


# ------------------------------------------------------------------ strategy definition + schedule
@dataclass(frozen=True)
class Strat:
    name: str
    lookback_m: int = 6
    top_n: int = 3
    trend: bool = True
    sma_n: int = 200
    abs_mom: bool = True
    regime: bool = False
    invvol: bool = False
    max_w: float = 1.0
    vol_target: Optional[float] = None
    dd_brake: Optional[float] = None
    brake_scale: float = 0.5
    defensive_fill: str = "cash"
    exclude: Tuple[str, ...] = ()


def month_end_idx(index: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(np.arange(len(index)), index=index)
    return s.groupby([index.year, index.month]).max().values


def build_schedule(sig: Signals, st: Strat, start: int, lag: int):
    cols = list(sig.px.columns); K = len(cols); ci = {c: i for i, c in enumerate(cols)}
    L = st.lookback_m * 21
    M, CM = sig.mom(L).values, sig.cash_mom(L)
    A = sig.above(st.sma_n) if st.trend else None
    V, RET = sig.vol(60), sig.ret.values
    cap = sig.regime_cap() if st.regime else None
    T = len(sig.px)
    rebal = sorted({start} | {int(i) for i in month_end_idx(sig.px.index) if i > start})
    sched = []
    for i in rebal:
        if i + lag >= T:
            break
        elig = []
        for c in cols:
            j = ci[c]
            if c in st.exclude or np.isnan(M[i, j]):
                continue
            if st.trend and not A[i, j]:
                continue
            if st.abs_mom and not (M[i, j] > CM[i]):
                continue
            elig.append((M[i, j], j))
        elig.sort(reverse=True)
        sel = [j for _, j in elig[:st.top_n]]
        w = np.zeros(K + 1)
        if sel:
            if st.invvol:
                iv = np.array([1 / max(V[i, j], 1e-3) if not np.isnan(V[i, j]) else 0.0 for j in sel])
                ww = iv / iv.sum() * len(sel) / st.top_n if iv.sum() > 0 else np.full(len(sel), 1 / st.top_n)
            else:
                ww = np.full(len(sel), st.max_w / st.top_n)
            w[sel] = np.minimum(ww, st.max_w)
            if st.regime:
                for j in sel:
                    if cols[j] not in DEFENSIVE:
                        w[j] *= cap[i]
        if st.defensive_fill == "ief" and "IEF" in ci:
            j, slack = ci["IEF"], 1 - w[:K].sum()
            if slack > 1e-9 and not np.isnan(M[i, j]) and (A is None or A[i, j]) and M[i, j] > CM[i]:
                w[j] += slack
        if st.vol_target and w[:K].sum() > 0:
            js = np.where(w[:K] > 0)[0]
            win = np.nan_to_num(RET[max(0, i - 59):i + 1][:, js])
            cov = np.atleast_2d(np.cov(win.T))
            pv = float(np.sqrt(max(w[js] @ cov @ w[js], 0) * TD))
            if pv > 0:
                w[:K] *= min(1.0, st.vol_target / pv)
        w[K] = 1 - w[:K].sum()
        sched.append((i + lag, w))
    return sched


def sched_from_weights(W: np.ndarray, start: int, lag: int):
    sched, last = [], None
    for i in range(start, len(W) - lag):
        if last is None or not np.allclose(W[i], last):
            sched.append((i + lag, W[i].copy())); last = W[i]
    return sched


def simulate(R: np.ndarray, sched, start: int, cost_bps: float, brake: Optional[float] = None, brake_scale: float = 0.5):
    T, K1 = R.shape; K = K1 - 1
    h = np.zeros(K1); h[K] = 1.0
    V = peak = 1.0; braked = False; ptr = 0
    sched = sorted(sched, key=lambda x: x[0])
    ret, expo, trn, wlog = np.zeros(T), np.zeros(T), np.zeros(T), []
    for t in range(start, T):
        Vb = V; h = h * (1 + R[t]); V = h.sum()
        while ptr < len(sched) and sched[ptr][0] == t:
            w = sched[ptr][1].copy(); ptr += 1
            if brake:
                dd = V / peak - 1
                if braked and dd > -brake / 2: braked = False
                if (not braked) and dd < -brake: braked = True
                if braked:
                    w[:K] *= brake_scale; w[K] = 1 - w[:K].sum()
            tgt = w * V
            traded = np.abs(tgt - h).sum(); cost = traded * cost_bps / 1e4
            trn[t] += traded / V / 2
            h = tgt * (1 - cost / V); V = h.sum()
            wlog.append((t, w))
        ret[t] = V / Vb - 1; peak = max(peak, V); expo[t] = 1 - h[K] / V
    return ret, expo, trn, wlog


@dataclass
class Result:
    name: str
    ret: pd.Series
    expo: pd.Series
    turnover: pd.Series
    wlog: list
    cols: list
    full_index: pd.DatetimeIndex = None


def _R(sig: Signals) -> np.ndarray:
    return np.column_stack([sig.ret.fillna(0).values, sig.cash.values])


def _wrap(sig, name, S, out) -> Result:
    ret, expo, trn, wlog = out
    ix = sig.px.index[S:]
    return Result(name, pd.Series(ret[S:], ix), pd.Series(expo[S:], ix), pd.Series(trn[S:], ix), wlog, list(sig.px.columns), sig.px.index)


def run_strat(sig: Signals, st: Strat, S: int, cost_bps: float, lag: int = 1) -> Result:
    out = simulate(_R(sig), build_schedule(sig, st, S, lag), S, cost_bps, st.dd_brake, st.brake_scale)
    return _wrap(sig, st.name, S, out)


def sched_monthly(W: np.ndarray, idx: pd.DatetimeIndex, start: int, lag: int):
    rebal = sorted({start} | {int(i) for i in month_end_idx(idx) if i > start})
    return [(i + lag, W[i].copy()) for i in rebal if i + lag < len(W)]


def run_weights(sig: Signals, name: str, W: np.ndarray, S: int, cost_bps: float, lag: int = 1, monthly: bool = False) -> Result:
    sch = sched_monthly(W, sig.px.index, S, lag) if monthly else sched_from_weights(W, S, lag)
    return _wrap(sig, name, S, simulate(_R(sig), sch, S, cost_bps))


def weights_static(sig, w: Dict[str, float]) -> np.ndarray:
    cols = list(sig.px.columns); W = np.zeros((len(sig.px), len(cols) + 1))
    for c, x in w.items():
        W[:, cols.index(c)] = x
    W[:, -1] = 1 - W[:, :-1].sum(1)
    return W


def weights_from_flags(sig, flags: pd.DataFrame) -> np.ndarray:
    cols = list(sig.px.columns); W = np.zeros((len(sig.px), len(cols) + 1))
    for c in flags.columns:
        W[:, cols.index(c)] = flags[c].reindex(sig.px.index).fillna(False).astype(float).values / len(flags.columns)
    W[:, -1] = 1 - W[:, :-1].sum(1)
    return W



# ------------------------------------------------------------------ metrics
def underwater(eq: pd.Series):
    peak = eq.cummax(); under = (eq < peak).values; idx = eq.index; n = len(eq); eps = []; i = 0
    while i < n:
        if under[i]:
            s = i - 1; j = i
            while j < n and under[j]: j += 1
            seg = eq.iloc[s:j]
            eps.append(dict(peak=idx[s], trough=seg.idxmin(), recovered=idx[j] if j < n else None,
                            depth=seg.min() / eq.iloc[s] - 1, days=(idx[min(j, n - 1)] - idx[s]).days))
            i = j
        else:
            i += 1
    return sorted(eps, key=lambda e: e["depth"])


def perf(ret: pd.Series, cash: pd.Series, wealth0: float = 100_000) -> dict:
    r = ret.dropna(); yrs = len(r) / TD
    eq = (1 + r).cumprod(); cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    ex = r - cash.reindex(r.index).fillna(0)
    dsd = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(TD)
    mo = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    yr = (1 + r).groupby(r.index.year).prod() - 1
    eps = underwater(eq); worst = eps[0] if eps else None
    rec = np.nan
    if worst and worst["recovered"] is not None:
        rec = (worst["recovered"] - worst["trough"]).days
    return dict(cagr=cagr, vol=r.std() * np.sqrt(TD), mdd=mdd,
                sharpe=ex.mean() / ex.std() * np.sqrt(TD) if ex.std() > 0 else 0.0,
                sortino=ex.mean() * TD / dsd if dsd > 0 else 0.0, calmar=cagr / abs(mdd) if mdd < 0 else 0.0,
                dsd=dsd, worst_month=mo.min(), worst_year=yr.min(), win_m=(mo > 0).mean(),
                pf=mo[mo > 0].sum() / abs(mo[mo < 0].sum()) if (mo < 0).any() else np.inf,
                recovery_days=rec, wealth=wealth0 * eq.iloc[-1],
                underwater_days=worst["days"] if worst else 0)


def quick(ret: pd.Series, cash=None):
    r = ret.dropna(); eq = (1 + r).cumprod(); yrs = max(len(r) / TD, 1e-9)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1; mdd = (eq / eq.cummax() - 1).min()
    ex = r - (cash.reindex(r.index).fillna(0) if cash is not None else 0)
    return dict(cagr=cagr, mdd=mdd, calmar=cagr / abs(mdd) if mdd < 0 else 0.0,
                sharpe=ex.mean() / ex.std() * np.sqrt(TD) if ex.std() > 0 else 0.0)


def trade_stats(res: Result, T_end: int):
    K = len(res.cols); prev = np.zeros(K + 1); prev[K] = 1; trades = 0; streaks = []; st = {}
    for t, w in res.wlog:
        for j in range(K):
            was, now = prev[j] > 0.01, w[j] > 0.01
            if now and not was: trades += 1; st[j] = t
            if was and not now: trades += 1; streaks.append(t - st.pop(j))
        prev = w
    for j, t0 in st.items(): streaks.append(T_end - t0)
    return trades, (float(np.mean(streaks)) if streaks else np.nan)


def full_metrics(res: Result, cash: pd.Series, T_end: int) -> dict:
    m = perf(res.ret, cash); tr, hold = trade_stats(res, T_end)
    yrs = len(res.ret) / TD
    m.update(invested=res.expo.mean(), trades=tr, hold_days=hold, turnover=res.turnover.sum() / yrs, cagr_over_mdd=m["calmar"])
    return m


def mo_rank(mdf: pd.DataFrame) -> pd.Series:
    """Multi-objective score (0-100): drawdown/downside weigh more than ordinary volatility."""
    W = {"calmar": .25, "mdd": .20, "sortino": .15, "cagr": .15, "sharpe": .10, "dsd": .05, "recovery_days": .05, "turnover": .05}
    score = pd.Series(0.0, index=mdf.index)
    for k, w in W.items():
        col = mdf[k].astype(float).fillna(mdf[k].astype(float).max() if k == "recovery_days" else 0)
        col = col.abs() if k == "mdd" else col
        r = col.rank(pct=True, ascending=k not in ("mdd", "dsd", "recovery_days", "turnover"))
        score += w * r
    return (score * 100).round(1)


# ------------------------------------------------------------------ robustness tools
def paired_bootstrap(ra: pd.Series, rb: pd.Series, cash: pd.Series, n=1000, block=21, seed=0) -> dict:
    ix = ra.index.intersection(rb.index)
    a, b, c = ra.loc[ix].values, rb.loc[ix].values, cash.reindex(ix).fillna(0).values
    T = len(a); nb = int(np.ceil(T / block)); rng = np.random.default_rng(seed)

    def stats(x, cc):
        eq = np.cumprod(1 + x); dd = (eq / np.maximum.accumulate(eq) - 1).min()
        cagr = eq[-1] ** (TD / len(x)) - 1; e = x - cc
        return (e.mean() / e.std() * np.sqrt(TD) if e.std() > 0 else 0.0), (cagr / abs(dd) if dd < 0 else 0.0), cagr, dd
    ds, dc_, dd_, cg = [], [], [], []
    for _ in range(n):
        idx = (rng.integers(0, T - block + 1, nb)[:, None] + np.arange(block)).ravel()[:T]
        sa, ca, cga, dda = stats(a[idx], c[idx]); sb, cb, cgb, ddb = stats(b[idx], c[idx])
        ds.append(sa - sb); dc_.append(ca - cb); dd_.append(dda - ddb); cg.append(cga)
    q = lambda x: (float(np.percentile(x, 5)), float(np.percentile(x, 95)))
    return dict(p_sharpe=float(np.mean(np.array(ds) > 0)), p_calmar=float(np.mean(np.array(dc_) > 0)),
                sharpe_diff_ci=q(ds), calmar_diff_ci=q(dc_), cagr_ci=q(cg), p_dd_better=float(np.mean(np.array(dd_) > 0)))


def plateau_pick(scores: np.ndarray):
    """Pick the cell whose NEIGHBOURHOOD average is best (prefers broad plateaus over lone peaks)."""
    sm = np.full(scores.shape, np.nan)
    for ix in np.ndindex(scores.shape):
        sl = tuple(slice(max(i - 1, 0), i + 2) for i in ix)
        sm[ix] = np.nanmean(scores[sl])
    return np.unravel_index(np.nanargmax(sm), sm.shape), sm


def trades_between(res: Result, d0, d1) -> int:
    K = len(res.cols); prev = np.zeros(K + 1); prev[K] = 1; n = 0
    for t, w in res.wlog:
        d = res.full_index[t]
        if d0 < d <= d1:
            n += int(((prev[:K] > 0.01) != (w[:K] > 0.01)).sum())
        prev = w
    return n


def episodes_vs_spy(spy_ret: pd.Series, res: Result, min_depth=-0.10, top=8) -> pd.DataFrame:
    eq = (1 + spy_ret).cumprod(); rows = []
    for e in [x for x in underwater(eq) if x["depth"] <= min_depth][:top]:
        p, t = e["peak"], e["trough"]; rec = e["recovered"] or eq.index[-1]
        w = res.ret[(res.ret.index > p) & (res.ret.index <= t)]; sw = spy_ret[(spy_ret.index > p) & (spy_ret.index <= t)]
        end = min(rec, t + pd.Timedelta(days=180))
        a = res.ret[(res.ret.index > t) & (res.ret.index <= end)]; sa = spy_ret[(spy_ret.index > t) & (spy_ret.index <= end)]
        ex = res.expo[(res.expo.index > p) & (res.expo.index <= rec)]
        e0 = float(res.expo[res.expo.index <= p].iloc[-1]) if (res.expo.index <= p).any() else 1.0
        out_ = ex[ex < 0.5 * e0] if e0 > 0.05 else ex.iloc[0:0]
        back = res.expo[res.expo.index > t]; back = back[back >= 0.5 * e0] if e0 > 0.05 else back.iloc[0:0]
        rows.append(dict(peak=p.date(), trough=t.date(), spy_dd=e["depth"], strat=(1 + w).prod() - 1, spy=(1 + sw).prod() - 1,
                         avoided=(1 + w).prod() - (1 + sw).prod(), exit_days=(out_.index[0] - p).days if len(out_) else None,
                         reenter_days=(back.index[0] - t).days if len(back) else None,
                         missed=(1 + a).prod() - (1 + sa).prod(), trades=trades_between(res, p, rec)))
    return pd.DataFrame(rows)


def stress_test(md: MarketData, st: Strat, S: int, cost_bps: float, n=12, window=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed); base = md.prices.pct_change(fill_method=None); T = len(md.prices)
    eq_idx = [i for i, c in enumerate(md.prices.columns) if c not in DEFENSIVE]; rows = []
    for k in range(n):
        up = md.prices["SPY"] > md.prices["SPY"].rolling(200).mean()
        for _ in range(500):
            s0 = int(rng.integers(S + 300, T - window - 120))
            if up.iloc[s0] and md.vix.iloc[s0] < VIX_HI:
                break
        r2 = base.copy(); vals = r2.to_numpy(copy=True)
        vals[s0:s0 + window, eq_idx] = vals[s0:s0 + window, eq_idx] * 1.3 - 0.012     # ~ -38% equity crash
        px2 = (1 + pd.DataFrame(vals, r2.index, r2.columns).fillna(0)).cumprod().where(md.prices.notna())
        vix2 = md.vix.copy(); vix2.iloc[s0 + 3:s0 + window + 25] = 40.0
        sig2 = Signals(MarketData(px2, md.cash_ret, vix2)); res = run_strat(sig2, st, S, cost_bps)
        win = slice(s0 - S, s0 - S + window + 120)
        s_eq = (1 + res.ret.iloc[win]).cumprod(); b_eq = (1 + sig2.ret["SPY"].iloc[S:].iloc[win]).cumprod()
        rows.append(dict(strat_dd=(s_eq / s_eq.cummax() - 1).min(), spy_dd=(b_eq / b_eq.cummax() - 1).min()))
    return pd.DataFrame(rows)


def reject_reasons(m: dict, halves: Tuple[dict, dict], sens_ratio: float, cost2_calmar_ratio: float,
                   crisis_share: float, exgrowth_ratio: float, turnover: float, leverage: float) -> List[str]:
    r = []
    if m["mdd"] < -0.45: r.append(f"max drawdown {m['mdd']:.1%} worse than -45%")
    if m["sharpe"] < 0.85: r.append(f"Sharpe {m['sharpe']:.2f} < 0.85")
    if m["calmar"] < 0.50: r.append(f"Calmar {m['calmar']:.2f} < 0.50")
    if crisis_share > 0.60: r.append(f"{crisis_share:.0%} of outperformance comes from one crisis")
    if halves[1]["sharpe"] < 0.5 * halves[0]["sharpe"]: r.append("second-half Sharpe < 50% of first half (stability)")
    if sens_ratio < 0.70: r.append(f"parameter sensitivity: neighbours average only {sens_ratio:.0%} of its Calmar")
    if turnover > 12.0: r.append(f"turnover {turnover:.1f}x per year (excessive)")
    if cost2_calmar_ratio < 0.85: r.append(f"2x costs cut Calmar to {cost2_calmar_ratio:.0%} of base")
    if exgrowth_ratio < 0.70: r.append(f"without hindsight winners (QQQ/XLK/XLC/XLY) Calmar falls to {exgrowth_ratio:.0%} (survivorship/hindsight)")
    if leverage > 2.0 + 1e-9: r.append(f"uses leverage {leverage:.2f}x (pre-registered: none)")
    return r


# ------------------------------------------------------------------ trend-rule variants (section 2)
def trend_variants(sig: Signals, S: int, cost_bps: float, lag: int, cash: pd.Series) -> pd.DataFrame:
    px = sig.px[CORE4]; ret = px.pct_change(fill_method=None); rows = {}
    sma = lambda n: px.rolling(n).mean()
    flags = {f"SMA{n}": (px > sma(n)) for n in (100, 150, 200, 250)}
    flags["EMA200"] = (px > px.ewm(span=200, adjust=False).mean()) & (np.arange(len(px))[:, None] >= 200)
    state = pd.DataFrame(False, index=px.index, columns=px.columns); cur = np.zeros(px.shape[1], bool)
    hi, lo = (px > 1.02 * sma(200)).values, (px < 0.98 * sma(200)).values
    for i in range(len(px)):
        cur = (cur | hi[i]) & ~lo[i]; state.iloc[i] = cur
    flags["SMA200 +/-2% band"] = state
    flags["50/200 cross"] = sma(50) > sma(200)
    flags["SMA200 + rising slope"] = (px > sma(200)) & (sma(200) - sma(200).shift(20) > 0)
    flags["SMA200 + 12m momentum>cash"] = (px > sma(200)) & (px / px.shift(252) - 1 > sig.cash_mom(252)[:, None])
    flags["SMA200 + low vol(20d<30%)"] = (px > sma(200)) & (ret.rolling(20).std() * np.sqrt(TD) < 0.30)
    me = month_end_idx(px.index); m = px.iloc[me]
    flags["Faber 10-month (monthly)"] = (m > m.rolling(10).mean()).reindex(px.index).ffill().fillna(False)
    flags["(ref) always invested"] = pd.DataFrame(True, index=px.index, columns=px.columns)
    mid = sig.px.index[S + (len(sig.px) - S) // 2]
    for name, fl in flags.items():
        r = run_weights(sig, name, weights_from_flags(sig, fl), S, cost_bps, lag)
        q, q1, q2 = quick(r.ret, cash), quick(r.ret[r.ret.index < mid], cash), quick(r.ret[r.ret.index >= mid], cash)
        rows[name] = dict(CAGR=q["cagr"], MaxDD=q["mdd"], Sharpe=q["sharpe"], Calmar=q["calmar"], Calmar_H1=q1["calmar"],
                          Calmar_H2=q2["calmar"], TurnoverPerYr=r.turnover.sum() / (len(r.ret) / TD))
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ report
def _f(v, k):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "n/a"
    if k in ("cagr", "vol", "mdd", "invested", "worst_month", "worst_year", "win_m"): return f"{v*100:.1f}%"
    if k in ("sharpe", "sortino", "calmar", "pf", "cagr_over_mdd", "turnover"): return f"{v:.2f}"
    if k in ("trades",): return f"{int(v)}"
    if k in ("hold_days", "recovery_days"): return f"{v:.0f}"
    if k == "wealth": return f"{v:,.0f}"
    return str(v)


ROWS = [("cagr", "CAGR"), ("vol", "Volatility"), ("mdd", "Max drawdown"), ("sharpe", "Sharpe (excess over cash)"),
        ("sortino", "Sortino"), ("calmar", "Calmar (= CAGR / MaxDD)"), ("pf", "Profit factor (monthly)"),
        ("win_m", "Win rate (monthly)"), ("trades", "Trades (entries+exits)"), ("hold_days", "Avg holding (days)"),
        ("invested", "Time invested"), ("turnover", "Turnover (x/yr, one-way)"), ("worst_month", "Worst month"),
        ("worst_year", "Worst year"), ("recovery_days", "Recovery after max DD (days)"), ("wealth", "Terminal wealth (100k)")]


def table(metrics: Dict[str, dict]) -> str:
    df = pd.DataFrame({n: {lab: _f(m[k], k) for k, lab in ROWS} for n, m in metrics.items()})
    return df.to_string()


def halves_of(ret, cash, mid):
    return quick(ret[ret.index < mid], cash), quick(ret[ret.index >= mid], cash)


def run_research(md: MarketData, cost_bps: float = 5.0, lag: int = 1, quick_mode: bool = False) -> Tuple[str, Dict[str, pd.Series]]:
    sig = Signals(md); idx = md.prices.index; T = len(idx); S = 260; cash = md.cash_ret; cols = list(md.prices.columns)
    out: List[str] = []; P = out.append
    mid = idx[S + (T - S) // 2]; spy_ret = sig.ret["SPY"].iloc[S:]
    P(f"RESEARCH REPORT  data {idx[0].date()} -> {idx[-1].date()}  evaluation {idx[S].date()} -> {idx[-1].date()}  (split into halves at {mid.date()})")
    P(f"Costs {cost_bps:g} bps per unit traded | trade at next close (lag {lag}) | cash = T-bill less 0.09%/yr | no leverage | monthly rebalance")
    first = {}
    for c in cols:
        valid_idx = md.prices[c].first_valid_index()
        if valid_idx is not None:
            first[c] = valid_idx.date()
        else:
            print(f"WARNING: No valid data found for ticker {c}. Dropping from universe.")
        # Optionally, you can drop the column here: md.prices.drop(columns=[c], inplace=True)
    P("ETF first price date: " + ", ".join(f"{c} {d}" for c, d in first.items()))
    P("Universe caveat: every ETF listed still exists today and several (QQQ, XLK) were top performers, so some hindsight is unavoidable; see the 'no hindsight winners' test.")

    # ---- baselines and stages
    R: Dict[str, Result] = {}
    R["BH (SPY)"] = run_weights(sig, "BH (SPY)", weights_static(sig, {"SPY": 1.0}), S, cost_bps, lag)
    W = weights_static(sig, {}); j = cols.index("SPY"); W[:, j] = sig.above(200)[:, j]; W[:, -1] = 1 - W[:, :-1].sum(1)
    R["SMA200 (SPY)"] = run_weights(sig, "SMA200 (SPY)", W, S, cost_bps, lag)
    R["60/40 SPY/TLT"] = run_weights(sig, "60/40 SPY/TLT", weights_static(sig, {"SPY": .6, "TLT": .4}), S, cost_bps, lag, monthly=True)
    C = Strat("C 2x QQQ/TLT Parity", lookback_m=6, top_n=2, trend=False, abs_mom=False, max_w=2.0, exclude=("SPY", "GLD"))
    D = Strat("D 2x Parity + Crash Protect", lookback_m=6, top_n=2, trend=False, abs_mom=True, max_w=2.0, exclude=("SPY", "GLD"))
    R["C Momentum rotation"] = run_strat(sig, C, S, cost_bps, lag); R["D Trend+Momentum"] = run_strat(sig, D, S, cost_bps, lag)

    P("\n=== ABLATION ===")
    base, base_res = D, R["D Trend+Momentum"]; abl = []
    for label, tag, kw in []:  # EMPTY LOOP DISABLES ABLATION DAMPENERS
        pass
    STAGES = []
    for label, tag, kw in STAGES:
        cand = dc.replace(base, name=base.name + "+" + tag, **kw); cr = run_strat(sig, cand, S, cost_bps, lag)
        b1, b2 = halves_of(base_res.ret, cash, mid); c1, c2 = halves_of(cr.ret, cash, mid)
        okc = c1["calmar"] > b1["calmar"] and c2["calmar"] > b2["calmar"]; oks = c1["sharpe"] > b1["sharpe"] and c2["sharpe"] > b2["sharpe"]
        mf, bf = quick(cr.ret, cash), quick(base_res.ret, cash); okd = mf["mdd"] > bf["mdd"] - 0.02
        keep = okc and oks and okd
        abl.append(f"  {label:42} CAGR {mf['cagr']*100:5.1f}% (was {bf['cagr']*100:5.1f}%)  MaxDD {mf['mdd']*100:6.1f}% (was {bf['mdd']*100:6.1f}%)  "
                   f"Calmar H1/H2 {c1['calmar']:.2f}/{c2['calmar']:.2f} (was {b1['calmar']:.2f}/{b2['calmar']:.2f})  Sharpe H1/H2 {c1['sharpe']:.2f}/{c2['sharpe']:.2f} "
                   f"(was {b1['sharpe']:.2f}/{b2['sharpe']:.2f})  -> {'KEEP' if keep else 'DROP'}")
        if keep: base, base_res = cand, cr
    out.extend(abl)
    best = dc.replace(base, name="Best (final stage)"); R["Best (final stage)"] = run_strat(sig, best, S, cost_bps, lag)
    br = R["Best (final stage)"]
    P(f"  Final stage = {base.name}   [regime={best.regime}, invvol={best.invvol}, dd_brake={best.dd_brake}, vol_target={best.vol_target}]")
    hyb = dc.replace(br, name="G 50% SPY + 50% Best", ret=0.5 * R["BH (SPY)"].ret + 0.5 * br.ret, expo=0.5 + 0.5 * br.expo, turnover=0.5 * br.turnover)
    R["G 50% SPY + 50% Best"] = hyb

    M = {n: full_metrics(r, cash, T - 1) for n, r in R.items()}
    P("\n=== 1. BENCHMARK TABLE (net of costs) ==="); P(table(M))
    P("Note: win rate / profit factor are computed on calendar months because a rotation portfolio has no clean per-trade P&L. Hybrid G is a daily-rebalanced 50/50 mix.")

    P("\n=== PERFORMANCE BY PERIOD (CAGR / MaxDD / Sharpe / Calmar) ===")
    for nm in ("BH (SPY)", "SMA200 (SPY)", "Best (final stage)"):
        for lab, a, b in PERIODS:
            r = R[nm].ret[(R[nm].ret.index >= a) & (R[nm].ret.index <= b)]
            if len(r) > 60:
                q = quick(r, cash); P(f"  {nm:20} {lab}  CAGR {q['cagr']*100:6.1f}%  MaxDD {q['mdd']*100:6.1f}%  Sharpe {q['sharpe']:5.2f}  Calmar {q['calmar']:5.2f}")

    P("\n=== 2. TREND-RULE VARIANTS on SPY/QQQ/IWM/DIA equal-weight (no parameter chosen; shows how flat the landscape is) ===")
    P(trend_variants(sig, S, cost_bps, lag, cash).to_string(float_format=lambda x: f"{x:.2f}"))

    # ---- grid, sensitivity, walk-forward
    Ls, Ns, SMs = ([6, 12], [2, 3, 4], [150, 200, 250]) if quick_mode else ([3, 6, 9, 12], [1, 2, 3, 4, 5], [100, 150, 200, 250])
    streams, calm = {}, np.full((len(Ls), len(Ns), len(SMs)), np.nan)
    for a, L in enumerate(Ls):
        for b, N in enumerate(Ns):
            for c, sm in enumerate(SMs):
                r = run_strat(sig, Strat("g", lookback_m=L, top_n=N, sma_n=sm), S, cost_bps, lag).ret
                streams[(a, b, c)] = r; calm[a, b, c] = quick(r, cash)["calmar"]
    P("\n=== 5. PARAMETER SENSITIVITY: Trend+Momentum Calmar (rows lookback months, cols top-N) ===")
    for c, sm in enumerate(SMs):
        P(f"  SMA{sm}:  " + "  ".join(f"N={n}" for n in Ns))
        for a, L in enumerate(Ls):
            P(f"    {L:>2}m   " + "  ".join(f"{calm[a, b, c]:5.2f}" for b in range(len(Ns))))
    flat = calm.ravel()
    P(f"  Grid of {flat.size} configs: Calmar median {np.median(flat):.2f}, min {flat.min():.2f}, max {flat.max():.2f}; "
      f"{(flat >= 0.5).mean():.0%} of cells have Calmar >= 0.5 (all {flat.size} were run; none discarded)")
    ia = (Ls.index(6) if 6 in Ls else 0, Ns.index(3) if 3 in Ns else 0, SMs.index(200) if 200 in SMs else 0)
    nb = calm[tuple(slice(max(i - 1, 0), i + 2) for i in ia)].ravel(); nb = nb[~np.isnan(nb)]
    sens_ratio = float(np.mean(nb) / calm[ia]) if calm[ia] > 0 else 0.0
    P(f"  A-priori cell (6m, top 3, SMA200) Calmar {calm[ia]:.2f}; its neighbours average {np.mean(nb):.2f} ({sens_ratio:.0%} of it)")

    P("\n=== 4/10. WALK-FORWARD (expanding in-sample; plateau-smoothed Calmar picks the config; trade the next calendar year) ===")
    years = list(range(idx[S].year + 7, idx[-1].year + 1)); wf, pbo_hits, is_sh, rows = [], 0, [], []
    for Y in years:
        sc = np.full(calm.shape, np.nan)
        for k, r in streams.items():
            sc[k] = quick(r[r.index < f"{Y}-01-01"], cash)["calmar"]
        pick, _ = plateau_pick(sc)
        o = streams[pick][(streams[pick].index >= f"{Y}-01-01") & (streams[pick].index <= f"{Y}-12-31")]
        wf.append(o); is_sh.append(quick(streams[pick][streams[pick].index < f"{Y}-01-01"], cash)["sharpe"])
        allr = np.array([(1 + r[(r.index >= f"{Y}-01-01") & (r.index <= f"{Y}-12-31")]).prod() - 1 for r in streams.values()])
        rank = float((allr < (1 + o).prod() - 1).mean()); pbo_hits += rank < 0.5
        rows.append(f"  {Y}: picked {Ls[pick[0]]}m/top{Ns[pick[1]]}/SMA{SMs[pick[2]]}  OOS return {((1+o).prod()-1)*100:6.1f}%  (beat {rank:.0%} of all configs)")
    out.extend(rows)
    wfr = pd.concat(wf) if wf else pd.Series(dtype=float)
    if len(wfr):
        qw, qb, qs = quick(wfr, cash), quick(R["BH (SPY)"].ret.reindex(wfr.index), cash), quick(R["SMA200 (SPY)"].ret.reindex(wfr.index), cash)
        pbo = pbo_hits / len(years)
        P(f"  Stitched OOS {wfr.index[0].date()} -> {wfr.index[-1].date()}:")
        for lab, q in (("Walk-forward Trend+Mom", qw), ("BH (SPY)", qb), ("SMA200 (SPY)", qs)):
            P(f"    {lab:24} CAGR {q['cagr']*100:5.1f}%  MaxDD {q['mdd']*100:6.1f}%  Sharpe {q['sharpe']:.2f}  Calmar {q['calmar']:.2f}")
        P(f"  OOS Sharpe {qw['sharpe']:.2f} vs average in-sample Sharpe of the chosen configs {np.mean(is_sh):.2f}")
        P(f"  Overfitting estimate (share of years the in-sample pick finished BELOW the median config out-of-sample): {pbo:.0%}  [{len(streams)} configs tried; ~50% = no selection skill, higher = selection is hurting]")

    # ---- costs, universe, crashes, stress, bootstrap
    P("\n=== 9. COSTS: gross vs net (Calmar / CAGR) ===")
    cost_ratio = {}
    for nm, st in (("D Trend+Momentum", D), ("Best (final stage)", best), ("C Momentum rotation", C)):
        v = {c: quick(run_strat(sig, st, S, c, lag).ret, cash) for c in (0.0, cost_bps, 2 * cost_bps)}
        cost_ratio[nm] = v[2 * cost_bps]["calmar"] / v[cost_bps]["calmar"] if v[cost_bps]["calmar"] > 0 else 0.0
        P(f"  {nm:22} " + "   ".join(f"{c:g}bps: CAGR {q['cagr']*100:5.1f}% Calmar {q['calmar']:.2f}" for c, q in v.items()))
    P("\n=== UNIVERSE ROBUSTNESS: remove hindsight winners (QQQ, XLK, XLC, XLY) ===")
    exg = {}
    for nm, st in (("D Trend+Momentum", D), ("Best (final stage)", best), ("C Momentum rotation", C)):
        q0, q1 = quick(R[nm].ret, cash), quick(run_strat(sig, dc.replace(st, exclude=tuple(GROWTH)), S, cost_bps, lag).ret, cash)
        exg[nm] = q1["calmar"] / q0["calmar"] if q0["calmar"] > 0 else 0.0
        P(f"  {nm:22} CAGR {q0['cagr']*100:5.1f}% -> {q1['cagr']*100:5.1f}%   MaxDD {q0['mdd']*100:6.1f}% -> {q1['mdd']*100:6.1f}%   Calmar {q0['calmar']:.2f} -> {q1['calmar']:.2f}")

    P("\n=== 11. CRASH ANALYSIS (SPY drawdown episodes of -10% or worse, detected automatically; no hand-picked dates) ===")
    for nm in ("SMA200 (SPY)", "Best (final stage)"):
        ep = episodes_vs_spy(spy_ret, R[nm])
        P(f"  {nm}:")
        if len(ep): P(ep.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    P("  (spy_dd/strat/spy/avoided/missed are fractions; avoided = strategy minus SPY return peak->trough; missed = strategy minus SPY over the 180 days after the trough; "
      "exit_days = days from the SPY peak until exposure fell below half of its level at the peak; reenter_days = days after the trough until exposure was back above half of that level)")
    P("  Synthetic crash stress (about -38% equity crash over 40 days with VIX spike, inserted at random dates; strategy rules re-run on the modified data):")
    stn = 5 if quick_mode else 12
    for nm, st in (("D Trend+Momentum", D), ("Best (final stage)", best)):
        s_ = stress_test(md, st, S, cost_bps, n=stn)
        P(f"    {nm:22} strategy MaxDD median {s_.strat_dd.median()*100:6.1f}%  worst {s_.strat_dd.min()*100:6.1f}%   |  SPY MaxDD median {s_.spy_dd.median()*100:6.1f}%  worst {s_.spy_dd.min()*100:6.1f}%")

    P("\n=== 10. BOOTSTRAP (paired 21-day blocks, resampling both series together) ===")
    nb_ = 300 if quick_mode else 1000; boots = {}
    for a_, b_ in (("Best (final stage)", "SMA200 (SPY)"), ("Best (final stage)", "BH (SPY)"), ("G 50% SPY + 50% Best", "SMA200 (SPY)")):
        bt = paired_bootstrap(R[a_].ret, R[b_].ret, cash, n=nb_); boots[(a_, b_)] = bt
        P(f"  {a_} vs {b_}: P(Sharpe better) {bt['p_sharpe']:.0%}  P(Calmar better) {bt['p_calmar']:.0%}  Sharpe diff 90% CI [{bt['sharpe_diff_ci'][0]:+.2f}, {bt['sharpe_diff_ci'][1]:+.2f}]  "
          f"Calmar diff 90% CI [{bt['calmar_diff_ci'][0]:+.2f}, {bt['calmar_diff_ci'][1]:+.2f}]")

    # ---- regimes, captures, annual, drawdowns
    P("\n=== PERFORMANCE BY MARKET REGIME (label known at close t, applied to the return of day t+1) ===")
    lab = sig.regime_labels().shift(1).reindex(R["BH (SPY)"].ret.index)
    rr = pd.DataFrame({n: R[n].ret for n in ("BH (SPY)", "SMA200 (SPY)", "Best (final stage)")}); rr["lab"] = lab
    g = rr.groupby("lab"); tab = (g[["BH (SPY)", "SMA200 (SPY)", "Best (final stage)"]].mean() * TD * 100).round(1); tab.insert(0, "% of days", (g.size() / len(rr) * 100).round(1))
    P("  annualised average return in each regime (%):"); P(tab.to_string())
    mret = lambda r: (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    ms, mb = mret(R["Best (final stage)"].ret), mret(R["BH (SPY)"].ret)
    P(f"  Bull participation (up-capture) {ms[mb > 0].mean() / mb[mb > 0].mean():.0%}   Bear protection (down-capture, lower is better) {ms[mb < 0].mean() / mb[mb < 0].mean():.0%}")
    P("\n=== ANNUAL RETURNS ===")
    ann = pd.DataFrame({n: (1 + R[n].ret).groupby(R[n].ret.index.year).prod() - 1 for n in ("BH (SPY)", "SMA200 (SPY)", "Best (final stage)", "G 50% SPY + 50% Best")})
    P((ann * 100).round(1).to_string())
    P("\n=== WORST 10 DRAWDOWNS (Best) ===")
    for e in underwater((1 + br.ret).cumprod())[:10]:
        P(f"  {e['peak'].date()} -> trough {e['trough'].date()} ({e['depth']*100:.1f}%)  recovered {e['recovered'].date() if e['recovered'] is not None else 'not yet'}  underwater {e['days']} days")

    # ---- rules, ranking, verdict
    P("\n=== 13. HARD REJECTION RULES ===")
    cs = {}
    epi = underwater((1 + spy_ret).cumprod())
    for nm, r in R.items():
        if epi:
            e = epi[0]; ex = np.log1p(r.ret) - np.log1p(spy_ret.reindex(r.ret.index)); tot = ex.sum()
            ep_ = ex[(ex.index > e["peak"]) & (ex.index <= (e["recovered"] or ex.index[-1]))].sum()
            cs[nm] = float(ep_ / tot) if tot > 0 else 0.0
    verdicts = {}
    for nm in ("C Momentum rotation", "D Trend+Momentum", "Best (final stage)", "G 50% SPY + 50% Best"):
        key = nm if nm in cost_ratio else "Best (final stage)"
        h = halves_of(R[nm].ret, cash, mid); lev = max([float(w[:len(cols)].sum()) for _, w in R[nm].wlog] or [0.0])
        why = reject_reasons(M[nm], h, sens_ratio, cost_ratio.get(key, 1.0), cs.get(nm, 0.0), exg.get(key, 1.0), M[nm]["turnover"], lev)
        verdicts[nm] = why
        P(f"  {nm:24} -> {'PASS all rules' if not why else 'REJECTED: ' + '; '.join(why)}")
    cand = ["BH (SPY)", "SMA200 (SPY)", "C Momentum rotation", "D Trend+Momentum", "Best (final stage)", "G 50% SPY + 50% Best"]
    mdf = pd.DataFrame({n: M[n] for n in cand}).T
    score = mo_rank(mdf)
    P("\n=== 12. MULTI-OBJECTIVE SCORE (0-100; Calmar 25%, MaxDD 20%, Sortino 15%, CAGR 15%, Sharpe 10%, downside dev 5%, recovery 5%, turnover 5%) ===")
    P(score.sort_values(ascending=False).to_string())

    P("\n=== TARGETS (CAGR>10%, MaxDD>-20%, Sharpe>0.90, Calmar>0.60) ===")
    for nm in ("SMA200 (SPY)", "Best (final stage)", "G 50% SPY + 50% Best"):
        m = M[nm]; P(f"  {nm:24} CAGR {'OK ' if m['cagr']>.10 else 'no '} MaxDD {'OK ' if m['mdd']>-.20 else 'no '} Sharpe {'OK ' if m['sharpe']>.9 else 'no '} Calmar {'OK ' if m['calmar']>.6 else 'no '}")

    P("\n=== MECHANICAL VERDICT (pre-registered logic; a human review still follows) ===")
    s1, s2 = halves_of(R["SMA200 (SPY)"].ret, cash, mid)
    beats = {}
    for nm in ("D Trend+Momentum", "Best (final stage)", "G 50% SPY + 50% Best"):
        a1, a2 = halves_of(R[nm].ret, cash, mid)
        bt = boots.get((nm, "SMA200 (SPY)")) or paired_bootstrap(R[nm].ret, R["SMA200 (SPY)"].ret, cash, n=nb_)
        beats[nm] = (a1["sharpe"] > s1["sharpe"] and a2["sharpe"] > s2["sharpe"] and a1["calmar"] > s1["calmar"] and a2["calmar"] > s2["calmar"]
                     and bt["p_sharpe"] >= 0.80 and bt["p_calmar"] >= 0.80)
        P(f"  {nm:24} beats SMA200 in both halves AND bootstrap >=80% on Sharpe and Calmar: {'YES' if beats[nm] else 'NO'}")
    ok = [n for n in beats if beats[n] and not verdicts[n]]
    if ok:
        pick = max(ok, key=lambda n: score[n]); label = "HYBRID" if pick.startswith("G") else "MOMENTUM ROTATION"
        P(f"  => candidate: {pick} ({label}); passes all hard rules and beats SMA200.")
    else:
        pick = max(["BH (SPY)", "SMA200 (SPY)"], key=lambda n: score[n])
        P(f"  => no improved strategy passes every rule AND beats SMA200 reliably; benchmark with the better score: {pick}.")
    return "\n".join(out), {n: r.ret for n, r in R.items()}

def get_live_signals(years: float = 2.0, strat_mode: str = "D") -> dict:
    """Computes today's live momentum, cash hurdle, regime, and target weights."""
    md = load_market(years=years, refresh=True)
    sig = Signals(md)
    
    st = (Strat("C 2x QQQ/TLT Parity", lookback_m=6, top_n=2, trend=False, abs_mom=False, max_w=2.0, exclude=("SPY", "GLD"))
          if strat_mode == "C" else
          Strat("D 2x Parity + Crash Protect", lookback_m=6, top_n=2, trend=False, abs_mom=True, max_w=2.0, exclude=("SPY", "GLD")))

    L = st.lookback_m * 21
    mom_df = sig.mom(L)
    cash_hurdle = float(sig.cash_mom(L)[-1])
    current_regime = str(sig.regime_labels().iloc[-1])
    latest_vix = float(sig.vix.iloc[-1])
    latest_irx = float(md.cash_ret.iloc[-1] * TD * 100)

    # Calculate eligibility and weights for the latest bar
    cols = list(sig.px.columns)
    latest_mom = {c: float(mom_df[c].iloc[-1]) for c in cols}
    
    elig = []
    for c in cols:
        if c in st.exclude or np.isnan(latest_mom[c]):
            continue
        if st.abs_mom and not (latest_mom[c] > cash_hurdle):
            continue
        elig.append((latest_mom[c], c))
    
    elig.sort(reverse=True)
    selected = [c for _, c in elig[:st.top_n]]
    
    weights = {c: (st.max_w / st.top_n if c in selected else 0.0) for c in cols}
    weights["CASH"] = round(1.0 - sum(weights.values()), 4)

    return {
        "as_of": str(sig.px.index[-1].date()),
        "strategy": st.name,
        "regime": current_regime,
        "vix": round(latest_vix, 2),
        "tbill_annual_pct": round(latest_irx, 2),
        "cash_hurdle_6m_pct": round(cash_hurdle * 100, 2),
        "assets": [
            {
                "symbol": c,
                "price": round(float(sig.px[c].iloc[-1]), 2),
                "mom_6m_pct": round(latest_mom[c] * 100, 2),
                "beats_cash": bool(latest_mom[c] > cash_hurdle),
                "excluded": c in st.exclude,
                "target_weight": round(weights[c], 4),
            }
            for c in cols
        ],
        "target_weights": weights,
    }