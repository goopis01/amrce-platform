"""Causal, bar-by-bar single-symbol backtester.
Rules: signal on the CLOSE of bar i -> entry at OPEN of bar i+1 (+slippage). If stop and target are both inside
one bar, the STOP is assumed to hit first. Stops fill at the worse of stop/open (gaps). Costs on every fill.
Limitation (by design for M2): one position at a time; multi-symbol portfolio engine is roadmap M5."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from ..core.pipeline import StrategyPipeline
from ..types import Decision, PortfolioState, Position, Side


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    decisions: pd.DataFrame


class _Open:
    pass


class BacktestEngine:
    def __init__(self, cfg, pipeline: StrategyPipeline):
        self.cfg, self.pipe = cfg, pipeline

    def _fee(self, px, qty):
        return px * qty * self.cfg.costs.commission_bps / 1e4

    def _slip(self, px, buying):
        s = px * self.cfg.costs.slippage_bps / 1e4
        return px + s if buying else px - s

    def run(self, symbol: str, bars: pd.DataFrame, warmup: Optional[int] = None,
            feats: Optional[pd.DataFrame] = None) -> BacktestResult:
        bt = self.cfg.backtest
        feats = self.pipe.prepare(bars) if feats is None else feats
        warm = bt.warmup_bars if warmup is None else warmup
        self.pipe.reset()
        O, H, L, C = (feats[k].values for k in ("open", "high", "low", "close"))
        self.realized = bt.initial_equity
        pf = PortfolioState(self.realized, self.realized, self.realized, self.realized)
        pos, pending = None, None
        trades, eq, decs = [], [], []
        last_day = last_wk = None

        for i in range(len(feats)):
            ts = feats.index[i]
            day, wk = ts.date(), tuple(ts.isocalendar())[:2]
            if day != last_day:
                pf.equity_sod, pf.stops_today, last_day, pf.today = pf.equity, {}, day, day
            if wk != last_wk:
                pf.equity_sow, last_wk = pf.equity, wk

            if pending is not None and pos is None:           # 1) fill at this bar's open
                pos = self._open(pending, O[i], i, ts)
                pending = None
            if pos is not None:                               # 2) manage with this bar's range
                if self._manage(pos, O[i], H[i], L[i], C[i], i, ts, trades, pf, symbol):
                    pos = None
            unreal = 0.0 if pos is None else pos.sign * (C[i] - pos.entry_fill) * pos.qty_open
            pf.equity = self.realized + unreal                # 3) mark to market
            pf.peak_equity = max(pf.peak_equity, pf.equity)
            eq.append(pf.equity)

            if pos is None and pending is None and i >= warm and i < len(feats) - 1:   # 4) decide
                pf.positions = []
                d = self.pipe.evaluate(symbol, feats.iloc[: i + 1], pf)
                decs.append({"ts": ts, "stage": d.stage, "reason": d.reason,
                             "regime": d.regime.value if d.regime else None, "score": d.score})
                if d.approved:
                    pending = (d, ts)

        if pos is not None:                                    # close at last close
            self._exit(pos, C[-1], pos.qty_open, "end", True, feats.index[-1], trades, pf, symbol, len(feats) - 1)
            eq[-1] = self.realized
        cols = ["symbol", "side", "setup", "regime", "signal_ts", "entry_ts", "exit_ts", "entry", "exit", "qty",
                "net_pnl", "r", "bars", "reason"]
        return BacktestResult(pd.DataFrame(trades, columns=cols), pd.Series(eq, index=feats.index, name="equity"),
                              pd.DataFrame(decs))

    # ---- internals ----
    def _open(self, pending, open_px, i, ts):
        d, sig_ts = pending
        pl = d.plan
        p = _Open()
        p.plan, p.sign, p.qty_total, p.qty_open = pl, pl.side.sign, d.qty, d.qty
        p.entry_fill = self._slip(open_px, pl.side is Side.LONG)
        p.stop, p.tp1, p.tp2, p.tp1_done = pl.stop, pl.tp1, pl.tp2, False
        p.init_risk = d.qty * abs(p.entry_fill - pl.stop)
        p.pnl, p.fees = 0.0, self._fee(p.entry_fill, d.qty)
        self.realized -= p.fees
        p.opened_i, p.opened_ts, p.signal_ts = i, ts, sig_ts
        return p

    def _exit(self, p, px, qty, reason, slip, ts, trades, pf, symbol, i):
        buying = p.sign < 0
        fill = self._slip(px, buying) if slip else px
        gross, fee = p.sign * (fill - p.entry_fill) * qty, self._fee(fill, qty)
        p.pnl += gross
        p.fees += fee
        self.realized += gross - fee
        p.qty_open -= qty
        p.exit_px_last = fill
        if p.qty_open <= 1e-9:
            net = p.pnl - p.fees
            trades.append([symbol, p.plan.side.value, p.plan.setup, p.plan.regime.value, p.signal_ts, p.opened_ts,
                           ts, p.entry_fill, fill, p.qty_total, net, net / p.init_risk if p.init_risk else 0.0,
                           i - p.opened_i, reason])
            if net < 0:
                pf.consecutive_losses += 1
                if pf.consecutive_losses >= self.cfg.risk.loss_streak_cooldown:
                    pf.cooldown_until = ts.date() + pd.Timedelta(days=1)   # pause rest of today + next session
                    pf.consecutive_losses = 0
                if reason == "stop":
                    pf.stops_today[symbol] = pf.stops_today.get(symbol, 0) + 1
            else:
                pf.consecutive_losses = 0
            return True
        return False

    def _manage(self, p, o, h, l, c, i, ts, trades, pf, symbol) -> bool:
        bt, s = self.cfg.backtest, p.sign
        if (l <= p.stop) if s > 0 else (h >= p.stop):                       # stop first
            px = min(p.stop, o) if s > 0 else max(p.stop, o)
            return self._exit(p, px, p.qty_open, "stop", True, ts, trades, pf, symbol, i)
        if bt.tp1_fraction > 0 and not p.tp1_done and ((h >= p.tp1) if s > 0 else (l <= p.tp1)):
            self._exit(p, p.tp1, p.qty_total * bt.tp1_fraction, "tp1", False, ts, trades, pf, symbol, i)
            p.tp1_done = True
            if bt.move_stop_to_be_after_tp1:
                p.stop = p.entry_fill
        if (h >= p.tp2) if s > 0 else (l <= p.tp2):
            px = max(p.tp2, o) if s > 0 else min(p.tp2, o)
            return self._exit(p, px, p.qty_open, "tp2", False, ts, trades, pf, symbol, i)
        if i - p.opened_i >= bt.time_stop_bars:
            return self._exit(p, c, p.qty_open, "time", True, ts, trades, pf, symbol, i)
        return False
