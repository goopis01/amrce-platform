"""Performance metrics (spec section 13). Expectancy = WinRate*AvgWin - LossRate*AvgLoss."""
from __future__ import annotations
import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame, equity: pd.Series, bars_per_year: int) -> dict:
    n = len(trades)
    out = {"trades": n}
    ret = equity.pct_change().dropna()
    dd = (equity / equity.cummax() - 1).min() if len(equity) else 0.0
    years = max(len(equity) / bars_per_year, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) else 0.0
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    std, dstd = ret.std(), ret[ret < 0].std()
    out.update({
        "net_return": total, "cagr": cagr, "max_drawdown": dd,
        "sharpe": float(ret.mean() / std * np.sqrt(bars_per_year)) if std and std > 0 else 0.0,
        "sortino": float(ret.mean() / dstd * np.sqrt(bars_per_year)) if dstd and dstd > 0 else 0.0,
        "calmar": float(cagr / abs(dd)) if dd < 0 else 0.0,
    })
    if n == 0:
        return out
    wins, losses = trades[trades.net_pnl > 0], trades[trades.net_pnl <= 0]
    wr = len(wins) / n
    avg_w = wins.net_pnl.mean() if len(wins) else 0.0
    avg_l = losses.net_pnl.mean() if len(losses) else 0.0
    loss_sum = abs(losses.net_pnl.sum())
    streak = best = 0
    for v in (trades.net_pnl <= 0):
        streak = streak + 1 if v else 0
        best = max(best, streak)
    r = trades.r
    out.update({
        "win_rate": wr, "avg_win": avg_w, "avg_loss": avg_l,
        "expectancy": wr * avg_w + (1 - wr) * avg_l,
        "avg_r": r.mean(), "expectancy_r": r.mean(),
        "profit_factor": float(wins.net_pnl.sum() / loss_sum) if loss_sum > 0 else float("inf"),
        "tail_loss_r_cvar5": float(r[r <= r.quantile(0.05)].mean()),
        "max_consec_losses": best, "net_profit": trades.net_pnl.sum(),
        "recovery_factor": float(trades.net_pnl.sum() / (abs(dd) * equity.iloc[0])) if dd < 0 else 0.0,
        "avg_bars_held": trades.bars.mean(),
    })
    return out
