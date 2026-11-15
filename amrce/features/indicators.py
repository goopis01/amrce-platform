"""Pure, causal indicator functions: value at row t uses only rows <= t."""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    out = 100 - 100 / (1 + au / ad.replace(0, np.nan))
    return out.where(ad != 0, 100.0).where(au.notna())


def stochastic(df: pd.DataFrame, k: int = 14, smooth: int = 7, d: int = 3) -> pd.DataFrame:
    ll, hh = df["low"].rolling(k).min(), df["high"].rolling(k).max()
    raw = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    kk = raw.rolling(smooth).mean()
    return pd.DataFrame({"stoch_k": kk, "stoch_d": kk.rolling(d).mean()})


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid, sd = s.rolling(n).mean(), s.rolling(n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    return pd.DataFrame({"bb_mid": mid, "bb_up": up, "bb_lo": lo, "bbw": (up - lo) / mid})


def rolling_pct_rank(s: pd.Series, n: int) -> pd.Series:
    """Percentile (0-100) of the latest value inside its trailing n-window (causal)."""
    return s.rolling(n, min_periods=max(20, n // 4)).apply(lambda x: (x <= x[-1]).mean() * 100, raw=True)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    hc = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ho = np.empty(len(df))
    ho[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ho[i] = (ho[i - 1] + hc.iloc[i - 1]) / 2
    ho = pd.Series(ho, index=df.index)
    return pd.DataFrame({"ha_open": ho, "ha_close": hc,
                         "ha_high": pd.concat([df["high"], ho, hc], axis=1).max(axis=1),
                         "ha_low": pd.concat([df["low"], ho, hc], axis=1).min(axis=1)})


def narrow_range(df: pd.DataFrame, n: int = 7) -> pd.Series:
    rng = df["high"] - df["low"]
    return rng <= rng.rolling(n).min()
