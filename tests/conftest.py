import numpy as np, pandas as pd, pytest
from amrce.config import load_config


@pytest.fixture
def cfg():
    return load_config()


def make_bars(n=900, seed=1, start="2024-01-02 09:30", freq="1h", drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="America/New_York")
    ret = rng.normal(drift, 0.004, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0005, n))
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(1_000, 50_000, n).astype(float)}, index=idx)


@pytest.fixture
def bars():
    return make_bars()
