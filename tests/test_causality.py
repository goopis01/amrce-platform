import pytest
from amrce.features.registry import build_feature_pipeline, REGISTRY
from amrce.features.causality import assert_causal
from amrce.features import indicators as ind


def test_enabled_feature_modules_are_causal(cfg, bars):
    fp = build_feature_pipeline(cfg)
    assert_causal(fp.compute, bars, cut=600)


def test_indicators_are_causal(bars):
    import pandas as pd
    fn = lambda b: pd.concat([ind.stochastic(b), ind.bollinger(b.close), ind.heikin_ashi(b),
                              ind.rsi(b.close, 2).rename("rsi2"), ind.atr(b).rename("atr")], axis=1)
    assert_causal(fn, bars, cut=500)


def test_look_ahead_guard_actually_catches_leaks(bars):
    leaky = lambda b: b[["close"]].assign(lead=b.close.shift(-1))
    with pytest.raises(AssertionError):
        assert_causal(leaky, bars, cut=500)


def test_unbuilt_modules_fail_loudly(cfg, bars):
    with pytest.raises(NotImplementedError):
        REGISTRY["structure"](cfg.features.params).compute(bars)
