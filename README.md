# AMRCE: Adaptive Multi-Regime Confluence Engine (architecture scaffold)

**Stage: M1 + M2 done (architecture and plumbing). No trading strategy exists yet.** Nothing here claims an edge.
The tests prove the *machinery* is correct (causal features, no same-bar fills, risk limits, fail-closed AI layer).

## Quick start
```bash
pip install -e ".[dev]"      # numpy pandas pyyaml (+pytest)
pytest -q                    # 30 tests should pass

# run the baseline benchmark on real data (daily, 15y)
pip install yfinance
python -m amrce.cli backtest --symbol SPY --tf 1D --years 15 --source yahoo --null 100
```

## Data flow (one decision per closed bar)
```
DataProvider ──> validators ──> FeaturePipeline (modules, causal)
                                     │
                       RegimeClassifier (rules; HMM + drift = M4)
                                     │
        SetupDetectors (per regime) ─┴─> confirmations >= 2 ─> ConfluenceScorer (0-100)
                                     │
                          build plan (R:R net of costs, ATR stop bounds)
                                     │
                    Reviewer (off | shadow | Claude)  ── veto / downgrade only
                                     │
         sizing.compute_qty ─> RiskEngine.approve ─> ExecutionGuard ─> Broker
```
`core/pipeline.py::StrategyPipeline.evaluate()` returns a `Decision` that records **which stage stopped the trade and why**.

## Layout
| Path | Responsibility | State |
|---|---|---|
| `types.py`, `config.py`, `config/default.yaml` | Shared types, typed config (all thresholds) | done |
| `data/` | `DataProvider`, validators, CSV + Alpaca loaders; OpenMarket stub | CSV/Alpaca done, OpenMarket stub |
| `features/` | `FeatureModule` contract, indicators, `trend`, `volatility`; 4 modules are stubs | 2 of 6 built |
| `regime/` | Rule classifier (hysteresis) built; `hmm.py`, `drift.py` stubs | rules done |
| `signals/` | `SetupDetector`, `ConfluenceScorer` built; 5 setups are stubs | framework done |
| `ai/` | Strict JSON schema, `Null`/`Shadow`/`Claude` reviewers (fail-closed) | done, API call untested |
| `risk/` | Sizing, risk engine (loss limits, buckets, cooldown), kill switch | done |
| `execution/` | `ExecutionGuard`, `Broker` interface, `SimBroker`; Alpaca paper stub | guard done |
| `backtest/` | Causal engine, metrics, walk-forward windows, Monte Carlo | done (single symbol) |
| `live/` | Runner | stub |

## Non-negotiable rules baked into the design
1. Features at row *t* use only rows <= *t* (`features/causality.py` + tests).
2. Signal on bar close -> fill at **next open** + slippage; stop assumed first if stop and target share a bar.
3. The reviewer can veto or lower size, never raise it; any error => veto (fail closed).
4. Daily-loss breach trips a kill switch; loss-streak cooldown expires (a bug the tests caught).
5. Config only in YAML; one change = one new version of the config.

## Roadmap (build order)
- **M1** Architecture, types, config, contracts: done
- **M2** Backtest engine, risk, sizing, metrics, tests: done
- **M3a** Baseline (trend + RSI2 pullback), benchmark report, random-entry null test, CLI: done
- **M3b** Real modules: momentum, price_action, structure, breakout; setups S1-S5; score components; CLI to run a backtest on Alpaca/CSV data
- **M4** Baseline + ablation harness; HMM + drift; walk-forward runner; Monte Carlo + null test
- **M5** Multi-symbol portfolio engine (buckets, VaR)
- **M6** Shadow mode -> paper trading -> go-live checklist

## Known limitations of this scaffold
- Backtest engine is single-symbol, one position at a time.
- No intraday session flattening or event calendar yet.
- `ClaudeReviewer` API call has not been exercised against the live API (tests use an injected fake).
- OpenMarket data/execution adapter waits on confirming a headless REST API.
