"""python -m amrce.cli backtest --symbol SPY --tf 1D --years 15 --source yahoo --null 100
   python -m amrce.cli exposure --symbols SPY QQQ IWM EFA TLT GLD --years 15"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
import pandas as pd
from .backtest.report import format_report, run_report


def load_bars(source: str, symbol: str, tf: str, years: float, csv_dir: str):
    start = (datetime.now() - timedelta(days=int(365 * years))).strftime("%Y-%m-%d")
    if source == "yahoo":
        from .data.yahoo import YahooData
        return YahooData().get_bars(symbol, tf, start=start)
    if source == "alpaca":
        from .data.alpaca import AlpacaData
        return AlpacaData().get_bars(symbol, tf, start=start)
    from .data.csv_loader import CSVData
    return CSVData(csv_dir).get_bars(symbol, tf)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="amrce")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--symbol", default="SPY")
    b.add_argument("--tf", default="1D", choices=["1D", "1h", "15m"])
    b.add_argument("--years", type=float, default=10)
    b.add_argument("--source", default="yahoo", choices=["yahoo", "alpaca", "csv"])
    b.add_argument("--csv-dir", default="data_csv")
    b.add_argument("--null", type=int, default=100, help="random-entry null runs (0 to skip)")
    b.add_argument("--shorts", action="store_true")
    e = sub.add_parser("exposure")
    e.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD"])
    e.add_argument("--years", type=float, default=15)
    e.add_argument("--source", default="yahoo", choices=["yahoo", "alpaca", "csv"])
    e.add_argument("--csv-dir", default="data_csv")
    e.add_argument("--target-vol", type=float, default=0.15)
    e.add_argument("--cost-bps", type=float, default=2.0)
    r = sub.add_parser("research")
    r.add_argument("--years", type=float, default=22)
    r.add_argument("--cost-bps", type=float, default=5.0)
    r.add_argument("--quick", action="store_true", help="smaller grids, faster")
    r.add_argument("--refresh", action="store_true", help="re-download data")
    a = ap.parse_args(argv)

    if a.cmd == "research":
        from pathlib import Path
        from .research import load_market, run_research
        txt, rets = run_research(load_market(a.years, refresh=a.refresh), a.cost_bps, quick_mode=a.quick)
        print(txt)
        Path("reports").mkdir(exist_ok=True)
        Path("reports/research_report.txt").write_text(txt)
        pd.DataFrame(rets).to_csv("reports/research_daily_returns.csv")
        print("\nSaved reports/research_report.txt and reports/research_daily_returns.csv")
        return

    if a.cmd == "exposure":
        from .backtest.exposure import format_exposure, run_exposure
        closes = pd.concat({s: load_bars(a.source, s, "1D", a.years, a.csv_dir)["close"] for s in a.symbols}, axis=1).dropna()
        print(f"Loaded {len(closes)} common days {closes.index[0].date()} -> {closes.index[-1].date()}")
        print(format_exposure(closes, run_exposure(closes, a.cost_bps, target_vol=a.target_vol)))
        return

    bars = load_bars(a.source, a.symbol, a.tf, a.years, a.csv_dir)
    print(f"Loaded {len(bars)} bars {bars.index[0].date()} -> {bars.index[-1].date()}")
    rep = run_report(bars, a.symbol, a.tf, a.null, a.shorts)
    print(format_report(a.symbol, a.tf, rep))
    from .backtest.extras import report_extras
    print(report_extras(bars, rep["cfg"]))


if __name__ == "__main__":
    main()