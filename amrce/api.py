from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from amrce.research import get_live_signals
from amrce.execution.alpaca_live import AlpacaRebalancer

app = FastAPI(title="AMRCE Quantitative Execution Terminal")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class RebalancePayload(BaseModel):
    strategy_mode: str = "D"   # "C" (17.3% CAGR) or "D" (11.9% CAGR + Crash Protect)
    dry_run: bool = True


@app.get("/api/dashboard")
def get_dashboard_data(strategy_mode: str = "D"):
    signals = get_live_signals(years=2.0, strat_mode=strategy_mode)
    
    # Load cumulative equity curve from saved research CSV
    csv_path = Path("reports/research_daily_returns.csv")
    chart_data = []
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        eq = (1 + df).cumprod() * 100_000
        # Downsample to monthly points for fast frontend rendering
        monthly_eq = eq.resample("ME").last()
        for dt, row in monthly_eq.iterrows():
            chart_data.append({
                "date": dt.strftime("%Y-%m"),
                "SPY": round(float(row.get("BH (SPY)", 100000)), 0),
                "Strat_C": round(float(row.get("C Momentum rotation", 100000)), 0),
                "Strat_D": round(float(row.get("Best (final stage)", 100000)), 0),
            })

    return {"signals": signals, "equity_curve": chart_data}


@app.get("/api/alpaca/account")
def get_alpaca_account():
    try:
        reb = AlpacaRebalancer(paper=True)
        return reb.get_portfolio_state()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/alpaca/rebalance")
def trigger_rebalance(payload: RebalancePayload):
    try:
        sig = get_live_signals(years=2.0, strat_mode=payload.strategy_mode)
        reb = AlpacaRebalancer(paper=True)
        orders = reb.rebalance(sig["target_weights"], dry_run=payload.dry_run)
        return {"signals": sig, "orders": orders}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))