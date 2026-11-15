from __future__ import annotations
import os
from typing import Dict, List
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


class AlpacaRebalancer:
    def __init__(self, api_key: str | None = None, secret_key: str | None = None, paper: bool = True):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.paper = paper
        if not self.api_key or not self.secret_key:
            raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY environment variables.")
        self.client = TradingClient(self.api_key, self.secret_key, paper=self.paper)

    def get_portfolio_state(self) -> dict:
        acct = self.client.get_account()
        positions = self.client.get_all_positions()
        equity = float(acct.equity)
        pos_map = {}
        for p in positions:
            mv = float(p.market_value)
            pos_map[p.symbol] = {
                "qty": float(p.qty),
                "market_value": mv,
                "current_weight": round(mv / equity, 4) if equity > 0 else 0.0,
                "unrealized_pl": float(p.unrealized_pl),
            }
        return {
            "equity": equity,
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "paper": self.paper,
            "positions": pos_map,
        }

    def rebalance(self, target_weights: Dict[str, float], dry_run: bool = True) -> List[dict]:
        """
        Synchronizes Alpaca positions with target_weights (e.g. {'QQQ': 1.0, 'TLT': 1.0}).
        Executes SELL orders before BUY orders so buying power is freed first.
        """
        state = self.get_portfolio_state()
        equity = state["equity"]
        positions = state["positions"]
        
        # Safety check: never exceed 2.0x gross exposure
        gross_leverage = sum(v for k, v in target_weights.items() if k != "CASH")
        if gross_leverage > 2.0 + 1e-6:
            raise ValueError(f"ExecutionGuard Veto: Gross leverage {gross_leverage:.2f}x exceeds 2.0x limit.")

        sells, buys = [], []
        all_symbols = set(positions.keys()) | {k for k in target_weights.keys() if k != "CASH"}

        for sym in sorted(all_symbols):
            tgt_w = target_weights.get(sym, 0.0)
            tgt_val = round(equity * tgt_w, 2)
            cur_val = positions.get(sym, {}).get("market_value", 0.0)
            diff_val = round(tgt_val - cur_val, 2)

            # Ignore tiny drift under $50 or 1% of equity
            if abs(diff_val) < max(50.0, equity * 0.01):
                continue

            if diff_val < 0:
                sells.append((sym, abs(diff_val), tgt_w))
            elif diff_val > 0:
                buys.append((sym, diff_val, tgt_w))

        orders_log = []
        # 1. Execute SELLS first
        for sym, notional, tgt_w in sells:
            action = {"symbol": sym, "side": "SELL", "notional_usd": notional, "target_weight": tgt_w, "status": "DRY_RUN"}
            if not dry_run:
                # If target weight is 0, liquidate entire position cleanly
                if tgt_w == 0.0 and sym in positions:
                    self.client.close_position(sym)
                    action["status"] = "CLOSED_POSITION"
                else:
                    req = MarketOrderRequest(symbol=sym, notional=notional, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                    resp = self.client.submit_order(req)
                    action["status"] = str(resp.status)
            orders_log.append(action)

        # 2. Execute BUYS second
        for sym, notional, tgt_w in buys:
            action = {"symbol": sym, "side": "BUY", "notional_usd": notional, "target_weight": tgt_w, "status": "DRY_RUN"}
            if not dry_run:
                req = MarketOrderRequest(symbol=sym, notional=notional, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                resp = self.client.submit_order(req)
                action["status"] = str(resp.status)
            orders_log.append(action)

        return orders_log