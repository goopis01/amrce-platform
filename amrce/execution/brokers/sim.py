"""Minimal in-memory broker for dry runs (the backtest engine does its own bar-level fills)."""
from __future__ import annotations
from typing import Dict, List
from ...types import Order, Position
from .base import Broker


class SimBroker(Broker):
    def __init__(self, equity: float = 100_000):
        self._equity, self.orders, self._n = equity, {}, 0

    def place(self, order: Order) -> str:
        self._n += 1
        oid = f"sim-{self._n}"
        self.orders[oid] = order
        return oid

    def cancel(self, order_id): self.orders.pop(order_id, None)
    def positions(self) -> List[Position]: return []
    def equity(self) -> float: return self._equity
