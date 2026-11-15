"""Last line of defence between a decision and a broker. Re-checks hard limits independently of the risk engine."""
from __future__ import annotations
from ..risk.killswitch import KillSwitch
from ..types import Order
from .brokers.base import Broker


class ExecutionGuard:
    def __init__(self, broker: Broker, killswitch: KillSwitch, max_notional_pct: float = 0.25):
        self.broker, self.kill, self.max_notional_pct = broker, killswitch, max_notional_pct
        self._seen = set()

    def submit(self, order: Order, price: float) -> str:
        if self.kill.active():
            raise PermissionError("kill switch active")
        if not order.idempotency_key:
            raise ValueError("idempotency_key required")
        if order.idempotency_key in self._seen:
            raise ValueError("duplicate order")
        if order.qty * price > self.max_notional_pct * self.broker.equity():
            raise PermissionError("order notional exceeds hard cap")
        self._seen.add(order.idempotency_key)
        return self.broker.place(order)
