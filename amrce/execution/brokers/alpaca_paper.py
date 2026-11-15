"""Alpaca PAPER broker. STUB (roadmap M6). Must always use paper=True until the go-live checklist passes."""
from __future__ import annotations
from .base import Broker


class AlpacaPaperBroker(Broker):
    def place(self, order): raise NotImplementedError("M6")
    def cancel(self, order_id): raise NotImplementedError("M6")
    def positions(self): raise NotImplementedError("M6")
    def equity(self): raise NotImplementedError("M6")
