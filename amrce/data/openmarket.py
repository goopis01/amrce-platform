"""Crypto perps via OpenMarket. STUB.

Open question to resolve first: does OpenMarket expose a headless REST API (see 'API Keys' / 'API Docs'
in its console)? The MCP route needs a live browser tab and is not suitable for a server loop.
Fallback for research: export history with the MCP `points` tool to CSV and use CSVData.
"""
from __future__ import annotations
from .base import DataProvider


class OpenMarketData(DataProvider):
    def get_bars(self, symbol, timeframe, start=None, end=None):
        raise NotImplementedError("Implement after confirming OpenMarket REST access; use CSVData meanwhile.")
