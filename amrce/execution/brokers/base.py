from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from ...types import Order, Position


class Broker(ABC):
    @abstractmethod
    def place(self, order: Order) -> str: ...
    @abstractmethod
    def cancel(self, order_id: str) -> None: ...
    @abstractmethod
    def positions(self) -> List[Position]: ...
    @abstractmethod
    def equity(self) -> float: ...
