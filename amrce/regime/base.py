from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from ..types import Regime


class RegimeClassifier(ABC):
    """Stateful: feed it bars sequentially. `features` ends at the current closed bar."""
    def reset(self) -> None: ...

    @abstractmethod
    def classify(self, features: pd.DataFrame) -> Regime: ...
