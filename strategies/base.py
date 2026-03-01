from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

StrategyOutput = Dict[str, Any]

class Strategy(ABC):
    name: str

    @abstractmethod
    def evaluate(self, data_by_tf: dict[int, pd.DataFrame]) -> StrategyOutput:
        """Return {name, signal, confidence, meta}.

        data_by_tf maps MT5 timeframe -> feature dataframe.
        """

        raise NotImplementedError
