from __future__ import annotations
from typing import List, Tuple
import pandas as pd
from strategies.base import Strategy, StrategyOutput

class EnsembleEngine:
    def __init__(self, strategies: List[Strategy], weights: dict[str, float] | None = None, min_conf: float = 0.55):
        self.strategies = strategies
        self.weights = weights or {s.name: 1.0 for s in strategies}
        self.min_conf = min_conf

    def run(self, data_by_tf: dict[int, pd.DataFrame]) -> Tuple[dict, List[StrategyOutput]]:
        outputs: List[StrategyOutput] = []
        score = 0.0
        total = 0.0

        for s in self.strategies:
            out = s.evaluate(data_by_tf)
            outputs.append(out)

            sig = (out.get("signal") or "HOLD").upper()
            conf = float(out.get("confidence") or 0.0)
            w = float(self.weights.get(out.get("name", s.name), 1.0))

            if sig == "HOLD" or conf < self.min_conf:
                continue

            x = 1.0 if sig == "BUY" else -1.0
            score += w * conf * x
            total += w * conf

        if total == 0:
            final = {"signal": "HOLD", "confidence": 0.0}
        else:
            final = {"signal": "BUY" if score > 0 else "SELL", "confidence": min(1.0, abs(score) / total)}

        return final, outputs
