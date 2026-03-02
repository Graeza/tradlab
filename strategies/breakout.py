from __future__ import annotations
import pandas as pd
from strategies.base import Strategy, StrategyResult, Signal
class BreakoutStrategy(Strategy):
    name = "BREAKOUT"
    # Breakouts work best in trend regimes
    allowed_trends = {"TREND"}

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]):
        df = next(iter(data_by_tf.values()))
        if df is None or len(df) < self.lookback + 2:
            return StrategyResult(name=self.name, signal=Signal.HOLD, confidence=0.0, meta={"reason": "insufficient_bars"})

        recent = df.tail(self.lookback + 1)
        prev = recent.iloc[:-1]
        last = recent.iloc[-1]

        hh = float(prev["high"].max())
        ll = float(prev["low"].min())
        c = float(last["close"])

        if c > hh:
            return StrategyResult(name=self.name, signal=Signal.BUY, confidence=0.60, meta={"hh": hh, "close": c})
        if c < ll:
            return StrategyResult(name=self.name, signal=Signal.SELL, confidence=0.60, meta={"ll": ll, "close": c})
        return StrategyResult(name=self.name, signal=Signal.HOLD, confidence=0.35, meta={"hh": hh, "ll": ll, "close": c})
