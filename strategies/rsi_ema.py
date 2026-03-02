from __future__ import annotations
import pandas as pd
from strategies.base import Strategy, StrategyResult, Signal
class RSIEMAStrategy(Strategy):
    name = "RSI_EMA"
    # RSI mean-reversion works best in ranges
    allowed_trends = {"RANGE"}

    def __init__(self, rsi_low: float = 30, rsi_high: float = 70, ema_fast: int = 10, ema_slow: int = 21):
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]):
        # Default uses the first (primary) df in dict
        df = next(iter(data_by_tf.values()))
        if df is None or df.empty:
            return StrategyResult(name=self.name, signal=Signal.HOLD, confidence=0.0, meta={"reason": "no_data"})

        row = df.iloc[-1]
        rsi = float(row.get("RSI", 50))
        ema_f = float(row.get(f"close_EMA{self.ema_fast}", row.get("close", 0)))
        ema_s = float(row.get(f"close_EMA{self.ema_slow}", row.get("close", 0)))

        if rsi <= self.rsi_low and ema_f > ema_s:
            return StrategyResult(name=self.name, signal=Signal.BUY, confidence=0.62, meta={"rsi": rsi, "ema": (ema_f, ema_s)})
        if rsi >= self.rsi_high and ema_f < ema_s:
            return StrategyResult(name=self.name, signal=Signal.SELL, confidence=0.62, meta={"rsi": rsi, "ema": (ema_f, ema_s)})

        return StrategyResult(name=self.name, signal=Signal.HOLD, confidence=0.4, meta={"rsi": rsi, "ema": (ema_f, ema_s)})
