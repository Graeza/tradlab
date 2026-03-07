from __future__ import annotations
import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal


class RSIEMAStrategy(Strategy):
    name = "RSI_EMA"
    # RSI mean-reversion works best in ranges
    allowed_trends = {"RANGE"}

    def __init__(
        self,
        rsi_low: float = 30,
        rsi_high: float = 70,
        ema_fast: int = 10,
        ema_slow: int = 21,
        spike_cooldown_bars: int = 3,
        spike_atr_mult: float = 2.0,
    ):
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.spike_cooldown_bars = int(spike_cooldown_bars)
        self.spike_atr_mult = float(spike_atr_mult)

    @staticmethod
    def _find_atr_col(df: pd.DataFrame) -> str | None:
        for c in ("ATR", "ATR14", "atr", "atr14"):
            if c in df.columns:
                return c
        return None

    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]):
        # Default uses the first (primary) df in dict
        df = next(iter(data_by_tf.values()))
        if df is None or df.empty:
            return StrategyResult(name=self.name, signal=Signal.HOLD, confidence=0.0, meta={"reason": "no_data"})

        # Live trading: ALWAYS use the last CLOSED candle (exclude forming bar)
        if len(df) < 2:
            return StrategyResult(name=self.name, signal=Signal.HOLD, confidence=0.0, meta={"reason": "insufficient_bars"})

        closed = df.iloc[:-1]
        row = closed.iloc[-1]

        # Optional spike cooldown (helps Boom/Crash chop after spikes)
        atr_col = self._find_atr_col(closed)
        if atr_col and {"high", "low"}.issubset(closed.columns) and self.spike_cooldown_bars > 0:
            tail = closed.tail(max(self.spike_cooldown_bars, 1))
            try:
                atr = float(row.get(atr_col, 0.0) or 0.0)
                if atr > 0:
                    ranges = (tail["high"].astype(float) - tail["low"].astype(float)).abs()
                    if float(ranges.max()) > float(self.spike_atr_mult) * atr:
                        return StrategyResult(
                            name=self.name,
                            signal=Signal.HOLD,
                            confidence=0.0,
                            meta={"reason": "spike_cooldown", "max_range": float(ranges.max()), "atr": atr},
                        )
            except Exception:
                pass

        rsi = float(row.get("RSI", 50))
        ema_f = float(row.get(f"close_EMA{self.ema_fast}", row.get("close", 0)))
        ema_s = float(row.get(f"close_EMA{self.ema_slow}", row.get("close", 0)))

        if rsi <= self.rsi_low and ema_f > ema_s:
            return StrategyResult(name=self.name, signal=Signal.BUY, confidence=0.62, meta={"rsi": rsi, "ema": (ema_f, ema_s)})

        if rsi >= self.rsi_high and ema_f < ema_s:
            return StrategyResult(name=self.name, signal=Signal.SELL, confidence=0.62, meta={"rsi": rsi, "ema": (ema_f, ema_s)})

        # HOLD should not contribute to ensemble confidence
        return StrategyResult(name=self.name, signal=Signal.HOLD, confidence=0.0, meta={"rsi": rsi, "ema": (ema_f, ema_s)})
