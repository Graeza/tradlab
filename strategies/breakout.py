from __future__ import annotations
import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal


class BreakoutStrategy(Strategy):
    name = "BREAKOUT"
    # Breakouts work best in trend regimes
    allowed_trends = {"TREND"}

    def __init__(
        self,
        lookback: int = 20,
        buffer_atr_mult: float = 0.30,   # require close beyond HH/LL by ATR buffer
        wick_to_body_max: float = 2.0,   # reject breakouts with rejection wicks
    ):
        self.lookback = int(lookback)
        self.buffer_atr_mult = float(buffer_atr_mult)
        self.wick_to_body_max = float(wick_to_body_max)

    @staticmethod
    def _find_atr_col(df: pd.DataFrame) -> str | None:
        for c in ("ATR", "ATR14", "atr", "atr14"):
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _wick_metrics(row: pd.Series) -> tuple[float, float]:
        # returns (upper_wick, lower_wick) in price units
        o = float(row.get("open", row.get("close", 0.0)) or 0.0)
        h = float(row.get("high", 0.0) or 0.0)
        l = float(row.get("low", 0.0) or 0.0)
        c = float(row.get("close", 0.0) or 0.0)
        upper = h - max(o, c)
        lower = min(o, c) - l
        return float(max(upper, 0.0)), float(max(lower, 0.0))

    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]):
        df = next(iter(data_by_tf.values()))
        if df is None or len(df) < self.lookback + 2:
            return StrategyResult(name=self.name, signal=Signal.HOLD, confidence=0.0, meta={"reason": "insufficient_bars"})

        # Live trading: ALWAYS use the last CLOSED candle (exclude forming bar)
        closed = df.iloc[:-1]
        if len(closed) < self.lookback + 1:
            return StrategyResult(name=self.name, signal=Signal.HOLD, confidence=0.0, meta={"reason": "insufficient_closed_bars"})

        recent = closed.tail(self.lookback + 1)
        prev = recent.iloc[:-1]
        last = recent.iloc[-1]

        hh = float(prev["high"].max())
        ll = float(prev["low"].min())
        c = float(last["close"])

        # ATR buffer (if available)
        buffer = 0.0
        atr_col = self._find_atr_col(recent)
        if atr_col:
            try:
                atr = float(last.get(atr_col, 0.0) or 0.0)
                if atr > 0:
                    buffer = float(self.buffer_atr_mult) * atr
            except Exception:
                buffer = 0.0

        # Wick rejection filter
        upper_wick, lower_wick = self._wick_metrics(last)
        o = float(last.get("open", last.get("close", 0.0)) or 0.0)
        body = abs(float(last.get("close", 0.0) or 0.0) - o)
        body = max(body, 1e-9)

        # BUY breakout: close must clear HH + buffer; reject if large upper wick (rejection)
        if c > (hh + buffer):
            if (upper_wick / body) > float(self.wick_to_body_max):
                return StrategyResult(
                    name=self.name,
                    signal=Signal.HOLD,
                    confidence=0.0,
                    meta={"reason": "buy_breakout_rejected_wick", "hh": hh, "close": c, "buffer": buffer, "upper_wick": upper_wick, "body": body},
                )
            return StrategyResult(name=self.name, signal=Signal.BUY, confidence=0.60, meta={"hh": hh, "close": c, "buffer": buffer})

        # SELL breakout: close must clear LL - buffer; reject if large lower wick (rejection)
        if c < (ll - buffer):
            if (lower_wick / body) > float(self.wick_to_body_max):
                return StrategyResult(
                    name=self.name,
                    signal=Signal.HOLD,
                    confidence=0.0,
                    meta={"reason": "sell_breakout_rejected_wick", "ll": ll, "close": c, "buffer": buffer, "lower_wick": lower_wick, "body": body},
                )
            return StrategyResult(name=self.name, signal=Signal.SELL, confidence=0.60, meta={"ll": ll, "close": c, "buffer": buffer})

        return StrategyResult(name=self.name, signal=Signal.HOLD, confidence=0.0, meta={"hh": hh, "ll": ll, "close": c, "buffer": buffer})
