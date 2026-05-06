from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal


class RSI3MAExtremeStrategy(Strategy):
    """Daily RSI(3) + 3-period RSI moving-average extreme strategy.

    Intended for non-Boom symbols such as Wall Street 30 and XAUUSD.
    It uses closed D1 candles when available and emits:
      - SELL when RSI/RSI-MA enter or reject the overbought zone above 80.
      - BUY when RSI/RSI-MA enter or reject the oversold zone below 20.
    """

    name = "RSI3_MA_EXTREME"

    def __init__(
        self,
        *,
        daily_tf: int = 16408,
        rsi_period: int = 3,
        rsi_ma_period: int = 3,
        low_level: float = 20.0,
        high_level: float = 80.0,
        confidence: float = 0.70,
    ):
        self.daily_tf = int(daily_tf)
        self.rsi_period = int(rsi_period)
        self.rsi_ma_period = int(rsi_ma_period)
        self.low_level = float(low_level)
        self.high_level = float(high_level)
        self.confidence = max(0.0, min(1.0, float(confidence)))

    def _pick_daily_df(self, data_by_tf: dict[int, pd.DataFrame]) -> Optional[pd.DataFrame]:
        for key in (self.daily_tf, 1440):
            df = data_by_tf.get(int(key))
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        return None

    def evaluate(
        self,
        data_by_tf: dict[int, pd.DataFrame],
        context: Optional[Mapping[str, Any]] = None,
    ) -> StrategyResult:
        return self._evaluate(data_by_tf)

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.where(~((loss == 0.0) & (gain > 0.0)), 100.0)
        rsi = rsi.where(~((gain == 0.0) & (loss > 0.0)), 0.0)
        rsi = rsi.where(~((gain == 0.0) & (loss == 0.0)), 50.0)
        return rsi

    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]):
        df = self._pick_daily_df(data_by_tf)
        if df is None or df.empty:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {"reason": "no_daily_data", "daily_tf": self.daily_tf})

        required = self.rsi_period + self.rsi_ma_period + 1
        if len(df) < required:
            return StrategyResult(
                self.name,
                Signal.HOLD,
                0.0,
                {"reason": "insufficient_daily_bars", "bars": int(len(df)), "required": int(required)},
            )

        if "close" not in df.columns:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {"reason": "missing_close"})

        if "time" in df.columns:
            work = df.drop_duplicates(subset=["time"]).sort_values("time")
        else:
            work = df.copy()
        close = work["close"].astype(float)
        rsi = self._rsi(close, self.rsi_period)
        rsi_ma = rsi.rolling(self.rsi_ma_period).mean()

        latest = pd.DataFrame({"rsi": rsi, "rsi_ma": rsi_ma}).dropna().tail(2)
        if len(latest) < 2:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {"reason": "indicators_not_ready"})

        prev = latest.iloc[0]
        curr = latest.iloc[1]
        prev_rsi = float(prev["rsi"])
        prev_ma = float(prev["rsi_ma"])
        rsi_now = float(curr["rsi"])
        ma_now = float(curr["rsi_ma"])

        crossed_above_high = (
            prev_rsi <= self.high_level
            and prev_ma <= self.high_level
            and rsi_now > self.high_level
            and ma_now > self.high_level
        )
        crossed_below_low = (
            prev_rsi >= self.low_level
            and prev_ma >= self.low_level
            and rsi_now < self.low_level
            and ma_now < self.low_level
        )
        bearish_rejection = (
            prev_rsi >= prev_ma
            and rsi_now < ma_now
            and rsi_now >= self.high_level
            and ma_now >= self.high_level
        )
        bullish_rejection = (
            prev_rsi <= prev_ma
            and rsi_now > ma_now
            and rsi_now <= self.low_level
            and ma_now <= self.low_level
        )

        base_meta = {
            "rsi_period": self.rsi_period,
            "rsi_ma_period": self.rsi_ma_period,
            "low_level": self.low_level,
            "high_level": self.high_level,
            "daily_tf": self.daily_tf,
            "prev_rsi": prev_rsi,
            "prev_rsi_ma": prev_ma,
            "rsi": rsi_now,
            "rsi_ma": ma_now,
            "crossed_above_high": bool(crossed_above_high),
            "crossed_below_low": bool(crossed_below_low),
            "bearish_rejection": bool(bearish_rejection),
            "bullish_rejection": bool(bullish_rejection),
        }

        if crossed_below_low or bullish_rejection:
            return StrategyResult(
                self.name,
                Signal.BUY,
                self.confidence,
                {**base_meta, "reason": "rsi3_ma3_oversold"},
            )

        if crossed_above_high or bearish_rejection:
            return StrategyResult(
                self.name,
                Signal.SELL,
                self.confidence,
                {**base_meta, "reason": "rsi3_ma3_overbought"},
            )

        return StrategyResult(self.name, Signal.HOLD, 0.0, {**base_meta, "reason": "no_extreme_cross"})
