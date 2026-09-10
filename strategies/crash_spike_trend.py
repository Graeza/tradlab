from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal
from strategies.boom_sell_decay import BoomSellDecayStrategy
from strategies.boom_spike_trend import BoomSpikeTrendStrategy


_META_KEY_TRANSLATIONS = {
    "upper_wick": "lower_wick",
    "lower_wick": "upper_wick",
    "upper_ratio": "lower_ratio",
    "lower_ratio": "upper_ratio",
    "h1_support": "h1_resistance",
    "h1_resistance": "h1_support",
    "dist_to_h1_support": "dist_to_h1_resistance",
    "dist_to_h1_resistance": "dist_to_h1_support",
    "near_h1_support": "near_h1_resistance",
    "near_h1_resistance": "near_h1_support",
    "impulse_down_m5": "impulse_up_m5",
    "impulse_up_m5": "impulse_down_m5",
    "trend_up_h4": "trend_down_h4",
    "close_off_high_atr": "close_off_low_atr",
    "close_below_ema": "close_above_ema",
    "prior_push_above_ema": "prior_push_below_ema",
}

_NEGATED_META_KEYS = {
    "h1_support",
    "h1_resistance",
    "m15_open",
    "m15_close",
    "m15_high",
    "m15_low",
    "m15_ema",
    "h4_last_close",
    "h4_ema50",
}

_META_VALUE_TRANSLATIONS = {
    "bullish": "bearish",
    "bearish": "bullish",
    "spike_sell": "spike_buy",
    "trend_buy": "trend_sell",
    "trend_sell": "trend_buy",
    "boom_sell_decay": "crash_buy_recovery",
    "blocked_by_h1_trend": "blocked_by_h1_trend",
    "too_close_to_h1_support": "too_close_to_h1_resistance",
}


def _mirror_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reflect prices so a downward Crash spike looks like an upward Boom spike."""
    mirrored = df.copy()
    if {"open", "high", "low", "close"}.issubset(mirrored.columns):
        original_high = mirrored["high"].copy()
        original_low = mirrored["low"].copy()
        mirrored["open"] = -mirrored["open"]
        mirrored["close"] = -mirrored["close"]
        mirrored["high"] = -original_low
        mirrored["low"] = -original_high

    for column in ("RSI", "rsi"):
        if column in mirrored.columns:
            mirrored[column] = 100.0 - mirrored[column]
    return mirrored


def _translate_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    translated: dict[str, Any] = {}
    for key, value in meta.items():
        new_key = _META_KEY_TRANSLATIONS.get(str(key), str(key))
        if key == "m15_high":
            new_key = "m15_low"
        elif key == "m15_low":
            new_key = "m15_high"
        if key in _NEGATED_META_KEYS and isinstance(value, (int, float)):
            value = -value
        if isinstance(value, str):
            value = _META_VALUE_TRANSLATIONS.get(value, value)
        translated[new_key] = value
    translated["mirrored_from_boom_strategy"] = True
    return translated


def _reverse_signal(signal: Signal) -> Signal:
    if signal == Signal.BUY:
        return Signal.SELL
    if signal == Signal.SELL:
        return Signal.BUY
    return Signal.HOLD


class _MirroredBoomStrategy(Strategy):
    boom_strategy_type: type[Strategy]

    def __init__(self, **kwargs: Any):
        self._boom_strategy = self.boom_strategy_type(**kwargs)

    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]) -> StrategyResult:
        mirrored_data = {tf: _mirror_frame(df) for tf, df in data_by_tf.items()}
        result = self._boom_strategy.evaluate(mirrored_data)
        return StrategyResult(
            name=self.name,
            signal=_reverse_signal(result.signal),
            confidence=result.confidence,
            meta=_translate_meta(result.meta),
        )


class CrashSpikeTrendStrategy(_MirroredBoomStrategy):
    """Directional mirror of the Boom spike/trend strategy for Crash indices."""

    name = "CRASH_SPIKE_TREND"
    boom_strategy_type = BoomSpikeTrendStrategy


class CrashBuyRecoveryStrategy(_MirroredBoomStrategy):
    """Buy the recovery after a downward Crash spike exhausts."""

    name = "CRASH_BUY_RECOVERY"
    boom_strategy_type = BoomSellDecayStrategy
