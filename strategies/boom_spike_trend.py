from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal
from utils.indicators import add_h1_context_to_df


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty or len(df) < period + 2:
        return float("nan")
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v)


def _bb_width(df: pd.DataFrame, period: int = 20, stdev: float = 2.0) -> float:
    if df is None or df.empty or len(df) < period + 2:
        return float("nan")
    close = df["close"]
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + stdev * sd
    lower = ma - stdev * sd
    last_ma = float(ma.iloc[-1])
    if last_ma == 0.0 or np.isnan(last_ma):
        return float("nan")
    return float((upper.iloc[-1] - lower.iloc[-1]) / abs(last_ma))


def _wick_exhaustion(last: pd.Series, wick_to_body: float = 2.0) -> Tuple[bool, Dict[str, Any]]:
    o = float(last.get("open", 0.0))
    c = float(last.get("close", 0.0))
    h = float(last.get("high", 0.0))
    l = float(last.get("low", 0.0))

    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    denom = max(body, 1e-9)
    upper_ratio = upper_wick / denom
    lower_ratio = lower_wick / denom

    meta = {
        "body": body,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "upper_ratio": upper_ratio,
        "lower_ratio": lower_ratio,
    }
    return (upper_ratio >= wick_to_body), meta


class BoomSpikeTrendStrategy(Strategy):
    name = "BOOM_SPIKE_TREND"
    allowed_trends = None
    allowed_vols = None

    def __init__(
        self,
        tf_m5: Optional[int] = None,
        tf_m15: Optional[int] = None,
        tf_h1: Optional[int] = None,
        tf_h4: Optional[int] = None,
        bb_width_thresh: float = 0.012,
        atr_compress_ratio: float = 0.75,
        wick_to_body: float = 2.0,
        impulse_pct: float = 0.0006,
        impulse_atr_mult: float = 0.75,
        use_h1_filter: bool = True,
        use_h1_sr_filter: bool = False,
        h1_sr_buffer_atr_mult: float = 0.75,
        allow_neutral_h1: bool = True,
    ):
        self.tf_m5 = tf_m5
        self.tf_m15 = tf_m15
        self.tf_h1 = tf_h1
        self.tf_h4 = tf_h4

        self.bb_width_thresh = float(bb_width_thresh)
        self.atr_compress_ratio = float(atr_compress_ratio)
        self.wick_to_body = float(wick_to_body)
        self.impulse_pct = float(impulse_pct)
        self.impulse_atr_mult = float(impulse_atr_mult)

        self.use_h1_filter = bool(use_h1_filter)
        self.use_h1_sr_filter = bool(use_h1_sr_filter)
        self.h1_sr_buffer_atr_mult = float(h1_sr_buffer_atr_mult)
        self.allow_neutral_h1 = bool(allow_neutral_h1)

    def _pick_tf(self, data_by_tf: dict, tf: Optional[int]) -> Optional[pd.DataFrame]:
        if tf is not None and tf in data_by_tf:
            return data_by_tf[tf]
        return None

    def _infer(self, data_by_tf: dict) -> Dict[str, pd.DataFrame]:
        items = [(k, v) for k, v in (data_by_tf or {}).items() if isinstance(v, pd.DataFrame) and not v.empty]
        if not items:
            return {}

        def score(item):
            _, df = item
            if "dt" in df.columns and len(df) >= 3:
                try:
                    dts = pd.to_datetime(df["dt"])
                    med = dts.diff().dropna().dt.total_seconds().median()
                    return float(med) if pd.notna(med) else float("inf")
                except Exception:
                    return float("inf")
            return float("inf")

        items_scored = sorted(items, key=score)
        out = {}
        labels = ["m5", "m15", "h1", "h4"]
        for i, (_, df) in enumerate(items_scored[:4]):
            out[labels[i]] = df
        return out

    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]):
        if not data_by_tf:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {"reason": "no_data"})

        m5 = self._pick_tf(data_by_tf, self.tf_m5)
        m15 = self._pick_tf(data_by_tf, self.tf_m15)
        h1 = self._pick_tf(data_by_tf, self.tf_h1)
        h4 = self._pick_tf(data_by_tf, self.tf_h4)

        if m5 is None or m15 is None or h1 is None or h4 is None:
            inferred = self._infer(data_by_tf)
            m5 = m5 or inferred.get("m5")
            m15 = m15 or inferred.get("m15")
            h1 = h1 or inferred.get("h1")
            h4 = h4 or inferred.get("h4")

        if m5 is None or m15 is None or h1 is None or h4 is None:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {"reason": "missing_timeframes"})

        if len(m5) < 30 or len(m15) < 60 or len(h1) < 60 or len(h4) < 60:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {"reason": "insufficient_bars"})

        atr_col_m15 = None
        for c in ("ATR", "ATR14", "atr", "atr14"):
            if c in m15.columns:
                atr_col_m15 = c
                break

        if h1 is not None and not h1.empty:
            try:
                m15 = add_h1_context_to_df(
                    m15,
                    h1,
                    atr_col=atr_col_m15,
                    sr_atr_buffer_mult=self.h1_sr_buffer_atr_mult,
                )
            except Exception:
                pass

        h4_close = h4["close"].astype(float)
        h4_ema50 = h4_close.ewm(span=50, adjust=False).mean().iloc[-1]
        h4_last = float(h4_close.iloc[-1])
        trend_up_h4 = h4_last >= float(h4_ema50)
        trend_down_h4 = h4_last <= float(h4_ema50)

        last_m15 = m15.iloc[-1]
        h1_trend = str(last_m15.get("h1_trend", "neutral")).lower()
        near_h1_support = bool(last_m15.get("near_h1_support", False))
        near_h1_resistance = bool(last_m15.get("near_h1_resistance", False))

        bb_w = _bb_width(m15)
        atr14 = _atr(m15, 14)
        atr50 = _atr(m15, 50)
        atr_compress = (
            (not np.isnan(atr14))
            and (not np.isnan(atr50))
            and atr50 > 0
            and atr14 <= atr50 * self.atr_compress_ratio
        )
        bb_compress = (not np.isnan(bb_w)) and bb_w <= self.bb_width_thresh

        ex_ok, ex_meta = _wick_exhaustion(last_m15, wick_to_body=self.wick_to_body)

        # rsi = float(last_m15.get("RSI", 50.0))
        # rsi_ok_for_sell = rsi >= 55.0
        # rsi_ok_for_buy = rsi <= 55.0

        rsi = float(last_m15.get("RSI", 50.0))

        # reversal / exhaustion sell gate
        rsi_ok_for_spike_sell = rsi >= 55.0

        # continuation gates
        rsi_ok_for_trend_buy = rsi <= 60.0
        rsi_ok_for_trend_sell = rsi >= 40.0

        compression = bb_compress and atr_compress

        m5_close = m5["close"].astype(float)
        last = float(m5_close.iloc[-1])
        prev = float(m5_close.iloc[-2])

        impulse_down_pct = (prev != 0) and ((prev - last) / abs(prev) >= self.impulse_pct)
        impulse_up_pct = (prev != 0) and ((last - prev) / abs(prev) >= self.impulse_pct)

        atr_m5 = _atr(m5, 14)
        impulse_down_atr = (not np.isnan(atr_m5)) and ((prev - last) >= self.impulse_atr_mult * float(atr_m5))
        impulse_up_atr = (not np.isnan(atr_m5)) and ((last - prev) >= self.impulse_atr_mult * float(atr_m5))

        impulse_down = impulse_down_pct or impulse_down_atr
        impulse_up = impulse_up_pct or impulse_up_atr

        meta = {
            "trend_up_h4": bool(trend_up_h4),
            "h1_trend": h1_trend,
            "h1_support": last_m15.get("h1_support"),
            "h1_resistance": last_m15.get("h1_resistance"),
            "dist_to_h1_support": last_m15.get("dist_to_h1_support"),
            "dist_to_h1_resistance": last_m15.get("dist_to_h1_resistance"),
            "near_h1_support": near_h1_support,
            "near_h1_resistance": near_h1_resistance,
            "bb_width_m15": float(bb_w) if not np.isnan(bb_w) else None,
            "atr14_m15": float(atr14) if not np.isnan(atr14) else None,
            "atr50_m15": float(atr50) if not np.isnan(atr50) else None,
            "compression": bool(compression),
            "exhaustion": bool(ex_ok),
            "rsi_m15": float(rsi),
            "impulse_down_m5": bool(impulse_down),
            "impulse_up_m5": bool(impulse_up),
            **ex_meta,
        }

        if compression and ex_ok and impulse_down and rsi_ok_for_spike_sell:
            if self.use_h1_filter:
                sell_allowed = (h1_trend == "bearish") or (self.allow_neutral_h1 and h1_trend == "neutral")
                if not sell_allowed:
                    return StrategyResult(self.name, Signal.HOLD, 0.0, {**meta, "reason": "blocked_by_h1_trend"})
                if self.use_h1_sr_filter and near_h1_support:
                    return StrategyResult(self.name, Signal.HOLD, 0.0, {**meta, "reason": "too_close_to_h1_support"})

            tightness = 1.0
            if meta["bb_width_m15"]:
                tightness = max(0.0, min(1.0, (self.bb_width_thresh / max(meta["bb_width_m15"], 1e-9)) / 2.0))
            conf = float(max(0.55, min(0.95, 0.65 + 0.3 * tightness)))
            if h1_trend == "bearish":
                conf = min(conf + 0.03, 0.95)
            meta["mode"] = "spike_sell"
            return StrategyResult(self.name, Signal.SELL, conf, meta)

        if trend_up_h4 and (not ex_ok) and impulse_up and rsi_ok_for_trend_buy:
            if self.use_h1_filter:
                buy_allowed = (h1_trend == "bullish") or (self.allow_neutral_h1 and h1_trend == "neutral")
                if not buy_allowed:
                    return StrategyResult(self.name, Signal.HOLD, 0.0, {**meta, "reason": "blocked_by_h1_trend"})
                if self.use_h1_sr_filter and near_h1_resistance:
                    return StrategyResult(self.name, Signal.HOLD, 0.0, {**meta, "reason": "too_close_to_h1_resistance"})

            conf = 0.55
            if h1_trend == "bullish":
                conf = min(conf + 0.03, 0.95)
            meta["mode"] = "trend_buy"
            return StrategyResult(self.name, Signal.BUY, conf, meta)
 
        if trend_down_h4 and (not ex_ok) and impulse_down and rsi_ok_for_trend_sell:
            if self.use_h1_filter:
                sell_allowed = (h1_trend == "bearish") or (self.allow_neutral_h1 and h1_trend == "neutral")
                if not sell_allowed:
                    return StrategyResult(self.name, Signal.HOLD, 0.0, {**meta, "reason": "blocked_by_h1_trend"})
                if self.use_h1_sr_filter and near_h1_support:
                    return StrategyResult(self.name, Signal.HOLD, 0.0, {**meta, "reason": "too_close_to_h1_support"})

            conf = 0.55
            if h1_trend == "bearish":
                conf = min(conf + 0.03, 0.95)

            meta["mode"] = "trend_sell"
            return StrategyResult(self.name, Signal.SELL, conf, meta)            

        return StrategyResult(self.name, Signal.HOLD, 0.0, {**meta, "reason": "no_setup"})
