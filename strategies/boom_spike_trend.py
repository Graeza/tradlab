from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty or len(df) < period + 2:
        return float("nan")
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
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
    return float((upper.iloc[-1] - lower.iloc[-1]) / last_ma)


def _wick_exhaustion(last: pd.Series, *, wick_to_body: float = 2.0) -> tuple[bool, Dict[str, Any]]:
    o = float(last.get("open", 0.0))
    c = float(last.get("close", 0.0))
    h = float(last.get("high", 0.0))
    l = float(last.get("low", 0.0))

    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    # Avoid division by zero: treat tiny body as "small body"
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
    # Boom spike SELL setups typically show strong rejection up (large upper wick)
    return (upper_ratio >= wick_to_body), meta


class BoomSpikeTrendStrategy(Strategy):
    """Boom 300–1000 multi-timeframe spike + trend strategy.

    Intended usage:
      - Entries on M5 confirmation
      - Setup on M15 compression + exhaustion
      - Context on H1/H4 (trend + volatility regime)

    Notes:
      - We compute ATR/BB width locally (features pipeline doesn't include these yet).
      - We use strategy.name keys for ensemble weighting ("BOOM_SPIKE_TREND").
    """

    name = "BOOM_SPIKE_TREND"
    # Spikes are more likely in low-vol compression; keep active in all trends but prefer LOW_VOL regime via multipliers.
    allowed_trends = None
    allowed_vols = None

    def __init__(
        self,
        *,
        tf_m5: Optional[int] = None,
        tf_m15: Optional[int] = None,
        tf_h1: Optional[int] = None,
        tf_h4: Optional[int] = None,
        bb_width_thresh: float = 0.012,   # normalized band width threshold
        atr_compress_ratio: float = 0.75, # ATR(14) < ATR(50) * ratio
        wick_to_body: float = 2.0,
        impulse_pct: float = 0.0006,      # 0.06% drop on M5 as impulse confirm
    ):
        # If you pass explicit MT5 timeframe ints, we'll use them; otherwise we'll infer by "most recent" keys.
        self.tf_m5 = tf_m5
        self.tf_m15 = tf_m15
        self.tf_h1 = tf_h1
        self.tf_h4 = tf_h4

        self.bb_width_thresh = float(bb_width_thresh)
        self.atr_compress_ratio = float(atr_compress_ratio)
        self.wick_to_body = float(wick_to_body)
        self.impulse_pct = float(impulse_pct)

    def _pick_tf(self, data_by_tf: dict[int, pd.DataFrame], tf: Optional[int]) -> Optional[pd.DataFrame]:
        if tf is not None and tf in data_by_tf:
            return data_by_tf[tf]
        return None

    def _infer(self, data_by_tf: dict[int, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        # Best-effort inference by bar count (higher TF usually fewer bars for same window).
        # We'll sort by median delta between timestamps if possible; fallback to length ordering.
        items = [(k, v) for k, v in (data_by_tf or {}).items() if isinstance(v, pd.DataFrame) and not v.empty]
        if not items:
            return {}

        def score(item):
            k, df = item
            if "dt" in df.columns and len(df) >= 3:
                try:
                    dts = pd.to_datetime(df["dt"])
                    med = (dts.diff().dropna().dt.total_seconds().median())
                    return float(med) if pd.notna(med) else float("inf")
                except Exception:
                    return float("inf")
            return float("inf")

        items_scored = sorted(items, key=score)
        # smallest delta ≈ lowest TF (M5), next ≈ M15, then H1, then H4
        out: Dict[str, pd.DataFrame] = {}
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

        # Live trading: ALWAYS use the last CLOSED candle (exclude forming bar).
        # Many live feeds include a still-forming last bar; using it causes repaint/false signals.
        def _closed(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty or len(df) < 2:
                return df
            return df.iloc[:-1]

        m5 = _closed(m5)
        m15 = _closed(m15)
        h1 = _closed(h1)
        h4 = _closed(h4)


        if m5 is None or m15 is None or h1 is None or h4 is None:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {"reason": "missing_timeframes"})

        if len(m5) < 30 or len(m15) < 60 or len(h1) < 60 or len(h4) < 60:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {"reason": "insufficient_bars"})

        # --- Context: HTF trend (H4 drift) ---
        h4_close = h4["close"].astype(float)
        h4_ema50 = h4_close.ewm(span=50, adjust=False).mean().iloc[-1]
        h4_last = float(h4_close.iloc[-1])
        trend_up = h4_last >= float(h4_ema50)

        # Additional guard: avoid taking SELL fades when H1 is strongly up (common Boom behavior)
        h1_close = h1["close"].astype(float)
        h1_ema50 = h1_close.ewm(span=50, adjust=False).mean().iloc[-1]
        h1_last = float(h1_close.iloc[-1])
        atr_h1 = _atr(h1, 14)
        h1_strong_up = (h1_last >= float(h1_ema50)) and (not np.isnan(atr_h1)) and ((h1_last - float(h1_ema50)) >= 0.5 * float(atr_h1))

        # --- Setup: M15 compression + exhaustion ---
        bb_w = _bb_width(m15)
        atr14 = _atr(m15, 14)
        atr50 = _atr(m15, 50)
        atr_compress = (not np.isnan(atr14) and not np.isnan(atr50) and atr50 > 0 and atr14 <= atr50 * self.atr_compress_ratio)
        bb_compress = (not np.isnan(bb_w) and bb_w <= self.bb_width_thresh)

        ex_ok, ex_meta = _wick_exhaustion(m15.iloc[-1], wick_to_body=self.wick_to_body)

        # Optional RSI filter if present (from features pipeline)
        rsi = float(m15.iloc[-1].get("RSI", 50.0))
        rsi_ok_for_sell = rsi >= 55.0  # mild overbought / stretched up before spike
        rsi_ok_for_buy = rsi <= 55.0

        compression = bb_compress and atr_compress

        # --- Trigger: M5 impulse (confirm breakdown) ---
        m5_close = m5["close"].astype(float)        
        last = float(m5_close.iloc[-1])
        prev = float(m5_close.iloc[-2])

        # Percent impulse (fallback) + ATR-normalized impulse (preferred)
        impulse_down_pct = (prev > 0) and ((prev - last) / prev >= self.impulse_pct)
        impulse_up_pct = (prev > 0) and ((last - prev) / prev >= self.impulse_pct)

        # Backward-safe default if attribute missing for any reason
        atr_mult = float(getattr(self, "impulse_atr_mult", 0.75))

        atr_m5 = _atr(m5, 14)
        impulse_down_atr = (not np.isnan(atr_m5)) and ((prev - last) >= atr_mult * float(atr_m5))
        impulse_up_atr = (not np.isnan(atr_m5)) and ((last - prev) >= atr_mult * float(atr_m5))

        impulse_down = impulse_down_pct or impulse_down_atr
        impulse_up = impulse_up_pct or impulse_up_atr

        meta = {
            "trend_up_h4": bool(trend_up),
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

        # --- Decision logic ---
        # Spike SELL: compression + exhaustion + impulse down
        if compression and ex_ok and impulse_down and rsi_ok_for_sell:
            # confidence scales with how tight BB width is
            tightness = 1.0
            if meta["bb_width_m15"]:
                tightness = max(0.0, min(1.0, (self.bb_width_thresh / max(meta["bb_width_m15"], 1e-9)) / 2.0))
            conf = float(max(0.55, min(0.95, 0.65 + 0.3 * tightness)))
            meta["mode"] = "spike_sell"
            return StrategyResult(self.name, Signal.SELL, conf, meta)

        # Trend BUY: HTF uptrend + no exhaustion + impulse up (continuation after pullback)
        if trend_up and (not ex_ok) and impulse_up and rsi_ok_for_buy:
            conf = 0.55
            meta["mode"] = "trend_buy"
            return StrategyResult(self.name, Signal.BUY, conf, meta)

        return StrategyResult(self.name, Signal.HOLD, 0.0, {**meta, "reason": "no_setup"})
