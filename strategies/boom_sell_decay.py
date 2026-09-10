from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal
from utils.indicators import add_h1_context_to_df


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty or len(df) < period + 2:
        return float("nan")
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v)


def _bb_width(df: pd.DataFrame, period: int = 20, stdev: float = 2.0) -> float:
    if df is None or df.empty or len(df) < period + 2:
        return float("nan")
    close = df["close"].astype(float)
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + stdev * sd
    lower = ma - stdev * sd
    last_ma = float(ma.iloc[-1])
    if last_ma == 0.0 or np.isnan(last_ma):
        return float("nan")
    return float((upper.iloc[-1] - lower.iloc[-1]) / abs(last_ma))


def _wick_metrics(last: pd.Series) -> Dict[str, float]:
    o = float(last.get("open", 0.0))
    c = float(last.get("close", 0.0))
    h = float(last.get("high", 0.0))
    l = float(last.get("low", 0.0))

    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    denom = max(body, 1e-9)

    return {
        "body": float(body),
        "upper_wick": float(max(upper_wick, 0.0)),
        "lower_wick": float(max(lower_wick, 0.0)),
        "upper_ratio": float(max(upper_wick, 0.0) / denom),
        "lower_ratio": float(max(lower_wick, 0.0) / denom),
    }


class BoomSellDecayStrategy(Strategy):
    name = "BOOM_SELL_DECAY"
    allowed_trends = None
    allowed_vols = None

    def __init__(
        self,
        tf_m5: Optional[int] = None,
        tf_m15: Optional[int] = None,
        tf_h1: Optional[int] = None,
        tf_h4: Optional[int] = None,
        wick_to_body_min: float = 2.2,
        min_close_off_high_atr: float = 0.20,
        m15_rsi_min: float = 58.0,
        m15_ema_span: int = 20,
        require_close_below_ema: bool = True,
        require_prior_push_above_ema_atr: float = 0.35,
        impulse_pct: float = 0.0007,
        impulse_atr_mult: float = 0.85,
        bb_width_max: float = 0.035,
        use_h1_filter: bool = True,
        allow_neutral_h1: bool = True,
        use_h1_sr_filter: bool = True,
        h1_sr_buffer_atr_mult: float = 0.75,
        use_h4_filter: bool = False,
    ):
        self.tf_m5 = tf_m5
        self.tf_m15 = tf_m15
        self.tf_h1 = tf_h1
        self.tf_h4 = tf_h4

        self.wick_to_body_min = float(wick_to_body_min)
        self.min_close_off_high_atr = float(min_close_off_high_atr)
        self.m15_rsi_min = float(m15_rsi_min)
        self.m15_ema_span = int(m15_ema_span)
        self.require_close_below_ema = bool(require_close_below_ema)
        self.require_prior_push_above_ema_atr = float(require_prior_push_above_ema_atr)

        self.impulse_pct = float(impulse_pct)
        self.impulse_atr_mult = float(impulse_atr_mult)
        self.bb_width_max = float(bb_width_max)

        self.use_h1_filter = bool(use_h1_filter)
        self.allow_neutral_h1 = bool(allow_neutral_h1)
        self.use_h1_sr_filter = bool(use_h1_sr_filter)
        self.h1_sr_buffer_atr_mult = float(h1_sr_buffer_atr_mult)
        self.use_h4_filter = bool(use_h4_filter)

    def _pick_tf(self, data_by_tf: dict, tf: Optional[int]) -> Optional[pd.DataFrame]:
        if tf is not None and tf in data_by_tf:
            return data_by_tf[tf]
        return None

    def _infer(self, data_by_tf: dict) -> Dict[str, pd.DataFrame]:
        items = [
            (k, v)
            for k, v in (data_by_tf or {}).items()
            if isinstance(v, pd.DataFrame) and not v.empty
        ]
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

    @staticmethod
    def _find_rsi(last: pd.Series) -> float:
        for c in ("RSI", "rsi"):
            if c in last.index:
                return float(last.get(c, 50.0))
        return 50.0

    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]):
        if not data_by_tf:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {"reason": "no_data"})

        m5 = self._pick_tf(data_by_tf, self.tf_m5)
        m15 = self._pick_tf(data_by_tf, self.tf_m15)
        h1 = self._pick_tf(data_by_tf, self.tf_h1)
        h4 = self._pick_tf(data_by_tf, self.tf_h4)

        if m5 is None or m15 is None or h1 is None:
            inferred = self._infer(data_by_tf)
            m5 = m5 or inferred.get("m5")
            m15 = m15 or inferred.get("m15")
            h1 = h1 or inferred.get("h1")
            h4 = h4 or inferred.get("h4")

        if m5 is None or m15 is None or h1 is None:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {"reason": "missing_timeframes"})

        if len(m5) < 30 or len(m15) < 60 or len(h1) < 60:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {"reason": "insufficient_bars"})

        atr_col_m15 = None
        for c in ("ATR", "ATR14", "atr", "atr14"):
            if c in m15.columns:
                atr_col_m15 = c
                break

        try:
            m15 = add_h1_context_to_df(
                m15,
                h1,
                atr_col=atr_col_m15,
                sr_atr_buffer_mult=self.h1_sr_buffer_atr_mult,
            )
        except Exception:
            pass

        last_m15 = m15.iloc[-1]
        prev_m15 = m15.iloc[-2]

        close_m15 = m15["close"].astype(float)
        ema_m15 = close_m15.ewm(span=self.m15_ema_span, adjust=False).mean()

        last_close = float(last_m15["close"])
        last_open = float(last_m15["open"])
        last_high = float(last_m15["high"])
        last_low = float(last_m15["low"])
        last_ema = float(ema_m15.iloc[-1])

        prev_close = float(prev_m15["close"])
        prev_ema = float(ema_m15.iloc[-2])

        atr_m15 = _atr(m15, 14)
        atr_m5 = _atr(m5, 14)
        bb_w_m15 = _bb_width(m15)

        wick = _wick_metrics(last_m15)
        upper_exhaustion = wick["upper_ratio"] >= self.wick_to_body_min

        close_off_high = 0.0
        if not np.isnan(atr_m15) and atr_m15 > 0:
            close_off_high = (last_high - last_close) / atr_m15

        rsi_m15 = self._find_rsi(last_m15)

        close_below_ema = last_close < last_ema
        prior_push_above_ema = False
        if not np.isnan(atr_m15) and atr_m15 > 0:
            recent_push = max(last_high - last_ema, prev_m15["high"] - prev_ema)
            prior_push_above_ema = float(recent_push) >= self.require_prior_push_above_ema_atr * atr_m15

        m5_close = m5["close"].astype(float)
        m5_last = float(m5_close.iloc[-1])
        m5_prev = float(m5_close.iloc[-2])

        impulse_down_pct = (m5_prev != 0.0) and ((m5_prev - m5_last) / abs(m5_prev) >= self.impulse_pct)
        impulse_down_atr = (not np.isnan(atr_m5)) and ((m5_prev - m5_last) >= self.impulse_atr_mult * float(atr_m5))
        impulse_down = bool(impulse_down_pct or impulse_down_atr)

        h1_trend = str(last_m15.get("h1_trend", "neutral")).lower()
        near_h1_support = bool(last_m15.get("near_h1_support", False))
        near_h1_resistance = bool(last_m15.get("near_h1_resistance", False))

        h4_ok = True
        h4_last_close = None
        h4_ema50 = None
        if self.use_h4_filter and h4 is not None and not h4.empty and len(h4) >= 60:
            h4_close = h4["close"].astype(float)
            h4_last_close = float(h4_close.iloc[-1])
            h4_ema50 = float(h4_close.ewm(span=50, adjust=False).mean().iloc[-1])
            h4_ok = h4_last_close <= h4_ema50

        bb_ok = np.isnan(bb_w_m15) or (bb_w_m15 <= self.bb_width_max)
        close_off_high_ok = close_off_high >= self.min_close_off_high_atr
        rsi_ok = rsi_m15 >= self.m15_rsi_min

        ema_gate_ok = True
        if self.require_close_below_ema:
            ema_gate_ok = close_below_ema

        setup_ok = (
            upper_exhaustion
            and close_off_high_ok
            and rsi_ok
            and ema_gate_ok
            and prior_push_above_ema
            and impulse_down
            and bb_ok
            and h4_ok
        )

        meta = {
            "upper_ratio": wick["upper_ratio"],
            "lower_ratio": wick["lower_ratio"],
            "body": wick["body"],
            "upper_wick": wick["upper_wick"],
            "lower_wick": wick["lower_wick"],
            "close_off_high_atr": float(close_off_high),
            "rsi_m15": float(rsi_m15),
            "m15_close": float(last_close),
            "m15_open": float(last_open),
            "m15_high": float(last_high),
            "m15_low": float(last_low),
            "m15_ema": float(last_ema),
            "close_below_ema": bool(close_below_ema),
            "prior_push_above_ema": bool(prior_push_above_ema),
            "impulse_down_m5": bool(impulse_down),
            "atr_m15": None if np.isnan(atr_m15) else float(atr_m15),
            "atr_m5": None if np.isnan(atr_m5) else float(atr_m5),
            "bb_width_m15": None if np.isnan(bb_w_m15) else float(bb_w_m15),
            "h1_trend": h1_trend,
            "near_h1_support": bool(near_h1_support),
            "near_h1_resistance": bool(near_h1_resistance),
            "h4_last_close": h4_last_close,
            "h4_ema50": h4_ema50,
        }

        if not setup_ok:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {**meta, "reason": "no_setup"})

        if self.use_h1_filter:
            sell_allowed = (h1_trend == "bearish") or (self.allow_neutral_h1 and h1_trend == "neutral")
            if not sell_allowed:
                return StrategyResult(self.name, Signal.HOLD, 0.0, {**meta, "reason": "blocked_by_h1_trend"})

        if self.use_h1_sr_filter and near_h1_support:
            return StrategyResult(self.name, Signal.HOLD, 0.0, {**meta, "reason": "too_close_to_h1_support"})

        conf = 0.62

        conf += min(0.10, max(0.0, (wick["upper_ratio"] - self.wick_to_body_min) * 0.03))
        conf += min(0.08, max(0.0, (rsi_m15 - self.m15_rsi_min) * 0.004))
        conf += 0.05 if near_h1_resistance else 0.0
        conf += 0.04 if h1_trend == "bearish" else 0.0
        conf += 0.03 if close_below_ema else 0.0
        conf += 0.03 if impulse_down_atr else 0.0

        conf = float(max(0.55, min(0.95, conf)))

        meta["mode"] = "boom_sell_decay"
        return StrategyResult(self.name, Signal.SELL, conf, meta)
