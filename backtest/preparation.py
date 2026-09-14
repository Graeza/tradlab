"""Data normalization and causal preparation for backtest simulation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

import numpy as np
import pandas as pd

from backtest.contracts import DuplicatePolicy
from backtest.signal_analysis import infer_bar_seconds
from core.features import build_features
from utils.regime import detect_regime


REQUIRED_BAR_COLUMNS = ("time", "open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class DataQualitySummary:
    row_count: int
    first_time_s: int | None
    last_time_s: int | None
    duplicate_rows: int
    dropped_rows: int
    gap_count: int
    timeframe_seconds: int


@dataclass(frozen=True, slots=True)
class PreparedBacktestContext:
    symbol: str
    primary_tf: int
    timeframes: tuple[int, ...]
    bars_by_tf: Mapping[int, pd.DataFrame]
    features_by_tf: Mapping[int, pd.DataFrame]
    alignment_by_tf: Mapping[int, np.ndarray]
    primary_time_s: np.ndarray
    decision_time_s: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    spreads: np.ndarray
    regimes: tuple[dict[str, object], ...]
    quality_by_tf: Mapping[int, DataQualitySummary]
    requested_warmup_bars: int
    effective_warmup_bars: int
    preparation_timings_s: Mapping[str, float]


def normalize_bars(
    bars: pd.DataFrame,
    *,
    timeframe_seconds: int | None = None,
    duplicate_policy: DuplicatePolicy = "error",
    invalid_policy: str = "error",
) -> tuple[pd.DataFrame, DataQualitySummary]:
    """Normalize bars into a source-independent, strictly ordered representation."""

    if bars is None:
        bars = pd.DataFrame()
    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"missing required bar columns: {', '.join(missing)}")
    frame = bars.copy()
    original_count = len(frame)
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    for column in REQUIRED_BAR_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required_values = frame.loc[:, REQUIRED_BAR_COLUMNS].to_numpy(dtype=float, copy=False)
    valid = np.isfinite(required_values).all(axis=1)
    dropped = int((~valid).sum())
    if dropped and invalid_policy == "error":
        raise ValueError(f"found {dropped} rows with invalid required values")
    if invalid_policy not in ("error", "drop"):
        raise ValueError(f"unsupported invalid_policy: {invalid_policy}")
    frame = frame.loc[valid].copy()
    frame["time"] = frame["time"].astype(np.int64)
    for column in REQUIRED_BAR_COLUMNS[1:]:
        frame[column] = frame[column].astype(np.float64)

    duplicate_count = int(frame.duplicated("time", keep=False).sum())
    if duplicate_count:
        if duplicate_policy == "error":
            raise ValueError(f"found {duplicate_count} rows with duplicate timestamps")
        keep = "first" if duplicate_policy == "keep_first" else "last"
        frame = frame.drop_duplicates("time", keep=keep)
    elif duplicate_policy not in ("error", "keep_first", "keep_latest"):
        raise ValueError(f"unsupported duplicate_policy: {duplicate_policy}")

    frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    if "spread" in frame:
        frame["spread"] = pd.to_numeric(frame["spread"], errors="coerce").astype(float)
        if bool((frame["spread"].dropna() < 0).any()):
            raise ValueError("spread must not be negative")
    duration = int(timeframe_seconds or (infer_bar_seconds(frame) if len(frame) >= 2 else 0))
    if duration <= 0:
        raise ValueError("timeframe duration must be positive")
    times = frame["time"].to_numpy(dtype=np.int64, copy=False)
    gaps = int(np.sum(np.diff(times) > duration)) if len(times) > 1 else 0
    frame["expected_close_time"] = frame["time"] + duration
    if "dt" not in frame:
        frame["dt"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    summary = DataQualitySummary(
        row_count=len(frame),
        first_time_s=int(times[0]) if len(times) else None,
        last_time_s=int(times[-1]) if len(times) else None,
        duplicate_rows=duplicate_count,
        dropped_rows=original_count - len(frame),
        gap_count=gaps,
        timeframe_seconds=duration,
    )
    return frame, summary


def causal_alignment(
    primary_decision_times_s: np.ndarray,
    source_open_times_s: np.ndarray,
    source_timeframe_seconds: int,
) -> np.ndarray:
    """Return the latest source candle closed at each primary decision time."""

    close_times = np.asarray(source_open_times_s, dtype=np.int64) + int(source_timeframe_seconds)
    result = np.searchsorted(close_times, primary_decision_times_s, side="right") - 1
    result = result.astype(np.int64, copy=False)
    result.setflags(write=False)
    return result


def prepare_backtest(
    *,
    symbol: str,
    bars_by_tf: Mapping[int, pd.DataFrame],
    timeframes: tuple[int, ...] | list[int],
    primary_tf: int,
    warmup_bars: int,
    minimum_lookback_bars: int = 1,
    duplicate_policy: DuplicatePolicy = "error",
) -> PreparedBacktestContext:
    """Normalize and precompute every input consumed by the simulation kernel."""

    started = perf_counter()
    normalized: dict[int, pd.DataFrame] = {}
    quality: dict[int, DataQualitySummary] = {}
    for tf in timeframes:
        source = bars_by_tf.get(tf)
        if source is None or source.empty:
            normalized[int(tf)] = pd.DataFrame()
            continue
        normalized[int(tf)], quality[int(tf)] = normalize_bars(source, duplicate_policy=duplicate_policy)
    if primary_tf not in normalized or normalized[primary_tf].empty:
        raise ValueError(f"No primary bars for tf={primary_tf}")
    normalization_done = perf_counter()

    features = {
        tf: build_features(frame).reset_index(drop=True) if not frame.empty else pd.DataFrame()
        for tf, frame in normalized.items()
    }
    features_done = perf_counter()
    primary = normalized[primary_tf]
    primary_duration = quality[primary_tf].timeframe_seconds
    times = _readonly(primary["time"].to_numpy(dtype=np.int64, copy=True))
    decisions = _readonly(times + primary_duration)
    alignment: dict[int, np.ndarray] = {}
    for tf, frame in normalized.items():
        if frame.empty:
            alignment[tf] = _readonly(np.full(len(primary), -1, dtype=np.int64))
        else:
            alignment[tf] = causal_alignment(
                decisions,
                frame["time"].to_numpy(dtype=np.int64, copy=False),
                quality[tf].timeframe_seconds,
            )
    alignment_done = perf_counter()

    primary_features = features[primary_tf]
    regimes = tuple(
        detect_regime(primary_features.iloc[: int(alignment[primary_tf][i]) + 1])
        if alignment[primary_tf][i] >= 0 else {"trend": "UNKNOWN", "vol": "UNKNOWN"}
        for i in range(len(primary))
    )
    finished = perf_counter()
    spreads = (
        primary["spread"].to_numpy(dtype=float, na_value=np.nan, copy=True)
        if "spread" in primary else np.full(len(primary), np.nan, dtype=float)
    )
    effective_warmup = max(int(warmup_bars), int(minimum_lookback_bars))
    return PreparedBacktestContext(
        symbol=symbol,
        primary_tf=int(primary_tf),
        timeframes=tuple(int(tf) for tf in timeframes),
        bars_by_tf=normalized,
        features_by_tf=features,
        alignment_by_tf=alignment,
        primary_time_s=times,
        decision_time_s=decisions,
        opens=_readonly(primary["open"].to_numpy(dtype=float, copy=True)),
        highs=_readonly(primary["high"].to_numpy(dtype=float, copy=True)),
        lows=_readonly(primary["low"].to_numpy(dtype=float, copy=True)),
        closes=_readonly(primary["close"].to_numpy(dtype=float, copy=True)),
        spreads=_readonly(spreads),
        regimes=regimes,
        quality_by_tf=quality,
        requested_warmup_bars=int(warmup_bars),
        effective_warmup_bars=effective_warmup,
        preparation_timings_s={
            "normalization": normalization_done - started,
            "features": features_done - normalization_done,
            "alignment": alignment_done - features_done,
            "regimes": finished - alignment_done,
            "total": finished - started,
        },
    )


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array
