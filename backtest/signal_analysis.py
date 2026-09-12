from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


SIGNAL_RESULT_BASE_COLUMNS = [
    "time_s", "symbol", "strategy", "signal", "confidence",
    "regime_trend", "regime_vol", "signal_close", "next_open", "point_size",
]


def _positive_ints(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values if int(value) > 0}))


def infer_bar_seconds(df: pd.DataFrame) -> int:
    if df is None or len(df) < 2:
        raise ValueError("At least two bars are required to infer timeframe duration")
    differences = pd.Series(df["time"]).astype(int).diff().dropna()
    differences = differences[differences > 0]
    if differences.empty:
        raise ValueError("Cannot infer timeframe duration from duplicate timestamps")
    return int(differences.median())


def slice_closed_bars(
    df: pd.DataFrame,
    decision_time_s: int,
    timeframe_seconds: int,
    time_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return only candles whose closing time is at/before the decision time."""
    if df is None or df.empty:
        return pd.DataFrame()
    if time_values is None:
        time_values = df["time"].to_numpy(copy=False)
    latest_closed_open_time = int(decision_time_s) - int(timeframe_seconds)
    end_idx = int(np.searchsorted(time_values, latest_closed_open_time, side="right"))
    return df.iloc[:end_idx]


def score_signals(
    strategy_outputs: pd.DataFrame,
    primary_bars: pd.DataFrame,
    *,
    point_size: float,
    horizons: Iterable[int] = (1, 3, 6, 12),
    targets: Iterable[int] = (50, 100, 250, 500),
) -> pd.DataFrame:
    """Score every strategy and final-ensemble signal in directional points.

    Signal-close results measure prediction quality. Next-open results measure the
    points available from the first realistically tradable bar. MFE/MAE use the
    largest requested horizon and are also measured from the next open.
    """
    horizons = _positive_ints(horizons)
    targets = _positive_ints(targets)
    point_size = float(point_size)
    if point_size <= 0:
        raise ValueError("point_size must be greater than zero")
    if strategy_outputs is None or strategy_outputs.empty or primary_bars is None or primary_bars.empty:
        return pd.DataFrame(columns=SIGNAL_RESULT_BASE_COLUMNS)

    bars = primary_bars.reset_index(drop=True)
    bar_index_by_time = {int(value): index for index, value in enumerate(bars["time"])}
    # There is one strategy-output row per strategy, so deduplicate the repeated
    # final ensemble signal at each timestamp.
    seen: set[tuple[int, str, str, str]] = set()
    results: list[dict] = []
    max_horizon = max(horizons, default=1)
    columns = list(strategy_outputs.columns)
    for values in strategy_outputs.itertuples(index=False, name=None):
        raw_output = dict(zip(columns, values))
        candidates = (
            {
                **raw_output,
                "strategy": str(raw_output.get("strategy") or "UNKNOWN"),
                "signal": str(raw_output.get("signal") or "HOLD").upper(),
            },
            {
                **raw_output,
                "strategy": "FINAL_ENSEMBLE",
                "signal": str(raw_output.get("final_signal") or "HOLD").upper(),
                "confidence": float(raw_output.get("final_confidence") or 0.0),
            },
        )
        for candidate in candidates:
            signal = candidate["signal"]
            if signal not in ("BUY", "SELL"):
                continue
            key = (
                int(candidate["time_s"]), str(candidate.get("symbol") or ""),
                str(candidate["strategy"]), signal,
            )
            if key in seen:
                continue
            seen.add(key)

            signal_index = bar_index_by_time.get(key[0])
            entry_index = None if signal_index is None else signal_index + 1
            if entry_index is None or entry_index >= len(bars):
                continue

            direction = 1.0 if signal == "BUY" else -1.0
            signal_close = float(bars.iloc[signal_index]["close"])
            next_open = float(bars.iloc[entry_index]["open"])
            row = {
                column: candidate.get(column) for column in SIGNAL_RESULT_BASE_COLUMNS
                if column not in ("signal_close", "next_open", "point_size")
            }
            row.update({
                "signal_close": signal_close,
                "next_open": next_open,
                "point_size": point_size,
            })

            for horizon in horizons:
                future_index = signal_index + horizon
                if future_index >= len(bars):
                    row[f"signal_close_points_{horizon}"] = None
                    row[f"next_open_points_{horizon}"] = None
                    row[f"correct_{horizon}"] = None
                    continue
                future_close = float(bars.iloc[future_index]["close"])
                close_points = direction * (future_close - signal_close) / point_size
                open_points = direction * (future_close - next_open) / point_size
                row[f"signal_close_points_{horizon}"] = close_points
                row[f"next_open_points_{horizon}"] = open_points
                row[f"correct_{horizon}"] = bool(open_points > 0)

            window_end = min(len(bars), entry_index + max_horizon)
            window = bars.iloc[entry_index:window_end]
            if signal == "BUY":
                favorable = (window["high"].astype(float) - next_open) / point_size
                adverse = (next_open - window["low"].astype(float)) / point_size
            else:
                favorable = (next_open - window["low"].astype(float)) / point_size
                adverse = (window["high"].astype(float) - next_open) / point_size
            row["mfe_points"] = max(0.0, float(favorable.max()))
            row["mae_points"] = max(0.0, float(adverse.max()))
            row["bars_to_mfe"] = int(favorable.to_numpy().argmax()) + 1
            row["bars_to_mae"] = int(adverse.to_numpy().argmax()) + 1
            for target in targets:
                reached = favorable[favorable >= float(target)]
                row[f"hit_plus_{target}"] = not reached.empty
                row[f"bars_to_plus_{target}"] = (
                    int(window.index.get_loc(reached.index[0])) + 1 if not reached.empty else None
                )
            results.append(row)

    return pd.DataFrame(results)


def summarize_signals(signal_results: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Create a compact strategy/symbol/side summary for quick comparisons."""
    horizons = _positive_ints(horizons)
    if signal_results is None or signal_results.empty:
        return pd.DataFrame(columns=["symbol", "strategy", "signal", "signals"])

    rows: list[dict] = []
    grouped = signal_results.groupby(["symbol", "strategy", "signal"], dropna=False, sort=True)
    for (symbol, strategy, signal), group in grouped:
        row = {
            "symbol": symbol,
            "strategy": strategy,
            "signal": signal,
            "signals": int(len(group)),
            "avg_mfe_points": float(group["mfe_points"].mean()),
            "median_mfe_points": float(group["mfe_points"].median()),
            "avg_mae_points": float(group["mae_points"].mean()),
            "median_mae_points": float(group["mae_points"].median()),
        }
        for horizon in horizons:
            points = pd.to_numeric(group[f"next_open_points_{horizon}"], errors="coerce").dropna()
            row[f"scored_{horizon}"] = int(len(points))
            row[f"correct_pct_{horizon}"] = float((points > 0).mean() * 100.0) if len(points) else None
            row[f"avg_points_{horizon}"] = float(points.mean()) if len(points) else None
            row[f"median_points_{horizon}"] = float(points.median()) if len(points) else None
        for column in group.columns:
            if column.startswith("hit_plus_"):
                row[f"{column}_pct"] = float(group[column].astype(bool).mean() * 100.0)
        rows.append(row)
    return pd.DataFrame(rows)
