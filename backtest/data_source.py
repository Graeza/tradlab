from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from core.database import MarketDatabase


@dataclass(frozen=True)
class BacktestData:
    """Container for historical bars by timeframe."""

    symbol: str
    bars_by_tf: Dict[int, pd.DataFrame]


def load_bars_from_db(
    db: MarketDatabase,
    symbol: str,
    timeframes: list[int],
    limit: int = 200_000,
    time_min_s: Optional[int] = None,
    time_max_s: Optional[int] = None,
) -> BacktestData:
    """Load bars from MarketDatabase.

    Notes:
      - `limit` is applied per timeframe.
      - Optional time window filters are applied after load.
    """
    out: Dict[int, pd.DataFrame] = {}
    for tf in timeframes:
        df = db.load_bars(symbol, tf, limit=limit)
        if df is None or df.empty:
            out[tf] = pd.DataFrame()
            continue

        if time_min_s is not None:
            df = df[df["time"] >= int(time_min_s)]
        if time_max_s is not None:
            df = df[df["time"] <= int(time_max_s)]
        out[tf] = df.reset_index(drop=True)

    return BacktestData(symbol=symbol, bars_by_tf=out)


def load_bars_from_csv(
    symbol: str,
    csv_by_tf: Dict[int, str],
    time_min_s: Optional[int] = None,
    time_max_s: Optional[int] = None,
) -> BacktestData:
    """Load bars from CSV files keyed by timeframe.

    CSV must contain at least: time, open, high, low, close
    (and optionally tick_volume, spread, real_volume).
    """
    out: Dict[int, pd.DataFrame] = {}
    for tf, path in csv_by_tf.items():
        df = pd.read_csv(path)
        if "time" not in df.columns:
            raise ValueError(f"CSV for tf={tf} missing required column 'time': {path}")

        # ensure ordering
        df = df.drop_duplicates(subset=["time"]).sort_values("time")

        if time_min_s is not None:
            df = df[df["time"] >= int(time_min_s)]
        if time_max_s is not None:
            df = df[df["time"] <= int(time_max_s)]

        # add dt for parity with DB loads
        if "dt" not in df.columns:
            df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)

        out[int(tf)] = df.reset_index(drop=True)
    return BacktestData(symbol=symbol, bars_by_tf=out)
