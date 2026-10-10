"""Resolve strategy timeframes without importing the MT5 terminal package."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


# MT5 encodes hourly timeframes differently from their duration in minutes.
# Prefer native MT5 keys when both representations are supplied.
_TIMEFRAME_KEYS = {
    5: (5,),
    15: (15,),
    60: (16385, 60),
    240: (16388, 240),
}


def get_timeframe(
    data_by_tf: Mapping[int, pd.DataFrame],
    minutes: int,
    *,
    explicit_key: int | None = None,
) -> pd.DataFrame | None:
    """Select a nonempty frame by its key, honoring an explicit key exactly.

    Missing frames remain missing; dictionary order and observed candle spacing
    must not cause another timeframe to take their place.
    """
    keys = (explicit_key,) if explicit_key is not None else _TIMEFRAME_KEYS[minutes]
    for key in keys:
        frame = data_by_tf.get(key)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame
    return None
