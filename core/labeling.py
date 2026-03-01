from __future__ import annotations
import pandas as pd
import numpy as np

def make_labels_from_bars(bars: pd.DataFrame, symbol: str, timeframe: int, horizon_bars: int) -> pd.DataFrame:
    """Create labels for bars where future close exists.

    y_class: +1 (BUY), -1 (SELL), 0 (flat)
    """

    if bars is None or bars.empty or len(bars) <= horizon_bars:
        return pd.DataFrame(columns=["symbol","timeframe","time","horizon_bars","future_return","y_class"])

    df = bars[["time","close"]].copy()
    df["future_close"] = df["close"].shift(-horizon_bars)
    df = df.dropna(subset=["future_close"]).copy()

    df["future_return"] = (df["future_close"] / df["close"]) - 1.0
    # classification label
    eps = 0.00005
    df["y_class"] = np.where(df["future_return"] > eps, 1, np.where(df["future_return"] < -eps, -1, 0))

    out = pd.DataFrame({
        "symbol": symbol,
        "timeframe": timeframe,
        "time": df["time"].astype(int),
        "horizon_bars": int(horizon_bars),
        "future_return": df["future_return"].astype(float),
        "y_class": df["y_class"].astype(int),
    })
    return out
