from __future__ import annotations

import pandas as pd
import numpy as np
import hashlib

from config.settings import FEATURE_SET_VERSION
from utils.indicators import calculate_rsi, calculate_ema, calculate_macd, analyze_positive_candles
from utils.encoding import encode_signal


def _feature_set_id(cols: list[str]) -> str:
    payload = "\n".join([str(c) for c in cols]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def attach_feature_version(df: pd.DataFrame) -> pd.DataFrame:
    """Attach feature_set_version + feature_set_id columns.

    feature_set_id is a short hash of the dataframe column order (excluding dt).
    """
    cols = [c for c in df.columns if c != "dt"]
    df = df.copy()
    df["feature_set_version"] = int(FEATURE_SET_VERSION)
    df["feature_set_id"] = _feature_set_id(cols)
    return df


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build feature dataframe from raw bars.

    Expects columns: time, open, high, low, close, tick_volume, spread, real_volume
    Returns bars + feature columns (keeps 'time' for joining).
    """

    df = bars.copy()
    df = df.drop_duplicates(subset=["time"]).sort_values("time")
    # Basic returns
    df["ret_1"] = df["close"].pct_change()
    df["log_ret_1"] = np.log(df["close"]).diff()
    # Indicators from your file
    df = calculate_rsi(df, period=14)
    df = calculate_ema(df, "close", period=10)
    df = calculate_ema(df, "close", period=21)
    df = calculate_macd(df)
    df = analyze_positive_candles(df, lookback=50, slope_threshold=0.0)

    # Clean infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Feature versioning metadata
    df = attach_feature_version(df)
    return df
