# NOTE: This file is based on your previous version.
# It assumes columns like: open, high, low, close; and for trend uses DateTime.
# If your pipeline uses a different datetime column, adjust analyze_stock_trend accordingly.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import talib as ta


# =========================================================
#  SHARED HELPERS
# =========================================================

def linear_regression_trend(y_values):
    """
    Fit a linear regression to a 1D array of values.
    Returns slope and intercept.
    """
    y = np.array(y_values).reshape(-1, 1)
    x = np.arange(len(y_values)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    return slope, intercept

# =========================================================
#  INDICATORS
# =========================================================

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def calculate_ema(df, column, period=10):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    ema_series = ta.EMA(df[column].values, timeperiod=period)
    df[f"{column}_EMA{period}"] = ema_series

    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    close = df['close'].values

    macd, signal_line, hist = ta.MACD(
        close,
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal
    )

    df['MACD'] = macd
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = hist

    return df

# =========================================================
#  TREND ANALYSIS
# =========================================================

def analyze_stock_trend(df):
    """
    Fit a regression line to closing prices over time.
    Returns slope, intercept, trendline array, and trend label.
    """
    df['Date_Ordinal'] = df['DateTime'].map(pd.Timestamp.toordinal)

    slope, intercept = linear_regression_trend(df['close'].values)
    trendline = slope * np.arange(len(df)) + intercept

    trend = 'Bullish' if slope > 0 else 'Bearish'
    df['Trend'] = trend

    return slope, intercept, trendline, trend

def analyze_positive_candles(df, lookback=50, slope_threshold=0.0):
    """
    Analyze trend of positive candles using regression on opens and highs.
    """
    positive = df[df['close'] > df['open']]

    if len(positive) < lookback:
        df['positive_candles'] = 'Insufficient data'
        return df

    recent = positive.tail(lookback)

    open_slope, _ = linear_regression_trend(recent['open'].values)
    high_slope, _ = linear_regression_trend(recent['high'].values)

    if open_slope > slope_threshold and high_slope > slope_threshold:
        trend = 'Rising'
    elif open_slope < -slope_threshold and high_slope < -slope_threshold:
        trend = 'Falling'
    else:
        trend = 'Sideways'

    df['positive_candles'] = None
    df.at[df.index[-1], 'positive_candles'] = trend

    return df
