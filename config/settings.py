from __future__ import annotations
import MetaTrader5 as mt5
import keyring as kr

# --- Trading universe ---
SYMBOL_LIST = [
    "Boom 1000 Index",
    "Boom 900 Index",
    "Boom 500 Index",
    "Boom 600 Index",
    "Boom 300 Index"
]

TIMEFRAME_LIST = [
    mt5.TIMEFRAME_M5,
    mt5.TIMEFRAME_M15,
    mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_H4,
]

PRIMARY_TIMEFRAME = mt5.TIMEFRAME_M5

# --- Looping ---
LOOP_SLEEP_SECONDS = 300

# --- DB ---
DB_PATH = "market_data.db"

# --- Feature versioning ---
# Bump this integer whenever you change the feature engineering pipeline (add/remove/rename columns).
FEATURE_SET_VERSION = 1

# --- ML ---
USE_ML_STRATEGY = True
ML_MODEL_PATH = "models/ml_strategy.joblib"

# --- ML Experiment tracking ---
# Training script appends one JSON line per run.
EXPERIMENT_LOG_PATH = "ml/experiments/experiments.jsonl"

# --- Backtesting ---
# Default behavior: decide at bar close, fill at next bar open.
BACKTEST_STARTING_CASH = 10_000.0
BACKTEST_WARMUP_BARS = 200
BACKTEST_OUT_DIR = "backtests/latest"

# --- Ensemble ---
ENSEMBLE_MIN_CONF = 0.55
STRATEGY_WEIGHTS = {
    "RSI_EMA": 1.0,
    "BREAKOUT": 1.0,
    "ML": 1.2,
    "BOOM_SPIKE_TREND": 1.3,
}

# --- Labeling ---
# horizon in bars for PRIMARY_TIMEFRAME (e.g. 12 bars on M5 = 60 minutes)
LABEL_HORIZON_BARS = 12

# --- Safety ---
ALLOW_NEW_TRADES_DEFAULT = True   # GUI can override at runtime

# --- MT5 Login ---
login = int(kr.get_password("mt5", "login"))
server = kr.get_password("mt5", "server")
password = kr.get_password("mt5", "password")

# --- Regime gating multipliers (for ensemble weighting adjustments) ---
REGIME_WEIGHT_MULTIPLIERS = {
    # Trend vs Range
    "TREND": {
        "RSI_EMA": 0.7,
        "BREAKOUT": 1.3,
        "ML": 1.0,
        "BOOM_SPIKE_TREND": 1.05,
    },
    "RANGE": {
        "RSI_EMA": 1.3,
        "BREAKOUT": 0.7,
        "ML": 1.0,
        "BOOM_SPIKE_TREND": 1.10,
    },

    # Volatility
    "HIGH_VOL": {
        "RSI_EMA": 0.9,
        "BREAKOUT": 1.1,
        "ML": 1.0,
        "BOOM_SPIKE_TREND": 0.85,
    },
    "LOW_VOL": {
        "RSI_EMA": 1.1,
        "BREAKOUT": 0.9,
        "ML": 1.0,
        "BOOM_SPIKE_TREND": 1.25,
    },
}