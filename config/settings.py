from __future__ import annotations
import MetaTrader5 as mt5
import keyring as kr

# --- Trading universe ---
SYMBOL_LIST = [
    # "EURUSD",
    # "GBPUSD",
    "Boom 1000 Index"
]

TIMEFRAME_LIST = [
    mt5.TIMEFRAME_M5,
    mt5.TIMEFRAME_M15,
    # mt5.TIMEFRAME_H1,
]

PRIMARY_TIMEFRAME = mt5.TIMEFRAME_M5

# --- Looping ---
LOOP_SLEEP_SECONDS = 300

# --- DB ---
DB_PATH = "market_data.db"

# --- ML ---
USE_ML_STRATEGY = True
ML_MODEL_PATH = "models/ml_strategy.joblib"

# --- Ensemble ---
ENSEMBLE_MIN_CONF = 0.55
STRATEGY_WEIGHTS = {
    "RSI_EMA": 1.0,
    "BREAKOUT": 1.0,
    "ML": 1.2,
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