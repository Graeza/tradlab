from __future__ import annotations
import os
import MetaTrader5 as mt5
import keyring as kr


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)

# --- Trading universe ---
SYMBOL_LIST = [
    "Boom 1000 Index",
    "Boom 900 Index",
    "Boom 500 Index",
    "Boom 600 Index",
    "Boom 300 Index",
    "Wall Street 30",
    "XAUUSD",
]

# Symbols added outside the Boom synthetic universe use a dedicated daily
# RSI(3)/RSI-MA(3) strategy instead of the Boom-focused strategy stack.
NEW_SYMBOL_STRATEGY_SYMBOLS = [
    "Wall Street 30",
    "XAUUSD",
]

TIMEFRAME_LIST = [
    mt5.TIMEFRAME_M5,
    mt5.TIMEFRAME_M15,
    mt5.TIMEFRAME_M30,
    mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_H4,
    mt5.TIMEFRAME_D1,
    mt5.TIMEFRAME_W1,
]

PRIMARY_TIMEFRAME = mt5.TIMEFRAME_M5

# --- Looping ---
LOOP_SLEEP_SECONDS = 300

# --- DB ---
DB_PATH = _project_path("market_data.db")

# --- Feature versioning ---
# Bump this integer whenever you change the feature engineering pipeline (add/remove/rename columns).
FEATURE_SET_VERSION = 1

# --- ML ---
USE_ML_STRATEGY = True
ML_MODEL_PATH = _project_path("models", "ml_strategy.joblib")
ML_CANDIDATES_DIR = _project_path("models", "candidates")
ML_REQUIRE_SYMBOL_MODEL = True
ML_MIN_CANDIDATE_ACCURACY = 0.55

# --- ML Experiment tracking ---
# Training script appends one JSON line per run.
EXPERIMENT_LOG_PATH = _project_path("ml", "experiments", "experiments.jsonl")

# --- Backtesting ---
# Default behavior: decide at bar close, fill at next bar open.
BACKTEST_STARTING_CASH = 10_000.0
BACKTEST_WARMUP_BARS = 200
BACKTEST_OUT_DIR = _project_path("backtests", "latest")
DATA_QUALITY_OUT_DIR = _project_path("backtests", "data_quality")

# --- Ensemble ---
ENSEMBLE_MIN_CONF = 0.55
ENSEMBLE_MIN_VOTE_GAP = 0.10
STRATEGY_WEIGHTS = {
    "RSI_EMA": 1.0,
    "BREAKOUT": 1.0,
    "ML": 1.2,
    "BOOM_SPIKE_TREND": 1.3,
    "BOOM_SELL_DECAY": 1.45,
    "RSI3_MA_EXTREME": 1.0,
}

# --- Labeling ---
# horizon in bars for PRIMARY_TIMEFRAME (e.g. 12 bars on M5 = 60 minutes)
LABEL_HORIZON_BARS = 12

# --- Safety ---
ALLOW_NEW_TRADES_DEFAULT = True   # GUI can override at runtime

# --- MT5 Login / Account Profiles ---
# Add credentials directly in config if preferred:
# ACCOUNT_CREDENTIALS = {
#   "DEMO": {"login": 12345678, "server": "Broker-Demo", "password": "demo-pass"},
#   "LIVE": {"login": 87654321, "server": "Broker-Live", "password": "live-pass"},
# }
#
# Safety defaults:
# - ACTIVE_ACCOUNT_PROFILE is DEMO by default.
# - LIVE profile is blocked unless ALLOW_LIVE_TRADING is True.
ACTIVE_ACCOUNT_PROFILE = os.getenv("MT5_ACCOUNT_PROFILE", "DEMO").strip().upper()
ALLOW_LIVE_TRADING = os.getenv("MT5_ALLOW_LIVE", "0").strip().lower() in {"1", "true", "yes", "on"}


def is_live_trading_allowed() -> bool:
    """Return current LIVE-trading gate, allowing runtime override from env."""
    return os.getenv("MT5_ALLOW_LIVE", "1" if ALLOW_LIVE_TRADING else "0").strip().lower() in {"1", "true", "yes", "on"}


def set_live_trading_allowed(enabled: bool) -> None:
    """Persist runtime LIVE-trading toggle for current process."""
    os.environ["MT5_ALLOW_LIVE"] = "1" if enabled else "0"


ACCOUNT_CREDENTIALS = {
    "DEMO": {
        "login": int(kr.get_password("mt5", "login")),
        "server": kr.get_password("mt5", "server"),
        "password": kr.get_password("mt5", "password"),
    },
    "LIVE": {
        "login": int(kr.get_password("mt5_live", "login")) if kr.get_password("mt5_live", "login") else None,
        "server": kr.get_password("mt5_live", "server"),
        "password": kr.get_password("mt5_live", "password"),
    },
}


def get_mt5_credentials(profile: str | None = None) -> tuple[int, str, str, str]:
    resolved_profile = (profile or ACTIVE_ACCOUNT_PROFILE).strip().upper()
    if resolved_profile not in ACCOUNT_CREDENTIALS:
        raise ValueError(f"Unknown MT5 profile '{resolved_profile}'. Valid profiles: {list(ACCOUNT_CREDENTIALS)}")
    if resolved_profile == "LIVE" and not is_live_trading_allowed():
        raise ValueError(
            "LIVE profile is blocked by default for safety. Set ALLOW_LIVE_TRADING=True (or MT5_ALLOW_LIVE=1) to enable."
        )

    creds = ACCOUNT_CREDENTIALS[resolved_profile]
    login = creds.get("login")
    server = creds.get("server")
    password = creds.get("password")

    if not login or not server or not password:
        raise ValueError(f"Incomplete credentials for MT5 profile '{resolved_profile}'.")

    return int(login), str(server), str(password), resolved_profile

# --- Regime gating multipliers (for ensemble weighting adjustments) ---
REGIME_WEIGHT_MULTIPLIERS = {
    # Trend vs Range
    "TREND": {
        "RSI_EMA": 0.7,
        "BREAKOUT": 1.3,
        "ML": 1.0,
        "BOOM_SPIKE_TREND": 1.05,
        "BOOM_SELL_DECAY": 1.10,
    },
    "RANGE": {
        "RSI_EMA": 1.3,
        "BREAKOUT": 0.7,
        "ML": 1.0,
        "BOOM_SPIKE_TREND": 1.10,
        "BOOM_SELL_DECAY": 0.90,
    },

    # Volatility
    "HIGH_VOL": {
        "RSI_EMA": 0.9,
        "BREAKOUT": 1.1,
        "ML": 1.0,
        "BOOM_SPIKE_TREND": 0.85,
        "BOOM_SELL_DECAY": 1.15,
    },
    "LOW_VOL": {
        "RSI_EMA": 1.1,
        "BREAKOUT": 0.9,
        "ML": 1.0,
        "BOOM_SPIKE_TREND": 1.25,
        "BOOM_SELL_DECAY": 1.05,
    },
}
