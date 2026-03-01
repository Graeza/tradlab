from __future__ import annotations

import time
import joblib
import os

from core.mt5_init import initialize_mt5
from core.data_fetcher import DataFetcher
from core.database import MarketDatabase
from core.data_pipeline import DataPipeline
from core.ensemble import EnsembleEngine
from core.orchestrator import Orchestrator

from strategies.rsi_ema import RSIEMAStrategy
from strategies.breakout import BreakoutStrategy
from strategies.ml_strategy import MLStrategy

from config.settings import (
    SYMBOL_LIST, TIMEFRAME_LIST, PRIMARY_TIMEFRAME, LOOP_SLEEP_SECONDS,
    DB_PATH, USE_ML_STRATEGY, ML_MODEL_PATH,
    ENSEMBLE_MIN_CONF, STRATEGY_WEIGHTS, LABEL_HORIZON_BARS
)

# keep your existing modules compatible
from risk_manager import RiskManager
from trade_executor import TradeExecutor

def build_strategies():
    strategies = [
        RSIEMAStrategy(),
        BreakoutStrategy(),
    ]
    if USE_ML_STRATEGY and os.path.exists(ML_MODEL_PATH):
            model = joblib.load(ML_MODEL_PATH)
            strategies.append(MLStrategy(model))
        else:
            print("ML model not found — running without ML strategy.")
return strategies

def main():
    initialize_mt5()

    db = MarketDatabase(DB_PATH)
    fetcher = DataFetcher()
    pipeline = DataPipeline(fetcher, db)

    strategies = build_strategies()
    ensemble = EnsembleEngine(strategies, weights=STRATEGY_WEIGHTS, min_conf=ENSEMBLE_MIN_CONF)

    risk = RiskManager()
    executor = TradeExecutor()

    bot = Orchestrator(
        pipeline=pipeline,
        ensemble=ensemble,
        risk_manager=risk,
        executor=executor,
        db=db,
        symbols=SYMBOL_LIST,
        timeframes=TIMEFRAME_LIST,
        primary_tf=PRIMARY_TIMEFRAME,
        label_horizon_bars=LABEL_HORIZON_BARS
    )
    bot.run_forever(sleep_s=LOOP_SLEEP_SECONDS)

if __name__ == "__main__":
    main()
