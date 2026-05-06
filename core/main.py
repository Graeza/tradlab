from __future__ import annotations

import time
import joblib
import os

from core.mt5_worker import MT5Client
from core.data_fetcher import DataFetcher
from core.database import MarketDatabase
from core.data_pipeline import DataPipeline
from core.ensemble import EnsembleEngine
from core.orchestrator import Orchestrator
from core.ml_model_registry import MLModelRegistry

from strategies.rsi_ema import RSIEMAStrategy
from strategies.breakout import BreakoutStrategy
from strategies.ml_strategy import MLStrategy
from strategies.rsi3_ma_extreme import RSI3MAExtremeStrategy
from strategies.symbol_scoped import SymbolScopedStrategy

from config.settings import (
    SYMBOL_LIST, TIMEFRAME_LIST, PRIMARY_TIMEFRAME, LOOP_SLEEP_SECONDS, NEW_SYMBOL_STRATEGY_SYMBOLS,
    DB_PATH, USE_ML_STRATEGY, ML_MODEL_PATH, ML_CANDIDATES_DIR, ML_REQUIRE_SYMBOL_MODEL, ML_MIN_CANDIDATE_ACCURACY,
    ENSEMBLE_MIN_CONF, STRATEGY_WEIGHTS, LABEL_HORIZON_BARS, REGIME_WEIGHT_MULTIPLIERS
)

from risk_manager import RiskManager
from trade_executor import TradeExecutor


def build_strategies():
    new_symbol_set = set(NEW_SYMBOL_STRATEGY_SYMBOLS)
    legacy_symbols = [s for s in SYMBOL_LIST if s not in new_symbol_set]

    strategies = [
        SymbolScopedStrategy(RSIEMAStrategy(), allowed_symbols=legacy_symbols),
        SymbolScopedStrategy(BreakoutStrategy(), allowed_symbols=legacy_symbols),
        SymbolScopedStrategy(RSI3MAExtremeStrategy(), allowed_symbols=NEW_SYMBOL_STRATEGY_SYMBOLS),
    ]

    if USE_ML_STRATEGY:
        registry = MLModelRegistry(
            candidates_dir=ML_CANDIDATES_DIR,
            fallback_model_path=ML_MODEL_PATH,
            require_symbol_model=ML_REQUIRE_SYMBOL_MODEL,
            min_candidate_accuracy=ML_MIN_CANDIDATE_ACCURACY,
            log=print,
        )
        strategies.append(
            SymbolScopedStrategy(
                MLStrategy(
                    model=None,
                    bundle_registry=registry,
                    default_primary_tf=int(PRIMARY_TIMEFRAME),
                ),
                allowed_symbols=legacy_symbols,
            )
        )
    else:
        print("ML strategy disabled by configuration.")

    return strategies


def main():
    mt5 = MT5Client()
    mt5.start()  # initializes MT5 on the MT5Worker thread

    try:
        db = MarketDatabase(DB_PATH)
        fetcher = DataFetcher(mt5)
        pipeline = DataPipeline(fetcher, db)

        strategies = build_strategies()
        ensemble = EnsembleEngine(
            strategies,
            weights=STRATEGY_WEIGHTS,
            min_conf=ENSEMBLE_MIN_CONF,
            regime_multipliers=REGIME_WEIGHT_MULTIPLIERS,
        )

        risk = RiskManager(mt5)
        executor = TradeExecutor(mt5)

        bot = Orchestrator(
            pipeline=pipeline,
            ensemble=ensemble,
            risk_manager=risk,
            executor=executor,
            db=db,
            symbols=SYMBOL_LIST,
            timeframes=TIMEFRAME_LIST,
            primary_tf=PRIMARY_TIMEFRAME,
            label_horizon_bars=LABEL_HORIZON_BARS,
        )
        bot.run_forever(sleep_s=LOOP_SLEEP_SECONDS)
    finally:
        # Always shut down MT5 on exit
        mt5.shutdown()


if __name__ == "__main__":
    main()
