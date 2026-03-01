from __future__ import annotations
import time
import threading
from typing import Callable, Optional, Any
from core.data_pipeline import DataPipeline
from core.ensemble import EnsembleEngine
from core.database import MarketDatabase
from core.labeling import make_labels_from_bars

class Orchestrator:
    def __init__(self, pipeline: DataPipeline, ensemble: EnsembleEngine, risk_manager, executor, db: MarketDatabase,
                 symbols: list[str], timeframes: list[int], primary_tf: int, label_horizon_bars: int,
                 log: Optional[Callable[[str], None]] = None, allow_new_trades_getter: Optional[Callable[[], bool]] = None, decision_callback: Optional[Callable[[str, dict, list], None]] = None):
        self.pipeline = pipeline
        self.ensemble = ensemble
        self.risk = risk_manager
        self.executor = executor
        self.db = db
        self.symbols = symbols
        self.timeframes = timeframes
        self.primary_tf = primary_tf
        self.label_horizon_bars = label_horizon_bars
        self.log = log or (lambda s: print(s, flush=True))
        self.allow_new_trades_getter = allow_new_trades_getter or (lambda: True)
        self.decision_callback = decision_callback

    def run_forever(self, sleep_s: int = 300, stop_event: Optional[threading.Event] = None):
        stop_event = stop_event or threading.Event()
        self.log("[BOT] Started")

        while not stop_event.is_set():
            for symbol in self.symbols:
                if stop_event.is_set():
                    break
                try:
                    data_by_tf = self.pipeline.update_symbol(symbol, self.timeframes)

                    if self.primary_tf not in data_by_tf:
                        self.log(f"[WARN] {symbol}: no primary timeframe data")
                        continue

                    final_signal, outputs = self.ensemble.run(data_by_tf)
                    
                    # Emit structured decision event (GUI can subscribe)
                    if self.decision_callback:
                        try:
                            self.decision_callback(symbol, final_signal, outputs)
                        except Exception as e:
                            self.log(f"[WARN] decision_callback failed: {e}")
                    self.log(f"[SIGNAL] {symbol}: {final_signal} | details={[(o.get('name'), o.get('signal'), o.get('confidence')) for o in outputs]}")

                    # Delayed labeling (primary timeframe)
                    bars = self.db.load_bars(symbol, self.primary_tf, limit=5000)
                    labels = make_labels_from_bars(bars, symbol, self.primary_tf, self.label_horizon_bars)
                    if not labels.empty:
                        n = self.db.upsert_labels(labels)
                        self.log(f"[LABEL] {symbol}: upserted {n} labels")

                    # SAFETY SWITCH: allow data + signals, but block opening new trades
                    if not self.allow_new_trades_getter():
                        self.log(f"[SAFE MODE] Entries blocked. Would have acted on {symbol}: {final_signal}")
                        continue

                    trade_params = self.risk.assess(final_signal, symbol)
                    if not trade_params:
                        self.log(f"[RISK] {symbol}: rejected")
                        continue

                    res = self.executor.execute(trade_params)
                    self.log(f"[EXEC] {symbol}: {res}")


                    # SAFETY SWITCH: block ONLY new trades
                    if not self.allow_new_trades_getter():
                        self.log(f"[SAFE MODE] Entries blocked. Would have acted on {symbol}: {final_signal}")
                        continue

                    trade_params = self.risk.assess(final_signal, symbol)
                    
                except Exception as e:
                    self.log(f"[ERROR] {symbol}: {e}")

            # responsive stop
            total = max(1, int(sleep_s))
            for _ in range(total):
                if stop_event.is_set():
                    break
                time.sleep(1)

        self.log("[BOT] Stopped")
