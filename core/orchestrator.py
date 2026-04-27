from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Any

from core.data_pipeline import DataPipeline
from core.ensemble import EnsembleEngine
from core.database import MarketDatabase
from core.performance_tracker import PerformanceTracker
from core.labeling import make_labels_from_bars
from utils.regime import detect_regime


class Orchestrator:
    def __init__(
        self,
        pipeline: DataPipeline,
        ensemble: EnsembleEngine,
        risk_manager,
        executor,
        db: MarketDatabase,
        symbols: list[str],
        timeframes: list[int],
        primary_tf: int,
        label_horizon_bars: int,
        log: Optional[Callable[[str], None]] = None,
        allow_new_trades_getter: Optional[Callable[[], bool]] = None,
        decision_callback: Optional[Callable[[str, dict, list], None]] = None,
        enforce_single_position_per_symbol: bool = True,
        max_positions_per_symbol: int = 1,
        max_total_open_positions: int = 0,
        one_entry_per_closed_bar: bool = True,
        enable_trade_cooldown: bool = False,
        trade_cooldown_minutes: int = 0,
        enable_max_daily_trades: bool = False,
        max_daily_trades_per_symbol: int = 0,
        max_daily_trades_total: int = 0,
        auto_close_profits: bool = False,
        auto_close_profits_threshold: float = 0.0,
    ):
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
        self.perf = PerformanceTracker()
        self.current_session_id: int | None = None
        self.current_session_started_at: datetime | None = None

        self.enforce_single_position_per_symbol = bool(enforce_single_position_per_symbol)
        self.max_positions_per_symbol = max(1, int(max_positions_per_symbol))
        self.max_total_open_positions = max(0, int(max_total_open_positions))
        self.one_entry_per_closed_bar = bool(one_entry_per_closed_bar)
        self.enable_trade_cooldown = bool(enable_trade_cooldown)
        self.trade_cooldown_minutes = max(0, int(trade_cooldown_minutes))
        self.enable_max_daily_trades = bool(enable_max_daily_trades)
        self.max_daily_trades_per_symbol = max(0, int(max_daily_trades_per_symbol))
        self.max_daily_trades_total = max(0, int(max_daily_trades_total))

        self._last_entry_bar_by_symbol: dict[str, str] = {}
        self._last_trade_time_by_symbol: dict[str, datetime] = {}
        self._daily_trade_counts_by_symbol: dict[tuple[str, str], int] = {}
        self._daily_trade_count_total: dict[str, int] = {}

        self.auto_close_profits = bool(auto_close_profits)
        self.auto_close_profits_threshold = float(auto_close_profits_threshold)
    
    def set_trade_session(self, session_id: int | None, started_at: datetime | None = None) -> None:
        self.current_session_id = session_id
        self.current_session_started_at = started_at

    def update_entry_policy(
        self,
        *,
        enforce_single_position_per_symbol: bool | None = None,
        max_positions_per_symbol: int | None = None,
        max_total_open_positions: int | None = None,
        one_entry_per_closed_bar: bool | None = None,
        enable_trade_cooldown: bool | None = None,
        trade_cooldown_minutes: int | None = None,
        enable_max_daily_trades: bool | None = None,
        max_daily_trades_per_symbol: int | None = None,
        max_daily_trades_total: int | None = None,
        auto_close_profits: bool | None = None,
        auto_close_profits_threshold: float | None = None,

    ) -> None:
        if enforce_single_position_per_symbol is not None:
            self.enforce_single_position_per_symbol = bool(enforce_single_position_per_symbol)
        if max_positions_per_symbol is not None:
            self.max_positions_per_symbol = max(1, int(max_positions_per_symbol))
        if max_total_open_positions is not None:
            self.max_total_open_positions = max(0, int(max_total_open_positions))
        if one_entry_per_closed_bar is not None:
            self.one_entry_per_closed_bar = bool(one_entry_per_closed_bar)
        if enable_trade_cooldown is not None:
            self.enable_trade_cooldown = bool(enable_trade_cooldown)
        if trade_cooldown_minutes is not None:
            self.trade_cooldown_minutes = max(0, int(trade_cooldown_minutes))
        if enable_max_daily_trades is not None:
            self.enable_max_daily_trades = bool(enable_max_daily_trades)
        if max_daily_trades_per_symbol is not None:
            self.max_daily_trades_per_symbol = max(0, int(max_daily_trades_per_symbol))
        if max_daily_trades_total is not None:
            self.max_daily_trades_total = max(0, int(max_daily_trades_total))
        if auto_close_profits is not None:
            self.auto_close_profits = bool(auto_close_profits)
        if auto_close_profits_threshold is not None:
            self.auto_close_profits_threshold = float(auto_close_profits_threshold)

    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _latest_closed_bar_key(self, primary_df) -> str | None:
        try:
            if primary_df is None or len(primary_df.index) == 0:
                return None
            idx = primary_df.index[-1]
            if hasattr(idx, "to_pydatetime"):
                idx = idx.to_pydatetime()
            if isinstance(idx, datetime):
                if idx.tzinfo is None:
                    idx = idx.replace(tzinfo=timezone.utc)
                return idx.astimezone(timezone.utc).isoformat()
            return str(idx)
        except Exception:
            return None

    def _managed_open_positions(self, symbol: str | None = None) -> int:
        try:
            return int(self.executor.count_open_positions(symbol=symbol))
        except Exception:
            return 0

    def _prune_old_trade_counts(self, keep_days: int = 7) -> None:
        today = self._utc_now().date()
        valid_days = {
            (today - timedelta(days=offset)).isoformat()
            for offset in range(max(1, int(keep_days)))
        }
        self._daily_trade_count_total = {
            day: count for day, count in self._daily_trade_count_total.items() if day in valid_days
        }
        self._daily_trade_counts_by_symbol = {
            key: count for key, count in self._daily_trade_counts_by_symbol.items() if key[0] in valid_days
        }

    def _entry_policy_block_reason(self, symbol: str, primary_df) -> str | None:
        open_for_symbol = self._managed_open_positions(symbol=symbol)
        if self.enforce_single_position_per_symbol and open_for_symbol >= max(1, int(self.max_positions_per_symbol)):
            return f"max_positions_per_symbol ({open_for_symbol}/{self.max_positions_per_symbol})"

        if self.max_total_open_positions > 0:
            open_total = self._managed_open_positions(symbol=None)
            if open_total >= int(self.max_total_open_positions):
                return f"max_total_open_positions ({open_total}/{self.max_total_open_positions})"

        bar_key = self._latest_closed_bar_key(primary_df)
        if self.one_entry_per_closed_bar and bar_key and self._last_entry_bar_by_symbol.get(symbol) == bar_key:
            return f"already_traded_closed_bar ({bar_key})"

        now = self._utc_now()
        if self.enable_trade_cooldown and self.trade_cooldown_minutes > 0:
            last_trade_at = self._last_trade_time_by_symbol.get(symbol)
            if last_trade_at is not None:
                elapsed = (now - last_trade_at).total_seconds()
                required = float(self.trade_cooldown_minutes) * 60.0
                if elapsed < required:
                    remaining = max(0, int((required - elapsed + 59) // 60))
                    return f"cooldown_active ({remaining}m remaining)"

        if self.enable_max_daily_trades:
            self._prune_old_trade_counts()
            today = now.date().isoformat()
            per_symbol = self._daily_trade_counts_by_symbol.get((today, symbol), 0)
            total = self._daily_trade_count_total.get(today, 0)

            if self.max_daily_trades_per_symbol > 0 and per_symbol >= int(self.max_daily_trades_per_symbol):
                return f"max_daily_trades_per_symbol ({per_symbol}/{self.max_daily_trades_per_symbol})"

            if self.max_daily_trades_total > 0 and total >= int(self.max_daily_trades_total):
                return f"max_daily_trades_total ({total}/{self.max_daily_trades_total})"

        return None

    def _record_successful_entry(self, symbol: str, primary_df, res: dict[str, Any]) -> None:
        event_time = res.get("event_time")
        now = event_time if isinstance(event_time, datetime) else self._utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        self._last_trade_time_by_symbol[symbol] = now
        bar_key = self._latest_closed_bar_key(primary_df)
        if bar_key:
            self._last_entry_bar_by_symbol[symbol] = bar_key

        today = now.date().isoformat()
        key = (today, symbol)
        self._daily_trade_counts_by_symbol[key] = int(self._daily_trade_counts_by_symbol.get(key, 0)) + 1
        self._daily_trade_count_total[today] = int(self._daily_trade_count_total.get(today, 0)) + 1
        self._prune_old_trade_counts()


    def _serialize_exec_result(self, res: dict[str, Any]) -> str:
        try:
            safe = {}
            for k, v in (res or {}).items():
                if k == "raw":
                    continue
                if isinstance(v, datetime):
                    safe[k] = v.astimezone(timezone.utc).isoformat()
                else:
                    safe[k] = v
            return json.dumps(safe, default=str)
        except Exception:
            return ""

    def _log_open_trade(self, symbol: str, final_signal: dict, res: dict[str, Any]) -> None:
        session_id = self.current_session_id
        if not session_id or not res.get("ok"):
            return
        position_id = int(res.get("position_id") or 0)
        if position_id <= 0:
            position_id = int(res.get("order_ticket") or 0)
        self.db.log_trade_open(
            session_id=session_id,
            event_time=res.get("event_time"),
            symbol=symbol,
            side=str(res.get("action") or ""),
            volume=res.get("volume"),
            entry_price=res.get("price"),
            initial_sl=res.get("sl"),
            initial_tp=res.get("tp"),
            position_id=position_id if position_id > 0 else None,
            order_ticket=int(res.get("order_ticket") or 0) or None,
            deal_ticket=int(res.get("deal_ticket") or 0) or None,
            strategy_name=str(final_signal.get("winning_strategy") or ""),
            comment=str(final_signal.get("signal") or ""),
            raw_result_json=self._serialize_exec_result(res),
        )

    def _apply_trailing_stop_logic(self) -> None:
        session_id = self.current_session_id
        events = self.executor.manage_trailing_stops()
        for ev in events:
            if ev.get("ok"):
                self.log(
                    f"[TRAIL] {ev.get('symbol')} pos={ev.get('position_id')} "
                    f"SL {ev.get('old_sl')} -> {ev.get('new_sl')} @ price={ev.get('live_price')}"
                )
                if session_id and ev.get("position_id"):
                    try:
                        self.db.log_trade_stop_event(
                            session_id=session_id,
                            position_id=int(ev["position_id"]),
                            symbol=str(ev.get("symbol") or ""),
                            event_time=ev.get("event_time"),
                            event_type="TRAIL",
                            sl=ev.get("new_sl"),
                            tp=ev.get("tp"),
                            source="executor.trailing",
                            note=(
                                f"old_sl={ev.get('old_sl')} live_price={ev.get('live_price')} "
                                f"initial_risk={ev.get('initial_risk')}"
                            ),
                        )
                    except Exception as e:
                        self.log(f"[WARN] trail journal log failed for {ev.get('symbol')}: {e}")
                        
            elif ev.get("reason") not in (
                None,
                "not_in_profit",
                "trigger_not_reached",
                "not_better_than_current",
                "step_not_reached",
                "no_current_sl",
                "no_anchor_sl",
                "zero_initial_risk",
            ):
                self.log(f"[WARN] trailing stop check failed for {ev.get('symbol')}: {ev}")

    def _auto_close_profits_positions(self) -> None:
        if not bool(self.auto_close_profits):
            return

        threshold = float(self.auto_close_profits_threshold or 0.0)

        try:
            events = self.executor.auto_close_profits(
                min_profit=threshold
            )
        except Exception as e:
            self.log(f"[WARN] auto-close profits failed: {e}")
            return

        if not events:
            return

        for ev in events:
            symbol = str(ev.get("symbol") or "")
            ticket = ev.get("position_id")
            profit = ev.get("profit")
            if ev.get("ok"):
                self.log(
                    f"[AUTO CLOSE] {symbol} BUY pos={ticket} closed in profit "
                    f"(profit={profit})"
                )
            else:
                self.log(
                    f"[WARN] auto-close profits failed for {symbol} BUY pos={ticket}: "
                    f"{ev.get('reason') or ev.get('error') or ev}"
                )

    def run_forever(self, sleep_s: int = 300, stop_event: Optional[threading.Event] = None):
        """Main loop.

        Fixes:
        - Remove duplicated SAFETY SWITCH / assess block.
        - Ensure primary timeframe presence check is correct.
        - Keep control flow clean: decide -> (optional) execute.
        """
        stop_event = stop_event or threading.Event()
        self.log("[BOT] Started")

        while not stop_event.is_set():
            try:
                self._apply_trailing_stop_logic()
            except Exception as e:
                self.log(f"[WARN] pre-loop trailing management failed: {e}")
            try:
                self._auto_close_profits_positions()
            except Exception as e:
                self.log(f"[WARN] pre-loop auto-close profits failed: {e}")
            for symbol in self.symbols:
                if stop_event.is_set():
                    break
                try:
                    data_by_tf = self.pipeline.update_symbol(symbol, self.timeframes)
                    if self.primary_tf not in data_by_tf:
                        self.log(f"[WARN] {symbol}: no primary timeframe data")
                        continue

                    primary_df = data_by_tf.get(self.primary_tf)
                    regime = detect_regime(primary_df) if primary_df is not None else {"trend": "UNKNOWN", "vol": "UNKNOWN"}
                    final_signal, outputs = self.ensemble.run(data_by_tf, regime=regime,context={
                            "symbol": symbol,
                            "primary_tf": self.primary_tf,
                        },
                    )
                    # attach regime for downstream components
                    if isinstance(final_signal, dict):
                        final_signal["regime"] = regime
                        if outputs:
                            winner = max(outputs, key=lambda o: float(o.get("confidence", 0.0) or 0.0))
                            final_signal["winning_strategy"] = str(winner.get("name") or "")
                            final_signal["comment"] = str(winner.get("signal") or "")

                    # Performance tracking (non-blocking metadata)
                    self.perf.add_prediction(
                        symbol=symbol,
                        df_primary=primary_df,
                        horizon_bars=self.label_horizon_bars,
                        final=final_signal,
                        outputs=outputs,
                    )
                    self.perf.update_with_bars(symbol, primary_df)

                    # Emit structured decision event (GUI can subscribe)
                    if self.decision_callback:
                        try:
                            self.decision_callback(symbol, final_signal, outputs)
                        except Exception as e:
                            self.log(f"[WARN] decision_callback failed: {e}")

                    self.log(
                        f"[SIGNAL] {symbol}: {final_signal} | details={[(o.get('name'), o.get('signal'), o.get('confidence')) for o in outputs]}"
                    )

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

                    entry_block = self._entry_policy_block_reason(symbol, primary_df)
                    if entry_block:
                        self.log(f"[ENTRY POLICY] {symbol}: blocked -> {entry_block}")
                        continue

                    trade_params = self.risk.assess(final_signal, symbol)
                    if not trade_params:
                        self.log(f"[RISK] {symbol}: rejected")
                        continue

                    trade_params["strategy_name"] = str(final_signal.get("winning_strategy") or "")
                    trade_params["comment"] = str(final_signal.get("comment") or "ModularBot")
                    res = self.executor.execute(trade_params)
                    self.log(f"[EXEC] {symbol}: {res}")

                    if isinstance(res, dict) and res.get("ok"):
                        try:
                            self._record_successful_entry(symbol, primary_df, res)
                        except Exception as e:
                            self.log(f"[WARN] entry policy state update failed for {symbol}: {e}")
                        try:
                            self._log_open_trade(symbol, final_signal, res)
                        except Exception as e:
                            self.log(f"[WARN] journal open log failed for {symbol}: {e}")

                    try:
                        self._apply_trailing_stop_logic()
                    except Exception as e:
                        self.log(f"[WARN] trailing management failed after {symbol}: {e}")
                    try:
                        self._auto_close_profits_positions()
                    except Exception as e:
                        self.log(f"[WARN] idle auto-close profits failed: {e}")
                except Exception as e:
                    self.log(f"[ERROR] {symbol}: {e}")

            # responsive stop
            total = max(1, int(sleep_s))
            for tick in range(total):
                if stop_event.is_set():
                    break
                time.sleep(1)
                try:
                    self._apply_trailing_stop_logic()
                except Exception as e:
                    self.log(f"[WARN] idle trailing management failed: {e}")
                if tick % 2 == 1:
                    try:
                        self._auto_close_profits_positions()
                    except Exception as e:
                        self.log(f"[WARN] timed auto-close profits failed: {e}")

        self.log("[BOT] Stopped")
