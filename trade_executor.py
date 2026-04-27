from __future__ import annotations

import time
import math
from datetime import datetime, time as dtime , timezone
from typing import Optional, Any

import MetaTrader5 as mt5  # for constants only

from core.mt5_worker import MT5Client


class TradeExecutor:
    """Executes trades via MT5 with optional execution guardrails and trailing stops.

    IMPORTANT: All MT5 terminal calls are serialized via MT5Client.
    """

    def __init__(
        self,
        mt5_client: MT5Client,
        enable_spread_filter: bool = True,
        max_spread_points: int = 200,
        enable_session_filter: bool = False,
        session_start_hour: int = 0,
        session_end_hour: int = 24,
        allow_weekends: bool = True,
        max_retries: int = 0,
        retry_delay_ms: int = 250,
        magic: int = 123456,
        comment: str = "ModularBot",
        min_allowed_lot: float = 0.0,   # 0.0 = disabled
        force_symbol_fixed_lot: bool = True,
        boom_crash_fixed_sl_tp: bool = True,
        boom_crash_sl_tp_offset: float = 10.0,
        enable_trailing_stop: bool = False,
        trailing_trigger_rr: float = 0.5,
        trailing_distance_rr: float = 0.5,
        trailing_step_rr: float = 0.10,
        blocked_symbols: Optional[set[str]] = None
    ):
        self.mt5 = mt5_client

        self.enable_spread_filter = bool(enable_spread_filter)
        self.max_spread_points = int(max_spread_points)

        self.enable_session_filter = bool(enable_session_filter)
        self.session_start_hour = int(session_start_hour)
        self.session_end_hour = int(session_end_hour)
        self.allow_weekends = bool(allow_weekends)

        self.max_retries = int(max_retries)
        self.retry_delay_ms = int(retry_delay_ms)

        self.magic = int(magic)
        self.comment = str(comment)
        self.min_allowed_lot = float(min_allowed_lot)
        self.force_symbol_fixed_lot = bool(force_symbol_fixed_lot)
        self._trailing_anchor_sl: dict[int, float] = {}
        self.boom_crash_fixed_sl_tp = bool(boom_crash_fixed_sl_tp)
        self.boom_crash_sl_tp_offset = float(boom_crash_sl_tp_offset)
        self.enable_trailing_stop = bool(enable_trailing_stop)
        self.trailing_trigger_rr = float(trailing_trigger_rr)
        self.trailing_distance_rr = float(trailing_distance_rr)
        self.trailing_step_rr = max(0.0, float(trailing_step_rr))
        self.blocked_symbols: set[str] = blocked_symbols if blocked_symbols is not None else set()

    def _fixed_lot_for_symbol(self, symbol: str) -> Optional[float]:
        s = symbol.lower()
        if "boom 1000" in s or "boom 900" in s or "boom 500" in s or "boom 600" in s:
            return 0.2
        if "boom 300" in s:
            return 0.5
        return None

    def _min_allowed_lot_for_symbol(self, symbol: str) -> float:
        s = symbol.lower()
        if "boom 1000" in s or "boom 900" in s or "boom 500" in s or "boom 600" in s:
            return 0.2
        if "boom 300" in s:
            return 0.5
        return 0.0  # disabled for other symbols

    def _normalize_volume(self, symbol: str, volume: float) -> float:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol info not found for {symbol}")

        min_vol = float(getattr(info, "volume_min", 0.0) or 0.0)
        max_vol = float(getattr(info, "volume_max", 0.0) or 0.0)
        step = float(getattr(info, "volume_step", 0.0) or 0.0)

        if min_vol <= 0 or max_vol <= 0 or step <= 0:
            # Fall back: return requested volume as-is
            return float(volume)

        # Clamp
        volume = max(min_vol, min(float(volume), max_vol))

        # Floor-to-step (never round up)
        steps = math.floor((volume - min_vol) / step + 1e-12)
        normalized = min_vol + steps * step

        # Round decimals based on step (0.01->2, 0.1->1, 1.0->0)
        decimals = max(0, int(round(-math.log10(step))) if step < 1 else 0)
        return round(normalized, decimals)

    def _within_session(self) -> bool:
        now = datetime.now()
        if not self.allow_weekends and now.weekday() >= 5:
            return False

        if not self.enable_session_filter:
            return True

        sh = max(0, min(23, int(self.session_start_hour)))
        eh = max(0, min(24, int(self.session_end_hour)))

        if sh == eh:
            return True

        t = now.time()
        start = dtime(sh, 0, 0)
        end = dtime((eh % 24), 0, 0)

        if sh < eh:
            return start <= t < end
        return t >= start or t < end

    def _spread_points(self, symbol: str) -> Optional[int]:
        info = self.mt5.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            return None

        if not bool(getattr(info, "visible", True)):
            self.mt5.symbol_select(symbol, True)

        info_spread = float(getattr(info, "spread", 0.0) or 0.0)
        if info_spread > 0:
            return int(round(info_spread))

        pt = float(getattr(info, "point", 0.0) or 0.0)
        if pt <= 0:
            return None

        spread = (float(tick.ask) - float(tick.bid)) / pt
        return int(round(spread))

    # -------------------------
    # Stop-level helpers
    # -------------------------
    def _stops_min_distance(self, info) -> float:
        """Minimum allowed SL/TP distance from current price in *price units*.

        MT5 provides trade_stops_level in POINTS; convert using point.
        """
        point = float(getattr(info, "point", 0.0) or 0.0)
        stops_level_pts = float(getattr(info, "trade_stops_level", 0.0) or 0.0)
        stop_level_alt = float(getattr(info, "stop_level", 0.0) or 0.0)  # some brokers

        lvl_pts = max(stops_level_pts, stop_level_alt, 0.0)
        if point <= 0:
            return 0.0
        if lvl_pts <= 0:
            # conservative fallback if broker doesn't report
            return point * 10.0
        return lvl_pts * point

    def _round_price(self, info, price: float | None) -> float | None:
        if price is None:
            return None
        digits = getattr(info, "digits", None)
        if digits is None:
            return float(price)
        return round(float(price), int(digits))
    
    def _adjust_sl_tp_to_stops(self, info, action: str, price: float, sl, tp):
        """Widen SL/TP to meet broker min stop distance if required."""
        min_dist = self._stops_min_distance(info)
        if min_dist <= 0:
            return sl, tp

        a = action.upper()

        if sl is not None and float(sl) != 0.0:
            sl = float(sl)
            if a == "BUY":
                if (price - sl) < min_dist:
                    sl = price - min_dist
            else:
                if (sl - price) < min_dist:
                    sl = price + min_dist

        if tp is not None and float(tp) != 0.0:
            tp = float(tp)
            if a == "BUY":
                if (tp - price) < min_dist:
                    tp = price + min_dist
            else:
                if (price - tp) < min_dist:
                    tp = price - min_dist

        sl = self._round_price(info, sl) if sl not in (None, 0.0) else sl
        tp = self._round_price(info, tp) if tp not in (None, 0.0) else tp
        return sl, tp
    
    def _apply_fixed_sl_tp_offset(self, symbol: str, action: str, price: float, sl, tp, info):
        if not self.boom_crash_fixed_sl_tp:
            return sl, tp
        sym_l = symbol.lower()
        if "boom" not in sym_l and "crash" not in sym_l:
            return sl, tp
        offset = float(self.boom_crash_sl_tp_offset or 0.0)
        if offset <= 0:
            return sl, tp
        if action.upper() == "BUY":
            sl = price - offset
            tp = price + offset
        else:
            sl = price + offset
            tp = price - offset
        return self._round_price(info, sl), self._round_price(info, tp)

    def _position_matches(self, pos) -> bool:
        try:
            pos_magic = int(getattr(pos, "magic", 0) or 0)
        except Exception:
            pos_magic = 0
        if pos_magic and pos_magic != self.magic:
            return False
        pos_comment = str(getattr(pos, "comment", "") or "")
        if pos_magic == 0 and self.comment and pos_comment and pos_comment != self.comment:
            return False
        return True

    def _position_side(self, pos) -> str:
        ptype = int(getattr(pos, "type", -1))
        if ptype == int(mt5.POSITION_TYPE_BUY):
            return "BUY"
        if ptype == int(mt5.POSITION_TYPE_SELL):
            return "SELL"
        return "UNKNOWN"

    def _position_live_price(self, symbol: str, side: str) -> float | None:
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        if side == "BUY":
            return float(getattr(tick, "bid", 0.0) or 0.0)
        if side == "SELL":
            return float(getattr(tick, "ask", 0.0) or 0.0)
        return None

    def _modify_position_sl_tp(self, *, position_id: int, symbol: str, sl: float | None, tp: float | None) -> Any:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": int(position_id),
            "sl": 0.0 if sl in (None, 0, 0.0) else float(sl),
            "tp": 0.0 if tp in (None, 0, 0.0) else float(tp),
        }
        return mt5.order_send(request)

    def _rr_triggered(self, current_profit_dist: float, initial_risk: float) -> bool:
        trigger = max(0.0, float(self.trailing_trigger_rr)) * initial_risk
        return current_profit_dist >= (trigger - 1e-12)

    def _rr_step_passed(self, candidate_sl: float, current_sl: float | None, initial_risk: float, side: str) -> bool:
        if current_sl in (None, 0, 0.0):
            return True
        step = max(0.0, float(self.trailing_step_rr)) * initial_risk
        if step <= 0:
            return True
        if side == "BUY":
            return (candidate_sl - float(current_sl)) >= (step - 1e-12)
        return (float(current_sl) - candidate_sl) >= (step - 1e-12)
    
    def _get_anchor_sl(self, pos) -> float | None:
        position_id = int(getattr(pos, "ticket", 0) or 0)
        if position_id <= 0:
            return None

        cached = self._trailing_anchor_sl.get(position_id)
        if cached not in (None, 0, 0.0):
            return float(cached)

        live_sl = getattr(pos, "sl", None)
        if live_sl in (None, 0, 0.0):
            return None

        live_sl = float(live_sl)
        self._trailing_anchor_sl[position_id] = live_sl
        return live_sl
    
    def _compute_trailing_sl(self, pos) -> tuple[float | None, dict[str, Any] | None]:
        side = self._position_side(pos)
        if side not in ("BUY", "SELL"):
            return None, {"reason": "unsupported_position_type"}

        position_id = int(getattr(pos, "ticket", 0) or 0)
        symbol = str(getattr(pos, "symbol", "") or "")
        info = self.mt5.symbol_info(symbol)
        if info is None:
            return None, {"reason": "no_symbol_info"}

        entry_price = float(getattr(pos, "price_open", 0.0) or 0.0)
        current_sl = getattr(pos, "sl", None)
        current_tp = getattr(pos, "tp", None)

        if entry_price <= 0:
            return None, {"reason": "no_entry_price"}

        if current_sl in (None, 0, 0.0):
            return None, {"reason": "no_current_sl"}

        current_sl = float(current_sl)

        anchor_sl = self._get_anchor_sl(pos)
        if anchor_sl in (None, 0, 0.0):
            return None, {"reason": "no_anchor_sl"}

        anchor_sl = float(anchor_sl)
        initial_risk = abs(entry_price - anchor_sl)
        if initial_risk <= 0:
            return None, {
                "reason": "zero_initial_risk",
                "entry_price": entry_price,
                "anchor_sl": anchor_sl,
            }

        live_price = self._position_live_price(symbol, side)
        if live_price is None or live_price <= 0:
            return None, {"reason": "no_live_price"}

        profit_dist = (live_price - entry_price) if side == "BUY" else (entry_price - live_price)
        if profit_dist <= 0:
            return None, {
                "reason": "not_in_profit",
                "profit_dist": profit_dist,
                "initial_risk": initial_risk,
            }

        if not self._rr_triggered(profit_dist, initial_risk):
            return None, {
                "reason": "trigger_not_reached",
                "profit_dist": profit_dist,
                "initial_risk": initial_risk,
            }

        trail_gap = max(0.0, float(self.trailing_distance_rr)) * initial_risk

        if side == "BUY":
            candidate_sl = live_price - trail_gap
            candidate_sl = min(candidate_sl, live_price - self._stops_min_distance(info))
            candidate_sl = self._round_price(info, candidate_sl)

            if candidate_sl is None or candidate_sl <= entry_price:
                candidate_sl = self._round_price(info, entry_price)

            if candidate_sl is None or candidate_sl <= current_sl + 1e-12:
                return None, {
                    "reason": "not_better_than_current",
                    "candidate_sl": candidate_sl,
                    "current_sl": current_sl,
                }

        else:
            candidate_sl = live_price + trail_gap
            candidate_sl = max(candidate_sl, live_price + self._stops_min_distance(info))
            candidate_sl = self._round_price(info, candidate_sl)

            if candidate_sl is None or candidate_sl >= entry_price:
                candidate_sl = self._round_price(info, entry_price)

            if candidate_sl is None or candidate_sl >= current_sl - 1e-12:
                return None, {
                    "reason": "not_better_than_current",
                    "candidate_sl": candidate_sl,
                    "current_sl": current_sl,
                }

        if not self._rr_step_passed(candidate_sl, current_sl, initial_risk, side):
            return None, {
                "reason": "step_not_reached",
                "candidate_sl": candidate_sl,
                "current_sl": current_sl,
                "initial_risk": initial_risk,
            }

        adj_sl, adj_tp = self._adjust_sl_tp_to_stops(info, side, live_price, candidate_sl, current_tp)
        candidate_sl = None if adj_sl in (None, 0, 0.0) else float(adj_sl)
        if candidate_sl is None:
            return None, {"reason": "candidate_invalid_after_adjust"}

        meta = {
            "position_id": position_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "live_price": live_price,
            "anchor_sl": anchor_sl,
            "initial_risk": initial_risk,
            "profit_dist": profit_dist,
            "current_sl": current_sl,
            "current_tp": None if current_tp in (None, 0, 0.0) else float(current_tp),
            "candidate_sl": candidate_sl,
            "candidate_tp": None if adj_tp in (None, 0, 0.0) else float(adj_tp),
        }
        return candidate_sl, meta
    
    def _cleanup_trailing_anchors(self, open_positions) -> None:
        open_ids = {
            int(getattr(p, "ticket", 0) or 0)
            for p in open_positions
            if int(getattr(p, "ticket", 0) or 0) > 0
        }
        stale = [pid for pid in self._trailing_anchor_sl.keys() if pid not in open_ids]
        for pid in stale:
            self._trailing_anchor_sl.pop(pid, None)

    def manage_trailing_stops(self) -> list[dict[str, Any]]:
        if not self.enable_trailing_stop:
            return []

        try:
            positions = self.mt5.positions_get() or []
            self._cleanup_trailing_anchors(positions)
        except Exception as e:
            return [{"ok": False, "reason": "positions_get_failed", "error": str(e)}]

        events: list[dict[str, Any]] = []
        for pos in positions:
            try:
                if not self._position_matches(pos):
                    continue
                position_id = int(getattr(pos, "ticket", 0) or 0)
                symbol = str(getattr(pos, "symbol", "") or "")
                if position_id <= 0 or not symbol:
                    continue

                new_sl, meta = self._compute_trailing_sl(pos)
                if new_sl is None:
                    print(
                        f"[TRAIL SKIP] symbol={symbol} pos={position_id} "
                        f"reason={meta.get('reason') if isinstance(meta, dict) else 'unknown'} "
                        f"meta={meta}"
                    )
                    continue

                tp = meta.get("candidate_tp") if meta else None
                print(
                    f"[TRAIL DEBUG] symbol={symbol} pos={position_id} "
                    f"side={meta.get('side')} entry={meta.get('entry_price')} "
                    f"live={meta.get('live_price')} old_sl={meta.get('current_sl')} "
                    f"new_sl={new_sl} risk={meta.get('initial_risk')} "
                    f"profit_dist={meta.get('profit_dist')}"
                )

                result = self._modify_position_sl_tp(
                    position_id=position_id,
                    symbol=symbol,
                    sl=new_sl,
                    tp=tp,
                )

                retcode = int(getattr(result, "retcode", -1) or -1) if result is not None else -1

                print(
                    f"[TRAIL DEBUG] modify symbol={symbol} pos={position_id} "
                    f"retcode={retcode} comment={getattr(result, 'comment', '')}"
                )

                ok = result is not None and retcode in {
                    int(mt5.TRADE_RETCODE_DONE),
                    int(mt5.TRADE_RETCODE_DONE_PARTIAL),
                    int(mt5.TRADE_RETCODE_PLACED),
                }

                event = {
                    "ok": ok,
                    "event_type": "TRAIL",
                    "position_id": position_id,
                    "symbol": symbol,
                    "side": meta.get("side") if meta else self._position_side(pos),
                    "old_sl": meta.get("current_sl") if meta else None,
                    "new_sl": new_sl,
                    "tp": tp,
                    "live_price": meta.get("live_price") if meta else None,
                    "entry_price": meta.get("entry_price") if meta else None,
                    "initial_risk": meta.get("initial_risk") if meta else None,
                    "profit_dist": meta.get("profit_dist") if meta else None,
                    "retcode": retcode,
                    "comment": getattr(result, "comment", "") if result is not None else "",
                    "event_time": datetime.now(timezone.utc),
                }
                events.append(event)
            except Exception as e:
                events.append({
                    "ok": False,
                    "event_type": "TRAIL",
                    "reason": "exception",
                    "error": str(e),
                    "position_id": int(getattr(pos, "ticket", 0) or 0),
                    "symbol": str(getattr(pos, "symbol", "") or ""),
                    "event_time": datetime.now(timezone.utc),
                })
        return events

    #-------Position / Close Helper Functions-------
    def _managed_positions(self, symbol: Optional[str] = None):
        try:
            pos = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
        except TypeError:
            pos = self.mt5.positions_get()
        pos_list = list(pos) if pos else []
        if symbol is not None:
            pos_list = [p for p in pos_list if str(getattr(p, "symbol", "") or "") == str(symbol)]
        return [p for p in pos_list if self._position_matches(p)]

    def count_open_positions(self, symbol: Optional[str] = None) -> int:
        return len(self._managed_positions(symbol=symbol))

    def has_open_position(self, symbol: str) -> bool:
        return self.count_open_positions(symbol=symbol) > 0

    def _positions(self):
        pos = self.mt5.positions_get()
        return list(pos) if pos else []

    def _close_position_obj(self, p) -> bool:
        symbol = getattr(p, "symbol", "")
        volume = float(getattr(p, "volume", 0.0) or 0.0)
        ticket = int(getattr(p, "ticket", 0) or 0)
        ptype = int(getattr(p, "type", -1))

        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"[EXECUTOR] close skip: no tick for {symbol}")
            return False

        if ptype == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)
        elif ptype == mt5.POSITION_TYPE_SELL:
            order_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)
        else:
            print(f"[EXECUTOR] close skip: unknown position type ticket={ticket}")
            return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 50,
            "magic": self.magic,
            "comment": f"{self.comment}-close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        ok = result is not None and int(getattr(result, "retcode", 0)) == mt5.TRADE_RETCODE_DONE
        if not ok:
            print(f"[EXECUTOR] close failed ticket={ticket} result={result}")
        return ok

    def close_all_positions(self) -> int:
        count = 0
        for p in self._positions():
            if self._close_position_obj(p):
                count += 1
        return count

    def close_positions_by_side(self, side: str) -> int:
        side = str(side).upper()
        count = 0
        for p in self._positions():
            ptype = int(getattr(p, "type", -1))
            if side == "BUY" and ptype != mt5.POSITION_TYPE_BUY:
                continue
            if side == "SELL" and ptype != mt5.POSITION_TYPE_SELL:
                continue
            if self._close_position_obj(p):
                count += 1
        return count

    def close_positions_by_tickets(self, tickets: list[int]) -> int:
        wanted = {int(t) for t in tickets if int(t) > 0}
        if not wanted:
            return 0

        count = 0
        for p in self._positions():
            ticket = int(getattr(p, "ticket", 0) or 0)
            if ticket not in wanted:
                continue
            if self._close_position_obj(p):
                count += 1
        return count

    def close_positions_in_profit(self) -> int:
        count = 0
        for p in self._positions():
            profit = float(getattr(p, "profit", 0.0) or 0.0)
            if profit > 0 and self._close_position_obj(p):
                count += 1
        return count

    def close_positions_in_loss(self) -> int:
        count = 0
        for p in self._positions():
            profit = float(getattr(p, "profit", 0.0) or 0.0)
            if profit < 0 and self._close_position_obj(p):
                count += 1
        return count

    def auto_close_profits(self, min_profit: float = 0.0) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        threshold = float(min_profit or 0.0)

        for p in self._managed_positions():
            try:
                symbol = str(getattr(p, "symbol", "") or "")
                if "boom" not in symbol.lower():
                    continue

                side = self._position_side(p)
                if side != "BUY":
                    continue

                profit = float(getattr(p, "profit", 0.0) or 0.0)
                if profit <= threshold:
                    continue

                position_id = int(getattr(p, "ticket", 0) or 0)
                ok = self._close_position_obj(p)

                events.append({
                    "ok": bool(ok),
                    "event_type": "AUTO_CLOSE_PROFITS",
                    "position_id": position_id,
                    "symbol": symbol,
                    "side": side,
                    "profit": profit,
                    "event_time": datetime.now(timezone.utc),
                    "reason": None if ok else "close_failed",
                })
            except Exception as e:
                events.append({
                    "ok": False,
                    "event_type": "AUTO_CLOSE_PROFITS",
                    "position_id": int(getattr(p, "ticket", 0) or 0),
                    "symbol": str(getattr(p, "symbol", "") or ""),
                    "side": self._position_side(p) if p is not None else "UNKNOWN",
                    "profit": float(getattr(p, "profit", 0.0) or 0.0) if p is not None else 0.0,
                    "event_time": datetime.now(timezone.utc),
                    "reason": "exception",
                    "error": str(e),
                })

        return events

    # Backward-compatible alias
    def auto_close_profitable_boom_buys(self, min_profit: float = 0.0) -> list[dict[str, Any]]:
        return self.auto_close_profits(min_profit=min_profit)

    def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        symbol = str(params["symbol"])
        action = str(params["action"]).upper()
        requested_lot = float(params["lot_size"])

        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"ok": False, "reason": "no_tick", "symbol": symbol}

        # Treat 0 / None as "not set"
        sl = params.get("sl")
        tp = params.get("tp")
        sl = None if sl in (None, 0, 0.0) else float(sl)
        tp = None if tp in (None, 0, 0.0) else float(tp)

        deviation = int(params.get("deviation") or 20)
        strategy_name = str(params.get("strategy_name") or "")
        strategy_comment = str(params.get("comment") or self.comment)

        # Extra spread gate (price-based) for synthetics (kept from your version)
        spread_price = float(tick.ask) - float(tick.bid)
        if "boom" in symbol.lower() or "crash" in symbol.lower():
            if spread_price > 3.0:   # adjust after observing
                return {"ok": False, "reason": "spread_price_too_high", "symbol": symbol,  "spread_price": spread_price}

        if not self._within_session():
            print(f"[EXECUTOR] BLOCK session_filter symbol={symbol}")
            return {"ok": False, "reason": "session_filter_blocked", "symbol": symbol}
        
        if symbol in getattr(self, "blocked_symbols", set()):
            print(f"[EXECUTOR] BLOCK symbol_blocked symbol={symbol}")
            return {
                "ok": False,
                "reason": "symbol_blocked",
                "symbol": symbol,
            }
        
        if self.enable_spread_filter:
            sym_l = symbol.lower()

            # Use price spread for Boom / Crash indices
            if "boom" in sym_l or "crash" in sym_l:
                tick_now = self.mt5.symbol_info_tick(symbol)
                if tick_now is None:
                    return {"ok": False, "reason": "no_tick", "symbol": symbol}

                spread_price = float(tick_now.ask) - float(tick_now.bid)
                max_spread_price = 3.0

                if spread_price > max_spread_price:
                    print(f"[EXECUTOR] BLOCK spread_price_too_high symbol={symbol} spread_price={spread_price}")
                    return {"ok": False,"reason": "spread_price_too_high", "symbol": symbol, "spread_price": spread_price}

            else:
                sp = self._spread_points(symbol)
                if sp is None:
                    return {"ok": False, "reason": "spread_unknown", "symbol": symbol}

                if sp > int(self.max_spread_points):
                    print(f"[EXECUTOR] BLOCK spread_too_high symbol={symbol} sp={sp} max={self.max_spread_points}")
                    return {"ok": False, "reason": "spread_too_high", "symbol": symbol, "spread_points": sp, "max_spread_points": int(self.max_spread_points)}

        # -------------------------
        # LOT LOGIC (fixed)
        # -------------------------
        lot_req = requested_lot

        # 1) Fixed lot override (checkbox)
        if self.force_symbol_fixed_lot:
            fixed = self._fixed_lot_for_symbol(symbol)
            if fixed is not None:
                lot_req = float(fixed)

        # 2) Global minimum lot
        if self.min_allowed_lot > 0:
            lot_req = max(lot_req, float(self.min_allowed_lot))

        # 3) Per-symbol minimum lot (Boom indices)
        min_lot = self._min_allowed_lot_for_symbol(symbol)
        if min_lot > 0:
            lot_req = max(lot_req, float(min_lot))

        lot = self._normalize_volume(symbol, lot_req)

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        price = float(tick.ask) if action == "BUY" else float(tick.bid)

        info = self.mt5.symbol_info(symbol)
        if info is None:
            return {"ok": False, "reason": "no_symbol_info", "symbol": symbol}

        # -------------------------
        # STOP / LIMIT ENFORCEMENT
        # -------------------------
        sl, tp = self._apply_fixed_sl_tp_offset(symbol, action, price, sl, tp, info)
        sl, tp = self._adjust_sl_tp_to_stops(info, action, price, sl, tp)   

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": 0.0 if sl in (None, 0, 0.0) else float(sl),
            "tp": 0.0 if tp in (None, 0, 0.0) else float(tp),
            "deviation": deviation,
            "magic": self.magic,
            "comment": strategy_comment or self.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": getattr(mt5, "ORDER_FILLING_FOK", 1),
        }

        attempts = max(1, self.max_retries + 1)
        last_result = None
        for i in range(attempts):
            result = mt5.order_send(request)
            last_result = result
            retcode = int(getattr(result, "retcode", -1) or -1) if result is not None else -1
            if result is not None and retcode in {
                int(mt5.TRADE_RETCODE_DONE),
                int(mt5.TRADE_RETCODE_PLACED),
                int(mt5.TRADE_RETCODE_DONE_PARTIAL),
            }:
                return {
                    "ok": True,
                    "symbol": symbol,
                    "action": action,
                    "volume": lot,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "event_time": datetime.now(timezone.utc),
                    "retcode": retcode,
                    "order_ticket": int(getattr(result, "order", 0) or 0) or None,
                    "deal_ticket": int(getattr(result, "deal", 0) or 0) or None,
                    "position_id": int(getattr(result, "order", 0) or 0) or None,
                    "strategy_name": strategy_name,
                    "comment": strategy_comment,
                    "raw": result,
                }
            if i + 1 < attempts:
                time.sleep(max(0, self.retry_delay_ms) / 1000.0)

        return {
            "ok": False,
            "symbol": symbol,
            "action": action,
            "volume": lot,
            "price": price,
            "sl": sl,
            "tp": tp,
            "retcode": int(getattr(last_result, "retcode", -1) or -1) if last_result is not None else -1,
            "comment": getattr(last_result, "comment", "") if last_result is not None else "",
            "reason": "order_send_failed",
            "raw": last_result,
        }
