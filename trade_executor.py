from __future__ import annotations

import time
import math
from datetime import datetime, time as dtime
from typing import Optional, Any

import MetaTrader5 as mt5  # for constants only

from core.mt5_worker import MT5Client


class TradeExecutor:
    """Executes trades via MT5 with optional execution guardrails.

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
        allow_weekends: bool = False,
        max_retries: int = 0,
        retry_delay_ms: int = 250,
        magic: int = 123456,
        comment: str = "ModularBot",
        min_allowed_lot: float = 0.0,   # 0.0 = disabled
        force_symbol_fixed_lot: bool = False,
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

    def _round_price(self, info, price: float) -> float:
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
    
    def execute(self, params: dict[str, Any]):
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

        # Extra spread gate (price-based) for synthetics (kept from your version)
        spread_price = float(tick.ask) - float(tick.bid)
        if "Boom" in symbol or "Crash" in symbol:
            if spread_price > 3.0:   # adjust after observing
                return {
                    "ok": False,
                    "reason": "spread_price_too_high",
                    "symbol": symbol,
                    "spread_price": spread_price,
                }

        if not self._within_session():
            print(f"[EXECUTOR] BLOCK session_filter symbol={symbol}")
            return {"ok": False, "reason": "session_filter_blocked", "symbol": symbol}

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
                    return {
                        "ok": False,
                        "reason": "spread_price_too_high",
                        "symbol": symbol,
                        "spread_price": spread_price,
                    }

            else:
                sp = self._spread_points(symbol)
                if sp is None:
                    return {"ok": False, "reason": "spread_unknown", "symbol": symbol}

                if sp > int(self.max_spread_points):
                    print(f"[EXECUTOR] BLOCK spread_too_high symbol={symbol} sp={sp} max={self.max_spread_points}")
                    return {
                        "ok": False,
                        "reason": "spread_too_high",
                        "symbol": symbol,
                        "spread_points": sp,
                        "max_spread_points": int(self.max_spread_points),
                    }

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
        sl, tp = self._adjust_sl_tp_to_stops(info, action, price, sl, tp)   

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": int(order_type),
            "price": float(price),
            "sl": sl,
            "tp": tp,
            "deviation": int(deviation),
            "magic": int(self.magic),
            "comment": str(self.comment),
        }

        attempts = 0
        while True:
            attempts += 1
            print(f"[EXECUTOR] SENDING request={request}")
            result = mt5.order_send(request)

            retcode = getattr(result, "retcode", None)
            comment = getattr(result, "comment", None)
            request_id = getattr(result, "request_id", None)

            print(f"[EXECUTOR] RESULT retcode={retcode} comment={comment} request_id={request_id}")
            print(f"[EXECUTOR] last_error={self.mt5.last_error()}")

            if retcode is None:
                return result

            if retcode == mt5.TRADE_RETCODE_DONE:
                return result

            if attempts > self.max_retries:
                return result

            time.sleep(max(0.0, self.retry_delay_ms / 1000.0))
