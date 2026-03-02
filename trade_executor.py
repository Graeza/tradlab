import time
from datetime import datetime, time as dtime
from typing import Optional

import MetaTrader5 as mt5


class TradeExecutor:
    """Executes trades via MT5 with optional execution guardrails.

    Guardrails are intentionally lightweight and UI-configurable:
    - Spread filter (points)
    - Session window filter (local time)
    - Retry logic for transient MT5 failures
    """

    def __init__(
        self,
        enable_spread_filter: bool = True,
        max_spread_points: int = 50,
        enable_session_filter: bool = False,
        session_start_hour: int = 0,
        session_end_hour: int = 24,
        allow_weekends: bool = False,
        max_retries: int = 0,
        retry_delay_ms: int = 250,
    ):
        self.enable_spread_filter = bool(enable_spread_filter)
        self.max_spread_points = int(max_spread_points)

        self.enable_session_filter = bool(enable_session_filter)
        self.session_start_hour = int(session_start_hour)
        self.session_end_hour = int(session_end_hour)
        self.allow_weekends = bool(allow_weekends)

        self.max_retries = int(max_retries)
        self.retry_delay_ms = int(retry_delay_ms)

    def _normalize_volume(self, symbol, volume):
        info = mt5.symbol_info(symbol)

        if info is None:
            raise RuntimeError(f"Symbol info not found for {symbol}")

        min_vol = info.volume_min
        max_vol = info.volume_max
        step = info.volume_step

        # Clamp to min/max
        volume = max(min_vol, min(volume, max_vol))

        # Snap to nearest valid step
        steps = round((volume - min_vol) / step)
        normalized = min_vol + steps * step

        return round(normalized, 2)

    def _within_session(self) -> bool:
        now = datetime.now()
        # weekday(): Mon=0 ... Sun=6
        if not self.allow_weekends and now.weekday() >= 5:
            return False

        if not self.enable_session_filter:
            return True

        sh = max(0, min(23, int(self.session_start_hour)))
        eh = max(0, min(24, int(self.session_end_hour)))

        # Treat end==start as "full day"
        if sh == eh:
            return True

        t = now.time()
        start = dtime(sh, 0, 0)
        end = dtime((eh % 24), 0, 0)

        # Normal window (e.g. 8->17)
        if sh < eh:
            return start <= t < end
        # Overnight window (e.g. 20->6)
        return t >= start or t < end

    def _spread_points(self, symbol) -> Optional[int]:
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            return None
        if not info.visible:
            mt5.symbol_select(symbol, True)
        pt = float(info.point or 0.0)
        if pt <= 0:
            return None
        spread = (float(tick.ask) - float(tick.bid)) / pt
        return int(round(spread))

    def execute(self, params):
        symbol = params["symbol"]
        action = params["action"]
        requested_lot = params["lot_size"]
        sl = params.get("sl")
        tp = params.get("tp")
        deviation = int(params.get("deviation") or 20)

        if not self._within_session():
            return {
                "ok": False,
                "reason": "session_filter_blocked",
                "symbol": symbol,
            }

        if self.enable_spread_filter:
            sp = self._spread_points(symbol)
            if sp is None:
                return {"ok": False, "reason": "spread_unknown", "symbol": symbol}
            if sp > int(self.max_spread_points):
                return {
                    "ok": False,
                    "reason": "spread_too_high",
                    "symbol": symbol,
                    "spread_points": sp,
                    "max_spread_points": int(self.max_spread_points),
                }

        # Normalize lot size to valid symbol volume
        lot = self._normalize_volume(symbol, requested_lot)

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"ok": False, "reason": "no_tick", "symbol": symbol}

        price = tick.ask if action == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": float(sl) if sl is not None else 0.0,
            "tp": float(tp) if tp is not None else 0.0,
            "deviation": deviation,
            "magic": 123456,
            "comment": "ModularBot",
        }

        attempts = 0
        while True:
            attempts += 1
            result = mt5.order_send(request)
            # MT5 result is usually an object with retcode; handle dict-style too
            retcode = getattr(result, "retcode", None)
            if retcode is None:
                # if result is a dict-like or unknown, just return it
                return result

            if retcode == mt5.TRADE_RETCODE_DONE:
                return result

            if attempts > self.max_retries:
                return result

            time.sleep(max(0.0, float(self.retry_delay_ms) / 1000.0))
