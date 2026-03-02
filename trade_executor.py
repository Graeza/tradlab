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
        max_spread_points: int = 50,
        enable_session_filter: bool = False,
        session_start_hour: int = 0,
        session_end_hour: int = 24,
        allow_weekends: bool = False,
        max_retries: int = 0,
        retry_delay_ms: int = 250,
        magic: int = 123456,
        comment: str = "ModularBot",
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

        pt = float(getattr(info, "point", 0.0) or 0.0)
        if pt <= 0:
            return None

        spread = (float(tick.ask) - float(tick.bid)) / pt
        return int(round(spread))

    def execute(self, params: dict[str, Any]):
        symbol = str(params["symbol"])
        action = str(params["action"]).upper()
        requested_lot = float(params["lot_size"])
        sl = params.get("sl")
        tp = params.get("tp")
        deviation = int(params.get("deviation") or 20)

        if not self._within_session():
            return {"ok": False, "reason": "session_filter_blocked", "symbol": symbol}

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

        lot = self._normalize_volume(symbol, requested_lot)

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL

        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"ok": False, "reason": "no_tick", "symbol": symbol}

        price = float(tick.ask) if action == "BUY" else float(tick.bid)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": int(order_type),
            "price": float(price),
            "sl": float(sl) if sl is not None else 0.0,
            "tp": float(tp) if tp is not None else 0.0,
            "deviation": int(deviation),
            "magic": int(self.magic),
            "comment": str(self.comment),
        }

        attempts = 0
        while True:
            attempts += 1
            result = self.mt5.order_send(request)

            retcode = getattr(result, "retcode", None)
            if retcode is None:
                return result

            if retcode == mt5.TRADE_RETCODE_DONE:
                return result

            if attempts > self.max_retries:
                return result

            time.sleep(max(0.0, self.retry_delay_ms / 1000.0))
