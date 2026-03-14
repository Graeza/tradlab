from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, time as dtime
from typing import Optional, Dict, Any


@dataclass
class Position:
    symbol: str
    side: str  # BUY (long) or SELL (short)
    qty: float
    entry_price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    anchor_sl: Optional[float] = None


@dataclass
class Fill:
    time_s: int
    symbol: str
    side: str
    qty: float
    price: float
    reason: str


class SimBroker:
    """Deterministic broker used by the bar-close / next-open backtest.

    This version mirrors several live execution-guard controls:
      - allow_new_trades
      - blocked_symbols
      - session / weekend filters
      - trailing-stop management (bar-based approximation)
    """

    def __init__(
        self,
        starting_cash: float = 10_000.0,
        commission_per_trade: float = 0.0,
        allow_new_trades: bool = True,
        blocked_symbols: Optional[set[str]] = None,
        enable_session_filter: bool = False,
        session_start_hour: int = 0,
        session_end_hour: int = 24,
        allow_weekends: bool = False,
        enable_trailing_stop: bool = False,
        trailing_trigger_rr: float = 1.0,
        trailing_distance_rr: float = 0.5,
        trailing_step_rr: float = 0.10,
    ):
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.equity = float(starting_cash)
        self.commission_per_trade = float(commission_per_trade)

        self.allow_new_trades = bool(allow_new_trades)
        self.blocked_symbols = set(blocked_symbols or set())
        self.enable_session_filter = bool(enable_session_filter)
        self.session_start_hour = int(session_start_hour)
        self.session_end_hour = int(session_end_hour)
        self.allow_weekends = bool(allow_weekends)
        self.enable_trailing_stop = bool(enable_trailing_stop)
        self.trailing_trigger_rr = float(trailing_trigger_rr)
        self.trailing_distance_rr = float(trailing_distance_rr)
        self.trailing_step_rr = max(0.0, float(trailing_step_rr))

        self.position: Optional[Position] = None
        self.pending_order: Optional[Dict[str, Any]] = None
        self.fills: list[Fill] = []
        self.equity_curve: list[dict] = []

    def _dt(self, time_s: int) -> datetime:
        return datetime.fromtimestamp(int(time_s), tz=timezone.utc)

    def _within_session(self, time_s: int) -> bool:
        dt = self._dt(time_s)
        if not self.allow_weekends and dt.weekday() >= 5:
            return False

        if not self.enable_session_filter:
            return True

        sh = max(0, min(23, int(self.session_start_hour)))
        eh = max(0, min(24, int(self.session_end_hour)))
        if sh == eh:
            return True

        t = dt.time()
        start = dtime(sh, 0, 0)
        end = dtime((eh % 24), 0, 0)
        if sh < eh:
            return start <= t < end
        return t >= start or t < end

    def can_open_new_trade(self, *, time_s: int, symbol: str) -> bool:
        if not self.allow_new_trades:
            return False
        if symbol in self.blocked_symbols:
            return False
        return self._within_session(time_s)

    def queue_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        reason: str = "signal",
    ) -> None:
        side = str(side).upper()
        if side not in ("BUY", "SELL"):
            return
        if qty <= 0:
            return
        self.pending_order = {
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "sl": sl,
            "tp": tp,
            "reason": reason,
        }

    def _apply_commission(self) -> None:
        if self.commission_per_trade > 0:
            self.cash -= self.commission_per_trade

    def on_bar_open(self, time_s: int, symbol: str, open_price: float) -> None:
        if not self.pending_order:
            return
        if self.pending_order.get("symbol") != symbol:
            return

        side = self.pending_order["side"]
        qty = float(self.pending_order["qty"])
        sl = self.pending_order.get("sl")
        tp = self.pending_order.get("tp")
        reason = str(self.pending_order.get("reason") or "signal")

        if self.position is not None:
            if self.position.side == side:
                self.pending_order = None
                return
            self.close_position(time_s, open_price, reason="flip")

        self._apply_commission()
        sl_value = float(sl) if sl is not None else None
        self.position = Position(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=float(open_price),
            sl=sl_value,
            tp=float(tp) if tp is not None else None,
            anchor_sl=sl_value,
        )
        self.fills.append(Fill(time_s=time_s, symbol=symbol, side=side, qty=qty, price=float(open_price), reason=reason))
        self.pending_order = None

    def _maybe_update_trailing_stop(self, high: float, low: float, close: float) -> None:
        if not self.enable_trailing_stop or self.position is None:
            return

        pos = self.position
        entry = float(pos.entry_price)
        anchor_sl = pos.anchor_sl
        if anchor_sl is None:
            return

        initial_risk = abs(entry - float(anchor_sl))
        if initial_risk <= 0:
            return

        if pos.side == "BUY":
            best_price = max(float(high), float(close))
            profit_dist = best_price - entry
            if profit_dist <= 0 or profit_dist < self.trailing_trigger_rr * initial_risk:
                return

            candidate_sl = best_price - (self.trailing_distance_rr * initial_risk)
            candidate_sl = max(candidate_sl, entry)
            if pos.sl not in (None, 0, 0.0):
                step_needed = self.trailing_step_rr * initial_risk
                if step_needed > 0 and (candidate_sl - float(pos.sl)) < (step_needed - 1e-12):
                    return
            pos.sl = float(candidate_sl)
            return

        best_price = min(float(low), float(close))
        profit_dist = entry - best_price
        if profit_dist <= 0 or profit_dist < self.trailing_trigger_rr * initial_risk:
            return

        candidate_sl = best_price + (self.trailing_distance_rr * initial_risk)
        candidate_sl = min(candidate_sl, entry)
        if pos.sl not in (None, 0, 0.0):
            step_needed = self.trailing_step_rr * initial_risk
            if step_needed > 0 and (float(pos.sl) - candidate_sl) < (step_needed - 1e-12):
                return
        pos.sl = float(candidate_sl)

    def on_bar(self, time_s: int, symbol: str, high: float, low: float, close: float) -> None:
        if self.position is not None and self.position.symbol == symbol:
            self._maybe_update_trailing_stop(high=float(high), low=float(low), close=float(close))

            pos = self.position
            exit_price = None
            exit_reason = None

            if pos.side == "BUY":
                if pos.sl is not None and low <= pos.sl:
                    exit_price = float(pos.sl)
                    exit_reason = "stop"
                elif pos.tp is not None and high >= pos.tp:
                    exit_price = float(pos.tp)
                    exit_reason = "takeprofit"
            else:
                if pos.sl is not None and high >= pos.sl:
                    exit_price = float(pos.sl)
                    exit_reason = "stop"
                elif pos.tp is not None and low <= pos.tp:
                    exit_price = float(pos.tp)
                    exit_reason = "takeprofit"

            if exit_price is not None:
                self.close_position(time_s, exit_price, reason=exit_reason or "exit")

        self._mark_to_market(symbol, float(close))
        self.equity_curve.append(
            {
                "time": int(time_s),
                "equity": float(self.equity),
                "cash": float(self.cash),
                "pos_side": None if self.position is None else self.position.side,
                "pos_qty": 0.0 if self.position is None else float(self.position.qty),
                "pos_entry": None if self.position is None else float(self.position.entry_price),
                "pos_sl": None if self.position is None else self.position.sl,
                "pos_tp": None if self.position is None else self.position.tp,
            }
        )

    def _mark_to_market(self, symbol: str, price: float) -> None:
        eq = float(self.cash)
        if self.position is not None and self.position.symbol == symbol:
            pos = self.position
            if pos.side == "BUY":
                eq += (price - pos.entry_price) * pos.qty
            else:
                eq += (pos.entry_price - price) * pos.qty
        self.equity = float(eq)

    def close_position(self, time_s: int, price: float, reason: str = "exit") -> None:
        if self.position is None:
            return
        pos = self.position
        self._apply_commission()

        pnl = (price - pos.entry_price) * pos.qty if pos.side == "BUY" else (pos.entry_price - price) * pos.qty
        self.cash += float(pnl)
        self.fills.append(Fill(time_s=time_s, symbol=pos.symbol, side="CLOSE", qty=pos.qty, price=float(price), reason=reason))
        self.position = None
        self.equity = float(self.cash)
