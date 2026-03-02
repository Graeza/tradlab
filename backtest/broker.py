from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Position:
    symbol: str
    side: str  # BUY (long) or SELL (short)
    qty: float
    entry_price: float
    sl: Optional[float] = None
    tp: Optional[float] = None


@dataclass
class Fill:
    time_s: int
    symbol: str
    side: str
    qty: float
    price: float
    reason: str


class SimBroker:
    """Very small deterministic broker.

    Assumptions (MVP):
      - Single position per symbol.
      - Decisions happen on bar close, fills happen at **next bar open**.
      - No partial fills.
      - PnL uses (exit - entry) * qty for BUY, reversed for SELL.
      - Optional commission_per_trade is applied on entry+exit.
    """

    def __init__(
        self,
        starting_cash: float = 10_000.0,
        commission_per_trade: float = 0.0,
    ):
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.equity = float(starting_cash)
        self.commission_per_trade = float(commission_per_trade)

        self.position: Optional[Position] = None
        self.pending_order: Optional[Dict[str, Any]] = None
        self.fills: list[Fill] = []
        self.equity_curve: list[dict] = []  # {time, equity, cash, pos_side, pos_qty, pos_entry}

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
        """Execute any queued order at the given open_price."""
        if not self.pending_order:
            return
        if self.pending_order.get("symbol") != symbol:
            return

        side = self.pending_order["side"]
        qty = float(self.pending_order["qty"])
        sl = self.pending_order.get("sl")
        tp = self.pending_order.get("tp")
        reason = str(self.pending_order.get("reason") or "signal")

        # If there is an existing position:
        # - Same side: ignore
        # - Opposite side: close then open (flip)
        if self.position is not None:
            if self.position.side == side:
                self.pending_order = None
                return
            self.close_position(time_s, open_price, reason="flip")

        self._apply_commission()
        self.position = Position(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=float(open_price),
            sl=float(sl) if sl is not None else None,
            tp=float(tp) if tp is not None else None,
        )
        self.fills.append(Fill(time_s=time_s, symbol=symbol, side=side, qty=qty, price=float(open_price), reason=reason))
        self.pending_order = None

    def on_bar(self, time_s: int, symbol: str, high: float, low: float, close: float) -> None:
        """Update equity and apply SL/TP exits using OHLC (intrabar approximation)."""
        # intrabar SL/TP check (simple):
        if self.position is not None and self.position.symbol == symbol:
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
            else:  # SELL
                if pos.sl is not None and high >= pos.sl:
                    exit_price = float(pos.sl)
                    exit_reason = "stop"
                elif pos.tp is not None and low <= pos.tp:
                    exit_price = float(pos.tp)
                    exit_reason = "takeprofit"

            if exit_price is not None:
                self.close_position(time_s, exit_price, reason=exit_reason or "exit")

        # mark-to-market equity at close
        self._mark_to_market(symbol, float(close))
        self.equity_curve.append(
            {
                "time": int(time_s),
                "equity": float(self.equity),
                "cash": float(self.cash),
                "pos_side": None if self.position is None else self.position.side,
                "pos_qty": 0.0 if self.position is None else float(self.position.qty),
                "pos_entry": None if self.position is None else float(self.position.entry_price),
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
