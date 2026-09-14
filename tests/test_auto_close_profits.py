import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


_fake_mt5 = types.ModuleType("MetaTrader5")
_fake_mt5.POSITION_TYPE_BUY = 0
_fake_mt5.POSITION_TYPE_SELL = 1

_fake_worker = types.ModuleType("core.mt5_worker")
_fake_worker.MT5Client = object

with patch.dict(
    sys.modules,
    {"MetaTrader5": _fake_mt5, "core.mt5_worker": _fake_worker},
):
    from trade_executor import TradeExecutor


class _AutoCloseExecutor(TradeExecutor):
    def __init__(self, positions):
        super().__init__(object())
        self.positions = positions
        self.closed_tickets = []

    def _managed_positions(self, symbol=None):
        return self.positions

    def _position_side(self, position):
        return "BUY" if position.type == _fake_mt5.POSITION_TYPE_BUY else "SELL"

    def _close_position_obj(self, position):
        self.closed_tickets.append(position.ticket)
        return True


def _position(ticket, symbol, side, profit):
    position_type = (
        _fake_mt5.POSITION_TYPE_BUY
        if side == "BUY"
        else _fake_mt5.POSITION_TYPE_SELL
    )
    return SimpleNamespace(
        ticket=ticket,
        symbol=symbol,
        type=position_type,
        profit=profit,
    )


class AutoCloseProfitsTests(unittest.TestCase):
    def test_closes_profitable_boom_buys_and_crash_sells(self):
        executor = _AutoCloseExecutor([
            _position(1, "Boom 500 Index", "BUY", 2.0),
            _position(2, "Crash 500 Index", "SELL", 3.0),
        ])

        events = executor.auto_close_profits()

        self.assertEqual(executor.closed_tickets, [1, 2])
        self.assertEqual([event["position_id"] for event in events], [1, 2])

    def test_ignores_opposite_sides_other_symbols_and_threshold(self):
        executor = _AutoCloseExecutor([
            _position(1, "Boom 500 Index", "SELL", 5.0),
            _position(2, "Crash 500 Index", "BUY", 5.0),
            _position(3, "XAUUSD", "SELL", 5.0),
            _position(4, "Crash 1000 Index", "SELL", 1.0),
        ])

        events = executor.auto_close_profits(min_profit=1.0)

        self.assertEqual(executor.closed_tickets, [])
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
