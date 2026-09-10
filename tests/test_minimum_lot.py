import unittest
from types import SimpleNamespace

from backtest.risk import BacktestRiskManager
from trade_executor import TradeExecutor


class _FakeMT5:
    def __init__(self, minimums):
        self.minimums = minimums

    def symbol_info(self, symbol):
        minimum = self.minimums.get(symbol)
        if minimum is None:
            return None
        return SimpleNamespace(volume_min=minimum)


class MinimumLotTests(unittest.TestCase):
    def test_live_executor_uses_broker_minimum_for_every_symbol_family(self):
        executor = TradeExecutor(
            _FakeMT5({"Crash 500 Index": 0.2, "XAUUSD": 0.01, "Wall Street 30": 0.1})
        )

        self.assertEqual(executor._minimum_lot_for_symbol("Crash 500 Index"), 0.2)
        self.assertEqual(executor._minimum_lot_for_symbol("XAUUSD"), 0.01)
        self.assertEqual(executor._minimum_lot_for_symbol("Wall Street 30"), 0.1)

    def test_unknown_symbol_has_no_minimum_override(self):
        executor = TradeExecutor(_FakeMT5({}))

        self.assertIsNone(executor._minimum_lot_for_symbol("Unknown"))

    def test_backtest_uses_supplied_broker_minimum_for_any_symbol(self):
        risk = BacktestRiskManager(
            min_confidence=0.5,
            force_symbol_fixed_lot=True,
            minimum_lot=0.01,
        )

        result = risk.assess(
            {"signal": "BUY", "confidence": 0.9},
            equity=10_000.0,
            entry_price=100.0,
            symbol="XAUUSD",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.qty, 0.01)


if __name__ == "__main__":
    unittest.main()
