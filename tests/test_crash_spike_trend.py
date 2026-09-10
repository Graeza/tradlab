import unittest

import pandas as pd

from strategies.base import StrategyResult, Signal
from strategies.crash_spike_trend import (
    CrashBuyRecoveryStrategy,
    CrashSpikeTrendStrategy,
    _mirror_frame,
)


class _StubBoomStrategy:
    def __init__(self, result: StrategyResult):
        self.result = result
        self.received = None

    def evaluate(self, data_by_tf):
        self.received = data_by_tf
        return self.result


class CrashStrategyMirrorTests(unittest.TestCase):
    def test_mirror_frame_swaps_extremes_and_inverts_rsi(self):
        source = pd.DataFrame(
            {"open": [100.0], "high": [106.0], "low": [97.0], "close": [102.0], "RSI": [28.0]}
        )

        mirrored = _mirror_frame(source)

        self.assertEqual(float(mirrored.loc[0, "open"]), -100.0)
        self.assertEqual(float(mirrored.loc[0, "high"]), -97.0)
        self.assertEqual(float(mirrored.loc[0, "low"]), -106.0)
        self.assertEqual(float(mirrored.loc[0, "close"]), -102.0)
        self.assertEqual(float(mirrored.loc[0, "RSI"]), 72.0)
        pd.testing.assert_frame_equal(source, pd.DataFrame(
            {"open": [100.0], "high": [106.0], "low": [97.0], "close": [102.0], "RSI": [28.0]}
        ))

    def test_spike_strategy_reverses_sell_to_buy_and_translates_metadata(self):
        strategy = CrashSpikeTrendStrategy()
        stub = _StubBoomStrategy(
            StrategyResult(
                "BOOM_SPIKE_TREND",
                Signal.SELL,
                0.81,
                {"mode": "spike_sell", "upper_ratio": 3.2, "near_h1_support": True},
            )
        )
        strategy._boom_strategy = stub

        result = strategy.evaluate({5: pd.DataFrame({"open": [1], "high": [2], "low": [0], "close": [1]})})

        self.assertEqual(result.name, "CRASH_SPIKE_TREND")
        self.assertEqual(result.signal, Signal.BUY)
        self.assertEqual(result.confidence, 0.81)
        self.assertEqual(result.meta["mode"], "spike_buy")
        self.assertEqual(result.meta["lower_ratio"], 3.2)
        self.assertTrue(result.meta["near_h1_resistance"])
        self.assertTrue(result.meta["mirrored_from_boom_strategy"])

    def test_recovery_strategy_reverses_decay_sell_to_buy(self):
        strategy = CrashBuyRecoveryStrategy()
        strategy._boom_strategy = _StubBoomStrategy(
            StrategyResult("BOOM_SELL_DECAY", Signal.SELL, 0.74, {"mode": "boom_sell_decay"})
        )

        result = strategy.evaluate({})

        self.assertEqual(result.name, "CRASH_BUY_RECOVERY")
        self.assertEqual(result.signal, Signal.BUY)
        self.assertEqual(result.meta["mode"], "crash_buy_recovery")

    def test_hold_signal_remains_hold(self):
        strategy = CrashSpikeTrendStrategy()
        strategy._boom_strategy = _StubBoomStrategy(
            StrategyResult("BOOM_SPIKE_TREND", Signal.HOLD, 0.0, {"reason": "no_setup"})
        )

        result = strategy.evaluate({})

        self.assertEqual(result.signal, Signal.HOLD)


if __name__ == "__main__":
    unittest.main()
