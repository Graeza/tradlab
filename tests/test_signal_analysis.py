import unittest

import pandas as pd

from backtest.signal_analysis import score_signals, slice_closed_bars, summarize_signals


class SignalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.bars = pd.DataFrame({
            "time": [0, 300, 600, 900, 1200],
            "open": [100.0, 101.0, 102.0, 100.0, 104.0],
            "high": [101.0, 103.0, 104.0, 105.0, 106.0],
            "low": [99.0, 100.0, 98.0, 99.0, 103.0],
            "close": [100.0, 102.0, 99.0, 104.0, 105.0],
        })

    def test_scores_buy_signal_from_close_and_next_open(self):
        outputs = pd.DataFrame([{
            "time_s": 0,
            "symbol": "Test Index",
            "strategy": "TEST",
            "signal": "BUY",
            "confidence": 0.8,
            "final_signal": "BUY",
            "final_confidence": 0.7,
            "regime_trend": "UP",
            "regime_vol": "NORMAL",
        }])

        results = score_signals(
            outputs, self.bars, point_size=0.5, horizons=(1, 3), targets=(4,),
        )

        strategy = results[results["strategy"] == "TEST"].iloc[0]
        self.assertEqual(strategy["signal_close_points_1"], 4.0)
        self.assertEqual(strategy["next_open_points_1"], 2.0)
        self.assertEqual(strategy["next_open_points_3"], 6.0)
        self.assertEqual(strategy["mfe_points"], 8.0)
        self.assertEqual(strategy["mae_points"], 6.0)
        self.assertTrue(strategy["hit_plus_4"])
        self.assertEqual(len(results[results["strategy"] == "FINAL_ENSEMBLE"]), 1)

    def test_scores_sell_direction_and_builds_summary(self):
        outputs = pd.DataFrame([{
            "time_s": 300,
            "symbol": "Test Index",
            "strategy": "SELL_TEST",
            "signal": "SELL",
            "confidence": 0.9,
            "final_signal": "HOLD",
            "final_confidence": 0.0,
            "regime_trend": "DOWN",
            "regime_vol": "HIGH",
        }])

        results = score_signals(outputs, self.bars, point_size=1.0, horizons=(1,), targets=(1,))
        summary = summarize_signals(results, horizons=(1,))

        self.assertEqual(results.iloc[0]["next_open_points_1"], 3.0)
        self.assertEqual(summary.iloc[0]["correct_pct_1"], 100.0)
        self.assertEqual(summary.iloc[0]["avg_points_1"], 3.0)

    def test_rejects_invalid_point_size(self):
        with self.assertRaisesRegex(ValueError, "point_size"):
            score_signals(pd.DataFrame([{"time_s": 0}]), self.bars, point_size=0)


class ClosedTimeframeSliceTests(unittest.TestCase):
    def test_only_exposes_higher_timeframe_bars_closed_by_decision(self):
        hourly = pd.DataFrame({"time": [0, 3600, 7200], "close": [1.0, 2.0, 3.0]})

        at_0130 = slice_closed_bars(hourly, decision_time_s=5400, timeframe_seconds=3600)
        at_0200 = slice_closed_bars(hourly, decision_time_s=7200, timeframe_seconds=3600)

        self.assertEqual(at_0130["time"].tolist(), [0])
        self.assertEqual(at_0200["time"].tolist(), [0, 3600])


if __name__ == "__main__":
    unittest.main()
