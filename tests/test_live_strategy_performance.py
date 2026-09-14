import os
import tempfile
import unittest

from core.database import MarketDatabase
from core.performance_tracker import PerformanceTracker


class LiveStrategyPerformanceTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.db = MarketDatabase(self.path)

    def tearDown(self):
        self.db.close_thread_connection()
        os.unlink(self.path)

    def test_upserts_compact_rows_for_each_trade_session(self):
        first_session = self.db.create_trade_session("2026-09-14T10:00:00+00:00")
        second_session = self.db.create_trade_session("2026-09-14T11:00:00+00:00")

        self.db.save_live_strategy_performance(first_session, [{
            "name": "CRASH_SPIKE_TREND",
            "n": 3,
            "win_rate": 2 / 3,
            "avg_return": 0.01,
            "avg_abs_ret": 0.02,
            "expectancy": 0.01,
        }])
        self.db.save_live_strategy_performance(first_session, [{
            "name": "CRASH_SPIKE_TREND",
            "n": 4,
            "win_rate": 0.75,
            "avg_return": 0.015,
            "avg_abs_ret": 0.025,
            "expectancy": 0.015,
        }])
        self.db.save_live_strategy_performance(second_session, [{
            "name": "FINAL",
            "n": 1,
            "win_rate": 1.0,
            "avg_return": 0.03,
            "avg_abs_ret": 0.03,
            "expectancy": 0.03,
        }])

        first_rows = self.db.list_live_strategy_performance(first_session)
        second_rows = self.db.list_live_strategy_performance(second_session)

        self.assertEqual(len(first_rows), 1)
        self.assertEqual(first_rows[0]["n"], 4)
        self.assertEqual(first_rows[0]["win_rate"], 0.75)
        self.assertEqual(second_rows[0]["name"], "FINAL")

    def test_compact_tracker_rows_exclude_regime_breakdowns_and_reset(self):
        tracker = PerformanceTracker()
        tracker.stats["TEST"] = {
            "n": 2.0,
            "wins": 1.0,
            "sum_ret": 0.04,
            "sum_abs_ret": 0.06,
        }
        tracker.stats_by_regime["TEST@UP/HIGH"] = {
            "n": 1.0,
            "wins": 1.0,
            "sum_ret": 0.03,
            "sum_abs_ret": 0.03,
        }

        rows = tracker.strategy_summary_rows()

        self.assertEqual({row["name"] for row in rows}, {"FINAL", "TEST"})
        test_row = next(row for row in rows if row["name"] == "TEST")
        self.assertEqual(test_row["avg_return"], 0.02)

        tracker.reset()
        self.assertEqual(tracker.pending_count(), 0)
        self.assertEqual(tracker.strategy_summary_rows()[0]["n"], 0)


if __name__ == "__main__":
    unittest.main()
