import pandas as pd
import pytest

from core.performance_tracker import PerformanceTracker


SYMBOL = "Crash 500 Index"
REGIME = {"trend": "DOWN", "vol": "HIGH"}


def _add_prediction(tracker, bars, signal, *, outputs=None, horizon=1):
    tracker.add_prediction(
        symbol=SYMBOL,
        df_primary=bars,
        horizon_bars=horizon,
        final={"signal": signal, "regime": REGIME},
        outputs=outputs if outputs is not None else [{"name": "TEST", "signal": signal}],
    )


def _rows_by_name(tracker):
    return {row["name"]: row for row in tracker.summary_rows()}


@pytest.mark.parametrize(
    "signal,future_close,expected_return,expected_win_rate",
    [
        ("BUY", 110.0, 0.10, 1.0),
        ("BUY", 90.0, -0.10, 0.0),
        ("SELL", 90.0, 0.10, 1.0),
        ("SELL", 110.0, -0.10, 0.0),
        ("BUY", 100.0, 0.0, 0.0),
        ("SELL", 100.0, 0.0, 0.0),
        ("buy", 110.0, 0.10, 1.0),
        ("sell", 90.0, 0.10, 1.0),
    ],
)
def test_returns_follow_prediction_direction(
    signal, future_close, expected_return, expected_win_rate
):
    bars = pd.DataFrame({"time": [0, 300], "close": [100.0, future_close]})
    tracker = PerformanceTracker()
    _add_prediction(tracker, bars.iloc[:1], signal)

    tracker.update_with_bars(SYMBOL, bars)

    rows = _rows_by_name(tracker)
    assert set(rows) == {"FINAL", "FINAL@DOWN/HIGH", "TEST", "TEST@DOWN/HIGH"}
    for row in rows.values():
        assert row["n"] == 1
        assert row["win_rate"] == expected_win_rate
        assert row["avg_ret"] == pytest.approx(expected_return)
        assert row["expectancy"] == pytest.approx(expected_return)
        assert row["avg_abs_ret"] == pytest.approx(abs(expected_return))
    assert tracker.pending_count() == 0


def test_final_and_strategy_predictions_use_their_own_directions():
    bars = pd.DataFrame({"time": [0, 300], "close": [100.0, 90.0]})
    tracker = PerformanceTracker()
    _add_prediction(
        tracker,
        bars.iloc[:1],
        "SELL",
        outputs=[
            {"name": "BUY_TEST", "signal": "BUY"},
            {"name": "SELL_TEST", "signal": "SELL"},
        ],
    )

    tracker.update_with_bars(SYMBOL, bars)

    rows = tracker.summary_rows()
    expected = {
        "FINAL": 0.10,
        "FINAL@DOWN/HIGH": 0.10,
        "SELL_TEST": 0.10,
        "SELL_TEST@DOWN/HIGH": 0.10,
        "BUY_TEST": -0.10,
        "BUY_TEST@DOWN/HIGH": -0.10,
    }
    assert {row["name"] for row in rows} == set(expected)
    for row in rows:
        assert row["n"] == 1
        assert row["avg_ret"] == pytest.approx(expected[row["name"]])
        assert row["expectancy"] == pytest.approx(expected[row["name"]])
        assert row["win_rate"] == (1.0 if expected[row["name"]] > 0 else 0.0)
    assert all(row["expectancy"] > 0 for row in rows[:4])
    assert all(row["expectancy"] < 0 for row in rows[4:])


def test_average_return_combines_buy_and_sell_predictions():
    bars = pd.DataFrame({"time": [0, 300, 600], "close": [100.0, 90.0, 99.0]})
    tracker = PerformanceTracker()
    _add_prediction(tracker, bars.iloc[:1], "SELL")
    _add_prediction(tracker, bars.iloc[:2], "BUY")

    tracker.update_with_bars(SYMBOL, bars)

    for row in tracker.summary_rows():
        assert row["n"] == 2
        assert row["win_rate"] == 1.0
        assert row["avg_ret"] == pytest.approx(0.10)
        assert row["expectancy"] == pytest.approx(0.10)
        assert row["avg_abs_ret"] == pytest.approx(0.10)


def test_hold_predictions_are_excluded_without_hiding_strategy_results():
    bars = pd.DataFrame({"time": [0, 300], "close": [100.0, 90.0]})
    tracker = PerformanceTracker()
    _add_prediction(
        tracker,
        bars.iloc[:1],
        "HOLD",
        outputs=[
            {"name": "HOLD_TEST", "signal": "HOLD"},
            {"name": "SELL_TEST", "signal": "SELL"},
        ],
    )

    tracker.update_with_bars(SYMBOL, bars)

    rows = _rows_by_name(tracker)
    assert set(rows) == {"FINAL", "SELL_TEST", "SELL_TEST@DOWN/HIGH"}
    assert rows["FINAL"]["n"] == 0
    assert rows["FINAL"]["expectancy"] == 0.0
    assert rows["SELL_TEST"]["n"] == 1
    assert rows["SELL_TEST"]["avg_ret"] == pytest.approx(0.10)


def test_prediction_waits_for_horizon_and_is_scored_once():
    bars = pd.DataFrame({"time": [0, 300, 600], "close": [100.0, 95.0, 90.0]})
    tracker = PerformanceTracker()
    _add_prediction(tracker, bars.iloc[:1], "SELL", horizon=2)

    tracker.update_with_bars(SYMBOL, bars.iloc[:2])

    assert tracker.pending_count() == 1
    assert _rows_by_name(tracker)["FINAL"]["n"] == 0

    tracker.update_with_bars(SYMBOL, bars)
    snapshot = tracker.snapshot()
    assert snapshot["pending_total"] == 0
    for row in snapshot["rows"]:
        assert row["n"] == 1
        assert row["avg_ret"] == pytest.approx(0.10)

    tracker.update_with_bars(SYMBOL, bars)
    assert tracker.snapshot() == snapshot


def test_small_returns_preserve_epsilon_win_classification():
    bars = pd.DataFrame({"time": [0, 300], "close": [100.0, 99.0]})
    tracker = PerformanceTracker(eps=0.02)
    _add_prediction(tracker, bars.iloc[:1], "SELL")

    tracker.update_with_bars(SYMBOL, bars)

    for row in tracker.summary_rows():
        assert row["n"] == 1
        assert row["win_rate"] == 0.0
        assert row["avg_ret"] == pytest.approx(0.01)
        assert row["expectancy"] == pytest.approx(0.01)
