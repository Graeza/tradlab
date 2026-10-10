import numpy as np
import pandas as pd
import pytest

from strategies.base import Signal
from strategies.boom_sell_decay import BoomSellDecayStrategy
from strategies.boom_spike_trend import BoomSpikeTrendStrategy
from strategies.crash_spike_trend import (
    CrashBuyRecoveryStrategy,
    CrashSpikeTrendStrategy,
)


SPIKE_STRATEGIES = (BoomSpikeTrendStrategy, CrashSpikeTrendStrategy)
DECAY_STRATEGIES = (BoomSellDecayStrategy, CrashBuyRecoveryStrategy)
STRATEGIES = SPIKE_STRATEGIES + DECAY_STRATEGIES


def _bars(minutes, *, slope=0.2, start=1000.0, with_dt=True):
    close = start + np.arange(240) * slope
    frame = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.04,
            "low": close - 0.24,
            "close": close,
            "RSI": 50.0,
        }
    )
    if with_dt:
        frame["dt"] = pd.date_range(
            "2026-01-01", periods=len(frame), freq=f"{minutes}min", tz="UTC"
        )
    return frame


def _market(*, minute_keys=False, with_dt=True):
    m5 = _bars(5, slope=0.02, with_dt=with_dt)
    m5.loc[m5.index[-1], ["open", "high", "low", "close"]] += 3.0
    return {
        5: m5,
        15: _bars(15, slope=0.01, start=1030.0, with_dt=with_dt),
        60 if minute_keys else 16385: _bars(60, slope=0.4, with_dt=with_dt),
        240 if minute_keys else 16388: _bars(240, slope=0.8, with_dt=with_dt),
    }


def _assert_evaluated(result, strategy_type):
    assert result.meta["h1_trend"] == "bullish"
    if strategy_type in SPIKE_STRATEGIES:
        assert result.signal == Signal.BUY
        assert result.meta["mode"] == "trend_buy"
        if strategy_type is BoomSpikeTrendStrategy:
            assert result.meta["trend_up_h4"] is True
        else:
            assert result.meta["trend_down_h4"] is False
    else:
        assert result.meta["reason"] == "no_setup"


@pytest.mark.parametrize("strategy_type", STRATEGIES)
@pytest.mark.parametrize("minute_keys", (False, True))
@pytest.mark.parametrize("with_dt", (False, True))
def test_extra_timeframes_and_order_preserve_strategy_results(
    strategy_type, minute_keys, with_dt
):
    market = _market(minute_keys=minute_keys, with_dt=with_dt)
    strategy = strategy_type()
    baseline = strategy.evaluate(market)
    _assert_evaluated(baseline, strategy_type)

    extras = {
        1: _bars(1, slope=-0.4, with_dt=with_dt),
        30: _bars(30, slope=-0.4, with_dt=with_dt),
        16408: _bars(1440, slope=-0.4, with_dt=with_dt),
        32769: _bars(10080, slope=-0.4, with_dt=with_dt),
    }
    # Match the configured timeframe order, then reverse and interleave it.
    hour_keys = (60, 240) if minute_keys else (16385, 16388)
    all_frames = {**market, **extras}
    orders = (
        (5, 15, 30, *hour_keys, 16408, 32769, 1),
        tuple(reversed(tuple(all_frames))),
        (32769, hour_keys[1], 30, 15, hour_keys[0], 16408, 1, 5),
    )
    for order in orders:
        result = strategy.evaluate({key: all_frames[key] for key in order})
        assert result == baseline


@pytest.mark.parametrize("strategy_type", STRATEGIES)
def test_gapped_timestamps_do_not_relabel_known_timeframes(strategy_type):
    market = _market()
    # A missing-bar sequence changes observed cadence but not the timeframe.
    market[5]["dt"] = pd.date_range(
        "2026-01-01", periods=240, freq="30min", tz="UTC"
    )
    market[16385]["dt"] = pd.date_range(
        "2026-01-01", periods=240, freq="120min", tz="UTC"
    )
    _assert_evaluated(strategy_type().evaluate(market), strategy_type)


@pytest.mark.parametrize("strategy_type", STRATEGIES)
def test_partial_explicit_override_selects_the_requested_frame(strategy_type):
    market = _market()
    expected = strategy_type().evaluate(market)
    market[9001] = market[16385]
    market[16385] = _bars(60, slope=-0.4)

    result = strategy_type(tf_h1=9001).evaluate(market)

    assert result == expected
    _assert_evaluated(result, strategy_type)


@pytest.mark.parametrize("strategy_type", STRATEGIES)
def test_all_explicit_override_keys_are_honored(strategy_type):
    market = _market()
    expected = strategy_type().evaluate(market)
    custom_keys = {5: 9001, 15: 9002, 16385: 9003, 16388: 9004}
    custom_market = {custom_keys[key]: frame for key, frame in market.items()}

    result = strategy_type(
        tf_m5=9001, tf_m15=9002, tf_h1=9003, tf_h4=9004
    ).evaluate(custom_market)

    assert result == expected


@pytest.mark.parametrize("strategy_type", STRATEGIES)
@pytest.mark.parametrize("explicit_key", (60, 16385, 9001))
def test_missing_explicit_override_does_not_fall_back(strategy_type, explicit_key):
    market = _market(minute_keys=explicit_key == 16385)

    result = strategy_type(tf_h1=explicit_key).evaluate(market)

    assert result.signal == Signal.HOLD
    assert result.meta["reason"] == "missing_timeframes"


@pytest.mark.parametrize("strategy_type", STRATEGIES)
@pytest.mark.parametrize("explicit_key", (60, 16385))
def test_explicit_hour_key_wins_when_both_aliases_exist(strategy_type, explicit_key):
    market = _market(minute_keys=explicit_key == 60)
    expected = strategy_type().evaluate(market)
    alternate_key = 16385 if explicit_key == 60 else 60
    market[alternate_key] = _bars(60, slope=-0.4)

    result = strategy_type(tf_h1=explicit_key).evaluate(market)

    assert result == expected


@pytest.mark.parametrize("strategy_type", STRATEGIES)
def test_native_hour_keys_are_preferred_by_default(strategy_type):
    market = _market()
    expected = strategy_type().evaluate(market)
    market[60] = _bars(60, slope=-0.4)
    market[240] = _bars(240, slope=-0.4)

    result = strategy_type().evaluate(dict(reversed(tuple(market.items()))))

    assert result == expected


@pytest.mark.parametrize("strategy_type", STRATEGIES)
def test_empty_explicit_override_does_not_use_a_valid_alias(strategy_type):
    market = _market()
    market[60] = market[16385].iloc[:0]

    result = strategy_type(tf_h1=60).evaluate(market)

    assert result.signal == Signal.HOLD
    assert result.meta["reason"] == "missing_timeframes"


REQUIRED_FRAMES = tuple(
    (strategy_type, key)
    for strategy_type in STRATEGIES
    for key in ((5, 15, 16385, 16388) if strategy_type in SPIKE_STRATEGIES else (5, 15, 16385))
)


@pytest.mark.parametrize("strategy_type, missing_key", REQUIRED_FRAMES)
@pytest.mark.parametrize("empty", (False, True), ids=("absent", "empty"))
def test_missing_required_timeframe_is_not_replaced_by_extra_data(
    strategy_type, missing_key, empty
):
    market = _market()
    market[30] = _bars(30, slope=-0.4)
    market[16408] = _bars(1440, slope=-0.4)
    if empty:
        market[missing_key] = market[missing_key].iloc[:0]
    else:
        del market[missing_key]

    result = strategy_type().evaluate(market)

    assert result.signal == Signal.HOLD
    assert result.meta["reason"] == "missing_timeframes"


@pytest.mark.parametrize("strategy_type, short_key", REQUIRED_FRAMES)
def test_insufficient_bars_in_required_timeframe_return_hold(strategy_type, short_key):
    market = _market()
    minimum = 30 if short_key == 5 else 60
    market[short_key] = market[short_key].iloc[: minimum - 1]

    result = strategy_type().evaluate(market)

    assert result.signal == Signal.HOLD
    assert result.meta["reason"] == "insufficient_bars"


@pytest.mark.parametrize("strategy_type", (BoomSpikeTrendStrategy, BoomSellDecayStrategy))
@pytest.mark.parametrize("invalid_frame", (None, "not a dataframe"))
def test_non_dataframe_required_frame_returns_hold(strategy_type, invalid_frame):
    market = _market()
    market[16385] = invalid_frame

    result = strategy_type().evaluate(market)

    assert result.signal == Signal.HOLD
    assert result.meta["reason"] == "missing_timeframes"


@pytest.mark.parametrize("strategy_type", DECAY_STRATEGIES)
def test_decay_can_evaluate_without_optional_h4(strategy_type):
    market = _market()
    expected = strategy_type().evaluate(market)
    del market[16388]

    result = strategy_type(use_h4_filter=False).evaluate(market)

    assert result == expected
    _assert_evaluated(result, strategy_type)


@pytest.mark.parametrize("strategy_type", DECAY_STRATEGIES)
@pytest.mark.parametrize("state", ("absent", "empty", "short"))
def test_enabled_decay_h4_filter_requires_available_history(strategy_type, state):
    market = _market()
    if state == "absent":
        del market[16388]
    else:
        market[16388] = market[16388].iloc[: 0 if state == "empty" else 59]

    result = strategy_type(use_h4_filter=True).evaluate(market)

    assert result.signal == Signal.HOLD
    expected_reason = "insufficient_bars" if state == "short" else "missing_timeframes"
    assert result.meta["reason"] == expected_reason


@pytest.mark.parametrize("strategy_type", DECAY_STRATEGIES)
def test_enabled_decay_h4_filter_selects_actual_h4(strategy_type):
    market = _market()
    market[30] = _bars(30, slope=-0.4)

    result = strategy_type(use_h4_filter=True).evaluate(market)

    assert result.meta["h1_trend"] == "bullish"
    assert result.meta["h4_last_close"] == float(market[16388].iloc[-1]["close"])
    assert result.meta["h4_ema50"] is not None
