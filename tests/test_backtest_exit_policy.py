"""Check the simulated exit policy against live Boom/Crash execution rules."""

import sys
import subprocess
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backtest.broker import SimBroker
from backtest.risk import BacktestRiskManager


# The policy helpers do not contact a terminal. Keep the Windows-only imports
# scoped, as in test_auto_close_profits, rather than installing an MT5 shim.
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


_TIME = 1_704_067_200  # Monday, 2024-01-01 UTC.
_SYMBOLS = [f"{family} {index} Index" for family in ("Boom", "Crash")
            for index in (300, 500, 600, 900, 1000)]


def _assess(symbol, side, **settings):
    risk = BacktestRiskManager(tp_rr=3.0, fallback_sl_pct=0.04, **settings)
    params = risk.assess(
        {"signal": side, "confidence": 1.0},
        equity=10_000.0,
        entry_price=100.0,
        symbol=symbol,
    )
    assert params is not None
    return params


def _open(broker, symbol, side, *, qty=1.0, spread_points=None):
    sl, tp = (90.0, 110.0) if side == "BUY" else (110.0, 90.0)
    broker.queue_order(symbol, side, qty, sl=sl, tp=tp)
    broker.on_bar_open(_TIME, symbol, 100.0, spread_points=spread_points)
    assert broker.position is not None


@pytest.mark.parametrize("symbol", _SYMBOLS)
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_fixed_stops_match_live_policy_even_with_different_reward_ratio(symbol, side):
    params = _assess(
        symbol, side, boom_crash_fixed_sl_tp=True, boom_crash_sl_tp_offset=5.0,
    )
    executor = TradeExecutor(object(), boom_crash_sl_tp_offset=5.0)
    expected = executor._apply_fixed_sl_tp_offset(
        symbol, side, 100.0, sl=96.0, tp=112.0,
        info=SimpleNamespace(digits=8),
    )

    assert (params.sl, params.tp) == pytest.approx(expected)
    assert abs(params.sl - 100.0) == pytest.approx(5.0)
    assert abs(params.tp - 100.0) == pytest.approx(5.0)
    assert params.fixed_sl_tp_offset == pytest.approx(5.0)


@pytest.mark.parametrize("side", ["BUY", "SELL"])
@pytest.mark.parametrize(
    ("symbol", "enabled", "offset"),
    [
        ("Boom 300 Index", False, 10.0),
        ("Crash 1000 Index", True, 0.0),
        ("Boom 900 Index", True, -10.0),
        ("XAUUSD", True, 10.0),
    ],
)
def test_normal_reward_ratio_remains_when_fixed_policy_is_inactive(symbol, enabled, offset, side):
    params = _assess(
        symbol, side,
        boom_crash_fixed_sl_tp=enabled,
        boom_crash_sl_tp_offset=offset,
    )

    assert abs(params.sl - 100.0) == pytest.approx(4.0)
    assert abs(params.tp - 100.0) == pytest.approx(12.0)
    assert params.fixed_sl_tp_offset is None


@pytest.mark.parametrize("symbol", ["Boom 500 Index", "Crash 500 Index"])
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_fixed_stops_are_centered_on_actual_fill_including_entry_costs(symbol, side):
    broker = SimBroker(point_size=0.5, slippage_points=2.0)
    # Decision-time levels are deliberately far from the next open. The
    # fixed-distance metadata must travel with the order to the actual fill.
    params = _assess(
        symbol, side, boom_crash_fixed_sl_tp=True, boom_crash_sl_tp_offset=5.0,
    )
    broker.queue_order(
        symbol, side, params.qty, params.sl, params.tp,
        fixed_sl_tp_offset=params.fixed_sl_tp_offset,
    )
    broker.on_bar_open(_TIME, symbol, 120.0, spread_points=4.0)

    position = broker.position
    assert position is not None
    expected_entry = 122.0 if side == "BUY" else 118.0
    expected_sl = expected_entry - 5.0 if side == "BUY" else expected_entry + 5.0
    expected_tp = expected_entry + 5.0 if side == "BUY" else expected_entry - 5.0
    assert position.entry_price == pytest.approx(expected_entry)
    assert position.sl == pytest.approx(expected_sl)
    assert position.tp == pytest.approx(expected_tp)
    assert position.anchor_sl == pytest.approx(expected_sl)


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_absolute_stops_are_preserved_without_fixed_distance_metadata(side):
    broker = SimBroker(point_size=0.5, slippage_points=2.0)
    sl, tp = (96.0, 112.0) if side == "BUY" else (104.0, 88.0)
    broker.queue_order("Boom 300 Index", side, 1.0, sl=sl, tp=tp)
    broker.on_bar_open(_TIME, "Boom 300 Index", 120.0, spread_points=4.0)

    assert broker.position.sl == sl
    assert broker.position.tp == tp
    assert broker.position.anchor_sl == sl


class _PolicyExecutor(TradeExecutor):
    def __init__(self, symbol, side, profit):
        super().__init__(object())
        self.position = SimpleNamespace(ticket=1, symbol=symbol, side=side, profit=profit)

    def _managed_positions(self, symbol=None):
        return [self.position]

    def _position_side(self, position):
        return position.side

    def _close_position_obj(self, position):
        return True


@pytest.mark.parametrize(
    ("symbol", "side", "close", "threshold"),
    [
        ("Boom 300 Index", "BUY", 102.0, 0.0),
        ("Crash 1000 Index", "SELL", 98.0, 0.0),
        ("bOoM 500 Index", "BUY", 102.0, 0.0),
        ("Boom 600 Index", "SELL", 98.0, 0.0),
        ("Crash 900 Index", "BUY", 102.0, 0.0),
        ("XAUUSD", "BUY", 102.0, 0.0),
        ("Boom 300 Index", "BUY", 100.0, 0.0),
        ("Crash 300 Index", "SELL", 102.0, 0.0),
        ("Boom 1000 Index", "BUY", 102.0, 2.0),
        ("Crash 500 Index", "SELL", 97.0, 2.0),
    ],
)
def test_early_profit_close_eligibility_matches_live_policy(symbol, side, close, threshold):
    profit = close - 100.0 if side == "BUY" else 100.0 - close
    executor = _PolicyExecutor(symbol, side, profit)
    expected_close = bool(executor.auto_close_profits(min_profit=threshold))
    broker = SimBroker(auto_close_profits=True, auto_close_profits_threshold=threshold)
    _open(broker, symbol, side)

    broker.on_bar(_TIME + 300, symbol, high=103.0, low=97.0, close=close)

    assert (broker.position is None) == expected_close
    if expected_close:
        assert broker.fills[-1].reason == "auto_close_profit"
        assert broker.fills[-1].price == close
        assert broker.fills[-1].side == "CLOSE"
    else:
        assert len(broker.fills) == 1


@pytest.mark.parametrize("symbol,side,close", [
    ("Boom 500 Index", "BUY", 102.0),
    ("Crash 500 Index", "SELL", 98.0),
])
def test_early_profit_close_is_disabled_for_legacy_broker_defaults(symbol, side, close):
    broker = SimBroker()
    _open(broker, symbol, side)

    broker.on_bar(_TIME + 300, symbol, high=103.0, low=97.0, close=close)

    assert broker.position is not None
    assert len(broker.fills) == 1


@pytest.mark.parametrize("symbol,side,close", [
    ("Boom 500 Index", "BUY", 102.0),
    ("Crash 500 Index", "SELL", 98.0),
])
@pytest.mark.parametrize("qty,threshold,should_close", [
    (1.0, 2.0, False),
    (3.0, 2.0, True),
    (3.0, 6.0, False),
])
def test_early_close_threshold_uses_position_profit_and_strict_comparison(
    symbol, side, close, qty, threshold, should_close,
):
    broker = SimBroker(auto_close_profits=True, auto_close_profits_threshold=threshold)
    _open(broker, symbol, side, qty=qty)

    broker.on_bar(_TIME + 300, symbol, high=103.0, low=97.0, close=close)

    assert (broker.position is None) == should_close


@pytest.mark.parametrize("symbol,side,close", [
    ("Boom 500 Index", "BUY", 102.0),
    ("Crash 500 Index", "SELL", 98.0),
])
@pytest.mark.parametrize("close_spread,should_close", [(2.0, True), (6.0, False)])
def test_early_close_eligibility_accounts_for_entry_and_closing_spread(
    symbol, side, close, close_spread, should_close,
):
    broker = SimBroker(auto_close_profits=True, point_size=0.5)
    _open(broker, symbol, side, spread_points=2.0)
    # Entry half-spread is 0.5; closing half-spread is either 0.5 or 1.5.
    # The latter consumes the whole two-unit market move, so it must stay open.
    broker.on_bar(
        _TIME + 300, symbol, high=103.0, low=97.0, close=close,
        spread_points=close_spread,
    )

    assert (broker.position is None) == should_close


@pytest.mark.parametrize("symbol,side,slippage,close,threshold", [
    ("Boom 500 Index", "BUY", 0.2, 101.3, 0.9),
    ("Crash 500 Index", "SELL", 0.2, 98.7, 0.9),
    ("Boom 500 Index", "BUY", 0.0, 100.2, 0.0),
    ("Crash 500 Index", "SELL", 0.0, 99.8, 0.0),
])
@pytest.mark.parametrize("extra_profit", [0.0, 0.000001])
@pytest.mark.parametrize("qty", [1.0, 200.0])
def test_decimal_rounding_does_not_turn_threshold_equality_into_profit(
    symbol, side, slippage, close, threshold, extra_profit, qty,
):
    broker = SimBroker(
        auto_close_profits=True,
        auto_close_profits_threshold=threshold * qty,
        point_size=1.0,
        default_spread_points=0.2,
        slippage_points=slippage,
    )
    _open(broker, symbol, side, qty=qty)
    close += extra_profit if side == "BUY" else -extra_profit

    broker.on_bar(_TIME + 300, symbol, high=102.0, low=98.0, close=close)

    assert (broker.position is None) == (extra_profit > 0)


@pytest.mark.parametrize("symbol,side,close", [
    ("Boom 500 Index", "BUY", 99.0),
    ("Crash 500 Index", "SELL", 101.0),
])
def test_early_close_uses_candle_close_without_assuming_a_profitable_intrabar_fill(symbol, side, close):
    broker = SimBroker(auto_close_profits=True)
    _open(broker, symbol, side)

    broker.on_bar(_TIME + 300, symbol, high=109.0, low=91.0, close=close)

    assert broker.position is not None
    assert len(broker.fills) == 1


@pytest.mark.parametrize(
    ("symbol", "side", "high", "low", "close", "expected_reason", "expected_fill"),
    [
        ("Boom 500 Index", "BUY", 105.0, 89.0, 102.0, "stop", 90.0),
        ("Boom 500 Index", "BUY", 111.0, 95.0, 102.0, "takeprofit", 110.0),
        ("Boom 500 Index", "BUY", 111.0, 89.0, 102.0, "stop", 90.0),
        ("Crash 500 Index", "SELL", 111.0, 95.0, 98.0, "stop", 110.0),
        ("Crash 500 Index", "SELL", 105.0, 89.0, 98.0, "takeprofit", 90.0),
        ("Crash 500 Index", "SELL", 111.0, 89.0, 98.0, "stop", 110.0),
    ],
)
def test_protective_exits_take_priority_over_profitable_candle_close(
    symbol, side, high, low, close, expected_reason, expected_fill,
):
    broker = SimBroker(auto_close_profits=True)
    _open(broker, symbol, side)

    broker.on_bar(_TIME + 300, symbol, high=high, low=low, close=close)

    assert broker.position is None
    assert broker.fills[-1].reason == expected_reason
    assert broker.fills[-1].price == expected_fill


@pytest.mark.parametrize("symbol,side,close,expected_entry,expected_exit", [
    ("Boom 500 Index", "BUY", 104.0, 101.5, 101.0),
    ("Crash 500 Index", "SELL", 96.0, 98.5, 99.0),
])
def test_early_close_fills_charge_slippage_and_commission(
    symbol, side, close, expected_entry, expected_exit,
):
    broker = SimBroker(
        auto_close_profits=True,
        auto_close_profits_threshold=0.25,
        point_size=0.5,
        slippage_points=2.0,
        commission_per_trade=2.0,
    )
    _open(broker, symbol, side, spread_points=2.0)
    # Current marked profit is 0.5, which qualifies before future closing
    # slippage/commission. The eventual fill has -0.5 PnL plus two commissions.
    broker.on_bar(_TIME + 300, symbol, high=105.0, low=95.0, close=close, spread_points=8.0)

    assert broker.position is None
    assert broker.fills[0].price == pytest.approx(expected_entry)
    assert broker.fills[-1].price == pytest.approx(expected_exit)
    assert broker.fills[-1].reason == "auto_close_profit"
    assert broker.cash == pytest.approx(9_995.5)
    assert broker.equity == pytest.approx(broker.cash)


def test_another_symbols_bar_cannot_trigger_early_close():
    broker = SimBroker(auto_close_profits=True)
    _open(broker, "Boom 500 Index", "BUY")

    broker.on_bar(_TIME + 300, "Crash 500 Index", high=105.0, low=95.0, close=102.0)

    assert broker.position is not None
    assert len(broker.fills) == 1


def test_real_engine_forwards_fixed_distance_and_applies_early_close_without_mt5():
    # Isolate the feature-version setting from config.settings' eager MT5 and
    # credential imports. The real preparation/engine/broker/risk code runs in a
    # subprocess so this stub cannot change imports in any other tests.
    script = textwrap.dedent("""
        import sys
        import types
        import numpy as np
        import pandas as pd

        settings = types.ModuleType("config.settings")
        settings.FEATURE_SET_VERSION = 1
        sys.modules["config.settings"] = settings

        from backtest.broker import SimBroker
        from backtest.engine import run_prepared_backtest
        from backtest.preparation import prepare_backtest
        from backtest.risk import BacktestRiskManager

        class OnceEnsemble:
            def __init__(self, side):
                self.side = side
                self.calls = 0

            def run(self, data_by_tf, regime=None, context=None):
                self.calls += 1
                signal = self.side if self.calls == 1 else "HOLD"
                final = {"signal": signal, "confidence": 1.0}
                return final, [{"name": "once", **final}]

        class LegacyBroker(SimBroker):
            # External subclasses using the existing signature must still work
            # for orders that do not request fixed-distance execution.
            def queue_order(self, symbol, side, qty, sl=None, tp=None, reason="signal"):
                super().queue_order(symbol, side, qty, sl, tp, reason)

        for symbol, side in (("Boom 500 Index", "BUY"), ("Crash 500 Index", "SELL")):
            for mode in ("fixed", "normal", "early_close"):
                closes = [100.0, 100.0, 120.5, 120.5, 120.5, 120.5]
                if mode == "early_close":
                    closes[2:] = [125.0 if side == "BUY" else 115.0] * 4
                bars = pd.DataFrame({
                    "time": 1704067200 + np.arange(6, dtype=np.int64) * 300,
                    "open": [100.0, 100.0, 120.0, 120.0, 120.0, 120.0],
                    "high": [101.0, 101.0, 126.0, 126.0, 126.0, 126.0],
                    "low": [99.0, 99.0, 114.0, 114.0, 114.0, 114.0],
                    "close": closes,
                    "spread": [2.0] * 6,
                })
                context = prepare_backtest(
                    symbol=symbol, bars_by_tf={5: bars}, timeframes=(5,),
                    primary_tf=5, warmup_bars=1,
                )
                broker_class = LegacyBroker if mode == "normal" else SimBroker
                broker = broker_class(
                    point_size=1.0, slippage_points=1.0,
                    auto_close_profits=(mode == "early_close"),
                    auto_close_profits_threshold=1.0,
                )
                result = run_prepared_backtest(
                    context=context, ensemble=OnceEnsemble(side),
                    broker=broker,
                    risk=BacktestRiskManager(
                        boom_crash_fixed_sl_tp=(mode != "normal"),
                        boom_crash_sl_tp_offset=10.0,
                        fallback_sl_pct=0.1, tp_rr=3.0,
                    ),
                )
                entry = 122.0 if side == "BUY" else 118.0
                assert result.fills.iloc[0]["price"] == entry
                assert result.diagnostics["orders_queued"] == 1
                if mode == "early_close":
                    assert broker.position is None
                    assert len(result.fills) == 2
                    assert result.fills.iloc[-1]["reason"] == "auto_close_profit"
                    assert result.fills.iloc[-1]["price"] == (123.0 if side == "BUY" else 117.0)
                else:
                    assert broker.position is not None
                    assert len(result.fills) == 1
                    if mode == "fixed":
                        expected = (112.0, 132.0) if side == "BUY" else (128.0, 108.0)
                    else:
                        expected = (108.0, 156.0) if side == "BUY" else (132.0, 84.0)
                    assert (broker.position.sl, broker.position.tp) == expected
                    assert tuple(result.equity_curve.iloc[-1][["pos_sl", "pos_tp"]]) == expected
    """)
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
