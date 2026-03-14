from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from backtest.broker import SimBroker
from backtest.risk import BacktestRiskManager
from backtest.metrics import compute_metrics, BacktestMetrics
from utils.regime import detect_regime
from core.features import build_features
from core.ensemble import EnsembleEngine


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    fills: pd.DataFrame
    metrics: BacktestMetrics


def _precompute_features(bars_by_tf: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    feats_by_tf: Dict[int, pd.DataFrame] = {}
    for tf, bars in bars_by_tf.items():
        if bars is None or bars.empty:
            feats_by_tf[tf] = pd.DataFrame()
            continue
        feats = build_features(bars)
        feats_by_tf[tf] = feats.reset_index(drop=True)
    return feats_by_tf


def _slice_up_to_time(df: pd.DataFrame, time_s: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df["time"] <= int(time_s)].copy()


def run_backtest_next_open(
    *,
    symbol: str,
    bars_by_tf: Dict[int, pd.DataFrame],
    timeframes: list[int],
    primary_tf: int,
    ensemble: EnsembleEngine,
    risk: Optional[BacktestRiskManager] = None,
    broker: Optional[SimBroker] = None,
    warmup_bars: int = 200,
    tag: str = "mvp",
) -> BacktestResult:
    """Bar-close decision, next-bar-open execution backtest."""
    risk = risk or BacktestRiskManager()
    broker = broker or SimBroker()

    primary_bars = bars_by_tf.get(primary_tf)
    if primary_bars is None or primary_bars.empty:
        raise ValueError(f"No primary bars for tf={primary_tf}")

    feats_by_tf = _precompute_features({tf: bars_by_tf.get(tf, pd.DataFrame()) for tf in timeframes})

    n = len(primary_bars)
    start_i = max(warmup_bars, 1)
    end_i = n - 2
    if end_i <= start_i:
        raise ValueError("Not enough bars for backtest after warmup")

    for i in range(start_i, end_i + 1):
        t = int(primary_bars.loc[i, "time"])
        current_open = float(primary_bars.loc[i, "open"])
        next_open = float(primary_bars.loc[i + 1, "open"])

        data_by_tf: Dict[int, pd.DataFrame] = {}
        for tf in timeframes:
            df = feats_by_tf.get(tf)
            data_by_tf[tf] = _slice_up_to_time(df, t)

        primary_df = data_by_tf.get(primary_tf, pd.DataFrame())
        regime = detect_regime(primary_df) if primary_df is not None and not primary_df.empty else {"trend": "UNKNOWN", "vol": "UNKNOWN"}

        broker.on_bar_open(time_s=t, symbol=symbol, open_price=current_open)

        final_signal, outputs = ensemble.run(data_by_tf, regime=regime)
        if isinstance(final_signal, dict):
            final_signal["regime"] = regime

        params = risk.assess(
            signal=final_signal,
            equity=broker.equity,
            entry_price=next_open,
            regime=regime,
            symbol=symbol,
        )
        if params is not None and broker.can_open_new_trade(time_s=t, symbol=symbol):
            broker.queue_order(
                symbol=symbol,
                side=str(final_signal.get("signal")),
                qty=params.qty,
                sl=params.sl,
                tp=params.tp,
            )

        broker.on_bar(
            time_s=t,
            symbol=symbol,
            high=float(primary_bars.loc[i, "high"]),
            low=float(primary_bars.loc[i, "low"]),
            close=float(primary_bars.loc[i, "close"]),
        )

    equity_curve = pd.DataFrame(broker.equity_curve)
    fills = pd.DataFrame([f.__dict__ for f in broker.fills])
    metrics = compute_metrics(equity_curve, fills)
    return BacktestResult(equity_curve=equity_curve, fills=fills, metrics=metrics)
