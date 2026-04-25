from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import json

import numpy as np
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
    strategy_outputs: pd.DataFrame
    metrics: BacktestMetrics
    diagnostics: dict[str, int]


def _precompute_features(bars_by_tf: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    feats_by_tf: Dict[int, pd.DataFrame] = {}
    for tf, bars in bars_by_tf.items():
        if bars is None or bars.empty:
            feats_by_tf[tf] = pd.DataFrame()
            continue
        feats = build_features(bars)
        feats_by_tf[tf] = feats.reset_index(drop=True)
    return feats_by_tf


def _slice_up_to_time(df: pd.DataFrame, time_s: int, time_values: Optional[np.ndarray] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if time_values is None:
        time_values = df["time"].to_numpy(copy=False)
    end_idx = int(np.searchsorted(time_values, int(time_s), side="right"))
    return df.iloc[:end_idx]


def _read_spread_points(bars: pd.DataFrame, i: int) -> Optional[float]:
    if bars is None or bars.empty or "spread" not in bars.columns:
        return None
    v = bars.iloc[int(i)].get("spread")
    if pd.isna(v):
        return None
    return float(v)


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
    feat_times_by_tf: Dict[int, np.ndarray] = {
        tf: feats["time"].to_numpy(copy=False) if feats is not None and not feats.empty and "time" in feats.columns else np.array([], dtype=np.int64)
        for tf, feats in feats_by_tf.items()
    }

    n = len(primary_bars)
    start_i = max(warmup_bars, 1)
    end_i = n - 2
    if end_i <= start_i:
        raise ValueError("Not enough bars for backtest after warmup")
    
    strategy_output_rows: list[dict] = []
    diagnostics: dict[str, int] = {
        "bars_processed": 0,
        "actionable_signals": 0,
        "risk_rejected": 0,
        "spread_rejected": 0,
        "broker_blocked": 0,
        "orders_queued": 0,
    }

    for i in range(start_i, end_i + 1):
        diagnostics["bars_processed"] += 1
        t = int(primary_bars.loc[i, "time"])
        current_open = float(primary_bars.loc[i, "open"])
        next_open = float(primary_bars.loc[i + 1, "open"])
        spread_points = _read_spread_points(primary_bars, i)

        data_by_tf: Dict[int, pd.DataFrame] = {}
        for tf in timeframes:
            df = feats_by_tf.get(tf)
            data_by_tf[tf] = _slice_up_to_time(df, t, feat_times_by_tf.get(tf))

        primary_df = data_by_tf.get(primary_tf, pd.DataFrame())
        regime = detect_regime(primary_df) if primary_df is not None and not primary_df.empty else {"trend": "UNKNOWN", "vol": "UNKNOWN"}

        broker.on_bar_open(time_s=t, symbol=symbol, open_price=current_open, spread_points=spread_points)

        final_signal, outputs = ensemble.run(data_by_tf, regime=regime, context={
                "symbol": symbol,
                "primary_tf": int(primary_tf),
            },
        )
        if isinstance(final_signal, dict):
            final_signal["regime"] = regime

        for out in outputs:
            strategy_output_rows.append({
                "time_s": int(t),
                "symbol": str(symbol),
                "strategy": str(out.get("name", "")),
                "signal": str(out.get("signal", "HOLD")),
                "confidence": float(out.get("confidence", 0.0) or 0.0),
                "meta_json": json.dumps(out.get("meta", {}) or {}, ensure_ascii=False),
                "final_signal": str(final_signal.get("signal", "HOLD")),
                "final_confidence": float(final_signal.get("confidence", 0.0) or 0.0),
                "regime_trend": str(regime.get("trend", "UNKNOWN")),
                "regime_vol": str(regime.get("vol", "UNKNOWN")),
            })

        action = str((final_signal or {}).get("signal") or "HOLD").upper()
        confidence = float((final_signal or {}).get("confidence") or 0.0)
        if action in ("BUY", "SELL") and confidence >= float(getattr(risk, "min_confidence", 0.0)):
            diagnostics["actionable_signals"] += 1
            spread_cap = int(getattr(risk, "exec_max_spread_points", 0) or getattr(risk, "max_spread_points", 0) or 0)
            if bool(getattr(risk, "enable_spread_filter", False)) and spread_points is not None and spread_cap > 0:
                if float(spread_points) > float(spread_cap):
                    diagnostics["spread_rejected"] += 1

        params = risk.assess(
            signal=final_signal,
            equity=broker.equity,
            entry_price=next_open,
            regime=regime,
            symbol=symbol,
            spread_points=spread_points,
            point_size=getattr(broker, "point_size", 1.0),
        )
        if params is None and action in ("BUY", "SELL") and confidence >= float(getattr(risk, "min_confidence", 0.0)):
            diagnostics["risk_rejected"] += 1

        can_open = broker.can_open_new_trade(time_s=t, symbol=symbol)
        if params is not None and can_open:
            broker.queue_order(
                symbol=symbol,
                side=str(final_signal.get("signal")),
                qty=params.qty,
                sl=params.sl,
                tp=params.tp,
            )
            diagnostics["orders_queued"] += 1
        elif params is not None and not can_open:
            diagnostics["broker_blocked"] += 1

        broker.on_bar(
            time_s=t,
            symbol=symbol,
            high=float(primary_bars.loc[i, "high"]),
            low=float(primary_bars.loc[i, "low"]),
            close=float(primary_bars.loc[i, "close"]),
            spread_points=spread_points,
        )

    equity_curve = pd.DataFrame(broker.equity_curve)
    fills = pd.DataFrame([f.__dict__ for f in broker.fills])
    strategy_outputs = pd.DataFrame(strategy_output_rows)
    metrics = compute_metrics(equity_curve, fills)
    return BacktestResult(
        equity_curve=equity_curve,
        fills=fills,
        strategy_outputs=strategy_outputs,
        metrics=metrics,
        diagnostics=diagnostics,
    )
