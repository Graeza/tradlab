from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional
import json

import numpy as np
import pandas as pd

from backtest.broker import SimBroker
from backtest.contracts import BacktestProgress, CancellationCheck, ProgressCallback, RunPhase
from backtest.metrics import BacktestMetrics, compute_metrics
from backtest.preparation import PreparedBacktestContext, prepare_backtest
from backtest.risk import BacktestRiskManager
from backtest.signal_analysis import score_signals, summarize_signals
from core.ensemble import EnsembleEngine


class BacktestCancelled(RuntimeError):
    """Raised at a deterministic checkpoint when cancellation is requested."""


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    fills: pd.DataFrame
    strategy_outputs: pd.DataFrame
    final_signals: pd.DataFrame
    signal_results: pd.DataFrame
    signal_summary: pd.DataFrame
    metrics: BacktestMetrics
    diagnostics: dict[str, int]
    preparation_timings_s: dict[str, float]


def run_prepared_backtest(
    *,
    context: PreparedBacktestContext,
    ensemble: EnsembleEngine,
    risk: Optional[BacktestRiskManager] = None,
    broker: Optional[SimBroker] = None,
    signal_horizons: tuple[int, ...] = (1, 3, 6, 12),
    signal_targets: tuple[int, ...] = (50, 100, 250, 500),
    run_id: str = "",
    progress_interval_bars: int = 1_000,
    progress_callback: ProgressCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
) -> BacktestResult:
    """Run the deterministic kernel against an already prepared context.

    Existing DataFrame-oriented strategies are supported by a bounded legacy
    adapter: causal slice end-points are precomputed and no historical search,
    feature calculation, regime calculation, or I/O occurs in this loop.
    """

    risk = risk or BacktestRiskManager()
    broker = broker or SimBroker()
    interval = max(1, int(progress_interval_bars))
    n = len(context.primary_time_s)
    start_i = max(context.effective_warmup_bars, 1)
    end_i = n - 2
    if end_i < start_i:
        raise ValueError("Not enough bars for backtest after warmup")

    strategy_records: list[dict] = []
    final_records: list[dict] = []
    diagnostics = {
        "bars_processed": 0,
        "actionable_signals": 0,
        "risk_rejected": 0,
        "spread_rejected": 0,
        "broker_blocked": 0,
        "orders_queued": 0,
    }
    started = perf_counter()
    total_bars = end_i - start_i + 1
    for offset, i in enumerate(range(start_i, end_i + 1), start=1):
        if offset == 1 or offset % interval == 0:
            if cancellation_check is not None and cancellation_check():
                raise BacktestCancelled(f"backtest cancelled at primary index {i}")
            if progress_callback is not None:
                elapsed = perf_counter() - started
                remaining = (elapsed / offset) * (total_bars - offset) if offset else None
                progress_callback(BacktestProgress(
                    run_id=run_id,
                    symbol=context.symbol,
                    phase=RunPhase.SIMULATING,
                    completed_units=offset,
                    total_units=total_bars,
                    elapsed_s=elapsed,
                    estimated_remaining_s=remaining,
                ))

        diagnostics["bars_processed"] += 1
        time_s = int(context.primary_time_s[i])
        spread_points = _optional_spread(context.spreads[i])
        broker.on_bar_open(
            time_s=time_s,
            symbol=context.symbol,
            open_price=float(context.opens[i]),
            spread_points=spread_points,
        )

        data_by_tf = {
            tf: context.features_by_tf[tf].iloc[: int(context.alignment_by_tf[tf][i]) + 1]
            if int(context.alignment_by_tf[tf][i]) >= 0 else context.features_by_tf[tf].iloc[:0]
            for tf in context.timeframes
        }
        regime = dict(context.regimes[i])
        final_signal, outputs = ensemble.run(
            data_by_tf,
            regime=regime,
            context={"symbol": context.symbol, "primary_tf": context.primary_tf},
        )
        final_signal = dict(final_signal or {})
        final_signal["regime"] = regime
        final_records.append({
            "time_s": time_s,
            "symbol": context.symbol,
            "signal": str(final_signal.get("signal", "HOLD")),
            "confidence": float(final_signal.get("confidence", 0.0) or 0.0),
            "vote_gap": float(final_signal.get("vote_gap", 0.0) or 0.0),
            "net_score": float(final_signal.get("net_score", 0.0) or 0.0),
            "regime_trend": str(regime.get("trend", "UNKNOWN")),
            "regime_vol": str(regime.get("vol", "UNKNOWN")),
        })
        for output in outputs:
            strategy_records.append({
                "time_s": time_s,
                "symbol": context.symbol,
                "strategy": str(output.get("name", "")),
                "signal": str(output.get("signal", "HOLD")),
                "confidence": float(output.get("confidence", 0.0) or 0.0),
                "meta": output.get("meta", {}) or {},
                "final_signal": str(final_signal.get("signal", "HOLD")),
                "final_confidence": float(final_signal.get("confidence", 0.0) or 0.0),
                "regime_trend": str(regime.get("trend", "UNKNOWN")),
                "regime_vol": str(regime.get("vol", "UNKNOWN")),
            })

        action = str(final_signal.get("signal") or "HOLD").upper()
        confidence = float(final_signal.get("confidence") or 0.0)
        actionable = action in ("BUY", "SELL") and confidence >= float(getattr(risk, "min_confidence", 0.0))
        if actionable:
            diagnostics["actionable_signals"] += 1
            spread_cap = int(getattr(risk, "exec_max_spread_points", 0) or getattr(risk, "max_spread_points", 0) or 0)
            if bool(getattr(risk, "enable_spread_filter", False)) and spread_points is not None and spread_cap > 0 and spread_points > spread_cap:
                diagnostics["spread_rejected"] += 1

        params = risk.assess(
            signal=final_signal,
            equity=broker.equity,
            entry_price=float(context.opens[i + 1]),
            regime=regime,
            symbol=context.symbol,
            spread_points=spread_points,
            point_size=broker.point_size,
        )
        if params is None and actionable:
            diagnostics["risk_rejected"] += 1
        can_open = broker.can_open_new_trade(time_s=time_s, symbol=context.symbol)
        if params is not None and can_open:
            broker.queue_order(context.symbol, action, params.qty, params.sl, params.tp)
            diagnostics["orders_queued"] += 1
        elif params is not None:
            diagnostics["broker_blocked"] += 1
        broker.on_bar(
            time_s=time_s,
            symbol=context.symbol,
            high=float(context.highs[i]),
            low=float(context.lows[i]),
            close=float(context.closes[i]),
            spread_points=spread_points,
        )

    # A decision on n-2 is executable at n-1; process that bar without making a
    # decision that would require unavailable n data.
    last_i = end_i + 1
    last_spread = _optional_spread(context.spreads[last_i])
    broker.on_bar_open(int(context.primary_time_s[last_i]), context.symbol, float(context.opens[last_i]), last_spread)
    broker.on_bar(
        int(context.primary_time_s[last_i]), context.symbol,
        float(context.highs[last_i]), float(context.lows[last_i]), float(context.closes[last_i]), last_spread,
    )
    if cancellation_check is not None and cancellation_check():
        raise BacktestCancelled("backtest cancelled during final checkpoint")

    equity_curve = pd.DataFrame(broker.equity_curve)
    fills = pd.DataFrame([fill.__dict__ for fill in broker.fills])
    strategy_outputs = pd.DataFrame(strategy_records)
    if "meta" in strategy_outputs:
        strategy_outputs["meta_json"] = strategy_outputs.pop("meta").map(
            lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
    final_signals = pd.DataFrame(final_records)
    signal_results = score_signals(
        strategy_outputs,
        context.bars_by_tf[context.primary_tf],
        point_size=broker.point_size,
        horizons=signal_horizons,
        targets=signal_targets,
    )
    signal_summary = summarize_signals(signal_results, signal_horizons)
    return BacktestResult(
        equity_curve=equity_curve,
        fills=fills,
        strategy_outputs=strategy_outputs,
        final_signals=final_signals,
        signal_results=signal_results,
        signal_summary=signal_summary,
        metrics=compute_metrics(equity_curve, fills),
        diagnostics=diagnostics,
        preparation_timings_s=dict(context.preparation_timings_s),
    )


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
    signal_horizons: tuple[int, ...] = (1, 3, 6, 12),
    signal_targets: tuple[int, ...] = (50, 100, 250, 500),
    progress_interval_bars: int = 1_000,
    progress_callback: ProgressCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
    duplicate_policy: str = "error",
) -> BacktestResult:
    """Compatibility entry point that prepares data before invoking the kernel."""

    context = prepare_backtest(
        symbol=symbol,
        bars_by_tf=bars_by_tf,
        timeframes=timeframes,
        primary_tf=primary_tf,
        warmup_bars=warmup_bars,
        duplicate_policy=duplicate_policy,  # type: ignore[arg-type]
    )
    return run_prepared_backtest(
        context=context,
        ensemble=ensemble,
        risk=risk,
        broker=broker,
        signal_horizons=signal_horizons,
        signal_targets=signal_targets,
        run_id=tag,
        progress_interval_bars=progress_interval_bars,
        progress_callback=progress_callback,
        cancellation_check=cancellation_check,
    )


def _optional_spread(value: float) -> float | None:
    return None if np.isnan(value) else float(value)
