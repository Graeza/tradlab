from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional
import json
import os

import pandas as pd

from backtest.metrics import BacktestMetrics
from ml.experiment_tracker import append_jsonl
from config.settings import EXPERIMENT_LOG_PATH


def save_backtest_outputs(
    out_dir: str,
    equity_curve: pd.DataFrame,
    fills: pd.DataFrame,
    metrics: BacktestMetrics,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    eq_path = os.path.join(out_dir, "equity_curve.csv")
    fills_path = os.path.join(out_dir, "fills.csv")
    metrics_path = os.path.join(out_dir, "metrics.json")

    equity_curve.to_csv(eq_path, index=False)
    fills.to_csv(fills_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        payload = {"metrics": asdict(metrics), "extra": extra or {}}
        json.dump(payload, f, indent=2)

    paths["equity_curve"] = eq_path
    paths["fills"] = fills_path
    paths["metrics"] = metrics_path
    return paths


def log_backtest_experiment(
    tag: str,
    symbol: str,
    timeframes: list[int],
    primary_tf: int,
    metrics: BacktestMetrics,
    params: Dict[str, Any],
    artifacts: Optional[Dict[str, str]] = None,
) -> None:
    """Append a single JSONL experiment record using the same tracker as ML training."""
    record = {
        "type": "backtest",
        "tag": tag,
        "symbol": symbol,
        "timeframes": [int(x) for x in timeframes],
        "primary_tf": int(primary_tf),
        "params": params,
        "metrics": asdict(metrics),
        "artifacts": artifacts or {},
    }
    append_jsonl(EXPERIMENT_LOG_PATH, record)
