"""Run a bar-close / next-bar-open backtest.

Example:
  python scripts/run_backtest.py --symbol "Boom 1000 Index" --primary-tf 5 --tfs 5 15

Notes:
  - Timeframes are in minutes for CLI convenience. They are mapped to MT5 timeframe constants.
  - Data source is the local SQLite DB (DB_PATH). Make sure you've already collected bars.
"""

from __future__ import annotations

import argparse
import os

import sys
from datetime import datetime, timezone

# Ensure project root (parent of /scripts) is on sys.path when running as a script.
# Without this, `from core...` fails when executed as `python scripts/run_backtest.py`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# MT5 timeframe constants: prefer MetaTrader5 if available; fallback to core.mt5_worker constants.
try:
    import MetaTrader5 as _mt5
except Exception:
    _mt5 = None

if _mt5 is not None:
    mt5 = _mt5
else:
    from core.mt5_worker import MT5Client as mt5  # constants-only fallback

import joblib

# import MetaTrader5 as mt5

from config.settings import (
    DB_PATH,
    ML_MODEL_PATH,
    USE_ML_STRATEGY,
    ENSEMBLE_MIN_CONF,
    STRATEGY_WEIGHTS,
    REGIME_WEIGHT_MULTIPLIERS,
    BACKTEST_STARTING_CASH,
    BACKTEST_WARMUP_BARS,
    BACKTEST_OUT_DIR,
)

from core.database import MarketDatabase
from core.ensemble import EnsembleEngine
from strategies.rsi_ema import RSIEMAStrategy
from strategies.breakout import BreakoutStrategy
from strategies.ml_strategy import MLStrategy
from strategies.boom_spike_trend import BoomSpikeTrendStrategy

from backtest.data_source import load_bars_from_db
from backtest.broker import SimBroker
from backtest.risk import BacktestRiskManager
from backtest.engine import run_backtest_next_open
from backtest.report import save_backtest_outputs, log_backtest_experiment

TF_MIN_TO_MT5 = {
    1: mt5.TIMEFRAME_M1,
    2: mt5.TIMEFRAME_M2,
    3: mt5.TIMEFRAME_M3,
    4: mt5.TIMEFRAME_M4,
    5: mt5.TIMEFRAME_M5,
    6: mt5.TIMEFRAME_M6,
    10: mt5.TIMEFRAME_M10,
    12: mt5.TIMEFRAME_M12,
    15: mt5.TIMEFRAME_M15,
    20: mt5.TIMEFRAME_M20,
    30: mt5.TIMEFRAME_M30,
    60: mt5.TIMEFRAME_H1,
    240: mt5.TIMEFRAME_H4,
    1440: mt5.TIMEFRAME_D1,
}


def build_strategies():
    strategies = [RSIEMAStrategy(), BreakoutStrategy(),BoomSpikeTrendStrategy()]

    if USE_ML_STRATEGY and os.path.exists(ML_MODEL_PATH):
        bundle = joblib.load(ML_MODEL_PATH)
        if isinstance(bundle, dict) and "model" in bundle:
            strategies.append(
                MLStrategy(
                    bundle["model"],
                    feature_cols=bundle.get("feature_cols"),
                    model_version=bundle.get("model_version") or bundle.get("version"),
                    schema_version=bundle.get("schema_version", 1),
                    strict_schema=bundle.get("strict_schema", True),
                    class_to_signal=bundle.get("class_to_signal"),
                    fillna_value=bundle.get("fillna_value"),
                    feature_set_version=bundle.get("feature_set_version"),
                    feature_set_id=bundle.get("feature_set_id"),
                )
            )
        else:
            strategies.append(MLStrategy(bundle))
    return strategies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, required=True)
    ap.add_argument("--tfs", nargs="+", type=int, required=True, help="Timeframes in minutes, e.g. 5 15")
    ap.add_argument("--primary-tf", type=int, required=True, help="Primary timeframe in minutes")
    ap.add_argument("--limit", type=int, default=200_000, help="Bars per timeframe to load")
    ap.add_argument("--cash", type=float, default=BACKTEST_STARTING_CASH)
    ap.add_argument("--warmup", type=int, default=BACKTEST_WARMUP_BARS)
    ap.add_argument("--out", type=str, default=BACKTEST_OUT_DIR)
    ap.add_argument("--tag", type=str, default="next_open_mvp")
    ap.add_argument("--start", type=str, default="", help="Optional start date YYYY-MM-DD (UTC)")
    ap.add_argument("--end", type=str, default="", help="Optional end date YYYY-MM-DD (UTC)")
    args = ap.parse_args()

    timeframes: list[int] = []
    tf_values = set(TF_MIN_TO_MT5.values())
    for m in args.tfs:
        # Accept either minutes (5,15,60) OR raw MT5 timeframe constants (e.g., 16385, 16388)
        if m in TF_MIN_TO_MT5:
            timeframes.append(TF_MIN_TO_MT5[m])
        elif m in tf_values:
            timeframes.append(int(m))
        else:
            raise SystemExit(f"Unsupported timeframe (minutes or MT5 constant): {m}")

    tf_values = set(TF_MIN_TO_MT5.values())
    if args.primary_tf in TF_MIN_TO_MT5:
        primary_tf = TF_MIN_TO_MT5[args.primary_tf]
    elif args.primary_tf in tf_values:
        primary_tf = int(args.primary_tf)
    else:
        raise SystemExit(f"Unsupported primary timeframe (minutes or MT5 constant): {args.primary_tf}")

    # Optional time window
    def _date_to_utc_s(s: str) -> int:
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    time_min_s = _date_to_utc_s(args.start) if args.start.strip() else None
    time_max_s = _date_to_utc_s(args.end) if args.end.strip() else None

    db = MarketDatabase(DB_PATH)
    data = load_bars_from_db(
        db,
        args.symbol,
        timeframes=timeframes,
        limit=args.limit,
        time_min_s=time_min_s,
        time_max_s=time_max_s,
    )

    strategies = build_strategies()
    ensemble = EnsembleEngine(
        strategies,
        weights=STRATEGY_WEIGHTS,
        min_conf=ENSEMBLE_MIN_CONF,
        regime_multipliers=REGIME_WEIGHT_MULTIPLIERS,
    )

    broker = SimBroker(starting_cash=args.cash)
    risk = BacktestRiskManager()

    res = run_backtest_next_open(
        symbol=args.symbol,
        bars_by_tf=data.bars_by_tf,
        timeframes=timeframes,
        primary_tf=primary_tf,
        ensemble=ensemble,
        risk=risk,
        broker=broker,
        warmup_bars=args.warmup,
        tag=args.tag,
    )

    artifacts = save_backtest_outputs(
        out_dir=args.out,
        equity_curve=res.equity_curve,
        fills=res.fills,
        metrics=res.metrics,
        extra={"symbol": args.symbol, "primary_tf": args.primary_tf, "tfs": args.tfs, "tag": args.tag},
    )

    log_backtest_experiment(
        tag=args.tag,
        symbol=args.symbol,
        timeframes=[int(x) for x in args.tfs],
        primary_tf=int(args.primary_tf),
        metrics=res.metrics,
        params={"cash": args.cash, "warmup": args.warmup, "limit": args.limit},
        artifacts=artifacts,
    )

    print("\n=== Backtest Complete ===")
    print(f"Equity: {res.metrics.start_equity:.2f} -> {res.metrics.end_equity:.2f}")
    print(f"Total return: {res.metrics.total_return*100:.2f}%")
    print(f"Max drawdown: {res.metrics.max_drawdown*100:.2f}%")
    print(f"Trades: {res.metrics.n_trades} | Win rate: {res.metrics.win_rate*100:.1f}% | Avg trade PnL: {res.metrics.avg_trade_pnl:.2f}")
    if res.metrics.profit_factor is not None:
        print(f"Profit factor: {res.metrics.profit_factor:.2f}")
    print(f"Outputs saved to: {args.out}")


if __name__ == "__main__":
    main()
