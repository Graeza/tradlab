"""Run a bar-close / next-bar-open backtest.

Example:
  python scripts/run_backtest.py --symbol "Boom 1000 Index" --primary-tf 5 --tfs 5 15

Notes:
  - Timeframes are in minutes for CLI convenience. They are mapped to MT5 timeframe constants.
  - Data source is the local SQLite DB (DB_PATH). Make sure you've already collected bars.
  - This script can mirror a large subset of the live GUI settings into the backtest path.
    Live-only controls such as retries, deviation, and real spread checks are accepted for
    auditability but remain informational in OHLC-only backtests.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

# Ensure project root (parent of /scripts) is on sys.path when running as a script.
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

from config.settings import (
    DB_PATH,
    ML_MODEL_PATH,
    ML_CANDIDATES_DIR,
    ML_REQUIRE_SYMBOL_MODEL,
    ML_MIN_CANDIDATE_ACCURACY,
    USE_ML_STRATEGY,
    ENSEMBLE_MIN_CONF,
    ENSEMBLE_MIN_VOTE_GAP,
    STRATEGY_WEIGHTS,
    REGIME_WEIGHT_MULTIPLIERS,
    BACKTEST_STARTING_CASH,
    BACKTEST_WARMUP_BARS,
    BACKTEST_OUT_DIR,
)

from core.database import MarketDatabase
from core.ensemble import EnsembleEngine
from core.ml_model_registry import MLModelRegistry
from strategies.rsi_ema import RSIEMAStrategy
from strategies.breakout import BreakoutStrategy
from strategies.ml_strategy import MLStrategy
from strategies.boom_spike_trend import BoomSpikeTrendStrategy
from strategies.boom_sell_decay import BoomSellDecayStrategy

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


def _parse_bool(s: str | bool) -> bool:
    if isinstance(s, bool):
        return s
    v = str(s).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _csv_to_set(s: str) -> set[str]:
    return {x.strip() for x in str(s or "").split(",") if x.strip()}


def build_strategies(
    *,
    symbol: str,
    primary_tf: int,
    use_rsi: bool = True,
    use_breakout: bool = True,
    use_ml: bool = USE_ML_STRATEGY,
    use_boom: bool = True,
    use_boom_sell: bool = True,
    ml_model_path: str | None = None,
):
    strategies = []

    if use_rsi:
        strategies.append(RSIEMAStrategy())

    if use_breakout:
        strategies.append(BreakoutStrategy())

    if use_boom:
        strategies.append(BoomSpikeTrendStrategy())

    if use_boom_sell:
        strategies.append(BoomSellDecayStrategy())

    chosen_ml_path = str(ml_model_path or "").strip() or None

    if use_ml and USE_ML_STRATEGY:
        registry = MLModelRegistry(
            candidates_dir=str(ML_CANDIDATES_DIR),
            fallback_model_path=str(ML_MODEL_PATH),
            explicit_override_path=chosen_ml_path,
            require_symbol_model=bool(ML_REQUIRE_SYMBOL_MODEL),
            min_candidate_accuracy=float(ML_MIN_CANDIDATE_ACCURACY),
            log=print,
        )
        strategies.append(
            MLStrategy(
                model=None,
                bundle_registry=registry,
                default_symbol=str(symbol),
                default_primary_tf=int(primary_tf),
            )
        )

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
    ap.add_argument("--ml-model-path", type=str, default=str(ML_MODEL_PATH), help="Optional explicit ML model bundle path for this backtest. Defaults to the promoted live model path.")
    ap.add_argument("--start", type=str, default="", help="Optional start date YYYY-MM-DD (UTC)")
    ap.add_argument("--end", type=str, default="", help="Optional end date YYYY-MM-DD (UTC)")
    ap.add_argument("--min-vote-gap", type=float, default=ENSEMBLE_MIN_VOTE_GAP, help="Minimum normalized vote gap required to act")

    # Strategy controls from GUI
    ap.add_argument("--use-rsi", type=_parse_bool, default=True)
    ap.add_argument("--use-breakout", type=_parse_bool, default=True)
    ap.add_argument("--use-ml", type=_parse_bool, default=bool(USE_ML_STRATEGY))
    ap.add_argument("--use-boom", type=_parse_bool, default=True)
    ap.add_argument("--weight-rsi", type=float, default=float(STRATEGY_WEIGHTS.get("RSI_EMA", 1.0)))
    ap.add_argument("--weight-breakout", type=float, default=float(STRATEGY_WEIGHTS.get("BREAKOUT", 1.0)))
    ap.add_argument("--weight-ml", type=float, default=float(STRATEGY_WEIGHTS.get("ML", 1.0)))
    ap.add_argument("--weight-boom", type=float, default=float(STRATEGY_WEIGHTS.get("BOOM_SPIKE_TREND", 1.0)))
    ap.add_argument("--ensemble-min-conf", type=float, default=float(ENSEMBLE_MIN_CONF))
    ap.add_argument("--use-boom-sell", type=_parse_bool, default=True)
    ap.add_argument("--weight-boom-sell", type=float, default=1.45)

    # Risk controls from GUI
    ap.add_argument("--risk-max-pct", type=float, default=1.0)
    ap.add_argument("--risk-min-conf", type=float, default=0.60)
    ap.add_argument("--risk-sl-atr", type=float, default=2.0)
    ap.add_argument("--risk-tp-rr", type=float, default=1.5)
    ap.add_argument("--risk-fallback-sl", type=float, default=0.003)
    ap.add_argument("--risk-max-spread", type=int, default=0)
    ap.add_argument("--risk-base-dev", type=int, default=0)

    # Execution-guard controls from GUI
    ap.add_argument("--allow-new-trades", type=_parse_bool, default=True)
    ap.add_argument("--blocked-symbols", type=str, default="")
    ap.add_argument("--enable-session-filter", type=_parse_bool, default=False)
    ap.add_argument("--session-start-hour", type=int, default=0)
    ap.add_argument("--session-end-hour", type=int, default=24)
    ap.add_argument("--allow-weekends", type=_parse_bool, default=False)
    ap.add_argument("--enable-spread-filter", type=_parse_bool, default=False)
    ap.add_argument("--exec-max-spread", type=int, default=0)
    ap.add_argument("--force-fixed-lot", type=_parse_bool, default=False)
    ap.add_argument("--fixed-sl-tp", type=_parse_bool, default=False)
    ap.add_argument("--sl-tp-offset", type=float, default=0.0)
    ap.add_argument("--enable-trailing-stop", type=_parse_bool, default=False)
    ap.add_argument("--trailing-trigger-rr", type=float, default=1.0)
    ap.add_argument("--trailing-distance-rr", type=float, default=0.5)
    ap.add_argument("--trailing-step-rr", type=float, default=0.10)
    ap.add_argument("--max-retries", type=int, default=0)
    ap.add_argument("--retry-delay-ms", type=int, default=0)
    args = ap.parse_args()

    timeframes: list[int] = []
    tf_values = set(TF_MIN_TO_MT5.values())
    for m in args.tfs:
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

    strategies = build_strategies(
        symbol=args.symbol,
        primary_tf=args.primary_tf,
        use_rsi=bool(args.use_rsi),
        use_breakout=bool(args.use_breakout),
        use_ml=bool(args.use_ml),
        use_boom=bool(args.use_boom),
        use_boom_sell=bool(args.use_boom_sell),
        ml_model_path=args.ml_model_path,
    )
    if not strategies:
        raise SystemExit("No strategies enabled for backtest")

    ensemble = EnsembleEngine(
        strategies,
        weights={
            "RSI_EMA": float(args.weight_rsi),
            "BREAKOUT": float(args.weight_breakout),
            "ML": float(args.weight_ml),
            "BOOM_SPIKE_TREND": float(args.weight_boom),
            "BOOM_SELL_DECAY": float(args.weight_boom_sell),
        },
        min_conf=float(args.ensemble_min_conf),
        min_vote_gap=max(0.0, min(1.0, float(args.min_vote_gap))),
        regime_multipliers=REGIME_WEIGHT_MULTIPLIERS,
    )

    broker = SimBroker(
        starting_cash=args.cash,
        allow_new_trades=bool(args.allow_new_trades),
        blocked_symbols=_csv_to_set(args.blocked_symbols),
        enable_session_filter=bool(args.enable_session_filter),
        session_start_hour=int(args.session_start_hour),
        session_end_hour=int(args.session_end_hour),
        allow_weekends=bool(args.allow_weekends),
        enable_trailing_stop=bool(args.enable_trailing_stop),
        trailing_trigger_rr=float(args.trailing_trigger_rr),
        trailing_distance_rr=float(args.trailing_distance_rr),
        trailing_step_rr=float(args.trailing_step_rr),
    )
    risk = BacktestRiskManager(
        max_risk_pct=float(args.risk_max_pct),
        min_confidence=float(args.risk_min_conf),
        sl_atr_mult=float(args.risk_sl_atr),
        tp_rr=float(args.risk_tp_rr),
        fallback_sl_pct=float(args.risk_fallback_sl),
        max_spread_points=int(args.risk_max_spread),
        base_deviation_points=int(args.risk_base_dev),
        force_symbol_fixed_lot=bool(args.force_fixed_lot),
        boom_crash_fixed_sl_tp=bool(args.fixed_sl_tp),
        boom_crash_sl_tp_offset=float(args.sl_tp_offset),
        enable_spread_filter=bool(args.enable_spread_filter),
        exec_max_spread_points=int(args.exec_max_spread),
    )

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

    extra = {
        "symbol": args.symbol,
        "primary_tf": args.primary_tf,
        "tfs": args.tfs,
        "tag": args.tag,
        "ml_model_path": str(args.ml_model_path),
        "strategy_settings": {
            "use_rsi": bool(args.use_rsi),
            "use_breakout": bool(args.use_breakout),
            "use_ml": bool(args.use_ml),
            "use_boom": bool(args.use_boom),
            "use_boom_sell": bool(args.use_boom_sell),
            "weights": {
                "RSI_EMA": float(args.weight_rsi),
                "BREAKOUT": float(args.weight_breakout),
                "ML": float(args.weight_ml),
                "BOOM_SPIKE_TREND": float(args.weight_boom),
                "BOOM_SELL_DECAY": float(args.weight_boom_sell),
            },
            "ensemble_min_conf": float(args.ensemble_min_conf),
            "ensemble_min_vote_gap": float(args.min_vote_gap),
        },
        "risk_settings": {
            "max_risk_pct": float(args.risk_max_pct),
            "min_confidence": float(args.risk_min_conf),
            "sl_atr_mult": float(args.risk_sl_atr),
            "tp_rr": float(args.risk_tp_rr),
            "fallback_sl_pct": float(args.risk_fallback_sl),
            "max_spread_points": int(args.risk_max_spread),
            "base_deviation_points": int(args.risk_base_dev),
        },
        "execution_guard": {
            "allow_new_trades": bool(args.allow_new_trades),
            "blocked_symbols": sorted(_csv_to_set(args.blocked_symbols)),
            "enable_session_filter": bool(args.enable_session_filter),
            "session_start_hour": int(args.session_start_hour),
            "session_end_hour": int(args.session_end_hour),
            "allow_weekends": bool(args.allow_weekends),
            "enable_spread_filter": bool(args.enable_spread_filter),
            "exec_max_spread": int(args.exec_max_spread),
            "force_fixed_lot": bool(args.force_fixed_lot),
            "fixed_sl_tp": bool(args.fixed_sl_tp),
            "sl_tp_offset": float(args.sl_tp_offset),
            "enable_trailing_stop": bool(args.enable_trailing_stop),
            "trailing_trigger_rr": float(args.trailing_trigger_rr),
            "trailing_distance_rr": float(args.trailing_distance_rr),
            "trailing_step_rr": float(args.trailing_step_rr),
            "max_retries": int(args.max_retries),
            "retry_delay_ms": int(args.retry_delay_ms),
            "notes": [
                "retries and retry_delay_ms are informational only in deterministic backtests",
                "spread filters are accepted for parity tracking but are no-op unless spread data is present in your historical bars",
                "base_deviation_points is informational only in backtests",
            ],
        },
    }

    artifacts = save_backtest_outputs(
        out_dir=args.out,
        equity_curve=res.equity_curve,
        fills=res.fills,
        strategy_outputs=res.strategy_outputs,
        metrics=res.metrics,
        extra=extra,
    )

    log_backtest_experiment(
        tag=args.tag,
        symbol=args.symbol,
        timeframes=[int(x) for x in args.tfs],
        primary_tf=int(args.primary_tf),
        metrics=res.metrics,
        params={
            "cash": args.cash,
            "warmup": args.warmup,
            "limit": args.limit,
            **extra["strategy_settings"],
            **extra["risk_settings"],
            **extra["execution_guard"],
        },
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
