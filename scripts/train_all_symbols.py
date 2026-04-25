from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from config.settings import (
    SYMBOL_LIST,
    PRIMARY_TIMEFRAME,
    LABEL_HORIZON_BARS,
    FEATURE_SET_VERSION,
    ML_CANDIDATES_DIR,
)
from scripts.export_dataset import export as export_dataset
from scripts.train_model import main as train_main


def _safe_fs_name(value: str) -> str:
    s = str(value or "").strip()
    s = re.sub(r"[^\w.\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return s or "unknown"


def _candidate_path(*, symbol: str, timeframe: int, model_version: str, schema_version: int) -> str:
    symbol_safe = _safe_fs_name(symbol)
    date_tag = datetime.utcnow().strftime("%Y-%m-%d")
    filename = (
        f"{model_version}"
        f"__{symbol_safe}"
        f"__tf{int(timeframe)}"
        f"__h{int(LABEL_HORIZON_BARS)}"
        f"__fs{int(FEATURE_SET_VERSION)}"
        f"__sv{int(schema_version)}"
        f"__{date_tag}.joblib"
    )
    return str(Path(ML_CANDIDATES_DIR) / symbol_safe / f"tf_{int(timeframe)}" / filename)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch export+train one model per symbol/timeframe.")
    p.add_argument("--symbols", default="", help="Comma-separated symbols. Defaults to SYMBOL_LIST.")
    p.add_argument("--timeframes", default="", help="Comma-separated MT5 timeframe ints. Defaults to PRIMARY_TIMEFRAME.")
    p.add_argument("--csv-dir", default="datasets", help="Where to write per-symbol dataset CSV files.")
    p.add_argument("--model-version", default=None, help="Model version base tag (default: ml_YYYY-MM-DD).")
    p.add_argument("--schema-version", type=int, default=1)
    p.add_argument("--strict-schema", action="store_true")
    p.add_argument("--limit", type=int, default=200000)
    p.add_argument(
        "--validation-policy",
        choices=("holdout", "walk_forward"),
        default="walk_forward",
        help="Validation policy passed to train_model.py (default: walk_forward).",
    )
    p.add_argument("--wf-folds", type=int, default=4, help="Walk-forward folds for train_model.py")
    p.add_argument("--wf-min-train-frac", type=float, default=0.50, help="Minimum training fraction before first fold")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()] or list(SYMBOL_LIST)
    timeframes = [int(x.strip()) for x in str(args.timeframes).split(",") if x.strip()] or [int(PRIMARY_TIMEFRAME)]
    model_version = str(args.model_version or f"ml_{datetime.utcnow().strftime('%Y-%m-%d')}")

    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        for timeframe in timeframes:
            symbol_safe = _safe_fs_name(symbol)
            dataset_path = str(csv_dir / f"{symbol_safe}__tf{int(timeframe)}.csv")
            print(f"[BATCH] Exporting dataset for {symbol} tf={timeframe} -> {dataset_path}")
            export_dataset(
                symbol=symbol,
                timeframe=int(timeframe),
                out_csv=dataset_path,
                limit=int(args.limit),
                horizon_bars=int(LABEL_HORIZON_BARS),
            )

            out_model = _candidate_path(
                symbol=symbol,
                timeframe=int(timeframe),
                model_version=model_version,
                schema_version=int(args.schema_version),
            )
            print(f"[BATCH] Training model for {symbol} tf={timeframe} -> {out_model}")
            import sys

            old_argv = list(sys.argv)
            try:
                sys.argv = [
                    "train_model.py",
                    "--csv", dataset_path,
                    "--model-path", out_model,
                    "--model-version", model_version,
                    "--schema-version", str(int(args.schema_version)),
                    "--symbol", symbol,
                    "--timeframe", str(int(timeframe)),
                    "--horizon-bars", str(int(LABEL_HORIZON_BARS)),
                    "--validation-policy", str(args.validation_policy),
                    "--wf-folds", str(int(args.wf_folds)),
                    "--wf-min-train-frac", str(float(args.wf_min_train_frac)),
                ]
                if bool(args.strict_schema):
                    sys.argv.append("--strict-schema")
                train_main()
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    main()
