from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd

from core.database import MarketDatabase
from config.settings import DB_PATH, PRIMARY_TIMEFRAME, LABEL_HORIZON_BARS


def _coerce_time_int64(df: pd.DataFrame, col: str = "time") -> pd.DataFrame:
    """Coerce merge key to int64 (drops rows where conversion fails)."""
    if col not in df.columns:
        raise SystemExit(f"Missing '{col}' column.")
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[col])
    out[col] = out[col].astype("int64")
    return out


def export(
    symbol: str,
    timeframe: int = PRIMARY_TIMEFRAME,
    out_csv: str = "dataset.csv",
    limit: int = 200000,
    horizon_bars: int = LABEL_HORIZON_BARS,
    warn_bad_sign_rate: float = 0.01,
) -> str:
    """Export a supervised training dataset by joining features + labels."""
    db = MarketDatabase(DB_PATH)

    feats = db.load_features(symbol, timeframe, limit=limit)
    if feats.empty:
        raise SystemExit("No features found.")

    labels = pd.read_sql_query(
        "SELECT time, future_return, y_class FROM labels "
        "WHERE symbol=? AND timeframe=? AND horizon_bars=?",
        db.conn,
        params=(symbol, timeframe, int(horizon_bars)),
    )

    if labels.empty:
        raise SystemExit(
            f"No labels found for symbol={symbol} timeframe={timeframe} horizon_bars={horizon_bars}."
        )

    # Normalize merge key dtypes to avoid silent empty/partial merges
    feats = _coerce_time_int64(feats, "time")
    labels = _coerce_time_int64(labels, "time")

    ds = feats.merge(labels, on="time", how="inner")

    if ds.empty:
        raise SystemExit(
            "Merged dataset is empty. Likely a time dtype mismatch, missing labels/features, or horizon mismatch."
        )

    # Metadata columns (useful for debugging downstream training)
    ds["label_horizon_bars"] = int(horizon_bars)
    ds["symbol"] = symbol
    ds["timeframe"] = int(timeframe)

    # Sanity check: y_class direction should broadly match future_return direction
    # (Assumes y_class is -1/0/+1; adjust if your encoding differs.)
    if "future_return" in ds.columns and "y_class" in ds.columns:
        bad_pos = ds[(ds["future_return"] > 0) & (ds["y_class"] < 0)]
        bad_neg = ds[(ds["future_return"] < 0) & (ds["y_class"] > 0)]
        bad_rate = (len(bad_pos) + len(bad_neg)) / max(len(ds), 1)

        if bad_rate > float(warn_bad_sign_rate):
            print(
                f"[WARN] y_class sign disagrees with future_return in ~{bad_rate:.2%} of rows. "
                "This often indicates an off-by-one alignment bug in labeling."
            )

    ds.to_csv(out_csv, index=False)
    print(f"Wrote {len(ds)} rows -> {out_csv}")
    return out_csv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a dataset CSV from the bot's SQLite DB (features + labels)."
    )
    p.add_argument("--symbol", default="EURUSD", help="Symbol to export")
    p.add_argument("--timeframe", type=int, default=PRIMARY_TIMEFRAME, help="MT5 timeframe int")
    p.add_argument("--out", default="dataset.csv", help="Output CSV filename")
    p.add_argument("--limit", type=int, default=200000, help="Max feature rows to load")
    p.add_argument(
        "--horizon",
        type=int,
        default=LABEL_HORIZON_BARS,
        help="Label horizon in bars (must match labels.horizon_bars in DB)",
    )
    p.add_argument(
        "--warn-bad-sign-rate",
        type=float,
        default=0.01,
        help="Warn if y_class sign disagrees with future_return above this fraction (e.g. 0.01 = 1%)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(
        symbol=args.symbol,
        timeframe=args.timeframe,
        out_csv=args.out,
        limit=args.limit,
        horizon_bars=args.horizon,
        warn_bad_sign_rate=args.warn_bad_sign_rate,
    )