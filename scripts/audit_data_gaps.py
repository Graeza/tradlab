from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root (parent of /scripts) is on sys.path when running as a script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import DB_PATH, SYMBOL_LIST, TIMEFRAME_LIST, DATA_QUALITY_OUT_DIR


# Minute-style values -> seconds
_MINUTE_TF_SECONDS = {
    1: 60,
    2: 120,
    3: 180,
    4: 240,
    5: 300,
    6: 360,
    10: 600,
    12: 720,
    15: 900,
    20: 1200,
    30: 1800,
    60: 3600,
    120: 7200,
    180: 10800,
    240: 14400,
    360: 21600,
    480: 28800,
    720: 43200,
    1440: 86400,
    10080: 604800,
    43200: 2592000,  # approximate month
}

# MT5 enum values used in your project -> seconds
# This mirrors the way your app/settings can pass MT5 constants directly.
TF_SECONDS = {
    **_MINUTE_TF_SECONDS,

    # Common MT5 constants seen in this project
    16385: 3600,    # TIMEFRAME_H1
    16386: 7200,    # TIMEFRAME_H2
    16387: 10800,   # TIMEFRAME_H3
    16388: 14400,   # TIMEFRAME_H4
    16390: 21600,   # TIMEFRAME_H6
    16392: 28800,   # TIMEFRAME_H8
    16396: 43200,   # TIMEFRAME_H12
    16408: 86400,   # TIMEFRAME_D1
    32769: 604800,  # TIMEFRAME_W1
    49153: 2592000, # TIMEFRAME_MN1 approx
}


@dataclass
class SeriesSummary:
    symbol: str
    timeframe: int
    expected_spacing_s: int
    n_rows: int
    first_time: int | None
    last_time: int | None
    n_gap_events: int
    missing_bars_total: int
    max_gap_bars: int
    max_gap_seconds: int
    completeness_ratio: float | None
    chart_path: str
    gaps_csv_path: str


def _safe_name(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "-", "_"):
            keep.append("_")
        else:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _parse_csv_ints(text: str) -> list[int]:
    vals: list[int] = []
    for part in str(text or "").split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    return vals


def _to_utc_dt(series_or_scalar):
    return pd.to_datetime(series_or_scalar, unit="s", utc=True)


def _load_series(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: int,
    time_min_s: int | None = None,
    time_max_s: int | None = None,
) -> pd.DataFrame:
    q = """
        SELECT time, open, high, low, close, tick_volume, spread, real_volume
        FROM bars
        WHERE symbol = ? AND timeframe = ?
    """
    params: list[object] = [symbol, timeframe]

    if time_min_s is not None:
        q += " AND time >= ?"
        params.append(int(time_min_s))
    if time_max_s is not None:
        q += " AND time <= ?"
        params.append(int(time_max_s))

    q += " ORDER BY time ASC"

    df = pd.read_sql_query(q, conn, params=params)
    if df.empty:
        return df

    df["time"] = df["time"].astype(int)
    df["dt"] = _to_utc_dt(df["time"])
    return df


def _find_gaps(df: pd.DataFrame, expected_s: int) -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return pd.DataFrame(
            columns=[
                "prev_time", "time", "prev_dt", "dt",
                "delta_s", "expected_s", "missing_bars",
                "first_missing_time", "last_missing_time",
                "first_missing_dt", "last_missing_dt",
                "missing_times_utc",
            ]
        )

    work = df[["time", "dt"]].copy()
    work["prev_time"] = work["time"].shift(1)
    work["prev_dt"] = work["dt"].shift(1)
    work["delta_s"] = work["time"] - work["prev_time"]
    work["expected_s"] = int(expected_s)

    gaps = work[work["delta_s"] > expected_s].copy()
    if gaps.empty:
        gaps["missing_bars"] = []
        return gaps

    gaps["missing_bars"] = ((gaps["delta_s"] // expected_s) - 1).astype(int)
    gaps["first_missing_time"] = (gaps["prev_time"] + expected_s).astype(int)
    gaps["last_missing_time"] = (gaps["time"] - expected_s).astype(int)
    gaps["first_missing_dt"] = _to_utc_dt(gaps["first_missing_time"])
    gaps["last_missing_dt"] = _to_utc_dt(gaps["last_missing_time"])

    def _missing_times_for_gap(row: pd.Series) -> str:
        times = range(int(row["first_missing_time"]), int(row["time"]), int(expected_s))
        return "|".join(datetime.fromtimestamp(t, tz=timezone.utc).isoformat() for t in times)

    gaps["missing_times_utc"] = gaps.apply(_missing_times_for_gap, axis=1)

    return gaps[[
        "prev_time", "time", "prev_dt", "dt",
        "delta_s", "expected_s", "missing_bars",
        "first_missing_time", "last_missing_time",
        "first_missing_dt", "last_missing_dt",
        "missing_times_utc",
    ]].reset_index(drop=True)


def _calc_completeness(df: pd.DataFrame, expected_s: int) -> float | None:
    if df.empty:
        return None
    if len(df) == 1:
        return 1.0

    first_t = int(df["time"].iloc[0])
    last_t = int(df["time"].iloc[-1])
    span = last_t - first_t
    if span < 0:
        return None

    expected_rows = int(span // expected_s) + 1
    if expected_rows <= 0:
        return None

    return float(len(df) / expected_rows)


def _plot_series(
    df: pd.DataFrame,
    gaps: pd.DataFrame,
    expected_s: int,
    symbol: str,
    timeframe: int,
    out_png: str,
) -> None:
    fig, axes = plt.subplots(
        2, 1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price = axes[0]
    ax_delta = axes[1]

    if df.empty:
        ax_price.set_title(f"{symbol} | TF={timeframe} | no data")
        ax_price.text(0.5, 0.5, "No bars found", ha="center", va="center", transform=ax_price.transAxes)
        ax_delta.text(0.5, 0.5, "No spacing data", ha="center", va="center", transform=ax_delta.transAxes)
    else:
        ax_price.plot(df["dt"], df["close"], linewidth=1.0)
        ax_price.set_title(f"{symbol} | TF={timeframe} | Close with detected gaps")
        ax_price.set_ylabel("Close")

        for row in gaps.itertuples(index=False):
            ax_price.axvspan(row.prev_dt, row.dt, alpha=0.20)

        if len(df) >= 2:
            d = df[["dt", "time"]].copy()
            d["delta_s"] = d["time"].diff()
            ax_delta.plot(d["dt"], d["delta_s"], linewidth=1.0)
            ax_delta.axhline(expected_s, linestyle="--", linewidth=1.0)
            ax_delta.set_ylabel("Δ sec")
            ax_delta.set_xlabel("UTC time")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _date_to_utc_s(s: str) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def _tf_to_seconds(tf: int) -> int | None:
    tf = int(tf)

    # direct mapping for minute-style values and known MT5 constants
    sec = TF_SECONDS.get(tf)
    if sec is not None:
        return sec

    # fallback: if someone passes a "minutes" value we know implicitly
    if tf > 0 and tf < 10_000:
        # not all integers are valid MT5/minute tfs, so only trust the explicit table above
        return None

    return None

def main():
    ap = argparse.ArgumentParser(description="Audit SQLite OHLCV data for missing-bar gaps and generate charts.")
    ap.add_argument("--db", type=str, default=DB_PATH)
    ap.add_argument("--out", type=str, default=DATA_QUALITY_OUT_DIR)
    ap.add_argument("--symbols", nargs="*", default=None, help="Optional symbol list. Default: all configured symbols with data.")
    ap.add_argument("--timeframes", nargs="*", type=int, default=None, help="Optional timeframe list. Default: config TIMEFRAME_LIST.")
    ap.add_argument("--start", type=str, default="", help="Optional YYYY-MM-DD UTC lower bound.")
    ap.add_argument("--end", type=str, default="", help="Optional YYYY-MM-DD UTC upper bound.")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    charts_dir = os.path.join(out_dir, "charts")
    gaps_dir = os.path.join(out_dir, "gap_details")
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(gaps_dir, exist_ok=True)

    time_min_s = _date_to_utc_s(args.start.strip()) if args.start.strip() else None
    time_max_s = _date_to_utc_s(args.end.strip()) if args.end.strip() else None

    conn = sqlite3.connect(args.db)

    try:
        configured_symbols = list(args.symbols) if args.symbols else list(SYMBOL_LIST)
        configured_tfs = list(args.timeframes) if args.timeframes else list(TIMEFRAME_LIST)

        cur = conn.execute(
            "SELECT DISTINCT symbol, timeframe FROM bars ORDER BY symbol, timeframe"
        )
        existing_pairs = {(str(r[0]), int(r[1])) for r in cur.fetchall()}

        pairs: list[tuple[str, int]] = [
            (sym, tf)
            for sym in configured_symbols
            for tf in configured_tfs
            if (sym, tf) in existing_pairs
        ]

        if not pairs:
            print("[AUDIT] No matching symbol/timeframe pairs found in bars table.")
            return

        summaries: list[SeriesSummary] = []

        for symbol, timeframe in pairs:
            expected_s = _tf_to_seconds(int(timeframe))
            if not expected_s:
                print(f"[AUDIT] Skipping unsupported timeframe mapping: symbol={symbol} tf={timeframe}")
                continue

            print(f"[AUDIT] Checking {symbol} tf={timeframe} ...")
            df = _load_series(
                conn=conn,
                symbol=symbol,
                timeframe=int(timeframe),
                time_min_s=time_min_s,
                time_max_s=time_max_s,
            )

            gaps = _find_gaps(df, expected_s=expected_s)
            completeness = _calc_completeness(df, expected_s=expected_s)

            sym_safe = _safe_name(symbol)
            base = f"{sym_safe}_tf{timeframe}"

            chart_path = os.path.join(charts_dir, f"{base}.png")
            gaps_csv_path = os.path.join(gaps_dir, f"{base}_gaps.csv")

            if gaps.empty:
                pd.DataFrame(
                    columns=[
                        "prev_time", "time", "prev_dt", "dt",
                        "delta_s", "expected_s", "missing_bars",
                        "first_missing_time", "last_missing_time",
                        "first_missing_dt", "last_missing_dt",
                        "missing_times_utc",
                    ]
                ).to_csv(gaps_csv_path, index=False)
            else:
                gaps.to_csv(gaps_csv_path, index=False)

            _plot_series(
                df=df,
                gaps=gaps,
                expected_s=expected_s,
                symbol=symbol,
                timeframe=int(timeframe),
                out_png=chart_path,
            )

            summary = SeriesSummary(
                symbol=symbol,
                timeframe=int(timeframe),
                expected_spacing_s=int(expected_s),
                n_rows=int(len(df)),
                first_time=int(df["time"].iloc[0]) if not df.empty else None,
                last_time=int(df["time"].iloc[-1]) if not df.empty else None,
                n_gap_events=int(len(gaps)),
                missing_bars_total=int(gaps["missing_bars"].sum()) if not gaps.empty else 0,
                max_gap_bars=int(gaps["missing_bars"].max()) if not gaps.empty else 0,
                max_gap_seconds=int(gaps["delta_s"].max()) if not gaps.empty else 0,
                completeness_ratio=float(completeness) if completeness is not None else None,
                chart_path=chart_path,
                gaps_csv_path=gaps_csv_path,
            )
            summaries.append(summary)

        summary_df = pd.DataFrame([asdict(x) for x in summaries])
        summary_csv = os.path.join(out_dir, "summary.csv")
        summary_json = os.path.join(out_dir, "summary.json")

        summary_df.to_csv(summary_csv, index=False)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump([asdict(x) for x in summaries], f, indent=2)

        print("\n=== Data Gap Audit Complete ===")
        print(f"Series checked: {len(summaries)}")
        print(f"Summary CSV:  {summary_csv}")
        print(f"Summary JSON: {summary_json}")
        print(f"Charts dir:   {charts_dir}")
        print(f"Gap CSV dir:  {gaps_dir}")

        if not summary_df.empty:
            worst = summary_df.sort_values(
                by=["missing_bars_total", "n_gap_events", "max_gap_bars"],
                ascending=False
            ).head(10)
            print("\nWorst series by missing bars:")
            print(worst[[
                "symbol", "timeframe", "n_rows", "n_gap_events",
                "missing_bars_total", "max_gap_bars", "completeness_ratio"
            ]].to_string(index=False))

    finally:
        conn.close()


if __name__ == "__main__":
    main()