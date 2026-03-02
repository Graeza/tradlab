#!/usr/bin/env python3
"""List experiment runs from the JSONL log.

The log is appended by:
  - scripts/train_model.py  (ML training)
  - scripts/run_backtest.py (backtests)

Usage examples:
  python scripts/list_experiments.py
  python scripts/list_experiments.py --limit 20 --sort -accuracy
  python scripts/list_experiments.py --type backtest --sort -total_return
  python scripts/list_experiments.py --contains Boom
  python scripts/list_experiments.py --since 2026-01-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


def _try_parse_dt(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _load_default_log_path() -> str:
    try:
        from config.settings import EXPERIMENT_LOG_PATH  # type: ignore

        return str(EXPERIMENT_LOG_PATH)
    except Exception:
        return os.path.join("ml", "experiments", "experiments.jsonl")


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _metric(run: Dict[str, Any], key: str) -> Optional[float]:
    m = run.get("metrics")
    if isinstance(m, dict) and key in m:
        try:
            return float(m[key])
        except Exception:
            return None
    if key in run:
        try:
            return float(run[key])
        except Exception:
            return None
    return None


def _fmt(v: Any, width: int) -> str:
    s = "" if v is None else str(v)
    if len(s) > width:
        return s[: width - 1] + "…"
    return s


def _print_table(rows: List[List[str]], headers: List[str]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    sep = "  "
    print(sep.join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print(sep.join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        print(sep.join(r[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> int:
    ap = argparse.ArgumentParser(description="List ML + backtest experiments from JSONL log.")
    ap.add_argument("--path", default=None, help="Path to experiments.jsonl (defaults to config EXPERIMENT_LOG_PATH).")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--sort", default="-time", help="Sort key. Examples: -time, -accuracy, -total_return")
    ap.add_argument("--since", default=None)
    ap.add_argument("--contains", default=None)
    ap.add_argument("--type", default=None, choices=["ml", "backtest"], help="Filter by record type")
    args = ap.parse_args()

    path = args.path or _load_default_log_path()
    if not os.path.exists(path):
        print(f"[list_experiments] Log not found: {path}", file=sys.stderr)
        return 2

    since_dt = _try_parse_dt(args.since) if args.since else None
    needle = args.contains.lower() if args.contains else None
    type_filter = args.type

    runs: List[Dict[str, Any]] = []
    for run in _iter_jsonl(path):
        rtype = str(run.get("type") or "ml").lower()
        run["_type"] = rtype

        # time parsing: support multiple fields
        t = run.get("utc_ts") or run.get("time") or run.get("timestamp") or run.get("created_at")
        tdt = _try_parse_dt(str(t)) if t is not None else None
        run["_parsed_time"] = tdt

        if type_filter and rtype != type_filter:
            continue
        if since_dt and (tdt is None or tdt < since_dt):
            continue
        if needle:
            hay = " ".join(
                str(run.get(k, "") or "")
                for k in (
                    "model_type",
                    "model_version",
                    "dataset_path",
                    "output_model_path",
                    "tag",
                    "symbol",
                )
            ).lower()
            if needle not in hay:
                continue
        runs.append(run)

    if not runs:
        print("[list_experiments] No runs matched filters.")
        return 0

    sort_key = args.sort.strip()
    desc = sort_key.startswith("-")
    key = sort_key[1:] if desc else sort_key

    def sort_fn(r: Dict[str, Any]):
        if key in ("time", "date", "timestamp"):
            return r.get("_parsed_time") or datetime.min
        if key in ("accuracy", "acc"):
            return _metric(r, "accuracy") or float("-inf")
        if key in ("f1", "f1_macro", "macro_f1"):
            return _metric(r, "f1_macro") or _metric(r, "f1") or float("-inf")
        if key in ("total_return", "return"):
            return _metric(r, "total_return") or float("-inf")
        if key in ("max_drawdown", "dd"):
            return _metric(r, "max_drawdown") or float("-inf")
        return str(r.get(key) or "")

    runs.sort(key=sort_fn, reverse=desc)

    headers = ["Time", "Type", "Name", "Acc", "Ret", "MaxDD", "FeatVer", "FeatId", "Artifacts"]
    rows: List[List[str]] = []
    for r in runs[: max(0, args.limit)]:
        tdt = r.get("_parsed_time")
        t = tdt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(tdt, datetime) else (r.get("utc_ts") or "")
        rtype = r.get("_type")

        if rtype == "backtest":
            name = f"{r.get('symbol','')}:{r.get('tag','')}"
            acc = None
            ret = _metric(r, "total_return")
            mdd = _metric(r, "max_drawdown")
            fver = ""
            fid = ""
        else:
            name = f"{r.get('model_type','')}:{r.get('model_version','')}"
            acc = _metric(r, "accuracy")
            ret = None
            mdd = None
            fver = r.get("feature_set_version")
            fid = r.get("feature_set_id")

        artifacts = r.get("artifacts") if isinstance(r.get("artifacts"), dict) else {}
        art_s = ",".join(sorted(artifacts.keys()))

        rows.append(
            [
                _fmt(t, 19),
                _fmt(rtype, 8),
                _fmt(name, 28),
                _fmt(f"{acc:.4f}" if isinstance(acc, float) else "", 8),
                _fmt(f"{ret*100:.2f}%" if isinstance(ret, float) else "", 8),
                _fmt(f"{mdd*100:.2f}%" if isinstance(mdd, float) else "", 8),
                _fmt(fver, 7),
                _fmt(fid, 12),
                _fmt(art_s, 18),
            ]
        )

    _print_table(rows, headers)
    print(f"\nTotal runs in view: {len(runs)} (showing {min(len(runs), args.limit)})")
    print(f"Log: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
