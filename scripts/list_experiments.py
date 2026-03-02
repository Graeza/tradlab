#!/usr/bin/env python3

List ML experiment runs from a JSONL log produced by ml/experiment_tracker.py.

Usage examples:
  python scripts/list_experiments.py
  python scripts/list_experiments.py --limit 20 --sort -accuracy
  python scripts/list_experiments.py --contains xgboost
  python scripts/list_experiments.py --since 2026-01-01
  python scripts/list_experiments.py --path ml/experiments/experiments.jsonl

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _try_parse_dt(s: str) -> Optional[datetime]:
    s = s.strip()
    if not s:
        return None
    # Accept ISO-like strings
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
        # Python 3.11+ ISO parsing
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _load_default_log_path() -> str:
    # Prefer config/settings.py if available, but allow script to run standalone
    try:
        from config.settings import EXPERIMENT_LOG_PATH  # type: ignore
        return str(EXPERIMENT_LOG_PATH)
    except Exception:
        return os.path.join("ml", "experiments", "experiments.jsonl")


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError:
                # Skip malformed lines, but keep going
                continue


def _get_metric(run: Dict[str, Any], key: str) -> Optional[float]:
    metrics = run.get("metrics") or {}
    if isinstance(metrics, dict) and key in metrics:
        try:
            return float(metrics[key])
        except Exception:
            return None
    # Some logs might store metrics at top-level
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
    # Compute widths
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for c in range(cols):
            widths[c] = max(widths[c], len(r[c]))
    sep = "  "
    header_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule_line = sep.join("-" * widths[i] for i in range(cols))
    print(header_line)
    print(rule_line)
    for r in rows:
        print(sep.join(r[i].ljust(widths[i]) for i in range(cols)))


def main() -> int:
    ap = argparse.ArgumentParser(description="List ML experiments from JSONL log.")
    ap.add_argument("--path", default=None, help="Path to experiments.jsonl (defaults to config EXPERIMENT_LOG_PATH).")
    ap.add_argument("--limit", type=int, default=50, help="Max rows to print (default 50).")
    ap.add_argument("--sort", default="-time", help="Sort key: time, accuracy, f1, model. Prefix with '-' for desc. Default -time.")
    ap.add_argument("--since", default=None, help="Filter runs on/after date/time (e.g. 2026-01-01 or 2026-01-01T10:00:00).")
    ap.add_argument("--contains", default=None, help="Case-insensitive substring filter across model/dataset/notes/model_path.")
    args = ap.parse_args()

    path = args.path or _load_default_log_path()
    if not os.path.exists(path):
        print(f"[list_experiments] Log not found: {path}", file=sys.stderr)
        return 2

    since_dt = _try_parse_dt(args.since) if args.since else None
    needle = args.contains.lower() if args.contains else None

    runs: List[Dict[str, Any]] = []
    for run in _iter_jsonl(path):
        # Parse run time
        t = run.get("time") or run.get("timestamp") or run.get("created_at")
        tdt = _try_parse_dt(str(t)) if t is not None else None
        run["_parsed_time"] = tdt
        if since_dt and (tdt is None or tdt < since_dt):
            continue
        if needle:
            hay = " ".join(
                str(run.get(k, "") or "")
                for k in ("model_name", "dataset_path", "notes", "model_path", "run_id")
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
    key = key.lower()

    def sort_fn(r: Dict[str, Any]):
        if key in ("time", "date", "timestamp"):
            return r.get("_parsed_time") or datetime.min
        if key in ("accuracy", "acc"):
            return _get_metric(r, "accuracy") or float("-inf")
        if key in ("f1", "f1_macro", "macro_f1"):
            return _get_metric(r, "f1_macro") or _get_metric(r, "f1") or float("-inf")
        if key in ("model", "model_name"):
            return str(r.get("model_name") or "")
        return str(r.get(key) or "")

    runs.sort(key=sort_fn, reverse=desc)

    # Build display rows
    headers = ["Time", "Model", "Acc", "F1(macro)", "FeatVer", "FeatId", "Dataset", "Model Path"]
    rows: List[List[str]] = []
    for r in runs[: max(0, args.limit)]:
        tdt = r.get("_parsed_time")
        t = tdt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(tdt, datetime) else (r.get("time") or r.get("timestamp") or "")
        model = r.get("model_name") or r.get("model") or ""
        acc = _get_metric(r, "accuracy")
        f1 = _get_metric(r, "f1_macro") or _get_metric(r, "f1")
        fver = r.get("feature_set_version") or (r.get("features") or {}).get("feature_set_version") if isinstance(r.get("features"), dict) else ""
        fid = r.get("feature_set_id") or (r.get("features") or {}).get("feature_set_id") if isinstance(r.get("features"), dict) else ""
        ds = r.get("dataset_path") or ""
        mp = r.get("model_path") or ""
        rows.append([
            _fmt(t, 19),
            _fmt(model, 18),
            _fmt(f"{acc:.4f}" if isinstance(acc, float) else "", 8),
            _fmt(f"{f1:.4f}" if isinstance(f1, float) else "", 8),
            _fmt(fver, 7),
            _fmt(fid, 12),
            _fmt(ds, 28),
            _fmt(mp, 32),
        ])

    _print_table(rows, headers)
    print(f"
Total runs in view: {len(runs)} (showing {min(len(runs), args.limit)})")
    print(f"Log: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
