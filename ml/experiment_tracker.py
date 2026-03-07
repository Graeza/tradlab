from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from datetime import datetime, timezone


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append a single JSON record to a JSONL file.

    This is intentionally lightweight (no external deps) and is safe to use for
    small/medium experiment logs.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Ensure record is JSON-serializable (best-effort)
    safe = json.loads(json.dumps(record, default=str))

    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")


def build_run_record(
    *,
    model_type: str,
    model_version: str,
    dataset_path: str,
    n_rows: int,
    n_features: int,
    feature_cols: list[str],
    label_col: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, Any],
    feature_set_version: int | None = None,
    feature_set_id: str | None = None,
    output_model_path: str | None = None,
    notes: str | None = None,
    utc_ts: str | None = None,

) -> dict[str, Any]:
    """Standard experiment run record schema."""
    if utc_ts is None:
        utc_ts = datetime.now(timezone.utc).isoformat()
        
    return {
        "utc_ts": utc_ts,
        "model_type": model_type,
        "model_version": model_version,
        "dataset_path": dataset_path,
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "feature_cols": list(feature_cols),
        "label_col": label_col,
        "feature_set_version": feature_set_version,
        "feature_set_id": feature_set_id,
        "params": dict(params),
        "metrics": dict(metrics),
        "output_model_path": output_model_path,
        "notes": notes,
    }
