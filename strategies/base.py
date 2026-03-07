from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Union

import pandas as pd

# Backward-compatible output type used elsewhere in the codebase / GUI.
StrategyOutput = Dict[str, Any]

class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass(frozen=True)
class StrategyResult:
    """Strict strategy evaluation output.

    Invariants:
      - name: non-empty string
      - signal: BUY/SELL/HOLD
      - confidence: float in [0.0, 1.0]
      - meta: dict (always present)
    """
    name: str
    signal: Signal
    confidence: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> StrategyOutput:
        return {
            "name": self.name,
            "signal": self.signal.value,
            "confidence": float(self.confidence),
            "meta": dict(self.meta) if self.meta is not None else {},
        }

def _coerce_signal(v: Any) -> Signal:
    if isinstance(v, Signal):
        return v
    s = str(v or "HOLD").upper()
    if s == "BUY":
        return Signal.BUY
    if s == "SELL":
        return Signal.SELL
    return Signal.HOLD

def _coerce_confidence(v: Any) -> float:
    try:
        c = float(v)
    except Exception:
        c = 0.0
    if c < 0.0:
        return 0.0
    if c > 1.0:
        return 1.0
    return c

def normalize_result(
    raw: Union[StrategyResult, Mapping[str, Any], None],
    *,
    fallback_name: str,
) -> StrategyResult:
    """Normalize a raw strategy output into a strict StrategyResult."""
    if raw is None:
        return StrategyResult(name=fallback_name, signal=Signal.HOLD, confidence=0.0, meta={"reason": "no_output"})

    if isinstance(raw, StrategyResult):
        name = str(raw.name or fallback_name).strip() or fallback_name
        return StrategyResult(
            name=name,
            signal=_coerce_signal(raw.signal),
            confidence=_coerce_confidence(raw.confidence),
            meta=dict(raw.meta) if raw.meta is not None else {},
        )

    # Mapping/dict-like
    try:
        name = str(raw.get("name") or fallback_name).strip() or fallback_name  # type: ignore[attr-defined]
        sig = raw.get("signal", "HOLD")  # type: ignore[attr-defined]
        conf = raw.get("confidence", 0.0)  # type: ignore[attr-defined]
        meta = raw.get("meta", {})  # type: ignore[attr-defined]
    except Exception:
        # If it's not dict-like, hard fail to HOLD
        return StrategyResult(name=fallback_name, signal=Signal.HOLD, confidence=0.0, meta={"reason": "bad_output_type"})

    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        meta = {"meta_raw": meta}

    return StrategyResult(
        name=name,
        signal=_coerce_signal(sig),
        confidence=_coerce_confidence(conf),
        meta=meta,
    )

class Strategy(ABC):
    """Base class for all strategies.

    Implement `_evaluate(...)` and return either:
      - StrategyResult (preferred), or
      - dict-like {name, signal, confidence, meta} (will be normalized)
    """

    name: str

    # Optional regime gating. If set, the ensemble can skip a strategy in mismatched regimes.
    # Example values: {"TREND"}, {"RANGE"}, {"HIGH_VOL"}, {"LOW_VOL"}
    allowed_trends: set[str] | None = None
    allowed_vols: set[str] | None = None

    def is_active(self, regime: dict[str, Any] | None) -> bool:
        """Return True if the strategy should be evaluated in the given regime.

        Default: active everywhere.

        Override for custom logic, or set `allowed_trends` / `allowed_vols`.
        """
        if not regime:
            return True

        trend = str(regime.get("trend", "")).upper()
        vol = str(regime.get("vol", "")).upper()

        if self.allowed_trends is not None and trend and trend not in {t.upper() for t in self.allowed_trends}:
            return False
        if self.allowed_vols is not None and vol and vol not in {v.upper() for v in self.allowed_vols}:
            return False
        return True

    def evaluate(self, data_by_tf: dict[int, pd.DataFrame]) -> StrategyResult:
        raw = self._evaluate(data_by_tf)
        fallback = getattr(self, "name", self.__class__.__name__)
        res = normalize_result(raw, fallback_name=str(fallback))

        # Enforce that returned name stays stable for weighting/perf tracking.
        # If a strategy returned a different name, we override to the declared name but keep a warning.
        declared = str(getattr(self, "name", "") or res.name).strip() or res.name
        if res.name != declared:
            meta = dict(res.meta)
            meta.setdefault("warnings", [])
            try:
                meta["warnings"].append(f"name_override:{res.name}->{declared}")
            except Exception:
                pass
            res = StrategyResult(name=declared, signal=res.signal, confidence=res.confidence, meta=meta)

        return res

    @abstractmethod
    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]) -> Union[StrategyResult, Mapping[str, Any], None]:
        """Strategy-specific implementation."""
        raise NotImplementedError
