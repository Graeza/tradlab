from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import joblib


def _safe_fs_name(value: str) -> str:
    s = str(value or "").strip()
    s = re.sub(r"[^\w.\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return s or "unknown"


class MLModelRegistry:
    """
    Resolves an ML bundle for a (symbol, timeframe) pair.

    Resolution order:
      1) explicit override path, if provided and exists
      2) latest candidate in models/candidates/<symbol>/tf_<tf>/*.joblib
      3) fallback promoted live model path
    """

    def __init__(
        self,
        *,
        candidates_dir: str,
        fallback_model_path: str,
        explicit_override_path: Optional[str] = None,
        require_symbol_model: bool = False,
        min_candidate_accuracy: float = 0.0,
        log=None,
    ):
        self.candidates_dir = str(candidates_dir)
        self.fallback_model_path = str(fallback_model_path)
        self.explicit_override_path = str(explicit_override_path or "").strip() or None
        self.require_symbol_model = bool(require_symbol_model)
        self.min_candidate_accuracy = max(0.0, min(1.0, float(min_candidate_accuracy)))
        self.log = log or (lambda *_args, **_kwargs: None)
        self._cache: Dict[str, dict] = {}
        self._resolved_path_cache: Dict[Tuple[str, int], Optional[str]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()
        self._resolved_path_cache.clear()

    def _candidate_dir(self, symbol: str, timeframe: int) -> Path:
        return Path(self.candidates_dir) / _safe_fs_name(symbol) / f"tf_{int(timeframe)}"

    def _score_candidate(self, path: Path) -> tuple[float, float]:
        try:
            loaded = joblib.load(str(path))
            if not isinstance(loaded, dict):
                return (float("-inf"), path.stat().st_mtime)
            metrics = loaded.get("train_metrics") if isinstance(loaded.get("train_metrics"), dict) else {}
            acc = metrics.get("accuracy")
            if acc is None:
                return (float("-inf"), path.stat().st_mtime)
            acc = float(acc)
            if acc < self.min_candidate_accuracy:
                return (float("-inf"), path.stat().st_mtime)
            return (acc, path.stat().st_mtime)
        except Exception:
            return (float("-inf"), path.stat().st_mtime)

    def _latest_candidate_path(self, symbol: str, timeframe: int) -> Optional[str]:
        d = self._candidate_dir(symbol, timeframe)
        if not d.exists() or not d.is_dir():
            return None

        files = sorted(
            [p for p in d.glob("*.joblib") if p.is_file()],
            key=lambda p: (p.stat().st_mtime, str(p.name)),
            reverse=True,
        )
        return str(files[0]) if files else None

    def _best_candidate_path(self, symbol: str, timeframe: int) -> Optional[str]:
        d = self._candidate_dir(symbol, timeframe)
        if not d.exists() or not d.is_dir():
            return None

        files = [p for p in d.glob("*.joblib") if p.is_file()]
        if not files:
            return None

        best = max(files, key=self._score_candidate)
        best_score = self._score_candidate(best)
        if best_score[0] == float("-inf"):
            return None
        return str(best)

    def resolve_path(self, symbol: str, timeframe: int) -> Optional[str]:
        key = (str(symbol), int(timeframe))
        if key in self._resolved_path_cache:
            return self._resolved_path_cache[key]

        path = None

        if self.explicit_override_path and os.path.exists(self.explicit_override_path):
            path = self.explicit_override_path
        else:
            candidate = self._best_candidate_path(symbol, timeframe)
            if not candidate:
                candidate = self._latest_candidate_path(symbol, timeframe)
            if candidate and os.path.exists(candidate):
                path = candidate
            elif (not self.require_symbol_model) and self.fallback_model_path and os.path.exists(self.fallback_model_path):
                path = self.fallback_model_path

        self._resolved_path_cache[key] = path
        return path

    def get_bundle(self, symbol: str, timeframe: int) -> tuple[Optional[dict], Optional[str]]:
        path = self.resolve_path(symbol, timeframe)
        if not path:
            return None, None

        try:
            if path not in self._cache:
                loaded = joblib.load(path)
                if isinstance(loaded, dict) and "model" in loaded:
                    self._cache[path] = loaded
                else:
                    self._cache[path] = {"model": loaded}
            return self._cache[path], path
        except Exception as e:
            self.log(f"[ML] Failed to load bundle for {symbol} tf={timeframe} from {path}: {e}")
            return None, path

    def describe(self, symbol: str, timeframe: int) -> dict[str, Any]:
        path = self.resolve_path(symbol, timeframe)
        return {
            "symbol": str(symbol),
            "timeframe": int(timeframe),
            "resolved_path": path,
            "uses_override": bool(self.explicit_override_path and path == self.explicit_override_path),
            "uses_fallback": bool(path and path == self.fallback_model_path),
        }