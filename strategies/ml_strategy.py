from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal
from utils.indicators import add_h1_context_to_df


class MLStrategy(Strategy):
    name = "ML"

    def __init__(
        self,
        model=None,
        feature_cols: Optional[List[str]] = None,
        *,
        model_version: Optional[str] = None,
        schema_version: int = 1,
        strict_schema: bool = True,
        class_to_signal: Optional[Dict[object, str]] = None,
        drop_cols: Optional[List[str]] = None,
        fillna_value: Optional[float] = None,
        feature_set_version: Optional[int] = None,
        feature_set_id: Optional[str] = None,
        use_h1_meta: bool = True,
        h1_tf: int = 60,
        h1_sr_buffer_atr_mult: float = 0.50,
        bundle_registry=None,
        default_symbol: Optional[str] = None,
        default_primary_tf: Optional[int] = None,
    ):
        """
        ML-backed strategy with feature-schema enforcement.

        Supports either:
          1) a fixed already-loaded model bundle, or
          2) dynamic bundle resolution via bundle_registry using (symbol, timeframe)
        """
        self.model = model
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.model_version = model_version
        self.schema_version = int(schema_version)
        self.strict_schema = bool(strict_schema)
        self.class_to_signal = class_to_signal
        self.drop_cols = set((drop_cols or []))
        self.fillna_value = fillna_value
        self.feature_set_version = int(feature_set_version) if feature_set_version is not None else None
        self.feature_set_id = str(feature_set_id) if feature_set_id is not None else None

        self.use_h1_meta = bool(use_h1_meta)
        self.h1_tf = int(h1_tf)
        self.h1_sr_buffer_atr_mult = float(h1_sr_buffer_atr_mult)

        self.bundle_registry = bundle_registry
        self.default_symbol = str(default_symbol) if default_symbol else None
        self.default_primary_tf = int(default_primary_tf) if default_primary_tf is not None else None

        self._expected_cols = None  # type: Optional[List[str]]
        if self.feature_cols:
            self._expected_cols = list(self.feature_cols)
        else:
            cols = getattr(self.model, "feature_names_in_", None) if self.model is not None else None
            if cols is not None:
                try:
                    self._expected_cols = [str(c) for c in list(cols)]
                except Exception:
                    self._expected_cols = None

        self._expected_schema_id = self._schema_id(self._expected_cols) if self._expected_cols else None

    @staticmethod
    def _schema_id(cols: Optional[List[str]]) -> Optional[str]:
        if not cols:
            return None
        payload = "\n".join([str(c) for c in cols]).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def _apply_bundle(self, bundle: dict, resolved_path: Optional[str] = None) -> None:
        self.model = bundle.get("model")
        self.feature_cols = list(bundle.get("feature_cols") or []) or None
        self.model_version = bundle.get("model_version") or bundle.get("version")
        self.schema_version = int(bundle.get("schema_version", 1))
        self.strict_schema = bool(bundle.get("strict_schema", True))
        self.class_to_signal = bundle.get("class_to_signal")
        self.fillna_value = bundle.get("fillna_value")
        self.feature_set_version = (
            int(bundle.get("feature_set_version"))
            if bundle.get("feature_set_version") is not None
            else None
        )
        self.feature_set_id = (
            str(bundle.get("feature_set_id"))
            if bundle.get("feature_set_id") is not None
            else None
        )

        if self.feature_cols:
            self._expected_cols = list(self.feature_cols)
        else:
            cols = getattr(self.model, "feature_names_in_", None) if self.model is not None else None
            if cols is not None:
                try:
                    self._expected_cols = [str(c) for c in list(cols)]
                except Exception:
                    self._expected_cols = None
            else:
                self._expected_cols = None

        self._expected_schema_id = self._schema_id(self._expected_cols) if self._expected_cols else None
        self._resolved_model_path = resolved_path

    def _resolve_runtime_bundle(self, context: Optional[Dict[str, Any]] = None) -> tuple[Optional[dict], Optional[str], Optional[str], Optional[int]]:
        context = context or {}
        symbol = context.get("symbol", self.default_symbol)
        primary_tf = context.get("primary_tf", self.default_primary_tf)

        if symbol is None or primary_tf is None:
            return None, None, symbol, primary_tf

        if self.bundle_registry is None:
            return None, None, symbol, primary_tf

        bundle, path = self.bundle_registry.get_bundle(str(symbol), int(primary_tf))
        return bundle, path, str(symbol), int(primary_tf)

    def _infer_live_feature_cols(self, df: pd.DataFrame) -> List[str]:
        cols = []
        for c in df.columns:
            if c in self.drop_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(str(c))
        return cols

    @staticmethod
    def _pred_to_signal(pred: Any) -> str:
        s = str(pred).upper()
        if s in {"BUY", "SELL", "HOLD"}:
            return s

        # common numeric conventions
        try:
            x = int(pred)
            if x > 0:
                return "BUY"
            if x < 0:
                return "SELL"
            return "HOLD"
        except Exception:
            return "HOLD"

    def _classes_default_mapping(self) -> Dict[object, str]:
        cls = list(getattr(self.model, "classes_", []) or [])
        if set(cls) == {-1, 0, 1}:
            return {-1: "SELL", 0: "HOLD", 1: "BUY"}
        if set(cls) == {0, 1, 2}:
            return {0: "SELL", 1: "HOLD", 2: "BUY"}
        if set(cls) == {0, 1}:  # default: 0=SELL, 1=BUY
            return {0: "SELL", 1: "BUY"}
        return {}

    def _clean_X(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.fillna_value is None:
            if X.isna().any().any():
                bad_cols = [str(c) for c in X.columns[X.isna().any()].tolist()]
                raise ValueError(f"NaN/inf in features: {bad_cols[:25]}")
            return X
        return X.fillna(float(self.fillna_value))

    @staticmethod
    def _find_atr_col(df: pd.DataFrame) -> Optional[str]:
        for c in ("ATR", "ATR14", "atr", "atr14"):
            if c in df.columns:
                return c
        return None
    def _evaluate(self, data_by_tf: Dict[int, pd.DataFrame], context: Optional[Dict[str, Any]] = None):
        df = next(iter(data_by_tf.values()))
        if df is None or df.empty:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {"reason": "no_data"},
            }

        # Dynamic per-symbol/per-timeframe model resolution
        if self.bundle_registry is not None:
            bundle, resolved_path, ctx_symbol, ctx_tf = self._resolve_runtime_bundle(context)
            if not bundle or "model" not in bundle:
                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
                        "reason": "no_model_bundle_for_context",
                        "symbol": ctx_symbol,
                        "primary_tf": ctx_tf,
                        "resolved_model_path": resolved_path,
                    },
                }
            self._apply_bundle(bundle, resolved_path=resolved_path)

        if self.model is None:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {"reason": "no_model_loaded"},
            }

        # DataFetcher already returns closed-bar data, so use the latest row directly.
        if len(df) < 1:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {"reason": "insufficient_bars"},
            }

        # Keep a separate debug/context copy so we do NOT mutate the inference schema.
        debug_df = df
        if self.use_h1_meta:
            h1_df = data_by_tf.get(self.h1_tf)
            if h1_df is not None and not h1_df.empty:
                try:
                    debug_df = add_h1_context_to_df(
                        df.copy(),
                        h1_df,
                        atr_col=self._find_atr_col(df),
                        sr_atr_buffer_mult=self.h1_sr_buffer_atr_mult,
                    )
                except Exception:
                    debug_df = df

        meta_ctx = {}
        try:
            last_dbg = debug_df.iloc[-1]
            meta_ctx = {
                "h1_trend": str(last_dbg.get("h1_trend", "neutral")).lower(),
                "h1_support": last_dbg.get("h1_support"),
                "h1_resistance": last_dbg.get("h1_resistance"),
                "dist_to_h1_support": last_dbg.get("dist_to_h1_support"),
                "dist_to_h1_resistance": last_dbg.get("dist_to_h1_resistance"),
                "near_h1_support": bool(last_dbg.get("near_h1_support", False)),
                "near_h1_resistance": bool(last_dbg.get("near_h1_resistance", False)),
            }
        except Exception:
            meta_ctx = {}

        # --- Feature set version gating ---
        if self.feature_set_version is not None:
            if "feature_set_version" not in df.columns:
                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
                        **meta_ctx,
                        "reason": "missing_feature_set_version",
                        "expected_feature_set_version": self.feature_set_version,
                        "schema_version": self.schema_version,
                        "model_version": self.model_version,
                    },
                }
            try:
                live_v = int(df["feature_set_version"].iloc[-1])
            except Exception:
                live_v = None
            if live_v != self.feature_set_version:
                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
                        **meta_ctx,
                        "reason": "feature_set_version_mismatch",
                        "expected_feature_set_version": self.feature_set_version,
                        "live_feature_set_version": live_v,
                        "schema_version": self.schema_version,
                        "model_version": self.model_version,
                    },
                }

        if self.feature_set_id is not None:
            if "feature_set_id" not in df.columns:
                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
                        **meta_ctx,
                        "reason": "missing_feature_set_id",
                        "expected_feature_set_id": self.feature_set_id,
                        "schema_version": self.schema_version,
                        "model_version": self.model_version,
                    },
                }
            live_id = str(df["feature_set_id"].iloc[-1])
            if live_id != self.feature_set_id:
                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
                        **meta_ctx,
                        "reason": "feature_set_id_mismatch",
                        "expected_feature_set_id": self.feature_set_id,
                        "live_feature_set_id": live_id,
                        "schema_version": self.schema_version,
                        "model_version": self.model_version,
                    },
                }

        live_cols = self._infer_live_feature_cols(df)
        live_schema_id = self._schema_id(live_cols)

        expected_cols = self._expected_cols
        expected_schema_id = self._expected_schema_id

        # If we have an expected schema and strict mode, reject on drift (including order drift)
        if expected_cols and self.strict_schema:
            if live_cols != expected_cols:
                missing = [c for c in expected_cols if c not in live_cols]
                extra = [c for c in live_cols if c not in expected_cols]

                first_mismatch = None
                for i in range(min(len(live_cols), len(expected_cols))):
                    if live_cols[i] != expected_cols[i]:
                        first_mismatch = {"index": i, "expected": expected_cols[i], "got": live_cols[i]}
                        break

                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
                        **meta_ctx,
                        "reason": "feature_schema_mismatch",
                        "expected_n": len(expected_cols),
                        "got_n": len(live_cols),
                        "missing_n": len(missing),
                        "extra_n": len(extra),
                        "missing": missing[:25],
                        "extra": extra[:25],
                        "first_mismatch": first_mismatch,
                        "expected_schema_id": expected_schema_id,
                        "live_schema_id": live_schema_id,
                        "schema_version": self.schema_version,
                        "model_version": self.model_version,
                    },
                }

            feature_cols = expected_cols
        else:
            if self.feature_cols is not None:
                feature_cols = list(self.feature_cols)
            else:
                cols = getattr(self.model, "feature_names_in_", None)
                if cols is not None:
                    try:
                        feature_cols = [str(c) for c in list(cols)]
                    except Exception:
                        feature_cols = live_cols
                else:
                    feature_cols = live_cols

        if not feature_cols:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {**meta_ctx, "reason": "no_features"},
            }

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {
                    **meta_ctx,
                    "reason": "missing_features",
                    "missing": missing[:25],
                    "missing_n": len(missing),
                    "expected_schema_id": expected_schema_id,
                    "live_schema_id": live_schema_id,
                    "schema_version": self.schema_version,
                    "model_version": self.model_version,
                },
            }

        try:
            X = df.loc[df.index[-1:], feature_cols].copy()
            X = self._clean_X(X)
        except Exception as e:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {
                    **meta_ctx,
                    "reason": "bad_features",
                    "error": str(e),
                    "expected_schema_id": expected_schema_id,
                    "live_schema_id": live_schema_id,
                    "schema_version": self.schema_version,
                    "model_version": self.model_version,
                },
            }

        conf = 0.55
        signal = "HOLD"
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[0]
                idx = int(np.argmax(proba))
                conf = float(np.max(proba))
                pred = getattr(self.model, "classes_", [None] * len(proba))[idx]
            else:
                pred = self.model.predict(X)[0]

            mapping = self.class_to_signal or self._classes_default_mapping()
            if mapping and pred in mapping:
                signal = str(mapping[pred]).upper()
            else:
                signal = self._pred_to_signal(pred)

        except Exception as e:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {
                    **meta_ctx,
                    "reason": "model_error",
                    "error": str(e),
                    "expected_schema_id": expected_schema_id,
                    "live_schema_id": live_schema_id,
                    "schema_version": self.schema_version,
                    "model_version": self.model_version,
                },
            }

        return {
            "name": self.name,
            "signal": signal,
            "confidence": float(conf),
            "meta": {
                **meta_ctx,
                "features_n": int(X.shape[1]),
                "filled_na": bool(self.fillna_value is not None),
                "expected_schema_id": expected_schema_id,
                "live_schema_id": live_schema_id,
                "schema_version": self.schema_version,
                "model_version": self.model_version,
            },
        }
    
    def evaluate(self, data_by_tf: Dict[int, pd.DataFrame], context: Optional[Dict[str, Any]] = None):
        raw = self._evaluate(data_by_tf, context=context)
        return StrategyResult(
            name=str(raw.get("name", self.name)),
            signal=Signal(str(raw.get("signal", "HOLD")).upper()),
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            meta=dict(raw.get("meta", {}) or {}),
        )