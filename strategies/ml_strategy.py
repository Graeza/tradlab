from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal
class MLStrategy(Strategy):
    name = "ML"

    def __init__(
        self,
        model,
        feature_cols: list[str] | None = None,
        *,
        model_version: str | None = None,
        schema_version: int = 1,
        strict_schema: bool = True,
        class_to_signal: dict[object, str] | None = None,
        drop_cols: list[str] | None = None,
        fillna_value: float | None = None,
        feature_set_version: int | None = None,
        feature_set_id: str | None = None,
    ):
        """ML-backed strategy with feature-schema enforcement.

        The core failure mode in production ML trading systems is *schema drift*:
        columns added/removed/renamed, or ordering changes between training and live.

        This strategy can run in a strict mode that will REFUSE to trade if:
          - the live feature schema differs from what the model expects
          - required features are missing
          - features contain NaN/inf (unless fillna_value is provided)

        Parameters
        ----------
        model:
            Any sklearn-like estimator or pipeline implementing predict / predict_proba.
        feature_cols:
            Explicit trained feature list. If provided, treated as the expected schema.
            If None, we try model.feature_names_in_ (sklearn).
            If that isn't available, we fall back to inferring numeric columns from df.
        model_version:
            Optional string to identify model build/version (recommended via saved bundle).
        schema_version:
            Optional integer for your own schema versioning (recommended via saved bundle).
        strict_schema:
            If True (default), and we have an expected schema, reject when live schema differs.
        class_to_signal:
            Optional mapping from model class labels to {BUY, SELL, HOLD}.
        drop_cols:
            Optional extra columns to drop when inferring numeric columns.
        fillna_value:
            If None (default), any NaN/inf in features rejects the signal (HOLD).
            If set (e.g., 0.0), NaN/inf are replaced with that value.
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

        # Expected schema (if we can resolve it now)
        self._expected_cols: list[str] | None = None
        if self.feature_cols:
            self._expected_cols = list(self.feature_cols)
        else:
            cols = getattr(self.model, "feature_names_in_", None)
            if cols is not None:
                try:
                    self._expected_cols = [str(c) for c in list(cols)]
                except Exception:
                    self._expected_cols = None

        self._expected_schema_id = self._schema_id(self._expected_cols) if self._expected_cols else None

    @staticmethod
    def _schema_id(cols: list[str] | None) -> str | None:
        if not cols:
            return None
        payload = "\n".join([str(c) for c in cols]).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def _default_drop_cols(self) -> set[str]:
        # Common non-features that may be present in your pipeline DB/features tables
        return {
            "time", "dt", "datetime", "timestamp",
            "symbol", "tf", "timeframe",
            "label", "target", "y",
            "feature_set_version", "feature_set_id",
        } | set(self.drop_cols)

    def _infer_live_feature_cols(self, df: pd.DataFrame) -> list[str]:
        numeric = df.select_dtypes(include=["number"]).copy()
        numeric.drop(columns=[c for c in self._default_drop_cols() if c in numeric.columns], inplace=True, errors="ignore")
        return list(numeric.columns)

    def _clean_X(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.fillna_value is None:
            if X.isna().any().any():
                raise ValueError("NaN/inf in feature vector")
        else:
            X = X.fillna(self.fillna_value)
        return X

    def _pred_to_signal(self, pred) -> str:
        # Explicit mapping wins
        if self.class_to_signal is not None and pred in self.class_to_signal:
            return str(self.class_to_signal[pred]).upper()

        # Direct string labels
        if isinstance(pred, str):
            up = pred.upper()
            if up in ("BUY", "SELL", "HOLD"):
                return up

        # Numeric mapping
        try:
            v = float(pred)
            return "BUY" if v > 0 else "SELL" if v < 0 else "HOLD"
        except Exception:
            return "HOLD"

    def _classes_default_mapping(self) -> dict[object, str]:
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            return {}

        try:
            cls = list(classes)
        except Exception:
            return {}

        # Common supervised label sets
        if set(cls) >= {-1, 1}:  # {-1,0,1} or {-1,1}
            mapping = {-1: "SELL", 1: "BUY"}
            if 0 in set(cls):
                mapping[0] = "HOLD"
            return mapping

        if set(cls) == {0, 1}:  # default: 0=SELL, 1=BUY
            return {0: "SELL", 1: "BUY"}

        return {}

    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]):
        df = next(iter(data_by_tf.values()))
        if df is None or df.empty:
            return {
                "name": self.name,
                "signal": "HOLD",
                    "confidence": 0.0,
                "meta": {"reason": "no_data"},
            }

        # Live trading: ALWAYS use the last CLOSED candle (exclude forming bar)
        if len(df) < 2:
            return {
                "name": self.name,
                "signal": "HOLD",
                    "confidence": 0.0,
                "meta": {"reason": "insufficient_bars"},
            }

        closed = df.iloc[:-1]
        if closed is None or closed.empty:
            return {
                "name": self.name,
                "signal": "HOLD",
                    "confidence": 0.0,
                "meta": {"reason": "insufficient_closed_bars"},
            }

        df = closed

        # --- Feature set version gating (optional but recommended) ---
        if self.feature_set_version is not None:
            if "feature_set_version" not in df.columns:
                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
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
                # find first index mismatch when both lists have at least one element
                for i in range(min(len(live_cols), len(expected_cols))):
                    if live_cols[i] != expected_cols[i]:
                        first_mismatch = {"index": i, "expected": expected_cols[i], "got": live_cols[i]}
                        break

                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
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
            # Non-strict / unknown expected schema:
            # use explicit feature_cols if provided, else model.feature_names_in_, else inferred live numeric columns
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
            return {"name": self.name, "signal": "HOLD", "confidence": 0.0, "meta": {"reason": "no_features"}}

        # Missing features gate (even in non-strict mode)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return {
                "name": self.name,
                "signal": "HOLD",
                    "confidence": 0.0,
                "meta": {
                    "reason": "missing_features",
                    "missing": missing[:25],
                    "missing_n": len(missing),
                    "expected_schema_id": expected_schema_id,
                    "live_schema_id": live_schema_id,
                    "schema_version": self.schema_version,
                    "model_version": self.model_version,
                },
            }

        # Build feature vector in the EXACT expected order
        try:
            X = df.loc[df.index[-1:], feature_cols].copy()  # df already excludes forming bar
            X = self._clean_X(X)
        except Exception as e:
            return {
                "name": self.name,
                "signal": "HOLD",
                    "confidence": 0.0,
                "meta": {
                    "reason": "bad_features",
                    "error": str(e),
                    "expected_schema_id": expected_schema_id,
                    "live_schema_id": live_schema_id,
                    "schema_version": self.schema_version,
                    "model_version": self.model_version,
                },
            }

        # Predict
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
                "features_n": int(X.shape[1]),
                "filled_na": bool(self.fillna_value is not None),
                "expected_schema_id": expected_schema_id,
                "live_schema_id": live_schema_id,
                "schema_version": self.schema_version,
                "model_version": self.model_version,
            },
        }
