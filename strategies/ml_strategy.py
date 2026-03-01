from __future__ import annotations
import pandas as pd
import numpy as np
from strategies.base import Strategy

class MLStrategy(Strategy):
    name = "ML"

    def __init__(self, model, feature_cols: list[str] | None = None):
        self.model = model
        self.feature_cols = feature_cols  # if None, will infer numeric columns

    def evaluate(self, data_by_tf: dict[int, pd.DataFrame]):
        df = next(iter(data_by_tf.values()))
        if df is None or df.empty:
            return {"name": self.name, "signal": "HOLD", "confidence": 0.0, "meta": {"reason": "no_data"}}

        row = df.iloc[-1]
        if self.feature_cols is None:
            X = row.select_dtypes(include=["number"]).to_frame().T
        else:
            X = pd.DataFrame([row[self.feature_cols].values], columns=self.feature_cols)

        # Support both proba and direct prediction
        conf = 0.55
        signal = "HOLD"
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[0]
                # assume class order [-1,0,1] or [0,1] etc; best-effort mapping:
                idx = int(np.argmax(proba))
                conf = float(np.max(proba))
                pred = self.model.classes_[idx]
            else:
                pred = self.model.predict(X)[0]

            # Map pred to BUY/SELL/HOLD
            if str(pred).upper() in ("BUY", "SELL", "HOLD"):
                signal = str(pred).upper()
            else:
                try:
                    v = float(pred)
                    signal = "BUY" if v > 0 else "SELL" if v < 0 else "HOLD"
                except Exception:
                    signal = "HOLD"

        except Exception as e:
            return {"name": self.name, "signal": "HOLD", "confidence": 0.0, "meta": {"error": str(e)}}

        return {"name": self.name, "signal": signal, "confidence": float(conf), "meta": {}}
