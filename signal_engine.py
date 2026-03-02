# Backward-compatible wrapper. Prefer using core/ensemble.py + strategies/* for multi-strategy setups.
class SignalEngine:
    def __init__(self, strategy):
        self.strategy = strategy

    def generate_signal(self, df):
        # Old contract: strategy takes df. New contract: strategies take dict[tf]->df.
        if hasattr(self.strategy, "evaluate"):
            out = self.strategy.evaluate({0: df})
            if hasattr(out, "to_dict"):
                out = out.to_dict()
            return {"signal": out.get("signal", "HOLD"), "confidence": float(out.get("confidence", 0.0) or 0.0), "meta": out.get("meta", {})}
        return self.strategy.generate_signal(df)
