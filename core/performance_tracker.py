from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from strategies.base import StrategyOutput


def _direction_from_return(r: float, eps: float = 0.0) -> str:
    if r > eps:
        return "BUY"
    if r < -eps:
        return "SELL"
    return "HOLD"


@dataclass
class PendingPrediction:
    symbol: str
    bar_time: int  # df['time'] (seconds)
    close: float
    horizon_bars: int
    final: Dict[str, Any]
    outputs: List[StrategyOutput]  # StrategyOutput list
    regime: Dict[str, Any]


class PerformanceTracker:
    """
    In-memory tracker:
      - add_prediction() at decision time
      - update_with_bars() to resolve pending once future bar exists
      - summary_rows() for GUI table
    """

    def __init__(self, max_pending_per_symbol: int = 2000, eps: float = 0.0):
        self.max_pending_per_symbol = max_pending_per_symbol
        self.eps = eps

        # symbol -> list[PendingPrediction]
        self.pending: Dict[str, List[PendingPrediction]] = {}

        # stats["RSI_EMA"] = {"n":..., "wins":..., "sum_ret":..., "sum_abs_ret":...}
        self.stats: Dict[str, Dict[str, float]] = {}
        self.stats_final: Dict[str, float] = {"n": 0.0, "wins": 0.0, "sum_ret": 0.0, "sum_abs_ret": 0.0}
        self.stats_by_regime: Dict[str, Dict[str, float]] = {}
        self.stats_final_by_regime: Dict[str, Dict[str, float]] = {}

    def add_prediction(
        self,
        symbol: str,
        df_primary: pd.DataFrame,
        horizon_bars: int,
        final: Dict[str, Any],
        outputs: List[StrategyOutput],
    ) -> None:
        if df_primary is None or df_primary.empty:
            return
        
        regime = dict(final.get("regime") or {})

        # Use the latest complete bar
        last = df_primary.iloc[-1]
        bar_time = int(last["time"])
        close = float(last["close"])

        # Prevent duplicate predictions on the same bar_time
        pend = self.pending.setdefault(symbol, [])
        if pend and pend[-1].bar_time == bar_time:
            return

        pend.append(PendingPrediction(
            symbol=symbol,
            bar_time=bar_time,
            close=close,
            horizon_bars=horizon_bars,
            final=dict(final),
            outputs=[dict(o) for o in outputs],
            regime=regime
        ))

        # Trim
        if len(pend) > self.max_pending_per_symbol:
            self.pending[symbol] = pend[-self.max_pending_per_symbol:]

    def update_with_bars(self, symbol: str, df_primary: pd.DataFrame) -> None:
        """
        Resolve any pending predictions where we now have bar_time + horizon_bars available.
        Assumes df_primary is sorted by time ascending.
        """
        pend = self.pending.get(symbol)
        if not pend or df_primary is None or df_primary.empty:
            return

        # Build an index: time -> row position
        times = df_primary["time"].astype(int).tolist()
        pos_by_time = {t: i for i, t in enumerate(times)}
        closes = df_primary["close"].astype(float).tolist()

        resolved_count = 0
        new_pending: List[PendingPrediction] = []

        for p in pend:
            i = pos_by_time.get(p.bar_time)
            if i is None:
                # bar not in df window; drop it (or keep it) — keep for now
                new_pending.append(p)
                continue

            j = i + p.horizon_bars
            if j >= len(closes):
                # not enough future data yet
                new_pending.append(p)
                continue

            c0 = p.close
            c1 = float(closes[j])
            ret = (c1 - c0) / c0 if c0 != 0 else 0.0
            label = _direction_from_return(ret, eps=self.eps)

            trend = str((p.regime or {}).get("trend", "UNKNOWN")).upper()
            vol = str((p.regime or {}).get("vol", "UNKNOWN")).upper()
            reg_key = f"{trend}/{vol}"

            # Update FINAL stats
            final_sig = str(p.final.get("signal", "HOLD")).upper()
            if final_sig != "HOLD":
                self._update_bucket(self.stats_final, final_sig, label, ret)
                # final by regime
                b = self.stats_final_by_regime.setdefault(reg_key, {"n": 0.0, "wins": 0.0, "sum_ret": 0.0, "sum_abs_ret": 0.0})
                self._update_bucket(b, final_sig, label, ret)

            # Update per-strategy stats
            for o in p.outputs:
                name = str(o.get("name", "UNKNOWN"))
                sig = str(o.get("signal", "HOLD")).upper()
                if sig == "HOLD":
                    continue
                bucket = self.stats.setdefault(name, {"n": 0.0, "wins": 0.0, "sum_ret": 0.0, "sum_abs_ret": 0.0})
                self._update_bucket(bucket, sig, label, ret)
                # by regime
                rk_name = f"{name}@{reg_key}"
                bucket_r = self.stats_by_regime.setdefault(rk_name, {"n": 0.0, "wins": 0.0, "sum_ret": 0.0, "sum_abs_ret": 0.0})
                self._update_bucket(bucket_r, sig, label, ret)
            resolved_count += 1

        self.pending[symbol] = new_pending

    def _update_bucket(self, bucket: Dict[str, float], pred_sig: str, label_sig: str, ret: float) -> None:
        bucket["n"] += 1.0
        if pred_sig == label_sig:
            bucket["wins"] += 1.0
        bucket["sum_ret"] += ret
        bucket["sum_abs_ret"] += abs(ret)

    def summary_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        # FINAL (global)
        rows.append(self._row_from_bucket("FINAL", self.stats_final))

        # FINAL by regime
        for reg_key, b in self.stats_final_by_regime.items():
            rows.append(self._row_from_bucket(f"FINAL@{reg_key}", b))

        # Strategies (global)
        for name, b in self.stats.items():
            rows.append(self._row_from_bucket(name, b))

        # Strategies by regime
        for name, b in self.stats_by_regime.items():
            rows.append(self._row_from_bucket(name, b))

        # Sort by expectancy then win_rate then n
        rows.sort(key=lambda r: (r["expectancy"], r["win_rate"], r["n"]), reverse=True)
        return rows

    def _row_from_bucket(self, name: str, b: Dict[str, float]) -> Dict[str, Any]:
        n = float(b.get("n", 0.0))
        wins = float(b.get("wins", 0.0))
        sum_ret = float(b.get("sum_ret", 0.0))
        sum_abs = float(b.get("sum_abs_ret", 0.0))

        win_rate = (wins / n) if n > 0 else 0.0
        avg_ret = (sum_ret / n) if n > 0 else 0.0
        avg_abs = (sum_abs / n) if n > 0 else 0.0

        # Expectancy ≈ average return per prediction
        expectancy = avg_ret

        return {
            "name": name,
            "n": int(n),
            "win_rate": win_rate,
            "avg_ret": avg_ret,
            "avg_abs_ret": avg_abs,
            "expectancy": expectancy,
        }