from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class BacktestRiskParams:
    qty: float
    sl: Optional[float]
    tp: Optional[float]


class BacktestRiskManager:
    """Simple risk policy for backtests.

    This is intentionally MT5-free.

    Notes:
      - qty is in *price units* (not lots). If you later want lot sizing,
        introduce a symbol-spec lookup.
      - Stop distance uses regime atr_pct when available; fallback to pct.
    """

    def __init__(
        self,
        max_risk_pct: float = 1.0,
        min_confidence: float = 0.60,
        sl_atr_mult: float = 2.0,
        tp_rr: float = 1.5,
        fallback_sl_pct: float = 0.003,
        max_leverage: float = 2.0,
    ):
        self.max_risk_pct = float(max_risk_pct)
        self.min_confidence = float(min_confidence)
        self.sl_atr_mult = float(sl_atr_mult)
        self.tp_rr = float(tp_rr)
        self.fallback_sl_pct = float(fallback_sl_pct)
        self.max_leverage = float(max_leverage)

    def assess(
        self,
        signal: Dict[str, Any],
        equity: float,
        entry_price: float,
        regime: Optional[Dict[str, Any]] = None,
    ) -> Optional[BacktestRiskParams]:
        if not isinstance(signal, dict):
            return None

        action = str(signal.get("signal") or "HOLD").upper()
        conf = float(signal.get("confidence") or 0.0)
        if action == "HOLD" or conf < self.min_confidence:
            return None

        if equity <= 0 or entry_price <= 0:
            return None

        reg = regime if isinstance(regime, dict) else {}
        atr_pct = float(reg.get("atr_pct") or 0.0)
        if atr_pct > 0:
            sl_dist = entry_price * atr_pct * self.sl_atr_mult
        else:
            sl_dist = entry_price * self.fallback_sl_pct

        if sl_dist <= 0:
            return None

        # risk money per trade
        risk_money = equity * (self.max_risk_pct / 100.0)
        # simplified per-unit loss if stop hit
        per_unit_loss = sl_dist
        qty = risk_money / per_unit_loss
        if qty <= 0:
            return None

        # leverage clamp: position value <= equity * max_leverage
        max_qty = (equity * self.max_leverage) / entry_price
        qty = min(qty, max_qty)
        if qty <= 0:
            return None

        if action == "BUY":
            sl = entry_price - sl_dist
            tp = entry_price + sl_dist * self.tp_rr
        else:
            sl = entry_price + sl_dist
            tp = entry_price - sl_dist * self.tp_rr

        return BacktestRiskParams(qty=float(qty), sl=float(sl), tp=float(tp))
