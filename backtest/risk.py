from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class BacktestRiskParams:
    qty: float
    sl: Optional[float]
    tp: Optional[float]


class BacktestRiskManager:
    """Backtest-side approximation of the live risk and entry policy.

    Notes:
      - qty is in simulated units, not broker lots.
      - max_spread_points and base_deviation_points are kept for parity/audit, but
        they are informational unless you later provide historical spread/slippage data.
    """

    def __init__(
        self,
        max_risk_pct: float = 1.0,
        min_confidence: float = 0.60,
        sl_atr_mult: float = 2.0,
        tp_rr: float = 1.5,
        fallback_sl_pct: float = 0.003,
        max_leverage: float = 2.0,
        max_spread_points: int = 0,
        base_deviation_points: int = 0,
        force_symbol_fixed_lot: bool = False,
        boom_crash_fixed_sl_tp: bool = False,
        boom_crash_sl_tp_offset: float = 0.0,
        enable_spread_filter: bool = False,
        exec_max_spread_points: int = 0,
    ):
        self.max_risk_pct = float(max_risk_pct)
        self.min_confidence = float(min_confidence)
        self.sl_atr_mult = float(sl_atr_mult)
        self.tp_rr = float(tp_rr)
        self.fallback_sl_pct = float(fallback_sl_pct)
        self.max_leverage = float(max_leverage)
        self.max_spread_points = int(max_spread_points)
        self.base_deviation_points = int(base_deviation_points)
        self.force_symbol_fixed_lot = bool(force_symbol_fixed_lot)
        self.boom_crash_fixed_sl_tp = bool(boom_crash_fixed_sl_tp)
        self.boom_crash_sl_tp_offset = float(boom_crash_sl_tp_offset)
        self.enable_spread_filter = bool(enable_spread_filter)
        self.exec_max_spread_points = int(exec_max_spread_points)

    @staticmethod
    def _fixed_lot_for_symbol(symbol: str) -> Optional[float]:
        s = str(symbol or "").lower()
        if "boom 1000" in s or "boom 900" in s or "boom 500" in s or "boom 600" in s:
            return 0.2
        if "boom 300" in s:
            return 0.5
        return None

    def assess(
        self,
        signal: Dict[str, Any],
        equity: float,
        entry_price: float,
        regime: Optional[Dict[str, Any]] = None,
        symbol: str = "",
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

        if self.boom_crash_fixed_sl_tp:
            sym_l = str(symbol or "").lower()
            if ("boom" in sym_l or "crash" in sym_l) and float(self.boom_crash_sl_tp_offset) > 0:
                sl_dist = float(self.boom_crash_sl_tp_offset)

        if sl_dist <= 0:
            return None

        risk_money = equity * (self.max_risk_pct / 100.0)
        per_unit_loss = sl_dist
        qty = risk_money / per_unit_loss
        if qty <= 0:
            return None

        max_qty = (equity * self.max_leverage) / entry_price
        qty = min(qty, max_qty)
        if qty <= 0:
            return None

        if self.force_symbol_fixed_lot:
            fixed_qty = self._fixed_lot_for_symbol(symbol)
            if fixed_qty is not None and fixed_qty > 0:
                qty = float(fixed_qty)

        if action == "BUY":
            sl = entry_price - sl_dist
            tp = entry_price + sl_dist * self.tp_rr
        else:
            sl = entry_price + sl_dist
            tp = entry_price - sl_dist * self.tp_rr

        return BacktestRiskParams(qty=float(qty), sl=float(sl), tp=float(tp))
