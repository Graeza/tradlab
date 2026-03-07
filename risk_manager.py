"""Risk management and position sizing.

Policy:
- Whether a signal is tradable (confidence / spread gating)
- How big the trade should be (risk-based sizing)
- Where SL/TP should go (volatility-based distance)

All MT5 calls are serialized via MT5Client to keep MT5 single-threaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.mt5_worker import MT5Client

@dataclass(frozen=True)
class RiskDecision:
    symbol: str
    action: str
    lot_size: float
    sl: float
    tp: float
    deviation: int

    def to_params(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "lot_size": float(self.lot_size),
            "sl": float(self.sl),
            "tp": float(self.tp),
            "deviation": int(self.deviation),
        }

class RiskManager:
    def __init__(
        self,
        mt5: MT5Client,
        max_risk_pct: float = 0.75,
        min_confidence: float = 0.55,
        sl_atr_mult: float = 2.5,
        tp_rr: float = 1.8,
        fallback_sl_pct: float = 0.003,
        max_spread_points: int = 25000,
        base_deviation_points: int = 30,
    ):
        self.mt5 = mt5
        self.max_risk_pct = float(max_risk_pct)
        self.min_confidence = float(min_confidence)
        self.sl_atr_mult = float(sl_atr_mult)
        self.tp_rr = float(tp_rr)
        self.fallback_sl_pct = float(fallback_sl_pct)
        self.max_spread_points = int(max_spread_points)
        self.base_deviation_points = int(base_deviation_points)

    def _equity(self) -> Optional[float]:
        acc = self.mt5.account_info()
        if acc is None:
            return None
        eq = getattr(acc, "equity", None)
        bal = getattr(acc, "balance", None)
        # Prefer equity (includes floating PnL); fallback to balance.
        val = eq if (eq is not None and float(eq) > 0) else bal
        return float(val) if (val is not None and float(val) > 0) else None

    @staticmethod
    def _money_per_lot_for_move(symbol_info, price_delta: float) -> float:
        """Approx loss for 1.0 lot if price moves by `price_delta`."""
        tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
        tick_value = float(getattr(symbol_info, "trade_tick_value", 0.0) or 0.0)
        if tick_size <= 0:
            tick_size = float(getattr(symbol_info, "point", 0.0) or 0.0)
        if tick_size <= 0 or tick_value <= 0:
            return 0.0
        ticks = abs(price_delta) / tick_size
        return float(ticks * tick_value)

    @staticmethod
    def _min_stop_distance(info) -> float:
        """Minimum allowed SL/TP distance from current price in *price units*.

        MT5 provides trade_stops_level in POINTS; convert using point.
        """
        point = float(getattr(info, "point", 0.0) or 0.0)
        stops_level_pts = float(getattr(info, "trade_stops_level", 0.0) or 0.0)
        stop_level_alt = float(getattr(info, "stop_level", 0.0) or 0.0)

        lvl_pts = max(stops_level_pts, stop_level_alt, 0.0)
        if point <= 0:
            return 0.0
        if lvl_pts <= 0:
            # conservative fallback if broker doesn't report
            return point * 10.0
        return lvl_pts * point

    def assess(self, signal: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """Return trade params dict for TradeExecutor, or None to reject."""
        def dbg(msg: str):
            # Toggle this to False when you're done debugging
            print(msg)

        # --- basic validation ---
        dbg(f"[RISK DEBUG] Checking {symbol} signal={signal}")
        if not isinstance(signal, dict):
            return None

        action = str(signal.get("signal") or "HOLD").upper()
        conf = float(signal.get("confidence") or 0.0)

        if action == "HOLD":
            dbg("[RISK DEBUG] Reject: HOLD signal")
            return None
        if conf < float(self.min_confidence):
            dbg(f"[RISK DEBUG] Reject: confidence {conf} < min_confidence {self.min_confidence}")
            return None

        # --- ensure symbol available ---
        self.mt5.symbol_select(symbol, True)

        info = self.mt5.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        eq = self._equity()
        if info is None or tick is None or eq is None:
            dbg(f"[RISK DEBUG] Reject: MT5 data missing info={info} tick={tick} equity={eq}")
            return None

        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid <= 0 or ask <= 0:
            dbg(f"[RISK DEBUG] Reject: bad prices bid={bid} ask={ask}")
            return None

        point = float(getattr(info, "point", 0.0) or 0.0)
        digits = getattr(info, "digits", None)
        tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
        tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)

        spread_price = ask - bid
        spread_points = (spread_price / point) if point > 0 else 0.0

        dbg(
            f"[RISK DEBUG] prices bid={bid} ask={ask} diff={spread_price} "
            f"point={point} digits={digits} tick_size={tick_size} tick_value={tick_value}"
        )

        # --- spread guard ---
        sym_l = symbol.lower()
        is_synth = ("boom" in sym_l) or ("crash" in sym_l)

        # Price-based limits for synthetics (recommended)
        spread_price_limits = {
            "boom 1000": 3.0,
            "boom 900": 3.0,
            "boom 600": 3.0,
            "boom 500": 3.0,
            "boom 300": 3.0,
        }

        max_spread_price = None
        for k, v in spread_price_limits.items():
            if k in sym_l:
                max_spread_price = float(v)
                break

        if is_synth and max_spread_price is not None:
            dbg(f"[RISK DEBUG] spread_price={spread_price} max_spread_price={max_spread_price}")
            if spread_price > max_spread_price:
                dbg(f"[RISK DEBUG] Reject: spread_price too high spread_price={spread_price} max_spread_price={max_spread_price}")
                return None
        else:
            info_spread = float(getattr(info, "spread", 0.0) or 0.0)
            sp_pts = info_spread if info_spread > 0 else spread_points
            dbg(f"[RISK DEBUG] spread_points={sp_pts} max={self.max_spread_points} (info.spread={info_spread})")
            if sp_pts > float(self.max_spread_points):
                dbg(f"[RISK DEBUG] Reject: spread too high spread_points={sp_pts} max_spread_points={self.max_spread_points}")
                return None

        # --- entry, SL/TP ---
        entry_price = ask if action == "BUY" else bid

        regime = signal.get("regime") if isinstance(signal.get("regime"), dict) else {}
        atr_pct = float(regime.get("atr_pct") or 0.0)

        if atr_pct > 0:
            sl_dist = entry_price * atr_pct * float(self.sl_atr_mult)
        else:
            sl_dist = entry_price * float(self.fallback_sl_pct)

        # Safety: enforce broker minimum stop distance (stops level)
        min_stop = self._min_stop_distance(info)
        if min_stop > 0 and sl_dist < min_stop:
            sl_dist = max(min_stop, entry_price * float(self.fallback_sl_pct))

        if action == "BUY":
            sl = entry_price - sl_dist
            tp = entry_price + sl_dist * float(self.tp_rr)
        else:
            sl = entry_price + sl_dist
            tp = entry_price - sl_dist * float(self.tp_rr)

        # --- sizing ---
        risk_money = float(eq) * (float(self.max_risk_pct) / 100.0)
        per_lot_loss = float(self._money_per_lot_for_move(info, sl_dist) or 0.0)

        dbg(f"[RISK DEBUG] sl_dist={sl_dist} min_stop={min_stop} risk_money={risk_money} per_lot_loss={per_lot_loss}")

        if per_lot_loss <= 0:
            dbg("[RISK DEBUG] Reject: per_lot_loss <= 0")
            return None

        raw_lot = risk_money / per_lot_loss
        if raw_lot <= 0:
            dbg(f"[RISK DEBUG] Reject: raw_lot <= 0 ({raw_lot})")
            return None

        # --- deviation ---
        sp_for_dev = spread_points
        deviation = int(max(float(self.base_deviation_points), float(self.base_deviation_points) + sp_for_dev))
        deviation_cap = 500
        deviation = max(int(self.base_deviation_points), min(deviation, deviation_cap))

        decision = RiskDecision(
            symbol=symbol,
            action=action,
            lot_size=float(raw_lot),
            sl=float(sl),
            tp=float(tp),
            deviation=int(deviation),
        )

        dbg(f"[RISK DEBUG] APPROVED: lot={raw_lot} sl={sl} tp={tp} deviation={deviation}")
        return decision.to_params()
