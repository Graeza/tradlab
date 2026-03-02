from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import pandas as pd


@dataclass(frozen=True)
class BacktestMetrics:
    start_equity: float
    end_equity: float
    total_return: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    avg_trade_pnl: float
    profit_factor: Optional[float]


def compute_metrics(equity_curve: pd.DataFrame, fills: pd.DataFrame) -> BacktestMetrics:
    if equity_curve is None or equity_curve.empty:
        return BacktestMetrics(
            start_equity=0.0,
            end_equity=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            n_trades=0,
            win_rate=0.0,
            avg_trade_pnl=0.0,
            profit_factor=None,
        )

    eq = equity_curve["equity"].astype(float).reset_index(drop=True)
    start = float(eq.iloc[0])
    end = float(eq.iloc[-1])
    total_ret = (end / start - 1.0) if start > 0 else 0.0

    # max drawdown
    peak = eq.cummax()
    dd = (eq - peak) / peak.replace(0, math.nan)
    max_dd = float(dd.min()) if len(dd) else 0.0
    if math.isnan(max_dd):
        max_dd = 0.0

    # trades: infer round-trips from fills
    n_entries = int((fills["side"] == "BUY").sum() + (fills["side"] == "SELL").sum()) if fills is not None and not fills.empty else 0
    n_trades = n_entries

    # approximate trade pnl from CLOSE fills: needs broker to record cash after close to be perfect.
    # Here we compute per trade PnL by pairing entry and close fills in order.
    win_rate = 0.0
    avg_pnl = 0.0
    pf = None

    if fills is not None and not fills.empty:
        fills = fills.reset_index(drop=True)
        entries = []
        trade_pnls = []
        for r in fills.itertuples(index=False):
            side = str(r.side)
            if side in ("BUY", "SELL"):
                entries.append(r)
            elif side == "CLOSE" and entries:
                e = entries.pop(0)
                entry_side = str(e.side)
                pnl = (float(r.price) - float(e.price)) * float(e.qty) if entry_side == "BUY" else (float(e.price) - float(r.price)) * float(e.qty)
                trade_pnls.append(float(pnl))

        if trade_pnls:
            wins = [p for p in trade_pnls if p > 0]
            losses = [-p for p in trade_pnls if p < 0]
            win_rate = float(len(wins) / len(trade_pnls))
            avg_pnl = float(sum(trade_pnls) / len(trade_pnls))
            if losses:
                pf = float(sum(wins) / sum(losses)) if sum(losses) > 0 else None

    return BacktestMetrics(
        start_equity=start,
        end_equity=end,
        total_return=float(total_ret),
        max_drawdown=float(max_dd),
        n_trades=int(n_trades),
        win_rate=float(win_rate),
        avg_trade_pnl=float(avg_pnl),
        profit_factor=pf,
    )
