from __future__ import annotations

import MetaTrader5 as mt5  # constants only
from core.mt5_worker import MT5Client


def list_positions(mt5c: MT5Client, symbol: str | None = None):
    if symbol:
        return mt5c.positions_get(symbol=symbol) or []
    return mt5c.positions_get() or []


def close_position(mt5c: MT5Client, pos) -> tuple[bool, str]:
    symbol = pos.symbol
    volume = float(pos.volume)
    ticket = int(pos.ticket)

    tick = mt5c.symbol_info_tick(symbol)
    if tick is None:
        return False, f"No tick for {symbol}"

    if int(pos.type) == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = float(tick.bid)
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = float(tick.ask)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": ticket,
        "symbol": symbol,
        "volume": volume,
        "type": int(order_type),
        "price": float(price),
        "deviation": 20,
        "magic": 234000,
        "comment": "GUI close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result is None:
        return False, f"order_send returned None for {symbol} ticket={ticket}"
    if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
        return False, f"Close failed retcode={getattr(result,'retcode',None)} {getattr(result,'comment','')}"
    return True, f"Closed {symbol} ticket={ticket} vol={volume}"


def close_positions(mt5c: MT5Client, mode: str = "all") -> dict:
    mode = (mode or "all").lower()
    positions = list_positions(mt5c)

    closed, failed = [], []
    for p in positions:
        profit = float(getattr(p, "profit", 0.0))
        if mode == "positive" and profit <= 0:
            continue
        if mode == "negative" and profit >= 0:
            continue

        ok, msg = close_position(mt5c, p)
        (closed if ok else failed).append(msg)

    return {"mode": mode, "attempted": len(positions), "closed": closed, "failed": failed}
