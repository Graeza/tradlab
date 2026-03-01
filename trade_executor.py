import MetaTrader5 as mt5

class TradeExecutor:

    def _normalize_volume(self, symbol, volume):
        info = mt5.symbol_info(symbol)

        if info is None:
            raise RuntimeError(f"Symbol info not found for {symbol}")

        min_vol = info.volume_min
        max_vol = info.volume_max
        step = info.volume_step

        # Clamp to min/max
        volume = max(min_vol, min(volume, max_vol))

        # Snap to nearest valid step
        steps = round((volume - min_vol) / step)
        normalized = min_vol + steps * step

        return round(normalized, 2)

    def execute(self, params):
        symbol = params["symbol"]
        action = params["action"]
        requested_lot = params["lot_size"]

        # Normalize lot size to valid Deriv volume
        lot = self._normalize_volume(symbol, requested_lot)

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL

        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if action == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "magic": 123456,
            "comment": "ModularBot",
        }

        result = mt5.order_send(request)
        return result
