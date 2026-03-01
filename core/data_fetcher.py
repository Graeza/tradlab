from __future__ import annotations

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

class DataFetcher:
    """Fetch raw OHLCV bars from MT5."""

    def fetch_window(self, symbol: str, timeframe: int, n_bars: int = 2000) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = df["time"].astype(int)
        df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    def fetch_since(self, symbol: str, timeframe: int, last_time_s: int | None, n_bars: int = 2000) -> pd.DataFrame:
        """Fetch bars and return only rows newer than last_time_s (epoch seconds)."""
        df = self.fetch_window(symbol, timeframe, n_bars=n_bars)
        if df.empty:
            return df
        if last_time_s is None:
            return df
        return df[df["time"] > int(last_time_s)].copy()

    def latest_tick(self, symbol: str) -> dict | None:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        return {"bid": float(tick.bid), "ask": float(tick.ask), "spread": float(tick.ask - tick.bid)}
