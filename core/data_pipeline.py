from __future__ import annotations

import pandas as pd
from core.data_fetcher import DataFetcher
from core.database import MarketDatabase
from core.features import build_features

class DataPipeline:
    def __init__(self, fetcher: DataFetcher, db: MarketDatabase):
        self.fetcher = fetcher
        self.db = db

    def update_symbol(self, symbol: str, timeframes: list[int], n_bars: int = 2000) -> dict[int, pd.DataFrame]:
        """Fetch new bars, upsert them, build features, upsert features.

        Returns dict: timeframe -> latest features df
        """

        out: dict[int, pd.DataFrame] = {}
        for tf in timeframes:
            last = self.db.get_last_bar_time(symbol, tf)
            new_bars = self.fetcher.fetch_since(symbol, tf, last_time_s=last, n_bars=n_bars)

            # keep only core bar cols
            if not new_bars.empty:
                cols = ["time","open","high","low","close","tick_volume","spread","real_volume"]
                for c in cols:
                    if c not in new_bars.columns:
                        new_bars[c] = None
                new_bars = new_bars[cols].drop_duplicates(subset=["time"]).sort_values("time")
                self.db.upsert_bars(new_bars, symbol, tf)

            # load enough bars to compute indicators safely
            bars = self.db.load_bars(symbol, tf, limit=3000)
            if bars.empty:
                continue
            feats = build_features(bars)
            # keep numeric & categorical features; store everything except dt index
            if "dt" in feats.columns:
                feats = feats.drop(columns=["dt"])
            self.db.upsert_features(feats, symbol, tf)
            out[tf] = feats

        return out
