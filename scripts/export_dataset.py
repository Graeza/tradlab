from __future__ import annotations
import pandas as pd
from core.database import MarketDatabase
from config.settings import DB_PATH, PRIMARY_TIMEFRAME, LABEL_HORIZON_BARS

def export(symbol: str, timeframe: int = PRIMARY_TIMEFRAME, out_csv: str = "dataset.csv"):
    db = MarketDatabase(DB_PATH)
    feats = db.load_features(symbol, timeframe, limit=200000)
    if feats.empty:
        raise SystemExit("No features found.")
    # Join labels
    labels = pd.read_sql_query(
        "SELECT time, future_return, y_class FROM labels WHERE symbol=? AND timeframe=? AND horizon_bars=?",
        db.conn,
        params=(symbol, timeframe, LABEL_HORIZON_BARS)
    )
    ds = feats.merge(labels, on="time", how="inner")
    ds.to_csv(out_csv, index=False)
    print(f"Wrote {len(ds)} rows -> {out_csv}")

if __name__ == "__main__":
    export(symbol="EURUSD")
