from __future__ import annotations

import re
import sqlite3
import threading
from typing import Iterable

import pandas as pd


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_ident(name: str) -> str:
    """Quote an SQLite identifier safely using double quotes.

    We also validate identifiers to avoid SQL injection via column names.
    """
    if not _IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return f'"{name}"'


class MarketDatabase:
    """SQLite storage for bars/features/labels with upsert semantics.

    Threading:
      - SQLite connections should not be shared across threads.
      - We keep a per-thread connection using thread-local storage.
    """

    def __init__(self, db_path: str = "market_data.db") -> None:
        self.db_path = db_path
        self._local = threading.local()

        # Initialize schema once (in the constructing thread) using that thread's connection.
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        # Compatibility with existing code that expects self.conn
        return self._get_conn()

    def close_thread_connection(self) -> None:
        """Close the calling thread's SQLite connection (optional hygiene)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bars (
                symbol TEXT NOT NULL,
                timeframe INTEGER NOT NULL,
                time INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                tick_volume REAL,
                spread REAL,
                real_volume REAL,
                PRIMARY KEY (symbol, timeframe, time)
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                symbol TEXT NOT NULL,
                timeframe INTEGER NOT NULL,
                time INTEGER NOT NULL,
                feature_set_version INTEGER,
                feature_set_id TEXT,
                -- dynamic feature columns are stored here via ALTER TABLE when needed
                PRIMARY KEY (symbol, timeframe, time)
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS labels (
                symbol TEXT NOT NULL,
                timeframe INTEGER NOT NULL,
                time INTEGER NOT NULL,
                horizon_bars INTEGER NOT NULL,
                future_return REAL,
                y_class INTEGER,
                PRIMARY KEY (symbol, timeframe, time, horizon_bars)
            );
            """
        )
        self.ensure_feature_meta_columns()
        self.conn.commit()

    def ensure_feature_meta_columns(self) -> None:
        """Add feature_set_version/feature_set_id columns for older DBs."""
        cur = self.conn.execute("PRAGMA table_info(features);")
        existing = {row[1] for row in cur.fetchall()}
        if "feature_set_version" not in existing:
            self.conn.execute("ALTER TABLE features ADD COLUMN feature_set_version INTEGER;")
        if "feature_set_id" not in existing:
            self.conn.execute("ALTER TABLE features ADD COLUMN feature_set_id TEXT;")

    # -------- Bars --------
    def get_last_bar_time(self, symbol: str, timeframe: int) -> int | None:
        cur = self.conn.execute(
            "SELECT MAX(time) FROM bars WHERE symbol=? AND timeframe=?",
            (symbol, timeframe),
        )
        val = cur.fetchone()[0]
        return int(val) if val is not None else None

    def upsert_bars(self, df: pd.DataFrame, symbol: str, timeframe: int) -> int:
        if df is None or df.empty:
            return 0
        needed = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        for c in needed:
            if c not in df.columns:
                df[c] = None
        rows = [
            (
                symbol,
                timeframe,
                int(r.time),
                float(r.open) if r.open is not None else None,
                float(r.high) if r.high is not None else None,
                float(r.low) if r.low is not None else None,
                float(r.close) if r.close is not None else None,
                float(r.tick_volume) if r.tick_volume is not None else None,
                float(r.spread) if r.spread is not None else None,
                float(r.real_volume) if r.real_volume is not None else None,
            )
            for r in df.itertuples(index=False)
        ]

        self.conn.executemany(
            """INSERT INTO bars(symbol,timeframe,time,open,high,low,close,tick_volume,spread,real_volume)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol,timeframe,time) DO UPDATE SET
                 open=excluded.open,
                 high=excluded.high,
                 low=excluded.low,
                 close=excluded.close,
                 tick_volume=excluded.tick_volume,
                 spread=excluded.spread,
                 real_volume=excluded.real_volume
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    def load_bars(self, symbol: str, timeframe: int, limit: int = 3000) -> pd.DataFrame:
        q = """
            SELECT time, open, high, low, close, tick_volume, spread, real_volume
            FROM bars
            WHERE symbol=? AND timeframe=?
            ORDER BY time DESC
            LIMIT ?
        """
        df = pd.read_sql_query(q, self.conn, params=(symbol, timeframe, limit))
        if df.empty:
            return df
        df = df.sort_values("time")
        df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    # -------- Features --------
    def ensure_feature_columns(self, columns: Iterable[str]) -> None:
        cur = self.conn.execute("PRAGMA table_info(features);")
        existing = {row[1] for row in cur.fetchall()}

        for c in columns:
            if c in ("symbol", "timeframe", "time"):
                continue
            if c in existing:
                continue

            col_type = "REAL"
            if c == "feature_set_version":
                col_type = "INTEGER"
            elif c == "feature_set_id":
                col_type = "TEXT"

            self.conn.execute(f"ALTER TABLE features ADD COLUMN {_quote_ident(c)} {col_type};")
            existing.add(c)

        self.conn.commit()

    def upsert_features(self, df: pd.DataFrame, symbol: str, timeframe: int) -> int:
        if df is None or df.empty:
            return 0
        if "time" not in df.columns:
            raise ValueError("features df must include 'time'")

        df2 = df.copy()
        df2["symbol"] = symbol
        df2["timeframe"] = timeframe

        cols = ["symbol", "timeframe", "time"] + [c for c in df2.columns if c not in ("symbol", "timeframe", "time")]
        self.ensure_feature_columns(cols)

        placeholders = ",".join(["?"] * len(cols))
        col_list = ",".join([_quote_ident(c) for c in cols])
        assignments = ",".join([f"{_quote_ident(c)}=excluded.{_quote_ident(c)}" for c in cols if c not in ("symbol", "timeframe", "time")])

        sql = f"""INSERT INTO features({col_list})
                  VALUES({placeholders})
                  ON CONFLICT(symbol,timeframe,time) DO UPDATE SET {assignments}
               """

        rows = []
        for r in df2[cols].itertuples(index=False, name=None):
            rr = list(r)
            rr[2] = int(rr[2])  # time
            rows.append(tuple(rr))

        self.conn.executemany(sql, rows)
        self.conn.commit()
        return len(rows)

    def load_features(self, symbol: str, timeframe: int, limit: int = 3000) -> pd.DataFrame:
        # Select all columns for this symbol/timeframe
        q = """
            SELECT *
            FROM features
            WHERE symbol=? AND timeframe=?
            ORDER BY time DESC
            LIMIT ?
        """
        df = pd.read_sql_query(q, self.conn, params=(symbol, timeframe, limit))
        if df.empty:
            return df
        df = df.sort_values("time")
        df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    # -------- Labels --------
    def upsert_labels(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        cols = ["symbol", "timeframe", "time", "horizon_bars", "future_return", "y_class"]
        df2 = df[cols].copy()
        sql = """INSERT INTO labels(symbol,timeframe,time,horizon_bars,future_return,y_class)
                 VALUES(?,?,?,?,?,?)
                 ON CONFLICT(symbol,timeframe,time,horizon_bars) DO UPDATE SET
                   future_return=excluded.future_return,
                   y_class=excluded.y_class
              """
        rows = []
        for r in df2.itertuples(index=False, name=None):
            rr = list(r)
            rr[2] = int(rr[2])
            rr[3] = int(rr[3])
            rows.append(tuple(rr))
        self.conn.executemany(sql, rows)
        self.conn.commit()
        return len(rows)

    def get_unlabeled_times(self, symbol: str, timeframe: int, horizon_bars: int, max_rows: int = 5000) -> pd.DataFrame:
        q = """
            SELECT b.time
            FROM bars b
            LEFT JOIN labels l
              ON l.symbol=b.symbol AND l.timeframe=b.timeframe AND l.time=b.time AND l.horizon_bars=?
            WHERE b.symbol=? AND b.timeframe=? AND l.time IS NULL
            ORDER BY b.time ASC
            LIMIT ?
        """
        return pd.read_sql_query(q, self.conn, params=(horizon_bars, symbol, timeframe, max_rows))
