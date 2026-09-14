from __future__ import annotations

import re
import sqlite3
import threading
from datetime import datetime, timezone
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

def _to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)

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
        conn.row_factory = sqlite3.Row
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
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                stopped_at TEXT,
                duration_seconds INTEGER DEFAULT 0,
                trade_count INTEGER DEFAULT 0,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                total_net REAL DEFAULT 0.0,
                buy_net REAL DEFAULT 0.0,
                sell_net REAL DEFAULT 0.0,
                notes TEXT DEFAULT ''
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_open_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                event_time TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                volume REAL,
                entry_price REAL,
                initial_sl REAL,
                initial_tp REAL,
                last_sl REAL,
                last_tp REAL,
                position_id INTEGER,
                order_ticket INTEGER,
                deal_ticket INTEGER,
                strategy_name TEXT DEFAULT '',
                comment TEXT DEFAULT '',
                raw_result_json TEXT DEFAULT '',
                FOREIGN KEY(session_id) REFERENCES trade_sessions(id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_stop_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                position_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                event_time TEXT NOT NULL,
                event_type TEXT NOT NULL,
                sl REAL,
                tp REAL,
                source TEXT DEFAULT '',
                note TEXT DEFAULT '',
                FOREIGN KEY(session_id) REFERENCES trade_sessions(id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                position_id INTEGER,
                deal_ticket INTEGER,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                volume REAL,
                entry_time TEXT,
                exit_time TEXT,
                entry_price REAL,
                exit_price REAL,
                initial_sl REAL,
                initial_tp REAL,
                last_sl REAL,
                last_tp REAL,
                first_trailing_sl REAL,
                last_trailing_sl REAL,
                gross_profit REAL DEFAULT 0.0,
                commission REAL DEFAULT 0.0,
                swap REAL DEFAULT 0.0,
                fee REAL DEFAULT 0.0,
                net_profit REAL DEFAULT 0.0,
                outcome TEXT,
                strategy_name TEXT,
                comment TEXT,
                UNIQUE(session_id, position_id),
                FOREIGN KEY(session_id) REFERENCES trade_sessions(id)
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

    # -------- Trade Journal --------
    def create_trade_session(self, started_at: datetime | str) -> int:
        started = _to_iso(started_at)
        cur = self.conn.execute(
            "INSERT INTO trade_sessions(started_at) VALUES(?)",
            (started,),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def log_trade_open(
        self,
        session_id: int,
        *,
        event_time: datetime | str | None,
        symbol: str,
        side: str,
        volume: float | None,
        entry_price: float | None,
        initial_sl: float | None,
        initial_tp: float | None,
        position_id: int | None = None,
        order_ticket: int | None = None,
        deal_ticket: int | None = None,
        strategy_name: str = "",
        comment: str = "",
        raw_result_json: str = "",
    ) -> None:
        event_ts = _to_iso(event_time or datetime.now(timezone.utc))
        self.conn.execute(
            """
            INSERT INTO trade_open_events(
                session_id, event_time, symbol, side, volume, entry_price,
                initial_sl, initial_tp, last_sl, last_tp,
                position_id, order_ticket, deal_ticket, strategy_name, comment, raw_result_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                int(session_id), event_ts, str(symbol), str(side), volume, entry_price,
                initial_sl, initial_tp, initial_sl, initial_tp,
                position_id, order_ticket, deal_ticket, strategy_name, comment, raw_result_json
            ),
        )
        if position_id and int(position_id) > 0:
            self.log_trade_stop_event(
                session_id=session_id,
                position_id=int(position_id),
                symbol=symbol,
                event_time=event_ts,
                event_type="OPEN",
                sl=initial_sl,
                tp=initial_tp,
                source="executor",
                note="initial stops",
            )
        self.conn.commit()

    def log_trade_stop_event(
        self,
        *,
        session_id: int,
        position_id: int,
        symbol: str,
        event_time: datetime | str | None,
        event_type: str,
        sl: float | None,
        tp: float | None,
        source: str = "",
        note: str = "",
    ) -> None:
        event_ts = _to_iso(event_time or datetime.now(timezone.utc))
        self.conn.execute(
            """
            INSERT INTO trade_stop_events(session_id, position_id, symbol, event_time, event_type, sl, tp, source, note)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (int(session_id), int(position_id), str(symbol), event_ts, str(event_type), sl, tp, str(source), str(note)),
        )
        self.conn.commit()

    def _stop_summary_for_position(self, session_id: int, position_id: int) -> dict[str, Any]:
        cur = self.conn.execute(
            """
            SELECT event_type, sl, tp, event_time
            FROM trade_stop_events
            WHERE session_id=? AND position_id=?
            ORDER BY event_time ASC, id ASC
            """,
            (int(session_id), int(position_id)),
        )
        rows = cur.fetchall()
        first_trail = None
        last_trail = None
        last_sl = None
        last_tp = None
        for row in rows:
            sl = row["sl"]
            tp = row["tp"]
            if sl is not None:
                last_sl = sl
            if tp is not None:
                last_tp = tp
            if str(row["event_type"]).upper() == "TRAIL" and sl is not None:
                if first_trail is None:
                    first_trail = sl
                last_trail = sl
        return {
            "first_trailing_sl": first_trail,
            "last_trailing_sl": last_trail,
            "last_sl": last_sl,
            "last_tp": last_tp,
        }

    def save_session_report(self, session_id: int, report: dict[str, Any]) -> None:
        trades = list(report.get("trades") or [])
        for tr in trades:
            position_id = int(tr.get("position_id") or 0)
            stop_summary = self._stop_summary_for_position(session_id, position_id) if position_id > 0 else {}
            initial_sl = tr.get("initial_sl")
            initial_tp = tr.get("initial_tp")
            last_sl = stop_summary.get("last_sl", tr.get("last_sl"))
            last_tp = stop_summary.get("last_tp", tr.get("last_tp"))
            first_trailing_sl = stop_summary.get("first_trailing_sl")
            last_trailing_sl = stop_summary.get("last_trailing_sl")
            net = float(tr.get("net", 0.0) or 0.0)
            outcome = "WIN" if net > 0 else "LOSS" if net < 0 else "FLAT"
            self.conn.execute(
                """
                INSERT INTO trade_journal(
                    session_id, position_id, deal_ticket, symbol, side, volume,
                    entry_time, exit_time, entry_price, exit_price,
                    initial_sl, initial_tp, last_sl, last_tp,
                    first_trailing_sl, last_trailing_sl,
                    gross_profit, commission, swap, fee, net_profit, outcome,
                    strategy_name, comment
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id, position_id) DO UPDATE SET
                    deal_ticket=excluded.deal_ticket,
                    symbol=excluded.symbol,
                    side=excluded.side,
                    volume=excluded.volume,
                    entry_time=excluded.entry_time,
                    exit_time=excluded.exit_time,
                    entry_price=excluded.entry_price,
                    exit_price=excluded.exit_price,
                    initial_sl=excluded.initial_sl,
                    initial_tp=excluded.initial_tp,
                    last_sl=excluded.last_sl,
                    last_tp=excluded.last_tp,
                    first_trailing_sl=excluded.first_trailing_sl,
                    last_trailing_sl=excluded.last_trailing_sl,
                    gross_profit=excluded.gross_profit,
                    commission=excluded.commission,
                    swap=excluded.swap,
                    fee=excluded.fee,
                    net_profit=excluded.net_profit,
                    outcome=excluded.outcome,
                    strategy_name=excluded.strategy_name,
                    comment=excluded.comment
                """,
                (
                    int(session_id),
                    position_id if position_id > 0 else None,
                    tr.get("deal_ticket"),
                    str(tr.get("symbol") or ""),
                    str(tr.get("side") or ""),
                    tr.get("volume"),
                    _to_iso(tr.get("open_time")),
                    _to_iso(tr.get("close_time")),
                    tr.get("open_price"),
                    tr.get("close_price"),
                    initial_sl,
                    initial_tp,
                    last_sl,
                    last_tp,
                    first_trailing_sl,
                    last_trailing_sl,
                    tr.get("profit"),
                    tr.get("commission"),
                    tr.get("swap"),
                    tr.get("fee"),
                    net,
                    outcome,
                    tr.get("strategy_name"),
                    tr.get("comment"),
                ),
            )

        start = report.get("start")
        stop = report.get("stop")
        duration_seconds = 0
        if isinstance(start, datetime) and isinstance(stop, datetime):
            duration_seconds = max(0, int((stop - start).total_seconds()))
        total_net = float(report.get("total_net", 0.0) or 0.0)
        buy_net = float(report.get("buy_net", 0.0) or 0.0)
        sell_net = float(report.get("sell_net", 0.0) or 0.0)
        win_count = sum(1 for tr in trades if float(tr.get("net", 0.0) or 0.0) > 0)
        loss_count = sum(1 for tr in trades if float(tr.get("net", 0.0) or 0.0) < 0)
        self.conn.execute(
            """
            UPDATE trade_sessions
            SET stopped_at=?, duration_seconds=?, trade_count=?, win_count=?, loss_count=?,
                total_net=?, buy_net=?, sell_net=?
            WHERE id=?
            """,
            (
                _to_iso(stop), duration_seconds, len(trades), win_count, loss_count,
                total_net, buy_net, sell_net, int(session_id)
            ),
        )
        self.conn.commit()

    def list_trade_sessions(self, limit: int = 200) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT id, started_at, stopped_at, duration_seconds, trade_count, win_count, loss_count,
                   total_net, buy_net, sell_net, notes
            FROM trade_sessions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        return [dict(r) for r in cur.fetchall()]

    def list_journal_trades(self, session_id: int) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT *
            FROM trade_journal
            WHERE session_id=?
            ORDER BY COALESCE(exit_time, entry_time) DESC, id DESC
            """,
            (int(session_id),),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_open_event_for_position(self, session_id: int, position_id: int) -> dict[str, Any] | None:
        cur = self.conn.execute(
            """
            SELECT *
            FROM trade_open_events
            WHERE session_id=? AND position_id=?
            ORDER BY id ASC
            LIMIT 1
            """,
            (int(session_id), int(position_id)),
        )
        row = cur.fetchone()
        return dict(row) if row else None