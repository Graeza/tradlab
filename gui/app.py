from __future__ import annotations

import os
import time
from datetime import datetime
import queue
import subprocess
import sys
import joblib
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from collections import deque

from core.mt5_init import initialize_mt5
from core.data_fetcher import DataFetcher
from core.database import MarketDatabase
from core.data_pipeline import DataPipeline
from core.ensemble import EnsembleEngine
from core.orchestrator import Orchestrator
from core.bot_controller import BotController

from strategies.rsi_ema import RSIEMAStrategy
from strategies.breakout import BreakoutStrategy
from strategies.ml_strategy import MLStrategy

from config.settings import (
    SYMBOL_LIST, TIMEFRAME_LIST, PRIMARY_TIMEFRAME, LOOP_SLEEP_SECONDS,
    DB_PATH, USE_ML_STRATEGY, ML_MODEL_PATH,
    ENSEMBLE_MIN_CONF, STRATEGY_WEIGHTS, LABEL_HORIZON_BARS, REGIME_WEIGHT_MULTIPLIERS
)

from risk_manager import RiskManager
from trade_executor import TradeExecutor
from utils.mt5_positions import close_positions, list_positions
from utils.mt5_account import get_account_summary


class TrainWorker(QtCore.QObject):
    line = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)  # (ok, message)

    def __init__(self, steps: list[list[str]]):
        super().__init__()
        self.steps = steps

    @QtCore.Slot()
    def run(self):
        try:
            for cmd in self.steps:
                self.line.emit("[TRAIN] " + " ".join(cmd))
                p = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert p.stdout is not None
                for ln in p.stdout:
                    ln = ln.rstrip("\n")
                    if ln:
                        self.line.emit(ln)
                rc = p.wait()
                if rc != 0:
                    self.finished.emit(False, f"Command failed (exit={rc}): {' '.join(cmd)}")
                    return
            self.finished.emit(True, "Training completed")
        except Exception as e:
            self.finished.emit(False, f"Training error: {e}")

class LogPump(QtCore.QObject):
    line = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.q: "queue.Queue[str]" = queue.Queue()

    def write(self, s: str):
        s = str(s).rstrip("\n")
        if s:
            self.q.put(s)

    @QtCore.Slot()
    def flush(self):
        while not self.q.empty():
            self.line.emit(self.q.get())

class DecisionBus(QtCore.QObject):
    decision = QtCore.Signal(dict)  # payload dict

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot Dashboard")
        self.resize(1150, 720)

        self.log = LogPump()

        self.decisions = {}  # symbol -> payload dict

        # --- Equity curve history (last N points) ---
        self.eq_t = deque(maxlen=600)      # timestamps
        self.eq_equity = deque(maxlen=600) 
        self.eq_balance = deque(maxlen=600)
        self.eq_last = None  # (equity, balance) last plotted values

        self.bus = DecisionBus()
        self.bus.decision.connect(self.on_decision)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Controls
        top = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start Bot")
        self.btn_stop = QtWidgets.QPushButton("Stop Bot")
        self.btn_stop.setEnabled(False)
        self.chk_allow = QtWidgets.QCheckBox("Allow New Trades")
        self.chk_allow.setChecked(True)  # default

        self.btn_close_pos = QtWidgets.QPushButton("Close POSITIVE")
        self.btn_close_neg = QtWidgets.QPushButton("Close NEGATIVE")
        self.btn_close_all = QtWidgets.QPushButton("Close ALL")
        self.btn_refresh = QtWidgets.QPushButton("Refresh Positions")

        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)
        top.addWidget(self.chk_allow)
        top.addStretch(1)
        top.addWidget(self.btn_close_pos)
        top.addWidget(self.btn_close_neg)
        top.addWidget(self.btn_close_all)
        top.addWidget(self.btn_refresh)
        layout.addLayout(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)

        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumBlockCount(6000)
        split.addWidget(self.txt)

        right = QtWidgets.QWidget()
        rlayout = QtWidgets.QVBoxLayout(right)

        tabs = QtWidgets.QTabWidget()
        rlayout.addWidget(tabs)

        # --- Tab 1: Positions ---
        pos_tab = QtWidgets.QWidget()
        pos_layout = QtWidgets.QVBoxLayout(pos_tab)

        self.tbl = QtWidgets.QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(["Ticket", "Symbol", "Type", "Volume", "Open Price", "Profit"])
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        pos_layout.addWidget(self.tbl)

        self.lbl_totals = QtWidgets.QLabel("Totals: —")
        self.lbl_totals.setStyleSheet("font-weight:600;")
        pos_layout.addWidget(self.lbl_totals)

        self.lbl_status = QtWidgets.QLabel("Starting…")
        pos_layout.addWidget(self.lbl_status)

        tabs.addTab(pos_tab, "Positions")

        # --- Tab 2: Strategy Debug ---
        dbg_tab = QtWidgets.QWidget()
        dbg_layout = QtWidgets.QVBoxLayout(dbg_tab)

        top_dbg = QtWidgets.QHBoxLayout()
        self.cmb_symbol = QtWidgets.QComboBox()
        self.cmb_symbol.addItems(SYMBOL_LIST)
        top_dbg.addWidget(QtWidgets.QLabel("Symbol:"))
        top_dbg.addWidget(self.cmb_symbol)

        self.cmb_symbol.currentTextChanged.connect(self.render_debug)

        top_dbg.addStretch(1)
        dbg_layout.addLayout(top_dbg)

        self.lbl_final = QtWidgets.QLabel("Final: —")
        dbg_layout.addWidget(self.lbl_final)

        self.tbl_debug = QtWidgets.QTableWidget(0, 5)
        self.tbl_debug.setHorizontalHeaderLabels(["Strategy", "Signal", "Confidence", "Eff W", "Meta"])

        header = self.tbl_debug.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)

        dbg_layout.addWidget(self.tbl_debug)

        tabs.addTab(dbg_tab, "Strategy Debug")

        # --- Tab 3: Portfolio ---
        port_tab = QtWidgets.QWidget()
        port_layout = QtWidgets.QVBoxLayout(port_tab)

        self.lbl_account = QtWidgets.QLabel("Account: —")
        self.lbl_account.setStyleSheet("font-weight: 600;")
        port_layout.addWidget(self.lbl_account)

        grid = QtWidgets.QGridLayout()
        port_layout.addLayout(grid)

        def add_row(row, label):
            lab = QtWidgets.QLabel(label)
            val = QtWidgets.QLabel("—")
            val.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(lab, row, 0)
            grid.addWidget(val, row, 1)
            return val

        self.v_balance = add_row(0, "Balance")
        self.v_equity = add_row(1, "Equity")
        self.v_profit = add_row(2, "Floating PnL")
        self.v_margin = add_row(3, "Margin Used")
        self.v_free_margin = add_row(4, "Free Margin")
        self.v_margin_level = add_row(5, "Margin Level (%)")
        self.v_risk_used = add_row(6, "Risk Used (%)")   
        self.v_leverage = add_row(7, "Leverage") 

        # --- Equity Curve Chart ---
        self.eq_plot = pg.PlotWidget()
        self.eq_plot.setBackground(None)  # uses default theme background
        self.eq_plot.showGrid(x=True, y=True, alpha=0.25)
        self.eq_plot.setTitle("Equity / Balance (live)")
        self.eq_plot.setLabel("left", "Value")
        self.eq_plot.setLabel("bottom", "Time (s)")

        # Lines
        self.eq_curve_equity = self.eq_plot.plot([], [], pen=pg.mkPen(width=2), name="Equity")
        self.eq_curve_balance = self.eq_plot.plot([], [], pen=pg.mkPen(width=2, style=QtCore.Qt.PenStyle.DashLine), name="Balance")
        self.eq_curve_peak = self.eq_plot.plot([], [], pen=pg.mkPen(width=1), name="Peak Equity")

        port_layout.addWidget(self.eq_plot)

        port_layout.addStretch(1)
        tabs.addTab(port_tab, "Portfolio")

        # --- Tab 4: Performance ---
        perf_tab = QtWidgets.QWidget()
        perf_layout = QtWidgets.QVBoxLayout(perf_tab)

        self.lbl_perf_status = QtWidgets.QLabel("Performance: —")
        self.lbl_perf_status.setStyleSheet("font-weight:600; color: gray;")
        perf_layout.addWidget(self.lbl_perf_status)

        self.tbl_perf = QtWidgets.QTableWidget(0, 5)
        self.tbl_perf.setHorizontalHeaderLabels(["Name", "N", "Win %", "Avg Ret", "Expectancy"])
        self.tbl_perf.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        perf_layout.addWidget(self.tbl_perf)

        tabs.addTab(perf_tab, "Performance")

        # --- Tab 5: ML Training ---
        train_tab = QtWidgets.QWidget()
        train_layout = QtWidgets.QVBoxLayout(train_tab)

        form = QtWidgets.QFormLayout()
        train_layout.addLayout(form)

        self.train_symbol = QtWidgets.QComboBox()
        self.train_symbol.addItems(SYMBOL_LIST)
        form.addRow("Symbol", self.train_symbol)

        self.train_timeframe = QtWidgets.QComboBox()
        # Use the configured timeframe ints directly
        for tf in TIMEFRAME_LIST:
            self.train_timeframe.addItem(str(tf), tf)
        form.addRow("Timeframe", self.train_timeframe)

        self.train_csv = QtWidgets.QLineEdit("dataset.csv")
        form.addRow("Dataset CSV", self.train_csv)

        self.train_model_version = QtWidgets.QLineEdit(f"ml_{datetime.utcnow().strftime('%Y-%m-%d')}")
        form.addRow("Model version", self.train_model_version)

        self.train_schema_version = QtWidgets.QSpinBox()
        self.train_schema_version.setRange(1, 10_000)
        self.train_schema_version.setValue(1)
        form.addRow("Schema version", self.train_schema_version)

        self.train_strict_schema = QtWidgets.QCheckBox("Strict schema (refuse to trade on drift)")
        self.train_strict_schema.setChecked(True)
        train_layout.addWidget(self.train_strict_schema)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_export_ds = QtWidgets.QPushButton("Export Dataset")
        self.btn_train_ml = QtWidgets.QPushButton("Train Model")
        self.btn_export_train = QtWidgets.QPushButton("Export + Train")
        self.btn_reload_ml = QtWidgets.QPushButton("Reload ML Model")
        self.btn_reload_ml.setEnabled(False)
        btn_row.addWidget(self.btn_export_ds)
        btn_row.addWidget(self.btn_train_ml)
        btn_row.addWidget(self.btn_export_train)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_reload_ml)
        train_layout.addLayout(btn_row)

        self.lbl_train_status = QtWidgets.QLabel("Training: —")
        self.lbl_train_status.setStyleSheet("font-weight:600; color: gray;")
        train_layout.addWidget(self.lbl_train_status)

        train_layout.addStretch(1)
        tabs.addTab(train_tab, "ML Training")

        split.addWidget(right)
        split.setSizes([740, 410])

        # Log timer
        self.log.line.connect(self._append_log)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.log.flush)
        self.timer.start()

        # Auto-refresh positions every 2 seconds
        self.pos_timer = QtCore.QTimer(self)
        self.pos_timer.setInterval(2000)  # ms
        self.pos_timer.timeout.connect(self.refresh_positions)
        self.pos_timer.timeout.connect(self.refresh_portfolio)
        self.pos_timer.timeout.connect(self.refresh_performance)
        self.pos_timer.start()

        # Init MT5 and bot
        initialize_mt5()
        self.lbl_status.setText("MT5 connected")

        self.db = MarketDatabase(DB_PATH)
        self.fetcher = DataFetcher()
        self.pipeline = DataPipeline(self.fetcher, self.db)

        strategies = self._build_strategies()

        self.ensemble = EnsembleEngine(
            strategies,
            weights=STRATEGY_WEIGHTS,
            min_conf=ENSEMBLE_MIN_CONF,
            regime_multipliers=REGIME_WEIGHT_MULTIPLIERS,
        )
        self.risk = RiskManager()
        self.executor = TradeExecutor()

        self.orch = Orchestrator(
            pipeline=self.pipeline,
            ensemble=self.ensemble,
            risk_manager=self.risk,
            executor=self.executor,
            db=self.db,
            symbols=SYMBOL_LIST,
            timeframes=TIMEFRAME_LIST,
            primary_tf=PRIMARY_TIMEFRAME,
            label_horizon_bars=LABEL_HORIZON_BARS,
            log=self.log.write,
            allow_new_trades_getter=lambda: self.chk_allow.isChecked(),
            decision_callback=lambda symbol, final, outputs: self.bus.decision.emit({"symbol": symbol, "final": final, "outputs": outputs})
        )
        self.chk_allow.stateChanged.connect(
            lambda _: self.log.write(f"[UI] Allow New Trades = {self.chk_allow.isChecked()}")
        )
        self.controller = BotController(self.orch)
        # Trigger initial debug view
        self.render_debug(self.cmb_symbol.currentText())
        # Wire buttons
        self.btn_start.clicked.connect(self.start_bot)
        self.btn_stop.clicked.connect(self.stop_bot)
        self.btn_close_pos.clicked.connect(lambda: self.close_mode("positive"))
        self.btn_close_neg.clicked.connect(lambda: self.close_mode("negative"))
        self.btn_close_all.clicked.connect(lambda: self.close_mode("all"))
        self.btn_refresh.clicked.connect(self.refresh_positions)

        # Training buttons
        self.btn_export_ds.clicked.connect(self.export_dataset)
        self.btn_train_ml.clicked.connect(self.train_model)
        self.btn_export_train.clicked.connect(self.export_and_train)
        self.btn_reload_ml.clicked.connect(self.reload_ml_model)

        self.refresh_positions()
        self.refresh_portfolio()
        self.refresh_performance()

        # training thread holder
        self._train_thread = None  # type: QtCore.QThread | None

    def _build_strategies(self):
        strategies = [RSIEMAStrategy(), BreakoutStrategy()]

        if USE_ML_STRATEGY and os.path.exists(ML_MODEL_PATH):
            try:
                bundle = joblib.load(ML_MODEL_PATH)
                if isinstance(bundle, dict) and "model" in bundle:
                    strategies.append(
                        MLStrategy(
                            bundle["model"],
                            feature_cols=bundle.get("feature_cols"),
                            model_version=bundle.get("model_version") or bundle.get("version"),
                            schema_version=bundle.get("schema_version", 1),
                            strict_schema=bundle.get("strict_schema", True),
                            class_to_signal=bundle.get("class_to_signal"),
                            fillna_value=bundle.get("fillna_value"),
                                feature_set_version=bundle.get("feature_set_version"),
                                feature_set_id=bundle.get("feature_set_id"),
                        )
                    )
                else:
                    strategies.append(MLStrategy(bundle))
                self.log.write(f"[ML] Loaded model: {ML_MODEL_PATH}")
            except Exception as e:
                self.log.write(f"[ML] Failed to load model: {e}")
        else:
            self.log.write("[ML] No model found — running without ML")

        return strategies

    def _run_training_steps(self, steps: list[list[str]]):
        if self._train_thread is not None:
            self.log.write("[TRAIN] Training already running")
            return

        self.lbl_train_status.setText("Training: running…")
        self.btn_export_ds.setEnabled(False)
        self.btn_train_ml.setEnabled(False)
        self.btn_export_train.setEnabled(False)
        self.btn_reload_ml.setEnabled(False)

        worker = TrainWorker(steps)
        th = QtCore.QThread(self)
        worker.moveToThread(th)

        worker.line.connect(self.log.write)

        def _done(ok: bool, msg: str):
            self.lbl_train_status.setText(f"Training: {'OK' if ok else 'FAILED'} — {msg}")
            self.btn_export_ds.setEnabled(True)
            self.btn_train_ml.setEnabled(True)
            self.btn_export_train.setEnabled(True)
            self.btn_reload_ml.setEnabled(ok and os.path.exists(ML_MODEL_PATH))
            th.quit()
            th.wait(2000)
            self._train_thread = None

        worker.finished.connect(_done)
        th.started.connect(worker.run)

        self._train_thread = th
        th.start()

    def _script_path(self, filename: str) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", filename))

    @QtCore.Slot()
    def export_dataset(self):
        symbol = self.train_symbol.currentText()
        tf = int(self.train_timeframe.currentData())
        out_csv = self.train_csv.text().strip() or "dataset.csv"

        steps = [[
            sys.executable,
            self._script_path("export_dataset.py"),
            "--symbol", symbol,
            "--timeframe", str(tf),
            "--out", out_csv,
        ]]
        self._run_training_steps(steps)

    @QtCore.Slot()
    def train_model(self):
        out_csv = self.train_csv.text().strip() or "dataset.csv"
        model_version = self.train_model_version.text().strip() or f"ml_{datetime.utcnow().strftime('%Y-%m-%d')}"
        schema_version = int(self.train_schema_version.value())
        strict = self.train_strict_schema.isChecked()

        cmd = [
            sys.executable,
            self._script_path("train_model.py"),
            "--csv", out_csv,
            "--model-version", model_version,
            "--schema-version", str(schema_version),
        ]
        if strict:
            cmd.append("--strict-schema")

        self._run_training_steps([cmd])

    @QtCore.Slot()
    def export_and_train(self):
        symbol = self.train_symbol.currentText()
        tf = int(self.train_timeframe.currentData())
        out_csv = self.train_csv.text().strip() or "dataset.csv"
        model_version = self.train_model_version.text().strip() or f"ml_{datetime.utcnow().strftime('%Y-%m-%d')}"
        schema_version = int(self.train_schema_version.value())
        strict = self.train_strict_schema.isChecked()

        export_cmd = [
            sys.executable,
            self._script_path("export_dataset.py"),
            "--symbol", symbol,
            "--timeframe", str(tf),
            "--out", out_csv,
        ]
        train_cmd = [
            sys.executable,
            self._script_path("train_model.py"),
            "--csv", out_csv,
            "--model-version", model_version,
            "--schema-version", str(schema_version),
        ]
        if strict:
            train_cmd.append("--strict-schema")

        self._run_training_steps([export_cmd, train_cmd])

    @QtCore.Slot()
    def reload_ml_model(self):
        if getattr(self.controller, "is_running", False):
            self.log.write("[ML] Stop the bot before reloading the model")
            return
        try:
            strategies = self._build_strategies()
            self.ensemble = EnsembleEngine(
                strategies,
                weights=STRATEGY_WEIGHTS,
                min_conf=ENSEMBLE_MIN_CONF,
                regime_multipliers=REGIME_WEIGHT_MULTIPLIERS,
            )
            # hot swap on orchestrator (safe because bot is stopped)
            self.orch.ensemble = self.ensemble
            self.log.write("[ML] Reloaded strategies/ensemble")
        except Exception as e:
            self.log.write(f"[ML] Reload failed: {e}")

    @QtCore.Slot()
    def start_bot(self):
        self.controller.start(sleep_s=LOOP_SLEEP_SECONDS)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log.write("[UI] Start pressed")

    @QtCore.Slot()
    def stop_bot(self):
        self.controller.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log.write("[UI] Stop pressed")

    @QtCore.Slot()
    def refresh_positions(self):
        pos = list_positions()

        self.tbl.setUpdatesEnabled(False)
        self.tbl.setRowCount(0)

        total_profit = 0.0
        winners = 0
        losers = 0

        for p in pos:
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)

            typ = "BUY" if int(p.type) == 0 else "SELL"
            profit = float(p.profit)

            total_profit += profit
            if profit > 0:
                winners += 1
            elif profit < 0:
                losers += 1

            vals = [
                str(p.ticket),
                p.symbol,
                typ,
                f"{p.volume:.2f}",
                f"{p.price_open:.5f}",
                f"{profit:.2f}",
            ]

            for c, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(v)

                if c == 2:  # Type column
                    if typ == "BUY":
                        item.setForeground(QtCore.Qt.GlobalColor.darkGreen)
                    else:
                        item.setForeground(QtCore.Qt.GlobalColor.darkRed)

                # Optional: align numbers nicer
                if c in (3, 4, 5):
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignRight)
                else:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)

                # Color profit column
                if c == 5:
                    if profit > 0:
                        item.setForeground(QtCore.Qt.GlobalColor.darkGreen)
                    elif profit < 0:
                        item.setForeground(QtCore.Qt.GlobalColor.darkRed)
                    else:
                        item.setForeground(QtCore.Qt.GlobalColor.darkGray)

                self.tbl.setItem(r, c, item)

        self.tbl.setUpdatesEnabled(True)

        # totals label
        if hasattr(self, "lbl_totals"):
            color = "darkgreen" if total_profit > 0 else "darkred" if total_profit < 0 else "gray"
            self.lbl_totals.setText(
                f"Positions: {len(pos)} | Winners: {winners} | Losers: {losers} | "
                f"<span style='color:{color};'>Floating PnL: {total_profit:.2f}</span>"
            )

        self.lbl_status.setText(f"Open positions: {len(pos)}")

    def close_mode(self, mode: str):
        self.log.write(f"[UI] Closing positions: {mode}")
        summary = close_positions(mode=mode)
        for m in summary.get("closed", []):
            self.log.write("[CLOSE] " + m)
        for m in summary.get("failed", []):
            self.log.write("[CLOSE][FAIL] " + m)
        self.refresh_positions()

    @QtCore.Slot(str)
    def _append_log(self, line: str):
        self.txt.appendPlainText(line)

    @QtCore.Slot(dict)
    def on_decision(self, payload: dict):
        sym = payload.get("symbol")
        if not sym:
            return
        self.decisions[sym] = payload

        # If the currently selected symbol matches, refresh UI
        if self.cmb_symbol.currentText() == sym:
            self.render_debug(sym)

    @QtCore.Slot()
    def refresh_portfolio(self):
        s = get_account_summary()
        if not s.get("ok"):
            self.lbl_account.setText(f"Account: ERROR - {s.get('error')}")
            return

        cur = s.get("currency", "")
        self.lbl_account.setText(f"Account: {s.get('login')} @ {s.get('server')} ({cur})")

        bal = float(s.get("balance", 0.0) or 0.0)
        eq  = float(s.get("equity", 0.0) or 0.0)
        pnl = float(s.get("profit", 0.0) or 0.0)
        m   = float(s.get("margin", 0.0) or 0.0)
        fm  = float(s.get("margin_free", 0.0) or 0.0)
        ml  = float(s.get("margin_level", 0.0) or 0.0)
        lev = int(s.get("leverage", 0) or 0)

        # Risk used (% of equity tied up in margin)
        risk_used = (m / eq) * 100.0 if eq > 0 else 0.0

        # --- Labels ---
        self.v_balance.setText(f"{bal:.2f} {cur}")
        self.v_equity.setText(f"{eq:.2f} {cur}")

        pnl_color = "darkgreen" if pnl > 0 else "darkred" if pnl < 0 else "gray"
        self.v_profit.setText(f"<span style='color:{pnl_color}; font-weight:600'>{pnl:.2f} {cur}</span>")

        self.v_margin.setText(f"{m:.2f} {cur}")
        self.v_free_margin.setText(f"{fm:.2f} {cur}")
        self.v_margin_level.setText(f"{ml:.2f}")
        self.v_leverage.setText(f"1:{lev}")

        # Risk Used color
        if risk_used < 10:
            ru_color = "darkgreen"
        elif risk_used < 30:
            ru_color = "orange"
        else:
            ru_color = "darkred"
        self.v_risk_used.setText(f"<span style='color:{ru_color}; font-weight:600'>{risk_used:.2f}%</span>")

        # --- Equity curve update (append only on change) ---
        if hasattr(self, "eq_curve_equity"):
            last = self.eq_last
            cur_pair = (round(eq, 2), round(bal, 2))  # rounding avoids tiny float noise

            if last != cur_pair:
                self.eq_last = cur_pair

                now = time.time()
                self.eq_t.append(now)
                self.eq_equity.append(eq)
                self.eq_balance.append(bal)

                t0 = self.eq_t[0]
                x = [t - t0 for t in self.eq_t]

                self.eq_curve_equity.setData(x, list(self.eq_equity))
                self.eq_curve_balance.setData(x, list(self.eq_balance))

                # --- Peak equity (drawdown reference) ---
                peak_series = []
                peak = float("-inf")
                for v in self.eq_equity:
                    if v > peak:
                        peak = v
                    peak_series.append(peak)

                self.eq_curve_peak.setData(x, peak_series)

                # Optional: show drawdown in title
                dd = peak - eq
                dd_pct = (dd / peak * 100.0) if peak > 0 else 0.0
                self.eq_plot.setTitle(f"Equity / Balance | Equity: {eq:.2f} {cur} | DD: {dd:.2f} ({dd_pct:.2f}%)")

                self.eq_plot.enableAutoRange(axis="y", enable=True)
                self.eq_plot.setTitle(f"Equity / Balance (Equity: {eq:.2f} {cur})")

    def _styled_item(self, text: str, signal: str | None = None) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)

        s = (signal or "").upper()
        if s == "BUY":
            item.setForeground(QtCore.Qt.GlobalColor.darkGreen)
        elif s == "SELL":
            item.setForeground(QtCore.Qt.GlobalColor.darkRed)
        elif s == "HOLD":
            item.setForeground(QtCore.Qt.GlobalColor.darkGray)

        # make it easier to scan
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
        return item

    def render_debug(self, symbol: str):
        payload = self.decisions.get(symbol)
        if not payload:
            return

        final = payload.get("final", {}) or {}
        outputs = payload.get("outputs", []) or []

        # Optional: show highest-confidence strategies first
        outputs = sorted(outputs, key=lambda o: float(o.get("confidence", 0.0) or 0.0), reverse=True)

        # Final signal/conf
        final_sig = str(final.get("signal", "—")).upper()
        final_conf = float(final.get("confidence", 0.0) or 0.0)

        # Regime info
        reg = final.get("regime", {}) or {}
        trend = str(reg.get("trend", "—"))
        vol = str(reg.get("vol", "—"))
        adx = float(reg.get("adx", 0.0) or 0.0)
        atrp = float(reg.get("atr_pct", 0.0) or 0.0)

        # Single label update (includes regime)
        self.lbl_final.setText(
            f"Final: {final_sig} (conf={final_conf:.2f}) | "
            f"Regime: {trend}/{vol} | ADX={adx:.1f} ATR%={atrp * 100:.2f}%"
        )

        # Color final label
        if final_sig == "BUY":
            self.lbl_final.setStyleSheet("font-weight: 700; color: darkgreen;")
        elif final_sig == "SELL":
            self.lbl_final.setStyleSheet("font-weight: 700; color: darkred;")
        else:
            self.lbl_final.setStyleSheet("font-weight: 700; color: dimgray;")

        # Table
        self.tbl_debug.setRowCount(0)

        for o in outputs:
            r = self.tbl_debug.rowCount()
            self.tbl_debug.insertRow(r)

            name = str(o.get("name", ""))
            sig = str(o.get("signal", "")).upper()
            conf = f"{float(o.get('confidence', 0.0) or 0.0):.2f}"

            meta_dict = o.get("meta") or {}
            eff_w = float(meta_dict.get("effective_weight", 1.0) or 1.0)

            self.tbl_debug.setItem(r, 0, self._styled_item(name))
            self.tbl_debug.setItem(r, 1, self._styled_item(sig, signal=sig))
            self.tbl_debug.setItem(r, 2, self._styled_item(conf, signal=sig))
            # Effective weight coloring
            w_item = QtWidgets.QTableWidgetItem(f"{eff_w:.2f}")
            w_item.setToolTip(
                f"Effective Weight\nBase × Trend × Volatility\n= {eff_w:.2f}"
            )

            if eff_w > 1.05:
                w_item.setForeground(QtCore.Qt.GlobalColor.darkGreen)   # boosted
            elif eff_w < 0.95:
                w_item.setForeground(QtCore.Qt.GlobalColor.darkRed)     # suppressed
            else:
                w_item.setForeground(QtCore.Qt.GlobalColor.darkGray)    # neutral

            w_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            self.tbl_debug.setItem(r, 3, w_item)
            self.tbl_debug.setItem(r, 4, self._styled_item(str(meta_dict)))
    
    @QtCore.Slot()
    def refresh_performance(self):
        rows = self.orch.perf.summary_rows()
        pending_total = sum(len(v) for v in self.orch.perf.pending.values())
        final_n = int(self.orch.perf.stats_final.get("n", 0))

        ts = datetime.now().strftime("%H:%M:%S")
        self.lbl_perf_status.setText(
            f"Performance: scored={final_n} | pending={pending_total} | "
            f"horizon={LABEL_HORIZON_BARS} bars | updated={ts}"
        )

        self.tbl_perf.setUpdatesEnabled(False)
        self.tbl_perf.setRowCount(0)

        for row in rows:
            r = self.tbl_perf.rowCount()
            self.tbl_perf.insertRow(r)

            name = str(row["name"])
            n = str(row["n"])
            win = f"{row['win_rate'] * 100:.1f}"
            avg = f"{row['avg_ret'] * 100:.3f}%"  # percent
            exp = f"{row['expectancy'] * 100:.3f}%"

            self.tbl_perf.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            self.tbl_perf.setItem(r, 1, QtWidgets.QTableWidgetItem(n))
            # Win% with color
            win_rate = float(row["win_rate"])
            win_item = QtWidgets.QTableWidgetItem(win)

            if win_rate > 0.55:
                win_item.setForeground(QtCore.Qt.GlobalColor.darkGreen)
            elif win_rate < 0.45:
                win_item.setForeground(QtCore.Qt.GlobalColor.darkRed)
            else:
                win_item.setForeground(QtCore.Qt.GlobalColor.darkGray)

            win_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            self.tbl_perf.setItem(r, 2, win_item)
            self.tbl_perf.setItem(r, 3, QtWidgets.QTableWidgetItem(avg))
            exp_item = QtWidgets.QTableWidgetItem(exp)

            exp_val = float(row["expectancy"])
            if exp_val > 0:
                exp_item.setForeground(QtCore.Qt.GlobalColor.darkGreen)
            elif exp_val < 0:
                exp_item.setForeground(QtCore.Qt.GlobalColor.darkRed)
            else:
                exp_item.setForeground(QtCore.Qt.GlobalColor.darkGray)

            exp_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            self.tbl_perf.setItem(r, 4, exp_item)
            
        self.tbl_perf.setUpdatesEnabled(True)

def run():
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()

if __name__ == "__main__":
    run()
