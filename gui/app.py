from __future__ import annotations

import os
import time
import json
from datetime import datetime
import queue
import subprocess
import sys
import shutil
from typing import Optional

import joblib
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from collections import deque

from core.mt5_worker import MT5Client
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
    ENSEMBLE_MIN_CONF, STRATEGY_WEIGHTS, LABEL_HORIZON_BARS, REGIME_WEIGHT_MULTIPLIERS,
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


class BacktestWorker(QtCore.QObject):
    line = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)  # (ok, message)

    def __init__(self, cmd: list[str]):
        super().__init__()
        self.cmd = cmd

    @QtCore.Slot()
    def run(self):
        try:
            self.line.emit("[BACKTEST] " + " ".join(self.cmd))
            p = subprocess.Popen(
                self.cmd,
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
                self.finished.emit(False, f"Backtest failed (exit={rc})")
                return
            self.finished.emit(True, "Backtest completed")
        except Exception as e:
            self.finished.emit(False, f"Backtest error: {e}")


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
        self.resize(1200, 740)

        self.log = LogPump()
        self.decisions = {}  # symbol -> payload dict

        # --- Equity curve history (last N points) ---
        self.eq_t = deque(maxlen=600)      # timestamps
        self.eq_equity = deque(maxlen=600)
        self.eq_balance = deque(maxlen=600)
        self.eq_last = None  # (equity, balance)

        self.bus = DecisionBus()
        self.bus.decision.connect(self.on_decision)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # --- Controls row ---
        top = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start Bot")
        self.btn_stop = QtWidgets.QPushButton("Stop Bot")
        self.btn_stop.setEnabled(False)

        self.chk_allow = QtWidgets.QCheckBox("Allow New Trades")
        self.chk_allow.setChecked(True)

        # MT5 status badge (auto-updated)
        self.lbl_mt5_badge = QtWidgets.QLabel("MT5: —")
        self.lbl_mt5_badge.setStyleSheet(
            "padding:4px 8px; border-radius:10px; font-weight:600; background:#555; color:white;"
        )
        self.btn_mt5_reconnect = QtWidgets.QPushButton("Reconnect MT5")

        self.btn_close_pos = QtWidgets.QPushButton("Close POSITIVE")
        self.btn_close_neg = QtWidgets.QPushButton("Close NEGATIVE")
        self.btn_close_all = QtWidgets.QPushButton("Close ALL")
        self.btn_refresh = QtWidgets.QPushButton("Refresh Positions")

        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)
        top.addWidget(self.chk_allow)
        top.addWidget(self.lbl_mt5_badge)
        top.addWidget(self.btn_mt5_reconnect)
        top.addStretch(1)
        top.addWidget(self.btn_close_pos)
        top.addWidget(self.btn_close_neg)
        top.addWidget(self.btn_close_all)
        top.addWidget(self.btn_refresh)
        layout.addLayout(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)

        # --- Log pane ---
        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumBlockCount(6000)
        split.addWidget(self.txt)

        right = QtWidgets.QWidget()
        rlayout = QtWidgets.QVBoxLayout(right)

        tabs = QtWidgets.QTabWidget()
        rlayout.addWidget(tabs)

        # ========== TAB: Positions ==========
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

        # ========== TAB: Strategy Debug ==========
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

        # ========== TAB: Portfolio ==========
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

        # Equity curve chart
        self.eq_plot = pg.PlotWidget()
        self.eq_plot.setBackground(None)
        self.eq_plot.showGrid(x=True, y=True, alpha=0.25)
        self.eq_plot.setTitle("Equity / Balance (live)")
        self.eq_plot.setLabel("left", "Value")
        self.eq_plot.setLabel("bottom", "Time (s)")

        self.eq_curve_equity = self.eq_plot.plot([], [], pen=pg.mkPen(width=2), name="Equity")
        self.eq_curve_balance = self.eq_plot.plot([], [], pen=pg.mkPen(width=2, style=QtCore.Qt.PenStyle.DashLine), name="Balance")
        self.eq_curve_peak = self.eq_plot.plot([], [], pen=pg.mkPen(width=1), name="Peak Equity")

        port_layout.addWidget(self.eq_plot)
        port_layout.addStretch(1)
        tabs.addTab(port_tab, "Portfolio")

        # ========== TAB: Performance ==========
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

        # ========== TAB: Strategies ==========
        strat_tab = QtWidgets.QWidget()
        strat_layout = QtWidgets.QVBoxLayout(strat_tab)

        self.chk_use_rsi = QtWidgets.QCheckBox("Enable RSI/EMA strategy")
        self.chk_use_rsi.setChecked(True)
        self.chk_use_breakout = QtWidgets.QCheckBox("Enable Breakout strategy")
        self.chk_use_breakout.setChecked(True)
        self.chk_use_ml = QtWidgets.QCheckBox("Enable ML strategy")
        self.chk_use_ml.setChecked(bool(USE_ML_STRATEGY and os.path.exists(ML_MODEL_PATH)))

        strat_layout.addWidget(self.chk_use_rsi)
        strat_layout.addWidget(self.chk_use_breakout)
        strat_layout.addWidget(self.chk_use_ml)

        wgrp = QtWidgets.QGroupBox("Strategy Weights (base)")
        wform = QtWidgets.QFormLayout(wgrp)

        self.w_rsi = QtWidgets.QDoubleSpinBox()
        self.w_rsi.setRange(0.0, 100.0)
        self.w_rsi.setSingleStep(0.1)
        self.w_rsi.setValue(float(STRATEGY_WEIGHTS.get("RSIEMAStrategy", 1.0)))

        self.w_breakout = QtWidgets.QDoubleSpinBox()
        self.w_breakout.setRange(0.0, 100.0)
        self.w_breakout.setSingleStep(0.1)
        self.w_breakout.setValue(float(STRATEGY_WEIGHTS.get("BreakoutStrategy", 1.0)))

        self.w_ml = QtWidgets.QDoubleSpinBox()
        self.w_ml.setRange(0.0, 100.0)
        self.w_ml.setSingleStep(0.1)
        self.w_ml.setValue(float(STRATEGY_WEIGHTS.get("MLStrategy", 1.0)))

        wform.addRow("RSI/EMA", self.w_rsi)
        wform.addRow("Breakout", self.w_breakout)
        wform.addRow("ML", self.w_ml)
        strat_layout.addWidget(wgrp)

        self.spin_min_conf = QtWidgets.QDoubleSpinBox()
        self.spin_min_conf.setRange(0.0, 1.0)
        self.spin_min_conf.setSingleStep(0.01)
        self.spin_min_conf.setValue(float(ENSEMBLE_MIN_CONF))
        strat_layout.addWidget(QtWidgets.QLabel("Ensemble Min Confidence"))
        strat_layout.addWidget(self.spin_min_conf)

        btns = QtWidgets.QHBoxLayout()
        self.btn_apply_strat = QtWidgets.QPushButton("Apply Strategy Settings")
        btns.addWidget(self.btn_apply_strat)
        btns.addStretch(1)
        strat_layout.addLayout(btns)

        strat_layout.addStretch(1)
        tabs.addTab(strat_tab, "Strategies")

        # ========== TAB: Experiments ==========
        exp_tab = QtWidgets.QWidget()
        exp_layout = QtWidgets.QVBoxLayout(exp_tab)

        exp_top = QtWidgets.QHBoxLayout()
        self.btn_exp_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_exp_promote = QtWidgets.QPushButton("Promote Selected Model")
        exp_top.addWidget(self.btn_exp_refresh)
        exp_top.addStretch(1)
        exp_top.addWidget(self.btn_exp_promote)
        exp_layout.addLayout(exp_top)

        self.tbl_exp = QtWidgets.QTableWidget(0, 8)
        self.tbl_exp.setHorizontalHeaderLabels([
            "Type", "Time", "Name", "Accuracy", "Macro F1", "Feat Ver", "Feat ID", "Model Path"
        ])
        self.tbl_exp.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        exp_layout.addWidget(self.tbl_exp)

        self.lbl_exp_status = QtWidgets.QLabel("Experiments: —")
        self.lbl_exp_status.setStyleSheet("color: gray;")
        exp_layout.addWidget(self.lbl_exp_status)

        tabs.addTab(exp_tab, "Experiments")

        # ========== TAB: Backtest ==========
        bt_tab = QtWidgets.QWidget()
        bt_layout = QtWidgets.QVBoxLayout(bt_tab)

        bt_form = QtWidgets.QFormLayout()
        bt_layout.addLayout(bt_form)

        self.bt_symbol = QtWidgets.QComboBox()
        self.bt_symbol.addItems(SYMBOL_LIST)
        bt_form.addRow("Symbol", self.bt_symbol)

        self.bt_tfs = QtWidgets.QLineEdit(",".join(str(t) for t in TIMEFRAME_LIST))
        bt_form.addRow("Timeframes (comma)", self.bt_tfs)

        self.bt_primary_tf = QtWidgets.QComboBox()
        for tf in TIMEFRAME_LIST:
            self.bt_primary_tf.addItem(str(tf), tf)
        self.bt_primary_tf.setCurrentText(str(PRIMARY_TIMEFRAME))
        bt_form.addRow("Primary TF", self.bt_primary_tf)

        self.bt_start = QtWidgets.QLineEdit("")
        self.bt_end = QtWidgets.QLineEdit("")
        bt_form.addRow("Start (YYYY-MM-DD, optional)", self.bt_start)
        bt_form.addRow("End (YYYY-MM-DD, optional)", self.bt_end)

        self.btn_run_backtest = QtWidgets.QPushButton("Run Backtest (next open fills)")
        bt_layout.addWidget(self.btn_run_backtest)

        self.lbl_bt_status = QtWidgets.QLabel("Backtest: —")
        self.lbl_bt_status.setStyleSheet("color: gray;")
        bt_layout.addWidget(self.lbl_bt_status)

        bt_layout.addStretch(1)
        tabs.addTab(bt_tab, "Backtest")

        # ========== TAB: ML Training ==========
        train_tab = QtWidgets.QWidget()
        train_layout = QtWidgets.QVBoxLayout(train_tab)

        form = QtWidgets.QFormLayout()
        train_layout.addLayout(form)

        self.train_symbol = QtWidgets.QComboBox()
        self.train_symbol.addItems(SYMBOL_LIST)
        form.addRow("Symbol", self.train_symbol)

        self.train_timeframe = QtWidgets.QComboBox()
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

        # ========== TAB: Risk ==========
        risk_tab = QtWidgets.QWidget()
        risk_layout = QtWidgets.QVBoxLayout(risk_tab)

        risk_form = QtWidgets.QFormLayout()
        risk_layout.addLayout(risk_form)

        def _ds(minv, maxv, step, val):
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(minv, maxv)
            w.setDecimals(6)
            w.setSingleStep(step)
            w.setValue(val)
            return w

        # placeholder defaults; synced after self.risk created
        self.risk_max_risk_pct = _ds(0.01, 20.0, 0.10, 1.0)
        risk_form.addRow("Max risk per trade (%)", self.risk_max_risk_pct)

        self.risk_min_conf = _ds(0.0, 1.0, 0.01, 0.60)
        risk_form.addRow("Min confidence", self.risk_min_conf)

        self.risk_sl_atr = _ds(0.1, 20.0, 0.1, 2.0)
        risk_form.addRow("SL ATR multiplier", self.risk_sl_atr)

        self.risk_tp_rr = _ds(0.1, 20.0, 0.1, 1.5)
        risk_form.addRow("TP R:R", self.risk_tp_rr)

        self.risk_fallback_sl = _ds(0.0001, 0.10, 0.0001, 0.003)
        risk_form.addRow("Fallback SL % of price", self.risk_fallback_sl)

        self.risk_max_spread = QtWidgets.QSpinBox()
        self.risk_max_spread.setRange(0, 10_000)
        self.risk_max_spread.setValue(50)
        risk_form.addRow("Max spread (points)", self.risk_max_spread)

        self.risk_base_dev = QtWidgets.QSpinBox()
        self.risk_base_dev.setRange(0, 10_000)
        self.risk_base_dev.setValue(20)
        risk_form.addRow("Base deviation (points)", self.risk_base_dev)

        self.btn_apply_risk = QtWidgets.QPushButton("Apply Risk Settings")
        risk_layout.addWidget(self.btn_apply_risk)
        risk_layout.addStretch(1)
        tabs.addTab(risk_tab, "Risk")

        # ========== TAB: Execution Guard ==========
        ex_tab = QtWidgets.QWidget()
        ex_layout = QtWidgets.QVBoxLayout(ex_tab)

        ex_form = QtWidgets.QFormLayout()
        ex_layout.addLayout(ex_form)

        self.ex_enable_spread = QtWidgets.QCheckBox("Enable spread filter")
        self.ex_enable_spread.setChecked(True)
        ex_form.addRow(self.ex_enable_spread)

        self.ex_max_spread = QtWidgets.QSpinBox()
        self.ex_max_spread.setRange(0, 10_000)
        self.ex_max_spread.setValue(50)
        ex_form.addRow("Max spread (points)", self.ex_max_spread)

        self.ex_enable_session = QtWidgets.QCheckBox("Enable session filter (local time)")
        self.ex_enable_session.setChecked(False)
        ex_form.addRow(self.ex_enable_session)

        self.ex_session_start = QtWidgets.QSpinBox()
        self.ex_session_start.setRange(0, 23)
        self.ex_session_start.setValue(0)
        ex_form.addRow("Session start hour", self.ex_session_start)

        self.ex_session_end = QtWidgets.QSpinBox()
        self.ex_session_end.setRange(0, 24)
        self.ex_session_end.setValue(24)
        ex_form.addRow("Session end hour", self.ex_session_end)

        self.ex_allow_weekends = QtWidgets.QCheckBox("Allow weekends")
        self.ex_allow_weekends.setChecked(False)
        ex_form.addRow(self.ex_allow_weekends)

        self.ex_max_retries = QtWidgets.QSpinBox()
        self.ex_max_retries.setRange(0, 20)
        self.ex_max_retries.setValue(0)
        ex_form.addRow("Max retries", self.ex_max_retries)

        self.ex_retry_delay = QtWidgets.QSpinBox()
        self.ex_retry_delay.setRange(0, 60_000)
        self.ex_retry_delay.setValue(250)
        ex_form.addRow("Retry delay (ms)", self.ex_retry_delay)

        self.btn_apply_exec_guard = QtWidgets.QPushButton("Apply Execution Guard")
        ex_layout.addWidget(self.btn_apply_exec_guard)

        self.lbl_exec_guard_status = QtWidgets.QLabel("Execution Guard: —")
        self.lbl_exec_guard_status.setStyleSheet("color: gray;")
        ex_layout.addWidget(self.lbl_exec_guard_status)

        ex_layout.addStretch(1)
        tabs.addTab(ex_tab, "Execution Guard")

        split.addWidget(right)
        split.setSizes([740, 430])

        # --- Log timer ---
        self.log.line.connect(self._append_log)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.log.flush)
        self.timer.start()

        # Auto-refresh positions/account/performance
        self.pos_timer = QtCore.QTimer(self)
        self.pos_timer.setInterval(2000)
        self.pos_timer.timeout.connect(self.refresh_positions)
        self.pos_timer.timeout.connect(self.refresh_portfolio)
        self.pos_timer.timeout.connect(self.refresh_performance)
        self.pos_timer.start()

        # --- Init MT5 (single-thread worker) and bot ---
        self.mt5 = MT5Client()
        self.mt5.start()
        self.lbl_status.setText("MT5 connected")

        self.db = MarketDatabase(DB_PATH)
        self.fetcher = DataFetcher(self.mt5)
        self.pipeline = DataPipeline(self.fetcher, self.db)

        strategies = self._build_strategies()
        self.ensemble = EnsembleEngine(
            strategies,
            weights=STRATEGY_WEIGHTS,
            min_conf=ENSEMBLE_MIN_CONF,
            regime_multipliers=REGIME_WEIGHT_MULTIPLIERS,
        )
        self.risk = RiskManager(self.mt5, )
        self.executor = TradeExecutor(self.mt5, )

        # Sync UI defaults with live objects
        self._sync_risk_ui_from_live()
        self._sync_exec_guard_ui_from_live()

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
            decision_callback=lambda symbol, final, outputs: self.bus.decision.emit({
                "symbol": symbol,
                "final": final,
                "outputs": outputs,
            }),
        )

        self.chk_allow.stateChanged.connect(lambda _: self.log.write(f"[UI] Allow New Trades = {self.chk_allow.isChecked()}"))

        self.controller = BotController(self.orch)

        # Trigger initial debug view + experiments
        self.render_debug(self.cmb_symbol.currentText())
        self.refresh_experiments()

        # --- Wire buttons ---
        self.btn_start.clicked.connect(self.start_bot)
        self.btn_stop.clicked.connect(self.stop_bot)
        self.btn_close_pos.clicked.connect(lambda: self.close_mode("positive"))
        self.btn_close_neg.clicked.connect(lambda: self.close_mode("negative"))
        self.btn_close_all.clicked.connect(lambda: self.close_mode("all"))
        self.btn_refresh.clicked.connect(self.refresh_positions)
        self.btn_mt5_reconnect.clicked.connect(self.reconnect_mt5)

        # Strategy/experiments/backtest
        self.btn_apply_strat.clicked.connect(self.apply_strategy_settings)
        self.btn_exp_refresh.clicked.connect(self.refresh_experiments)
        self.btn_exp_promote.clicked.connect(self.promote_selected_model)
        self.btn_run_backtest.clicked.connect(self.run_backtest)

        # Risk/execution guard
        self.btn_apply_risk.clicked.connect(self.apply_risk_settings)
        self.btn_apply_exec_guard.clicked.connect(self.apply_execution_guard)

        # Training buttons
        self.btn_export_ds.clicked.connect(self.export_dataset)
        self.btn_train_ml.clicked.connect(self.train_model)
        self.btn_export_train.clicked.connect(self.export_and_train)
        self.btn_reload_ml.clicked.connect(self.reload_ml_model)

        self.refresh_positions()
        self.refresh_portfolio()
        self.refresh_performance()

        self._train_thread = None
        self.bt_thread = None

    # ---------- Paths / helpers ----------

    def _script_path(self, filename: str) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", filename))

    def _experiments_path(self) -> str:
        try:
            from config.settings import EXPERIMENT_LOG_PATH  # type: ignore
            return str(EXPERIMENT_LOG_PATH)
        except Exception:
            return os.path.join("ml", "experiments", "experiments.jsonl")

    def _set_badge(self, text: str, ok: bool):
        self.lbl_mt5_badge.setText(text)
        if ok:
            self.lbl_mt5_badge.setStyleSheet(
                "padding:4px 8px; border-radius:10px; font-weight:600; background:#1f7a1f; color:white;"
            )
        else:
            self.lbl_mt5_badge.setStyleSheet(
                "padding:4px 8px; border-radius:10px; font-weight:600; background:#8a1f1f; color:white;"
            )

    def _sync_risk_ui_from_live(self):
        try:
            self.risk_max_risk_pct.setValue(float(self.risk.max_risk_pct))
            self.risk_min_conf.setValue(float(self.risk.min_confidence))
            self.risk_sl_atr.setValue(float(self.risk.sl_atr_mult))
            self.risk_tp_rr.setValue(float(self.risk.tp_rr))
            self.risk_fallback_sl.setValue(float(self.risk.fallback_sl_pct))
            self.risk_max_spread.setValue(int(self.risk.max_spread_points))
            self.risk_base_dev.setValue(int(self.risk.base_deviation_points))
        except Exception:
            pass

    def _sync_exec_guard_ui_from_live(self):
        try:
            self.ex_enable_spread.setChecked(bool(self.executor.enable_spread_filter))
            self.ex_max_spread.setValue(int(self.executor.max_spread_points))
            self.ex_enable_session.setChecked(bool(self.executor.enable_session_filter))
            self.ex_session_start.setValue(int(self.executor.session_start_hour))
            self.ex_session_end.setValue(int(self.executor.session_end_hour))
            self.ex_allow_weekends.setChecked(bool(self.executor.allow_weekends))
            self.ex_max_retries.setValue(int(self.executor.max_retries))
            self.ex_retry_delay.setValue(int(self.executor.retry_delay_ms))
        except Exception:
            pass

    # ---------- Strategy building / hot updates ----------

    def _build_strategies(self, enabled: Optional[dict] = None):
        enabled = enabled or {
            "RSIEMAStrategy": True,
            "BreakoutStrategy": True,
            "MLStrategy": bool(USE_ML_STRATEGY),
        }

        strategies = []
        if enabled.get("RSIEMAStrategy", True):
            strategies.append(RSIEMAStrategy())
        if enabled.get("BreakoutStrategy", True):
            strategies.append(BreakoutStrategy())

        if enabled.get("MLStrategy", True) and USE_ML_STRATEGY and os.path.exists(ML_MODEL_PATH):
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

    @QtCore.Slot()
    def apply_strategy_settings(self):
        enabled = {
            "RSIEMAStrategy": self.chk_use_rsi.isChecked(),
            "BreakoutStrategy": self.chk_use_breakout.isChecked(),
            "MLStrategy": self.chk_use_ml.isChecked(),
        }

        strategies = self._build_strategies(enabled=enabled)

        # Update ensemble in-place (orchestrator holds reference)
        self.ensemble.strategies = strategies
        self.ensemble.weights = {
            "RSIEMAStrategy": float(self.w_rsi.value()),
            "BreakoutStrategy": float(self.w_breakout.value()),
            "MLStrategy": float(self.w_ml.value()),
        }
        self.ensemble.min_conf = float(self.spin_min_conf.value())

        self.log.write(
            f"[UI] Applied strategies={[s.name for s in strategies]} weights={self.ensemble.weights} min_conf={self.ensemble.min_conf}"
        )

    # ---------- Experiments ----------

    @QtCore.Slot()
    def refresh_experiments(self):
        path = self._experiments_path()
        rows = []

        if not os.path.exists(path):
            self.lbl_exp_status.setText(f"Experiments: log not found: {path}")
            self.tbl_exp.setRowCount(0)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rec = json.loads(ln)
                    except Exception:
                        continue
                    rows.append(rec)
        except Exception as e:
            self.lbl_exp_status.setText(f"Experiments: failed to read: {e}")
            return

        rows = list(reversed(rows))  # newest first
        self.tbl_exp.setRowCount(0)

        def get_metric(rec, key):
            m = rec.get("metrics") or {}
            return m.get(key) if isinstance(m, dict) else rec.get(key)

        for rec in rows[:500]:
            r = self.tbl_exp.rowCount()
            self.tbl_exp.insertRow(r)

            rtype = str(rec.get("type") or "ml")
            ts = str(rec.get("time") or rec.get("timestamp") or "")
            name = str(rec.get("model_version") or rec.get("name") or rec.get("run_name") or "")
            acc = get_metric(rec, "accuracy")
            f1 = get_metric(rec, "macro_f1")
            feat_ver = rec.get("feature_set_version") or rec.get("feature_version") or ""
            feat_id = rec.get("feature_set_id") or rec.get("feature_id") or ""
            model_path = str(rec.get("model_path") or rec.get("artifact_path") or "")

            vals = [rtype, ts, name, acc, f1, feat_ver, feat_id, model_path]
            for c, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem("" if v is None else str(v))
                if c == 7:
                    item.setToolTip(str(v))
                self.tbl_exp.setItem(r, c, item)

        self.lbl_exp_status.setText(f"Experiments: {len(rows)} runs (showing up to 500)")

    @QtCore.Slot()
    def promote_selected_model(self):
        row = self.tbl_exp.currentRow()
        if row < 0:
            self.log.write("[EXP] No experiment selected")
            return

        model_path_item = self.tbl_exp.item(row, 7)
        if not model_path_item:
            self.log.write("[EXP] Selected row has no model path")
            return

        model_path = model_path_item.text().strip()
        if not model_path:
            self.log.write("[EXP] Selected row has empty model path")
            return

        if not os.path.exists(model_path):
            self.log.write(f"[EXP] Model path not found: {model_path}")
            return

        try:
            os.makedirs(os.path.dirname(ML_MODEL_PATH) or ".", exist_ok=True)
            shutil.copy2(model_path, ML_MODEL_PATH)
            self.log.write(f"[EXP] Promoted model -> {ML_MODEL_PATH}")
            self.btn_reload_ml.setEnabled(True)
        except Exception as e:
            self.log.write(f"[EXP] Promote failed: {e}")

    # ---------- Backtest ----------

    @QtCore.Slot()
    def run_backtest(self):
        script = self._script_path("run_backtest.py")
        if not os.path.exists(script):
            self.lbl_bt_status.setText("Backtest: missing scripts/run_backtest.py (apply backtest patch)")
            self.log.write("[BACKTEST] scripts/run_backtest.py not found")
            return

        symbol = self.bt_symbol.currentText()
        tfs = [t.strip() for t in self.bt_tfs.text().split(",") if t.strip()]
        primary_tf = str(self.bt_primary_tf.currentData())

        cmd = [sys.executable, script, "--symbol", symbol, "--primary-tf", primary_tf]
        if tfs:
            cmd += ["--tfs"] + tfs
        if self.bt_start.text().strip():
            cmd += ["--start", self.bt_start.text().strip()]
        if self.bt_end.text().strip():
            cmd += ["--end", self.bt_end.text().strip()]

        self.lbl_bt_status.setText("Backtest: running…")
        self.lbl_bt_status.setStyleSheet("font-weight:600; color: gray;")
        self.btn_run_backtest.setEnabled(False)

        self.bt_thread = QtCore.QThread(self)
        self.bt_worker = BacktestWorker(cmd)
        self.bt_worker.moveToThread(self.bt_thread)
        self.bt_thread.started.connect(self.bt_worker.run)
        self.bt_worker.line.connect(self._append_log)

        def _done(ok: bool, msg: str):
            self.btn_run_backtest.setEnabled(True)
            self.lbl_bt_status.setText(f"Backtest: {msg}")
            self.lbl_bt_status.setStyleSheet(
                "font-weight:600; color: green;" if ok else "font-weight:600; color: red;"
            )
            try:
                self.refresh_experiments()
            except Exception:
                pass
            self.bt_thread.quit()
            self.bt_thread.wait(1500)

        self.bt_worker.finished.connect(_done)
        self.bt_thread.start()

    # ---------- Training helpers ----------

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
            enabled = {
                "RSIEMAStrategy": self.chk_use_rsi.isChecked(),
                "BreakoutStrategy": self.chk_use_breakout.isChecked(),
                "MLStrategy": self.chk_use_ml.isChecked(),
            }
            strategies = self._build_strategies(enabled=enabled)
            self.ensemble = EnsembleEngine(
                strategies,
                weights=self.ensemble.weights,
                min_conf=self.ensemble.min_conf,
                regime_multipliers=REGIME_WEIGHT_MULTIPLIERS,
            )
            self.orch.ensemble = self.ensemble
            self.log.write("[ML] Reloaded strategies/ensemble")
        except Exception as e:
            self.log.write(f"[ML] Reload failed: {e}")

    # ---------- Bot control ----------

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
    def reconnect_mt5(self):
        try:
            initialize_mt5()
            self.log.write("[MT5] Reconnect requested")
            self.refresh_portfolio()
        except Exception as e:
            self.log.write(f"[MT5] Reconnect failed: {e}")

    # ---------- Risk / execution guard ----------

    @QtCore.Slot()
    def apply_risk_settings(self):
        try:
            self.risk.max_risk_pct = float(self.risk_max_risk_pct.value())
            self.risk.min_confidence = float(self.risk_min_conf.value())
            self.risk.sl_atr_mult = float(self.risk_sl_atr.value())
            self.risk.tp_rr = float(self.risk_tp_rr.value())
            self.risk.fallback_sl_pct = float(self.risk_fallback_sl.value())
            self.risk.max_spread_points = int(self.risk_max_spread.value())
            self.risk.base_deviation_points = int(self.risk_base_dev.value())
            self.log.write(
                f"[RISK] Applied: risk%={self.risk.max_risk_pct} min_conf={self.risk.min_confidence} "
                f"SLx={self.risk.sl_atr_mult} RR={self.risk.tp_rr} max_spread={self.risk.max_spread_points} dev={self.risk.base_deviation_points}"
            )
        except Exception as e:
            self.log.write(f"[RISK] Apply failed: {e}")

    @QtCore.Slot()
    def apply_execution_guard(self):
        try:
            self.executor.enable_spread_filter = bool(self.ex_enable_spread.isChecked())
            self.executor.max_spread_points = int(self.ex_max_spread.value())
            self.executor.enable_session_filter = bool(self.ex_enable_session.isChecked())
            self.executor.session_start_hour = int(self.ex_session_start.value())
            self.executor.session_end_hour = int(self.ex_session_end.value())
            self.executor.allow_weekends = bool(self.ex_allow_weekends.isChecked())
            self.executor.max_retries = int(self.ex_max_retries.value())
            self.executor.retry_delay_ms = int(self.ex_retry_delay.value())

            self.lbl_exec_guard_status.setText(
                f"Execution Guard: spread={'ON' if self.executor.enable_spread_filter else 'OFF'} "
                f"(max {self.executor.max_spread_points}pt), session={'ON' if self.executor.enable_session_filter else 'OFF'} "
                f"({self.executor.session_start_hour}-{self.executor.session_end_hour}), "
                f"retries={self.executor.max_retries}@{self.executor.retry_delay_ms}ms"
            )
            self.log.write("[EXEC_GUARD] Applied execution guard settings")
        except Exception as e:
            self.log.write(f"[EXEC_GUARD] Apply failed: {e}")

    # ---------- UI refreshers ----------

    @QtCore.Slot()
    def refresh_positions(self):
        pos = list_positions(self.mt5, self.mt5)

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

                if c == 2:
                    item.setForeground(QtCore.Qt.GlobalColor.darkGreen if typ == "BUY" else QtCore.Qt.GlobalColor.darkRed)

                if c in (3, 4, 5):
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignRight)
                else:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)

                if c == 5:
                    if profit > 0:
                        item.setForeground(QtCore.Qt.GlobalColor.darkGreen)
                    elif profit < 0:
                        item.setForeground(QtCore.Qt.GlobalColor.darkRed)
                    else:
                        item.setForeground(QtCore.Qt.GlobalColor.darkGray)

                self.tbl.setItem(r, c, item)

        self.tbl.setUpdatesEnabled(True)

        color = "darkgreen" if total_profit > 0 else "darkred" if total_profit < 0 else "gray"
        self.lbl_totals.setText(
            f"Positions: {len(pos)} | Winners: {winners} | Losers: {losers} | "
            f"<span style='color:{color};'>Floating PnL: {total_profit:.2f}</span>"
        )
        self.lbl_status.setText(f"Open positions: {len(pos)}")

    def close_mode(self, mode: str):
        self.log.write(f"[UI] Closing positions: {mode}")
        summary = close_positions(self.mt5, self.mt5, mode=mode)
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

        if self.cmb_symbol.currentText() == sym:
            self.render_debug(sym)

    @QtCore.Slot()
    def refresh_portfolio(self):
        s = get_account_summary(self.mt5)
        if not s.get("ok"):
            err = s.get("error")
            self.lbl_account.setText(f"Account: ERROR - {err}")
            self._set_badge("MT5: DISCONNECTED", ok=False)
            return

        cur = s.get("currency", "")
        self.lbl_account.setText(f"Account: {s.get('login')} @ {s.get('server')} ({cur})")
        self._set_badge(f"MT5: CONNECTED {s.get('login')}@{s.get('server')}", ok=True)

        bal = float(s.get("balance", 0.0) or 0.0)
        eq = float(s.get("equity", 0.0) or 0.0)
        pnl = float(s.get("profit", 0.0) or 0.0)
        m = float(s.get("margin", 0.0) or 0.0)
        fm = float(s.get("margin_free", 0.0) or 0.0)
        ml = float(s.get("margin_level", 0.0) or 0.0)
        lev = int(s.get("leverage", 0) or 0)

        risk_used = (m / eq) * 100.0 if eq > 0 else 0.0

        self.v_balance.setText(f"{bal:.2f} {cur}")
        self.v_equity.setText(f"{eq:.2f} {cur}")

        pnl_color = "darkgreen" if pnl > 0 else "darkred" if pnl < 0 else "gray"
        self.v_profit.setText(f"<span style='color:{pnl_color}; font-weight:600'>{pnl:.2f} {cur}</span>")

        self.v_margin.setText(f"{m:.2f} {cur}")
        self.v_free_margin.setText(f"{fm:.2f} {cur}")
        self.v_margin_level.setText(f"{ml:.2f}")
        self.v_leverage.setText(f"1:{lev}")

        if risk_used < 10:
            ru_color = "darkgreen"
        elif risk_used < 30:
            ru_color = "orange"
        else:
            ru_color = "darkred"
        self.v_risk_used.setText(f"<span style='color:{ru_color}; font-weight:600'>{risk_used:.2f}%</span>")

        # Equity curve update
        last = self.eq_last
        cur_pair = (round(eq, 2), round(bal, 2))
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

            peak_series = []
            peak = float("-inf")
            for v in self.eq_equity:
                if v > peak:
                    peak = v
                peak_series.append(peak)
            self.eq_curve_peak.setData(x, peak_series)

            self.eq_plot.enableAutoRange(axis="y", enable=True)
            self.eq_plot.setTitle(f"Equity / Balance (Equity: {eq:.2f} {cur})")

    def _styled_item(self, text: str, signal: Optional[str] = None) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)
        s = (signal or "").upper()
        if s == "BUY":
            item.setForeground(QtCore.Qt.GlobalColor.darkGreen)
        elif s == "SELL":
            item.setForeground(QtCore.Qt.GlobalColor.darkRed)
        elif s == "HOLD":
            item.setForeground(QtCore.Qt.GlobalColor.darkGray)
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
        return item

    def render_debug(self, symbol: str):
        payload = self.decisions.get(symbol)
        if not payload:
            self.lbl_final.setText("Final: —")
            self.tbl_debug.setRowCount(0)
            return

        final = payload.get("final") or {}
        outputs = payload.get("outputs") or []

        self.lbl_final.setText(
            f"Final: {final.get('signal', '—')} | Conf={final.get('confidence', 0):.2f} | Regime={final.get('regime', '—')}"
        )

        self.tbl_debug.setRowCount(0)
        for o in outputs:
            r = self.tbl_debug.rowCount()
            self.tbl_debug.insertRow(r)

            strat = str(o.get("name", ""))
            sig = str(o.get("signal", ""))
            conf = float(o.get("confidence", 0.0) or 0.0)
            effw = o.get("eff_weight")
            meta = o.get("meta")

            self.tbl_debug.setItem(r, 0, self._styled_item(strat, sig))
            self.tbl_debug.setItem(r, 1, self._styled_item(sig, sig))
            self.tbl_debug.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{conf:.2f}"))
            self.tbl_debug.setItem(r, 3, QtWidgets.QTableWidgetItem("" if effw is None else str(effw)))
            self.tbl_debug.setItem(r, 4, QtWidgets.QTableWidgetItem(str(meta)[:500]))

    @QtCore.Slot()
    def refresh_performance(self):
        rows = self.orch.perf.summary_rows()
        pending_total = sum(len(v) for v in self.orch.perf.pending.values())
        self.lbl_perf_status.setText(f"Performance: {len(rows)} strategies | pending={pending_total}")

        self.tbl_perf.setRowCount(0)
        for row in rows:
            r = self.tbl_perf.rowCount()
            self.tbl_perf.insertRow(r)
            vals = [
                str(row.get("name")),
                str(row.get("n")),
                f"{float(row.get('win_rate', 0.0) or 0.0) * 100.0:.1f}%",
                f"{float(row.get('avg_return', 0.0) or 0.0):.5f}",
                f"{float(row.get('expectancy', 0.0) or 0.0):.5f}",
            ]
            for c, v in enumerate(vals):
                self.tbl_perf.setItem(r, c, QtWidgets.QTableWidgetItem(v))


    def closeEvent(self, event):
        # Ensure background bot + MT5 worker are stopped cleanly on exit.
        try:
            try:
                if getattr(self, "bot", None) is not None:
                    self.bot.stop()
            except Exception:
                pass
            try:
                if getattr(self, "mt5", None) is not None:
                    self.mt5.shutdown()
            except Exception:
                pass
        finally:
            super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
