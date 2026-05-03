from __future__ import annotations

import os
import time
import json
import re
from datetime import datetime, timezone
import queue
import subprocess
import sys
import shutil
from typing import Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest import report
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
from core.ml_model_registry import MLModelRegistry

from strategies.rsi_ema import RSIEMAStrategy
from strategies.breakout import BreakoutStrategy
from strategies.ml_strategy import MLStrategy
from strategies.boom_spike_trend import BoomSpikeTrendStrategy
from strategies.boom_sell_decay import BoomSellDecayStrategy

from config.settings import (
    SYMBOL_LIST, TIMEFRAME_LIST, PRIMARY_TIMEFRAME, LOOP_SLEEP_SECONDS,
    DB_PATH, USE_ML_STRATEGY, ML_MODEL_PATH, ML_CANDIDATES_DIR, FEATURE_SET_VERSION, ML_REQUIRE_SYMBOL_MODEL, ML_MIN_CANDIDATE_ACCURACY,
    ENSEMBLE_MIN_CONF, ENSEMBLE_MIN_VOTE_GAP, STRATEGY_WEIGHTS, LABEL_HORIZON_BARS, REGIME_WEIGHT_MULTIPLIERS, 
    DATA_QUALITY_OUT_DIR,BACKTEST_STARTING_CASH, BACKTEST_WARMUP_BARS, BACKTEST_OUT_DIR,
    is_live_trading_allowed, set_live_trading_allowed
)

from risk_manager import RiskManager
from trade_executor import TradeExecutor
from utils.mt5_positions import close_positions, list_positions
from utils.mt5_account import get_account_summary
from utils.indicators import calculate_rsi, calculate_ema, calculate_macd



def _qt_message_filter(mode, context, message):
    if "QObject::connect(QStyleHints, QStyleHints): unique connections require a pointer to member function of a QObject subclass" in message:
        return
    QtCore.qDefaultMessageHandler(mode, context, message)


class TrainWorker(QtCore.QObject):
    line = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)  # (ok, message)

    def __init__(self, steps: list[list[str]], cwd: str | None = None):
        super().__init__()
        self.steps = steps
        self.cwd = cwd
        self._stop = False
        self._proc = None

    @QtCore.Slot()
    def run(self):
        ok = True
        msg = "Training completed"

        try:
            for i, cmd in enumerate(self.steps, start=1):
                if self._stop:
                    ok = False
                    msg = "Cancelled"
                    self.line.emit("[TRAIN] Cancelled.")
                    break

                self.line.emit(f"[TRAIN] Step {i}/{len(self.steps)}: {' '.join(cmd)}")

                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=self.cwd,
                )

                assert self._proc.stdout is not None
                for ln in self._proc.stdout:
                    if self._stop:
                        try:
                            self._proc.terminate()
                        except Exception:
                            pass
                        ok = False
                        msg = "Cancelled"
                        self.line.emit("[TRAIN] Cancelled during subprocess run.")
                        break

                    ln = ln.rstrip("\n")
                    if ln:
                        self.line.emit(ln)

                rc = self._proc.wait()
                self._proc = None

                if self._stop:
                    ok = False
                    msg = "Cancelled"
                    break

                if rc != 0:
                    ok = False
                    msg = f"Step {i} failed (exit code {rc})"
                    self.line.emit(f"[TRAIN] {msg}")
                    break

        except Exception as e:
            ok = False
            msg = f"Worker exception: {e}"
            self.line.emit(f"[TRAIN] {msg}")

        self.finished.emit(ok, msg)

    def stop(self):
        self._stop = True
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass

class BacktestWorker(QtCore.QObject):
    line = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)  # (ok, message)

    def __init__(self, cmd: list[str], cwd: str | None = None):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd

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
                cwd=self.cwd,
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


class ManualOHLCDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None, default_symbol: str = "", default_timeframe: int = 1):
        super().__init__(parent)
        self.setWindowTitle("Manual OHLC Entry")
        self.setModal(True)
        self.resize(420, 320)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.symbol = QtWidgets.QComboBox()
        self.symbol.addItems(SYMBOL_LIST)
        if default_symbol and default_symbol in SYMBOL_LIST:
            self.symbol.setCurrentText(default_symbol)
        form.addRow("Symbol", self.symbol)

        self.timeframe = QtWidgets.QComboBox()
        for tf in TIMEFRAME_LIST:
            self.timeframe.addItem(str(tf), tf)
        idx = self.timeframe.findData(default_timeframe)
        if idx >= 0:
            self.timeframe.setCurrentIndex(idx)
        form.addRow("Timeframe", self.timeframe)

        self.bar_time = QtWidgets.QDateTimeEdit()
        self.bar_time.setCalendarPopup(True)
        self.bar_time.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.bar_time.setTimeSpec(QtCore.Qt.TimeSpec.UTC)
        self.bar_time.setDateTime(QtCore.QDateTime.currentDateTimeUtc())
        form.addRow("Bar Time (UTC)", self.bar_time)

        self.open_price = QtWidgets.QDoubleSpinBox()
        self.open_price.setDecimals(8)
        self.open_price.setMaximum(1_000_000_000)
        form.addRow("Open", self.open_price)

        self.high_price = QtWidgets.QDoubleSpinBox()
        self.high_price.setDecimals(8)
        self.high_price.setMaximum(1_000_000_000)
        form.addRow("High", self.high_price)

        self.low_price = QtWidgets.QDoubleSpinBox()
        self.low_price.setDecimals(8)
        self.low_price.setMaximum(1_000_000_000)
        form.addRow("Low", self.low_price)

        self.close_price = QtWidgets.QDoubleSpinBox()
        self.close_price.setDecimals(8)
        self.close_price.setMaximum(1_000_000_000)
        form.addRow("Close", self.close_price)

        self.tick_volume = QtWidgets.QDoubleSpinBox()
        self.tick_volume.setDecimals(2)
        self.tick_volume.setMaximum(1_000_000_000)
        form.addRow("Tick Volume", self.tick_volume)

        self.spread = QtWidgets.QDoubleSpinBox()
        self.spread.setDecimals(2)
        self.spread.setMaximum(1_000_000)
        form.addRow("Spread", self.spread)

        self.real_volume = QtWidgets.QDoubleSpinBox()
        self.real_volume.setDecimals(2)
        self.real_volume.setMaximum(1_000_000_000)
        form.addRow("Real Volume", self.real_volume)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def payload(self) -> dict:
        return {
            "symbol": self.symbol.currentText().strip(),
            "timeframe": int(self.timeframe.currentData()),
            "time": int(self.bar_time.dateTime().toSecsSinceEpoch()),
            "open": float(self.open_price.value()),
            "high": float(self.high_price.value()),
            "low": float(self.low_price.value()),
            "close": float(self.close_price.value()),
            "tick_volume": float(self.tick_volume.value()),
            "spread": float(self.spread.value()),
            "real_volume": float(self.real_volume.value()),
        }


class AccountSwitchDialog(QtWidgets.QDialog):
    def __init__(self, current_profile: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Switch MT5 Account")
        self.setModal(True)
        self.resize(390, 130)
        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel("Choose your account profile. LIVE may execute real-money orders.")
        info.setWordWrap(True)
        layout.addWidget(info)
        self.profile = QtWidgets.QComboBox()
        self.profile.addItems(["DEMO", "LIVE"])
        idx = self.profile.findText(str(current_profile).upper())
        if idx >= 0:
            self.profile.setCurrentIndex(idx)
        form = QtWidgets.QFormLayout()
        form.addRow("Account Profile", self.profile)
        layout.addLayout(form)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_profile(self) -> str:
        return self.profile.currentText().strip().upper()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot Dashboard")
        self.resize(1200, 740)

        self.log = LogPump()
        self.decisions = {}  # symbol -> payload dict

        self.bot_start_time: datetime | None = None
        self.bot_stop_time: datetime | None = None
        self.trade_session_id: int | None = None

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

        self.chk_allow_live = QtWidgets.QCheckBox("Allow Live Trading")
        self.chk_allow_live.setChecked(bool(is_live_trading_allowed()))
        self.chk_allow_live.setToolTip("Required to execute orders while MT5 is on LIVE profile.")

        # MT5 status badge (auto-updated)
        self.lbl_mt5_badge = QtWidgets.QLabel("MT5: —")
        self.lbl_mt5_badge.setStyleSheet(
            "padding:2px 4px; border-radius:6px; font-weight:600; background:#555; color:white;"
        )
        self.btn_mt5_reconnect = QtWidgets.QPushButton("Reconnect MT5")
        self.btn_switch_account = QtWidgets.QPushButton("Switch Account")
        self.lbl_now = QtWidgets.QLabel("Time: —")
        self.lbl_now.setStyleSheet("font-weight:600; color: gray;")

        self.cmb_close_mode = QtWidgets.QComboBox()
        self.cmb_close_mode.addItems(["Selected", "All", "Profits", "Losses", "Buys", "Sells"])

        self.btn_close_selected = QtWidgets.QPushButton("Close")

        close_row = QtWidgets.QHBoxLayout()
        close_row.addWidget(QtWidgets.QLabel("Close:"))
        close_row.addWidget(self.cmb_close_mode)
        close_row.addWidget(self.btn_close_selected)
        close_row.addStretch(1)   

        self.btn_refresh = QtWidgets.QPushButton("Refresh Positions")

        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)
        top.addWidget(self.chk_allow)
        top.addWidget(self.chk_allow_live)
        top.addWidget(self.lbl_mt5_badge)
        top.addWidget(self.btn_mt5_reconnect)
        top.addWidget(self.btn_switch_account)
        top.addWidget(self.lbl_now)
        top.addStretch(1)
        top.addWidget(self.btn_refresh)
        top.addLayout(close_row)
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

        # ========== TAB: Strategies ==========
        strat_tab = QtWidgets.QWidget()
        strat_layout = QtWidgets.QVBoxLayout(strat_tab)

        self.chk_use_rsi = QtWidgets.QCheckBox("Enable RSI/EMA strategy")
        self.chk_use_rsi.setChecked(True)

        self.chk_use_breakout = QtWidgets.QCheckBox("Enable Breakout strategy")
        self.chk_use_breakout.setChecked(True)

        self.chk_use_ml = QtWidgets.QCheckBox("Enable ML strategy")
        self.chk_use_ml.setChecked(bool(USE_ML_STRATEGY))

        self.chk_use_boom = QtWidgets.QCheckBox("Enable Boom Spike/Trend strategy")
        self.chk_use_boom.setChecked(True)

        self.chk_use_boom_sell = QtWidgets.QCheckBox("Enable Boom Sell Decay strategy")
        self.chk_use_boom_sell.setChecked(True)

        strat_layout.addWidget(self.chk_use_rsi)
        strat_layout.addWidget(self.chk_use_breakout)
        strat_layout.addWidget(self.chk_use_ml)
        strat_layout.addWidget(self.chk_use_boom)
        strat_layout.addWidget(self.chk_use_boom_sell)

        wgrp = QtWidgets.QGroupBox("Strategy Weights (base)")
        wform = QtWidgets.QFormLayout(wgrp)

        self.w_rsi = QtWidgets.QDoubleSpinBox()
        self.w_rsi.setRange(0.0, 100.0)
        self.w_rsi.setSingleStep(0.1)
        self.w_rsi.setValue(float(STRATEGY_WEIGHTS.get("RSI_EMA", 1.0)))

        self.w_breakout = QtWidgets.QDoubleSpinBox()
        self.w_breakout.setRange(0.0, 100.0)
        self.w_breakout.setSingleStep(0.1)
        self.w_breakout.setValue(float(STRATEGY_WEIGHTS.get("BREAKOUT", 1.0)))

        self.w_ml = QtWidgets.QDoubleSpinBox()
        self.w_ml.setRange(0.0, 100.0)
        self.w_ml.setSingleStep(0.1)
        self.w_ml.setValue(float(STRATEGY_WEIGHTS.get("ML", 1.0)))

        self.w_boom = QtWidgets.QDoubleSpinBox()
        self.w_boom.setRange(0.0, 100.0)
        self.w_boom.setSingleStep(0.1)
        self.w_boom.setValue(float(STRATEGY_WEIGHTS.get("BOOM_SPIKE_TREND", 1.3)))

        self.w_boom_sell = QtWidgets.QDoubleSpinBox()
        self.w_boom_sell.setRange(0.0, 100.0)
        self.w_boom_sell.setSingleStep(0.1)
        self.w_boom_sell.setValue(float(STRATEGY_WEIGHTS.get("BOOM_SELL_DECAY", 1.45)))

        wform.addRow("RSI/EMA", self.w_rsi)
        wform.addRow("Breakout", self.w_breakout)
        wform.addRow("ML", self.w_ml)
        wform.addRow("Boom Spike/Trend", self.w_boom)
        wform.addRow("Boom Sell Decay", self.w_boom_sell)
        strat_layout.addWidget(wgrp)

        self.spin_min_conf = QtWidgets.QDoubleSpinBox()
        self.spin_min_conf.setRange(0.0, 1.0)
        self.spin_min_conf.setSingleStep(0.01)
        self.spin_min_conf.setValue(float(ENSEMBLE_MIN_CONF))
        strat_layout.addWidget(QtWidgets.QLabel("Ensemble Min Confidence"))
        strat_layout.addWidget(self.spin_min_conf)

        self.spin_min_vote_gap = QtWidgets.QDoubleSpinBox()
        self.spin_min_vote_gap.setRange(0.0, 1.0)
        self.spin_min_vote_gap.setSingleStep(0.01)
        self.spin_min_vote_gap.setValue(float(ENSEMBLE_MIN_VOTE_GAP))
        strat_layout.addWidget(QtWidgets.QLabel("Minimum Vote Gap"))
        strat_layout.addWidget(self.spin_min_vote_gap)

        btns = QtWidgets.QHBoxLayout()
        self.btn_apply_strat = QtWidgets.QPushButton("Apply Strategy Settings")
        btns.addWidget(self.btn_apply_strat)
        btns.addStretch(1)
        strat_layout.addLayout(btns)

        strat_layout.addStretch(1)
        tabs.addTab(strat_tab, "Strategies")

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

        ex_columns = QtWidgets.QHBoxLayout()
        ex_columns.setSpacing(16)
        ex_layout.addLayout(ex_columns)

        # ----- Left column -----
        ex_left = QtWidgets.QVBoxLayout()
        ex_columns.addLayout(ex_left, 1)

        grp_exec_orders = QtWidgets.QGroupBox("Order Controls")
        form_exec_orders = QtWidgets.QFormLayout(grp_exec_orders)

        self.ex_force_fixed_lot = QtWidgets.QCheckBox("Minimum lot size")
        self.ex_force_fixed_lot.setChecked(True)
        form_exec_orders.addRow(self.ex_force_fixed_lot)

        self.ex_fixed_sl_tp = QtWidgets.QCheckBox("Use SL/TP offset")
        self.ex_fixed_sl_tp.setChecked(True)
        form_exec_orders.addRow(self.ex_fixed_sl_tp)

        self.ex_sl_tp_offset = QtWidgets.QDoubleSpinBox()
        self.ex_sl_tp_offset.setDecimals(2)
        self.ex_sl_tp_offset.setRange(0.0, 1_000_000.0)
        self.ex_sl_tp_offset.setValue(10.0)
        form_exec_orders.addRow("SL/TP offset (price)", self.ex_sl_tp_offset)

        ex_left.addWidget(grp_exec_orders)

        grp_exec_trailing = QtWidgets.QGroupBox("Trailing Stop")
        form_exec_trailing = QtWidgets.QFormLayout(grp_exec_trailing)

        self.ex_enable_trailing = QtWidgets.QCheckBox("Enable trailing stop")
        self.ex_enable_trailing.setChecked(True)
        form_exec_trailing.addRow(self.ex_enable_trailing)

        self.ex_trailing_trigger_rr = QtWidgets.QDoubleSpinBox()
        self.ex_trailing_trigger_rr.setDecimals(2)
        self.ex_trailing_trigger_rr.setRange(0.10, 20.00)
        self.ex_trailing_trigger_rr.setSingleStep(0.10)
        self.ex_trailing_trigger_rr.setValue(0.50)
        form_exec_trailing.addRow("Trail trigger (R)", self.ex_trailing_trigger_rr)

        self.ex_trailing_distance_rr = QtWidgets.QDoubleSpinBox()
        self.ex_trailing_distance_rr.setDecimals(2)
        self.ex_trailing_distance_rr.setRange(0.05, 20.00)
        self.ex_trailing_distance_rr.setSingleStep(0.05)
        self.ex_trailing_distance_rr.setValue(0.50)
        form_exec_trailing.addRow("Trail distance (R)", self.ex_trailing_distance_rr)

        self.ex_trailing_step_rr = QtWidgets.QDoubleSpinBox()
        self.ex_trailing_step_rr.setDecimals(2)
        self.ex_trailing_step_rr.setRange(0.01, 10.00)
        self.ex_trailing_step_rr.setSingleStep(0.01)
        self.ex_trailing_step_rr.setValue(0.10)
        form_exec_trailing.addRow("Trail step (R)", self.ex_trailing_step_rr)

        ex_left.addWidget(grp_exec_trailing)

        grp_exec_filters = QtWidgets.QGroupBox("Market Filters")
        form_exec_filters = QtWidgets.QFormLayout(grp_exec_filters)

        self.ex_enable_spread = QtWidgets.QCheckBox("Enable spread filter")
        self.ex_enable_spread.setChecked(True)
        form_exec_filters.addRow(self.ex_enable_spread)

        self.ex_max_spread = QtWidgets.QSpinBox()
        self.ex_max_spread.setRange(0, 10_000)
        self.ex_max_spread.setValue(50)
        form_exec_filters.addRow("Max spread (points)", self.ex_max_spread)

        self.ex_enable_session = QtWidgets.QCheckBox("Enable session filter (local time)")
        self.ex_enable_session.setChecked(False)
        form_exec_filters.addRow(self.ex_enable_session)

        self.ex_session_start = QtWidgets.QSpinBox()
        self.ex_session_start.setRange(0, 23)
        self.ex_session_start.setValue(0)
        form_exec_filters.addRow("Session start hour", self.ex_session_start)

        self.ex_session_end = QtWidgets.QSpinBox()
        self.ex_session_end.setRange(0, 24)
        self.ex_session_end.setValue(24)
        form_exec_filters.addRow("Session end hour", self.ex_session_end)

        self.ex_allow_weekends = QtWidgets.QCheckBox("Allow weekends")
        self.ex_allow_weekends.setChecked(True)
        form_exec_filters.addRow(self.ex_allow_weekends)

        ex_left.addWidget(grp_exec_filters)

        grp_exec_retry = QtWidgets.QGroupBox("Retries")
        form_exec_retry = QtWidgets.QFormLayout(grp_exec_retry)

        self.ex_max_retries = QtWidgets.QSpinBox()
        self.ex_max_retries.setRange(0, 20)
        self.ex_max_retries.setValue(0)
        form_exec_retry.addRow("Max retries", self.ex_max_retries)

        self.ex_retry_delay = QtWidgets.QSpinBox()
        self.ex_retry_delay.setRange(0, 60_000)
        self.ex_retry_delay.setValue(250)
        form_exec_retry.addRow("Retry delay (ms)", self.ex_retry_delay)

        ex_left.addWidget(grp_exec_retry)
        ex_left.addStretch(1)

        # ----- Right column -----
        ex_right = QtWidgets.QVBoxLayout()
        ex_columns.addLayout(ex_right, 1)

        grp_exec_symbols = QtWidgets.QGroupBox("Symbol Controls")
        form_exec_symbols = QtWidgets.QFormLayout(grp_exec_symbols)

        self.ex_block_symbols = QtWidgets.QListWidget()
        self.ex_block_symbols.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        self.ex_block_symbols.setMinimumHeight(100)

        for sym in SYMBOL_LIST:
            item = QtWidgets.QListWidgetItem(str(sym))
            self.ex_block_symbols.addItem(item)

        form_exec_symbols.addRow("Block trading on symbols", self.ex_block_symbols)

        ex_right.addWidget(grp_exec_symbols)

        grp_exec_positions = QtWidgets.QGroupBox("Position Limits")
        form_exec_positions = QtWidgets.QFormLayout(grp_exec_positions)

        self.ex_single_pos_per_symbol = QtWidgets.QCheckBox("Only one open position per symbol")
        self.ex_single_pos_per_symbol.setChecked(True)
        form_exec_positions.addRow(self.ex_single_pos_per_symbol)

        self.ex_max_pos_per_symbol = QtWidgets.QSpinBox()
        self.ex_max_pos_per_symbol.setRange(1, 100)
        self.ex_max_pos_per_symbol.setValue(2)
        form_exec_positions.addRow("Max positions per symbol", self.ex_max_pos_per_symbol)

        self.ex_max_total_positions = QtWidgets.QSpinBox()
        self.ex_max_total_positions.setRange(0, 500)
        self.ex_max_total_positions.setSpecialValueText("Disabled")
        self.ex_max_total_positions.setValue(0)
        form_exec_positions.addRow("Max total open positions", self.ex_max_total_positions)

        self.ex_one_entry_per_bar = QtWidgets.QCheckBox("Only one entry per closed bar")
        self.ex_one_entry_per_bar.setChecked(False)
        form_exec_positions.addRow(self.ex_one_entry_per_bar)

        self.ex_auto_close_profits = QtWidgets.QCheckBox(
            "Auto-close profits"
        )
        self.ex_auto_close_profits.setChecked(True)
        form_exec_positions.addRow(self.ex_auto_close_profits)

        self.ex_auto_close_profits_threshold = QtWidgets.QDoubleSpinBox()
        self.ex_auto_close_profits_threshold.setDecimals(2)
        self.ex_auto_close_profits_threshold.setRange(0.0, 1_000_000.0)
        self.ex_auto_close_profits_threshold.setValue(0.0)
        form_exec_positions.addRow(
            "Min profit to auto-close",
            self.ex_auto_close_profits_threshold,
        )

        ex_right.addWidget(grp_exec_positions)

        grp_exec_limits = QtWidgets.QGroupBox("Cooldown / Daily Limits")
        form_exec_limits = QtWidgets.QFormLayout(grp_exec_limits)

        self.ex_enable_cooldown = QtWidgets.QCheckBox("Enable symbol cooldown")
        self.ex_enable_cooldown.setChecked(True)
        form_exec_limits.addRow(self.ex_enable_cooldown)

        self.ex_cooldown_minutes = QtWidgets.QSpinBox()
        self.ex_cooldown_minutes.setRange(0, 10_080)
        self.ex_cooldown_minutes.setSpecialValueText("Disabled")
        self.ex_cooldown_minutes.setValue(10)
        form_exec_limits.addRow("Cooldown (minutes)", self.ex_cooldown_minutes)

        self.ex_enable_daily_limits = QtWidgets.QCheckBox("Enable daily trade limits")
        self.ex_enable_daily_limits.setChecked(False)
        form_exec_limits.addRow(self.ex_enable_daily_limits)

        self.ex_daily_trades_per_symbol = QtWidgets.QSpinBox()
        self.ex_daily_trades_per_symbol.setRange(0, 10_000)
        self.ex_daily_trades_per_symbol.setSpecialValueText("Disabled")
        self.ex_daily_trades_per_symbol.setValue(0)
        form_exec_limits.addRow("Max daily trades / symbol", self.ex_daily_trades_per_symbol)

        self.ex_daily_trades_total = QtWidgets.QSpinBox()
        self.ex_daily_trades_total.setRange(0, 10_000)
        self.ex_daily_trades_total.setSpecialValueText("Disabled")
        self.ex_daily_trades_total.setValue(0)
        form_exec_limits.addRow("Max daily trades total", self.ex_daily_trades_total)

        ex_right.addWidget(grp_exec_limits)
        ex_right.addStretch(1)

        self.btn_apply_exec_guard = QtWidgets.QPushButton("Apply Execution Guard")
        ex_layout.addWidget(self.btn_apply_exec_guard)

        self.lbl_exec_guard_status = QtWidgets.QLabel("Execution Guard: —")
        self.lbl_exec_guard_status.setStyleSheet("color: gray;")
        ex_layout.addWidget(self.lbl_exec_guard_status)

        ex_layout.addStretch(1)
        tabs.addTab(ex_tab, "Execution Guard")

        # ========== TAB: Positions ==========
        pos_tab = QtWidgets.QWidget()
        pos_layout = QtWidgets.QVBoxLayout(pos_tab)

        self.tbl = QtWidgets.QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(["Ticket", "Symbol", "Type", "Volume", "Open Price", "Current Price", "Profit"])
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)

        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

        pos_layout.addWidget(self.tbl)

        self.lbl_totals = QtWidgets.QLabel("Totals: —")
        self.lbl_totals.setStyleSheet("font-weight:600;")
        pos_layout.addWidget(self.lbl_totals)

        self.lbl_status = QtWidgets.QLabel("Starting…")
        pos_layout.addWidget(self.lbl_status)

        tabs.addTab(pos_tab, "Positions")

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
        self.lbl_bot_start_time = QtWidgets.QLabel("Start Time: —")
        self.lbl_session_duration = QtWidgets.QLabel("Session Duration: —")
        self.lbl_bot_stop_time = QtWidgets.QLabel("Stop Time: —")
       
        perf_layout.addWidget(self.lbl_bot_start_time)
        perf_layout.addWidget(self.lbl_session_duration)
        perf_layout.addWidget(self.lbl_bot_stop_time)

        self.tbl_perf = QtWidgets.QTableWidget(0, 5)
        self.tbl_perf.setHorizontalHeaderLabels(["Name", "N", "Win %", "Avg Ret", "Expectancy"])
        self.tbl_perf.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        perf_layout.addWidget(self.tbl_perf)

        tabs.addTab(perf_tab, "Performance")

        # ========== TAB: Trade Journal ==========
        journal_tab = QtWidgets.QWidget()
        journal_layout = QtWidgets.QVBoxLayout(journal_tab)

        self.lbl_journal_status = QtWidgets.QLabel("Trade Journal: —")
        self.lbl_journal_status.setStyleSheet("font-weight:600; color: gray;")
        journal_layout.addWidget(self.lbl_journal_status)

        self.tbl_sessions = QtWidgets.QTableWidget(0, 8)
        self.tbl_sessions.setHorizontalHeaderLabels([
            "Session ID", "Started", "Stopped", "Duration", "Trades", "Wins", "Losses", "Net"
        ])
        self.tbl_sessions.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_sessions.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl_sessions.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_sessions.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        journal_layout.addWidget(self.tbl_sessions)

        self.tbl_journal = QtWidgets.QTableWidget(0, 15)
        self.tbl_journal.setHorizontalHeaderLabels([
            "Pos ID", "Symbol", "Side", "Volume", "Entry Time", "Exit Time",
            "Entry", "Exit", "Initial SL", "Initial TP", "Last SL", "Last TP",
            "1st Trail SL", "Last Trail SL", "Net"
        ])
        self.tbl_journal.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_journal.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        journal_layout.addWidget(self.tbl_journal)

        tabs.addTab(journal_tab, "Trade Journal")

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

        self.lbl_debug_ml_model = QtWidgets.QLabel("Resolved ML model: —")
        self.lbl_debug_ml_model.setWordWrap(True)
        self.lbl_debug_ml_model.setStyleSheet("color: gray;")
        dbg_layout.addWidget(self.lbl_debug_ml_model)

        self.tbl_debug = QtWidgets.QTableWidget(0, 5)
        self.tbl_debug.setHorizontalHeaderLabels(["Strategy", "Signal", "Confidence", "Eff W", "Meta"])
        header = self.tbl_debug.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Interactive)
        dbg_layout.addWidget(self.tbl_debug)

        tabs.addTab(dbg_tab, "Strategy Debug")

        # ========== TAB: Backtest ==========
        bt_tab = QtWidgets.QWidget()
        bt_layout = QtWidgets.QVBoxLayout(bt_tab)

        bt_form = QtWidgets.QFormLayout()
        bt_layout.addLayout(bt_form)

        self.bt_symbol = QtWidgets.QComboBox()
        self.bt_symbol.addItems(SYMBOL_LIST)
        bt_form.addRow("Symbol", self.bt_symbol)

        self.bt_selected_only = QtWidgets.QCheckBox("Selected symbol only")
        self.bt_selected_only.setChecked(True)
        bt_form.addRow("Scope", self.bt_selected_only)

        self.bt_tfs = QtWidgets.QLineEdit(",".join(str(t) for t in TIMEFRAME_LIST))
        bt_form.addRow("Timeframes (comma)", self.bt_tfs)

        self.bt_primary_tf = QtWidgets.QComboBox()
        for tf in TIMEFRAME_LIST:
            self.bt_primary_tf.addItem(str(tf), tf)
        self.bt_primary_tf.setCurrentText(str(PRIMARY_TIMEFRAME))
        bt_form.addRow("Primary TF", self.bt_primary_tf)

        self.bt_use_candidate_model = QtWidgets.QCheckBox("Use candidate model path for ML backtests")
        self.bt_use_candidate_model.setChecked(False)
        bt_form.addRow("ML model override", self.bt_use_candidate_model)

        self.bt_ml_model_path = QtWidgets.QLineEdit("")
        self.bt_ml_model_path.setPlaceholderText("Optional .joblib path for candidate model")
        bt_form.addRow("Candidate model path", self.bt_ml_model_path)

        self.bt_start = QtWidgets.QLineEdit("")
        self.bt_end = QtWidgets.QLineEdit("")
        bt_form.addRow("Start (YYYY-MM-DD, optional)", self.bt_start)
        bt_form.addRow("End (YYYY-MM-DD, optional)", self.bt_end)

        bt_btn_row = QtWidgets.QHBoxLayout()

        self.btn_run_backtest = QtWidgets.QPushButton("Run Backtest (next open fills)")
        self.btn_audit_data_gaps = QtWidgets.QPushButton("Audit DB Gaps")
        self.btn_backfill_data_gaps = QtWidgets.QPushButton("Backfill DB Gaps")
        self.btn_manual_ohlc = QtWidgets.QPushButton("Manual OHLC Entry")
        self.btn_bt_plot_chart = QtWidgets.QPushButton("Candlestick Chart")

        bt_btn_row.addWidget(self.btn_run_backtest)
        bt_btn_row.addWidget(self.btn_audit_data_gaps)
        bt_btn_row.addWidget(self.btn_backfill_data_gaps)
        bt_btn_row.addWidget(self.btn_manual_ohlc)
        bt_btn_row.addWidget(self.btn_bt_plot_chart)
        bt_btn_row.addStretch(1)

        bt_layout.addLayout(bt_btn_row)

        bt_chart_row = QtWidgets.QHBoxLayout()
        bt_chart_row.addWidget(QtWidgets.QLabel("Display from:"))
        self.bt_chart_start_date = QtWidgets.QDateEdit()
        self.bt_chart_start_date.setCalendarPopup(True)
        self.bt_chart_start_date.setDisplayFormat("yyyy-MM-dd")
        self.bt_chart_start_date.setDate(QtCore.QDate.currentDate().addMonths(-1))
        bt_chart_row.addWidget(self.bt_chart_start_date)
        bt_chart_row.addWidget(QtWidgets.QLabel("Indicators:"))
        self.bt_ind_rsi = QtWidgets.QCheckBox("RSI")
        self.bt_ind_rsi.setChecked(True)
        self.bt_ind_ema10 = QtWidgets.QCheckBox("EMA10")
        self.bt_ind_ema10.setChecked(True)
        self.bt_ind_ema21 = QtWidgets.QCheckBox("EMA21")
        self.bt_ind_ema21.setChecked(True)
        self.bt_ind_macd = QtWidgets.QCheckBox("MACD")
        self.bt_ind_macd.setChecked(False)
        self.bt_ind_macd_signal = QtWidgets.QCheckBox("MACD Signal")
        self.bt_ind_macd_signal.setChecked(False)
        self.bt_ind_macd_hist = QtWidgets.QCheckBox("MACD Hist")
        self.bt_ind_macd_hist.setChecked(False)
        bt_chart_row.addWidget(self.bt_ind_rsi)
        bt_chart_row.addWidget(self.bt_ind_ema10)
        bt_chart_row.addWidget(self.bt_ind_ema21)
        bt_chart_row.addWidget(self.bt_ind_macd)
        bt_chart_row.addWidget(self.bt_ind_macd_signal)
        bt_chart_row.addWidget(self.bt_ind_macd_hist)
        bt_chart_row.addStretch(1)
        bt_layout.addLayout(bt_chart_row)

        self.bt_chart_widget = pg.GraphicsLayoutWidget()
        self.bt_price_plot = self.bt_chart_widget.addPlot(row=0, col=0)
        self.bt_price_plot.setLabel("left", "Price")
        self.bt_price_plot.showGrid(x=True, y=True, alpha=0.2)
        self.bt_price_plot.setMenuEnabled(True)
        self.bt_price_plot.setMouseEnabled(x=True, y=True)
        self.bt_price_plot.setClipToView(True)
        self.bt_price_plot.addLegend(offset=(10, 10))

        self.bt_indicator_plot = self.bt_chart_widget.addPlot(row=1, col=0)
        self.bt_indicator_plot.setLabel("left", "Indicators")
        self.bt_indicator_plot.setLabel("bottom", "Bars")
        self.bt_indicator_plot.showGrid(x=True, y=True, alpha=0.2)
        self.bt_indicator_plot.setMouseEnabled(x=True, y=True)
        self.bt_indicator_plot.setXLink(self.bt_price_plot)
        self.bt_indicator_plot.addLegend(offset=(10, 10))
        self.bt_chart_widget.ci.layout.setRowStretchFactor(0, 3)
        self.bt_chart_widget.ci.layout.setRowStretchFactor(1, 2)
        bt_layout.addWidget(self.bt_chart_widget, 1)

        self.lbl_bt_status = QtWidgets.QLabel("Backtest: —")
        self.lbl_bt_status.setStyleSheet("color: gray;")
        bt_layout.addWidget(self.lbl_bt_status)
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

        # ========== TAB: Experiments ==========
        exp_tab = QtWidgets.QWidget()
        exp_layout = QtWidgets.QVBoxLayout(exp_tab)

        exp_top = QtWidgets.QHBoxLayout()
        self.btn_exp_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_exp_promote = QtWidgets.QPushButton("Promote Selected Model")
        self.btn_exp_use_for_backtest = QtWidgets.QPushButton("Use Selected Model For Backtest")
        self.btn_exp_use_for_backtest.clicked.connect(self.use_selected_experiment_model_for_backtest)
        exp_top.addWidget(self.btn_exp_refresh)
        exp_top.addStretch(1)
        exp_top.addWidget(self.btn_exp_promote)
        exp_top.addWidget(self.btn_exp_use_for_backtest)
        exp_layout.addLayout(exp_top)

        self.tbl_exp = QtWidgets.QTableWidget(0, 8)
        self.tbl_exp.setHorizontalHeaderLabels([
            "Type", "Time", "Name", "Win Rate / Accuracy (%)", "Profit Factor / Macro F1", "Return % / Feat Ver", "Max DD % / Feat ID", "Model Path"
        ])
        self.tbl_exp.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        exp_layout.addWidget(self.tbl_exp)

        self.lbl_exp_status = QtWidgets.QLabel("Experiments: —")
        self.lbl_exp_status.setStyleSheet("color: gray;")
        exp_layout.addWidget(self.lbl_exp_status)

        tabs.addTab(exp_tab, "Experiments")

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
        self.pos_timer.timeout.connect(self.refresh_clock)
        self.pos_timer.start()

        # --- Init MT5 (single-thread worker) and bot ---
        self.mt5 = MT5Client()
        self.mt5.start()
        self._update_positions_status_label()

        self.db = MarketDatabase(DB_PATH)
        self.fetcher = DataFetcher(self.mt5)
        self.pipeline = DataPipeline(self.fetcher, self.db)

        strategies = self._build_strategies()
        self.ensemble = EnsembleEngine(
            strategies,
            weights=STRATEGY_WEIGHTS,
            min_conf=ENSEMBLE_MIN_CONF,
            min_vote_gap=ENSEMBLE_MIN_VOTE_GAP,
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
            allow_new_trades_getter=lambda: self.chk_allow.isChecked() and (str(getattr(self.mt5, "profile", "DEMO")).upper() != "LIVE" or self.chk_allow_live.isChecked()),
            decision_callback=lambda symbol, final, outputs: self.bus.decision.emit({
                "symbol": symbol,
                "final": final,
                "outputs": outputs,
            }),
            enforce_single_position_per_symbol=bool(self.ex_single_pos_per_symbol.isChecked()),
            max_positions_per_symbol=int(self.ex_max_pos_per_symbol.value()),
            max_total_open_positions=int(self.ex_max_total_positions.value()),
            one_entry_per_closed_bar=bool(self.ex_one_entry_per_bar.isChecked()),
            enable_trade_cooldown=bool(self.ex_enable_cooldown.isChecked()),
            trade_cooldown_minutes=int(self.ex_cooldown_minutes.value()),
            enable_max_daily_trades=bool(self.ex_enable_daily_limits.isChecked()),
            max_daily_trades_per_symbol=int(self.ex_daily_trades_per_symbol.value()),
            max_daily_trades_total=int(self.ex_daily_trades_total.value()),
            auto_close_profits=bool(self.ex_auto_close_profits.isChecked()),
            auto_close_profits_threshold=float(self.ex_auto_close_profits_threshold.value()) if hasattr(self, "ex_auto_close_profits_threshold") else 0.0,
        )

        self.chk_allow.stateChanged.connect(lambda _: self.log.write(f"[UI] Allow New Trades = {self.chk_allow.isChecked()}"))
        self.chk_allow_live.stateChanged.connect(self.on_allow_live_toggled)

        self.controller = BotController(self.orch)
        self.tbl_sessions.itemSelectionChanged.connect(self.on_trade_session_selected)
        self.refresh_trade_sessions()

        # Trigger initial debug view + experiments
        self.render_debug(self.cmb_symbol.currentText())
        self.refresh_experiments()

        # --- Wire buttons ---
        self.btn_start.clicked.connect(self.start_bot)
        self.btn_stop.clicked.connect(self.stop_bot)
        self.btn_close_selected.clicked.connect(self.close_positions_by_mode)
        self.btn_refresh.clicked.connect(self.refresh_positions)
        self.btn_mt5_reconnect.clicked.connect(self.reconnect_mt5)
        self.btn_switch_account.clicked.connect(self.open_account_switch_dialog)

        # Strategy/experiments/backtest
        self.btn_apply_strat.clicked.connect(self.apply_strategy_settings)
        self.btn_exp_refresh.clicked.connect(self.refresh_experiments)
        self.btn_exp_promote.clicked.connect(self.promote_selected_model)
        self.btn_run_backtest.clicked.connect(self.run_backtest)
        self.btn_audit_data_gaps.clicked.connect(self.run_data_gap_audit)
        self.btn_backfill_data_gaps.clicked.connect(self.run_backfill_data_gaps)
        self.btn_manual_ohlc.clicked.connect(self.open_manual_ohlc_dialog)
        self.btn_bt_plot_chart.clicked.connect(self.plot_backtest_chart)

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
        self.refresh_clock()

        self._train_thread = None
        self.bt_thread = None

    # ---------- Paths / helpers ----------

    def _script_path(self, filename: str) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", filename))

    def _experiments_path(self) -> str:
        # Prefer explicit config setting if present; otherwise default to a stable, project-root-relative path.
        # Using an absolute path avoids 'works in terminal but not in GUI' issues due to different CWDs.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        try:
            from config.settings import EXPERIMENT_LOG_PATH  # type: ignore
            p = str(EXPERIMENT_LOG_PATH)
            if not os.path.isabs(p):
                p = os.path.abspath(os.path.join(project_root, p))
            return p
        except Exception:
            return os.path.abspath(os.path.join(project_root, "ml", "experiments", "experiments.jsonl"))

    def _set_badge(self, text: str, ok: bool, live: bool = False):
        self.lbl_mt5_badge.setText(text)
        if live:
            self.lbl_mt5_badge.setStyleSheet(
                "padding:4px 8px; border-radius:10px; font-weight:600; background:#b8860b; color:white;"
            )
        elif ok:
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

            self.ex_enable_trailing.setChecked(bool(getattr(self.executor, "enable_trailing_stop", False)))
            self.ex_trailing_trigger_rr.setValue(float(getattr(self.executor, "trailing_trigger_rr", 1.0)))
            self.ex_trailing_distance_rr.setValue(float(getattr(self.executor, "trailing_distance_rr", 0.5)))
            self.ex_trailing_step_rr.setValue(float(getattr(self.executor, "trailing_step_rr", 0.10)))

            blocked = set(getattr(self.executor, "blocked_symbols", set()) or set())
            for i in range(self.ex_block_symbols.count()):
                item = self.ex_block_symbols.item(i)
                item.setSelected(item.text() in blocked)

            orch = getattr(self, "orch", None)
            if orch is not None:
                self.ex_single_pos_per_symbol.setChecked(bool(getattr(orch, "enforce_single_position_per_symbol", True)))
                self.ex_max_pos_per_symbol.setValue(max(1, int(getattr(orch, "max_positions_per_symbol", 1))))
                self.ex_max_total_positions.setValue(max(0, int(getattr(orch, "max_total_open_positions", 0))))
                self.ex_one_entry_per_bar.setChecked(bool(getattr(orch, "one_entry_per_closed_bar", True)))
                self.ex_enable_cooldown.setChecked(bool(getattr(orch, "enable_trade_cooldown", False)))
                self.ex_cooldown_minutes.setValue(max(0, int(getattr(orch, "trade_cooldown_minutes", 0))))
                self.ex_enable_daily_limits.setChecked(bool(getattr(orch, "enable_max_daily_trades", False)))
                self.ex_daily_trades_per_symbol.setValue(max(0, int(getattr(orch, "max_daily_trades_per_symbol", 0))))
                self.ex_daily_trades_total.setValue(max(0, int(getattr(orch, "max_daily_trades_total", 0))))

        except Exception:
            pass

    def _selected_position_tickets(self) -> list[int]:
        tickets: list[int] = []
        seen: set[int] = set()

        try:
            for item in self.tbl.selectedItems():
                row = item.row()
                ticket_item = self.tbl.item(row, 0)
                if ticket_item is None:
                    continue
                ticket = int((ticket_item.text() or "").strip())
                if ticket <= 0 or ticket in seen:
                    continue
                seen.add(ticket)
                tickets.append(ticket)
        except Exception:
            return []

        return tickets

    @QtCore.Slot()
    def close_positions_by_mode(self):
        try:
            mode = self.cmb_close_mode.currentText().strip().lower()

            if mode == "selected":
                tickets = self._selected_position_tickets()
                if not tickets:
                    self.log.write("[CLOSE] No positions selected. Select one or more rows first.")
                    return
                closed = self.executor.close_positions_by_tickets(tickets)
                self.log.write(f"[CLOSE] mode=selected requested={len(tickets)} closed={closed} tickets={tickets}")
                self.refresh_positions()
                return

            if mode == "all":
                closed = self.executor.close_all_positions()

            elif mode == "profits":
                closed = self.executor.close_positions_in_profit()

            elif mode == "losses":
                closed = self.executor.close_positions_in_loss()

            elif mode == "buys":
                closed = self.executor.close_positions_by_side("BUY")

            elif mode == "sells":
                closed = self.executor.close_positions_by_side("SELL")

            else:
                self.log.write(f"[CLOSE] Unknown close mode: {mode}")
                return

            self.log.write(f"[CLOSE] mode={mode} closed={closed}")
            self.refresh_positions()

        except Exception as e:
            self.log.write(f"[CLOSE] Failed: {e}")

    def _backtest_target_symbols(self) -> list[str]:
        if self.bt_selected_only.isChecked():
            sym = self.bt_symbol.currentText().strip()
            return [sym] if sym else []
        return list(SYMBOL_LIST)

    def _blocked_symbols_set(self) -> set[str]:
        out: set[str] = set()
        try:
            if not hasattr(self, "ex_block_symbols") or self.ex_block_symbols is None:
                return out

            for i in range(self.ex_block_symbols.count()):
                item = self.ex_block_symbols.item(i)
                if item is not None and item.isSelected():
                    sym = item.text().strip()
                    if sym:
                        out.add(sym)
        except Exception:
            pass
        return out

    def _set_backtest_buttons_enabled(self, enabled: bool) -> None:
        self.btn_run_backtest.setEnabled(enabled)
        if hasattr(self, "btn_audit_data_gaps"):
            self.btn_audit_data_gaps.setEnabled(enabled)
        if hasattr(self, "btn_manual_ohlc"):
            self.btn_manual_ohlc.setEnabled(enabled)

    def _run_next_backtest_job(self):
        if not getattr(self, "_bt_queue", None):
            ok_count = sum(1 for ok, _, _ in getattr(self, "_bt_results", []) if ok)
            total = len(getattr(self, "_bt_results", []))
            self._set_backtest_buttons_enabled(True)

            if total == 0:
                self.lbl_bt_status.setText("Backtest: nothing ran")
                self.lbl_bt_status.setStyleSheet("font-weight:600; color: red;")
            elif ok_count == total:
                self.lbl_bt_status.setText(f"Backtest: completed for {ok_count}/{total} symbol(s)")
                self.lbl_bt_status.setStyleSheet("font-weight:600; color: green;")
            else:
                self.lbl_bt_status.setText(f"Backtest: completed with failures ({ok_count}/{total} OK)")
                self.lbl_bt_status.setStyleSheet("font-weight:600; color: orange;")

            try:
                self.refresh_experiments()
            except Exception:
                pass

            self.bt_thread = None
            self.bt_worker = None
            return

        symbol, cmd = self._bt_queue.pop(0)
        self.lbl_bt_status.setText(f"Backtest: running {symbol} ({len(self._bt_results)+1}/{len(self._bt_results)+len(self._bt_queue)+1})")
        self.lbl_bt_status.setStyleSheet("font-weight:600; color: gray;")

        self.bt_thread = QtCore.QThread(self)
        self.bt_worker = BacktestWorker(cmd, cwd=PROJECT_ROOT)
        self.bt_worker.moveToThread(self.bt_thread)

        self.bt_worker.line.connect(self.log.write)
        self.bt_thread.started.connect(self.bt_worker.run)

        def _done(ok: bool, msg: str, _symbol=symbol):
            self._bt_results.append((ok, _symbol, msg))
            color = "green" if ok else "red"
            self.lbl_bt_status.setText(f"Backtest: {_symbol} -> {msg}")
            self.lbl_bt_status.setStyleSheet(f"font-weight:600; color: {color};")
            self.bt_thread.quit()
            self.bt_thread.wait(1500)
            self._run_next_backtest_job()

        self.bt_worker.finished.connect(_done)
        self.bt_thread.start()

    def _safe_fs_name(self, value: str) -> str:
        out = []
        for ch in str(value):
            out.append(ch if ch.isalnum() else "_")
        s = "".join(out)
        while "__" in s:
            s = s.replace("__", "_")
        return s.strip("_")

    def _set_backtest_buttons_enabled(self, enabled: bool) -> None:
        self.btn_run_backtest.setEnabled(enabled)
        if hasattr(self, "btn_audit_data_gaps"):
            self.btn_audit_data_gaps.setEnabled(enabled)
        if hasattr(self, "btn_backfill_data_gaps"):
            self.btn_backfill_data_gaps.setEnabled(enabled)
        if hasattr(self, "btn_manual_ohlc"):
            self.btn_manual_ohlc.setEnabled(enabled)

    def _candidate_model_path(self) -> str:
        symbol = self.train_symbol.currentText().strip()
        tf = int(self.train_timeframe.currentData())
        model_version = (self.train_model_version.text() or "").strip() or f"ml_{datetime.utcnow().strftime('%Y-%m-%d')}"
        schema_version = int(self.train_schema_version.value())

        symbol_safe = self._safe_fs_name(symbol)
        tf_safe = str(tf)
        date_tag = datetime.utcnow().strftime("%Y-%m-%d")

        filename = (
            f"{model_version}"
            f"__{symbol_safe}"
            f"__tf{tf_safe}"
            f"__h{LABEL_HORIZON_BARS}"
            f"__fs{FEATURE_SET_VERSION}"
            f"__sv{schema_version}"
            f"__{date_tag}.joblib"
        )

        return os.path.abspath(
            os.path.join(
                ML_CANDIDATES_DIR,
                symbol_safe,
                f"tf_{tf_safe}",
                filename,
            )
        )

    @QtCore.Slot()
    def use_selected_experiment_model_for_backtest(self):
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

        self.bt_ml_model_path.setText(model_path)
        self.bt_use_candidate_model.setChecked(True)
        self.log.write(f"[EXP] Backtest model path set -> {model_path}")

    def _resolve_debug_ml_model_path(self, symbol: str) -> str:
        try:
            if not getattr(self, "ensemble", None):
                return "—"

            for s in getattr(self.ensemble, "strategies", []):
                if getattr(s, "name", "") == "ML":
                    registry = getattr(s, "bundle_registry", None)
                    if registry is None:
                        return str(getattr(s, "_resolved_model_path", "") or ML_MODEL_PATH or "—")

                    path = registry.resolve_path(str(symbol), int(PRIMARY_TIMEFRAME))
                    return str(path or "No model resolved")
            return "ML strategy not active"
        except Exception as e:
            return f"Resolve failed: {e}"

    # ---------- Strategy building / hot updates ----------
    def _build_strategies(self, enabled: Optional[dict] = None):
        enabled = enabled or {
            "RSIEMAStrategy": True,
            "BreakoutStrategy": True,
            "MLStrategy": bool(USE_ML_STRATEGY),
            "BoomSpikeTrendStrategy": True,
            "BoomSellDecayStrategy": True,
        }

        strategies = []

        if enabled.get("RSIEMAStrategy", True):
            strategies.append(RSIEMAStrategy())

        if enabled.get("BreakoutStrategy", True):
            strategies.append(BreakoutStrategy())

        if enabled.get("BoomSpikeTrendStrategy", True):
            strategies.append(BoomSpikeTrendStrategy())

        if enabled.get("BoomSellDecayStrategy", True):
            strategies.append(BoomSellDecayStrategy())

        if enabled.get("MLStrategy", True) and USE_ML_STRATEGY:
            try:
                registry = MLModelRegistry(
                    candidates_dir=ML_CANDIDATES_DIR,
                    fallback_model_path=ML_MODEL_PATH,
                    require_symbol_model=ML_REQUIRE_SYMBOL_MODEL,
                    min_candidate_accuracy=ML_MIN_CANDIDATE_ACCURACY,
                    log=self.log.write,
                )
                strategies.append(
                    MLStrategy(
                        model=None,
                        bundle_registry=registry,
                        default_primary_tf=int(PRIMARY_TIMEFRAME),
                    )
                )
                self.log.write(
                    f"[ML] Dynamic registry enabled: candidates={ML_CANDIDATES_DIR} fallback={ML_MODEL_PATH}"
                )
            except Exception as e:
                self.log.write(f"[ML] Failed to initialize dynamic ML registry: {e}")
        else:
            self.log.write("[ML] ML strategy disabled")

        return strategies

    @QtCore.Slot()
    def apply_strategy_settings(self):
        enabled = {
            "RSIEMAStrategy": self.chk_use_rsi.isChecked(),
            "BreakoutStrategy": self.chk_use_breakout.isChecked(),
            "MLStrategy": self.chk_use_ml.isChecked(),
            "BoomSpikeTrendStrategy": self.chk_use_boom.isChecked(),
            "BoomSellDecayStrategy": self.chk_use_boom_sell.isChecked(),
        }

        strategies = self._build_strategies(enabled=enabled)

        # Update ensemble in-place (orchestrator holds reference)
        self.ensemble.strategies = strategies
        self.ensemble.weights = {
            "RSI_EMA": float(self.w_rsi.value()),
            "BREAKOUT": float(self.w_breakout.value()),
            "ML": float(self.w_ml.value()),
            "BOOM_SPIKE_TREND": float(self.w_boom.value()),
            "BOOM_SELL_DECAY": float(self.w_boom_sell.value()),
        }
        self.ensemble.min_conf = float(self.spin_min_conf.value())
        self.ensemble.min_vote_gap = float(self.spin_min_vote_gap.value())

        self.log.write(
            f"[UI] Applied strategies={[s.name for s in strategies]} weights={self.ensemble.weights} "
            f"min_conf={self.ensemble.min_conf} min_vote_gap={self.ensemble.min_vote_gap}"
        )
        self.render_debug(self.cmb_symbol.currentText())

    # ---------- Experiments ----------

    @QtCore.Slot()
    def refresh_experiments(self):
        path = self._experiments_path()
        self.log.write(f"[EXP] GUI reading experiments from: {os.path.abspath(path)}")
        print("[EXP][GUI] reading:", os.path.abspath(path))
        rows = []

        if not os.path.exists(path):
            self.lbl_exp_status.setText(f"Experiments: log not found: {path}")
            self.tbl_exp.setRowCount(0)
            self.log.write(f"[EXP] Log not found: {path}")
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
                    if isinstance(rec, dict):
                        rows.append(rec)
        except Exception as e:
            self.lbl_exp_status.setText(f"Experiments: failed to read: {e}")
            self.tbl_exp.setRowCount(0)
            self.log.write(f"[EXP] Failed to read log: {e}")
            return

        self.tbl_exp.setRowCount(0)

        def get_metric(rec, *keys):
            m = rec.get("metrics")
            if isinstance(m, dict):
                for k in keys:
                    if k in m and m[k] is not None:
                        return m[k]
            for k in keys:
                if k in rec and rec[k] is not None:
                    return rec[k]
            return None

        def metric_percent(rec, pct_keys: tuple[str, ...], ratio_keys: tuple[str, ...]):
            v = get_metric(rec, *pct_keys)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    return v
            r = get_metric(rec, *ratio_keys)
            if r is None:
                return None
            try:
                return float(r) * 100.0
            except Exception:
                return r

        def format_cell(v):
            if isinstance(v, bool):
                return str(v)
            if isinstance(v, (int, float)):
                return f"{float(v):.4f}"
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return ""
                try:
                    return f"{float(s):.4f}"
                except Exception:
                    return s
            return "" if v is None else str(v)

        def get_artifact_path(rec):
            arts = rec.get("artifacts")
            if isinstance(arts, dict):
                return (
                    arts.get("model")
                    or arts.get("model_path")
                    or arts.get("output_model_path")
                    or arts.get("metrics")
                    or arts.get("equity_curve")
                    or ""
                )
            return str(
                rec.get("output_model_path")
                or rec.get("model_path")
                or rec.get("artifact_path")
                or ""
            )

        def parse_macro_f1(rec):
            f1 = get_metric(rec, "macro_f1", "f1_macro", "f1")
            if f1 is not None:
                return f1

            report_text = get_metric(rec, "report")
            if not isinstance(report_text, str):
                return None

            # sklearn classification_report includes:
            # macro avg       <precision> <recall> <f1-score> <support>
            m = re.search(
                r"macro avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)",
                report_text,
            )
            if not m:
                return None
            try:
                return float(m.group(3))
            except Exception:
                return None

        def build_ml_name(rec, model_path: str):
            if model_path:
                stem = os.path.splitext(os.path.basename(model_path))[0]
                parts = [p for p in stem.split("__") if p]
                if len(parts) >= 3:
                    name_parts = [parts[1].replace("_", " ")]
                    for p in parts[2:]:
                        if p.startswith(("tf", "h", "fs", "sv")):
                            name_parts.append(p)
                    if name_parts:
                        return " | ".join(name_parts)
                if stem:
                    return stem

            return str(
                rec.get("run_name")
                or rec.get("name")
                or rec.get("model_type")
                or rec.get("model_version")
                or ""
            )

        def build_backtest_name(rec):
            symbol = str(rec.get("symbol") or "").strip()
            tag = str(rec.get("tag") or "").strip()
            if symbol and tag:
                symbol_safe = self._safe_fs_name(symbol)
                cleaned_tag = tag
                suffix = f"_{symbol_safe}"
                if cleaned_tag.endswith(suffix):
                    cleaned_tag = cleaned_tag[: -len(suffix)]
                return f"{symbol} | {cleaned_tag}"

            return str(
                rec.get("name")
                or f"{symbol} [{tag}]".strip()
            )

        def format_experiment_time(raw_ts):
            ts = "" if raw_ts is None else str(raw_ts).strip()
            if not ts:
                return ""
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return f"{dt.strftime('%Y-%m-%d')} | {dt.strftime('%H:%M:%S')}"
            except Exception:
                return ts

        def sort_key(rec):
            return str(
                rec.get("utc_ts")
                or rec.get("time")
                or rec.get("timestamp")
                or rec.get("created_at")
                or ""
            )

        rows = sorted(rows, key=sort_key, reverse=True)

        for rec in rows[:500]:
            r = self.tbl_exp.rowCount()
            self.tbl_exp.insertRow(r)

            rtype = str(rec.get("type") or "ml")
            raw_ts = (
                rec.get("utc_ts")
                or rec.get("time")
                or rec.get("timestamp")
                or ""
            )
            ts = format_experiment_time(raw_ts)

            if rtype == "backtest":
                name = build_backtest_name(rec)
                win_rate = get_metric(rec, "win_rate")
                acc = (float(win_rate) * 100.0) if win_rate is not None else None
                f1 = get_metric(rec, "profit_factor")
                feat_ver = metric_percent(rec, ("total_return_pct",), ("total_return",))
                feat_id = metric_percent(rec, ("max_drawdown_pct",), ("max_drawdown",))
                model_path = str(get_artifact_path(rec))
            else:
                model_path = str(get_artifact_path(rec))
                name = build_ml_name(rec, model_path)
                acc = float(get_metric(rec, "accuracy", "acc") * 100.0)
                f1 = parse_macro_f1(rec)
                feat_ver = metric_percent(rec, ("total_return_pct",), ("total_return",))
                if feat_ver is None:
                    feat_ver = rec.get("feature_set_version") or rec.get("feature_version") or ""
                feat_id = metric_percent(rec, ("max_drawdown_pct",), ("max_drawdown",))
                if feat_id is None:
                    feat_id = rec.get("feature_set_id") or rec.get("feature_id") or ""

            vals = [rtype, ts, name, acc, f1, feat_ver, feat_id, model_path]
            for c, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(format_cell(v))
                if c == 7:
                    item.setToolTip(str(v))
                self.tbl_exp.setItem(r, c, item)

        self.lbl_exp_status.setText(f"Experiments: {len(rows)} runs (showing up to 500)")
        self.log.write(f"[EXP] Loaded {len(rows)} runs from {path}")

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
            self.log.write(f"[EXP] Promoted model: {model_path} -> {ML_MODEL_PATH}")
            self.lbl_exp_status.setText(f"Experiments: promoted -> {os.path.basename(model_path)}")
            self.btn_reload_ml.setEnabled(True)
        except Exception as e:
            self.log.write(f"[EXP] Promote failed: {e}")

    # ---------- Backtest ----------
    def _parse_backtest_date_range(self) -> tuple[Optional[int], Optional[int]]:
        start_ts: Optional[int] = None
        end_ts: Optional[int] = None

        start_text = (self.bt_start.text() or "").strip()
        end_text = (self.bt_end.text() or "").strip()

        if start_text:
            start_dt = datetime.strptime(start_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            start_ts = int(start_dt.timestamp())
        if end_text:
            end_dt = datetime.strptime(end_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_ts = int(end_dt.timestamp()) + (24 * 60 * 60) - 1
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise ValueError("start date is after end date")
        return start_ts, end_ts

    def _load_chart_bars(self, symbol: str, timeframe: int) -> pd.DataFrame:
        df = self.db.load_bars(symbol=symbol, timeframe=timeframe, limit=20_000)
        if df is None or df.empty:
            return pd.DataFrame()

        start_ts, end_ts = self._parse_backtest_date_range()
        chart_start_dt = self.bt_chart_start_date.date().toPython()
        chart_start_ts = int(datetime.combine(chart_start_dt, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        if start_ts is None or chart_start_ts > start_ts:
            start_ts = chart_start_ts
        if start_ts is not None:
            df = df[df["time"] >= int(start_ts)]
        if end_ts is not None:
            df = df[df["time"] <= int(end_ts)]
        return df.reset_index(drop=True)

    def _plot_candles(self, plot: pg.PlotItem, df: pd.DataFrame) -> None:
        x = np.arange(len(df), dtype=float)
        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)

        # Candle wick segments using NaN separators.
        wick_x = np.empty(len(df) * 3, dtype=float)
        wick_y = np.empty(len(df) * 3, dtype=float)
        wick_x[0::3] = x
        wick_x[1::3] = x
        wick_x[2::3] = np.nan
        wick_y[0::3] = l
        wick_y[1::3] = h
        wick_y[2::3] = np.nan
        plot.plot(wick_x, wick_y, pen=pg.mkPen((200, 200, 200), width=1), name="Wicks")

        bull = c >= o
        bear = ~bull
        width = 0.65
        if np.any(bull):
            bull_item = pg.BarGraphItem(
                x=x[bull],
                y0=o[bull],
                y1=c[bull],
                width=width,
                brush=pg.mkBrush(30, 170, 80, 180),
                pen=pg.mkPen(30, 170, 80, 220),
            )
            plot.addItem(bull_item)
        if np.any(bear):
            bear_item = pg.BarGraphItem(
                x=x[bear],
                y0=o[bear],
                y1=c[bear],
                width=width,
                brush=pg.mkBrush(210, 60, 60, 180),
                pen=pg.mkPen(210, 60, 60, 220),
            )
            plot.addItem(bear_item)

    @QtCore.Slot()
    def plot_backtest_chart(self):
        try:
            symbol = self.bt_symbol.currentText().strip()
            timeframe = int(self.bt_primary_tf.currentData())
            if not symbol:
                self.lbl_bt_status.setText("Backtest: select a symbol for charting")
                self.lbl_bt_status.setStyleSheet("font-weight:600; color: red;")
                return

            df = self._load_chart_bars(symbol=symbol, timeframe=timeframe)
            if df.empty:
                self.lbl_bt_status.setText("Backtest: no bars found for selected range")
                self.lbl_bt_status.setStyleSheet("font-weight:600; color: red;")
                return

            # Calculate indicators only once, then display based on checkbox selection.
            df = calculate_ema(df.copy(), "close", 10)
            df = calculate_ema(df, "close", 21)
            df = calculate_rsi(df, 14)
            df = calculate_macd(df)

            self.bt_price_plot.clear()
            self.bt_indicator_plot.clear()
            self.bt_price_plot.addLegend(offset=(10, 10))
            self.bt_indicator_plot.addLegend(offset=(10, 10))
            self._plot_candles(self.bt_price_plot, df)

            x = np.arange(len(df), dtype=float)
            if self.bt_ind_ema10.isChecked() and "close_EMA10" in df.columns:
                self.bt_price_plot.plot(x, df["close_EMA10"].to_numpy(dtype=float), pen=pg.mkPen("#f5c542", width=1.5), name="EMA10")
            if self.bt_ind_ema21.isChecked() and "close_EMA21" in df.columns:
                self.bt_price_plot.plot(x, df["close_EMA21"].to_numpy(dtype=float), pen=pg.mkPen("#3fa7ff", width=1.5), name="EMA21")

            if self.bt_ind_rsi.isChecked() and "RSI" in df.columns:
                self.bt_indicator_plot.plot(x, df["RSI"].to_numpy(dtype=float), pen=pg.mkPen("#b182ff", width=1.5), name="RSI")
                self.bt_indicator_plot.addLine(y=30, pen=pg.mkPen((150, 150, 150), style=QtCore.Qt.PenStyle.DashLine))
                self.bt_indicator_plot.addLine(y=70, pen=pg.mkPen((150, 150, 150), style=QtCore.Qt.PenStyle.DashLine))

            if self.bt_ind_macd.isChecked() and "MACD" in df.columns:
                self.bt_indicator_plot.plot(x, df["MACD"].to_numpy(dtype=float), pen=pg.mkPen("#00d1b2", width=1.3), name="MACD")
            if self.bt_ind_macd_signal.isChecked() and "MACD_Signal" in df.columns:
                self.bt_indicator_plot.plot(x, df["MACD_Signal"].to_numpy(dtype=float), pen=pg.mkPen("#ff8c42", width=1.3), name="MACD Signal")
            if self.bt_ind_macd_hist.isChecked() and "MACD_Hist" in df.columns:
                hist = df["MACD_Hist"].fillna(0.0).to_numpy(dtype=float)
                self.bt_indicator_plot.addItem(
                    pg.BarGraphItem(
                        x=x,
                        y0=np.zeros_like(hist),
                        y1=hist,
                        width=0.65,
                        brush=pg.mkBrush(120, 120, 255, 120),
                        pen=pg.mkPen(120, 120, 255, 180),
                    )
                )

            self.bt_price_plot.setTitle(f"{symbol} TF {timeframe} Candles (drag to pan, wheel to zoom)")
            self.bt_indicator_plot.setTitle("Selected indicators")
            self.bt_price_plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            self.bt_indicator_plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            self.lbl_bt_status.setText(f"Backtest: chart loaded ({len(df)} bars)")
            self.lbl_bt_status.setStyleSheet("font-weight:600; color: green;")
        except ValueError as e:
            self.lbl_bt_status.setText(f"Backtest: invalid chart range ({e})")
            self.lbl_bt_status.setStyleSheet("font-weight:600; color: red;")
        except Exception as e:
            self.lbl_bt_status.setText(f"Backtest: chart failed ({e})")
            self.lbl_bt_status.setStyleSheet("font-weight:600; color: red;")

    @QtCore.Slot()
    def run_backtest(self):
        try:
            if getattr(self, "bt_thread", None) is not None:
                self.log.write("[BACKTEST] Backtest already running")
                return

            symbols = self._backtest_target_symbols()
            if not symbols:
                self.lbl_bt_status.setText("Backtest: no symbol selected")
                self.lbl_bt_status.setStyleSheet("font-weight:600; color: red;")
                return

            tfs = [x.strip() for x in self.bt_tfs.text().split(",") if x.strip()]
            if not tfs:
                self.lbl_bt_status.setText("Backtest: no timeframes specified")
                self.lbl_bt_status.setStyleSheet("font-weight:600; color: red;")
                return

            primary_tf = str(self.bt_primary_tf.currentData())

            ml_model_path_for_backtest = str(ML_MODEL_PATH)
            if self.bt_use_candidate_model.isChecked():
                candidate_path = (self.bt_ml_model_path.text() or "").strip()
                if candidate_path:
                    ml_model_path_for_backtest = candidate_path

            if self.bt_use_candidate_model.isChecked():
                candidate_path = (self.bt_ml_model_path.text() or "").strip()
                if not candidate_path:
                    self.lbl_bt_status.setText("Backtest: candidate model override is enabled but no path is set")
                    self.lbl_bt_status.setStyleSheet("font-weight:600; color: red;")
                    return
                if not os.path.exists(candidate_path):
                    self.lbl_bt_status.setText("Backtest: candidate model path not found")
                    self.lbl_bt_status.setStyleSheet("font-weight:600; color: red;")
                    self.log.write(f"[BACKTEST] Candidate model path not found: {candidate_path}")
                    return
                
        except Exception as e:
            self.log.write(f"[BACKTEST] GUI handler failed: {e}")
            self.lbl_bt_status.setText(f"Backtest: GUI handler failed ({e})")
            self.lbl_bt_status.setStyleSheet("font-weight:600; color: red;")  
        
        base_cmd_common = [
            sys.executable,
            self._script_path("run_backtest.py"),
            "--primary-tf", primary_tf,
            "--ml-model-path", ml_model_path_for_backtest,
            "--tfs", *tfs,
            "--cash", str(BACKTEST_STARTING_CASH),
            "--warmup", str(BACKTEST_WARMUP_BARS),
            "--ensemble-min-conf", str(float(self.spin_min_conf.value())),
            "--min-vote-gap", str(float(self.spin_min_vote_gap.value())),
            "--use-rsi", str(self.chk_use_rsi.isChecked()),
            "--use-breakout", str(self.chk_use_breakout.isChecked()),
            "--use-ml", str(self.chk_use_ml.isChecked()),
            "--use-boom", str(self.chk_use_boom.isChecked()),
            "--use-boom-sell", str(self.chk_use_boom_sell.isChecked()),
            "--weight-rsi", str(float(self.w_rsi.value())),
            "--weight-breakout", str(float(self.w_breakout.value())),
            "--weight-ml", str(float(self.w_ml.value())),
            "--weight-boom", str(float(self.w_boom.value())),
            "--weight-boom-sell", str(float(self.w_boom_sell.value())),
            "--risk-max-pct", str(float(self.risk_max_risk_pct.value())),
            "--risk-min-conf", str(float(self.risk_min_conf.value())),
            "--risk-sl-atr", str(float(self.risk_sl_atr.value())),
            "--risk-tp-rr", str(float(self.risk_tp_rr.value())),
            "--risk-fallback-sl", str(float(self.risk_fallback_sl.value())),
            "--risk-max-spread", str(int(self.risk_max_spread.value())),
            "--risk-base-dev", str(int(self.risk_base_dev.value())),
            "--allow-new-trades", str(self.chk_allow.isChecked()),
            "--blocked-symbols", ",".join(self._blocked_symbols_set()),
            "--enable-session-filter", str(self.ex_enable_session.isChecked()),
            "--session-start-hour", str(int(self.ex_session_start.value())),
            "--session-end-hour", str(int(self.ex_session_end.value())),
            "--allow-weekends", str(self.ex_allow_weekends.isChecked()),
            "--enable-spread-filter", str(self.ex_enable_spread.isChecked()),
            "--exec-max-spread", str(int(self.ex_max_spread.value())),
            "--force-fixed-lot", str(self.ex_force_fixed_lot.isChecked()),
            "--fixed-sl-tp", str(self.ex_fixed_sl_tp.isChecked()),
            "--sl-tp-offset", str(float(self.ex_sl_tp_offset.value())),
            "--enable-trailing-stop", str(self.ex_enable_trailing.isChecked()),
            "--trailing-trigger-rr", str(float(self.ex_trailing_trigger_rr.value())),
            "--trailing-distance-rr", str(float(self.ex_trailing_distance_rr.value())),
            "--trailing-step-rr", str(float(self.ex_trailing_step_rr.value())),
            "--max-retries", str(int(self.ex_max_retries.value())),
            "--retry-delay-ms", str(int(self.ex_retry_delay.value())),
        ]

        if self.bt_start.text().strip():
            base_cmd_common += ["--start", self.bt_start.text().strip()]
        if self.bt_end.text().strip():
            base_cmd_common += ["--end", self.bt_end.text().strip()]

        self._bt_queue = []
        for symbol in symbols:
            out_dir = os.path.abspath(os.path.join(BACKTEST_OUT_DIR, self._safe_fs_name(symbol)))
            cmd = list(base_cmd_common) + [
                "--symbol", symbol,
                "--out", out_dir,
                "--tag", f"next_open_{self._safe_fs_name(symbol)}",
            ]
            self._bt_queue.append((symbol, cmd))

        self._bt_results = []
        self._set_backtest_buttons_enabled(False)
        self._run_next_backtest_job()

    @QtCore.Slot()
    def open_manual_ohlc_dialog(self):
        default_symbol = self.bt_symbol.currentText().strip() if hasattr(self, "bt_symbol") else ""
        default_tf = int(self.bt_primary_tf.currentData()) if hasattr(self, "bt_primary_tf") else int(PRIMARY_TIMEFRAME)

        dlg = ManualOHLCDialog(self, default_symbol=default_symbol, default_timeframe=default_tf)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        bar = dlg.payload()
        if bar["high"] < max(bar["open"], bar["close"], bar["low"]):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid OHLC",
                "High must be greater than or equal to open, close, and low.",
            )
            return
        if bar["low"] > min(bar["open"], bar["close"], bar["high"]):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid OHLC",
                "Low must be less than or equal to open, close, and high.",
            )
            return

        try:
            self.db.conn.execute(
                """
                INSERT INTO bars(symbol,timeframe,time,open,high,low,close,tick_volume,spread,real_volume)
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
                (
                    bar["symbol"],
                    int(bar["timeframe"]),
                    int(bar["time"]),
                    float(bar["open"]),
                    float(bar["high"]),
                    float(bar["low"]),
                    float(bar["close"]),
                    float(bar["tick_volume"]),
                    float(bar["spread"]),
                    float(bar["real_volume"]),
                ),
            )
            self.db.conn.commit()
            ts_utc = datetime.fromtimestamp(int(bar["time"]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            msg = f"Manual OHLC saved: {bar['symbol']} tf={bar['timeframe']} t={ts_utc}"
            self.log.write(f"[MANUAL_OHLC] {msg}")
            self.lbl_bt_status.setText(f"Backtest: {msg}")
            self.lbl_bt_status.setStyleSheet("font-weight:600; color: green;")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Failed", f"Could not save OHLC bar:\n{e}")

    @QtCore.Slot()
    def run_data_gap_audit(self):
        if getattr(self, "bt_thread", None) is not None:
            self.log.write("[AUDIT] Another backtest/audit job is already running")
            return

        symbols = self._backtest_target_symbols()
        tfs = [x.strip() for x in self.bt_tfs.text().split(",") if x.strip()]
        out_dir = os.path.abspath(DATA_QUALITY_OUT_DIR)

        cmd = [
            sys.executable,
            self._script_path("audit_data_gaps.py"),
            "--db", str(DB_PATH),
            "--out", out_dir,
        ]

        if symbols:
            cmd += ["--symbols", *symbols]

        if tfs:
            cmd += ["--timeframes", *tfs]

        if self.bt_start.text().strip():
            cmd += ["--start", self.bt_start.text().strip()]
        if self.bt_end.text().strip():
            cmd += ["--end", self.bt_end.text().strip()]

        self.lbl_bt_status.setText("Backtest: running data-gap audit…")
        self.lbl_bt_status.setStyleSheet("font-weight:600; color: gray;")
        self._set_backtest_buttons_enabled(False)

        self.bt_thread = QtCore.QThread(self)
        self.bt_worker = BacktestWorker(cmd, cwd=PROJECT_ROOT)
        self.bt_worker.moveToThread(self.bt_thread)

        self.bt_worker.line.connect(self.log.write)
        self.bt_thread.started.connect(self.bt_worker.run)

        def _done(ok: bool, msg: str):
            self._set_backtest_buttons_enabled(True)
            if ok:
                self.lbl_bt_status.setText(f"Backtest: data-gap audit completed ({out_dir})")
                self.lbl_bt_status.setStyleSheet("font-weight:600; color: green;")
            else:
                self.lbl_bt_status.setText(f"Backtest: data-gap audit failed ({msg})")
                self.lbl_bt_status.setStyleSheet("font-weight:600; color: red;")
            self.bt_thread.quit()
            self.bt_thread.wait(1500)
            self.bt_thread = None
            self.bt_worker = None

        self.bt_worker.finished.connect(_done)
        self.bt_thread.start()

    @QtCore.Slot(bool, str)
    def _on_data_gap_audit_finished(self, ok: bool, message: str):
        self.btn_run_backtest.setEnabled(True)
        self.btn_audit_data_gaps.setEnabled(True)

        if ok:
            out_dir = os.path.abspath(DATA_QUALITY_OUT_DIR)
            self.lbl_bt_status.setText(f"Backtest: data-gap audit completed ({out_dir})")
            self.lbl_bt_status.setStyleSheet("font-weight:600; color: #1f7a1f;")
            self.log.write(f"[AUDIT] Completed. Outputs saved to: {out_dir}")
        else:
            self.lbl_bt_status.setText(f"Backtest: data-gap audit failed ({message})")
            self.lbl_bt_status.setStyleSheet("font-weight:600; color: #b00020;")
            self.log.write(f"[AUDIT] Failed: {message}")

    @QtCore.Slot()
    def run_backfill_data_gaps(self):
        if getattr(self, "bt_thread", None) is not None:
            self.log.write("[BACKFILL] Another backtest/audit/backfill job is already running")
            return

        symbols = self._backtest_target_symbols() if hasattr(self, "_backtest_target_symbols") else [self.bt_symbol.currentText().strip()]
        tfs = [x.strip() for x in self.bt_tfs.text().split(",") if x.strip()]

        cmd = [
            sys.executable,
            self._script_path("backfill_data_gaps.py"),
            "--db", str(DB_PATH),
            "--audit-dir", os.path.abspath(DATA_QUALITY_OUT_DIR),
            "--rerun-audit",
        ]

        if symbols:
            cmd += ["--symbols", *symbols]
        if tfs:
            cmd += ["--timeframes", *tfs]

        self.lbl_bt_status.setText("Backtest: backfilling DB gaps…")
        self.lbl_bt_status.setStyleSheet("font-weight:600; color: gray;")
        self._set_backtest_buttons_enabled(False)

        self.bt_thread = QtCore.QThread(self)
        self.bt_worker = BacktestWorker(cmd, cwd=PROJECT_ROOT)
        self.bt_worker.moveToThread(self.bt_thread)

        self.bt_thread.started.connect(self.bt_worker.run)
        self.bt_worker.line.connect(self._append_log)

        def _done(ok: bool, msg: str):
            self._set_backtest_buttons_enabled(True)
            self.lbl_bt_status.setText(
                f"Backtest: {'backfill completed' if ok else 'backfill failed'} — {msg}"
            )
            self.lbl_bt_status.setStyleSheet(
                "font-weight:600; color: green;" if ok else "font-weight:600; color: red;"
            )

            try:
                self.bt_thread.quit()
                self.bt_thread.wait(1500)
            finally:
                self.bt_thread = None
                self.bt_worker = None

        self.bt_worker.finished.connect(_done)
        self.bt_thread.start()

    # ---------- Training helpers ----------
    def _run_training_steps(self, steps: list[list[str]]):
        if getattr(self, "_train_thread", None) is not None:
            self.log.write("[TRAIN] Training already running")
            return

        self.lbl_train_status.setText("Training: running candidate build…")
        self.btn_export_ds.setEnabled(False)
        self.btn_train_ml.setEnabled(False)
        self.btn_export_train.setEnabled(False)
        self.btn_reload_ml.setEnabled(False)

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        worker = TrainWorker(steps, cwd=project_root)
        th = QtCore.QThread(self)
        worker.moveToThread(th)

        self._train_worker = worker
        self._train_thread = th

        worker.line.connect(self.log.write)
        th.started.connect(worker.run)

        def _done(ok: bool, msg: str):
            self.lbl_train_status.setText(f"Training: {'OK' if ok else 'FAILED'} — {msg}")

            self.btn_export_ds.setEnabled(True)
            self.btn_train_ml.setEnabled(True)
            self.btn_export_train.setEnabled(True)
            self.btn_reload_ml.setEnabled(bool(ok))

            if ok:
                self.log.write("[TRAIN] All steps completed successfully")
                if hasattr(self, "refresh_experiments"):
                    self.refresh_experiments()
            else:
                self.log.write(f"[TRAIN] Training failed: {msg}")

            th.quit()

        worker.finished.connect(_done)
        worker.finished.connect(worker.deleteLater)
        th.finished.connect(th.deleteLater)

        def _thread_finished():
            self._train_worker = None
            self._train_thread = None

        th.finished.connect(_thread_finished)

        th.start()

    @QtCore.Slot()
    def export_dataset(self):
        symbol = self.train_symbol.currentText()
        tf = int(self.train_timeframe.currentData())
        out_csv = (self.train_csv.text() or "").strip() or "dataset.csv"

        self.lbl_train_status.setText(f"Training: exporting dataset… ({symbol}, tf={tf})")
        self.log.write(f"[ML] Export dataset: symbol={symbol} tf={tf} -> {out_csv}")

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "scripts.export_dataset",
            "--symbol", symbol,
            "--timeframe", str(tf),
            "--out", out_csv,
        ]
        self._run_training_steps([cmd])

    @QtCore.Slot()
    def train_model(self):
        symbol = self.train_symbol.currentText()
        tf = int(self.train_timeframe.currentData())
        out_csv = (self.train_csv.text() or "").strip() or "dataset.csv"
        model_version = (self.train_model_version.text() or "").strip() or f"ml_{datetime.utcnow().strftime('%Y-%m-%d')}"
        schema_version = int(self.train_schema_version.value())
        strict = bool(self.train_strict_schema.isChecked())

        candidate_model_path = self._candidate_model_path()

        self.lbl_train_status.setText("Training: training candidate model…")
        self.log.write(
            f"[ML] Train candidate: csv={out_csv} model_version={model_version} "
            f"schema_version={schema_version} strict={strict} -> {candidate_model_path}"
        )

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "scripts.train_model",
            "--csv", out_csv,
            "--model-path", str(candidate_model_path),
            "--model-version", model_version,
            "--schema-version", str(schema_version),
            "--symbol", symbol,
            "--timeframe", str(tf),
            "--horizon-bars", str(LABEL_HORIZON_BARS),
        ]
        if strict:
            cmd.append("--strict-schema")

        self._run_training_steps([cmd])

    @QtCore.Slot()
    def export_and_train(self):
        symbol = self.train_symbol.currentText()
        tf = int(self.train_timeframe.currentData())
        out_csv = (self.train_csv.text() or "").strip() or "dataset.csv"
        model_version = (self.train_model_version.text() or "").strip() or f"ml_{datetime.utcnow().strftime('%Y-%m-%d')}"
        schema_version = int(self.train_schema_version.value())
        strict = bool(self.train_strict_schema.isChecked())

        candidate_model_path = self._candidate_model_path()

        self.lbl_train_status.setText(f"Training: export + train candidate… ({symbol}, tf={tf})")
        self.log.write(
            f"[ML] Export+Train candidate: symbol={symbol} tf={tf} csv={out_csv} "
            f"model_version={model_version} schema_version={schema_version} strict={strict} "
            f"-> {candidate_model_path}"
        )

        export_cmd = [
            sys.executable,
            "-u",
            "-m",
            "scripts.export_dataset",
            "--symbol", symbol,
            "--timeframe", str(tf),
            "--out", out_csv,
        ]

        train_cmd = [
            sys.executable,
            "-u",
            "-m",
            "scripts.train_model",
            "--csv", out_csv,
            "--model-path", str(candidate_model_path),
            "--model-version", model_version,
            "--schema-version", str(schema_version),
            "--symbol", symbol,
            "--timeframe", str(tf),
            "--horizon-bars", str(LABEL_HORIZON_BARS),
        ]
        if strict:
            train_cmd.append("--strict-schema")

        self._run_training_steps([export_cmd, train_cmd])

    @QtCore.Slot()
    def reload_ml_model(self):
        # Safety: never reload while bot is running
        if getattr(self.controller, "is_running", False):
            self.log.write("[ML] Stop the bot before reloading the model")
            return

        try:
            enabled = {
                "RSIEMAStrategy": self.chk_use_rsi.isChecked(),
                "BreakoutStrategy": self.chk_use_breakout.isChecked(),
                "MLStrategy": self.chk_use_ml.isChecked(),
                "BoomSpikeTrendStrategy": self.chk_use_boom.isChecked(),
            }

            # Preserve existing ensemble knobs if available
            prev_weights = getattr(getattr(self, "ensemble", None), "weights", None)
            prev_min_conf = getattr(getattr(self, "ensemble", None), "min_conf", None)
            prev_min_vote_gap = getattr(getattr(self, "ensemble", None), "min_vote_gap", None)

            strategies = self._build_strategies(enabled=enabled)

            self.ensemble = EnsembleEngine(
                strategies,
                weights=prev_weights if prev_weights is not None else None,
                min_conf=prev_min_conf if prev_min_conf is not None else ENSEMBLE_MIN_CONF,
                min_vote_gap=prev_min_vote_gap if prev_min_vote_gap is not None else ENSEMBLE_MIN_VOTE_GAP,
                regime_multipliers=REGIME_WEIGHT_MULTIPLIERS,
            )

            # Ensure orchestrator uses the new ensemble
            self.orch.ensemble = self.ensemble

            self.log.write(
                "[ML] Reloaded strategies/ensemble "
                "(ML models will now resolve dynamically per symbol/timeframe)"
            )
            self.lbl_train_status.setText(
                "Training: strategies reloaded, dynamic ML model resolution active"
            )

            # Optional: refresh experiments so you immediately see the latest run
            if hasattr(self, "refresh_experiments"):
                self.refresh_experiments()
            if hasattr(self, "render_debug"):
                self.render_debug(self.cmb_symbol.currentText())

        except Exception as e:
            self.log.write(f"[ML] Reload failed: {e}")
            self.lbl_train_status.setText(f"Training: reload failed ({e})")
            
    # ---------- Bot control ----------
    @QtCore.Slot()
    def start_bot(self):
        self.bot_start_time = datetime.now(timezone.utc)
        self.bot_stop_time = None
        try:
            self.trade_session_id = self.db.create_trade_session(self.bot_start_time)
            if hasattr(self.orch, "set_trade_session"):
                self.orch.set_trade_session(self.trade_session_id, self.bot_start_time)
            self.log.write(f"[JOURNAL] Opened trade session #{self.trade_session_id}")
        except Exception as e:
            self.trade_session_id = None
            self.log.write(f"[JOURNAL] Failed to open DB trade session: {e}")

        self.controller.start(sleep_s=LOOP_SLEEP_SECONDS)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.lbl_bot_start_time.setText(self.bot_start_time.strftime("Start Time: %Y-%m-%d %H:%M:%S"))
        self.lbl_bot_stop_time.setText("Stop Time: —")

        self.log.write(f"[BOT] Started at {self.bot_start_time}")

    @QtCore.Slot()
    def stop_bot(self):
        self.controller.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        self.bot_stop_time = datetime.now(timezone.utc)
        self.lbl_bot_stop_time.setText(self.bot_stop_time.strftime("Stop Time: %Y-%m-%d %H:%M:%S"))

        self.log.write(f"[BOT] Stopped at {self.bot_stop_time}")
        report = self._build_session_trade_report()
        if self.trade_session_id:
            try:
                self.db.save_session_report(self.trade_session_id, report)
                self.log.write(f"[JOURNAL] Saved session #{self.trade_session_id} to DB")
            except Exception as e:
                self.log.write(f"[JOURNAL] Failed to save session #{self.trade_session_id}: {e}")

        if hasattr(self.orch, "set_trade_session"):
            self.orch.set_trade_session(None, None)

        self.refresh_trade_sessions(select_session_id=self.trade_session_id)
        self._show_session_trade_report(report)


    @QtCore.Slot()
    def on_allow_live_toggled(self):
        enabled = self.chk_allow_live.isChecked()
        set_live_trading_allowed(enabled)
        self.log.write(f"[UI] Allow Live Trading = {enabled}")

    @QtCore.Slot()
    def reconnect_mt5(self):
        """Best-effort reconnect without touching MT5 from the GUI thread.

        MT5 calls must go through MT5Client (worker thread).
        """
        try:
            ok = self.mt5.initialize(login=self.mt5.login, server=self.mt5.server, password=self.mt5.password)
            if not ok:
                raise RuntimeError(f"MT5 initialize() failed: {self.mt5.last_error()}")
            self.log.write("[MT5] Reconnect requested")
            self.refresh_portfolio()
        except Exception as e:
            self.log.write(f"[MT5] Reconnect failed: {e}")

    @QtCore.Slot()
    def open_account_switch_dialog(self):
        dlg = AccountSwitchDialog(getattr(self.mt5, "profile", "DEMO"), self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        target_profile = dlg.selected_profile()
        current_profile = str(getattr(self.mt5, "profile", "DEMO")).upper()
        if target_profile == current_profile:
            self.log.write(f"[MT5] Already on {target_profile} profile.")
            return
        if target_profile == "LIVE":
            ans = QtWidgets.QMessageBox.warning(
                self,
                "Confirm LIVE trading",
                "Are you sure you want to switch to LIVE trading? Orders will use real money.",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                self.log.write("[MT5] LIVE profile switch cancelled.")
                return
        try:
            ok = self.mt5.switch_profile(target_profile)
            if not ok:
                raise RuntimeError(f"MT5 initialize() failed: {self.mt5.last_error()}")
            self.log.write(f"[MT5] Switched profile to {target_profile}.")
            self._update_positions_status_label()
            self.refresh_portfolio()
            self.refresh_positions()
        except Exception as e:
            self.log.write(f"[MT5] Switch profile failed: {e}")

    def _update_positions_status_label(self):
        profile = str(getattr(self.mt5, "profile", "DEMO")).upper()
        self.lbl_status.setText(f"MT5 connected ({profile})")

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
            self.executor.force_symbol_fixed_lot = bool(self.ex_force_fixed_lot.isChecked())
            # GUI-driven fixed SL/TP offsets for Boom/Crash
            if hasattr(self.executor, 'boom_crash_fixed_sl_tp'):
                self.executor.boom_crash_fixed_sl_tp = bool(self.ex_fixed_sl_tp.isChecked())
                self.executor.boom_crash_sl_tp_offset = float(self.ex_sl_tp_offset.value())
            self.executor.enable_spread_filter = bool(self.ex_enable_spread.isChecked())
            self.executor.max_spread_points = int(self.ex_max_spread.value())
            self.executor.enable_session_filter = bool(self.ex_enable_session.isChecked())
            self.executor.session_start_hour = int(self.ex_session_start.value())
            self.executor.session_end_hour = int(self.ex_session_end.value())
            self.executor.allow_weekends = bool(self.ex_allow_weekends.isChecked())
            self.executor.max_retries = int(self.ex_max_retries.value())
            self.executor.retry_delay_ms = int(self.ex_retry_delay.value())
            
            blocked_symbols = set()
            for item in self.ex_block_symbols.selectedItems():
                blocked_symbols.add(item.text().strip())

            self.executor.blocked_symbols = blocked_symbols
            self.executor.enable_trailing_stop = bool(self.ex_enable_trailing.isChecked())
            self.executor.trailing_trigger_rr = float(self.ex_trailing_trigger_rr.value())
            self.executor.trailing_distance_rr = float(self.ex_trailing_distance_rr.value())
            self.executor.trailing_step_rr = float(self.ex_trailing_step_rr.value())

            if hasattr(self, "orch") and self.orch is not None:
                self.orch.update_entry_policy(
                    enforce_single_position_per_symbol=bool(self.ex_single_pos_per_symbol.isChecked()),
                    max_positions_per_symbol=int(self.ex_max_pos_per_symbol.value()),
                    max_total_open_positions=int(self.ex_max_total_positions.value()),
                    one_entry_per_closed_bar=bool(self.ex_one_entry_per_bar.isChecked()),
                    enable_trade_cooldown=bool(self.ex_enable_cooldown.isChecked()),
                    trade_cooldown_minutes=int(self.ex_cooldown_minutes.value()),
                    enable_max_daily_trades=bool(self.ex_enable_daily_limits.isChecked()),
                    max_daily_trades_per_symbol=int(self.ex_daily_trades_per_symbol.value()),
                    max_daily_trades_total=int(self.ex_daily_trades_total.value()),
                    auto_close_profits=bool(
                        self.ex_auto_close_profits.isChecked()
                    ),
                    auto_close_profits_threshold=float(
                        self.ex_auto_close_profits_threshold.value()
                    ) if hasattr(self, "ex_auto_close_profits_threshold") else 0.0,
                )

            fixed = "ON" if self.executor.force_symbol_fixed_lot else "OFF"
            spread = "ON" if self.executor.enable_spread_filter else "OFF"
            session = "ON" if self.executor.enable_session_filter else "OFF"
            trailing = "ON" if getattr(self.executor, "enable_trailing_stop", False) else "OFF"
            blocked_txt = ", ".join(sorted(self.executor.blocked_symbols)) if getattr(self.executor, "blocked_symbols", None) else "—"
            single_pos = "ON" if self.ex_single_pos_per_symbol.isChecked() else "OFF"
            one_bar = "ON" if self.ex_one_entry_per_bar.isChecked() else "OFF"
            cooldown = "ON" if self.ex_enable_cooldown.isChecked() else "OFF"
            daily_limits = "ON" if self.ex_enable_daily_limits.isChecked() else "OFF"

            self.lbl_exec_guard_status.setText(
                "spread={spread} (max {max_spread}pt)\n"
                "session={session} ({start}-{end})\n"
                "weekends={weekends}\n"
                "retries={retries}@{delay}ms\n"
                "fixed_lot={fixed}\n"
                "fixed_sl_tp={fixed_sl_tp} (offset {offset:g})\n"
                "trailing={trailing} (trigger {trigger:.2f}R, distance {distance:.2f}R, step {step:.2f}R)\n"
                "blocked_symbols={blocked}\n"
                f"auto_close_profits={self.ex_auto_close_profits.isChecked()})"
                .format(
                    spread=spread,
                    max_spread=self.executor.max_spread_points,
                    session=session,
                    start=self.executor.session_start_hour,
                    end=self.executor.session_end_hour,
                    weekends="ON" if self.executor.allow_weekends else "OFF",
                    retries=self.executor.max_retries,
                    delay=self.executor.retry_delay_ms,
                    fixed=fixed,
                    fixed_sl_tp="ON" if getattr(self.executor, "boom_crash_fixed_sl_tp", False) else "OFF",
                    offset=getattr(self.executor, "boom_crash_sl_tp_offset", 0.0),
                    single_pos=single_pos,
                    max_symbol=int(self.ex_max_pos_per_symbol.value()),
                    max_total=int(self.ex_max_total_positions.value()),
                    one_bar=one_bar,
                    cooldown=cooldown,
                    cooldown_min=int(self.ex_cooldown_minutes.value()),
                    daily_limits=daily_limits,
                    daily_symbol=int(self.ex_daily_trades_per_symbol.value()),
                    daily_total=int(self.ex_daily_trades_total.value()),
                    trailing=trailing,
                    trigger=float(getattr(self.executor, "trailing_trigger_rr", 1.0)),
                    distance=float(getattr(self.executor, "trailing_distance_rr", 0.5)),
                    step=float(getattr(self.executor, "trailing_step_rr", 0.10)),
                    blocked=blocked_txt,
                )
            )
            self.log.write(
                f"[EXEC GUARD] Applied single_pos={single_pos} max/symbol={int(self.ex_max_pos_per_symbol.value())} "
                f"max_total={int(self.ex_max_total_positions.value())} one_per_bar={one_bar} "
                f"cooldown={cooldown}({int(self.ex_cooldown_minutes.value())}m) "
                f"daily_limits={daily_limits} per_symbol={int(self.ex_daily_trades_per_symbol.value())} "
                f"total={int(self.ex_daily_trades_total.value())}"
            )
        except Exception as e:
            self.log.write(f"[EXEC_GUARD] Apply failed: {e}")

    # ---------- UI refreshers ----------
    @QtCore.Slot()
    def refresh_clock(self):
        now = datetime.now()
        self.lbl_now.setText(now.strftime("Time: %Y-%m-%d %H:%M:%S"))

    @QtCore.Slot()
    def refresh_positions(self):
        self._update_positions_status_label()
        pos = list_positions(self.mt5)
        selected_tickets = set(self._selected_position_tickets())

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

            current_price = float(getattr(p, "price_current", 0.0) or 0.0)
            if current_price <= 0:
                tick = self.mt5.symbol_info_tick(getattr(p, "symbol", ""))
                if tick is not None:
                    current_price = float(getattr(tick, "bid", 0.0) if typ == "BUY" else getattr(tick, "ask", 0.0) or 0.0)

            vals = [
                str(p.ticket),
                p.symbol,
                typ,
                f"{p.volume:.2f}",
                f"{p.price_open:.5f}",
                f"{current_price:.5f}",
                f"{profit:.2f}",
            ]

            for c, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(v)

                if c == 2:
                    item.setForeground(QtCore.Qt.GlobalColor.darkGreen if typ == "BUY" else QtCore.Qt.GlobalColor.darkRed)

                if c in (3, 4, 5, 6):
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignRight)
                else:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)

                if c == 6:
                    if profit > 0:
                        item.setForeground(QtCore.Qt.GlobalColor.darkGreen)
                    elif profit < 0:
                        item.setForeground(QtCore.Qt.GlobalColor.darkRed)
                    else:
                        item.setForeground(QtCore.Qt.GlobalColor.darkGray)

                self.tbl.setItem(r, c, item)


            if int(getattr(p, "ticket", 0) or 0) in selected_tickets:
                self.tbl.selectRow(r)

        self.tbl.setUpdatesEnabled(True)

        color = "darkgreen" if total_profit > 0 else "darkred" if total_profit < 0 else "gray"
        self.lbl_totals.setText(
            f"Positions: {len(pos)} | Winners: {winners} | Losers: {losers} | "
            f"<span style='color:{color};'>Floating PnL: {total_profit:.2f}</span>"
        )

    def close_mode(self, mode: str):
        self.log.write(f"[UI] Closing positions: {mode}")
        summary = close_positions(self.mt5, mode=mode)
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
        acct_profile = getattr(self.mt5, "profile", "DEMO")
        is_live = str(acct_profile).upper() == "LIVE"
        profile_tag = f"[{acct_profile}]"
        self.lbl_account.setText(f"Account {profile_tag}: {s.get('login')} @ {s.get('server')} ({cur})")
        self._set_badge(
            f"MT5: CONNECTED {profile_tag} {s.get('login')}@{s.get('server')}",
            ok=True,
            live=is_live,
        )

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

    def render_debug(self, symbol: Optional[str] = None):
        symbol = symbol or self.cmb_symbol.currentText()
        payload = self.decisions.get(symbol)
        if not payload:
            self.lbl_final.setText("Final: —")
            self.tbl_debug.setRowCount(0)
            return
        self.lbl_debug_ml_model.setText(
            f"Resolved ML model: {self._resolve_debug_ml_model_path(symbol)}"
        )
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

        self.lbl_bot_start_time.setText(
            self.bot_start_time.strftime("Start Time: %Y-%m-%d %H:%M:%S")
            if self.bot_start_time else "Start Time: —"
        )

        self.lbl_bot_stop_time.setText(
            self.bot_stop_time.strftime("Stop Time: %Y-%m-%d %H:%M:%S")
            if self.bot_stop_time else "Stop Time: —"
        )

        self.lbl_session_duration.setText(
            f"Session Duration: {self._format_duration(self.bot_start_time, self.bot_stop_time)}"
        )

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

    def _build_session_trade_report(self) -> dict:
        if not self.bot_start_time:
            raise RuntimeError("No session start time recorded")

        end_time = self.bot_stop_time or datetime.now(timezone.utc)

        deals = self.mt5.history_deals_get(self.bot_start_time, end_time)
        if deals is None:
            err = self.mt5.last_error()
            raise RuntimeError(f"history_deals_get failed: {err}")

        # MT5 constants
        DEAL_ENTRY_IN = getattr(self.mt5, "DEAL_ENTRY_IN", 0)
        DEAL_ENTRY_OUT = getattr(self.mt5, "DEAL_ENTRY_OUT", 1)
        ORDER_TYPE_BUY = getattr(self.mt5, "ORDER_TYPE_BUY", 0)
        ORDER_TYPE_SELL = getattr(self.mt5, "ORDER_TYPE_SELL", 1)

        by_position: dict[int, dict] = {}

        for d in deals:
            position_id = int(getattr(d, "position_id", 0) or 0)
            if position_id <= 0:
                continue

            row = by_position.setdefault(position_id, {
                "position_id": position_id,
                "symbol": str(getattr(d, "symbol", "")),
                "side": "",
                "open_time": None,
                "close_time": None,
                "volume": 0.0,
                "open_price": None,
                "close_price": None,
                "profit": 0.0,
                "commission": 0.0,
                "swap": 0.0,
                "fee": 0.0,
                "net": 0.0,
                "closed": False,
            })

            entry = int(getattr(d, "entry", -1))
            dtype = int(getattr(d, "type", -1))
            t = datetime.fromtimestamp(float(getattr(d, "time", 0)), tz=timezone.utc)

            if entry == DEAL_ENTRY_IN:
                row["open_time"] = row["open_time"] or t
                row["volume"] = float(getattr(d, "volume", 0.0) or 0.0)
                row["open_price"] = float(getattr(d, "price", 0.0) or 0.0)
                if dtype == ORDER_TYPE_BUY:
                    row["side"] = "BUY"
                elif dtype == ORDER_TYPE_SELL:
                    row["side"] = "SELL"

            elif entry == DEAL_ENTRY_OUT:
                row["close_time"] = t
                row["close_price"] = float(getattr(d, "price", 0.0) or 0.0)
                row["profit"] += float(getattr(d, "profit", 0.0) or 0.0)
                row["commission"] += float(getattr(d, "commission", 0.0) or 0.0)
                row["swap"] += float(getattr(d, "swap", 0.0) or 0.0)
                row["fee"] += float(getattr(d, "fee", 0.0) or 0.0)
                row["closed"] = True

        trades = []
        total_net = 0.0
        buy_net = 0.0
        sell_net = 0.0

        for row in by_position.values():
            if not row["closed"]:
                continue

            if self.trade_session_id and row["position_id"] > 0:
                try:
                    open_ev = self.db.get_open_event_for_position(self.trade_session_id, row["position_id"])
                except Exception:
                    open_ev = None
                if open_ev:
                    row["initial_sl"] = open_ev.get("initial_sl")
                    row["initial_tp"] = open_ev.get("initial_tp")
                    row["last_sl"] = open_ev.get("last_sl")
                    row["last_tp"] = open_ev.get("last_tp")
                    row["strategy_name"] = open_ev.get("strategy_name") or ""
                    row["comment"] = open_ev.get("comment") or ""
                else:
                    row["initial_sl"] = None
                    row["initial_tp"] = None
                    row["last_sl"] = None
                    row["last_tp"] = None
                    row["strategy_name"] = ""
                    row["comment"] = ""
            else:
                row["initial_sl"] = None
                row["initial_tp"] = None
                row["last_sl"] = None
                row["last_tp"] = None
                row["strategy_name"] = ""
                row["comment"] = ""

            row["net"] = row["profit"] + row["commission"] + row["swap"] + row["fee"]
            total_net += row["net"]

            if row["side"] == "BUY":
                buy_net += row["net"]
            elif row["side"] == "SELL":
                sell_net += row["net"]

            trades.append(row)

        trades.sort(key=lambda x: x["close_time"] or x["open_time"] or datetime.min.replace(tzinfo=timezone.utc))

        return {
            "start": self.bot_start_time,
            "stop": end_time,
            "count": len(trades),
            "total_net": total_net,
            "buy_net": buy_net,
            "sell_net": sell_net,
            "trades": trades,
        }

    def _show_session_trade_report(self, report: dict):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Session Trade Report")
        dlg.resize(1000, 560)

        layout = QtWidgets.QVBoxLayout(dlg)

        start = report["start"].astimezone().strftime("%Y-%m-%d %H:%M:%S")
        stop = report["stop"].astimezone().strftime("%Y-%m-%d %H:%M:%S")

        lbl = QtWidgets.QLabel(
            f"Start: {start}\n"
            f"Stop:  {stop}\n"
            f"Closed Trades: {report['count']}\n"
            f"Total PnL: {report['total_net']:.2f}\n"
            f"Buy PnL:   {report['buy_net']:.2f}\n"
            f"Sell PnL:  {report['sell_net']:.2f}"
        )
        lbl.setStyleSheet("font-weight:600;")
        layout.addWidget(lbl)

        table = QtWidgets.QTableWidget(0, 13)
        table.setHorizontalHeaderLabels([
            "Position ID", "Symbol", "Side", "Volume",
            "Open Time", "Close Time", "Open Price", "Close Price",
            "Initial SL", "Initial TP", "Last SL", "Last TP", "Net PnL"
        ])
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)

        for tr in report["trades"]:
            r = table.rowCount()
            table.insertRow(r)

            vals = [
                str(tr["position_id"]),
                str(tr["symbol"]),
                str(tr["side"]),
                f"{tr['volume']:.2f}",
                tr["open_time"].astimezone().strftime("%Y-%m-%d %H:%M:%S") if tr["open_time"] else "—",
                tr["close_time"].astimezone().strftime("%Y-%m-%d %H:%M:%S") if tr["close_time"] else "—",
                f"{tr['open_price']:.5f}" if tr["open_price"] is not None else "—",
                f"{tr['close_price']:.5f}" if tr["close_price"] is not None else "—",
                f"{tr['initial_sl']:.5f}" if tr.get("initial_sl") is not None else "—",
                f"{tr['initial_tp']:.5f}" if tr.get("initial_tp") is not None else "—",
                f"{tr['last_sl']:.5f}" if tr.get("last_sl") is not None else "—",
                f"{tr['last_tp']:.5f}" if tr.get("last_tp") is not None else "—",
                f"{tr['net']:.2f}",
            ]

            for c, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(v)
                table.setItem(r, c, item)

        layout.addWidget(table)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        btns.button(QtWidgets.QDialogButtonBox.StandardButton.Close).clicked.connect(dlg.close)
        layout.addWidget(btns)

        dlg.exec()

    @QtCore.Slot()
    def refresh_trade_sessions(self, select_session_id: int | None = None):
        try:
            sessions = self.db.list_trade_sessions(limit=200)
        except Exception as e:
            self.lbl_journal_status.setText(f"Trade Journal: failed to load sessions ({e})")
            return

        self.tbl_sessions.setRowCount(0)
        selected_row = -1
        for row in sessions:
            r = self.tbl_sessions.rowCount()
            self.tbl_sessions.insertRow(r)
            vals = [
                str(row.get("id") or ""),
                str(row.get("started_at") or "—"),
                str(row.get("stopped_at") or "—"),
                str(row.get("duration_seconds") or 0),
                str(row.get("trade_count") or 0),
                str(row.get("win_count") or 0),
                str(row.get("loss_count") or 0),
                f"{float(row.get('total_net') or 0.0):.2f}",
            ]
            for c, v in enumerate(vals):
                self.tbl_sessions.setItem(r, c, QtWidgets.QTableWidgetItem(v))
            if select_session_id is not None and int(row.get("id") or 0) == int(select_session_id):
                selected_row = r

        self.lbl_journal_status.setText(f"Trade Journal: {len(sessions)} session(s)")
        if selected_row >= 0:
            self.tbl_sessions.selectRow(selected_row)
            self.refresh_trade_journal_for_session(int(select_session_id))
        elif sessions:
            self.tbl_sessions.selectRow(0)
            self.refresh_trade_journal_for_session(int(sessions[0].get("id") or 0))
        else:
            self.tbl_journal.setRowCount(0)

    @QtCore.Slot()
    def on_trade_session_selected(self):
        row = self.tbl_sessions.currentRow()
        if row < 0:
            return
        item = self.tbl_sessions.item(row, 0)
        if item is None:
            return
        try:
            session_id = int(item.text())
        except Exception:
            return
        self.refresh_trade_journal_for_session(session_id)

    def refresh_trade_journal_for_session(self, session_id: int):
        try:
            trades = self.db.list_journal_trades(session_id)
        except Exception as e:
            self.lbl_journal_status.setText(f"Trade Journal: failed to load trades ({e})")
            return

        self.tbl_journal.setRowCount(0)
        for tr in trades:
            r = self.tbl_journal.rowCount()
            self.tbl_journal.insertRow(r)
            vals = [
                str(tr.get("position_id") or ""),
                str(tr.get("symbol") or ""),
                str(tr.get("side") or ""),
                f"{float(tr.get('volume') or 0.0):.2f}",
                str(tr.get("entry_time") or "—"),
                str(tr.get("exit_time") or "—"),
                f"{float(tr.get('entry_price')):.5f}" if tr.get("entry_price") is not None else "—",
                f"{float(tr.get('exit_price')):.5f}" if tr.get("exit_price") is not None else "—",
                f"{float(tr.get('initial_sl')):.5f}" if tr.get("initial_sl") is not None else "—",
                f"{float(tr.get('initial_tp')):.5f}" if tr.get("initial_tp") is not None else "—",
                f"{float(tr.get('last_sl')):.5f}" if tr.get("last_sl") is not None else "—",
                f"{float(tr.get('last_tp')):.5f}" if tr.get("last_tp") is not None else "—",
                f"{float(tr.get('first_trailing_sl')):.5f}" if tr.get("first_trailing_sl") is not None else "—",
                f"{float(tr.get('last_trailing_sl')):.5f}" if tr.get("last_trailing_sl") is not None else "—",
                f"{float(tr.get('net_profit') or 0.0):.2f}",
            ]
            for c, v in enumerate(vals):
                self.tbl_journal.setItem(r, c, QtWidgets.QTableWidgetItem(v))

    def _format_duration(self, start: datetime | None, stop: datetime | None) -> str:
        if not start:
            return "—"

        end = stop or datetime.now(start.tzinfo) if start.tzinfo else datetime.now()
        secs = max(0, int((end - start).total_seconds()))

        hours = secs // 3600
        minutes = (secs % 3600) // 60
        seconds = secs % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    QtCore.qInstallMessageHandler(_qt_message_filter)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
