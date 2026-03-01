from __future__ import annotations

import os
import queue
import joblib
from PySide6 import QtCore, QtWidgets

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
    ENSEMBLE_MIN_CONF, STRATEGY_WEIGHTS, LABEL_HORIZON_BARS
)

from risk_manager import RiskManager
from trade_executor import TradeExecutor
from utils.mt5_positions import close_positions, list_positions

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

        self.tbl_debug = QtWidgets.QTableWidget(0, 4)
        self.tbl_debug.setHorizontalHeaderLabels(["Strategy", "Signal", "Confidence", "Meta"])
        self.tbl_debug.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        dbg_layout.addWidget(self.tbl_debug)

        tabs.addTab(dbg_tab, "Strategy Debug")

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
        self.pos_timer.start()

        # Init MT5 and bot
        initialize_mt5()
        self.lbl_status.setText("MT5 connected")

        db = MarketDatabase(DB_PATH)
        fetcher = DataFetcher()
        pipeline = DataPipeline(fetcher, db)

        strategies = [RSIEMAStrategy(), BreakoutStrategy()]
        if USE_ML_STRATEGY and os.path.exists(ML_MODEL_PATH):
            try:
                model = joblib.load(ML_MODEL_PATH)
                strategies.append(MLStrategy(model))
                self.log.write(f"[ML] Loaded model: {ML_MODEL_PATH}")
            except Exception as e:
                self.log.write(f"[ML] Failed to load model: {e}")
        else:
            self.log.write("[ML] No model found — running without ML")

        ensemble = EnsembleEngine(strategies, weights=STRATEGY_WEIGHTS, min_conf=ENSEMBLE_MIN_CONF)
        risk = RiskManager()
        executor = TradeExecutor()

        self.orch = Orchestrator(
            pipeline=pipeline,
            ensemble=ensemble,
            risk_manager=risk,
            executor=executor,
            db=db,
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

        # Wire buttons
        self.btn_start.clicked.connect(self.start_bot)
        self.btn_stop.clicked.connect(self.stop_bot)
        self.btn_close_pos.clicked.connect(lambda: self.close_mode("positive"))
        self.btn_close_neg.clicked.connect(lambda: self.close_mode("negative"))
        self.btn_close_all.clicked.connect(lambda: self.close_mode("all"))
        self.btn_refresh.clicked.connect(self.refresh_positions)

        self.refresh_positions()

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

        final = payload.get("final", {})
        outputs = payload.get("outputs", [])

        # Optional: show highest-confidence strategies first
        outputs = sorted(outputs, key=lambda o: float(o.get("confidence", 0.0)), reverse=True)

        final_sig = str(final.get("signal", "—")).upper()
        final_conf = float(final.get("confidence", 0.0) or 0.0)
        self.lbl_final.setText(f"Final: {final_sig} (conf={final_conf:.2f})")

        if final_sig == "BUY":
            self.lbl_final.setStyleSheet("font-weight: 700; color: darkgreen;")
        elif final_sig == "SELL":
            self.lbl_final.setStyleSheet("font-weight: 700; color: darkred;")
        else:
            self.lbl_final.setStyleSheet("font-weight: 700; color: dimgray;")

        self.tbl_debug.setRowCount(0)

        for o in outputs:
            r = self.tbl_debug.rowCount()
            self.tbl_debug.insertRow(r)

            name = str(o.get("name", ""))
            sig = str(o.get("signal", "")).upper()
            conf = f"{float(o.get('confidence', 0.0)):.2f}"
            meta = str(o.get("meta", {}))

            # ✅ These must be INSIDE the loop
            self.tbl_debug.setItem(r, 0, self._styled_item(name))
            self.tbl_debug.setItem(r, 1, self._styled_item(sig, signal=sig))
            self.tbl_debug.setItem(r, 2, self._styled_item(conf, signal=sig))
            self.tbl_debug.setItem(r, 3, self._styled_item(meta))

def run():
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()

if __name__ == "__main__":
    run()
