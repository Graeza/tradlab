from __future__ import annotations

import threading
import queue
from dataclasses import dataclass
from concurrent.futures import Future
from typing import Any, Callable, Optional

import MetaTrader5 as mt5

from config.settings import login as cfg_login, server as cfg_server, password as cfg_password


@dataclass
class _Call:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    fut: Future


class MT5Worker(threading.Thread):
    """Single-thread MT5 execution worker.

    All MT5 API calls that touch the terminal should be routed through this worker
    to avoid thread-safety issues.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._q: "queue.Queue[_Call | None]" = queue.Queue()
        self._stop = threading.Event()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        fut: Future = Future()
        self._q.put(_Call(fn=fn, args=args, kwargs=kwargs, fut=fut))
        return fut

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)

    def run(self) -> None:
        while not self._stop.is_set():
            item = self._q.get()
            if item is None:
                break
            try:
                res = item.fn(*item.args, **item.kwargs)
                item.fut.set_result(res)
            except Exception as e:
                item.fut.set_exception(e)


class MT5Client:
    """Synchronous MT5 client that serializes all terminal calls via MT5Worker."""

    def __init__(
        self,
        login: int | None = None,
        server: str | None = None,
        password: str | None = None,
        *,
        initialize_on_start: bool = True,
    ):
        self.login = int(login) if login is not None else int(cfg_login)
        self.server = str(server) if server is not None else str(cfg_server)
        self.password = str(password) if password is not None else str(cfg_password)

        self._worker = MT5Worker()
        self._started = False
        self._initialize_on_start = bool(initialize_on_start)

    # --- lifecycle ---
    def start(self) -> None:
        if self._started:
            return
        self._worker.start()
        self._started = True
        if self._initialize_on_start:
            ok = self.initialize(login=self.login, server=self.server, password=self.password)
            if not ok:
                raise RuntimeError(f"MT5 initialize() failed: {self.last_error()}")

    def shutdown(self) -> None:
        if not self._started:
            return
        try:
            # Best effort: try to shutdown terminal connection from the worker
            self._call(mt5.shutdown)
        finally:
            self._worker.stop()
            self._started = False

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self._started:
            raise RuntimeError("MT5Client is not started. Call mt5_client.start() first.")
        fut = self._worker.submit(fn, *args, **kwargs)
        return fut.result()

    # --- wrappers (terminal-touching) ---
    def initialize(self, **kwargs: Any) -> bool:
        return bool(self._call(mt5.initialize, **kwargs))

    def last_error(self) -> Any:
        # last_error() is cheap, but keep it serialized for consistency
        return self._call(mt5.last_error)

    def account_info(self):
        return self._call(mt5.account_info)

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        return bool(self._call(mt5.symbol_select, symbol, enable))

    def symbol_info(self, symbol: str):
        return self._call(mt5.symbol_info, symbol)

    def symbol_info_tick(self, symbol: str):
        return self._call(mt5.symbol_info_tick, symbol)

    def positions_get(self, **kwargs: Any):
        return self._call(mt5.positions_get, **kwargs)

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int):
        return self._call(mt5.copy_rates_from_pos, symbol, timeframe, start_pos, count)

    def order_send(self, request: dict):
        return self._call(mt5.order_send, request=request)

    # --- expose constants for convenience (do not touch terminal) ---
    @property
    def const(self):
        return mt5
