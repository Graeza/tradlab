from __future__ import annotations
import MetaTrader5 as mt5
from config.settings import login as cfg_login, server as cfg_server, password as cfg_password

def initialize_mt5(
    login: int | None = None,
    server: str | None = None,
    password: str | None = None
) -> None:
    """Initialize MT5 terminal connection."""

    login = login if login is not None else cfg_login
    server = server if server is not None else cfg_server
    password = password if password is not None else cfg_password

    ok = mt5.initialize(login=login, server=server, password=password)

    if not ok:
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
