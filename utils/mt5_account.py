from __future__ import annotations

from core.mt5_worker import MT5Client


def get_account_summary(mt5c: MT5Client) -> dict:
    acc = mt5c.account_info()
    if acc is None:
        return {"ok": False, "error": "account_info() failed"}

    return {
        "ok": True,
        "login": getattr(acc, "login", None),
        "server": getattr(acc, "server", None),
        "currency": getattr(acc, "currency", ""),
        "balance": float(getattr(acc, "balance", 0.0) or 0.0),
        "equity": float(getattr(acc, "equity", 0.0) or 0.0),
        "profit": float(getattr(acc, "profit", 0.0) or 0.0),
        "margin": float(getattr(acc, "margin", 0.0) or 0.0),
        "margin_free": float(getattr(acc, "margin_free", 0.0) or 0.0),
        "margin_level": float(getattr(acc, "margin_level", 0.0) or 0.0),
        "leverage": int(getattr(acc, "leverage", 0) or 0),
    }
