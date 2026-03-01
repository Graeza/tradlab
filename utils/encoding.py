from __future__ import annotations

def encode_signal(signal: str) -> int:
    """Map BUY/SELL/HOLD -> +1/-1/0."""
    s = (signal or "HOLD").upper()
    if s == "BUY":
        return 1
    if s == "SELL":
        return -1
    return 0
