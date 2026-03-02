"""Backtesting package.

Design goals:
- Reuse the same Strategy/Ensemble code paths as live.
- Bar-close decision, **next-bar-open execution** (default).
- Keep it simple and deterministic; add realism (spread/slippage) later.
"""
