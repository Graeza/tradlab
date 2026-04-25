"""Backtesting package.

Design goals:
- Reuse the same Strategy/Ensemble code paths as live.
- Bar-close decision, **next-bar-open execution** (default).
- Keep it deterministic while modeling practical execution frictions
  (spread/slippage/session gates) when configured.
"""
