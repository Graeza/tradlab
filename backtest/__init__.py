"""Backtesting package.

Design goals:
- Reuse the same Strategy/Ensemble code paths as live.
- Bar-close decision, **next-bar-open execution** (default).
- Keep it deterministic while modeling practical execution frictions
  (spread/slippage/session gates) when configured.
"""

from backtest.contracts import (
    AnalysisConfig,
    BacktestProgress,
    BacktestRunRequest,
    ExecutionConfig,
    OutputConfig,
    RunPhase,
    RunStatus,
    StrategySelection,
    SymbolMetadata,
)

__all__ = [
    "AnalysisConfig",
    "BacktestProgress",
    "BacktestRunRequest",
    "ExecutionConfig",
    "OutputConfig",
    "RunPhase",
    "RunStatus",
    "StrategySelection",
    "SymbolMetadata",
]
