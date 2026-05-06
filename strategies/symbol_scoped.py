from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal, normalize_result


def _norm_symbol(value: Any) -> str:
    return str(value or "").strip().casefold()


class SymbolScopedStrategy(Strategy):
    """Run a strategy only for selected symbols.

    The wrapped strategy keeps its original name so ensemble weights,
    performance tracking, and GUI debug output continue to use the same
    strategy identifiers.
    """

    def __init__(
        self,
        inner: Strategy,
        *,
        allowed_symbols: Optional[Iterable[str]] = None,
        blocked_symbols: Optional[Iterable[str]] = None,
    ):
        self.inner = inner
        self.name = str(getattr(inner, "name", inner.__class__.__name__))
        self.allowed_symbols = {_norm_symbol(s) for s in (allowed_symbols or []) if str(s).strip()}
        self.blocked_symbols = {_norm_symbol(s) for s in (blocked_symbols or []) if str(s).strip()}

    def evaluate(
        self,
        data_by_tf: dict[int, pd.DataFrame],
        context: Optional[Mapping[str, Any]] = None,
    ) -> StrategyResult:
        context = context or {}
        symbol = _norm_symbol(context.get("symbol"))

        if self.allowed_symbols and symbol not in self.allowed_symbols:
            return StrategyResult(
                name=self.name,
                signal=Signal.HOLD,
                confidence=0.0,
                meta={
                    "reason": "symbol_not_enabled_for_strategy",
                    "symbol": context.get("symbol"),
                    "allowed_symbols": sorted(self.allowed_symbols),
                },
            )

        if self.blocked_symbols and symbol in self.blocked_symbols:
            return StrategyResult(
                name=self.name,
                signal=Signal.HOLD,
                confidence=0.0,
                meta={
                    "reason": "symbol_blocked_for_strategy",
                    "symbol": context.get("symbol"),
                    "blocked_symbols": sorted(self.blocked_symbols),
                },
            )

        try:
            raw = self.inner.evaluate(data_by_tf, context=context)  # type: ignore[call-arg]
        except TypeError:
            raw = self.inner.evaluate(data_by_tf)

        return normalize_result(raw, fallback_name=self.name)

    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]):
        # Not used because evaluate() handles context-aware scoping.
        return StrategyResult(self.name, Signal.HOLD, 0.0, {"reason": "use_evaluate"})
