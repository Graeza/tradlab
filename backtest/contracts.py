"""Public, dependency-free contracts for the redesigned backtest engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Mapping


JSONScalar = str | int | float | bool | None
EngineMode = Literal["legacy", "optimized", "compare"]
DetailLevel = Literal["summary", "signals", "full"]
DuplicatePolicy = Literal["error", "keep_first", "keep_latest"]


class RunStatus(str, Enum):
    QUEUED = "queued"
    LOADING = "loading"
    PREPARING = "preparing"
    SIMULATING = "simulating"
    ANALYZING = "analyzing"
    PERSISTING = "persisting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class RunPhase(str, Enum):
    LOADING = "loading"
    PREPARING = "preparing"
    SIMULATING = "simulating"
    ANALYZING = "analyzing"
    PERSISTING = "persisting"


@dataclass(frozen=True, slots=True)
class SymbolMetadata:
    point_size: float
    minimum_lot: float = 0.0


@dataclass(frozen=True, slots=True)
class StrategySelection:
    name: str
    enabled: bool = True
    weight: float = 1.0
    parameters: Mapping[str, JSONScalar] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    signal_horizons: tuple[int, ...] = (1, 3, 6, 12)
    signal_targets: tuple[int, ...] = (50, 100, 250, 500)


@dataclass(frozen=True, slots=True)
class OutputConfig:
    root_directory: Path
    detail_level: DetailLevel = "signals"
    diagnostics_format: Literal["parquet", "csv"] = "parquet"
    equity_curve_stride: int = 1
    retain_failed_temporary_files: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    max_workers: int = 1
    fail_fast: bool = False
    progress_interval_bars: int = 1_000
    prediction_batch_size: int = 8_192
    duplicate_policy: DuplicatePolicy = "error"


@dataclass(frozen=True, slots=True)
class BacktestRunRequest:
    symbols: tuple[str, ...]
    timeframes: tuple[int, ...]
    primary_timeframe: int
    start_time_s: int | None
    end_time_s: int | None
    warmup_bars: int
    starting_cash: float
    symbol_metadata: Mapping[str, SymbolMetadata]
    strategies: tuple[StrategySelection, ...]
    ensemble: Mapping[str, Any] = field(default_factory=dict)
    risk: Mapping[str, Any] = field(default_factory=dict)
    broker: Mapping[str, Any] = field(default_factory=dict)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(Path("backtest_outputs")))
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    engine_mode: EngineMode = "optimized"
    random_seed: int = 0
    tag: str = ""

    def normalized(self, *, strict_duplicates: bool = False) -> "BacktestRunRequest":
        """Validate and return a canonical request.

        Duplicate symbols/timeframes are removed in first-seen order unless strict
        validation is requested. Validation deliberately happens before any loader
        or strategy is invoked.
        """

        symbols = _deduplicate(self.symbols, "symbols", strict_duplicates)
        timeframes = _deduplicate(self.timeframes, "timeframes", strict_duplicates)
        if not symbols:
            raise ValueError("symbols must not be empty")
        if not timeframes:
            raise ValueError("timeframes must not be empty")
        if self.primary_timeframe not in timeframes:
            raise ValueError("primary_timeframe must occur in timeframes")
        if self.start_time_s is not None and self.end_time_s is not None and self.end_time_s < self.start_time_s:
            raise ValueError("end_time_s must be greater than or equal to start_time_s")
        _positive("warmup_bars", self.warmup_bars)
        _positive("starting_cash", self.starting_cash)
        _positive("max_workers", self.execution.max_workers)
        _positive("progress_interval_bars", self.execution.progress_interval_bars)
        _positive("prediction_batch_size", self.execution.prediction_batch_size)
        _positive("equity_curve_stride", self.output.equity_curve_stride)
        for name, values in (
            ("signal_horizons", self.analysis.signal_horizons),
            ("signal_targets", self.analysis.signal_targets),
        ):
            if not values:
                raise ValueError(f"{name} must not be empty")
            for value in values:
                _positive(name, value)
        if self.engine_mode not in ("legacy", "optimized", "compare"):
            raise ValueError(f"unsupported engine_mode: {self.engine_mode}")
        if self.output.detail_level not in ("summary", "signals", "full"):
            raise ValueError(f"unsupported detail_level: {self.output.detail_level}")
        if self.execution.duplicate_policy not in ("error", "keep_first", "keep_latest"):
            raise ValueError(f"unsupported duplicate_policy: {self.execution.duplicate_policy}")
        for symbol in symbols:
            metadata = self.symbol_metadata.get(symbol)
            if metadata is None:
                raise ValueError(f"missing symbol metadata for {symbol!r}")
            _positive(f"symbol_metadata[{symbol!r}].point_size", metadata.point_size)
            if metadata.minimum_lot < 0:
                raise ValueError(f"symbol_metadata[{symbol!r}].minimum_lot must be non-negative")
        for strategy in self.strategies:
            if not strategy.name.strip():
                raise ValueError("strategy name must not be empty")
            if strategy.weight < 0:
                raise ValueError(f"strategy {strategy.name!r} weight must be non-negative")
        return replace(self, symbols=symbols, timeframes=timeframes)

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = asdict(self.normalized())
        payload["output"]["root_directory"] = str(self.output.root_directory)
        return payload


@dataclass(frozen=True, slots=True)
class BacktestProgress:
    run_id: str
    symbol: str
    phase: RunPhase
    completed_units: int
    total_units: int | None
    elapsed_s: float
    estimated_remaining_s: float | None
    message: str = ""


CancellationCheck = Callable[[], bool]
ProgressCallback = Callable[[BacktestProgress], None]


def _deduplicate(values: tuple[Any, ...], name: str, strict: bool) -> tuple[Any, ...]:
    result = tuple(dict.fromkeys(values))
    if strict and len(result) != len(values):
        raise ValueError(f"duplicate {name} are not allowed in strict mode")
    return result


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
