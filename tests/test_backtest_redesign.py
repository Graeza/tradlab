from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.contracts import BacktestRunRequest, ExecutionConfig, OutputConfig, SymbolMetadata
from backtest.preparation import causal_alignment, normalize_bars, prepare_backtest


def test_request_normalizes_duplicates_and_is_manifest_serializable():
    request = BacktestRunRequest(
        symbols=("Boom 1000", "Boom 1000"),
        timeframes=(5, 60, 5),
        primary_timeframe=5,
        start_time_s=100,
        end_time_s=200,
        warmup_bars=10,
        starting_cash=10_000,
        symbol_metadata={"Boom 1000": SymbolMetadata(point_size=0.01)},
        strategies=(),
        output=OutputConfig(Path("results")),
    )

    normalized = request.normalized()

    assert normalized.symbols == ("Boom 1000",)
    assert normalized.timeframes == (5, 60)
    assert normalized.to_manifest_dict()["output"]["root_directory"] == "results"
    with pytest.raises(ValueError, match="duplicate symbols"):
        request.normalized(strict_duplicates=True)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"symbols": ()}, "symbols must not be empty"),
        ({"primary_timeframe": 15}, "primary_timeframe"),
        ({"execution": ExecutionConfig(max_workers=0)}, "max_workers"),
    ],
)
def test_request_rejects_invalid_configuration(changes, message):
    values = dict(
        symbols=("X",), timeframes=(5,), primary_timeframe=5,
        start_time_s=None, end_time_s=None, warmup_bars=1, starting_cash=100,
        symbol_metadata={"X": SymbolMetadata(0.01)}, strategies=(),
    )
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        BacktestRunRequest(**values).normalized()


def test_higher_timeframe_alignment_only_exposes_closed_candle():
    primary_decisions = np.arange(10 * 3600 + 5 * 60, 11 * 3600 + 5 * 60, 5 * 60)
    h1_opens = np.array([9 * 3600, 10 * 3600], dtype=np.int64)

    aligned = causal_alignment(primary_decisions, h1_opens, 60 * 60)

    assert aligned.tolist()[:-1] == [0] * 11
    assert aligned.tolist()[-1] == 1


def test_normalization_duplicate_policy_gaps_and_readonly_preparation():
    bars = pd.DataFrame({
        "time": [300, 0, 300, 900],
        "open": [2.0, 1.0, 2.5, 3.0],
        "high": [3.0, 2.0, 3.5, 4.0],
        "low": [1.0, 0.5, 2.0, 2.5],
        "close": [2.5, 1.5, 3.0, 3.5],
        "spread": [1.0, 1.0, 2.0, 1.0],
    })
    normalized, quality = normalize_bars(
        bars, timeframe_seconds=300, duplicate_policy="keep_latest"
    )
    assert normalized["time"].tolist() == [0, 300, 900]
    assert normalized.loc[1, "open"] == 2.5
    assert quality.duplicate_rows == 2
    assert quality.gap_count == 1

    primary = pd.DataFrame({
        "time": np.arange(20, dtype=np.int64) * 300,
        "open": np.linspace(100, 119, 20),
        "high": np.linspace(101, 120, 20),
        "low": np.linspace(99, 118, 20),
        "close": np.linspace(100.5, 119.5, 20),
    })
    context = prepare_backtest(
        symbol="X", bars_by_tf={5: primary}, timeframes=(5,), primary_tf=5,
        warmup_bars=2, minimum_lookback_bars=4,
    )
    assert context.effective_warmup_bars == 4
    assert not context.opens.flags.writeable
    assert context.alignment_by_tf[5][0] == 0
    with pytest.raises(ValueError):
        context.opens[0] = 0


def test_normalization_rejects_negative_spread():
    bars = pd.DataFrame({
        "time": [0, 60], "open": [1, 1], "high": [2, 2],
        "low": [0, 0], "close": [1, 1], "spread": [0, -1],
    })
    with pytest.raises(ValueError, match="spread"):
        normalize_bars(bars, timeframe_seconds=60)
