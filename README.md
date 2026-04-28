# Trading Bot Refactor (Expandable / ML-ready / Multi-strategy)

## What you get
- Incremental MT5 data collection
- SQLite storage with **UPSERT** (no duplicates)
- 3-layer data model: `bars` / `features` / `labels`
- Multi-strategy execution via `EnsembleEngine`
- Your previous `indicators.py` placed in `utils/indicators.py`

## Folder layout
```
trading_bot_refactor/
  core/         # orchestrator + pipeline + db + ensemble
  strategies/   # plug-in strategies (RSI/EMA, Breakout, ML)
  utils/        # indicators + helpers
  config/       # settings
  models/       # your .joblib models go here
```

## Run
1) Install requirements (ensure MT5 terminal is installed & logged in):
```bash
pip install -r requirements.txt
```

2) Configure symbols/timeframes in `config/settings.py`

3) Start:
```bash
python -m core.main
```

## ML: per-symbol / per-timeframe models
- Candidate models are discovered from `models/candidates/<symbol_safe>/tf_<timeframe>/*.joblib`.
- If no candidate exists, runtime fallback behavior is controlled by:
  - `ML_REQUIRE_SYMBOL_MODEL` (if `True`, bot will HOLD when a symbol-specific model is missing),
  - `ML_MODEL_PATH` (promoted fallback model),
  - `ML_MIN_CANDIDATE_ACCURACY` (minimum validation score for candidate selection; now defaults to `0.55`).
- Training now defaults to a walk-forward validation policy (expanding windows) rather than a single holdout.
- Registry candidate scoring prefers walk-forward metrics when available and enforces the quality floor on both mean and worst fold accuracy.

Train all symbols in one pass:

```bash
python -m scripts.train_all_symbols --strict-schema

# optional explicit policy tuning
python -m scripts.train_all_symbols --validation-policy walk_forward --wf-folds 4 --wf-min-train-frac 0.5
```


## Architecture visuals
For a visual walkthrough of system structure and runtime behavior, see:

- `docs/architecture.md`

The document includes:
- high-level component architecture
- decision-loop sequence diagram
- ML model selection flow
- end-to-end data lifecycle

## Adding a new strategy
1) Create `strategies/my_strategy.py` implementing `Strategy.evaluate(data_by_tf)`
2) Add it in `core/main.py -> build_strategies()`
3) Optionally add a weight in `config/settings.py -> STRATEGY_WEIGHTS`

## Notes
- Labels are generated in a delayed manner (needs future bars).
- `utils/indicators.py` contains your RSI/EMA/MACD + positive candle regression utilities.


## GUI (industry-style dashboard)
Run:

```bash
python -m gui.app
```

Includes:
- Start / Stop bot buttons
- Live log console
- Open positions table
- Close positions: POSITIVE / NEGATIVE / ALL
