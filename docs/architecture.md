# System Architecture Diagrams

These diagrams provide a visual map of how the trading system is structured and what happens during each execution loop.

## 1) High-level architecture

```mermaid
flowchart LR
    subgraph Runtime[Live runtime]
        A[core.main]
        B[Orchestrator]
        C[EnsembleEngine]
        D[RiskManager]
        E[TradeExecutor]
        G[PerformanceTracker]
    end

    subgraph Data[Data + feature pipeline]
        M[MT5Client / MT5Worker]
        F[DataFetcher]
        P[DataPipeline]
        DB[(SQLite MarketDatabase)]
        L[Labeling]
    end

    subgraph Strategies[Strategy plugins]
        S1[RSI/EMA]
        S2[Breakout]
        S3[MLStrategy]
        R[MLModelRegistry]
        CAND[(models/candidates/...)]
    end

    subgraph UX[Control + observability]
        GUI[gui.app]
        BC[BotController]
    end

    GUI --> BC --> B
    A --> M
    A --> F
    A --> P
    A --> C
    A --> D
    A --> E

    M --> F --> P --> DB
    P --> L --> DB

    S1 --> C
    S2 --> C
    S3 --> C
    R --> S3
    CAND --> R

    B --> P
    B --> C
    B --> D
    B --> E
    B --> G
    B --> DB
    E --> M
```

## 2) Decision cycle (what happens in each loop)

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant P as DataPipeline
    participant DB as MarketDatabase
    participant EN as EnsembleEngine
    participant RM as RiskManager
    participant EX as TradeExecutor

    loop Every sleep interval
        O->>P: update/fetch bars by symbol+timeframe
        P->>DB: UPSERT bars/features
        O->>DB: load latest data window
        O->>EN: evaluate strategies + aggregate signal
        EN-->>O: final action + confidence + winning strategy

        alt signal is HOLD
            O-->>O: skip entry
        else actionable BUY/SELL
            O->>RM: validate risk, size, limits
            RM-->>O: allow/deny + params
            alt risk approved
                O->>EX: place trade
                EX-->>O: execution result
                O->>DB: log trade open / state
            else risk denied
                O-->>O: log blocked reason
            end
        end

        O->>EX: manage trailing stops / auto-close profits
        EX-->>O: stop/close events
        O->>DB: journal stop/close updates
    end
```

## 3) ML model selection path

```mermaid
flowchart TD
    A[MLStrategy evaluate] --> B{Symbol+TF model exists?}
    B -- Yes --> C[MLModelRegistry chooses best candidate]
    C --> D{Meets min quality floor?}
    D -- Yes --> E[Use selected candidate]
    D -- No --> F[Use fallback model path]
    B -- No --> G{ML_REQUIRE_SYMBOL_MODEL?}
    G -- True --> H[Return HOLD]
    G -- False --> F
    E --> I[Contribute vote to EnsembleEngine]
    F --> I
    H --> I
```

## 4) Data lifecycle

```mermaid
flowchart LR
    MT5[MT5 market bars] --> Fetch[DataFetcher]
    Fetch --> Bars[(bars table)]
    Bars --> Feat[Feature engineering]
    Feat --> Features[(features table)]
    Bars --> LabelGen[Delayed label generation]
    LabelGen --> Labels[(labels table)]

    Features --> Train[scripts/train_model.py]
    Labels --> Train
    Train --> Artifacts[(models/candidates/...joblib)]
    Artifacts --> Runtime[MLModelRegistry + MLStrategy runtime]
```

## Reading tip

If you are new to the codebase, read in this order:
1. `core/main.py` (wiring)
2. `core/orchestrator.py` (runtime loop)
3. `core/data_pipeline.py` + `core/database.py` (state + persistence)
4. `core/ensemble.py` and `strategies/*` (decision logic)
5. `risk_manager.py` + `trade_executor.py` (execution and safety)
