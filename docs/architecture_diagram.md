# FIE Architecture Diagram

This is the canonical architecture diagram (v1.13.0). It renders automatically on
GitHub. To export a PNG for a paper or slide deck, paste the Mermaid block below
into [mermaid.live](https://mermaid.live) → **Actions → PNG**.

```mermaid
flowchart TD
    U["User / App<br/>wraps any LLM call with @monitor()"]
    U -->|prompt| FP

    subgraph GUARD["PRE-FLIGHT GUARD — runs before your model, under 20ms"]
        direction TB
        FP["Fast-path check<br/>known-attack and whitelist hashes (instant)"]
        L["12 detection layers, all at once<br/>regex · prompt-guard · GCG suffix · many-shot ·<br/>indirect injection · multilingual · romanisation · …"]
        AGG["Weighted vote + domain and session boosts"]
        FP --> L --> AGG
    end

    AGG --> ROUTE{"How risky<br/>is it?"}
    ROUTE -->|"high — clear attack"| BLOCK["BLOCKED<br/>your model never runs · event logged"]
    ROUTE -->|"borderline"| LG["LlamaGuard tie-breaker<br/>fail-secure: blocks if unavailable"]
    ROUTE -->|"low — safe"| LLM
    LG -->|unsafe| BLOCK
    LG -->|safe| LLM

    LLM["Your LLM runs<br/>answer returned to the user immediately"]
    LLM --> MON

    subgraph MON["OUTPUT MONITORING — in the background, zero added latency"]
        direction TB
        SHADOW["Shadow ensemble: 3 models answer in parallel<br/>disagreement = strongest hallucination signal"]
        FSV["Failure Signal Vector → XGBoost classifier (under 10ms)"]
        JURY["DiagnosticJury: 3 specialist agents review the answer"]
        SHADOW --> FSV --> JURY
    end

    MON --> OUT{"Verdict?"}
    OUT -->|"answer is correct"| VAL["VALIDATED<br/>delivered as-is"]
    OUT -->|"hallucination"| COR["CORRECTED<br/>shadow consensus replaces the answer"]
    OUT -->|"attack confirmed"| BLK2["BLOCKED"]

    BLOCK -.confirmed labels.-> FB["Feedback store<br/>next identical prompt → instant decision"]
    BLK2 -.-> FB
    FB -.learned hash.-> FP

    classDef user   fill:#e0e7ff,stroke:#4f46e5,color:#1e1b4b;
    classDef guard  fill:#fef3c7,stroke:#d97706,color:#78350f;
    classDef gate   fill:#ede9fe,stroke:#7c3aed,color:#4c1d95;
    classDef block  fill:#fee2e2,stroke:#dc2626,color:#7f1d1d;
    classDef safe   fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef mon    fill:#cffafe,stroke:#0891b2,color:#164e63;
    classDef warn   fill:#ffedd5,stroke:#ea580c,color:#7c2d12;
    classDef store  fill:#f1f5f9,stroke:#64748b,color:#1e293b;

    class U user;
    class FP,L,AGG guard;
    class ROUTE,OUT gate;
    class BLOCK,BLK2 block;
    class LLM,LG,VAL safe;
    class SHADOW,FSV,JURY mon;
    class COR warn;
    class FB store;
```

## What each colour means

| Colour | Meaning |
|--------|---------|
| Blue   | The incoming user request |
| Amber  | The pre-flight guard (12 detection layers, before the model) |
| Purple | A decision / routing point |
| Red    | Request blocked — the model never runs, or the answer is suppressed |
| Green  | The safe path: your LLM and a validated answer |
| Cyan   | Background hallucination monitoring (shadow ensemble, XGBoost, jury) |
| Orange | A corrected answer (shadow consensus) |
| Grey   | The learning loop — confirmed labels become instant future decisions |

For the full, animated, every-component view, open
[`fie_architecture.html`](fie_architecture.html) in a browser, or read
[`ARCHITECTURE.md`](ARCHITECTURE.md).
