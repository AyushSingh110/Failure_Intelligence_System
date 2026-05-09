# FIE Architecture Diagram

Paste the Mermaid code below at **https://mermaid.live** → click **Actions → PNG** → save as `fig_architecture.png`

```mermaid
flowchart TD
    A["User / Application\n─────────────────\nSends a prompt"]

    A -->|Incoming Prompt| P1

    subgraph PIPE ["  Pre-LLM Guardrail Pipeline  "]
        direction TB
        P1["1  ·  Prompt Injection Detection"]
        P2["2  ·  Jailbreak Detection"]
        P3["3  ·  Many-Shot Jailbreak Detection"]
        P4["4  ·  Model Extraction Detection"]
        P5["5  ·  Prompt Leakage Detection"]
        P6["6  ·  Sensitive Data Detection"]
        P7["7  ·  Toxicity Detection"]
        P8["8  ·  Bias Detection"]
        P9["9  ·  Off-Topic Detection"]
        P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8 --> P9
    end

    P9 --> GATE{"Risk Score\n> Threshold?"}

    GATE -->|"YES  —  Attack Detected"| BLOCK["BLOCK REQUEST\nLog event · Alert dashboard"]
    GATE -->|"NO  —  Safe"| LLM["LLM Engine\nGroq  /  Ollama"]

    LLM --> HAL["10  ·  Hallucination Monitor\nPost-response verification"]
    HAL --> OUT["Final Response\nConfidence score · Attack type · Latency"]
    OUT -->|Response delivered| A

    %% Colours
    style A    fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a,font-weight:bold
    style PIPE fill:#fafafa,stroke:#6b7280
    style P1   fill:#fef3c7,stroke:#d97706,color:#1c1917
    style P2   fill:#fef3c7,stroke:#d97706,color:#1c1917
    style P3   fill:#fef3c7,stroke:#d97706,color:#1c1917
    style P4   fill:#fef3c7,stroke:#d97706,color:#1c1917
    style P5   fill:#fef3c7,stroke:#d97706,color:#1c1917
    style P6   fill:#fef3c7,stroke:#d97706,color:#1c1917
    style P7   fill:#fef3c7,stroke:#d97706,color:#1c1917
    style P8   fill:#fef3c7,stroke:#d97706,color:#1c1917
    style P9   fill:#fef3c7,stroke:#d97706,color:#1c1917
    style GATE fill:#f3e8ff,stroke:#9333ea,color:#3b0764,font-weight:bold
    style BLOCK fill:#fee2e2,stroke:#ef4444,color:#7f1d1d,font-weight:bold
    style LLM  fill:#dcfce7,stroke:#16a34a,color:#14532d,font-weight:bold
    style HAL  fill:#fff7ed,stroke:#ea580c,color:#431407
    style OUT  fill:#dcfce7,stroke:#16a34a,color:#14532d,font-weight:bold
```

## What each colour means

| Colour | Meaning |
|--------|---------|
| Blue   | User / Application |
| Yellow | Detection layers (pre-LLM) |
| Purple | Decision gate |
| Red    | Attack blocked |
| Green  | Safe path — LLM + final output |
| Orange | Hallucination monitor |
