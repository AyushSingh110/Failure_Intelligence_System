# Contributing to Failure Intelligence Engine

Thank you for your interest in contributing to FIE. This document explains how to get started, how to report issues, and how to submit changes.

FIE is a solo-maintained research project that doubles as a live system. Contributions that are well-scoped, reproducible, and don't require big architectural changes merge fastest.

---

## Ways to Contribute

| Type | How |
| --- | --- |
| Bug reports | Open an issue using the **Bug Report** template |
| False positive / false negative | Open an issue using the **Detection Report** template |
| Feature suggestions | Open an issue using the **Feature Request** template |
| Code — bug fix or detection improvement | Fork → branch → PR against `main` |
| New detection layer | Read [Adding a Detection Layer](#adding-a-detection-layer) first |
| Benchmark prompts | Read [Contributing Benchmark Prompts](#contributing-benchmark-prompts) first |
| Documentation | Fix errors or improve examples via PR — no issue needed |

---

## Getting Started

### 1. Fork and clone

```bash
git clone https://github.com/AyushSingh110/Failure_Intelligence_System.git
cd Failure_Intelligence_System
```

### 2. Set up the backend

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY, MONGODB_URI, etc.
uvicorn app.main:app --reload --port 8000
```

### 3. Set up the frontend

```bash
cd Frontend
npm install
cp .env.local.example .env.local   # fill in VITE_API_URL, VITE_GOOGLE_CLIENT_ID
npm run dev
```

### 4. Verify the encoder loads

```bash
python -c "from engine.encoder import get_encoder; e = get_encoder(); print('encoder available:', e.available)"
```

Should print `encoder available: True`. If not, run `pip install sentence-transformers`.

---

## Project Structure

```
app/
  routes/           FastAPI route modules (one domain per file)
  auth.py           User management + JWT sessions
  schemas.py        Pydantic request/response models

engine/
  agents/
    adversarial/
      specialist.py           DiagnosticJury adversarial specialist — main detection pipeline
      multilingual_romanisation.py  Layer 3d: cross-lingual romanisation detector (v1.13.0)
    hallucination/            Hallucination-focused jury agents
  detector/
    consistency.py            Semantic clustering + exact-match fallback
    ensemble.py               Pairwise disagreement scoring across shadow models
    embedding.py              Sentence-transformer + n-gram similarity
    entropy.py                Shannon entropy on token logprobs
  encoder.py                  Lazy-loaded SentenceEncoder singleton (all-MiniLM-L6-v2)
  groq_service.py             Shadow model fan-out via Groq API
  hard_positive_collector.py  Staging pipeline for human-review of blocked prompts

evaluation/
  datasets/
    unknown_*_bench_v3.py     Held-out benchmark sets (200 prompts each, 4 attack families)
    freeze_benchmarks.py      SHA-256 manifest generator — run after any dataset change
  hallucination/
    run_eval.py               TruthfulQA harness: H1 (XGBoost FSV) vs H2 (ensemble baseline)
  ablation/
    run_fsv_ablation.py       SHAP ablation for XGBoost Failure Signal Vector

scripts/
  retrain_pair_v4.py          Production PAIR v4 training script (Exp 8, 3× weighting)
  extend_benchmarks.py        Groq-powered benchmark extension tool

fie/                          Published as fie-sdk on PyPI — separate importable package
  adversarial.py              Public @fie.monitor decorator
  models/                     Shipped model artifacts (PAIR v4 classifier + metadata)

Frontend/src/
  pages/                      One file per dashboard page
  components/                 Shared UI components
  lib/                        Auth utilities
```

---

## Running the Evaluations

### Adversarial benchmark

```bash
python evaluation/run_benchmark.py
```

This runs PAIR v4 against all 12 benchmark sets. Results print to stdout and save to `evaluation/results/`.

### TruthfulQA hallucination eval

```bash
GROQ_API_KEY=your_key python evaluation/hallucination/run_eval.py
```

The run is resume-safe via `evaluation/hallucination/results/raw_responses.jsonl` — interrupted runs pick up where they left off. Expect ~4–6 hours for the full 817-question set on Groq free tier.

### FSV SHAP ablation

```bash
python evaluation/ablation/run_fsv_ablation.py
```

Outputs a feature importance curve to `evaluation/ablation/results/fsv_ablation/`.

---

## Adding a Detection Layer

FIE's adversarial detection pipeline is structured as composable layers in `engine/agents/adversarial/specialist.py`. The romanisation detector (`multilingual_romanisation.py`, v1.13.0) is the reference implementation.

**Minimum contract for a new layer:**

```python
def run_<name>_detection(prompt: str) -> tuple[str | None, float, dict]:
    """
    Returns:
        root_cause  — string constant if detected, None otherwise
        confidence  — float in [0.0, 1.0]
        evidence    — dict with keys explaining what was found
    """
```

**Wiring it into specialist.py:**

1. Import at the top of `specialist.py`.
2. Call it early in the `run()` method — before the PAIR classifier.
3. Add to `_high_conf_structural` if confidence ≥ 0.80 should short-circuit.
4. Add to the `elif` priority chain for the `root_cause` assignment.
5. Add to `active_confidences`.
6. Add an evidence block under `evidence["<name>"]`.
7. Add a mitigation string to the `_MITIGATIONS` dict.

**Smoke test your layer** against the relevant benchmark before opening a PR:

```bash
python -c "
from evaluation.datasets.unknown_multilingual_bench_v3 import _PROMPTS
from engine.agents.adversarial.<your_module> import run_<name>_detection
hits = sum(1 for p in _PROMPTS[:30] if run_<name>_detection(p['prompt'])[0] is not None)
print(f'Hit rate: {hits}/30')
"
```

---

## Contributing Benchmark Prompts

The `evaluation/datasets/unknown_*_bench_v3.py` files are the held-out test sets — they are frozen and should not be used for training. Contributions here must follow the format strictly.

Each entry:

```python
{
    "id": "v3_<family>_<three_digit_number>",
    "prompt": "...",
    "category": "<attack_family>",
    "expected": "BLOCK",
    "notes": "optional: what makes this structurally distinct"
},
```

After adding entries, regenerate the SHA-256 manifest:

```bash
python evaluation/datasets/freeze_benchmarks.py
```

Do not edit existing entries — only append. The manifest enforces this.

---

## Submitting a Pull Request

1. Branch from `main`:

   ```bash
   git checkout -b fix/short-descriptive-name
   ```

2. Keep the PR focused — one concern per PR.

3. Make sure the backend starts cleanly:

   ```bash
   uvicorn app.main:app --reload
   ```

4. Make sure the frontend builds:

   ```bash
   cd Frontend && npm run build
   ```

5. If your change touches a detection layer, include smoke test output in the PR description.

6. Open the PR against `main` with a title that describes *what changed*, not what you did.

---

## Code Style

- **Python** — PEP 8, type hints throughout, functions under ~50 lines. No class hierarchies unless the domain demands it.
- **JavaScript / React** — functional components only. Inline styles to match the existing design system.
- **Comments** — only when the *why* is non-obvious. Do not describe what the code does.
- **No docstrings longer than one line.** The function signature and name should carry the meaning.

---

## Questions

Open a [GitHub Discussion](https://github.com/AyushSingh110/Failure_Intelligence_System/discussions) or email [ayushsingh355vns@gmail.com](mailto:ayushsingh355vns@gmail.com). No question is too small.
