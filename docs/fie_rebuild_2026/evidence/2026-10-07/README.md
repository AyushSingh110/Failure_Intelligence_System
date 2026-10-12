# Evidence — 2026-10-07 planning audit

Raw outputs behind [BASELINE_AUDIT.md](../../BASELINE_AUDIT.md). Baseline commit `24cb2e9`,
conda env `failure-engine` (Python 3.10.19), PAIR v6.3b, all 25 model files matching
`scripts/model_manifest.json`.

These are audit probes, not experiments. They had no pre-registered hypothesis. WP-001 turns
them into tracked harness suites.

| File | What it is |
| --- | --- |
| `hermetic_env.sh` | Environment used for every run. Pre-sets each key found in `.env` to empty so no real credential can load, sets CI's four values, disables telemetry, redirects the blocked-prompt log. Set `SP` to a scratch directory before sourcing it |
| `pytest_baseline_summary.txt` | The 87 test outcomes |
| `probe_baseline.py` → `probe_report.json` | Sections A–J: shipped configuration by routing zone, determinism, domain inference, the `threshold` argument, lite-install simulation, keyword framing, long context, script pilot, cache key, latency |
| `probe_followup.py` → `probe_followup.json` | Framing prefixes on 763 attacks with and without domain inference, padding dose-response, `langdetect` variance |

## Reproduce

From the repository root, in Git Bash:

```bash
export SP=/path/to/an/empty/scratch/dir
source docs/fie_rebuild_2026/evidence/2026-10-07/hermetic_env.sh
unset GROQ_API_KEY          # tiebreaker unreachable = the fail-secure configuration
PYTHONIOENCODING=utf-8 "$PY" docs/fie_rebuild_2026/evidence/2026-10-07/probe_baseline.py "$SP/probe_report.json"
PYTHONIOENCODING=utf-8 "$PY" docs/fie_rebuild_2026/evidence/2026-10-07/probe_followup.py "$SP/probe_followup.json"
```

Runtimes on the development laptop: about 9.5 and 14 minutes.

## What the probes do and do not touch

- They import the working-tree `fie` package and call `scan_prompt(..., use_llama_guard=False)`.
- `fie.multilingual.translate_to_english` is replaced by a stub, so no prompt is sent to a
  translation service. "offline" returns `None`; "translated" returns a fixed benign English
  sentence.
- They clear the in-process scan cache between configurations.
- They write only to the output path given on the command line and to `$FIE_FEEDBACK_PATH`.
- One run is expected to differ from the committed JSON by at most one XSTest-safe verdict:
  the pipeline has a non-deterministic path (audit §5.7).

## Caveats

- Section E simulates a lite install by disabling PAIR and the meta-classifier on a full
  install. It is not a clean-environment measurement.
- Section H uses six hand-written benign prompts per language. It is a pilot.
- Section J and the padding timings are from a laptop, single process, short prompts.
