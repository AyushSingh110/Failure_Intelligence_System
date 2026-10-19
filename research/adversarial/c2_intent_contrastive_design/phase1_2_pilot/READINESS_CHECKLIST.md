# C2 Phase 1.2 — Pilot readiness checklist

| # | Item | Status | Evidence |
| --- | --- | --- | --- |
| 1 | 200 unique rows sampled deterministically, no model prediction used | **Done** | `sampling_manifest.json` (seed 20261018, 21 strata, no shortages) |
| 2 | Every row traceable to its original file and line, with hashes | **Done, verified** | `owner_only/pilot_mapping.json`; `logs/integrity_check.json` → `mapping_recovers_original_rows: true` |
| 3 | Two blinded packets, same 200 examples (+20 hidden repeats), different orders | **Done, verified** | 220 rows each; `same_200_examples`, `orders_differ`; minimum repeat gap 69 (A) / 53 (B) |
| 4 | No labels, sources, scores or other answers in the packets; annotations blank | **Done, verified** | `no_label_score_or_source_columns`, `all_annotations_blank` |
| 5 | Reviewer guide and schema | **Done** | `REVIEWER_GUIDE.md`, `annotation_schema.json` |
| 6 | Agreement / self-consistency / unclear / adjudication script | **Done; self-tested on synthetic in-memory data only** | `scripts/compute_pilot_agreement.py --self-test` (identical → κ = 1.0; independent → κ = 0.008). The real run refuses until annotations exist |
| 7 | Reviewer assignments R1, R2, ADJ | **Missing: owner (OD-C2-4)** | Placeholders in `PILOT_PROTOCOL.md` §2 |
| 8 | Owner approval to start the pilot | **Missing** | — |
| 9 | Human annotations | **None exist** | `annotations_returned/` is empty |
| 10 | Acceptance criteria evaluated | **Not evaluated** | Requires item 9 |
| 11 | Workload | **Estimate** | ≈ 10–15 person-hours (`PILOT_PROTOCOL.md` §6) |
| 12 | Prior evidence and protected paths untouched | **Verified** | E53/E54/E55, Phase 1 and Phase 1.1 checksums verify; `git_status_protected` empty; model digests unchanged |
| 13 | Checksums for this package | **Done** | `SHA256SUMS_PHASE1_2.txt` |

## Verdict

**Ready for human annotation. Not started.**

Next human action: the owner names R1, R2 and ADJ, approves the pilot, and sends each reviewer only their
own packet plus the guide and schema.
