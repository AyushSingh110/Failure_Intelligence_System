# C2 Phase 1.1 — Readiness checklist

Status words:

- **Verified:** measured locally in Phase 1 or 1.1.
- **Estimate:** reasoned, not measured.
- **Depends:** waiting on an owner decision.
- **Missing:** not available.

No performance improvement is claimed or estimated.

| # | Requirement | Status | Evidence / gap |
| --- | --- | --- | --- |
| 1 | ≈ 1,300 rows identified for human review | **Verified** (counts) | Red-team 485, HarmfulQA 465, ToxicChat 251 (180 toxic_query + 71 jailbreak), llama paraphrase 160 (82 refusal confirmations + 78 normal), 22 cross-label conflicts (overlapping): `audit/training_data_audit.json` |
| 2 | Review protocol | **Ready (spec)** | [C2_PHASE1_1_HUMAN_REVIEW_PROTOCOL.md](C2_PHASE1_1_HUMAN_REVIEW_PROTOCOL.md) |
| 3 | Reviewers (2 + adjudicator) | **Depends: OD-C2-4** | Not named. A single reviewer cannot satisfy the protocol |
| 4 | Review effort | **Estimate** | ≈ 100 reviewer-hours including pilot, 10% audit, adjudication and pair review |
| 5 | Refusal and dual-use rules | **Depends: OD-C2-2, OD-C2-3** | Recommended defaults in the decision gate |
| 6 | ≥ 480 reviewed training pairs | **Missing** | Only 1 of 595 mined candidates passes the surface screens (`audit/phase1_1_pair_candidates.json`). Benign twins must be written (human, or model-drafted under OD-C2-4). 595 mined rows remain usable as topic seeds; 195 unique benign partners |
| 7 | ≥ 200 held-out reviewed pairs | **Missing** | Same as 6. Held-out pairs need their own seeds and families |
| 8 | Local drafting tool | **Unverified** | Ollama binary and 6 models on disk (manifests read). Not started; no inference. Only `llama3.2` (2.0 GB) fits the 4 GB GPU |
| 9 | Topic-group coverage for pairs (12 groups) | **Estimate** | Group assignment of the mined seeds is not done (needs review); per-group counts Unknown |
| 10 | Source-held-out (LOSO) partitions | **Verified feasible** | 12 sources have ≥ 100 `tr` rows. Group ids exist (source, file); aug_v63 seed families recoverable (median back-translation-to-seed cosine 0.94; 6 ambiguous) |
| 11 | Register-held-out partitions | **Verified feasible** | hh-rlhf (red-team + helpful), alpaca (two source names), medical (MedQuad + medqa) are identifiable |
| 12 | Calibration and development splits | **Specified; not built** | Group-aware rules in design §7.1. Built only after the labels exist |
| 13 | Independent frozen final test | **Missing; Depends: OD-C2-5** | Not constructible offline. Options and freeze procedure in [C2_PHASE1_1_FINAL_TEST_OPTIONS.md](C2_PHASE1_1_FINAL_TEST_OPTIONS.md) |
| 14 | Second encoder for deduplication | **Depends: OD-C2-6** | Lexical MinHash possible locally; a semantic second encoder needs a download |
| 15 | Training environment (frozen encoder) | **Verified** | `failure-engine`: sklearn 1.7.2, numpy 2.2.6, scipy, onnxruntime, tokenizers; no torch (OD-C2-7). 20 CPU threads, RTX 3050 4 GB. Cached MiniLM embeddings for all 4,236 training rows (E53, checksum-verified) |
| 16 | A0 reproduction path | **Verified** | E53 gate: A0 = shipped head (agreement 100%, max \|Δ\| 1.5e-6) with recipe row order (E53 D9) |
| 17 | Artifact provenance and checksums | **Verified for existing evidence** | E53, E54, E55 and Phase 1 `SHA256SUMS.txt` verify. Phase 1.1 docs in `SHA256SUMS_PHASE1_1.txt`. E53–E55 `results/` not in git (OD-C2-1) |
| 18 | Manipulation (M) training data | **Missing** | 71 local candidates only; OD-C2-8 default is no M claims |
| 19 | RAM available for 7–8B CPU-offload drafting | **Unknown** | Not measured (`wmic` unavailable) |

## Verdict

**Not ready for C2 data construction or training.**

- **Blocking now:**
  - OD-C2-4 (reviewers);
  - OD-C2-2 and OD-C2-3 (label rules);
  - OD-C2-5 (final test, which must be frozen before any training).
- **Largest practical gaps:**
  - writing ≈ 680 benign twins, because mining yields topic seeds but not minimal pairs;
  - ≈ 100 reviewer-hours.
- **Already in place:**
  - compute;
  - the A0 baseline path;
  - leave-one-source-out and register partitions;
  - provenance.

## Next permitted step after owner approval

Stage S1 only: the 200-row taxonomy pilot. Its acceptance gates are κ ≥ 0.60 on H and M, self-consistency
≥ 90%, and `unclear` ≤ 15%.
