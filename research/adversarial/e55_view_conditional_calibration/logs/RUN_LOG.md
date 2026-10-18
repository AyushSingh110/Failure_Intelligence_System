# E55 run log (append-only)

- 2026-10-17T17:41:40Z scripts/e54_views.py and scripts/e54_common.py copied verbatim from E54 (sha256 cfa9dd49… and 85c87d46…, identical to E54).
- 2026-10-17T17:41:40Z fixtures/e55_variants_v1.json sha256=6b88a1cafd4425aa28df0a1af77c349f6d77abcee128735e40927cbdf2509f01 authored in-session by the assistant before any E55 scoring.
- 2026-10-17T17:42:31Z PREREGISTRATION.md sha256=681bfceb3718b3d832e3cfbfb6445f65a23b6c107f02ade03e6f9c867cbd3584 written before any E55 item was built or scored. HEAD 29bd6e2, tree clean.
- 2026-10-17T17:43:42Z s01 E54 reproduction OK: pad84 {'recall': 0.5583333333333333, 'n': 120}, long {'fpr': 0.2236024844720497, 'n': 161}, macro -5.47
- 2026-10-17T17:44:00Z s01 built 21710 items, 36537 unique texts; calibration bins single 786, multi 369; overlap train {'filler_only': 0, 'tqa_cal': 0, 'tqa_eval': 0, 'halu_cal': 0, 'halu_eval': 0, 'benign_frame': 0, 'benign_manip': 0, 'mmlu_new_cal': 0, 'mmlu_new': 0, 'techdocs_cal': 0, 'techdocs': 0, 'code': 0, 'code_cal': 0, 'secdocs': 0}; overlap e54 {'filler_only': 0, 'tqa_cal': 0, 'tqa_eval': 0, 'halu_cal': 0, 'halu_eval': 0, 'benign_frame': 0, 'benign_manip': 0, 'mmlu_new_cal': 0, 'mmlu_new': 0, 'techdocs_cal': 0, 'techdocs': 0, 'code': 0, 'code_cal': 0, 'secdocs': 0}
- 2026-10-17T18:01:14Z s02 whole: 21010 texts 1002s; BL-0002 dec 0 val 0 max 5.0e-05; E54 max 0.0e+00 over 2143; passed=True
- 2026-10-17T18:05:16Z s02 views: 15527 extra texts 236s; records 21710
- 2026-10-17T18:06:09Z s03 decisions: primary failed; {"H1": false, "H1_target_recall": 0.5452859350850078, "H2": true, "H2a": true, "H2b": true, "H2c": true, "H3": false, "A6_framing": false, "A7_manipulation": false, "bins_adequate": true, "accept_as_research_candidate": false, "H0_retained": true, "status": "failed"}
- 2026-10-17T18:07:15Z s04 latency: <=64: V0 15/21 V1 30/51 ms; 65-256: V0 42/78 V1 130/226 ms; 257-512: V0 80/102 V1 306/352 ms; >512: V0 84/89 V1 400/529 ms
- 2026-10-17T18:08:02Z s05 integrity passed=True
- 2026-10-17T18:09:00Z SHA256SUMS.txt written over 25 files (append-only RUN_LOG.md excluded)
