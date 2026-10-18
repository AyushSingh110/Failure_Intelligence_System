# E54 run log (append-only)

- 2026-10-17T12:58:07Z fixtures/long_benign_authored_v1.jsonl sha256=78702e377f116d851a70d20a9b02b20cc8e1e6620743f24bc4f7af0a4412daa2 authored in-session by the assistant (Claude) before any E54 scoring; no external model or API used.
- 2026-10-17T12:59:21Z PREREGISTRATION.md sha256=5aa0d37065994d5b44c29064e09b5f88a0937d4b3840a1699e875572128a2024 written before any E54 view or candidate scoring. HEAD 700db5f, tree clean.
- 2026-10-17T13:01:04Z s01 built 8467 items; unique texts 20147 (whole 8312, extra views 11835); overlap with training {}
- 2026-10-17T13:05:33Z s02 whole: 8312 texts in 227s; gate BL-0002 dec mm 0, val mm 0, max 5.0e-05; E53 max |d| 5.0e-16 over 7720 items; passed=True
- 2026-10-17T13:10:11Z s02 views: 11835 extra view texts in 274s; per-prompt records for 8467 items
- 2026-10-17T13:10:38Z s03 metrics written; H1 []; H2 []; H3 []; useful []; H0 retained True
- 2026-10-17T13:11:52Z s06 EXPLORATORY view-conditional threshold analysis written (not a decision input)
- 2026-10-17T13:13:48Z s04 latency written: <=64: V0 p50 12ms V3 p50 31ms; 65-256: V0 p50 42ms V3 p50 201ms; 257-512: V0 p50 80ms V3 p50 976ms; >512: V0 p50 81ms V3 p50 1232ms
- 2026-10-17T13:14:15Z s05 integrity passed=True
- 2026-10-17T13:15:13Z SHA256SUMS.txt written over 25 files (append-only RUN_LOG.md excluded)
