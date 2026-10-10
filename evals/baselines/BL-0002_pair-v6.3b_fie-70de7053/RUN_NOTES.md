# Run notes — run metadata

**Not part of the deterministic artifact.** Everything here depends on the machine and the moment, or is non-deterministic by design. It is not byte-compared and not used to decide whether two runs agree.

## Cold start

| Profile | import fie (s) | warm-up (s) | first scan (s) | worker wall time (s) |
| --- | --- | --- | --- | --- |
| sdk-offline-failsecure | 0.846 | 7.007 | 0.0303 | 1137.738 |
| lite-simulated | 0.446 | 0.015 | 0.0016 | 24.494 |

## Warm latency (latency suite)

Protocol: result cache cleared before every scan; one discarded pass; per-prompt median over the measured passes. Passes measured: 3.

| Prompts | mean ms | p50 ms | p95 ms | p99 ms | max ms |
| --- | --- | --- | --- | --- | --- |
| 260 | 84.32 | 37.21 | 299.25 | 335.26 | 346.74 |

By input (length buckets):

| Dataset / variant | Prompts | mean chars | mean ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- |
| xstest_safe / base | 100 | 46 | 28.07 | 26.36 | 38.29 |
| harmbench / base | 100 | 96 | 39.06 | 38.74 | 50.49 |
| harmbench / pad_after:84 | 20 | 764 | 175.15 | 176.66 | 203.99 |
| harmbench / pad_after:168 | 20 | 1420 | 284.91 | 283.47 | 317.46 |
| harmbench / pad_after:336 | 20 | 2732 | 300.37 | 298.25 | 339.04 |

## Per-suite scan time and throughput

Timed while the deterministic records were produced (cache not cleared between prompts of a suite, so an exact duplicate prompt returns from cache).

| Suite | Profile | Scans | mean chars | scan time (s) | scans / s | mean ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| std.xstest | sdk-offline-failsecure | 448 | 44 | 13.58 | 33.0 | 30.31 | 38.7 |
| std.orbench_hard | sdk-offline-failsecure | 250 | 116 | 11.1 | 22.5 | 44.39 | 54.52 |
| std.jailbreakbench | sdk-offline-failsecure | 134 | 1421 | 35.62 | 3.8 | 265.84 | 393.49 |
| std.harmbench | sdk-offline-failsecure | 387 | 89 | 15.06 | 25.7 | 38.93 | 52.86 |
| std.strongreject | sdk-offline-failsecure | 242 | 167 | 14.27 | 17.0 | 58.97 | 87.05 |
| std.sorrybench | sdk-offline-failsecure | 387 | 138 | 31.89 | 12.1 | 82.4 | 187.89 |
| std.advbench | sdk-offline-failsecure | 168 | 74 | 17.49 | 9.6 | 104.08 | 203.58 |
| risk.long_input | sdk-offline-failsecure | 1820 | 1202 | 404.3 | 4.5 | 222.14 | 349.69 |
| risk.framing | sdk-offline-failsecure | 4052 | 324 | 333.42 | 12.2 | 82.29 | 307.86 |
| pilot.script | sdk-offline-failsecure | 144 | 70 | 6.89 | 20.9 | 47.85 | 69.59 |
| lite.std | lite-simulated | 1848 | 199 | 23.56 | 78.4 | 12.75 | 51.77 |

## D. Stability — the shipped behaviour, language detector unseeded

The canonical reproducibility profile fixes the language-detector seed. This suite removes it, which is how the product ships, and repeats identical passes. A prompt is unstable when its verdict differs between passes. More passes find more: a prompt that flips rarely can be missed.

| Dataset | n | Passes | Flagged per pass | Unstable prompts | Row indexes |
| --- | --- | --- | --- | --- | --- |
| xstest_safe | 250 | 8 | 132, 132, 132, 132, 132, 132, 132, 132 | 0 | — |
| jailbreakbench | 134 | 3 | 130, 130, 130 | 0 | — |

## Guard accounting

| Profile | Worker exit | Audit events seen by the hook | Classes resolved while unpickling | Blocked imports attempted |
| --- | --- | --- | --- | --- |
| sdk-offline-failsecure | 0 | 723143 | 11 | engine, storage |
| lite-simulated | 0 | 24468 | 0 | engine, joblib, storage |

Classes resolved by unpickling (recorded only; no allowlist is enforced — decision OD-6):

`builtins.bytearray`, `joblib.numpy_pickle.NumpyArrayWrapper`, `numpy._core.multiarray.scalar`, `numpy.dtype`, `numpy.ndarray`, `sklearn.calibration.CalibratedClassifierCV`, `sklearn.calibration._CalibratedClassifier`, `sklearn.calibration._SigmoidCalibration`, `sklearn.svm._classes.LinearSVC`, `xgboost.core.Booster`, `xgboost.sklearn.XGBClassifier`.
