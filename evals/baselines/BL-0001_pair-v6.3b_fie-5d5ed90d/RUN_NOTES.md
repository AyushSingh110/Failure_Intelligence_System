# Run notes — run metadata

**Not part of the deterministic artifact.** Everything here depends on the machine and the moment, or is non-deterministic by design. It is not byte-compared and not used to decide whether two runs agree.

## Cold start

| Profile | import fie (s) | warm-up (s) | first scan (s) | worker wall time (s) |
| --- | --- | --- | --- | --- |
| sdk-offline-failsecure | 0.404 | 2.304 | 0.0195 | 738.218 |
| lite-simulated | 0.297 | 0.015 | 0.0016 | 11.537 |

## Warm latency (latency suite)

Protocol: result cache cleared before every scan; one discarded pass; per-prompt median over the measured passes. Passes measured: 3.

| Prompts | mean ms | p50 ms | p95 ms | p99 ms | max ms |
| --- | --- | --- | --- | --- | --- |
| 260 | 56.75 | 35.52 | 167.16 | 171.53 | 174.36 |

By input (length buckets):

| Dataset / variant | Prompts | mean chars | mean ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- |
| xstest_safe / base | 100 | 46 | 24.49 | 25.98 | 37.55 |
| harmbench / base | 100 | 96 | 37.33 | 37.68 | 50.83 |
| harmbench / pad_after:84 | 20 | 764 | 109.66 | 119.76 | 129.37 |
| harmbench / pad_after:168 | 20 | 1420 | 156.2 | 162.06 | 169.78 |
| harmbench / pad_after:336 | 20 | 2732 | 162.79 | 166.32 | 173.01 |

## Per-suite scan time and throughput

Timed while the deterministic records were produced (cache not cleared between prompts of a suite, so an exact duplicate prompt returns from cache).

| Suite | Profile | Scans | mean chars | scan time (s) | scans / s | mean ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| std.xstest | sdk-offline-failsecure | 448 | 44 | 9.71 | 46.1 | 21.67 | 31.24 |
| std.orbench_hard | sdk-offline-failsecure | 250 | 116 | 11.52 | 21.7 | 46.09 | 73.09 |
| std.jailbreakbench | sdk-offline-failsecure | 134 | 1421 | 20.9 | 6.4 | 155.94 | 196.24 |
| std.harmbench | sdk-offline-failsecure | 387 | 89 | 15.85 | 24.4 | 40.96 | 60.16 |
| std.strongreject | sdk-offline-failsecure | 242 | 167 | 14.35 | 16.9 | 59.3 | 98.45 |
| std.sorrybench | sdk-offline-failsecure | 387 | 138 | 19.98 | 19.4 | 51.62 | 89.12 |
| std.advbench | sdk-offline-failsecure | 168 | 74 | 6.3 | 26.7 | 37.51 | 50.6 |
| risk.long_input | sdk-offline-failsecure | 1820 | 1202 | 226.01 | 8.1 | 124.18 | 185.18 |
| risk.framing | sdk-offline-failsecure | 4052 | 324 | 241.32 | 16.8 | 59.55 | 166.85 |
| pilot.script | sdk-offline-failsecure | 144 | 70 | 6.21 | 23.2 | 43.11 | 64.83 |
| lite.std | lite-simulated | 1848 | 199 | 10.96 | 168.5 | 5.93 | 25.52 |

## D. Stability — the shipped behaviour, language detector unseeded

The canonical reproducibility profile fixes the language-detector seed. This suite removes it, which is how the product ships, and repeats identical passes. A prompt is unstable when its verdict differs between passes. More passes find more: a prompt that flips rarely can be missed.

| Dataset | n | Passes | Flagged per pass | Unstable prompts | Row indexes |
| --- | --- | --- | --- | --- | --- |
| xstest_safe | 250 | 8 | 132, 133, 132, 132, 132, 132, 133, 132 | 1 | 99 |
| jailbreakbench | 134 | 3 | 130, 130, 130 | 0 | — |

## Guard accounting

| Profile | Worker exit | Audit events seen by the hook | Classes resolved while unpickling | Blocked imports attempted |
| --- | --- | --- | --- | --- |
| sdk-offline-failsecure | 0 | 723127 | 11 | engine, storage |
| lite-simulated | 0 | 24443 | 0 | engine, joblib, storage |

Classes resolved by unpickling (recorded only; no allowlist is enforced — decision OD-6):

`builtins.bytearray`, `joblib.numpy_pickle.NumpyArrayWrapper`, `numpy._core.multiarray.scalar`, `numpy.dtype`, `numpy.ndarray`, `sklearn.calibration.CalibratedClassifierCV`, `sklearn.calibration._CalibratedClassifier`, `sklearn.calibration._SigmoidCalibration`, `sklearn.svm._classes.LinearSVC`, `xgboost.core.Booster`, `xgboost.sklearn.XGBClassifier`.
