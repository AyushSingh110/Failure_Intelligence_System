# Run notes — run metadata

**Not part of the deterministic artifact.** Everything here depends on the machine and the moment, or is non-deterministic by design. It is not byte-compared and not used to decide whether two runs agree.

## Cold start

| Profile | import fie (s) | warm-up (s) | first scan (s) | worker wall time (s) |
| --- | --- | --- | --- | --- |
| reference-v6.2 | 0.321 | 1.831 | 0.0126 | 81.341 |

## Per-suite scan time and throughput

Timed while the deterministic records were produced (cache not cleared between prompts of a suite, so an exact duplicate prompt returns from cache).

| Suite | Profile | Scans | mean chars | scan time (s) | scans / s | mean ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| std.xstest | reference-v6.2 | 448 | 44 | 9.46 | 47.3 | 21.12 | 30.28 |
| std.orbench_hard | reference-v6.2 | 250 | 116 | 7.94 | 31.5 | 31.77 | 42.61 |
| std.jailbreakbench | reference-v6.2 | 134 | 1421 | 16.2 | 8.3 | 120.91 | 156.66 |
| std.harmbench | reference-v6.2 | 387 | 89 | 11.55 | 33.5 | 29.83 | 46.62 |
| std.strongreject | reference-v6.2 | 242 | 167 | 11.59 | 20.9 | 47.89 | 82.03 |
| std.sorrybench | reference-v6.2 | 387 | 138 | 16.45 | 23.5 | 42.51 | 76.11 |
| std.advbench | reference-v6.2 | 168 | 74 | 5.17 | 32.5 | 30.79 | 43.58 |

## Guard accounting

| Profile | Worker exit | Audit events seen by the hook | Classes resolved while unpickling | Blocked imports attempted |
| --- | --- | --- | --- | --- |
| reference-v6.2 | 0 | 153231 | 11 | engine, storage |

Classes resolved by unpickling (recorded only; no allowlist is enforced — decision OD-6):

`builtins.bytearray`, `joblib.numpy_pickle.NumpyArrayWrapper`, `numpy._core.multiarray.scalar`, `numpy.dtype`, `numpy.ndarray`, `sklearn.calibration.CalibratedClassifierCV`, `sklearn.calibration._CalibratedClassifier`, `sklearn.calibration._SigmoidCalibration`, `sklearn.svm._classes.LinearSVC`, `xgboost.core.Booster`, `xgboost.sklearn.XGBClassifier`.
