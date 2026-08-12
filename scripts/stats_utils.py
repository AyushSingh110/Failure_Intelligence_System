"""
Confidence intervals for every reported metric.

WHY
---
Every headline number in this project is currently a bare point estimate:
"53.6% XSTest over-refusal", "85.8% macro recall". The first question any
reviewer asks is *"plus or minus what, over how many samples?"* — and without
an interval there is no way to tell a real improvement from noise.

XSTest has 250 prompts. A difference of 2 points on 250 samples is well inside
sampling error. Reporting it as an improvement would be wrong, and the only
way to know is to compute the interval.

METHOD
------
Bootstrap percentile intervals: resample the evaluation set with replacement
B times, recompute the metric on each resample, take the 2.5th and 97.5th
percentiles. No distributional assumption, works for any metric, and is the
standard choice when the sampling distribution of the statistic is unknown.

For comparing two systems on the SAME prompts, use `paired_bootstrap_diff`.
Comparing two independent intervals and checking whether they overlap is a
weaker test — overlapping intervals do not imply a non-significant difference.
The paired test uses the fact that both systems saw identical inputs, which is
exactly the situation in a guardrail comparison.

Usage:
    from scripts.stats_utils import bootstrap_ci, paired_bootstrap_diff

    lo, hi = bootstrap_ci(labels, preds, metric="recall")
    print(f"recall {recall:.1%} [95% CI {lo:.1%}–{hi:.1%}]")
"""
from __future__ import annotations

import numpy as np

# Fixed so every reported interval is reproducible. Change it and every number
# moves slightly, which is exactly why it must be pinned and recorded.
DEFAULT_SEED = 42
DEFAULT_B = 10_000


def _metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute one scalar metric. Returns nan when undefined for a resample."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    if metric == "recall":            # of the attacks, how many were caught
        return tp / (tp + fn) if (tp + fn) else float("nan")
    if metric == "precision":
        return tp / (tp + fp) if (tp + fp) else float("nan")
    if metric == "fpr":               # of the benign, how many were blocked
        return fp / (fp + tn) if (fp + tn) else float("nan")
    if metric == "over_refusal":      # alias: fraction of safe prompts blocked
        return fp / (fp + tn) if (fp + tn) else float("nan")
    if metric == "accuracy":
        total = tp + fp + fn + tn
        return (tp + tn) / total if total else float("nan")
    if metric == "f1":
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else float("nan")
    raise ValueError(f"unknown metric: {metric}")


def bootstrap_ci(
    y_true,
    y_pred,
    metric: str = "recall",
    b: int = DEFAULT_B,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for one metric.

    Returns (point_estimate, ci_low, ci_high).

    `y_true` / `y_pred` are 0/1 arrays over the SAME prompts, in the same order.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    point = _metric(y_true, y_pred, metric)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    stats = np.array([_metric(y_true[i], y_pred[i], metric) for i in idx])
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return point, float("nan"), float("nan")

    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_bootstrap_diff(
    y_true,
    pred_a,
    pred_b,
    metric: str = "recall",
    b: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Test whether system B differs from system A on the same evaluation set.

    Resamples PROMPTS (not predictions), so both systems are always compared on
    the identical resample — that pairing is what gives the test its power.

    Returns the observed difference, its CI, a two-sided bootstrap p-value, and
    a `significant` flag (CI excludes zero).
    """
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    n = len(y_true)

    obs = _metric(y_true, pred_b, metric) - _metric(y_true, pred_a, metric)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    diffs = np.array([
        _metric(y_true[i], pred_b[i], metric) - _metric(y_true[i], pred_a[i], metric)
        for i in idx
    ])
    diffs = diffs[~np.isnan(diffs)]

    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    # Two-sided p: how often the resampled difference lands on the other side
    # of zero from the observed effect.
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "metric":      metric,
        "difference":  float(obs),
        "ci_low":      lo,
        "ci_high":     hi,
        "p_value":     float(min(p, 1.0)),
        "significant": bool(lo > 0 or hi < 0),
        "n":           int(n),
        "seed":        seed,
        "bootstraps":  int(b),
    }


def format_ci(point: float, lo: float, hi: float, pct: bool = True) -> str:
    """Render as '53.6% [47.4–59.8]' for direct use in a table."""
    if pct:
        return f"{point:.1%} [{lo:.1%}–{hi:.1%}]"
    return f"{point:.4f} [{lo:.4f}–{hi:.4f}]"


if __name__ == "__main__":
    # Self-check: a 250-prompt set (XSTest size) at a 53.6% rate has an
    # interval roughly +/-6 points. That width is the entire argument for
    # reporting intervals — a 2-point "improvement" would be indistinguishable
    # from noise at this sample size.
    rng = np.random.default_rng(0)
    n = 250
    y_true = np.zeros(n, dtype=int)          # all benign, as in XSTest
    y_pred = (rng.random(n) < 0.536).astype(int)
    point, lo, hi = bootstrap_ci(y_true, y_pred, metric="over_refusal")
    print(f"XSTest-sized over-refusal: {format_ci(point, lo, hi)}")
    print(f"interval width: {(hi - lo) * 100:.1f} percentage points")
