"""Metrics: hand-checked counts and intervals, and the two-axis rule."""
from __future__ import annotations

import math

import pytest

from evals import latency, metrics
from evals.metrics import SingleAxisError
from _helpers import REPO_ROOT

pytest.importorskip("numpy")


def _rec(flagged, zone=None, dataset="d", variant="base", idx=0, layers=("pair_classifier",),
         degraded=(), status="ok", type_="JAILBREAK_ATTEMPT"):
    zone = zone or ("clear_block" if flagged else "allow")
    return {"dataset": dataset, "variant": variant, "idx": idx, "source_idx": idx,
            "input_sha256": f"{idx:064d}", "expected": "attack", "flagged": flagged, "zone": zone,
            "type": type_ if flagged else None, "conf": 0.9 if flagged else 0.0,
            "layers_fired": list(layers) if flagged else [],
            "layer_scores": {"pair_classifier": 0.9 if flagged else 0.0, "regex": 0.0},
            "degraded": list(degraded), "status": status}


# ── Wilson interval ──────────────────────────────────────────────────────────

def test_wilson_matches_independently_computed_values():
    """Expected values come from solving the Wilson quadratic directly, not from metrics.wilson."""
    lo, hi = metrics.wilson(132, 250)
    assert lo == pytest.approx(0.46616, abs=5e-6) and hi == pytest.approx(0.58899, abs=5e-6)
    lo, hi = metrics.wilson(130, 134)
    assert lo == pytest.approx(0.92576, abs=5e-6) and hi == pytest.approx(0.98833, abs=5e-6)
    assert metrics.wilson(0, 0) == [0.0, 0.0]
    lo, hi = metrics.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.27753, abs=5e-6)
    lo, hi = metrics.wilson(10, 10)
    assert hi == pytest.approx(1.0) and hi <= 1.0 and lo == pytest.approx(0.72247, abs=5e-6)


# ── cells ────────────────────────────────────────────────────────────────────

def test_attack_cell_counts_and_rate():
    recs = [_rec(True, idx=i) for i in range(7)] + [_rec(False, idx=i) for i in range(7, 10)]
    recs[0]["zone"] = "uncertain_block"
    c = metrics.cell(recs, "attack", REPO_ROOT)
    assert (c["n"], c["flagged"], c["not_flagged"]) == (10, 7, 3)
    assert c["confusion"] == {"tp": 7, "fn": 3}
    assert c["zones"] == {"allow": 3, "uncertain_block": 1, "clear_block": 6}
    assert c["rate"]["metric"] == "recall" and c["rate"]["point"] == pytest.approx(0.7)
    lo, hi = c["rate"]["bootstrap95"]
    assert 0.3 <= lo < 0.7 < hi <= 1.0
    assert c["rate"]["wilson95"] == metrics.wilson(7, 10)
    assert c["scan_errors"] == 0 and c["degraded_records"] == 0
    assert c["coverage"] == {"pair_classifier": 1.0, "regex": 1.0}


def test_benign_cell_uses_over_refusal_and_fp_tn():
    recs = [_rec(True, idx=i) for i in range(4)] + [_rec(False, idx=i) for i in range(4, 10)]
    c = metrics.cell(recs, "benign", REPO_ROOT)
    assert c["confusion"] == {"fp": 4, "tn": 6}
    assert c["rate"]["metric"] == "over_refusal" and c["rate"]["point"] == pytest.approx(0.4)


def test_duplicates_are_counted_and_unique_reported():
    recs = [_rec(True, idx=0), _rec(True, idx=0), _rec(False, idx=1)]
    c = metrics.cell(recs, "attack", REPO_ROOT)
    assert c["n"] == 3 and c["n_unique"] == 2


def test_degraded_layers_and_scan_errors_are_surfaced():
    recs = [_rec(False, idx=0, degraded=["pair_classifier"]), _rec(False, idx=1, status="error:RuntimeError"),
            _rec(True, idx=2)]
    c = metrics.cell(recs, "attack", REPO_ROOT)
    assert c["degraded_records"] == 1 and c["scan_errors"] == 1
    assert c["coverage"]["pair_classifier"] == pytest.approx(2 / 3)
    assert c["coverage"]["regex"] == 1.0


def test_drivers_name_the_layers_and_types_behind_flags():
    recs = [_rec(True, idx=0), _rec(True, idx=1, layers=("direct_harm", "pair_classifier")),
            _rec(True, idx=2, type_="DIRECT_HARMFUL_REQUEST"), _rec(False, idx=3)]
    c = metrics.cell(recs, "benign", REPO_ROOT)
    assert dict(c["drivers"]["layers"]) == {"pair_classifier": 2, "direct_harm+pair_classifier": 1}
    assert dict(c["drivers"]["types"]) == {"JAILBREAK_ATTEMPT": 2, "DIRECT_HARMFUL_REQUEST": 1}


def test_pilot_cells_carry_no_rate():
    c = metrics.cell([_rec(True, idx=i) for i in range(6)], "benign", REPO_ROOT, with_intervals=False)
    assert "rate" not in c and c["flagged"] == 6


def test_bootstrap_is_reproducible_and_uses_the_published_implementation():
    flags = [1] * 130 + [0] * 4
    a = metrics.rate_block(flags, "attack", REPO_ROOT)
    b = metrics.rate_block(list(flags), "attack", REPO_ROOT)
    assert a == b
    stats = metrics.stats_utils(REPO_ROOT)
    assert (stats.DEFAULT_SEED, stats.DEFAULT_B) == (42, 10_000)
    point, lo, hi = stats.bootstrap_ci([1] * 134, flags, metric="recall")
    assert a["point"] == pytest.approx(point) and a["bootstrap95"] == [lo, hi]


# ── macro / micro ────────────────────────────────────────────────────────────

def test_macro_is_the_unweighted_mean_and_differs_from_micro():
    sets = [[1] * 9 + [0], [1] * 50 + [0] * 50]            # 90% of 10, 50% of 100
    macro = metrics.macro_block(sets)
    assert macro["point"] == pytest.approx(0.70) and macro["sets"] == 2
    lo, hi = macro["bootstrap95"]
    assert lo < 0.70 < hi
    pooled = [f for s in sets for f in s]
    assert sum(pooled) / len(pooled) == pytest.approx(59 / 110)
    assert metrics.macro_block(sets) == macro, "stratified bootstrap must be reproducible"


def test_canonical_headline_arithmetic():
    """The approved v6.3b counts give a macro recall of 88.25%."""
    counts = [(130, 134), (326, 387), (218, 242), (316, 387)]
    macro = metrics.macro_block([[1] * k + [0] * (n - k) for k, n in counts])
    assert round(100 * macro["point"], 2) == 88.25
    assert sum(k for k, _ in counts) == 990 and sum(n for _, n in counts) == 1150
    reference = [(129, 134), (317, 387), (217, 242), (291, 387)]
    ref = metrics.macro_block([[1] * k + [0] * (n - k) for k, n in reference])
    assert round(100 * ref["point"], 2) == 85.76


# ── the two-axis rule ────────────────────────────────────────────────────────

def test_security_table_requires_both_axes():
    attack = {"jailbreakbench": {"flagged": 130, "n": 134}}
    benign = {"xstest_safe": {"flagged": 132, "n": 250}}
    assert metrics.security_table(attack, benign) == {"attack": attack, "benign": benign}
    with pytest.raises(SingleAxisError, match="benign utility is missing"):
        metrics.security_table(attack, {})
    with pytest.raises(SingleAxisError, match="attack performance is missing"):
        metrics.security_table({}, benign)
    with pytest.raises(SingleAxisError, match="Single-axis run. Not a security result."):
        metrics.security_table(attack, None)


# ── latency statistics ───────────────────────────────────────────────────────

def test_percentiles_and_describe():
    values = list(range(1, 101))
    assert latency.percentile(values, 50) == pytest.approx(50.5)
    assert latency.percentile(values, 95) == pytest.approx(95.05)
    assert latency.percentile([7.0], 99) == 7.0 and latency.percentile([], 50) == 0.0
    d = latency.describe([10, 20, 30, 40])
    assert d == {"n": 4, "mean_ms": 25.0, "p50_ms": 25.0, "p95_ms": 38.5, "p99_ms": 39.7, "max_ms": 40.0}
    assert latency.describe([]) == {"n": 0}


def test_stability_reports_prompts_whose_verdict_changed(tmp_path):
    import json
    out = tmp_path / "nondeterministic"
    out.mkdir()
    rows = [{"dataset": "x", "pass": 0, "n": 5, "flagged_idx": [1, 2]},
            {"dataset": "x", "pass": 1, "n": 5, "flagged_idx": [1, 2, 4]},
            {"dataset": "x", "pass": 2, "n": 5, "flagged_idx": [1, 2]},
            {"dataset": "y", "pass": 0, "n": 3, "flagged_idx": [0]},
            {"dataset": "y", "pass": 1, "n": 3, "flagged_idx": [0]}]
    (out / "stability.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    res = latency.stability(tmp_path)
    assert res["x"] == {"n": 5, "passes": 3, "flagged_per_pass": [2, 3, 2], "unstable_idx": [4]}
    assert res["y"]["unstable_idx"] == []
    assert latency.stability(tmp_path, "absent") is None


def test_warm_latency_takes_the_per_prompt_median(tmp_path):
    import json
    out = tmp_path / "nondeterministic"
    out.mkdir()
    rows = []
    for p, values in enumerate([(10, 100), (30, 300), (20, 200)], start=1):
        for idx, ms in enumerate(values):
            rows.append({"pass": p, "idx": idx, "dataset": "d", "variant": "base" if idx == 0 else "pad_after:84",
                         "chars": 50 * (idx + 1), "ms": ms})
    (out / "latency.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    res = latency.warm_latency(tmp_path)
    assert res["overall"]["n"] == 2 and res["overall"]["passes"] == 3
    assert res["overall"]["mean_ms"] == 110.0            # medians 20 and 200
    assert res["by_bucket"]["d / base"]["p50_ms"] == 20.0
    assert res["by_bucket"]["d / pad_after:84"]["mean_chars"] == 100
