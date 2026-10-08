"""
The human report.

REPORT.md is a deterministic artifact, so rendering is checked two ways: against
a stored golden file, byte for byte, and for the properties the approved plan
requires (exact counts first, both axes, labels, wording).

Regenerate the golden file deliberately:  EVALS_UPDATE_GOLDEN=1 pytest tests/evals/test_report.py
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

from evals import canonical, report, transforms
from _helpers import REPO_ROOT

GOLDEN = Path(__file__).parent / "data" / "report_golden.md"


def _rate(k, n, metric):
    return {"metric": metric, "point": k / n, "bootstrap95": [max(0.0, k / n - 0.05), min(1.0, k / n + 0.05)],
            "wilson95": [max(0.0, k / n - 0.04), min(1.0, k / n + 0.04)]}


def _brief(k, n, axis):
    conf = {"tp": k, "fn": n - k} if axis == "attack" else {"fp": k, "tn": n - k}
    return {"n": n, "n_unique": n, "flagged": k, "rate": _rate(k, n, "recall" if axis == "attack" else "over_refusal"),
            "zones": {"allow": n - k, "uncertain_block": k // 4, "clear_block": k - k // 4}, "confusion": conf}


def _fingerprint() -> dict:
    return {
        "schema_version": 1,
        "keys": {"dataset_key": "d" * 64, "config_key": "c" * 64, "subject_key": "5" * 64, "env_key": "e" * 64},
        "identity": {
            "subject": {"fie_tree_sha256": "a1" * 32, "fie_py_files": 35, "fie_version": "1.18.0"},
            "git": {"commit": "c0ffee" + "0" * 34, "dirty_subject": False, "dirty_other": True},
            "models": {"sdk-offline-failsecure": {
                "pair_classifier": {"file": "pair_intent_classifier_v6_3b.pkl", "sha256": "9c682b28" + "0" * 56,
                                    "size": 11349, "in_manifest": True}},
                "lite-simulated": {}},
            "model_manifest": {"path": "scripts/model_manifest.json", "release_tag": "models-v1.18.0"},
            "declared": {"sdk-offline-failsecure": {"pair": {"declared_version": "v6.3b", "threshold": 0.5}}},
            "datasets": {"jailbreakbench": {"content_sha256": "c2" * 32, "rows": 134, "unique": 134, "label": "attack"},
                         "xstest_safe": {"content_sha256": "a3" * 32, "rows": 250, "unique": 250, "label": "benign"}},
            "suites": {}, "fixtures": {},
            "harness": {"version": "1.0.0", "schema_version": 1, "tree_sha256": "h" * 64, "py_files": 16},
        },
        "configuration": {"sdk-offline-failsecure": {
            "translation": "unavailable (harness stub returns None)", "langdetect_seed": 0,
            "subject": {"pair": {"loaded": True, "threshold": 0.5}, "meta": {"loaded": True, "threshold": 0.41},
                        "tiebreaker": "disabled", "operator_overrides": {}, "framing_dampen_factor": 0.72,
                        "thresholds": {"scan": 0.65, "attack": {"JAILBREAK_ATTEMPT": 0.65, "TOKEN_SMUGGLING": 0.88}},
                        "scan_args": {"use_llama_guard": False, "domain": None}}}},
        "environment": {},
    }


def _summary() -> dict:
    head_attack = {"jailbreakbench": _brief(130, 134, "attack")}
    head_benign = {"xstest_safe": _brief(132, 250, "benign")}
    return {
        "schema_version": 1, "canonical": True, "keys": _fingerprint()["keys"],
        "primary_profile": "sdk-offline-failsecure",
        "profiles": {"sdk-offline-failsecure": {
            "title": "canonical reproducibility/evaluation profile",
            "summary": "Every network path closed. Not the production runtime environment.",
            "pair_declared_version": "v6.3b", "pair_sha256_8": "9c682b28"}},
        "dataset_display": {"jailbreakbench": "JailbreakBench", "xstest_safe": "XSTest (safe)",
                            "advbench": "AdvBench (case study)"},
        "suites": {"std.jailbreakbench": {"group": "standard", "cells": [
            dict(_brief(130, 134, "attack"), dataset="jailbreakbench", variant="base", axis="attack")]}},
        "single_axis_notice": None,
        "headline": {
            "complete": False, "attack_sets": ["jailbreakbench"], "benign_sets": ["xstest_safe"],
            "attack": head_attack, "benign": head_benign,
            "macro_recall": {"point": 0.970149, "bootstrap95": [0.94, 0.993], "sets": 1},
            "micro_recall": {"flagged": 130, "n": 134, "point": 0.970149, "bootstrap95": [0.94, 0.993],
                             "wilson95": [0.926, 0.988]},
            "clear_block_only_macro_recall": {"point": 0.910448, "bootstrap95": [0.86, 0.955], "sets": 1},
            "contrast": {}, "case_study": {"advbench": _brief(163, 168, "attack")},
            "case_study_note": "Case study. Excluded from macro and micro recall.",
        },
        "approved_counts": {"profile": "sdk-offline-failsecure", "source": "approved plan",
                            "datasets": {"jailbreakbench": {"expected_flagged": 130, "flagged": 130, "n": 134, "match": True}},
                            "all_match": True},
        "zones": {"xstest_safe": {"n": 250, "allow": 118, "uncertain_block": 63, "clear_block": 69}},
        "risk": {
            "long_input": {"sample": "First 120 unique rows by position.",
                           "attack": [{"variant": "pad_after:84", "n": 120, "flagged": 12,
                                       "rate": _rate(12, 120, "recall"), "base_caught": 104,
                                       "base_caught_still_caught": 11, "layers_when_caught": [["direct_harm", 9]]}],
                           "benign": [{"variant": "pad_after:84", "dataset": "xstest_safe", "n": 250, "flagged": 5,
                                       "flagged_unpadded_same_rows": 132, "rate": _rate(5, 250, "over_refusal")}]},
            "framing": {"attack": [{"variant": "frame:legal", "n": 763, "flagged": 445,
                                    "rate": _rate(445, 763, "recall"), "unframed_flagged": 674,
                                    "base_caught_now_missed": 229, "base_missed_now_caught": 0}],
                        "benign": [{"variant": "frame:legal", "dataset": "xstest_safe", "n": 250, "flagged": 40,
                                    "unframed_flagged": 132, "rate": _rate(40, 250, "over_refusal")}]},
            "script_pilot": {"label": "PILOT", "note": "Counts only. Six hand-written prompts per group.",
                             "rows": [{"variant": "base", "group": "hindi", "n": 6, "flagged": 6,
                                       "zones": {"clear_block": 6}, "layers": [["gcg_suffix", 6]]}]},
            "lite": {"classifier_loaded": False, "classifier_load_error": "missing dependency: joblib",
                     "scans": 1848, "scans_reporting_full_coverage": 1848, "share_reporting_full_coverage": 1.0,
                     "scans_where_classifier_scored_zero_with_status_ok": 1848, "honest": False,
                     "reading": "The classifier did not load, yet every scan reported an empty degraded_layers list.",
                     "by_dataset": {"jailbreakbench": _brief(14, 134, "attack")}},
        },
        "integrity": {"datasets_verified": 2, "models_verified": {"sdk-offline-failsecure": 1}, "scan_errors": 0,
                      "degraded_records": 0,
                      "hermetic": {"sdk-offline-failsecure": {
                          "guard_ok": True, "non_canary_events": 0, "canaries_denied": 5, "audit_hook_ran": True,
                          "prearm": ["platform.uname", "urllib3.ipv6_probe"], "models_reverified_after_run": True}}},
        "comparison": {"baseline_id": None, "status": "no baseline selected"},
    }


def _render(summary=None) -> str:
    profiles = transforms.load_profiles(REPO_ROOT)
    suites = transforms.load_suites(REPO_ROOT)
    return report.render(summary or _summary(), _fingerprint(), profiles, suites)


def test_report_matches_the_golden_file_byte_for_byte():
    text = _render()
    data = text.encode("utf-8")
    if os.environ.get("EVALS_UPDATE_GOLDEN") == "1" or not GOLDEN.exists():
        canonical.write_bytes(GOLDEN, data)
    assert canonical.read_lf(GOLDEN) == data, (
        "REPORT.md rendering changed. If intended, regenerate with EVALS_UPDATE_GOLDEN=1 and review the diff.")


def test_rendering_is_deterministic_and_lf_only():
    a, b = _render(), _render(copy.deepcopy(_summary()))
    assert a == b
    assert "\r" not in a


def test_exact_counts_come_before_percentages():
    text = _render()
    assert "**130 / 134**" in text and "**132 / 250**" in text
    line = next(l for l in text.splitlines() if "JailbreakBench" in l and "130 / 134" in l)
    assert line.index("130 / 134") < line.index("97.0%")


def test_both_axes_appear_in_the_headline():
    text = _render()
    head = text[text.index("## 3. Headline"):text.index("## 4. Routing zones")]
    assert "Attack recall" in head and "Benign over-refusal" in head
    assert "Read the two axes together" in head


def test_single_axis_run_is_declared_as_not_a_security_result():
    summary = _summary()
    summary["headline"] = None
    summary["single_axis_notice"] = "a security table needs both axes; benign utility is missing."
    text = _render(summary)
    assert text.splitlines()[2].startswith("> **Single-axis run. Not a security result.**")
    head = text[text.index("## 3. Headline"):]
    assert "Single-axis run. Not a security result." in head
    assert "Macro recall" not in text and "Attack recall |" not in text


def test_model_version_and_hash_are_named_in_the_title_and_the_headline():
    text = _render()
    assert text.count("PAIR v6.3b (`9c682b28`)") >= 2
    assert "`sdk-offline-failsecure`" in text


def test_profile_wording_follows_the_approved_decision():
    text = _render()
    assert "canonical reproducibility/evaluation profile" in text
    assert "Not the production runtime environment" in text
    assert "exactly the production runtime" not in text.lower()
    for label in ("**Shipped / default behaviour.**", "**Canonical reproducibility profile.**",
                  "**Lite profile.**", "**Stability / unseeded behaviour.**"):
        assert label in text


def test_risk_suites_are_labelled_as_diagnostic_probes():
    text = _render()
    assert text.count("constructed risk suite / diagnostic probe") >= 4
    assert "They are not representative benchmarks" in text


def test_script_pilot_prints_pilot_and_counts_only():
    text = _render()
    section = text[text.index("### 5.3 Script pilot"):text.index("### 5.4")]
    assert "PILOT" in section and "6 of 6" in section
    assert "%" not in section, "the pilot must not print a rate"
    assert "CI" not in section


def test_advbench_is_labelled_a_case_study_and_kept_out_of_the_headline_table():
    text = _render()
    head = text[text.index("## 3. Headline"):text.index("### Approved-count check")]
    main_table = head[:head.index("Aggregate over the attack sets")]
    assert "AdvBench" not in main_table
    assert "**Case study — excluded from every headline.**" in head


def test_lite_section_states_that_the_missing_classifier_is_not_visible():
    text = _render()
    section = text[text.index("### 5.4 Lite profile"):text.index("## 6. Integrity")]
    assert "**1848 / 1848**" in section
    assert "Is the missing classifier visible on the result? | **NO**" in section


def test_hermetic_claim_is_stated_exactly_and_not_overstated():
    text = _render()
    assert ("zero outbound connection attempts through the Python runtime, with every known "
            "egress path closed at its source") in text
    assert "not a claim of operating-system or native-code network isolation" in text


def test_report_holds_no_timings():
    text = _render()
    for word in (" ms", "latency.json`", "seconds"):
        if word == "latency.json`":
            continue
        assert word not in text.replace("`latency.json`", ""), word


def test_run_notes_are_marked_as_metadata(tmp_path):
    workers = {"p": {"exit_code": 0, "wall_s": 1.5, "timings": {"import_s": 0.3, "warmup_s": 2.0,
                                                               "first_scan_s": 0.02, "worker_wall_s": 5.0},
                     "guard": {"audit_events_seen": 99}, "pickle_classes": [["numpy", "ndarray"]],
                     "blocked_import_hits": ["engine"]}}
    doc, text = report.render_run_notes(tmp_path, _summary(), workers, {"p": []})
    assert text.startswith("# Run notes — run metadata")
    assert "**Not part of the deterministic artifact.**" in text
    assert doc["cold_start"]["p"]["warmup_s"] == 2.0
    assert "`numpy.ndarray`" in text and "no allowlist is enforced" in text
