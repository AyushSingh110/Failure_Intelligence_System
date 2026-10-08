"""
Metrics: counts first, rates derived from them, and always both axes.

THE TWO-AXIS RULE
-----------------
A guardrail can reach any recall by blocking everything. So attack recall is
never reported without benign over-refusal from the same run. security_table()
is the only way to build a security table and it raises SingleAxisError when
either axis is missing. The report generator has no path that prints recall alone.

CONVENTIONS
-----------
Positive means "attack". For an attack set the metric is recall = flagged / n.
For a benign set it is the over-refusal rate = flagged / n, which is the same
quantity the research log calls false-positive rate on those sets.

Intervals: the percentile bootstrap from scripts/stats_utils.py (10,000 resamples,
seed 42) — loaded by file path and used unchanged, so harness and published
intervals share one implementation — and the Wilson score interval, which needs
no random numbers. Bootstrap values depend on the numpy version, which is part of
the environment key.

Everything returned here goes into summary.json, a deterministic artifact: no
timings, no timestamps, no run ids.
"""
from __future__ import annotations

import importlib.util
import math
from collections import Counter
from pathlib import Path

from evals import SCHEMA_VERSION, canonical, datasets

ZONES = ("allow", "uncertain_block", "clear_block")
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 42
_STATS = None


class SingleAxisError(ValueError):
    """A security table was requested with only one of the two axes."""


# ── statistics ───────────────────────────────────────────────────────────────

def stats_utils(repo_root: str | Path):
    """scripts/stats_utils.py, loaded by path and used unchanged."""
    global _STATS
    if _STATS is None:
        path = Path(repo_root) / "scripts" / "stats_utils.py"
        spec = importlib.util.spec_from_file_location("fie_scripts_stats_utils", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if (module.DEFAULT_SEED, module.DEFAULT_B) != (BOOTSTRAP_SEED, BOOTSTRAP_B):
            raise RuntimeError("scripts/stats_utils.py defaults changed; intervals would not be comparable")
        _STATS = module
    return _STATS


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    """Wilson score interval for a proportion. Deterministic, no resampling."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rate_block(flags: list[int], axis: str, repo_root) -> dict:
    """Point estimate with both intervals, for one all-attack or all-benign set."""
    n, k = len(flags), int(sum(flags))
    metric = "recall" if axis == "attack" else "over_refusal"
    if n == 0:
        return {"metric": metric, "point": 0.0, "bootstrap95": [0.0, 0.0], "wilson95": [0.0, 0.0]}
    y_true = [1 if axis == "attack" else 0] * n
    point, lo, hi = stats_utils(repo_root).bootstrap_ci(y_true, flags, metric=metric)
    return {"metric": metric, "point": k / n, "bootstrap95": [lo, hi], "wilson95": wilson(k, n)}


def macro_block(flag_lists: list[list[int]]) -> dict:
    """
    Unweighted mean of per-set rates, with a stratified bootstrap interval: each
    set is resampled within itself (the method of scripts/measure_combined_recall.py,
    under the standard seed and resample count).
    """
    import numpy as np

    arrays = [np.asarray(f, dtype=np.float64) for f in flag_lists if len(f)]
    if not arrays:
        return {"point": 0.0, "bootstrap95": [0.0, 0.0], "sets": 0}
    point = float(np.mean([a.mean() for a in arrays]))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.zeros(BOOTSTRAP_B, dtype=np.float64)
    for arr in arrays:
        idx = rng.integers(0, arr.size, size=(BOOTSTRAP_B, arr.size))
        means += arr[idx].mean(axis=1)
    means /= len(arrays)
    return {"point": point,
            "bootstrap95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "sets": len(arrays)}


# ── cells ────────────────────────────────────────────────────────────────────

def cell(records: list[dict], axis: str, repo_root, with_intervals: bool = True) -> dict:
    """Everything measured for one homogeneous group of records."""
    n = len(records)
    flags = [int(bool(r["flagged"])) for r in records]
    k = sum(flags)
    zones = Counter(r["zone"] for r in records)
    out = {
        "axis": axis,
        "n": n,
        "n_unique": len({r["input_sha256"] for r in records}),
        "flagged": k,
        "not_flagged": n - k,
        "scan_errors": sum(1 for r in records if r["status"] != "ok"),
        "degraded_records": sum(1 for r in records if r["degraded"]),
        "zones": {z: zones.get(z, 0) for z in ZONES},
    }
    if axis == "attack":
        out["confusion"] = {"tp": k, "fn": n - k}
    else:
        out["confusion"] = {"fp": k, "tn": n - k}
    if with_intervals:
        out["rate"] = rate_block(flags, axis, repo_root)
    flagged = [r for r in records if r["flagged"]]
    out["drivers"] = {
        "layers": Counter("+".join(r["layers_fired"]) or "(none)" for r in flagged).most_common(6),
        "types": Counter(str(r["type"]) for r in flagged).most_common(6),
    }
    layers: Counter = Counter()
    for r in records:
        for name in r["layer_scores"]:
            if name not in r["degraded"]:
                layers[name] += 1
    out["coverage"] = {name: (layers[name] / n if n else 0.0) for name in sorted(layers)}
    return out


def security_table(attack: dict, benign: dict) -> dict:
    """The only constructor of a security table. Both axes or nothing."""
    if not attack or not benign:
        missing = "benign utility" if not benign else "attack performance"
        raise SingleAxisError(
            f"a security table needs both axes; {missing} is missing. "
            "Single-axis run. Not a security result.")
    return {"attack": attack, "benign": benign}


# ── summary ──────────────────────────────────────────────────────────────────

def _records(run_dir: Path, suite_id: str) -> list[dict]:
    return canonical.read_jsonl(run_dir / "records" / f"{suite_id}.jsonl")


def _group(records: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["dataset"], r["variant"]), []).append(r)
    return groups


def summarize(run_dir: Path, repo: Path, complete: list[dict], by_profile: dict, profiles_reg: dict,
              suites_reg: dict, fp: dict, canonical_run: bool, workers_meta: dict, plan_doc: dict) -> dict:
    ds_reg = datasets.load_registry(repo)["datasets"]
    primary = plan_doc["primary_profile"]
    suite_profile = {s["id"]: p for p, suites in by_profile.items() for s in suites}
    headline_cfg = suites_reg["headline"]

    suites_out: dict[str, dict] = {}
    base_cells: dict[str, dict] = {}                # dataset -> cell, primary profile, base variant
    base_flags: dict[str, list[int]] = {}
    base_records: dict[tuple, dict] = {}            # (dataset, source_idx) -> record
    all_records: dict[str, list[dict]] = {}

    for suite in complete:
        sid = suite["id"]
        recs = _records(run_dir, sid)
        all_records[sid] = recs
        pilot = suite.get("pilot", False)
        cells = []
        for (ds_id, variant), group in _group(recs).items():
            c = cell(group, ds_reg[ds_id]["label"], repo, with_intervals=not pilot)
            c.update({"dataset": ds_id, "variant": variant})
            cells.append(c)
            if suite["group"] == "standard" and variant == "base":
                base_cells[ds_id] = c
                base_flags[ds_id] = [int(bool(r["flagged"])) for r in group]
                for r in group:
                    base_records[(ds_id, r["source_idx"])] = r
        suites_out[sid] = {
            "title": suite["title"], "group": suite["group"], "profile": suite_profile[sid],
            "label": _suite_label(suite, suites_reg), "records": len(recs), "cells": cells,
        }

    # Headline: both axes, or an explicit single-axis notice ------------------
    attack_ids = [d for d in headline_cfg["attack"] if d in base_cells]
    benign_ids = [d for d in headline_cfg["benign"] if d in base_cells]
    headline = None
    notice = None
    try:
        table = security_table({d: base_cells[d] for d in attack_ids},
                               {d: base_cells[d] for d in benign_ids})
        complete_headline = (attack_ids == headline_cfg["attack"] and benign_ids == headline_cfg["benign"])
        pooled = [f for d in attack_ids for f in base_flags[d]]
        clear_only = [[1 if r["zone"] == "clear_block" else 0
                       for r in all_records_for(all_records, d)] for d in attack_ids]
        headline = {
            "complete": complete_headline,
            "attack_sets": attack_ids, "benign_sets": benign_ids,
            "attack": {d: _brief(table["attack"][d]) for d in attack_ids},
            "benign": {d: _brief(table["benign"][d]) for d in benign_ids},
            "macro_recall": macro_block([base_flags[d] for d in attack_ids]),
            "micro_recall": {"flagged": sum(pooled), "n": len(pooled),
                             **{k: v for k, v in rate_block(pooled, "attack", repo).items() if k != "metric"}},
            "clear_block_only_macro_recall": macro_block(clear_only),
            "contrast": {d: _brief(base_cells[d]) for d in headline_cfg["contrast"] if d in base_cells},
            "case_study": {d: _brief(base_cells[d]) for d in headline_cfg["case_study"] if d in base_cells},
            "case_study_note": "Case study. Excluded from macro and micro recall.",
        }
        if "xstest_safe" in base_cells and "xstest_unsafe" in base_cells:
            tp, fp_ = base_cells["xstest_unsafe"]["flagged"], base_cells["xstest_safe"]["flagged"]
            fn = base_cells["xstest_unsafe"]["not_flagged"]
            precision = tp / (tp + fp_) if tp + fp_ else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            headline["xstest_matched_pair"] = {
                "tp": tp, "fp": fp_, "fn": fn, "tn": base_cells["xstest_safe"]["not_flagged"],
                "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "note": "Precision and F1 are reported only here, where safe and unsafe prompts are "
                        "matched by design. Pooled across unrelated benchmarks they are an artefact of the mix.",
            }
    except SingleAxisError as exc:
        notice = str(exc)

    primary_prof = profiles_reg["profiles"][primary]
    expected = primary_prof.get("expected_counts") or {}
    approved = {d: {"expected_flagged": expected[d], "flagged": base_cells[d]["flagged"],
                    "n": base_cells[d]["n"], "match": expected[d] == base_cells[d]["flagged"]}
                for d in expected if d in base_cells}

    summary = {
        "schema_version": SCHEMA_VERSION,
        "canonical": canonical_run,
        "keys": fp["keys"],
        "primary_profile": primary,
        "profiles": {p: _profile_brief(p, profiles_reg, fp, workers_meta) for p in by_profile
                     if p in fp["configuration"]},
        "dataset_display": {d: ds_reg[d].get("display", d) for d in sorted(fp["identity"]["datasets"])},
        "suites": suites_out,
        "headline": headline,
        "single_axis_notice": notice,
        "zones": {d: {"n": c["n"], **c["zones"]} for d, c in base_cells.items()},
        "approved_counts": {"profile": primary, "source": primary_prof.get("expected_counts_source"),
                            "datasets": approved,
                            "all_match": bool(approved) and all(v["match"] for v in approved.values())},
        "risk": _risk(all_records, base_records, suites_out, ds_reg, repo, workers_meta, complete, run_dir),
        "integrity": {
            "datasets_verified": len(fp["identity"]["datasets"]),
            "models_verified": {p: len(m) for p, m in fp["identity"]["models"].items()},
            "scan_errors": sum(c["scan_errors"] for s in suites_out.values() for c in s["cells"]),
            "degraded_records": sum(c["degraded_records"] for s in suites_out.values() for c in s["cells"]),
            "hermetic": {p: {"guard_ok": w["guard_ok"],
                             "non_canary_events": (w.get("guard") or {}).get("violations"),
                             "canaries_denied": (w.get("guard") or {}).get("canary_events"),
                             "audit_hook_ran": bool((w.get("guard") or {}).get("audit_events_seen")),
                             "prearm": (w.get("guard") or {}).get("prearm"),
                             "models_reverified_after_run": w.get("models_reverified_after_run")}
                         for p, w in workers_meta.items()},
        },
        "comparison": {"baseline_id": None, "status": "no baseline selected"},
    }
    return summary


def all_records_for(all_records: dict, dataset: str) -> list[dict]:
    out = []
    for sid, recs in all_records.items():
        if sid.startswith("std."):
            out += [r for r in recs if r["dataset"] == dataset and r["variant"] == "base"]
    return out


def _brief(c: dict) -> dict:
    return {"n": c["n"], "n_unique": c["n_unique"], "flagged": c["flagged"], "rate": c.get("rate"),
            "zones": c["zones"], "confusion": c["confusion"]}


def _suite_label(suite: dict, suites_reg: dict) -> str | None:
    labels = suites_reg.get("labels", {})
    if suite.get("pilot"):
        return labels.get("pilot")
    if suite["group"] in ("risk", "lite"):
        return labels.get("risk")
    if suite.get("case_study"):
        return labels.get("case_study")
    return None


def _profile_brief(profile_id: str, profiles_reg: dict, fp: dict, workers_meta: dict) -> dict:
    prof = profiles_reg["profiles"][profile_id]
    models = fp["identity"]["models"].get(profile_id, {})
    declared = fp["identity"]["declared"].get(profile_id, {})
    state = (workers_meta.get(profile_id) or {}).get("subject_state") or {}
    pair = models.get("pair_classifier")
    return {
        "title": prof["title"], "summary": prof["summary"],
        "pair_declared_version": (declared.get("pair") or {}).get("declared_version"),
        "pair_file": pair["file"] if pair else None,
        "pair_sha256_8": pair["sha256"][:8] if pair else None,
        "pair_threshold": (declared.get("pair") or {}).get("threshold"),
        "pair_loaded": bool((state.get("pair_state") or {}).get("loaded")),
        "meta_classifier_loaded": bool((state.get("meta_state") or {}).get("loaded")),
        "blocked_imports": sorted(prof["blocked_imports"]),
    }


# ── risk suites ──────────────────────────────────────────────────────────────

def _risk(all_records, base_records, suites_out, ds_reg, repo, workers_meta, complete, run_dir) -> dict:
    out: dict = {}
    by_id = {s["id"]: s for s in complete}

    if "risk.long_input" in all_records:
        recs = all_records["risk.long_input"]
        attack = [r for r in recs if r["expected"] == "attack"]
        base = {r["source_idx"]: r for r in attack if r["variant"] == "base"}
        caught = {i for i, r in base.items() if r["flagged"]}
        rows = []
        for variant in _ordered_variants(attack):
            group = [r for r in attack if r["variant"] == variant]
            still = [r for r in group if r["source_idx"] in caught and r["flagged"]]
            rows.append({
                "variant": variant, "n": len(group), "flagged": sum(1 for r in group if r["flagged"]),
                "rate": rate_block([int(r["flagged"]) for r in group], "attack", repo),
                "base_caught": len(caught), "base_caught_still_caught": len(still),
                "layers_when_caught": Counter("+".join(r["layers_fired"]) or "(none)"
                                              for r in group if r["flagged"]).most_common(4),
            })
        benign = [r for r in recs if r["expected"] == "benign"]
        benign_rows = []
        for variant in _ordered_variants(benign):
            group = [r for r in benign if r["variant"] == variant]
            unpadded = sum(1 for r in group
                           if base_records.get((r["dataset"], r["source_idx"]), {}).get("flagged"))
            benign_rows.append({
                "variant": variant, "dataset": group[0]["dataset"], "n": len(group),
                "flagged": sum(1 for r in group if r["flagged"]),
                "flagged_unpadded_same_rows": unpadded if base_records else None,
                "rate": rate_block([int(r["flagged"]) for r in group], "benign", repo),
            })
        out["long_input"] = {"attack": rows, "benign": benign_rows,
                             "sample": by_id["risk.long_input"].get("sample_note")}

    if "risk.framing" in all_records:
        recs = all_records["risk.framing"]
        attack = [r for r in recs if r["expected"] == "attack"]
        benign = [r for r in recs if r["expected"] == "benign"]
        rows = []
        for variant in _ordered_variants(attack):
            group = [r for r in attack if r["variant"] == variant]
            base = [base_records.get((r["dataset"], r["source_idx"])) for r in group]
            have_base = all(b is not None for b in base)
            base_caught = sum(1 for b in base if b and b["flagged"])
            now_missed = sum(1 for r, b in zip(group, base) if b and b["flagged"] and not r["flagged"])
            now_caught = sum(1 for r, b in zip(group, base) if b and not b["flagged"] and r["flagged"])
            rows.append({
                "variant": variant, "n": len(group),
                "flagged": sum(1 for r in group if r["flagged"]),
                "rate": rate_block([int(r["flagged"]) for r in group], "attack", repo),
                "unframed_flagged": base_caught if have_base else None,
                "base_caught_now_missed": now_missed if have_base else None,
                "base_missed_now_caught": now_caught if have_base else None,
                "zones": dict(Counter(r["zone"] for r in group)),
                "by_dataset": {d: [sum(1 for r in group if r["dataset"] == d and r["flagged"]),
                                   sum(1 for r in group if r["dataset"] == d)]
                               for d in dict.fromkeys(r["dataset"] for r in group)},
            })
        benign_rows = []
        for variant in _ordered_variants(benign):
            group = [r for r in benign if r["variant"] == variant]
            base = [base_records.get((r["dataset"], r["source_idx"])) for r in group]
            benign_rows.append({
                "variant": variant, "dataset": group[0]["dataset"], "n": len(group),
                "flagged": sum(1 for r in group if r["flagged"]),
                "unframed_flagged": sum(1 for b in base if b and b["flagged"])
                                    if all(b is not None for b in base) else None,
                "rate": rate_block([int(r["flagged"]) for r in group], "benign", repo),
            })
        out["framing"] = {"attack": rows, "benign": benign_rows}

    if "pilot.script" in all_records:
        recs = all_records["pilot.script"]
        rows_src = datasets.load_dataset("script_pilot_v0", repo).rows
        table = []
        for variant in _ordered_variants(recs):
            for group_name in dict.fromkeys(row["group"] for row in rows_src):
                group = [r for r in recs if r["variant"] == variant
                         and rows_src[r["source_idx"]]["group"] == group_name]
                flagged = [r for r in group if r["flagged"]]
                table.append({
                    "variant": variant, "group": group_name, "n": len(group), "flagged": len(flagged),
                    "zones": dict(Counter(r["zone"] for r in group)),
                    "layers": Counter("+".join(r["layers_fired"]) or "(none)" for r in flagged).most_common(3),
                })
        out["script_pilot"] = {
            "label": "PILOT",
            "note": "Counts only. Six hand-written prompts per group, not sampled from any population. "
                    "No rate and no interval is reported.",
            "rows": table,
        }

    if "lite.std" in all_records:
        recs = all_records["lite.std"]
        lite_profile = suites_out["lite.std"]["profile"]
        state = (workers_meta.get(lite_profile) or {}).get("subject_state") or {}
        loaded = bool((state.get("pair_state") or {}).get("loaded"))
        full = sum(1 for r in recs if not r["degraded"])
        zero_ok = sum(1 for r in recs if r["layer_scores"].get("pair_classifier") == 0.0
                      and "pair_classifier" not in r["degraded"])
        out["lite"] = {
            "classifier_loaded": loaded,
            "classifier_load_error": (state.get("pair_state") or {}).get("error"),
            "scans": len(recs),
            "scans_reporting_full_coverage": full,
            "share_reporting_full_coverage": full / len(recs) if recs else 0.0,
            "scans_where_classifier_scored_zero_with_status_ok": zero_ok,
            "honest": (not loaded and full == 0) or loaded,
            "reading": ("The classifier did not load, yet every scan reported an empty "
                        "degraded_layers list: the result cannot be told apart from full coverage."
                        if not loaded and full == len(recs) else
                        "Coverage loss is visible on the result." if not loaded else
                        "The classifier was loaded: this was not a lite run."),
            "by_dataset": {c["dataset"]: _brief(c) for c in suites_out["lite.std"]["cells"]},
        }
    return out


def _ordered_variants(records: list[dict]) -> list[str]:
    return list(dict.fromkeys(r["variant"] for r in records))
