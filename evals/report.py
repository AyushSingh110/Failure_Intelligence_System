"""
Human-readable output.

REPORT.md is a DETERMINISTIC artifact: it is rendered from summary.json and
fingerprint.json only, so two runs with equal fingerprints produce the same bytes.
It contains no timings and no stability numbers — those are run metadata and go
to RUN_NOTES.md.

Rules the report enforces:

  * exact counts first, percentages derived from them
  * every table names the model version and hash it describes
  * attack performance is never printed without benign utility; a run that
    covers one axis opens with "Single-axis run. Not a security result."
  * risk suites carry the label "constructed risk suite / diagnostic probe"
  * the script pilot prints counts and the word PILOT — never a rate
  * the profile is called the canonical reproducibility/evaluation profile,
    never the production runtime
"""
from __future__ import annotations

from pathlib import Path

from evals import latency

RISK_LABEL = "constructed risk suite / diagnostic probe"


def pct(x) -> str:
    return "—" if x is None else f"{100.0 * x:.1f}%"


def ci(pair) -> str:
    return "—" if not pair else f"[{100.0 * pair[0]:.1f}, {100.0 * pair[1]:.1f}]"


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _subject_line(summary: dict, fp: dict) -> str:
    prim = summary["profiles"].get(summary["primary_profile"], {})
    ident = fp["identity"]
    version = prim.get("pair_declared_version") or "no classifier"
    sha = prim.get("pair_sha256_8") or "—"
    commit = (ident["git"]["commit"] or "unknown")[:7]
    return (f"PAIR {version} (`{sha}`) · fie {ident['subject']['fie_version']} · "
            f"source tree `{ident['subject']['fie_tree_sha256'][:8]}` · commit `{commit}`")


def _rate_row(axis_label: str, name: str, brief: dict) -> list:
    rate = brief.get("rate") or {}
    return [axis_label, name, f"**{brief['flagged']} / {brief['n']}**", pct(rate.get("point")),
            ci(rate.get("bootstrap95")), ci(rate.get("wilson95"))]


def render(summary: dict, fp: dict, profiles_reg: dict, suites_reg: dict) -> str:
    L: list[str] = []
    add = L.append
    primary = summary["primary_profile"]
    prim = summary["profiles"].get(primary, {})
    ident = fp["identity"]
    cfg = fp["configuration"].get(primary, {})
    display = dict(summary.get("dataset_display", {}))

    add("# FIE evaluation report")
    add("")
    if summary.get("single_axis_notice"):
        add("> **Single-axis run. Not a security result.**")
        add("> " + summary["single_axis_notice"])
        add("")
    add(table(["", ""], [
        ["Subject", _subject_line(summary, fp)],
        ["Profile", f"`{primary}` — {prim.get('title', '')}"],
        ["Status", "**CANONICAL RUN**" if summary["canonical"]
         else "**NOT CANONICAL** — may be inspected; cannot be pinned or used as a baseline"],
        ["Harness", f"evals {ident['harness']['version']}, schema {ident['harness']['schema_version']}, "
                    f"source `{ident['harness']['tree_sha256'][:8]}`"],
    ]))
    add("")
    add(f"The `{primary}` profile is a **{prim.get('title', '')}**. {prim.get('summary', '')}")
    add("")

    # 1 ------------------------------------------------------------------------
    add("## 1. Four behaviours this report keeps apart")
    add("")
    apart = profiles_reg["behaviours_to_keep_apart"]
    add(table(["", "Behaviour", "Where its numbers are"], [
        ["A", "**Shipped / default behaviour.** " + apart["A_shipped_default"],
         "Not measured directly by a deterministic suite. Sections 5.4 and `RUN_NOTES.md` show where it differs"],
        ["B", "**Canonical reproducibility profile.** " + apart["B_canonical_reproducibility_profile"],
         "Sections 3, 4, 5.1–5.3"],
        ["C", "**Lite profile.** " + apart["C_lite_profile"], "Section 5.4"],
        ["D", "**Stability / unseeded behaviour.** " + apart["D_stability_unseeded"],
         "`RUN_NOTES.md` (run metadata: it differs from run to run by design)"],
    ]))
    add("")

    # 2 ------------------------------------------------------------------------
    add("## 2. What was evaluated")
    add("")
    add("### 2.1 Subject")
    add("")
    rows = [["fie version", ident["subject"]["fie_version"]],
            ["fie source tree (SHA-256 over `fie/**/*.py`)", f"`{ident['subject']['fie_tree_sha256']}`"],
            ["git commit", f"`{ident['git']['commit']}`"],
            ["measured paths clean", "yes" if ident["git"]["dirty_subject"] is False else
             ("NO" if ident["git"]["dirty_subject"] else "unknown")],
            ["other uncommitted changes in the tree", "yes" if ident["git"]["dirty_other"] else "no"],
            ["model manifest", f"`{ident['model_manifest']['path']}`, release `{ident['model_manifest']['release_tag']}`"]]
    add(table(["Item", "Value"], rows))
    add("")
    for profile_id, models in ident["models"].items():
        if not models:
            add(f"Profile `{profile_id}` loads **no model files**.")
            add("")
            continue
        add(f"Model files verified for profile `{profile_id}` (hash checked before `fie` was imported, "
            f"again after the run):")
        add("")
        add(table(["Role", "File", "SHA-256", "In manifest"],
                  [[role, f"`{m['file']}`", f"`{m['sha256']}`", "yes" if m["in_manifest"] else "**NO**"]
                   for role, m in sorted(models.items())]))
        add("")

    add("### 2.2 Profile, and how it differs from an unmodified user environment")
    add("")
    add("Applied at run time inside the worker process. No production file is edited.")
    add("")
    add(table(["#", "Deviation", "Why", "Changes what is measured?"],
              [[d["id"], d["what"], d["why"], d["changes_what_is_measured"]]
               for d in profiles_reg["deviations"]]))
    add("")

    subj = cfg.get("subject", {})
    if subj:
        add("### 2.3 Thresholds and detector configuration")
        add("")
        add(table(["Setting", "Value"], [
            ["PAIR classifier loaded", "yes" if subj["pair"]["loaded"] else "**no**"],
            ["PAIR threshold", subj["pair"]["threshold"]],
            ["Meta-classifier loaded / threshold", f"{'yes' if subj['meta']['loaded'] else 'no'} / {subj['meta']['threshold']}"],
            ["Tiebreaker", subj["tiebreaker"]],
            ["Translation", cfg.get("translation")],
            ["Language-detector seed", cfg.get("langdetect_seed")],
            ["Operator threshold overrides", subj["operator_overrides"] or "none"],
            ["Fallback scan threshold", subj["thresholds"]["scan"]],
            ["Framing dampening factor", subj["framing_dampen_factor"]],
            ["Scan arguments", ", ".join(f"{k}={v}" for k, v in sorted(subj["scan_args"].items()))],
        ]))
        add("")
        add("Per-attack-type thresholds (the uncertain band is `[0.60 × T, T)`): "
            + ", ".join(f"{k} {v}" for k, v in sorted(subj["thresholds"]["attack"].items())) + ".")
        add("")

    add("### 2.4 Datasets")
    add("")
    add(table(["Dataset", "Rows", "Unique", "Label", "Content SHA-256"],
              [[display.get(k, k), v["rows"], v["unique"], v["label"], f"`{v['content_sha256'][:16]}…`"]
               for k, v in sorted(ident["datasets"].items())]))
    add("")
    add("Content hashes are over canonically serialized rows and do not depend on line endings. "
        "Row order is file order. Nothing is deduplicated.")
    add("")

    # 3 ------------------------------------------------------------------------
    head = summary.get("headline")
    add("## 3. Headline — attack performance and benign utility")
    add("")
    add(f"Subject: {_subject_line(summary, fp)}. Profile: `{primary}`.")
    add("")
    if head is None:
        add("**Single-axis run. Not a security result.** One of the two axes was not measured, so no "
            "headline is reported. Per-suite counts follow.")
        add("")
        rows = []
        for sid, suite in summary["suites"].items():
            for c in suite["cells"]:
                rows.append([sid, display.get(c["dataset"], c["dataset"]), c["variant"], c["axis"],
                             f"{c['flagged']} / {c['n']}"])
        add(table(["Suite", "Dataset", "Variant", "Axis", "Flagged / n"], rows))
        add("")
    else:
        if not head["complete"]:
            add("> This run covers only part of the headline set. The table shows what was measured.")
            add("")
        rows = []
        for d in head["attack_sets"]:
            rows.append(_rate_row("Attack recall", display.get(d, d), head["attack"][d]))
        for d in head["benign_sets"]:
            rows.append(_rate_row("Benign over-refusal", display.get(d, d), head["benign"][d]))
        add(table(["Axis", "Set", "Flagged / n", "Rate", "95% CI (bootstrap)", "95% CI (Wilson)"], rows))
        add("")
        macro, micro, clear = head["macro_recall"], head["micro_recall"], head["clear_block_only_macro_recall"]
        add(table(["Aggregate over the attack sets", "Value", "95% CI (bootstrap)"], [
            [f"Macro recall (unweighted mean of {macro['sets']} sets)", f"**{100 * macro['point']:.2f}%**",
             ci(macro["bootstrap95"])],
            [f"Micro recall (pooled, {micro['flagged']} / {micro['n']})", pct(micro["point"]),
             ci(micro["bootstrap95"])],
            ["Macro recall counting clear blocks only", pct(clear["point"]), ci(clear["bootstrap95"])],
        ]))
        add("")
        add("Read the two axes together. Attack recall says nothing by itself: a guard that blocks "
            "everything scores 100%. \"Clear blocks only\" is the recall left if every uncertain-band "
            "block were let through — the lower bound for any deployment that does not block that band.")
        add("")
        if head.get("contrast"):
            rows = [_rate_row("Attack recall (contrast set)", display.get(d, d), b)
                    for d, b in head["contrast"].items()]
            add(table(["Axis", "Set", "Flagged / n", "Rate", "95% CI (bootstrap)", "95% CI (Wilson)"], rows))
            add("")
        pair = head.get("xstest_matched_pair")
        if pair:
            add(f"XSTest matched pair: TP {pair['tp']}, FP {pair['fp']}, FN {pair['fn']}, TN {pair['tn']} — "
                f"precision {pct(pair['precision'])}, recall {pct(pair['recall'])}, F1 {pct(pair['f1'])}. "
                + pair["note"])
            add("")
        if head.get("case_study"):
            add("**Case study — excluded from every headline.** " + head["case_study_note"])
            add("")
            rows = [_rate_row("Attack recall (case study)", display.get(d, d), b)
                    for d, b in head["case_study"].items()]
            add(table(["Axis", "Set", "Flagged / n", "Rate", "95% CI (bootstrap)", "95% CI (Wilson)"], rows))
            add("")

    approved = summary["approved_counts"]
    if approved["datasets"]:
        add("### Approved-count check")
        add("")
        add(f"Against the counts approved for profile `{approved['profile']}` ({approved['source']}).")
        add("")
        add(table(["Dataset", "Flagged / n", "Approved", "Match"],
                  [[display.get(d, d), f"{v['flagged']} / {v['n']}", v["expected_flagged"],
                    "yes" if v["match"] else "**NO**"] for d, v in approved["datasets"].items()]))
        add("")

    # 4 ------------------------------------------------------------------------
    if summary["zones"]:
        add("## 4. Routing zones")
        add("")
        add("How each set divides between the three zones inside `scan_prompt`. The product collapses "
            "them into one boolean; `uncertain_block` is the band a REVIEW state would expose.")
        add("")
        add(table(["Set", "n", "allow", "uncertain block", "clear block"],
                  [[display.get(d, d), z["n"],
                    f"{z['allow']} ({pct(z['allow'] / z['n'])})",
                    f"{z['uncertain_block']} ({pct(z['uncertain_block'] / z['n'])})",
                    f"{z['clear_block']} ({pct(z['clear_block'] / z['n'])})"]
                   for d, z in summary["zones"].items()]))
        add("")

    # 5 ------------------------------------------------------------------------
    risk = summary.get("risk") or {}
    if risk:
        add("## 5. Constructed risk suites / diagnostic probes")
        add("")
        add(f"Everything in this section is a **{RISK_LABEL}**. These suites record how the system "
            "behaves on inputs built for the purpose. They are not representative benchmarks and "
            "support no claim about any population of real prompts.")
        add("")
    if "long_input" in risk:
        li = risk["long_input"]
        add(f"### 5.1 Long input — {RISK_LABEL}")
        add("")
        add("What it is: one fixed benign filler sentence added before or after a fixed, position-chosen "
            "sample of attack prompts. What it is not: an estimate of behaviour on long real prompts, "
            "other fillers, or text mixed into the attack.")
        if li.get("sample"):
            add("")
            add(li["sample"])
        add("")
        add(table(["Variant (words of filler)", "Flagged / n", "Recall", "95% CI (bootstrap)",
                   "Of those caught unpadded, still caught", "Layers when caught"],
                  [[r["variant"], f"{r['flagged']} / {r['n']}", pct(r["rate"]["point"]),
                    ci(r["rate"]["bootstrap95"]), f"{r['base_caught_still_caught']} / {r['base_caught']}",
                    ", ".join(f"{a} ×{b}" for a, b in r["layers_when_caught"]) or "—"]
                   for r in li["attack"]]))
        add("")
        add("Benign axis, same manipulation:")
        add("")
        add(table(["Dataset", "Variant", "Flagged / n", "Over-refusal", "95% CI (bootstrap)",
                   "Same rows, unpadded"],
                  [[display.get(r["dataset"], r["dataset"]), r["variant"], f"{r['flagged']} / {r['n']}",
                    pct(r["rate"]["point"]), ci(r["rate"]["bootstrap95"]),
                    "—" if r["flagged_unpadded_same_rows"] is None else f"{r['flagged_unpadded_same_rows']} / {r['n']}"]
                   for r in li["benign"]]))
        add("")
    if "framing" in risk:
        fr = risk["framing"]
        add(f"### 5.2 Framing — {RISK_LABEL}")
        add("")
        add("What it is: four fixed, hand-written templates applied to every attack prompt. What it is "
            "not: an adaptive attack, or an estimate of how often framing works in general. It measures "
            "these four strings.")
        add("")
        add(table(["Template", "Flagged / n", "Recall", "95% CI (bootstrap)", "Unframed flagged",
                   "Caught unframed, missed framed", "Missed unframed, caught framed"],
                  [[r["variant"], f"{r['flagged']} / {r['n']}", pct(r["rate"]["point"]),
                    ci(r["rate"]["bootstrap95"]), _n(r["unframed_flagged"]),
                    _n(r["base_caught_now_missed"]), _n(r["base_missed_now_caught"])]
                   for r in fr["attack"]]))
        add("")
        add("Benign axis, same templates:")
        add("")
        add(table(["Dataset", "Template", "Flagged / n", "Over-refusal", "95% CI (bootstrap)", "Unframed flagged"],
                  [[display.get(r["dataset"], r["dataset"]), r["variant"], f"{r['flagged']} / {r['n']}",
                    pct(r["rate"]["point"]), ci(r["rate"]["bootstrap95"]), _n(r["unframed_flagged"])]
                   for r in fr["benign"]]))
        add("")
    if "script_pilot" in risk:
        sp = risk["script_pilot"]
        add("### 5.3 Script pilot — PILOT")
        add("")
        add("**PILOT.** " + sp["note"])
        add("")
        add("`base` is the canonical reproducibility profile, in which translation is unavailable. "
            "`base+translation_stub` replaces the translator with one fixed benign English sentence: "
            "a counterfactual, not a measurement of any translation service.")
        add("")
        add(table(["Pass", "Group (benign prompts)", "Flagged (count)", "Zones", "Layers that fired"],
                  [[r["variant"], r["group"], f"{r['flagged']} of {r['n']}",
                    ", ".join(f"{k} {v}" for k, v in sorted(r["zones"].items())),
                    ", ".join(f"{a} ×{b}" for a, b in r["layers"]) or "—"] for r in sp["rows"]]))
        add("")
    if "lite" in risk:
        lt = risk["lite"]
        add(f"### 5.4 Lite profile — {RISK_LABEL}")
        add("")
        add("What it is: the working tree's code with the ML packages unimportable, which is the code "
            "path of a base `pip install`. What it is not: a measurement of the wheel published on PyPI.")
        add("")
        add(table(["Self-report check", "Value"], [
            ["Classifier loaded", "yes" if lt["classifier_loaded"] else "**no**"],
            ["Loader's reason", lt["classifier_load_error"] or "—"],
            ["Scans", lt["scans"]],
            ["Scans whose result reported full coverage (`degraded_layers == []`)",
             f"**{lt['scans_reporting_full_coverage']} / {lt['scans']}** ({pct(lt['share_reporting_full_coverage'])})"],
            ["Scans where the classifier scored 0.0 with status ok",
             f"{lt['scans_where_classifier_scored_zero_with_status_ok']} / {lt['scans']}"],
            ["Is the missing classifier visible on the result?", "yes" if lt["honest"] else "**NO**"],
        ]))
        add("")
        add(lt["reading"])
        add("")
        rows = []
        canon = {}
        for d in lt["by_dataset"]:
            for suite in summary["suites"].values():
                if suite["group"] == "standard":
                    for c in suite["cells"]:
                        if c["dataset"] == d and c["variant"] == "base":
                            canon[d] = c
        for d, b in lt["by_dataset"].items():
            c = canon.get(d)
            rows.append([display.get(d, d), "attack" if "tp" in b["confusion"] else "benign",
                         f"{b['flagged']} / {b['n']}", pct((b.get("rate") or {}).get("point")),
                         f"{c['flagged']} / {c['n']}" if c else "—"])
        add(table(["Dataset", "Axis", "Lite: flagged / n", "Lite: rate",
                   "Canonical reproducibility profile: flagged / n"], rows))
        add("")

    # 6 ------------------------------------------------------------------------
    integ = summary["integrity"]
    add("## 6. Integrity")
    add("")
    rows = [["Datasets verified by content hash", integ["datasets_verified"]],
            ["Scan errors", integ["scan_errors"]],
            ["Records with degraded layers", integ["degraded_records"]]]
    for p, n in integ["models_verified"].items():
        rows.append([f"Model files verified — `{p}`", n])
    for p, h in integ["hermetic"].items():
        rows.append([f"Hermetic proof — `{p}`",
                     ("ok" if h["guard_ok"] else "**FAILED**")
                     + f": {h['non_canary_events']} non-canary event(s), {h['canaries_denied']} canaries denied, "
                       f"audit hook ran: {'yes' if h['audit_hook_ran'] else 'NO'}, "
                       f"models re-verified after run: {'yes' if h['models_reverified_after_run'] else 'NO'}, "
                       f"pre-arm warm-ups: {', '.join(h['prearm'] or []) or 'none'}"])
    add(table(["Check", "Result"], rows))
    add("")
    add("The hermetic claim, stated exactly: **zero outbound connection attempts through the Python "
        "runtime, with every known egress path closed at its source.** It is not a claim of "
        "operating-system or native-code network isolation.")
    add("")

    # 7 ------------------------------------------------------------------------
    add("## 7. Comparability")
    add("")
    keys = summary["keys"]
    add(table(["Key", "Value", "If it differs between two runs"], [
        ["dataset_key", f"`{keys['dataset_key'][:16]}…`", "incomparable"],
        ["config_key", f"`{keys['config_key'][:16]}…`", "comparable only as a declared configuration change"],
        ["subject_key", f"`{keys['subject_key'][:16]}…`", "expected: this is the change being measured"],
        ["env_key", f"`{keys['env_key'][:16]}…`", "verdicts comparable with a warning; latency is not"],
    ]))
    add("")
    add("Deterministic artifacts are byte-identical only when all four keys match.")
    add("")

    # 8 ------------------------------------------------------------------------
    add("## 8. What changed")
    add("")
    comp = summary["comparison"]
    if comp["baseline_id"] is None:
        add("No baseline was selected for this run; baseline-versus-candidate comparison is not part of WP-001.")
        if approved["datasets"]:
            add("")
            add("Against the approved counts for this profile: "
                + ("**all match.**" if approved["all_match"] else "**MISMATCH — see section 3.**"))
    add("")
    add("## 9. Not in this file")
    add("")
    add("Latency, per-suite run times and the stability (unseeded) results are run metadata. They "
        "differ from run to run and live in `RUN_NOTES.md`, `latency.json` and `known_unstable.json`, "
        "outside the byte-compared set.")
    add("")
    return "\n".join(L)


def _n(value) -> str:
    return "—" if value is None else str(value)


# ── run notes (metadata) ─────────────────────────────────────────────────────

def render_run_notes(run_dir: Path, summary: dict, workers_meta: dict, by_profile: dict):
    """
    RUN_NOTES.md and its data. Run metadata: not deterministic, not byte-compared.
    Returns (document, markdown) or None when there is nothing to report.
    """
    doc: dict = {"cold_start": {}, "per_suite": {}, "warm_latency": None, "stability": None,
                 "workers": {}}
    for profile_id, w in workers_meta.items():
        t = w.get("timings") or {}
        doc["cold_start"][profile_id] = {k: t.get(k) for k in
                                         ("import_s", "warmup_s", "first_scan_s", "worker_wall_s")}
        doc["workers"][profile_id] = {
            "exit_code": w.get("exit_code"), "wall_s": w.get("wall_s"),
            "audit_events_seen": (w.get("guard") or {}).get("audit_events_seen"),
            "pickle_classes_observed": w.get("pickle_classes", []),
            "blocked_import_hits": w.get("blocked_import_hits", []),
        }
    for profile_id, suites in by_profile.items():
        for suite in suites:
            if suite["kind"] == "scan":
                t = latency.suite_timing(run_dir, suite["id"])
                if t:
                    doc["per_suite"][suite["id"]] = dict(t, profile=profile_id)
            elif suite["kind"] == "latency":
                doc["warm_latency"] = latency.warm_latency(run_dir, suite["id"])
            elif suite["kind"] == "stability":
                doc["stability"] = latency.stability(run_dir, suite["id"])

    L: list[str] = []
    add = L.append
    add("# Run notes — run metadata")
    add("")
    add("**Not part of the deterministic artifact.** Everything here depends on the machine and the "
        "moment, or is non-deterministic by design. It is not byte-compared and not used to decide "
        "whether two runs agree.")
    add("")
    add("## Cold start")
    add("")
    add(table(["Profile", "import fie (s)", "warm-up (s)", "first scan (s)", "worker wall time (s)"],
              [[p, _n(v["import_s"]), _n(v["warmup_s"]), _n(v["first_scan_s"]), _n(v["worker_wall_s"])]
               for p, v in doc["cold_start"].items()]))
    add("")
    if doc["warm_latency"]:
        wl = doc["warm_latency"]
        add("## Warm latency (latency suite)")
        add("")
        add(f"Protocol: {wl['protocol']}. Passes measured: {wl['overall']['passes']}.")
        add("")
        o = wl["overall"]
        add(table(["Prompts", "mean ms", "p50 ms", "p95 ms", "p99 ms", "max ms"],
                  [[o["n"], o["mean_ms"], o["p50_ms"], o["p95_ms"], o["p99_ms"], o["max_ms"]]]))
        add("")
        add("By input (length buckets):")
        add("")
        add(table(["Dataset / variant", "Prompts", "mean chars", "mean ms", "p50 ms", "p95 ms"],
                  [[k, v["n"], v["mean_chars"], v["mean_ms"], v["p50_ms"], v["p95_ms"]]
                   for k, v in wl["by_bucket"].items()]))
        add("")
    if doc["per_suite"]:
        add("## Per-suite scan time and throughput")
        add("")
        add("Timed while the deterministic records were produced (cache not cleared between prompts "
            "of a suite, so an exact duplicate prompt returns from cache).")
        add("")
        add(table(["Suite", "Profile", "Scans", "mean chars", "scan time (s)", "scans / s", "mean ms", "p95 ms"],
                  [[k, v["profile"], v["n"], v["mean_chars"], v["scan_time_s"], v["scans_per_s"],
                    v["mean_ms"], v["p95_ms"]] for k, v in doc["per_suite"].items()]))
        add("")
    if doc["stability"]:
        add("## D. Stability — the shipped behaviour, language detector unseeded")
        add("")
        add("The canonical reproducibility profile fixes the language-detector seed. This suite removes "
            "it, which is how the product ships, and repeats identical passes. A prompt is unstable when "
            "its verdict differs between passes. More passes find more: a prompt that flips rarely can "
            "be missed.")
        add("")
        add(table(["Dataset", "n", "Passes", "Flagged per pass", "Unstable prompts", "Row indexes"],
                  [[d, v["n"], v["passes"], ", ".join(map(str, v["flagged_per_pass"])),
                    len(v["unstable_idx"]), ", ".join(map(str, v["unstable_idx"])) or "—"]
                   for d, v in doc["stability"].items()]))
        add("")
    add("## Guard accounting")
    add("")
    add(table(["Profile", "Worker exit", "Audit events seen by the hook", "Classes resolved while unpickling",
               "Blocked imports attempted"],
              [[p, v["exit_code"], v["audit_events_seen"], len(v["pickle_classes_observed"]),
                ", ".join(v["blocked_import_hits"]) or "—"] for p, v in doc["workers"].items()]))
    add("")
    classes = sorted({f"{m}.{n}" for v in doc["workers"].values() for m, n in v["pickle_classes_observed"]})
    if classes:
        add("Classes resolved by unpickling (recorded only; no allowlist is enforced — decision OD-6):")
        add("")
        add(", ".join(f"`{c}`" for c in classes) + ".")
        add("")
    return doc, "\n".join(L)
