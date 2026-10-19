"""
C2 Phase 1.2 — pilot agreement, self-consistency, unclear rates and adjudication workload.

Run ONLY after both reviewers have returned their packets:
  annotations_returned/reviewer_a_returned.csv
  annotations_returned/reviewer_b_returned.csv
(the issued packets, filled in; locked columns unchanged).

  python compute_pilot_agreement.py            # real run; writes results/ (exclusive-create)
  python compute_pilot_agreement.py --self-test  # synthetic in-memory check of the code; writes nothing

Metrics (preregistered in PILOT_PROTOCOL.md section 5):
  * Cohen's kappa on the 200 originals (repeat copies excluded) for H (4 categories, 'unclear' kept)
    and M (3 categories), with seeded bootstrap 95% CIs (2,000 resamples over rows); raw confusion
    matrices; percent agreement; PABAK; prevalence.
  * kappa is marked NOT EVALUABLE for a dimension when, for either reviewer, the least frequent
    non-'unclear' value has fewer than 10 of the 200 rows; then percent agreement (>= 90%) and the
    confusion matrix are reported instead (fixed rule).
  * self-consistency per reviewer on the 20 repeat pairs: identical (M, H) on both copies.
  * unclear rate per reviewer and dimension over the 200 originals.
  * adjudication queue: rows where M or H differ, plus escalation rules.
"""
from __future__ import annotations

import csv
import hashlib
import json
import socket
import sys
from collections import Counter
from pathlib import Path

PILOT = Path(__file__).resolve().parent.parent
socket.socket.connect = socket.create_connection = socket.getaddrinfo = (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))  # type: ignore
sys.dont_write_bytecode = True
BOOT, SEED = 2000, 20261018
H_VALUES = ["harmful", "dual_use", "none", "unclear"]
M_VALUES = ["yes", "no", "unclear"]


def kappa(a, b, cats):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po


def kappa_block(a, b, cats, rng):
    import numpy as np
    k, po = kappa(a, b, cats)
    a, b = np.array(a), np.array(b)
    boots = []
    for _ in range(BOOT):
        ix = rng.integers(0, len(a), len(a))
        kk, _ = kappa(list(a[ix]), list(b[ix]), cats)
        if kk == kk:
            boots.append(kk)
    q = len(cats)
    pabak = (q * po - 1) / (q - 1)
    conf = {f"{x}|{y}": int(((a == x) & (b == y)).sum()) for x in cats for y in cats}
    prev = {"a": dict(Counter(a.tolist())), "b": dict(Counter(b.tolist()))}
    minority = min(min((v for kx, v in Counter(r.tolist()).items() if kx != "unclear"), default=0) for r in (a, b))
    nonunclear_values = [len({x for x in r.tolist() if x != "unclear"}) for r in (a, b)]
    evaluable = minority >= 10 and min(nonunclear_values) >= 2
    return {"kappa": k, "kappa_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
            "percent_agreement": po, "pabak": pabak, "confusion_a_by_b": conf, "prevalence": prev,
            "kappa_evaluable": bool(evaluable), "n": int(len(a))}


def analyse(a_rows, b_rows, mapping):
    import numpy as np
    rng = np.random.default_rng(SEED)
    items = mapping["items"]
    originals = sorted(p for p, v in items.items() if v["repeat_of"] is None)
    A = {r["pilot_id"]: r for r in a_rows}; Bm = {r["pilot_id"]: r for r in b_rows}
    H = kappa_block([A[p]["H_harmful_intent"] for p in originals], [Bm[p]["H_harmful_intent"] for p in originals], H_VALUES, rng)
    M = kappa_block([A[p]["M_manipulation"] for p in originals], [Bm[p]["M_manipulation"] for p in originals], M_VALUES, rng)
    reps = [(v["repeat_of"], p) for p, v in items.items() if v["repeat_of"]]
    selfc = {}
    for name, R in (("a", A), ("b", Bm)):
        same = sum(R[o]["M_manipulation"] == R[c]["M_manipulation"] and R[o]["H_harmful_intent"] == R[c]["H_harmful_intent"] for o, c in reps)
        selfc[name] = {"identical": same, "pairs": len(reps), "rate": same / len(reps)}
    unclear = {name: {"H": sum(R[p]["H_harmful_intent"] == "unclear" for p in originals) / len(originals),
                      "M": sum(R[p]["M_manipulation"] == "unclear" for p in originals) / len(originals)}
               for name, R in (("a", A), ("b", Bm))}
    conflict_ids = {x for pair in mapping["conflict_pairs"] for x in pair["pilot_ids"]}
    queue = []
    for p in originals:
        a, b = A[p], Bm[p]
        reasons = []
        if a["M_manipulation"] != b["M_manipulation"]: reasons.append("M_disagree")
        if a["H_harmful_intent"] != b["H_harmful_intent"]: reasons.append("H_disagree")
        for r in (a, b):
            if "unclear" in (r["M_manipulation"], r["H_harmful_intent"]): reasons.append("unclear")
            if r["H_harmful_intent"] == "dual_use": reasons.append("dual_use")
            if r["Q_confidence"] == "low": reasons.append("low_confidence")
            if r["M_manipulation"] == "yes" and r["H_harmful_intent"] == "harmful": reasons.append("M_and_H")
        if p in conflict_ids: reasons.append("cross_label_conflict")
        if items[p]["source"] == "anthropic_rlhf_redteam" and "none" in (a["H_harmful_intent"], b["H_harmful_intent"]):
            reasons.append("redteam_H_none")
        if items[p]["refusal_cue"]: reasons.append("refusal_confirmation")
        if reasons:
            queue.append({"pilot_id": p, "reasons": sorted(set(reasons))})
    crit = {
        "H": (H["kappa"] >= 0.60) if H["kappa_evaluable"] else (H["percent_agreement"] >= 0.90),
        "M": (M["kappa"] >= 0.60) if M["kappa_evaluable"] else (M["percent_agreement"] >= 0.90),
        "self_consistency": all(v["rate"] >= 0.90 for v in selfc.values()),
        "unclear": all(v["H"] <= 0.15 and v["M"] <= 0.15 for v in unclear.values()),
    }
    return {"H": H, "M": M, "self_consistency": selfc, "unclear_rate": unclear,
            "adjudication_queue_size": len(queue),
            "adjudication_reason_counts": dict(Counter(r for q in queue for r in q["reasons"])),
            "criteria": crit, "criteria_basis": {"H": "kappa" if H["kappa_evaluable"] else "percent_agreement (kappa not evaluable)",
                                                 "M": "kappa" if M["kappa_evaluable"] else "percent_agreement (kappa not evaluable)"},
            "pilot_passed": all(crit.values())}, queue


def validate(rows, packet, mapping, schema):
    errs = []
    issued = {r["pilot_id"]: r for r in packet}
    if {r["pilot_id"] for r in rows} != set(issued):
        errs.append("pilot_id set differs from the issued packet")
    for r in rows:
        p = r["pilot_id"]
        if p in issued and r["text"] != issued[p]["text"]:
            errs.append(f"{p}: locked text changed")
        if p in mapping["items"] and hashlib.sha256(r["text"].encode("utf-8")).hexdigest() != mapping["items"][p]["text_sha256"]:
            errs.append(f"{p}: text hash mismatch")
        f = schema["fields"]
        for k in ("M_manipulation", "H_harmful_intent", "Q_confidence", "not_a_user_input", "toxic_only"):
            if r.get(k, "") not in f[k]["values"]:
                errs.append(f"{p}: {k}={r.get(k)!r} not allowed")
        ctx = [c for c in r.get("C_context", "").split(";") if c]
        if not ctx or any(c not in f["C_context"]["values"] for c in ctx) or ("none" in ctx and len(ctx) > 1):
            errs.append(f"{p}: C_context invalid")
        if "unclear" in (r.get("M_manipulation"), r.get("H_harmful_intent")) and r.get("Q_confidence") != "low":
            errs.append(f"{p}: Q must be low when unclear")
        if not r.get("reviewer_id") or not r.get("timestamp_utc"):
            errs.append(f"{p}: reviewer_id/timestamp missing")
    return errs


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def self_test():
    """Synthetic, in-memory only. Proves the arithmetic; produces no annotation file."""
    import numpy as np
    rng = np.random.default_rng(1)
    mapping = json.loads((PILOT / "owner_only/pilot_mapping.json").read_text(encoding="utf-8"))
    def synth(agree):
        out = []
        for p in sorted(mapping["items"]):
            h = ["harmful", "none", "dual_use"][int(rng.integers(0, 3))]
            out.append({"pilot_id": p, "H_harmful_intent": h, "M_manipulation": "no" if rng.random() > .1 else "yes",
                        "Q_confidence": "high"})
        return out
    a = synth(1)
    b = [dict(r) for r in a]                       # identical -> kappa must be 1
    res, _ = analyse(a, b, mapping)
    assert abs(res["H"]["kappa"] - 1) < 1e-12 and res["H"]["percent_agreement"] == 1
    b2 = synth(0)                                  # independent -> kappa near 0
    res2, _ = analyse(a, b2, mapping)
    assert abs(res2["H"]["kappa"]) < 0.25
    print("self-test passed (synthetic, in memory; nothing written):",
          {"identical_kappa_H": res["H"]["kappa"], "independent_kappa_H": round(res2["H"]["kappa"], 3)})


def main():
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    ra, rb = PILOT / "annotations_returned/reviewer_a_returned.csv", PILOT / "annotations_returned/reviewer_b_returned.csv"
    if not (ra.exists() and rb.exists()):
        raise SystemExit("No returned annotations yet: the pilot has not been run. Nothing computed.")
    mapping = json.loads((PILOT / "owner_only/pilot_mapping.json").read_text(encoding="utf-8"))
    schema = json.loads((PILOT / "annotation_schema.json").read_text(encoding="utf-8"))
    a, b = read_csv(ra), read_csv(rb)
    errs = validate(a, read_csv(PILOT / "reviewer_packets/reviewer_a.csv"), mapping, schema) + \
           validate(b, read_csv(PILOT / "reviewer_packets/reviewer_b.csv"), mapping, schema)
    if errs:
        raise SystemExit("Returned annotations are invalid:\n" + "\n".join(errs[:50]))
    res, queue = analyse(a, b, mapping)
    out = PILOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "pilot_agreement.json", "x", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    with open(out / "adjudication_queue.csv", "x", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pilot_id", "reasons", "text", "a_M", "a_H", "a_C", "a_Q", "b_M", "b_H", "b_C", "b_Q",
                    "adjudicator_id", "final_M", "final_H", "final_C", "final_Q", "rationale", "timestamp_utc"])
        A = {r["pilot_id"]: r for r in a}; Bm = {r["pilot_id"]: r for r in b}
        for q in queue:
            p = q["pilot_id"]
            w.writerow([p, ";".join(q["reasons"]), A[p]["text"], A[p]["M_manipulation"], A[p]["H_harmful_intent"],
                        A[p]["C_context"], A[p]["Q_confidence"], Bm[p]["M_manipulation"], Bm[p]["H_harmful_intent"],
                        Bm[p]["C_context"], Bm[p]["Q_confidence"]] + [""] * 7)
    print(json.dumps({k: res[k] for k in ("criteria", "criteria_basis", "pilot_passed", "adjudication_queue_size")}, indent=1))


if __name__ == "__main__":
    main()
