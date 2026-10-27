"""
E-R1 Part B — validate completed v4 NEI human review and compute the representativeness
audit (PROTOCOL_v4.md §5, §7, §8).

    python research/reliability/e_r1_part_b/validate_nei_review.py [--roles smoke calibration test]
    python research/reliability/e_r1_part_b/validate_nei_review.py --selftest

Reads the reviewer's filled packet_<role>.csv, the private mapping_<role>.json and
ATTESTATION_<role>.txt from data/fever/review_v4/. Exit code 0 only if every check
passes. On success writes e_r1_part_b/manifest/nei_v4_adjudicated.json and
nei_v4_representativeness.json (ids, decisions and counts only; no text).
This script never fills annotation fields and never decides acceptance itself: it
only checks the human's record for completeness and consistency.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REVIEW = REPO / "research/reliability/data/fever/review_v4"
MAN = REPO / "research/reliability/e_r1_part_b/manifest"
YN = {"Y", "N"}
FLAGS = ["accidental_support", "accidental_refutation", "wrong_sense_or_entity", "unusable_passage",
         "insufficiently_relevant"]
STATUSES = {"ACCEPTED", "EXCLUDED", "UNRESOLVED"}
ATTEST_PHRASES = ["reviewer", "date", "without model", "llm"]
TYPE_WORDS = {"film": {"film", "movie"}, "music": {"album", "song", "single", "band", "musician", "singer", "ep"},
              "tv": {"tv", "series", "episode", "season"}, "book": {"novel", "book", "comics", "manga"},
              "play": {"play", "musical", "opera"}, "game": {"game"},
              "place": {"city", "town", "county", "state", "river", "village"}}


def title_type(title: str) -> str:
    m = re.search(r"-LRB-(.+)-RRB-$", title)
    if not m:
        return "undisambiguated"
    words = set(re.findall(r"[a-z]+", m.group(1).lower()))
    for t, ws in TYPE_WORDS.items():
        if words & ws:
            return t
    return "other_disambiguated"


def validate_role(role: str, base: Path, errors: list[str]) -> list[dict]:
    mapping = json.loads((base / f"mapping_{role}.json").read_text(encoding="utf-8"))["records"]
    by_id = {r["review_id"]: r for r in mapping}
    att = base / f"ATTESTATION_{role}.txt"
    if not att.exists():
        errors.append(f"{role}: missing {att.name}")
    else:
        raw = att.read_text(encoding="utf-8")
        text = raw.lower()
        for p in ATTEST_PHRASES:
            if p not in text:
                errors.append(f"{role}: attestation lacks '{p}'")
        for label in ("Reviewer name or id:", "Review date(s):", "Signature / initials:"):
            m = re.search(re.escape(label) + r"[ \t]*(\S.*)", raw)
            if not m:
                errors.append(f"{role}: attestation field '{label}' is not filled in")
    rows = list(csv.DictReader((base / f"packet_{role}.csv").open(encoding="utf-8-sig")))
    seen, out = set(), []
    for row in rows:
        rid = row.get("review_id", "")
        rec = by_id.get(rid)
        where = f"{role}:{rid}"
        if rec is None:
            errors.append(f"{where}: review id not in the frozen mapping")
            continue
        if rid in seen:
            errors.append(f"{where}: duplicate row")
        seen.add(rid)
        if hashlib.sha256(row.get("claim", "").encode()).hexdigest() != rec["claim_sha256"]:
            errors.append(f"{where}: claim text differs from the frozen claim")
        sel = row.get("selected_candidate_id", "").strip().upper()
        letters = {c["display_letter"] for c in rec["candidates"]}
        subj = row.get("correct_entity_or_subject", "").strip().upper()
        conf = row.get("reviewer_confidence", "").strip().lower()
        status = row.get("adjudication_status", "").strip().upper()
        flags = {f: row.get(f, "").strip().upper() for f in FLAGS}
        rationale = row.get("rationale", "").strip()
        if not row.get("reviewer_id", "").strip():
            errors.append(f"{where}: reviewer_id empty")
        if not rationale:
            errors.append(f"{where}: rationale required")
        if conf not in {"high", "medium", "low"}:
            errors.append(f"{where}: reviewer_confidence must be high/medium/low")
        if status not in STATUSES:
            errors.append(f"{where}: adjudication_status must be one of {sorted(STATUSES)}")
        for f, v in flags.items():
            if v not in YN:
                errors.append(f"{where}: {f} must be Y or N")
        if sel == "NONE":
            if subj not in {"N/A", ""}:
                errors.append(f"{where}: correct_entity_or_subject must be N/A when NONE")
            if status == "ACCEPTED":
                errors.append(f"{where}: NONE cannot be ACCEPTED")
            if status == "EXCLUDED" and "Y" not in flags.values():
                errors.append(f"{where}: EXCLUDED NONE needs at least one Y reason")
        elif sel in letters:
            if subj not in {"Y", "N", "UNSURE"}:
                errors.append(f"{where}: correct_entity_or_subject must be Y/N/UNSURE for a selection")
            clean = subj == "Y" and all(v == "N" for v in flags.values()) and conf != "low"
            if status == "ACCEPTED" and not clean:
                errors.append(f"{where}: ACCEPTED requires subject Y, all flags N, confidence not low")
            if clean and status != "ACCEPTED":
                errors.append(f"{where}: a clean high/medium selection must be ACCEPTED (or explain via UNRESOLVED + low)")
            if (conf == "low" or subj == "UNSURE") and status != "UNRESOLVED":
                errors.append(f"{where}: low confidence or UNSURE subject must be UNRESOLVED")
            if status == "EXCLUDED" and "Y" not in flags.values() and subj != "N":
                errors.append(f"{where}: EXCLUDED selection needs a defect flag or subject N")
        else:
            errors.append(f"{where}: selected_candidate_id '{sel}' is not NONE or one of {sorted(letters)}")
        if not rec["candidates"] and sel != "NONE":
            errors.append(f"{where}: no candidates were offered; only NONE is valid")
        chosen = next((c for c in rec["candidates"] if c["display_letter"] == sel), None)
        out.append({
            "review_id": rid, "item_id": rec["item_id"], "fever_id": rec["fever_id"], "role": role,
            "decision": status, "selected_letter": sel, "selected_hidden_rank": chosen["rank"] if chosen else None,
            "selected_title": chosen["title"] if chosen else None,
            "selected_page_id": chosen["page_id"] if chosen else None,
            "selected_sentence_indices": chosen["sentence_indices"] if chosen else None,
            "reasons": sorted(f for f, v in flags.items() if v == "Y"), "confidence": conf,
            "subject": subj, "n_candidates": len(rec["candidates"]),
            "title_type": title_type((chosen or (rec["candidates"][0] if rec["candidates"] else {"title": ""}))["title"]),
            "claim_tokens": len(rec["claim"].split()),
            "multi_capitalised_spans": len({c["matched_span"] for c in rec["candidates"]}) > 1,
        })
    missing = set(by_id) - seen
    if missing:
        errors.append(f"{role}: {len(missing)} frozen claims have no review row (e.g. {sorted(missing)[:3]})")
    return out


def representativeness(decisions: list[dict]) -> dict:
    rep, flags = {}, []
    for role in sorted({d["role"] for d in decisions}):
        ds = [d for d in decisions if d["role"] == role]
        acc = [d for d in ds if d["decision"] == "ACCEPTED"]
        exc = [d for d in ds if d["decision"] != "ACCEPTED"]
        tt_all = collections.Counter(d["title_type"] for d in ds)
        tt_exc = collections.Counter(d["title_type"] for d in exc)
        rate = len(exc) / len(ds) if ds else 0.0
        rep[role] = {
            "claims": len(ds), "accepted": len(acc), "excluded": sum(d["decision"] == "EXCLUDED" for d in ds),
            "unresolved": sum(d["decision"] == "UNRESOLVED" for d in ds), "not_accepted_rate": rate,
            "none_selected": sum(d["selected_letter"] == "NONE" for d in ds),
            "selected_letter": dict(collections.Counter(d["selected_letter"] for d in acc)),
            "selected_hidden_rank": dict(collections.Counter(d["selected_hidden_rank"] for d in acc)),
            "reasons": dict(collections.Counter(r for d in ds for r in d["reasons"])),
            "confidence": dict(collections.Counter(d["confidence"] for d in ds)),
            "title_type_all": dict(tt_all), "title_type_not_accepted": dict(tt_exc),
            "mean_claim_tokens": {"accepted": sum(d["claim_tokens"] for d in acc) / max(len(acc), 1),
                                  "not_accepted": sum(d["claim_tokens"] for d in exc) / max(len(exc), 1)},
            "multi_span_share": {"accepted": sum(d["multi_capitalised_spans"] for d in acc) / max(len(acc), 1),
                                 "not_accepted": sum(d["multi_capitalised_spans"] for d in exc) / max(len(exc), 1)},
        }
        if rate > 0.35:
            flags.append(f"{role}: {rate:.0%} not accepted > 35%")
        for t, n in tt_exc.items():
            if exc and n / len(exc) > 0.5 and tt_all[t] / len(ds) < 0.25:
                flags.append(f"{role}: title type '{t}' is {n / len(exc):.0%} of exclusions but "
                             f"{tt_all[t] / len(ds):.0%} of claims")
    return {"per_role": rep, "stop_rule_flags": flags,
            "stop_rule": "PROTOCOL_v4 §8: >35% not accepted in a role, or one title type >50% of exclusions "
                         "while <25% of claims -> report bias and revise design before smoke"}


def kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def run(roles: list[str], base: Path, write: bool) -> int:
    errors, decisions = [], []
    for role in roles:
        decisions += validate_role(role, base, errors)
    second = base / "second_review_packet_completed.csv"
    agreement = None
    if second.exists():
        primary = {d["review_id"]: d for d in decisions}
        rows = list(csv.DictReader(second.open(encoding="utf-8-sig")))
        pairs = [(primary[r["review_id"]], r) for r in rows if r["review_id"] in primary]
        agreement = {"n": len(pairs),
                     "kappa_selection": kappa([p["selected_letter"] for p, _ in pairs],
                                              [r["selected_candidate_id"].strip().upper() for _, r in pairs]),
                     "kappa_accepted": kappa([p["decision"] == "ACCEPTED" for p, _ in pairs],
                                             [r["adjudication_status"].strip().upper() == "ACCEPTED" for _, r in pairs])}
    print(f"{len(decisions)} rows checked, {len(errors)} errors")
    for e in errors[:60]:
        print("  ERROR", e)
    if errors:
        return 1
    rep = representativeness(decisions)
    print(json.dumps({r: {k: v[k] for k in ("claims", "accepted", "excluded", "unresolved", "not_accepted_rate")}
                      for r, v in rep["per_role"].items()}, indent=1))
    print("stop-rule flags:", rep["stop_rule_flags"] or "none")
    if write:
        (MAN / "nei_v4_adjudicated.json").write_text(json.dumps(
            {"schema": "e-r1-partB-nei-v4-adjudicated/1", "decisions": decisions, "second_review": agreement},
            indent=1), encoding="utf-8")
        (MAN / "nei_v4_representativeness.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


def selftest() -> int:
    """Synthetic, clearly marked rows in a temp copy; proves the checks fire. Never touches review_v4."""
    with tempfile.TemporaryDirectory(prefix="nei-v4-selftest-") as tmp:
        t = Path(tmp)
        shutil.copy(REVIEW / "mapping_smoke.json", t / "mapping_smoke.json")
        rows = list(csv.DictReader((REVIEW / "packet_smoke.csv").open(encoding="utf-8-sig")))
        mapping = {r["review_id"]: r for r in json.loads((t / "mapping_smoke.json").read_text(encoding="utf-8"))["records"]}

        def write(mutate):
            out = []
            for i, row in enumerate(rows):
                row = dict(row)
                first = mapping[row["review_id"]]["candidates"][0]["display_letter"]
                row.update({"selected_candidate_id": first, "correct_entity_or_subject": "Y",
                            **{f: "N" for f in FLAGS}, "reviewer_confidence": "high",
                            "rationale": "SYNTHETIC SELFTEST ROW", "adjudication_status": "ACCEPTED",
                            "reviewer_id": "SELFTEST"})
                mutate(i, row)
                out.append(row)
            with (t / "packet_smoke.csv").open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(out)
            (t / "ATTESTATION_smoke.txt").write_text(
                "SELFTEST (synthetic)\nReviewer name or id: SELFTEST\nReview date(s): n/a\n"
                "done without model or LLM assistance\nSignature / initials: SELFTEST\n", encoding="utf-8")

        cases = {
            "valid": (lambda i, r: None, 0),
            "bad letter": (lambda i, r: r.update(selected_candidate_id="Z") if i == 0 else None, 1),
            "accepted with flag": (lambda i, r: r.update(accidental_support="Y") if i == 0 else None, 1),
            "none accepted": (lambda i, r: r.update(selected_candidate_id="NONE", correct_entity_or_subject="N/A")
                              if i == 0 else None, 1),
            "low conf accepted": (lambda i, r: r.update(reviewer_confidence="low") if i == 0 else None, 1),
            "empty rationale": (lambda i, r: r.update(rationale="") if i == 0 else None, 1),
            "claim edited": (lambda i, r: r.update(claim=r["claim"] + " x") if i == 0 else None, 1),
            "valid NONE": (lambda i, r: r.update(selected_candidate_id="NONE", correct_entity_or_subject="N/A",
                                                 wrong_sense_or_entity="Y", adjudication_status="EXCLUDED")
                           if i == 0 else None, 0),
        }
        bad = 0
        for name, (mut, expect) in cases.items():
            write(mut)
            code = run(["smoke"], t, write=False)
            ok = (code == 0) == (expect == 0)
            print(f"SELFTEST {name}: exit {code} -> {'ok' if ok else 'WRONG'}\n")
            bad += not ok
        print("SELFTEST", "PASSED" if not bad else f"FAILED ({bad})")
        return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="+", default=["smoke", "calibration", "test"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else run(a.roles, REVIEW, write=True))
