"""
C2 Phase 1.2 — deterministic, blinded 200-row taxonomy-pilot sample and two reviewer packets.

Selection uses only: source identity, file/line position, fixed regex cues (refusal text,
harm-topic vocabulary, from the Phase 1 audit) and the Phase 1 cross-label conflict list
(embedding cosine). No classifier prediction, score or test outcome is used.

Original data files are read, never written. Reviewers see the full original prompt text
(not the 512-character training cut) and nothing else: no label, source, score or other
reviewer's answers.

Outputs (exclusive-create, under phase1_2_pilot/):
  reviewer_packets/reviewer_a.csv, reviewer_packets/reviewer_b.csv   (220 rows each, blank annotations)
  owner_only/pilot_mapping.json                                      (pilot_id -> original row, label, source, hashes)
  sampling_manifest.json                                             (algorithm, seed, strata, counts, limitations)
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import socket
import sys
from pathlib import Path

PILOT = Path(__file__).resolve().parent.parent
C2 = PILOT.parent
REPO = C2.parents[2]
E53 = REPO / "research/adversarial/e53_frozen_encoder_head_diagnostic"
socket.socket.connect = socket.create_connection = socket.getaddrinfo = (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))  # type: ignore
sys.dont_write_bytecode = True

SEED = 20261018
N_REPEATS = 20
MIN_REPEAT_GAP = 50
TRAIN_FILES = ["data/pair_training/train.decontam.jsonl", "data/pair_training_v6/dataset_v6.decontam.jsonl",
               "data/pair_training_v6/augmented_v6.decontam.jsonl", "data/pair_training_v6/augmented_v6_3_train.jsonl",
               "data/pair_training_v6/augmented_v6_3b_benign_train.jsonl"]
# (stratum, sources, n, n_with_harm_vocab_cue or None, extra filter)
STRATA = [
    ("S2_redteam", ["anthropic_rlhf_redteam"], 40, 20, None),
    ("S3_harmfulqa", ["harmfulqa"], 40, 15, None),
    ("S4a_toxicchat_jailbreak", ["toxicchat"], 15, None, ("category", "jailbreak")),
    ("S4b_toxicchat_toxic_query", ["toxicchat"], 15, None, ("category", "toxic_query")),
    ("S5a_llama_refusal", ["llama_paraphrase:llama3.1:8b"], 10, None, ("refusal", True)),
    ("S5b_llama_non_refusal", ["llama_paraphrase:llama3.1:8b"], 10, None, ("refusal", False)),
    ("S6_retained_attack_harmful_behaviors", ["mlabonne/harmful_behaviors"], 6, None, None),
    ("S6_retained_attack_forbidden_qs", ["TrustAIRLab/forbidden_question_set"], 6, None, None),
    ("S6_retained_attack_maliciousinstruct", ["walledai/MaliciousInstruct"], 4, None, None),
    ("S6_retained_attack_jbb_behaviors", ["JailbreakBench/JBB-Behaviors"], 3, None, None),
    ("S6_retained_attack_aug_v63_seeds", ["aug_v63"], 6, None, ("mode", "seed")),
    ("S7_retained_benign_alpaca", ["alpaca"], 4, 2, None),
    ("S7_retained_benign_tatsu_alpaca", ["tatsu-lab/alpaca"], 2, 1, None),
    ("S7_retained_benign_hh_helpful", ["anthropic_rlhf_helpful"], 7, 4, None),
    ("S7_retained_benign_law_stackexchange", ["dim/law_stackexchange"], 6, 2, None),
    ("S7_retained_benign_medquad", ["keivalya/MedQuad-MedicalQnADataset"], 3, None, None),
    ("S7_retained_benign_medqa", ["medalpaca/medical_meadow_medqa"], 3, 2, None),
    ("S7_retained_benign_mbpp", ["google-research-datasets/mbpp"], 2, None, None),
    ("S7_retained_benign_hand_written", ["hand_written"], 4, 2, None),
    ("S7_retained_benign_aug_v63b", ["aug_v63b"], 4, 2, None),
]
N_CONFLICT_PAIRS = 5


def sha256(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def norm_sha1(t: str) -> str:
    return hashlib.sha1(" ".join(t.lower().split()).encode("utf-8")).hexdigest()


def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="") as f:
        f.write(text)


def main() -> None:
    import numpy as np
    rng = np.random.default_rng(SEED)
    audit = json.loads((C2 / "audit/training_data_audit.json").read_text(encoding="utf-8"))
    refusal = re.compile(audit["cue_regex"]["refusal_text"], re.I)
    harm = re.compile(audit["cue_regex"]["harm_topic"], re.I)

    # ── rebuild the v6.3b corpus from the original files, keeping file + line ──
    manifest = [json.loads(l) for l in open(E53 / "manifests/train_corpus.jsonl", encoding="utf-8")]
    rows, seen = [], set()
    file_hashes = {}
    for rel in TRAIN_FILES:
        raw = (REPO / rel).read_bytes()
        file_hashes[rel] = hashlib.sha256(raw).hexdigest()
        for line_no, line in enumerate(raw.decode("utf-8").splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            p = (r.get("prompt") or "").strip()
            if not p or norm_sha1(p) in seen:
                continue
            seen.add(norm_sha1(p))
            label = 1 if r.get("label", 0) in (1, "1", "attack", True) else 0
            rows.append({"i": len(rows), "file": rel, "line": line_no, "text": p, "label": label,
                         "source": r.get("source"), "category": r.get("category"), "mode": r.get("mode")})
    assert len(rows) == len(manifest)
    for r, m in zip(rows, manifest):
        assert sha256(r["text"][:512]) == m["text_sha256"] and r["source"] == m["source"] and r["label"] == m["label"]

    E = np.load(E53 / "cache/embeddings.npy")
    eidx = {h: k for k, h in enumerate(json.loads((E53 / "cache/embedding_index.json").read_text(encoding="utf-8")))}
    X = E[[eidx[m["text_sha256"]] for m in manifest]]

    selected: dict[int, str] = {}
    sel_vecs: list = []

    def near_dup(i):
        if not sel_vecs:
            return False
        return float((np.vstack(sel_vecs) @ X[i]).max()) >= 0.95

    def take(i, stratum):
        selected[i] = stratum
        sel_vecs.append(X[i])

    # ── S1: cross-label conflict pairs (intentionally related rows) ──────────
    conflict_ids = sorted({i for s in audit["sources"].values() for i in s["conflict_row_ids"]})
    pairs = []
    for i in conflict_ids:
        opp = [j for j in range(len(rows)) if rows[j]["label"] != rows[i]["label"]]
        j = opp[int(np.argmax(X[opp] @ X[i]))]
        key = tuple(sorted((i, j)))
        if key not in pairs:
            pairs.append(key)
    order = rng.permutation(len(pairs))
    s1 = []
    for k in order:
        a, b = pairs[k]
        if a in selected or b in selected:
            continue
        take(a, "S1_cross_label_conflict_pair"); take(b, "S1_cross_label_conflict_pair")
        s1.append({"rows": [a, b], "cosine": round(float(X[a] @ X[b]), 4)})
        if len(s1) == N_CONFLICT_PAIRS:
            break

    # ── S2..S7 ──────────────────────────────────────────────────────────────
    availability, shortages = {}, {}
    for name, sources, n, n_cue, filt in STRATA:
        pool = [r["i"] for r in rows if r["source"] in sources and r["i"] not in selected]
        if filt:
            key, val = filt
            if key == "refusal":
                pool = [i for i in pool if bool(refusal.search(rows[i]["text"])) == val]
            else:
                pool = [i for i in pool if rows[i][key] == val]
        cue = [i for i in pool if harm.search(rows[i]["text"])]
        nocue = [i for i in pool if not harm.search(rows[i]["text"])]
        availability[name] = {"pool": len(pool), "with_harm_vocab_cue": len(cue)}
        plan = [(cue, n_cue), (nocue, n - n_cue)] if n_cue is not None else [(pool, n)]
        got = 0
        for cand, want in plan:
            taken = 0
            for k in rng.permutation(len(cand)):
                i = cand[k]
                if i in selected or near_dup(i):
                    continue
                take(i, name); taken += 1; got += 1
                if taken == want:
                    break
            if taken < want:
                shortages[f"{name}:{'cue' if cand is cue else 'nocue' if cand is nocue else 'any'}"] = want - taken
        availability[name]["selected"] = got
    if len(selected) != 200:
        raise SystemExit(f"STOP: selected {len(selected)} rows, not 200; shortages {shortages}")

    # ── pilot ids, repeats, independent orders ───────────────────────────────
    sel = sorted(selected)
    ids = set()
    def new_id():
        while True:
            x = "P" + "".join(f"{v:02x}" for v in rng.integers(0, 256, 4))
            if x not in ids:
                ids.add(x); return x
    pid = {i: new_id() for i in sel}
    repeat_rows = sorted(int(sel[k]) for k in rng.choice(len(sel), N_REPEATS, replace=False))
    rid = {i: new_id() for i in repeat_rows}          # second pilot id for the repeat copy
    items = [(pid[i], i) for i in sel] + [(rid[i], i) for i in repeat_rows]

    def make_order():
        # Shuffle the 200 originals, then insert each repeat copy at a seeded random position
        # at least MIN_REPEAT_GAP away from its original. An insertion never shrinks an
        # existing gap, so every earlier constraint stays satisfied.
        perm = [(pid[i], i) for i in (sel[k] for k in rng.permutation(len(sel)))]
        for i in (repeat_rows[k] for k in rng.permutation(len(repeat_rows))):
            p = next(n for n, (q, _) in enumerate(perm) if q == pid[i])
            valid = [n for n in range(len(perm) + 1) if n <= p - MIN_REPEAT_GAP or n >= p + 1 + MIN_REPEAT_GAP]
            perm.insert(int(valid[int(rng.integers(0, len(valid)))]), (rid[i], i))
        pos = {q: n for n, (q, _) in enumerate(perm)}
        assert all(abs(pos[pid[i]] - pos[rid[i]]) >= MIN_REPEAT_GAP for i in repeat_rows)
        return perm
    order_a, order_b = make_order(), make_order()
    assert [p for p, _ in order_a] != [p for p, _ in order_b]

    cols = ["position", "pilot_id", "text", "reviewer_id", "M_manipulation", "H_harmful_intent", "C_context",
            "Q_confidence", "not_a_user_input", "toxic_only", "same_topic_pair_note", "notes", "timestamp_utc"]
    for name, order in (("reviewer_a", order_a), ("reviewer_b", order_b)):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\r\n")
        w.writerow(cols)
        for n, (p, i) in enumerate(order, start=1):
            w.writerow([n, p, rows[i]["text"]] + [""] * (len(cols) - 3))
        write_new(PILOT / "reviewer_packets" / f"{name}.csv", "﻿" + buf.getvalue())

    mapping = {
        "WARNING": "OWNER ONLY. Never give this file to reviewers: it reveals original labels and sources.",
        "seed": SEED,
        "items": {p: {"corpus_row": i, "file": rows[i]["file"], "line": rows[i]["line"],
                      "text_sha256": sha256(rows[i]["text"]), "train_text_sha256_512": manifest[i]["text_sha256"],
                      "original_label": rows[i]["label"], "source": rows[i]["source"], "category": rows[i]["category"],
                      "mode": rows[i]["mode"], "stratum": selected[i],
                      "repeat_of": None if p == pid[i] else pid[i],
                      "is_preregistered_repeat_pair": i in rid,
                      "truncated_in_training": len(rows[i]["text"]) > 512,
                      "harm_vocab_cue": bool(harm.search(rows[i]["text"])),
                      "refusal_cue": bool(refusal.search(rows[i]["text"])),
                      "position_a": next(n for n, (q, _) in enumerate(order_a, 1) if q == p),
                      "position_b": next(n for n, (q, _) in enumerate(order_b, 1) if q == p)}
                  for p, i in items},
        "conflict_pairs": [{"pilot_ids": [pid[a], pid[b]], "cosine": c["cosine"]} for c in s1 for a, b in [c["rows"]]],
    }
    write_new(PILOT / "owner_only" / "pilot_mapping.json", json.dumps(mapping, indent=2, sort_keys=True) + "\n")

    from collections import Counter
    manifest_out = {
        "seed": SEED, "algorithm": [
            "rebuild the v6.3b corpus from the five original files in recipe order with SHA-1 normalised dedup; verify every row against E53 train_corpus (512-char hash, source, label)",
            "S1: 5 cross-label conflict pairs (Phase 1 audit, nearest opposite-label cosine >= 0.92), both rows kept intentionally",
            "S2..S7: per stratum, seeded permutation of the pool; harm-vocabulary-cue quota where set; skip a row if cosine >= 0.95 to an already selected row",
            f"{N_REPEATS} preregistered repeat rows (seeded) get a second pilot id; each reviewer sees 220 items; every repeat pair >= {MIN_REPEAT_GAP} positions apart in both packets",
            "reviewer packets: independent seeded orders; columns = position, pilot_id, text, blank annotation fields"],
        "uses_no_model_prediction": True,
        "strata": {k: v for k, v in Counter(selected.values()).items()},
        "availability": availability, "shortages": shortages,
        "original_label_counts_in_sample": dict(Counter(str(rows[i]["label"]) for i in sel)),
        "rows_with_harm_vocab_cue": sum(bool(harm.search(rows[i]["text"])) for i in sel),
        "rows_with_refusal_cue": sum(bool(refusal.search(rows[i]["text"])) for i in sel),
        "rows_truncated_in_training_shown_in_full": sum(len(rows[i]["text"]) > 512 for i in sel),
        "conflict_pairs": len(s1), "repeats": N_REPEATS, "items_per_packet": len(items),
        "original_file_sha256": file_hashes,
        "e53_inputs_sha256": {"manifests/train_corpus.jsonl": hashlib.sha256((E53 / "manifests/train_corpus.jsonl").read_bytes()).hexdigest(),
                              "cache/embeddings.npy": hashlib.sha256((E53 / "cache/embeddings.npy").read_bytes()).hexdigest()},
        "limitations": [
            "not statistically representative of the corpus: strata are fixed quotas chosen to exercise the rubric",
            "harm-vocabulary and refusal cues are regex heuristics, not labels",
            "dual-use / ambiguous rows cannot be pre-identified without review; law_stackexchange, aug_v63b, hand_written and medqa strata are included as likely sources",
            "instruction-manipulation candidates are scarce: only the 71 ToxicChat jailbreak rows (15 sampled)",
            "texts are shown as single user messages; hh-rlhf rows are first human turns of longer conversations and that context is not available locally"],
    }
    write_new(PILOT / "sampling_manifest.json", json.dumps(manifest_out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest_out[k] for k in ("strata", "shortages", "original_label_counts_in_sample",
                                                   "rows_with_harm_vocab_cue", "rows_with_refusal_cue",
                                                   "rows_truncated_in_training_shown_in_full")}, indent=1))


if __name__ == "__main__":
    main()
