"""
C2 Phase 1 — read-only audit of the PAIR v6.3b training corpus.

No model is trained or scored. Inputs are existing, checksum-verified artifacts:
  E53 manifests/train_corpus.jsonl, cache/texts.jsonl, cache/embeddings.npy (+ index),
  results/scores_a5.csv (row-random out-of-fold and leave-one-source-out scores saved by E53).
Lexical cue rates use fixed, documented regexes: they are heuristics, not labels.

Output (exclusive-create): audit/training_data_audit.json
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[2]
E53 = REPO / "research/adversarial/e53_frozen_encoder_head_diagnostic"


def _blocked(*a, **k):
    raise RuntimeError("network forbidden in C2 Phase 1")


socket.socket.connect = socket.create_connection = socket.getaddrinfo = _blocked  # type: ignore
sys.dont_write_bytecode = True

CUES = {
    # instruction-hierarchy manipulation (override / persona / extraction / delimiter)
    "manipulation": re.compile(r"\b(ignore (all |the )?(previous|prior|above)|disregard|jailbreak|\bDAN\b|do anything now|"
                               r"developer mode|you are now|pretend (to be|you are)|roleplay|role-play|act as|"
                               r"no (restrictions|filters|guidelines)|system prompt|from now on)\b", re.I),
    # harmful-request action framing
    "request_action": re.compile(r"\b(how (do|can|would|to)|give me|tell me|write|create|make|generate|explain how|"
                                 r"step[- ]by[- ]step|instructions? (for|to)|help me)\b", re.I),
    # harm-topic vocabulary (topic, not intent)
    "harm_topic": re.compile(r"\b(kill|murder|bomb|explosive|weapon|gun|poison|drug|meth|cocaine|hack|malware|virus|"
                             r"steal|theft|fraud|scam|suicide|racist|terror|abuse|rape|assault|attack|violence)\w*\b", re.I),
    "refusal_text": re.compile(r"^\s*(i can'?t|i cannot|i'm not able|i am not able|i won'?t|sorry, but|i'm sorry)", re.I),
    "conversation_marker": re.compile(r"(\bHuman:|\bAssistant:|\bUser:|\bAI:)"),
}


def norm(t):
    return hashlib.sha1(" ".join(t.lower().split()).encode()).hexdigest()


def main() -> None:
    import numpy as np

    for line in (E53 / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        d, rel = line.split("  ", 1)
        if hashlib.sha256((E53 / rel).read_bytes()).hexdigest() != d:
            raise SystemExit(f"STOP: E53 artifact changed: {rel}")
    corpus = [json.loads(l) for l in open(E53 / "manifests/train_corpus.jsonl", encoding="utf-8")]
    texts = {json.loads(l)["sha256"]: json.loads(l)["text"] for l in open(E53 / "cache/texts.jsonl", encoding="utf-8")}
    E = np.load(E53 / "cache/embeddings.npy")
    order = json.loads((E53 / "cache/embedding_index.json").read_text(encoding="utf-8"))
    row = {h: k for k, h in enumerate(order)}
    X = E[[row[r["text_sha256"]] for r in corpus]]
    lab = np.array([r["label"] for r in corpus]); src = np.array([r["source"] for r in corpus])
    T = [texts[r["text_sha256"]] for r in corpus]
    a5 = {int(r["i"]): r for r in csv.DictReader(open(E53 / "results/scores_a5.csv", encoding="utf-8"))}

    S = X @ X.T
    np.fill_diagonal(S, -1)
    nn_same_label = np.where(lab[None, :] == lab[:, None], S, -1).max(1)
    nn_other_label = np.where(lab[None, :] != lab[:, None], S, -1).max(1)
    nn_other_label_idx = np.where(lab[None, :] != lab[:, None], S, -1).argmax(1)

    out = {"n_rows": len(corpus), "label_counts": {str(k): int(v) for k, v in collections.Counter(lab.tolist()).items()},
           "sources": {}, "thresholds": {"near_dup": 0.95, "cross_label_conflict": 0.92, "topic_matched_candidate": 0.80},
           "cue_regex": {k: v.pattern for k, v in CUES.items()}}
    for s in sorted(set(src.tolist())):
        idx = np.nonzero(src == s)[0]
        L = int(lab[idx[0]])
        words = np.array([len(T[i].split()) for i in idx])
        cue = {k: float(np.mean([bool(CUES[k].search(T[i])) for i in idx])) for k in CUES}
        within = S[np.ix_(idx, idx)]
        near_dup_rows = int((within >= 0.95).any(1).sum())
        exact = len(idx) - len({norm(T[i]) for i in idx})
        conflicts = [int(i) for i in idx if nn_other_label[i] >= 0.92]
        topic_pairs = int((nn_other_label[idx] >= 0.80).sum())
        oof = [float(a5[corpus[i]["i"]]["rowcv_oof_score"]) for i in idx if corpus[i]["split"] == "tr"]
        loso = [float(a5[corpus[i]["i"]]["loso_score"]) for i in idx if corpus[i]["split"] == "tr" and a5[corpus[i]["i"]]["loso_score"]]
        if L == 1:
            disagree_oof = float(np.mean([x < 0.2 for x in oof])) if oof else None
            disagree_loso = float(np.mean([x < 0.2 for x in loso])) if loso else None
        else:
            disagree_oof = float(np.mean([x > 0.8 for x in oof])) if oof else None
            disagree_loso = float(np.mean([x > 0.8 for x in loso])) if loso else None
        cats = collections.Counter(str(corpus[i]["category"]) for i in idx)
        modes = collections.Counter(str(corpus[i]["mode"]) for i in idx)
        files = collections.Counter(corpus[i]["file"] for i in idx)
        splits = collections.Counter(corpus[i]["split"] for i in idx)
        # nearest other-label row: which sources it comes from (register confound)
        nn_src = collections.Counter(src[nn_other_label_idx[i]] for i in idx)
        out["sources"][s] = {
            "label": L, "rows": len(idx), "files": dict(files), "splits": dict(splits),
            "category_values": dict(cats.most_common(12)), "n_category_values": len(cats), "modes": dict(modes),
            "words_mean": float(words.mean()), "words_median": float(np.median(words)), "words_p90": float(np.quantile(words, .9)),
            "share_truncated_512_chars": float(np.mean([len(T[i]) >= 512 for i in idx])),
            "cue_rates": cue,
            "exact_duplicates_after_normalisation": int(exact),
            "rows_with_within_source_near_duplicate_ge_0.95": near_dup_rows,
            "rows_with_opposite_label_neighbour_ge_0.92": len(conflicts),
            "conflict_row_ids": [int(corpus[i]["i"]) for i in conflicts][:50],
            "rows_with_opposite_label_neighbour_ge_0.80": topic_pairs,
            "mean_nn_cos_same_label": float(nn_same_label[idx].mean()),
            "mean_nn_cos_opposite_label": float(nn_other_label[idx].mean()),
            "nearest_opposite_label_row_source": dict(nn_src.most_common(5)),
            "e53_rowcv_disagreement_rate": disagree_oof,
            "e53_loso_disagreement_rate": disagree_loso,
            "disagreement_definition": "attack: score < 0.2; benign: score > 0.8 (E53 saved scores, tr rows only)",
        }
    # topic-vs-label confound across the whole corpus
    ht = np.array([bool(CUES["harm_topic"].search(t)) for t in T])
    out["harm_topic_vocab_by_label"] = {"attack": float(ht[lab == 1].mean()), "benign": float(ht[lab == 0].mean()),
                                        "P(attack | harm_topic)": float(lab[ht].mean()), "P(attack | no harm_topic)": float(lab[~ht].mean())}
    words = np.array([len(t.split()) for t in T])
    out["length_by_label"] = {"attack_mean_words": float(words[lab == 1].mean()), "benign_mean_words": float(words[lab == 0].mean())}
    # aug_v63: recover seed families (back-translations -> nearest seed)
    a = np.nonzero(src == "aug_v63")[0]
    seeds = [i for i in a if corpus[i]["mode"] == "seed"]; bts = [i for i in a if corpus[i]["mode"] != "seed"]
    if seeds:
        sim = S[np.ix_(bts, seeds)]
        best = sim.max(1); second = np.sort(sim, 1)[:, -2] if len(seeds) > 1 else best
        out["aug_v63_family_recovery"] = {"seeds": len(seeds), "back_translations": len(bts),
                                          "bt_to_nearest_seed_cos_median": float(np.median(best)),
                                          "bt_with_margin_lt_0.05_over_second_seed": int((best - second < 0.05).sum()),
                                          "bt_nearest_seed_cos_lt_0.7": int((best < 0.7).sum())}
    # llama paraphrase rows that are refusals labelled attack
    lp = np.nonzero(src == "llama_paraphrase:llama3.1:8b")[0]
    out["llama_paraphrase_refusal_rows"] = [int(corpus[i]["i"]) for i in lp if CUES["refusal_text"].search(T[i])]
    path = HERE / "audit" / "training_data_audit.json"
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print("written", path)


if __name__ == "__main__":
    main()
