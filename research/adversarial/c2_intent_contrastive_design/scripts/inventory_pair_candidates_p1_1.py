"""
C2 Phase 1.1 — read-only inventory of the near-neighbour minimal-pair candidates.

For each attack-labelled training row whose nearest benign-labelled row has MiniLM
cosine >= 0.80 (the 595 candidates of Phase 1), record ids and surface metrics only:
cosine, length ratio, token Jaccard, harm-vocabulary count difference, source pair,
register match, review status, and benign-partner reuse. No text is written, nothing is
generated, no model is trained or scored (cosines come from E53's cached embeddings).

Output (exclusive-create): audit/phase1_1_pair_candidates.json
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[2]
E53 = REPO / "research/adversarial/e53_frozen_encoder_head_diagnostic"
socket.socket.connect = socket.create_connection = socket.getaddrinfo = (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))  # type: ignore
sys.dont_write_bytecode = True

REGISTER = {  # source -> register family (for "same register" pairs)
    "anthropic_rlhf_redteam": "hh-rlhf", "anthropic_rlhf_helpful": "hh-rlhf",
    "alpaca": "alpaca", "tatsu-lab/alpaca": "alpaca",
}
REVIEW_REQUIRED = {"anthropic_rlhf_redteam", "harmfulqa", "toxicchat", "llama_paraphrase:llama3.1:8b"}
REFUSAL = re.compile(r"^\s*(i can'?t|i cannot|i'm not able|i am not able|i won'?t|sorry, but|i'm sorry)", re.I)


def toks(t):
    return set(re.findall(r"[a-z0-9']+", t.lower()))


def main() -> None:
    import numpy as np
    for line in (E53 / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        d, rel = line.split("  ", 1)
        if hashlib.sha256((E53 / rel).read_bytes()).hexdigest() != d:
            raise SystemExit(f"STOP: E53 artifact changed: {rel}")
    audit = json.loads((HERE / "audit/training_data_audit.json").read_text(encoding="utf-8"))
    harm = re.compile(audit["cue_regex"]["harm_topic"], re.I)
    corpus = [json.loads(l) for l in open(E53 / "manifests/train_corpus.jsonl", encoding="utf-8")]
    texts = {json.loads(l)["sha256"]: json.loads(l)["text"] for l in open(E53 / "cache/texts.jsonl", encoding="utf-8")}
    E = np.load(E53 / "cache/embeddings.npy")
    row = {h: k for k, h in enumerate(json.loads((E53 / "cache/embedding_index.json").read_text(encoding="utf-8")))}
    X = E[[row[r["text_sha256"]] for r in corpus]]
    lab = np.array([r["label"] for r in corpus])
    T = [texts[r["text_sha256"]] for r in corpus]
    A = np.nonzero(lab == 1)[0]; Bn = np.nonzero(lab == 0)[0]
    S = X[A] @ X[Bn].T
    best = S.argmax(1); cos = S.max(1)
    cand = []
    for k, ai in enumerate(A):
        if cos[k] < 0.80:
            continue
        bi = Bn[best[k]]
        ta, tb = T[ai], T[bi]
        la, lb = len(ta.split()), len(tb.split())
        ja = toks(ta); jb = toks(tb)
        ra, rb = corpus[ai]["source"], corpus[bi]["source"]
        cand.append({
            "attack_row": corpus[ai]["i"], "benign_row": corpus[bi]["i"],
            "attack_source": ra, "benign_source": rb,
            "same_register": REGISTER.get(ra, ra) == REGISTER.get(rb, rb),
            "cosine": round(float(cos[k]), 4),
            "length_ratio": round(la / max(lb, 1), 3),
            "token_jaccard": round(len(ja & jb) / max(len(ja | jb), 1), 3),
            "harm_vocab_diff": len(harm.findall(ta)) - len(harm.findall(tb)),
            "attack_split": corpus[ai]["split"], "benign_split": corpus[bi]["split"],
            "attack_row_review_required": ra in REVIEW_REQUIRED,
            "attack_row_is_refusal": bool(REFUSAL.search(ta)),
        })
    reuse = collections.Counter(c["benign_row"] for c in cand)
    for c in cand:
        c["benign_partner_reuse"] = reuse[c["benign_row"]]
    screens = lambda c: (0.67 <= c["length_ratio"] <= 1.5 and c["token_jaccard"] >= 0.4 and abs(c["harm_vocab_diff"]) <= 1)
    passing = [c for c in cand if screens(c)]
    summary = {
        "n_candidates": len(cand),
        "by_attack_source": dict(collections.Counter(c["attack_source"] for c in cand).most_common()),
        "by_benign_source": dict(collections.Counter(c["benign_source"] for c in cand).most_common()),
        "same_register": sum(c["same_register"] for c in cand),
        "unique_benign_partners": len(reuse), "max_benign_partner_reuse": max(reuse.values()) if reuse else 0,
        "benign_partners_reused_ge_3": sum(1 for v in reuse.values() if v >= 3),
        "attack_side_review_required": sum(c["attack_row_review_required"] for c in cand),
        "attack_side_refusal": sum(c["attack_row_is_refusal"] for c in cand),
        "pass_preregistered_surface_screens": len(passing),
        "pass_screens_attack_side_not_review_required": sum(not c["attack_row_review_required"] for c in passing),
        "pass_screens_unique_benign_partners": len({c["benign_row"] for c in passing}),
        "screen_definition": "0.67<=length_ratio<=1.5 and token_jaccard>=0.4 and |harm_vocab_diff|<=1 (C2_PHASE1_DESIGN.md 4.2-4.3)",
        "cosine_quantiles": {q: float(np.quantile([c["cosine"] for c in cand], q)) for q in (0.1, 0.5, 0.9)},
        "token_jaccard_median": float(np.median([c["token_jaccard"] for c in cand])),
        "harm_vocab_diff_distribution": dict(sorted(collections.Counter(c["harm_vocab_diff"] for c in cand).items())),
        "note": "Nearest-neighbour similarity is a candidate filter only. None of these is a verified intent contrast until reviewed.",
    }
    out = {"summary": summary, "candidates": cand}
    with open(HERE / "audit/phase1_1_pair_candidates.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
