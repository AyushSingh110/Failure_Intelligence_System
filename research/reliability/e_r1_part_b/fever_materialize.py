"""
E-R1 Part B, Phases 3–4 — deterministic sample and bounded evidence extraction.

    python -I research/reliability/e_r1_part_b/fever_materialize.py

Inputs (git-ignored, research/reliability/data/fever/raw/): paper_dev.jsonl,
paper_test.jsonl, wiki-pages.zip (Zenodo 10.5281/zenodo.4925954).

Procedure (preregistered in manifest/PROTOCOL.md, fixed before any passage was read):
  1. Dedupe: within each split, drop EVERY row whose normalised claim text occurs
     more than once (no keep-first: duplicates may disagree on label).
  2. Sample, seed 20261027, rows sorted by FEVER id, per label:
       paper_dev  -> smoke 10 + calibration 120   (smoke excluded from calibration)
       paper_test -> sealed test 280
  3. Gold evidence (SUPPORTS / REFUTES): every evidence set is kept; the PRIMARY set
     shown to models is the first set with the fewest sentences (ties: listed order).
  4. NOT ENOUGH INFO: a related, non-gold passage = first two non-empty sentences of
     the page whose title equals the LONGEST span (1..6 tokens) of the claim that
     starts with a capitalised token; ties -> leftmost. Marked related_not_gold.
  5. One streaming pass over wiki-pages.zip keeps only pages referenced by step 3 or
     matched as candidates by step 4. Nothing else is extracted.
  6. Any unresolved page or sentence index, or empty sentence, is a data-integrity
     failure: the item is EXCLUDED and counted, never re-sampled, never filled from
     live Wikipedia.

Outputs:
  data/fever/derived/{smoke,calibration}.jsonl     items with labels
  data/fever/derived/test_sealed.jsonl            items WITHOUT labels
  data/fever/derived/test_sealed_labels.jsonl     labels, read only by the sealed scorer
  e_r1_part_b/manifest/sample_manifest.json       hashes, counts, exclusions (tracked; no text)
Standard library only. No network.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

VERSION = "fever_materialize/2"   # /2: NFC title matching (see key())
SEED = 20261027
QUOTAS = {"smoke": 10, "calibration": 120, "test": 280}
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
MAX_SPAN, NEI_SENTENCES = 6, 2

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "research/reliability/data/fever/raw"
DERIVED = REPO / "research/reliability/data/fever/derived"
MANIFEST = REPO / "research/reliability/e_r1_part_b/manifest/sample_manifest.json"

# FEVER's PTB-style escapes in titles and sentences -> plain text (published mapping)
ESCAPES = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}",
           "-COLON-": ":", "``": '"', "''": '"'}


def unescape(text: str) -> str:
    for k, v in ESCAPES.items():
        text = text.replace(k, v)
    return re.sub(r"\s+([.,;:!?)\]}'])", r"\1", re.sub(r"([(\[{])\s+", r"\1", text)).strip()


def key(title: str) -> str:
    """Page-title match key. The claim files spell titles in NFD (decomposed accents),
    the dump in NFC; matching on NFC of both sides joins them. Identifiers are kept as written."""
    return unicodedata.normalize("NFC", title)


def norm_claim(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (RAW / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def dedupe(rows: list[dict]) -> tuple[list[dict], int]:
    counts = collections.Counter(norm_claim(r["claim"]) for r in rows)
    kept = [r for r in rows if counts[norm_claim(r["claim"])] == 1]
    return kept, len(rows) - len(kept)


def sample(rows: list[dict], label: str, n: int, salt: str) -> list[dict]:
    pool = sorted((r for r in rows if r["label"] == label), key=lambda r: r["id"])
    return random.Random(f"{SEED}:{salt}:{label}").sample(pool, n)


_TOKEN = re.compile(r"[A-Za-z0-9][\w'\-\.&]*")


def candidate_titles(claim: str) -> list[str]:
    """Spans starting with a capitalised token, longest first, then leftmost."""
    toks = [t.rstrip(".,;:!?").removesuffix("'s") for t in _TOKEN.findall(claim)]
    toks = [t for t in toks if t]
    spans = []
    for i, t in enumerate(toks):
        if not t[0].isupper():
            continue
        for n in range(1, MAX_SPAN + 1):
            if i + n <= len(toks):
                spans.append((n, -i, "_".join(toks[i:i + n])))
    return [key(s) for _, _, s in sorted(spans, reverse=True)]


def evidence_sets(row: dict) -> list[list[tuple[str, int]]]:
    return [[(key(it[2]) if it[2] else it[2], it[3]) for it in s] for s in row["evidence"]]


def primary_set(sets: list[list[tuple[str, int]]]) -> list[tuple[str, int]]:
    return min(sets, key=len)   # min() keeps the first among equals


def parse_lines(raw: str) -> dict[int, str]:
    out = {}
    for line in raw.split("\n"):
        parts = line.split("\t")
        if parts and parts[0].isdigit():
            out[int(parts[0])] = parts[1] if len(parts) > 1 else ""
    return out


def main() -> int:
    dev, test = load("paper_dev.jsonl"), load("paper_test.jsonl")
    dev_k, dev_drop = dedupe(dev)
    test_k, test_drop = dedupe(test)

    chosen = {"smoke": [], "calibration": [], "test": []}
    for label in LABELS:
        pool = sample(dev_k, label, QUOTAS["smoke"] + QUOTAS["calibration"], "paper_dev")
        chosen["smoke"] += pool[:QUOTAS["smoke"]]
        chosen["calibration"] += pool[QUOTAS["smoke"]:]
        chosen["test"] += sample(test_k, label, QUOTAS["test"], "paper_test")

    gold_pages, nei_cands = set(), set()
    for rows in chosen.values():
        for r in rows:
            if r["label"] == "NOT ENOUGH INFO":
                nei_cands.update(candidate_titles(r["claim"]))
            else:
                gold_pages.update(p for s in evidence_sets(r) for p, _ in s)

    zpath = RAW / "wiki-pages.zip"
    wanted = gold_pages | nei_cands
    pages: dict[str, dict[int, str]] = {}
    members, skipped = 0, []
    with zipfile.ZipFile(zpath) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            # The archive also carries macOS __MACOSX/._*.jsonl resource forks
            # (binary, not JSONL) and license.html; only wiki-pages/*.jsonl is data.
            if not (info.filename.startswith("wiki-pages/") and info.filename.endswith(".jsonl")):
                skipped.append(info.filename)
                continue
            members += 1
            with zf.open(info) as fh:
                for line in fh:
                    rec = json.loads(line)
                    k = key(rec["id"])
                    if k in wanted:
                        pages[k] = parse_lines(rec.get("lines", ""))

    DERIVED.mkdir(parents=True, exist_ok=True)
    excluded, counts, hashes = [], collections.Counter(), {}
    shared_pages = ({p for r in dev for s in evidence_sets(r) for p, _ in s if p}
                    & {p for r in test for s in evidence_sets(r) for p, _ in s if p})
    files = {role: (DERIVED / f"{'test_sealed' if role == 'test' else role}.jsonl").open("w", encoding="utf-8")
             for role in chosen}
    labels_fh = (DERIVED / "test_sealed_labels.jsonl").open("w", encoding="utf-8")
    for role, rows in chosen.items():
        for r in sorted(rows, key=lambda r: r["id"]):
            item = {"item_id": f"{role}-{r['id']}", "fever_id": r["id"], "role": role, "claim": r["claim"]}
            if r["label"] == "NOT ENOUGH INFO":
                title = next((t for t in candidate_titles(r["claim"]) if t in pages), None)
                sents = [(i, s) for i, s in sorted(pages.get(title, {}).items()) if s.strip()][:NEI_SENTENCES]
                if not title or len(sents) < 1:
                    excluded.append({"fever_id": r["id"], "role": role, "label": r["label"],
                                     "reason": "no_related_page" if not title else "related_page_empty"})
                    continue
                item.update(evidence_kind="related_not_gold", related_page=title,
                            passage=[{"page": title, "sentence_index": i, "text": s} for i, s in sents])
                item["all_gold_sets"] = []
            else:
                sets = evidence_sets(r)
                bad = [(p, i) for s in sets for p, i in s
                       if p not in pages or i not in pages[p] or not pages[p][i].strip()]
                prim = primary_set(sets)
                if any(x in bad for x in prim):
                    excluded.append({"fever_id": r["id"], "role": role, "label": r["label"],
                                     "reason": "unresolved_gold_evidence",
                                     "unresolved": [list(x) for x in bad if x in prim]})
                    continue
                item.update(evidence_kind="gold",
                            passage=[{"page": p, "sentence_index": i, "text": pages[p][i]} for p, i in prim],
                            all_gold_sets=[[list(x) for x in s] for s in sets],
                            unresolved_in_other_sets=len([x for x in bad if x not in prim]),
                            page_shared_across_splits=any(p in shared_pages for p, _ in prim))
            item["passage_text"] = " ".join(unescape(p["text"]) for p in item["passage"])
            item["primary_set_sentences"] = len(item["passage"])
            item["primary_set_pages"] = len({p["page"] for p in item["passage"]})
            if role == "test":
                labels_fh.write(json.dumps({"item_id": item["item_id"], "label": r["label"]}) + "\n")
            else:
                item["label"] = r["label"]
            files[role].write(json.dumps(item, ensure_ascii=False) + "\n")
            counts[(role, r["label"])] += 1
    for fh in (*files.values(), labels_fh):
        fh.close()
    for p in sorted(DERIVED.glob("*.jsonl")):
        hashes[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size,
                          "rows": sum(1 for _ in p.open(encoding="utf-8"))}

    manifest = {
        "schema": "e-r1-partB-sample/1",
        "script": VERSION,
        "script_sha256": sha256_file(Path(__file__)),
        "seed": SEED,
        "quotas_per_label": QUOTAS,
        "sources": {n: {"sha256": sha256_file(RAW / n), "bytes": (RAW / n).stat().st_size}
                    for n in ("paper_dev.jsonl", "paper_test.jsonl", "wiki-pages.zip")},
        "dedupe_rows_dropped": {"paper_dev": dev_drop, "paper_test": test_drop},
        "zip_members_scanned": members,
        "zip_members_skipped": {"count": len(skipped),
                                "macosx_resource_forks": sum(1 for s in skipped if s.startswith("__MACOSX")),
                                "other": [s for s in skipped if not s.startswith("__MACOSX")]},
        "pages_requested": {"gold": len(gold_pages), "nei_candidates": len(nei_cands)},
        "pages_extracted": len(pages),
        "gold_pages_missing_from_archive": sorted(gold_pages - set(pages))[:50],
        "kept_counts": {f"{role}|{lab}": counts[(role, lab)] for role in QUOTAS for lab in LABELS},
        "excluded": excluded,
        "excluded_counts": dict(collections.Counter(f"{e['role']}|{e['label']}|{e['reason']}" for e in excluded)),
        "derived_files": hashes,
        "escape_mapping": ESCAPES,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("dedupe_rows_dropped", "zip_members_scanned", "pages_requested",
                                               "pages_extracted", "kept_counts", "excluded_counts")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
