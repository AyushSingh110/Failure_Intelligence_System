"""
E-R1 Part B — NEI construction v3 (Option A; manifest/PROTOCOL_v3.md).

    python -I research/reliability/e_r1_part_b/fever_nei_v3.py

Sampling, seeds, claims, labels and role assignments are those of fever_materialize/2
(imported, not re-implemented). Gold items are copied byte-for-byte from the v2
derived files. Only NOT ENOUGH INFO passages are rebuilt:

  1. Spans: every 1..6-token claim span starting with a capitalised token (v2 rule).
  2. Candidates: every dump page whose title equals a span OR a span followed by a
     disambiguator "_-LRB-...-RRB-" (e.g. Mud_-LRB-2012_film-RRB-). One streaming
     pass over all titles in wiki-pages.zip.
  3. Disambiguation pages ("... may refer to", "-LRB-disambiguation-RRB-") are not
     candidates.
  4. Lead = first two NON-ARTEFACT sentences. Artefacts (skipped, never edited):
     empty lines; image residue (starts with thumb/left/right/upright/frame then "|").
     Skipped indices are recorded.
  5. Score = |content(claim) - tokens(span)| ∩ content(lead + disambiguator)|, with a
     fixed stop-word list and a fixed suffix stemmer. Highest score wins; ties ->
     longer span, then undisambiguated title, then title order.
  6. Score 0 -> NO DEFENSIBLE PAGE: item excluded and counted. Never forced.

Outputs: data/fever/derived_v3/ (git-ignored; v2 files untouched) and
e_r1_part_b/manifest/sample_manifest_v3.json (tracked; no text).
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fever_materialize as v2   # noqa: E402

VERSION = "fever_nei/3"
REPO = v2.REPO
V2_DIR = v2.DERIVED
OUT_DIR = REPO / "research/reliability/data/fever/derived_v3"
MANIFEST = REPO / "research/reliability/e_r1_part_b/manifest/sample_manifest_v3.json"
LEAD_SENTENCES = 2
LEAD_LINES_KEPT = 8          # lines stored per candidate page (enough to skip artefacts)

STOP = frozenset("""a an the and or but if of to in on at by for with from as is are was were be been being
has have had do does did not no it its this that these those he she they his her their him them who whom
which what when where why how than then so such there here also only just more most very can could
would should will may might must shall about into over under after before between during through out up
down off again further once all any both each few other some own same too s t one two""".split())
_DISAMBIG = re.compile(r"_-LRB-(.+)-RRB-$")
_ARTEFACT = re.compile(r"^\s*(thumb|left|right|upright|frame)\b[^|]*\|", re.IGNORECASE)
_DISAMBIG_PAGE = re.compile(r"\bmay (also )?refer to\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content(text: str) -> set[str]:
    return {stem(w) for w in _WORD.findall(v2.unescape(text).lower()) if w not in STOP}


def spans_with_tokens(claim: str) -> list[tuple[str, int]]:
    """(title key, token count) for every capitalised-start span, v2 tokenisation."""
    toks = [t.rstrip(".,;:!?").removesuffix("'s") for t in v2._TOKEN.findall(claim)]
    toks = [t for t in toks if t]
    out = []
    for i, t in enumerate(toks):
        if t[0].isupper():
            for n in range(1, v2.MAX_SPAN + 1):
                if i + n <= len(toks):
                    out.append((v2.key("_".join(toks[i:i + n])), n))
    return out


def lead(lines: dict[int, str]) -> tuple[list[tuple[int, str]], list[int]]:
    kept, skipped = [], []
    for i, s in sorted(lines.items()):
        if not s.strip() or _ARTEFACT.match(s):
            skipped.append(i)
            continue
        kept.append((i, s))
        if len(kept) == LEAD_SENTENCES:
            break
    return kept, skipped


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    # identical selection to v2
    dev, test = v2.load("paper_dev.jsonl"), v2.load("paper_test.jsonl")
    dev_k, _ = v2.dedupe(dev)
    test_k, _ = v2.dedupe(test)
    nei = {"smoke": [], "calibration": [], "test": []}
    label = "NOT ENOUGH INFO"
    pool = v2.sample(dev_k, label, v2.QUOTAS["smoke"] + v2.QUOTAS["calibration"], "paper_dev")
    nei["smoke"], nei["calibration"] = pool[: v2.QUOTAS["smoke"]], pool[v2.QUOTAS["smoke"]:]
    nei["test"] = v2.sample(test_k, label, v2.QUOTAS["test"], "paper_test")

    span_keys = {k for rows in nei.values() for r in rows for k, _ in spans_with_tokens(r["claim"])}

    # one pass over every title: keep exact or disambiguated matches of any span
    cands: dict[str, list[dict]] = collections.defaultdict(list)
    with zipfile.ZipFile(v2.RAW / "wiki-pages.zip") as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if not (info.filename.startswith("wiki-pages/") and info.filename.endswith(".jsonl")):
                continue
            with zf.open(info) as fh:
                for line in fh:
                    rec = json.loads(line)
                    k = v2.key(rec["id"])
                    m = _DISAMBIG.search(k)
                    base = k[: m.start()] if m else k
                    if base in span_keys:
                        lines = v2.parse_lines(rec.get("lines", ""))
                        cands[base].append({"title": k, "source_id": rec["id"],
                                            "disambig": m.group(1) if m else "",
                                            "lines": {i: s for i, s in sorted(lines.items())[:LEAD_LINES_KEPT]}})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    built, excluded, decisions = {}, [], {}
    for role, rows in nei.items():
        for r in rows:
            item_id = f"{role}-{r['id']}"
            claim_c = content(r["claim"])
            best, considered = None, 0
            for span, ntok in spans_with_tokens(r["claim"]):
                span_c = content(span.replace("_", " "))
                for c in cands.get(span, []):
                    if "disambiguation" in c["disambig"].lower():
                        continue
                    sents, skipped = lead(c["lines"])
                    if not sents or _DISAMBIG_PAGE.search(sents[0][1]):
                        continue
                    considered += 1
                    page_c = content(" ".join(s for _, s in sents) + " " + c["disambig"].replace("_", " "))
                    score = len((claim_c - span_c) & page_c)
                    rank = (score, ntok, c["disambig"] == "", [-ord(ch) for ch in c["title"]])
                    if best is None or rank > best[0]:
                        best = (rank, c, sents, skipped, span)
            if best is None or best[0][0] == 0:
                excluded.append({"item_id": item_id, "fever_id": r["id"], "role": role,
                                 "reason": "no_candidate" if best is None else "no_defensible_page_overlap_0",
                                 "candidates_considered": considered})
                continue
            rank, c, sents, skipped, span = best
            built[item_id] = {
                "item_id": item_id, "fever_id": r["id"], "role": role, "claim": r["claim"],
                "evidence_kind": "related_not_gold", "related_page": c["title"],
                "related_page_source_id": c["source_id"], "matched_span": span,
                "passage": [{"page": c["source_id"], "sentence_index": i, "text": s} for i, s in sents],
                "artefact_lines_skipped": skipped, "selection_score": rank[0],
                "candidates_considered": considered, "all_gold_sets": [],
                "passage_text": " ".join(v2.unescape(s) for _, s in sents),
                "primary_set_sentences": len(sents), "primary_set_pages": 1,
                "construction": VERSION,
            }
            decisions[item_id] = c["title"]

    # write: gold items copied verbatim from v2, NEI items rebuilt; order and label isolation as v2
    counts, changed = collections.Counter(), collections.Counter()
    for name in ("smoke.jsonl", "calibration.jsonl", "test_sealed.jsonl"):
        src = (V2_DIR / name).read_text(encoding="utf-8").splitlines()
        v2_items = {json.loads(l)["item_id"]: l for l in src}
        role = {"test_sealed.jsonl": "test"}.get(name, name.split(".")[0])
        out_lines = []
        order = sorted({*v2_items, *(k for k, v in built.items() if v["role"] == role)},
                       key=lambda k: int(k.rsplit("-", 1)[1]))
        for iid in order:
            if iid in built:
                item = dict(built[iid])
                if role != "test":
                    item["label"] = label
                out_lines.append(json.dumps(item, ensure_ascii=False))
                old = v2_items.get(iid)
                changed["nei_new_item" if old is None else
                        ("nei_same_page" if json.loads(old).get("related_page") == item["related_page"]
                         else "nei_page_changed")] += 1
                counts[(role, "NEI")] += 1
            else:
                it = json.loads(v2_items[iid])
                if it["evidence_kind"] == "gold":
                    out_lines.append(v2_items[iid])          # byte-identical
                    counts[(role, "gold")] += 1
                else:
                    changed["nei_dropped_by_v3"] += 1
        (OUT_DIR / name).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    # labels file: v2 labels for every item still present
    present = {json.loads(l)["item_id"] for l in (OUT_DIR / "test_sealed.jsonl").read_text(encoding="utf-8").splitlines()}
    lab_lines = [l for l in (V2_DIR / "test_sealed_labels.jsonl").read_text(encoding="utf-8").splitlines()
                 if json.loads(l)["item_id"] in present]
    new_test_nei = [k for k in present if k in built and k not in
                    {json.loads(l)["item_id"] for l in lab_lines}]
    lab_lines += [json.dumps({"item_id": k, "label": label}) for k in sorted(new_test_nei)]
    (OUT_DIR / "test_sealed_labels.jsonl").write_text("\n".join(lab_lines) + "\n", encoding="utf-8")

    files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size,
                      "rows": sum(1 for _ in p.open(encoding="utf-8"))} for p in sorted(OUT_DIR.glob("*.jsonl"))}
    manifest = {
        "schema": "e-r1-partB-sample/3", "construction": VERSION, "script_sha256": sha(Path(__file__)),
        "base_selection": "fever_materialize/2 (imported)", "v2_script_sha256": sha(Path(v2.__file__)),
        "archive_sha256": sha(v2.RAW / "wiki-pages.zip"),
        "span_keys": len(span_keys), "candidate_bases_found": len(cands),
        "candidate_pages_total": sum(len(v) for v in cands.values()),
        "counts": {f"{r}|{k}": v for (r, k), v in sorted(counts.items())},
        "nei_change_vs_v2": dict(changed),
        "excluded": excluded,
        "excluded_counts": dict(collections.Counter(f"{e['role']}|{e['reason']}" for e in excluded)),
        "derived_files": files,
        "rules": {"lead_sentences": LEAD_SENTENCES, "stop_words": len(STOP), "stemmer": "strip ing/ed/es/s if len>4",
                  "artefact_regex": _ARTEFACT.pattern, "disambiguation_page_regex": _DISAMBIG_PAGE.pattern,
                  "tie_break": "score, span tokens, undisambiguated, title order"},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("candidate_bases_found", "candidate_pages_total", "counts",
                                               "nei_change_vs_v2", "excluded_counts")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
