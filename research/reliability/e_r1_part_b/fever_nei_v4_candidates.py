"""
E-R1 Part B — v4 NEI candidate generation and blind review packets (PROTOCOL_v4.md §3–§6).

    python -I research/reliability/e_r1_part_b/fever_nei_v4_candidates.py

Candidates are PROPOSALS for a human reviewer; nothing here accepts or rejects a passage.
Never opens test_sealed_labels.jsonl. One streaming pass over wiki-pages.zip.

Outputs (git-ignored, research/reliability/data/fever/review_v4/):
  packet_<role>.csv / packet_<role>.html   reviewer-facing (no ids, no ranks, no labels)
  second_review_packet.csv / .html          optional 50-claim second review
  mapping_<role>.json                       PRIVATE: review id -> FEVER id, candidates' provenance
  REVIEW_INSTRUCTIONS.md                    how to fill the packet
Tracked (no text): e_r1_part_b/manifest/nei_v4_candidates_manifest.json
"""
from __future__ import annotations

import collections
import csv
import hashlib
import html
import json
import random
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fever_materialize as v2   # noqa: E402
import fever_nei_v3 as v3        # noqa: E402  (lead(), content(), artefact regexes)

VERSION = "fever_nei_v4_candidates/1"
REPO = v2.REPO
OUT = REPO / "research/reliability/data/fever/review_v4"
MANIFEST = REPO / "research/reliability/e_r1_part_b/manifest/nei_v4_candidates_manifest.json"
MAX_CANDIDATES, LINES_KEPT = 5, 12
LETTERS = "ABCDE"
ANNOTATION_FIELDS = ["selected_candidate_id", "correct_entity_or_subject", "accidental_support",
                     "accidental_refutation", "wrong_sense_or_entity", "unusable_passage",
                     "insufficiently_relevant", "reviewer_confidence", "rationale",
                     "adjudication_status", "reviewer_id"]

# fixed type-cue table: disambiguator words <- claim words (PROTOCOL_v4 §3.4)
TYPE_CUES = {
    "film":   ({"film", "movie"}, {"film", "filmed", "movie", "directed", "director", "star", "stars", "starring",
                                   "starred", "cast", "actor", "actress", "box", "screenplay", "premiered"}),
    "music":  ({"album", "song", "single", "band", "musician", "singer", "rapper", "ep"},
               {"album", "song", "single", "band", "music", "musician", "singer", "record", "records",
                "recorded", "released", "chart", "charts", "tour", "rapper", "ep", "lyrics"}),
    "tv":     ({"tv", "series", "episode", "season", "show", "miniseries"},
               {"series", "episode", "episodes", "season", "television", "tv", "aired", "sitcom", "show"}),
    "book":   ({"novel", "book", "comics", "comic", "manga", "magazine", "poem"},
               {"novel", "book", "written", "author", "published", "wrote", "comic", "comics"}),
    "play":   ({"play", "musical", "opera"}, {"play", "theatre", "theater", "stage", "musical", "opera", "broadway"}),
    "game":   ({"video", "game"}, {"game", "games", "video", "console"}),
    "sports": ({"footballer", "team", "club", "basketball", "baseball", "cricketer", "wrestler"},
               {"team", "club", "league", "played", "plays", "football", "season", "championship"}),
    "place":  ({"city", "town", "county", "state", "river", "village", "province"},
               {"city", "town", "located", "capital", "population", "river", "village", "state"}),
}
_ID_NUM = re.compile(r"(\d+)$")


def cue_types(words: set[str], side: int) -> set[str]:
    return {t for t, sets in TYPE_CUES.items() if words & sets[side]}


def matched_maximal_spans(claim: str, available: set[str]) -> list[tuple[str, int, int]]:
    """(title key, start, length) for matched spans not strictly inside another matched span."""
    toks = [t.rstrip(".,;:!?").removesuffix("'s") for t in v2._TOKEN.findall(claim)]
    toks = [t for t in toks if t]
    spans = []
    for i, t in enumerate(toks):
        if t[0].isupper():
            for n in range(1, v2.MAX_SPAN + 1):
                if i + n <= len(toks):
                    k = v2.key("_".join(toks[i:i + n]))
                    if k in available:
                        spans.append((k, i, n))
    maximal = [s for s in spans if not any(o is not s and o[1] <= s[1] and s[1] + s[2] <= o[1] + o[2]
                                           and o[2] > s[2] for o in spans)]
    return sorted(set(maximal), key=lambda s: (-s[2], s[1]))


def artefact_notes(sents, skipped, title_disambig: str) -> list[str]:
    notes = []
    if skipped:
        notes.append(f"lines skipped as markup: {skipped}")
    first = sents[0][1].strip() if sents else ""
    if first and first[0].islower():
        notes.append("lead sentence lacks its subject (dump strips the bold title)")
    if re.match(r"^(For |This article is about|Not to be confused)", first):
        notes.append("hatnote")
    if any(s.rstrip().endswith(":") for _, s in sents):
        notes.append("ends with a list colon (list items not shown)")
    if re.search(r"\[[^\]]*[ˈˌəɪʊæɛɔʃʒθðŋ][^\]]*\]", " ".join(s for _, s in sents)):
        notes.append("contains IPA pronunciation")
    return notes


def windows(lines: dict[int, str]) -> tuple[tuple[list, list], tuple[list, list]]:
    lead, skipped = v3.lead(lines)
    used = {i for i, _ in lead} | set(skipped)
    rest = {i: s for i, s in lines.items() if i not in used and (not lead or i > lead[-1][0])}
    later, skipped2 = v3.lead(rest)
    return (lead, skipped), (later, skipped2)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    if "test_sealed_labels" in " ".join(sys.argv):
        raise SystemExit("refusing: this step never reads the sealed label file")
    dev, test = v2.load("paper_dev.jsonl"), v2.load("paper_test.jsonl")
    dev_k, _ = v2.dedupe(dev)
    test_k, _ = v2.dedupe(test)
    lab = "NOT ENOUGH INFO"
    pool = v2.sample(dev_k, lab, v2.QUOTAS["smoke"] + v2.QUOTAS["calibration"], "paper_dev")
    claims = {"smoke": pool[: v2.QUOTAS["smoke"]], "calibration": pool[v2.QUOTAS["smoke"]:],
              "test": v2.sample(test_k, lab, v2.QUOTAS["test"], "paper_test")}
    assert all(r["label"] == lab for rows in claims.values() for r in rows)

    # all spans of all claims (any length), to learn which bases exist in the dump
    span_keys = {k for rows in claims.values() for r in rows for k, _ in v3.spans_with_tokens(r["claim"])}
    pages: dict[str, list[dict]] = collections.defaultdict(list)
    with zipfile.ZipFile(v2.RAW / "wiki-pages.zip") as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if not (info.filename.startswith("wiki-pages/") and info.filename.endswith(".jsonl")):
                continue
            with zf.open(info) as fh:
                for line in fh:
                    rec = json.loads(line)
                    k = v2.key(rec["id"])
                    m = v3._DISAMBIG.search(k)
                    base = k[: m.start()] if m else k
                    if base in span_keys:
                        lines = v2.parse_lines(rec.get("lines", ""))
                        pages[base].append({"title": k, "source_id": rec["id"], "disambig": m.group(1) if m else "",
                                            "lines": dict(sorted(lines.items())[:LINES_KEPT])})
    available = set(pages)

    OUT.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    per_role_ncand = {r: collections.Counter() for r in claims}
    reasons_count = collections.Counter()
    order_rng_seed = "20261027:v4-order:{}"
    cand_rng = random.Random("20261027:v4-cand")
    second_pool = []
    blank_hashes = {}
    for role, rows in claims.items():
        records = []
        for r in sorted(rows, key=lambda r: r["id"]):
            claim_words = set(re.findall(r"[a-z0-9]+", r["claim"].lower()))
            want_types = cue_types(claim_words, 1)
            per_span = []
            for k, start, n in matched_maximal_spans(r["claim"], available):
                opts = []
                for pg in pages[k]:
                    if "disambiguation" in pg["disambig"].lower():
                        continue
                    (lead, sk), (later, sk2) = windows(pg["lines"])
                    if not lead or v3._DISAMBIG_PAGE.search(lead[0][1]):
                        continue
                    dis_words = set(re.findall(r"[a-z0-9]+", pg["disambig"].lower().replace("_", " ")))
                    cue = bool(cue_types(dis_words, 0) & want_types)
                    opts.append((not cue, pg["disambig"] != "", pg["title"], pg, lead, sk, later, sk2, cue))
                opts.sort(key=lambda o: o[:3])
                if opts:
                    per_span.append((k, n, opts))
            # round-robin over maximal spans (longest first): every span's page 1 lead, with
            # span 1's later window right after its lead; then every span's page 2; ...
            cands = []
            for depth in range(10):
                for si, (k, n, opts) in enumerate(per_span):
                    if depth < len(opts):
                        cands.append((f"span{si + 1}_page{depth + 1}_lead", k, n, opts[depth], "lead"))
                        if si == 0 and depth == 0 and opts[0][6]:
                            cands.append(("span1_page1_later_window", k, n, opts[0], "later"))
            seen, uniq = set(), []
            for c in cands:
                sig = (c[3][2], c[4])
                if sig not in seen:
                    seen.add(sig)
                    uniq.append(c)
            cands = uniq[:MAX_CANDIDATES]
            per_role_ncand[role][len(cands)] += 1
            built = []
            for rank, (reason, span, n, opt, which) in enumerate(cands, 1):
                pg = opt[3]
                sents, skipped = (opt[4], opt[5]) if which == "lead" else (opt[6], opt[7])
                built.append({
                    "rank": rank, "generation_reason": reason, "matched_span": span, "span_tokens": n,
                    "type_cue_match": opt[8], "title": pg["title"], "page_id": pg["source_id"],
                    "window": which, "sentence_indices": [i for i, _ in sents],
                    "raw_sentences": [s for _, s in sents],
                    "display_text": " ".join(v2.unescape(s) for _, s in sents),
                    "artefact_notes": artefact_notes(sents, skipped, pg["disambig"]),
                    "overlap_metadata_only": len((v3.content(r["claim"]) - v3.content(span.replace("_", " ")))
                                                 & v3.content(" ".join(s for _, s in sents))),
                })
                reasons_count[reason] += 1
            letters = list(LETTERS[: len(built)])
            cand_rng.shuffle(letters)
            for c, L in zip(built, letters):
                c["display_letter"] = L
            records.append({"fever_id": r["id"], "item_id": f"{role}-{r['id']}", "claim": r["claim"],
                            "claim_sha256": hashlib.sha256(r["claim"].encode()).hexdigest(),
                            "candidates": sorted(built, key=lambda c: c["display_letter"])})
            stats[f"{role}|claims"] += 1
            stats[f"{role}|no_candidate"] += (len(built) == 0)
        random.Random(order_rng_seed.format(role)).shuffle(records)
        for i, rec in enumerate(records, 1):
            rec["review_id"] = f"R4-{role}-{i:03d}"
        if role in ("smoke", "calibration"):
            second_pool += records
        write_packet(role, records)
        blank_hashes[f"packet_{role}.csv"] = sha(OUT / f"packet_{role}.csv")
        (OUT / f"mapping_{role}.json").write_text(json.dumps(
            {"PRIVATE": "do not give to the reviewer", "construction": VERSION, "role": role,
             "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")

    second = random.Random("20261027:v4-second").sample(sorted(second_pool, key=lambda r: r["review_id"]), 50)
    write_packet("second_review", second, name="second_review_packet")
    blank_hashes["second_review_packet.csv"] = sha(OUT / "second_review_packet.csv")

    manifest = {
        "schema": "e-r1-partB-nei-v4-candidates/1", "construction": VERSION,
        "script_sha256": sha(Path(__file__)), "protocol_v4_sha256": sha(MANIFEST.parent / "PROTOCOL_v4.md"),
        "archive_sha256": sha(v2.RAW / "wiki-pages.zip"),
        "sealed_label_file_opened": False,
        "claims_per_role": {r: len(v) for r, v in claims.items()},
        "candidate_count_distribution": {r: dict(sorted(c.items())) for r, c in per_role_ncand.items()},
        "claims_with_no_candidate": {r: stats[f"{r}|no_candidate"] for r in claims},
        "generation_reason_counts": dict(reasons_count),
        "blank_packet_sha256": blank_hashes,
        "mapping_sha256": {f"mapping_{r}.json": sha(OUT / f"mapping_{r}.json") for r in claims},
        "second_review_ids": [r["review_id"] for r in second],
        "seeds": {"order": "20261027:v4-order:<role>", "candidate_letters": "20261027:v4-cand",
                  "second_review": "20261027:v4-second"},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("claims_per_role", "candidate_count_distribution",
                                               "claims_with_no_candidate", "generation_reason_counts")}, indent=1))
    return 0


def write_packet(role: str, records: list[dict], name: str | None = None) -> None:
    name = name or f"packet_{role}"
    head = ["review_id", "claim"]
    for L in LETTERS:
        head += [f"{L}_title", f"{L}_page_id", f"{L}_passage", f"{L}_artefact_notes"]
    head += ANNOTATION_FIELDS
    with (OUT / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for rec in records:
            row = [rec["review_id"], rec["claim"]]
            by = {c["display_letter"]: c for c in rec["candidates"]}
            for L in LETTERS:
                c = by.get(L)
                row += ([c["title"].replace("_", " ").replace("-LRB-", "(").replace("-RRB-", ")"), c["page_id"],
                         c["display_text"], "; ".join(c["artefact_notes"])] if c else ["", "", "", ""])
            row += [""] * len(ANNOTATION_FIELDS)
            w.writerow(row)
    parts = [f"<!doctype html><meta charset='utf-8'><title>{name}</title><style>body{{font:15px/1.45 system-ui,sans-serif;"
             "max-width:900px;margin:24px auto;padding:0 16px}}.c{{border:1px solid #ccc;border-radius:6px;padding:8px 10px;"
             "margin:6px 0}}.n{{color:#a33;font-size:13px}}h2{{margin-top:32px}}.claim{{font-size:17px;background:#f4f4f4;"
             "padding:8px}}</style>",
             f"<h1>{name}</h1><p>Read-only view. Record decisions in <b>{name}.csv</b>. "
             "Every claim is a FEVER NOT ENOUGH INFO claim. Passages are <b>related passages, not FEVER gold evidence</b>. "
             "Choose the one candidate about the claim's intended subject that does not by itself settle the claim, "
             "or NONE. Letters are in random order and carry no ranking.</p>"]
    for rec in records:
        parts.append(f"<h2>{rec['review_id']}</h2><div class='claim'>{html.escape(rec['claim'])}</div>")
        if not rec["candidates"]:
            parts.append("<p><i>No candidate could be generated: select NONE.</i></p>")
        for c in rec["candidates"]:
            t = c["title"].replace("_", " ").replace("-LRB-", "(").replace("-RRB-", ")")
            note = f"<div class='n'>artefact notes: {html.escape('; '.join(c['artefact_notes']))}</div>" if c["artefact_notes"] else ""
            parts.append(f"<div class='c'><b>{c['display_letter']}</b> — <b>{html.escape(t)}</b> "
                         f"<small>(page id: {html.escape(c['page_id'])})</small><br>{html.escape(c['display_text'])}{note}</div>")
    (OUT / f"{name}.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
