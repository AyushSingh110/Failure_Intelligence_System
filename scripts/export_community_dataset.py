"""
Turn community feedback into a publishable dataset.

Community-reported false positives are the most valuable data this project can
collect: over-refusal is its largest documented weakness, and no amount of
scraping produces the label "this prompt was safe and you blocked it" — a human
has to assert it.

Publishing them is also the honest bargain. People submitted reports to improve
an open project; the resulting dataset should be open too, not a private
training asset.

Usage:
    python scripts/export_community_dataset.py                     # write JSONL
    python scripts/export_community_dataset.py --stats             # counts only
    python scripts/export_community_dataset.py --push Ayush-Singh9791/fie-community-feedback

Review before publishing. `--push` refuses to run without --reviewed, because
the raw feed is unmoderated and someone will eventually paste something they
should not have.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "community"


def build_records(raw: list[dict]) -> list[dict]:
    """
    Normalise stored reports into a flat dataset schema.

    Flattened rather than nested because most consumers (pandas, HF datasets,
    a quick grep) handle flat columns far better than nested dicts, and the
    layer scores are more useful as a single JSON column than as 12 sparse ones.
    """
    out = []
    for r in raw:
        verdict = r.get("fie_verdict") or {}
        out.append({
            "prompt":        r.get("prompt", ""),
            # The human label: what the reporter says the truth is.
            #   false_positive -> the prompt was SAFE (FIE wrongly blocked)
            #   missed_attack  -> the prompt was ADVERSARIAL (FIE wrongly allowed)
            "reported_label": "safe" if r.get("kind") == "false_positive" else "adversarial",
            "report_kind":    r.get("kind", ""),
            # What FIE said, so the disagreement is reconstructable.
            "fie_is_attack":   bool(verdict.get("is_attack", False)),
            "fie_attack_type": verdict.get("attack_type"),
            "fie_confidence":  float(verdict.get("confidence", 0.0) or 0.0),
            "fie_layers_fired": ",".join(verdict.get("layers_fired") or []),
            "fie_layer_scores": json.dumps(verdict.get("layer_scores") or {}, sort_keys=True),
            "reported_at":     r.get("reported_at", ""),
            "source":          r.get("source", "demo"),
        })
    return out


DATASET_CARD = """---
license: apache-2.0
task_categories:
- text-classification
language:
- en
tags:
- ai-safety
- guardrails
- over-refusal
- red-teaming
pretty_name: FIE Community Feedback
---

# FIE Community Feedback

Community-reported mistakes made by the [Failure Intelligence Engine]({repo}),
an open-source LLM guardrail. Every row is a case where a human said the
detector got it wrong.

## Why this exists

Over-refusal — blocking prompts that are perfectly safe — is FIE's largest
documented weakness. On standardised benchmarks it flags **53.6% of safe XSTest
prompts** and **90.4% of OR-Bench-hard prompts**, and a 20B guard model fails
the same test at 80%. The blind spot is field-wide.

Measuring it on benchmarks is one thing; collecting the cases real people hit is
another. That is what this dataset is. Nobody can scrape the label *"this prompt
was safe and you blocked it"* — a person has to assert it.

## Fields

| Field | Description |
| --- | --- |
| `prompt` | The text that was scanned |
| `reported_label` | Ground truth per the reporter: `safe` or `adversarial` |
| `report_kind` | `false_positive` (safe, blocked) or `missed_attack` (adversarial, allowed) |
| `fie_is_attack` | Whether FIE flagged it |
| `fie_attack_type` | The attack class FIE assigned, if any |
| `fie_confidence` | FIE's confidence, 0-1 |
| `fie_layers_fired` | Which detection layers fired |
| `fie_layer_scores` | Per-layer scores as a JSON object |
| `reported_at` | UTC timestamp |

## Composition

- **{n_total}** reports
- **{n_fp}** false positives (safe prompts that were blocked)
- **{n_fn}** missed attacks

## Limitations — read these

- **Labels are unverified.** They reflect one reporter's judgement, not
  adjudicated ground truth. Some "false positives" are genuinely borderline.
- **Self-selected.** People report what surprised them, so the distribution is
  skewed toward edge cases and away from ordinary traffic.
- **Small.** Treat it as an evaluation and error-analysis set, not a training
  corpus, until it is much larger.
- **Adversarial by construction.** It contains prompts designed to probe a
  safety classifier. Handle accordingly.

## Privacy

No IP addresses, cookies, fingerprints, accounts or session identifiers were
collected. Submissions are opt-in and cannot be linked to each other or to a
person. See the [privacy policy](https://failure-intelligence-system.pages.dev/privacy).

## Citation

```bibtex
@software{{fie,
  author = {{Singh, Ayush}},
  title  = {{Failure Intelligence Engine}},
  year   = {{2026}},
  url    = {{{repo}}}
}}
```

Apache-2.0. Generated {generated}.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="print counts and exit")
    ap.add_argument("--push", metavar="REPO_ID", help="upload to a HuggingFace dataset repo")
    ap.add_argument("--reviewed", action="store_true",
                    help="confirm you have read the records before publishing")
    args = ap.parse_args()

    # storage.database.get_db() is a pure accessor by design — it never
    # connects, so a request path can't accidentally trigger a blocking DNS
    # lookup. A standalone script therefore has to open the connection itself,
    # or it silently falls back to the local JSONL and reports "no feedback"
    # while the real records sit in MongoDB.
    try:
        from storage.database import initialize_vault
        initialize_vault()
    except Exception as exc:
        print(f"note: MongoDB unavailable ({type(exc).__name__}), reading local JSONL only")

    from engine.demo_feedback import export_all, stats

    if args.stats:
        print(json.dumps(stats(), indent=2))
        return 0

    raw = export_all()
    if not raw:
        print("No community feedback recorded yet.")
        return 1

    records = build_records(raw)
    kinds = Counter(r["report_kind"] for r in records)
    n_fp = kinds.get("false_positive", 0)
    n_fn = kinds.get("missed_attack", 0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = OUT_DIR / "community_feedback.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    card = DATASET_CARD.format(
        repo="https://github.com/AyushSingh110/Failure_Intelligence_System",
        n_total=len(records), n_fp=n_fp, n_fn=n_fn,
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    (OUT_DIR / "README.md").write_text(card, encoding="utf-8")

    print(f"wrote {jsonl}  ({len(records)} records)")
    print(f"  false positives : {n_fp}")
    print(f"  missed attacks  : {n_fn}")
    print(f"wrote {OUT_DIR / 'README.md'}")

    if not args.push:
        print("\nReview data/community/community_feedback.jsonl, then publish with:")
        print(f"  python {Path(__file__).name} --push <user>/<dataset> --reviewed")
        return 0

    if not args.reviewed:
        print(
            "\nREFUSING to publish without --reviewed.\n"
            "The feed is unmoderated. Read community_feedback.jsonl first and remove\n"
            "anything a person should not have submitted — once it is public, it is public."
        )
        return 1

    import os
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        try:
            from dotenv import dotenv_values
            token = (dotenv_values(ROOT / ".env").get("HUGGING_FACE_TOKEN") or "").strip()
        except Exception:
            token = ""
    if not token:
        print("ERROR: no HF token (set HF_TOKEN or HUGGING_FACE_TOKEN).")
        return 1

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(args.push, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(OUT_DIR),
        repo_id=args.push,
        repo_type="dataset",
        commit_message=f"community feedback: {len(records)} reports",
    )
    print(f"\npublished: https://huggingface.co/datasets/{args.push}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
