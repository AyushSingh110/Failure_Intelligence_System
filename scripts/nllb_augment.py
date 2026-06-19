"""Augment PAIR training data with translations via local NLLB-200.

Uses Meta's NLLB-200 (No Language Left Behind) — a 200-language translation
model that runs fully locally with no API key and no ToS violations.

This script takes the PAIR training JSONL, translates each attack prompt into
N target languages, and writes a new augmented JSONL. The translated prompts
keep the original label (attack=1) and add `source_lang`/`target_lang` fields.

Usage (Windows CMD — checkpoint-safe):

    pip install transformers sentencepiece torch
    python scripts\nllb_augment.py --input data\pair_training\train.jsonl --output data\pair_training\train_augmented.jsonl
    python scripts\nllb_augment.py --input data\pair_training\train.jsonl --output data\pair_training\train_augmented.jsonl --resume

Model download: ~2.5 GB on first run (facebook/nllb-200-distilled-600M).
Use --model facebook/nllb-200-distilled-1.3B for higher quality (5 GB).

Checkpoint: data/nllb_augment_checkpoint.jsonl — one row per completed
translation. Safe to Ctrl+C and re-run with --resume.

Target languages (10, chosen for attack-vector diversity + dataset coverage):
  French, Spanish, German, Chinese, Arabic, Hindi, Japanese, Korean, Turkish, Russian
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_PATH = ROOT / "data" / "nllb_augment_checkpoint.jsonl"

# NLLB-200 language codes for the 10 target languages
TARGET_LANGS = {
    "fra_Latn": "French",
    "spa_Latn": "Spanish",
    "deu_Latn": "German",
    "zho_Hans": "Chinese (Simplified)",
    "arb_Arab": "Arabic",
    "hin_Deva": "Hindi",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "tur_Latn": "Turkish",
    "rus_Cyrl": "Russian",
}

SRC_LANG = "eng_Latn"


def _load_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("text") or ""
            label_raw = obj.get("label") if "label" in obj else obj.get("is_attack", 0)
            if prompt and int(bool(label_raw)) == 1:  # attacks only
                rows.append({"prompt": prompt, "label": 1})
    return rows


def _checkpoint_key(prompt: str, tgt_lang: str) -> str:
    return f"{tgt_lang}||{prompt[:100]}"


def _load_checkpoint() -> set[str]:
    if not CHECKPOINT_PATH.exists():
        return set()
    seen: set[str] = set()
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                key = _checkpoint_key(obj.get("original_prompt", ""), obj.get("target_lang_code", ""))
                seen.add(key)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input",  required=True, help="Input JSONL (attack prompts)")
    parser.add_argument("--output", required=True, help="Output augmented JSONL")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint — skip already-translated prompts")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M",
                        help="NLLB model ID (default: facebook/nllb-200-distilled-600M)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Translation batch size (default: 8, reduce if OOM)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Max attack prompts to process (default: all)")
    args = parser.parse_args()

    dataset = _load_dataset(args.input)
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]
    print(f"Loaded {len(dataset)} attack prompts from {args.input}")

    already_done: set[str] = _load_checkpoint() if args.resume else set()
    if already_done:
        print(f"Resuming: {len(already_done)} translations already done")

    # ── Load NLLB model ───────────────────────────────────────────────────────
    print(f"Loading NLLB model: {args.model} (may download ~2-5 GB on first run)...")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
    except ImportError:
        print("ERROR: transformers and torch required.")
        print("  pip install transformers sentencepiece torch")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=torch.float16).to(device)
    print(f"Model loaded on {device}")

    def translate_batch(texts: list[str], tgt_lang: str) -> list[str]:
        tokenizer.src_lang = SRC_LANG
        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=256,
                num_beams=2,
            )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # ── Translate ─────────────────────────────────────────────────────────────
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_file   = open(CHECKPOINT_PATH, "a", encoding="utf-8")
    output_file = open(output_path, "a", encoding="utf-8")

    total_written = 0
    prompts = [r["prompt"] for r in dataset]

    for tgt_code, tgt_name in TARGET_LANGS.items():
        print(f"\nTranslating to {tgt_name} ({tgt_code})...")

        pending_indices = [
            i for i, r in enumerate(dataset)
            if _checkpoint_key(r["prompt"], tgt_code) not in already_done
        ]

        if not pending_indices:
            print(f"  All {len(dataset)} prompts already translated — skipping")
            continue

        print(f"  {len(pending_indices)}/{len(dataset)} prompts to translate")

        for batch_start in range(0, len(pending_indices), args.batch_size):
            batch_indices = pending_indices[batch_start: batch_start + args.batch_size]
            batch_texts   = [prompts[i] for i in batch_indices]

            try:
                translated = translate_batch(batch_texts, tgt_code)
            except Exception as exc:
                print(f"  [warn] batch failed: {exc} — skipping batch")
                continue

            for idx, (orig_i, trans) in enumerate(zip(batch_indices, translated)):
                orig_row = dataset[orig_i]
                record = {
                    "prompt":            trans,
                    "label":             1,
                    "original_prompt":   orig_row["prompt"],
                    "source_lang_code":  SRC_LANG,
                    "target_lang_code":  tgt_code,
                    "target_lang_name":  tgt_name,
                    "augmented":         True,
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1
                key = _checkpoint_key(orig_row["prompt"], tgt_code)
                already_done.add(key)

            done = batch_start + len(batch_indices)
            if done % (args.batch_size * 5) == 0:
                output_file.flush()
                ckpt_file.flush()
                print(f"  [{done}/{len(pending_indices)}] flushed", end="\r", flush=True)

            # Avoid thrashing GPU — brief pause between batches
            if device == "cpu":
                time.sleep(0.1)

        output_file.flush()
        ckpt_file.flush()
        print(f"  {tgt_name} done. Total written so far: {total_written}")

    output_file.close()
    ckpt_file.close()

    print(f"\nDone. {total_written} augmented rows written to {output_path}")
    print("Next: retrain PAIR classifier on the augmented dataset.")
    print("  python scripts\\retrain_pair_v4.py --train data\\pair_training\\train_augmented.jsonl")


if __name__ == "__main__":
    main()
