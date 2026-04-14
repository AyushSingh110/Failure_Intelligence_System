"""
test_ground_truth.py — Tests the GT pipeline in isolation.

Calls run_ground_truth_pipeline() directly — no server, no shadow models, no FIE API.
Just the GT logic itself.

Run:
    python test_ground_truth.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.verifier.ground_truth_pipeline import run_ground_truth_pipeline

SEP = "─" * 65


def run(label, prompt, primary_output, root_cause, jury_confidence=0.80, shadow_outputs=None):
    print()
    print(SEP)
    print(f"  TEST: {label}")
    print(f"  Prompt        : {prompt}")
    print(f"  Model answer  : {primary_output}")
    print(f"  Root cause    : {root_cause}  (jury confidence: {jury_confidence:.0%})")
    print(SEP)

    gt = run_ground_truth_pipeline(
        prompt          = prompt,
        primary_output  = primary_output,
        root_cause      = root_cause,
        jury_confidence = jury_confidence,
        shadow_outputs  = shadow_outputs or [],
        shadow_weights  = None,
    )

    print(f"  Source        : {gt.source or 'none'}")
    print(f"  Confidence    : {gt.confidence:.0%}")
    print(f"  From cache    : {gt.from_cache}")
    print(f"  Verified ans  : {gt.verified_answer or '(none — use original or shadow)'}")
    print(f"  Escalate?     : {gt.requires_escalation}")
    if gt.escalation_reason:
        print(f"  Escalation    : {gt.escalation_reason[:100]}")

    print()
    print("  Pipeline trace:")
    for step in gt.pipeline_trace:
        print(f"    → {step}")

    # Verdict
    print()
    if gt.verified_answer and not gt.requires_escalation:
        if gt.verified_answer.strip().lower() != primary_output.strip().lower():
            print(f"  VERDICT: CORRECTED  '{primary_output}' → '{gt.verified_answer}'")
        else:
            print(f"  VERDICT: CONFIRMED  Model answer matches GT")
    elif gt.requires_escalation:
        print(f"  VERDICT: ESCALATED  Human review needed")
    else:
        print(f"  VERDICT: FALLBACK   Using shadow consensus")

    return gt


# ── Test cases ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 65)
    print("  GROUND TRUTH PIPELINE — ISOLATION TEST")
    print("=" * 65)

    # 1. Classic factual wrong answer — should Wikidata override
    run(
        label          = "Telephone inventor — wrong (Edison)",
        prompt         = "Who invented the telephone?",
        primary_output = "Thomas Edison",
        root_cause     = "FACTUAL_HALLUCINATION",
        jury_confidence= 0.80,
        shadow_outputs = ["Alexander Graham Bell", "Alexander Graham Bell", "Bell"],
    )

    # 2. Capital of Australia — wrong (Sydney) — should Wikidata override
    run(
        label          = "Capital of Australia — wrong (Sydney)",
        prompt         = "What is the capital of Australia?",
        primary_output = "Sydney",
        root_cause     = "FACTUAL_HALLUCINATION",
        jury_confidence= 0.75,
        shadow_outputs = ["Canberra", "Canberra", "Canberra"],
    )

    # 3. Permanent fact — H2O — should confirm, NOT route to Serper
    run(
        label          = "Chemical formula for water — correct",
        prompt         = "What is the chemical formula for water?",
        primary_output = "H2O",
        root_cause     = "FACTUAL_HALLUCINATION",
        jury_confidence= 0.60,
        shadow_outputs = ["H2O", "H2O", "H2O"],
    )

    # 4. Permanent fact — wrong H2O2 — should catch even though formula
    run(
        label          = "Chemical formula for water — wrong (H2O2)",
        prompt         = "What is the chemical formula for water?",
        primary_output = "H2O2",
        root_cause     = "FACTUAL_HALLUCINATION",
        jury_confidence= 0.80,
        shadow_outputs = ["H2O", "H2O", "H2O"],
    )

    # 5. Temporal question — should route to Serper
    run(
        label          = "Temporal — current CEO of OpenAI",
        prompt         = "Who is the current CEO of OpenAI?",
        primary_output = "Sam Altman",
        root_cause     = "TEMPORAL_KNOWLEDGE_CUTOFF",
        jury_confidence= 0.70,
        shadow_outputs = ["Sam Altman", "Sam Altman", "Elon Musk"],
    )

    # 6. Moon landing — correct answer — should confirm
    run(
        label          = "Moon landing — correct (Neil Armstrong)",
        prompt         = "Who was the first person to walk on the moon?",
        primary_output = "Neil Armstrong",
        root_cause     = "FACTUAL_HALLUCINATION",
        jury_confidence= 0.65,
        shadow_outputs = ["Neil Armstrong", "Neil Armstrong", "Neil Armstrong"],
    )

    # 7. Moon landing — wrong (Buzz Aldrin) — should override
    run(
        label          = "Moon landing — wrong (Buzz Aldrin)",
        prompt         = "Who was the first person to walk on the moon?",
        primary_output = "Buzz Aldrin",
        root_cause     = "FACTUAL_HALLUCINATION",
        jury_confidence= 0.80,
        shadow_outputs = ["Neil Armstrong", "Neil Armstrong", "Neil Armstrong"],
    )

    # 8. Cache test — run same question twice, second should be cache hit
    print()
    print("=" * 65)
    print("  CACHE TEST — run same question twice")
    print("=" * 65)
    run(
        label          = "Cache test — first run (cache miss expected)",
        prompt         = "What is the capital of France?",
        primary_output = "Paris",
        root_cause     = "FACTUAL_HALLUCINATION",
        jury_confidence= 0.80,
        shadow_outputs = ["Paris", "Paris", "Paris"],
    )
    run(
        label          = "Cache test — second run (cache HIT expected if confidence>=0.90)",
        prompt         = "What is the capital of France?",
        primary_output = "Berlin",
        root_cause     = "FACTUAL_HALLUCINATION",
        jury_confidence= 0.80,
        shadow_outputs = ["Paris", "Paris", "Paris"],
    )

    print()
    print("=" * 65)
    print("  Done. Check trace above for each pipeline decision.")
    print("=" * 65)
    print()
