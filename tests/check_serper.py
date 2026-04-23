"""
Quick Serper API check — 3 test questions.
Run: python check_serper.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

from engine.verifier.serper_verifier import verify_with_serper

tests = [
    {
        "question": "Who is the current CEO of OpenAI?",
        "answer":   "Sam Altman",
    },
    {
        "question": "What is the capital of Australia?",
        "answer":   "Sydney",          # wrong — should trigger override
    },
    {
        "question": "Who won the 2024 US Presidential election?",
        "answer":   "Donald Trump",
    },
]

print("=" * 55)
print("  Serper API Live Check")
print("=" * 55)

for i, t in enumerate(tests, 1):
    print(f"\n[{i}] Q: {t['question']}")
    print(f"     Answer submitted: {t['answer']}")

    result = verify_with_serper(t["question"], t["answer"])

    if result.skip:
        print(f"     STATUS : SKIPPED — {result.error}")
        print("     → Serper API not configured or unreachable")
    elif result.found:
        match = "CONFIRMED" if result.matches_output else "CONTRADICTED"
        print(f"     STATUS : {match}")
        print(f"     Source : {result.source}")
        print(f"     Confidence : {result.confidence:.2f}")
        if not result.matches_output:
            print(f"     Correct answer: {result.grounded_answer[:100]}")
    else:
        print(f"     STATUS : No results found — {result.error}")

print()
print("=" * 55)
