import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fie.monitor import monitor   # ← from local fie/monitor.py
from fie.client import FIEClient
from fie.config import FIEConfig

#Configuration
GROQ_API_KEY = "Key"
FIE_API_KEY  = "FIE_API_KEY"
FIE_URL      = "http://localhost:8000"

# Logging setup 
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("test_local")

# ── Check keys ────────────────────────────────────────────────────────────────
if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
    print("\nERROR: GROQ_API_KEY looks invalid. It should start with 'gsk_'")
    sys.exit(1)

if not FIE_API_KEY or not FIE_API_KEY.startswith("fie-"):
    print("\nERROR: FIE_API_KEY looks invalid. It should start with 'fie-'")
    sys.exit(1)

# ── Groq primary LLM helper ───────────────────────────────────────────────────

def call_groq(prompt: str, force_answer: str = "") -> str:
    """
    Calls Groq llama-3.1-8b-instant with the given prompt.
    If force_answer is provided, returns it directly without calling Groq.
    (Used for deliberate wrong-answer tests.)
    """
    if force_answer:
        return force_answer

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model    = "llama-3.1-8b-instant",
            messages = [
                {"role": "system",  "content": "Answer the question directly and concisely. One sentence maximum."},
                {"role": "user",    "content": prompt},
            ],
            max_tokens  = 80,
            temperature = 0.0,
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        print("ERROR: groq package not installed. Run: pip install groq")
        sys.exit(1)
    except Exception as exc:
        print(f"Groq call failed: {exc}")
        return "ERROR"


#FIE monitor call 

def call_fie(prompt: str, primary_output: str, label: str) -> dict:
    """Sends (prompt, primary_output) to FIE and returns the result."""
    config = FIEConfig(fie_url=FIE_URL, api_key=FIE_API_KEY)
    client = FIEClient(config)
    result = client.monitor(
        prompt             = prompt,
        primary_output     = primary_output,
        primary_model_name = "test-groq-llama-3.1-8b",
        latency_ms         = 500,
        run_full_jury      = True,
    )
    return result or {}


# Result parser 

def parse_result(r: dict) -> dict:
    fsv   = r.get("failure_signal_vector") or {}
    fix   = r.get("fix_result")            or {}
    gt    = r.get("ground_truth")          or {}
    jury  = r.get("jury")                  or {}
    pv    = jury.get("primary_verdict")    or {} if jury else {}

    return {
        "high_risk":     r.get("high_failure_risk", False),
        "archetype":     r.get("archetype", "UNKNOWN"),
        "agreement":     fsv.get("agreement_score", 0.0),
        "entropy":       fsv.get("entropy_score", 0.0),
        "verdict":       pv.get("root_cause", "—"),
        "confidence":    pv.get("confidence_score", 0.0),
        "fix_applied":   fix.get("fix_applied", False),
        "fix_strategy":  fix.get("fix_strategy", "—"),
        "fix_output":    fix.get("fixed_output", ""),
        "gt_source":     gt.get("source", "none"),
        "escalated":     r.get("requires_human_review", False),
        "summary":       r.get("failure_summary", ""),
    }


#Test cases 
FAILURE_TESTS = [
    {
        "name":          "Telephone inventor — wrong (Edison)",
        "prompt":        "Who invented the telephone?",
        "wrong_answer":  "Thomas Edison",
        "correct":       "Alexander Graham Bell",
    },
    {
        "name":          "Capital of Australia — wrong (Sydney)",
        "prompt":        "What is the capital of Australia?",
        "wrong_answer":  "Sydney",
        "correct":       "Canberra",
    },
    {
        "name":          "First moon landing — wrong (Buzz Aldrin)",
        "prompt":        "Who was the first person to walk on the moon?",
        "wrong_answer":  "Buzz Aldrin",
        "correct":       "Neil Armstrong",
    },
    {
        "name":          "Chemical formula for water — wrong (H2O2)",
        "prompt":        "What is the chemical formula for water?",
        "wrong_answer":  "H2O2",
        "correct":       "H2O",
    },
    {
        "name":          "15 x 15 — wrong (215)",
        "prompt":        "What is the result of 15 multiplied by 15?",
        "wrong_answer":  "215",
        "correct":       "225",
    },
]

#Correct Tests
CORRECT_TESTS = [
    {
        "name":          "Capital of France — correct",
        "prompt":        "What is the capital of France?",
        "use_groq":      True,
    },
    {
        "name":          "Square root of 144 — correct",
        "prompt":        "What is the square root of 144?",
        "use_groq":      True,
    },
    {
        "name":          "H2O — correct",
        "prompt":        "What is the chemical formula for water?",
        "correct_answer": "H2O",
        "use_groq":      False,
    },
    {
        "name":          "Neil Armstrong — correct",
        "prompt":        "Who was the first person to walk on the moon?",
        "correct_answer": "Neil Armstrong",
        "use_groq":      False,
    },
    {
        "name":          "15 x 15 = 225 — correct",
        "prompt":        "What is the result of 15 multiplied by 15?",
        "correct_answer": "225",
        "use_groq":      False,
    },
]


#Decorator test 

@monitor(
    fie_url       = FIE_URL,
    api_key       = FIE_API_KEY,
    model_name    = "groq-llama-decorator-test",
    run_full_jury = True,
    mode          = "monitor",   
    log_results   = True,
)
def ask_groq_with_monitor(prompt: str) -> str:
    """A simple Groq function wrapped with the local @monitor decorator."""
    return call_groq(prompt)


#Main runner 

def run_tests():
    results_a = []   # failure test results
    results_b = []   # correct test results

    sep = "─" * 70

    print()
    print("  FIE LOCAL DECORATOR TEST")
    print("  Testing: primary outlier ")
    print("  Testing: GT pipeline gate ")
  

    #decorator checks
    print()
    print("@monitor decorator smoke test")
    answer = ask_groq_with_monitor(prompt="What is 2 + 2?")
    print(f"  Groq answered: '{answer}'")
    time.sleep(5)
    print()

    #Failure tests
    print(sep)
    print(" Wrong answers (FIE SHOULD detect these as failures)")
    print(sep)

    for i, tc in enumerate(FAILURE_TESTS, 1):
        print(f"\n[A{i}] {tc['name']}")
        print(f"     Prompt:    {tc['prompt']}")
        print(f"     Submitted: {tc['wrong_answer']}  ← deliberately wrong")
        print(f"     Correct:   {tc['correct']}")

        r   = call_fie(tc["prompt"], tc["wrong_answer"], tc["name"])
        p   = parse_result(r)
        detected = p["high_risk"] or p["fix_applied"]

        status = "TRUE POSITIVE  ✓" if detected else "FALSE NEGATIVE ✗ (FIE missed this!)"
        print(f"     Result:    {status}")
        print(f"     FIE:       risk={p['high_risk']} | archetype={p['archetype']} | "
              f"agreement={p['agreement']:.2f} | verdict={p['verdict']} ({p['confidence']:.0%})")
        if p["fix_applied"]:
            print(f"     Fix:       [{p['fix_strategy']}] → '{p['fix_output'][:80]}'")
        if p["escalated"]:
            print(f"     Escalated: Yes (human review queued)")

        results_a.append({
            "name":     tc["name"],
            "detected": detected,
            "parsed":   p,
        })
        time.sleep(2)

    # Correct answer tests 
    print()
    print(sep)
    print(" Correct answers (FIE should NOT flag these)")
    print(sep)

    for i, tc in enumerate(CORRECT_TESTS, 1):
        if tc.get("use_groq"):
            answer = call_groq(tc["prompt"])
        else:
            answer = tc["correct_answer"]

        print(f"\n[B{i}] {tc['name']}")
        print(f"     Prompt:    {tc['prompt']}")
        print(f"     Submitted: {answer}")

        r   = call_fie(tc["prompt"], answer, tc["name"])
        p   = parse_result(r)
        flagged = p["high_risk"] or p["fix_applied"]

        status = "TRUE NEGATIVE  " if not flagged else "FALSE POSITIVE (FIE wrongly flagged!)"
        print(f"Result:{status}")
        print(f"FIE: risk={p['high_risk']} | archetype={p['archetype']} | "
              f"agreement={p['agreement']:.2f} | verdict={p['verdict']} ({p['confidence']:.0%})")
        if p["fix_applied"]:
            print(f"  Fix wrongly applied: [{p['fix_strategy']}] → '{p['fix_output'][:80]}'")

        results_b.append({
            "name":    tc["name"],
            "flagged": flagged,
            "answer":  answer,
            "parsed":  p,
        })
        time.sleep(2)

    #Summary 
    tp = sum(1 for r in results_a if r["detected"])
    fn = len(results_a) - tp
    tn = sum(1 for r in results_b if not r["flagged"])
    fp = len(results_b) - tn

    recall      = tp / len(results_a) if results_a else 0
    specificity = tn / len(results_b) if results_b else 0
    fpr         = fp / len(results_b) if results_b else 0

    print()
    print("  SUMMARY")
    print()
    print(" Failure detection (recall)")
    print(f" True positives:  {tp}/{len(results_a)}  (FIE correctly flagged wrong answers)")
    print(f" False negatives: {fn}/{len(results_a)}  (FIE missed real failures)")
    print(f" Recall:          {recall:.0%}   target: > 80%")
    print()
    print(" Correct answer stability (specificity)")
    print(f" True negatives:  {tn}/{len(results_b)}  (FIE correctly left correct answers alone)")
    print(f" False positives: {fp}/{len(results_b)}  (FIE wrongly flagged correct answers)")
    print(f" Specificity:     {specificity:.0%}   target: > 70%")
    print(f" False pos rate:  {fpr:.0%}   target: < 30%")
    print()

    if recall >= 0.80 and specificity >= 0.70:
        print("  OVERALL: PASS Both targets met.")
    elif recall >= 0.80:
        print("  OVERALL: PARTIAL  Recall OK but too many false positives.")
        print("  The primary-outlier fix may need threshold adjustment.")
    elif specificity >= 0.70:
        print("  OVERALL: PARTIAL Low false positives but FIE is missing failures.")
    else:
        print("  OVERALL: FAIL  Neither target met. Check server logs for errors.")

    print()
    print("  Previous run FPR (before fix): 80%")
    print(f"  This run FPR (after fix):  {fpr:.0%}")
    improvement = 0.80 - fpr
    if improvement > 0:
        print(f"  Improvement: {improvement:.0%} reduction in false positives ✓")
    else:
        print("  Improvement: No improvement check fix applied correctly.")
    print()

    # Per-case detail table 
    print("  DETAILED RESULTS")
    print(f"  {'Test':<42} {'Expected':<10} {'Got':<10} {'OK?'}")
    print(f"  {'─'*42} {'─'*10} {'─'*10} {'─'*5}")
    for r in results_a:
        got = "FLAGGED" if r["detected"] else "STABLE"
        ok  = "✓" if r["detected"] else "✗"
        print(f"  {r['name'][:42]:<42} {'FLAGGED':<10} {got:<10} {ok}")
    for r in results_b:
        got = "FLAGGED" if r["flagged"] else "STABLE"
        ok  = "✓" if not r["flagged"] else "✗"
        print(f"  {r['name'][:42]:<42} {'STABLE':<10} {got:<10} {ok}")
    print()


if __name__ == "__main__":
    run_tests()
