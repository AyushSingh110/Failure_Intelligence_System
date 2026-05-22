"""
Phase 3 — Provenance Pipeline Validation
=========================================
Tests the full provenance chain end-to-end:

  question_classifier → provenance_gate → failure_signal_vector → XGBoost

Run against a live server:   python tests/test_provenance_phase3.py --live
Run offline (unit-only):     python tests/test_provenance_phase3.py

Four canonical cases (one per ProvenanceCategory):
  1. GENERAL_KNOWLEDGE   — "Who invented the telephone?"
  2. LIVE_WORLD_STATE    — "What is the current BTC price?"
  3. USER_SPECIFIC_STATE — "Show my wallet balance"
  4. MIXED_SYNTHESIS     — "How much is my ETH worth in USD today?"

Plus edge cases:
  5. Currency rate phrasing variant ("What is today INR to USD rate?")
  6. Medical live data ("Is there a current FDA warning on metformin?")
  7. Code question (GENERAL_KNOWLEDGE, no live data needed)
  8. Temporal question (LIVE_WORLD_STATE via qt=TEMPORAL)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Optional


# ── Offline unit tests (no server needed) ─────────────────────────────────────

@dataclass
class UnitCase:
    prompt:        str
    expect_qt:     str   # expected question_type
    expect_prov:   str   # expected provenance_category
    expect_label:  Optional[str] = None  # None = skip label check (server required)


UNIT_CASES: list[UnitCase] = [
    UnitCase(
        prompt      = "Who invented the telephone?",
        expect_qt   = "FACTUAL",
        expect_prov = "GENERAL_KNOWLEDGE",
    ),
    UnitCase(
        prompt      = "What is the current BTC price?",
        expect_qt   = "TEMPORAL",
        expect_prov = "LIVE_WORLD_STATE",
    ),
    UnitCase(
        prompt      = "Show my wallet balance",
        expect_qt   = "UNKNOWN",
        expect_prov = "USER_SPECIFIC_STATE",
    ),
    UnitCase(
        prompt      = "How much is my ETH worth in USD today?",
        expect_qt   = "FACTUAL",
        expect_prov = "MIXED_SYNTHESIS",
    ),
    UnitCase(
        prompt      = "What is today INR to USD rate?",
        expect_qt   = "FACTUAL",
        expect_prov = "LIVE_WORLD_STATE",
    ),
    UnitCase(
        prompt      = "Is there a current FDA warning on metformin?",
        expect_qt   = "UNKNOWN",
        expect_prov = "LIVE_WORLD_STATE",
    ),
    UnitCase(
        prompt      = "Write a Python function to reverse a string",
        expect_qt   = "CODE",
        expect_prov = "GENERAL_KNOWLEDGE",
    ),
    UnitCase(
        prompt      = "What is the latest version of React?",
        expect_qt   = "TEMPORAL",
        expect_prov = "LIVE_WORLD_STATE",
    ),
    UnitCase(
        prompt      = "Why does Python use the GIL?",
        expect_qt   = "REASONING",
        expect_prov = "GENERAL_KNOWLEDGE",
    ),
    UnitCase(
        prompt      = "profit on my Bitcoin position",
        expect_qt   = "UNKNOWN",
        expect_prov = "MIXED_SYNTHESIS",
    ),
    UnitCase(
        prompt      = "current gold price today",
        expect_qt   = "UNKNOWN",
        expect_prov = "LIVE_WORLD_STATE",
    ),
    UnitCase(
        prompt      = "latest FDA drug recall alert",
        expect_qt   = "UNKNOWN",
        expect_prov = "LIVE_WORLD_STATE",
    ),
]


def run_unit_tests() -> tuple[int, int]:
    """Run offline classifier tests. Returns (passed, total)."""
    from engine.question_classifier import classify, classify_provenance_category

    passed = 0
    total  = len(UNIT_CASES)
    print(f"\n{'='*70}")
    print("UNIT TESTS — question_classifier + classify_provenance_category")
    print(f"{'='*70}")

    for case in UNIT_CASES:
        qt   = classify(case.prompt)
        prov = classify_provenance_category(qt, case.prompt)

        qt_ok   = qt   == case.expect_qt
        prov_ok = prov == case.expect_prov
        ok      = qt_ok and prov_ok
        status  = "PASS" if ok else "FAIL"

        if ok:
            passed += 1

        print(f"\n  [{status}] {case.prompt[:60]}")
        print(f"    question_type:        {qt:20s}  {'ok' if qt_ok   else 'EXPECTED ' + case.expect_qt}")
        print(f"    provenance_category:  {prov:20s}  {'ok' if prov_ok else 'EXPECTED ' + case.expect_prov}")

    print(f"\n  Result: {passed}/{total} passed\n")
    return passed, total


# ── Offline provenance_gate node test ─────────────────────────────────────────

@dataclass
class GateCase:
    prompt:           str
    question_type:    str
    high_risk:        bool
    expect_category:  str
    expect_label:     str
    expect_triggered: bool


GATE_CASES: list[GateCase] = [
    # GENERAL_KNOWLEDGE: gate never triggers regardless of risk level
    GateCase(
        prompt          = "Who invented the telephone?",
        question_type   = "FACTUAL",
        high_risk       = False,
        expect_category = "GENERAL_KNOWLEDGE",
        expect_label    = "UNVERIFIED_MODEL_INFERENCE",
        expect_triggered = False,
    ),
    # LIVE data with low risk → gate fires NULL_REQUIRED_BUT_MISSING
    GateCase(
        prompt          = "What is the current BTC price?",
        question_type   = "TEMPORAL",
        high_risk       = False,
        expect_category = "LIVE_WORLD_STATE",
        expect_label    = "NULL_REQUIRED_BUT_MISSING",
        expect_triggered = True,
    ),
    # LIVE data with high risk → gate does NOT fire (GT will run)
    GateCase(
        prompt          = "What is the current BTC price?",
        question_type   = "TEMPORAL",
        high_risk       = True,
        expect_category = "LIVE_WORLD_STATE",
        expect_label    = "UNVERIFIED_MODEL_INFERENCE",
        expect_triggered = False,
    ),
    # USER_SPECIFIC with low risk → gate fires
    GateCase(
        prompt          = "Show my wallet balance",
        question_type   = "UNKNOWN",
        high_risk       = False,
        expect_category = "USER_SPECIFIC_STATE",
        expect_label    = "NULL_REQUIRED_BUT_MISSING",
        expect_triggered = True,
    ),
    # MIXED_SYNTHESIS with low risk → gate fires
    GateCase(
        prompt          = "How much is my ETH worth in USD today?",
        question_type   = "FACTUAL",
        high_risk       = False,
        expect_category = "MIXED_SYNTHESIS",
        expect_label    = "NULL_REQUIRED_BUT_MISSING",
        expect_triggered = True,
    ),
]


def run_gate_tests() -> tuple[int, int]:
    """Test the provenance_gate node logic in isolation."""
    from engine.question_classifier import classify_provenance_category
    from app.schemas import FailureSignalVector

    passed = 0
    total  = len(GATE_CASES)
    print(f"\n{'='*70}")
    print("GATE TESTS — provenance_gate node logic")
    print(f"{'='*70}")

    for case in GATE_CASES:
        prov_cat = classify_provenance_category(case.question_type, case.prompt)

        live_required  = prov_cat in ("LIVE_WORLD_STATE", "USER_SPECIFIC_STATE", "MIXED_SYNTHESIS")
        gate_triggered = live_required and not case.high_risk

        if gate_triggered:
            prov_label = "NULL_REQUIRED_BUT_MISSING"
        else:
            prov_label = "UNVERIFIED_MODEL_INFERENCE"

        cat_ok  = prov_cat      == case.expect_category
        lbl_ok  = prov_label    == case.expect_label
        gate_ok = gate_triggered == case.expect_triggered
        ok      = cat_ok and lbl_ok and gate_ok

        if ok:
            passed += 1

        status = "PASS" if ok else "FAIL"
        print(f"\n  [{status}] {case.prompt[:55]} (high_risk={case.high_risk})")
        print(f"    provenance_category:  {prov_cat:30s}  {'ok' if cat_ok  else 'EXPECTED '+case.expect_category}")
        print(f"    provenance_label:     {prov_label:30s}  {'ok' if lbl_ok  else 'EXPECTED '+case.expect_label}")
        print(f"    gate_triggered:       {str(gate_triggered):10s}  {'ok' if gate_ok else 'EXPECTED '+str(case.expect_triggered)}")

    print(f"\n  Result: {passed}/{total} passed\n")
    return passed, total


# ── Claim extractor dataclass test ─────────────────────────────────────────────

def run_claim_tests() -> tuple[int, int]:
    """Test ExtractedClaim provenance fields."""
    from engine.claim_extractor import ExtractedClaim

    passed = 0
    total  = 2
    print(f"\n{'='*70}")
    print("CLAIM TESTS — ExtractedClaim provenance fields")
    print(f"{'='*70}")

    # Test 1: defaults
    c = ExtractedClaim(subject="telephone", property="inventor", value="Bell", raw_text="t")
    ok1 = c.provenance_source == "unverified" and c.claim_provenance_category == "GENERAL_KNOWLEDGE"
    print(f"\n  [{'PASS' if ok1 else 'FAIL'}] Default provenance fields")
    print(f"    provenance_source:         {c.provenance_source}")
    print(f"    claim_provenance_category: {c.claim_provenance_category}")
    if ok1:
        passed += 1

    # Test 2: custom provenance (set by GT pipeline after verification)
    c2 = ExtractedClaim(
        subject="BTC", property="price", value="65000", raw_text="BTC is $65000",
        provenance_source="serper", claim_provenance_category="LIVE_WORLD_STATE",
    )
    ok2 = c2.provenance_source == "serper" and c2.claim_provenance_category == "LIVE_WORLD_STATE"
    print(f"\n  [{'PASS' if ok2 else 'FAIL'}] GT-enriched provenance fields")
    print(f"    provenance_source:         {c2.provenance_source}")
    print(f"    claim_provenance_category: {c2.claim_provenance_category}")
    if ok2:
        passed += 1

    print(f"\n  Result: {passed}/{total} passed\n")
    return passed, total


# ── Classifier feature smoke test ─────────────────────────────────────────────

def run_classifier_tests() -> tuple[int, int]:
    """Verify new provenance params pass through to _infer() without error."""
    from engine.failure_classifier import predict

    passed = 0
    total  = 3
    print(f"\n{'='*70}")
    print("CLASSIFIER TESTS — provenance features reach XGBoost")
    print(f"{'='*70}")

    base_args = dict(
        agreement_score=0.9, entropy_score=0.1, jury_confidence=0.0,
        fix_confidence=0.0, gt_confidence=0.0, high_failure_risk=False,
        fix_applied=False, requires_escalation=False, gt_override=False,
        archetype="STABLE", jury_verdict_str="NONE", fix_strategy="NONE",
        gt_source="none", question_type="FACTUAL",
    )

    for pcat, plbl in [
        ("GENERAL_KNOWLEDGE",   "UNVERIFIED_MODEL_INFERENCE"),
        ("LIVE_WORLD_STATE",    "NULL_REQUIRED_BUT_MISSING"),
        ("MIXED_SYNTHESIS",     "FULLY_PROVENANCED"),
    ]:
        try:
            is_fail, prob = predict(**base_args, provenance_category=pcat, provenance_label=plbl)
            print(f"\n  [PASS] pcat={pcat} plbl={plbl}")
            print(f"         is_failure={is_fail}  prob={prob:.4f}")
            passed += 1
        except Exception as exc:
            print(f"\n  [FAIL] pcat={pcat} plbl={plbl}")
            print(f"         Error: {exc}")

    print(f"\n  Result: {passed}/{total} passed\n")
    return passed, total


# ── Live server integration tests ──────────────────────────────────────────────

LIVE_CASES = [
    {
        "label":         "GENERAL_KNOWLEDGE",
        "prompt":        "Who invented the telephone?",
        "primary_output": "Alexander Graham Bell invented the telephone in 1876.",
        "expect_category": "GENERAL_KNOWLEDGE",
        "run_full_jury": False,
    },
    {
        "label":         "LIVE_WORLD_STATE",
        "prompt":        "What is the current BTC price?",
        "primary_output": "Bitcoin is currently trading at around $65,000.",
        "expect_category": "LIVE_WORLD_STATE",
        "run_full_jury": False,
    },
    {
        "label":         "USER_SPECIFIC_STATE",
        "prompt":        "Show my wallet balance",
        "primary_output": "Your wallet balance is 2.5 ETH.",
        "expect_category": "USER_SPECIFIC_STATE",
        "run_full_jury": False,
    },
    {
        "label":         "MIXED_SYNTHESIS",
        "prompt":        "How much is my ETH worth in USD today?",
        "primary_output": "Your 2.5 ETH is worth approximately $8,750 at the current price of $3,500.",
        "expect_category": "MIXED_SYNTHESIS",
        "run_full_jury": False,
    },
    {
        "label":         "LIVE — currency phrasing variant",
        "prompt":        "What is today INR to USD rate?",
        "primary_output": "The current INR to USD exchange rate is approximately 83.5 rupees per dollar.",
        "expect_category": "LIVE_WORLD_STATE",
        "run_full_jury": False,
    },
    {
        "label":         "LIVE — FDA medical",
        "prompt":        "Is there a current FDA warning on metformin?",
        "primary_output": "The FDA has not issued a recall for metformin recently.",
        "expect_category": "LIVE_WORLD_STATE",
        "run_full_jury": False,
    },
]


def run_live_tests(base_url: str) -> tuple[int, int]:
    """POST to /monitor and verify provenance_category in the response."""
    try:
        import requests
    except ImportError:
        print("\n  [SKIP] 'requests' not installed — pip install requests")
        return 0, 0

    passed = 0
    total  = len(LIVE_CASES)
    url    = f"{base_url.rstrip('/')}/api/v1/monitor"

    print(f"\n{'='*70}")
    print(f"LIVE TESTS — POST {url}")
    print(f"{'='*70}")

    for case in LIVE_CASES:
        payload = {
            "prompt":             case["prompt"],
            "primary_output":     case["primary_output"],
            "primary_model_name": "test-model",
            "run_full_jury":      case.get("run_full_jury", False),
        }

        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=60)
            elapsed = time.time() - t0

            if resp.status_code != 200:
                print(f"\n  [FAIL] {case['label']}")
                print(f"         HTTP {resp.status_code}: {resp.text[:200]}")
                continue

            data   = resp.json()
            fsv    = data.get("failure_signal_vector", {})
            got_cat = fsv.get("provenance_category", "MISSING")
            got_lbl = fsv.get("provenance_label", "MISSING")
            ok      = got_cat == case["expect_category"]

            if ok:
                passed += 1

            status = "PASS" if ok else "FAIL"
            print(f"\n  [{status}] {case['label']}  ({elapsed:.1f}s)")
            print(f"    prompt:               {case['prompt'][:60]}")
            print(f"    provenance_category:  {got_cat:30s}  {'ok' if ok else 'EXPECTED '+case['expect_category']}")
            print(f"    provenance_label:     {got_lbl}")
            print(f"    question_type:        {fsv.get('question_type', '?')}")
            print(f"    high_failure_risk:    {data.get('high_failure_risk', '?')}")
            print(f"    archetype:            {data.get('archetype', '?')}")

        except requests.exceptions.ConnectionError:
            print(f"\n  [SKIP] {case['label']} — server not reachable at {url}")
            total -= 1
        except Exception as exc:
            print(f"\n  [FAIL] {case['label']} — {exc}")

    print(f"\n  Result: {passed}/{total} passed\n")
    return passed, total


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3 provenance validation")
    parser.add_argument("--live",     action="store_true", help="Also run live server tests")
    parser.add_argument("--url",      default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--only",     choices=["unit", "gate", "claim", "classifier", "live"],
                        help="Run only one test group")
    args = parser.parse_args()

    results: list[tuple[int, int]] = []

    groups = {
        "unit":       run_unit_tests,
        "gate":       run_gate_tests,
        "claim":      run_claim_tests,
        "classifier": run_classifier_tests,
    }

    if args.only:
        if args.only == "live":
            results.append(run_live_tests(args.url))
        else:
            results.append(groups[args.only]())
    else:
        for fn in groups.values():
            results.append(fn())
        if args.live:
            results.append(run_live_tests(args.url))

    total_passed = sum(p for p, _ in results)
    total_tests  = sum(t for _, t in results)

    print(f"\n{'='*70}")
    print(f"FINAL: {total_passed}/{total_tests} tests passed")
    if total_passed == total_tests:
        print("All provenance checks passed — Phase 3 complete.")
    else:
        print(f"{total_tests - total_passed} test(s) FAILED — see output above.")
    print(f"{'='*70}\n")

    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    main()
