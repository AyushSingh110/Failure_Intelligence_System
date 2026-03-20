"""
verify_fixes.py — Run this on your machine to confirm which files are fixed.
Place this in your project root and run: python verify_fixes.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("FIE FIX VERIFICATION")
print("=" * 60)

errors = []

# Check 1: consistency.py threshold
try:
    from engine.detector.consistency import SEMANTIC_SIMILARITY_THRESHOLD
    if SEMANTIC_SIMILARITY_THRESHOLD == 0.72:
        print("✅ consistency.py  — threshold = 0.72")
    else:
        print(f"❌ consistency.py  — threshold = {SEMANTIC_SIMILARITY_THRESHOLD} (should be 0.72)")
        errors.append("consistency.py threshold not 0.72")
except Exception as e:
    print(f"❌ consistency.py  — import error: {e}")
    errors.append(str(e))

# Check 2: entropy.py has compute_entropy_from_counts
try:
    from engine.detector.entropy import compute_entropy_from_counts
    result = compute_entropy_from_counts({"paris": 1, "the capital": 3}, 4)
    import math
    expected = round(-(1/4*math.log2(1/4) + 3/4*math.log2(3/4)) / math.log2(4), 4)
    if abs(result - expected) < 0.001:
        print(f"✅ entropy.py      — compute_entropy_from_counts works ({result})")
    else:
        print(f"❌ entropy.py      — wrong result: {result} (expected {expected})")
        errors.append("entropy computation wrong")
except ImportError:
    print("❌ entropy.py      — compute_entropy_from_counts NOT FOUND (old file)")
    errors.append("entropy.py not updated")
except Exception as e:
    print(f"❌ entropy.py      — error: {e}")
    errors.append(str(e))

# Check 3: failure_agent.py uses compute_entropy_from_counts
try:
    with open("engine/agents/failure_agent.py", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if "compute_entropy_from_counts" in content and "ensemble_fires" in content:
        print("✅ failure_agent.py — entropy guard + compute_entropy_from_counts")
    elif "compute_entropy_from_counts" not in content:
        print("❌ failure_agent.py — missing compute_entropy_from_counts (old file)")
        errors.append("failure_agent.py not updated")
    elif "ensemble_fires" not in content:
        print("❌ failure_agent.py — missing ensemble_fires guard (old file)")
        errors.append("failure_agent.py missing ensemble_fires")
except Exception as e:
    print(f"❌ failure_agent.py — {e}")

# Check 4: routes.py uses compute_entropy_from_counts
try:
    with open("app/routes.py", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if "compute_entropy_from_counts" in content and "ensemble_fires" in content:
        print("✅ routes.py        — entropy guard + compute_entropy_from_counts")
    elif "compute_entropy_from_counts" not in content:
        print("❌ routes.py        — missing compute_entropy_from_counts (old file)")
        errors.append("routes.py not updated")
    elif "ensemble_fires" not in content:
        print("❌ routes.py        — missing ensemble_fires guard")
        errors.append("routes.py missing ensemble_fires")
except Exception as e:
    print(f"❌ routes.py        — {e}")

# Check 5: labeling.py Rule 3 guard
try:
    with open("engine/archetypes/labeling.py", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if "entropy > 0.0 or agreement" in content:
        print("✅ labeling.py      — Rule 3 entropy guard present")
    else:
        print("❌ labeling.py      — Rule 3 entropy guard MISSING (old file)")
        errors.append("labeling.py not updated")
except Exception as e:
    print(f"❌ labeling.py      — {e}")

# Check 6: ensemble.py has SEMANTIC_SIMILARITY_THRESHOLD
try:
    with open("engine/detector/ensemble.py", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if "SEMANTIC_SIMILARITY_THRESHOLD" in content and "return 1.0" in content:
        print("✅ ensemble.py      — SEMANTIC_SIMILARITY_THRESHOLD + return 1.0 fix")
    else:
        print("❌ ensemble.py      — fix missing (old file)")
        errors.append("ensemble.py not updated")
except Exception as e:
    print(f"❌ ensemble.py      — {e}")

print()
print("=" * 60)
if not errors:
    print("ALL FIXES VERIFIED ✅ — restart server and test")
else:
    print(f"FIXES MISSING ❌ — {len(errors)} file(s) need updating:")
    for e in errors:
        print(f"  • {e}")
print("=" * 60)