import requests
import json

BASE = "http://localhost:8000/api/v1"

print("=" * 65)
print("DEBUG TEST — Seeing exactly what Ollama returns")
print("=" * 65)

# Test 1: Paris
print("\nTEST 1 — Paris")
print("-" * 40)
r = requests.post(f"{BASE}/monitor", json={
    "prompt": "What is the capital of France?",
    "primary_output": "Paris",
    "primary_model_name": "gpt-4",
    "run_full_jury": False,   # skip jury — faster
}, timeout=120)
data = r.json()

print("Shadow model outputs:")
for s in data.get("shadow_model_results", []):
    print(f"  {s['model_name']}: '{s['output_text'][:80]}'")

fsv = data.get("failure_signal_vector", {})
print(f"\nAnswer counts:")
for ans, count in fsv.get("answer_counts", {}).items():
    print(f"  [{count}x] '{ans[:70]}'")

print(f"\nagreement={fsv.get('agreement_score')} entropy={fsv.get('entropy_score')}")
print(f"archetype={data.get('archetype')}")
print(f"high_risk={data.get('high_failure_risk')}")

# Test 2: Edison
print("\n\nTEST 2 — Edison (wrong answer)")
print("-" * 40)
r = requests.post(f"{BASE}/monitor", json={
    "prompt": "Who invented the telephone?",
    "primary_output": "Thomas Edison invented the telephone.",
    "primary_model_name": "gpt-4",
    "run_full_jury": False,
}, timeout=120)
data = r.json()

print("Shadow model outputs:")
for s in data.get("shadow_model_results", []):
    print(f"  {s['model_name']}: '{s['output_text'][:80]}'")

fsv = data.get("failure_signal_vector", {})
print(f"\nAnswer counts:")
for ans, count in fsv.get("answer_counts", {}).items():
    print(f"  [{count}x] '{ans[:70]}'")

print(f"\nagreement={fsv.get('agreement_score')} entropy={fsv.get('entropy_score')}")
print(f"archetype={data.get('archetype')}")
print(f"high_risk={data.get('high_failure_risk')}")