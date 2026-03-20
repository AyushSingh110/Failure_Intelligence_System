"""
test_sdk.py — Tests the FIE SDK end to end.

Run with:
    python test_sdk.py

Make sure before running:
    Terminal 1: uvicorn app.main:app --port 8000
    Terminal 2: ollama serve
"""
import logging
import time

# Show all FIE logs in terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s"
)

from fie import monitor

print("=" * 60)
print("FIE SDK TEST")
print("=" * 60)
print()

# ── Test 1: Stable answer (all models should agree) ─────────
print("TEST 1 — Stable answer (should return STABLE)")
print("-" * 40)

@monitor(
    fie_url="http://localhost:8000",
    model_name="gpt-4-test",
    async_mode=False,      # sync so we see result before moving on
    log_results=True,
)
def stable_llm(prompt: str) -> str:
    """Simulates a correct LLM answer."""
    return "Paris"

start = time.time()
answer = stable_llm("What is the capital of France?")
elapsed = round((time.time() - start) * 1000)
print(f"Answer to user: '{answer}'")
print(f"Total time including FIE: {elapsed}ms")
print()

# ── Test 2: Wrong answer (should trigger MODEL_BLIND_SPOT) ───
print("TEST 2 — Wrong answer (should return MODEL_BLIND_SPOT)")
print("-" * 40)

@monitor(
    fie_url="http://localhost:8000",
    model_name="gpt-4-test",
    async_mode=False,
    log_results=True,
)
def wrong_llm(prompt: str) -> str:
    """Simulates a wrong LLM answer."""
    return "Thomas Edison invented the telephone."

start = time.time()
answer = wrong_llm("Who invented the telephone?")
elapsed = round((time.time() - start) * 1000)
print(f"Answer to user: '{answer}'")
print(f"Total time including FIE: {elapsed}ms")
print()

# ── Test 3: Async mode (user never waits) ───────────────────
print("TEST 3 — Async mode (user gets answer instantly)")
print("-" * 40)

@monitor(
    fie_url="http://localhost:8000",
    model_name="gpt-4-async",
    async_mode=True,       # FIE runs in background
    log_results=True,
)
def async_llm(prompt: str) -> str:
    return "Quantum entanglement is when particles travel faster than light."

start = time.time()
answer = async_llm("Explain quantum entanglement.")
user_wait = round((time.time() - start) * 1000)
print(f"Answer to user: '{answer[:60]}...'")
print(f"User waited: {user_wait}ms (FIE runs in background)")
print("(FIE result will appear in a few seconds below)")
print()

# Wait for the background thread to finish
print("Waiting for background FIE analysis...")
time.sleep(120)   # give Ollama time to respond

print()
print("=" * 60)
print("All tests complete.")
print("Check MongoDB Atlas → fie_database → inferences")
print("You should see new records from these test calls.")
print("=" * 60)