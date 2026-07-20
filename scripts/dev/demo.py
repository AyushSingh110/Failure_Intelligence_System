"""
FIE Live Demo
─────────────
Uses Groq's weakest model (llama-3.1-8b-instant) so FIE has more failures to catch.
Every prompt/response is sent to the FIE server and logged to the dashboard.

Setup (one-time):
    Set in .env:
        GROQ_API_KEY=gsk_...
        FIE_API_KEY=...        ← copy from dashboard → Settings → API Key
        FIE_URL=http://localhost:8000   ← or your deployed URL

Run:
    python -m tests.demo
"""
import sys
import os
import logging

# ── Load .env ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ── Logging: show FIE analysis in terminal ────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "  %(message)s",
    stream = sys.stdout,
)
# Silence noisy third-party loggers
for _noisy in ("httpx", "httpcore", "groq", "urllib3", "requests"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ── FIE monitor decorator ─────────────────────────────────────────────────────
from fie.monitor import monitor

@monitor(
    model_name    = "groq-llama-3.1-8b-instant",
    run_full_jury = True,
    mode          = "correct",   # FIE corrects wrong answers in real-time
    log_results   = True,
)
def chat(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model       = "llama-3.1-8b-instant",   # weakest Groq model → more failures caught
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = 300,
        temperature = 0.7,                       # some randomness → more mistakes
    )
    return resp.choices[0].message.content.strip()


# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        FIE  —  Failure Intelligence Engine  Demo             ║
║   Model : Groq  llama-3.1-8b-instant  (weakest, on purpose) ║
║   FIE   : {url:<45}║
║   Mode  : correct  (FIE fixes wrong answers in real-time)    ║
╚══════════════════════════════════════════════════════════════╝
  Dashboard → http://localhost:5173   (open in browser)
  Type  'exit'  to quit.
"""

# Prompts that tend to make the 8b model stumble — good for demo
SUGGESTED = [
    "What is 17 × 24?",
    "How many days are in 15 weeks?",
    "Is the Great Wall of China visible from space?",
    "What year did World War I end?",
    "What is the capital of Australia?",
    "Explain why the sky is green.",
    "If a train travels 90 km/h for 2.5 hours, how far does it go?",
]


def run():
    fie_url = os.getenv("FIE_URL", "http://localhost:8000")
    print(BANNER.format(url=fie_url))

    print("  Suggested prompts (copy-paste these for good demo screenshots):")
    for i, s in enumerate(SUGGESTED, 1):
        print(f"    {i}. {s}")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Bye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("  Bye.")
            break

        # Enter a number to use a suggested prompt
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(SUGGESTED):
                user_input = SUGGESTED[idx]
                print(f"You: {user_input}")

        print()
        answer = chat(prompt=user_input)
        print(f"AI:  {answer}")
        print()


if __name__ == "__main__":
    if sys.platform == "win32":
        os.system("")  # enable ANSI on Windows
    run()
