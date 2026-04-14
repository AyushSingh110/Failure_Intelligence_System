import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fie.monitor import monitor

GROQ_API_KEY = "Key"
FIE_API_KEY  = "FIE_API_KEY"
FIE_URL      = "http://localhost:8000"


@monitor(
    fie_url       = FIE_URL,
    api_key       = FIE_API_KEY,
    model_name    = "groq-llama-3.3-70b",
    run_full_jury = True,
    mode          = "correct",   
    log_results   = True,
)
def chat(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model    = "llama-3.3-70b-versatile",
        messages = [{"role": "user", "content": prompt}],
        max_tokens  = 200,
        temperature = 0.0,
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    if sys.platform == "win32":
        os.system("")

    print("\n  FIE Chatbot  powered by Groq + Failure Intelligence Engine")
    print("  Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        answer = chat(prompt=user_input)
        print(f"\nAI:  {answer}\n")
