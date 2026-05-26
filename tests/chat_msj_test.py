"""
Terminal chat tester for FIE adversarial detection.

Flow per message:
  1. Run scan_prompt() on the user input (updated MSJ + injection layer)
  2. If BLOCKED  → print detection report, do NOT send to model
  3. If UNCERTAIN → print warning, send to model with CWD notice in system prompt
  4. If CLEAR    → send to model normally
  5. Log to FIE dashboard via /monitor if FIE_API_KEY is set in .env

Model: llama-3.1-8b-instant (fast, weak, ideal for seeing jailbreak susceptibility)

Run:
    python tests/chat_msj_test.py
"""
from __future__ import annotations

import os
import sys
import time
import textwrap
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from fie.adversarial import scan_prompt, build_cwd_injection, _ATTACK_THRESHOLDS

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
FIE_API_KEY   = os.environ.get("FIE_API_KEY", "")
FIE_API_URL   = os.environ.get("FIE_API_URL", "http://localhost:8000")
GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
CHAT_MODEL    = "llama-3.1-8b-instant"   # weak model — shows jailbreak susceptibility clearly
MAX_HISTORY   = 20                        # turns to keep in context window

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

DIVIDER = DIM + "─" * 72 + RESET


def _wrap(text: str, width: int = 72, indent: str = "  ") -> str:
    lines = []
    for para in text.split("\n"):
        if para.strip():
            lines.extend(textwrap.wrap(para, width=width, initial_indent=indent, subsequent_indent=indent))
        else:
            lines.append("")
    return "\n".join(lines)


def _print_header() -> None:
    print()
    print(BOLD + CYAN + "  FIE Adversarial Chat Tester" + RESET)
    print(CYAN + "  Model: " + CHAT_MODEL + RESET)
    print(CYAN + "  Dashboard: " + (FIE_API_URL if FIE_API_KEY else "disabled (set FIE_API_KEY in .env)") + RESET)
    print(DIVIDER)
    print(DIM + "  Commands: 'quit' to exit | 'clear' to reset history | 'history' to show turns" + RESET)
    print(DIVIDER)
    print()


def _print_block_report(result, prompt: str) -> None:
    threshold = _ATTACK_THRESHOLDS.get(result.attack_type, 0.65)
    print()
    print(BOLD + RED + "  !! BLOCKED — ADVERSARIAL PROMPT DETECTED" + RESET)
    print(RED + DIVIDER + RESET)
    print(f"  {BOLD}Attack type :{RESET}  {result.attack_type}")
    print(f"  {BOLD}Confidence  :{RESET}  {result.confidence:.4f}  (threshold: {threshold})")
    print(f"  {BOLD}Layers fired:{RESET}  {', '.join(result.layers_fired)}")

    # MSJ-specific detail
    ev = result.evidence
    msj_ev = ev.get("many_shot", {})
    if msj_ev:
        print()
        print(f"  {BOLD}MSJ Evidence:{RESET}")
        print(f"    Pair count        : {msj_ev.get('pair_count', 'N/A')}")
        print(f"    Power-law danger  : {msj_ev.get('power_law_danger', 'N/A')}")
        print(f"    Harmful ratio     : {msj_ev.get('harmful_ratio', 'N/A')}")
        print(f"    Behavioral drift  : {msj_ev.get('behavioral_drift', 'N/A')}")
        print(f"    Q entropy         : {msj_ev.get('q_entropy', 'N/A')}")
        print(f"    A-turn compliance : {msj_ev.get('a_turn_compliance', 'N/A')}")
        print(f"    Escalation        : {msj_ev.get('escalation', 'N/A')}")
        print(f"    Fiction framing   : {msj_ev.get('fiction_framing', 'N/A')}")
        signals = msj_ev.get("signals_fired", [])
        if signals:
            print(f"\n  {BOLD}Signals fired:{RESET}")
            for s in signals:
                print(f"    - {s}")
        last_q = msj_ev.get("last_q_preview", "")
        if last_q:
            print(f"\n  {BOLD}Final query preview:{RESET}")
            print(_wrap(last_q[:200]))

    # Regex/injection detail
    regex_ev = ev.get("regex", {})
    if regex_ev.get("matched_text"):
        print(f"\n  {BOLD}Matched pattern:{RESET}  \"{regex_ev['matched_text'][:120]}\"")

    if ev.get("cwd_recommended"):
        print()
        print(YELLOW + "  CWD injection recommended for this session." + RESET)

    print()
    print(f"  {BOLD}Mitigation:{RESET}")
    print(_wrap(result.mitigation or "Review and harden your input sanitization."))
    print()
    print(RED + DIVIDER + RESET)
    print()


def _print_uncertain_report(result) -> None:
    threshold = _ATTACK_THRESHOLDS.get(result.attack_type, 0.65)
    print()
    print(BOLD + YELLOW + "  !! UNCERTAIN — Suspicious prompt (CWD injected, sending to model)" + RESET)
    print(YELLOW + DIVIDER + RESET)
    print(f"  Attack type : {result.attack_type}")
    print(f"  Confidence  : {result.confidence:.4f}  (threshold: {threshold})")
    print(f"  Layers fired: {', '.join(result.layers_fired)}")
    print(YELLOW + DIVIDER + RESET)
    print()


def _call_groq(messages: list[dict]) -> tuple[str, float]:
    """Call Groq, return (response_text, latency_ms)."""
    t0 = time.perf_counter()
    resp = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model":       CHAT_MODEL,
            "messages":    messages,
            "temperature": 0.7,
            "max_tokens":  1024,
        },
        timeout=30,
    )
    latency = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    return text, latency


def _log_to_dashboard(prompt: str, output: str, scan_result, latency_ms: float) -> str:
    """Log to FIE /monitor dashboard. Returns status string for display."""
    if not FIE_API_KEY:
        return "skipped (no FIE_API_KEY)"
    try:
        r = requests.post(
            f"{FIE_API_URL}/api/v1/monitor",
            headers={"X-API-Key": FIE_API_KEY, "Content-Type": "application/json"},
            json={
                "prompt":             prompt,
                "primary_output":     output,
                "primary_model_name": CHAT_MODEL,
                "latency_ms":         latency_ms,
                "run_full_jury":      False,
            },
            timeout=10,
        )
        if r.status_code == 200:
            return "logged OK"
        return f"HTTP {r.status_code}: {r.text[:80]}"
    except Exception as e:
        return f"failed: {str(e)[:80]}"


def main() -> None:
    if not GROQ_API_KEY:
        print(RED + "ERROR: GROQ_API_KEY not set in .env" + RESET)
        sys.exit(1)

    _print_header()

    history: list[dict] = []   # OpenAI-format message history
    turn_num = 0

    while True:
        try:
            user_input = input(BOLD + "You  > " + RESET).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + DIM + "  Session ended." + RESET)
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print(DIM + "  Goodbye." + RESET)
            break

        if user_input.lower() == "clear":
            history.clear()
            turn_num = 0
            print(DIM + "  History cleared." + RESET + "\n")
            continue

        if user_input.lower() == "history":
            print()
            for i, m in enumerate(history):
                role  = m["role"].upper()
                snip  = m["content"][:120].replace("\n", " ")
                print(f"  [{i}] {role}: {snip}...")
            print()
            continue

        turn_num += 1
        print()

        # ── Step 1: Adversarial scan ──────────────────────────────────────────
        t_scan = time.perf_counter()
        scan   = scan_prompt(user_input)
        scan_ms = (time.perf_counter() - t_scan) * 1000

        msj_threshold  = _ATTACK_THRESHOLDS.get("MANY_SHOT_JAILBREAK", 0.68)
        type_threshold = _ATTACK_THRESHOLDS.get(scan.attack_type or "", 0.65)
        uncertain_floor = type_threshold * 0.60

        # ── Step 2: BLOCKED ───────────────────────────────────────────────────
        if scan.is_attack:
            _print_block_report(scan, user_input)
            dash_status = _log_to_dashboard(user_input, "[BLOCKED]", scan, scan_ms)
            print(DIM + f"  dashboard: {dash_status}" + RESET + "\n")
            continue

        # ── Step 3: UNCERTAIN — send with CWD injected ────────────────────────
        cwd_active = False
        if scan.attack_type and scan.confidence >= uncertain_floor:
            _print_uncertain_report(scan)
            cwd_active = True

        # ── Step 4: Build messages for Groq ───────────────────────────────────
        system_msg = {"role": "system", "content": "You are a helpful assistant."}
        messages   = [system_msg] + history[-MAX_HISTORY:] + [{"role": "user", "content": user_input}]

        if cwd_active:
            messages = build_cwd_injection(messages)

        # ── Step 5: Call Groq ─────────────────────────────────────────────────
        try:
            ai_reply, latency_ms = _call_groq(messages)
        except Exception as exc:
            print(RED + f"  Groq error: {exc}" + RESET + "\n")
            continue

        # ── Step 6: Display response ──────────────────────────────────────────
        print(BOLD + "AI   > " + RESET, end="")
        # Print with wrapping, aligned to indent
        lines = ai_reply.split("\n")
        print(lines[0])
        for line in lines[1:]:
            if line.strip():
                print("       " + line)
            else:
                print()

        # Scan status footer
        scan_label = (
            GREEN + "CLEAR" + RESET if not scan.attack_type else
            YELLOW + f"UNCERTAIN ({scan.attack_type} {scan.confidence:.3f})" + RESET
        )
        dash_status = _log_to_dashboard(user_input, ai_reply, scan, latency_ms)
        print()
        print(DIM + f"  scan={scan_label}{DIM}  model_latency={latency_ms:.0f}ms  scan_latency={scan_ms:.1f}ms" + RESET)
        print(DIM + f"  dashboard: {dash_status}" + RESET)
        print(DIVIDER)
        print()

        # ── Step 7: Update history ────────────────────────────────────────────
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": ai_reply})

        # ── Step 8: Log to dashboard (best-effort) ────────────────────────────
        _log_to_dashboard(user_input, ai_reply, scan, latency_ms)


if __name__ == "__main__":
    main()
