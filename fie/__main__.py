"""
fie CLI — run adversarial prompt scanning from the terminal.

Usage:
    python -m fie detect "Your prompt text here"
    fie detect "Your prompt text here"
    echo "prompt" | fie detect -

Options:
    --output json     Output raw JSON instead of colored summary
    --threshold 0.5   Minimum confidence to flag as attack (default: 0.5)
    --quiet           Exit code only (0 = safe, 1 = attack detected)
"""
from __future__ import annotations

import argparse
import json
import sys


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _red(t: str)    -> str: return _color(t, "91")
def _green(t: str)  -> str: return _color(t, "92")
def _yellow(t: str) -> str: return _color(t, "93")
def _bold(t: str)   -> str: return _color(t, "1")
def _dim(t: str)    -> str: return _color(t, "2")


def _cmd_detect(args: argparse.Namespace) -> int:
    from fie.adversarial import scan_prompt

    prompt = args.prompt
    if prompt == "-":
        prompt = sys.stdin.read()

    result = scan_prompt(prompt.strip())

    if args.output == "json":
        print(json.dumps({
            "is_attack":    result.is_attack,
            "attack_type":  result.attack_type,
            "category":     result.category,
            "confidence":   result.confidence,
            "layers_fired": result.layers_fired,
            "matched_text": result.matched_text,
            "mitigation":   result.mitigation,
            "evidence":     result.evidence,
        }, indent=2))
        return 1 if result.is_attack else 0

    if args.quiet:
        return 1 if result.is_attack else 0

    # Colored human-readable output
    if result.is_attack:
        status = _red(_bold("ATTACK DETECTED"))
        conf_str = _red(f"{result.confidence:.0%}")
    else:
        status = _green(_bold("SAFE"))
        conf_str = _green("0%")

    print()
    print(f"  {_bold('FIE Adversarial Scan')}")
    print(f"  {'─' * 40}")
    print(f"  Status     : {status}")

    if result.is_attack:
        print(f"  Attack type: {_yellow(result.attack_type or '')}")
        print(f"  Category   : {result.category or '—'}")
        print(f"  Confidence : {conf_str}")
        print(f"  Layers     : {', '.join(result.layers_fired) if result.layers_fired else '—'}")

        if result.matched_text:
            snippet = result.matched_text[:80] + ("…" if len(result.matched_text) > 80 else "")
            print(f"  Matched    : {_dim(repr(snippet))}")

        print()
        print(f"  {_bold('Mitigation')}")
        for line in (result.mitigation or "").split(". "):
            line = line.strip()
            if line:
                print(f"  • {line}{'.' if not line.endswith('.') else ''}")
    else:
        print(f"  Confidence : {conf_str}")

    print()
    return 1 if result.is_attack else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fie",
        description="FIE — Failure Intelligence Engine CLI",
    )
    sub = parser.add_subparsers(dest="command")

    detect_parser = sub.add_parser("detect", help="Scan a prompt for adversarial attacks")
    detect_parser.add_argument("prompt", help='Prompt text to scan, or "-" to read from stdin')
    detect_parser.add_argument("--output", choices=["text", "json"], default="text")
    detect_parser.add_argument("--threshold", type=float, default=0.5,
                               help="Minimum confidence to flag as attack (default: 0.5)")
    detect_parser.add_argument("--quiet", action="store_true",
                               help="Suppress output — exit code only (0=safe, 1=attack)")

    args = parser.parse_args()

    if args.command == "detect":
        sys.exit(_cmd_detect(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
