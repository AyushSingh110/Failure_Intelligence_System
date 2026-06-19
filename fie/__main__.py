"""
fie CLI — run adversarial prompt scanning from the terminal.

Usage:
    fie detect "Your prompt text here"
    fie detect -                         # read from stdin
    fie explain "Your prompt text here"  # layer-by-layer breakdown
    fie benchmark harmbench              # run HarmBench eval with checkpoint/resume
    fie benchmark jailbreakbench         # run JailbreakBench eval with checkpoint/resume

Options (detect):
    --output json     Output raw JSON instead of colored summary
    --threshold 0.5   Minimum confidence to flag as attack (default: 0.5)
    --quiet           Exit code only (0 = safe, 1 = attack detected)

Options (explain):
    --output json     Output raw JSON instead of colored table

Options (benchmark):
    --resume          Resume from last checkpoint (skip already-scanned prompts)
    --n N             Max prompts per category (harmbench only)
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
                print(f"  * {line}{'.' if not line.endswith('.') else ''}")
    else:
        print(f"  Confidence : {conf_str}")

    print()
    return 1 if result.is_attack else 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from fie.adversarial import scan_prompt

    prompt = args.prompt
    if prompt == "-":
        prompt = sys.stdin.read()

    result = scan_prompt(prompt.strip())

    if args.output == "json":
        print(json.dumps({
            "is_attack":    result.is_attack,
            "attack_type":  result.attack_type,
            "confidence":   result.confidence,
            "layer_scores": result.layer_scores,
            "layers_fired": result.layers_fired,
            "evidence":     result.evidence,
            "mitigation":   result.mitigation,
        }, indent=2))
        return 1 if result.is_attack else 0

    # Colored layer-by-layer breakdown
    print()
    print(f"  {_bold('FIE Explain')}  --  layer-by-layer breakdown")
    print(f"  {'-' * 50}")
    ellipsis = "..." if len(prompt) > 80 else ""
    print(f"  Prompt   : {_dim(repr(prompt[:80] + ellipsis))}")
    print()

    # Sort layers: fired first (by confidence desc), then silent
    fired   = {name for name in result.layers_fired}
    scores  = result.layer_scores or {}
    all_layers = sorted(scores.keys(), key=lambda n: -scores.get(n, 0.0))

    print(f"  {'Layer':<22}  {'Score':>6}  {'Fired':>5}  Evidence")
    print(f"  {'-'*22}  {'-'*6}  {'-'*5}  {'-'*30}")

    for layer in all_layers:
        score    = scores.get(layer, 0.0)
        did_fire = layer in fired
        score_s  = f"{score:.3f}"
        fire_s   = _red("YES") if did_fire else _dim("no ")
        score_c  = _red(score_s) if did_fire else (_yellow(score_s) if score > 0.30 else _dim(score_s))

        ev = result.evidence.get(layer, {})
        ev_str = ""
        if isinstance(ev, dict):
            if "matched_text" in ev:
                ev_str = repr(ev["matched_text"][:40])
            elif "pair_probability" in ev:
                ev_str = f"pair_prob={ev['pair_probability']:.3f}"
            elif "danger_score" in ev:
                ev_str = f"danger={ev['danger_score']:.3f}"
            elif "translated_text" in ev:
                ev_str = f"translated: {str(ev['translated_text'])[:35]}"
            elif ev:
                first_key = next(iter(ev))
                ev_str = f"{first_key}={str(ev[first_key])[:30]}"

        print(f"  {layer:<22}  {score_c:>6}  {fire_s:>5}  {_dim(ev_str)}")

    print()
    if result.evidence.get("meta_classifier"):
        mc = result.evidence["meta_classifier"]
        print(f"  {_bold('Meta-classifier')} : prob={mc.get('probability', 0):.3f}  "
              f"threshold={mc.get('threshold', '-')}  source={mc.get('source', '-')}")
        print()

    if result.is_attack:
        print(f"  {_bold('Verdict')}  : {_red(_bold('ATTACK'))} -- {_yellow(result.attack_type or '')}  "
              f"(conf={result.confidence:.2f})")
        print()
        print(f"  {_bold('Mitigation')}")
        for line in (result.mitigation or "").split(". "):
            line = line.strip()
            if line:
                print(f"  * {line}{'.' if not line.endswith('.') else ''}")
    else:
        print(f"  {_bold('Verdict')}  : {_green(_bold('SAFE'))}  (max layer score={max(scores.values(), default=0):.3f})")

    print()
    return 1 if result.is_attack else 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    if args.suite == "harmbench":
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "eval_harmbench",
            os.path.join(os.path.dirname(__file__), "..", "data", "eval_harmbench.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Patch sys.argv so the module's argparse picks up our flags
        orig_argv = sys.argv
        sys.argv = ["fie benchmark harmbench"]
        if args.resume:
            sys.argv.append("--resume")
        if args.n:
            sys.argv += ["--n", str(args.n)]
        try:
            mod.main()
        finally:
            sys.argv = orig_argv
        return 0

    elif args.suite == "jailbreakbench":
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "eval_jailbreakbench",
            os.path.join(os.path.dirname(__file__), "..", "data", "eval_jailbreakbench.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        orig_argv = sys.argv
        sys.argv = ["fie benchmark jailbreakbench"]
        if args.resume:
            sys.argv += ["--resume", args.resume]
        try:
            mod.main()
        finally:
            sys.argv = orig_argv
        return 0

    else:
        print(f"Unknown benchmark suite: {args.suite}")
        print("Available: harmbench, jailbreakbench")
        return 1


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

    explain_parser = sub.add_parser("explain", help="Layer-by-layer breakdown of why a prompt was flagged")
    explain_parser.add_argument("prompt", help='Prompt text to explain, or "-" to read from stdin')
    explain_parser.add_argument("--output", choices=["text", "json"], default="text")

    bench_parser = sub.add_parser("benchmark", help="Run a standard benchmark evaluation")
    bench_parser.add_argument("suite", choices=["harmbench", "jailbreakbench"],
                              help="Benchmark suite to run")
    bench_parser.add_argument("--resume", nargs="?", const=True, default=None,
                              help="Resume from checkpoint. For jailbreakbench, pass the raw JSONL path.")
    bench_parser.add_argument("--n", type=int, default=None,
                              help="Max behaviors per category (harmbench only)")

    args = parser.parse_args()

    if args.command == "detect":
        sys.exit(_cmd_detect(args))
    elif args.command == "explain":
        sys.exit(_cmd_explain(args))
    elif args.command == "benchmark":
        sys.exit(_cmd_benchmark(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
