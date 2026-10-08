"""
Command line for the evaluation harness.

    python -m evals baseline               the one command: run everything, check the counts
    python -m evals run [options]          general runner
    python -m evals verify-determinism     run twice in separate processes, compare bytes
    python -m evals selftest               guard canaries, model hashes, dataset hashes
    python -m evals pin RUN_DIR            copy a canonical run into evals/baselines/
    python -m evals show RUN_DIR           print a run's report

Run it from a source checkout; no install step is needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evals import HARNESS_VERSION, SCHEMA_VERSION

# Exit codes. Each failure class has its own so a caller can tell them apart.
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_INCOMPLETE = 2          # a suite did not finish, or a scan raised
EXIT_MODEL_INTEGRITY = 3
EXIT_DATASET_INTEGRITY = 4
EXIT_HERMETIC = 5            # network/process attempt, or guard self-test failed
EXIT_BASELINE_MISMATCH = 6
EXIT_NONDETERMINISTIC = 7
EXIT_FINGERPRINT_MISMATCH = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="FIE evaluation harness — measures the guardrail under named, "
                    "fingerprinted configurations. Not part of the serving path.",
    )
    parser.add_argument("--version", action="version",
                        version=f"evals {HARNESS_VERSION} (schema {SCHEMA_VERSION})")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("selftest", help="guard canaries, model hashes, dataset hashes; no scanning")

    b = sub.add_parser("baseline", help="run every suite and check the standard counts")
    b.add_argument("--out", help="parent directory for the run (default: evals/runs/)")
    b.add_argument("--resume", metavar="RUN_DIR", help="finish an interrupted run")

    r = sub.add_parser("run", help="general runner")
    r.add_argument("--plan", default="baseline",
                   choices=["baseline", "deterministic", "standard", "reference"],
                   help="which suites to run (default: baseline)")
    r.add_argument("--profile", help="primary profile (default: the canonical reproducibility profile)")
    r.add_argument("--suites", help="comma-separated suite ids; marks the run non-canonical")
    r.add_argument("--limit", type=int, help="first N rows per part; marks the run non-canonical")
    r.add_argument("--out", help="parent directory for the run (default: evals/runs/)")
    r.add_argument("--resume", metavar="RUN_DIR", help="finish an interrupted run")
    r.add_argument("--hash-seed", default="0", help="PYTHONHASHSEED for the workers (default 0)")
    r.add_argument("--model-sha256", action="append", default=[], metavar="ROLE=SHA256",
                   help="measure an unpublished model; marks the run non-canonical")

    v = sub.add_parser("verify-determinism",
                       help="run the deterministic suites twice in separate processes and compare")
    v.add_argument("--all", action="store_true",
                   help="every deterministic suite (required before pinning); default: standard suites")
    v.add_argument("--profile", help="primary profile (default: canonical)")
    v.add_argument("--out", help="parent directory for the runs (default: evals/runs/)")

    p = sub.add_parser("pin", help="copy a canonical run's deterministic files into evals/baselines/")
    p.add_argument("run_dir")
    p.add_argument("--canonical", action="store_true", help="pin as the canonical baseline (BL-…)")
    p.add_argument("--reference", action="store_true", help="pin as a reference reproduction (REF-…)")
    p.add_argument("--determinism-proof", metavar="FILE",
                   help="DETERMINISM.json written by `verify-determinism --all`")

    s = sub.add_parser("show", help="print a run's or a baseline's report")
    s.add_argument("run_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_OK

    if args.command == "show":
        report = Path(args.run_dir) / "REPORT.md"
        if not report.exists():
            print(f"no REPORT.md in {args.run_dir}", file=sys.stderr)
            return EXIT_USAGE
        sys.stdout.buffer.write(report.read_bytes())
        return EXIT_OK

    # Everything below runs under the orchestrator's guard.
    from evals import commands
    return commands.dispatch(args)
