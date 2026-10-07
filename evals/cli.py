"""Command line for the evaluation harness. Commands are added step by step."""
from __future__ import annotations

import argparse

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
    parser.add_subparsers(dest="command")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_OK
    return EXIT_USAGE
