"""
FIE evaluation harness.

Measures the guardrail under named, fingerprinted configurations. It is not part
of the request-serving path and is not shipped in the wheel.

Run from the repository root:

    python -m evals baseline
"""

# Bump HARNESS_VERSION on any change to harness behaviour.
# Bump SCHEMA_VERSION when a deterministic artifact changes shape: two runs with
# different schema versions are never compared.
HARNESS_VERSION = "1.0.0"
SCHEMA_VERSION = 1
