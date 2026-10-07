# EXECUTION_nnn_<short-name>

Copy this file to `EXECUTION_nnn_<short-name>.md` when a work package is approved. Fill every
section. Write "None" with a reason where a section does not apply; never delete a heading.
"Implemented successfully" is not an acceptable entry anywhere: say what changed, what was
measured, and what the numbers were.

# Work Package
WP-nnn — name. Link to its row in [ROADMAP.md](../ROADMAP.md) §8 and its acceptance criteria in §10.

# Date
Start and end, ISO format.

# Objective
One or two sentences. What is true after this package that was not true before.

# Approved Scope
Exactly what the owner approved, quoted or linked. List anything explicitly out of scope.
List every breaking change, named before it was made.

# Pre-implementation State
Branch, commit hash, `git status`, environment (Python and key package versions), model
hashes, and the test baseline (command, counts, duration).

# Files Changed
Table: path · added / modified / deleted · one-line reason. Paste `git diff --stat`.

# Implementation Performed
What was built, in the order it was built. Enough detail that another engineer could redo it.

# Design Decisions
Each choice made during the work: options considered, the one taken, why, and the decision
id added to the master log (`D-nnn`).

# Tests Executed
Every command run, including ones that failed.

# Test Results
Counts and durations. For failures: the test name and the message.

# Metrics Before
Harness report file and its configuration fingerprint. Attack recall on all five sets and
over-refusal on both benign sets, with 95% CIs. Latency. Any package-specific metric.

# Metrics After
The same table. Then the paired comparison: difference, CI, p-value for each row.
State plainly which rows got worse.

# Errors Encountered
Every error, in the order met, including dead ends.

# Root Cause
For each error: the actual cause, not the symptom.

# Resolution
For each error: the exact fix, with file and line.

# Regressions
Anything that got worse, whether or not it was fixed. Golden-file diffs explained line by line.

# Security Impact
New or removed attack surface, egress, stored data, credentials, dependencies.

# Product Impact
What an integrator sees differently. Compatibility. Migration notes.

# Research Impact
Which published numbers or claims this affects. Experiments it enables or invalidates.

# Documentation Updated
Each file touched, and confirmation that the master log was updated in the same session.

# Known Limitations
What this package does not solve, and what was measured only weakly.

# Next Recommended Work
The next package and why, plus anything discovered that should reorder the roadmap.
