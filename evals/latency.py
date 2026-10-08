"""
Timing statistics. Everything here is RUN METADATA.

Latency depends on the machine, its load and its package builds. It never enters
a deterministic artifact, and two latency results are comparable only when the
environment key matches.

Standard library only.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already sorted list; q in [0, 100]."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def describe(values: list[float]) -> dict:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean_ms": round(sum(vals) / len(vals), 2),
        "p50_ms": round(percentile(vals, 50), 2),
        "p95_ms": round(percentile(vals, 95), 2),
        "p99_ms": round(percentile(vals, 99), 2),
        "max_ms": round(vals[-1], 2),
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def suite_timing(run_dir: Path, suite_id: str) -> dict | None:
    """Per-prompt wall time recorded while a scan suite produced its records."""
    rows = _read_jsonl(run_dir / "timing" / f"{suite_id}.jsonl")
    if not rows:
        return None
    ms = [r["ms"] for r in rows]
    total_s = sum(ms) / 1000.0
    out = describe(ms)
    out["scan_time_s"] = round(total_s, 2)
    out["scans_per_s"] = round(len(ms) / total_s, 1) if total_s else None
    out["mean_chars"] = round(sum(r["chars"] for r in rows) / len(rows))
    return out


def warm_latency(run_dir: Path, suite_id: str = "latency") -> dict | None:
    """
    The latency suite: each prompt's MEDIAN across the measured passes, then
    statistics over prompts. Also grouped by variant, which gives length buckets.
    """
    rows = _read_jsonl(run_dir / "nondeterministic" / f"{suite_id}.jsonl")
    if not rows:
        return None
    per_prompt: dict[int, list[float]] = defaultdict(list)
    info: dict[int, dict] = {}
    for r in rows:
        per_prompt[r["idx"]].append(r["ms"])
        info[r["idx"]] = r
    medians = {idx: median(v) for idx, v in per_prompt.items()}
    overall = describe(list(medians.values()))
    overall["passes"] = max(r["pass"] for r in rows)
    buckets: dict[str, list[float]] = defaultdict(list)
    chars: dict[str, list[int]] = defaultdict(list)
    for idx, value in medians.items():
        key = f"{info[idx]['dataset']} / {info[idx]['variant']}"
        buckets[key].append(value)
        chars[key].append(info[idx]["chars"])
    by_bucket = {}
    for key in buckets:
        d = describe(buckets[key])
        d["mean_chars"] = round(sum(chars[key]) / len(chars[key]))
        by_bucket[key] = d
    return {"overall": overall, "by_bucket": by_bucket,
            "protocol": "result cache cleared before every scan; one discarded pass; "
                        "per-prompt median over the measured passes"}


def stability(run_dir: Path, suite_id: str = "stability") -> dict | None:
    """Prompts whose verdict changed between identical unseeded passes."""
    rows = _read_jsonl(run_dir / "nondeterministic" / f"{suite_id}.jsonl")
    if not rows:
        return None
    out = {}
    for ds_id in dict.fromkeys(r["dataset"] for r in rows):
        passes = [r for r in rows if r["dataset"] == ds_id]
        sets = [set(p["flagged_idx"]) for p in passes]
        union, common = set().union(*sets), set.intersection(*sets)
        out[ds_id] = {
            "n": passes[0]["n"], "passes": len(passes),
            "flagged_per_pass": [len(s) for s in sets],
            "unstable_idx": sorted(union - common),
        }
    return out
