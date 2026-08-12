"""
Measure what the meta-classifier actually contributes, with confidence intervals.

Three configurations on the same held-out prompts:

    off      FIE_DISABLE_META=1        weighted vote only
    broken   the shipped model         6 of 11 features constant zero, never
                                       sees pair_classifier
    fixed    retrained model           features derived from the live pipeline

Answers two questions that cannot be answered by staring at the code:

  1. Did fixing the feature names help, hurt, or do nothing?
  2. Does the meta-classifier beat plain weighted voting at all? If it does
     not, the honest move is to delete it — an ablation result, not a failure.

Every comparison is a PAIRED bootstrap on identical prompts, because the
question is "does B differ from A on the same inputs", and comparing two
independent intervals for overlap is a weaker, wronger test.

Usage:
    python scripts/eval_meta_impact.py --eval data/benchmark_audit/jbb_clean.jsonl
    python scripts/eval_meta_impact.py --eval <attacks.jsonl> --benign <benign.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def load_jsonl(path: Path, default_label: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = o.get("prompt") or o.get("text") or o.get("goal") or ""
            if not prompt:
                continue
            if default_label is not None:
                label = default_label
            else:
                raw = o.get("label", o.get("is_attack", 1))
                label = int(bool(raw))
            rows.append({"prompt": prompt, "label": label})
    return rows


def score_in_subprocess(prompts: list[str], env_extra: dict) -> list[int]:
    """
    Run scans in a fresh interpreter.

    A subprocess is used because the meta-classifier is cached in module state
    after first load — flipping an env var in-process would not reload it, and
    the second configuration would silently reuse the first one's model. This
    is the same class of bug the whole exercise is about, so it is not a risk
    worth taking for convenience.
    """
    script = (
        "import json,sys,os;"
        "sys.path.insert(0, r'%s');"
        "from fie.adversarial import scan_prompt, warmup;"
        "warmup();"
        "prompts=json.load(sys.stdin);"
        "print('@@'+json.dumps([int(scan_prompt(p).is_attack) for p in prompts]))"
    ) % str(ROOT)

    env = {**os.environ, **env_extra, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(prompts), text=True, capture_output=True, env=env,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise RuntimeError(f"scoring subprocess produced no result:\n{proc.stderr[-1500:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="JSONL of attack prompts")
    ap.add_argument("--benign", help="JSONL of benign prompts (for FPR/over-refusal)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows for a quick run")
    ap.add_argument("--out", default="data/meta_impact_report.json")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.eval), default_label=1)
    if args.benign:
        rows += load_jsonl(Path(args.benign), default_label=0)
    if args.limit:
        rows = rows[: args.limit]

    prompts = [r["prompt"] for r in rows]
    y_true = [r["label"] for r in rows]
    n_att = sum(y_true)
    print(f"evaluating {len(rows)} prompts ({n_att} attack / {len(rows) - n_att} benign)\n")

    configs = {
        "meta_off": {"FIE_DISABLE_META": "1"},
        "meta_on":  {"FIE_DISABLE_META": "0"},
    }
    preds = {}
    for name, env in configs.items():
        print(f"  scoring [{name}] ...")
        preds[name] = score_in_subprocess(prompts, env)

    from stats_utils import bootstrap_ci, format_ci, paired_bootstrap_diff

    report = {"n": len(rows), "n_attack": n_att, "n_benign": len(rows) - n_att,
              "configs": {}, "comparisons": []}

    metrics = ["recall", "precision", "f1"]
    if len(rows) - n_att > 0:
        metrics.append("over_refusal")

    print()
    for name, p in preds.items():
        report["configs"][name] = {}
        print(f"[{name}]")
        for m in metrics:
            point, lo, hi = bootstrap_ci(y_true, p, metric=m)
            report["configs"][name][m] = {"point": point, "ci_low": lo, "ci_high": hi}
            print(f"   {m:13s} {format_ci(point, lo, hi)}")
        print()

    print("[paired comparison: meta_on vs meta_off]")
    for m in metrics:
        r = paired_bootstrap_diff(y_true, preds["meta_off"], preds["meta_on"], metric=m)
        report["comparisons"].append(r)
        verdict = "SIGNIFICANT" if r["significant"] else "not significant"
        print(f"   {m:13s} {r['difference']:+.4f}  "
              f"CI[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]  p={r['p_value']:.4f}  {verdict}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
