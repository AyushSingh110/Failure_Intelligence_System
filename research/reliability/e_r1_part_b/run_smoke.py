"""
E-R1 Part B, Phase 5 — 30-item smoke run of the core arms (feasibility only).

    python -I research/reliability/e_r1_part_b/run_smoke.py

Reads ONLY data/fever/derived/smoke.jsonl. Never opens calibration or the test files.
Arms: V0, V1 (rescoring V0), V2, V3, CO (claim-only), C+, C- (manifest/PROTOCOL.md §5).

Network: the only permitted connection is the local Ollama server at 127.0.0.1:11434.
Every other connect / DNS lookup is refused and recorded, and fails the run.

Outputs:
  data/fever/runs/<stamp>_smoke/generations.jsonl       raw model text (git-ignored)
  e_r1_part_b/results/<stamp>_smoke_summary.json        aggregates only (tracked)
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OLLAMA = ("127.0.0.1", 11434)
MODEL = "llama3.1:8b"
SMOKE = REPO / "research/reliability/data/fever/derived/smoke.jsonl"
RUNS = REPO / "research/reliability/data/fever/runs"
RESULTS = REPO / "research/reliability/e_r1_part_b/results"
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 2048}
NUM_PREDICT = {"V0": 100, "V3": 160, "CO": 20}

# ── network guard: loopback Ollama only ──────────────────────────────────────
ATTEMPTS: list[str] = []
_real_connect, _real_gai = socket.socket.connect, socket.getaddrinfo


def _connect(self, address):
    if isinstance(address, tuple) and address[0] in ("127.0.0.1", "localhost") and address[1] == OLLAMA[1]:
        return _real_connect(self, address)
    ATTEMPTS.append(f"connect {address!r}")
    raise ConnectionRefusedError("E-R1 smoke: only 127.0.0.1:11434 is allowed")


def _gai(host, *a, **k):
    if host in ("127.0.0.1", "localhost"):
        return _real_gai(host, *a, **k)
    ATTEMPTS.append(f"getaddrinfo {host!r}")
    raise socket.gaierror("E-R1 smoke: DNS blocked")


socket.socket.connect, socket.getaddrinfo = _connect, _gai

from engine.reliability_contract import parse_verdict_strict          # noqa: E402
from engine.verifier import serper_verifier as sv                     # noqa: E402

# ── prompts ──────────────────────────────────────────────────────────────────
V0_TEMPLATE = """You are a fact-checker comparing a model output to a live search result.

Original question: {q}

Model output: {o}

Live search result (from Google): {s}

Answer with EXACTLY this format (3 lines):
VERDICT: CONSISTENT or INCONSISTENT or UNCERTAIN
CONFIDENCE: 0.0 to 1.0
GROUNDED_ANSWER: <most accurate answer based on the search result>"""

V3_TEMPLATE = """You are verifying a claim against an evidence passage. Use ONLY the passage; ignore anything else you know.

Claim: {claim}

Passage: {passage}

Choose one label:
SUPPORTED - the passage states information that makes the claim true.
REFUTED - the passage states information that makes the claim false.
NOT_ENOUGH_INFO - the passage does not settle the claim.

Answer with EXACTLY these two lines:
LABEL: SUPPORTED or REFUTED or NOT_ENOUGH_INFO
QUOTE: <the exact sentence or phrase from the passage that decides the label, or NONE>"""

CO_TEMPLATE = """Decide whether the following claim is true, using only your own knowledge.

Claim: {claim}

Answer with EXACTLY one line:
LABEL: SUPPORTED or REFUTED or NOT_ENOUGH_INFO"""


def check_v0_template_matches_production() -> None:
    """V0 must be the production prompt: every template line must occur in its source."""
    src = inspect.getsource(sv._verify_via_groq_detailed)
    lines = [l for l in V0_TEMPLATE.replace("{q}", "{prompt[:200]}").replace("{o}", "{primary_output[:300]}")
             .replace("{s}", "{search_answer[:400]}").splitlines() if l.strip()]
    missing = [l for l in lines if l not in src]
    if missing:
        raise SystemExit(f"V0 template drifted from production: {missing}")


def v0_prompt(item: dict) -> str:
    q = f"Is the following claim correct? {item['claim']}"
    return V0_TEMPLATE.format(q=q[:200], o=item["claim"][:300], s=item["passage_text"][:400])


# ── decisions ────────────────────────────────────────────────────────────────
S, C, N = "SUPPORTED", "CONTRADICTED", "NEI"
GOLD = {"SUPPORTS": S, "REFUTES": C, "NOT ENOUGH INFO": N}


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower().strip(" .\"'")


def parse_label(text: str) -> tuple[str | None, str]:
    labels = [l.split(":", 1)[1].strip().upper() for l in text.splitlines() if l.strip().upper().startswith("LABEL:")]
    quotes = [l.split(":", 1)[1].strip() for l in text.splitlines() if l.strip().upper().startswith("QUOTE:")]
    if len(labels) != 1 or labels[0] not in ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"):
        return None, quotes[0] if quotes else ""
    return labels[0], quotes[0] if quotes else ""


def decide_v3(text: str, passage: str) -> tuple[str, str]:
    label, quote = parse_label(text)
    if label is None:
        return N, "parse_failure"
    if label == "NOT_ENOUGH_INFO":
        return N, "explicit_nei"
    q = _norm(quote)
    if q in ("", "none") or len(q.split()) < 3 or q not in _norm(passage):
        return N, "quote_not_in_passage"
    return (S if label == "SUPPORTED" else C), "quote_ok"


def decide_co(text: str) -> tuple[str, str]:
    label, _ = parse_label(text)
    if label is None:
        return N, "parse_failure"
    return {"SUPPORTED": S, "REFUTED": C, "NOT_ENOUGH_INFO": N}[label], "explicit"


def decide_v0(text: str) -> str:
    matches, _conf, _ans = sv._parse_groq_verdict(text)
    return S if matches else C


def decide_v1(text: str) -> tuple[str, str]:
    p = parse_verdict_strict(text, "GROUNDED_ANSWER")
    return {"SUPPORTED": S, "CONTRADICTED": C}.get(p.state.value, N), p.reason


def decide_v2(item: dict) -> str:
    matches, _c, _s = sv._heuristic_match(item["claim"], item["passage_text"][:400])
    return S if matches else C


# ── Ollama ───────────────────────────────────────────────────────────────────
def generate(prompt: str, num_predict: int) -> tuple[str, float, dict]:
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {**OPTIONS, "num_predict": num_predict}}).encode()
    req = urllib.request.Request(f"http://{OLLAMA[0]}:{OLLAMA[1]}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    meta = {k: out.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count", "eval_count",
                                    "eval_duration", "done_reason")}
    return out.get("response", ""), dt, meta


def ollama_get(path: str) -> dict:
    with urllib.request.urlopen(f"http://{OLLAMA[0]}:{OLLAMA[1]}{path}", timeout=30) as r:
        return json.loads(r.read())


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                used, util = (int(x) for x in out.stdout.strip().split(","))
                self.samples.append((used, util))
            except Exception:
                pass
            self.stop.wait(1.0)


# ── metrics ──────────────────────────────────────────────────────────────────
def wilson_lb(k: int, n: int, z: float = 1.96) -> float | None:
    if n == 0:
        return None
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def metrics(golds: list[str], preds: list[str]) -> dict:
    cm = Counter(zip(golds, preds))
    out = {"confusion[gold->pred]": {g: {p: cm[(g, p)] for p in (S, C, N)} for g in (S, C, N)}}
    for lab in (S, C):
        tp, pp, gp = cm[(lab, lab)], sum(cm[(g, lab)] for g in (S, C, N)), sum(cm[(lab, p)] for p in (S, C, N))
        out[lab] = {"predicted": pp, "precision": tp / pp if pp else None, "precision_wilson_lb": wilson_lb(tp, pp),
                    "recall": tp / gp if gp else None}
    out["false_supported_on_refutes"] = cm[(C, S)]
    out["false_supported_on_nei"] = cm[(N, S)]
    out["coverage_non_nei"] = sum(1 for p in preds if p != N) / len(preds)
    out["accuracy"] = sum(cm[(x, x)] for x in (S, C, N)) / len(preds)
    return out


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    check_v0_template_matches_production()
    items = [json.loads(l) for l in SMOKE.read_text(encoding="utf-8").splitlines()]
    assert len(items) == 30 and all(i["role"] == "smoke" for i in items), "smoke file mismatch"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"{stamp}_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)

    tags = {m["name"]: m["digest"] for m in ollama_get("/api/tags")["models"]}
    version = ollama_get("/api/version")["version"]
    gpu = GpuSampler()
    gpu.start()
    t_start = time.perf_counter()

    # warm-up (not scored, not timed per item): load the model once
    _, warm_s, warm_meta = generate("Reply with OK.", 2)
    ps = ollama_get("/api/ps")

    rows, timings = [], {"V0": [], "V3": [], "CO": []}
    gens = (run_dir / "generations.jsonl").open("w", encoding="utf-8")
    for item in items:
        rec = {"item_id": item["item_id"], "gold": GOLD[item["label"]], "evidence_kind": item["evidence_kind"],
               "passage_chars": len(item["passage_text"]), "primary_set_sentences": item["primary_set_sentences"]}
        prompts = {"V0": v0_prompt(item),
                   "V3": V3_TEMPLATE.format(claim=item["claim"], passage=item["passage_text"][:2000]),
                   "CO": CO_TEMPLATE.format(claim=item["claim"])}
        out = {}
        for arm, prompt in prompts.items():
            text, dt, meta = generate(prompt, NUM_PREDICT[arm])
            timings[arm].append(dt)
            out[arm] = text
            gens.write(json.dumps({"item_id": item["item_id"], "arm": arm, "prompt_sha256": sha(prompt),
                                   "output": text, "seconds": dt, "ollama": meta}, ensure_ascii=False) + "\n")
        rec["V0"] = decide_v0(out["V0"])
        rec["V1"], rec["V1_reason"] = decide_v1(out["V0"])
        t0 = time.perf_counter()
        rec["V2"] = decide_v2(item)
        rec["V2_seconds"] = time.perf_counter() - t0
        rec["V3"], rec["V3_reason"] = decide_v3(out["V3"], item["passage_text"][:2000])
        rec["CO"], rec["CO_reason"] = decide_co(out["CO"])
        rec["C+"], rec["C-"] = S, N
        rows.append(rec)
        print(f"{item['item_id']:>22} gold={rec['gold']:<12} V0={rec['V0']:<12} V1={rec['V1']:<12} "
              f"V3={rec['V3']:<12} CO={rec['CO']}", flush=True)
    gens.close()
    wall = time.perf_counter() - t_start
    gpu.stop.set()
    gpu.join(timeout=5)
    (run_dir / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    golds = [r["gold"] for r in rows]
    per_arm = {arm: metrics(golds, [r[arm] for r in rows]) for arm in ("V0", "V1", "V2", "V3", "CO", "C+", "C-")}
    reasons = {arm: dict(Counter(r[f"{arm}_reason"] for r in rows)) for arm in ("V1", "V3", "CO")}
    raw_lines = [json.loads(l) for l in (run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    truncated = Counter(g["arm"] for g in raw_lines if (g["ollama"] or {}).get("done_reason") == "length")

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None

    summary = {
        "schema": "e-r1-partB-smoke/1",
        "run": stamp,
        "feasibility_only": True,
        "items": len(rows),
        "gold_counts": dict(Counter(golds)),
        "model": MODEL, "model_digest": tags.get(MODEL), "ollama_version": version,
        "options": OPTIONS, "num_predict": NUM_PREDICT,
        "prompt_template_sha256": {"V0": sha(V0_TEMPLATE), "V3": sha(V3_TEMPLATE), "CO": sha(CO_TEMPLATE)},
        "smoke_file_sha256": hashlib.sha256(SMOKE.read_bytes()).hexdigest(),
        "timing_seconds": {
            "wall_total": wall, "warmup_load": warm_s, "warmup_meta": warm_meta,
            **{arm: {"mean": sum(t) / len(t), "p50": pct(t, .5), "p95": pct(t, .95), "max": max(t)}
               for arm, t in timings.items()},
            "V2_mean": sum(r["V2_seconds"] for r in rows) / len(rows),
        },
        "hardware": {
            "gpu_mem_used_mib_max": max((s[0] for s in gpu.samples), default=None),
            "gpu_util_pct_max": max((s[1] for s in gpu.samples), default=None),
            "gpu_util_pct_mean": (sum(s[1] for s in gpu.samples) / len(gpu.samples)) if gpu.samples else None,
            "gpu_samples": len(gpu.samples),
            "ollama_ps": [{k: m.get(k) for k in ("name", "size", "size_vram")} for m in ps.get("models", [])],
        },
        "parse": reasons,
        "truncated_generations_by_arm": dict(truncated),
        "metrics": per_arm,
        "network_attempts_blocked": ATTEMPTS,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{stamp}_smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(out_path)
    print(json.dumps({"timing": summary["timing_seconds"], "hardware": summary["hardware"],
                      "parse": reasons, "net": ATTEMPTS}, indent=1, default=str))
    return 1 if ATTEMPTS else 0


if __name__ == "__main__":
    sys.exit(main())
