"""
E53 step 2 — embed every unique text exactly once with the verified production
ONNX encoder (prefix + text, one text per call, as fie/layers/pair.py does).

Outputs (exclusive-create):
  cache/embeddings.npy           float32 (n, 384), row order = cache/embedding_index.json
  cache/embedding_index.json     list of text sha256 in row order
  cache/embedding_provenance.json  encoder hashes, counts, output hashes, timing
"""
from __future__ import annotations

import json
import time

import e53_common as C


def main() -> None:
    import numpy as np

    pinned = C.verify_pinned()
    rows = C.read_jsonl(C.CACHE / "texts.jsonl")
    order = [r["sha256"] for r in rows]
    t0 = time.perf_counter()
    out = np.zeros((len(rows), 384), dtype=np.float32)
    for k, r in enumerate(rows):
        out[k] = C.embed_one(r["text"])
        if (k + 1) % 1000 == 0:
            print(f"{k + 1}/{len(rows)} {time.perf_counter() - t0:.0f}s", flush=True)
    elapsed = time.perf_counter() - t0

    with open(C.CACHE / "embeddings.npy", "xb") as f:
        np.save(f, out)
    C.write_json_new(C.CACHE / "embedding_index.json", order)
    C.write_json_new(C.CACHE / "embedding_provenance.json", {
        "encoder": "fie/onnx_encoder.py OnnxEncoder loaded by file path; one text per call; "
                   "prefix + text; normalize_embeddings=True",
        "prefix": C.PAIR_PREFIX,
        "pinned_artifacts": pinned,
        "onnx_encoder_py_sha256": C.sha256_file(C.REPO / "fie" / "onnx_encoder.py"),
        "texts_jsonl_sha256": C.sha256_file(C.CACHE / "texts.jsonl"),
        "n": len(rows), "dim": 384, "dtype": "float32",
        "embeddings_npy_sha256": C.sha256_file(C.CACHE / "embeddings.npy"),
        "elapsed_s": round(elapsed, 1),
        "norm_min": float(np.linalg.norm(out, axis=1).min()),
        "norm_max": float(np.linalg.norm(out, axis=1).max()),
    })
    C.log(f"s02 embedded {len(rows)} unique texts in {elapsed:.0f}s (one text per call, ONNX)")


if __name__ == "__main__":
    main()
