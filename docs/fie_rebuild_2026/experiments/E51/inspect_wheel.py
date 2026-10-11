"""
E51 static wheel inspection. Reads the archive; imports and executes nothing from it.

    python inspect_wheel.py --manifest scripts/model_manifest.json --out <file.json> <label>=<wheel> [...]

For each wheel: identity, modules, metadata, entry points, start-up code, every model file with
its SHA-256 and manifest status, the classifier loader's preference list as written in the
wheel's own source, and the resulting default load set.

"verified" means: the file's SHA-256 equals the manifest entry for that path. Listing and
hashing is all that happens here; no model file is opened by an unpickler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

MODEL_SUFFIXES = (".pkl", ".joblib", ".pickle", ".onnx", ".bin", ".safetensors", ".pt")


def inspect(label: str, wheel: Path, manifest: dict) -> dict:
    data = wheel.read_bytes()
    z = zipfile.ZipFile(wheel)
    names = z.namelist()
    listed = {a["path"]: a["sha256"] for a in manifest["artifacts"]}

    def text(name: str) -> str:
        return z.read(name).decode("utf-8", errors="replace")

    meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
    meta = text(meta_name)
    ep_name = next((n for n in names if n.endswith(".dist-info/entry_points.txt")), None)

    models = []
    for n in sorted(names):
        if n.lower().endswith(MODEL_SUFFIXES):
            sha = hashlib.sha256(z.read(n)).hexdigest()
            if n not in listed:
                status = "unmanifested"
            else:
                status = "verified" if listed[n] == sha else "hash_mismatch"
            models.append({"path": n, "size": z.getinfo(n).file_size, "sha256": sha, "manifest_status": status})

    # The loader's preference order, read from the wheel's own source text (not executed).
    loader_file = next((n for n in ("fie/layers/pair.py", "fie/adversarial.py") if n in names
                        and "pair_intent_classifier" in text(n)), None)
    preference = []
    if loader_file:
        for name in re.findall(r"pair_intent_classifier[\w.]*?\.pkl", text(loader_file)):
            if name not in preference:
                preference.append(name)
    present = {Path(m["path"]).name: m for m in models}
    default_classifier = next((p for p in preference if p in present), None)
    load_set = [present[n] for n in (default_classifier, "meta_clf.pkl") if n and n in present]
    forced_env = "FIE_PAIR_VERSION" in (text(loader_file) if loader_file else "")

    return {
        "label": label, "filename": wheel.name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
        "file_count": len(names),
        "metadata": {
            "version": re.search(r"^Version: (.+)$", meta, re.M).group(1).strip(),
            "requires_python": (re.search(r"^Requires-Python: (.+)$", meta, re.M) or [None, None])[1],
            "requires_dist": re.findall(r"^Requires-Dist: (.+)$", meta, re.M),
            "provides_extra": re.findall(r"^Provides-Extra: (.+)$", meta, re.M),
        },
        "entry_points": text(ep_name) if ep_name else None,
        "fie_modules": sorted(n for n in names if n.startswith("fie/") and n.endswith(".py")),
        "has": {k: (v in names) for k, v in {
            "fie.policy": "fie/policy.py", "fie.layers.pair": "fie/layers/pair.py",
            "fie.onnx_encoder": "fie/onnx_encoder.py", "fie._lite": "fie/_lite.py"}.items()},
        "startup_code": [n for n in names if n.endswith(".pth") or Path(n).name in ("sitecustomize.py",
                                                                                   "usercustomize.py")],
        "model_files": models,
        "model_counts": {s: sum(1 for m in models if m["manifest_status"] == s)
                         for s in ("verified", "unmanifested", "hash_mismatch")},
        "encoder_files_present": [m["path"] for m in models if m["path"].endswith(".onnx")],
        "loader": {
            "source_file": loader_file, "preference_order": preference,
            "honours_FIE_PAIR_VERSION": forced_env,
            "default_classifier": default_classifier,
            "default_load_set": [{"path": m["path"], "manifest_status": m["manifest_status"]} for m in load_set],
            "default_load_set_all_verified": bool(load_set) and all(m["manifest_status"] == "verified"
                                                                    for m in load_set),
            "determined_statically": loader_file is not None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("wheels", nargs="+", help="label=path")
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    result = {}
    for item in args.wheels:
        label, path = item.split("=", 1)
        result[label] = inspect(label, Path(path), manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    for label, r in result.items():
        print(label, r["sha256"][:16], r["size"], "files", r["file_count"], "models", r["model_counts"],
              "default", r["loader"]["default_classifier"], "has", r["has"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
