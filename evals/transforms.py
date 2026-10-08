"""
Suite registry and deterministic input construction.

A suite is one or more PARTS. A part names a dataset, an optional row selection,
and an ordered list of VARIANTS. A variant is a pure function of the base prompt:

    base                the prompt as stored
    pad_after:<words>   prompt, blank line, then the filler sentence repeated
    pad_before:<words>  the filler sentence repeated, blank line, then the prompt
    frame:<template>    the prompt inserted into a fixed template

Items are produced in a fixed order — parts as listed, variants as listed, rows
in file order — so a suite is the same sequence on every run and for every model.
Row selection is by POSITION, never by what the subject does with the row.

Suite kinds map to code through a fixed table in the worker. Nothing in a
registry is ever imported, evaluated or executed.

Standard library only.
"""
from __future__ import annotations

from pathlib import Path

from evals import canonical
from evals.datasets import Dataset, DatasetIntegrityError

SUITES_PATH = "evals/registry/suites.json"
PROFILES_PATH = "evals/registry/profiles.json"
SUITE_KINDS = ("scan", "latency", "stability")
PRIMARY = "@primary"


class RegistryError(Exception):
    """A suite or profile registry is malformed."""


# ── registries ───────────────────────────────────────────────────────────────

def load_profiles(repo_root: str | Path) -> dict:
    try:
        reg = canonical.read_json(Path(repo_root) / PROFILES_PATH)
    except (OSError, ValueError) as exc:
        raise RegistryError(f"cannot read {PROFILES_PATH}: {exc}") from exc
    if reg.get("canonical_profile") not in reg.get("profiles", {}):
        raise RegistryError("profiles.json: canonical_profile is not a defined profile")
    for name, prof in reg["profiles"].items():
        for key in ("title", "models", "required_roles", "blocked_imports", "translation",
                    "langdetect_seed", "env", "expect"):
            if key not in prof:
                raise RegistryError(f"profile '{name}' has no '{key}'")
    return reg


def load_suites(repo_root: str | Path) -> dict:
    try:
        reg = canonical.read_json(Path(repo_root) / SUITES_PATH)
    except (OSError, ValueError) as exc:
        raise RegistryError(f"cannot read {SUITES_PATH}: {exc}") from exc
    seen = set()
    for suite in reg.get("suites", []):
        for key in ("id", "group", "kind", "profile", "title", "parts"):
            if key not in suite:
                raise RegistryError(f"suite {suite.get('id', '?')!r} has no '{key}'")
        if suite["kind"] not in SUITE_KINDS:
            raise RegistryError(f"suite '{suite['id']}': unknown kind {suite['kind']!r}")
        if suite["id"] in seen:
            raise RegistryError(f"duplicate suite id: {suite['id']}")
        seen.add(suite["id"])
        for part in suite["parts"]:
            if "dataset" not in part:
                raise RegistryError(f"suite '{suite['id']}': a part has no dataset")
            for variant in part.get("variants", ["base"]):
                parse_variant(variant)
    return reg


def suite_datasets(suite: dict) -> list[str]:
    out = []
    for part in suite["parts"]:
        if part["dataset"] not in out:
            out.append(part["dataset"])
    return out


def suite_definition_hash(suite: dict) -> str:
    return canonical.hash_obj(suite)


# ── fixtures ─────────────────────────────────────────────────────────────────

class Fixtures:
    def __init__(self, filler: str = "", filler_words: int = 0, templates: dict | None = None,
                 hashes: dict | None = None) -> None:
        self.filler = filler
        self.filler_words = filler_words
        self.templates = templates or {}
        self.hashes = hashes or {}


def load_fixtures(repo_root: str | Path, suites_registry: dict) -> Fixtures:
    """Load and verify the fixtures the suite registry declares. Hashes are of content."""
    spec = suites_registry.get("fixtures", {})
    root = Path(repo_root)
    fx = Fixtures()

    pad = spec.get("padding_filler")
    if pad:
        raw = canonical.read_lf(root / pad["path"]).decode("utf-8")
        filler = raw.rstrip("\n")
        digest = canonical.sha256_bytes(filler.encode("utf-8"))
        if digest != pad["sha256"]:
            raise DatasetIntegrityError(
                f"fixture {pad['path']} is not the pinned filler (content hash {digest[:16]}…)")
        if len(filler.split()) != pad["words"]:
            raise DatasetIntegrityError(
                f"fixture {pad['path']}: {len(filler.split())} words, registry says {pad['words']}")
        fx.filler, fx.filler_words = filler, pad["words"]
        fx.hashes["padding_filler"] = digest

    frames = spec.get("framing_templates")
    if frames:
        doc = canonical.read_json(root / frames["path"])
        digest = canonical.hash_obj(doc)
        if digest != frames["content_sha256"]:
            raise DatasetIntegrityError(
                f"fixture {frames['path']} is not the pinned template set (content hash {digest[:16]}…)")
        for name, tpl in doc["templates"].items():
            if tpl["template"].count("{p}") != 1:
                raise DatasetIntegrityError(f"template '{name}' must contain exactly one {{p}}")
        fx.templates = {name: tpl["template"] for name, tpl in doc["templates"].items()}
        fx.hashes["framing_templates"] = digest
    return fx


# ── variants ─────────────────────────────────────────────────────────────────

def parse_variant(variant: str) -> tuple[str, str | None]:
    if variant == "base":
        return "base", None
    kind, _, arg = variant.partition(":")
    if kind in ("pad_after", "pad_before") and arg.isdigit() and int(arg) > 0:
        return kind, arg
    if kind == "frame" and arg:
        return kind, arg
    raise RegistryError(f"unknown variant: {variant!r}")


def apply_variant(prompt: str, variant: str, fx: Fixtures) -> str:
    kind, arg = parse_variant(variant)
    if kind == "base":
        return prompt
    if kind in ("pad_after", "pad_before"):
        words = int(arg)
        if not fx.filler or words % fx.filler_words:
            raise RegistryError(
                f"variant {variant!r}: word count must be a multiple of the filler's "
                f"{fx.filler_words or '?'} words")
        padding = fx.filler * (words // fx.filler_words)
        return prompt + "\n\n" + padding if kind == "pad_after" else padding + "\n\n" + prompt
    template = fx.templates.get(arg)
    if template is None:
        raise RegistryError(f"variant {variant!r}: no such framing template")
    return template.replace("{p}", prompt)


def select_rows(ds: Dataset, select: dict | None) -> list[tuple[int, dict]]:
    """Choose rows by position. Returns (source_idx, row) pairs in file order."""
    rows = list(enumerate(ds.rows))
    if not select:
        return rows
    if "first" in select:
        return rows[: int(select["first"])]
    if "first_unique" in select:
        out, seen = [], set()
        for idx, row in rows:
            if row["prompt"] in seen:
                continue
            seen.add(row["prompt"])
            out.append((idx, row))
            if len(out) == int(select["first_unique"]):
                break
        return out
    raise RegistryError(f"unknown row selection: {select!r}")


def build_items(suite: dict, datasets: dict[str, Dataset], fx: Fixtures,
                limit: int | None = None) -> list[dict]:
    """
    Expand a suite into its ordered list of inputs.

    `limit` keeps only the first N rows of each (part, variant). It exists for
    smoke tests; a limited run is never canonical.
    """
    items: list[dict] = []
    for part in suite["parts"]:
        ds = datasets[part["dataset"]]
        rows = select_rows(ds, part.get("select"))
        if limit is not None:
            rows = rows[:limit]
        tag = part.get("tag")
        for variant in part.get("variants", ["base"]):
            label = f"{variant}+{tag}" if tag else variant
            for source_idx, row in rows:
                text = apply_variant(row["prompt"], variant, fx)
                items.append({
                    "idx": len(items),
                    "dataset": ds.id,
                    "source_idx": source_idx,
                    "variant": label,
                    "expected": ds.label,
                    "text": text,
                    "input_sha256": canonical.sha256_bytes(text.encode("utf-8")),
                    "runtime": part.get("runtime", {}),
                })
    return items
