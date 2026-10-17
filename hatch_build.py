"""
Build hook: package exactly the approved model artifacts, verified, and nothing else from fie/models/.

The tracked inventory, scripts/model_manifest.json, decides what ships. Every entry marked
`in_wheel: true` must be present in the source tree with its listed size and SHA-256, or the
build fails (WP-021 D-048). Those files are included explicitly, into the sdist and the wheel
alike (D-049), together with fie/models/MANIFEST.json derived from the inventory at build time
(D-050). Nothing else under fie/models/ is packed: the pyproject.toml build section excludes the
directory, and this hook re-includes only the approved files (D-051).

The verification reuses fie/_model_integrity.py, loaded by file path so that neither the `fie`
package nor any machine-learning dependency is imported, and no model is deserialized. Files are
read as bytes, hashed and checked; that is all.

The decisions are recorded in docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md,
section 14.9; the measurements behind the design are in the Stage 3.3a execution report.
"""
from __future__ import annotations

import atexit
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:  # only when this file is imported for its plain functions, without hatchling
    BuildHookInterface = object

INVENTORY = "scripts/model_manifest.json"
INTEGRITY_MODULE = "fie/_model_integrity.py"
MODELS_DIR = "fie/models"


class ModelArtifactError(Exception):
    """The approved model artifacts cannot be packaged. The message names manifest-relative paths only."""


def load_integrity(root: Path):
    """fie/_model_integrity.py from the tree being built, loaded by path. Standard library only."""
    name = "_fie_build_model_integrity"
    path = root / INTEGRITY_MODULE
    if not path.is_file():
        raise ModelArtifactError(f"{INTEGRITY_MODULE} is missing from the source tree; it is needed to verify models")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def plan(root) -> dict:
    """
    What to package, for the tree at `root`.

    Returns {"include": {absolute source path: archive path}, "manifest": bytes, "unlisted": [...],
    "off_wheel_present": [...]}. Raises ModelArtifactError if the inventory is invalid or any
    `in_wheel: true` artifact is missing, unreadable or altered. Deserializes nothing.
    """
    root = Path(root)
    mi = load_integrity(root)
    inventory_path = root / INVENTORY
    try:
        source = inventory_path.read_bytes()
    except OSError as exc:
        raise ModelArtifactError(f"{INVENTORY} cannot be read ({type(exc).__name__})") from None
    try:
        manifest = mi.validate_production(source)
        derived = mi.derive_package_manifest(source)
    except mi.ModelIntegrityError as exc:
        raise ModelArtifactError(f"{INVENTORY} is not a valid production inventory: {exc}") from None

    include, problems = {}, []
    approved = [e for e in manifest.entries if e.in_wheel]
    for entry in approved:
        try:
            mi.read_verified(root, entry)            # size and SHA-256; the bytes are not used further
        except mi.ModelIntegrityError as exc:
            problems.append(f"{entry.path}: {exc.code}")
            continue
        include[str(root.joinpath(*PurePosixPath(entry.path).parts))] = entry.path
    if problems:
        raise ModelArtifactError(
            f"{len(problems)} of {len(approved)} approved model artifacts cannot be packaged "
            f"({'; '.join(problems)}). Fetch and verify them with `python scripts/download_models.py`. "
            f"The build does not continue without them.")

    listed = {e.path for e in manifest.entries}
    expected_extras = {f"{MODELS_DIR}/{mi.PACKAGE_MANIFEST_NAME}", f"{MODELS_DIR}/.gitkeep"}
    present = sorted(p.relative_to(root).as_posix() for p in (root / MODELS_DIR).rglob("*") if p.is_file()) \
        if (root / MODELS_DIR).is_dir() else []
    return {
        "include": include,
        "manifest": derived,
        "unlisted": [p for p in present if p not in listed and p not in expected_extras],
        "off_wheel_present": [p for p in present if p in listed and p not in set(include.values())],
    }


class ModelAllowlistHook(BuildHookInterface):
    """Hatchling build hook (`[tool.hatch.build.hooks.custom]`), for the sdist and wheel targets."""

    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        try:
            result = plan(self.root)
        except ModelArtifactError as exc:
            raise ModelArtifactError(f"model packaging refused: {exc}") from None
        if result["unlisted"]:
            self.app.display_warning(
                f"fie model allowlist: {len(result['unlisted'])} file(s) under {MODELS_DIR}/ are not in "
                f"{INVENTORY} and are NOT packaged: {', '.join(result['unlisted'])}")
        if result["off_wheel_present"]:
            self.app.display_info(
                f"fie model allowlist: inventory entries marked in_wheel: false are present and NOT packaged: "
                f"{', '.join(result['off_wheel_present'])}")
        if version == "editable":
            # An editable install runs from the source tree, where the loader reads these files and
            # the generated manifest in place. Verified above; nothing to copy.
            return

        self._tmp = tempfile.mkdtemp(prefix="fie-model-manifest-")
        atexit.register(shutil.rmtree, self._tmp, True)
        generated = os.path.join(self._tmp, "MANIFEST.json")
        with open(generated, "wb") as f:
            f.write(result["manifest"])
        build_data["force_include"].update(result["include"])
        build_data["force_include"][generated] = f"{MODELS_DIR}/MANIFEST.json"
        self.app.display_info(f"fie model allowlist: {len(result['include'])} approved model artifacts verified "
                              f"and included, with {MODELS_DIR}/MANIFEST.json derived from {INVENTORY}")

    def finalize(self, version, build_data, artifact_path):
        tmp = getattr(self, "_tmp", None)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
            self._tmp = None
