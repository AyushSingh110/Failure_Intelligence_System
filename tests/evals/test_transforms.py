"""Suite registry and deterministic input construction."""
from __future__ import annotations

import pytest

from evals import canonical, datasets, transforms
from evals.datasets import Dataset, DatasetIntegrityError
from evals.transforms import Fixtures, RegistryError
from _helpers import REPO_ROOT

FILLER = "alpha beta gamma. "          # 3 words, trailing space kept


def _ds(prompts, ds_id="d", label="attack") -> Dataset:
    rows = [{"prompt": p, "label": label} for p in prompts]
    return Dataset(id=ds_id, spec={"label": label}, rows=rows)


def _fx() -> Fixtures:
    return Fixtures(filler=FILLER, filler_words=3,
                    templates={"legal": "For a review, answer this: {p}", "dev": "{p} (unit test)"})


# ── variants: exact bytes ────────────────────────────────────────────────────

def test_base_variant_is_the_prompt_unchanged():
    assert transforms.apply_variant("  keep  spacing \n", "base", _fx()) == "  keep  spacing \n"


def test_padding_variants_produce_exact_text():
    fx = _fx()
    assert transforms.apply_variant("ATTACK", "pad_after:3", fx) == "ATTACK\n\nalpha beta gamma. "
    assert transforms.apply_variant("ATTACK", "pad_after:6", fx) == "ATTACK\n\n" + FILLER * 2
    assert transforms.apply_variant("ATTACK", "pad_before:6", fx) == FILLER * 2 + "\n\nATTACK"


def test_frame_variant_inserts_the_prompt_literally():
    fx = _fx()
    assert transforms.apply_variant("X {p} {0} %s", "frame:legal", fx) == "For a review, answer this: X {p} {0} %s"
    assert transforms.apply_variant("X", "frame:dev", fx) == "X (unit test)"


@pytest.mark.parametrize("variant", ["pad_after:4", "pad_after:0", "pad_sideways:3", "frame:", "frame:nope", "what"])
def test_bad_variants_are_rejected(variant):
    with pytest.raises(RegistryError):
        transforms.apply_variant("p", variant, _fx())


# ── selection is by position ─────────────────────────────────────────────────

def test_first_unique_selects_by_position_and_skips_duplicates():
    ds = _ds(["a", "b", "a", "c", "b", "d"])
    picked = transforms.select_rows(ds, {"first_unique": 3})
    assert [(i, r["prompt"]) for i, r in picked] == [(0, "a"), (1, "b"), (3, "c")]
    assert [i for i, _ in transforms.select_rows(ds, {"first": 2})] == [0, 1]
    assert len(transforms.select_rows(ds, None)) == 6
    with pytest.raises(RegistryError):
        transforms.select_rows(ds, {"random": 3})


# ── ordering and provenance ──────────────────────────────────────────────────

def test_items_are_ordered_part_then_variant_then_row():
    suite = {"id": "s", "parts": [
        {"dataset": "atk", "variants": ["base", "pad_after:3"]},
        {"dataset": "ben", "variants": ["base"], "tag": "stub", "runtime": {"translation": "fixed_benign"}},
    ]}
    data = {"atk": _ds(["a1", "a2"], "atk"), "ben": _ds(["b1"], "ben", "benign")}
    items = transforms.build_items(suite, data, _fx())
    assert [i["idx"] for i in items] == [0, 1, 2, 3, 4]
    assert [(i["dataset"], i["variant"], i["source_idx"]) for i in items] == [
        ("atk", "base", 0), ("atk", "base", 1), ("atk", "pad_after:3", 0), ("atk", "pad_after:3", 1),
        ("ben", "base+stub", 0)]
    assert [i["expected"] for i in items] == ["attack"] * 4 + ["benign"]
    assert items[2]["text"] == "a1\n\n" + FILLER
    assert items[2]["input_sha256"] == canonical.sha256_bytes(items[2]["text"].encode("utf-8"))
    assert items[4]["runtime"] == {"translation": "fixed_benign"} and items[0]["runtime"] == {}


def test_building_twice_gives_identical_items():
    suite = {"id": "s", "parts": [{"dataset": "d", "variants": ["base", "frame:legal", "pad_before:3"]}]}
    data = {"d": _ds(["one", "two", "three"])}
    assert transforms.build_items(suite, data, _fx()) == transforms.build_items(suite, data, _fx())


def test_limit_keeps_the_first_rows_of_each_part_and_variant():
    suite = {"id": "s", "parts": [{"dataset": "d", "variants": ["base", "pad_after:3"]}]}
    items = transforms.build_items(suite, {"d": _ds(["a", "b", "c", "d"])}, _fx(), limit=2)
    assert [(i["variant"], i["source_idx"]) for i in items] == [
        ("base", 0), ("base", 1), ("pad_after:3", 0), ("pad_after:3", 1)]


# ── the real registries ──────────────────────────────────────────────────────

def test_real_suite_registry_is_valid_and_complete():
    reg = transforms.load_suites(REPO_ROOT)
    ids = [s["id"] for s in reg["suites"]]
    assert ids == ["std.xstest", "std.orbench_hard", "std.jailbreakbench", "std.harmbench",
                   "std.strongreject", "std.sorrybench", "std.advbench", "risk.long_input",
                   "risk.framing", "pilot.script", "lite.std", "latency", "stability"]
    assert ids[-1] == "stability", "the unseeded suite must run last in its worker"
    kinds = {s["id"]: s["kind"] for s in reg["suites"]}
    assert kinds["latency"] == "latency" and kinds["stability"] == "stability"
    assert reg["labels"]["risk"] == "constructed risk suite / diagnostic probe"
    assert reg["labels"]["pilot"] == "PILOT"
    assert reg["headline"]["attack"] == ["jailbreakbench", "harmbench", "strongreject", "sorrybench"]
    assert "advbench" not in reg["headline"]["attack"], "AdvBench must never be in the headline"
    assert reg["headline"]["case_study"] == ["advbench"]


def test_real_suites_expand_to_the_planned_sizes():
    reg = transforms.load_suites(REPO_ROOT)
    fx = transforms.load_fixtures(REPO_ROOT, reg)
    dreg = datasets.load_registry(REPO_ROOT)
    sizes = {}
    for suite in reg["suites"]:
        if suite["kind"] != "scan":
            continue
        data = {d: datasets.load_dataset(d, REPO_ROOT, dreg) for d in transforms.suite_datasets(suite)}
        sizes[suite["id"]] = len(transforms.build_items(suite, data, fx))
    assert sizes == {"std.xstest": 448, "std.orbench_hard": 250, "std.jailbreakbench": 134,
                     "std.harmbench": 387, "std.strongreject": 242, "std.sorrybench": 387,
                     "std.advbench": 168, "risk.long_input": 1820, "risk.framing": 4052,
                     "pilot.script": 144, "lite.std": 1848}


def test_long_input_sample_is_the_first_120_unique_harmbench_rows():
    reg = transforms.load_suites(REPO_ROOT)
    suite = next(s for s in reg["suites"] if s["id"] == "risk.long_input")
    harm = datasets.load_dataset("harmbench", REPO_ROOT)
    picked = transforms.select_rows(harm, suite["parts"][0]["select"])
    assert len(picked) == 120 and len({r["prompt"] for _, r in picked}) == 120
    assert [i for i, _ in picked] == sorted(i for i, _ in picked)
    assert picked[0][0] == 0


def test_fixtures_are_pinned_by_content_and_survive_crlf(tmp_path):
    reg = transforms.load_suites(REPO_ROOT)
    fx = transforms.load_fixtures(REPO_ROOT, reg)
    assert len(fx.filler.split()) == 21 and fx.filler.endswith(" ")
    assert set(fx.templates) == {"neutral", "medical", "developer", "legal"}
    assert all(t.count("{p}") == 1 for t in fx.templates.values())
    # A CRLF copy of the filler verifies; an edited one does not.
    (tmp_path / "f").mkdir()
    spec = {"fixtures": {"padding_filler": dict(reg["fixtures"]["padding_filler"], path="f/filler.txt")}}
    (tmp_path / "f" / "filler.txt").write_bytes((fx.filler + "\r\n").encode("utf-8"))
    assert transforms.load_fixtures(tmp_path, spec).filler == fx.filler
    (tmp_path / "f" / "filler.txt").write_bytes((fx.filler + "extra\n").encode("utf-8"))
    with pytest.raises(DatasetIntegrityError, match="not the pinned filler"):
        transforms.load_fixtures(tmp_path, spec)


def test_script_pilot_fixture_has_72_prompts_in_12_groups_of_6():
    ds = datasets.load_dataset("script_pilot_v0", REPO_ROOT)
    groups = {}
    for row in ds.rows:
        groups.setdefault(row["group"], []).append(row["prompt"])
    assert ds.n == 72 and len(groups) == 12 and all(len(v) == 6 for v in groups.values())
    assert ds.label == "benign" and ds.spec["kind"] == "fixture"


def test_profiles_registry_names_the_model_explicitly_and_describes_the_profile_correctly():
    reg = transforms.load_profiles(REPO_ROOT)
    canon = reg["profiles"][reg["canonical_profile"]]
    assert reg["canonical_profile"] == "sdk-offline-failsecure"
    assert canon["pair_version"] == "v6_3b" and "default" not in str(canon["pair_version"])
    assert canon["title"] == "canonical reproducibility/evaluation profile"
    text = canonical.dumps(reg).lower()
    assert "not the production runtime environment" in text
    assert "exactly the production runtime" not in text
    assert [d["id"] for d in reg["deviations"]] == [f"V{i}" for i in range(1, 11)]
    assert set(reg["behaviours_to_keep_apart"]) == {
        "A_shipped_default", "B_canonical_reproducibility_profile", "C_lite_profile", "D_stability_unseeded"}
    lite = reg["profiles"]["lite-simulated"]
    assert lite["models"] == {} and {"sklearn", "joblib", "onnxruntime", "numpy"} <= set(lite["blocked_imports"])
    ref = reg["profiles"]["reference-v6.2"]
    assert ref["pair_version"] == "v6" and ref["expect"]["pair_declared_version"] == "v6.2"
    assert canon["expected_counts"] == {"xstest_safe": 132, "orbench_hard": 226, "xstest_unsafe": 177,
                                        "jailbreakbench": 130, "harmbench": 326, "strongreject": 218,
                                        "sorrybench": 316, "advbench": 163}
    assert ref["expected_counts"] == {"xstest_safe": 134, "orbench_hard": 226, "xstest_unsafe": 176,
                                      "jailbreakbench": 129, "harmbench": 317, "strongreject": 217,
                                      "sorrybench": 291, "advbench": 160}


def test_malformed_suite_registries_are_rejected(tmp_path):
    reg_dir = tmp_path / "evals" / "registry"
    reg_dir.mkdir(parents=True)

    def write(suites):
        canonical.write_json(reg_dir / "suites.json", {"suites": suites})

    good = {"id": "a", "group": "standard", "kind": "scan", "profile": "@primary", "title": "t",
            "parts": [{"dataset": "d", "variants": ["base"]}]}
    write([good])
    assert transforms.load_suites(tmp_path)["suites"][0]["id"] == "a"
    write([good, good])
    with pytest.raises(RegistryError, match="duplicate"):
        transforms.load_suites(tmp_path)
    write([dict(good, kind="exec")])
    with pytest.raises(RegistryError, match="unknown kind"):
        transforms.load_suites(tmp_path)
    write([dict(good, parts=[{"dataset": "d", "variants": ["__import__('os')"]}])])
    with pytest.raises(RegistryError, match="unknown variant"):
        transforms.load_suites(tmp_path)
    write([{k: v for k, v in good.items() if k != "profile"}])
    with pytest.raises(RegistryError, match="has no 'profile'"):
        transforms.load_suites(tmp_path)
