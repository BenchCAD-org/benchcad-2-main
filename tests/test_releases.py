"""releases/: one JSON per release (releases/README.md). The one for this
checkout's harness version states what this checkout runs, and the 2.0 one
states what the published reference lines ran on."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.common.ecad_graph.spatial_reference import scorer_digest  # noqa: E402
from harness import run as R                                         # noqa: E402

FIELDS = {"schema_version", "release", "status", "code", "harness_version", "datasets",
          "scorer_digest", "budgets", "reference_lines", "not_comparable_with"}


def releases() -> dict[str, dict]:
    return {p.name: json.loads(p.read_text()) for p in sorted((ROOT / "releases").glob("benchcad-*.json"))}


def test_every_release_has_the_fields_and_names_itself():
    rs = releases()
    assert set(rs) >= {"benchcad-2.0.json", "benchcad-2.1-beta.json"}
    for name, r in rs.items():
        assert set(r) == FIELDS, name
        assert name == f"benchcad-{r['release']}.json" and r["harness_version"] == r["release"]
        assert set(r["code"]) == {"repo", "tag", "commit"}
        # A release's JSON is written before its tag: a draft, with no commit yet.
        assert len(r["code"]["commit"] or "") == 40 or (r["status"] == "draft" and r["code"]["commit"] is None)
        assert set(r["budgets"]) == {"rounds", "output_tokens", "reasoning_carried", "compaction_tokens"}
        for ds in filter(None, r["datasets"].values()):
            assert set(ds) == {"repo", "revision", "commit", "dataset_version"} and len(ds["commit"]) == 40
        assert all(f"benchcad-{v}.json" in rs for v in r["not_comparable_with"])
        assert r["release"] in (ROOT / "releases/README.md").read_text()


def test_this_checkouts_release_matches_the_code():
    r = releases()[f"benchcad-{R.HARNESS_VERSION}.json"]
    assert r["budgets"] == {"rounds": R.DEFAULT_ROUNDS, "output_tokens": R.DEFAULT_TOKEN_BUDGET,
                            "reasoning_carried": True, "compaction_tokens": R.COMPACT_TOKENS}
    assert r["scorer_digest"] == scorer_digest()
    assert f"`v{R.HARNESS_VERSION}`" in (ROOT / "README.md").read_text()


def test_the_reference_lines_name_their_release():
    ref = json.loads((ROOT / "results/core_reference.json").read_text())
    r = releases()[f"benchcad-{ref['harness_version']}.json"]
    assert r["reference_lines"] == "results/core_reference.json"
    assert (r["code"]["tag"], r["budgets"]["rounds"]) == (ref["version"], ref["rounds"])
    assert r["scorer_digest"] == ref["scorer_digest"]
    assert (r["datasets"]["core"]["repo"].split("/")[1], r["datasets"]["core"]["dataset_version"]) == \
        (ref["case_set"], ref["dataset_version"])
