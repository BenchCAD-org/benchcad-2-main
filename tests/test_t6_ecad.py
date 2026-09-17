"""T6 smoke test on the pcb2schematic fixture under tests/fixtures/t6: the
reference graph submitted as the answer scores 1.0, the empty template scores 0,
and three mutations pin the metric's middle tiers. Real boards are data and
stay out of the repo (envs/t6_pcb2schematic/cases/, gitignored)."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envs.verifiers.ecad import score  # noqa: E402

CASES = sorted(p.parent.parent for p in (REPO / "tests/fixtures/t6").glob("*/gt/gt_graph.json"))
EMPTY = {"schema": "pcb2schematic/1.0", "components": [], "nets": [], "incidences": []}


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_reference_scores_one(case):
    r = score(case, case / "gt/gt_graph.json")
    assert r["score"] == pytest.approx(1.0), r
    assert all(v == pytest.approx(1.0) for v in r["channels"].values()), r["channels"]
    # A finished search says so: exact, and no wall named.
    assert r["exact_search"] is True and r["search_limit"] is None, r


def test_search_walls_come_from_the_task_and_name_the_wall_they_hit(tmp_path):
    """[verifier] timeout_sec / node_budget in task.toml are the phi search's
    two walls (the ECAD repo's declaration). A search stopped by one returns
    a lower bound with exact_search = false and search_limit naming the
    wall, so the reader knows whether more time would move the number."""
    case = CASES[0]
    task = {"verifier": {"timeout_sec": 3600.0, "node_budget": 1}}
    # A submission that leaves the anchors unusable forces a search: every
    # designator renamed, connectivity kept.
    g = json.loads((case / "gt/gt_graph.json").read_text())
    ren = {c["id"]: f"X{i}" for i, c in enumerate(g["components"])}
    for c in g["components"]:
        c["terminals"] = [t.replace(c["id"] + ".", ren[c["id"]] + ".", 1) for t in c["terminals"]]
        c["id"] = ren[c["id"]]
    g["incidences"] = [[next(ren[k] + t[len(k):] for k in ren if t.startswith(k + ".")), n] for t, n in g["incidences"]]
    sub = tmp_path / "pred_graph.json"; sub.write_text(json.dumps(g))
    r = score(case, sub, task)
    assert r["exact_search"] is False and r["search_limit"] == "node_budget", r
    assert "node_budget" in r["note"]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_empty_scores_zero(case, tmp_path):
    p = tmp_path / "pred_graph.json"
    p.write_text(json.dumps(EMPTY))
    assert score(case, p)["score"] == 0.0


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_half_graph_scores_between(case, tmp_path):
    """Drop every other incidence: the score must land strictly inside (0, 1),
    which is what separates a metric from a pass/fail gate."""
    g = json.loads((case / "gt/gt_graph.json").read_text())
    g["incidences"] = g["incidences"][::2]
    p = tmp_path / "pred_graph.json"
    p.write_text(json.dumps(g))
    s = score(case, p)["score"]
    assert 0.0 < s < 1.0, s


PINNED = {
    # (case, mutation) -> score. These move when the metric moves; 1.0 and 0.0 survive
    # a lot of damage to a ruler, the middle tier is what actually pins it.
    ("case1", "missing_C1"): 0.47619,        # one component dropped: S_C 2/3 x S_N 5/7
    ("case1", "U1.1_to_VCC"): 0.0,           # a signal pin shorted to a rail: P_short -> 0
}


def _mutate(g, how):
    g = json.loads(json.dumps(g))
    if how == "missing_C1":
        g["components"] = [c for c in g["components"] if c["id"] != "C1"]
        g["incidences"] = [i for i in g["incidences"] if not i[0].startswith("C1.")]
    elif how == "U1.1_to_VCC":
        g["incidences"] = [["U1.1", "VCC"] if i == ["U1.1", "N1"] else i for i in g["incidences"]]
    elif how == "half_incidences":
        g["incidences"] = g["incidences"][::2]
    return g


@pytest.mark.parametrize("key", sorted(PINNED), ids=["/".join(k) for k in sorted(PINNED)])
def test_pinned_tiers(key, tmp_path):
    case, how = key
    g = _mutate(json.loads((REPO / "tests/fixtures/t6" / case / "gt/gt_graph.json").read_text()), how)
    p = tmp_path / "pred_graph.json"
    p.write_text(json.dumps(g))
    assert score(REPO / "tests/fixtures/t6" / case, p)["score"] == pytest.approx(PINNED[key], abs=1e-5)


def test_garbage_is_zero_not_exception(tmp_path):
    p = tmp_path / "pred_graph.json"
    p.write_text("{not json")
    r = score(CASES[0], p)
    assert r["score"] == 0.0 and "error" in r
