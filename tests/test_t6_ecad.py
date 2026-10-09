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
    assert r["lower_bound"] is False


def test_search_walls_come_from_the_task_and_name_the_wall_they_hit(tmp_path):
    """[verifier] timeout_sec / node_budget in task.toml are the phi search's
    two walls (the ECAD repo's declaration). A search stopped by one returns
    a lower bound with exact_search = false and search_limit naming the
    wall, so the reader knows whether more time would move the number."""
    case = CASES[0]
    task = {"verifier": {"timeout_sec": 3600.0, "node_budget": 1}}
    # A submission that leaves the anchors unusable AND cannot be proven
    # optimal at once forces a search: every designator renamed, and every
    # terminal moved to the next incidence's net. (The matcher since the ECAD
    # repo's change 57 proves a renamed-but-correct graph from its bound before any
    # node is spent, so connectivity kept no longer reaches the wall.)
    g = json.loads((case / "gt/gt_graph.json").read_text())
    ren = {c["id"]: f"X{i}" for i, c in enumerate(g["components"])}
    for c in g["components"]:
        c["terminals"] = [t.replace(c["id"] + ".", ren[c["id"]] + ".", 1) for t in c["terminals"]]
        c["id"] = ren[c["id"]]
    inc = [[next(ren[k] + t[len(k):] for k in ren if t.startswith(k + ".")), n] for t, n in g["incidences"]]
    g["incidences"] = [[t, n] for (t, _), (_, n) in zip(inc, inc[1:] + inc[:1])]
    sub = tmp_path / "pred_graph.json"; sub.write_text(json.dumps(g))
    # Since the ECAD repo's change 58 the search has a second stage: when the
    # branch-and-bound stops at a wall, the MILP (HiGHS) takes the rest of the
    # deadline and closes a board this size in well under a second -- so the
    # walls are only ever reported when the MILP stage is unavailable too.
    r = score(case, sub, task)
    assert r["exact_search"] is True and r["search_limit"] is None, r
    with _bnb_only():
        r = score(case, sub, task)
    assert r["exact_search"] is False and r["search_limit"] == "node_budget", r
    assert "node_budget" in r["note"]


class _bnb_only:
    """The matcher with its MILP stage switched off, so the branch-and-bound's
    walls are observable (the ECAD repo's change 58 folds a MILP in after it)."""
    def __enter__(self):
        from envs.common.ecad_graph import name_aware
        self.na, self.saved = name_aware, name_aware._milp
        name_aware._milp = None
    def __exit__(self, *exc):
        self.na._milp = self.saved


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
    # C1 read as 1 uF instead of 100 nF. 0.666667 until ecad change 100 (the value
    # term cost the whole component, S_C 2/3); values are not read since.
    ("case1", "C1_value_x10"): 1.0,
}


def _mutate(g, how):
    g = json.loads(json.dumps(g))
    if how == "missing_C1":
        g["components"] = [c for c in g["components"] if c["id"] != "C1"]
        g["incidences"] = [i for i in g["incidences"] if not i[0].startswith("C1.")]
    elif how == "U1.1_to_VCC":
        g["incidences"] = [["U1.1", "VCC"] if i == ["U1.1", "N1"] else i for i in g["incidences"]]
    elif how == "C1_value_x10":
        for c in g["components"]:
            if c["id"] == "C1":
                c["value"] = 1e-6
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


def test_the_greedy_incumbent_answers_to_the_deadline(tmp_path):
    """The greedy pass that builds the search's first incumbent is itself
    |free_gt| x |cand| inner solves, and on 2026-09-19 a submission with two
    and a half times the reference's nets spent longer than the scorer's
    subprocess limit in there -- before the first deadline check -- so the
    record could never be judged. Since the ECAD repo's change 58 that pass is a
    lifted alternation (one assignment + one inner solve per round) and the
    recursion checks the clock every eight nodes, so with timeout_sec = 0 the
    call must come back at once -- either exact, when the board closes before
    the first clock check, or as a bound that names the deadline."""
    case = CASES[0]
    g = json.loads((case / "gt/gt_graph.json").read_text())
    ren = {c["id"]: f"X{i}" for i, c in enumerate(g["components"])}
    for c in g["components"]:
        c["terminals"] = [t.replace(c["id"] + ".", ren[c["id"]] + ".", 1) for t in c["terminals"]]
        c["id"] = ren[c["id"]]
    inc = [[next(ren[k] + t[len(k):] for k in ren if t.startswith(k + ".")), n] for t, n in g["incidences"]]
    g["incidences"] = [[t, n] for (t, _), (_, n) in zip(inc, inc[1:] + inc[:1])]
    sub = tmp_path / "pred_graph.json"; sub.write_text(json.dumps(g))
    import time
    t0 = time.monotonic()
    with _bnb_only():
        r = score(case, sub, {"verifier": {"timeout_sec": 0.0, "node_budget": 200_000}})
    assert time.monotonic() - t0 < 30.0
    if r["exact_search"]:
        assert r["search_limit"] is None and r["lower_bound"] is False, r
    else:
        assert r["search_limit"] == "deadline" and r["timed_out"] is True and r["lower_bound"] is True, r
    assert 0.0 <= r["score"] <= 1.0


def _brute_force_optimum(w):
    """Best total weight over every injective row -> column map (small tables)."""
    import itertools
    n_r, n_c = len(w), len(w[0]) if w else 0
    best = 0
    for cols in itertools.permutations(range(n_c), min(n_r, n_c)):
        rows = range(n_r) if n_r <= n_c else itertools.combinations(range(n_r), n_c)
        for rs in ([list(rows)] if n_r <= n_c else rows):
            best = max(best, sum(w[i][j] for i, j in zip(rs, cols)))
    return best


def test_max_weight_matching_matches_the_brute_force_optimum():
    """The C assignment finds the optimum on rectangular tables with zeros
    (checked against exhaustive enumeration -- the pure-Python Hungarian it
    replaced left with the ECAD repo's change 58), and only ever pairs a row with a
    column of positive weight, once."""
    import random

    from envs.common.ecad_graph.matcher import max_weight_matching
    rng = random.Random(7)
    for _ in range(200):
        n_r, n_c = rng.randint(1, 6), rng.randint(1, 6)
        w = [[rng.choice([0, 0, 0, 1, 2, 3]) for _ in range(n_c)] for _ in range(n_r)]
        a = max_weight_matching(w)
        assert sum(w[i][j] for i, j in enumerate(a) if j >= 0) == _brute_force_optimum(w)
        assert all(j == -1 or w[i][j] > 0 for i, j in enumerate(a))
        cols = [j for j in a if j >= 0]
        assert len(cols) == len(set(cols))
    assert max_weight_matching([]) == [] and max_weight_matching([[]]) == [-1]


def test_every_record_names_its_code_and_its_solvers():
    """A T6 record carries ecad's own scorer_digest (six files, as ecad
    defines it), a digest over every vendored file (name_aware.py and milp.py,
    where the matcher changes land, included) with its version and file list,
    and the solvers the host could run (ecad change 90)."""
    from envs.verifiers.ecad import ECAD_SOURCE, ecad_vendored_digest
    case = CASES[0]
    r = score(case, case / "gt/gt_graph.json")
    vd = r["ecad_vendored_digest"]
    files = sorted(p.name for p in (REPO / "envs/common/ecad_graph").glob("*.py"))
    assert vd == ecad_vendored_digest() and vd["version"] == 1 and vd["files"] == files
    assert {"name_aware.py", "milp.py", "matcher.py"} <= set(vd["files"]) and len(vd["sha256"]) == 64
    assert r["scorer_version"] == ECAD_SOURCE and ECAD_SOURCE.startswith("BenchCAD-org/ecad 61e6018")
    assert set(r["solvers"]) == {"scipy", "milp", "hungarian_c"}
    assert r["solvers"]["milp"] is True and r["solvers"]["scipy"]


def test_the_vendored_digest_sees_a_matcher_change(tmp_path, monkeypatch):
    """The point of the second digest: an edit to name_aware.py (outside
    scorer_digest's six files) changes it."""
    import shutil
    from envs.verifiers import ecad
    from envs.common.ecad_graph import spatial_reference
    src = REPO / "envs/common/ecad_graph"
    fake = tmp_path / "envs/common/ecad_graph"
    shutil.copytree(src, fake, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "envs/verifiers").mkdir(parents=True)
    monkeypatch.setattr(ecad, "__file__", str(tmp_path / "envs/verifiers/ecad.py"))
    before = ecad.ecad_vendored_digest()["sha256"]
    with open(fake / "name_aware.py", "a") as f:
        f.write("\n# a change\n")
    assert ecad.ecad_vendored_digest()["sha256"] != before
    assert spatial_reference.scorer_digest() == spatial_reference.scorer_digest()   # ecad's own is untouched


# ── position mode (ecad f021ef4, re-stamped 61e6018), through ecad's verifier.py / score_t6 ──────
#
# A case is scored in position mode when gt/correspondence.json says so; the
# only real board whose position-mode grading files may live in git is the open
# dev sample examples/task6/cases/case1 (ecad's `pcb2schematic` board, the same
# bytes as bank case1). Its correspondence.json and spatial_reference.json, from
# ecad verbatim (the sidecar from f021ef4, the correspondence re-stamped at
# 61e6018 for the change 98/#100 scorer), are under tests/fixtures/t6_position/case1/gt/.

POS_FILES = REPO / "tests/fixtures/t6_position/case1/gt"
# ecad f021ef4 tasks/pcb2schematic-wheelleg-buck/grading/correspondence.json, verbatim
HELD = {"mode": "held",
        "reason": "Spatial reference unavailable: authoritative placed terminal geometry is incomplete. "
                  "Excluded from the official 20-case T6 positional bank; --mode legacy is historical/debug only."}


def _position_case(tmp_path, name="case1"):
    import shutil
    case = tmp_path / name
    shutil.copytree(REPO / "examples/task6/cases/case1", case)
    for f in ("correspondence.json", "spatial_reference.json"):
        shutil.copyfile(POS_FILES / f, case / "gt" / f)
    return case


def _write(tmp_path, obj, name="pred_graph.json"):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


def test_position_mode_scores_a_real_board_through_the_verifier(tmp_path):
    """The reference with its sidecar attached is a perfect 2.0-position answer
    and scores 1.0 through ecad's verifier; a 2.0 answer off by a hair in every
    centre scores the same (inside every gate); one dropped component lowers
    it. The record says which scorer and mode produced it."""
    from envs.verifiers.ecad import position_mode, position_reference
    case = _position_case(tmp_path)
    assert position_mode(case)
    ref = position_reference(case)
    assert ref["schema"] == "pcb2schematic/2.0-position" and ref["terminals"]
    r = score(case, _write(tmp_path, ref))
    assert r["status"] == "ok" and r["score"] == pytest.approx(1.0), r
    assert r["correspondence_mode"] == "position" and r["metric_version"] == "v2-position"
    assert r["scorer"] == "ecad verifier.py / score_t6" and r["lower_bound"] is False
    assert r["components_matched"] == r["components_gt"] == len(ref["components"])
    assert r["spatial_reference_sha256"] == json.loads((POS_FILES / "correspondence.json").read_text())[
        "spatial_reference_sha256"]
    assert "score_v1" not in r            # the position diagnostic is not an optimised graph IoU

    nudged = json.loads(json.dumps(ref))
    for c in nudged["components"]:
        c["center"] = [min(1.0, c["center"][0] + 0.001), c["center"][1]]
    assert score(case, _write(tmp_path, nudged, "n.json"))["score"] == pytest.approx(1.0)

    drop = json.loads(json.dumps(ref))
    gone = drop["components"].pop()
    tset = set(gone["terminals"])
    drop["terminals"] = [t for t in drop["terminals"] if t["id"] not in tset]
    drop["incidences"] = [i for i in drop["incidences"] if i[0] not in tset]
    r = score(case, _write(tmp_path, drop, "d.json"))
    assert r["status"] == "ok" and 0.0 < r["score"] < 1.0, r


def test_position_case_rejects_a_1_0_graph_with_a_reason(tmp_path):
    """On a position-mode case a 2.0-position submission scores, and a 1.0
    submission (no centres, no terminals) is rejected cleanly: status
    invalid_prediction and an error naming the schema it needed -- a 0 that
    says why, never a silent one, and never a legacy fallback."""
    from envs.verifiers.ecad import position_reference
    case = _position_case(tmp_path)
    ok = score(case, _write(tmp_path, position_reference(case), "ok.json"))
    assert ok["status"] == "ok" and ok["score"] == pytest.approx(1.0)
    r = score(case, case / "gt/gt_graph.json")          # the 1.0 reference itself
    assert r["status"] == "invalid_prediction" and r["score"] == 0.0, r
    assert "2.0-position" in r["error"] and "channels" not in r


def test_a_case_without_a_correspondence_keeps_the_legacy_path(tmp_path):
    """robot-mainboard-b (case13) and wheelleg-buck (case15) have no spatial
    reference and get no correspondence.json: they are scored exactly as
    before -- named correspondence, called directly -- and their prompt is
    the unchanged TASK.md. A position-mode case is asked for 2.0-position."""
    from envs.common.episode import _task_brief
    from envs.verifiers.ecad import position_mode
    case = CASES[0]
    assert not (case / "gt/correspondence.json").exists() and not position_mode(case)
    r = score(case, case / "gt/gt_graph.json")
    assert r["status"] == "ok" and r["correspondence_mode"] == "legacy" and r["score"] == pytest.approx(1.0)
    assert r["score_v1"] == pytest.approx(1.0)
    assert _task_brief(case) == (REPO / "envs/t6_pcb2schematic/TASK.md").read_text()
    assert "2.0-position" not in _task_brief(case)
    pos = _position_case(tmp_path)
    assert _task_brief(pos) == (REPO / "envs/t6_pcb2schematic/TASK_position.md").read_text()
    assert "pcb2schematic/2.0-position" in _task_brief(pos)


def test_a_held_board_is_a_status_not_a_number(tmp_path):
    """A board ecad declares `held` comes back status held with score None --
    never a number, never silently dropped (summarize lists it by name)."""
    case = _position_case(tmp_path)
    (case / "gt/spatial_reference.json").unlink()
    (case / "gt/correspondence.json").write_text(json.dumps(HELD))
    r = score(case, case / "gt/gt_graph.json")
    assert r["status"] == "held" and r["score"] is None and "Spatial reference unavailable" in r["held"], r
    from tools.summarize import t6_mean_note
    note = t6_mean_note([{"case": "case15", "status": "held", "score": None, "mode": "held"}])
    assert "held, no number: case15" in note


def test_an_incomplete_legacy_search_through_the_verifier_is_an_evaluator_error(tmp_path):
    """Through ecad's verifier an unfinished legacy search raises inside
    score_t6: the record is status evaluator_error with score None -- not 0 and
    not a lower bound passed off as a score."""
    case = CASES[0]
    g = json.loads((case / "gt/gt_graph.json").read_text())
    ren = {c["id"]: f"X{i}" for i, c in enumerate(g["components"])}
    for c in g["components"]:
        c["terminals"] = [t.replace(c["id"] + ".", ren[c["id"]] + ".", 1) for t in c["terminals"]]
        c["id"] = ren[c["id"]]
    inc = [[next(ren[k] + t[len(k):] for k in ren if t.startswith(k + ".")), n] for t, n in g["incidences"]]
    g["incidences"] = [[t, n] for (t, _), (_, n) in zip(inc, inc[1:] + inc[:1])]
    sub = _write(tmp_path, g)
    task = {"verifier": {"timeout_sec": 3600.0, "node_budget": 1}}
    with _bnb_only():
        r = score(case, sub, task, mode="legacy")
    assert r["status"] == "evaluator_error" and r["score"] is None, r
    assert "legacy search incomplete: node_budget" in r["error"]
    # the same submission on the default (no correspondence) path is still a bound, as before
    with _bnb_only():
        r = score(case, sub, task)
    assert r["status"] == "ok" and r["lower_bound"] is True and r["search_limit"] == "node_budget"


def test_a_tampered_position_case_is_an_evaluator_error(tmp_path):
    """The activation checks ecad's verifier makes are made here: a view_top.png
    that is not the one the sidecar was measured on, or a scorer that is not
    the one the correspondence names, is the evaluator's failure -- no number."""
    from envs.verifiers.ecad import position_reference
    case = _position_case(tmp_path)
    sub = _write(tmp_path, position_reference(case))
    (case / "input/views/view_top.png").write_bytes(b"not the render")
    r = score(case, sub)
    assert r["status"] == "evaluator_error" and r["score"] is None and "render digest" in r["error"], r
    case = _position_case(tmp_path, "case1b")
    c = json.loads((case / "gt/correspondence.json").read_text())
    c["scorer_sha256"] = "0" * 64
    (case / "gt/correspondence.json").write_text(json.dumps(c))
    r = score(case, sub)
    assert r["status"] == "evaluator_error" and r["score"] is None and "scorer implementation" in r["error"], r


def test_the_vendored_scorer_is_the_one_every_correspondence_names():
    """scorer_digest() over the vendored files is the scorer_sha256 ecad
    61e6018 wrote into every position-mode correspondence.json."""
    from envs.common.ecad_graph.spatial_reference import scorer_digest
    assert json.loads((POS_FILES / "correspondence.json").read_text())["scorer_sha256"] == scorer_digest()


# Position-mode tiers on the real dev board (n_1 is the board's V+, n_2 its GND).
# (mutation) -> (score, fatal rail faults). Pinned at ecad 61e6018 (change 98, change 100);
# under f021ef4 the same graphs scored 0.714286 / 0.5 / 0.5 / 0.166667, because
# position coverage was 0 (no short or open judged, V+ tied to GND survived at
# 0.5) and a wrong value cost the whole component.
POSITION_PINNED = {
    "split_V+": (0.357143, []),              # S_N 5/7 x P_open (1 - 3.0 x 1/6)
    "shatter_V+": (0.0, ["open"]),           # 4 of 5 V+ terminals outside its largest piece: fatal open
    "V+_absorbs_GND": (0.0, ["short"]),      # every GND terminal on V+: fatal short
    "values_x10": (1.0, []),                 # values are not read
}


def _position_mutate(g, how):
    g = json.loads(json.dumps(g))
    inc = g["incidences"]
    if how == "split_V+":
        moved = set([t for t, n in inc if n == "n_1"][:2])
        g["nets"].append({"id": "n_1b"})
        g["incidences"] = [[t, "n_1b"] if t in moved else [t, n] for t, n in inc]
    elif how == "shatter_V+":
        out = []
        for k, (t, n) in enumerate(inc):
            if n == "n_1":
                g["nets"].append({"id": f"n_1_{k}"})
                n = f"n_1_{k}"
            out.append([t, n])
        g["incidences"] = out
    elif how == "V+_absorbs_GND":
        g["incidences"] = [[t, "n_1" if n == "n_2" else n] for t, n in inc]
    elif how == "values_x10":
        for c in g["components"]:
            if c.get("value"):
                c["value"] *= 10
    return g


@pytest.mark.parametrize("how", sorted(POSITION_PINNED))
def test_position_pinned_tiers(how, tmp_path):
    from envs.verifiers.ecad import position_reference
    case = _position_case(tmp_path)
    r = score(case, _write(tmp_path, _position_mutate(position_reference(case), how)))
    want, faults = POSITION_PINNED[how]
    assert r["status"] == "ok" and r["score"] == pytest.approx(want, abs=1e-6), r
    assert [f["fault"] for f in r["fatal_rail_faults"]] == faults, r["fatal_rail_faults"]
    assert r["fatal_power_shorts"] == [f for f in r["fatal_rail_faults"] if f["fault"] == "short"]
    assert r["short_open_evidence"]["coverage"] == pytest.approx(1.0)   # identity from the spatial match


STAGE = REPO / "bank_staging/t6_position"


@pytest.mark.skipif(not (STAGE / "manifest.json").exists(),
                    reason="no bank_staging/t6_position (tools/stage_t6_position.py; gitignored bank data)")
def test_every_staged_board_scores_its_own_reference(tmp_path):
    """The staged grading files (gitignored: held-out ground truth), on a copy
    of each bank case: the position reference scores 1.0, the 1.0 graph is
    rejected with a reason, and the two held boards are not staged at all."""
    import shutil
    from envs.verifiers.ecad import position_reference
    m = json.loads((STAGE / "manifest.json").read_text())
    bank = Path.home() / "benchcad-cases"
    assert set(m["not_staged"]) == {"case13", "case15"} and not (STAGE / "case13").exists()
    for cid, e in sorted(m["staged"].items()):
        src = bank / e["env"] / "cases" / cid
        if not src.exists():
            pytest.skip(f"no bank case at {src}")
        case = tmp_path / cid
        (case / "gt").mkdir(parents=True)
        (case / "input/views").mkdir(parents=True)
        shutil.copyfile(src / "gt/gt_graph.json", case / "gt/gt_graph.json")
        shutil.copyfile(src / "input/views/view_top.png", case / "input/views/view_top.png")
        # every staged file, a replaced gt_graph.json included (ecad change 104: case16)
        assert set(e["files"]) <= {"gt/correspondence.json", "gt/spatial_reference.json", "gt/gt_graph.json"}
        for f in e["files"]:
            shutil.copyfile(STAGE / cid / f, case / f)
        r = score(case, _write(tmp_path, position_reference(case), f"{cid}.json"))
        assert r["status"] == "ok" and r["score"] == pytest.approx(1.0), (cid, r)
        r = score(case, case / "gt/gt_graph.json")
        assert r["status"] == "invalid_prediction" and r["score"] == 0.0, (cid, r)


def test_every_example_in_the_position_task_text_is_a_valid_2_0_graph():
    """A model copies the example. TASK_position.md's examples (its own and
    ecad's) are all pcb2schematic/2.0-position and pass the schema; TASK.md
    keeps its 1.0 example for the legacy cases."""
    import re
    from envs.common.ecad_graph import validate
    text = (REPO / "envs/t6_pcb2schematic/TASK_position.md").read_text()
    blocks = re.findall(r"```(python|json)\n(.*?)```", text, re.S)
    assert len(blocks) == 2, blocks
    for lang, body in blocks:
        if lang == "python":
            ns = {}
            exec(body, ns)
            obj = ns["result"]
        else:
            obj = json.loads(body)
        assert obj["schema"] == "pcb2schematic/2.0-position"
        validate(obj)
    assert '"schema": "pcb2schematic/1.0"' in (REPO / "envs/t6_pcb2schematic/TASK.md").read_text()


def test_the_t6_mean_note_counts_boards_from_the_records():
    """t6_mean_note's board counts come from the records summarised (distinct
    cases), not from a constant; no other bank's means are quoted."""
    from tools.summarize import t6_mean_note
    rows = ([{"case": c, "mode": "position", "status": "ok", "score": 0.5} for c in ("case03", "case04", "case05")]
            + [{"case": "case03", "mode": "position", "status": "ok", "score": 0.7},     # a second run
               {"case": "case13", "mode": "legacy", "status": "ok", "score": 0.1},
               {"case": "case15", "mode": "held", "status": "held", "score": None}])
    note = t6_mean_note(rows)
    assert "3 of the 5 T6 boards here scored in position mode" in note and "over 4 records" in note
    assert "legacy-path records" in note and "case13" in note and "held, no number: case15" in note
    assert "0.6149" not in note and "18 of" not in note


def test_the_position_brief_spatial_section_is_ecads_file_verbatim():
    """The spatial section has one home in ecad, lib/t6_spatial_identity.md
    (change 99: sixty copies drifted once), vendored here as
    envs/common/t6_spatial_identity.md. TASK_position.md's section must be that
    file, compared the way ecad's check_all compares its own copies: from the
    heading to the end, stripped."""
    heading = "## Spatial identity (T6 position mode)"
    want = (REPO / "envs/common/t6_spatial_identity.md").read_text(encoding="utf-8").strip()
    text = (REPO / "envs/t6_pcb2schematic/TASK_position.md").read_text(encoding="utf-8")
    assert want.startswith(heading) and text.count(heading) == 1
    assert (heading + text.split(heading, 1)[1]).strip() == want


def test_the_scorer_digest_covers_every_vendored_module():
    """scorer_digest() hashes an explicit list (ecad change 110), so a module added
    to the vendored tree would not move it: the list must be the tree."""
    from envs.common.ecad_graph import spatial_reference as sr
    root = Path(sr.__file__).parent
    assert sorted(p.name for p in root.glob("*.py")) == sorted(sr.SCORER_FILES)
    assert sr.scorer_digest() == "f4dd141038e763497de4cb31a714b4de0cc5a96cac01cab20df5733f9992e3fa"
