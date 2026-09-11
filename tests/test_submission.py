"""The fixed submission layout of the assembly tasks (envs/common/submission.py).

    equivalence     the new layout scores EXACTLY what the same geometry scores
                    as an old-style named cq.Assembly (to 1e-9 -- measured 0.0
                    -- on t2/t4/t5, for the reference AND for a wrong answer)
    rebuild         the scored assembly comes from parts x instances, so the
                    optional assembly/assembly.step changes nothing at all
    supplied parts  a re-export of the supplied part passes (bytes are not the
                    test), a rotated re-export passes (the check is pose-free),
                    a rebuilt part scores 0 for its TYPE with the reason -- and
                    the assembly score still uses what was submitted
    missing / extra a bom id with no part file, and a part file that is not a
                    bom id: each a named 0, never a crash
    transforms      not 4x4, not finite, not a rotation: the instance is dropped
                    with the reason and its type is charged, nothing raises
    unreadable      a part file that is not a STEP, and a submission that cannot
                    be rebuilt at all: 0 with the reason, never an exception
    old layout      a single STEP still scores the same, marked deprecated
    the tools       tools.use_part / export_part / submit_assembly write a
                    submission that parses and scores 1.0
    the episode     a complete submission directory wins over a leftover
                    final.step

Nothing under tests/fixtures/ is modified; every submission is built in tmp.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from envs.geom import ocp_hashcode_fix  # noqa: E402

ocp_hashcode_fix()
import cadquery as cq  # noqa: E402

from envs.common import submission as S  # noqa: E402
from envs.common.caseformat import resolve_part, solids, transform  # noqa: E402
from envs.common.episode import _artifact  # noqa: E402
from envs.common.score_case import fmt, score_case  # noqa: E402

FX = REPO / "tests/fixtures"
ASSEMBLY_TASKS = ["t2", "t4", "t5"]
# The keys both layouts must agree on: the headline, both its factors and every
# legacy column. A difference in any of them would mean the layout, not the
# answer, moved the number.
SAME = ("score", "asm_v1", "asm_v1_raw", "avg_part", "iou", "iou_raw", "hit",
        "rubric", "part_gen", "rubric_final")
EXACT = 1e-9


# ── building submissions ───────────────────────────────────────────────────
def _gt_instances(case: Path) -> list[dict]:
    return json.loads((case / "gt/instances.json").read_text())["instances"]


def _write(root: Path, parts: dict[str, Path], recs: list[dict],
           assembly_step=None) -> Path:
    """A submission directory: `parts` copied in under their ids, `recs` as
    instances.json (a bare list, as the layout specifies)."""
    sub = root / S.SUB_ROOT
    (sub / S.PARTS).mkdir(parents=True, exist_ok=True)
    (sub / S.ASSEMBLY).mkdir(parents=True, exist_ok=True)
    for pid, src in parts.items():
        shutil.copyfile(src, sub / S.PARTS / f"{pid}.step")
    (sub / S.ASSEMBLY / S.INSTANCES).write_text(json.dumps(recs, indent=1) + "\n")
    if assembly_step is not None:
        cq.exporters.export(cq.Workplane(obj=assembly_step), str(sub / S.ASSEMBLY / S.ASSEMBLY_STEP))
    return sub


def _oracle(case: Path, root: Path, **kw) -> Path:
    """The reference, submitted in the new layout: every part type's own file
    (caseformat.resolve_part) plus gt/instances.json's placements."""
    recs = _gt_instances(case)
    parts = {r["part_id"]: resolve_part(case, r["part_id"]) for r in recs}
    return _write(root, parts,
                  [{"part_id": r["part_id"], "instance_id": r["instance_id"], "transform": r["T"]}
                   for r in recs], **kw)


def _old_style(sub: Path, out: Path) -> Path:
    """The same submission as ONE STEP with children named `<part_id>_i<k>` --
    the layout every number so far was measured with. Written here from the
    part files and the transforms, without going through the module under
    test, so "the two layouts agree" is a measurement and not a tautology."""
    recs = json.loads((sub / S.ASSEMBLY / S.INSTANCES).read_text())
    a = cq.Assembly(name="asm")
    cache: dict[str, list] = {}
    for r in recs:
        pid = r["part_id"]
        if pid not in cache:
            cache[pid] = solids(sub / S.PARTS / f"{pid}.step")
        placed = [transform(s, r["transform"]) for s in cache[pid]]
        shape = placed[0] if len(placed) == 1 else cq.Compound.makeCompound(placed)
        a.add(cq.Workplane(obj=shape), name=r["instance_id"])
    a.save(str(out), "STEP")
    return out


def _reexport(src: Path, out: Path, rotate: float = 0.0) -> Path:
    """`src` imported and written out again -- what a program that reads a
    supplied part and re-exports it produces. Optionally turned in its own
    frame, which is legitimate: the part frame is what the instance transform
    is relative to."""
    shape = cq.importers.importStep(str(src)).val()
    if rotate:
        shape = shape.rotate((0, 0, 0), (0, 0, 1), rotate)
    cq.exporters.export(cq.Workplane(obj=shape), str(out))
    return out


def _not_the_part(src: Path, out: Path) -> Path:
    """A block 25 % too long where the part should be: the classic "I rebuilt
    it rather than using the one I was given" answer. (A block of the exact
    bounding box would not do here -- two of the synthetic fixture's parts ARE
    boxes, and the check would rightly pass.)"""
    shape = cq.importers.importStep(str(src)).val()
    b = shape.BoundingBox()
    box = cq.Workplane("XY").box(b.xlen * 1.25, b.ylen, b.zlen).val().translate(
        ((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, (b.zmin + b.zmax) / 2))
    cq.exporters.export(cq.Workplane(obj=box), str(out))
    return out


def _supplied_id(case: Path) -> str:
    """A part id this case SUPPLIES (T5 models some of its parts itself)."""
    return next(r["part_id"] for r in _gt_instances(case)
                if S.supplied_part_file(case, r["part_id"]) is not None)


# ── 1. equivalence with the old layout ─────────────────────────────────────
@pytest.mark.parametrize("task", ASSEMBLY_TASKS)
def test_new_layout_scores_exactly_what_the_old_one_scores(task, tmp_path):
    """The whole point of the rebuild: parts x instances is the assembly the
    model would have handed over as one named STEP, so the score cannot depend
    on which way it arrived."""
    case = FX / task / "case1"
    sub = _oracle(case, tmp_path)
    old = _old_style(sub, tmp_path / "old_style.step")
    rn, ro = score_case(case, sub), score_case(case, old)
    assert rn["metric"] == ro["metric"]
    for k in SAME:
        if k in ro and ro[k] is not None:
            assert k in rn and rn[k] is not None, k
            assert abs(float(rn[k]) - float(ro[k])) <= EXACT, (k, rn[k], ro[k], fmt(rn), fmt(ro))
    assert abs(rn["score"] - 1.0) <= 1e-4, fmt(rn)           # and it is the oracle
    assert "error" not in rn, rn.get("error")
    assert rn["submission"]["layout"] == "directory"
    assert rn["submission"]["n_instances"] == len(_gt_instances(case))
    assert not rn["submission"]["failures"] and not rn["submission"]["zeroed_types"]


@pytest.mark.parametrize("task", ASSEMBLY_TASKS)
def test_equivalence_holds_for_a_wrong_answer_too(task, tmp_path):
    """The oracle is 1.0 both ways, which a bug could reproduce by accident.
    The same comparison with one instance displaced by 9 mm gives fractional
    numbers on every column, and they still have to be identical."""
    case = FX / task / "case1"
    sub = _oracle(case, tmp_path)
    recs = json.loads((sub / S.ASSEMBLY / S.INSTANCES).read_text())
    recs[0]["transform"][0][3] = float(recs[0]["transform"][0][3]) + 9.0
    (sub / S.ASSEMBLY / S.INSTANCES).write_text(json.dumps(recs))
    rn = score_case(case, sub)
    ro = score_case(case, _old_style(sub, tmp_path / "old_wrong.step"))
    assert 0.0 < rn["score"] < 1.0, fmt(rn)          # a real answer, not a corner case
    for k in SAME:
        if k in ro and ro[k] is not None:
            assert abs(float(rn[k]) - float(ro[k])) <= EXACT, (k, rn[k], ro[k])


@pytest.mark.parametrize("task", ASSEMBLY_TASKS)
def test_pairing_is_by_name_not_by_geometry(task, tmp_path):
    """Names come from the part FILE names, so both scorers take their
    name-first path; nothing is attributed by invariants."""
    case = FX / task / "case1"
    r = score_case(case, _oracle(case, tmp_path))
    assert r["asm_v1_detail"]["pairing"] == "names"
    assert r["avg_part_detail"]["pairing"] == "names"
    assert not r["avg_part_detail"]["extra_children"]


# ── 2. the assembly is rebuilt: assembly.step is inert ─────────────────────
def test_optional_assembly_step_is_ignored(tmp_path):
    """A wrong assembly.step beside a right parts x instances changes nothing:
    it is never read. (The reverse of the failure the layout exists to close --
    an answer that shows one geometry in the assembly and another in the
    parts.)"""
    case = FX / "t2/case1"
    plain = score_case(case, _oracle(case, tmp_path / "plain"))
    lie = cq.Workplane("XY").box(400, 400, 400).val()
    r = score_case(case, _oracle(case, tmp_path / "with_step", assembly_step=lie))
    for k in SAME:
        if k in plain and plain[k] is not None:
            assert abs(float(r[k]) - float(plain[k])) <= EXACT, (k, r[k], plain[k])
    assert r["submission"]["assembly_step"] == S.ASSEMBLY_STEP
    assert "never scored" in r["submission"]["assembly_step_note"]


# ── 3. supplied parts: the file must BE the supplied part ──────────────────
@pytest.mark.parametrize("rotate", [0.0, 37.0], ids=["reexported", "reexported_and_turned"])
def test_a_reexport_of_the_supplied_part_passes(rotate, tmp_path):
    """Bytes are not the test. A program that imports the supplied STEP and
    writes it out again -- with or without a different part frame -- is
    submitting the supplied part, and the identity check is pose-free."""
    case = FX / "t2/case1"
    pid = _supplied_id(case)
    sub = _oracle(case, tmp_path)
    _reexport(resolve_part(case, pid), sub / S.PARTS / f"{pid}.step", rotate=rotate)
    parsed = S.parse(sub, case_dir=case)
    assert parsed.supplied[pid]["ok"], parsed.supplied[pid]
    assert parsed.supplied[pid]["max_rel_diff"] <= S.IDENTITY_TOL
    assert not parsed.zeroed_types
    assert (sub / S.PARTS / f"{pid}.step").read_bytes() != resolve_part(case, pid).read_bytes()


@pytest.mark.parametrize("task", ["t2"])
def test_a_rebuilt_part_scores_zero_for_its_type_with_the_reason(task, tmp_path):
    """A block where a supplied part should be: that TYPE is 0 in avg_part,
    with a named reason, while the assembly score still measures what was
    actually submitted (T2's headline is asm_v1 alone, so it is untouched by
    the verdict and simply charges the wrong shape).

    T2 is the only task this applies to now: T4 supplies no part at all and T5
    supplies only its purchased types, so a rebuilt part there is the answer
    being measured (the two tests below)."""
    case = FX / task / "case1"
    pid = _supplied_id(case)
    sub = _oracle(case, tmp_path)
    _not_the_part(resolve_part(case, pid), sub / S.PARTS / f"{pid}.step")
    r = score_case(case, sub)
    rec = r["submission"]
    assert rec["supplied_parts"][pid]["ok"] is False
    assert pid in rec["zeroed_types"]
    assert "not the supplied" in rec["zeroed_types"][pid]
    assert [f["code"] for f in rec["failures"]] == ["not_the_supplied_part"]
    assert "submission:" in r["error"] and pid in r["error"]
    by = {row["part_id"]: row for row in r["avg_part_detail"]["per_type"]}
    assert by[pid]["mean"] == 0.0 and by[pid]["zeroed"]
    assert all(row["mean"] == 1.0 for k, row in by.items() if k != pid)
    assert r["avg_part"] < 1.0
    assert r["asm_v1"] < 1.0                       # the block is not the part: the union suffers
    assert 0.0 <= r["score"] <= 1.0
    if r["metric"] == "part_x_asm_v1":             # T4: avg_part is a factor of the headline
        assert abs(r["score"] - r["avg_part"] * r["asm_v1"]) <= 1e-12


def test_t4_models_every_part_and_none_is_identity_checked(tmp_path):
    """T4 supplies no 3-D: every part type is the answer, so none of them is
    compared with a supplied file and a part built wrong is SCORED as a part
    rather than disqualified. The distinction matters: a disqualified type is
    0 by verdict whatever it looks like, while a scored one is measured -- and
    measuring is the whole point of the task."""
    case = FX / "t4/case1"
    recs = _gt_instances(case)
    assert all(S.supplied_part_file(case, r["part_id"]) is None for r in recs), \
        "the t4 fixture must supply no part geometry at all"
    sub = _oracle(case, tmp_path)
    parsed = S.parse(sub, case_dir=case)
    assert parsed.supplied == {} and not parsed.zeroed_types
    pid = recs[0]["part_id"]
    _not_the_part(resolve_part(case, pid), sub / S.PARTS / f"{pid}.step")
    r = score_case(case, sub)
    assert r["submission"]["zeroed_types"] == {}
    assert [f["code"] for f in r["submission"]["failures"]] == []
    by = {row["part_id"]: row for row in r["avg_part_detail"]["per_type"]}
    assert 0.0 < by[pid]["mean"] < 1.0, by[pid]          # measured, not a verdict
    assert all(row["mean"] == 1.0 for k, row in by.items() if k != pid)
    assert r["metric"] == "part_x_asm_v1"
    assert abs(r["score"] - r["avg_part"] * r["asm_v1"]) <= 1e-12
    assert r["score"] < 1.0


def test_t5_models_its_drawing_parts_and_they_are_not_identity_checked(tmp_path):
    """T5's made-to-print part is the answer, so it is scored as a part rather
    than compared with a supplied file (there is none); its purchased parts are
    checked exactly as T2's."""
    case = FX / "t5/case1"
    modelled = [r["part_id"] for r in _gt_instances(case)
                if S.supplied_part_file(case, r["part_id"]) is None]
    assert modelled, "the t5 fixture must have a part with no supplied STEP"
    sub = _oracle(case, tmp_path)
    parsed = S.parse(sub, case_dir=case)
    assert set(parsed.supplied) == {r["part_id"] for r in _gt_instances(case)} - set(modelled)
    # the modelled part built wrong: scored as a part, not disqualified
    pid = modelled[0]
    _not_the_part(resolve_part(case, pid), sub / S.PARTS / f"{pid}.step")
    r = score_case(case, sub)
    assert pid not in r["submission"]["zeroed_types"]
    by = {row["part_id"]: row for row in r["avg_part_detail"]["per_type"]}
    assert by[pid]["mean"] < 1.0 and r["score"] < 1.0


# ── 4. missing and extra ids ───────────────────────────────────────────────
def test_a_missing_bom_id_is_a_named_zero(tmp_path):
    case = FX / "t4/case1"
    recs = _gt_instances(case)
    drop = recs[-1]["part_id"]
    kept = [r for r in recs if r["part_id"] != drop]
    sub = _write(tmp_path, {r["part_id"]: resolve_part(case, r["part_id"]) for r in kept},
                 [{"part_id": r["part_id"], "instance_id": r["instance_id"], "transform": r["T"]}
                  for r in kept])
    r = score_case(case, sub)
    codes = {f["code"]: f for f in r["submission"]["failures"]}
    assert "missing_part_type" in codes and codes["missing_part_type"]["id"] == drop
    assert drop in r["submission"]["zeroed_types"]
    by = {row["part_id"]: row for row in r["avg_part_detail"]["per_type"]}
    assert by[drop]["mean"] == 0.0
    assert drop in r["asm_v1_detail"]["missing"]
    assert r["score"] < 1.0 and 0.0 <= r["score"]


def test_an_extra_part_type_earns_nothing_and_inflates_the_union(tmp_path):
    case = FX / "t2/case1"
    sub = _oracle(case, tmp_path)
    widget = tmp_path / "widget.step"
    cq.exporters.export(cq.Workplane("XY").box(8, 8, 8), str(widget))
    shutil.copyfile(widget, sub / S.PARTS / "widget.step")
    recs = json.loads((sub / S.ASSEMBLY / S.INSTANCES).read_text())
    recs.append({"part_id": "widget", "instance_id": "widget_i1",
                 "transform": [[1, 0, 0, 60], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
    (sub / S.ASSEMBLY / S.INSTANCES).write_text(json.dumps(recs))
    r = score_case(case, sub)
    codes = {f["code"]: f["id"] for f in r["submission"]["failures"]}
    assert codes.get("extra_part_type") == "widget"
    assert r["asm_v1"] < 1.0                        # it is in the union it should not be in
    assert r["avg_part_detail"]["extra_children"] == ["widget_i1"]
    # It is not a reference type, so it earns nothing of its own -- and it is
    # charged a second time through the alignment: avg_part reuses the
    # bounding-box centre asm_v1 computed over the WHOLE submission, which a
    # body outside the assembly's extent moves, so the honest instances are
    # compared in a shifted frame. Pre-existing avg_part behaviour, recorded
    # here rather than asserted away (see the PR's follow-up note).
    assert r["avg_part"] < 1.0


def test_a_part_file_that_is_never_placed_is_a_named_zero(tmp_path):
    case = FX / "t2/case1"
    recs = _gt_instances(case)
    idle = recs[-1]["part_id"]
    sub = _write(tmp_path, {r["part_id"]: resolve_part(case, r["part_id"]) for r in recs},
                 [{"part_id": r["part_id"], "instance_id": r["instance_id"], "transform": r["T"]}
                  for r in recs if r["part_id"] != idle])
    parsed = S.parse(sub, case_dir=case)
    assert "places no instance of it" in parsed.zeroed_types.get(idle, "")
    assert score_case(case, sub)["score"] < 1.0


# ── 5. malformed transforms ────────────────────────────────────────────────
@pytest.mark.parametrize("bad,expect", [
    ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], "not 4x4"),
    ([[1, 0, 0, float("nan")], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "not finite"),
    ([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]], "not a rotation"),
    ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], "det(R)"),
    ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]], "bottom row"),
    ("nonsense", "not a 4x4"),
], ids=["shape", "nan", "scaled", "mirrored", "bottom_row", "not_a_matrix"])
def test_a_malformed_transform_drops_that_instance_with_the_reason(bad, expect, tmp_path):
    """Never a crash and never a silent pass: the instance is not rebuilt, the
    reason names it, and the metrics charge the type exactly as they charge any
    instance that is not there."""
    case = FX / "t4/case1"
    sub = _oracle(case, tmp_path)
    recs = json.loads((sub / S.ASSEMBLY / S.INSTANCES).read_text())
    victim = recs[0]["part_id"]
    recs[0]["transform"] = bad
    (sub / S.ASSEMBLY / S.INSTANCES).write_text(json.dumps(recs))
    r = score_case(case, sub)
    fails = [f for f in r["submission"]["failures"] if f["code"] == "bad_transform"]
    assert len(fails) == 1 and expect in fails[0]["reason"], (fails, expect)
    assert fails[0]["scope"] == "instance"
    assert r["submission"]["n_instances"] == len(recs) - 1
    by = {row["part_id"]: row for row in r["avg_part_detail"]["per_type"]}
    assert by[victim]["mean"] == 0.0
    assert 0.0 <= r["score"] < 1.0


def test_an_instance_without_a_part_file_is_a_named_zero(tmp_path):
    case = FX / "t2/case1"
    sub = _oracle(case, tmp_path)
    recs = json.loads((sub / S.ASSEMBLY / S.INSTANCES).read_text())
    recs.append({"part_id": "ghost", "instance_id": "ghost_i1",
                 "transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
    (sub / S.ASSEMBLY / S.INSTANCES).write_text(json.dumps(recs))
    parsed = S.parse(sub, case_dir=case)
    assert [f.code for f in parsed.failures] == ["missing_part_file"]
    assert len(parsed.instances) == len(recs) - 1


def test_a_part_file_that_is_not_a_step_is_a_named_zero(tmp_path):
    """The rebuild must not raise out of the scorer on a submission's own
    defect: an unreadable part file is a named 0 for its type, its instances
    are not rebuilt, and the rest of the answer is still scored."""
    case = FX / "t2/case1"
    sub = _oracle(case, tmp_path)
    pid = _gt_instances(case)[-1]["part_id"]
    (sub / S.PARTS / f"{pid}.step").write_text("not a step file\n")
    parsed = S.parse(sub, case_dir=case)
    codes = {f.code: f for f in parsed.failures}
    assert "unreadable_part" in codes and codes["unreadable_part"].id == pid
    assert "unusable_part_file" in codes
    assert pid not in parsed.shapes and pid in parsed.zeroed_types
    r = score_case(case, sub)
    assert 0.0 <= r["score"] < 1.0 and "cannot be read as STEP" in r["error"]


@pytest.mark.parametrize("what", ["no_instances", "empty_dir", "unreadable", "no_records"])
def test_an_unscorable_submission_is_zero_with_a_reason(what, tmp_path):
    """A submission that cannot be rebuilt at all: score 0, the reason in the
    record, no exception (and `score` present, because the task declares one)."""
    case = FX / "t2/case1"
    sub = _oracle(case, tmp_path)
    ip = sub / S.ASSEMBLY / S.INSTANCES
    if what == "no_instances":
        ip.unlink()
    elif what == "empty_dir":
        shutil.rmtree(sub)
        (sub / S.PARTS).mkdir(parents=True)
    elif what == "unreadable":
        ip.write_text("{not json")
    else:
        ip.write_text("[]")
    r = score_case(case, sub)
    assert r["score"] == 0.0 and r["iou"] == 0.0
    assert r["error"].startswith("submission:") and len(r["error"]) > 20
    assert r["submission"]["layout"] in ("directory",)


# ── 6. the old layout still works ──────────────────────────────────────────
@pytest.mark.parametrize("task", ASSEMBLY_TASKS)
def test_a_single_step_is_still_scored_and_marked_deprecated(task, tmp_path):
    """Every number measured before this layout existed was produced this way;
    they have to stay comparable, so the path is kept and labelled rather than
    removed."""
    case = FX / task / "case1"
    r = score_case(case, case / "gt/gt.step")
    assert abs(r["score"] - 1.0) <= 1e-4, fmt(r)
    assert r["submission"] == {"layout": "single_step", "deprecated": True,
                              "note": r["submission"]["note"]}
    assert "old layout" in r["submission"]["note"]
    assert "error" not in r


# ── 7. the tools the sandbox stages write this layout ──────────────────────
@pytest.fixture()
def staged_tools(tmp_path, monkeypatch):
    """tools.py exactly as the sandbox writes it, imported in a working
    directory staged like a case (envs/common/sandbox.py TOOLS_PY)."""
    import importlib.util
    from envs.common.sandbox import TOOLS_PY
    work = tmp_path / "work"
    work.mkdir()
    (work / "tools.py").write_text(TOOLS_PY)
    monkeypatch.chdir(work)
    spec = importlib.util.spec_from_file_location("staged_tools", work / "tools.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, work


def test_the_tools_write_a_submission_that_parses_and_scores_one(staged_tools):
    """use_part + export_part + submit_assembly, on the staged input of a real
    fixture: the result parses, every part is the supplied part, and it is the
    oracle."""
    import numpy as np
    tools, work = staged_tools
    case = FX / "t2/case1"
    shutil.copytree(case / "input", work, dirs_exist_ok=True)
    recs = _gt_instances(case)
    tools.use_part(recs[0]["part_id"])                             # the supplied file, copied
    tools.export_part(f"step_files/{recs[1]['part_id']}.step", recs[1]["part_id"])   # by path
    tools.export_part(cq.importers.importStep(                     # as geometry
        str(case / f"input/step_files/{recs[2]['part_id']}.step")), recs[2]["part_id"])
    assert tools.submission_ready() is False                       # no instances.json yet
    tools.submit_assembly([
        {"part_id": recs[0]["part_id"], "transform": recs[0]["T"]},                  # nested 4x4
        {"part_id": recs[1]["part_id"], "transform": np.array(recs[1]["T"])},        # numpy
        {"part_id": recs[2]["part_id"], "instance_id": recs[2]["instance_id"],
         "transform": [x for row in recs[2]["T"] for x in row]},                     # flat 16
    ])
    assert tools.submission_ready() is True
    parsed = S.parse(work / S.SUB_ROOT, case_dir=case)
    assert not parsed.failures, [f.as_dict() for f in parsed.failures]
    assert all(v["ok"] for v in parsed.supplied.values())
    assert [i.instance_id for i in parsed.instances] == [r["instance_id"] for r in recs]
    r = score_case(case, work / S.SUB_ROOT)
    assert abs(r["score"] - 1.0) <= 1e-4, fmt(r)


def test_submit_assembly_refuses_an_instance_with_no_part_file(staged_tools):
    """The model learns in-round instead of scoring zero at the end."""
    tools, work = staged_tools
    with pytest.raises(FileNotFoundError, match="export_part"):
        tools.submit_assembly([{"part_id": "part_01", "transform": [[1, 0, 0, 0], [0, 1, 0, 0],
                                                                    [0, 0, 1, 0], [0, 0, 0, 1]]}])
    assert not (work / S.SUB_ROOT / S.ASSEMBLY / S.INSTANCES).exists()


def test_submit_assembly_refuses_a_transform_that_is_not_4x4(staged_tools, tmp_path):
    tools, work = staged_tools
    cq.exporters.export(cq.Workplane("XY").box(4, 4, 4), str(tmp_path / "p.step"))
    tools.export_part(tmp_path / "p.step", "part_01")
    with pytest.raises(ValueError, match="4x4"):
        tools.submit_assembly([{"part_id": "part_01", "transform": [1, 2, 3]}])


def test_a_cadquery_location_is_an_acceptable_transform(staged_tools):
    tools, work = staged_tools
    cq.exporters.export(cq.Workplane("XY").box(4, 4, 4), str(work / "p.step"))
    tools.export_part(work / "p.step", "part_01")
    tools.submit_assembly([{"part_id": "part_01",
                            "transform": cq.Location(cq.Vector(1, 2, 3))}])
    T = json.loads((work / S.SUB_ROOT / S.ASSEMBLY / S.INSTANCES).read_text())[0]["transform"]
    assert [row[3] for row in T[:3]] == [1.0, 2.0, 3.0]
    assert S.rigid_reason(T) is None


def test_finish_requires_an_answer(staged_tools):
    """`tools.finish` is what the harness appends. No `result` and no
    submission/ is a stated failure the model gets one retry on, not a silent
    zero; a complete submission/ needs no `result` at all."""
    tools, work = staged_tools
    with pytest.raises(ValueError, match="nothing was submitted"):
        tools.finish({})
    cq.exporters.export(cq.Workplane("XY").box(4, 4, 4), str(work / "p.step"))
    tools.export_part(work / "p.step", "part_01")
    tools.submit_assembly([{"part_id": "part_01", "transform": [[1, 0, 0, 0], [0, 1, 0, 0],
                                                                [0, 0, 1, 0], [0, 0, 0, 1]]}])
    assert Path(tools.finish({})).name == S.SUB_ROOT
    # a leftover `result` does not win, and does not break the run either: it is
    # exported beside the submission as a convenience copy
    assert Path(tools.finish({"result": cq.Workplane("XY").box(2, 2, 2)})).name == S.SUB_ROOT
    assert (work / "final.step").exists()
    # and with no submission, a part task is unchanged: `result` -> final.step
    (work / S.SUB_ROOT / S.ASSEMBLY / S.INSTANCES).unlink()
    (work / "final.step").unlink()
    assert Path(tools.finish({"result": cq.Workplane("XY").box(2, 2, 2)})).name == "final.step"


# ── 8. what the episode picks up ───────────────────────────────────────────
def test_the_episode_prefers_a_complete_submission_over_final_step(tmp_path):
    """On an assembly task the directory is the answer; a `result` the model
    kept for its own bookkeeping is exported too and must not win."""
    case = FX / "t2/case1"
    box = SimpleNamespace(dir=tmp_path)
    assert _artifact(box) is None
    (tmp_path / "final.step").write_text("ISO-10303-21;\n")
    assert _artifact(box) == tmp_path / "final.step"
    _oracle(case, tmp_path)
    assert _artifact(box) == tmp_path / S.SUB_ROOT
    # incomplete: no instances.json -> the STEP is the answer again
    (tmp_path / S.SUB_ROOT / S.ASSEMBLY / S.INSTANCES).unlink()
    assert _artifact(box) == tmp_path / "final.step"


def test_locate_accepts_the_four_things_a_caller_has_in_hand(tmp_path):
    case = FX / "t2/case1"
    sub = _oracle(case, tmp_path)
    assert S.locate(sub) == sub                                   # the directory itself
    assert S.locate(tmp_path) == sub                              # the working directory
    assert S.locate(sub / S.ASSEMBLY / S.INSTANCES) == sub        # the instances file
    assert S.locate(case / "gt/gt.step") is None                  # a single STEP: the old layout
    assert S.locate(tmp_path / "nope") is None
    assert S.is_ready(sub) and not S.is_ready(tmp_path / "nope")


def test_the_staged_tools_and_the_parser_agree_on_the_layout():
    """`tools.py` runs INSIDE the container and cannot import
    envs.common.submission, so the path names are written twice. This is the
    guard against the two copies drifting -- a renamed directory on one side
    only would make every submission unparseable, with the model doing
    everything the prompt asked."""
    from envs.common.sandbox import TOOLS_PY
    assert f'SUBMISSION = Path("{S.SUB_ROOT}")' in TOOLS_PY
    assert f'SUBMISSION / "{S.PARTS}"' in TOOLS_PY
    assert f'SUBMISSION / "{S.ASSEMBLY}" / "{S.INSTANCES}"' in TOOLS_PY
    assert f'SUBMISSION / "{S.ASSEMBLY}" / "{S.ASSEMBLY_STEP}"' in TOOLS_PY
    assert f'(root / "{S.ASSEMBLY}" / "{S.INSTANCES}").exists()' in TOOLS_PY
