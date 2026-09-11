"""The headline of every task, through the declared verifier (docs/METRICS.md).

    T1 part_v1 (free)      T3 part_v1 (pinned)      T2 asm_v1 (+ avg_part column)
    T4 part_x_asm_v1 = avg_part x asm_v1 (pinned)   T5 the same, free

  oracle      the reference submitted as itself -> 1.0 on every fixture t1..t5
  dumb        the GT bounding-box block -> strictly below 1; on T4 / T5 the
              product is below both factors
  misplaced   T4: correct parts, one displaced -> avg_part about (n-1)/n (the
              displaced instance scores ~0), asm_v1 drops, product below both
  replaced    T4: correct placement, one part swapped for a box -> both drop
  repeated    a type with three identical posts is removed as a WHOLE by
              asm_v1: its headroom is all three posts' volume
  solid gate  a shell / face compound / empty file scores 0.0 as a part and
              as a child in an assembly; a reference shell raises
  range       every reported score is in [0, 1]
  scope       T5 declares `avg_part_types = "modelled"`: the mean runs over the
              part types whose bom.json row says source = "drawing", the
              supplied types are scored and reported but not averaged in, and a
              case with no modelled type at all is UNSCORABLE (avg_part None)

Synthetic assemblies are written here in the case format (case.json via
write_manifest, parts through caseformat's own layout constants, never a
literal input path); nothing under tests/fixtures/ is modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from envs.geom import ocp_hashcode_fix  # noqa: E402

ocp_hashcode_fix()
import cadquery as cq  # noqa: E402

from envs import tasks as T  # noqa: E402
from envs.common import part_metric as pm  # noqa: E402
from envs.common.asm_v1 import asm_v1  # noqa: E402
from envs.common.avg_part import avg_part  # noqa: E402
from envs.common.avg_part import avg_part as _avg_part  # noqa: E402
from envs.common.caseformat import INSTANCES_FORMAT, STEP_DIR, write_manifest  # noqa: E402
from envs.common.score_case import fmt, score_case  # noqa: E402
from tools.dryrun_case import _score_key, _score_value  # noqa: E402

FX = REPO / "tests/fixtures"
GEOM = ["t1", "t2", "t3", "t4", "t5"]
HEADLINE = {"t1": "part_v1", "t2": "asm_v1", "t3": "part_v1", "t4": "part_x_asm_v1", "t5": "part_x_asm_v1"}
ENV = {"t2": "t2_realparts2assembly", "t4": "t4_parts2assembly", "t5": "t5_drawings2assembly"}
TOL = 1e-4


# ── synthetic assemblies in the case format ────────────────────────────────
def _parts():
    """Four part types, `post` x3. Volumes: base 14400, post 3 x 1508,
    bracket 1280, pin 72 (mm^3)."""
    return {"base": cq.Workplane("XY").box(60, 40, 6).val(),
            "post": cq.Workplane("XY").circle(4).extrude(30).val(),
            "bracket": cq.Workplane("XY").box(16, 10, 8).val(),
            "pin": cq.Workplane("XY").box(3, 3, 8).val()}


PLACES = {"base": [(0, 0, 0)],
          "post": [(-22, 0, 18), (0, 12, 18), (22, 0, 18)],
          "bracket": [(0, -12, 7)],
          "pin": [(-20, -14, 7)]}


def _depose(solid):
    b = solid.BoundingBox()
    c = ((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, (b.zmin + b.zmax) / 2)
    return solid.translate((-c[0], -c[1], -c[2]))


def _T(t):
    return [[1, 0, 0, t[0]], [0, 1, 0, t[1]], [0, 0, 1, t[2]], [0, 0, 0, 1]]


def _placed(parts, places, override=None, replace=None, skip=()):
    """[(name, solid)] named `<part_id>_i<k>`: the de-posed part moved to its
    place. `override` re-places a type, `replace` substitutes a solid for one
    named instance, `skip` leaves types out."""
    out = []
    for pid, sol in parts.items():
        if pid in skip:
            continue
        dp = _depose(sol)
        for k, t in enumerate((override or {}).get(pid, places[pid]), 1):
            name = f"{pid}_i{k}"
            s = (replace or {}).get(name, dp.translate(t))
            out.append((name, s))
    return out


def _save_named(path: Path, named):
    a = cq.Assembly(name="asm")
    for n, s in named:
        a.add(cq.Workplane(obj=s), name=n)
    a.save(str(path), "STEP")
    return path


def _write_case(root: Path, env: str, parts, places, drawing_parts=()) -> Path:
    """A case in the format: supplied parts under the input STEP directory
    (caseformat.STEP_DIR), modelled parts under gt/parts, gt/gt.step from
    the placed solids, gt/instances.json, input/bom.json, case.json."""
    (root / "input" / STEP_DIR).mkdir(parents=True)
    (root / "gt/parts").mkdir(parents=True)
    items, inst = [], []
    for pid, sol in parts.items():
        dp = _depose(sol)
        if pid in drawing_parts:
            cq.exporters.export(cq.Workplane(obj=dp), str(root / "gt/parts" / f"{pid}.step"))
            items.append({"part_id": pid, "quantity": len(places[pid]), "source": "drawing",
                          "file": f"part_drawings/{pid}.pdf"})
        else:
            cq.exporters.export(cq.Workplane(obj=dp), str(root / "input" / STEP_DIR / f"{pid}.step"))
            items.append({"part_id": pid, "quantity": len(places[pid]), "source": "step",
                          "file": f"{STEP_DIR}/{pid}.step"})
        for k, t in enumerate(places[pid], 1):
            inst.append({"instance_id": f"{pid}_i{k}", "part_id": pid, "T": _T(t)})
    if not any(root.glob("gt/parts/*")):
        (root / "gt/parts").rmdir()
    (root / "input/bom.json").write_text(json.dumps(
        {"items": items, "n_part_types": len(items), "n_instances": len(inst)}))
    (root / "gt/instances.json").write_text(json.dumps(
        {"format": INSTANCES_FORMAT, "frame": "part frame -> assembly frame, row-major 4x4, mm",
         "instances": inst}))
    _save_named(root / "gt/gt.step", _placed(parts, places))
    write_manifest(root, id=root.name, env=env, kind="assembly",
                   source={"repo": "synthetic", "split": "test"}, synthetic=True,
                   generator={"tool": "tests/test_headlines.py"}, redaction={"status": "clean"})
    return root


@pytest.fixture(scope="module")
def t4_case(tmp_path_factory):
    return _write_case(tmp_path_factory.mktemp("t4") / "synth", ENV["t4"], _parts(), PLACES)


@pytest.fixture(scope="module")
def t5_case(tmp_path_factory):
    """The bracket and the pin are modelled from drawings (gt/parts); the rest supplied."""
    return _write_case(tmp_path_factory.mktemp("t5") / "synth", ENV["t5"], _parts(), PLACES,
                       drawing_parts=("bracket", "pin"))


def _dumb_block(gt_step: Path, out: Path) -> Path:
    gt = cq.importers.importStep(str(gt_step)).val()
    b = gt.BoundingBox()
    box = cq.Workplane("XY").box(b.xlen, b.ylen, b.zlen).val().translate(
        ((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, (b.zmin + b.zmax) / 2))
    cq.exporters.export(cq.Workplane(obj=box), str(out))
    return out


def _in_unit(x) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x) and 0.0 <= x <= 1.0


def _assert_range(r: dict):
    """Every reported score in [0, 1]: the headline, the factors, the legacy
    columns, and every per-type / per-instance number underneath."""
    for k in ("score", "asm_v1", "avg_part", "part_x_asm_v1", "iou", "iou_term", "surf_f1", "pix_fg",
              "hit", "hit_prec", "hit_f1", "rubric"):
        if k in r and r[k] is not None:
            assert _in_unit(r[k]), (k, r[k])
    for row in (r.get("asm_v1_detail") or {}).get("per_type", []):
        if row.get("score") is not None:
            assert _in_unit(row["score"]), row
    ap = r.get("avg_part_detail") or {}
    for row in ap.get("per_instance", []):
        assert _in_unit(row["score"]), row
        for k in ("iou_term", "surf_f1", "pix_fg"):
            if k in row:
                assert _in_unit(row[k]), row
    for row in ap.get("per_type", []):
        assert _in_unit(row["mean"]) and all(_in_unit(x) for x in row["scores"]), row


# ── declarations ───────────────────────────────────────────────────────────
def test_headlines_are_declared():
    assert set(T.METRICS) == {"legacy", "asm_v1", "part_v1", "part_x_asm_v1", "ecad_v2"}
    t = T.load("t1_drawing2part")
    assert (t.metric, t.orientation, t.pose_mode) == ("part_v1", "free", "iou24_aligned")
    t = T.load("t3_part2step")
    assert (t.metric, t.orientation, t.pose_mode) == ("part_v1", "pinned", "lab")
    t = T.load("t2_realparts2assembly")
    assert (t.metric, t.orientation, t.pose_mode) == ("asm_v1", "free", "iou24_aligned")
    t = T.load("t4_parts2assembly")
    assert (t.metric, t.orientation, t.pose_mode) == ("part_x_asm_v1", "pinned", "lab")
    t = T.load("t5_drawings2assembly")
    assert (t.metric, t.orientation, t.pose_mode) == ("part_x_asm_v1", "free", "iou24_aligned")
    assert T.load("t6_pcb2schematic").metric == "ecad_v2"       # the ecad verifier's own Metric V2


def test_the_avg_part_scope_is_declared():
    """T5 declares `avg_part_types = "modelled"`; every other task keeps "all".
    Declared, never sniffed: a case with one step_files/ entry more or less
    must not silently change what the mean is over, and a misspelling must
    raise rather than quietly average over everything again."""
    from envs.verifiers.assembly import avg_part_types
    assert set(T.AVG_PART_TYPES) == {"all", "modelled"} and T.DEFAULT_AVG_PART_TYPES == "all"
    assert T.load("t5_drawings2assembly").avg_part_types == "modelled"
    for tid in ("t1_drawing2part", "t2_realparts2assembly", "t3_part2step",
                "t4_parts2assembly", "t6_pcb2schematic"):
        assert T.load(tid).avg_part_types == "all", tid
    assert avg_part_types(None) == "all" and avg_part_types({"verify": {}}) == "all"
    assert avg_part_types({"verify": {"avg_part_types": "modelled"}}) == "modelled"
    assert avg_part_types(T.load("t5_drawings2assembly")) == "modelled"
    with pytest.raises(ValueError):
        avg_part_types({"verify": {"avg_part_types": "modeled"}})      # one l, not a scope


# ── 1. oracle ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("task", GEOM)
def test_oracle_scores_one(task):
    d = FX / task / "case1"
    r = score_case(d, d / "gt/gt.step")
    assert r["metric"] == HEADLINE[task]
    assert abs(r["score"] - 1.0) <= TOL, fmt(r)
    assert _score_key(r) == "score" and _score_value(r) == r["score"]
    if task in ("t2", "t4", "t5"):
        assert abs(r["asm_v1"] - 1.0) <= TOL and abs(r["avg_part"] - 1.0) <= TOL
        ap = r["avg_part_detail"]
        assert ap["pairing"] == "geometry" and ap["n_paired"] == ap["n_instances"] and not ap["extra_children"]
        assert all(row["score"] == 1.0 for row in ap["per_instance"])
    if task in ("t4", "t5"):
        assert r["part_x_asm_v1"] == r["score"] == r["avg_part"] * r["asm_v1"]
    if task in ("t1", "t3"):
        assert r["identical"] is True
    _assert_range(r)


def test_oracle_named_children_pair_by_names(t4_case, tmp_path):
    """The oracle re-exported through cq.Assembly with `<part_id>_i<k>`
    names: the names path, every instance paired, 1.0 -- and the tessellation
    identity rule is what makes it exactly 1.0 (one STEP round trip changes
    the triangles of the same vertices; see part_metric)."""
    sub = _save_named(tmp_path / "named.step", _placed(_parts(), PLACES))
    r = score_case(t4_case, sub)
    assert r["metric"] == "part_x_asm_v1" and abs(r["score"] - 1.0) <= TOL, fmt(r)
    ap = r["avg_part_detail"]
    assert ap["pairing"] == "names" and ap["n_paired"] == 6 and ap["n_types"] == 4
    assert {row["part_id"]: row["n_instances"] for row in ap["per_type"]} == {"base": 1, "post": 3, "bracket": 1, "pin": 1}
    assert r["asm_v1_detail"]["pairing"] == "names"
    _assert_range(r)


# ── 2. dumb ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("task", GEOM)
def test_dumb_block_scores_below_one(task, tmp_path):
    d = FX / task / "case1"
    r = score_case(d, _dumb_block(d / "gt/gt.step", tmp_path / "dumb.step"))
    assert r["score"] < 1.0 - 1e-3, fmt(r)
    if task in ("t4", "t5"):
        assert r["score"] == r["part_x_asm_v1"] <= min(r["avg_part"], r["asm_v1"]) < 1.0
        assert r["asm_v1"] == 0.0 and r["avg_part"] == 0.0           # nothing paired, every type missing
    if task in ("t1", "t3"):
        assert r["iou_term"] <= 0.01 and r["identical"] is False
    _assert_range(r)


# ── 3. T4: one misplaced / one replaced ────────────────────────────────────
def test_t4_one_misplaced_instance(t4_case, tmp_path):
    """Correct parts, the bracket moved 20 mm along x (clear of everything).
    Its part_v1 in the reference frame is ~0 (samples outside the reference
    cube, surface points beyond tau, render shifted), so avg_part is about
    (n-1)/n over the four types; asm_v1 drops (the bracket clips to 0 and
    pulls the others down); the product is below both factors."""
    sub = _save_named(tmp_path / "moved.step",
                      _placed(_parts(), PLACES, override={"bracket": [(20, -12, 7)]}))
    r = score_case(t4_case, sub)
    ap = r["avg_part_detail"]
    by = {row["part_id"]: row for row in ap["per_type"]}
    assert by["bracket"]["mean"] < 0.1, by["bracket"]
    for pid in ("base", "post", "pin"):
        assert by[pid]["mean"] == 1.0, by[pid]
    assert abs(r["avg_part"] - (3 + by["bracket"]["mean"]) / 4) <= 1e-6
    assert 0.7 <= r["avg_part"] <= 0.8
    v1 = {row["part_id"]: row for row in r["asm_v1_detail"]["per_type"]}
    assert v1["bracket"]["score"] == 0.0 and 0.0 < r["asm_v1"] < 1.0
    assert r["score"] == r["part_x_asm_v1"] < min(r["avg_part"], r["asm_v1"])
    _assert_range(r)


def test_t4_one_part_replaced_by_a_box(t4_case, tmp_path):
    """Correct placement, post_i2 replaced by a box of the post's bounding
    box: that instance is not the part (its iou does not beat the reference
    post's own bounding-cylinder baseline; surface and render disagree), so
    avg_part drops through the post type's mean, and asm_v1 drops (the box
    adds volume the reference does not have)."""
    post = _depose(_parts()["post"])
    b = post.BoundingBox()
    box = cq.Workplane("XY").box(b.xlen, b.ylen, b.zlen).val().translate(PLACES["post"][1])
    sub = _save_named(tmp_path / "boxed.step", _placed(_parts(), PLACES, replace={"post_i2": box}))
    r = score_case(t4_case, sub)
    ap = r["avg_part_detail"]
    by = {row["part_id"]: row for row in ap["per_type"]}
    assert by["post"]["n_paired"] == 3
    scores = sorted(by["post"]["scores"])
    assert scores[0] < 0.8 and scores[1] == scores[2] == 1.0, by["post"]
    assert r["avg_part"] < 1.0 - 0.02 and r["asm_v1"] < 1.0 - 1e-3, fmt(r)
    assert r["score"] < min(r["avg_part"], r["asm_v1"]) + 1e-12
    _assert_range(r)


def test_t4_pinned_orientation_is_charged_t5_free_is_not(t4_case, t5_case, tmp_path):
    """The bracket (16 x 10 x 8) turned 90 degrees about z in place. T4 is
    pinned: the instance is compared at the delivered pose and loses. T5 is
    free: the 24-rotation search about the reference instance's centre finds
    the turn, and iou24_aligned applies it to all three terms, so avg_part
    stays 1.0 -- the misorientation is asm_v1's to charge, and it does."""
    turned = _depose(_parts()["bracket"]).rotate((0, 0, 0), (0, 0, 1), 90).translate(PLACES["bracket"][0])
    sub = _save_named(tmp_path / "turned.step", _placed(_parts(), PLACES, replace={"bracket_i1": turned}))
    r4 = score_case(t4_case, sub)
    r5 = score_case(t5_case, sub)
    b4 = {row["part_id"]: row for row in r4["avg_part_detail"]["per_type"]}["bracket"]
    b5 = {row["part_id"]: row for row in r5["avg_part_detail"]["per_type"]}["bracket"]
    assert b4["mean"] < 0.9 < b5["mean"], (b4, b5)
    i5 = next(x for x in r5["avg_part_detail"]["per_instance"] if x["part_id"] == "bracket")
    assert i5["rotation_applied"] is True and i5["surf_f1"] >= 0.999 and i5["pix_fg"] >= 0.999
    for r in (r4, r5):
        assert {row["part_id"]: row for row in r["asm_v1_detail"]["per_type"]}["bracket"]["score"] < 0.9
        _assert_range(r)


def test_t5_modelled_part_is_scored_against_gt_parts(t5_case, tmp_path):
    """T5: the bracket and the pin exist only under gt/parts. A submission
    that models the pin 1 mm too long still pairs it (names) and scores it
    below 1 on the reference instance; the supplied posts stay at 1.0."""
    long_pin = cq.Workplane("XY").box(3, 3, 9).val().translate(PLACES["pin"][0])
    sub = _save_named(tmp_path / "longpin.step", _placed(_parts(), PLACES, replace={"pin_i1": long_pin}))
    r = score_case(t5_case, sub)
    by = {row["part_id"]: row for row in r["avg_part_detail"]["per_type"]}
    assert by["pin"]["n_paired"] == 1 and 0.0 < by["pin"]["mean"] < 1.0, by["pin"]
    assert by["post"]["mean"] == 1.0 and by["base"]["mean"] == 1.0 and by["bracket"]["mean"] == 1.0
    assert r["avg_part_detail"]["pairing"] == "names"
    assert r["score"] < 1.0
    _assert_range(r)


# ── 4. repeated type: removed as a whole ───────────────────────────────────
def test_repeated_type_is_removed_as_a_whole(t4_case):
    """`post` x3 in the oracle. Leave-one-TYPE-out: the headroom
    1 - baseline_post is the volume of ALL THREE posts over the assembly
    volume (voxel tolerance), three times what removing one post would
    leave -- and every type still scores 1.0."""
    r = asm_v1(t4_case / "gt/gt.step", t4_case / "gt/gt.step", t4_case, pinned=True)
    by = {row["part_id"]: row for row in r["per_type"]}
    assert by["post"]["n_instances"] == 3 and by["post"]["quantity"] == 3
    v_post = np.pi * 16 * 30
    vol = {"base": 14400.0, "post": 3 * v_post, "bracket": 1280.0, "pin": 72.0}
    total = sum(vol.values())
    head = 1.0 - by["post"]["baseline"]
    assert abs(head - vol["post"] / total) <= 0.03, (head, vol["post"] / total)   # 64^3 inflates thin posts a little
    assert head > 2.5 * (v_post / total)                       # not one post's worth
    assert all(abs(row["score"] - 1.0) <= TOL for row in r["per_type"])
    assert abs(r["asm_v1"] - 1.0) <= TOL


def test_dropping_one_post_costs_the_whole_type_headroom(t4_case, tmp_path):
    """Two of three posts submitted: the type's baseline is unchanged (all
    posts out either way), its gain is two posts' worth, so score_post is
    about 2/3 -- and avg_part's post type is exactly 2/3 (one instance
    unpaired at 0)."""
    sub = _save_named(tmp_path / "two_posts.step",
                      _placed(_parts(), PLACES, override={"post": PLACES["post"][:2]}))
    r = score_case(t4_case, sub)
    v1 = {row["part_id"]: row for row in r["asm_v1_detail"]["per_type"]}["post"]
    assert v1["n_instances"] == 2 and 0.55 <= v1["score"] <= 0.75, v1
    ap = {row["part_id"]: row for row in r["avg_part_detail"]["per_type"]}["post"]
    assert ap["n_paired"] == 2 and sorted(ap["scores"]) == [0.0, 1.0, 1.0] and abs(ap["mean"] - 2 / 3) < 1e-6
    _assert_range(r)


# ── 5. solid gate ──────────────────────────────────────────────────────────
def _shell_of(solid):
    return cq.Shell.makeShell(solid.Faces())


def _faces_of(solid):
    return cq.Compound.makeCompound(solid.Faces()[:3])


@pytest.mark.parametrize("what", ["shell", "faces", "empty", "garbage"])
def test_solid_gate_on_a_part(what, tmp_path):
    gt = FX / "t1/case1/gt/gt.step"
    sub = tmp_path / f"{what}.step"
    solid = pm.load_shape(gt)
    if what == "shell":
        cq.exporters.export(cq.Workplane(obj=_shell_of(solid)), str(sub))
    elif what == "faces":
        cq.exporters.export(cq.Workplane(obj=_faces_of(solid)), str(sub))
    elif what == "empty":
        sub.write_text("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n")
    else:
        sub.write_text("not a step file\n")
    r = score_case(FX / "t1/case1", sub)
    assert r["score"] == 0.0 and r["coverage"] == 1.0, r
    assert r["iou_term"] == r["surf_f1"] == r["pix_fg"] == 0.0
    assert "unusable" in r["error"]
    if what in ("shell", "faces"):
        assert "no solid" in r["error"], r["error"]
        # Without the gate the shell of the reference scores as a near-perfect part.
        assert pm.solid_gate(pm.load_shape(sub)) is not None
    _assert_range(r)


def test_solid_gate_on_an_assembly_child(t4_case, tmp_path):
    """The bracket submitted as its shell, in the right place: that instance
    is 0.0 on avg_part (no solid), the others 1.0, so the product drops to
    0.75. (asm_v1 alone cannot see it: its voxeliser fills any closed
    surface, so the shell voxelises like the solid -- the gate is the part
    metric's job.)"""
    shell = _shell_of(_depose(_parts()["bracket"]).translate(PLACES["bracket"][0]))
    sub = _save_named(tmp_path / "shell_child.step", _placed(_parts(), PLACES, replace={"bracket_i1": shell}))
    r = score_case(t4_case, sub)
    ap = r["avg_part_detail"]
    row = next(x for x in ap["per_instance"] if x["part_id"] == "bracket")
    assert row["child"] == "bracket_i1" and row["score"] == 0.0 and "no solid" in row["error"], row
    by = {t["part_id"]: t["mean"] for t in ap["per_type"]}
    assert by == {"base": 1.0, "post": 1.0, "bracket": 0.0, "pin": 1.0}
    assert abs(r["avg_part"] - 0.75) <= 1e-6 and r["score"] == r["avg_part"] * r["asm_v1"] <= 0.75
    _assert_range(r)


def test_reference_without_a_solid_raises(tmp_path):
    """A reference shell is a broken case, not a score: part_v1 raises, and
    so does the assembly verifier when a gt/parts file is a shell."""
    gt = FX / "t1/case1/gt/gt.step"
    shell = tmp_path / "ref_shell.step"
    cq.exporters.export(cq.Workplane(obj=_shell_of(pm.load_shape(gt))), str(shell))
    with pytest.raises(ValueError, match="no solid"):
        pm.score_part_v1(shell, gt, orientation="free", pose_mode="iou24_aligned")
    case = _write_case(tmp_path / "broken", ENV["t5"], _parts(), PLACES, drawing_parts=("bracket",))
    cq.exporters.export(cq.Workplane(obj=_shell_of(_depose(_parts()["bracket"]))), str(case / "gt/parts/bracket.step"))
    with pytest.raises(ValueError, match="no solid"):
        score_case(case, case / "gt/gt.step")


# ── 6. range on every synthetic submission ─────────────────────────────────
def test_every_score_in_unit_interval(t4_case, t5_case, tmp_path):
    subs = {
        "oracle": t4_case / "gt/gt.step",
        "dumb": _dumb_block(t4_case / "gt/gt.step", tmp_path / "dumb.step"),
        "no_pin": _save_named(tmp_path / "no_pin.step", _placed(_parts(), PLACES, skip=("pin",))),
        "far": _save_named(tmp_path / "far.step", _placed(_parts(), {k: [(x + 500, y, z) for x, y, z in v]
                                                                       for k, v in PLACES.items()})),
        "extra": _save_named(tmp_path / "extra.step", _placed(_parts(), PLACES) +
                             [("widget_i1", cq.Workplane("XY").box(6, 6, 6).val().translate((20, -14, 6)))]),
    }
    for name, sub in subs.items():
        for case in (t4_case, t5_case):
            r = score_case(case, sub)
            assert "score" in r and r["metric"] == "part_x_asm_v1", name
            _assert_range(r)
    r = score_case(t4_case, subs["extra"])
    assert r["avg_part_detail"]["extra_children"] == ["widget_i1"] and r["avg_part"] == 1.0
    assert r["asm_v1"] < 1.0                                    # the widget inflates the union


def test_avg_part_without_alignment_scores_zero(t4_case):
    """No asm_v1 alignment to reuse -> 0 with the reason, never an exception."""
    r = avg_part(t4_case, t4_case / "gt/gt.step", orientation="pinned", pose_mode="lab",
                 asm={"asm_v1": 0.0, "error": "no submission instances"})
    assert r["avg_part"] == 0.0 and "no alignment" in r["error"]
    assert all(t["mean"] == 0.0 for t in r["per_type"])


# ── 7. the T5 scope: the mean runs over the MODELLED types only ────────────
def _v1(case: Path, sub: Path, *, pinned: bool = False):
    return asm_v1(case / "gt/gt.step", sub, case, pinned=pinned)


def test_t5_counts_and_flags_the_types_it_averaged(t5_case):
    """The detail says what the mean was over: the totals, the types left out
    and why each one was. `t5_case` supplies `base` and `post` and has the
    bracket and the pin modelled from drawings."""
    ap = score_case(t5_case, t5_case / "gt/gt.step")["avg_part_detail"]
    assert ap["avg_part_types"] == "modelled"
    assert (ap["n_types"], ap["n_types_in_mean"], ap["n_types_excluded"]) == (4, 2, 2)
    assert ap["excluded_types"] == ["base", "post"]
    by = {row["part_id"]: row for row in ap["per_type"]}
    for pid in ("bracket", "pin"):
        assert by[pid]["in_mean"] is True and "excluded_reason" not in by[pid], by[pid]
    for pid in ("base", "post"):
        assert by[pid]["in_mean"] is False, by[pid]
        assert "source='step'" in by[pid]["excluded_reason"], by[pid]
        assert by[pid]["mean"] == 1.0 and by[pid]["n_paired"] == by[pid]["n_instances"]
    assert len(ap["per_instance"]) == 6                       # every instance still scored


def test_t5_a_supplied_type_that_fails_is_scored_but_not_averaged(t5_case, t4_case, tmp_path):
    """`post_i2` replaced by a box of the post's own bounding box: a SUPPLIED
    type, submitted wrong. It is scored and reported -- the client has to see
    that a supplied part was mishandled -- and it must not re-enter the mean
    through the back door, because exclusion is decided by the BOM's `source`
    and never by a score. asm_v1 charges it, so the headline still drops.

    The same submission on T4, which declares `all`, drops avg_part itself:
    that contrast IS the change."""
    post = _depose(_parts()["post"])
    b = post.BoundingBox()
    box = cq.Workplane("XY").box(b.xlen, b.ylen, b.zlen).val().translate(PLACES["post"][1])
    sub = _save_named(tmp_path / "boxed_supplied.step", _placed(_parts(), PLACES, replace={"post_i2": box}))

    r5 = score_case(t5_case, sub)
    by = {row["part_id"]: row for row in r5["avg_part_detail"]["per_type"]}
    assert by["post"]["in_mean"] is False and min(by["post"]["scores"]) < 0.8, by["post"]
    assert r5["avg_part"] == 1.0                              # bracket and pin are still verbatim
    assert r5["asm_v1"] < 1.0 - 1e-3 and r5["score"] == r5["asm_v1"], fmt(r5)

    r4 = score_case(t4_case, sub)
    assert r4["avg_part_detail"]["n_types_excluded"] == 0
    assert r4["avg_part"] < 1.0 - 0.02 and r4["avg_part"] < r5["avg_part"], fmt(r4)
    _assert_range(r5)
    _assert_range(r4)


def test_t5_submitting_only_the_supplied_parts_earns_nothing(t5_case, tmp_path):
    """The arithmetic the scope exists for. A submission that re-exports the
    supplied parts at the right places and models neither drawing part scores
    avg_part **0.0** under `modelled` (both modelled types unpaired) -- and
    0.5 under `all`, which is the free-1.0s problem: on the real T5 case1, 16
    of 21 types are supplied and `all` would hand it 16/21 = 0.76."""
    sub = _save_named(tmp_path / "supplied_only.step",
                      _placed(_parts(), PLACES, skip=("bracket", "pin")))
    r = score_case(t5_case, sub)
    by = {row["part_id"]: row for row in r["avg_part_detail"]["per_type"]}
    for pid in ("bracket", "pin"):
        assert by[pid]["in_mean"] is True and by[pid]["n_paired"] == 0 and by[pid]["mean"] == 0.0, by[pid]
    assert by["base"]["mean"] == by["post"]["mean"] == 1.0     # the supplied halves are perfect
    assert r["avg_part"] == 0.0 and r["score"] == 0.0, fmt(r)

    same = _avg_part(t5_case, sub, orientation="free", pose_mode="iou24_aligned",
                     asm=_v1(t5_case, sub), types="all")
    assert same["avg_part"] == 0.5 and same["n_types_in_mean"] == 4
    _assert_range(r)


def test_t5_a_modelled_type_with_no_submitted_part_scores_zero(t5_case, tmp_path):
    """A BOM type with nothing submitted for it: the reference instance stays
    unpaired and scores 0, and because the type is in scope the mean carries
    it (bracket 1.0, pin 0.0 -> 0.5)."""
    sub = _save_named(tmp_path / "no_pin.step", _placed(_parts(), PLACES, skip=("pin",)))
    r = score_case(t5_case, sub)
    by = {row["part_id"]: row for row in r["avg_part_detail"]["per_type"]}
    assert by["pin"]["in_mean"] is True and by["pin"]["n_paired"] == 0
    assert by["pin"]["scores"] == [0.0] and by["pin"]["mean"] == 0.0
    assert abs(r["avg_part"] - 0.5) <= 1e-9, [(k, v["mean"], v["in_mean"]) for k, v in by.items()]
    row = next(x for x in r["avg_part_detail"]["per_instance"] if x["part_id"] == "pin")
    assert row["child"] is None and row["error"] == "no submitted instance paired"
    _assert_range(r)


def test_t5_with_no_drawing_row_is_unscorable_not_zero_and_not_one(tmp_path):
    """A case that declares `modelled` and has no `source = "drawing"` row: the
    mean is undefined, so `avg_part` and the headline are **None** with an
    `unscorable_reason`. Not 0.0 (which would read as "the model built
    nothing") and not 1.0 ("every part right") -- neither is a claim this case
    can make about a model. Every type is still scored and reported."""
    case = _write_case(tmp_path / "all_supplied", ENV["t5"], _parts(), PLACES)   # no drawing_parts
    r = score_case(case, case / "gt/gt.step")
    assert r["metric"] == "part_x_asm_v1"
    assert r["avg_part"] is None and r["part_x_asm_v1"] is None
    assert "score" in r and r["score"] is None                # declared, and not computable here
    assert "drawing" in r["unscorable_reason"] and "modelled" in r["unscorable_reason"]
    ap = r["avg_part_detail"]
    assert (ap["n_types"], ap["n_types_in_mean"], ap["n_types_excluded"]) == (4, 0, 4)
    assert all(row["mean"] == 1.0 and row["in_mean"] is False for row in ap["per_type"]), ap["per_type"]
    assert all(row["score"] == 1.0 for row in ap["per_instance"])
    assert abs(r["asm_v1"] - 1.0) <= TOL                      # the assembly half is untouched
    # A reader that keys on the headline must see "no number", never the
    # legacy IoU standing in for it.
    assert _score_key(r) == "score" and _score_value(r) is None
    assert "UNSCORABLE" in fmt(r)


def test_t5_with_no_bom_is_unscorable(tmp_path):
    """The BOM is what says which types were modelled (docs/CASE_FORMAT.md), so
    a case without one cannot define the mean either -- same answer, its own
    reason. (asm_v1 needs the BOM too and reports its own error.)"""
    case = _write_case(tmp_path / "no_bom", ENV["t5"], _parts(), PLACES,
                       drawing_parts=("bracket", "pin"))
    (case / "input/bom.json").unlink()
    r = score_case(case, case / "gt/gt.step")
    assert r["avg_part"] is None and r["score"] is None
    assert "bom.json" in r["unscorable_reason"], r["unscorable_reason"]
    assert r["avg_part_detail"]["n_types_in_mean"] == 0
    assert _score_value(r) is None


def test_t2_and_t4_still_average_over_every_type(t4_case):
    """The scope changed on T5 only. T2 and T4 supply every part, so their
    `avg_part` keeps averaging over all of them -- on T2 a diagnostic column,
    on T4 half the headline with nothing else to average."""
    r = score_case(t4_case, t4_case / "gt/gt.step")
    ap = r["avg_part_detail"]
    assert ap["avg_part_types"] == "all"
    assert (ap["n_types"], ap["n_types_in_mean"], ap["n_types_excluded"]) == (4, 4, 0)
    assert ap["excluded_types"] == [] and all(row["in_mean"] for row in ap["per_type"])
    assert abs(r["avg_part"] - 1.0) <= TOL and abs(r["score"] - 1.0) <= TOL
    d = FX / "t2/case1"
    r2 = score_case(d, d / "gt/gt.step")
    assert r2["avg_part_detail"]["avg_part_types"] == "all"
    assert r2["avg_part_detail"]["n_types_excluded"] == 0 and r2["metric"] == "asm_v1"

