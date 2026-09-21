"""The meshing guard (envs/geom/meshguard.py) and what the scorers do with it.

Trigger: on 2026-09-18 a gpt-6-astra T4 submission's wire clip -- a circle
swept along a spline with transition="round", BRepCheck invalid -- meshed in
64 s at relative deflection 0.1 and never at the metric's 0.05, and the
scorer sat on a two-part assembly for 79 minutes. `bad_sweep()` below is that
clip, rebuilt from the model's own construction.

    identity      the worker's mesh is the in-place mesh (same triangles,
                  vertices to 1e-12) on a held-out reference part; a moved
                  instance meshes the same on both SIDES of a comparison
    the budget    the sweep raises UnmeshableShape within the budget, the
                  worker is killed and the next call works
    part_v1       the sweep is a named zero (`error: ... unmeshable ...`)
                  within the budget, not a hang
    assembly      the T4 fixture with one part replaced by the sweep: that
                  type is 0 in avg_part with the reason, its instance is
                  named under asm_v1's `excluded_instances`, and the other
                  types still score 1.0
    legacy iou    iou_step_vs_step scores an unmeshable submission 0.0
The budgets here are seconds (monkeypatched), the default is MESH_TIMEOUT_S.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from envs.geom import meshguard as mg  # noqa: E402
from envs.geom import ocp_hashcode_fix  # noqa: E402

ocp_hashcode_fix()
import cadquery as cq  # noqa: E402

FX = REPO / "tests/fixtures"
BUDGET = 6.0                       # seconds: enough for any part here, far under the sweep


def bad_sweep(r: float = 2.1):
    """The clip that started this, rebuilt from the model's own construction:
    a wire bail -- lines and tangent-constrained splines, two of the splines
    2.9 mm long -- swept by a 2.1 mm circle with transition="round". The bend
    radius is under the tube radius, so the tube crosses itself; 8 faces,
    BRepCheck invalid, and it survives a STEP round trip unchanged (314 KB,
    like the original). Measured here: 4 s at deflection 1.0 (125 k
    triangles), 7 s at 0.5, 25 s at 0.2, 64 s at 0.1, unbounded at 0.05."""
    theta = np.deg2rad(22)
    sn, cs = np.sin(theta), np.cos(theta)

    def V(x, q):
        return cq.Vector(float(x), float(q * sn), float(70 - q * cs))

    lower = [(-29.11, 20.45), (-29.23, 31.1), (-26.85, 40.22), (-19.45, 53.09), (-7.61, 61.36),
             (6.85, 64.26), (20.25, 60.25), (30.14, 51.62), (33.25, 40.03), (32.43, 29.61),
             (31.17, 24.18), (29.04, 20.54)]
    a, b, c = V(10, 0), V(-10, 0), V(-12.6, 1.35)
    d, e, f = V(-28.0, 18.8), V(28.3, 18.8), V(12.6, 1.35)
    edges = [cq.Edge.makeLine(a, b),
             cq.Edge.makeSpline([b, V(-11.4, .22), c], tangents=[cq.Vector(-1, 0, 0), V(-1, 1) - V(0, 0)]),
             cq.Edge.makeLine(c, d),
             cq.Edge.makeSpline([d] + [V(x, q) for x, q in lower] + [e],
                                tangents=[V(-1, 1.13) - V(0, 0), V(-1, -1.13) - V(0, 0)]),
             cq.Edge.makeLine(e, f),
             cq.Edge.makeSpline([f, V(11.4, .22), a], tangents=[V(-1, -1) - V(0, 0), cq.Vector(-1, 0, 0)])]
    path = cq.Wire.assembleEdges(edges)
    plane = cq.Plane(origin=a, normal=(-1, 0, 0), xDir=(0, 1, 0))
    return cq.Workplane(plane).circle(r).sweep(path, isFrenet=True, transition="round").val()


@pytest.fixture(autouse=True)
def _short_budget(monkeypatch):
    monkeypatch.setattr(mg, "MESH_TIMEOUT_S", BUDGET)


def _same(a, b, tol=1e-12) -> bool:
    return (a[1].shape == b[1].shape and np.array_equal(a[1], b[1])
            and a[0].shape == b[0].shape and float(np.abs(a[0] - b[0]).max()) <= tol)


# ── identity with the in-place mesh ────────────────────────────────────────
@pytest.mark.parametrize("deflection", [0.05, 0.01])
def test_worker_mesh_is_the_in_place_mesh(deflection):
    from envs.common.part_metric import load_shape
    ref = FX / "t1/case1/gt/gt.step"
    a = mg.tessellate_in_place(load_shape(ref), deflection)
    b = mg.tessellate(load_shape(ref), deflection)
    assert len(a[1]) > 100
    assert _same(a, b), (len(a[1]), len(b[1]))


def test_the_face_walk_is_cadquerys_on_a_shape_that_meshes_whole():
    """With every face triangulated the mesh is exactly what cadquery's
    Shape.tessellate returns: same vertices, same triangles, same order."""
    from envs.common.part_metric import load_shape
    ref = FX / "t1/case1/gt/gt.step"
    shape = load_shape(ref)
    verts, tris = shape.tessellate(0.05, mg.ANGULAR_DEFLECTION)
    V = np.array([[v.x, v.y, v.z] for v in verts]); T = np.array(tris, dtype=np.int64)
    got = mg.tessellate_in_place(load_shape(ref), 0.05)
    assert _same((V, T), got)
    assert mg.faces_without_triangles(shape) == []


def test_a_face_left_without_triangles_is_meshed_again_and_the_mesh_is_whole(monkeypatch, capsys):
    """T1 held-out case11 (2026-09-21): one fillet face got no triangulation
    at angular 0.1, cadquery's walk skipped it, and the hollow mesh scored a
    correct part iou24 0.46. Here one face of the fixture part has its
    triangulation stripped after the first pass; the ladder meshes it on its
    own and the returned mesh covers every face."""
    from OCP.BRepTools import BRepTools
    from envs.common.part_metric import load_shape
    shape = load_shape(FX / "t1/case1/gt/gt.step")
    whole = mg.tessellate_in_place(load_shape(FX / "t1/case1/gt/gt.step"), 0.05)
    victim = max(shape.Faces(), key=lambda f: f.Area())
    real_mesh = type(shape).mesh

    def mesh_then_strip_one_face(self, tol, ang):
        real_mesh(self, tol, ang)
        BRepTools.Clean_s(victim.wrapped)                 # no triangulation on this face any more
    monkeypatch.setattr(type(shape), "mesh", mesh_then_strip_one_face)
    V, T = mg.tessellate_in_place(shape, 0.05)
    assert mg.faces_without_triangles(shape) == []       # the victim has triangles again
    err = capsys.readouterr().err
    assert "1 of" in err and "meshed again at [(2.0, 1.0)]" in err   # the first rung: same angular, doubled deflection
    # nothing missing: the mesh covers the whole surface, and the patch is
    # close to the fine pass (a nudged deflection, not a coarser angle)
    def area(V, T):
        a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2)
    assert abs(area(V, T) - area(*whole)) <= 0.01 * area(*whole)
    assert 0.8 * len(whole[1]) <= len(T) <= len(whole[1])


def test_a_face_no_rung_meshes_is_sealed_never_zeroed(monkeypatch, capsys):
    """A T4 submission's clip (2026-09-21) had one 1 mm^2 planar face that no
    fine setting meshes; the first version of the fallback raised and the
    part scored 0 (0.733 -> 0.089 on the case). Now a face the whole ladder
    fails on is sealed with a fan over its boundary, and the mesh is still
    closed for the voxeliser; only a shape with NO meshable face raises."""
    from envs.common.part_metric import load_shape
    shape = load_shape(FX / "t1/case1/gt/gt.step")
    victim = max((f for f in shape.Faces() if len(f.Wires()) == 1), key=lambda f: f.Area())   # no inner wires: the fan is the face
    monkeypatch.setattr(mg, "faces_without_triangles", lambda s: [victim])
    # every rung "fails": whatever the per-face mesher does, the face reads as empty
    import OCP.BRep as _brep
    real_tri = _brep.BRep_Tool.Triangulation_s
    monkeypatch.setattr(_brep.BRep_Tool, "Triangulation_s",
                        staticmethod(lambda f, loc: None if f.IsSame(victim.wrapped) else real_tri(f, loc)))
    V, T = mg.tessellate_in_place(shape, 0.05)
    assert "1 sealed with a boundary fan" in capsys.readouterr().err
    assert len(T) > 0 and len(V) > 0
    # the seal is a fan over the victim's boundary: its area is close to the face's
    fv, ft = mg._boundary_fan(victim)
    a, b, c = fv[ft[:, 0]], fv[ft[:, 1]], fv[ft[:, 2]]
    assert abs(float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2) - victim.Area()) <= 0.05 * victim.Area()
    # none meshable at all: raises
    monkeypatch.setattr(mg, "faces_without_triangles", lambda s: list(s.Faces()))
    with pytest.raises(mg.UnmeshableShape, match="none of the .* faces"):
        mg.tessellate_in_place(load_shape(FX / "t1/case1/gt/gt.step"), 0.05)


def test_a_moved_instance_meshes_the_same_on_both_sides():
    """Two readings of one placed part through the worker are identical --
    the property the assembly identity relies on (both sides go through the
    worker; the in-place mesh of a LOCATED shape may differ by a few
    triangles, see the module docstring)."""
    from envs.common.part_metric import load_shape
    ref = FX / "t1/case1/gt/gt.step"
    loc = cq.Location(cq.Vector(10, 20, 30), cq.Vector(1, 1, 0), 37)
    a = mg.tessellate(load_shape(ref).moved(loc), 0.05)
    b = mg.tessellate(load_shape(ref).moved(loc), 0.05)
    assert _same(a, b, tol=0.0)


def test_the_budget_kills_the_mesher_and_the_next_call_works():
    t = time.time()
    with pytest.raises(mg.UnmeshableShape, match="did not finish within"):
        mg.tessellate(bad_sweep(), 0.05)
    assert time.time() - t < BUDGET + 5
    assert mg._WORKER is None                              # killed and forgotten
    V, T = mg.tessellate(cq.Workplane("XY").box(10, 20, 30).val(), 0.05)
    assert len(T) == 12 and mg._WORKER is not None and mg._WORKER.alive()


def test_the_sweep_is_a_shape_the_gates_accept():
    """The clip is not garbage: one solid of positive volume, and it does mesh
    at a coarse deflection -- it is the fine mesh that never comes."""
    from envs.common.part_metric import solid_gate
    s = bad_sweep()
    assert solid_gate(s) is None
    V, T = mg.tessellate(s, 2.0)
    assert len(T) > 1000


def test_a_worker_that_dies_is_reported_and_replaced():
    """A worker that dies between calls is replaced without a word; one that
    dies DURING a call (an OCCT crash) is that call's UnmeshableShape."""
    import threading
    mg.tessellate(cq.Workplane("XY").box(1, 1, 1).val(), 0.5)
    mg._WORKER.proc.kill()
    mg._WORKER.proc.wait()
    V, T = mg.tessellate(cq.Workplane("XY").box(1, 1, 1).val(), 0.5)
    assert len(T) == 12
    threading.Timer(1.0, lambda: mg._WORKER.proc.kill()).start()
    with pytest.raises(mg.UnmeshableShape, match="died"):
        mg.tessellate(bad_sweep(), 0.05)
    V, T = mg.tessellate(cq.Workplane("XY").box(1, 1, 1).val(), 0.5)
    assert len(T) == 12


# ── part_v1 ────────────────────────────────────────────────────────────────
def test_part_v1_scores_the_sweep_zero_with_the_reason(tmp_path):
    from envs.common.part_metric import score_part_v1
    sub = tmp_path / "sub.step"
    cq.exporters.export(cq.Workplane(obj=bad_sweep()), str(sub))
    t = time.time()
    r = score_part_v1(FX / "t1/case1/gt/gt.step", sub, orientation="free", pose_mode="iou24_aligned")
    assert time.time() - t < BUDGET + 10
    assert r["score"] == 0.0 and r["coverage"] == 1.0
    assert "unmeshable" in r["error"] and "did not finish" in r["error"]


# ── the assembly scorers ───────────────────────────────────────────────────
def test_assembly_scores_the_rest_when_one_part_is_unmeshable(tmp_path):
    from envs.common import submission as S
    from envs.common.caseformat import resolve_part
    from envs.common.score_case import score_case
    case = FX / "t4/case1"
    recs = json.loads((case / "gt/instances.json").read_text())["instances"]
    sub = tmp_path / S.SUB_ROOT
    (sub / S.PARTS).mkdir(parents=True)
    (sub / S.ASSEMBLY).mkdir(parents=True)
    for r in recs:
        shutil.copyfile(resolve_part(case, r["part_id"]), sub / S.PARTS / f"{r['part_id']}.step")
    (sub / S.ASSEMBLY / S.INSTANCES).write_text(json.dumps(
        [{"part_id": r["part_id"], "instance_id": r["instance_id"], "transform": r["T"]} for r in recs]))
    bad = recs[-1]["part_id"]
    cq.exporters.export(cq.Workplane(obj=bad_sweep()), str(sub / S.PARTS / f"{bad}.step"))

    t = time.time()
    r = score_case(case, sub)
    elapsed = time.time() - t
    # the bad part is meshed twice under the budget (instances() and the
    # part_v1 gate); everything else is ordinary scoring work
    assert elapsed < 2 * BUDGET + 60, elapsed
    ap = {row["part_id"]: row for row in r["avg_part_detail"]["per_type"]}
    assert ap[bad]["mean"] == 0.0
    assert all("unmeshable" in row["error"] for row in r["avg_part_detail"]["per_instance"]
               if row["part_id"] == bad)
    assert all(row["mean"] == 1.0 for pid, row in ap.items() if pid != bad), ap
    v1 = r["asm_v1_detail"]
    assert [d["name"] for d in v1["excluded_instances"]] == [recs[-1]["instance_id"]]
    assert "unmeshable" in v1["excluded_instances"][0]["reason"]
    by = {row["part_id"]: row for row in v1["per_type"]}
    assert by[bad]["n_instances"] == 0 and "unmeshable" in by[bad]["note"]
    assert by[bad]["score"] == 0.0
    # leave-one-type-out with one type absent: the others are pulled down to
    # v_j / (v_j + v_absent) (tests/test_asm_v1.py), measured, not zeroed
    assert all(0.0 < row["score"] < 1.0 for pid, row in by.items() if pid != bad), by
    assert r["metric"] == "part_x_asm_v1"
    assert 0.0 < r["score"] < 1.0
    assert abs(r["score"] - r["avg_part"] * r["asm_v1"]) <= 1e-12


# ── the legacy column ──────────────────────────────────────────────────────
def test_legacy_iou_scores_an_unmeshable_submission_zero(tmp_path):
    from envs.geom import iou_step_vs_step
    sub = tmp_path / "sub.step"
    cq.exporters.export(cq.Workplane(obj=bad_sweep()), str(sub))
    t = time.time()
    assert iou_step_vs_step(FX / "t1/case1/gt/gt.step", sub, 64) == 0.0
    assert time.time() - t < BUDGET + 10
