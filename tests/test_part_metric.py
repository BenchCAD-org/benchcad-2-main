"""part_v1 (envs/common/part_metric.py, change 26): the ruler for T1 / T3.

Same discipline as test_examples.py -- synthetic geometry in git
(tests/fixtures/t1/case1, tests/fixtures/t3/case1), no benchmark data -- plus
six expert-scored STEP pairs that pin the numbers when they are on disk (skipped,
loudly, when they are not).

  oracle        the reference submitted as-is scores 1.0 on every term
  dumb          the GT bounding-box block earns nothing on iou_term
  mirror        a mirrored chiral part is NOT recovered by the 24 proper rotations
  quarter turn  T1 (free, iou24_aligned) repairs it on all three terms; T3 (pinned) does not
  background    the silhouette samples the frame corner, never assumes white
  coverage      a term that fails drops out with the weights renormalised
  fixtures      six reference pairs pin the SURFACE, PIXEL and iou numbers
                its fixture set against the true voxelisation
  noise         the sampler the iou term used until change 40 disagrees with itself,
                the voxeliser that replaced it does not -- both on record
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from envs import tasks as T                                          # noqa: E402
from envs.common import part_metric as pm                            # noqa: E402
from envs.common.score_case import fmt, load_task, score_case        # noqa: E402

FX = REPO / "tests/fixtures"
PART_CASES = [FX / "t1/case1", FX / "t3/case1"]
SYNTH_T1 = FX / "t1/case1"
SYNTH_T3 = FX / "t3/case1"
FIXTURES = Path.home() / "cad-agent-work" / "part_metric_fixtures"
FIXTURE_ROWS = sorted(p.parent for p in FIXTURES.glob("row*/expected.json")) if FIXTURES.exists() else []


def _ids(paths):
    return [str(p.relative_to(FX)) for p in paths]


def _export(shape, path: Path) -> Path:
    from envs.geom import ocp_hashcode_fix
    ocp_hashcode_fix()
    import cadquery as cq
    cq.exporters.export(cq.Workplane(obj=shape), str(path))
    return path


def _gt_shape(case: Path):
    return pm.load_shape(case / "gt/gt.step")


def _declared(case: Path) -> tuple[str, str]:
    v = load_task(case)["verify"]
    return v["orientation"], v.get("pose_mode", "expert-fit")


def _dumb_block(case: Path, out: Path) -> Path:
    """The dumb solution of test_examples.py: a solid block of the GT's bbox."""
    from envs.geom import ocp_hashcode_fix
    ocp_hashcode_fix()
    import cadquery as cq
    b = _gt_shape(case).BoundingBox()
    box = cq.Workplane("XY").box(b.xlen, b.ylen, b.zlen).val().translate(
        ((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, (b.zmin + b.zmax) / 2))
    return _export(box, out)


def _chiral():
    """A box with three unequal corner cuts: no mirror plane, no inversion
    centre, so its mirror image is a different part."""
    from envs.geom import ocp_hashcode_fix
    ocp_hashcode_fix()
    import cadquery as cq
    return (cq.Workplane("XY").box(40, 24, 12)
            .cut(cq.Workplane("XY").box(8, 8, 12).translate((16, 8, 0)))
            .cut(cq.Workplane("XY").box(4, 4, 12).translate((18, -10, 0)))
            .cut(cq.Workplane("XY").box(12, 12, 6).translate((-14, 6, 3)))).val()


# ------------------------------------------------------------ declaration ----
def test_declared_not_sniffed():
    """T1 / T3 declare part_v1 in task.toml; the rest stay legacy. The scorer
    dispatches on the declaration, never on the task id."""
    t1, t3 = T.load("t1_drawing2part"), T.load("t3_part2step")
    assert (t1.metric, t1.pose_mode, t1.orientation) == ("part_v1", "iou24_aligned", "free")
    assert (t3.metric, t3.pose_mode, t3.orientation) == ("part_v1", "expert-fit", "pinned")
    # The assembly tasks declare their own headlines (asm_v1 on T2, avg_part x
    # asm_v1 on T4 / T5); their pose_mode is the per-instance part_v1's inside
    # the assembly (free -> iou24_aligned, pinned -> expert-fit). T6 has its own verifier.
    assert T.load("t2_realparts2assembly").metric == "asm_v1"
    for tid in ("t4_parts2assembly", "t5_drawings2assembly"):
        assert T.load(tid).metric == "part_x_asm_v1", tid
    assert T.load("t6_pcb2schematic").metric == "ecad_v2"
    for tid in ("t2_realparts2assembly", "t4_parts2assembly", "t5_drawings2assembly", "t6_pcb2schematic"):
        assert T.load(tid).metric != "part_v1", tid
    assert T.load("t4_parts2assembly").pose_mode == "expert-fit" and T.load("t5_drawings2assembly").pose_mode == "iou24_aligned"
    assert set(T.METRICS) == {"legacy", "asm_v1", "part_v1", "part_x_asm_v1", "ecad_v2"} and set(T.POSE_MODES) == {"expert-fit", "iou24_aligned"}
    # The scope of avg_part's mean over part types is declared the same way:
    # T5 averages the modelled types only (16 of its 21 types are supplied),
    # everything else averages over all of them (docs/METRICS.md).
    assert set(T.AVG_PART_TYPES) == {"all", "modelled"}
    assert T.load("t5_drawings2assembly").avg_part_types == "modelled"
    assert T.load("t2_realparts2assembly").avg_part_types == "all"
    assert T.load("t4_parts2assembly").avg_part_types == "all"


def test_legacy_dispatch_keeps_old_result_shape(tmp_path):
    """A task that does not declare part_v1 gets exactly the legacy dict."""
    from envs.verifiers.part import score
    r = score(SYNTH_T3, SYNTH_T3 / "gt/gt.step", {"verify": {"orientation": "pinned"}})
    assert set(r) == {"iou", "gt_sha256", "gt_hash_source"} and abs(r["iou"] - 1.0) < 1e-4
    with pytest.raises(ValueError):
        score(SYNTH_T3, SYNTH_T3 / "gt/gt.step", {"verify": {"metric": "part_v2"}})


# ----------------------------------------------------------------- oracle ----
@pytest.mark.parametrize("case", PART_CASES, ids=_ids(PART_CASES))
def test_oracle_scores_one(case):
    """(1) gt.step submitted through the declared verifier: part_v1 == 1.0 on
    every term, the legacy `iou` still present and 1.0."""
    r = score_case(case, case / "gt/gt.step")
    assert r["metric"] == "part_v1" and r["coverage"] == 1.0
    assert abs(r["score"] - 1.0) < 1e-3, fmt(r)
    assert abs(r["iou_term"] - 1.0) < 1e-3 and r["surf_f1"] == 1.0 and r["pix_fg"] == 1.0
    assert abs(r["iou"] - 1.0) < 1e-4                      # legacy key, legacy value
    assert r["baseline"] < 1.0 and r["rotation_applied"] is False
    orientation, pose_mode = _declared(case)
    assert r["orientation"] == orientation and r["pose_mode"] == pose_mode
    assert ("iou24" in r) == (orientation == "free") and ("iou_pinned" in r) == (orientation == "pinned")
    assert "part_v1=" in fmt(r)


# ------------------------------------------------------------------- dumb ----
@pytest.mark.parametrize("case", PART_CASES, ids=_ids(PART_CASES))
def test_dumb_scores_less(case, tmp_path):
    """(2) the GT bounding-box block is no better than a bounding primitive:
    iou_term 0 (the 1e-3 rule allows at most ~0.002 for an exact tie with
    the box baseline), and part_v1 well below the oracle's 1.0."""
    orientation, pose_mode = _declared(case)
    r = pm.score_part_v1(case / "gt/gt.step", _dumb_block(case, tmp_path / "dumb.step"),
                         orientation=orientation, pose_mode=pose_mode)
    assert r["coverage"] == 1.0
    assert r["iou_term"] <= 0.01, r
    assert r["score"] < 0.9, r
    assert 0.0 < r["surf_f1"] < 1.0 and 0.0 < r["pix_fg"] < 1.0


# ----------------------------------------------------------------- mirror ----
def test_mirror_is_not_a_rotation(tmp_path):
    """(3) det = +1 only. The mirror image of a chiral part is recovered by
    one of the 24 IMPROPER signed permutations and by none of the 24 proper
    ones -- so the proper search must not give it iou_term 1.0. The test has
    teeth: with the improper half admitted the mirror scores measurably
    higher, and the analytic un-mirroring shows surf_f1 1.0 (it IS the exact
    mirror)."""
    ref = _export(_chiral(), tmp_path / "chiral.step")
    mir = _export(_chiral().mirror("YZ"), tmp_path / "mirror.step")
    same = pm.score_part_v1(ref, ref, orientation="free", pose_mode="expert-fit")
    assert same["iou24"] == 1.0 and same["iou_term"] == 1.0

    r = pm.score_part_v1(ref, mir, orientation="free", pose_mode="expert-fit")
    assert r["iou24"] < 0.9 and r["iou_term"] < 0.5, r

    # The mirror is exact: undo it analytically and the surfaces coincide.
    flip = np.diag([-1.0, 1.0, 1.0])
    rp, mp = pm.surface_points(pm.load_shape(ref)), pm.surface_points(pm.load_shape(mir))
    assert pm.surface_f1(mp @ flip.T, rp)["surf_f1"] == 1.0

    # The improper half would have admitted it: best over det = -1 beats best
    # over det = +1. Recomputed through the voxeliser's own pieces, which
    # shares nothing with score_part_v1's bookkeeping.
    ctx = pm.reference_iou_context(pm.load_shape(ref))
    V, T = pm.tessellate(pm.load_shape(mir), pm.IOU_DEFLECTION)
    idx = pm.world_indices(*pm.occupancy(V, T, frame=pm.mesh_frame(V)))
    best = {1: 0.0, -1: 0.0}
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, j in enumerate(perm):
                M[i, j] = signs[i]
            d = round(float(np.linalg.det(M)))
            g = pm.paste(pm.place(pm.rotate_indices(idx, M), "self")[0])
            best[d] = max(best[d], pm.grid_iou(ctx["gvox"], g))
    assert best[1] == pytest.approx(r["iou24"], abs=1e-9)
    assert best[-1] > best[1] + 0.05, best
    assert len(pm.rotations()) == 24 and all(round(float(np.linalg.det(M))) == 1 for M in pm.rotations())


# ----------------------------------------------------------- quarter turn ----
@pytest.fixture(scope="module")
def quarter_turned(tmp_path_factory):
    """The synthetic T1/T3 reference (one geometry, both tasks) turned 90 deg about Z."""
    out = tmp_path_factory.mktemp("rot") / "gt_rot90z.step"
    return _export(_gt_shape(SYNTH_T1).rotate((0, 0, 0), (0, 0, 1), 90), out)


def test_quarter_turn_free_vs_pinned(quarter_turned):
    """(4) T1 (free) finds the turn: iou24 stays 1.0. T3 (pinned) searches
    nothing: iou_pinned collapses and iou_term is 0. Through the declared
    verifiers, so this also checks the dispatch."""
    t1 = score_case(SYNTH_T1, quarter_turned)
    assert t1["orientation"] == "free" and t1["iou24"] >= 0.99 and t1["iou_term"] >= 0.99, fmt(t1)
    assert t1["iou1"] < 0.5                                    # the delivered pose is wrong
    t3 = score_case(SYNTH_T3, quarter_turned)
    assert t3["orientation"] == "pinned" and "iou24" not in t3 and t3["rotation_search"] is False
    assert t3["iou_pinned"] < 0.5 and t3["iou_term"] == 0.0, fmt(t3)
    assert t3["score"] < t1["score"] - 0.3


def test_pose_mode_iou24_aligned_vs_lab(quarter_turned):
    """The this repository deviation, on and off. In `iou24_aligned` the
    rotation iou24 found is applied before surf_f1 / pix_fg, so a
    quarter-turned oracle scores 1.0 on both; in `expert-fit` (the reference
    behaviour) those two see the delivered pose and do not."""
    gt = SYNTH_T1 / "gt/gt.step"
    aligned = pm.score_part_v1(gt, quarter_turned, orientation="free", pose_mode="iou24_aligned")
    plain = pm.score_part_v1(gt, quarter_turned, orientation="free", pose_mode="expert-fit")
    assert aligned["rotation_applied"] is True and plain["rotation_applied"] is False
    assert aligned["iou24"] == plain["iou24"]                    # the search itself is the same
    assert aligned["surf_f1"] >= 0.999 and aligned["pix_fg"] >= 0.999, aligned
    assert plain["surf_f1"] < 0.7 and plain["pix_fg"] < 0.8, plain
    assert aligned["score"] > 0.99 > 0.8 > plain["score"]
    R = np.array(aligned["rotation"])
    assert round(float(np.linalg.det(R))) == 1 and not np.allclose(R, np.eye(3))
    # A pinned orientation has nothing to align to: the contradiction raises
    # in the scorer, not only in check_tasks (which guards the declaration).
    with pytest.raises(ValueError, match="iou24_aligned"):
        pm.score_part_v1(gt, quarter_turned, orientation="pinned", pose_mode="iou24_aligned")
    with pytest.raises(ValueError):
        pm.score_part_v1(gt, quarter_turned, orientation="free", pose_mode="icp")


# ------------------------------------------------------------- background ----
def test_silhouette_samples_the_corner():
    """(5) A tinted, still visually white, background must not turn the whole
    frame into foreground. Assuming white does exactly that."""
    bg, part = (232, 232, 236), pm.PART_COLOR
    img = np.full((64, 64, 3), bg, dtype=np.uint8)
    img[20:40, 10:50] = part                                 # the part
    img[30, 10:50] = (30, 30, 30)                            # a drawn edge across it
    sil = pm.silhouette(img)
    assert sil.sum() == 20 * 40 - 40                         # part minus the edge line
    assert not sil[0, 0] and not sil[10, 10]
    white_assumed = np.abs(img.astype(np.int16) - 255).max(axis=-1) > 12
    assert white_assumed.all()                               # the failure the corner sample prevents
    # pix_fg on two tinted frames that differ on the part is < 1; identical ones give 1.
    other = img.copy()
    other[20:30, 10:50] = bg
    assert pm.pix_fg(img, other) == pytest.approx(1 - 400 / 760)
    assert pm.pix_fg(img, img.copy()) == 1.0


def test_render_background_is_sampled_not_assumed(tmp_path):
    """The same pair rendered on white and on a tint scores the same pix_fg
    (to a rasterisation hair), with the same silhouette area -- and not 1.0."""
    gt = SYNTH_T1 / "gt/gt.step"
    dumb = _dumb_block(SYNTH_T1, tmp_path / "dumb.step")
    ref_mesh, cand_mesh = pm.render_mesh(pm.load_shape(gt)), pm.render_mesh(pm.load_shape(dumb))
    tint = (232, 232, 236)
    a_w, b_w = pm.render_composite(*ref_mesh), pm.render_composite(*cand_mesh)
    a_t, b_t = pm.render_composite(*ref_mesh, background=tint), pm.render_composite(*cand_mesh, background=tint)
    assert a_w.shape == (524, 524, 3) and tuple(a_t[0, 0]) == tint and tuple(a_w[0, 0]) == (255, 255, 255)
    assert pm.silhouette(a_t).sum() == pytest.approx(pm.silhouette(a_w).sum(), rel=0.01)
    pw, pt = pm.pix_fg(a_w, b_w), pm.pix_fg(a_t, b_t)
    assert pw < 0.95 and pt < 0.95, (pw, pt)
    assert pt == pytest.approx(pw, abs=0.02)


# --------------------------------------------------------------- coverage ----
def test_coverage_renormalises_when_a_term_fails(tmp_path, monkeypatch):
    gt = SYNTH_T1 / "gt/gt.step"
    dumb = _dumb_block(SYNTH_T1, tmp_path / "dumb.step")
    full = pm.score_part_v1(gt, dumb, orientation="free", pose_mode="iou24_aligned")

    def boom(*a, **k):
        raise RuntimeError("no GL context")
    monkeypatch.setattr(pm, "render_composite", boom)
    r = pm.score_part_v1(gt, dumb, orientation="free", pose_mode="iou24_aligned")
    assert "pix_fg" not in r and r["missing"] == {"pix_fg": "RuntimeError: no GL context"}
    assert r["coverage"] == pytest.approx(0.8)
    assert r["score"] == pytest.approx((0.5 * full["iou_term"] + 0.3 * full["surf_f1"]) / 0.8)
    assert r["iou_term"] == full["iou_term"] and r["surf_f1"] == full["surf_f1"]


def test_reference_failure_raises(tmp_path):
    """A broken reference is a broken case, not a score: it raises, it is
    not dropped into `missing` and scored 0 at coverage 0."""
    gt = SYNTH_T1 / "gt/gt.step"
    with pytest.raises(Exception):
        pm.score_part_v1(tmp_path / "nonexistent.step", gt, orientation="free", pose_mode="iou24_aligned")
    empty = tmp_path / "empty.step"
    empty.write_text("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n")
    with pytest.raises(Exception):
        pm.score_part_v1(empty, gt, orientation="free", pose_mode="iou24_aligned")
    # The same candidate against a good reference is fine -- so the raise above
    # came from the reference side.
    r = pm.score_part_v1(gt, empty, orientation="free", pose_mode="iou24_aligned")
    assert r["score"] == 0.0 and r["coverage"] == 1.0 and "unusable" in r["error"]


def test_record_keys_follow_the_arguments(tmp_path):
    """`n_samples` in the record is the argument, not the constant; `fmt`
    prints the record's weights, not literals."""
    gt = SYNTH_T3 / "gt/gt.step"
    r = pm.score_part_v1(gt, gt, orientation="pinned", pose_mode="expert-fit", n_samples=5000)
    assert r["n_samples"] == 5000 and "iou_n_samples" not in r
    r["weights"] = {"iou_term": 0.5, "surf_f1": 0.3, "pix_fg": 0.2}
    line = pm.fmt(r)
    assert "0.50*iou_term" in line and "0.30*surf_f1" in line and "0.20*pix_fg" in line


def test_unreadable_submission_scores_zero(tmp_path):
    bad = tmp_path / "bad.step"
    bad.write_text("not a step file\n")
    r = pm.score_part_v1(SYNTH_T1 / "gt/gt.step", bad, orientation="free", pose_mode="iou24_aligned")
    assert r["score"] == 0.0 and r["coverage"] == 1.0 and "unusable" in r["error"]
    assert r["iou_term"] == r["surf_f1"] == r["pix_fg"] == 0.0


def test_normalise_iou_is_mains_norm_iou():
    """`norm_iou`: clip((x - x0) / (1 - x0), 0, 1), with
    x0 >= 1 -- a reference that IS its own primitive -- answered explicitly
    (1.0 only for a perfect x). change 40 dropped the variant that carried 1e-3 on
    both sides of the quotient to dodge the same division by zero: it moved
    every other score by 1e-3 / (1 - x0) and was not what the other repo
    compares against."""
    assert pm.normalise_iou(1.0, 1.0) == 1.0
    assert pm.normalise_iou(0.999, 1.0) == 0.0          # baseline 1: nothing short of perfect
    assert pm.normalise_iou(1.0, 0.5) == 1.0
    assert pm.normalise_iou(0.5, 0.5) == 0.0            # a tie with the primitive earns nothing
    assert pm.normalise_iou(0.75, 0.5) == pytest.approx(0.5)
    assert pm.normalise_iou(0.2, 0.5) == 0.0
    assert pm.WEIGHTS == {"iou_term": 0.5, "surf_f1": 0.3, "pix_fg": 0.2}     # 2026-09-16
    assert pm.fuse({"iou_term": 1.0, "surf_f1": 1.0, "pix_fg": 1.0}) == (1.0, 1.0)
    assert pm.fuse({"iou_term": 0.5}) == (pytest.approx(0.5), pytest.approx(0.5))
    assert pm.fuse({}) == (0.0, 0.0)


# --------------------------------------------------------------- fixtures ----
@pytest.mark.parametrize("row", FIXTURE_ROWS or [pytest.param(
    None, marks=pytest.mark.skip(reason=f"expert-fit fixtures not on disk at {FIXTURES} "
                                        "(six cand.step / ref.step / expected.json rows)"))],
    ids=[r.name for r in FIXTURE_ROWS] or ["absent"])
def test_expert_fit_fixtures(row):
    """The reference pairs' SURFACE and PIXEL numbers: surf_f1
    (every tau) and pix_fg within 0.01, at the delivered pose, which is what
    were computed at the delivered pose (pose_mode "expert-fit").

    The iou half of these rows is in `test_expert_fit_fixtures_iou_pending_republish`:
    the recorded iou values are checked by the tests below."""
    exp = json.loads((row / "expected.json").read_text())["expected"]
    ref, cand = row / "ref.step", row / "cand.step"
    rp, cp = pm.surface_points(pm.load_shape(ref)), pm.surface_points(pm.load_shape(cand))
    for tau in (0.005, 0.01, 0.02, 0.05):
        assert pm.surface_f1(cp, rp, tau=tau)["surf_f1"] == pytest.approx(exp[f"surf_f1_{tau:g}"], abs=0.01), tau
    a = pm.render_composite(*pm.render_mesh(pm.load_shape(ref)))
    b = pm.render_composite(*pm.render_mesh(pm.load_shape(cand)))
    assert pm.pix_fg(a, b) == pytest.approx(exp["pix_fg"], abs=0.01)
    # And the whole thing through the scorer, in expert-fit mode: the two terms the
    # reference pins, and the fused score from THIS repo's iou term.
    full = pm.score_part_v1(ref, cand, orientation="free", pose_mode="expert-fit")
    assert full["surf_f1"] == pytest.approx(exp["surf_f1_0.02"], abs=0.01)
    assert full["pix_fg"] == pytest.approx(exp["pix_fg"], abs=0.01)
    assert full["score"] == pytest.approx(0.5 * full["iou_term"] + 0.3 * exp["surf_f1_0.02"]
                                          + 0.2 * exp["pix_fg"], abs=0.02)


# The movement of the iou half, measured on this machine when change 40 replaced the
# sampled estimator with the true voxelisation. Recorded here, not asserted:
# the recorded values described the old estimator; the republished set is checked
# the set. Left as an xfail so that the day the republished fixtures land the
# test goes GREEN and says so (strict: an unexpected pass is a failure).
EXPERT_FIT_IOU_MOVED = {                    # row: (expert-fit/old iou24, new iou24, expert-fit/old norm, new norm)
    "row01_pan_head_screw_000035_s20260505_0": (0.1044, 0.1033, 0.0000, 0.0000),
    # row02 moved 0.4919 -> 0.4935 when geom.voxel started snapping vertices
    # to a 2**-20 lattice (a bolt whose whole-millimetre faces sat on cell
    # boundaries), and its norm 0.1328 -> 0.1565 when the primitive baseline
    # went to the cube-touch test (the cylinder floor 0.4159 -> 0.3986).
    "row02_bolt_000037_s20260505_0": (0.5432, 0.4935, 0.1088, 0.1577),
    "row03_rl__hex_nut_squash": (0.4629, 0.6000, 0.0000, 0.0000),
    "row04_circlip_000175_s20260505_1": (0.4450, 0.8875, 0.0000, 0.7453),   # norm 0.7418 -> 0.7453, cube-touch floor
    "row05_t1_part_1553_r1": (0.8122, 0.7777, 0.6369, 0.5900),
    "row06_t1_part_0393_r1": (0.9840, 0.9787, 0.9667, 0.9408),
}


# The six fixtures were republished against the true voxelisation on
# 2026-09-16. With the 2**-20 snap on both sides and
# the cube-touch primitives, all six agree to 0.02.
EXPERT_FIT_UNSNAPPED: set[str] = set()


@pytest.mark.parametrize("row", FIXTURE_ROWS or [pytest.param(
    None, marks=pytest.mark.skip(reason="expert-fit fixtures not on disk"))],
    ids=[r.name for r in FIXTURE_ROWS] or ["absent"])
def test_expert_fit_fixtures_iou_republished(row):
    """The republished iou family (true voxelisation, pad res + 5)
    against ours, to 0.02."""
    if row.name in EXPERT_FIT_UNSNAPPED:
        pytest.xfail("voxelised without the 2**-20 snap")
    exp = json.loads((row / "expected.json").read_text())["expected"]
    r = pm.iou_terms(pm.load_shape(row / "ref.step"), pm.load_shape(row / "cand.step"), search=True)
    assert r["iou24"] == pytest.approx(exp["iou24"], abs=0.02)
    assert r["iou1"] == pytest.approx(exp["iou1"], abs=0.02)
    assert r["baseline"] == pytest.approx(exp["iou_baseline"], abs=0.02)
    assert r["iou_term"] == pytest.approx(exp["iou24_norm"], abs=0.02)


@pytest.mark.parametrize("row", FIXTURE_ROWS or [pytest.param(
    None, marks=pytest.mark.skip(reason="expert-fit fixtures not on disk"))],
    ids=[r.name for r in FIXTURE_ROWS] or ["absent"])
def test_expert_fit_fixtures_iou_is_what_we_recorded(row):
    """The iou numbers are the ones
    change 40 measured and wrote down, to 1e-3, so a later change to the term shows
    up here."""
    want = EXPERT_FIT_IOU_MOVED.get(row.name)
    if want is None:
        pytest.skip(f"no recorded movement for {row.name}")
    r = pm.iou_terms(pm.load_shape(row / "ref.step"), pm.load_shape(row / "cand.step"), search=True)
    assert r["iou24"] == pytest.approx(want[1], abs=1e-3), (row.name, r["iou24"])
    assert r["iou_term"] == pytest.approx(want[3], abs=1e-3), (row.name, r["iou_term"])
    assert r["iou1"] <= r["iou24"] + 1e-12                 # the search can only help
    assert 0.0 <= r["iou_term"] <= 1.0


# ------------------------------------------------------- re-exported oracle ----
@pytest.mark.parametrize("case", PART_CASES, ids=_ids(PART_CASES))
def test_reexported_oracle_floor_is_on_record(case, tmp_path):
    """The realistic oracle: the reference geometry re-exported through
    CadQuery (a model's program never returns the byte-identical file). All
    three terms are stable under re-tessellation. The iou term used not to be
    -- with the sampled estimator a re-export of a thick part measured 0.998
    on the synthetic block and 0.62 on the former examples/t3/example1 (48k
    triangles, re-ordered on export; a real case, out of git since change 31) -- and
    since change 40 it is: the bound below is kept as it was, and it now passes with
    a wide margin (the assertion after it is the one with teeth)."""
    orientation, pose_mode = _declared(case)
    re = _export(_gt_shape(case), tmp_path / "reexport.step")
    assert re.read_bytes() != (case / "gt/gt.step").read_bytes()
    r = pm.score_part_v1(case / "gt/gt.step", re, orientation=orientation, pose_mode=pose_mode)
    assert r["surf_f1"] >= 0.999 and r["pix_fg"] >= 0.999, r
    assert r["iou_term"] >= 0.5 and r["score"] >= 0.8, r
    assert r["iou_term"] >= 0.999 and r["score"] >= 0.999, r        # since change 40


# ------------------------------------------------------------------ noise ----
def test_iou_sampling_noise_is_on_record():
    """WHY the iou term stopped sampling, kept as a measurement where a
    change will be noticed: the estimator it used -- 20,000 area-weighted
    surface samples marked into the grid and filled along Z -- disagrees with
    ITSELF on one mesh at two seeds by more than 0.1, while 200,000 samples
    are stable. Sample count alone would have fixed the seed noise and not the
    fill, which bridged an impeller's blades (+83 % cells) and missed material
    on a split ring (-33 %): opposite directions, so no correction existed.
    The estimator is still in the module for this test and nowhere else."""
    V, Tr = pm.tessellate(_gt_shape(SYNTH_T1), pm.IOU_DEFLECTION)
    o = np.array([-.5, -.5, -.5])

    def vox(n, seed):
        return pm.sampled_voxels(pm._centred(pm.sample_surface(V, Tr, n=n, seed=seed)), o, 1.0)
    noisy = pm.grid_iou(vox(20000, 0), vox(20000, 1))
    stable = pm.grid_iou(vox(200000, 0), vox(200000, 1))
    assert stable >= 0.99, stable
    assert noisy < stable - 0.1, (noisy, stable)
    # And the replacement, on the same mesh: no seed exists to vary, and the
    # occupancy of the same geometry meshed 20x finer is the same cells.
    fine = pm.tessellate(_gt_shape(SYNTH_T1), pm.IOU_DEFLECTION / 20)
    fr = pm.mesh_frame(V)
    a = pm.paste(pm.world_indices(*pm.occupancy(V, Tr, frame=fr)))
    b = pm.paste(pm.world_indices(*pm.occupancy(*fine, frame=fr)))
    assert pm.grid_iou(a, b) == 1.0, pm.grid_iou(a, b)
    assert "seed" not in pm.iou_terms.__doc__


# ------------------------------------------------------ surface identity ----
def _spline_loft():
    """A lofted B-spline solid, 2.5 mm thick and 40 mm long."""
    import cadquery as cq
    pts = [(0, 0), (10, 4), (22, 5), (34, 3), (40, 0), (34, -3), (22, -5), (10, -4)]
    return (cq.Workplane("XY").spline(pts, includeCurrent=False).close()
            .workplane(offset=2.5).spline([(x * 0.7, y * 0.7 + 1.5) for x, y in pts], includeCurrent=False).close()
            .loft(ruled=False))


def test_same_surface_under_another_triangulation_is_the_reference(tmp_path):
    """Identity level 3 (`surface_identity`). The three held-out T3 parts
    whose oracle scored 0.995-0.999 (2026-09-17) came back from the STEP
    round trip with every B-spline edge re-approximated and every face
    re-triangulated: same surface to a few um, vertices 1-3 mm apart, so the
    vertex-identical gate (level 1) did not fire and the triangulation-
    sensitive terms lost 0.5 %. A synthetic loft round-trips bit-exact, so
    the different triangulation is made here by rebuilding the same solid
    from two halves fused back (the faces are split, the mesh is not the
    reference's). It must score exactly 1.0 by the surface gate; a mirror
    and a 2 mm feature change must not pass it."""
    import cadquery as cq
    loft = _spline_loft()
    ref = _export(loft.val(), tmp_path / "loft.step")
    left = loft.intersect(cq.Workplane("XY").box(20, 100, 100, centered=(False, True, True)))
    right = loft.intersect(cq.Workplane("XY").box(100, 100, 100, centered=(False, True, True)).translate((20, 0, 0)))
    same = _export(left.union(right, clean=True).val(), tmp_path / "loft_rebuilt.step")
    r = pm.score_part_v1(ref, same, orientation="pinned", pose_mode="expert-fit")
    assert r["score"] == 1.0 and r["identical"] and r["identical_by"] == "surface", \
        {k: r.get(k) for k in ("score", "identical", "identical_by", "surface_identity")}
    si = r["surface_identity"]
    assert si["distance_p999_mm"] <= si["limit_mm"]

    cut = _export(loft.cut(cq.Workplane("XY").box(2, 2, 2).translate((22, 0, 1))).val(), tmp_path / "loft_cut.step")
    c = pm.score_part_v1(ref, cut, orientation="pinned", pose_mode="expert-fit")
    assert not c["identical"] and c["score"] < 1.0, {k: c.get(k) for k in ("score", "identical", "identical_by")}

    mir = _export(loft.mirror("XZ").val(), tmp_path / "loft_mirror.step")
    m = pm.score_part_v1(ref, mir, orientation="pinned", pose_mode="expert-fit")
    assert not m["identical"], {k: m.get(k) for k in ("score", "identical", "identical_by")}


# ------------------------------------------------ degenerate triangles ----
def test_degenerate_triangles_never_reach_the_surface_distance():
    """OCC's face mesher leaves collinear zero-area slivers along trimmed
    edges (40-120 per T5 held-out reference), and vtkImplicitPolyDataDistance
    dies on them with SIGBUS / SIGSEGV -- not an exception -- and not on every
    sample (2026-09-18: two gpt-6-astra T5 submissions took the scorer down
    twice each, inside the identity gate). They carry no area, so the
    sampler never draws from them: dropping them changes no distance, and a
    mesh with nothing else left is refused before VTK sees it."""
    V = np.array([[x, y, z] for x in (0, 20) for y in (0, 20) for z in (0, 20)], dtype=float)
    T = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    sliver = np.array([[0, 0, 0], [0, 0, 7], [0, 0, 20], [0, 5, 0], [0, 5, 0]], dtype=float)   # collinear, and coincident
    Vd = np.vstack([V, sliver]); n = len(V)
    Td = np.vstack([T, [[n, n + 1, n + 2]] * 30, [[n + 3, n + 4, n + 3]] * 10, [[n, n + 2, n + 1]] * 10])
    kept = pm.surface_triangles(Vd, Td)
    assert len(kept) == 12 and set(map(tuple, kept)) == set(map(tuple, T))
    clean = pm.surface_distance(V, T, V + 0.01, T)
    assert pm.surface_distance(Vd, Td, V + 0.01, T) == pytest.approx(clean)
    assert pm.surface_distance(V + 0.01, T, Vd, Td) == pytest.approx(clean)
    with pytest.raises(ValueError):
        pm.surface_triangles(Vd, Td[12:])                          # nothing but slivers
    with pytest.raises(ValueError):
        pm.surface_triangles(V, np.array([[0, 1, 99]]))            # index out of range
    with pytest.raises(ValueError):
        pm.surface_triangles(np.vstack([V, [[np.nan, 0, 0]]]), np.array([[0, 1, 8]]))
    # ... and the identity gate reports such a mesh, it does not raise or die.
    import cadquery as cq, tempfile
    box = pm.load_shape(_export(cq.Workplane("XY").box(20, 20, 20).val(), Path(tempfile.mkdtemp()) / "box.step"))
    orig = pm.tessellate
    try:
        pm.tessellate = lambda shape, deflection: (Vd, Td[12:])    # every triangle a sliver
        k, rec = pm.surface_identity(box, box, search=False)
    finally:
        pm.tessellate = orig
    assert k is None and "unusable" in rec.get("note", ""), rec
