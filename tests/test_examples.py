"""End-to-end tests of tests/fixtures/t1..t6 -- touching no benchmark data.

Isolation is the whole point of this file: `envs/*/cases` is a gitignored
symlink to data outside the repo, so any test that needs a real case skips or
errors on a fresh clone -- and in pytest output "skipped" looks too much like
"passed". The cases here are synthetic (boxes and cylinders), live in git,
always run, and carry no benchmark geometry, so heldout and license rules do
not apply.

Two checks per task, on the declared headline (`score` from score_case):
  (1) the full-marks solution scores 1.0   -- a scorer that fails a known-correct answer is broken
  (2) the dumb solution scores LESS        -- a score that cannot separate good from bad carries no information
(2) is the necessary complement of (1): a scorer that always returns 1.0 passes (1).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EX = REPO / "tests/fixtures"
TASKS = sorted(p.name for p in EX.iterdir() if p.is_dir() and (p / "expected.json").exists())


def _run(code: str, cwd: Path) -> Path:
    """Execute a solution in cwd and export `result` to STEP the way the
    sandbox does (a named cq.Assembly is saved with its children, anything
    else goes through exporters.export). Returns the STEP path."""
    out = cwd / "out.step"
    prog = code + (
        "\nimport json as _json\n"
        "if isinstance(result, (dict, list)):\n"
        f"    open({str(out.with_suffix('.json'))!r}, 'w').write(_json.dumps(result))\n"
        "else:\n"
        "    import cadquery as _cq\n"
        "    if hasattr(result, 'save') and hasattr(result, 'children'):\n"
        f"        result.save({str(out)!r}, 'STEP')\n"
        "    else:\n"
        f"        _cq.exporters.export(result, {str(out)!r})\n")
    r = subprocess.run([sys.executable, "-c", prog], cwd=cwd, capture_output=True, text=True)
    if out.with_suffix(".json").exists():
        return out.with_suffix(".json")
    assert out.exists(), f"solution produced no STEP:\n{r.stderr[-1500:]}"
    return out


def _score(case: Path, submission: Path) -> float:
    """Score the way the env does: score_case dispatches on the case's declared
    task.toml, and `score` is the declared headline (part_v1, asm_v1,
    avg_part x asm_v1, the ecad graph metric). Not the raw voxel IoU: a
    scorer whose headline fails a known-correct answer must fail here."""
    sys.path.insert(0, str(REPO))
    from envs.common.score_case import score_case
    r = score_case(case, submission)
    assert "score" in r, f"{case}: no headline `score` in the record ({r.get('metric')})"
    return r["score"]


@pytest.mark.parametrize("task", TASKS)
def test_oracle_scores_one(task, tmp_path):
    """(1) full-marks solution -> 1.0. This tests the ruler, not the task."""
    d = EX / task
    exp = json.loads((d / "expected.json").read_text())
    if exp.get("placeholder"):
        pytest.skip(f"{task} is a placeholder: skeleton only, not defined yet")
    work = tmp_path / task
    work.mkdir()
    (work / "case1").symlink_to(d / "case1", target_is_directory=True)
    step = _run((d / "solution.py").read_text(), work)
    got = _score(d / "case1", step)
    # expected.json's `oracle_iou` is the full-marks value of the declared headline (historical key name).
    assert abs(got - exp["oracle_iou"]) <= exp["tol"], f"{task}: full-marks solution scored only {got:.6f}"


@pytest.mark.parametrize("task", TASKS)
def test_dumb_scores_less(task, tmp_path):
    """(2) the dumb solution (a solid block of the GT bounding box) must score BELOW full marks.

    Without this, a scorer that does `return 1.0` passes every (1) test.
    """
    d = EX / task
    exp = json.loads((d / "expected.json").read_text())
    if exp.get("placeholder"):
        pytest.skip(f"{task} is a placeholder")
    work = tmp_path / task
    work.mkdir()
    sys.path.insert(0, str(REPO))
    if (d / "case1/gt/gt_graph.json").exists():
        # the dumb graph: valid, empty -- the submission template of the ECAD task
        dumb = work / "pred_graph.json"
        dumb.write_text(json.dumps({"schema": "pcb2schematic/1.0", "components": [], "nets": [], "incidences": []}))
        got = _score(d / "case1", dumb)
        assert got < exp["oracle_iou"] - exp["tol"], f"{task}: empty graph scored {got:.6f}"
        return
    from envs.common.score import _ocp_hashcode_fix
    _ocp_hashcode_fix()
    import cadquery as cq
    gt = cq.importers.importStep(str(d / "case1/gt/gt.step")).val()
    b = gt.BoundingBox()
    box = cq.Workplane("XY").box(b.xlen, b.ylen, b.zlen).val().translate(
        ((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, (b.zmin + b.zmax) / 2))
    dumb = work / "dumb.step"
    cq.exporters.export(cq.Workplane(obj=box), str(dumb))
    got = _score(d / "case1", dumb)
    assert got < exp["oracle_iou"] - exp["tol"], \
        f"{task}: trivial floor {got:.6f} is not below full marks -- this case cannot measure anything"
