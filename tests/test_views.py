"""The T3/T4 reference-view rule: of the four views only the (1,1,1) view is
exact, the other three are rendered from their nominal directions rotated by
a small random angle, the draw is recorded under gt/ (never under input/),
and no renderer goes into the sandbox.

Rendering happens off-screen through VTK, on the synthetic fixtures under
tests/fixtures/t3 and t4."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("CADENV_LOCAL", "1")       # docker is used when present; otherwise stage locally

from envs.common import bench_views as bv  # noqa: E402
from envs.common.caseformat import load_case  # noqa: E402
from envs.common.views import (COMPOSITE_SIZE, VIEWS_JSON, default_seed, perturbation, render_case_views,  # noqa: E402
                               render_part_sheets, sheet_names)

T3 = REPO / "tests/fixtures/t3/case1"
T4 = REPO / "tests/fixtures/t4/case1"
FORBIDDEN = ("bench_views", "views.py", "CAMERA_POSITIONS", "PARALLEL_SCALE")


def _recovered_rotation(cam: dict) -> tuple[float, np.ndarray]:
    """The rigid rotation that takes the nominal camera frame (position, its
    derived view-up, their cross product) to the recorded one, and its angle
    in degrees -- recovered from the dict, not read from `angle_deg`."""
    def frame(pos, up):
        p = np.asarray(pos, float); p /= np.linalg.norm(p)
        u = np.asarray(up, float); u /= np.linalg.norm(u)
        return np.stack([p, u, np.cross(p, u)], axis=1)
    F0 = frame(cam["nominal"], bv.nominal_view_up(cam["nominal"]))
    F1 = frame(cam["position"], cam["view_up"])
    R = F1 @ F0.T
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    return angle, R


def test_perturbation_by_seed():
    p = perturbation(12345)
    assert json.loads(json.dumps(p)) == p                       # JSON-serialisable, round-trips
    assert p["seed"] == 12345 and p["angle_deg_range"] == [3.0, 8.0]
    assert [c["nominal"] for c in p["cameras"]] == [list(map(float, c)) for c in bv.CAMERA_POSITIONS]
    assert perturbation(12345) == p                             # same seed, same draw
    assert perturbation(12346) != p

    exact = p["cameras"][bv.EXACT_CAMERA]
    assert exact["nominal"] == [1.0, 1.0, 1.0]
    assert exact["position"] == [1.0, 1.0, 1.0] and exact["angle_deg"] == 0.0
    a, R = _recovered_rotation(exact)
    assert a < 1e-9 and np.allclose(R, np.eye(3))
    for i, cam in enumerate(p["cameras"]):
        if i == bv.EXACT_CAMERA:
            continue
        a, R = _recovered_rotation(cam)
        assert 3.0 <= a <= 8.0, (i, a)
        assert abs(a - cam["angle_deg"]) < 1e-6
        assert abs(np.linalg.norm(cam["axis"]) - 1.0) < 1e-9
        assert np.allclose(R, bv.rotation_matrix(cam["axis"], cam["angle_deg"]))
        assert abs(np.linalg.norm(cam["position"]) - np.sqrt(3)) < 1e-9        # rigidly rotated, not rescaled
        assert abs(np.dot(cam["position"], cam["view_up"])) < 1e-9             # up stays orthogonal to the view
    angles = [c["angle_deg"] for i, c in enumerate(p["cameras"]) if i != bv.EXACT_CAMERA]
    assert len(set(angles)) == 3                                # three independent draws


def test_seed_rendering_is_deterministic(tmp_path):
    step = T3 / "gt/gt.step"
    a = bv.composite_for_step(step, tmp_path / "a.png", perturb=perturbation(7))
    b = bv.composite_for_step(step, tmp_path / "b.png", perturb=perturbation(7))
    c = bv.composite_for_step(step, tmp_path / "c.png", perturb=perturbation(8))
    n = bv.composite_for_step(step, tmp_path / "n.png")         # nominal cameras
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()
    from PIL import Image
    A, N = (np.asarray(Image.open(p).convert("RGB")) for p in (a, n))
    assert A.shape == N.shape == (524, 524, 3)
    s, g = 256, 4
    quad = [(g, g), (2 * g + s, g), (g, 2 * g + s), (2 * g + s, 2 * g + s)]
    same = [np.array_equal(A[y:y + s, x:x + s], N[y:y + s, x:x + s]) for x, y in quad]
    assert same == [False, True, False, False]                  # only the top-right (1,1,1) view is the nominal render


def test_fixtures_record_the_draw_under_gt_only():
    for case in (T3, T4):
        m = load_case(case).manifest
        assert VIEWS_JSON in [e["path"] for e in m["gt"]], case
        assert not any(e["path"].endswith("views.json") for e in m["input"]), case
        rec = json.loads((case / VIEWS_JSON).read_text())
        seed = m["generator"]["views"]["seed"]
        assert m["generator"]["views"]["tool"] == "tools/render_views.py"
        assert seed == default_seed(m["env"], m["id"])
        assert rec == perturbation(seed)
    from PIL import Image
    assert Image.open(T3 / "input/views.png").size == (1412, 1412)
    assert Image.open(T4 / "input/views.png").size == (1412, 1412)


def _t4_sheets() -> list[str]:
    """Both sheets of every part type of the T4 fixture's BOM, relative to input/."""
    bom = json.loads((T4 / "input/bom.json").read_text())
    return sheet_names(it["part_id"] for it in bom["items"])


def test_t4_ships_one_sheet_pair_per_part_type():
    """One 2x2 sheet pair per BOM part type, each the size of views.png, and
    no strip: the old parts_views.png put 1 + 2n rows of four views in one
    image (1630 x 4960 px for seven types) that the API's long-edge cap
    shrank past legibility. The fixture is what a case looks like."""
    from PIL import Image
    m = load_case(T4).manifest
    listed = [e["path"] for e in m["input"]]
    sheets = _t4_sheets()
    assert len(sheets) == 2 * len(m["parts"]) == 6
    for rel in sheets:
        assert f"input/{rel}" in listed, rel
        assert Image.open(T4 / "input" / rel).size == (1412, 1412), rel
    assert sorted(listed) == sorted(["input/bom.json", "input/views.png"] + [f"input/{r}" for r in sheets])
    assert not (T4 / "input/parts_views.png").exists()
    # the layout is views.png's: white gutters of the 2x2 composite at the same places
    import numpy as np
    for rel in ["views.png"] + sheets:
        a = np.asarray(Image.open(T4 / "input" / rel).convert("RGB"))
        s, g = COMPOSITE_SIZE, 4                                    # gutters: 0..3, s+4..s+7, 2s+8..2s+11
        for y in (0, g - 1, s + g, s + 2 * g - 1, 2 * s + 2 * g, 2 * s + 3 * g - 1):
            assert (a[y] == 255).all(), (rel, y)
        for x in (0, g - 1, s + g, s + 2 * g - 1, 2 * s + 2 * g, 2 * s + 3 * g - 1):
            assert (a[:, x] == 255).all(), (rel, x)


def test_a_part_sheet_shows_the_part():
    """`_alone` fills the frame with the part on its own scale; `_in_assembly`
    shows the same part red at assembly scale with the rest ghosted. The
    fixture's `base` is a 60 x 40 x 6 plate and a post is a d10 x 30 rod, so
    the alone sheets have the same footprint in the frame while in the
    assembly the post is a small red thing and the plate a large one."""
    import numpy as np
    from PIL import Image

    def ink(rel, red=False):
        a = np.asarray(Image.open(T4 / "input" / rel).convert("RGB")).astype(int)
        if red:                                                         # the highlight, lit or in shade
            return ((a[:, :, 0] > 60) & (a[:, :, 0] > 2 * a[:, :, 1]) & (a[:, :, 0] > 2 * a[:, :, 2])).mean()
        return (a.sum(axis=2) < 3 * 245).mean()                         # anything drawn on the white

    # measured: 0.142 / 0.098 alone, 0.133 / 0.022 red in the assembly, 0.17 ink there
    alone_base, alone_post = ink("parts/base_alone.png"), ink("parts/post_1_alone.png")
    assert alone_base > 0.10 and alone_post > 0.05, (alone_base, alone_post)
    red_base, red_post = ink("parts/base_in_assembly.png", red=True), ink("parts/post_1_in_assembly.png", red=True)
    assert red_base > 4 * red_post > 0.04, (red_base, red_post)
    # the ghosted rest is there: the in-assembly sheet draws far more than its red part
    assert ink("parts/post_1_in_assembly.png") > 3 * red_post
    # and both sheets of a type are different pictures
    assert (T4 / "input/parts/base_alone.png").read_bytes() != (T4 / "input/parts/base_in_assembly.png").read_bytes()


def test_part_sheets_use_the_recorded_cameras(tmp_path):
    """The sheets are drawn with the case's four cameras, never a re-drawn
    seed: two perturbations share the exact (1,1,1) view and differ in the
    other three, quadrant by quadrant, on every sheet."""
    import numpy as np
    from PIL import Image
    a = render_part_sheets(T4, perturbation(7), tmp_path / "a")
    b = render_part_sheets(T4, perturbation(8), tmp_path / "b")
    c = render_part_sheets(T4, perturbation(7), tmp_path / "c")
    assert sorted(a) == sorted(b) == sorted(c) == sorted(_t4_sheets())
    s, g = COMPOSITE_SIZE, 4
    quad = [(g, g), (2 * g + s, g), (g, 2 * g + s), (2 * g + s, 2 * g + s)]
    for rel in a:
        A, B, C = (np.asarray(Image.open(d[rel]).convert("RGB")) for d in (a, b, c))
        assert np.array_equal(A, C), rel                                # same seed, same picture
        same = [np.array_equal(A[y:y + s, x:x + s], B[y:y + s, x:x + s]) for x, y in quad]
        assert same == [False, True, False, False], (rel, same)


def test_a_stale_sheet_is_replaced(tmp_path):
    """input/parts/ is a function of gt/ and the seed: a sheet for a type the
    case no longer has does not survive a re-render."""
    import shutil
    d = tmp_path / "case1"
    shutil.copytree(T4, d)
    stale = d / "input/parts/gone_alone.png"
    stale.write_bytes((d / "input/parts/base_alone.png").read_bytes())
    render_case_views(d, load_case(T4).manifest["generator"]["views"]["seed"])
    assert not stale.exists()
    assert sorted(p.name for p in (d / "input/parts").iterdir()) == sorted(Path(r).name for r in _t4_sheets())


def test_render_case_views_reproduces_the_fixture(tmp_path):
    """The recorded seed re-renders the fixture's cameras exactly and its images
    to within rasteriser noise.

    The cameras are the case definition, so `gt/views.json` must match to the
    last digit -- it is JSON and portable. The PNG bytes are NOT: the renderer
    is deterministic on one machine (two fresh renders here are byte-identical)
    but VTK and libpng differ across environments, and the committed fixture
    was produced in another one -- measured, 12,685 vs 12,691 bytes for the
    same picture. A byte assertion on a render therefore tests the build, not
    the code, so this compares the pixels with a tolerance instead.
    """
    import shutil
    d = tmp_path / "case1"
    shutil.copytree(T4, d)
    seed = load_case(T4).manifest["generator"]["views"]["seed"]
    p = render_case_views(d, seed)
    assert p == json.loads((T4 / VIEWS_JSON).read_text())
    assert (d / VIEWS_JSON).read_bytes() == (T4 / VIEWS_JSON).read_bytes()      # the camera record is untouched
    assert not (d / "input/parts_views.png").exists()
    from PIL import Image
    import numpy as np
    for n in ["views.png"] + _t4_sheets():
        got = Image.open(d / "input" / n).convert("RGB")
        want = Image.open(T4 / "input" / n).convert("RGB")
        assert got.size == want.size, n
        diff = np.abs(np.asarray(got, float) - np.asarray(want, float))
        # A one-environment difference is a handful of edge pixels; a wrong
        # camera, a wrong part or a wrong colour moves whole regions.
        assert diff.mean() < 1.0, f"{n}: mean |diff| {diff.mean():.3f}"
        assert (diff.max(axis=2) > 24).mean() < 0.01, (
            f"{n}: {(diff.max(axis=2) > 24).mean():.4f} of pixels differ visibly")


def _docker() -> bool:
    from envs.common.sandbox import _docker_ready
    try:
        return bool(_docker_ready())
    except Exception:                                   # noqa: BLE001
        return False


@pytest.mark.parametrize("case", [T3, T4], ids=["t3", "t4"])
def test_no_renderer_reaches_the_sandbox(case, tmp_path):
    from envs.common.sandbox import TOOLS_PY, Sandbox
    for s in FORBIDDEN:
        assert s not in TOOLS_PY, s
    # docker can only mount paths under $HOME (see Sandbox._mount_works)
    root = Path.home() / "cad-agent-work" / "dryrun" if _docker() else tmp_path
    root.mkdir(parents=True, exist_ok=True)
    wd = Path(tempfile.mkdtemp(prefix="views_", dir=root))
    Sandbox(case, wd)
    staged = sorted(p for p in wd.rglob("*") if p.is_file())
    names = [str(p.relative_to(wd)) for p in staged]
    assert "views.png" in names
    if case is T4:
        # the per-part sheets are staged, under parts/, and nothing else is
        assert [n for n in names if n.startswith("parts/")] == _t4_sheets(), names
        assert "parts_views.png" not in names
    assert not any(n.startswith("gt") or n == "case.json" or n.endswith("views.json") for n in names), names
    for p in staged:
        assert not any(s in p.name for s in ("bench_views", "views.py")), p
        if p.suffix in (".py", ".json", ".md", ".txt", ".toml"):
            text = p.read_text(errors="replace")
            for s in FORBIDDEN:
                assert s not in text, (p, s)


@pytest.mark.parametrize("case", [T3, T4], ids=["t3", "t4"])
def test_prompt_says_three_views_are_perturbed(case):
    task = (REPO / "envs" / load_case(case).manifest["env"] / "TASK.md").read_text()
    assert "exactly ( 1,  1,  1)" in task
    assert "rotated" in task and "3 to 8 degrees" in task
