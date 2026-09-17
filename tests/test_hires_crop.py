"""tools.crop cuts from the 600 dpi master a drawing sheet is staged with.

The sheet the model is shown is 300 dpi (and the API downscales it further);
a crop of that cannot add detail. Staging renders every PDF page twice, and
crop maps a box in the sheet's pixels onto the 2x master.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.common.sandbox import HIRES_DIR, HIRES_FACTOR, TOOLS_PY, _rasterize_pdf  # noqa: E402

PDF = ROOT / "tests/fixtures/t1/case1/input/drawing.pdf"



@pytest.fixture
def wd(tmp_path):
    """A work dir under $HOME. With the sandbox image present, Sandbox mounts
    the directory into docker, and the VM shares only $HOME: a tmp_path under
    /private/var mounts as an empty directory and Sandbox refuses it."""
    import shutil
    d = Path.home() / "cad-agent-work" / "pytest" / tmp_path.name / "wd"
    d.parent.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d.parent, ignore_errors=True)

def test_master_is_exactly_twice_the_sheet(tmp_path):
    pdf = tmp_path / "drawing.pdf"
    shutil.copy(PDF, pdf)
    sheet = _rasterize_pdf(pdf)[0]                 # the sheet, then its four tiles
    master = tmp_path / HIRES_DIR / "drawing.png"
    assert master.exists()
    with Image.open(sheet) as a, Image.open(master) as b:
        assert (b.width, b.height) == (a.width * HIRES_FACTOR, a.height * HIRES_FACTOR)


def test_crop_reads_the_master_in_sheet_coordinates(tmp_path):
    pdf = tmp_path / "drawing.pdf"
    shutil.copy(PDF, pdf)
    sheet = _rasterize_pdf(pdf)[0]
    (tmp_path / "tools.py").write_text(TOOLS_PY)
    box = (100, 80, 400, 260)                       # in the sheet's pixels
    r = subprocess.run([sys.executable, "-c",
                        f"import tools; print(tools.crop({sheet.name!r}, {box!r}))"],
                       cwd=tmp_path, capture_output=True, text=True, check=True)
    out = tmp_path / r.stdout.strip()
    with Image.open(out) as im:
        assert im.size == ((box[2] - box[0]) * HIRES_FACTOR, (box[3] - box[1]) * HIRES_FACTOR)


def test_crop_without_a_master_is_unchanged(tmp_path):
    Image.new("RGB", (300, 200), "white").save(tmp_path / "views.png")
    (tmp_path / "tools.py").write_text(TOOLS_PY)
    r = subprocess.run([sys.executable, "-c",
                        "import tools; print(tools.crop('views.png', (10, 20, 110, 70)))"],
                       cwd=tmp_path, capture_output=True, text=True, check=True)
    with Image.open(tmp_path / r.stdout.strip()) as im:
        assert im.size == (100, 50)


def test_master_is_hidden_from_the_prompt():
    from envs.common.episode import _visible
    assert not _visible(Path(HIRES_DIR))


def test_staged_drawings_are_png_only_with_tiles(tmp_path, wd):
    """The sandbox holds one form of every drawing: the sheet, its four tiles
    and the hidden master. The PDF is rendered and removed."""
    import os
    from envs.common.sandbox import Sandbox, tile_grid
    os.environ["CADENV_LOCAL"] = "1"
    case = ROOT / "tests/fixtures/t1/case1"
    Sandbox(case, wd)
    names = sorted(str(p.relative_to(wd)) for p in wd.rglob("*") if p.is_file())
    assert not any(n.endswith(".pdf") for n in names), names
    assert "drawing.png" in names
    assert tile_grid(297, 210) == (1, 1)                              # the fixture is an A4 sheet: no tiles
    assert not [n for n in names if n.startswith("drawing_tile_")]
    assert "_hires/drawing.png" in names


def test_tiles_of_a_large_sheet(tmp_path):
    """An A0 sheet (a T2 assembly drawing) is 3 x 4 overlapping tiles, each
    rendered with its long edge at TILE_PX."""
    from envs.common.sandbox import TILE_PX, _rasterize_pdf
    import pymupdf
    pdf = tmp_path / "drawing.pdf"
    doc = pymupdf.open(); page = doc.new_page(width=1189 / 25.4 * 72, height=841 / 25.4 * 72)
    page.draw_rect(pymupdf.Rect(150, 150, 400, 400), fill=(0, 0, 0)); doc.save(str(pdf)); doc.close()
    files = _rasterize_pdf(pdf)
    tiles = [f for f in files if "_tile_" in f.name]
    # one drawn shape at the top-left: only r1c1 has ink inside the margin;
    # the other eleven are blank paper and are not written
    assert [t.name for t in tiles] == ["drawing_tile_r1c1.png"]
    with Image.open(tiles[0]) as t, Image.open(files[0]) as sheet:
        assert max(t.size) == TILE_PX
        assert t.width / sheet.width > 1.0 / 4                          # more than a bare quarter: the overlap


def test_blank_tiles_are_skipped_but_content_tiles_kept(tmp_path):
    from envs.common.sandbox import _rasterize_pdf
    import pymupdf
    pdf = tmp_path / "drawing.pdf"
    doc = pymupdf.open(); page = doc.new_page(width=1189 / 25.4 * 72, height=841 / 25.4 * 72)
    for x, y in ((150, 150), (1950, 1100), (2900, 2000)):               # r1c1, r2c3, r3c4 in page points
        page.draw_rect(pymupdf.Rect(x, y, x + 200, y + 200), fill=(0, 0, 0))
    doc.save(str(pdf)); doc.close()
    tiles = sorted(f.name for f in _rasterize_pdf(pdf) if "_tile_" in f.name)
    assert tiles == ["drawing_tile_r1c1.png", "drawing_tile_r2c3.png", "drawing_tile_r3c4.png"], tiles


def test_tile_grid_follows_the_sheet_size():
    from envs.common.sandbox import tile_grid
    assert tile_grid(420, 297) == (1, 2)          # A3
    assert tile_grid(594, 420) == (2, 2)          # A2
    assert tile_grid(1189, 841) == (3, 4)         # A0
    assert tile_grid(210, 297) == (1, 1)          # A4: no tiles


def test_staged_bom_names_the_png_not_the_pdf(tmp_path, wd):
    """bom.json in the sandbox points at what is there: the drawing part's
    PNG, not the PDF the case stores (T5)."""
    import json, os
    from envs.common.sandbox import Sandbox
    os.environ["CADENV_LOCAL"] = "1"
    case = ROOT / "tests/fixtures/t5/case1"
    Sandbox(case, wd)
    files = [it["file"] for it in json.loads((wd / "bom.json").read_text())["items"]]
    assert files and not any(f.endswith(".pdf") for f in files), files
    for f in files:
        assert (wd / f).exists(), f
    # the case's own bom.json is untouched
    assert any(it["file"].endswith(".pdf") for it in json.loads((case / "input/bom.json").read_text())["items"])


def test_staged_bom_carries_the_parts_list_mapping(tmp_path, wd):
    """bom.json in the sandbox says how the drawing's parts list maps to the
    ids (case.json parts_list.note) -- it differs per case."""
    import json, os
    from envs.common.sandbox import Sandbox
    os.environ["CADENV_LOCAL"] = "1"
    case = ROOT / "examples/task2/cases/case1"
    if not case.exists():
        return
    Sandbox(case, wd)
    staged = json.loads((wd / "bom.json").read_text())
    pl = json.loads((case / "case.json").read_text())["parts_list"]
    assert staged["parts_list"].startswith(pl["note"])
    # the declared, sheet-validated table becomes each part's `item`
    assert {it["part_id"]: it["item"] for it in staged["items"]} == {v: int(k) for k, v in pl["table"].items()}


def test_tiles_are_on_disk_but_not_seeds(tmp_path):
    """A round carries one image per sheet: the tiles stay in the working
    directory (shown when the model writes or copies one at the top level,
    like any PNG) and are not prompt images (a T2 case seeded 13, a T5 case
    20, every round)."""
    from envs.common.episode import _seed_images
    from envs.common.sandbox import _rasterize_pdf
    import pymupdf
    wd = tmp_path / "wd"; wd.mkdir()
    pdf = wd / "drawing.pdf"
    doc = pymupdf.open(); page = doc.new_page(width=1189 / 25.4 * 72, height=841 / 25.4 * 72)
    for x, y in ((150, 150), (1950, 1100), (2900, 2000)):
        page.draw_rect(pymupdf.Rect(x, y, x + 200, y + 200), fill=(0, 0, 0))
    doc.save(str(pdf)); doc.close()
    files = _rasterize_pdf(pdf); pdf.unlink()
    assert sum("_tile_" in f.name for f in files) == 3
    seeds = _seed_images(wd)
    assert [p.name for p in seeds] == ["drawing.png"]
