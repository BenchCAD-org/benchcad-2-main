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
    d = ROOT / "work" / "pytest" / tmp_path.name / "wd"
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


def test_master_is_listed_in_the_prompt():
    """Every model gets the same inputs: the 2x master is a listed file, not
    a name the listing skips and `ls -a` finds (Anthropic's review of the
    public harness, 2026-09-21)."""
    from envs.common.episode import _visible
    assert _visible(Path(HIRES_DIR)) and not HIRES_DIR.startswith("_")


def test_staged_drawings_are_png_only_with_tiles(tmp_path, wd):
    """The sandbox holds one form of every drawing: the sheet, its four tiles
    and the 2x master under hires/. The PDF is rendered and removed."""
    import os
    from envs.common.sandbox import Sandbox, tile_grid
    os.environ["CADENV_LOCAL"] = "1"
    case = ROOT / "tests/fixtures/t1/case1"
    Sandbox(case, wd)
    names = sorted(str(p.relative_to(wd)) for p in wd.rglob("*") if p.is_file())
    assert not any(n.endswith(".pdf") for n in names), names
    assert "drawing.png" in names
    assert not [n for n in names if n.startswith("drawing_tile_")]    # an A4 sheet: no tiles
    assert "hires/drawing.png" in names


def _sheet(tmp_path, w_mm, h_mm, shapes, text=()):
    """A synthetic sheet: a frame, a footer line, black rectangles, and
    numeric text at given em sizes (mm), so the reading area and the
    lettering rule can be exercised without a real drawing."""
    import pymupdf
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "drawing.pdf"
    doc = pymupdf.open(); page = doc.new_page(width=w_mm / 25.4 * 72, height=h_mm / 25.4 * 72)
    R = page.rect
    page.draw_rect(pymupdf.Rect(R.x0 + 5, R.y0 + 5, R.x1 - 5, R.y1 - 5), color=(0, 0, 0), width=1)   # frame
    page.insert_text((20, R.y1 - 8), "BenchCAD  A2  SHEET 1 / 1", fontsize=8)                        # footer
    for x, y in shapes:
        page.draw_rect(pymupdf.Rect(x, y, x + 200, y + 200), fill=(0, 0, 0))
    for x, y, mm, txt in text:
        page.insert_text((x, y), txt, fontsize=mm / 25.4 * 72)
    doc.save(str(pdf)); doc.close()
    return pdf


def test_the_sheet_shown_is_the_reading_area(tmp_path):
    """The frame, the footer and the paper around the content are cropped
    away: an A2 sheet whose only content sits in its top-left quarter is
    shown as that quarter (plus the margin), not as the whole sheet."""
    from envs.common.sandbox import _rasterize_pdf
    pdf = _sheet(tmp_path, 594, 420, [(150, 150)], text=[(160, 400, 4.0, "120.5")])
    files = _rasterize_pdf(pdf)
    with Image.open(files[0]) as sheet:
        # the content spans ~x 150..350 pt, y 150..400 pt of a 1684 x 1191 pt page
        assert sheet.width < sheet.height * 1.2 and sheet.width < 0.4 * 1684 * 300 / 72 * 1.05


def test_tiles_only_when_the_lettering_asks(tmp_path):
    """Large lettering: no tiles, whatever the sheet size. Small lettering
    on a big reading area: the smallest grid that lifts it to TILE_MIN_CAP
    px, tiles rendered with their long edge at TILE_PX."""
    from envs.common.sandbox import TILE_PX, _rasterize_pdf
    big = _sheet(tmp_path / "big", 594, 420, [(150, 150), (1300, 900)],
                 text=[(200, 500, 6.0, "670.9"), (1350, 1100, 6.0, "12")])
    assert not [f for f in _rasterize_pdf(big) if "_tile_" in f.name]
    small = _sheet(tmp_path / "small", 594, 420, [(30, 30), (1450, 950)],          # corner to corner
                   text=[(60, 400, 2.8, "670.9"), (1500, 1100, 2.8, "12")])
    files = _rasterize_pdf(small)
    tiles = [f for f in files if "_tile_" in f.name]
    assert tiles, "2.8 mm lettering across an A2 reading area is 8.6 px after the downscale: tiles"
    with Image.open(tiles[0]) as t:
        assert abs(max(t.size) - TILE_PX) <= 1                      # pymupdf rounds the clip render


def test_blank_tiles_are_skipped_but_content_tiles_kept(tmp_path):
    """Two content corners of an A0 reading area with small lettering: the
    grid covers the area, the blank pieces are not written, the pieces with
    ink are."""
    from envs.common.sandbox import _rasterize_pdf
    pdf = _sheet(tmp_path, 1189, 841, [(150, 150), (2900, 2000)],
                 text=[(200, 500, 2.5, "1189"), (2950, 2300, 2.5, "841")])
    tiles = sorted(f.name for f in _rasterize_pdf(pdf) if "_tile_" in f.name)
    assert tiles and tiles[0] == "drawing_tile_r1c1.png" and len(tiles) < 9, tiles


def test_tile_grid_follows_the_lettering_not_the_paper():
    from envs.common.sandbox import tile_grid
    # Cap heights at the 2048 px edge the strictest provider keeps (OpenAI).
    assert tile_grid(594, 420, 6.2) == (1, 1)          # the A2 assembly sheets: 15 px, no tiles
    assert tile_grid(594, 420, 2.8) == (2, 2)          # an A2 part drawing with 2.8 mm lettering: four (a 1x2 tile shows 9.6 px at 2048)
    assert tile_grid(420, 297, 2.4) == (1, 2)          # A3 at 2.4 mm: 8.2 px, two tiles (10.4 px at 2576 was none)
    assert tile_grid(420, 297, 3.0) == (1, 1)          # A3 at 3.0 mm: 10.2 px, just enough
    assert tile_grid(1189, 841, 2.5) == (3, 4)         # A0 at 2.5 mm: 12 tiles of ~330 mm
    assert tile_grid(210, 297, 2.4) == (1, 1)          # A4: 11.6 px
    assert tile_grid(594, 420, None) == (2, 2)         # outline lettering: taken at 2.5 mm, an A2 tiles (four at 2048)
    assert tile_grid(420, 297, None) == (1, 2)         # ... an A3 at 2.5 mm: 8.5 px at 2048, two


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
    wd = tmp_path / "wd"
    pdf = _sheet(wd, 1189, 841, [(150, 150), (1950, 1100), (2900, 2000)],
                 text=[(200, 500, 2.5, "1189"), (2000, 1400, 2.5, "500"), (2950, 2300, 2.5, "841")])
    files = _rasterize_pdf(pdf); pdf.unlink()
    assert sum("_tile_" in f.name for f in files) >= 3
    seeds = _seed_images(wd)
    assert [p.name for p in seeds] == ["drawing.png"]
