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

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.common.sandbox import HIRES_DIR, HIRES_FACTOR, TOOLS_PY, _rasterize_pdf  # noqa: E402

PDF = ROOT / "tests/fixtures/t1/case1/input/drawing.pdf"


def test_master_is_exactly_twice_the_sheet(tmp_path):
    pdf = tmp_path / "drawing.pdf"
    shutil.copy(PDF, pdf)
    (sheet,) = _rasterize_pdf(pdf)
    master = tmp_path / HIRES_DIR / "drawing.png"
    assert master.exists()
    with Image.open(sheet) as a, Image.open(master) as b:
        assert (b.width, b.height) == (a.width * HIRES_FACTOR, a.height * HIRES_FACTOR)


def test_crop_reads_the_master_in_sheet_coordinates(tmp_path):
    pdf = tmp_path / "drawing.pdf"
    shutil.copy(PDF, pdf)
    (sheet,) = _rasterize_pdf(pdf)
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
