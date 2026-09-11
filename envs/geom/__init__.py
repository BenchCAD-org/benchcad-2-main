"""Geometry primitives: one implementation of everything both kinds need.

The rule: if a function is used by both part and assembly tasks, it lives here; if only
one kind uses it, it stays in that kind's verifier.

⚠️ **No function at this layer gets a default for `res`; the caller is forced to pass it
explicitly.**
Resolution is not "one constant repeated three times"; it is three numbers for three
different purposes, each with its own constraint:
  * whole-object IoU res=64      -- the convention used for cross-comparison with
                                    BenchCAD; change it and the historical results
                                    are void
  * per-instance compare res=128 -- it has to resolve a pin of radius 0.02, which at
                                    64 gets only two or three voxels and is all noise
  * alignment self-check         -- no voxelisation at all; uses centroid residuals
Providing a default is an invitation for someone later to "unify them", and the moment
they do, cross-comparison is silently destroyed.
"""
from .tessellate import ocp_hashcode_fix, tessellate_all          # noqa: F401
from .normalize import normalized_mesh                            # noqa: F401
from .voxel import solid_voxels, surface_voxels, to_dense         # noqa: F401
from .iou import grid_iou, iou_step_vs_step                       # noqa: F401
from .rotate import ROT24, best_rotation                          # noqa: F401
from .primitive import bbox_box, inscribed_cylinders, trivial_floor  # noqa: F401
