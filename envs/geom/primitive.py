"""The trivial baseline: how much "doing nothing" scores.

The floor is **an upper bound on the free score**, so the baseline has to be "the
strongest mindless answer constructible from the task inputs alone", not the laziest one.
Both conditions are necessary:
  * strongest -- a lazy baseline underestimates the free score. The first version of the
    ecad floor took "the first legal solution in catalogue order", which ended up
    measuring "picked the wrong one" rather than "did not do the thing that had to be
    done".
  * constructible -- GT's convex hull is a stronger mindless answer (measured 0.11 higher
    than the box), but **it cannot be computed without GT's 3-D outline first**, which is
    using the answer as the baseline. A box and a cylinder can be built by reading three
    dimensions off the drawing; a convex hull cannot.

Measured: on a corpus dominated by plate-like parts the cylinder earns almost nothing
(of 546 tasks the cylinder fits better than the box in only 3, lifting the mean by just
0.0014), but on a corpus with many parts of revolution it is 111/387 with a mean of
+0.0767 -- **the gain is corpus dependent, so keeping the cylinder is what makes the two
comparable.**
"""
from __future__ import annotations

import tempfile
from pathlib import Path


def bbox_box(step: Path, out: Path):
    import cadquery as cq
    b = cq.importers.importStep(str(step)).val().BoundingBox()
    cq.exporters.export(cq.Workplane().box(b.xlen, b.ylen, b.zlen), str(out))
    return out


def inscribed_cylinders(step: Path, out_dir: Path):
    """One inscribed cylinder along each of the three principal axes, its diameter
    taken as the smaller of the other two axes."""
    import cadquery as cq
    b = cq.importers.importStep(str(step)).val().BoundingBox()
    dims = [b.xlen, b.ylen, b.zlen]
    outs = []
    for ax in range(3):
        h = dims[ax]
        r = min(dims[(ax + 1) % 3], dims[(ax + 2) % 3]) / 2.0
        if r <= 0 or h <= 0:
            continue
        p = Path(out_dir) / f"cyl{ax}.step"
        cq.exporters.export(cq.Workplane(("XY", "YZ", "XZ")[ax]).circle(r).extrude(h), str(p))
        outs.append(p)
    return outs


def trivial_floor(gt: Path, res: int) -> dict:
    """max(box, best cylinder). `res` must be passed explicitly."""
    from .iou import iou_step_vs_step
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        f_box = iou_step_vs_step(gt, bbox_box(gt, td / "box.step"), res)
        f_cyl = max((iou_step_vs_step(gt, c, res) for c in inscribed_cylinders(gt, td)),
                    default=0.0)
    return {"floor": max(f_box, f_cyl), "floor_box": f_box, "floor_cyl": f_cyl}
