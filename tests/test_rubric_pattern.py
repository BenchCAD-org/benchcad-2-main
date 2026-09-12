"""Regression tests for duplicate-part arrangement and the penalty for missing parts --
symmetric equivalents must score full marks, while wrong placements and missing parts
must lose points.

The counterpart of tests/test_assembly_symmetry.py: that file covers "a part's **own**
symmetry group must not be penalised", this one covers the opposite direction, "several
instances of the same part placed in the wrong spots must not get away with it", plus
"a missing part must be charged against GT's count, not against the count that is
there".

Both holes were found by measurement (ASM-01, 5 duplicate part types / 18 instances):
* orient used to take "the most similar one of the same type" within a group, so all 3
  instances of one part type placed in the same single orientation counted as three
  correct ones -- the collapsed answer scored a perfect 1.00 on orientation. Changed to
  a one-to-one Hungarian match within the group.
* When tallying multisets by type pair, collapsing only pushed contact / relative
  distance from 1.00 down to 0.77 / 0.79, so a physically absurd answer still totalled
  0.839. layout now compares the pairwise distance spectrum between instances of the
  same part type directly, which is all zeros after collapsing.

⚠️ The denominator is always max(#GT instances, #predicted instances), never "the ones
that matched" -- for an answer missing one column, every item except the BOM must be
charged against GT's 4 columns; counting against the 3 that are there makes the missing
one free.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cadquery as cq

from envs.common.rubric_asm import rubric


class RubricPatternTest(unittest.TestCase):
    """One base plate + 4 columns spaced evenly around a circle: the most typical
    array."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="cad-rubric-pattern-")
        self.tmp = Path(self._tmp.name)
        self.plate = cq.Workplane("XY").box(80, 80, 6).translate((0, 0, -3)).val()
        self.post = cq.Workplane("XY").circle(4).extrude(20).val()
        self.spots = [(24, 0), (0, 24), (-24, 0), (0, -24)]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _save(self, spots, name: str, plate: bool = True) -> Path:
        asm = cq.Assembly(name=name)
        if plate:
            asm.add(self.plate, name="plate")
        for k, (x, y) in enumerate(spots):
            asm.add(self.post.translate((x, y, 0)), name=f"post_{k}")
        p = self.tmp / f"{name}.step"
        (asm.export if hasattr(asm, "export") else asm.save)(str(p), "STEP")
        return p

    def test_perfect_is_full_marks(self) -> None:
        gt = self._save(self.spots, "gt")
        r = rubric(gt, self._save(self.spots, "same"))
        self.assertGreaterEqual(r["layout"], 0.999)
        self.assertGreaterEqual(r["total"], 0.99)

    def test_permuting_identical_instances_is_equivalent(self) -> None:
        """Two instances of the same part swap positions -- the geometry is identical,
        so it must still be full marks. The fix must not hit symmetry by mistake."""
        gt = self._save(self.spots, "gt2")
        shuffled = self.spots[2:] + self.spots[:2]
        r = rubric(gt, self._save(shuffled, "perm"))
        self.assertGreaterEqual(r["layout"], 0.999)
        self.assertGreaterEqual(r["total"], 0.99)

    def test_rotating_a_symmetric_part_about_its_axis_is_free(self) -> None:
        """A column is a body of revolution, so rotating it 30 degrees about its own
        axis is geometrically exactly equivalent -- orientation must not lose a single
        point.

        ⚠️ This became a hard requirement only once the item was made continuous. The old
        version cut at IoU >= 0.5, so the threshold hid the noise; using a hard IoU as
        the score directly, a cylinder rotated 30 degrees gives only 0.852 and rotated 90
        degrees only 0.733. What is compared now is surface similarity with a **1-voxel
        tolerance**, which is back to 1.000.
        """
        gt = self._save(self.spots, "gt_sym")
        asm = cq.Assembly(name="spun")
        asm.add(self.plate, name="plate")
        for k, (x, y) in enumerate(self.spots):
            spun = self.post.rotate((0, 0, 0), (0, 0, 1), 30.0)
            asm.add(spun.translate((x, y, 0)), name=f"post_{k}")
        p = self.tmp / "spun.step"
        (asm.export if hasattr(asm, "export") else asm.save)(str(p), "STEP")
        r = rubric(gt, p)
        self.assertGreaterEqual(r["orient"], 0.99)
        self.assertGreaterEqual(r["total"], 0.99)

    def test_collapsed_array_loses_layout(self) -> None:
        """All 4 columns piled onto the same position -- the count is right, the
        orientations are right, but the arrangement is wrong."""
        gt = self._save(self.spots, "gt3")
        r = rubric(gt, self._save([self.spots[0]] * 4, "collapse"))
        self.assertGreaterEqual(r["bom"], 0.999)   # the count is still correct
        self.assertLess(r["layout"], 0.45)         # every same-type pairwise distance collapses to 0
        self.assertLess(r["total"], 0.8)

    def test_shrunk_array_loses_layout(self) -> None:
        """The array radius halved -- subtler than collapsing, since count, orientation
        and contact may all still be right."""
        gt = self._save(self.spots, "gt4")
        small = [(x / 2, y / 2) for x, y in self.spots]
        r = rubric(gt, self._save(small, "shrunk"))
        self.assertLess(r["layout"], 1.0)

    def test_missing_part_is_charged_against_gt_count(self) -> None:
        """One column missing -- every item except the BOM must be charged against
        **GT's 4 columns**, not against the 3 that are there.

        Counted against the count that is there, the remaining 3 being all correct would
        be full marks and the missing one would be free.
        Taking the denominator as max(#GT instances, #predicted instances) is what closes
        that hole.
        """
        gt = self._save(self.spots, "gt6")
        r = rubric(gt, self._save(self.spots[:3], "missing"))
        self.assertLess(r["bom"], 1.0)
        self.assertLessEqual(r["orient"], 0.85)    # capped at 4/5, so full marks are impossible
        self.assertLess(r["layout"], 0.75)         # a whole batch is missing from the distance spectrum

    def test_edge_items_not_applicable_for_single_instance(self) -> None:
        """With only one instance there are no "edges" to speak of, so fit/layout are
        not applicable and renormalising the weights loses no points."""
        gt = self._save([], "gt5")                 # the base plate only
        r = rubric(gt, self._save([], "one"))
        self.assertIsNone(r["layout"])
        self.assertIsNone(r["fit"])
        self.assertGreaterEqual(r["total"], 0.99)


if __name__ == "__main__":
    unittest.main()
