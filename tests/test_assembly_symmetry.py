"""Regression tests for symmetry-equivalent assembly poses.

Assembly scoring must compare occupied geometry, not raw rotation matrices.  A
rotation that belongs to a part's geometric symmetry group is the same pose for
evaluation purposes; a superficially similar rotation of an asymmetric part is
not.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cadquery as cq

from envs.common.rubric_asm import rubric
from envs.common.score_asm import assembly_score


class AssemblySymmetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="cad-assembly-symmetry-")
        self.tmp = Path(self._tmp.name)
        self.base = cq.Workplane("XY").box(18, 12, 4).translate((0, 0, -4))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _score(self, name: str, gt_part, pred_part) -> tuple[dict, dict]:
        def save(part, path: Path) -> None:
            assembly = cq.Assembly(name="asm")
            assembly.add(self.base, name="base")
            assembly.add(part, name="moving")
            assembly.save(str(path), "STEP")

        gt = self.tmp / f"{name}_gt.step"
        pred = self.tmp / f"{name}_pred.step"
        save(gt_part, gt)
        save(pred_part, pred)
        score = assembly_score(gt, pred)
        return score, rubric(gt, pred, placement=score["hit"])

    @staticmethod
    def _gear(teeth: int = 12, keyed: bool = False):
        shape = cq.Workplane("XY").circle(7).extrude(4)
        tooth = cq.Workplane("XY").box(4, 3, 4).translate((8, 0, 2))
        for k in range(teeth):
            shape = shape.union(
                tooth.rotate((0, 0, 0), (0, 0, 1), 360 * k / teeth)
            )
        if keyed:
            # This small off-axis boss intentionally destroys the 12-fold symmetry.
            shape = shape.union(
                cq.Workplane("XY").box(2, 2, 2).translate((0, 8.8, 5))
            )
        return shape.translate((30, 0, 0))

    def _assert_equivalent(self, score: dict, rb: dict) -> None:
        self.assertGreaterEqual(score["iou_align"], 0.98)
        self.assertEqual(score["hit"], 1.0)
        self.assertGreaterEqual(score["part_iou_mean"], 0.95)
        self.assertEqual(rb["orient"], 1.0)

    def test_axisymmetric_part_accepts_arbitrary_axial_rotation(self) -> None:
        cylinder = cq.Workplane("XY").circle(6).extrude(10).translate((30, 0, 0))
        rotated = cylinder.rotate((30, 0, 0), (30, 0, 1), 37)
        self._assert_equivalent(*self._score("cylinder", cylinder, rotated))

    def test_gear_accepts_one_tooth_pitch(self) -> None:
        gear = self._gear(teeth=12)
        rotated = gear.rotate((30, 0, 0), (30, 0, 1), 30)
        self._assert_equivalent(*self._score("gear", gear, rotated))

    def test_square_part_accepts_quarter_turn(self) -> None:
        square = cq.Workplane("XY").box(10, 10, 4).translate((30, 0, 0))
        rotated = square.rotate((30, 0, 0), (30, 0, 1), 90)
        self._assert_equivalent(*self._score("square", square, rotated))

    def test_asymmetric_key_is_not_treated_as_gear_symmetry(self) -> None:
        keyed = self._gear(teeth=12, keyed=True)
        rotated = keyed.rotate((30, 0, 0), (30, 0, 1), 15)
        score, rb = self._score("keyed_gear", keyed, rotated)
        self.assertLess(score["hit"], 1.0)
        self.assertLess(rb["orient"], 1.0)


if __name__ == "__main__":
    unittest.main()
