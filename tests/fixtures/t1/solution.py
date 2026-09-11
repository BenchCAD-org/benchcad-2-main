"""Full-marks answer: rebuild the reference step by step. Must score 1.0."""
import cadquery as cq

result = (cq.Workplane("XY").box(40, 24, 12)
          .faces(">Z").workplane().hole(8)
          .faces(">Z").workplane().circle(6).extrude(6))
