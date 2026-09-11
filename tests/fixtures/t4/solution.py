"""Full-marks answer: model every part and place it where the reference has it. Must score 1.0."""
import cadquery as cq

# Nothing is supplied: every part is built here, already at its place in the assembly.
base = cq.Workplane("XY").box(60, 40, 6).val()          # modelled from the reference views
post_1 = cq.Workplane("XY").circle(5).extrude(30).translate((-20, 0, 3)).val()          # modelled from the reference views
post_2 = cq.Workplane("XY").circle(5).extrude(30).translate((20, 0, 3)).val()          # modelled from the reference views

result = cq.Assembly(name="asm")
for name, solid in {"base_i1": base, "post_1_i1": post_1, "post_2_i1": post_2}.items():
    result.add(cq.Workplane(obj=solid), name=name)
