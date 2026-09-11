"""Full-marks answer: put the supplied parts back where the reference has them. Must score 1.0."""
import cadquery as cq

# The supplied parts are de-posed (translated to their bbox centre); move them back.
base = cq.Workplane("XY").box(60, 40, 6).val()          # modelled from its drawing
post_1 = cq.importers.importStep("case1/input/step_files/post_1.step").val().translate((-20.0, 0.0, 18.0))
post_2 = cq.importers.importStep("case1/input/step_files/post_2.step").val().translate((20.0, 0.0, 18.0))

result = cq.Assembly(name="asm")
for name, solid in {"base_i1": base, "post_1_i1": post_1, "post_2_i1": post_2}.items():
    result.add(cq.Workplane(obj=solid), name=name)
