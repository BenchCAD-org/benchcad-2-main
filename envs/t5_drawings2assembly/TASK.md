# T5 - drawing set + assembly drawing -> assembly

`drawing.png` is the assembly drawing (`drawing_p2.png` ... per further
page); `drawing_tile_r<i>c<j>.png` are overlapping pieces of the same sheet
(rows top to bottom, columns left to right) at a resolution where the
lettering is legible; a tile that would be blank paper is not written.
`part_drawings/part_NN.png` are the drawings of the parts to model, in
millimetres, each with its tiles beside it.
`step_files/part_NN.step`
are the purchased parts, each centred on its bounding box and given a random
axis-aligned 90-degree rotation. `bom.json` gives each part's `source`
(`drawing` or `step`) and count; part ids need not match the drawing's item
numbers, and hardware on the drawing that is not in it is not part of the
answer.

Model the drawing parts and place every instance of every part. Submit with
the tools: `tools.export_part(solid, part_id)` for each modelled part,
`tools.use_part(part_id)` for each supplied one, then
`tools.submit_assembly(instances)`, where each instance is
`{"part_id": ..., "transform": T}` and `T` is a 4x4 in millimetres mapping
the part file's frame to the assembly frame. Global position, scale and
axis-aligned orientation are free.
