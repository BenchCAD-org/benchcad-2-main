# T5 - drawing set + assembly drawing -> assembly

`drawing.pdf` (raster `drawing.png`, `drawing_p2.png` ... per page) is the
assembly drawing. `part_drawings/part_NN.pdf` (raster beside it) are the
drawings of the parts to model, in millimetres. `step_files/part_NN.step`
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
