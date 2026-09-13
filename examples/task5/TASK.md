# T5 - drawing set + assembly drawing -> assembly

`drawing.png` is the assembly drawing (`drawing_p2.png` ... per further page).
Every sheet is shown to you downscaled -- layout clear, lettering not -- and
is also cut into tiles, `<sheet>_tile_r<i>c<j>.png` (row i from the top,
column j from the left, neighbours overlapping by 10 %), each at a resolution
where every dimension is legible; a tile that would be blank paper is not
written. Read the numbers on the tiles; use the sheet for where things are and
for `tools.crop` coordinates. `part_drawings/part_NN.png` are the drawings of
the parts to model, in millimetres, each with its tiles beside it, first-angle
projection unless the sheet's title block says third-angle.
`step_files/part_NN.step` are the supplied parts, each centred on its bounding
box and given a random axis-aligned 90-degree rotation. `bom.json` gives each
part's id, `name`, `source` (`drawing` or `step`), `quantity` and `item` (its
balloon number on the assembly drawing). The parts list may show standard
fasteners that are not in `bom.json`; those are not part of the answer.

Model the drawing parts at the drawn size and place every instance of every
part. Submit with the tools: `tools.export_part(solid, part_id)` for each
modelled part, `tools.use_part(part_id)` for each supplied one, then
`tools.submit_assembly(instances)`, where each instance is
`{"part_id": ..., "transform": T}` and `T` is a rigid 4x4 (a proper
rotation, no mirror or scale, plus a translation in millimetres) mapping
the part file's frame to the assembly frame. The global position and the
axis-aligned orientation of the whole assembly are free.
