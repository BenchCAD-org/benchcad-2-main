# T2 - part STEPs + assembly drawing -> assembly

`step_files/part_NN.step` are the parts, each centred on its bounding box and
given a random axis-aligned 90-degree rotation. `drawing.png` is the assembly
drawing (`drawing_p2.png` ... per further page). You are shown it downscaled:
the layout is clear, the lettering is not. The same sheet is also cut into
tiles, `drawing_tile_r<i>c<j>.png` (row i from the top, column j from the
left, neighbours overlapping by 10 %), each at a resolution where every number
and label is legible; a tile that would be blank paper is not written. The
tiles are in the directory, not in this message. Use `drawing.png` for where
things are and for `tools.crop` coordinates. `bom.json` gives each part's id,
`quantity` and
`item` (its balloon number on the drawing), and its `parts_list` note says how
the drawing's parts list is laid out. The parts list may show standard
fasteners that are not in `bom.json`; those are not part of the answer. Two
part ids can be the same geometry under two names; then either file may take
either place.

Place every instance. Submit with the tools: `tools.use_part(part_id)` for
every part, then `tools.submit_assembly(instances)`, where each instance is
`{"part_id": ..., "transform": T}` and `T` is a rigid 4x4 (a proper
rotation, no mirror or scale, plus a translation in millimetres) mapping
the part file's frame to the assembly frame. The global position and the
axis-aligned orientation of the whole assembly are free.
