# T2 - part STEPs + assembly drawing -> assembly

`step_files/part_NN.step` are the parts, each centred on its bounding box and
given a random axis-aligned 90-degree rotation. `drawing.png` is the
assembly drawing (`drawing_p2.png` ... per further page);
`drawing_tile_r<i>c<j>.png` are overlapping pieces of the same sheet (rows
top to bottom, columns left to right) at a resolution where the lettering
is legible; a tile that would be blank paper is not written. `bom.json`
gives each part's id and `quantity`, and its `parts_list` note says how the
drawing's parts list maps to the ids. Hardware on the drawing that is not
in `bom.json` is not part of the answer; two ids may have identical
geometry, and either may then go in either place.

Place every instance. Submit with the tools: `tools.use_part(part_id)` for
every part, then `tools.submit_assembly(instances)`, where each instance is
`{"part_id": ..., "transform": T}` and `T` is a rigid 4x4 (a proper
rotation, no mirror or scale, plus a translation in millimetres) mapping
the part file's frame to the assembly frame. The global position and the
axis-aligned orientation of the whole assembly are free.
