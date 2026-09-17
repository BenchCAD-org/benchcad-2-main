# T2 - part STEPs + assembly drawing -> assembly

`step_files/part_NN.step` are the parts, each centred on its bounding box
and given a random axis-aligned 90-degree rotation. `drawing.png` is the
assembly drawing (`drawing_p2.png` per further page): the sheet's drawing
area (frame, footer and parts-list table cropped away; the parts list is
`bom.json`) at a resolution where the balloons and dimensions read. Only
when a sheet's lettering is too small to read whole is it also on disk as
tiles `drawing_tile_r<i>c<j>.png` (row i from the top, column j from the
left, 10 % overlap); the directory listing shows whether there are any.
`bom.json` gives each part's id, `quantity` and `item` (its balloon number
on the drawing); its `parts_list` note says how the drawing's parts list
is laid out. The parts list may show standard fasteners that are not in
`bom.json`; those are not part of the answer. Two part ids can be the same
geometry under two names; then either file may take either place.

Place every instance: `use_part` for every part, then `submit_assembly`
with one `{"part_id", "transform"}` per instance, `transform` a rigid 4x4
(proper rotation, no mirror or scale, translation in millimetres) mapping
the part file's frame to the assembly frame. The whole assembly's position
and axis-aligned orientation are free.
