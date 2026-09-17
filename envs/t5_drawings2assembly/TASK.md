# T5 - drawing set + assembly drawing -> assembly

`drawing.png` is the assembly drawing (`drawing_p2.png` per further page).
`part_drawings/part_NN.png` are the drawings of the parts to model, in
millimetres, first-angle projection unless the sheet's header says
third-angle. Every sheet is its drawing area (frame, footer and parts-list
table cropped away; the parts list is `bom.json`) at a resolution where
the dimensions read. Only when a sheet's lettering is too small to read
whole is it also on disk as tiles `<sheet>_tile_r<i>c<j>.png` (row i from
the top, column j from the left, 10 % overlap); the directory listing shows
whether there are any. `step_files/part_NN.step` are the supplied parts,
each centred on its bounding box and given a random axis-aligned 90-degree
rotation. `bom.json` gives each part's id, `name`, `source` (`drawing` or
`step`), `quantity` and `item` (its balloon number on the assembly
drawing). The parts list may show standard fasteners that are not in
`bom.json`; those are not part of the answer.

Model the drawing parts at the drawn size and place every instance of every
part: `export_part` for each modelled part, `use_part` for each supplied
one, then `submit_assembly` with one `{"part_id", "transform"}` per
instance, `transform` a rigid 4x4 (proper rotation, no mirror or scale,
translation in millimetres) mapping the part file's frame to the assembly
frame. The whole assembly's position and axis-aligned orientation are free.
