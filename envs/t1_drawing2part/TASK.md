# T1 - part drawing -> part

`drawing.png` is the engineering drawing of one part, in millimetres,
first-angle projection unless its title block says third-angle. You are
shown it downscaled: the layout is clear, the lettering is not. The same
sheet is also cut into tiles, `drawing_tile_r<i>c<j>.png` (row i from the
top, column j from the left, neighbours overlapping by 10 %), each at a
resolution where every dimension is legible; a tile that would be blank
paper is not written. The tiles are in the directory, not in this message.
Use `drawing.png` for where things are and for `tools.crop` coordinates. A
multi-page drawing adds `drawing_p2.png` and its tiles.

Write a CadQuery program that builds the part and leaves the solid in
`result`, axis-aligned to the drawing's views.
