# T1 - part drawing -> part

`drawing.png` is the engineering drawing of one part, in millimetres,
first-angle projection unless its title block says third-angle.
`drawing_tile_r<i>c<j>.png` are overlapping pieces of the same sheet (rows
top to bottom, columns left to right) at a resolution where the lettering
is legible; a tile that would be blank paper is not written. Read
dimensions there, measure on `drawing.png`. A multi-page drawing adds
`drawing_p2.png` and its tiles.

Write a CadQuery program that builds the part and leaves the solid in
`result`, axis-aligned to the drawing's views.
