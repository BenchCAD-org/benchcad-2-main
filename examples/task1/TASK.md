# T1 - part drawing -> part

`drawing.png` is the engineering drawing of one part, in millimetres,
first-angle projection unless its header says third-angle. It is the
sheet's drawing area (frame, footer and parts-list table cropped away) at a
resolution where the dimensions read. Only when a sheet's lettering is too
small to read whole is it also on disk as tiles `drawing_tile_r<i>c<j>.png`
(row i from the top, column j from the left, 10 % overlap); the directory
listing shows whether there are any. A further page is `drawing_p2.png`.

Write a CadQuery program that builds the part and leaves the solid in
`result`, axis-aligned to the drawing's views.
