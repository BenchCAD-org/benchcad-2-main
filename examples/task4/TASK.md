# T4 - four views -> parts + assembly

`views.png` is a 2x2 sheet of four parallel-projection views of one
assembled mechanism, all at the same scale. Every camera sits on the given direction from the
centre and looks at the centre:

    top-left     approx. (-1, -1,  1)     top-right     exactly ( 1,  1,  1)
    bottom-left  approx. (-1,  1, -1)     bottom-right  approx. ( 1, -1, -1)

"approx." means that camera (position and up vector together) is rotated
3 to 8 degrees about the centre off that direction; the rotation is not given.
`parts_views.png` is the same four cameras, one row per part, that part red
and the rest translucent. `bom.json` names each part and gives its count.
There is no 3-D input.

Model every part and place every instance. Submit with the tools:
`tools.export_part(solid, part_id)` for every part, then
`tools.submit_assembly(instances)`, where each instance is
`{"part_id": ..., "transform": T}` and `T` is a 4x4 in millimetres mapping
your part's frame to the assembly frame. World XYZ of the assembly is world
XYZ in the views; position and scale are free.
