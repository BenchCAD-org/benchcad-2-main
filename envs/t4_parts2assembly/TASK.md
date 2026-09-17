# T4 - four views -> parts + assembly

`views.png` is a 2x2 sheet of four parallel-projection views of one
assembled mechanism, all at the same scale. Every camera sits on the given
direction from the centre and looks at the centre, with world +Z up in the
image:

    top-left     approx. (-1, -1,  1)     top-right     exactly ( 1,  1,  1)
    bottom-left  approx. (-1,  1, -1)     bottom-right  approx. ( 1, -1, -1)

"approx." means that camera (position and up vector together) is rotated
3 to 8 degrees about the centre off that direction; the rotation is not given.
`parts/<part_id>_alone.png` and `parts/<part_id>_in_assembly.png` are 2x2
sheets with the same four cameras, one pair per part: `_alone` is one
instance of the part by itself at its own scale (its shape), `_in_assembly`
is the part red inside the ghosted assembly at assembly scale (its relative
size and position). `bom.json` names each part and gives its `quantity`.
There is no 3-D input.

Model every part and place every instance: `export_part` for every part,
then `submit_assembly` with one `{"part_id", "transform"}` per instance,
`transform` a rigid 4x4 (proper rotation, no mirror or scale, plus a
translation) mapping your part's frame to the assembly frame. World XYZ in
your code is world XYZ in the views; position and scale are free (any unit,
the same for every part).
