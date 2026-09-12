# T3 - four views -> part

`views.png` is a 2x2 sheet of four parallel-projection views of one part,
all at the same scale. Every camera sits on the given direction from the
part's centre and looks at the centre, with world +Z up in the image:

    top-left     approx. (-1, -1,  1)     top-right     exactly ( 1,  1,  1)
    bottom-left  approx. (-1,  1, -1)     bottom-right  approx. ( 1, -1, -1)

"approx." means that camera (position and up vector together) is rotated
3 to 8 degrees about the centre off that direction; the rotation is not given.

Write a CadQuery program that builds the part and leaves the solid in
`result`. World XYZ in your code is world XYZ in the views; position and
scale are free.
