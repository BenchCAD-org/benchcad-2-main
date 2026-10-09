## Spatial identity (T6 position mode)

This section defines the frame the hand-in example above is written in. The
graph carries `"schema": "pcb2schematic/2.0-position"` and
`"coordinate_reference": "view_top/full-image"`; every component carries a
`center: [x, y]`; and a top-level `terminals` list gives every terminal a
`parent_component` and a `relative_position`. Everything else — `nets`,
`incidences`, component types, each component's terminal-ID list — is
exactly as described above.

```json
{
  "schema": "pcb2schematic/2.0-position",
  "coordinate_reference": "view_top/full-image",
  "components": [{"id": "c1", "type": "resistor", "center": [0.4, 0.3],
                  "terminals": ["t1", "t2"]}],
  "terminals": [
    {"id": "t1", "parent_component": "c1", "relative_position": [-0.5, 0.0]},
    {"id": "t2", "parent_component": "c1", "relative_position": [0.5, 0.0]}
  ],
  "nets": [{"id": "n1"}],
  "incidences": [["t1", "n1"]]
}
```

Use the full, uncropped `views/view_top.png`, including its surrounding margin.
Origin is the upper-left image edge; x goes right, y goes down; divide pixel x
by image width and pixel y by image height. The lower-right edge is [1, 1].
All components, including bottom-side parts, use this same top projection.
To transfer a point from `view_bottom`, keep x and use `height - y`. Oblique
views help interpretation but are never coordinate references.

A component center is the midpoint of the axis-aligned bounding box of its
terminal pad centers — **built from the pads, not from the package body**. For a
part whose plastic body overhangs its pads, such as a connector with its pins
along one edge, the body's centre and the pad bounding box's centre are
different points and only the second one is correct. A terminal made of several
electrically common physical pads uses their mean center.
For each terminal, subtract the component center in pixels, then divide both
offsets by the larger of that bounding box's width and height **in pixels**.
This is a parent-local, board-axis-aligned frame; do not undo component rotation.
Coordinates lie in [-0.5, 0.5].

If that bounding box has no extent — one terminal, or several whose centers
coincide once electrically common pads are averaged — there is nothing to divide
by. Give every one of that component's terminals `[0, 0]`. This is not a
degenerate case to work around: a coaxial jack whose shield lugs average to the
same point as its centre pin genuinely has no internal geometry to report, and
`[0, 0]` is the right answer rather than a fallback.

A component pairs with one of a compatible type at a compatible place. The
compatible types are the ones listed with the schema above, and nothing wider:
being in the right place does not rescue a component whose type is refused.

How close is close enough depends on the part. A component whose nearest
same-type neighbour is far away is matched from further off than one sitting in
a dense cluster, because a wider gate there cannot pair it with the wrong part.
The gate is never narrower than 0.005 normalized distance and never reaches
past the midpoint to the nearest part it could be confused with; terminals work
the same way within their matched parent, from 0.025. So aim for the real centre
and do not spend effort chasing precision far below the size of the part: small
errors do not subtract topology credit. Unmatched components
and incidences still reduce their existing score channels. Terminal IDs and
list order may be freely renamed/reordered while keeping geometry/connectivity.
There is no global permutation fallback. These spatial rules supersede older
name-based or terminal-order correspondence descriptions above.
