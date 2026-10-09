# T6 - PCB renders -> terminal-net graph

`views/` holds six renders of one assembled PCB: `view_top`, `view_bottom`,
and two obliques of each side (`_obl_a`, `_obl_b`). A board with copper
layers between the outer two also has `view_inner<n>.png` per inner layer:
a top-down drawing of that layer's copper (traces, pours, vias), framed like
`view_top`; a 2-layer board has none. `README.md` has board-specific notes.

Recover the components and which terminals share a net. Leave a dict in
`result`, a terminal-net incidence graph
**with spatial identity**. The schema line, the component centres and the
top-level `terminals` list are all required; a graph without them is rejected
outright rather than scored low. "Spatial identity" below defines the frame:

```python
result = {
  "schema": "pcb2schematic/2.0-position",
  "coordinate_reference": "view_top/full-image",
  "components": [
    {"id": "R1", "type": "resistor", "terminals": ["R1.1", "R1.2"],
     "center": [0.41, 0.65]}
  ],
  "terminals": [
    {"id": "R1.1", "parent_component": "R1", "relative_position": [-0.5, 0.0]},
    {"id": "R1.2", "parent_component": "R1", "relative_position": [0.5, 0.0]}
  ],
  "nets": [{"id": "n1"}],
  "incidences": [["R1.1", "n1"]]
}
```

`type` is one of `resistor`, `capacitor`, `capacitor_polarized`, `inductor`,
`ferrite_bead`, `fuse`, `diode`, `led`, `transistor`, `ic`, `connector`,
`crystal`, `switch`, `transformer`, `buzzer`, `test_point`. Get it right: a type
the table below does not forgive drops the component and every connection on it.
The only latitude runs toward the generic — a `ferrite_bead` answered
`inductor`, a `capacitor_polarized` answered `capacitor`, or a `fuse` answered
`resistor` or `ferrite_bead` is accepted; the reverse is not.

`terminals` lists every pin, and the list order is free. Which
pin is which is carried by each terminal's `relative_position`, not by its
place in the list: getting pin 1 of an `ic` or a keyed `connector` right, or
polarity right on a `diode`, `led` or `capacitor_polarized`, means putting
that terminal at the right offset, not putting it first. No two terminals of
one component are interchangeable, a resistor's included: swap the two ends of
a resistor and each end is on the other's net, which is charged like any other
wrong connection. Component values are
not asked for and not scored. Net ids are yours, the power rails
included — nothing is matched by name, so `GND` and `n17` score the same. A
terminal is in at most one net; leave out what you cannot see.

**Names are yours.** Component, terminal and net IDs are internal references
only. Nothing is matched by name — correspondence comes from geometry and type,
so a printed reference designator buys you nothing and transcribing one is not
part of the job. Name things whatever helps you keep track. The only rule is
that the IDs inside one submission must be consistent with each other: a
terminal's `parent_component` must name a component you listed, and an
incidence must name a terminal you listed.

**A terminal is an electrical contact, not a piece of copper.** Most pads are
one pin each, so most of the time these are the same thing. Where they differ,
count contacts: a coaxial jack's four shield lugs are one ground contact, a
connector's mounting tabs are not signal pins, and a tact switch's four pads
are two contacts internally paired. Write one terminal per contact. One
terminal too many and one too few cost the same, so do not transcribe every
pad to be safe and do not merge two genuinely separate contacts just because
they share a net: an IC with four ground pins has four terminals. A pad the
package body covers — an exposed thermal pad under a QFN, say — is not
something a render can show and is not counted; leave it out.

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
