# Input

`views/` — six standardized renders of the assembled board from fixed
viewpoints: top, bottom, two top obliques and two bottom obliques. Each side
gets an oblique pair at complementary azimuths, so what one view occludes the
other shows.

A camera sees outer copper only. If the board has copper layers between the
outer two, each is here as well — `view_inner1.png`, `view_inner2.png`, …
from the top surface down: a top-down drawing of that layer's copper alone,
framed pixel-for-pixel like `view_top` and seen from above. A trace that
enters a via on the photograph continues at the same pixel on the inner view.
No `view_inner*.png` means a 2-layer board.

That is everything. No netlist, no schematic, no board file. Components may
sit on both sides.
