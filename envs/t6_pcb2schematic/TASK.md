# T6 - PCB renders -> terminal-net graph

`views/` holds six renders of one assembled 2-layer PCB: `view_top`,
`view_bottom`, and two obliques of each side (`_obl_a`, `_obl_b`). All
routing is on the outer layers. `README.md` has board-specific notes.

Recover the components and which terminals share a net. Leave a dict in
`result`:

```python
result = {
  "schema": "pcb2schematic/1.0",
  "components": [{"id": "C2", "type": "capacitor", "terminals": ["C2.1", "C2.2"],
                  "value": 1e-7, "value_unit": "F"},
                 {"id": "U1", "type": "ic", "terminals": ["U1.1", "U1.2"]}],
  "nets": [{"id": "GND"}, {"id": "N1"}],
  "incidences": [["C2.1", "N1"], ["C2.2", "GND"], ["U1.1", "N1"]]}
```

`id` is the silkscreen designator where readable. Net ids are yours; name a
power rail as the board does (`GND`, `VCC`, `3V3`). A terminal is in at most
one net; leave out what you cannot see.
