"""Graph schema, observability derivation and the Soft-MCS Graph-IoU metric
for the pcb2schematic / inverse-ECAD task family.

Source of truth. Tasks vendor a byte-identical copy into `environment/` and
`grading/`; `scripts/check_all.py` fails if the copies drift.
"""

from .matcher import MatchResult, decompose, graph_iou  # noqa: F401
from .observability import derive_gt                   # noqa: F401
from .schema import (Component, Graph, SchemaError,     # noqa: F401
                     SCHEMA_VERSION, dump_graph, load_graph, validate)
