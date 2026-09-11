"""Read task.toml: a task's kind, tools, verifier and orientation are DECLARED, never sniffed.

The old way was `score_case.is_assembly()` looking for a `parts/` directory in the case
and dispatching the scorer on that, and the sandbox injecting the standard-parts library
on `startswith("t5_")`. Both infer the TASK's type from the DATA's shape: a fifth
verifier means another branch, and a case with one directory more or less is silently
scored as a different task -- wrong, without an error.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Headline metrics a task may declare under [verify] (docs/METRICS.md).
#   legacy         raw 64^3 voxel IoU (parts) / orientation-aware IoU + hit + rubric (assemblies)
#   asm_v1         per-part-type leave-one-TYPE-out IoU gain, normalised by (1 - baseline) (T2, #24)
#   part_v1        0.40 iou_term + 0.35 surf_f1 + 0.25 pix_fg (T1 / T3, #26)
#   part_x_asm_v1  avg_part x asm_v1: the per-instance part_v1 inside the aligned
#                  assembly, averaged per part type, times asm_v1 (T4 / T5)
#   ecad_v2        the terminal-net graph metric S_C * S_T * S_N * P_short * P_open (T6)
# Every headline is in [0, 1]. Omitted means legacy. A name outside this tuple
# fails tools/check_tasks.py -- a misspelled metric must not silently score as legacy.
METRICS = ("legacy", "asm_v1", "part_v1", "part_x_asm_v1", "ecad_v2")
DEFAULT_METRIC = "legacy"
# How part_v1 poses the candidate for surf_f1 / pix_fg (docs/METRICS.md, "Pose").
# Read by the part verifier (T1 / T3) and by the per-instance part_v1 inside
# assemblies (avg_part: T2 / T4 / T5).
#   lab            the lab's reference behaviour: terms at the delivered pose
#   iou24_aligned  the best-of-24 proper rotation iou24 found is applied first (T1, T2, T5)
POSE_MODES = ("lab", "iou24_aligned")
DEFAULT_POSE_MODE = "lab"
# Which part TYPES avg_part averages over (docs/METRICS.md, "The T5 scope rule").
#   all        every reference part type. T2 (a diagnostic column: every part is
#              supplied there) and T4 (half the headline, and every part is
#              supplied there too, so there is nothing else to average).
#   modelled   only the types whose input/bom.json row says source = "drawing":
#              the parts the model had to build. T5 supplies 16 of its 21 types
#              as step_files/, so under "all" a submission that merely
#              re-exports them collects 16 free 1.0s and avg_part >= 0.76
#              before it models anything -- the part half of the score has to
#              measure the parts the model actually had to build.
# Declared, never sniffed from the case directory: a case with one step_files/
# entry more or less must not silently change what the mean is over.
AVG_PART_TYPES = ("all", "modelled")
DEFAULT_AVG_PART_TYPES = "all"


@dataclass(frozen=True)
class Task:
    id: str
    kind: str                 # part | assembly | ecad -- selects the verifier
    submission: str
    given: str                # nothing_3d | purchased_parts_step | all_parts_step | renders_only
    inputs: tuple[str, ...]
    renderer: str            # none | shared -- whether the reference renderer is injected
    inject: tuple[str, ...]   # extra assets the sandbox injects
    verify_entry: str         # "envs.verifiers.part:score"
    orientation: str          # free (2-D input, orientation not charged) | pinned (rendered input, orientation is ability) | n/a
    oracle: Path | None
    # Band exclusion: cases that cannot measure ability stay out of the mean.
    # They are still scored and still reported, just listed separately -- the
    # reason is "this case cannot measure", not "this model should not be
    # scored", and hiding the number would turn it into the latter.
    #
    # Two granularities, chosen by whether the property is constant within a
    # family -- which has to be measured first:
    #   band_exclude_families  the property follows from the family's structure,
    #                          every variant alike (a 2-part family: fit is N/A, 7/7)
    #   band_exclude_cases     the property follows from sampled parameters and
    #                          varies within the family (slotted_din_rail: 8/10 sub-voxel, 2/10 fine)
    band_exclude_families: list[str]
    band_exclude_cases: list[str]
    band_exclude_reason: str
    source_repo: str          # benchcad-2 | the heldout corpus | the data pipeline | the ECAD source repository
    split: str                # open | heldout -- mixing them contaminates, irreversibly
    data_root: Path
    default_dataset: str
    generator: str
    # Headline metric and its pose handling. Declared, never inferred from the
    # task id or the case directory; omitted means legacy. See METRICS above.
    metric: str = DEFAULT_METRIC
    pose_mode: str = DEFAULT_POSE_MODE
    # The scope of avg_part's mean over part types. See AVG_PART_TYPES above.
    avg_part_types: str = DEFAULT_AVG_PART_TYPES

    MEASURES = {
        "nothing_3d": "model the shape from the drawing (+ placement, for an assembly)",
        "purchased_parts_step": "model the made-to-print parts from drawings + place; purchased parts taken as supplied",
        "all_parts_step": "placement only -- every part's geometry is supplied, modelling is not measured",
        "renders_only": "recover schematic connectivity from renders of the assembled board (a graph, not geometry)",
    }

    @property
    def measures(self) -> str:
        """What the task measures. Derived from `given`, never free text -- it
        decides what the score means. Measured on one batch of machines: T2
        0.92 vs T5 0.65, and the difference is exactly the modelling step.
        Two numbers placed side by side without `given` get compared directly."""
        return self.MEASURES[self.given]

    def verifier(self):
        mod, fn = self.verify_entry.split(":")
        import importlib
        return getattr(importlib.import_module(mod), fn)

    def dataset(self, name: str | None = None) -> Path:
        return REPO / self.data_root / (name or self.default_dataset) / "cases"


def load(task_id: str) -> Task:
    p = REPO / "envs" / task_id / "task.toml"
    if not p.exists():
        raise FileNotFoundError(f"{task_id} has no task.toml -- every task must declare itself")
    d = tomllib.loads(p.read_text())
    t, v = d["task"], d["verify"]
    orc = v.get("oracle")
    return Task(id=t["id"], kind=t["kind"], submission=t["submission"],
                given=t["given"],
                inputs=tuple(t.get("inputs", ())),
                renderer=d.get("tools", {}).get("renderer", "none"),
                inject=tuple(d.get("tools", {}).get("inject", ())),
                verify_entry=v["entry"], orientation=v["orientation"],
                oracle=(REPO / orc) if orc else None,
                band_exclude_families=v.get("band_exclude_families", []),
                band_exclude_cases=v.get("band_exclude_cases", []),
                band_exclude_reason=v.get("band_exclude_reason", ""),
                source_repo=d["data"]["source_repo"], split=d["data"]["split"],
                data_root=Path(d["data"]["root"]),
                default_dataset=d["data"]["default_dataset"],
                generator=d["data"]["generator"],
                metric=v.get("metric", DEFAULT_METRIC),
                pose_mode=v.get("pose_mode", DEFAULT_POSE_MODE),
                avg_part_types=v.get("avg_part_types", DEFAULT_AVG_PART_TYPES))


def all_tasks() -> list[Task]:
    return [load(p.parent.name) for p in sorted((REPO / "envs").glob("*/task.toml"))]


if __name__ == "__main__":
    for t in all_tasks():
        print(f"{t.id:24s} kind={t.kind:9s} orient={t.orientation:7s} "
              f"metric={t.metric:7s} avg_part_types={t.avg_part_types:8s} "
              f"inject={list(t.inject)}  verify={t.verify_entry}")
