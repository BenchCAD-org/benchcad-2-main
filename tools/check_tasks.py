#!/usr/bin/env python
"""Contract gate: every task must declare completely, resolve its verifier,
and name an oracle.

Modelled on the ecad session's check_all.py, which fails any task without an
oracle. Without one, a change to the scorer leaves nothing that can tell us
whether it is still the same ruler. The gate goes up first; oracles are filled
in per task.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envs.tasks import AVG_PART_TYPES, METRICS, POSE_MODES, all_tasks  # noqa: E402

KINDS = {"part", "assembly", "ecad"}
ORIENT = {"free", "pinned", "n/a"}          # n/a: the answer is not geometry (T6)
GIVEN = {"nothing_3d", "purchased_parts_step", "all_parts_step", "renders_only"}
SPLIT = {"open", "heldout"}
RENDERER = {"none", "shared"}


def main() -> int:
    bad = []
    for t in all_tasks():
        if t.kind not in KINDS:
            bad.append(f"{t.id}: kind={t.kind!r} not in {KINDS}")
        # A misspelled exclusion entry fails SILENTLY: the name matches no
        # directory, the case stays in the mean, and the reported mean looks
        # entirely normal. So every name must resolve to a real directory.
        try:
            root = REPO / "envs" / t.id / "cases"
            fams = {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else None
        except OSError:
            fams = None
        if fams is not None:
            for f in t.band_exclude_families:
                if f not in fams:
                    bad.append(f"{t.id}: band_exclude_families entry {f!r} does not exist")
            for c in t.band_exclude_cases:
                if not (root / c).is_dir():
                    bad.append(f"{t.id}: band_exclude_cases entry {c!r} does not exist")
        if (t.band_exclude_families or t.band_exclude_cases) and not t.band_exclude_reason:
            bad.append(f"{t.id}: exclusion list without band_exclude_reason -- "
                       f"the reason is the entire point of the field")
        if t.renderer not in RENDERER:
            bad.append(f"{t.id}: renderer={t.renderer!r} not in {RENDERER}")
        if t.split not in SPLIT:
            bad.append(f"{t.id}: split={t.split!r} not in {SPLIT}")
        if t.given not in GIVEN:
            bad.append(f"{t.id}: given={t.given!r} not in {GIVEN}")
        if t.orientation not in ORIENT:
            bad.append(f"{t.id}: orientation={t.orientation!r} not in {ORIENT}")
        # A misspelled metric must not silently score as legacy.
        if t.metric not in METRICS:
            bad.append(f"{t.id}: metric={t.metric!r} not in {set(METRICS)}")
        if t.pose_mode not in POSE_MODES:
            bad.append(f"{t.id}: pose_mode={t.pose_mode!r} not in {set(POSE_MODES)}")
        if t.pose_mode == "iou24_aligned" and t.orientation != "free":
            bad.append(f"{t.id}: pose_mode=iou24_aligned needs orientation=free "
                       f"(a pinned pose searches nothing to align to)")
        # A misspelled scope must not silently average over every type again --
        # that is the whole value the declaration has over sniffing the case.
        if t.avg_part_types not in AVG_PART_TYPES:
            bad.append(f"{t.id}: avg_part_types={t.avg_part_types!r} not in {set(AVG_PART_TYPES)}")
        # `modelled` means "the types the model had to build from a drawing".
        # A task that supplies every part has none, so the mean would be
        # undefined on every one of its cases -- a declaration that scores
        # nothing, which is worth failing here rather than per case at run time.
        if t.avg_part_types == "modelled" and t.given == "all_parts_step":
            bad.append(f"{t.id}: avg_part_types=modelled with given=all_parts_step -- "
                       f"every part is supplied, so no part type was modelled and "
                       f"the mean would be undefined on every case")
        try:
            t.verifier()
        except Exception as e:                        # noqa: BLE001
            bad.append(f"{t.id}: verifier failed to resolve: {type(e).__name__}: {e}")
        if t.oracle is None:
            bad.append(f"{t.id}: no oracle declared")
        elif not t.oracle.exists():
            print(f"  ! {t.id}: oracle dir does not exist yet "
                  f"({t.oracle.relative_to(REPO)}) -- to be added")
        print(f"  {t.id:24s} kind={t.kind:9s} orient={t.orientation:7s} "
              f"metric={t.metric:7s} avg_part_types={t.avg_part_types:8s} "
              f"renderer={t.renderer} inject={list(t.inject)}")
        print(f"  {'':24s} given={t.given:22s} corpus {t.source_repo} / {t.split}")
    if bad:
        print("\nFAILED:")
        for b in bad:
            print("  x", b)
        return 1
    print(f"\n{len(all_tasks())} task declaration(s) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
