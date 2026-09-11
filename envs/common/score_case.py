#!/usr/bin/env python
"""Pick the scoring convention from the task's own contract -- callers (harness /
verify_case / dryrun_case) only ever call this one function.

The convention is written in the task's own `task.toml`:

    [verify]
    entry = "envs.verifiers.assembly:score"
    orientation = "free"          # or pinned; decides which IoU is the headline metric
                                  # for an assembly task

All this module does is **find that entry and call it**; it no longer decides the task
type itself.

⚠️ It used to decide by whether the environment directory's name contained `assembly`,
which is a criterion that leaks: generate a case somewhere else (debugging, a temporary
directory) and `parent.parent` is no longer the environment directory, so an assembly
task is treated as a part task and produces a single IoU with no per-instance hit rate
-- (measured: hit once when running the oracle check from a scratch directory).
It now reads the contract, and only falls back to that guesswork when no contract can be
read (also taking "does it carry per-instance pose ground truth" into account).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=64)
def load_task(case_dir: Path) -> dict | None:
    """The task contract for a case. A case in the case format names its env
    in case.json, and that env's task.toml is the contract wherever the case
    sits (tests/fixtures/, a scratch copy, a data tree). Otherwise walk up from the
    case directory and take the first task.toml. None when nothing declares."""
    import json
    import tomllib
    cj = Path(case_dir) / "case.json"
    if cj.exists():
        try:
            env = json.loads(cj.read_text()).get("env")
            f = Path(__file__).resolve().parents[2] / "envs" / env / "task.toml"
            if f.exists():
                return tomllib.loads(f.read_text())
        except Exception:                                      # noqa: BLE001
            pass
    for base in (Path(case_dir).parent.parent, Path(case_dir).parent):
        f = base / "task.toml"
        if f.exists():
            try:
                return tomllib.loads(f.read_text())
            except Exception:                                  # noqa: BLE001
                return None
    return None


def _entry(task: dict | None):
    """The contract's entry -> a callable. Returns None when it cannot be resolved, so
    the caller takes the fallback path."""
    ref = ((task or {}).get("verify") or {}).get("entry")
    if not ref or ":" not in ref:
        return None
    mod, fn = ref.split(":", 1)
    try:
        import importlib
        return getattr(importlib.import_module(mod), fn)
    except Exception:                                          # noqa: BLE001
        return None


def is_assembly(case_dir: Path) -> bool:
    """The fallback criterion (used when there is no task.toml): the environment
    directory's name contains assembly, or the case carries per-instance pose ground
    truth."""
    d = Path(case_dir)
    return ("assembly" in d.parent.parent.name
            or (d / "gt/instances.json").exists()
            or (d / "gt/poses.json").exists())


def score_case(case_dir: Path, step: Path) -> dict:
    case_dir, step = Path(case_dir), Path(step)
    if not step or not step.exists():
        return {"iou": 0.0, "score": 0.0, "error": "no submission"}
    task = load_task(case_dir)
    fn = _entry(task)
    if fn is not None:
        # Arity from the signature, not by catching TypeError: a TypeError
        # raised INSIDE a verifier would otherwise re-dispatch to the two-arg
        # call and, for a part task, come back as a legacy record with no
        # `score`, no `metric` and no error.
        import inspect
        if len(inspect.signature(fn).parameters) >= 3:
            return fn(case_dir, step, task)
        return fn(case_dir, step)             # verifiers that take (case, submission) only
    # -- fallback: no contract, or the entry cannot be resolved -----------------
    if is_assembly(case_dir):
        from envs.verifiers.assembly import score as _asm
        return _asm(case_dir, step, task)
    from envs.verifiers.part import score as _part
    return _part(case_dir, step, task)


def fmt(r: dict) -> str:
    if r.get("metric") == "part_v1":
        from envs.verifiers.part import fmt as _fmt
        return _fmt(r)
    from envs.verifiers.assembly import fmt as _fmt
    return _fmt(r)
