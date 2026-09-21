#!/usr/bin/env python
"""Prove that a case runs end to end: the three gates every example must pass.

    python tools/dryrun_case.py tests/fixtures/t1/case1 [--deep] [--work DIR] [--json OUT]

  1. format   check_case (with --deep, assemblies are rebuilt from parts + instances)
  2. oracle   the reference itself, submitted, scores 1.0 through the declared verifier
  3. stage    input/ is staged into a real Sandbox (docker if present, else CADENV_LOCAL=1),
              a trivial program runs there, its export is scored -- any number, no crash,
              and gt/ must not be visible from inside

Exit status 0 when all three pass. Prints one line per gate and, with --json,
writes the full report. This is what tests/test_examples_run.py runs on every
case under envs/<env>/cases/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from envs.common.caseformat import GT_FILE, check_case, load_case  # noqa: E402
from envs.common.score_case import score_case  # noqa: E402

TRIVIAL = {
    "part": "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)\ntools.export(result, 'final.step')\n",
    "assembly": ("import cadquery as cq\nresult = cq.Assembly(name='asm')\n"
                 "result.add(cq.Workplane('XY').box(10, 10, 10), name='trivial_i1')\n"
                 "tools.export(result, 'final.step')\n"),
    "ecad": ("result = {'schema': 'pcb2schematic/1.0', 'components': [], 'nets': [], 'incidences': []}\n"
             "tools.export(result, 'pred_graph.json')\n"),
}


def _score_value(r: dict) -> float | None:
    """The headline: `score` when the verifier's declared metric sets one
    (part_v1 on T1 / T3, asm_v1 on T2, avg_part x asm_v1 on T4 / T5, T6's
    graph metric), else `iou` (a legacy declaration).

    A record that HAS a `score` key set to None declared a headline and could
    not compute it (a T5 case whose avg_part scope leaves no part type in the
    mean, docs/METRICS.md). That is None here, never the legacy `iou`: falling
    through would report a diagnostic as the headline and an unscorable case
    would pass a gate at 1.0.
    """
    v = r.get("score")
    if isinstance(v, (int, float)):
        return float(v)
    if "score" in r:
        return None
    return float(r["iou"]) if isinstance(r.get("iou"), (int, float)) else None


def _score_key(r: dict) -> str | None:
    if "score" in r:
        return "score"
    return "iou" if isinstance(r.get("iou"), (int, float)) else None


def dryrun(case_dir: Path, *, deep: bool, work: Path | None) -> dict:
    case = Path(case_dir).resolve()
    rep = {"case": str(case), "gates": {}}
    t0 = time.time()
    fr = check_case(case, deep=deep)
    rep["gates"]["format"] = {"ok": fr.ok, "errors": fr.errors, "warnings": fr.warnings, "seconds": round(time.time() - t0, 1)}
    if not fr.ok:
        return rep
    c = load_case(case)

    t0 = time.time()
    gt = case / GT_FILE[c.kind]
    try:
        r = score_case(case, gt)
        v = _score_value(r)
        rep["gates"]["oracle"] = {"ok": v is not None and abs(v - 1.0) < 1e-3, "score": v,
                                  "metric": r.get("metric", _score_key(r)), "key": _score_key(r),
                                  "seconds": round(time.time() - t0, 1), "error": r.get("error")}
    except Exception as ex:                                     # noqa: BLE001
        import traceback
        rep["gates"]["oracle"] = {"ok": False, "score": None, "seconds": round(time.time() - t0, 1),
                                  "exception": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()[-2000:]}

    t0 = time.time()
    from envs.common.sandbox import Sandbox
    root = work or REPO / "work" / "dryrun"
    root.mkdir(parents=True, exist_ok=True)
    wd = Path(tempfile.mkdtemp(prefix=case.name + "_", dir=root))
    stage = {"ok": False, "work": str(wd)}
    try:
        sb = Sandbox(case, wd)
        staged = sorted(str(p.relative_to(wd)) for p in wd.rglob("*") if p.is_file() and not p.name.startswith("_"))
        leaks = [p for p in staged if p.startswith("gt/") or p == "case.json"]
        stage["staged"] = staged
        stage["leaks"] = leaks
        stage["mode"] = "docker" if sb.docker else "local"
        res = sb.run("import tools\n" + TRIVIAL[c.kind], timeout=300)
        stage["rc"] = res.returncode
        stage["stderr_tail"] = res.stderr[-400:]
        out = wd / ("pred_graph.json" if c.kind == "ecad" else "final.step")
        stage["export_exists"] = out.exists()
        if out.exists():
            r2 = score_case(case, out)
            stage["trivial_score"] = _score_value(r2)
            stage["trivial_error"] = r2.get("error")
        stage["ok"] = res.returncode == 0 and out.exists() and not leaks and stage.get("trivial_score") is not None
    except Exception as ex:                                     # noqa: BLE001
        import traceback
        stage["exception"] = f"{type(ex).__name__}: {ex}"
        stage["traceback"] = traceback.format_exc()[-2000:]
    stage["seconds"] = round(time.time() - t0, 1)
    rep["gates"]["stage"] = stage
    rep["ok"] = all(g.get("ok") for g in rep["gates"].values())
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case"); ap.add_argument("--deep", action="store_true")
    ap.add_argument("--work", type=Path, default=None); ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    rep = dryrun(Path(a.case), deep=a.deep, work=a.work)
    for name, g in rep["gates"].items():
        extra = ""
        if name == "oracle":
            extra = f"score={g['score']} metric={g.get('metric')} key={g.get('key')}"
        elif name == "stage":
            extra = f"mode={g.get('mode')} staged={len(g.get('staged', []))} leaks={g.get('leaks')} trivial={g.get('trivial_score')}"
            if g.get("exception"): extra += f" {g['exception']}"
        elif g.get("errors"):
            extra = "; ".join(g["errors"])
        print(f"{'ok  ' if g.get('ok') else 'FAIL'} {name:7s} {g.get('seconds', 0):6.1f}s  {extra}")
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1) + "\n")
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
