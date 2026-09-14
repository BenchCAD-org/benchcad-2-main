#!/usr/bin/env python
"""Summarise one or more results files from harness/run.py.

    python tools/summarize.py results/*.json
    python tools/summarize.py results/run.json --by task     # default
    python tools/summarize.py results/run.json --by case

One line per case (task, case, headline score, the columns beside it, what
ended the episode, tokens), then the mean per task and the mean of the task
means. A case with no score (an API failure, a submission that could not be
scored) is listed with its error and excluded from the means -- the count of
scored cases is printed next to every mean so a partial run cannot pass for a
full one.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def task_of(rec: dict) -> str:
    """t2_realparts2assembly -> T2; tasks/task2/... or examples/task2/... -> T2."""
    p = rec.get("case", "")
    for part in Path(p).parts:
        if part.startswith("task") and part[4:].isdigit():
            return "T" + part[4:]
        if len(part) > 2 and part[0] == "t" and part[1].isdigit() and part[2] == "_":
            return "T" + part[1]
    return "?"


def rows_of(files: list[Path]) -> list[dict]:
    rows = []
    for f in files:
        d = json.loads(Path(f).read_text())
        for rec in d.get("cases", []):
            s = rec.get("score") or {}
            rows.append({
                "file": Path(f).name, "model": rec.get("model", d.get("model")), "task": task_of(rec),
                "case": rec.get("case_id"), "score": s.get("score"),
                "cols": {k: s[k] for k in ("iou", "avg_part", "asm_v1", "hit", "score_v1") if isinstance(s.get(k), (int, float))},
                "error": (rec.get("error") or rec.get("skipped") or (s.get("error") if s else None) or ""),
                "tokens": rec.get("tokens", {}), "seconds": rec.get("seconds"),
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--by", choices=("task", "case"), default="task")
    a = ap.parse_args(argv)
    rows = rows_of(a.results)
    if not rows:
        print("no case records in", ", ".join(map(str, a.results)))
        return 1
    print(f"{'task':4s} {'case':10s} {'score':>8s}  {'beside it':36s} {'in_tok':>9s} {'out_tok':>8s} {'s':>6s}  note")
    for r in sorted(rows, key=lambda r: (r["task"], r["case"] or "")):
        sc = "-" if r["score"] is None else f"{r['score']:.4f}"
        cols = " ".join(f"{k}={v:.3f}" for k, v in r["cols"].items())[:36]
        note = (r["error"][:60] if r["score"] is None else "")
        print(f"{r['task']:4s} {str(r['case']):10s} {sc:>8s}  {cols:36s} {r['tokens'].get('input', 0):>9d} {r['tokens'].get('output', 0):>8d} {str(r['seconds'] or ''):>6s}  {note}")
    by: dict[str, list[float]] = {}
    total = {}
    for r in rows:
        total[r["task"]] = total.get(r["task"], 0) + 1
        if r["score"] is not None:
            by.setdefault(r["task"], []).append(float(r["score"]))
    print()
    means = {}
    for t in sorted(total):
        v = by.get(t, [])
        means[t] = statistics.mean(v) if v else None
        m = "-" if not v else f"{means[t]:.4f}"
        print(f"{t}: mean {m:>8s}  ({len(v)}/{total[t]} cases scored)")
    scored = [m for m in means.values() if m is not None]
    if scored:
        print(f"mean of task means: {statistics.mean(scored):.4f}  over {len(scored)}/{len(total)} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
