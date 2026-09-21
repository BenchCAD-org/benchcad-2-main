#!/usr/bin/env python
"""Re-score the records of a harness results file with the current scorer.

    uv run python tools/rescore.py results/run.json [--only t2,t5] [--workers 2]
        [--only-unscored] [--map /home/old/=/home/new/ ...]

Every record that has an artifact is scored again exactly as the harness
scores it (harness.run.score_in_subprocess: a child interpreter, the same
time bounds); `score` and `seconds_score` are replaced, everything else --
the episode, its tokens, its transcript paths -- is kept. The previous file
is saved next to it as `<name>.before-rescore.json` the first time, and a
`rescored` note (UTC time, scorer tree's git head when known) is added to
the file's header. Records without an artifact (no submission, skipped,
episode error) are left alone.

Why: the scorer changes while a sweep runs (2026-09-18: the mesh guard, the
free-rotation alignment for references off their axes), and the user's
rule is "keep the model runs, re-judge the cases with the new scorer". It
is also how a run made with `--score-workers 0` (episodes recorded
unscored) gets its scores, on whichever machine has the cores: the T6
matcher can take an hour a board.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.run import score_in_subprocess, show


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--only", default="", help="comma-separated task prefixes (t2,t5); default every task")
    ap.add_argument("--cases", default="", help="comma-separated case ids; default every case")
    ap.add_argument("--workers", type=int, default=2, help="scorer processes at once")
    ap.add_argument("--only-unscored", action="store_true",
                    help="only records a --score-workers 0 run left unscored (score null, unscored true)")
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite a path prefix of every record's case and step before scoring "
                         "(repeatable, applied in order): the file was made on another machine")
    a = ap.parse_args(argv)
    maps = [m.split("=", 1) for m in a.map]

    def local(p: str) -> str:
        for old, new in maps:
            if p.startswith(old):
                p = new + p[len(old):]
        return p
    doc = json.loads(a.results.read_text())
    recs = doc["cases"] if isinstance(doc, dict) else doc
    only = {t.strip() for t in a.only.split(",") if t.strip()}
    ids = {c.strip() for c in a.cases.split(",") if c.strip()}

    def wanted(r: dict) -> bool:
        if not r.get("step") or "error" in r or "skipped" in r:
            return False
        if a.only_unscored and not r.get("unscored"):
            return False
        # the env directory is somewhere on the path (a --cases dir of
        # symlinks puts it right above the case, a bank puts cases/ between)
        task = next((p.split("_")[0] for p in Path(r["case"]).parts if re.match(r"t[1-6](_|$)", p)), "")
        return (not only or task in only) and (not ids or r["case_id"] in ids)

    todo = [r for r in recs if wanted(r)]
    print(f"{a.results}: {len(recs)} records, {len(todo)} to re-score", flush=True)
    if not todo:
        return 0
    backup = a.results.with_name(a.results.stem + ".before-rescore.json")
    if not backup.exists():
        backup.write_text(a.results.read_text())
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10, check=False).stdout.strip() or None
    except Exception:                                          # noqa: BLE001
        head = None

    def one(r: dict) -> tuple[dict, dict | None, str | None, float]:
        t0 = time.time()
        try:
            return r, score_in_subprocess(Path(local(r["case"])), Path(local(r["step"]))), None, time.time() - t0
        except Exception as e:                                 # noqa: BLE001
            return r, None, f"{type(e).__name__}: {e}", time.time() - t0

    def write() -> None:
        if isinstance(doc, dict):
            doc["rescored"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "scorer": head}
        a.results.write_text(json.dumps(doc, indent=1, default=str) + "\n")

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for n, f in enumerate(as_completed([ex.submit(one, r) for r in todo]), 1):
            r, score, err, secs = f.result()
            old = r.get("score")
            old_head = old.get("score") if isinstance(old, dict) else old
            if score is not None:
                r["score"], r["seconds_score"] = score, round(secs, 1)
                r.pop("score_error", None)
                r.pop("unscored", None)
                new_head = score.get("score") if isinstance(score, dict) else score
                delta = (f"{old_head:.3f} -> {new_head:.3f}" if isinstance(old_head, (int, float))
                         and isinstance(new_head, (int, float)) else f"{old_head} -> {new_head}")
                print(f"  [{n}/{len(todo)}] {r['case_id']}: {delta}  {show(score)[:110]}  ({secs:.0f}s)", flush=True)
            else:
                r["score_error"] = err
                print(f"  [{n}/{len(todo)}] {r['case_id']}: scorer failed, score kept: {err}", flush=True)
            write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
