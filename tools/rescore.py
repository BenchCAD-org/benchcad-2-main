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

Only the answer the record was scored on is judged again. A record
names its answer by path, and a lane that shared the path once replaced the
answer under three rescores in a row. Before scoring, the answer on disk is
compared with the record's `artifact_sha256` (harness/artifact_hash.py).
Records made before that field existed are compared with a snapshot manifest
(--snapshot answers_snapshot_*.manifest.json). The comparison can come out
three ways:

  - mismatch (or answer missing): the record is not scored. Its score is
    kept and `artifact_check` says why.
  - unverified (neither the record nor a snapshot says which answer it
    was): skipped, unless --allow-unverified is given.
  - either kind of refusal: the run exits 2.

T6: a record whose answer is a pcb2schematic/1.0 graph, on a case that is now
scored in position mode (gt/correspondence.json), is the answer to the task
the case asked before. It is re-scored on the legacy path it was first scored
on (ecad.score mode="historical"; the record's score says `forced_mode:
historical`, `correspondence_mode: legacy`), never in position mode, where it
would become invalid_prediction 0 and the historical number would be lost.
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

from harness.artifact_hash import Snapshots, check
from harness.run import MemoryBudgetExceeded, score_in_subprocess, score_memory_cap, show
from envs.verifiers.ecad import historical_submission


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
    ap.add_argument("--snapshot", action="append", default=[], type=Path, metavar="MANIFEST",
                    help="a snapshot manifest ({\"files\": {path: sha256}}, e.g. answers_snapshot_*.manifest.json) "
                         "that says which answer a record made before artifact_sha256 was scored on (repeatable)")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="also re-score records whose answer neither the record nor a snapshot vouches for; "
                         "their artifact_check says unverified")
    a = ap.parse_args(argv)
    maps = [m.split("=", 1) for m in a.map]
    snaps = Snapshots(a.snapshot) if a.snapshot else None

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

    def one(r: dict) -> tuple[dict, dict, dict | None, str | None, float]:
        t0 = time.time()
        step = Path(local(r["step"]))
        chk = check(r, step, snaps)
        if chk["status"] in ("mismatch", "missing") or (chk["status"] == "unverified" and not a.allow_unverified):
            return r, chk, None, None, 0.0
        try:
            case = Path(local(r["case"]))
            if historical_submission(case, step):
                # a 1.0 answer on a case now scored in position mode: judged on
                # the legacy path it was first judged on, labelled so, never
                # turned into invalid_prediction 0
                return r, chk, score_in_subprocess(case, step, mode="historical"), None, time.time() - t0
            return r, chk, score_in_subprocess(case, step), None, time.time() - t0
        except MemoryBudgetExceeded as e:                      # infrastructure: no score, re-score alone
            return r, chk, None, f"{type(e).__name__}: {e}", time.time() - t0
        except Exception as e:                                 # noqa: BLE001
            return r, chk, None, f"{type(e).__name__}: {e}", time.time() - t0

    n_scored = 0

    def write() -> None:
        if isinstance(doc, dict) and n_scored:
            doc["rescored"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "scorer": head}
        a.results.write_text(json.dumps(doc, indent=1, default=str) + "\n")

    refused, unverified = [], []
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for n, f in enumerate(as_completed([ex.submit(one, r) for r in todo]), 1):
            r, chk, score, err, secs = f.result()
            old = r.get("score")
            old_head = old.get("score") if isinstance(old, dict) else old
            at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if chk["status"] in ("mismatch", "missing"):
                # judging the file would score another answer under this record's name
                r["artifact_check"] = {**chk, "at": at}
                refused.append(r)
                print(f"  [{n}/{len(todo)}] {r['case_id']}: REFUSED, score kept: {chk['why']}", flush=True)
                write()
                continue
            if chk["status"] == "unverified" and not a.allow_unverified:
                unverified.append(r)
                print(f"  [{n}/{len(todo)}] {r['case_id']}: skipped, answer unverified ({chk['why']})", flush=True)
                continue
            if score is not None:
                n_scored += 1
                r["artifact_check"] = {**chk, "at": at}
                r["score"], r["seconds_score"] = score, round(secs, 1)
                r["score_memory_cap"] = score_memory_cap()
                r.pop("score_error", None)
                r.pop("memory_budget_exceeded", None)
                r.pop("unscored", None)
                new_head = score.get("score") if isinstance(score, dict) else score
                delta = (f"{old_head:.3f} -> {new_head:.3f}" if isinstance(old_head, (int, float))
                         and isinstance(new_head, (int, float)) else f"{old_head} -> {new_head}")
                print(f"  [{n}/{len(todo)}] {r['case_id']}: {delta}  {show(score)[:110]}  ({secs:.0f}s)", flush=True)
            else:
                r["score_error"] = err
                if err and err.startswith("MemoryBudgetExceeded"):
                    r["memory_budget_exceeded"] = True
                print(f"  [{n}/{len(todo)}] {r['case_id']}: scorer failed, score kept: {err}", flush=True)
            write()
    if refused or unverified:
        print(f"{len(refused)} refused (the answer on disk is not the one scored), {len(unverified)} unverified "
              "(no artifact_sha256; give --snapshot, or --allow-unverified to judge them anyway)", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
