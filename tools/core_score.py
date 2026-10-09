#!/usr/bin/env python
"""The BenchCAD 2.0 Core headline from harness/run.py results files.

    python tools/core_score.py results/core/<model>/*.json --dataset <core dir> [--json out.json]

The rule, per (model, effort):
- Every Core case must be resolved: scored, or no submission (= 0). A scorer
  zero is a real 0.
- Pending is not 0 and not left out: a scorer timeout or memory budget, an
  episode ended by an API / infrastructure failure, an evaluator_error, a T6
  board with no position-mode record, a case with no record. Any pending case
  makes the run INCOMPLETE and no headline is printed (--provisional prints a
  labelled mean over the resolved cases).
- A case's resolved reps are averaged first; the headline is the plain mean
  over the cases (100 for Core). Per-task means are diagnostics.
- T6 counts only position-mode records (correspondence_mode position).
- A run at other than 30 rounds is a smoke run, not a reportable score.
- Records covering fewer cases than the dataset holds (dataset.json n_cases,
  from --dataset-info) are a SUBSET: per-task lines, no headline.

Also per effort: mean $ per case (tools/prices.json, list prices of that exact
model id, an estimate; "price unknown" for any other id), mean input / output
tokens per case, median wall-clock per case.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUNDS = 30
PENDING_ERR = re.compile(r"memory|MemoryBudget|OOM|timeout|timed out|Timeout|evaluator_error|"
                         r"APIError|APIConnection|Connection|ServerError|InternalServerError|exhausted|"
                         r"RateLimit|\b429\b|\b5\d\d\b", re.I)


def task_of(case: str) -> str:
    for part in Path(case).parts:
        if part.startswith("task") and part[4:].isdigit():
            return "T" + part[4:]
        if len(part) > 2 and part[0] == "t" and part[1].isdigit() and part[2] == "_":
            return "T" + part[1]
    return "?"


def resolve(rec: dict) -> tuple[str, float | None, str]:
    """('scored' | 'no_submission' | 'pending', value, why) for one record."""
    task = task_of(rec.get("case", ""))
    err = rec.get("error") or ""
    if rec.get("pending_score"):                     # the episode finished; its score never landed
        return "pending", None, "episode finished, not scored yet (re-run to score it)"
    if rec.get("skipped"):
        return "pending", None, "skipped: " + str(rec["skipped"])[:80]
    if err:
        if "CallOverBudget" in err:                  # the model's own reply ran past the budget
            return "no_submission", 0.0, "call over budget"
        return "pending", None, err[:100]
    s = rec.get("score")
    if s is None:
        if rec.get("unscored"):
            return "pending", None, "not scored yet (tools/rescore.py)"
        return "no_submission", 0.0, "no submission"
    if task == "T6":
        if s.get("status") == "invalid_prediction":
            return "no_submission", 0.0, "invalid_prediction"
        if s.get("correspondence_mode") != "position":
            return "pending", None, f"T6 not in position mode ({s.get('correspondence_mode') or s.get('status')})"
        if s.get("status") not in (None, "ok") or not isinstance(s.get("score"), (int, float)):
            return "pending", None, f"T6 {s.get('status')}"
        return "scored", float(s["score"]), ""
    if isinstance(s.get("score"), (int, float)):
        return "scored", float(s["score"]), ""
    e = str(s.get("error") or s.get("status") or "")
    if PENDING_ERR.search(e):
        return "pending", None, "scorer: " + e[:90]
    return "no_submission", 0.0, "nothing scorable: " + e[:80]


def prices() -> dict:
    p = ROOT / "tools/prices.json"
    return json.loads(p.read_text()) if p.exists() else {}


def price_of(model: str | None, table: dict) -> dict | None:
    """The list price of this exact model id (or a dated snapshot of it,
    <id>-YYYYMMDD), else None: a price is never borrowed from another model."""
    if not model or model.startswith("_"):
        return None
    if model in table:
        return table[model]
    m = re.fullmatch(r"(.+)-\d{8}", model)
    return table.get(m.group(1)) if m else None


def usd(model: str, tok: dict, table: dict) -> float | None:
    p = price_of(model, table)
    if not p or not tok:
        return None
    cached = tok.get("cached", 0) or 0
    return ((max(0, (tok.get("input", 0) or 0) - cached) * p["input"] + cached * p["cached"]
             + (tok.get("output", 0) or 0) * p["output"]) / 1e6)


def expected_cases(dataset: Path | None) -> dict[str, set[str]] | None:
    """{task: {case ids}} from the downloaded Core tree, or None."""
    if not dataset:
        return None
    out: dict[str, set[str]] = {}
    for cj in sorted(Path(dataset).rglob("case.json")):
        out.setdefault(task_of(str(cj.parent)), set()).add(cj.parent.name)
    return out


def harness_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:                                            # noqa: BLE001
        return None


def record_revs(files: list[Path]) -> dict[str, int]:
    """{harness commit: records} as stamped at episode time ("unknown" for
    records from before the stamp)."""
    out: dict[str, int] = {}
    for f in files:
        for rec in json.loads(Path(f).read_text()).get("cases", []):
            k = rec.get("harness_commit") or "unknown"
            out[k] = out.get(k, 0) + 1
    return out


def summarize(files: list[Path], dataset: Path | None = None, n_core: int | None = None) -> list[dict]:
    """One summary per (model, effort) found in `files`. `n_core`: the cases
    the dataset holds; fewer cases than that is a subset, with no headline."""
    want = expected_cases(dataset)
    table = prices()
    groups: dict[tuple, dict] = {}
    for f in files:
        d = json.loads(Path(f).read_text())
        for rec in d.get("cases", []):
            model = rec.get("model") or d.get("model")
            effort = rec.get("effort") or d.get("effort")
            g = groups.setdefault((model, effort), {"rounds": set(), "cases": {}, "files": set()})
            g["rounds"].add(rec.get("rounds") or d.get("rounds"))
            g["files"].add(str(f))
            key = (task_of(rec.get("case", "")), rec.get("case_id") or Path(rec.get("case", "")).name)
            g["cases"].setdefault(key, []).append(rec)
    out = []
    for (model, effort), g in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        keys = set(g["cases"])
        if want is not None:
            keys |= {(t, c) for t, cs in want.items() for c in cs}
        per_case, pending, costs, tin, tout, secs, sent = {}, [], [], [], [], [], set()
        for key in sorted(keys):
            recs = g["cases"].get(key, [])
            vals = []
            why = "no record"
            for r in recs:
                if "effort_sent" in r:
                    sent.add(r["effort_sent"])
                state, v, why_r = resolve(r)
                if state != "pending":
                    vals.append(v)
                else:
                    why = why_r
                tok = r.get("tokens") or {}
                c = usd(model, tok, table)
                if c is not None:
                    costs.append(c)
                tin.append(tok.get("input", 0) or 0)
                tout.append(tok.get("output", 0) or 0)
                if r.get("seconds"):
                    secs.append(float(r["seconds"]))
            if vals:
                per_case[key] = statistics.mean(vals)
            else:
                pending.append({"task": key[0], "case": key[1], "why": why})
        n = len(keys)
        tasks = sorted({k[0] for k in keys})
        per_task = {t: {"mean": (statistics.mean([v for k, v in per_case.items() if k[0] == t])
                                 if any(k[0] == t for k in per_case) else None),
                        "resolved": sum(1 for k in per_case if k[0] == t),
                        "cases": sum(1 for k in keys if k[0] == t)} for t in tasks}
        rounds = sorted(r for r in g["rounds"] if r is not None)
        subset = n_core is not None and n < n_core
        complete = not pending and n > 0 and not subset
        out.append({
            "model": model, "effort": effort, "effort_sent": sorted(sent, key=str),
            "rounds": rounds[0] if len(rounds) == 1 else rounds,
            "smoke": rounds != [ROUNDS],
            "n_cases": n, "n_core": n_core, "subset": subset,
            "resolved": len(per_case), "pending": pending, "complete": complete,
            "core_mean": statistics.mean(per_case.values()) if complete else None,
            "provisional_mean": statistics.mean(per_case.values()) if per_case else None,
            "per_task": per_task,
            "usd_per_case": (sum(costs) / n) if costs and n else None,
            "price": "list" if price_of(model, table) else "unknown",
            "input_tokens_per_case": (sum(tin) / n) if n else None,
            "output_tokens_per_case": (sum(tout) / n) if n else None,
            "median_seconds_per_case": statistics.median(secs) if secs else None,
            "files": sorted(g["files"]),
        })
    return out


def fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def report(rows: list[dict], provisional: bool = False, reference: dict | None = None) -> str:
    lines = []
    for r in rows:
        lines.append(f"== {r['model']}  effort {r['effort']}"
                     + (f" (sent {', '.join(map(str, r['effort_sent']))})" if r["effort_sent"]
                        and r["effort_sent"] != [r["effort"]] else "") + f"  rounds {r['rounds']}")
        if r["smoke"]:
            lines.append(f"   SMOKE RUN, rounds={r['rounds']}: not a reportable score "
                         f"(reported numbers use {ROUNDS} rounds)")
        of = f" ({r['n_cases']}/{r['n_core']} Core cases)" if r.get("subset") else ""
        if r["complete"]:
            lines.append(f"   CORE MEAN {r['core_mean']:.4f}   over {r['n_cases']} cases")
        elif r["pending"]:
            lines.append(f"   INCOMPLETE: {len(r['pending'])} of {r['n_cases']} cases pending{of}, no headline")
        else:
            lines.append(f"   SUBSET {r['n_cases']}/{r['n_core']}: these records cover {r['n_cases']} of the "
                         f"{r['n_core']} Core cases, no headline")
        if not r["complete"] and provisional and r["provisional_mean"] is not None:
            lines.append(f"   provisional mean over the {r['resolved']} resolved cases: "
                         f"{r['provisional_mean']:.4f} (NOT the headline)")
        if reference:
            ref = next((x for x in reference.get("lines", [])
                        if x.get("model") == r["model"] and x.get("effort") == r["effort"]), None)
            if ref:
                lines.append(f"   reference ({reference.get('version', '')}): {fmt(ref.get('core_mean'))}"
                             + (f"  [{ref['noise_note']}]" if ref.get("noise_note") else ""))
        lines.append("   " + "  ".join(f"{t} {fmt(v['mean'], 3)} ({v['resolved']}/{v['cases']})"
                                       for t, v in r["per_task"].items()))
        cost = (f"${fmt(r['usd_per_case'], 2)}" if r.get("price", "list") == "list"
                else f"price unknown for {r['model']}")
        lines.append(f"   per case: {cost}  in {fmt(r['input_tokens_per_case'], 0)} tok  "
                     f"out {fmt(r['output_tokens_per_case'], 0)} tok  median {fmt(r['median_seconds_per_case'], 0)} s")
        for p in r["pending"][:20]:
            lines.append(f"   pending {p['task']} {p['case']}: {p['why']}")
        if len(r["pending"]) > 20:
            lines.append(f"   ... {len(r['pending']) - 20} more pending (see the JSON)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--dataset", type=Path, default=None,
                    help="the downloaded Core tree: every case in it is expected (a missing one is pending)")
    ap.add_argument("--dataset-info", type=Path, default=None,
                    help="the dataset root holding dataset.json (default: --dataset)")
    ap.add_argument("--json", type=Path, default=None, help="write the summary here")
    ap.add_argument("--provisional", action="store_true",
                    help="on an incomplete run, also print the mean over the resolved cases, labelled")
    ap.add_argument("--compare", type=Path, default=None, help="a core_reference.json to print beside each line")
    a = ap.parse_args(argv)
    files = [f for f in a.results if f.exists()]
    revs = record_revs(files)
    meta = {"harness_commits": revs, "checkout_commit": harness_commit()}
    info = a.dataset_info or a.dataset
    if info and (Path(info) / "dataset.json").exists():
        ds = json.loads((Path(info) / "dataset.json").read_text())
        meta["dataset"] = {k: ds.get(k) for k in ("name", "version", "scorer_digest", "n_cases")}
    rows = summarize(files, a.dataset,
                     (meta.get("dataset") or {}).get("n_cases"))
    if not rows:
        print("no case records in", ", ".join(map(str, a.results)))
        return 1
    ref = json.loads(a.compare.read_text()) if a.compare else None
    known = [k for k in revs if k != "unknown"]
    ran = (", ".join(f"{k} ({n})" for k, n in sorted(revs.items(), key=lambda kv: -kv[1]))
           if len(revs) > 1 else known[0] if known else f"unknown (checkout {meta['checkout_commit']})")
    print(f"harness {ran}  dataset "
          + (f"{meta['dataset']['name']} {meta['dataset']['version']}  scorer_digest "
             f"{str(meta['dataset']['scorer_digest'])[:12]}" if "dataset" in meta else "-"))
    print(report(rows, a.provisional, ref))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({**meta, "efforts": rows}, indent=1) + "\n")
        print(f"summary -> {a.json}")
    return 0 if all(r["complete"] for r in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
