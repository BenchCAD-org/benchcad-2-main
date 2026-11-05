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
- Every rep must resolve. A case with a rep that did not is pending, and is
  never averaged over the reps that did -- until 2026-11-05 the unresolved rep
  was dropped silently and the case kept its headline, so a run could report a
  mean over 2 reps of one case and 3 of another. A rep is the record's `rep`,
  or the results file for a record that does not name one (one file is one run
  of one rep); two records of the same rep -- a lane re-run into a second
  results file -- are that one rep, and it resolves if either of them did.
- A case's resolved reps are averaged first; the headline is the plain mean
  over the cases (100 for Core). Per-task means are diagnostics.
- T6 counts only position-mode records (correspondence_mode position).
- A run at other than 60 rounds and a 500,000 output-token budget (the
  setting since 2026-10-18, 2.1; 40 rounds and 500,000 from
  2026-10-16, 2.1-beta; 30 rounds and no budget before, 2.0) is a smoke run,
  not a reportable score; so is one whose records come from more than one harness
  version (harness_version, stamped by harness/run.py; "2.0" for records
  from before the stamp).
- Records covering fewer cases than the dataset holds (dataset.json n_cases,
  from --dataset-info) are a SUBSET: per-task lines, no headline.
- The Public Set (dataset.json name benchcad-2.0-public; tools/run_core.sh
  --set public) is scored by the same rule and labelled PUBLIC: for anyone to
  look at, not a reportable score.

Also per effort: mean $ per case (tools/prices.json, list prices of that exact
model id, an estimate; "price unknown" for any other id; a model priced by
prompt length is priced per request from tokens.per_call, and records without it
are a labelled estimate), mean input / output tokens per case without the
replies the harness discarded and asked again for (reported apart as
discarded_*), median wall-clock per case.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUNDS = 60
TOKEN_BUDGET = 500_000
UNSTAMPED = "2.0"                       # records from before harness/run.py stamped harness_version
PUBLIC_SET = "benchcad-2.0-public"      # dataset.json name of the open Public Set (run_core.sh --set public)
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


ESTIMATED = "estimated: per-call sizes not recorded"


def _call_usd(p: dict, inp: float, cached: float, out: float) -> float:
    """One request at its tier: a prompt over long_context.over_input_tokens (all
    its input, cache reads and writes included) pays the long-context rates for
    the whole request (tools/prices.json, _long_context)."""
    lc = p.get("long_context")
    r = lc if lc and inp > lc["over_input_tokens"] else p
    return (max(0.0, inp - cached) * r["input"] + cached * r["cached"] + out * r["output"]) / 1e6


def usd_of(model: str, tok: dict, table: dict) -> tuple[float | None, str | None]:
    """(the case's list-price $, None) when its calls' sizes are recorded
    (tokens.per_call) or the model has one price at every prompt length;
    (an estimate, ESTIMATED) for a tiered model whose record holds only per-case
    sums: each call at the case's mean prompt size. (None, None): no price."""
    p = price_of(model, table)
    if not p or not tok:
        return None, None
    calls = tok.get("per_call")
    if calls:
        return sum(_call_usd(p, c.get("input", 0) or 0, c.get("cached", 0) or 0, c.get("output", 0) or 0)
                   for c in calls), None
    inp, cached, out = (tok.get(k, 0) or 0 for k in ("input", "cached", "output"))
    n = tok.get("calls") or 0
    if not p.get("long_context") or not n:
        return _call_usd({k: v for k, v in p.items() if k != "long_context"}, inp, cached, out), None
    return n * _call_usd(p, inp / n, cached / n, out / n), ESTIMATED


def usd(model: str, tok: dict, table: dict) -> float | None:
    return usd_of(model, tok, table)[0]


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


def record_versions(files: list[Path]) -> dict[str, int]:
    """{harness version: records} (UNSTAMPED for records from before the stamp)."""
    out: dict[str, int] = {}
    for f in files:
        d = json.loads(Path(f).read_text())
        for rec in d.get("cases", []):
            k = rec.get("harness_version") or d.get("harness_version") or UNSTAMPED
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
            g = groups.setdefault((model, effort), {"rounds": set(), "budgets": set(), "versions": set(),
                                                    "cases": {}, "files": set()})
            g["versions"].add(rec.get("harness_version") or d.get("harness_version") or UNSTAMPED)
            g["rounds"].add(rec.get("rounds") or d.get("rounds"))
            g["budgets"].add(rec.get("token_budget") or d.get("token_budget"))
            g["files"].add(str(f))
            key = (task_of(rec.get("case", "")), rec.get("case_id") or Path(rec.get("case", "")).name)
            # Which rep this record is, so an unresolved one is not covered by
            # another rep of the case. Records from before the harness stamped
            # `rep` are grouped by their results file, which is one run of one
            # rep (harness/run.py writes one record per case per file).
            rep = rec.get("rep", d.get("rep"))
            g["cases"].setdefault(key, {}).setdefault(
                rep if rep is not None else f"file {Path(f).name}", []).append(rec)
    out = []
    for (model, effort), g in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        keys = set(g["cases"])
        if want is not None:
            keys |= {(t, c) for t, cs in want.items() for c in cs}
        per_case, pending, costs, tin, tout, secs, sent = {}, [], [], [], [], [], set()
        din, dout, dcalls, estimated = [], [], [], 0
        for key in sorted(keys):
            reps = g["cases"].get(key, {})
            vals, unresolved = [], []
            for label, recs in sorted(reps.items(), key=lambda kv: str(kv[0])):
                got, why_rep = [], ""
                for r in recs:
                    if "effort_sent" in r:
                        sent.add(r["effort_sent"])
                    state, v, why_r = resolve(r)
                    if state != "pending":
                        got.append(v)
                    else:
                        why_rep = why_r
                    tok = r.get("tokens") or {}
                    c, est = usd_of(model, tok, table)     # billed: discarded replies cost money too
                    if c is not None:
                        costs.append(c); estimated += bool(est)
                    # Replies the harness discarded and asked again for are not in the
                    # token means (since 2.1 they do not count toward the budget either);
                    # they are reported apart, as discarded_*. Input of a discarded call
                    # is known only where tokens.per_call is recorded.
                    pc = tok.get("per_call") or []
                    d_in = sum(c_.get("input", 0) or 0 for c_ in pc if c_.get("discarded"))
                    d_out = tok.get("output_discarded", 0) or 0
                    tin.append((tok.get("input", 0) or 0) - d_in)
                    tout.append((tok.get("output", 0) or 0) - d_out)
                    din.append(d_in); dout.append(d_out); dcalls.append(tok.get("calls_discarded", 0) or 0)
                    if r.get("seconds"):
                        secs.append(float(r["seconds"]))
                if got:
                    vals += got                     # the rep resolved (either record of it did)
                else:
                    unresolved.append((label, why_rep))
            # A rep that did not resolve makes the case pending: it is never
            # covered by the reps that did (see the rule at the top).
            if vals and not unresolved:
                per_case[key] = statistics.mean(vals)
            else:
                pending.append({"task": key[0], "case": key[1],
                                "why": unresolved[0][1] if unresolved else "no record",
                                "reps": [str(label) for label, _ in unresolved], "n_reps": len(reps)})
        n = len(keys)
        tasks = sorted({k[0] for k in keys})
        per_task = {t: {"mean": (statistics.mean([v for k, v in per_case.items() if k[0] == t])
                                 if any(k[0] == t for k in per_case) else None),
                        "resolved": sum(1 for k in per_case if k[0] == t),
                        "cases": sum(1 for k in keys if k[0] == t)} for t in tasks}
        rounds = sorted(g["rounds"], key=lambda r: (r is not None, r or 0))   # None: no round limit
        budgets = sorted(b for b in g["budgets"] if b)
        versions = sorted(g["versions"])
        subset = n_core is not None and n < n_core
        complete = not pending and n > 0 and not subset
        out.append({
            "model": model, "effort": effort, "effort_sent": sorted(sent, key=str),
            "rounds": rounds[0] if len(rounds) == 1 else rounds,
            "token_budget": (budgets[0] if len(budgets) == 1 else budgets) if budgets else None,
            "harness_version": versions[0] if len(versions) == 1 else versions,
            "smoke": (g["rounds"] != {ROUNDS} or budgets != [TOKEN_BUDGET] or None in g["budgets"]
                      or len(versions) > 1),
            "n_cases": n, "n_core": n_core, "subset": subset,
            "resolved": len(per_case), "pending": pending, "complete": complete,
            "core_mean": statistics.mean(per_case.values()) if complete else None,
            "provisional_mean": statistics.mean(per_case.values()) if per_case else None,
            "per_task": per_task,
            "usd_per_case": (sum(costs) / n) if costs and n else None,
            "price": "list" if price_of(model, table) else "unknown",
            # records priced as an estimate (a tiered model, no per-call sizes)
            "usd_estimated": f"{estimated} of {len(costs)} records {ESTIMATED}" if estimated else None,
            "input_tokens_per_case": (sum(tin) / n) if n else None,
            "output_tokens_per_case": (sum(tout) / n) if n else None,
            "discarded_input_tokens_per_case": (sum(din) / n) if n else None,
            "discarded_output_tokens_per_case": (sum(dout) / n) if n else None,
            "discarded_calls_per_case": (sum(dcalls) / n) if n else None,
            "median_seconds_per_case": statistics.median(secs) if secs else None,
            "files": sorted(g["files"]),
        })
    return out


def fmt_rounds(r) -> str:
    return "no limit" if r is None else ", ".join(map(fmt_rounds, r)) if isinstance(r, list) else str(r)


def fmt_versions(v) -> str:
    return " + ".join(v) if isinstance(v, list) else str(v)


def fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def report(rows: list[dict], provisional: bool = False, reference: dict | None = None,
           public: bool = False) -> str:
    lines = []
    head, cases = ("PUBLIC MEAN", "Public Set") if public else ("CORE MEAN", "Core")
    for r in rows:
        lines.append(f"== {r['model']}  effort {r['effort']}"
                     + (f" (sent {', '.join(map(str, r['effort_sent']))})" if r["effort_sent"]
                        and r["effort_sent"] != [r["effort"]] else "") + f"  rounds {fmt_rounds(r['rounds'])}  harness {fmt_versions(r.get('harness_version', UNSTAMPED))}")
        if r["smoke"] and isinstance(r.get("harness_version"), list):
            lines.append(f"   SMOKE RUN, harness versions {fmt_versions(r['harness_version'])} mixed: not a "
                         f"reportable score (run every case from one tagged release)")
        elif r["smoke"]:
            lines.append(f"   SMOKE RUN, rounds={fmt_rounds(r['rounds'])}, token budget "
                         f"{r.get('token_budget') or 'none'}: not a reportable score (reported numbers use "
                         f"a {TOKEN_BUDGET:,} output-token budget and "
                         + ("no round limit)" if ROUNDS is None else f"{ROUNDS} rounds)"))
        if public:
            lines.append("   PUBLIC SET: for anyone to look at, not a reportable score (report Core)")
        of = f" ({r['n_cases']}/{r['n_core']} {cases} cases)" if r.get("subset") else ""
        if r["complete"]:
            lines.append(f"   {head} {r['core_mean']:.4f}   over {r['n_cases']} cases")
        elif r["pending"]:
            lines.append(f"   INCOMPLETE: {len(r['pending'])} of {r['n_cases']} cases pending{of}, no headline")
        else:
            lines.append(f"   SUBSET {r['n_cases']}/{r['n_core']}: these records cover {r['n_cases']} of the "
                         f"{r['n_core']} {cases} cases, no headline")
        if not r["complete"] and provisional and r["provisional_mean"] is not None:
            lines.append(f"   provisional mean over the {r['resolved']} resolved cases: "
                         f"{r['provisional_mean']:.4f} (NOT the headline)")
        if reference:
            ref = next((x for x in reference.get("lines", [])
                        if x.get("model") == r["model"] and x.get("effort") == r["effort"]), None)
            if ref:
                lines.append(f"   reference ({reference.get('version', '')}; harness "
                             f"{ref.get('harness_version') or reference.get('harness_version', UNSTAMPED)}, "
                             f"{reference.get('rounds', 30)} rounds, "
                             f"{reference.get('harness', 'run before 2026-10-15')}): {fmt(ref.get('core_mean'))}"
                             + (f"  [{ref['noise_note']}]" if ref.get("noise_note") else ""))
        lines.append("   " + "  ".join(f"{t} {fmt(v['mean'], 3)} ({v['resolved']}/{v['cases']})"
                                       for t, v in r["per_task"].items()))
        cost = (f"${fmt(r['usd_per_case'], 2)}" if r.get("price", "list") == "list"
                else f"price unknown for {r['model']}")
        if r.get("usd_estimated"):
            cost += f" ({r['usd_estimated']})"
        lines.append(f"   per case: {cost}  in {fmt(r['input_tokens_per_case'], 0)} tok  "
                     f"out {fmt(r['output_tokens_per_case'], 0)} tok  median {fmt(r['median_seconds_per_case'], 0)} s")
        if r.get("discarded_calls_per_case"):
            lines.append(f"   discarded and asked again, per case: {fmt(r['discarded_calls_per_case'], 2)} calls, "
                         f"out {fmt(r['discarded_output_tokens_per_case'], 0)} tok"
                         + (f", in {fmt(r['discarded_input_tokens_per_case'], 0)} tok" if r.get("discarded_input_tokens_per_case") else "")
                         + " (billed, not in the means above)")
        for p in r["pending"][:20]:
            # Which rep is missing, when the case has more than one: the other
            # reps resolving does not resolve the case.
            of = (f" (rep {', '.join(p['reps'])} of {p['n_reps']})"
                  if p.get("reps") and p.get("n_reps", 0) > 1 else "")
            lines.append(f"   pending {p['task']} {p['case']}{of}: {p['why']}")
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
    meta = {"harness_versions": record_versions(files), "harness_commits": revs,
            "checkout_commit": harness_commit()}
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
    print(f"harness {', '.join(sorted(meta['harness_versions']))} at {ran}  dataset "
          + (f"{meta['dataset']['name']} {meta['dataset']['version']}  scorer_digest "
             f"{str(meta['dataset']['scorer_digest'])[:12]}" if "dataset" in meta else "-"))
    public = (meta.get("dataset") or {}).get("name") == PUBLIC_SET
    print(report(rows, a.provisional, ref, public))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({**meta, **({"set": "public"} if public else {}), "efforts": rows},
                                     indent=1) + "\n")
        print(f"summary -> {a.json}")
    return 0 if all(r["complete"] for r in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
