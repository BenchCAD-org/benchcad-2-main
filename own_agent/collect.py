#!/usr/bin/env python
"""One results row per trial (result.schema.json) from Harbor job directories, plus the line a run reports.

    python collect.py JOB_DIR [JOB_DIR ...] --out results.jsonl [--summary summary.json]

A row says what was scored and how it was got: the 2.1 cell (scored / no_submission / pending), wall time,
tokens and cost as the agent reported them, tool calls counted from its trajectory, the agent and its version,
the model and effort, the pass index, the meter's output-token count and whether the budget was hit, and the
pins of the task it ran (data, harness, scorer). Trajectories are
not copied: `trajectory` points at the trial's own agent/ folder, which is kept for audit.

The line follows 2.1's rule (tools/core_score.py): per (agent, model, effort), a case's passes are averaged,
then a plain mean over the cases; a pending cell means no headline (INCOMPLETE), and a run over fewer than
the 100 Core cases is a SUBSET. A cell is pending (no score, run it again) when the trial has no result, its
verifier could not finish, its /work or meter record is missing, the meter could not read a reply's usage, or
the agent ended on a provider error (an HTTP 429 or 5xx it did not get past: 2.1 retries those, here the case
is run again). A run whose trials do not all carry the 2.1 pins, or whose meter had another budget or
upstream, is NOT REPORTABLE.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

N_CORE = 100
STANDARD_BUDGET = 500_000
STANDARD_PINS = {"data_commit": "4b6fe17f63ef46090e372120d5863bcb91bef9de", "harness_version": "2.1",
                 "scorer_digest": "c4799cc240612b038362efc5f649fc05e4bf1ce33babec46f1237092a9a7137d"}
PIN_KEYS = ("data_commit", "harness_commit", "harness_version", "scorer_digest", "scorer_tree_sha256",
            "case_tree_sha256")
STANDARD_UPSTREAMS = {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com",
                      "gemini": "https://generativelanguage.googleapis.com"}


def _t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def _secs(timing):
    if not timing:
        return None
    a, b = _t(timing.get("started_at")), _t(timing.get("finished_at"))
    return round((b - a).total_seconds(), 1) if a and b else None


def tool_calls(trial: Path) -> int | None:
    """Tool calls in the agent's ATIF trajectory (agent/trajectory.json), embedded subagent trajectories
    (the trajectory's subagent_trajectories, nested) included."""
    tr = trial / "agent/trajectory.json"
    if not tr.exists():
        return None

    def walk(t) -> int:
        if not isinstance(t, dict):
            return 0
        return (sum(len(s.get("tool_calls") or []) for s in t.get("steps") or [])
                + sum(walk(sub) for sub in t.get("subagent_trajectories") or []))
    return walk(json.loads(tr.read_text()))


def provider_error(trial: Path) -> int | None:
    """The HTTP status of the last model request the meter forwarded, when it was a 429 or a 5xx (the meter's
    own 502 for an unreachable upstream included): the agent ended on a provider error, not on its own."""
    rq = trial / "artifacts/meter/requests.jsonl"
    if not rq.exists():
        return None
    last = None
    for line in rq.read_text().splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if not x.get("refused") and not x.get("blocked") and not x.get("websocket"):
            last = x
    st = (last or {}).get("status")
    return st if isinstance(st, int) and (st == 429 or st >= 500) else None


def missing(case: str, trial: str | None, why: str) -> dict:
    """A trial that should be there and has no result: pending."""
    return {"case": case, "task": None, "kind": None, "trial": trial, "started_at": None, "status": "pending",
            "score": None, "why": why, "agent": None, "agent_version": None, "model": None, "effort": None,
            "wall_time_s": None, "trial_time_s": None, "verifier_time_s": None, "timed_out": False,
            "exception": None, "tokens": {}, "output_tokens": None, "budget": None, "budget_hit": None,
            "refused_requests": None, "model_requests": None, "reportable": False, "websocket_refused": None,
            "cost_usd": None, "tool_calls": None, "pins": {}, "task_checksum": None, "trajectory": ""}


def row(trial: Path) -> dict | None:
    """One trial's row. The job's run.json (benchcad.py) names the wrapper's agent, which is the
    leaderboard's name for it, and how it reached the model."""
    rf = trial / "result.json"
    if not rf.exists():
        return missing(trial.name.split("__")[0], trial.name, "the trial has no result.json")
    r = json.loads(rf.read_text())
    cfg = r.get("config") or {}
    agent_cfg = cfg.get("agent") or {}
    kw = agent_cfg.get("kwargs") or {}
    info = r.get("agent_info") or {}
    ar = r.get("agent_result") or {}
    v = trial / "verifier"
    reward = json.loads((v / "reward.json").read_text()) if (v / "reward.json").exists() else None
    detail = json.loads((v / "detail.json").read_text()) if (v / "detail.json").exists() else {}
    exc = (r.get("exception_info") or {}).get("exception_type")
    collect_why = None
    mf = trial / "artifacts/manifest.json"
    work = [m for m in (json.loads(mf.read_text()) if mf.exists() else []) if m.get("source") == "/work"]
    if not work or work[0].get("status") != "ok":
        # /work never reached the verifier: its "no answer" is Harbor's, not the agent's
        status, score = "pending", None
        collect_why = "the agent's /work was not collected"
    elif reward is None:
        status, score = "pending", None
    elif reward.get("error"):
        status, score = "pending", None
    elif reward.get("no_answer"):
        status, score = "no_submission", 0.0
    else:
        status, score = "scored", float(reward["reward"])
    # the meter's own count is the budget's record (agent-reported tokens are kept beside it)
    um = trial / "artifacts/meter/usage.json"
    meter = json.loads(um.read_text()) if um.exists() else None
    if meter is None and status != "pending":
        status, score, collect_why = "pending", None, "no meter record: the budget cannot be shown to have held"
    elif meter and meter.get("usage_unread"):
        status, score, collect_why = "pending", None, f"{meter['usage_unread']} model replies whose usage the meter could not read"
    elif exc and exc != "AgentTimeoutError" and meter and not meter.get("budget_hit") and (pe := provider_error(trial)):
        status, score, collect_why = "pending", None, f"the agent ended on a provider error (HTTP {pe}, {exc})"
    why = collect_why or detail.get("why")
    pins = detail.get("pins") or {}
    model = (info.get("model_info") or {}).get("name") or agent_cfg.get("model_name")
    prov = (info.get("model_info") or {}).get("provider")
    return {
        "case": pins.get("case") or r.get("task_name", "").split("/")[-1],
        "task": pins.get("task"), "kind": pins.get("kind"),
        "trial": r.get("trial_name"), "started_at": r.get("started_at"),
        "status": status, "score": score, "why": why,
        "agent": info.get("name") or agent_cfg.get("name"), "agent_version": info.get("version"),
        "model": f"{prov}/{model}" if prov and model and "/" not in model else model,
        "effort": kw.get("reasoning_effort") or kw.get("effort"),
        "wall_time_s": _secs(r.get("agent_execution")),
        "trial_time_s": _secs({"started_at": r.get("started_at"), "finished_at": r.get("finished_at")}),
        "verifier_time_s": _secs(r.get("verifier")),
        "timed_out": exc == "AgentTimeoutError", "exception": exc,
        "tokens": {"input": ar.get("n_input_tokens"), "output": ar.get("n_output_tokens"),
                   "cached": ar.get("n_cache_tokens")},
        "output_tokens": None if meter is None else meter.get("output_tokens"),
        "budget": None if meter is None else meter.get("budget"),
        "budget_hit": None if meter is None else meter.get("budget_hit"),
        "refused_requests": None if meter is None else meter.get("refused"),
        "model_requests": None if meter is None else meter.get("generation_requests"),
        # a run whose meter had another budget or another upstream (harbor --extra-docker-compose can change
        # both) is a check, not a score to report
        "reportable": bool(meter) and meter.get("budget") == STANDARD_BUDGET
                      and (meter.get("upstreams") or STANDARD_UPSTREAMS) == STANDARD_UPSTREAMS
                      and all(pins.get(k) == v for k, v in STANDARD_PINS.items()),
        "max_in_flight": None if meter is None else meter.get("max_in_flight"),
        "over_budget": None if meter is None else meter.get("over_budget"),
        "blocked_requests": None if meter is None else meter.get("blocked"),
        "websocket_refused": None if meter is None else meter.get("websocket_refused"),
        "cost_usd": ar.get("cost_usd"),
        "tool_calls": tool_calls(trial),
        "pins": {k: pins.get(k) for k in PIN_KEYS},
        "task_checksum": r.get("task_checksum"),
        "trajectory": f"{trial.parent.name}/{trial.name}/agent",
    }


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("jobs", nargs="+", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--summary", type=Path)
    args = a.parse_args()
    rows = []
    for j in args.jobs:
        run = json.loads((j / "run.json").read_text()) if (j / "run.json").exists() else {}
        mine = []
        for t in sorted(p for p in j.iterdir() if p.is_dir()):
            mine.append(row(t))
        keys = {(x["agent"], x["model"], x["effort"]) for x in mine if x["agent"]}
        if len(keys) == 1:                       # a trial with no result belongs to the job's one line
            for x in mine:
                if not x["agent"]:
                    x["agent"], x["model"], x["effort"] = next(iter(keys))
        if run:
            # every case asked for gets its passes: a trial Harbor never wrote is a pending cell, not a gap
            asked = [c for c in (run.get("cases") or "").split(",") if c and c != "all"]
            names = {f"t{c.split('/')[0][4:]}-{c.split('/')[1]}" for c in asked} | {x["case"] for x in mine}
            for name in sorted(names):
                for _ in range(int(run.get("passes") or 1) - sum(x["case"] == name for x in mine)):
                    mine.append(missing(name, None, "a pass the run asked for has no trial"))
            for x in mine:
                # the wrapper's names are the line's: agent, model and effort as the run was asked for
                x["model_reported"] = x["model"]
                x["agent"], x["model"], x["effort"] = run["agent"], run.get("model") or x["model"], run.get("effort") or x["effort"]
                x["run"] = {k: run.get(k) for k in ("harbor_agent", "transport", "model_sent", "budget")}
        rows.extend(mine)
    rows.sort(key=lambda x: (x["agent"] or "", x["model"] or "", x["effort"] or "", x["case"], x["started_at"] or ""))
    seen = defaultdict(int)
    for x in rows:
        k = (x["agent"], x["model"], x["effort"], x["case"])
        x["pass_index"] = seen[k]
        seen[k] += 1
    args.out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    lines = defaultdict(lambda: defaultdict(list))
    for x in rows:
        lines[(x["agent"], x["model"], x["effort"])][x["case"]].append(x)
    out = []
    for (agent, model, effort), cases in sorted(lines.items(), key=lambda kv: tuple(str(s) for s in kv[0])):
        pending = sorted(c for c, xs in cases.items() if any(x["status"] == "pending" for x in xs))
        per_case = {c: statistics.mean(x["score"] for x in xs) for c, xs in cases.items() if c not in pending}
        n = len(cases)
        every = [x for xs in cases.values() for x in xs]
        # one line is one implementation: trials built from different pins are not averaged into it
        pinsets = {tuple(x["pins"].get(k) for k in PIN_KEYS if k != "case_tree_sha256") for x in every if x["pins"]}
        line = {"agent": agent, "model": model, "effort": effort, "n_cases": n, "pending": pending,
                "passes": sorted({len(xs) for xs in cases.values()}),
                "mean": round(statistics.mean(per_case.values()), 4) if per_case and not pending else None,
                "label": ("NOT REPORTABLE" if len(pinsets) > 1 or not all(x["reportable"] or x["status"] == "pending"
                                                                          for x in every)
                          else "INCOMPLETE" if pending else ("SUBSET" if n < N_CORE else "CORE")),
                "agent_versions": sorted({x["agent_version"] for x in every if x["agent_version"]}),
                "max_in_flight": max((x.get("max_in_flight") or 0 for x in every), default=0),
                # every trial that went past the budget, by how much (allowed: requests in flight then finish)
                "over_budget": [{"case": x["case"], "trial": x["trial"], "over_budget": x["over_budget"]}
                                for x in every if (x.get("over_budget") or 0) > 0],
                "usd_per_case": _mean([x["cost_usd"] for xs in cases.values() for x in xs]),
                "median_wall_time_s": _median([x["wall_time_s"] for xs in cases.values() for x in xs]),
                "timed_out": sum(x["timed_out"] for xs in cases.values() for x in xs),
                "budget_hit": sum(bool(x["budget_hit"]) for xs in cases.values() for x in xs),
                "mean_output_tokens": _mean([x["output_tokens"] for xs in cases.values() for x in xs])}
        out.append(line)
        print(f"{agent} {model} {effort}: {line['label']} n={n} mean={line['mean']} pending={len(pending)}")
    if args.summary:
        args.summary.write_text(json.dumps(out, indent=1) + "\n")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


if __name__ == "__main__":
    main()
