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
the 100 Core cases is a SUBSET.
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
    """Tool calls in the agent's ATIF trajectory (agent/trajectory.json), subagents' steps included."""
    tr = trial / "agent/trajectory.json"
    if not tr.exists():
        return None
    n = 0

    def walk(steps):
        nonlocal n
        for s in steps or []:
            n += len(s.get("tool_calls") or [])
            for sub in s.get("subagent_trajectories") or []:
                walk(sub.get("steps"))
    d = json.loads(tr.read_text())
    walk(d.get("steps"))
    return n


def row(trial: Path) -> dict | None:
    """One trial's row. The job's run.json (benchcad.py) names the wrapper's agent, which is the
    leaderboard's name for it, and how it reached the model."""
    rf = trial / "result.json"
    if not rf.exists():
        return None
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
                      and (meter.get("upstreams") or STANDARD_UPSTREAMS) == STANDARD_UPSTREAMS,
        "websocket_refused": None if meter is None else meter.get("websocket_refused"),
        "cost_usd": ar.get("cost_usd"),
        "tool_calls": tool_calls(trial),
        "pins": {k: pins.get(k) for k in ("data_commit", "harness_commit", "harness_version", "scorer_digest",
                                          "scorer_tree_sha256", "case_tree_sha256")},
        "task_checksum": r.get("task_checksum"),
        "trajectory": str((trial / "agent").resolve()),
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
        for t in sorted(p for p in j.iterdir() if p.is_dir()):
            if (x := row(t)):
                if run:
                    x["agent"] = run["agent"]
                    x["run"] = {k: run.get(k) for k in ("harbor_agent", "transport", "model_sent", "budget")}
                rows.append(x)
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
        line = {"agent": agent, "model": model, "effort": effort, "n_cases": n, "pending": pending,
                "passes": sorted({len(xs) for xs in cases.values()}),
                "mean": round(statistics.mean(per_case.values()), 4) if per_case and not pending else None,
                "label": ("NOT REPORTABLE" if not all(x["reportable"] for xs in cases.values() for x in xs)
                          else "INCOMPLETE" if pending else ("SUBSET" if n < N_CORE else "CORE")),
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
