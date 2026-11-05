#!/usr/bin/env python
"""BenchCAD 2.1 Core with your own agent, in one command (from a checkout of tag v2.1):

    python3 own_agent/benchcad.py run --agent claude-code --model anthropic/claude-opus-5-5 --effort high
    python3 own_agent/benchcad.py run --agent codex --model openai/gpt-6-astra --effort xhigh --cases task<N>/caseNNN

It does, in order, only what is not done yet:
  1. download the gated Core data (BenchCAD/benchcad-2.0-core, revision v2.1; an approved Hugging Face
     account's HF_TOKEN) to the same place tools/run_core.sh uses: $BENCHCAD_DATA, else
     ~/.cache/benchcad/benchcad-2.0-core;
  2. build the Harbor tasks from it with this checkout's 2.1 harness (run_build.sh, in Docker) to
     $BENCHCAD_CACHE/own-agent-tasks-<budget> (default ~/.cache/benchcad);
  3. `harbor run` them with the chosen agent, its model API pointed at the case's meter
     (http://meter:8080/<provider>, the agent's only way out, where the 500,000-token budget is kept);
  4. collect.py: results.jsonl (one row per trial) and summary.json (the line, by 2.1's rule) in the job folder;
  5. remove what holds Core files (unless --keep-tasks): the built tasks, and in each trial's artifacts/work
     everything but the agent's answer.py. Kept: result.json, the collect output, the trajectories, the meter's
     usage and request log, the verifier's record. Harbor removes the trial images itself (--delete, its default).

Under it is plain Harbor (BENCHCAD_HARBOR names the command, default `harbor`; 0.20.0 is the tested
version); --dry-run prints the commands. Each agent is given what Harbor's own installed agent
reads: its key from the environment (ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / LLM_API_KEY) and
the meter as its base URL. --effort goes to the agent's own effort setting (Harbor's reasoning_effort kwarg, which
all five take; Harbor itself refuses a value a model cannot take, e.g. any effort on a Gemini 2.5 model).
A model provider an agent has not been tested with through the meter is refused.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = Path(os.environ.get("BENCHCAD_CACHE", Path.home() / ".cache/benchcad"))
DATA = Path(os.environ.get("BENCHCAD_DATA", Path.home() / ".cache/benchcad/benchcad-2.0-core"))
DATA_REPO, DATA_REV = "BenchCAD/benchcad-2.0-core", "v2.1"
STANDARD_BUDGET = 500_000
METER = "http://meter:8080"
ROUTE = {"anthropic": f"{METER}/anthropic", "openai": f"{METER}/openai/v1", "gemini": f"{METER}/gemini"}

# agent -> how Harbor runs it, which providers it reaches through the meter (provider -> the base-URL
# variables its installed agent reads), its effort kwarg (None: Harbor has none), and its key variable
AGENTS = {
    "claude-code": {"harbor": ["-a", "claude-code"], "bare_model": True, "providers": {"anthropic": ["ANTHROPIC_BASE_URL"]},
                    "effort": "reasoning_effort", "keys": {"anthropic": "ANTHROPIC_API_KEY"}},
    "claude-code-bare": {"harbor": ["--agent", "agents.claude_code_bare:ClaudeCodeBare"], "bare_model": True,
                         "providers": {"anthropic": ["ANTHROPIC_BASE_URL"]},
                         "effort": "reasoning_effort", "keys": {"anthropic": "ANTHROPIC_API_KEY"}},
    # Codex runs on HTTP (agents/codex_http.py: its provider is the meter with supports_websockets = false):
    # Harbor's plain codex 0.166.0 opens the Responses WebSocket, which the meter refuses, and ends the turn.
    # Its name stays "codex"; the transport is in the job's run.json.
    "codex": {"harbor": ["--agent", "agents.codex_http:CodexHttp"], "providers": {"openai": ["OPENAI_BASE_URL"]},
              "effort": "reasoning_effort", "keys": {"openai": "OPENAI_API_KEY"},
              "transport": "http (codex provider = meter, supports_websockets = false)"},
    "gemini-cli": {"harbor": ["-a", "gemini-cli"], "providers": {"gemini": ["GOOGLE_GEMINI_BASE_URL"]},
                   "effort": "reasoning_effort", "keys": {"gemini": "GEMINI_API_KEY"}},
    "mini-swe-agent": {"harbor": ["-a", "mini-swe-agent"], "providers": {"openai": ["OPENAI_BASE_URL", "OPENAI_API_BASE"]},
                       "effort": "reasoning_effort", "keys": {"openai": "OPENAI_API_KEY"}},
    # Harbor 0.20.0's "openhands" installs the latest openhands-ai, which no longer has the openhands.core.main it
    # starts; "openhands-sdk" is the maintained OpenHands. Its effort defaults to high, so
    # the wrapper always names it and the record says what ran.
    "openhands": {"harbor": ["-a", "openhands-sdk"], "providers": {"openai": ["LLM_BASE_URL"]},
                  "effort": "reasoning_effort", "default_effort": "high", "keys": {"openai": "LLM_API_KEY"}},
}


def budget(s: str) -> int:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmM]?)", s.replace(",", "").replace("_", ""))
    if not m:
        raise argparse.ArgumentTypeError(f"{s!r}: a number of tokens, e.g. 500k")
    return int(float(m.group(1)) * {"": 1, "k": 1000, "m": 1_000_000}[m.group(2).lower()])


def sh(cmd: list[str], dry: bool, env: dict | None = None) -> None:
    shown = " ".join(f"{k}=…" if "KEY" in k else f"{k}={shlex.quote(v)}" for k, v in (env or {}).items())
    print("+", (shown + " " if shown else "") + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})


def ensure_data(dry: bool) -> Path:
    if (DATA / "MANIFEST.sha256").exists():
        return DATA                      # build.py checks every file it uses against the manifest
    token_file = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    if not os.environ.get("HF_TOKEN") and not token_file.exists() and not dry:
        raise SystemExit(f"no Hugging Face token: set HF_TOKEN (an account with access to {DATA_REPO}; "
                         "request it at https://benchcad.com/access.html)")
    sh(["uvx", "--quiet", "--from", "huggingface_hub", "hf", "download", DATA_REPO, "--repo-type", "dataset",
        "--revision", DATA_REV, "--local-dir", str(DATA)], dry, {"HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    return DATA


def head() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def lock(tokens: int, dry: bool):
    """One run at a time per task folder: a run rebuilds and removes it, so a second one would pull it away
    from under the first. Held until the process ends."""
    if dry:
        return None
    import fcntl
    CACHE.mkdir(parents=True, exist_ok=True)
    f = open(CACHE / f"own-agent-tasks-{tokens}.lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another benchcad.py run is using {CACHE / f'own-agent-tasks-{tokens}'}; wait for it, or "
                         "give this one its own BENCHCAD_CACHE")
    return f


def ensure_tasks(tokens: int, cases: str, dry: bool) -> Path:
    """The built tasks for this budget, reused only when they were built from this checkout's commit, hold
    every case asked for, and each one is still byte for byte what was built (its tree digest in the index)."""
    out = CACHE / f"own-agent-tasks-{tokens}"
    want = [c for c in cases.split(",") if c] if cases else None
    try:
        index = json.loads((out / "dataset.json").read_text())
    except (OSError, ValueError):
        index = {}
    built = {t["name"]: t["sha256"] for t in index.get("tasks", [])} if index.get("harness_commit") == head() else {}
    names = [f"t{c.split('/')[0][4:]}-{c.split('/')[1]}" for c in want] if want else sorted(built)
    if (want or len(built) == 100) and names and all(n in built for n in names):
        sys.path.insert(0, str(HERE))
        from build import tree_sha256
        if all((out / n).is_dir() and tree_sha256(out / n) == built[n] for n in names):
            return out
        print(f"note: {out} was changed since it was built; building again", file=sys.stderr)
    data = ensure_data(dry)
    sh(["rm", "-rf", str(out)], dry)
    sh([str(HERE / "run_build.sh"), str(data), str(out), "--budget", str(tokens)]
       + (["--cases", cases] if cases else []), dry)
    return out


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--agent", required=True, choices=sorted(AGENTS))
    r.add_argument("--model", required=True, help="provider/model-id, e.g. anthropic/claude-opus-5-5")
    r.add_argument("--effort", help="the agent's own effort setting (Harbor's reasoning_effort)")
    r.add_argument("--budget", type=budget, default=STANDARD_BUDGET, help="output tokens per case (default 500k)")
    r.add_argument("--cases", default="", help="task<N>/caseNNN,... (default: all 100)")
    r.add_argument("--passes", type=int, default=1, help="trials per case")
    r.add_argument("--n-concurrent", type=int, default=1)
    r.add_argument("--jobs-dir", type=Path, default=ROOT / "results/own_agent",
                   help="where Harbor writes the job (default results/own_agent/, beside run_core's results/core/)")
    r.add_argument("--job-name")
    r.add_argument("--keep-tasks", action="store_true",
                   help="keep the built tasks (they hold Core inputs and references) and the inputs in each "
                        "trial's artifacts/work; by default both are removed after collect, and a later run "
                        "rebuilds them from the downloaded data")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("harbor_args", nargs=argparse.REMAINDER, help="after --: passed to harbor run unchanged")
    args = a.parse_args()

    spec = AGENTS[args.agent]
    provider = args.model.split("/", 1)[0]
    if provider not in spec["providers"]:
        raise SystemExit(f"--agent {args.agent} reaches {', '.join(spec['providers'])} through the meter; "
                         f"{provider!r} has not been tested with it")
    if args.effort and not spec["effort"]:
        raise SystemExit(f"--agent {args.agent}: Harbor's agent has no effort setting, so --effort {args.effort} "
                         f"would not reach the model; leave it out")
    key = spec["keys"][provider]
    if not os.environ.get(key) and not args.dry_run:
        raise SystemExit(f"{key} is not set")
    if args.budget != STANDARD_BUDGET:
        print(f"note: budget {args.budget:,} is not 2.1's {STANDARD_BUDGET:,}: a plumbing check, not a reportable run",
              file=sys.stderr)

    held = lock(args.budget, args.dry_run)                    # noqa: F841 (held until exit)
    tasks = ensure_tasks(args.budget, args.cases, args.dry_run)
    env = {v: ROUTE[provider] for v in spec["providers"][provider]}
    env["PYTHONPATH"] = os.pathsep.join([str(HERE)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = args.job_name or (f"{args.agent}-{args.model.split('/')[-1]}-"
                            f"{args.effort or spec.get('default_effort') or 'default'}-{stamp}")
    # Harbor's claude-code sends "provider/model" as the model id once a base URL is set (it assumes a router
    # such as OpenRouter); the Anthropic API behind the meter takes the bare id
    model = args.model.split("/", 1)[1] if spec.get("bare_model") else args.model
    # --max-retries 0 is always sent: it overrides any retry setting a --config file might bring
    cmd = [*shlex.split(os.environ.get("BENCHCAD_HARBOR", "harbor")), "run", "-p", str(tasks), *spec["harbor"], "-m", model, "-o", str(args.jobs_dir),
           "--job-name", job, "-n", str(args.n_concurrent), "-k", str(args.passes), "--max-retries", "0", "-y"]
    effort = args.effort or spec.get("default_effort")
    if effort:
        cmd += ["--ak", f"{spec['effort']}={effort}"]
    # the base URL goes in the container too: not every installed agent forwards it from the host
    # (Harbor 0.20.0's gemini-cli does not), and --ae is what reaches the agent's process
    for v in spec["providers"][provider]:
        cmd += ["--ae", f"{v}={ROUTE[provider]}"]
    for c in [c for c in args.cases.split(",") if c]:
        cmd += ["-i", f"*t{c.split('/')[0][4:]}-{c.split('/')[1]}"]
    extra = args.harbor_args[1:] if args.harbor_args[:1] == ["--"] else args.harbor_args
    if any(x in ("-r", "--max-retries") or x.startswith("--max-retries=") for x in extra):
        # Harbor retries a trial whose agent ended on an API error -- which is how the meter ends a case at its
        # budget -- and a retry starts a fresh meter: the budget would no longer bound the case
        raise SystemExit("--max-retries is refused: a retried trial gets a fresh 500k budget. Re-run failed trials "
                         "by hand, after checking that the failure was not the budget.")
    sh(cmd + extra, args.dry_run, env)
    jd = args.jobs_dir / job
    run = {"agent": args.agent, "harbor_agent": " ".join(spec["harbor"]), "model": args.model, "model_sent": model,
           "effort": effort, "budget": args.budget, "reportable_budget": args.budget == STANDARD_BUDGET,
           "base_urls": {v: ROUTE[provider] for v in spec["providers"][provider]},
           "transport": spec.get("transport", "http"), "cases": args.cases or "all", "passes": args.passes}
    if not args.dry_run:
        (jd / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    sh([sys.executable, str(HERE / "collect.py"), str(jd), "--out", str(jd / "results.jsonl"),
        "--summary", str(jd / "summary.json")], args.dry_run)
    if not args.keep_tasks:
        cleanup(tasks, jd, args.dry_run)


IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".pdf"}


def cleanup(tasks: Path, job: Path, dry: bool) -> None:
    """Remove the Core copies a run leaves: the built tasks, and in each trial's /work copy the files the task
    staged there (by path) and every image or PDF (the agent's crops and renders of the drawings). What the agent
    made itself stays: answer.py, the modules it imports, its own geometry and notes."""
    import shutil
    work = sorted(job.glob("*/artifacts/work")) if job.exists() else []
    print(f"+ cleanup: remove {tasks}, and the staged inputs and images in {len(work)} trial artifacts/work", flush=True)
    if dry:
        return
    staged = {}
    for t in tasks.glob("*/environment/task") if tasks.exists() else []:
        staged[t.parent.parent.name] = [p.relative_to(t) for p in t.rglob("*")]
    for w in work:
        trial = w.parent.parent
        try:
            name = json.loads((trial / "result.json").read_text())["task_name"].split("/")[-1]
        except (OSError, ValueError, KeyError):
            name = trial.name.split("__")[0]
        for rel in sorted(staged.get(name, []), key=lambda r: len(r.parts), reverse=True):
            f = w / rel
            if f.is_symlink() or f.is_file():
                f.unlink(missing_ok=True)
            elif f.is_dir() and not any(f.iterdir()):
                f.rmdir()
        for f in sorted(w.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if f.suffix.lower() in IMAGES and (f.is_file() or f.is_symlink()):
                f.unlink(missing_ok=True)
    shutil.rmtree(tasks, ignore_errors=True)


if __name__ == "__main__":
    main()
