#!/usr/bin/env python
"""Build the BenchCAD 2.1 Core own-agent dataset: one Harbor task per Core case.

    python own_agent/build.py --data DATA --out OUT [--cases task<N>/caseNNN,...] [--budget N]

DATA     the gated Hugging Face dataset BenchCAD/benchcad-2.0-core at revision v2.1, as downloaded
         (task<N>/cases/caseNNN/{case.json,input,gt}, MANIFEST.sha256 checked here)
OUT      the task folders, OUT/<task>-<case>/. They hold Core case files (inputs and references): keep them
         on the machine that runs them and never publish them.

The 2.1 harness is this checkout (the repository root above own_agent/), at tag v2.1 and without local
changes. Its own code stages the inputs, writes the task text and scores, so a task says exactly what a
2.1 run says, minus the round limit and the fenced-block protocol, and it is scored by the 2.1 scorer.

Runs where the harness's own environment is (pymupdf renders the drawings): run_build.sh runs it in the
builder image (builder/Dockerfile, the 2.1 uv.lock), see README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # the 2.1 harness: this checkout
TEMPLATE = HERE / "template"
CORE_COMMIT = "4b6fe17f63ef46090e372120d5863bcb91bef9de"      # BenchCAD/benchcad-2.0-core revision v2.1
SCORER_DIGEST = "c4799cc240612b038362efc5f649fc05e4bf1ce33babec46f1237092a9a7137d"
TASKS = {"task1": "t1", "task2": "t2", "task3": "t3", "task4": "t4", "task5": "t5", "task6": "t6"}
AGENT_HOURS = 8                          # a runaway backstop only
TOKEN_BUDGET = 500_000                   # 2.1's budget, kept by the meter (template/environment/meter)
# What one answer run gets, as in 2.1 (envs.common.episode: "600 s, 2 GB, 1 CPU").
ANSWER_TIMEOUT_S = 600
# The scorer's time limit, as tools/run_core.sh sets it (BENCHCAD_SCORE_TIMEOUT_S=14400, which raises
# harness/run.py's SCORE_TIMEOUT_S and SCORE_TIMEOUT_ECAD_S alike). A score that runs out of it is no score
# (verified again), never a zero.
SCORE_TIMEOUT_S = 14400


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_sha256(root: Path, rel: list[str] | None = None) -> str:
    """Sorted relative paths with each file's sha256: the digest of a folder."""
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for p in files:
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + sha256(p).encode() + b"\n")
    return h.hexdigest()


def check_manifest(data: Path, cases: list[Path]) -> None:
    want = {}
    for line in (data / "MANIFEST.sha256").read_text().splitlines():
        digest, path = line.split(None, 1)
        want[path.strip()] = digest
    for c in cases:
        for f in sorted(c.rglob("*")):
            if f.is_file():
                rel = f.relative_to(data).as_posix()
                if want.get(rel) != sha256(f):
                    raise SystemExit(f"{rel}: not as MANIFEST.sha256 lists it (data revision other than v2.1?)")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


def harness_rev() -> tuple[str, str | None]:
    """This checkout's commit and its tag (None when HEAD is not tagged). Local changes to tracked files are
    refused: the tasks pin the commit, so it has to be what they were built from."""
    rev = git("rev-parse", "HEAD")
    if not rev:
        raise SystemExit(f"{ROOT}: not a git checkout (git clone the repository, then git checkout v2.1)")
    # tracked files anywhere, and untracked ones where the scorer and the sandbox come from
    if git("status", "--porcelain", "--untracked-files=no") or git(
            "status", "--porcelain", "--", "envs", "sandbox", "tools", "harness", "pyproject.toml", "uv.lock"):
        raise SystemExit(f"{ROOT}: has local changes; build from a clean checkout of tag v2.1")
    return rev, git("describe", "--tags", "--exact-match", "HEAD") or None


def instruction(case: Path, staged: Path) -> str:
    """The 2.1 system prompt (envs.common.episode.SYSTEM) for this case, with its budget and its
    fenced-block protocol replaced by the file answer contract. The task brief, directory listing, tools
    block and answer wording are the harness's own functions, unchanged."""
    from envs.common import episode as ep
    kind = ep._kind(case)
    seeds = [p.relative_to(staged).as_posix() for p in ep._seed_images(staged)]
    listing = "\n".join(f"  {n}" for n in ep._listing(staged))
    what = ep._submit_what(case).replace("\n             ", " ")
    return f"""You are solving a CAD task in a working directory, `/work`, over {ep.limits_text(None, TOKEN_BUDGET)}.

{ep._task_brief(case).strip()}

Directory (`/work`; `/task` holds the same files as they were given, read-only):
{listing}
tools.py (`import tools`; `/work` is on PYTHONPATH):
{ep.tools_help(case)}

Python 3.12: cadquery {ep.SANDBOX_CQ_VERSION} (cadquery-ocp 7.9), numpy, scipy, PIL,
trimesh, vtk, matplotlib, ezdxf. No network beyond your own model's API.

The input images: {", ".join(seeds) if seeds else "none"}.

Your answer is `/work/answer.py`: {what}
When you stop, it is run once, exactly as written, as a fresh `python` process in a copy of `/work`
as you left it (any final.step, pred_graph.json or submission/ in it removed first, tools.py and
sitecustomize.py restored), with this appended:

    import tools as _harness_tools
    _harness_tools.finish(globals())

That run has {ANSWER_TIMEOUT_S} s, one CPU and no network. What it leaves is what is scored. If there is no
`/work/answer.py`, or it does not run, or it leaves no answer, the case scores 0.

The budget counts every model call of the run, a subagent's too. Once it is spent, no further call is
served and the `/work/answer.py` that exists then is scored. `curl -s http://meter:8080/_meter/status`
shows the output tokens spent so far. The run also stops after {AGENT_HOURS} hours of wall-clock time.
"""


def oracle(case: Path, sol: Path) -> str:
    """solution/: the reference answer as an answer.py, built the way harness/run.py's mock/oracle builds
    its ```submit (the reference files are copied next to it instead of inlined)."""
    gt_step, gt_graph = case / "gt/gt.step", case / "gt/gt_graph.json"
    ref = sol / "_oracle"
    ref.mkdir(parents=True)
    if gt_graph.exists():
        from envs.verifiers.ecad import position_mode, position_reference
        r = position_reference(case) if position_mode(case) else json.loads(gt_graph.read_text())
        (ref / "ref.json").write_text(json.dumps(r) + "\n")
        return "import json\nresult = json.loads(open('_oracle/ref.json').read())\n"
    if (case / "gt/instances.json").exists():
        inst = json.loads((case / "gt/instances.json").read_text())["instances"]
        lines = ["import tools"]
        for pid in sorted({r["part_id"] for r in inst}):
            posed = case / "gt/parts" / f"{pid}.step"
            if not posed.exists() and (case / "input/step_files" / f"{pid}.step").exists():
                lines.append(f"tools.use_part({pid!r})")
            else:
                shutil.copy(posed, ref / f"{pid}.step")
                lines.append(f"tools.export_part('_oracle/{pid}.step', {pid!r})")
        recs = [{"part_id": r["part_id"], "instance_id": r["instance_id"], "transform": r["T"]} for r in inst]
        lines.append(f"tools.submit_assembly({json.dumps(recs)})")
        return "\n".join(lines) + "\n"
    shutil.copy(gt_step, ref / "ref.step")
    return "import cadquery as cq\nresult = cq.importers.importStep('_oracle/ref.step')\n"


def run_py_constants() -> dict:
    """HARNESS_VERSION and SCORER_CHILD (the code score_in_subprocess runs) from harness/run.py's source,
    with the module-level constants SCORER_CHILD names (run.py itself imports the provider SDKs)."""
    import ast
    tree = ast.parse((ROOT / "harness/run.py").read_text())
    ns: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("HARNESS_VERSION", "MEMORY_ERROR_EXIT", "MEMORY_ERROR_MARK", "SCORER_CHILD"):
                ns[name] = eval(compile(ast.Expression(node.value), "run.py", "eval"), {}, ns)
    if "SCORER_CHILD" not in ns or ns.get("MEMORY_ERROR_EXIT") != 87:
        raise SystemExit(f"{ROOT}/harness/run.py: SCORER_CHILD not found as expected")
    if ns.get("HARNESS_VERSION") != "2.1":
        raise SystemExit(f"{ROOT}: harness version {ns.get('HARNESS_VERSION')!r}, not 2.1; git checkout v2.1")
    return ns


def fill(text: str, **kw) -> str:
    for k, v in kw.items():
        text = text.replace("{{" + k + "}}", str(v))
    if "{{" in text:
        raise SystemExit(f"unfilled template field in: {text[text.index('{{'):][:60]}")
    return text


def build_case(data: Path, task: str, cid: str, out: Path, meta: dict) -> dict:
    from envs.common.sandbox import Sandbox
    from envs.common.score_case import load_task
    case = data / task / "cases" / cid
    cj = json.loads((case / "case.json").read_text())
    kind = cj["kind"]
    name = f"{TASKS[task]}-{cid}"
    d = out / name
    if d.exists():
        raise SystemExit(f"{d} exists; build into an empty OUT")
    shutil.copytree(TEMPLATE, d, ignore=shutil.ignore_patterns("__pycache__"))
    for sub in ("tests/scorer", "tests/sandbox", "environment/sandbox"):     # filled below; git keeps no empty folder
        (d / sub).mkdir(parents=True, exist_ok=True)
    # inputs, staged by the 2.1 sandbox: PDFs rendered to sheet PNGs, tiles and hires/ masters, bom.json
    # pointed at the PNGs, tools.py and sitecustomize.py written; gt/ and case.json never come across
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["CADENV_LOCAL"] = "1"
        box = Sandbox(case, Path(tmp) / "work")
        staged = d / "environment/task"
        shutil.copytree(box.dir, staged)
        text = instruction(case, staged)
    leaks = [p for p in staged.rglob("*") if p.name in ("gt", "case.json") or p.suffix == ".pdf"]
    if leaks:
        raise SystemExit(f"{name}: staged inputs hold {leaks}")
    (d / "instruction.md").write_text(text)
    # the verifier's copy of the case (case.json, input/, gt/), the 2.1 scorer and its lock
    shutil.copytree(case, d / "tests/case")
    shutil.copytree(ROOT / "envs", d / "tests/scorer/envs", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / f, d / "tests/scorer" / f)
    (d / "tests/scorer/tools").mkdir()
    shutil.copy(ROOT / "tools/core_score.py", d / "tests/scorer/tools/core_score.py")
    (d / "tests/scorer/harness_scorer_child.py").write_text(meta["scorer_child"])
    for f in ("requirements.txt", "sitecustomize.py"):
        shutil.copy(ROOT / "sandbox" / f, d / "tests/sandbox" / f)
        shutil.copy(ROOT / "sandbox" / f, d / "environment/sandbox" / f)
    shutil.copy(staged / "tools.py", d / "tests/run/tools.py")
    shutil.copy(staged / "sitecustomize.py", d / "tests/run/sitecustomize.py")
    # solution/: the reference answer
    sol = d / "solution"
    (sol / "answer.py").write_text(oracle(case, sol))
    t = load_task(case) or {}
    verify_s = ANSWER_TIMEOUT_S + SCORE_TIMEOUT_S + 600
    scorer_tree = tree_sha256(d / "tests/scorer")
    pins = {"case": name, "task": task, "case_id": cid, "env": cj["env"], "kind": kind,
            "data_repo": "BenchCAD/benchcad-2.0-core", "data_revision": "v2.1", "data_commit": CORE_COMMIT,
            "harness_repo": "BenchCAD-org/benchcad-2-main", "harness_tag": meta["harness_tag"], "harness_commit": meta["harness_commit"],
            "harness_version": "2.1", "scorer_digest": SCORER_DIGEST, "scorer_tree_sha256": scorer_tree,
            "case_tree_sha256": tree_sha256(d / "tests/case"), "answer_timeout_s": ANSWER_TIMEOUT_S,
            "score_timeout_s": SCORE_TIMEOUT_S,
            "agent_timeout_s": AGENT_HOURS * 3600, "token_budget": TOKEN_BUDGET, "metric": (t.get("verify") or {}).get("metric")}
    (d / "tests/pins.json").write_text(json.dumps(pins, indent=1) + "\n")
    dc = d / "environment/docker-compose.yaml"
    dc.write_text(fill(dc.read_text(), token_budget=TOKEN_BUDGET))
    mem = meta["verifier_memory_mb"].get(name, meta["verifier_memory_mb"]["default"])
    (d / "task.toml").write_text(fill((d / "task.toml").read_text(), name=name, kind=kind, task=TASKS[task],
                                      agent_timeout=AGENT_HOURS * 3600, verify_timeout=verify_s,
                                      verifier_memory_mb=mem))
    return {"name": name, "kind": kind, "sha256": tree_sha256(d), "inputs_sha256": tree_sha256(staged),
            "verifier_memory_mb": mem}


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--cases", default="", help="task<N>/caseNNN,... (default: all 100)")
    a.add_argument("--budget", type=int, default=TOKEN_BUDGET,
                   help=f"output tokens per case, thinking included (default {TOKEN_BUDGET:,}, 2.1's; anything else "
                        "is a plumbing check, not a reportable run)")
    args = a.parse_args()
    globals()["TOKEN_BUDGET"] = args.budget
    data, out = args.data.resolve(), args.out.resolve()
    ds = json.loads((data / "dataset.json").read_text())
    if ds.get("scorer_digest") != SCORER_DIGEST:
        raise SystemExit(f"dataset.json scorer_digest {ds.get('scorer_digest')} is not 2.1's {SCORER_DIGEST}")
    if (ds.get("name"), ds.get("version")) != ("benchcad-2.0-core", "1.0"):
        raise SystemExit(f"{data}: dataset {ds.get('name')} {ds.get('version')}, not benchcad-2.0-core 1.0")
    rel = json.loads((ROOT / "releases/benchcad-2.1.json").read_text())
    if rel.get("scorer_digest") != SCORER_DIGEST or rel["datasets"]["core"]["commit"] != CORE_COMMIT:
        raise SystemExit(f"{ROOT}: not the 2.1 release")
    consts = run_py_constants()
    sys.path.insert(0, str(ROOT))
    from envs.common.ecad_graph.spatial_reference import scorer_digest
    if scorer_digest() != SCORER_DIGEST:
        raise SystemExit(f"{ROOT}: scorer digest {scorer_digest()} is not 2.1's")
    want = [c.strip() for c in args.cases.split(",") if c.strip()]
    if want:
        cases = [data / w.split("/")[0] / "cases" / w.split("/")[1] for w in want]
    else:
        cases = sorted(data.glob("task[1-6]/cases/case*"))
    missing = [c for c in cases if not (c / "case.json").exists()]
    if missing:
        raise SystemExit(f"no such case: {missing}")
    check_manifest(data, cases)
    rev, tag = harness_rev()
    meta = {"harness_commit": rev, "harness_tag": tag, "scorer_child": consts["SCORER_CHILD"],
            "verifier_memory_mb": json.loads((HERE / "verifier_memory.json").read_text())}
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cases:
        r = build_case(data, c.parent.parent.name, c.name, out, meta)
        rows.append(r)
        print(f"{r['name']:12s} {r['kind']:8s} {r['sha256'][:12]}", flush=True)
    index = {"dataset": "benchcad-2.1-core-own-agent", "n_tasks": len(rows), "token_budget": TOKEN_BUDGET, "data_commit": CORE_COMMIT,
             "harness_commit": meta["harness_commit"], "harness_tag": tag, "scorer_digest": SCORER_DIGEST, "tasks": rows}
    (out / "dataset.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
