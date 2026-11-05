"""Score what the agent's answer left, with the 2.1 scorer, and write the verifier's files.

    /opt/scorer/bin/python /tests/run/score.py RUN_DIR RC

RUN_DIR is where answer.py ran (test.sh), RC its exit code. The answer is what harness/run.py takes from a
submitted program (envs.common.episode._artifact): a complete submission/ directory, else final.step, else
pred_graph.json. It is scored as harness/run.py scores (score_in_subprocess: envs.common.score_case in a child
interpreter, the 2.1 time limit, out-of-memory told apart from a score) and the cell is resolved by 2.1's own
rule (tools/core_score.resolve): scored, no submission (= 0), or pending (no score: the verifier's failure,
re-verify it).

/logs/verifier/reward.json  reward (the cell's value, 0 when there is none), error (1: pending, not a score),
                            no_answer (1: the case scores 0 because there is no answer)
/logs/verifier/detail.json  the pins, the resolution and the scorer's whole record
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

T = Path("/tests")
V = Path("/logs/verifier")
SCORER = T / "scorer"
sys.path.insert(0, str(SCORER))
sys.path.insert(0, str(SCORER / "tools"))

pins = json.loads((T / "pins.json").read_text())
run, rc = Path(sys.argv[1]), int(sys.argv[2])
detail = {"pins": pins, "answer_rc": rc}


def write(reward: float, error: int, no_answer: int, **kw) -> None:
    V.mkdir(parents=True, exist_ok=True)
    detail.update(kw)
    (V / "reward.json").write_text(json.dumps({"reward": round(float(reward), 6), "error": error,
                                               "no_answer": no_answer}) + "\n")
    (V / "detail.json").write_text(json.dumps(detail, indent=1, default=str) + "\n")
    if error or no_answer:
        (V / "reward.txt").write_text("0\n")
    print(json.dumps({"reward": reward, "error": error, "no_answer": no_answer, "status": detail.get("status"),
                      "why": detail.get("why")}))
    sys.exit(0)


def tree_sha256(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"\n")
    return h.hexdigest()


# the scorer is the one this task was built with, and the 2.1 one
if tree_sha256(SCORER) != pins["scorer_tree_sha256"]:
    write(0, 1, 0, status="pending", why="the scorer in this image is not the one the task pins")
from envs.common.ecad_graph.spatial_reference import scorer_digest  # noqa: E402
if scorer_digest() != pins["scorer_digest"]:
    write(0, 1, 0, status="pending", why=f"scorer digest {scorer_digest()} is not {pins['scorer_digest']}")

from envs.common.submission import SUB_ROOT, is_ready  # noqa: E402
from core_score import resolve  # noqa: E402

if rc == -97:
    # launch.py could not start the answer run: the verifier's failure, not the answer's
    write(0, 1, 0, status="pending", why="the answer run could not be started (launch failed)")
artifact = None
if rc == 0:
    if is_ready(run / SUB_ROOT):
        artifact = run / SUB_ROOT
    else:
        for name in ("final.step", "pred_graph.json"):
            if (run / name).exists():
                artifact = run / name
                break
detail["artifact"] = None if artifact is None else artifact.relative_to(run).as_posix()
if artifact is None:
    # 2.1: a submitted program that does not run, or leaves no answer, is no answer (record score None)
    v = resolve({"case": pins["task"], "score": None})
    why = "no /work/answer.py" if rc == -1 else (f"answer.py failed (exit {rc})" if rc else "answer.py left no answer")
    write(0, 0, 1, status=v[0], why=why, score=None)

# score_in_subprocess, as harness/run.py runs it (the child's code is the harness's own SCORER_CHILD)
child = (SCORER / "harness_scorer_child.py").read_text()
limit = pins["score_timeout_s"]          # tools/run_core.sh's BENCHCAD_SCORE_TIMEOUT_S
try:
    r = subprocess.run([sys.executable, "-c", child, str(T / "case"), str(artifact)], capture_output=True,
                       text=True, timeout=limit, cwd=str(SCORER), env={"PATH": "/opt/scorer/bin:/usr/bin:/bin",
                                                                       "PYTHONPATH": str(SCORER), "HOME": "/tmp"})
except subprocess.TimeoutExpired:
    write(0, 1, 0, status="pending", why=f"scorer timed out after {limit} s")
detail["scorer_stderr_tail"] = r.stderr.strip()[-4096:]
if r.returncode == 87 and "BENCHCAD_SCORER_MEMORY_ERROR" in r.stderr:
    write(0, 1, 0, status="pending", why="memory_budget_exceeded: the scorer ran out of memory")
if r.returncode != 0:
    write(0, 1, 0, status="pending", why=f"scorer exited {r.returncode}")
score = json.loads(r.stdout.strip().splitlines()[-1])
status, value, why = resolve({"case": pins["task"], "score": score})
if status == "scored":
    write(value, 0, 0, status=status, why=why, score=score)
if status == "no_submission":
    write(0, 0, 1, status=status, why=why, score=score)
write(0, 1, 0, status=status, why=why, score=score)
