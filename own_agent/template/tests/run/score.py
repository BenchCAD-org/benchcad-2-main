"""Score what the agent's answer left, with the 2.1 scorer, and write the verifier's files.

    /opt/scorer/bin/python -s /tests/run/score.py RUN_DIR RC PRIVATE_DIR

RUN_DIR holds what answer.py left, as its own uid copied it out (test.sh, collect_answer.py), RC its exit
code, PRIVATE_DIR the scorer's own HOME, TMPDIR and caches, made fresh after /tmp was emptied. The answer is
what harness/run.py takes from a submitted program (envs.common.episode._artifact): a complete submission/
directory, else final.step, else pred_graph.json. It is scored as harness/run.py scores
(score_in_subprocess: envs.common.score_case in a child interpreter, the 2.1 time limit, out-of-memory told
apart from a score) and the cell is resolved by 2.1's own rule (tools/core_score.resolve): scored, no
submission (= 0), or pending (no score: the verifier's failure, re-verify it).

A submission that the scorer cannot score is the answer's own 0, not a pending cell: only the scorer running
out of memory or time, or failing to import, is the verifier's. Otherwise an answer could buy itself a fresh
budget by crashing the scorer. The reason is recorded either way, and collect.py lists such cases.

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

MAX_JSON = 32 << 20                      # a prediction larger than this, or nested deeper, is malformed
MAX_DEPTH = 64
# the scorer's own failures that are ours, not the answer's: everything else is scored 0
INFRA_ERRORS = ("MemoryError", "ImportError", "ModuleNotFoundError")

pins = json.loads((T / "pins.json").read_text())
run, rc = Path(sys.argv[1]), int(sys.argv[2])
private = Path(sys.argv[3] if len(sys.argv) > 3 else "/run/private")
for sub in ("home", "tmp", "mpl", "cache", "config"):
    (private / sub).mkdir(parents=True, exist_ok=True)
detail = {"pins": pins, "answer_rc": rc}


def malformed(path: Path) -> str | None:
    """Why this JSON prediction cannot be scored, or None. The bytes are scanned, never parsed: a deeply
    nested document raises RecursionError inside json.loads itself."""
    size = path.stat().st_size
    if size > MAX_JSON:
        return f"{path.name} is {size} bytes, over the {MAX_JSON} the scorer takes"
    depth = top = 0
    in_str = esc = False
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            for c in chunk:
                if in_str:
                    if esc:
                        esc = False
                    elif c == 0x5C:
                        esc = True
                    elif c == 0x22:
                        in_str = False
                elif c == 0x22:
                    in_str = True
                elif c in (0x7B, 0x5B):
                    depth += 1
                    top = max(top, depth)
                elif c in (0x7D, 0x5D):
                    depth -= 1
    if top > MAX_DEPTH:
        return f"{path.name} nests {top} deep, over the {MAX_DEPTH} the scorer takes"
    return None


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

if tree_sha256(T / "case") != pins["case_tree_sha256"]:
    write(0, 1, 0, status="pending", why="the case in this image is not the one the task pins")

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
if artifact is not None and any(p.is_symlink() for p in [artifact, *(artifact.rglob("*") if artifact.is_dir() else [])]):
    write(0, 1, 0, status="pending", why="the collected answer holds a link")      # collect_answer.py makes none
detail["artifact"] = None if artifact is None else artifact.relative_to(run).as_posix()
if artifact is None:
    # 2.1: a submitted program that does not run, or leaves no answer, is no answer (record score None)
    v = resolve({"case": pins["task"], "score": None})
    why = "no /work/answer.py" if rc == -1 else (f"answer.py failed (exit {rc})" if rc else "answer.py left no answer")
    write(0, 0, 1, status=v[0], why=why, score=None)

# a prediction the scorer cannot read is the answer's 0, decided before the scorer sees it
for f in ([artifact] if artifact.suffix == ".json" else sorted(artifact.rglob("*.json"))
          if artifact.is_dir() else []):
    if (why := malformed(f)):
        v = resolve({"case": pins["task"], "score": None})
        write(0, 0, 1, status=v[0], why=f"the answer cannot be scored: {why}", score=None)

# score_in_subprocess, as harness/run.py runs it (the child's code is the harness's own SCORER_CHILD), with
# its own HOME, TMPDIR and caches: it shares the container with a run by another user, so it shares no state
child = (SCORER / "harness_scorer_child.py").read_text()
limit = pins["score_timeout_s"]          # tools/run_core.sh's BENCHCAD_SCORE_TIMEOUT_S
out_f, err_f = private / "scorer.stdout", private / "scorer.stderr"
env = {"PATH": "/opt/scorer/bin:/usr/bin:/bin", "PYTHONPATH": str(SCORER), "PYTHONNOUSERSITE": "1",
       "HOME": str(private / "home"), "TMPDIR": str(private / "tmp"), "MPLCONFIGDIR": str(private / "mpl"),
       "XDG_CACHE_HOME": str(private / "cache"), "XDG_CONFIG_HOME": str(private / "config")}
try:
    with open(out_f, "wb") as o, open(err_f, "wb") as e:
        rcode = subprocess.run([sys.executable, "-s", "-c", child, str(T / "case"), str(artifact)], stdout=o,
                               stderr=e, timeout=limit, cwd=str(SCORER), env=env).returncode
except subprocess.TimeoutExpired:
    write(0, 1, 0, status="pending", why=f"scorer timed out after {limit} s")


def tail(path: Path, n: int = 65536) -> str:
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - n))
        return f.read().decode("utf-8", "replace")


err = tail(err_f)
detail["scorer_stderr_tail"] = err.strip()[-4096:]
if rcode == 87 and "BENCHCAD_SCORER_MEMORY_ERROR" in err:
    write(0, 1, 0, status="pending", why="memory_budget_exceeded: the scorer ran out of memory")
if rcode != 0:
    last = [l for l in err.strip().splitlines() if l and not l[0].isspace()]
    kind = (last[-1].split(":")[0].split(".")[-1] if last else "")
    if kind in INFRA_ERRORS:
        write(0, 1, 0, status="pending", why=f"the scorer could not run: {kind}")
    write(0, 0, 1, status="no_submission", score=None, scorer_failed=kind or f"exit {rcode}",
          why=f"the answer could not be scored: {kind or f'the scorer exited {rcode}'}")
try:
    score = json.loads(tail(out_f).strip().splitlines()[-1])
except (ValueError, IndexError):
    write(0, 1, 0, status="pending", why="the scorer printed no score")
status, value, why = resolve({"case": pins["task"], "score": score})
if status == "scored":
    write(value, 0, 0, status=status, why=why, score=score)
if status == "no_submission":
    write(0, 0, 1, status=status, why=why, score=score)
write(0, 1, 0, status=status, why=why, score=score)
