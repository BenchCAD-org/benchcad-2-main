"""A MemoryError while scoring is the machine's, never the answer's.

Under the change 244 cgroup cap the kernel ends an over-budget scorer and the record
says memory_budget_exceeded. A MemoryError raised in-process (prlimit --as, one
allocation past the overcommit limit) took another road: the scorer's broad
`except Exception` handlers caught it and asm_v1.surface_indices returned an
empty surface, score.py's legacy iou scored the submission 0.0. And the
harness dropped a successful scorer's stderr, so the "unusable: MemoryError"
line left no trace in the record.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import run  # noqa: E402

# agent-side modules: their handlers guard the episode, not a score
NOT_SCORING = {"envs/common/sandbox.py", "envs/common/episode.py"}
# handlers that may stay broad, by (file, enclosing function): meshguard's worker
# loop answers the client with the error (the client raises it, below) and its
# close() only cleans up; the ECAD verifier CLI's handlers end in evaluator_error,
# which carries no score
MAY_CATCH = {("envs/geom/meshguard.py", "_serve"), ("envs/geom/meshguard.py", "close"),
             ("envs/common/ecad_graph/verifier.py", "main")}
BROAD = {"Exception", "BaseException", "<bare>"}
OOM = {"MemoryError", "Standard_OutOfMemory", "OOM_ERRORS"}
# pure-Python code with no OCCT in it may guard with MemoryError alone (the
# vendored ECAD matcher, kept byte-identical with ecad's own copy)
MEMORYERROR_ENOUGH = ("envs/common/ecad_graph/",)


def _module_tuples(tree) -> dict:
    """Module-level `NAME = <exception or tuple of them>`, so a handler naming
    the variable is read as what it holds."""
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out[n.targets[0].id] = n.value
    return out


def _caught(t, env=None, depth=0) -> set:
    if t is None:
        return {"<bare>"}
    if isinstance(t, ast.Tuple):
        return set().union(*(_caught(e, env, depth) for e in t.elts))
    if isinstance(t, ast.Name):
        if env and t.id in env and depth < 5:
            return {t.id} | _caught(env[t.id], env, depth + 1)
        return {t.id}
    if isinstance(t, ast.Attribute):
        return {t.attr}
    return set()


def _body_nodes(h):
    """The handler's statements, without the bodies of functions or classes defined in it."""
    stack = list(h.body)
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(n))


def _reraises(h) -> bool:
    """Every way out of the handler is the caught exception, re-raised: it ends in
    a bare `raise` and has no return, break, continue or other raise before it."""
    if not (h.body and isinstance(h.body[-1], ast.Raise) and h.body[-1].exc is None):
        return False
    return not any(isinstance(n, (ast.Return, ast.Break, ast.Continue))
                   or (isinstance(n, ast.Raise) and n.exc is not None) for n in _body_nodes(h))


def _is_guard(h, c: set, rel: str) -> bool:
    """`except <OOM tuple>[, others]: raise` -- one bare raise and nothing else."""
    single = len(h.body) == 1 and isinstance(h.body[0], ast.Raise) and h.body[0].exc is None
    full = "OOM_ERRORS" in c or {"MemoryError", "Standard_OutOfMemory"} <= c \
        or ("MemoryError" in c and rel.startswith(MEMORYERROR_ENOUGH))
    return single and full


def _unguarded(path: Path, rel: str) -> list[str]:
    """Handlers that could catch an out-of-memory error (broad: Exception,
    BaseException, bare, in a tuple or behind a module-level variable; or naming
    one of OOM) and neither re-raise it nor follow a guard in their try; and
    contextlib.suppress of any of those."""
    out, tree = [], ast.parse(path.read_text())
    env = _module_tuples(tree)

    def visit(node, fn):
        for child in ast.iter_child_nodes(node):
            f = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if isinstance(child, ast.Try) or type(child).__name__ == "TryStar":
                guarded = False
                for h in child.handlers:
                    c = _caught(h.type, env)
                    if _is_guard(h, c, rel):
                        guarded = True
                    elif (c & BROAD or c & OOM) and not guarded and not _reraises(h) \
                            and (rel, f) not in MAY_CATCH:
                        out.append(f"{rel}:{h.lineno} except {sorted(c)} in {f}")
            if isinstance(child, ast.Call):
                fn_name = getattr(child.func, "attr", getattr(child.func, "id", ""))
                if fn_name == "suppress" and any(_caught(a, env) & (BROAD | OOM) for a in child.args):
                    out.append(f"{rel}:{child.lineno} suppress in {f}")
            visit(child, f)
    visit(tree, "<module>")
    return out


def _oom(*_a, **_k):
    raise MemoryError("test")


def test_every_broad_handler_in_the_scorer_lets_memoryerror_through():
    """Everywhere under envs/common, envs/geom and envs/verifiers, subpackages
    included: a handler that would catch a MemoryError either re-raises it or
    is one of MAY_CATCH."""
    missing = []
    for top in ("envs/common", "envs/geom", "envs/verifiers"):
        for p in sorted((ROOT / top).rglob("*.py")):
            rel = str(p.relative_to(ROOT))
            if rel not in NOT_SCORING:
                missing += _unguarded(p, rel)
    assert not missing, missing


def test_the_lint_sees_every_form(tmp_path):
    src = tmp_path / "m.py"
    src.write_text(
        "import contextlib\n"
        "ERRS = (ValueError, Exception)\n"
        "def f():\n"
        "    try: g()\n"
        "    except (ValueError, Exception): pass\n"          # 5  broad in a tuple
        "    try: g()\n"
        "    except: pass\n"                                   # 7  bare
        "    try: g()\n"
        "    except BaseException: pass\n"                     # 9
        "    with contextlib.suppress(Exception): g()\n"       # 10
        "    try: g()\n"
        "    except OOM_ERRORS:\n"
        "        raise\n"
        "    except Exception: pass\n"                         # guarded
        "    try: g()\n"
        "    except Exception:\n"
        "        log()\n"
        "        raise\n"                                      # re-raises: fine
        "    try: g()\n"
        "    except Exception:\n"                              # 20 returns early, then raise
        "        if x: return 0\n"
        "        raise\n"
        "    try: g()\n"
        "    except ERRS: pass\n"                              # 24 broad behind a variable
        "    try: g()\n"
        "    except (OOM_ERRORS, KeyError):\n"
        "        raise\n"
        "    except Exception: pass\n"                         # guarded by a tuple guard
        "    try: g()\n"
        "    except MemoryError:\n"
        "        raise\n"
        "    except Exception: pass\n"                         # 32 MemoryError alone misses OCCT's
        "    try: g()\n"
        "    except Exception as e:\n"                         # 34 turned into another error
        "        raise ValueError(e)\n")
    got = [l.split(":")[1].split()[0] for l in _unguarded(src, "m.py")]
    assert got == ["5", "7", "9", "10", "20", "24", "32", "34"], got
    # in the vendored ECAD matcher (no OCCT) MemoryError alone is a guard
    assert "32" not in [l.split(":")[1].split()[0] for l in _unguarded(src, "envs/common/ecad_graph/m.py")]


def test_surface_indices_raises_instead_of_an_empty_surface(monkeypatch):
    import envs.geom.voxel  # noqa: F401
    from envs.common import asm_v1
    import trimesh
    from trimesh.voxel import creation
    monkeypatch.setattr(trimesh.Trimesh, "voxelized", _oom)
    monkeypatch.setattr(creation, "voxelize_subdivide", _oom)          # the path after change 245
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    with pytest.raises(MemoryError):
        asm_v1.surface_indices(v, np.array([[0, 1, 2]]), 16)


def test_legacy_iou_raises_instead_of_scoring_zero(monkeypatch):
    from envs.common import score
    step = ROOT / "tests/fixtures/t1/case1/gt/gt.step"
    real = score._normalized_mesh
    monkeypatch.setattr(score, "_normalized_mesh", lambda p: real(p) if p == step else _oom())
    with pytest.raises(MemoryError):
        score.iou_step_vs_step(step, Path("/answer.step"), res=16)


def test_meshguard_client_raises_the_workers_memoryerror(monkeypatch):
    from envs.geom import meshguard

    class W:
        n = 0
        tmp = Path("/tmp")

        def request(self, *_a):
            return {"error": "MemoryError: std::bad_alloc"}

    monkeypatch.setattr(meshguard, "_worker", lambda: W())
    monkeypatch.setattr(meshguard, "_write_brep", lambda *a: None)
    with pytest.raises(MemoryError):
        meshguard.tessellate(object(), 0.05)


def _fake_child(monkeypatch, rc, stdout, stderr):
    monkeypatch.delenv(run.SCORE_MEMORY_ENV, raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, rc, stdout, stderr))


def test_a_real_numpy_allocation_failure_is_a_memory_budget_record(tmp_path):
    """The real scorer child (run.SCORER_CHILD), its score_case swapped for one
    real allocation numpy cannot make: numpy raises _ArrayMemoryError, whose
    traceback's last line does not read "MemoryError" -- the child exits
    MEMORY_ERROR_EXIT by type and the record is memory_budget_exceeded."""
    driver = ("import sys, numpy as np\n"
              "import envs.common.score_case as sc\n"
              "def score_case(case, answer, mode=None):\n"
              "    try:\n"
              "        return np.zeros(1 << 50, dtype=np.uint8)\n"
              "    except MemoryError as x:\n"
              "        x.add_note('a note after the line')\n"
              "        raise\n"
              "sc.score_case = score_case\n"
              f"exec({run.SCORER_CHILD!r})\n")
    r = subprocess.run([sys.executable, "-c", driver, str(tmp_path), str(tmp_path / "a.step")],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert "_ArrayMemoryError" in r.stderr and r.returncode == run.MEMORY_ERROR_EXIT, r.stderr[-800:]
    with pytest.raises(run.MemoryBudgetExceeded, match="MemoryError"):   # numpy displays it as its base
        run.scorer_outcome(r, capped=False, limit=None)


def test_other_scorer_failures_stay_errors(monkeypatch, tmp_path):
    _fake_child(monkeypatch, 1, "", "Traceback ...\nValueError: no solid")
    with pytest.raises(RuntimeError, match="scorer exited 1"):
        run.score_in_subprocess(tmp_path, tmp_path / "a.step")


def test_stderr_tail_is_kept_on_success(monkeypatch, tmp_path):
    noise = "x" * 10_000 + "\niou: submission a.step unusable: ValueError: no solid"
    _fake_child(monkeypatch, 0, '\n{"score": 0.5}', noise)
    out = run.score_in_subprocess(tmp_path, tmp_path / "a.step")
    assert out["score"] == 0.5
    assert out["stderr_tail"].endswith("unusable: ValueError: no solid")
    assert len(out["stderr_tail"]) == run.STDERR_TAIL
    _fake_child(monkeypatch, 0, '\n{"score": 0.5}', "")
    assert "stderr_tail" not in run.score_in_subprocess(tmp_path, tmp_path / "a.step")


# ── the mesh worker killed from outside (a container's OOM kill, no cgroup scope) ──
SLEEPER = "import sys, time; sys.stdin.readline(); time.sleep(120)"


class SignalOnFlush:
    """A worker's stdin that sends `sig` to the worker as soon as a request is
    flushed to it: the death lands mid-call, every time, with no timer to race."""

    def __init__(self, proc, sig):
        self._proc, self._sig, self._f = proc, sig, proc.stdin

    def flush(self):
        import os
        self._f.flush()
        os.kill(self._proc.pid, self._sig)

    def __getattr__(self, name):
        return getattr(self._f, name)


def _fake_worker(tmp_path, sig):
    """A meshguard _Worker whose process takes the request and hangs; `sig` arrives mid-call."""
    from envs.geom import meshguard
    w = meshguard._Worker.__new__(meshguard._Worker)
    w.tmp, w.n, w._buf = Path(tmp_path), 0, b""
    w.proc = subprocess.Popen([sys.executable, "-c", SLEEPER], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    w.proc.stdin = SignalOnFlush(w.proc, sig)
    return w


def test_worker_sigkill_mid_call_is_infrastructure(tmp_path):
    import signal
    from envs.geom import meshguard
    w = _fake_worker(tmp_path, signal.SIGKILL)
    with pytest.raises(meshguard.MeshWorkerKilled, match="SIGKILL"):
        w.request({"deflection": 0.05}, cpu=60, wall=60)


@pytest.mark.parametrize("sig", ["SIGSEGV", "SIGABRT"])
def test_worker_crash_on_bad_geometry_stays_unmeshable(tmp_path, sig):
    import signal
    from envs.geom import meshguard
    w = _fake_worker(tmp_path, getattr(signal, sig))
    with pytest.raises(meshguard.UnmeshableShape, match="died"):
        w.request({"deflection": 0.05}, cpu=60, wall=60)


def test_meshguards_own_wall_kill_stays_unmeshable(tmp_path):
    import signal
    from envs.geom import meshguard
    w = _fake_worker(tmp_path, signal.SIGCONT)                 # harmless; the 2 s backstop fires first
    with pytest.raises(meshguard.UnmeshableShape, match="wall clock"):
        w.request({"deflection": 0.05}, cpu=1, wall=2)


def test_score_case_gives_no_score_when_the_worker_is_killed(tmp_path, monkeypatch):
    """End to end in-process: every mesh request lands on a worker that is
    SIGKILLed mid-call; score_case must raise, not return a score."""
    import signal
    from envs.geom import meshguard
    from envs.common.score_case import score_case
    monkeypatch.setattr(meshguard, "_worker", lambda: _fake_worker(tmp_path, signal.SIGKILL))
    case = ROOT / "tests/fixtures/t1/case1"
    with pytest.raises(meshguard.MeshWorkerKilled):
        score_case(case, case / "gt/gt.step")


def test_harness_records_a_killed_worker_as_memory_budget(tmp_path):
    """MeshWorkerKilled out of the real scorer child: exit MEMORY_ERROR_EXIT, no score."""
    driver = ("import envs.common.score_case as sc\n"
              "from envs.geom.meshguard import MeshWorkerKilled\n"
              "def score_case(case, answer, mode=None):\n"
              "    raise MeshWorkerKilled('mesher process killed by SIGKILL during the test')\n"
              "sc.score_case = score_case\n"
              f"exec({run.SCORER_CHILD!r})\n")
    r = subprocess.run([sys.executable, "-c", driver, str(tmp_path), str(tmp_path / "a.step")],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == run.MEMORY_ERROR_EXIT, r.stderr[-800:]
    with pytest.raises(run.MemoryBudgetExceeded, match="MeshWorkerKilled"):
        run.scorer_outcome(r, capped=False, limit=None)


def test_a_worker_oom_killed_before_the_request_is_infrastructure(tmp_path):
    """Alive at _worker()'s check, SIGKILLed before the request is written: the
    write's BrokenPipe is classified by the exit code, not read as unmeshable."""
    import signal
    from envs.geom import meshguard
    w = meshguard._Worker.__new__(meshguard._Worker)
    w.tmp, w.n, w._buf = Path(tmp_path), 0, b""
    w.proc = subprocess.Popen([sys.executable, "-c", SLEEPER], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    w.proc.send_signal(signal.SIGKILL)
    w.proc.wait()
    with pytest.raises(meshguard.MeshWorkerKilled):
        w.request({"deflection": 0.05, "pad": "x" * 1_000_000}, cpu=60, wall=60)   # over a pipe buffer


def test_cpu_budget_stays_under_a_finite_hard_limit(tmp_path):
    """Under a finite hard RLIMIT_CPU (ulimit -t, Slurm, docker --ulimit cpu) the
    budget must still arrive as SIGXCPU -- at the hard limit the kernel sends
    SIGKILL, which would read as an OOM kill."""
    code = ("import resource, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "resource.setrlimit(resource.RLIMIT_CPU, (5, 5))\n"
            "from envs.geom.meshguard import _cpu_limit\n"
            "_cpu_limit(60)\n"
            "soft, hard = resource.getrlimit(resource.RLIMIT_CPU)\n"
            "assert soft < hard == 5, (soft, hard)\n"
            "while True: pass\n")
    r = subprocess.run([sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True, timeout=60)
    import signal
    assert r.returncode == -signal.SIGXCPU, (r.returncode, r.stderr[-400:])


def test_occt_out_of_memory_is_reported_as_memoryerror(monkeypatch):
    """OCCT's Standard_OutOfMemory is not a Python MemoryError; the worker names
    it one, and the client raises MemoryError on that, not an unmeshable shape."""
    from OCP.Standard import Standard_OutOfMemory
    from envs.geom import meshguard
    assert meshguard._error_reply(Standard_OutOfMemory("BRepMesh")) == {"error": "MemoryError: BRepMesh"}
    assert meshguard._error_reply(ValueError("bad")) == {"error": "ValueError: bad"}

    class W:
        n, tmp = 0, Path("/tmp")

        def request(self, *_a):
            return meshguard._error_reply(Standard_OutOfMemory("BRepMesh"))

    monkeypatch.setattr(meshguard, "_worker", lambda: W())
    monkeypatch.setattr(meshguard, "_write_brep", lambda *a: None)
    with pytest.raises(MemoryError):
        meshguard.tessellate(object(), 0.05)


# ── OCCT's out of memory, in the scorer process itself ──
def _occt_oom():
    from OCP.Standard import Standard_OutOfMemory
    return Standard_OutOfMemory("BRepMesh: out of memory (test)")


def test_occt_out_of_memory_in_process_raises_instead_of_scoring_zero(monkeypatch):
    """Standard_OutOfMemory is a plain Exception to Python; the shared OOM tuple
    makes every guard let it through (score.py's legacy iou scored 0.0 on it)."""
    from OCP.Standard import Standard_OutOfMemory
    from envs.common import score
    from envs.geom.oom import OOM_ERRORS
    assert Standard_OutOfMemory in OOM_ERRORS and MemoryError in OOM_ERRORS
    step = ROOT / "tests/fixtures/t1/case1/gt/gt.step"
    real = score._normalized_mesh

    def mesh(p):
        if p == step:
            return real(p)
        raise _occt_oom()
    monkeypatch.setattr(score, "_normalized_mesh", mesh)
    with pytest.raises(Standard_OutOfMemory):
        score.iou_step_vs_step(step, Path("/answer.step"), res=16)


def _child(tmp_path, body: str, pre: str = ""):
    driver = (pre + "import envs.common.score_case as sc\n"
              "def score_case(case, answer, mode=None):\n" + body +
              "sc.score_case = score_case\n"
              f"exec({run.SCORER_CHILD!r})\n")
    return subprocess.run([sys.executable, "-c", driver, str(tmp_path), str(tmp_path / "a.step")],
                          capture_output=True, text=True, cwd=str(ROOT))


def test_occt_out_of_memory_out_of_the_scorer_child_is_a_memory_budget_record(tmp_path):
    r = _child(tmp_path, "    from OCP.Standard import Standard_OutOfMemory\n"
                         "    raise Standard_OutOfMemory('BRepMesh: out of memory')\n")
    assert r.returncode == run.MEMORY_ERROR_EXIT, r.stderr[-800:]
    with pytest.raises(run.MemoryBudgetExceeded, match="Standard_OutOfMemory"):
        run.scorer_outcome(r, capped=False, limit=None)


def test_the_marker_survives_a_traceback_that_cannot_print(tmp_path):
    r = _child(tmp_path, "    raise MemoryError('x')\n",
               pre="import traceback\n"
                   "def broken(*a, **k):\n"
                   "    raise RuntimeError('cannot print')\n"
                   "traceback.print_exc = broken\n")
    assert r.returncode == run.MEMORY_ERROR_EXIT and run.MEMORY_ERROR_MARK in r.stderr, r.stderr[-800:]
    with pytest.raises(run.MemoryBudgetExceeded):
        run.scorer_outcome(r, capped=False, limit=None)


# ── a budget the worker can no longer grant in full ──
def test_cpu_limit_says_when_it_cannot_grant_the_budget(tmp_path):
    code = ("import resource, sys, time\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "resource.setrlimit(resource.RLIMIT_CPU, (8, 8))\n"
            "from envs.geom.meshguard import _cpu_limit\n"
            "t = time.process_time()\n"
            "while time.process_time() - t < 2.5: pass\n"      # this worker has spent CPU already
            "used = time.process_time()\n"
            "print(_cpu_limit(1), _cpu_limit(8 - used))\n")
    r = subprocess.run([sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True, timeout=60)
    assert r.stdout.split() == ["True", "False"], (r.stdout, r.stderr[-400:])


RESTARTER = ("import json, sys\n"
             "req = json.loads(sys.stdin.readline())\n"
             "if not req.get('short_ok'):\n"
             "    print(json.dumps({'restart': 'cpu'}), flush=True); sys.exit(0)\n"
             "print(json.dumps({'ok': True, 'short_ok': True}), flush=True)\n"
             "sys.stdin.readline()\n")


def test_a_worker_that_cannot_grant_the_budget_is_replaced(tmp_path, monkeypatch):
    """The request goes to a fresh worker (same tmp, so its files are still there)
    instead of running on a shortened budget that depends on the cases before it."""
    from envs.geom import meshguard
    spawned = []

    def spawn(self):
        self.proc = subprocess.Popen([sys.executable, "-c", RESTARTER], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        self._buf = b""
        spawned.append(self.proc.pid)
    monkeypatch.setattr(meshguard._Worker, "_spawn", spawn)
    w = meshguard._Worker()
    try:
        reply = w.request({"deflection": 0.05}, cpu=60, wall=60)
    finally:
        w.close()
    assert reply == {"ok": True, "short_ok": True} and len(spawned) == 2
