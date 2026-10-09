"""harness/run.py --resume: what is kept, what is re-scored, what is run again.

A score that failed for the host's sake is re-scored from the answer on disk,
never re-run, so a lab does not pay for an episode twice; that includes
records written before the harness marked such failures (score_error).
Runs the mock oracle, no API key."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))
from envs.common.sandbox import is_secret_env                     # noqa: E402

os.environ.setdefault("CADENV_LOCAL", "1")
RUN = ROOT / "harness/run.py"
STEP_CASES = [p.parent for p in sorted((ROOT / "tests/fixtures").rglob("case.json")) if (p.parent / "gt/gt.step").exists()]


def run(*args, expect: int = 0):
    env = {k: v for k, v in os.environ.items() if not is_secret_env(k)}
    r = subprocess.run([sys.executable, str(RUN), *args], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == expect, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}"
    return r


def _out(r) -> dict:
    return json.loads((ROOT / r.stdout.strip().splitlines()[-1].split("-> ")[-1]).read_text())


@pytest.fixture
def home_work(tmp_path):
    """A work root under $HOME (with the sandbox image present, Docker's VM shares only $HOME)."""
    d = Path.home() / "cad-agent-work" / "pytest" / tmp_path.name
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("marks", [
    {"error": "ImportError: libGL.so.1: cannot open shared object file", "score_error": True},
    # the shape the harness wrote before score_error existed (main 5d36424..acc118d)
    {"error": "RuntimeError: scorer exited 1: ImportError: libGL.so.1: cannot open shared object file"},
    # finished, recorded, then the run was killed before its score landed
    {"pending_score": True},
    # a score past its time limit (T2 case004 on a 16 GB box at 3600 s), in the error text only
    {"error": "RuntimeError: scorer timed out after 3600 s (re-score later)"},
], ids=["score_error", "pre-1cb51ab", "killed-before-scoring", "timed-out"])
def test_resume_rescores_a_scorer_failure_and_never_reruns_its_episode(tmp_path, home_work, marks):
    """A score that failed for the host's sake (a missing system library, the
    memory budget) is re-scored on --resume from the answer already on disk:
    the episode is not run again, so the model is not paid for twice."""
    out = tmp_path / "r.json"
    case = STEP_CASES[0]
    first = _out(run("--model", "mock/oracle", "--cases", str(case.relative_to(ROOT)), "--rounds", "2",
                     "--out", str(out), "--work", str(home_work / "w")))
    rec = first["cases"][0]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert rec["harness_commit"].split("+")[0] == head, "the record names the harness that ran it"
    step = Path(rec["step"]); mtime = step.stat().st_mtime_ns
    rec.update(score=None, seconds=12345.0, **marks)
    out.write_text(json.dumps(first))
    r = run("--model", "mock/oracle", "--cases", str(case.relative_to(ROOT)), "--rounds", "2",
            "--out", str(out), "--work", str(home_work / "w"), "--resume")
    assert "resume: 0 kept, 1 to re-score, 0 to run" in r.stdout
    again = _out(r)["cases"][0]
    assert again.get("error") is None and "score_error" not in again
    assert again["score"]["iou"] == pytest.approx(1.0, abs=1e-4)
    assert again["seconds"] == 12345.0 and step.stat().st_mtime_ns == mtime     # the episode did not run


def test_an_episode_error_with_no_answer_is_still_rerun():
    sys.path.insert(0, str(ROOT / "harness"))
    import run as R
    assert not R._rescore_only({"error": "APIConnectionError: boom", "step": None})
    assert not R._rescore_only({"error": "RuntimeError: scorer exited 1", "step": "/no/such/answer.step"})


def test_a_run_task_by_task_into_one_file_keeps_every_task(tmp_path, home_work):
    """--resume into a file that holds other cases carries them over untouched.
    Measured 2026-10-09: --cases task3 then --cases task4 into one results file
    left task4's 25 records; task3's 28 were gone."""
    out = tmp_path / "r.json"
    args = ("--model", "mock/oracle", "--rounds", "1", "--out", str(out), "--resume")
    first = _out(run("--cases", "examples/task1", *args, "--work", str(home_work / "a")))["cases"]
    r = run("--cases", "examples/task2", *args, "--work", str(home_work / "b"))
    assert "resume: 0 kept, 0 to re-score, " in r.stdout and f"{len(first)} other cases carried over" in r.stdout
    both = _out(r)["cases"]
    assert [c for c in both if "/task1/" in c["case"]] == first
    n2 = sum(1 for _ in (ROOT / "examples/task2").rglob("case.json"))
    assert n2 and sum("/task2/" in c["case"] for c in both) == n2 and len(both) == len(first) + n2
    assert [c["case"] for c in both] == sorted((c["case"] for c in both), key=Path)


def test_a_score_may_take_longer_when_asked_and_a_timeout_says_how(monkeypatch):
    import run as R
    monkeypatch.delenv(R.SCORE_TIMEOUT_ENV, raising=False)
    assert (R.score_timeout(False), R.score_timeout(True)) == (3600, 4200)
    monkeypatch.setenv(R.SCORE_TIMEOUT_ENV, "14400")
    assert (R.score_timeout(False), R.score_timeout(True)) == (14400, 14400)

    def slow(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], kw["timeout"])
    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(RuntimeError) as e:
        R.score_in_subprocess(STEP_CASES[0], Path("/no/answer.step"))
    msg = str(e.value)
    assert "timed out after 14400 s" in msg and "--resume" in msg and f"{R.SCORE_TIMEOUT_ENV}=28800" in msg
    assert R._SCORER_FAILED.search("RuntimeError: " + msg)
