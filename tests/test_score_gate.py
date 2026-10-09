"""harness/run.py ScoreGate: the scoring pool grows as episodes finish.

Measured 2026-10-09 (Haiku high, Core, 30 rounds, 8 vCPU / 16 GB, two fixed
scorers): 97 of 100 episodes had finished and 19 were scored; the other 78 took
5-7 hours of scoring alone with the CPU at load 3/8 and 10 GB free.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))
import run as R                                                   # noqa: E402
from envs.common.sandbox import is_secret_env                     # noqa: E402


def test_the_allowance_rises_as_the_episodes_finish():
    """run_core's 8 vCPU / 14 GB budget: 10 workers, scorers 2 -> 4."""
    g = R.ScoreGate(2, 4, mem_gb=14, workers=10, episodes=20)
    seen = []
    for _ in range(20):
        seen.append(g.allowed())
        g.episode_done()
    seen.append(g.allowed())
    assert seen[0] == 2 and seen[-1] == 4 and seen == sorted(seen)
    assert R.ScoreGate(2, 2).allowed() == 2, "no memory given: fixed, as before"
    assert R.ScoreGate(3, 4, mem_gb=6, workers=10, episodes=0).allowed() == 3, "never below the start"


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.01)
    return cond()


def test_scores_waiting_at_the_gate_start_as_episodes_finish():
    g = R.ScoreGate(2, 4, mem_gb=14, workers=10, episodes=10)
    release = threading.Event()

    def score():
        with g:
            release.wait(10)
    threads = [threading.Thread(target=score) for _ in range(8)]
    for t in threads:
        t.start()
    assert _wait(lambda: g.busy == 2)
    time.sleep(0.1)
    assert g.busy == 2, "every episode slot busy: two scores at once"
    for _ in range(10):
        g.episode_done()
    assert _wait(lambda: g.busy == 4), "the episodes are done: their memory goes to scoring"
    time.sleep(0.1)
    assert g.busy == 4 and g.peak == 4
    release.set()
    for t in threads:
        t.join(10)
    assert g.busy == 0


def test_run_takes_the_elastic_flags(tmp_path):
    env = {k: v for k, v in os.environ.items() if not is_secret_env(k)}
    env.setdefault("CADENV_LOCAL", "1")
    r = subprocess.run([sys.executable, "harness/run.py", "--model", "mock/oracle", "--cases", "examples/task1",
                        "--rounds", "1", "--out", str(tmp_path / "r.json"), "--work", str(tmp_path / "w"),
                        "--unsafe-local", "--score-workers", "1", "--score-workers-max", "3", "--memory-gb", "6"],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "scorers 1->3 as episodes finish (6 GB shared)" in r.stdout
