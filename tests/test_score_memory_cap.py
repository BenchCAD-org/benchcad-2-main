"""The scorer's memory budget (harness.run: BENCHCAD_SCORE_MEMORY_MAX).

A score runs in its own cgroup scope with MemoryMax and MemorySwapMax=0; past
the cap the kernel ends that scorer only and the record says
`memory_budget_exceeded`, never a score. The wrapper's verdict is tested
everywhere with a fake memory.events; the real cap only where a user systemd
can apply it (Linux, e.g. WSL) -- elsewhere those tests skip.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import run  # noqa: E402

KILL_SELF = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"


def _wrapped(child_code: str, events: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", run._OOM_WRAPPER, sys.executable, "-c", child_code],
                          capture_output=True, text=True, env={"BENCHCAD_CGROUP_EVENTS": str(events)})


def test_no_budget_leaves_the_command_alone():
    cmd = [sys.executable, "-c", "print(1)"]
    assert run.memory_capped(cmd, None) == (cmd, False)
    assert run.memory_capped(cmd, "") == (cmd, False)


def test_budget_without_user_systemd_runs_uncapped(monkeypatch, capsys):
    monkeypatch.setitem(run._SCOPE_OK, "ok", False)
    monkeypatch.delitem(run._SCOPE_OK, "warned", raising=False)
    cmd = [sys.executable, "-c", "print(1)"]
    assert run.memory_capped(cmd, "10G") == (cmd, False)
    assert "WITHOUT a memory cap" in capsys.readouterr().err
    monkeypatch.setenv(run.SCORE_MEMORY_ENV, "10G")
    assert run.score_memory_cap() is None              # recorded as uncapped


def test_wrapper_names_an_oom_kill(tmp_path):
    ev = tmp_path / "memory.events"
    ev.write_text("low 0\nhigh 0\nmax 3\noom 1\noom_kill 1\n")
    r = _wrapped(KILL_SELF, ev)
    assert r.returncode == run.MEMORY_BUDGET_EXIT
    assert run.MEMORY_BUDGET_MARK in r.stderr


def test_wrapper_passes_other_deaths_through(tmp_path):
    ev = tmp_path / "memory.events"
    ev.write_text("oom 0\noom_kill 0\n")
    assert _wrapped(KILL_SELF, ev).returncode == 128 + 9        # a SIGKILL that was not the cap
    assert _wrapped("import sys; sys.exit(3)", ev).returncode == 3
    ok = _wrapped("print('fine')", ev)
    assert ok.returncode == 0 and ok.stdout.strip() == "fine"
    ev.write_text("oom_kill 2\n")
    assert _wrapped("print('fine')", ev).returncode == 0         # an earlier kill does not taint a clean exit


needs_scope = pytest.mark.skipif(not run._user_scope_available(),
                                 reason="no `systemd-run --user --scope` here (macOS, or Linux without a user systemd)")


@needs_scope
def test_cap_ends_a_scorer_past_it():
    cmd, capped = run.memory_capped([sys.executable, "-c", "x = bytearray(400 * 2**20); print(len(x))"], "150M")
    assert capped
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == run.MEMORY_BUDGET_EXIT and run.MEMORY_BUDGET_MARK in r.stderr


@needs_scope
def test_cap_leaves_a_scorer_under_it_alone():
    cmd, capped = run.memory_capped([sys.executable, "-c", "x = bytearray(40 * 2**20); print('ok', len(x))"], "150M")
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert capped and r.returncode == 0 and r.stdout.split()[0] == "ok"


@needs_scope
def test_score_in_subprocess_reports_the_budget(monkeypatch, tmp_path):
    """score_in_subprocess turns a cap kill into MemoryBudgetExceeded (no score)."""
    monkeypatch.setenv(run.SCORE_MEMORY_ENV, "150M")
    real = run.memory_capped
    monkeypatch.setattr(run, "memory_capped",
                        lambda cmd, limit: real([sys.executable, "-c", "x = bytearray(400 * 2**20)"], limit))
    with pytest.raises(run.MemoryBudgetExceeded, match="memory_budget_exceeded"):
        run.score_in_subprocess(tmp_path, tmp_path / "answer.step")
    assert run.score_memory_cap() == "150M"
