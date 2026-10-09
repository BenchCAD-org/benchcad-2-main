"""harness/run.py work directories: two runs started in the same second -- a launcher that starts
every effort of a case at once -- must not share a work root, a results file or an episode directory.
When they did, the lanes overwrote each other's final.step (and each episode's start deleted what another
lane had staged), and rescoring those records read the last writer's submission for all of them."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAMP = "20260927-120000"                                 # both runs start in this second


def _R():
    sys.path.insert(0, str(ROOT / "harness"))
    import run as R
    return R


def test_runs_started_in_the_same_second_get_their_own_work_root_and_results(tmp_path, monkeypatch):
    R = _R()
    monkeypatch.setenv("HOME", str(tmp_path))              # the default root lives under ~/cad-agent-work
    monkeypatch.chdir(tmp_path)
    high_1 = R.work_paths(None, None, "openai/gpt-6-astra", "high", 0, STAMP)
    high_2 = R.work_paths(None, None, "openai/gpt-6-astra", "high", 0, STAMP)
    xhigh = R.work_paths(None, None, "openai/gpt-6-astra", "xhigh", 0, STAMP)
    roots = {high_1[0], high_2[0], xhigh[0]}
    results = {high_1[1], high_2[1], xhigh[1]}
    assert len(roots) == 3, roots
    assert len(results) == 3, results


def test_efforts_of_one_case_get_their_own_episode_dirs(tmp_path):
    R = _R()
    case = tmp_path / "envs/t3_part2step_heldout/cases/some_part_00"
    case.mkdir(parents=True)
    root = tmp_path / "run"
    dirs = {R.episode_dir(root, 0, effort, case) for effort in ("low", "medium", "high", "xhigh")}
    assert len(dirs) == 4, dirs


def test_a_work_root_in_use_by_another_run_is_refused(tmp_path):
    R = _R()
    root = tmp_path / "run"
    held = R.claim_work_root(root)
    try:
        with pytest.raises(SystemExit):
            R.claim_work_root(root)                        # a second run pointed at the same --work
    finally:
        held.close()
    R.claim_work_root(root).close()                        # free again once the first run is gone


def test_a_file_system_without_flock_is_reported_as_such_not_as_in_use(tmp_path, monkeypatch):
    """Only EWOULDBLOCK means another run holds the root. NFS or WSL's /mnt/c answer flock with
    ENOLCK / EOPNOTSUPP, and calling that "in use" would send someone looking for a run that isn't there."""
    import errno
    import fcntl
    R = _R()

    def no_locks(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")
    monkeypatch.setattr(fcntl, "flock", no_locks)
    with pytest.raises(SystemExit) as got:
        R.claim_work_root(tmp_path / "run")
    msg = str(got.value)
    assert "No locks available" in msg and "flock" in msg
    assert "another harness run" not in msg
