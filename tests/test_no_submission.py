"""An episode that ends without a ```submit has no answer, Task 6 included.

The prompt says it in those words ("An episode that ends without a ```submit
has no answer and scores 0"), and envs/common/episode.py reports an answer only
when a submitted program ran and left one. harness/run.py used to add one back:
a `pred_graph.json` found in the working directory was taken as the answer
whenever the episode reported none. That scored two episodes the rule gives
nothing -- one that exported the graph in a round and then never submitted, one
whose submitted program wrote the file and then failed -- and over 3,755 T6
records of our own it had scored 5 of them, the largest 0.5008.

The real runner, with the episode stubbed out: no key, no sandbox, no score.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.common.sandbox import is_secret_env                             # noqa: E402

_spec = importlib.util.spec_from_file_location("core_score", ROOT / "tools/core_score.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)

CASE = ROOT / "tests/fixtures/t6/case1"
GRAPH = CASE / "gt/gt_graph.json"

# Runs harness/run.py's own main(), with run_episode replaced by an episode
# that leaves the reference graph in the working directory and reports no
# answer -- which is what an episode that ran out of rounds after an
# `export` round reports, and what one whose submitted program failed reports.
STUB = '''
import os, pathlib, sys
sys.path[:0] = [os.environ["BC_ROOT"], os.environ["BC_ROOT"] + "/harness"]
import run as R

GRAPH = pathlib.Path(os.environ["BC_GRAPH"]).read_text()
SUBMITTED = bool(os.environ.get("BC_SUBMITTED"))


def episode(case, work, call, max_rounds=None, token_budget=None, **kw):
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "pred_graph.json").write_text(GRAPH)
    rounds = [{"round": 1, "action": "exec", "rc": 0}]
    if SUBMITTED:
        rounds += [{"round": 2, "action": "submit"},
                   {"round": 2, "action": "submit_failed", "rc": 1, "stderr": "boom"}]
    else:
        rounds += [{"round": 2, "action": "no_submission"}]
    return {"code": "x = 1" if SUBMITTED else "", "step": None, "artifact": None,
            "rounds": rounds, "submitted": SUBMITTED,
            "ended_by": "submit" if SUBMITTED else "rounds", "output_tokens": 1000}


R.run_episode = episode
raise SystemExit(R.main())              # the runner parses sys.argv itself
'''


def _run(tmp_path: Path, submitted: bool) -> tuple[dict, str, Path]:
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    out, work = tmp_path / "r.json", tmp_path / "w"
    env = {k: v for k, v in os.environ.items() if not is_secret_env(k)}
    env.update({"BC_ROOT": str(ROOT), "BC_GRAPH": str(GRAPH), "CADENV_LOCAL": "1",
                "BC_SUBMITTED": "1" if submitted else ""})
    r = subprocess.run([sys.executable, str(stub), "--model", "mock/oracle", "--cases", str(CASE),
                        "--rounds", "2", "--out", str(out), "--work", str(work)],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}"
    [rec] = json.loads(out.read_text())["cases"]
    left = [p for p in work.rglob("pred_graph.json")]
    assert left, "the stub episode did leave a graph in the working directory"
    return rec, r.stdout, left[0]


@pytest.mark.parametrize("submitted", [False, True],
                         ids=["ran out of rounds", "submitted program failed"])
def test_a_graph_left_in_the_working_directory_is_not_an_answer(tmp_path, submitted):
    rec, out, graph = _run(tmp_path, submitted)
    assert rec["submitted"] is submitted
    assert rec["step"] is None, "the graph in the working directory was taken as the answer"
    assert rec["score"] is None and rec["artifact"] is None
    assert rec["artifact_sha256_of"] is None and rec["artifact_sha256"] is None
    assert "no submission" in out
    # It was not scored because the harness did not accept it, not because it
    # was gone: it is still there, and it is the reference itself.
    assert graph.read_text() == GRAPH.read_text()
    # And the headline counts the case as a real 0, not as pending.
    assert C.resolve(rec) == ("no_submission", 0.0, "no submission")


@pytest.fixture
def home_work(tmp_path):
    """A work root under $HOME (with the sandbox image present, Docker's VM shares only $HOME)."""
    d = Path.home() / "cad-agent-work" / "pytest" / tmp_path.name
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_a_submitted_graph_is_still_the_answer(tmp_path, home_work):
    """The other half of the rule, and what the fallback was thought to cover:
    an ECAD answer that WAS submitted is found and scored. envs/common/episode.py
    looks for the graph itself, so the runner needs no second look. The real
    episode in a real sandbox, with the mock oracle's answer."""
    out = tmp_path / "r.json"
    env = {k: v for k, v in os.environ.items() if not is_secret_env(k)}
    env["CADENV_LOCAL"] = "1"                   # Docker when it is there, locally when it is not
    r = subprocess.run([sys.executable, str(ROOT / "harness/run.py"), "--model", "mock/oracle",
                        "--cases", str(CASE), "--rounds", "2", "--out", str(out),
                        "--work", str(home_work / "w")],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}"
    [rec] = json.loads(out.read_text())["cases"]
    assert rec["submitted"] and rec["ended_by"] == "submit"
    assert Path(rec["step"]).name == "pred_graph.json" and rec["artifact"] == "step"
    assert rec["score"]["score"] == pytest.approx(1.0), "the reference graph must score 1.0"
    assert C.resolve(rec)[0] == "scored"
