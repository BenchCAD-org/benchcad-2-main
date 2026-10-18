"""The episode's limits: an output-token budget, and rounds when capped.

Once the model's reasoning goes back to it (tests/test_reasoning_carry.py)
it no longer re-derives it every round, and a round cap binds models that
work in many small rounds. Since 2026-10-18 (2.1-beta.2) the default is
500,000 output tokens and no round limit; 2.1-beta (2026-10-16) had 40
rounds and 500,000; --rounds N still caps the rounds, for a smoke run.

The episode ends at the first of a submission, the budget and (when given)
the rounds. The budget counts every call's output tokens (thinking
included), is checked between calls and never inside one, and every
observation says what is left.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CADENV_LOCAL", "1")
from envs.common import episode as E                             # noqa: E402

CASE = ROOT / "tests/fixtures/t3/case1"
SUBMIT = "```submit\nimport cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)\n```"
PY = "```python\nprint('measured')\n```"


@pytest.fixture(autouse=True)
def _local(monkeypatch):
    from envs.common import sandbox as S
    monkeypatch.setenv("CADENV_LOCAL", "1")
    monkeypatch.setattr(S, "_docker_ready", lambda: False)


def model(replies, out_tokens):
    """A call_fn answering call n with replies[n] and spending out_tokens[n]
    output tokens, as a provider's usage would report them."""
    seen: list[dict] = []
    usage: list[dict] = []

    def call(system, turns, plain=False):
        n = len(seen)
        seen.append({"system": system, "last": turns[-1]["text"]})
        usage.append({"input_tokens": 10, "output_tokens": out_tokens[min(n, len(out_tokens) - 1)]})
        return replies[min(n, len(replies) - 1)]
    call.usage, call.context_tokens, call.seen = usage, 1_000_000, seen
    return call


def test_the_header_and_the_prompt_are_unchanged_without_a_budget():
    """No --token-budget: the system prompt's sentence and the observation
    header every run before 2026-10-15 had, byte for byte."""
    assert E.limits_text(30, None) == "30 rounds"
    assert E.SYSTEM.format(task_brief="T", file_list="", tools_help="", submit_what="S",
                           limits=E.limits_text(30, None), cq_version="2.3.0").startswith(
        "You are solving a CAD task in a working directory, over 30 rounds.\n")
    assert E.round_status(12, 30) == "Round 12/30"


def test_with_no_round_limit_the_header_and_the_prompt_name_only_the_budget():
    assert E.round_status(12, None, 340_512, 500_000) == "Round 12 · output tokens 340,512 / 500,000"
    assert E.limits_text(None, 500_000) == ("at most 500,000 output tokens, your thinking included -- "
                                            "the episode ends when they are spent")
    assert E.SYSTEM.format(task_brief="T", file_list="", tools_help="", submit_what="S",
                           limits=E.limits_text(None, 500_000), cq_version="2.3.0").startswith(
        "You are solving a CAD task in a working directory, over at most 500,000 output tokens, ")


def test_with_a_budget_the_header_shows_what_is_left():
    assert E.round_status(12, 50, 340_512, 1_000_000) == "Round 12/50 · output tokens 340,512 / 1,000,000"
    assert E.limits_text(200, 500_000) == ("at most 200 rounds and 500,000 output tokens, your thinking "
                                           "included -- the episode ends at whichever runs out first")


def test_the_budget_ends_the_episode_between_calls_never_inside_one(tmp_path):
    """Mode B. Each call spends 200k: after call 3 the budget (500k) is
    spent, so no fourth call is made. Call 3 began with 400k spent and ran to
    its end (600k): a call is never cut, and its round was played. The
    observations count the tokens; the record says what ended it."""
    call = model([PY], [200_000])
    out = E.run_episode(CASE, tmp_path / "w", call, max_rounds=200, exec_timeout=60, token_budget=500_000)
    assert len(call.seen) == 3
    assert [r["action"] for r in out["rounds"]] == ["exec", "exec", "exec", "token_budget_spent", "no_submission"]
    assert out["ended_by"] == "token_budget" and out["output_tokens"] == 600_000 and not out["submitted"]
    assert call.seen[1]["last"].startswith("Round 1/200 · output tokens 200,000 / 500,000 -- exit 0")
    assert call.seen[2]["last"].startswith("Round 2/200 · output tokens 400,000 / 500,000 -- exit 0")
    assert "at most 200 rounds and 500,000 output tokens" in call.seen[0]["system"]


def test_a_submission_inside_the_budget_counts(tmp_path):
    call = model([PY, SUBMIT], [300_000])
    out = E.run_episode(CASE, tmp_path / "w", call, max_rounds=200, exec_timeout=60, token_budget=500_000)
    assert out["submitted"] and out["ended_by"] == "submit" and out["output_tokens"] == 600_000


def test_the_rounds_end_it_when_they_run_out_first(tmp_path):
    """Mode A: 50 rounds with a 1M output guard; small replies, so the
    rounds end it (here 3 of them)."""
    call = model([PY], [1_000])
    out = E.run_episode(CASE, tmp_path / "w", call, max_rounds=3, exec_timeout=60, token_budget=1_000_000)
    assert len(call.seen) == 3 and out["ended_by"] == "rounds"
    assert [r["action"] for r in out["rounds"]] == ["exec", "exec", "exec", "no_submission"]


def test_without_a_budget_only_the_rounds_count(tmp_path):
    call = model([PY], [10_000_000])
    out = E.run_episode(CASE, tmp_path / "w", call, max_rounds=2, exec_timeout=60)
    assert len(call.seen) == 2 and out["ended_by"] == "rounds"
    assert call.seen[1]["last"].startswith("Round 1/2 -- exit 0")


def test_the_defaults_are_500k_and_no_round_limit_and_are_the_reportable_setting(tmp_path):
    """Since 2026-10-18 a run with no limit flags has a 500,000 output-token
    budget and no round limit: the banner, the file and every record say so,
    and core_score reports it (no SMOKE). A round cap is a different setting
    (SMOKE); so is no budget, which needs a round cap; no limit at all, or a
    negative one, is refused."""
    import harness.run as R
    sys.path.insert(0, str(ROOT / "tools"))
    import core_score as C

    def run(*extra, out):
        return subprocess.run([sys.executable, str(ROOT / "harness/run.py"), "--model", "mock/dumb",
                               "--cases", str(CASE), "--unsafe-local", "--out", str(tmp_path / out),
                               "--work", str(tmp_path / ("w" + out)), *extra],
                              cwd=ROOT, capture_output=True, text=True)
    r = run(out="d.json")
    assert r.returncode == 0, r.stderr[-800:]
    assert "rounds no limit " in r.stdout and "token-budget 500,000 " in r.stdout
    out = json.loads((tmp_path / "d.json").read_text()); rec = out["cases"][0]
    assert out["token_budget"] == rec["token_budget"] == R.DEFAULT_TOKEN_BUDGET == 500_000
    assert out["rounds"] is rec["rounds"] is R.DEFAULT_ROUNDS is None and rec["ended_by"] == "submit"
    row = C.summarize([tmp_path / "d.json"])[0]
    assert not row["smoke"] and "SMOKE" not in C.report([row]) and "rounds no limit" in C.report([row])
    r = run("--rounds", "40", out="c.json")
    assert r.returncode == 0, r.stderr[-800:]
    row = C.summarize([tmp_path / "c.json"])[0]
    assert row["rounds"] == 40 and row["smoke"] and "SMOKE RUN, rounds=40" in C.report([row])
    r = run("--token-budget", "0", "--rounds", "5", out="n.json")
    assert r.returncode == 0, r.stderr[-800:]
    assert json.loads((tmp_path / "n.json").read_text())["cases"][0]["token_budget"] is None
    row = C.summarize([tmp_path / "n.json"])[0]
    assert row["smoke"] and "token budget none" in C.report([row])
    r = run("--token-budget", "0", out="z.json")
    assert r.returncode == 1 and "needs a limit" in r.stderr
    r = run("--token-budget", "-1", out="x.json")
    assert r.returncode == 1 and "0 for none" in r.stderr


def test_an_episode_at_the_defaults_has_no_round_cap_and_ends_on_the_budget(tmp_path):
    """The defaults, mocked usage of 10k output per call: past the 40 rounds
    2.1-beta allowed, the episode plays on until the budget is spent -- after
    the 50th call (500k), checked between calls, so no 51st call is made --
    and the record says the budget ended it. Each observation counts the
    rounds with no "/max"."""
    import harness.run as R
    call = model([PY], [10_000])
    out = E.run_episode(CASE, tmp_path / "w", call, max_rounds=R.DEFAULT_ROUNDS, exec_timeout=60,
                        token_budget=R.DEFAULT_TOKEN_BUDGET)
    assert len(call.seen) == 50 and out["ended_by"] == "token_budget" and out["output_tokens"] == 500_000
    assert [r["action"] for r in out["rounds"]] == ["exec"] * 50 + ["token_budget_spent", "no_submission"]
    assert call.seen[45]["last"].startswith("Round 45 \u00b7 output tokens 450,000 / 500,000 -- exit 0")
    assert "at most 500,000 output tokens, your thinking included" in call.seen[0]["system"]
    assert json.loads((tmp_path / "w_log/episode.json").read_text())["ended_by"] == "token_budget"


def test_an_episode_with_no_limit_at_all_is_refused(tmp_path):
    with pytest.raises(ValueError, match="needs a limit"):
        E.run_episode(CASE, tmp_path / "w", model([PY], [1]), max_rounds=None, exec_timeout=60)


def test_each_provider_names_the_field_it_counts(monkeypatch):
    import harness.run as R
    from types import SimpleNamespace as NS
    import anthropic, openai
    monkeypatch.setattr(R, "_MODEL_INFO", {})
    monkeypatch.setattr(anthropic, "Anthropic", lambda **_: NS())
    monkeypatch.setattr(openai, "OpenAI", lambda **_: NS())
    assert R.anthropic_call("claude-opus-5-5", None, [], "max").output_field == "usage.output_tokens (thinking included)"
    assert R.openai_responses_call("gpt-6-astra", None, [], "k", "max").output_field == "usage.output_tokens (reasoning included)"
    assert R.openai_compat_call("m", None, [], "k", "https://openrouter.ai/api/v1", "high").output_field == "usage.completion_tokens"
    assert R.DEFAULT_ROUNDS is None and R.DEFAULT_TOKEN_BUDGET == 500_000
