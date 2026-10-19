"""The episode's limits: rounds and an output-token budget.

Once the model's reasoning goes back to it (tests/test_reasoning_carry.py)
it no longer re-derives it every round, and the rounds bind. Since
2026-10-18 (2.1) the default is 60 rounds and 500,000 output tokens,
and the prompt states both and that no submission scores 0; 2.1-beta
(2026-10-16) had 40 rounds and 500,000.

The episode ends at the first of a submission, the rounds and the budget.
The budget counts the output tokens (thinking included) of every call the
episode played -- not a reply the harness threw away and asked again for --
is checked between calls and never inside one, and every observation says
what is left.
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


def test_the_defaults_are_60_rounds_and_500k_and_are_the_reportable_setting(tmp_path):
    """Since 2026-10-18 a run with no limit flags has 60 rounds and a 500,000
    output-token budget: the banner, the file and every record say both, and
    core_score reports it (no SMOKE). Other rounds, or no budget, are a
    different setting (SMOKE); no limit at all, or a negative one, is refused."""
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
    assert "rounds 60 " in r.stdout and "token-budget 500,000 " in r.stdout
    out = json.loads((tmp_path / "d.json").read_text()); rec = out["cases"][0]
    assert out["token_budget"] == rec["token_budget"] == R.DEFAULT_TOKEN_BUDGET == 500_000
    assert out["rounds"] == rec["rounds"] == R.DEFAULT_ROUNDS == 60 and rec["ended_by"] == "submit"
    assert rec["tokens"]["output_discarded"] == rec["tokens"]["calls_discarded"] == 0
    row = C.summarize([tmp_path / "d.json"])[0]
    assert not row["smoke"] and "SMOKE" not in C.report([row]) and "rounds 60 " in C.report([row])
    r = run("--rounds", "40", out="c.json")
    assert r.returncode == 0, r.stderr[-800:]
    row = C.summarize([tmp_path / "c.json"])[0]
    assert row["rounds"] == 40 and row["smoke"] and "SMOKE RUN, rounds=40" in C.report([row])
    r = run("--token-budget", "0", out="n.json")
    assert r.returncode == 0, r.stderr[-800:]
    assert json.loads((tmp_path / "n.json").read_text())["cases"][0]["token_budget"] is None
    row = C.summarize([tmp_path / "n.json"])[0]
    assert row["smoke"] and "token budget none" in C.report([row])
    r = run("--token-budget", "0", "--rounds", "0", out="z.json")
    assert r.returncode == 1 and "needs a limit" in r.stderr
    r = run("--token-budget", "-1", out="x.json")
    assert r.returncode == 1 and "0 for none" in r.stderr


def test_the_prompt_states_both_limits_and_that_no_submission_scores_0(tmp_path):
    """Alex's review (2026-10-18): 61 of the 229 episodes that used all 40
    rounds scored 0 for not submitting, and the prompt never said so."""
    import harness.run as R
    call = model([SUBMIT], [1_000])
    E.run_episode(CASE, tmp_path / "w", call, max_rounds=R.DEFAULT_ROUNDS, exec_timeout=60,
                  token_budget=R.DEFAULT_TOKEN_BUDGET)
    system = call.seen[0]["system"]
    assert system.startswith("You are solving a CAD task in a working directory, over at most 60 rounds "
                             "and 500,000 output tokens, your thinking included -- the episode ends at "
                             "whichever runs out first.\n")
    assert "An episode that\n             ends without a ```submit has no answer and scores 0." in system


def test_an_episode_at_the_defaults_plays_60_rounds(tmp_path):
    """The defaults, mocked usage of 5k output per call: past the 40 rounds
    2.1-beta allowed, the episode plays to round 60 (300k, inside the budget)
    and the rounds end it; every observation counts both limits."""
    import harness.run as R
    call = model([PY], [5_000])
    out = E.run_episode(CASE, tmp_path / "w", call, max_rounds=R.DEFAULT_ROUNDS, exec_timeout=60,
                        token_budget=R.DEFAULT_TOKEN_BUDGET)
    assert len(call.seen) == 60 and out["ended_by"] == "rounds" and out["output_tokens"] == 300_000
    assert call.seen[45]["last"].startswith("Round 45/60 \u00b7 output tokens 225,000 / 500,000 -- exit 0")
    assert json.loads((tmp_path / "w_log/episode.json").read_text())["ended_by"] == "rounds"


def test_an_episode_at_the_defaults_stops_at_500k_output(tmp_path):
    """150k output per call: the budget is spent after the 4th call (600k),
    checked between calls, so no 5th call is made; the 60 rounds are far off."""
    import harness.run as R
    call = model([PY], [150_000])
    out = E.run_episode(CASE, tmp_path / "w", call, max_rounds=R.DEFAULT_ROUNDS, exec_timeout=60,
                        token_budget=R.DEFAULT_TOKEN_BUDGET)
    assert len(call.seen) == 4 and out["ended_by"] == "token_budget" and out["output_tokens"] == 600_000
    assert call.seen[3]["last"].startswith("Round 3/60 \u00b7 output tokens 450,000 / 500,000 -- exit 0")


def test_with_no_round_limit_the_budget_alone_ends_it(tmp_path):
    """--rounds 0 (a smoke setting): no round cap; 10k per call, so the budget
    ends it after the 50th call, past the 40 and short of the 60 rounds."""
    call = model([PY], [10_000])
    out = E.run_episode(CASE, tmp_path / "w", call, max_rounds=None, exec_timeout=60, token_budget=500_000)
    assert len(call.seen) == 50 and out["ended_by"] == "token_budget" and out["output_tokens"] == 500_000
    assert call.seen[45]["last"].startswith("Round 45 \u00b7 output tokens 450,000 / 500,000 -- exit 0")


def test_an_episode_with_no_limit_at_all_is_refused(tmp_path):
    with pytest.raises(ValueError, match="needs a limit"):
        E.run_episode(CASE, tmp_path / "w", model([PY], [1]), max_rounds=None, exec_timeout=60)


def test_a_reply_the_harness_asks_again_for_is_not_counted(monkeypatch):
    """Alex's review (2026-10-18): a reply with no executable block is thrown
    away and asked again for (drive, rule 1) -- the model never sees it as a
    round -- and it used to count against the budget. Its usage is now marked
    discarded: still in the record's tokens, not in the budget."""
    import harness.run as R
    monkeypatch.setattr(R, "BACKOFF_S", 0)
    usage, replies = [], iter(["I will now think about the part.", PY])

    def send(system, turns, drop_images):
        usage.append({"input_tokens": 10, "output_tokens": 100_000})
        return next(replies)
    call = R.drive(send, "test")
    call.usage = usage
    assert call("S", [{"role": "user", "text": "Begin.", "images": []}]) == PY
    assert E.output_tokens(call) == 100_000
    assert usage[0]["discarded"] == "no executable block; asked again" and "discarded" not in usage[1]


def test_an_infrastructure_failure_asked_again_is_not_counted(tmp_path):
    """A call that fails for the network's sake is asked again for the same
    round; output it reported before failing is not the model's spend."""
    seen, usage = [], []

    def call(system, turns, plain=False):
        seen.append(1)
        usage.append({"input_tokens": 10, "output_tokens": 200_000})
        if len(seen) == 1:
            raise ConnectionError("stream reset by peer")
        return SUBMIT
    call.usage, call.context_tokens = usage, 1_000_000
    out = E.run_episode(CASE, tmp_path / "w", call, max_rounds=60, exec_timeout=60, token_budget=500_000)
    assert out["submitted"] and out["output_tokens"] == 200_000
    assert usage[0]["discarded"].startswith("infrastructure failure (ConnectionError)")


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
    assert R.DEFAULT_ROUNDS == 60 and R.DEFAULT_TOKEN_BUDGET == 500_000
