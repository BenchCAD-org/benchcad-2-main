"""harness/run.py gemini_call: the google-genai adapter, against a fake client.

The fake replaces only genai.Client; every chunk it streams is a REAL
google.genai.types.GenerateContentResponse, so the attribute paths the adapter
reads (candidates[0].content.parts[*].thought / thought_signature,
usage_metadata.thoughts_token_count, prompt_feedback.block_reason, the
FinishReason enum) are the SDK's, not a stand-in's.

Each test fails on the adapter before this change (a unary generate_content
with no effort, output counted as candidates only, r.text returned as is).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT))
import run as R                                                  # noqa: E402

from google import genai                                         # noqa: E402
from google.genai import errors, types                           # noqa: E402

FENCE = "```submit\nresult = 1\n```"
TURNS = [{"role": "user", "text": "Begin.", "images": []}]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda *_: None)


def chunk(*parts, finish=None, usage=None, block=None):
    """One streamed GenerateContentResponse. parts: (text, thought, signature)."""
    cands = []
    if parts or finish:
        cands = [types.Candidate(
            content=types.Content(role="model", parts=[
                types.Part(text=t, thought=th or None, thought_signature=sig) for t, th, sig in parts]),
            finish_reason=finish)]
    return types.GenerateContentResponse(
        candidates=cands or None,
        usage_metadata=types.GenerateContentResponseUsageMetadata(**usage) if usage else None,
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block) if block else None)


USAGE = {"prompt_token_count": 1000, "cached_content_token_count": 400,
         "candidates_token_count": 50, "thoughts_token_count": 200}


def ok_stream(text=FENCE, sig=b"sig-1"):
    return [chunk(("thinking about it", True, None)),
            chunk((text, False, None)),
            chunk(("", False, sig), finish="STOP", usage=USAGE)]


class FakeGenai:
    """genai.Client stand-in: each generate_content_stream call takes the next
    outcome -- a list of chunks (streamed) or an Exception (raised on the
    first read, where the SDK makes the request)."""

    def __init__(self, monkeypatch, *outcomes):
        self.outcomes, self.calls, self.client_kwargs = list(outcomes), [], None
        fake = self

        class Models:
            def generate_content_stream(self, model, contents, config):
                fake.calls.append({"model": model, "contents": contents, "config": config})
                out = fake.outcomes[min(len(fake.calls) - 1, len(fake.outcomes) - 1)]

                def gen():
                    if isinstance(out, Exception):
                        raise out
                    yield from out
                return gen()

        class Client:
            def __init__(self, **kw):
                fake.client_kwargs = kw
                self.models = Models()
        monkeypatch.setattr(genai, "Client", Client)


def bad_level(msg):
    return errors.ClientError(400, {"error": {"code": 400, "message": msg, "status": "INVALID_ARGUMENT"}})


def test_usage_counts_thoughts_as_output_and_the_cache_and_returns_only_the_answer(monkeypatch):
    FakeGenai(monkeypatch, ok_stream())
    usage: list = []
    out = R.gemini_call("gemini-3.8-flash", None, usage, "k")("sys", TURNS)
    assert out == FENCE, "a thought part is never the answer"
    u = {k: v for k, v in usage[0].items() if k != "seconds"}
    assert u == {"input_tokens": 1000, "cached_tokens": 400, "output_tokens": 250, "thoughts_tokens": 200}


def test_the_log_lines_are_the_ones_the_watchers_parse(monkeypatch, capsys):
    """run's gm_watch.py reads $ and empty replies from these two lines; the
    format is the Anthropic path's (usage: prompt N (cached C, ...), output O,
    S s / reasoning R chars, content K, stop=...)."""
    FakeGenai(monkeypatch, ok_stream())
    R.gemini_call("gemini-3.8-flash", None, [], "k")("sys", TURNS)
    log = capsys.readouterr().out
    assert re.search(r"^      usage: prompt 1,000 \(cached 400, thoughts 200\), output 250, \d+ s$", log, re.M), log
    assert re.search(r"^      reasoning 17 chars, content %d, stop=STOP$" % len(FENCE), log, re.M), log


def test_effort_is_sent_as_thinking_level_with_thoughts_on_and_the_model_ceiling(monkeypatch):
    fake = FakeGenai(monkeypatch, ok_stream())
    call = R.gemini_call("gemini-3.8-flash", None, [], "k")          # unset = the top, high
    call("sys", TURNS)
    cfg = fake.calls[0]["config"]
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.HIGH
    assert cfg.thinking_config.include_thoughts is True
    assert cfg.max_output_tokens == R.GEMINI_MAX_OUTPUT == 65_536
    assert cfg.system_instruction == "sys"
    assert call.effort_sent() == "high"
    fake2 = FakeGenai(monkeypatch, ok_stream())
    R.gemini_call("gemini-3.8-flash", None, [], "k", "low")("sys", TURNS)
    assert fake2.calls[0]["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_a_rejected_level_steps_down_and_the_record_says_what_was_sent(monkeypatch, capsys):
    """Measured 2026-10-08: a model without a level 400s naming it. One step
    down per rejection, remembered for the episode; 2.5 has no thinking_level
    at all and is then sent none."""
    fake = FakeGenai(monkeypatch, bad_level("Thinking level HIGH is not supported for this model. "
                                            "Please retry with other thinking level."), ok_stream())
    call = R.gemini_call("gemini-x", None, [], "k", "high")
    assert call("sys", TURNS) == FENCE
    assert [c["config"].thinking_config.thinking_level for c in fake.calls] == \
        [types.ThinkingLevel.HIGH, types.ThinkingLevel.MEDIUM]
    assert call.effort_sent() == "medium"
    assert "rejected thinking_level=HIGH; sending MEDIUM instead" in capsys.readouterr().out
    fake = FakeGenai(monkeypatch, bad_level("Thinking level is not supported for this model."), ok_stream())
    call = R.gemini_call("gemini-2.5-flash", None, [], "k", "medium")
    assert call("sys", TURNS) == FENCE
    assert fake.calls[1]["config"].thinking_config.thinking_level is None
    assert call.effort_sent() is None


def test_another_400_is_not_a_step_down(monkeypatch):
    fake = FakeGenai(monkeypatch, bad_level("Request contains an invalid argument."))
    with pytest.raises(errors.ClientError):
        R.gemini_call("gemini-3.8-flash", None, [], "k")("sys", TURNS)
    assert len(fake.calls) == 1


def test_max_tokens_with_no_block_is_truncated_and_retried_once_with_room(monkeypatch):
    cut = [chunk(("First I will measure the", False, None), finish="MAX_TOKENS", usage=USAGE)]
    fake = FakeGenai(monkeypatch, cut, ok_stream())
    usage: list = []
    assert R.gemini_call("gemini-3.8-flash", 8000, usage, "k")("sys", TURNS) == FENCE
    assert [c["config"].max_output_tokens for c in fake.calls] == [8000, 16000]
    assert usage[0]["truncated"] is True and "truncated" not in usage[1]
    # at the model's own ceiling there is no room to give: handed back, marked
    fake = FakeGenai(monkeypatch, cut)
    usage = []
    R.gemini_call("gemini-3.8-flash", None, usage, "k")("sys", TURNS)
    assert all(u["truncated"] for u in usage) and \
        {c["config"].max_output_tokens for c in fake.calls} == {65_536}


@pytest.mark.parametrize("finish", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "OTHER"])
def test_a_policy_finish_is_deterministic_and_named(monkeypatch, finish):
    fake = FakeGenai(monkeypatch, [chunk(("", False, None), finish=finish, usage=USAGE)])
    with pytest.raises(R.ReplyBlocked, match=f"finish_reason={finish}"):
        R.gemini_call("gemini-3.8-flash", None, [], "k")("sys", TURNS)
    assert len(fake.calls) == 1, "the same request draws the same verdict"


def test_a_blocked_prompt_is_logged_and_not_repeated(monkeypatch, capsys):
    fake = FakeGenai(monkeypatch, [chunk(usage={"prompt_token_count": 10}, block="PROHIBITED_CONTENT")])
    with pytest.raises(R.ReplyBlocked, match="block_reason=PROHIBITED_CONTENT"):
        R.gemini_call("gemini-3.8-flash", None, [], "k")("sys", TURNS)
    assert len(fake.calls) == 1
    assert "      blocked: block_reason=PROHIBITED_CONTENT" in capsys.readouterr().out


def test_the_models_earlier_turn_goes_back_with_its_thought_signature(monkeypatch):
    fake = FakeGenai(monkeypatch, ok_stream(sig=b"sig-A"), ok_stream(text=FENCE + " again"))
    call = R.gemini_call("gemini-3.8-flash", None, [], "k")
    first = call("sys", TURNS)
    call("sys", TURNS + [{"role": "assistant", "text": first, "images": []},
                         {"role": "user", "text": "Round 2.", "images": []}])
    model_turn = fake.calls[1]["contents"][1]
    assert model_turn.role == "model"
    assert [(p.text, p.thought_signature) for p in model_turn.parts] == [(FENCE, b"sig-A")]
    # a turn the episode rewrote (a summary) goes back as plain text
    call("sys", TURNS + [{"role": "assistant", "text": "summary", "images": []}])
    assert fake.calls[2]["contents"][1].parts[0].thought_signature is None


def test_the_clocks_are_the_other_first_party_paths(monkeypatch):
    """HttpOptions.timeout (ms) is the per-effort budget -- google-genai also
    sends it as X-Server-Timeout -- and every request's httpx read timeout is
    FIRST_PARTY_IDLE_S, so a silent stream is caught in a minute."""
    fake = FakeGenai(monkeypatch, ok_stream())
    R.gemini_call("gemini-3.8-flash", None, [], "k", "low")
    opts = fake.client_kwargs["http_options"]
    assert opts.timeout == R.CALL_BUDGET_S["low"] * 1000
    req = opts.httpx_client.build_request("POST", "https://x/v1beta/models/m:streamGenerateContent", timeout=None)
    to = req.extensions["timeout"]
    assert to["read"] == R.FIRST_PARTY_IDLE_S and to["connect"] == 30.0


def test_a_stream_past_its_budget_is_given_up(monkeypatch):
    FakeGenai(monkeypatch, ok_stream())
    clock = iter([0.0] + [10_000.0] * 50)
    monkeypatch.setattr(R.time, "time", lambda: next(clock))
    with pytest.raises(R.CallOverBudget):
        R.gemini_call("gemini-3.8-flash", None, [], "k", "low")("sys", TURNS)


def test_build_call_hands_gemini_its_effort(monkeypatch):
    fake = FakeGenai(monkeypatch, ok_stream())
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    call = R.build_call("gemini/gemini-3.8-flash", None, [], Path("."), "medium")
    call("sys", TURNS)
    assert fake.calls[0]["config"].thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert call.effort_sent() == "medium"


# ── auth: Gemini API key, Vertex express key, Vertex ADC, base URL ─────────────

def test_auth_routes_and_labels_carry_no_secret():
    assert R.gemini_auth({"GEMINI_API_KEY": "s1", "GOOGLE_API_KEY": "s2"}) == ({"api_key": "s1"}, "gemini-api-key")
    assert R.gemini_auth({"GOOGLE_API_KEY": "s2"}) == ({"api_key": "s2"}, "gemini-api-key")
    kw, label = R.gemini_auth({"GOOGLE_GENAI_USE_VERTEXAI": "true", "GOOGLE_API_KEY": "s3",
                               "GOOGLE_CLOUD_PROJECT": "p"})
    assert kw == {"vertexai": True, "api_key": "s3"} and label == "vertex-express-key"
    kw, label = R.gemini_auth({"GOOGLE_GENAI_USE_VERTEXAI": "True", "GOOGLE_CLOUD_PROJECT": "p"})
    assert kw == {"vertexai": True, "project": "p", "location": "global"}
    assert label == "vertex-adc project=p location=global"
    kw, label = R.gemini_auth({"GEMINI_API_KEY": "s1", "GEMINI_BASE_URL": "https://gw.example/v"})
    assert kw == {"api_key": "s1", "base_url": "https://gw.example/v"}
    assert label == "gemini-api-key base_url=https://gw.example/v"
    for env in ({}, {"GOOGLE_GENAI_USE_VERTEXAI": "1"}):
        with pytest.raises(SystemExit):
            R.gemini_auth(env)
    for env in ({"GEMINI_API_KEY": "secret-x"}, {"GOOGLE_GENAI_USE_VERTEXAI": "1", "GOOGLE_API_KEY": "secret-x"}):
        assert "secret-x" not in R.gemini_auth(env)[1]


def test_the_client_gets_the_route_and_the_base_url(monkeypatch):
    fake = FakeGenai(monkeypatch, ok_stream())
    R.gemini_call("gemini-3.8-flash", None, [], {"vertexai": True, "api_key": "s3",
                                                 "base_url": "https://gw.example/v"})("sys", TURNS)
    kw = fake.client_kwargs
    assert kw["vertexai"] is True and kw["api_key"] == "s3" and "base_url" not in kw
    assert kw["http_options"].base_url == "https://gw.example/v"


def test_build_call_takes_vertex_without_a_gemini_key_and_labels_the_record(monkeypatch):
    fake = FakeGenai(monkeypatch, ok_stream())
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    call = R.build_call("gemini/gemini-3.8-flash", None, [], Path("."))
    call("sys", TURNS)
    assert fake.client_kwargs["project"] == "proj" and call.auth == "vertex-adc project=proj location=global"
