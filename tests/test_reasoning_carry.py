"""The model's reasoning goes back to it on the next round (2026-10-15).

Anthropic's review of Core (2026-10-15): each round the harness sent the
model only its earlier text, so it worked its reasoning out again every
round. Each provider now sends a turn back as the API returned it:
Anthropic's thinking blocks with their signatures, OpenAI's reasoning items
with their encrypted_content, OpenRouter's reasoning_details (Gemini already
replayed its thought_signature). Each test fails on the harness before 2026-10-15.

The Anthropic blocks are bound to the conversation they were produced in, so
the conversation a request carries must be the previous request's plus the
new turns, byte for byte: an image is encoded once, when it is first sent.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CADENV_LOCAL", "1")
import harness.run as R                                          # noqa: E402

FENCE = "```submit\nresult = 1\n```"
TURNS = [{"role": "user", "text": "Begin.", "images": []}]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda *_: None)
    monkeypatch.setattr(R, "_MODEL_INFO", {})


def _resp(status: int):
    import httpx
    return httpx.Response(status, request=httpx.Request("POST", "https://x/v1/messages"))


def fake_anthropic(monkeypatch, replies, *, on_stream=None, transformations=None):
    """anthropic.Anthropic -> a client that answers request n with replies[n]:
    a thinking block signed `sig<n>` and the text. Records a deep copy of
    every request. `on_stream(kw, n)` may raise; `transformations(n)` gives
    the input_transformations of reply n."""
    import anthropic
    from anthropic.types import TextBlock, ThinkingBlock
    sent: list[dict] = []

    class Stream:
        def __init__(self, **kw):
            sent.append(copy.deepcopy(kw))
            if on_stream:
                on_stream(kw, len(sent))
            n = len([k for k in sent if not k.get("_failed")])
            self.n = n
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __iter__(self): return iter(())
        def get_final_message(self):
            text = replies[min(self.n, len(replies)) - 1]
            return NS(usage=NS(input_tokens=10, output_tokens=1000, cache_read_input_tokens=0,
                               cache_creation_input_tokens=0),
                      stop_reason="end_turn", stop_details=None,
                      input_transformations=(transformations(self.n) if transformations else []),
                      content=[ThinkingBlock(type="thinking", thinking="", signature=f"sig{self.n}"),
                               TextBlock(type="text", text=text)])

    class Client:
        def __init__(self, **_):
            self.messages = NS(stream=Stream)
            self.beta = NS(messages=NS(stream=Stream))

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    return sent


def _png(path: Path, w: int, h: int, colour) -> Path:
    from PIL import Image
    Image.new("RGB", (w, h), colour).save(path)
    return path


def _size(b64: str):
    from PIL import Image
    with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
        return im.size


# ── Anthropic ─────────────────────────────────────────────────────────────────

def test_an_episode_replays_every_earlier_thinking_block_on_an_unchanged_prefix(tmp_path, monkeypatch):
    """A real episode (local sandbox) of 14 rounds whose programs write two
    PNGs a round, one of them over MANY_IMAGE_PX: the conversation passes
    MANY_IMAGES on the way. Every request is the previous request plus the
    new turns, byte for byte -- the same system prompt, the same images in
    the same bytes they were first sent in -- and the turn appended is the
    previous reply's content blocks, its thinking and signature included. So
    the last request carries every earlier thinking block, and the API has
    nothing to drop."""
    from envs.common import episode as E
    from envs.common import sandbox as S
    monkeypatch.setattr(S, "_docker_ready", lambda: False)
    rounds = 14
    replies = [("```python\nfrom PIL import Image\n"
                f"Image.new('RGB', (2400, 700), ({n * 9}, 40, 90)).save('wide.png')\n"
                f"Image.new('RGB', (600, 400), (10, {n * 9}, 90)).save('small.png')\n"
                "print('ok')\n```") for n in range(1, rounds)]
    replies.append("```submit\nimport cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)\n```")
    sent = fake_anthropic(monkeypatch, replies)
    usage: list = []
    call = R.anthropic_call("claude-opus-5-5", None, usage, "max")
    call.usage, call.context_tokens = usage, 1_000_000
    out = E.run_episode(ROOT / "tests/fixtures/t3/case1", tmp_path / "w", call,
                        max_rounds=rounds + 2, exec_timeout=120)
    assert out["submitted"] and len(sent) == rounds

    def images(req):
        return [b["source"]["data"] for m in req["messages"] for b in m["content"] if b["type"] == "image"]
    assert len(images(sent[-1])) > R.MANY_IMAGES >= len(images(sent[2]))
    for k in range(1, rounds):
        prev, cur = sent[k - 1], sent[k]
        assert cur["system"] == prev["system"]
        n = len(prev["messages"])
        assert json.dumps(cur["messages"][:n]) == json.dumps(prev["messages"]), f"request {k + 1} edited the history"
        assert cur["messages"][n] == {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "", "signature": f"sig{k}"},
            {"type": "text", "text": replies[k - 1]}]}
        assert cur["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
        assert cur["betas"] == [R.THINKING_BINDING_BETA]
    last = [b["signature"] for m in sent[-1]["messages"] for b in m["content"] if b["type"] == "thinking"]
    assert last == [f"sig{k}" for k in range(1, rounds)]
    # the wide image went at the bound from its first request on, under MANY_IMAGES too
    assert max(max(_size(b)) for req in sent for b in images(req)) == R.MANY_IMAGE_PX
    assert call.edits == {"images_left_out": 0, "retries_without_images": 0}
    assert not any(u.get("thinking_dropped") for u in usage)


def test_a_retried_request_and_a_round_asked_again_carry_the_same_blocks(monkeypatch):
    """drive() repeating request 2 after a transient failure, and the episode
    asking a round again after an infrastructure failure, send the same
    conversation: reply 1's thinking block, byte for byte, both times."""
    fails = {"n": 0}

    def flaky(kw, n):
        if n >= 2 and fails["n"] < R.ATTEMPTS + 1:
            fails["n"] += 1
            kw["_failed"] = True
            raise ConnectionError("Connection error.")
    sent = fake_anthropic(monkeypatch, [FENCE.replace("1", "a"), FENCE], on_stream=flaky)
    call = R.anthropic_call("claude-opus-5-5", None, [], "max")
    reply = call("sys", TURNS)
    turns = TURNS + [{"role": "assistant", "text": reply, "images": []},
                     {"role": "user", "text": "Round 1/30 -- exit 0", "images": []}]
    with pytest.raises(ConnectionError):
        call("sys", turns)                                       # every attempt fails: the round's call fails
    assert call("sys", turns) == FENCE                           # the episode asks the round again
    carried = {"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": "sig1"},
                                                {"type": "text", "text": FENCE.replace("1", "a")}]}
    assert len(sent) == 1 + (R.ATTEMPTS + 1) + 1               # reply 1, the failures, the answer
    assert all(req["messages"][1] == carried for req in sent[1:])
    assert all(json.dumps(req["messages"]) == json.dumps(sent[1]["messages"]) for req in sent[1:])


def test_a_summary_call_sends_text_only(monkeypatch):
    """Episode's summarisation asks under its own system prompt, where no
    replayed block was produced: those calls (plain) send each turn as text.
    The next real round sends the blocks again."""
    sent = fake_anthropic(monkeypatch, [FENCE, "Summary.", FENCE])
    call = R.anthropic_call("claude-opus-5-5", None, [], "max")
    reply = call("sys", TURNS)
    turns = TURNS + [{"role": "assistant", "text": reply, "images": []},
                     {"role": "user", "text": "obs", "images": []}]
    call("handover", turns, plain=True)
    assert sent[1]["messages"][1] == {"role": "assistant", "content": [{"type": "text", "text": FENCE}]}
    call("sys", turns)
    assert sent[2]["messages"][1]["content"][0]["type"] == "thinking"


def test_thinking_off_carries_nothing_and_binds_nothing(monkeypatch):
    sent = fake_anthropic(monkeypatch, [FENCE])
    R.anthropic_call("claude-opus-5", None, [], "none")("sys", TURNS)
    assert sent[0]["thinking"] == {"type": "disabled"} and "betas" not in sent[0]


def test_what_the_api_dropped_is_recorded(monkeypatch, capsys):
    """The binding beta's input_transformations: each dropped block is
    counted on the call's usage record (the run's tokens.thinking_dropped)
    and kept as the API reported it."""
    from anthropic.types.beta import BetaThinkingDroppedInputTransformation as Dropped
    usage: list = []
    fake_anthropic(monkeypatch, [FENCE, FENCE], transformations=lambda n: [] if n == 1 else [
        Dropped(type="thinking_dropped", path="messages.1.content.0", reason="prefix_binding_mismatch")])
    call = R.anthropic_call("claude-opus-5-5", None, usage, "max")
    reply = call("sys", TURNS)
    call("sys", TURNS + [{"role": "assistant", "text": reply, "images": []},
                         {"role": "user", "text": "obs", "images": []}])
    assert "thinking_dropped" not in usage[0]
    assert usage[1]["thinking_dropped"] == 1
    assert usage[1]["input_transformations"] == [{"type": "thinking_dropped", "path": "messages.1.content.0",
                                                  "reason": "prefix_binding_mismatch"}]
    assert "earlier thinking dropped 1" in capsys.readouterr().out


def test_an_endpoint_without_the_binding_beta_and_a_refused_signature(monkeypatch, capsys):
    """A 400 naming the binding: sent again without it, for the episode. A
    400 on a replayed block's signature (an enforced account the binding
    could not be asked for): the API's documented recovery -- those turns
    go back as text from then on, the newer ones keep their blocks."""
    import anthropic

    def no_beta(kw, n):
        if "betas" in kw:
            kw["_failed"] = True
            raise anthropic.BadRequestError("thinking.block_binding: Extra inputs are not permitted",
                                            response=_resp(400), body=None)
    sent = fake_anthropic(monkeypatch, [FENCE, FENCE], on_stream=no_beta)
    call = R.anthropic_call("claude-opus-5-5", None, [], "max")
    reply = call("sys", TURNS)
    call("sys", TURNS + [{"role": "assistant", "text": reply, "images": []},
                         {"role": "user", "text": "obs", "images": []}])
    assert [("betas" in kw) for kw in sent] == [True, False, False]
    assert sent[-1]["thinking"] == {"type": "adaptive"}
    assert "rejected the thinking binding controls" in capsys.readouterr().out

    def bad_sig(kw, n):
        if n == 2:
            kw["_failed"] = True
            raise anthropic.BadRequestError("messages.1.content.0: Invalid `signature` in `thinking` block. "
                                            "The block is bound to a different conversation.",
                                            response=_resp(400), body=None)
    sent = fake_anthropic(monkeypatch, [FENCE.replace("1", "a"), FENCE.replace("1", "b"), FENCE], on_stream=bad_sig)
    call = R.anthropic_call("claude-opus-5-5", None, [], "max")
    r1 = call("sys", TURNS)
    turns = TURNS + [{"role": "assistant", "text": r1, "images": []}, {"role": "user", "text": "o1", "images": []}]
    r2 = call("sys", turns)
    assert sent[2]["messages"][1] == {"role": "assistant", "content": [{"type": "text", "text": r1}]}
    turns += [{"role": "assistant", "text": r2, "images": []}, {"role": "user", "text": "o2", "images": []}]
    call("sys", turns)
    assert sent[3]["messages"][1]["content"] == [{"type": "text", "text": r1}]
    assert sent[3]["messages"][3]["content"][0]["type"] == "thinking"
    assert "go back as text from now on" in capsys.readouterr().out


# ── the conversation is append-only ──────────────────────────────────────────

def test_an_image_is_sent_in_the_bytes_of_its_first_request(tmp_path):
    """SentImages: bounded once, at MANY_IMAGE_PX, and never re-encoded --
    not when the request passes MANY_IMAGES, not when the file changes on
    disk (a seed the model overwrote)."""
    wide = _png(tmp_path / "wide.png", 2400, 700, (1, 2, 3))
    images = R.SentImages()
    first = images(wide)
    assert _size(first) == (2000, 583)
    _png(wide, 300, 300, (9, 9, 9))
    assert images(wide) == first
    out, max_px = R.bound_images([{"role": "user", "text": "Begin.", "images": [str(wide)]}])
    assert max_px == R.MANY_IMAGE_PX


def test_the_last_resort_edits_are_counted(monkeypatch):
    """The two edits of a conversation already sent stay, as last resorts,
    and are counted: images left out past the API's ceiling, and drive()'s
    retry without images."""
    edits = R.prefix_edits()
    per = 7
    turns = [{"role": "user", "text": "Begin.", "images": ["seed.png"]}]
    for r in range(R.HARD_IMAGE_CAP // per + 2):
        turns += [{"role": "assistant", "text": "x", "images": []},
                  {"role": "user", "text": "o", "images": [f"r{r}_{k}.png" for k in range(per)]}]
    R.bound_images(turns, edits)
    assert edits == {"images_left_out": 1, "retries_without_images": 0}

    outcomes = iter([RuntimeError("400 invalid_image: could not process image"), FENCE])

    def send(system, turns, drop_images):
        o = next(outcomes)
        if isinstance(o, Exception):
            o.status_code = 400
            raise o
        return o
    call = R.drive(send, "t", edits)
    assert call("sys", [{"role": "user", "text": "hi", "images": ["a.png"]}]) == FENCE
    assert call.edits is edits and edits["retries_without_images"] == 1


# ── OpenAI (Responses API) ────────────────────────────────────────────────────

def fake_responses(monkeypatch, outputs):
    """responses.stream(...) answering request n with outputs[n]: a list of
    real output items (the final response's `output`)."""
    import openai
    sent: list[dict] = []

    class Stream:
        def __init__(self, **kw):
            sent.append(copy.deepcopy(kw))
            items = outputs[min(len(sent), len(outputs)) - 1]
            text = "".join(c.text for i in items if i.type == "message" for c in i.content)
            self.text = text
            self.final = NS(status="completed", incomplete_details=None, error=None, output=items,
                            usage=NS(input_tokens=100, input_tokens_details=NS(cached_tokens=0),
                                     output_tokens=500, output_tokens_details=NS(reasoning_tokens=400)))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __iter__(self):
            yield NS(type="response.output_text.delta", delta=self.text)
            yield NS(type="response.completed", response=self.final)
        def get_final_response(self): return self.final

    class Client:
        def __init__(self, **_):
            self.responses = NS(stream=lambda **kw: Stream(**kw))
    monkeypatch.setattr(openai, "OpenAI", Client)
    return sent


def _output(n: int, text: str, encrypted: bool = True):
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText, ResponseReasoningItem
    return [ResponseReasoningItem(id=f"rs_{n}", type="reasoning", summary=[],
                                  **({"encrypted_content": f"enc-{n}"} if encrypted else {})),
            ResponseOutputMessage(id=f"msg_{n}", type="message", role="assistant", status="completed",
                                  content=[ResponseOutputText(type="output_text", text=text, annotations=[])])]


def test_openai_replays_the_reasoning_items_statelessly(monkeypatch):
    """store=False, include reasoning.encrypted_content, and the next request
    carries reply 1's output items -- the reasoning item with its encrypted
    content, then the message -- exactly as the API returned them."""
    sent = fake_responses(monkeypatch, [_output(1, FENCE.replace("1", "a")), _output(2, FENCE)])
    call = R.openai_responses_call("gpt-6-astra", None, [], "k", "max")
    reply = call("sys", TURNS)
    assert sent[0]["store"] is False and sent[0]["include"] == ["reasoning.encrypted_content"]
    call("sys", TURNS + [{"role": "assistant", "text": reply, "images": []},
                         {"role": "user", "text": "obs", "images": []}])
    items = sent[1]["input"]
    assert items[1:3] == [i.to_dict(mode="json") for i in _output(1, FENCE.replace("1", "a"))]
    assert items[1] == {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "enc-1"}
    assert items[3]["role"] == "user"


def test_openai_reasoning_without_its_encrypted_content_goes_back_as_text(monkeypatch):
    sent = fake_responses(monkeypatch, [_output(1, FENCE, encrypted=False), _output(2, FENCE)])
    call = R.openai_responses_call("gpt-6-astra", None, [], "k", "max")
    reply = call("sys", TURNS)
    call("sys", TURNS + [{"role": "assistant", "text": reply, "images": []},
                         {"role": "user", "text": "obs", "images": []}])
    assert sent[1]["input"][1] == {"role": "assistant", "content": [{"type": "output_text", "text": FENCE}]}


# ── OpenRouter (chat.completions) ─────────────────────────────────────────────

def _chunk(content=None, details=None, finish=None):
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
    extra = {"reasoning_details": details} if details is not None else {}
    return ChatCompletionChunk(id="x", object="chat.completion.chunk", created=0, model="m", choices=[
        Choice(index=0, finish_reason=finish, delta=ChoiceDelta(role="assistant", content=content, **extra))])


def fake_chat(monkeypatch, *streams):
    import openai
    sent: list[dict] = []

    class Client:
        def __init__(self, **_):
            self.chat = NS(completions=NS(create=lambda **kw: (sent.append(copy.deepcopy(kw)),
                                                               iter(streams[min(len(sent), len(streams)) - 1]))[1]))
    monkeypatch.setattr(openai, "OpenAI", Client)
    return sent


def test_openrouter_replays_the_reasoning_details_merged_by_index(monkeypatch):
    """The details stream in pieces; the next request carries them whole,
    one per index, on the assistant message. xAI / OpenCode have no such
    field: their turns go back as text."""
    stream = [_chunk(details=[{"type": "reasoning.text", "text": "meas", "format": "f", "index": 0}]),
              _chunk(details=[{"type": "reasoning.text", "text": "ure", "index": 0, "signature": "s0"}]),
              _chunk(details=[{"type": "reasoning.encrypted", "data": "ZZ", "index": 1}]),
              _chunk(content=FENCE, finish="stop")]
    sent = fake_chat(monkeypatch, stream, [_chunk(content=FENCE, finish="stop")])
    call = R.openai_compat_call("anthropic/claude-opus-5-5", None, [], "k", "https://openrouter.ai/api/v1", "high")
    reply = call("sys", TURNS)
    turns = TURNS + [{"role": "assistant", "text": reply, "images": []}, {"role": "user", "text": "obs", "images": []}]
    call("sys", turns)
    assert sent[1]["messages"][2]["reasoning_details"] == [
        {"type": "reasoning.text", "text": "measure", "format": "f", "index": 0, "signature": "s0"},
        {"type": "reasoning.encrypted", "data": "ZZ", "index": 1}]
    sent = fake_chat(monkeypatch, stream, [_chunk(content=FENCE, finish="stop")])
    call = R.openai_compat_call("grok-4.5", None, [], "k", "https://api.x.ai/v1", None)
    call("sys", TURNS + [{"role": "assistant", "text": call("sys", TURNS), "images": []},
                         {"role": "user", "text": "obs", "images": []}])
    assert "reasoning_details" not in sent[1]["messages"][2]


def test_a_crossing_of_the_image_byte_cap_edits_the_conversation_once(tmp_path, monkeypatch):
    """Past REQUEST_IMAGE_BYTES the oldest observation images leave the request
    down to IMAGE_BYTES_LOW, and stay out: one edit per crossing, and every
    request until the next crossing is the previous one plus the new turns,
    byte for byte, its thinking blocks with it. (Leaving out just enough on
    every request edited every request from the first crossing on.)"""
    from PIL import Image
    shot = lambda n: (Image.effect_noise((300, 300), 64).convert("RGB").save(tmp_path / f"r{n}.png"),
                      str(tmp_path / f"r{n}.png"))[1]
    one = R._sent_bytes(Path(shot(0)), R.MANY_IMAGE_PX)
    monkeypatch.setattr(R, "REQUEST_IMAGE_BYTES", int(one * 8.5))        # the 9th image crosses
    monkeypatch.setattr(R, "IMAGE_BYTES_LOW", int(one * 5.5))            # a crossing comes down to 5
    rounds = 12
    sent = fake_anthropic(monkeypatch, [FENCE.replace("1", str(k)) for k in range(rounds + 1)])
    call = R.anthropic_call("claude-opus-5-5", None, [], "max")
    turns = [{"role": "user", "text": "Begin.", "images": []}]
    for r in range(1, rounds + 1):
        reply = call("sys", turns)
        turns += [{"role": "assistant", "text": reply, "images": []},
                  {"role": "user", "text": f"Round {r}/50 -- exit 0", "images": [shot(r)]}]
    call("sys", turns)
    n_img = lambda req: sum(b["type"] == "image" for m in req["messages"] for b in m["content"])
    edited = [k for k in range(1, len(sent))
              if json.dumps(sent[k]["messages"][:len(sent[k - 1]["messages"])]) != json.dumps(sent[k - 1]["messages"])]
    assert [n_img(req) for req in sent] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]
    assert edited == [9] and call.edits["images_left_out"] == 1
    last = [b["signature"] for m in sent[-1]["messages"] for b in m["content"] if b["type"] == "thinking"]
    assert last == [f"sig{k}" for k in range(1, rounds + 1)]
