#!/usr/bin/env python
"""One entry point for running a model against the cases.

    uv run python harness/run.py --model anthropic/claude-opus-5
    uv run python harness/run.py --model mock/oracle          # no key needed
    uv run python harness/run.py --model openai/gpt-5.5 --effort high --cases examples

`--model <provider>/<id>` picks the provider by prefix. Every provider speaks
the same `call_fn(system, turns) -> str` contract that envs.common.episode
defines, so episode.py is untouched: text and images only, never a provider's
own tool protocol, so scores stay comparable across vendors.

Providers and the key each one reads (first name that is set wins):

    anthropic/<id>    ANTHROPIC_API_KEY
    openai/<id>       OPENAI_API_KEY
    gemini/<id>       GEMINI_API_KEY, GOOGLE_API_KEY
    xai/<id>          XAI_API_KEY, GROK_API_KEY
    openrouter/<id>   OPENROUTER_API_KEY
    opencode/<id>     OPENCODE_ZEN_API_KEY, OPENCODE_API_KEY, ZEN_API_KEY
    mock/oracle       none - submits the reference back (gt/gt.step, or
                      gt/gt_graph.json for ECAD), for checking the plumbing.
                      It INLINES the whole reference into its own submission;
                      the largest example reference is 10.5 MB. The reply never
                      leaves this process.
    mock/dumb         none - submits a 10 mm box (or an empty graph), a
                      deliberate low score
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.common.episode import run_episode            # noqa: E402
from envs.common.score_case import fmt, score_case     # noqa: E402

# Thinking effort, as the API takes it. Measured against claude-opus-5 on
# 2026-09-11: `thinking.type.enabled` with a token budget is REJECTED for this
# model ("Use thinking.type.adaptive and output_config.effort"), and the
# accepted efforts are low | medium | high | max. Leaving it unset is not max --
# the same prompt spent 32 thinking tokens by default, 111 at effort=max -- so
# the default here is max and every run states it.
EFFORTS = ("none", "low", "medium", "high", "max")
DEFAULT_EFFORT = "max"
# OpenRouter takes low | medium | high; max maps onto high there.
OR_EFFORT = {"none": "none", "low": "low", "medium": "medium", "high": "high",
             "max": "high"}
# OpenAI's own knob on /chat/completions, `reasoning_effort`. Measured
# 2026-09-15 on gpt-5.4 and gpt-5.5: none | low | medium | high | xhigh are
# accepted and `minimal` / `max` are 400s, so max maps onto xhigh. Older
# families take fewer values (gpt-5: minimal..high, gpt-5.1: none..high, the
# o-series: low..high) and gpt-4.1 has no such parameter at all
# ("Unrecognized request argument supplied: reasoning_effort"), so a 400 that
# names the parameter steps the value down -- xhigh -> high -> not sent -- and
# the call is repeated; the step is remembered for the rest of the episode.
OPENAI_EFFORT = {"none": "none", "low": "low", "medium": "medium", "high": "high",
                 "max": "xhigh"}
# A formal run gets the full budget: 100 rounds at effort max. Smoke runs pass
# --rounds / --effort explicitly; the defaults are the contract the shipped
# eval.toml quotes.
DEFAULT_ROUNDS = 100
DEFAULT_MAX_TOKENS = 16000
ATTEMPTS = 4
BACKOFF_S = 5
# A reply with no content is retried once with this much more room, and on
# OpenRouter with the thinking bounded to this fraction of it, so content
# always has somewhere to go. See EmptyContent and openai_compat_call.
EMPTY_RETRY_BOOST = 2
REASONING_BOUND_FRAC = 0.6


class SkipCase(Exception):
    """This case is out of scope for the chosen model, with a reason."""


class EmptyContent(RuntimeError):
    """The provider returned a reply with no content, and why.

    Current hybrid reasoners spend `max_completion_tokens` on THINKING first
    and only then write the answer, so a budget that runs out mid-thought ends
    the call with `finish_reason="length"`, a full `reasoning` field and
    `content` empty. Measured 2026-09-11 on OpenRouter: qwen/qwen3.7-flash at
    max_completion_tokens=300 returned 1,183 characters of reasoning and no
    content at all.

    This is a FAILED call, not a turn, and it is transient by construction --
    it is raised so drive() retries it with room (see openai_compat_call). The
    message carries the attribution the log used to lack: before this, every
    such round printed only `reply has no executable block (len 0, tail '')`,
    which is indistinguishable from a model that wrote prose.
    """


@dataclass
class Provider:
    kind: str
    keys: tuple[str, ...] = ()
    base_url: str | None = None


PROVIDERS: dict[str, Provider] = {
    "anthropic/":  Provider("anthropic", ("ANTHROPIC_API_KEY",)),
    "openai/":     Provider("openai_compat", ("OPENAI_API_KEY",)),
    "gemini/":     Provider("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "xai/":        Provider("openai_compat", ("XAI_API_KEY", "GROK_API_KEY"),
                            "https://api.x.ai/v1"),
    "openrouter/": Provider("openai_compat", ("OPENROUTER_API_KEY",),
                            "https://openrouter.ai/api/v1"),
    "opencode/":   Provider("openai_compat",
                            ("OPENCODE_ZEN_API_KEY", "OPENCODE_API_KEY", "ZEN_API_KEY"),
                            "https://opencode.ai/zen/v1"),
    "mock/":       Provider("mock"),
}


def split_model(spec: str) -> tuple[str, Provider, str]:
    """'anthropic/claude-opus-5' -> ('anthropic/', Provider, 'claude-opus-5')."""
    for prefix, prov in PROVIDERS.items():
        if spec.startswith(prefix):
            model_id = spec[len(prefix):]
            if not model_id:
                raise SystemExit(f"--model {spec!r}: no model id after {prefix!r}")
            return prefix, prov, model_id
    raise SystemExit(
        f"--model {spec!r}: unknown provider prefix. Use one of: "
        + ", ".join(sorted(PROVIDERS)) + "\nSee --help for the key each one reads.")


def resolve_key(prefix: str, prov: Provider) -> str | None:
    """The first key that is set, or a message naming exactly which are missing."""
    if not prov.keys:
        return None
    for name in prov.keys:
        if os.environ.get(name):
            return os.environ[name]
    want = " or ".join(prov.keys)
    raise SystemExit(
        f"{prefix}* needs {want} in the environment; none of those is set.\n"
        f"Export one and retry, or use --model mock/oracle to exercise the "
        f"runner without any provider key.")


# The API's many-image rule, measured 2026-09-12 on claude-opus-5: a request
# with MORE than MANY_IMAGES image blocks (every turn's images count, seeds
# included) rejects any image with a dimension over MANY_IMAGE_PX with
# "exceed max allowed size for many-image requests"; 20 images of 2100 px
# pass, 21 do not, 21 of 2000 px pass. Every drawing sheet is 4200 px, so a
# long run with crops would fail every call from about round 7 on -- and
# drive()'s fallback would then drop ALL images, blinding the model for the
# rest of the episode. Two bounds keep a request under the rule:
#   KEEP_OBS_ROUNDS   observation images (crops, renders) older than this
#                     many rounds leave the request; their text stays, and
#                     the model can always crop again
#   MANY_IMAGE_PX     when a request still carries more than MANY_IMAGES
#                     images, each is downscaled to this on its longest side
#                     before encoding -- the API would have downscaled a
#                     4200 px sheet to ~2300 px anyway, so the cost is small
MANY_IMAGES = 20
MANY_IMAGE_PX = 2000
# An observation image (a render or plot the model made) stays in the
# request for this many rounds, then leaves (its text stays, and the model
# can make it again). Two rather than four: each round it stays is one more
# cached read of ~2-5k tokens per image, and a model that needs an old
# figure again has it on disk.
KEEP_OBS_ROUNDS = 2
# The downscale decision is made ONCE per episode, from the seed count, not
# per request from the live count. A per-request decision flipped as crops
# came and went: the seed images were re-encoded at a different size, the
# request's prefix changed from turn one, and the whole prompt cache was
# rewritten (1.25x) instead of read (0.1x). The seeds plus this many
# observation images per kept round is the request size the episode is
# sized for; above MANY_IMAGES it runs at MANY_IMAGE_PX from round one. On
# the high-resolution tier a 2300 px tile is downscaled to ~1970 px by the
# API anyway, so sending it at 2000 px loses nothing.
OBS_PER_ROUND = 2


def _b64(path: Path, max_px: int | None = None) -> str:
    """The PNG at `path`, base64; downscaled in memory to `max_px` on its
    longest side when it is larger than that (never written back)."""
    if max_px:
        import io
        from PIL import Image
        with Image.open(path) as im:
            if max(im.size) > max_px:
                k = max_px / max(im.size)
                buf = io.BytesIO()
                im.convert("RGB").resize((max(1, round(im.width * k)), max(1, round(im.height * k))),
                                         Image.LANCZOS).save(buf, "PNG")
                return base64.standard_b64encode(buf.getvalue()).decode()
    return base64.standard_b64encode(Path(path).read_bytes()).decode()


def image_label(path, label: str | None = None) -> str:
    """The text that precedes every image: its path in the working directory
    when the episode says it (`image_labels`, round one), else its file
    name. A round-one turn can carry twenty images; without a label the
    model cannot tell drawing_tile_r2c3.png from r3c4, and the prompt names
    files, not pictures. A label is the path the model must pass to
    tools.crop, so part_drawings/part_03.png, not part_03.png."""
    return f"[{label or Path(path).name}]"


def labelled(t: dict):
    """(path, label) per image of a turn."""
    imgs = t.get("images") or []
    labels = t.get("image_labels") or [None] * len(imgs)
    return list(zip(imgs, labels))


def bound_images(turns: list) -> tuple[list, int | None]:
    """The turns as a request should carry them: the first user turn keeps its
    images (the case's inputs), the last KEEP_OBS_ROUNDS user turns keep
    theirs, every other turn's images are dropped (text kept). Returns the
    trimmed turns and the per-image pixel limit to encode with (None when the
    request is under the many-image threshold)."""
    user_idx = [i for i, t in enumerate(turns) if t.get("role") != "assistant"]
    keep = set(user_idx[:1]) | set(user_idx[-KEEP_OBS_ROUNDS:])
    out = [dict(t, images=(t.get("images") or []) if i in keep else [],
                image_labels=(t.get("image_labels") or []) if i in keep else []) for i, t in enumerate(turns)]
    n_seed = len(out[user_idx[0]]["images"]) if user_idx else 0
    sized_for = n_seed + KEEP_OBS_ROUNDS * OBS_PER_ROUND
    n = sum(len(t["images"]) for t in out)
    # The live count still rules when a model crops more than the episode
    # was sized for: the API would reject the request otherwise.
    return out, (MANY_IMAGE_PX if max(sized_for, n) > MANY_IMAGES else None)


# A wedged stream is silent, so the clock that matters is httpx's PER-READ
# timeout, not wall time measured when a frame happens to arrive. The former
# batch runner (removed 2026-09-11; see git history) shortened exactly this
# (read=45) and noted the endpoints send keepalives every ~15s. Measuring wall time inside `for ch in stream` instead, as an
# earlier draft did, is wrong twice over: it never fires on a stream that is
# actually wedged (no frame = no check), and it kills a stream that paused and
# then RECOVERED, throwing away everything already received.
READ_IDLE_S = 90      # no bytes for this long = wedged; > any keepalive gap
# httpx has no total deadline. For the streamed providers this bounds the
# other phases (write, pool) while READ_IDLE_S bounds each read; for gemini,
# which does not stream, it is the per-read timeout too -- the only guard
# that path has. A stream that keeps sending frames is never cut off.
CALL_TIMEOUT_S = 1800


_DETERMINISTIC = ("invalid_image", "invalid image", "unsupported image",
                  "image_parse", "moderation", "content_policy",
                  "data policy", "invalid_request", "context_length")


def _looks_deterministic(msg: str) -> bool:
    """An in-stream error frame has no status; classify it by what it says.

    Retrying a data-policy or bad-image failure four times only burns the
    budget, and without this the image rescue can never fire on the three
    openai-compatible gateways.
    """
    low = (msg or "").lower()
    return any(k in low for k in _DETERMINISTIC)


def _status(e: BaseException) -> int | None:
    """The HTTP status `e` carries, or None.

    openai raises a bare APIError for an in-stream {"error": ...} frame (HTTP
    200 + SSE error, which is how OpenRouter and xAI report provider-side
    failures): it has no status_code, and its .code is whatever the provider
    put in the frame -- an int on OpenRouter, a STRING such as "invalid_image"
    on OpenAI-shaped bodies. google-genai raises ClientError with an int .code
    and no status_code. Only an int is a status. A string code is a label and
    is classified by _looks_deterministic with the message; comparing it
    against 400 raised TypeError inside the except handler, which escaped the
    retry loop and killed the case on the first such frame.
    """
    for v in (getattr(e, "status_code", None), getattr(e, "code", None),
              getattr(getattr(e, "response", None), "status_code", None)):
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


def _has_fence(text: str) -> bool:
    """Whether episode would find something executable in this reply."""
    from envs.common.episode import _blocks          # keep the rules in lockstep
    return any(_blocks(text or ""))


def drive(send, what: str):
    """Turn a provider's send(turns, drop_images) into episode's call_fn.

    Three rules carried over from the former batch runner (removed 2026-09-11;
    see git history), all provider-neutral:

    1. A reply with no executable fence is a FAILED call, not a turn. Measured
       on T3 gn866: a model that trails off on a channel marker reported
       status=completed with no fence, and 100 consecutive rounds burned 3M
       tokens on nothing. Retry it -- but never raise on the last attempt: the
       model may simply be writing prose, and killing the case there turned
       PART-0214 from scoring every round into a flat zero. Hand the text back
       and let the episode run its nudge round, which exists for this.
    2. A deterministic 4xx is not worth repeating, with one exception: if it
       names an image, drop the images and try once more. Measured: a 41-byte
       truncated PNG got resent five times, each time with the same bad image,
       and the case was lost. Losing one round of observation images beats
       losing the case.
    3. A stream that goes silent stays silent. Retrying the same context twice
       wedged two cases for two hours each; give up the round instead.
    """
    def call(system: str, turns: list) -> str:
        drop_images = False
        for attempt in range(ATTEMPTS):
            try:
                text = send(system, turns, drop_images)
            except Exception as e:                               # noqa: BLE001
                status = _status(e)
                msg = str(e)
                code = getattr(e, "code", None)
                if isinstance(code, str) and code not in msg:
                    msg = f"{code}: {msg}"       # a label, not a status: classify it
                if status is None and _looks_deterministic(msg):
                    status = 400
                if status is not None and 400 <= status < 500 and status != 429:
                    if ("image" in msg.lower() and not drop_images
                            and attempt < ATTEMPTS - 1
                            and any(t.get("images") for t in turns)):
                        print(f"      {status} names an image; dropping images "
                              f"and retrying once: {msg[:90]}", flush=True)
                        drop_images = True
                        continue
                    print(f"      deterministic {status}, not retrying: {msg[:100]}",
                          flush=True)
                    raise
                # Match the exception TYPE, not its message: neither
                # anthropic.APITimeoutError ("Request timed out or
                # interrupted") nor openai.APITimeoutError ("Request timed
                # out.") nor httpx.ReadTimeout ("timed out") contains the
                # string "ReadTimeout", so the old message test fired for none
                # of them and a wedged stream got the full retry budget.
                timeoutish = (isinstance(e, TimeoutError)
                              or "Timeout" in type(e).__name__
                              or "idle" in msg.lower())
                if timeoutish and attempt >= 1:
                    print(f"      stream stalled twice; giving up this round",
                          flush=True)
                    raise
                if attempt == ATTEMPTS - 1:
                    raise
                print(f"      {what} failed ({type(e).__name__}: {msg[:110]}), "
                      f"retry {attempt + 1}/{ATTEMPTS - 1}", flush=True)
                time.sleep(BACKOFF_S * 2 ** attempt)
                continue
            if _has_fence(text) or attempt == ATTEMPTS - 1:
                if not _has_fence(text):
                    print(f"      no executable block after {ATTEMPTS} tries; "
                          f"handing the text back for the episode's nudge round",
                          flush=True)
                return text
            print(f"      reply has no executable block (len {len(text or '')}, "
                  f"tail {(text or '')[-60:]!r}), retry {attempt + 1}", flush=True)
            time.sleep(BACKOFF_S)
        # Unreachable: every branch above either returns or raises on the last
        # attempt. Raising rather than returning "" matters -- episode counts an
        # empty string as a successful call and resets its dead-round counter,
        # so a persistent fault would be laundered into wasted rounds.
        raise RuntimeError(f"{what}: exhausted {ATTEMPTS} attempts")
    return call


# ── provider call_fns: all return call(system, turns) -> str ─────────────────

def anthropic_call(model: str, max_tokens: int, usage: list,
                   effort: str = DEFAULT_EFFORT):
    import anthropic
    client = anthropic.Anthropic()
    # As on the openai path: set for exactly ONE attempt by the truncation
    # raise below, cleared on every other outcome, so the boost never
    # compounds. max_tokens covers the thinking AND the answer on this API,
    # and at effort high / max the thinking alone can run past 16k tokens:
    # the reply then stops at max_tokens with the answer unwritten, and
    # without a retry with room that round is lost with nothing to show.
    room = {"on": False}

    def send(system: str, turns: list, drop_images: bool) -> str:
        turns, max_px = bound_images(turns)
        budget = max_tokens * EMPTY_RETRY_BOOST if room["on"] else max_tokens
        messages = []
        for t in turns:
            content = [{"type": "text", "text": t["text"]}] if t.get("text") else []
            for img, lab in ([] if drop_images else labelled(t)):
                content.append({"type": "text", "text": image_label(img, lab)})
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": _b64(img, max_px)}})
            messages.append({"role": "assistant" if t["role"] == "assistant" else "user",
                             "content": content or [{"type": "text", "text": "(empty)"}]})
        # "none" is thinking switched off, not an effort value: output_config
        # takes low..max only. (The other four are measured; none is the
        # documented `thinking.type.disabled` and is not yet measured here.)
        knobs = ({"thinking": {"type": "disabled"}} if effort == "none" else
                 {"thinking": {"type": "adaptive"},
                  "output_config": {"effort": effort}})
        # Prompt caching is NOT automatic on this API: without a cache_control
        # field nothing is cached, and every round re-bought the whole
        # transcript and every seed image at full price (the usage
        # accounting below was written as if it were on; it was not). A
        # top-level cache_control asks the API to cache the longest stable
        # prefix itself: reads are 0.1x the input price, writes 1.25x, and
        # the prefix -- system prompt, seed images, every earlier turn --
        # is the bulk of a round's input from round two on.
        # Stream: max_tokens this large trips the SDK's HTTP timeout otherwise.
        if room["on"]:
            print(f"      retrying with max_tokens={budget:,} so the answer has room "
                  f"after the thinking", flush=True)
        with client.messages.stream(model=model, system=system,
                                    max_tokens=budget, messages=messages,
                                    extra_body={"cache_control": {"type": "ephemeral"}},
                                    **knobs) as st:
            msg = st.get_final_message()
        u = msg.usage
        # episode resends the seed images every round and they are read from
        # the cache, so cache reads/writes are most of the input on later
        # rounds; counting only input_tokens undercounts the run. The two
        # are also kept apart, so a results file shows what caching bought.
        cached = getattr(u, "cache_read_input_tokens", 0) or 0
        written = getattr(u, "cache_creation_input_tokens", 0) or 0
        usage.append({"input_tokens": u.input_tokens + cached + written,
                      "cached_tokens": cached, "cache_write_tokens": written,
                      "output_tokens": u.output_tokens})
        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            print(f"      refusal (category={cat}); treating as an empty turn", flush=True)
            return ""
        think = "".join(getattr(b, "thinking", "") or ""
                        for b in msg.content if b.type == "thinking")
        text = "".join(b.text for b in msg.content if b.type == "text")
        if think or msg.stop_reason != "end_turn":
            print(f"      thinking {len(think):,} chars, content {len(text):,}"
                  + ("" if msg.stop_reason == "end_turn" else f", stop={msg.stop_reason}"),
                  flush=True)
        if msg.stop_reason == "max_tokens" and not _has_fence(text):
            # Truncated before (or inside) the answer: a failed call, not a
            # turn. Once, with room; the second time it is handed back as it
            # is, so a model that cannot stop thinking is not mistaken for a
            # broken adapter.
            if not room["on"]:
                room["on"] = True
                raise EmptyContent(f"stop_reason=max_tokens at {budget:,} tokens with no "
                                   f"executable block (thinking {len(think):,} chars)")
            room["on"] = False
            print("      still truncated with the doubled budget; handing the reply "
                  "back (this round is lost)", flush=True)
            return text
        room["on"] = False
        return text
    return drive(send, "anthropic call")


def _delta_reasoning(delta) -> str:
    """The thinking on one stream delta, whatever the gateway calls it.

    Measured on OpenRouter 2026-09-11 (inclusionai/ling-3.0-flash-vl:free and
    qwen/qwen3.7-flash): a thinking chunk carries BOTH

        delta.reasoning         -> "The"                       (a plain string)
        delta.reasoning_details -> [{"type": "reasoning.text",
                                    "text": "The", "format": "unknown",
                                    "index": 0}]

    Neither field is in the SDK's typed ChoiceDelta. They survive only because
    openai's BaseModel sets extra="allow", so they land in `delta.model_extra`
    and are reachable by getattr and by nothing else -- attribute access on the
    typed model is exactly what the adapter was missing. Only `content` was
    read, so a reply that was all thinking accumulated to "".

    The text is NOT part of the reply: it never goes back to the episode. The
    thinking is the model talking to itself, it is not an answer, and a fenced
    block that appears inside it is a draft that was reasoned about and often
    then rejected -- executing that as the submission would score a draft.
    """
    txt = getattr(delta, "reasoning", None)
    if isinstance(txt, str) and txt:
        return txt
    out = []
    for part in (getattr(delta, "reasoning_details", None) or []):
        t = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if isinstance(t, str):
            out.append(t)
    return "".join(out)


def _reasoning_tokens(usage) -> int | None:
    """usage.completion_tokens_details.reasoning_tokens, or None.

    Typed in the SDK but optional in practice, and some gateways send the
    details as a plain dict, so read it defensively: this number exists to be
    printed in a diagnosis and must never be the thing that raises.
    """
    det = getattr(usage, "completion_tokens_details", None)
    if det is None and isinstance(usage, dict):
        det = usage.get("completion_tokens_details")
    v = (det.get("reasoning_tokens") if isinstance(det, dict)
         else getattr(det, "reasoning_tokens", None))
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _is_openrouter(base_url: str | None) -> bool:
    """Whether `reasoning` is a parameter this endpoint understands.

    OpenRouter normalises a `reasoning` block across vendors; OpenAI, xAI and
    OpenCode would reject the unknown field with a 400, which drive() treats as
    deterministic and does not retry -- so send it to exactly one of them.
    """
    return "openrouter.ai" in (base_url or "")


def openai_compat_call(model: str, max_tokens: int, usage: list,
                       api_key: str, base_url: str | None,
                       effort: str = DEFAULT_EFFORT):
    """OpenAI, xAI, OpenRouter and OpenCode all speak /chat/completions.

    Streamed, so a per-read timeout (READ_IDLE_S on the client below) can
    catch a wedged stream: it reports no error and never ends, and a
    whole-request timeout would not fire while frames trickle. The former
    batch runner patched xai_adapter._stream for exactly this; here it is
    provider-neutral because every one of these four streams.

    Reasoning is accumulated separately from content and never returned: see
    _delta_reasoning for why the thinking is not the model's answer. An empty
    reply is raised as EmptyContent with the numbers that explain it, and the
    one retry drive() then makes is given room -- a doubled budget and, on
    OpenRouter, a bound on the thinking.
    """
    import httpx
    import openai
    client = openai.OpenAI(
        api_key=api_key, base_url=base_url,
        timeout=httpx.Timeout(CALL_TIMEOUT_S, read=READ_IDLE_S, connect=30.0))
    # Set for exactly ONE attempt, by the EmptyContent raise below, and cleared
    # on every other outcome: the boost is never compounded, so
    # max_tokens * EMPTY_RETRY_BOOST is the ceiling however many rounds run.
    room = {"on": False}
    # OpenAI's own effort knob (see OPENAI_EFFORT); None once stepped all the
    # way down. Only OpenAI itself gets it: xAI and OpenCode are not measured
    # against it, and OpenRouter has its `reasoning` block below.
    knob = {"effort": OPENAI_EFFORT[effort] if base_url is None else None}

    def _step_down(err: str) -> bool:
        """A 400 that names reasoning_effort: lower the knob one step, or drop
        it, and say so. False when there is nothing left to lower."""
        if "reasoning_effort" not in err or not knob["effort"]:
            return False
        was, knob["effort"] = knob["effort"], ("high" if knob["effort"] == "xhigh" else None)
        print(f"      {model} rejected reasoning_effort={was}; "
              + (f"sending {knob['effort']} instead" if knob["effort"]
                 else "sending no reasoning_effort")
              + " for the rest of this episode", flush=True)
        return True

    def send(system: str, turns: list, drop_images: bool) -> str:
        turns, max_px = bound_images(turns)
        messages: list = [{"role": "system", "content": system}]
        for t in turns:
            parts = [{"type": "text", "text": t["text"]}] if t.get("text") else []
            for img, lab in ([] if drop_images else labelled(t)):
                parts.append({"type": "text", "text": image_label(img, lab)})
                # detail "high" is the request for the full-resolution
                # pass; "auto" leaves it to the endpoint, and a drawing sheet
                # read at the low tier is a blur.
                parts.append({"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + _b64(img, max_px),
                    "detail": "high"}})
            messages.append({"role": "assistant" if t["role"] == "assistant" else "user",
                             "content": parts or [{"type": "text", "text": "(empty)"}]})
        budget = max_tokens * EMPTY_RETRY_BOOST if room["on"] else max_tokens
        extra: dict = {}
        if _is_openrouter(base_url):
            # The gateway's own effort knob, so "medium" means the same thing
            # here as it does on the anthropic path. It takes low | medium |
            # high, and an upstream is free to ignore it (measured: Novita's
            # free models do).
            extra["extra_body"] = {"reasoning": {"effort": OR_EFFORT[effort]}}
        if room["on"]:
            bound = None
            if _is_openrouter(base_url):
                bound = int(budget * REASONING_BOUND_FRAC)
                # Keep the effort, add the hard bound: the empty reply means
                # the thinking ate the whole budget, and effort alone did not
                # stop it.
                # OpenRouter rejects both knobs at once ("Only one of
                # reasoning.effort and reasoning.max_tokens"), and that 400 is
                # deterministic, so drive() does not retry it -- the case died
                # with no submission. Measured: 5 of 9 cases lost this way.
                # The bound is the whole point of this retry, so it replaces
                # the effort rather than joining it.
                extra["extra_body"] = {"reasoning": {"max_tokens": bound}}
            print(f"      retrying with max_completion_tokens={budget:,}"
                  + (f", reasoning.max_tokens={bound:,}" if bound else "")
                  + " so the content has room", flush=True)
        while True:
            if knob["effort"]:
                extra["reasoning_effort"] = knob["effort"]
            else:
                extra.pop("reasoning_effort", None)
            try:
                stream = client.chat.completions.create(
                    model=model, messages=messages, max_completion_tokens=budget,
                    stream=True, stream_options={"include_usage": True}, **extra)
                break
            except openai.BadRequestError as e:
                # Deterministic to drive(), so it must be handled here: the
                # same request minus one step of the knob is a different
                # request. Anything else is drive()'s to classify.
                if not _step_down(str(e)):
                    raise
        chunks, thinking = [], []
        seen_usage, rtokens, finish = None, None, None
        try:
            for ch in stream:
                # Usage can arrive on the final frame only (the spec) or on
                # every frame (Google's compat shim, some OpenRouter
                # providers). Keep the LAST one rather than summing, which
                # inflated a 10-token prompt to 30.
                if getattr(ch, "usage", None):
                    det = getattr(ch.usage, "prompt_tokens_details", None)
                    cached = (det.get("cached_tokens") if isinstance(det, dict)
                              else getattr(det, "cached_tokens", None)) or 0
                    seen_usage = {"input_tokens": ch.usage.prompt_tokens,
                                  "cached_tokens": cached,
                                  "output_tokens": ch.usage.completion_tokens}
                    # Same last-one-wins rule, except that a frame which omits
                    # the breakdown must not erase a count an earlier one gave.
                    rtokens = _reasoning_tokens(ch.usage) or rtokens
                if not ch.choices:
                    continue
                choice = ch.choices[0]
                # The reason the call ended arrives on the frame before the
                # usage frame, and it is the whole diagnosis for an empty
                # reply: "length" means the budget ran out, "stop" means the
                # model chose to say nothing.
                if getattr(choice, "finish_reason", None):
                    finish = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:                 # a usage-only or error frame
                    continue
                think_delta = _delta_reasoning(delta)
                if think_delta:
                    thinking.append(think_delta)
                if delta.content:
                    chunks.append(delta.content)
        finally:
            getattr(stream, "close", lambda: None)()
        if seen_usage is None:
            print("      (provider reported no usage; token counts for this "
                  "call are unknown, not zero)", flush=True)
        else:
            usage.append(seen_usage)
        text, think = "".join(chunks), "".join(thinking)
        if think:
            # So a client reading the log can see where the budget went. The
            # finish_reason is appended when it is not a clean stop, which is
            # the other reply this used to lose silently: content that IS
            # there but was cut off mid-program, so it carries no closing
            # fence and read as "the model wrote prose".
            print(f"      reasoning {len(think):,} chars"
                  + (f" / {rtokens:,} tokens" if rtokens else "")
                  + f", content {len(text):,}"
                  + (f", finish_reason={finish}" if finish and finish != "stop"
                     else ""), flush=True)
        if text:
            room["on"] = False
            return text
        why = (f"content empty, finish_reason={finish}, "
               f"reasoning_tokens={rtokens}")
        if not room["on"]:
            room["on"] = True                     # the next attempt gets room
            raise EmptyContent(why)
        # Room did not help: hand the empty reply back as before, so the
        # episode's nudge round still happens and a weak model is not
        # mistaken for a broken adapter. The log now says which it was.
        room["on"] = False
        print(f"      {why} with the doubled budget too; handing the empty "
              f"reply back (this round is lost)", flush=True)
        return ""
    return drive(send, "chat.completions call")


def gemini_call(model: str, max_tokens: int, usage: list, api_key: str):
    from google import genai
    from google.genai import types
    # google-genai defaults HttpOptions().timeout to None -- one hung unary
    # request would hang the whole run with no output and nothing to attribute
    # it to. There is no stream here, so this ceiling is the only guard.
    client = genai.Client(api_key=api_key,
                          http_options=types.HttpOptions(
                              timeout=CALL_TIMEOUT_S * 1000))

    def send(system: str, turns: list, drop_images: bool) -> str:
        turns, max_px = bound_images(turns)
        contents = []
        for t in turns:
            parts = [types.Part.from_text(text=t["text"])] if t.get("text") else []
            for img, lab in ([] if drop_images else labelled(t)):
                parts.append(types.Part.from_text(text=image_label(img, lab)))
                parts.append(types.Part.from_bytes(
                    data=base64.b64decode(_b64(img, max_px)), mime_type="image/png"))
            if not parts:
                parts = [types.Part.from_text(text="(empty)")]
            # Gemini names the assistant role "model", not "assistant".
            contents.append(types.Content(
                role="model" if t["role"] == "assistant" else "user", parts=parts))
        r = client.models.generate_content(
            model=model, contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=max_tokens))
        um = getattr(r, "usage_metadata", None)
        if um:
            usage.append({"input_tokens": getattr(um, "prompt_token_count", 0),
                          "output_tokens": getattr(um, "candidates_token_count", 0)})
        return r.text or ""
    return drive(send, "gemini call")


def mock_call(kind: str, case: Path):
    """No network. `oracle` submits the reference, `dumb` submits an empty answer.

    Both exist to exercise the entry point, the sandbox staging and the scoring
    chain without a provider key -- a green mock run proves the plumbing, never
    a model. The reply never leaves this process, so the oracle inlines the
    whole reference (the largest example reference is 10.5 MB).
    """
    if kind not in ("oracle", "dumb"):
        raise SystemExit(f"--model mock/{kind}: use mock/oracle or mock/dumb")
    case = Path(case)
    gt_step, gt_graph = case / "gt/gt.step", case / "gt/gt_graph.json"

    if gt_graph.exists():                      # ECAD: the answer is a graph
        if kind == "oracle":
            body = ("import json\n"
                    f"result = json.loads({json.dumps(gt_graph.read_text())})\n")
        else:
            body = ('result = {"schema": "pcb2schematic/1.0", "components": [], '
                    '"nets": [], "incidences": []}\n')
    elif (case / "gt/instances.json").exists() and kind == "oracle":
        # Assembly: the answer is the fixed submission layout the prompt asks
        # for (envs.common.submission), not one STEP. Placement is
        # gt/instances.json verbatim, and each T there is relative to the file
        # `resolve_part` names -- gt/parts/<id>.step when the case holds one,
        # else the staged input/step_files/<id>.step. So a type with a gt/parts
        # copy goes in through tools.export_part from that copy (inlined here
        # because gt/ is never staged), and only a type without one through
        # tools.use_part. Choosing use_part whenever an input file existed put
        # the de-posed input under a T meant for the posed gt/parts copy:
        # measured on the T5 sample, 13 of 21 types landed wrong and the
        # reference scored 0.03. The old single-STEP submission scored the same
        # reference, but it proved the deprecated path, not the one the task
        # description says to use.
        inst = json.loads((case / "gt/instances.json").read_text())["instances"]
        lines = ["import base64, pathlib", "import tools",
                 "_d = pathlib.Path('_oracle'); _d.mkdir(exist_ok=True)"]
        for pid in sorted({r["part_id"] for r in inst}):
            posed = case / "gt/parts" / f"{pid}.step"
            if not posed.exists() and (case / "input/step_files" / f"{pid}.step").exists():
                lines.append(f"tools.use_part({pid!r})")
            else:
                blob = base64.b64encode(posed.read_bytes()).decode()
                lines += [f"(_d / '{pid}.step').write_bytes(base64.b64decode('{blob}'))",
                          f"tools.export_part(str(_d / '{pid}.step'), {pid!r})"]
        recs = [{"part_id": r["part_id"], "instance_id": r["instance_id"], "transform": r["T"]}
                for r in inst]
        lines.append(f"tools.submit_assembly({json.dumps(recs)})")
        body = "\n".join(lines) + "\n"
    elif gt_step.exists():                     # geometry: the answer is a solid
        if kind == "oracle":
            # gt/ is never staged into the sandbox (it is a container; a host
            # path would not resolve either), so the oracle CARRIES the answer
            # in its reply -- which is what a perfect model would do anyway.
            blob = base64.b64encode(gt_step.read_bytes()).decode()
            # Write it into a SUBDIRECTORY. Sandbox._log copies
            # self.dir.glob("*.step") into the round log and episode falls back
            # to the last such artifact, so a top-level _oracle.step would be
            # picked up as the submission and scored against itself -- the
            # oracle would report 1.0 with the export chain completely broken,
            # which is the one thing this mock exists to prove.
            body = ("import base64, pathlib\n"
                    "import cadquery as cq\n"
                    "_d = pathlib.Path('_oracle'); _d.mkdir(exist_ok=True)\n"
                    f"(_d / 'ref.step').write_bytes(base64.b64decode('{blob}'))\n"
                    "result = cq.importers.importStep(str(_d / 'ref.step'))\n")
        else:
            body = ("import cadquery as cq\n"
                    "result = cq.Workplane('XY').box(10, 10, 10)\n")
    else:
        held = ", ".join(sorted(p.name for p in (case / "gt").glob("*"))) or "nothing"
        raise SkipCase(f"mock/{kind} handles gt.step or gt_graph.json; this gt holds {held}")

    reply = "Submitting.\n\n```submit\n" + body + "```\n"

    def call(system: str, turns: list) -> str:
        return reply
    return call


def build_call(spec: str, max_tokens: int, usage: list, case: Path,
               effort: str = DEFAULT_EFFORT):
    prefix, prov, model_id = split_model(spec)
    if prov.kind == "mock":
        return mock_call(model_id, case)
    key = resolve_key(prefix, prov)
    if prov.kind == "anthropic":
        return anthropic_call(model_id, max_tokens, usage, effort)
    if prov.kind == "openai_compat":
        return openai_compat_call(model_id, max_tokens, usage, key, prov.base_url, effort)
    if prov.kind == "gemini":
        return gemini_call(model_id, max_tokens, usage, key)
    raise SystemExit(f"provider kind {prov.kind!r} has no call implementation")


# ── cases ───────────────────────────────────────────────────────────────────

def _walk(root: Path):
    """Every file under `root`, following symlinked directories.

    `Path.rglob` stops at a symlinked directory on Python 3.13+
    (`recurse_symlinks=False` is the default there, and the keyword does not
    exist before it). The documented data layout makes `envs/<env>/cases` a
    symlink into the data tree, so a plain rglob silently found 7 of 11 cases
    -- a run that reports a clean sheet while skipping a third of the bank.
    """
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:                          # a symlink loop, or two links to one tree
            dirnames[:] = []
            continue
        seen.add(real)
        for f in filenames:
            yield Path(dirpath) / f


def discover(spec: str) -> list[Path]:
    """--cases tests/fixtures -> every tests/fixtures/t*/<case> that has a case.json."""
    root = Path(spec)
    if (root / "case.json").exists() or (root / "gt/gt.step").exists():
        return [root]
    files = list(_walk(root))
    out = [f.parent for f in files if f.name == "case.json"]
    if not out:                                   # legacy trees have no case.json
        out = [f.parent.parent for f in files
               if f.name == "gt.step" and f.parent.name == "gt"]
    return sorted(set(out))


def case_key(case: Path) -> str:
    """A work-dir name unique across the whole run.

    `case.name` is not: the default --cases tests/fixtures selects 14 cases whose
    basenames are `case` six times and `example1` six times, and Sandbox copies
    with dirs_exist_ok and never clears, so case N would inherit case N-1's
    inputs, final.step and pred_graph.json -- a correct ECAD submission scored
    0.0 because the last-usable-STEP fallback handed it the previous case's
    box. The former batch runner carried the same fix with the same reason
    (170 prodata cases sharing one work directory).

    The last three path components are the readable part; they are not unique
    on their own (envs/<env>/cases/<family>/<nn> drops <env>, and T1 and T3
    share a parts corpus, so `--cases envs` would fold their families
    together). A short hash of the full resolved path makes the key unique.
    """
    import hashlib
    p = Path(case).resolve()
    tag = hashlib.sha1(str(p).encode()).hexdigest()[:8]
    return "__".join(p.parts[-3:]) + "__" + tag


def show(score: dict) -> str:
    """fmt() assumes an `iou` key; the ECAD verifier reports different fields."""
    try:
        return fmt(score)
    except Exception:                                            # noqa: BLE001
        head = ("score", "iou", "metric_version", "components_matched",
                "components_gt", "incidences_pred", "incidences_gt")
        keep = [(k, score[k]) for k in head if k in score]
        return " ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}"
                        for k, v in keep) or "scored"


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _raise_fd_limit() -> None:
    """macOS starts a process at 256 descriptors. Twenty concurrent episodes
    (subprocess pipes, image files, HTTP streams) go straight through that,
    and the failure is OSError 24 inside the episode, scored as the model's
    zero. The former batch runner measured 31 of 49 records lost that way."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = min(hard, 65536) if hard != resource.RLIM_INFINITY else 65536
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ImportError, ValueError, OSError):
        pass


def _done(rec: dict) -> bool:
    """Whether a record from an earlier run of the same --out is final.

    A score, a skip, or an episode that ran to its end without submitting
    is a result. An error is the harness's or the network's, not the
    model's, and is re-run.
    """
    return not rec.get("error") and ("score" in rec or "skipped" in rec)


def _shard(cases: list, spec: str | None) -> list:
    """--shard k/n keeps cases k, k+n, k+2n, ... of the sorted list."""
    if not spec:
        return cases
    try:
        k, n = (int(x) for x in spec.split("/"))
        assert n > 0 and 0 <= k < n
    except (ValueError, AssertionError):
        raise SystemExit(f"--shard {spec!r}: expected k/n with 0 <= k < n")
    return cases[k::n]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one model over a set of cases.",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="<provider>/<id>, e.g. anthropic/claude-opus-5, "
                         "gemini/gemini-3-pro, mock/oracle")
    ap.add_argument("--cases", default="tests/fixtures",
                    help="a case dir, or a tree to search (default: tests/fixtures)")
    ap.add_argument("--effort", choices=EFFORTS, default=DEFAULT_EFFORT,
                    help="thinking effort (default: max). anthropic: "
                         "output_config.effort (none = thinking disabled); "
                         "openai: reasoning_effort, where max maps to xhigh; "
                         "openrouter: reasoning.effort, where max maps to high")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="default results/<model>_<timestamp>.json")
    ap.add_argument("--workers", type=int, default=1,
                    help="episodes run at once (threads: an episode waits on "
                         "HTTP or on its sandbox almost all of the time). "
                         "Each sandbox exec may take 2 GB; size the docker "
                         "host for workers x 2 GB")
    ap.add_argument("--max-execs", type=int, default=0,
                    help="at most this many sandbox executions at once in "
                         "this process, however many workers wait on the "
                         "API (CADENV_MAX_EXECS; 0 = no cap). Size it to the "
                         "docker host: each execution may take 2 GB")
    ap.add_argument("--rep", type=int, default=0,
                    help="repetition index: recorded in every record and "
                         "part of the work-dir name, so reps of one case can "
                         "run side by side (default 0)")
    ap.add_argument("--shard", default=None,
                    help="k/n: this process takes cases k, k+n, k+2n, ... of "
                         "the sorted list; give each machine its own k")
    ap.add_argument("--resume", action="store_true",
                    help="reuse --out: cases it already holds with a score, "
                         "a skip or a finished episode are kept, errors are "
                         "re-run, the rest are run")
    a = ap.parse_args()

    prefix, prov, model_id = split_model(a.model)
    cases = _shard(discover(a.cases), a.shard)
    if not cases:
        raise SystemExit(f"--cases {a.cases}: no cases found")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = a.out or (Path("results") /
                         f"{a.model.replace('/', '_')}_{stamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = a.work or (Path.home() / "cad-agent-work" / f"run_{stamp}")
    if a.workers > 1:
        _raise_fd_limit()
    if a.max_execs:
        # Read by envs.common.sandbox at import; the harness imports it
        # through episode above, so re-create the gate here.
        os.environ["CADENV_MAX_EXECS"] = str(a.max_execs)
        import envs.common.sandbox as _sb
        _sb.EXEC_GATE = _sb._exec_gate()

    # Records keyed by case path, in case order, so a resumed run and a
    # parallel run both write the same file a sequential run would.
    kept: dict[str, dict] = {}
    if a.resume and out_path.exists():
        prior = json.loads(out_path.read_text())
        kept = {r["case"]: r for r in prior.get("cases", []) if _done(r)}
        stamp = prior.get("started", stamp)
    todo = [c for c in cases if str(c) not in kept]
    print(f"model {a.model}  provider {prefix.rstrip('/')}  "
          f"cases {len(cases)}  rounds {a.rounds}  effort {a.effort}  "
          f"rep {a.rep}  workers {a.workers}"
          + (f"  max-execs {a.max_execs}" if a.max_execs else "")
          + (f"  shard {a.shard}" if a.shard else "")
          + (f"  resume: {len(kept)} kept, {len(todo)} to run" if a.resume else ""),
          flush=True)
    records: dict[str, dict] = dict(kept)
    import threading
    lock = threading.Lock()

    def write() -> None:
        out_path.write_text(json.dumps(
            {"model": a.model, "provider": prefix.rstrip("/"), "rounds": a.rounds,
             "effort": a.effort, "rep": a.rep, "started": stamp,
             "cases": [records[str(c)] for c in cases if str(c) in records]},
            indent=1, default=str) + "\n")

    def one(case: Path) -> dict:
        usage: list = []
        t0 = time.time()
        rec = {"case": str(case), "case_id": case.name, "model": a.model,
               "provider": prefix.rstrip("/"), "rounds": a.rounds,
               "effort": a.effort, "rep": a.rep}
        gt = case / "gt/gt.step"
        rec["gt_sha256"] = sha256(gt) if gt.exists() else None
        try:
            call = build_call(a.model, a.max_tokens, usage, case, a.effort)
            work = work_root / f"r{a.rep}__{case_key(case)}"
            # A directory from an earlier attempt (--resume re-running an
            # error) would be staged over, not replaced -- Sandbox copies with
            # dirs_exist_ok -- and its old submission could be scored.
            import shutil
            for stale in (work, work.parent / (work.name + "_log")):
                shutil.rmtree(stale, ignore_errors=True)
            res = run_episode(case, work, call, max_rounds=a.rounds)
            rec["submitted"] = res["submitted"]
            # run_episode only looks for .step artifacts, so an ECAD submission
            # (a graph dict exported as pred_graph.json) comes back with
            # step=None even when it succeeded. Finding the artifact is the
            # runner's job, so look for it here rather than widen episode's
            # contract.
            artifact = res["step"]
            if not artifact and (work / "pred_graph.json").exists():
                artifact = str(work / "pred_graph.json")
            rec["step"] = artifact
            # Which layout the answer arrived in: "submission_directory" on an
            # assembly task that used the fixed layout, "step" otherwise
            # (envs/common/episode.py _artifact). In the record because the two
            # are scored by different code paths.
            rec["artifact"] = res.get("artifact")
        except SkipCase as e:
            rec["skipped"] = str(e)
        except SystemExit:
            raise
        except Exception as e:                                   # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-2000:]
        rec["seconds"] = round(time.time() - t0, 1)
        rec["tokens"] = {
            "input": sum(u.get("input_tokens", 0) for u in usage),
            "cached": sum(u.get("cached_tokens", 0) for u in usage),
            "output": sum(u.get("output_tokens", 0) for u in usage),
            "calls": len(usage)}
        return rec

    done = 0

    def finish(case: Path, rec: dict) -> None:
        """Score and record. Runs on the MAIN thread only: the pixel term
        renders through vtk, and on macOS vtk's Cocoa window may only be
        created on the main thread -- from a worker thread it is an
        NSInternalInconsistencyException that aborts the whole process
        (measured: four episodes lost at once, results file empty)."""
        nonlocal done
        if "error" not in rec and "skipped" not in rec:
            t0 = time.time()
            try:
                artifact = rec.get("step")
                rec["score"] = score_case(case, Path(artifact)) if artifact else None
            except Exception as e:                               # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["traceback"] = traceback.format_exc()[-2000:]
            rec["seconds_score"] = round(time.time() - t0, 1)
        with lock:
            records[str(case)] = rec
            done += 1
            n = done
            write()
        score = rec.get("score")
        line = (show(score) if score
                else rec.get("skipped") and f"skipped: {rec['skipped']}"
                or rec.get("error") or "no submission")
        print(f"  [{n}/{len(todo)}] {case.name}: {line}  "
              f"({rec['seconds']}s)", flush=True)

    if a.workers <= 1:
        write()
        for case in todo:
            finish(case, one(case))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        write()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(one, c): c for c in todo}
            for f in as_completed(futs):
                finish(futs[f], f.result())
    print(f"\n{len(records)} cases -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
