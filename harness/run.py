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

# Thinking effort follows the model: a level is sent as it is, and only to a
# provider that has it -- nothing is remapped (PROVIDER_EFFORTS, checked by
# check_effort before any call is made).
#
#   anthropic   output_config.effort low | medium | high | xhigh | max, the
#               five Anthropic documents for Opus 5, Sonnet 5 and Fable 5.1;
#               none = thinking.type=disabled (on Opus 5 not at xhigh / max).
#               A model without effort (Haiku 4.5) answers 400 and the case
#               fails; that is intended, drive() treats it as deterministic.
#   openai      reasoning_effort none | low | medium | high | xhigh | max.
#               gpt-6-astra takes low..max and answers 400 to none (its model
#               page, read 2026-09-20: "reasoning.effort supports low, medium,
#               high, xhigh, and max"); gpt-5.4 and gpt-5.5 take none..xhigh
#               (measured 2026-09-15: `minimal` and `max` are 400s). Older
#               families take fewer values (gpt-5: minimal..high, gpt-5.1:
#               none..high, the o-series: low..high) and gpt-4.1 has no such
#               parameter at all ("Unrecognized request argument supplied:
#               reasoning_effort"), so a 400 that names the parameter steps
#               the value down -- max -> xhigh -> high -> not sent -- and the
#               call is repeated; the step is remembered for the rest of the
#               episode.
#   openrouter  reasoning.effort low | medium | high, and none.
#
# Unset, the level is the provider's top (top_effort): Anthropic max, OpenAI
# max, OpenRouter high. Leaving it unset at the API is not the top -- both
# APIs default to high (measured 2026-09-11 on claude-opus-5: 32 thinking
# tokens unset, 111 at max) -- and the level actually run is what the
# banner and every record carry.
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = None
PROVIDER_EFFORTS = {"anthropic/": EFFORTS,
                    "openai/": EFFORTS,
                    "openrouter/": ("none", "low", "medium", "high")}
OPENAI_EFFORT = {lvl: lvl for lvl in PROVIDER_EFFORTS["openai/"]}
# What an OpenAI 400 naming the effort lowers the knob to: one step, then off.
EFFORT_STEP_DOWN = {"max": "xhigh", "xhigh": "high"}
OR_EFFORT = {lvl: lvl for lvl in PROVIDER_EFFORTS["openrouter/"]}
# The rounds an episode gets when --rounds is not given; the eval.toml the
# examples ship quotes it (tools/make_dev_samples.py). 30, since 2026-09-17:
# on the trial runs the model submitted on its own well inside that, and
# the rounds past it bought re-crops, not answers.
DEFAULT_ROUNDS = 30
# No cap on a reply by default: the model gets its own maximum output
# (OpenAI / Gemini: the parameter is left out; Anthropic requires one, so
# the model's ceiling is sent -- what the Models API reports as max_tokens
# (_anthropic_model_info), else this table: 128k for the Claude 5 family
# and everything not listed, 64k for Haiku 4.5, Anthropic's own numbers; a
# model with a lower ceiling answers 400 and anthropic_call halves until
# accepted). A 16k cap was the thinking's cap at effort max, and every
# round that filled it lost its answer. 512k is the same reasoning. --max-tokens still sets one when a run wants it.
DEFAULT_MAX_TOKENS = None
ANTHROPIC_MAX_OUTPUT = {"claude-haiku-4-5": 64_000}
ANTHROPIC_MAX_OUTPUT_DEFAULT = 128_000
# OpenAI's own endpoint gets the same ceiling as a reply: 128k, which is
# gpt-6-astra's own maximum output and the Claude 5 family's -- so a reply
# on either provider is bounded the same way. Decided 2026-09-17 after the
# medium held-out run: a reasoning model can run for an hour on one prompt
# (an earlier harness measured 14 % of effort-max calls doing so and bounded
# them the same way), and an unbounded reply is an unbounded bill and an
# unbounded wait. Not a thinking cap in the 16k sense: no measured reply
# came near it (the longest, 48k output over a whole episode).
OPENAI_MAX_OUTPUT = 128_000
# Transient failures (connection errors, 5xx) are re-requested ATTEMPTS
# times with BACKOFF_S doubling -- 5/10/20/40/80 s, ~2.5 min of outage --
# before the round is given up. Measured 2026-09-18 over ~2,500 calls on
# two machines: 224 "Connection error." retries, and the 4-attempt budget
# (35 s) still burned 5 rounds; an outage of a minute or two is not the
# model's doing and a 30-round episode should not pay for it.
ATTEMPTS = 6
BACKOFF_S = 5
# A 429 is the endpoint saying "not now", not a failed call: it does not
# spend the ATTEMPTS budget above and never counts as a dead round. The
# wait is what the endpoint asks for (Retry-After, or OpenAI's
# x-ratelimit-reset-tokens / -requests), else RATE_LIMIT_BACKOFF_S doubling,
# capped; after RATE_LIMIT_ATTEMPTS waits in one call it is raised as a
# failure. Measured 2026-09-17: 12 episodes in flight on one org key
# (500k TPM) at ~100k prompt tokens a round drew a 429 every few calls,
# and 5-20 s backoffs inside a 60 s window were mostly wasted.
RATE_LIMIT_ATTEMPTS = 10
RATE_LIMIT_BACKOFF_S = 20
RATE_LIMIT_MAX_WAIT_S = 300
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


def top_effort(prefix: str) -> str | None:
    """The highest level the provider at `prefix` has -- what --effort means
    when it is not given -- or None for a provider with no effort knob (xai,
    opencode, gemini, mock: nothing is sent)."""
    levels = PROVIDER_EFFORTS.get(prefix)
    return levels[-1] if levels else None


def check_effort(prefix: str, effort: str | None) -> None:
    """Refuse a level the provider at `prefix` does not have, before any
    call is made: no remapping, a run at `xhigh` on OpenRouter is an error."""
    levels = PROVIDER_EFFORTS.get(prefix)
    name = prefix.rstrip("/")
    if levels is None:
        if effort is not None:
            raise SystemExit(f"{name} has no effort knob; leave --effort unset")
        return
    if effort not in levels:
        raise SystemExit(f"{name} has no effort {effort!r}; its levels are "
                         + ", ".join(levels))


# The API's many-image rule, measured 2026-09-12 on claude-opus-5: a request
# with MORE than MANY_IMAGES image blocks (every turn's images count, seeds
# included) rejects any image with a dimension over MANY_IMAGE_PX with
# "exceed max allowed size for many-image requests"; 20 images of 2100 px
# pass, 21 do not, 21 of 2000 px pass. Every image the episode has produced
# stays in the request (nothing is dropped: the model's own figures are its
# working memory, and a figure that left the request was re-made -- measured
# on gpt-6-astra, 49 of 49 rounds re-cropping the same box), so once the
# count passes MANY_IMAGES every image goes at MANY_IMAGE_PX from then on:
# one cache rewrite, then a stable prefix.
#   MANY_IMAGE_PX     the per-image bound over MANY_IMAGES images
#   HARD_IMAGE_CAP    the API's own ceiling on image blocks per request
#                     (600 on a 1M-context Anthropic model; OpenAI allows
#                     1500). Not a policy: a request past it is a 400, so
#                     the oldest observation images beyond it are left out
#                     and the log says so.
MANY_IMAGES = 20
MANY_IMAGE_PX = 2000
HARD_IMAGE_CAP = 600
# Both first-party APIs refuse a request over 32 MB, and a long episode's
# observation images are what fills it (Anthropic's review of the public
# harness, 2026-09-21: errors on the longest trajectories). The images of one
# request are bounded to this many bytes as they will be sent (PNG, base64
# adds a third), oldest observation images dropped first, seeds never; the
# log says how many. With the compaction point at COMPACT_TOKENS this rarely
# fires -- it is the wall behind the wall.
REQUEST_IMAGE_BYTES = 20_000_000


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


def _sent_bytes(path: Path, max_px: int | None) -> int:
    """What one image costs a request: its PNG bytes, scaled by the area
    ratio when it will be downscaled to max_px, times base64's 4/3."""
    try:
        size = path.stat().st_size
        if max_px:
            from PIL import Image
            with Image.open(path) as im:
                m = max(im.size)
            if m > max_px:
                size = int(size * (max_px / m) ** 2)
    except OSError:
        size = 0
    return size * 4 // 3


def bound_images(turns: list) -> tuple[list, int | None]:
    """The turns as a request should carry them: every image stays. Returns
    the turns and the per-image pixel limit to encode with (None under the
    many-image threshold). Past HARD_IMAGE_CAP images, or past
    REQUEST_IMAGE_BYTES of image payload, the oldest observation images are
    left out, oldest first, seeds never."""
    n = sum(len(t.get("images") or []) for t in turns)
    max_px = MANY_IMAGE_PX if n > MANY_IMAGES else None
    user_idx = [i for i, t in enumerate(turns) if t.get("role") != "assistant"]
    out = [dict(t) for t in turns]
    dropped = 0
    over = n - HARD_IMAGE_CAP
    for i in user_idx[1:]:                                   # never the seed turn
        if over <= 0:
            break
        imgs = list(out[i].get("images") or []); labs = list(out[i].get("image_labels") or [])
        k = min(over, len(imgs))
        out[i]["images"], out[i]["image_labels"] = imgs[k:], labs[k:] if labs else labs
        over -= k; dropped += k
    if dropped:
        max_px = MANY_IMAGE_PX
        print(f"      request had {n} images, over the API's {HARD_IMAGE_CAP}; the oldest "
              f"{dropped} observation images are left out of this request", flush=True)
    total = sum(_sent_bytes(Path(img), max_px) for t in out for img in (t.get("images") or []))
    seed = sum(_sent_bytes(Path(img), max_px) for img in (out[user_idx[0]].get("images") or [])) if user_idx else 0
    if total > REQUEST_IMAGE_BYTES:
        dropped_b = 0
        for i in user_idx[1:]:
            if total <= REQUEST_IMAGE_BYTES:
                break
            imgs = list(out[i].get("images") or []); labs = list(out[i].get("image_labels") or [])
            while imgs and total > REQUEST_IMAGE_BYTES:
                total -= _sent_bytes(Path(imgs.pop(0)), max_px)
                if labs:
                    labs.pop(0)
                dropped_b += 1
            out[i]["images"], out[i]["image_labels"] = imgs, labs
        print(f"      request's images would exceed {REQUEST_IMAGE_BYTES // 1_000_000} MB "
              f"(seeds alone {seed // 1_000_000} MB); the oldest {dropped_b} observation "
              f"images are left out of this request", flush=True)
    return out, max_px


# A wedged stream is silent, so the clock that matters is httpx's PER-READ
# timeout, not wall time measured when a frame happens to arrive. The former
# batch runner (removed 2026-09-11; see git history) shortened exactly this
# (read=45) and noted the endpoints send keepalives every ~15s. Measuring wall time inside `for ch in stream` instead, as an
# earlier draft did, is wrong twice over: it never fires on a stream that is
# actually wedged (no frame = no check), and it kills a stream that paused and
# then RECOVERED, throwing away everything already received.
READ_IDLE_S = 90      # no bytes for this long = wedged; > any keepalive gap
# OpenAI's own endpoint and the Anthropic API send NOTHING while a reasoning
# model thinks (no keepalive, no reasoning deltas on chat.completions), and
# a round-one request with a dozen sheets at 100k prompt tokens can think
# for minutes under load: measured 2026-09-17 on gpt-6-astra with 12
# episodes in flight, T2/T5 round one timed out at 90 s three times running
# and the cases were abandoned as dead. So on the first-party endpoints the
# per-read clock is the whole call budget; the 90 s wedge detector stays for
# the gateways (OpenRouter, xAI, OpenCode), which do send keepalives.
# ... but not the whole 1800 s: streams DO wedge -- 2026-09-17 23:00, four
# of four in flight went silent at once for 30+ minutes while a fresh
# request with the same images answered in 3 s -- so a silent stream is
# given up after FIRST_PARTY_IDLE_S and the request is simply made again,
# STALL_RETRIES times, before the round is discarded. 300 s covers the
# longest silent reasoning measured at high (< 60 s) with room for xhigh.
# 2026-09-18: with OpenAI spoken through the Responses API, reasoning
# summaries stream while the model thinks, and Anthropic streams its
# thinking deltas, so on both first-party endpoints silence means a dead
# stream, not a thinking one -- 60 s is plenty (the longest silence seen on a
# live stream today was under 60 s, on chat.completions). And every call
# gets a wall-clock budget by effort, CALL_BUDGET_S: nothing measured today
# answered after 90 s except a dead stream (median ~40 s, p90 60-90 s across
# ~400 calls at high and medium), so 5 minutes at high is a wide margin and
# 10 at xhigh / max leaves room for reasoning ten times longer. A call over
# budget is re-requested like a silent one (STALL_RETRIES), so the cost of a
# rare misfire is the tokens of one reply, not a case.
FIRST_PARTY_IDLE_S = 60
STALL_RETRIES = 3
CALL_BUDGET_S = {"none": 300, "low": 300, "medium": 300, "high": 300, "xhigh": 600, "max": 600}
CALL_BUDGET_DEFAULT_S = 300


class CallOverBudget(TimeoutError):
    """A stream still open past CALL_BUDGET_S. Not re-requested (the model
    would do the same reasoning again); the episode discards the round and
    says so in the next observation."""


def _over_budget(started: float, budget: float, what: str) -> None:
    if time.time() - started > budget:
        raise CallOverBudget(f"{what} still streaming after {budget:.0f} s; giving it up")
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


def _retry_after(e: BaseException) -> float | None:
    """The wait an endpoint asked for on a 429, in seconds, or None:
    `Retry-After` (seconds), else OpenAI's `x-ratelimit-reset-tokens` /
    `x-ratelimit-reset-requests` ("1.2s", "250ms", "1m3s"), else the
    "Please try again in 1.2s" / "in 250ms" an OpenAI error body says (an
    in-stream frame has no headers)."""
    import re
    headers = getattr(getattr(e, "response", None), "headers", None)
    if not headers:
        m = re.search(r"try again in ([0-9.]+)\s*(ms|s)\b", str(e))
        return float(m.group(1)) * (0.001 if m.group(2) == "ms" else 1) if m else None
    v = headers.get("retry-after")
    if v:
        try:
            return float(v)
        except ValueError:
            pass
    best = None
    for k in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        v = headers.get(k)
        if not v:
            continue
        secs = 0.0
        for num, unit in re.findall(r"([0-9.]+)(ms|s|m|h)", v):
            secs += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
        if secs > 0:
            best = max(best or 0.0, secs)
    return best


def _has_fence(text: str) -> bool:
    """Whether episode would find something executable in this reply."""
    from envs.common.episode import _blocks          # keep the rules in lockstep
    return any(_blocks(text or ""))


def drive(send, what: str):
    """Turn a provider's send(turns, drop_images) into episode's call_fn.

    Three rules carried over from the former batch runner (removed 2026-09-11;
    see git history), all provider-neutral:

    1. A reply with no executable fence is a FAILED call, not a turn. Measured
       on T3 one case: a model that trails off on a channel marker reported
       status=completed with no fence, and 100 consecutive rounds burned 3M
       tokens on nothing. Retry it -- but never raise on the last attempt: the
       model may simply be writing prose, and killing the case there turned
       part case 0214 from scoring every round into a flat zero. Hand the text back
       and let the episode run its nudge round, which exists for this.
    2. A deterministic 4xx is not worth repeating, with one exception: if it
       names an image, drop the images and try once more. Measured: a 41-byte
       truncated PNG got resent five times, each time with the same bad image,
       and the case was lost. Losing one round of observation images beats
       losing the case.
    3. A stream that goes silent stays silent: the per-read timeout ends it
       and the request is made again on a fresh connection, STALL_RETRIES
       times; only then is the round discarded.
    """
    def call(system: str, turns: list, plain: bool = False) -> str:
        """`plain`: the reply is prose (a summary, questions, answers --
        episode's context summarization), so rule 1 does not apply."""
        drop_images = False
        waits = 0                    # 429s waited out in this call; not attempts
        stalls = 0                   # silent streams re-requested in this call; not attempts
        attempt = 0
        while attempt < ATTEMPTS:
            try:
                text = send(system, turns, drop_images)
            except Exception as e:                               # noqa: BLE001
                status = _status(e)
                msg = str(e)
                code = getattr(e, "code", None)
                # The Responses stream reports an org rate limit as an SSE
                # error frame on an HTTP 200 -- a bare APIError with code
                # "rate_limit_exceeded", no status, no Retry-After -- not as
                # a 429. Measured 2026-09-18: 86 such frames in one log went
                # through the 5-20 s transient backoffs and burned rounds
                # ("round 25 call failed (APIError), round discarded").
                limited = status == 429 or code == "rate_limit_exceeded"
                if limited and waits < RATE_LIMIT_ATTEMPTS:
                    wait = _retry_after(e) or min(RATE_LIMIT_MAX_WAIT_S,
                                                  RATE_LIMIT_BACKOFF_S * 2 ** waits)
                    wait = min(RATE_LIMIT_MAX_WAIT_S, max(1.0, wait + 1.0))
                    waits += 1
                    print(f"      rate limited ({msg[:160]}); waiting {wait:.0f} s "
                          f"[{waits}/{RATE_LIMIT_ATTEMPTS}]", flush=True)
                    time.sleep(wait)
                    continue                                   # the attempt is not spent
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
                        attempt += 1
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
                if isinstance(e, CallOverBudget):
                    # Not a dead stream: the model was still answering. Asking
                    # again re-runs the same reasoning (measured 2026-09-18:
                    # T5 round one streamed for 43 minutes and ended in a
                    # server error; the earlier re-requests of it each did
                    # the same) -- so the round is given up at once and the
                    # episode tells the model. See episode.run_episode.
                    print(f"      {msg}; round given up", flush=True)
                    raise
                timeoutish = (isinstance(e, TimeoutError)
                              or "Timeout" in type(e).__name__
                              or "idle" in msg.lower())
                if timeoutish:
                    # A wedged stream is not the model's doing: make the
                    # request again, STALL_RETRIES times, each a fresh
                    # connection, before the round is discarded.
                    stalls += 1
                    if stalls > STALL_RETRIES:
                        print(f"      stream silent {STALL_RETRIES + 1} times; giving up this round",
                              flush=True)
                        raise
                    print(f"      stream silent; making the request again [{stalls}/{STALL_RETRIES}]",
                          flush=True)
                    continue                                   # not an attempt either
                if attempt == ATTEMPTS - 1:
                    raise
                print(f"      {what} failed ({type(e).__name__}: {msg[:110 if status != 429 else 400]}), "
                      f"retry {attempt + 1}/{ATTEMPTS - 1}", flush=True)
                time.sleep(BACKOFF_S * 2 ** attempt)
                attempt += 1
                continue
            if plain or _has_fence(text) or attempt == ATTEMPTS - 1:
                if not plain and not _has_fence(text):
                    print(f"      no executable block after {ATTEMPTS} tries; "
                          f"handing the text back for the episode's nudge round",
                          flush=True)
                return text
            print(f"      reply has no executable block (len {len(text or '')}, "
                  f"tail {(text or '')[-60:]!r}), retry {attempt + 1}", flush=True)
            time.sleep(BACKOFF_S)
            attempt += 1
        # Unreachable: every branch above either returns or raises on the last
        # attempt. Raising rather than returning "" matters -- episode counts an
        # empty string as a successful call and resets its dead-round counter,
        # so a persistent fault would be laundered into wasted rounds.
        raise RuntimeError(f"{what}: exhausted {ATTEMPTS} attempts")
    return call


# ── provider call_fns: all return call(system, turns) -> str ─────────────────

_MODEL_INFO: dict[str, dict | None] = {}


def _anthropic_model_info(client, model: str) -> dict | None:
    """What the Models API says `model` takes: its output ceiling
    (max_tokens), its context window (max_input_tokens) and the effort
    levels it has. Asked once per process. None when it cannot be asked
    (offline, a client without .models), said once; the built-in tables
    then stand in."""
    if model in _MODEL_INFO:
        return _MODEL_INFO[model]
    try:
        info = client.models.retrieve(model)
        eff = getattr(getattr(info, "capabilities", None), "effort", None)
        efforts: set[str] = set()
        for lvl in EFFORTS[1:]:
            cap = getattr(eff, lvl, None)
            if cap is not None and cap.supported:
                efforts.add(lvl)
        out = {"max_tokens": getattr(info, "max_tokens", None),
               "max_input_tokens": getattr(info, "max_input_tokens", None),
               "efforts": efforts}
    except Exception as e:                                       # noqa: BLE001
        print(f"      (models API unavailable for {model}, {type(e).__name__}; "
              f"using the built-in table)", flush=True)
        out = None
    _MODEL_INFO[model] = out
    return out


def anthropic_call(model: str, max_tokens: int, usage: list,
                   effort: str | None = DEFAULT_EFFORT):
    import anthropic
    effort = effort or top_effort("anthropic/")
    check_effort("anthropic/", effort)
    call_budget = CALL_BUDGET_S.get(effort, CALL_BUDGET_DEFAULT_S)
    # The same clock as the openai client: a wedged stream is caught by the
    # per-read timeout, the other phases by CALL_TIMEOUT_S. (anthropic 1.x is
    # built on httpx2; anthropic.Timeout is its Timeout.)
    client = anthropic.Anthropic(timeout=anthropic.Timeout(CALL_TIMEOUT_S, read=FIRST_PARTY_IDLE_S,
                                                           connect=30.0))
    info = _anthropic_model_info(client, model)
    # As on the openai path: set for exactly ONE attempt by the truncation
    # raise below, cleared on every other outcome, so the boost never
    # compounds. max_tokens covers the thinking AND the answer on this API,
    # and at effort high / max the thinking alone can run past 16k tokens:
    # the reply then stops at max_tokens with the answer unwritten, and
    # without a retry with room that round is lost with nothing to show.
    room = {"on": False}

    ceiling = {"n": max_tokens or (info or {}).get("max_tokens")
               or next((n for k, n in ANTHROPIC_MAX_OUTPUT.items()
                        if model.startswith(k)), ANTHROPIC_MAX_OUTPUT_DEFAULT)}

    # The level actually sent: the one asked for, or -- when the Models API
    # lists this model's levels and it is not among them -- the highest of
    # those below it (the lowest listed when none is), said once.
    level = effort
    if info and info["efforts"] and effort != "none" and effort not in info["efforts"]:
        listed = [lvl for lvl in EFFORTS[1:] if lvl in info["efforts"]]
        below = [lvl for lvl in listed if EFFORTS.index(lvl) < EFFORTS.index(effort)]
        level = below[-1] if below else listed[0]
        print(f"      {model} has no effort {effort}; sending {level}", flush=True)

    # none is thinking switched off; every other level rides on adaptive
    # thinking, the documented mode for every current model. A model that
    # rejects either (Haiku 4.5 takes neither) answers 400, which drive()
    # does not retry: the case fails, as intended -- nothing is remapped.
    knobs = ({"thinking": {"type": "disabled"}} if effort == "none" else
             {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}})

    def send(system: str, turns: list, drop_images: bool) -> str:
        turns, max_px = bound_images(turns)
        budget = ceiling["n"] * EMPTY_RETRY_BOOST if (room["on"] and max_tokens) else ceiling["n"]
        messages = []
        for t in turns:
            content = [{"type": "text", "text": t["text"]}] if t.get("text") else []
            for img, lab in ([] if drop_images else labelled(t)):
                content.append({"type": "text", "text": image_label(img, lab)})
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": _b64(img, max_px)}})
            messages.append({"role": "assistant" if t["role"] == "assistant" else "user",
                             "content": content or [{"type": "text", "text": "(empty)"}]})
        # Prompt caching is NOT automatic on this API: without a cache_control
        # field nothing is cached, and every round re-bought the whole
        # transcript and every seed image at full price (the usage
        # accounting below was written as if it were on; it was not). The
        # top-level cache_control asks the API to mark the last cacheable
        # block itself, so the longest stable prefix -- system prompt, seed
        # images, every earlier turn, the bulk of a round's input from round
        # two on -- is read at 0.1x the input price. The one-hour TTL (writes
        # 2x instead of 1.25x; Anthropic's review of the public harness,
        # 2026-09-21) keeps that prefix through a long sandbox exec, a
        # scoring pause or a rate-limit wait -- the five-minute default let
        # an agentic round re-buy the whole transcript after any such gap,
        # and at a 94 % cache-hit rate the dearer writes cost ~5 %.
        # Stream: max_tokens this large trips the SDK's HTTP timeout otherwise.
        if room["on"]:
            print(f"      retrying with max_tokens={budget:,} so the answer has room "
                  f"after the thinking", flush=True)
        while True:
            try:
                started = time.time()
                with client.messages.stream(model=model, system=system,
                                            max_tokens=budget, messages=messages,
                                            cache_control={"type": "ephemeral", "ttl": "1h"},
                                            **knobs) as st:
                    for _ in st:                                   # each event: the budget clock
                        _over_budget(started, call_budget, "anthropic call")
                    msg = st.get_final_message()
                break
            except anthropic.BadRequestError as e:
                # The model's ceiling is not published per model here; a 400
                # naming max_tokens says what it is ("... maximum of N") and
                # the value is halved until accepted, remembered per episode.
                # Any other 400 -- effort or adaptive thinking on a model
                # without them, thinking disabled at xhigh / max -- is
                # drive()'s deterministic 400.
                if "max_tokens" in str(e) and budget > 8000:
                    ceiling["n"] = budget = budget // 2
                    print(f"      {model} rejected max_tokens; sending {budget:,} "
                          f"for the rest of this episode", flush=True)
                    continue
                raise
        u = msg.usage
        # episode resends the seed images every round and they are read from
        # the cache, so cache reads/writes are most of the input on later
        # rounds; counting only input_tokens undercounts the run. The two
        # are also kept apart, so a results file shows what caching bought.
        cached = getattr(u, "cache_read_input_tokens", 0) or 0
        written = getattr(u, "cache_creation_input_tokens", 0) or 0
        secs = round(time.time() - started, 1)
        usage.append({"input_tokens": u.input_tokens + cached + written,
                      "cached_tokens": cached, "cache_write_tokens": written,
                      "output_tokens": u.output_tokens, "seconds": secs})
        print(f"      usage: prompt {u.input_tokens + cached + written:,} (cached {cached:,}, "
              f"cache write {written:,}), output {u.output_tokens:,}, {secs:.0f} s", flush=True)
        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            print(f"      refusal (category={cat}); treating as an empty turn", flush=True)
            return ""
        think = "".join(getattr(b, "thinking", "") or ""
                        for b in msg.content if b.type == "thinking")
        text = "".join(b.text for b in msg.content if b.type == "text")
        if think or msg.stop_reason != "end_turn":
            print(f"      reasoning {len(think):,} chars, content {len(text):,}"
                  + ("" if msg.stop_reason == "end_turn" else f", stop={msg.stop_reason}"),
                  flush=True)
        if msg.stop_reason == "max_tokens" and not _has_fence(text):
            # Truncated before (or inside) the answer: a failed call, not a
            # turn. With a --max-tokens cap: once more, with room; at the
            # model's own ceiling there is no room to give, and the reply is
            # handed back as it is. Either way the call is marked, so the
            # record counts how many rounds an episode lost this way
            # (tokens.truncated_replies): a lab reviewing the samples saw
            # Claude spend the whole 128k ceiling thinking on four of nine,
            # and the count is what says whether that costs a result.
            usage[-1]["truncated"] = True
            if not room["on"] and max_tokens:
                room["on"] = True
                raise EmptyContent(f"stop_reason=max_tokens at {budget:,} tokens with no "
                                   f"executable block (thinking {len(think):,} chars)")
            room["on"] = False
            print("      still truncated with the doubled budget; handing the reply "
                  "back (this round is lost)", flush=True)
            return text
        room["on"] = False
        return text
    call = drive(send, "anthropic call")
    # The window the Models API reported, for build_call's context_tokens.
    call.context_hint = (info or {}).get("max_input_tokens")    # type: ignore[attr-defined]
    return call


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


def openai_responses_call(model: str, max_tokens: int, usage: list,
                          api_key: str, effort: str | None = DEFAULT_EFFORT):
    """OpenAI's own endpoint, through the Responses API, streamed.

    Why not /chat/completions here (measured 2026-09-17/18 on gpt-6-astra):
    a round-one task request -- the system prompt that asks for the task
    plus "Begin." and the sheets -- returned no byte for 15 minutes on
    chat.completions, streamed or not, at low and at medium, while the same
    request without the system prompt answered in 4 s and, through the
    Responses API, answered in 10 s with its reasoning summaries streaming
    from the second second. The Responses API is also what an earlier
    harness of ours speaks. Its stream carries reasoning-summary deltas, so a
    thinking model is visibly alive and the per-read timeout means what it
    says; the summary text is never returned (it is not the answer).
    """
    import httpx
    import openai
    effort = effort or top_effort("openai/")
    check_effort("openai/", effort)
    client = openai.OpenAI(api_key=api_key,
                           timeout=httpx.Timeout(CALL_TIMEOUT_S, read=FIRST_PARTY_IDLE_S, connect=30.0),
                           max_retries=0)                 # drive() owns every retry, so the log says what happened
    room = {"on": False}
    knob = {"effort": OPENAI_EFFORT[effort]}
    call_budget = CALL_BUDGET_S.get(effort, CALL_BUDGET_DEFAULT_S)

    def _step_down(err: str) -> bool:
        if "effort" not in err or not knob["effort"]:
            return False
        was, knob["effort"] = knob["effort"], EFFORT_STEP_DOWN.get(knob["effort"])
        print(f"      {model} rejected reasoning.effort={was}; "
              + (f"sending {knob['effort']} instead" if knob["effort"] else "sending no reasoning.effort")
              + " for the rest of this episode", flush=True)
        return True

    def send(system: str, turns: list, drop_images: bool) -> str:
        turns, max_px = bound_images(turns)
        items: list = []
        for t in turns:
            if t["role"] == "assistant":
                items.append({"role": "assistant", "content": [{"type": "output_text", "text": t.get("text") or "(empty)"}]})
                continue
            parts = [{"type": "input_text", "text": t["text"]}] if t.get("text") else []
            for img, lab in ([] if drop_images else labelled(t)):
                parts.append({"type": "input_text", "text": image_label(img, lab)})
                parts.append({"type": "input_image", "detail": "high",
                              "image_url": "data:image/png;base64," + _b64(img, max_px)})
            items.append({"role": "user", "content": parts or [{"type": "input_text", "text": "(empty)"}]})
        budget = (max_tokens * EMPTY_RETRY_BOOST if room["on"] else max_tokens) if max_tokens else OPENAI_MAX_OUTPUT
        if room["on"]:
            print(f"      retrying with max_output_tokens={budget:,} so the content has room", flush=True)
        while True:
            reasoning = {"summary": "auto"}
            if knob["effort"]:
                reasoning["effort"] = knob["effort"]
            try:
                stream = client.responses.stream(model=model, instructions=system, input=items,
                                                 reasoning=reasoning, max_output_tokens=budget)
                break
            except openai.BadRequestError as e:
                if not _step_down(str(e)):
                    raise
        chunks, think, final, seen_usage, status, why_incomplete = [], [], None, None, None, None
        started = time.time()
        with stream as st:
            for ev in st:
                _over_budget(started, call_budget, "responses call")
                k = ev.type
                if k == "response.output_text.delta":
                    chunks.append(ev.delta)
                elif k == "response.reasoning_summary_text.delta":
                    think.append(ev.delta)
                elif k == "error":
                    raise RuntimeError(f"responses stream error: {getattr(ev, 'message', ev)}")
                elif k in ("response.completed", "response.incomplete", "response.failed"):
                    final = ev.response
            if final is None:
                final = st.get_final_response()
        status = getattr(final, "status", None)
        inc = getattr(final, "incomplete_details", None)
        why_incomplete = getattr(inc, "reason", None) if inc else None
        u = getattr(final, "usage", None)
        if u is not None:
            cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
            rtok = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", None)
            seen_usage = {"input_tokens": u.input_tokens, "cached_tokens": cached, "output_tokens": u.output_tokens,
                          "seconds": round(time.time() - started, 1)}
            if why_incomplete == "max_output_tokens" and not _has_fence("".join(chunks)):
                seen_usage["truncated"] = True                # the same mark as the Anthropic path
            usage.append(seen_usage)
            print(f"      usage: prompt {u.input_tokens:,} (cached {cached:,}), completion {u.output_tokens:,}"
                  + (f" (reasoning {rtok:,})" if rtok else "") + f", {seen_usage['seconds']:.0f} s", flush=True)
        else:
            print("      (provider reported no usage; token counts for this call are unknown, not zero)", flush=True)
        if status == "failed":
            err = getattr(final, "error", None)
            raise RuntimeError(f"response failed: {getattr(err, 'message', err)}")
        text, summary = "".join(chunks), "".join(think)
        if summary or status != "completed":
            print(f"      reasoning summary {len(summary):,} chars, content {len(text):,}"
                  + (f", status={status}" + (f" ({why_incomplete})" if why_incomplete else "") if status != "completed" else ""),
                  flush=True)
        if text:
            room["on"] = False
            return text
        why = f"content empty, status={status}, incomplete={why_incomplete}"
        if not room["on"] and max_tokens:
            room["on"] = True
            raise EmptyContent(why)
        room["on"] = False
        print(f"      {why}; handing the empty reply back (this round is lost)", flush=True)
        return ""
    return drive(send, "responses call")


def openai_compat_call(model: str, max_tokens: int, usage: list,
                       api_key: str, base_url: str | None,
                       effort: str | None = DEFAULT_EFFORT):
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
    # The provider's own levels, its top when none was given: OpenAI and
    # OpenRouter have a knob, xAI and OpenCode are sent nothing.
    who = "openai/" if base_url is None else "openrouter/" if _is_openrouter(base_url) else None
    if who:
        effort = effort or top_effort(who)
        check_effort(who, effort)
    client = openai.OpenAI(
        api_key=api_key, base_url=base_url,
        timeout=httpx.Timeout(CALL_TIMEOUT_S, read=FIRST_PARTY_IDLE_S if base_url is None else READ_IDLE_S,
                              connect=30.0))
    # Set for exactly ONE attempt, by the EmptyContent raise below, and cleared
    # on every other outcome: the boost is never compounded, so
    # max_tokens * EMPTY_RETRY_BOOST is the ceiling however many rounds run.
    room = {"on": False}
    # OpenAI's own effort knob (see OPENAI_EFFORT); None once stepped all the
    # way down. Only OpenAI itself gets it: xAI and OpenCode are not measured
    # against it, and OpenRouter has its `reasoning` block below.
    knob = {"effort": OPENAI_EFFORT[effort] if base_url is None else None}
    call_budget = CALL_BUDGET_S.get(effort, CALL_BUDGET_DEFAULT_S)

    def _step_down(err: str) -> bool:
        """A 400 that names reasoning_effort: lower the knob one step, or drop
        it, and say so. False when there is nothing left to lower."""
        if "reasoning_effort" not in err or not knob["effort"]:
            return False
        was, knob["effort"] = knob["effort"], EFFORT_STEP_DOWN.get(knob["effort"])
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
        budget = (max_tokens * EMPTY_RETRY_BOOST if room["on"] else max_tokens) if max_tokens else None
        extra: dict = {}
        if budget:
            extra["max_completion_tokens"] = budget
        elif base_url is None:
            extra["max_completion_tokens"] = OPENAI_MAX_OUTPUT       # the ceiling, not a cap to double
        if _is_openrouter(base_url):
            # The gateway's own effort knob, so "medium" means the same thing
            # here as it does on the anthropic path. It takes low | medium |
            # high, and an upstream is free to ignore it (measured: one host's
            # free models do).
            extra["extra_body"] = {"reasoning": {"effort": OR_EFFORT[effort]}}
        if room["on"] and budget:
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
                    model=model, messages=messages,
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
        started = time.time()
        try:
            for ch in stream:
                _over_budget(started, call_budget, "chat.completions call")
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
            seen_usage["seconds"] = round(time.time() - started, 1)
            usage.append(seen_usage)
            print(f"      usage: prompt {seen_usage['input_tokens']:,} "
                  f"(cached {seen_usage['cached_tokens']:,}), completion "
                  f"{seen_usage['output_tokens']:,}"
                  + (f" (reasoning {rtokens:,})" if rtokens else "") + f", {seen_usage['seconds']:.0f} s", flush=True)
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
        if not room["on"] and max_tokens:
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
                system_instruction=system,
                **({"max_output_tokens": max_tokens} if max_tokens else {})))
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

    def call(system: str, turns: list, plain: bool = False) -> str:
        return reply
    return call


# Context windows the runner assumes when --context-tokens is not given, by
# model-id prefix (longest match). The episode summarises its history the
# way Terminus 2 does when the provider's last-reported prompt size comes
# within SUMMARIZE_FREE_TOKENS of this; a context-length error summarises
# reactively whatever the number says, so a wrong entry costs one failed
# call, not the case.
# gpt-6-astra: 1,050,000 (OpenAI; requests over 272k input tokens are
# billed at 2x input / 1.5x output for the whole request -- a 30-round
# episode stays under that, a 100-round T2/T5 episode would not). The
# Claude 5 family: 1M from the Models API. Both providers therefore
# summarise at the same point.
# Where the episode summarises, whatever the window: a request that carries
# 180k tokens of transcript and observation images is close to the 32 MB
# request-size limit both first-party APIs enforce, and a run that compacts
# at the same point on every provider is comparable across them, which a
# run that compacts at each provider's own window is not (Anthropic's review
# of the public harness, 2026-09-21; OpenAI's own agents compact at about
# the same size). --context-tokens overrides in either direction. Measured
# on the gpt-6-astra sweep before this: the largest request of an episode
# was 90k tokens at the median, 142k at p90, 188k at most, so on 30-round
# episodes the point moves only the longest few.
COMPACT_TOKENS = 180_000
CONTEXT_TOKENS = {"claude-opus-5": 1_000_000, "claude-": 200_000,
                  "gpt-6": 1_050_000, "gpt-5": 400_000, "o3": 200_000, "o4": 200_000,
                  "gemini": 1_000_000}
DEFAULT_CONTEXT_TOKENS = 200_000


def context_tokens_for(model_id: str) -> int:
    best = None
    for prefix, n in CONTEXT_TOKENS.items():
        if model_id.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, n)
    return best[1] if best else DEFAULT_CONTEXT_TOKENS


def build_call(spec: str, max_tokens: int, usage: list, case: Path,
               effort: str | None = DEFAULT_EFFORT, context_tokens: int | None = None):
    prefix, prov, model_id = split_model(spec)
    effort = effort if effort is not None else top_effort(prefix)
    check_effort(prefix, effort)
    if prov.kind == "mock":
        call = mock_call(model_id, case)
    else:
        key = resolve_key(prefix, prov)
        if prov.kind == "anthropic":
            call = anthropic_call(model_id, max_tokens, usage, effort)
        elif prov.kind == "openai_compat" and prov.base_url is None:
            call = openai_responses_call(model_id, max_tokens, usage, key, effort)
        elif prov.kind == "openai_compat":
            call = openai_compat_call(model_id, max_tokens, usage, key, prov.base_url, effort)
        elif prov.kind == "gemini":
            call = gemini_call(model_id, max_tokens, usage, key)
        else:
            raise SystemExit(f"provider kind {prov.kind!r} has no call implementation")
    # What the episode reads to decide on summarising: the provider's own
    # count of the last prompt, and the window it has to fit in -- as given,
    # else as the provider reported it (anthropic's Models API), else the
    # table.
    call.usage = usage                                   # type: ignore[attr-defined]
    window = getattr(call, "context_hint", None) or context_tokens_for(model_id)
    call.context_window = window                         # type: ignore[attr-defined]
    call.context_tokens = context_tokens or min(window, COMPACT_TOKENS)   # type: ignore[attr-defined]
    return call


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
    (170 parametric cases sharing one work directory).

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


# The T6 verifier's own search deadline is 3600 s (task.toml); every other
# scorer is done in minutes -- unless a submitted part cannot be meshed:
# measured 2026-09-18, a swept B-spline wire clip (BRepCheck invalid) took
# 64 s at deflection 0.1 and never returned at the metric's, holding the
# scorer for 79 min. So a part/assembly score gets SCORE_TIMEOUT_S and a
# timeout is recorded as an error the case can be re-scored from later.
# 3600 since 2026-09-21 (was 900): a T5 assembly whose bought-in parts
# tessellate to 10^7 triangles (coil springs, caster assemblies) takes
# 15-30 min to score honestly, and a submission on it should not be
# recorded as an error for that.
SCORE_TIMEOUT_S = 3600
SCORE_TIMEOUT_ECAD_S = 4200


def score_in_subprocess(case: Path, artifact: Path) -> dict:
    """score_case in a child interpreter: the pixel term renders through
    vtk, whose Cocoa window on macOS may only be created on a process's main
    thread -- a child process has its own -- and the T6 graph search is pure
    Python that can run to its 3600 s deadline, which held the GIL of this
    process for 25 minutes on 2026-09-18 and starved every episode in
    flight. Scoring therefore leaves this process entirely, and can run from
    the worker that finished the episode."""
    import subprocess
    code = ("import json, sys\n"
            "from pathlib import Path\n"
            "from envs.common.score_case import score_case\n"
            "print('\\n' + json.dumps(score_case(Path(sys.argv[1]), Path(sys.argv[2])), default=str))")
    ecad = (case / "gt/gt_graph.json").exists()
    try:
        r = subprocess.run([sys.executable, "-c", code, str(case), str(artifact)],
                           capture_output=True, text=True,
                           timeout=SCORE_TIMEOUT_ECAD_S if ecad else SCORE_TIMEOUT_S,
                           cwd=str(Path(__file__).resolve().parents[1]))
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"scorer timed out after {e.timeout:.0f} s (re-score later)") from None
    if r.returncode != 0:
        raise RuntimeError(f"scorer exited {r.returncode}: {r.stderr.strip()[-600:]}")
    line = r.stdout.strip().splitlines()[-1]
    return json.loads(line)


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
                    help="thinking effort, sent as it is and only to a provider "
                         "that has it: anthropic output_config.effort "
                         "none..max (none = thinking disabled); openai "
                         "reasoning_effort none..max; openrouter "
                         "reasoning.effort none..high. Default: the "
                         "provider's top level (max, max, high)")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                    help=f"rounds per episode (default: {DEFAULT_ROUNDS})")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="cap on one reply (thinking included). Default: none "
                         "-- the model's own maximum")
    ap.add_argument("--context-tokens", type=int, default=None,
                    help="where the episode summarises its history (Terminus "
                         "2's way): when the last prompt comes within 8000 "
                         "tokens of this. Default: the smaller of the model's "
                         "window (harness/run.py CONTEXT_TOKENS) and "
                         "COMPACT_TOKENS (180000, the same point on every provider)")
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="default results/<model>_<timestamp>.json")
    ap.add_argument("--workers", type=int, default=1,
                    help="episodes run at once (threads: an episode waits on "
                         "HTTP or on its sandbox almost all of the time). "
                         "Each sandbox exec may take 2 GB; size the docker "
                         "host for workers x 2 GB")
    ap.add_argument("--score-workers", type=int, default=2,
                    help="scores running at once, each in a child process (default 2); "
                         "a worker hands its finished episode over and takes the next case. "
                         "0 records the episodes UNSCORED (score null, unscored true) for "
                         "tools/rescore.py on another machine -- the T6 matcher can take an "
                         "hour a board and need not hold the box that runs the episodes")
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
    defaulted = a.effort is None
    if defaulted:
        a.effort = top_effort(prefix)
    check_effort(prefix, a.effort)
    cases = _shard(discover(a.cases), a.shard)
    if not cases:
        raise SystemExit(f"--cases {a.cases}: no cases found")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = a.out or (Path("results") /
                         f"{a.model.replace('/', '_')}_{stamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = a.work or (REPO / "work" / f"run_{stamp}")
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
    effort_note = (f" ({prefix.rstrip('/')}'s top level)" if defaulted and a.effort else
                   " (no effort knob)" if a.effort is None else "")
    print(f"model {a.model}  provider {prefix.rstrip('/')}  "
          f"cases {len(cases)}  rounds {a.rounds}  effort {a.effort or '-'}{effort_note}  "
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
            call = build_call(a.model, a.max_tokens, usage, case, a.effort, a.context_tokens)
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
            "calls": len(usage),
            # Replies that stopped at the output ceiling before an executable
            # block (the thinking used it all): each is a round the episode
            # lost to its nudge, and the count is the measure of it.
            "truncated_replies": sum(1 for u in usage if u.get("truncated"))}
        return rec

    done = 0

    def score(case: Path, rec: dict) -> dict:
        """Score in a child process (score_in_subprocess), from whichever
        thread ran the episode: the process's GIL and main thread stay free
        for the episodes still in flight."""
        if "error" not in rec and "skipped" not in rec:
            if a.score_workers == 0:
                # Judged later, elsewhere: the record is final for --resume
                # (it has a "score" key) and tools/rescore.py fills it in.
                rec["score"] = None
                rec["unscored"] = bool(rec.get("step"))
                rec["seconds_score"] = 0.0
                return rec
            t0 = time.time()
            try:
                artifact = rec.get("step")
                rec["score"] = score_in_subprocess(case, Path(artifact)) if artifact else None
            except Exception as e:                               # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["traceback"] = traceback.format_exc()[-2000:]
            rec["seconds_score"] = round(time.time() - t0, 1)
        return rec

    def finish(case: Path, rec: dict) -> None:
        """Record a scored episode and print its line."""
        nonlocal done
        with lock:
            records[str(case)] = rec
            done += 1
            n = done
            write()
        score = rec.get("score")
        line = (show(score) if score
                else rec.get("skipped") and f"skipped: {rec['skipped']}"
                or rec.get("error") or ("unscored" if rec.get("unscored") else "no submission"))
        print(f"  [{n}/{len(todo)}] {case.name}: {line}  "
              f"({rec['seconds']}s)", flush=True)

    # Episodes and scoring are two pools: a worker hands its finished record
    # to the scoring pool and takes the next case at once, so a slow score
    # (T6's graph search can run to its 3600 s deadline) never idles an
    # episode slot; the scores themselves run in child processes.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    write()
    with ThreadPoolExecutor(max_workers=max(1, a.score_workers)) as scorers, \
         ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        episodes = {ex.submit(one, c): c for c in todo}
        # Each score is recorded the moment it lands (a done-callback on the
        # scoring thread), not after the last episode: the results file is
        # current at every moment and a crash loses one case, not the run.
        # Measured 2026-09-18 03:00: the earlier "score everything, then
        # record" shape held 14 finished episodes unrecorded for an hour.
        for f in as_completed(episodes):
            case = episodes[f]
            scorers.submit(score, case, f.result()).add_done_callback(
                lambda sf, c=case: finish(c, sf.result()))
        scorers.shutdown(wait=True)
    print(f"\n{len(records)} cases -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
