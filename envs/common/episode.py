#!/usr/bin/env python
"""Turn-based episode. Text and images only -- no provider tool protocol.

Each round the model replies with one fenced block:
  ```python   executed in the working directory; next round it sees stdout,
              stderr and any new images
  ```submit   the final answer: a complete CadQuery program leaving the final
              geometry in `result`. The harness runs it, exports STEP, and
              computes voxel IoU against the ground truth.

Three inherited lessons: the prompt states the interface and does not teach
strategy; the final round explicitly demands a submission (otherwise the model
spends the whole budget measuring and the case is wasted); and a failed
submission falls back to the last usable STEP in the log rather than scoring
zero.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .sandbox import Sandbox
from .submission import SUB_ROOT, is_ready

MAX_ROUNDS = 8
# How many consecutive failed calls before treating it as a real fault rather
# than one network hiccup.
MAX_DEAD_ROUNDS = 3
# The cadquery version in the sandbox image. Telling the model reduces API
# hallucination (measured: it called cadquery.selectors.and_, which exists in
# neither 2.3.0 nor 2.8.0).
SANDBOX_CQ_VERSION = "2.3.0"
# Permissive fence matching. Models write ```py / ```cadquery / ```Python, and
# also truncate a reply on a channel marker so the closing ``` is lost. The
# original strict regex classified all of those as no_block -- measured: T3
# gn866 produced 100 consecutive no_block rounds and wasted 3M tokens.
#   - language tag is case-insensitive; python|py|submit|cadquery accepted
#   - if the closing ``` is missing, take text to the end (a truncated block
#     still beats discarding the round)
_FENCE = re.compile(
    r"```[ \t]*(python|py|submit|cadquery)?[ \t]*\r?\n(.*?)(?:```|\Z)",
    re.DOTALL | re.IGNORECASE)
# Bare-code fallback: sometimes a model emits Python with no fence at all
# (measured: a reply ending in `print(f'...center=(...)')`). These signals
# separate "code" from "prose" well enough; a false positive costs one round
# (the model sees the error), which beats discarding the round entirely.
_CODEISH = re.compile(
    r"^\s*(import |from \w+ import |print\(|result\s*=|\w+\s*=\s*cq\.)",
    re.MULTILINE)

# The closing "your entire reply must be exactly one fenced block" line is an
# INTERFACE CONTRACT, not a strategy hint. Do not remove it. Without it, some
# models degrade into channelled output: the body carries only a plan,
# `monologue` appears in the text as a separator, and the code-block channel
# never reaches `text` at all (measured: 50-token completions, four consecutive
# retries the same way, none of 10 cases able to start). With it, they emit a
# well-formed fenced block immediately.
SYSTEM = """You are solving a CAD task in a working directory, over {rounds} rounds.

{task_brief}

Directory:
{file_list}
tools.py:
{tools_help}

Python 3.12: cadquery {cq_version} (cadquery-ocp 7.9), numpy, scipy, PIL,
trimesh, vtk, matplotlib, ezdxf. No network.

Each reply is exactly one fenced block and nothing outside it:
```python    runs as a fresh process in the directory (files persist, memory
             does not). You get back stdout and stderr (each limited to
             10,000 bytes: the first and last 5,000) and every PNG it wrote
             at the top level of the directory, each with its file name; a
             PNG written anywhere else is not shown. 600 s, 2 GB, 1 CPU.
```submit    your final answer: {submit_what}
             Ends the episode; this is what is scored.
"""


# Every input image is a seed image; there is no cap. A case's inputs are the
# task, and a model that has to fetch the 13th drawing itself is being handed
# a different task from one that gets all twelve. What bounds a request's
# image count is the harness (harness.run: observation images older than a
# few rounds leave the request, and a request over the API's many-image
# threshold is downscaled to its per-image limit), not the case.
_HIDDEN = ("tools.py", "_render.py", "sitecustomize.py")
# The standard-parts library has 269 files; listing them individually drowns
# the file listing. Report the directory and a count instead.
_SUMMARIZE_DIRS = ("stdparts",)


def _visible(p: Path) -> bool:
    return p.name not in _HIDDEN and not p.name.startswith("_")


def _listing(root: Path) -> list[str]:
    """The directory, compactly: one line per top-level file, one line per
    subdirectory naming what is inside; a drawing's tiles are a count next
    to their sheet, not one line each. (A T5 directory has 39 files; listed
    one per line the tiles alone were a third of the prompt.)"""
    def _names(files: list[Path]) -> str:
        sheets = [x.name for x in files if not _is_tile(x)]
        tiles = [x for x in files if _is_tile(x)]
        if len(sheets) > 8:
            shown = f"{sheets[0]}, {sheets[1]} ... {sheets[-1]} ({len(sheets)} files)"
        else:
            shown = ", ".join(sheets)
        if tiles:
            shown += f"; {len(tiles)} tiles <sheet>_tile_r<i>c<j>.png"
        return shown
    out = []
    top = [p for p in sorted(root.iterdir()) if _visible(p)]
    for p in top:
        if p.is_dir():
            kids = sorted(x for x in p.iterdir() if _visible(x))
            if p.name in _SUMMARIZE_DIRS:
                out.append(f"  {p.name}/   {len(kids)} files -- see index.json")
            else:
                out.append(f"  {p.name}/   {_names(kids)}")
    files = [p for p in top if not p.is_dir()]
    tiles = [p for p in files if _is_tile(p)]
    for p in files:
        if _is_tile(p):
            continue
        line = f"  {p.name}"
        stem = p.name[:-4] if p.suffix == ".png" else None
        mine = [t for t in tiles if stem and t.name.startswith(stem + "_tile_")]
        if mine:
            line += f"   + {len(mine)} tiles {stem}_tile_r<i>c<j>.png"
        out.append(line)
    return out


def _is_tile(p: Path) -> bool:
    return "_tile_r" in p.name


def _seed_images(root: Path) -> list[Path]:
    """Prompt images: top-level first (assembly drawing / reference views),
    then subdirectory images sorted by name. A drawing's tiles are NOT
    seeds: the sheet is the overview, the tiles are the legible copies on
    disk, and the model has them shown the way it has any image shown -- a
    PNG it writes or copies at the top level of the directory is attached
    next round -- so a round carries one image per sheet instead of one per
    tile (a T2 case seeded 13 images, a T5 case 20, every round)."""
    top = [p for p in sorted(root.iterdir()) if _visible(p) and p.suffix == ".png" and not _is_tile(p)]
    sub = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and _visible(d) and d.name not in _SUMMARIZE_DIRS:
            sub += [x for x in sorted(d.iterdir()) if _visible(x) and x.suffix == ".png" and not _is_tile(x)]
    return top + sub


def _blocks(raw: str):
    """Return (python block, submit block); for each kind, the last one."""
    raw = raw or ""
    py = sub = None
    for kind, body in _FENCE.findall(raw):
        if not kind:
            # An untagged fence must NOT be executed as Python
            # unconditionally. Models use it for ASCII sketches and tables
            # (measured twice in 462 rounds: "+----+ <- top, width 10" and
            # "view0 (yaw30)  view1 (yaw120)"); executing those buys a
            # SyntaxError and wastes the round. Accept only if it looks like
            # code.
            if _CODEISH.search(body):
                py = body
            continue
        k = kind.lower()
        if k in ("python", "py"):
            py = body
        else:                                   # submit / cadquery
            sub = body
    if py is None and sub is None and _CODEISH.search(raw):
        py = raw                                # bare-code fallback
    return py, sub


_ANSWERISH = re.compile(r"^\s*result\s*=|import\s+cadquery|from\s+cadquery", re.M)


def _is_answer(code: str | None) -> bool:
    """Is the recovered block AN ANSWER, or a script that only measures?

    A submitted program is appended with `tools.export(result, 'final.step')`
    before execution, so a measurement-only script without `result` cannot
    score anyway -- **the score was always correct**. This predicate does not
    protect the score, it protects the COUNTS: recording "the model never
    converged" as "we recovered its answer" is the mirror image of the parsing
    bug this fallback was added to fix.

    Measured elsewhere: two records that ran 100 rounds without submitting
    ended with 18- and 47-line numpy/PIL scripts, no `result` and no cadquery
    import -- not an answer in the wrong slot, no answer at all.
    """
    return bool(code and _ANSWERISH.search(code))


OUTPUT_LIMIT_BYTES = 10_000


def _limit_output(output: str, max_bytes: int = OUTPUT_LIMIT_BYTES) -> str:
    """Terminus 2's rule (Terminal-Bench 4.0): an observation is at most
    max_bytes, the first and last halves kept, the middle replaced by a
    line that says how much was omitted. Tail kept because a model prints
    its conclusion last; head kept because the first error is first."""
    output = output.strip()
    data = output.encode("utf-8")
    if len(data) <= max_bytes:
        return output
    half = max_bytes // 2
    first = data[:half].decode("utf-8", errors="ignore")
    last = data[-half:].decode("utf-8", errors="ignore")
    omitted = len(data) - len(first.encode("utf-8")) - len(last.encode("utf-8"))
    return (f"{first}\n[... output limited to {max_bytes} bytes; "
            f"{omitted} interior bytes omitted ...]\n{last}")


# Appended to the submitted program before it is run. `tools.finish` exports
# `result` to final.step when the program left one (T1 / T3 / T6 and the old
# single-STEP assembly answer, unchanged), and otherwise requires the fixed
# submission layout to be on disk -- so a program that leaves neither says so
# in its own stderr and gets the one retry, instead of ending as a silent zero.
FINALISE = "\n\nimport tools as _harness_tools\n_harness_tools.finish(globals())\n"


def _artifact(box: Sandbox):
    """What the program submitted. The fixed submission DIRECTORY wins when it
    is complete (envs.common.submission): on an assembly task it is the answer,
    and a `result` left over from the model's own bookkeeping is not. Otherwise
    the exported STEP, exactly as before."""
    sub = box.dir / SUB_ROOT
    if is_ready(sub):
        return sub
    # Then the exported STEP; then the graph an ECAD answer writes
    # (tools.export turns a dict into pred_graph.json). The T6 oracle used to
    # come back as "failed to execute" and burn its retry because only
    # final.step was looked for here.
    for name in ("final.step", "pred_graph.json"):
        f = box.dir / name
        if f.exists():
            return f
    return None


def _observation(rnd, res, max_rounds):
    parts = [f"Round {rnd}/{max_rounds} -- exit {res.returncode}"]
    if res.stdout.strip():
        parts.append(f"stdout:\n{_limit_output(res.stdout)}")
    if res.stderr.strip():
        parts.append(f"stderr:\n{_limit_output(res.stderr)}")
    if res.images:
        # Every PNG the round wrote is attached, newest first, and stays in
        # the conversation for the rest of the episode. A cap (three by name,
        # then six) made the model rename files to get the next one shown.
        parts.append("images produced: " + ", ".join(p.name for p in res.images))
    return "\n\n".join(parts), list(res.images)


def tools_help(case_dir: Path) -> str:
    """The tools block of the prompt, from the task declaration, so it never
    advertises a callable the sandbox does not stage. One line per tool:
    signature -> effect. What a tool is good for is the model's call."""
    from envs.common.score_case import load_task
    task = load_task(Path(case_dir)) or {}
    renderer = (task.get("tools") or {}).get("renderer", "none")
    kind = (task.get("task") or {}).get("kind", "part")
    given = (task.get("task") or {}).get("given", "")
    lines = []
    if kind == "assembly":
        lines.append("  export_part(solid_or_step_path, part_id) -> submission/parts/<part_id>.step")
        if given != "nothing_3d":
            lines.append("  use_part(part_id)            -> submission/parts/<part_id>.step, the supplied file unchanged")
        lines.append("  submit_assembly(instances)   -> submission/assembly/instances.json;"
                     " instances = [{part_id, transform[, instance_id]}]")
        lines.append("  export(solid, path)          -> STEP, for your own checks; not the answer")
    elif kind == "ecad":
        lines.append("  export(result, path)         -> pred_graph.json (result is the graph dict)")
    else:
        lines.append("  export(result, path)         -> STEP")
    if renderer == "shared":
        lines += ["  render(step, png)            -> isometric hidden-line view",
                  "  views(step, png)             -> four-orientation 2x2 sheet, the reference's renderer and angles"]
    has_sheets = any(str(i).endswith(".pdf") or "drawing" in str(i)
                     for i in (task.get("task") or {}).get("inputs", []))
    lines.append("  crop(png, (l, t, r, b)[, out]) -> a PNG of that box, in the file's own pixel coordinates"
                 + (" (a sheet file is larger than shown; the crop is cut from a 2x master)" if has_sheets else ""))
    return "\n".join(lines)


# What ```submit has to contain, per task kind: one wording for the system
# prompt and one for the three reminders (no block / rounds finished / the
# program crashed). An assembly answer is the submission/ directory the
# program writes with the tools; a `result` there is optional. The reminders
# used to demand "the final solid in `result`" on every task, which on an
# assembly task told the model to do something the scorer does not read.
SUBMIT_WHAT = {
    "part": "a complete CadQuery program that leaves the solid in `result`.",
    "assembly": "a complete program that writes submission/ with the tools\n"
                "             (export_part / use_part, then submit_assembly).",
    "assembly_nothing_3d": "a complete program that writes submission/ with the tools\n"
                           "             (export_part for every part, then submit_assembly).",
    "ecad": "a complete program that leaves the graph dict in `result`.",
}
ANSWER = {
    "part": "the complete CadQuery program for your best geometry, leaving the solid in `result`",
    "assembly": "the complete program that writes submission/ with the tools for your best assembly",
    "ecad": "the complete program that leaves your best graph dict in `result`",
}


def _kind(case_dir: Path) -> str:
    from envs.common.score_case import load_task
    task = load_task(Path(case_dir)) or {}
    return (task.get("task") or {}).get("kind", "part")


def _submit_what(case_dir: Path) -> str:
    from envs.common.score_case import load_task
    t = (load_task(Path(case_dir)) or {}).get("task") or {}
    kind = t.get("kind", "part")
    if kind == "assembly" and t.get("given") == "nothing_3d":
        return SUBMIT_WHAT["assembly_nothing_3d"]
    return SUBMIT_WHAT[kind]


def _task_brief(case_dir: Path) -> str:
    """The brief the system prompt is built from.

    Order: the case's own brief.txt, then the env named by case.json, then the
    legacy walk-up to the literal "cases" path component (a case is
    envs/<env>/cases/<name> flat, envs/<env>/cases/<family>/<nn> in
    the data tree, so a fixed parent count lands wrong).

    The walk-up used to be the only rule and ran before the brief.txt check, so
    a case outside an envs/<env>/cases/ tree raised
    `ValueError: tuple.index(x): x not in tuple` before the episode began --
    tests/fixtures/<task>/<case> could never be run at all.
    """
    own = case_dir / "brief.txt"
    if own.exists():
        return own.read_text()
    cj = case_dir / "case.json"
    if cj.exists():
        try:
            env = json.loads(cj.read_text()).get("env")
        except (ValueError, OSError):
            env = None
        if env:
            t = Path(__file__).resolve().parents[2] / "envs" / str(env) / "TASK.md"
            if t.exists():
                return t.read_text()
    parts = case_dir.parts
    if "cases" in parts:
        env_dir = Path(*parts[:len(parts) - 1 - parts[::-1].index("cases")])
        return (env_dir / "TASK.md").read_text()
    raise FileNotFoundError(
        f"{case_dir}: no brief.txt, case.json names no env with a TASK.md, and "
        f'the path has no "cases" component -- nothing to build a brief from')


# ── context summarization, the way Terminus 2 (Terminal-Bench 4.0) does it ──
# When the conversation nears the model's context limit, three calls with
# the same model replace it: (1) the agent summarises its own work so far,
# (2) a fresh agent, given the task, the summary and the directory, asks the
# questions the summary leaves open, (3) the old context answers them. The
# conversation then restarts as [seed turn, questions prompt, questions,
# answers + "continue"]. Proactively when the last request left fewer than
# SUMMARIZE_FREE_TOKENS free (Terminus 2: 8000), reactively when a call
# fails on context length -- then the newest turns are unwound first until
# the summary request itself fits. Images from the summarised turns leave
# with them (their files stay in the directory, and the hand-off says so);
# the seed images are re-sent as in every round.
SUMMARIZE_FREE_TOKENS = 8000
UNWIND_FREE_TOKENS = 4000
IMAGE_TOKEN_ESTIMATE = 4800          # the API's per-image ceiling; conservative on purpose
_CONTEXT_ERROR = re.compile(r"context.length|context_length|maximum context|too many tokens|"
                            r"prompt is too long|input length|exceeds the model|token limit", re.I)

SUMMARY_PROMPT = """You are about to hand off your work to another AI agent.
Please provide a comprehensive summary of what you have accomplished so far on this task:

Original Task: {brief}

Based on the conversation history, please provide a detailed summary covering:
1. **Major Actions Completed** - List each significant program you ran and what you learned from it.
2. **Important Information Learned** - A summary of crucial findings: measurements, dimensions, coordinates, file locations, errors, and the state of the working directory.
3. **Challenging Problems Addressed** - Any significant issues you encountered and how you resolved them.
4. **Current Status** - Exactly where you are in the task completion process.

Be comprehensive and detailed. The next agent needs to understand everything that has happened so far in order to continue."""

QUESTIONS_PROMPT = """You are picking up work from a previous AI agent on this task:

**Original Task:** {brief}

**Summary from Previous Agent:**
{summary}

**Current working directory:**
{listing}

**Last observation:**
{last_obs}

Please begin by asking several questions (at least five, more if necessary) about the current state of the solution that are not answered in the summary from the prior agent. After you ask these questions you will be on your own, so ask everything you need to know."""

HANDOFF_PROMPT = """Here are the answers the other agent provided.

{answers}

Continue working on this task from where the previous agent left off. You can no longer ask questions. The images from the earlier rounds are no longer in this conversation; their files are still in the directory, and any PNG you write at the top level is shown to you. Reply with one fenced block."""


def _est_tokens(system: str, turns: list) -> int:
    n = len(system) // 4
    for t in turns:
        n += len(t.get("text") or "") // 4 + IMAGE_TOKEN_ESTIMATE * len(t.get("images") or [])
    return n


def _plain(call_fn, system: str, turns: list) -> str:
    """A call whose reply is prose, not a fenced block (harness.run.drive
    retries fenceless replies; `plain` switches that off)."""
    try:
        return call_fn(system, turns, plain=True) or ""
    except TypeError:
        return call_fn(system, turns) or ""


def _needs_summary(call_fn) -> tuple[bool, int, int]:
    """(needed, last_prompt_tokens, limit) from what the provider reported
    for the last request and the limit the runner declared."""
    usage = getattr(call_fn, "usage", None)
    limit = getattr(call_fn, "context_tokens", None)
    if not usage or not limit:
        return False, 0, limit or 0
    last = int(usage[-1].get("input_tokens", 0) or 0)
    return limit - last < SUMMARIZE_FREE_TOKENS, last, limit


def _summarize(call_fn, system: str, turns: list, brief: str, box: Sandbox,
               rounds: list, why: str, log_n: int) -> list:
    """Terminus 2's three steps; returns the new conversation."""
    limit = getattr(call_fn, "context_tokens", None)
    history = list(turns)
    if why == "reactive" and limit:
        while len(history) > 1 and limit - _est_tokens(system, history) < UNWIND_FREE_TOKENS:
            history = history[:-2] if len(history) >= 3 else history[:1]
    seed = history[0]
    n_user = sum(1 for t in history[1:] if t.get("role") != "assistant")
    last_obs = next((t["text"] for t in reversed(history[1:]) if t.get("role") != "assistant"), "")
    plain_system = ("You are an AI agent handing over, or taking over, a CAD task in a working "
                    "directory. Reply in plain text.")
    summary = _plain(call_fn, plain_system, history + [{"role": "user", "text": SUMMARY_PROMPT.format(brief=brief), "images": []}])
    listing = "\n".join(_listing(box.dir))
    q_prompt = QUESTIONS_PROMPT.format(brief=brief, summary=summary, listing=listing,
                                       last_obs=_limit_output(last_obs, 4000))
    questions = _plain(call_fn, plain_system, [{"role": "user", "text": q_prompt, "images": []}])
    answers = _plain(call_fn, plain_system, history + [
        {"role": "user", "text": SUMMARY_PROMPT.format(brief=brief), "images": []},
        {"role": "assistant", "text": summary, "images": []},
        {"role": "user", "text": "The next agent has a few questions for you, please answer each of them one by one in detail:\n\n" + questions, "images": []}])
    try:
        (box.log_dir / f"summary_{log_n:02d}.json").write_text(json.dumps(
            {"why": why, "turns_summarised": n_user, "summary": summary, "questions": questions, "answers": answers}, indent=1))
    except OSError:
        pass
    rounds.append({"round": len(rounds) + 1, "action": "summarize", "why": why, "turns_summarised": n_user})
    print(f"      context summarised ({why}): {n_user} rounds of history -> summary + "
          f"{len(questions)} chars of questions + {len(answers)} chars of answers", flush=True)
    return [seed,
            {"role": "user", "text": q_prompt, "images": []},
            {"role": "assistant", "text": questions, "images": []},
            {"role": "user", "text": HANDOFF_PROMPT.format(answers=answers), "images": []}]


def run_episode(case_dir: Path, work_dir: Path, call_fn,
                max_rounds: int = MAX_ROUNDS, exec_timeout: int = 600) -> dict:
    """Run one case.

    call_fn(system, turns) -> str, where turns is
    [{"role", "text", "images": [Path, ...]}, ...].
    Returns the submitted code and the final STEP.
    """
    case_dir = Path(case_dir)
    box = Sandbox(case_dir, work_dir)
    task_brief = _task_brief(case_dir)
    files = _listing(box.dir)
    kind = _kind(case_dir)
    system = SYSTEM.format(task_brief=task_brief.strip(),
                           file_list="\n".join(f"  {n}" for n in files),
                           tools_help=tools_help(case_dir),
                           submit_what=_submit_what(case_dir),
                           rounds=max_rounds, cq_version=SANDBOX_CQ_VERSION)
    answer = ANSWER[kind]
    # Images must be collected RECURSIVELY. T5's part drawings live in the
    # part_drawings/ subdirectory; the original scan was top-level only, so the
    # "all part drawings -> assembly" task never actually showed the model any
    # part drawing -- it received one assembly drawing and the reference views,
    # and the premise of the task disappeared. Subdirectory images are part of
    # the prompt too.
    seed_imgs = _seed_images(box.dir)
    turns = [{"role": "user", "text": "Begin.", "images": seed_imgs,
              "image_labels": [p.relative_to(box.dir).as_posix() for p in seed_imgs]}]

    rounds, submitted = [], ""
    dead = 0
    summaries = 0
    for rnd in range(1, max_rounds + 1):
        needed, last_tokens, limit = _needs_summary(call_fn)
        if needed:
            print(f"      last request {last_tokens:,} tokens of a {limit:,} context; "
                  f"summarising proactively", flush=True)
            summaries += 1
            turns = _summarize(call_fn, system, turns, task_brief, box, rounds, "proactive", summaries)
        # A failed call should waste THIS ROUND, not the whole case. Measured
        # in v2: PART-0043 / -0199 / -0211 all hit a stream interruption on
        # the first call (the provider never sent response.completed), the
        # exception propagated out, and the case scored 0 with none of its 30
        # rounds used -- that is the network's score, not the model's.
        # Only after MAX_DEAD_ROUNDS consecutive failures is it a real fault,
        # and only then is the exception allowed out.
        try:
            try:
                raw = call_fn(system, turns)
            except Exception as e:                       # noqa: BLE001
                if not _CONTEXT_ERROR.search(str(e)) or summaries >= max_rounds:
                    raise
                print(f"      context length error ({str(e)[:80]}); summarising", flush=True)
                summaries += 1
                turns = _summarize(call_fn, system, turns, task_brief, box, rounds, "reactive", summaries)
                raw = call_fn(system, turns)
            dead = 0
        except Exception as e:                           # noqa: BLE001
            dead += 1
            rounds.append({"round": rnd, "action": "call_failed",
                           "error": f"{type(e).__name__}: {e}"})
            if dead >= MAX_DEAD_ROUNDS:
                print(f"      round {rnd} call failed ({type(e).__name__}), "
                      f"{dead} in a row -- treating as a real fault, "
                      f"abandoning this case", flush=True)
                raise
            print(f"      round {rnd} call failed ({type(e).__name__}), "
                  f"round discarded, continuing "
                  f"[{dead}/{MAX_DEAD_ROUNDS}]", flush=True)
            continue
        # The raw reply must be archived. When no parsable block comes back
        # (no_block), without it there is no way to tell whether the model
        # wrote nothing, used a different fence, or was truncated -- measured:
        # T2 produced 9 consecutive no_block rounds and only became
        # investigable after this line was added.
        try:
            (box.log_dir / f"round_{rnd:02d}_raw.txt").write_text(raw or "")
        except OSError:
            pass
        turns.append({"role": "assistant", "text": raw or "", "images": []})
        py, sub = _blocks(raw)
        if sub is not None:
            submitted = sub
            rounds.append({"round": rnd, "action": "submit"})
            break
        if py is None:
            rounds.append({"round": rnd, "action": "no_block"})
            # This nudge must RESTATE THE FORMAT IN FULL, not just say "not
            # found". A first-round parse failure is self-reinforcing: the
            # model receives a nudge instead of an observation, stays in prose
            # mode, and fails again. Measured in another session: 64-76% of
            # episodes failed to parse round one, dropping to 1.3-2.5% once
            # inside the tool loop (a 30-65x difference) -- so spending a few
            # dozen tokens to push the model into the loop is worth it.
            turns.append({"role": "user", "text":
                          "Your reply contained no runnable block, so nothing "
                          "was executed and the round was wasted.\n\n"
                          "Reply with exactly one fenced block and nothing else "
                          "outside it:\n\n"
                          "```python\n# code to run in the working directory\n```\n\n"
                          "or, when you are ready to answer:\n\n"
                          f"```submit\n# {answer}\n```\n\n"
                          "Start your reply with the opening ``` -- no preamble, "
                          "no analysis before it.",
                          "images": []})
            continue
        res = box.run(py, timeout=exec_timeout)
        rounds.append({"round": rnd, "action": "exec", "rc": res.returncode})
        text, imgs = _observation(rnd, res, max_rounds)
        turns.append({"role": "user", "text": text, "images": imgs})

    # No forced final submission and no rescue: an episode whose rounds run
    # out without a ```submit has no answer, and is scored as such (the
    # record says `no_submission`). The budget is in the prompt and every
    # observation counts the rounds; asking on the model's behalf measured
    # the harness, not the model (BenchCAD-main PR 53: the two rescues were
    # the whole apparent advantage over mini-swe-agent).
    if not submitted and rounds:
        rounds.append({"round": len(rounds) + 1, "action": "no_submission"})

    # The submitted program runs once, exactly as submitted. If it fails to
    # execute, or leaves no answer, there is no answer: no resubmission with
    # the error shown, no "last complete submission in the log", no "newest
    # STEP anywhere". Those were rescues (BenchCAD-main PR 53 removed the same
    # ones): a model whose final program does not run has not answered, and
    # the record says so -- `submit_failed` with the stderr tail -- so the
    # reader can tell a crash from a wrong shape.
    step = None
    if submitted.strip():
        res = box.run(submitted + FINALISE, timeout=exec_timeout)
        if res.ok:
            step = _artifact(box)
        if step is None:
            rounds.append({"round": len(rounds) + 1, "action": "submit_failed",
                           "rc": res.returncode,
                           "stderr": (res.stderr or "").strip()[-1500:]})

    out = {"code": submitted, "step": str(step) if step else None,
           "artifact": (None if step is None else
                        "submission_directory" if step.is_dir() else "step"),
           "rounds": rounds, "submitted": bool(submitted)}
    (box.log_dir / "episode.json").write_text(json.dumps(out, indent=1))
    return out
