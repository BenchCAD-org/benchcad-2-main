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
SYSTEM = """You are solving a CAD task in a working directory.

{task_brief}

Files in your directory:
{file_list}
{tools_help}

Python 3.12 with cadquery {cq_version} (cadquery-ocp 7.9), numpy, scipy, PIL,
trimesh, vtk, matplotlib and ezdxf. There is no network. Stick to APIs that
exist in those versions -- a call that does not exist raises at submit time.

```python    runs in the directory; you get back stdout, stderr, and up to
             three images it wrote (every image is sent with its file name)
```submit    your final answer: {submit_what}
             It ends the episode and is what gets scored.

The directory persists across rounds. You have {rounds} rounds.

Your entire reply must be exactly one fenced block and nothing else -- no
narration, no plan, no prose before or after it. Start the reply with ```.
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
    """File listing, descending one level into subdirectories.

    Otherwise `part_drawings` is a bare directory name and the model knows
    neither how many drawings are inside nor what they are called.
    """
    out = []
    for p in sorted(root.iterdir()):
        if not _visible(p):
            continue
        if p.is_dir():
            kids = sorted(x for x in p.iterdir() if _visible(x))
            if p.name in _SUMMARIZE_DIRS:
                out.append(f"{p.name}/   ({len(kids)} files -- see index.json)")
                continue
            out.append(f"{p.name}/")
            out += [f"  {p.name}/{x.name}" for x in kids]
        else:
            out.append(p.name)
    return out


def _seed_images(root: Path) -> list[Path]:
    """Prompt images: top-level first (assembly drawing / reference views),
    then subdirectory images sorted by name."""
    top = [p for p in sorted(root.iterdir()) if _visible(p) and p.suffix == ".png"]
    sub = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and _visible(d) and d.name not in _SUMMARIZE_DIRS:
            sub += [x for x in sorted(d.iterdir()) if _visible(x) and x.suffix == ".png"]
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


def _clip(s: str, head: int = 2000, tail: int = 4000) -> str:
    """Truncate long output while KEEPING THE TAIL.

    The original was `s[:2000]`, a head-only truncation -- but a model's
    measurements are almost always printed last (compute, then print), so that
    discarded exactly the part it most needed. Measured in another session:
    30% of stdout was truncated, and 36% of those lost the model's own
    conclusion line. Give more tail than head, and say how much was elided.
    """
    s = s.strip()
    if len(s) <= head + tail:
        return s
    return (f"{s[:head]}\n\n...[{len(s) - head - tail} characters elided]...\n\n{s[-tail:]}")


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
    f = box.dir / "final.step"
    return f if f.exists() else None


def _observation(rnd, res, max_rounds):
    parts = [f"Round {rnd}/{max_rounds} -- exit {res.returncode}"]
    if res.stdout.strip():
        parts.append(f"stdout:\n{_clip(res.stdout)}")
    if res.stderr.strip():
        parts.append(f"stderr:\n{_clip(res.stderr)}")
    if res.images:
        shown = res.images[:3]
        parts.append("images produced: " + ", ".join(p.name for p in res.images)
                     + (f" (the first {len(shown)} are attached)" if len(res.images) > len(shown) else ""))
    if rnd >= max_rounds - 1:
        parts.append("This is your final observation -- reply now with your "
                     "```submit answer, using the best you have.")
    return "\n\n".join(parts), list(res.images[:3])


def tools_help(case_dir: Path) -> str:
    """The tools block of the prompt, generated from the task declaration so it
    never advertises a callable the sandbox does not stage. Only a task
    with renderer = "shared" gets render/views; every task gets export and crop;
    an assembly task gets the three calls that write the fixed submission
    layout (envs.common.submission), because on those tasks the answer is that
    directory and not a single STEP."""
    from envs.common.score_case import load_task
    task = load_task(Path(case_dir)) or {}
    renderer = (task.get("tools") or {}).get("renderer", "none")
    kind = (task.get("task") or {}).get("kind", "part")
    given = (task.get("task") or {}).get("given", "")
    export_line = ("export(result, path) -> pred_graph.json (result is the graph dict)" if kind == "ecad"
                   else "export(result, path) -> STEP")
    lines = []
    if kind == "assembly":
        # The answer on an assembly task is the submission/ directory, so
        # the calls that write it come first; export() stays for the model's
        # own checks. use_part copies input/step_files/<part_id>.step, and a
        # task with given = nothing_3d (T4) has no such directory:
        # advertising it there is exactly the defect closed for
        # render/views.
        lines += ["  tools.py     export_part(geometry_or_step_path, part_id)",
                  "                                 -> submission/parts/<part_id>.step (one part TYPE)"]
        if given != "nothing_3d":
            lines += ["               use_part(part_id) -> copies the supplied step_files/<part_id>.step",
                      "                                    into submission/parts/, unchanged"]
        lines += ["               submit_assembly(instances[, assembly])",
                  "                                 -> submission/assembly/instances.json;",
                  "                                    instances = [{part_id, transform[, instance_id]}]",
                  "               export(geometry, path) -> STEP, for your own checks; not the answer"]
    else:
        lines += [f"  tools.py     {export_line}"]
    if renderer == "shared":
        lines += ["               render(step, png) -> isometric hidden-line view",
                  "               views(step, png)  -> four-orientation 2x2 sheet, same renderer",
                  "                                    and angles as the reference image"]
    has_sheets = any(str(i).endswith(".pdf") or "drawing" in str(i)
                     for i in (task.get("task") or {}).get("inputs", []))
    lines.append("               crop(png, box)    -> writes the box as a new PNG and returns its PATH\n"
                 "                                    (not an image); you see it next round")
    if has_sheets:
        lines.append("                                    (a drawing sheet is cut from a 2x-resolution master)")
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
                           submit_what=SUBMIT_WHAT[kind],
                           rounds=max_rounds, cq_version=SANDBOX_CQ_VERSION)
    answer = ANSWER[kind]
    # Images must be collected RECURSIVELY. T5's part drawings live in the
    # part_drawings/ subdirectory; the original scan was top-level only, so the
    # "all part drawings -> assembly" task never actually showed the model any
    # part drawing -- it received one assembly drawing and the reference views,
    # and the premise of the task disappeared. Subdirectory images are part of
    # the prompt too.
    seed_imgs = _seed_images(box.dir)
    turns = [{"role": "user", "text": "Begin.", "images": seed_imgs}]

    rounds, submitted = [], ""
    dead = 0
    for rnd in range(1, max_rounds + 1):
        # A failed call should waste THIS ROUND, not the whole case. Measured
        # in v2: PART-0043 / -0199 / -0211 all hit a stream interruption on
        # the first call (the provider never sent response.completed), the
        # exception propagated out, and the case scored 0 with none of its 30
        # rounds used -- that is the network's score, not the model's.
        # Only after MAX_DEAD_ROUNDS consecutive failures is it a real fault,
        # and only then is the exception allowed out.
        try:
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

    if not submitted and rounds:
        turns.append({"role": "user", "text":
                      f"Rounds are finished. Reply with only a ```submit block: {answer}.",
                      "images": []})
        py_f, sub = _blocks(call_fn(system, turns))
        # The forced-submit round must ALSO accept ```python. It originally
        # accepted only ```submit, so when a model returned its complete
        # program inside a ```python fence the fallback silently did nothing --
        # the case became a structural zero, and the log did not even record
        # submit_final, so it was invisible. Measured: minimax-m3 spent all
        # eight rounds iterating in exec and answered the final round with
        # exactly a ```python block. The program was there; we discarded it
        # over the tag. What is asked for is "the complete program for your
        # best geometry", and which tag it carries does not change that.
        if sub is None and py_f is not None:
            if _is_answer(py_f):
                sub = py_f
                rounds.append({"round": len(rounds) + 1,
                               "action": "submit_final_from_python"})
            else:
                rounds.append({"round": len(rounds) + 1, "action": "no_answer"})
        elif sub is not None:
            rounds.append({"round": len(rounds) + 1, "action": "submit_final"})
        elif py_f is None:
            rounds.append({"round": len(rounds) + 1, "action": "no_answer"})
        if sub is not None:
            submitted = sub

    # When the submitted program fails to execute, hand the error back and
    # allow one resubmission. Originally a crash ended the episode -- measured:
    # both hard zeros arose that way (once a nonexistent cadquery API, once a
    # fillet radius that overflowed into BRep_API not done), while the model's
    # understanding of the part had been adequate up to that point.
    # This is a separate small budget and does not consume a round: it is not
    # extra work for the model, it is showing the model its own error.
    step = None
    for attempt in range(2):
        if not submitted.strip():
            break
        res = box.run(submitted + FINALISE, timeout=exec_timeout)
        if res.ok:
            step = _artifact(box)
        if step is not None:
            break
        if attempt == 0:
            err = (res.stderr or "").strip()[-1500:] or f"exit {res.returncode}"
            turns.append({"role": "user", "text":
                          "Your submitted program failed to execute, so nothing "
                          f"was scored:\n\n{err}\n\nReply with one corrected "
                          "```submit block and nothing else. Keep the geometry "
                          "you had; fix only what the error names.", "images": []})
            py2, sub2 = _blocks(call_fn(system, turns))
            # The same trap as the forced-submit round: we ask for "one
            # corrected ```submit block", the model answers in ```python, sub2
            # is None, and the retry silently gives up. This path only runs
            # when the case is already in trouble, so it fails exactly when it
            # is most needed.
            if sub2 is None and _is_answer(py2):
                sub2 = py2
                rounds.append({"round": len(rounds) + 1,
                               "action": "resubmit_from_python"})
            elif sub2 is not None:
                rounds.append({"round": len(rounds) + 1, "action": "resubmit"})
            if sub2 is None:
                break
            submitted = sub2
    if step is None:
        # The same fallback as for a STEP, for the directory layout: the last
        # round that left a complete submission beats scoring zero.
        subs = [d for d in box.submissions() if is_ready(d)]
        step = subs[-1] if subs else None
    if step is None:
        cands = box.artifacts(".step")
        step = cands[-1] if cands else None

    out = {"code": submitted, "step": str(step) if step else None,
           "artifact": (None if step is None else
                        "submission_directory" if step.is_dir() else "step"),
           "rounds": rounds, "submitted": bool(submitted)}
    (box.log_dir / "episode.json").write_text(json.dumps(out, indent=1))
    return out
