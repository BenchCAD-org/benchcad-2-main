"""The tools the sandbox stages behave as the prompt says (audit follow-ups)."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.common.sandbox import TOOLS_PY  # noqa: E402


def _run(tmp_path: Path, code: str):
    (tmp_path / "tools.py").write_text(TOOLS_PY)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=tmp_path,
                          capture_output=True, text=True)


def test_export_writes_every_object_on_the_stack(tmp_path):
    r = _run(tmp_path, '''
        import cadquery as cq, tools
        w = cq.Workplane("XY").box(10, 10, 10).add(cq.Workplane("XY").box(5, 5, 5).translate((20, 0, 0)))
        p = tools.export(w, "two.step")
        print(len(cq.importers.importStep(str(p)).solids().vals()))
    ''')
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "2"


def test_submit_assembly_refuses_a_non_rigid_transform(tmp_path):
    r = _run(tmp_path, '''
        import cadquery as cq, tools
        tools.export_part(cq.Workplane("XY").box(4, 4, 4), "a")
        for T in ([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]],          # scale
                  [[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]):        # mirror
            try:
                tools.submit_assembly([{"part_id": "a", "transform": T}]); print("accepted")
            except ValueError as e:
                print("refused:", "not rigid" in str(e))
        tools.submit_assembly([{"part_id": "a", "transform": [[0,-1,0,5],[1,0,0,0],[0,0,1,0],[0,0,0,1]]}])
        print("rigid ok")
    ''')
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("\n")[:3] == ["refused: True", "refused: True", "rigid ok"]
    assert (tmp_path / "submission/assembly/instances.json").exists()


def test_artifact_finds_an_ecad_graph(tmp_path):
    from envs.common.episode import _artifact
    box = types.SimpleNamespace(dir=tmp_path)
    assert _artifact(box) is None
    (tmp_path / "pred_graph.json").write_text("{}")
    assert _artifact(box) == tmp_path / "pred_graph.json"
    (tmp_path / "final.step").write_text("")
    assert _artifact(box) == tmp_path / "final.step"


def test_seed_image_labels_are_paths_in_the_working_directory():
    from harness.run import bound_images, image_label, labelled
    turns = [{"role": "user", "text": "Begin.", "images": ["/w/part_drawings/part_03.png", "/w/drawing.png"],
              "image_labels": ["part_drawings/part_03.png", "drawing.png"]}]
    out, _ = bound_images(turns)
    assert [image_label(p, l) for p, l in labelled(out[0])] == ["[part_drawings/part_03.png]", "[drawing.png]"]
    # an observation image (no label given) is named by its file
    assert image_label("/log/round_03/crop_drawing.png") == "[crop_drawing.png]"


def test_export_refuses_a_malformed_graph_with_the_reason(tmp_path):
    r = _run(tmp_path, '''
        import tools
        good = {"schema": "pcb2schematic/1.0",
                "components": [{"id": "C1", "type": "capacitor", "terminals": ["C1.1", "C1.2"]}],
                "nets": [{"id": "GND"}, {"id": "N1"}],
                "incidences": [["C1.1", "N1"], ["C1.2", "GND"]]}
        print(tools.export(good, "pred_graph.json").name)
        bad = dict(good, incidences=[["C1.1", "POWER.1"], ["C1.2", "GND"], ["C1.2", "N1"], ["R9.1", "GND"]])
        try:
            tools.export(bad, "pred_graph.json"); print("accepted")
        except ValueError as e:
            msg = str(e); print("refused:", "POWER.1" in msg, "two nets" in msg, "R9.1" in msg)
    ''')
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("\n")[:2] == ["pred_graph.json", "refused: True True True"]


def test_exec_gate_bounds_concurrent_sandbox_runs(tmp_path, monkeypatch):
    """CADENV_MAX_EXECS=1: two episodes' executions cannot overlap, however
    many threads are in flight. Measured by wall time: two 1 s scripts take
    >= 2 s under the gate, and without it clearly less than that (interpreter
    start-up is paid by both, so only the difference is asserted)."""
    import threading
    import time
    from envs.common import sandbox as S
    monkeypatch.setenv("CADENV_LOCAL", "1")
    monkeypatch.setattr(S, "_docker_ready", lambda: False)
    case = ROOT / "tests/fixtures/t3/case1"
    boxes = [S.Sandbox(case, tmp_path / f"w{i}") for i in range(2)]
    code = "import time; time.sleep(1.0); print('done')"

    def timed(gate):
        monkeypatch.setattr(S, "EXEC_GATE", gate)
        t0 = time.time()
        ts = [threading.Thread(target=b.run, args=(code,), kwargs={"timeout": 30}) for b in boxes]
        for t in ts: t.start()
        for t in ts: t.join()
        return time.time() - t0

    monkeypatch.setenv("CADENV_MAX_EXECS", "1")
    assert S._exec_gate() is not None
    gated, free = timed(S._exec_gate()), timed(None)
    assert gated >= 2.0, gated
    assert free <= gated - 0.6, (free, gated)



def test_a_tile_the_model_copies_or_touches_at_the_top_level_is_attached(tmp_path, monkeypatch):
    """No tool for looking at a tile: the round attaches the PNGs the model
    wrote or copied at the top level, and a copied tile -- or a staged one
    it touches -- is exactly that."""
    from envs.common import sandbox as S
    monkeypatch.setenv("CADENV_LOCAL", "1")
    monkeypatch.setattr(S, "_docker_ready", lambda: False)
    case = ROOT / "tests/fixtures/t3/case1"
    wd = tmp_path / "w"
    box = S.Sandbox(case, wd)
    (wd / "sub").mkdir()
    (wd / "sub" / "drawing_tile_r1c2.png").write_bytes((wd / "views.png").read_bytes())
    r = box.run("import shutil, pathlib\nshutil.copy('sub/drawing_tile_r1c2.png', 'r1c2.png')\n"
                "pathlib.Path('views.png').touch()\n", timeout=60)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in r.images) == ["r1c2.png", "views.png"]


def test_context_summarization_replaces_the_history_the_terminus_way(tmp_path, monkeypatch):
    """When the last request left fewer than SUMMARIZE_FREE_TOKENS free, the
    episode makes Terminus 2's three calls (summary, questions, answers) and
    continues from [seed, questions prompt, questions, answers + continue];
    the seed images stay, the summarised rounds' images do not, and the
    record says `summarize`."""
    from envs.common import episode as E
    from envs.common import sandbox as S
    monkeypatch.setenv("CADENV_LOCAL", "1")
    monkeypatch.setattr(S, "_docker_ready", lambda: False)
    case = ROOT / "tests/fixtures/t3/case1"
    calls: list[dict] = []
    usage: list[dict] = []

    def call_fn(system, turns, plain=False):
        calls.append({"plain": plain, "n_turns": len(turns), "images": sum(len(t.get("images") or []) for t in turns),
                      "last": turns[-1]["text"][:40]})
        if plain:
            return {"You are about to hand off": "SUMMARY-TEXT", "picking up work": "Q1? Q2? Q3? Q4? Q5?",
                    "few questions for you": "A1. A2. A3. A4. A5."}[
                next(k for k in ("You are about to hand off", "picking up work", "few questions for you") if k in turns[-1]["text"])]
        # round 1: a program that writes a PNG; round 2: submit
        if len([t for t in turns if t.get("role") == "assistant"]) == 0:
            usage.append({"input_tokens": 195_000, "output_tokens": 10})     # 5,000 free of 200k -> summarise before round 2
            return "```python\nfrom PIL import Image\nImage.new('RGB', (8, 8), 'red').save('look.png')\nprint('ok')\n```"
        usage.append({"input_tokens": 3_000, "output_tokens": 10})
        return "```submit\nimport cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)\n```"
    call_fn.usage = usage
    call_fn.context_tokens = 200_000
    out = E.run_episode(case, tmp_path / "w", call_fn, max_rounds=5, exec_timeout=60)
    kinds = [c["plain"] for c in calls]
    assert kinds == [False, True, True, True, False], calls
    # the second real call sees the restarted conversation: seed + q-prompt + questions + handoff
    last = calls[-1]
    assert last["n_turns"] == 4 and last["images"] == 1 and "Here are the answers" in last["last"]
    assert [r["action"] for r in out["rounds"]] == ["exec", "summarize", "submit"]
    assert out["submitted"] and (tmp_path / "w_log" / "summary_01.json").exists()


def test_a_reply_withheld_round_after_round_ends_with_no_submission(tmp_path, monkeypatch):
    """harness.run.ReplyBlocked: the provider refused the conversation on
    policy. MAX_DEAD_ROUNDS of it in a row end the episode with no submission,
    a final record scored 0 -- not an exception, which leaves the case an
    error that --resume re-runs and Core counts as pending, forever. Any other
    failure still ends the case as a fault."""
    from envs.common import episode as E
    from envs.common import sandbox as S
    from harness.run import ReplyBlocked
    monkeypatch.setenv("CADENV_LOCAL", "1")
    monkeypatch.setattr(S, "_docker_ready", lambda: False)
    case = ROOT / "tests/fixtures/t3/case1"

    def blocked(system, turns, plain=False):
        raise ReplyBlocked("gemini withheld the reply: finish_reason=SAFETY")
    blocked.usage = []
    blocked.context_tokens = 200_000
    out = E.run_episode(case, tmp_path / "w", blocked, max_rounds=10, exec_timeout=60)
    assert [r["action"] for r in out["rounds"]] == ["call_failed"] * E.MAX_DEAD_ROUNDS + ["no_submission"]
    assert not out["submitted"] and out["step"] is None

    def down(system, turns, plain=False):
        raise ConnectionError("Connection error.")
    down.usage = []
    down.context_tokens = 200_000
    with pytest.raises(ConnectionError):
        E.run_episode(case, tmp_path / "w2", down, max_rounds=10, exec_timeout=60)


def test_an_over_budget_call_is_a_discarded_round_the_model_hears_about(tmp_path, monkeypatch):
    """harness.run.CallOverBudget: the model was still answering when the
    call's wall budget ran out. The round is discarded (counted like a failed
    call) and the next request carries one sentence saying so, since the
    same context would draw the same reasoning again."""
    from envs.common import episode as E
    from envs.common import sandbox as S
    from harness.run import CallOverBudget
    monkeypatch.setenv("CADENV_LOCAL", "1")
    monkeypatch.setattr(S, "_docker_ready", lambda: False)
    case = ROOT / "tests/fixtures/t3/case1"
    seen = []

    def call_fn(system, turns, plain=False):
        seen.append(turns[-1]["text"])
        if len(seen) == 1:
            raise CallOverBudget("responses call still streaming after 300 s; giving it up")
        return "```submit\nimport cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)\n```"
    call_fn.usage = []
    call_fn.context_tokens = 200_000
    out = E.run_episode(case, tmp_path / "w", call_fn, max_rounds=5, exec_timeout=60)
    assert [r["action"] for r in out["rounds"]] == ["call_failed", "submit"]
    assert "CallOverBudget" in out["rounds"][0]["error"]
    assert seen[1].startswith("Round 1: your reply did not finish within the time budget") and "Round 2 of 5" in seen[1]
    assert out["submitted"]


def test_the_default_sandbox_image_is_the_tag_run_core_builds_here(monkeypatch):
    """tools/run_core.sh builds benchcad-sandbox:arm64 on arm64/aarch64 and :amd64
    elsewhere; a direct harness/run.py looks for the same tag (it looked for
    :arm64 on x86 too). CADENV_DOCKER_IMAGE still wins."""
    import os
    import platform
    from envs.common import sandbox as S
    for machine, tag in (("arm64", "arm64"), ("aarch64", "arm64"), ("x86_64", "amd64"), ("AMD64", "amd64")):
        monkeypatch.setattr(platform, "machine", lambda m=machine: m)
        assert S._default_image() == f"benchcad-sandbox:{tag}"
    probe = [sys.executable, "-c", "import envs.common.sandbox as S; print(S.DOCKER_IMAGE)"]
    env = {k: v for k, v in os.environ.items() if k != "CADENV_DOCKER_IMAGE"}
    assert subprocess.run(probe, cwd=ROOT, env={**env, "CADENV_DOCKER_IMAGE": "mine:1"},
                          capture_output=True, text=True).stdout.strip() == "mine:1"


def _counting_episode(tmp_path, monkeypatch, script, max_rounds):
    """Run the t3 fixture with a call_fn that plays `script`, one entry per call:
    an exception to raise, or a reply. Returns (episode result, the text of the
    last turn each call was sent)."""
    from envs.common import episode as E
    from envs.common import sandbox as S
    monkeypatch.setenv("CADENV_LOCAL", "1")
    monkeypatch.setattr(S, "_docker_ready", lambda: False)
    seen = []

    def call_fn(system, turns, plain=False):
        seen.append(turns[-1]["text"])
        step = script[min(len(seen), len(script)) - 1]
        if isinstance(step, Exception):
            raise step
        return step
    call_fn.usage = []
    call_fn.context_tokens = 200_000
    out = E.run_episode(ROOT / "tests/fixtures/t3/case1", tmp_path / "w", call_fn,
                        max_rounds=max_rounds, exec_timeout=60)
    return out, seen


PRINT = "```python\nprint('ok')\n```"


def test_an_infrastructure_failure_does_not_spend_the_models_round(tmp_path, monkeypatch):
    """GDM's review (2026-10-12): a call that fails past drive()'s retries for
    the infrastructure's reasons is asked again for the same round, so the
    model still plays every one of its rounds."""
    from envs.common import episode as E
    out, seen = _counting_episode(tmp_path, monkeypatch,
                                  [ConnectionError("Connection error."), ConnectionError("Connection error."), PRINT], 30)
    assert [r["action"] for r in out["rounds"]].count("exec") == 30
    assert [r["round"] for r in out["rounds"] if r["action"] == "exec"] == list(range(1, 31))
    assert [r.get("infra") for r in out["rounds"][:2]] == [True, True] and len(seen) == 32
    assert out["rounds"][-1] == {"round": 31, "action": "no_submission"}     # numbered by the model's rounds
    # ... within a total per episode, so a flapping endpoint still ends the case
    flap = [ConnectionError("Connection error."), PRINT] * 20
    with pytest.raises(ConnectionError):
        _counting_episode(tmp_path / "flap", monkeypatch, flap, 30)
    assert E.MAX_INFRA_FAILURES < 20



def test_a_withheld_reply_still_spends_the_round(tmp_path, monkeypatch):
    """ReplyBlocked keeps its semantics: the provider's verdict on the model's
    conversation spends the round (and MAX_DEAD_ROUNDS in a row end the episode
    with no submission, test_a_reply_withheld_round_after_round_...). So do
    an over-budget reply and a context that no longer fits; the rest is the
    infrastructure's."""
    from envs.common import episode as E
    from harness.run import CallOverBudget, ReplyBlocked
    blocked = ReplyBlocked("gemini withheld the reply: finish_reason=SAFETY")
    out, seen = _counting_episode(tmp_path, monkeypatch, [blocked, blocked, PRINT], 4)
    assert [r["action"] for r in out["rounds"]] == ["call_failed", "call_failed", "exec", "exec", "no_submission"]
    assert not any(r.get("infra") for r in out["rounds"]) and len(seen) == 4
    assert E._models_round(blocked) and E._models_round(CallOverBudget("still streaming after 1800 s"))
    assert E._models_round(RuntimeError("Error code: 400 - prompt is too long: 1,050,000 tokens"))
    assert not any(E._models_round(e) for e in (ConnectionError("Connection error."), TimeoutError("idle"),
                                                 RuntimeError("Error code: 503 - overloaded")))
