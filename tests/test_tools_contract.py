"""The tools the sandbox stages behave as the prompt says (audit follow-ups)."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import types
from pathlib import Path

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
