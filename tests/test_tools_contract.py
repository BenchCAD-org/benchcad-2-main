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
