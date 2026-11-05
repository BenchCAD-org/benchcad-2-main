"""The harness never follows a link out of the model's working directory.

The model writes its own files in /work, so anything it leaves there is its own: a link is not. Following
one would let a program be scored on a file it only pointed at (the reference next to the case), have the
host write its code or copy its logs outside /work, or make a copy run away down a device. Each site that
reads the working directory is checked here: the submission layout, the artifact the episode scores, the
answer's hash, and the sandbox's snapshot, round log and program file.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.common import sandbox as sb                                     # noqa: E402
from envs.common.submission import is_ready, is_submission, locate        # noqa: E402
from harness.artifact_hash import artifact_sha256                         # noqa: E402


def _reference(tmp: Path) -> Path:
    """A reference answer outside the working directory, as a case's gt/ is."""
    gt = tmp / "gt"
    gt.mkdir()
    (gt / "gt.step").write_text("ISO-10303-21; the reference\n")
    (gt / "instances.json").write_text('{"instances": []}')
    return gt


def _submission(work: Path) -> Path:
    sub = work / "submission"
    (sub / "parts").mkdir(parents=True)
    (sub / "assembly").mkdir()
    return sub


def test_a_submission_of_links_is_not_a_submission(tmp_path):
    gt = _reference(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    sub = _submission(work)
    (sub / "parts/part_01.step").symlink_to(gt / "gt.step")
    (sub / "assembly/instances.json").symlink_to(gt / "instances.json")
    assert not is_ready(sub) and not is_ready(work)
    # the same layout, written rather than pointed at, is a submission
    (sub / "parts/part_01.step").unlink()
    (sub / "assembly/instances.json").unlink()
    (sub / "parts/part_01.step").write_text("a part the program wrote\n")
    (sub / "assembly/instances.json").write_text('{"instances": []}')
    assert is_ready(sub) and is_ready(work)


def test_a_submission_directory_that_is_a_link_is_not_found(tmp_path):
    gt = _reference(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (gt / "parts").mkdir()
    (gt / "parts/part_01.step").write_text("the reference part\n")
    (gt / "assembly").mkdir()
    (gt / "assembly/instances.json").write_text('{"instances": []}')
    (work / "submission").symlink_to(gt)
    assert locate(work) is None and not is_submission(work) and not is_ready(work)
    # nor one whose parts/ alone is a link
    (work / "submission").unlink()
    sub = _submission(work)
    (sub / "assembly/instances.json").write_text('{"instances": []}')
    (sub / "parts").rmdir()
    (sub / "parts").symlink_to(gt / "parts")
    assert not is_ready(sub)


def test_the_episode_scores_no_link_left_as_final_step(tmp_path):
    from envs.common.episode import _artifact

    class Box:
        pass
    gt = _reference(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    box = Box()
    box.dir = work
    assert _artifact(box) is None
    (work / "final.step").symlink_to(gt / "gt.step")
    assert _artifact(box) is None, "a link named final.step is not an answer"
    (work / "final.step").unlink()
    (work / "final.step").write_text("ISO-10303-21; what the program exported\n")
    assert _artifact(box) == work / "final.step"
    # a graph file, the same way
    (work / "final.step").unlink()
    (work / "pred_graph.json").symlink_to(gt / "instances.json")
    assert _artifact(box) is None


def test_an_answers_hash_is_of_what_the_program_wrote(tmp_path):
    gt = _reference(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / "final.step").symlink_to(gt / "gt.step")
    assert artifact_sha256(work / "final.step") is None
    sub = _submission(work)
    (sub / "assembly/instances.json").write_text('{"instances": []}')
    (sub / "parts/real.step").write_text("a part\n")
    (sub / "parts/linked.step").symlink_to(gt / "gt.step")
    kind, with_link = artifact_sha256(sub)
    (sub / "parts/linked.step").unlink()
    assert kind == "tree" and artifact_sha256(sub) == (kind, with_link), "the link counted for nothing"


def test_the_sandbox_shows_and_logs_no_link_the_program_left(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 4000)
    work = tmp_path / "work"
    work.mkdir()
    (work / "linked.png").symlink_to(outside / "secret.png")
    (work / "own.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"1" * 4000)
    box = sb.Sandbox.__new__(sb.Sandbox)                       # no case, no docker: only the file handling
    box.dir = work
    box.log_dir = tmp_path / "logs"
    box.log_dir.mkdir()
    box._round = 0
    assert set(box._snapshot()) == {"own.png"}, "a link is not one of the program's figures"
    # the round log copies what the program wrote, and nothing a link names
    (work / "linked.step").symlink_to(outside / "secret.png")
    (work / "own.step").write_text("ISO-10303-21;\n")
    res = sb.Result(0, "", "", [work / "own.png", work / "linked.png"])
    kept = box._log("print(1)", res)
    got = sorted(p.name for p in (box.log_dir / "round_01").iterdir())
    assert "linked.png" not in got and "linked.step" not in got
    assert "own.png" in got and "own.step" in got
    assert all(p.name != "linked.png" for p in kept), "a link is never handed back to the model"


def test_the_sandbox_writes_the_program_to_a_file_it_makes(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("the host's own file\n")
    work = tmp_path / "work"
    work.mkdir()
    (work / "_run.py").symlink_to(outside)
    box = sb.Sandbox.__new__(sb.Sandbox)
    box.dir = work
    box.log_dir = tmp_path / "logs"
    box.log_dir.mkdir()
    box._round = 0
    box._seen = {}
    box.docker = False
    box.run("print('the model's code')".replace("'s", "s"), timeout=30)
    assert outside.read_text() == "the host's own file\n", "the link was written through"
    assert not (work / "_run.py").is_symlink() and "print(" in (work / "_run.py").read_text()


def test_the_program_output_the_model_sees_is_bounded(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    box = sb.Sandbox.__new__(sb.Sandbox)
    box.dir = work
    box.log_dir = tmp_path / "logs"
    box.log_dir.mkdir()
    box._round = 0
    box._seen = {}
    box.docker = False
    n = sb.TAIL_BYTES // 1000 + 50
    res = box.run(f"print('x' * 999, flush=False)\nfor _ in range({n}): print('y' * 999)", timeout=120)
    assert len(res.stdout) <= sb.TAIL_BYTES and res.stdout.endswith("y" * 999 + "\n")
    assert res.returncode == 0
