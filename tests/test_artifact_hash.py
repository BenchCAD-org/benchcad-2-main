"""harness/artifact_hash.py and its check in tools/rescore.py: a record is judged again only on the
answer it was scored on. When two lanes shared an episode directory, one lane's final.step replaced the
other's, and three rescores in a row scored the replacement under the first lane's records."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.artifact_hash import Snapshots, artifact_sha256, check   # noqa: E402

CASE = REPO / "tests/fixtures/t3/case1"
A = CASE / "gt/gt.step"
B = REPO / "tests/fixtures/t1/case1/gt/gt.step"
OLD = {"score": 0.123, "metric": "part_v1"}


def _rescore_module():
    spec = importlib.util.spec_from_file_location("rescore_under_test", REPO / "tools/rescore.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _answer(tmp_path, src=A) -> Path:
    step = tmp_path / "r0__high__case1" / "final.step"
    step.parent.mkdir(exist_ok=True)
    shutil.copy(src, step)
    return step


def _record(step, **kw) -> dict:
    return {"case": str(CASE), "case_id": "case1", "model": "m", "effort": "high", "rep": 0,
            "step": str(step), "artifact": "step", "score": dict(OLD), "seconds_score": 1.0, **kw}


@pytest.fixture
def rescore(tmp_path, monkeypatch):
    """tools/rescore.py with a stand-in scorer: run(records, *args) -> (exit code, file, answers judged)."""
    mod = _rescore_module()
    judged = []

    def fake(case, step):
        judged.append(Path(step))
        return {"score": 0.777, "metric": "part_v1"}
    monkeypatch.setattr(mod, "score_in_subprocess", fake)

    def run(recs, *args):
        judged.clear()
        f = tmp_path / "r.json"
        f.write_text(json.dumps({"model": "m", "cases": recs}))
        code = mod.main([str(f), *args])
        return code, json.loads(f.read_text()), list(judged)
    return run


def test_a_swapped_answer_is_refused_and_its_score_kept(tmp_path, rescore):
    step = _answer(tmp_path)
    of, digest = artifact_sha256(step)
    rec = _record(step, artifact_sha256_of=of, artifact_sha256=digest)
    shutil.copy(B, step)                                  # another lane's answer lands on the same path
    code, out, judged = rescore([rec])
    assert code == 2 and judged == []
    r = out["cases"][0]
    assert r["score"] == OLD
    assert r["artifact_check"]["status"] == "mismatch"
    assert r["artifact_check"]["recorded"] == _sha(A) and r["artifact_check"]["on_disk"] == _sha(B)
    assert "rescored" not in out                          # the header does not claim a rescore that did not happen


def test_the_answer_the_record_names_is_judged_again(tmp_path, rescore):
    step = _answer(tmp_path)
    code, out, judged = rescore([_record(step, artifact_sha256_of="file", artifact_sha256=_sha(A))])
    assert code == 0 and judged == [step]
    r = out["cases"][0]
    assert r["score"]["score"] == 0.777
    assert r["artifact_check"]["status"] == "verified" and r["artifact_check"]["against"] == "record"
    assert out["rescored"]["at"]


def test_a_missing_answer_is_not_scored_as_no_submission(tmp_path, rescore):
    """score_case scores a path with nothing at it as "no submission" 0.0 -- over a real score, and
    for every record at once when a --map is wrong."""
    gone = tmp_path / "gone" / "final.step"
    for rec in (_record(gone, artifact_sha256_of="file", artifact_sha256=_sha(A)), _record(gone)):
        code, out, judged = rescore([rec], "--allow-unverified")
        assert code == 2 and judged == []
        assert out["cases"][0]["score"] == OLD and out["cases"][0]["artifact_check"]["status"] == "missing"


def test_a_record_from_before_the_field_is_checked_against_a_snapshot(tmp_path, rescore):
    step = _answer(tmp_path)
    good, bad = tmp_path / "good.manifest.json", tmp_path / "bad.manifest.json"
    good.write_text(json.dumps({"files": {str(step): _sha(A)}}))
    bad.write_text(json.dumps({"files": {str(step): _sha(B)}}))

    code, out, judged = rescore([_record(step)], "--snapshot", str(good))
    assert code == 0 and judged == [step]
    assert out["cases"][0]["artifact_check"]["status"] == "verified"
    assert out["cases"][0]["artifact_check"]["against"] == "good.manifest.json"

    code, out, judged = rescore([_record(step)], "--snapshot", str(bad))
    assert code == 2 and judged == [] and out["cases"][0]["score"] == OLD
    assert out["cases"][0]["artifact_check"]["status"] == "mismatch"

    # a path two snapshots list with different answers held two answers: nothing says which was scored
    code, out, judged = rescore([_record(step)], "--snapshot", str(good), "--snapshot", str(bad))
    assert code == 2 and judged == []
    assert "disagree" in out["cases"][0]["artifact_check"]["why"]


def test_a_snapshot_is_looked_up_under_the_records_own_path(tmp_path, rescore):
    """A file made on another machine: the snapshot lists the answer where the record says it was;
    --map says where it is here."""
    step = _answer(tmp_path)
    there = "/elsewhere/work/r0__high__case1/final.step"
    m = tmp_path / "there.manifest.json"
    m.write_text(json.dumps({"files": {there: _sha(A)}}))
    code, out, judged = rescore([_record(there)], "--snapshot", str(m), "--map", f"/elsewhere/work={tmp_path}")
    assert code == 0 and judged == [step]
    assert out["cases"][0]["step"] == there                 # the record keeps its own path


def test_an_unverifiable_record_is_skipped_unless_allowed(tmp_path, rescore):
    step = _answer(tmp_path)
    code, out, judged = rescore([_record(step)])
    assert code == 2 and judged == []
    assert out["cases"][0] == _record(step)                 # untouched: nothing is wrong with it, nothing vouches for it

    code, out, judged = rescore([_record(step)], "--allow-unverified")
    assert code == 0 and judged == [step]
    assert out["cases"][0]["score"]["score"] == 0.777
    assert out["cases"][0]["artifact_check"]["status"] == "unverified"


def test_a_submission_directory_is_hashed_as_a_tree(tmp_path):
    sub = tmp_path / "submission"
    (sub / "parts").mkdir(parents=True)
    (sub / "assembly").mkdir()
    shutil.copy(A, sub / "parts/part_01.step")
    shutil.copy(B, sub / "parts/part_02.step")
    (sub / "assembly/instances.json").write_text("[]")
    of, digest = artifact_sha256(sub)
    assert of == "tree"

    # the same digest from a manifest that lists the files, and not a sibling whose name shares the prefix
    files = {str(f): _sha(f) for f in sub.rglob("*") if f.is_file()}
    files[str(tmp_path / "submission_old/parts/part_01.step")] = _sha(B)
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"files": files}))
    ref = Snapshots([m]).reference(str(sub))
    assert ref["of"] == "tree" and ref["sha256"] == digest

    rec = {"step": str(sub), "artifact_sha256_of": of, "artifact_sha256": digest}
    assert check(rec, sub)["status"] == "verified"
    shutil.copy(A, sub / "parts/part_02.step")              # one part of another answer
    assert check(rec, sub)["status"] == "mismatch"
    assert check({"step": str(sub)}, sub, Snapshots([m]))["status"] == "mismatch"


def test_files_the_scorer_does_not_read_leave_the_digest_alone(tmp_path):
    """A .DS_Store from Finder, a __pycache__, an editor's backup or the never-scored assembly.step
    must not make an unchanged answer read as another one; a changed instances.json must."""
    sub = tmp_path / "submission"
    (sub / "parts").mkdir(parents=True)
    (sub / "assembly").mkdir()
    shutil.copy(A, sub / "parts/part_01.step")
    (sub / "assembly/instances.json").write_text('{"instances": []}')
    before = artifact_sha256(sub)
    rec = {"step": str(sub), "artifact_sha256_of": before[0], "artifact_sha256": before[1]}

    (sub / ".DS_Store").write_bytes(b"\0Bud1")
    (sub / "parts/.DS_Store").write_bytes(b"\0Bud1")
    (sub / "__pycache__").mkdir()
    (sub / "__pycache__/tools.cpython-312.pyc").write_bytes(b"\x00")
    (sub / "parts/part_01.step~").write_bytes(b"old")
    (sub / "parts/notes.txt").write_text("not a STEP file: the parser ignores it")
    (sub / "Thumbs.db").write_bytes(b"\x00")
    shutil.copy(B, sub / "assembly/assembly.step")         # optional, for a human reader, never scored
    assert artifact_sha256(sub) == before
    assert check(rec, sub)["status"] == "verified"
    m = tmp_path / "m.json"                                  # a snapshot that also listed the junk
    m.write_text(json.dumps({"files": {str(f): _sha(f) for f in sub.rglob("*") if f.is_file()}}))
    assert Snapshots([m]).reference(str(sub))["sha256"] == before[1]

    (sub / "assembly/instances.json").write_text('{"instances": [{"part_id": "part_01"}]}')
    assert check(rec, sub)["status"] == "mismatch"
