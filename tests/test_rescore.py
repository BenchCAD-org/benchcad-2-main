"""tools/rescore.py re-judges a results file's records with the current
scorer and keeps everything else."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _answer_sha(step: Path) -> dict:
    """The record's answer digest, as harness/run.py writes it (rescore judges only that answer)."""
    return {"artifact_sha256_of": "file", "artifact_sha256": hashlib.sha256(step.read_bytes()).hexdigest()}


def test_rescore_replaces_the_score_and_keeps_the_record(tmp_path):
    case = REPO / "tests/fixtures/t2/case1"
    rec = {"case": str(case), "case_id": "case1", "model": "m", "effort": "medium",
           "step": str(case / "gt/gt.step"), "artifact": "step", "seconds": 1.0,
           "tokens": {"input": 1, "cached": 0, "output": 1, "calls": 1},
           "score": {"score": 0.123, "metric": "asm_v1"}, "seconds_score": 0.0,
           **_answer_sha(case / "gt/gt.step")}
    skipped = {"case": str(case), "case_id": "case1b", "step": None, "artifact": None, "score": None}
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"model": "m", "cases": [rec, skipped]}))
    p = subprocess.run([sys.executable, str(REPO / "tools/rescore.py"), str(f), "--only", "t2"],
                       capture_output=True, text=True, timeout=900, cwd=REPO, check=False)
    assert p.returncode == 0, p.stderr[-800:]
    assert "0.123 -> 1.000" in p.stdout, p.stdout
    out = json.loads(f.read_text())
    assert out["rescored"]["at"]
    new = out["cases"][0]
    assert abs(new["score"]["score"] - 1.0) < 1e-4 and new["score"]["metric"] == "asm_v1"
    assert new["tokens"] == rec["tokens"] and new["seconds"] == 1.0   # the episode is untouched
    assert out["cases"][1] == skipped                                  # no artifact: left alone
    assert json.loads((tmp_path / "r.before-rescore.json").read_text())["cases"][0]["score"]["score"] == 0.123


def test_rescore_maps_paths_and_only_unscored(tmp_path):
    """A file made on another machine: --map rewrites the case and step prefixes;
    --only-unscored leaves scored records alone."""
    fake = "/elsewhere/bank"
    unscored = {"case": f"{fake}/t2/case1", "case_id": "case1", "step": f"{fake}/t2/case1/gt/gt.step",
                "artifact": "step", "seconds": 1.0, "tokens": {"input": 1, "cached": 0, "output": 1, "calls": 1},
                "score": None, "unscored": True, "seconds_score": 0.0,
                **_answer_sha(REPO / "tests/fixtures/t2/case1/gt/gt.step")}
    scored = {**unscored, "case_id": "case1_already", "score": {"score": 0.5, "metric": "asm_v1"}, "unscored": False}
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"model": "m", "cases": [unscored, scored]}))
    p = subprocess.run([sys.executable, str(REPO / "tools/rescore.py"), str(f), "--only-unscored",
                        "--map", f"{fake}/t2=" + str(REPO / "tests/fixtures/t2")],
                       capture_output=True, text=True, timeout=900, cwd=REPO, check=False)
    assert p.returncode == 0, p.stderr[-800:]
    out = json.loads(f.read_text())["cases"]
    assert "unscored" not in out[0] and abs(out[0]["score"]["score"] - 1.0) < 1e-4
    assert out[0]["case"].startswith(fake)                      # the record keeps its own paths
    assert out[1]["score"]["score"] == 0.5                       # untouched


def test_rescore_keeps_a_1_0_t6_answer_on_the_legacy_path_once_the_case_is_position(tmp_path):
    """Once a T6 case carries gt/correspondence.json (position mode), an old
    record's pcb2schematic/1.0 answer is re-scored on the legacy path it was
    first scored on -- labelled forced_mode historical, correspondence_mode
    legacy, the same number the legacy scorer gives -- never in position mode,
    where it would become invalid_prediction 0. A 2.0-position answer in the
    same file is scored in position mode."""
    import shutil
    sys.path.insert(0, str(REPO))
    from envs.verifiers.ecad import position_reference, score
    case = tmp_path / "t6_pcb2schematic" / "case1"
    shutil.copytree(REPO / "examples/task6/cases/case1", case)
    old = json.loads((case / "gt/gt_graph.json").read_text())
    gone = set(old["components"].pop()["terminals"])
    old["incidences"] = [i for i in old["incidences"] if i[0] not in gone]
    old_sub = tmp_path / "old" / "pred_graph.json"
    old_sub.parent.mkdir()
    old_sub.write_text(json.dumps(old))
    legacy_number = score(case, old_sub)["score"]            # before activation: the historical number
    assert 0.0 < legacy_number < 1.0
    for f in ("correspondence.json", "spatial_reference.json"):
        shutil.copyfile(REPO / "tests/fixtures/t6_position/case1/gt" / f, case / "gt" / f)
    assert score(case, old_sub)["status"] == "invalid_prediction"      # what a plain re-score would do
    new_sub = tmp_path / "new" / "pred_graph.json"
    new_sub.parent.mkdir()
    new_sub.write_text(json.dumps(position_reference(case)))
    base = {"case": str(case), "model": "m", "artifact": "graph", "seconds": 1.0,
            "tokens": {"input": 1, "cached": 0, "output": 1, "calls": 1}, "seconds_score": 0.0}
    recs = [{**base, "case_id": "case1_old", "step": str(old_sub), **_answer_sha(old_sub),
             "score": {"score": legacy_number, "metric": "ecad_v2"}},
            {**base, "case_id": "case1_new", "step": str(new_sub), **_answer_sha(new_sub),
             "score": {"score": 0.0, "metric": "ecad_v2"}}]
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"model": "m", "cases": recs}))
    p = subprocess.run([sys.executable, str(REPO / "tools/rescore.py"), str(f)],
                       capture_output=True, text=True, timeout=600, cwd=REPO, check=False)
    assert p.returncode == 0, p.stderr[-800:]
    got = {r["case_id"]: r["score"] for r in json.loads(f.read_text())["cases"]}
    o, n = got["case1_old"], got["case1_new"]
    assert o["forced_mode"] == "historical" and o["correspondence_mode"] == "legacy", o
    assert o["status"] == "ok" and abs(o["score"] - legacy_number) < 1e-9
    assert n["correspondence_mode"] == "position" and n["status"] == "ok" and abs(n["score"] - 1.0) < 1e-9
    assert "forced_mode" not in n
