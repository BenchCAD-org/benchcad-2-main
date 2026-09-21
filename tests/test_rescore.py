"""tools/rescore.py re-judges a results file's records with the current
scorer and keeps everything else."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_rescore_replaces_the_score_and_keeps_the_record(tmp_path):
    case = REPO / "tests/fixtures/t2/case1"
    rec = {"case": str(case), "case_id": "case1", "model": "m", "effort": "medium",
           "step": str(case / "gt/gt.step"), "artifact": "step", "seconds": 1.0,
           "tokens": {"input": 1, "cached": 0, "output": 1, "calls": 1},
           "score": {"score": 0.123, "metric": "asm_v1"}, "seconds_score": 0.0}
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
                "score": None, "unscored": True, "seconds_score": 0.0}
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
