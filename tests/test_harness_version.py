"""The harness version (README, Versions): harness/run.py stamps it in the
run's file and every record, and core_score carries it into the summary and
its lines. Records from more than one version are not a reportable score."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import core_score as C                                            # noqa: E402
from harness import run as R                                      # noqa: E402

CASE = ROOT / "tests/fixtures/t3/case1"


def test_the_version_is_2_1_beta():
    assert R.HARNESS_VERSION == "2.1-beta"


def test_the_runner_stamps_the_file_and_every_record_and_the_summary_carries_it(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "harness/run.py"), "--model", "mock/dumb",
                        "--cases", str(CASE), "--unsafe-local",
                        "--out", str(tmp_path / "o.json"), "--work", str(tmp_path / "w")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["harness_version"] == "2.1-beta"
    assert [c["harness_version"] for c in out["cases"]] == ["2.1-beta"]
    row = C.summarize([tmp_path / "o.json"])[0]
    assert row["harness_version"] == "2.1-beta" and not row["smoke"]
    assert "harness 2.1-beta" in C.report([row])
    C.main([str(tmp_path / "o.json"), "--json", str(tmp_path / "core_summary.json")])
    summary = json.loads((tmp_path / "core_summary.json").read_text())
    assert summary["harness_versions"] == {"2.1-beta": 1}
    assert summary["efforts"][0]["harness_version"] == "2.1-beta"


def rec(case, version=None):
    r = {"case": f"t3/{case}", "case_id": case, "model": "anthropic/m", "effort": "max",
         "rounds": 40, "token_budget": 500_000, "score": {"score": 0.5}}
    if version:
        r["harness_version"] = version
    return r


def test_mixed_versions_are_a_smoke_run_and_unstamped_records_are_2_0(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"cases": [rec("a", "2.1-beta"), rec("b")]}))
    row = C.summarize([f])[0]
    assert row["harness_version"] == ["2.0", "2.1-beta"] and row["smoke"]
    assert "harness versions 2.0 + 2.1-beta mixed" in C.report([row])
    assert C.record_versions([f]) == {"2.1-beta": 1, "2.0": 1}


def test_the_reference_lines_are_2_0_and_the_comparison_says_so(tmp_path):
    ref = json.loads((ROOT / "results/core_reference.json").read_text())
    assert ref["harness_version"] == "2.0"
    assert {line["harness_version"] for line in ref["lines"]} == {"2.0"}
    line = ref["lines"][0]
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"cases": [dict(rec("a", "2.1-beta"), model=line["model"],
                                            effort=line["effort"])]}))
    text = C.report(C.summarize([f]), reference=ref)
    assert "harness 2.1-beta" in text and "; harness 2.0, 30 rounds," in text
