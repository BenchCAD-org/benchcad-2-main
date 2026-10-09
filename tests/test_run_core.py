"""tools/run_core.sh --dry-run and tools/core_score.py: the one-command path, without a model or a download.

A fake Core tree is built from the test fixtures, with the dataset.json and
MANIFEST.sha256 the real package carries, and the script is run in dry-run
mode: every check (manifest, dataset version, scorer seal, effort, key) runs,
and the harness commands it would run are printed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.common.ecad_graph.spatial_reference import scorer_digest      # noqa: E402
from envs.common.sandbox import is_secret_env                             # noqa: E402
import importlib.util                                                     # noqa: E402
_spec = importlib.util.spec_from_file_location("core_score", ROOT / "tools/core_score.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)

SCRIPT = ROOT / "tools/run_core.sh"
FIXTURES = sorted(p.parent for p in (ROOT / "tests/fixtures").rglob("case.json"))


def fake_core(root: Path, version="1.0", digest=None) -> Path:
    for i, case in enumerate(FIXTURES[:3], 1):
        shutil.copytree(case, root / "task3" / "cases" / f"case{i:03d}")
    (root / "dataset.json").write_text(json.dumps({
        "name": "benchcad-2.0-core", "version": version, "n_cases": 3,
        "scorer_digest": digest or scorer_digest(), "case_counts": {"task3": 3}}, indent=2) + "\n")
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}"
             for p in sorted(root.rglob("*")) if p.is_file()]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return root


def run_core(*args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if not is_secret_env(k)}
    env.update({"BENCHCAD_PYTHON": sys.executable, "BENCHCAD_SKIP_SETUP": "1",
                "HF_HOME": str(Path(os.environ.get("TMPDIR", "/tmp")) / "run_core_test_hf_home"), **(env_extra or {})})
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True, env=env)


def test_dry_run_checks_everything_and_prints_the_published_settings(tmp_path):
    data = fake_core(tmp_path / "core")
    r = run_core("--model", "mock/oracle", "--data", str(data), "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "dataset verified" in r.stdout and "all checks passed" in r.stdout
    cmd = next(l for l in r.stdout.splitlines() if "would run:" in l)
    for want in ("--model mock/oracle", "--rounds 30", "--rep 0", "--resume", f"--cases {data}"):
        assert want in cmd, cmd
    assert "SMOKE" not in r.stdout


def test_a_smoke_run_says_so_and_efforts_each_get_a_run(tmp_path, monkeypatch):
    data = fake_core(tmp_path / "core")
    r = run_core("--model", "gemini/gemini-3.8-flash", "--effort", "low,high", "--rounds", "3",
                 "--data", str(data), "--dry-run", env_extra={"GEMINI_API_KEY": "x"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SMOKE RUN: rounds=3, not a reportable score" in r.stdout
    runs = [l for l in r.stdout.splitlines() if "would run:" in l]
    assert len(runs) == 2 and "--effort low" in runs[0] and "--effort high" in runs[1]


def test_it_refuses_before_any_spend(tmp_path):
    data = fake_core(tmp_path / "core")
    r = run_core("--model", "openai/gpt-6-astra", "--data", str(data), "--dry-run")
    assert r.returncode != 0 and "OPENAI_API_KEY" in r.stderr                  # no key
    r = run_core("--model", "openai/gpt-6-astra", "--effort", "ultra", "--data", str(data), "--dry-run",
                 env_extra={"OPENAI_API_KEY": "x"})
    assert r.returncode != 0 and "no effort 'ultra'" in r.stderr
    old = fake_core(tmp_path / "old", version="0.9")
    r = run_core("--model", "mock/oracle", "--data", str(old), "--dry-run")
    assert r.returncode != 0 and "expects benchcad-2.0-core 1.0" in r.stderr
    other = fake_core(tmp_path / "other", digest="0" * 64)
    r = run_core("--model", "mock/oracle", "--data", str(other), "--dry-run")
    assert r.returncode != 0 and "different T6 scorer" in r.stderr
    (data / "dataset.json").write_text((data / "dataset.json").read_text().replace('"1.0"', '"1.1"'))
    r = run_core("--model", "mock/oracle", "--data", str(data), "--dry-run")
    assert r.returncode != 0 and "no Hugging Face token" in r.stderr           # the manifest no longer matches:
                                                                                # a dry run checks access, not downloads


def _rec(case, score=None, **kw):
    return {"case": f"/d/{case}", "case_id": case.split("/")[-1], "model": "m/x", "effort": "high",
            "rounds": 30, "score": score, "tokens": {"input": 1000, "cached": 0, "output": 100}, "seconds": 10, **kw}


def test_core_rules(tmp_path):
    """Resolved = scored or no submission (0); errors are pending, not 0; reps
    averaged per case first; T6 only in position mode; a plain mean over cases."""
    recs = [
        _rec("task1/cases/case001", {"score": 0.5}), _rec("task1/cases/case001", {"score": 0.7}),   # reps: 0.6
        _rec("task2/cases/case001", None, submitted=False),                                        # no submission: 0
        _rec("task6/cases/case001", {"score": 0.9, "status": "ok", "correspondence_mode": "position"}),
        _rec("task6/cases/case002", {"score": 0.0, "status": "invalid_prediction",
                                     "correspondence_mode": "position"}),                          # 0
    ]
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"model": "m/x", "effort": "high", "rounds": 30, "cases": recs}))
    [row] = C.summarize([f])
    assert row["complete"] and row["n_cases"] == 4
    assert row["core_mean"] == pytest.approx((0.6 + 0 + 0.9 + 0) / 4)                # not a mean of task means
    pend = recs + [_rec("task3/cases/case001", None, error="MemoryBudgetExceeded: scorer over 4G"),
                   _rec("task6/cases/case003", {"score": 0.4, "status": "ok", "correspondence_mode": "legacy"})]
    f.write_text(json.dumps({"model": "m/x", "effort": "high", "rounds": 30, "cases": pend}))
    [row] = C.summarize([f])
    assert not row["complete"] and row["core_mean"] is None
    assert {p["case"] for p in row["pending"]} == {"case001", "case003"} and len(row["pending"]) == 2
    text = C.report([row], provisional=True)
    assert "INCOMPLETE: 2 of 6 cases pending" in text and "NOT the headline" in text
    smoke = [dict(r, rounds=3) for r in recs]
    f.write_text(json.dumps({"model": "m/x", "effort": "high", "rounds": 3, "cases": smoke}))
    [row] = C.summarize([f])
    assert row["smoke"] and "SMOKE RUN, rounds=3" in C.report([row])


def test_the_frozen_command_line_and_the_bare_form(tmp_path):
    """The line in the release email and on the access page must keep working
    unchanged: tools/run_core.sh --model gemini/<model-id> --effort low,medium,high;
    and a bare --model runs the provider's ladder (gemini: low, medium, high)."""
    data = fake_core(tmp_path / "core")
    key = {"GEMINI_API_KEY": "x"}
    frozen = run_core("--model", "gemini/some-model-id", "--effort", "low,medium,high",
                      "--data", str(data), "--dry-run", env_extra=key)
    bare = run_core("--model", "gemini/some-model-id", "--data", str(data), "--dry-run", env_extra=key)
    for r in (frozen, bare):
        assert r.returncode == 0, r.stdout + r.stderr
        runs = [l for l in r.stdout.splitlines() if "would run:" in l]
        assert [next(w for w in ("low", "medium", "high") if f"--effort {w} " in l) for l in runs] == \
            ["low", "medium", "high"]
        assert all("--model gemini/some-model-id" in l and "--rounds 30" in l and "--rep 0" in l for l in runs)
    anth = run_core("--model", "anthropic/claude-haiku-5-5", "--data", str(data), "--dry-run",
                    env_extra={"ANTHROPIC_API_KEY": "x"})
    assert anth.returncode == 0 and sum("would run:" in l for l in anth.stdout.splitlines()) == 5
    assert "gemini auth: gemini-api-key" in frozen.stderr


def test_the_provider_sdks_are_installed_and_the_summary_names_the_dataset(tmp_path, capsys):
    """A fresh clone failed every real model: uv sync without --group harness
    installs no provider SDK. And with --cases a subset, the summary still
    names the dataset it came from."""
    assert "uv sync --frozen --quiet --group harness" in SCRIPT.read_text()
    data = fake_core(tmp_path / "core")
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"model": "m/x", "effort": "high", "rounds": 30,
                             "cases": [_rec("task3/cases/case001", {"score": 1.0})]}))
    C.main([str(f), "--dataset", str(data / "task3/cases/case001"), "--dataset-info", str(data)])
    assert "dataset benchcad-2.0-core 1.0  scorer_digest" in capsys.readouterr().out


@pytest.mark.parametrize("cpus,mem,want", [
    (8, 14, (10, 2, 2, 4)),      # run's 8 vCPU / 16 GB box: 14 GB available
    (8, 30, (16, 4, 4, 4)), (8, 64, (16, 4, 4, 4)), (32, 256, (32, 4, 28, 4)), (1, 64, (2, 4, 1, 4)),
    (16, 6, (1, 1, 1, 4)), (2, 0, (1, 1, 1, 1))])
def test_the_memory_budget_sets_workers_scorers_and_executions(tmp_path, cpus, mem, want):
    """run.py 3 GB + 3 GB per scorer reserved; two episodes per remaining GB (two per CPU,
    32 at most); scorers max(1, min(4, GB/5)); one sandbox execution per 2 GB, and
    executions + scorers within the CPUs. 16 executions on 8 cores starved the scorers;
    on the 16 GB box 12 workers ran 2.7x the cases/h of the old default 5, with no OOM.
    --workers overrides the episodes only: the execution and scorer caps stay."""
    data = fake_core(tmp_path / "core")
    env = {"BENCHCAD_NPROC": str(cpus), "BENCHCAD_MEM_GB": str(mem)}
    r = run_core("--model", "mock/oracle", "--data", str(data), "--dry-run", env_extra=env)
    assert r.returncode == 0, r.stdout + r.stderr
    w, s, e, smax = want
    assert f"workers {w} per effort, auto: {cpus} CPUs, {mem} GB available" in r.stdout
    assert f"scorers {s}->{smax} as episodes finish" in r.stdout
    cmd = next(l for l in r.stdout.splitlines() if "would run:" in l)
    assert f"--workers {w} --score-workers {s} --max-execs {e} --score-workers-max {smax} --memory-gb {mem} " in cmd, cmd
    r = run_core("--model", "mock/oracle", "--data", str(data), "--dry-run", "--workers", "3", env_extra=env)
    assert (f"workers 3 per effort (--workers override; the memory budget would size {w}), "
            f"{e} sandbox executions at once, scorers {s}->{smax} as episodes finish") in r.stdout
    assert f"--workers 3 --score-workers {s} --max-execs {e} " in r.stdout


def test_a_killed_run_says_so_and_how_to_resume(tmp_path):
    """run.py killed (out of memory, exit 137): say what happened and how to go on."""
    data = fake_core(tmp_path / "core")
    fake = tmp_path / "python"
    fake.write_text(f"""#!/bin/bash
for a in "$@"; do [ "$a" = harness/run.py ] && kill -9 $$; done
exec {sys.executable} "$@"
""")
    fake.chmod(0o755)
    r = run_core("--model", "mock/oracle", "--data", str(data), env_extra={"BENCHCAD_PYTHON": str(fake)})
    assert r.returncode != 0
    assert "run.py was killed (likely out of memory); re-run the same command to resume" in r.stderr, r.stderr


def test_a_record_waiting_for_its_score_is_pending(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"model": "m/x", "effort": "high", "rounds": 30,
                             "cases": [_rec("task3/cases/case001", None, step="/a.step", pending_score=True)]}))
    [row] = C.summarize([f])
    assert not row["complete"] and row["pending"][0]["why"].startswith("episode finished, not scored")


def test_a_subset_prints_no_headline_and_the_union_does(tmp_path, capsys):
    """Measured 2026-10-09: a --cases task4 run printed "CORE MEAN 0.0063 over
    25 cases" as if it were the headline. Fewer cases than dataset.json's
    n_cases is a SUBSET: per-task lines, no CORE MEAN, exit 3."""
    data = fake_core(tmp_path / "core")                                   # n_cases 3
    f = tmp_path / "r.json"
    recs = [_rec(f"task3/cases/case00{i}", {"score": 0.5}) for i in (1, 2)]
    f.write_text(json.dumps({"model": "m/x", "effort": "high", "rounds": 30, "cases": recs}))
    assert C.main([str(f), "--dataset", str(data / "task3/cases/case001"), "--dataset-info", str(data)]) == 3
    out = capsys.readouterr().out
    assert "SUBSET 2/3" in out and "CORE MEAN" not in out and "T3 0.500 (2/2)" in out
    f.write_text(json.dumps({"model": "m/x", "effort": "high", "rounds": 30,
                             "cases": recs + [_rec("task3/cases/case003", {"score": 0.5})]}))
    assert C.main([str(f), "--dataset", str(data / "task3/cases/case003"), "--dataset-info", str(data)]) == 0
    assert "CORE MEAN 0.5000   over 3 cases" in capsys.readouterr().out


def test_a_price_belongs_to_one_model_id():
    """gemini-3.8-flash was billed at another Gemini's price ($0.08/case shown
    to its own maker). Only the exact id, or a dated snapshot of it, is priced."""
    table = C.prices()
    tok = {"input": 1_000_000, "cached": 0, "output": 0}
    assert C.usd("gemini/gemini-3.8-flash", tok, table) is None
    assert C.usd("anthropic/claude-haiku-5-5", tok, table) == pytest.approx(0.1)
    assert C.usd("anthropic/claude-haiku-5-5-20261001", tok, table) == pytest.approx(0.1)
    assert C.usd("anthropic/claude-haiku-5-5-fast", tok, table) is None
    f = {"model": "gemini/gemini-3.8-flash", "effort": "low", "rounds": 30, "n_cases": 1, "n_core": None,
         "subset": False, "complete": True, "core_mean": 0.1, "pending": [], "effort_sent": [], "smoke": False,
         "per_task": {}, "usd_per_case": None, "price": "unknown", "input_tokens_per_case": 1000,
         "output_tokens_per_case": 100, "median_seconds_per_case": 10}
    line = next(l for l in C.report([f]).splitlines() if "per case:" in l)
    assert "price unknown for gemini/gemini-3.8-flash" in line and "in 1000 tok" in line and "$" not in line


def test_the_table_names_the_harness_the_records_ran_on(tmp_path, capsys):
    """acc-1 printed "harness 8c34013": the checkout, pulled mid-run; run.py ran 1cb51ab."""
    f = tmp_path / "r.json"
    recs = [_rec("task3/cases/case001", {"score": 1.0}, harness_commit="1cb51ab00000"),
            _rec("task3/cases/case002", {"score": 1.0}, harness_commit="1cb51ab00000"),
            _rec("task3/cases/case003", {"score": 1.0}, harness_commit="db98eef00000")]
    f.write_text(json.dumps({"model": "m/x", "effort": "high", "rounds": 30, "cases": recs}))
    C.main([str(f), "--json", str(tmp_path / "s.json")])
    assert "harness 1cb51ab00000 (2), db98eef00000 (1)  dataset" in capsys.readouterr().out
    assert json.loads((tmp_path / "s.json").read_text())["harness_commits"] == {"1cb51ab00000": 2, "db98eef00000": 1}
    f.write_text(json.dumps({"model": "m/x", "effort": "high", "rounds": 30, "cases": recs[:1]}))
    C.main([str(f)])
    assert "harness 1cb51ab00000  dataset" in capsys.readouterr().out
    f.write_text(json.dumps({"model": "m/x", "effort": "high", "rounds": 30,
                             "cases": [_rec("task3/cases/case001", {"score": 1.0})]}))
    C.main([str(f)])
    assert "harness unknown (checkout " in capsys.readouterr().out


def test_run_core_gives_a_score_four_hours(tmp_path):
    data = fake_core(tmp_path / "core")
    r = run_core("--model", "mock/oracle", "--data", str(data), "--dry-run")
    assert r.returncode == 0 and "scorer timeout 14400 s per case" in r.stdout


def test_the_published_reference_ships_and_prints_beside_a_line(tmp_path, capsys):
    """results/core_reference.json: the published Core lines, for core_score --compare
    (run_core passes it). Bound to this dataset and this T6 scorer."""
    ref_path = ROOT / "results/core_reference.json"
    ref = json.loads(ref_path.read_text())
    assert (ref["case_set"], ref["dataset_version"]) == ("benchcad-2.0-core", "1.0")
    assert ref["scorer_digest"] == scorer_digest()
    assert len(ref["lines"]) == 25 and all(l["n_cases"] == 100 for l in ref["lines"])
    assert "--compare results/core_reference.json" in SCRIPT.read_text()
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"model": "anthropic/claude-haiku-5-5", "effort": "high", "rounds": 30,
                             "cases": [dict(_rec("task3/cases/case001", {"score": 0.1}),
                                            model="anthropic/claude-haiku-5-5")]}))
    C.main([str(f), "--compare", str(ref_path)])
    haiku = next(l for l in ref["lines"] if (l["model"], l["effort"]) == ("anthropic/claude-haiku-5-5", "high"))
    assert f"reference (v2.0-core1): {haiku['core_mean']:.4f}" in capsys.readouterr().out
