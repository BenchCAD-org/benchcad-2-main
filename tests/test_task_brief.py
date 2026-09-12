"""episode._task_brief: a case anywhere on disk resolves a brief."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.common.episode import _task_brief                    # noqa: E402


def test_brief_txt_wins(tmp_path):
    c = tmp_path / "case"
    c.mkdir()
    (c / "brief.txt").write_text("from brief.txt")
    (c / "case.json").write_text(json.dumps({"env": "t3_part2step"}))
    assert _task_brief(c) == "from brief.txt"


def test_case_json_env_resolves_outside_a_cases_tree(tmp_path):
    """This is the case that used to raise before the episode even started."""
    c = tmp_path / "anywhere" / "case"
    c.mkdir(parents=True)
    (c / "case.json").write_text(json.dumps({"env": "t3_part2step"}))
    want = (ROOT / "envs/t3_part2step/TASK.md").read_text()
    assert _task_brief(c) == want


def test_real_example_case_resolves():
    assert "T3" in _task_brief(ROOT / "tests/fixtures/t3/case1")


def test_legacy_cases_tree_still_walks_up(tmp_path):
    env = tmp_path / "t9_thing"
    (env / "cases" / "fam" / "00").mkdir(parents=True)
    (env / "TASK.md").write_text("legacy brief")
    assert _task_brief(env / "cases" / "fam" / "00") == "legacy brief"


def test_unresolvable_case_says_why(tmp_path):
    c = tmp_path / "orphan"
    c.mkdir()
    with pytest.raises(FileNotFoundError, match="nothing to build a brief from"):
        _task_brief(c)
