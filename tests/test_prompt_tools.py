"""The prompt advertises exactly the tools the sandbox stages ."""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envs.common.episode import tools_help  # noqa: E402
from envs.common.sandbox import TOOLS_PY  # noqa: E402

STAGED = set(re.findall(r"^def (\w+)\(", TOOLS_PY, re.M))


def test_advertised_tools_are_staged():
    for case in sorted((REPO / "tests/fixtures").glob("t*/*/case.json")):
        advertised = set(re.findall(r"^\s*(?:tools\.py\s+)?(\w+)\(", tools_help(case.parent), re.M))
        assert advertised <= STAGED, (case.parent, advertised - STAGED)
        assert {"export", "crop"} <= advertised


def test_no_renderer_means_no_render_lines():
    for case in sorted((REPO / "tests/fixtures").glob("t*/*/case.json")):
        h = tools_help(case.parent)
        assert "render(" not in h and "views(" not in h, (case.parent, h)


def test_ecad_export_is_described_as_a_graph():
    h = tools_help(REPO / "tests/fixtures/t6/case1")
    assert "pred_graph.json" in h
