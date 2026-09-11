"""Every fixture under tests/fixtures/ runs end to end: format check (deep), the
reference scores 1.0 through the declared verifier, and input/ stages into a
real Sandbox where a trivial program runs, exports, and is scored. This is the
gate a case has to pass before it counts as "in the repo" -- the format check
alone does not prove the harness can eat it (the first run of this test caught
a SyntaxError in the generated tools.py that every task shared)."""
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("CADENV_LOCAL", "1")       # docker is used when present; otherwise run locally
from tools.dryrun_case import dryrun  # noqa: E402

CASES = sorted(p.parent for p in (REPO / "tests/fixtures").glob("t*/*/case.json"))


@pytest.mark.parametrize("case", CASES, ids=[str(c.relative_to(REPO / "tests/fixtures")) for c in CASES])
def test_case_runs(case, tmp_path):
    rep = dryrun(case, deep=True, work=tmp_path if os.environ.get("CADENV_LOCAL") == "1" and not _docker() else None)
    bad = {k: v for k, v in rep["gates"].items() if not v.get("ok")}
    assert not bad, {k: {kk: vv for kk, vv in v.items() if kk in ("errors", "score", "error", "exception", "traceback", "stderr_tail", "leaks", "rc")} for k, v in bad.items()}


def _docker() -> bool:
    from envs.common.sandbox import _docker_ready
    try:
        return bool(_docker_ready())
    except Exception:                                   # noqa: BLE001
        return False
