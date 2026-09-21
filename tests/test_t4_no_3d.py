"""T4 supplies no 3-D: the model reads the reference views and the BOM, models
every part type itself, and assembles them.

Two things have to hold for that to be the task it claims to be, and neither is
visible in a score:

  1. nothing with part geometry is STAGED into the sandbox. `views.png`, the
     per-part sheets `parts/<id>_{alone,in_assembly}.png` and `bom.json`, and
     that is all -- no `.step` anywhere.
     Checked through the real `Sandbox`, the way
     tests/test_views.py::test_no_renderer_reaches_the_sandbox checks that no
     renderer gets in: the staging code is what the model actually sees, and a
     policy that only lives in a validator would not stop a case tree that
     still carried the files.
  2. a case that carries part STEPs is REJECTED by the format check, naming the
     file. T4's headline is part_x_asm_v1 = avg_part x asm_v1, and avg_part is
     free when the parts are handed over -- a submission need only re-export
     them -- so a stray `input/step_files/` entry quietly turns the task back
     into placement-only. That is a broken case, not a harmless extra, and the
     error has to be as loud as a stored raster's.

T2 (every part supplied, placement only) and T5 (the purchased types supplied,
the rest modelled from their drawings) keep theirs; the last test pins that the
ban is T4's alone.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("CADENV_LOCAL", "1")       # docker is used when present; otherwise stage locally

from envs.common.caseformat import INPUT_POLICY, STEP_DIR, check_case, write_manifest  # noqa: E402
from envs.common.views import sheet_names  # noqa: E402

T4 = REPO / "tests/fixtures/t4/case1"
# what the sandbox injects next to the inputs, in every task (envs/common/sandbox.py)
INJECTED = {"tools.py", "sitecustomize.py"}


def _docker() -> bool:
    from envs.common.sandbox import _docker_ready
    try:
        return bool(_docker_ready())
    except Exception:                                   # noqa: BLE001
        return False


def test_no_part_geometry_reaches_the_sandbox(tmp_path):
    from envs.common.sandbox import Sandbox
    # docker can only mount paths under $HOME (see Sandbox._mount_works)
    root = REPO / "work" / "dryrun" if _docker() else tmp_path
    root.mkdir(parents=True, exist_ok=True)
    wd = Path(tempfile.mkdtemp(prefix="t4_no3d_", dir=root))
    Sandbox(T4, wd)
    names = sorted(str(p.relative_to(wd)) for p in wd.rglob("*") if p.is_file())
    assert [n for n in names if n.endswith(".step")] == [], names
    assert [n for n in names if STEP_DIR in n] == [], names
    # the whole of what the model sees, listed: the renders and the BOM
    types = [p["part_id"] for p in json.loads((T4 / "case.json").read_text())["parts"]]
    assert sorted(n for n in names if n not in INJECTED) == sorted(["bom.json", "views.png"] + sheet_names(types)), names


def test_a_t4_case_that_supplies_parts_is_rejected(tmp_path):
    """The case is built honestly -- the part file is on disk AND in the manifest,
    de-posed, not a copy of a gt file -- so the only thing wrong with it is that
    T4 may not supply part geometry at all. That is the error it must produce."""
    from envs.common.caseformat import _cq
    d = tmp_path / "case1"
    shutil.copytree(T4, d)
    cq = _cq()
    (d / "input" / STEP_DIR).mkdir()
    cq.exporters.export(cq.Workplane("XY").box(7, 5, 3), str(d / "input" / STEP_DIR / "x.step"))
    m = json.loads((d / "case.json").read_text())
    write_manifest(d, id="case1", env=m["env"], kind="assembly", source=m["source"],
                   generator=m["generator"], synthetic=True, redaction=m["redaction"],
                   notes=m["notes"])
    assert f"input/{STEP_DIR}/x.step" in [e["path"] for e in json.loads((d / "case.json").read_text())["input"]]
    errs = check_case(d).errors
    assert [e for e in errs if e == f"input/{STEP_DIR}/x.step not allowed for t4"], errs


def test_the_ban_is_t4s_alone():
    """T2 and T5 supply part geometry by design, so the pattern stays allowed
    there: this is a contract per task, not a repo-wide rule."""
    step = f"{STEP_DIR}/body.step"
    import re
    def allows(tp: str) -> bool:
        return any(re.fullmatch(a, step) for a in INPUT_POLICY[tp]["allowed"])
    assert not allows("t4")
    assert allows("t2") and allows("t5")
    assert INPUT_POLICY["t4"]["required"] == ["views.png", "bom.json"]


def _rebuilt(d: Path):
    """case.json rewritten from what is on disk, as a generator would."""
    m = json.loads((d / "case.json").read_text())
    write_manifest(d, id="case1", env=m["env"], kind="assembly", source=m["source"],
                   generator=m["generator"], synthetic=False, redaction=m["redaction"],
                   notes=m["notes"])
    return check_case(d).errors


def test_the_old_strip_is_an_error_and_every_type_needs_its_sheets(tmp_path):
    """`parts_views.png` -- the 1 + 2n-row strip the per-part sheets replaced
    -- is rejected by name, not carried along as an extra; and a part type
    without both of its sheets, or a sheet for a type the case does not
    have, is a broken case. (The fixture is synthetic, where a missing
    required input is a warning; this copy is marked real.)"""
    d = tmp_path / "case1"
    shutil.copytree(T4, d)
    assert _rebuilt(d) == []
    (d / "input/parts_views.png").write_bytes((d / "input/views.png").read_bytes() + b"\0")
    assert [e for e in _rebuilt(d) if e == "input/parts_views.png not allowed for t4"]
    (d / "input/parts_views.png").unlink()
    (d / "input/parts/post_1_in_assembly.png").unlink()
    assert [e for e in _rebuilt(d) if e == "input/parts/post_1_in_assembly.png required for t4: part type 'post_1'"]
    (d / "input/parts/post_1_in_assembly.png").write_bytes((d / "input/parts/post_2_in_assembly.png").read_bytes() + b"\0")
    (d / "input/parts/ghost_alone.png").write_bytes((d / "input/parts/base_alone.png").read_bytes())
    errs = _rebuilt(d)
    assert [e for e in errs if e == "input/parts/ghost_alone.png: no part type 'ghost' in this case"], errs
    # the alias is the same rule as a stored raster elsewhere: an error, never a warning
    assert not [w for w in check_case(d).warnings if "parts_views" in w or "ghost" in w]
