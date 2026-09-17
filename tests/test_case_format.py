"""Every fixture under tests/fixtures/ validates against the case format, and the
assemblies rebuild from parts + instances. This is the test that keeps
docs/CASE_FORMAT.md honest: a fixture that drifts from the contract fails here
before any scorer sees it."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envs.common.caseformat import CJK, check_case, is_new_format, load_case  # noqa: E402

CASES = sorted(p.parent for p in (REPO / "tests/fixtures").glob("t*/*/case.json"))


@pytest.mark.parametrize("case", CASES, ids=[str(c.relative_to(REPO)) for c in CASES])
def test_example_validates(case):
    rep = check_case(case, deep=True)
    assert rep.ok, "\n".join(rep.errors)


def test_fixtures_cover_every_task():
    envs = {load_case(c).manifest["env"] for c in CASES}
    assert {e[:2] for e in envs} == {"t1", "t2", "t3", "t4", "t5", "t6"}


def test_no_legacy_layout_in_fixtures():
    """No legacy-layout directory may sit under tests/fixtures/: a fixture arrives
    in the format or not at all. A new one here is a failure, not a skip."""
    legacy = sorted(str(p.relative_to(REPO)) for p in (REPO / "tests/fixtures").glob("t*/*")
                    if p.is_dir() and (p / "gt").is_dir() and not is_new_format(p))
    assert legacy == [], legacy


def test_gt_never_in_input():
    """A case must not stage its answer: the sandbox copies input/ only."""
    from envs.common.sandbox import Sandbox  # noqa: F401  (import guards the API)
    for c in CASES:
        m = load_case(c).manifest
        assert not any(e["path"].startswith("gt/") for e in m["input"])


def test_manifest_carries_no_identity(tmp_path):
    """case.json is the shared artifact, so it names nothing about where the
    case came from: an old-style source block with a Drive link, an upstream
    case number, an e-mail in a note or a home path is rejected, and the
    fixtures themselves are clean."""
    import json
    import shutil
    from envs.common.caseformat import identity_leaks
    for c in CASES:
        assert identity_leaks(load_case(c).manifest) == []
    d = tmp_path / "case1"
    shutil.copytree(REPO / "tests/fixtures/t1/case1", d)
    m = json.loads((d / "case.json").read_text())
    m["source"] = {"repo": "the data pipeline", "split": "example", "source_case": "T1 / case 23",
                   "drive": {"case": "https://drive.google.com/drive/folders/xyz"}}
    m["notes"] = "see /Users/somebody/Desktop"
    m["redaction"] = {"status": "clean", "note": "author was someone@example.com"}
    (d / "case.json").write_text(json.dumps(m))
    errs = [e for e in check_case(d).errors if e.startswith("identity in manifest")]
    assert len(errs) == 5, errs



def test_input_names_are_one_vocabulary(tmp_path):
    """input/ names are the same in every task (docs/CASE_FORMAT.md): the sheet
    is drawing.pdf, supplied parts sit under step_files/. A case that spells
    them differently -- the legacy directory names, a renamed sheet -- fails the
    policy check as an error, not a warning, and its part ids no longer resolve.
    (write_manifest itself refuses such a tree, so the manifest is edited to
    list the renamed paths with their unchanged hashes.)

    T2 and T5 are the tasks that supply part geometry. T4 supplies none, so
    there is no step_files/ there to misspell -- the name is rejected outright,
    under any spelling (tests/test_t4_no_3d.py)."""
    import json
    import shutil
    for task, pdf in (("t2", "drawing.pdf"), ("t5", "drawing.pdf")):
        d = tmp_path / task / "case1"
        shutil.copytree(REPO / "tests/fixtures" / task / "case1", d)
        (d / "input" / "step_files").rename(d / "input" / "parts")
        if pdf:
            (d / "input" / pdf).rename(d / "input" / "sheet.pdf")
        m = json.loads((d / "case.json").read_text())
        for e in m["input"]:
            e["path"] = e["path"].replace("/step_files/", "/parts/").replace(f"input/{pdf}", "input/sheet.pdf")
        (d / "case.json").write_text(json.dumps(m))
        rep = check_case(d)
        bad = [e for e in rep.errors if "not allowed" in e]
        assert any("/parts/" in e and e.startswith("input/") for e in bad), (task, rep.errors)
        assert any("no part file for part_id" in e for e in rep.errors), (task, rep.errors)
        if pdf:
            assert any("input/sheet.pdf" in e for e in bad), (task, rep.errors)
            # synthetic fixtures may lack the sheet (a warning); a real case may not
            assert any(f"input/{pdf} required" in e for e in rep.errors + rep.warnings), (task, rep.errors, rep.warnings)


def test_cjk_hidden_in_a_step_escape_is_caught(tmp_path):
    """A CAD system writes a non-ASCII feature name as an ISO 10303-21 escape,
    so a raw-text CJK search over a STEP file finds nothing while the name is
    right there. Measured on the shipped T1 sample before this gate existed:
    zero raw hits, two after decoding, in `MANIFOLD_SOLID_BREP ( '<fillet>10' )`.
    The name says which language the part was modelled in, which is identity."""
    import shutil
    from envs.common.caseformat import decode_step_text
    d = tmp_path / "case1"
    shutil.copytree(REPO / "tests/fixtures/t1/case1", d)
    step = d / "gt/gt.step"
    raw = step.read_text()
    # The escape for U+5706 U+89D2, exactly as a CAD system writes a Chinese
    # feature name. Mark the first quoted entity name in the file, whatever
    # entity that is -- the leak is the escape, not the entity type.
    i = raw.index("PRODUCT('")
    marked = raw[:i] + "PRODUCT('\\X2\\57068A92\\X0\\" + raw[i + len("PRODUCT('"):]
    assert marked != raw
    step.write_text(marked)
    assert not CJK.search(marked), "the point of the test: raw search sees nothing"
    assert CJK.search(decode_step_text(marked)), "decoder must see it"
    errs = [e for e in check_case(d).errors if "STEP name" in e]
    assert errs, check_case(d).errors


def test_t6_inner_layer_views_are_input_vocabulary(tmp_path):
    """A board with copper layers between the outer two carries one
    views/view_inner<n>.png per inner layer (ECAD change 48); a 2-layer board has
    none. The name is the vocabulary; a view under any other name is not."""
    import json
    import shutil
    from envs.common.caseformat import sha256
    d = tmp_path / "t6" / "case1"
    shutil.copytree(REPO / "tests/fixtures/t6/case1", d)
    top = d / "input/views/view_top.png"
    m = json.loads((d / "case.json").read_text())
    for name in ("view_inner1.png", "view_inner2.png", "view_side.png", "view_inner.png"):
        shutil.copy(top, d / "input/views" / name)
        m["input"].append({"path": f"input/views/{name}", "sha256": sha256(top)})
    (d / "case.json").write_text(json.dumps(m))
    bad = [e for e in check_case(d).errors if "not allowed" in e]
    assert not any("view_inner1" in e or "view_inner2" in e for e in bad), bad
    assert any("view_side.png" in e for e in bad) and any("view_inner.png" in e for e in bad), bad
