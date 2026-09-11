"""The drawing gate on PDFs , the parts-list mapping , symbol
conservation  and vendor identifiers in model-facing JSON.

Drawings whose text is outlines cannot be read here; they are admitted only
with the delivery gate's report (provenance/redaction_report.json, status pass)."""
import json
import shutil
import sys
from pathlib import Path

import pymupdf
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envs.common.caseformat import check_case, sha256, write_manifest  # noqa: E402


def _outline_pdf(path: Path, meta: dict | None = None):
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.draw_line((10, 10), (100, 100))            # geometry only, no text objects
    if meta:
        doc.set_metadata(meta)
    doc.save(str(path)); doc.close()


def _text_pdf(path: Path, text: str, meta: dict | None = None):
    doc = pymupdf.open(); page = doc.new_page(width=300, height=200)
    page.insert_text((20, 40), text)
    if meta:
        doc.set_metadata(meta)
    doc.save(str(path)); doc.close()


def _t1_case(tmp_path, *, outline=True, text="", meta=None, report=None):
    c = tmp_path / "example9"
    (c / "input").mkdir(parents=True); (c / "gt").mkdir()
    shutil.copy(REPO / "tests/fixtures/t1/case1/gt/gt.step", c / "gt/gt.step")
    if outline:
        _outline_pdf(c / "input/drawing.pdf", meta)
    else:
        _text_pdf(c / "input/drawing.pdf", text, meta)
    red = {"status": "clean"}
    if report is not None:
        (c / "provenance").mkdir()
        (c / "provenance/redaction_report.json").write_text(json.dumps(report))
        red["report_sha256"] = sha256(c / "provenance/redaction_report.json")
    write_manifest(c, id="example9", env="t1_drawing2part", kind="part",
                   source={"repo": "synthetic", "split": "example"}, redaction=red)
    return c


def test_outline_pdf_without_report_is_an_error(tmp_path):
    rep = check_case(_t1_case(tmp_path))
    assert any("redaction_report.json is absent" in e for e in rep.errors), rep.errors


def test_outline_pdf_with_passing_report_is_admitted(tmp_path):
    rep = check_case(_t1_case(tmp_path, report={"status": "pass", "checks": {}}))
    assert rep.ok, rep.errors
    assert any("admitted on provenance/redaction_report.json" in w for w in rep.warnings)


def test_outline_pdf_with_failing_report_is_an_error(tmp_path):
    rep = check_case(_t1_case(tmp_path, report={"status": "fail", "checks": {}}))
    assert any("not pass" in e for e in rep.errors), rep.errors


def test_report_hash_is_pinned(tmp_path):
    c = _t1_case(tmp_path, report={"status": "pass"})
    (c / "provenance/redaction_report.json").write_text(json.dumps({"status": "pass", "edited": True}))
    rep = check_case(c)
    assert any("report_sha256" in e for e in rep.errors), rep.errors


def test_cjk_in_pdf_metadata_is_an_error(tmp_path):
    rep = check_case(_t1_case(tmp_path, outline=False, text="M6 THRU", meta={"title": "23_\u76f8\u673a\u56fa\u5b9a\u67b6"}))
    assert any("CJK in PDF metadata" in e for e in rep.errors), rep.errors


def test_metadata_values_are_reported(tmp_path):
    rep = check_case(_t1_case(tmp_path, outline=False, text="M6 THRU", meta={"author": "LAPTOP\\31292"}))
    assert rep.ok and any("PDF metadata present" in w for w in rep.warnings), (rep.errors, rep.warnings)


def test_parts_list_item_numbers_are_checked(tmp_path):
    c = tmp_path / "example8"
    shutil.copytree(REPO / "tests/fixtures/t2/case1", c)
    _text_pdf(c / "input/drawing.pdf", "1  plate   BenchCAD Pro")
    m = json.loads((c / "case.json").read_text())
    write_manifest(c, id="example8", env="t2_realparts2assembly", kind="assembly", source=m["source"], synthetic=False)
    m = json.loads((c / "case.json").read_text())
    m["parts_list"] = {"mapping": "item_number"}
    (c / "case.json").write_text(json.dumps(m))
    rep = check_case(c)
    # post_1 -> item 1 is on the sheet; post_2 (item 2) and base (no number, no name) are not
    assert any("no item number for" in e and "post_2" in e and "base" in e and "post_1" not in e.split("for")[1] for e in rep.errors), rep.errors


def test_symbol_conservation_on_text_pdf(tmp_path):
    """#16: declared counts are checked on the PDF's own text; fewer is an
    error, more needs a gain_note."""
    c = _t1_case(tmp_path, outline=False, text="\u00b10.05  5 X 45\u00b0  \u00d8 12")
    m = json.loads((c / "case.json").read_text())
    m["drawings"] = {"input/drawing.pdf": {"symbols": {"plusminus": 2, "degree": 1, "diameter": 1}}}
    (c / "case.json").write_text(json.dumps(m))
    rep = check_case(c)
    assert any("expected 2, found 1" in e for e in rep.errors), rep.errors
    m["drawings"]["input/drawing.pdf"]["symbols"] = {"plusminus": 0, "degree": 1, "diameter": 1}
    (c / "case.json").write_text(json.dumps(m))
    rep = check_case(c)
    assert any("net gain" in e and "without" in e for e in rep.errors), rep.errors
    m["drawings"]["input/drawing.pdf"]["gain_note"] = "restored from the complete raster"
    (c / "case.json").write_text(json.dumps(m))
    rep = check_case(c)
    assert rep.ok and any("net gain" in w for w in rep.warnings), (rep.errors, rep.warnings)


def test_vendor_identifiers_in_input_json_are_rejected():
    from envs.common.caseformat import vendor_strings
    hits = list(vendor_strings({"items": [{"part_id": "part_01", "quantity": 2, "name": "M03902-04-028-A"},
                                          {"part_id": "part_02", "name": "B6704ZZ deep groove bearing"},
                                          {"part_id": "part_03", "name": "AS2201F-01-06S (SMC)"},
                                          {"part_id": "part_04", "name": "312981249 fixing seat"},
                                          {"part_id": "part_05", "name": "Drive pulley"},
                                          {"part_id": "part_06", "name": "Deep-groove ball bearing"}]}))
    assert [h[1] for h in hits] == ["M03902-04-028-A", "B6704ZZ deep groove bearing", "AS2201F-01-06S (SMC)", "312981249 fixing seat"], hits
