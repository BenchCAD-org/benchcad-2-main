"""The case format shared by every task (docs/CASE_FORMAT.md).

One layout, five tasks:

    <env>/cases/<case_id>/
      case.json         manifest: id, env, kind, source, file hashes, gates
      input/            the only thing the model sees; staged into the sandbox verbatim
        drawing.pdf     T1 part drawing, T2/T5 assembly drawing (rasterized at stage time)
        bom.json        assembly tasks: part types and quantities
        step_files/<id>.step  T2/T5: supplied parts, de-posed (never T4: it supplies no 3-D)
        part_drawings/<id>.pdf  T5: sheets of the made-to-print parts
        views.png       T3/T4: the 2x2 reference render; views/view_*.png + README.md for T6
        parts/<id>_alone.png, parts/<id>_in_assembly.png  T4: one 2x2 pair per part type
      gt/               never staged
        gt.step         the scored answer (a part for T1/T3, an assembly for T2/T4/T5)
        parts/<id>.step the part types that had to be modelled (all of T4's, T5's
                        drawing parts): canonical geometry, own frame
        instances.json  assembly tasks: [{instance_id, part_id, T}] with T a 4x4 in mm
        views.json      T3/T4: the (seeded, perturbed) camera set the renders were made with

This module is the single reader/writer for that layout. Generators write
through `write_manifest`, verifiers read through `load_case`, and
`tools/check_cases.py` enforces the rules in `check_case`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

FORMAT = "benchcad-case/1"
INSTANCES_FORMAT = "benchcad-instances/1"
PART_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
INSTANCE_ID = re.compile(r"^([a-z][a-z0-9_]{0,63})_i([1-9][0-9]{0,3})$")
# ISO 10303-21 escapes hide text from every regex that reads a STEP file as
# bytes. `\\X2\\5706898910\\X0\\` is UTF-16BE hex for a Chinese feature name, and a
# CJK search over the raw file finds nothing: measured on the shipped T1 sample,
# 0 raw hits and 2 after decoding ('MANIFOLD_SOLID_BREP ( \'<fillet>10\', ... )').
# Decode before checking, or the gate certifies the encoding rather than the
# content -- the same mistake as reading a library's in-memory view instead of
# the delivered bytes.
_STEP_X2 = re.compile(r"\\X2\\((?:[0-9A-Fa-f]{4})+)\\X0\\")
_STEP_X = re.compile(r"\\X\\([0-9A-Fa-f]{2})")


def decode_step_text(s: str) -> str:
    """A STEP file's text with its \\X2\\ / \\X\\ escapes resolved."""
    def x2(m):
        try:
            return bytes.fromhex(m.group(1)).decode("utf-16-be", "replace")
        except ValueError:
            return m.group(0)

    def x1(m):
        try:
            return bytes.fromhex(m.group(1)).decode("latin-1", "replace")
        except ValueError:
            return m.group(0)
    return _STEP_X.sub(x1, _STEP_X2.sub(x2, s))


CJK = re.compile("[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")  # CJK ideographs, punctuation, full-width forms

# What each task's input/ may and must contain. One vocabulary across tasks:
# the sheet is `drawing.pdf` (T1's part drawing, T2/T5's assembly drawing),
# supplied part geometry is `step_files/<part_id>.step` (T2, T5), the bill
# of materials is `bom.json`, renders are `views.png` (+ one 2x2 pair per part
# type under `parts/` for T4), T5's per-part sheets are
# `part_drawings/<part_id>.pdf`. Earlier spellings of these names are errors,
# not aliases: one name per thing -- T4's old single strip `parts_views.png`
# (1630 x 4960 px for seven types, downscaled past legibility by the API's
# 2576 px long-edge cap) is rejected, not carried along.
# Rasters are derived at stage time from the PDFs and are never stored; only
# T3/T4 ship PNGs because their inputs are renders that have no PDF source.
#
# T4 supplies NO part geometry, and a `step_files/` entry there is an error in
# the same way a stored raster is an error elsewhere -- not an allowed extra.
# The task is "model every part type from the reference views, then assemble
# them" (`given = "nothing_3d"`), and its headline part_x_asm_v1 multiplies a
# per-part geometry score: hand the parts over and avg_part is free, because a
# submission need only re-export what it was given. So T4's part geometry is
# the ANSWER and lives under gt/parts/ (`resolve_part` already prefers it),
# with input/ carrying only the renders and the BOM. T2 (every part
# supplied, placement only) and T5 (the purchased types supplied, the rest
# modelled from their drawings) keep theirs.
DRAWING = "drawing.pdf"
STEP_DIR = "step_files"
PART_DRAWINGS_DIR = "part_drawings"
_STEP_FILE = STEP_DIR + r"/[a-z][a-z0-9_]*\.step"
PART_SHEETS_DIR = "parts"                     # T4: input/parts/<part_id>_{alone,in_assembly}.png
PART_SHEET_KINDS = ("alone", "in_assembly")
PART_SHEET = re.compile(PART_SHEETS_DIR + r"/(?P<part_id>[a-z][a-z0-9_]*)_(?P<kind>alone|in_assembly)\.png")
INPUT_POLICY = {
    "t1": {"required": [DRAWING], "allowed": [r"drawing\.pdf"], "png": False},
    "t2": {"required": [DRAWING, "bom.json"],
           "allowed": [r"drawing\.pdf", r"bom\.json", _STEP_FILE], "png": False},
    "t3": {"required": ["views.png"], "allowed": [r"views\.png"], "png": True},
    # T4's per-part sheets are required PER PART TYPE, which a static list
    # cannot say: check_case asks for both sheets of every type in the manifest.
    "t4": {"required": ["views.png", "bom.json"],
           "allowed": [r"views\.png", r"bom\.json", PART_SHEET.pattern], "png": True},
    "t5": {"required": [DRAWING, "bom.json"],
           "allowed": [r"drawing\.pdf", r"bom\.json", PART_DRAWINGS_DIR + r"/[a-z][a-z0-9_]*\.pdf",
                       _STEP_FILE], "png": False},
    # T6 pcb2schematic: six standardized renders of the assembled board and the
    # input note; the answer is a terminal-net graph, not a STEP.
    "t6": {"required": ["views/view_top.png", "views/view_bottom.png"],
           # view_inner<n>.png: one top-down copper drawing per inner layer of a
           # board with more than two (ECAD change 48); a 2-layer board has none.
           "allowed": [r"views/view_(top|bottom)(_obl_[ab])?\.png", r"views/view_inner[0-9]+\.png",
                       r"README\.md"], "png": True},
}
ASSEMBLY_TASKS = {"t2", "t4", "t5"}
KIND_OF = {"t1": "part", "t2": "assembly", "t3": "part", "t4": "assembly", "t5": "assembly", "t6": "ecad"}
GT_FILE = {"part": "gt/gt.step", "assembly": "gt/gt.step", "ecad": "gt/gt_graph.json"}


def task_prefix(env: str) -> str:
    m = re.match(r"^(t[1-6])(_|$)", env)
    if not m:
        raise ValueError(f"env id does not start with t1..t6: {env}")
    return m.group(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def listing(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


# ── geometry ─────────────────────────────────────────────────────────────────

def _cq():
    from envs.geom.tessellate import ocp_hashcode_fix
    ocp_hashcode_fix()
    import cadquery as cq
    return cq


def solids(step: Path) -> list:
    cq = _cq()
    shape = cq.importers.importStep(str(step))
    out = shape.solids().vals()
    return out if out else [shape.val()]


def invariants(solid) -> dict:
    """Pose-free description of one solid, used for identity checks."""
    import numpy as np
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid.wrapped, props)
    vol = props.Mass()
    ix, iy, iz = props.PrincipalProperties().Moments()
    surf = GProp_GProps()
    BRepGProp.SurfaceProperties_s(solid.wrapped, surf)
    area = surf.Mass()
    scale = vol ** (5.0 / 3.0) if vol > 0 else 1.0
    mom = sorted(float(m) / scale for m in (ix, iy, iz))
    return {"volume": float(vol), "area": float(area), "faces": len(solid.Faces()),
            "moments": [float(np.round(m, 6)) for m in mom]}


def geometry_class(sols) -> str:
    """Hash of the invariants at 4 significant digits, over every solid of the
    part file (a multi-solid part is one part type). Two part types with the
    same class are the same geometry under two names and are interchangeable
    for pairing (ASM-07 ships one bar as part_01 and part_07)."""
    if not isinstance(sols, (list, tuple)):
        sols = [sols]
    def sig(x): return 0.0 if x == 0 else float(f"{x:.4g}")
    rows = sorted(json.dumps([sig(i["volume"]), sig(i["area"]), i["faces"], [sig(m) for m in i["moments"]]])
                  for i in map(invariants, sols))
    return "g-" + hashlib.sha256("|".join(rows).encode()).hexdigest()[:12]


def transform(solid, T):
    """Apply a 4x4 (row-major, mm) to a solid."""
    from OCP.gp import gp_Trsf, gp_GTrsf, gp_Mat, gp_XYZ
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    cq = _cq()
    t = gp_Trsf()
    t.SetValues(T[0][0], T[0][1], T[0][2], T[0][3],
                T[1][0], T[1][1], T[1][2], T[1][3],
                T[2][0], T[2][1], T[2][2], T[2][3])
    return cq.Shape.cast(BRepBuilderAPI_Transform(solid.wrapped, t, True).Shape())


def resolve_part(case: Path, part_id: str) -> Path:
    """The file a part_id's frame is defined by: the answer under gt/parts when
    the part had to be modelled, otherwise the supplied input part under
    input/step_files (the same directory name in every assembly task)."""
    for rel in (f"gt/parts/{part_id}.step", f"input/{STEP_DIR}/{part_id}.step"):
        if (Path(case) / rel).exists():
            return Path(case) / rel
    raise FileNotFoundError(f"{case}: no file for part_id {part_id!r}")


def rebuild_assembly(case: Path):
    """The assembly implied by gt/parts + gt/instances.json, as a compound."""
    cq = _cq()
    inst = json.loads((case / "gt/instances.json").read_text())
    cache = {}
    placed = []
    for rec in inst["instances"]:
        pid = rec["part_id"]
        if pid not in cache:
            cache[pid] = solids(resolve_part(case, pid))
        for s in cache[pid]:
            placed.append(transform(s, rec["T"]))
    return cq.Compound.makeCompound(placed)



# ── drawings: what a PDF actually says ───────────────────────────────────────

CJK_FONT_HINT = re.compile(r"simsun|simhei|fangsong|kaiti|hysw|stsong|stfangsong|stkaiti|stxihei|pingfang|"
                           r"notosanscjk|notoserifcjk|sourcehan|msgothic|msmincho|meiryo|yugothic|malgun|"
                           r"microsoftyahei|yahei|dengxian|nsimsun|songti|heiti", re.I)


# engineering symbols a text rewrite can silently drop; AutoCAD codes fold to the glyph
SYMBOLS = {"\u00b1": ("\u00b1", "%%p", "%%P"), "\u00b0": ("\u00b0", "%%d", "%%D"), "\u00d8": ("\u00d8", "%%c", "%%C")}


def symbol_counts(text: str) -> dict:
    """Literal glyph counts, plus the AutoCAD code forms counted separately
    (dimension overrides carry `<>%%p0.02`; annotation text carries the glyph).
    Declared expectations are compared against the literal glyphs."""
    out = {}
    for glyph, forms in SYMBOLS.items():
        out[glyph] = text.count(glyph)
        out[forms[1]] = sum(text.count(f) for f in forms[1:])
    return out


# vendor-looking identifiers in model-facing JSON: a drawing/part number such as
# M03902-04-028-A or B6704ZZ, a catalogue code with a brand suffix, a bare 6+
# digit code. Keys that are ours (part_id, file, sha256, ...) are skipped.
VENDOR_ID = re.compile(r"(?<![\w/])(?:[A-Z]{1,4}\d{4,}[A-Z0-9-]*|\d{6,}|[A-Z]{2,}\d{2,}[A-Z]+\d+[A-Z0-9-]*)(?![\w/])")
VENDOR_SKIP_KEYS = {"part_id", "instance_id", "file", "path", "sha256", "drive_id", "drive_url", "url", "source_case",
                    "release", "asset", "tag", "note", "format", "schema"}


def vendor_strings(obj, key=None):
    """Yield (key, string) for every string value in a JSON tree that looks like a vendor identifier."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from vendor_strings(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from vendor_strings(v, key)
    elif isinstance(obj, str) and key not in VENDOR_SKIP_KEYS:
        if not obj.startswith("part_") and VENDOR_ID.search(obj):
            yield key, obj


def pdf_text_and_fonts(pdf: Path) -> tuple[str, list[str]]:
    """Extractable text of every page and the embedded font names. Text drawn
    as outlines yields an empty string -- that is why the DXF sidecar exists."""
    import pymupdf
    text, fonts = [], set()
    with pymupdf.open(str(pdf)) as doc:
        for page in doc:
            text.append(page.get_text("text"))
            for f in page.get_fonts():
                fonts.add(f[3])
    return "\n".join(text), sorted(fonts)


XMP_PACKET = re.compile(rb"<\?xpacket begin.*?<\?xpacket end[^>]*>", re.S)


def pdf_raw_identity(path: Path) -> list[str]:
    """Identity that lives in a PDF's BYTES and not in its document view.

    A CAD system writes an XMP packet carrying the machine name, the account
    id, the tool version, the internal working file name and two UUIDs. When
    the object ends up unreferenced -- which is what an incremental save
    leaves behind -- every API-level check passes: measured on five shipped
    drawings, `pymupdf` reported an empty Info dict and `get_xml_metadata()`
    of length 0 while 3,499 bytes of it sat in the file. So this reads the
    bytes. A gate that inspects a library's in-memory view is not inspecting
    the artefact.
    """
    try:
        b = path.read_bytes()
    except OSError as ex:                                       # noqa: BLE001
        return [f"unreadable: {ex}"]
    out = []
    m = XMP_PACKET.search(b)
    if m:
        out.append(f"XMP packet in the raw bytes ({len(m.group(0))} bytes); "
                   "save with garbage collection, not incrementally")
        t = m.group(0).decode("utf-8", "replace")
        for tag in ("dc:creator", "xmp:CreatorTool", "dc:title", "xmpMM:DocumentID"):
            hit = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", t, re.S)
            if hit:
                v = re.sub(r"<[^>]+>", "", hit.group(1)).strip()[:60]
                if v:
                    out.append(f"XMP {tag}: {v!r}")
    return out


def sheet_parts_list(pdf: Path) -> dict[str, str]:
    """{item number: part id} as printed on an assembly sheet's parts list:
    every `part_NN.step` word paired with the nearest integer word to its left
    on the same row. Measured on the two T2 sheets against an independent
    extraction: 21/21 and 20/20 pairs agree. Empty when the sheet has no text
    layer or no STEP FILE column."""
    try:
        import pymupdf
    except ImportError:                                  # pragma: no cover
        import fitz as pymupdf                           # type: ignore
    out: dict[str, str] = {}
    try:
        doc = pymupdf.open(str(pdf))
    except Exception:                                    # noqa: BLE001
        return out
    words = [w for pg in doc for w in pg.get_text("words")]
    files = [w for w in words if re.fullmatch(r"part_\d\d\.step", w[4])]
    items = [w for w in words if re.fullmatch(r"\d{1,2}", w[4])]
    for f in files:
        cy = (f[1] + f[3]) / 2
        cands = [w for w in items if abs((w[1] + w[3]) / 2 - cy) < 4 and w[2] <= f[0]]
        if cands:
            it = min(cands, key=lambda w: f[0] - w[2])
            out[it[4]] = f[4][:-5]
    return out


def pdf_metadata(pdf: Path) -> dict:
    """The PDF's Info dictionary as stored (creator, producer, author, title, ...)."""
    import pymupdf
    with pymupdf.open(str(pdf)) as doc:
        return {k: v for k, v in (doc.metadata or {}).items() if v}


def redaction_report(case: Path) -> dict | None:
    """provenance/redaction_report.json: the delivery gate's verification of the
    drawings (DXF-entity CJK and identity, symbol conservation, parts list vs
    BOM), produced by the data pipeline. This repo keeps only the PDFs the
    model sees; the DXF-level facts travel as this report."""
    p = case / "provenance" / "redaction_report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def drawing_text(case: Path, pdf_rel: str) -> tuple[str, str]:
    """(text, source) for a drawing: the PDF's own text, else ('', 'none')."""
    text, _ = pdf_text_and_fonts(case / pdf_rel)
    return (text, "pdf") if text.strip() else ("", "none")


# ── manifest ─────────────────────────────────────────────────────────────────

@dataclass
class Case:
    dir: Path
    manifest: dict
    @property
    def id(self): return self.manifest["id"]
    @property
    def env(self): return self.manifest["env"]
    @property
    def kind(self): return self.manifest["kind"]
    @property
    def input(self): return self.dir / "input"
    @property
    def gt_step(self): return self.dir / "gt/gt.step"
    @property
    def gt_file(self): return self.dir / GT_FILE[self.kind]
    @property
    def instances(self) -> list[dict]:
        p = self.dir / "gt/instances.json"
        return json.loads(p.read_text())["instances"] if p.exists() else []


def load_case(case_dir: Path) -> Case:
    d = Path(case_dir)
    m = json.loads((d / "case.json").read_text())
    if m.get("format") != FORMAT:
        raise ValueError(f"{d}: format {m.get('format')!r} != {FORMAT!r}")
    return Case(d, m)


def is_new_format(case_dir: Path) -> bool:
    # case.json alone: a fixture with no inputs has no input/ directory once
    # it has been through git, which keeps no empty directories.
    return (Path(case_dir) / "case.json").exists()


def write_manifest(case_dir: Path, *, id: str, env: str, kind: str, source: dict,
                   family: str | None = None, generator: dict | None = None,
                   gates: dict | None = None, redaction: dict | None = None,
                   synthetic: bool = False, notes: str = "") -> dict:
    """Write case.json from what is on disk: hashes every file under input/ and
    gt/, and for assemblies lists the part types with their geometry class."""
    d = Path(case_dir)
    m = {
        "format": FORMAT, "id": id, "env": env, "kind": kind, "family": family,
        "units": "mm", "synthetic": synthetic, "source": source,
        "input": [{"path": str(p.relative_to(d)), "sha256": sha256(p)} for p in listing(d / "input")],
        "gt": [{"path": str(p.relative_to(d)), "sha256": sha256(p)} for p in listing(d / "gt")],
        "generator": generator or {}, "gates": gates or {}, "redaction": redaction or {"status": "unknown"},
        "notes": notes,
    }
    if kind == "assembly":
        inst = json.loads((d / "gt/instances.json").read_text())["instances"]
        counts = {}
        for r in inst:
            counts[r["part_id"]] = counts.get(r["part_id"], 0) + 1
        parts = []
        for pid in sorted(counts):
            f = resolve_part(d, pid)
            sol = solids(f)
            parts.append({"part_id": pid, "quantity": counts[pid], "n_solids": len(sol),
                          "geometry_class": geometry_class(sol),
                          "source": "gt" if f.parts[-3:-1] == ("gt", "parts") else "input"})
        m["parts"] = parts
    (d / "case.json").write_text(json.dumps(m, indent=1) + "\n")
    return m


# ── checks ───────────────────────────────────────────────────────────────────

@dataclass
class Report:
    case: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    @property
    def ok(self): return not self.errors


# case.json is a shared artifact (it is what a reviewer or a runner reads first), so it
# carries no identity at all: no URLs, no source ids, no source case numbers, no accounts,
# no vendor strings. Provenance that must survive goes under provenance/ (never staged).
IDENTITY_KEYS = {"drive", "drive_id", "drive_url", "url", "urls", "href", "account", "author",
                 "owner", "source_case", "delivery_bom", "email"}
IDENTITY_TEXT = re.compile(r"://|drive\.google|docs\.google|[\w.+-]+@[\w-]+\.\w|\\\\|/Users/|/home/")


def identity_leaks(obj, path: str = "case.json") -> list[str]:
    """Every key or string in a manifest that names where the case came from."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in IDENTITY_KEYS:
                out.append(f"{path}.{k}: identity key")
            out += identity_leaks(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += identity_leaks(v, f"{path}[{i}]")
    elif isinstance(obj, str) and IDENTITY_TEXT.search(obj):
        out.append(f"{path}: {obj[:60]!r}")
    return out


def dev_sample_admission(manifest: dict) -> tuple[dict | None, str | None]:
    """`(generator.dev_sample, redaction.report_sha256)` when a manifest is a dev
    sample that pins the delivery report it was admitted on, else `(None, None)`.

    A dev sample under `examples/` is a copy of a real case minus `provenance/`,
    because that directory is what carries the source ids, the delivery reports,
    the source case numbers and the account names -- the whole reason the old
    `examples/` tree had to be purged from git. An outline PDF (text drawn as
    line art, so nothing here can read it) is admitted only on the delivery
    gate's report, and that report is exactly one of the things a sample must
    not ship. So the evidence and the copy cannot sit together, and the gate has
    to run where the evidence is: `tools/make_dev_samples.py` refuses to build a
    sample unless the SOURCE case has `provenance/redaction_report.json` with
    `status = "pass"` at the hash `case.json.redaction.report_sha256` names. The
    sample then carries that hash plus the `generator.dev_sample` marker, and
    the outline PDF becomes a warning here instead of an error.

    Both halves are required on purpose. The marker alone would let any case
    self-declare out of the gate; the hash alone is a field a real case has
    anyway. And neither is enough to admit a sample that was never built by the
    tool -- the tool is the only thing that can put the marker there, and
    `tests/test_dev_samples.py` takes each half away in turn to prove the error
    comes back.
    """
    dev = (manifest.get("generator") or {}).get("dev_sample")
    pin = (manifest.get("redaction") or {}).get("report_sha256")
    if isinstance(dev, dict) and isinstance(pin, str) and len(pin) == 64:
        return dev, pin
    return None, None


def check_case(case_dir: Path, *, deep: bool = False, res: int = 64) -> Report:
    d = Path(case_dir)
    rep = Report(str(d))
    E, W = rep.errors.append, rep.warnings.append
    try:
        c = load_case(d)
    except Exception as ex:                                     # noqa: BLE001
        E(f"case.json: {ex}")
        return rep
    m = c.manifest
    if m["id"] != d.name:
        E(f"id {m['id']!r} != directory name {d.name!r}")
    for leak in identity_leaks(m):
        E(f"identity in manifest: {leak}")
    try:
        tp = task_prefix(m["env"])
    except ValueError as ex:
        E(str(ex)); return rep
    if m["kind"] not in KIND_OF.values():
        E(f"kind {m['kind']!r}")
    elif m["kind"] != KIND_OF[tp]:
        E(f"{m['env']} must be kind={KIND_OF[tp]!r}")

    # R1 files: everything listed exists with the right hash, nothing unlisted.
    for section, root in (("input", d / "input"), ("gt", d / "gt")):
        listed = {e["path"]: e["sha256"] for e in m.get(section, [])}
        on_disk = {str(p.relative_to(d)): p for p in listing(root)} if root.is_dir() else {}
        for rel, h in listed.items():
            p = d / rel
            if not p.exists():
                E(f"{section}: listed file missing: {rel}")
            elif sha256(p) != h:
                E(f"{section}: hash mismatch: {rel}")
        for rel in on_disk:
            if rel not in listed:
                E(f"{section}: file on disk not in manifest: {rel}")
    for p in listing(d):
        if p.is_symlink():
            E(f"symlink inside case: {p.relative_to(d)}")
    for p in listing(d):
        if p.suffix in (".json", ".md", ".txt"):
            try:
                if CJK.search(p.read_text(encoding="utf-8")):
                    E(f"non-English text in {p.relative_to(d)}")
            except UnicodeDecodeError:
                E(f"not UTF-8: {p.relative_to(d)}")
        elif p.suffix.lower() in (".step", ".stp"):
            # A STEP file carries the modeller's feature and body names, and a
            # CAD system writes non-ASCII ones as ISO 10303-21 escapes, which no
            # raw-text search can see. Measured on the shipped T1 sample: zero
            # CJK in the raw bytes, two after decoding, in
            # MANIFOLD_SOLID_BREP ( '<fillet>10', ... ). Those names say which
            # language the part was modelled in, so they are identity.
            try:
                txt = decode_step_text(p.read_text(encoding="utf-8", errors="ignore"))
            except OSError as ex:                                   # noqa: BLE001
                E(f"{p.relative_to(d)}: unreadable: {ex}")
                continue
            hit = CJK.search(txt)
            if hit:
                E(f"{p.relative_to(d)}: CJK in a STEP name (escaped): {hit.group(0)!r}")
    if m["kind"] in GT_FILE and not (d / GT_FILE[m["kind"]]).exists():
        E(f"{GT_FILE[m['kind']]} missing")
    if m["kind"] == "ecad":
        try:
            from envs.common.ecad_graph import load_graph, validate
            obj = json.loads((d / "gt/gt_graph.json").read_text())
            validate(obj); g = load_graph(obj)
            if not g.components or not g.incidences:
                E("gt/gt_graph.json has no components or no incidences")
        except Exception as ex:                                 # noqa: BLE001
            E(f"gt/gt_graph.json: {ex}")

    # R2 input policy: PDFs for drawings, PNG only where the input is a render.
    pol = INPUT_POLICY[tp]
    rel_inputs = [e["path"][len("input/"):] for e in m.get("input", [])]
    for req in pol["required"]:
        if req not in rel_inputs:
            (W if m.get("synthetic") else E)(f"input/{req} required for {tp}")
    for rel in rel_inputs:
        if not any(re.fullmatch(a, rel) for a in pol["allowed"]):
            E(f"input/{rel} not allowed for {tp}")
        if rel.endswith(".png") and not pol["png"]:
            E(f"input/{rel}: rasters are derived at stage time, not stored")
    if tp == "t4":
        # One pair of sheets per part type (envs/common/views.py): a type
        # without its sheets cannot be modelled, and a sheet for a type the
        # case does not have is a stale render of some other case.
        types = [p["part_id"] for p in m.get("parts", [])]
        for pid in types:
            for k in PART_SHEET_KINDS:
                want = f"{PART_SHEETS_DIR}/{pid}_{k}.png"
                if want not in rel_inputs:
                    (W if m.get("synthetic") else E)(f"input/{want} required for {tp}: part type {pid!r}")
        for rel in rel_inputs:
            hit = PART_SHEET.fullmatch(rel)
            if hit and hit.group("part_id") not in types:
                E(f"input/{rel}: no part type {hit.group('part_id')!r} in this case")
    gt_hashes = {e["sha256"] for e in m.get("gt", [])}
    for e in m.get("input", []):
        if e["sha256"] in gt_hashes:
            E(f"{e['path']} is byte-identical to a gt/ file: answer leaks into input")

    # R2b drawings: this repo keeps only the PDFs the model sees. A PDF must
    # be shown to carry no CJK and no identity: extractable text, embedded font
    # names (a CJK subset is text even when extraction fails), and the Info
    # dictionary (creator / producer / author / title / subject / keywords).
    # Text drawn as outlines cannot be read here at all: such a PDF is admitted
    # only with the delivery gate's redaction report (provenance/redaction_report.json,
    # status = pass), which carries the DXF-entity-level facts from the data pipeline.
    drawing_texts: dict[str, tuple[str, str]] = {}
    report = redaction_report(d)
    for e in m.get("input", []):
        rel = e["path"]
        if not rel.endswith(".pdf"):
            continue
        try:
            text, fonts = pdf_text_and_fonts(d / rel)
            meta = pdf_metadata(d / rel)
        except Exception as ex:                                     # noqa: BLE001
            E(f"{rel}: cannot read PDF: {ex}"); continue
        bad_fonts = [f for f in fonts if CJK_FONT_HINT.search(f)]
        if bad_fonts:
            E(f"{rel}: embeds CJK font(s) {bad_fonts}")
        if CJK.search(text):
            E(f"{rel}: CJK in extractable text: {CJK.search(text).group(0)!r}")
        for hit in pdf_raw_identity(d / rel):
            E(f"{rel}: {hit}")
        for k, v in meta.items():
            if CJK.search(str(v)):
                E(f"{rel}: CJK in PDF metadata {k}={v!r}")
        ident = {k: v for k, v in meta.items() if k in ("author", "creator", "producer", "title", "subject", "keywords")}
        if ident:
            W(f"{rel}: PDF metadata present: " + ", ".join(f"{k}={v!r}" for k, v in ident.items()))
        src = "pdf"
        if not text.strip():
            src = "none"
            dev, pin = dev_sample_admission(m)
            if report is None and dev and pin:
                W(f"{rel}: outline text; dev sample of {dev.get('from')} -- admitted on the SOURCE case's "
                  f"delivery report, pinned by redaction.report_sha256 {pin[:12]}... and verified against "
                  f"that hash by {dev.get('tool')} when the sample was built. The report itself is not "
                  f"shipped: it is provenance (see the note in dev_sample_admission)")
            elif report is None:
                (W if m.get("synthetic") else E)(
                    f"{rel}: text is drawn as outlines and provenance/redaction_report.json is absent -- "
                    "the DXF-level checks live in the data pipeline's delivery gate; ship its report")
            elif str(report.get("status", "")).lower() != "pass":
                E(f"{rel}: provenance/redaction_report.json status is {report.get('status')!r}, not pass")
            else:
                W(f"{rel}: outline text; admitted on provenance/redaction_report.json (status pass)")
        drawing_texts[rel] = (text, src)
    if report is not None:
        rs = (m.get("redaction") or {}).get("report_sha256")
        actual = sha256(d / "provenance" / "redaction_report.json")
        if rs != actual:
            E(f"case.json redaction.report_sha256 {str(rs)[:12]}... != provenance/redaction_report.json {actual[:12]}...")

    # R2d symbol conservation: a redaction/translation pass can drop plus-minus,
    # degree and diameter glyphs without touching the geometry or the CJK count
    # (measured: 90 and 16 lost across 18 sheets on a cp1252-as-UTF-8 rewrite).
    # A case may declare the expected counts per drawing; when it does they
    # are checked, and the observed counts are always reported.
    declared_syms = (m.get("drawings") or {})
    for rel, (text, src) in drawing_texts.items():
        if src == "none":
            continue                # outline PDF: the declared counts come from the delivery report
        got = symbol_counts(text)
        exp = (declared_syms.get(rel) or {}).get("symbols")
        if isinstance(exp, dict):
            gains = []
            # the two encodings of one symbol are checked separately and never summed:
            # a rewrite that drops the high-byte glyph leaves the ASCII %%p code intact,
            # so a merged total, or a check on one form only, reads green through the loss
            names = {"plusminus": "\u00b1", "degree": "\u00b0", "diameter": "\u00d8",
                     "plusminus_code": "%%p", "degree_code": "%%d", "diameter_code": "%%c"}
            for glyph, n in exp.items():
                g = names.get(glyph, glyph)
                have = got.get(g, 0)
                if have < int(n):
                    E(f"{rel}: symbol {g!r} expected {n}, found {have} -- a text rewrite dropped glyphs")
                elif have > int(n):
                    gains.append(f"{g!r} +{have - int(n)}")
            if gains:
                # a source whose own text was already degraded (fonts not embedded) can
                # legitimately gain glyphs restored from a raster; the gain must be explained
                note = (declared_syms.get(rel) or {}).get("gain_note")
                if note:
                    W(f"{rel}: symbol net gain {', '.join(gains)} -- {note}")
                else:
                    E(f"{rel}: symbol net gain {', '.join(gains)} without drawings[...].gain_note -- unexplained additions")
        # Plain lookups, not `\u` escapes inside the f-string: pyproject says
        # >= 3.11 and 3.11 cannot compile a backslash there (measured: the
        # runner died at import on a 3.11 venv).
        pm, deg, dia = got["\u00b1"], got["\u00b0"], got["\u00d8"]
        W(f"{rel}: symbols plusminus {pm} (+{got['%%p']} in dimension codes), degree {deg} (+{got['%%d']}), "
          f"diameter {dia} (+{got['%%c']})" + ("" if isinstance(exp, dict) else " -- no expected counts declared"))
    # model-facing JSON must not carry vendor identifiers (drawing numbers, catalogue
    # codes with brand suffixes): the drawings are redacted, the BOM must be too
    for e in m.get("input", []):
        if e["path"].endswith(".json"):
            try:
                hits = list(vendor_strings(json.loads((d / e["path"]).read_text())))
            except Exception as ex:                                 # noqa: BLE001
                E(f"{e['path']}: not valid JSON: {ex}"); continue
            if hits:
                E(f"{e['path']}: vendor-looking identifier in model-facing JSON: {hits[0][0]}={hits[0][1]!r}"
                  + (f" (+{len(hits)-1} more)" if len(hits) > 1 else ""))
    n_pdf = sum(1 for e in m.get("input", []) if e["path"].endswith(".pdf"))
    if tp in ("t1", "t2", "t5") and not m.get("synthetic") and n_pdf == 0:
        E("no PDF under input/ was checked for CJK -- a drawing task with no drawing")

    # R2b' duplicated content streams: a PDF rewrite that appends a stream without
    # replacing the old one draws every label twice -- invisible on the page,
    # visible in the extracted text as every token doubled (measured on a
    # redaction pass: 88 spans -> 176). Reported per drawing.
    for rel, (text, src) in drawing_texts.items():
        if src != "pdf" or not text.strip():
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 8:
            from collections import Counter
            cnt = Counter(lines)
            doubled = sum(1 for ln, n in cnt.items() if n >= 2)
            if doubled >= 0.9 * len(cnt):
                W(f"{rel}: every distinct text line appears at least twice ({len(lines)} lines, {len(cnt)} distinct) -- "
                  "likely a duplicated content stream from a PDF rewrite; check page.get_contents()")
    # R2c T2/T5: the assembly drawing's parts list must map to the part ids
    if tp in ("t2", "t5") and not m.get("synthetic"):
        pl = m.get("parts_list")
        if not isinstance(pl, dict) or pl.get("mapping") not in ("item_number", "declared", "none"):
            E("case.json parts_list.mapping must be item_number | declared | none")
        elif pl["mapping"] == "declared":
            table = pl.get("table")
            if not isinstance(table, dict) or not table:
                E("parts_list.mapping = declared needs parts_list.table {item: part_id}")
            else:
                ids = [p_["part_id"] for p_ in m.get("parts", [])]
                vals = list(table.values())
                dup = sorted({v for v in vals if vals.count(v) > 1})
                if dup:
                    E(f"parts_list.table maps several items to one part: {dup}")
                miss = sorted(set(ids) - set(vals))
                if miss:
                    E(f"parts_list.table has no item for {miss}")
                # A declared table is checked against the sheet, not trusted:
                # each (item, file) pair must sit on one row of the parts list.
                txt, src = drawing_texts.get(f"input/{DRAWING}", ("", "none"))
                if src != "none":
                    rows = sheet_parts_list(d / "input" / DRAWING)
                    bad = {k: v for k, v in table.items() if rows.get(str(k)) not in (None, v)}
                    if bad:
                        E(f"parts_list.table disagrees with the sheet on items {sorted(bad)}: "
                          f"declared {bad}, sheet {{k: rows[k] for k in bad}}")
                    unread = [k for k in table if str(k) not in rows]
                    if unread:
                        W(f"parts_list.table items not readable off the sheet: {sorted(unread)}")
        elif pl["mapping"] == "none" and not pl.get("note"):
            E("parts_list.mapping = none needs a note (the case is not solvable as posed)")
        elif pl["mapping"] == "item_number":
            txt, src = drawing_texts.get(f"input/{DRAWING}", ("", "none"))
            if src == "none":
                E("parts_list item_number check: assembly drawing has no readable text (PDF or DXF)")
            else:
                tokens = set(re.findall(r"[A-Za-z0-9_-]+", txt))
                ids = [p_["part_id"] for p_ in m.get("parts", [])]
                missing = []
                for pid in ids:
                    num = re.sub(r"\D", "", pid)
                    forms = {pid, pid.upper(), num, num.lstrip("0") or "0", f"PART-{num}", f"part-{num}"}
                    if not (forms & tokens):
                        missing.append(pid)
                if missing:
                    E(f"parts list ({src} text) has no item number for {missing}")

    # R3 assemblies: parts + instances + the scored assembly agree by construction.
    if m["kind"] == "assembly":
        ip = d / "gt/instances.json"
        if not ip.exists():
            E("gt/instances.json missing"); return rep
        inst = json.loads(ip.read_text())
        if inst.get("format") != INSTANCES_FORMAT:
            E(f"instances.json format {inst.get('format')!r}")
        recs = inst.get("instances", [])
        ids = [r.get("instance_id") for r in recs]
        if len(set(ids)) != len(ids):
            E("duplicate instance_id")
        counts: dict[str, int] = {}
        for r in recs:
            pid = r.get("part_id", "")
            if not PART_ID.match(pid):
                E(f"bad part_id {pid!r}")
            mm = INSTANCE_ID.match(r.get("instance_id", ""))
            if not mm or mm.group(1) != pid:
                E(f"instance_id {r.get('instance_id')!r} must be <part_id>_i<k>")
            T = r.get("T")
            if not (isinstance(T, list) and len(T) == 4 and all(len(row) == 4 for row in T)):
                E(f"{r.get('instance_id')}: T must be 4x4")
            else:
                R = [[float(T[i][j]) for j in range(3)] for i in range(3)]
                det = (R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1]) - R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])
                       + R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0]))
                if abs(det - 1.0) > 1e-6 or T[3] != [0, 0, 0, 1]:
                    E(f"{r.get('instance_id')}: T is not a proper rigid transform (det={det:.6f})")
            counts[pid] = counts.get(pid, 0) + 1
            try:
                resolve_part(d, pid)
            except FileNotFoundError:
                E(f"no part file for part_id {pid!r} (instance {r.get('instance_id')})")
        declared = {p["part_id"]: p["quantity"] for p in m.get("parts", [])}
        if declared != counts:
            E(f"case.json parts quantities {declared} != instances {counts}")
        bom = d / "input/bom.json"
        if bom.exists():
            try:
                b = json.loads(bom.read_text())
                bc = {it["part_id"]: it["quantity"] for it in b.get("items", [])}
                if bc != counts:
                    E(f"input/bom.json quantities {bc} != instances {counts}")
            except Exception as ex:                             # noqa: BLE001
                E(f"input/bom.json: {ex}")
        # de-posed inputs: every input part sits at its own bbox centre
        for e in m.get("input", []):
            rel = e["path"]
            if rel.endswith(".step") and rel.startswith(f"input/{STEP_DIR}/"):
                try:
                    cq = _cq()
                    bb = cq.Compound.makeCompound(solids(d / rel)).BoundingBox()   # whole file, not per solid
                    cen = ((bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2)
                    diag = max(bb.DiagonalLength, 1e-9)
                    if max(abs(x) for x in cen) > 0.01 * diag:          # generators round; 1% of size
                        E(f"{rel}: not de-posed (bbox centre {tuple(round(x, 3) for x in cen)}, diag {diag:.1f})")
                except Exception as ex:                         # noqa: BLE001
                    E(f"{rel}: cannot load: {ex}")
        # cheap consistency: multiset of (volume, area) of gt.step solids == instances
        if not rep.errors:
            try:
                gt_inv = sorted((round(invariants(s)["volume"], 3), round(invariants(s)["area"], 3)) for s in solids(c.gt_step))
                exp = []
                cache = {}
                for r in recs:
                    pid = r["part_id"]
                    if pid not in cache:
                        cache[pid] = [invariants(s) for s in solids(resolve_part(d, pid))]
                    exp += [(round(i["volume"], 3), round(i["area"], 3)) for i in cache[pid]]
                exp.sort()
                if len(gt_inv) != len(exp):
                    E(f"gt.step has {len(gt_inv)} solids, parts x instances give {len(exp)}")
                else:
                    # 1e-4 relative: BRepGProp integrates extrusion / spline faces
                    # numerically, and a rotated or mirrored copy of the same solid
                    # comes back 1 mm^3 in 17,000 off (a T4 clamp half, measured);
                    # a different part is off by far more than 0.01 %.
                    for (gv, ga), (ev, ea) in zip(gt_inv, exp):
                        if abs(gv - ev) > 1e-4 * max(1.0, abs(ev)) or abs(ga - ea) > 1e-4 * max(1.0, abs(ea)):
                            E(f"gt.step solid (vol {gv}, area {ga}) has no matching instance (nearest vol {ev}, area {ea})")
                            break
            except Exception as ex:                             # noqa: BLE001
                E(f"geometry check failed: {ex}")
        if deep and not rep.errors:
            import tempfile
            from envs.common.score_asm import instance_shapes
            from envs.geom.iou import iou_step_vs_step
            cq = _cq()
            # Structure, not just volume: the rebuilt union can match gt.step to
            # 1.0 while gt.step hides instances inside a sub-assembly, and the
            # scorer pairs per instance. One leaf per instances.json entry.
            n_leaves = len(instance_shapes(c.gt_step))
            if n_leaves != len(c.instances):
                E(f"gt.step has {n_leaves} placed instances, instances.json {len(c.instances)}")
            rebuilt = rebuild_assembly(d)
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "rebuilt.step"
                cq.exporters.export(rebuilt, str(out))
                iou = iou_step_vs_step(c.gt_step, out, res)
            if iou >= 0.999:
                rep.warnings.append(f"rebuild IoU {iou:.4f}")
            else:
                # The voxel number is the mesher's, not the data's, on thin
                # B-spline parts (rollers, blades, spring clips): the two sides
                # tessellate at tolerances set by bounding boxes that are not
                # tight on splines, and 29 held-out T4 cases sat at 0.998-0.999
                # while every part intersected its reference exactly. Settle it
                # with exact B-rep booleans, solid by solid.
                exact = exact_rebuild_iou(rebuilt, c.gt_step)
                if exact["min"] < 0.999:
                    E(f"assembly rebuilt from parts+instances has IoU {iou:.4f} against gt.step; "
                      f"exact per-solid IoU min {exact['min']:.4f} ({exact['worst']})")
                else:
                    rep.warnings.append(f"rebuild IoU {iou:.4f} (voxel); exact per-solid IoU min {exact['min']:.6f}")
    return rep


def exact_rebuild_iou(rebuilt, gt_step: Path) -> dict:
    """Exact B-rep IoU of each rebuilt solid against the nearest gt.step solid
    (nearest by bbox centre and volume, each gt solid used once):
    {"min": worst IoU, "worst": that solid's index, "iou": all}."""
    gt = solids(gt_step)
    def cen(g):
        b = g.BoundingBox()
        return ((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, (b.zmin + b.zmax) / 2)
    used, ious, worst = set(), [], (1.0, None)
    for k, s in enumerate(rebuilt.Solids()):
        v, cs = s.Volume(), cen(s)
        free = [(i, g) for i, g in enumerate(gt) if i not in used]
        if not free:
            return {"min": 0.0, "worst": f"solid {k}: no gt solid left", "iou": ious}
        i, g = min(free, key=lambda ig: sum((a - b) ** 2 for a, b in zip(cen(ig[1]), cs)) + abs(ig[1].Volume() - v))
        used.add(i)
        try:
            inter = s.intersect(g).Volume()
        except Exception:                                   # noqa: BLE001  -- boolean failed: no overlap
            inter = 0.0
        gv = g.Volume()
        iou = inter / (v + gv - inter) if (v + gv - inter) > 0 else 0.0
        ious.append(iou)
        if iou < worst[0]:
            worst = (iou, f"solid {k} vs gt solid {i}")
    return {"min": worst[0], "worst": worst[1], "iou": ious}
