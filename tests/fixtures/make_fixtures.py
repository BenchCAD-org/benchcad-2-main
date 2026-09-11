"""Generate the synthetic fixtures under tests/fixtures/t1..t6: one minimal,
self-consistent case per task in the case format (docs/CASE_FORMAT.md), plus a
full-marks solution.

Why synthetic rather than a real case: the real cases live outside the repo
(`envs/*/cases` is a gitignored symlink), so any test that needs one skips on a
fresh clone -- and "skipped" looks too much like "passed". These fixtures are
boxes and cylinders built on the spot, contain no benchmark geometry, go into
git, and always run. Real cases never enter git: they live under
envs/<env>/cases/<id>/ in the same format, and that directory is gitignored.

Each <task>/ directory:
    case1/           a case in the format: case.json + input/ + gt/
    solution.py      full-marks answer; must score 1.0
    expected.json    expected score and tolerance
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repo root
from envs.common.caseformat import DRAWING, INSTANCES_FORMAT, PART_DRAWINGS_DIR, STEP_DIR, write_manifest  # noqa: E402
from envs.common.views import VIEWS_TOOL, default_seed, render_case_views  # noqa: E402
from envs.geom.tessellate import ocp_hashcode_fix  # noqa: E402

ocp_hashcode_fix()
import cadquery as cq  # noqa: E402

HERE = Path(__file__).resolve().parent

# t6 is pcb2schematic (BenchCAD-org/the ECAD source repository): the fixture is a
# three-component graph, and the "views" are one blank render so the input
# policy is exercised without any board imagery.
SPECS = {
    "t1": dict(env="t1_drawing2part",      kind="part",     given="nothing_3d",           note="one part drawing -> 3D"),
    "t2": dict(env="t2_realparts2assembly", kind="assembly", given="all_parts_step",       note="part STEPs + assembly drawing -> assembly"),
    "t3": dict(env="t3_part2step",         kind="part",     given="nothing_3d",           note="reference views -> STEP"),
    "t4": dict(env="t4_parts2assembly",    kind="assembly", given="nothing_3d",           note="reference views -> every part modelled, then assembled"),
    "t5": dict(env="t5_drawings2assembly", kind="assembly", given="purchased_parts_step", note="full drawing set -> assembly (purchased parts as STEP)"),
    "t6": dict(env="t6_pcb2schematic",     kind="ecad",     given="renders_only",         note="assembled-PCB renders -> terminal-net graph"),
}

PART_SRC = '''result = (cq.Workplane("XY").box(40, 24, 12)
          .faces(">Z").workplane().hole(8)
          .faces(">Z").workplane().circle(6).extrude(6))'''


def part_gt():
    """A block with a through hole and a boss: planes, cylinders, an inner and
    an outer contour -- enough for a voxel IoU to tell shapes apart."""
    return (cq.Workplane("XY").box(40, 24, 12)
            .faces(">Z").workplane().hole(8)
            .faces(">Z").workplane().circle(6).extrude(6))


# One spelling of the three assembly parts: the expression that builds each one
# where the reference has it. The generator evaluates these, and a solution that
# has to MODEL a part (T4's three, T5's base) gets the same text written into it
# -- so the fixture's geometry and its full-marks answer cannot drift apart.
ASM_SRC = {
    "base": 'cq.Workplane("XY").box(60, 40, 6).val()',
    "post_1": 'cq.Workplane("XY").circle(5).extrude(30).translate((-20, 0, 3)).val()',
    "post_2": 'cq.Workplane("XY").circle(5).extrude(30).translate((20, 0, 3)).val()',
}


def asm_parts():
    """Three parts in place: a base plate and two posts. Three parts give the
    pose solver a proper solution (fewer than three falls back to 24 rotations)."""
    return {pid: eval(src, {"cq": cq}) for pid, src in ASM_SRC.items()}   # noqa: S307  (our own source, above)


def _blank_pdf(path: Path):
    """A one-page empty A4 landscape PDF: the contract input where a drawing goes.
    The fixture measures the scorer, not perception, but the sandbox staging
    path (PDF -> raster beside it) runs on it, so the model-facing listing is
    the real one: drawing.png next to drawing.pdf."""
    import pymupdf
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(); doc.new_page(width=842, height=595); doc.save(str(path)); doc.close()


# T3/T4 views are REAL renders of the fixture's gt/ with the perturbed cameras
# (envs/common/views.py), at the production sizes. The seed is the one
# tools/render_views.py picks for the case, so running that tool on a fixture
# reproduces the same images and the same gt/views.json (boxes and cylinders
# on white: ~10 KB for the composite, ~150 KB for the T4 parts sheet).


def _views(case: Path, env: str) -> dict:
    seed = default_seed(env, case.name)
    render_case_views(case, seed)
    return {"tool": VIEWS_TOOL, "seed": seed}


def _depose(solid):
    """Translate to the bbox centre; return (de-posed solid, 4x4 that puts it back)."""
    b = solid.BoundingBox()
    c = ((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, (b.zmin + b.zmax) / 2)
    T = [[1, 0, 0, c[0]], [0, 1, 0, c[1]], [0, 0, 1, c[2]], [0, 0, 0, 1]]
    return solid.translate((-c[0], -c[1], -c[2])), T


def write(task: str, spec: dict):
    d = HERE / task
    case = d / "case1"
    # alignment/ (T6) is hand-made and its recorded.json was scored by another
    # grader, so it is not regenerated here: kept across a rebuild, and if the
    # graph below changes, tests/test_t6_ecad_alignment.py says so.
    keep = d / "alignment.keep"
    if (case / "alignment").exists():
        (case / "alignment").rename(keep)
    if case.exists():
        shutil.rmtree(case)
    (case / "input").mkdir(parents=True)
    (case / "gt").mkdir()
    if keep.exists():
        keep.rename(case / "alignment")
    (d / "solution.py").unlink(missing_ok=True)
    (d / "expected.json").unlink(missing_ok=True)

    if spec["kind"] == "ecad":
        graph = {
            "schema": "pcb2schematic/1.0",
            "components": [
                {"id": "R1", "type": "resistor",  "terminals": ["R1.1", "R1.2"], "value": 1000.0, "value_unit": "ohm"},
                {"id": "C1", "type": "capacitor", "terminals": ["C1.1", "C1.2"], "value": 1e-7,  "value_unit": "F"},
                {"id": "U1", "type": "ic",        "terminals": ["U1.1", "U1.2", "U1.3"]},
            ],
            "nets": [{"id": "VCC"}, {"id": "GND"}, {"id": "N1"}],
            "incidences": [["R1.1", "VCC"], ["R1.2", "N1"], ["U1.1", "N1"],
                           ["C1.1", "VCC"], ["C1.2", "GND"], ["U1.2", "GND"], ["U1.3", "VCC"]],
        }
        (case / "gt/gt_graph.json").write_text(json.dumps(graph, indent=1) + "\n")
        (case / "input/views").mkdir()
        from PIL import Image
        for v in ("view_top", "view_bottom"):
            Image.new("RGB", (64, 48), (240, 240, 240)).save(case / "input/views" / f"{v}.png")
        (case / "input/README.md").write_text("Synthetic fixture: the renders are blank; the answer is the three-component graph in the solution.\n")
        sol = ('"""Full-marks answer: the reference graph, written out as data. Must score 1.0."""\n'
               f"result = {json.dumps(graph, indent=1)}\n")
        write_manifest(case, id="case1", env=spec["env"], kind="ecad",
                       source={"repo": "synthetic", "split": "example"},
                       generator={"tool": "tests/fixtures/make_fixtures.py"}, synthetic=True,
                       redaction={"status": "clean"}, notes=spec["note"])
    elif spec["kind"] == "part":
        cq.exporters.export(part_gt(), str(case / "gt/gt.step"))
        # The contract input: a one-page empty PDF for T1 (the fixture measures
        # the scorer, not perception, but the sandbox staging path PDF -> raster
        # runs on it), a real four-view render for T3 (see _views).
        gen = {"tool": "tests/fixtures/make_fixtures.py"}
        if task == "t1":
            _blank_pdf(case / "input" / DRAWING)
        elif task == "t3":
            gen["views"] = _views(case, spec["env"])
        sol = f'"""Full-marks answer: rebuild the reference step by step. Must score 1.0."""\nimport cadquery as cq\n\n{PART_SRC}\n'
        write_manifest(case, id="case1", env=spec["env"], kind="part",
                       source={"repo": "synthetic", "split": "example"},
                       generator=gen, synthetic=True,
                       redaction={"status": "clean"}, notes=spec["note"])
    else:
        parts = asm_parts()
        # T5: the base is "made to print" (answer under gt/parts, its blank sheet
        # under input/part_drawings/); the posts are purchased (de-posed STEP under
        # input/step_files/). T2: every part is supplied de-posed under
        # input/step_files/ -- the same directory name in every task that supplies
        # parts. T4 supplies NO 3-D at all: every part type is modelled from the
        # reference views, so every part's canonical geometry is the ANSWER under
        # gt/parts/ and input/ carries only the two renders and the BOM
        # (caseformat.INPUT_POLICY rejects a step_files/ entry for t4).
        drawing_parts = {"base"} if task == "t5" else set()
        modelled_parts = set(parts) if task == "t4" else set()
        in_dir = case / "input" / STEP_DIR
        if task != "t4":
            in_dir.mkdir()
        # the contract sheet beside the parts, blank, for T2/T5 (see _blank_pdf);
        # T4's renders are made from gt/ once instances.json and the parts exist
        if task in ("t2", "t5"):
            _blank_pdf(case / "input" / DRAWING)
        instances, items, lines = [], [], []
        for pid, solid in parts.items():
            if pid in modelled_parts:
                # No supplied file, so the part frame is the de-posed one exactly as
                # for a supplied part, and instances.json still carries the 4x4 that
                # places it -- a T4 part type may have several instances, so its own
                # frame cannot be one instance's assembly pose.
                (case / "gt/parts").mkdir(exist_ok=True)
                dp, T = _depose(solid)
                cq.exporters.export(cq.Workplane(obj=dp), str(case / "gt/parts" / f"{pid}.step"))
                items.append({"part_id": pid, "quantity": 1, "source": "views"})
                lines.append(f'{pid} = {ASM_SRC[pid]}          # modelled from the reference views')
            elif pid in drawing_parts:
                (case / "gt/parts").mkdir(exist_ok=True)
                cq.exporters.export(cq.Workplane(obj=solid), str(case / "gt/parts" / f"{pid}.step"))
                _blank_pdf(case / "input" / PART_DRAWINGS_DIR / f"{pid}.pdf")
                T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                items.append({"part_id": pid, "quantity": 1, "source": "drawing", "file": f"{PART_DRAWINGS_DIR}/{pid}.pdf"})
                lines.append(f'{pid} = {ASM_SRC[pid]}          # modelled from its drawing')
            else:
                dp, T = _depose(solid)
                cq.exporters.export(cq.Workplane(obj=dp), str(in_dir / f"{pid}.step"))
                items.append({"part_id": pid, "quantity": 1, "source": "step", "file": f"{STEP_DIR}/{pid}.step"})
                t = tuple(T[i][3] for i in range(3))
                lines.append(f'{pid} = cq.importers.importStep("case1/input/{STEP_DIR}/{pid}.step").val().translate({t})')
            instances.append({"instance_id": f"{pid}_i1", "part_id": pid, "T": T})
        cq.exporters.export(cq.Workplane(obj=cq.Compound.makeCompound(list(parts.values()))), str(case / "gt/gt.step"))
        (case / "gt/instances.json").write_text(json.dumps({
            "format": INSTANCES_FORMAT,
            "frame": "T maps the part frame (gt/parts file if present, else the input part) to the assembly frame; row-major 4x4, mm",
            "instances": instances}, indent=1) + "\n")
        (case / "input/bom.json").write_text(json.dumps({
            "note": "quantities are the ground truth for the listed part types",
            "n_part_types": len(items), "n_instances": len(instances), "items": items}, indent=1) + "\n")
        body = "\n".join(lines)
        names = ", ".join(f'"{pid}_i1": {pid}' for pid in parts)
        head = ('"""Full-marks answer: model every part and place it where the reference has it. Must score 1.0."""\n'
                "import cadquery as cq\n\n"
                "# Nothing is supplied: every part is built here, already at its place in the assembly.\n"
                ) if task == "t4" else (
                '"""Full-marks answer: put the supplied parts back where the reference has them. Must score 1.0."""\n'
                "import cadquery as cq\n\n"
                "# The supplied parts are de-posed (translated to their bbox centre); move them back.\n")
        sol = (head +
               f"{body}\n\n"
               "result = cq.Assembly(name=\"asm\")\n"
               f"for name, solid in {{{names}}}.items():\n"
               "    result.add(cq.Workplane(obj=solid), name=name)\n")
        gen = {"tool": "tests/fixtures/make_fixtures.py"}
        if task == "t4":
            gen["views"] = _views(case, spec["env"])
        write_manifest(case, id="case1", env=spec["env"], kind="assembly",
                       source={"repo": "synthetic", "split": "example"},
                       generator=gen, synthetic=True,
                       redaction={"status": "clean"}, notes=spec["note"])

    (d / "solution.py").write_text(sol)
    (d / "expected.json").write_text(json.dumps({
        "task_kind": spec["kind"], "given": spec["given"], "note": spec["note"],
        "oracle_iou": 1.0, "tol": 1e-4,
        "placeholder": False,
    }, indent=1) + "\n")
    print(f"  {task}  {spec['kind']:9s} {spec['note']}")


if __name__ == "__main__":
    for t, sp in SPECS.items():
        write(t, sp)
