# Dev samples

Nine real cases, complete: the inputs a model sees, the reference answer, and
the scores our own scorer gives that reference. They are here so that an
evaluation can be **set up and aligned** — install the environment, run one
case, get the same numbers we get — before anyone runs the formal benchmark.

**These are not the formal bank.** They are nine cases we are happy to publish,
chosen to cover all six tasks. The formal bank and its answers are not in this
repository and will not be; a benchmark question is spent the moment it has been
seen. Aligning on these samples and scoring well on them says the *plumbing*
agrees, not that a model is good.

Each sample directory is:

```
task<N>/
  TASK.md            the prompt the model is given -- byte-identical to envs/<env>/TASK.md
  eval.toml          the evaluation contract: tool permissions, round and token budget,
                     timeouts, submission format, the scoring rule and the headline metric
  cases/<id>/
    case.json        the manifest: every input and reference file with its sha256
    input/           the ONLY thing staged into the container
    gt/              the reference answer; never staged
    expected.json    the numbers this case must reproduce
```

Same format as every case in this repo (`../docs/CASE_FORMAT.md`) minus
`provenance/`, plus `expected.json`. `provenance/` is where a case keeps where it
came from — source ids and links, delivery reports, account names — and none of
it is needed to reproduce a score, so none of it ships. The samples are built by
`../tools/make_dev_samples.py` and re-checked by `../tests/test_dev_samples.py`.

---

## (a) Install, and one command

```sh
uv sync --group harness --group dev              # host: scorers, tests, providers
docker build -t benchcad-sandbox:arm64 sandbox/  # the container submitted code runs in
```

Then, from the repository root:

```sh
# the whole sample set, end to end, no API key: the reference is submitted back
uv run python harness/run.py --model mock/oracle --cases examples --rounds 1

# the same with a deliberately wrong answer, to prove the score can move
uv run python harness/run.py --model mock/dumb   --cases examples --rounds 1

# a real model
uv run python harness/run.py --model anthropic/claude-opus-5 --cases examples
```

Per-case results are written to `results/<model>_<timestamp>.json`. Score one
submission on its own instead:

```sh
uv run python tools/verify_case.py examples/task1/cases/case1 my_part.step
uv run python tools/verify_case.py examples/task2/cases/case1 my_work_dir/submission
uv run python -m envs.verifiers.ecad examples/task6/cases/case1 my_graph.json
```

And check that a case runs end to end in your own environment — format, oracle,
and a real sandbox stage — with:

```sh
uv run python tools/dryrun_case.py examples/task1/cases/case1 --deep
uv run python tools/check_cases.py examples --deep        # all ten, format only
```

Without Docker the sandbox does **not** silently fall back; `CADENV_LOCAL=1` is
required to degrade to a local subprocess, and that mode has no isolation.

## (b) The fixed environment

Submitted code runs in one container, defined by `sandbox/Dockerfile` and built
from `sandbox/requirements.txt`. Nothing about it is left to the host.

| | |
|---|---|
| image | `benchcad-sandbox:arm64` (override with `CADENV_DOCKER_IMAGE`) |
| base | `python:3.12-slim-trixie` — Debian 13, Python 3.12 |
| network | **none** (`--network none`) |
| memory / CPU | `--memory 2g --cpus 1 --pids-limit 256` |
| filesystem | read-only root; `/work` (the case's working directory) is the only writable, persistent path; `/tmp` is a 256 MB tmpfs |
| environment | empty (`--env-file /dev/null`) plus `PYTHONPATH=/work` |
| privileges | `--security-opt no-new-privileges` |

Pinned packages, quoted from `sandbox/requirements.txt` (24 packages, every
version pinned; `pip freeze` of the live image):

```
cadquery==2.3.0            fonttools==4.63.0          python-dateutil==2.9.0.post0
cadquery-ocp==7.9.3.0      kiwisolver==1.5.0          scipy==1.17.1
cadquery-ocp-proxy==7.9.3.0  matplotlib==3.11.1       six==1.17.0
casadi==3.7.2              multimethod==2.0.2         trimesh==4.12.2
contourpy==1.3.3           nptyping==2.5.0            typing_extensions==4.16.0
cycler==0.12.1             numpy==1.26.4              typish==1.9.3
ezdxf==1.4.3               packaging==26.3            vtk==9.5.2
                           pillow==12.3.0
                           pyparsing==3.3.2
```

`nlopt==2.11.0` is installed separately with `--no-deps` — the reason is in the
Dockerfile, and it matters: the live image's own `pip freeze` output does not
resolve, because nlopt declares `numpy>=2` while numpy is pinned at 1.26.4 for
the cadquery-ocp pair. `sandbox/sitecustomize.py` is copied in as well; without
it every `.faces()` / `.edges()` selector raises, and `pip freeze` cannot see
it. **Identical package lists are not identical behaviour** — two images with
byte-identical freeze output disagreed on exactly that call.

The host side (scorers, harness, tests) is `pyproject.toml`:

```toml
requires-python = ">=3.11"
dependencies = ["cadquery>=2.5", "numpy", "pillow", "pymupdf", "scipy", "trimesh"]
[dependency-groups]
harness = ["anthropic", "openai==2.35.1", "google-genai>=2.23"]
dev     = ["pytest"]
```

The host's cadquery is deliberately **not** the container's: the container is
2.3.0, the host resolves 2.8.0. That is not an oversight — the scoring path was
checked to agree across both (see the round-trip note in (c), measured identical
to six decimals on each).

**Resources, measured on this repository (Apple M-series, macOS):**

| | peak RSS | wall time |
|---|---|---|
| score one part case (`verify_case.py`, task1) | 2.48 GB | 16 s |
| score one assembly case (`verify_case.py`, task5, 21 part types / 25 instances) | 3.14 GB | 69 s |
| the sandbox the model's code runs in, as configured | 2 GB cap | 1 CPU, 300 s per code block |
| `mock/oracle` over all samples, 1 round | — | 4 min 35 s, measured on the eight-sample set before T2 took its second case |
| building the samples from the case trees | — | 10 min 17 s |

Scoring runs on the **host**, not in the sandbox, and the two limits are
separate: the 2 GB cap applies to the model's own code, while the scorer above
peaks at 2.5-3.1 GB (OCCT tessellation plus the voxel grids). It is CPU-bound
and single-process, and the assembly metrics dominate because `asm_v1` removes
each part type in turn and re-voxelises. Budget **8 GB of RAM and 2+ cores** for
the host; one core is enough, it is just slower.

## (c) The samples

| sample | source task | headline metric | oracle | round trip | trivial floor | size |
|---|---|---|---|---|---|---|
| `task1/cases/case1` | T1 `drawing2part` — part drawing to part | `part_v1` | 1.000000 | 1.000000 | 0.098903 | 0.9 MB |
| `task2/cases/case1` | T2 `realparts2assembly` — supplied parts + assembly sheet | `asm_v1` | 1.000000 | 1.000000 | 0.000000 | 9.5 MB |
| `task2/cases/case2` | T2 `realparts2assembly` — one supplied part carries two solids | `asm_v1` | 1.000000 | 1.000000 | 0.000000 | 9.4 MB |
| `task3/cases/case1` | T3 `part2step` — rendered views to part | `part_v1` | 1.000000 | **0.999964** | 0.157033 | 1.4 MB |
| `task3/cases/case2` | T3 `part2step` | `part_v1` | 1.000000 | **0.999595** | 0.141540 | 1.4 MB |
| `task4/cases/case1` | T4 `parts2assembly` — supplied parts + views | `part_x_asm_v1` | 1.000000 | 1.000000 | 0.000000 | 13.6 MB |
| `task5/cases/case1` | T5 `drawings2assembly` — five part drawings + supplied STEP parts | `part_x_asm_v1` | 1.000000 | 1.000000 | 0.000000 | 11.5 MB |
| `task6/cases/case1` | T6 `pcb2schematic` — board renders to schematic graph | Metric V2 | 1.000000 | 1.000000 | 0.000000 | 0.2 MB |
| `task6/cases/case2` | T6 `pcb2schematic` | Metric V2 | 1.000000 | 1.000000 | 0.000000 | 0.5 MB |

48.5 MB in total, 162 files — T2 replacing one 9.5 MB sample with two (9.5 MB
and 9.4 MB) is the whole of the change from the previous 39.0 MB / 136 files.
`../docs/METRICS.md` defines every metric; each sample's `eval.toml` names the
one that applies and quotes its formula.

Every **oracle** is exactly 1.000000 and the two T3 **round trips** are a few
ten-thousandths short. Those shortfalls are facts about the metric rather than
about the case, and they are recorded, not rounded away, because **a client who
reproduces 1.0 where we get 0.999527 is not running our ruler** — which is the
entire point of these samples. Reproduce the digits, within `tolerance`.

- **round trip** is the reference imported and written back out with cadquery.
  This is what a *perfect* submission looks like on disk: a model cannot hand
  over our file, it hands over a program, and `tools.export` writes the result
  through cadquery's STEP writer. It is also exactly what `mock/oracle`
  submits, so it is the column an end-to-end run reproduces.
- On both **T3** samples the round trip used to cost about 0.15 of the
  headline, all of it `part_v1`'s voxel term (0.786 / 0.808) while the legacy
  raw 64³ `iou` stayed at 1.0000. That term is now a true solid voxelisation
  and reaches exactly 1.0 on both re-exports (the same cell count on both
  sides), so the round trip is 0.999957 / 0.999527. What is left is the lab's
  sampled `surf_f1` (0.9999 / 0.9986) and `pix_fg` (0.99997 / 1.0) —
  20,000 surface samples and a 524² render are not bit-stable under
  re-tessellation, and neither term was changed. The synthetic fixtures under
  `tests/fixtures/` are boxes and cylinders whose re-export is
  vertex-identical, so they do not show it at all — which is why having real
  cases in git is worth something.
- **T5**'s oracle is now exactly 1.0. It was 0.999202: one purchased part type
  scored 0.983244 against itself inside `avg_part`, on geometry that is
  literally identical, because the voxel term was a Monte-Carlo estimate of the
  solid. Note that `envs/verifiers/assembly.py` still records that era's 0.9891
  oracle floor in a comment. See "provisional numbers" below.
- The **assembly floors are 0.000000**, not small. A single bounding-box block
  places no part type at all, and `asm_v1` is scored per part type, so the
  trivial answer earns nothing rather than a little.

`mock/dumb` submits a 10 mm box (or, for T6, the empty graph) — a *different*
trivial answer from the bounding-box block, so its scores need not equal the
floor column above. For T6 they are the same answer and do match.

### How to read `expected.json`

One per case, and the thing to diff your own run against:

```json
{
 "computed_on": "34ef525…",                         which commit's scoring code measured this
 "computed_with": {"metric": "part_v1", "verifier": "envs.verifiers.part:score",
                   "worktree_dirty": false},
 "case": "case1", "env": "t3_part2step", "kind": "part",
 "metric": "part_v1",                               as declared in envs/<env>/task.toml
 "score_key": "score",                              which key of the result record is the headline
 "orientation": "pinned", "pose_mode": "lab",
 "verifier": "envs.verifiers.part:score",
 "metric_reference": "docs/METRICS.md",
 "tolerance": 0.0001,                               how far your number may be from ours
 "oracle":     {"submission": "gt/gt.step", "score": 1.0,      "columns": {...}, "note": "..."},
 "round_trip": {"submission": "gt/gt.step imported and re-exported …",
                "score": 0.999957, "columns": {...}, "note": "..."},
 "baseline":   {"submission": "a solid block of the reference's bounding box …",
                "score": 0.189367, "columns": {...}, "note": "..."},
 "gt_sha256": "…",                                  the reference the scores are against
 "how_to_check": "uv run python tools/verify_case.py examples/task3/cases/case1 <your.step>"
}
```

To align: score the reference yourself (`how_to_check`, with `gt/gt.step` as the
submission) and compare against `oracle.score` within `tolerance`. Then run
`harness/run.py --model mock/oracle` and compare against `round_trip.score`.
`columns` holds the diagnostics the verifier also reports (`iou`, `hit`,
`avg_part`, `asm_v1`, or the ECAD graph counts); a run whose headline matches but
whose `iou` does not has a different problem from one whose headline is simply
wrong, so both are worth comparing.

### Provisional numbers, and how to refresh them

`computed_on` is the commit whose scoring code produced the scores in that file.
The scores are a measurement of this repository, so they move when the scorers
move. Check whether yours are stale:

```sh
git log --oneline <computed_on>..HEAD -- \
    envs/verifiers envs/common/part_metric.py envs/common/asm_v1.py \
    envs/common/avg_part.py envs/common/score_asm.py envs/common/rubric_asm.py \
    envs/common/ecad_graph docs/METRICS.md
```

If that prints anything, re-measure — it rewrites `expected.json` and touches
nothing else, so the shipped case bytes cannot move when the numbers do:

```sh
uv run python tools/make_dev_samples.py --scores-only
```

**The oracle-exactness fix has landed**, and these files are its
re-measurement. `part_v1`'s voxel term is now BenchCAD-main's true solid
voxelisation rather than a Monte-Carlo estimate of it, and `avg_part` decides
instance identity on analytic invariants (`docs/METRICS.md`, "What the iou term
replaced, and why"). What moved: T5's oracle and round trip 0.999202 → 1.0,
the T2 sample's `avg_part` column 0.992628 → 1.0 (measured on the T2 sample
that has since been replaced, below; the two that replaced it were measured
after the fix and never carried the old number), and both T3 round trips
0.849 / 0.850 → 0.999957 / 0.999527. Every baseline, T1, T4 and both T6 samples are unchanged.
**Every iou number of every task is different from before that fix** — it is a
change of metric, not a repair with no consequences. Everything else in a
sample — the inputs, the reference, the manifest — is unaffected by it.

### T2's two samples

T2 shipped one sample until 2026-09-11 and now ships two, **replacing** it
rather than adding to it. Both are contributed, audited cases:

- `task2/cases/case1` — the clean standard example: 26 semantic instances,
  26 solids, 21 part types. This is the **same assembly** the replaced sample
  was built from (all 20 geometry classes and the per-`part_id` (class,
  quantity, n_solids) map are identical, and it is the same assembly sheet),
  independently de-posed. Shipping both would have cost 9.5 MB for a second
  de-posing of one assembly and no new coverage, so the older one is gone.
- `task2/cases/case2` — a **multi-solid semantic part**: 36 instances,
  37 solids, 20 part types, with `part_11.step` holding two solids that are
  placed together under one instance transform. It is here because
  `envs/common/score_asm.py` warns that over-splitting a multi-solid part
  wrecks the per-part numbers, and nothing in git exercised that until now.
  `avg_part` pairs `part_11` as one type, `n = 1`, `mean = 1.000000`.

Part ids and source geometries are preserved; whole supplied parts are de-posed
with `tools/asmlib.py` (seed 20260911) and the named reference leaves are
written by its `write_gt`. The contributor's audit reversed the rigid
transforms and measured zero symmetric-difference volume against the canonical
references. Only the English parts-table font and the PDF metadata are
normalised for publication.

Both were validated on macOS with the unmodified checker, deep:

```sh
uv run python tools/check_cases.py examples/task2/cases/case1 examples/task2/cases/case2 --deep
```

## (d) Evaluation configuration and the scoring scripts

| what | where |
|---|---|
| the evaluation contract, per task | `task<N>/eval.toml` — tool permissions, rounds, token cap, timeouts, submission format, the headline metric and its formula |
| the prompt the model is given | `task<N>/TASK.md`, byte-identical to `envs/<env>/TASK.md` |
| the task declaration it is derived from | `envs/<env>/task.toml` |
| the scorers | `envs/verifiers/part.py` (T1, T3), `envs/verifiers/assembly.py` (T2, T4, T5), `envs/verifiers/ecad.py` (T6) |
| the metric terms | `envs/common/part_metric.py`, `envs/common/asm_v1.py`, `envs/common/avg_part.py`, `envs/common/ecad_graph/` |
| the metric definitions | `docs/METRICS.md` — the contract; the verifier dispatches on the task's declaration, never on the task id or the case contents |
| score one submission | `tools/verify_case.py <case> <your.step>` (T1/T3) or `tools/verify_case.py <case> <your submission/ directory>` (T2/T4/T5, see `docs/CASE_FORMAT.md` "Submission layout") / `python -m envs.verifiers.ecad <case> <graph.json>` |
| run a model over cases | `harness/run.py --model <provider>/<id> --cases examples` |
| validate a case | `tools/check_cases.py <dir> --deep` |
| stage + score without a model | `tools/dryrun_case.py <case> --deep` |
| the task contracts themselves | `tools/check_tasks.py` |

`eval.toml` is generated, not written: every value in it is read out of the file
that owns it (`task.toml`, `envs/common/sandbox.py`, `envs/common/episode.py`,
`harness/run.py`), and `tests/test_dev_samples.py` compares it back against
those files. So it cannot quietly describe an evaluation that is not the one that
runs.

## (e) Formal evaluation

The formal bank is **not** shipped and neither are its answers. Two reasons, and
the second cannot be undone: the corpus contains purchased material with
unresolved licensing, and a heldout question is permanently spent once it has
been published — a miscomputed score can be recomputed, a question that has been
seen cannot be unseen.

The order of work is therefore:

1. **Align on these samples.** Reproduce `oracle.score` for all ten, then
   `round_trip.score` through `harness/run.py --model mock/oracle`. If those
   agree, your scoring chain is ours.
2. **Confirm the model / agent integration** on the samples with a real model:
   one case per task, default budget, and check the recorded episode — whether
   the answer arrived on its own or only under the forced final round
   (`submit_final` vs `submit_final_from_python` vs `no_answer`; the runner
   records which, and `no_answer` is kept out of every mean).
3. **Then the formal run**, on the bank, by arrangement: the case set, the split
   (`open` / `heldout` — mixing them contaminates, irreversibly), the number of
   rounds, and how many repeats. Ask for repeats: same-case spread was 0.67 at
   temperature 0.7, and a paired 8-vs-10-round comparison over 34 pairs gave a
   mean delta of −0.076 with a standard deviation of 0.303. A single run of a
   single configuration does not resolve an effect that size.

Before reading any number from a formal run, read "Reading the numbers" in the
repository `README.md`: raw means across tasks are not comparable (T4 measures
placement, T3 measures shape recovery), the distribution matters as much as the
mean, and some cases are declared unable to measure anything and are reported
separately rather than hidden.
