# Releases

One JSON per release, `benchcad-<release>.json`, pins what a score from that release
ran on:
- the code tag and its commit;
- the harness version every record carries;
- the Hugging Face dataset revisions;
- the T6 scorer digest;
- the limits.

A release's own JSON can't hold the commit of the tag it ships in. Until the tag is made,
its `status` is `draft` and its `commit` is `null`; `main` fills both in afterwards.

| Version | Status | Code tag | Datasets (Hugging Face revision) | Release |
|---|---|---|---|---|
| 2.1 | stable, recommended | [`v2.1`](https://github.com/BenchCAD-org/benchcad-2-main/tree/v2.1) | Core `v2.1` (`b7a2ec787d`), Public `v2.1` (`1e4cfa4fe7`) | [BenchCAD 2.1](https://github.com/BenchCAD-org/benchcad-2-main/releases/tag/v2.1) |
| 2.1-beta | pre-release, superseded | [`v2.1-beta`](https://github.com/BenchCAD-org/benchcad-2-main/tree/v2.1-beta) | Core `v2.1-beta` (`d159834275`), Public `v2.1-beta` (`4b5500320a`) | [BenchCAD 2.1-beta](https://github.com/BenchCAD-org/benchcad-2-main/releases/tag/v2.1-beta) |
| 2.0 | stable; reproduces the published lines | [`v2.0-core1`](https://github.com/BenchCAD-org/benchcad-2-main/tree/v2.0-core1) | Core `v2.0` (`d159834275`); Public not part of 2.0 | [BenchCAD 2.0](https://github.com/BenchCAD-org/benchcad-2-main/releases/tag/v2.0-core1) |

From 2.1 on, `tools/run_core.sh` at a release's tag downloads that release's
dataset revision (`--revision`), never `main`. The `v2.0-core1` and `v2.1-beta` checkouts
predate this and download `main`, which still holds exactly the data their revisions name.

The case files are the same in every release. 2.1 rebinds the task-6 boards to the 2.1 scorer
(each `gt/correspondence.json` names its digest, with its hash in `case.json`, `dataset.json` and
`MANIFEST.sha256`); nothing else in the data changes. The harness settings differ:

| | 2.0 | 2.1-beta | 2.1 |
|---|---|---|---|
| rounds | 30 | 40 | 60 |
| output tokens per case | no budget | 500,000, thinking included | 500,000, thinking included |
| the case ends at | a submit, or the rounds | a submit, the rounds or the budget | a submit, the rounds or the budget |
| the prompt says no submit scores 0 | no | no | yes |
| a discarded, re-asked reply counts against the budget | — | yes | no |
| reasoning carried across rounds | no (text only) | yes | yes |
| context summarised at | 180k tokens | 800k tokens | 800k tokens |
| T6 scorer digest | `f4dd1410…` | `f4dd1410…` | `df63b759…` (rule B: a prediction pairs only with a part no other part is more than 0.005 nearer to) |

Scores from different releases are not comparable. `reference_lines` names the
published lines run on a release: `results/core_reference.json` on `main`, all from 2.0.
2.1 has none yet.
