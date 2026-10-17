# Releases

One JSON per release, `benchcad-<release>.json`, pins what a score from that release
ran on:
- the code tag and its commit;
- the harness version every record carries;
- the Hugging Face dataset revisions;
- the T6 scorer digest;
- the limits.

| Version | Status | Code tag | Datasets (Hugging Face revision) | Release |
|---|---|---|---|---|
| 2.1-beta | pre-release, recommended | [`v2.1-beta`](https://github.com/BenchCAD-org/benchcad-2-main/tree/v2.1-beta) | Core `v2.1-beta` (`d159834275`), Public `v2.1-beta` (`4b5500320a`) | [BenchCAD 2.1-beta](https://github.com/BenchCAD-org/benchcad-2-main/releases/tag/v2.1-beta) |
| 2.0 | stable; reproduces the published lines | [`v2.0-core1`](https://github.com/BenchCAD-org/benchcad-2-main/tree/v2.0-core1) | Core `v2.0` (`d159834275`); Public not part of 2.0 | [BenchCAD 2.0](https://github.com/BenchCAD-org/benchcad-2-main/releases/tag/v2.0-core1) |

The Core data is the same commit in both releases. The harness settings differ:

| | 2.0 | 2.1-beta |
|---|---|---|
| rounds | 30 | 40 |
| output tokens per case | no budget | 500,000, thinking included |
| reasoning carried across rounds | no (text only) | yes |
| context summarised at | 180k tokens | 800k tokens |

Scores from different releases are not comparable. `reference_lines` names the
published lines run on a release: `results/core_reference.json` on `main`, all from 2.0.
2.1-beta has none yet.
