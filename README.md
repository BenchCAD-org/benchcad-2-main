<div align="center">

# BenchCAD 2.0

**Can a model build real CAD from drawings, photos and parts?**

[![Website](https://img.shields.io/badge/🌐%20Website-benchcad.com-2ea44f.svg)](https://benchcad.com)
[![Leaderboard](https://img.shields.io/badge/🏆%20Leaderboard-view-orange.svg)](https://benchcad.com/leaderboard.html)
[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-benchcad--2.0--core-yellow.svg)](https://huggingface.co/datasets/BenchCAD/benchcad-2.0-core)
[![Access](https://img.shields.io/badge/Core%20access-request-blue.svg)](https://benchcad.com/access.html)

</div>

---

Six CAD tasks, 100 Core cases. The model works in a sandbox, writes one Python
program per round, and hands in geometry or a schematic. Scoring is executed
geometry against the reference, deterministic, with no LLM judge.

| Task | Input → Output | Metric |
|---|---|---|
| **T1** | a part drawing → the part (CadQuery) | `part_v1` |
| **T2** | parts (STEP) + assembly drawing → the assembly | `asm_v1` |
| **T3** | four rendered views → the part | `part_v1` |
| **T4** | views of a mechanism → every part + the assembly | `avg_part × asm_v1` |
| **T5** | part drawings + parts + assembly drawing → the assembly | `avg_part × asm_v1` |
| **T6** | photos of a PCB → its schematic | `ecad_v2` |

## Quick start

Needs Docker, `libgl1`, and a Hugging Face account with
[Core access](https://benchcad.com/access.html).

```bash
git clone https://github.com/BenchCAD-org/benchcad-2-main && cd benchcad-2-main
export HF_TOKEN=...  GEMINI_API_KEY=...      # or ANTHROPIC_API_KEY / OPENAI_API_KEY
tools/run_core.sh --model gemini/<model-id> --effort low,medium,high
```

One command installs the environment, builds the sandbox, downloads and verifies the
dataset, runs all 100 cases and scores them. Run it again to resume. The summary is
`results/core/<model>/core_summary.json`; send that file to report a score.

| Flag | Meaning |
|---|---|
| `--model` | `anthropic/<id>`, `openai/<id>`, `gemini/<id>`, `xai/<id>`, `openrouter/<id>` |
| `--effort` | comma-separated levels, one run each (default: the provider's full ladder) |
| `--rounds` | default none, the published setting: the token budget ends a case; `--rounds N` caps the rounds, a smoke run |
| `--token-budget` | output tokens per case, thinking included; default `500000`, the published setting (`0` = none); anything else is a smoke run |
| `--cases` | a task or case directory, for a subset |
| `--set public` | run the Public set instead of Core |
| `--dry-run` | check keys, dataset access and scorers; no API calls |

How each provider's effort level is sent, auth options and the other flags:
[`docs/RUNNING.md`](docs/RUNNING.md).

## Versions

| Version | Tag | Use |
|---|---|---|
| **2.1-beta.2** | [`v2.1-beta.2`](https://github.com/BenchCAD-org/benchcad-2-main/releases/tag/v2.1-beta.2) | Recommended. Reasoning carried across rounds; 500k output tokens per case and no round limit, provisional. Core data unchanged. |
| 2.1-beta | [`v2.1-beta`](https://github.com/BenchCAD-org/benchcad-2-main/releases/tag/v2.1-beta) | Superseded: reasoning carried; 40 rounds and 500k output tokens per case. |
| 2.0 | [`v2.0-core1`](https://github.com/BenchCAD-org/benchcad-2-main/releases/tag/v2.0-core1) | Only to reproduce the published reference lines (30 rounds, thinking not carried). |

What each version pins (code commit, dataset revisions, scorer digest, limits):
[`releases/`](releases/README.md).

Don't mix versions or run from `main` for a reported score; check out a tag
(`git checkout v2.1-beta.2`). Every record and `core_summary.json` carry `harness_version`;
scores from different versions are not comparable.

## Public set

[`BenchCAD/benchcad-2.0-public`](https://huggingface.co/datasets/BenchCAD/benchcad-2.0-public):
100 more cases, same tasks and scorers, no overlap with Core. Accept its terms once
(one click, no review), then add `--set public`. Its scores are for trying the harness,
not for reporting. Nine samples also ship in `examples/`, no download needed.

## Rounds

500k output tokens per case, thinking included, and no round limit: the case ends at a `submit`
or once the budget is spent, checked between calls. (Since 2026-10-18, 2.1-beta.2; 2.1-beta: 40
rounds and 500k, the first to run out; 2.0, before 2026-10-16: 30 rounds, no token budget.) Each
observation starts `Round N · output tokens spent / 500,000`. A case that ends without a `submit` has no answer. A call that fails for infrastructure reasons (network, server error,
stalled stream) is retried and does not count as a round. (Changed 2026-10-12.)

Each round the model gets its earlier reasoning back, as its provider returned it:
Anthropic's thinking blocks with their signatures, OpenAI's encrypted reasoning items,
Gemini's thought signatures, OpenRouter's `reasoning_details`. (Changed 2026-10-15; before,
only the text went back.)

## Reproducing the published lines

`results/core_reference.json` holds the published lines: harness 2.0, 30 rounds, all 100 Core cases,
run before the 2026-10-15 change above -- a different setting from today's 500k tokens with no round limit.
Claude lines are one rep; `openai/gpt-6-astra` is the mean of 3 reps (T6: 1 rep), with
the spread in `noise_note`. `run_core` prints the published line beside yours. To rerun
them in their own setting, check out `v2.0-core1` first:

```bash
git checkout v2.0-core1
tools/run_core.sh --model anthropic/claude-opus-5-5     # ANTHROPIC_API_KEY
tools/run_core.sh --model openai/gpt-6-astra            # OPENAI_API_KEY
```

## More

- `docs/RUNNING.md`: providers, flags, clocks, running your own agent.
- `docs/METRICS.md`: every metric. `docs/CASE_FORMAT.md`: a case and a submission.
- `envs/<task>/TASK.md`: the prompt each task gives the model.

Contact: benchcad.team@gmail.com
