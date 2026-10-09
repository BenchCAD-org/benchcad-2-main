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

Needs Docker and a Hugging Face account with Core access
([request it here](https://benchcad.com/access.html)). Docker on Linux:
`curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER`, then log in
again (on a freshly booted box, apt may hold its lock for a minute). The scorers also need
`libgl1` (`sudo apt-get install -y libgl1`); `run_core.sh` installs it when it can.

```bash
git clone https://github.com/BenchCAD-org/benchcad-2-main && cd benchcad-2-main
export HF_TOKEN=...  GEMINI_API_KEY=...      # or ANTHROPIC_API_KEY / OPENAI_API_KEY
tools/run_core.sh --model gemini/<model-id> --effort low,medium,high
```

An Anthropic organization-level key also needs `ANTHROPIC_WORKSPACE_ID`.

That one command installs the environment, builds the sandbox, downloads and
verifies the dataset, runs all 100 cases, scores them and prints the result.
Run it again to resume. The summary is written to `results/core/<model>/core_summary.json`.

| Flag | Meaning |
|---|---|
| `--model` | `anthropic/<id>`, `openai/<id>`, `gemini/<id>`, `xai/<id>`, `openrouter/<id>` (**required**) |
| `--effort` | comma-separated, e.g. `low,medium,high`; one run each (default: the provider's ladder) |
| `--rounds` | default `30`, the published setting; anything else is labelled a smoke run |
| `--cases` | a task or case directory inside the dataset, for a subset; runs add up in one results file, and the table says `SUBSET n/100` until all cases are in |
| `--workers` | episodes in parallel (default: from a memory budget, printed at the start) |
| `--dry-run` | set everything up and check keys, dataset access and the scorers; no API calls |

**Gemini.** `--effort low|medium|high` is sent as `thinking_level`. Use `GEMINI_API_KEY`,
or Vertex AI with `GOOGLE_GENAI_USE_VERTEXAI=true` plus an express key in `GOOGLE_API_KEY`.
`GEMINI_BASE_URL` points at another endpoint. The adapter was smoke-tested on
`gemini-3.8-flash`; please check model-specific details (levels, endpoint, auth) on
your side, and report `effort_sent`, `auth`, the model id, rounds and rep with the numbers.

## Try the public examples first

Nine samples ship in `examples/`, no download needed:

```bash
uv run python harness/run.py --model mock/oracle --cases examples      # every score should be 1.0
uv run python harness/run.py --model gemini/<model-id> --cases examples/task3 --effort low
```

## Reporting a score

Send `results/core/<model>/core_summary.json` from a run at the published settings
(30 rounds, rep 0, all 100 cases resolved). A run with pending cases or other rounds is
marked INCOMPLETE or SMOKE and is not a reportable score.

## Dataset

Core, the 100 cases behind the published scores, is available under an agreement:
[benchcad.com/access.html](https://benchcad.com/access.html), then
[`BenchCAD/benchcad-2.0-core`](https://huggingface.co/datasets/BenchCAD/benchcad-2.0-core).
T1 6, T2 11, T3 28, T4 25, T5 18, T6 12. `run_core.sh` checks its `MANIFEST.sha256`
and its version before every run.

## More

- `docs/RUNNING.md`: the harness in detail (providers, flags, clocks, your own agent).
- `docs/METRICS.md`: every metric; `docs/CASE_FORMAT.md`: a case and a submission.
- `envs/<task>/TASK.md`: the prompt each task gives the model.

## Contact

benchcad.team@gmail.com
