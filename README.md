# BenchCAD 2

Six CAD reconstruction tasks with executable scorers, and nine development
samples. A model works in a sandboxed directory, one Python program per round,
and hands in geometry (or a netlist); the scorer compares it with the reference
by voxels, surfaces and renders. No LLM judge anywhere.

| task | the model gets | it hands in | headline |
|---|---|---|---|
| T1 | one part drawing: the sheet as PNG, with legible tiles of it on disk to open on demand | the part, a CadQuery solid | `part_v1`, orientation free |
| T2 | every part as STEP + the assembly drawing (sheet shown, tiles on disk) + `bom.json` | the assembly: the supplied parts placed | `asm_v1` |
| T3 | four rendered views of a part, no 3-D | the part | `part_v1`, orientation pinned |
| T4 | four views of a mechanism + two sheets per part (alone / in place), no 3-D | every part **and** the assembly | `avg_part × asm_v1` |
| T5 | five part drawings + sixteen supplied STEPs + the assembly drawing | the five parts **and** the assembly | `avg_part × asm_v1` |
| T6 | six renders of an assembled PCB | the terminal-net graph | `ecad_v2` |

`part_v1 = 0.5 iou_term + 0.3 surf_f1 + 0.2 pix_fg`; `asm_v1` is a per-part-type
leave-one-out voxel IoU gain; `avg_part` is `part_v1` of each submitted part against
its reference part. Every term and every headline is in [0, 1]. `docs/METRICS.md`
defines them; `docs/CASE_FORMAT.md` defines a case and a submission; each task's
prompt is `envs/<task>/TASK.md`.

## Requirements

- Python 3.12 via [uv](https://docs.astral.sh/uv/) (`uv sync` installs cadquery, OCP, vtk, ...).
- Docker for the sandbox the model's code runs in (on macOS: Docker Desktop or
  colima with a few GB). The image is built once from `sandbox/`; on an x86
  host tag it as you like and point `CADENV_DOCKER_IMAGE` at it. Without Docker,
  `CADENV_LOCAL=1` runs the model's code as a plain subprocess — fine for a look,
  not for a run you report.
- An API key for the provider you run: `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`,
  `OPENAI_API_KEY`, `GEMINI_API_KEY` or `XAI_API_KEY`.

```sh
uv sync --group harness --group dev
docker build -t benchcad-sandbox:arm64 sandbox/     # no network inside, 2 GB, 1 CPU
```

## Score in one command

The nine development samples are in `examples/` -- `examples/task1` ...
`examples/task6`, one or two cases each -- and a first run points there:

```sh
export OPENAI_API_KEY=...             # or ANTHROPIC_API_KEY
uv run python harness/run.py --model openai/gpt-6-astra --cases examples --out results/astra.json
uv run python harness/run.py --model anthropic/claude-opus-5 --cases examples --out results/opus5.json
uv run python tools/summarize.py results/astra.json results/opus5.json
```

`--cases` takes any directory and finds every case under it, so `examples` is
all nine samples, `examples/task3` one task, `examples/task3/cases/case1` one
case, and a path into your own case tree works the same way. The defaults are
the benchmark's contract: **30 rounds per case** (`--rounds` changes it) at the
provider's top effort (`--effort` changes it), no cap on a reply (the model's
own maximum output; `--max-tokens` sets one), every image the episode produced
kept in the conversation, the observation text limited the way Terminus 2
limits it (10,000 bytes, first and last halves), and context summarised the way
Terminus 2 summarises it when the model's window fills. An episode ends when
the model submits, or with no answer when its rounds run out -- nothing is
asked on its behalf. The run writes one JSON with a record per case (headline
`score`, the diagnostic columns, tokens, seconds), and `work/<run>/` keeps
every case's working directory and transcript; `summarize.py` prints the
per-case table, the mean per task and the mean of the task means, with the
count of scored cases next to every mean.

`--effort` follows the model: OpenAI takes `none | low | medium | high | xhigh`
(`reasoning.effort` on the Responses API, which is what OpenAI's own endpoint
is spoken through), Anthropic `none | low | medium | high | xhigh | max`
(`none` is thinking disabled; the rest is `output_config.effort` with adaptive
thinking), OpenRouter `none | low | medium | high` (`reasoning.effort` on
`/chat/completions`, which the gateways keep). A level the provider does not
have is refused at startup -- `max` on OpenAI is an error, not xhigh -- and
with no `--effort` the run uses the provider's top level (OpenAI xhigh,
Anthropic max), prints it and records it in the results. A reply's ceiling is
128k tokens on both first-party endpoints, thinking included (gpt-6-astra's
maximum and the Claude 5 family's; Anthropic's read from the Models API);
`--max-tokens` sets a lower one. The context window is 1M on both (gpt-6-astra
1.05M; the Claude 5 family's `max_input_tokens`), which is what the context
summarisation works against; `--context-tokens` overrides. Both adapters share
the clocks -- a 60 s idle limit on a stream (both carry reasoning progress while
the model thinks), a wall budget per call of 300 s up to `high` and 600 s at
`xhigh` / `max` past which the round is discarded and the model told in one
sentence -- the retry rules (a 429 waits for the endpoint's own hint; a silent
stream is re-requested three times), the image handling (every image kept; over
20 images, 2000 px each) and prompt caching (OpenAI's automatic prefix cache,
Anthropic's top-level `cache_control`); every usage record carries the call's
seconds. Scoring runs in child processes from a separate pool, so a slow score
never holds an episode. Images go to OpenAI at `detail: high`. Every provider
speaks the same text-and-images protocol, so scores compare across vendors.

Runs of any size:

```sh
uv run python harness/run.py --model anthropic/claude-opus-5 --cases bank --effort high \
    --workers 6 --max-execs 3 --rep 0 --shard 0/2 --out results/opus5_high_r0_s0.json --resume
```

`--workers N` runs N episodes at once in threads (an episode waits on the API
or on its sandbox nearly all of the time). `--max-execs M` caps the sandbox
executions running at once in that process, whatever N is; each execution
may take 2 GB, so M is what the Docker host has to hold. `--rep k` is
the repetition index, recorded in every record and in the work-dir name so
reps of one case can run side by side. `--shard k/n` gives this process every
n-th case starting at k, so machines split one case list without a queue.
`--resume` re-reads `--out` and skips cases already scored there, re-running
only errors, so a run interrupted anywhere continues with the same command.
`tools/summarize.py` takes any number of result files.

Budget for the full contract: a part case is tens of thousands of input
tokens per round (cached after round one); an assembly case starts at 12–20
images per round. Reckon on a few dollars per part case and around ten per
assembly case at 30 rounds.

## Check your setup before you spend anything

```sh
uv run python harness/run.py --model mock/oracle --cases examples --rounds 1
```

`mock/oracle` hands in the reference answer the way each prompt asks for it
(parts and an instances file on the assembly tasks, a solid or a graph on the
others) and must score 1.0 on every sample; `mock/dumb` hands in a box. Each
case's `expected.json` records those numbers and the commit they were measured
on. Two more checks:

```sh
uv run python tools/check_cases.py examples/task*/cases/* --deep    # the data is well formed
uv run python tools/verify_case.py examples/task1/cases/case1 your.step   # score one answer of yours
```

## Running your own agent

The harness is one file, `harness/run.py`: a provider turns the episode's
`(system, turns)` into a reply, nothing else is provider-specific. To plug in
another model or your own agent loop, add a provider there or drive
`envs.common.episode.run_episode` yourself with a `call_fn(system, turns) -> str`.
The interface a model sees — the prompt, the tools, the round protocol — is
`envs/common/episode.py` and `envs/common/sandbox.py`; the tools it can call are
`export`, `export_part`, `use_part`, `submit_assembly` and `crop`.

## These nine cases are development samples, not the formal bank

The formal cases and their answers are not in this repository and are not
distributed. Align on these samples first; the model or agent integration and
the formal run are agreed separately.
