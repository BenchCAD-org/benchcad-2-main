# BenchCAD 2.1 Core with your own agent

The 100 BenchCAD 2.1 Core cases as Harbor tasks (`harbor` on PyPI, 0.20.0), for running
an agent of your choice (Claude Code, Codex, Gemini CLI, mini-swe-agent, OpenHands) instead of the
harness's own loop. The inputs, the task text, the answer contract and the scorer are 2.1's; what changes
is who drives the model.

## Rules

- **500,000 output tokens per case**, thinking included, every model call counted (subagents and side
  models too). The count is kept at the model API by a meter the agent's traffic passes through, never
  by the agent. As in 2.1 it is checked before each request, so a request made while budget remains runs
  to its end; once the case has spent its budget, every further request is refused and the agent ends.
  The `/work/answer.py` that exists when the agent ends is scored. Requests sent at the same time
  (parallel subagents) are each admitted against the balance at that moment; every row records how many
  were in flight at most and how far the total went past the budget (`max_in_flight`, `over_budget`).
- **No round cap.** The agent works however it likes until it stops or the budget is spent.
- **Network: the model API only**, through the meter, and of it only the generation, token-count and
  model-list paths (no files, batches or other services). Nothing else is reachable while the agent works,
  ICMP included.
- **8 h wall clock**, as a runaway backstop only; what exists at that point is scored.
- **No retries.** A retried trial would start a fresh budget; `benchcad.py` refuses `--max-retries`.
- **Leaderboard rows are labelled by agent and model** (e.g. `codex` + `openai/gpt-6-astra`, effort
  `xhigh`) and kept apart from the standard 2.1 `run_core` track: the two settings are not comparable.
- **Codex runs over HTTP.** Codex opens a Responses WebSocket by default, which the meter cannot read
  token usage from, so the `codex` agent here points Codex at the meter with WebSockets off.

## Run

Needs Docker, [uv](https://docs.astral.sh/uv/), Harbor 0.20.0 (`uv tool install harbor==0.20.0`), a
Hugging Face account with [Core access](https://benchcad.com/access.html) and your provider's key.

```bash
git clone https://github.com/BenchCAD-org/benchcad-2-main && cd benchcad-2-main && git checkout v2.1
export HF_TOKEN=...  ANTHROPIC_API_KEY=...
python3 own_agent/benchcad.py run --agent claude-code --model anthropic/claude-opus-5-5 --effort high
```

One command downloads Core at revision `v2.1` to the same place as `tools/run_core.sh` (every file a task
uses is checked against its `MANIFEST.sha256`), builds the tasks from this checkout in Docker, runs `harbor
run` with the agent pointed at each case's meter, collects the results, and then removes what holds the
gated Core files: the built tasks, and in each trial's copy of `/work` the staged inputs and every image or
PDF. What the agent wrote itself stays (`answer.py`, its modules, its own geometry). Each run is a new
Harbor job, one at a time per budget (a lock); `--dry-run` prints the plain Harbor commands.

| Flag | Meaning |
|---|---|
| `--agent` | one of the agents below |
| `--model` | `provider/model-id` |
| `--effort` | the agent's own effort setting (Harbor's `reasoning_effort`) |
| `--passes N` | trials per case (default 1); a case's passes are averaged |
| `--cases` | `task<N>/caseNNN,...`, for a subset (default: all 100) |
| `--n-concurrent N` | trials in flight (default 1); each takes 2 CPUs and 8 GB |
| `--budget` | default `500k`; anything else is a plumbing check, never reportable |
| `--keep-tasks` | keep the built tasks and inputs instead of removing them |
| `-- ...` | passed to `harbor run` unchanged |

## Agents

| `--agent` | Harbor agent | provider | the meter is its |
|---|---|---|---|
| `claude-code` | `claude-code` | `anthropic/` | `ANTHROPIC_BASE_URL` |
| `claude-code-bare` | `claude-code` with `--bare` | `anthropic/` | `ANTHROPIC_BASE_URL` |
| `codex` | `codex`, over HTTP (`agents/codex_http.py`) | `openai/` | its model provider |
| `gemini-cli` | `gemini-cli` | `gemini/` | `GOOGLE_GEMINI_BASE_URL` |
| `mini-swe-agent` | `mini-swe-agent` | `openai/` | `OPENAI_BASE_URL` |
| `openhands` | `openhands-sdk` (effort default `high`) | `openai/` | `LLM_BASE_URL` |

Keys come from the environment as each agent reads them: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY`, `LLM_API_KEY` (OpenHands). Another agent can be run with plain `harbor run` on the built
tasks, as long as all its model traffic goes to `http://meter:8080/anthropic`, `/openai/v1` or `/gemini`
over HTTP; the meter reads Anthropic Messages, OpenAI Responses and Chat Completions, and Gemini
`generateContent`, streamed or not.

## Results

The job folder (default `results/own_agent/<job>/`) gets:

- `results.jsonl`: one row per trial ([`result.schema.json`](result.schema.json)): the 2.1 cell (`scored`,
  `no_submission` = 0, or `pending` = no score), the meter's output-token count and whether the budget was
  hit, wall time, the agent's own token and cost report, tool calls, agent and version, model, effort,
  pass index, and the pins (data, harness and scorer digests);
- `summary.json`: the line, by 2.1's rule: a case's passes averaged, then the plain mean over the cases.
  Its `label` is `CORE` (all 100 cases scored), `SUBSET`, `INCOMPLETE` (a pending cell: no line until that
  case is run again) or `NOT REPORTABLE` (another budget, a meter not in front of the real APIs, or trials
  that do not all carry the 2.1 pins of one build).

A cell is pending when the trial has no result, the verifier could not finish, `/work` or the meter
record is missing, the meter could not read a reply's usage, or the agent ended on a provider error it did
not get past (a 429 or 5xx: 2.1 retries those calls; here the case is run again, a new trial, never a
Harbor retry).

To report a score, send `summary.json` and `results.jsonl` of a `CORE` run to benchcad.team@gmail.com,
naming the agent, its version, the model and the effort. Keep the job folder: the trajectories, the
meter's request log and the verifier's records are what an audit looks at.

`python3 own_agent/collect.py JOB [JOB ...] --out results.jsonl --summary summary.json` rebuilds both from
Harbor job folders, e.g. after several runs into different jobs.

## How a task works

| | |
|---|---|
| agent sees | `/task`: the case's inputs as the 2.1 sandbox stages them (drawings rendered to sheets, tiles and `hires/` masters), with the 2.1 `tools.py`, read-only; `/work`: a copy, the working directory |
| agent image | the 2.1 sandbox image (Python 3.12, cadquery 2.3.0, the pinned set) plus curl, git, procps and tmux for an agent's own setup |
| instruction | the 2.1 system prompt for the case, written by the harness's own functions, with the round protocol replaced by the answer contract |
| answer | `/work/answer.py`, as a 2.1 `submit`: a CadQuery program leaving `result` (T1, T3), a program writing `submission/` with the tools (T2, T4, T5), a program leaving the graph in `result` (T6) |
| meter | its own compose service and the agent's only way out (Harbor's egress allowlist names it alone): `/anthropic`, `/openai` and `/gemini` go to the providers' public APIs; `curl http://meter:8080/_meter/status` shows what is spent |
| limits | 2 CPUs, 8 GB, no GPU |
| verifier | a separate container the agent never sees, with the reference and the 2.1 scorer in its locked environment: it runs `answer.py` once in a copy of `/work` (600 s, one CPU, no network) and scores what it leaves as `harness/run.py` does |

Each task pins the scorer twice (`tests/pins.json`: the digest of the scorer tree it was built with, and
the 2.1 task-6 digest `c4799cc2…`) and the case's own digest; the verifier checks all three and gives no
score on a mismatch. The answer runs as an unprivileged user that cannot read the reference; what it
leaves is copied out by that same user (links followed, regular files only) and only that copy is scored.

Differences from a `run_core` run, all outside the scorer: the input images are named in the instruction
for the agent to open, not attached to the first turn; the answer run's memory cap is 6 GiB of address
space instead of a 2 GB container limit.

`own_agent/run_build.sh DATA OUT [--cases ...]` builds the tasks without running them; `harbor run -p OUT
-a oracle` scores every reference answer (1.0) and `-a nop` scores no answer (0).

## Checks

- `tests/test_own_agent.py` (offline, part of the suite): the meter in front of a stub model API for every
  provider shape, streamed and not; the budget check and the refusals; WebSocket refusal; unreadable usage;
  `collect.py`'s rows and line; the wrapper's commands; the builder on the test fixtures.
- The reference answers score 1.0 on six Core cases, one per task, and no answer scores 0 on the same six,
  through Harbor 0.20.0.
- The five agent CLIs, each against a stub model API with dummy keys (`metertest/cli_tests.sh`): every one
  sent its model traffic through the meter, every reply's usage was read, and each stopped once refused.
  A meter test task (`metertest/make_metertest.sh`) showed that nothing but the meter is reachable from the
  agent and that the answer written before the budget ran out is scored.
- One real-model plumbing check with a 5,000-token budget, where the meter's output-token count matched
  the agent's own report.
