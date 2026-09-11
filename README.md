# BenchCAD 2

Six CAD reconstruction tasks with their scorers, and nine development samples.

| task | input | deliverable | headline score |
|---|---|---|---|
| T1 | one part drawing (PDF, rasterised at stage time) | the part | `part_v1`, free orientation |
| T2 | every part as STEP + the assembly drawing | the assembly | `asm_v1` |
| T3 | four reference views, no 3-D | the part | `part_v1`, pinned orientation |
| T4 | four views + a per-part highlight sheet, no 3-D | every part **and** the assembly | `avg_part x asm_v1` |
| T5 | five part drawings + sixteen supplied STEPs | the five parts **and** the assembly | `avg_part x asm_v1` |
| T6 | six renders of an assembled PCB | the terminal-net graph | `ecad_v2` |

`part_v1 = 0.40 iou_term + 0.35 surf_f1(tau=0.02) + 0.25 pix_fg`, every term and
every headline in [0, 1]. `docs/METRICS.md` defines all of them; `docs/CASE_FORMAT.md`
defines a case and a submission.

## Install

```sh
uv sync --group harness --group dev
docker build -t benchcad-sandbox:arm64 sandbox/     # no network inside, 2 GB, 1 CPU
```

## Run everything, no API key needed

```sh
uv run python harness/run.py --model mock/oracle --cases examples --rounds 1
```

Every sample scores its `expected.json` `round_trip` value. Then a real model:

```sh
ANTHROPIC_API_KEY=... uv run python harness/run.py \
    --model anthropic/claude-opus-5 --cases examples --rounds 100 --effort max
```
Defaults are 100 rounds at effort max. `openrouter/<id>` (OPENROUTER_API_KEY),
`openai/<id>`, `gemini/<id>` and `xai/<id>` also work.

## Checking your setup against ours

Each case ships an `expected.json` with three numbers and the commit they were
measured on: `oracle` (the reference submitted as the answer — the top of the
scale), `round_trip` (the same reference after a STEP round trip, which is what
a submitted program necessarily produces) and `baseline` (a trivial answer that
knows only the overall size). Reproduce `round_trip` and your scoring chain
matches ours.

```sh
uv run python tools/check_cases.py examples/task*/cases/* --deep
uv run python tools/verify_case.py examples/task1/cases/case1 your.step
```

## These nine cases are development samples, not the formal bank

The formal cases and their answers are not in this repository and are not
distributed. Align on these samples first; the model or agent integration and
the formal run are agreed separately.
