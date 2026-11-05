#!/bin/bash
# The meter with each real agent CLI, against the stub API (stub_compose.yaml): no model is called, nothing spent.
#   own_agent/metertest/cli_tests.sh JOBS_DIR STUBLOG_DIR [agent ...]     (default: the five the wrapper runs)
# Each agent runs one case (CASE=task<N>/caseNNN, default the download's first task-1 case) through
# benchcad.py (its own base-URL and key variables, dummy keys), with the verifier off and a 12 min agent limit. cli_report.py then says, per agent: did its model traffic reach the
# meter, was every reply's usage read, was it refused after the budget, and did it stop.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
jobs=$(mkdir -p "$1" && cd "$1" && pwd); logs=$(mkdir -p "$2" && cd "$2" && pwd); shift 2
agents=("$@"); [ ${#agents[@]} -gt 0 ] || agents=(claude-code-bare codex gemini-cli mini-swe-agent openhands)
one=${CASE:-task1/$(ls "${BENCHCAD_DATA:-$HOME/.cache/benchcad/benchcad-2.0-core}/task1/cases" | head -1)}
model_of() {   # bash 3 (macOS) has no associative arrays
  case "$1" in
    claude-code|claude-code-bare) echo anthropic/claude-opus-5-5 ;;
    gemini-cli) echo gemini/gemini-2.5-pro ;;
    *) echo openai/gpt-6-astra ;;
  esac
}
export STUB_CONTEXT="$here/stubapi" ANTHROPIC_API_KEY=stub-key OPENAI_API_KEY=stub-key GEMINI_API_KEY=stub-key LLM_API_KEY=stub-key
for a in "${agents[@]}"; do
  mkdir -p "$logs/$a"; rm -f "$logs/$a/requests.jsonl"
  echo "== $a start $(date -u +%FT%TZ)"
  STUB_LOG_DIR="$logs/$a" python3 "$here/../benchcad.py" run --agent "$a" --model "$(model_of "$a")" \
      --cases "$one" --jobs-dir "$jobs" --job-name "cli-$a" -- \
      --extra-docker-compose "$here/stub_compose.yaml" --disable-verification --agent-timeout-multiplier 0.025 --override-memory-mb 4096
  echo "== $a rc=$? $(date -u +%FT%TZ)"
done
python3 "$here/cli_report.py" "$jobs" "$logs" "${agents[@]}"
