#!/usr/bin/env bash
# Run the effort x rep matrix of harness/run.py on one machine, resumably.
#
#   tools/matrix.sh <model> <cases> <shard k/n> <workers per effort> <rep>...
#
#   tools/matrix.sh anthropic/claude-opus-5 bank 0/1 3 0 1 2 3 4   # one machine, all
#                                                                  # cases, 5 x 3 episodes
#                                                                  # in flight, 5 reps
#   MAX_EXECS=2 tools/matrix.sh ...    # per effort process, at most this many sandbox
#                                      # executions at once (harness --max-execs)
#
# One rep at a time; within a rep the five efforts run as five processes side
# by side (so 5 x <workers> episodes are in flight and no effort's tail leaves
# workers idle), each writing results/<tag>_<effort>_r<rep>_s<k>.json and a
# log next to it. EFFORTS follows the model: every level is sent verbatim and
# harness/run.py refuses one the provider lacks, so an Anthropic or a
# gpt-6-astra matrix is EFFORTS="low medium high xhigh max" (the default) and a
# gpt-5.5 matrix is EFFORTS="low medium high xhigh" (no max there). EFFORTS="none ..."
# adds a thinking-off run (OpenAI reasoning_effort none -- not on gpt-6-astra --,
# Anthropic thinking disabled).
# ROUNDS defaults to the harness's own default (30). Every
# process runs with --resume, so re-running the same command after a crash or
# a reboot picks up where it stopped. Reps are ordered outermost so that
# stopping early leaves complete efforts for the reps that finished.
set -u
if [ $# -lt 5 ]; then sed -n 2,22p "$0"; exit 2; fi
MODEL=$1; CASES=$2; SHARD=$3; WORKERS=$4; shift 4
EFFORTS=${EFFORTS:-"low medium high xhigh max"}
ROUNDS=${ROUNDS:-30}
MAX_EXECS=${MAX_EXECS:-0}
TAG=$(echo "$MODEL" | tr '/:' '__')
K=${SHARD%%/*}
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/logs
for REP in "$@"; do
    echo "[$(date +%F' '%T)] rep $REP: efforts $EFFORTS, shard $SHARD, $WORKERS workers each"
    pids=()
    for E in $EFFORTS; do
        OUT="results/${TAG}_${E}_r${REP}_s${K}.json"
        WORK="$HOME/cad-agent-work/${TAG}_${E}_r${REP}_s${K}"
        uv run --no-sync python -u harness/run.py --model "$MODEL" --cases "$CASES" \
            --effort "$E" --rounds "$ROUNDS" --rep "$REP" --shard "$SHARD" \
            --workers "$WORKERS" --max-execs "$MAX_EXECS" --resume --out "$OUT" --work "$WORK" \
            > "results/logs/${TAG}_${E}_r${REP}_s${K}.log" 2>&1 &
        pids+=($!)
    done
    wait "${pids[@]}"
    echo "[$(date +%F' '%T)] rep $REP done"
done
