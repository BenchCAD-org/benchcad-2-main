#!/usr/bin/env bash
# Run the effort x rep matrix of harness/run.py on one machine, resumably.
#
#   tools/matrix.sh <model> <cases> <shard k/n> <workers per effort> <rep>...
#
#   tools/matrix.sh openai/gpt-5.5 bank 0/2 8 0 1 2 3 4     # this machine takes every
#                                                          # 2nd case from 0; 5 reps
#
# One rep at a time; within a rep the five efforts run as five processes side
# by side (so 5 x <workers> episodes are in flight and no effort's tail leaves
# workers idle), each writing results/<tag>_<effort>_r<rep>_s<k>.json and a
# log next to it. Every process runs with --resume, so re-running the same
# command after a crash or a reboot picks up where it stopped. Reps are
# ordered outermost so that stopping early leaves complete efforts for the
# reps that finished.
set -u
if [ $# -lt 5 ]; then sed -n 2,16p "$0"; exit 2; fi
MODEL=$1; CASES=$2; SHARD=$3; WORKERS=$4; shift 4
EFFORTS=${EFFORTS:-"none low medium high max"}
ROUNDS=${ROUNDS:-100}
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
            --workers "$WORKERS" --resume --out "$OUT" --work "$WORK" \
            > "results/logs/${TAG}_${E}_r${REP}_s${K}.log" 2>&1 &
        pids+=($!)
    done
    wait "${pids[@]}"
    echo "[$(date +%F' '%T)] rep $REP done"
done
