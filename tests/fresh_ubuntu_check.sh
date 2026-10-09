#!/usr/bin/env bash
# The fresh-box check: tools/run_core.sh --dry-run in a bare ubuntu:24.04 container, as a
# lab's first command on a new machine. Needs only Docker on the host.
#
#   tests/fresh_ubuntu_check.sh [<core data dir>]
#
# Inside: curl and git only (what a fresh cloud image has), then run_core.sh installs uv,
# syncs, finds OCP cannot load (libGL.so.1), installs libgl1, passes the scorer
# self-test, and -- given a data dir -- runs the mock oracle over the examples. A `docker`
# stub stands in for the host's Docker during the dry run; the examples then run in a local
# sandbox inside the throwaway container (no stub).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-}
TMP=$(mktemp -d)
git archive HEAD | tar -x -C "$TMP"
mkdir -p "$TMP/.stub" && printf '#!/bin/sh\nexit 0\n' > "$TMP/.stub/docker" && chmod +x "$TMP/.stub/docker"
docker run --rm -v "$TMP:/src:ro" ${DATA:+-v "$DATA:/data:ro"} ubuntu:24.04 bash -c '
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq curl ca-certificates git >/dev/null 2>&1
    cp -a /src /repo && cd /repo && export PATH=/repo/.stub:$PATH
    if [ -d /data ]; then
        tools/run_core.sh --model mock/oracle --data /data --dry-run
        echo "== mock oracle over the examples, local sandbox inside this throwaway container"
        export PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; CADENV_LOCAL=1 uv run --no-sync python harness/run.py --model mock/oracle --cases examples --rounds 1 \
            --unsafe-local --out /tmp/o.json --work /tmp/w | grep -E "^\s+\[[0-9]+/[0-9]+\]" | cut -c1-90
    else
        tools/run_core.sh --model mock/oracle --dry-run || true    # stops at the dataset without HF_TOKEN
    fi'
rm -rf "$TMP"
