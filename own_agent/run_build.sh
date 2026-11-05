#!/bin/bash
# Build the task folders in the builder image (no network, inputs read-only):
#   own_agent/run_build.sh DATA OUT [--cases task<N>/caseNNN,...] [--budget N]
# DATA: the v2.1 Core download; OUT: an empty directory. The 2.1 harness is this checkout.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
data=$(cd "$1" && pwd); mkdir -p "$2"; out=$(cd "$2" && pwd); shift 2
ctx=$(mktemp -d); trap 'rm -rf "$ctx"' EXIT
cp "$root/own_agent/builder/Dockerfile" "$root/pyproject.toml" "$root/uv.lock" "$ctx"/
docker build -q -t benchcad21-own-builder "$ctx" >/dev/null
# a linked worktree's .git names the main repository's git directory by its host path: mount that too, there
common=$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)
gitmount=(); case "$common" in "$root"/*) ;; *) gitmount=(-v "$common:$common:ro") ;; esac
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$data:/data:ro" -v "$root:/repo:ro" ${gitmount[@]+"${gitmount[@]}"} -v "$out:/out" \
  benchcad21-own-builder /opt/scorer/bin/python /repo/own_agent/build.py --data /data --out /out "$@"
