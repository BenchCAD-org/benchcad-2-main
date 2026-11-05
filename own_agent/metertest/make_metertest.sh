#!/bin/bash
# The meter test task: a built task with a stub Anthropic API behind its meter.
#   metertest/make_metertest.sh BUILT_TASK_DIR OUT_DIR
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$1; out=$2
rm -rf "$out"; cp -a "$src" "$out"
mkdir -p "$out/environment/stubapi" "$out/environment/probe_answer"
cp -r "$out/solution/answer.py" "$out/solution/_oracle" "$out/environment/probe_answer/"
printf 'COPY probe_answer/ /probe_answer/\n' >> "$out/environment/Dockerfile"
cp "$here/stubapi/stub_api.py" "$out/environment/stubapi/stub_api.py"
printf 'FROM python:3.12-slim-trixie\nCOPY stub_api.py /stub_api.py\nCMD ["python", "/stub_api.py"]\n' > "$out/environment/stubapi/Dockerfile"
python3 - "$out" <<'PY'
import sys, re
from pathlib import Path
d = Path(sys.argv[1])
c = d / "environment/docker-compose.yaml"
s = c.read_text()
s = s.replace('      METER_BUDGET:', '      METER_UPSTREAM_ANTHROPIC: "http://stubapi:8000"\n      METER_BUDGET:')
s += """  stubapi:
    build:
      context: ./stubapi
    environment:
      STUB_OUTPUT_TOKENS: "200000"
    networks:
      - default
"""
c.write_text(s)
t = d / "task.toml"
t.write_text(t.read_text().replace('name = "benchcad-2.1-core/', 'name = "benchcad-2.1-core-metertest/'))
PY
echo "$out"
