#!/bin/bash
# The reference answer (Harbor's oracle agent copies solution/ to /solution and runs this): answer.py and the
# reference files it reads, as harness/run.py's mock/oracle submits them.
set -euo pipefail
cp -r /solution/_oracle /work/_oracle
cp /solution/answer.py /work/answer.py
