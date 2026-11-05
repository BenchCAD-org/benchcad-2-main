#!/bin/bash
# Run /work/answer.py once in a copy of /work, as harness/run.py runs a submitted program, then score what it
# left with the 2.1 scorer (run/score.py). Writes /logs/verifier/reward.json and detail.json.
V=/logs/verifier; mkdir -p "$V"; rm -f "$V/reward.json" "$V/reward.txt" "$V/detail.json"
trap '[ -f "$V/reward.json" ] || /usr/local/bin/python3 /tests/run/fail.py "verifier stopped early"' EXIT
SCORE() { /opt/scorer/bin/python /tests/run/score.py "$W" "$1"; exit 0; }

W=/run/answer; rm -rf "$W"
# a link is not an answer (it would be read here, as root)
if [ ! -f /work/answer.py ] || [ -L /work/answer.py ]; then mkdir -p "$W"; echo "no /work/answer.py"; SCORE -1; fi
cp -a /work "$W" || { /usr/local/bin/python3 /tests/run/fail.py "could not copy /work"; exit 0; }
# what answer.py must make itself is removed; the harness's tools are restored
rm -rf "$W/submission" "$W/final.step" "$W/pred_graph.json" "$W/_run.py" "$W/_render.py"
cp /tests/run/tools.py /tests/run/sitecustomize.py "$W"/
cat /work/answer.py /tests/run/finalise.py > "$W/_run.py"
chown -R 61000:61000 "$W"

# 600 s, one CPU, no network (the verifier has none), uid 61000, an empty environment
if (cd "$W" && timeout -k 10 600 /usr/local/bin/python3 /tests/run/launch.py "$W" \
      > "$V/answer.stdout" 2> "$V/answer.stderr"); then rc=0; else rc=$?; fi
# launch.py's own failure (before the answer ran) is the verifier's: no score, not a zero
[ $rc -eq 97 ] && grep -q '^BENCHCAD_LAUNCH_FAILED' "$V/answer.stderr" && rc=-97
echo "answer.py exit $rc"; tail -c 1500 "$V/answer.stderr"
# nothing it started outlives it; what it left is copied out by its own uid (links followed, regular files only),
# then made root's, and only that copy is scored
/usr/local/bin/python3 /tests/run/reap.py
W0=$W; W=/run/scored; rm -rf "$W"; mkdir -p "$W"; chown 61000:61000 "$W"
timeout -k 10 300 /usr/local/bin/python3 /tests/run/collect_answer.py "$W0" "$W" 2>> "$V/answer.stderr" \
  || { /usr/local/bin/python3 /tests/run/fail.py "could not collect the answer's files"; exit 0; }
/usr/local/bin/python3 /tests/run/reap.py
chown -R 0:0 "$W" && chmod -R go-w "$W"
SCORE "$rc"
