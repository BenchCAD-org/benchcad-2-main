#!/bin/bash
# Run /work/answer.py once in a copy of /work, as harness/run.py runs a submitted program, then score what it
# left with the 2.1 scorer (run/score.py). Writes /logs/verifier/reward.json and detail.json.
#
# The answer runs as uid 61000 and shares the container with the scorer, which runs as root beside the
# reference, so every step here assumes the answer is trying to be scored without solving the case:
#  - the verifier's log directory is root's alone while the answer runs, so no reward can be written for it;
#  - /work is copied in by file descriptor, regular files only, so no link of the answer's is ever followed;
#  - every file the verifier writes into the run directory is removed first, so root never writes through a
#    link the answer left; the verifier's own files are hashed before the run and checked again after it;
#  - a launch failure is reported on a file only root can write, not on stderr or an exit code;
#  - what the answer left is copied out by its own uid, which cannot read the reference;
#  - /tmp is emptied and the scorer gets its own HOME, TMPDIR and caches, outside /tmp.
V=/logs/verifier
chmod 0755 /logs 2>/dev/null
mkdir -p "$V"; chown 0:0 "$V"; chmod 0700 "$V"
rm -f "$V/reward.json" "$V/reward.txt" "$V/detail.json"
install -d -m 700 -o 0 -g 0 /run/state
: > /run/state/launch_failed
done_() { chmod 0755 "$V" 2>/dev/null; }
trap '[ -f "$V/reward.json" ] || /usr/local/bin/python3 /tests/run/fail.py "verifier stopped early"; done_' EXIT
FAIL() { /usr/local/bin/python3 /tests/run/fail.py "$1"; exit 0; }
SCORE() { /opt/scorer/bin/python -s /tests/run/score.py "$W" "$1" /run/private; exit 0; }

/usr/local/bin/python3 /tests/run/seal.py record /run/state/seal.json || FAIL "could not seal the verifier"

W=/run/answer; rm -rf "$W"
# a link is not an answer (it would be read here, as root), and neither is a file of unreasonable size
if [ ! -f /work/answer.py ] || [ -L /work/answer.py ] || [ "$(stat -c %s /work/answer.py)" -gt 16777216 ]; then
    mkdir -p "$W"; echo "no usable /work/answer.py"; SCORE -1
fi
/usr/local/bin/python3 /tests/run/copy_in.py /work "$W" 2>> "$V/answer.stderr" || FAIL "could not copy /work"
# the harness's own modules and the run script, each written where nothing of the answer's remains
rm -f "$W/tools.py" "$W/sitecustomize.py" "$W/_run.py"
cp /tests/run/tools.py /tests/run/sitecustomize.py "$W"/ || FAIL "could not install the harness tools"
cat /work/answer.py /tests/run/finalise.py > "$W/_run.py" || FAIL "could not write the answer run"
chown -R 61000:61000 "$W"

# 600 s, one CPU, no network (the verifier has none), uid 61000, an empty environment
if (cd "$W" && timeout -k 10 600 /usr/local/bin/python3 /tests/run/launch.py "$W" /run/state/launch_failed \
      > "$V/answer.stdout" 2> "$V/answer.stderr"); then rc=0; else rc=$?; fi
# the launcher's own failure (before the answer ran) is the verifier's: no score, not a zero
[ -s /run/state/launch_failed ] && rc=-97
echo "answer.py exit $rc"; tail -c 1500 "$V/answer.stderr"

# nothing it started outlives it; what it left is copied out by its own uid (links followed, regular files
# only), then made root's, and only that copy is scored
/usr/local/bin/python3 /tests/run/reap.py
W0=$W; W=/run/scored; rm -rf "$W"; mkdir -p "$W"; chown 61000:61000 "$W"
timeout -k 10 300 /usr/local/bin/python3 /tests/run/collect_answer.py "$W0" "$W" 2>> "$V/answer.stderr" \
  || FAIL "could not collect the answer's files"
/usr/local/bin/python3 /tests/run/reap.py
chown -R 0:0 "$W" && chmod -R go-w "$W"
/usr/local/bin/python3 /tests/run/wipe_tmp.py /run/private || FAIL "could not clear /tmp for the scorer"
/usr/local/bin/python3 /tests/run/seal.py verify /run/state/seal.json 2>> "$V/answer.stderr" \
  || FAIL "the verifier's own files changed while the answer ran"
SCORE "$rc"
