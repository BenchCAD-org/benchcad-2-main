"""The verifier's own files, hashed before the answer runs and checked again before it is scored.

    python3 /tests/run/seal.py record SEAL_FILE
    python3 /tests/run/seal.py verify SEAL_FILE      (exit 1 and a line on stderr if anything changed)

The answer runs as another user and cannot write to /tests, but it could leave a link where the verifier
writes (its tools, its _run.py) and so have root write through it. The destinations are removed before every
copy; this is the check that nothing got through anyway.
"""
import hashlib
import json
import os
import stat
import sys

WATCH = ["/tests/test.sh", "/tests/pins.json", "/tests/run", "/tests/scorer/harness_scorer_child.py",
         "/tests/scorer/tools/core_score.py"]


def digest(path: str) -> dict:
    out = {}
    st = os.lstat(path)
    if stat.S_ISDIR(st.st_mode):
        for name in sorted(os.listdir(path)):
            out.update(digest(os.path.join(path, name)))
        return out
    if not stat.S_ISREG(st.st_mode):
        out[path] = f"not a regular file: {stat.S_IFMT(st.st_mode):o}"
        return out
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    out[path] = h.hexdigest()
    return out


def now() -> dict:
    out = {}
    for p in WATCH:
        try:
            out.update(digest(p))
        except OSError as e:
            out[p] = f"{type(e).__name__}"
    return out


mode, seal = sys.argv[1], sys.argv[2]
if mode == "record":
    fd = os.open(seal, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with open(fd, "w", closefd=True) as f:
        json.dump(now(), f)
    sys.exit(0)
was = json.load(open(seal))
is_ = now()
bad = sorted(k for k in set(was) | set(is_) if was.get(k) != is_.get(k))
for k in bad:
    sys.stderr.write(f"seal: {k} changed during the answer run\n")
sys.exit(1 if bad else 0)
