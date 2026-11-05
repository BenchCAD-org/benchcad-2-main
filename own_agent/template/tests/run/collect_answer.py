"""Copy what the answer run left (submission/, final.step, pred_graph.json) from RUN_DIR to OUT_DIR as the answer
run's own unprivileged user, following symlinks and taking regular files only.

    python3 /tests/run/collect_answer.py RUN_DIR OUT_DIR

The scorer runs as root next to the reference (/tests/case), so it never reads the answer run's directory
itself: a link from there to the reference would be followed. This copy is made by the uid that ran the
answer, which cannot read /tests, so a link out of the run directory copies nothing."""
import os
import shutil
import stat
import sys
from pathlib import Path

UID = 61000
NAMES = ("submission", "final.step", "pred_graph.json")

run, out = Path(sys.argv[1]), Path(sys.argv[2])
os.setgroups([])
os.setgid(UID)
os.setuid(UID)


def copy(src: Path, dst: Path) -> None:
    try:
        st = os.stat(src)                    # follows a link, as the answer's own user
    except OSError as e:
        print(f"collect: skipped {src.relative_to(run)}: {type(e).__name__}", file=sys.stderr)
        return
    if stat.S_ISDIR(st.st_mode):
        dst.mkdir(exist_ok=True)
        try:
            entries = sorted(os.listdir(src))
        except OSError as e:
            print(f"collect: skipped {src.relative_to(run)}/: {type(e).__name__}", file=sys.stderr)
            return
        for name in entries:
            copy(src / name, dst / name)
    elif stat.S_ISREG(st.st_mode):
        try:
            with open(src, "rb") as f, open(dst, "wb") as g:
                shutil.copyfileobj(f, g)
        except OSError as e:
            dst.unlink(missing_ok=True)
            print(f"collect: skipped {src.relative_to(run)}: {type(e).__name__}", file=sys.stderr)


for name in NAMES:
    if os.path.lexists(run / name):
        copy(run / name, out / name)
