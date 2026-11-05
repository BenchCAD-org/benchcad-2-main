"""Copy /work into the run directory, as root, taking regular files only.

    python3 /tests/run/copy_in.py SRC DST

Nothing in SRC is opened by a path the answer could have turned into a link: the walk descends by directory
file descriptors, every name is checked with lstat, links and device files are skipped, and the bytes copied
are capped. So neither a link out of /work (to the reference, to /dev/zero, to one of the verifier's own
files) nor a special file can reach the run directory or make the copy run away.
"""
import os
import stat
import sys

MAX_FILE = 4 << 30                       # one file
MAX_TOTAL = 8 << 30                      # the whole copy
MAX_DEPTH = 64
# answer.py must leave these itself, and the harness's own modules come from /tests, so they are not copied in
SKIP_AT_TOP = {"submission", "final.step", "pred_graph.json", "_run.py", "_render.py", "tools.py",
               "sitecustomize.py"}

total = 0
skipped: list[str] = []


def copy_dir(sfd: int, out: str, depth: int) -> None:
    """Copy the directory open at sfd into the (already made) directory `out`."""
    global total
    if depth > MAX_DEPTH:
        skipped.append(f"{out}: deeper than {MAX_DEPTH}")
        return
    try:
        names = sorted(os.listdir(sfd))
    except OSError as e:
        skipped.append(f"{out}: {type(e).__name__}")
        return
    for name in names:
        if depth == 0 and name in SKIP_AT_TOP:
            continue
        target = os.path.join(out, name)
        try:
            st = os.lstat(name, dir_fd=sfd)
        except OSError as e:
            skipped.append(f"{target}: {type(e).__name__}")
            continue
        if stat.S_ISLNK(st.st_mode):
            skipped.append(f"{target}: a link")
            continue
        if stat.S_ISDIR(st.st_mode):
            try:
                child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY, dir_fd=sfd)
            except OSError as e:
                skipped.append(f"{target}: {type(e).__name__}")
                continue
            try:
                os.mkdir(target, 0o755)
            except FileExistsError:
                pass
            try:
                copy_dir(child, target, depth + 1)
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(st.st_mode):
            skipped.append(f"{target}: not a regular file")
            continue
        if st.st_size > MAX_FILE or total + st.st_size > MAX_TOTAL:
            skipped.append(f"{target}: over the size cap")
            continue
        try:
            rfd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=sfd)
        except OSError as e:
            skipped.append(f"{target}: {type(e).__name__}")
            continue
        try:
            wfd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        except OSError as e:
            os.close(rfd)
            skipped.append(f"{target}: {type(e).__name__}")
            continue
        try:
            with open(rfd, "rb", closefd=True) as f, open(wfd, "wb", closefd=True) as g:
                while chunk := f.read(1 << 20):
                    total += len(chunk)
                    if total > MAX_TOTAL:
                        raise OSError("over the total size cap")
                    g.write(chunk)
        except OSError as e:
            skipped.append(f"{target}: {type(e).__name__}: {e}")


src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
top = os.open(src, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.chdir(dst)
    copy_dir(top, ".", 0)
finally:
    os.close(top)
for line in skipped:
    sys.stderr.write(f"copy_in: skipped {line}\n")
