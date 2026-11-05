"""Empty /tmp, dotfiles and all, between the answer run and the scorer, and make the scorer's own private
directories. The answer runs as another user but shares /tmp and HOME with the root scorer, so anything it
left there -- a link where a library writes, a matplotlib or OCCT cache, a planted module -- is gone before
the scorer starts, and the scorer's HOME, TMPDIR, cache and config directories are made fresh here, root's
alone, outside /tmp.

    python3 /tests/run/wipe_tmp.py PRIVATE_DIR [DIR_TO_EMPTY]        (DIR_TO_EMPTY defaults to /tmp)
"""
import os
import shutil
import stat
import sys

private = sys.argv[1]
shared = sys.argv[2] if len(sys.argv) > 2 else "/tmp"

for name in sorted(os.listdir(shared)):
    p = os.path.join(shared, name)
    try:
        st = os.lstat(p)
        if stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode):
            shutil.rmtree(p, ignore_errors=True)
        else:
            os.unlink(p)
    except OSError as e:
        sys.stderr.write(f"wipe_tmp: {p}: {type(e).__name__}\n")

shutil.rmtree(private, ignore_errors=True)
for sub in ("home", "tmp", "mpl", "cache", "config"):
    os.makedirs(os.path.join(private, sub), mode=0o700)
os.chmod(private, 0o700)
