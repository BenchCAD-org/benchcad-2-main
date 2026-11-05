"""Start the answer run as harness/run.py's sandbox does, as far as a process inside a container can: one CPU,
not root, an empty environment, PYTHONPATH the run directory, a cap on processes, and an address-space cap
(2.1 caps the run at 2 GB of memory with docker --memory; a cgroup is not available in here, and address space
is not memory -- OCCT and numpy reserve far more than they touch -- so the cap is 6 GiB of address space).

The run's uid is 61000, not 1000: RLIMIT_NPROC counts every process of a uid on the host kernel, and uid 1000
is the host user on many machines (on a shared host whose user ran many processes, the answer could not
even start). Anything that fails here, before the answer runs, is written to the marker file given as the
second argument, and the verifier reports no score instead of a zero. The marker is opened here, as root, in
a directory the answer cannot write, and is not inherited across the exec, so only this code can write it:
the answer cannot buy itself a "no score" by printing a marker or exiting 97."""
import os
import resource
import sys

UID = 61000
LAUNCH_FAILED = 97

work, marker = sys.argv[1], sys.argv[2]
fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
os.set_inheritable(fd, False)
try:
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpus[0]})
    as_bytes = int(os.environ.get("SUB_AS_BYTES", str(6 * 1024 ** 3)))
    resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
    resource.setrlimit(resource.RLIMIT_NPROC, (1024, 1024))
    os.setgroups([])
    os.setgid(UID)
    os.setuid(UID)
    os.chdir(work)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "PYTHONPATH": work,
           "PYTHONDONTWRITEBYTECODE": "1", "XDG_CACHE_HOME": "/tmp/.cache-sandbox"}
    os.execve("/usr/local/bin/python3", ["python", "_run.py"], env)
except OSError as e:
    os.write(fd, f"BENCHCAD_LAUNCH_FAILED {type(e).__name__}: {e}\n".encode())
    sys.stderr.write(f"BENCHCAD_LAUNCH_FAILED {type(e).__name__}: {e}\n")
    sys.exit(LAUNCH_FAILED)
