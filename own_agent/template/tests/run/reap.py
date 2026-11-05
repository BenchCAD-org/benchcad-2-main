"""Kill every process of the answer run's uid (61000), until none is left: nothing the answer started may
outlive its run and touch what is scored."""
import os
import signal
import time

UID = 61000
for _ in range(50):
    pids = []
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                if os.stat(f"/proc/{p}").st_uid == UID:
                    pids.append(int(p))
            except OSError:
                pass
    if not pids:
        break
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(0.1)
