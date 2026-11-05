# The agent's image: the BenchCAD 2.1 sandbox (benchcad-2-main v2.1 sandbox/Dockerfile: Debian 13, Python 3.12,
# cadquery 2.3.0 / cadquery-ocp 7.9.3.0, the pinned set, the HashCode shim) plus what an installed agent needs to
# set itself up (curl, git, procps, tmux), and the case's staged inputs in /task and, as the working copy, /work.
# No reference geometry and no scorer: those are only in the verifier's image (tests/).
FROM python:3.12-slim-trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglx-mesa0 libglvnd0 libgl1-mesa-dri \
        libegl1 libegl-mesa0 libgles2 libgbm1 \
        libx11-6 libx11-xcb1 libxext6 libxrender1 libsm6 libice6 \
        ca-certificates curl git procps tmux \
    && rm -rf /var/lib/apt/lists/*
COPY sandbox/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt \
    && pip install --no-cache-dir --no-deps nlopt==2.11.0
COPY sandbox/sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py
COPY task/ /task/
COPY task/ /work/
RUN chmod -R a-w /task
WORKDIR /work
ENV PYTHONPATH=/work PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/.cache
