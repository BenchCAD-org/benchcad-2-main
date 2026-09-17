# The container submitted code runs in.
#
# This image previously existed only as a local tag on one laptop
# (benchcad-sandbox:arm64) with no definition in the repo -- meaning the
# "reproducible container" did not exist. A fresh clone could not start the
# sandbox, and sandbox.py would fall through to checking CADENV_LOCAL and
# report an error that has nothing to do with a missing image.
#
# This file was MEASURED from that image, not written from memory:
#   /etc/os-release   Debian 13 (trixie)
#   PYTHON_VERSION    3.12.14
#   pip freeze        24 packages, all pinned (see requirements.txt)
#   dpkg -l           the libGL / libX11 set that vtk links against
#
# Build:     docker build -t benchcad-sandbox:arm64 sandbox/
# Override:  CADENV_DOCKER_IMAGE=<other tag>
FROM python:3.12-slim-trixie

# vtk / OCC offscreen rendering needs these. A missing one does not fail at
# import -- it throws on the first render, with an exception that does not
# name a library. Install the whole set rather than adding them on demand.
# Without libegl1 / libegl-mesa0 / libgles2 vtk has no offscreen context at
# all in a container (no X, EGL fails to load, no OSMesa) and a first
# Render() is a SIGSEGV (rc 139), measured 2026-09-17; with them it renders
# through mesa's llvmpipe (a T4 case's views and part sheets in 3.2 s).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglx-mesa0 libglvnd0 libgl1-mesa-dri \
        libegl1 libegl-mesa0 libgles2 libgbm1 \
        libx11-6 libx11-xcb1 libxext6 libxrender1 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# nlopt must be installed separately with --no-deps, or the build cannot
# resolve at all: nlopt 2.11.0 declares numpy>=2, while numpy is pinned to
# 1.26.4 (required by the cadquery-ocp pair). pip returns ResolutionImpossible.
#
# The live image has both coexisting -- `import nlopt` works under numpy 1.26.4
# -- which means that image was built incrementally and ITS OWN pip freeze
# output cannot be resolved. Only trying to rebuild it surfaces this.
#
# nlopt is used only by cadquery's Assembly.solve(). The scoring path never
# touches it (grep across envs/ and tools/verify_case.py: zero hits). It stays
# so that the API surface available to submitted code does not change.
RUN pip install --no-cache-dir --no-deps nlopt==2.11.0

# This layer was VERIFIED, not reasoned about. Two images with byte-identical
# `pip freeze` output disagreed: the same
#   cq.Workplane("XY").box(20,20,10).faces(">Z")
# passed on the live image and raised
#   AttributeError: TopoDS_Face has no attribute 'HashCode'
# on the rebuilt one -- because the live image carries a sitecustomize.py that
# pip freeze cannot see.
#
# cadquery 2.3 calls TopoDS_*.HashCode; cadquery-ocp 7.9 removed it. Without
# the shim EVERY .faces() / .edges() selector fails, which is most of CadQuery.
# The harness used to prepend this patch to each program it executed: that
# fixed our own runs and left the image broken for everyone else -- a plain
# `python -c` in this container could not use a selector at all. It belongs to
# the environment, not to one caller.
#
# "Identical package list" is not "identical behaviour". Only running geometry
# shows the difference.
COPY sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py

# Working directory for submitted code. The harness mounts the case inputs and
# tools.py here and sets PYTHONPATH=/work so `import tools` resolves.
WORKDIR /work
ENV PYTHONPATH=/work

# The sandbox has no network (the harness passes --network none). Nothing here
# may assume otherwise.
CMD ["python"]
