# The verifier's image (environment_mode = separate), built from tests/ and never seen by the agent.
#   /usr/local/bin/python3   the 2.1 sandbox's Python (cadquery 2.3.0 ...): runs the agent's answer.py
#   /opt/scorer              the 2.1 harness's own locked environment (benchcad-2-main v2.1 uv.lock): scores it
#   /tests/scorer/envs       the 2.1 scorer, byte for byte (tests/pins.json: its tree digest and the T6 digest)
#   /tests/case              the case: case.json, input/, gt/
# Layers before the case are the same for every task, so a host builds them once.
FROM python:3.12-slim-trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglx-mesa0 libglvnd0 libgl1-mesa-dri \
        libegl1 libegl-mesa0 libgles2 libgbm1 \
        libx11-6 libx11-xcb1 libxext6 libxrender1 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*
COPY sandbox/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt \
    && pip install --no-cache-dir --no-deps nlopt==2.11.0
COPY sandbox/sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py
RUN pip install --no-cache-dir uv==0.11.16
COPY scorer/pyproject.toml scorer/uv.lock /opt/scorer-src/
RUN cd /opt/scorer-src && UV_PROJECT_ENVIRONMENT=/opt/scorer uv sync --frozen --no-dev --no-install-project \
        --python /usr/local/bin/python3.12
COPY scorer/ /tests/scorer/
COPY run/ /tests/run/
COPY case/ /tests/case/
COPY test.sh pins.json /tests/
RUN chmod -R go-rwx /tests && chmod +x /tests/test.sh
