# Where build.py runs: the 2.1 harness's own locked environment (this checkout's uv.lock, the same as
# the verifier's /opt/scorer), so the inputs are staged and the reference answers made by 2.1's own code.
FROM python:3.12-slim-trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglx-mesa0 libglvnd0 libgl1-mesa-dri \
        libegl1 libegl-mesa0 libgles2 libgbm1 \
        libx11-6 libx11-xcb1 libxext6 libxrender1 libsm6 libice6 git \
    && rm -rf /var/lib/apt/lists/* && git config --system --add safe.directory '*'
RUN pip install --no-cache-dir uv==0.11.16
COPY pyproject.toml uv.lock /opt/scorer-src/
RUN cd /opt/scorer-src && UV_PROJECT_ENVIRONMENT=/opt/scorer uv sync --frozen --no-dev --no-install-project \
        --python /usr/local/bin/python3.12
