# syntax=docker/dockerfile:1
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

# Runnable ToOp image, intended for a laptop (Jupyter / interactive shell / VS Code dev container).
#
# This is NOT the image used by the root ./Dockerfile. That one is a bare dev container base: it
# layers uv + protoc onto python:3.13 and copies no project files, leaving dependency installation
# to .devcontainer/onCreateCommand.sh. This image installs the project instead.
#
# The `uv` release is pinned by digest, matching the style of the root Dockerfile.
FROM ghcr.io/astral-sh/uv:0.11.19@sha256:b46b03ddfcfbf8f547af7e9eaefdf8a39c8cebcba7c98858d3162bd28cf536f6 AS uv

# Python 3.11 because every pyproject.toml in this repo declares requires-python ">=3.11,<3.12".
# Deliberately tag-pinned rather than digest-pinned: a digest must be verified against the registry,
# and inventing one would be worse than none. To harden, run
#   docker buildx imagetools inspect python:3.11-slim-bookworm
# and append the reported @sha256:... here.
FROM python:3.11-slim-bookworm

COPY --from=uv /uv /usr/local/bin/uv

# git is required at build time: uv-dynamic-versioning derives each package version from git tags.
# ca-certificates is required for any HTTPS the packages perform (e.g. CGMES boundary downloads).
RUN apt-get -qq update \
 && apt-get -qq -y install --no-install-recommends git ca-certificates \
 && apt-get -qq -y autoremove \
 && apt-get autoclean \
 && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Extra conveniences for interactive/dev-container use. Off by default to keep the runtime slim.
# Note: protoc is intentionally NOT installed here. packages/interfaces_pkg/src/compile_proto.sh
# needs the exact protoc 33.1 pinned by the root ./Dockerfile, and Debian ships a different version.
# The generated *_pb2.py files are committed, so regeneration is a rare, deliberate task -- use the
# root Dockerfile for it.
ARG DEV_TOOLS=false
RUN if [ "$DEV_TOOLS" = "true" ]; then \
      apt-get -qq update \
   && apt-get -qq -y install --no-install-recommends vim htop zip unzip curl less procps \
   && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log; \
    fi

# The project environment lives OUTSIDE /app on purpose.
#
# /app is bind-mounted from the Windows host at runtime. Creating the venv at /app/.venv would put a
# Linux virtualenv on the Windows filesystem: it would collide with any natively-created .venv, and
# every import would drag thousands of small files across the WSL2 9p/virtiofs bridge. Relocating it
# keeps `uv sync` / `uv run` as the only interface while the environment sits on Linux-native
# storage. UV_PROJECT_ENVIRONMENT is uv's supported way to do this.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_PREFERENCE=system \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# The whole context is copied (including .git, ~9MB) so that uv-dynamic-versioning can resolve
# versions and so the six editable path-dependencies under packages/ resolve at build time. At
# runtime the host repo is bind-mounted over /app; because the editable install records
# /app/packages/*/src, source edits on the host take effect with no rebuild.
COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-groups

# Bytecode is compiled once, above, and ships in the image (and therefore in the toop-venv volume,
# which Docker seeds from it). Turning the flag off from here on keeps the entrypoint's re-sync from
# recompiling all ~38k files on every container start, which costs ~6s each time. Placed after the
# sync so it does not invalidate that layer.
ENV UV_COMPILE_BYTECODE=0

# Keep every byproduct of an in-container run off the bind mount.
#
# /app is the Windows checkout. Without this, the container writes __pycache__ next to the sources
# it imports, and those .pyc files record their origin as /app/packages/... Because the bind mount
# gives both sides byte-identical mtime and size, a later *native Windows* run happily reuses that
# container-built bytecode: pytest then reports test modules as `<- ..\..\app\packages\...`, and
# tracebacks point at paths that do not exist on the host. It works in reverse too -- Windows-built
# .pyc shadowing the container's. Confusing rather than fatal, but it makes a native-vs-container
# comparison untrustworthy, which is exactly when it matters.
#
# PYTHONPYCACHEPREFIX moves the whole cache tree to a container-local path (it mirrors the source
# layout underneath, so nothing is lost and imports stay fast). The pytest cache is redirected for
# the same reason: it lives at the rootdir, which is the bind mount.
#
# The host side needs the mirror image of this; see "Testing notes" in CLAUDE.md.
ENV PYTHONPYCACHEPREFIX=/tmp/pycache \
    PYTEST_ADDOPTS="-o cache_dir=/tmp/pytest_cache"

# Optional NVIDIA GPU support. Requires an NVIDIA GPU, a Windows driver with WSL2 CUDA support, and
# `docker compose --profile gpu`. Roughly doubles the image size, hence opt-in.
#
# Three things here are load-bearing and easy to get wrong:
#
#  1. `--python` is mandatory. Unlike `uv sync` / `uv run`, `uv pip` ignores UV_PROJECT_ENVIRONMENT;
#     it resolves VIRTUAL_ENV or ./.venv. Neither exists at build time (.venv/ is excluded by
#     Dockerfile.dockerignore), so without this the build fails outright rather than installing to
#     the wrong place.
#
#  2. The version is pinned to whatever uv.lock already resolved, read back from the environment
#     installed above rather than hardcoded here. `jax[cuda12]` unpinned resolves to the newest
#     release, which would silently violate the `jax (>=0.5.3,<0.6.0)` pin in
#     packages/dc_solver_pkg/pyproject.toml and desynchronise the image from the lock.
#
#  3. TOOP_CUDA is recorded in the image so entrypoint.sh knows to re-sync with --inexact. The CUDA
#     wheels are deliberately absent from uv.lock, so a plain `uv sync --frozen` treats them as
#     extraneous and uninstalls them on every container start -- silently downgrading the GPU
#     service back to CPU. See docker/README.md.
#
#  4. The two UV_ settings below are about getting ~2.8GB of wheels down a domestic link, not about
#     correctness. uv defaults to many concurrent downloads and a 30s read timeout; with cudnn at
#     762MB and cublas at 554MB, a dozen parallel streams each crawl, the timeout trips, and uv
#     restarts that wheel from zero -- which looks exactly like a hung build and can loop forever.
#     Fewer streams each get a usable share of the bandwidth, and the long timeout tolerates a stall.
#     Scoped to this RUN so they do not affect the runtime image or the entrypoint's sync.
ARG INSTALL_CUDA=false
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_CUDA" = "true" ]; then \
      JAX_VERSION="$("$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import jax; print(jax.__version__)')" \
   && echo "Installing jax[cuda12]==${JAX_VERSION} (pinned to the version resolved by uv.lock)" \
   && UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=2 \
      uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" "jax[cuda12]==${JAX_VERSION}"; \
    fi
ENV TOOP_CUDA=${INSTALL_CUDA}

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# Copied next to the entrypoint rather than left in the bind-mounted /app, so the container can still
# report its compute device when the repository is mounted over, absent, or mid-checkout.
COPY docker/device_banner.py /usr/local/bin/device_banner.py
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8888 6006

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uv", "run", "jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password=", \
     "--ServerApp.root_dir=/app"]
