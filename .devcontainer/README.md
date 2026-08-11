# Dev container notes

Background for three things that look odd in `Dockerfile` and `devcontainer.json` and are
deliberate. They live here rather than as inline comments because `devcontainer.json` is checked by
the `check-json` pre-commit hook, which parses strict JSON and rejects comments.

## `UV_PROJECT_ENVIRONMENT=/opt/venv`

The project environment is deliberately **not** in the workspace folder.

The workspace is a bind mount from the host, and on Windows that means it sits on NTFS across the
WSL2 filesystem bridge. A Linux virtualenv there would collide with any `.venv` created natively on
the host — `uv sync` in the container would write a Linux environment straight over the developer's
Windows one — and every import would drag thousands of small files across the bridge.

Relocating it keeps `uv sync` and `uv run` as the only interface, exactly as documented; only the
location moves. The environment stays on container-local storage, and the interpreter path is the
same regardless of host OS.

## `python:3.11-slim-bookworm` and `UV_PYTHON_PREFERENCE=system`

The base image's Python version is pinned to match `requires-python = ">=3.11,<3.12"`, and uv is told
to prefer it over downloading a managed interpreter. Both halves are load-bearing.

The image previously used `python:3.13`. Since no package in this repo accepts 3.13, `uv sync`
silently downloaded its own CPython 3.11 into `/root/.local/share/uv/python/` and pointed the
virtualenv at it:

```
home = /root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin
```

That path lives in the container's writable layer, not in any volume. So a `/opt/venv` kept on a
named volume — which is what makes container restarts fast — referenced an interpreter that ceased
to exist the moment the container was replaced. The next container found a dangling symlink and
started over:

```
warning: Ignoring existing virtual environment linked to non-existent Python interpreter:
         /opt/venv/bin/python3 -> python
Removed virtual environment at: /opt/venv
```

Every fresh container therefore re-downloaded 29.4 MB of CPython and reinstalled all 315 packages,
which defeats the point of persisting the environment at all.

With the interpreter supplied by the base image, `pyvenv.cfg` reads `home = /usr/local/bin`, which is
present in every container built from that image. A second container against the same volume now
reports `Checked 315 packages in 347ms` and starts immediately. Matching the base image to
`requires-python` also removes a silent version mismatch and cuts the image roughly in half.

Note that `python:3.11-slim-bookworm` ships neither `git` nor `ca-certificates`, and the `Dockerfile`
installs both: `uv-dynamic-versioning` derives every package version from git tags, so `uv sync`
fails outright without git.

## `"python.defaultInterpreterPath": "/opt/venv/bin/python"`

Follows from the above: with the environment outside the workspace, the Python extension cannot find
an interpreter by scanning the workspace folder, so the path is given explicitly.

The tracked `.vscode/settings.json` sets the same key to `${workspaceFolder}/.venv/bin/python`,
which is correct on the host but does not exist in the container. If the extension ever picks up the
workspace value inside the container, select the `/opt/venv` interpreter manually once and VS Code
will remember it.

## `${localEnv:HOME}${localEnv:USERPROFILE}/.ssh`

The two variables are concatenated on purpose. Windows sets `USERPROFILE` and leaves `HOME` unset;
Linux and macOS do the reverse. Exactly one of the pair expands, so the result is the correct home
directory on any host.

Previously the mount read `${localEnv:HOME}/.ssh`, which resolved to `/.ssh` on Windows and failed
the container start outright.
