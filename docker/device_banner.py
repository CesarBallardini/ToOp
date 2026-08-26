# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Report whether JAX is actually running on the GPU or has fallen back to the CPU.

Three of the four ways the container GPU setup can break are silent: the container starts, nothing
errors, and every computation quietly runs on the CPU. The only reliable signal is which device JAX
hands back, so this module asks it directly and says so in plain language.

It is used in two places:

- ``docker/entrypoint.sh`` runs it as a script at container start.
- Analysis scripts import :func:`print_device_banner` and call it before doing any work.

Deliberately importing nothing from ``toop_engine_*``: it must stay usable even when the project
environment is half-installed, which is exactly when a device problem is most likely.
"""

# This file is a console diagnostic -- printing is the entire point, so T201 does not apply.
# ruff: noqa: T201

import contextlib
import importlib.util
import os
import sys
from typing import Optional

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

BANNER_WIDTH = 72
CUDA_PLUGIN = "jax_cuda12_plugin"
SANITY_SIZE = 1024


def _probe_gpu() -> tuple[list["jax.Device"], str]:
    """Try to obtain GPU devices from JAX, and explain it if that is not possible.

    Returns
    -------
    tuple[list["jax.Device"], str]
        The GPU devices JAX reports (empty if none are usable), and a human-readable reason for the
        absence. The reason is an empty string when GPUs were found.
    """
    pinned = os.environ.get("JAX_PLATFORMS", "")
    if pinned and "cuda" not in pinned and "gpu" not in pinned:
        return [], f"JAX_PLATFORMS={pinned!r} pins JAX to the CPU. Unset it to attempt the GPU."

    try:
        devices = jax.devices("gpu")
    except RuntimeError as exc:
        detail = str(exc).strip()
        summary = detail.splitlines()[0] if detail else "no GPU backend"
        if importlib.util.find_spec(CUDA_PLUGIN) is None:
            return [], (
                f"the CUDA build of JAX is not installed ({CUDA_PLUGIN} missing). Rebuild the image with "
                "INSTALL_CUDA=true, i.e. `docker compose --profile gpu build toop-gpu`."
            )
        return [], f"the CUDA plugin is installed but no device is visible -- {summary}. Check `nvidia-smi` first."

    if not devices:
        return [], "JAX reports zero GPU devices."
    return devices, ""


def _verify_gpu_executes(device: "jax.Device") -> str:
    """Run a trivial computation on ``device`` to prove it works rather than merely being listed.

    Parameters
    ----------
    device : jax.Device
        The JAX device to place the computation on.

    Returns
    -------
    str
        An empty string on success, or the reason the computation failed.
    """
    try:
        with jax.default_device(device):
            total = float(jnp.ones(SANITY_SIZE, dtype=jnp.float32).sum())
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if total != float(SANITY_SIZE):
        return f"sanity check returned {total}, expected {float(SANITY_SIZE)}"
    return ""


def _vram_budget_gib(device: "jax.Device") -> Optional[float]:
    """Return the VRAM budget JAX has reserved on ``device``, in GiB, or None if unavailable.

    Parameters
    ----------
    device : jax.Device
        The JAX device to query.

    Returns
    -------
    Optional[float]
        The budget in GiB, or None when the device does not report memory statistics.
    """
    with contextlib.suppress(Exception):
        limit = (device.memory_stats() or {}).get("bytes_limit")
        if limit:
            return float(limit) / 2**30
    return None


def print_device_banner() -> bool:
    """Print which device JAX will use, attempting the GPU first and falling back to the CPU.

    Returns
    -------
    bool
        True when the GPU was found *and* proved able to execute, False when running on CPU.
    """
    rule = "=" * BANNER_WIDTH
    print(rule)

    if jax is None:
        print("  COMPUTE DEVICE: unknown -- JAX is not importable in this environment")
        print(rule)
        return False

    gpus, reason = _probe_gpu()
    failure = _verify_gpu_executes(gpus[0]) if gpus else ""

    if gpus and not failure:
        device = gpus[0]
        print(f"  COMPUTE DEVICE: GPU  <<< {getattr(device, 'device_kind', 'unknown GPU')}")
        print(f"  devices       : {gpus}")
        budget = _vram_budget_gib(device)
        if budget is not None:
            print(f"  VRAM budget   : {budget:.2f} GiB (JAX pre-allocates ~75% of the card by default)")
        print("  note          : ToOp forces float64; consumer GeForce cards run it at ~1/32 of float32 speed,")
        print("                  so the GPU only pays off on grids large enough to fill it.")
        print(rule)
        return True

    if gpus and failure:
        reason = f"a GPU is listed but could not execute -- {failure}"

    print("  COMPUTE DEVICE: CPU ONLY  <<< the GPU is NOT being used")
    print(f"  devices       : {jax.devices()}")
    print(f"  reason        : {reason or 'no GPU backend available'}")
    print(rule)
    return False


if __name__ == "__main__":
    print_device_banner()
    # Always succeed: this is a diagnostic, and must never stop a container from starting.
    sys.exit(0)
