<!--
Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file,
you can obtain one at https://mozilla.org/MPL/2.0/.
Mozilla Public License, version 2.0
-->

# Run ToOp on Windows: a container path, and the five bugs behind the native one

Pull request from `feat/add-docker-compose` into `main`. This file is the PR description.

ToOp had no supported path on Windows. This branch adds one -- a Docker Compose setup that is the
recommended route -- and, separately, fixes the five defects that stopped the DC optimizer from
running *natively* on a Windows host. All five are **invisible on Linux**, which is why CI never
caught any of them, and they surfaced in sequence: each was hidden behind the one before it.

Four of the five are the same underlying mistake -- **`dtype=int` in numpy is the platform's C
`long`, so it is int32 on Windows and int64 everywhere else.** It appears as an explicit `dtype=`
argument (section 2), as a value split across lines that a line-based sweep misses (also section 2),
as an implicit width that must match a *neighbouring* array (section 4), and as an API *default*
with no `dtype=` in sight at all (section 5). The last is the one worth remembering: it means the
grep that catches the others is not sufficient.

Nothing here changes behaviour on Linux. That claim is argued from the code in
[Why this is a no-op on Linux](#why-this-is-a-no-op-on-linux) and tested in
[Verification](#verification).

## Scope

`git diff origin/main...HEAD` -> **51 files changed, 5522 insertions(+), 4231 deletions(-)**.

Diff against `origin/main`, not a local `main`: this branch was cut from a `main` that is now 15
commits behind, so `git diff main...HEAD` reports 315 files and 43 999 insertions, almost all of it
other people's merged work.

| Commit | Files | Subject |
|---|---|---|
| `d70e1a9` | 15 | `feat:` container setup for running ToOp on Windows |
| `31cceec` | 17 | `style:` normalize the tree to LF line endings -- **see the review note below** |
| `1b764b5` | 4 | `fix(docker):` make the container GPU path work, and report the device in use |
| `7773fc7` | 18 | `fix:` make the DC optimizer run natively on Windows |
| `cd1457d` | 1 | `build(docker):` survive slow CUDA wheel downloads, isolate the bytecode cache |
| `320e25f` | 2 | `docs:` record the Windows fixes and the container test workflow |

Plus, still uncommitted at the time of writing: the `_ConnectionBase` fix (fix 3 below, 2 files), one
leftover `dtype=int` site in `topology_computations.py`, the `uv.lock` repair, and this file.

### How to review this efficiently

**`31cceec` is 4054 insertions / 4038 deletions of pure line-ending change** -- 95 % of the deletions
in the whole PR. It carries no semantic content. Review it with whitespace suppressed, or skip it:

```bash
git show -w 31cceec | head -40          # near-empty, as it should be
git diff -w origin/main...HEAD          # the whole PR, minus the noise
```

It exists because the repo had mixed CRLF/LF and no `.gitattributes`; `d70e1a9` adds one, and this
commit applies it in a single pass so the normalization never lands mixed into a behavioural diff.

The rest splits cleanly in two, and the halves are independent:

- **Container path** -- `d70e1a9`, `1b764b5`, `cd1457d`. Almost entirely new files, plus a
  `.devcontainer` and `pyproject.toml` touch-up. Rationale in [`docker/README.md`](docker/README.md).
- **Native Windows fixes** -- `7773fc7` and the uncommitted remainder. This is the only part that
  touches `packages/`, and it is what most of this document is about.

---

# Part 1 -- the container path

`docker-compose.yaml` at the repo root is the entry point; `docker/Dockerfile` is the image that
actually runs the project. The pre-existing root `./Dockerfile` is an unrelated dev-container *base*
used to regenerate protobuf gencode -- the two are not interchangeable, and `docker/README.md` says
so explicitly, because confusing them is the obvious first mistake.

```bash
docker compose up -d                          # Jupyter Lab on :8888
docker compose exec toop bash                 # interactive; use `uv run ...`
docker compose --profile gpu up -d toop-gpu   # NVIDIA variant, Jupyter on :8889
```

Design points that are load-bearing and non-obvious. Each caused a silent failure before it was
pinned down, and all are documented at length in [`docker/README.md`](docker/README.md):

- The uv environment lives at **`/opt/venv`**, not `.venv`, because the workspace is a bind mount and
  a host `.venv` seen through it would otherwise shadow the container's. `uv pip` ignores
  `UV_PROJECT_ENVIRONMENT` and needs an explicit `--python`; `uv sync` and `uv run` do not.
- The base is `python:3.11-slim-bookworm` with `UV_PYTHON_PREFERENCE=system`, matching
  `requires-python`. Raising it makes uv download its own CPython into the layer, which invalidates
  any venv kept on a volume whenever the container is replaced.
- `git` and `ca-certificates` are installed explicitly -- `uv-dynamic-versioning` needs git to
  resolve package versions, so `uv sync` fails outright on the bare slim base.
- **GPU (`1b764b5`)**: four things must all hold, and three of them fail *silently* -- the container
  starts, nothing errors, and everything runs on CPU. `jax[cuda12]` pinned to the version `uv.lock`
  resolved (unpinned it violates `jax (>=0.5.3,<0.6.0)`); `TOOP_CUDA=true` so `entrypoint.sh` syncs
  with `--inexact`, since the CUDA wheels are deliberately absent from the lock and a plain
  `--frozen` sync treats them as extraneous and uninstalls them on *every* start; a **separate**
  `toop-venv-gpu` volume, because Docker seeds a named volume from the image only while that volume
  is still empty, so a shared one shadows the baked-in wheels; and the
  `deploy.resources.reservations.devices` block, which the compose profile alone does not imply.
- Because three of those fail silently, `docker/device_banner.py` **proves** the device by running a
  computation rather than trusting `jax.devices()`. It prints at container start, falls back to CPU,
  and names the cause: wheels missing, `JAX_PLATFORMS` pinned, or the card not passed through.
- **`cd1457d`** scopes `UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=2` to the CUDA `RUN` layer -- 3.5 GB
  of wheels otherwise times out in a way that looks like a hang -- and sets
  `PYTHONPYCACHEPREFIX=/tmp/pycache` plus `PYTEST_ADDOPTS="-o cache_dir=/tmp/pytest_cache"` so the
  container and the Windows host stop trading `__pycache__` through the bind mount. That
  cross-contamination produced failures which moved depending on which platform had run last.

---

# Part 2 -- the native Windows fixes

## 1. TensorBoard directory name contained colons

`dc/main.py` built the log directory from `str(datetime.datetime.now())`, e.g.
`'2026-08-26 11:10:00.685782'`. Colons are illegal in Windows paths, so `SummaryWriter` raised
before the first epoch ran:

```
OSError: [WinError 123] ... 'C:\...\results/test/2026-08-26 11:10:00.685782'
  File ".../topology_optimizer/dc/main.py", line 274, in main
    writer = SummaryWriter(f"{args.tensorboard_dir}/{datetime.datetime.now()}")
```

**Fix.** Use the filesystem-safe format the AC side had already adopted at
`ac/optimizer.py:298` — one convention, not two:

```python
run_timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S_%f")
writer = SummaryWriter(f"{args.tensorboard_dir}/{run_timestamp}")
```

Microsecond precision keeps two runs started in the same second from colliding.

---

## 2. `dtype=int` is the platform's C `long`

`numpy` resolves `dtype=int` to the C `long`: 64-bit on Linux/macOS, **32-bit on Windows**. The
codebase used it throughout for index and sentinel arrays, so Windows built a *different grid
representation* from every other platform.

`int_max()` — which correctly keys off jax's `jax_enable_x64` flag and returns 2⁶³−1 everywhere —
was then combined with `dtype=int` in the same expression. The two conventions agree only on Linux.

### Three distinct failure modes

**a. Hard error.** `jnp.array(...)` refuses the down-cast:

```python
>>> jnp.array(int_max(), dtype=int)          # on Windows
OverflowError: Python int too large to convert to C long
```

This took out 10 tests in `tests/dc/genetic_functions`, and `test_mutate_disconnections.py` failed
at *collection* — a `@parametrize` decorator called it at module scope, so all 32 tests in the file
never ran.

**b. Silent corruption.** `jnp.full(...)` does **not** raise; it wraps:

| | sentinel stored in `action_index` | round-trips |
|---|---|---|
| Windows, before | `-1` | ❌ |
| Windows, after | `9223372036854775807` | ✅ |
| Linux | `9223372036854775807` | ✅ |

`-1` is a *valid* negative index that wraps to the last element, rather than an out-of-bounds
sentinel that gets dropped — so padding corrupted real data instead of being ignored.

**c. Scan-carry mismatch.** An int32 array meeting an int64 one inside `jax.lax.scan`:

```
TypeError: scan body function carry input and carry output must have equal types
  loop_carry[1].repertoire.genotypes.nodal_injections_optimized.pst_tap_idx
  has type int32[800,1,2] but the corresponding output carry component has type int64[800,1,2]
```

### Fix

A single canonical helper in `dc_solver/jax/types.py`, beside the existing `int_max()`:

```python
def int_dtype() -> jnp.dtype:
    """Canonical integer dtype for the current jax precision mode."""
    return jnp.dtype(jnp.int_)
```

`jnp.int_` keys off jax's own x64 flag rather than the C `long`, so it is int64 on every platform.
Applied across 18 files: **74 lines are exactly `dtype=int` → `dtype=int_dtype()`** (or
`.astype(int)` → `.astype(int_dtype())`), plus imports.

**On Linux `int_dtype() == np.dtype(int)`, so every one of those 74 substitutions is textually
identical there — the change is a provable no-op on Linux** and converges Windows onto Linux's
existing behaviour.

Two changes go beyond the mechanical substitution:

- **`fix_dtypes()`** passed `nodal_injections_optimized` through untouched, which is exactly how
  `pst_tap_idx` escaped normalization and ended up a different width from the rest of the carry. It
  now normalizes the PST taps too.
- **`empty_repertoire()`**'s `jnp.tile` inherited whatever width the static information carried; it
  now normalizes instead of inheriting.

### Auditing this change: line-based greps miss it

The substitution was applied mechanically, and one site escaped -- because the call spans lines and
`int_max()` and `dtype=int` are not on the same one:

```python
    return ActionIndexComputations(
        action=jnp.full(
            (
                batch_size,
                1,
            ),
            int_max(),
            dtype=int,        # <- topology_computations.py, default_topology()
        ),
```

A `sed`/`grep` keyed on lines containing `int_max` sees nothing here, and so did the verification
grep written the same way -- the audit shared the bug with the fix. It surfaced as
`test_default_topology` failing with `Array([[-1], ...], dtype=int32)` against the expected
`int_max()`, i.e. exactly failure mode (b) above, in the one place the sweep had not reached.

Audit with a matcher that spans lines. This walks every `jnp.*`/`np.*` call and flags those combining
`int_max` with a bare `dtype=int`:

```python
for m in re.finditer(r"(?:jnp|np)\.\w+\((?:[^()]|\([^()]*\))*?\)", src, re.S):
    if "int_max" in m.group(0) and re.search(r"\bdtype\s*=\s*int\s*[,)]", m.group(0)):
        ...
```

Across `packages/*/src` and `packages/*/tests` that now reports exactly **one** remaining site,
`jax/inputs.py:66` -- the `tot_stat` case under "Still open", which is left alone deliberately.

The ~280 other bare `dtype=int` occurrences in the tree are *not* flagged and were *not* touched:
they build small index arrays that never carry a 2^63-1 sentinel, so on Windows they are int32 but
correct. Converting them wholesale would be a much larger diff with no defect behind it. The rule
this PR establishes is narrower and checkable: **`int_max()` and `dtype=int` must never appear in the
same call.**

The rule is recorded under "JAX conventions" in `CLAUDE.md`. It applies to `tests/` as well — the
shared `conftest.py` fixture was building int32 `ActionSet`s (`substation_correspondence=i32[2000]`),
which produced the last three failures.

---

## 3. `multiprocessing.Pipe()` returns a different type on Windows

`mp.Pipe()` yields a `Connection` on POSIX but a **`PipeConnection`** on Windows, and the two are
*siblings* rather than parent and child:

```python
>>> a, b = mp.Pipe()                      # on Windows
>>> type(a).__mro__
(PipeConnection, _ConnectionBase, object)
>>> isinstance(a, Connection)
False
```

`runner.py:70` annotated the parameter as the narrow `Connection`. Under `ENABLE_BEARTYPE=true` —
which `[tool.pytest_env]` sets for every test run — beartype rejected the argument, the child process
died on entry, and the parent blocked forever in `parent_conn.recv()` until the 300 s timeout. A
hang, not an error, which is why it was easy to mistake for a slow test.

**Fix.** Annotate the shared base, which is what the parameter actually means:

```python
from multiprocessing.connection import _ConnectionBase as Connection
```

`_ConnectionBase` is private but stable across CPython 3.x, and this is strictly a *widening* —
`Connection` is itself a `_ConnectionBase`, so everything previously accepted still is. Applied in
both places carrying the annotation: `dc_solver/jax/benchmarks/runner.py` and
`topology_optimizer/benchmark/benchmark_utils.py` (`run_task_process`), the latter being the
multiprocessing path of the pipeline itself.

Result on Windows: `test_runner.py` went from **hanging for 300 s** to **2 passed in 47 s**.

---

## 4. The same C `long`, in `find_bridges.py`

`find_bridges()` compares the bridge pairs networkx returns against the grid's own `(from, to)`
pairs, using the structured-view trick the code cites in place (`stackoverflow.com/a/8317403`, kept
as text rather than a link -- Stack Overflow answers 403 to `markdown-link-check`). The view's field
formats are derived from `from_to_node.dtype`, but were applied to *both* arrays -- while `bridges`
was built with the C `long`:

```python
bridges = np.array(bridges, dtype=int)               # int32 on Windows: 2 cols = 8 bytes
from_to_node = np.c_[from_node, to_node]             # int64 from the caller: 2 cols = 16 bytes
dtype = {"names": [...], "formats": ncols * [from_to_node.dtype]}     # 2 x int64 = 16 bytes
np.intersect1d(from_to_node.view(dtype), bridges.view(dtype), ...)    # <- raises on `bridges`
```

```
ValueError: When changing to a larger dtype, its size must be a divisor of the total size in bytes
of the last axis of the array.
```

**Fix.** Derive the width from the array it must match, rather than naming one:

```python
from_to_node = np.c_[from_node, to_node]
bridges = np.array(bridges, dtype=from_to_node.dtype)
```

`int_dtype()` would also have worked today, but it would re-break the moment a caller passes
something narrower -- the invariant is *"these two agree"*, not *"both are int64"*, so the code now
says that. `from_to_node` is simply built a few lines earlier; nothing else moves.

Accounted for **12 failures** across `test_postprocess_powsybl.py` and
`test_validate_loadflow_results.py`.

## 5. `np.random.randint`'s default is also the C `long`

`test_case9241_pp` failed before reaching any grid logic:

```python
seed=np.random.randint(2**32)     # example_grids.py
```

```
ValueError: high is out of bounds for int32
```

This one matters out of proportion to its size, because **the audit rule from section 2 does not
catch it**. There is no `dtype=` and no `.astype()` -- the C `long` arrives as an API *default*. Any
numpy entry point that mints integers has the same exposure.

**Fix**, and both halves are load-bearing:

```python
def _bisection_seed() -> int:
    return int(np.random.randint(2**32, dtype=np.int64))
```

- `dtype=np.int64` makes `high=2**32` representable. int64 is what the default already resolved to
  on Linux, so the sampled range is unchanged there.
- `int(...)` is not cosmetic. Passing an explicit `dtype` switches the return from a Python `int` to
  a numpy scalar, and networkx's `py_random_state` accepts only the former:

  ```
  >>> type(np.random.randint(100))                      # no dtype
  <class 'int'>
  >>> type(np.random.randint(2**32, dtype=np.int64))    # with dtype
  <class 'numpy.int64'>
  ValueError: 3125985628 cannot be used to generate a random.Random instance
  ```

  Fixing only the width swaps one Windows failure for a different one. The first attempt here did
  exactly that.

---

---

## Why this is a no-op on Linux

Two independent arguments, and they agree.

**Textual.** On Linux and macOS `int_dtype()` and `np.dtype(int)` are the same dtype, so all 74
substituted lines mean exactly what they meant before:

```python
>>> int_dtype() == np.dtype(int)     # Linux, macOS
True
>>> int_dtype() == np.dtype(int)     # Windows
False        # int64 vs int32 -- this is the entire bug
```

The `_ConnectionBase` change is a strict *widening*: `Connection` is itself a `_ConnectionBase`, so
everything the old annotation accepted the new one still accepts. The `strftime` change alters only a
directory *name*.

**Empirical.** The full `dc_solver` suite in the Linux container, with the branch applied:
**531 passed, 8 skipped, 7 xfailed, 0 failed** (39:57, serial). That run also confirmed the cache
isolation from `cd1457d` was in effect -- pytest reported `cachedir: /tmp/pytest_cache`, and no test
module was collected through a `C:\...` path.

## Verification

| Suite | Windows, at `origin/main` | Windows, with this PR | Linux container |
|---|---|---|---|
| `tests/dc/test_main.py` | all fail (`WinError 123`) | **8 passed** | **8 passed** |
| `tests/dc/genetic_functions` | 74 collected -- 63 passed / 11 failed, +1 file uncollectable | **106 passed** | **106 passed** |
| `test_mutate_disconnections.py` | 0 run (collection error) | **32 passed** | **32 passed** |
| `jax/test_topology_computations.py` | 6 passed / 3 failed | **9 passed** | **9 passed** |
| `jax/test_runner.py` | hangs, aborts the session at its 300 s timeout | **2 passed** (47 s) | **2 passed** |
| `packages/dc_solver_pkg/tests` (full, serial) | 426 passed / **103 failed** | **468 passed / 63 failed** | **531 passed, 0 failed** |

`ruff check` and `ruff format --check` are clean on every file this PR touches.

### The full-suite comparison, done properly

Windows still fails a substantial number of `dc_solver` tests **with or without this PR** -- see
[The 63 that still fail on Windows](#the-63-that-still-fail-on-windows). Closing those is not in
scope. What matters for review is the *delta*, measured by extracting `origin/main` read-only and
overriding the editable installs with `PYTHONPATH`, so both runs see identical fixtures:

```bash
git archive 1b764b5 | tar -x -C /tmp/toop_prefix     # read-only; no worktree, no stash
```

Both runs, full `packages/dc_solver_pkg/tests`, serial, same machine:

| | `origin/main` (`1b764b5`) | with this PR |
|---|---|---|
| failed | **103** | **63** |
| passed | 426 | **468** |
| skipped / xfailed | 8 / 7 | 8 / 7 |

Comparing the two failure sets as node ids:

| | count |
|---|---|
| **fixed** by this PR (failed before, pass now) | **40** |
| **introduced** by this PR (passed before, fail now) | **0** |
| unchanged -- fail both before and after | 63 |

Measured in two stages, because sections 4 and 5 were found by characterising what remained after
sections 1-3: 103 -> 76 failures from sections 1-3, then 76 -> 63 from sections 4 and 5. The
introduced set was empty at both stages.

The 40 fixed span ten files, and include the whole `test_loadflows_match` family:

```
postprocessing/test_postprocess_powsybl.py       16
postprocessing/test_validate_loadflow_results.py  5
jax/test_busbar_outage.py                         5
preprocessing/test_powsybl_backend.py             3   <- see the correction below
preprocessing/test_loadflows_match.py             3
jax/test_topology_computations.py                 3
jax/test_compute_batch.py                         2
test_example_grids.py                             1
jax/test_cross_coupler_flow.py                    1
test_fixtures.py                                  1
                                                 --
                                                 40
```

The introduced set is empty. An earlier draft of this document reported one regression
(`test_default_topology`); that was the multi-line `dtype=int` site described above, and it is fixed.

Sections 4 and 5 are confirmed a no-op on Linux by their own container run -- the three files they
touch, `test_example_grids.py` + both `postprocessing/` suites: **111 passed, 2 skipped, 6 xfailed,
0 failed**.


**Do not measure this with `git stash push -- packages/`.** If anything interrupts the run between
push and pop, the work is stranded in a stash that is easy to lose. Use `git worktree add`, or the
`git archive` extraction above.

## Correction: `test_loadflows_match*` was **not** pre-existing

An earlier revision of this file listed the three `test_powsybl_backend.py::test_loadflows_match*`
tests as pre-existing Windows failures caused by a genuine numerical disagreement between the JAX DC
solver and pypowsybl -- quoting flow arrays that differed by 0.94 % and 158 %, and arguing the UCTE
case looked like a different slack treatment.

**That was wrong, and the delta above disproves it.** All three fail at `origin/main` and pass with
this PR:

```
$ uv run pytest packages/dc_solver_pkg/tests/preprocessing/test_powsybl_backend.py -q -p no:randomly
17 passed in 61.97s
```

The flow arrays *were* the symptom, but the cause was upstream: int32 index and sentinel arrays
built a different grid before the loadflow ever ran, so the two solvers were not being asked the
same question. The comparison was misdiagnosed as a solver disagreement because the earlier
"pre-existing?" check ran against a `HEAD` that already carried part of the fix.

`test_powsybl_backend.py` is expected to report **17 passed** on Windows, matching the container.

## The 63 that still fail on Windows

All 63 also fail at `origin/main`, so this PR neither causes nor addresses them. Characterising the
previous 76 is what turned up sections 4 and 5 -- 13 of them were the same C-`long` bug in files the
original sweep had not covered. What is left no longer shows that signature:

| Cluster | Count | Dominant signature |
|---|---|---|
| `postprocessing/test_validate_loadflow_results.py` | 31 | `AssertionError: n_N-n_N does not match` |
| `postprocessing/test_postprocess_powsybl.py` | 17 | `AssertionError: DC Better` / numeric mismatch |
| `preprocessing/test_parallel_switch_edge_cases.py` | 4 | `assert False` |
| `jax/test_bsdf.py` | 3 | numeric mismatch |
| `jax/test_topology_looper.py` | 2 | numeric mismatch |
| one each: `test_example_grids`, `test_loadflows_match`, `test_postprocess_pandapower`, `test_multi_outages`, `test_compute_batch`, `test_busbar_outage` | 6 | mixed |

These are numeric disagreements rather than dtype errors, and the remaining `test_example_grids.py`
failure is `test_case57_backends_match` -- pandapower against pypowsybl, not Windows against Linux.
The most likely common cause is the `tot_stat` sentinel described under "Still open", which is known
to change the loadflow result on the Windows XLA CPU backend and is deliberately untouched here.
Closing them is separate work.

---

## Still open

### `tot_stat` padding: `2**63-1` as an int64 gather index misbehaves on the Windows XLA CPU backend

`convert_tot_stat` in `jax/inputs.py:63` is the one remaining place that computes its own
C-`long`-width int max. Converting it to `int_dtype()` makes the Windows and Linux inputs
byte-identical — same dtype, same pad value, same valid range — and **Windows then fails the base
loadflow 16/16 while Linux passes on the same data** (`n_1` absmax 236.17 vs 245.6385). Windows at
`HEAD`, with int32 padding and the sentinel wrapping to −1, coincidentally matches Linux exactly.

That change was therefore **deliberately reverted** rather than shipped unvalidated. `inputs.py:63`
is self-consistent as long as it is left alone.

**This has a deadline.** Every Windows run of `tests/dc/test_main.py` emits, five times:

```
jax/_src/ops/scatter.py:93: FutureWarning: scatter inputs have incompatible types: cannot safely
cast value from dtype=int64 to dtype=int32 with jax_numpy_dtype_promotion='standard'.
In future JAX releases this will result in an error.
```

That is `tot_stat_jax.at[sub_id, : c_l[sub_id]].set(...)` — an int32 array receiving int64 values,
silently down-cast today and announced for rejection. The next JAX bump turns it into a hard failure
on Windows. It cannot surface in CI, because `tot_stat` is already int64 on Linux and no cast
happens there.

Investigating properly means finding where the extreme index reaches a gather/scatter, and whether a
smaller out-of-bounds sentinel (e.g. `n_branches`) is the right fix. That warning's stack is the
best available pointer.

### `uv.lock`: four `numba` specifier lines — resolved, and benign

Previously flagged here as unexplained. It is the opposite: **`HEAD`'s lock was stale, and the
working tree repairs it.** All four values now agree exactly with the committed manifests, which
were never modified on this branch:

| `uv.lock` block | package | manifest section | manifest declares | lock at `HEAD` |
|---|---|---|---|---|
| `dev` | `contingency_analysis` | `[dependency-groups]` | `numba>=0.60.0` | `>=0.61.2` |
| `dev` | `grid_helpers` | `[dependency-groups]` | `numba>=0.60.0` | `>=0.61.2` |
| `requires-dist` | `dc_solver` | uv-dynamic-versioning hook | `numba (>=0.60.0,<0.67.0)` | `>=0.58.1,<0.67.0` |
| `requires-dist` | `importer` | uv-dynamic-versioning hook | `numba (>=0.60.0,<0.67.0)` | `>=0.58.1,<0.67.0` |

`uv sync` — which the container entrypoint runs on every start — refreshed that recorded metadata to
match. The whole diff against `HEAD` is **4 insertions, 4 deletions**, with **no resolved version or
wheel hash changes** (`git diff -U0 HEAD -- uv.lock | grep -c sha256` -> `0`). Nothing is installed
differently; only the specifier metadata the lock had recorded is corrected.

Keep it in the PR. Dropping it leaves `uv sync --frozen` — used by `docker/Dockerfile` — validating a
lock against manifests it disagrees with.
