<!--
Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file,
you can obtain one at https://mozilla.org/MPL/2.0/.
Mozilla Public License, version 2.0
-->

# Run ToOp on Windows: a container path, and the seven bugs behind the native one

Pull request from `feat/add-docker-compose` into `main`. This file is the PR description.

ToOp had no supported path on Windows. This branch adds one -- a Docker Compose setup that is the
recommended route -- and, separately, fixes the seven defects that stopped ToOp's test suite from
passing *natively* on a Windows host. All seven are **invisible on Linux**, which is why CI never
caught any of them, and they surfaced in sequence: each was hidden behind the one before it.

Five of the six are the same underlying mistake -- **`dtype=int` in numpy is the platform's C
`long`, so it is int32 on Windows and int64 everywhere else.** It appears as an explicit `dtype=`
argument (section 2), as a value split across lines that a line-based sweep misses (also section 2),
as an implicit width that must match a *neighbouring* array (section 4), as an API *default* with no
`dtype=` in sight at all (section 5), and as a **test assertion** comparing a correct int64 column
against bare `int` (section 6). The last two are the ones worth remembering: they mean the grep that
catches the first three is not sufficient, and that `tests/` is as exposed as `src/`.

The remaining two are Windows platform semantics rather than integer width: `multiprocessing.Pipe()`
returning a different class (section 3), and a temp file that cannot be reopened by name
(section 7).

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

## 6. The same C `long`, in a test's *assertion*

`interfaces_pkg`'s `test_interface_helpers.py` checks that
`get_empty_dataframe_from_model()` gives integer columns:

```python
assert df.dtypes["a"] == int
```

```
AssertionError: assert dtype('int64') == int
```

**The production code is correct here.** pandera resolves `Series[int]` to int64 on every platform,
including Windows -- verified directly:

```
>>> get_empty_dataframe_from_model(M).dtypes["a"]      # on Windows
int64
```

It is the *assertion* that is wrong. `df.dtypes["a"] == int` asks "is this the platform's C `long`?"
when it means "is this an integer column", and on Windows the honest answer to the first question is
no. This is the fourth distinct shape of the same bug, and the one furthest from where anyone would
look for it: **neither audit rule finds it**, because the mistake is in a `tests/` assertion rather
than in an array constructor, and there is no `dtype=` anywhere on the line.

**Fix.** Compare against the concrete width, in all four places:

```python
assert df.dtypes["a"] == np.int64
```

`np.int64` rather than the string `"int64"`, which also works. The string matches the idiom two lines
away (`assert df.index.dtype == "object"`) and needs no import, but a typo in it fails *silently and
misleadingly*:

| | result |
|---|---|
| `dtype == "in64"` (typo'd string) | `False` -- fails as `assert dtype('int64') == 'in64'` |
| `dtype == np.in64` (typo'd symbol) | `AttributeError`, immediately |

In this file of all files, a typo that fails looking exactly like a real dtype mismatch is the wrong
trade for saving an import.

Two near-misses worth recording. `pd.Int64Dtype()` looks like the natural typed choice and is
**wrong** -- it is the nullable extension dtype, and compares `False` against a numpy int64 column.
And `int_dtype()`, the helper this PR adds for exactly this purpose, is **unavailable here**:
`interfaces_pkg` is the base of the dependency chain, so importing `dc_solver` would invert it.

One of the four,
`df.index.get_level_values(1).dtype == int`, had never even been reached -- the assertion above it
failed first, so it would have surfaced only after the others were fixed.

### Scope note

This is the one change in this PR outside `dc_solver_pkg` / `topology_optimizer_pkg`, and it is a
test-only change in a package the PR otherwise does not touch. It is included because it is the same
defect class, it is three words, and leaving it would mean `interfaces_pkg` stays red on Windows for
a reason already fully diagnosed here.

It is **not** a regression from this branch, and nothing in the branch could have caused it:

- `packages/interfaces_pkg/tests/test_interface_helpers.py` is unchanged by this PR (before this fix).
- `interfaces_pkg` has **no Python source change** on the branch -- only `pyproject.toml` (adds
  `grpcio-tools>=1.72`, used only by `compile_proto.sh`), the script itself, and its lock.
- The mechanism is platform-only: `np.dtype("int64") == int` is `False` on Windows, `True` on Linux.

---

## 7. `NamedTemporaryFile` cannot be reopened by name on Windows

`load_pandapower_net_via_grid2opt_for_powsybl()` round-trips a grid through a Matpower `.mat` file:

```python
with tempfile.NamedTemporaryFile(suffix=".mat", delete=True) as tmpfile:
    _ = pandapower.converter.to_mpc(net, tmpfile.name)      # <- reopens the path to write it
    pypowsybl_network = pypowsybl.network.load(tmpfile.name, loading_params)
```

`NamedTemporaryFile` keeps its own handle open for the duration of the `with`. On POSIX a second
process or library may open the same path anyway; **on Windows that handle is exclusive**, so
`to_mpc()` fails before writing anything:

```
PermissionError: [Errno 13] Permission denied: 'C:\...\Temp\tmp_4g_0z8y.mat'
  scipy/io/matlab/_mio.py:45
```

Reproducible in three lines, with no grid and no pandapower involved:

```python
import tempfile
with tempfile.NamedTemporaryFile(suffix=".mat", delete=True) as t:
    open(t.name, "w").write("x")
# Windows: PermissionError: [Errno 13] Permission denied
# Linux:   succeeds
```

**Fix.** Hand out a path inside a temporary *directory*, which we never open ourselves:

```python
with tempfile.TemporaryDirectory() as tmpdir:
    mat_path = Path(tmpdir) / "net.mat"
    _ = pandapower.converter.to_mpc(net, str(mat_path))
    pypowsybl_network = pypowsybl.network.load(str(mat_path), loading_params)
```

Same semantics on both platforms, same automatic cleanup, and no exclusive handle for anyone to trip
over. `pathlib` was already imported in the module.

This is the only `NamedTemporaryFile` in `packages/*/src` or `packages/*/tests`, so the trap does not
recur elsewhere.

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

### A measurement retraction, first

An earlier revision of this document reported a full-suite Windows comparison of **103 failures at
`origin/main` -> 63 with this PR, 40 fixed, 0 introduced**. **Those counts were wrong and are
withdrawn.** They were produced in a contaminated environment:

- The Windows runs set `PYTEST_ADDOPTS` but **not `PYTHONPYCACHEPREFIX`**, so they read and wrote
  `__pycache__` inside the repository. Container runs from before `cd1457d` -- the commit that adds
  container-side cache isolation -- had written their own `.pyc` into that same bind-mounted tree.
  The bind mount gives both sides byte-identical mtime and size, so Python trusts whichever got there
  first. This is exactly the hazard `docker/README.md` documents, walked into while doing the very
  native-vs-container comparison it warns about.
- They also invoked `.venv/Scripts/python.exe -m pytest` directly rather than `uv run pytest`, so the
  environment was never synced first.

The decisive check: with `__pycache__` cleared, `PYTHONPYCACHEPREFIX` exported and `uv run`, the same
serial `dc_solver` suite reached 45 % with **zero** failures **while sharing the machine with a
concurrent container run**. The contaminated run had 36 failures at that same point on an *idle*
machine. Since contention can only add failures, never remove them, the earlier counts cannot have
been real.

**What this does and does not affect.** It invalidates the aggregate pass/fail *counts* only. Every
fix in this PR rests on a demonstrated mechanism reproduced directly -- `OverflowError`, the `-1`
wraparound, the `.view()` itemsize `ValueError`, `high is out of bounds for int32`,
`dtype('int64') == int` -- most of them in a one-line interpreter check rather than a test run, and
each with a targeted before/after on the affected file. Those stand. The Linux no-op argument also
stands: container runs are cache-isolated by the image (`PYTHONPYCACHEPREFIX=/tmp/pycache`,
`cachedir: /tmp/pytest_cache`) and have been consistently clean throughout.

### The clean measurement

Full six-package suite, native Windows, cache isolated, via `uv run`, `-n 4 --dist loadgroup`, on an
otherwise idle machine. **Ran to completion in 34 m 23 s:**

```
1 failed, 1923 passed, 24 skipped, 8 xfailed, 21 errors
```

| | Count | Attribution |
|---|---|---|
| passed | **1923** | — |
| errors | 21 | **environmental** — all Kafka, all `DockerException: 500 Server Error for http+docker://localnpipe/version`. The Docker daemon was unhealthy; these tests need a broker and never started. |
| failed | 1 | `test_load_pandapower_net_via_grid2opt_for_powsybl` — **since fixed, see section 7** |

**Nothing in `dc_solver_pkg` or `topology_optimizer_pkg` failed** — the two packages this PR modifies.

That single failure was the `NamedTemporaryFile` bug, fixed after this measurement and verified on
both platforms (Windows `1 passed in 6.44s`, container `1 passed in 3.73s`). **With it fixed, the
expected Windows result is 1924 passed and zero failures**, leaving only the 21 Kafka errors from the
unhealthy Docker daemon — which are not test failures and clear once the daemon is restarted.

### Per-fix evidence

Since the aggregate counts are withdrawn, each fix stands on its own reproduction:

| Fix | Mechanism, reproduced directly | After |
|---|---|---|
| 1 TensorBoard colons | `OSError: [WinError 123]` before the first epoch | `tests/dc/test_main.py` 8 passed |
| 2 `dtype=int` | `OverflowError` on `jnp.array`; `jnp.full` silently storing `-1` | `genetic_functions` 106 passed; `test_mutate_disconnections` 32 passed (was uncollectable) |
| 3 `_ConnectionBase` | beartype rejects `PipeConnection`; child dies, parent blocks in `recv()` | `test_runner.py` 2 passed in 47 s (was a 300 s timeout) |
| 4 `find_bridges` | `ValueError: When changing to a larger dtype ...` from mismatched itemsize | the `.view()` failures pass |
| 5 `np.random.randint` | `ValueError: high is out of bounds for int32`, then `... cannot be used to generate a random.Random instance` | `test_case9241_pp` passes |
| 6 test assertion | `assert dtype('int64') == int` -> False on Windows only | `test_interface_helpers.py` 4 passed, both platforms |
| 7 `NamedTemporaryFile` | `PermissionError` reopening an open temp file by name | `test_load_pandapower_net_via_grid2opt_for_powsybl` passes on both |

### Reproducing these numbers

The exact shell preamble -- cache isolation, Ray cleanup, and the invocation -- is in
[`docker/README.md`](docker/README.md#the-full-git-bash-session-setup), with the container
equivalent. Three things there are easy to skip and each one invalidates a comparison: the cache
variables must be exported *before* `python` starts, `uv run` must not be bypassed, and the two sides
must be run one at a time.

### Which compute device each run actually used

Every number in this document was produced on **CPU, on both platforms**. That is deliberate: it
leaves the operating system as the only variable. Running one leg on a GPU would have made the two
incomparable, and float64 differences would have muddied exactly the numeric failures being
attributed.

Verified directly, not inferred from configuration:

| Environment | `jax.devices()` | GPU |
|---|---|---|
| Windows host `.venv` | `[CpuDevice(id=0)]`, `jax_cuda12_plugin` not installed | **no, and no path to it** |
| container `toop` (CPU service) | `[CpuDevice(id=0)]` | no |
| container `toop-gpu` (`gpu` profile) | `[CudaDevice(id=0)]` | **yes** |

**Native Windows cannot use the GPU at all.** JAX publishes CUDA wheels for Linux x86-64 only; there
is no native-Windows CUDA jaxlib. The GPU path on a Windows machine is the `toop-gpu` container via
WSL2, which is why the compose setup carries a separate GPU service rather than a flag.

The GPU service itself is confirmed working (`1b764b5`): `device_banner.py` reports
`CudaDevice(id=0)` on an RTX 3050 Laptop (driver 526.56, 4 GB), and a float64 matmul executes on it —
1.50 s on the first call, 0.19 s once compiled. `device_banner.py` proves this by *running* a
computation rather than trusting the device list, because three of the four things that enable the
GPU fail silently (see Part 1).

**The solver suite has never been run on the GPU.** That is a genuine gap, not an omission from this
PR's scope — the fixes here are dtype- and path-related and are device-independent. Two things to
expect when someone does it: with ~3 GiB usable VRAM, `lf_config.batch_size` will likely need
lowering from 8, or `XLA_PYTHON_CLIENT_PREALLOCATE=false`; and ToOp forces `jax_enable_x64`
(`dc/main.py:239`, `preprocess/convert_to_jax.py:65`) while consumer GeForce cards run float64 at
~1/32 of float32, so the GPU only pays off on grids large enough to fill it. Measure any GPU run
twice — the first costs ~30 s in CUDA context creation and XLA compilation.

## Correction: `test_loadflows_match*` was **not** pre-existing

An earlier revision of this file listed the three `test_powsybl_backend.py::test_loadflows_match*`
tests as pre-existing Windows failures caused by a genuine numerical disagreement between the JAX DC
solver and pypowsybl -- quoting flow arrays that differed by 0.94 % and 158 %, and arguing the UCTE
case looked like a different slack treatment.

**That was wrong.** All three pass on Windows:

```
$ uv run pytest packages/dc_solver_pkg/tests/preprocessing/test_powsybl_backend.py -q -p no:randomly
17 passed in 61.97s
```

They also pass in the clean full-suite run. What remains uncertain is *why* the earlier run showed
them failing: that baseline came from the contaminated environment described under
[Verification](#a-measurement-retraction-first), so the "fails at `origin/main`, passes here"
framing is no longer supported either. What is certain is the negative claim -- **these are not a
standing JAX-vs-pypowsybl numerical disagreement on Windows**, and the flow arrays quoted in the
earlier revision should not be treated as evidence of one.

`test_powsybl_backend.py` is expected to report **17 passed** on Windows, matching the container.

## What still fails on Windows

### Nothing, once the Docker daemon is healthy

The suite's only real failure — `test_load_pandapower_net_via_grid2opt_for_powsybl` — is **fixed in
this PR**, see [section 7](#7-namedtemporaryfile-cannot-be-reopened-by-name-on-windows). It was
pre-existing (`grid_helpers_pkg/src` had zero changes on this branch before that fix) but it is the
same class of defect as the rest, so it is closed here rather than deferred.

### Twenty-one Kafka errors — environmental

Every one is `DockerException: ... 500 Server Error for http+docker://localnpipe/version`. The Docker
daemon was unhealthy for the duration. These need a broker and never started, so they are not
failures of the suite. Re-run once `docker ps` answers cleanly, or exclude with `-m "not kafka"`.

### Worker crashes under memory pressure — resolved, and worth knowing

Earlier runs saw three tests kill their xdist worker outright
(`node down: Not properly terminated`, no traceback):

| Test | Package |
|---|---|
| `test_stored_action_set_large_performance` | `interfaces_pkg` |
| `test_make_action_repo_large@performance` | `dc_solver_pkg` |
| `test_mutate_on_same_repository_with_different_keys` | `topology_optimizer_pkg` |

**All three pass on an idle machine** — 157.94 s, and 290.63 s for the slowest — and the run that
contains them finished in 34 m rather than 1 h 44 m. They are the three heaviest tests in the suite,
and they die only when a *second* suite is competing for memory on the same host, which is what was
happening during the earlier measurements.

An earlier revision of this file called them "reproducible crashes under `-n 4`" and, before that,
attributed one of them to timeout-under-contention. Both were wrong in different directions. The
accurate statement is: **do not run the native and container suites concurrently** — a 4-worker run
on each side is 8 JAX processes plus Ray, and the heaviest tests are the ones that die.

The failure mode is worth documenting even so, because it is silent and bimodal: xdist replaces the
dead worker, and if the replacement cannot initialise under the same pressure the controller waits on
it forever. Full chain and `py-spy` signature in
[`docker/README.md`](docker/README.md). Passing `--max-worker-restart=0` turns the hang into an
immediate error naming the test.

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
