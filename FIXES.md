<!--
Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file,
you can obtain one at https://mozilla.org/MPL/2.0/.
Mozilla Public License, version 2.0
-->

# Code fixes on `feat/add-docker-compose`

Changes to `packages/` made so the DC optimizer runs natively on Windows. Container and tooling
changes are documented in [`docker/README.md`](docker/README.md); this file covers only the code.

Both defects are **invisible on Linux**, which is why CI never caught either. The second was hidden
behind the first.

---

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

The rule is recorded under "JAX conventions" in `CLAUDE.md`. It applies to `tests/` as well — the
shared `conftest.py` fixture was building int32 `ActionSet`s (`substation_correspondence=i32[2000]`),
which produced the last three failures.

---

## Verification

| Suite | Windows before | Windows after | Linux container |
|---|---|---|---|
| `tests/dc/test_main.py` | all fail (`WinError 123`) | **8 passed** | **8 passed** |
| `tests/dc/genetic_functions` | 74 collected, 63 passed / 11 failed (+1 file uncollectable) | **106 passed** | **106 passed** |
| `test_mutate_disconnections.py` | 0 (collection error) | **32 passed** | **32 passed** |

Windows now collects and passes exactly what Linux does. `ruff check` reports the same 6 pre-existing
errors as `HEAD`; formatting is clean.

No regressions: comparing failure sets against a pristine `HEAD` on Windows produced an empty
"introduced by the fix" set, and one test that failed at `HEAD` now passes
(`test_scoring_functions.py::test_translate_topology`).

---

## Known pre-existing Windows failures — not caused by these changes

Each verified by running the same file against a pristine `HEAD` and diffing the failure sets.

| Test | Cause |
|---|---|
| `test_powsybl_backend.py::test_loadflows_match` | pypowsybl sign-convention mismatch — the two flow vectors come back as near-exact negations (`-125.0` vs `125.0`, `-46.19` vs `38.31`) |
| `test_powsybl_backend.py::test_loadflows_match_bat_hvdc_shunt_svc` | same |
| `test_powsybl_backend.py::test_loadflows_match_ucte` | same |
| `jax/benchmarks/test_runner.py::test_runner` | hangs to the 300 s timeout. `mp.Pipe()` returns `PipeConnection` on Windows, but `runner.py:70` annotates the parameter as `Connection`; with `ENABLE_BEARTYPE=true` the child process is killed on entry and the parent blocks forever in `parent_conn.recv()`. Same family as the dtype bug: an annotation that is only true on POSIX. `benchmark_utils.run_task_process` carries the identical annotation. |

`test_powsybl_backend.py` is therefore expected to report **3 failed, 14 passed** on Windows. All
four pass in the container.

Before concluding that your own change broke something on Windows, compare against `HEAD` in a
worktree — **not** with `git stash push -- packages/`, which strands the work in a stash if
interrupted between push and pop:

```bash
T=packages/dc_solver_pkg/tests/preprocessing/test_powsybl_backend.py
git worktree add /tmp/toop-head HEAD
(cd /tmp/toop-head && uv run pytest $T -q -p no:randomly --tb=no | tail -1)
git worktree remove /tmp/toop-head
```

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

### `uv.lock` carries four `numba` specifier edits with no matching `pyproject.toml` change

`>=0.61.2` → `>=0.60.0` in two dev groups, `>=0.58.1,<0.67.0` → `>=0.60.0,<0.67.0` in two
`requires-dist` blocks. These mirror manifest constraints that no tracked `pyproject.toml` declares,
so the lock and the manifests disagree — and `uv sync --frozen` in the Dockerfile validates one
against the other. Origin unknown; resolve before merging.
