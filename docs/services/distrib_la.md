# distrib_la — distributed dense linear algebra over a JAX device mesh

`services/distrib_la/`. Independently installable (`pyproject.toml`,
src-layout); depends on `lxkit` + jax and nothing else in LORRAX. Every
vendor library it can reach — ScaLAPACK, SLATE, cuSOLVERMp — appears in
exactly one dependency edge (`distrib_la.loader`, which `dlopen`s a `.so`
by path) and in zero declared dependencies.

## Purpose

One door for `eigh`, `cholesky` and `solve_lu` on an `('x','y')` device
mesh, so a driver says *what* to compute and *where*, never *which
library*. Before this service the same three ops were dispatched from ten
`src/` modules through four packages (`ffi.linalg`, `ffi.slate`,
`ffi.scalapack`, `common.cholesky_2d`), with per-call-site guard ladders
that disagreed; the worst measured consequence of that drift was a silent
route change that ran to completion with `rc=0` and a QP gap of **−161 eV**.
Backend selection is declarative data, backend *capability* is probed once,
and a resolved backend name is a **promise** that every guard passed.

The service is the door. `import distrib_la` and use top-level names;
`from distrib_la.plan import …` from outside the package is a layering
failure that `tests/test_layering.py` fails on (with a red twin).

## API

| name | what it is |
|---|---|
| `plan(op, mesh, *, backend='auto', n=None) -> Plan` | Resolve once, then call. **Eager** — dlopens, probes, reads `jax.process_count()`. |
| `Plan(A)` / `Plan.batched(A_stack)` | One tile at `P('x','y')` / a stack at `P(None,'x','y')`. **Trace-safe** — no dlopen, no `device_put`, no process count inside. |
| `Plan.is_native`, `.backend`, `.describe()` | The resolved fact, readable; never something a caller must branch on to be correct. |
| `Plan.native_fn` | A pure closure for a fusion-critical site that needs the math inside its own `jit`. Native backends only. |
| `factor(op, A, mesh, *, backend, ...) -> FactorToken` | Factor once. |
| `solve(token, B) -> jax.Array` | Back-solve many. The token carries the handle (ScaLAPACK `ipiv`, cuSOLVERMp raw buffer, SLATE `SlateLowerL`). |
| `resolve_backend(op, requested, mesh, *, n=None) -> str` | The **raising** probe. `n` is decoupled from the operand so a caller that will pad can ask before it has built anything. |
| `list_backends(op, mesh) -> dict` | The **never-raising** report, for startup banners. |
| `BACKEND_CHOICES`, `EIGH_BACKENDS`, `CHOLESKY_BACKENDS`, `LU_BACKENDS`, `OPS`, `NATIVE` | The vocabulary. Importable with **no `.so` anywhere on the machine** — a deck parser must not need the FFI layer to read a deck. |
| `mesh_key(mesh)`, `mesh_platform(mesh)`, `mesh_is_cpu(mesh)` | Stable hashable mesh identity (axes, extents, platform, device ids) and its two predicates. `mesh_key` is for any cache whose stored value does **not** retain the mesh; `id(mesh)` there is the documented drift. |
| `dial_key()` | Factory-time dials folded into one tuple, for kernel cache keys. |
| `probe_target`, `has_target` | Capability, with the ABSENT / BROKEN split (`lxkit.probe`). |
| `dispatch_batched_eigh(A, mesh, backend)` | The one legacy entry point kept for `gw.qsgw_density`. |

Two phases and they stay two: only platform and handler guards can fire at
resolve time — operand dtype, rank and extent are trace-time facts — so a
single-phase API would have to lie about when it checked.

## Contract

* **Promise semantics.** A returned backend name means every guard passed:
  platform, compiled handler, one-process-per-device coverage, mesh
  geometry, divisibility. The call cannot then fail for an availability or
  geometry reason.
* **Explicit requests REFUSE**, with the failed guard named and a fix.
  Only `auto` demotes, and it announces once, from the rank it happened on.
  A grammar error (an unknown spelling) falls to the gate DEFAULT.
* **Exception types are part of the API**: `ValueError` / `RuntimeError`,
  each constructible from one string. `bandstructure/bse_setup.py:386-403`
  re-raises `type(exc)(_why)`; a service-specific exception class would
  escape that handler and delete the `use_low_mem_eigh` refusal message.
* **Layout.** Eigenvalues come back replicated; eigenvectors as **COLUMNS**,
  on every backend. Batched inputs are `P(None,'x','y')`, single tiles
  `P('x','y')`. `ensure_sharding` is the one place that is spelled.
* **Donation** is declared per op, not per call site:
  `eigh` donates nothing, `cholesky` donates argument 0, `solve_lu` donates
  0 and 1. A donated operand must be a fresh value at the call site.
* **Tokens are opaque.** `FactorToken` exposes `op`, `backend`, `mesh`, `n`,
  `nbatch` and no factor. "Never reshard it, feed it back verbatim" is the
  type rather than a comment. It is deliberately **not** a JAX pytree, so a
  `jit` boundary refuses it by name instead of tracing a block-cyclic
  handle (pinned by `test_a_factor_token_cannot_enter_the_composed_kernel`).
* **`solve()` checks `B` against the token's `n` and `nbatch`**, so a
  mismatched RHS refuses instead of corrupting a solve or hanging in a
  collective.
* **`factor`/`solve` is not the fused route.** `factor('solve_lu', …)`
  raises `NotImplementedError` for any backend but ScaLAPACK; one factor per
  solve is `plan('solve_lu', mesh, backend=…).batched(A, B)`.
* **Env grants capability, never selects a backend.** `LORRAX_FFI_SO` /
  `LORRAX_FFI_HOST_SO` pin which `.so` to open. `distrib_la.resolve` reads
  no environment at all. An explicit pin that cannot be honoured is a
  refusal, never a fall-through.

## Backends

Preference is declarative data with every supported platform represented,
including the declared-untested tier.

| op | vocabulary | `auto` | `distributed` resolves to |
|---|---|---|---|
| `eigh` | `auto off distributed cusolvermp slate scalapack` | `native` | cpu → **scalapack**, CUDA → **cusolvermp**, ROCm → **slate** |
| `cholesky` | `auto off native2d cusolvermp slate` | `native` | — (its CPU story is a channel-policy ladder in the caller, not one library) |
| `solve_lu` | `auto off distributed cusolvermp scalapack` | `native` | cpu → **scalapack**, CUDA → **cusolvermp** |

* `native` is the floor everywhere and the measured default at every
  production tile size. `auto` never picks an FFI backend.
* `native2d` is the 2-D block-distributed tiled Cholesky (`_native2d`, the
  old `common/cholesky_2d`). Pure JAX, every platform, but a *different
  algorithm*: at n=10k on P=128 the replicated-reshard route costs 1.6
  GB/device and this one 5 MB/device. `auto` never picks it — a cost model
  that differs by three orders of magnitude in **both** directions is the
  caller's decision.
* **ROCm is DECLARED-UNTESTED.** The preference rows exist so the routing
  question has an answer; LORRAX builds no ROCm `.so`. Known measurement
  gap, recorded in `resolve.py`: on the jaxes this tree runs,
  `Device.platform` is `'gpu'` for both vendors, so a real ROCm mesh lands
  on the CUDA row. Disambiguating is **one row** and needs a machine.
* **SLATE `heev` on a host mesh is refused outright** — bug L-2,
  deterministic SIGSEGV against MKL/LibSci LAPACK. Two contract cells carry
  the skip by design.
* **There is no ScaLAPACK `potrf` handler anywhere in the tree**, and no
  SLATE `getrf`. The vocabularies above are the whole truth.
* **The two platform `.so`s share their SLATE.** Both `liblorrax_ffi.so` and
  `liblorrax_ffi_host.so` carry `NEEDED libslate.so.2` and `NEEDED
  libblaspp.so.2`, resolved out of different builds; `ld.so` keys a loaded
  object by SONAME, so the first one opened decides which
  `blas::get_device_count()` the other calls, and the host build's is a
  compiled-in 0. Both loaders therefore open **CUDA before cpu**
  (`_open_cuda_before_host`). See Antipatterns.

## Tests

Four tiers, markers `services` + `distrib_la`, applied by a collection hook
(a `pytestmark` in a conftest is silent — `tests/test_service_selection.py`
measures that the marks arrived).

| tier | file | needs |
|---|---|---|
| L-a shape/contract algebra | `test_distrib_la_shape_algebra.py` | nothing — a laptop, milliseconds |
| L-b emulated multi-device | `test_distrib_la_emulated_mesh.py` | `XLA_FLAGS` set by the SERVICE conftest; **skips**, never asserts, below 4 devices |
| L-c real multi-process | `test_distrib_la_multiproc.py` | `srun -n 4`; shared `check_*(mesh, …)` bodies + a `__main__` CLI (`_CLI_CELLS`) — same functions, no duplicated logic |
| contract + wiring | `test_distrib_la_contract.py` | the `.so` pins; every refusal constructibly fires |
| C++ / ELF acceptance | `test_so_acceptance.py` | binutils + a pinned `.so`; reads the ELF, never dlopens |
| import isolation | `test_distrib_la_import_isolation.py` | `python -S` subprocess; `sys.modules` AND `sys.path` asserted, plus a red twin and a with-lorrax-still-passes |
| skip honesty | `test_distrib_la_skip_honesty.py` | a machine profile; ABSENT = skip, BUILT-AND-BROKEN = **FAIL** |

* **Hostile geometry is mandatory**: a real 2×2 with non-dividing extents
  and padding round-trips, with the anti-tautology self-assertion (the pad
  divisor must be provably non-vacuous).
* **Every check ships with the case where it returns FALSE.** No exceptions.
* Run it standalone: `pytest services/distrib_la/tests` (never loads
  `tests/conftest.py`). Run it from the monorepo: `pytest -m distrib_la`.
  Deselect: `--no-services` / `--only-service=NAME`, never a second `-m`
  (`pyproject` sets `addopts = "-m 'not extra'"` and an explicit `-m`
  REPLACES it, silently re-enabling 26 deselected suites).
* Perlmutter floor, 2026-08-07, HEAD `32e61fe`, BUILD_NOTES pins: full-suite
  `-m distrib_la` **130 cells / 0 failed / 22 skipped**; service-only by
  path **250 / 0 / 3**.

## Performance

Recorded baselines (never slow tests) in
`services/distrib_la/bench/baselines/{cpu,gpu}{1x1,2x2}.json`: one row per
(op × backend × shape), with `seconds`/`min`/`max`, `compile_seconds`, the
resolved backend, and — for a combination that cannot run — the refusal
text instead of a number. All rows below: Perlmutter, **jobid 56444350**,
1 node, jax `0.7.0.dev20260807`, complex128, BUILD_NOTES `.so` pins.
Regression detection = diffing baseline files across branches. **Step 6
owns new measurements; these are step 2's and are cited, not re-run.**

**`auto` resolves to `native` everywhere, and the numbers say why.**

| op | backend | mesh | shape (nq, n) | s |
|---|---|---|---|---|
| eigh | native | cpu 1×1 | (2, 1024) | 1.446 |
| eigh | scalapack | cpu 1×1 | (2, 1024) | 1.451 |
| eigh | scalapack | cpu 2×2 | (2, 1024) | 1.099 |
| eigh | native | gpu 1×1 | (2, 1024) | 0.044 (via `plan`, `cusolvermp` 0.0435) |
| cholesky | native2d | gpu 2×2 | (8, 256) | 0.022 |
| cholesky | native2d | cpu 2×2 | (2, 1024) | 0.375 |

**cuSOLVERMp `eigh` on a 4-process 2×2 costs a flat ~1.55 s PER MATRIX,
independent of size.** That is the collective, not the factorization:

| shape (nq, n) | total s | s / matrix |
|---|---|---|
| (2, 64) | 3.117 | 1.56 |
| (2, 256) | 3.201 | 1.60 |
| (2, 1024) | 3.525 | 1.76 |
| (8, 256) | 12.713 | **1.59** |

64×64 and 1024×1024 cost the same. A per-matrix cost that does not move
with n⁴× the work is a fixed collective/context charge, so cuSOLVERMp is
for the regime where a tile does **not fit on one device**, and using it
below that is pure loss. `slate` `eigh` on the same 2×2 is 0.876–4.555 s
(cheaper than cuSOLVERMp at every shape measured, and still ~200× the
native 1×1 time).

**SLATE `factor`+`solve` (potrf + two trsm passes), the route H3 moved
onto**, cpu 2×2: 0.727–2.945 s; gpu 2×2: 1.397–7.237 s. Its
**correctness** is the load-bearing result, and it is first-execution
evidence — the SLATE trsm back-solve had never run before step 2:

| leg | residual vs the native reference | bar |
|---|---|---|
| SLATE trsm, CPU c128, per-q | 3.8e-16 / 4.5e-16 | rtol 1e-12, NOT relaxed |
| SLATE trsm, GPU c128, per-q | 1.0e-15 / 6.9e-16 | " |
| SLATE token round trip, GPU 2×2 | 4.4e-16 | " |
| SLATE token round trip, CPU 2×2 | 4.3e-16 | " |
| cuSOLVERMp token round trip, GPU 2×2 | 5.6e-16 | " |
| ScaLAPACK LU + ipiv, CPU 2×2 | 3.855e-13 | native LU control 3.856e-13 |
| batched-vs-serial eigh | **bit-identical** (0.0 in W and Z) | — |

## Antipatterns

* **Editing `src/ffi/<name>/`.** `src/ffi/linalg/`, `src/ffi/slate/`,
  `src/ffi/scalapack/` and `src/common/cholesky_2d.py` are **deleted**
  (17 files, commit `b3f3675`). A backend lives in
  `services/distrib_la/src/distrib_la/_<name>.py` plus one row in each of
  `loader._*_TARGET_SYMBOLS`, `resolve._SPEC`/`BACKEND_CHOICES` and
  `plan._IMPL`. `docs/dev/linalg_ffi.md` § "Adding a backend" is the
  procedure; the C++ under `src/ffi/cpp/` did not move and its target
  strings are frozen.
* **Selecting a backend from the environment.** There is no env var that
  does it and adding one is the antipattern. Deck keys choose
  (`eigh_backend`, `distributed_cholesky`, `distributed_lu`,
  `w_dyson_solver`); env only says which `.so` exists.
* **Comparing Z across meshes.** Eigenvectors are gauge-dependent: a
  degenerate subspace has no canonical basis and two meshes will return
  different (equally correct) columns. Compare **eigenvalues**, or a
  gauge-invariant contraction (`Z diag(W) Zᴴ`, subspace projectors,
  `|Zᴴ Z'|`). A test that diffs Z across meshes will pass on Si and fail on
  the first degenerate system.
* **Reading token internals.** `token._factor` is private and its layout is
  block-cyclic on that mesh's specific grid. Reaching for `.shape` on a
  token is the same error one level up — it has `.n` and `.nbatch`.
  Anything that needs the factor's *bytes* wants a different API, not this
  one.
* **Branching on `token.backend` / `plan.backend` outside this package.**
  They are introspection (a banner, a test's message). A caller that
  branches on them has re-implemented the resolver, and the two will drift.
* **Re-deriving a mesh identity by hand.** Use `mesh_key(mesh)`. `id(mesh)`
  as a cache key is only safe when the cached value retains the mesh, and
  the case where it does not is exactly where somebody re-spells it.
* **Bypassing the `sys.path` bootstrap.** `services/*/src` is on no
  `PYTHONPATH` any launcher sets: `lx` rewrites the container `PYTHONPATH`
  to exactly `<checkout>/src` and the Shifter image pip-installs nothing.
  A module that imports `distrib_la` must call `ffi._services.ensure_on_path()`
  first (`tests/test_service_path_bootstrap.py` runs the bare-`sys.path`
  import in a subprocess). This is transitional plumbing with an owner
  decision behind it — do not paper over it with a `sys.path.insert` of
  your own, and do not assume a green pytest leg proves it: the service
  conftest puts services on the path during collection, so a broken
  bootstrap is a **green-suite / red-cluster** failure.
* **Assuming the two platform `.so`s are independent.** They share
  `libslate.so.2` / `libblaspp.so.2` by SONAME. Opening the host library
  first gives every CUDA SLATE handler a `blas::get_device_count()` of 0
  and a `FAILED_PRECONDITION` naming a device count nobody set. Do not
  "fix" that by unsetting `CUDA_VISIBLE_DEVICES` — measured, it is not a
  visible-device problem; the failing leg had exactly one visible GPU.
* **Making the refusal path a `try/except` at the call site.** An explicit
  backend that cannot be honoured must raise out of the driver. Six probe
  calls in `isdf/core.py` exist only for their raise; wrapping any of them
  turns a loud refusal into a silent different-backend run
  (`tests/test_charge_zeta_route.py` pins all six).
