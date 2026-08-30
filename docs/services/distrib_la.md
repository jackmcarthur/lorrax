# distrib_la — distributed dense linear algebra over a JAX device mesh

`services/distrib_la/`. Independently installable (`pyproject.toml`,
src-layout); its runtime dependencies are `lxkit`, JAX, and NumPy. It imports
nothing from LORRAX's `src/` or `common` packages. Every vendor library it can
reach — ScaLAPACK/PBLAS, SLATE, cuSOLVERMp/cuBLASMp — appears behind exactly
one dependency edge (`distrib_la.loader`, which `dlopen`s an optional provider `.so` by path)
and in zero declared Python dependencies.

## Standalone installation and capability model

Python 3.12 or newer is required. From this source tree, install `lxkit` and
`distrib_la` as two distributions; the editable spelling is useful for
development but is not required:

```bash
cd services/distrib_la
python -m pip install -e ../lxkit -e '.[test]'
python -c "import distrib_la as d; print(d.BATCHED_ROUTE_CHOICES)"
python -m pytest tests/test_distrib_la_shape_algebra.py \
  tests/test_distrib_la_emulated_mesh.py \
  tests/test_distrib_la_import_isolation.py \
  tests/test_distrib_la_batch_reshard.py \
  tests/test_distrib_la_matmul.py \
  tests/test_distrib_la_matmul_plan.py
```

An installed consumer imports only `distrib_la`'s top-level names. It does
not call LORRAX's source-path bootstrap and does not need a LORRAX Python
installation. Native JAX routes, capability reporting, and backend vocabulary
work with no shared library present. NumPy is declared directly because the
process-local placement and cuSOLVERMp context code import it at runtime; it
is not merely a test dependency.

`loader.get_lib()` has one optional import: it tries `h5py` in a caught block
before `dlopen` so an already-installed h5py establishes the safe
process-wide HDF5 symbol order. Absence is accepted and no public service API
uses h5py, so it is intentionally not a hard dependency or package extra.

The optional FFI provider is a separate capability. Set `LORRAX_FFI_SO`
(CUDA) or `LORRAX_FFI_HOST_SO` (CPU) to an absolute path to a compatible
library. Those historical environment names describe the provider ABI; they
do not create an upward Python dependency. The current provider is built by
LORRAX's C++ tree and exports a versioned handler ABI mirrored by this package.
An absent `.so` does not break import, while an explicit missing, unstamped in
strict mode, or ABI-incompatible pin refuses rather than falling through.
`batched_route='batch_reshard'` with `backend='off'` is the completely
provider-free spelling.

## Known limitation — `distributed_eigh` hangs at a 3×3 mesh for n ≥ 3072

**Read this before you plan a large deck.** `distributed_eigh` is not
currently safe on a 3×3 device mesh once the matrix reaches n = 3072. It does
not fail and it does not raise; it hangs silently and forever. The cuSOLVERMp
banner prints, nothing follows it, and the job sits there until the scheduler
or a human kills it. Runs were killed at 420 s and, in one case, at 900 s
without the call ever returning.

The break is bracketed tightly. At a 3×3 mesh, n = 2049 completes in 6.9 s
with a maximum eigenvalue error of 1.5e−11, and n = 3072 never returns. Every
larger size tested at that mesh — 3072, 4098, 6144, 8190 — hangs the same way.
A 2×2 mesh at n = 8192 is fine, finishing in 9.3 s, so this is specific to the
3×3 geometry rather than a general size ceiling.

**The allocator is exonerated.** The hang reproduces identically under the
`platform` allocator, which is what the fleet runs today, and under the
recommended BFC settings. Whatever is wrong at 3×3 was wrong before the
allocator question was ever asked, and it will still be wrong if the allocator
recommendation is rejected. The card reported 36.4 GiB free at the moment of
the call, so this is not memory starvation either.

What this means in practice is that **large-deck users must not assume
`distributed_eigh` works past roughly n ≈ 2k until this is closed.** If your
stage calls `eigh` on a big matrix, run it on a 2×2 mesh, or keep the matrix
below the bracket, or use a non-distributed route. A 3×3 mesh with a large
matrix will not give you a wrong answer — it will give you no answer at all,
which in a batch queue is an expensive way to find out.

Two things are still owed here. The first is a control: a 4×4 leg at n = 8192
is what separates "3×3 is odd because it is not a power of two" from "anything
past 2×2 is broken". That leg never got a placement before the measuring
window closed, and the log it did leave carries no cuSOLVERMp line at all, so
its non-zero exit is a queue artifact and must not be read as a hang. The
second is the root cause itself, which needs a multi-node GPU allocation to
chase and therefore has not been attempted. One unconfirmed lead, offered as a
lead: at a 3×3 grid, n = 2049 gives a local block of 683, below the usual 1024
tile, while n = 3072 gives exactly 1024 per rank and so more than one block
column per rank for the first time. A hang appearing exactly where the 2-D
block-cyclic distribution stops being trivial on a non-power-of-two grid is a
plausible library-side story, but nobody has confirmed it.

The owner's disposition is **revisit soon**. This is a `distrib_la` solver
defect and it belongs to whoever owns `distrib_la`.

The seven-leg evidence table, the exact reproduction and the probe live in the
2026-08-09 amendment to `tests/KNOWN_FAILURES.md` (§1, "`distributed_eigh`
hangs at a 3×3 mesh for every n ≥ 3072"). The run artifacts are under
`/pscratch/sd/j/jackm/sigma_scaling_0809/_reports/` in legs `la_p4_bfc85`,
`la_p9_bfc85`, `la_p9_platform`, `la_p9_small_bfc85`, `la_p9_3072`,
`la_p9_mid` and `la_p9_6144`; the probe is
`sigma_scaling_0809/probe_pressure_sq.py`, and the prose write-up is
`SIGMA_SCALING.md` §8. `la_p16_8192` is the leg that did not run.

## Purpose

The same top-level door also owns the square-matrix polar/SVD operation used
for parallel-transport links.  It composes the planned distributed Hermitian
eigensolver rather than introducing a fifth vendor dependency or allowing a
caller-side dense SVD.

One door for `polar_factor`, `eigh`, `cholesky`, `solve_lu` and `matmul` on an
`('x','y')` device mesh, so a driver says *what* to compute and *where*,
never *which library*. Before the GEMM surface landed, the same three solver ops were dispatched from ten
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

The polar additions are:

| name | what it is |
|---|---|
| polar_factor(A, mesh, backend='distributed', rcond=None) -> (L, s) | Cached one-shot square polar factor. A and L are P('x','y'); descending s is replicated. |
| plan_polar_factor(mesh, n=..., backend='distributed', rcond=None) -> PolarPlan | Eagerly resolve once for a streamed k-point loop. The returned operation is trace-safe. |
| PolarPlan(A) | One physical square matrix only. No batch axis: the preprocessing design streams one neighbour at a time. |

### Polar factor / SVD contract

The operation diagonalizes the Hermitian dilation

    H = [[0, A], [A.H, 0]]

with one planned eigh call and forms L = U V.H from its positive-eigenvalue
subspace.  It never diagonalizes A.H A: that Gram construction squares the
condition number and loses precisely the small overlap singular values needed
as the manifold-quality diagnostic.  Singular values are non-negative and
descending, matching NumPy SVD order.

The physical input is exactly one rank-2 square float64 or complex128 array at
P('x','y') on the supplied mesh.  The service refuses rank, shape, dtype,
mesh-axis, divisibility, and concrete-layout mismatches before numerical work.
There is no implicit device_put or full-matrix reshard.  L has the same shape,
dtype and sharding.  Only s, a length-n real vector, is replicated.

rcond is relative to max(s).  None means n times machine epsilon for the real
component dtype.  Directions at or below the cutoff do not contribute to L.
Consequently a rank-deficient matrix returns the unique polar partial
isometry.  The service does not invent a backend-dependent unitary pairing of
independent left and right null spaces.  Full-rank overlap matrices return the
usual unitary polar factor.

P('x','y') requires the physical extent to divide both mesh axes.  For a
non-divisible logical band count, zero-pad rows and columns to the next common
multiple, factor that physical matrix, and slice the leading logical block of
L and leading logical singular values.  The zero pad is safe because its null
directions are thresholded out.  plan_polar_factor refuses a non-divisible
physical extent and reports the minimum pad extent rather than rounding
silently.

Planning and execution remain separate.  Hoist plan_polar_factor out of the
IBZ loop; it resolves and probes the backend once.  PolarPlan is trace-safe and
the dilation, planned eigh, masking, and final sharded GEMM are cached as one
fused operation per mesh/shape/backend/dtype/cutoff signature.  The convenient
polar_factor call caches that plan for eager streamed calls, but deliberately
refuses entry from an outer trace and points to the planned form.

Expected array scaling over the design envelope is O(n^2/P) per process for
every matrix-shaped value.  The dilation and its eigenvectors each have 4n^2
global elements; A and L each have n^2; no n^2 object is replicated.  The only
replicated result is n real singular values.  Runtime is the cost of one 2n
Hermitian eigensolve plus one n-cubic distributed GEMM.  The 3x3 large-eigh
hang documented at the top applies to the dilation extent 2n as well.

### Planned factorization API

| name | what it is |
|---|---|
| `plan(op, mesh, *, backend='auto', n=None, batched_route='auto') -> Plan` | Resolve once, then call. **Eager** — dlopens, probes, reads `jax.process_count()`. `batched_route='batch_reshard'` opts into the staged local route. |
| `Plan(A)` / `Plan.batched(A_stack)` | One tile at `P('x','y')` / a stack at `P(None,'x','y')`. **Trace-safe** — no dlopen, no `device_put`, no process count inside. |
| `Plan.is_native`, `.backend`, `.describe()` | The resolved fact, readable; never something a caller must branch on to be correct. |
| `Plan.batched_route`, `BATCHED_ROUTES`, `BATCHED_ROUTE_CHOICES`, `BATCHED_SCAN_UNROLL` | HOW a stack runs: a `lax.scan` over the single-matrix op, the backend's stacked entry, or staged batch-axis movement around a local native kernel. **The one place that decides.** Public selection is `auto|batch_reshard`; `scan`/`backend_batched` remain internal resolutions. |
| `Plan.native_fn` | A pure closure for a fusion-critical site that needs the math inside its own `jit`. Native backends only. |
| `factor(op, A, mesh, *, backend, ...) -> FactorToken` | Factor once. |
| `solve(token, B) -> jax.Array` | Back-solve many. The token carries the handle (ScaLAPACK `ipiv`, cuSOLVERMp raw buffer, SLATE `SlateLowerL`). |
| `resolve_backend(op, requested, mesh, *, n=None) -> str` | The **raising** probe. `n` is decoupled from the operand so a caller that will pad can ask before it has built anything. |
| `list_backends(op, mesh) -> dict` | The **never-raising** report, for startup banners. |
| `BACKEND_CHOICES`, `EIGH_BACKENDS`, `CHOLESKY_BACKENDS`, `LU_BACKENDS`, `OPS`, `NATIVE` | The vocabulary. Importable with **no `.so` anywhere on the machine** — a deck parser must not need the FFI layer to read a deck. |
| `mesh_key(mesh)`, `mesh_platform(mesh)`, `mesh_is_cpu(mesh)` | Stable hashable mesh identity (axes, extents, platform, device ids) and its two predicates. `mesh_key` is for any cache whose stored value does **not** retain the mesh; `id(mesh)` there is the documented drift. |
| `dial_key()` | Factory-time dials folded into one tuple, for kernel cache keys. |
| `probe_target`, `has_target` | Capability, with the ABSENT / BROKEN split (`lxkit.probe`). |
| `dispatch_batched_eigh(A, mesh, backend, *, batched_route='auto')` | The one legacy entry point kept for `gw.qsgw_density`; it passes the same public route selection into `plan`. |
| `matmul(A, B, C=None, *, mesh, alpha=1, beta=0, transa='N', transb='N', backend='auto', batched_route='auto')` | Top-level distributed GEMM. Rank 2 uses `P('x','y')`; rank 3 uses `P(None,'x','y')`. Unlike `plan`, `backend='auto'` selects a distributed provider. |
| `resolve_matmul_backend(requested, mesh, *, batched_route='auto') -> str`, `MATMUL_BACKEND_CHOICES` | Raising GEMM-provider probe and its public vocabulary. `cusolvermp` is an accepted alias for `cublasmp`; `off` is legal only for the provider-free staged route. |
| `gemm_plan(mesh, *, m, k, n, nq, dtype, backend='auto', alpha=1, beta=0) -> GemmPlan` | Resolve, probe, warm and COMPILE one N,N GEMM shape ONCE — the `matmul` analogue of `plan_polar_factor`. `GemmPlan(A, B, C=None, *, out=None)` is trace-safe: safe inside a caller's own `jax.jit`/`lax.scan`. |

Two phases and they stay two: only platform and handler guards can fire at
resolve time — operand dtype, rank and extent are trace-time facts — so a
single-phase API would have to lie about when it checked.

One standalone limitation is intentional: under the native backend,
`Plan.__call__`/automatic `Plan.batched` directly implement `eigh` only.
Native Cholesky and LU channel policy remains caller-owned, so an installed
consumer that wants those operations through the array-returning plan surface
must select `batched_route='batch_reshard'`; Cholesky may alternatively select
`backend='native2d'`, and either op may select an available FFI backend. The
opaque `factor()`/`solve()` token surface is separate and is not changed by
the batch route.

`batched_route` owns only the execution of an array-returning
`Plan.batched` call. It does not choose whether an application uses a Plan,
an opaque `FactorToken`, or a larger multi-channel schedule. LORRAX's coupled
transverse-zeta caller is one deliberate higher-level policy: for an eligible
all-fresh fit, its `auto` request tries certified local `batch_reshard`, then
a distributed split-factor token, then sequential channels as capacity
requires. Explicit `batch_reshard` never switches to the token route; failure
to fit the coupled local live set selects sequential calls that retain the
explicit route. Partial reuse is sequential. Both coupled schedules share Z
construction but keep three ordered q-batch solves; there is no public or
private fused three-channel cuSOLVERMp route.

## Contract

* Polar/SVD is a composite, not a resolver operation: its backend argument is
  passed once to plan('eigh', ..., n=2*n), so there is no second backend
  vocabulary or demotion ladder to drift.
* Rank-deficient polar output is value-level comparable through the partial
  isometry and singular spectrum.  Individual dilation eigenvectors remain
  gauge-dependent and must never be compared across meshes.

* **Promise semantics.** A returned backend name means every guard passed:
  platform, **known-broken combinations**, compiled handler,
  one-process-per-device coverage, mesh geometry, divisibility. The call
  cannot then fail for an availability or geometry reason.
* **Explicit requests REFUSE**, with the failed guard named and a fix.
  Only `auto` demotes, and it announces once, from the rank it happened on.
  A grammar error (an unknown spelling) falls to the gate DEFAULT.
* **A COST NOTICE is not a demote.** An explicit `distributed`/`cusolvermp`
  eigh below n=16384 prints what that route costs (rank 0, once per mesh
  geometry) and then returns it unchanged. The measurement is in
  § Performance; the reason it is a notice and not a demote is the line
  above this one.
* **Exception types are part of the API**: `ValueError` / `RuntimeError`,
  each constructible from one string. `bandstructure/bse_setup.py:386-403`
  re-raises `type(exc)(_why)`; a service-specific exception class would
  escape that handler and delete the `use_low_mem_eigh` refusal message.
* **Layout.** Eigenvalues come back replicated; eigenvectors as **COLUMNS**,
  on every backend. Plan and GEMM batched inputs are `P(None,'x','y')`, single
  tiles `P('x','y')`; GEMM returns the same-rank face layout. `ensure_sharding`
  is the one place that is spelled.
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
* **`factor`/`solve` is the split-token surface, not `Plan.batched`.**
  ScaLAPACK and cuSOLVERMp LU expose getrf/getrs through one opaque token, so
  callers may factor once and apply getrs repeatedly. The array-returning
  `plan('solve_lu', mesh, backend=…).batched(A, B)` instead owns one complete
  factor+solve call per input batch and may select the service's staged local
  route.
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

The table above is the `Plan` resolver. GEMM has its own provider vocabulary
and deliberately different default:

| `matmul` request | CUDA | CPU | ROCm |
|---|---|---|---|
| `auto` or `distributed` | **cuBLASMp** | **PBLAS** (`pdgemm`/`pzgemm`) | **SLATE** (`slate::multiply`) |
| `cusolvermp` | **cuBLASMp** alias | refuse | refuse |
| `cublasmp` | **cuBLASMp** | refuse | refuse |
| `scalapack` | refuse | **PBLAS** | refuse |
| `slate` | **SLATE** | **SLATE** | **SLATE** (declared-untested) |
| `off` | provider-free `batch_reshard` only | provider-free `batch_reshard` only | provider-free `batch_reshard` only |

* For `Plan`, `native` is the floor everywhere and the measured default at
  every production tile size; its `auto` never picks an FFI backend. For
  top-level `matmul`, `auto` means the platform's distributed provider.
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
* **SLATE `eigh` on a MULTI-RANK CUDA mesh is refused at n ≥ 4096** — bug
  L-4, the CUDA sibling of L-2: SIGSEGV, `srun` rc=139, every rank of the
  job down (jobid 56457930). Size-scoped, because it *returns* at
  n ≤ 2048 on the same mesh. A 1×1 CUDA mesh at that size is unmeasured
  and therefore not refused.
* **cuSOLVERMp `eigh` with `compute_evecs=False` is refused** — bug L-3,
  `cusolverMpSyevd` status=7 at every n. A library defect (`bufferSize`
  succeeds; the handler only forwards the flag). Permanent, not a
  stopgap: LORRAX wants `compute_evecs=True` everywhere (owner,
  2026-08-07).
* **ELPA — REGISTERED CANDIDATE, not built, not wired.** Neither shipped
  distributed eigensolver is good in the regime that matters: cuSOLVERMp
  has a ~1.55 s per-matrix floor, and SLATE has a *cheaper* floor
  (0.40 s at n=64) but worse asymptotics (5.44 s at n=2048, already 2.8×
  cuSOLVERMp) and it **crashes at 4096** — i.e. it dies exactly where a
  distributed eigh becomes necessary. Building ELPA as the "dream stack"
  eigensolver is under consideration by the owner. Nothing in this tree
  depends on it; the row exists so the question has a written answer.
* **There is no ScaLAPACK `potrf` handler anywhere in the tree**, and no
  SLATE `getrf`. The vocabularies above are the whole truth.
* **The two platform `.so`s share their SLATE.** Both `liblorrax_ffi.so` and
  `liblorrax_ffi_host.so` carry `NEEDED libslate.so.2` and `NEEDED
  libblaspp.so.2`, resolved out of different builds; `ld.so` keys a loaded
  object by SONAME, so the first one opened decides which
  `blas::get_device_count()` the other calls, and the host build's is a
  compiled-in 0. Both loaders therefore open **CUDA before cpu**
  (`_open_cuda_before_host`). See Antipatterns.

## Distributed matrix multiplication

`matmul` is a top-level operation rather than a `Plan` op:

```python
D = distrib_la.matmul(
    A, B, C, mesh=mesh, alpha=alpha, beta=beta,
    transa='N', transb='N', backend='auto', batched_route='auto')
```

It computes `D = alpha * op(A) @ op(B) + beta * C`, where each operation
code is `N`, `T`, or `C`. `C` is optional only when `beta == 0`; in that case
the service creates a zero addend. Real operands refuse complex `alpha` or
`beta`. All operands must have the same dtype, the contraction dimensions
must agree, and a supplied `C` must have exactly the output shape.

| input rank | public input sharding | batch contract | output |
|---|---|---|---|
| 2 | `P('x','y')` | one matrix, internally lifted to batch 1 | rank 2, `P('x','y')` |
| 3 | `P(None,'x','y')` | A, B, and C have the same nonempty leading batch | rank 3, `P(None,'x','y')` |

With `batched_route='auto'`, the resolved provider receives the original
face-sharded matrices: cuBLASMp on CUDA, PBLAS `pdgemm`/`pzgemm` on CPU, or
`slate::multiply` on ROCm. Explicit `cublasmp`, `scalapack`, and `slate`
requests never demote, and `cusolvermp` names its cuBLASMp sibling. Provider
handlers accept `float64` and `complex128`, require an exact 2-D `('x','y')`
mesh with y-minor process order, one JAX process per cell, and exact face
tiling. They alias/donate `C` to the output. cuBLASMp and SLATE also require a
square mesh; PBLAS supports rectangular grids. `backend='off'` has no
provider and is therefore legal only with the staged route.

With `batched_route='batch_reshard'`, the service pads a leading batch `B`
to `Bp = ceil(B/(Px*Py)) * Px*Py`, using zero A, B, and C matrices. Zero is a
safe synthetic GEMM row and all padded results are discarded. Each operand
then follows these collectives inside one `shard_map`:

| stage for a generic `X: (Bp,R,C)` | per-device shape | collective |
|---|---|---|
| incoming face | `(Bp, R/Px, C/Py)` | — |
| x forward | `(Bp/Px, R, C/Py)` | x `all_to_all(split_axis=0, concat_axis=1)` |
| y forward | `(Bp/(Px*Py), R, C)` | y `all_to_all(split_axis=0, concat_axis=2)` |

Local devices apply `op(A)` and `op(B)`, run `jnp.matmul`, and add `beta*C`.
For `D: (Bp,M,N)`, the literal inverse is deliberately in reverse order:

| stage for D | per-device shape | collective |
|---|---|---|
| local result | `(Bp/(Px*Py), M, N)` | — |
| y inverse | `(Bp/Px, M, N/Py)` | y `all_to_all(split_axis=2, concat_axis=0)` |
| x inverse | `(Bp, M/Px, N/Py)` | x `all_to_all(split_axis=1, concat_axis=0)` |
| returned face | original leading batch only | drop the padded rows |

This is staged device-to-device movement, never a host gather and never a
direct sharding constraint that could rematerialize a full stack. It accepts
a ragged leading batch, but deliberately does **not** pad matrix dimensions:
physical rows of A/B/C must divide `Px`, physical columns must divide `Py`,
and output M/N must do the same. These checks run before placement or a
collective. Rank, batch, dtype, contraction, operation-code, scalar, and
missing-C errors also refuse eagerly.

The staged route is a capacity tradeoff. Each device holds
`ceil(B/(Px*Py))` complete A, B, and D matrices (plus C when `beta != 0`),
the still-live caller faces/padded faces, collective exchange buffers, and
JAX GEMM workspace. With `beta == 0`, no synthetic C is allocated or
exchanged.
Rank-2 input is padded to `Px*Py`, so each device still holds one complete
operand set even though only one row is real. Use it only when those complete
matrices fit comfortably; the provider route remains the default for matrices
that require 2-D distribution for capacity.

Provider-specific refusals still matter when a provider is selected. A
non-`off` provider request is resolved and probed even with
`batched_route='batch_reshard'`; use `backend='off'` for a provider-free call.
An unavailable handler, wrong platform, partial-world mesh, unsupported
provider dtype, or incompatible face extent refuses rather than falling back.
Multi-rank cuBLASMp accepts only `transa='N', transb='N'`. The real P=4 gate
found that transpose-A returns a wrong result, while transpose-B can return
rank-divergent `INVALID_VALUE` and deadlock. Both are refused before the
provider call; pretranspose into the ordinary face layout, select PBLAS/SLATE,
or use the staged route.

## Planned GEMM — a trace-safe cuBLASMp N,N call for hot loops

`matmul()` resolves its provider and probes capability at every call. That
is correct for an eager call site, but a caller that runs G construction
or a per-tau Sigma projection inside its own `jax.jit`/`lax.scan` needs the
same two-phase split `Plan`/`PolarPlan` already give the solver ops: an
EAGER phase (dlopen, probe, mesh geometry, cuBLASMp communicator) that runs
ONCE, and a closure built from its result that touches none of that.
`gemm_plan`/`GemmPlan` (`distrib_la.matmul_plan`) is that split for GEMM,
modelled directly on `plan_polar_factor`/`PolarPlan` — the one existing
precedent for driving an FFI call from inside a composed, jitted kernel.

```python
plan = distrib_la.gemm_plan(
    mesh, m=m, k=k, n=n, nq=nq, dtype=dtype, backend='auto')
# ... hoisted out of the k/tau loop; by here the cuBLASMp communicator
# exists and both kernel variants have already run once on dummy data ...
D = plan(A, B)                 # inside jit/scan: no dlopen, no probe
```

Deliberately narrower than `matmul()`:

* **N,N only** — there is no `transa`/`transb` anywhere in the module.
  Multi-rank cuBLASMp's transpose modes are the ones `matmul()` itself
  refuses (§ "Distributed matrix multiplication" above); a caller with a
  transposed operand pretransposes into the complementary face layout
  once, which is exactly what a two-face `psi_nmu`/`psi_mun` bundle does.
* **One replicated leading batch, fixed at construction** — `A` is
  `(nq,m,k)`, `B` is `(nq,k,n)`, `C`/`D` are `(nq,m,n)`, all
  `P(None,'x','y')`. `nq` holds k-points; a spinor axis is not a second
  batch — flatten it into m/k/n, or call the SAME plan `ns` times in a
  small, statically unrolled Python loop. `nq=1` is a legal,
  zero-overhead rank-2-equivalent plan.
* **cuBLASMp only, today** — `lorrax_scalapack_batched_gemm` and
  `lorrax_slate_batched_gemm` are claimed by `distrib_la.loader`'s target
  table but have no C++ definition anywhere in this tree
  (`KNOWN_LORRAX_ISSUES.md`, "services/distrib_la loader vs src/ffi" row;
  confirmed again by `nm -D` on the pinned CUDA library, which exports
  only `CublasMpBatchedGemmFfi`). A request that `resolve_matmul_backend`
  would send to either provider refuses at `gemm_plan()` construction, by
  name, using the same capability probe `matmul()` uses.
* **Provider route only** — `backend='off'` refuses by name.
  `batch_reshard` materializes complete A, B, C and D on every device; the
  reason to reach for a *planned* GEMM at all is a G/Sigma-sized operand
  that must never be that, so this surface never selects it.

**Output liveness.** A plan built with `beta=0` (the default, and what
every G/T/Sigma GEMM in the `low_mem_bands` audit needs) additionally
compiles a kernel that builds its zero addend with `jnp.zeros` INSIDE the
same compiled program as the GEMM FFI call, so a repeated call never pays
`matmul()`'s separate top-level `jax.jit` dispatch for a missing `C`
(`matmul.py:433-437`) — one compiled program, not two. Passing an existing
buffer as `out=` skips the internal zero-fill entirely and donates that
buffer's storage to the provider instead, for a caller threading a scratch
accumulator through a `lax.scan` carry. Neither path removes cuBLASMp's own
requirement of a live `C` argument: the FFI handler binds it unconditionally
(`src/ffi/cpp/cublasmp/batched_gemm_ffi.cc`, `.Arg<AnyBuffer>() // C`), so
there is no provider-level "no C at all" mode — what this surface removes
is the extra Python-level allocation and compiled program, not the C++
argument. State that distinction when reporting the memory win.

`out=` is refused on a `beta!=0` plan: `C` and `out` both reach the same
compiled kernel, and that kernel's `beta` is fixed at *plan construction*,
not chosen per call, so `out=`'s "content is ignored" contract only holds
when the plan itself was built with `beta=0` — on a `beta!=0` plan the
buffer's stale content would silently be scaled by `beta` and folded into
the result. Pass `C=` on such a plan instead, where the accumulate is
explicit at the call site.

**Verified**, `services/distrib_la/tests/test_distrib_la_matmul_plan.py`
(emulated CPU mesh — the eager refusal ladder only: `backend='off'`, a
resolved non-cuBLASMp provider, mesh topology, dtype, malformed shapes;
real execution cannot be reached without a CUDA mesh) and
`test_distrib_la_multiproc.py`'s `gemm_plan_cublasmp` CLI cell, on
Perlmutter, `lx run -G 4 -n 4 ... --mesh 2x2 --only gemm_plan`: numerics
against `A @ B` (complex128 and float64, relative ~1e-16), called eagerly,
inside a `jax.jit`, inside a `lax.scan` (the actual per-tau/per-k hot-loop
shape), through the `beta!=0` accumulate path, through the donated `out=`
path, and across five repeated calls with fresh operands. `matmul_cublasmp`
in the same suite (the pre-existing `matmul()` path, unchanged by this
work) passed on the same real 2x2 mesh in the same run, confirming no
regression.

## Tests

test_distrib_la_polar.py is the synthetic complex/real polar tier.  It covers
NumPy-SVD parity, unitary and repeated-singular-value degeneracies,
ill-conditioning below the Gram-eigh resolution floor, numerical rank
deficiency, an all-zero matrix, non-divisible logical padding, output
shardings, planned tracing, and refusal/red-control cases.  It runs on the
four-device emulated CPU mesh and does not require a vendor library.

The suite spans local algebra, emulated devices, real processes, FFI/ELF
acceptance, import isolation, and skip honesty. Markers `services` +
`distrib_la` are applied by a collection hook
(a `pytestmark` in a conftest is silent — `tests/test_service_selection.py`
measures that the marks arrived).

| tier | file | needs |
|---|---|---|
| L-a shape/contract algebra | `test_distrib_la_shape_algebra.py` | nothing — a laptop, milliseconds |
| L-b emulated multi-device | `test_distrib_la_emulated_mesh.py` | `XLA_FLAGS` set by the SERVICE conftest; **skips**, never asserts, below 4 devices |
| route-c staged movement | `test_distrib_la_batch_reshard.py` | four emulated CPU devices; all three local kernels, ragged batches, inverse round trip + wrong-order red twin |
| GEMM provider + staged contract | `test_distrib_la_matmul.py` | four emulated CPU devices; backend vocabulary, rank-2 and ragged rank-3 GEMM, transpose codes, and exact x/y + y/x schedule |
| planned GEMM eager refusal ladder | `test_distrib_la_matmul_plan.py` | nothing but jax; `gemm_plan()`'s pure helpers plus its `backend='off'`/non-cuBLASMp/topology/dtype/shape refusals on an emulated CPU mesh — real cuBLASMp execution is CUDA-only and is leg L-c's `gemm_plan_cublasmp` cell |
| L-c real multi-process | `test_distrib_la_multiproc.py` | `srun -n 4`; shared `check_*(mesh, …)` bodies + a `__main__` CLI (`_CLI_CELLS`) — same functions, no duplicated logic |
| contract + wiring | `test_distrib_la_contract.py` | the `.so` pins; every refusal constructibly fires |
| C++ / ELF acceptance | `test_so_acceptance.py` | binutils + a pinned `.so`; reads the ELF, never dlopens |
| import isolation | `test_distrib_la_import_isolation.py` | `python -S` subprocess; `sys.modules` AND `sys.path` asserted, plus a red twin and a with-lorrax-still-passes |
| skip honesty | `test_distrib_la_skip_honesty.py` | a machine profile; ABSENT = skip, BUILT-AND-BROKEN = **FAIL** |

* **Hostile geometry is mandatory**: a real 2×2 with non-dividing extents
  and padding round-trips, with the anti-tautology self-assertion (the pad
  divisor must be provably non-vacuous).
* **Every check ships with the case where it returns FALSE.** No exceptions.
* From an installed editable package in `services/distrib_la`, run
  `python -m pytest`. From the repository root, run
  `python -m pytest services/distrib_la/tests`; the service conftest loads
  first and creates four emulated CPU devices before JAX imports. Neither
  spelling loads the monorepo `tests/conftest.py`. Before the GEMM surface
  landed, the source-only/emulated floor at commit `0ba29095` was **82
  passed**, including **13** focused route-(c) cells. The current total also
  includes `test_distrib_la_matmul.py`; use collection/run output rather than
  treating the historical count as a present invariant.
* Run it as part of LORRAX with `python -m pytest -m distrib_la`.
  Deselect: `--no-services` / `--only-service=NAME`, never a second `-m`
  (`pyproject` sets `addopts = "-m 'not extra'"` and an explicit `-m`
  REPLACES it, silently re-enabling 26 deselected suites).
* The scheduler-specific real-GPU gate is deliberately separate from the
  installable package's local suite. On Perlmutter it is:

  ```bash
  export LX_BASE_MODULE=lorrax_J070
  lx run -N 1 -G 4 -n 4 python3 -u \
    services/distrib_la/tests/test_distrib_la_multiproc.py \
    --mesh 2x2 --only batch_reshard_local_ops
  ```

  On 2026-08-15, commit `0ba29095`, that solver gate ran a ragged batch of five on
  four CUDA ranks and passed `complex128` and `float64` for `eigh`,
  `cholesky`, and `solve_lu`: **2 cells / 0 failures**. The gate asserts the
  output shardings before its test-only host readback. It predates `matmul`
  and is not evidence for either GEMM provider execution or staged GEMM on
  real CUDA processes.
* `python -m pytest` runs the FFI/ELF and machine-profile gates as well. On
  Perlmutter, the profile promises working provider libraries, so the full
  suite requires the documented `LORRAX_FFI_{,HOST_}SO` pins and dependent
  library paths; missing promised capabilities fail skip honesty instead of
  being counted as acceptable standalone skips.
* Perlmutter floor, 2026-08-07, HEAD `eeece71`, BUILD_NOTES pins: full-suite
  `-m distrib_la` **130 cells / 0 failed / 22 skipped** (was 124 / 8 failed
  before the SONAME fix); service-only by path **250 / 0 / 3**, of which
  lxkit is 120 and unchanged.

## Performance

Recorded baselines (never slow tests) in
`services/distrib_la/bench/baselines/{cpu,gpu}{1x1,2x2}.json`: one row per
(op × backend × shape), with `seconds`/`min`/`max`, `compile_seconds`, the
resolved backend, and — for a combination that cannot run — the refusal
text instead of a number. All rows below: Perlmutter, **jobid 56447670**,
1 node, jax `0.7.0.dev20260807`, complex128, BUILD_NOTES `.so` pins.
Regression detection = diffing baseline files across branches.

> The jobid above read **56444350** until 2026-08-07. That was stale — it
> is the step-2 L-c multiproc leg, not the leg the committed baselines
> came from. Every row in all four baseline files carries `"jobid":
> "56447670"`, which is the number now printed here.

> **The eight `cholesky`/`slate` rows in the baseline files are now
> PESSIMISTIC and have not been regenerated.** They were measured against
> the per-q Python loops in `factor`/`solve`, which re-compiled SLATE's
> kernel once per matrix; those loops are scans as of
> `feat/batched-canonical-2026-08-08` and § "The batched surface is a
> scan" measures the same route 35× faster on gpu 2×2. Regenerating the
> sweep is a perf leg on a shared GPU pool and is deliberately NOT folded
> into that change — read the A/B table for this route, not
> `baselines/*.json`, until a regeneration leg lands.

**For the three `Plan` ops, `auto` resolves to `native` everywhere, and the
numbers say why.** (`matmul` uses the separate provider default above.)

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
with n⁴× the work is a fixed collective/context charge.

### The crossover, and where the ~1.55 s actually goes

Step 6's eigh investigation (jobid **56447670**, 4 processes on a real
2×2 A100 mesh, complex128, warm medians, one matrix per call). Rows are in
`baselines/{cpu,gpu}2x2.json` tagged `"leg": "step6.cross_size"`.

| n | native replicated | cuSOLVERMp | SLATE | cuSOLVERMp / native |
|---|---|---|---|---|
| 64 | **0.00149** | 1.586 | 0.401 | 1064× |
| 256 | **0.00412** | 1.561 | 0.546 | 378× |
| 1024 | **0.02662** | 1.754 | 1.387 | 66× |
| 2048 | **0.07275** | 1.932 | 5.444 | 27× |
| 4096 | **0.39550** | 2.739 | **SIGSEGV** | 6.9× |

**Is SLATE the reasonable one?** No. It has the cheaper floor — 0.40 s vs
1.59 s at n=64, and it beats cuSOLVERMp up to n≈1024 — but its scaling is
worse (5.44 s at 2048, already 2.8× cuSOLVERMp) and it **dies at 4096**,
which is where distributed eigh starts to be the thing you actually need.
Faster floor, worse asymptotics, crashes in the capacity regime. That is
the whole reason ELPA is a registered candidate (§ Backends).

**The ~1.55 s is not ours and is not reachable.** `LORRAX_FFI_PROFILE=1`
splits the warm n=64 call as: `cusolverMpSyevd` **99.998%**, `plan()` +
`resolve_backend` 9.6 µs, cuSOLVERMp context-cache HIT 0.72 µs,
descriptors + `bufferSize` ≤ 0.02 ms. Cold-only extras, outside the warm
number: one-time context bootstrap 0.66–0.81 s, XLA compile ~0.79 s.
There is no in-tree edit that moves this. `block_size` does not either
(64/128/256/default at n=4096: 2.44/2.41/2.44/2.70 s).

**CPU tells the same story without needing an extrapolation.** ScaLAPACK
`pzheevd` never beat native replicated on one node, and the gap widens:

| n | native replicated | scalapack | ratio |
|---|---|---|---|
| 64 | **0.00063** | 0.00255 | 4.0× |
| 256 | **0.01626** | 0.01717 | 1.06× |
| 1024 | **0.27120** | 0.55437 | 2.04× |
| 2048 | **1.48904** | 4.12157 | 2.77× |

`Plan`'s `auto` → `native` (`resolve.py`) is therefore **vindicated on both
platforms**, at every size anyone has measured. Do not change it.

### `distributed` eigh is for CAPACITY, not speed — and that regime is UNMEASURED

Every row above **fits on one device**, so every row above is a regime
where a distributed library has no reason to win. Extrapolating two fitted
power-law exponents puts break-even near **n ≈ 1.9 × 10⁴**, which is
**4.6× past the largest n measured (4096)** and is sensitive to the fit
window (last-three-points ≈ 1.4 × 10⁴; all-five ≈ 2.8 × 10⁴). Read the
**decade**, never the digits. It lands in the same decade as the
single-device capacity wall: an n×n complex128 matrix plus its
eigenvector copy and workspace stops fitting in 40 GB around n ≈ 2.7 × 10⁴.

So the honest summary is: *below the capacity wall, native wins by
between 1064× and 6.9×; at and above it, native cannot run at all and
nobody has measured what the alternatives cost.* An explicit
`distributed`/`cusolvermp` eigh below n=16384 now prints that on rank 0,
once per mesh geometry, and **still runs exactly what was asked for** —
explicit requests are never demoted.

### Open measurement gaps (inherit this map; do not re-derive it)

1. **MULTI-NODE — the biggest gap by a distance.** Every number in this
   document is one node, 4 ranks. The case `distributed` exists for is a
   mesh spanning nodes, and there is not one measurement of it.
2. **The capacity regime itself** (n ≳ 2.7 × 10⁴, matrices too large for
   one device). Unmeasured, so the crossover stays an extrapolation.
3. **float64 rows.** Everything here is complex128. The real-symmetric
   path is exercised by contract cells but has no baseline row.
4. **`compute_evecs=False`.** Refused on cuSOLVERMp (bug L-3), so its
   cost is unknown on every backend — including the ones where it works.
5. **A genuine CPU partition.** The "cpu" legs ran on the GPU pool's nodes
   with `JAX_PLATFORMS=cpu`; the Milan CPU partition has one census leg
   (jobid 56446562) and no perf leg.
6. **Non-square 1-D meshes.** ScaLAPACK accepts them (square *blocks*, not
   a square grid) and nothing has been timed on one.
7. **Batched vs serial, split by backend.** ~~Never isolated.~~ Partly
   closed by the A/B in § "The batched surface is a scan": on cpu 2×2 at
   nq=8/n=64 the scan route costs 0.0115 s against the ScaLAPACK stacked
   entry's 0.01125 s — 2% for a route with no C++ batching at all — so the
   stacked entry's advantage at THIS shape is small. What stays open is
   whether that holds at large `nq`, where one descriptor and one
   workspace should start to tell.
8. **The (2048, 4096] window for bug L-4.** SLATE CUDA eigh's true crash
   threshold is somewhere in there; 4096 is just the smallest size anyone
   watched it die at.
9. **The 4×4 control at n = 8192.** Owed to the 3×3 hang written up at the
   top of this document. Until someone runs it, we cannot say whether the
   hang is a non-power-of-two-grid problem or a "anything past 2×2"
   problem, and those two have very different blast radii. Root-causing
   the hang itself needs a multi-node GPU allocation.

### The batched surface is a scan, and the route toggle is one place

**BUILT.** `Plan.batched` is a `lax.scan` over this package's own
single-matrix operation, at `BATCHED_SCAN_UNROLL = 1` — a named constant,
deliberately changeable, changed only with a measurement behind it.
Backend-native stacked entry points (ScaLAPACK's eigh `many`, cuSOLVERMp's
`batched_potrf`, both `batched_solve_lu`) are backend-internal
optimizations **behind that same interface**, never a second public
surface. `dispatch_batched_eigh` used to be that second surface — it
carried its own `getattr` capability probe and its own serial loop — and
is now `plan(...).batched(A)`.

The point is not the loop. A Python loop over `nb` matrices is `nb`
separate calls the compiler never sees together, and there is nowhere in
it to put "run this batch some other way". A scan is one node, so the
choice collapses to **`Plan.batched_route`, the one place that decides**:

* **(a)** `ROUTE_SCAN` — a scan of the *distributed* single-matrix op.
  The default, and the definition of the surface.
* **(b)** `ROUTE_BACKEND_BATCHED` — the backend's stacked entry where the
  library has one. Its saving is in C++, around ONE descriptor and ONE
  workspace, and no scan can recover it.
* **(c)** `ROUTE_BATCH_RESHARD` — reshard
  `(q, μ_x, ν_y) → (q_xy, μ, ν)`, run the op locally with the native
  JAX kernel per matrix, and reshard matrix outputs back through the literal
  inverse exchange. This serves batches of matrices below single-device
  capacity **without paying any distributed-library fixed cost** — which the
  tables above show is the entire cost below n ≈ 10⁴.

(a)/(b)/(c) behind one toggle point is how small-system, non-distributed
linalg happens **without a parallel API**.

**Route (c) is built without an upward package dependency.** A direct
`P(None,'x','y') → P(('x','y'),None,None)` constraint is not a tile
permutation: GSPMD has lowered it as replicate-then-partition, with a measured
64× per-rank residency blow-up (job 7882974). The package therefore owns a
small private implementation in `_batch_reshard`; it imports no LORRAX
`common` module. Forward movement and its literal inverse run inside one
`shard_map`, using only device collectives. For padded global batch `Bp`, the
local shapes and exact `all_to_all` arguments are:

| step | operation | local shape after the step |
|---|---|---|
| input face | `P(None,'x','y')` | `(Bp, N/Px, N/Py)` |
| forward x | `all_to_all('x', split_axis=0, concat_axis=1, tiled=True)` | `(Bp/Px, N, N/Py)` |
| forward y | `all_to_all('y', split_axis=0, concat_axis=2, tiled=True)` | `(Bp/(Px·Py), N, N)` |
| inverse y | `all_to_all('y', split_axis=2, concat_axis=0, tiled=True)` | `(Bp/Px, N, N/Py)` |
| inverse x | `all_to_all('x', split_axis=1, concat_axis=0, tiled=True)` | `(Bp, N/Px, N/Py)` |

Thus x splits batch and joins matrix rows, then y splits batch and joins
matrix columns; the inverse must run y then x. A wrong-order inverse is
shape-correct on a square mesh, so the test suite contains both a bit-exact
movement round trip and a red twin proving that x-then-y scrambles the data.
No operand or output gathers through the host.

The local kernels are `jnp.linalg.eigh`, `jnp.linalg.cholesky`, and
`jnp.linalg.solve`, covering all three array-returning `Plan.batched` ops.
Outputs preserve the ordinary service contract:

* `eigh`: eigenvalues `(B,N)` are restored with a device
  `all_gather(('x','y'), axis=0, tiled=True)` and returned replicated at
  `P()`; eigenvectors take the inverse exchanges and return at
  `P(None,'x','y')`.
* `cholesky`: factors take the inverse exchanges and return at
  `P(None,'x','y')`.
* `solve_lu`: solutions take the inverse exchanges and return at
  `P(None,'x','y')`.

Ragged batch handling is internal. `Bp = ceil(B/(Px·Py))·Px·Py`, and
synthetic rows are dropped after the inverse. Eigh gets zero-Hermitian rows;
Cholesky and LU replace each synthetic A with identity; LU's synthetic RHS
is zero. Those are safe inputs to the local kernels and prevent padded
Cholesky/LU rows from producing singular-factor NaNs. This padding covers
only the leading batch. Matrix faces must tile exactly
(`N % Px == N % Py == 0`), and LU RHS columns must obey
`NRHS % Py == 0`; rank/shape/dtype/extent violations refuse eagerly before
placement or collective entry. A consumer may pad a matrix/RHS extent before
the call and slice afterward, but the package cannot infer that transformation
without changing the mathematical problem.

Backend-only `block_size` is consumed at the route boundary and never reaches
`jnp.linalg.*`. `compute_evecs` is likewise consumed for `eigh`, because the
public result always contains eigenvectors; any other unsupported keyword
refuses with `TypeError` instead of leaking into a local kernel.

The public opt-in is construction-time:
`plan(..., batched_route='batch_reshard')`; `Plan.batched(...)` remains the
single call surface. Route selection and backend selection are orthogonal:
the requested backend is still resolved and, if explicit, probed before route
(c) runs its native kernel. Use `backend='off'` when no provider capability is
intended. `auto` preserves the historical choice exactly: native `eigh` and
FFI backends with a stacked entry resolve to `backend_batched`; remaining FFI
backends resolve to `scan`. `Plan.describe()` records both the requested and
resolved backend and the requested and resolved batch route.

Plan construction also calls the package-local `warm_mesh_cliques(mesh)`.
It is a cached no-op for GPU/NCCL, non-MPI CPU transports, single-process
runs, and already-warmed meshes. On multi-process JAX CPU with
`JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`, it synchronously compiles tiny
`psum`s for the x, y, and flattened `(x,y)` cliques on the main thread before
the route's `all_to_all`s can be compiled from an intra-op worker.

**Capacity is the route's hard boundary.** After the two forward exchanges,
each device holds `Bp/(Px·Py)` complete `N×N` matrices and runs a native
dense solver on them. At least one full matrix, its matrix-shaped output, and
the solver workspace must fit on one device; multiple local batch elements
increase that peak further. The exchanges are volume-preserving, but they do
not make a single matrix smaller. When one full matrix cannot fit, retain the
default/distributed tile route rather than selecting `batch_reshard`.

#### What the restructure cost and bought

A/B on the same nodes with the BUILD_NOTES pins, `origin/main` (`21d68e0`)
against `feat/batched-canonical-2026-08-08`, real 4-process 2×2 meshes,
nq=8, n=64, complex128. `compiles` is XLA compilations on the cold call,
counted off jax's own compile log; `warm` is the median of three
subsequent calls.

| leg | case | route | compiles | cold s | warm s |
|---|---|---|---|---|---|
| cpu 2×2 | eigh, `main` | Python loop | 12 | 0.307 | 0.0237 |
| cpu 2×2 | eigh, branch | **(a) scan** | **1** | 0.109 | **0.0115** |
| cpu 2×2 | eigh, `main` | (b) stacked | 1 | 0.063 | 0.01130 |
| cpu 2×2 | eigh, branch | (b) stacked | 1 | 0.063 | 0.01125 |
| cpu 2×2 | SLATE cholesky factor+solve, `main` | 2 Python loops | 165 | 3.370 | 3.2578 |
| cpu 2×2 | SLATE cholesky factor+solve, branch | **2 scans** | **3** | 0.177 | **0.0087** |
| gpu 2×2 | eigh/cuSOLVERMp, `main` | Python loop | 11 | 14.250 | 12.518 |
| gpu 2×2 | eigh/cuSOLVERMp, branch | **(a) scan** | **1** | 14.177 | 12.702 |
| gpu 2×2 | SLATE cholesky factor+solve, `main` | 2 Python loops | 165 | 6.234 | 5.887 |
| gpu 2×2 | SLATE cholesky factor+solve, branch | **2 scans** | **3** | 0.841 | **0.166** |

**Every result is BIT-IDENTICAL across the two trees** — W and Z for both
eigh legs and X for the cholesky legs, `np.array_equal` true, on cpu and
on gpu. This is an interior restructure and the arrays say so.

The eigh/cuSOLVERMp row is the only one where the branch is not faster on
a single run, and repeating it says it is not slower either. Warm seconds
over three alternating runs on the same allocation:

| run | `main` | branch |
|---|---|---|
| 1 | 12.518 | 12.702 |
| 2 | 12.613 | 12.459 |
| 3 | 12.537 | 12.503 |
| mean | **12.556** | **12.555** |

The ordering lands on the wrong side of the difference as often as the
right one. That is what a **flat per-matrix collective charge** looks
like: § "The crossover" measures cuSOLVERMp at 1.56/1.60/1.76/1.59 s per
matrix across four shapes with **99.998% of it inside
`cusolverMpSyevd`**. Nothing in that leg is ours to move, and the scan
did not move it.

**The compile count is the result to read.** The old serial route compiled
per iteration on any wrapper without a jit cache; the scan compiles once,
whatever `nq` is. SLATE is where that mattered most —
`_slate.distributed_cholesky` and `_slate.distributed_trsm` build their
`shard_map` at eager top level with no `jax.jit` around it and no
per-signature cache, the shape every other wrapper here has and those two
do not, so an `nbatch`-long Python loop re-traced and re-compiled the
kernel `nbatch` times. 165 compiles → 3, and 5.9 s → 0.17 s of GPU wall.

**An eager scan is not enough, and the first measurement said so.**
Without a `jax.jit` cache per signature the scan is correct and *slower*:
the cpu eigh row came out at 0.080 s warm against 0.024 s for the Python
loop, because `lax.scan` called eagerly re-traces and re-lowers the whole
loop every call while the Python loop's `nq` backend calls each landed in
the wrapper's own `_JIT_CACHE`. The same scan as a pre-compiled executable
timed 0.0114 s, so ~69 ms of that 80 was pure retrace. `plan._SCAN_CACHE`
is the fix and it is keyed on `mesh_key`, which is strictly finer than the
`(Px, Py)` the backends key their MPI/NCCL contexts on — a cache that
cannot hand back an executable whose baked-in context handle has moved on.

**`lax.scan` over a distributed FFI call is viable on BOTH platforms, and
CUDA was the open question.** The host answer was already recorded (job
7889132, ScaLAPACK, P=4). The route that actually *ships* is cuSOLVERMp
and SLATE over NCCL + `cal_comm`, whose failure mode is a HANG with no
traceback, and no CUDA mesh had ever been pointed at it. It traces,
compiles and runs: `eigh/cusolvermp` above is that route, on a real
4-process 2×2 A100 mesh, residual 2.296e-15.

*Experiment note, not a plan:* a possible **fourth** route, for the
capacity regime only (matrices too large for one device **and**
latency-bound) — overlap `k` independent distributed ops from a C++-side
context pool. Feasible on CUDA (a second cuSOLVERMp context costs
**0.065 s**, measured) but NCCL needs one communicator per concurrent
slot; off-limits on the host leg (Cray MPICH's default
`MPI_THREAD_SINGLE`, with documented deadlock history at `H5Fclose` —
`src/ffi/phdf5/ARCHITECTURE.md`); contention-bound at compute-heavy sizes
anyway. ELPA targets the same slice properly.

**Any such restructuring must keep the batched-vs-serial bit-identity
gate green** (`tests/multi_device/batched_eigh_dispatch_gate.py`, and its
adopted twin `check_batched_eigh_dispatch` in the L-c suite: the two
routes agree to **0 ulp** in both W and Z — job 7889132 at nq=6/n=32, and
again at nq=8/n=64 with the scan on both sides of the comparison). That
gate is the guard on this whole design. Route (c) additionally has direct
native-reference, inverse-round-trip, and wrong-inverse red-twin coverage;
the real P=4 shared check is `check_batch_reshard_local_ops`. The older gate
continues to prove scan and backend-stacked execution did not drift. Its
`_force_serial` argument is `Plan.batched`'s private `_route` override —
the toggle is what makes the gate able to run two routes over one set of
operands at all.

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

The cuSOLVERMp LU token's rank-private pivot row is sized to the vendor
contract ``LOCr(M_A) + MB_A``.  It is still O(n/Px) per matrix/rank; the
extra block is required storage for distributed row interchanges, not a
replicated global pivot.

## Antipatterns

* Calling jnp.linalg.svd at a consumer, or using eigh(A.H @ A), bypasses the
  service and is numerically weaker.  Use polar_factor, or hoist one
  plan_polar_factor for a streamed loop.
* Passing a replicated/host overlap matrix and relying on an implicit reshard
  is refused.  Build the overlap directly at P('x','y'); pad before placement
  when the logical band count does not tile the mesh.

* **Editing `src/ffi/<name>/`.** `src/ffi/linalg/`, `src/ffi/slate/`,
  `src/ffi/scalapack/` and `src/common/cholesky_2d.py` are **deleted**
  (17 files, commit `b3f3675`). A backend lives in
  `services/distrib_la/src/distrib_la/_<name>.py` plus one row in each of
  `loader._*_TARGET_SYMBOLS`, `resolve._SPEC`/`BACKEND_CHOICES` and
  `plan._IMPL`. `docs/dev/linalg_ffi.md` § "Adding a backend" is the
  procedure; the C++ under `src/ffi/cpp/` did not move and its target
  strings are frozen.
* **Writing a Python loop over the batch axis.** That is what
  `Plan.batched` is for, and looping outside it hands the compiler `nb`
  separate calls, re-traces any wrapper that builds its `shard_map`
  eagerly, and — the real cost — puts the batch somewhere no route toggle
  can reach. Measured on the two SLATE wrappers that have no jit cache:
  165 compiles and 5.9 s of GPU wall for an 8-matrix factor+solve, against
  3 compiles and 0.17 s through the scan.
* **Adding a second batched entry point for a backend that has one.** A
  stacked FFI entry is one `plan._IMPL` row and it is then taken
  automatically; exposing it as its own public function recreates the
  `dispatch_batched_eigh` split, where two places decided the same thing
  and one of them forgot the eigenvector normaliser.
* **Calling `lax.scan` over an FFI wrapper eagerly, uncached.** It is
  correct and it is slower than the loop: ~69 ms per call of retrace and
  relower at nq=8/n=64 on cpu 2×2. Route it through `Plan.batched`, which
  owns `plan._SCAN_CACHE`.
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
* **Confusing an installed package with LORRAX's source-checkout
  bootstrap.** A normal consumer installs `lxkit` and `distrib_la` and then
  imports the latter directly; it must not import `ffi._services` or edit
  `sys.path`. Only an uninstalled LORRAX checkout needs the transitional
  `ffi._services.ensure_on_path()` call, because `lx` sets container
  `PYTHONPATH` to `<checkout>/src` and does not install `services/*/src`.
  `tests/test_service_path_bootstrap.py` covers that LORRAX integration
  seam, while the package's own import-isolation suite proves the installed
  service has no upward dependency.
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
