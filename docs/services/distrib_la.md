# distrib_la — distributed dense linear algebra over a JAX device mesh

`services/distrib_la/` is the one door for `eigh`, `cholesky`, `solve_lu`,
`matmul` and the polar factor on an `('x','y')` device mesh. A caller says
what to compute and where; which library runs is a resolved fact it can read
but never branches on. The package is independently installable
(src-layout, Python ≥ 3.12); its runtime dependencies are `lxkit`, JAX and
NumPy, and it imports nothing from LORRAX's `src/`. ScaLAPACK/PBLAS, SLATE and
cuSOLVERMp/cuBLASMp are reached through one edge, `distrib_la.loader`, which
`dlopen`s the LORRAX FFI `.so` by path; none is a declared Python dependency.

Import top-level names only. `from distrib_la.<submodule> import …` from
outside the package fails `tests/test_layering.py`.

## Installation and capability

```bash
cd services/distrib_la
python -m pip install -e ../lxkit -e '.[test]'
python -m pytest            # four emulated CPU devices, no vendor library needed
```

Native JAX routes, capability reporting and the backend vocabulary work with
no shared library present. The FFI provider is a separate capability: the CUDA
and host libraries are found through a sealed bundle or pinned by
`LORRAX_FFI_SO` / `LORRAX_FFI_HOST_SO`, whose rules
[`docs/dev/env_vars.md`](../dev/env_vars.md) owns. The loader expects handler
ABI 5 (`loader.LORRAX_FFI_ABI_VERSION`). An absent `.so` does not break
import; an explicit pin that is missing, mis-stamped or ABI-incompatible
refuses instead of falling through. The environment grants capability and
never selects a backend: `distrib_la.resolve` reads no environment.

Both platform libraries link `libslate.so.2` and `libblaspp.so.2` by SONAME,
and the first library opened decides which copy the other binds. The loader
therefore opens the CUDA library before the host one in any process that can
use CUDA (`loader._open_cuda_before_host`); opening the host library first
gives every CUDA SLATE handler `blas::get_device_count() == 0`.

## API

| name | contract |
|---|---|
| `plan(op, mesh, *, backend='auto', n=None, batched_route='batch_reshard', budget_bytes=None) -> Plan` | Resolve once. **Eager**: dlopens, probes, reads `jax.process_count()`. `op` is `eigh`, `cholesky` or `solve_lu`. Passing `n` runs the divisibility guard at resolve time. |
| `Plan(A)` / `Plan.batched(A_stack, ...)` | One tile at `P('x','y')` / a stack at `P(None,'x','y')`. **Trace-safe**. Operands are moved to the contract layout by `ensure_sharding`. |
| `Plan.backend`, `.is_native`, `.batched_route`, `.route_for(shape, dtype)`, `.describe()`, `.donates` | The resolved facts. Introspection only; a caller never branches on them to be correct. |
| `Plan.native_fn` | Pure trace-safe closure for a caller that needs the math inside its own `jit`. Native backends only. |
| `factor(op, A, mesh, *, backend, ...) -> FactorToken`, `solve(token, B)` | Factor once, back-solve many. The token carries the backend handle (ScaLAPACK `ipiv`, cuSOLVERMp raw buffer, SLATE `SlateLowerL`). |
| `resolve_backend(op, requested, mesh, *, n=None, compute_evecs=True) -> str` | The **raising** probe. `n` is independent of any operand, so a caller that will pad can ask first. |
| `list_backends(op, mesh) -> dict` | The **never-raising** report, for startup banners. |
| `BACKEND_CHOICES`, `EIGH_BACKENDS`, `CHOLESKY_BACKENDS`, `LU_BACKENDS`, `OPS`, `NATIVE`, `BATCHED_ROUTES`, `BATCHED_ROUTE_CHOICES`, `BATCHED_ROUTE_DEFAULT`, `MATMUL_BACKEND_CHOICES` | Vocabulary, importable with no `.so` on the machine, so a deck parser needs no FFI layer. |
| `mesh_key(mesh)`, `mesh_platform(mesh)`, `mesh_is_cpu(mesh)` | Hashable mesh identity (axes, extents, platform, device ids) and its predicates. Use `mesh_key` for any cache whose value does not retain the mesh. |
| `dial_key()`, `probe_target`, `has_target`, `backend_module` | Factory-time cache-key aggregate; capability probes with the ABSENT/BROKEN split. |
| `dispatch_batched_eigh(A, mesh, backend='distributed', *, batched_route='batch_reshard')` | `plan('eigh', …).batched(A)` for `gw.qsgw_density`. |
| `matmul(A, B, C=None, *, mesh, alpha=1, beta=0, transa='N', transb='N', backend='auto', batched_route='batch_reshard', budget_bytes=None)`, `resolve_matmul_backend` | Distributed GEMM, § [matmul](#matmul). |
| `gemm_plan(...) -> GemmPlan`, `local_gemm_plan(...)` | Resolve-once N,N GEMM for hot loops, § [Planned GEMM](#planned-gemm). |
| `panel_matmul(A, B, *, mesh, panel_bytes)` | Face GEMM with bounded contraction panels, § [Bounded face products](#bounded-face-products). |
| `contract_faces(b_X, b_Y, weights, start, stop, *, mesh, return_transpose=False)` | `(b_X·w) @ b_Yᴴ` for row faces `[b,m,K]` at `P(None,'x',None)` / `P(None,'y',None)` (or `[b,μ,s,K]`, spin merged into μ) with replicated weights `[b,K]` and interval `[start,stop)`; output `[b,m,m]` at `P(None,'x','y')`. Local GEMMs only, no collective, no provider. |
| `polar_factor`, `plan_polar_factor`, `PolarPlan` | Square polar factor / SVD, § [Polar factor](#polar-factor-and-spectral-directions). |
| `right_singular_vectors`, `leading_eigenvectors`, `retain_leading_eigenvectors` | Eager spectral-direction selection on face or batch-layout stacks, § [Polar factor](#polar-factor-and-spectral-directions). |
| `local_batch`, `batch_layout`, `is_batch_layout` | q-local kernels with resident operands, § [q-local batch](#q-local-batch-with-resident-operands). |
| `hermitian_part`, `hermitian_block`, `join_columns`, `diagonal_like`, `on_face`, `face_sharding` | Face-pinned block glue, § [Face-pinned block glue](#face-pinned-block-glue). |
| `workspace_bytes_per_rank`, `matmul_workspace_bytes_per_rank`, `fits_local` | Allocation-free workspace and capacity queries, § [Workspace queries](#workspace-queries). |
| `plan_subspace`, `plan_local_subspace`, `LocalSubspacePlan`, `plan_orthogonalization` | Iterative-eigensolver subspace plans: [Davidson](davidson.md), [orthogonalization](orthogonalization.md). |

## Contract

* **Two phases.** `plan()`, `plan_polar_factor()` and `gemm_plan()` are eager;
  what they return is trace-safe. Only platform and handler guards fire at
  resolve time; operand dtype, rank and extent are checked at call time.
* **Promise semantics.** A returned backend name means every guard passed:
  vocabulary, platform, known-broken combinations, compiled handler,
  one JAX process per device, mesh geometry, divisibility. The call cannot then
  fail for an availability or geometry reason.
* **Explicit requests refuse**, naming the failed guard and the fix. Only
  `auto` demotes, and it announces once on rank 0. An explicit
  `distributed`/`cusolvermp` eigh below n = 16384 prints a rank-0 cost notice
  (once per mesh geometry) and still runs what was asked.
* **Exceptions are `ValueError` / `RuntimeError`** (plus `TypeError` /
  `NotImplementedError` for misuse), each constructible from one string.
  `bandstructure/bse_setup.py` re-raises `type(exc)(why)`; a service-specific
  exception class would break that handler.
* **Layout.** Single tiles at `P('x','y')`, stacks at `P(None,'x','y')`.
  Eigenvalues return replicated; eigenvectors return as **columns**
  (`A Q = Q diag(λ)`) on every backend. `ensure_sharding` is the one
  FFI-adjacent reshard: a tracer gets `with_sharding_constraint`, an array
  already in layout is returned untouched, anything else is placed
  process-locally.
* **Donation is per op** (`DONATES`): `eigh` donates nothing, `cholesky`
  argument 0, `solve_lu` arguments 0 and 1. A donated operand must be a fresh
  value at the call site.
* **Tokens are opaque.** `FactorToken` exposes `op`, `backend`, `mesh`, `n`,
  `nbatch` and no factor. It is not a pytree, so a `jit` boundary refuses it
  by name. `solve()` checks `B` against `n` and `nbatch`. `factor`/`solve` is
  the split surface (getrf once, getrs many); `plan('solve_lu').batched(A, B)`
  is one complete factor + solve per call.
* **Native Cholesky and LU are caller-owned.** Under `backend='native'`,
  `Plan(A)` implements `eigh` only; `cholesky`/`solve_lu` raise
  `NotImplementedError` because their native path is a caller channel policy.
  Reach them through `batched_route='batch_reshard'`, `backend='native2d'`
  (Cholesky), or an FFI backend.

## Backends

| op | vocabulary | `auto` | `distributed` |
|---|---|---|---|
| `eigh` | `auto off distributed cusolvermp slate scalapack` | `native` everywhere | cpu → `scalapack`, CUDA → `cusolvermp`, ROCm → `slate` |
| `cholesky` | `auto off native2d cusolvermp slate` | `cusolvermp` on a CUDA mesh with px ≥ 2 and py ≥ 2 when compiled, else `native` | not in the vocabulary |
| `solve_lu` | `auto off distributed cusolvermp scalapack` | as `cholesky` | cpu → `scalapack`, CUDA → `cusolvermp`; ROCm refuses |

`off` (and the spelling `native`) resolve to `native` unconditionally. `auto`
never picks an FFI backend on a CPU mesh and never picks `native2d`.

* **`native2d`** is the pure-JAX 2-D block-distributed tiled Cholesky, on
  every platform. It is a different algorithm with a different memory
  profile: at n = 10k on P = 128 the replicated route costs 1.6 GB/device and
  `native2d` 5 MB/device. Divisibility of `n` into tiles is checked at
  resolve time.
* **Geometry guards.** cuSOLVERMp and SLATE eigh need a square mesh
  (`cusolverMpSyevd` deadlocks on rectangular blocks). SLATE otherwise needs a
  square or N×1 mesh (1×N hits a stride assert). ScaLAPACK needs a square or
  1-D mesh (square descriptor blocks). Every FFI backend needs one JAX process
  per device.
* **Known-broken combinations refuse at resolve time:**
  - SLATE `eigh` on a CPU mesh (bug L-2: host `heev` SIGSEGVs, even on 1×1).
    Use `distributed` (ScaLAPACK `pzheevd`).
  - cuSOLVERMp `eigh` with `compute_evecs=False` (bug L-3: `cusolverMpSyevd`
    status 7 at every n). Use `compute_evecs=True`.
  - SLATE `eigh` at n ≥ 4096 on a multi-rank CUDA mesh (bug L-4: SIGSEGV that
    kills every rank). It runs at n ≤ 2048; use `distributed`.
* **Unguarded defect:** cuSOLVERMp `eigh` on a 3×3 mesh hangs silently for
  n ≥ 3072 (n = 2049 completes; 2×2 at n = 8192 completes). The resolver
  does not refuse it. Keep large eigh (including the 2n polar dilation) off
  3×3 meshes. Evidence: `tests/KNOWN_FAILURES.md`, "`distributed_eigh` hangs
  at a 3×3 mesh".
* **ROCm is declared-untested.** LORRAX builds no ROCm `.so`, and JAX reports
  `Device.platform == 'gpu'` for both vendors, so a ROCm mesh currently
  resolves as CUDA (`resolve.FFI_PLATFORMS`).
* **Handler inventory.** There is no ScaLAPACK `potrf` and no SLATE
  `getrf` handler. Adding a backend is one `_<name>.py` module plus one row in
  each of `loader`'s target table, `resolve._SPEC`/`BACKEND_CHOICES` and
  `plan._IMPL`; [`docs/dev/linalg_ffi.md`](../dev/linalg_ffi.md) § "Adding a
  backend" is the procedure. The C++ lives under `src/ffi/cpp/`, and its
  target strings are frozen.

## Batched routes

`Plan.batched_route` is the one place that decides how a stack runs:

* **(a) `scan`**: `lax.scan` (`BATCHED_SCAN_UNROLL = 1`) over the backend's
  single-matrix op, compiled once per (op, backend, `mesh_key`, signature).
* **(b) `backend_batched`**: the library's stacked entry (ScaLAPACK eigh,
  cuSOLVERMp potrf, both LU backends, `native2d`, native `jnp.linalg.eigh`),
  one descriptor and one workspace for the stack.
* **(c) `batch_reshard`**: move the batch axis onto the mesh, run the native
  JAX kernel (`jnp.linalg.eigh`/`cholesky`/`solve`) on whole local matrices,
  move matrix outputs back. No distributed-library call.

Public selection is `batched_route ∈ {'batch_reshard', 'auto'}`;
`BATCHED_ROUTE_DEFAULT = 'batch_reshard'` for `plan`, `dispatch_batched_eigh`,
`matmul` and `resolve_matmul_backend`. The polar dilation, `gemm_plan` and
`factor`/`solve` keep their provider plans. Route and backend selection are
orthogonal: an explicit backend is still resolved and probed before route (c)
runs its native kernel; `backend='off'` is the provider-free spelling.

Explicit `auto` resolves to (b) when the backend has a stacked entry, else
(a). On CUDA with handlers present:

| request | 1×1 | 2×2, 4×4 |
|---|---|---|
| `eigh`, backend `auto`/`off` | native (b) | native (b) |
| `eigh`, backend `distributed` | cuSOLVERMp (a) | cuSOLVERMp (a) |
| `cholesky` / `solve_lu`, backend `auto` | native, caller-owned (no array-returning route) | cuSOLVERMp (b) |
| `matmul`, backend `auto` | cuBLASMp | cuBLASMp |

**Capacity route.** With `batched_route='auto'` and a `budget_bytes` every
rank shares, `Plan.route_for` picks (c) for a provider eigh stack when
`fits_local` admits its per-rank whole matrices (input and output,
`ceil(nb/P)` each) plus the local kernel workspace, and the provider route
otherwise. `matmul(..., budget_bytes=...)` applies the same rule to A, B and D.

**Route (c) movement.** Forward exchanges and their literal inverse run in one
`shard_map`, so GSPMD never sees a face→batch reshard it could lower as
replicate-then-partition. For padded batch `Bp = ceil(B/(Px·Py))·Px·Py`:

| step | collective (`tiled=True`) | local shape |
|---|---|---|
| input face | — | `(Bp, N/Px, N/Py)` |
| forward x | `all_to_all('x', split_axis=0, concat_axis=1)` | `(Bp/Px, N, N/Py)` |
| forward y | `all_to_all('y', split_axis=0, concat_axis=2)` | `(Bp/(Px·Py), N, N)` |
| inverse y | `all_to_all('y', split_axis=2, concat_axis=0)` | `(Bp/Px, N, N/Py)` |
| inverse x | `all_to_all('x', split_axis=1, concat_axis=0)` | `(Bp, N/Px, N/Py)` |

The inverse must run y then x; x-then-y is shape-correct on a square mesh and
scrambles the data (the test suite carries that red twin). Eigenvalues are
restored with one device `all_gather` and returned replicated; eigenvectors,
Cholesky factors and LU solutions return at `P(None,'x','y')`. Nothing crosses
the host.

* **Ragged batches** are zero-padded before the first exchange. Synthetic
  local slots never enter a dense kernel (scalar `fori_loop` + `lax.cond` on
  the global q index) and are dropped after the inverse.
* **Matrix faces are not padded:** `N % Px == N % Py == 0`, and LU RHS columns
  `NRHS % Py == 0`. Shape, rank, dtype and extent violations refuse before
  placement or any collective. Pad the matrix yourself and slice afterwards.
* **Keywords.** `block_size` (and `compute_evecs` for eigh) are dropped at the
  route boundary; any other keyword raises `TypeError`.
* **Capacity is the hard boundary.** Each device holds `ceil(B/P)` complete
  `N×N` matrices, their outputs and the native solver workspace. When one
  matrix does not fit one device, use `batched_route='auto'`.
* **CPU/MPI transport.** Route (c) planning calls `warm_mesh_cliques(mesh)`,
  which on multi-process CPU with `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`
  compiles tiny `psum`s over x, y and (x,y) on the main thread before any
  `all_to_all` is compiled from an XLA worker thread. It is a cached no-op
  elsewhere.
* **Batch-layout input.** An eigh stack already at `P(('x','y'),None,None)` is
  solved rank-locally with no movement, whatever the plan's route.

## matmul

`matmul` computes `D = alpha·op(A)·op(B) + beta·C` with `op ∈ {N, T, C}`. `C`
may be omitted only when `beta == 0`. Operands share one dtype; real operands
refuse complex `alpha`/`beta`.

| input rank | sharding | output |
|---|---|---|
| 2 | `P('x','y')` | rank 2, `P('x','y')` |
| 3 | `P(None,'x','y')`, same nonempty leading batch for A, B, C | rank 3, `P(None,'x','y')` |

Every physical input face and the output face must tile the mesh
(rows % Px, columns % Py); these checks run before placement.

**Provider route** (`batched_route='auto'`): `auto`/`distributed` resolve to
cuBLASMp on CUDA, PBLAS on CPU, SLATE on ROCm; `cusolvermp` is an alias for
`cublasmp`; explicit names never demote.

| request | CUDA | CPU | ROCm |
|---|---|---|---|
| `auto`, `distributed` | cuBLASMp | PBLAS | SLATE |
| `cublasmp`, `cusolvermp` | cuBLASMp | refuse | refuse |
| `scalapack` | refuse | PBLAS | refuse |
| `slate` | SLATE | SLATE | SLATE |
| `off` | staged route only | staged route only | staged route only |

Only `lorrax_cublasmp_batched_gemm` has a C++ handler. The PBLAS and SLATE
GEMM targets are declared in the loader but not built, so on CPU any provider
matmul refuses at the capability probe; use `backend='off'` with the staged
route there. Provider calls need float64 or complex128, an exact 2-D
`('x','y')` mesh with y-minor process order, one JAX process per cell, and
exact face tiling; cuBLASMp and SLATE also need a square mesh. The provider
aliases `C` to the output. On cuBLASMp, `T`/`C` operation codes are served by
a device face transpose of the operand followed by the N,N call (one extra
operand-sized distributed buffer per transposed operand); the native
transpose descriptors are never used.

**Staged route** (`batched_route='batch_reshard'`, the default): the route (c)
exchanges for A, B and C (C only when `beta ≠ 0`), a local `jnp.matmul`, and
the inverse exchanges for D. A rank-2 call is lifted to batch 1 and padded to
`Px·Py`, so every device holds one complete operand set. Each device holds
`ceil(B/P)` complete A, B, D (and C) matrices plus exchange buffers; use it
only when those fit. A non-`off` backend is still resolved and probed.

## Planned GEMM

`gemm_plan(mesh, *, m, k, n, nq, dtype, backend='auto', alpha=1, beta=0,
layout='face', reduction_axis=None, out_spec=None, enable_active_range=False,
warmup=True) -> GemmPlan` resolves, probes and (by default) compiles and runs
one N,N GEMM shape once. `GemmPlan(A, B, C=None, *, out=None)` is trace-safe:
no dlopen, no probe, no new `jit` wrapper inside a caller's `jit`/`lax.scan`.

* **Shapes.** `A (nq,m,k)`, `B (nq,k,n)`, `C`/`D (nq,m,n)`. `nq` is fixed at
  construction and holds k-points; flatten a spinor axis into m/k/n or call
  the plan once per spin. `nq = 1` is a legal rank-2 equivalent. float64 or
  complex128.
* **N,N only.** There is no `transa`/`transb`; pretranspose into the
  complementary face layout once.
* **`alpha`, `beta` are baked in** as FFI attributes. With `beta = 0` the plan
  also compiles a kernel that creates the zero addend inside the same program;
  `out=` donates an existing buffer instead (legal only when `beta = 0`, since
  a `beta ≠ 0` plan would scale the stale content). Pass `C` or `out`, not
  both. The cuBLASMp handler always binds a live `C` buffer; the plan removes
  the extra Python-level allocation and program, not that argument.
* **`GemmPlan.local_call`** runs the same GEMM from inside a caller's manual
  `shard_map` over the plan's mesh, on bare local tiles
  `A (nq,m/px,k/py)`, `B (nq,k/px,n/py)`, `C (nq,m/px,n/py)`. `__call__`
  cannot be entered there.

| layout | operands | implementation | active-range kernel |
|---|---|---|---|
| `face` | all at `P(None,'x','y')`; `m % px`, `k % px`, `k % py`, `n % py` | cuBLASMp; CUDA only | cuBLASMp descriptor views of each owner intersection |
| `axis`, CUDA | `A (q,m_X,k)`, `B (q,k,n_Y)`, `D (q,m_X,n_Y)`; complete local k | `local_gemm_plan`, no collective | cuBLAS pointer/leading-dimension views, fixed XLA-owned workspace |
| `axis`, CPU | as above | `local_gemm_plan`, no collective | JAX dot panels of power-of-two widths ≤ 256 |

The face layout refuses `backend='off'` by name (the staged route would
materialize complete operands on every device), and on CPU it refuses at the
provider probe, since no PBLAS/SLATE GEMM handler exists. For `layout='axis'`,
`out_spec=P(None,'x',None)` or `P(None,None,'y')` keeps a single centroid axis
sharded, and `reduction_axis='y'` / `'x'` contracts a tiled centroid axis with
a local GEMM followed by `psum_scatter` (two-axis output only).

**Active ranges.** With `enable_active_range=True`,
`GemmPlan.active_range(A, B, lo, hi, C=None, *, out=None, weights=None)`
contracts the exact interval `[lo, hi)` of k without changing allocation
shapes; `lo`, `hi` are integer scalars or `(nq,)`, `weights` is `(nq, k)`.
`GemmPlan.prepare_active_range(lo, hi)` validates host bounds eagerly and
returns `prepared(A, B, C=None, *, out=None, weights=None)` with the bounds
captured as constants (CUDA passes them as FFI metadata, with no per-call
device-to-host copy). Each prepared callable compiles lazily on first use and
the service caches no bound variants, so keep one per interval rather than
creating it inside a loop. CUDA applies weights to the full A tile once
before the call; CPU applies them inside the selected panels.
[Active ranges in planned distributed GEMM](../dev/active_gemm_ranges.md) owns
the descriptor-view design and validation.

### GEMM plans without dummy execution

`gemm_plan(..., warmup=False)` and `local_gemm_plan(..., warmup=False)` return
the same trace-safe callables without allocating dummy operands or compiling
and running standalone warmup GEMMs. Shape validation, provider probing and
communicator setup still happen at construction; the first real call pays
compilation and descriptor/workspace initialization. Use it when a one-shot
outer `jit` compiles the GEMM with its surrounding operations; keep the
default to move first-use cost ahead of a timed hot loop.

## Polar factor and spectral directions

`polar_factor(A, mesh, *, backend='distributed', rcond=None) -> (L, s)` and
`plan_polar_factor(mesh, *, n, backend='distributed', rcond=None) -> PolarPlan`
diagonalize the Hermitian dilation

$$H = \begin{pmatrix} 0 & A \\ A^\dagger & 0 \end{pmatrix}, \qquad
H \begin{pmatrix} u_i \\ \pm v_i \end{pmatrix} = \pm\sigma_i \begin{pmatrix} u_i \\ \pm v_i \end{pmatrix}$$

with one planned eigh of extent 2n (`batched_route='auto'`), read
$(u_i, v_i)$ as $\sqrt2$ times the halves of the n positive-eigenvalue
eigenvectors, and form
$L = \sum_{\sigma_i > \text{rcond}\cdot\sigma_\text{max}} u_i v_i^\dagger$. It never forms $A^\dagger A$, which squares the condition
number and loses the small overlap singular values used as a quality
diagnostic.

* **Input:** one rank-2 square float64/complex128 array at `P('x','y')`, with
  `n` divisible by both mesh axes. No implicit `device_put` or reshard; rank,
  shape, dtype, mesh-axis, divisibility and layout mismatches refuse first.
* **Output:** `L` with A's shape, dtype and sharding; `s` (length n, real,
  descending, replicated). No n² object is replicated.
* **`rcond`** is relative to max(s); `None` means n·eps(real dtype). Directions
  at or below the cutoff are dropped, so a rank-deficient A returns the unique
  polar partial isometry; a full-rank A returns the unitary polar factor.
* **Padding.** For a non-divisible logical extent, zero-pad rows and columns
  to the next common multiple, factor, and slice the leading block of L and
  the leading singular values; the pad's null directions fall below the
  cutoff. `plan_polar_factor` refuses a non-divisible `n` and reports the
  minimum pad extent.
* **Planning.** Hoist `plan_polar_factor` out of the k-point loop; the
  `PolarPlan` call is one fused, cached executable per
  (mesh, n, backend, rcond, dtype). `polar_factor` caches plans for eager
  streamed calls and refuses a tracer operand.
* **Cost:** one 2n Hermitian eigensolve plus one n³ GEMM; every matrix value is
  O(n²/P) per process (the dilation and its eigenvectors hold 4n² elements).
* **Comparison:** compare L and s across meshes; individual dilation
  eigenvectors are gauge-dependent.

`right_singular_vectors(W, tau, *, eigh_plan, column_extent, ...)` returns the
right singular directions with σ/σ_max > `tau`, closing whole multiplets at
the cut (`multiplet_tol`, default 1e-6) and optionally capped by `max_rank`.
`leading_eigenvectors(W, r, *, eigh_plan, column_extent, ...)` returns the
leading `r` eigenvectors with the boundary multiplet closed; `rcond` further
bounds the width by eigenvalues above `rcond·max|λ|`. Both are eager: `W` is a
square face `P('x','y')`, a face stack `P(None,'x','y')`, or a batch-layout
stack (solved rank-locally, returned in batch layout, `real_rows` marking
synthetic trailing slots). `eigh_plan` is the caller's resolved eigh plan (of
extent 2m for singular vectors); its route owns the local/distributed choice.
Only the O(m) spectra cross the host, broadcast from one process as bit
patterns so every rank makes the same cut. On a cuSOLVERMp plan the dilation
is solved as the positive matrix $I + H/\lVert W\rVert_F$ and the eigenvalues
mapped back by $(\lambda - 1)\lVert W\rVert_F$ (provider route only), which avoids a reproduced
cuSOLVERMp STEDC convergence failure without changing eigenvectors or cuts.

## q-local batch with resident operands

A factor used against many right-hand sides is placed once in the batch
layout, and only the right-hand sides move per call:

```python
F_b = distrib_la.batch_layout(F_face, mesh)          # once: (Bp, n, n) q-local
run = distrib_la.local_batch(lambda F, Z: F @ Z, mesh, resident=(0,))
X = run(F_b, Z_face)                                  # per call: Z moves, F does not
```

`batch_layout` accepts a face stack `(B,M,N)` at `P(None,'x','y')` (moved by
the route (c) exchanges) or a fully replicated `(B,...)` array (sliced
locally), and returns `(Bp,...)` at `P(('x','y'),None,...)` with zero pad
rows. Rank `x·Py + y` owns rows `[rank·Bp/P, (rank+1)·Bp/P)`, the order the
exchanges produce, so a resident row meets the RHS row of the same global q.
Pad rows never reach the kernel. Face operands must tile the mesh; resident
operands may have any trailing shape (a `(Bp, n)` pivot table, for example).
A `resident` position whose operand is not in batch layout refuses before
tracing. Per call the RHS costs `2·ceil(B/P)` whole blocks of exchange; the
resident operand costs nothing after placement. Capacity is the route (c)
boundary: `ceil(B/P)` whole matrices plus their RHS blocks per rank.

## Bounded face products

`panel_matmul(A, B, *, mesh, panel_bytes)` forms `A @ B` by broadcasting
contraction panels inside a `shard_map` scan. `A` is `[q,m,k]` at
`P(None,'x','y')`; `B` is `[q,k,n]` in the same layout or `[q,s,k,n]` at
`P(None,None,'x','y')`, in which case each A panel is broadcast once outside
the sample loop. Output faces keep both mesh axes; no complete row or column
is gathered. The service picks a contraction width dividing both shard
extents with `itemsize·q·width·(m/Px + n/Py) ≤ panel_bytes`. That bounds the
operand panels only: the caller admits input/output faces, compiled
temporaries (`memory_analysis()`) and provider workspace.

## Face-pinned block glue

Eager slicing, concatenation and `a + aᴴ` of face-sharded operands come out
replicated, so a `[b,R,R]` block would occupy 16R² bytes per rank instead of
16R²/(Px·Py). These helpers run the same elementwise program with the
operand's face as output sharding (one executable per function, layout and
statics), bitwise equal to the eager form:

| call | result |
|---|---|
| `hermitian_part(a)` | `(a + aᴴ)/2` for `[b,R,R]` |
| `hermitian_block(block, off, corner)` | `[[block, offᴴ], [off, corner]]`, `[b,R+r,R+r]` |
| `join_columns(a, b)` | column panels `[b,n,R]` and `[b,n,r]` concatenated |
| `diagonal_like(values, like)` | `diag(values)` in `like`'s dtype and face |
| `on_face(fn, out, *operands, **static)` | a module-level `fn` with outputs placed on `out` |

Traced operands and unsharded host arrays take the plain function. No helper
gathers, pads or reshards.

## Workspace queries

All three are allocation-free and accept float64 or complex128 only.

* `workspace_bytes_per_rank(plan, op, shapes, dtype)` returns the device
  workspace per rank of a resolved CUDA `Plan` or `GemmPlan`. For distributed
  `eigh` (`((n,n),)` or `((batch,n,n),)`, cuSOLVERMp only) it is the vendor
  workspace rounded to 256 bytes plus one private operand tile
  `(n/Px)·(n/Py)·itemsize`, which keeps eigh non-donating despite destructive
  tridiagonalization; do not add the tile again. Local eigh returns the
  cuSolverDn workspace plus a 4-byte info word, once per staged-route call
  and once per stack member on the native stacked route. For `gemm` (`((m,k),(k,n))` after
  transpose staging), one vendor workspace per context persists across
  calls, so budget `max(GEMM workspace) + max(concurrent eigh scratch)`; the
  staged route returns the compiled local-matmul temporary. Query
  collectively on the actual mesh. Non-CUDA meshes and other providers
  refuse.
* `matmul_workspace_bytes_per_rank(mesh, shapes, dtype, *, backend,
  batched_route)` is the same query for a `matmul` route.
* `fits_local(plan, op, shapes, dtype, budget_bytes)` answers the
  fit-on-one-device question: `Σ prod(shape)·itemsize + workspace ≤ budget`
  for the caller's whole per-rank live set of complete matrices. The workspace
  is the cuSolverDn query on CUDA, the LAPACK `?heevd` optimum on hosts
  (complex: lwork = 2n + n², lrwork = 1 + 5n + 2n², liwork = 3 + 5n), or the
  compiled local GEMM temporary. Pass a budget every rank agrees on (a deck
  value, never a per-rank measurement): the answer selects collectives.

The native doors are `lrx_eigh_workspace_bytes` and
`lrx_gemm_workspace_bytes`. Operand and output carriers, transpose staging,
communication buffers and persistent context resources are the caller's to
admit.

## Performance

For every matrix that fits one device, the distributed libraries' cost is
their fixed per-call charge, which is why `auto` eigh resolves to native and
route (c) is the default. Complex128, one node, 4 ranks on a 2×2 A100 mesh,
warm, one matrix per call:

| n | native replicated (s) | cuSOLVERMp (s) | SLATE (s) | cuSOLVERMp / native |
|---|---|---|---|---|
| 64 | 0.00149 | 1.586 | 0.401 | 1064× |
| 256 | 0.00412 | 1.561 | 0.546 | 378× |
| 1024 | 0.02662 | 1.754 | 1.387 | 66× |
| 2048 | 0.07275 | 1.932 | 5.444 | 27× |
| 4096 | 0.39550 | 2.739 | SIGSEGV (L-4) | 6.9× |

cuSOLVERMp eigh costs a flat ~1.55 s per matrix; 99.998% of a warm call is
inside `cusolverMpSyevd` (plan and resolve ≈ 10 µs, context-cache hit < 1 µs),
and `block_size` does not move it. Cold extras: context bootstrap
0.66–0.81 s, XLA compile ~0.8 s. On CPU (one node), ScaLAPACK `pzheevd` never
beats native replicated eigh (4.0× slower at n = 64, 2.77× at n = 2048).

Extrapolated break-even for distributed eigh is n ≈ 2 × 10⁴ (the fit window
moves it between 1.4 and 2.8 × 10⁴), the same decade where an n×n complex128
matrix, its eigenvectors and workspace stop fitting in 40 GB (n ≈ 2.7 × 10⁴).
`distributed` eigh is therefore a capacity route, not a speed route.

The route (c) exchanges and the scan route compile once per signature; a
Python loop over the batch recompiled SLATE's eager `shard_map` wrappers per
matrix (165 compiles and 5.9 s against 3 compiles and 0.17 s for an 8-matrix
factor + solve on GPU 2×2).

### Open measurement gaps

Every number above is one node with 4 ranks and n ≤ 4096, complex128. The
regime `distributed` exists for, a matrix too large for one device or a mesh
spanning nodes, is unmeasured, as are float64 rows and the SLATE CUDA eigh
crash threshold inside (2048, 4096].

## Tests

`services/distrib_la/tests/` runs from the package (`python -m pytest`) or the
repository root (`python -m pytest services/distrib_la/tests`); the service
conftest creates four emulated CPU devices before JAX imports and applies the
`services` and `distrib_la` markers through a collection hook. Inside the
LORRAX suite, select with `-m distrib_la` and deselect with `--no-services` /
`--only-service=NAME`, never a second `-m`: an explicit `-m` replaces
`addopts = "-m 'not extra'"`.

| tier | file | needs |
|---|---|---|
| shape and contract algebra | `test_distrib_la_shape_algebra.py` | nothing |
| emulated multi-device, route (c), matmul, gemm_plan refusals, polar | `test_distrib_la_{emulated_mesh,batch_reshard,matmul,matmul_plan,polar}.py` and siblings | four emulated CPU devices; skips below four |
| real multi-process | `test_distrib_la_multiproc.py` (`check_*` bodies plus a `__main__` CLI over `_CLI_CELLS`) | one process per device |
| `.so` contract and ELF acceptance | `test_distrib_la_contract.py`, `test_so_acceptance.py` | pinned libraries; binutils |
| import isolation | `test_distrib_la_import_isolation.py` | a `python -S` subprocess |
| skip honesty | `test_distrib_la_skip_honesty.py` | a machine profile: absent capability skips, built-and-broken fails |

Every check ships with its red twin, and the real 2×2 cells use non-dividing
extents with padding round trips. On Perlmutter the real-process gate is

```bash
lx run -N 1 -G 4 -n 4 python3 -u \
  services/distrib_la/tests/test_distrib_la_multiproc.py --mesh 2x2 --only batch_reshard_local_ops
```

(`--only gemm_plan` for the planned GEMM). `tests/multi_device/batched_eigh_dispatch_gate.py`
and its twin `check_batched_eigh_dispatch` require the scan and stacked routes
to agree to 0 ulp in both W and Z; `Plan.batched(..., _route=...)` is the
private override that makes that comparison possible.

## Antipatterns

* **Calling `jnp.linalg.svd` or `eigh(A.H @ A)` at a consumer.** Use
  `polar_factor`, or hoist `plan_polar_factor` for a streamed loop.
* **Passing a replicated or host matrix and relying on an implicit reshard.**
  Build it at `P('x','y')`, padded to tile the mesh.
* **A Python loop over the batch axis.** Use `Plan.batched`; a loop hands the
  compiler `nb` separate calls and puts the batch where no route can reach it.
* **Exposing a backend's stacked entry as a second public function.** It is one
  `plan._IMPL` row and is then taken automatically.
* **`lax.scan` over an FFI wrapper eagerly.** Without the per-signature cache
  it retraces and relowers every call; `Plan.batched` owns that cache.
* **Selecting a backend from the environment.** Deck keys choose
  (`eigh_backend`, `distributed_cholesky`, `distributed_lu`, `w_dyson_solver`);
  the environment only says which `.so` exists.
* **Comparing eigenvectors across meshes.** Degenerate subspaces have no
  canonical basis. Compare eigenvalues or gauge-invariant contractions
  (`Z diag(W) Zᴴ`, projectors, `|Zᴴ Z'|`).
* **Reading token internals or branching on `token.backend` /
  `plan.backend`.** The factor layout is block-cyclic on that mesh; a caller
  that branches on the backend has re-implemented the resolver.
* **Hand-rolled mesh identities.** Use `mesh_key(mesh)`; `id(mesh)` is safe
  only when the cached value retains the mesh.
* **Wrapping a refusal in `try/except` at a call site.** An explicit backend
  that cannot be honoured must raise out of the driver;
  `tests/test_charge_zeta_route.py` pins the probe calls in `isdf/core.py`
  that exist for their raise.
* **Editing `sys.path` or importing a LORRAX path helper from a consumer.** An
  installed consumer imports `distrib_la` directly;
  `tests/test_service_path_bootstrap.py` covers the checkout integration.
