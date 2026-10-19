# distrib_la: backends and the `linalg` dial

This page says which library computes each `distrib_la` operation on each
platform, how a request becomes a library, and how one deck key chooses
between whole-matrix and mesh-distributed algebra. It is for a user deciding
`linalg = local | distributed`, and for a developer adding or debugging a
backend. Read [the API](api.md) first for the layouts and the promise
semantics; the native targets and their source files are tabulated on
[the FFI layer](../../architecture/ffi_layout.md#dense-linear-algebra-targets).

## Where the libraries come from

`distrib_la` links no vendor library. Its one edge to them is
`distrib_la.loader`, which opens the LORRAX native pair at run time:
`liblorrax_ffi.so` (the CUDA leg) for CUDA meshes and `liblorrax_ffi_host.so`
(the host leg) for CPU meshes. Each leg is found, in order, at its pin
(`LORRAX_FFI_SO`, `LORRAX_FFI_HOST_SO`), in the checkout's build directory, or
on `sys.path`; it must carry the handler ABI in `loader.LORRAX_FFI_ABI_VERSION`,
the mirror of `src/ffi/common/ffi_loader.py`'s. A missing library leaves the
package importable and every FFI backend refused by name; a pinned library
that is missing, unstamped under strict mode, or of another ABI refuses
instead of falling through. How the pair is built, sealed and pinned is
[Building the FFI libraries](../../installation/ffi-build.md).

**The environment grants capability; it never selects.** Nothing in
`distrib_la.resolve` reads `os.environ`. A library's presence makes a backend
possible; the deck's `linalg` dial and the caller's explicit request choose it.

**Load order.** Both legs can link SLATE, and when they do they share the
SONAMEs `libslate.so.2` and `libblaspp.so.2`: the first leg opened decides
which copy the other binds. The host build is `gpu_backend=none`, whose
`blas::get_device_count()` is a compiled-in 0, so if it wins, every CUDA SLATE
handler sees no device and refuses. In any process that can use CUDA the
loader therefore opens the CUDA leg first (`loader._open_accelerator_before_host`,
and the same rule in `src/ffi/common/ffi_loader.py`). Before its first
`dlopen` the loader also imports `h5py` if it is installed, so h5py's HDF5
symbols bind first; h5py is not a dependency.

## The deck dial: `linalg = local | distributed` {#the-deck-dial}

Every dense solve in GW has two storage plans. **local** keeps whole per-q
matrices and schedules them q-parallel over the devices; it is mesh-invariant
and is the numerical control. **distributed** factors one matrix over the
whole mesh in a 2-D block-cyclic layout; it is the only plan whose
factorization memory divides by `P`, and it agrees with the local plan to
$\kappa\varepsilon$, not bitwise, because the summation order depends on the
process grid. `linalg` (default `local`; [input reference](../../input_reference.md))
is the only deck key for this choice. `gw_config.resolve_linalg` interprets it
once into a `LinalgResolution`, and stage code reads that record rather than
re-interpreting the dial:

| `LinalgResolution` field | consumer | `local` | `distributed` |
|---|---|---|---|
| `w_dyson_solver` | the W Dyson solve (`gw.w_isdf`) | per-q local LU | `plan('solve_lu', backend='distributed').batched` |
| `batched_route` | every `distrib_la` stack | `batch_reshard` | `auto` |
| `eigh_backend` | the `fH_q` eigensolve of htransform and exciton bands, `bse.vq_interp`'s `C_q` | `auto` (native, q-batched) | `distributed`: ScaLAPACK on CPU, cuSOLVERMp on CUDA |
| `sc_eigh` | the QSGW `H_k` eigensolve (`SCConfig.eigh`) | `auto` | `distributed` |
| `charge_zeta_solve` | the charge ζ factor | `rank_truncate` | `rank_truncate` |

The charge ζ factor is the rank-truncating eigh pseudo-inverse in both
layouts, and its eigh is always the replicated local one:
it carries the rank conditioning of $V_q$. Its eigh runs on the face through the
checked plan (`isdf.core._factor_c_q_face`), checked on the kept columns; the kept
rank can differ between grids only for an eigenvalue within rounding of the cut. Per-stage
plans and per-rank memory are in
[dense solves and GEMMs on P devices](../../architecture/dense_linear_algebra.md#stage-plans).

A deck that names a field directly (`distributed_zeta_solve`,
`distributed_cholesky`, `distributed_lu`, `w_dyson_solver`,
`distrib_la_batched_route`, `charge_zeta_solve`, `transverse_zeta_solve`,
`eigh_backend`, `sc_eigh`, `use_low_mem_eigh`) refuses by name with "use
`linalg = local | distributed`", so one key decides the layout of every
stage. No driver flag overrides it. The transverse ζ LU is the whole-tile
local LU on every layout, so it has no field.

## From a request to a library

### The vocabulary

| op | requested names | `auto` resolves to | `distributed` resolves to |
|---|---|---|---|
| `eigh` | `auto off distributed cusolvermp slate scalapack one_device` | `native` everywhere | CPU → `scalapack`, CUDA → `cusolvermp`, ROCm → `one_device` for `n ≤ 16384`, else `slate` ([size route](#size-route)) |
| `cholesky` | `auto off native2d cusolvermp slate` | `cusolvermp` on a CUDA mesh with `P_x ≥ 2` and `P_y ≥ 2` when compiled, else `native` | not in the vocabulary |
| `solve_lu` | `auto off distributed cusolvermp scalapack slate one_device` | as `cholesky` (ROCm: `native`) | CPU → `scalapack`, CUDA → `cusolvermp`, ROCm → `one_device` for `n ≤ 8192`, else `slate` ([size route](#size-route)) |
| `eigh`, `solve_lu` with a room (`budget_bytes`) | as above | as above | CUDA → `one_device` where one whole matrix's priced program fits the room, else `cusolvermp` ([room leg](#measured-route)) |

- `off` and its spelling `native` resolve to `native` unconditionally: the
  pure-JAX path (`jnp.linalg.eigh`, or the caller's own Cholesky and LU
  channel policy).
- `auto` never picks an FFI backend on a CPU mesh and never picks `native2d`.
  `auto` eigh is native because a q-batched local eigh solves `P` matrices at
  once, while a distributed library solves one matrix `P` ways and walks the
  batch serially. When `auto` Cholesky or LU demotes a compiled cuSOLVERMp
  (a 1-D mesh) or cannot load its handler on a 2-D mesh, rank 0 says so once.
  The candidate is a per-leg row (`resolve._AUTO_FFI`, CUDA only); on a leg
  without one (ROCm) `auto` is `native` with no probe and no notice.
- `distributed` means "spread one matrix over the whole mesh with this
  platform's distributed library". There is no `distributed` Cholesky, because
  the CPU Cholesky story is a channel policy in the caller, not one library.
- `native2d` is the pure-JAX 2-D block-distributed tiled Cholesky, on every
  platform. It is a different algorithm with a different memory profile: at
  `n = 10⁴` on `P = 128` the replicated route costs 1.6 GB per device and
  `native2d` 5 MB. Its tile decomposition is checked at resolve time.
- **ROCm is declared, not tested.** The platform key comes from the device
  vendor (`lxkit.device_vendor`: the client's platform and version strings and
  the device kind), so a ROCm mesh resolves as `rocm`, never as CUDA. The
  table's ROCm entries are the resolution only: `distributed` eigh names
  SLATE, the one library of the three with a ROCm build, and so does
  `distributed` LU (`slate::getrf` PartialPiv + `slate::getrs`, ROCM leg only;
  see [SLATE LU](#slate-lu)), both behind the leg's size route
  ([size route](#size-route)). The ROCm leg (`liblorrax_ffi_rocm.so`) carries
  the HIP SLATE family; a leg built without it refuses a SLATE request at the
  capability probe (guard 4).

### SLATE LU (ROCM leg) {#slate-lu}

`_slate.batched_distributed_solve_lu(A, B, mesh=)` and the split pair
`batched_distributed_getrf(A, mesh=)` → `(LU, piv)` /
`batched_distributed_getrs(LU, piv, B, mesh=)` keep the ScaLAPACK and
cuSOLVERMp contract: `A` `(Nq, N, N)` and `B` `(Nq, N, NRHS)` at
`P(None,'x','y')`, both donated, `X` in `B`'s sharding. Each matrix spans the
whole mesh, block-cyclic by relabeling as the eigh: the handler reads each
rank's own `(N/p, N/q)` buffer with a tile `nb` dividing both local edges
(`_slate.lu_layout`: the largest divisor at most 256, else one tile per rank),
so SLATE factors $\Pi A \Pi^T$ and solves for $\Pi B$. On a square mesh X
returns in A's row order with no data movement; on a `p×1` mesh one
`all_to_all` orders its rows (`_slate._rows_in_order`). Square or `p×1` meshes,
`NRHS % q == 0`. The operands are copied into SLATE-owned tiles (uniform
stride) and back. Measured at P4 2×2 (pool log `distla_lugate1`, steady s,
NRHS = n): n 8192 f64 nb 128 / 256 / 512 / 4096 (one tile) = 3.23 / 2.17 /
2.79 / 9.59; c128 3.87 / 3.63 / 4.18 / 26.1. `getrs` is now the larger half
(c128 n 8192: getrf 1.03 s, getrs 2.58 s). It runs only beyond the size
route's bound. `piv` is int64 `(Nq, P·2N)` at `P(None,('x','y'))`: each
rank's copy of SLATE's replicated pivots as (tile index, offset) pairs, opaque.
The fused entry runs the same getrf + getrs core, so the split pair reproduces
it bit for bit. Handlers: `cpp/slate/batched_solve_lu_ffi.cc`. CUDA does not
build them (its LU is cuSOLVERMp).

### The guard ladder {#guard-ladder}

`resolve_backend` applies every guard in one fixed order, all at resolve time,
so a returned name cannot fail later for an availability or geometry reason:

1. **Vocabulary.** The name belongs to the op; `distributed` is first replaced
   by the platform's library and then checked as if named.
   - **1b. Size.** On a leg with a `ONE_DEVICE_MAX_N` row (ROCm only) and a
     known `n` within the bound, `distributed` resolves to `one_device`
     instead ([size route](#size-route)); announced once per decision and
     recorded on the plan (`Plan.size_route`, `Plan.describe()`).
2. **Platform.** cuSOLVERMp is CUDA-only, ScaLAPACK host-only; SLATE is
   declared on CUDA, CPU and ROCm.
3. **Known-broken combinations**, each a deterministic crash with no Python
   traceback that the capability probe cannot see:
   - SLATE `eigh` on a CPU mesh: the host `heev` segfaults, even on a 1×1
     mesh at `n = 64`. Use `distributed` (ScaLAPACK `pzheevd`).
   - cuSOLVERMp `eigh` with `compute_evecs=False`: `cusolverMpSyevd` returns
     status 7 at every `n`. LORRAX always wants eigenvectors; ask for them
     and ignore them.
   - SLATE `eigh` at `n ≥ 4096` on a multi-rank CUDA mesh: a segfault that
     kills every rank. It runs at `n ≤ 2048`.
4. **Capability.** `loader.probe_target` confirms the handler is usable and
   distinguishes "the library would not load" from "the library has no such
   handler"; the fixes differ. Every handler of a multi-handler backend (the
   LU families) must probe usable.
5. **Coverage.** One JAX process per mesh device, because each library's MPI
   or NCCL context is per process.
6. **Geometry.**
   - cuSOLVERMp and SLATE `eigh` need a square mesh: `cusolverMpSyevd`
     deadlocks inside a collective on rectangular blocks instead of
     returning an error.
   - cuSOLVERMp Cholesky and LU need `P_x ≥ 2` and `P_y ≥ 2`; their
     block-cyclic layout degenerates on a 1-D mesh.
   - SLATE otherwise needs a square or `N × 1` mesh (below).
   - ScaLAPACK needs a square or 1-D mesh, because `pXheevd` and `pXgetrf`
     need square descriptor blocks.
7. **Divisibility.** When `n` is given, it must divide by both mesh axes: the
   one-tile-per-rank layout has no ragged tiles.

After the ladder succeeds, an explicit cuSOLVERMp `eigh` below `n = 16384`
prints a once-per-geometry rank-0 notice that distributed eigh is a capacity
route; the request is still honoured.

### What runs where {#what-runs-where}

| operation | CUDA mesh | CPU mesh |
|---|---|---|
| eigh, local | `jnp.linalg.eigh` (cuSolverDn in jaxlib) | `jnp.linalg.eigh` (LAPACK in jaxlib) |
| eigh, distributed | cuSOLVERMp `syevd` | ScaLAPACK `pzheevd` / `pdsyevd` |
| LU solve, distributed | cuSOLVERMp batched `getrf` / `getrs` | ScaLAPACK `pXgetrf` / `pXgetrs` |
| Cholesky, explicit request | cuSOLVERMp batched `potrf` / `potrs` | SLATE `potrf` / `trsm`; `native2d` |
| GEMM, distributed | `panel_matmul` (XLA SUMMA; local products on cuBLAS) | `panel_matmul` (XLA SUMMA; local products on JAX dot panels) |
| SLATE, any op | only when the CUDA leg was built against a `gpu_backend=cuda` SLATE | `potrf`, `trsm`; `eigh` refused (guard 3) |

The Perlmutter CUDA leg is built without SLATE
(`config/perlmutter/build_ffi_cuda.sh` points `LORRAX_SLATE_INSTALL_DIR` at an
empty prefix, and the sealed bundle's CUDA leg has no `libslate` dependency),
so an explicit `slate` on a CUDA mesh refuses at guard 4, naming the missing
handler. Device-side distributed algebra on NVIDIA is cuSOLVERMp, which
communicates through NCCL.

**Handler inventory.** There is no ScaLAPACK Cholesky and no SLATE LU
handler. `lorrax_slate_batched_potrf` / `_batched_trsm` exist on both legs,
but no Python wrapper calls them. No native GEMM handler is called.

## Distributed is a capacity route {#capacity-route}

### The size route (ROCm) {#size-route}

`distributed` names "one matrix over the whole mesh" because some matrix
does not fit one device. Where it does fit, the library's fixed cost is the
whole cost. On the ROCM leg SLATE's host stages (`hb2st` on rank 0, `stedc` on
HostTask) cost 0.22-0.29 s per eigh at `n = 32-128` (native 0.0016-0.0045 s,
SLATE LU is 3.9-35x native even
block-cyclic (n 512-8192; pool log `distla_lugate1`). So the ROCM leg declares a size row,
`resolve.ONE_DEVICE_MAX_N`: `distributed` with a known `n` at or below it
resolves to `one_device`, beyond it to `slate`.

`one_device` (`distrib_la._one_device`) keeps the FFI operand contract
(`P('x','y')` tile or `P(None,'x','y')` stack in, the same layout out). Route
(c)'s staged exchange moves each matrix whole onto one device, the leg's native
kernel solves it (`native_eigh`, `jnp.linalg.solve`), and the inverse
exchange returns the face layout. A stack spreads `ceil(B/P)` matrices per rank,
so the P-scaling that `distributed` exists for is kept. A single matrix is a
stack of one: rank 0 solves it and the inverse exchange scatters that one
result. Per round a rank holds at most `max_n²` elements of whole matrices,
in equal rounds otherwise. The plan's checked chain applies as to a library.

This is not the silent downgrade the resolver forbids. That rule protects the
replicated factor's O(nq·μ³)-with-no-P-scaling wall and the route changes no one
sees. The size route is declared in a leg row, printed once per decision
(rank 0) and kept in `Plan.describe()`. An explicit `slate` is still honoured
as named. `one_device` refuses (`GATE one_device`) on a leg without a row or
beyond the bound. The capacity provider of an `auto` plan with a room
(`resolve_eigh_capacity`) stays the library. It runs only stacks of which not
one whole matrix per rank fits the room, and `one_device` would need exactly
that.

Measured at P4 2×2 MI250X, complex128, steady s (pool log `distla_bound1`):

| n | eigh `one_device` | native, 1 GCD | SLATE 2×2 (before) | LU `one_device` (NRHS = n) | native LU |
|---|---|---|---|---|---|
| 32 | 0.0061 | 0.0016 | 0.236 | — | — |
| 128 | 0.0091 | 0.0045 | 0.289 | — | — |
| 2048 | 0.151 | 0.142 | 3.97 | 0.051 | 0.034 |
| 8192 | 6.45 | 6.37 | 17.3 | 1.06 | 0.94 |
| 16384 | 42.4 (peak 27.9 GB/rank) | 42.1 | 76.4 | beyond the bound | wrong (below) |

The SLATE column at n ≥ 2048 is from the SLATE-perf lane. The LU bound is 8192.
The native complex128 LU on one GCD (jaxlib getrf/getrs on rocSOLVER, ROCm
7.14.1) returns a wrong solution with no error at n = 12288 and 16384 (residual
8e-3; pool log `distla_lubig`). float64 at 16384 and complex128 at n ≤ 11616 are
exact (`distla_lubig2`). The checked solve refused the one-device result by name
(`GATE distrib_la_result_check`). Beyond 8192 the LU is SLATE's.

### The CUDA room leg and the measured size route {#measured-route}

CUDA has no size row. It is a room leg (`resolve.ONE_DEVICE_BY_ROOM`): given a
room, `distributed` resolves to `one_device` where one whole matrix's priced
program fits it (`resolve.one_device_max_n`, the largest such `n`), else to
`cusolvermp`. Without a room it is `cusolvermp`.

A `distributed` plan resolved to `one_device` then decides each call signature
against main's incumbent (`plan._measured_rounds`): `cusolvermp` on CUDA,
`one_device` in the round rule's count on ROCm. The candidates are the
`one_device` round counts that could beat it, priced by XLA's
`memory_analysis` of their compiled programs. The fewest-round candidate whose
per-rank bytes are within the incumbent's and within the room runs, else the
incumbent. Every rank's figures are agreed over the coordination service's
key-value store, with the largest candidate bytes against the smallest
incumbent bytes. A room that differs across ranks is refused by name. The
agreed figures are cached on disk and in memory per signature without the
room, so a later room is decided by arithmetic: no exchange, no disk read, no
compile. Each decision is a route-ledger line. The result is what
`Plan.decision` reports ([decision-keyed programs](api.md#eigh-stack)).
Vendor workspaces allocated at run time (jaxlib's cuSOLVER `getrf`/`getrs`
scratch) are outside the analysis.

For a matrix that fits one device, a distributed library's cost is its fixed
per-call charge, almost all of it inside `cusolverMpSyevd`, which the local
kernel does not pay. That is why `auto` eigh resolves to native and route (c)
is the default stack route: `distributed` is for a matrix too large for one
device or a mesh spanning nodes, never for speed. Measurements behind the
rule are in the docstring of `distrib_la.resolve._announce_eigh_fixed_cost`.

**cuSOLVERMp eigh is block-cyclic by relabeling.** A `P('x','y')` operand is
one `(n/p, n/p)` tile per rank. Handing cuSOLVERMp the same local buffers with
a smaller square block `mb | n/p` (`mb ≤ 256`, `_cusolvermp._block_size`)
makes it read them as $\Pi A \Pi^T$ for a block permutation $\Pi$: the
eigenvalues are A's, and one `all_to_all` over `x` returns the eigenvector rows
to ascending order. The tridiagonalization then keeps every rank busy as the
trailing matrix shrinks. Measured (complex128, median seconds, one tile per
rank → `mb = 250`, the un-permute included): P16 `n = 16000` 17.50 → 10.59,
`n = 8000` 4.46 → 3.56; P64 `n = 16000` 20.05 → 15.65, `n = 8000`
7.13 → 6.47; no change at `n/p ≤ 256`. Where the largest divisor of `n/p` up
to 256 is below 128 (`n/p` prime or a small multiple of one: 3954 = 2·3·659
gives 6), the solve pads each tile to the smallest edge with a divisor in
[128, 256] (`_cusolvermp.solve_layout`, at most 127 rows per rank: 7908 →
7912, block 172). The padded rows are zero except a diagonal of distinct
sentinels below the Gershgorin bound, so they are the lowest eigenpairs and
are dropped before the result leaves the wrapper. The cuSOLVERMp handlers
zero and read the solver's `info` after every `syevd`, `potrf`, `potrs`,
`getrf` and `getrs` and return a nonzero value as an error naming the
routine (`src/ffi/cpp/cusolvermp/info.h`). LU and Cholesky keep one tile per rank,
because block-cyclic `getrs`/`potrs` are 2–3× slower and `getrf` gains
nothing.

## Inside the ScaLAPACK handlers {#inside-the-scalapack-handlers}

These failures live in `src/ffi/cpp/scalapack/`, below every Python guard: the
handler is compiled and the probe passes, and the failure is numerical or
environmental.

- **Eigenvector workspace.** `pXheevd` can return `INFO = 0` with correct
  eigenvalues and a garbage `Z`: its back-transform (`pXunmtr`/`pXormtr`)
  needs more `WORK` than `pXheevd`'s published `LWORK` formula or MKL's query
  provides. `eigh_ffi.cc` floors `LWORK` at
  `max(NB(NB−1)/2, (NP0+MQ0)·NB) + NB² + 8N`, the back-transform's own bound
  plus the five length-`N` vectors `pXheevd` carves off the front. An
  eigenvalue-only test does not test an eigensolver: check $AZ = Z\,\mathrm{diag}(w)$.
- **The workspace query is mandatory.** MKL's `pzheevd` rejects the netlib
  minimum with `INFO = −16` and asks for far more on multi-rank grids, so a
  failed query is fatal and the handler uses `max(query, formula)`. The
  workspace is allocated inside the handler, invisible to XLA's memory
  planner; `LORRAX_DEBUG_PRINT=1` prints it per call.
- **SLATE's ScaLAPACK overlay is refused.** `libslate_scalapack_api.so`
  redefines `pzheevd_`, `pdsyevd_`, `pzgetrf_`, `pdgetrf_`, `pzgetrs_` and
  `pdgetrs_`, so an `LD_PRELOAD` would replace every routine this backend
  calls while `resolve` still reports `scalapack`. The overlay assumes rank
  `mx + my·p` for shard `(mx, my)` where LORRAX's mesh puts it on `mx·q + my`,
  and its shims hard-wire `info = 0`. `blacs_grid.h` resolves the provider of
  each routine (`dlsym` + `dladdr`) and refuses, naming it;
  `LORRAX_SCALAPACK_ALLOW_SLATE_API=1` downgrades the refusal to one stderr
  line for deliberate measurement.
- **MKL thread team.** At production grids `pzheevd`/`pzgetrf` issue
  thousands of small BLAS calls between latency-bound BLACS collectives, and a
  wide MKL team starves MPI progress (24× slower at a 12×12 grid, `n = 2448`).
  The handlers pin the calling thread's team through
  `mkl_set_num_threads_local` (`common/mkl_thread_pin.h`) to
  `min(current, 4)`; `LORRAX_SCALAPACK_MKL_THREADS` overrides, and the pin is a
  no-op on a non-MKL ScaLAPACK.

## SLATE {#slate}

SLATE serves explicit Cholesky on CPU meshes. It is also the library a ROCm
`distributed` eigh resolves to, which no LORRAX build serves (guard 4). Its Python wrappers are in `distrib_la._slate` (reached through
`distrib_la.backend_module('slate')`, never imported directly):
`distributed_cholesky` (`slate::potrf`, returning an opaque lower factor) and
`distributed_eigh` (`slate::heev`, refused on CPU and above `n = 2048` on
multi-rank CUDA). Three pieces make SLATE read JAX-sharded data:

1. **Local transpose.** JAX tiles are row-major and SLATE tiles column-major.
   Each rank transposes its own `(n/p, n/q)` shard to `(n/q, n/p)` inside a
   `shard_map`, which is the original block in column-major order, with no
   inter-rank traffic.
2. **Rank remap.** SLATE's `fromDevices`/`fromScaLAPACK` place tile `(i, j)`
   on rank `i + j·p`, while JAX's mesh puts shard `(mx, my)` on rank
   `mx·q + my`. `src/ffi/cpp/slate/context.cc` rebuilds the communicator with
   `MPI_Comm_split` key `(r / q) + (r % q)·p`, so SLATE's tile-to-rank map
   equals JAX's shard-to-rank map.
3. **One tile per rank.** The tile size is `nb = n / max(p, q)`, the only
   value for which JAX's contiguous block equals SLATE's block-cyclic tile on
   a multi-rank mesh; a `block_size` override is accepted only on a 1×1 mesh.

That last rule fixes the geometry SLATE accepts: a square mesh, or `N × 1`.
With both axes above 1 and `p ≠ q`, no square tile gives one tile per rank on
both axes and the result would be silently wrong. A `1 × q` mesh has local
stride `n` but tile size `n/q`, which trips a SLATE assertion
(`internal_batch.hh`: `group.ld[m] == Mij.stride()`) and aborts every rank; use
the transposed `q × 1` mesh. `heev` additionally needs `p = q`. An exception
thrown inside SLATE's OpenMP tasks cannot be caught by the handler and
terminates every rank, so every layout precondition is checked in Python first
(`_slate.validate_mesh`, `_slate.validate_tile_layout`, and the mirrored
geometry guard in `resolve._check_geometry`). Each wrapper requires
`p·q == jax.process_count()`.

On a CPU mesh the host handlers (`src/ffi/cpp/slate/host_ffi.cc`) build SLATE
matrices with `fromScaLAPACK` on host buffers and run `Target::HostTask`; the
transposes, rank remap and validation are the same. The host leg links a
`gpu_backend=none` SLATE built by `src/ffi/cpp/stage/slate_build_perlmutter.sh
cpu` on Perlmutter ([Building the FFI libraries](../../installation/ffi-build.md)).

## GEMM providers and active-range kernels

Every `matmul` request except `off` runs `distrib_la.panel_matmul`, the
batched 2-D SUMMA in XLA, on every platform. The retired names `cublasmp`,
`cusolvermp`, `scalapack` and `slate` are still accepted and run it too;
`off` is the staged route only. The face GEMM needs `float64` or
`complex128`, a square `('x','y')` mesh and exact face tiling, and takes
`transa`/`transb` itself. It replaced cuBLASMp, which it beat 1.4-1.5x on
the ζ-projector and SC-rotation shapes at P4 (sandbox claim 3987).

[Active ranges](api.md#active-ranges) are implemented per route:

| route | operands | kernel |
|---|---|---|
| CUDA face | both at `P(None,'x','y')` | `panel_matmul`'s local panel products on `lorrax_cublas_local_active_range_gemm` (host-known intervals: `…_prepared_active_range_gemm`) |
| CUDA axis | A `P(None,'x',None)`, B `P(None,None,'y')` | `lorrax_cublas_local_active_range_gemm`: local BLAS pointer views, no collective |
| other face | both at `P(None,'x','y')` | `panel_matmul`'s local panel products on the axis kernel below |
| ROCM axis | as CUDA axis | `lorrax_active_gemm_device`: device-side bounds, one run-time-compiled kernel |
| other axis | as CUDA axis | `_active_local.active_local_matmul`: JAX dot panels |

`_active_local.ACTIVE_LOCAL_ROUTE` holds the row per leg (CUDA `native`,
ROCM `device`, cpu `interval`).  `native` is the cuBLAS/rocBLAS pointer-view
handler (`common/gpu_blas.h`); it reads the bounds to the host and syncs the
stream on every call, which on MI250X drains the stream inside Sigma's tau loop
(Fe P=4 Sigma tau 13.5 s against 9.1 s masked).  `device`
(`_active_local_device`, `src/ffi/cpp/active_gemm/`) is one run-time-compiled,
leg-neutral kernel that reads the bounds on the device: per call no host read,
no sync, no cond, no operand or weighted copy and no workspace; each row's K
loop runs over its active width only and never loads inactive storage; weights
are applied on load; invalid bounds give all-NaN.  On gfx90a / gfx94x the inner
product runs on the FP64 matrix cores (`v_mfma_f64_16x16x4f64`, complex128 as the
exact four-product form; MI250X, Fe W shape c128[64,156,936] at full K: 1.13 ms
vs 0.81 ms rocBLAS and 1.87 ms on the FMA path); other targets take the FMA path.  The CUDA leg adopts it by
listing the source.  The `masked` and `interval` routes stay as the XLA
references; `tests/test_active_local_native_gate.py` gates both native routes
against them on the same device.  The cuBLAS kernel stays the CUDA arm: on
windowed bounds it is 3.7-5.0x faster than the JAX
panels at P4, and equal on a full window (sandbox claim 3987).

- **Local cuBLAS views.** `_active_local_cuda.active_local_cuda` views
  row-major tiles as column-major transposes, $C^T = B^T A^T$, and moves the
  base pointers by the interval while keeping leading dimensions and strides;
  consecutive rows with equal bounds share one strided-batched call. The
  handler binds a dedicated cuBLAS handle to the call's stream with a 4 MiB
  workspace that XLA owns (an explicit scratch output), reset on every call so
  no state leaks between invocations. Weighting forms one weighted-A tile.
- **JAX panels.** A `lax.while_loop` walks the interval in disjoint slices of
  power-of-two widths up to 256 columns, chosen by `lax.switch`, so at most
  nine statically shaped dots compile regardless of `K`, and changing the
  bounds recompiles nothing. Weights are applied inside each slice. Rows with
  equal intervals stay batched; differing intervals scan the rows.

## Adding a backend {#adding-a-backend}

Every step is inside `services/distrib_la/`, except a loader row in `src/` when
the LORRAX loader also reaches the handler.

1. Write `distrib_la/_<name>.py`, modelled on `_cusolvermp.py`, whose
   docstring names the three per-routine decisions: donation, handle versus
   array return, and output normalization.
2. Register the C++ handler symbol in `distrib_la/loader.py`
   (`_CUDA_TARGET_SYMBOLS` / `_HOST_TARGET_SYMBOLS`); this makes the
   capability probe work, with the absent-versus-broken split. If
   `src/ffi/common/ffi_loader.py` also reaches the handler, add the row there.
3. In `distrib_la/resolve.py`: the name in `BACKEND_CHOICES` per op, one
   `(op, backend) → (targets, platforms)` row in `_SPEC`, any geometry rule in
   `_check_geometry`, a `_DISTRIBUTED_DEFAULT` row if it is a platform's
   `distributed` answer, and a branch in `backend_module`.
4. One row in `distrib_la/plan.py`'s `_IMPL`: the single-matrix entry, the
   stacked entry and an output normalizer. A missing stacked entry is filled
   by `lax.scan` over the single one; a single-matrix entry that returns a
   library handle sets `one_handle=True`. Normalize conventions here, never at
   call sites.
5. The eigh vocabulary reaches the resolver's validation through
   `gw_config.eigh_backend_choices()`, which reads `BACKEND_CHOICES`; nothing
   else changes.
6. Add a check beside the existing one in `services/distrib_la/bench/`
   (`cusolvermp_eigh_test.py`) that asserts the
   residual ($AZ - Z\,\mathrm{diag}(w)$, or the factor's reconstruction) on a
   real mesh, and update the tables on this page.
