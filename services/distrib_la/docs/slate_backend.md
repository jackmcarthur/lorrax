# The SLATE backend — distributed dense linear algebra from JAX

This is the backend-maintainer companion to the public service guide at
`docs/services/distrib_la.md`. `distrib_la` remains independently installable:
its Python package imports no LORRAX module, and SLATE is an optional FFI
capability rather than a Python dependency. The current compatible provider
is built by LORRAX's `src/ffi/cpp/slate/` tree; the build sections below
document that provider, not an installation prerequisite for native routes.

The backend implementation is consolidated in private module
`distrib_la._slate`, reached by maintainers through the public
`distrib_la.backend_module("slate")` lookup. Consumers use `plan`, `factor`,
`solve`, and top-level `matmul`; importing `_slate` directly is outside the
public package API.
The old `ffi.slate` re-export and split `cholesky.py`, `trsm.py`, `eigh.py`,
`batched.py`, and `context.py` files no longer exist.

JAX FFI wrappers around [SLATE](https://icl.utk.edu/slate/) (a tile-based
MPI + GPU dense linear algebra library from ICL).  Currently exposes:

- `distributed_cholesky(A, mesh)` → `SlateLowerL`  (`slate::potrf`)
- `distributed_trsm(A_or_handle, B, mesh, ...)`   (`slate::trsm`)
- `batched_distributed_matmul(A, B, C, mesh, ...)` (`slate::multiply`)
- `distributed_eigh(A, mesh)` → `(W, Q)`           (`slate::heev`; W AND
  true column eigenvectors — `A @ Q == Q @ diag(W)` at ~1e-14.  The
  historical "eigvec layout artifact" was root-caused 2026-07-10: the
  FFI read stale device tiles without `tileGetForReading` — heev's
  back-transform leaves the valid Z copy on the host — plus a missing
  local-transpose pair on top. Fixed in `distrib_la._slate` and
  `src/ffi/cpp/slate/eigh_ffi.cc`.)

cholesky, trsm and eigh hit machine-precision residuals (~1e-16 /
~1e-14) on meshes with `p == q` or `q == 1` (N×1) where
`p * q == jax.process_count()` — including rectangular RHS (`m != n`)
for trsm, which used to abort the whole job.  `1×q` meshes are rejected
(SLATE-internal stride assertion — see Restrictions).

## Public quick start

```python
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from distrib_la import factor, matmul, solve

mesh = Mesh(np.asarray(jax.devices()).reshape(p, q), ('x', 'y'))
A = jax.device_put(A_stack, NamedSharding(mesh, P(None, 'x', 'y')))
B = jax.device_put(B_stack, NamedSharding(mesh, P(None, 'x', 'y')))

token = factor('cholesky', A, mesh, backend='slate', n=A.shape[-1])
X = solve(token, B)  # solves A[q] @ X[q] == B[q]

# The public door lifts rank-2 operands or accepts the rank-3 face stacks
# above. backend='auto' also selects SLATE on a ROCm mesh.
D = matmul(A, B, mesh=mesh, backend='slate')
```

The token intentionally hides `SlateLowerL`; it cannot be reshared or passed
through `jit`. Backend maintainers who need the raw primitives may use
`slate = distrib_la.backend_module('slate')`, but application code should not.

Local service tests and the current Perlmutter multi-process spelling are:

```bash
cd services/distrib_la
python -m pytest tests/test_distrib_la_contract.py -k slate
python -m pytest tests/test_distrib_la_matmul.py

cd ../..
export LX_BASE_MODULE=lorrax_J070
lx run -N 1 -G 4 -n 4 python3 -u \
  services/distrib_la/tests/test_distrib_la_multiproc.py \
  --mesh 2x2 --only slate
```

The claims-style performance driver is
`services/distrib_la/bench/bench_distrib_la.py`. It records baselines and is
not a pass/fail test.

## Design rationale

Three coordinated pieces make SLATE play nicely with JAX-sharded data:

1. **Local transpose** (Python).  JAX is row-major; SLATE tiles are
   col-major.  Each rank locally transposes its own
   `(n/p, n/q)` shard to `(n/q, n/p)` via `shard_map` + `jnp.transpose`
   — pure local op, no inter-rank comm.  The transposed bytes are the
   original block in col-major layout, which SLATE reads correctly.

2. **MPI rank remap** (C++ `src/ffi/cpp/slate/context.cc`).  SLATE's `fromDevices`
   hardcodes `GridOrder::Col` (rank of tile `(i, j)` = `i + j*p`),
   while JAX's C-order mesh reshape puts shard `(mx, my)` on rank
   `mx*q + my`.  These don't agree for `p != 1, q != 1`, so we
   `MPI_Comm_split` with key `= (jax_rank / q) + (jax_rank % q) * p`
   to rebuild the comm such that SLATE's tile-to-rank assignment matches
   JAX's shard-to-rank assignment.

3. **GPU-aware MPI** (wired up by the `lx` launch environment, build-time by
   `src/ffi/cpp/run_shifter.sh` through `src/ffi/cpp/in_container.sh`).
   Cray MPICH on Perlmutter does GPU-Direct
   RDMA over Slingshot (the closest equivalent to NCCL for any
   MPI-based library) only when `MPICH_GPU_SUPPORT_ENABLED=1` and
   `libmpi_gtl_cuda.so` is loaded.  shifter's `--module=mpich`
   explicitly *unsets* the env var per `/etc/shifter/udiRoot.conf`,
   so `in_container.sh` re-asserts it inside the container.  The
   module also `LD_PRELOAD`s the CUDA-12 copy of `libmpi_gtl_cuda.so.0`
   from the stage so the loader doesn't pick up shifter's CUDA-11-linked
   one and look for `libcudart.so.11.0`.

The combination yields the standard SLATE convention (`side='L',
uplo='L', op='N'`) for the trsm of a Cholesky factor: no inverted
sides, no opaque-handle gymnastics inside the user's mental model.

## Distributed GEMM

The public `distrib_la.matmul` door maps `backend='slate'` to
`slate::multiply`. It accepts rank-2 `P('x','y')` matrices or rank-3
`P(None,'x','y')` stacks and returns the same-rank face layout. Rank 2 is
lifted to a one-element stack before the private wrapper; the handler loops
over rank-3 stack elements while reusing the process-grid context. Operation
codes are `N`, `T`, and `C`, and the full contract is
`alpha * op(A) @ op(B) + beta * C` for `float64` or `complex128`.

The CUDA handler in `gemm_ffi.cc` constructs matrices with `fromDevices` and
runs `slate::multiply(..., Target::Devices)`. The host handler in
`gemm_host_ffi.cc` uses `fromScaLAPACK` and `Target::HostTask`. Both consume
the same JAX face layout, reuse the context's MPI rank remap, and alias C as D
when XLA honours donation. The wrapper locally transposes row-major JAX tiles
on entry and transposes the result back on exit, just as the other SLATE
surfaces do.

This provider path is the default `matmul` route on ROCm; explicit `slate`
also selects it on CUDA or CPU when the matching handler exists. It is
distinct from `batched_route='batch_reshard'`, which exchanges faces x then y,
runs local JAX GEMM, and exchanges D back y then x. A non-`off` SLATE request
is still capability-probed even with that staged route; use `backend='off'`
for a provider-free staged call. Leading-batch padding belongs only to the
staged route. The SLATE provider loops the batch as supplied and requires all
physical and output matrix extents to tile the process grid exactly.

## Restrictions / known gaps

- `mesh` must have axes named `('x', 'y')`.  No 3-D-mesh / sub-mesh
  selection yet (see *GWJAX adaptation* below for the workaround).
- `p * q == jax.process_count()`.  Partial-world / sub-comm calls
  aren't implemented.
- The solver wrappers require `p == q` or `q == 1` (N×1) — with both axes > 1
  and `p != q`, the
  square SLATE tile size cannot give one tile per rank on both axes, so
  JAX's block shards and SLATE's block-cyclic tiles diverge (silent
  wrong answers).  `1×q` grids additionally hit a size-dependent SLATE
  assertion (`internal_batch.hh:290: group.ld[m] == Mij.stride()`,
  SIGABRT on all ranks): local stride `lld = n` ≠ tile size `nb = n/q`,
  and SLATE's device-region batching wants uniform strides.  `p ≥ q`
  meshes have `lld == nb` and are safe. `validate_tile_layout` in
  `distrib_la._slate` rejects both classes; use the transposed (q×1) mesh.
  The batched wrappers' per-slice `(1, Py)` sub-grid is the same stride
  class — production-validated on 2×2, but the 1×4 `nbatch=8, n=128`
  assert repro documented below remains an accepted risk there.
- `heev` additionally requires `p == q` (SLATE's algorithm).
- For solver matrices, default tile size `nb = n // max(p, q)` is the ONLY
  layout-consistent
  value on multi-rank meshes (one tile per rank); `block_size=`
  overrides are rejected there and only allowed on a 1×1 mesh.  The
  same invariant is why trsm's X needed per-dimension tile sizes in the
  FFI (solve dim conforms with A's `nb`, free dim gets one tile per
  rank): the old square-`nb` X on a rectangular RHS aborted every rank
  via an uncatchable `blas::Error` from a SLATE OpenMP task (2×2) or
  silently mis-assembled B (1×4).  Fixed 2026-07-10 in
  `src/ffi/cpp/slate/{trsm,batched_trsm}_ffi.cc`. Distributed GEMM also
  requires a square process grid: with one face tile per operand, SLATE's
  inner tile count is `Py` for A but `Px` for B on an N/N rectangular grid.
  The public resolver, private wrapper, and C++ handler all refuse that
  geometry before entering SLATE; PBLAS remains valid on rectangular grids.
- Exceptions thrown inside SLATE's OpenMP tasks CANNOT be caught by the
  FFI handler's try/catch — they `std::terminate` all ranks.  Layout
  preconditions must be validated Python-side before invoking the FFI
  (`validate_mesh` + `validate_tile_layout` + per-wrapper shape checks).
- 3 local transposes per cholesky+trsm chain.  At `n ~ 1k–4k` on
  4 A100s the chain runs ~150–500 ms; below ~1k the transposes are a
  visible fraction of total time.
- SLATE GEMM accepts only F64/C128, matching operand dtypes, `N/T/C` operation
  codes, a nonempty matching rank-3 batch (or the public rank-2 lifting), and
  A/B/D physical extents divisible by the corresponding mesh axes. A supplied
  C must match D exactly; `C=None` is legal only when `beta == 0`.

## GWJAX adaptation: batched `(Nq, Nmu, Nmu)` cholesky+trsm

User scenario: a batch of `Nmu × Nmu` Hermitian PD matrices, depth
`Nq` (1–8000).  Each `Nmu × Nmu` is too big for one GPU, so each must
be 2-D-sharded; multiple batch elements should be processed in
parallel across the available GPUs.

**Implemented** in `distrib_la._slate` plus
`src/ffi/cpp/slate/{batched_potrf,batched_trsm}_ffi.cc`. The sharding
contract is `P('x', None, 'y')`: batch across `'x'`, inner matrix
across `'y'`.  Each X-row of the mesh gets its own MPI sub-comm of
size `Py` (`MPI_Comm_split` by `x_rank`, via
`lrx_slate_subrow_context_create`), and the FFI handler loops over the
per-rank batch on that sub-comm with SLATE grid `p=1, q=Py`.

SLATE has no native batched `potrf`, so the inner loop is a plain
C++ `for` — but the sub-comm setup is shared across iterations, which
is the only amortisable bit.  The real batched kernel lives in
cuSOLVERMp; a parallel cuSOLVERMp implementation with a true batched
API is planned.

Correctness verified on a 2×2 mesh (`nbatch=8, n=128, c128`) at
machine precision (~2.3e-16 for cholesky+trsm).  1×4 mesh currently
trips an internal SLATE assertion in `internal_batch.hh:290` — out
of scope; cuSOLVERMp is the answer if you need rectangular meshes.

## Future work

- **Sub-mesh / partial-world support.**  Generalise the sub-row
  `MPI_Comm_split` trick to arbitrary sub-meshes for ops other than
  the batched-along-X case, so callers can compose a SLATE op across
  an arbitrary `n < world_size` process slice.  Today
  `validate_mesh` rejects anything where `p*q != jax.process_count()`.
- **Batched cuSOLVERMp.**  cuSOLVERMp exposes a real batched
  `syevd` / `potrf` (NCCL-backed) that will be much faster than SLATE's
  for-loop for small-n Hermitian batches; SLATE remains the AMD-GPU
  fallback path (HIP/Frontier).
- ~~**`heev` eigenvectors.**~~  RESOLVED 2026-07-10 — stale MOSI tile
  read + missing transpose pair; see `src/ffi/cpp/slate/eigh_ffi.cc`.
- **`gels` / least-squares.**  Useful for the ISDF fitting paths
  (separate ticket).

## Files

- `services/distrib_la/src/distrib_la/_slate.py` — Python contexts,
  validation, handles, and all single/stacked wrappers.
- `services/distrib_la/src/distrib_la/factor.py` — the public opaque-token
  Cholesky factor/solve path.
- `services/distrib_la/src/distrib_la/loader.py` — provider discovery,
  handler registration, ABI check, and CUDA-before-host load order.
- `src/ffi/cpp/slate/{ctx.h,context.cc}` — provider contexts and MPI remap.
- `src/ffi/cpp/slate/{eigh,potrf,trsm,gemm}_ffi.cc` — CUDA handlers per op;
  GEMM calls `slate::multiply`.
- `src/ffi/cpp/slate/batched_{potrf,trsm}_ffi.cc` — batched sub-row handlers.
- `src/ffi/cpp/slate/host_ffi.cc` and `gemm_host_ffi.cc` — CPU variants in
  `liblorrax_ffi_host.so`.
- `src/ffi/cpp/stage/slate_build_perlmutter.sh` — reproducible provider
  SLATE builds (see below).

## Building

### 1. SLATE itself (host-side, Cray PE — not in the container)

```
bash src/ffi/cpp/stage/slate_build_perlmutter.sh gpu    # gpu_backend=cuda
bash src/ffi/cpp/stage/slate_build_perlmutter.sh cpu    # gpu_backend=none
```

Idempotent; installs under `$HOME/software/slate_builds/{gpu,cpu}/install`
from a source checkout pinned at `v2025.05.28-1` (override:
`LORRAX_SLATE_COMMIT`).  Module stacks (script-loaded, per NERSC docs):
GPU = `PrgEnv-gnu cray-libsci cmake cudatoolkit craype-accel-nvidia80`;
CPU = same minus the CUDA pair (explicitly unloaded — with
`craype-accel-nvidia80` loaded the CC wrapper links `libmpi_gtl_cuda`,
whose `libcuda.so.1` driver dependency is a portability landmine).
The script's comments explain every non-obvious flag, especially
`-DSCALAPACK_LIBRARIES=""` (ScaLAPACK lives *inside* wrapper-linked
libsci; empty string keeps the tester's reference checks without a bogus
`-lscalapack`).  If the login node is loaded, run it through the
allocation: `srun --jobid=$SLURM_JOBID --overlap -N1 -n1 -c 48 bash
src/ffi/cpp/stage/slate_build_perlmutter.sh gpu`.

**Which build where**: GPU nodes use `gpu/`; CPU nodes use `cpu/`.
SLATE's execution target is a *runtime* option, and the cuda build does
run host-side (`--target t` — verified on both node types; Perlmutter
CPU nodes happen to ship `libcuda.so.1`), but the `none` build is the
config that carries to non-NVIDIA machines and never drags CUDA/GTL
into a CPU-node link.

### 2. The FFI .so (inside the container)

```
bash src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh
```

Login node works when shifter cooperates; otherwise run with
`SLURM_JOBID` set to go through a compute node.  To build against a
non-default SLATE without touching the in-tree `build/` (which other
sessions may be using):

```
export LORRAX_SLATE_INSTALL_DIR=$HOME/software/slate_builds/gpu/install
export LORRAX_FFI_BUILD_DIR=$HOME/software/slate_builds/ffi_build_gpu
bash src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh
# then at run time:
export LORRAX_FFI_SO=$LORRAX_FFI_BUILD_DIR/liblorrax_ffi.so   # loader override
export LORRAX_SLATE_INSTALL_DIR=...                            # LDLIB override
```

### CPU story — host platform SUPPORTED (2026-07-10)

The SLATE ops run on the JAX CPU backend through host handler variants
(`src/ffi/cpp/slate/host_ffi.cc`, plus `gemm_host_ffi.cc` for multiply):
`fromScaLAPACK()` on the host buffers (same 2-D
block-cyclic layout + GridOrder::Col as `fromDevices`, so the local
transposes, comm rank-remap, and every mesh/tile validation carry
unchanged), `Target::HostTask`, plain `memcpy` staging, no stream ctx.
They compile ONLY into a separate CUDA-free library:

```
bash src/ffi/cpp/build_host.sh     # → host/build/liblorrax_ffi_host.so
```

built host-side (Cray PE, no container) against the `cpu`
(`gpu_backend=none`) SLATE install, with the XLA FFI headers staged from
the container's jaxlib (the runtime the .so must match).  The script
fails if the result links any CUDA-stack library.

`distrib_la.loader` registers the host handlers under `platform="cpu"` and
the CUDA ones under `platform="CUDA"` — same target names, so
`jax.ffi.ffi_call` sites resolve by lowering platform exactly like
jaxlib's cpu (lapack) vs CUDA (cusolver) kernel split.  Wrapper call
sites pick the library from the MESH's device platform
(`context.ensure_registered`), so slate-on-CPU-devices works even inside
a GPU-backend process.  Input-file selection: `distributed_cholesky =
slate` now passes through on the CPU backend (still never auto-picked;
still fails loudly with build pointers when the library is absent).

Tests: `services/distrib_la/tests/test_distrib_la_contract.py::test_slate_*_cpu` (1×1 CPU
mesh, skipif-clean without the host lib) + the CLI matrix under
`JAX_PLATFORMS=cpu` for multi-rank CPU meshes.  The first additional
host backend landed the same way: `ffi.scalapack` (Cray LibSci
pXgetrf+pXgetrs, `distributed_lu = scalapack`) compiles into the same
library and registration table — see `distrib_la._scalapack` (the C++ is
still `src/ffi/cpp/scalapack/`).

Dual-lib caveat (GPU nodes): both SLATE builds install `libslate.so.2`,
so when the CUDA FFI library loads first its `libslate` satisfies the
host library's own dependencies too — the in-process `*_cpu` tests on a GPU
node exercise the host HANDLERS (`fromScaLAPACK` + `Target::HostTask`)
against the cuda-built SLATE running host-side.  That is a supported
SLATE configuration, and the `gpu_backend=none` binary itself is
validated where it actually deploys — CPU nodes, where only the host
library loads (bare-metal Milan runs: 7/7 pytest, 2×2 + 4×1 CLI clean,
2026-07-10).

**THE OTHER ORDER IS NOT SUPPORTED, and it is not benign (2026-08-07).**
`libblaspp.so.2` collides the same way, and the two builds are NOT
interchangeable in that direction: the host build is `gpu_backend=none`,
so its `blas::get_device_count()` is a compiled-in 0.  Open the HOST
library first and every CUDA SLATE handler in the process refuses —

    FAILED_PRECONDITION: slate.potrf: blas::get_device_count()=0 but JAX
    one-process-per-GPU model requires exactly 1.

— with exactly one visible GPU, because the count came from a library
compiled without CUDA at all.  Measured with `dladdr` in both legs of the
distrib_la suite (`_reports_step4/discrim_{marker,svc}.txt`): the failing
leg resolved `blas::get_device_count` into
`slate_builds/cpu/install/lib64/libblaspp.so.2` and read 0; the green leg
resolved it into `slate/install/lib64/libblaspp.so.2` and read 1.  What
lost the race was a MODULE-SCOPE `probe_target(..., "cpu")` at pytest
collection, so this is a production hazard too: any GPU run whose first
FFI touch is a host target gets a device-less SLATE.

Both loaders therefore open **CUDA before cpu**
(`distrib_la.loader._open_cuda_before_host`, and the same rule in
`src/ffi/common/ffi_loader.py`).  `test_so_acceptance.py::test_check_5_*`
is the ratchet on the premise: it fails the day the two stacks stop
sharing a SONAME, which is the day to delete the rule.

Tests: `services/distrib_la/tests/test_distrib_la_contract.py` (the
`slate_*` cells, and the SONAME-race section) and
`test_distrib_la_multiproc.py --mesh 2x2`; run via `lx run` / `lx test`
inside an allocation.  The benches are `services/distrib_la/bench/`.

### Batched variant — `_slate.py`, `batched_{potrf,trsm}_ffi.cc`

For the GWJAX `(Nq, Nmu, Nmu)` workload, `batched_distributed_cholesky`
and `batched_distributed_trsm` take a 3-D input sharded
`P('x', None, 'y')`: batch across `'x'`, each `Nmu × Nmu` across `'y'`.
Each X-row of the mesh gets its own MPI sub-comm of size `Py`
(`MPI_Comm_split` by `x_rank`), and the FFI handler loops over the
per-rank batch calling `slate::potrf` / `slate::trsm` on each slice.

SLATE has no native batched potrf, so the "batching" is literally a
C++ for-loop — but the sub-comm setup is shared across iterations,
which is the only bit that matters for amortising Python↔XLA dispatch
overhead. See the `batched_distributed_cholesky` and
`batched_distributed_trsm` docstrings in `distrib_la._slate` for the full
shape contract and `test_distrib_la_multiproc.py` for the real 4-rank
correctness gate.

## References

- **SLATE Users' Guide** (SWAN-010):
  https://icl.utk.edu/files/publications/2020/icl-utk-1390-2020.pdf
  — chapter 4.1 on tile layout, chapter 4.5 on the C++ matrix API.
- SLATE source: `sources/SLATE` (cloned from
  https://github.com/icl-utk-edu/slate).
- NERSC GPU-aware MPI:
  https://docs.nersc.gov/development/programming-models/mpi/cray-mpich/
- NERSC GPU affinity (`CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`):
  https://docs.nersc.gov/jobs/affinity/
