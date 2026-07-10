# `ffi.slate` — distributed dense linear algebra on GPUs from JAX

JAX FFI wrappers around [SLATE](https://icl.utk.edu/slate/) (a tile-based
MPI + GPU dense linear algebra library from ICL).  Currently exposes:

- `distributed_cholesky(A, mesh)` → `SlateLowerL`  (`slate::potrf`)
- `distributed_trsm(A_or_handle, B, mesh, ...)`   (`slate::trsm`)
- `distributed_eigh(A, mesh)` → `(W, Q)`           (`slate::heev`; W AND
  true column eigenvectors — `A @ Q == Q @ diag(W)` at ~1e-14.  The
  historical "eigvec layout artifact" was root-caused 2026-07-10: the
  FFI read stale device tiles without `tileGetForReading` — heev's
  back-transform leaves the valid Z copy on the host — plus a missing
  local-transpose pair on top.  Fixed in eigh.py / cpp/eigh_ffi.cc.)

cholesky, trsm and eigh hit machine-precision residuals (~1e-16 /
~1e-14) on meshes with `p == q` or `q == 1` (N×1) where
`p * q == jax.process_count()` — including rectangular RHS (`m != n`)
for trsm, which used to abort the whole job.  `1×q` meshes are rejected
(SLATE-internal stride assertion — see Restrictions).

## Quick start

```python
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ffi.slate import distributed_cholesky, distributed_trsm

mesh = Mesh(np.asarray(jax.devices()).reshape(p, q), ('x', 'y'))
A    = jax.device_put(A_np, NamedSharding(mesh, P('x', 'y')))   # Hermitian PD
B    = jax.device_put(B_np, NamedSharding(mesh, P('x', 'y')))

L_handle = distributed_cholesky(A, mesh=mesh)
X_fwd    = distributed_trsm(L_handle, B, mesh=mesh, op='N')   # L X = B
X_adj    = distributed_trsm(L_handle, B, mesh=mesh, op='C')   # L^H X = B

L_jax = L_handle.to_jax_lower()   # standard row-major lower-tri L
```

Smoke tests / benches (via the LORRAX module — `module load lorrax`
sets the select_gpu / Cray MPICH / LD_PRELOAD / XLA memory flags):

```bash
lxalloc
lxrun python3 -u -m common.slate_cholesky_trsm_test -n 256 --dtype c128
lxrun python3 -u -m common.slate_batched_test --nbatch 8 -n 128 --mesh 2x2 --dtype c128
lxrun python3 -u -m common.slate_chol_trsm_bench --mesh 2x2 -n 1024
```

## Design rationale

Three coordinated pieces make SLATE play nicely with JAX-sharded data:

1. **Local transpose** (Python).  JAX is row-major; SLATE tiles are
   col-major.  Each rank locally transposes its own
   `(n/p, n/q)` shard to `(n/q, n/p)` via `shard_map` + `jnp.transpose`
   — pure local op, no inter-rank comm.  The transposed bytes are the
   original block in col-major layout, which SLATE reads correctly.

2. **MPI rank remap** (C++ `cpp/context.cc`).  SLATE's `fromDevices`
   hardcodes `GridOrder::Col` (rank of tile `(i, j)` = `i + j*p`),
   while JAX's C-order mesh reshape puts shard `(mx, my)` on rank
   `mx*q + my`.  These don't agree for `p != 1, q != 1`, so we
   `MPI_Comm_split` with key `= (jax_rank / q) + (jax_rank % q) * p`
   to rebuild the comm such that SLATE's tile-to-rank assignment matches
   JAX's shard-to-rank assignment.

3. **GPU-aware MPI** (wired up by `config/modulefiles/lorrax/...lua`
   → `lxrun` / `lxshell`, build-time by `cpp/run_shifter.sh`, both via
   `cpp/in_container.sh`).  Cray MPICH on Perlmutter does GPU-Direct
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

## Restrictions / known gaps

- `mesh` must have axes named `('x', 'y')`.  No 3-D-mesh / sub-mesh
  selection yet (see *GWJAX adaptation* below for the workaround).
- `p * q == jax.process_count()`.  Partial-world / sub-comm calls
  aren't implemented.
- `p == q` or `q == 1` (N×1) — with both axes > 1 and `p != q`, the
  square SLATE tile size cannot give one tile per rank on both axes, so
  JAX's block shards and SLATE's block-cyclic tiles diverge (silent
  wrong answers).  `1×q` grids additionally hit a size-dependent SLATE
  assertion (`internal_batch.hh:290: group.ld[m] == Mij.stride()`,
  SIGABRT on all ranks): local stride `lld = n` ≠ tile size `nb = n/q`,
  and SLATE's device-region batching wants uniform strides.  `p ≥ q`
  meshes have `lld == nb` and are safe.  `validate_tile_layout`
  (context.py) rejects both classes; use the transposed (q×1) mesh.
  The batched wrappers' per-slice `(1, Py)` sub-grid is the same stride
  class — production-validated on 2×2, but the 1×4 `nbatch=8, n=128`
  assert repro documented below remains an accepted risk there.
- `heev` additionally requires `p == q` (SLATE's algorithm).
- Default tile size `nb = n // max(p, q)` is the ONLY layout-consistent
  value on multi-rank meshes (one tile per rank); `block_size=`
  overrides are rejected there and only allowed on a 1×1 mesh.  The
  same invariant is why trsm's X needed per-dimension tile sizes in the
  FFI (solve dim conforms with A's `nb`, free dim gets one tile per
  rank): the old square-`nb` X on a rectangular RHS aborted every rank
  via an uncatchable `blas::Error` from a SLATE OpenMP task (2×2) or
  silently mis-assembled B (1×4).  Fixed 2026-07-10 in
  `cpp/{trsm,batched_trsm}_ffi.cc`.
- Exceptions thrown inside SLATE's OpenMP tasks CANNOT be caught by the
  FFI handler's try/catch — they `std::terminate` all ranks.  Layout
  preconditions must be validated Python-side before invoking the FFI
  (`validate_mesh` + `validate_tile_layout` + per-wrapper shape checks).
- 3 local transposes per cholesky+trsm chain.  At `n ~ 1k–4k` on
  4 A100s the chain runs ~150–500 ms; below ~1k the transposes are a
  visible fraction of total time.

## GWJAX adaptation: batched `(Nq, Nmu, Nmu)` cholesky+trsm

User scenario: a batch of `Nmu × Nmu` Hermitian PD matrices, depth
`Nq` (1–8000).  Each `Nmu × Nmu` is too big for one GPU, so each must
be 2-D-sharded; multiple batch elements should be processed in
parallel across the available GPUs.

**Implemented** — see [`batched.py`](batched.py),
[`cpp/batched_potrf_ffi.cc`](cpp/batched_potrf_ffi.cc),
[`cpp/batched_trsm_ffi.cc`](cpp/batched_trsm_ffi.cc).  The sharding
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
  read + missing transpose pair; see `cpp/eigh_ffi.cc`.
- **`gels` / least-squares.**  Useful for the ISDF fitting paths
  (separate ticket).

## Files

- `cholesky.py` — `distributed_cholesky` + `SlateLowerL` handle.
- `trsm.py`     — `distributed_trsm` (handle + plain-array paths).
- `eigh.py`     — `distributed_eigh`.
- `batched.py`  — `batched_distributed_{cholesky,trsm}` + `SlateBatchedLowerL`.
- `context.py`  — per-(p,q) `MPI_Comm` + `SlateCtx` cache + `validate_mesh`
                  + `get_or_init_subrow_context` for the batched sub-comm.
- `cpp/ctx.h`           — `SlateCtx` struct.
- `cpp/context.cc`      — `lrx_slate_{context,subrow_context}_create/destroy/init_mpi`.
- `cpp/{eigh,potrf,trsm}_ffi.cc` — XLA FFI handlers per op (CUDA).
- `cpp/batched_{potrf,trsm}_ffi.cc` — batched variants (sub-row comm, CUDA).
- `cpp/host_ffi.cc` — host-platform variants of all five handlers
  (JAX CPU backend; compiled only into `liblorrax_ffi_host.so`).
- `scripts/stage_cray.sh` — populate `$HOME/software/lorrax_slate_cray/stage`
  with libsci + libmpi_gtl_cuda + xpmem + lustreapi (run once per
  module update).
- `scripts/build_perlmutter.sh` — reproducible SLATE builds (see below).

## Building

### 1. SLATE itself (host-side, Cray PE — not in the container)

```
bash src/ffi/slate/scripts/build_perlmutter.sh gpu    # gpu_backend=cuda
bash src/ffi/slate/scripts/build_perlmutter.sh cpu    # gpu_backend=none
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
src/ffi/slate/scripts/build_perlmutter.sh gpu`.

**Which build where**: GPU nodes use `gpu/`; CPU nodes use `cpu/`.
SLATE's execution target is a *runtime* option, and the cuda build does
run host-side (`--target t` — verified on both node types; Perlmutter
CPU nodes happen to ship `libcuda.so.1`), but the `none` build is the
config that carries to non-NVIDIA machines and never drags CUDA/GTL
into a CPU-node link.

### 2. The FFI .so (inside the container)

```
bash src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
```

Login node works when shifter cooperates; otherwise run with
`SLURM_JOBID` set to go through a compute node.  To build against a
non-default SLATE without touching the in-tree `build/` (which other
sessions may be using):

```
export LORRAX_SLATE_INSTALL_DIR=$HOME/software/slate_builds/gpu/install
export LORRAX_FFI_BUILD_DIR=$HOME/software/slate_builds/ffi_build_gpu
bash src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
# then at run time:
export LORRAX_FFI_SO=$LORRAX_FFI_BUILD_DIR/liblorrax_ffi.so   # loader override
export LORRAX_SLATE_INSTALL_DIR=...                            # LDLIB override
```

### CPU story — host platform SUPPORTED (2026-07-10)

The slate ops run on the JAX CPU backend through host handler variants
(`cpp/host_ffi.cc`): `fromScaLAPACK()` on the host buffers (same 2-D
block-cyclic layout + GridOrder::Col as `fromDevices`, so the local
transposes, comm rank-remap, and every mesh/tile validation carry
unchanged), `Target::HostTask`, plain `memcpy` staging, no stream ctx.
They compile ONLY into a separate CUDA-free library:

```
bash src/ffi/common/cpp/host/build_host.sh     # → host/build/liblorrax_ffi_host.so
```

built host-side (Cray PE, no container) against the `cpu`
(`gpu_backend=none`) SLATE install, with the XLA FFI headers staged from
the container's jaxlib (the runtime the .so must match).  The script
fails if the result links any CUDA-stack library.

`ffi_loader.py` registers the host handlers under `platform="cpu"` and
the CUDA ones under `platform="CUDA"` — same target names, so
`jax.ffi.ffi_call` sites resolve by lowering platform exactly like
jaxlib's cpu (lapack) vs CUDA (cusolver) kernel split.  Wrapper call
sites pick the library from the MESH's device platform
(`context.ensure_registered`), so slate-on-CPU-devices works even inside
a GPU-backend process.  Input-file selection: `distributed_cholesky =
slate` now passes through on the CPU backend (still never auto-picked;
still fails loudly with build pointers when the library is absent).

Tests: `tests/test_ffi_linalg_contract.py::test_slate_*_cpu` (1×1 CPU
mesh, skipif-clean without the host lib) + the CLI matrix under
`JAX_PLATFORMS=cpu` for multi-rank CPU meshes.  Future host backends
(e.g. ScaLAPACK getrf for the `distributed_lu` axis) join the same
library and registration table.

Tests under `src/common/slate_*_test.py` and `slate_*_bench.py`; run
via `lxrun` (inside an `lxalloc`-created allocation).

### Batched variant — `batched.py`, `batched_{potrf,trsm}_ffi.cc`

For the GWJAX `(Nq, Nmu, Nmu)` workload, `batched_distributed_cholesky`
and `batched_distributed_trsm` take a 3-D input sharded
`P('x', None, 'y')`: batch across `'x'`, each `Nmu × Nmu` across `'y'`.
Each X-row of the mesh gets its own MPI sub-comm of size `Py`
(`MPI_Comm_split` by `x_rank`), and the FFI handler loops over the
per-rank batch calling `slate::potrf` / `slate::trsm` on each slice.

SLATE has no native batched potrf, so the "batching" is literally a
C++ for-loop — but the sub-comm setup is shared across iterations,
which is the only bit that matters for amortising Python↔XLA dispatch
overhead.  See `batched.py` docstring for the full shape contract and
`src/common/slate_batched_test.py` for a 4-GPU 2×2 correctness test
(machine precision).

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
