# `ffi.slate` — distributed dense linear algebra on GPUs from JAX

JAX FFI wrappers around [SLATE](https://icl.utk.edu/slate/) (a tile-based
MPI + GPU dense linear algebra library from ICL).  Currently exposes:

- `distributed_cholesky(A, mesh)` → `SlateLowerL`  (`slate::potrf`)
- `distributed_trsm(A_or_handle, B, mesh, ...)`   (`slate::trsm`)
- `distributed_eigh(A, mesh)` → `(W, Q)`           (`slate::heev`; eigvals
  are good, eigvecs have an unresolved layout artifact — see eigh.py)

cholesky and trsm hit machine-precision residuals (~1–3e-16) on **any
p × q mesh** where `p * q == jax.process_count()`.

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

Smoke tests / benches (require `LORRAX_SELECT_GPU=1`,
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`):

- `python3 -m common.slate_cholesky_trsm_test -n 256 --dtype c128`
- `python3 -m common.slate_chol_trsm_bench --mesh 2x2 -n 1024`

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

3. **GPU-aware MPI** (`cpp/run_shifter.sh` + `cpp/in_container.sh`).
   Cray MPICH on Perlmutter does GPU-Direct RDMA over Slingshot
   (the closest equivalent to NCCL for any MPI-based library)
   only when `MPICH_GPU_SUPPORT_ENABLED=1` and `libmpi_gtl_cuda.so`
   is loaded.  shifter's `--module=mpich` explicitly *unsets* the
   env var, so `in_container.sh` re-asserts it after shifter starts.

The combination yields the standard SLATE convention (`side='L',
uplo='L', op='N'`) for the trsm of a Cholesky factor: no inverted
sides, no opaque-handle gymnastics inside the user's mental model.

## Restrictions / known gaps

- `mesh` must have axes named `('x', 'y')`.  No 3-D-mesh / sub-mesh
  selection yet (see *GWJAX adaptation* below for the workaround).
- `p * q == jax.process_count()`.  Partial-world / sub-comm calls
  aren't implemented.
- `heev` requires `p == q` (SLATE's algorithm); cholesky/trsm don't.
- `heev` eigenvectors are wrong by a layout transform we haven't fully
  pinned down — eigvals are correct.  Use cholesky/trsm if at all
  possible.
- Default tile size: `nb = n // max(p, q)` (overridable via `block_size=`).
  Performance is sensitive to this for large `n` — sweep if it matters.
- 3 local transposes per cholesky+trsm chain.  At `n ~ 1k–4k` on
  4 A100s the chain runs ~150–500 ms; below ~1k the transposes are a
  visible fraction of total time.

## GWJAX adaptation: batched `(Nq, Nmu, Nmu)` cholesky+trsm

User scenario: a batch of `Nmu × Nmu` Hermitian PD matrices, depth
`Nq` (1–8000).  Each `Nmu × Nmu` is too big for one GPU, so each must
be 2-D-sharded; multiple batch elements should be processed in
parallel across the available GPUs.

Recommended pattern:

1. Build a JAX 3-D mesh `('batch', 'x', 'y')` of shape
   `(B, Px, Py)` where `B * Px * Py == jax.process_count()`.
   Choose `Px, Py` so a single `Nmu × Nmu` matrix tiles cleanly across
   the `(x, y)` sub-mesh; pick `B` so `Nq / B` is a manageable per-rank
   batch.

2. Reshard the batch from `(q, mu_X, nu_Y)` to `(q_BATCH, mu, nu_Y)`
   (one all-to-all up front) so each `'batch'`-row of the mesh owns
   `Nq/B` matrices and the inner `(x, y)` sub-mesh shards each matrix.

3. Loop in Python over the per-rank batch, calling
   `distributed_cholesky` / `distributed_trsm` for each matrix using a
   2-D sub-mesh restricted to the `(x, y)` axes.  **Today this requires
   a sub-mesh API the FFI doesn't expose** — the validator rejects
   meshes that don't cover the full process count.  See *Future work*.

4. Reshard outputs back to whatever layout the rest of GWJAX wants.

For `Nq = 8000` and per-call wall ≈ 100 ms, total ≈ 800 s on one
batch row — divide by `B` to parallelise across rows.  This dominates
unless we add a batched FFI primitive (see Future work).

## Future work

- **Sub-mesh / partial-world support.**  Allow
  `distributed_cholesky(A, mesh=sub_mesh_of_('x','y'))` where
  `sub_mesh_of_(...).size < jax.process_count()`.  Each unique sub-mesh
  would have its own `MPI_Comm_split`'d sub-comm and its own
  `SlateCtx`.  Required for the GWJAX batched pattern above to actually
  parallelise across the `'batch'` mesh axis.
- **Batched primitives.**  Single FFI call accepting a `(Nq, Nmu, Nmu)`
  input that loops potrf internally on each batch slice.  Amortises the
  Python ↔ XLA dispatch overhead (~50 ms per call right now); could
  also fuse with the in-place trsm to avoid the L-handle round-trip.
- **`heev` eigenvectors.**  See `eigh.py` for the layout artifact;
  unresolved.
- **`gels` / least-squares.**  Useful for the ISDF fitting paths
  (separate ticket).

## Files

- `cholesky.py` — `distributed_cholesky` + `SlateLowerL` handle.
- `trsm.py`     — `distributed_trsm` (handle + plain-array paths).
- `eigh.py`     — `distributed_eigh`.
- `context.py`  — per-(p,q) `MPI_Comm` + `SlateCtx` cache + `validate_mesh`.
- `cpp/ctx.h`           — `SlateCtx` struct.
- `cpp/context.cc`      — `lrx_slate_context_create/destroy/init_mpi`.
- `cpp/{eigh,potrf,trsm}_ffi.cc` — XLA FFI handlers per op.
- `scripts/stage_cray.sh` — populate `/pscratch/.../lorrax_slate_cray/stage`
  with libsci + libmpi_gtl_cuda + xpmem + lustreapi (run once per
  module update).

Build: `bash src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh`.
Tests under `src/common/slate_*_test.py` and `slate_*_bench.py`.

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
