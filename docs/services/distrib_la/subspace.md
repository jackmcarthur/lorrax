# distrib_la: the active subspace

Iterative eigensolvers (Davidson, Lanczos) grow a basis of vectors, project
the operator into it and diagonalize the small projected matrix. Under `jit`
every array shape is fixed at compilation, while the basis grows at run time.
This page describes the `distrib_la` plans that reconcile the two: buffers of
fixed capacity, of which only an active prefix or window is ever computed on.
It is for someone writing or debugging an iterative solver; the solvers
themselves are [Iterative eigensolvers](../../architecture/iterative_eigensolvers.md),
and the layouts and promise semantics are [the API](api.md).

## The objects

A plan fixes three integers before tracing: the **capacity** $M$ (the most
vectors the basis can hold), the Ritz count `n_eig` (the eigenpairs returned)
and the maximum **block width** $b$ (the most vectors added or orthogonalized
at once; `max_block_size`, default `n_eig`). Every array below is `complex128`
and allocated once at its full size.

| object | shape | meaning |
|---|---|---|
| basis $V$ | `(M, *vector_shape)` | basis vectors as rows; rows past the active count are never read |
| images $HV$ | `(M, *vector_shape)` | the operator applied to each basis row |
| projected matrix $H$ | `(M, M)` | $H_{ij} = \langle v_i \vert H \vert v_j\rangle$ on the active block |
| block $P$ | `(b', *vector_shape)`, $b' \le b$ | new directions (corrections, Krylov block) |
| coefficients $C$ | `(M, b')` | overlaps $\bar V P^{\mathsf T}$, or Ritz coefficients `(M, n_eig)` |

The run-time state is a pair of integer scalars, `start` and `count` (or the
active size `active`), naming the rows in use. They are traced values, so
changing them never recompiles; changing $M$, $b$ or `vector_shape` does.
`vector_shape` is the caller's: `(n,)` for a flat vector, `(n_spinor,
n_G)` for a plane-wave band, `(n_c, n_v, n_k)` for a BSE trial vector. When the
vectors are distributed, `vector_sharding` is a `NamedSharding` whose first
(row) axis is unsharded and whose vector axes cover every mesh axis of extent
above 1, so no vector is ever held whole by fewer than all ranks.

## Planning

```python
from distrib_la import plan_subspace

plan = plan_subspace(capacity=M, n_eig=k, vector_sharding=sharding,
                     max_block_size=b, native_collectives=False)
print(plan.workspace_specs)          # scratch the plan adds, before lowering
```

`plan_subspace(*, capacity, n_eig, vector_sharding=None, max_block_size=None,
native_collectives=False)` checks `1 ≤ n_eig ≤ capacity < 2³¹`, JAX x64 and the
sharding rule above, then resolves one provider from the local device
platform:

| provider | when | how the operations run |
|---|---|---|
| `LocalSubspacePlan` | a GPU process | the `lorrax_active_subspace_*` handlers: cuBLAS and cuSolverDn on the active dimensions only |
| `CpuSubspacePlan` | a CPU process | NumPy BLAS/LAPACK inside `jax.pure_callback` on the active rows |
| `DistributedSubspacePlan` | `vector_sharding` over more than one device | wraps either provider in `shard_map`: each rank works on its own vector tile and only coefficient matrices are summed across the mesh |

`plan_local_subspace(*, capacity, n_eig, max_block_size=None)` is the
one-device CUDA plan alone; it refuses on a CPU process and when any of the
six handlers (`eigh`, `project`, `reconstruct`, `ortho`, `store`, `gram`) is
missing, naming it. There is no CPU fallback for a GPU process and no
padded-math fallback for a missing handler, because either would hide a cost
the caller planned around.

## The operations

Each operation touches only the active rows, so cost follows the basis size
in use, not the capacity.

| method | computes | data that moves between ranks |
|---|---|---|
| `store(V, HV, P, HP, start, count)` | writes `count` new rows at `start`, in place (input/output aliased) | none |
| `project(V, HV, active, H, start, count)` | the new rows and columns of $H$ against the first `active` rows | the new panel only, `psum`'d |
| `gram(V, P, active, start=)` | $C = \bar V_{[start, start+active)} P^{\mathsf T}$ | `(M, b')` coefficients, `psum`'d |
| `eigh(H, active)` | the lowest `n_eig` eigenpairs of the leading `active × active` block; missing values are `+∞`, unused columns zero | none (replicated input) |
| `reconstruct(V, HV, C, active, template, columns=, compute_image=, start=)` | $X = C^{\mathsf T} V$ and $HX = C^{\mathsf T} HV$ over the active rows | none |
| `orthogonalize(V, P, active, start=)` | CGS2 of $P$ against the selected rows (below) | coefficients only |
| `qr(P)` | reduced QR of a block (TSQR when distributed) | small R factors |
| `normalize(P, rank)` | whitens the first `rank` rows of a correction block by a second Gram | through `project`/`eigh` |

`project` reduces only the new panel because $H$ is already replicated:
re-summing the old block would multiply it by the rank count.

**TSQR.** The distributed `qr` factors each rank's local tile, gathers only
the reduced $R$ factors (`(P·b, b)`, the `qr_stacked_r` scratch), takes the QR
of that small stack and applies each rank's slice of its $Q$ to the local
factor. Nearly dependent blocks therefore never pass through normal equations,
which would square their condition number. A tile with fewer entries than the
block width contributes its reduced height.

## CGS2 orthogonalization {#cgs2}

$$
C = \bar Q\,P^{\mathsf T}, \qquad P \leftarrow P - C^{\mathsf T} Q,
$$

applied twice, with $Q$ the selected basis rows $[start, start+count)$ and the
vector axes flattened. Two passes bound the projection round-off; they cannot
repair a basis that is not orthonormal or a different inner product (the
symplectic BSE orthogonalization is a separate operation). The operation does
not normalize $P$, detect its rank, or orthogonalize vectors within the block;
`qr` and `normalize` do that. An empty interval returns $P$ unchanged, and
inactive basis rows may hold NaNs.

`plan_subspace(...).orthogonalize` is one route. For a caller that needs CGS2
without an eigensolver, `plan_orthogonalization(*, capacity, max_block_size,
vector_sharding=None, native_collectives=False)` returns the callable alone,
with no eigensolver workspace:

```python
from distrib_la import plan_orthogonalization

ortho = plan_orthogonalization(capacity=512, max_block_size=32,
                               vector_sharding=sharding)
# basis (512, *vector_shape), block (b ≤ 32, *vector_shape), complex128
block = ortho(basis, block, count, start=start)   # start, count may be traced
```

Operands must be `complex128` with `basis.shape[0] == capacity`, matching
vector shapes and `1 ≤ block.shape[0] ≤ max_block_size`. `start` and `count`
must agree across ranks, and every rank must issue the calls in the same
order. `ortho.backend` names the resolved implementation:

| `backend` | when | mechanism |
|---|---|---|
| `cuda` | one GPU | local cuBLAS CGS2, block input aliased to the output |
| `cuda_jax_collectives` | distributed GPU (default) | local Gram; an in-place subtraction fused with the next Gram; a final in-place subtraction; two JAX `psum`s of the fixed `capacity × b` coefficients |
| `cuda_nccl` | distributed GPU, `native_collectives=True`, one GPU per process on the whole `('x','y')` mesh | one native handler updates the block in place and all-reduces only the `count × b` active coefficients per pass; a fixed-size exchange first checks that every rank holds the same interval |
| `jax_collectives` | distributed CPU | host callbacks on local tiles, capacity-sized JAX coefficient reductions |
| `cpu_callback` | one CPU device | both passes in one NumPy callback |

`native_collectives=True` is opt-in because the NCCL context it reuses (the
service's cuSOLVERMp context) has its own setup time and device memory;
planning with it is collective, and its calls must not come from concurrent
host threads. Each route probes its handlers at planning
(`lorrax_active_subspace_ortho`; `_gram`, `_subtract`, `_subtract_gram`;
`_distributed_ortho`) and refuses by name when one is missing.

## Memory and synchronization

`workspace_specs` lists the XLA-owned scratch a plan adds, so a caller can
admit it before lowering:

| entry | shape | purpose |
|---|---|---|
| `eigh` | `M² + lwork` complex128 | the projected eigensolve; `lwork` is cuSolverDn's query at planning |
| `eigenvalues`, `info` | `M` float64, one int32 | eigensolve outputs |
| `gram` | `(M, b)` complex128 | coefficients |
| `blas` | 4 MiB | cuBLAS workspace bound to each call's stream |
| `qr_r`, `qr_stacked_r` | `(b, b)`, `(P·b, b)` complex128 | QR factors |
| `orthogonalization_ranges` | `2P` int32 | the `cuda_nccl` interval check |

Native context allocations, CPU callback temporaries and host LAPACK
workspace are not in it; read the compiled schedule from
`memory_analysis()`. An undonated block must stay unchanged, so XLA may copy
it; a surrounding `jit` can donate it when the caller no longer needs it.

Every CUDA operation copies its small `start`/`count` descriptor to the host
and synchronizes the stream before calling cuBLAS or cuSolverDn with those
dimensions. A solver loop is therefore fully staged in one executable but is
not a host-free CUDA graph. On CPU the callbacks give fixed compiled shapes and
active arithmetic, but their host transfers and LAPACK workspace are outside
the memory contract.

## Consumers

Planned Davidson (`solvers.davidson_fixed`, used by `psp.run_nscf`), block
and single-vector Lanczos (`solvers.lanczos`) and thick-restart Lanczos
(`solvers.thick_restart_lanczos`), used by `bse.bse_lanczos` and
`bse.exciton_bands`, all take their plan from `plan_subspace`
([Iterative eigensolvers](../../architecture/iterative_eigensolvers.md)).
