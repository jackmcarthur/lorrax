# Planned Davidson

`solvers.plan_davidson` builds a complex128 block Davidson solver with fixed
capacity and explicit operator data; `plan_local_davidson` is the same solver
on one device. The whole solve is one compiled `lax.while_loop`, and different
k-points are data arguments to the same executable. Its native linear algebra
enters through [`distrib_la.plan_subspace`](#shared-subspace-service). The host
API `solvers.davidson.davidson` delegates to the same implementation.

## API

```python
from solvers import plan_local_davidson
from solvers.davidson_fixed import CONVERGED

# Both callbacks are traceable; data is an array pytree, not captured H data.
# apply_h(data, vectors) -> vectors
# precondition(data, residuals, eigenvalues, vectors) -> corrections
plan = plan_local_davidson(
    apply_h, precondition,
    n_eig=128, capacity=1280, vector_shape=(nspinor, ngkmax),
)
print(plan.workspace_specs)
executable = plan.solve.lower(data, initial, 1e-8, 100).compile()
print(executable.memory_analysis())
values, vectors, info = executable(data, initial, 1e-8, 100)
assert int(info.status) == CONVERGED
```

* **Shapes.** `initial` and the returned vectors are `(n_eig, *vector_shape)`
  (row eigenvectors). `1 ≤ n_eig ≤ prod(vector_shape)` and
  `capacity ≥ 2·n_eig` are required and never adjusted. `solve(data,
  initial, tolerance=1e-8, max_iterations=100, stall_patience=0)`; tolerance,
  iteration budget and stall patience are dynamic, not compilation keys.
* **Reuse.** Build the plan and executable once and reuse them across
  same-shape Hamiltonians; a new plan inside a k-point loop recompiles.
  `lower` accepts `jax.ShapeDtypeStruct` leaves, so the buffer schedule can be
  inspected before the arrays exist.
* **Operator.** `apply_h` must accept full `n_eig` blocks and power-of-two
  tail widths; only retained corrections are applied. The preconditioner
  receives a full residual block. For independent per-k solves, callbacks
  must not contain inter-rank collectives, because solves stop at different
  iterations.
* **Distributed vectors.** Pass
  `vector_sharding=NamedSharding(mesh, P(None, "x", "y", None))` to
  `plan_davidson`: the first axis is the replicated subspace row axis, and
  every persistent buffer and conditional output keeps the trailing vector
  layout. Explicit shard maps flatten only each rank's local tile; CGS2 uses
  two batched coefficient reductions and never gathers a vector. Incremental
  projection reduces only new matrix entries. `tests/test_davidson_planned.py`
  fails on any synchronous or asynchronous all-gather in the compiled HLO.
* **Host API.** Pass `data=payload` and callbacks that take the payload first
  when arrays are distributed; JAX cannot close over non-addressable arrays.
  `solvers.davidson.LAST_RUN` holds one final snapshot with its status.

**Result.** `DavidsonInfo` holds `status`, `iterations`, `matvecs` (including
the initial vectors), `restarts`, `active_size` and per-root `residuals`.
`CONVERGED` means every residual norm is below
`tolerance·max(1, |eigenvalue|)`. The other statuses are `ITERATION_LIMIT`,
`NO_DIRECTIONS`, `BAD_INITIAL`, `NONFINITE`, `BAD_TOLERANCE` and `STALLED`
(`stall_patience > 0` iterations without a 1% reduction of the largest
residual). Inspect the status on every return: a bad initial block or a zero
iteration budget does not return certified Ritz pairs.

## Memory and active work

The two persistent vector buffers (basis and images) hold
`2·capacity·prod(vector_shape)` complex128 elements; the projected matrix is
`capacity²` and the coefficient block `capacity·n_eig`. `workspace_specs`
lists these plus the subspace provider's native scratch before lowering.
`memory_analysis()` reports the compiled XLA schedule, including XLA-visible
callback temporaries; CUDA context and handle memory, provider caches,
allocator reservations, arrays outside the executable and private allocations
inside an operator callback are not in it.

Only active work is done. Projection updates only new rows and columns; CGS2
takes an active count and window start; Ritz reconstruction uses the active
prefix; a partial H block is decomposed into exact power-of-two pieces. Rank
discovery uses the relative Gram cutoff in `solvers/subspace_numerics.py`, and
normalization touches only retained directions. No projected eigensolve or
vector GEMM runs over unused capacity. Capacity-sized arrays never cross a
conditional output (projections and stores alias in place), so only small
correction blocks do. Initial reservation and zero fills are fixed-size;
iteration never grows them.

On CUDA, runtime dimensions reach cuBLAS and cuSOLVER through the provider;
BLAS scratch is explicit and eigensolver scratch is queried at planning time.
Each native call copies small dimension descriptors to the host and
synchronizes the stream, so the loop is fully staged but is not a host-free
CUDA graph. On CPU the provider uses NumPy BLAS/LAPACK callbacks: fixed
compiled shapes and active arithmetic, but host transfers and LAPACK workspace
fall outside the memory contract. There is no GPU-to-CPU fallback.

## Shared subspace service

`distrib_la.plan_subspace(*, capacity, n_eig, vector_sharding=None,
max_block_size=None, native_collectives=False)` resolves the provider for
Davidson and Lanczos before tracing: runtime-sized native BLAS/LAPACK on CUDA
(`plan_local_subspace`), host LAPACK callbacks on CPU. `max_block_size`
defaults to `n_eig`; Lanczos declares its block width when it differs from the
Ritz count. `start` and `active` arguments are interval start and **count**,
and an empty window is legal. The plan provides active Gram, reconstruction,
CGS2, projected eigensolve, aliased stores and incremental projection.

For Lanczos it also provides block TSQR: each rank factors its local tile,
exchanges only reduced R factors, and applies its small factor from the
stacked-R QR, so nearly dependent blocks never go through normal equations.
`qr_stacked_r` bounds the replicated coefficient stack; tiles shorter than the
block width contribute their reduced R height.

The CUDA provider must export the `ActiveSubspace*Ffi` handlers of the
canonical `src/ffi/cpp/CMakeLists.txt` target; a missing target refuses by name
at planning, with no padded-math or CPU fallback. The
[FFI layout](../architecture/ffi_layout.md) owns provider selection. The
[orthogonalization service](orthogonalization.md) owns CGS2, its coefficient
communication and the correction-buffer alias.

Tests: `tests/test_davidson_fixed.py`, `tests/test_davidson_planned.py` and
`services/distrib_la/tests/test_active_subspace.py` cover complex and
degenerate spectra, restarts, every failure status, changed operator data,
rank-one correction tails, poisoned inactive storage and distributed-input
refusal.
