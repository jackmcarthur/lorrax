# Planned Davidson

`solvers.plan_davidson` builds a complex128 Davidson solver with fixed
capacity and explicit operator data. `plan_local_davidson` specializes the
same implementation to one Hamiltonian on one device. The complete solve is a single compiled
`lax.while_loop`; different k-points are explicit data arguments to that
same executable. Its native operations enter through
[`distrib_la`](distrib_la.md). The `solvers.davidson` host API delegates to this same implementation;
there is no growing-shape production Davidson loop or shape-warmup ladder.

## API

```python
from solvers import plan_local_davidson
from solvers.davidson_fixed import CONVERGED

# Both functions are traceable. data is an array pytree, not captured H data.
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

`initial` and the returned vectors have shape `(n_eig, *vector_shape)`.
`capacity >= 2*n_eig` is required and is never silently changed. Reuse the
plan and executable across same-shape Hamiltonians; constructing a new
factory inside a k-point loop defeats that reuse. `max_iterations` and the
positive scalar tolerance are dynamic arguments, not compilation keys.

The operator must accept full `n_eig` blocks and power-of-two tail widths.
Only retained corrections are sent to it. The preconditioner receives a
full residual block. For an independent local k-point solve, callbacks must not contain inter-rank
collectives: those solves can stop at different iterations. For distributed
vectors, pass `vector_sharding=NamedSharding(mesh, P(None, "x", "y", None))`
to `plan_davidson`. The first axis is the replicated subspace row axis;
trailing vector axes retain that layout in every persistent buffer. Explicit
shard maps flatten only each rank's local vector tile. CGS2 uses two batched
reductions of coefficient panels, never a vector gather. Incremental projection
reduces only new matrix entries so retained global entries are not counted P times.
Correction and normalization conditional outputs explicitly retain vector
sharding. Without these constraints, GSPMD gathered whole correction blocks
at conditional boundaries despite correctly sharded final outputs. The P4
diagonal-operator HLO regression checks synchronous and asynchronous gather
names and exercises shaped X/Y vector axes.

The CPU service uses active NumPy BLAS/LAPACK callbacks as a compatibility
implementation. It preserves fixed compiled shapes and active arithmetic, but
host transfer costs and private LAPACK workspace are not covered by the CUDA
memory/performance contract. There is no silent GPU-to-CPU fallback.

For the host convenience API, pass `data=payload` and callbacks taking that
payload first when arrays are distributed; closing over non-addressable arrays
is not supported by JAX. `LAST_RUN` records a final snapshot with explicit
status, rather than per-iteration host transfers. Native callers should use
`DavidsonInfo` directly.

`DavidsonInfo` contains `status`, `iterations`, `matvecs`, `restarts`,
`active_size`, and per-root residual norms. CONVERGED means every norm is
less than `tolerance * max(1, abs(eigenvalue))`. Other statuses distinguish
an exhausted iteration budget, no independent directions, a deficient
initial block, nonfinite computation, an invalid tolerance, and a stalled residual. The optional dynamic fifth
argument `stall_patience` stops after that many iterations without a 1%
reduction of the largest residual (zero disables it). Inspect the
status: an invalid initial block or zero iteration budget does not return
certified Ritz pairs. Matvec counts include initial vectors.

## Memory and active work

The two persistent vector buffers contain `2*capacity*prod(vector_shape)`
complex128 elements. Projected matrices and native workspaces have fixed
maximum shapes. `workspace_specs` describes persistent buffers and native
scratch before lowering; `memory_analysis()` reports the compiled XLA
buffer schedule, including XLA-visible callback temporaries. CUDA context/handle
memory, provider-owned caches or workspace, allocator reservations, and application
arrays outside this executable are not part of that XLA byte count. The generic
solver cannot declare private allocations made by an arbitrary operator callback. `lower` also
accepts `jax.ShapeDtypeStruct` leaves for the data and initial block, so the
compiled buffer schedule can be inspected before allocating those arrays.

Runtime dimensions reach cuBLAS and cuSOLVER through the existing provider
library. The BLAS scratch allocation is explicit, and eigensolver scratch is
queried at planning time and checked against the active-size requirement.
The handlers copy small dimension descriptors to the host and synchronize
the stream. Thus the Python iteration is fully staged, but native calls
still have host size synchronization; this is not a host-free CUDA graph.

Projection updates only new rows/columns. CGS2 accepts an active count and optional window start; Ritz reconstruction
uses only the active prefix. Rank discovery uses the shared relative
Gram cutoff in `solvers/subspace_numerics.py`; subsequent normalization processes only retained directions.
The H application decomposes a partial block into exact power-of-two pieces.
There is no projected eigensolve or vector GEMM over the unused capacity.

Capacity arrays never cross a conditional output: XLA generated full-array
copies even for identity branches in the initial implementation. The provider
now exposes aliased active-range stores, and projections alias their projected
matrix. Only small correction blocks cross conditionals. Initial reservation
and zero fills remain fixed-size allocations; iteration does not grow them.

Davidson uses `store_project(v, hv, p, hp, h, start, count)` to insert a block
and update its projected Hamiltonian panel together. The returned tuple is
`(v, hv, h)`. Nonempty insertions define the active prefix as `start + count`;
empty insertions preserve all three arrays. The CUDA handler reads and checks
the range once, then performs the same copies and GEMMs as the separate
operations. It still synchronizes once to read that range. This removes a
second metadata transfer and wait, without changing the GEMM dimensions or
allocating another vector buffer. The distributed service reduces only the
projection result and keeps vectors partitioned; it never reduces the already
global entries of `h` again. The CPU provider composes its existing operations.

## Provider and verification scope

Rebuild the canonical CUDA `src/ffi/cpp/CMakeLists.txt` target to obtain the
`ActiveSubspace*Ffi` handlers. An older provider refuses by name when a plan
is requested. No separate shared library, private driver binding, or silent
backend fallback is used. The provider selection follows the existing
[`FFI contract`](../architecture/ffi_layout.md).

The Perlmutter investigation is recorded in sandbox run
`runs/DEV/388_davidson_jit_20260913`, allocation `58264396`; its report owns
the timing tables and evidence paths. Focused tests live in
`tests/test_davidson_fixed.py` and
`services/distrib_la/tests/test_active_subspace.py`. They cover complex and
degenerate spectra, restarts, failure statuses, explicit changed operator
data, rank-one correction tails, poisoned inactive storage, and distributed
input refusal. The PSP benchmarks use the production Hamiltonian application;
they are solver comparisons, not a QE total-potential certification or a
certification of the complete NSCF writer/scheduler.

## Shared subspace service

`distrib_la.plan_subspace(capacity=..., n_eig=..., vector_sharding=...,
max_block_size=...)` owns provider resolution for Davidson and Lanczos.
`max_block_size` defaults to `n_eig`; Lanczos declares its correction width
separately when it differs from the requested Ritz-vector count. `start` and
`active` arguments mean interval start and **count**, including an empty window.
The provider includes active Gram, reconstruction, CGS2, projected eigensolve,
aliased store and incremental projection operations.
`store_project` combines the latter pair for consumers that immediately
project an inserted block. It requires `ActiveSubspaceStoreProjectFfi` in the
CUDA provider; standalone store and projection remain available.

The same service provides stable block TSQR for Lanczos. Each rank factors
its local vector tile, exchanges only reduced R factors, and applies its
small factor from the stacked-R QR. Nearly dependent blocks do not use
normal equations. `qr_stacked_r` bounds the replicated coefficient stack;
QR backend temporaries appear in the compiled memory schedule. Local tiles
shorter than the block width contribute their reduced R height.

Native-provider consumers must rebuild the canonical provider with this
source revision: active Gram is a new target, and the window descriptors
for reconstruction/CGS2 have changed. The provider probe refuses a missing
target; it does not silently select padded math or a CPU fallback.
