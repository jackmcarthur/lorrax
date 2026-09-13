# Planned local Davidson

`solvers.plan_local_davidson` builds a complex128 Davidson solver for one
Hamiltonian on one CUDA device. The complete solve is a single compiled
`lax.while_loop`; different k-points are explicit data arguments to that
same executable. Its native operations enter through
[`distrib_la`](distrib_la.md). The existing `solvers.davidson` API supports
host-driven and distributed callers.

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
full residual block. Neither callback may contain inter-rank collectives:
independent Hamiltonians can take different numbers of iterations. This
local route refuses distributed vector operands, also under an outer JIT.
CPU and distributed-vector Davidson remain outside this route's contract.

`DavidsonInfo` contains `status`, `iterations`, `matvecs`, `restarts`,
`active_size`, and per-root residual norms. CONVERGED means every norm is
less than `tolerance * max(1, abs(eigenvalue))`. Other statuses distinguish
an exhausted iteration budget, no independent directions, a deficient
initial block, nonfinite computation, and an invalid tolerance. Inspect the
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

Projection updates only new rows/columns. CGS2 and Ritz reconstruction use
only the active prefix. Rank discovery shares the legacy solver's relative
Gram cutoff; subsequent normalization processes only retained directions.
The H application decomposes a partial block into exact power-of-two pieces.
There is no projected eigensolve or vector GEMM over the unused capacity.

Capacity arrays never cross a conditional output: XLA generated full-array
copies even for identity branches in the initial implementation. The provider
now exposes aliased active-range stores, and projections alias their projected
matrix. Only small correction blocks cross conditionals. Initial reservation
and zero fills remain fixed-size allocations; iteration does not grow them.

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
