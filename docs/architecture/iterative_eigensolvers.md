# Iterative eigensolvers

LORRAX finds a few extremal eigenpairs of large operators it never stores as
matrices: the plane-wave Kohn–Sham Hamiltonian of the NSCF step (Davidson) and
the BSE Hamiltonian (Lanczos). This page describes `src/solvers/`'s planned
Davidson and Lanczos solvers: their API, their fixed-capacity storage, and the
results they certify. It is for a developer calling or changing a solver.
Read [the active subspace](../services/distrib_la/subspace.md) first: both
solvers keep their basis in its fixed-capacity buffers and do all their dense
algebra through its plans.

## Why fixed capacity

A Krylov or Davidson basis grows by a block per iteration, but under `jit`
every array shape is fixed when the program compiles. Both solvers therefore
allocate the basis once at its maximum size, the capacity $M$, and carry the
number of rows in use as a traced integer. The whole solve is one compiled
`lax.while_loop` (or `fori_loop`), so a solve at a different k-point, or of a
different Hamiltonian of the same shape, reuses the executable. Each dense
operation (projection, Gram, eigensolve, reconstruction, orthogonalization)
runs over the active rows only, so arithmetic follows the basis in use, not
$M$; the unused tail is allocated but never read or cleared.

## Planned Davidson {#planned-davidson}

`solvers.plan_davidson(apply_h, precondition, *, n_eig, capacity,
vector_shape, vector_sharding=None)` builds a `complex128` block Davidson
solver for the `n_eig` lowest eigenpairs; `plan_local_davidson` is the same
solver on one device. It requires `1 ≤ n_eig ≤ prod(vector_shape)` and
`capacity ≥ 2·n_eig` and never adjusts them.

```python
from solvers import plan_local_davidson
from solvers.davidson_fixed import CONVERGED

# apply_h(data, vectors) -> vectors
# precondition(data, residuals, eigenvalues, vectors) -> corrections
plan = plan_local_davidson(apply_h, precondition, n_eig=128, capacity=1280,
                           vector_shape=(n_spinor, n_G))
exe = plan.solve.lower(data, initial, 1e-8, 100).compile()   # once
values, vectors, info = exe(data, initial, 1e-8, 100)        # per Hamiltonian
assert int(info.status) == CONVERGED
```

- **Operator data is an argument, not a closure.** `data` is a pytree of the
  Hamiltonian's arrays, so one executable serves every k-point; a new plan
  inside a k-point loop recompiles. `lower` accepts `jax.ShapeDtypeStruct`
  leaves, so the buffer schedule can be inspected before the arrays exist.
- **Shapes.** `initial` and the returned vectors are `(n_eig, *vector_shape)`,
  one eigenvector per row. `solve(data, initial, tolerance=1e-8,
  max_iterations=100, stall_patience=0)`: tolerance, iteration budget and
  stall patience are traced, not compilation keys.
- **One iteration.** Solve the projected eigenproblem on the active block,
  reconstruct the Ritz vectors $x$ and their images $Hx$, form residuals
  $r = Hx - \lambda x$, precondition them, orthogonalize the corrections
  against the basis by CGS2, whiten them and drop dependent directions (a
  relative Gram cutoff, `solvers/subspace_numerics.py`), apply $H$ to the
  retained ones, store both and extend the projected matrix by the new rows
  and columns only. When the next block would exceed the capacity, the basis
  restarts from the current Ritz vectors.
- **The operator callback** must accept full `n_eig` blocks and power-of-two
  tail widths: a partial block of `r` corrections is applied as exact
  power-of-two pieces, so no padded vector is ever multiplied. For
  independent per-k solves the callbacks must contain no inter-rank
  collective, because solves stop at different iterations.
- **Distributed vectors.** `vector_sharding=NamedSharding(mesh, P(None, "x",
  "y", None))` keeps the row axis replicated and every vector on all ranks;
  CGS2 uses two batched coefficient reductions and no vector is gathered.
- **Host API.** `solvers.davidson.davidson` delegates to the same
  implementation. Pass `data=payload` and callbacks that take the payload
  first when arrays are distributed, since JAX cannot close over
  non-addressable arrays. `solvers.davidson.LAST_RUN` holds the final
  snapshot and its status.

**What a return certifies.** `DavidsonInfo` holds `status`, `iterations`,
`matvecs` (the initial block included), `restarts`, `active_size` and the
per-root `residuals`. `CONVERGED` means every residual norm is below
`tolerance·max(1, |λ|)`. The other statuses are `ITERATION_LIMIT`,
`NO_DIRECTIONS` (every correction was dependent), `BAD_INITIAL`, `NONFINITE`,
`BAD_TOLERANCE` and `STALLED` (`stall_patience > 0` iterations without a 1 %
reduction of the largest residual). Read the status on every return: a bad
initial block or a zero iteration budget returns values that are not
certified Ritz pairs.

**Memory.** The basis and its images hold `2·capacity·prod(vector_shape)`
`complex128` elements, the projected matrix `capacity²` and the coefficient
block `capacity·n_eig`; `plan.workspace_specs` adds the subspace provider's
scratch. Capacity-sized buffers are updated in place and never pass through a
conditional output, which would copy them; only correction-sized blocks do.
`memory_analysis()` of the compiled solve reports the XLA schedule; native
context memory, allocator reservations and allocations inside the operator
callback are outside it.

## Planned Lanczos {#planned-lanczos}

`solvers.lanczos.block_lanczos_eig_jit(matvec, n, n_eig=20, block_size=4,
max_iter=50, *, …, subspace_plan=None, vector_shape=None,
structured_vectors=False, support=None)` runs block Lanczos with block width
$b$; `block_size=1` is the single-vector solver.
`block_lanczos_eig_jit_converged(…, rtol=1e-6, atol=1e-8, check_every=4)`
runs the same recurrence in a `lax.while_loop` that stops when the lowest
`n_eig` Ritz values stop moving.

**Geometry.** With $M_b = \max(1, \min(\texttt{max\_iter}, \lfloor n_a/b\rfloor))$
block iterations, where $n_a$ is the number of supported entries, the basis
holds $(M_b + 1)$ blocks: the extra block is the residual block of the last
step. The subspace plan must therefore have `capacity = (M_b+1)·b` and
`n_eig` equal to the requested root count, or the solver refuses; a geometry
change needs a new plan. Distributed callers build the plan with their vector
sharding before their outer `jit` and pass it in; an omitted plan resolves the
local or CPU plan, which must not be used for distributed vectors.
`subspace_plan=False` selects the fixed-shape reference arithmetic used to
check the planned path.

**Vector layout.** The basis is row-oriented, `(M_b+1, b, *vector_shape)`,
with `vector_shape = (n,)` by default. The BSE callers pass `vector_shape =
(n_c, n_v, n_k)` with their X/Y sharding and `structured_vectors=True`, so the
basis keeps both distributed axes, callbacks see `(b, *vector_shape)` blocks,
and Ritz vectors return as `(n_eig, *vector_shape)`; the flat default would
redistribute every block.

**Start block.** A random complex Gaussian block, drawn only on the entries
`support` marks (the BSE passes its physical transitions) and zero elsewhere,
then orthonormalized; the supported count $n_a$ bounds the Krylov depth.

**One step.** Apply the operator to block $j$, orthogonalize the result by
CGS2 against the reorthogonalization window
$[\max(0, j - n_{\mathrm{reorth}})\,b,\ (j+1)\,b)$ (the current block
included) with two batched overlap reductions, then factor it by TSQR, which
exchanges only small $R$ factors, never a vector. `n_reorth = -1` (the
default) means the whole basis. The planned path is the only route; old
vectors outside the window and the unused capacity tail cost nothing.

**Ritz pairs.** The projected block-tridiagonal matrix is built from the
completed blocks only, its eigensolve gets the real dimension `j·b`, and the
Ritz vectors are reconstructed from the completed rows.

**What a return certifies.** Convergence of the converged variant means
$\max_i |\lambda_i - \lambda_i^{\mathrm{prev}}| < \mathrm{rtol}\cdot\max(|\lambda_i|, \mathrm{atol})$
between checks `check_every` blocks apart; it is eigenvalue stability, not an
eigenpair residual. A consumer that needs a residual evaluates
$HX - X\Lambda$ itself.

## Thick-restart Lanczos

`solvers.thick_restart_lanczos.thick_restart_lanczos_eig(…, m_max=80,
n_keep=n_eig+10, …, subspace_plan=None)` bounds memory by restarting: after
`m_max` steps it keeps `n_keep` Ritz pairs and continues from them, so a
cycle's projected matrix is an arrowhead (a diagonal block of the kept Ritz
values plus one coupling row) followed by the ordinary tridiagonal part. It
requires $0 < $ `n_eig` $\le$ `n_keep` $<$ `m_max`, and its subspace plan has
`capacity = m_max + 1` (the extra slot holds the residual vector) and
`n_eig = n_keep`. Two programs compile, one for the first cycle and one for
every restart cycle. Each step orthogonalizes only the initialized prefix;
a restart reconstructs the kept rows without slicing the basis, and stale
slots stay allocated but excluded until they are overwritten.

## Consumers

| solver | caller | operator |
|---|---|---|
| planned Davidson | `psp.run_nscf` ([NSCF Hamiltonian](nscf_hamiltonian.md)); `bse.bse_lanczos` (`--solver davidson`) | $H_k$ on the G sphere; the BSE Hamiltonian |
| block Lanczos | `bse.bse_lanczos`, `bse.exciton_bands` | the BSE Hamiltonian |
| thick-restart Lanczos | `bse.bse_lanczos` | the BSE Hamiltonian |

Which BSE solver each CLI flag selects is [BSE](bse.md).
