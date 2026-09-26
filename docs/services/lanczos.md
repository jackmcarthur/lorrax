# Planned Lanczos active algebra

`solvers.lanczos.lanczos_eig_jit`, `block_lanczos_eig_jit` and
`block_lanczos_eig_jit_converged` take a `subspace_plan` from
[`distrib_la.plan_subspace`](davidson.md#shared-subspace-service), resolved
before the recurrence is staged. Distributed callers build it with their
vector sharding before their outer `jit` and pass it in. An omitted plan
resolves the default local/CPU plan, which must not be used for distributed
vectors. `subspace_plan=False` selects the fixed-shape reference arithmetic
used by A/B tests.

## Geometry

For block width `b`, `M = max(1, min(max_iter, n // b))`. The plan must have
`capacity = (M+1)·b` (the extra block is the recurrence sentinel) and
`n_eig` equal to the requested Ritz root count; a mismatch raises. The
single-vector entry point has `b = 1`. A geometry change needs a new plan and
executable.

The basis is row-oriented `(M+1, b, *vector_shape)`, with
`vector_shape = (n,)` by default. Distributed BSE and exciton callers pass
`vector_shape = (nc, nv, nk)` with the matching X/Y sharding, so the capacity
buffer keeps both distributed axes and flattening the two Krylov axes never
transposes the whole buffer. With `structured_vectors=True`, callbacks see
`(b, *vector_shape)` blocks and Ritz vectors return as
`(n_eig, *vector_shape)`; production callers use this mode, because the flat
default may redistribute each block.

## Active work

* **QR.** TSQR on local shards, exchanging only small R factors (including
  for the initial block); no complete vector is gathered.
* **Reorthogonalization.** CGS2 over the active interval
  `[max(0, j − n_reorth)·b, (j+1)·b)`, including the current block, with two
  batched overlap reductions per iteration, so excluded old vectors and the
  unused capacity tail cost nothing. It is the only route.
* **Convergence-driven solves** build only completed projected blocks, pass
  the actual `j·b` dimension to the service eigensolver, and reconstruct from
  the completed rows only. No changing vector shape enters the iteration.
* **Native cost.** CUDA active algebra uses runtime dimensions, with small
  host synchronizations for the dimension descriptors. The CPU plan's
  behaviour is the service's; it is not active-size native BLAS.

Convergence means the lowest Ritz values stabilize at the checkpoint cadence;
it is not a certified eigenpair residual. A consumer that needs one evaluates
`H X − X Λ`.

## Thick restart

`solvers.thick_restart_lanczos.thick_restart_lanczos_eig` uses the same
service with `capacity = m_max + 1` and `n_eig = n_keep`
(`n_eig ≤ n_keep < m_max`), because restart retains `n_keep` Ritz pairs. Its
`sharding` argument is forwarded when resolving an omitted plan. Each step
orthogonalizes only the initialized prefix; restart reconstructs the retained
rows without slicing the full basis, and stale slots stay allocated but
excluded until overwritten, so no capacity-sized buffer is cleared.

Tests: `tests/test_lanczos_planned.py` covers full and partial active windows,
finite-depth subspaces compared through phase-invariant projectors, and
early-convergence eigenpairs with unused storage remaining.
