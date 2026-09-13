# Planned Lanczos active algebra

`solvers.lanczos.lanczos_eig_jit`, `block_lanczos_eig_jit`, and
`block_lanczos_eig_jit_converged` accept a `subspace_plan` from `distrib_la`.
The plan is resolved before the recurrence is staged. Distributed callers
construct it with their vector sharding before their outer JIT, then pass it
into the numerical entrypoint. An omitted plan resolves the service's default
local/CPU implementation; it must not be used to gather distributed vectors.

For block width `b`, let `M = min(max_iter, n // b)` (clamped to at least
one). The plan requires `capacity=(M+1)*b`, including the final recurrence
sentinel, and `n_eig` equal to the requested Ritz root count. The single-vector
entrypoint has `b=1`. Geometry changes require a new plan and executable.

The planned basis is row-oriented `(M+1,b,*vector_shape)`, with
`vector_shape=(n,)` by default. Distributed BSE and exciton callers pass
`vector_shape=(nc,nv,nk)` and the corresponding X/Y sharding, so the retained
capacity buffer preserves both distributed vector axes. Flattening its two
Krylov axes does not transpose the complete vector buffer at every iteration.
With `structured_vectors=True`, callbacks and returned Ritz vectors use
`(b,*vector_shape)` and `(n_eig,*vector_shape)`, respectively. The recurrence
keeps these axes throughout. Production callers use this mode; the default
flat callback/output compatibility API may redistribute current blocks.
The service QR uses local vector shards and exchanges only the small R
factors for TSQR, including the initial block. It never gathers a complete
vector for QR.
CGS2 uses the active interval
`[max(0,j-n_reorth)*b, (j+1)*b)`, including the current block, and retains
two batched overlap reductions per iteration. Partial reorthogonalization
therefore skips both old excluded vectors and the unused capacity tail.
The legacy MGS setting retains its sequential arithmetic.

Convergence-driven solves build only completed projected blocks and pass
the actual `j*b` dimension to the service eigensolver. Final reconstruction
uses only those completed basis rows. No changing vector shape enters the
iteration. CUDA active algebra uses native runtime dimensions; CPU behavior
is owned by the service and is not a claim of active-size native CPU BLAS.
As for the planned Davidson service, CUDA dimension descriptors currently
require small host synchronizations.

The convergence criterion is unchanged: stabilization of the lowest Ritz
values at the requested checkpoint cadence, not an independently certified
eigenpair residual. Consumers requiring residual certification must evaluate
`H X - X lambda`. The Hermiticity diagnostic remains unchanged.

`subspace_plan=False` is the fixed-shape reference arithmetic used by the
A/B tests. Production callers use the resolved plan. Numerical tests are in
`tests/test_lanczos_planned.py`; they distinguish full and partial active
windows, compare finite-depth subspaces with phase-invariant projectors, and
check independently constructed early-convergence eigenpairs with unused
storage remaining. GPU runtime, HLO and collective evidence are recorded by
the sandbox planned-solvers investigation, not inferred from source layout.

`solvers.thick_restart_lanczos.thick_restart_lanczos_eig` uses the same
service with `capacity=m_max+1` and `n_eig=n_keep`, since restart retains
`n_keep` Ritz pairs. Its existing `sharding` argument is forwarded when
resolving an omitted plan. Each step orthogonalizes only the initialized
prefix; restart reconstructs the retained rows without slicing the full
basis. Unused old slots remain allocated and are excluded until overwritten,
so restart does not clear a complete capacity-sized vector buffer. The
arrowhead coupling and iteration budget are unchanged.
