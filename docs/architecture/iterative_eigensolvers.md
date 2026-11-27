# Iterative eigensolvers

LORRAX finds a few extremal eigenpairs of large operators it never stores as
matrices: the plane-wave Kohn–Sham Hamiltonian of the NSCF step (Davidson) and
the BSE Hamiltonian (Davidson, Lanczos, and for the full problem the
[structure-preserving Davidson](#structure-preserving-davidson)). This page
describes `src/solvers/`'s planned
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
`capacity ≥ min(2·n_eig, prod(vector_shape))`. Storage capacity is capped at
the declared vector-space dimension, including any padding: more independent
vectors cannot exist. This matters for near/full-spectrum solves, where an
unreachable larger capacity also enlarges the ROCm capacity eigensolve.
Corrections are limited to the available basis slots; an exhausted full space
that still fails its true-residual test returns `NO_DIRECTIONS`, not convergence.

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

## Structure-preserving Davidson {#structure-preserving-davidson}

`solvers.bse_sp_davidson.sp_davidson(apply_pair, X0, *, precondition=None,
data=(), n_converge=None, capacity=None, max_iter=100, tol=1e-8,
stall_patience=20, lock_converged=True, margin_lock=1.0,
on_indefinite="refuse", metric_tol=1e-8, refusal_notes=None, plan_key=None, verbose=True)`
finds the lowest roots of the full (non-Tamm–Dancoff) Bethe–Salpeter problem

$$H\begin{pmatrix}X\\Y\end{pmatrix}=\omega\begin{pmatrix}X\\Y\end{pmatrix},\qquad
H=\begin{pmatrix}A&B\\-\bar B&-\bar A\end{pmatrix},\qquad A=A^\dagger,\quad B=B^{\mathsf T},$$

with the pairs normalised $X^\dagger X-Y^\dagger Y=1$. It is the solver behind
`full_bse_solver = davidson` ([BSE](bse.md#solvers)). Its only access to the
operator is the real-linear pair applier (`bse_stack_matvec.build_bse_stack_pair_matvec`),

$$M(x)=Ax+B\bar x=\mathrm{pair}(x,+1),\qquad \mathrm{pair}(x,-1)=-\mathrm i\,M(\mathrm i x), \tag{1}$$

so only the sign $s=+1$ is ever applied. One trial vector through `pair` is
one **pair call**.

**The method.** The basis vectors $w_j\in\mathbb C^N$ are orthonormal in the
real inner product, $\mathrm{Re}(W^\dagger W)=I$, and the trial space is
$\mathrm{span}_{\mathbb C}\{[w_j;\bar w_j]\}$. With
$\mathcal K=\begin{pmatrix}A&B\\\bar B&\bar A\end{pmatrix}$ and
$\Sigma=\mathrm{diag}(I,-I)$,

$$Z=\begin{pmatrix}W\\\bar W\end{pmatrix},\qquad Z^\dagger Z=2I,\quad
Z^\dagger\mathcal KZ=2M_r,\quad Z^\dagger\Sigma Z=2\mathrm iS,\qquad
M_r=\mathrm{Re}(W^\dagger MW),\quad S=\mathrm{Im}(W^\dagger W). \tag{2}$$

$M_r$ is real symmetric and $S$ real antisymmetric. The Galerkin projection
of $Hz=\omega z$ on $z=Zc$ is

$$M_r\,c=\omega\,(\mathrm iS)\,c, \tag{3}$$

solved on the host by Shao et al., Linear Algebra Appl. 488 (2016) 148,
Algorithm 1: $M_r=LL^{\mathsf T}$, the Hermitian $\mathrm iL^{-1}SL^{-\mathsf T}$
has the eigenvalues $\pm1/\omega$, and $c=L^{-\mathsf T}y/\sqrt{2/\omega}$. The
Ritz vectors, their residual and the correction of an open root are

$$X=\sum_jc_jw_j,\qquad Y=\overline{\sum_j\bar c_jw_j},\qquad
X_i^\dagger X_j-Y_i^\dagger Y_j=c_i^\dagger(2\mathrm iS)c_j=\delta_{ij}, \tag{4}$$

$$R_X=\sum_jc_j\,Mw_j-\omega X,\qquad R_Y=-\overline{\sum_j\bar c_j\,Mw_j}-\omega Y,\qquad
\rho=\frac{\big(\lVert R_X\rVert^2+\lVert R_Y\rVert^2\big)^{1/2}}{\big(\lVert X\rVert^2+\lVert Y\rVert^2\big)^{1/2}}, \tag{5}$$

$$t_X=\frac{R_X}{D-\omega+\epsilon},\qquad t_Y=-\frac{R_Y}{D+\omega+\epsilon},\qquad
a=t_X+\bar t_Y,\qquad b=-\mathrm i\,(t_X-\bar t_Y). \tag{6}$$

The residual needs no pair call. $D$ is the real diagonal of $A$: the diagonal
of $H$ is $[D;-D]$, because $A$ is Hermitian and $B$ sits in the off-diagonal
blocks. $a$ and $b$ are the two real-form components of $[t_X;t_Y]$, scaled by
one real factor per root; adding both puts the correction and its partner
$[\bar t_Y;\bar t_X]$ into the trial space for two pair calls. A root passes
when $\rho<\mathrm{tol}\cdot\max(1,|\omega|)$, the planned Davidson's rule
($\lVert z\rVert_2^2=1+2\lVert Y\rVert^2$ replaces $\lVert x\rVert=1$). When
the next block would not fit the capacity, the basis restarts on the real
form of the Ritz block,

$$Q=\mathrm{orth}_{\mathbb R}\,[\mathrm{Re}\,C,\ \mathrm{Im}\,C],\qquad
W\leftarrow Q^{\mathsf T}W,\quad MW\leftarrow Q^{\mathsf T}MW,\quad
M_r\leftarrow Q^{\mathsf T}M_rQ,\quad S\leftarrow Q^{\mathsf T}SQ, \tag{7}$$

which spans the Ritz vectors and their partners and costs no pair call.

**Real coefficients only.** $M$ is not complex-linear, so every coefficient
that multiplies a basis vector whose image is carried along is real: the
Gram–Schmidt coefficients $\mathrm{Re}(W^\dagger p)$, the whitening of the
real Gram matrix, the rotation (7). The plan's `orthogonalize` uses complex
coefficients and is not used; the module's `_real_cgs2` is `gram`, its real
part, `reconstruct` and a subtraction on the same plan. The rank rule is
`solvers.subspace_numerics._rank_whitener` on the real Gram matrix.

**What it guarantees when $\mathcal K\succ0$.** $M_r$ is the compression of
$\mathcal K$, so (3) is a Hermitian definite pencil: the Ritz values are real,
come in exact $\pm$ pairs, and the $j$-th positive one is at or above the
$j$-th exact root. The normalisation (4) holds to rounding, inside a
degenerate multiplet too. $\lambda_{\min}(M_r)\ge\lambda_{\min}(\mathcal K)$ is
the stability margin, free every iteration; it equals
$\lambda_{\min}(\mathcal K)$ once the softest mode is among the roots.
$\mathrm{Re}(P^\dagger MP)$ of each new block is symmetric exactly when
$A=A^\dagger$ and $B=B^{\mathsf T}$: its relative asymmetry above `metric_tol`
refuses (`GATE sp_davidson_operator`); the check runs on every block, the start block first.

**When $\mathcal K$ is not positive definite.** A projected $M_r$ with
$\lambda_{\min}\le10^{-13}\max_i|(M_r)_{ii}|$ is a certificate that
$\mathcal K$ is not positive definite: the reference state is unstable, and
some root is imaginary, complex, or real and not positive. `on_indefinite =
"refuse"` (the default) raises `GATE sp_davidson_definite` with
$\lambda_{\min}(M_r)$ at the iteration where the sign first showed, the
iteration, and the option that continues. That number is an upper bound of
$\lambda_{\min}(\mathcal K)$, not a converged value: a NiO state unstable by
$-72$ meV was refused at $-41$ meV in iteration 5.
`"continue"` solves on. A general (QZ) solve of the small pencil identifies
the unstable sector: complex roots, one representative with
$\mathrm{Im}\,\omega>0$, $\mathrm{Re}\,\omega\ge0$ per pair or quadruplet, and
real roots of positive type ($c^\dagger\mathrm iSc>0$) with $\omega\le0$.
A root near zero is fixed only to $\varepsilon s^2/|\omega|$, $s$ the scale of
$M_r$, so the real part of a non-real root counts as zero below a multiple of
that (`ROOT_NOISE`): an imaginary root with a rounding-negative real part is
still the representative of its pair. Degenerate representatives each take
their own partner.
Eigenvectors of roots that are not conjugate partners are $M_r$- and
$S$-orthogonal. With $K$ the real span of the sector and its partners, the
complement $\{d:K^{\mathsf T}Sd=0\}=\{d:K^{\mathsf T}M_rd=0\}$ therefore holds
the stable roots only, $M_r$ is positive definite on it, and the Cholesky
solve gives the stable roots with all of the properties above. The better
conditioned of the two descriptions is used: $M_rK$ loses
$\varepsilon/|\lambda|$ digits in a soft direction of the sector, and the
stable roots then stall at that level. Real roots of negative norm are
$\Sigma$-orthonormalised inside a multiplet (a general solve returns any basis
of it). The returned order is unstable roots, non-positive
real roots, positive roots; an unstable root has $X^\dagger X-Y^\dagger Y=0$
and is scaled to $X^\dagger X+Y^\dagger Y=1$. The restart keeps the unstable
eigenvectors and their partners.

**A host loop on planned buffers.** The planned Davidson above is one
`lax.while_loop` because its only dense step is a Hermitian eigensolve that
`distrib_la` serves on every device leg, and because NSCF runs it once per
k-point. Here the projected problem needs a real Cholesky factor, triangular
solves and, for an unstable reference state, a general complex eigensolve.
JAX has no device kernel for the last and `distrib_la` no per-leg route for
the others, so one `while_loop` would need a host callback every iteration or
a second, host-side copy of the iteration for the unstable case. The solve is
one per run, a few tens of iterations of $2n_{open}$ pair calls each. The loop
is therefore on the host: per iteration three kernel dispatches (four at a
restart), the applier's power-of-two pieces and one small host solve. Shapes
stay fixed all the same. $W$, $MW$ and every block
have a fixed capacity, the kernels are the plan's `store`, `gram` and
`reconstruct` with traced counts, so nothing recompiles as the basis grows,
and `plan_key` reuses the kernel set across solves. Every process runs the
loop on the same replicated small arrays; rank 0's host solve is taken by
every rank through one all-reduce per iteration (`_from_rank0`, the identity
in one process), so a host LAPACK that rounds differently from rank to rank
cannot split the control flow.

| step | plan operation | collectives (optimized HLO, 2×2 CPU mesh) |
|---|---|---|
| Ritz vectors, residuals, corrections (4)–(6) | `reconstruct`, row norms | two all-reduces of $(n_b,)$ norms |
| orthogonalise the $2n_b$ corrections | `gram` and `reconstruct` twice, twice; block `gram` twice | six ordered reductions: four $(M, 2n_b)$ and two $(2n_b, 2n_b)$ coefficient matrices |
| apply $M$ to the kept rows | the caller's applier, power-of-two widths | the applier's own |
| store, new panels of $M_r$ and $S$ | `store`, `gram` twice | two ordered reductions of $(M, 2n_b)$ |
| restart (7) | `reconstruct`, `store` | none |

$M$ is the capacity and $n_b$ the block (roots and margin roots). An ordered
reduction is `lxkit.ordered_psum`: the per-rank partials of a coefficient
matrix are gathered and summed in rank order. No vector is gathered and no
vector-sized array is replicated: the largest array in any kernel's optimized
HLO is a rank's own tile of the basis
(`tests/test_bse_sp_davidson_cpu.py`, CPU provider; the device legs run the
same `DistributedSubspacePlan` with their native tile kernels).

**What a return certifies.** `sp_davidson` returns `(omega, X, Y, info)`;
`X` and `Y` are `(n_b, *vector_shape)` in the layout of `X0`. `info['status']`
is `CONVERGED` only when the lowest `n_tested` roots pass (5); the other
statuses (`ITERATION_LIMIT`, `STALLED`, `NO_DIRECTIONS`) are returned, not
raised, and the caller's gate refuses on them (`bse_nontda`, `GATE
nontda_davidson_converged`). `info` also holds `pair_calls` (the
$2n_b$ of the start block included), `pair_calls_per_root`,
`root_corrections`, `restarts`, `lam_min`, `indefinite_at`, `n_unstable`,
`n_negative`, `metric_asym`, `orth_err` and the measured `sigma_err`
$=\max|X_i^\dagger X_j-Y_i^\dagger Y_j-\delta_{ij}|$. Under `"continue"` the
unstable and non-positive roots take block rows; `n_stable_tested` is the
number of positive roots certified, and a caller that needs `n_converge` of
them gives the block that many margin rows (`GATE nontda_davidson_block`).

`CONVERGED` certifies eigenpairs, not that they are the lowest. As for the
planned Davidson, a root whose eigenvector neither the start block nor the
corrections reach is not found. The BSE start block (the lowest transitions
and five random vectors) reaches every symmetry sector. One case defeats it:
an exactly diagonal operator, where a unit start vector is an exact
eigenvector and the diagonal preconditioner returns the Ritz vector itself.
There a start with fewer low-transition vectors than requested roots (a margin
below five) converged to higher eigenpairs in 47 of 432 stress solves, and in
none of 144 at the default margin of eight. The same stress run met `STALLED`
with exact roots when a multiplet was larger than the block (3 of 100), and at
$\lambda_{\min}=-10^{-12}$ of the scale, below what the pencil resolves.

**Memory.** $W$ and $MW$ hold $2M$ vectors; $M$ counts real-form vectors and
defaults to $20\,n_{converge}$ (the depth in iterations of the planned
Davidson's $10\,n_{eig}$, twice its vectors), at least six blocks. The host
holds $M_r$ and $S$, $2M^2$ reals.

**Measured, and where** (CPU, JAX 0.12, 2026-11-27; no GPU, no multi-process
run, no material). On the synthetic pencils of `tests/bse_full_oracle.py` at
240 transitions, on one device and on four emulated devices, the roots are
within $4\times10^{-14}$ eV of the dense oracle for a positive definite
$\mathcal K$; within $10^{-11}$ eV, the soft root included, with a soft mode
down to $\lambda_{\min}=10^{-11}$ Ry; and the unstable roots (one or two
imaginary pairs, a complex quadruplet, a negative real root) within
$5\times10^{-12}$ eV (`tests/test_bse_sp_davidson_cpu.py`). The soft root of
a stored matrix is itself defined only to
$\varepsilon\lVert A\rVert/2\omega_1$ ($5\times10^{-10}$ eV at
$\lambda_{\min}=10^{-9}$ Ry): the dense oracle and this solver sit that far
from the analytic value, together. On the 128-transition spinor payload
(8 roots and 8 margin roots, 0.01 meV) it takes 15 iterations and 436 pair
calls, 54.5 per root, against 207 applications of $A$, 25.9 per root, for the
planned Davidson on the Tamm–Dancoff problem (a factor 2.1), and 22.2 to 35.2
per root for the structure-preserving Lanczos of `solvers.bse_sp_lanczos`,
whose Krylov space is close to the whole window at that size
(`tests/test_bse_full_davidson_cpu.py`).

## Consumers

| solver | caller | operator |
|---|---|---|
| planned Davidson | `psp.run_nscf` ([NSCF Hamiltonian](nscf_hamiltonian.md)); `bse.bse_lanczos` (`solver = davidson`, the default; also `bse.exciton_bands` per $Q$) | $H_k$ on the G sphere; the BSE Hamiltonian |
| block Lanczos | `bse.bse_lanczos`, `bse.exciton_bands` (`solver = block_lanczos`) | the BSE Hamiltonian |
| thick-restart Lanczos | `bse.bse_lanczos` | the BSE Hamiltonian |
| structure-preserving Davidson | `bse.bse_nontda` (`full_bse = true` with `full_bse_solver = davidson`) | the full BSE pair applier $AX+B\bar X$ |

Which BSE solver each CLI flag selects is [BSE](bse.md).
