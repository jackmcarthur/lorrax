# The NSCF Hamiltonian

`psp.run_nscf` computes Kohn–Sham bands on a k-grid from a converged QE charge
density and writes them as a `WFN.h5`, optionally with pseudobands. This page
describes how it applies the plane-wave Hamiltonian $H_k$ inside its Davidson
solve: the compact nonlocal-pseudopotential coupling, the batched application
and the per-k schedule. It is for a developer changing the NSCF step or the
Hamiltonian operators in `src/psp/`. The Davidson solver itself is
[Iterative eigensolvers](iterative_eigensolvers.md); producing the QE inputs
is [Inputs from DFT](../preprocessing.md).

## The operator

At one k-point the Hamiltonian acts on a block of bands
$\psi$ of shape `(n_vec, n_spinor, n_G)` on the k-dependent G sphere, padded to
the largest sphere `ngkmax` with a mask so every k has one shape:

$$
H_k\psi = T_k\psi + \mathcal F^{-1}\!\left[V_{\mathrm{scf}}(\mathbf r)\,\mathcal F\psi\right]
        + Z_k^\dagger E\,Z_k\,\psi .
$$

$T_k$ is the diagonal kinetic energy $|\mathbf k + \mathbf G|^2$; the local
potential $V_{\mathrm{scf}}$ acts on the FFT box; the nonlocal term uses the
projectors $Z_k$, shape `(R_tot, n_G)` with one row per (atom, projector) and
$R_{\mathrm{tot}}$ rows in all, and the coupling $E$, shape
`(n_spinor, n_spinor, R_tot, R_tot)`, which carries the pseudopotential's
$D$ coefficients and, with spin–orbit coupling, their spin blocks.
`psp.dft_operators.apply_H_k_from_G` applies all three to the G-sphere
representation directly.

## Compact nonlocal coupling

$E$ is block diagonal: a row couples only to rows of the same atom, and every
atom of one pseudopotential channel has the same block $E^{(c)}$, shape
`(n_spinor, n_spinor, R_c, R_c)` for that channel's `R_c` projectors. The
dense matrix spends almost all its storage and arithmetic on structural zeros.
`psp.vnl_ops.compact_vnl_coupling(setup)` builds a `CompactVNLCoupling` pytree
that stores each channel's $E^{(c)}$ once, with one contiguous span of
projector rows covering that channel's atoms:

- The action (`psp.vnl_ops.apply_projector_coupling`) reshapes each channel's
  span to `(n_atoms_c, R_c, n_spinor, n_vec)` and contracts it with $E^{(c)}$
  in one GEMM, folding atoms and vectors into one axis. There is no padding
  inside a block, no `R_tot × R_tot` operand and no per-atom launch; every
  spin and projector off-diagonal within a block is kept.
- The number of compiled coupling groups equals the number of channels, not
  atoms.
- `psp.vnl_ops.projector_coupling_diagonal(Z, coupling)` gives the matching
  spin-summed diagonal of $Z^\dagger E Z$ for the preconditioner.
- `apply_projector_coupling` is the one owner of the compact-versus-dense
  dispatch.
- Non-contiguous channel rows or an inconsistent block shape raise
  `ValueError`, because the span representation would silently couple the
  wrong rows.

`setup_H_k_from_kvec(..., compact_vnl=True)` places the compact pytree in
`HamiltonianK.vnl_E`. The default, `compact_vnl=False`, keeps the dense array
for consumers that read `vnl_E` as an array, including the Sternheimer
solvers. Compact coupling removes the structural-zero work from every
application but not from setup: `VNLSetup` still builds the dense matrix, and
the projectors and the $Z\psi$ projections are unchanged. For a small system
one dense GEMM can be faster than the per-channel blocks; measure the
crossover there.

The ionic structure-factor scan (`psp.ionic_gspace`) guards each atom's phase
evaluation with `lax.cond`, so padded atoms evaluate no full-G exponential;
the scan stays reverse-mode differentiable and the active atom count stays a
run-time operand.

## The NSCF solve

`run_nscf` builds one planned local Davidson solver
([Iterative eigensolvers](iterative_eigensolvers.md#planned-davidson)) for
`nbnd` bands with capacity `DEFAULT_M_MAX_FACTOR · nbnd = 10 · nbnd`, compiled
once for the common padded G sphere and reused at every k-point with that
k-point's Hamiltonian arrays as data; each solve runs at most 100 iterations
with stall patience 20. A non-converged solve is reported before export.

- **Batched application.** `apply_H_k_batched(..., vector_batch=32)` applies
  $H_k$ to fixed blocks of 32 vectors in a `lax.map`, then to an exact-width
  tail, which bounds the FFT-box temporaries to 32 vectors and never applies
  $H$ to a zero vector. The plane-wave initial guess uses the same operation.
- **k-point rounds.** Setup selects G-sphere shapes on the host, so every rank
  stages each round of `P` k-points in the same order, which keeps the
  compiled programs identical across ranks; each rank keeps only its own
  k-point and the ranks then solve concurrently. Setup is therefore repeated
  on every rank, and no all-k Hamiltonian or guess is cached on the device.
  Assembling the global result and writing the WFN keep their own memory
  costs.
