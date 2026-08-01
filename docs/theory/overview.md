## ISDF + GW Formalism (COHSEX focus)

This page summarizes the working equations used in this codebase. It is a condensed, renderable version of the notes in docs/dev/archive/isdf_context.md.

### Wavefunctions and notation

We work with spinor wavefunctions on a real-space grid r and k-points k:

- psi_nk(r): band n, k-point k, and spinor index s (suppressed when clear)
- cnk(G): plane-wave coefficients on FFT grid G
- FFTs connect cnk(G) ↔ psi_nk(r)

### ISDF basis selection

1. Build charge density from a chosen band window (valence + some conduction)
2. Select interpolation points r_mu via k-means/CVT over the density

### Density-product approximation

For each crystal momentum transfer q, approximate the density product:

rho_q(mnk, r) = psi*_{m,k−q}(r) psi_{n,k}(r) ≈ sum_mu zeta_{q,mu}(r) psi*_{m,k−q}(r_mu) psi_{n,k}(r_mu)

We solve a least-squares system for zeta_{q,mu}(r):

Z_q(r_mu, r) = sum_k P*_{k−q}(r_mu, r) P_k(r_mu, r)
C_q(r_mu, r_nu) = sum_k P*_{k−q}(r_mu, r_nu) P_k(r_mu, r_nu)
C_q zeta_q = Z_q

where P_k(r, r_mu) = sum_n psi_{n,k}(r) psi_{n,k}(r_mu). Spinor structure is carried in psi but zeta_q is spin-independent.

### Coulomb matrix elements in the ISDF basis

Define z_q,mu(r) = e^{−i q·r} zeta_{q,mu}(r). In reciprocal space:

V_{q,mu,nu} = sum_G z*_{q,mu}(G) v_q(G) z_{q,nu}(G)

with v_q(G) the Coulomb kernel (truncated in 2D; q=G=0 handled analytically/MC per BerkeleyGW conventions).

### Green’s function and COHSEX

At t=0 (static limit), we build G from occupied and unoccupied subspaces. In practice we use psi evaluated on r_mu and zeta-derived V_mu,nu on the k-grid, transform to mixed R-space as needed, and form

sigma_X ∝ G ∘ V  (element-wise in the mixed representation),

then project back to band space Sigma_kij by contracting with psi. Screened exchange (SX) and Coulomb-hole (COH) can be obtained if W replaces V (via chi0).

### Practical notes

- All large arrays are sharded over devices using JAX NamedSharding and PartitionSpec
- Wavefunction FFTs are performed once per window, then reduced to psi(r_mu)
- q-loops materialize only the minimal data needed (zeta for one q at a time)

For implementation details, see `src/gw/gw_jax.py` and `src/gw/w_isdf.py`.



### Self consistency (updated 2026-07-31)

Self-consistent updates iterate the V_Hartree and Sigma_GW contributions until the quasiparticle Hamiltonian is stationary. The DFT Hamiltonian is K (kinetic) + I (ionic local + nonlocal) + H (Hartree) + Vxc; `gw.kin_ion_io` writes the K+I elements (and, by default, the exact V_H) to file so the run can rebuild V_Hartree + Sigma_GW itself, and `psp.get_dipole_mtxels` supplies the dipole matrix elements for the q→0 head correction to W.

QSGW (`qp_solver = self_consistent`, `gw_config.QPSolver`) is wired for all compute modes via the mode-agnostic sigma dispatch and verified end-to-end; the loop knobs (`sc_max_iter`, `sc_tol_ev`, rCROP/linear acceleration) are deck keys. See manual §4.5 for the solver family.