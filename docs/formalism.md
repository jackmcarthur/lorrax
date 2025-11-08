## ISDF + GW Formalism (COHSEX focus)

This page summarizes the working equations used in this codebase. It is a condensed, renderable version of the notes in docs/misc/isdf_context.md.

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

For implementation details, see `src/isdf/gw_isdf/cohsex_jax.py` and `src/isdf/gw_isdf/w_isdf.py`.



### Self consistency
To do self consistent updates, we need to iterate the V_hartree and Sigma_GW contributions to the self energy until the wavefunctions remain unchanged.

The DFT hamiltonian is K (kinetic E) + I (ionic local + nonlocal E) + H (hartree E) + Vxc.
We use isdf.psp.kin_ion_io to write the K+I elements to file so we can update V_hartree+Sigma_GW ourselves.

We also use the output of the interpolation-point finding code (kmeans clustering step weighted by the charge density) and the dipole matrix elements from isdf.psp.get_dipole_mtxels to calculate the "head correction" to the screened interaction W. Theoretically these should probably all be done on startup of cohsex_jax but they are all kind of slow (like 10+ seconds each) and we shouldn't do that until they have been profiled and highly optimized with JAX for performance.

The self consistent iterations right now are untested (alpha only, not converging well).