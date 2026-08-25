# Uniform gauge vertices and contact

`psp.dft_operators.static_gauge_vertices_matrix_k` is the single public
operator boundary for the static, spatially uniform electromagnetic
derivatives of LORRAX's plane-wave norm-conserving DFT Hamiltonian.  It returns

\[
\Gamma_a = +{\partial H\over\partial K_a},\qquad
\Lambda_{ab}=+{\partial^2 H\over\partial K_a\partial K_b},
\]

where Cartesian `K` is in `1/bohr`.  `Gamma` has units `Ry*bohr` and `Lambda`
has units `Ry*bohr^2`.  These are neutral Hamiltonian derivatives: neither an
electron-charge sign nor a factor of `1/c` is included.  The caller applies
the factors implied by its one, explicitly stated minimal-coupling convention.

The kinetic contribution to the first derivative is `2(K+G)`, and the
nonlocal contribution uses the existing positive `dV_NL/dK` convention.  The
contact is

\[
\Lambda^{\rm kin}_{ab,mn}=2\delta_{ab}\langle\psi_m|\psi_n\rangle
  + {\partial^2(V_{\rm NL})_{mn}\over\partial K_a\partial K_b}.
\]

The beta projectors and their first derivative remain owned by
`build_Z_and_dZ`; its exact spelling is differentiated once to obtain the
second beta derivative.  A band block is contracted with beta, `d beta`, and
`d2 beta` once, producing only the low-rank coefficients
`c=<beta|psi>`, `dc`, and `d2c`.  `V_NL`, its current, and its contact then
close as `c^dag E c` and its two product-rule derivatives entirely in
coefficient space.  No band-by-G derivative wavefunction is materialized.
The same Kleinman--Bylander projectors and spinor `E` matrices, including
their spin-orbit coupling, therefore own the Hamiltonian, current, and
contact.  `E` is k-independent in this canonical norm-conserving
pseudopotential representation, so there are no `dE` terms.  Re-expansion
through beta onto G space remains confined to the existing apply-to-ket
operator door and `psp.vnl_ops.apply_vnl_derivatives_to_ket`, used only when
a Sternheimer action genuinely requires the first- or second-derivative
G-space action.  Local multiplicative `V_scf` is invariant under a uniform
vector potential and has no derivative here.

These are operator derivatives at fixed `psi_G`.  The k dependence of an
eigenstate or occupied projector is response-state drift and belongs to the
existing Sternheimer/covariant-response layer; it must not be folded into
`Gamma` or `Lambda` a second time.

## Scope boundary

Uniform `A` is equivalent to a momentum shift and is independent of a path
between the endpoints of a nonlocal kernel.  At finite wavevector that is no
longer true: gauging `V_NL(r,r')` requires one selected Wilson-line path, whose
first and second functional derivatives define the nonlocal current and
contact.  The project has not bound such a path.  Passing nonzero
`q_cart_bohr_inv` therefore refuses with `GATE EM-VERTEX-FINITE-Q-WILSON`.
Endpoint-averaged velocities and a raw projected Dirac-alpha vertex are not
accepted substitutes because they do not define the contact or guarantee the
finite-q Ward identity.

The coefficient and returned matrix arrays remain on their JAX devices, with
the band axis kept last in each coefficient block so existing band-block
sharding is preserved.  A low-memory consumer can persist or stream the
result of `vnl_projector_coefficients_k` and pass it back to
`static_gauge_vertices_matrix_k` without repeating any G contraction.  The
owner neither loads a WFN nor performs an FFT, symmetry operation, gather, or
file write.  Inputs must use the canonical paired `k`/`G` representation, with
padded wavefunction columns already zero.
