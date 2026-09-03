# Core ISDF and GW theory

This chapter gives the common theory used by the current static, plasmon-pole,
and multipole drivers. It stops where a focused chapter takes ownership.
Rydberg units are used internally.

## 1. Wavefunctions and transition densities

For band \(n\), crystal momentum \(\mathbf k\), and spinor component \(a\),

$$
\psi_{n\mathbf k a}(\mathbf r)
=\frac{1}{\sqrt{N_r}}\sum_{\mathbf G}
c_{n\mathbf k a}(\mathbf G)
e^{i(\mathbf k+\mathbf G)\cdot\mathbf r}.
$$

At momentum transfer \(\mathbf q\), the charge-channel transition density is

$$
\rho_{mn\mathbf k}^{\mathbf q}(\mathbf r)
=\sum_a
\psi^*_{m,\mathbf k-\mathbf q,a}(\mathbf r)
\psi_{n\mathbf k a}(\mathbf r).
$$

The spin trace is physical: Coulomb screening couples charge, while spinor
indices remain open in Green functions and self-energy until the appropriate
contraction. Bispinor response extends the channel index but does not change
the ISDF argument below.

## 2. ISDF factorization

Choose interpolation points \(\{\mathbf r_\mu\}\) from the band density and
approximate every transition density by

$$
\rho_{mn\mathbf k}^{\mathbf q}(\mathbf r)
\approx
\sum_\mu \zeta_{q\mu}(\mathbf r)
\rho_{mn\mathbf k}^{\mathbf q}(\mathbf r_\mu).
$$

The interpolation vectors are independent of the band pair. This replaces a
pair-product object indexed by \((m,n,\mathbf k,\mathbf r)\) with band values
at \(N_\mu\) points and \(N_\mu\) interpolation vectors.

Writing \(P_{\mathbf k}(\mu,\mathbf r)\) for the occupied band contraction
entering the normal equations, the Galerkin system has the schematic form

$$
C_q(\mu,\nu)
=\sum_{\mathbf k}
P^*_{\mathbf k-\mathbf q}(\mu,\nu)P_{\mathbf k}(\mu,\nu),
$$

$$
Z_q(\mu,\mathbf r)
=\sum_{\mathbf k}
P^*_{\mathbf k-\mathbf q}(\mu,\mathbf r)P_{\mathbf k}(\mu,\mathbf r),
\qquad
C_q\zeta_q=Z_q.
$$

Production forms the open-spin pair density, solves this system in bounded
real-space chunks, and accumulates the solution directly into its reciprocal
G-flat representation. The precise current algorithm is
[G-flat zeta and V](isdf-zeta-vq.md).

## 3. Coulomb interaction in the ISDF basis

Remove the Bloch phase,

$$
z_{q\mu}(\mathbf r)=e^{-i\mathbf q\cdot\mathbf r}\zeta_{q\mu}(\mathbf r),
$$

and let \(\widetilde z_{q\mu}(\mathbf G)\) be its orthonormal Fourier
transform. Then

$$
V_{q,\mu\nu}
=\sum_{\mathbf G}
\widetilde z^*_{q\mu}(\mathbf G)
v(\mathbf q+\mathbf G)
\widetilde z_{q\nu}(\mathbf G).
$$

The Coulomb kernel owns dimensional truncation and the analytic
\(\mathbf q+\mathbf G=0\) treatment. The ISDF contraction owns only the basis
change. \(V_q\) is stored as a flat-q matrix with its two centroid axes
distributed over the two-dimensional process mesh.

## 4. Polarizability and screening

The independent-particle response is

$$
\chi^0_{q,\mu\nu}(z)
=\sum_{vc\mathbf k}
M^\mu_{vc\mathbf k}(q)
M^{\nu *}_{vc\mathbf k}(q)
\left[
\frac{1}{z-\Delta_{vc\mathbf k}}
-\frac{1}{z+\Delta_{vc\mathbf k}}
\right],
$$

where \(\Delta_{vc\mathbf k}=\epsilon_{c\mathbf k}
-\epsilon_{v,\mathbf k-\mathbf q}\). Production does not materialize the
transition-pair tensor. A separable time representation builds one occupied
and one empty Green function,

$$
G_A(\mathbf k,t)
=\sum_{n\in A}
\psi_{n\mathbf k}\psi^\dagger_{n\mathbf k}
e^{-i\epsilon_{n\mathbf k}t},
$$

then contracts their lattice transforms. Static and GN-PPM rules are described
in [Minimax quadrature](minimax-quadrature.md); complex MPA samples are
described in [Multipole frequency integration](THEORY_mpa_implementation.md).

Without time reversal the two ordered particle-hole orientations of the
bracket above carry different residues — the \(+\Delta\) pole the object built
from \(\overline{\psi_v}\psi_c\), the \(-\Delta\) pole the conjugate of the
\(-\mathbf q\) forward object — so \(\chi^0_q(i\omega)\) has an anti-Hermitian,
magnetisation-odd part that vanishes only at \(\omega=0\). The production
imaginary-axis route keeps it by weighting the two orientations
independently, \(\chi^0_q(i\omega_p)=F_q+\overline{F_{-q}}\), on a
measured-broken-TR deck only; the derivation, the two-residue plasmon-pole
model it feeds and the Σ assignment are
[`docs/dev/notes/DERIVATION_gnppm_nonhermitian.md`](../dev/notes/DERIVATION_gnppm_nonhermitian.md).

Screening follows from the Dyson equation

$$
W_q(z)=\left[I-V_q\chi^0_q(z)\right]^{-1}V_q,
\qquad
W_{c,q}(z)=W_q(z)-V_q.
$$

The local and distributed linear solvers implement the same equation.
Whitening is not part of the current formulation.

## 5. Self-energy

The GW self-energy is

$$
\Sigma(1,2)=iG(1,2)W(1^+,2).
$$

In the ISDF representation, one time node builds \(G_{\mathbf k}(t)\) and
\(W_{\mathbf q}(t)\), lattice-transforms them, multiplies them elementwise,
and projects back to the requested band subspace. The shared spatial map is
schematically

$$
G_{\mathbf k}(t),W_{\mathbf q}(t)
\longrightarrow
\mathcal F\!\left[
\mathcal F^{-1}G\circ\mathcal F^{-1}W
\right]
\longrightarrow \Sigma_{\mathbf k,ij}(t).
$$

The frequency ansatz changes how time nodes and scalar projection
coefficients are obtained, not this spatial contraction:

| mode | frequency model |
|---|---|
| `x_only` | bare exchange |
| `cohsex` | static screened exchange plus Coulomb hole |
| `gn_ppm` | one pole fixed by \(W(0)\) and \(W(i\omega_p)\); on a measured-broken-TR deck two Hermitian residues \(R_\pm=B\pm D\) from the Hermitian and anti-Hermitian parts of \(W(i\omega_p)\), \(R_+\) on the empty and \(R_-\) on the occupied branch ([derivation](../dev/notes/DERIVATION_gnppm_nonhermitian.md)) |
| `hl_ppm` | one pole fixed by static screening and a sum rule |
| `mpa` | several complex poles fitted from two sampling lines; at `34228021` its contour completion is still time-reversal-symmetric and does not preserve the magnetic odd channel, so it is not a fallback for magnetic GN-PPM |

The band projection is

$$
\Sigma_{\mathbf k,ij}
=\sum_{ab\mu\nu}
\psi^*_{i\mathbf k a}(\mathbf r_\mu)
\Sigma_{\mathbf k,a\mu,b\nu}
\psi_{j\mathbf k b}(\mathbf r_\nu).
$$

Real or imaginary window projections must be applied before this band map
unless the map is known to commute with that projection. In general
\(K[\operatorname{Re}X]\ne\operatorname{Re}K[X]\).

The four-current implementation phases and route boundary are stated once on
[Four-current heads and frequency](four-current-head-corrections.md#four-current-phase-status).

## 6. Long-wavelength terms

At \(\mathbf q=0\), the Coulomb divergence multiplies a response that vanishes
with a power of \(q\). LORRAX stores the nonsingular body and treats the
analytic head separately. For screening,

$$
\chi_{00}(\mathbf q,z)
=q_aS_{ab}(z)q_b+\mathcal O(q^3),
$$

with Cartesian reciprocal coordinates. The convention and independent
dipole/Sternheimer builders are fixed in
[The S-tensor convention](s-tensor-convention.md).

The exchange head is direction dependent and produces
longitudinal-transverse exciton splitting. Its finite-\(\mathbf q\) and
mini-BZ-averaged meanings are separated in
[LT splitting and the exchange head](lt-exchange-head.md).

## 7. Quasiparticle Hamiltonian

The stored one-body operator is kinetic plus ionic; the
[direct field](hartree.md) is live:

$$
H_{\mathrm{QP}}=(T+V_{\mathrm{ion}})+H_{\mathrm{dir}}+\Sigma_{xc}.
$$

A dynamic self-energy is first interpolated on its real-frequency grid. The
QSGW map Hermitianizes matrix elements at the associated quasiparticle
energies and diagonalizes the result. States outside the explicitly computed
Sigma band range receive the established scissor continuation; that
continuation is not another self-energy ansatz.

In a self-consistent calculation, updated energies and active orbitals feed
the next response and self-energy build. A frozen pole model is therefore not
a self-consistent full-frequency calculation.

## 8. Computational invariants

Only layout facts forced by the physics belong here:

| object | logical layout | reason |
|---|---|---|
| \(\psi_{\mathbf k}(\mu)\) | flat k; centroid axis sharded | band GEMMs remain contiguous |
| \(V_q,\chi_q,W_q\) | `P(None,'x','y')` | both matrix axes distributed |
| \(\Sigma_{\mathbf k,ij}(\omega)\) | k/band matrix, optionally sharded | avoid replicated frequency cubes |
| zeta and fitted poles on disk | irreducible q wedge | symmetry commutes with their non-FFT construction |

The lattice FFT convolution consumes full-zone k/q data. Irreducible-zone or
time-reversal reductions are used for non-FFT work and storage, followed by an
explicit unfold before the convolution. SlabIO owns distributed large-array
I/O; no rank should materialize an \(N_\mu^2\) matrix merely to write or fit it.

## 9. Present validity boundary

The mature static, PPM, and insulating MPA modes support their documented
driver contracts. Bispinor MPA uses the packed current owner, and its ordered
contour completion preserves the magnetic odd residue. Metallic
sample geometries exist, but
fractional occupations, intraband response, and denominator cells that
straddle zero still require explicit physics. A code path that lacks those
terms must refuse rather than manufacture a gap or threshold occupations.
