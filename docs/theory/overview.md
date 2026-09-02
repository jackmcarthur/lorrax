# Theory map

LORRAX evaluates GW and related response functions in an interpolative
separable density-fitting (ISDF) basis. The common calculation is

$$
\{\psi_{n\mathbf k},\epsilon_{n\mathbf k}\}
\longrightarrow \zeta_{q\mu}
\longrightarrow (V_q,\chi^0_q)
\longrightarrow W_q
\longrightarrow \Sigma_{\mathbf k}(\omega)
\longrightarrow H_{\mathrm{QP}}.
$$

Each arrow has one detailed owner:

| question | read |
|---|---|
| How do ISDF, screening, self-energy, and QSGW fit together? | [Core ISDF and GW theory](physics.md) |
| How are interpolation vectors and Coulomb matrices formed? | [G-flat zeta and V](isdf-zeta-vq.md) |
| How is the direct Hartree field built from charge and current? | [Direct Hartree field](hartree.md) |
| Which symmetry convention controls irreducible-zone work and unfolding? | [Symmetry](symmetry.md) |
| What problem does the Sigma quadrature solve, and what must any method obey? | [The Sigma(omega) quadrature problem](sigma-quadrature-problem.md) |
| How are static and GN/HL-PPM frequency integrals separated? | [Minimax quadrature](minimax-quadrature.md) |
| What fixes the HL plasmon pole? | [HL-GPP derivation](hl-gpp-derivation.md) |
| How are MPA samples, poles, and Sigma windows constructed? | [Multipole frequency integration](THEORY_mpa_implementation.md) |
| What is the long-wavelength response convention? | [S-tensor convention](s-tensor-convention.md) |
| Why is the exchange head direction dependent? | [LT splitting and the exchange head](lt-exchange-head.md) |
| How do the four-current (bispinor) channels treat q→0, and which carry frequency? | [Four-current heads and frequency](four-current-head-corrections.md) |

The theory pages state equations, conventions, validity domains, and the few
data layouts forced by those equations. Exact input defaults belong to the
[input reference](../input_reference.md). Module ownership and dependency
rules belong to [architecture](../architecture/codebase.md). Historical
measurements and implementation campaigns belong under `docs/reports` or
`docs/dev`, not here.

Three principles recur throughout:

1. Occupation selects a spectral branch; it does not redefine a signed band
   energy.
2. Expensive pair sums are replaced by separable Green-function contractions,
   and symmetry is used only where the operation commutes with unfolding.
3. Scalar quadrature, distributed storage, and spatial physics have distinct
   owners. A numerical rule never knows about bands or HDF5; SlabIO never
   decides physics.
