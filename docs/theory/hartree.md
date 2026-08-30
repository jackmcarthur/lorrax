# Direct Hartree field

`kin_ion.h5` contains only kinetic and ionic operators. GW builds the direct
field live from its wavefunctions and occupations.

## Sources

$$
\rho(\mathbf r)=\sum_{\mathbf k n}w_{\mathbf k}f_{n\mathbf k}
\psi^\dagger_{n\mathbf k}(\mathbf r)\psi_{n\mathbf k}(\mathbf r).
$$

Here $f_{n\mathbf k}$ is the physical occupation; a screening cutoff does not
truncate the density. Four-component bispinors also form the signed Dirac
current

$$
J_i(\mathbf r)/c=\sum_{\mathbf k n}w_{\mathbf k}f_{n\mathbf k}
\psi^\dagger_{n\mathbf k}(\mathbf r)\alpha_i
\psi_{n\mathbf k}(\mathbf r).
$$

Charge and current use the same orbitals, occupations, and inverse FFT. $J/c$
is a current, not a second charge density.

## G-space solve

$$
V_H(\mathbf G)=\frac{8\pi\rho(\mathbf G)}{|\mathbf G|^2},\qquad
\mathbf G\ne0.
$$

The periodic zero mode is zero; `sys_dim=2` applies the slab Coulomb factor.
For bispinors, the same kernel acts on transverse current:

$$
A_i(\mathbf G)=s_{TT}v(\mathbf G)
\left(\delta_{ij}-\frac{G_iG_j}{|\mathbf G|^2}\right)J_j(\mathbf G)/c.
$$

Here $s_{TT}$ is the shared transverse-metric sign and $A_i(0)=0$. The
band-space operator is

$$
H^{\mathrm{dir}}_{mn\mathbf k}=
\langle m\mathbf k|V_H|n\mathbf k\rangle+
\left\langle m\mathbf k\left|\sum_i\alpha_iA_i\right|n\mathbf k\right\rangle.
$$

Scalar runs omit the vector term. One packed matrix-element sweep returns
`P(None,'x','y')` over the two band axes.

## Lifecycle

One-shot GW uses DFT orbitals. Density-self-consistent GW rebuilds the sources
and field from the current orbitals at every iteration, then contracts the
field in the DFT basis. Screening is a separate GW stage. Hartree always uses
the WFN FFT grid: there is no stored, folded, or ISDF Hartree selector.

API ownership, schedules, and evidence scope:
`docs/dev/rho_vh_2d_design.md` (repository only).
