# Direct Hartree field

LORRAX builds the direct field from the wavefunctions and occupations used by
the current calculation. It does this when GW runs. The `kin_ion.h5` file
contains only the kinetic and ionic operators.

## Charge and current

The scalar source is

$$
\rho(\mathbf r)=\sum_{\mathbf k n}w_{\mathbf k}f_{n\mathbf k}
\psi^\dagger_{n\mathbf k}(\mathbf r)\psi_{n\mathbf k}(\mathbf r).
$$

Here $w_{\mathbf k}$ is the k-point weight and $f_{n\mathbf k}$ is the full
occupation weight. A screening band cutoff does not truncate this sum.

A four-component bispinor calculation also forms the signed Dirac current

$$
J_i(\mathbf r)/c=\sum_{\mathbf k n}w_{\mathbf k}f_{n\mathbf k}
\psi^\dagger_{n\mathbf k}(\mathbf r)\alpha_i
\psi_{n\mathbf k}(\mathbf r).
$$

This is a current density, not a second charge density. Charge and current are
contracted from the same orbitals, occupations and inverse FFT.

## Reciprocal-space solve

In Rydberg units, the periodic scalar solve is

$$
V_H(\mathbf G)=\frac{8\pi\rho(\mathbf G)}{|\mathbf G|^2},\qquad
\mathbf G\ne0.
$$

The periodic average $V_H(\mathbf G=0)$ is set to zero. A two-dimensional
calculation applies the slab Coulomb factor selected by `sys_dim`.

For a bispinor, the same Coulomb kernel acts on the transverse part of the
current,

$$
A_i(\mathbf G)=s_{TT}v(\mathbf G)
\left(\delta_{ij}-\frac{G_iG_j}{|\mathbf G|^2}\right)J_j(\mathbf G)/c.
$$

The transverse projector and its sign convention are shared with the other
current vertices. The zero mode is again zero.

The band-space direct operator is

$$
H^{\mathrm{dir}}_{mn\mathbf k}=
\langle m\mathbf k|V_H|n\mathbf k\rangle+
\left\langle m\mathbf k\left|\sum_i\alpha_iA_i\right|n\mathbf k\right\rangle.
$$

The vector term is absent in a scalar calculation. LORRAX packs the scalar
and vector actions into one matrix-element sweep. The result remains sharded
over both band axes.

## Self-consistency

A one-shot calculation uses the DFT orbitals and their physical occupations.
A density-self-consistent calculation rebuilds charge, current and the direct
field from the current occupied orbitals at every iteration. It then forms
the matrix elements in that same basis. Screening is rebuilt by its own GW
path; it is not part of the Hartree solve.

ISDF compresses screening and self-energy objects. The direct field instead
uses the wavefunction FFT grid.

Implementation ownership and parallel schedules are described in the
repository-only note `docs/dev/rho_vh_2d_design.md`.
