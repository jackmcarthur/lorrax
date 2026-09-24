# Direct Hartree field

GW builds the direct field live from its own wavefunctions and occupations.
`kin_ion.h5` carries only $T+V_{\rm loc}+V_{\rm NL}$. There is one
implementation, `gw.kin_ion_io.compute_hartree_matrix`, and it always works
on the WFN FFT grid: there is no stored, folded or ISDF Hartree.

## Sources

$$
\rho(\mathbf r)=\sum_{\mathbf k n}w_{\mathbf k}f_{n\mathbf k}
\psi^\dagger_{n\mathbf k}(\mathbf r)\psi_{n\mathbf k}(\mathbf r),
\qquad
J_i(\mathbf r)/c=\sum_{\mathbf k n}w_{\mathbf k}f_{n\mathbf k}
\psi^\dagger_{n\mathbf k}(\mathbf r)\alpha_i\psi_{n\mathbf k}(\mathbf r).
$$

$f_{n\mathbf k}$ is the physical occupation, fractional on a metal; the
screening band window does not truncate the density. The signed Dirac
current $J$ exists only for four-component bispinors. It is a current, not a
second charge density, and enters only through the transverse projector
below.

Both sums come from one density scan, `gw.qsgw_density.rho_from_wfns`, over
the star wedge of k with weights $|\text{star}|/N_k$, star-averaged by the
FFT-grid pullback. The one-shot driver and the self-consistent map share it.
The current is then projected onto the polar-vector representation of the
magnetic group (`symmetry_maps.project_polar_fft_field`, antiunitary rows
included).

## G-space solve

$$
V_H(\mathbf G)=\frac{8\pi\rho(\mathbf G)}{|\mathbf G|^2},\qquad
A_i(\mathbf G)=s_{TT}\,v(\mathbf G)
\left(\delta_{ij}-\frac{G_iG_j}{|\mathbf G|^2}\right)\frac{J_j(\mathbf G)}{c},
\qquad V_H(0)=A_i(0)=0 .
$$

`sys_dim = 2` replaces $v$ by the slab-truncated kernel. $s_{TT}=-1$ is the
Coulomb-gauge spatial-metric sign, `vcoul.COULOMB_GAUGE_TT_SIGN`: the same
sign the bare TT exchange tiles carry. The scalar Poisson owner supplies $v$
to both fields (`psp.dft_operators.transverse_potential_from_current`). The
periodic zero mode is exactly zero, so no mini-BZ head enters the direct
field.

The band operator is

$$
H^{\mathrm{dir}}_{mn\mathbf k}=
\langle m\mathbf k|V_H|n\mathbf k\rangle+
\Big\langle m\mathbf k\Big|\sum_i\alpha_iA_i\Big|n\mathbf k\Big\rangle ,
$$

and scalar runs omit the second term.

## Algorithm and cost

1. **Sources.** One inverse FFT per (wedge k, occupied band). Bands are
   sharded over all ranks, and one reduction forms $\rho$ and $J$.
2. **Poisson.** Two 3-D FFTs per field component, replicated on every rank. This is
   negligible beside step 3, and it makes $V_H(\mathbf r)$ bit-identical
   across ranks.
3. **Matrix elements.** One k-scan over the star wedge
   (`common.mtxel_sweep.sweep_matrix_elements`). Every rank holds a G slab
   of every band, so $N_k$ is a trip count, not a parallel axis. Per wedge k
   the cost is $n_b$ FFTs, a local multiply by the potential, and one
   $n_b\times n_b$ contraction over G. The output is
   $(N_k, n_b, n_b)$ Ry at `P(None,'x','y')`, broadcast from the wedge to
   the full zone on device.

## Lifecycle

One-shot GW builds the field once from the DFT orbitals. Density-self-consistent
GW (`density_self_consistent = true`; bispinor QSGW refuses without it,
`GATE bispinor_self_consistency_requires_live_four_current`) rebuilds
$\rho$, $J$ and both fields from the current orbitals at every map
(`gw.sc_iteration.rebuild_hartree_dft_basis`) and contracts them in the DFT
basis. The frozen DFT field is then dropped from the Σ stage, so neither
field is counted twice.

Sigma writers store the aggregate `Hdir` beside its components $V_H$ and
$H_T$. They refuse unless $H^{\rm dir}=V_H+H_T$ holds exactly on the rows
they write.

The schedules and API are in `docs/dev/rho_vh_2d_design.md` (repository
only).
