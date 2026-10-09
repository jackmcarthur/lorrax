# Direct Hartree field

GW builds the direct field from its own wavefunctions and occupations.
`kin_ion.h5` carries only $T+V_{\rm loc}+V_{\rm NL}$. There is one
ordinary-WFN implementation, `gw.hartree.direct_field_matrices`, which works
on the WFN FFT grid. Atomic reconstruction additionally retains a direct
receiving band matrix from the shared fitting frame, as described below;
it does not fit the Hartree matrix through ISDF points.

## Sources

$$
\rho(\mathbf r)=\sum_{\mathbf k n}w_{\mathbf k}f_{n\mathbf k}
\psi^\dagger_{n\mathbf k}(\mathbf r)\psi_{n\mathbf k}(\mathbf r),
\qquad
J_i(\mathbf r)/c=\sum_{\mathbf k n}w_{\mathbf k}f_{n\mathbf k}
\psi^\dagger_{n\mathbf k}(\mathbf r)\alpha_i\psi_{n\mathbf k}(\mathbf r).
$$

$f_{n\mathbf k}$ is the physical occupation, fractional on a metal; the
screening band window does not truncate the density. On bispinor decks
$\psi$ is the normalized RKB four-component lift
([carrier](bispinor-gw.md#lift)), and $\rho$ and $J$ come from one scan of
it: the direct field is the full four-current Hartree
$V_H[\rho]+\boldsymbol\alpha\cdot\mathbf A[J]$ on one carrier, and
$\int\rho$ is the electron count exactly. The signed Dirac
current $J$ exists only for four-component bispinors. It is a current, not a
second charge density, and enters only through the transverse projector
below.

Both sums come from one density scan, `gw.qsgw_density.rho_from_wfns`, over
the star wedge of k with weights $|\text{star}|/N_k$, star-averaged by the
FFT-grid pullback. The one-shot driver and the self-consistent map share it.
Inside that scan the current is projected once onto the polar-vector
representation of the magnetic group (`symmetry_maps.project_polar_fft_field`,
antiunitary rows included); its receipt (the movement of the raw field, the
covariance residual of the projected one) is the one the run reports.

### Reconstructed fixed-source charge

For `atomic_reconstruction_dir` with `bispinor_gw = coulomb_only`, the source
and receiving orbitals use the same full-WFN Löwdin factor on their smooth
and atomic pieces. Both large and small components enter $\rho$. Source
weights are the physical occupations and full-zone k quadrature; the
occupied-band weight in the ISDF loss never becomes a density weight.
Physical bands and their padded transport carrier have separate domains.

`gw.augmentation_hartree_receiving.build_resident_receiving_J` reuses the
ordinary FFT Poisson and matrix sweep for the smooth field. The existing
`isdf.atomic_hartree` owner supplies the compact local corrections and the
periodic neutral-cell mean terms. The smooth receiving overlap is measured
in the shared frame, rather than replaced by an identity matrix. Atomic
endpoint samples remain distributed over both mesh axes; bounded band
tiles construct the native FILE-wedge operator at `P(None,'x','y')` before
the fitting stage releases its orbital store.

For a fixed occupied source the local Hartree expression is linear in the
receiving density. `isdf.atomic_hartree.charge_hartree_functional` therefore
supplies its radial adjoints once, including compensation, both smooth/local
cross terms, exact monopole enrichment and both periodic mean terms. Applying
the angular projection's transpose gives point weights $u_\delta$ and
$u_{\rm PS}$, so its atomic receiving matrix is

$$
J^{\rm local}_{mn}=
\sum_p u_\delta(p)\left[
\psi^{\rm PS\dagger}_m(p)\delta\psi_n(p)
+\delta\psi_m^\dagger(p)\psi^{\rm PS}_n(p)
+\delta\psi_m^\dagger(p)\delta\psi_n(p)\right]
+\sum_p u_{\rm PS}(p)\psi^{\rm PS\dagger}_m(p)\psi^{\rm PS}_n(p)
+\sum_a b_a M^a_{00,mn}.
$$

Here $p$ runs over atom, radius and angular sample, all four spinor components
are contracted, and $M_{00}$ is the independently evaluated physical correction
monopole. The point weights include the single $N_{\rm FFT}/\Omega$ conversion
from grid-normalized orbital products; the exact monopole covector $b$ has no
such factor. The response is transposed without conjugation because it acts
linearly on the receiving density. Bounded ket tiles contract these weights
with the distributed orbital samples directly, avoiding band-pair radial
clouds while retaining the original decomposition into physical terms.

`gw.augmentation_hartree` authenticates the density, frame, geometry and
operator recipe, serves the saved native matrix with the canonical FILE/TR
map, and lets the existing Sigma assembly apply a requested basis rotation
once. The restart bundle stores the matrix with its source/operator binding
and logical payload checksum. A missing or changed reconstructed source
refuses before large restart tensors are read. The admitted lifecycle is
fixed-source `one_shot_dft`; an updated density requires reconstruction of
its source and is currently refused. Unaugmented runs retain the live path.

The reconstruction changes the interaction vertices while retaining the
original DFT energies and ionic/XC references. A consistent all-electron
one-body reference and frozen-core counterterms require a separate physical
comparison; correctness of this direct field alone does not establish
all-electron quasiparticle energies.

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
