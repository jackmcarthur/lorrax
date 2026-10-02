# LORRAX for BerkeleyGW users: inputs, outputs and comparison

This page is for someone who knows BerkeleyGW and wants to run LORRAX on the
same mean field and compare the results. It maps BerkeleyGW's input keys to
LORRAX's, documents every file the GW driver writes (format, columns, units,
band and k conventions), lists the conventions that make the two codes compute
different numbers from the same wavefunctions, and gives the measured
agreement. Run the [quickstart](../quickstart.md) first; deck keys are owned by
the [input reference](../input_reference.md).

## 1. How the two codes divide the work

BerkeleyGW computes the inverse dielectric matrix $\varepsilon^{-1}_{GG'}(q,\omega)$
in a plane-wave basis with `epsilon.x`, then evaluates $\Sigma$ for listed
k points and bands with `sigma.x`. LORRAX does both in one driver,
`python -m gw.gw_jax`, and replaces the plane-wave basis of $\chi_0$ and $W$
by interpolative separable density fitting (ISDF): pair densities are fitted
on $N_\mu$ real-space points (centroids), so $\chi_0$, $V$ and $W$ are
$N_\mu \times N_\mu$ matrices per $q$ ([theory map](../theory/overview.md)).
$N_\mu$ is the convergence parameter that LORRAX adds and BerkeleyGW does not
have ([ISDF exchange accuracy](../theory/isdf-exchange-accuracy.md)).

Both codes read the same `WFN.h5` from QE's `pw2bgw.x`. LORRAX additionally
needs, from its own preprocessing drivers ([drivers](../drivers.md)):

| file | producer | replaces in BerkeleyGW |
|---|---|---|
| `centroids_frac_<N>.txt` | `centroid.kmeans_cli` | the G-vector basis of $\varepsilon$ |
| `kin_ion.h5`, $\langle T + V_\mathrm{loc} + V_\mathrm{NL}\rangle$ | `gw.kin_ion_io` | `vxc.dat` / `kih.dat` (§4.5) |
| `dipole.h5`, velocity matrix elements | `psp.get_dipole_mtxels` | the shifted-$q_0$ calculation of the $q \to 0$ head (§4.2) |

and QE's `data-file-schema.xml` beside `WFN.h5`, from which it reads the
symmetry operations and time-reversal flags. Let $n_\mathrm{occ}$ be the
number of occupied bands, the largest `ifmax` of the WFN (for a spinor WFN it
equals the number of electrons). LORRAX computes $\Sigma$ for every band from
the lowest up to $n_\mathrm{occ} + $ `ncond`, at every k point of the WFN's
irreducible wedge. The WFN must be spin-unpolarized (`nspin = 1`, scalar or
spinor) and complex: a collinear spin-polarized WFN (`nspin = 2`) refuses at
load (`wfn_loader.WfnLoader`); use a noncollinear calculation for magnetism.

## 2. Inputs: BerkeleyGW keys and their LORRAX counterparts {#inputs}

| BerkeleyGW key (file) | LORRAX | note |
|---|---|---|
| `frequency_dependence 0` (sigma) | `compute_mode = cohsex` | static COHSEX |
| `frequency_dependence 1` | `compute_mode = hl_ppm` | Hybertsen–Louie plasmon pole |
| `frequency_dependence 3` | `compute_mode = gn_ppm` | Godby–Needs; probe frequency `ppm_omega_p` (default 2 Ry) |
| `frequency_dependence 2` | `compute_mode = mpa`, `sigma_w_model = shared_pole` | full frequency; LORRAX fits W with real poles shared per $q$ ([shared-pole W](../theory/shared-pole-w-model.md)) instead of tabulating $\varepsilon^{-1}(\omega)$ |
| `number_bands` (epsilon) | `number_bands_chi` | bands in $\chi_0$ |
| `number_bands` (sigma) | `number_bands_sigma` | bands in the $\Sigma$ sum; `number_bands` sets both |
| `band_index_min/max`, `diag` | `ncond`, or `number_bands_protected` (the total band count of the request: every occupied band plus conduction bands up to that total) | $\Sigma$ bands are always $[1, n_\mathrm{occ} + $`ncond`$]$; `nval` does not change them ([its roles](../input_reference.md#system)) |
| `epsilon_cutoff`, `screened_coulomb_cutoff` | none | W is built in the ISDF basis from the same $V_q$ as $\Sigma_x$ (§4.1) |
| `bare_coulomb_cutoff` (sigma) | `bare_coulomb_cutoff` | default the WFN's `ecutwfc` in both codes |
| `cell_average_cutoff` (sigma) | `mc_average_vcoul_body` | §4.1 |
| `write_vcoul` (sigma) | `use_bgw_vcoul`, `bgw_vcoul_file` | LORRAX reads that table (§4.1) |
| `qpoints` with a shifted $q_0$ (epsilon) | none | the $q \to 0$ head comes from `dipole.h5` (`wcoul0_source`); head overrides `vhead`, `whead_0freq`, `whead_imfreq` (§4.2) |
| `exact_static_ch` (sigma) | none | LORRAX's COH is a partial band sum (§4.3) |
| `cell_slab_truncation` | `sys_dim = 2` | an untruncated bulk run is `sys_dim = 3`; the key is required |
| `cell_box_truncation`, `cell_wire_truncation`, `spherical_truncation` | none documented | |
| `screening_semiconductor`, `screening_metal` | none | the class is read from the WFN occupations; a metal needs `occ_smearing_width_ry` ([metals](metals.md)) |
| `broadening` (full frequency) | `sigma_regularization_ev` | $\eta$ of $\Sigma(\omega)$, default 0.25 eV |
| `frequency_dependence_method`, `delta_frequency_eval`, `max_frequency_eval` | `sigma_omega_min_ev`, `sigma_omega_max_ev`, `sigma_omega_step_ev` | the real-axis grid on which $\Sigma(\omega)$ is stored |
| `invalid_gpp_mode` | `ppm_invalid_mode` | `static_limit` = mode 3, `zero` = mode 0, `2ry` = mode 2 |
| `finite_difference_form`, `finite_difference_spacing` | none | fixed central difference at ±0.5 eV (§4.6) |
| `vxc.dat`, `kih.dat`, `dont_use_vxcdat` | none | §4.5 |
| `no_symmetries_q_grid` | none | LORRAX uses every symmetry QE reports |
| k-point list (sigma) | none | every k of the WFN wedge |
| `degeneracy_check_override` | none | a band count that cuts a degenerate multiplet refuses |
| `eqp_corrections` (kernel, absorption) | `--eqp` of `bse.bse_jax` | §7 |
| `use_momentum` (absorption) | `psp.get_dipole_mtxels --skip-vnl` | §7 |
| `number_val_bands`, `number_cond_bands` | `--n-val`, `--n-cond` | counted in bands, spinor bands included |
| `energy_resolution`, `number_iterations` | `--eta-eV`, `--n-iter` of `bse.absorption_haydock` | |

## 3. Outputs {#outputs}

### 3.1 What the GW driver writes

All files go beside the deck unless a path is given. Energies in text files
are eV; HDF5 datasets are eV when their name ends in `_ev` and Ry otherwise.
Every text file starts with one line `# Generated by LORRAX <version> at <UTC time>`.

| file | written when | content | BerkeleyGW counterpart |
|---|---|---|---|
| `eqp0.dat`, `eqp1.dat` | every run except a non-converged SC run | QP energies on the wedge, BerkeleyGW format | `eqp0.dat`, `eqp1.dat` |
| `sigma_diag.dat` | every run | $\Sigma$ diagonal decomposition per k and band | `sigma_hp.log` |
| `eqp_g0w0.dat` | dynamic one-shot (PPM, MPA) | $H_0 + \Sigma_{xc}(E_\mathrm{DFT})$, real and imaginary | — |
| `sigma_mnk.h5` | dynamic modes (`gn_ppm`, `hl_ppm`, `mpa`) | the full band matrix $\Sigma_{mn}(k, \omega)$ on the ω grid | — |
| `tmp/isdf_tensors_<N_mu>.h5` | `write_restart_tensors = true` (default) | the restart bundle: $V_q$, static $W_0$, ψ at the centroids, energies, head | `eps0mat.h5`, `epsmat.h5` (for the BSE) |
| `qp_wfn_rotations.h5` | every run | the QP eigensystem $U$, $E_\mathrm{QP}$ | — |
| `WFN_qp.h5` | see §3.6 | the WFN rotated into QP states, with QP energies | — |
| `dipole_qsgw.h5` | SC runs with a velocity head | velocity matrix between QP states | `vmtxel` |
| `gwjax.out` | every run | the run report | `sigma.out` |

Rank 0 writes the text files. Band and k conventions differ by file; §3.2–3.3
state them.

### 3.2 `eqp0.dat` and `eqp1.dat`

The format is BerkeleyGW's (`gw.eqp_bgw.write_bgw_eqp`). After the provenance
line, one block per k point of the WFN's irreducible wedge, in the WFN's order:

```text
kx ky kz nspin*nbands          Fortran (3f13.9, i8): k in crystal coordinates
ispin iband E_DFT E_QP         Fortran (2i8, 2f15.9): one row per band, energies in eV
```

The band index is 1-based and absolute, as in BerkeleyGW. Bands $1 \dots
n_\mathrm{occ} + $`ncond` are written; spin is always 1 (spinor and
spin-unpolarized WFNs). Energies are absolute, on the DFT eigenvalue scale,
unshifted. Past the first line the file is byte-compatible with BerkeleyGW's;
SC and `eqp2.dat` files carry a few more `#` lines, so pass
`grep -v '^#' eqp1.dat` to a BerkeleyGW tool.

**One-shot runs** (`qp_solver = one_shot_dft`, what the default `auto`
resolves to unless the deprecated `self_consistent = true` is set) write the
diagonal quantities BerkeleyGW writes. With the DFT-basis diagonal

$$\Delta_n(E) = \mathrm{Re}\big[\langle n|T + V_\mathrm{ion}|n\rangle + V_{H,nn} + \Sigma_{x,nn} + \Sigma_{c,nn}(E)\big] - E^\mathrm{DFT}_n,$$

`eqp0.dat` holds $E^\mathrm{DFT}_n + \Delta_n(E^\mathrm{DFT}_n)$ and
`eqp1.dat` holds $E^\mathrm{DFT}_n + Z_n\,\Delta_n(E^\mathrm{DFT}_n)$ with
$Z_n = (1 - \partial_\omega\mathrm{Re}\,\Sigma_{c,nn})^{-1}$ (§4.6), written
for every $Z$, including $Z$ outside $(0, 1]$
(`gw.eqp_bgw.assemble_eqp`). No QP equation is solved. For static COHSEX,
$\Sigma = \Sigma_\mathrm{SX} + \Sigma_\mathrm{COH}$ and $Z = 1$, so the two
files are equal. The $\Sigma$ diagonals are averaged over degenerate DFT
states, as BerkeleyGW does (`no_degen_averaging`, `degen_avg_tol_ry`, default
$10^{-6}$ Ry; `gw.degen_average`).

**Self-consistent runs** (`qp_solver = self_consistent`) write the converged
QSGW spectrum to both files: row $n$ pairs the $n$-th DFT energy with the
$n$-th eigenvalue of the accepted map's Hamiltonian, so the band label is an
energy order, not a state identity ([self-consistency](../self_consistency.md)).
A run that ends not converged refuses before writing them; the per-map
`eqp0_iterNNNN.dat` remain.

`eqp2.dat` (`write_eqp2 = true`, dynamic one-shot only) holds the fixed-$\Sigma$
eigenvalue fixed point: the stored $\Sigma(\omega)$ matrix is iterated in the
evolving QP basis, without rebuilding $W$.

### 3.3 `sigma_diag.dat`

Written by `file_io.sigma_output.write_sigma_to_file`. Header lines name the
decomposition (`# sigXC = sigX + sigC` or `# sigTOT = sigSX + sigCOH`), the
optional columns, and the k basis. Each k block is:

```text
k-point 0:
# kcrys    0.000000000    0.000000000    0.000000000
----------------------------------------------------------------------------------------------------
n=0   sigX=<re>  sigC=<re>+<im>i  sigXC=<re>+<im>i  VH=<re>  Eo=<re>  Z=<re>
```

`k-point K` is the 0-based position in the WFN's wedge, and `# kcrys` its
crystal coordinate; join blocks to another code by that coordinate modulo a
reciprocal lattice vector, never by position. **`n` is 0-based**: LORRAX
`n=18` is BerkeleyGW band 19. A complex column prints `re+ imi` when any entry
has an imaginary part above $10^{-10}$. All values are eV.

| column | meaning | `sigma_hp.log` column to compare |
|---|---|---|
| `sigX` | bare exchange $\Sigma_x$ (dynamic modes) | `X` |
| `sigC` | $\Sigma_c(E_\mathrm{DFT})$, complex, $q \to 0$ head included; extrapolated past the band count when band extrapolation is on | the sum of the columns `SX-X` and `CH′` (plasmon pole); `Re Cor` (full frequency) |
| `sigXC` | `sigX + sigC` | `Sig′` |
| `sigSX` | screened exchange (static modes; bare $\Sigma_x$ under `x_only`) | `X + (SX−X)` |
| `sigCOH` | Coulomb hole, partial band sum (static modes) | `CH′`, not `CH` (§4.3) |
| `sigTOT` | `sigSX + sigCOH` | `Sig′` |
| `VH` | Hartree diagonal, recomputed from the orbitals; `Hdir` on bispinor runs (Hartree plus the transverse direct term) | — (BerkeleyGW's `Vxc` column subtracts the exchange–correlation potential instead, §4.5) |
| `Eo` | $E_\mathrm{DFT}$ | `Emf`, not BerkeleyGW's `Eo` |
| `Z` | renormalization at $E_\mathrm{DFT}$ (dynamic one-shot) | `Znk` |
| `sigCC`, `sigTT`, `sigCT` | bispinor runs: charge–charge, transverse–transverse and mixed (CT + TC) parts; they sum to the total | — |
| `sigC_odd` | runs with measured broken time reversal on GN-PPM or MPA: the odd part of $\Sigma_c$ | — |
| `sigC_raw`, `eqp0_raw`, `eqp1_raw` | one-shot runs with band extrapolation: $\Sigma_c$, eqp0 and eqp1 from the band sum truncated at `number_bands_sigma`, from the same evaluation | `SX-X` plus `CH′`, and `Eqp0′`, `Eqp1′`, at the same band count (§4.3–4.4) |

The file has no `kin_ion` column, so eqp0 cannot be rebuilt from it alone; the
debug table `sigma_freq_debug.dat` (`sigma_freq_debug_output = true`) has
`kin_ion`, `V_H`, `x_bare`, the $\Sigma_c$ columns and `eqp0`, `eqp1` per row,
tab-separated with a named header. In SC runs the diagonals are those of the
final map's $\Sigma$ rotated back to the DFT basis, and there are no raw
columns.

### 3.4 `sigma_mnk.h5`

The frequency-dependent self-energy as full band matrices, written by
`file_io.sigma_output.write_sigma_omega_h5` (dynamic modes only; a one-shot at
finalize, an SC run once after convergence in the last map's QP basis).

| dataset | shape | meaning |
|---|---|---|
| `omega_ev` | $(n_\omega)$ | the ω grid, eV, relative to a reference energy: attributes `omega_reference_ev` and `omega_reference_provenance` (`midgap`, `vbm` or `fixed-N mu`, from `fermi_reference`), and the broadening `sigma_regularization_ev` |
| `sigma_c_kij_ev` | $(n_\omega, n_k, n_b, n_b)$ | $\Sigma_c(\omega)$, $q \to 0$ head included |
| `sigma_sx_kij_ev` | $(n_k, n_b, n_b)$ | bare $\Sigma_x$, despite the name |
| `hartree_kij_ev` | $(n_k, n_b, n_b)$ | Hartree (bispinor: plus the transverse direct term, also stored split) |
| `sigma_total_kij_ev` | $(n_\omega, n_k, n_b, n_b)$ | $\Sigma_c(\omega) + \Sigma_x + V_H$ (no kinetic or ionic term) |
| `sigma_eval_rel_ev` | $(n_k, n_b)$ | the energy each state's $\Sigma$ was read at, relative to the reference; attribute `sigma_eval_provenance` = `at_e_dft` (one-shot) or `self_consistent_qp` |
| `irr_idx_k`, `sym_idx_k` | $(n_{k,\mathrm{full}})$ | for each full-grid point, its stored row and the symmetry operation that maps it |
| `sigma_c_extrap_{inf,last,sigma}_kn_ev`, `sigma_c_extrap_beta_kn` | $(n_k, n_b)$ | band-extrapolation fit results, when it ran |
| `sigma_xc_qsgw_kij_ev`, `qp_omega0_ev` | | only with `write_qsgw_datasets = true`: the Hermitian static QSGW $\Sigma_{xc}$ and the eigenvalues of $H_0 + \Sigma(\omega \approx 0)$ |

Bands are 0-based from band 0. The k axis holds one representative per
symmetry star of the full grid (`k_storage = "ibz"`); this can be fewer rows
than the WFN's k list when the WFN stores symmetry-related points (for
example $k$ and $-k$), and the file stores no k coordinates. The cubes are not
degeneracy-averaged. A file whose `lorrax_io_committed` attribute is 0 is an
interrupted write.

`python -m gw.eqp_bgw <run_dir>` rebuilds `eqp0.dat` and `eqp1.dat` from
`sigma_mnk.h5`, `kin_ion.h5`, `WFN.h5` and `qp_wfn_rotations.h5`
(`--finite-difference-spacing`, default 0.5 eV); it refuses an SC file
(`GATE eqp_bgw_self_consistent_run`), whose eqp files are already final.

### 3.5 The restart bundle and the static W

`tmp/isdf_tensors_<N_mu>.h5` holds what a later GW restart or a BSE run
needs: `V_qmunu` (the bare interaction in the centroid basis per stored $q$),
`W0_qmunu` (the static screened interaction, valid when attribute `W0_ready`
is true), the parent wavefunctions at the centroids (`psi_parent_y`,
`psi_parent_y_mun`), `enk_full` (DFT energies, Ry), the band window, the
centroid table hashes, and the $q \to 0$ head (`vhead`, `whead`,
`S_cart_head`, `G0_mu_nu`). For a dynamic $W$ the stored $W_0$ is the model at
$\omega = 0$. $q$ is stored on the irreducible parents when the centroids are
closed under the symmetry orbits, otherwise on the full grid; readers unfold
it. The BSE consumes this file in place of BerkeleyGW's `eps0mat.h5` and
`epsmat.h5` ([BSE inputs](../architecture/bse.md#inputs)). `tmp/zeta_q.h5`
holds the fitted interpolation vectors $\zeta_q(G)$.

### 3.6 `qp_wfn_rotations.h5` and `WFN_qp.h5`

`qp_wfn_rotations.h5` holds `U_mnk` $(n_k, n_b, n_b)$ with
$U_{mn}(k) = \langle m_\mathrm{DFT}|n_\mathrm{QP}\rangle$, the QP energies
(`E_qp_nk_rydberg`, `E_qp_nk_hartree`), `band_range`, the k tables and the
source WFN's fingerprint. In a one-shot run these are the eigenpairs of the
full Hermitian QSGW matrix $T + V_\mathrm{ion} + V_H + \Sigma_{xc}(E_\mathrm{DFT})$,
not the diagonal eqp0 values; the gap printed in `gwjax.out` is the gap of
this matrix. htransform, the BSE and SC seeding read the file.

`WFN_qp.h5` is a BerkeleyGW-layout WFN with the bands of the QP window
rotated, $c^\mathrm{QP}_n = \sum_m U_{mn} c^\mathrm{DFT}_m$, and their
energies replaced by $E_\mathrm{QP}$ (Ry). SC runs write it at the end
(`write_wfn_h5`, default true). A one-shot run writes it only when $\Sigma$
and the WFN share one k set, which a symmetry-reduced WFN does not; build it
then with `python -m postprocess.rotate_wfn_to_qp WFN.h5 qp_wfn_rotations.h5`.

### 3.7 `dipole.h5` and `dipole_qsgw.h5`

`dipole.h5` (`psp.get_dipole_mtxels`) holds `dipole_cart` $(3, n_k, n_b, n_b)$,
the velocity matrix $\langle mk|\hat v_a|nk\rangle$ in Ry atomic units
(Ry·bohr) on the full grid, `band_energies` (Ry), and provenance stamps
(WFN fingerprint, band window, V_NL treatment and sign). The velocity is
$\hat v = \partial_k H$, nonlocal pseudopotential term included
([QP velocity](../theory/qp-velocity.md)). `dipole_qsgw.h5`, written by an SC
run with a velocity head, has the same layout with $U^\dagger v U$ between the
final QP states, `band_energies` = $E_\mathrm{QP}$, and `basis = "qp"`; it
pairs only with `WFN_qp.h5`.

### 3.8 `gwjax.out`

The report, written by `gw.production_report.GWProductionReport`, has these
blocks in order: the run header (input, method); `CONFIGURATION PROVENANCE`
(every resolved choice and where it came from); `PROCESSOR ARCHITECTURE`;
`METHOD AND PHYSICAL PATHWAYS` (including the degenerate sets);
`NUMERICAL ENVIRONMENT`; `CRYSTAL SYMMETRY AND BRILLOUIN-ZONE SAMPLING`
(operations, wedge, time-reversal pathways); `BAND SPACES AND ENERGY COVERAGE`
(1-based band ranges); on SC runs one block per map (energies, gap, max
$|\Delta E|$, stage times, the head-velocity block) and the verdict line
`SC verdict: CONVERGED …`; `QUADRATURE WINDOWS`; `DYNAMIC SIGMA ENERGY COVERAGE`
(`COMPLETE` or `INCOMPLETE`); `FUNDAMENTAL GAP` (the full-matrix effective-H
gap, §3.6); `MAJOR-STAGE TIMING`; `MAJOR-STAGE DEVICE AND HOST MEMORY`;
`WARNINGS`; `OUTPUT FILES AND INPUTS`; and the closing line
`LORRAX GW calculation completed.` or `… REFUSED (… GATE <name>)`.
`LORRAX_DEBUG_PRINT=1` adds the full component output
([startup block](../environment/overview.md#startup-block)).

## 4. Conventions that move numbers {#conventions}

Each item is a place where the two codes, given the same WFN, compute a
different object by default, and how to make them compute the same one.

### 4.1 The Coulomb interaction at small q

**Point value or mini-BZ average.** Near $q + G = 0$ the bare interaction
$8\pi/|q+G|^2$ varies strongly across the mini Brillouin zone around each grid
point, and a point value misrepresents the zone it stands for. Both codes can
replace it by its average over the mini-zone, but they select different
elements:

- **BerkeleyGW**: `cell_average_cutoff` (Ry, in `sigma.inp`) is a cutoff on
  $|q+G|^2$; every element below it is averaged. The same cutoff decides at
  which $q$ the wings of $\varepsilon^{-1}$ are rescaled (`fixwings`, applied
  at $q_0$ and at every $q$ with $|q|^2$ below the cutoff). Its default is
  $10^{12}$, so every $q$ is averaged and wing-rescaled, on an untruncated
  run with `screening_semiconductor`; on every other run (any truncation,
  `screening_metal`, `screening_graphene`) it is $10^{-12}$, and only
  $q + G = 0$ is averaged (`Sigma/inread_sig.f90`).
- **LORRAX**: `mc_average_vcoul_body = true` (default, 3D bulk only) replaces,
  at every $q \ne 0$, the one element with the smallest $|q+G|$ by its
  mini-zone average, and rescales no wings (`gw.v_q_g_flat.v_head_fn_in_V`);
  `false` keeps point values. On a slab the flag has no effect.

The two codes compute the same bare interaction only with
`cell_average_cutoff 1.0d-12` and `mc_average_vcoul_body = false`; that is the
pairing every agreement number in §6 uses. On the static-COHSEX Si deck of
§6, LORRAX's default `true` against that BerkeleyGW run differs by 136 meV
(mean absolute) in $\Sigma_x$; `false` differs by 0.35 meV. LORRAX's `true`
is the nearer counterpart of BerkeleyGW's semiconductor default, not an
identical one. `cell_average_cutoff` is a `sigma.inp` key only: `epsilon.x`
stops on it as an unexpected keyword, and epsilon always averages only
$q + G = 0$.

**One bare interaction.** LORRAX builds one bare interaction $V_q$ in the
ISDF basis from the plane waves with $|q+G|^2 \le$ `bare_coulomb_cutoff`
(default the WFN's `ecutwfc`) and uses it both in $\Sigma_x$ and in
$W = (1 - V\chi_0)^{-1}V$. BerkeleyGW uses `bare_coulomb_cutoff` for
$\Sigma_x$ and the smaller dielectric cutoff (`epsilon_cutoff`) for the
screened part. Matching `bare_coulomb_cutoff` makes $\Sigma_x$ comparable;
$\Sigma_c$ agrees as BerkeleyGW's `epsilon_cutoff` and LORRAX's $N_\mu$ are
each converged.

**BerkeleyGW's own table.** `use_bgw_vcoul = true` with
`bgw_vcoul_file = <run>/vcoul` replaces LORRAX's $v(q+G)$ by the table
BerkeleyGW writes under `write_vcoul` (`gw.compute_vcoul`). Take it from the
`sigma.x` run: `epsilon.x` writes the table at its shifted $q_0$ without
averaging. The first row ($q = 0$, $G = 0$) is BerkeleyGW's averaged bare head.

### 4.2 The q → 0 head

At $q = 0$ the $G = 0$ elements of $v$ and $W$ diverge, and each code replaces
them by a finite cell average.

- **BerkeleyGW** computes $\varepsilon^{-1}$ at a small shifted $q_0$
  (the first `qpoints` line of `epsilon.inp`) and combines its head with the
  mini-BZ-averaged $v$.
- **LORRAX** builds the macroscopic response analytically from the velocity
  matrix elements of `dipole.h5`: a tensor $S_{ab}(\omega)$ with
  $\chi_{00}(q\to0) = q_a S_{ab} q_b$. The head values are cell averages,
  $v_h = \langle v\rangle_\mathrm{mBZ}$ and
  $W_h(\omega) = \langle v/(1 - v\,q\cdot S^\mathrm{eff}(\omega)\cdot q)\rangle_\mathrm{mBZ}$,
  and they enter $\Sigma$ as band-diagonal shifts
  ([four-current heads §3](../theory/four-current-head-corrections.md#charge-head)).
  `head_correction = full` (default) folds the local fields into $S$;
  `wcoul0_source = s_tensor` (default) takes $S$ from `dipole.h5`.

To impose BerkeleyGW's head, give the head scalars in Ry:

| deck key | value | where to read it |
|---|---|---|
| `vhead` | $v_h$ | row 1 of the sigma-side `vcoul` table |
| `whead_0freq` | $W_h(0) = v_h\,\varepsilon^{-1}_{00}(q_0, 0)$ | the `epshead` of `eps0mat.h5` times $v_h$ |
| `whead_imfreq` | $W_h(i\omega_p)$, GN-PPM only | not printed by a stock BerkeleyGW build |

An override applies only when `vhead` and the $W$ value of that frequency are
both set (`gw.head_correction.resolve_head_override`). A GN-PPM deck with
`vhead` and `whead_0freq` but no `whead_imfreq` mixes an overridden static
head with a computed dynamic one and prints `[head] MIXED HEAD`.
`wcoul0_source = epshead` instead reads `eps0mat.h5` (beside the deck)
directly; it is static only, reusing $\varepsilon^{-1}_{00}(0)$ at every
frequency, so it is right for COHSEX and wrong for a dynamic $\Sigma$.

**The plasmon-pole head.** On GN-PPM and HL-PPM, LORRAX fits one pole to the
correlation part of the head, $W^c_h = W_h - v_h$, at $\omega = 0$ and the probe
$z$ (`gw.head_correction.fit_head_ppm`), and adds its band-diagonal
$\Sigma^{c,\mathrm{head}}_n(\omega)$ to $\Sigma_c$
(`gw.ppm_pipeline._compute_analytic_head_diag`; equations in
[four-current heads §3.3](../theory/four-current-head-corrections.md#ppm-head)).
BerkeleyGW fits its GPP to the $G = G' = 0$ element of its $q_0$ dielectric
matrix like any other element. The two heads agree when the two $W_h$ agree at
both frequencies, which is why GN-PPM matching needs `whead_imfreq`.

### 4.3 The Coulomb hole: partial sum or static remainder

LORRAX's Coulomb hole is a partial sum over the `number_bands_sigma` bands,
$\Sigma^\mathrm{COH}_{mm'} = -\tfrac12 \sum_{n<N} \langle m n|W - v|n m'\rangle$
(`gw.cohsex_sigma`, band mask `sigma_sum`), and its dynamic $\Sigma_c$ is a
partial sum too (before the band extrapolation of §4.4); neither has a static
remainder. BerkeleyGW can add one, a correction of the partial sum toward the
exact static CH computed by closure: the whole difference on COHSEX, half of
it on a dynamic run. Which of BerkeleyGW's
numbers is then the partial sum depends on the mode and `exact_static_ch`
(`Sigma/mtxel_cor.f90`, `Sigma/write_result_hp.f90`, `Sigma/sigma_main.f90`):

| BerkeleyGW run | unprimed `CH`, `Sig`, `Eqp0`, `Eqp1`, and `eqp0.dat`, `eqp1.dat` | primed `CH′`, `Sig′`, `Eqp0′`, `Eqp1′` | compare LORRAX with |
|---|---|---|---|
| COHSEX, `exact_static_ch 1` (its default) | the exact static CH by closure | not printed | nothing: not a partial sum |
| COHSEX, `exact_static_ch 0` | partial sum plus remainder, which equals the exact static CH | the partial sum | the primed columns |
| plasmon pole, `exact_static_ch 0` (its default) | the partial sum | equal to the unprimed (printed for GN only) | either |
| plasmon pole, `exact_static_ch 1` | partial sum plus remainder | the partial sum | the primed columns |

The eqp files always hold the unprimed values, so in the rows with a
remainder take the comparison from `sigma_hp.log`, never from BerkeleyGW's
eqp files. A full-frequency run at its default `exact_static_ch 0` has no
remainder; its log has no CH column (§6 compares `Re Cor`).

### 4.4 The band sum: extrapolation

By default LORRAX extrapolates $\Sigma_c$ past `number_bands_sigma`
(`use_band_extrapolation`; [band extrapolation](../theory/band-extrapolation.md));
BerkeleyGW truncates the sum. The extrapolated `sigC`, `eqp0.dat` and
`eqp1.dat` are therefore not BerkeleyGW's quantity at the same band count. A
one-shot run with extrapolation on also writes the truncated numbers
(`sigC_raw`, `eqp0_raw`, `eqp1_raw` in `sigma_diag.dat`), which equal an
extrapolation-off run's `sigC` and eqp files. Compare those, or run with
`use_band_extrapolation = false`. $\chi_0$ is truncated at
`number_bands_chi` in both codes.

### 4.5 The exchange–correlation potential

BerkeleyGW forms $E^\mathrm{QP} = E_\mathrm{mf} + \Sigma - V_{xc}$ with
$V_{xc}$ from `vxc.dat` (or `kih.dat`). LORRAX reads neither: it builds
$H = T + V_\mathrm{ion} + V_H + \Sigma_{xc}$ from `kin_ion.h5` and a Hartree
potential it computes from the orbitals ([Hartree](../theory/hartree.md)). The
implied $V_{xc} = E_\mathrm{DFT} - \langle T + V_\mathrm{ion} + V_H\rangle$
agrees with QE's own $\langle V_{xc}\rangle$ to 0.85 meV at Γ and 2.95 meV at
M on monolayer CrI₃ (8×8 grid, bands 100–144). Both codes print absolute
energies on the DFT scale. Align each code to its own valence-band maximum
(or Fermi level) before comparing QP energies, because the head and the band
tail shift all states nearly rigidly.

### 4.6 The frequency derivative

LORRAX's eqp1 linearizes at $E_\mathrm{DFT}$, with $\partial_\omega\Sigma_c$
from a central difference at ±0.5 eV on the sampled ω grid (one-sided within
0.5 eV of a grid edge; `gw.eqp_bgw`). BerkeleyGW's plasmon-pole eqp1 uses
the finite difference set by `finite_difference_form` and
`finite_difference_spacing` (default 1.0 eV); its full-frequency eqp1 is
solved on its own frequency grid. This step moves eqp1 more than eqp0 (§6).

### 4.7 Symmetry and k order

LORRAX uses every symmetry operation and time-reversal flag in QE's
`data-file-schema.xml`, and writes energies on the WFN's irreducible wedge in
WFN order. BerkeleyGW writes the k points listed in `sigma.inp`. Join the two
by crystal coordinate modulo a reciprocal lattice vector.

## 5. Comparing like for like: a checklist

1. One WFN, one band count: `number_bands` = BerkeleyGW's `number_bands` in
   both `epsilon.inp` and `sigma.inp`, on a band edge that does not cut a
   degenerate multiplet.
2. `bare_coulomb_cutoff` equal in both; `sys_dim` matching the truncation.
3. `mc_average_vcoul_body` paired with `cell_average_cutoff` (§4.1).
4. `use_band_extrapolation = false`, or compare the `_raw` columns (§4.4).
5. COHSEX: BerkeleyGW `exact_static_ch 0`, compare against `CH′` and `Sig′`
   from `sigma_hp.log`; plasmon pole: leave `exact_static_ch` at 0 (§4.3).
6. Head: either accept the two codes' heads (they agree well on insulators,
   §6) or impose BerkeleyGW's with `vhead`/`whead_0freq` (and `whead_imfreq`
   for GN-PPM) (§4.2).
7. Centroids: enough for the target accuracy
   ([ISDF exchange accuracy](../theory/isdf-exchange-accuracy.md)).
8. Join by k coordinate, convert LORRAX's 0-based `n` in `sigma_diag.dat`,
   and align each code to its own VBM before comparing QP energies.

## 6. How closely they agree {#agreement}

**Full-frequency G₀W₀, Si.** Same WFN (diamond Si with spin–orbit coupling,
4×4×4 grid, 8 irreducible k, 25 Ry), same 100-band sum for $\chi_0$ and
$\Sigma$, $\Sigma$ for bands 1–24. BerkeleyGW: contour deformation (300 real
and 15 imaginary frequencies, 0.25 eV broadening, ε cutoff 25 Ry,
`cell_average_cutoff 1.0d-12`). LORRAX: one-shot shared-pole full frequency at
its defaults except `use_band_extrapolation = false` and
`mc_average_vcoul_body = false`; 1100 centroids.

| quantity | BerkeleyGW | LORRAX | difference |
|---|---|---|---|
| indirect gap, on the grid | 1.0798 eV | 1.0799 eV | +0.1 meV |
| direct gap | 3.0706 eV | 3.0721 eV | +1.5 meV |
| eqp1, 12 states within ±1 eV of midgap, each VBM-aligned | | | std 0.6 meV, max 1.7 meV |
| eqp1, 142 states within ±10 eV | | | std 6.4 meV, max 22.8 meV (eqp0: 2.5 / 8.7 meV) |

LORRAX took 55 s on one node of 4 A100 GPUs. With band extrapolation on (the
LORRAX default) the indirect gap is 1.1256 eV, +45.8 meV: the extrapolated
tail, which BerkeleyGW truncates, lowers every state by about 155 meV and
opens the gap. The remaining differences come from the frequency treatment
(contour deformation against the shared-pole W and its quadrature), the head
(§4.2), the ISDF basis and the derivative (§4.6).

**Static COHSEX, Si.** Si 4×4×4 (48 operations, 8 irreducible k), 25 Ry,
60 bands, 960 centroids, $\Sigma$ for 16 bands at all 64 k. BerkeleyGW:
`exact_static_ch 0`, `cell_average_cutoff 1.0d-12`; LORRAX:
`mc_average_vcoul_body = false`, BerkeleyGW's head given as `vhead` and
`whead_0freq`. Mean absolute (largest) differences: screened exchange
0.10 (0.27) meV against `X + (SX−X)`, Coulomb hole 0.35 (1.26) meV against
`CH′`, total 0.34 (1.07) meV against `Sig′`.

No metal and no slab has a matched comparison on the current code.

## 7. BSE and absorption {#bse}

The BSE reads the GW restart bundle (§3.5), so the GW run's Coulomb and head
conventions (§4.1–4.2) carry into the kernel. To compare with BerkeleyGW's
`kernel.x` and `absorption.x`:

- **Dipole operator.** BerkeleyGW's `use_momentum` uses $\hat p$ only. Write
  that with `python -m psp.get_dipole_mtxels -i deck.in --skip-vnl --out
  dipole_p_only.h5` and pass it as `--dipole`. The default `dipole.h5`
  includes the nonlocal term and is the physical velocity.
- **QP energies.** `--eqp` takes a BerkeleyGW-format `eqp1.dat`, LORRAX's or
  BerkeleyGW's. The reader (`file_io.restart_bundle.read_eqp_energies`)
  requires the file's k blocks to be the WFN's irreducible wedge in WFN order
  (it refuses otherwise) and to cover the BSE window. A BerkeleyGW file
  qualifies when `sigma.x` ran on the WFN's k list in that order. To compare
  kernels, give both codes the same QP energies.
- **Band counts.** `--n-val` and `--n-cond` count bands, spinor bands
  included, as BerkeleyGW's `number_val_bands` and `number_cond_bands` do. The
  occupied count comes from the WFN's `ifmax` or `--n-occ`. A window edge
  inside a degenerate multiplet (a Kramers pair under spin–orbit coupling)
  refuses unless `--band-degeneracy snap` widens it.
- **Spectrum.** The dipole of a transition $t = (v, c, k)$ is the position
  matrix element, obtained from the velocity in `dipole.h5`:
  $d^\alpha_t = \langle ck|\hat r_\alpha|vk\rangle = v^\alpha_{cv}/\big(i(E_c - E_v)\big)$
  (`bse.absorption_common.slice_dipole_to_bse_window` drops the constant
  factor $-i$, which cancels in every absorption quantity). With $H$ the TDA
  BSE Hamiltonian and $z = \omega + i\eta$,
  $$g^\alpha(z) = \frac{\langle d^\alpha|(z - H)^{-1}|d^\alpha\rangle}{\lVert d^\alpha\rVert^2}
  = \cfrac{1}{z - a_1 - \cfrac{b_1^2}{z - a_2 - \cfrac{b_2^2}{z - a_3 - \cdots}}},$$
  where $a_n$, $b_n$ are the Lanczos coefficients of $H$ started from
  $d^\alpha/\lVert d^\alpha\rVert$, and
  $\varepsilon_2^\alpha(\omega) = \frac{16\pi^2}{V N_k n_\mathrm{spin}
  n_\mathrm{spinor}}\lVert d^\alpha\rVert^2\,\big(-\mathrm{Im}\,g^\alpha(\omega+i\eta)/\pi\big)$,
  BerkeleyGW's Haydock formula (`bse.absorption_haydock`, `python -m
  bse.absorption_haydock`). Match `--eta-eV` to `energy_resolution`,
  `--n-iter` to `number_iterations`, and `--V-cell` (bohr³) to the cell;
  `--n-spin` and `--n-spinor` (defaults 1 and 2, the spinor case) enter the
  prefactor and must describe the WFN. The outputs
  `absorption_haydock_b1_eh.dat` (b2, b3) are the three Cartesian
  polarizations. Compare $\varepsilon_2(\omega)$, not raw sums of $|d|^2$.
- **Eigenvectors.** `bse.bse_jax --lanczos --tda --bse --write-eigs N
  --dipole FILE` writes `eigenvectors.h5` in BerkeleyGW's layout (eV, valence
  axis reversed; [BSE outputs](../architecture/bse.md#outputs)). An
  $\varepsilon_2$ summed over a truncated set of eigenstates converges slowly
  in peak height; compare full spectra through the Haydock route, and compare
  eigenvectors only through gauge-invariant quantities.
  `bse.absorption_common.eps2_from_exciton_dipoles` evaluates the
  sum-over-states $\varepsilon_2$ from the stored per-state dipoles; its
  energies and $\eta$ are in Ry.
