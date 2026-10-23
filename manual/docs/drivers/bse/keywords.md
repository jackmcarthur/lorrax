---
driver: bse
page: keywords
---

# Input keywords

`bse.bse_jax` Every deck key the BSE driver and `bse.absorption_haydock` read, and the four command-line flags.
{: .page-summary }

<div class="chain-map" data-driver="bse" data-page="keywords"></div>

The BSE settings are deck keys in the deck passed with `-i` (the GW deck), under BerkeleyGW's `absorption.x` names wherever the meaning is the same. A key is written `key = value`, or as in BerkeleyGW's inputs: `key value` without `=`, and a bare flag (`full_bse`) for `full_bse = true`, so `absorption.inp` lines can be pasted. The command line keeps `-i`, `--px`/`--py`, `--report-file` and `--out-prefix`. An old command-line flag refuses with its new name ("`--n-val` is now deck key `number_val_bands_fine`"), and so does an old deck key ("`bse_k_grid` is now `kgrid_fine`"). A key that only another route reads refuses on the route that would ignore it. Defaults are the drivers' own (`src/bse/bse_jax.py` `BSE_KEYS`, `src/bse/absorption_haydock.py` `HAYDOCK_KEYS`); the keys are listed in `src/gw/gw_config.py` `BSE_DECK_KEYS`.

## `bse.bse_jax` {: #bse-jax }

Required
{: .kw-group }

**Command line**

### `-i`, `--input` {: .kw data-type="path" data-default="required" }
The GW deck. Locates the ISDF tensor file, `wfn_file`, the centroid table and every key below.

**Deck keys**

### `number_val_bands_fine` {: .kw data-type="int" data-default="required" }
Valence bands of the BSE window, counted down from the highest occupied band, spinor bands included. BerkeleyGW: `number_val_bands_fine`.

Without `kgrid_fine` the window is solved on the GW grid, as BerkeleyGW's absorption with `skip_interpolation`. Clamped to the occupied count.
{: .kw-details }

### `number_cond_bands_fine` {: .kw data-type="int" data-default="required" }
Conduction bands of the BSE window, from the lowest unoccupied band up. BerkeleyGW: `number_cond_bands_fine`.

### `nval`, `ncond`, `number_bands` {: .kw data-type="int" data-default="required" }
The GW band window the bundle was written with. The bundle authenticates against them, so they must be the GW run's values.

### `centroids_file` {: .kw data-type="path" data-default="required" }
Centroid table; its content hash authenticates the bundle, and under `kgrid_fine` it is the interpolation's point set.

Optional
{: .kw-group }

**Kernel and spin**

### `full_bse` {: .kw data-type="bool" data-default="false" }
Solve the full $\begin{pmatrix}A&B\\-B^*&-A^*\end{pmatrix}$ problem. Off: the Tamm–Dancoff resonant block $A$, as in BerkeleyGW. `dipole_file`, the spectra and the Haydock recursion need the TDA.

### `spin_singlet`, `spin_triplet` {: .kw data-type="bool" data-default="singlet" }
Non-spinor WFN only: the singlet kernel $D + 2V - W$ (default) or the triplet kernel $D - W$. BerkeleyGW: `spin_singlet`, `spin_triplet`.

A spinor WFN has one kernel, $D + V - W$ (exchange weight 1), and either key refuses there. The spin case changes only the scalar exchange weight; every case runs the same matvec. A collinear spin-polarized WFN refuses.
{: .kw-details }

### `local_fields` {: .kw data-type="bool" data-default="false" }
Drop the screened direct term: $D + w V$, the RPA with local fields. BerkeleyGW: `local_fields`. Never the default.

### `spinor` {: .kw data-type="bool" data-default="the WFN's" }
Assertion only: refuses unless it matches the WFN's spinor count. BerkeleyGW: `spinor`.

### `extended_kernel` {: .kw data-type="bool" data-default="false" }
Read by [`bse.bsemat_cli`](../berkeleygw-export.md#bsemat) only. In a solver deck it prints one NOTE: LORRAX does not interpolate the kernel, and `full_bse` selects the non-TDA problem.

**Bands and energies**

### `highest_occupied_band` {: .kw data-type="int" data-default="WFN ifmax" }
Occupied-band count that places the window $[n_{\mathrm{occ}} - n_v,\, n_{\mathrm{occ}} + n_c)$. BerkeleyGW: `highest_occupied_band`.

### `eqp_corrections`, `eqp_co_corrections` {: .kw data-type="bool or path" data-default="—" }
QP energies in `eqp1.dat` format on the GW grid's irreducible wedge. BerkeleyGW: `eqp_corrections` (no fine grid), `eqp_co_corrections` (with `kgrid_fine`, where the energies enter the band interpolation).

A bare key reads BerkeleyGW's file name beside the deck (`eqp.dat`, `eqp_co.dat`); a value names the file (LORRAX's GW writes `eqp1.dat`). `eqp_co_corrections` without `kgrid_fine`, or `eqp_corrections` with it, refuses. Without a fine grid the window is re-sliced on the corrected spectrum.
{: .kw-details }

### `qp_rotations_file` {: .kw data-type="path" data-default="—" }
The GW run's `qp_wfn_rotations.h5`: solve in the eigenbasis of its $H_{QP}$; the window counts QP states. Exclusive with the eqp keys; refuses `kgrid_fine`.

### `wfn_file` {: .kw data-type="path" data-default="WFN.h5" }
WFN for the occupied count, the cell volume, the spinor count, bundle authentication and the band interpolation; relative to the deck.

### `band_degeneracy` {: .kw data-type="enum" data-default="strict" }
What happens when a window edge cuts a degenerate multiplet (a Kramers pair under spin–orbit coupling). `strict` refuses and names counts that work; `snap` widens the window outward; `off` proceeds.

`strict` `snap` `off`
{: .kw-values }

### `degeneracy_check_override` {: .kw data-type="bool" data-default="false" }
BerkeleyGW's name for `band_degeneracy = off`.

### `degeneracy_tol_ry` {: .kw data-type="float" data-default="7.3499e-05 (1 meV)" }
Energy tolerance in Ry for "same multiplet".

**Fine grid**
{: #fine-grid }

### `kgrid_fine` {: .kw data-type="int[3]" data-default="—" }
The fine BSE grid $\bk_{\mathrm{fi}}$, `NX NY NZ`. Replaces BerkeleyGW's `WFN_fi`: ψ, $E^{\mathrm{QP}}$, the dipoles and $W$ move to $\bk_{\mathrm{fi}}$ before the solve, by the [H-transform](../../utilities/htransform-fine-grid.md) and the coarse-to-fine kernel interpolation.

Each axis must be at least the coarse one; the two grids need not nest.
{: .kw-details }

### `kgrid_fine_shift` {: .kw data-type="float[3]" data-default="0 0 0" }
kgrid.x's offset $\delta\bk$ of the fine grid, in fractions of one fine-grid step: $\bk_i = (i + \delta\bk)/N$. `0.5 0.5 0.5` is half a step; BerkeleyGW's Si example uses `0.47 0.37 0.31`.

A shift breaks the grid's symmetry and samples the zone evenly. Only $\bk - \bk'$ enters the direct term, so $W$ stays on the unshifted fine $\bq$ grid; the $\bQ = 0$ exchange tile does not depend on $\bk$. `eigenvectors.h5` holds the shifted $\bk$. Values are reduced modulo 1. [Exciton bands](../exciton-bands/keywords.md) and TD-aGW refuse a shift.
{: .kw-details }

### `number_cond_bands_coarse` {: .kw data-type="int" data-default="ncond" }
Conduction bands of the coarse (GW) grid that enter the band interpolation. BerkeleyGW: `number_cond_bands_coarse`.

Coarse much wider than fine is the expected case: the GW grid carries the conduction sums, the BSE is a near-gap problem. Fewer than `number_cond_bands_fine` refuses; fewer than four above it warns, because the top of the interpolated window then lies in the f-transform's shoulder. Above `ncond`, the extra bands are read from the WFN as guard bands.
{: .kw-details }

### `number_val_bands_coarse` {: .kw data-type="int" data-default="—" }
Accepted and ignored, with a warning: the H-transform always fits every valence band of the GW window.

### `w_head_densify` {: .kw data-type="enum" data-default="srlr" }
Long-range model split off $W$ before the densification under `kgrid_fine`. `srlr`, the only model: the plane-wave channels $\bq+\bG_a\to0$ carry the long range from the bundle's `W0_lr`, fitted on the coarse grid and added back at every fine $\bq$. It needs a GW restart with `W0_lr` (a static Dyson $W$: `cohsex`, `gn_ppm`, `hl_ppm`). `c1` and `legacy` were removed and refuse with one line. See [Short- and long-range W](../../theory/w-range-splitting.md).

`srlr`
{: .kw-values }

### `head_minibz_average` {: .kw data-type="bool" data-default="false" }
Read under `kgrid_fine`: rebuilds the $V_{\bq=0}$ tile with the fine grid's mini-BZ head. Must match the GW run.

### `vhead`, `whead_0freq` {: .kw data-type="float" data-default="—" }
Overrides of the bundle's $V^{\mathrm{head}}$ and $W^{\mathrm{head}}(\omega = 0)$ (Ry·bohr³). Cross-code validation knobs.

### `wfn_fi_q_chunk` {: .kw data-type="int" data-default="0" }
Fine-grid points per batch of the band interpolation; `0` uses the coarse $N_k$. Memory only.

### `htransform_qr_eps` {: .kw data-type="float" data-default="1e-3" }
Basis control of the band interpolation, on the [htransform](../../utilities/htransform.md#htransform_qr_eps) page.

**Solver**

### `number_eigenvalues` {: .kw data-type="int" data-default="5" }
Number of lowest eigenpairs. BerkeleyGW: `number_eigenvalues`.

### `solver` {: .kw data-type="enum" data-default="davidson" }
Eigensolver of the TDA problem. `davidson`: block Davidson with the exact diagonal preconditioner, stopped when every root's residual is below `davidson_tol_mev`. `block_lanczos`: block Lanczos at a fixed Krylov dimension or to `lanczos_rtol`. `trlan`: thick-restart Lanczos within `trlan_m_max` slots. A key that only another solver reads refuses. The full BSE (`full_bse`) uses the dense non-TDA solver.

BerkeleyGW's `diagonalization` (dense), `diagonalization_primme` and `lanczos` (a Lanczos spectrum) are other methods, so their names are not used.
{: .kw-details }

`davidson` `block_lanczos` `trlan`
{: .kw-values }

### `solver_max_iter` {: .kw data-type="int" data-default="max(30, min(200, N/2))" }
Iteration budget of every solver, with $N$ the transition count: Davidson's cap on block iterations, Lanczos's Krylov dimension, TRLan's matvec budget.

### `davidson_tol_mev` {: .kw data-type="float" data-default="0.01" }
Davidson stopping rule in meV: every root's residual $\lVert Hx - \Omega x\rVert$ below it. A solve that reaches `solver_max_iter` first refuses (`GATE davidson_converged`).

### `davidson_precond`, `davidson_margin`, `davidson_margin_tol_mev`, `davidson_m_max`, `davidson_olsen`, `davidson_eps_shift` {: .kw data-type="mixed" data-default="exact, 8, davidson_tol_mev, 10·n_eig, false, 1e-3" }
Davidson internals: the preconditioner diagonal (`exact`, `bare`, `auto`), extra unreported roots and the residual at which they lock, the subspace size, the Olsen correction, the regularization in Ry.

### `block_size`, `lanczos_rtol`, `lanczos_check_every`, `lanczos_n_reorth` {: .kw data-type="mixed" data-default="1, 0, 4, −1" }
`block_lanczos` internals: block width, convergence threshold (0 = fixed dimension) and its cadence, reorthogonalization window (−1 = full; keep it for spinor spectra).

### `trlan_m_max`, `trlan_n_keep` {: .kw data-type="int" data-default="max(3 n_eig, 60), n_eig + 10" }
`trlan` internals: basis slots and Ritz vectors kept across a restart.

**Dipoles, spectra and output files**

### `dipole_file` {: .kw data-type="path" data-default="—" }
`dipole.h5` (`psp.get_dipole_mtxels`), relative to the deck. With it, and the TDA, each state's $\langle 0\lvert\hat r\rvert S\rangle$ goes to `eigenvalues_b1.dat`…`_b3.dat` (`eigenvalues.dat` along a deck `polarization`), the spectra below are written, and the Haydock recursion is seeded. Under `kgrid_fine` it is the coarse WFN's file, carried through the band interpolation.

### `use_momentum`, `use_velocity` {: .kw data-type="bool" data-default="—" }
BerkeleyGW's operator choice, checked against `dipole_file`'s `skip_vnl` stamp. LORRAX chooses the operator when the dipole file is made: `psp.get_dipole_mtxels --skip-vnl` is the momentum $p$ alone, the default is the velocity $p + i[V_{\mathrm{NL}}, r]$.

### `dipole_position_fix` {: .kw data-type="bool" data-default="false" }
Under `kgrid_fine` with QP energies: the dipoles are the exact $\langle c\lvert r\rvert v\rangle$ of the fine QP states instead of $v/\Delta E$ of their Rayleigh quotients. Opt in for strongly band-dependent QP shifts.

### `polarization` {: .kw data-type="float[3]" data-default="unset" }
Light polarization $p_1\,p_2\,p_3$ along $\mathbf b_1\,\mathbf b_2\,\mathbf b_3$, as in BerkeleyGW. The Cartesian vector $\sum_i p_i \mathbf b_i$ is built and normalized. Unset: one spectrum per Cartesian axis (files `_b1`, `_b2`, `_b3`); set: one spectrum (files without the suffix).

### `energy_resolution` {: .kw data-type="float" data-default="0.1" }
Broadening in eV. BerkeleyGW: `energy_resolution`.

### `gaussian_broadening`, `lorentzian_broadening` {: .kw data-type="bool" data-default="gaussian" }
Profile of the spectra from the eigensolve: a Gaussian of standard deviation `energy_resolution` (the default, as BerkeleyGW's diagonalization) or a Lorentzian of half width `energy_resolution`. The Haydock spectrum is Lorentzian only.

### `delta_frequency`, `omega_min_ev`, `omega_max_ev` {: .kw data-type="float" data-default="0.01, 0, 15" }
Frequency grid of the spectra in eV, endpoints included. BerkeleyGW: `delta_frequency`.

### `no_eps1` {: .kw data-type="bool" data-default="false" }
Skips Kramers–Kronig; the $\varepsilon_1$ column is `1`.

With `dipole_file` the eigensolve writes BerkeleyGW's `absorption_eh.dat` (the computed states, complete below the highest) and `absorption_noeh.dat` (the window's transitions): $\omega$, $\varepsilon_2$, $\varepsilon_1$ and the density of states.
{: .kw-details }

### `write_eigenvectors` {: .kw data-type="int" data-default="0" }
States written to `eigenvectors.h5`: 0 none, $N > 0$ the lowest $N$, $N < 0$ every computed state. BerkeleyGW: `write_eigenvectors`.

### `helicity_table` {: .kw data-type="bool" data-default="false" }
With `dipole_file`: each state's in-plane circular strengths in the report.

### `haydock`, `number_iterations` {: .kw data-type="bool, int" data-default="false, 200" }
After the eigensolve, `number_iterations` Haydock steps on the same payload in the same process; writes `absorption_haydock.h5` and `absorption_haydock*_eh.dat`. BerkeleyGW: `haydock`, `number_iterations`. Needs the TDA and `dipole_file`.

**Exciton density of states**

### `kpm_dos` {: .kw data-type="bool" data-default="false" }
Computes the KPM density of states of $H$ and exits, with `kpm_window_count`, `kpm_n_moments`, `kpm_n_random`, `kpm_n_lanczos`, `kpm_emin_ev`, `kpm_emax_ev`, `kpm_plot_file`.

**Command line**

### `--px`, `--py` {: .kw data-type="int" data-default="startup mesh" }
Device mesh over the $(c, v)$ transition axes. Given together, square, and using every device.

### `--report-file` {: .kw data-type="path" data-default="bse.out" }
Run report.

### `--out-prefix` {: .kw data-type="path" data-default="absorption" }
Prefix of the absorption files from the eigensolve (`absorption_haydock` for the Haydock files).

## Absorption (`bse.absorption_haydock`)

A separate entry point: the TDA screened recursion ($D + w V - W$), at least two devices, band degeneracy `strict`. It reads `number_val_bands_fine`, `number_cond_bands_fine`, `highest_occupied_band`, the eqp keys, `dipole_file` (default `dipole.h5` beside the deck), `dipole_position_fix`, `use_momentum`/`use_velocity`, the spin keys, `number_iterations`, `energy_resolution`, `delta_frequency`, `omega_min_ev`, `omega_max_ev`, `no_eps1`, `polarization` and the fine-grid keys, as above. The cell volume and the spinor count are read from the WFN. The command line is `-i` and `--out-prefix` (default `absorption_haydock`). `gaussian_broadening` and `local_fields` refuse.

## Renamed and retired

| Old | Now |
|---|---|
| `--n-val`, `--n-cond`, `--n-occ` | `number_val_bands_fine`, `number_cond_bands_fine`, `highest_occupied_band` |
| `--tda` (default off) | TDA is the default; `full_bse = true` for the full BSE |
| `--bse`, `--rpa` (default RPA) | the screened kernel is the default; `local_fields = true` for the RPA |
| `--eqp` | `eqp_corrections` / `eqp_co_corrections` |
| `--dipole`, `--qp-rotations` | `dipole_file`, `qp_rotations_file` |
| `--n-eig`, `--write-eigs` | `number_eigenvalues`, `write_eigenvectors` (BerkeleyGW's sign convention) |
| `--solver lanczos`, `--max-lanczos-iter`, `--n-reorth` | `solver = block_lanczos`, `solver_max_iter`, `lanczos_n_reorth` |
| `--haydock-iter`, `--haydock-eta-ev`, `--haydock-n-omega`, `--haydock-pol` | `haydock` + `number_iterations`, `energy_resolution`, `delta_frequency`, `polarization` |
| `--eta-eV`, `--n-iter`, `--n-omega`, `--pol` (Haydock) | `energy_resolution`, `number_iterations`, `delta_frequency`, `polarization` |
| `--V-cell`, `--n-spinor`, `--n-spin` (Haydock) | removed: read from the WFN |
| `--lanczos` | removed (it changed nothing) |
| other `--davidson-*`, `--trlan-*`, `--lanczos-*`, `--kpm-*` | the deck key of the same name in snake case |
| `bse_k_grid` | `kgrid_fine` |
| `--ring-timing`, `w_head_densify = legacy`, `LORRAX_LANCZOS_REORTH` | retired |
