---
driver: bse
page: files
---

# Files

`bse.bse_jax` Formats of the files the BSE driver and `bse.absorption_haydock` read and write.
{: .page-summary }

<div class="chain-map" data-driver="bse" data-page="files"></div>

## Inputs {: #inputs }

Relative `wfn_file` and `centroids_file` resolve against the deck's directory; paths given on the command line resolve against the working directory.

| File | Deck key | Produced by | Note |
|---|---|---|---|
| deck | `-i` | user | The GW deck. Its directory locates the restart bundle. |
| `isdf_tensors_<N_mu>.h5` | — | gw | Restart bundle. Exactly one, beside the deck or in its `tmp/`. Every route; read, never written. |
| `WFN.h5` | `wfn_file` | nscf, or DFT + pw2bgw | Occupied count, cell volume, bundle authentication; under `kgrid_fine`, the band interpolation. `WFN_qp.h5` from a QSGW run is accepted. |
| `centroids_frac_<N>.txt` | `centroids_file` | centroids | Content hash authenticates the bundle; under `bse_k_grid`, the interpolation points. On a downfolded bundle, the parent table recorded in the bundle. |
| `eqp1.dat` | `eqp_corrections` | gw | Eigensolve route and `bse.absorption_haydock`. Diagonal QP energies on the irreducible wedge, unfolded on load. Format: [GW files](../gw/files.md#eqp). |
| `dipole.h5` | `dipole_file` | dipole | Velocity matrix elements and `band_energies`. `bse.bse_jax`: the per-state dipoles, `absorption_eh.dat`/`absorption_noeh.dat` and the Haydock seed (TDA). `bse.absorption_haydock`: always, default `dipole.h5`. |
| `dipole_qsgw.h5` | `dipole_file` | gw (QSGW) | Same layout, stamped `basis = qp`; pairs only with `WFN_qp.h5`. |
| `dipole.h5` beside the deck | — | dipole | The head tensor $S$ when the bundle has no `S_cart_head`: under `kgrid_fine`, and in a $W(0)$ rebuild. |
| `zeta_q.h5` beside the bundle | — | gw | Only for a $W(0)$ rebuild, and for `head_minibz_average = true` under `kgrid_fine`. |
| `eps0mat.h5` beside the deck | — | BerkeleyGW `epsilon.x` | Only for a $W(0)$ rebuild with `wcoul0_source = epshead`. |

No file from the [htransform](../../utilities/htransform.md) utility is read: under `kgrid_fine` the band interpolation runs inside the driver ([Theory §1.3](theory.md#grid-bands)).

## What is read from the bundle {: #bundle }

The bundle's full schema belongs to the [GW Files](../gw/files.md#restart) page. The BSE reads these names from it:

| Dataset | Read for |
|---|---|
| `lorrax_io_committed` | Write-commit flag; an uncommitted bundle refuses. |
| `psi_parent_y`, `psi_parent_y_mun`, `psi_parent_k_rows`, `band_window`, `band_window_schema`, `enk_full`, `kgrid` | Must all be present: the current raw-parent format. |
| `enk_full` | $E_{n\bk}$ in Ry on the full coarse grid, shape $(N_k, N_b)$. |
| `kgrid` | The coarse grid $(N_x, N_y, N_z)$. |
| `psi_parent_y` | $\psi_{n\bk}(\br_\mu)$ on the parent k rows, $(k_{\mathrm{parent}}, N_b, n_{\mathrm{spinor}}, N_\mu)$; only the window bands are read and unfolded to the full grid. |
| `psi_parent_k_rows` | Parent rows, checked against the WFN's wedge. |
| `band_window`, `band_window_split`, `band_window_carrier`, `n_rmu_logical` | Authentication against the deck's band keys and $N_\mu$. |
| `zeta_fit_windows` | The ζ-fit legs that must hold the BSE window; older bundles fall back on `band_window`. |
| `V_qmunu` (`V_ready`) | The $V_{\bq=0}(\mu,\nu)$ tile only. |
| `W0_qmunu` (`W0_ready`) | $W_\bq(\mu,\nu;0)$, unfolded from its q parents to the full grid on load. Absent or not ready: rebuilt in memory. |
| `G0_mu_nu`, `vhead`, `whead` | Head vector $g_0(\mu)$ and the head scalars, Ry·bohr³. |
| `S_cart_head` | Head tensor $S$, under `kgrid_fine`. |
| `W0_lr/` | Long-range channels, under `w_head_densify = srlr`. |
| `qp_state_source_provenance` | Fingerprint of the WFN the bundle's ψ and $E$ came from. |
| `downfold_provenance/` | Parent centroid table and kept rows of a downfolded bundle. |
| `@centroids_charge_md5` | Centroid content hash. |

## Outputs {: #outputs }

<div class="file-list" markdown>

`bse.out`
:   Run report of the eigensolve route (`--report-file`): pathways, environment, window, exciton energies in eV, stage timings and the files read and written.

`eigenvalues.dat`, `eigenvalues_b1.dat` … `eigenvalues_b3.dat`
:   Eigensolve route: every root in BerkeleyGW's format, under its names: `eigenvalues.dat` for one polarization (a deck `polarization`, or no `dipole_file`), `eigenvalues_b1.dat` … `_b3.dat` for the three Cartesian axes. A triplet run writes the same files and rows, with BerkeleyGW's ` mg ` in the column header.

`absorption_eh.dat`, `absorption_noeh.dat` (or `absorption_b1_eh.dat` … `_b3_noeh.dat`)
:   Eigensolve route with `dipole_file` and the TDA: BerkeleyGW's four header lines and `4f16.9` rows ($\omega$, $\varepsilon_2$, $\varepsilon_1$, DOS), named per polarization as the eigenvalue files.

`eigenvectors.h5`
:   `write_eigenvectors`: the lowest $N$ exciton energies and amplitudes in BerkeleyGW's layout, read unchanged by its readers: `exciton_header/versionnumber`, 32-bit integer scalars, `kpts` $(N_k, 3)$, the four eigenvector datasets $(N_Q, N, N_k, n_c, n_v, n_s, 2)$ with the valence index counted down from the VBM, `spin_kernel` 3 spinor, 1 singlet, 0 triplet, 2 local fields, and the deck WFN's `mf_header` (on a `kgrid_fine` grid with the k list replaced and the per-k band data left out). Under the TDA the left and de-excitation datasets are zeros; with `full_bse` they hold $X$, $-Y$ and $Y$. With `dipole_file` it also holds the per-state dipoles as `exciton_data/dipoles`, a LORRAX addition that BerkeleyGW readers ignore.

`bse_dos_kpm.png`, `bse_dos_kpm.npz`
:   `kpm_dos = true`: the exciton density of states, its Chebyshev moments and the window edges (`kpm_plot_file`).

`absorption_haydock.h5`
:   `bse.absorption_haydock`: $\varepsilon_2$, $\varepsilon_1$ and the independent-particle spectrum on the frequency grid, with the continued-fraction coefficients.

`absorption_haydock_b1_eh.dat`, `…_b2_eh.dat`, `…_b3_eh.dat`
:   `bse.absorption_haydock`: one four-column spectrum per Cartesian polarization $x$, $y$, $z$.

standard output
:   Progress and timing on every route.

</div>

Rank 0 writes `bse.out`, the eigenvalue files, `eigenvectors.h5` and the Haydock files. `eigenvectors.h5` goes to the working directory and is overwritten. The restart bundle is never modified, and no cache is written.

The eigenvalue files, `eigenvectors.h5` and the `absorption_haydock_b{1,2,3}_eh.dat` spectra are written in BerkeleyGW's formats, which the BerkeleyGW manual documents ([absorption outputs](https://manual.berkeleygw.org/4.0/absorption-overview/), [`eigenvectors.h5`](https://manual.berkeleygw.org/4.0/eigenvectors_h5_spec/)). The rest of this page describes the files that are specific to LORRAX.

## bse.out {: #report }

Plain text, written on the eigensolve route only. Sections in order: the input deck; the device mesh; the pathways (Hamiltonian, interaction, eigensolver, iteration mode, band boundary); the WFN environment and k sampling; under `kgrid_fine`, the band-interpolation compression and quality receipts; the window (valence and conduction counts, padded and physical transition counts, requested roots); the exciton spectrum, one line `S<nnn>  <energy in eV>` per root; the stage timings (restart input, eigensolve, exciton dipoles, eigenvector write); the files read and written.

## bse_dos_kpm.npz {: #kpm }

NumPy archive beside the plot, same base name. The plot shows $\rho(E)$ with the window edges, and the raw and Jackson-damped moment decay.

| Key | Content |
|---|---|
| `mu_raw`, `mu_damped` | Chebyshev moments $\mu_n$, $n = 0 \ldots M$, before and after Jackson damping. |
| `jackson_sigma` | The Jackson factors $g_n$. |
| `E_grid_eV` | 2000 energies, eV. |
| `rho` | $\rho(E)$ per eV, clipped at zero; normalized to unit area ($\mu_0 \approx 1$). |
| `e_center_ry`, `half_width_ry` | The scaling of $H$ onto $[-1, 1]$, Ry. |
| `e_min_ry`, `e_max_ry` | Spectral bounds from Lanczos (or `kpm_emin_ev`, `kpm_emax_ev`), Ry. |
| `n_moments`, `n_random` | $M$ and $R$. |

## absorption_haydock.h5 {: #haydock-h5 }

Written as `<--out-prefix>.h5` by `absorption_common.write_absorption_h5`. The polarization axis is Cartesian $x, y, z$.

```tree
absorption_haydock.h5
  @route                     str = haydock
  @V_cell_bohr3              float
  @N_k                       int
  @eta_eV                    float
  @n_spin                    int
  @n_spinor                  int
  @n_iter                    int
  omega_eV                   float64 (n_omega)
  eps2                       float64 (n_omega, 3)
  eps1                       float64 (n_omega, 3)
  jdos                       float64 (n_omega, 3)
  haydock_alphas             float64 (3, n_iter)
  haydock_betas              float64 (3, n_iter)
  haydock_norms              float64 (3)
```

| Name | Content |
|---|---|
| `@V_cell_bohr3`, `@eta_eV`, `@n_spin`, `@n_spinor`, `@n_iter` | The run's cell volume and spinor count (from the WFN; `n_spin` = 1), `energy_resolution` and `number_iterations`. |
| `@N_k` | $N_k$ of the BSE grid, in the prefactor $\kappa$. |
| `omega_eV` | The frequency grid, eV. |
| `eps2`, `eps1`, `jdos` | Columns 2, 3 and 4 of the `.dat` files; `eps1` is ones with `--no-eps1`. |
| `haydock_alphas` | $a_n = \langle s_n\lvert H\rvert s_n\rangle$, Ry. |
| `haydock_betas` | $b_n = \lVert r_n\rVert$, Ry; $b_n$ couples steps $n$ and $n+1$, so the last one is not used. |
| `haydock_norms` | $\lVert d^\alpha\rVert$, bohr. |

The `eps2`, `eps1` and `jdos` arrays are columns 2, 3 and 4 of the `.dat` spectra. The spectrum can be re-evaluated on another grid or $\eta$ from the stored coefficients, without the matvec: $\varepsilon^\alpha_2(\omega) = -\frac{\kappa}{\pi}\lVert d^\alpha\rVert^2\,\mathrm{Im}\,g^\alpha(\omega + i\eta)$ with $g^\alpha$ the continued fraction in $a_n$, $b_n$ ([Theory §5.4](theory.md#abs-haydock)).

