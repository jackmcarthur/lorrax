---
driver: gw
page: keywords
---

# Input keywords

`gw.gw_jax` Every deck key gw_jax reads, grouped by what it controls.
{: .page-summary }

<div class="chain-map" data-driver="gw" data-page="keywords"></div>

Keys are written `key = value`, one per line, under a `[cohsex]` section header in the deck passed with `-i`; keys are case-insensitive, a comment starts with `#`, and an unknown or retired key refuses with its line number. Energies are in Ry unless the key name ends in `_ev`. Defaults are read from the keyword schema (`src/gw/gw_config.py`, `_DEFAULTS`).

Required
{: .kw-group }

**Bands**
{: #bands }

Give `number_bands`, and the Σ window either as `nval` and `ncond` or as `number_bands_protected`.

### `number_bands` {: .kw data-type="int" data-default="required" }
Total bands in the GW calculation: the top of the $\chi^0$ and Σ band sums. Alias `nband`.

Sets both sums unless `number_bands_chi` or `number_bands_sigma` is given; the loaded band range is rounded up to the process count with zero pad bands. Choose it for $\chi^0$, which is not extrapolated, with $N \ge 2n_\mathrm{occ}$.
{: .kw-details }

### `nval` {: .kw data-type="int" data-default="required" }
Required unless `number_bands_protected` is given. Valence bands in the requested QP window, $b_1 = n_\mathrm{elec} - n_\mathrm{val}$.

The Σ/QP matrix always starts at band 0; `nval` sets the ζ fit's right window and, on dynamic runs, the semicore class.
{: .kw-details }

### `ncond` {: .kw data-type="int" data-default="required" }
Required unless `number_bands_protected` is given. Conduction bands in the Σ/QP window $[0,\, n_\mathrm{elec} + n_\mathrm{cond})$.

Under QSGW this window is the rotating QP matrix; bands above it form a scissored tail.
{: .kw-details }

### `number_bands_protected` {: .kw data-type="int" data-default="instead of nval, ncond" }
The Σ window as one count instead of `nval` and `ncond`: active bands for the Σ matrix elements, from the lowest valence band up to this total ($b_3$).

Giving it beside `nval` or `ncond` refuses (`GATE band_request_forms`); it must lie between the occupied count and `number_bands`. A convergence parameter of QSGW ([QSGW](../../theory/qsgw.md)).
{: .kw-details }

**System and occupations**
{: #system }

### `sys_dim` {: .kw data-type="int" data-default="required" }
Coulomb truncation: `3` bulk, `2` slab with vacuum along $c$.

A deck without it refuses (`GATE sys_dim_required`). It enters $V_\bq$, $W$ and $V^{\mathrm H}$.
{: .kw-details }

`3` `2`
{: .kw-values }

**Files**

### `centroids_file` {: .kw data-type="path" data-default="required" }
ISDF point table from the [centroids](../../utilities/centroids.md) utility.

Optional
{: .kw-group }

**Screening**
{: #screening }

### `compute_mode` {: .kw data-type="enum" data-default="auto" }
Σ treatment. See [frequency treatments](../../theory/frequency-treatments.md).

`auto` resolves from the legacy keys to `cohsex` (or `x_only`, `gn_ppm`, `hl_ppm`) and never selects `mpa`. A metal admits only `mpa` and `x_only`.
{: .kw-details }

`x_only` `cohsex` `gn_ppm` `hl_ppm` `mpa`
{: .kw-values }

### `sigma_w_model` {: .kw data-type="enum" data-default="mpa" }
Model of $W^c$ under `compute_mode = mpa`.

`tangential` is the production model ([Tangential W model](../../theory/tangential-w.md)); `mpa` fits poles element by element. Naming the key with another `compute_mode` refuses. The old value `shared_pole` is still accepted, with a note in the log.
{: .kw-details }

`tangential` `mpa`
{: .kw-values }

### `sigma_w_accuracy` {: .kw data-type="enum" data-default="production" }
Tier of the tangential recipe.

`relaxed` also sets an omitted `sigma_quadrature_eps` to `5e-4`. Requires `sigma_w_model = tangential`.
{: .kw-details }

`production` `relaxed`
{: .kw-values }

### `sigma_frequency_integration` {: .kw data-type="enum" data-default="pole_model" }
Frequency integration of $\Sigma^c$.

`pole_model` is the time-node route of the `sigma_w_model` pole models. `contour_deformation` integrates on the imaginary axis and adds the real-axis residues, on the exact ISDF $W(z)$; it needs `compute_mode = mpa`, `head_correction = off`, `use_band_extrapolation = false`, `linalg = local` and `screening_diagrams = w_rpa`, and refuses the `sigma_w_*`, `mpa_*`, `write_w` and `write_poles` keys.
{: .kw-details }

`pole_model` `contour_deformation`
{: .kw-values }

### `screening_diagrams` {: .kw data-type="enum" data-default="w_rpa" }
Diagrams summed by $W$: the RPA Dyson series, the ladder-corrected resolvent, or the RPA resolvent.

`w_bse` runs one-shot under `cohsex`, `gn_ppm` and elementwise `mpa` on a time-reversal-symmetric WFN; `w_rpa_resolvent` under `cohsex` and `gn_ppm` only. Both need an orbit-closed centroid set.
{: .kw-details }

`w_rpa` `w_bse` `w_rpa_resolvent`
{: .kw-values }

### `write_poles` {: .kw data-type="bool" data-default="false" }
Exports the stored tangential factors $\tilde b_{j\bq}(\mu)=\sqrt{2\Omega_{j\bq}}\,b_{j\bq}(\mu)$ and $\Omega^2_{j\bq}$ to `tmp/mpa/<map>_poles.h5`, with $b$ the residue-normalised factor of [Tangential W model §2.1](../../theory/tangential-w.md#model-form).

A static-W consumer needs nothing else, since $W(0) = V - 2b\,\Omega^{-1} b^\dagger = V - \tilde b\,\Omega^{-2} \tilde b^\dagger$. Requires `compute_mode = mpa` and `sigma_w_model = tangential`.
{: .kw-details }

### `minimax_target_error` {: .kw data-type="float" data-default="1e-6" }
Accuracy target of the χ⁰ time rules, in a currency set by each rule.

Absolute error in Ry⁻¹ for the static rule, the HL probe and the insulating MPA $i\varpi_2$ sample; relative for the GN probe certificate and, against $1/\varpi$, for the MPA line samples. The tangential bank ignores it and uses its tier's tolerance ([Rules and keys](../../theory/time-frequency.md#frame-keys)).
{: .kw-details }

### `minimax_max_nodes` {: .kw data-type="int" data-default="64" }
Node cap of the static, HL-probe and insulating-MPA rules; order cap of one panel of an MPA line rule.

A static rule that misses its target at the cap refuses. The GN probe and the tangential bank ignore it; the GN probe stops at 64 nodes ([minimax](../../theory/time-frequency.md#frame-keys)).
{: .kw-details }

### `minimax_energy_reference` {: .kw data-type="enum" data-default="midgap" }
Energy zero of the band energies in the χ⁰ time rules. It cancels from $\chi^0$.

`none` (aliases `raw`, `zero`) uses the raw energies. A number in the deck reaches the resolver as a string and refuses when the first rule is built ([minimax](../../theory/time-frequency.md#frame-keys)).
{: .kw-details }

`midgap` `vbm` `cbm` `none`
{: .kw-values }

### `ladder_probe_chunk` {: .kw data-type="int" data-default="0" }
Probe columns per block of the ladder solve (`w_bse`, `w_rpa_resolvent`).

`0` solves one block; a positive value bounds memory and changes no value.
{: .kw-details }

### `sigma_w_support_sites_ev` {: .kw data-type="string" data-default="(empty)" }
Replaces both tangential support ladders, written `"<line list> | <imaginary list>"` in eV.

A support study dial, announced as a debug warning; the sites enter the recipe hash.
{: .kw-details }

### `write_w` {: .kw data-type="bool" data-default="false" }
Dumps the whole tangential $W^c$ sample bank. Debug.

About 16 times the size of the pole export; no consumer reads it. Same requirements as `write_poles`.
{: .kw-details }

### `screening_method` {: .kw data-type="enum" data-default="minimax" }
Frequency treatment of $\chi^0$. `minimax` is the only value.

Any other value refuses, `ctsp` included.
{: .kw-details }

**Quasiparticles and self-consistency**
{: #qp }

The QSGW map is on [quasiparticle self-consistency](../../theory/qsgw.md).

### `qp_solver` {: .kw data-type="enum" data-default="auto" }
How QP energies are extracted.

`auto` resolves to `self_consistent` when the deprecated `self_consistent = true` is set, else to `one_shot_dft`. `one_shot_dft` diagonalizes one Hermitian $H$ with $\Sigma^{xc}$ at $E_{n\bk}$; `self_consistent` runs the QSGW loop; no QP equation is solved on either route.
{: .kw-details }

`auto` `one_shot_dft` `self_consistent`
{: .kw-values }

### `sc_tol_ev` {: .kw data-type="float" data-default="1e-4" }
Convergence: $\max|E^{\mathrm{out}} - E^{\mathrm{in}}|$ over the QP matrix of one map, in eV.

The production recipe uses `1e-3`.
{: .kw-details }

### `sc_max_iter` {: .kw data-type="int" data-default="30" }
QSGW map budget, at least 1.

Reaching it unconverged refuses and keeps the per-map tables.
{: .kw-details }

### `density_self_consistent` {: .kw data-type="bool" data-default="true under QSGW" }
Rebuilds $V^{\mathrm H}$ from the rotated occupied orbitals on every QSGW map.

An explicit `false` keeps the DFT $V^{\mathrm H}$ as an announced comparison mode; on a bispinor deck it refuses.
{: .kw-details }

### `sc_head_update` {: .kw data-type="enum" data-default="off" }
How the $\bq \to 0$ head follows each QSGW map.

Unnamed on a tangential QSGW deck with the head on, it becomes `parallel_transport` if `parallel_transport_file` exists, else `dft_velocity` if `dipole.h5` exists. `interband_commutator` refuses on a metal; any value but `off` refuses with `head_correction = off`.
{: .kw-details }

`off` `parallel_transport` `dft_velocity` `interband_commutator`
{: .kw-values }

### `sc_semicore` {: .kw data-type="enum" data-default="dft" }
Semicore class: pinned at its DFT block (`dft`) or reading Σ at its own QP energy (`qp`).

`dft` `qp`
{: .kw-values }

### `sc_frozen_core_bands` {: .kw data-type="int" data-default="0" }
The lowest $N$ bands keep their DFT block and stay in every band sum.

Use `sc_semicore` for semicore states.
{: .kw-details }

### `sc_initial_qp_rotations_file` {: .kw data-type="path" data-default="(empty)" }
Seeds the loop with $U\,\mathrm{diag}(E)\,U^\dagger$ from a `qp_wfn_rotations.h5`.

With an authenticated `sc_checkpoint.h5` beside it the run continues that trajectory. Requires `qp_solver = self_consistent`.
{: .kw-details }

### `sc_history_depth` {: .kw data-type="int" data-default="20" }
Anderson history depth, at least 1. Each entry costs $16\,n_k N_b^2$ bytes, held $2(m+1)$ times.

### `sc_on_ibz` {: .kw data-type="bool" data-default="true" }
Carries $H$, $E$ and $U$ on the star wedge; Σ stays on the full grid.

### `sc_accelerator` {: .kw data-type="enum" data-default="anderson" }
Mixing of $H$. `anderson` is the only value; `rcrop` and `linear` refuse.

### `sc_dump_dir` {: .kw data-type="path" data-default="(empty)" }
Directory for the per-map energy and rotation history. Diagnostic.

**Bands**

### `number_bands_sigma` {: .kw data-type="int" data-default="number_bands" }
Top of the Σ band sum only.

The band-extrapolation brackets are fractions of this count. Same conflict rule as `number_bands_chi`.
{: .kw-details }

### `number_bands_chi` {: .kw data-type="int" data-default="number_bands" }
Top of the $\chi^0$ band sum only.

Naming it beside an explicit `number_bands` with a different value refuses. The band edge takes a strict degeneracy check.
{: .kw-details }

### `zeta_nband` {: .kw data-type="int" data-default="padded band top" }
Narrows the band window of the ζ fit without changing either band sum.

Range $[1, \max(N_\chi, N_\Sigma)]$; a value above the top refuses. Production uses `number_bands`.
{: .kw-details }

**System and occupations**

### `occ_smearing_width_ry` {: .kw data-type="float" data-default="unset" }
Fermi–Dirac $k_BT$ of a metal's occupations, in Ry.

Required when the WFN occupations are fractional and refused when they are integer; `WFN.h5` does not store it. See [Metals](../../theory/metals.md).
{: .kw-details }

### `fermi_reference` {: .kw data-type="enum" data-default="midgap" }
Origin $E_F$ of the Σ frequency axis.

An insulator takes `midgap` or `vbm`; a metal requires `mp1_fixed_n` (fixed-$N$ chemical potential), which refuses on an insulator.
{: .kw-details }

`midgap` `vbm` `mp1_fixed_n`
{: .kw-values }

### `occ_broadening` {: .kw data-type="float" data-default="0.0" }
First-order Methfessel–Paxton width of the smeared QSGW head on an insulator, in eV.

A positive value needs `qp_solver = self_consistent` with `sc_head_update` = `parallel_transport` or `dft_velocity`, and refuses beside `occ_smearing_width_ry`.
{: .kw-details }

### `occupation_clamp_tol` {: .kw data-type="float" data-default="1e-8" }
Snaps metallic occupations within this distance of 0 or 1 to exactly 0 or 1.

Applied inside the fixed-$N$ root; `0.0` disables it.
{: .kw-details }

**Self-energy**
{: #sigma }

### `sigma_regularization_ev` {: .kw data-type="float" data-default="0.25" }
Retarded broadening η of $\Sigma^c(\omega)$, in eV, one value for every treatment.

Must be finite and positive. The crossing windows' node count grows with their width over η ([time–frequency transforms §2.3, §4](../../theory/time-frequency.md#sampling-count)).
{: .kw-details }

### `sigma_quadrature_eps` {: .kw data-type="float" data-default="1e-4" }
Accuracy certificate of every Σ quadrature rule: $\eta\,|Q(d) - 1/d| \le \varepsilon$ on a crossing window, $|d|\,|Q(d) - 1/d| \le \varepsilon$ on a sign-definite one.

In $(0, 1)$. `5e-4` when omitted under `sigma_w_accuracy = relaxed`. Currencies: [time–frequency transforms §4.3](../../theory/time-frequency.md#win-currency).
{: .kw-details }

### `sigma_omega_min_ev` {: .kw data-type="float" data-default="unset" }
Lower edge of the Σ(ω) range, in eV relative to $E_F$. A minimum extent; the range still grows over the requested states.

### `sigma_omega_max_ev` {: .kw data-type="float" data-default="unset" }
Upper edge of the Σ(ω) range, in eV relative to $E_F$.

### `sigma_out_of_grid` {: .kw data-type="enum" data-default="cover" }
Where a QSGW $\Sigma(E)$ outside the sampled range reads.

`cover` grows the range over every requested state; `clamp` reads the nearest edge and `static` reads $\omega = 0$, and both need `sigma_omega_min_ev` and `sigma_omega_max_ev` set.
{: .kw-details }

`cover` `clamp` `static`
{: .kw-values }

### `sigma_omega_step_ev` {: .kw data-type="float" data-default="0.25" }
Step of the Σ(ω) grid, in eV.

On shell it only places the range edges; Σ is sampled at this step only under `sigma_omega_grid_debug`.
{: .kw-details }

### `sigma_omega_patches_ev` {: .kw data-type="string" data-default="(empty)" }
`"lo:hi, …"` patches in eV that replace the contiguous Σ range.

A QP energy inside a hole refuses. `lo:hi:eta` triples are coarse windows for the semicore class, each read at its own η (not below `sigma_regularization_ev`).
{: .kw-details }

### `sigma_window_edge_factor` {: .kw data-type="float" data-default="1.5" }
Product-window margin, state edge = factor × η. Adds no broadening ([windows](../../theory/time-frequency.md#win-products)).

### `no_degen_averaging` {: .kw data-type="bool" data-default="false" }
Disables BerkeleyGW-style averaging of the written Σ diagonals over degenerate DFT states.

### `degen_avg_tol_ry` {: .kw data-type="float" data-default="1e-6" }
Degeneracy tolerance of that averaging, in Ry.

### `sigma_omega_grid_debug` {: .kw data-type="bool" data-default="false" }
Computes $\Sigma^c(\omega)$ on the whole grid instead of on shell.

Debug. Adds the Σ(ω) cube and `omega_ev` to `sigma_mnk.h5`, takes eqp1's $Z$ from a ±0.5 eV central difference, reads the frequency debug table on the grid, and keeps the $n_\omega$-deep cube resident. `write_eqp2` stays retired.
{: .kw-details }

**Band extrapolation**
{: #band-extrapolation }

The model is on [band extrapolation](../../theory/band-extrapolation.md).

### `use_band_extrapolation` {: .kw data-type="bool" data-default="true" }
Extrapolates the $\Sigma^c$ band sum past `number_bands_sigma` from three bracket sums.

Refuses when `number_bands_sigma` $< 2n_\mathrm{occ}$. On a stage that does not consume it the default switches itself off; an explicit `true` there refuses.
{: .kw-details }

### `band_extrapolation_bracket_scheme` {: .kw data-type="enum" data-default="total_fractions" }
The three bracket band counts.

`total_fractions` takes 70, 85 and 100 % of $N_\Sigma$. Naming it with extrapolation off refuses.
{: .kw-details }

`total_fractions` `conduction_fractions` `conduction_energy_midpoint`
{: .kw-values }

### `band_extrapolation_estimator` {: .kw data-type="enum" data-default="spectral_shell" }
Estimator applied to the brackets. `spectral_shell` is the only value.

**Head**
{: #head }

The $\bq \to 0$ keys. Their physics is on [the q → 0 head](../../theory/q0-head.md).

### `head_correction` {: .kw data-type="enum" data-default="full" }
Γ-cell policy.

`full` is the macroscopic head with wings and body fold, `no_local_fields` the direct head only, `off` a debug skip. Unset on `bispinor_gw = full_tangential` it becomes `no_local_fields`. On a time-reversal-broken metal the tangential route refuses `full`.
{: .kw-details }

`full` `no_local_fields` `off`
{: .kw-values }

### `head_minibz_average` {: .kw data-type="bool" data-default="false" }
Coulomb head as a mini-BZ cell average instead of a point value.

A BSE run on a densified `kgrid_fine` must use the same value.
{: .kw-details }

### `wcoul0_source` {: .kw data-type="enum" data-default="s_tensor" }
Source of the direct head: `dipole.h5` (`s_tensor`) or `eps0mat.h5` (`epshead`, static only).

`s_tensor` `epshead`
{: .kw-values }

### `wcoul0_eta` {: .kw data-type="float" data-default="0.0" }
Broadening of the head evaluation, in Ry.

### `bgw_metal_q0_treatment` {: .kw data-type="enum" data-default="exact" }
Metallic $\bq = 0$ convention.

`bgw_q0shift` emulates BerkeleyGW's shifted $q_0$; it needs `compute_mode = mpa` and `sys_dim = 3`, and refuses an explicit `mc_average_vcoul_body = true`.
{: .kw-details }

`exact` `bgw_q0shift`
{: .kw-values }

### `bgw_metal_q0_vector` {: .kw data-type="string" data-default="0 0 0.125" }
Reduced coordinates of the shifted $q_0$ under `bgw_q0shift`. Three finite numbers, not all zero.

### `mc_average_placement` {: .kw data-type="enum" data-default="off" }
Where the $\bq \ne 0$ mini-BZ average enters: in $V$ (`off`) or on W's head after the Dyson solve (`bgw`).

`schur_avg` is declared and refuses.
{: .kw-details }

`off` `bgw`
{: .kw-values }

### `mc_average_placement_vcoul` {: .kw data-type="path" data-default="(empty)" }
BerkeleyGW `write_vcoul` dump that supplies the mini-BZ average under `mc_average_placement = bgw`.

### `vhead` {: .kw data-type="float" data-default="computed" }
Override of the bare Coulomb head.

### `whead_0freq` {: .kw data-type="float" data-default="computed" }
Override of the static screened head $W_h(\omega = 0)$.

### `whead_imfreq` {: .kw data-type="float" data-default="computed" }
Override of the screened head at the GN probe $i\omega_p$.

**ISDF and Coulomb**
{: #isdf }

### `zeta_cutoff` {: .kw data-type="float" data-default="WFN ecutwfc" }
G sphere $|\bq+\bG|^2 \le$ cutoff on which $\zeta_{\bq\mu}(\bG)$ is stored, in Ry.

Must be at least `bare_coulomb_cutoff` and at most the density cutoff.
{: .kw-details }

### `bare_coulomb_cutoff` {: .kw data-type="float" data-default="WFN ecutwfc" }
G sphere of the sum in $V_\bq(\mu,\nu)$, in Ry.

Same meaning as BerkeleyGW's `bare_coulomb_cutoff`.
{: .kw-details }

### `zeta_rcond` {: .kw data-type="float" data-default="1e-8" }
Rank cut of the charge ζ Gram, relative to its largest eigenvalue.

See [ISDF and V(q)](../../theory/isdf.md).
{: .kw-details }

### `mc_average_vcoul_body` {: .kw data-type="bool" data-default="true" }
Monte Carlo mini-BZ average of $v(\bq+\bG)$ at every $\bq \ne 0$.

`false` averages only the $\bq+\bG = 0$ cell. Forced to `false` by `bgw_metal_q0_treatment = bgw_q0shift`.
{: .kw-details }

### `use_bgw_vcoul` {: .kw data-type="bool" data-default="false" }
Reads $v(\bq,\bG)$ from a BerkeleyGW `vcoul` file instead of building it.

Cross-code comparison only.
{: .kw-details }

### `bgw_vcoul_file` {: .kw data-type="path" data-default="(empty)" }
The BerkeleyGW `vcoul` file read under `use_bgw_vcoul`.

### `bgw_vcoul_sym_wfn` {: .kw data-type="path" data-default="(empty)" }
Auxiliary WFN whose symmetry folds the q points onto BerkeleyGW's irreducible list.

For a main WFN written without symmetry.
{: .kw-details }

**Files**
{: #files }

A relative path resolves against the deck's directory for `wfn_file`, `centroids_file`, `centroids_file_current`, `kin_ion_file`, `parallel_transport_file`, `static_gauge_hall_file`, `bgw_vcoul_file`, `bgw_vcoul_sym_wfn`, `mpa_fit_reuse_file`, `sc_initial_qp_rotations_file`, `report_file`, `sigma_diag_file`, `eqp0_file`, `eqp1_file` and `sigma_omega_h5_file`. `sigma_freq_debug_file` and `sc_dump_dir` are used as given, so a relative value resolves against the launch directory. Formats are on the [Files](files.md) page.

### `wfn_file` {: .kw data-type="path" data-default="WFN.h5" }
Mean-field wavefunctions, BerkeleyGW format.

### `kin_ion_file` {: .kw data-type="path" data-default="kin_ion.h5" }
One-body operator $\langle m\bk|T + V^{\mathrm{ion}}|n\bk\rangle$ from [kin-ion](../../utilities/kin-ion.md). Authenticated against the run.

### `parallel_transport_file` {: .kw data-type="path" data-default="parallel_transport.h5" }
Link and velocity artifact from [dipole](../../utilities/dipole.md), read by the velocity heads.

### `report_file` {: .kw data-type="path" data-default="gwjax.out" }
The run report.

### `sigma_diag_file` {: .kw data-type="path" data-default="sigma_diag.dat" }
Σ decomposition table.

### `eqp0_file` {: .kw data-type="path" data-default="eqp0.dat" }
Zeroth-order QP energies, BerkeleyGW format.

### `eqp1_file` {: .kw data-type="path" data-default="eqp1.dat" }
Linearized QP energies, BerkeleyGW format.

### `sigma_omega_h5_file` {: .kw data-type="path" data-default="sigma_mnk.h5" }
Σ matrix elements of dynamic runs.

### `write_wfn_h5` {: .kw data-type="bool" data-default="true" }
Writes `WFN_qp.h5`. `qp_wfn_rotations.h5` is written regardless.

A one-shot run on a symmetry-reduced WFN skips the file with a printed message: its Σ is on the full zone, not the WFN's k set ([WFN_qp.h5](files.md#wfn-qp)).
{: .kw-details }

### `write_qsgw_datasets` {: .kw data-type="bool" data-default="false" }
Adds the Hermitian QSGW $\Sigma^{xc}$ matrix `sigma_xc_qsgw_kij_ev` to `sigma_mnk.h5`.

Only a dynamic `compute_mode` writes `sigma_mnk.h5`. The ladder `qp_omega0_ev` is added only under `sigma_omega_grid_debug = true`; `qp_static_cohsex_ev` is never written, since `cohsex` writes no `sigma_mnk.h5`.
{: .kw-details }

### `sigma_freq_debug_output` {: .kw data-type="bool" data-default="false" }
Writes the per-band Σ table and the head attribution. Debug.

On shell a one-shot run reads the table from its on-shell values; a dynamic QSGW run omits it unless `sigma_omega_grid_debug = true`.
{: .kw-details }

### `sigma_freq_debug_file` {: .kw data-type="path" data-default="sigma_freq_debug.dat" }
Path of that table.

**Restart and layout**
{: #restart }

### `restart` {: .kw data-type="bool" data-default="false" }
Reads `tmp/isdf_tensors_<N_mu>.h5` instead of refitting ζ and rebuilding $V$, after authentication.

A bundle is never selected from its presence alone. Conditions are on the [Dependencies](dependencies.md#caches) page.
{: .kw-details }

### `write_restart_tensors` {: .kw data-type="bool" data-default="true" }
Writes the restart bundle. With `false`, a downstream BSE run refuses.

### `restart_q_storage` {: .kw data-type="enum" data-default="auto" }
q set of the restart bundle.

`auto` stores the irreducible wedge when the centroid set is orbit-closed and the q grid reduced, the full grid otherwise; `ibz` refuses a set that cannot be stored on the wedge.
{: .kw-details }

`auto` `ibz`
{: .kw-values }

### `qp_rotations_k_storage` {: .kw data-type="enum" data-default="auto" }
k set of `qp_wfn_rotations.h5`.

`auto` keeps the wedge when its round trip reproduces the full-grid arrays exactly; `ibz` refuses otherwise.
{: .kw-details }

`auto` `ibz` `full`
{: .kw-values }

**Solver**
{: #solver }

### `memory_per_device_gb` {: .kw data-type="float" data-default="0.0" }
Per-device memory budget for the chunk planners, in GB. `0` detects it.

### `linalg` {: .kw data-type="enum" data-default="local" }
Layout of the dense solves (ζ, the $W$ Dyson solve, eigensolvers).

`local` keeps whole $N_\mu \times N_\mu$ matrices per task; `distributed` block-distributes them. Changes no result and does not invalidate a restart.
{: .kw-details }

`local` `distributed`
{: .kw-values }

### `vq_g_chunk_size` {: .kw data-type="int" data-default="0" }
G-axis tile of the $V_\bq$ product. `0` lets the planner choose.

### `gamma_contract_mode` {: .kw data-type="enum" data-default="take" }
Kernel variant of one spinor contraction. All three are mathematically identical.

`take` `einsum` `scan`
{: .kw-values }

**Spinors**
{: #spinors }

### `bispinor` {: .kw data-type="bool" data-default="false" }
Four-component run: four-channel ζ fit, transverse Σ channels and a second centroid table.

### `bispinor_gw` {: .kw data-type="enum" data-default="bare_transverse" }
Four-current screening model.

`full_tangential` needs `compute_mode = mpa`, `sigma_w_model = tangential` and `screening_diagrams = w_rpa`, and refuses `head_correction = full`; the old value `full_shared_pole` is still accepted. `full_static_cohsex` is one-shot on a slab.
{: .kw-details }

`bare_transverse` `full_tangential` `full_static_cohsex`
{: .kw-values }

### `centroids_file_current` {: .kw data-type="path" data-default="(empty)" }
Transverse-channel centroid table. Bispinor runs only.

### `static_gauge_hall_file` {: .kw data-type="path" data-default="(empty)" }
Hall pseudovector for the packed Γ-cell completion.

Empty means $\sigma_H = 0$; a named file that is absent refuses.
{: .kw-details }

### `sigma_lorentz_debug_output` {: .kw data-type="bool" data-default="false" }
Writes each four-current map's on-shell CC, CT+TC and TT matrices. Debug.

**Multipole sampling**
{: #mpa }

These keys configure the elementwise MPA fit (`sigma_w_model = mpa`). Under `tangential` with the head on, the first seven set the sampling of the head fit and the last three refuse; with the head off all ten refuse (`GATE tangential_unused_inputs`).

### `mpa_n_poles` {: .kw data-type="int" data-default="8" }
Poles per matrix element, 1 to 16.

### `mpa_sampling_alpha` {: .kw data-type="int" data-default="1 insulator, 2 metal" }
Exponent of the nested sampling partition $\omega_n = \omega_m s_n^\alpha$.

`1` `2`
{: .kw-values }

### `mpa_sampling_schedule` {: .kw data-type="enum" data-default="nested" }
Continuation of the sampling above eight poles.

`nested` `leon`
{: .kw-values }

### `mpa_pole_solver` {: .kw data-type="enum" data-default="loewner" }
Pole identification method.

`loewner` `companion` `thiele`
{: .kw-values }

### `mpa_varpi_near_ry` {: .kw data-type="float" data-default="0.2" }
Height of the near complex sampling line, in Ry. Must be below `mpa_varpi_far_ry`.

### `mpa_varpi_far_ry` {: .kw data-type="float" data-default="2.0" }
Height of the far complex sampling line, in Ry.

### `mpa_metal_origin_shift_ry` {: .kw data-type="float" data-default="2e-5" }
Metal only: height of the first near-line sample, $z = i\cdot$shift, in Ry.

Refused on an insulator; must lie below `mpa_varpi_near_ry`.
{: .kw-details }

### `mpa_pole_batch_size` {: .kw data-type="int" data-default="4" }
Fitted-pole slabs resident during the MPA Σ, 1 to 8. Memory schedule only.

### `mpa_fit_reuse_file` {: .kw data-type="path" data-default="(empty)" }
A finalized MPA fit reused read-only, after authentication.

Requires `compute_mode = mpa` and `qp_solver = one_shot_dft`.
{: .kw-details }

### `mpa_overwrite_completed_artifacts` {: .kw data-type="bool" data-default="false" }
Permits replacing a completed MPA sample store or finalized fit.

**Plasmon pole**
{: #ppm }

Read by `compute_mode = gn_ppm | hl_ppm`.

### `ppm_omega_p` {: .kw data-type="float" data-default="2.0" }
Second probe frequency, in Ry: $i\omega_p$ for GN, real $\omega_p$ for HL.

An HL deck sets it above every $\chi^0$ transition energy.
{: .kw-details }

### `ppm_invalid_mode` {: .kw data-type="enum" data-default="static_limit" }
Treatment of modes with no valid $\Omega^2$.

`static_limit` (alias `infinity`) is BerkeleyGW mode 3, `zero` (alias `skip`) mode 0, `2ry` mode 2; `imaginary` refuses.
{: .kw-details }

`static_limit` `zero` `2ry`
{: .kw-values }

### `ppm_fallback_omega` {: .kw data-type="float" data-default="2.0" }
Pole used by `ppm_invalid_mode = 2ry`, in Ry.

### `ppm_head_omega_h_ry` {: .kw data-type="float" data-default="computed" }
Override of the $\bq \to 0$ head pole, in Ry. For BerkeleyGW comparisons.

**Contour deformation**
{: #cd }

The `cd_*` keys tune the contour-deformation integral of $\Sigma^c(\omega)$ over the frequency of $W$. They are read only with `sigma_frequency_integration = contour_deformation` and refuse otherwise (`GATE cd_applicability`). The physics is on [Contour deformation](../../theory/contour-deformation.md).

### `cd_imaginary_nodes` {: .kw data-type="int" data-default="96" }
Gauss–Legendre nodes on the imaginary axis, under $u=\eta\tan\theta$ in two panels. At least 5.

### `cd_knot_step_ev` {: .kw data-type="float" data-default="0.0625" }
Spacing $h$ of the real-axis knots $jh+i\eta$, in eV. $0<h\le\eta/4$.

### `cd_residue_reach_ev` {: .kw data-type="string" data-default="auto" }
Extent of the knot line, in eV.

`auto` is the largest live $\lvert E-E_l\rvert$ over the requested energies, the own-energy rows and the Σ band sum. An explicit value below it refuses (`GATE cd_residue_coverage`).
{: .kw-details }

### `cd_own_energy_band_min` {: .kw data-type="int" data-default="1" }
First band (1-based) of the own-energy rows $\Sigma^c_{mm}(E_{m\bk})$ and of the `auto` reach.

The Σ cube keeps the whole window. Raise it to leave semicore rows out of the real-axis reach.
{: .kw-details }

### `cd_chi0_rule_tolerance` {: .kw data-type="float" data-default="1e-8" }
Relative tolerance of the response-bank rules that give $\chi^0$ at every CD point. In $[10^{-13},10^{-2})$.

### `cd_chi0_rule_capacity` {: .kw data-type="int" data-default="1088" }
Largest node count of one $\chi^0$ rule set on the knot windows. In $[192, 8192]$.

### `cd_chi0_band_windows` {: .kw data-type="enum" data-default="off" }
`auto` splits each knot window's pair space into crossing and remote band windows, each with its own $\chi^0$ rule. Insulators only.

`off` `auto`
{: .kw-values }

### `cd_chi0_remote_margin_ev` {: .kw data-type="float" data-default="4.0" }
With `cd_chi0_band_windows = auto`: how far above the window's largest $\lvert\mathrm{Re}\,z\rvert$ a remote band window starts, in eV. In $(0, 100]$.

### `cd_chi0_rules_file` {: .kw data-type="path" data-default="(empty)" }
Precomputed $\chi^0$ rules from `python -m gw.cd.chi0_rules`, relative to the deck.

Checked against the run's energies, occupations, tolerance, points and windows; a mismatch refuses (`GATE cd_chi0_rules_identity`). Empty: the rules are built in the job and written to `cd_chi0_rules.pkl`.
{: .kw-details }

### `cd_checkpoint` {: .kw data-type="enum" data-default="off" }
Per-chunk accumulator checkpoint `tmp/cd_checkpoint.h5`.

`write` rewrites it after every chunk; `resume` continues from an authenticated one, then writes. A resume is never inferred from the file.
{: .kw-details }

`off` `write` `resume`
{: .kw-values }

### `cd_knot_chunk` {: .kw data-type="int" data-default="0" }
Imaginary nodes or knots per streamed chunk. `0` lets the tile planner size it from the bytes per point.

### `cd_band_tile` {: .kw data-type="int" data-default="0" }
Bands of the $l$ sum contracted at a time. `0` lets the planner size it; any tile changes only the summation order.

## Keys read by other drivers {: #other }

The gw deck is also the deck of the preprocessing utilities and BSE, so the schema holds their keys. `gw_jax` accepts them and reads a few of them only to authenticate `dipole.h5`.

### `ecutrho` {: .kw data-type="float" data-default="WFN ecutwfc" }
Density cutoff of [kin-ion](../../utilities/kin-ion.md) and [dipole](../../utilities/dipole.md), in Ry.

### `screened_coulomb_cutoff` {: .kw data-type="float" data-default="WFN ecutwfc" }
Sphere of the plane-wave $\chi_\bq(\bG,\bG')$ path, in Ry. No driver reads it.

### `kgrid_fine` {: .kw data-type="string" data-default="(empty)" }
[BSE](../bse/keywords.md#fine-grid) and [htransform](../../utilities/htransform-fine-grid.md): fine k grid onto which the BSE problem (or the htransform's fine bands) is interpolated.

### `w_head_densify` {: .kw data-type="enum" data-default="c1" }
[BSE](../bse/keywords.md): coarse-to-fine treatment of the W head.

### `htransform_rank_multiplier` {: .kw data-type="float" data-default="20.0" }
[htransform](../../utilities/htransform.md): search ceiling of the QRCP basis.

### `htransform_qr_eps` {: .kw data-type="float" data-default="1e-3" }
[htransform](../../utilities/htransform.md): rank criterion of the QRCP basis.

### `htransform_qrcp_seed` {: .kw data-type="int" data-default="0" }
[htransform](../../utilities/htransform.md): seed of the randomized QRCP.

### `get_centroids_fi` {: .kw data-type="bool" data-default="false" }
[htransform](../../utilities/htransform.md): computes fine-grid wavefunctions at the coarse centroids.

### `kgrid_fine_shift`, `number_cond_bands_coarse` {: .kw data-type="string, int" data-default="(empty)" }
[BSE](../bse/keywords.md#fine-grid) and [htransform](../../utilities/htransform-fine-grid.md): kgrid.x's offset of `kgrid_fine` and the coarse conduction bands of the band interpolation.

### `wfn_fi_q_chunk` {: .kw data-type="int" data-default="0" }
[htransform](../../utilities/htransform.md): fine q points processed at once; `0` takes the coarse k count.

## Retired {: #retired }

Deprecated keys are honored with a note and refuse when they contradict their named replacement.

### `nband` {: .kw data-type="int" data-default="unset" }
Deprecated alias of `number_bands`. Naming both with different values refuses.

### `do_screened` {: .kw data-type="bool" data-default="true" }
Deprecated. Use `compute_mode`; `false` means `x_only`.

### `use_ppm_sigma` {: .kw data-type="bool" data-default="false" }
Deprecated. Use `compute_mode = gn_ppm` or `hl_ppm`.

### `ppm_model` {: .kw data-type="enum" data-default="gn" }
Deprecated. Use `compute_mode = gn_ppm` or `hl_ppm`.

### `self_consistent` {: .kw data-type="bool" data-default="false" }
Deprecated. Use `qp_solver = self_consistent`.

### `sigma_at_dft_energies` {: .kw data-type="bool" data-default="false" }
Deprecated. Use `qp_solver = one_shot_dft`, which `auto` resolves to.

### `sigma_band_extrapolation` {: .kw data-type="bool" data-default="unset" }
Deprecated alias of `use_band_extrapolation`. Naming both with different values refuses.

### `do_G0` {: .kw data-type="bool" data-default="true" }
Deprecated. Use `head_correction`; `false` means `off`.

### `zeta_ridge` {: .kw data-type="float" data-default="0.0" }
Retired Tikhonov ridge of the charge ζ Gram. The default parses; any other value refuses at the ζ fit. `zeta_rcond` sets the cut.

### `sc_mixing` {: .kw data-type="float" data-default="1.0" }
Retired. The default parses; any other value refuses (`GATE sc_mixing_retired`). Delete the key.

Retired values refuse by name: `qp_solver = fixed_point` (use `one_shot_dft` or `self_consistent`), `sc_accelerator = rcrop | linear` (use `anderson`), `band_extrapolation_estimator = band_index_only` (use `spectral_shell`), `screening_method = ctsp` (use `minimax`), `bispinor_gw = charge_hall_cubature` (use `full_static_cohsex`), `bispinor_gw = pauli_reference_bare_transverse | isometric_kinetic_balance_bare_transverse` (use `bare_transverse`), `ppm_invalid_mode = imaginary`.

Removed keys refuse with the reason; delete them:

| Key | Replacement |
|---|---|
| `distributed_zeta_solve`, `distributed_cholesky`, `distributed_lu`, `w_dyson_solver`, `distrib_la_batched_route`, `charge_zeta_solve`, `transverse_zeta_solve`, `eigh_backend`, `sc_eigh`, `use_low_mem_eigh` | `linalg` |
| `band_chunk_size`, `low_mem_bands`, `r_chunk_size`, `gflat_chunk_size` | `memory_per_device_gb` |
| `ppm_sigma_target_error`, `ppm_sigma_max_nodes`, `mpa_sigma_sector_target_error`, `mpa_sigma_max_nodes`, `sigma_quadrature_reduction_seconds`, `sigma_quadrature_reduction_steps` | `sigma_quadrature_eps` |
| `sigma_regularization_floor_ev` | `sigma_regularization_ev` |
| `occ_smearing_family` | `occ_smearing_width_ry` (metals are Fermi–Dirac) |
| `use_shipped_minimax_tables`, `regenerate_minimax_tables`, `sigma_quadrature_cache_dir` | none: rules are built in each run |
| `slab_io`, `use_ffi_io`, `gspace_mode`, `hartree_source`, `bispinor_tt_head_correction`, `transverse_zeta_rcond`, `strict_keys` | none |
| `sc_exact_degeneracy_tol_ev`, `sc_tail_fit`, `sc_buffer_nbands`, `sc_buffer_mode`, `sigma_at_dft_extrapolate`, `occupation_window_threshold` | none (widen `nval` / `ncond` for the buffer keys) |
| `sigma_omega_accumulation = kij_stream` | none; other values of the key are ignored with a warning |
| `write_eqp2`, `eqp2_tol_ev`, `eqp2_max_iter`, `eqp2_accelerator`, `eqp2_history_depth`, `eqp2_file` | none: the fixed-Σ eqp2 loop is retired (`GATE eqp2_retired`) |

`chunk_size`, `output_file` and `eqp_output_file` are ignored with a warning: chunking is planner-owned, and `eqp0.dat` and `eqp1.dat` are always written (`sigma_diag_file` names the Σ table).

