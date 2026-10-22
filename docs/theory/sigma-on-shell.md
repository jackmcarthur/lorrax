# On-shell Σ_c

A dynamic Σ_c (GN/HL-PPM, MPA, tangential, the four-current sectors) is
integrated from its time-domain nodes σ(τ) by the minimax window quadrature
([Σ(ω) quadrature problem](sigma-quadrature-problem.md)). Every consumer of
the result reads it at the energies of the states it belongs to: the QSGW
ansatz

    Σ^QSGW_mn(k) = ½ [ Σ_mn(k, E_mk) + Σ_mn(k, E_nk) ]ʰ

and the eqp writers' band diagonal Σ_nn(E_n) with its slope. So the τ sweep
folds each σ(τ) into those reads directly instead of into an ω grid. This is
the production evaluation; the ω-grid path is kept as an opt-in debug
(`sigma_omega_grid_debug = true`, [below](#grid-debug)), and fixed-Σ evSC
(`eqp2.dat`) is retired ([below](#retired)).

## What is folded {#fold}

A window with nodes (t_i, α_i), pole sign s and reference E_ref adds
`c_i(ω) σ(t_i)` with `c_i(ω) = pref·α_i·exp(−i(E_ref − s|ω|)t_i)`
(`gw.ppm_accumulators`). On shell the coefficient is evaluated per state
(k, b) on the device, once per node, and the fold is

    pair[0, k, m, n] += c_i(ω_km) σ_mn(t_i)        Σ_mn(E_m)
    pair[1, k, m, n] += c_i(ω_kn) σ_mn(t_i)        Σ_mn(E_n)
    slot[j, k, n]    += c_i(ω^(j)_kn) σ_nn(t_i)    Σ_nn on table j
    slot[−1, k, n]   += i s s_ext t_i c_i(ω_kn) σ_nn(t_i)    dΣ_nn/dω

with `s_ext = ±1` the window's frequency half. The last row is the exact ω
derivative of the same quadrature. Each σ(τ) is read once; the output is
`(2, k, nb, nb)` plus `(n_slot, k, nb, p_y)` diagonal slots. An anti-Hermitian window closes
`(Z − Z†)/2i` with the slots exchanged: Z(ω_m)† is conj(Z(ω_m)ᵀ), whose
frequency rides the column index, so `pair[0] += (Z₀ − Z₁ᵀ*)/2i` and
`pair[1] += (Z₁ − Z₀ᵀ*)/2i`. The window planner plans on the distinct
evaluation energies (`ppm_sigma.on_shell_request`).

The coefficient inputs of every window are placed once per Σ call as one
replicated `WindowTable`: per window the padded nodes `t`, `α`, the node
count, `own` (`(n_table, n_parent, nb)` bool: the states whose energy the
window's frequencies hold) and four scalars (prefactor, E_ref, s, s_ext).
The per-state energies are the request's parent-row tables, placed once by
the accumulator. A window reads its row on device behind a device counter, so
it transfers nothing. The run prints the table's size per device
(`Sigma window table: N windows, X MiB per device`).

The derivative row inherits the value's quadrature certificate in the squared
currency: on one pole through the production rule (`sigma_quadrature_eps =
1e-4`, η = 0.25 eV), the derivative's error is 1.3 ε·η⁻² on a crossing box and
0.18 ε·|d|⁻² on a sign-definite one (`tests/test_sigma_on_shell_cpu.py`,
`test_single_pole_derivative_at_production_eps_eta`).

## The evaluation tables {#tables}

The energies are fixed before the sweep, measured from the body's ω
reference, and read by the readers' rules
(`dynamic_sigma.sigma_on_shell_request` builds and prints the request once
per Σ call):

| table | energies | rule | reader |
|---|---|---|---|
| `pair` | E_eval | `sigma_out_of_grid` (`qsgw_utils.sigma_eval_omega`) | the QSGW pair, the band-extrapolation fit, dΣ/dω |
| `eval0` | E_eval | uncovered → Σ(ω = 0) | eqp1 at E_eval (SC) |
| `dft0` | E_DFT | uncovered → Σ(ω = 0) | eqp0, `sigC(Edft)` |
| `zero` | 0 | — | SC grid-edge census (`sigma_out_of_grid = static` only) |
| `edge` | the nearer range edge | — | SC grid-edge census (same) |

A table equal to an earlier one is not repeated (one-shot: one table, two
slots). "Covered" is the extent of `config.omega_grid_ev`, which the SC
`cover` growth moves as before; the grid's step is not read. Z is
`1/(1 − Re dΣ/dω)` on covered states and 1 elsewhere
(`eqp_bgw.on_shell_z_factor`, the one Z of eqp1 and the SC tail weights). It
prints one line counting states with Z outside (0, 1], and a non-finite Z
refuses (`GATE z_factor_finite`). The readers check that they ask for the
energies the sweep used (`GATE on_shell_reads`).

Z is the analytic derivative, not BerkeleyGW's ±0.5 eV central difference
(`finite_difference_spacing`). The two differ by the curvature of Σ over
±0.5 eV.

The SC grid-edge census (`sc_iteration._sc_edge_ambiguity`) compares
Re Σ_nn(0) with Re Σ_nn at the nearer edge of the sampled range (or of the SC
growth window, where that reaches further) from the `zero` and `edge` tables.

## Symmetry {#symmetry}

The sweep runs on the parent k rows at the parents' energies
(`GATE on_shell_star_invariant`: the energies are equal across a star), and
the wedge→BZ transport is the grid's. On an antiunitary row it transposes,
`Σ(k') = Σ(k)ᵀ`, and a transpose exchanges which state's frequency a slot
carries: there `pair[0](k') = pair[1](k)ᵀ` (`mpa.sigma._on_shell_unfold_fn`).
Diagonals are unchanged.

## Band extrapolation and the head {#extrapolation}

Every cumulative band count keeps its diagonal slots; the first and last keep
the pair. The extrapolated Σ is the same real per-state combination as on the
grid, applied to the pair (`½(w_m + w_n)`) and to each slot (`w_n`), so dΣ/dω
and the eqp tables are extrapolated too. The fit reads each state's `pair`
diagonal, the value H reads. The q→0 head (`head_correction.HeadSigmaDiag`,
now also GN/HL-PPM's) adds its closed-form value on each table and its exact
derivative to the derivative slot.

## Output {#output}

`sigma_mnk.h5` keeps its dataset names: `sigma_c_kij_ev` and
`sigma_total_kij_ev` hold the pair `(2, nk, nb, nb)` (stamped `pair_slots`),
beside `sigma_c_onshell_<table>_kn_ev`, `onshell_<table>_rel_ev` and
`sigma_c_onshell_dsigma_kn`, under a root `sigma_evaluation = on_shell`.
There is no `omega_ev`, so a reader that needs an ω axis fails on it. The
eqp receipt carries `eqp_assembly_conditioned_sigma_c_onshell_ev`
(C(E_DFT), C(E_eval), dC/dω, coverage) in place of the C(ω) curve, and
`eqp_bgw.make_eqp_bgw` reassembles from it.

## The ω-grid debug path {#grid-debug}

`sigma_omega_grid_debug = true` (default false) evaluates Σ_c on the whole ω
grid instead, as before 2026-10 and bit for bit as `c2975e450` with the same
key (UPGRADE_NOTES, "Sigma(omega) grid debug restored"). The same window
runner folds each σ(τ) into every grid frequency with the `(capacity, n_ω)`
coefficient (`DeviceOmegaAccumulator.grid`); every reader then interpolates
the cube instead of reading a table. It writes:

- `sigma_mnk.h5` with the `(n_ω, nk, nb, nb)` cubes `sigma_c_kij_ev` /
  `sigma_total_kij_ev` and the dataset `omega_ev` (the grid, relative to
  `omega_reference_ev`), streamed ω slab by ω slab; no `sigma_evaluation`
  marker and no `onshell_*` tables.
- eqp0 at the interpolated Σ_c(E_DFT) and eqp1 with Z from the ±0.5 eV
  central difference on the grid (`eqp_bgw.compute_z_factor_from_omega_grid`);
  the eqp receipt carries the conditioned C(ω) curve
  (`eqp_assembly_conditioned_sigma_c_omega_diag_ev`), which
  `python -m gw.eqp_bgw` reassembles from (`--finite-difference-spacing`).
- `sigma_freq_debug.dat` (with `sigma_freq_debug_output`) read on the C(ω)
  curve, the head interpolated on the grid; `qp_omega0_ev` (with
  `write_qsgw_datasets`), the H₀ + Σ_x + Σ_c(ω ≈ 0) ladder, on an
  unsharded cube only (the rank-0 appendix names a band-sharded one, under
  `sigma_omega_layout = sharded`, as omitted).
- Under SC: the QSGW pair interpolated from the cube, Z from the stencil,
  the cube rotated to the DFT basis at the end so `sigma_diag.dat` carries
  `sigC`/`sigXC` at E_DFT, and the edge census read on the cube.
- A one-shot tangential sweep checkpoint (`tmp/sigma_checkpoint_oneshot.h5`,
  `file_io.sigma_checkpoint`), written when the sweep took at least 5× its
  predicted write.

**Cost.** The accumulator holds the cube: `16·n_ω·nk_wedge·nb²/(p_x p_y)`
bytes per rank on the band tiles, against `16·2·nk_wedge·nb²/(p_x p_y)` for
the on-shell pair, and the window table carries `n_ω`-wide coefficient rows.
The full-BZ cube, its head add, the QSGW interpolation, the writer's eV
cubes (made one at a time) and the SC rotation each hold a cube more.
Measured on Fe 8^3 (MPA shared pole, SC, 4 A100 ranks, n_ω = 271, 59 k,
38 bands): `sigma_mnk.h5` is 714 MiB against 7.9 MiB (one 352 MiB cube
each for Σ_c and Σ_total), the τ sweep 122.3 s against 105.3 s (+16 %), and
the per-rank device peak is unchanged at 16.34 GiB because the W response
sets it. A one-shot run also writes a 2.99 GiB sweep checkpoint.

The Z the grid gives is the smoothed stencil Z, not the exact derivative,
so eqp1 under the grid differs from the default by up to the amounts the
on-shell entry reported (UPGRADE_NOTES, "Sigma_c on shell only"). The Z
guard (`GATE z_factor_finite`) applies on shell only. The grid path has a
CPU test (`tests/test_sigma_on_shell_cpu.py::test_grid_fold_matches_closed_form`,
`::test_grid_cube_diagonal_at_on_shell_energies_is_the_on_shell_slots`,
`tests/test_sigma_window_table_cpu.py::test_window_pick_matches_host_coefficient`).

## Retired {#retired}

Fixed-Σ evSC and what only it used are gone (UPGRADE_NOTES, "Sigma_c on shell
only"). The deck keys refuse by name at parse time (`GATE eqp2_retired`).

- `write_eqp2`, `eqp2_tol_ev`, `eqp2_max_iter`, `eqp2_accelerator`,
  `eqp2_history_depth`, `eqp2_file`: fixed-Σ evSC (`eqp2.dat`) re-read one
  Σ(ω) at moving energies in a rotating basis; with it went the rCROP
  accelerator and the EQP2 scissor partition.
- On shell: `qp_omega0_ev` (there is no ω = 0 matrix) and an SC run's
  DFT-basis Σ_c(E_DFT) (the pair's slots belong to the QP basis's band
  indices, so no rotation acts on them). `sigma_diag.dat` omits the `sigC`
  and `sigXC` columns under SC and names them in a `# omitted:` header line;
  the SC eqp files are the SC spectrum and do not depend on it. Both come
  back under the grid debug path.
- On shell, `sigma_freq_debug.dat` keeps its one-shot columns from the
  on-shell operand and the head at E_DFT; under SC it is omitted by name.
- A one-shot on-shell sweep is checkpointed as its pair and slots
  (`tmp/sigma_checkpoint_oneshot_on_shell.h5`, `file_io.sigma_checkpoint`),
  under the payoff rule of the grid path. Its identity adds the evaluation
  and the request's tables (E_eval, E_DFT, coverage) to the grid's, so moved
  energies recompute and neither evaluation reads the other's file. An SC
  run continues at map granularity (`sc_seed/sc_checkpoint.h5`).
- `python -m gw.plane_wave_pipeline --deck` runs on shell only: the JSON keeps
  `sigc_at_e_ev` (the exact on-shell `dft0` diagonal) and has no `omega_ev`
  or `sigc_omega_ev` curve.
