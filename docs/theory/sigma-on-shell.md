# On-shell Σ_c

A dynamic Σ_c (GN/HL-PPM, MPA, shared-pole, the four-current sectors) is
integrated from its time-domain nodes σ(τ) by the minimax window quadrature
([Σ(ω) quadrature problem](sigma-quadrature-problem.md)). Every consumer of
the result reads it at the energies of the states it belongs to: the QSGW
ansatz

    Σ^QSGW_mn(k) = ½ [ Σ_mn(k, E_mk) + Σ_mn(k, E_nk) ]ʰ

and the eqp writers' band diagonal Σ_nn(E_n) with its slope. So the τ sweep
folds each σ(τ) into those reads directly instead of into an ω grid. This is
the only evaluation: the ω-grid path (`sigma_omega_grid_debug`) is retired
([below](#retired)).

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

## Retired with the grid {#retired}

The ω-grid evaluation and everything only it fed are gone (UPGRADE_NOTES,
"Sigma_c on shell only"). The deck keys refuse by name at parse time
(`GATE omega_grid_retired`).

- `sigma_omega_grid_debug` and the `(n_ω, nk, nb, nb)` cube, its
  `sigma_mnk.h5` `omega_ev` axis and the grid's ±0.5 eV Z stencil.
- `write_eqp2`, `eqp2_tol_ev`, `eqp2_max_iter`, `eqp2_accelerator`,
  `eqp2_history_depth`, `eqp2_file`: fixed-Σ evSC (`eqp2.dat`) re-read one
  Σ(ω) at moving energies in a rotating basis; with it went the rCROP
  accelerator and the EQP2 scissor partition.
- `qp_omega0_ev` (the H₀ + Σ_x + Σ_c(ω = 0) ladder): on shell there is no
  ω = 0 matrix.
- An SC run's DFT-basis Σ_c(E_DFT): the pair's slots belong to the QP basis's
  band indices, so no rotation acts on them. `sigma_diag.dat` omits the
  `sigC` and `sigXC` columns under SC and names them in a `# omitted:` header line; the SC
  eqp files are the SC spectrum and do not depend on it.
- `sigma_freq_debug.dat` keeps its one-shot columns from the on-shell
  operand and the head at E_DFT; under SC it is omitted by name.
- The C(ω) curve in the eqp receipt: `eqp_bgw.make_eqp_bgw` refuses an old
  ω-grid `sigma_mnk.h5` (`GATE omega_grid_receipt_retired`), and its
  `--finite-difference-spacing` option is gone.
- The MPA sweep checkpoint (the on-shell sweep is recomputed on restart; an
  on-shell checkpoint is a follow-up).
- `python -m gw.plane_wave_pipeline --deck`: the JSON keeps `sigc_at_e_ev`
  (now the exact on-shell `dft0` diagonal) and drops `omega_ev` and
  `sigc_omega_ev`.
