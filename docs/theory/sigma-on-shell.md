# On-shell Σ_c

A dynamic Σ_c (GN/HL-PPM, MPA, shared-pole, the four-current sectors) is
integrated from its time-domain nodes σ(τ) by the minimax window quadrature
([Σ(ω) quadrature problem](sigma-quadrature-problem.md)). Every consumer of
the result reads it at the energies of the states it belongs to: the QSGW
ansatz

    Σ^QSGW_mn(k) = ½ [ Σ_mn(k, E_mk) + Σ_mn(k, E_nk) ]ʰ

and the eqp writers' band diagonal Σ_nn(E_n) with its slope. So the τ sweep
folds each σ(τ) into those reads directly instead of into an ω grid. This is
the default; `sigma_omega_grid_debug = true` keeps the grid.

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
`(2, k, nb, nb)` plus `(n_slot, k, nb, p_y)` diagonal slots, against
`(n_ω, k, nb, nb)` on the grid. An anti-Hermitian window closes
`(Z − Z†)/2i` with the slots exchanged: Z(ω_m)† is conj(Z(ω_m)ᵀ), whose
frequency rides the column index, so `pair[0] += (Z₀ − Z₁ᵀ*)/2i` and
`pair[1] += (Z₁ − Z₀ᵀ*)/2i`. The window planner is unchanged; it plans on
the distinct evaluation energies (`ppm_sigma.on_shell_request`).

## The evaluation tables {#tables}

The energies are fixed before the sweep, measured from the body's ω
reference, and read by the grid readers' own rules, so the on-shell numbers
are the grid's without its interpolation:

| table | energies | rule | reader |
|---|---|---|---|
| `pair` | E_eval | `sigma_out_of_grid` (`qsgw_utils.sigma_eval_omega`) | the QSGW pair, the band-extrapolation fit, dΣ/dω |
| `eval0` | E_eval | uncovered → Σ(ω = 0) | eqp1 at E_eval (SC) |
| `dft0` | E_DFT | uncovered → Σ(ω = 0) | eqp0, `sigC(Edft)` |

A table equal to an earlier one is not repeated (one-shot: one table, two
slots). "Covered" is the extent of `config.omega_grid_ev`, which the SC
`cover` growth moves as before; the grid's step is not read. Z is
`1/(1 − Re dΣ/dω)` on covered states and 1 elsewhere, which is what the
grid's two clamped probes gave there. The readers check that they ask for the
energies the sweep used (`GATE on_shell_reads`).

Z is the analytic derivative, not BerkeleyGW's ±0.5 eV central difference
(`finite_difference_spacing`). The two differ by the curvature of Σ over
±0.5 eV.

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

## What needs the grid {#grid}

These read Σ at energies or in a basis the sweep did not evaluate. On shell
they refuse or are omitted by name:

- `qp_solver = eqp2` (fixed-Σ evSC) refuses (`GATE eqp2_needs_omega_grid`):
  it re-reads one Σ(ω) at moving energies in a rotating basis.
- An SC run's DFT-basis Σ_c(E_DFT) (`sigma_diag.dat` `sigC`, written NaN).
  The pair's slots belong to the QP basis's band indices, so no rotation acts
  on them. The SC eqp files are the SC spectrum and do not depend on it.
- `sigma_freq_debug.dat`, `qp_omega0_ev`, and the SC grid-edge ambiguity
  census.
- The MPA sweep checkpoint (on shell the sweep is recomputed on restart).
