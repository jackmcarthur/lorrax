# Self-consistent GW (QSGW) in LORRAX — safe parameters and pitfalls

**Status as of 2026-09-03.** `qp_solver = self_consistent` is the shipped
quasiparticle self-consistent loop. It converged in 13–15 maps to 0.07 meV on
Si (4×4×4, 80 bands, 504 centroids) for both `compute_mode = mpa` and
`gn_ppm`, and in 13 maps on monolayer MoS2 (GN-PPM). Every statement below
carries the run it was measured on; the sandbox reports are named in
[Evidence](#evidence). Nothing here is a theorem about your material.

This page owns the *how to run it safely* facts. The keys themselves are
defined in [input_reference.md](input_reference.md); the drivers are in
[drivers.md](drivers.md); the Σ(ω) quadrature is in
[dev/crossing-rule-cost-law.md](dev/crossing-rule-cost-law.md).

## What the loop does

One map is `H → rotate ψ → χ₀ → W → screening model (poles) → Σ(ω) → H'`.
The map is iterated with rCROP (`sc_accelerator = rcrop`, history 5) until
every non-scissored band moves by less than `sc_tol_ev`. Within the active
subspace (`nval + ncond` bands around E_F) each band is in one of three
classes (`gw/band_partition.py`):

The loop has one k-set invariant: its retained H, E, U, every k-indexed
`SigmaResult` table, and density-SC Hartree components all carry exactly the
loop's k-set (the star wedge when `sc_on_ibz = true`). The full BZ exists only
inside a map while the k-grid FFT builds Sigma, and in the separate one-shot
writer path. At the map boundary one named seam selects the complete retained
result and its defining U together; diagnostics and writers consume that
selection and never select or broadcast an individual Sigma table again.

| class | diagonal of H' | off-diagonals |
|---|---|---|
| protected (inside the Σ(ω) grid) | full Σ at the QP energy | kept, protected×protected only |
| non-protected, inside the grid | Σ at the band's own energy | zeroed |
| outside the grid | α·E_DFT + β, refit every map from the in-range corrections (`sc_tail_fit = frontier`) | zeroed |

The loop is driven by `eqp0` (the Σ evaluated at the current energies). No
Z-factor enters the iteration; `eqp1` is written as the BerkeleyGW-style
linearized output only. After convergence `eqp1` and `eqp0` agree to about
1 meV, which is a cheap check that the fixed point is real.

## Production requirements (owner rulings, 2026-09-03 evening)

The deck below is the *diagnostic* deck the convergence study used. It is
not a production setting, and band structures drawn from it are not
interpretable. A production self-consistent run must satisfy:

- at least **20 conduction bands in the Σ window** (`ncond >= 20`); the loop's
  partition decides which of them are protected;
- **centroids ≥ 10 × `number_bands`** (80 bands → at least 800), the nearest
  orbit-closed count at or above that; 192 or 504 centroids are diagnostic
  only;
- the ζ/ISDF fit built on the Gram matrix of **all bands that enter Σ**
  (`zeta_nband = number_bands`), not a 16–20-band window; if the strict rank
  ceiling refuses, the owner decides `zeta_rcond`, the run does not drop to a
  smaller basis;
- `use_band_extrapolation = true` (the default), named explicitly;
- on metals the MPA route only (GN-PPM refuses fractional occupations by design, owner ruling 2026-09-03); the two-level (frozen-W inner) loop is discontinued and stays a diagnostic branch;
- band-structure interpolation (htransform) fitting the whole WFN band set and
  returning at least **16 corrected conduction bands**, guard bands ≥ 8. A
  band-structure workflow must request its own dense uniform NSCF/WFN for
  htransform and a separate QE `calculation='bands'` along the same path; it
  must not inherit the GW screening mesh as its DFT reference. Start Si-class
  cells at 8x8x8, or use the first material-specific grid whose certificate
  passes. Four returned conduction bands destroy the interpolant. The
  **htransform coarse k-grid is a production convergence parameter independent
  of the GW screening grid**: densify it until the energy-ordered,
  per-path-VBM-aligned QE certificate is at most **20 meV for every plotted
  cell whose QE energy lies in the inclusive [-8,+8] eV window**. The receipt
  must also report the all-state maximum; cells outside the window do not gate.
  Whole-WFN fitting, a larger Galerkin rank, guard bands, and a different
  f-transform scale do not replace this grid test. On Si, 4x4x4 and 6x6x6 miss
  by 120.424 and 33.905 meV, while 8x8x8 passes at 10.869 meV; do not publish a
  curve from the coarser diagnostic grids merely because the GW correction
  itself was computed there. On monolayer MoS2, the 64x64x1 and 72x72x1
  all-state maxima remain 26.987 and 31.646 meV on the lowest valence pair,
  about -60.2 eV relative to the path VBM, while their [-8,+8] eV maxima are
  9.327 and 8.117 meV. This non-monotone deep-pair error is a known finite-mesh
  interpolation limitation outside the publication window, not a state-label
  or f-transform-scale correction.

## The diagnostic deck the study converged on

These are the live keys of the Si run behind the band structures in
`reports/si_bands_dft_g0w0_qsgw_2026-09-03` (sandbox, superseded by the
production reruns of 2026-09-03 evening). Keys marked *deck* are not defaults
and must be chosen per material.

```ini
qp_solver = self_consistent
sc_max_iter = 30             # deck; default 20 (13–15 maps when healthy)
sc_tol_ev = 1e-4             # default
sc_accelerator = rcrop       # default
sc_history_depth = 5         # default
nval = 8                     # deck; default 5
ncond = 8                    # deck; default 5
number_bands = 80            # deck — see pitfall 1
compute_mode = mpa           # or gn_ppm
linalg = distributed         # one layout dial for production-sized matrices
low_mem_bands = true         # automatic band chunks; default false
mpa_n_poles = 8              # default
sigma_omega_min_ev = -15.0   # deck; default -5 — see pitfall 3
sigma_omega_max_ev = 15.0    # deck; default +5
sigma_omega_step_ev = 0.25   # default
sigma_regularization_ev = 0.25   # default; literal η — see pitfall 4
sigma_quadrature_eps = 1e-4      # default — see pitfall 5
use_band_extrapolation = false   # diagnostic study; production = true — see pitfall 6
restart = true                   # deck — see pitfall 7
zeta_rcond = 1e-10               # deck; production default 1e-8
```

with 504 ISDF centroids for 80 bands (about 6 per band; a production run needs 10 per band, see above).

## Pitfalls, in the order they bite

**1. An under-resolved screening support makes the map non-contractive.**
On Si with 24 bands and 192 centroids the same loop plateaus at 1–70 meV and
never converges, for both ansätze, with any accelerator. A 0.1 meV kick to one
eigenvalue moved the on-shell Σ_c by 3 meV and the H eigenvalue by 1.4–1.8 meV
(gain 14–18); at 80 bands / 504 centroids the gain is 0.3 and the loop
contracts. The symptom is a residual that stalls or oscillates in the meV
range after the first few maps. The remedy is resolution, not loop settings:
about 6 centroids per band and enough bands that the top of the active window
is far from `number_bands`. A 12× centroid count is past what the ζ fit
certifies at the default rank ceiling (strict mode refuses); 6× is the usable
top on the Si cell.

**2. The pole refit, not the physics, is the discontinuity.** The
imaginary-axis samples of χ₀ and W respond linearly and minutely to an
eigenvalue change (decade scaling exactly 10). The MPA Loewner/Padé solve from
16 samples per element is not identifiable: many pole sets reproduce the
samples to tolerance, and a 1e-7-level change in the samples lands on a
different member (poles move 50–100 eV). Two pole sets that agree on the
imaginary axis differ on the real axis, so Σ(ω) jumps by 10–20 meV somewhere in
the cube. The response does not scale with the kick, so this is a jump between
equivalent fits, not a gradient, and no Z-guard, damping, or mixing schedule can
converge it. GN-PPM modes that cross Ω² < 0 into the `static_limit` branch
carry about 1e-12 of the residue mass and are not the cause. Resolution
(pitfall 1) makes the ambiguity harmless for Σ on the real axis; a frozen-W
inner loop (branch `feat/qsgw-two-level-2026-09-03`, not on main) removes it
structurally.

**3. The Σ(ω) grid must contain every protected band at every k, and must not
be wider than it needs to be.** The default grid is ±5 eV around E_F; a
protected window of 8 valence bands on Si reaches −12 eV. A protected band
outside the grid triggers the all-caps warning from
`BandPartition.warn_if_protected_outside_grid`, and its off-diagonals then mix
edge-clamped Σ values into the eigenproblem. Set the grid to the window plus a
margin (2 eV is enough for the frozen rules, pitfall 12). Do not chase
semicore states by widening: the node count of the crossing rules grows like
bandwidth × ln(10/ε)/(π η), and a [−90, +20] eV CrI3 grid at η = 0.25 eV cost
80 min per Σ evaluation on 16 GPUs. Keep the grid within ±15 eV and let deeper
states take the scissor tail (owner ruling 2026-09-03). Deep bands therefore
move with the frontier fit, not with their own Σ; say so when you plot them.

**4. `sigma_regularization_ev` is the physics, not a knob.** Since 2026-09-03
η is literally the Lorentzian broadening every ansatz runs at; the automatic
floor and `sigma_regularization_floor_ev` are gone, and `mpa_sigma_max_nodes`
is gone with the pair ceiling. Halving η roughly doubles the crossing-rule
node count and changes Im Σ; do not use it to buy speed. Decks written before
that date that spell either retired key refuse by name.

**5. `sigma_quadrature_eps` below 1e-4 may refuse at the round-off gate.** The
planner refuses a rule whose round-off amplification exceeds 0.05·ε/6e-8 (83
at ε = 1e-4, 8.3 at 1e-5). That refusal is a certification, not a failure:
tightening ε does not make Σ more accurate once round-off dominates. The
builder/consumer κ-cap mismatch that produced spurious refusals at ε = 1e-5
on sign-definite rules is fixed in `sigma_box_plan.py` (2026-09-03).

**6. `use_band_extrapolation` is TRUE by default and is the production setting**
(owner, 2026-09-03: "use band extrapolation in future runs"). Every
self-consistency result in the 2026-09-03 diagnostic study was measured with
it FALSE; the production reruns carry it TRUE with one FALSE control. Name the
key explicitly in an SC deck either way, and if an extrapolation-on loop fails
where its control converges, report it rather than tune around it.

**7. `restart = true` reuses the ISDF/W tensors of a finished run and is the
right way to start a loop from a converged one-shot; it is not safe against a
directory holding valid MPA pole stores it would overwrite.** Point a restart
at a copy or a variant directory (sandbox rule: never mutate a completed run).

**8. Degeneracies are symmetrized only when exact.** `sc_exact_degeneracy_tol_ev`
is 0.1 meV, deliberately below any physical splitting (MoS2's SOC-split K pair
is 1.7–3.6 meV). Do not raise it to make a loop converge; a near-degenerate
pair that will not settle is a window-edge or state-identity problem
(protected windows must close multiplets at every k), never a reason for
damping.

**9. Budget.** A healthy loop converges in 13–15 maps. If the residual has
not fallen below 1 meV by map 20 it will not converge at 60; stop and fix the
deck (pitfalls 1, 3, 6). Each map costs one full Σ evaluation. Read its cost
from the three Σ rows of the stage table: on Si b80/c504 at P4 a cold run is
"Sigma rule plan" 180 s (box-rule fitting, cached by box and tolerance, paid
once per rule set and reused by the frozen rules across maps), "Sigma tau
sweep" 6 s (the actual contraction, 700 τ nodes), "Sigma other" 9 s
(2026-09-03, `runs/DEV/123`). A QSGW map that shows minutes of rule planning
after map 1 is rebuilding rules; check `sc_fixed_rebuilds_this_iteration`.

**10. Convergence is judged on the non-scissored bands only**
(`protected_band_convergence`). Scissored bands are α·E_DFT + β with the
coefficients refit each map, so including them would re-count in-range drift.
`max|dE|` in the log is over that set; a "converged" loop says nothing about
the tail's own Σ.

**11. Metals and spin-orbit systems are untested in the loop.** MPA on
metals (`mpa_material_class = metal`) has wider pole widths and partial
occupations; the non-identifiability of pitfall 2 is worse when the fit family
is richer. Bi (bispinor) and Na are the pending cases.

**12. Quadrature rules are frozen across maps** (2026-09-03). The first map
plans and certifies one rule per product window on the window's box padded by
`sc` state padding; later maps reuse the rule (`cache=hit:sc-fixed`) and refuse
rather than rebuild if a state leaves its padded box. One-shot results are
bit-identical with and without the freeze. The eqp1 file is written from the
converged map.

**13. Map gain is a diagnostic, not a controller.** From map 2 onward the
driver prints `SC map gain: max |dSigma_on-shell| / max |dE_in| = ...`, using
the adjacent changes over the non-scissored set, and stores the same gain and
worst-state tuple in the `eqp0_iterNNNN.dat` / `eqp1_iterNNNN.dat` comments.
The ratio predicted every failure in the 2026-09-03 study; a value above 1 is
evidence that the sampled map is not contracting. It does not change damping,
convergence, refusal, or any other control decision (TASTE 59); pitfall 9's
budget rule still owns when a run stops.

## Evidence

Sandbox reports (paths under
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/reports/`):
`sc_map_sensitivity_2026-09-03` (claim 661, stage-wise gain and the frozen-W
control), `si_centroid_ladder_sc_2026-09-03` (resolution ladder),
`si_bands_dft_g0w0_qsgw_2026-09-03` (claim 668, the converged decks and band
structures), `sc_fixed_rules_eqp1_2026-09-03` (claim 662, frozen rules),
`sigma_eta_literal_no_ceiling_2026-09-03` (claims 637/639),
`qsgw_two_level_2026-09-03` (the frozen-W inner loop, branch only).
