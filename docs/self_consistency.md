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
protected window of 8 valence bands on Si reaches −12 eV. Protected states outside the requested window are reported by
`BandPartition.warn_if_protected_outside_grid` in debug output. Accepted
maps grow sampled support before evaluating those states; the production
record gives one partition summary per map. The common energy-dependent
pad controls hysteresis and quadrature certification (pitfall 16). Do not chase
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

**11. Spin-orbit systems are untested in the loop; metals are covered by pitfalls 16-19.** MPA on
metals (`mpa_material_class = metal`) has wider pole widths and partial
occupations; the non-identifiability of pitfall 2 is worse when the fit family
is richer. Bi (bispinor) and Na are the pending cases.

**12. Quadrature rules are frozen across maps** (2026-09-03). The first map
plans and certifies one rule per product window on the window's box padded by
`sc` state padding; later maps reuse the rule (`cache=hit:sc-fixed`) and, when
no existing certificate with the same causal conjugation covers a window,
rebuild that rule on the escaped box with the same padding
(`rebuild:sc-fixed`, counted in the geometry receipt and printed per map;
main 0cfaf059). One-shot results are bit-identical with and without the
freeze. When no crossing certificate exists for a causal sign at initialization,
its sign-definite rules reserve the zero-side channel down to eta; this covers
newly live low poles while preserving the sign topology. The He growth gate
requires zero rebuilds on its growth map. The eqp1 file is written from the
converged map.

**13. Map gain is a diagnostic, not a controller.** From map 2 onward the
driver prints `SC map gain: max |dSigma_on-shell| / max |dE_in| = ...`, using
the adjacent changes over the non-scissored set, and stores the same gain and
worst-state tuple in the `eqp0_iterNNNN.dat` / `eqp1_iterNNNN.dat` comments.
The ratio predicted every failure in the 2026-09-03 study; a value above 1 is
evidence that the sampled map is not contracting. It does not change damping,
convergence, refusal, or any other control decision (TASTE 59); pitfall 9's
budget rule still owns when a run stops.

**14. The accelerator must see the same state set as the criterion.** The
rCROP carry is the whole active window in the DFT basis, but the scissored
bands are re-derived from their own (alpha, beta) refit each map and move by
electron-volts per map (Na eta=0.5, 86 bands on a [-15,+18] eV grid: bands
41-86 at 30-96 eV moved 0.7-3.9 eV per map while the in-range bands 5-10 moved
35-70 meV). When those entries entered the least squares, the accelerator's
trials wandered (entry mu 2.09 eV at map 2, in-range residual 3.7 eV). The
Gram and the residual norms are now taken over the non-scissored block only,
the per-k outer product of the current ``protected | in_range`` identity mask;
the update
still mixes the full carry and the scissored rows keep following the map. A
full window (a semiconductor with every band in range) has weights of exactly
1.0 and is bit identical to the unweighted solve. The log line is
``SC rCROP metric: Gram over the per-k non-scissored DFT identity block``.
The caller owns a per-state Boolean mask, refreshed only on accepted calls;
rCROP forms its outer product on device. Discarded probes cannot change
the quadratic form used for the accepted history.
On metals a local gain above 1 can remain at Fermi-crossing states, where the
exchange term responds to an occupation flip under the smearing width; that is
the physics of the map, not the accelerator, and is read from pitfall 13's
line.

**15. One quadrature acceptance on every path.** The one-shot planner, the
fixed-SC initializer and its rebuilds all require the certified sup error at
or below `sigma_quadrature_eps`; a build that misses gets one retry with five
times the reduction budget and then refuses, naming the window, its box,
the achieved sup, the node count and the remedy. The 2026-09-03 bypass
(`enforce_sup_error=False`) let Na retain a conduction pole-tail rule at
sup=0.0405 against eps=1e-4 with 906 nodes in every self-consistent arm; its
actual support has a 24-node rule at eps (sandbox lane QUADCHECK). The cause
was the SC pad: the ten-percent pole pad pushed a strictly negative support
across zero and asked for a crossing rule. The pad now keeps sign-definite
supports sign-definite (the zero-side edge moves at most halfway to zero;
a support that really crosses later is a box escape and rebuilds), and the
disk cache never returns a certificate above eps. Do not loosen eps to admit
a rule.

**16. One energy-dependent pad supports classification and quadrature.**
A DFT-labelled band enters the protected set when its current all-k energy
range lies entirely inside the requested, mu-anchored window. Previously
protected members remain protected while their range lies within that window
padded by `pad(E) = 0.5 eV + 0.10 |E - mu|`. The pad is 0.5 eV at mu,
1.5 eV at 10 eV and 2.5 eV at 20 eV. This replaces the inward 2 eV edge
margin and frozen sorted-index set: near-mu motion is small, while Na's
spectrum stretches by about ten percent and the upper states moved 1.2–2.2 eV.
Classification runs on every map, including rCROP trials, but trials do not
commit classification memory or grow the sampled grid. On an accepted escape
map the identity remains protected, its grid grows before Sigma evaluation,
and convergence is forbidden. The next accepted map scissors it; re-entry
requires the entire band range inside the requested window. The production
summary names the escape's band, k, energy and pad; debug output carries all
per-state rows. Quadrature retains its rebuild-on-box-escape behavior.
The quadrature state support uses the same pad; pole padding and certified
acceptance are unchanged. The sampled omega grid is the REQUESTED grid at
map 0 for every state, so SC iteration 1 equals the one-shot and a state
outside the requested window keeps the one-shot treatment (pre-padding the
grid re-evaluated such states and moved the GN-PPM invariance fixture by
0.28 eV, 2026-09-05). When a retained state drifts past the requested
bounds, only the outer sampled endpoints grow, to the escaped energy plus
its pad (`SC sampled-support growth: ...` in debug output), rounded outward
in blocks of four samples. Old samples remain unchanged and the quadrature
session keeps the grown support on later maps and trials.
Certificates reserve this prospective external-frequency extent, including
the growth pad and grid rounding, without evaluating those samples at map 0.
Padding intermediate states alone does not cover external-frequency growth.
Changed product-window membership, state support or pole drift can still
require a rebuild. Each frequency branch reserves outward growth only:
negative frequencies toward the prospective lower extent, nonnegative
frequencies toward the upper extent. Its inner endpoint remains unchanged;
reserving unsampled support toward zero can turn a sign-definite pole tail
into an expensive crossing rule. Genuine new pole channels follow the
reported rebuild-on-escape path.
Contained initial certificates of the same causal type share their nodes. A
replacement certificate must also use the same relative or peak-relative
error criterion; a small sign-definite window must not inherit a large
crossing rule just because its box is contained. A near-boundary cache
request is accepted only after re-auditing the unchanged
nodes on the union box at the same sup-error and runtime-noise limits; the
small boundary threshold selects audits, never relaxes a certificate.
Interior holes still require an explicit patch. Quadrature
nodes remain frozen while their certified boxes cover the map.

**17. The active-window scissor law stays frozen at map 0.** States inside
the active Sigma band window but outside the retained self-consistent block
follow the affine law fitted at map 0 (`SC scissor: frozen from map 0
(...)`). Measured against a per-map refit at convergence, with the pad and
per-k identity partition of pitfalls 16 and 18 (Na eta=0.5, +19 eV, trusted
5-10, three arms to accepted map 16, sandbox claim 946): the refitted law
drifts from alpha 1.051 to 1.140 and beta -1.23 to -1.72 eV over the loop,
the refit arm converges more slowly (per-pair motion at 14->16 up to 19 meV
against 7 meV frozen; residual 11.1 against 4.9 meV) and settles 127 meV
away rigidly, with bands 5-8 up to 95 meV and bands 9-10 up to 411 meV apart
after alignment, while the Fermi band's shape agrees to 5.8 meV. The far
states' correction following the trusted block's stretch every map feeds
back through the sum over states; one fixed correction, as in standard
QSGW, is the closure. The sum-band tail beyond the Sigma window keeps its
per-map refit: freezing that distinct tail moved the Si b80/c504 gap by 22
meV at map 6.

**18. Partition identities are per k, and Hamiltonian masks use the carry's
basis.** Overlap assignment against reference DFT multiplets finds the sorted
QP columns carrying each identity on every map. Classification uses those raw
assigned energies. Whole reference multiplets are protected locally at each
k; promotion at one k does not transitively promote the same label at every
other k. The Hamiltonian and rCROP carry are in the DFT basis, so their masks
are `(k, DFT identity)`, with sorted-column correspondence printed explicitly.
Applying sorted-column masks directly to that carry would protect the wrong
states at a crossing. Scissor fits preserve paired DFT/QP identity columns,
including protected states retained outside the requested edge; they never
sort QP samples independently. Metal Fermi-class masks follow the same
per-k input-column assignment. The same masks select the non-scissored convergence
criterion, rCROP Gram block and identity comments. The eqp body retains sorted
eigenvalues and the identity comments retain DFT-band labels. Each map reports
bands protected at all k and k rows where the sorted protected columns differ
from the reference labels.
For the motion readout, the first map output supplies a fixed QP reference
labelled by its overlap with whole DFT multiplets. Later input and output
columns are assigned to those reference multiplets over all active candidates;
their block means define the identity criterion even if a multiplet splits.
`SC_identity` comments describe eqp0 motion in both eqp0 and eqp1 files and
use the file's k-block index (the first integer in a body row is spin).

**19. On metals the self-consistent set should stop where the quasiparticle
stops being well defined; convergence time is set by the largest Z in the
set.** Na eta=0.5 (86 bands, 8x8x8, 896 centroids), three historical trusted sets were run
with the former 2 eV inward margin: band 5 alone (window top +11 eV), bands 5-10
(+21 eV) and bands 5-13 (+24 eV). Per accepted map, the unmixed output
motion of every band with map-0 quasiparticle weight Z in (0.55, 0.8)
(bands 5-8, Z from the eqp1 diagnostic `(eqp1 - E_DFT)/(eqp0 - E_DFT)`)
halves; bands 9-13, whose Z is 0.97-1.74 (more than a plasmon energy above
mu, where Re Sigma(omega) is flat or rising on shell), walk monotonically at
map gain about 1 and reach their fixed point 1.3-2 eV above G0W0 only after
12-14 accepted maps. The Fermi band's shape is a window-independent
observable: its k-resolved energies agree to under 10 meV across the three
sets (bandwidth 6.831 / 6.827 / 6.837 eV against 6.581 DFT-seeded), while
its absolute position moves 41 meV when bands 11-13 join the set (bands
5 alone and 5-10 agree to 2.5 meV). Bands 5-10 converge to about 1 meV per
map by accepted map 22 and reach a 2 meV rCROP residual at map 28; 5-13 is
still at 5-9 meV per map on bands 12-13 at map 26. Those measurements used `sigma_omega_max_ev = 21` to select 5-10.
Under the all-k entry rule in pitfall 16 the same +21 eV request admits
5-12 and promotes 13, because the inward margin has been removed; that run
then paces like the old +24 arm (bands 11-13 at 240-440 meV per accepted
pair at map 10). A window value no longer implies the old membership: on
Na, `sigma_omega_max_ev = 19` gives trusted 5-10 (band 11 tops out at
20.92 eV absolute), and with that set the pad partition tracks the
margin-based record map for map (aligned per-pair motion within a few meV
from map 6 on, residual 4.9 meV at map 16 against 7.2, Fermi band shape
within 2.1 meV at map 16 and 2.9 meV of the converged reference; sandbox
claim 937). Production on Na is therefore `sigma_omega_max_ev = 19`,
trusted 5-10. Report the actual identity masks and the Fermi-window
observables, and do not infer observable convergence from the rCROP
L-infinity residual alone while Z >= 1 states walk. A
partition by quasiparticle well-definedness (Z below about 0.9 at every k
at map 0) instead of by energy window is registered; in Na the multiplet
chain linked bands 5-10 under global promotion; local per-k closure now
avoids that transitive union. The single measurement that
decides the next structural change is a small-kick response at a retained
input: chi0, the body W and the final trusted-block H scale linearly
(ratio 10.1-10.6 for a 10x kick) while the fitted scalar head does not
(48.6, pole count changes), which names the head MPA refit as the first
non-smooth stage of the map.

## Evidence

Sandbox reports (paths under
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/reports/`):
`sc_map_sensitivity_2026-09-03` (claim 661, stage-wise gain and the frozen-W
control), `si_centroid_ladder_sc_2026-09-03` (resolution ladder),
`si_bands_dft_g0w0_qsgw_2026-09-03` (claim 668, the converged decks and band
structures), `sc_fixed_rules_eqp1_2026-09-03` (claim 662, frozen rules),
`sigma_eta_literal_no_ceiling_2026-09-03` (claims 637/639),
`qsgw_two_level_2026-09-03` (the frozen-W inner loop, branch only).
