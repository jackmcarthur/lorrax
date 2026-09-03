# DESIGN — the self-consistent (QSGW) loop: one update law, band classes fixed by index

Owner's scheme, 2026-09-03; architecture by the coordinator; implementation and the
window-consistency study by a Codex lane (`runs/DEV/105_sc_loop_2026-09-03` in the sandbox).
Supersedes the active/inactive partition, the per-k in-range masks with hysteresis, the
buffer bands, the frontier / all-conduction / buffer-edge tail laws and the affine
out-of-range scissor fit that `sc_iteration.py`, `band_partition.py`, `scissor.py` and the
SC half of `band_extrapolation.py` carry today (about 11k lines, 37 consumer sites in
`sc_iteration.py` alone).

## 1. Objects

**`SigmaTable`** — Σ_c(ω; k, m, n) on the deck's uniform, Fermi-relative grid
[ω_min, ω_max] for the protected band window P (rows P, columns P), plus the static
pieces (Σ_x, V_H, head terms) and, new, the single-frequency block Σ_mn(E_F) for m ∈ P,
n ∈ U (see §2). Built once per iteration in the current QP basis exactly as today
(screening and Σ recomputed with rotated wavefunctions), or held fixed for the
`eqp2` eigenvalue loop. `SigmaTable.at(E)` is piecewise-linear interpolation on the
grid and knows nothing about policies; a request outside the grid is a programming
error there, not a clamp.

**`BandClasses`** — decided once at setup, by band INDEX, from the deck: P = the Σ
window `[b0, b3)`; U = the sum-band tail `[b3, b4_user)` (Σ elements among U are never
computed); mesh padding is ignored. Membership never changes during the loop: no
reclassification, no hysteresis margin, no per-k in-range mask, no buffer bands.

**`EvaluationPolicy`** — the map from the current QP energies to the frequencies at
which Σ is read for a P×P pair. Two named values, both implemented for the study;
the owner picks one from the measurement and the other is deleted:
- `fermi` (the owner's rule): both E_m and E_n inside [ω_min, ω_max] → (E_m, E_n);
  otherwise (E_F, E_F).
- `clamp`: (clip(E_m), clip(E_n)) always, clip to [ω_min, ω_max].

**`effective_sigma(table, classes, policy, E, E_F) -> Σ̃_k`** — THE update law, one
function, Hermitian by construction:
- P×P: ½[Σ_mn(ω_m) + Σ_mn(ω_n)] with (ω_m, ω_n) from the policy, Hermitized;
- P×U and U×P: Σ_mn(E_F) — the U states rotate with the protected block;
- U×U: zero off-diagonal; diagonal E^DFT_n + Δ;
- Δ (the scissor): one number per iteration and spin, the mean over k and over the two
  highest protected conduction bands of [Σ̃_nn − V_xc,nn]. No affine fit, no per-band
  law. Its sensitivity is measured (§4), not tuned.

**`QpHamiltonian`** — H_k = h0_k + Σ̃_k − V_xc,k in the DFT basis (the rotate-back seam as
today), `eigh` → (E, U) ascending; the Fermi level as today (midgap for insulators,
occupations for metals). The carry is H in the DFT basis so rCROP / linear mixing stay
meaningful; no state tracking, no damping of near-degenerate pairs (TASTE 59).

**Loop** — iterate the map to `max|ΔE| < sc_tol_ev` over P. The `eqp2` fixed-table loop
is the same `effective_sigma` on a fixed `SigmaTable`; it stops having its own partition
path.

## 2. The P×U block

`effective_sigma` needs Σ_mn(E_F) for m ∈ P, n ∈ U: one frequency, not the ω table. On
the packed route the accumulator already builds Σ at the mesh-divisible full band
extent and strips to the window (`gw/mpa/sigma.py`, face-carrier note), so the columns
exist; on the legacy layout the projector's column extent must be widened for that one
frequency. The lane prices both; if the legacy widening is expensive, the block is
produced by a single-ω evaluation at E_F (it is static in ω), which is cheap.

## 3. Robustness (the owner asked)

- Rule P×P-inside is standard QSGW (mode A). Robust.
- Rule P×P-outside → Σ(E_F) is a hard switch when a QP energy crosses ω_max or ω_min
  between iterations: Σ̃_mn jumps by Σ_mn(E_edge) − Σ_mn(E_F), which for a diagonal
  element is O(100 meV) and can make a state that straddles the edge cycle — the
  window-edge oscillation of claims 620/624 under the old tail laws had this shape. The
  `clamp` policy is continuous in E, so the fixed point exists and the accelerators
  converge; for a state one eV beyond the edge the edge value is also closer to Σ(E)
  than Σ(E_F) is. Expected outcome of the study: `clamp` wins on both consistency and
  iteration count; the owner rules on the measurement.
- Rules P×U at E_F and U×U scissor are the standard static outer block. Because P/U
  is by index there is no reclassification; the only artifact is a mismatch between the
  protected block's correction at its top and Δ, visible at the P/U boundary and
  measured by the nb_P ladder (§4).
- Metals: classes by index, Fermi level from occupations, and the E_F-evaluated P×U
  block is the natural choice; nothing metal-specific is added.

## 4. The study (acceptance)

Systems: MoS2 3×3 GN-PPM (the existing SC deck family, P4, `sigma_omega_step_ev = 0.25`)
and Si 4×4×4 (GN-PPM and MPA). Arms, each run to `sc_tol_ev = 1e-4`:
- ω window ±10 eV vs ±15 eV (same P, same nband);
- policy `fermi` vs `clamp`;
- nb_P ladder (e.g. 8, 16, 24 protected bands at fixed nband).
Deliverables per arm: per-iteration max|ΔE| and its history plot, iteration count,
final QP energies on the k-grid, and the band structure along the standard k-path from
the `bandstructure` driver rendered to PNG (matplotlib; ±10 vs ±15 overlaid; `fermi` vs
`clamp` overlaid; nb_P ladder overlaid). Tables: max/RMS deviation over P bands between
the two windows within ±8 eV of E_F; the P/U boundary mismatch; iteration counts.
"Robust" means: the two windows agree to ≤ 10 meV over the bands within ±8 eV, no
window-edge oscillation (monotone max|ΔE| after the first two iterations), and the
nb_P ladder moves the lower bands by ≤ 10 meV. Report the numbers whatever they are.

## 5. Deprecation (same branch, second commit, after the study passes)

Delete: `BandPartition` and `band_partition.py`, `_state_partition`,
`_partition_hysteresis_margin_ev`, `sc_buffer_*` keys and `_apply_sc_buffer_partition`,
`sc_tail_fit` and the frontier / all_conduction / buffer_edges laws,
`band_extrapolation`'s SC rulings (`sc_tolerance_ruling`, `static_limit_tail_ruling`),
`_scissor_E_qp_for_outofrange`, `_apply_scissor_partition_policy` and the SC use of
`ScissorFit`, `one_sided_core_mask` in `build_qsgw_sigma_xc`, the `eqp2` partition
path, with their tests and fixtures; retired keys refuse by name. The Σ-sum band
brackets (`band_brackets`/`band_counts`) are a Σ-convergence device, not SC: untouched.
Keys kept: `sc_max_iter`, `sc_tol_ev`, `sc_accelerator`, `sc_history_depth`,
`sc_mixing`, `sc_exact_degeneracy_tol_ev`, `sc_eigh`, plus the one the study decides
(`fermi`/`clamp` becomes the fixed behaviour, not a key). Target: the loop in ≤ 1.5k lines.
