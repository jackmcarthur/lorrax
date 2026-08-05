# ROOT CAUSE — 4-GPU vs 16-GPU eqp divergence (synthesis of agents A/B/C, 2026-07-08)

Inputs: `agent_A_stage_bisect.md`, `agent_B_padding_audit.md`, `agent_C_ffi_shard_audit.md`
(same run pair: `runs/MoS2/Z_memplanner_validation_2026-07-06/{A_charge,B_bispinor}/head_{4g,16g}`).

## 1. Root cause

**One root cause, two manifestations: computations that should run on the LOGICAL μ extent
are run on the PADDED extent (n_rmu_padded = round_up(n_rmu, P)), so the pad extent — which
changes with device count — deterministically changes the answer.** This is a *real
pad-extent defect*, not a P-stochastic reduction-order effect. It is then amplified to eV by
two known ill-conditioned consumers (near-singular transverse CCT; on-pole GN-PPM Σ_C), which
is a *separate robustness defect* that also needs fixing but is not the invariance carrier.

### Manifestation 1 — bispinor (PROVEN)

The transverse ζ solve (`isdf/core.py:factor_c_q` identity-pad passthrough →
`_ridge_indef_solve`, ~line 1283) runs the batched pivoted LU at the **padded** extent. The
"logical block bit-identical" guarantee claimed in `_identity_pad_block_diagonal`'s docstring
(isdf/core.py:781-791) is **false for LU**: each pad extent (668 / 672 / 684) yields a
different, per-extent-deterministic ζ_T, because shape-dependent LU roundoff is amplified O(1)
in the near-null modes of the indefinite transverse CCT. n=672 hits a catastrophic resonance
(Σ^B tile(2,2) trace −0.153 → −117.9 eV; ζ_mu2 rel 5.5e2; V_TT_22 rel 1.35e5).

Evidence chain (each link independently measured):
1. **B, dispositive:** at FIXED P=4, `LORRAX_EXTRA_MU_PAD=4` (668→672 = the P=16 extent)
   reproduces the P=16 numbers to 5 digits — Σ^B tile22 −117.914395 vs −117.914143, eqp0
   diff max 2.535 eV / median 69 meV / 270/270 bands, the identical 4g↔16g signature.
   Pad extent alone, at fixed device count, IS the bug.
2. **C, solver exonerated:** 16g rerun with `cusolvermp_lu=off` (single-device
   `jnp.linalg.solve`) reproduces the same wrong ζ_T to rel ≤3.5e-7; cuSolverMp 4×4-grid
   unit tests pass at 1e-15. The corruption is in the padded *system*, not the solver.
3. **B, ridge exonerated:** subtracting the pad contribution from the ridge trace
   (`padtest_4g_pad4_ridgefix`) changes nothing (−117.914395 unchanged).
4. **A+B+C agree the carrier is Σ^B:** sigma_diag (sigSX/sigCOH/VH) and kin_ion are
   P-invariant to 1e-6 eV; the corrupted ζ_T enters eqp only through `results.sig_x`
   (bare Breit Σ^B, `sigma_x_bispinor.py`), which sigma_diag never prints — resolving
   A's "Σ identical but eqp moves" paradox.

### Manifestation 2 — charge / GN-PPM (strongly indicated; one confirming experiment pending)

The PPM mode census and adaptive minimax window statistics run over the **padded** μ² mode
space at P=16 (C: n_total_modes 16·1216² vs 16·1204²; invalid count = 4g + 464,640 pad modes
**+ 2 flipped real modes**; masked-Ω stats feeding `ppm_windows._build_windows_for_branch`
change → minimax node counts 15→13 (b_slab) and 15→14 (single)). A **discrete, P-dependent
algorithm change** in the Σ_C fit — while everything upstream (ζ_C 2.4e-7 rel, V_q, Σ_X,
V_H) is P-invariant at print precision (A/B/C all measured Σ_X identical to 1e-6 eV at all
1261–1280 (k,n) despite 0-vs-12 pad rows).

The eV magnitude (max ΔΣ_C 5.67 eV, max Δeqp 5.64 eV) is then near-pole amplification: the
run sits ON GN-PPM poles (|Im Σ_C| 1e4–6.2e6 eV on 1030/1280 bands; d/|Im| median 2.3e-7).
On-pole Σ_C(E_dft) is intrinsically ill-posed — even a perfectly pad-clean fit will move
these bands by O(0.1 eV) under any 1e-7 perturbation. So for charge the fix is BOTH: make
the census/window stats pad-invariant (defect), AND treat near-pole evaluation as a
robustness/tolerance problem (`ppm_invalid_mode` handling), not something a gate can pin
to μeV.

### Reconciling the conflict (A vs B/C)

Agent A's headline "μ-padding exonerated" is **wrong for the eV-scale bug** and stands only
for the ~1e-7 baseline: A observed that the pad-free bispinor charge channel (640 % 16 = 0)
still shows ~2.5e-7 ζ noise, and that all pad rows/cols of dumped tensors are exactly zero.
Both observations are correct — but they only exonerate padding as the cause of the *tiny
baseline* divergence, and only rule out *pad-content* (pad-row value) leaks. The actual leak
is a *pad-shape* effect (solve extent, census extent), invisible to pad-row inspection.
B's controlled fixed-P pad-flip reproducing the exact eV signature is a strictly stronger
form of evidence (intervention vs correlation) and overrides A's inference. A's baseline
finding remains valid and important: after the pad fixes, a ~1e-7 rel P-dependent
reduction-order floor in ζ remains, which is the quantity the gate tolerance must absorb.

### The one experiment that closes the remaining gap

Manifestation 2 has C's mechanism evidence (node counts, census) but not yet B-style proof.
**Decisive experiment:** rerun `A_charge` at P=4 with `LORRAX_EXTRA_MU_PAD=12` (1204→1216 =
the P=16 extent). If it reproduces the 16g minimax node counts (13/14), the 2 flipped modes,
and the 16g Σ_C/eqp to reduction-order residual, the charge divergence is also
pad-extent-deterministic and the whole bug is closed under one root cause. (B's charge
pad-flip control was on the B_bispinor COHSEX charge chain, which has no GN-PPM stage — it
did not test this.) ~30 min on a 1-node alloc; alloc 55674933 may still be live.

## 2. Fix plan (NOT implemented — per directive)

Both invariance defects are "compute on logical extent" fixes; the robustness items are
separate and smaller in urgency but required for a meaningful gate.

### Fix 1 — transverse ζ solve at logical extent (the bispinor bug)
- **File:** `sources/lorrax_D/src/isdf/core.py` (`factor_c_q` transverse passthrough,
  `_ridge_indef_solve` ~1283).
- **Approach (per B §2, confirmed local by sharding):** L is replicated
  (`in_specs=P(None,None)`), Z is column-sharded → slice both to `n_rmu_logical` before
  `jnp.linalg.solve`, zero-fill ζ pad rows after. Also compute the ridge from the logical
  trace and logical n (C side-finding 5c). Fix or delete the false docstring guarantee at
  isdf/core.py:781-791; if `_identity_pad_block_diagonal` keeps other callers, document
  that the guarantee holds only for Cholesky (charge, measured ≤1e-7) not LU.
- **Size:** ~20–40 lines + docstring. Makes ζ_T bit-independent of pad extent.

### Fix 2 — PPM census/window stats on logical modes (the charge defect)
- **Files:** `sources/lorrax_D/src/gw/minimax_screening.py` (mode census, invalid counting,
  GN-PPM fit inputs), `src/gw/ppm_windows.py` (`_build_windows_for_branch`,
  `_masked_stats_device`).
- **Approach:** exclude pad-μ modes from n_total_modes, the invalid-mode census, and the
  masked-Ω min/max/count that drive adaptive node selection — via an explicit logical-mode
  mask (pad Wc=0 already makes B_pad exactly 0 per B §3; the leak is purely in the
  statistics/counters, so the mask is cheap). Node counts and invalid counts must become
  provably P-independent; the "2 flipped real modes" should disappear once stats match.
- **Size:** small-moderate, ~40–80 lines. Verify with the pending pad-12 experiment
  (before/after node-count comparison).

### Fix 3 — robustness (separate defects, needed for the gate tolerance to be honest)
- **Transverse CCT conditioning:** ζ_T is solver-noise-dominated even at sane extents
  (pad 684 moves tile22 by 14% — B). Rank-revealing treatment of the indefinite CCT
  (absolute-cutoff pseudoinverse or Bunch-Kaufman with null-space handling) in
  `isdf/core.py`, and/or transverse centroid-count guidance. Without this, Σ^B remains
  hypersensitive to the residual 1e-7 input noise and no tight cross-P tolerance is
  achievable for bispinor. Medium-size, its own branch.
- **GN-PPM near-pole handling:** `ppm_invalid_mode` wiring (known parsed-but-unread knob)
  + a documented statement that Σ_C(E_dft) on bands with |Im Σ_C| above a threshold is
  ill-posed and excluded from invariance guarantees. Small.
- Not defects to fix here but to keep: A's verified zero-pad-row invariant (cheap regression
  assert), C's donation-safe `cusolvermp_batched_test` replacement, C's legacy-LU-fallback
  planner sizing + NCCL crash ticket.

## 3. Multi-device eqp-invariance gate design

Constraints: must not require 16 GPUs (portable per sandbox policy); must separate the
deterministic pad defect from the irreducible ~1e-7 reduction-order floor; must not gate on
ill-posed quantities (on-pole Σ_C, near-null ζ_T components).

**Fixture:** MoS2 3×3 (existing `Z_memplanner_validation`-style inputs), two legs:
charge GN-PPM (A-type, but with evaluation energies / band set chosen OFF poles — assert
|Im Σ_C| < 100 eV on all gated bands, or restrict the eqp comparison to such bands) and
bispinor COHSEX+Breit (B-type, gates Σ^B/ζ_T). Deliberately choose n_rmu NOT divisible by 4
(e.g. keep 1204 charge / 668 transverse) so pads differ between the two legs of each test.

**Tier 1 — pad-extent invariance at fixed P (the sharp test, runs on 1 GPU):**
promote B's `LORRAX_EXTRA_MU_PAD` hook to a supported test-only knob (config key or env,
plumbed in `common/meta.py` / `runtime/padding.py` / `gw/gw_init.py` as B prototyped).
Run P=1 base vs P=1 with extra pad chosen to hit the P=4 and P=16 extents. After Fixes 1+2,
ζ (all channels), V_q tiles, PPM node counts/invalid census, Σ_X, Σ^B, Σ_C, eqp0/eqp1 must be
**bit-identical** (tol 0; everything downstream of a shape-invariant solve is deterministic
at fixed P). This is the regression gate for the actual defect and costs one cheap run pair.

**Tier 2 — cross-P invariance, 1 GPU vs 4 GPU (practical proxy for 4-vs-16):**
same fixture, P=1 vs P=4 (different pad extents AND different reduction topology). Expected
residual after fixes = the ~1e-7 rel ζ floor (A §Amplification, B §4) through
well-conditioned consumers. Tolerances, each justified by measurement:
- ζ_C, V_q_CC: frob-rel ≤ 1e-6 (measured floor 2–9e-7).
- Σ_X / Σ_SX / Σ_COH / V_H diag: ≤ 1e-5 eV (measured ≤1e-6 eV across P today).
- PPM node counts + logical invalid-mode count: exactly equal (integers; Fix 2 acceptance).
- eqp0/eqp1 on gated (off-pole) bands: ≤ 1 meV charge. Bispinor eqp/Σ^B tolerance is set
  AFTER Fix 3 by measuring the fixed-shape solver-noise floor (today ~14% tile wobble makes
  any tight bound dishonest; gate provisionally on ζ_T pad-invariance from Tier 1 plus a
  documented loose Σ^B bound, tightened when the rank-revealing solve lands).
- 4-vs-16 GPU is run once manually post-fix for confirmation and thereafter only ad hoc;
  the routine CI gate is Tier 1 (1 GPU) + Tier 2 (≤4 GPU).

**Non-gates:** on-pole Σ_C bands (report |Im Σ_C| census instead), ζ_T near-null components
(gauge/noise — gate physical Σ^B and eqp, per the bispinor-tt-noncovariance lesson).

## AS-FIXED (2026-07-08, session D — fixes on `agent/memplanner-cleanup`)

### §6 charge confirmation experiment — RESULT (split verdict)

Ran `A_charge` at P=4 with `LORRAX_EXTRA_MU_PAD=12` (1204→1216 = the P=16 extent;
run dir `A_charge/padtest_4g_pad12`, knob-only pre-fix code, manifest_padtest.yaml):

| quantity | 4g base | 4g+pad12 | 16g | verdict |
|---|---|---|---|---|
| n_total_modes | 16·1204² | **16·1216² (=16g)** | 16·1216² | census inflation **pad-deterministic — CONFIRMED** |
| GN invalid | 255,980 | **720,620 = base + 464,640 pad + 0** | 720,622 (= +2 real flips) | pad share exact; the 2 real flips are NOT pad-driven |
| minimax nodes (b_slab / single) | 15 / 15 | **15 / 15** | 13 / 14 | node-count change **NOT pad-driven** |
| max\|Re ΔΣ_C\| vs base | — | **5.14 eV** | 5.67 eV | eV-scale Σ_C shift **IS pad-driven** (Σ_X = 0 to 1e-6 eV) |
| max\|Re ΔΣ_C\| vs 16g | 5.67 eV | 1.12 eV | — | pad extent explains most, not all, of 4g↔16g |

Refinement of §Manifestation 2: under the production `ppm_invalid_mode='zero'`,
pad modes never entered `B_mask` (they are `valid=False`), so the masked-Ω window
stats were already pad-free — the §4-observed node-count change is cross-P
reduction-order noise flipping near-threshold (divergent-Ω) modes in/out of the
valid set, whose max feeds the adaptive window intervals.  The pad-deterministic
eV carrier in the charge chain is instead the **padded-extent solves** (the per-q
W-solve LU, and at multi-GPU the distributed ζ-Cholesky extent), amplified by
on-pole Σ_C(E_dft) evaluation.  The census/`unfulfilled` inflation is a separate,
exactly-pad-deterministic diagnostic defect.

### Fixes landed (commit series on `agent/memplanner-cleanup`)

1. **Knob**: `LORRAX_EXTRA_MU_PAD` promoted to a permanent env-only test knob —
   `runtime/padding.py:padded_mu_extent` (single source of truth), consumed by
   `Meta.from_system`, `gw_init` (transverse refresh), `wfn_transforms`
   (ψ centroid load), `v_q_g_flat` (V-tile pad), `v_q_bispinor` (reader pad).
   The two V_q-side local `_pad` helpers previously computed their own round-up
   — the knob exposed that as a shape crash (`dot_general 1204 vs 1216`), now
   unified.
2. **Fix 1 (bispinor ζ_T)**: `isdf/core.py:solve_zeta` — both the per-q
   `jnp.linalg.solve` path and the cuSolverMp getrf/getrs path μ-slice the
   indefinite system to `n_rmu_logical` before the solve and zero-fill ζ pad
   rows after; ridge uses the LOGICAL trace (`fit_zeta_to_h5` computes it on
   the logical block) and logical n.  cuSolverMp falls back to the per-q path
   when the logical extent isn't per-axis mesh-divisible.  The false
   "logical block bit-identical" docstring guarantee corrected
   (Cholesky ≤1e-7 measured; LU: fails O(1)).
3. **Fix 1b (charge logical-extent solves)**: `w_isdf._get_w_solve_fn` per-q
   Dyson LU sliced to logical (+ zero-filled W pad rows);
   `factor_c_q` single-device dense Cholesky factorises at logical extent and
   re-embeds (multi-device distributed factorisations keep the padded extent —
   block-cyclic layouts require mesh divisibility; measured ≤1e-7 rel).
4. **Fix 2 (census/stats)**: `ppm_sigma._prepare_sigma_state` takes a logical-μ
   mask — `n_total_modes`, `n_invalid`, and `B_mask` (hence the masked-Ω window
   stats and the `2ry` contraction mask) count logical modes only;
   `fit_ppm`/`fit_gn_ppm_from_wc_pair` compute `unfulfilled` over logical modes.
5. **Gates**: Tier 1 in-suite `tests/test_mu_pad_invariance.py` (fixed-P pad
   flip, bit-identical, 1 GPU; bispinor pins the historically catastrophic 672
   extent).  Tier 2 script `tests/multi_device/eqp_invariance_cross_p.py`
   (P=1 vs P=4, §gate tolerances).

### Post-fix validation (head config reruns, `postfix_{4g,16g}` in both cells)

**No leak into the logical path:** both `postfix_4g` runs are **bit-identical**
to their pre-fix `head_4g` outputs modulo the timestamp header (at P=4 neither
system has a μ pad — every fix is an exact no-op there).  Full pytest suite:
**247 passed / 0 failed** (all four golden e2e gates + IBZ gate green); final suite including the two new Tier-1 gates: **249 passed / 0 failed**.

**B_bispinor (the eV bug): FIXED.**

| quantity | pre-fix 4g↔16g | post-fix 4g↔16g |
|---|---|---|
| Σ^B tile(2,2) trace at 16g | **−117.914143 eV** | **−0.152608 eV** (= 4g to all printed digits; every tile matches) |
| eqp0 / eqp1 max\|Δ\| | 2.535 eV (270/270 bands > 1 meV) | **1.45e-7 eV** (0 bands > 1 meV) |
| sigma_diag (Σ_SX/Σ_COH/V_H) | identical | identical |

**A_charge (GN-PPM): pad defects fixed; on-pole/noise residual remains and is
now cleanly separated.**

| quantity | pre-fix 4g↔16g | post-fix 4g↔16g |
|---|---|---|
| n_total_modes | 16·1204² vs 16·1216² | **16·1204² at both** (census logical) |
| GN invalid | 255,980 vs 720,622 | 255,980 vs **255,982** (the 2 cross-P noise flips persist; pad share gone) |
| minimax nodes (b_slab/single) | 15/15 vs 13/14 | 15/15 vs 13/14 (**unchanged — not a pad defect**, see §6 result) |
| Σ_X diag | identical (1e-6 eV) | identical |
| eqp0 max\|Δ\|, \|Im Σ_C\| < 100 eV (26 bands) | 0.27 eV | 0.27 eV |
| eqp0 max\|Δ\|, on-pole (1254 bands) | 5.64 eV (median 74 meV) | 5.64 eV (median 61 meV) |
| 16g pre-fix vs post-fix | — | ≤ 4.8e-5 eV (the W-LU extent change itself was a 1e-5-eV-scale carrier at 16g) |

Honest accounting of the surviving A_charge cross-P spread — neither part is a
pad-extent defect and neither was forced here:

1. **On-pole ill-posedness** — this run evaluates Σ_C(E_dft) essentially ON
   GN-PPM poles (only 26/1280 bands below \|Im Σ_C\| = 100 eV, none below
   1 eV).  The irreducible cross-P inputs noise (~1e-7 rel: reduction
   topology + the distributed ζ-Cholesky, which must factorise at the padded
   extent) is amplified by up to ~1e6.  Fix-3 territory (`ppm_invalid_mode`
   robustness + documented exclusion of on-pole bands from invariance
   guarantees).
2. **Node-count flips** — the adaptive minimax windows read the masked-Ω
   max, which is dominated by the most nearly-invalid valid mode (Ω diverges
   at the validity threshold), so a ~1e-7 W wobble flips 2 modes and
   discretely changes the node count (15→13/14).  A robustness fix (capped /
   robustified window stats or validity hysteresis) would change golden
   numbers and is deliberately NOT part of this series.

**Gate status:** Tier 1 (fixed-P pad flip, 1 GPU) added to the suite as
`tests/test_mu_pad_invariance.py`:

- **bispinor leg: full BIT-IDENTITY holds** (pad 4 pins the historically
  catastrophic 672 extent) — the static COHSEX+Σ^B chain is now exactly
  pad-extent-invariant at fixed P.
- **gnppm leg:** census / node counts / invalid count / unfulfilled and Σ_X
  are **exactly equal**; Σ_C/eqp are gated at ≤ 2e-4 eV.  §3's "bit-identical"
  aspiration is NOT achievable for the dynamic chain: in-memory arrays keep
  the padded μ extent, so XLA fusion tiling regroups partial sums per extent
  — measured 1–2 ULP on ζ (1.7e-16 rel) and V (2.3e-16 rel) at P=1 with ψ
  bit-identical — and the near-singular GN-PPM fit ratio (denominators at
  the 1e-14 validity threshold) amplifies those ULPs to a measured 6.3e-5 eV
  on Σ_C (2.2e-5 eV even off-pole).  Removing that would mean never padding
  μ in any kernel (an architecture change), or the Fix-3 robustness work.
  Probe artifacts: `postfix_probe/` in this dir (`probe_diff.py` + the
  pad0/pad12 P=1 run logs; run dirs at `sources/lorrax_D/.padprobe/`).

Tier 2 (P=1 vs P=4) added as `tests/multi_device/` (`run_tier2.sh` driver +
`eqp_invariance_cross_p.py compare`; needs 4 GPUs, not in the suite).  First
post-fix measurement (both fixtures **PASS** under calibrated, documented
tolerances):

| quantity | measured P=1↔P=4 | gate |
|---|---|---|
| gnppm ζ_C | 2.8e-6 frob-rel | ≤ 1e-5 (dense-vs-distributed solver contrast) |
| gnppm Σ_X | 1e-6 eV | ≤ 1e-5 eV |
| gnppm minimax node counts | **exactly equal** | exact (census-fix acceptance ✓) |
| gnppm n_total_modes | exactly equal | exact |
| gnppm invalid split | ±8 flips / 3.7M | reported, not gated (Fix-3: near-threshold flips) |
| gnppm off-pole eqp | 69 meV | ≤ 0.1 eV PROVISIONAL — each flipped mode discretely adds/drops its pole term under `ppm_invalid_mode='zero'`; the §gate 1 meV target needs Fix-3, not pad hygiene |
| gnppm on-pole eqp | 0.43 eV | reported only |
| bispinor ζ_T (μ1/μ2/μ3) | **1.8–2.1e-7** | ≤ 1e-3 provisional (Fix-1 acceptance — c.f. 0.9–5.5e2 pre-fix) |
| bispinor ζ_C | 0.25 frob | reported only — this 32-band fixture's CCT is rank-deficient (640 μ > pair rank), ζ_C carries solver-dependent null components; physical Σ/eqp are the gate |
| bispinor Σ_SX (incl. Σ^B) | 1.85e-4 eV | ≤ 1e-3 eV provisional |
| bispinor eqp0 | 34 meV | ≤ 50 meV provisional (production 3×3 cell measures 1.45e-7 eV at 4g↔16g — the loose bound is a property of the tiny fixture) |

## Artifacts index
- Repro runs: `runs/MoS2/Z_memplanner_validation_2026-07-06/B_bispinor/padtest_4g_{base,pad4,pad4_ridgefix,pad16}` (+ `manifest_padtest.yaml`), `head_16g_luoff`.
- Comparators: this dir — `agent_A_diff_outputs.py`, `agent_B_tmp/*`, `zeta_compare.py`, `potrs_4x4_check.py`.
- `sources/lorrax_D` clean at bb95bc3 (all agents reverted instrumentation).
