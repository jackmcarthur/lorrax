# Agent B — μ-padding audit (device-count invariance bug, 2026-07-08)

**Verdict up front:** the bispinor 4g↔16g eqp divergence (2.5 eV) is a **proven
transverse μ-pad defect**, reproduced at FIXED P=4 by forcing the 16-GPU pad extent
(668→672). The charge GN-PPM divergence is **not** a pad leak: Σ_X is identical to
1e-6 eV across P despite 0-vs-12 pad rows; its Σ_C spread lives on GN-PPM pole bands
(|Im Σ_C| = 2 eV … 1e7 eV) at relative size ~1e-5–1e-7 of the pole magnitude —
shape/reduction roundoff amplified by near-pole conditioning.

All temporary instrumentation in `sources/lorrax_D` was reverted (`git checkout --`).

---

## 1. The decisive experiment (fixed P, pad-only flip)

MoS2 3×3 bispinor COHSEX (640 charge + 668 transverse centroids), all runs on the
same code (`agent/memplanner-cleanup`), same inputs, same WFN/centroids, **4 GPUs**:

| run | pad extent (T) | Σ^B tile(2,2) trace | Bare Σ_X diag k=0, band 2 | dirs |
|---|---|---|---|---|
| `padtest_4g_base` (control) | 668 (no pad) | **−0.152608 eV** | −30.5881 eV | `runs/MoS2/Z_memplanner_validation_2026-07-06/B_bispinor/padtest_4g_base` |
| `padtest_4g_pad4` (`LORRAX_EXTRA_MU_PAD=4`) | 672 (= P=16 extent) | **−117.914395 eV** | −32.8091 eV | `.../padtest_4g_pad4` |
| `padtest_4g_pad16` | 684 | −0.173712 eV | −30.5884 eV | `.../padtest_4g_pad16` |
| reference `head_4g` (P=4) | 668 | −0.152608 eV | −30.5881 eV | `.../head_4g` |
| reference `head_16g` (P=16) | 672 | **−117.914143 eV** | −32.8091 eV | `.../head_16g` |

- `pad4` at P=4 reproduces the P=16 numbers to 5 digits (residual 2.5e-6 rel =
  reduction-order roundoff). eqp0 diff base↔pad4: **max 2.535 eV, median 0.069 eV,
  270/270 bands > 1 meV — the identical signature of the real 4g↔16g diff.**
- The on-disk `v_q_bispinor.h5` diff between base↔pad4 is byte-level the same
  signature as head_4g↔head_16g: `V_qmunu_TT_22` maxdiff 3.79e12 (rel 1.35e5),
  all TT tiles rel ≥ 1; `V_qmunu_CC` rel 3e-8 (roundoff).
- Divergence enters at the **transverse ζ solve**: `zeta_q_mu2.h5` rel diff 8.6e2
  (mu1: 0.99, mu3: 1.0) while charge `zeta_q.h5` rel 5.5e-8.
- The eqp carrier is `results.sig_x` (bare DHF+Breit exchange) inside
  `eqp0 = kin_ion + V_H + Σ_x + Σ_c` (`gw/eqp_bgw.py:compute_eqp_diag`); the printed
  sigma_diag columns (sigSX/sigCOH/VH) and kin_ion.h5 are **identical** across P,
  which is why sigma_diag looked clean while eqp moved.

**Charge chain end-to-end control:** in base↔pad4 (charge got 4 extra pad rows too),
sigSX/sigCOH/sigTOT/VH are identical to 1e-6 eV at all 9 k — the charge pipeline
(ζ Cholesky → V_q → χ0 → W solve → COHSEX Σ) is pad-neutral end-to-end.

## 2. Mechanism (narrowed, one candidate exonerated)

The transverse channels factor via the identity-padded indefinite LU path:
`isdf/core.py:factor_c_q` (vertex_mu_L≠0 → passthrough of
`_identity_pad_block_diagonal(C_q)`), then `solve_zeta`'s `_ridge_indef_solve`
(`isdf/core.py:~1283`): `ridge = 1e-12·|tr(L)|/n`, `jnp.linalg.solve(L+ridge·I, Z)`.

- **Exonerated:** ridge-trace pad contamination (`tr(L_padded)=tr_log+n_pad`,
  `n=padded`). A run with the pad contribution subtracted
  (`padtest_4g_pad4_ridgefix`) still gives −117.914395 — not the mechanism.
- **Remaining mechanism:** the batched pivoted LU (`jnp.linalg.solve`) at padded
  extent does **not** honor the "logical block bit-identical" guarantee claimed in
  `_identity_pad_block_diagonal`'s docstring (isdf/core.py:781-791). Empirically each
  padded extent (668 / 672 / 684) yields a **different, per-extent-deterministic**
  ζ_T — which is why P=4-pad672 == P=16 exactly. The transverse CCT (Hermitian
  indefinite, γ̃² channel worst) is near-singular at the 1e-12-ridge scale, so
  shape-dependent LU roundoff (cuSOLVER blocking differs with n) is amplified O(1)
  in near-null modes; at n=672 it hits a catastrophic resonance (tile22 ×770).
  Note ζ_T's null-space junk is NOT annihilated in physical Σ^B (tile22 moved
  117 eV; even the "sane" 684 pad moves it 14%) — the transverse fit is
  solver-noise-dominated at these centroid counts, refining the earlier
  "TT Σ^B well-conditioned" assessment.

**Fix directions** (not applied — audit only):
1. Device-invariance: run the per-q transverse LU on the **logical** extent — L is
   replicated (`in_specs=P(None,None)`) and Z is column-sharded, so slicing μ to
   `n_rmu_logical` before `jnp.linalg.solve` and zero-filling ζ pad rows after is
   local and cheap. This makes ζ_T bit-independent of the pad extent (hence of P).
2. Robustness (separate issue): the near-null modes of the indefinite transverse
   CCT need a rank-revealing treatment (fixed-absolute-cutoff pseudoinverse /
   Bunch-Kaufman / channel-specific centroid count) — otherwise ζ_T remains
   hypersensitive to ANY perturbation even at fixed shape.

## 3. Per-consumer pad audit (code + measurement)

n_rmu_padded = round_up(n_rmu, P) (`common/meta.py:133`; transverse copy at
`gw/gw_init.py:207-208`; ψ-side `runtime/padding.py:round_up_to_mesh_product`).

| consumer | location | pad handling | verdict |
|---|---|---|---|
| ψ centroid load μ-pad | `common/wfn_transforms.py:1852-1858` (`_reshard_all`) | `jnp.pad` zeros | **neutral** (proven: Σ_X charge identical across P) |
| band pad (b_id_4, past-mnband, user-band) | `wfn_transforms.py:1808-1878` | explicit zero blocks | **neutral** |
| C_q / CCT bilinears | `gw/isdf_fitting.py:391-397` | bilinear in zero-pad ψ → exact zero pad rows | **neutral** |
| Cholesky identity-pad (charge) | `isdf/core.py:765-820, factor_c_q` | block-diag [C,I], √1=1 | **neutral to ≤1e-7 rel** (measured ζ_C 5.5e-8) |
| **LU identity-pad (transverse)** | `isdf/core.py:factor_c_q` passthrough + `_ridge_indef_solve` | claimed block-exact | **VIOLATED — the bug.** ζ_T changes wholesale with pad extent |
| ζ gflat accumulate + G-slot sentinel mask | `isdf_fitting.py:944-975` | pad G-slots masked to zero pre-write | neutral (P-independent, ngkmax fixed) |
| disk round-trips (ζ, V tiles) | SlabIO `valid_shape` (writers); readers re-pad zeros (`zeta_reader.py:167-178`, `v_q_bispinor.py:497-545`) | logical on disk, zero pad on read | **neutral** |
| V_q tiles + g0 head | `v_q_g_flat.py` | ζ̃ pad rows zero → V pad rows/cols zero | **neutral** (Σ_X identical across P) |
| IBZ unfold `take_along_axis` | `symmetry_maps.py:285-310, 392-470` | fwd_perm padded with identity block; logical perm values < n_logical never index pads; pad L rows zero → phase 1 | **neutral by construction** (not exercised in these runs — orbit closure failed → full-BZ) |
| W solve (μ pad) | `w_isdf.py:206-268` | A=I−Vχ pad block = I exactly, RHS pad = 0 | **neutral** (measured: sigSX/sigCOH identical under pad flip) |
| W solve (q pad, nk=9→16 at P=16) | `w_isdf.py:240-262` | zero-padded q rows solved as I·W=0, sliced off | **neutral** |
| GN-PPM fit | `minimax_screening.py:361-411` | pad Wc=0 → denom 0 → `safe=False` → B_pad = −0.5·0·ω = 0 exactly | **neutral** (pad entries only inflate the diagnostic `unfulfilled`/`n_invalid` counters) |
| Σ_X/Σ^B projection | `sigma_x_bispinor.py:161-181`, cohsex kernels | V zero-padded to ψ's padded extent; bilinear in zero-pad ψ | **neutral** given clean inputs (the corrupt V_TT tile values are logical, upstream) |
| eqp assembly | `gw_output.py:272-340`, `eqp_bgw.py` | logical band extents only | neutral (it just exposes the Σ_x corruption) |

## 4. Charge system (A_charge, GN-PPM) — what its 3.8–5.7 eV spread is

- `|d Σ_X| = 0` (≤1e-6 eV print precision) at ALL 1261 (k,n) across 4g↔16g despite
  the pad-row asymmetry (0 vs 12 rows) — no pad leak anywhere in ζ_C/V_q/Σ_X.
- `d Σ_C` up to 5.7 eV, but exclusively on bands with |Im Σ_C| = 2 eV … 1e7 eV
  (GN-PPM poles at E_dft for the deep/semicore bands; e.g. the 5.66 eV mover has
  |Im Σ_C| = 4.5e5 eV). Ratio d/|Im|: median 2.3e-7, max 5.4e-2. Only 2 of 1261
  entries have |Im| < 100 eV, and their d ≤ 5e-4 eV.
- Interpretation: P-dependent (and, per §2, shape/pad-dependent at 1e-7 rel) roundoff
  in ζ/W is amplified by on-pole denominators. Same class as the base↔pad4 charge
  ζ wobble (5.5e-8 rel) — a conditioning problem, not a pad-row leak. It will not be
  fixed by pad hygiene alone; on-pole Σ_C(E_dft) evaluation is intrinsically
  ill-posed there.

## 5. Artifacts

- Repro runs + manifest: `runs/MoS2/Z_memplanner_validation_2026-07-06/B_bispinor/{padtest_4g_base,padtest_4g_pad4,padtest_4g_pad4_ridgefix,padtest_4g_pad16,manifest_padtest.yaml}`
- Diff/parse scripts: `reports/device_invariance_2026-07-08/agent_B_tmp/{diff_runs.py,diff_vq_bisp.py,diff_kinion.py,perk.py,charge_ratio.py}`
- The temporary knobs used (now reverted): `LORRAX_EXTRA_MU_PAD` (meta.py /
  runtime/padding.py / gw_init.py), `LORRAX_RIDGE_TRACE_PAD_CORRECTION`
  (isdf/core.py), `LORRAX_COORD_PORT` (runtime/__init__.py — needed to avoid JAX
  coordinator collisions between concurrent agent steps in the shared allocation;
  see KNOWN_SANDBOX_ERRORS.md entry 2026-07-08).
