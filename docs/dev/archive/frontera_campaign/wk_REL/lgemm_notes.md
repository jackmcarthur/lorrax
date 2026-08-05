# wk_REL — L-GEMM f64-split projection relowering (lever 1) implementation log (2026-07-28)

Implements lever 1 of wk_REL/docs/RESHARD_OVERHEAD_MEMO.md (Sec. 4.4 exit (a) /
Sec. 7 ordered list, item 1): movement-only relowering of the projection
right-einsums so XLA:CPU can no longer promote the f64 channel operands to
c128 (the HLO-proven ~400 MB/channel `convert` copies + Eigen zgemm at 2×
required flops, 295 GF/s vs 1263 GF/s measured BLAS on the same contraction).

Tree: /work2/08271/jackmc/frontera/lorrax @ 0225b5f, WORKING TREE ONLY (not
committed; orchestrator merges).  NOTHING TRS-based (owner veto); collectives
untouched (payload dtype/shape/order/replica-groups byte-identical — the
FFI direct-MPI chain is the SEPARATE owner-gated lever L3).

## What changed (src/gw/ppm_tau_kernel.py; FINAL tree state after measurement)

Both projection bodies keep their mathematics and their collectives; ONLY the
lowering of the large right contraction was in scope:

- `_project_ri_local` (two-channel, crossing + default plan) — RELOWERED,
  KEPT: each channel's f64 × c128 einsum 'ksxty,ktyn->ksxn' is expressed as
  TWO f64 dgemms against Re ψ / Im ψ + one `lax.complex` recombine (4
  dgemms/τ total).  The owner-held channel algebra (independent σ_R / σ_I
  chains) is untouched — only the complex ψ operand's representation changes.
  Previously XLA promoted the f64 channel operand to c128 (~400 MB convert
  materialization per channel) and ran a full zgemm per channel = 2× required
  flops.
- `_project_x_local` (merged Laplace plan) — RELOWERING TRIED AND REFUTED,
  REVERTED to its single complex einsum (refutation recorded in its
  docstring; the tried form is preserved in
  wk_REL/docs/patches/lgemm_full_2026-07-28.patch).  This body has NO promotion pathology
  (genuine complex × complex at minimal flops); the 4-dgemm form regressed it
  because Eigen's f64 batched dot is per-flop SLOWER than its zgemm at these
  shapes (~172 vs 295 GF/s measured, decomposition below).
- Small left dots stay genuinely complex (measured 1.6e-3 of right flops —
  memo Sec. 2.2); stacked psum_scatter pair / merged single-chain collectives
  byte-identical; output dtypes unchanged.  Docstrings updated in both bodies
  + the factory with measured numbers and claim scopes.

Ungated (like AK.9 stacking and the axis-order swap — movement-only,
value-level identical, 1e-12 parity-gated; NOT bit-exact: the dgemm split
sums the same products in a different order than the promoted zgemm).

Envelope statement (design-envelope rule): the flop halving and the
promotion-copy removal are shape-independent (any n_atoms / N_μ / nb / nk /
P); GEMM share GROWS with nb (∝ nb·μ²/P vs comm ∝ nb·μ/√P — memo Sec. 5), so
the lever's relative value rises with the size campaign.  Mixed-dtype dots
promote on XLA:GPU as well → neutral-or-better on both backends (measured on
XLA:CPU only; claim scoped).

New test: tests/test_projection_lgemm.py (test_sanity_gates_jax.py 4-device
pattern) — compiled-HLO pins on BOTH projector modules: (1) no rank≥2
f64→c128 convert; (2) large right dots f64, only the small left dots c128;
(3) exactly 2 c128 reduce-scatters at the exact pre-relowering payload shapes
(stacked leading 2 two-channel / no leading 2 merged); (4) 1e-12 numpy parity.

## Gates

Gate order: unit gates first (lgemm_gate.sbatch), A/B only on PASS
(lgemm_ab.sbatch).

### GATE 1 — py_compile + HLO pin + P=4 production kernels (job 7878939): PASS

Cells: pycompile (compileall src + the new test), lgemm (the HLO-pin/parity
unit test, 4 emulated devices), p4 (wk_REL/probes/check_channel_hermiticity.py
--stage p4 — the production τ kernels through the real shard_map/psum_scatter
tails vs independent numpy references at 1e-12, now exercising the relowered
bodies; also re-proves S_R + i·S_I = X through the relowered kernels).

Results (job 7878939, all cells rc=0):
- pycompile: rc=0.
- lgemm unit test: PASS ×2 (two-channel + merged) — compiled modules carry
  NO rank≥2 f64→c128 convert, ≥4 f64 dgemms, only left dots c128, and
  exactly 2 c128 reduce-scatters at the exact pre-relowering payload shapes
  (stacked leading 2 / merged no-leading-2); numpy parity ≤1e-12.
- p4: PASS all six gates — G1 S_R+i·S_I=X 2.5e-16/3.1e-16 (Laplace/crossing),
  G2 two-channel vs numpy 7.3e-16/6.8e-16, G3 merged vs numpy 6.7e-16/7.1e-16
  — indistinguishable from the pre-relowering chmerge values (identical seeds).
- (CUDA plugin banner in the log = pre-existing venv noise, JAX_PLATFORMS=cpu.)

### GATE 2 — restart-gated A/B nb=128 + nb=256 at P=64 (job 7878942): DONE
### (both-bodies-relowered tree; verdict SPLIT — kept two-channel, refuted merged)

Passes (AC.4 harness, J.7 restart-gated, cache-cold, coll=mpi AS.7 env):
  k128L/k128Ls   nb=128 default flags (two-channel, XLA fft), prod + staged
  k128Lf/k128Lfs nb=128 + FFT_FFI+FUSED, prod + staged (the 43.2 s baseline row)
  l1Lmf/l1Lmfs   nb=256 MERGE=1 + FFT_FFI+FUSED (composed stack), prod + staged
Gates: parity 1e-12 dat + sigma_mnk.h5 tensors; per-τ project_rs rows vs
baselines (43.2 s @nb=128 FFI-fused two-channel; 47.6 s XLA+stacking; 24.6 s
@nb=256 merged XLA-staged — no prior FFI-staged-merged row exists, noted);
colltable unchanged-collectives; tau-kernel HLO L-GEMM pins at production
shapes (0 μ-sized c128 dots, 0 μ-sized f64→c128 converts, f64 dgemms present,
rs payload shapes byte-equal to the chmerge tables).

Memo projection at nb=128: project_rs 43.2 → 19.8 s IF Eigen dgemm reaches
the BLAS rate; if it materially lags, the FFI MKL GEMM handler (memo lever
1(b), the FFT-FFI playbook) is the named follow-up — NOT implemented here.

## Results — job 7878942 (6 passes, all rc=0, cache-cold, coll=mpi)

PARITY: sigma_diag/eqp0/eqp1 max|diff| = 0.000e+00 (text precision) ALL SIX
passes.  H5 TENSOR GATE (all datasets, tol 1e-12): PASS ×6 — worst dataset
sigma_c_kij_ev: 2.222e-14 eV (maxrel 3.0e-15) @nb=128 (fftffi-era baseline
diffs were 2.41-2.59e-14 — same ULP class, the relowering adds nothing
beyond reassociation); 1.335e-14 eV @nb=256.

COLLTABLE + HLO: collectives byte-unchanged at both shapes — tau-kernel
modules rs=2, all-gather=0, payloads exactly the chmerge tables
(c128[2,16,2,624,16]+c128[2,16,16,16] @nb=128 two-channel;
c128[2,16,2,312,32]+c128[2,16,32,32] crossing and
c128[16,2,312,32]+c128[16,32,32] merged @nb=256 — merged halving intact);
"NO collective carries a full (mu,mu) tile" at μ=4962 AND μ=2475.
L-GEMM pin at production shape: the promoted `c128[16,1248,128]
dot(%convert_bitcast_fusion...)` pair of the baseline dumps
(run_SIGMA_fft_fft1_c module_0551) is REPLACED by four
`f64[16,1248,128] dot(..., %real/%imag_bitcast_fusion)` dgemms; ZERO
μ-sized f64→c128 converts.  (The job's "c128 dots at mu-sized shapes"
grep printed 1 per module — that is the G-BUILD dot c128[16,1248,1248]
('ksxn,knty->ksxty', complex × complex by nature, present identically in
every baseline dump, outside lever scope.  Refined pin = zero c128 dots
at the (μ, nb)-shaped right-product class, which holds.)

sigma.exec (production passes):
  k128L  (XLA, default flags)  262.564  vs 272.040 baseline / 278-280
                               neutral band → −3.5% vs baseline, −6% vs
                               band (best XLA nb=128 prod on this tree
                               WITHOUT the merge; k128m with merge: 262.4)
  k128Lf (FFI+fused)            66.470  vs 71.906 → **−5.44 s (−7.6%)**
  l1Lmf  (merge+FFI, BOTH bodies split — the REFUTED tree state)
                                39.694  vs 35.591 (run_CHMERGE_l1mf) →
                                **+4.1 s REGRESSION** (see verdict)

Staged per-τ project_rs (the lever's row):
  k128Ls  (XLA)      45.500 s/176τ vs 47.6 r2-XLA ref (−4.4%; same-run FFT
          rows read +25% high — the ±25% staged-noise band applies)
  k128Lfs (FFI)      **38.688 s/176τ vs 43.2 BASELINE → −10.5%**
          (245.7 → 219.8 ms/τ); omega_project 0.389 s
  l1Lmfs  (merge+FFI, refuted tree) 25.509 s/173τ vs 24.60 l1ms-XLA ref
          (+3.7% — and the FFI layout should have HELPED by ~10% per the
          nb=128 fftffi precedent → real merged-body regression)

## VERDICT + gap decomposition (the honest read)

The de-promotion WORKED (HLO-proven, above) and the two-channel body wins
everywhere measured.  But the memo's 19.8 s projection assumed dgemm at the
BLAS rate; measured instead, per the Sec. 5 decomposition (project_rs
219.8 ms/τ − collectives 49.9 − casts/left ~11.6 − skew ~9.6):

  right-GEMM ≈ 148.7 ms/τ for 2.55e10 real flops ≈ **172 GF/s — Eigen's
  f64 batched dot is per-flop SLOWER than its own zgemm (295 GF/s)** and
  7.3× below the 1263 GF/s BLAS roofline.

Consequences, both measured in this job:
1. Two-channel body: flops halve (2 promoted zgemms → 4 dgemms), rate drops
   295→172 → net −25.9 ms/τ = −10.5% project_rs.  KEPT.
2. Merged body: flops were already minimal (1 zgemm); the split trades rate
   295→172 at EQUAL flops → +60 ms/τ on Laplace windows; composed nb=256
   regressed (project_rs +0.9 s staged, sigma.exec +4.1 s prod).  REVERTED
   (job 3 re-verifies the reverted tree at nb=256; the nb=128 verdicts
   stand — merge was OFF in those passes so the merged kernel was neither
   built nor dispatched).

## Named, not done

- **FFI MKL GEMM handler for the projection (memo Sec. 4.4 exit (b), the
  FFT-FFI playbook: MklThreadScope, input_output_aliases, announce-or-
  refuse, two-plan)** — the lever that reaches the BLAS rate.  Required to
  realize the remaining ~19 s at nb=128 (dgemm 172 → ~1263 GF/s ⇒
  project_rs → ~20-22 s) and it would serve BOTH bodies (zgemm through MKL
  for the merged plan, dgemm pairs for the two-channel).  Explicitly NOT
  implemented here per instruction: XLA's own dgemm "still lags BLAS
  materially" — confirmed by measurement.
- The memo's L0/L3/L1 collective levers (common primitive, FFI direct-MPI
  chain, strip overlap): separate owner-gated workstream, untouched here.

### GATE 3 — re-gate + focused nb=256 re-run of the reverted (FINAL) tree: PASS

- job 7878976 (gate rerun on the final tree): pycompile=0, lgemm unit
  test PASS ×2 (merged module now pins the RESTORED complex right dot +
  still-forbidden promotions), stage-p4 PASS (all six 1e-16 gates,
  values identical to the chmerge round — same seeds).
- job 7878977 (lgemm_ab2, composed stack merge+FFI+fused at nb=256,
  both passes rc=0):
  - l1Lmf2 prod sigma.exec **35.497 s** vs run_CHMERGE_l1mf 35.591 (and
    vs the refuted both-split tree's 39.694) — the regression is GONE;
    band restored, ~neutral prod (the crossing-window win is inside the
    d2h-absorbed device wait at prod row semantics).
  - l1Lmfs2 staged project_rs **20.612 s/173τ** vs 24.60 l1ms XLA-staged
    ref (−16%) and vs 25.51 on the refuted tree — first FFI-staged-merged
    row at this shape; consistent with FFI layout benefit + the kept
    two-channel win on the 96 crossing windows (model −2.4 s) + merged
    windows restored (+4.9 s recovered vs refuted tree).
  - Parity exact-0 text ×2; h5 tensor gate PASS, worst dataset
    sigma_c_kij_ev 5.340e-15 eV (maxrel 6.3e-16) — BETTER than the
    chmerge-era 3.6e-15→ same class; merge announce printed.
  - HLO (rank-0, cache-cold): crossing module = 4× f64[16,624,256]
    dots on %real/%imag fusions + 2 c128[16,256,32] left dots; merged
    module = c128[16,624,256] right dot RESTORED + c128[16,256,32]
    left; ZERO μ-sized f64→c128 converts in both; rs=2/ag=0 both
    modules, payloads byte-equal to the chmerge tables.

## Final scoreboard (the tree as it stands)

| row | before | after | Δ |
|---|---|---|---|
| project_rs staged, nb=128 FFI-fused (BASELINE) | 43.2 | **38.69** | **−10.5%** |
| project_rs staged, nb=128 XLA | 47.6 | 45.50 | −4.4% (±25% band) |
| project_rs staged, nb=256 merged+FFI composed | 24.60 (XLA ref) | **20.61** | −16% |
| sigma.exec prod, nb=128 FFI-fused | 71.906 | **66.470** | −7.6% |
| sigma.exec prod, nb=128 XLA | 272.040 / 278-280 band | 262.564 | −3.5% / −6% |
| sigma.exec prod, nb=256 composed | 35.591 | 35.497 | neutral |

Claim scope: MoS2 4×4 deck, μ_pad=4992 (nb=128) / 2496 (nb=256), 8×8 mesh
on 32 nodes × 2 ranks, coll=mpi (AS.7), XLA:CPU/Eigen, cache-cold, jobs
7878939/7878942/7878976/7878977.  Full pytest sweep NOT re-run (change is
scoped to ppm_tau_kernel + a new self-contained test; the suite's
pre-existing ledgered failures are independent — chmerge attribution
7878915).

## Files touched (worktree only, NOT committed)

- src/gw/ppm_tau_kernel.py — `_project_ri_local` f64-split relowering
  (KEPT) + docstrings with measured numbers; `_project_x_local` docstring
  refutation record (body reverted to single complex einsum); factory
  docstring lever bullet.
- tests/test_projection_lgemm.py — NEW: compiled-HLO pin + parity unit
  test (4-device pattern), both projector modules.
- wk_REL: lgemm_gate.sbatch, lgemm_ab.sbatch, lgemm_ab2.sbatch (+ inner
  scripts they generate), lgemm_full_2026-07-28.patch (the refuted
  both-bodies-split form), lgemm_notes.md (this file).
- Run dirs: mos2_4x4_test/run_LGEMM_{k128L,k128Ls,k128Lf,k128Lfs,l1Lmf,
  l1Lmfs,l1Lmf2,l1Lmfs2} (one run per dir, J.7 restart-gated).
