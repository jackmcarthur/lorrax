# wk_REL — Laplace channel merge implementation log (2026-07-28)

Owner ruling implemented (OWNER_DECISIONS.md, RULINGS 2026-07-28 evening):
Laplace channel merge CONDITIONALLY APPROVED — (a) falsification gates incl.
the P=4 production-kernel S_R+i·S_I=X check at 1e-12, (b) tolerance-parity
A/B at nb=128+256, (c) the bilinearity math incorporated into repo docs.
HARD VETO respected: nothing TRS-involving anywhere (memo checks T1-T3 and
the half-BZ route are NOT implemented, NOT tested, NOT documented as
options); crossing ("core") windows keep their two-channel structure — they
dispatch the pre-existing kernel, whose math body is untouched.

Tree: /work2/08271/jackmc/frontera/lorrax @ 5918cf6, WORKING TREE ONLY (not
committed; orchestrator merges).  Derivation source:
wk_REL/docs/DERIVATION_channel_hermiticity.md (verdicts 1-6 licensed the merge by
bilinearity + consumer analysis; verdict 5 excludes crossing; verdict 6 kills
the Toeplitz recovery route).

## Design (two-plan doctrine)

Default plan (flag off): current code byte-untouched — two real-channel GEMM
chains per (τ, window), stacked psum_scatter pair, (σ_re, σ_im) D2H, host
consumer recombines.

Gated plan `LORRAX_SIGMA_LAPLACE_MERGE=1`: window-level dispatch in
`ppm_sigma._integrate_tau_windows_for_branch`:
- Laplace windows (project="full", project_code=0): a sibling production
  kernel (`_get_sigma_tau_kernel(..., merged_x=True)`) whose projection tail
  is ONE complex chain X = ψ†σψ (`ppm_tau_kernel._project_x_local`) — one
  GEMM pair per (τ, k-batch), ONE psum_scatter per mesh axis at HALF the
  stacked-pair payload (one c128 tile vs stacked re+im, both complex), half
  the per-τ D2H bytes.  The accumulator consumes X directly
  (`add_tau(X, None, ...)`; `_project_tau_onto_omega_np` computes coeff·X).
- Crossing windows (project="imag"): ALWAYS the existing two-channel kernel.
  A merged tile reaching the crossing consumer raises (dispatch-bug guard).
Both kernels are AOT-prewarmed by `precompile_sigma` on a merged-plan run.
Announce: one driver line per Σ stage ("Σc channel plan: LAPLACE MERGE ...").
Axis-order swap and contraction order inherited from the two-channel body
(μ_Y first; large partial reduce-scatters over node-local 'y' groups).
Scaling honesty: no new N_μ²-global object on any rank; the merged plan only
removes work/bytes (flat in n_atoms, N_μ, nk, nb, P; backend-agnostic —
composes with LORRAX_FFT_FFI(_FUSED), gated below).

Deck-level ε_H observability (memo Stage B item iii): env-gated
`LORRAX_PPM_HERM_DIAG=1` in `ppm_sigma.fit_ppm` prints
max_q|B_q−B_q†|/max|B| and the Ω_q symmetry residual over ALL q (production
gate was q=0 only, rtol 1e-6).  Diagnostic, not a merge gate — the merge
needs no hermiticity.

## Files touched (worktree only, NOT committed)

- src/gw/ppm_tau_kernel.py   — `_laplace_merge_enabled`, `_project_x_local`,
  `_make_project_ri_reduce_scatter(merged_x=)`, kernel factories keyed on
  merged_x, precompile of both plans; channel-structure docstrings rewritten
  (what each path computes, why the merge is legal, why crossing is
  excluded); stale OWNER-HELD notes replaced by the ruling.
- src/gw/ppm_accumulators.py — merged-X consumer in
  `_project_tau_onto_omega_np` (sigma_im=None ⇒ coeff·X; raises on
  project_code=1), single-tile async-D2H in `_TauAccumulator.add_tau`.
- src/gw/ppm_sigma.py        — per-window kernel dispatch, driver announce,
  `fit_ppm` ε_H diag, docstring updates.
- manual/07_frequency_integration/7.5_sigma_assembly.md — new display-LaTeX
  subsection "Channel structure of the projection" (X = ψ†σψ = S_R+i·S_I by
  bilinearity; consumer identity c·(S_R+i·S_I)=c·X; Toeplitz non-uniqueness;
  crossing counterexample), citing the derivation memo; the old
  "projection does not commute with taking real parts" parenthetical
  (wrong as stated — bilinearity is exactly the commutation) replaced.
- wk_REL/probes/check_channel_hermiticity.py — the memo §4 falsification script
  (Stage A + P=4 gate; TRS checks absent per veto).
- wk_REL/harness/chmerge_gate.sbatch, wk_REL/harness/chmerge_ab.sbatch — gate harnesses.

## GATE 1 — falsification protocol (job 7878863, dev 1 node): PASS

Run FIRST, per owner gate order, before any A/B.

Stage A (P=1, 1×1 mesh, σ^τ assembled from the real building blocks —
build_G_tau + make_flat_k_{i,f}fftn at G_FFT7D_SPEC/V_FFT5D_SPEC + the
_build_W_t_q phase build; cross-checked against an independent np.fft
reference at 6.0e-16/8.0e-16):
- L1 Laplace σ_R symmetry / σ_I antisymmetry: 6.4e-17 / 1.3e-16  PASS
- L2 bilinearity (S_R+i·S_I)−X: 4.0e-16; X hermiticity 3.4e-16   PASS
- C1 crossing σ_R symmetry residual: 3.473e-01 — O(1) as §2.2 demands
  (falsification check: a small value would have REFUTED the derivation
  and stopped the workstream)                                     PASS
- C2 crossing σ(t)†=σ(−t): 2.0e-16                                PASS
- ε_H tracking: L1 residual 1.061e-6 @ ε=1e-6, 1.061e-8 @ ε=1e-8 —
  ratio 100.0, exactly linear                                     PASS

P=4 gate (2×2 mesh, 4 XLA host devices, single process; PRODUCTION
`_get_sigma_tau_kernel` kernels through the real shard_map/psum_scatter
tails; ns=2, symmetric random mask_B, masked E_A):
- G1 S_R+i·S_I = X (two-channel vs merged production kernels):
  Laplace 2.53e-16, crossing 3.11e-16   (gate 1e-12)              PASS
- G2 two-channel kernel vs numpy reference: 7.3e-16 / 6.8e-16     PASS
- G3 merged kernel vs numpy reference: 6.7e-16 / 7.1e-16          PASS
Topology honesty: single-process 4-device (XLA host devices), not 4 ranks —
the shard_map/psum_scatter code path is identical; multi-process collectives
are covered by the P=64 A/B (gate 2).

py_compile sweep (venv 3.12, compileall src/): rc=0                PASS

Stage B disposition: memo items (i)/(ii) are TRS checks — dropped per the
owner veto.  Item (iii), deck-level ε_H, runs inside gate 2 via
LORRAX_PPM_HERM_DIAG=1 on the actual fitted B_q at nb=128 AND nb=256.

## GATE 2 — restart-gated A/B at nb=128 + nb=256 (job 7878867): PASS

Six passes, all with the merge ON (k128m/k128ms/k128mf vs
run_AQ_c4962_p64_mpi; l1m/l1ms/l1mf vs run_L1_b256; *s = staged sub-rows,
*f = + LORRAX_FFT_FFI=1 + LORRAX_FFT_FFI_FUSED=1 composition).  All six
rc=0, cache-cold, coll=mpi (AS.7 env), merge announce printed in every log.

τ-node split (window tables; bounds the whole-loop win): nb=128 = 80
Laplace / 96 crossing of 176; nb=256 = 77 / 96 of 173 — the merge halves
projection work on ~45% of τ nodes, so predicted project_rs delta ≈ −22%.

PARITY (sigma_perf_ab_parity.py, tol 1e-12): max|diff| = 0.000e+00 on
sigma_diag/eqp0/eqp1, ALL SIX passes — text-precision exact (9-decimal
files, ~5e-10 eV resolution; the h5 gate below is the discriminating one).

H5 TENSOR GATE (h5_sigma_compare.py, all datasets, tol 1e-12): PASS ×6.
Honest max diffs (worst dataset sigma_c_kij_ev, value-level NOT bit-exact,
as designed — complex-GEMM association):
  k128m/k128ms 2.418e-14 eV (maxrel 3.28e-15); k128mf 2.587e-14 eV;
  l1m/l1ms 3.555e-15 eV (maxrel 4.19e-16);     l1mf 5.330e-15 eV.
Merge adds essentially nothing beyond the known pref-fold/FFI few-ULP level
(fftffi-era diffs were 2.414e-14 / 2.541e-14 at nb=128).
Note prod ≡ staged bitwise per engine (m vs ms identical diffs), as in
every previous round.

sigma.exec (production passes):
  nb=128: k128m  262.416 s  vs baseline 272.040 / this-tree neutral band
          278-280 → ~−6% wall (merge-only, XLA fft path);
          k128mf 64.227 s  vs FFI+fused reference 71.906 → −7.7 s (−11%):
          the win COMPOSES with the FFT backend (projection is 66% of the
          FFI τ wall; τ-weighted prediction ≈ −6.5 s).
  nb=256: l1m    78.537 s  vs run_L1_b256 86.99 / this-tree ref 90.2
          (−10..13%);
          l1mf   35.591 s  vs 86.99 XLA baseline (2.44×; first FFI figure
          at this shape, no prior FFI-b256 reference).

Staged per-τ sub-rows (project_rs = the projection GEMMs + both
reduce-scatters; staged rows carry the documented ±25% cross-run noise):
  nb=128 (k128ms, 176 τ): project_rs 47.64 (r2 stacking ref) → 40.36 s
    (−15%); omega_project 0.71 → 0.41 s (merged host consumer = ONE
    complex multiply, no re/im recombination).
  nb=256 (l1ms, 173 τ):  project_rs 35.6 → 24.60 s (−31%; prediction −22%);
    the projection bucket falls 39% → 29% of τ dispatch;
    omega_project 1.18 → 0.88 s.

COLLTABLE / HLO payload check (rank-0 dumps, both shapes): two tau-kernel
modules per run as designed —
  two-channel (crossing) module: rs=2, payloads c128[2,16,2,624,16] +
    c128[2,16,16,16] (nb=128) / c128[2,16,2,312,32] + c128[2,16,32,32]
    (nb=256) — the stacked re/im pair, unchanged;
  merged (Laplace) module: rs=2, payloads c128[16,2,624,16] +
    c128[16,16,16] / c128[16,2,312,32] + c128[16,32,32] — the stacked
    leading 2 is GONE: collective payload exactly halves per Laplace τ,
    message count unchanged (2/τ).  all-gather 0 in all modules.
Scaling guard: "NO collective carries a full (mu,mu) tile" at both
mu=4962 and mu=2475 (colltable) — the low-memory scaling target respected.

Deck-level ε_H (memo Stage B item iii, LORRAX_PPM_HERM_DIAG=1, RAW fitted
tensors over ALL q, normalized by global max):
  nb=128 / c4962: B_q residual 1.048e-01, Ω_q symmetry residual 9.999e-01;
  nb=256 / c2475: B_q residual 2.471e-06, Ω_q 4.447e-04.
FINDING: on the c4962 deck the raw fitted B_q is NOT numerically Hermitian
(10%!) and Ω_q has O(1)-asymmetric elements — consistent with fallback/
mask-boundary elements flipping on one side of the diagonal (the fit's
safe/valid thresholds are roundoff-sensitive; note the diag measures RAW
tensors, before the σ kernel's B_mask gating, so the masked operand may be
cleaner).  Strongly deck-dependent (c2475 is 5 orders cleaner).  This
VINDICATES both the memo's "measure, don't assume" clause and the owner's
rejection of any symmetry-assuming reduction (triangle-only transport
would have silently symmetrized a 10% asymmetry on c4962).  The merge is
indifferent — bilinearity — and the 1e-12 h5 gates above are the proof.

## pytest sweep (jobs 7878868 + attribution rerun)

Full-suite `pytest -q` in-container: IDENTICAL error/failure pattern with
the merge flag OFF and ON (same E/F progress string, both rc=1) — the flag
changes nothing in the suite.  The mid-run termination without a summary
is the LEDGERED PRE-EXISTING full-suite behavior on this branch: the
scorecard records `test_slate_cholesky_trsm_cpu` hanging/dying in-container
("PRE-EXISTING, not audit-introduced", bisected job 7877804; "the
full-pytest cell wk_REL/results/logs/pytest_v.7877798.out hits the same test") plus 14
known failures "reproduced bit-for-bit on untouched 19aeece" (13×
test_file_io stale zeta fixtures, 2× test_bse_w_omega_chain shard_map
carry) — which is why the house harness (pytest_v.sbatch) runs a curated
SUBSET.  ATTRIBUTION CLOSED (job 7878915, chmerge_pytest2.7878915.out): the
PRISTINE 5918cf6 worktree produces the BYTE-IDENTICAL pytest progress
string and rc=1 as the dirty tree with the flag off AND on (three cells,
identical output incl. the summary-less termination — the suite dies the
same way on the untouched tree).  Verdict: pre-existing at HEAD,
independent of this change and of the merge flag.  The pristine worktree
lives at /work2/08271/jackmc/frontera/wt-chmerge-pristine (remove with
`git worktree remove` after the orchestrator is done).

## Named, not done / hand-off notes

- chmerge_pytest2 attribution: DONE (job 7878915, pristine ≡ dirty ≡
  dirty+flag — see pytest section above).
- ε_H on c4962 (B_q 10% non-Hermitian RAW; masked operand not measured
  separately): worth a one-off look at whether the asymmetry lives
  entirely in fallback/invalid elements that B_mask drops — one more diag
  line in _prepare_sigma_state would answer it.  Physics outputs are
  1e-12-gated regardless.
- Owner docs condition (c) satisfied in-tree: manual 7.5 subsection +
  kernel/accumulator docstrings; commit is the orchestrator's call
  (nothing committed here).
- run dirs: mos2_4x4_test/run_CHMERGE_{k128m,k128ms,k128mf,l1m,l1ms,l1mf}
  (one run per dir, J.7 restart-gated, gw.in restart=true from the
  reference decks).
