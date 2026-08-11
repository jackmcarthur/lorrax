# wk_REL — Σ-stage perf implementation log (2026-07-28)
# SCORECARD: written up as SPEEDUP_SCORECARD.md ## AY (AY.4; nb>=256 d2h_wait attribution is named-not-done — job 7878233 l1 passes failed).

Branch `fix/zq-band-gather-device-invariance`, working tree ONLY (not committed).
Baseline for every A/B and parity gate: `/scratch2/08271/jackmc/mos2_4x4_test/run_AQ_c4962_p64_mpi`
(AQ 4962c, P=64 8×8, coll=mpi, cache-cold; sigma.exec **272.040 s**, 176 τ @ 1.51 s).
Evidence base: HLO module_0912.jit__tau_kernel + `wk_REL/results/sigma_perf_candidates.json`.

## Landed in tree (round 1, verified by job 7878038)

1. **Instrumentation** (`ppm_sigma.py`, `ppm_tau_kernel.py`):
   - always-on rows `sigma.tau.dispatch`, `sigma.tau.host_accum` (µs overhead),
     `sigma.finalize` (per branch), `sigma.host_gather` (once per stage);
   - `LORRAX_SIGMA_TAU_TIMING=1` → stage-split τ kernel (w_phase / G_build /
     G_ifft / V_ifft / GW_mult_fft / project_rs), blocking rows, same op sequence;
   - `jax_profile.trace_section` session per branch (first window), active only
     under ISDF_JAX_PROFILE_DIR.
2. **Finalize-tail comms fix** (AK.9 second lever; `ppm_accumulators.py` `host_tiles()`,
   `ppm_sigma.py` `_SigmaBranchTiles` + driver): Σ branch tiles stay per-rank host
   numpy across branches; ONE end-of-stage gather replaces 4× (device re-upload +
   full-slab 64-process allgather). Outputs bit-identical by construction.

### Round 1 verdict (job 7878038, two passes, cache-cold)
- **Parity: exact 0.0** max|diff| on sigma_diag.dat / eqp0.dat / eqp1.dat, BOTH passes.
- pass A (production): sigma.exec **273.278 s** vs 272.040 baseline (+0.5%, node noise);
  `sigma.finalize` 0.000×4, `sigma.host_gather` 0.244×1; all Finished→Started seams 0 s.
- pass B (staged): sigma.exec 313.503 s (instrumented, NOT comparable); **per-τ split of
  295.0 s dispatch: w_phase 5.70 / G_build 11.17 / G_ifft 79.63 / V_ifft 16.54 /
  GW_mult_fft 95.74 / project_rs 84.75; host numpy ω-projection only 0.71 s total.**
  → FFT-adjacent layout churn = 191.9 s (65%) of the staged τ time; collectives+dots
  (project_rs) 84.7 s; W-phase small (candidate #3's exp specialization low-value).
- Tau-kernel HLO byte-equivalent to baseline (fft 3, transposes 6, rs 4, ag 0). ✓
- **CLAIM-DECAY vs candidates.json**: the analysts' "4-5 s/branch finalize tail" was
  the async-D2H deque drain (by design); the baseline's own Finished→Started seams
  were 0-1 s, so the eliminated gather cost only ~1-4 s at nb=128. The fix's value is
  AK.9's nb²-growth term (~237 MB/branch at b160), not this deck's seconds.
- AK.6c pad check: `pad_sigma_window` resolves to identity on both axes here
  (nb=128, 8×8; HLO shows n=128 unpadded). NOTE: nb=128 ≡ 0 mod 64 → this deck cannot
  discriminate per-axis vs old product rule; the discriminating case is nb=70 @ 8×10.

## Landed in tree (round 2, awaiting clean verification)

3. **AK.9 collective stacking** (`ppm_tau_kernel._make_project_ri_reduce_scatter`):
   both re/im channels ride ONE psum_scatter per mesh axis (stacked leading axis),
   4 → 2 collectives/τ at identical bytes. Per-channel einsum/cast sequence
   token-identical (OWNER_HOLD respected). Expect HLO: rs 4→2, payloads
   2×5.11→1×10.22 MB ('x'), 2×0.07→1×0.13 MB ('y').

### Round 2 (job 7878092) verdict
- pass A = INFRA FAILURE, not code: c208-020 apptainer `squashfuse_ll failed
  to mount` (tasks 8-9 exited 255 at container creation) → 62/64 ranks
  DEADLINE_EXCEEDED at RegisterTask → rc=255. No Python frame reached.
  Node excluded in subsequent jobs (`--exclude=c208-020`).
- pass B (staged, WITH stacking): rc=0, **parity exact 0.0** on
  sigma_diag/eqp0/eqp1 → AK.9 stacking verified bit-exact in practice.
- HLO (module_0922.jit__project_ri_reduce_scatter): **reduce-scatter 4 → 2**;
  payloads c128[2,16,16,2,624] on stride-8 'x' groups + c128[2,16,16,16] on
  consecutive 'y' groups — exactly as designed.
- Staged rows w/ stacking: project_rs **84.7 → 47.6 s**; BUT collective-free
  rows moved +22-28% between runs (G_ifft 79.6→101.9, V_ifft 16.5→25.6,
  GW_mult_fft 95.7→101.9) with staged TOTAL ~unchanged (295.0→295.5 s) →
  staged rows carry ±25% cross-run noise; at ~10 MB messages the stacking
  mostly relocates latency/skew wait rather than removing wall at this shape.
  Claim recorded with measured domain per house rule.

## Landed in tree (round 3, iteration 1 pending job 7878110)

4. **Axis-order swap** (owner-approved, movement-only): projection contracts
   μ_Y first — the LARGE stacked partial reduce-scatters over 'y'
   (consecutive-rank groups, node-local pairs), the small final block over
   stride-8 'x'. Per-channel order becomes ψ*·(σ·ψ): value-identical, NOT
   bit-exact — 1e-12 parity-gated. Shared body `_project_ri_local` now feeds
   both the standalone projector and the fused kernel.
5. **Monolithic fused τ kernel** (`LORRAX_SIGMA_FUSED_TAU`, default OFF until
   gates pass): ONE shard_map over rank-local tiles (isdf/core.py
   c_q_from_psi_sm pattern) — build_G_tau local einsum, fft_helpers
   `local_ifftn3/local_fftn3` (the family's documented inside-shard_map
   kernels; wrappers can't nest) at same axes/norm, same multiply expression,
   shared `_project_ri_local`. Targets the flat-k helper-boundary transposes
   (module_0912 fused_computation.6 double transpose of the 398 MB tile).

### Iteration 1 VERDICT (SIGMA_iter job 7878110, restart-gated, 4 passes)
All four passes rc=0 and **parity exact 0.0** (sigma_diag/eqp0/eqp1) — the
restart-gated deck reproduces the full-run outputs bitwise, validating the
AC.4 harness for Σ-only iteration.
- a (stacking+swap, prod): sigma.exec **278.049** (baseline 272.040,
  round-1 273.278). The axis swap is parity-exact-0 and HLO-clean but shows
  no measurable win at this shape (Δ within the ±5 s cross-run band).
- c (FUSED, prod): sigma.exec **278.959** — performance-NEUTRAL; d's single
  fused row: 265.06 s/176 τ = 1.506 s/τ ≡ decomposed path.
- **Fused-module HLO REGRESSION: transposes 20 (18 large) vs 6 in the
  decomposed module** (rs=2, ag=0 fine). XLA:CPU re-anchored layouts WORSE
  inside the merged shard_map — the hoped-for k-minor end-to-end layout did
  not materialize; the flat-k helper boundaries were not the binding
  constraint at this shape.  Both owner gates (neutrality + near-zero large
  transposes) FAILED → per instruction, not forced.

## host_accum at nb=256 (run_L1_b256): attribution analysis + gate in flight

Claimed: "host ω-projection explodes with nb (72.7 s / 84% at nb=256 vs
0.7 s at nb=128)".  The comparison mixes ROW SEMANTICS: 0.7 s was the
STAGED pass (host-only; device wait absorbed by blocking stage rows), while
72.7 s is the PRODUCTION pass, where `sigma.tau.host_accum` documentedly
absorbs the device-compute wait through the lag-2 drain's `np.asarray`.
The comparable production numbers are: nb=128/μ=4962 host_accum **243.5 s**
→ nb=256/μ=2475 host_accum **72.7 s** — it TRACKS THE DEVICE TILE (399 →
100 MB), not nb².  L1 per-τ wall (LoopProgress): 33 s/71 τ = 0.465 s/τ ≈
the device-kernel estimate at a 4×-smaller μ² tile.  Working hypothesis:
the L1 σ loop is still DEVICE-bound; host ω-projection is ~3 s there
(0.71 s at nb=128 × 4 for the nb² tile).
Actions landed (working tree, post-dc30af4):
- `sigma.tau.host_accum` split into `sigma.tau.d2h_wait` (device wait) and
  `sigma.tau.omega_project` (pure numpy) — the observable now discriminates.
- `_project_tau_onto_omega_np`: pref folded into the (n_ω,) coeff and the
  redundant `.astype` full-copy dropped — 3 full-size array passes → 1 per
  τ-shard.  pref-fold is value-identical NOT bitwise (flagged in-code);
  astype-drop is bit-exact.
### VERDICT (jobs 7878233 k128{a,b} + 7878276 l1{a,b}) — prediction confirmed
All four passes parity PASS (k128 vs AQ baseline; l1 vs run_L1_b256 —
l1's 17 s failures in 7878233 were a harness input-link bug, fixed by
cloning the reference run's link set; b256 decks link WFN_b256/dipole_b256/
kin_ion_b256).
- nb=128 prod: d2h_wait 245.0+24.4 s vs **omega_project 0.71 s**; staged:
  omega_project 0.42 s.  sigma.exec 279.9 (neutral, 278-280 band).
- nb=256 prod: d2h_wait 71.2+7.3 s vs **omega_project 1.18 s**; sigma.exec
  90.2 vs 87.0 reference.  **The "72.7 s host_accum wall at nb=256" was
  the row absorbing DEVICE wait — the host ω-projection is ~1 s.**
- nb=256 staged device split (l1b, 173 τ, 91.2 s): **project_rs 35.6 s
  (39%) — now the LARGEST bucket** — G_ifft 20.4 + GW_mult_fft 19.5 +
  V_ifft 6.4 (FFT/layout 51%), G_build 7.1 (7.8%), w_phase 1.5.
Unified nb-scaling picture for rung 2+: the projection GEMMs scale
∝ nb·μ² (share 16% → 39% from nb=128→256 at μ²/4) and become the σ wall;
their re/im channel structure is OWNER-HELD (derivation-level work, AK.9 /
candidates #2 — the "half the zgemm flops" lever).  FFT/layout buckets
scale ∝ μ² and are CLOSED as structural on XLA:CPU (see below).
[**REFUTED 2026-07-29, banner added 2026-08-11.** "CLOSED as structural"
survived about 24 h: the MKL-DFTI FFT FFI (job 7878727) took the trio 191.9 s →
4.99 s, `FFI_EVIDENCE_AUDIT.md` F22.  Two further corrections from
2026-08-11: the scaling is ∝ μ² **and ∝ nk**, and nk is the axis that
dominates — the FFT is 16%/60%/85% of the τ dispatch at 9/64/216 k-points on
GPU; and the projection GEMMs do NOT "become the σ wall" at every scale —
`project_rs` falls to 9% of the τ dispatch at Si 6×6×6 while the FFT rises to
85%.  See
`tests/known_failures/2026-08-11-gnppm-sigma-performance-claims-adjudicated.md`.]
Band
slicing (owner item 3): the FFTs never see nb; the only full-nb-under-mask
cost is G_build = 7.1 s/91 s at nb=256 → a ~3-5 s win costing per-extent
recompiles — sized, available, not landed (below envelope threshold).
Host thread: exonerated at both shapes; the landed projector cleanup
(pref-fold + no-copy) is parity-exact-0 at both shapes and keeps
omega_project ≈ 1 s.

## ω-cube full-slab gather at nb=256 (coordinator query, answered)

The 687.87 MB/rank collective (c128[41,16,256,256]) in run_L1_b256's table
is the END-OF-STAGE reconstruction of Σ_c(ω,k,m,n) onto every rank — that
run DID import this working tree (its log has the sigma.tau/host_gather
rows), so the 4→1 finalize fix was already active: ONE such gather (old
code: 4 per-branch gathers totalling ~2× that volume) + the driver's
`jnp.asarray(sigma_kij_host)` re-upload.  `sigma.host_gather` measured it
at 0.833 s at nb=256.  Fully removing the nb²·n_ω replication means the
downstream consumers (head injection, diag interpolation, sigma_mnk write)
must accept a sharded/rank-0 ω-cube — AK.9 names this "its own workstream";
not attempted here.  Growth to watch: ~2.75 GB/rank replicated at nb=512.

## chi0-vs-sigma kernel comparison (owner directive, 2026-07-28)

HLO: chi0's τ-scan module (jit_minimax_tau_integrate_chi, same cold dumps)
has fft 3 / transposes 22 (16 large) / copies 28 / collectives 0 — chi0 is
NOT transpose-free; where it crosses dot→fft (2 G's per τ) it pays the
same class of full-tile transposes sigma pays.  Its 0.90 vs 1.51 s/τ
advantage at equal μ (despite TWO G builds + TWO FFTs per τ) decomposes as:
(a) LAYOUT: chi0's post-FFT consumer is the elementwise spin-trace
    'Rambn,Rambn->Rmn' — layout-agnostic, consumes the FFT output k-minor,
    NEVER returns to dot layout.  Sigma must return (ψ-projection GEMM),
    and additionally round-trips at the G·W multiply because the flat-k
    helper's exit reshape (k-minor → flat k-major) anchors layouts
    (module_0912 fused_computation.6, ~1.7 GB/τ);
(b) no per-τ collectives; (c) lax.scan (no per-τ dispatch); (d) no
    ω-projection/D2H tail.
Portable piece — REFUTED with HLO evidence before re-implementation: the
candidate "scoped convolution helper" (ifft·mult·fft in one region) was
already CONTAINED in the refuted fused module, and its dump
(run_SIGMA_iter_iter1_c module_0551) shows XLA STILL computed the
multiplies k-major and transposed to k-minor for each FFT
(transpose.50/.52/.54 on the 398 MB tile — the same double-transpose as
the helper-boundary path).  The anchor is XLA:CPU's layout assignment at
the flat(k-major)↔3-D(k-minor) reshape — present in ANY implementation of
the convolution — not the helper/jit seams.  chi0 escapes only because it
never FFTs the big tile back (its consumer is an elementwise contraction
into the SMALL (nk,μ,μ) χ); sigma's ifft→mult→fft over the same μ² object
forces the round trip under XLA:CPU regardless of module structure.  Two
independent implementations at identical τ rate (1.510 vs 1.506 s/τ)
confirm.  The dot↔fft layout alternation is CLOSED as not-removable on
this backend without changing what the kernel computes (owner-held) or
the backend (GPU relowers layouts differently — re-open there if the GPU
lattice campaign needs it).
[**REFUTED, banner added 2026-08-11.** The FFI removed it on this very
backend the next day (`FFI_EVIDENCE_AUDIT.md` F22).  The "re-open there"
invitation was taken up on GPU and the answer is that the FFT is 85% of the τ
dispatch at Si 6×6×6/216 k — `2026-08-11-gnppm-sigma-performance-claims-adjudicated.md`.]
Owner item (3), band slicing: the FFTs never see the band axis (they act on
the μ² pair object); the full-nb-under-mask cost sits ONLY in the G-build
GEMM (contracts all nb with masked phases — val branch uses ~26 of 128).
Sizing: G_build = 11.2 s of 295 s staged at nb=128; grows ∝nb·μ² — check
l1b's G_build row at nb=256 before investing (union-band-range slicing is
exact; costs per-extent recompiles, bucketable).

### Fusion: tried and REFUTED at nb=128/μ=4962/P=64 (claim-scope record)
Reverted from the tree 2026-07-28; re-apply patch preserved at
`wk_REL/docs/patches/fused_tau_refuted_2026-07-28.patch`.  Scope conditions: measured
ONLY at MoS2 4×4, nb=128, μ_pad=4992, 8×8 mesh, XLA:CPU (DUCC fft,
Eigen dots).  Mechanism of refutation: the per-τ wall is NOT dominated by
the helper-boundary transposes the fusion removes — XLA emits MORE
transposes when free to re-anchor inside one shard_map region, and the τ
rate is unchanged (1.506 vs 1.51 s/τ).  Also costs the per-τ sub-row
observability (single 265 s row).  Re-open only with concrete evidence
from a different shape/backend (e.g. GPU, or nb≥256 profiles showing the
boundary transposes dominate there).

## Iteration harness
`$D/sigma_iter.sbatch` (SIGMA_iter, ITER_TAG env): AC.4 restart-gating —
restart=true + tmp/ symlinked from round-1 pass A's parity-verified
isdf_tensors_4962.h5 (14 GB; J.7 satisfied: identical gw.in/centroids).
Re-enters at restart_load → screening → Σ; two passes (prod + staged) per
iteration + parity + colltable + tau-kernel HLO counts.

## Queue (owner-approved sequence)
1. Clean verdict for AK.9 stacking (SIGMA_iter or rerun).
2. Quick win: psum_scatter axis-order swap — large payload to the 'y'
   (consecutive-rank, node-local-pair) groups, small to stride-8 'x'.
   Owner-approved as movement-only; parity-gated (value-identical, contraction
   order swaps per channel).
3. MAIN: monolithic one-shard_map τ kernel (zeta-style; isdf/core.py pair
   pipeline as reference): rank-local tiles, ONE layout, fft_helpers' local
   kernels (`local_ifftn3`/`local_fftn3` — the family's documented
   inside-shard_map entry points; shard_map cannot nest, so the flat-k
   reshape is inlined around the SAME single-source local FFT), projection
   dots in the same layout. Gates: zero collectives in fft steps, near-zero
   large transposes, parity 1e-12, per-τ sub-row A/B vs round-1 pass-B table.
