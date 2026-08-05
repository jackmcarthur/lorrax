# wk_REL — contract_bands_block_reshard: the shared band projection+reshard primitive (2026-07-28)

Owner-directed architectural primitive (this workstream's charter): ONE
source of truth for the multi-stage band projection + reshard, isolated in
`src/common/contract_bands.py` for future agents to study; sigma adoption;
gated FFI MKL GEMM body; BSE adoption MAP ONLY (no BSE wiring).

Tree: /work2/08271/jackmc/frontera/lorrax @ 3c44494, WORKING TREE ONLY (not
committed; orchestrator merges).  Prereads: RESHARD_OVERHEAD_MEMO.md (the
lever list, the impl=mpi world-collective-first contract Sec. 4.2, the BSE
audit Sec. 6), lgemm_notes.md (de-promotion lowering), ffi mklfft/scalapack
handler patterns, QUALITY_PATTERNS.md, ARTIFACT_MAP conventions.

⚠ TREE CONFOUND (recorded before any measurement): the working tree ALSO
carries a separate in-flight owner-ordered change (Laplace merge made the
DEFAULT; `LORRAX_SIGMA_LAPLACE_MERGE` flag removed from ppm_tau_kernel/
ppm_sigma; landed on disk mid-workstream by another agent).  Consequences
for this workstream's rows: nb=128 perf refs (project_rs 38.688 / sigma.exec
66.470, job 7878942) are merge-OFF numbers — the nb=128 rows here are
NEW-BASELINE rows, not direct A/Bs; nb=256 refs (20.612 / 35.497, job
7878977) ran MERGE=1 and ARE directly comparable.  Value parity is
plan-invariant (bilinearity, chmerge gates) so all parity/h5 gates remain
valid against the original baseline outputs.

## 1. API sketch (src/common/contract_bands.py)

    from common.contract_bands import (
        contract_bands_block_reshard,     # THE primitive (factory)
        bands_gemm_ffi_enabled,           # LORRAX_BANDS_GEMM_FFI (factory-time)
        ensure_grouped_collectives_ready, # impl=mpi world-first warm-up
    )

    project = contract_bands_block_reshard(
        mesh_xy,
        channels="none" | "split_reim",   # split_reim = sigma two-channel plan
        extra="none" | "leading" | "minor",  # caller stack axis position
        axes=("x", "y"),                  # ax_x shards mu/m, ax_y shards nu/n
        divisibility_hint="...")          # appended to the refusal message

    out = project(psi_left, O, psi_right)

Math: out[extra?, k, m, n] = sum_{s,mu} sum_{s',nu}
      conj(psi_left)[k,m,s,mu] · O[extra?,k,s,mu,s',nu] · psi_right[k,s',nu,n]

Canonical shardings:
    psi_left  (nk, m, s, mu)              P(None,None,None,'x')
    O         (nk, s, mu, s', nu)         P(None,None,'x',None,'y')
      leading (E, nk, s, mu, s', nu)      P(None,None,None,'x',None,'y')
      minor   (nk, s, mu, s', nu, E)      P(None,None,'x',None,'y',None)
    psi_right (nk, s', nu, n)             P(None,None,'y',None)
    out       (nk, m_X, n_Y) (+E)         P(None,'x','y') (+None)
    channels="split_reim": returns the (S_R, S_I) tuple (crossing plan).

Raison d'etre (stated in the module docstring): NO (m,n,k)-, (m,mu)- or
(mu,mu)-sized object is ever materialized on one rank; the chain is
right-contract -> psum_scatter('y', LARGE payload, consecutive-rank
groups) -> left-contract -> psum_scatter('x', small payload).

Encoded policies (each with evidence cites in the docstring):
1. large-payload-on-node-local-axis: 'y' contracted FIRST; a mesh whose
   minor axis is not ax_y is REFUSED (memo Sec. 6.1 finding #1 is exactly
   this inversion in the BSE inner loop).
2. AK.9 stacking: channel/extra slices ride ONE collective per mesh axis.
3. de-promoted f64-split lowering wherever a real operand would meet a
   complex one in the large right GEMM (automatic, dtype-driven);
   complex×complex stays a single zgemm (f64-split refuted there, job
   7878942).
4. impl=mpi world-collective-first warm-up (memo Sec. 4.2): the factory
   calls ensure_grouped_collectives_ready() — one world-clique barrier
   collective, once per process, only under
   JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi multi-process.  Standalone
   consumers (tests/benches/BSE drivers) inherit it instead of dying at
   their first grouped psum_scatter.
5. divisibility guard: actionable refusal naming the per-axis pad fix
   (never the p_x*p_y product rule).

Gated CPU GEMM body: LORRAX_BANDS_GEMM_FFI=1 routes ONLY the large right
contraction through `lorrax_mklblas_gemm_batch`
(src/ffi/mklblas/cpp/gemm_batch_ffi.cc — cblas_{d,z}gemm_batch, row-major
NN, C[i] = A[i] @ B[i % BB] broadcast rule for the extra-stacked batch,
MklThreadScope thread pin, LORRAX_MKLBLAS_THREADS auto|off|N,
LORRAX_MKLBLAS_LOG).  input_output_aliases: none — no GEMM output can
legally alias an operand buffer (documented; contrast the in-place FFT
handlers).  Announce-or-refuse per house rules; the flag REFUSES on
non-CPU meshes (CUDA keeps the native XLA lowering — cuBLAS is already
optimal) and refuses extra="minor" (contracted axis not GEMM-reachable
without a full-tile transpose).  Kernel caches key on
bands_gemm_ffi_enabled() (ppm_tau_kernel pipeline/cache keys).
Build: config/frontera/build_ffi_host.sh (the TU rides the existing MKL
link line; CMake block mirrors mklfft's); loader table entry in
ffi/common/ffi_loader.py `_HOST_TARGET_SYMBOLS`.

## 2. Sigma adoption (behavior-identical, gated)

`gw.ppm_tau_kernel._make_project_ri_reduce_scatter` is now a thin
Σ-specific wrapper: merged_x=False -> channels="split_reim",
merged_x=True -> channels="none".  The historical `_project_ri_local` /
`_project_x_local` bodies were REMOVED — their op sequences live verbatim
in the primitive's `_body_split_reim` / `_body_none` (this was the
subsumption, not a rewrite); the Σ-owned channel algebra (crossing
consumer needs (S_R,S_I); Laplace consumer forms only c·X) stays in the
factory docstring with the derivation cites.  Kernel cache keys extended
with the GEMM-FFI flag.  Consumers of the factory (tests, reshard_ubench)
unchanged.

## 3. Gates

Gate order: cbands_gate.sbatch (build + unit + microbench + pool probe)
first; cbands_ab.sbatch (32N/P=64 restart-gated A/B, GEMM off AND on,
nb=128 + nb=256) only on PASS.

### GATE 1 — cbands_gate (job 7879008): PASS (all cells rc=0)

- build-host -> build_host_BANDS: rc=0; nm exports MklBlasGemmBatchHostFfi
  + MklFftFlatKHostFfi + ScalapackEighHostFfi + phdf5 (superset lib; the
  A/B uses it for ALL passes).
- pycompile: rc=0.
- tests/test_contract_bands.py: rc=0 — 7/7 PASS (parity ≤1e-14 for
  complex/split_reim/leading/minor/real-O plans; rs=2/module at exact
  payload shapes incl. stacked leading-2, leading-3, trailing-3; zero
  rank≥2 f64→c128 converts; all four refusals actionable; FFI plan:
  1/4/1 custom-calls as designed, parity ≤1e-14, rs payloads identical
  to the XLA plan, minor-order refusal fires).
- tests/test_projection_lgemm.py on the ADOPTED tree: rc=0 — the
  pre-existing sigma-factory HLO pins hold unchanged (subsumption is
  lowering-identical).
- check_channel_hermiticity --stage p4: rc=0 — all six gates at
  2.5e-16..7.3e-16, values identical to the chmerge/lgemm rounds (same
  seeds) through the adopted kernels.
- cbands_ubench + gemm_pool_probe: Sec. 4 / Sec. 5.
- (Traceback lines in the log = the pre-existing CUDA-plugin discovery
  banner, JAX_PLATFORMS=cpu venv noise — same as lgemm-gate 7878939.)

Incidental P=64 coverage: the CONCURRENT LMDEF_ab job (7879005, the
other agent's merge-default A/B) snapshotted the tree at 20:27:51 —
AFTER this workstream's adoption edits (20:24-20:26) — and its two
XLA-path production passes ran the ADOPTED kernels end-to-end at P=64:
rc=0, parity=0, h5 PASS, rs=2/ag=0 (k128 sigma.exec 252.394, l1
76.674).  Not this workstream's gate, but real adopted-tree coverage on
the XLA (non-FFT-FFI) route this A/B does not re-run.

### GATE 2 — cbands_ab (job 7879010, 32N×2/P=64, cache-cold, coll=mpi, all 8 passes rc=0): PASS

PARITY: sigma_diag/eqp0/eqp1 max|diff| = 0.000e+00 (text precision) ALL
EIGHT passes vs the original baseline dirs (B128=run_AQ_c4962_p64_mpi,
BL1=run_L1_b256).  H5 TENSOR GATE (all datasets, tol 1e-12): PASS ×8 —
worst dataset sigma_c_kij_ev 2.495e-14 eV @nb=128 (GEMM-on rows 2.2e-14 —
same ULP class as the lgemm/chmerge eras), 9.970e-15 eV @nb=256.

COLLTABLE + HLO (rank-0, cache-cold): "NO collective carries a full
(mu,mu) tile" at μ=4962 AND μ=2475, all four prod dumps; tau-kernel
modules rs=2 / all-gather=0 in every pass; reduce-scatter payloads
BYTE-EQUAL GEMM-off vs GEMM-on and exactly the chmerge tables:
  nb=128 two-channel c128[2,16,2,624,16]+c128[2,16,16,16];
  nb=128 merged      c128[16,2,624,16]+c128[16,16,16];
  nb=256 crossing    c128[2,16,2,312,32]+c128[2,16,32,32];
  nb=256 merged      c128[16,2,312,32]+c128[16,32,32].
Dot/custom-call pins, per module (WANT = designed):
  GEMM off: two-channel 4 f64 μ-dots + 0 custom-calls; merged 0 f64 +
  2 c128 μ-dots (G-build + merged right); converts 0 everywhere.
  GEMM ON:  two-channel 4 mklblas custom-calls + 0 f64 μ-dots; merged
  1 custom-call; the only surviving c128 μ-dot is the G-build dot
  (present identically in every baseline dump); converts 0.

ROWS (sigma.exec prod / project_rs staged; d2h_wait absorbs device wait
on prod rows as always):

| pass | config | sigma.exec | project_rs (staged) |
|---|---|---|---|
| b128f/b128fs | nb=128 FFI-fused, GEMM off | 58.313 | 29.407 s/176τ |
| g128f/g128fs | nb=128 FFI-fused, GEMM ON  | **49.224** | **19.622 s/176τ** |
| l256f/l256fs | nb=256 composed, GEMM off | 35.234 | 20.565 s/173τ |
| m256f/m256fs | nb=256 composed, GEMM ON  | **29.979** | **14.162 s/173τ** |

ADOPTION NEUTRALITY (the directly comparable shape): l256f 35.234 vs
35.497 ref (−0.7%) and l256fs project_rs 20.565 vs 20.612 ref (−0.2%) —
row-neutral within noise; plus the concurrent LMDEF_ab XLA-path passes on
the adopted tree (parity 0, h5 PASS).  nb=128 GEMM-off rows are NEW
BASELINES (merge-default confound, header note): 29.407/58.313 vs the
merge-off 38.688/66.470.

## 3b. GEMM VERDICT (owner question): the FFI MKL GEMM REALIZES the memo's ~19 s

- nb=128 staged project_rs: 29.407 → **19.622 s** (−33%; 167→111.5
  ms/τ).  The memo's "~19.8 s at BLAS rate" was projected for the
  merge-OFF mix; on the merged-default mix the measured GEMM-ON row
  lands at 19.6 s — the MKL-rate GEMM delivers the memo's number.
- nb=128 prod sigma.exec: 58.313 → **49.224** (−15.6%).
- nb=256 composed: project_rs 20.565 → **14.162** (−31%); sigma.exec
  35.234 → **29.979** (−14.9%).
- Campaign arc for the composed nb=256 stack: 272.0 (AQ era) → 35.6
  (chmerge) → 35.5 (lgemm) → **29.98** with the primitive + FFI GEMM.
- Residual per-τ project_rs at nb=128 GEMM-ON ≈ 111.5 ms/τ ≈ collect-
  ives (~50 ms shim chain) + casts/left/stacks + skew per the memo
  Sec. 5 decomposition — the next levers are L3 (FFI direct-MPI chain)
  + L1 (strip overlap), which slot BEHIND this primitive's interface.
Claim scope: MoS2 4×4 deck, μ_pad=4992 (nb=128) / 2496 (nb=256), 8×8
mesh on 32 nodes × 2 ranks, coll=mpi (AS.7), FFT-FFI-fused stack,
cache-cold, jobs 7879008/7879010; XLA:CPU only (the flag refuses on
CUDA by design).

## 4. Owner question 1 — extra/omega-dim order (leading vs minor): MEASURED

cbands_ubench (job 7879008 cell ubench): production local shapes
(O_loc = (16,2,624,2,624) c128 ≈ 398 MB/device, m=n=128), 2×2
emulated-device mesh, min-of-8 reps after warm-up:

| case | ms/call (min) | right-GEMM GF/s-equiv |
|---|---|---|
| extra=none  E=1 | 286.18 | 356.7 |
| extra=leading E=2 | 595.62 | 342.7 |
| extra=minor   E=2 | 588.38 | 347.0 |
| extra=leading E=4 | 1159.35 | 352.2 |
| extra=minor   E=4 | 1130.53 | 361.1 |

MINOR wins by 1.2% (E=2) and 2.5% (E=4) — inside the single-node noise
class; per-E cost is flat (no stacking penalty in either order).
Measured domain: single node, 4 emulated devices sharing one 28-core
pool, collectives are SHM copies — this prices the local GEMM/stack
lowering (the 71% component of the production row), NOT the wire; and
the FFI GEMM plan REFUSES minor (contracted axis not GEMM-reachable), so
adopters wanting the MKL-rate body must use leading.  Default stays
"leading" (matches the production stacked-collective payload tables
byte-for-byte); the numbers above are the recorded answer to the owner
question.

## 5. Owner question 2 — pool-scaling probe: MEASURED — NOT pool-miswired; Eigen per-thread quality + in-module overhead

gemm_pool_probe (job 7879008), single process, GF/s (min-of-5, warm).
prod-* = the production right-contraction shapes (16 batches
(1248×1248)@(1248×128)); sq-* = equivalent-flops square GEMM.  Width
controlled by the CPU AFFINITY MASK AT CLIENT CREATION (taskset) —
**that is the knob that works**: the XLA:CPU client sizes its Eigen pool
from the usable-core count once at creation.

| width | prod-f64 xla/MKL | prod-c128 xla/MKL | sq-f64 xla/MKL | sq-c128 xla/MKL | xla/MKL ratio band |
|---|---|---|---|---|---|
| 1  | 19.6 / 45.5 | 20.6 / 56.9 | 21.0 / 67.8 | 21.4 / 67.5 | 2.3–3.2× |
| 6  | 105.1 / 226.1 | 114.4 / 294.2 | 113.9 / 315.6 | 121.9 / 319.7 | 2.2–2.8× |
| 14 | 225.0 / 444.3 | 251.0 / 571.1 | 258.1 / 571.1 | 279.2 / 608.6 | 2.0–2.3× |
| 28 | 388.5 / 609.0 | 467.7 / 848.7 | 498.3 / 817.7 | 521.1 / 820.5 | 1.6–1.9× |

Controls (both at 28 cores):
- OMP_NUM_THREADS=1: xla rates UNCHANGED (361.6/452.4/496.3/536.6) —
  **OMP_NUM_THREADS does not govern the Eigen pool** (documented, as the
  owner suspected).
- XLA_FLAGS=--xla_cpu_multi_thread_eigen=false: ACCEPTED (no F-abort)
  but a NO-OP for rates (370.4/470.1/492.8/537.8) — the flag exists in
  DebugOptions (strings scan: xla_cpu_multi_thread_eigen,
  intra_op_parallelism_threads, ExecutableRunOptions::
  set_intra_op_thread_pool) but the thunks runtime ignores it here.

INTERPRETATION (per the owner's contract):
- Scaling is NOT flat: 19.6→388.5 GF/s over 1→28 threads (19.8× = 71%
  parallel efficiency; MKL itself only scales 13.4× = 48%, DRAM-bound
  earlier).  **No pool/sharding wiring defect** — the client pool under
  the production taskset (28 cores/rank) is correctly sized; the
  batched-skinny shape scales as well as the square one (no cost-model
  shard-count pathology at these shapes).
- The bare dot saturates **1.6–1.9× below MKL at width 28** (2.3–3.2×
  at 1 thread) — per-thread Eigen kernel quality, inside the owner's
  "2-4× below MKL" Eigen-verdict band.
- NEW datum the probe exposes: the bare isolated dot runs 388 (dgemm) /
  468 (zgemm) GF/s — substantially FASTER than the same contraction
  measured inside the production module (172 dgemm / 295 zgemm, memo
  Sec. 4.4/lgemm_notes): the remaining ~2× of the production deficit is
  NOT the dot kernel itself but the surrounding module (fused
  real/imag-extract streams feeding the dots, stack copies) and/or
  in-job co-tenancy (2 ranks/node).  The FFI-GEMM A/B row (job 2) is
  the measurement that says how much of the production row MKL-rate
  GEMMs actually recover — quoted in the gate table below, not
  projected from this probe.

## 6. BSE adoption map (owner review — NOTHING WIRED)

Reference: memo Sec. 6 audit.  All shapes below at the audit cell
(mu_pad=4992/P=64: mu_loc=nu_loc=624, nk=16, ns=2).

### 6.1 #1 bse_stack_matvec._w_stack decode (bse_stack_matvec.py:126-131) — THE target

Current per trial (inside the lax.scan body, :108-132):

    A    = psum_scatter(einsum("kctM,MNtsk->cNsk", conj(psi_c_X), U_b),
                        "x", dim=0)     # LARGE (c_full, nu_loc, ns, nk) on 'x'  <- INVERTED
    WXcv = psum_scatter(einsum("kvsN,cNsk->cvk", psi_v_Y, A),
                        "y", dim=1)     # small (c_loc, v_full, ns, nk) -> v on 'y'

Adoption (movement-only, value-level identical; per-trial primitive call
= the AXIS+ORDER FIX, memo finding #1):

    decode_b = contract_bands_block_reshard(mesh, channels="none")
    #   psi_left  = psi_c_X            (nk, c_full, t, mu_loc)  — ALREADY canonical
    #   psi_right = psi_v_Y transposed (nk, s, nu_loc, v_full)  — one hoisted
    #               rank-local transpose per SOLVE (with the M_X/M_Y hoists),
    #               NOT per matvec
    #   O         = U_b in layout (k, t, M, s, N)
    out_b (nk, c_X, v_Y) -> local transpose -> (c_loc, v_loc, nk)

Layout prerequisite (the real work item): U_b currently emerges
(M, N, t, s, k) because the conv chain runs local_ifftn3/fftn3 over MINOR
k axes.  The primitive wants k-LEADING (k, t, M, s, N).  Two routes:
  (a) relabel the encode einsum output ("kctM,cksN->ktMsN" instead of
      "->MNtsk") and run the conv with the flat-k FFI backend
      (LORRAX_FFT_FFI), which READS k-major natively — the layouts the
      primitive wants and the FFT-FFI wants are the SAME layout, so the
      two adoptions compose instead of fighting;
  (b) XLA path: same relabel + move the FFT axes handling to leading-k
      3-D form (transpose cost priced by HLO before claiming — pattern
      QP#4; do not wire without the trace).
Effect on movement: large decode payload (11-12 MB/trial class at the
audit shape) moves from the stride-8 zero-locality 'x' groups to the
consecutive-rank node-local 'y' groups — the exact owner-approved sigma
swap, transposed to (c,v).  Collective COUNT per trial unchanged in this
step (2 psum_scatter + the 2 encode all_gathers).

Trial stacking (memo L4(b), 4b -> 4 collectives/matvec) — requires a
TWO-PHASE primitive variant, flagged for owner decision, NOT designed
into the current API: calling the primitive with extra="leading" over
trials would need O = stacked U (b × T-family) — exactly the memory
bound the stack matvec exists to hold at ONE T.  The memory-safe form
is: scan body runs the primitive's stage-1 GEMM only (U_b × psi_v ->
r_b (k, t, M_loc, v_full), ~11.5 MB/trial), scan STACKS r_b; after the
scan, ONE stacked psum_scatter('y'), one batched left GEMM with
conj(psi_c), ONE stacked psum_scatter('x').  That preserves one-T-alive,
moves 2b collectives -> 2, and costs b×11.5 MB of stacked partials.  The
encode all_gathers admit the same treatment (stack X_b/R across trials).
API implication: expose the primitive's two stages
(`stage_right_local` + `finish_stacked`) or an
`extra="scan_stacked"` plan — owner review before any wiring.

Call-count arithmetic unchanged from the memo: ~4800 grouped
collectives/solve today; Option A alone re-lanes the large half of those
bytes onto the node-local axis; the two-phase form divides the count by b.

### 6.2 #2 bse_ring_comm._apply_W_from_T (TDA :345-360; non-TDA :585-600) — clean drop-in

These plain-jit einsum pairs ("kctM,bMNtsk->bcNsk" then
"kvsN,bcNsk->bcvk", partitioner-chosen collectives, c-replicated
intermediate by construction) already materialize the b-stacked T
(sh.T = (b, M_loc, N_loc, t, s, k)) — so here extra="leading" IS free:

    apply = contract_bands_block_reshard(mesh, extra="leading")
    #   O = T-derived U in layout (b, k, t, M, s, N) (same conv-layout
    #       note as 6.1), psi_left = psi_c_X, psi_right = transposed psi_v_Y
    #   out (b, nk, c_X, v_Y) -> local transpose -> sh.X (b, c, v, k)

This converts partitioner-chosen collectives into the structural stacked
psum_scatter pair with the right axis order (features 1-4 all gained) and
retires the pre-optimization pattern per bse_stack_matvec's own
retirement plan (:43-54).  Live callers to repoint afterwards:
bse_feast.estimate_spectral_bounds_sharded, bse_kpm.py:129,
absorption_haydock.py:208, bse_pseudopoles.py:234; non-TDA family
(bse_w_exact.py:102, bse_nontda.py:76) only after the B2 encode fix.

### 6.3 #3 vq_interp.make_eval_vq (:1024-1056) — NOT a contract_bands instance (honest verdict)

V = V_SR + conj(A_x) @ A_y.T is an OUTER PRODUCT of a thin (mu, nG)
factor producing the (mu_X, nu_Y) tile: the contracted axis (nG) is
replicated, the sharded axes are the OUTPUT axes — structurally the
transpose of this primitive (which contracts the sharded axes).  Forcing
it through contract_bands would be shape gymnastics with no movement win.
The right fix is a small structural sibling (shard_map + all_gather of
the thin factor per axis + local GEMM), plus hoisting the per-Q eager
hermitize (exciton_bands.py:630-661) — separate design, owner review.
The primitive's warm-up + axis-policy doctrines apply to that sibling.

### 6.4 #4 density snapshot (bse_ring_comm :892-916) — leave

Single-axis psum_scatter mirror, already the good pattern; the two-stage
primitive does not apply (one collective, one contraction).  No action.

### 6.5 Doc hygiene

bse_simple.py:27-30 stale claim ("uses the psum_scatter trick" — it does
not) — still flagged from the memo Sec. 6.2; fix belongs with the 6.2
retirement, not this workstream.

## 7. Files touched (worktree only, NOT committed)

- src/common/contract_bands.py — NEW: the primitive (+ warm-up + FFI gate).
- src/ffi/mklblas/cpp/gemm_batch_ffi.cc — NEW: MKL batched-GEMM handler.
- src/ffi/common/cpp/host/CMakeLists.txt — mklblas TU in the MKL block.
- src/ffi/common/ffi_loader.py — lorrax_mklblas_gemm_batch target entry.
- src/gw/ppm_tau_kernel.py — bodies subsumed; factory delegates to the
  primitive; cache keys + module header updated; channel-algebra doc
  consolidated into the factory docstring.
- src/gw/ppm_sigma.py, src/gw/ppm_accumulators.py — stale body-name
  docstring references repointed (docstring-only).
- tests/test_contract_bands.py — NEW: parity/HLO-pin/refusal/FFI gates.
- docs/dev/staged_reshard_primitive.md — NEW (owner-added deliverable):
  the API documentation for future resharding-primitive builders — API
  contract + refusals, the never-materialize-(m,n,k) doctrine, every
  special technique with its evidence pointer (AK.9, node-local-'y'
  policy, f64-split de-promotion + refutation, FFI GEMM gating/CUDA
  refusal/no-legal-alias note, warm-up contract, divisibility refusals,
  HLO-pin methodology, measured omega-order verdict incl.
  FFI-requires-leading), the future-change gating recipe (parity
  classes, colltable, AY.2 cache-cold rule), and the BSE adoption map
  incl. the two-phase-variant open API decision.
- wk_REL: cbands_gate.sbatch, cbands_ab.sbatch, cbands_ubench.py,
  gemm_pool_probe.py, contract_bands_notes.md (this file), + inner
  scripts the sbatches generate.
- Run dirs: mos2_4x4_test/run_CBANDS_* (one run per dir, J.7).

## 8. Named, not done

- BSE wiring (all of Sec. 6) — map only, per instruction.
- Two-phase primitive variant for scan-stacked decode collectives
  (6.1) — needs an owner API decision.
- L3 FFI direct-MPI reduce-scatter chain + L1 strip overlap — the
  primitive is the interface they slot behind (two-plans doctrine);
  untouched here.
- GPU-lattice check of the primitive (lowers by construction; measured
  on XLA:CPU only — claim scoped).
