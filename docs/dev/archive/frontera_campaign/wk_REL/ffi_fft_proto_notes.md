# FFT-FFI prototype notes (2026-07-28) — MKL FFT (DFTI API) flat-k backend for the Σ τ path

Tree: /work2/08271/jackmc/frontera/lorrax @ b436e47, WORKING TREE ONLY (not committed).
Baseline for every gate: run_AQ_c4962_p64_mpi (sigma.exec 272.040 s, 176 τ @ 1.51 s).
Prior structural closure: wk_REL/docs/sigma_perf_results.md — G_ifft 79.6 + GW_mult_fft 95.7 +
V_ifft 16.5 = 191.9 s of the 295 s staged τ time is layout churn anchored in XLA:CPU's
fft custom-call needing the k axes minor-most, CLOSED as not-removable by any XLA-side
re-arrangement (fusion refuted with HLO evidence).  This prototype removes the anchor
itself: the FFT leaves XLA.

NAMING (owner rule): the backend is **MKL FFT driven through the DFTI descriptor API**
— a genuine O(N log N) fast Fourier transform at ANY k-count (mixed radix, arbitrary
lengths).  It is NOT a DFT-as-matmul; that formulation (candidates.json #2's "fold the
nk=16 DFT into a 16x16 matmul") is owner-vetoed and is not what this does.  "DFTI" is
Intel's name for the descriptor interface to its FFT engine, nothing else.

## Design (two-plan doctrine)

Default plan: the XLA path, byte-for-byte untouched (flag off = zero change).
Gated plan: `LORRAX_FFT_FFI=1` — `common/fft_helpers.py` (still THE single FFT entry
point, owner rule) returns an FFI-backed flat-k variant from `make_flat_k_fft`:
one `jax.ffi.ffi_call` per rank inside a `shard_map` over the FLAT (nk, *trail)
layout — the 3-D k-minor form is never created.  The handler
(`src/ffi/mklfft/cpp/fft_flat_k_ffi.cc`, targets `lorrax_mklfft_flat_k` /
`lorrax_mklfft_gw_conv`) reads the dot-layout tile in place via DFTI STRIDE
DESCRIPTORS: FFT-axis element strides {nky·nkz·T, nkz·T, T}, batch of T transforms at
DISTANCE 1 along the unit-stride trail.  Announce-or-refuse: an explicit flag with a
missing handler / non-CPU mesh RAISES with the `probe_target` reason (pattern #8), it
never silently runs the XLA path.

Second entry point: `LORRAX_FFT_FFI_FUSED=1` (read in `gw/ppm_tau_kernel.py`) routes
the τ kernel's step through `make_flat_k_gw_conv`:
`sigma_k = FFT[IFFT[G_k]·IFFT[W_q](bcast)·(-1/√Nk)]` in ONE host call, chunked
(per-thread ~4 MiB compact buffers) so the R-space G tile never materializes; the
multiplier is folded into the forward DFTI scale.

norm conventions: computed in fft_helpers to match jnp.fft exactly
(ifftn backward 1/N, ortho 1/√N, forward 1; fftn mirrored) and shipped as a plain
scale attribute — ONE source for the convention.  Values are ~1e-15 relative to
jnp.fft (different FFT engine), NOT bit-identical; every gate below is stated with
that ULP caveat.

Envelope honesty: every extent/stride from runtime buffer dims + kgrid attributes; the
batch is whatever trail is sharded onto the rank — no deck constants, no N_mu^2 global
tile on any rank (scaling target respected; fused V_R arena is (nk, mx_loc, my_loc) ≈
100 MB at this shape, i.e. the sharded W tile, and is malloc'd outside the XLA planner
— logged via LORRAX_MKLFFT_LOG, same class as the scalapack workspace).

Threading: the handler parallelizes its chunk loop with OpenMP
(`LORRAX_MKLFFT_THREADS` auto/off/N, strict grammar per the AW audit lesson) and pins
MKL to 1 thread per team member (blacs_grid.h MklThreadScope pattern, local
MPI-free copy).  Chunk size `LORRAX_MKLFFT_CHUNK` (default 16384 trail elements).
The owner's 6-way-cap question is answered by measurement (unit-gate sweep below).

Scope note for the A/B: `LORRAX_FFT_FFI=1` applies to every eligible flat-k call site,
so in the restart-gated deck the χ0/W-solve FFTs also ride the FFI backend
(screening re-runs before Σ).  Rows are separate; χ0's consumer is layout-agnostic so
little change is expected there — reported alongside, not conflated.

Files touched (working tree):
- src/ffi/mklfft/cpp/fft_flat_k_ffi.cc            NEW: the two handlers
- src/common/fft_helpers.py                       gated backend + fused factory
- src/gw/ppm_tau_kernel.py                        fused gate, cache keys, donation audit
- src/ffi/common/cpp/host/CMakeLists.txt          mklfft TU (MKL group; -fopenmp)
- src/ffi/common/ffi_loader.py                    host target table entries
- config/frontera/build_ffi_host.sh               export gate (WANT symbols)
Harness (wk_REL): fft_ffi_unit_gate.py, fftffi_build_gate.sbatch, fftffi_sigma_ab.sbatch.

## DONATION AUDIT (owner directive 2026-07-28 — reported distinctly from the kernel)

Inventory of every jit on the τ path (ppm_tau_kernel.py) and each operand's fate:

| jit | operands | donatable? | state before | state now |
|---|---|---|---|---|
| `_tau_kernel` (production top-level, 1/τ) | ψ×4, E_A, mask_A, B_q, Ω_q, mask_B, E_ref×2, t | **none** — every array arg is loop-invariant across the 176-τ loop (reused next τ); donation illegal | no donation (correct) | unchanged |
| `_build_W_t_q` (inner) | B_q, Ω_q, mask_B persist | none | no donation (correct) | unchanged |
| `_sigma_kij_kernel` (inner) | W_q dead after | W_q — but **INERT in production**: an inner jit traced into `_tau_kernel`; donation acts only at top-level dispatch (house fact: SPEEDUP_SCORECARD.md audit row (d), "donation is inert under the fused r-chunk jit") | `donate_argnums=(8,)` believed active | kept + documented as inert-in-production |
| staged `_G_ifft_j` | G_k (399 MB) dead after | yes | **undonated** | `donate_argnums=(0,)` |
| staged `_V_ifft_j` | W_q (100 MB) dead after | yes — the in-code comment even flagged "W_q is NOT donated here … one extra buffer" | **undonated** | `donate_argnums=(0,)` |
| staged `_mult_fft_j` | G_R (399 MB), V_R (100 MB) dead after | yes (G_R aliases the same-shape output) | **undonated** | `donate_argnums=(0, 1)` |
| staged `_conv_j` (new, fused) | G_k, W_q dead after | yes | n/a | `donate_argnums=(0, 1)` |
| staged `_project_j` | sigma_k (399 MB) dead after; ψ×2 loop-invariant | sigma_k only | **undonated** | `donate_argnums=(1,)` |

Reuse check (the "buffer reused ⇒ donation illegal" clause): in the staged dispatcher
G_k/W_q/G_R/V_R/sigma_k are each consumed by exactly one later stage and never touched
again; `sec.watch` block-until-ready runs in the PRODUCING stage's section, before the
donating consumer, so no post-donation use exists.  ψ/E/mask operands are
loop-invariant and are never donated anywhere.  `precompile_sigma` is unaffected: the
production `.lower()` path lowers the same jit objects; the staged prewarm executes
once on fresh dummies (ψ = the real, undonated arrays).

Production-path intermediates: at the `_tau_kernel` boundary there is nothing legal to
donate, so the 5 measured materializations of c128[16,2,624,2,624] + the 1.15 GiB
preallocated temp (archaeology below) are an XLA-liveness/layout-copy artifact inside
one module — unreachable by `donate_argnums` by construction.  The lever that DOES
reach them is the FFI in-place alias: both handlers declare
`input_output_aliases={0: 0}`, and when the operand is dead XLA grants the alias and
**the transform lands back in the input buffer — the terminal form of donation (zero
extra big tiles)**; the layout-copy materializations disappear with the layout
requirement itself.  In-place is a design requirement of this prototype wherever the
engine allows it, realized at the XLA-BUFFER level for every shape used.  At the DFTI
level the first engine literally ran DFTI_INPLACE on the strided tile (unit-gate round
1 exercised both branches), but measurement showed the strided↔strided form pays one
full-tile memory pass PER RADIX DIMENSION (2.8× single-thread cost), so the final
engine transforms each chunk through a per-thread ~512 KiB L2 buffer and writes the
result back into the very same aliased buffer region — still zero extra tiles, the
only per-rank scratch being the thread-local chunk buffers (+ the sharded-V arena in
the fused handler, logged).  When the operand is still live XLA inserts a protective
copy — expected only in eager diagnostic use.  Measured deltas: HLO copy/transpose
counts + memory-report temp in the A/B epilogue (results below).

## ARCHAEOLOGY (sigma buffer/donation/remat/temp record, as directed)

- **wk_REL/results/sigma_perf_candidates.json, candidate #2 evidence** — THE prior flag on Σ
  buffer behavior: "16 copy( + 7 transpose( ops in a 409-line module; preallocated
  temp **1.15 GiB (memory-usage-report), 5 materializations of c128[16,2,624,2,624]**";
  summary repeats "~3 more full-object transpose/copy passes forced by the fft<->dot
  layout alternation (1.15 GiB preallocated temp)".  Folded in: the A/B epilogue greps
  the fresh memory report; target is a large drop in temp + materializations.
- **src/gw/ppm_tau_kernel.py:330 (pre-change tree)** — in-code acknowledgment of the
  staged donation gap: "W_q is NOT donated here (the fused kernel donates it into the
  _V_ifftn temp); one extra (nq, μ_X, μ_Y)/rank buffer while the knob is on".  Closed
  by this audit (see table).  Note the belief encoded there ("the fused kernel donates
  it") is itself weakened by the inner-jit inertness fact below.
- **SPEEDUP_SCORECARD.md:3728, audit row (d)** — "3 new donations … all three run
  under the fused r-chunk jit, **where donation is inert anyway**": the house
  precedent used above to classify `_sigma_kij_kernel`'s `(8,)` as inert in
  production.
- **SPEEDUP_SCORECARD.md:2605 (T.4)** — donate_argnums + cache-hoisting precedent
  (`_reshard_z`), and **:799 (K-series)** — the bounded ζ back-solve routed to the
  `donate_argnums` variant; **:5546 / wk_AD/scorecard_AD.md:240** — the W-solve
  donation contract upstream of Σ ("A and B are always freshly built …
  solve_lu DONATES both operands, and V is still needed").
- **wk_REL/results/audit_findings.json #334-338** — χ0-donation comment pointing at a
  nonexistent call site (donation-contract doc hygiene; upstream of Σ).
- Honest scope: no record was found of a τ-path-specific donation writeup beyond the
  in-tree W_q comments; the "past agent flagged poor sigma donation" datum resolves to
  candidates.json #2's temp/materialization evidence plus the staged-path comment
  above.  Searched: lorrax_setup/{SPEEDUP_SCORECARD.md, SESSION_REPORT*.md, wk_*/}
  for donat/remat/buffer/temp/materiali (sigma-adjacent hits only are cited).

## Gates

1. **Build + numerical unit gate + thread sweep** (job 7878708, dev 1 node):
   handler vs jnp.fft at production LOCAL shapes (16,2,624,2,624)/(16,624,624),
   fused vs decomposed composition, odd shapes/all norms/ragged chunks, donation
   smoke.  Gate max rel err 1e-13 (expect ~1e-15).
   **RESULTS (7878708): UNIT GATE PASS, all cells.**
   - build-host: clean, both handlers exported, ScaLAPACK/BLACS DT_NEEDED intact.
   - G tile ifftn/fftn ortho: max rel err **0.0 (bit-identical to jnp at THIS
     kgrid** — 4·4·1 is pure radix-2/4 and the 0.25 ortho scale is exact; the
     general claim stays "value-identical ~1e-15", per the odd shapes below).
   - fused gw_conv vs decomposed composition: 2.8e-16 (scale-fold rounding, as
     designed).  Round trip 1.4e-16.  Donation smoke (jit donate_argnums →
     DFTI_INPLACE branch): PASS, both handlers.
   - odd (3,2,1)×(5,7,3), all four norms, ragged chunk=13: 1.1e-16 … 4.0e-16.
   - THREAD SWEEP (round 1, strided→strided engine), G-tile ifft / gw_conv ms:
     1thr 349.6/700.9 · 2 247.7/424.7 · 4 198.6/285.5 · 6 180.2/240.0 ·
     8 173.0/215.6 · 14 163.7/191.2 · 28 161.7/182.1; XLA standalone-jit
     reference 125.7 (ifft incl. its transposes) / 1145.5 (3-FFT staged
     composition).  → The FUSED conv is 6.3× the XLA composition, but the
     PLAIN strided→strided form was SLOWER than XLA's transpose+DUCC: the
     4×4×1 FFT is computed dimension-by-dimension, so every radix pass
     streamed the full 400 MB tile.  FIXED before the A/B: the plain engine
     now transforms strided→per-thread-compact (L2-resident radix passes) +
     one contiguous scatter-copy out — the same shape that makes the conv
     fast — and the default chunk was resized to 512 KiB/thread (L2), env
     LORRAX_MKLFFT_CHUNK.  Re-gate: job 7878719 (results below).
   - 6-way-cap answer, round 1: throughput rises monotonically to 28 threads
     (no BLACS-style oversubscription cliff — this is streaming, not
     latency-bound collectives); the knee is at ~8-14 threads.  Final policy
     set after the re-gate.

   **RE-GATE (job 7878719, compact-chunk engine): build=0 gate=0, all 15
   correctness cells PASS (same 0.0 / ~1e-16 errors).**
   - Thread sweep (G ifft / gw_conv ms): 1t 366/687 · 2 254/427 · 4 198/286 ·
     6 179/238 · 8 170/214 · 14 159/184 · 28 **151/163**; XLA references
     127.8 / 1141.4.
   - Chunk sweep at 28t: 512→16384 spans 150.4-162.5 (ifft) / 162.5-184.5
     (conv); **auto (512 KiB/L2 ⇒ 2048 at nk=16) ≈ best** — keep auto.
   - **6-way-cap ANSWER: no cap.**  Scaling is monotone to the full 28-core
     team (a 6-cap costs +19% on the plain transform and +46% on the fused
     conv); policy `LORRAX_MKLFFT_THREADS=auto` = ambient omp max.  The
     AW-style catastrophic oversubscription cliff does not exist here —
     streaming FFT work, no latency-bound collectives inside the handler.
   - Standalone interpretation (single process, eager-jit dispatch):
     * FUSED conv = **163 ms vs 1141 ms** for the XLA-equivalent staged
       composition (7.0×) — the fused entry point is the decisive lever.
     * PLAIN ifft = 151 ms vs 128 ms XLA standalone: ONE strided pass over
       the 400 MB tile costs ~70 ms in either engine (the 16 k-lines are
       24.9 MB apart), and plain = read-pass + write-pass.  The compact
       rewrite moved the radix passes into L2 (conv gained 182→163) but the
       two streaming passes bound the plain form.  Whether the production
       module (no per-call buffer churn, no jit-boundary materialization)
       shows the same ordering is exactly what A/B passes a/b measure.
2. **Σ-only restart-gated A/B** at nb=128 vs run_AQ_c4962_p64_mpi
   (fftffi_sigma_ab.sbatch, job 7878727; AC.4 harness; parity tol 1e-12 on
   sigma_diag/eqp0 with the ULP caveat — exact-0 is NOT expected, unlike
   pure-movement rounds; at THIS kgrid the plain transforms measured
   bit-identical, the fused scale-fold did not): passes
   a (prod FFI) / b (staged FFI) / c (prod FFI+fused) / d (staged fused), per-τ
   sub-rows vs the r1 pass-B table (G_ifft 79.63 / V_ifft 16.54 / GW_mult_fft 95.74),
   tau-kernel HLO transpose/fft/copy counts, memory-report temp (baseline
   module_0912: Total 1.45 GiB, preallocated-temp 1.15 GiB).
   Scope note: LORRAX_FFT_FFI=1 also routes the χ0/W flat-k FFTs (screening
   re-runs in the restart deck) — chi/W rows reported separately.
   FFT_THREADS=auto (28), chunk auto.

   **RESULTS (job 7878727): ALL FOUR PASSES rc=0, ALL FOUR PARITY PASS.**

   sigma.exec (production):
   | run | sigma.exec | vs 272.040 baseline |
   |---|---|---|
   | AQ baseline (XLA) | 272.040 | — |
   | r1 prod (XLA, instrumented tree) | 273.278 | 1.00× |
   | **a: FFI decomposed** | **77.646** | **3.50×** |
   | **c: FFI fused gw_conv** | **71.906** | **3.78×** |
   (staged passes, not wall-comparable: b 75.736, d 69.367; pass walls for
   the whole restart-gated run: 246/184/178/167 s.)

   Per-τ staged sub-rows, 176-τ totals (r1 pass-B XLA reference → pass b FFI):
   | row | XLA (r1) | FFI (b) | ratio |
   |---|---|---|---|
   | w_phase | 5.70 | 5.65 | — |
   | G_build | 11.17 | 11.15 | — |
   | **G_ifft** | **79.63** | **2.36** | **33.7×** |
   | **V_ifft** | **16.54** | **2.78** | **5.9×** |
   | **GW_mult_fft** | **95.74** | **5.71** | **16.8×** |
   | project_rs | 84.75 (r1) / 47.6 (r2 stacking) | 43.14 | — |
   | staged dispatch total | 295.0 | 71.64 | 4.1× |
   Pass d (fused): the trio collapses further into ONE row
   `sigma.tau.GW_conv_ffi` = **4.99 s** (vs 10.85 decomposed FFI, vs 191.9
   XLA — **38.5×**); staged total 65.73.  The τ-kernel FFT/layout cost at
   nb=128/μ=4962/P=64 is CLOSED: 65% of the τ wall → ~7% (b) / ~5% (d).
   The Σ wall is now the ψ-projection: project_rs 43.2 s = 66% of pass-d
   dispatch — the OWNER-HELD re/im channel lever (perf-log prediction
   confirmed at nb=128, not just nb=256).

   Tau-kernel HLO (module_0551, passes a and c):
   - xla fft ops **3 → 0**; custom-call lorrax_mklfft ×3 (a) / ×1 (c);
   - transposes **6 → 0** (large transposes 0; the refuted XLA fusion had 20);
   - copies 20 (a) / 18 (c) — small tiles only; reduce-scatter 2, all-gather 0.
   Memory (buffer-assignment / memory-usage-report, same module):
   - baseline: Total 1.45 GiB, preallocated-temp 1.15 GiB;
   - a: Total 1.54 GiB, temp **1.24 GiB (+90 MiB — honest regression**: the
     decomposed path materializes the broadcast V_R as its own buffer where
     the XLA path fused it into the multiply);
   - c: Total 1.45 GiB, temp 1.15 GiB — **equal to baseline**; peak live
     1.257 GB both.  The hoped-for temp SHRINK did not materialize: XLA
     still reserves ~3 tile-lifetimes across the kernel (G build output,
     conv in-place alias, projection stacks) — the FFI aliases prevented
     growth, they did not shrink the arena.  Time, not memory, is the win.
   χ0 bonus (same flag, screening re-ran on FFI): chi.exec **7.76 → 2.52 s**
   (3.1×), W.exec 13.0 → 14.0 (noise) — chi0's dot→fft input-side
   transposes were also real, its layout-agnostic consumer only spared the
   output side.

   Parity: sigma_diag/eqp0/eqp1 max|diff| = **0.000e+00** on all four passes
   — BUT those are 9-decimal text files (~5e-10 eV resolution), so this
   line alone cannot distinguish "bit-identical" from "identical to text
   precision".  The h5 tensor gate below answers it.

   Threading answer (the 6-way-cap question, FINAL): the handler's OpenMP
   chunk team ran at `auto` = the ambient 28 threads/rank (unit-gate log
   line `threads=28`; MKL pinned to 1 inside each member, AW pattern).  In
   production the engine hits real streaming bandwidth: G_ifft 2.36 s/176 τ
   = **13.4 ms per 400 MB tile ≈ 60 GB/s/rank (~120 GB/s/node with 2
   ranks)** — the unit gate's standalone 151 ms was dispatch/alloc-bound,
   not engine-bound.  NO cap: scaling is monotone to 28 (a 6-cap costs
   +19%/+46% standalone); the AW oversubscription cliff does not exist here
   (streaming FFT work, no latency-bound collectives in the handler).

   Fused-multiply verdict: **it paid** — −5.74 s on sigma.exec (77.6 →
   71.9), trio 10.85 → 4.99 s staged, and it restores the baseline memory
   arena (temp 1.24 → 1.15 GiB).  Worth keeping as the recommended FFI
   configuration (both flags on), with the decomposed path retained for
   row-level observability.

3. **sigma_mnk.h5 tensor gate** (job 7878845, wk_REL/probes/h5_sigma_compare.py —
   the .dat parity lines are 9-decimal text and cannot distinguish
   bit-exact from value-level): **PASS, tol 1e-12, all datasets, all four
   passes.**  Worst dataset `sigma_c_kij_ev`: a/b max|diff| 2.414e-14 eV
   (maxrel 3.271e-15), c/d 2.541e-14 eV (3.443e-15).  So the exact-0 .dat
   parity is a TEXT-PRECISION artifact (real differences ~2.5e-14 eV sit
   far below the ~5e-10 eV text resolution): the runs are value-identical
   at the engine-swap level, NOT bit-identical — even though the plain
   transforms measured bit-equal on raw tiles at this kgrid, the
   end-to-end pipeline (FFI χ0/W upstream, scale placement, fused
   scale-fold in c/d) differs at the few-ULP level, exactly as the design
   predicted.  Note a≡b and c≡d produce identical diffs (prod vs staged of
   the same engine reproduce each other bitwise, as in previous rounds).

## Verdict

**The prototype works and the claimed mechanism is confirmed end-to-end.**
Replacing XLA:CPU's minor-most-anchored fft custom-call with MKL FFT (DFTI
API) stride-descriptor handlers reading the dot-layout tile in place:

- kills the layout transposes outright (tau-kernel HLO: 6 → 0 transposes,
  3 → 0 xla fft ops) — the structural closure in sigma_perf_results.md is
  now bypassed rather than contradicted: the anchor left with the custom-call;
- collapses the FFT trio 191.9 → 4.99 s (fused) / 10.85 s (decomposed) of
  staged τ time — the 65% layout share of the τ wall is gone;
- **sigma.exec 272.0 → 71.9 s (3.78×)** at AQ 4962c/P=64, parity-gated at
  1e-12 on both the text outputs AND the full sigma_mnk.h5 tensors
  (measured 2.5e-14 eV); χ0 gains 3.1× on the same flag for free;
- the fused G·W entry point pays (−5.7 s + baseline-equal memory arena) —
  recommended config is BOTH flags on; decomposed mode kept for rows;
- threading: auto (full 28-thread team, MKL pinned to 1 inside), ~60
  GB/s/rank in production; NO 6-way cap — the AW cliff does not apply;
- memory: honest neutral — temp arena 1.15 GiB (c) unchanged vs baseline,
  +90 MiB (a); the aliasing prevents growth, it does not shrink the arena.

Measured domain: MoS2 4×4, nb=128, μ_pad=4992, 8×8 mesh, XLA:CPU, MKL
2020.1, kgrid (4,4,1).  The Σ wall is now the ψ-projection
(project_rs 43.2 s = 66% of the staged τ dispatch) — the owner-held re/im
channel derivation (OWNER_DECISIONS #1) is the next real lever, exactly as
the nb-scaling analysis predicted.  Risks + follow-ups: see final report;
NOT committed (working tree only), per instruction.
