# BSE ring/stack matvec — efficiency audit (JOINT FINDINGS)

3-agent audit of the inner machinery of LORRAX's BSE ring/stack matvec. Owner's
questions: are the matmuls contiguous; are the intermediate/persistent arrays laid
out optimally; are buffer donations optimal.

- **LAYOUT-AUDITOR** — array layouts, transposes, donations, dtype (+ synthesis owner).
- **TRACER** — all GPU measurement (Perlmutter A100, job 56010372, 4×A100, module-free
  srun+shifter, JAX x64). Dossier: `trace_dossier.md`; raw HLO/xprof under `raw_traces/`.
- **COMMS-AUDITOR** — collectives, overlap, topology.

Scope: `sources/lorrax_A` @ `agent/bse-phase2` HEAD `6ca714b`, read-only. Investigation
only — **no code changes**. Two measured regimes: **fixture** (MoS2 gnppm gate:
nc2/nv2/ns2/nk9/μ400 — latency regime) and **inflated** (nc48/nv48/ns2/nk16/μ800 —
compute/bandwidth regime).

---

## Executive summary

The BSE matvec is **never compute-bound** — it is HBM-bandwidth-bound at production
size (stack inflated 1×1: arithmetic intensity **0.039 FLOP/byte**, warm 15.1 ms ≈
2.5× the HBM-traffic floor) and collective-latency-bound at fixture size. So the levers
rank by **bytes moved and collectives launched**, not FLOPs. Two orthogonal top items on
two different matvecs:

- **Stack W-term (production):** the screened-direct term round-trips a 655 MB T-tensor
  ~6-8× through HBM; **two materialized full-T transposes** bracket the FFT (a structural
  k-batch↔k-minor reorder) = **15.2% of device wall at inflated** (36.7% at fixture, the
  #1 category). Halving the byte width (c64 where the existing fp32 option already
  applies) is the single biggest, lowest-risk knob; eliminating one transpose recovers
  ~7-8%.
- **Ring/full matvec (resolvent + spectral bounds):** the q=0 exchange is
  over-communicated (24 explicit collectives on the non-TDA resolvent SOLVE, 6 per
  `apply_V_ring`) where a stack-style batched exchange needs ~8.

Plus a clean load-time finding — the stack↔ring choice is a **measured memory-for-comm
trade with a crossover at nt≈2-3** — and three low-risk cleanups (hoist the recomputed
pair-amplitudes, drop cosmetic donations, an nt-aware dispatch).

---

## 1. Per-einsum layout table — STACK matvec hot path (`bse_stack_matvec._matvec`/`_w_stack`)

Persistent layouts (`bse_io` loaders + `make_bse_shardings`), all complex128:
`psi_c_X`(nk,c,ns,μ_loc) P(_,_,_,'x') / `psi_v_Y`(nk,v,ns,ν_loc) P(_,_,_,'y') → **k
outermost, μ/ν innermost**; `W_R`(μ_loc,ν_loc,kx,ky,kz) P('x','y',_,_,_) → **k-axes
minor**; `V_q0`(μ,ν) P('x','y'); `X`(nt,c_loc,v_loc,nk) P(_,'x','y',_) → **k innermost**.

| # | stage | einsum (src) | contracted | k role | layout / transpose |
|---|-------|--------------|-----------|--------|--------------------|
| E1/E4 | V pair-amp M_Y,M_X | `kcsm,kvsm->kcvm` (bse_serial:14) | s (spinor) | batch | ns∈{1,2} reduction, not GEMM. **Recomputed every matvec, X-independent → P3 hoist.** |
| E2 | V encode S | `kcvN,bcvk->bN` (:139) | k,c,v | summed | GSPMD → **2 all-reduce** on tiny S=`c128[8,400]`; no X-transpose materialized (M_Y sliced, not X all-gathered). |
| E3 | V solve U | `MN,bN->bM` (:141) | N | — | tiny (μ×ν), 1 all-reduce. |
| E5 | V decode VX | `kcvM,bM->bcvk` (:144) | M | free | broadcast over k; out matches sh.X. |
| D | D term | `(ε_c−ε_v)·X` (:134) | — | — | elementwise, layout-agnostic; k-inner matches X. |
| W1 | W encode R | `kvsN,cvk->cksN` (:97) | v | batch | after `all_gather(y,v)`; small (band). |
| W2 | W encode T | `kctM,cksN->MNtsk` (:99) | c | batch | **L2**: batched-GEMM emits k outermost; next reshape+cuFFT need k minor → **materialized full-T transpose** `transpose.196` dims{3,2,1,0}, 655 MB. |
| F | FFT conv | ifft·(W_R×)·fft axes(4,5,6) (:104-106) | — | — | k-axes minor-contiguous (correct for cuFFT); W_R broadcast-mult local. |
| W3 | W decode A | `kctM,MNtsk->cNsk` (:113) | t,M | batch | **L3** (mirror): cuFFT emits k minor, decode-GEMM wants k batch → **materialized full-T transpose** `transpose.197` dims{4,1,3,2,0}, 655 MB (fused w/ ortho ×0.25). |
| W4 | W decode WX | `kvsN,cNsk->cvk` (:117) | s,N | batch | out (c,v,k) k-inner = sh.X; `psum_scatter`×2 completes μ/ν, scatters c→x,v→y. |

**L2/L3 CONFIRMED (TRACER HLO, `stack_inflated_1x1_nt1.hlo`):** two real `transpose` ops
inside kInput fusions — **NOT free bitcasts** (a permutation `{4,1,3,2,0}` cannot be a
bitcast). Root cause exactly as predicted: the encode/decode are **batched-GEMMs over k**
(k as the outer batch dim of the dot output); **cuFFT requires k as the three minor axes**
→ the reorder is **structurally irreducible without changing the GEMM formulation**.
Persist per-rank at 2×2 (164 MB/rank each). GEMM operand layouts are uniformly row-major
`{3,2,1,0}` — **no surprise operand transposes** beyond these two, no col-major
mismatches, and the encode/decode chains are captured into CUDA command-buffers (graphs).
So the answer to "are the matmuls contiguous": **yes, the cuBLAS operands are clean; the
only layout cost is the two irreducible T-reorders around the FFT.**

## 1b. Persistent-array layouts — NO relayout needed
`psi`(k,band,ns,μ): dominant consumer = encode/decode GEMMs, which want k-outer-batch ✓,
band-contract ✓, μ-minor-free ✓. `W_R`(μ,ν,k): FFT wants k-minor ✓, multiply broadcasts ✓.
`V_q0`: tiny. Each persistent array is already optimal for its dominant consumer; a
load-time relayout buys nothing measurable. (`compute_pair_amplitude`'s spinor contraction
with μ-innermost is awkward but ns∈{1,2} makes it a cheap reduction — not a lever.)

---

## 2. The bandwidth thesis (the pivot — TRACER)

Stack inflated 1×1 nt1: `bytes_accessed=8.96 GB` vs `flops=349 MFLOP`. The W-term touches
the 655 MB T-tensor ~7×: encode-GEMM out → **L2 transpose** → IFFT (r+w) → W_R×T_R → FFT
(r+w) → **L3 transpose** → decode-GEMM read. Device-measured category split (14.07 ms/call,
no collective masking):

| category | % device wall | ms | note |
|----------|--------------:|---:|------|
| gemm | 35.3% | 4.97 | **bandwidth**-bound (349 MFLOP total): z884gemm reads/writes the 655 MB operands |
| fft | 28.4% | 4.00 | IFFT+FFT pair — algorithmically required |
| elementwise/fusion | 19.3% | 2.71 | W_R×T_R multiply + norms + V-term |
| **copy/transpose** | **17.0%** | **2.39** | **the two T-transposes = `input_transpose_fusion_2` 1.07 + `_3` 1.07 = 2.14 ms = 15.2% of wall** |

At fixture size copy/transpose is the **#1 category at 36.7%**. The two L2/L3 transposes are
~2 of ~7 full-T round-trips ≈ **19-28% of W-term HBM traffic** — the concrete
avoidable-bandwidth number.

---

## 3. Buffer donations — all cosmetic (LAYOUT L5, TRACER-confirmed)

`donate_argnums` aliases an input buffer to a jit **OUTPUT**, never to an internal
intermediate; an *unusable* donation declines-and-frees the input **without a copy**.
TRACER's decisive `copy(` grep of the ring HLO: the full-T buffer is **NEVER copied**.

| site | donated | jit outputs | verdict |
|------|---------|-------------|---------|
| `bse_ring_comm.py:361` `apply_W_from_T` | T | WX (diff shape) | declined → **no full-T copy**, cosmetic |
| `bse_lanczos.py:240` `solve_bse_sharded` | W_q [200,200,3,3,1] | replicated eigenpairs | same declined class (W_R=ifftn(W_q) is a fresh buffer, no aliasable output) → expected cosmetic; confirmed by the ring analog + mechanism |
| `absorption_haydock.py:221` | W_q | Haydock coeffs | same cosmetic no-op |

**Fix:** drop the unusable `donate_argnums` (one-liners; removes the recurring warning,
changes nothing measurable). **Separate minor finding (P-JIT):** the ring's
`encode_T_ring`(shard_map)→`apply_W_from_T`(separate jit) boundary emits ~5-11 small
(~1-5 MB/rank) layout-conversion copies — ~40× smaller than a full-T pass, and the stack's
single-scan W-term has **zero** such boundary copies (its encode+FFT+decode are one
shard_map). Ring-only, retirement-bound.

---

## 4. dtype — FLAG ONLY (no default change)

Everything is complex128 (`jax_enable_x64`). Since the entire W-term is HBM-bandwidth-bound,
a **complex64** matvec ~halves T (655→328 MB) and every one of its ~7 round-trips (incl. the
L2/L3 transposes, the FFTs, the pair-amplitudes) → ~2× on the whole W-term. The **2e-9
closure gate and eigenvalue solves must stay fp64**, but the **existing fp32-GMRES option**
(bse_feast; `sqrt_nk` already follows input dtype) means a c64 matvec is a drop-in for the
FEAST contour solves + spectral-bound Lanczos where fp32 is **already validated** — no *new*
accuracy risk there. TF32 is irrelevant at c128 and would break the 2e-9 gate at c64 → keep
OFF for closure runs. cuBLAS autotuning (`--xla_gpu_autotune_level`) worth an A/B on the
inflated batched GEMMs. All flag-only.

---

## 5. Collectives, overlap, memory-for-comm (COMMS-AUDITOR + TRACER)

**Collective inventory / matvec (2×2, measured):**
- **Stack (production):** V = **2 all-reduce** (src139/141, tiny tensors); W (scan body) =
  4 async/trial (2 all-gather + 2 reduce-scatter) → ~6 @nt1, ~34 @nt8. Count scales ×nt
  (the scan serializes trials).
- **Ring TDA (spectral bounds):** ~11 (8 ppermute + 1 all-reduce + 2 reduce-scatter),
  **fixed in nt** (trials batched on the T axis).
- **Full non-TDA (resolvent SOLVE, `screening=True`, include_W=False):** 4×`apply_V_ring` =
  **24 explicit/matvec** = PHASE2_LOG's ~20 ms SOLVE. A stack-style batched exchange = ~8.

**Overlap is ALREADY on (decisive).** TRACER: max concurrency **7-9** at every 2×2 config —
collectives on dedicated per-GPU NCCL streams, GEMM/FFT/transpose on compute streams, run
concurrently. The latency-hiding scheduler is ALREADY default-on and IS the source of that
overlap — **measured flag A/B (stack inflated 2×2 nt8): default 33.93 ms, `=true` 38.70 ms
(no-op), `=false` 43.90 ms (+29%)**. So "overlap already on" rests on a hard 29% floor, and
the while-loop barrier below is structural — no compiler flag crosses it. Yet collectives still eat
32-96% of device time because the cost model is **(collective COUNT) × (per-barrier
sync-wait)**, not β·bytes and not launch latency: the same all-gather measures GPU:0=621µs vs
GPU:3=3081µs (5× straggler spread on a ~6 KB message) — the "duration" is barrier-sync
surfacing **upstream compute imbalance**, not data movement. Two residual structural costs:
(1) sheer collective count × sync-wait; (2) the stack W-scan's `while`-loop iteration
**barrier** serializes the independent (carry=None) trials (~3 ms trial-to-trial spacing = one
collective-wait apiece; collective fraction falls 72.8%→32.1% nt1→nt8). **So the TOP latency
lever is COUNT-REDUCTION** — P-NT (ring: fixed ~11 at low nt) and C1 (exchange 24→8) both cut
*barriers*. Source-level bounded-unroll (P6) breaks the while-barrier to *pipeline* cross-trial
collectives but is narrow (below); the scheduler flag is not a lever.

**Memory-for-comm crossover (measured, inflated 2×2):** ring batches trials → collectives +
compute FIXED in nt, memory LINEAR; stack scans → memory FLAT (one T), collectives + compute
×nt. Ring nt1 = 12.35 ms / 347 MB **beats** stack nt1 = 18.80 ms / 819 MB; stack nt8 = 33.93
ms / 578 MB **beats** ring nt8 = 55.13 ms / 2647 MB (ring's linear T hits 10.5 GB @1×1 nt8 vs
stack's flat 1.36 GB — the concrete why-stack-exists). **Crossover ≈ nt 2-3 on BOTH axes →
empirical dispatch threshold nt≤2 → ring, nt≥3 → stack.** Honesty caveat:
the ring's low-nt *memory* edge is inflated-only (at the fixture, ring nt1 = stack nt1 =
183.9 MB, equal); what is **robust across sizes** is the ~1.5× low-nt *wall* edge and that the
ring is never larger at nt1.

**Topology / multi-node:** ppermute rings are topology-blind (each step crosses Slingshot
separately); NCCL all_gather/reduce-scatter do hierarchical intra-then-inter-node. **Design
lock:** keep the stack's collective formulation for scale-out; do not revive ppermute rings.
The k-replicated FFT (zero FFT comm) is a strong multi-node property.

---

## 6. Prioritized recommendations

Framing: both matvecs are memory/latency-bound, never compute-bound, so levers rank by
**bytes moved (bandwidth regime) and barriers launched (latency regime)** — and the top lever
depends on which regime dominates a given solve:
- **Bandwidth regime** (production CrI3-scale, large μ/nk, block solves): **P1 c64** (halves
  the whole HBM-bound W-term) and **P4 transpose-elim** are top.
- **Latency regime** (fixture/gate sizes, single-vector solves, the resolvent SOLVE):
  collective cost = count × sync-wait, so **count-reduction — P2 C1 (24→8) and P-NT** — is top.

These are orthogonal (two different matvecs / two regimes), not competing; P1 and P2 are
co-tops. The table numbers each but the regime note above governs which to reach for first.

| # | change | expected gain (evidence) | risk | effort | lands |
|---|--------|--------------------------|------|--------|-------|
| **P1** | **c64 mixed-precision matvec** (FLAG only) — extend the existing fp32-GMRES option to FEAST contour solves + spectral-bound Lanczos | **~2×** on the whole HBM-bound W-term (halves T + all ~7 round-trips incl. L2/L3 + pair-amps). Biggest single lever. | low WHERE fp32 already used; **2e-9/eigenvalue paths stay fp64** | med | resolvent-local + load-cast |
| **P2** | **C1: apply_V_ring rings→2 all_gathers** + local `compute_pair_amplitude` einsum (ring/resolvent) | resolvent 24→16 explicit near-term; 24→8 real fix (batched exchange, fixed in b). Also deletes the per-step `dynamic_slice(psi, origin·chunk)` band-rotation arithmetic; speeds spectral-bound Lanczos. | low (mirrors the proven stack V-term) | low near-term | solver-program (ring) |
| **P3** | **hoist M_X/M_Y pair-amps** — precompute once/solve, pass as matvec args | CONFIRMED recompute for **ALL** iterative solvers (Lanczos/Davidson/FEAST): the matvec is a per-iteration black-box jit with psi as arg, so XLA can't hoist the X-independent M across calls. = 2 GEMMs/matvec, **472 MB each @inflated** ≈ ~10% of matvec bytes. Zero comms. | low | low (signature change) | resolvent-local / caller-side |
| **P4** | **L2/L3: eliminate one W-term T-transpose** — k-minor (all_gather) encode, or fuse the FFT norm/scale to drop a pass | **15.2% of device wall @inflated** (2.14 ms; 36.7% @fixture). One → ~7-8%. | med (custom/strided FFT may not net; measure) | med-high | resolvent-local |
| **P-NT** | **nt-aware dispatch** — route `solve_bse_sharded` bs==1 (single-vector Lanczos) through the ring; **nt≤2 → ring, nt≥3 → stack** | Production repointed bs==1 onto the stack (bse_lanczos:159); at nt=1 the ring is **~1.5× faster** (12.35 vs 18.80 ms) and never larger. Also cuts barrier COUNT (ring fixed ~11 vs stack ×nt). Refines PHASE2_LOG deferred-#1's memory premise (inverted at low nt); does not overturn a validated call. Ring TDA is bit-exact to dense (2.3e-15) and already exists. | low | low (dispatch re-add) | solver-program |
| **P5** | **drop cosmetic donate_argnums** (bse_lanczos:240, haydock:221, ring:361/581) | CONFIRMED cosmetic — no full-buffer copy at any site. Silences the recurring warning. | none | trivial | load/solver-program |
| **P6** | **bounded-unroll the stack scan (=2-4)** — narrow mid-nt EXPERIMENT, not a recommendation | Breaks the while-loop iteration **barrier** to *pipeline* cross-trial collectives (trial i+1 encode issues while trial i decode drains) — its value is NOT "compute to hide comm" (within-trial overlap already on), it's barrier-removal. Real but narrow: gated 2× live-T → mid-nt/mid-mesh only, above the P-NT crossover; at nt≤2 use ring; at CrI3-prod nt8 collectives are only 32% (compute-dominated) so it buys little. | low if gated | low | experiment (memory-gated) |
| **P7** | **XLA flags** — cuBLAS autotune only | latency-hiding-scheduler is a **confirmed non-lever**: it is ALREADY default-on and is the *source* of the measured overlap — off = **+29%** (stack inflated 2×2 nt8: default 33.93 ms, off 43.90 ms; =true is a no-op, 38.70 ms). The residual comm cost is structural (count × sync-wait + while-barrier), removable only by source change (P-NT/P6). Autotune worth an A/B on inflated GEMMs; TF32 must stay OFF (breaks 2e-9 @c64). | low | trivial | run-config |
| P-JIT | fuse ring `encode_T_ring`+`apply_W_from_T` into one jit (ring-only) | removes ~5-11 small (~1-5 MB) boundary layout copies; stack already has none. | low | low | solver-program (ring) |

**Bytes-halving stack** (compound on the bandwidth-bound W-term): P1 c64 (×0.5 all) →
P4 transpose-elim (−7-15%) → P3 pair-amp hoist (−~10%). P1 is the single biggest,
lowest-risk knob where fp32 is already permitted.

---

## Disagreements / honest caveats (all reconciled)

- **P-NT vs ring retirement.** PHASE2_LOG deferred-#1 marks the ring retirement-bound on a
  *memory* argument ("stack bounds T strictly better"). TRACER's crossover (nt 2-3) shows that
  premise is inverted at low nt. All three agents agree P-NT **refines** the assumption (keep a
  small-nt path), it does not contradict a validated call — the stack is correct for block
  solvers (nt≥4). The *only* live regression is `solve_bse_sharded` bs==1;
  `estimate_spectral_bounds_sharded` and the resolvent already keep the ring.
- **P-NT vs P6 are not co-equal.** P-NT (use ring) is the low-nt fix; P6 (unroll) is a
  block-solve lever and would *worsen* the low-nt memory traffic, so they target different nt
  regimes, not the same problem.
- **L2/L3 irreducibility.** The two transposes are structural given batched-GEMM-over-k + cuFFT
  k-minor; P4 trades one transpose for a larger encode collective (or a fused-norm pass) — net
  win is regime-dependent (measure). It is not free elimination.
- **Donation mechanism.** LAYOUT predicted "unusable donation → no copy"; TRACER's HLO confirmed
  it (an earlier dossier draft wrongly called the ring donation a full-T copy; corrected). The
  ring boundary copies are small layout-conversions, a separate two-jit-split artifact.
- **c64 (P1)** gain is inferred from the measured AI=0.039 bandwidth-bound profile, not a c64
  A/B wall (not run — a nice-to-have measurement).

Artifacts: `trace_dossier.md` + `raw_traces/` (this dir). Source lines cite HEAD `6ca714b`.
