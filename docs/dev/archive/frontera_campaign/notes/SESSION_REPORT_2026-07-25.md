# LORRAX CPU-at-scale campaign — consolidated report (2026-07-25)

## ⭐ HEADLINE (final): trustworthy 12×12 physics achieved
**MoS₂ 12×12 G0W0 on 40 CPU nodes: DFT gap 1.7010 eV → QP gap +2.6475 eV at
606 centroids** (GW correction +0.947 eV) — the first physically-sane eqp this
deck has ever produced, after the campaign found and fixed, in order: three
launch/memory bugs, an unsharded 271 GB intermediate, an I/O-seam zero-copy
abort, an unconverged-quadrature V_H (replaced by an exact route validated to
0.0001 eV against QE), a V_H double-count in the post-hoc CLI (caught by the
new sanity guards on their first outing), and — the deep one — **the nosym-deck
ψ*(−r) corruption in the eager WFN unfold** (every historical 12×12 artifact
was affected; every norm-based check was blind to it by symmetry). The
276c/1194c/1998c gap rebuilds (μ-convergence table) are the last rows pending.

Branch: `fix/zq-band-gather-device-invariance` @ `0f9e4dc` (not pushed).
Full handoff: `lorrax/docs/dev/HANDOFF_cpu_frontera_2026-07.md` · Ops playbook:
`$WORK/LORRAX_FRONTERA_ADVICE.md` · Working log: `SPEEDUP_SCORECARD.md`.

## 1. What landed (three refactor waves, all bit-exact-gated)

### Wave 1 — CPU enablement & correctness (merged @ 971baef)
| win | measured effect |
|---|---|
| z_q band-gather device-invariance bug fixed (+ regression test) | wrong-at-P>1 → invariant <1e-9 |
| phdf5_host CPU WFN read (shared unfold kernel, zero build) | htransform 23→16 s; bit-exact |
| Mesh-aware htransform loader (ψ band-sharded, not replicated) | −O(nb·ngk) replicated bytes/rank |
| §5b process-local WFN load | 40–80 node runs unlocked (79 MB/rank vs 151 GiB wall) |
| Band-chunk uniform pad (compile fix) + latent UH_bc dim bug | −23% Galerkin on non-uniform chunks |
| phdf5 FFI shared-core CPU port (3 CUDA seams switched, one MPI-IO core) | CPU FFI read bit-exact; host lib builds CUDA-free |
| Exciton: rank-parallel mini-BZ head (GW-single-sourced) | rank-count-invariant, 1/P work |
| Exciton: build_cq face-sharded | −13.4 GB/proc host gather |
| CPU-safe linalg auto-routes + CLI bootstrap + gated GPU fallback | 9 CLIs run on CPU nodes |

### Wave 2 — maintainability & unification audit (merged @ 708c9d1)
- **F (wfn-read):** htransform's read was duplicative (n_bc+1 loads → 1; unfold
  halved; ~10% wall at fixture scale, grows with n_bc). Verdict AGAINST adopting
  gwjax's async/PsiGStore machinery — measured overlap is 0.000 (AsyncWfnReader
  is production-dead code). Triplicated load-dance unified into
  `load_psi_gflat_padded`; psp reader dedup.
- **G (infra):** loader's 3 host build paths → one `_kplan`+assemble scaffold
  (~90 dup lines gone, §5b re-lowering wart fixed); `runtime.bootstrap()` 2-line
  header across 9 CLIs; public FFI capability probe `has_target()`; isdf resolver
  ladder consolidated. Verdict: unfold algebra + FFI platform tables were
  ALREADY well-factored; duplication was in scaffolding, now gone.

### Wave 3 — the distributed-linalg facade (merged @ 0f9e4dc)
`src/ffi/linalg/`: `resolve_backend(op, requested, mesh)` — vocabulary →
platform → compiled-capability (`has_target`) → process-coverage → geometry
(square-mesh deadlock guard) → divisibility, ALL at resolve time with uniform
errors; `list_backends()` introspection; `backend_module()` single import seam;
native impls first-class. New input key `eigh_backend` (CLI now an override).
Docs: `docs/dev/linalg_ffi.md` written standalone-library-grade.
Gates: route pins 7-passed, worker_cap, loader suite 15-passed, htransform
bit-exact (incl. input-key path), gw eqp max|Δ|=1e-6 eV.

## 2. Production timing — MoS2 12×12 on 40 CPU nodes
- WFN: `/scratch2/.../mos2_80ry_12x12/WFN.h5` (15.65 GB, nosym → nq=144 full-BZ,
  mnband=400, ngkmax=8603, FFT 36×36×135, SOC spinors).
- Memory model CALIBRATED: real peak ≈ **2.02×** planner HWM (from MaxRSS of the
  cancelled-not-OOMed 40-node attempt). Where the 2.02× lives: AUDIT PENDING (J).
- run1 (276c, nband=160, 8×10 mesh) — a productive failure ladder:
  - v1 (7874158): died 2 min — FATAL bogus XLA flag
    (`--intra_op_parallelism_threads` doesn't exist in this jaxlib; F-aborts).
    ADVICE §10 corrected: `taskset` is the real thread mechanism.
  - v2 (7874236): mesh up, plan + physics route correct, then
    **RESOURCE_EXHAUSTED: 271 GB single allocation** inside the ζ-fit pair
    pipeline (`z_q_from_psi_sm` → `_pair_pipeline_sm`) — a ψ(r)-chunk-shaped
    intermediate (nk·bc·ns·r_chunk ≈ 129 GB ×~2) materializing UNSHARDED when
    the P=80 planner picks one full-r chunk. **The campaign's first
    empirically-caught lim P→∞ replication bug** — invisible at P=4 (24×
    smaller r_chunks) and very plausibly the origin of the 2.02×
    model-vs-reality memory gap. Forensics handed to audit J (fix in flight).
  - v3 (7874242): **ζ-FIT COMPLETED — 1692 s at P=80** (first-ever multi-node
    12×12 completion; vs 3503 s at P=4 → 2.07× — sublinear on ib0 as expected).
    MaxRSS 39.0 GB/rank — confirms J's 2-slot Stage-C model (31.7 GB + stages).
    zeta_q.h5 written clean (5.49 GB). THEN: all 80 ranks SIGABRT'd entering the
    screening stage — node0 memory spiking 22→53 GB in the last 30 s sample —
    **J's "unmodelled Stage F" manifesting at only 276c**.
  - v4 root-cause (J): NOT an OOM — an XLA raw-buffer CHECK (zero-copy
    device_put of a dying host buffer) at **V_q entry**: the H5PY_ALLGATHER
    read path pulls the whole (nq,μ,ngkmax) ζ onto EVERY rank ×2 (12.69 GB/rank
    vs 79 MB sharded — 160× the modelled term). FIXED: per-rank hyperslab
    sharded read + block_until_ready; gated P=4/P=8 (eqp 1e-6, ht bit-exact) +
    unit tests across all partition specs. Merged; v5 resubmitted (full refit —
    restart=true can't skip ζ because it needs the tensors file the crashed
    stage writes; ζ-reuse-from-clean-zeta_q.h5 flagged as a cheap feature).
  - Next wall pre-computed by the new Stage-F model term: the restart-tensor
    WRITE gather is still replicated (27 GB @μ=2412, 461 GB @10k) — cure is the
    env fix (mpi4py + parallel h5py → PHDF5_HOST) or a sharded write.
  - **CONDITIONING DATUM (v5 telemetry):** n_keep = **276/276 on every q** —
    zero truncation at μ=276 (rcond 1e-8); the pair-density space is nowhere
    near over-complete at this μ. The 1998c run (workstream K) supplies the
    next point on the curve.
  - v6 (restart, σ-window 70→80): completed in 14m22s but **QP gap −135 eV =
    GARBAGE — restart with a CHANGED band window reuses tensors built under
    the old window (silent misindexing)**. Lesson: restart requires identical
    windows; a loud attrs-guard in isdf_tensors is queued with J. v7
    (7874380, full rerun, consistent window 80) in flight.
  - Σ-stage divisibility: window must be a multiple of lcm(p_x,p_y)=40 at
    8×10; proper zero-pad fix (meta.py) in progress (J).
  - **v5 (7874338, merged fixes): CLEARED the wall end-to-end.** ζ-fit
    **1324 s (−21.7% vs v3** — the identity-take elision was costing ~21% of
    ζ-fit in pure memory traffic); V_q all 144 q in ~2 min (the exact all-rank
    SIGABRT point before); `isdf_tensors_276.h5` written (379 MB) → **post-ISDF
    restart iteration now possible (~minutes instead of ~30 min)**. Memory
    model-vs-real: **1.22×** (was 2.02×), model now names its own binder.
    Node mem peaked 36 GB and FELL BACK (was 22→53-climbing-to-abort).
    Screening cleared cleanly (~9 GB/rank, matching audit predictions); then
    died at Σ entry on the **Phase-1 divisibility guard working as designed**:
    σ window 70 (26+44) doesn't divide the 8×10 mesh (needs multiple of 40) —
    pre-hardening this was a silent-wrong-results case. Workaround: ncond 44→54
    (window 80) + **restart from isdf_tensors (first use of the restart
    workflow — jumps straight to screening/Σ)** as v6 (7874375). Proper fix
    (pad the σ window in meta.py, zero-pad rows stripped from outputs) queued
    with J, gated.
- run2 (1194c, 3 r-chunks, ~71 GB/rank = 78% envelope): staged, submits after
  run1 validates. **PENDING**
- Free wins found in prep: nband 120→160 costs nothing (same 80-pad);
  `band_chunk_size=16` default silently double-chunks at P=80 (pinned to 160).

## 3. The scale ladder (2k → 10k+ centroids) — the wall map (STAGED, ready to run)
Owner directive: never replicate any N_μ-dimensioned object avoidably; find the
lim P→∞ bottlenecks; probe conditioning toward 10k+ centroids.
**Verdict from the calibrated analysis: beyond μ ≈ 3000–4000 there is NO correct
configuration today — not slow, none.** The ranked walls:

| # | wall | scaling | binds at | fix |
|---|---|---|---|---|
| **0** | ζ-fit Stage-C arena (`fit_one_rchunk` pair pipeline) **sharded ~2×, not ~P** — the 271 GB OOM ask is 1.90× smaller than fully-unsharded (516 GB); planner assumed /P (9.9× off) | O(nk·ns²·μ·cr / 2) | **μ≈4000** even at max chunking (81); already bit at 276c/1-chunk | shard_map specs fix at isdf/core.py:2203 — **with J now** |
| 1a | charge replication-cap gate MIS-SPECIFIED: tests the whole (nq,μ,μ) stack but the factorization already q-batches (true transient flat ~12-13 GB, μ-independent) | gate artifact | refuses at μ>1365 needlessly | test one batch, not the stack — cheap |
| 1b | redundant O(nq·μ³) eigh on EVERY rank in replicated_rank_truncate | compute | 41 min @2k, 5.5 h @4k, 86 h @10k | distributed_rank_truncate on ffi/linalg (SLATE/ScaLAPACK — cusolvermp deadlocks on rectangular 8×10) — design with J |
| 2 | ζ-writer h5py_allgather, O(μ) on ONE rank | memory | μ≈4650 | serialized per-rank hyperslab writes (pure h5py), or port write_ffi.cc |

- **Chunking strategy REVERSED** by wall #0: "few large chunks" was wrong; all
  rungs re-pinned so 9.9×model ≤ ~40 GB (run1→9 chunks, run2/3→27, run4/5/6→81).
- **Staged**: kmeans `centroids_frac_1998.txt` DONE (0.1% pad @P=80; ~2400 set
  still generating); rungs run3_c1194_b400 / run4_c2000_b160 / run5_c2000_b400 /
  run6_c2400_b160 (cap-raised controls, dev queue) + RUNBOOK.md (dev-QOS
  choreography, go/no-go gates, wall table to 10k) + re-runnable
  preflight_ladder.py/stage_rungs.py. The kmeans "mangling bug" was argparse
  prefix-matching (`--out` doesn't exist → ate `--out-suffix`); pass neither.
- **Conditioning observability gap:** the rank truncation's `keep` count prints
  nothing from jit — one `jax.debug.print(keep.sum)` makes the ladder's central
  question (dropped-mode growth vs μ) measurable. Flagged to J.
- **Bands:** 400 = this WFN's hard ceiling. Pseudobands is NOT wired into GW
  (library-only, hook unpopulated, nk>1 ingest broken). Recommendation: QE
  regeneration (nbnd≈1600, ~63 GB WFN, -ndiag 1, regen dipole/kin_ion) — but
  only AFTER walls #0/#1 are fixed (nband=1600 wants μ≫3000).
- Bonus launch finding: the container's default JAX AOT cache is
  machine-mismatched (SIGILL risk) — `ISDF_JAX_CACHE_DIR=""` stays mandatory.

## 4. μ-replication audit (lim P→∞) — DONE (J); 5 fixes merged
Full object-by-object table in J's report (scorecard "J"); the essentials:

**The 271 GB OOM, root-caused to 0.03%:** not shard specs — the band
`all_gather` at `z_q_from_psi_sm` (isdf/core.py:685) gathers `(nk,bc,ns,cr)`
UNSHARDED (129 GB at full r), and the y-compaction `jnp.take` (from the original
invariance fix) adds an identical 129 GB copy XLA can't fold for traced indices
(+12.9 GB carries = 270.89 GB vs 270.98 observed). The term is μ-INDEPENDENT
(bc×cr) — why it bit at just 276 centroids.

**Implemented + gated bit-exact at P=4 AND P=8 (gw eqp 1e-6 eV; ht bit-exact):**
1. Identity-take elision → Stage C 271→142 GB (full-r), 31.7→17.4 GB (9-chunk).
2. Planner now models the gathered-ψ term (27.4→285.4 GB at the OOM config; picks
   4-5 chunks — the failure mode cannot recur).
3. htransform `fH_R_rep` de-replication: **−51 GB/rank** (+−11.4 GB temp).
4. Replication-cap gate fixed (tests one q-batch, not the stack — the true
   transient is flat ~12-13 GB μ-independent): rank_truncate now reachable at
   full-BZ 12×12 with no cap surgery, no resolving-route changes.
5. Truncation observability: `n_keep/q, λ_max/q, λ_min(kept)/q` logged — the
   ladder's central conditioning signal (verified live).

**The 2.02× memory mystery: RESOLVED.** (a) the planner undercounted its own
binding Stage-C transient by up to 10.4× (fixed); (b) the model stops at Stage E
— χ₀/W/PPM/Σ are unmodelled, including a REPLICATED W (`w_isdf.py:283`,
break-μ≈4.4k, one-line fix candidate) and an un-jitted PPM fit holding ~15
concurrent (nq,μ,μ) eager arrays. Discriminator staged: LORRAX_EXIT_AFTER_ZETA
MaxRSS comparison.

**Structural designs ready to implement (next wave):**
- ψ-gather → all-to-all('y') + all-gather('x'): 129→12.9 GB/rank, bit-exact by
  construction (band order preserved); removes wall #0 entirely.
- Distributed truncated pinv: per-q SLATE/ScaLAPACK eigh at P('x','y'),
  replicated (μ,) spectrum → local identical truncation → column-scale (never
  materialize B replicated) + SUMMA back-solve replacing the per-r-chunk factor
  all-gather (230 GB comm/r-chunk @10k). O(nq·μ³)→O(nq·μ³/P): ~86h→~1h @10k.
  PREREQ: host lib rebuild with SLATE eigh (current wtA lib has ONLY phdf5
  symbols — no distributed eigh is compiled today; resolver correctly refuses).
- ζ-writer: cheapest fix is ENVIRONMENT (mpi4py + parallel h5py makes the
  already-written PHDF5_HOST path reachable — no code change), plus a byte-cap
  guard and per-q streaming fallback in `_to_host`.

**Conditioning for 10k:** knobs enumerated (zeta_rcond=1e-8 env-only — should
become an input key; ridge floors; two unlogged np.linalg.pinv in vq_interp).
Leading indicator: the moment n_keep < n_log per q (pair-density rank
over-completion) — now visible thanks to fix 5.

## 5. Test-suite health — DONE (H): all three suites GREEN on CPU
- `test_zeta_mesh_invariance.py`: **4/4 PASSED (16m13s)** — first-ever full
  completion on this branch; the backend-aware cap fix confirmed.
- `test_wfn_loader_eager.py`: 15 passed / 1 expected-skip.
- `wfn_loader_backend_parity_test.py`: **ALL PASS truly multi-process for the
  first time** (world=4, eager vs phdf5 FFI: max|Δ|=0.0, gvecs exact). One
  test-side fix (process_allgather tiled=True) — merged.
- Compile-cache cold measurement: 122 compiles = **only 6.8 s pure compile** at
  fixture scale — the storm's production cost is tracing/dispatch ×
  rank-replication, not compilation; warm-hit measurement PENDING. Note:
  htransform's CLI never calls ensure_jax_compile_cache (only gw_jax does).
- Top time sinks at fixture scale: (1) bring-up/import/dispatch (>100 s of a
  121 s wall vs ~4 s numerics); (2) rank-replicated compiles (invariant
  ~122-138/rank); (3) jax.distributed startup fragility under node sharing
  (DEADLINE_EXCEEDED; mitigation = node pinning/stagger or longer timeout).
- H's end-to-end merged-branch gates were stranded by holder expiries — but
  superseded: the merged branch was gated repeatedly afterward (post-merge
  bit-exact at 708c9d1; J's P=4/P=8 gates at 9c2d551/658b0de; production v5/v6).

## 5b. ROOT-CAUSED + GUARDED (numerical fix IN PROGRESS, workstream N) —
## the "garbage eqp" was never a parallel bug: unconverged ISDF V_H in a
## catastrophic cancellation (present in ALL historical 12×12 runs)
STATUS (updated after workstream N landed): the exact-route V_H is
**implemented, merged, default-on** (full deck inheritance, provenance attrs,
no-double-count seam at SigmaResult entry so the SC loop is covered). N's
error DECOMPOSITION — both prior hypotheses disconfirmed with data:
- **Convention: CORRECT.** QE ran assume_isolated='2D' → the 2D-truncated
  Coulomb is right (full-3D is 3× WORSE: 115.7 vs 38.7 eV rms). The convention
  matters enormously (½∫ρV_H: 437.8 vs 244.0 Ry) but was not wrong.
- **ISDF quadrature error: real (82.6 eV @276c/160b; 2.3 eV at the fixture's
  converged μ=399) and now ELIMINATED by the exact route.** Fixture implied-Vxc
  goes [−145,+88] → [−6.3,−1.5] eV; legacy mode bit-identical (no regression).
- **FINAL RESOLUTION (Q): everyone was right, and the bug was in neither
  place anyone looked.** The owner's challenge ("V_NL has been correct for
  months") — CORRECT: the psp code was never wrong; every table/cutoff
  hypothesis was killed by bisection. N's measurement (V_NL≈0.09 eV on this
  deck) — ALSO correct. The cause: **the eager WFN-loader's `ntran≤1` branch
  never TRS-augmented `sym_mats_k`**, so `unfold_psi` classified the IDENTITY
  as a time-reversal row and silently returned `iσ_y·conj(ψ)` on an un-negated
  G-list — ψ*(−r) for ψ(r) at every k of every **`nosym` deck**. The
  transformation preserves norms, overlaps, ⟨T⟩ and ⟨V_H⟩ *exactly* (why all
  historical validation passed); it breaks only τ-dependent terms on
  non-centrosymmetric systems. The 12×12 deck is the first nosym deck ever
  run. The phdf5/phdf5_host path was ALWAYS CORRECT (silent backend-parity
  violation, invisible to the parity test which never covered nosym) — so the
  production GW ψ and all ΔSigXC sweep data are VALID; only the 1-node
  eager-generated kin_ion.h5 was corrupted.
- **Gate PASSED: 12×12 H0 vs QE kih.dat rms 38.69 eV → 0.0001 eV** (max
  0.0006); implied Vxc matches vxc.dat to 0.0001 eV; fixtures bit-exact no-op.
  Fix merged (TRS-augment + hard guard + regression test).
- REGENERATION QUEUE: 12×12 kin_ion.h5 with the fixed loader + N's --hartree
  at ≥160 bands (fixed 120-band file staged at wk_Q/), then rerun the 276c
  pipeline → the first trustworthy 12×12 eqp. dipole.h5 probably clean
  (phdf5-generated); cheap to redo for provenance. Unit trap recorded:
  kin_ion.h5 is in RYDBERG, kih/vxc.dat in eV.
- Process lesson (recorded): the synthetic-fixture "ntran=1 degeneracy" seen
  during phdf5_host testing weeks earlier WAS this bug — it was scoped around
  as a fixture edge case instead of chased. Edge cases in symmetry tables are
  never just fixture artifacts.
**Root cause (M):** `H0 = ⟨T+V_ion+V_NL⟩ (exact plane-wave kin_ion.h5) +
⟨V_H⟩ (ISDF centroid quadrature)` — two ~500 eV terms cancelling. At 276
centroids / 160 bands (1.7 c/band vs the fixture's 8.7, serving 144 k-points),
the ISDF V_H carries ~10% error → **rms 46.6 eV in H0** while Σ stays physical
(Σ never differences an exact quantity). Bit-identical at 4×4 and 8×10;
nband-dependent (VH[k0,n0] = 367/505 vs 460.55 true); ground truth was on disk
(QE pw2bgw kih.dat: kih+vxc−E_DFT ≈ 0.01 eV). The historical 07-23 "reference"
run has the same rms 47.1 eV defect. **Every parallel milestone stands; the
campaign caught a pre-existing scientific-validity issue.**
- Merged: print-only implied-Vxc sanity guard in gw_output (fires on both
  broken runs, silent on the fixture — would have caught this on 07-23).
- Designed: fold V_H into kin_ion.h5 at generation (exact FFT-grid route;
  `psp/get_DFT_mtxels.get_kin_ion(include_hartree=True)` exists; 3 gaps
  enumerated incl. a latent 2D-truncation flag inconsistency + a provenance
  attr to prevent double-counting). Makes eqp0 centroid-count-independent.
- The K sweep (606→2406c) now doubles as the V_H-convergence experiment:
  per-rung readout dH0 = (kin_ion + VH) − kih_QE, require rms ≪ 1 eV.
- Bonus finding: dipole.h5 head S(ω) built from a 120-band transition space
  while the run window is 160 — separate consistency item.

## 5b-prior. The hunt (kept for the record — how it was localized)
v8 (full pipeline, consistent window, rc=0, ζ 1330 s) still yields **QP gap
−136 eV** — so this is NOT the restart-window issue: a genuine mesh/scale-
dependent correctness bug at 8×10/P=80 that the P=4 (2×2) and P=8 (2×4) eqp
gates (1e-6) do NOT catch. Same failure family as the original z_q bug (silent
wrong results at untested device layouts). Clues: (a) v8's V_q trace differs
from v5's by 27% though V_q is band-window-independent (5.006e9 vs 6.912e9);
(b) at P=80, μ_pad=320 vs n_rmu=276 → **44 zero-pad centroids** (P=4 has ZERO
padding — 276/4 divides — exactly why small-P gates pass); truncation telemetry
correctly shows n_log=276. Discriminators in flight:
- A/B: restart from v8's tensors on a 4×4 mesh (job 7874616) — sane ⇒ Σ-stage
  8×10 bug; garbage ⇒ ISDF-stage corruption.
- Cheap repro: the tiny fixture on 80 EMULATED host devices, 8×10 mesh, 1 node
  (job 7874618) — if it reproduces, we get a minutes-scale debug loop.
- **NARROWED (K's decomposition): Σxc is SANE** (VBM σXC −15.075, CBM −9.092,
  ΔSigXC +5.983 eV — physical); the garbage is confined to **H0/VH assembly**
  (per-band VH 100.98→505.51 eV in one run). Read conditioning off ΔSigXC
  (sigma_diag.dat), not the eqp gap, until fixed.
- **RESOLVED DIRECTION (both discriminators in): the parallel machinery is
  EXONERATED.** The 80-emulated-device fixture (8×10 mesh, 25% μ-pad) passes
  at 1e-6 eV, and the 4×4/16-proc A/B reproduces the production garbage
  BIT-IDENTICALLY (−136.0137 eV at both meshes). The corruption is
  deterministic and mesh/P-independent → a data/config-side H0 issue (stale or
  mismatched kin_ion.h5/dipole.h5 for this window/nband, or a units/indexing
  slip the 12×12 deck exposes). Under root-cause by workstream M (forensics on
  the on-disk artifacts; eqp_g0w0.dat sanity + kin_ion dims are the key
  checks). **All P=80 pipeline milestones stand.**

## 5c-live. STABILITY VERDICT SO FAR (owner's criterion): NOT YET STABLE
Sweep points in: at μ=606 (first rung where truncation activates, n_keep
583/606) ΔSigXC swings **+5.98 → −0.14 eV** vs 276c; CBM σ_XC moves 6.6 eV and
even bare σ_x moves 5.7 eV. By the owner's criterion (small-μ observables must
be stable before attributing large-μ errors), **no attribution is defensible
at ≤606c**. The 1194/1998 rungs (running) discriminate: monotone settling ⇒
276/606 simply underconverged; erratic ⇒ misbehavior in the newly-exercised
truncated regime (first production use of rank-truncate with n_keep<n_log).
Kept-rank curve so far: 276/276 → ~583/606 → ~1015/1998 → ~1052/2406.
μ=2406 rung DIED at the memory envelope (MaxRSS 58.5 GB/rank,
RESOURCE_EXHAUSTED → Gloo symptom): K's sweep used 27 chunks where the ladder
prescribed 81 for μ≥2000, stacked on the measured 18.9 GB `_solve_all_at_once`
gather — the wall J's in-flight SUMMA fix removes.
ALSO IN FLIGHT: N = exact-route V_H + full input-file parameter inheritance
for get_DFT_mtxels + the Coulomb-convention error decomposition (2D-truncated
V_q[0] in sig_h vs QE's full-3D Hartree — the owner's systematic-error
hypothesis, tested host-side); J = the two structural large-μ fixes
(all-to-all ψ-gather; SUMMA back-solve + distributed-eigh plumbing via
ScaLAPACK pzheevd since SLATE heev is broken).

## 5c. Conditioning vs μ — the ceiling is MEASURED (K)
- μ=276: n_keep 276/276 (no truncation). **μ=1998: n_keep ≈ 1011–1018 of 1998**,
  cut exactly at rcond=1e-8 with NO spectral gap → **the pair-density numerical
  rank saturates at ~1015 for nband=160**. Centroids beyond ~1200 add nothing
  at this band window; the informative probe is nband=400 (does the ceiling
  scale with bands?) — run6_c2400@160b is predicted worthless.
- Implication for the 10k-centroid ambition: μ scaling must be accompanied by
  band-window scaling — μ_useful ≈ O(rank of the ψψ pair space), not free.
- 5-point ΔSigXC-vs-μ sweep RUNNING on normal queue (276/606/1194/1998/2406;
  jobs 7874609-12; readout: prof_k/sigma_sweep.py).
- Two more measured scale walls (K): `_solve_all_at_once` all-gathers the whole
  (nq,μ,μ) factor → **18.9 GB/rank measured vs 0.115 modelled** → envelope wall
  at μ≈4400 (J's SUMMA back-solve design is the cure, now with data); one
  compiler-flagged remat: 1.475 GB `_reshard_all` all-gather scaling with
  μ·nband.

## 5d. The μ ladder after J's structural round (wall #0 CURED)
- **ψ-gather all-to-all MERGED**: Stage-C 9.4× smaller (129→12.9 GB at run1
  scale; constant 1.47M→157k B/cr); MoS2 276c goes 4 chunks → 1; gated
  bit-identical P=4-vs-P=8. That wall's own ceiling: μ≈68k.
- **SUMMA back-solve: honestly REVERTED** — flat-mesh column sharding makes
  the 'x'-psum mix unrelated column blocks (NaN with rc=0, caught by eqp float
  counts). Correct fix requires the ζ-reshard rewrite; bundled with the
  distributed-eigh work.
- **Current binding walls for the 10k push** (P=80, 72 GB/dev):
  1. `F_tensor_write` SlabIO allgather: **μ≈3,960** — CHEAPEST LIFT: the env
     fix (mpi4py + h5py HDF5_MPI=ON → PHDF5_HOST) raises it to ≈50,100.
     (Defer the venv mutation until the sweep jobs finish.)
  2. Replicated rank-truncate eigh: μ≈4,000 by TIME (5.5h @4k, 86h @10k) —
     needs the distributed eigh (ScaLAPACK pzheevd handler; SLATE heev is
     broken at the library level).
  3. `B_cct_chol`: μ≈16,700 — next after those.
  Both #1+#2 together open 10k+.
- Ops notes: wt-J now shared with N's in-progress Hartree work (J committed
  only its own two files); fixture symlink overwrite incident (restored;
  switch gate stagers to cp -L / add a read-only fixture guard).

## 6. Open items (ranked, post-campaign)
1. **Exact-route V_H** (M's design): fold V_H into kin_ion.h5 at generation —
   makes eqp0 centroid-independent and closes the 500 eV cancellation
   analytically. Includes fixing the latent 2D-truncation flag inconsistency.
   THE prerequisite for trusting any 12×12+ eqp.
2. **Distributed rank-truncated pinv** (J's full design incl. SUMMA back-solve;
   now with measured urgency: `_solve_all_at_once` hits 18.9 GB/rank, wall
   μ≈4400; compute wall ~5.5h @4k). PREREQ: unified host FFI lib with SLATE
   eigh — workstream L is building exactly that.
3. **Structural ψ-gather fix** (J's all-to-all design: 129→12.9 GB/rank —
   removes wall #0 entirely, beyond the merged elision).
4. **Environment fix**: mpi4py + parallel h5py → PHDF5_HOST reachable → kills
   the remaining replicated I/O seams (reads fixed in code; the WRITE gather
   is still the μ≈4650 wall).
5. P>1 compile-cache deadlock (storm lever; compilation itself is only ~7 s/rank
   — the cost is tracing/dispatch × ranks).
6. Suite gap: add a multi-device gn_ppm rung (the reduce-scatter path had zero
   coverage until J's gate).
7. dipole.h5 regeneration for nband>120 windows (head S(ω) consistency).
8. Owner calls: delete dead AsyncWfnReader; zero pad-slot bands inside load();
   bootstrap for run_nscf/run_sternheimer/kmeans_cli; ζ-ladder legacy
   explicit-cusolvermp semantics; ζ-reuse-from-clean-zeta_q.h5 restart feature.

## 7. Downfolding readiness — DONE (L): the distributed-linalg verdict
- **SLATE built on Frontera for the FIRST TIME** (the old host lib was
  cusolvermp-only; SLATE was Perlmutter-only — the "existing slate host lib"
  premise was wrong). Unified lib: `$WORK/lorrax_ffi_unified/build_host/
  liblorrax_ffi_host.so` — all 9 host targets (phdf5×3, slate×5, scalapack×1).
  ScaLAPACK needed a hand-injected MKL BLACS link (CMake gap). Build hygiene
  notes: SLATE's own CMake needed Fortran dropped; /work2 Lustre stalls under
  the 40-node job → stage source to node-local /tmp (2-min build).
- **Bench (facade, CPU meshes, relerr ≤1e-15, ZERO hangs over 10 reps):**
  cholesky 400: FFI 0.111 s vs native 0.204 (1.8×); solve_lu 400×200:
  **6.1×**; solve_lu in the μ=3000 chain: **11.5×**; sharded SᴴWS GEMM chain
  0.07-0.38 s at μ=1200/3000. FFI per-call floor 0.10-0.26 s (first call
  1.4-2.7 s compile).
- **Verdict: cholesky + solve_lu + the sharded overlap/downfold chain are
  production-ready at downfolding sizes.** Distributed eigh is BLOCKED:
  SLATE host heev SIGSEGVs deterministically even at n=64 on a 1×1 mesh
  (library-level, not LORRAX; MPI/threading/symbols ruled out) — use native
  eigh meanwhile (0.93 s @1200, 11.4 s @3000, 3.7e-15).
- **Critical constraint for production runs:** FFI linalg runs only on square
  (or N×1) meshes — an 8×10 silently resolves to native everywhere. Choose
  8×8 or 10×10 when FFI linalg matters.
- Two bugs documented: L-1 facade guard gap (cholesky+slate at 2×4 resolves
  then crashes — violates the resolve-is-a-promise contract; small fix) and
  L-2 the heev SIGSEGV. Two doc corrections: linalg_ffi.md's "FFI 100-600×
  slower" claim was a batched-GPU result (backwards for CPU single-tile);
  FI_PROVIDER=mlx (not tcp) for honest inter-node latencies.

## 7b. The audit wave (O/P/Q) — all landed & gated
- **O (silent-failure hardening):** 3 latent bugs fixed (dead ζ basis guard
  probing a renamed dataset; zeta_is_done written-never-read → half-written ζ
  flowed downstream rc=0; BSE restart-file picked LEXICOGRAPHICALLY — 1194
  beats 276). New: common/sanity.py (7 sharded-safe gates, LORRAX_SANITY),
  barrier() replacing 7 swallowed collectives, failfast excepthook, rc
  discipline across 19 mains, implied-Vxc detector on the post-hoc eqp CLI
  (recalibrated after a false-positive on semicore physics — gates caught it),
  eqp writers NaN-refuse + float-count re-parse. 17/17 + fixture eqp PASS.
  Adjudicated: the fixture's own kin_ion/eqp_ref embed an unphysical mean
  field → fixture regeneration queued; guards stay loud on it by design.
- **P (loose ends):** all 10 backlog items closed (L-1 geometry guard with
  16-mesh exhaustive equivalence; NEW probe_target() three-state diagnosis —
  the old bool told users to rebuild a healthy library; ζ-reuse restart with
  effective-value provenance; allow_abbrev=False ×47; bootstrap stragglers;
  htransform cache hook; AsyncWfnReader deleted; fixture write-protection
  with root cause; linalg docs corrected incl. the backwards perf claim;
  docs/dev/env_vars.md registry). G1-G5 all green incl. a real SLATE potrf
  call through the promise (relerr 2.2e-16).
- **Q (the nosym unfold bug):** see §5b — the campaign's deepest fix.

## 7c. Conditioning: the band-window ceiling is CONFIRMED + a new wall (overnight)
- **n_keep telemetry across rungs (nband=160, rcond=1e-8): μ=1998 → ~1015;
  μ=2406 → ~1053. +20% centroids → +3.6% kept rank — the pair-density rank
  ceiling is set by the BAND WINDOW, not μ** (the owner's ≥8×bands rule
  mechanism, now measured). And λ_min_kept/λ_max = rcond EXACTLY — no spectral
  gap; the truncation is purely the knob (the owner's "threshold too high"
  critique confirmed as well-posed).
- Overnight probes RUNNING (72 nodes each, 144 ranks, 12×12 square mesh —
  first square-mesh production + fresh device count): **A** (7875070)
  c2406×nband=400, does the ceiling scale with the window; **B** (7875071)
  c1998×rcond=1e-10, does the plateau move with the knob. Both deliver n_keep
  in the first hour regardless of tail survival.
- **NEW WALL (top engineering item): a per-r-chunk anonymous-memory RAMP** —
  the 40-node μ1998/81-chunk run died bad_alloc at chunk ~32, climbing
  +2.9 GB/node/chunk (MaxRSS 63.9 GB vs planner 18.9); extrapolated need
  ~264 GB/node ⇒ high-chunk-count configs cannot finish at any wall time.
  Mitigated tonight (15 chunks); tonight's P=144 pair discriminates
  fixed-per-chunk vs working-set-proportional. LEAK HUNT = next workstream.
- htransform GW-QP bandstructure delivered (workstream R): DFT 1.7008 /
  QP 2.6513 eV on-path, GW makes the gap indirect (CBM → Λ/Q valley);
  interpolation rms 1.9 meV in the gap region. Two parser traps queued
  (eqp0.dat format + missing eV→Ry conversion, both silent-failure class)
  + the fixture's K-point is M-type (wrong for hexagonal).

## 9. Second wave (2026-07-26): S/T/U + the physics program
### T — the memory ramp: cured (glibc, not JAX)
`live_arrays` byte-constant across 30 chunks — the ramp was **glibc heap
retention** (dynamic mmap-threshold ratchet + 28 XLA worker arenas; leak ∝
back-solve FLOPs, 0.10-0.16 GB/1e9 cMACs). Cure: `tune_glibc_malloc()` +
`malloc_trim` per chunk → **+0.176 → +0.0003 GB/rank/chunk (~600×)**, zero
time cost. Bonus: a per-chunk jit recompile fixed (solve stage **9.1×**);
run_A/B's actual killer identified TO THE BYTE (`nq·16·(μ_pad²+2μ_log²)`
back-solve arena, unmodelled) and eliminated by the new
**`distributed_zeta_solve = auto|replicated|per_q|distributed`** key
(per-q tile gather: 27.76→0.19 GB/rank @c1998; 40.48→0.28 @c2406;
per_q bit-identical to replicated). **c2406/b400 now fits at 19.8 GB/rank.**

### U — symmetry is now MEASURED at load, not inferred
`WfnLoader` computes the spin-resolved density (QE-style k-weighted, raw IBZ
ψ — the unfold is never trusted) and checks: TRS via the Kramers identity
m_{−k}=−m_k (the naive Σw·m test would false-fail valley-coupled MoS₂);
spatial ops on their little groups; ∫ρ=nelec. `SymMaps` GATES TRS
augmentation on the verdict; flags-vs-measurement disagreement is loud.
Synthetic magnetic control correctly refuses TRS rows. 4.8 s at 12×12,
9 ms cached. Gaps stated: U_spinor cancels from the trace (extend to the
m-vector to close); nspin=2 unsupported loudly.

### S — the V_H architecture (owner's design) + the convergence measurement
`kin_ion.h5` stores `kin_ion` (pristine) + `v_hartree` (full exact window
matrix) SEPARATELY; **`hartree_source = auto|stored|isdf|gspace`**
(stored → ISDF V_q0 → G-space). Migration gates bit-identical (eqp0+eqp1
0.00e+00 vs the fold-in route). QSGW basis-rotation bug caught preemptively
(stored V_H is DFT-basis; rotated-basis Σ needs U·V_H·U† — fixed, hardware-
untested). Exact-route scaling audited honestly: 1/P today; the fix design
(band-sharded ρ FFTs + one 1.4 MB psum, replicated Poisson) makes in-loop
QSGW affordable (~0.5 s/iteration at P=80). ISDF V_H CONVERGENCE (measured):
≤1% relative at ~6 c/band; plateau 0.11-0.20% past ~9 c/band,
rcond-INDEPENDENT; semicore BEST-served (0.005%); high-conduction tail worst.
**The criterion caveat: 1% of ~500 eV = eV-scale frontier errors that don't
cancel (4.15 eV of gap at 606c)** → stored/gspace is the quantitative
default; isdf is the survey tier. Exactly the owner's ordering.

### The hypothesis scorecard (what survived measurement)
| hypothesis | verdict |
|---|---|
| ngk-variation causes the compile storm (owner+me) | ✗ — rank replication |
| async read would speed htransform (implied) | ✗ — overlap measured 0.000 |
| restart-window mismatch caused v6 garbage (me) | ✗ — real cause deeper |
| V_NL code broken at 80 Ry (N) | ✗ — loader nosym unfold bug (owner's pushback was the pivot) |
| kin_ion exonerated (M) | ✗ — circular; Q re-opened and found the unfold bug |
| 2D-truncation convention mismatch (owner+me) | ✗ — convention correct; but the probe found the coverage trap |
| ρ misses bands when nval<nelec (owner) | ✗ — tr G=26.000; BUT exposed the dead-code duplicate band_ranges |
| threshold rcond=1e-8 too high (owner) | ✓ — 98% of modes survive at 1e-10 |
| kept-rank ceiling set by band window (owner's 8× rule) | ✓ — 1053→1677 with nband 160→400 |
| ISDF V_H can reach ≤1% (owner) | ✓ (relative) — at ~6 c/band; criterion nuance above |
| memory ramp = JAX leak (assumed) | ✗ — glibc heap retention |

## 10. Third wave (07-26): the owner's distributed-linalg + V_H program — LANDED
### V/W — ScaLAPACK pzheevd + the real `distributed` ζ tier (HEAD b1efa0f)
- MKL 2020u1 (via intel/19.1.1) is Frontera's only ScaLAPACK — handler added
  with ZERO new link deps. Test ladder as prescribed: 1×1 → 2×2 → 4×4 at
  n=64/512/2048/2016/2448 (residual 1e-15; 1.99-2.89 s/matrix at production μ).
- **Silent-corruption trap caught in pzheevd itself**: INFO=0 + correct
  eigenvalues + GARBAGE eigenvectors on short back-transform workspace
  (residual 1.40 with perfect orthonormality) — only the strict
  eigenpair-contract test caught it; LWORK now floored with the pXunmtr bound.
- `distributed_zeta_solve='distributed'` is REAL: distributed eigh → local
  truncation (replicated spectrum, n_keep telemetry kept) → 2D-sharded C⁺ →
  stacked GEMM C⁺@Z both P(None,'x','y') (the owner's lstsq-free scheme; Z
  consumed in native layout — J.9's trap avoided). Gates: eqp 1e-6 at 2×2 &
  4×4; 2×2-vs-4×4 AND distributed-vs-replicated **0.00e+00**; rectangular
  refused at resolve. HLO audit: collective bytes −38%, no O(μ²)-per-q excess.
- Honest verdict: loses on wall at small n (7× at 512), wins on MEMORY
  (65 MB vs 9.36 GB/rank) and P-scaling — removes the ~4k time wall.
  Square-or-1D mesh only. CPU `auto` eigh deliberately stays native.
- Transverse: already fully 2D-sharded (pXgetrf/rs); the real gap is a silent
  replicated fallback on non-dividing n_log — queued as a pad-numerics item.
- Slate-eigh-on-CPU now refuses at resolve (was a SIGSEGV); build-script
  SIGPIPE bug + two main-broken contract tests fixed en route.

### X — G-space V_H strong scaling (merged 217d356)
- Real blockers: `jax.devices()` AT MODULE IMPORT in get_DFT_mtxels (pinned
  every psp-importing CLI to one process — broad fix); no process-local loads
  (added WfnLoader.load_process_local + a latent every-rank-gets-process-0's-
  device bug fixed).
- Design: k-partitioned round-robin ρ (P=1 = identical op sequence ⇒ EXACT
  parity), Poisson replicated by design, documented assembly gather.
- Gates: kin_ion bit-identical at every P; v_hartree 2.1e-14 (below the
  28-vs-56-thread noise floor); scaling 159→14.3 s at P=16 (70%; residual =
  P-invariant load floor QSGW never pays). **HLO contract met exactly: ONE
  1.40 MB ρ all-reduce + ONE 8.3 MB assembly gather + 144 B — nothing else.**
- QSGW: ~3.2 s/invocation at P=80 (S's 0.5 s corrected); psi_rotation=U opens
  ρ-from-rotated-orbitals; invalidate_hartree_cache() prevents iteration-0
  V_H freeze. hartree_source=gspace end-to-end eqp identical to stored.

## 8. Still in flight at report time
- K's ΔSigXC-vs-μ sweep (606/1194/1998/2406; jobs 7874609-12, hours) — doubles
  as the V_H-convergence experiment (readout recipe in §5b).
- Branch: `fix/zq-band-gather-device-invariance` @ f7f45d1, local only (NOT
  pushed). 16 substantive commits + merges this campaign.

### 8a. RESOLVED / re-scoped by workstream AC (2026-07-26 evening)

**"Flagship A (c2406 × nband=400, P=144) is running" was wrong.** Job 7875551
never finished r-chunk 1 of 15 in 7 h 26 m — it was submitted at 12:11, before
AA merged at 16:20, so `auto` resolved the ζ back-solve to the *defeated*
`per_q` (Y.2's form: the whole `(nq,μ,μ)` = 13.81 GB/rank all-gather, `nq`=144
times per chunk ≈ 2 TB/rank/chunk). Zero `LoopProgress` milestones is the
signature; scorecard **AC.0** / `wk_AC/POSTMORTEM_7875551.md`.

**Salvaged from it, complete over all 144 q — the largest-μ conditioning datum
the campaign has:** `n_keep` at μ=2406, nband=400, rcond=1e-8 =
**1675–1680, mean 1676.31** (43.7 % → **69.7 %** of `n_log` as nband goes
160 → 400). 2.5× the bands buys 1.594× the rank; the ≥8×-bands rule's ~2600
does not materialise, and the cut is rcond-bound (retained κ = 1/rcond = 1e8).

**Stage table, first real chunk timings at 2406c / b432 / P=144** (job 7876086,
AA-fixed `per_q` + `PHDF5_HOST`, stopped by the owner at 3/15 for cluster
priority — healthy, not failed):

| stage | measured |
|---|---|
| startup → planner | ≈2 min (warm page cache) |
| replicated `rank_truncate` eigh, 144 × n=2448 | **≈17 min** |
| ζ r-chunk, steady state | **283.1 s** = `z_q_build` 55.0 + back-solve 228.1 (80.5 %) |
| ζ per-chunk `h5_write` (accumulate) | **6 ms** |
| **projected full ζ-fit (15 chunks)** | **≈71 min** — 2.1× better than T.7's 2.5 h |
| V_q / screening / Σ | **NOT REACHED** — still open |

so the owner's "post-ζ ≤ 50 % of the ζ-fit wall" criterion is a **≈35 min**
budget, and closing it needs one `full` run to end-of-fit plus two ≤2 h gating
cells (`wk_AC/runAC.sbatch` modes `vq` and `restart`), not another 8 h run.

**New blocking finding: `distributed_zeta_solve = distributed` does not survive
P=144** (job 7876062, scorecard **AC.2**). `pzheevd` completed, then `C⁺`
formation died on Gloo (`ReduceScatter`/`AllGather`, 306 × `Socket closed`) —
**not memory**, MaxRSS 10.69 GB/rank. And on the same nodes the eigh it replaces
is **faster**: replicated ≈7 s/matrix vs `pzheevd` ≈12 s/matrix, i.e. the tier
is **1.75× slower AND fatal** at this rank count. V/W gated it only at P ≤ 16.
Until that is fixed, **`per_q` (what `auto` picks at this μ) is the production
tier on a 12×12 mesh.**

## Method tradeoffs (measured)

Workstream Y. The owner's directive was to make HLO tracing a *standing practice*
— "extract timing, communication and memory data from HLO traces at least
occasionally … to contribute to the report on tradeoffs between different
methods". This section is the first output of that practice, and the harness
that produced it is checked in so the next one is a one-liner:

    /scratch2/08271/jackmc/lorrax_setup/wk_Y/submit_probe.sh <tag> <P> <wall_min> KEY=VAL...
    /scratch2/08271/jackmc/lorrax_setup/wk_Y/analyze_probe.py <rundir> [...] [--csv]

`probe.sbatch` runs the MoS₂ 12×12 production deck with the rank-0 HLO dump (K.0's
verified flag set), the code's own stage timers, per-r-chunk RSS/`live_arrays`
(T.1's instrumentation), a node memory sampler and a `timeout`-based self-kill;
`analyze_probe.py` turns a run dir into one block: stage walls, per-chunk cadence
and its z_q/back-solve split, the collective table by op **and by module**, the
RSS slope, and planner-HWM-vs-measured. Both are documented in their own headers.
13 cells, all MoS₂ 12×12 / 80 Ry / nk=144 / nband=160, `r_chunk=6480`, three
r-chunks measured of 27–28, `LORRAX_EXIT_AFTER_ZETA=1` except the V_H row.
The two μ=1998 cells are the follow-up that settles anomaly 1 and were run to the
first r-chunk only (`MAXCHUNK=1`), which is all their deciding artifact needs.

### (a) ζ back-solve tier — `distributed_zeta_solve`

Per-r-chunk walls, rank 0, from `LORRAX_RCHUNK_DEBUG`. `back-solve` is the phase
the tier actually changes; `z_q` is the tier-invariant control.

| μ | P (mesh) | tier | chunk [s] | z_q [s] | **back-solve [s]** | full fit (×n_chunks) |
|---|---|---|---|---|---|---|
| 276 | 16 (4×4) | replicated | 77.6 | 38.4 | **38.6** | 2094 s |
| 276 | 16 (4×4) | per_q | 510.4 | 39.6 | **470.4** | 13 781 s |
| 276 | 16 (4×4) | distributed | 59.4 | 39.3 | **19.6** | 1604 s |
| 276 | 64 (8×8) | replicated | 34.1 | 18.6 | **14.9** | 954 s |
| 276 | 64 (8×8) | per_q | 624.4 | 18.9 | **604.9** | 17 482 s |
| 276 | 64 (8×8) | distributed | 31.8 | 18.8 | **12.3** | 890 s |
| 606 | 64 (8×8) | replicated | 59.4 | 21.3 | **37.6** | 1663 s |
| 606 | 64 (8×8) | distributed | 45.2 | 21.7 | **23.0** | 1266 s |

Every row is the mean over 3 completed r-chunks (all eight cells finished theirs;
none was cut short by the self-kill). Note what that costs: the P=16 `per_q` cell
needed **1797 s of wall for the same three chunks the other two P=16 cells finished
in ~180–230 s**, and the back-solve is **92 % of a per_q chunk at P=16 and 97 % at
P=64**, against 50 % (replicated) and 33 % (distributed). The tier's cadence is also
flat to 0.03 % across chunks (624.29 / 624.32 / 624.49 s), so this is a structural
cost, not a warm-up artefact. Re-read the live values any time with
`python3 analyze_probe.py runs/*/ --csv` (`wk_Y/runs/MATRIX.csv` is the snapshot
behind this table).

Collective traffic of the ζ factor + back-solve modules only (optimized HLO,
**bytes per execution of the module** — see the caveat below), separated from the
run-wide traffic by module name:

| μ / P | tier | modules that carry it | bytes | instrs (whole run) |
|---|---|---|---|---|
| 276 / 16 | replicated | `_solve_all_at_once` 238.9 MB + `_reshard_z` 806.2 MB + `_fn` 225.2 MB | **1270 MB** | 43 |
| 276 / 16 | distributed | `_block` 1122.7 MB + `_fn` 59.7 MB | **1182 MB** (−7 %) | 30 |
| 276 / 64 | replicated | 265.4 + 223.4 + 215.3 | **704 MB** | 43 |
| 276 / 64 | distributed | 625.2 + 33.2 | **658 MB** (−7 %) | 30 |
| 606 / 64 | replicated | 1061.7 + 978.3 + 446.8 | **2487 MB** | 43 |
| 606 / 64 | distributed | 1309.4 + 132.7 | **1442 MB** (−42 %) | 30 |

`_reshard_z` is **absent from every `distributed` dump** — V.6's central claim,
reproduced on the production deck at two P and two μ. The instruction count falls
43 → 30 in every distributed cell.

`_solve_all_at_once`'s gather is `c128[144, μ_pad, μ_pad]` = `nq·μ_pad²·16` to the
byte at both sizes (238.9 MB at μ_pad=288, 265.4 MB at 320, 943.7 MB at 640).

**Verdict on the three tiers, from these numbers:**

* **`distributed` wins on wall at every point measured**, including the small ones
  where V.3's library-level ladder predicted it would lose (7× slower at n=512).
  The reason the end-to-end result differs from the ladder: V.4's explicit `C⁺`
  turns the per-r-chunk back-solve into **one** GEMM instead of two and deletes
  `_reshard_z`, and the r-chunk loop pays that 27–28 times while the `pzheevd`
  factorisation is paid **once**. 1.97× on the back-solve at P=16/μ=276, 1.22× at
  P=64/μ=276, 1.63× at P=64/μ=606 — and the byte advantage grows with μ (−7 % →
  −42 %), i.e. the crossover V/W described is already behind us on this deck.
* **`per_q` is a memory escape hatch and nothing else — it is 12–40× slower.**
  This is the sharpest number in the table and it was not previously measured.
  See "anomalies" below for *why*, which is worse than the wall figure suggests.
* **`replicated` remains the right default at small μ**, and `auto`'s 4 GiB
  `LORRAX_ZETA_GATHER_CAP_GIB` gate correctly keeps `per_q` switched off there
  (276c ⇒ 0.19 GB, 606c ⇒ 0.94 GB, both far under the cap).

Strong scaling P=16 → 64 (4× the ranks, exactly 4.01× less r per rank) at μ=276:
chunk 2.28× (replicated) / 1.87× (distributed); z_q 2.07×; back-solve 2.58×
(replicated) / 1.59× (distributed). The distributed tier scales *worse* — expected,
since its collectives are the part that does not shrink with P.

### (b) `hartree_source` — the V_H tradeoff at P=16

Full pipeline (V_q → W → Σ) on an intentionally truncated ζ (2 r-chunks), so the
*costs* are production-real and the *physics* is not. Everything else identical.

| source | Σ stage [s] | its V_H sub-timers | Δ vs `stored` | extra collectives | total recorded | wall |
|---|---|---|---|---|---|---|
| `stored` | 317.2 (self 7.3) | — (one h5 read of `v_hartree`) | — | **none** | 651.8 s | 828 s |
| `isdf` | 318.5 (self 7.1) | — (reuses V_q[0], already built for W) | **+1.2 s ≈ free** | **none** | 651.3 s | 828 s |
| `gspace` | 337.9 (self 18.5) | bootstrap 0.08, ρ sweep 13.40 (psum 10.85), assembly gather 0.20 | **+20.7 s (+6.5 % of Σ, +3.5 % of the run)** | **exactly 3** | 674.8 s | 853 s |

The three extra collectives `gspace` introduces, diffed instruction-by-instruction
against the `stored` dump:

| op | shape | bytes | what |
|---|---|---|---|
| all-reduce | `f64[36,36,135]` | **1 399 680** | the ρ psum |
| all-gather | `c128[16,9,80,80]` | **14 745 600** | the ⟨mk\|V_H\|nk⟩ assembly gather |
| all-gather | `s32[16,9]` | 576 | the k-index payload |

**and nothing else — X.6's contract reproduced independently at P=16, on the
production deck, at a different band window (80 vs 120).** `stored` adds zero
collectives to the program.

H0 quality, same run: `stored` and `gspace` give an implied Vxc of
**[−24.262, −4.455] eV, identical to the printed digit**, guard silent. `isdf`
gives **[−162.85, +578.33] eV** and O's guard fires on 11 434 of 11 520 (k,n).
*Caveat, stated because it matters:* that `isdf` range is dominated by the
deliberate ζ truncation (2 of 27 r-chunks ⇒ a ~7 %-integrated V_q[0]), not by
centroid convergence — **S §5 remains the authority on `isdf` accuracy**. What
this cell *does* establish cleanly is (i) the cost ordering, (ii) that
`stored ≡ gspace` on H0 in a real driver run, and (iii) that the silent-failure
guard correctly catches an under-integrated ζ, which is a useful side result.

### Which tier / which source, when

**ζ tier.** Use **`distributed`** whenever the mesh is square (P ∈ {16, 64, 144,
196, …}) — it was faster than `replicated` at every point measured here, it is
bit-identical (V.5), it removes the only O(μ³) no-P-scaling stage, and its
advantage grows with μ. On a rectangular mesh it refuses at resolve time and
**`replicated` is the correct fallback**; do not reach for `per_q` to get
"distributed-like" memory there, because at 276–606 centroids it costs 12–40× the
back-solve wall for a 1.6× module-peak reduction (below). Keep `per_q` for what
`auto` already uses it for: the regime above `LORRAX_ZETA_GATHER_CAP_GIB` where
the replicated gather genuinely will not fit (μ_pad ≳ 2000 at nq=144) — but note
that at μ_pad=2048 it was **measured** to buy only 1.67× on module peak (anomaly 1),
not the order of magnitude T.5's table implies, so budget it as 1.67× and not more. **Production advice: prefer 144 ranks (12×12) over 80 (8×10)** — square
buys the tier *and* removes the K.2 rematerialisation (below).

**And the honest framing for both.** Above ~600 centroids none of these choices is
the binding constraint any more: `F_tensor_write`'s ζ writer gather is (anomaly 2),
it is P-independent, and it killed a 1998-centroid run outright. Tier selection buys
1.2–2× on a phase that is ~50 % of the ζ-fit; fixing the writer is what buys the next
rung of the centroid ladder. Order the work accordingly.

**V_H source.** `gspace` costs **+3.5 % of a full run** and buys the exact
mean-field with three collectives totalling 16 MB; `stored` costs nothing beyond
one 33 MB h5 read and gives the identical answer, but requires a pre-generated
new-format `kin_ion.h5` for the deck. So: **`stored` when the file exists**
(quantitative work, repeated runs on one deck), **`gspace` when it does not or
when the density will change** (QSGW in-loop, new decks, one-offs) — 20 s is not
worth a generation step. **`isdf` is genuinely free** and is the survey tier only;
its price is accuracy, not time, and S §6 quantifies it (eV-scale frontier error
at production μ). Nothing measured here changes that ordering.

### Anomalies the traces revealed (the standing value of the practice)

1. **`per_q` does not do what its name says, on XLA:CPU.** The optimized HLO of
   `jit(_solve_one_q_and_update)` contains the *same* two all-gathers as
   `_solve_all_at_once`, ending in `c128[144,288,288]` — **the whole (nq, μ, μ)
   stack** — and only *then* a `dynamic_slice` for the single q. The buffer
   assignment names it in the peak breakdown:
   `jit(_solve_one_q_and_update)/dynamic_slice … 238 878 720 bytes (30.2 %)` of a
   790 MB module peak, against 1240 MB for `_solve_all_at_once`. So per_q's
   measured memory win at 276c is **1.57×, not the 144× T.5 projected**, and
   because the module is executed once per q it moves **144 × 238.9 MB = 34.4 GB
   per r-chunk** where `replicated` moves 238.9 MB — which is exactly why it is
   12–40× slower. XLA did not sink the q-slice through the μ-axis all-gather even
   though the two commute.

   **Settled at production μ** (jobs 7875703/7875704: 1998 centroids, μ_pad=2048,
   P=64, 8×8, one r-chunk). Peak stack-trace breakdowns of the two solve modules:

   | | `per_q` | `replicated` |
   |---|---|---|
   | module peak | **11.98 GB** | **19.95 GB** |
   | total allocated | 11.16 GiB | 27.16 GiB |
   | the gather buffer | `dynamic_slice` **10 871 635 968 B (90.8 %)** | `slice` **9 662 519 808 B (48.4 %)** |
   | the logical-extent copy | — | `complex` **9 197 577 216 B (46.1 %)** |

   Both gather numbers are exact: `nq·μ_pad·(μ_pad + μ_pad/P_x)·16` =
   144·2048·2304·16 for per_q, `nq·μ_pad²·16` = 144·2048²·16 for replicated, and
   the replicated copy is `nq·μ_log²·16` = 144·1998²·16 — i.e. **T.3's arena
   formula is confirmed component by component**, and `per_q`'s own gather buffer
   (10.87 GB) is **LARGER than the replicated gather it exists to avoid**
   (9.66 GB). **T.5's "9.36 GB → 0.065 GB (144×)" is refuted.** The real saving is
   1.67× on peak / 2.43× on total-allocated, and it comes entirely from `per_q`
   not making T.3's two logical-extent copies — not from gathering less.
   Consequence for planning: **T.7's residency table needs revising** — its
   "c2406 with per_q → back-solve arena 0.28 GB, 19.8 GB/rank total" becomes
   ≈**14.96 GB** of arena by the measured law (μ_pad=2448, P_x=12), so the c2406
   config's headroom is far smaller than T.7 concluded. The tier remains
   numerically exact (T.6's bit-identity gate is untouched); only its memory and
   latency claims were wrong.
   
   **One production-μ cadence point, from the cell that died** (`d_1998_P64_rep`,
   1998c, P=64): its single completed r-chunk was **267.1 s (z_q 23.0 + back-solve
   242.6 s)** — the replicated back-solve grows 14.9 → 242.6 s for μ 276 → 1998 at
   fixed P, i.e. ≈μ^2.2, against a μ² arithmetic expectation. The `per_q` cell at the
   same size had **still not completed one r-chunk** more than 20 minutes into its
   chunk loop, so the 12–40× penalty measured at 276c does not shrink with μ — which
   is what the Y.2 mechanism predicts, since the 144 serial executions do not get
   cheaper relative to one batched call.
2. **The biggest collective in a ζ-fit is not in the ζ tier — and it is now a
   measured cause of death, not a projection.** In every cell it is
   `F_tensor_write`'s unsharded G-flat gather, `c128[nq, μ_pad, ngkmax]` =
   `nq·μ_pad·ngkmax·16` to the byte: **5.71 GB** at μ_pad=288, **6.34 GB** at 320,
   **12.69 GB at 606 centroids** — **P-independent**, so adding nodes does not
   shrink it. The 1998-centroid replicated cell (job 7875704, P=64, 32 nodes)
   **died rc=1 on it**, in `multihost_utils._handle_array_process_allgather`:

       RESOURCE_EXHAUSTED: Out of memory allocating 40 594 046 976 bytes
       144 · 2048 · 8603 · 16                     = 40 594 046 976   ✓ to the byte

   It had completed r-chunk 1 (266 s) and died at the ζ write. **The planner named
   `F_tensor_write` as the binder and sized it at 19.97 GB/dev —
   2.03× too small**, i.e. it flagged the right term and under-predicted the number
   that killed the job. This is J.10 / V.7's named wall with an exact formula, a
   three-point measured curve *and* a death certificate.

   The margin, exactly: the same `[rchunk_dbg]` line reports
   **`live=60.685 GB`/rank** at 1998c/P=64 (against 8.3 GB at 276c), so the 40.59 GB
   writer request lands on top of 60.7 GB already live — **101 GB against the
   96 GB/rank a 2-rank Frontera node affords.** It misses by ~5 %. That is why this
   term, not the ζ solve, sets the centroid ceiling, and why **no tier choice
   changes it** (per_q and distributed alter the back-solve; the writer is outside
   both). At the flagship's P=144 the live term shards 2.25× further while the
   writer only drops to 39.96 GB (μ_pad=2016) — so it fits, but the headroom is
   thin and this collective is what consumes it. Installing mpi4py +
   `HDF5_MPI=ON` h5py (so `PHDF5_HOST` becomes reachable) remains the fix and is
   now the highest-priority item on the centroid ladder. (O's fail-fast worked
   exactly as designed: named rank 56/64, exited rc=1 without teardown.)
3. **Two loader collectives are P-LINEAR and unmodelled by the planner**:
   `s32[P·nk, 36,36,135]` = `P·nk·n_rtot·4` (1.61 GB at P=16 → **6.45 GB at
   P=64**) and `c128[P·nk, 8603]` = `P·nk·ngkmax·16` (0.32 → 1.27 GB). Together
   7.7 GB/rank at P=64 and a projected **17.4 GB/rank at P=144** — i.e. this pair
   grows into the dominant memory term exactly where the campaign is heading, and
   the ISDF memory model does not know about it. K.1 #3/#8 flagged the family at
   one P; this pins the scaling law at two.
4. **K.2's involuntary full rematerialisation is GONE — zero occurrences in all
   13 cells.** K.2 measured 80 of them (1/rank) at P=80 on the **8×10** mesh, on a
   `{[1,10,1,1,8]<=[8,10]T(1,0)} → {[1,8,1,1,10]<=[80]}` reshard. At 4×4 and 8×8
   the two shardings in that pair are indistinguishable, so the transposition is
   free. Reading this together with anomaly 1's mesh requirement: **square meshes
   are worth choosing for two independent reasons.** (One caveat: 4×4 and 8×8 vs
   8×10 is square-vs-rectangular *and* smaller; a 10×10 point would nail it.)
5. **Planner accuracy is drifting toward under-prediction as μ grows, and it has
   already crossed.** `HWM estimate × 2 ranks` vs the measured node peak: **1.54×**
   conservative at P=16, **1.11×** at P=64, **0.98× at 606c/P=64**, and at
   1998c/P=64 its own named binder is **2.03× under-sized** against the allocation
   that killed the job (19.97 GB/dev modelled, 40.59 GB requested — anomaly 2).
   It is no longer a safety margin at production μ.
   The binder is `A_centroid_load` in all 13 cells (not `C_fit_one_rchunk` as at
   P=80/19440), because the smaller r-chunk moved the binding stage.
6. **The T.2 glibc cure holds everywhere**: RSS slope ≤ 0 in every multi-chunk
   cell (−0.08 to −0.09 GB/rank/chunk, i.e. chunk 1's one-off transient being
   returned, not a ramp), and `jax.live_arrays()` byte-constant across chunks in
   all of them. T.4 also holds: rank-0 compile counts are 195–208 for a ζ-only run
   and 475 for the full pipeline — no per-chunk multiple.

**Methodological caveat that any future reader of these tables needs.** The HLO
gives bytes **per instruction per execution**; it does not give execution counts.
`replicated` executes its solve module once per r-chunk, `per_q` executes it
`nq` = 144 times, and the two look nearly identical in a naive byte total. The
`by module` breakdown plus the code's loop structure is what makes the table
honest — `analyze_probe.py` prints the former and this note is the latter.

### Follow-up (workstream AA, same day): three of these anomalies are now closed, with before/after cells from this same harness

The value of the practice, concretely: anomalies 1, 3 and 5 above were turned into
code changes and re-measured with `submit_probe.sh … SRC=<worktree>/src` — **no new
instrumentation**, three cells (`wk_Y/runs/AA_P16_rep_after`, `AA_P16_per_after`,
`AA_P64_606_rep_after`; jobs 7875721 / 7875722 / 7875725), and `MATRIX.csv`
regenerated so they sit next to the originals. Full write-up: scorecard **AA**.

**Anomaly 3 (the two P-LINEAR loader collectives) — they were never loader code.**
Both are JAX's own `multihost_utils.assert_equal`, which
`jax.device_put(numpy, <a multi-process NamedSharding>)` calls unconditionally to
check that every rank passed the same array (`jax/_src/dispatch.py::
_device_put_sharding_impl`), and `assert_equal` is `process_allgather(x,
tiled=True)`. A debug assertion, invisible in the source, was all-gathering `P ×`
every host table the loader stages — on device *and* on the host. Replaced by an
explicit process-local placement (`common.collectives.device_put_process_local`:
slice this rank's own shard, `make_array_from_single_device_arrays`, zero
collectives).

| 276c, P=16, identical parameters | before (`a_P16_rep`) | after (`AA_P16_rep_after`) |
|---|---|---|
| collective instructions | 43 | **32** |
| total collective bytes | 11.380 GB | **9.448 GB** (−17.0 %) |
| `jit(_identity_fn)` module total | 8 447 058 752 B | **6 514 756 160 B** |
| `s32[2304,36,36,135]` loader index gather | 1 612 431 360 B | **gone** |
| `c128[2304,8603]` τ-phase gather | 317 140 992 B | **gone** |
| steady-state r-chunk wall | 79.7 s | 78.5 s (unchanged, as intended) |

The delta is `1 932 302 592 B` — **to the byte** the two named gathers plus the six
small tables. The same comparison at **P=64** (`c_606_P64_rep` → the third cell,
606 centroids): `jit(_identity_fn)` 20 889 990 400 → **13 132 468 480 B**, a delta
of **7 757 521 920 B = 7.76 GB/rank**, again byte-exact against the closed forms;
43 → **32** collective instructions and 24.398 → **16.640 GB** run-wide (−31.8 %),
at an unchanged wall (59.39 → 59.15 s/chunk). Extrapolating the two measured
points gives **17.5 GB/rank returned at P=144**: the term this section called
"the dominant memory term exactly where the campaign is heading" is simply
deleted, for zero numerical change. After the fix the biggest collective in a
ζ-fit is `F_tensor_write`'s ζ writer gather at every size — anomaly 2, which
remains the top item.

**Anomaly 1 (`per_q` does not do the per-q gather) — closed, 144×.** The cure is to
take the q-slice INSIDE a `shard_map`, where the two `lax.all_gather`s that rebuild
the `(1, μ, μ)` tile are written on an already-sliced operand and there is nothing
left for the partitioner to hoist. Optimized HLO of `jit(_solve_one_q_and_update)`,
276c / P=16:

| | before (`a_P16_per`) | after (`AA_P16_per_after`) |
|---|---|---|
| the module's collectives | `c128[144,72,288]` + `c128[144,288,288]` | `c128[1,288,72]` + `c128[1,288,288]` |
| **gathered bytes per execution** | **238 878 720** | **1 658 880** — 144.0×, exactly nq |
| module `Total bytes used` | 753.49 MiB | **529.70 MiB** |
| gather traffic per r-chunk (×144 execs) | 34.4 GB | **239 MB** ≈ `replicated`'s 238.9 MB |
| **back-solve wall** (steady-state chunks) | **470.4 s** | **36.9 s** |
| chunk wall | 510.9 s | **78.8 s** |

So the sharpest number in the tradeoff table — "`per_q` is a memory escape hatch and
nothing else, 12–40× slower" — **no longer holds.** The tier is now within 2 % of
`replicated` on wall (36.9 s vs 36.1 s back-solve) while its live gather is `nq`×
smaller. At the production point this section settled (μ_pad = 2048, nq = 144, 8×8)
the per-execution gather becomes `μ²·16·(1 + 1/p_y)` = **75.5 MB**, against the
**10.87 GB** measured above. The advice changes accordingly: on a **square** mesh
`distributed` is still fastest (19.6 s at P=16/276c), but on a **rectangular** mesh
`per_q` is now a free memory win instead of a 12–40× penalty — and T.7's residency
table returns to ≈0.1 GB of back-solve arena at c2406/P=144 rather than the 14.96 GB
computed from the defeated form. The rewrite is bit-identical: `per_q` vs `auto` on
the P=4 fixture gives `max|Δ| = 0.00e+00`, and P=8 vs P=4 likewise.

**Anomaly 5 (planner accuracy) — both causes found; the writer term was sizing the
wrong tensor.** The 1998c death is `nq·μ_pad·ngkmax·16` (the G-flat ζ write); the
model only carried `nq·μ²·16` (the V/W write). `F_t` is now
`2 · max(V_tensor, gflat_tensor)`, which reproduces the byte-exact 40 594 046 976
allocation (the `2 ×` being the gathered device buffer plus `process_allgather`'s
closing host copy) instead of under-sizing it 2.10×. A second, P-INDEPENDENT
persistent term covers the loader tables the anomaly-3 fix leaves resident. Effect
at the re-run cells: `F_tensor_write` goes from absent-from-the-top-4 to
**14.09 GB/dev** at 276c/P=16, and at 606c/P=64 it **becomes the binder at
31.16 GB/dev** where the old model reported `A_centroid_load` at 10.33 GB/dev.
`planner/measured` at that cell therefore goes **0.48× (under) → 1.45×
(conservative)**, and **every point in the matrix is now conservative** — the
1.54 → 1.11 → 0.98 → 2.03×-under drift this section reported no longer exists.
(The measured node peak at 606c is unchanged at 43 GB/node: the ζ-writer gather,
not the loader gather, is what sets it, and AA models that term rather than
removing it — removing it is the `PHDF5_HOST` item, still anomaly 2.) Chunk
selection and wall are unchanged (`band_chunk` / `r_chunk` / `q_chunk` identical
before and after at both points): this makes the planner honest, not cautious.

Anomalies 2, 4 and 6 are untouched by AA and stand exactly as written above.
