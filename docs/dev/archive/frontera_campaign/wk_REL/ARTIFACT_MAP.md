# ARTIFACT MAP — LORRAX Frontera campaign institutional memory (built 2026-07-28)

Scope: /scratch2/08271/jackmc/lorrax_setup (+ $WORK/frontera). Read this before
adding artifacts; match the conventions in the final section.

## 1. Top-level layout

- `/work2/08271/jackmc/frontera/LORRAX_FRONTERA_ADVICE.md` — the ops playbook (1104 lines, summarized in §2 below).
- `/work2/08271/jackmc/frontera/lorrax` — main checkout, branch `fix/zq-band-gather-device-invariance` (Frontera work; `main` = Perlmutter-pristine). Worktrees `wt-*` beside it. Handoff doc IN-repo: `docs/dev/HANDOFF_cpu_frontera_2026-07.md`.
- `/scratch2/08271/jackmc/lorrax_setup/` — campaign hub: SPEEDUP_SCORECARD.md (9475 lines, the ledger), LORRAX_CONTEXT_BRIEF.md (onboarding), SESSION_REPORT_2026-07-{25,27}.md, py312.sif container, holder infra (`alloc_run.sh`, `current_holder_jid`, `holder40.sbatch`, `hold40.*`), ~200 `<tag>.<jobid>.out` job logs + their `.sbatch` files, `bs_groundtruth_meshless.dat` (htransform gate truth), `multidev_run.sh`/`multidev_compare.py` (GPU ladder), `aq_c5000_driver.sh` + `aq_c5000_logs/` (AQ 5000c kmeans; ACCEPTED count 4962).
- `/scratch2/08271/jackmc/mos2_4x4_test/` — AJ's 4×4/30 Ry fast deck (dev proxy; REGENERATE ~4.5 min; `gw800_merged.sbatch`, `aq_rehearsal.sbatch`). `/scratch2/08271/jackmc/mos2_80ry_12x12/` — production WFN deck. `/scratch2/08271/jackmc/lorrax_mos2_12x12/` — production run dirs (one dir per run).

## 2. ADVICE file section summaries (LORRAX_FRONTERA_ADVICE.md)

- §0 TL;DR: single-node single-process CPU is the default; 16 GB RTX GPUs OOM 12×12; compile cache now SAFE at P>1 (buys ~1% CPU, 3.6× GPU); use venv python explicitly; `memory_per_device_gb=90`.
- §1 Paths: repo/venv/container/allocation `PHY25006`; container REQUIRED (glibc 2.17 host); apptainer blocked on login nodes.
- §2 /home1 200k INODE quota locks you out — never let caches land in $HOME.
- §3 Launch: `module load tacc-apptainer`, bind /home1,/work2,/scratch*; `--nv` GPU-only; `development` = 2 concurrent jobs max; node sweet spot 4–8, not max.
- §4 Compile cache: fixed by AH (shared process-invariant key + coordination-service agreement, atomic writes); leave ON, point at $SCRATCH; "prefer-no-gather" lines are log noise; compilation is NOT a bottleneck on CPU.
- §5/§5b Memory is the wall: planner under-counts CPU peak ~40–50 GB (set budget 90); centroid ψ load used to GROW with world_size (fixed by process-local load).
- §6 Multi-node startup: `CUDA_VISIBLE_DEVICES=""` + explicit `JAX_COORDINATOR_ADDRESS` (per-launch port) → 18 s; `GLOO_SOCKET_IFNAME` is INERT with jax.
- **§6a ⚠ NEVER `distributed_cholesky=off`** — bypasses the rank-truncation cure, run "succeeds" with garbage physics (QP gap −161 eV). Use `auto`; ALWAYS `grep "Computing L_q"` for `path=replicated_rank_truncate`; cap knob `LORRAX_ZETA_REPLICATE_CAP_GIB`.
- §6b Generate `dipole.h5` (multi-node) and `kin_ion.h5` (`gw.kin_ion_io`, NOT psp.get_DFT_mtxels) BEFORE long GW; `restart=true` reuses ζ. §6c `process_allgather(tiled=True)` multi-host bug class. §6d ONE RUN PER DIRECTORY (`tmp/zeta_q.h5` collision).
- §7 QE prep (`-ndiag 1`, wfn2hdf). §8 input knobs: `slab_io`, budget 90, nband ≳ n_centroids; keys must be inside `[cohsex]`.
- Timings/cost ladder: dev-iterate ONLY on the 4×4 deck (183 s P=4); 12×12 is a ~1.5 h 4-node job min; do wall arithmetic vs the 2 h dev limit (ζ written only at fit END).
- §9 GPU multidev ladder: cuSOLVERMp needs SQUARE mesh; Bug A (per-rank cache deadlock, superseded by AH) + Bug B (Z_q band all_gather remainder-chunk misalignment — the campaign's marquee P>1 correctness bug, fixed + `test_zeta_mesh_invariance` gate).
- §10 CPU beats RTX ~3× for FP64; 2 ranks/node × 28 threads via `taskset` (XLA_FLAGS intra_op flag F-ABORTS — does not exist); dev loop = ssh into holder node, `srun --overlap --jobid`.
- §10b Gloo ran on em1 (1 GbE) ALL campaign; `runtime.pin_gloo_interface()` auto-pins ib0 (`LORRAX_GLOO_IFNAME` override) → ζ-fit 9.6×, Σ 4.6×, pipeline 6.5× at 606c/P=80, bit-identical. ⚠ CLAIM-DECAY: every pre-07-27 collective wall is an em1 number.
- **§10c Intel-MPI: DELETE `FI_PROVIDER=tcp`** (rtx workaround carried by mistake) → native `mlx`/UCX (1.07 µs vs 10.9 µs; pzheevd 12 s/q → 0.5–0.9 s/q). NEVER `--bind /dev` (breaks device nodes); stage host rdma/UCX libs per wk_AS recipe; dial = `LORRAX_MPI_PROVIDER=auto`; trust only the `I_MPI_DEBUG=4` provider banner. JAX collectives can ride MPI (`impl=mpi`: needs THREAD_MULTIPLE-patched MPIwrapper + `LORRAX_MPI_FINALIZE_FIX=skip_atexit`; 1.18× vs gloo/ib0 at P=16). Gloo/ib0 stays the certified default.
- §10d Per-variable transport audit (AU): `FI_PROVIDER_PATH` is REQUIRED in-container; TACC UCX module vars must be setdefault'd (2× at 1 MiB allreduce); login env exports `FI_PROVIDER=mlx` — the auto-unset is load-bearing.
- **§11 Parallel-agent workflow**: 40-node holder + `alloc_run.sh` (`srun --overlap --jobid` from INSIDE a holder node; script/`-m` only, never `python -c`); one git worktree per agent under /work2, orchestrator merges; compile storm = rank replication (~138 modules/rank, size-invariant).

## 3. CONTEXT_BRIEF digest (lorrax_setup/LORRAX_CONTEXT_BRIEF.md)

LORRAX = JAX GW-BSE (ISDF ζ-fit → GN-PPM Σ → htransform → BSE), FP64/c128,
2D ('x','y') mesh via shard_map. CPU nodes are the production target.
Repo map: wfn_loader (eager/phdf5/phdf5_host backends), psi_G_store,
wfn_transforms, htransform, gw_jax/isdf/core (solver-kind resolvers),
ffi/common/dispatch (eigh backends), bse/, runtime.bootstrap().
Test infra: alloc_run.sh + holder; cohsex_debug fixture (60c, eqp_ref gate
atol 1e-4 rtol 1e-6); bs_groundtruth_meshless.dat (max|Δ|<1e-8).
Rules: bit-exact physics; edit only in your worktree; do NOT commit — orchestrator merges;
log measured wins to SPEEDUP_SCORECARD.md under your workstream letter.
⚠ STALE: brief still says cache disabled `ISDF_JAX_CACHE_DIR=""` — superseded by AH (leave ON).

## 4. SPEEDUP_SCORECARD.md — structure + sigma-relevant sections

9475 lines, append-only. `## <LETTER> — <one-line headline>` per workstream, then
`### <L>.<n>` subsections. Letters used: A–Z (phase 1/2), AA–AX (07-26/27).
Sigma-relevant digests:

- **AK (7371) — GW forensics + sigma pad**: Σ τ-loop code is BYTE-IDENTICAL to July-25 (no introduced regression); Σ does not strong-scale — 457 MFLOP/s/core on 1 node collapses to 68 off-node (em1 floor; 4 psum_scatters per τ = re/im × x/y). AK.6c: `pad_sigma_window` over-padded to the p_x·p_y PRODUCT — per-axis fix = −12.4% sigma.exec at 8×10, avoids 3.2–4.0× latent tile waste on square meshes. AK.9 named-not-done: halve 4→2 psum_scatters/τ (stack re/im); `_to_host_np(sigma_kij, tiled=False)` = P-independent ~237 MB/branch gather onto EVERY rank (grows nb²); V_q never re-A/B'd. Sharded W solve (AD.4) verified 5.5× at production μ.
- **AC (5001) — flagship 2406c/P=144 recovery**: `distributed_zeta_solve=distributed` did NOT survive P=144 (Gloo ReduceScatter death in C⁺ formation — transport not memory; em1-scoped) → went out on `per_q` (283 s/chunk flat, 71 min ζ projected). AC.3b: `zeta_q.h5` size test is a TRAP under PHDF5_HOST. AC.4: the post-ζ (screening/Σ/eqp) restart gating harness — the pattern for Σ-only work from `isdf_tensors_*.h5`.
- **AP (8294) — pzheevd/eigh**: the 30-min P=144 eigh was `FI_PROVIDER=tcp` — provider unset (mlx) gives 0.91 s/q (0.49 with NB=64); provider×block-size interact (verbs+one-block pathological at P=144). Local MKL zheevd (0.61 s, 657 GF/s) is 11.5× the `jnp.linalg.eigh` the replicated route runs. Two-plan memo (wk_AP/DESIGN_MEMO_zeta_eigh_two_plans.md): Plan A q-parallel local eigh (μ≤~8k, ~5 s); Plan B tuned 2D pzheevd for 50k+.
- **AW (8874) — MKL-thread cliff**: harness-wide `MKL_NUM_THREADS=28` is catastrophic INSIDE ScaLAPACK handlers at P≥64 (12×12 grid: 11.28 s/q @14 thr → 0.463 @4 = 24×) — a SECOND independent cause of AC.2's 30-min eigh; fixed in-handler (`MklThreadScope`, cap 4, `LORRAX_SCALAPACK_MKL_THREADS` override); harness 28 must STAY (plan-A local route needs it). Invisible on every 4×4 gate — scale-dependent pattern.

Other sigma-adjacent anchors: AL (Σ 288→62 s at 606c/P=80 on ib0; GW/ζ=0.53
marginal FAIL, remaining gap = structural Σ comm), AK.2 (owner invariant
GW ≤ 0.5×ζ), J.7 (σ band-window mesh pad + restart-window guard — restart with
a changed window silently produces garbage), AD (sharded W, ONE eqp assembly,
GN-PPM coverage), AN (two W Dyson plans), AS.7 (certified AQ launch env),
wk_REL/results/sigma_perf_candidates.json (see §5).

## 5. Per-workstream dirs (lorrax_setup/wk_*) — one line each; [σ] = sigma-relevant measurements

- wk_AA — loader all-gather/device_put fixes + planner F_tensor_write; gate.7875720.out.
- wk_AB — PHDF5_HOST env build (mpi4py + HDF5_MPI h5py); CUTOVER.md, scorecard_AB.md.
- wk_AC [σ] — flagship recovery: runAC.sbatch (production template), stage_table.py (log→stage-table reducer), POSTMORTEM_7875551.md, watch.sh, runs/.
- wk_AD (+_gpu,_scalapack) [σ] — sharded W solve + eqp unify + GN-PPM rung; gate_[a-e].sbatch/.out; scorecard_AD.md.
- wk_AE (+_gpu) — phdf5 WRITE core ported to host FFI lib.
- wk_AF — distributed-ζ payload chunking, P=144 gate; scorecard_AF.md.
- wk_AG — rtx centroid-load hang root cause (P>1 compile-cache deadlock).
- wk_AH — shared compile cache repair; scorecard_AH.md; 104 files of gates.
- wk_AI — restart-tensor writer 1.7 MB/s→8.2 GB/s (collective + striping); scorecard_AI.md, wbench.py.
- wk_AK [σ] — GW forensics: stage_diff.py, sigxc_cmp.py, ib_gloo_run.py, gate_ak/gate_cohsex.sbatch, A/B run dirs.
- wk_AL [σ] — ib0 pin gates; gate40_ib0.sbatch is the 40-node template.
- wk_AM — defaults alignment (slab_io=auto) gates.
- wk_AN [σ] — two W Dyson plans; run_AN_* on the 4×4 deck, HLO collective tables.
- wk_AO — centroid-load collective hygiene; reshard-92.6s verdict.
- wk_AP [σ] — provider benches: pz_bench.c, matrix{1..5}.sh, apbench.sbatch, DESIGN_MEMO_zeta_eigh_two_plans.md, AP_scorecard_draft.md, logs/.
- wk_AR — jax upstream issue/PR drafts (Gloo interface selection).
- wk_AS — in-container comms certification: as*.out, mpiw_thr_install (THREAD_MULTIPLE wrapper), sitedir (finalize fix), staging recipe as_inner.sh.
- wk_AU — transport env A/B cells (logs/); harness consolidation.
- wk_AV — LORRAX knob audit (COLLECTIVE_CHUNK_MB verdict).
- wk_AX — bispinor propagation runs + eqp_stats.py.
- wk_BC — exciton-CPU workstream. wk_D/wk_E/wk_G/wk_H/wk_I — phase-2 workstreams (D=htransform compiles, E=linalg backend, G=infra audit, H=test health, I=linalg facade); wk_D-style dirs are cohsex_debug fixture COPIES used as run dirs (§6d rule).
- wk_ENV [σ] — AT/AV/AW env audits: A{T,V,W}_harness_spec.md, aw_mkl_matrix*.{sh,out} (the MKL cliff data), pzlu_bench.c, aw_gates out.
- wk_J [σ] — μ-replication audit + J.7 σ-window runs (sgOdd*/sgP4 = sigma window gates; a2a*, gn*, gw*, ht* run dirs).
- wk_L — SLATE + first host FFI lib build. wk_M (in scorecard; artifacts in run dirs) — 12×12 H0/VH corruption root cause.
- wk_N — exact V_H folded into kin_ion gates. wk_P — loose ends P.1–P.11. wk_Q — nosym/TRS loader-bug discriminating measurements. wk_R — 12×12 QP bandstructure.
- wk_S — V_H source (stored|isdf|gspace) convergence runs. wk_T — leak-hunt (glibc trim). wk_U — WFN symmetry measurement. wk_V [σ] — ScaLAPACK pzheevd backend + distributed ζ tier (gate_*.txt, audit_hlo.py). wk_X — G-space V_H strong scaling (scorecard_X.md). wk_Y — standing HLO/timing probe harness. wk_Z — recompile-hazard audit + linalg PLAN API.
- wk_gateO{,2,3}, wk_eqpO_legacy, wk_merge{,2}, wk_repro80 — O-hardening / merge-verification / P=80-repro gate-run fixture copies.
- **wk_REL (this dir)** — current release/size-campaign workstream: SIZE_CAMPAIGN_BRIEF.md (owner-directed size escalation, 32 nodes/P=64 8×8, SEQUENCED AFTER sigma-perf), sigma_perf_candidates.json (ranked σ candidates — #1: replace the 16-pt flat-k FFT round-trip in ppm_tau_kernel with a DFT-matrix GEMM, est. 25–45% of sigma.exec at μ=4962/P=64), audit_findings.json (e.g. Gloo-pin warnings rank-0-gated), rtx/verify/bisect/pytest gate scripts + outs.
- Letters without a wk dir: AJ → mos2_4x4_test; AQ → aq_c5000_* + mos2_4x4_test/aq_rehearsal.sbatch; AT/AV/AW → wk_ENV.

## 6. Session report digests

SESSION_REPORT_2026-07-25.md (948 lines):
- Headline: first trustworthy 12×12 physics — DFT 1.7010 eV → QP gap 2.6475 eV at 606c/40 nodes, after fixing (among others) the nosym ψ*(−r) WFN-unfold corruption that invalidated EVERY historical 12×12 artifact.
- Three merged refactor waves (CPU enablement/correctness, unification, linalg facade), all bit-exact gated.
- The run1 failure ladder v1–v7 is the canonical debugging narrative: bogus XLA flag, 271 GB unsharded lim-P→∞ intermediate, V_q allgather 160× blowup, σ-window divisibility, restart-with-changed-window garbage.
- Memory model calibrated (2.02×→1.22× vs reality); per_q back-solve made competitive (470→37 s).
- Conditioning datum: n_keep=276/276 at μ=276 (zero truncation).

SESSION_REPORT_2026-07-27.md (165 lines):
- First complete e2e GW at 2406c/P=144: 45 m 35 s rc=0, QP gap 2.7271 eV (+80 meV vs 606c, μ-convergence from below); isdf_tensors_2406.h5 restart-usable.
- Gloo was on em1 ALL campaign → ib0 pin landed: 3.9× total at 606c/P=80, byte-identical five ways; the "ζ blocker" was an em1 artifact, no code regression.
- Restart writer 1.7 MB/s → 8182 MB/s (collective MPI-IO + striping hints; lfs absent in container).
- Owner invariant: ib0-native GW/ζ = 0.53 marginal FAIL; next levers = Σ psum_scatter halving, `_to_host_np` gather.
- Env-audit round (AT/AU/AV/AW) + bispinor (AX) merged; branch @ 9f20cb6, ~52 local commits, NOT pushed.

## 7. REPORTING CONVENTIONS (match these exactly)

1. **The ledger is SPEEDUP_SCORECARD.md** (`/scratch2/08271/jackmc/lorrax_setup/`). Each workstream APPENDS one `## <LETTER> — <one-line headline with the verdict>` section (letter assigned by the orchestrator; next free letters follow AX). House style, in order: `### <L>.0` context/root-cause; numbered `### <L>.n` subsections with MEASURED tables (jobids inline, e.g. "job 7876530"); a GATES subsection (bit-identical eqp0/eqp1/eqp_g0w0/sigma_diag comparisons, or eqp vs eqp_ref.dat max|Δ|=1e-6 eV, stated PASS/FAIL); `### <L>.n — files touched (worktree only, NOT committed)`; and a closing `### <L>.n — named, not done`. Corrections to OLD sections are made in place with `> ⚠ CLAIM-DECAY` blockquote banners (never silently edited).
2. **Per-workstream dir** `lorrax_setup/wk_<LETTER>/` holds: gate sbatch scripts (`gate_*.sbatch`, `*_inner.sh` container-side pair), job logs named `<tag>.<jobid>.out`, analysis/one-shot scripts (`*_cmp.py`, `stage_table.py` pattern), and any DESIGN_MEMO_*.md / POSTMORTEM_<jobid>.md / scorecard_<L>.md draft. JSON deliverables (candidate lists, audit findings) also live here.
3. **TEST RUNS**: batch via sbatch (dev queue, 2-job cap) writing `<tag>.%j.out` into the launching dir; interactive steps via the shared holder (`alloc_run.sh`, jobid in `current_holder_jid`). One RUN per DIRECTORY (§6d); production runs get their own dir under `/scratch2/08271/jackmc/lorrax_mos2_12x12/` or `mos2_4x4_test/run_*`.
4. **Gate discipline**: every claimed win must carry (a) a physics gate (byte-identical or ≤1e-6 eV eqp), (b) the route check (`grep "Computing L_q"`), (c) jobids, and (d) an honesty note on co-tenancy/noise if the holder was shared. Cross-cutting corrections also get edited into LORRAX_FRONTERA_ADVICE.md ("CORRECTED <date>" boxes).
5. **Code**: edits stay in the agent's worktree/branch, NOT committed (unless the section header says COMMITTED on branch); the orchestrator merges. Session-level rollups go in SESSION_REPORT_<date>.md.

## 8. DECK LINEAGES in mos2_4x4_test (size campaign; added 2026-07-29)

Two INDEPENDENT physical calculations now live in the same directory.  They
are distinguished purely by filename tag — nothing is shared except the .upf
pseudopotentials and the harness scripts.  **Never gate one against the other**
(owner ruling 2026-07-29: a different cutoff is a different calculation).

### Lineage A — 30 Ry (ecutwfc 30 / ecutrho 120), FFT 24x24x80 = 46080
One SCF (`scf.in -> out/`), then per-nbnd staged-density NSCFs.  Staging the
density is legitimate ONLY because the cutoff is fixed within the lineage.

    nbnd  outdir        WFN                 vxc/kih/RHO tag   deck descriptor
    128   out/          WFN.h5              (untagged)        deck.in
    256   b256_out/     WFN_b256.h5         _b256             deck_b256.in
    512   b512_out/     WFN_b512.h5         _b512             deck_b512.in
    1024  b1024_out/    WFN_b1024.h5        _b1024            deck_b1024.in
    per-nbnd: kin_ion_<tag>.h5, dipole_<tag>.h5, centroids_<tag>_c<N>.txt
Band axis TERMINATES at 1024 here (QE npwx_g ceiling; see brief).

### Lineage B — 45 Ry (ecutwfc 45 / ecutrho 180), FFT 25x25x100 = 62500
Its OWN SCF (`scf_r45.in -> out_r45/`) — a full regen, not staged from 30 Ry.

    nbnd  outdir            WFN                  tag          deck descriptor
    1024  r45_b1024_out/    WFN_r45_b1024.h5     _r45_b1024   (reference only)
    2048  r45_b2048_out/    WFN_r45_b2048.h5     _r45_b2048   deck_r45_b2048.in
    centroids_r45_b2048_c<N>.txt  -- MUST be regenerated on the 62500 grid;
    reusing any 30 Ry centroid file is INVALID.
`r45_b1024_{a,b,c}_out/` are the parallelization-probe outdirs; the winner is
symlinked as `r45_b1024_out`.

### Harness conventions added this session
- `wk_REL/srcpin_<sha>/` — read-only `git archive` of the lorrax repo, used as
  PYTHONPATH by every batch job so a moving working tree cannot perturb a run.
- Ladder GW templates are RUNG-PARAMETERIZED; pass rung knobs at submit:
  `sbatch --export=ALL,CENTFILE=...,TAG=...,WEAPONS=on|off,SHARDED=on|off,RCHUNK=N`
  (`l7_b1024_bigmu.sbatch` 30 Ry, `l6_r45_b2048.sbatch` 45 Ry,
  `diag_b512_weap.sbatch` the rung-4-size control).
- `wk_REL/harness/qpgap.sh <run_dir> [nval]` — the QP indirect-gap extractor, its
  convention CERTIFIED by reproducing ladder rungs 3 and 4 from their own eqp
  files (3.4594/3.1603 and 3.2194/2.9778).
- QUEUES: QE work -> `small` (2 nodes, 48 h, cap independent of the GW ladder);
  P=64 GW ladder runs -> `normal` (the `development` 2 h wall and its 2-job cap
  shared with the concurrent GEMM/BSE workstream both bind at ladder sizes).
  The ARCHITECTURE is unchanged in every case: P=64, 8x8 mesh, 32 nodes x 2.

### LOCAL QE TOOL PATCHES (outside the lorrax repo; both announced)
| file | line | change | rebuilt | backup |
|---|---|---|---|---|
| PW/src/memory_report.f90 | 486 | `npwx_g*npol < nbndx` | 07-28 21:24 | — |
| PW/src/c_bands.f90 | 308 | `nbndx > ipw*npol` | 07-29 01:40 | `c_bands.f90.orig_prepatch_20260729` |
Both correct the same npol-blindness (spinor state dim is npwx*npol).  Neither
computes anything — both are pure guards.  30 Ry artifacts predate the 01:40
rebuild; the 45 Ry b1024 reference predates it by 4 minutes, which is what makes
it a valid independent gate for the second patch.
