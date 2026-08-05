# LORRAX CPU-adaptation / perf scorecard

## PHASE 2 (2026-07-25 pm): maintainability + unification audit
Question under audit: are the CPU changes maximally shared with GPU infra (not
duplicated), and is htransform's WFN reading at parity with gwjax's streaming
(PsiGStore + AsyncWfnReader io_callback/scan pattern) and psp/dft readers?
Workstreams: F=wfn-read unification (wt-F), G=cpu-infra maintainability/consolidation
(wt-G), H=tests-green + perf at current scale (wt-H). Holder 7874023 (2h), runner
alloc_run.sh unchanged. Known consolidation candidates spotted in prep: wfn_loader has
THREE host build paths (_eager_build, _eager_build_process_local, _phdf5_host_build);
7 CLIs carry a duplicated 6-line bootstrap header; htransform streams ψ synchronously
per chunk while gwjax overlaps read/compute.

Goal: make GPU-targeted LORRAX paths run well on multi-process CPU, kill XLA
compile storms, fix CPU bugs. Track quantified wins here.

## Baselines (measured earlier this effort)
- htransform G-accumulation (big system nk=144,nb=70,nr=174960,n_mu=276,4x4 mesh):
  ~158s, **2208 XLA compiles**. Root cause: eager per-k/per-chunk jnp-op dispatch
  (each un-jitted op re-compiles per chunk) + persistent compile cache disabled
  (P>1 deadlock) + WFN loader was mesh-LESS → replicated ψ per rank.
- Small fixture (WFNsmall, nk=9,nb=8,1 chunk, 2x2 CPU mesh) eager htransform:
  26s wall, 131 compiles.

## Changes landed (this session, branch fix/zq-band-gather-device-invariance + WIP)
| # | Change | File(s) | Status | Measured effect |
|---|--------|---------|--------|-----------------|
| 1 | phdf5_host WFN backend: host h5py union read + shared on-device vectorised unfold kernel (band-sharded, compiles unfold ONCE via lru_cache). CPU twin of the GPU phdf5 FFI, zero new build. | src/file_io/wfn_loader.py | **VALIDATED** | bit-exact vs ground truth (world=4 real-sym WFN, full unfold). htransform wall 23→16s (~30%). Compiles ~unchanged (read is not the storm). |
| 2 | Wire device mesh into htransform loader so ψ is band-sharded (phdf5_host on CPU / phdf5 FFI on GPU) instead of replicated on every rank. | src/bandstructure/htransform.py | **VALIDATED** | mesh-ful eager (§5b) == pre-change mesh-less bit-exact → to_rchunk handles band-sharded ψ; no GPU-path regression expected (falls to phdf5 FFI). |
| 3 | LORRAX_WFN_BACKEND env override + auto-pick: mesh-less loader always eager (safety); multi-proc CPU → phdf5_host. | src/file_io/wfn_loader.py _auto_pick_backend | **VALIDATED** | pytest 3-passed; fixes crash where env forced a mesh-requiring backend onto htransform's metadata loader. |

### Measured (WFNsmall, 2x2/4-rank CPU, 1 band chunk)
- eager: 23s wall, 132 compiles, G-accum 0.83s
- phdf5_host: 16s wall, 140 compiles, G-accum 0.55s
- **3-way correctness: max|Δ energy| = 0.000e+00 (bit-exact).**

## BSE audit findings (agent) + fixes
| # | Finding | File | Status |
|---|---------|------|--------|
| B1 | Production BSE CLI `bse_jax.py` lacked the CPU-fallback bootstrap (`set_default_env` + `fallback_to_cpu_if_no_gpu_backend`) → dies "Unknown backend: 'gpu'" on CPU-only node if launcher sets JAX_PLATFORM_NAME=gpu. | src/bse/bse_jax.py | **FIXED** (mirrors exciton_bands bootstrap; behavior-preserving on GPU) |
| B2 | Same missing guard in tool CLIs bse_kpm/bse_feast/bse_w_exact/davidson_absorption. | src/bse/*.py | TODO (low-risk batch) |
| B3 | `absorption_haydock.py:158` hard-requires ≥2 devices → blocks single-proc CPU smoke. Gate on process_count. | src/bse/absorption_haydock.py | TODO |
| B4 | `--eigh-backend cusolvermp` GPU+square-mesh only; default `auto`→jnp.linalg.eigh is CPU-safe. Document/guard. | ffi/common/dispatch.py | note only |
| B5 | `w_omega_chain.py` per-iteration device_get sync (perf, W-tooling not production). | src/bse/w_omega_chain.py | TODO |

BSE core is healthy: the 3 hot paths (stack-matvec Lanczos, Haydock, exciton lax.scan)
are each single-jit, CPU-portable, no compile storm; sharded loader is correct.

## Exciton bandstructure / finite-Q interpolation audit (agent)
Path (bse/exciton_bands.py) is ALREADY hardened: per-Q solve is ONE lax.scan (one
compile serves all Q), eval_vq takes Q as runtime arg (one compile), compute_wfns_fi
batches q at fixed shapes, ALL loaders get mesh_xy (no replicated-loader anti-pattern).
No compile storm here. Findings are opt-in CPU-breaks + replicated-memory + host loops:
| # | Finding | File | Status |
|---|---------|------|--------|
| X1 | `--eigh-backend cusolvermp/slate` GPU+square-mesh only; reject on CPU with clear msg (default auto=jnp.eigh is safe). | exciton_bands.py / dispatch.py | TODO |
| X2 | `fallback_to_cpu_if_no_gpu_backend` only matches "Unknown backend: 'gpu'", not "Unable to initialize backend 'cuda'" — infra-wide CPU-run gate. Broaden CAREFULLY (risk: masking real GPU errors). | runtime/__init__.py:226 | needs runtime confirm |
| X3 | `--head-minibz-average`: per-Q host Sobol QMC (2^18×10) in Python loop, run on every rank, not rank-0 gated. | exciton_bands.py:653 | TODO (opt-in) |
| X4 | `vq_interp.build_cq` returns full (nq,nμ,nμ) c128 to host on every proc (~13GB at nq=144,nμ=2412); diagnostics default ON (LORRAX_SKIP_VQ_GATES=1 to disable). | vq_interp.py:390,969 | TODO (mem@scale) |
| X5 | psp/finite_q_head_interp.py = off-path serial numpy PoC; do NOT wire into production (use sharded head_wing_schur). | psp/finite_q_head_interp.py | note only |

## MERGE STATUS — all 4 workstreams merged into feature branch (971baef)
Committed each worktree branch and merged into fix/zq-band-gather-device-invariance:
D bc58cc1 (ff) → E 0c25ae7 → A 50ba500 → BC 971baef. All clean, no conflicts (D+E
htransform.py overlap auto-merged: file has both band_pad_to ×2 and set_default_env ×6).
All merged files py_compile OK. Consolidated verification in progress (merged htransform
bit-exact vs ground truth + zeta/loader suite). NOT pushed yet.

## Shared test infra
- 40-node dev holder: job 7873900 (nodes in hold40.nodes.7873900), ~2h wall.
- Runner: `alloc_run.sh <N> <tasks/node> <src> <workdir> python -m mod|-u script.py args`
  (ssh→srun --jobid into the holder; standard CPU jax.distributed env; verified
  RANK 0-3/4 across 2 nodes). Use a SCRIPT/-m, not inline -c.
- QOS: qdevelopment=2 jobs/40 nodes; qnormal=75 jobs/1280 nodes (overflow).

## Workstream E — DONE (wt-linalg-backend), pending my verification
- Finding: switchable input-file linalg keys ALREADY exist (distributed_cholesky,
  distributed_lu, eigh_backend; auto default, CPU-safe). No redundant key added.
- Fixed the ONE real CPU hole: isdf/core._resolve_solver_kind_transverse LU `auto`
  returned cusolvermp_lu on a CPU 2D mesh → guarded with not _mesh_is_cpu (mirrors
  charge resolver). Explicit user choice left unguarded.
- Zeta test made backend-aware (sharded_cholesky on CPU / cusolvermp on GPU).
- Bootstrap unified across 6 CLIs (bse_kpm, bse_feast, bse_w_exact, davidson_absorption
  = none→canonical; absorption_haydock, htransform = partial→canonical).
- fallback_to_cpu broadened to match "Unable to initialize backend 'cuda|gpu|rocm'"
  but GATED on _gpu_is_present() (no /dev/nvidia* or CUDA_VISIBLE_DEVICES="") so a real
  GPU-node cuda-init failure is RE-RAISED, not masked. Good call.
- 9 files in wt-E. VERIFIED by me: 8 CLIs (bse_kpm/feast/w_exact/davidson_absorption/
  absorption_haydock/htransform/bse_jax/exciton_bands) import clean on CPU. Reviewed
  isdf/core LU guard + runtime fallback — correct. **I fixed a bug** in E's fallback:
  bare `raise` outside the except block → "No active exception to re-raise" on the
  real-error path; changed to capture `caught=exc` and `raise caught`.
- verify.py **RESULT: PASS** — LU auto→lu (fix), explicit cusolvermp respected,
  cholesky auto→sharded_cholesky, dispatch_eigh auto native (err 1.3e-15), cusolvermp
  cleanly rejected (no hang), 9 CLIs import clean, _gpu_is_present False. E READY TO MERGE.
  (Full zeta pytest = slow wrapper over logic verify.py already confirmed.)
- Merge deferred until agents done + /work2 git quiet (avoid hang on shared object store).
- MERGE NOTE: E touched htransform.py (bootstrap header) AND D owns htransform.py
  (compute) — different regions, should merge clean.

## Workstream BC — DONE + verified (wt-exciton-cpu), reviewed OK
- Task 1: minibz_head_vlr now rank-split with jax.random.fold_in(base_key, global_slot)
  → rank-count-INVARIANT (head=27.38911..., bit-identical nproc 1/2/4), physics
  single-sourced from gw/coulomb/base.py (minibz_voronoi_batches, _minibz_kernel_bare,
  minibz_average) + gw.vcoul.wrap_points_to_voronoi. process_allgather all-reduce.
  Each rank does 1/P of the QMC work. Signature unchanged (call site untouched).
- Task 2: build_cq drops the _to_host process_allgather (was 13.4 GB/proc at nq=144,
  n_μ=2412) → returns face-sharded P(None,'x','y') device array. Consumers updated:
  run_gates (gathers only C_q[0] diag slice), prepare_coarse (slice+hermitize on device,
  reshards face3→qb3 per chunk; numpy path preserved). Reviewed: single caller (:1088)
  → run_gates+prepare_coarse only, fully handled.
- Correctness (agent, real multi-rank): sharded spec (None,'x','y'), gathered value vs
  numpy ref relerr 1.7e-16; prepare_coarse consumer path vs numpy eigh 5.7e-14.
- Files: src/bse/vq_interp.py, src/bse/exciton_bands.py. No gw/ edits needed. READY TO MERGE.
- No file overlap with E. (D & E overlap on htransform.py — resolve at merge.)

## Workstream A — DONE + verified (wt-phdf5-ffi-cpu), reviewed OK
- SHARED-CORE refactor (the design the user wanted): one compile flag LORRAX_FFI_NO_CUDA;
  the collective MPI-IO read core compiles from the SAME TUs (context.cc, read_ffi.cc,
  api.cc) into both GPU + host libs. Only 3 seams switch: handler binding, index copy-in
  (cudaMemcpy D2H vs host read), device-staging tail (cudaMemcpyAsync H2D vs std::memcpy;
  cudaMallocHost vs aligned_alloc).
- Host FFI BUILT clean, CUDA-free (readelf: no libcuda; NEEDED = hdf5/mpi/stdc++/gcc/c).
  Caught+fixed a GPU-path bug (stale (void)stream) so the CUDA build still compiles.
- auto-pick: GPU CUDA FFI → else CPU host FFI (probes host lib for the phdf5 read symbol
  via hasattr) → else h5py twin (phdf5_host, DEMOTED to documented fallback). GRACEFUL:
  current deployed host lib lacks the handlers → falls back to twin → zero regression;
  rebuild with config/frontera/build_ffi_host.sh to enable the FFI read on CPU.
- Correctness (job 7874002, 4 CPU ranks 2x2, WFNsmall): auto→'phdf5' host FFI; k=ibz &
  full_bz FFI-vs-eager max|Δ|=0.0 bit-exact; FFI-vs-h5py-twin 0.0. Async ffi::Future
  union path works on XLA CPU runtime.
- Files: ffi/phdf5/cpp/{ctx.h,context.cc,read_ffi.cc,api.cc(new)}, ffi/common/cpp/{api.cc,
  ffi_helpers.h,CMakeLists.txt,host/CMakeLists.txt}, ffi/common/ffi_loader.py,
  ffi/phdf5/{context.py,read.py}, file_io/wfn_loader.py, config/frontera/build_ffi_host.sh.
  Host write path NOT ported (read-only host lib; fine — CPU only needs reads). READY TO MERGE.
- MERGE: A edits wfn_loader.py (auto-pick) — check D didn't also (told D to flag if so).

## Workstream D — DONE + verified (wt-htransform-compiles), reviewed OK
- MEASUREMENT corrected the hypothesis: per-k ngk is ALREADY uniform (loader pads to
  ngkmax, sentinel box_index, zero G-slot) — every heavy kernel compiles ONCE/rank.
  The 2208 = ~16 ranks × ~138 compiles/rank (per-rank count is problem-size-invariant:
  140/rank tiny ≈ 138/rank big). Dominant factor = SPMD rank replication, not shape.
- The ONE real residual: band-CHUNK remainder width (non-uniform last chunk → to_rchunk
  + _accum recompile). Fixed via band-axis uniform zero-pad (band_pad_to param), mirroring
  the ngkmax pad. ALSO fixed a latent shape-mismatch bug (UH_bc used bc while ψ had the
  round_up(bc,p_band) width). Padded bands × ψ zero-pad-bands = 0 contribution.
- Measured: forced [3,3,2] chunks → to_rchunk 2→1, _accum 2→1 compiles; Galerkin
  39.66→30.65s (**−23%**), bit-exact (max|Δ|=0.0 vs ground truth). Win grows with size /
  any non-uniform remainder. 2x2 uniform: no regression, bit-exact.
- Files: htransform.py (band_pad_to=_bc, UH_bc pad to w, accum key on w),
  wfn_transforms.py (iter_psi_rchunk_bandwise band_pad_to param). wfn_loader.py UNTOUCHED
  (no overlap with A). READY TO MERGE. Overlap w/ E = htransform.py (E header vs D compute,
  non-adjacent → clean).

## >>> NEXT BIG LEVER (D's key finding): the persistent compile cache <<<
2208≈16×138 is rank replication. With a SHARED jax_compilation_cache_dir, ranks 1-15 hit
rank-0's compiled modules (identical SPMD local shapes) instead of each recompiling ~138 →
the real path to killing 158s. BLOCKED by the P>1 compile-cache DEADLOCK (bug A: per-rank
cache diverges on hit/miss → cross-process compile barrier hangs; that's why runs set
ISDF_JAX_CACHE_DIR=""). Fixing that deadlock is the highest-value remaining storm work.

## Active parallel workstreams (agents, worktree-isolated)
- **D**: htransform ngk-uniform compile fix (variable #Gvecs/k → variable shapes →
  recompiles; apply padded-ngkmax + dynamic-slice zero-fill, the DFT-code pattern). STORM LEVER.
- **BC**: exciton — (1) unify --head-minibz-average to GW's minibz backend, rank-split + jax.random + psum; (2) build_cq sharded C_q(μ_X,ν_Y) from X/Y-sharded ψ (gwjax pattern), not host-gathered.
- **A**: phdf5 FFI CPU-enable via SHARED core (switch only device-staging GPU/CPU), retire the h5py twin to fallback.
- **E**: replace hardcoded cusolvermp with input-file-configurable switchable-linalg dispatch; unify jax gpu/cpu bootstrap across all CLIs.

## Test status (CPU, this branch)
tests/test_wfn_loader_eager.py + test_zeta_mesh_invariance.py: **18 passed, 1 skipped,
1 failed**. The 1 failure (`test_rank_truncate_refuses_above_the_replication_cap`)
hardcodes `cusolvermp_cholesky` (GPU-only dispatch) → fails on CPU regardless of my
changes = pre-existing CPU-incompat test, NOT a regression. All my new + touched tests pass.

### Key insight on the storm
132 compiles for a SINGLE band chunk ⇒ the storm is per-chunk recompilation of
eager (un-jitted) ops in to_rchunk (local FFT) + _accum + kpath, NOT the WFN read.
The big lever is wrapping the per-chunk hot path in ONE jit (stable shapes) so
chunks 2..N hit the trace cache, and/or re-enabling the persistent compile cache
(blocked by the P>1 deadlock). phdf5_host is a correctness+memory+modest-wall win,
not the storm fix.

## Open questions / next
- Measure phdf5_host vs eager htransform: compile count + wall time + correctness
  (3-way diff vs pre-change ground truth). Job 7873561.
- BSE core CPU/perf audit (agent a512...).
- Exciton bandstructure / Q-interp CPU/perf audit (agent a9a6...).
- The htransform compile storm is dominated by eager per-chunk jnp ops in
  to_rchunk/accum, NOT the WFN read — a separate fix (wrap hot loop in one jit /
  enable compile cache) is the bigger lever there.

## Workstream G — DONE (wt-G, audit-cpu-infra): CPU/GPU shared-infra audit + consolidation
Maintainability pass, behavior bit-exact (no perf claim). What landed:
1. wfn_loader.py: the "learn my band block / assemble sharded" scaffold is now
   written ONCE — new `WfnLoader._assemble_process_local` (cached
   `_sharded_zero_proto_fn` proto → `_local_shard_and_global_offset` → per-rank
   fill → `make_array_from_single_device_arrays`) used by BOTH
   `_eager_build_process_local` (§5b) and `_phdf5_host_build`; new
   `WfnLoader._kplan` single-sources the IBZ-union bookkeeping for
   `_phdf5_build` + `_phdf5_host_build`. Fixed the §5b re-lowering wart
   (fresh `jax.jit(lambda...)` per load → lru-cached proto factory).
   `_eager_build` stays the one numpy unfold core. Backend contract untouched.
2. CLI bootstrap: new `runtime.bootstrap()` (set_default_env →
   init_jax_distributed → fallback_to_cpu_if_no_gpu_backend; importable
   without jax). 9 CLIs converted to the 2-line header: bse_jax, bse_kpm,
   bse_feast, bse_w_exact, davidson_absorption, absorption_haydock,
   htransform, exciton_bands, gw_jax. NOT converted (deliberate, headers
   differ): run_nscf/run_sternheimer/kmeans_cli (no CPU fallback),
   get_dipole_mtxels (no set_default_env before import jax — pre-existing),
   profile_batched (init inside main).
3. FFI seam: `ffi_loader.has_target()/has_phdf5_read(platform)` — the
   symbol-probe now lives in ffi_loader; `_auto_pick_backend` no longer
   reaches into `_HOST_TARGET_SYMBOLS`, and the CUDA branch now also checks
   the union-read handler (a partial CUDA build previously auto-picked
   phdf5 and would have crashed at read time).
4. isdf/core.py: `_resolve_channel_ladder()` — the off/cusolvermp/explicit-
   backend/auto(CPU-guard) ladder shared by `_resolve_solver_kind_charge`
   and `_resolve_solver_kind_transverse`; both are now thin wrappers
   (slate/scalapack guards + charge replication-cap branch as hooks).
Gates (all on holder 7874023): tests/test_wfn_loader_eager.py 15 passed/1
skipped; test_rank_truncate_refuses_above_the_replication_cap PASSED (CPU);
htransform 2 nodes x 2 ranks (2x2 mesh) vs bs_groundtruth_meshless.dat:
max|dE| = 0.0 for BOTH backend=auto and forced LORRAX_WFN_BACKEND=eager
(the §5b path). py_compile clean on all 13 touched files. Not committed.

## Workstream F — WFN-read parity & unification audit (wt-F, branch audit-wfn-read-unify) — DONE
Audit verdicts (measured-fact based):
- htransform ADOPTING AsyncWfnReader: **NO** — psi_G_store.py:195 records overlap_frac=0.000
  (depth-2 prefetch, MoS2 3x3) and the big-system profile had load ~0.5s vs fft ~9s/print
  (read ≈5% of stream). AsyncWfnReader (wfn_loader.py:1423) is currently dead in production.
- htransform ADOPTING PsiGStore host tiles/io_callback: **NO** — the Galerkin sweep reads each
  band ONCE over the full r axis (single r-chunk); PsiGStore's payoff is re-read amortization
  across MANY r-chunks + scan-internal FFT-box aliasing, which to_rchunk already gives htransform.
- Retire iter_psi_rchunk_bandwise: **NO** — it stays the shared streaming iterator (htransform +
  bse/vq_interp), already single-sourced on to_rchunk_inner with the gw kernels.
- qe_save_reader (QE XML+rho), zeta_loader (SlabIO, WfnLoader-shaped), bse_io restart readers:
  **no unification** — different data/formats; bse_io's per-rank h5py hyperslab readers are a
  possible future SlabIO consolidation (out of F scope).
What WAS unified (bit-exact, gate-verified):
- NEW common/wfn_transforms.load_psi_gflat_padded — single source for the
  load -> cap-at-file-nbands -> zero-pad-band-axis -> reshard dance that was triplicated
  (psi_G_store._populate_from_loader, load_centroids_band_chunked, iter_psi_rchunk_bandwise;
  the iter copy was MISSING the past-mnband cap = latent CrI3-class crash, now fixed for free).
- htransform now loads psi(G-flat) ONCE per run (streaming_galerkin_solve ~:137) and the window
  serves BOTH centroid sampling (load_centroids_band_chunked(psi_G_flat=...)) and the r-stream
  (iter_psi_rchunk_bandwise(psi_G_flat=...) slices per-chunk on device via one cached
  dynamic-slice jit). File reads for the Galerkin: n_bc+1 loader.load calls -> 1; symmetry
  unfold work halved. G-flat window is the same tensor load_centroids already held (~6-11% of
  FFT box) so residency through the Q loop is proven-affordable at CrI3 scale.
- psp/get_DFT_mtxels: _gvecs_full_cache/_ngk_full_cache now reuse the passed WfnLoader
  (loader-internal caches) instead of opening a duplicate per-path WfnLoader (second file
  handle + duplicate (ngktot,3) gvecs table per rank).
Measured (cohsex fixture, 2 nodes x 2 ranks, forced 2 band chunks via LORRAX_GALERKIN_CHUNK_GIB=0.004,
same pinned idle nodes, warm cache, JAX_LOG_COMPILES wrapper):
- BEFORE (main @419f57e): wall 13.6s, Galerkin 3.83s, 580 compiles total (145/rank)
- AFTER  (wt-F):          wall 12.2s, Galerkin 3.67s, 572 compiles total (143/rank)
- Gate: bandstructure.dat max|dE| = 0.000e+00 vs bs_groundtruth_meshless.dat AND vs baseline
  output (n=648) — bit-exact. Win scales with n_bc (per-chunk read+unfold eliminated).
For workstream G (wfn_loader.py, not touched): (1) AsyncWfnReader is production-dead — fold that
into the 3-path consolidation decision; (2) loader band round-up can return REAL file bands in
pad slots — every consumer masks them; zeroing them inside load would simplify consumer contracts.
Infra note: two runs died DEADLINE_EXCEEDED because alloc_run.sh pins the coordinator to the
holder head node while srun --overlap may place the step elsewhere under step contention —
pinning -w <nodes> + coordinator=first pinned node works reliably.

## Workstream I — distributed-linalg facade (wt-I, branch linalg-facade, base 708c9d1) — DONE
NEW package src/ffi/linalg/ (resolve.py + dispatch.py + __init__.py): ONE
interface from JAX-side calls to backend choice (native|cusolvermp|slate|
scalapack per op: eigh/cholesky/solve_lu).  resolve_backend() applies ALL
guards at RESOLVE time in fixed order: vocabulary -> platform -> compiled-
capability (ffi_loader.has_target — NEW: slate-less build + slate request now
fails at resolve with the available-backends list, was call-time) -> process-
coverage (NEW) -> geometry (square-mesh syevd-deadlock, SLATE 1xq stride,
scalapack square-or-1D) -> divisibility.  list_backends() introspection;
backend_module() = the one import seam (isdf/core's 5 post-resolution FFI
imports + 4 bench scripts routed through it).  ffi/common/dispatch.py -> shim.
isdf/core keeps the channel policy ladder BYTE-IDENTICAL on top (replication
cap, rank-truncate refusal, route strings); its _require_{slate,scalapack}_ffi
+ inline geometry checks deleted in favor of facade calls; _mesh_is_cpu now
aliases ffi.linalg.mesh_is_cpu.  Config: NEW input key eigh_backend
(gw_config _DEFAULTS/_NORMALIZE_STR/validation/BackendConfig) — input file is
source of truth, --eigh-backend (htransform, exciton_bands) now an OVERRIDE
(default None -> key).  bse_setup/vq_interp resolve ONCE up front.  Docs:
docs/dev/linalg_ffi.md (architecture, backend comparison table, config
surface, examples, add-a-backend, sharp edges) + HANDOFF cross-link.
Deliberately NOT unified: cublasmp (gw/w_isdf fused gemm/W-solve — not a
selectable op), common/*_test.py wrapper-contract tests (test backend
internals directly), the explicit-cusolvermp-on-CPU legacy honor in the ζ-fit
ladder (pinned semantics).
Gates (holder 7874023): wk_I/verify.py 50+ checks PASS (auto->native on CPU
all ops; cusolvermp-on-CPU + slate-not-compiled + coverage/geometry/
divisibility rejections at resolve; list_backends sane; route strings pinned);
pytest zeta_mesh_invariance -k cap + wfn_loader_eager 15p/1s + charge_zeta_
route 7p green; htransform 2x2 (2 nodes x 2 ranks) bit-exact vs
bs_groundtruth_meshless.dat (max|dE|=0.0), ALSO with get_centroids_fi=true
exercising the bse_setup resolve path; gw cohsex 1 node 2x1 eqp vs eqp_ref
max|d|=1.0e-06 eV (tol 1e-3).  py_compile clean (14 files).  Not committed.

## MoS2-12x12 prep (2026-07-25) — production-scale 40-node CPU GW, STAGED NOT SUBMITTED
Staged: `/scratch2/08271/jackmc/lorrax_mos2_12x12/{run1,run2_c1194}/` (own dir each —
ADVICE 6d `tmp/zeta_q.h5` is not centroid-namespaced). Inputs `gw.in` + `run40.sbatch` /
`run40_c1194.sbatch` (normal queue, -N40 --ntasks-per-node=2, taskset 0-27/28-55,
`XLA_FLAGS=--intra_op_parallelism_threads=28`, ib0 Gloo, `ISDF_JAX_CACHE_DIR=""`,
`HDF5_USE_FILE_LOCKING=FALSE`, explicit coordinator, date stamps + node-0 memory sampler).
Inputs symlink the EXISTING `/scratch2/08271/jackmc/mos2_80ry_12x12/` WFN.h5 (15.65 GB;
nrk=144 full BZ, ntran=1, mnband=400, ngkmax=8603, FFT 36x36x135 -> n_rtot=174960,
nspinor=2, nelec=26) + dipole.h5 + kin_ion.h5 + centroids_frac_{276,1194}.txt.

**Calibration win — the planner's HWM under-count is now a NUMBER, not "40-50 GB".**
Job 7873510 (40 nodes x 1 rank, 276c, r_chunk=174960): model HWM 48.30 GB/dev vs
`sacct MaxRSS` 97,754,104 K = **97.75 GB** -> **real/model = 2.02x**. That job was
CANCELLED (user, 19:00 of 25:00), NOT OOM-killed — correcting the "40 nodes OOMs" read.
Consistent with run_276c (P=4, model 46.75 -> completed on 192 GB) and ADVICE 6a's
budget-90 P=4 OOM (model 76.5 -> ~155 GB + rank-truncate replication).
=> **sizing rule for CPU: real peak/rank ~ 2.0x the printed HWM estimate.**

**Config (verified by running the REAL `plan_gflat_chunks` on a synthetic 8x10 mesh via
`--xla_force_host_platform_device_count=80`; script `lorrax_mos2_12x12/preflight.py`):**
| | run1 (276c) | run2 (1194c) |
|---|---|---|
| mu_pad = round_up(n_rmu,80) | 320 (13.8% pad) | 1200 (0.5% pad) |
| band_chunk / r_chunk | 160 / **174960 (1 chunk)** | 160 / **58320 (3 chunks)** |
| model HWM (binder C_fit_one_rchunk) | 27.65 GB/dev | 35.18 GB/dev |
| real @2.02x | ~56 GB/rank (112/node) | ~71 GB/rank (142/node) |

- **nband 120 -> 160 is FREE at world_size=80**: `b4 = round_up(nband,80) = 160` either way,
  so the historical 120 pads 40 all-zero band slots at identical cost. Band pad waste 0%.
- **`band_chunk_size` default 16 is a trap**: gw_init passes it as an *override* whenever >0,
  and `_bump_bc(16)` = 80 at P=80 -> two band chunks. Pinned to 160 = one.
- **`vq_g_chunk_size`**: auto = `_pick_g_chunk(8603)` = 1229 (8603 = 7*1229) -> 7 tiles.
  Pinned to 8603 = one tile; Stage E is <1 GB either way.
- **r_chunk quantization**: `n_rtot/p_xy = 174960/80 = 3^7`, so the only r_chunk values that
  both divide n_rtot evenly AND are multiples of p_xy give **1, 3, 9, 27 or 81** chunks.
  2 chunks is unreachable — and would be 76.8 GB model / ~155 GB real anyway.
- Replication cap `nq*n_rmu^2*16`: 276c = 0.16 GiB, **1194c = 3.06 GiB < 4 GiB** ->
  `replicated_rank_truncate` stays available with NO `LORRAX_ZETA_REPLICATE_CAP_GIB` override.
- `distributed_cholesky`/`distributed_lu` = **auto** in both (ADVICE 6a). Note ADVICE 8 still
  says `= off` for CPU — that text is SUPERSEDED by 6a and should be fixed.
- q axis (nq=144) is not mesh-sharded (L_q/gflat_acc/rchunk shard mu on ('x','y')),
  so 144-vs-80 divisibility is a non-issue. ADVICE 5b eager-load wall is gone at 80 ranks:
  process-local band-sharded load -> psi(G-flat) global 6.34 GB = 79 MB/rank.

**Open (to be filled by the timed run):** 40-node x 2-rank wall for 276c (predicted ~35-45 min,
2 h requested) and 1194c (predicted ~80-115 min, 3 h requested); first real P=80 scaling point
against P=4 (zeta-fit 3503 s, job 7871465) and P=1 (~140 min).

## MoS2 scale ladder (2026-07-25) — hunting the lim P->inf walls, 40 nodes x 2 ranks (P=80)
Staged under `/scratch2/08271/jackmc/lorrax_mos2_12x12/` (run3..run6 + RUNBOOK.md,
preflight_ladder.py, stage_rungs.py, centroids_gen/). **Nothing submitted.**

**TWO LAUNCH BUGS FOUND THE HARD WAY (fix before any 40-node run):**
1. **`XLA_FLAGS=--intra_op_parallelism_threads=N` DOES NOT EXIST in this jaxlib** and
   XLA **hard-aborts every rank**: `F0725 parse_flags_from_env.cc:234] Unknown flag in
   XLA_FLAGS`. MEASURED: job **7874158** died in 2 min on all 80 ranks. **ADVICE 10
   recommends this flag — the advice is WRONG for this build and should be corrected.**
   Only `xla_cpu_multi_thread_eigen` / `xla_cpu_parallel_codegen_split_count` exist;
   there is no thread-COUNT flag. XLA:CPU sizes its Eigen pool from process CPU
   AFFINITY, so **`taskset` IS the thread cap** (which is how ADVICE 10's own 2x28
   numbers were measured). All staged sbatch corrected.
2. The container's default JAX **persistent AOT cache is machine-mismatched**:
   `cpu_aot_loader.cc:220 ... +prefer-no-scatter not supported on the host machine ...
   could lead to SIGILL`. Another reason `ISDF_JAX_CACHE_DIR=""` must stay set.

**THE #1 lim P->inf BLOCKER: charge-channel `replicated_rank_truncate`.**
`isdf/core._factor_c_q_replicated` does `with_sharding_constraint(C_log, P())` —
FULLY REPLICATED — then `jnp.linalg.eigh` per q. Three distinct problems:
- *The gate is mis-specified.* `_replicate_charge_ok` tests the WHOLE stack
  `nq*n_mu^2*16 <= 4 GiB` -> refuses at **n_mu > 1365** (nq=144). But
  `factor_c_q_replicated_batched` already batches q at
  `_REPLICATED_FACTOR_MAX_BATCH_BYTES = 4 GiB`, so the real per-rank transient is
  **flat at ~12-13 GB independent of mu**. The gate should test one BATCH, not the
  stack. Cheapest fix on the board.
- *Real memory wall*: one `(mu,mu)` matrix = `mu^2*16` replicated **per rank** —
  O(mu^2), NOT O(mu^2/P). 1.6 GB/rank at mu=10k. (Contrast the SHARDED
  `L_q = nq*mu^2*16/P` = 2.88 GB/rank at the same mu — the rest of the code is fine.)
- *The killer is REDUNDANT COMPUTE*: all 80 ranks run the identical `nq x eigh(mu,mu)`,
  zero parallel speedup, O(nq*mu^3). Calibrated on `zeta_fit.cholesky = 6.8 s @
  mu_pad=280` (job 7871465): **535 s @ mu=1200, ~41 min @ 2000, ~5.5 h @ 4000,
  ~86 h @ 10000.** Compute bites long before memory.
- **Consequence: beyond n_mu ~ 3000-4000 there is NO correct configuration today.**
  Raising the cap keeps correctness but is unaffordable; the only above-cap
  alternative is `sharded_cholesky` = the ADVICE 6a physics destroyer.
- **Fix:** a `distributed_rank_truncate` route (distributed Hermitian eigh on the
  block-cyclic (mu,mu) tile -> truncated pinv as distributed matmuls). Everything
  needed exists after workstream I: `src/ffi/linalg/` has `eigh` as a selectable op
  with SLATE/ScaLAPACK/cuSolverMp backends. Guards: cusolverMpSyevd DEADLOCKS on
  rectangular meshes (8x10 is rectangular -> SLATE/ScaLAPACK on CPU), and the
  truncation must run at the LOGICAL mu extent (`solve_at_logical`).

**WALL #2: the zeta writer.** `slab_io = h5py_allgather` gathers `nq*n_mu*ngkmax*16`
onto ONE rank: 23.7 GB @1194c, 39.6 GB @2000c, **crosses the ~92 GB/task envelope at
n_mu ~ 4650**. O(mu), single-process. And it is **the only multi-host writer that
exists** — `src/ffi/phdf5/cpp/write_ffi.cc` was NOT ported to the CUDA-free host lib
(only the READ core was). Fix: serialized per-rank hyperslab writes (pure h5py,
O(mu/P), `SlabIO.write_slab` already takes `valid_shape=`), or port write_ffi.cc.

**Ladder (all preflighted with the REAL planner on a real 8x10 mesh of 80 forced
host devices; `real` = MEASURED 2.02x ratio):**
| rung | n_mu(pad) | nband(b4) | r_chunk | model HWM | real/rank | cap gate | writer |
|---|---|---|---|---|---|---|---|
| run3_c1194_b400 | 1194(1200) | 400(400) | 58320 x3 | 35.77 GB | 72.3 GB | 3.06 GiB ok | 23.7 GB |
| run4_c2000_b160 | 1998(2000) | 160(160) | 19440 x9 | 20.60 GB | 41.6 GB | **8.58 REFUSE** | 39.6 GB |
| run5_c2000_b400 | 1998(2000) | 400(400) | 19440 x9 | 21.60 GB | 43.6 GB | **8.58 REFUSE** | 39.6 GB |
| run6_c2400_b160 | ~2400 | 160(160) | 19440 x9 | 24.75 GB | 50.0 GB | **12.36 REFUSE** | 47.6 GB |
`r_chunk` is quantized: `n_rtot/p_xy = 3^7`, so only **1/3/9/27/81** chunks exist.
Chunk count is forced up with mu (1 -> 3 -> 9); that part scales correctly.

**kmeans:** `centroids_frac_1998.txt` GENERATED (target 2000, orbit-quantized;
mu_pad 2000, 0.1% pad) in `lorrax_mos2_12x12/centroids_gen/`; ~2400 set still
running. **There is no `--out` flag** — argparse prefix-matched the old
`--out X` onto `--out-suffix`, which is the whole filename-mangling bug; pass
NEITHER and the file is auto-named `centroids_frac_<n_unique>.txt`. Generation has
its own wall: the pivoted-Cholesky prune holds two `(nk,ns^2,M,M)` c128 pair
densities = `2*144*4*M^2*16` (M ~ oversample*N_c): 53.9 GB @M=1710, 152.8 GB @M=2880,
384 GB @M=4566 (the ADVICE 5 OOM). Sharded path needs `M % n_dev == 0` and M is
orbit-determined/unpredictable -> use `--no-shard` + `--oversample` to cap M.

**OBSERVABILITY GAP (blocks the owner's central question):** the rank truncation
prints **nothing**. `keep = lam > rcond*lam_max` is inside a jit
(`isdf/core._rank_trunc_factor`) with no count emitted, so there is **no way to see
how many mu modes get dropped or whether that grows with mu**. One
`jax.debug.print` of `keep.sum(-1)` would make the ladder's central question
directly observable. Until then the proxy is a `zeta_rcond` (env `LORRAX_ZETA_RCOND`)
sweep 1e-6/1e-8/1e-10 with eqp diff. Best pre-run conditioning signal is in the
CENTROID log instead: `After pruning: N centroids (rank=R)` (R saturating = basis
rank-deficient) plus `[pivoted_cholesky] picked-pivot residuals first/mid/last`
(1194c reference: 6.319e-06 / 1.066e-08 / 1.028e-11) and `tr(R_k)/tr(G)`.

**10x bands:** 400 is this WFN's hard ceiling. `solvers/pseudobands{,_v2}.py` are
library-only (no CLI), **not wired into GW at all** (no `_DEFAULTS` key, nothing in
`src/gw/` imports them), no tests, the `band_norms` hook `gw_init` reads is never
populated by `WfnLoader`, and `psp/run_nscf.py`'s WFN ingest is **broken for nk>1**
(writes every band into ik=0). Recommendation: QE nscf regeneration (nbnd~1600 ->
~63 GB WFN, `-ndiag 1` per ADVICE 7, regenerate dipole.h5+kin_ion.h5) — but do
NEITHER until wall #1 is fixed: nband=1600 wants n_mu >> 3000, where no correct
route exists.

### ADDENDUM — WALL #0 MEASURED (job 7874236), supersedes the r_chunk sizing above
Someone submitted `run1/run40.sbatch` (corrected, no bad XLA_FLAGS) at the full
40 nodes x 2 ranks. It got **everything right** — `Backend: CPU Devices: 80
Mesh: 8x10 Processes: 80`, `band_chunk=160`, `r_chunk=174960 (1 chunks)`,
`persistent=0.25 GB/dev`, and the correct
`Computing L_q = rank-truncated pinv [path=replicated_rank_truncate]` — then died
at 418 s with
`RESOURCE_EXHAUSTED: Out of memory allocating 270976942488 bytes` (**270.98 GB in
ONE buffer**) at `isdf/core.py:2203 fit_one_rchunk`, `sacct MaxRSS` only 8.4 GB
(refused before allocating).
- The planner modelled that same transient at **27.41 GB/rank** ->
  **real single-buffer ask = 9.9x the model.**
- The FULLY UNSHARDED `(nk,ns^2,mu,cr)` pair density is 515.98 GB; the request is
  only **1.90x smaller**, i.e. the Stage-C arena is sharded by ~2, **NOT by P=80**.
  This is exactly the "N_mu intermediate materialized on far fewer than P procs"
  defect class. **It bites at 276 centroids** — before the replication cap and
  before the writer.
- **"Few and large chunks" is therefore the WRONG strategy.** All six rungs
  re-pinned so `9.9 x model <= ~40 GB`: run1 19440 (9), run2/3 6480 (27),
  run4/5 2160 (81), run6 2160 (81). 81 is the max the quantization allows
  (`n_rtot/p_xy = 3^7`), so **wall #0 caps this machine at mu ~ 4000 on its own**,
  independent of walls #1/#2.
- Next step is NOT more chunking: check the `in_specs`/`out_specs` of the Stage-C
  `shard_map` at `isdf/core.py:2203` against `gflat_memory_model._stage_C_slope`
  (which assumes `slots*(nk,ns^2,mu,cr)/p_xy + (nq,mu,cr)/p_xy`). A residual ~2x
  suggests one operand replicated on one mesh axis.

---

## J — lim P→∞ μ-replication audit (worktree wt-J, branch `mu-replication-audit` @ 0f9e4dc)

### J.0 — ROOT CAUSE of the 270.98 GB Stage-C allocation (job 7874236)
It is **not** a mis-specified `shard_map` spec. Every `in_specs`/`out_specs` in
`z_q_from_psi_sm` is correct. The offender is `isdf/core.py` step (3), the
**band `all_gather` of the FULL-r ψ(r) slab**:

```
psi_Y_bc_full_r = jax.lax.all_gather(psi_Y_bc_local_full_r,
                                     axis_name=('x','y'), axis=1, tiled=True)
# -> (nk, bpd_max_global, ns, n_zchunk)   REPLICATED in bands, FULL in r
```
Each rank computes its 1/P band block over the **whole** r-chunk, then gathers
bands, then slices its own `r_loc` (the r-slice *must* follow the gather —
band/r coherence). So between (3) and (3b) there exists one object that is
**replicated on the band axis AND unsharded on the r axis — no /P at all**:

    nk · bpd_max_global · ns · cr · 16

Exact arithmetic at run1 (nk=144, band window 160, ns=2, cr=n_rtot=174960,
μ_pad=320, mesh 8×10):

| term | bytes |
|---|---|
| gathered ψ(r) slab (step 3) | 128.99 GB |
| `jnp.take` band-compaction copy (step 3a) | 128.99 GB |
| 2 pair-density carries `(nk,ns,r_loc,μ_loc,ns)` | 12.90 GB |
| **total** | **270.888 GB** |
| **actual OOM** | **270.977 GB** (Δ = 88 MB of masks/slicer/δP) |

Match to **0.03 %**. The "sharded by ~2 not by P" reading was a coincidence of
2 copies × 129 GB; the true statement is *sharded by 1*.

### J.1 — fixes implemented + gated (all bit-exact, 2 mesh shapes each)

| fix | file | effect |
|---|---|---|
| **Elide the identity band-compaction `jnp.take`** — the permutation is the identity whenever `bpd_per_bc == bpd_max` (every full band chunk), but XLA cannot fold a traced-index take, so it allocated a second full 129 GB slab. Static host-side check `_y_compact_identity`. | `isdf/core.py` (:521, :634, :716) | Stage C **271 → 142 GB** at cr=n_rtot; **31.7 → 17.4 GB** at run1's mitigated cr=19440 |
| **Stage-C model term for the gathered ψ(r) slab** (`+2·nk·band_chunk·ns·16` per cr, **no /P**), and `band_chunk` threaded into `_stage_C_slope`. | `gw/gflat_memory_model.py` | model at cr=n_rtot **27.41 → 285.4 GB** (real 270.98 → 5 % conservative). Planner now picks r_chunk ≈ 43–47 k (**4–5 chunks**) instead of 1, so the OOM cannot recur |
| **De-replicate `fH_R_rep`** — `jax.device_put(fH_R, rep)` made `nk·rank²·16` on every rank (≈51 GB at rank 4716) plus an 11.4 GB replicated `(bs,rank,rank)` einsum temp. Replaced with the pattern already proven in `bse_setup.py::_fourier`: keep `fH_R` at `P(None,'x','y')`, pin the einsum output, reshard→q, then hermitize. | `bandstructure/htransform.py` (:887-945, :972) | removes the **#1 htransform offender**; its break-μ was ≈3.1 k centroids |
| **`_replicate_rank_truncate_ok`** — the charge cap tested the whole `nq·μ²` stack, but `factor_c_q_replicated_batched` already q-batches at its own 4 GiB bound, so the replicated transient is **one batch, flat in nq**. New criterion for the `rank_truncate` branch only (`cholesky` untouched → no existing route changes). | `isdf/core.py` (:936, :1090) | `rank_truncate` (the §6a physics cure) reachable at full-BZ 12×12 μ=2412 without `LORRAX_ZETA_REPLICATE_CAP_GIB` surgery |
| **Rank-truncation observability** — `jax.debug.print` of `n_keep/q`, `λ_max/q`, `λ_min(kept)/q` inside the eigh factor (`LORRAX_ZETA_RANK_LOG=0` silences). | `isdf/core.py` (:1288) | the ladder's central conditioning signal is now visible |

**Gates (all PASS):**
- GW cohsex eqp vs `eqp_ref.dat`, tol 1e-3: **P=4 and P=8 → max|Δ| = 1.0e-6 eV** (file print precision), 0/1888 over tol.
- htransform vs `bs_groundtruth_meshless.dat`, gate 1e-8: **P=4 and P=8 → max|ΔE| = 0.000e+00, bit-identical**.

### J.2 — ranked REPLICATED-SCALING offenders (P=80, mesh 8×10, break-μ = μ at which one rank exceeds 90 GB)

| # | object | file:line | per-rank bytes | break-μ |
|---|---|---|---|---|
| 0 | gathered FULL-r ψ(r) slab | `isdf/core.py:685/696` | `nk·bc·ns·cr·16` (×2) | μ-independent, **cr-capped**: forced ≤81 chunks ⇒ hard wall ~μ 4 k. FIXED ×2; structural all-to-all fix designed |
| 1 | `fH_R_rep` | `htransform.py:894` | `nk·rank²·16` | **≈3.1 k** — **FIXED** |
| 2 | SlabIO `H5PY_ALLGATHER` `_to_host` (ζ writer, and V/S/W0 restart) | `_slab_io_allgather.py:68` | `n_q_disk·μ·ngkmax·16` / `nq·μ²·16`, ×2 (device+host) | **2.3 k** full-BZ ζ / **4.4 k** W0. **LIVE on Frontera** — venv has no mpi4py and h5py has no MPI ⇒ `PHDF5_HOST` unreachable, always falls back |
| 3 | `solve_w` returns W **replicated** | `w_isdf.py:283` (`rep_3d = P(None,None,None)`) | `nq·μ²·16` ×2 (static+probe) | **4.4 k**; and it re-runs every SC iteration |
| 4 | `_kpath_batch` `mat` temp (was replicated) | `htransform.py:906` | `bs·rank²·16` | ≈8 k — **FIXED** |
| 5 | replicated eigh in `replicated_rank_truncate` | `isdf/core.py:1212` | bounded (~12 GB) — **TIME** not memory: O(nq·μ³) redundant on every rank, **zero P-scaling** (~5.5 h @4 k, ~86 h @10 k) | compute wall ~4 k |
| 6 | `S`, `S_chol`, `fH_k0_rep`, `fH_gamma_rt` | `htransform.py:380/883/867` | `rank²·16` each | ~37 k (and `S` is literally `I`) |
| 7 | `B_at_mu`/`B_rep` | `htransform.py:379`, `bse_setup.py:180` | `rank·ns·μ·16 ≈ 64μ²` | ~37 k |
| 8 | `Fch` host mirror (NOT under `keep_host_mirrors`) | `vq_interp.py:670/711` | `nq·μ·nG·16` | ~10 k |
| 9 | `gflat_to_rmu` band-slice replication | `wfn_transforms.py:1006` | `nk·nb_pad·ns·μ·16` | ~65 k (htransform path only) |
| 10 | `_load_ring_subset` whole-array `V/W0/psi_full` reads | `bse_io.py:1313/1315/1330` | `nq·μ²·16` | single-device-only by contract, **unguarded** |
| 11 | `sigma_kij_host` + its `process_allgather` copy | `ppm_sigma.py:739/831` | `2·n_ω·nk·nb²·16` host | no μ, but unmodelled |

### J.3 — the 2.02× model-vs-MaxRSS mystery: RESOLVED (two parts)
1. **Stage C's missing gathered-ψ term** (J.0) — the model undercounted the
   binding r-chunk arena by up to 10.4×. Whether it shows as 2× or 10× depends
   only on whether Stage C was the binder in that calibration run. **Fixed.**
2. **The model stops at Stage E (`E_v_q`).** `gflat_memory_model.plan_gflat_chunks`
   has no term for χ₀, W, the PPM fit or Σ — and those peak *after* ζ-fit on top
   of a resident `V_q`. In particular `fit_gn_ppm_from_wc_pair`
   (`minimax_screening.py:408-439`) is **not jitted**: ~15 concurrent `(nq,μ,μ)`
   eager arrays with zero XLA buffer reuse, fed by the **replicated** W of J.2#3.
   Discriminating measurement: `LORRAX_EXIT_AFTER_ZETA=1`; if MaxRSS/model
   collapses to ~1.05–1.3× (the documented CPU figure) the remainder is all
   post-ζ. Ruled out: gvec/symmetry tables (≤0.2 GB), io_callback host buffers
   (ζ-fit only, already modelled).

### J.4 — conditioning toward 10 k centroids
Knobs that govern it, all in one place now: `zeta_rcond` (default **1e-8**,
`LORRAX_ZETA_RCOND`) — the only *principled* cure, drops λ < rcond·λ_max;
`zeta_ridge` (default **0**, `LORRAX_ZETA_RIDGE`) — opt-in Tikhonov, superseded
by rank_truncate; the hard `1e-14·|tr C|` Cholesky floor; `LU_RIDGE = 1e-12`
(transverse); `htransform` SVD `rtol = 1e-8` + the `1e-10·mean(diag S)` ridge at
`htransform.py:881`; `EPS_LR = 1e-8` and `np.linalg.pinv` (default rcond!) in
`vq_interp.py:812/1159`. **What was logged: nothing about the truncation itself**
— now fixed (J.1). Watch as μ grows: `n_keep/q` vs `n_log` (the moment
`n_keep < n_log` is the moment μ has over-completed the pair-density rank), and
`λ_max/λ_min(kept)`. Two things to add next: (a) make `zeta_rcond` an input-file
key (it is only a kwarg + env today), (b) `vq_interp`'s two bare
`np.linalg.pinv` calls inherit numpy's default rcond and are unlogged.

### J.5 — the post-ζ SIGABRT (job 7874242): the V_q ζ read is unsharded on the allgather backend

**Symptom.** ζ-fit completed clean (1692 s, MaxRSS 39.0 GB/rank — confirming
`_GATHERED_PSI_SLOTS = 2` at cr=19440), `zeta_q.h5` written (5.49 GB), then all
80 ranks died at `16:24:14.6` with

```
F0725 16:24:14 raw_buffer.h:149] Check failed: buffer_.IsConcrete()
```

→ `LOG(FATAL)` → `abort()` → rc=134. Not a Python exception, not
`RESOURCE_EXHAUSTED`, not the OOM-killer (that would be SIGKILL).

**Stage.** Proven by the restart job 7874331: it died with
`FileNotFoundError: tmp/isdf_tensors_276.h5` at `gw_init.py:746`, i.e. the
crashed stage is the one that BUILDS/writes that file — V_q assembly, the first
thing after ζ-fit. `mem_node0` shows 22 → 53 GB in the final 30 s sample with
`buff_cache` 7 → 12 GB (the 5.49 GB ζ file entering page cache on 2 ranks/node).

**Cause.** `gw/v_q_g_flat.py::_make_read_all_ibz` reads ALL q in ONE call
(deliberately — it fixed a trace-cache-miss storm). On the `H5PY_ALLGATHER`
backend `_slab_io_allgather.read_slab` reads **the whole global tensor into host
numpy on every rank, twice** (`read_host` + the zero-padded `host`) and only
then `device_put`s it to the sharded layout — by explicit design (its docstring:
*"the FFI backend reads sharded directly, this backend reads the full slab on
every rank then device_puts"*). At this run's dims:

    n_q · μ_pad · ngkmax · 16 × 2 = 144 · 320 · 8603 · 16 × 2 = 12.69 GB/rank
                                                             = 25.4 GB/node

against a **79 MB/rank** sharded read — **160× the modelled `zeta_slab` term.**
That is the fast-growing allocation, and the node was still climbing when it died.

**Why a CHECK and not an OOM** — two mechanisms, both plausible, both fixed:
(a) an allocation failed inside the PJRT host-transfer layer, the buffer's async
value went to error, and the next raw-buffer access CHECK-failed instead of
propagating; (b) XLA:CPU can adopt a large host numpy **zero-copy** in
`device_put`, and `read_host`/`host` fall out of scope when `read_slab` returns
— if the definition event is still pending the raw buffer is left non-concrete.
(b) is favoured by the abort being instantaneous and simultaneous on all 80
ranks at only 53 GB of 192, and by its size-dependence (small arrays get
copied, which is why no fixture ever hit it). Not decidable from the log alone.

**Fix (implemented, gated):** `_AllgatherBackend.read_slab` now reads only the
**process-local shard** when a `partition_spec` is given — each rank already has
its own h5py handle, so it reads its own hyperslab and assembles via
`jax.make_array_from_single_device_arrays`. Byte-identical (each shard takes
exactly the elements the sharding assigns it; positions past `valid_shape` stay
zero, the μ-pad contract). Plus `jax.block_until_ready` before returning on BOTH
paths, so the source numpy outlives the transfer — that closes mechanism (b)
independently.

| read | pre-fix / rank | post-fix / rank |
|---|---|---|
| V_q batched ζ (μ=276) | 12.69 GB | 0.079 GB |
| bispinor V-tile, ×7 tiles (μ=276) | 0.47 GB | 0.003 GB |
| bispinor V-tile, ×7 tiles (μ=2412) | 27.25 GB | 0.170 GB |

**Stage-F model term** (the coordinator's ask — it binds before the corrected
Stage C at this scale): `plan_gflat_chunks` gained `slab_io_replicates` (wired
from `cfg.backend.slab_io` in `gw_init.py`) and a `F_tensor_write` peak =
`2·(n_q, μ, μ)` **unsharded** when the backend is `H5PY_ALLGATHER` — the
`_to_host` gather on the restart-tensor WRITE, which is still replicated and is
the next wall: **μ=276 → 0.47 GB, μ=2412 → 27 GB, μ=10k → 461 GB.**

**Operational notes.**
- Job 7874331 is a dead end and cannot reproduce the crash — `restart=true`
  needs `isdf_tensors_276.h5`, which was never written.
- There is no "reuse existing ζ" path (`fit_zeta` only *validates* the basis at
  `gw_init.py:90` and always refits), so a rerun pays the 1692 s ζ-fit again
  even though `zeta_q.h5` is clean and complete. Cheap feature worth adding.
- The real cure for the whole class remains offender #2: install `mpi4py` +
  h5py built `HDF5_MPI=ON` so `PHDF5_HOST` becomes reachable; then both the
  read and the write are per-rank hyperslabs and `slab_io_replicates=False`.

**Ruled out** (the coordinator's candidate list): χ₀ `Gv_k`/`Gc_k` are
`(nk,ns,μ,ns,μ)` at `P(None,None,'x',None,'y')` = **11.8 MB/rank**; `chi_R`
2.9 MB/rank; `V_acc` 2.9 MB/rank; the replicated W (`w_isdf.py:283`) is 236
MB/rank at μ=276 (real, but the μ≥2.4k wall, not this); the un-jitted PPM fit is
never reached. **There is no (G,G') object anywhere in the tree** — grep for
`ngkmax, ngkmax` / `ngkmax**2` returns nothing; v(G) is diagonal and stored
`(nq, ngkmax)`. The ngkmax·μ scaling the coordinator suspected is real, but it
is in the **I/O seam**, not the physics kernels.

**J.5 gates (job 7874332, all PASS):**
- GW cohsex eqp vs `eqp_ref.dat`, tol 1e-3: **P=4 and P=8 → max|Δ| = 1.0e-6 eV**,
  0/1888 over tol; physics route confirmed `replicated_rank_truncate` at both.
- htransform vs `bs_groundtruth_meshless.dat`, gate 1e-8: **P=4 and P=8 →
  max|ΔE| = 0.000e+00, bit-identical**; Γ round-trip 2.776e-17.
- Unit (job 7874335): `read_slab` sharded fast path vs the whole-slab result,
  8 devices, mesh 2x4, μ-pad 11→16 so the pad zero-fill is exercised —
  **ALL PASS, EXACT** for every spec the codebase uses:
  `P(None,('x','y'),None)` (V_q ζ), `P(None,'x','y')` (bispinor V tiles),
  `P(None,None,('x','y'))`, `P('x',None,'y')`, `P()` (replicated), plus an
  offset sub-slab read.  Incidental finding: JAX raises `IndivisibleError` for
  uneven NamedShardings, so the shard loop's empty/uneven handling is
  defensive only.

### J.6 — MEASURED, production (MoS2 12×12, 276c, P=80, mesh 8×10, r_chunk=19440)

**ζ-fit wall time, job 7874242 (pre-fix) vs 7874338 (post-fix, merged `658b0de`):**

| | 7874242 | 7874338 | Δ |
|---|---|---|---|
| ζ-fit elapsed | 1692 s | **1324 s** | **−21.7 %** |
| chunk-loop total | 1905.5 s | **1489.3 s** | **−21.8 %** |

Same r_chunk, same 9 chunks, same node count — the delta is the Stage-C
identity-take elision (J.1): at band_chunk=160 on P=80, `bpd_per_bc == bpd_max`
so the compaction permutation IS the identity and the 129 GB/rank second copy of
the band-gathered ψ(r) slab is now skipped entirely. It was costing ~21 % of
ζ-fit in pure memory traffic.

**Planner, same run — both model fixes live and validated:**

```
HWM estimate  = 31.96 GB/dev (38% of budget) [binder: C_fit_one_rchunk]
  C_fit_one_rchunk....   31.96 <=      (was 6.70, binder A_centroid_load)
  A_centroid_load.....    6.70
  F_tensor_write......    0.55         (new Stage-F term)
```

Model 31.96 vs 7874242's measured MaxRSS 39.0 GB/rank at the same r_chunk =
**1.22×**, down from the 2.02× that opened this workstream and inside the
documented 1.05–1.3× CPU band. NOTE the model still assumes
`_GATHERED_PSI_SLOTS = 2` while this run has the elision (1 live slot), so it is
now ~14 GB/dev conservative; 7874338's MaxRSS is the measurement that decides
whether the constant can drop to 1 (expect ~20–24 GB/rank if so).

**V_q entry (the 7874242 SIGABRT point) — CLEARED, CONFIRMED.** 7874338 ran the
full batched ζ read and all 144 q of the CC tile (`q=143/144: kernel=0.13s`),
printed `V_q computed: Shape (144, 320, 320), V_q=0 trace 6911688774.4958`,
`Wavefunctions built`, `Chunked ISDF path complete`, wrote
`tmp/isdf_tensors_276.h5` (the file whose absence proved the crash stage), and
proceeded into the minimax screening window. **0 occurrences** of `IsConcrete` /
`Check failed` / `RESOURCE_EXHAUSTED` / `Aborted`. Node memory peaked at 36 GB
and fell back to 18 GB (2 ranks/node), against 22→53-and-climbing-to-abort
before. Total V_q stage cost: ~2 min.

## Workstream H — test-suite health + perf at current scale (wt-H, audit-tests-perf @ 419f57e = pre-F/G merged HEAD)
- **tests/test_zeta_mesh_invariance.py: 4/4 PASSED in 973s (16m13s)** — FIRST full completion on this
  branch (run solo, 1 node/1 task on holder 7874023). Includes the formerly-failing
  `test_rank_truncate_refuses_above_the_replication_cap` (backend-aware cap fix confirmed good).
- **tests/test_wfn_loader_eager.py: 15 passed, 1 skipped (MoS2 3x3 WFN absent — expected) in 210s.**
- **src/common/wfn_loader_backend_parity_test.py: ALL PASS** (eager vs phdf5 FFI, world=4, 2x2 mesh,
  1 node x 4 ranks, --mpi=pmi2 + Intel-MPI/phdf5 env + LORRAX_FFI_HOST_SO=lorrax_ffi_wtA host lib):
  load(ibz) max|Δ|=0.0, load(full_bz) max|Δ|=0.0, gvecs exact. REQUIRED a test-side fix:
  `_replicate_to_host` used `process_allgather(tiled=False)` which raises on multi-process global
  arrays ("only supports tiled=True") → switched to tiled=True (identity for 1-proc). File:
  wt-H/src/common/wfn_loader_backend_parity_test.py (uncommitted, in worktree).
- End-to-end gates on merged branch: **PENDING** — htransform 2x2 (2 attempts) and gw_jax 4x2 both
  died in jax.distributed startup DEADLINE_EXCEEDED (holder contention: 2nd attempts self-inflicted,
  three multi-rank steps landed on shared nodes because launch_bg2.sh dropped the -w pin), then both
  holders (7874023, 7874160) expired. Last completed references: gw 4x2 eqp gate PASS max|Δ|=1.00e-06 eV
  (cpua.7873492, 96s job wall; also 7873360 at 173s); htransform 3-way bit-exact max|Δ|=0.0 (pre-merge, scorecard above).
- **Compile-cache prize (single-rank htransform, WFNsmall, ISDF_JAX_CACHE_DIR=fresh, cache enabled via
  ensure_jax_compile_cache in wrapper — NOTE htransform CLI itself never calls it, only gw_jax does):**
  COLD run completed: wall post-import 121.2s, 122 XLA compiles totaling 6.8s pure-XLA-compile,
  cache written (122 entries, 592K). WARM re-run **PENDING** (holder expired). Bound from measurements:
  warm saves ≤~7s XLA-compile + tracing per rank at fixture scale; the real prize is at production scale
  (158s-class / 2208 compiles = 16 ranks x ~138) where rank-replicated compilation dominates.
- Top-3 time sinks at fixture scale (from JAX_LOG_COMPILES + code timers, 1x1 CPU rank):
  1. FIXED overhead: process bring-up + eager dispatch — 121.2s wall vs ~4s accounted numerics
     (Galerkin 2.63s, G-accum 0.36s, SVD 0.69s, centroids 0.93s) + 6.8s XLA compile; >100s is
     import/jax-init/per-op dispatch, not physics.
  2. FIXED overhead: rank-replicated compile storm — 122 compiles/rank, problem-size-invariant
     (~138/rank at scale); killable by shared persistent cache (P>1 deadlock = separate workstream).
  3. jax.distributed startup fragility at P>1 — ~1min-class init with DEADLINE_EXCEEDED under any
     node sharing (reproduced 2x); scaling work: stagger/step-pin launches or raise init timeout.

## K — 40-node HLO/buffer profiling at 1998 centroids + the conditioning sweep (2026-07-25)

### K.0 — profiling recipe that WORKS on this jaxlib (0.9.1)
```
XLA_FLAGS="--xla_dump_to=$DIR --xla_dump_hlo_as_text \
           --xla_debug_buffer_assignment_show_max=400"     # rank 0 only
JAX_LOG_COMPILES=1                                          # rank 0 only
LORRAX_ZETA_RANK_LOG=1                                      # n_keep telemetry (default on)
```
All three XLA flags verified present in `jaxlib/libjax_common.so`'s flag table
AND by a real 1-process container run (RC=0) *before* the 40-node submit — the
`--intra_op_parallelism_threads` F-abort lesson. Gate `if [ "$SLURM_PROCID" = 0 ]`
inside the `bash -lc`; 80 ranks dumping the same ~180 modules is self-inflicted I/O.
Cost: **27 MB / 1416 files** for ~180 modules, no measurable compile slowdown.
The dump gives, per module, `*-buffer-assignment.txt` (every allocation + size,
`Total bytes used`, peak live ranges, and an **op_name stack-trace breakdown of the
peak**) and `*-memory-usage-report.txt`. Profiled sbatch:
`run4_c2000_b160/run40_run4_c2000_b160_PROF.sbatch` (original untouched); it folds
a 60 s 1-task flag probe in front of the 80-rank srun and falls back to
dump-off rather than burning the slot. Analyser: `prof_k/analyze_hlo.py`.

Job **7874385**, 40 nodes, dev, 1998c/b160, r_chunk=2160 (81 chunks).
Planner: `Devices 80 Mesh 8x10`, `persistent 1.67 GB/dev`, `HWM 18.91 GB/dev
[binder: F_tensor_write]`, route = `replicated_rank_truncate` (correct).

### K.1 — TOP MEASURED BUFFERS PER RANK vs the memory model
| # | module (op) | measured/rank | modelled | verdict |
|---|---|---|---|---|
| 1 | `_solve_all_at_once` **peak 18.88 GB**, total alloc 27.98 GB | `all-gather c128[144,2000,2000]` = **9.216 GB** + 2 x `c128[144,1998,1998]` = 9.198 GB each | L_q modelled at **0.115 GB/rank** (÷P) | **80x worse than modelled — NEW #1 offender** |
| 2 | `_fn` (replicated eigh, Stage B) peak **8.62 GB** | `all-gather c128[67,2000,2000]` 4.288 + eigh custom-call 4.279 + `V*inv_sqrt` 4.279 | runbook "eigh replicated ~12.9 GB" | **better than feared**; q-batch = 67 (3 batches: 67/67/10) |
| 3 | `_identity_fn` = `process_allgather` **8.062 GB** | `all-gather s32[11520,36,36,135]` (11520 = 80 procs x nk 144, 36x36x135 = n_rtot) | **NOT MODELLED AT ALL** | **scales with P, not mu** — 16 GB at P=160, 32 GB at P=320 |
| 4 | `_local_fft` (centroid load) peak 4.84 GB | 3 live `c128[144,2,2,36,36,135]` @ 1.612 GB | `A_centroid_load 8.12 GB` | modelled, conservative |
| 5 | Stage-C `jit_fn` peak **3.56 GB** | `all_gather c128[144,160,2,2160]` = **1.593 GB** = the J.0 band-gathered full-r psi(r) slab | slope term `2*nk*bc*ns*16` per r-unit | **only ONE slot live** — the identity-take elision works; `_GATHERED_PSI_SLOTS=2` is 2x conservative |
| 6 | `_kernel` (G->box FFT) peak 1.84 GB | fft 1.612 GB + `g_index_` param 100.8 MB | — | fine |
| 7 | `_reshard_all` peak 1.83 GB | `all-gather c128[144,2000,160,2]` = **1.475 GB** | `psi_copies` = 0.664 GB for all 4 copies | **2.7x model**; this IS the SPMD remat below |
| 8 | `_identity_fn` `c128[11520,8603]` **1.586 GB** | (nk x ngkmax) x 80 procs | NOT MODELLED | P-linear, same family as #3 |
| 9-10 | `jit_fn` pair-density 1.86 GB (`kmna,knbr->karmb`), `_kernel` 1.40 GB | — | `C_fit_one_rchunk 6.97 GB` | modelled, conservative |

### K.2 — REMATERIALIZATION: found, named, and it scales with mu
`80 occurrences` (exactly 1/rank) of
```
[SPMD] Involuntary full rematerialization. The compiler cannot go from sharding
{devices=[1,10,1,1,8]<=[8,10]T(1,0) ...} to {devices=[1,8,1,1,10]<=[80] ...}
for %transpose.0 = c128[144,200,160,2] ... op_name="jit(_reshard_all)/transpose"
-> SPMD will replicate the tensor and then partition it
```
The dim-1 extent tracks mu_pad/10 exactly (**32 at 276c -> 200 at 1998c**), so this
grows linearly with mu AND with nband. In the dump it lands as a **1.475 GB
all-gather of the whole (nk, mu_pad, nband, ns) psi_rmu** on every rank, against an
18.4 MB shard. It is an x-major <-> y-major reshard XLA cannot do efficiently
(upstream b/433785288, "fixed by Shardy in the future"). Projection: 3.69 GB/rank
at nband=400, 18.4 GB/rank at mu=10k/nb=400. **Not fatal yet, but it is real remat
and it is the only one the compiler flags.**

### K.3 — the NEW #1 lim P->inf offender: `isdf/core.py:1930 _solve_all_at_once`
```python
@jax.jit
def _solve_all_at_once(L_q_sharded, Z_col):
    L_full_rep = jax.lax.with_sharding_constraint(L_q_sharded, L_batch_rep_shard)
    return _sharded_cho_solve_batch(L_full_rep, Z_col)   # in_specs=(P(None,None,None), ...)
```
It takes the correctly-sharded `L_q` (115.2 MB/rank, exactly the model's ÷P term),
**all-gathers the entire (nq, mu_pad, mu_pad) factor to 9.216 GB on every rank**, then
`_pinv_matmul_logical` makes two more full-size copies (the slice to logical 1998 and
the conjugate, via f64 `real`/`imag`/`negate` at 4.599 GB each). **Peak 18.88 GB/rank,
27.98 GB total allocated, entirely un-sharded.** Scaling `3*nq*mu^2*16`:
mu 2000 -> 18.9 GB (measured) | 2400 -> 27.2 | 3000 -> 42.5 | 4000 -> 75.5 |
**~4400 -> 92 GB = the envelope.** This bites *before* wall #1's eigh time and before
wall #2's writer. The bounded alternative **already exists two functions up** —
`_solve_batch_and_update` (line 1921) does the identical thing per q-batch with
`donate_argnums` — so the cheapest fix is a size gate that routes to it past some
`nq*mu^2` bound; the real fix is a distributed matmul back-solve.

### K.4 — 1998c TRUNCATION TELEMETRY (the conditioning datum)
```
[zeta rank_truncate] n_log=1998 rcond=1.0e-08
  n_keep/q = 1011 ... 1018   (median ~1014, over the 67-q batch)
  lam_max/q   ~ 2.054e-02   (flat to 5 digits across q)
  lam_min_kept ~ 2.06e-10   -> lam_min/lam_max = 1.00e-08 == rcond exactly
```
vs **276/276 kept at mu=276**. So: **at mu=1998 the rank truncation discards ~49% of
the basis**, and `lam_min_kept/lam_max` sits exactly at rcond — i.e. the spectrum is
*continuous* through the cut, there is no spectral gap, the cut is set purely by
`zeta_rcond`. **The numerical rank of the pair density has saturated near ~1015 at
nband=160.** Direct consequence: going 1194 -> 1998 -> 2406 centroids adds
essentially no new basis directions, so **run6_c2400 is near-worthless** and the
interesting question is whether the ~1015 ceiling is set by the *band window* ->
**run5_*_b400 is the informative rung, not run6**.

### K.5 — cadence, and why 7874385 was cancelled
Preamble (import -> zeta-fit start) 1353 s incl. the whole replicated eigh
(much faster than the runbook's mu^3 upper bound of ~41 min).
zeta-fit cadence at r_chunk=2160: chunks 1->5 in 798 s = **~200 s/chunk**, ETA 12126 s
=> **81 chunks = ~4.5 h against a 2 h dev wall**. Since `zeta_q.h5` is written only
after the whole fit, the run could produce no zeta and no eqp. Cancelled at 39 min
with every profiling deliverable already extracted.
**81 chunks is 1.74x more expensive than the work requires** (276c did 9 chunks x 165 s
= 1489 s; pure mu-scaling predicts 9300 s, not 16200 s) — the extra is per-chunk fixed
overhead. Wall #0's 9.9x over-ask is now MODELLED (the gathered-psi term, J.1) and the
measured Stage-C module peak is only **3.56 GB/rank at cr=2160**, so
**r_chunk should go back up to 6480 (27 chunks)** for run4/5/6: ~11 GB/rank Stage C,
and the fit drops to ~2.9 h. All sweep dirs below use 6480.

### K.6 — BLOCKER for the QP-vs-mu conditioning test (found while building it)
`eqp0.dat` / `eqp_g0w0.dat` at MoS2 12x12 are **numerically broken in absolute terms**,
with the CORRECT `replicated_rank_truncate` route and a completed 276c run (7874386):
- DFT indirect gap parses correctly as **1.7010 eV** (matches the reference), so the
  file format / band indexing is fine (nval=26, VBM=band 26).
- **QP indirect gap = -122.13 eV.** eqp col 4 is `Re[H0 + Sigma_xc(E_DFT)]`, and the
  `VH` term inside H0 is per-band nonsense: it ranges **100.98 -> 505.51 eV** across
  bands in one run. Runbook sanity gate #4 (QP gap positive) therefore FAILS at the
  baseline rung for reasons that have nothing to do with ISDF conditioning.
- `Sigma_xc` itself is **sane and usable**: at 276c, VBM (Eo=-5.2527) `sigX=-15.687
  sigC=+0.612 sigXC=-15.075`; CBM (Eo=-3.5517) `sigX=-7.410 sigC=-1.683 sigXC=-9.092`;
  gap correction `dSigXC = +5.983 eV`.
=> **Use `dSigXC` from `sigma_diag.dat`, not the eqp gap, as the conditioning
observable** until the H0/VH bookkeeping is fixed. Script: `prof_k/sigma_sweep.py`.

### K.7 — the centroid sweep now running (what the ladder should have been)
Normal queue (does NOT touch the dev budget), 40 nodes, 8 h, nband=160, gn_ppm,
`restart=false`, r_chunk=6480, identical in every other respect:
`sweep_c606` **7874609**, `sweep_c1194` **7874610**, `sweep_c1998` **7874611**,
`sweep_c2406` **7874612**  (+ the existing 276c point in `run1/`).
Read out with `python3 prof_k/sigma_sweep.py` -> `dSigXC` vs n_mu. Smooth monotone
convergence = basis still meaningful; scatter/sign flips = it is not. Given K.4
(rank saturates ~1015), the prediction is that dSigXC stops moving somewhere around
mu ~ 1200 and starts to jitter at 1998/2406.

### K.8 — go/no-go
- **run5_c2000_b400: GO**, but re-pin `r_chunk = 6480` and put it on the **normal**
  queue with an 8 h wall, not dev/2 h. It is the rung that tests whether the ~1015
  rank ceiling is a band-window artefact.
- **run6_c2400_b160: NO-GO as a physics rung.** n_keep already discards half the basis
  at 1998; 2406 centroids at nband=160 buys nothing. (`sweep_c2406` is running purely
  as the noise end-point of the conditioning curve.)
- **run4 as staged (dev/2 h/81 chunks): NO-GO** — it cannot finish; 4.5 h of fit.

### J.7 — sigma band-window mesh pad + restart band-window guard

**Fix 1 — pad the QP band window to the mesh (the 7874338 Phase-1 guard).**
`ppm_tau_kernel._make_project_ri_reduce_scatter` reduce-scatters m over `'x'`
and n over `'y'`, AND `ppm_accumulators._MemoryTileSink` holds
Sigma_c(w,k,m,n) at `P(None,None,'x','y')` — so both need `m % p_x == 0` and
`n % p_y == 0`.  `common/meta.py` rounds `b_id_4` (the FULL window) to
`world_size` but never the sigma window, which is
**`b3 - b0 = nelec + ncond`** (NOT `nval + ncond` — worth knowing, it is why
this is so easy to trip).

New `ppm_sigma.pad_sigma_window` / `strip_sigma_window`: zero-pad the band axis
of `psi_proj_xr` (axis 1) and `psi_proj_yn` (axis 3) up to a multiple of
`p_x·p_y`, run the whole branch at the padded extent, and strip at ONE seam —
`_run_sigma_branch`'s return.  `nb_proj` stays the real window everywhere the
caller can see, so the host Sigma buffer and the eqp writer never see the pad.
Exact by construction: every output `Sigma[k,m,n]` is an INDEPENDENT
contraction `psi*_m . sigma . psi_n`, so appending zero bands adds zero
rows/cols without perturbing any existing element.  `precompile_sigma` pads
identically (else the AOT signature diverges from runtime and pjit re-traces).
`_H5Sink` asserts no pad (KIJ_STREAM is single-process ⇒ 1x1 mesh ⇒ no pad).

**Fix 2 — restart band-window provenance guard.**
`write_restart_state_to_h5` now stamps `band_window` (b0,b1,b2,b3,b4) and
`n_rmu_logical`; `assert_restart_window_matches` refuses a load whose window or
centroid count differs, naming BOTH windows.  This is the job-7874375 failure
mode: window-70 tensors reused at window 80 gave a **QP gap of -135 eV while
every stage reported success**.  Files without the attrs pass through
(back-compat, so existing restarts are not stranded).

**Gates — GN-PPM path, job 7874393 (`compute_mode = gn_ppm`, mesh 2x4):**

| run | src | window | result |
|---|---|---|---|
| 1 | MAIN `658b0de` | 30 (stock fixture) | **rc=1 — guard fires** |
| 2 | wt-J | 30 | **rc=0 — repaired** |
| 3 | MAIN `658b0de` | 29 (ncond 4->3) | **rc=1 — guard fires** |
| 4 | wt-J | 29 | **rc=0 — repaired** |
| 5 | wt-J, P=1 (1x1) | 29, unpadded | rc=0 — reference |

Exact pre-fix error (job 7874613, **stock** fixture, no modification):
`sigma reduce-scatter needs the band window divisible by the mesh: m=30 must be
a multiple of p_x=2 and n=30 of p_y=4`.

**Physics gate — padded P=8 vs unpadded P=1: `max|Δ| = 0.00e+00 eV`,
BIT-IDENTICAL over 2349 values.**  Exactly as the independence argument
predicts.

Note the bug was reachable on the STOCK regression fixture at any mesh whose
`p_y` does not divide 30 — it was never exercised only because the fixture is
normally run on the COHSEX path (`compute_mode` default resolves to cohsex,
whose `wavefunction_bundle.project` has no reduce-scatter and no divisibility
requirement).  The GN-PPM path had no multi-device regression coverage at all.

**Restart guard unit (job 7874382): ALL PASS** — matching window passes;
b3 70->80 raises and names both; changed `n_rmu` raises; attr-less legacy file
passes through.

## M — root cause of the MoS2 12x12 H0/VH corruption (2026-07-25)

**VERDICT: not a parallel bug, not an aux-file dim mismatch, not a units/indexing
slip. `H0 = <T+V_ion+V_NL> + <V_H>` is a CATASTROPHIC CANCELLATION of two ~500 eV
terms computed by two DIFFERENT numerical routes, and the ISDF route (V_H) is not
converged at 276 centroids. This has been true of EVERY 12x12 run ever made,
including the "proven reference" `run_276c`.**

### Data flow (static trace)
`eqp0.dat` col-4 = `eqp_g0w0.dat` Re column = **the same number**, both built in
`gw/gw_output.py::write_results`:
- `kin_ion_diag_ev = diag(results.kin_ion_ry)*RYD_TO_EV` <- `file_io/kin_ion.py::
  load_kin_ion_submatrix` reads `kin_ion.h5[band_start:band_stop]`, EXACT plane-wave
  `T + V_loc + V_NL` written by `gw/kin_ion_io.py` (no V_H, no ISDF).
- `hartree_diag_ev = diag(sig_h)*RYD_TO_EV` <- `gw/cohsex_sigma.py::
  _make_cohsex_kernels.hartree` (:109-121), a **pure ISDF centroid quadrature**:
  `rho[mu]=sum_k psi*_i psi_j G_ij` at centroids -> `Vrho=V_q[0]@rho/nk` ->
  `<m|V_H|n> = sum_mu psi*_m(r_mu) Vrho[mu] psi_n(r_mu)`.
- `eqp0 = kin_ion + V_H + sigX + Re sigC(E_DFT)` (`gw/eqp_bgw.py::compute_eqp_diag`).
No process_index / world-size / pad dependence anywhere on this path.

### Discriminating evidence (all from on-disk artifacts, zero new jobs)
1. **`eqp_g0w0.dat` is NOT sane** — it is bit-identical to eqp0 col-4. Both carry H0.
2. **A/B verdict (job 7874616, 4x4 mesh / P=16 restart)**: `VH[k0,n0] = 505.510617`,
   identical to v8 at 8x10 / P=80 **to all printed digits**. Deterministic,
   mesh- and process-count-INDEPENDENT. P=80 machinery exonerated.
3. **Aux files are dimensionally FINE**: `kin_ion.h5` = (144,120,120), `dipole.h5`
   `deltaE`(144,120,120)/`dipole_cart`(3,144,120,120). Window needs b3=80 <= 120.
   (Minor: the q=0 head S(omega) is therefore built from a 120-band transition
   space while the run's polarizability window is nband=160 — under-converged head,
   not the H0 bug.)
4. **GROUND TRUTH EXISTS ON DISK**: `/scratch2/08271/jackmc/mos2_80ry_12x12/out/kih.dat`
   (pw2bgw `<nk|T+V_ion+V_H|nk>`, eV) and `out/vxc.dat`. Check: `kih + vxc - E_DFT`
   = 0.008..0.02 eV. So the exact per-(k,n) H0 is known.

| run | mu | nband | P/mesh | dH0 = H0_LORRAX - kih_QE, k=0 (eV) |
|---|---|---|---|---|
| run1 (v8, job 7874386) | 276 | 160 | 80 / 8x10 | min -45.1 max **+179.7** rms **46.6** |
| ab_4x4 (job 7874616) | 276 | 160 | 16 / 4x4 | **identical: rms 46.6** |
| run_276c ("reference") | 276 | 120 | - | min -149.9 max +69.3 rms **47.1** |

5. **kin_ion.h5 is EXONERATED**: `VH_true = kih_QE - kin_ion_LORRAX` comes out
   positive, smooth and physically ordered (460 eV for the Mo semicore n=0 down to
   ~30 eV for the vacuum-localized n=40/70) — the two codes share the G=0/alpha-Z
   convention. All of dH0 sits in the ISDF `sig_h`.
6. **The fixture is accurate, so there is no code bug**: `tests/regression/gnppm_debug`
   (mu=399, nk=9, nband=46) reproduces implied `Vxc = E_DFT - kin_ion - V_H` in
   **[-28.3, -6.0] eV** over all 414 (k,n) — within ~1 eV of QE's `vxc.dat` for the
   same material. The 12x12 gives **[-191.6, +31.7] eV**.
7. **The ISDF solve itself is healthy**: log shows `n_keep/q = 276` for every q
   (`rcond=1e-8`) — full rank kept. The basis is simply too SMALL, not truncated.
   Centroids/band: fixture **399/46 = 8.7**, 12x12 **276/160 = 1.7** (5x worse), and
   the 12x12 must also serve 144 k-points vs 9.
8. **nband-sensitivity confirms it is the ISDF and nothing else**: identical
   `kin_ion.h5`, identical density, but `VH[k0,n0]` = 367.36 (nband=120) vs 505.51
   (nband=160) vs **460.55 true**. Only zeta changed.

### Root cause
For MoS2 the Mo 4s/4p semicore states give `<T+V_ion+V_NL> = -502 eV` and
`<V_H> = +461 eV`; `H0 = -42 eV` is their 1%-level difference. `kin_ion` is exact;
`V_H` is an ISDF centroid quadrature. A ~10% ISDF pair-product error at 276
centroids therefore lands on H0 as **~50 eV rms**, wrecking every QP energy
(gap -136 eV) while Sigma — which never differences against an exact quantity —
still looks physical. Silent by construction: every stage reports success.

### Fix
**Implemented (wt-J, `mu-replication-audit`, `src/gw/gw_output.py`) — H0 sanity gate.**
New `_warn_on_unphysical_h0()`, called from `write_results` before the eqp writers.
Uses the exact DFT identity `E_DFT = kin_ion + V_H + V_xc`: prints the implied Vxc
range every run and raises a loud WARNING outside [-50, +2] eV. **Print-only — zero
numerics touched.** Gated on artifacts:
- gnppm fixture (mu=399): `implied Vxc in [-28.281, -6.027] eV` -> **silent, PASS**
- run1 k=0: 36/80 bad, worst k=0 n=70 Vxc=-191.6 -> **fires**
- run_276c k=0: 42/70 bad, worst Vxc=+134.5 -> **fires** (would have caught this
  on 2026-07-23, before four days of production runs)

**Designed (structural, NOT implemented) — make H0 exact, kill the cancellation.**
Never compute the mean-field V_H by ISDF. Evaluate `<mk|V_H|nk>` on the FFT grid
like any local potential (`psp/get_DFT_mtxels.compute_local_V_k`), folding it into
`kin_ion.h5` at generation time so the ~500 eV G-space cancellation closes
analytically inside one exact routine. Most of the machinery exists:
`psp/get_DFT_mtxels.get_kin_ion(include_hartree=True)` already does exactly this.
Three gaps to close:
  (a) the production CLI `gw/kin_ion_io.py` has NO Hartree branch at all;
  (b) `get_DFT_mtxels.py:673` hardwires `compute_hartree_potential_real(...,
      truncation_2d=False)` with the comment *"Set to True for 2D slab truncation
      matching ISDF"* — a latent inconsistency for every sys_dim=2 deck, while
      `kin_ion.h5` is stamped `truncation_2d=TRUE`;
  (c) no provenance attr, so `write_results` would double-count (it unconditionally
      adds `sig_h`). Stamp `has_hartree` on `kin_ion.h5` and skip `sig_h` when set.
Cost is one extra local-potential matrix-element pass over the sigma window —
negligible next to Sigma — and it makes eqp0 centroid-count-independent.

**Interim for production:** raise the centroid count. The in-flight sweep
(jobs 7874609-12, mu = 606/1194/1998/2406) is the confirming experiment; evaluate
each with `dH0 = (kin_ion.h5 diag * 13.6057 + VH from sigma_diag.dat) - kih.dat`
against `/scratch2/08271/jackmc/mos2_80ry_12x12/out/kih.dat` and require
rms << 1 eV before trusting any eqp0. Note mu=1998 is still only 12.5 centroids
per band vs the fixture's 8.7 but against 16x the k-points — do not assume it
converges; the exact-V_H fix is the durable answer.

## Workstream L — 2D distributed-linalg FFI made REAL on Frontera CPU (downfolding readiness)

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_L/` (`build_slate_host.sh`,
`smoke.py`, `diag.py`, `downfold_bench.py`, `bench_*.json`, job logs
`build.7874407 / diag.7874620 / seq.7874653 / bench2.7874654`).

### 1. BUILD — first SLATE on Frontera, and the first complete host FFI lib
The premise that "SLATE compiled on this machine before" is **FALSE**: the old
`lorrax_ffi/build/liblorrax_ffi.so` exports only `cusolvermp` symbols and
`config/frontera/build_ffi.sh:48` pins `-DLORRAX_SLATE_INSTALL_DIR=$STAGE/_no_slate`
(a deliberately non-existent dir). SLATE had only ever been built on Perlmutter.

Built SLATE v2025.05.28-1 (`ded15290`), `gpu_backend=none`, inside the py312
container with g++-12 + **MKL 2020.1 LP64** + Intel MPI 2020.4 →
`$WORK/slate_builds/cpu/install` (16 MB `libslate.so.2`). Two deviations from
`build_perlmutter.sh`, both required: (a) `-Dblas=mkl` instead of `libsci`;
(b) SLATE declares `LANGUAGES CXX Fortran` but uses Fortran **nowhere** (only the
commented-out `scalapack_api`) — the container has no gfortran, so the script
patches it to `LANGUAGES CXX`. `-Dbuild_tests=no`, `-DCMAKE_INSTALL_LIBDIR=lib64`.

Unified host lib → `$WORK/lorrax_ffi_unified/build_host/liblorrax_ffi_host.so`
(528 K, CUDA-free). ScaLAPACK comes from MKL and is **not** wired up by
`src/ffi/common/cpp/host/CMakeLists.txt` — it had to be injected via
`-DCMAKE_SHARED_LINKER_FLAGS="-lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 …"`.
`nm -D` proof — all 9 `_HOST_TARGET_SYMBOLS` present as `T`:
`PhdfRead{,Kchunk,KchunkUnion}HostFfi`, `Slate{Eigh,Potrf,Trsm,BatchedPotrf,BatchedTrsm}HostFfi`,
`ScalapackBatchedSolveLuHostFfi` (+ `lrx_slate_init_mpi`, `lrx_slate_context_create`).
Runtime needs `LD_LIBRARY_PATH` ⊇ MKL + **Intel compiler RT** (`libimf/libsvml/libirng/libintlc`) + slate lib64.

### 2. FACADE — works, on the real library, first time on CPU
16 ranks / 8 nodes, `JAX_PLATFORMS=cpu`, `LORRAX_FFI_HOST_SO`=unified lib:
`ffi_loader.has_target` → **True for all 9 targets**; on a 4×4 CPU mesh
`list_backends` reports `slate: available` / `scalapack: available`, and
`resolve_backend('eigh','slate',4x4) → 'slate'`, `('cholesky','slate') → 'slate'`,
`('solve_lu','scalapack') → 'scalapack'`. Capability probing, platform guard and
`auto`→`native` CPU-safety all behave exactly as documented.

### 3. GEOMETRY — the headline constraint for a 8×10-shaped production mesh
**No SLATE/ScaLAPACK op runs on a rectangular mesh.** Measured rejections (clean
`ValueError` at resolve time, no hang):
- `eigh` + slate on 2×4 / 4×2 / 2×8 / 8×2 → rejected (square-mesh guard).
- `solve_lu` + scalapack on 2×4 / 4×2 → rejected (square-or-1D descriptor guard).
- Usable shapes on CPU: **square (2×2, 4×4) or N×1**. `1×q` is rejected (SLATE stride assert).

**BUG L-1 (guard inconsistency, resolve-time promise violated).**
`resolve_backend('cholesky','slate', mesh 2×4)` returns `'slate'`, but the very
next call raises. `ffi/linalg/resolve.py::_check_geometry` only rejects `px==1 and py>1`
for slate cholesky, while `ffi/slate/context.py::validate_tile_layout:123`
(call time) rejects **`p>1 and q>1 and p!=q`**. This breaks the documented
contract ("a returned FFI name is a *promise*… the subsequent call cannot fail
for an availability/geometry reason"). Repro: `wk_L/smoke.py --shapes 2x4`.
Fix: mirror the `p!=q` rule into `_check_geometry`'s `backend == "slate"` branch.

### 4. BUG L-2 (blocker) — SLATE host `heev` SIGSEGVs; everything else is clean
`lorrax_slate_eigh` → `slate::heev(…, Target::HostTask)` **segfaults rank 0,
deterministically**, at every configuration tried:
n = 64 / 512 / 1200; mesh 1×1, 2×2, 4×4; intra-node (shm) and inter-node;
`compute_evecs=True` **and** `False` (so it is *not* the Z back-transform /
tile copy-out path in `host_ffi.cc`); and with SLATE built both
`-Dblas_threaded=true` (mkl_gnu_thread) **and** `false` (mkl_sequential).
Because it reproduces on a **1×1 mesh, single rank, single process**, it is not
MPI, not the comm remap, not the LORRAX layout contract.
Excluded: `MPI_Query_thread` → **3 (MULTIPLE)**; `ldd -r` shows **no** unresolved
symbols in `libslate/liblapackpp`; MKL 2020.1 exports the full 2-stage set
(`zhetrd_he2hb`, `zhetrd_hb2st`, `zhb2st_kernels`, `zstedc`, `zsteqr`, `zunmtr`).
Remaining suspect: SLATE 2025.05's host `heev` against MKL's LAPACK 3.8 (the
Perlmutter validation was Cray LibSci). Repro:
`wk_L/diag.py --px 1 --py 1 --ops mpi,ctx,potrf512,eigh512`.
**On the SAME library/context, `potrf`, `trsm` and ScaLAPACK `getrf/getrs` all
work**: potrf n=512 residual `1.47e-16` (2×2), `1.84e-16` (1×1).

### 5. DOWNFOLDING BENCH — 4×4 mesh, all arrays face-sharded `P('x','y')`, c128
`S_cross^H W_big S_cross` (μ_big→μ_small) + cholesky + solve_lu, 10 reps.
GEMMs are one `shard_map`: all_gather + local GEMM, and SUMMA-style
`psum_scatter` reduce-scatter for the `A^H B` leg. Nothing ever replicated.
**Median seconds over reps (first call in parens); 8 nodes × 2 ranks:**

| op (size) | FFI backend | FFI median | native median | FFI win |
|---|---|---|---|---|
| cholesky (400) | slate | **0.111** (2.48) | 0.204 | 1.8× |
| solve_lu (400, nrhs 200) | scalapack | **0.256** (1.48) | 1.563 | **6.1×** |
| cholesky (400), μ_big=3000 run | slate | 0.116 (2.66) | 0.071 | 0.6× |
| solve_lu (400, nrhs 200), μ_big=3000 run | scalapack | **0.240** (1.82) | 2.766 | **11.5×** |
| eigh (1200) | — (L-2) | — | 0.929 | — |
| eigh (3000) | — (L-2) | — | 10.19–11.44 | — |
| gemm W·S (1200) | native shard_map | 0.070 | 0.070 | — |
| gemm S^H·T (1200) | native shard_map | 0.031 | 0.031 | — |
| gemm W·S (3000) | native shard_map | 0.333 | 0.342 | — |
| gemm S^H·T (3000) | native shard_map | 0.050 | 0.048 | — |

1 node × 16 ranks (pure shm MPI, 3 threads/rank), μ_big=1200: cholesky[slate]
0.142 vs native 0.020; solve_lu[scalapack] 0.416 vs native 0.429 — **the FFI
advantage is a multi-node effect**: it grows with rank spread, because native
gathers the tile while the FFI keeps it distributed.

**Numerics: perfect, and stable over all 10 reps** (vs replicated numpy at μ=1200):
`W_down` 9.29e-16, cholesky 1.48e-16, solve_lu 3.96e-16, eigh evals 2.78e-15,
eigh residual `‖AQ−QΛ‖/‖A‖` 3.69e-15. **Zero hangs** in any op/mesh/size.
Per-call FFI floor 0.10–0.26 s; first call 1.4–2.7 s (ctx create + XLA compile),
amortized from rep 1 — so hoist `resolve_backend` and the first call out of loops.

**This contradicts `docs/dev/linalg_ffi.md`'s "FFI is 100–600× slower" line** —
that was measured for *batched* eigh on GPU. For a *single* tile on a CPU mesh
the FFI wins; the doc's claim should be scoped to the batched-GPU regime.

### 6. VERDICT — downfolding readiness on Frontera CPU
- **SOLID:** `cholesky` (slate) and `solve_lu` (scalapack) at n = 400–1200, and
  the sharded `S^H W S` GEMM chain at μ_big = 1200 and 3000. 1e-16-class
  residuals, no drift, no hangs.
- **BLOCKED:** distributed `eigh` via FFI (bug L-2). Use `eigh_backend=auto`
  (native `jnp.linalg.eigh`) — measured 0.93 s at n=1200 / ~11 s at n=3000, correct
  to 3.7e-15. Perfectly usable; just not distributed.
- **MESH:** the downfolding driver must run FFI linalg on a **square** (or N×1)
  mesh. A production 8×10 gets `native` for everything. If FFI cholesky/LU is
  wanted, choose 64 ranks (8×8) or 100 (10×10), and keep n divisible by both axes.
- **DO:** resolve once outside the loop; keep everything `P('x','y')`; treat the
  first FFI call as a ~2 s warm-up. **Note:** `scalapack.batched_distributed_solve_lu`
  **donates BOTH A and B** — rebuild them if you need residuals (cost us a rerun).
- **CAVEAT:** all inter-node numbers used `FI_PROVIDER=tcp` (IPoIB), an rtx/mlx4
  workaround. Frontera CLX is ConnectX-6 and TACC's own `impi` modulefile sets
  `FI_PROVIDER=mlx` + UCX tunings; TACC also documents `ibrun`, not
  `srun --mpi=pmi2`, as the supported container-MPI launcher. The FFI latencies
  above are therefore **pessimistic** — retest on `mlx` before tuning.

### J.8 — wall #0 STRUCTURALLY CURED: ψ-gather all-to-all (implemented + gated)

`isdf/core.py` step (3) no longer gathers bands at the FULL r-chunk and slices
r afterwards.  The movement it needs IS an all-to-all, so it is now written as
one:

```
(3a) lax.all_to_all(psi_local, 'y', split_axis=r, concat_axis=band, tiled=True)
(3b) lax.all_gather(.., 'x', axis=band, tiled=True)
```

Rank (x,y) ships its y'-th r-block to (x,y') and receives every (x,y'')'s bands
at r-block y; the remaining band blocks come from the 'x' gather.  Same byte
volume, pure data movement, and the r-slice disappears.

**Band order is preserved exactly** — load-bearing for `g_axis`,
`y_compact_idx` and the `psi_*_X` slice.  all_to_all concatenates sources in
'y' order, all_gather in 'x' order, so block (x,y) lands at
`(x·p_y + y)·bpd_max` — precisely where `all_gather(('x','y'), tiled=True)`
put it (that flattens row-major with 'x' slowest).

| | per-rank bytes | at run1 (nk=144, bc=160, ns=2, cr=174960) |
|---|---|---|
| before | `nk·bc·ns·cr·16` | 129 GB |
| after | `nk·bc·ns·cr·16 / p_y` | **12.9 GB** (p_y=10) |

Stage-C slope `489.6·μ + const`, const **1,474,560 → 156,672 B/cr** (9.4×).
Planner `_stage_C_slope` updated (`shard=p_y` on the gathered slab, plus the
all-to-all source term `nk·(bc/p_xy)·ns`), `_GATHERED_PSI_SLOTS` kept at 2.

**Effect at MoS2 12×12 / 276c: r_chunk goes from 44,291 (4 chunks) to the
FULL 174,960 — one chunk.**

**Gates (job 7874705) — ALL PASS:**
- GW eqp vs `eqp_ref.dat` 1e-3: P=4 **1.0e-6 eV**, P=8 **1.0e-6 eV**.
- htransform vs ground truth: P=4 and P=8 **max|ΔE| = 0.000e+00, bit-exact**.
- Mesh-invariance, eqp P=4 vs P=8 direct: **max|Δ| = 0.00e+00** — bit-identical
  across device counts, confirming the band-order argument empirically.

### J.9 — SUMMA back-solve: ATTEMPTED, REVERTED (design corrected)

Implemented, gated, and it **failed the gate** — reverted.  Worth recording
precisely, because the design note in J.3 was wrong in a way that is not
obvious:

`Z` enters the back-solve at `P(None, None, ('x','y'))` — columns sharded over
the **flat** mesh.  So ranks sharing a `y` index hold **different** column
blocks, and the `psum` over `'x'` in a 2-D SUMMA sums partial products built
from unrelated columns.  Result: NaNs (the gate caught them as float-count
deficits in eqp — 14 at P=4, 26 at P=8; `rc=0`, no crash, silent garbage).

The constraint the corrected design must respect: **a block-sharded (μ,μ)
factor requires ranks that share a column block to cooperate on the μ
contraction.**  With columns spread over all P, every rank needs the whole
operator.  So the fix is not a drop-in — it must also change Z/ζ to
columns-on-'y' (μ rows on 'x'), which changes the downstream
`_reshard_zeta_r_XY_to_mu_XY` plan, a path with documented Involuntary-Remat
OOM history.  That is a real piece of work, not a mechanical swap, and it
should be done together with the distributed eigh rather than bolted onto the
replicated one.

The all-gather it targets remains: `with_sharding_constraint(L, P(None,None,None))`
at `isdf/core.py`, `nq·μ²·16` per r-chunk (230 GB at μ=10k).

### J.10 — the μ ladder after J.8 (MoS2 12×12, P=80, target 72.2 GB/dev)

| wall | ceiling | status |
|---|---|---|
| #0 ψ-gather (Stage C) | μ ≈ 65k → **68k**, and 4 chunks → **1 chunk** | **CURED (J.8)** |
| `F_tensor_write` — SlabIO allgather, `2·nq·μ²` UNSHARDED | **μ ≈ 3,960** | **now the binding wall**; ENVIRONMENTAL fix (mpi4py + `HDF5_MPI=ON` h5py ⇒ PHDF5_HOST) lifts it to **μ ≈ 50,100** |
| replicated rank-truncate eigh — O(nq·μ³) with NO P-scaling | **μ ≈ 4,000** (TIME, not memory: ~5.5 h at 4k, ~86 h at 10k) | needs the distributed eigh (ScaLAPACK `pzheevd`; SLATE host heev SIGSEGVs per workstream L) |
| `B_cct_chol` `(nq + 2·nk·ns²)·μ²/P` | μ ≈ 16,700 | next after those two |

**So J.8 did not raise the practical ceiling on its own — it removed the wall
that was capping it at ~4k *for a different reason* (chunk quantisation), and
the two ~4k walls now standing are (a) an environment fix and (b) the
distributed eigh.** Doing (a) costs a venv rebuild and buys μ ≈ 3,960 → 16,700
(B_cct_chol then binds); doing (b) as well is what opens 10k+.

## N — exact mean-field V_H folded into kin_ion (wt-J, `mu-replication-audit`) + the H0 error decomposition (2026-07-25)

Implements M's designed structural fix, and then **overturns M's attribution**:
the ISDF V_H was *not* the dominant H0 error on the 12×12, and the Coulomb
convention is *not* mismatched. Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_N/`
(`analyze_h0.py`, `analyze_fixture.py`, `vh_convention_probe.py`, `term_probe.py`,
`kin_ion_vh160.h5`, `kin_ion_nohar160.h5`, `vh_conv_12x12.npz`, jobs
`genA.7874704 / genB.7874710 / genD.7874797`).

### Implementation (5 files, worktree only, NOT committed)
- `psp/get_DFT_mtxels.py` — new `spin_degeneracy_factor` (2 only for nspin=1 ∧
  nspinor=1; `wfn.nelec` is a *band* count, so this is what turns it into
  charge — silently wrong by 2× before), `valence_density_from_kpoint`
  (single-sourced per-k ρ quadrature), `build_hartree_potential` (Poisson +
  **hard `∫ρ d³r` check**, raises >0.1 % — the cheap guard against a factor-2 or
  grid-normalisation slip in a ~500 eV term). `get_kin_ion` gap (b) closed:
  `truncation_2d=ctx.truncation_2d` instead of the hardwired `False`, plus a
  grid-mismatch guard (the old path would have silently broadcast-failed
  whenever `wfn.grid_rho` was the 2× grid — i.e. `include_hartree=True` had
  never actually run).
- `gw/kin_ion_io.py` — gaps (a)+(c) and the **full parameter-inheritance
  mandate**. `--hartree` (DEFAULT ON) / `--no-hartree`; chunked
  `build_valence_density_chunked` (one k of ψ resident); `--sys_dim` may only
  *confirm* the deck, never contradict it (raises); band floor `nelec+ncond`
  enforced against the deck; `grid_rho` vs FFT-box guard; provenance stamped:
  `has_hartree`, `hartree_truncation_2d`, `input_file`, `wfn_file`, `nval`,
  `ncond`, `nband_input`, `nelec_bands`, `bispinor`, `nspinor`, `fft_grid`.
- `file_io/kin_ion.py` (+`__init__` exports) — `read_kin_ion_provenance`,
  `kin_ion_has_hartree`, `validate_kin_ion_against_run` (refuses a sys_dim /
  nk / band-window disagreement at load time, prints which mode is active).
- `gw/sigma_dispatch.py` — **the single no-double-count seam**: where `sig_h`
  enters `SigmaResult`, zero it when `has_hartree`. One line covers eqp{0,1},
  `sigma_total = Σ_xc + V_H` (the eigh operand), the fixed-point h₀, the SC
  iteration map and `sigma_diag.dat` — consistent by construction, not by each
  consumer remembering.
- `gw/gw_output.py` — `GWResults.kin_ion_has_hartree`; `write_results` skips
  the Hartree diagonal explicitly; `_warn_on_unphysical_h0` keeps working in
  both modes and now prints a *different diagnosis* per mode (telling an
  exact-V_H user to raise the centroid count would send them down the wrong
  path). `gw/gw_jax.py` — validate + plumb the flag.

### Old-vs-new contract
`has_hartree=True` ⇒ ⟨mk|V_H|nk⟩ is inside `kin_ion`; the GW run adds **no**
`sig_h` (`sigma_diag.dat`'s VH column reads 0.000 *by design*). Attribute
absent/False ⇒ legacy semantics, unchanged. Back-compat is safe in the
dangerous direction only by the guard: a NEW file read by OLD code would double
count, but `_warn_on_unphysical_h0` fires hard on it.

### Gate 1 — fixtures (host-side, no GW run needed)
**gnppm (μ=399, nval=26=nelec, the converged fixture) — the decisive validation
of the new route:** VH_ISDF vs the new exact FFT-grid V_H over 414 (k,n),
values up to 594 eV: **rms 2.30 eV, mean +0.60 eV (0.5 %)**. Two completely
independent methods agree. Implied Vxc [−28.281, −6.027] (old) → **[−30.021,
−6.356] (new)**, 0/414 outside the physical window either way.
**cohsex fixture (nval=4 < nelec=26):** implied Vxc [−144.98, +88.42] eV, 86/120
bad (old) → **[−6.273, −1.539] eV, 0/120 bad (new)**. Also found: with
`nval < nelec` the ISDF `build_Gij` occupation projector marks the first
`min(nelec, nb_sigma)` *window-local* bands occupied, i.e. global 22..29 —
4 conduction bands in, 22 occupied bands out. The fixture's ISDF ρ is
qualitatively wrong; only decks with `nval == nelec` (12×12, gnppm) are safe.
Its committed `kin_ion.h5` is also stale (210 eV rms vs a fresh regen).

### Gate 2 — THE 12×12 gate vs `out/kih.dat` (144 k × 120 bands)
| H0 route | rms(H0 − kih) |
|---|---|
| legacy kin_ion + ISDF V_H @276c (production) | 50.76 eV |
| **exact V_H, 2D-truncated (the fix)** | **38.69 eV** |
| exact V_H, 3D periodic | 115.68 eV |
| best scalar 2D/3D mix (a=0.80) | 26.65 eV |

**GATE NOT MET (38.7 eV ≫ 1 eV) — and the residual is not V_H.**
Regenerated legacy file is **bit-identical** (max|Δ| = 0.000e+00) to the on-disk
120-band `kin_ion.h5`, so the refactor is a proven no-op and the on-disk file is
exactly what current code produces.

### THE ERROR DECOMPOSITION (the owner's criterion)
1. **Coulomb convention — NOT the problem, hypothesis disconfirmed.** QE's
   `scf.in`/`nscf.in` both carry `assume_isolated='2D'`, so the truncated
   (Ismail-Beigi/Sohier) Coulomb is the *correct* convention for H0 assembly,
   and it is what the fix uses. Measured: 3D is **3× worse** (115.7 vs 38.7 eV
   rms) and no mixture of the two beats 26.7 eV. The ISDF V_q[0] convention also
   agrees — it reproduces the exact 2D V_H to 0.5 % on gnppm. (VH_2D − VH_3D is
   huge, 141 eV rms, ½∫ρV_H = 437.8 vs 244.0 Ry — so the convention *matters*
   enormously; it is simply not mis-set.)
2. **ISDF quadrature — real but secondary, and now moot.** At μ=399/46 bands it
   is 2.3 eV rms (0.5 %); at μ=276/160 bands on the 12×12 it is 82.6 eV rms vs
   the exact V_H, confirming M's under-convergence finding. The fix removes this
   error class entirely and makes H0 centroid-count-independent.
3. **The dominant residual is `kin_ion`'s IONIC terms — a second, independent
   bug M's evidence #5 mistakenly exonerated** (that exoneration was circular:
   `VH_true = kih − kin_ion` was *assumed* to be QE's V_H). Same material, same
   cell (702.20 bohr³, c = 12 Å), same pseudos (md5-identical), k=0 band 0
   (Mo semicore), Ry→eV, from `term_probe.py`:

   | deck | ecutwfc | T | V_loc | V_NL | V_H | H0 | implied Vxc |
   |---|---|---|---|---|---|---|---|
   | cohsex fixture | 16 | 24.1 | −463.7 | **+37.6** | 358.8 | −43.4 | physical |
   | gnppm fixture | 30 | 29.0 | −682.2 | **+24.1** | 593.6 | −35.5 | **−29.9 ✓** |
   | mos2 12×12 | 80 | 24.0 | **−526.6** | **+0.09** | 602.4 | **+99.9** | **−166 ✗** |

   V_H agrees across systems to 1.5 % (593.6 vs 602.4) — the new route is
   consistent everywhere. The ionic terms are not: V_loc is +155.6 eV and V_NL
   −24.0 eV off relative to the physically-verified 30 Ry case, and **V_NL has
   collapsed to ≈0** at 80 Ry. Net +135 eV, which is the whole H0 error.
4. **Not aliasing / not grid resolution.** Doubling the FFT box (36,36,135) →
   (72,72,270), rebuilding V_loc, ρ and V_H on it, moves every term by
   **≤ 2×10⁻⁴ eV** on both decks. Also not the V_NL radial table (n_q=4000,
   q_max=√ecutwfc·1.01 ⇒ dq only 2.2× coarser at 80 Ry than at 16 Ry, and
   `_table_interp` clamps rather than zeroes). The ecut dependence points at the
   ionic-potential construction itself (species structure factors / SR–LR split
   in `build_local_ionic_potential_on_G_total` + `vnl_ops`) — **open, and the
   next workstream's target.** `wk_N/term_probe.py` is the ready-made tool.

### Gate 3 — fixture eqp (cohsex_debug, P=4, job 7874710)
Three cases, all rc=0. Against `eqp_ref.dat`: **`sigSX`, `sigCOH`, `sigTOT`,
`Eo` are bit-identical (max|Δ| = 0.000e+00) in all three** — Σ is untouched, as
designed. Legacy-file case also reproduces the reference `VH` column exactly ⇒
**zero regression in legacy mode**. New mode: VH column → 0 (contract), eqp0
col-4 moves −168.45 → −73.88 eV at (k=0,n=2), i.e. the documented delta is
*large* here and that is the correct outcome — `eqp_ref` was frozen on a stale
`kin_ion.h5` *and* an ISDF V_H built from a 4-band pseudo-density (see Gate 1).
The exact-V_H run is the only one of the three with a physical implied Vxc.

**Bottom line:** the V_H fix is implemented, validated to 0.5 % against an
independent method, and makes H0 centroid-independent — but it does **not** by
itself rescue the 12×12, because a second and larger bug lives in the ionic
part of `kin_ion` at high ecutwfc. Do not trust any 12×12 eqp until that is
fixed; the raise-the-centroid-count interim from M will not help either.

---

## O — SILENT-FAILURE HARDENING (branch `safety-hardening` @ a1969c3, wt-F)

Target: the campaign's dominant failure signature — **rc=0 but garbage**
(QP gap −136 eV with every stage "successful"; a NaN back-solve caught only by
counting floats in `eqp0.dat`; a restart silently misindexing bands). No
cluster jobs used; verification = `py_compile` over all of `src/` + a
19-assertion pure-python suite that runs on the login node.

### New infrastructure
- `src/common/sanity.py` — cheap stage-boundary gates in the
  `_warn_on_unphysical_h0` style. `check_finite` / `check_hermitian` /
  `check_positive` / `check_in_range` / `check_sign` / `check_shape` /
  `check_count`. One device reduction → **one** host fetch of ≤4 scalars per
  call; sharded-safe (reduces on device, never gathers). Switch:
  `LORRAX_SANITY=0|1|strict` (default 1 = warn loudly and keep running;
  `strict` raises `SanityError` — use in CI). Grep token on failure:
  `*** LORRAX SANITY FAILURE`.
- `src/common/collectives.py` — `barrier(name)` replacing **7** copies of
  `try: sync_global_devices(...) except Exception: pass`. Single-process ⇒
  skip; multi-process ⇒ **never swallow** (a swallowed barrier is how one rank
  sails past a collective its peers are blocked in → hang → rc 0).
- `runtime.install_failfast_excepthook()` (now part of `bootstrap()`) — in a
  multi-rank run an uncaught exception prints a rank-tagged banner, flushes,
  and `os._exit(1)` without unwinding, so srun sees a non-zero task and the
  job's rc stops lying. No-op at P=1; opt out with `LORRAX_FAILFAST=0`.

### Two latent bugs found and fixed (both silent by construction)
1. **The `zeta_q.h5` basis guard was dead code for every production run.**
   `gw_init._check_zeta_h5_matches_basis` probed `f['zeta_q']`, but the G-flat
   writer creates `zeta_q_G` — so `f.get(...)` returned `None` and the guard
   passed on exactly the stale-ζ collisions it was written to catch. Now probes
   both layouts, plus `zeta_is_done` and the centroid FFT grid.
2. **`zeta_is_done` was written by the writer and read by nobody.** A ζ from a
   fit that died mid-write (this happened repeatedly: SIGABRT at V_q entry,
   RESOURCE_EXHAUSTED in Stage C) was indistinguishable on disk from a complete
   one. `ZetaLoader` now refuses it (`LORRAX_ALLOW_PARTIAL_ZETA=1` to override).

### Gates added (stage → invariant)
V_q: finite + `tr V_{q=0} > 0` + hermiticity of the q=0 tile + finite G0 ·
restart load: finite V_q / ψ / E_nk + positive trace · screening: finite W per
role + hermiticity on the ω-imaginary branch · Σ: finite Σ_x/V_H + **Σ_x
diagonal strictly negative** (negative-definite by construction) + magnitude
bracket · QP: finite kin_ion + Σ_total before `eigh`, finite E_qp after ·
**eqp writers: NaN/Inf refusal + post-write re-parse asserting the exact finite-
float count** — the check that caught the NaN back-solve, now structural ·
`compute_eqp_diag`: **implied `Vxc = E_DFT − (kin_ion + V_H)` in the physical
window** — same invariant and same bounds as `gw_output._warn_on_unphysical_h0`
(single-sourced from it at import), extending that guard to the post-hoc
`make_eqp_bgw` CLI, which rebuilds eqp{0,1} from on-disk `kin_ion.h5` +
`sigma_mnk.h5` and had none · htransform:
finite S/ctilde/E_nk + bandwidth bracket + finite `bandstructure.dat` ·
`dipole.h5`: loud coverage report + warning when the file's `nbands` < the run's
Σ window (**the known 120-vs-160 head inconsistency now announces itself**) and
when its `nk` disagrees · BSE `_find_restart_file`: multiple
`isdf_tensors_*.h5` matches used to resolve **lexicographically** (1194 sorts
before 276) — now picks newest and names what it passed over.

### rc discipline
19 CLI mains had `main()` bare, discarding the return value (`htransform`,
`exciton_bands` already returned 0 into the void). All now
`raise SystemExit(main())`. `bse_jax` delegated to `bse_kpm.main` /
`bse_feast.main` and then hardcoded `SystemExit(0)` — now propagates.
`common/eigh_block_sweep.py` swallowed every per-block error and returned 0
unconditionally (a CI run could never fail) — now counts failures and returns 1.

### Exception audit
143 broad handlers (63 `except: pass`). Classified: the large majority are
benign (capability probes, cleanup `close()`, optional-telemetry, compile-cache
init). Fixed the error-hiding ones: the 7 swallowed barriers, the two
`ISDF_ZCT_STAGE_CAP_*` env parses (a typo silently disabled the cap the user
thought was set), `minimax` node-refinement failure (silently degraded the τ
quadrature every χ₀ is built on — now warns), `eigh_block_sweep`'s
`jax.distributed.initialize` swallow.

### Deferred to file owners (N owns these in wt-J — proposals only, no edits)
- `psp/get_DFT_mtxels.py:913,920` — broad `except` demotes "failed to write
  kin_ion.h5" to a warning and falls through to `return 0`. **A run producing no
  kin_ion.h5 exits 0.** Highest-severity rc leak found.
- `kin_ion.h5` coverage/provenance: reader (`file_io/kin_ion.py`) checks only
  `band_stop <= nb_total`; it never reads `attrs["nb"]/["nk"]/["sys_dim"]/
  ["truncation_2d"]`, and the two writers disagree on what they stamp
  (`gw/kin_ion_io.py` stamps 6 attrs, `psp/get_DFT_mtxels.py` stamps only a
  description). Proposal: make `get_DFT_mtxels` stamp the same 6 + the new
  `has_hartree`, then add `assert_kin_ion_matches(nk, sys_dim, truncation_2d,
  has_hartree)` with the missing-attr pass-through idiom.
- `gw/gw_output.py:282` — the 7th swallowed barrier (same one-line fix).

### Recalibration after gate job 7874812 (a false positive, and what it taught)
The first version of the eqp gate bracketed the **raw QP shift**
`Δ = eqp0 − E_DFT` at ±50 eV. It fired on **72 of 120 states of the
cohsex_debug fixture — in the same job where that run reproduced its reference
eqp to 1e-6 eV**. The reasoning was wrong, not the threshold: `Δ = Σ_xc − V_xc`,
and deep semicore bands carry bare-exchange `Σ_x` of order −100 eV against an
equally large `V_xc`, so `|Δ| ≫ 50 eV` is *correct physics* there. Replaced with
the implied-`Vxc` identity, which is insensitive to legitimately large Σ by
construction (Σ does not appear in it). Pinned by two regression tests: a
healthy run with `max|Δ| > 50 eV` must stay silent; a 60 eV `kin_ion + V_H`
cancellation error must fire.

**Placement, corrected after gate 7874822.** The recalibrated gate was first
put in `compute_eqp_diag`, which the live driver path also uses — where
`gw_output.write_results` already runs `_warn_on_unphysical_h0` on the same
arrays immediately before the writer. Result: one broken H0 reported itself
twice, in two wordings, on the identical 86 of 120 states. The gate now lives
on `make_eqp_bgw` only (the post-hoc CLI, genuinely unguarded); the live path
is left entirely to N's merged guard, which owns that policy. Pinned by
`test_compute_eqp_diag_does_not_duplicate_the_mean_field_gate`.

**The fixture's H0 is genuinely defective — the firing is a TRUE positive.**
Gate 7874822 shows N's own merged `_warn_on_unphysical_h0` firing on
cohsex_debug: implied Vxc in **[−144.976, +88.420] eV, 86 of 120 (k,n)
outside [−50, 2]**. This is consistent with N's Gate 3 finding (line ~1348)
that `eqp_ref` was frozen on a stale `kin_ion.h5` *and* an ISDF V_H built from
a 4-band pseudo-density. So "fixture eqp reproduces eqp_ref to 1e-6 eV" and
"the fixture's mean field is unphysical" are both true and not in tension —
`eqp_ref` is a frozen LORRAX reference, not ground truth. **Consequence for
gating: the criterion "no sanity firing on the fixture" cannot be met by any
correct implied-Vxc guard until the fixture's `kin_ion.h5` is regenerated.**
That is N's call, not a reason to widen the window.

Methodology note, including a mistake worth recording: while diagnosing the
false positive I reconstructed the implied Vxc on the login node from
`sigma_diag.dat` + `eqp0.dat` and got [−140, +97] eV. I dismissed that as
invalid because in COHSEX mode `sigma_diag.dat` prints `sigSX` (*screened*
exchange) while eqp{0,1} use `sig_x` (*bare*) — a real discrepancy, worth
tens of eV. But the measured truth is [−145, +88]: **the reconstruction was
essentially right**, offset by the ~5–9 eV screened-minus-bare difference, and
it had already predicted this outcome. I over-corrected and discarded a
usable signal. The reusable lesson cuts both ways: derive each gate's
invariant rather than pattern-matching it, but do not throw away a
cheap approximate measurement just because it is approximate.

### Seam gap caught in production: `make_eqp_bgw` vs N's `has_hartree` contract
Job 7874840 ran the post-hoc CLI in `sweep_c606` against Q's regenerated
V_H-folded `kin_ion.h5`. The implied-Vxc guard fired on **all 10080 entries**
([−626.96, −87.61] eV) and the rebuilt QP gap was **−453 eV**, with rc=0.

Root cause: N's no-double-count seam lives in `gw/sigma_dispatch.py`, which
covers the **live driver only**. `make_eqp_bgw` predates the contract and kept
adding `sigma_mnk.h5`'s ISDF Hartree column on top of an already-folded
`kin_ion`. The mixed case is the *normal* one for this CLI, not an edge case:
Σ_xc does not depend on V_H, so pointing it at a regenerated `kin_ion.h5` is
precisely how one re-derives QP energies without re-running Σ.

Fix (wt-F, `gw/eqp_bgw.py`): read `file_io.kin_ion.kin_ion_has_hartree` and,
when set, zero the CLI's Hartree term — the exact mirror of the live seam.
No formula change was needed in the guard: N's identity is mode-agnostic by
design, so `E_DFT − (kin_ion + V_H)` with V_H suppressed *is*
`E_DFT − kin_ion`. Legacy (attr-less) files take an untouched path.

**Gate 7874844 — all three green:**
| check | result |
|---|---|
| jax unit suite (2 new contract tests) | **19 passed, 0 failed** |
| legacy regression, attr-less kin_ion, pre-fix vs post-fix | **eqp0.dat + eqp1.dat BIT-IDENTICAL** |
| 606c repro (the 7874840 scenario) | guard **SILENT**; suppressed V_H mean 366.692 eV |

### μ-CONVERGENCE DATUM — MoS₂ 12×12, 606 centroids, exact-route V_H
**DFT gap 1.7010 eV → QP gap 2.6475 eV** (GW correction **+0.947 eV**).
Physical, and the first trustworthy 12×12 QP gap of the campaign: it is
centroid-count-independent by construction, since H₀ now takes V_H from the
exact FFT-grid route rather than the ISDF quadrature. Compare against the
276c/1194c/1998c rungs as they land — under the exact route the gap should be
*flat* in μ, which is the whole point of N's fold-in and the cleanest available
test of it.

### Status
Not measured on hardware (no cluster jobs by directive). `py_compile` clean over
all of `src/`; `tests/test_sanity_gates.py` **19/19 pass with plain python3 on
the login node**. `tests/test_sanity_gates_jax.py` written and READY-TO-RUN
(needs jax + h5py; 1 process, seconds) — covers the device/sharded reduction
path and the HDF5 provenance guards. Note: `pytest` cannot collect this repo's
`tests/` on the login node at all (`tests/harness.py:200` uses PEP-585
`dict[str, str]` in a runtime annotation; login python3 is 3.7) — use the
standalone `python3 tests/test_sanity_gates.py` runner.

## P — accumulated loose ends closed (wt-G, branch `loose-ends`, base a1969c3)

Development/unification polish, not research. 10 items; **9 done, 1 partial**.
59 files changed (+765 / −178) + 2 new files. **NOT committed** (per directive).
No cluster jobs used (queue owned by other workstreams).

### P.1 — BUG L-1 CLOSED: the facade's resolve-is-a-promise contract holds again
`ffi/linalg/resolve.py::_check_geometry` gained SLATE's `px>1 and py>1 and
px!=py` rejection, mirroring `ffi/slate/context.py::validate_tile_layout:123`
exactly (same rule, same order, same message shape). Before:
`resolve_backend('cholesky','slate', 2x4)` returned `'slate'` and the very next
call raised. Verified by **exhaustive equivalence** over all 16 shapes in
{1,2,4,8}²: resolve-time and call-time rules now agree on every one. Confirmed
the fix is safe for the production route — `isdf/core`'s `slate_cholesky` uses
the NON-batched `distributed_cholesky` (whole-mesh `p,q`), so the mesh-level
rule is the one that applies; the batched wrappers' `(1,Py)` sub-grid path
(which legitimately opts out via `allow_row_grid`) is untouched.

### P.2 — linalg_ffi.md corrected (the "100–600× slower" line was backwards)
The claim is now scoped to its actual regime (**batched eigh on GPU**) and
paired with L's measured CPU table: cholesky 400 **1.8×**, solve_lu 400×200
**6.1×**, solve_lu in the μ=3000 chain **11.5× FASTER** than native, with the
counter-example (1 node × 16 ranks: native wins) that explains WHY — the FFI
advantage is a multi-node effect. Added the cost model to design against
(**first call 1.4–2.7 s**, **per-call floor 0.10–0.26 s** ⇒ hoist
`resolve_backend` and the first call out of loops; don't route matrices whose
native solve is already < ~0.25 s). New sharp edges: the **square-or-N×1 mesh
constraint** (an 8×10 production mesh gets `native` for everything — pick 8×8
or 10×10 if FFI linalg matters), **bug L-2** (SLATE host `heev` SIGSEGVs at
1×1/n=64; everything ruled out except SLATE-2025.05-vs-MKL-LAPACK-3.8),
scalapack's **donate-both-A-and-B**, and **`FI_PROVIDER=mlx`, not `tcp`** —
every inter-node number L measured is therefore pessimistic.

### P.3 — MKL ScaLAPACK/BLACS link made permanent; SLATE patch documented
`src/ffi/common/cpp/host/CMakeLists.txt` now resolves ScaLAPACK itself, in the
SLATE branch (where `scalapack/cpp/solve_lu_ffi.cc` is compiled):
`-DLORRAX_SCALAPACK_LIBRARIES` (vendor-agnostic escape hatch) → MKL probe under
`-DLORRAX_MKL_ROOT`/`$MKLROOT` → loud WARNING. **Verified by a real cmake
configure on the login node**: the generated link line is byte-identical to the
one L hand-injected —
`-Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl`.
BLACS flavour and threading layer are cache vars (a BLACS/MPI mismatch links
fine then hangs inside `blacs_gridinit`). `-Wl,--no-as-needed` is commented as
mandatory. `project(... LANGUAGES CXX)` is now explicitly load-bearing (no
gfortran in the container). `config/frontera/build_ffi_host.sh` gained the
SLATE group as an opt-in (`LORRAX_SLATE_HOST_INSTALL_DIR` set ⇒ all 9 targets),
records SLATE's 3 external-source deviations as a documented patch step
(`LANGUAGES CXX Fortran → CXX`; `-Dblas=mkl` not libsci; build on node-local
/tmp because /work2 Lustre stalls under a 40-node job), and post-verifies
exported handler symbols + ScaLAPACK/BLACS in DT_NEEDED.

### P.4 — the argparse abbreviation trap, closed as a CLASS
`allow_abbrev=False` on **all 49** `ArgumentParser`s under `src/` (47 applied;
2 skipped — `gw/kin_ion_io.py`, `psp/get_DFT_mtxels.py` — workstream N owns
them; one-line each, listed for N). AST-located inserts, so multi-line call
formatting is preserved. This is the bug that silently ate kmeans_cli's `--out`
onto `--out-suffix` and mangled a centroid file. Gated both ways: with the flag
`--out` now falls through to `parse_known_args` extras (or errors); a control
parser without it still swallows it.

### P.5 — bootstrap stragglers converted
`psp/run_nscf.py`, `psp/run_sternheimer.py`, `centroid/kmeans_cli.py` now use
`runtime.bootstrap()`. **BEHAVIOUR CHANGE, called out in each file:** they
previously ran `set_default_env()` + `init_jax_distributed()` with NO CPU
fallback; `bootstrap()` adds `fallback_to_cpu_if_no_gpu_backend()`, so on a node
with no usable GPU backend these three now fall back to CPU instead of dying at
the first jax call. Verified `bootstrap()` still precedes each module's own
`import jax` (the before-import contract).

### P.6 — htransform now enables the JAX compile cache
It was the one driver that never called `ensure_jax_compile_cache` (H's
finding). Same call/pattern as `gw_jax._warm_start`, placed after `parse_args`,
failures logged and swallowed. The `ISDF_JAX_CACHE_DIR=""` opt-out is honoured
inside the helper (no test needed at the call site) and is documented there as
the mandatory P>1 setting on this machine.

### P.7 — AsyncWfnReader deleted (production-dead, measured overlap 0.000)
87 lines from `file_io/wfn_loader.py`; `__all__` trimmed to `["WfnLoader"]`.
Confirmed zero live references first (AST-level: no Name/Attribute/import/
`__all__` use anywhere in the repo). **`common/async_io.AsyncDispatcher` is
KEPT** — it drives the SlabIO write side. The two historical mentions
(`psi_G_store.py`, `async_io.py` docstrings) were rewritten to record *why* the
class is gone and how to rebuild it on AsyncDispatcher if CrI3-scale ever wants
it, rather than deleted.

### P.8 — ζ-RESTART: `tmp/zeta_q.h5` is now reused instead of always refit
Before, `gw_init` only VALIDATED the μ extent of an existing ζ and refit
regardless. Now the fit is skipped when the file is provably the same fit.

* `isdf_header` gained `fit_provenance` (JSON) + `stamp_fit_provenance()`,
  written **after** `mark_zeta_done` on purpose — a job killed between the two
  leaves a complete-but-unstamped file, which refits. Every failure mode
  (missing file, unreadable header, `zeta_is_done=False`, absent provenance,
  legacy file, changed centroid table) costs compute; **none costs
  correctness**. That asymmetry is the design: this cache sits in front of the
  step whose silent misuse produced −135 eV once already (job 7874375).
* Provenance = band windows, both cutoffs, `charge_zeta_solve`,
  `gamma_contract_mode`, `write_ibz_only`, bispinor, gspace_mode, band_norms
  digest, fft_grid, ecutwfc/ecutrho, source-WFN path+size, schema version.
  **Deliberately excludes** `n_rmu_padded` / process count / mesh / chunk plan —
  ζ is device-count invariant, so a P=4 ζ must stay reusable at P=80.
  **`zeta_rcond`/`zeta_ridge` record the EFFECTIVE (env-overridden) values**,
  mirroring `isdf/core:1278-1279` — recording the cfg values would have been a
  real hole (drop `LORRAX_ZETA_RCOND` on a rerun and it would have matched).
* Loud banner on reuse; on mismatch the log NAMES the changed keys
  (`on-disk=… now=…`). `LORRAX_FORCE_REFIT=1` overrides. Reuse is
  charge-channel-only — a bispinor run also produces `transverse_wfn_data`,
  which is not on disk, so it always refits.
* gw_init.py was clear to edit: checked N's wt-J diff first (7 files, gw_init
  not among them).

### P.9 — fixture protection (the 2026-07-25 symlink incident)
Root cause found: `wkJ_gate.sbatch` staged the fixture with `ln -sf`, so the
driver's `sigma_mnk.h5` write landed **through the link, on the checked-in
fixture**. Three layers now:
1. `tests/harness.protect_fixtures()` chmods every **git-tracked** file under
   `tests/regression/` to `a-w`; `tests/conftest.pytest_sessionstart` calls it.
   The git-tracked predicate matters: `sigma_mnk.h5` is in `_FIXTURE_IGNORE`
   (the driver writes a file of that name) **and** is a checked-in artifact —
   a name-based rule would have skipped precisely the victim.
2. `harness.copy_fixture` restores owner-write on the run-dir COPY, so Tier-2
   variants that mutate their input file still work.
3. `wkJ_gate.sbatch` fixed: `ln -sf` → `cp -rL … ; chmod -R u+w`; the fixture
   README documents the copy-never-symlink convention. (Backup:
   `wkJ_gate.sbatch.bak-preP`.)
**APPLIED** `a-w` to the fixtures in **wt-G and the main checkout** (git sees no
mode change — only the x-bit is tracked). **wt-J left alone** (N is active
there); one line to protect it too, in the report.

### P.10 — env-var registry: `docs/dev/env_vars.md` + `tools/env_audit.py`
Full AST audit of every `os.environ`/`os.getenv` read under `src/` (52 names)
plus the 12 C++ `getenv()`s, categorised runtime-physics / runtime-IO / debug /
deprecated / build-only / external, each with default + read site.
**Consistency verdict: NO drift.** Every LORRAX var read at >1 site agrees
(`LORRAX_FORCE_FULL_BZ` ×5 all `'0'`; `LORRAX_PHDF5_STRIPE_COUNT` ×3 all `'16'`;
`_STRIPE_SIZE_FS` ×3 all `'4M'`; `LORRAX_MEM_DEBUG` ×4, `LORRAX_RCHUNK_DEBUG`
×2 all presence-tests). The scanner's "multiple defaults" flags on
`JAX_PLATFORMS` / `CUDA_VISIBLE_DEVICES` / `XLA_PYTHON_CLIENT_*` /
`TF_GPU_ALLOCATOR` are all `setdefault`(writer)+`get`(reader) pairs — the
intended pattern, not drift.
**Promotion candidates, ranked** (#1 is workstream M's recommendation):
1. **`LORRAX_ZETA_RCOND`** — the key `zeta_rcond` already exists, but
   `isdf/core:1279` reads the ENV with the key as *fallback*, so the env wins
   **silently and unlogged**. At μ=1998 this number decides `n_keep ≈ 1015/1998`
   with no spectral gap — it sets the numerical rank of the pair-density space.
   A sweep whose rcond came from an env var is a run whose central parameter is
   not in its own input file. Fix = the existing `_sc_env` idiom
   (`gw_config.py:1272`): key is truth, env overrides **loudly**.
2. `LORRAX_ZETA_RIDGE` (same site, same shape, do it together).
3. `LORRAX_ZETA_REPLICATE_CAP_GIB` (gates which ζ route a run takes).
4. `ISDF_ZCT_STAGE_CAP_GB/_FRAC`, `ISDF_CHUNK_TARGET_UTILIZATION` → memory keys.
5. `LORRAX_WFN_BACKEND` → backend config section.
Explicitly **do not** promote debug/build vars or `ISDF_JAX_CACHE_DIR`.
Bonus: `LORRAX_GFLAT_CHUNK_SIZE` in `docs/theory/isdf-zeta-vq.md:479` is stale —
nothing reads it; the knob is the input key `gflat_chunk_size`.

### P — verification
**Login-node (RUN, all green):** `wk_P/run_verify_login.sh` →
**72 passed / 0 failed / 0 skipped**, plus `py_compile` **56/56** over every
touched file. Covers: the L-1 guard's full 16-shape truth table (executed from
the shipping source via AST extraction — no jax needed), all 49 parsers +
a live argparse demonstration of the `--out`/`--out-suffix` bug and its fix,
`isdf_header` provenance round-trip against real h5py, the ζ-reuse wiring +
provenance-content invariants, fixture protection + idempotence + the victim
file, AsyncWfnReader liveness, bootstrap ordering, and the htransform cache
call site. (`module load python3/3.9.2 phdf5/1.10.4` — the runner does it.)
A cmake configure smoke test additionally proved the ScaLAPACK link line.

**Cluster gate — WRITTEN, READY TO RUN, NOT RUN (no jobs by directive):**
`wk_P/gate.sbatch` (+ `wk_P/g4_guard.py`), 4 nodes × 2 ranks, ~30 min:
G1 eqp regression at P=4/P=8 (behaviour-neutrality of the argparse/bootstrap/
dead-code/harness changes) · G2 ζ-reuse end-to-end (run twice ⇒ "FIT SKIPPED"
+ byte-identical eqp; then changed band window ⇒ must REFIT and name the key;
then `LORRAX_FORCE_REFIT=1` ⇒ must refit; plus a provenance-stamp dump) ·
G3 htransform bit-exact vs the meshless ground truth + compile cache populates ·
G4 the L-1 guard against the REAL unified host lib on 2×4 — the promise must
either resolve-and-succeed or refuse at resolve time, never both · G5 the three
converted CLIs on a CPU node (the fallback behaviour change).

### P — deferred / notes for the orchestrator
- **PARTIAL (item 4):** `src/gw/kin_ion_io.py:76` and
  `src/psp/get_DFT_mtxels.py:727` still lack `allow_abbrev=False` — workstream N
  owns both. One-line each: add `allow_abbrev=False, ` as the first argument to
  the `argparse.ArgumentParser(` call.
- `wt-J`'s fixture is still writable. To match:
  `cd /work2/08271/jackmc/frontera/wt-J && git ls-files -z tests/regression |
   xargs -0 chmod a-w` (left undone because N is working there).
- The ζ-reuse feature changes what a rerun-in-place DOES. Nothing reuses without
  a byte-identical provenance stamp, and only files written by this branch carry
  one — so every pre-existing `zeta_q.h5` refits exactly as before. The first
  behaviour change any user sees is a *second* run in the same directory.

## Q — the IONIC-TERM bug is a WFN-LOADER bug: `nosym` WFNs were silently time-reversed (wt-J, 2026-07-25)

**Root cause (one line): for any WFN with `ntran <= 1` (a `nosym` deck — which
is exactly the 12×12), `SymMaps` left `sym_mats_k` at length 1 instead of
TRS-augmenting it to `2·ntran`, so `unfold_psi` computed
`n_sym_spatial = 1 // 2 = 0`, classified the IDENTITY row (`sym_idx = 0`) as a
time-reversal row, and returned `iσ_y·conj(ψ)` on an un-negated G-list — i.e. it
replaced ψ(r) by ψ*(−r) at EVERY k of EVERY nosym WFN read through the `eager`
backend.**

N's attribution ("V_loc +155.6 eV off, V_NL collapsed 24.1 → 0.09") was the
correct *symptom* but the wrong *organ*: nothing in `psp/` is broken. The
projector code, the radial tables, the KB coefficients and the local-potential
builder were all fed a wavefunction that had been inverted through the origin.
The owner's prior ("V_NL has been correct every time I've tried it for months")
was right — every deck previously tried had `ntran ≥ 2`.

### Why it hid for so long — the invariance pattern is a perfect camouflage
ψ → iσ_y·ψ*(−r) is norm-, orthogonality-, and |ψ_G|-preserving, so:

| quantity | invariant under the corruption? | measured |
|---|---|---|
| ‖ψ‖, ⟨ψ_m\|ψ_n⟩ | exactly | 1.000000, max\|S−I\| = 4.5e-15 |
| ⟨T⟩ (depends only on \|ψ_G\|² vs \|k+G\|) | exactly | 24.0447 eV both ways |
| ⟨V_H⟩ (ρ is built from the SAME inverted ψ ⇒ V_H inverts too) | exactly | 602.402 eV, agreed to 1.5 % with the good deck |
| ⟨V_loc⟩ (ions at the TRUE τ; MoS₂ has **no** inversion centre) | **NO** | −526.594 → −704.964 eV (Δ = −178) |
| ⟨V_NL⟩ (KB projectors at the TRUE τ) | **NO** | 0.086 → 36.590 eV (426×) |

That is why every internal consistency check N ran (∫ρ = nelec, FFT-box
doubling, radial-table q_max sweeps, Coulomb-convention probes) passed: they are
all invariant under the very transformation that was corrupting the answer.

### The discriminating measurements (`/scratch2/08271/jackmc/lorrax_setup/wk_Q/`)
1. `vnl_bisect.py` (job 7874808) — **kills every cutoff/table hypothesis.**
   On the 12×12's own ψ, V_NL is already collapsed at a 9 Ry G-sphere
   (0.058 eV) — the loss is not at high G. Radial tables built with
   q_max=√30 vs √80 agree to **1.1e-6 relative** for all 16 projectors and both
   V_loc SR tables. `q > q_max count = 0`. Hypotheses 1–3 of the brief: dead.
2. `unfold_probe.py` (job 7874811) — **the smoking gun.** At the 12×12's k=0 the
   symmetry op is the *identity* and `gvecs(full_bz)[0]` is bit-equal to the raw
   file G-list, yet `loader.load()` returns something with
   `max|ψ_prod − ψ_raw| = 0.285`. V_NL from the RAW h5 coefficients = **36.59 eV**;
   from the loader = **0.086 eV**. Control: gnppm (ntran=2) has
   `ψ_prod ≡ ψ_raw` at k=0 (Δ = 0.0) and its k=1 (a genuine TRS row) reproduces
   the raw-path V_NL exactly (24.1993 both ways) — the unfolder is right when
   it is *asked* to unfold.
3. `phase_probe.py` (job 7874817) — **identifies the transformation exactly.**
   |ψ_prod/ψ_raw| = 0.98764 on spinor 0 and 1.01242 = 1/0.98764 on spinor 1,
   constant across G ⇒ the spinor components were swapped-and-conjugated:
   `ψ_prod = iσ_y·conj(ψ_raw)`. `translations[0] = 0`, `U_spinor[0] = I`, so the
   only way to get that is the mis-derived `n_sym_spatial`.
4. `cross_overlap.py` (job 7874813) — the two decks' Γ states have identical
   eigenvalues (−4.862 vs −4.805 Ry, whole ladder matches) but overlap only
   0.07–0.62. Both are internally orthonormal, so one of them is a unitary
   image of the truth — the 12×12 one.

### THE decisive, circularity-free measurement — same-deck term identity vs `kih.dat`
`same_deck_identity.py`, 12×12 k=0, 30 bands, LORRAX's own T/V_loc/V_NL/V_H vs
pw2bgw `out/kih.dat` (`kih + vxc = el` verified to 1e-4 eV, so it is exact truth):

| | T | V_loc | V_NL | V_H | H0 | kih_QE | resid | implied V_NL |
|---|---|---|---|---|---|---|---|---|
| band 0, **before** | 24.045 | −526.594 | 0.086 | 602.402 | +99.939 | −41.926 | **+141.87** | −141.78 |
| band 0, **after**  | 24.045 | **−704.964** | **36.590** | 602.402 | **−41.927** | −41.926 | **−0.001** | +36.591 |

Over 30 bands at k=0: **rms(H0 − kih) 73.7953 eV → 0.0002 eV**;
`rms(V_NL_LORRAX − V_NL_implied_by_QE)` = 0.0002 eV. Every term is now
independently correct — the identity closes to 0.2 meV.

### The fix — 2 hunks, `src/common/symmetry_maps.py`
- `SymMaps.__init__`, `ntran <= 1` branch (~:879): TRS-augment
  `sym_mats_k = concat([S, −S])` exactly as the general branch does at ~:972.
  One line of behaviour; the rest is the comment explaining the camouflage.
- `unfold_psi` (~:808): hard guard —
  `len(sym_mats_k) != 2*len(U_spinor_spatial)` now raises instead of silently
  reinterpreting the identity row as time reversal.
- `tests/test_symmetry_unfold.py`: new
  `test_trivial_symmetry_branch_trs_augments_sym_mats_k` (builds a `ntran=1`
  SymMaps, asserts the augmented shape and that the identity row round-trips ψ).

### Blast radius — which artifacts are invalid
The corruption lived in `WfnLoader._eager_build` only. The `phdf5` / `phdf5_host`
table path derives `n_tran = sym.sym_matrices.shape[0]` (= 1, correct) and was
always right — so this was also a silent **backend-parity violation** that the
existing parity test never saw because every fixture has `ntran ≥ 2`.

- **Corrupt** — anything a *single-process* tool produced from a `nosym` WFN.
  For the 12×12 that is `kin_ion.h5` (written by the 1-node `kinion.sbatch`),
  and therefore every H0, `eqp0/1.dat`, `eqp_g0w0.dat`, `sigma_diag.dat` and
  `sigma_mnk.h5` downstream of it. **Regenerate.**
- **Probably clean** — `dipole.h5` (4-rank `dipole4.sbatch` ⇒ phdf5* path) and
  the multi-rank GW ψ reads themselves. Cheap to regenerate; recommended for
  provenance hygiene now that the two backends provably agree.
- **Unaffected** — every fixture (`cohsex_debug` ntran=12, `gnppm_debug` 2,
  `bispinor_debug` 2, `si_cohsex_debug` 48) and every symmetry-reduced deck.

### Gates
- **(a) fixture no-op — PASSED, bit-exact.** Regenerating both fixture
  `kin_ion.h5` files with the fix against N's pre-fix files:
  `cohsex (9,40,40) max|Δ| = 0.000000e+00`, `gnppm (9,80,80) max|Δ| = 0.000000e+00`.
  The 30 Ry deck cannot change — it has symmetry, so the branch is untouched.
- **(b) the same-deck term identity at k=0 (30 bands) — PASSED:
  rms(H0 − kih) 73.7953 eV → 0.0002 eV.**
- **(c) THE 12×12 GATE — PASSED. Regenerated `kin_ion.h5` (`-n 120`, exact V_H),
  full 144 k × 120 bands (17 280 states) vs `out/kih.dat`:
  `rms(H0 − kih) = 0.0001 eV`, `max|Δ| = 0.0006 eV`.**
  Implied `Vxc` now lands on QE's `vxc.dat` to the same 0.0001 eV
  (range [−24.262, −3.498] vs QE's [−24.263, −3.498]).
  Per-k rms is flat at 1.5–2.0e-4 eV across all 144 k; per-band rms is
  ≤ 1e-3 eV across all 120 bands. This is the campaign's final physics gate:
  **38.69 eV → 0.0001 eV, a 4·10⁵× reduction.**

| H0 route (144 k × 120 bands) | rms(H0 − kih) |
|---|---|
| production on-disk `kin_ion.h5` (legacy, no V_H) | 275.80 eV |
| N's exact-V_H fix alone (corrupted ψ) | 38.69 eV |
| **exact V_H + this loader fix** | **0.0001 eV** |

(The 38.6940 eV control reproduces N's reported 38.69 exactly, which validates
the gate script against N's independent `analyze_h0.py`.)

**Units gotcha, for whoever writes the next gate:** `kin_ion.h5` stores
**Rydberg**; `kih.dat` / `vxc.dat` are **eV**. A first pass at the gate compared
them raw and produced a plausible-looking 22.86 eV that was nothing but the
missing 13.6057 factor. `wk_Q/gate_rms.py` now carries the conversion.

### Artifacts
`/scratch2/08271/jackmc/lorrax_setup/wk_Q/`: `vnl_bisect.py`, `unfold_probe.py`,
`phase_probe.py`, `cross_overlap.py`, `same_deck_identity.py` (the reusable
per-band term-identity tool — point it at any deck with a `kih.dat`; it is the
assumption-free way to attribute an H0 error to a specific term),
`localize.py`, `gate_rms.py`, `cmp_h5.py`; jobs `bisect.7874808 /
ident.7874810 / unfold.7874811 / xover.7874813 / phase.7874817 /
gates.7874823 / loc.7874827 / gate2.7874829`; regenerated
`kin_ion_vh120_FIXED.h5` (ready to install as the 12×12's `kin_ion.h5`).

`tests/test_symmetry_unfold.py`: 17 passed (incl. the new guard), job 7874827.

### P.11 — G4 RE-RUN: **GREEN** (job 7874834, rc=0) + a second real bug found

The coordinator's G4 finding was real, and root-causing it turned up a defect
worth more than the original item.

**It was NOT a stale symbol table.** `_HOST_TARGET_SYMBOLS` maps
`lorrax_slate_potrf → SlatePotrfHostFfi`, and `nm -D` confirms the unified lib
exports all 9. Given a correct table and present symbols, `has_target` can only
return False if **`get_lib()` raised** — which it did.

**Root cause (two layers).**
1. *Environment (mine).* The gate's container bind list omitted **`/opt/intel`**,
   so MKL and the Intel compiler runtime did not exist inside the container at
   all — no `LD_LIBRARY_PATH` could fix that. `dlopen` failed on
   `libmkl_scalapack_lp64.so`. Secondary: the unified lib's RUNPATH covers MKL /
   SLATE `lib64` / IMPI `lib/release`, but its DT_NEEDED also needs
   `libhdf5.so.310`, `libfabric.so.1` and the Intel compiler RT
   (`libimf`/`libsvml`/`libirng`/`libintlc`). L's proven recipe also needs
   `srun --mpi=pmi2` + `I_MPI_PMI_LIBRARY` + `I_MPI_FABRICS`. The gate now
   mirrors `wk_L/bench2.sbatch` exactly.
2. **CODE DEFECT (the one that matters, now fixed).** `has_target` wrapped
   `get_lib()` in a bare `except Exception: return False`, so **"the library
   would not load" and "the handler is not compiled" collapsed into one bool** —
   and `resolve_backend` rendered it as *"its FFI handler is not compiled into
   the cpu FFI library … To enable it, rebuild with build_host.sh"*. That
   diagnosis is actively wrong: it sends you to rebuild a library whose symbols
   are all present, while the actual cause (a container bind / one
   `LD_LIBRARY_PATH` entry) goes unmentioned. This is precisely the
   silent-wrong-behaviour class the campaign is stamping out — the *effect* was
   an N×1 production mesh silently downgraded to `native` with a misleading
   reason.

**Fix:** new `ffi_loader.probe_target(target, platform) -> (usable, reason)`
separating the three states that have three different fixes — *unknown target*
(typo/wrong platform), *library would not load* (path/deps/glibc — **says
nothing about whether the handler is compiled**, and names `LD_LIBRARY_PATH` /
`ldd` as the fix), *loaded but does not export the symbol* (**the only
"rebuild" case**, and the only one that now quotes `build_hint`). `has_target`
is retained as `probe_target(...)[0]` for auto-pick fallback logic (unchanged
semantics for `_auto_pick_backend` / `has_phdf5_read`). `resolve.py` guard 3
quotes the reason; its duplicate `_BUILD_HINTS` table (a second, drifting copy
of the build command) was deleted — that string now lives only in
`ffi_loader._PLATFORMS[...]["build_hint"]`.

**Test defect also fixed.** G4 v1 asserted only THAT a shape was refused, not
WHY. Since guard 3 (capability) precedes guard 5 (geometry), a library-load
failure made every rectangular case "pass" for entirely the wrong reason — only
8×1 (which must SUCCEED) exposed it. Each expected refusal now asserts the guard
that produced it, and a preflight aborts loudly if the lib will not load.

**G4 result (job 7874834, 1 node × 8 ranks, unified host lib):**

| check | result |
|---|---|
| all 9 host targets `probe_target` → available | PASS |
| `cholesky+slate` 2×4 refused **by the geometry guard** | PASS |
| `cholesky+slate` 4×2 refused **by the geometry guard** | PASS |
| `cholesky+slate` 1×8 refused **by the stride guard** | PASS |
| `cholesky+slate` **8×1 resolves to slate** | PASS |
| 8×1 **promised call SUCCEEDS** (real SLATE potrf) | PASS |
| 8×1 numerics `‖LLᴴ−A‖/‖A‖` | **2.21e-16** |
| `list_backends` explains, never raises | PASS |

`=== G4: ALL PASS ===`, `rc=0`. The intermediate run (7874833) is itself
evidence the new diagnostic works: it printed
*"the cpu FFI library could not be loaded: OSError: libmkl_scalapack_lp64.so …
NOTE: this says nothing about whether lorrax_slate_potrf is compiled"* —
exactly the message the old code could not produce.

**Login suite re-run: 92 passed / 0 failed / 0 skipped; py_compile 57/57.**
(+20 checks pinning the three-way probe distinction, that a load failure never
claims "not compiled", that the build hint appears only on the partial-build
path, and that every `_HOST_TARGET_SYMBOLS` entry matches the lib's real
nm-verified exports.)

**Files touched by this round:** `src/ffi/common/ffi_loader.py` (probe_target,
`_LIB_PATHS`/`_loaded_path`, has_target delegates), `src/ffi/linalg/resolve.py`
(guard 3 + docstring, `_BUILD_HINTS` removed), `docs/dev/linalg_ffi.md`
(guard-3 section + the RUNPATH/LD_LIBRARY_PATH facts).

---

## R — MoS2 12×12 GW quasiparticle bandstructure (htransform), 2026-07-26

> **⚠ CLAIM-DECAY (AL, 2026-07-27): em1-scoped collective walls.** Every multi-node JAX CPU-collective WALL TIME in this section was measured while Gloo was bound to Frontera's 1 GbE management NIC (`em1`, 129.114.x.x) — the campaign-wide default before `runtime.pin_gloo_interface()` landed (AK.10/AL). Byte counts, HLO collective inventories and residency ledgers remain valid; the wall-time consequences of collective-bound stages re-price on ib0 (measured 3.3x whole-pipeline at 785c/P=16, and see AL for 606c/P=80). Single-node numbers are unaffected.

**Deliverable.** DFT-vs-QP overlaid band plot on Γ–M–K–Γ from the 606-centroid
GN-PPM run, job **7875020** (development, 8 nodes × 2 ranks = 16 devices,
mesh 4×4). Workdir `/scratch2/08271/jackmc/lorrax_setup/wk_R/band/`.

| artifact | path |
|---|---|
| main figure (broken y-axis: manifold + semicore) | `wk_R/band/bandstructure_mos2_gw.png` |
| gap zoom (VBM−3 … CBM+3 eV) | `wk_R/band/bandstructure_mos2_gw_zoom.png` |
| raw bands | `wk_R/band/bandstructure_{dft,qp}.dat` (Ry, VBM-shifted to 0) |

**Gaps.** DFT on-path **1.7008 eV** vs the known 12×12 DFT gap 1.7010 eV
(**0.2 meV** — the interpolation self-check). QP on-path **2.6513 eV**
(+0.9505 eV vs DFT); full-BZ eqp0 reference 2.6475 eV. The 3.8 meV difference is
geometry, not error: the full-BZ QP CBM sits at (2/3, 1/6, 0), which the
Γ–M–K–Γ path misses; on-path the QP CBM is at the Λ/Q valley between K and Γ,
0.225 eV *below* the K→K direct QP gap of 2.8765 eV. GW therefore moves the CBM
off K — the DFT gap is direct at K, the QP gap is indirect.

**Timing (8 nodes, 16 ranks, nk=144, nb=70, n_μ=606, n_r=174960).**
DFT pass 1064 s, QP pass 973 s (Galerkin 889 s each, ψ window load 84 s,
band_chunk auto-sized to 7 = 5.26 GB/chunk, 10 chunks). Peak node0 ~35 GB of
192. Whole two-pass + plot job: 34 min wall inside the 2 h dev limit.

**Two traps worth writing down.**
1. `htransform --eqp-file` **cannot read LORRAX's own `eqp0.dat`.**
   `read_eqp_energies` (htransform.py:683) wants the BerkeleyGW *log* form
   (`k-point N:` / `n=… EQP=…`); LORRAX's GW writes the *columnar* BGW form, and
   `initialize_wfns` **swallows the parse failure with a log line**, silently
   interpolating DFT energies while the run reports success. Same trap is
   already documented at `bse/exciton_bands.py:470`. Worked around here with
   `wk_R/band/make_eqp_ht.py` (columnar → log form). This is a real code gap —
   htransform should either read the columnar form or hard-fail.
2. `read_eqp_energies` does **no unit conversion**: its return value replaces
   `enk_sigma`, which is in **Rydberg**, while BGW eqp files are in eV. Feeding
   the eV numbers straight in is a silent 13.6× physics error. The converter
   divides by `common.units.RYD_TO_EV`.
   Gate that catches both: the log must say
   `Using EQP energies from … for band window (0, 70)`.

**The `RANK-DEFICIENT` warning is far more pessimistic than reality.** With
n_μ=606 the Galerkin basis has rank 1212 = nspinor·n_μ against 144×70 = 10080
states, and htransform prints *"Capacity rule: nk·nb < rank(ψ_μ), i.e. nb < 8.42
here"*. Measured on-grid interpolation error at the 13 path points that ARE
coarse-grid k-points: **max 10.2 meV / rms 1.9 meV over all 70 bands, 2.2 meV
over the frontier bands** (QP: 10.7 / 2.2 / 2.3 meV). fH(k=0) eigenvalue error
9–10 meV; Γ Δε ≤ 0.34 mRy; FFT Γ round-trip 4.4e-15. The printed capacity rule
should not be read as a usability gate at this scale.

**Interpolation artifacts (what to look at in the figure).** Step/kink between
adjacent path points, by band block:

| band block | DFT step / kink | QP step / kink |
|---|---|---|
| valence 0–25 | 0.28 / 0.27 eV | 0.35 / 0.25 eV |
| low conduction 26–39 | 0.41 / 0.22 eV | 0.44 / 0.26 eV |
| mid 40–59 | 0.52 / 0.60 eV | 0.70 / 0.81 eV |
| **window top 60–69** | **0.64 / 0.87 eV** | **0.92 / 1.68 eV** |

Artifacts grow monotonically toward the top of the Σ window and are essentially
absent in the frontier bands — the expected hard-truncation edge effect of the
70-band window, consistent with the `exciton_bands.py` note that a ~640-centroid
set cannot orthonormalise the high oscillatory bands. **Nothing above ~14 eV of
the plotted zoom is trustworthy at the few-hundred-meV level; the gap region
is.** The residual kinks inside the zoom window (≤0.27 eV, both at path pt 74 in
valence bands 18/19) are at genuine band crossings and are an artifact of
`energies_sorted` (energies are sorted per k, so a crossing shows as a kink in
the sorted-index curve) — not of the interpolation.

**Note on inputs.** `sweep_c606/eqp0.dat` (rebuilt by job 7874847, QP gap
+2.6475 eV) is the sane one and pins the band window to nval=26/ncond=44.
`run1/eqp0.dat` (window 80, 276 centroids) is the **broken pre-fix** file — QP
gap **−0.4925 eV** — and must not be used.

## overnight A/B (2026-07-26) — SUBMITTED, 72 nodes x 2 ranks = 144 devices (12x12 mesh)

> **⚠ CLAIM-DECAY (AL, 2026-07-27): em1-scoped collective walls.** Every multi-node JAX CPU-collective WALL TIME in this section was measured while Gloo was bound to Frontera's 1 GbE management NIC (`em1`, 129.114.x.x) — the campaign-wide default before `runtime.pin_gloo_interface()` landed (AK.10/AL). Byte counts, HLO collective inventories and residency ledgers remain valid; the wall-time consequences of collective-bound stages re-price on ib0 (measured 3.3x whole-pipeline at 785c/P=16, and see AL for 606c/P=80). Single-node numbers are unaffected.
Owner-approved overnight submission. Codebase 1ee52b2 (Stage-C all-to-all fix, sharded
slab reads, exact-route V_H, nosym unfold). Re-preflighted from scratch with the UPDATED
planner on a REAL 12x12 mesh of 144 forced host devices (`preflight_144.py`, job 7875067).

| job id | dir | n_mu (pad) | nband (b4) | r_chunk | model HWM | binder | wall |
|---|---|---|---|---|---|---|---|
| **7875070** | `run_A_c2406_b400/` | 2406 (2448, 1.7%) | 400 (432, 7.4%) | 11664 x15 | 28.58 GB | F_tensor_write | 12 h |
| **7875071** | `run_B_c1998_rcond10/` | 1998 (2016, 0.9%) | 160 (288, 44%) | 11664 x15 | 19.30 GB | F_tensor_write | 6 h |
Sizing: `n_rtot = 174960 = 2^4*3^7*5`, `n_rtot/144 = 1215`, so legal `r_chunk = 144*d`
with `d | 1215` -> chunk counts 1/3/5/9/15/27/45/81/135/243/405/1215 (far more choice
than the 8x10 mesh's 1/3/9/27/81). Both cap-raised (`LORRAX_ZETA_REPLICATE_CAP_GIB=16`;
needs 14 and 10 respectively). B also exports `LORRAX_ZETA_RCOND=1e-10`.

**THE OBSERVABLE ALREADY HAS A RESULT (from last night's 40-node pair).** The n_keep
telemetry now exists (`isdf/core.py:1328` `jax.debug.print`) — this closes the
observability gap flagged in the scale-ladder entry. At nband=160, rcond=1e-8:
- `n_log=1998  n_keep/q = 1013..1018`
- `n_log=2406  n_keep/q = 1050..1056`
**+20% centroids bought +3.6% kept rank.** The ceiling (~1050) is set by the BAND
WINDOW, not by mu — which is what job A tests by raising nband 160 -> 400 at mu=2406.
And the cut has **no spectral gap**: `lam_max ~ 2.054e-2`, `lam_min_kept ~ 2.052e-10`,
ratio == rcond EXACTLY. The truncation is set by the knob, not the physics — which is
what job B tests by dropping rcond two decades.

**NEW DEFECT — a per-r-chunk MEMORY RAMP (this is the current #1 blocker).**
Job 7874803 (P=80, mu1998, r_chunk=2160, 81 chunks) climbed **+2.9 GB/node/chunk** from
a 32 GB base — 32 -> 69 -> 93 -> 111 -> 121 GB — and died `std::bad_alloc` at r-chunk
~32/81 after 2.2 h. MaxRSS **63.9 GB/rank**, node peak **137/192 GB**, while the planner
predicted HWM **18.91 GB** (binder F_tensor_write). Ratio 3.4x, and *growing with chunk
index* — i.e. NOT a static under-count, an accumulation. `buff/cache` stayed flat at
9 GB throughout, so it is anonymous memory, not page cache.
- Linear extrapolation: 81 chunks would need ~264 GB/node. **That configuration cannot
  finish at any wall time.**
- MITIGATION APPLIED TONIGHT (not a fix): **15 chunks instead of 81**. Under a
  fixed-per-chunk ramp that is ~24 GB/node of accumulation instead of ~130. The updated
  planner also lets 15 chunks keep `F_tensor_write` as the binder with a modest
  `C_fit_one_rchunk` stage (14.16 / 10.57 GB), so nothing is traded away.
- The two candidate models (leak per chunk FIXED vs proportional to r_chunk) are not yet
  distinguishable from one data point. **Tonight's runs are the discriminator**: same
  code, 15 chunks, P=144 — if the ramp per chunk is unchanged at ~2.9 GB/node the leak is
  fixed-per-chunk; if it falls ~1.8x it tracks the per-rank working set.
- **Why this still gets submitted:** the n_keep observable is printed at the L_q stage
  (log line ~1810) BEFORE `Started zeta fitting` (line ~2121). Both jobs therefore
  deliver their primary science in the first ~1 h even if the fit loop later dies.

**P-independent walls at these sizes** (do NOT shrink with node count):
`F_tensor_write = 2*nq*mu^2*16` UNSHARDED = 27.6 GB (A) / 18.7 GB (B);
writer gather `nq*n_mu*ngkmax*16` = 47.7 / 39.6 GB on ONE rank;
replicated rank-truncate eigh ~76 / ~42 min REDUNDANT on all 144 ranks.

**Band-pad note:** at 144 ranks `round_up(160,144) = 288`, so job B pays a 44% band-axis
pad. It does NOT contaminate the A/B: pad bands are zero-filled
(`meta.py` b_id_4 contract), so pair densities, the CCT spectrum and n_keep are
numerically identical to a b4=160 run — only FLOPs are wasted. Job A's `b4=432 > mnband=400`
is handled by the loader's past-file clamp.

**Accepted inconsistency:** `dipole.h5` (the q=0 Coulomb head) was built on a 120-band
transition space, narrower than either run's window. Documented in both input headers.
`kin_ion_vh120_FIXED.h5` (144,120,120) is used by both; 120 >= window b3=80.

**Telemetry to grep tomorrow** (in priority order):
1. `grep -aoE "n_log=[0-9]+ rcond=[^ ]+ n_keep/q=\[[0-9 ]+" gw.*.out | head` — THE result.
   A: does n_keep leave ~1050 and head towards ~2600? B: does it leave ~1015?
2. `grep -aF "lam_min_kept" gw.*.out | head -1` — is `lam_min_kept/lam_max` still pinned
   exactly at rcond (no gap), or did a real spectral edge appear?
3. `grep -aF "Computing L_q" gw.*.out | sort -u` — MUST stay `path=replicated_rank_truncate`.
4. `awk '{for(i=1;i<=NF;i++) if($i~/used_GB=/){split($i,a,"=");print a[2]}}' mem_node0.*.log`
   — the ramp slope per chunk; this is the leak discriminator.
5. `grep -aE "r-chunk +[0-9]+ / 15|bad_alloc|RESOURCE_EXHAUSTED|Finished zeta"` — did the
   tail survive.

---

## S — V_H becomes an explicitly-resolved SOURCE (`stored` | `isdf` | `gspace`), and the measured price of each (wt-G, branch `vh-default`, base 1ee52b2)

Two directives, in order. (1) Restore the ISDF `V_q[0]` route as the default and
show it reproduces the exact V_H to O(1 %) at reasonable μ. (2) *Superseding the
storage design:* `kin_ion.h5` should by default carry the exact ⟨mk|V_H|nk⟩ as a
**separate array**, `kin_ion` staying pristine, with resolution order
stored → isdf → gspace and an input key to override.

Both are implemented and gated. The convergence measurement is the reason the
second directive is the right one: the ≤1 % target on V_H is **reachable but not
sufficient**, so the file format has to let a run choose its V_H source per-run
rather than baking one in at generation time.

### 1. MECHANISM VERDICT — the "ρ misses occupied bands when nval < nelec" hypothesis is REFUTED

The coordinator's addendum (and N's Gate-1 note at line ~1290) held that
`cohsex_sigma.hartree` builds ρ from the σ-window valence bands, so a deck with
`nval < nelec` (fit start `b1 = nelec − nval > 0`) drops the excluded occupied
bands out of ρ — a structural error no μ could fix. **It does not.**

- **Code.** `build_Gij` multiplies `wfns.yr(s.sigma)`, and `s` is
  `wavefunction_bundle.BandSlices`, whose `sigma = slice(0, b3 − b0)` — i.e.
  global `[0, nelec+ncond)`, *starting at b0*. `nocc = min(nelec, nb_sigma)
  = nelec`, so the projector marks global bands `[0, nelec)`: **every** occupied
  band, semicore included, for any `nval`.
- **The trap that produced the wrong reading.** `Meta.band_ranges.sigma` is a
  *second, conflicting* definition, `(b1, b3)`. Under that reading the fixture's
  window is bands 22..29 and "22 occupied bands are missing" follows. But
  `Meta.band_ranges` is **dead code — `grep -rn band_ranges src/ tests/` finds
  only its own definition**. Fixed by comment, not deletion (`common/meta.py`,
  `gw/cohsex_sigma.build_Gij`).
- **Measured, on the exact deck the hypothesis was proposed for.**
  `wk_S/vh_mu_probe.py` prints the resolved coverage. cohsex fixture
  (`nval=4`, `nelec=26`, `b0..b4 = (0, 22, 26, 30, 40)`):
  `sigma slice = global [0, 30) ; Gij marks the first 26 occupied => global
  [0, 26)` and **`tr Gij per k = 26.000`**. Not 8, not 4.
- **The positive half of the proof — μ moves it, band window frozen.**
  If the deficit were missing charge, ISDF/exact would sit at ≈4/26 = 0.15 for
  *every* μ. Measured ratio over the 26 occupied bands, identical deck and
  identical band window at all four μ:

  | μ (cohsex fixture) | 60 | 200 | 398 | 785 |
  |---|---|---|---|---|
  | mean ISDF/exact over occ | **0.651** | 1.015 | 0.972 | — |
  | rms error on occ V_H | **38.80 %** | 4.85 % | 3.09 % | **1.69 %** |

  A 23× reduction from centroids alone. **The fixture's unphysical mean field
  (implied Vxc [−144.98, +88.42] eV, O line ~1443) is ISDF under-convergence at
  1.5 centroids/band, not a ρ-coverage defect.** N's "0.5 % agreement of two
  incomplete densities" reading does not apply: both densities were complete.

### 2. THE IMPLEMENTATION — V_H as a resolved source, not a file mode (10 files, worktree only, NOT committed)

**File format.** `gw.kin_ion_io` now writes, by default, TWO datasets:
`kin_ion` = ⟨T+V_loc+V_NL⟩ **pristine**, and `v_hartree` = the exact FFT-grid
⟨mk|V_H|nk⟩ (Ry) — the **full matrix**, so a QSGW rotation has something it can
transform (~15 MB at 12×12/80 bands). `--no-hartree` writes the ionic-only file;
`--fold-hartree` reproduces N's legacy add-into-the-values format for artifact
regeneration only. The exact-V_H build is factored into one
`compute_hartree_matrix(wfn, sym, meta, …)` shared by the CLI and the driver's
`gspace` route, so `stored` and `gspace` cannot drift apart.

**Resolution.** New input key `hartree_source = auto|stored|isdf|gspace`,
validated at *parse* time (`gw_config`), resolved once by
`file_io.kin_ion.resolve_hartree_source`: `auto` → stored array if present, else
a legacy folded file, else isdf. Explicit requests are honoured **except** on a
folded file, where any other source would double count ~500 eV — that raises and
names the fix rather than producing a plausible wrong number.

**The seam migrated, not duplicated.** N's no-double-count line in
`gw/sigma_dispatch.py` becomes the source seam at the same single point where
`sig_h` enters `SigmaResult`: `folded` → zero it; `stored`/`gspace` → **replace**
it with the exact matrix; `isdf` → keep it. Everything downstream (the eigh
operand, the fixed-point h₀, the SC map, eqp{0,1}, `sigma_diag.dat`) is therefore
consistent by construction, and the VH column stops reading 0.000 by design.
`gw/eqp_bgw.py`'s CLI mirror follows the same three-way rule (suppress for
folded, **substitute** for stored). `gw_output`'s guard is source-aware: the
identity was already mode-agnostic, only the printed diagnosis differs.

**QSGW correctness catch, found while wiring it.** In the SC loop
`compute_sigma_xc` runs against **rotated** ψ, so every Σ channel it returns is
in the QP basis and `sc_iteration` rotates the lot back with `O_DFT = U·O_QP·U†`.
The stored V_H is a fixed **DFT-basis** operator; substituting it raw would have
made the rotate-back return `U·V_H·U†` — a basis error inside a 500 eV term with
no other symptom. New `hartree_basis_rotation=U_qp` kwarg rotates it in
(`O_QP = U†·O_DFT·U`) first. One-shot is unaffected (default `None`).

**Two structural guards added.**
- `cohsex_sigma.build_Gij` **raises** if `nb_sigma < nelec`, the only way the
  Hartree ρ could ever drop occupied bands (see §1). Unreachable today for
  `ncond ≥ 0`; the guard is there so a future band-window change cannot
  reintroduce it quietly.
- `kin_ion_io` **refuses a multiprocess launch**. The CLI is single-process by
  construction (`load_kpoint_fftbox` passes `sharding=None` and builds a 1×1
  mesh; the k loop is host-side Python; the output is a numpy array), so
  `srun -n P` meant P ranks redoing the whole job and overwriting each other's
  `kin_ion.h5` at rc=0. `LORRAX_KIN_ION_ALLOW_MULTIPROC=1` overrides.

**Back-compat is safe in the dangerous direction.** A new-format file has
pristine `kin_ion` and `has_hartree=False`, so an *old* reader treats it as
ionic-only and correctly adds its own ISDF V_H. Only the reverse (a folded file
read as pristine) double counts, and `_warn_on_unphysical_h0` fires hard on that.

Files: `gw/kin_ion_io.py`, `file_io/kin_ion.py` (+`__init__` exports),
`gw/gw_config.py`, `gw/gw_jax.py`, `gw/sigma_dispatch.py`, `gw/sc_iteration.py`,
`gw/gw_output.py`, `gw/eqp_bgw.py`, `gw/cohsex_sigma.py`, `common/meta.py`,
`tests/test_sanity_gates_jax.py` (7 new tests).

### 3. GATES — all green (jobs 7875316, 7875318, **7875549**)

| # | gate | result |
|---|---|---|
| — | `py_compile` all 224 files under `src/` + tests | **OK** |
| — | `tests/test_sanity_gates.py` (login-runnable, plain python3) | **23 passed / 0 failed** |
| G1 | `tests/test_sanity_gates_jax.py` (P=1, in container) | **25 passed / 0 failed** (7 new) |
| G9 | same suite **re-run against the final source** (job 7875555, after the attr-precedence and basis-rotation edits) | **25 passed / 0 failed** |
| G2 | new-default `kin_ion.h5` on 3 decks: pristine `kin_ion` + `v_hartree`, `has_hartree=False` | **OK** |
| G3 | **decomposition identity** `kin_ion + v_hartree` vs N/Q's folded file | **max\|Δ\| = 0.000e+00 on all 3 decks** |
| G4 | `--fold-hartree` vs `wk_Q/gnppm_kin_ion_vh_Q.h5` | **BIT-IDENTICAL** |
| G5 | fixture eqp P=4, `auto` → **stored**; implied Vxc | **[−6.273, −1.539] eV, guard SILENT** |
| G6 | **THE MIGRATION GATE** — same deck, legacy **folded** file | eqp0 **and** eqp1 vs G5: **max\|Δ\| = 0.00e+00, 496 values** |
| G7 | legacy ionic-only file, `auto` → **isdf**, vs `eqp_ref.dat` | **PASS — 1888 values, max\|Δ\| = 1.0e-6 eV** |
| G8 | explicit `hartree_source=isdf` **on a stored file** | override honoured (`resolved=isdf`); Σ columns still 1e-6 vs `eqp_ref` |

**G6 is the gate the design change turns on, and it is exact.** The stored-array
route and the fold-in route it replaces produce **bit-identical QP energies**.
The one difference anywhere in the outputs is `sigma_diag.dat`'s VH column:

```
folded:  n=0  sigSX= -7.458231  sigCOH= -4.706926  sigTOT= -12.165157  VH=   0.000000  Eo= -59.162438
stored:  n=0  sigSX= -7.458231  sigCOH= -4.706926  sigTOT= -12.165157  VH= 591.647161  Eo= -59.162438
```

— i.e. the column that used to read 0.000 *by design* now reports the actual
mean-field Hartree (591.647 eV, matching the generator's own printout to
1e-4 eV). That is the intended improvement, not a regression. (Method note: the
gate as first scripted compared `eqp_test.dat`, which for this deck is the
`sigma_diag` dump, and "failed" at max|Δ| = 592 eV on exactly 270 of 1620 values
= one column of 270 rows. Comparing the actual `eqp0_test.dat` / `eqp1_test.dat`
is what answers the question the gate was asking.)

The three routes are therefore proven equivalent where they must be and distinct
where they should be: `stored ≡ folded` bit-exactly on QP energies; `isdf`
reproduces the frozen fixture reference to 1e-6 eV; an explicit `isdf` override
on a stored file demonstrably takes the ISDF branch (implied Vxc moves from
[−6.3, −1.5] to a different range, and the printed resolution says `isdf`).
The implied-Vxc guard fires on the ISDF fixture runs ([−144.976, +88.420] eV) —
a **true positive** for the μ=60 under-convergence quantified in §1, exactly as
O predicted, and it is silent on both exact routes.

### 4. GROUND TRUTH — assumption-free exact V_H

`V_H_exact(k,n) = RYD · [ diag(kin_ion --hartree) − diag(kin_ion default) ]`,
both files from the same code, same deck, same ψ: every ionic term cancels
identically. Q's gate already pins the `--hartree` file to `out/kih.dat` at
**rms 0.0001 eV / max 0.0006 eV over 144 k × 120 bands**, so this reference is
QE-verified *for every band*, not just the occupied ones.
Cross-check against N's `wk_N/vh_conv_12x12.npz` (built pre-loader-fix, on
time-reversed ψ): **rms 6.2e-14 eV** — confirming Q's invariance argument
(ρ(r) → ρ(−r) ⇒ ⟨n|V_H|n⟩ unchanged) holds to roundoff for all 120 bands.

### 5. **THE CONVERGENCE TABLE** — ISDF V_H vs exact, MoS₂

Errors are per-(k,n) diagonals; "%" is rms(Δ)/rms(V_H_exact) over the group.
`dGAP` = ΔV_H(CBM, k_c) − ΔV_H(VBM, k_v), i.e. what the ISDF V_H would do to
the **DFT gap** if H₀ took V_H from this route. "c/band" = μ / deck `nband`
(the ζ-fit window b4). Tool: `wk_S/vh_mu_probe.py` + `vh_curve.py`.

**(a) Production scale — MoS₂ 12×12, 80 Ry, nk=144, nband=160, 120-band reference.**
μ=606 is the real production `sweep_c606` run's `sigma_mnk.h5` Hartree column
(40 nodes × 2 ranks, correct ψ). μ=276 (`run1`) is **unusable as a data point**:
that job ran against Q's folded `kin_ion.h5`, so `sigma_dispatch` zeroed `sig_h`
and the column on disk is identically 0 — the seam worked.

| group (bands) | rms eV | max eV | rms % | rms meV |
|---|---|---|---|---|
| semicore, n=0–7 | 4.013 | 6.248 | 0.680 | 4013 |
| n=0–11 (semicore + S 3s) | 3.614 | 6.248 | 0.683 | 3614 |
| valence n=12–25 | 0.838 | 3.305 | 0.206 | 838 |
| **all occupied, n=0–25** | **2.531** | 6.248 | **0.541** | 2531 |
| **frontier ±5, n=21–30** | **1.624** | 3.633 | **0.370** | 1624 |
| n=0–35 (up to E_VBM+7.7 eV) | 2.555 | 7.797 | 0.555 | 2555 |
| **whole σ window n=0–69** | **32.556** | **209.0** | **8.647** | — |

Per-band: n=0 (Mo 4s) **0.005 %**, n=2–7 (Mo 4p) 0.74–0.83 %, n=8–11 0.64–0.76 %,
n=12–25 0.12–0.38 %, n=26–30 0.39–0.48 %. **First band above 1 % is n=34
(E_VBM+2.3 eV)**; from there it runs away — 7.7 % at n=40, 46 % at n=60, 65 % at
n=66. **The error is NOT semicore-dominated; it is high-conduction-dominated.**
The deepest state is the best-served of all 70 (density-weighted k-means puts its
centroids exactly where |ψ_4s|² lives); the worst are the diffuse states above
the CBM that the charge-density weight never samples.

**(b) The μ ladder — gnppm fixture (MoS₂ 3×3, 30 Ry, nk=9, nband=46, nval=26=nelec).**

| μ | c/band | occ rms % | occ eV | semi rms % | frontier rms % | frontier meV | ALL rms % | n≤1 % up to | **dGAP eV** |
|---|---|---|---|---|---|---|---|---|---|
| 150 | 3.26 | 2.792 | 13.03 | 2.310 | 2.450 | 10863 | 5.546 | n=1 | **−31.27** |
| 296 | 6.43 | 1.000 | 4.67 | 1.175 | 1.097 | 4862 | 2.080 | n=1 | **−8.84** |
| 399 | 8.67 | **0.113** | 0.53 | 0.072 | 0.153 | 679 | 0.427 | n=39 | **−0.64** |
| 586 | 12.74 | 0.197 | 0.92 | 0.192 | 0.253 | 1122 | 0.518 | n=35 | **+1.59** |
| 1007 | 21.89 | 0.166 | 0.77 | 0.170 | 0.170 | 754 | 0.399 | n=39 | **+0.33** |

**(c) The same ladder with `nval < nelec` — cohsex fixture (MoS₂, 16 Ry, nband=40).**
Band window frozen; only μ varies. This is the mechanism experiment of §1.

| μ | c/band | occ rms % | occ eV | semi rms % | frontier rms % | ALL rms % | dGAP eV |
|---|---|---|---|---|---|---|---|
| 60 (shipped) | 1.50 | **38.799** | 177.4 | 43.185 | 25.432 | 37.246 | +33.56 |
| 200 | 5.00 | 4.853 | 22.2 | 6.217 | 1.617 | 4.572 | −4.03 |
| 398 | 9.95 | 3.085 | 14.1 | 3.495 | 1.529 | 2.896 | +5.98 |
| 785 | 19.62 | 1.688 | 7.7 | 1.869 | 0.723 | 1.587 | +2.44 |

### 6. **THE VERDICT**

**(i) The ≤1 % target is met — at ≈6–9 centroids per ζ-fit band.**
gnppm crosses 1 % on the occupied manifold at μ≈296 (6.4 c/band) and is at
0.11 % by μ=399 (8.7 c/band) — the owner's 0.5 % datum reproduced (my
whole-σ-window figure at μ=399 is 0.427 %). The 12×12 at μ=606 is already at
**0.54 % on all occupied bands and 0.37 % (1.6 eV) on the frontier ±5**, i.e.
the ≤1 % bar is cleared at 3.8 c/band for every band that matters. Semicore is
**better** than the owner expected, not worse (0.68 %, and 0.005 % on the
deepest band); the exception the target should have carved out is the
**high-conduction tail** (>1 % above E_VBM+2.3 eV at μ=606, catastrophic above
+10 eV).

**(ii) …and ≤1 % is the wrong criterion. This is the finding that matters.**
V_H is ≈500 eV, so **1 % = 5 eV of H₀**, and the VBM and CBM errors do not
cancel. At the 12×12's production μ=606, at the K point where both band edges
sit (k=104): ΔV_H(VBM, n=25) = **+1.91 eV**, ΔV_H(CBM, n=26) = **−2.25 eV** ⇒
the DFT gap would read **1.701 − 4.15 = −2.45 eV**. Coherent across the
degenerate pairs (n=24,25: +2.06/+1.91; n=26,27: −2.25/−2.23), so it is
systematic error, not sampling noise. The frontier column in every table above
is in **eV, not meV**: the smallest frontier error anywhere in this campaign is
**679 meV** (gnppm, μ=399, 8.7 c/band), and the smallest gap error is
**330 meV** at 21.9 c/band. For 50 meV QP energies V_H must be right to
≈1×10⁻⁴ relative — **100× tighter than the directive's 1 %**.

**(iii) Raising μ does not get there: the error PLATEAUS and is non-monotonic.**
gnppm, occupied-manifold rms: 2.79 % → 1.00 % → **0.113 % → 0.197 % → 0.166 %**.
Past ≈9 c/band, 2.5× more centroids buys nothing; `dGAP` wanders
−0.64 → +1.59 → +0.33 eV. This is a floor, not a tail.

**And the floor is NOT the ζ rank truncation** — which answers the mandate's
question about run_B (`zeta_rcond=1e-10`). Job 7875349, gnppm at μ fixed = 1007,
`zeta_rcond` swept over four orders of magnitude:

| `zeta_rcond` | occ rms % | occ eV | frontier meV | ALL rms % | dGAP eV |
|---|---|---|---|---|---|
| 1e-6 | 0.179 | 0.833 | 602 | 0.650 | −1.059 |
| **1e-8 (default)** | 0.166 | 0.774 | 754 | 0.399 | +0.333 |
| 1e-10 | 0.162 | 0.754 | 749 | 0.396 | +0.392 |
| 1e-12 | **0.162** | 0.754 | 749 | 0.396 | +0.392 |

1e-10 and 1e-12 are **bit-identical** ⇒ rank truncation is already inactive at
1e-10, and going from the default 1e-8 to no truncation at all buys **2 %
relative** (0.166 → 0.162 %). The ~0.16 % floor is the intrinsic ζ-representability
limit of the centroid set, not a numerical truncation artifact. **Keeping more
modes does not help V_H specifically.**

**(iv) The doc guidance the new `hartree_source` key needs — "at what μ does
`isdf` match `stored` to ≤1 %".** Directly from the tables: **≈6 centroids per
ζ-fit band** for the occupied manifold and the frontier ±5 (gnppm crosses 1 % at
μ=296/6.4 c/band; the 12×12 is at 0.54 %/0.37 % with only 3.8 c/band), and
**≈9 c/band** to reach the 0.1–0.2 % plateau. It never matches to better than
that. For the σ window *as a whole* — including conduction states more than
~2 eV above the CBM — `isdf` does not reach 1 % at any μ measured: the 12×12 at
μ=606 is 8.6 % over bands 0–69 and 65 % at n=66.

**Consequence for the directive as written.** `isdf` is the default whenever the
file offers nothing better, and it *is* accurate to ≤1 % at sane centroid counts
for the bands that matter, as required. But **H₀ assembled from it does not yield
trustworthy QP energies at any μ measured here** — the accuracy H₀ needs is 1e-4,
not 1e-2, and the route plateaus two orders of magnitude short. That is precisely
why the second directive (store the exact V_H in the file, resolve the source
per-run) is the right structure: it makes the accurate route the *automatic* one
for any freshly generated `kin_ion.h5` — `auto` picks `stored` — while leaving
`isdf` one input key away for the QSGW / in-loop case that needs a distributed
V_H, and leaving every legacy file's behaviour exactly as it was. The remaining
work is §7: distributing the density build so `gspace` is affordable in-loop.

### 7. STRONG-SCALING AUDIT of the exact route — honest answer: **it does not scale at all today**

*(design note only; nothing implemented — mandated)*

**Where the work is** (`gw/kin_ion_io.py` + `psp/get_DFT_mtxels.py`), 12×12 with
N_r = 36·36·135 = 174 960, nk = 144, nb = 120, nspinor = 2, ngkmax = 8603,
n_occ = 26; one 3-D FFT = 5·N·log₂N = 1.52e7 flop:

| phase | flops | peak memory | distributed? |
|---|---|---|---|
| ρ build (`build_valence_density_chunked`) | 7 488 FFTs = **1.1e11** | ψ_k box 26·2·N_r·16 B = **146 MB**; ρ 1.4 MB | **no** |
| Poisson (`build_hartree_potential`) | 2 FFTs = **3.1e7** | 1.4 MB | **no** (and doesn't matter) |
| ⟨mk\|V_H\|nk⟩ (in the k loop) | 69 120 FFTs **1.05e12** + projection GEMM **1.4e11** | ψ_k box 120·2·N_r·16 B = **672 MB** | **no** |
| output | — | `kin_ion_all` (nk,nb,nb) c128 = **33 MB** host numpy | **no** |

**Measured** (job 7875316, 1 node, 1 rank): 1.05 s/k without V_H, 1.79 s/k with
⇒ 151 s → 258 s for 144 k. The V_H fold-in costs **+71 %** on `kin_ion`
generation — trivial in absolute terms (≈12 GFLOP/s effective, one socket).

**Why P buys nothing.** `load_kpoint_fftbox` calls `loader.load(..., sharding=None)`
and constructs a **1×1 mesh inline**; the k loop, the ρ accumulation and the
matrix-element assembly are host-side Python `for` loops; `kin_ion_all` is numpy;
and the CLI never calls `runtime.bootstrap()`. Strong-scaling efficiency is
**1/P**, and the memory HWM is **P-invariant** (per node it is P× *worse* if you
naively `srun -n P` — which is what the new guard in §2 now refuses).
`psp.get_DFT_mtxels.get_kin_ion` (the library variant) is only cosmetically
better: it takes a `mesh_xy` and stores ψ k-sharded, but `compute_valence_density`
then loops `for ik` accumulating into a **replicated** ρ (each iteration gathers
one k off the sharded axis) and the H_k loop does `wfn_k_sharded[i,:nb]` +
`np.asarray(total_k)` per k. Only the resident ψ shrinks; none of the work does.

**Where it breaks.** Scaling to a 24×24 deck (nk=576, nb=400, N_r ≈ 7e5):
ψ_k box **8.96 GB on one device**, `kin_ion_all` **1.47 GB**, flops **6.3e13**
⇒ ≈90 min serial, on a node you cannot add nodes to.

**What it would take to make it in-loop QSGW-capable** (all machinery exists):
1. `runtime.bootstrap()` + the 2-D mesh in `kin_ion_io.main`, as the other 9 CLIs do.
2. **ρ: near-perfect strong scaling, cheaply.** Replace the per-k
   `load_kpoint_fftbox` with the band-sharded G-flat load
   (`load_psi_gflat_padded`, `P(None,('x','y'),None,None)`) + `to_rchunk` /
   `to_rchunk_inner` per (k-chunk, band-chunk); accumulate |ψ|² into a per-rank ρ
   over the **full** r grid (real f64, 1.4 MB — replicating it is free) and finish
   with **one `psum`**. Bands *and* k shard, the FFT flops divide by P exactly,
   and the only collective is a 1.4 MB all-reduce per iteration. The existing
   `band_pad_to` zero-pad contract already guarantees pad bands contribute 0, so
   correctness reduces to "each occupied band owned by exactly one rank".
3. **Poisson: keep it replicated.** It is 3e-5 of the total flops and 1.4 MB;
   a sharded FFT here buys nothing and costs an all-to-all. Revisit above
   N_r ≈ 1e8.
4. **⟨mk|V_H|nk⟩: shard k over ('x','y')**, run `compute_local_V_k` rank-locally
   (no cross-rank reduction — each k block is independent, exactly the shape
   `htransform`'s G-accum already uses), and write `kin_ion` through `SlabIO`
   per-rank hyperslabs instead of a rank-0 numpy array.
5. **The caveat that actually decides it.** The QSGW loop carries ψ **only at the
   μ centroids** (the `Wavefunctions` bundle) plus the rotation U. An exact
   FFT-grid ρ per SC iteration means re-expanding the rotated ψ back to G-flat
   and re-running nk·n_occ 3-D FFTs **every iteration**; the ISDF route gets its
   ρ from arrays already resident, with one einsum over the centroid axis and no
   FFT at all. Even perfectly distributed, the exact route pays a full FFT sweep
   per iteration that the default route does not — ≈1.2e12 flop/iteration for the
   12×12, i.e. **≈0.5 s on 80 ranks**. That is affordable. The blocker is
   engineering (steps 1–4), not cost.

### 8. Cheap accuracy upgrades for the default route — ideas, one paragraph

Two ζ interpolations sit inside the kernel, not one: ρ(r) ≈ Σ_y ζ_y(r)ρ(y) *and*
ψ*_mψ_n(r) ≈ Σ_x ζ_x(r)ψ*_m(x)ψ_n(x); `V_q[0] = ζ†vζ` carries both. The
density-side one can be removed outright — build ρ exactly on the FFT grid and
Poisson-solve it (per §7 that is the *cheap, perfectly-scalable* part, 1.1e11
flop and a 1.4 MB all-reduce), then replace `Vrho = V_q[0]·ρ/nk` by the exact
V_H sampled at the centroids times the ζ dual weights `w_x = ∫ζ_x d³r` (computable
once from the ζ file). That is a contained change to one `jnp.einsum` and it is
the *only* upgrade that also survives QSGW, since ρ is what the SC loop updates.
It must be **measured, not assumed**: the two interpolation errors partially
cancel today, so a half-exact hybrid can be worse — the harness for that
measurement is exactly `wk_S/vh_mu_probe.py` + `vh_curve.py`. A semicore-only
additive V_H correction (the addendum's suggestion) is the *wrong* lever on this
evidence: semicore ρ is reproduced to 0.005 % on the deepest band, and what
degrades bands 2–11 is the **bra-ket** side (|ψ_4p|² is nodal and sharply
peaked), which a potential-side correction cannot touch; the correct form there
is a band-selective **exact matrix element** for a small fixed band set, which is
static across QSGW iterations for frozen semicore. Finally, the dominant error
(high conduction) is a centroid-*placement* problem: k-means weights by ρ^0.6,
which is exactly where diffuse conduction states are not — `--weight-bands` /
`--rho-power` / `--prune-window vc_x_vc` are the targeted knobs and are already
CLI-exposed.

### Artifacts
`/scratch2/08271/jackmc/lorrax_setup/wk_S/`: `vh_mu_probe.py` (isolates
`cohsex_sigma.hartree` on the real production ISDF path; prints the resolved ρ
band coverage and `tr Gij` — the tool that settled §1), `vh_table.py`,
`vh_curve.py`, `g8_check.py` (the decomposition identity);
`kin_ion_noVH120.h5`, `kin_ion_vh120_S.h5`, `kin_ion_STORED120.h5` (the
**new-format 12×12 file, ready to install**), the fixture triples,
`mu_out/vh_{gn,cx,rc,m12}_*.npz`, `logs/`; sbatch `s1_kinion` `s2_gate`
`s4_m12probe` `s5_fixture` `s6_musweep` `s7_rcond` **`s8_gate`** `s9_recheck`;
jobs **7875316** (kin_ion regen), **7875318** (pre-redesign unit + eqp gates),
**7875332** (μ ladders), **7875349** (rcond probe), **7875549** (the redesign
gate suite G1–G8), 7875322 (12×12 μ=276 probe, 1-node, still running at
hand-off), 7875323.

### Not finished at hand-off
- **12×12 μ=276 probe** (job 7875322, 1 node × 4 ranks) — would give the second
  production-scale point on the μ curve. It ran 50 min without reaching the
  Hartree kernel; the ζ fit at `r_chunk=2160` on 4 devices is simply slow. Not
  load-bearing: the μ dependence is established by the two fixture ladders and
  the 12×12 anchor at μ=606.
- **`hartree_source=gspace` has no cluster gate.** The code path is wired and
  shares `compute_hartree_matrix` with the CLI (so `stored` and `gspace` cannot
  disagree numerically by construction), and G3 proves that function's output
  against N/Q's artifacts bit-exactly — but no run has exercised the driver's
  on-the-fly branch end to end. Cheap to add: the cohsex fixture with
  `hartree_source = gspace` must reproduce G5's eqp0 exactly.
- **QSGW basis rotation of the stored V_H is untested on hardware.** The bug is
  real and the fix is derived (§2), but the fixture is one-shot, so no gate
  covers `hartree_basis_rotation`. A `qp_solver = self_consistent` fixture run
  with a stored file is the missing check.

**Reusable gotcha for the next agent:** `#SBATCH -n 4` puts `SLURM_NTASKS=4` in
the environment and `runtime.bootstrap()` reads it as the JAX process count, so
**any single-process tool launched with plain `apptainer exec` (no `srun`) inside
a multi-task allocation blocks ~5 min in `jax.distributed.initialize` and dies
`DEADLINE_EXCEEDED: RegisterTask`**. Cost me one wasted job (7875320) and half of
another. Export `JAX_PROCESS_COUNT=1 JAX_PROCESS_INDEX=0`, or wrap in `srun -n1`.
Also: **never pipe an sbatch step through `grep` alone** — the filter swallows the
traceback and a silent failure looks like a slow success; tee to a log first.

## T — the per-r-chunk ANONYMOUS-MEMORY RAMP: root-caused + cured; plus the ζ back-solve per-q gather tier (wt-J, branch `leak-hunt`, base 1ee52b2 — NOT committed)

> **⚠ CLAIM-DECAY (AL, 2026-07-27): em1-scoped collective walls.** Every multi-node JAX CPU-collective WALL TIME in this section was measured while Gloo was bound to Frontera's 1 GbE management NIC (`em1`, 129.114.x.x) — the campaign-wide default before `runtime.pin_gloo_interface()` landed (AK.10/AL). Byte counts, HLO collective inventories and residency ledgers remain valid; the wall-time consequences of collective-bound stages re-price on ib0 (measured 3.3x whole-pipeline at 785c/P=16, and see AL for 606c/P=80). Single-node numbers are unaffected.

### T.0 — discriminator readout from the three deaths (and the two survivors)

Least-squares fit of `mem_node0.*.log` `used_GB` sampled at each `LoopProgress`
r-chunk milestone, so the in-chunk sawtooth is averaged out rather than tracked.

| job | μ (pad) | nband (pad) | r_chunk (chunks) | P | **GB/node/chunk** | outcome |
|---|---|---|---|---|---|---|
| 7874386 `run1` 276c | 276 (288) | 160 | 19440 (9) | 80 | **+0.57** (≈noise) | completed |
| 7874609 `sweep_c606` | 606 (640) | 160 | 19440 (9) | 80 | **+4.08** | completed |
| 7874803 `sweep_c1998` | 1998 (2016) | 160 | 2160 (81) | 80 | **+3.40** | `bad_alloc` ~chunk 30/81 |
| 7875071 `run_B` | 1998 (2016) | 160 (288) | 11664 (15) | 144 | **+13.10** | OOM **in chunk 5/15** |
| 7875070 `run_A` | 2406 (2448) | 400 (432) | 11664 (15) | 144 | (1 milestone) | died **in chunk 2/15** |

**Neither candidate model in the overnight entry is right.** Not fixed-per-chunk
(2.9 → 13.1 at the same μ), and not proportional to the per-rank working set.
Fitting `leak ∝ nq·μ^a·(r_chunk/P)^b` over the four measured points gives
**b = 1.0, a = 2.4** — i.e. the ramp tracks the **back-solve FLOP count**
`nq·μ²·(r_chunk/P)`, at **0.10–0.16 GB per 10⁹ complex MACs**, flat across a 16×
span in that product and across two mesh shapes.  Memory ∝ *work done* rather
than ∝ any buffer is an allocator signature, and that is what it is.

Also settled: **the ramp is present at 276c too** (0.57 GB/node/chunk), just 23×
smaller — invisible inside the sawtooth over only 9 chunks.  And **run_A/run_B
did not die of the ramp alone** — see T.3.

### T.1 — the measurement that named it (jobs 7875321 / 7875330 / 7875331, 40 nodes each)

30 s node-level `free` is far too coarse.  New instrumentation (wt-J): per-r-chunk
**host RSS** from `/proc/self/status` (`device.memory_stats()` returns `None` on
the CPU backend, so this is the only faithful per-rank observable), the
`jax.live_arrays()` total, and per-phase deltas (`d_zq`/`d_solve` inside
`fit_one_rchunk`, `d_fit`/`d_acc` around it).  Repro: MoS2 12×12, **606
centroids, P=80, mesh 8×10, r_chunk forced to 2160 (81 chunks)**,
`LORRAX_MAX_RCHUNKS=30`, `LORRAX_EXIT_AFTER_ZETA=1` — same geometry as
7874609/7874803, ~32 s/chunk, ~25 min/run.

| condition | **rss slope GB/rank/chunk** | rss chunk 1→30 | fit wall/chunk | `live_arrays` |
|---|---|---|---|---|
| baseline (1ee52b2) | **+0.176** | 2.46 → 7.71 GB | 32.4 s (zq 9.7 + solve 22.7) | **16.483 GB, EXACTLY constant** |
| + `_reshard_z` hoist only | **+0.171** (unchanged) | 2.34 → 7.74 GB | 33.6 s | 16.483 GB constant |
| + glibc malloc tuning (via env) | **+0.001** | 1.68 → **1.54** GB | 32.6 s | 16.483 GB constant |
| **CODE DEFAULTS, no env at all** (7875348) | **+0.0003** | 1.80 → **1.66** GB (flat from chunk 13) | 32.5 s | 16.483 GB constant |

`live_arrays()` constant to the byte across every chunk **rules out hypothesis 1
(jax buffer retention) outright**.  The XLA-compile hypothesis is ruled out too:
the `_reshard_z` recompile fix (T.4) removes the only per-chunk compilation and
leaves the slope untouched.

### T.2 — ROOT CAUSE: glibc heap retention driven by XLA:CPU's transient churn

XLA:CPU allocates and frees every intermediate through plain `malloc`/`free`.
glibc's mmap threshold is **dynamic**: the first time an mmap'd block is freed it
raises the threshold to that block's size (capped at 32 MB) and sets
`trim_threshold = 2 × mmap_threshold`.  After that every allocation below 32 MB
comes from the sbrk heap / per-thread arenas, and heap memory returns to the OS
only when the *top* of the heap is free.  With 28 XLA worker threads churning
multi-MB contraction scratch inside the replicated `B(BᴴZ)` back-solve that
never happens — so RSS ratchets in proportion to the number of contraction
blocks executed, i.e. **to the FLOP count**, exactly the law measured in T.0.
Anonymous memory, page cache flat, `live_arrays` flat: all three follow.

**Fix — both halves default ON, numerics untouched:**
- `runtime/__init__.py::tune_glibc_malloc()`, called from `set_default_env()`
  (hence `bootstrap()`, hence all 9 CLIs) **before `import jax`**:
  `mallopt(M_MMAP_THRESHOLD, 1 MB)` + `mallopt(M_TRIM_THRESHOLD, 128 MB)`.
  Pinning `M_MMAP_THRESHOLD` also **disables** the dynamic adjustment, so every
  ≥1 MB allocation is mmap'd and `munmap`'d on free.  Off with
  `LORRAX_MALLOC_TUNE=0`; sized by `LORRAX_MALLOC_MMAP_MB` / `LORRAX_MALLOC_TRIM_MB`.
- `gw/isdf_fitting.py`: one `malloc_trim(0)` per r-chunk.  Off with
  `LORRAX_MALLOC_TRIM=0`.

**Effect: +0.176 → +0.001 GB/rank/chunk (176×), at 32.4 → 32.6 s/chunk (noise),
and steady-state RSS 2.46 → 1.54 GB/rank as a bonus.**  Extrapolated to the
overnight configs: run_B's 15-chunk fit no longer accumulates ~98 GB/rank;
7874803's 81-chunk fit no longer accumulates ~16 GB/rank.

### T.3 — the OTHER defect this exposed: the back-solve arena is 3×`nq·μ²`, unmodelled

`run_B` (7875071) did not die of the ramp — it died on a **single allocation of
27,759,200,256 B**, and that number is exact:

    nq·16·(μ_pad² + 2·μ_log²) = 144·16·(2016² + 2·1998²) = 27,759,200,256   ✓ to the byte

That is `isdf/core.py::_solve_all_at_once` → `_sharded_cho_solve_batch` →
`_pinv_matmul_logical`: the replicated `(nq, μ_pad, μ_pad)` factor **plus** two
logical-extent copies (`solve_at_logical`'s μ-slice of B, and `B_log.conj().T`),
per rank, unsharded, **re-materialised on every r-chunk**, and entirely
unmodelled by the planner (which reported `C_fit_one_rchunk` = 10.57 GB there).
`run_A` at μ_pad 2448 / μ_log 2406 is **40.48 GB/rank** — which is why it died in
chunk 2 with almost no ramp yet.

### T.4 — `_reshard_z` was recompiled on every r-chunk on every rank (time, not memory)

`isdf/core.py::solve_zeta` defined its `@partial(jax.jit, donate_argnums=(0,))
_reshard_z` **inside the function body**, so every call made a fresh function
object — a fresh key for JAX's identity-keyed trace/lower/compile caches.
`JAX_LOG_COMPILES=1` on the fixture (75 r-chunks × 4 ranks):

    Finished XLA compilation of jit(_reshard_z)  x 300      <- #1 compile, by 40%
    total compiles 1722

Hoisted into `_solve_cache` beside the other kernels:

    jit(_reshard_z)  300 -> 4        (one per rank)
    total compiles   1722 -> 1426
    zeta_fit.chunk.solve   5.140 s -> 0.563 s   (9.1x — the solve phase was 91% recompile)
    zeta_fit.chunk_loop   15.013 s -> 10.462 s  (1.43x)

Not the memory ramp (T.1 row 2 proves that), but a real wall-clock/compile-storm
win, and it removes an antipattern the codebase has already been bitten by twice
— see the `_EXTRACT_DIAG_KERNEL_CACHE` / `_QSGW_BUILD_KERNEL_CACHE` comments in
`gw/qsgw_utils.py`, which exist for exactly this reason.

### T.5 — NEW TIER: `distributed_zeta_solve` (per-q gather in the back-solve)

> **⚠ CORRECTED BY Y.2 (2026-07-26).** The per-rank gather figures below
> ("9.36 GB → 0.065 GB, 144×") are **refuted by the optimized HLO**: XLA:CPU
> materialises the whole `(nq, μ, μ)` all-gather and applies the per-q
> `dynamic_slice` afterwards, so `per_q`'s own gather buffer is *larger* than the
> replicated one (10.87 vs 9.66 GB at μ_pad=2048, measured). The real benefit is
> 1.67× on module peak, and it costs **12–40× the back-solve wall**. The tier is
> still numerically exact (T.6 stands). Read Y.2 before using this tier.


Coordinator-scoped addition.  `isdf/core.py::_solve_one_q_and_update` gathers
**one `(μ, μ)` tile at a time** instead of the whole `(q_batch, μ, μ)` stack,
looping q inside the r-chunk.  Same `_sharded_cho_solve_batch` kernel at batch 1,
so each q sees exactly the arithmetic it saw inside the batched call; only the
gathered extent changes.  Both slices are taken INSIDE the jit off a traced `q`
(identical shapes for every q ⇒ **one trace, one compile, one executable** for the
whole loop; `Z_col` is never sliced eagerly).  `donate_argnums=(2,)` chains the
accumulator exactly as `_solve_batch_and_update` does; a Python loop, not
`lax.scan`, for the documented SPMD-accumulator-replication reason.

**Per-rank gather at MoS2 12×12 / μ_pad = 2016, nq = 144: 9.36 GB → 0.065 GB
(144×).**  Counting the two logical-extent copies the whole T.3 arena goes
**27.76 GB → 0.19 GB/rank**; at μ_pad = 2448 (`run_A`) **40.48 GB → 0.28 GB**.

New input key (append-only in `gw_config.py`; no other key touched):

    distributed_zeta_solve = auto | replicated | per_q | distributed   # default auto

- `replicated` — today's whole-batch gather.
- `per_q` — the new tier.
- `distributed` — **rejected at resolve time** naming the two missing pieces:
  the ScaLAPACK `pzheevd` handler (SLATE's host `heev` SIGSEGVs, bug L-2) and the
  ζ column re-layout (columns-on-`('x','y')` → columns-on-`'y'`, scorecard J.9).
- `auto` — `replicated` while `nq·μ_pad²·16` fits under `LORRAX_ZETA_GATHER_CAP_GIB`
  (**4 GiB, deliberately separate from `LORRAX_ZETA_REPLICATE_CAP_GIB`** — that
  one gates the *factorization* route and production raises it to 16), `per_q`
  above.  Resolved ladder: fixture (0.6 MB) → replicated; c606 (0.94 GB) →
  replicated; **c1998 (9.36 GB) → per_q; c2406 (13.81 GB) → per_q**.

Threading: `gw_config` → `BackendConfig` → `gw_init` (both charge and transverse
`fit_zeta` calls) → `isdf_fitting.fit_zeta_to_h5` → `_resolve_zeta_gather` →
`fit_one_rchunk` (in the kernel cache key) → `solve_phase` → `solve_zeta`.
**Deliberately NOT in the ζ provenance hash** (`gw_init.py:184`) — the tier is
numerically neutral, so putting it there would needlessly invalidate on-disk ζ.

### T.6 — gates (all PASS except one PRE-EXISTING failure, see below)

- `py_compile` on all five touched files.
- GW cohsex fixture **P=4** vs `eqp_ref.dat` @1e-3: **max|Δ| = 1.0e-06 eV**, 0/1888 over tol.
- GW cohsex fixture **P=8** vs `eqp_ref.dat` @1e-3: **max|Δ| = 1.0e-06 eV**, 0/1888.
- **P=8 vs P=4 directly: max|Δ| = 0.00e+00** — bit-identical, mesh-invariance intact.
- `distributed_zeta_solve=auto` (fixture ⇒ resolves `replicated`) vs ref: **1.0e-06 eV**.
- `distributed_zeta_solve=per_q` vs ref: **1.0e-06 eV**; and
  **per_q vs auto directly: max|Δ| = 0.00e+00 — BIT-IDENTICAL.**
- New route-pin `test_zeta_gather_tier_ladder_is_pinned` (tests/test_zeta_mesh_invariance.py): **PASS**.
- Repro slope → 0 (T.1).

**PRE-EXISTING FAILURE, not from this work:**
`test_rank_truncate_refuses_above_the_replication_cap` fails identically on the
**untouched main checkout @1ee52b2** (job 7875510) and in wt-J (7875509):
`ibz74_rank_truncate` returns `replicated_rank_truncate` where the test asserts
`RAISE:`.  The test still encodes the pre-J.1 contract; J.1's
`_replicate_rank_truncate_ok` deliberately made that stack reachable.  **Left for
J/N to update** — touching it here would only create a merge conflict.

### T.7 — predicted ceiling for the 2406/b400 configs

> **⚠ CORRECTED BY Y.2 (2026-07-26).** The "with per_q" rows below use the
> refuted 0.19/0.28 GB back-solve arena. Measured law:
> `nq·μ_pad·(μ_pad + μ_pad/P_x)·16` ⇒ **≈14.96 GB/rank** for c2406 at P=144, not
> 0.28 GB, so c2406's total is ≈34.5 GB/rank rather than 19.8. Use `distributed`
> (V.4) — it genuinely gathers no `(μ,μ)` object — wherever the mesh is square.


With the ramp at zero and the back-solve on `per_q`, per-rank ζ-fit residency
(P=144, mesh 12×12):

| config | back-solve arena | resident L_q | Stage C | total | node (2 ranks) |
|---|---|---|---|---|---|
| c2406 b432, 15 chunks — **before** | 40.48 | 13.81 | 4.03 | 60.0 GB/rank | 120 GB / 192 |
| c2406 b432, 15 chunks — **with per_q** | **0.28** | 13.81 | 4.03 | **19.8 GB/rank** | **40 GB / 192** |
| c2406 b432, 81 chunks — with per_q | 0.28 | 13.81 | 0.75 | 16.5 GB/rank | 33 GB / 192 |
| c1998 b288, 15 chunks — with per_q | 0.19 | 9.36 | 2.69 | 13.4 GB/rank | 27 GB / 192 |

**"Can they now run 81 chunks?" — yes, and they no longer need to.**  The chunk
count was only ever a lever against the ramp; with the ramp gone the binder is
r-independent, so **15 chunks is the right choice** (81 chunks costs 5.4× more
chunk iterations for the same total FLOPs and buys 3.3 GB/rank).  Both A and B
now fit at 15 chunks with >150 GB/node of headroom, and the binder reverts to the
one J.10 already names: `F_tensor_write`'s unsharded `2·nq·μ²` SlabIO allgather
(27.6 GB for A).  Wall for A: ~600 s/chunk × 15 ≈ 2.5 h of ζ-fit (scaling
run_B's measured 274 s/chunk by (2448/2016)²), well inside the 12 h wall.

### T.8 — next lever (named, not done)

`B_log` — `solve_at_logical`'s μ-slice of `L_q` — is **loop-invariant**, yet it is
re-materialised inside the per-r-chunk solve.  Hoisting the logical slice out of
the chunk loop removes one of the three copies in T.3.  Under `per_q` that is now
worth only ~0.1 GB/rank, so it is a `replicated`-tier optimisation; record it in
case the tier is ever forced back on.

---

## U — the WFN's symmetries are now MEASURED, not inferred (wt-F, branch `trs-density-check`, base 1ee52b2, 2026-07-26)

**One line: a WFN file never says whether time reversal holds, so LORRAX
now builds the spin-resolved charge density from the file's own raw IBZ
wavefunctions at load time and asks the density — and `SymMaps` refuses
to select a time-reversal row when the answer is no, whatever `ntran`
and the k-weights imply.**

§Q's ψ(r) → ψ*(−r) corruption was the fourth bug in a family whose common
ancestor is *inferring physics from flags*. This closes the family
structurally rather than fixing the fourth instance.

### What it measures (default ON at `WfnLoader.__init__`)

| arm | statement | verdict lands as |
|---|---|---|
| **TRS** | for every ± k-pair in the file, `m_{−k}(r) = −m_k(r)` where `m = Σ_occ ψ†σψ`; gate on `max_pair ‖w·m_k + w·m_{−k}‖∞ / ‖ρ‖∞` | `loader.trs_holds`, `sym.trs_allowed` |
| **spatial table** | for each claimed `{S\|τ}`, `ρ(Sr+τ) = ρ(r)` on the k-points `S` fixes | per-op residual + named loud warning |
| **invariants** | `∫ρ d³r = N_elec`, `ρ ≥ −ε` | `report.invariants_ok` |
| **flags-vs-measurement** | does the k list need TRS to cover its own kgrid? | `report.trs_implied_by_mesh` |

### The three things that were easy to get wrong, and how they were avoided

1. **`‖Σ_k w_k m_k‖ = 0` is NOT the right test.** For any IBZ reduced
   using TRS the weights count k̄'s time-reversed image but the stored ψ
   only supplies the k̄ half. MoS₂ (D₃ₕ, no inversion, spin-valley
   coupling: large `m_z` at K, opposite at K′) would have failed a naive
   total-`m` test while being perfectly nonmagnetic. The **± k-pair**
   form is exactly zero under TRS with no symmetrization step — no
   circularity, no false alarm.
2. **Coverage is adequate exactly where it matters.** A pair is testable
   iff −k is also in the file. TRS-folded IBZs have poor coverage — but
   QE only TRS-folds when TRS is *assumed*, and a magnetic system cannot
   produce such a mesh; its k-set is spatially reduced (or `nosym`), and
   that is closed under k → −k. So the cases where TRS *can* be broken
   are precisely the cases with full coverage. Measured: 12×12 **100 %**,
   gnppm 100 %, bispinor 100 %, cohsex_debug 78 %, si_cohsex 12 %.
3. **The raw IBZ-weighted ρ is NOT point-group symmetric** (it is
   ρ_full's unsymmetrized precursor), so testing a general op against it
   would flag every legitimate op — and symmetrizing it first would make
   the test circular. The spatial arm therefore restricts each op to its
   **little group**; for a Γ-centred mesh Γ is in every little group, so
   every op is still tested (measured: 12/12, 2/2, 48/48).

### Non-circularity, stated so it can be audited

Built from the raw IBZ coefficients (`wfns/coeffs` hyperslab + the
`box_index(k='ibz')` table). No `unfold_psi`, no `k='full_bz'`, no
`SymMaps` — the unfold is the thing under test. The only shared
machinery is the density quadrature itself, `psp.get_DFT_mtxels.
valence_density_from_kpoint`, **imported, never re-derived**: all four
Pauli components come out of it via the polarisation identity
(`m_x = D(a+b) − ρ`, `m_y = ρ − D(a+ib)`), costing 4 quadrature calls
instead of 2 — the deliberate price of having one FFT-normalisation
convention in the tree. `tests/…::test_raw_read_matches_the_loader_ibz_path`
pins the reader bit-exactly to `loader.load(k='ibz')` so the deliberate
independence cannot become silent drift.

### TRS × non-symmorphic × spinors — the messy interplay, written down

Full derivations (T1)–(T5) + (S1) live in the
`src/common/density_symmetry_check.py` module docstring, with the unfold
counterpart in `unfold_psi`'s docstring and an AUDIT MAP at the
`sym_mats_k` augmentation site. Executable in
`tests/test_density_symmetry_check.py::test_T1_*`, `test_T3_*`,
`test_polarisation_identities_*`. Summary:

- **Θ = iσ_y K**, `Θ² = −1`. On coefficients it has **two halves**:
  `c_{Θ,−k}(G′) = iσ_y·conj(c(−G′))` — the spinor factor *and* the
  negation of the G list. **Applying one without the other is exactly
  §Q** (`ψ(r) → ψ*(−r)`: norm-, orthogonality- and ⟨T⟩-preserving, hence
  invisible; O(100 eV) wrong in V_loc/V_NL). In LORRAX the spinor half
  lives in `unfold_psi`/`trs_augment_U` and the G half in
  `sym_mats_k[sym_idx] = −S` flowing into `gvecs(k='full_bz')`; the only
  thing keeping them in step is the `len(sym_mats_k) == 2·ntran` guard.
- **`σ_y σ_i σ_y = −σ_i*` for all i ⇒ `m_{Θψ} = −m_ψ`.** That is the
  whole basis of the verdict, and it is a real-space identity: **no
  symmetry op, no τ, no umklapp vector, no `U_spinor` appears in the TRS
  arm.** A bug in any of those cannot move the TRS verdict — which is
  what makes the measurement an independent check on the unfold rather
  than a restatement of it.
- **Umklapp / TRIM** is handled by integer k-grid arithmetic (`−k ≡ k`
  ⇒ self-paired ⇒ `m_k ≡ 0`); densities, never coefficient lists, are
  compared, so no `G₀` bookkeeping exists to get wrong.
- **Non-symmorphic τ under TRS**: `tau_phase_row` is fed `S_full = −S`,
  so `exp(−i(−S·G)·τ) = exp(+i(S·G)·τ)` — the conjugate of the spatial
  phase, which is what Θ demands. No separate τ for TRS rows.
  **Verified end-to-end on `si_cohsex_debug`, which is genuinely
  non-symmorphic (`tnp = π` ⇒ `τ_frac = 1/2` glides): 48/48 ops pass at
  ≤ 3.0e-9.**
- **Why the spatial arm can use ρ at all**: ρ is a spinor TRACE, so
  `U_spinor` cancels (`ψ†U†Uψ = ψ†ψ`), the τ-phase cancels (modulus),
  and unitary mixing inside the occupied manifold cancels. Only the
  real-space point map `r → mtrx⁻¹·r + τ/2π` survives. **Corollary,
  stated plainly: this check validates `mtrx` and `tnp`, NOT `U_spinor`
  and NOT the τ-phase** (those remain covered by
  `tests/test_symmetry_unfold.py` + the §Q shape guard). Extending the
  spatial arm to `m` — an AXIAL vector, `m(Sr+τ) = det(R)·R·m(r)` with
  the CARTESIAN `R` — would close that gap; deliberate future work.
- **(T5) the band cut-off**: `nocc = max(ifmax)` is an index cut, and
  under TRS `ε_{n,−k} = ε_{n,k}` band by band, so the cut is
  TRS-consistent even for a metal. The one hazard is a cut inside a
  degenerate multiplet; `report.manifold_gap` is measured and quoted in
  any failure message so an auditor can rule that channel in or out.
- **`nspin = 2` is explicitly UNSUPPORTED, not silently wrong.** LORRAX's
  reader has no spin axis (`coeffs` axis 1 is the spinor axis
  everywhere), so the collinear channels are not addressable. The check
  says so and leaves the verdict permissive — worth knowing, since a
  `nspin=2` deck is the most likely place for TRS to actually be broken.

### The gate: how a broken-TRS verdict is enforced

`sym_mats_k` **keeps its `2·ntran` length** (`unfold_psi` hard-requires
that shape — §Q), and the gate lives in a new `_sym_mats_k_search` used
by `create_kpoint_symmetry_map`, `find_symmetry_ops_simple` and
`find_irreducible_bz_points`. With TRS disallowed those can only return
`sym_idx < ntran`, so **no ψ is conjugated anywhere in the pipeline, on
any backend** (`SymMaps` is the sole producer of `sym_idx_k`/`sym_idx_q`,
which every backend consumes). If the file's IBZ genuinely needed TRS,
`find_symmetry_ops_simple` now **raises** naming the count of
unreachable k rather than falling back to identity.

### Measured cost (Frontera CPU, 1 node, 28 threads)

| deck | grid | nk file | nk used | check | of which io / fft |
|---|---|---|---|---|---|
| cohsex_debug (ntran=12) | 15×15×60 | 4 | 4 | **1.37 s** | 0.01 / 0.64 |
| gnppm_debug (ntran=2) | 24×24×80 | 9 | 9 | **1.16 s** | 0.08 / 1.07 |
| bispinor_debug (ntran=2) | 30×30×120 | 9 | 9 | **2.69 s** | 0.27 / 2.39 |
| si_cohsex (ntran=48, glides) | 24×24×24 | 8 | 8 | **0.41 s** | 0.01 / 0.34 |
| **MoS2 12×12 (ntran=1)** | 36×36×135 | 144 | **12** | **4.77 s** | 0.56 / 4.18 |

Second and later constructions of the same file: **0.009 s** (verdict
cached per `(path, mtime, size)`). 12×12 k-ladder — cost linear, verdict
and residual order flat, which is the justification for the default
`LORRAX_TRS_MAX_K=12`:

| max_k | nk | seconds | m_rel |
|---|---|---|---|
| 8 | 9 | 3.63 | 2.5e-13 |
| **12** | **12** | **4.77** | **1.9e-13** |
| 24 | 24 | 11.4 | 9.6e-14 |
| 48 | 48 | 20.8 | 4.9e-14 |
| 0 (all) | 144 | 213 | 3.1e-14 |

Subsampling is a **sufficiency** choice, not an approximation: each ±
pair is an independent exact identity, so a subset is a *sharper*
statement than the sum (no cancellation between pairs can mask a
violation). The sample is built ±-closed and seeded with one k per op's
little group so the spatial arm stays fully testable.

**COST GOTCHA, measured, worth knowing beyond this workstream.** The
first cold `WfnLoader` on the 12×12 measured **96 s**, of which the check
itself was ~10 s. Breakdown: `import psp.get_DFT_mtxels` = **48.9 s
cold / 0.39 s warm**; `import jax` = **8.1 s cold / 1.5 s warm**; first
`jnp.fft.ifftn` (XLA compile) = 8.3 s; `box_index` = 0.09 s. **It is
100 % Lustre page-cache on the import graph, not work** — the same tax
`import jax` already pays. Nothing in the check does 50 s of anything.
On a warm node the marginal cost is the table above.

### Tolerances, and why they are not marginal

`LORRAX_TRS_TOL = 1e-6` on `‖m‖∞/‖ρ‖∞`; `LORRAX_TRS_SPATIAL_TOL = 1e-4`
on the ρ-invariance residual. Measured on every real deck:

| deck | m_rel | max spatial residual |
|---|---|---|
| MoS2 12×12 | **7.3e-14 … 1.9e-13** | 0.0 (identity, exact) |
| cohsex_debug | 9.8e-12 | 1.74e-09 (12 ops) |
| gnppm_debug | 5.0e-11 | 6.74e-11 |
| bispinor_debug | 1.4e-10 | 1.68e-10 |
| si_cohsex | 1.8e-10 | 3.01e-09 (48 ops, glides) |

So the gates sit **4–6 orders above the measured floor and 4–10 orders
below a real signal** (a spin-polarised manifold gives `m_rel = 1.000`
exactly; a bogus op gives 0.73–0.92). Not a knife edge in either
direction.

### Gates

- **(a) fixture (ntran ≥ 2, TRS-invariant) — PASSED.** cohsex_debug:
  `trs_holds=True`, **12/12 spatial ops pass**, `∫ρ = 26.000000`
  (rel err **0.0e+00**), 1.37 s. gnppm/bispinor/si_cohsex likewise;
  si_cohsex exercises **48 non-symmorphic ops**, all pass.
- **(b) the 12×12 nosym deck — PASSED, and it flags NOTHING.**
  `trs_holds=True` (MoS₂ nonmagnetic), `m_rel = 1.93e-13`, the spatial
  table trivial-passes (identity, residual **exactly 0.0**),
  `∫ρ = 26.000000` (rel **1.4e-16**), `trs_implied_by_mesh=False`,
  `SymMaps.trs_allowed=True`, `max(sym_idx_k)=0`, `nk_tot=144`, and
  `loader.load(k=[0])` is **bit-identical to the raw IBZ ψ** (`Δ = 0.0`)
  — i.e. the check confirms §Q's fix and correctly reports the physics
  was never the problem.
  **Synthetic TRS-broken control — PASSED:** a Γ-only 2-spinor deck with
  a fully spin-polarised occupied manifold gives `trs_holds=False` with
  `m_rel = 1.000` (m_z ≡ ρ), while its Kramers-paired twin
  (`ψ_partner(G) = iσ_y conj(ψ(−G))`, built from (★) itself) gives
  `m_rel < 1e-15`. Both decks are constructed G-even and band-normalised
  so their declared inversion really is a symmetry and `∫ρ = nocc` —
  i.e. **TRS is the only arm that can fire**. The gate then holds:
  `sym.trs_allowed=False`, `sym_mats_k` still `2·ntran`,
  `_sym_mats_k_search` = `ntran`, `max(sym_idx_k) < ntran`,
  `max(sym_idx_q) < ntran`.
- **(c) full loader suite green — PASSED.** `test_wfn_loader_eager.py`
  + `test_symmetry_unfold.py` + `test_wfn_transforms.py` +
  `test_density_symmetry_check.py`: **62 passed, 3 skipped**
  (job 7875511). `LORRAX_TRS_CHECK=0` is a true no-op: 32 passed,
  1 skipped. `tests/test_file_io.py` shows **13 failed / 20 passed —
  IDENTICAL on the pristine main checkout** (same job, side by side):
  pre-existing, unrelated to this branch.
- **(d) py_compile — PASSED** on all five touched files.

### Files

- **NEW** `src/common/density_symmetry_check.py` (the measurement +
  the (T1)–(T5)/(S1) derivations).
- **NEW** `tests/test_density_symmetry_check.py` (12 tests: 3 are an
  executable audit of the Pauli algebra, 5 synthetic, 4 real-fixture
  incl. a deliberately corrupted symmetry block).
- `src/file_io/wfn_loader.py` — runs the check at construction; sets
  `trs_holds` / `density_symmetry`; passes the verdict via
  `_sym_wfn_stub`; carries the "what is and is not established" block.
- `src/common/symmetry_maps.py` — `SymMaps(wfn, *, allow_trs=None)`,
  `trs_allowed`, `_sym_mats_k_search` threaded through the three search
  sites, the hard raise, the AUDIT MAP, and the (★) derivation in
  `unfold_psi`.
- `docs/dev/env_vars.md` — `LORRAX_TRS_CHECK` / `_TOL` / `_SPATIAL_TOL`
  / `_MAX_K` registered in §1.

**NOT committed.** Artifacts + sbatch/scripts:
`/scratch2/08271/jackmc/lorrax_setup/wk_U/`; jobs `7875319` (deck probe),
`7875329`, `7875333`, `7875336`, `7875396` (cost breakdown), `7875501`
(import breakdown), `7875504`, `7875511` (final).

### Notes for the orchestrator

1. **Import coupling to workstream S.** The check imports
   `valence_density_from_kpoint` and `spin_degeneracy_factor` from
   `psp/get_DFT_mtxels.py` (FENCED — read only, not edited). If S changes
   either signature, this breaks; the import is inside a `try/except`
   that degrades to a warning + permissive verdict, so it fails soft.
2. **Known-correct warning noise.** `tests/test_wfn_loader_eager.py`'s
   synthetic deck has random coefficients, so it genuinely has no TRS
   and no normalisation — the check now says so (6 RuntimeWarnings).
   The suite stays green. Left alone on purpose (another suite's
   fixture); making its bands Kramers-paired would silence it honestly.
3. **Deferred, ranked**: (i) extend the spatial arm to `m` with the
   axial-vector rotation, closing the `U_spinor` gap noted above;
   (ii) add a spin axis to the reader so `nspin=2` becomes measurable;
   (iii) if the cold-import tax ever matters at P=144, move the check
   from `__init__` to `_ensure_sym()` — equally protective, since
   `SymMaps` is the sole consumer.

---

## V/W — the ScaLAPACK `pzheevd` eigh backend + the `distributed` ζ tier (wt-J, branch `scalapack-eigh`, base 8841d5e, 2026-07-26 — NOT committed)

> **⚠ CLAIM-DECAY (AL, 2026-07-27): fabric unverified for the MPI walls; em1 for the Gloo walls.** The JAX/Gloo collective walls here are em1-scoped (see the AL banner above other sections). `pzheevd` runs over the container's Intel MPI, NOT Gloo — which fabric/provider it actually used (ofi/mlx vs tcp over em1) was never measured, so its 30-min-class walls (AC.2) are unpriced on the real fabric too, in both directions. Re-verify before planning on them.

**One line: the charge factor is no longer replicated on every rank. `C_q`
is eigendecomposed distributed (MKL ScaLAPACK `pzheevd`), truncated on the
replicated spectrum, and the truncated pseudo-inverse `C⁺` stays 2D-sharded;
the back-solve is a stacked GEMM `C⁺@Z` with BOTH operands 2D-sharded — and
it moves 38% FEWER collective bytes than the tier it replaces, because it
also deletes `_reshard_z`.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_V/` (`run.sh`, `build.sh`,
`eigh_ladder.py`, `audit_hlo.py`, `hlo_dist/`, `hlo_rep/`, `out_*.txt`,
`gate_*.txt`, `P4dist/ P16dist/ P8dist/ P8auto/ P4rep/`).
Library: `$WORK/lorrax_ffi_unified/build_host_V/liblorrax_ffi_host.so`
(versioned stage — the working `build_host/` is untouched).

### V.1 — MKL ScaLAPACK on Frontera: the facts that decide the implementation

* **No mkl module is needed.** The default `intel/19.1.1 + impi/19.0.9`
  stack sets `MKLROOT=/opt/intel/compilers_and_libraries_2020.1.217/linux/mkl`
  (which is MKL **2020 Update 1**, despite the module name). There is no
  netlib scalapack module on Frontera — MKL is the only ScaLAPACK.
* `libmkl_scalapack_lp64.{so,a}` exports `pzheevd_`, `pzheevr_`, `pzheevx_`,
  `pzhegvx_`, and also `numroc_`/`descinit_` (which are **not** in the BLACS
  lib). `libmkl_blacs_intelmpi_lp64.so` exports the C-BLACS set
  (`Csys2blacs_handle`, `Cblacs_gridinit/gridinfo/gridexit`) — but
  `mkl_blacs.h` declares only the Fortran names, so the C ones must be
  declared by hand (we already did, for solve_lu).
* **The link line P's CMake already emits is correct and needs nothing new**
  (`src/ffi/common/cpp/host/CMakeLists.txt`):
  `-L$MKLROOT/lib/intel64_lin -Wl,--no-as-needed -lmkl_scalapack_lp64
   -lmkl_blacs_intelmpi_lp64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core
   -lgomp -lpthread -lm -ldl`. The eigh handler adds **zero** link deps.
* **`pzheevd` implements `JOBZ='V'` ONLY** — netlib's source says "(NOT
  IMPLEMENTED YET)" for `'N'`, and MKL returns `INFO = -1`. So the handler
  hard-wires `'V'`; there is no `compute_evecs` knob (and no dead branch).
* **The workspace query is MANDATORY, not an optimisation.** MKL returns an
  `LWORK` far above the netlib `LWMIN` on multi-rank grids (measured **368×**
  at N=20000 on a 2×2 grid) and **rejects the netlib minimum with
  `INFO = -16`**. The handler treats a failed query as fatal and uses
  `max(query, reference formula)`. Measured here: n=2016 on a 4×4 grid →
  WORK 11.4 MB + RWORK 6.0 MB + IWORK 0.06 MB per rank — modest, but it is
  `malloc`'d inside the FFI and therefore **invisible to the planner**, so
  `LORRAX_SCALAPACK_EIGH_LOG=1` prints it.
* **THE TRAP, found the hard way: `pXheevd` can return `INFO = 0` with
  CORRECT eigenvalues and a SILENTLY GARBAGE `Z`.** The eigenvector
  back-transform `pXunmtr`/`pXormtr` is a *separate* workspace requirement
  that `pXheevd`'s published `LWORK` formula does not always cover — and
  neither does MKL's query. Measured: real symmetric n=32 on a 1×1 grid,
  PDSYEVD asks for **2305** doubles, PDORMTR needs **3072** plus the ~5N
  of leading TAU/D/E/D2/E2 offsets pXheevd carves off the front ⇒ **3232**.
  At 2305 the run returns `INFO = 0`, `max|Δλ| = 7.1e-15`, `‖ZᴴZ−I‖ =
  6.0e-15` — and `‖AZ−ZΛ‖/‖A‖ = 1.40`. The ONLY visible symptom is one
  `PDORMTR parameter number 16 had an illegal value` line on stderr from
  PXERBLA, which is trivially lost in a 16-rank log. **The complex path
  escaped it purely by luck** (`pzheevd`'s formula, `N+(NP0+MQ0+NB)·NB`,
  happens to be bigger than `pzunmtr`'s bound). The handler now floors
  `LWORK` with `max((NB(NB−1))/2, (NP0+MQ0)·NB) + NB² + 8N` on BOTH paths.
  Caught by the new strict-eigenpair contract cell — an eigenvalue-only
  test would have shipped this.
* Descriptor rules (verified against MKL, not just the docs): `MB_A == NB_A`
  (else `INFO = -706`), `IA = JA = 1` (else `INFO = -4`), `descz` must match
  `desca` in M/N/MB/NB/RSRC/CTXT (else `INFO = -1205`), **A is destroyed**.
  No minimum NB. ⇒ same geometry class as `pXgetrf`: **square or 1-D mesh**,
  block `g = N / max(Px, Py)`.
* `pzheevr` (MRRR) is the recorded fallback: supports `JOBZ='N'` and
  eigenvalue subsets, needs ~n²/p instead of ~3n²/p workspace, but is
  measurably worse on clustered spectra (‖ZᴴZ−I‖ 1.9e-12 vs `pzheevd`'s
  7.6e-14 on a 500-fold cluster) — and the charge CCT *is* clustered. One
  MKL hang bug exists for `pzheevr`/`pzheevx` but **ILP64 only** (fixed
  2025.1); our LP64 build is unaffected.

### V.2 — the handler (10 targets now in the host lib)

`src/ffi/scalapack/cpp/eigh_ffi.cc` → `ScalapackEighHostFfi` /
`lorrax_scalapack_eigh`. **Batched**: one FFI call factors the whole
`(Nq, N, N)` stack, reusing one descriptor and one workspace, so `Nq`
matrices cost one collective-serialisation round instead of `Nq`.
`src/ffi/scalapack/cpp/blacs_grid.h` is new and now holds the Fortran-ABI
prototypes + the per-`SlateCtx` BLACS-context cache that `solve_lu_ffi.cc`
used to keep privately (it now includes the header; one cache, one copy).
Python: `ffi/scalapack/eigh.py` (`batched_distributed_eigh` +
`distributed_eigh` for the one-tile facade path).

Facade (`ffi/linalg/resolve.py`):
* `('eigh','scalapack')` registered, host-only, square-or-1-D geometry —
  the eigh geometry guard no longer force-rejects `p != q` for *every*
  backend, only for cusolvermp/slate (whose reasons are a deadlock and a
  library requirement, neither of which applies to `pXheevd`).
* **new vocabulary `'distributed'`** = "spread ONE tile over the mesh with
  this platform's distributed eigh" → `scalapack` on cpu **permanently**,
  `cusolvermp` on CUDA (`_DISTRIBUTED_DEFAULT`).
* **`('eigh','slate')` on a CPU mesh is now a resolve-time REFUSAL** naming
  bug L-2 and the replacement. It used to resolve happily and then SIGSEGV
  the job with no traceback — the worst class of broken promise (L-1's
  lesson: the handler IS compiled, so no capability probe catches it).

> **Deviation from the brief, deliberate, flagged.** The brief asked for the
> CPU **`auto`** eigh default to become scalapack too. It must not:
> `resolve_backend('eigh','auto')` is what `bse_setup`/`vq_interp`/htransform
> branch on, and flipping it would route the htransform fH_q eigh through
> `nq` serial distributed solves — slower by the measured 100–600×/matrix
> AND not bit-exact (different gauge), breaking htransform's
> `max|ΔE| = 0.000e+00` gate. `auto` therefore keeps its documented meaning
> ("the fastest eigh for this call" = native batched) and `distributed` is
> the name for "one tile over the mesh", permanently ScaLAPACK on CPU.

Build: `config/frontera/build_ffi_host.sh` unchanged except the target list
— **and one real bug fixed**: its `nm -D "$SO" | grep -q ...` symbol check
runs under `set -o pipefail`, so `grep -q`'s early exit SIGPIPEs `nm` and
the pipeline returns 141 ⇒ every symbol grep matched EARLY was reported
`MISSING`. It only ever "passed" for the symbols sitting at the end of the
table. Now reads the table once into a variable.

Infra finding for anyone else driving the host FFI from a holder job:
**`FI_PROVIDER_PATH=$I_MPI_ROOT/intel64/libfabric/lib/prov` is required**,
or `MPI_Init_thread` dies `OFI addrinfo() failed ... No data available`
regardless of `FI_PROVIDER`. wk_L's sbatch got it from the outer module
env; an `ssh + srun --overlap` into a holder does not.

### V.3 — TEST LADDER (library level, c128 Hermitian PD, `P(None,'x','y')`)

`FI_PROVIDER=tcp` (IPoIB) throughout, so every inter-node time is
**pessimistic** (scorecard L's caveat).

| mesh (nodes) | n | ‖AZ−ZΛ‖/‖A‖ | ‖ZᴴZ−I‖ | max\|Δλ\|/λ_max | ‖C⁺−C⁺_native‖/‖C⁺‖ | t/matrix | native (replicated) |
|---|---|---|---|---|---|---|---|
| 1×1 (1) | 64 | 1.08e-15 | 1.26e-14 | 7.2e-16 | 2.3e-15 | 0.009 s | — |
| 1×1 (1) | 512 | 1.33e-15 | 6.5e-14 | 1.8e-15 | 4.8e-15 | 0.28 s | — |
| 2×2 (2) | 64 | 1.19e-15 | 1.50e-14 | 1.1e-15 | 2.6e-15 | 0.012 s | — |
| 2×2 (2) | 512 | 2.12e-15 | 1.09e-13 | 2.5e-15 | 6.4e-15 | 0.54 s | **0.076 s** |
| 2×2 (2) | 2048 | 4.00e-15 | 3.95e-13 | 5.3e-15 | 1.18e-14 | 2.33 s | **1.44 s** |
| **4×4 (8)** | **2016** (μ_pad, c1998) | — | — | — | — | **1.99 s** | ~1.4 s* |
| **4×4 (8)** | **2448** (μ_pad, c2406) | — | — | — | — | **2.89 s** | ~2.5 s* |

\* the native column is `jnp.linalg.eigh` on the fully REPLICATED stack,
measured at 2×2 and (\*) extrapolated by n³ to the two 4×4 sizes — it is
P-INDEPENDENT by construction (every rank does the whole thing), so the
2×2 measurement IS the 4×4 value.  Latencies are the pessimistic
`FI_PROVIDER=tcp` figure and exclude a 7–29 s FIRST call (BLACS grid +
XLA compile), amortised from rep 1 — hoist it out of loops.

**Read this table honestly: `pzheevd` does not beat the replicated native
eigh on WALL TIME at these sizes and rank counts** — it is 7× slower at
n=512/4 ranks and roughly at parity (0.7–1.2×) at n≈2000–2450 on 16 ranks.
What it buys is the two things wall time does not show: (a) the
`(nq, μ, μ)` factor is **never replicated** — 65 MB/rank of local block
instead of 9.36 GB/rank of gathered stack at MoS2 12×12; and (b) the cost
**scales with P**, where the native path is flat, so the crossover moves
with both μ and rank count and the ~4k-centroid TIME wall (J.2 #5,
~86 h at μ=10k) stops existing.  At 16 ranks the curve has just reached
parity; at 144 it is the only option.

The 4×4 rows are **timing only**: three attempts to also collect the
correctness metrics there died in the harness's *reference* path with
`jax.distributed` Gloo `DEADLINE_EXCEEDED` / coordination `Socket closed`
— the documented infra flakiness, under two live 72-node flagship jobs
plus a second agent's holder.  `pzheevd` itself ran clean at both sizes
every time (per-rank workspace: 11.4 MB WORK + 6.0 MB RWORK at n=2016;
17.2 MB + 8.7 MB at n=2448).  Correctness at 4×4 is instead covered
end-to-end by the P=16 fixture gate below.

Reruns on a fixed grid are **bit-deterministic**. The load-bearing column is
the last correctness one: `C⁺ = Z diag(1/λ) Zᴴ` — the object the ζ-fit
actually consumes — agrees with `jnp.linalg.eigh` to **1e-14** even though
the eigenvectors themselves do not and must never be compared across meshes
(gauge). Facade path (`resolve_backend('eigh','distributed') → scalapack`
→ `dispatch_eigh` on one `(n,n)` tile) verified end-to-end at 1×1 and 2×2:
residual 1.06e-15 / 1.11e-15.

### V.4 — `distributed_zeta_solve = distributed` — the tier is REAL now

`isdf/core.py`: `_resolve_zeta_gather`'s rejection is replaced by the path.
New charge route string **`distributed_rank_truncate`** (existing route
strings byte-identical — verified by the P=8 `auto` run below, which still
prints `path=replicated_rank_truncate` / `tier=replicated`).

Per q, nothing O(μ²) is ever replicated:

```
C_q, C⁺  (nq, μ, μ)  P(None,'x','y')      V  (nq, μ, μ)  P(None,'x','y')
λ        (nq, μ)     replicated           Z, ζ (nq, μ, r) P(None,'x','y')
```

1. **`pzheevd(C_q)`** over the whole mesh → `λ` replicated (ScaLAPACK's own
   contract: `W` is a global output on every grid process), `V` 2D-sharded.
2. **LOCAL identical truncation** `keep = λ > rcond·λ_max` — no collective,
   and no chance of a rank-dependent cut. `n_keep`/`λ_max`/`λ_min(kept)`
   telemetry preserved verbatim (`LORRAX_ZETA_RANK_LOG`).
3. **`C⁺ = Σ_keep vᵢvᵢᴴ/λᵢ` formed 2D-sharded**, in ONE `shard_map`:
   all-gather V along `'x'` (μ²/Py) → local einsum → `psum_scatter` along
   `'y'`. Explicit `C⁺` (not the factor `B` with `BBᴴ = C⁺`) costs one extra
   `nq·μ³` at fit time and **halves** the per-r-chunk back-solve: one GEMM
   instead of two — and the r-chunk loop runs 9–81 times.
4. **Back-solve `ζ = C⁺Z`**, one `shard_map`: all-gather `C⁺`'s row block
   along `'y'` (μ²/Px), all-gather Z's column block along `'x'` (μ·r/Py),
   local einsum. No psum. Output lands at `P(None,'x','y')` — so the tier
   also **skips `_reshard_z` entirely** and pays only the second leg of the
   output reshard. q-batched under the existing
   `LORRAX_ZETA_GATHER_CAP_GIB`.

**J.9's trap, avoided rather than fixed.** The reason the first SUMMA
attempt NaN'd is that `Z` had been resharded to `P(None,None,('x','y'))` —
columns over the FLAT mesh — so ranks sharing a `y` index hold unrelated
column blocks and a `psum` over `'x'` sums partial products from different
columns. This tier never does that reshard: it consumes Z in the layout
`z_q_from_psi_sm` builds it in.

**Padded extent, exactly.** ScaLAPACK descriptors need `n % Px == n % Py == 0`,
which `n_rmu_logical` generally is not and `n_rmu_padded` always is, so this
route factors the identity-padded block-diagonal `[C_log 0; 0 I]`. The
blocks do not mix, so `C⁺`'s logical block is `pinv(C_log)`. One subtlety
that WOULD have made the cut device-dependent and is handled exactly: the
pad contributes `n_pad − n_log` eigenvalues equal to 1, so when
`λ_max(C_log) < 1` (true at the fixture: 1.7e-3) a naive `λ[-1]` would use
the PAD's 1.0 as `λ_max` and truncate harder. Ascending order makes the fix
exact — `λ_max = λ[-1] if λ[-1] > 1 else λ[n_log-1]`.

Transverse channels: `distributed` resolves to `per_q` (visible in the
banner, which prints request and resolution side by side) rather than
raising — ONE key drives both channels and a bispinor run must not die in
the transverse fit after the charge fit succeeded. See W.3.

### V.5 — GATES (cohsex fixture, `wk_V/gate_*.txt`)

| gate | result |
|---|---|
| **P=4 (2×2), tier=distributed**, eqp vs `eqp_ref.dat` @1e-3 | **PASS, max\|Δ\| = 1.0e-06 eV** (file print precision), 0/1888 over tol |
| **P=16 (4×4), tier=distributed**, eqp vs ref @1e-3 | **PASS, max\|Δ\| = 1.0e-06 eV**, 0/1888 |
| **mesh-shape invariance 2×2 vs 4×4** (distributed vs distributed) | **max\|Δ\| = 0.00e+00** — identical at print precision |
| **distributed vs replicated** (P4 distributed vs P8 `auto`→replicated) | **max\|Δ\| = 0.00e+00** |
| **P=8 (2×4), tier=distributed** | **REFUSED at resolve time**, before any work: `eigh backend 'scalapack': mesh 2x4 unsupported — pXheevd needs square descriptor blocks (MB == NB)…`. This is the gate, not a failure: rectangular production meshes must reject cleanly, never hang. |
| **P=8 (2×4), `auto` (regression)** | **PASS 1.0e-06 eV**, route still `replicated_rank_truncate` / `tier=replicated` — existing tiers byte-identical |
| `n_keep` telemetry on the new route | present: `[zeta rank_truncate/distributed] n_pad=64 rcond=1.0e-08 n_keep/q=[64 …]` |

The P=4/P=16 agreement being *exactly* 0.00e+00 rather than the expected
~1e-10-class is a property of the fixture, not of the tier: μ=60 is fully
rank-retaining (`n_keep = 60/60`, `n_keep = 64/64` padded) and well
conditioned, so the two gauges give the same ζ to well below eqp print
precision. At production conditioning the ~κ·ε class still applies.

**Route-pin test updated** (`tests/test_zeta_mesh_invariance.py::
test_zeta_gather_tier_ladder_is_pinned`, PASSES): `auto` may never pick
`distributed` at any size; `charge_zeta_solve='cholesky'` + distributed
refuses; missing mesh refuses; transverse → `per_q`.

**Two cells of `tests/test_ffi_linalg_contract.py` fixed** (both broken on
the untouched main checkout @8841d5e — verified by running them there):

* `test_compute_wfns_fi_slate_matches_native_cpu` called SLATE's host
  `heev` and **SIGSEGV'd the whole pytest process** — no traceback, no
  summary, and every later cell in the file silently unrun. Replaced by
  `..._slate_refused_on_cpu` (asserts the new resolve-time refusal) plus a
  new `..._scalapack_matches_native_cpu` that runs the same
  `compute_wfns_fi` comparison on the backend that works there.
* `test_slate_eigh_true_eigenvectors_cpu` (the direct wrapper cell) is
  **skipped** with the bug reference instead of SIGSEGV-ing, and replaced
  in coverage by a new `test_scalapack_eigh_true_eigenvectors_cpu`
  asserting the same STRICT contract (`A@Z == Z diag(W)`, unitary Z, rerun
  bit-determinism) on the backend that works. That cell is what caught the
  PDORMTR workspace trap above.
* `test_compute_wfns_fi_rejects_bad_backend` asserted
  `match="eigh_backend"` against a message that has always read "eigh
  backend must be one of …" (op-generic — `resolve.py` must not hard-code
  per-consumer config-key names). Regex relaxed to `"backend must be one
  of"`.

### V.6 — **HLO COLLECTIVE AUDIT** (rank-0 `--xla_dump_to`, K's recipe; `wk_V/audit_hlo.py`)

Same fixture, same 2×2 mesh, `distributed_zeta_solve = replicated` vs
`= distributed`, optimized HLO, every collective's RESULT bytes counted
(tuple results included — an all-to-all's tuple IS its payload).

| | replicated | distributed |
|---|---|---|
| modules scanned | 376 | 378 |
| collectives | 81 | 72 |
| **total collective bytes** | **0.2484 GB** | **0.1535 GB** (−38%) |
| all-to-all | 139,651,200 | 45,691,200 |
| collective-permute | 59,083,272 | 763,200 |
| all-gather | 48,189,956 | 105,473,156 |
| reduce-scatter | 0 | 129,600 |

Where it comes from, instruction by instruction:

* **replicated**: `jit(_solve_all_at_once)` all-gathers
  `c128[9,60,60]` = **518,400 B** = `nq·μ²·16` — *the whole factor*, on
  every r-chunk — plus `c128[9,30,60]`. And `jit(_reshard_z)` costs
  **64.8 MB all-to-all + 29.2 MB collective-permute** moving the whole
  `nq·μ·r` tensor from `P(None,'x','y')` to `P(None,None,('x','y'))`.
* **distributed**: the factor is touched only as `c128[9,30,60]` =
  **259,200 B** = `μ²/Px` (in `jit(_fn)`, once per fit, plus a
  `c128[9,30,30]` reduce-scatter), and `jit(_block)`'s largest collective is
  the **Z** column gather `c128[9,60,6750]` = 58.3 MB = `μ·r/Py`.
  `_reshard_z` **does not appear at all.**

**Assertion demanded by the brief — no collective moves an O(μ²)-per-q
object beyond the intended factor distribution: HOLDS.** In the distributed
dump the largest `(μ,μ)`-class collective anywhere in the run is 691 kB, and
it belongs to `_solve_w` (the downstream W solve), not the ζ tier. The ζ
tier's own factor traffic is exactly the two intended pieces, `μ²/Py`
(forming `C⁺`) and `μ²/Px` (applying it).

Scaling this to MoS2 12×12 (nq=144, μ_pad=2016, r_chunk=11664, 12×12 mesh),
per rank per r-chunk: replicated **9.36 GB** of factor gather alone, against
`nq·(μ²/Px + μ·r/Py)·16` = **5.3 GB** here, with a 36.8 MB live transient
per q instead of a 9.36 GB one — plus the deleted `_reshard_z`.

### V.7 — what this unblocks, and what it does not

J.10 named two ~4k-centroid walls. This removes the second one: the
replicated rank-truncate eigh, `O(nq·μ³)` with **no P-scaling at all**
(~5.5 h at μ=4k, ~86 h at μ=10k on 28 cores). It is now `O(nq·μ³/P)`.
The remaining wall is unchanged and environmental: `F_tensor_write`'s
unsharded `2·nq·μ²` SlabIO allgather at **μ ≈ 3,960**, cured by installing
mpi4py + `HDF5_MPI=ON` h5py so `PHDF5_HOST` becomes reachable.

**Hard constraint to plan production around:** this tier needs a **square
or 1-D** mesh. The current production 8×10 (P=80) and 12×12 (P=144) split
differently — 12×12 is square and works; 8×10 does **not** and refuses at
resolve time. Choose 64/100/144/196 ranks if the tier is wanted.

### W.3 — transverse channel: STATUS, and why there is little to mirror

*(assessed, not implemented — flagged as such)*

The transverse route **already is** the distributed pattern this workstream
built for the charge channel: `distributed_lu = scalapack` →
`solver_kind='scalapack_lu'` → `ffi.scalapack.batched_distributed_solve_lu`
(`pXgetrf` + `pXgetrs`), with `L_log` and `Z_log` both pinned to
`P(None,'x','y')` inside `_dist_ridged_lu` and the result reshard through
the same `_reshard_zeta_mu_X_r_Y_to_mu_XY` single all-to-all. Nothing is
gathered. There is no eigh-based truncation to mirror: the transverse CCTᵘ
is Hermitian **indefinite** (γ̃ⁱ⊗γ̃ⁱ has both eigenvalue signs), so the
spectral cut that cures the charge channel does not apply — its
conditioner is the `LU_RIDGE = 1e-12·|tr|/n` lift.

The real transverse gap, recorded for whoever picks it up:

1. **The distributed LU silently falls back.** `solve_zeta` runs the
   transverse solve at the **LOGICAL** μ extent (load-bearing —
   ROOT_CAUSE.md 2026-07-08: solving the identity-padded system amplifies
   pad-shape LU roundoff O(1) in the near-null transverse modes), but the
   block-cyclic descriptors need `n_log % Px == n_log % Py == 0`. When they
   do not divide, it warns and drops to the per-q replicated
   `jnp.linalg.solve`. At production `n_log` is rarely divisible ⇒ the
   distributed transverse route is mostly unreachable *in practice*. Fixing
   it properly means either padding to a mesh-divisible extent with a
   **provably** roundoff-neutral pad for an indefinite system (the thing
   ROOT_CAUSE.md says does not hold), or a descriptor with a ragged last
   block (ScaLAPACK supports that; the one-tile-per-rank JAX layout does
   not).
2. **It is gated by the wrong key for a user who set
   `distributed_zeta_solve`.** Today: `distributed_lu`. Unifying the two
   keys is a config-surface decision, not an implementation one.
3. A `pzheevr`-based indefinite path is NOT the answer — an eigh of an
   indefinite matrix gives no truncation criterion.

**Recommendation: do not "mirror" the charge work here.** Spend the effort
on (1), which is a numerics question about the pad, not a linalg-library one.

### V.7b — deliberately NOT done (one-line follow-ups, named)

* The **input key `eigh_backend`** still validates against
  `auto|off|cusolvermp|slate` in `gw_config.py`, so `scalapack` /
  `distributed` are reachable from `ffi.linalg` and from the ζ tier but
  not from `cohsex.in` or `--eigh-backend`. Widening it (gw_config +
  the htransform / exciton_bands CLI choices) would let a too-large
  htransform `fH_q` tile use the distributed CPU eigh. Additive and safe;
  left out because nothing in this workstream needs it and the key's
  vocabulary is a config-surface decision.
* `pzheevr` is researched and recorded (V.1) but not wired. Wire it only
  if a `pzheevd` bug shows up or an eigenvalue-SUBSET solve is wanted —
  same descriptor contract, one more TU.

### V.7c — LIBRARY STAGE (do not clobber the working one)

`$WORK/lorrax_ffi_unified/build_host_V/liblorrax_ffi_host.so` (10 targets).
The pre-existing `build_host/` (9 targets) is untouched, so every existing
job keeps its library.  Point `LORRAX_FFI_HOST_SO` at `build_host_V` to use
the eigh handler; rebuild with
`LORRAX_ROOT=<worktree> LORRAX_SLATE_HOST_INSTALL_DIR=$WORK/slate_builds/cpu/install
 LORRAX_FFI_HOST_STAGE=$WORK/lorrax_ffi_unified/build_host_V
 config/frontera/build_ffi_host.sh` **inside the container with
`--bind …,/opt/intel,…`**.

### V.8 — cleanliness / files touched (worktree only, NOT committed)

New: `src/ffi/scalapack/cpp/eigh_ffi.cc`, `src/ffi/scalapack/cpp/blacs_grid.h`,
`src/ffi/scalapack/eigh.py`.
Modified: `src/ffi/scalapack/cpp/solve_lu_ffi.cc` (now includes the shared
header; its private BLACS cache deleted), `src/ffi/scalapack/__init__.py`,
`src/ffi/common/ffi_loader.py`, `src/ffi/common/cpp/host/CMakeLists.txt`,
`config/frontera/build_ffi_host.sh`, `src/ffi/linalg/resolve.py`,
`src/ffi/linalg/dispatch.py`, `src/isdf/core.py`, `src/gw/isdf_fitting.py`,
`tests/test_zeta_mesh_invariance.py`, `docs/dev/linalg_ffi.md`.
No dead branches: `compute_evecs` is not exposed (pzheevd has no `'N'`),
`slate` eigh on CPU is refused rather than kept as an unreachable option,
and `dispatch_eigh`'s slate/scalapack tail is one call, not two.

## X — the exact (G-space) V_H now STRONG-SCALES: one distributed kernel, three consumers, ONE psum (wt-G, branch `gspace-vh-scaling`, base 8841d5e — NOT committed)

> **⚠ CLAIM-DECAY (AL, 2026-07-27): em1-scoped collective walls.** Every multi-node JAX CPU-collective WALL TIME in this section was measured while Gloo was bound to Frontera's 1 GbE management NIC (`em1`, 129.114.x.x) — the campaign-wide default before `runtime.pin_gloo_interface()` landed (AK.10/AL). Byte counts, HLO collective inventories and residency ledgers remain valid; the wall-time consequences of collective-bound stages re-price on ib0 (measured 3.3x whole-pipeline at 785c/P=16, and see AL for 606c/P=80). Single-node numbers are unaffected.

S's §7 audit said the exact-V_H route "does not scale at all today" — `sharding=None`,
a 1×1 mesh built inline, a host-side Python k loop, a numpy output, efficiency 1/P —
and set out what it would take. This is that work: built to S's blueprint with two
deliberate departures, measured on the 12×12 production deck at P = 1/4/8/16, and
audited instruction-by-instruction in the optimized HLO.

**Headline.** The shared kernel `compute_hartree_matrix` goes from **159.4 s at P=1
to 14.3 s at P=16** (11.2×, 70 % efficiency) on MoS₂ 12×12 / 120 bands; the whole
`kin_ion.h5` generation goes 311.3 s → 31.7 s. The communication is **one 1.4 MB
all-reduce plus one 8.3 MB assembly all-gather per invocation — and nothing else**.
`hartree_source=gspace` now runs distributed inside the driver and reproduces the
`stored` route's QP energies exactly.

### X.1 — WHAT WAS ACTUALLY IN THE WAY (and it was not the k loop)

`gw.kin_ion_io` could not have been distributed by "adding a mesh", because
**importing it already destroyed the ability to join a distributed world**:

```
psp/get_DFT_mtxels.py:90   try: devs = jax.devices(); print(f"JAX: {len(devs)} ...")
```

— a device banner at *module import*. `jax.devices()` brings up the XLA backend, and
`jax.distributed.initialize()` refuses to run afterwards ("must be called before any
JAX calls that might initialise the XLA backend"). Every CLI that imports the psp
stack was pinned to one process **whatever its own header did**; the first `srun -n 4`
attempt died in exactly that assertion. It is now `report_devices()`, a function
nobody calls at import. **Generalisable: a diagnostic print at import time cost this
code base a distributed CLI.**

The second blocker is why S's design needed adapting. `WfnLoader.load` only ever
returns a **global** array: every rank must request the same `(bands, k)` window and
each owns a band shard of one logical object. That is right for the GW pipeline and
wrong for any kernel whose parallelism is over k — rank *r* asking for `k=[7]` while
rank *s* asks for `k=[9]` builds a global array whose "shards" are pieces of different
physical objects. There was no primitive for *this rank's own data*. There is now:

* `WfnLoader.load_process_local(bands=, k=)` → a **single-device** array on
  `jax.local_devices()[0]`; same `_eager_build`, same symmetry unfold, no collective,
  no cross-rank shape agreement, so each rank may load a different window and run
  ordinary `jax.jit` on it. Combining results then becomes an explicit, auditable
  step instead of something XLA's SPMD partitioner infers.
* `common.wfn_transforms.load_kpoint_fftbox_local(...)` + `process_local_mesh()` on
  top of it. `load_kpoint_fftbox` is now a thin wrapper over the former (values
  unchanged for its six existing single-process callers).

`process_local_mesh()` also fixes a latent bug: `load_kpoint_fftbox` built its 1×1
mesh from `jax.devices()[:1]` — i.e. **process 0's device on every rank**, a mesh no
rank but 0 can compute on. It is `jax.local_devices()[:1]` now: identical at P=1,
correct at P>1.

### X.2 — THE COMMUNICATION CONTRACT (the whole design, in four lines)

| stage | partition | collective |
|---|---|---|
| ρ(r) | (k, band-chunk), round-robin | **ONE psum, nx·ny·nz·8 B = 1.400 MB** |
| Poisson V_H(r) | **replicated by design** | **none** |
| ⟨mk\|V_H\|nk⟩ | k, round-robin; each k rank-local, no reduction | one all-gather (8.29 MB/rank) + a 144 B index gather |
| ψ load | per rank, only its own (k, band) windows | none — process-local |

**Poisson stays replicated, deliberately.** Two 3-D FFTs on a 1.4 MB array = 3.1e7
flop against the sweep's 1.2e12 — 3e-5 of the work. Sharding it would trade a free
duplicated computation for an all-to-all (a distributed 3-D FFT is two transposes)
and buy back 1.4 MB per rank. Every rank solves the same Poisson equation from the
same replicated ρ and gets bit-identical V_H(r) — which is *also* what makes the
k-partitioned matrix sweep trivially rank-invariant. Revisit above N_r ≈ 1e8.

**The assembly gather is accepted, not engineered away**, exactly as the directive
allows: 8.29 MB per rank in / 33.2 MB out at 12×12 / 120 bands (15 MB at 80 bands),
**once** per invocation, and it hands every rank the object both consumers want — a
file write on rank 0, a replicated operand for `sigma_dispatch`.

**Two departures from S's §7 sketch, both load-bearing.**

1. *ρ is k-partitioned first, band-chunked only when P > nk* (§7 proposed band-sharded
   G-flat ψ + `to_rchunk`). k is the axis the **dominant** sweep needs — matrix
   elements are 1.05e12 of the 1.2e12 flops and cannot be band-split without
   introducing a reduction — so k-partitioning *both* sweeps lets one process-local
   ψ-load idiom serve both and halves the I/O. Decisively, it makes the P=1 sweep the
   *identical sequence of operations* the serial code performed, which is what buys
   **exact** bit-parity instead of 1e-16 agreement (gate 1). Band chunking
   (`ceil(P/nk)` contiguous chunks of the occupied manifold) is layered on only in the
   P > nk regime where k alone stops filling the machine.
2. *Round-robin work assignment, not contiguous blocks.* nk need not divide P (the
   cohsex fixture has nk=9): contiguous blocking gives 3/3/3/0 at P=4, round-robin
   gives 3/2/2/2. Each gathered item carries its own global index in the payload, so
   the assignment may be any permutation and the result does not depend on the
   gather's process ordering.

No density math was forked: `valence_density_from_kpoint` gained an `nocc=None`
spelling ("every band you were handed contributes") for the band-chunked sweep and is
still the single quadrature for `compute_valence_density`, the chunked CLI, and the
distributed sweep.

### X.3 — THE CLI's MULTI-RANK GUARD BECAME THE COORDINATED-WRITE PATH

S's guard refused `srun -n P` outright. It now *distributes* and writes once: both k
loops (V_H matrix and kin_ion) are partitioned, each ends in one indexed gather, rank
0 alone opens the HDF5 file, and `sync_global_devices` keeps the peers alive until it
is closed. What is still fatal is the *other* multi-rank failure mode — a launcher
advertising P tasks while `jax.distributed` joined a world of 1, where every task
computes everything and every task believes it is rank 0. That is detected by
comparing `SLURM_NTASKS` against `jax.process_count()`, and raised before any file is
opened. (`LORRAX_KIN_ION_ALLOW_MULTIPROC` is gone; multi-rank is the supported mode.)

### X.4 — GATE 1: PARITY — **exact where it can be, 5e-16 relative where it cannot**

| # | comparison | `kin_ion` | `v_hartree` |
|---|---|---|---|
| P1 | cohsex fixture, P=1, vs S's `cohsex_kin_ion_STORED.h5` | **BIT-IDENTICAL** | **BIT-IDENTICAL** |
| P2 | **MoS₂ 12×12 / 120 b, P=1 @56 threads, vs S's `kin_ion_STORED120.h5`** | **BIT-IDENTICAL** | **BIT-IDENTICAL** |
| P3 | 12×12, P=4 vs P=1 | **BIT-IDENTICAL** | 2.132e-14 Ry (4.81e-16 rel; 2.90e-13 eV on the diagonal) |
| P4 | 12×12, P=8 vs P=1 | **BIT-IDENTICAL** | 2.134e-14 Ry (4.82e-16 rel) |
| P5 | 12×12, P=16 vs P=1 | **BIT-IDENTICAL** | 2.134e-14 Ry (4.82e-16 rel) |
| P6 | *control:* 12×12 P=1 @28 threads vs P=1 @56 threads | 3.554e-14 | 3.555e-14 |

Read P3–P6 together. `kin_ion` is bit-identical at **every** P because that sweep is
k-partitioned with no reduction anywhere — each k's matrix is computed by the same
code on one device. `v_hartree` moves by 2.1e-14 Ry, and only through ρ: the psum
sums 4/8/16 partials instead of 144 terms in k order. And **the control row is the
verdict**: running the *same serial code* on 28 threads instead of 56 moves the answer
by 3.55e-14 — **larger than anything the distribution does**. The P-dependence of this
kernel is strictly below the thread-count dependence XLA:CPU already had.

(P2 is also why the first 12×12 P=1 comparison looked like a 3.55e-14 "failure": S
generated the reference on a whole node, i.e. 56 XLA threads, while the scaling ladder
uses one socket per rank. Matching the thread count makes it bit-identical.)

**Unit suite** `tests/test_sanity_gates_jax.py`: **34 passed / 0 failed** (S left it at
25; +9 new — partition completeness and balance, the "P≤nk is the serial sweep"
bit-parity precondition, the `nocc=None` slicing identity, P=1 collective identities,
the process-local load vs the legacy wrapper, the QSGW rotation seam at U=1 and under
a band swap, the cache invalidator, and the rewritten broken-launch guard).

### X.5 — GATE 2: STRONG SCALING (MoS₂ 12×12, 80 Ry, nk=144, nb=120, N_r=36·36·135)

One rank = one Frontera socket (28 threads, `taskset`-pinned); 2 ranks/node. All four
runs serial on a dedicated 8-node allocation. Times are `timing` sections, i.e. they
exclude process startup and `jax.distributed` init.

| P | nodes×ranks | **total** | **build_V_H (THE KERNEL)** | ρ sweep | ⟨V_H⟩ matrix | kin_ion sweep | load_wfn (fixed) |
|---|---|---|---|---|---|---|---|
| 1 | 1×1 | 311.33 s | **159.42 s** | 26.82 | 132.59 | 144.87 | 6.76 |
| 4 | 2×2 | 94.57 s | **47.86 s** | 9.82 | 37.62 | 39.14 | 6.90 |
| 8 | 4×2 | 56.11 s | **28.75 s** | 6.88 | 19.71 | 19.76 | 6.33 |
| 16 | 8×2 | 31.72 s | **14.28 s** | 1.74 | 11.93 | 10.37 | 6.05 |

| P | kernel speedup | kernel eff. | compute-only speedup¹ | compute-only eff. | total speedup | total eff. |
|---|---|---|---|---|---|---|
| 4 | 3.33× | **83 %** | 3.49× | **87 %** | 3.29× | 82 % |
| 8 | 5.55× | 69 % | 6.23× | 78 % | 5.55× | 69 % |
| 16 | **11.17×** | **70 %** | **12.19×** | **76 %** | 9.81× | 61 % |

¹ `build_V_H + kin_ion sweep + its gather` = everything that is actually per-k work.

Per-phase: the **matrix-element sweep** (the dominant 1.05e12 flops) runs 11.1× at
P=16; the **kin_ion sweep** runs 14.0× (87 % of ideal); the **ρ sweep** runs 15.4×.
S's serial baseline was 1.79 s/k on a whole node; this is **0.171 s/k at P=16** for the
same two sweeps.

**Where the missing 30 % is, honestly.** It is *not* communication (§X.6). It is the
P-invariant floor: `load_wfn` (6.0–6.9 s — WFN header + the measured TRS/symmetry
check of §U) plus V_loc/V_NL setup, which is 2 % of the P=1 wall and **19 % of the
P=16 wall**. That is textbook Amdahl on a fixed serial section, and **a QSGW loop pays
it zero times per iteration** — the loader is already open. Strip it and the P=16
total efficiency is 76 %, not 61 %. The ρ-sweep column is also noticeably noisy
(6.88 s at P=8 vs 1.74 s at P=16 for only 2× less work): that phase is WFN-read bound
at this size and inherits Lustre variance.

One measured lesson worth keeping: the **first** collective of a Gloo/CPU run costs
**≈12 s** (XLA lowering of the shard_map module + communicator handshake) and every
later one costs milliseconds. Left inside the ρ sweep it read as 70 % of that phase
and destroyed the scaling signal. `compute_hartree_matrix` now fires the identical
reduction on a zero array first (`vh_collective_bootstrap`, 0.04–0.10 s once warm);
`runtime.nccl_warmup`'s generic psums do **not** cover it, because they lower a
different module.

### X.6 — GATE 3: **THE HLO COLLECTIVE TABLE** (rank-0 `--xla_dump_to`, P=4, 12×12/120 b)

**250 optimized-HLO modules. 10 collective instructions. All of them accounted for.**

| # | module | op | operand bytes | result bytes | replica_groups | what it is |
|---|---|---|---|---|---|---|
| 1 | `module_0063.jit__identity_fn` | all-gather | 32 | 128 | `[1,4]<=[4]` | `nccl_warmup` staging |
| 2 | `module_0067.jit_sum` | all-reduce | 8 | 8 | `[2,2]<=[4]` | `nccl_warmup` (x-axis) |
| 3 | `module_0067.jit_sum` | all-reduce | 8 | 8 | `[2,2]<=[2,2]T(1,0)` | `nccl_warmup` (y-axis) |
| 4 | `module_0071.jit__identity_fn` | all-gather | 16 | 64 | `[1,4]<=[4]` | `nccl_warmup` staging |
| 5 | `module_0075.jit_sum` | all-reduce | 8 | 8 | `[2,2]<=[2,2]T(1,0)` | `nccl_warmup` |
| 6 | `module_0077.jit_sum` | all-reduce | 8 | 8 | `[2,2]<=[4]` | `nccl_warmup` |
| **7** | **`module_0079.jit__body`** | **all-reduce `f64[36,36,135]`** | **1 399 680** | **1 399 680** | **`{{0,1,2,3}}`** | **THE ρ psum — `jit(_body)/shard_map/psum`** |
| **8** | **`module_0337.jit__identity_fn`** | **all-gather `c128[1,36,120,120]→[4,…]`** | **8 294 400** | **33 177 600** | **`[1,4]<=[4]`** | **the (k, nb, nb) assembly gather** |
| 9 | `module_0339.jit__identity_fn` | all-gather `s32[1,36]→[4,36]` | 144 | 576 | `[1,4]<=[4]` | the k-index payload |
| 10 | `module_0499.jit__identity_fn` | all-gather `u32[1]→u32[4]` | 4 | 16 | `[1,4]<=[4]` | `sync_global_devices` before exit |

Rows 1–6 are the deliberate pre-run warmup (≤ 32 B each). Row 10 is the write barrier
(4 B). **The kernel's entire traffic is rows 7–9: one 1.400 MB all-reduce, one 8.29 MB
all-gather and a 144 B index gather.** Rows 8–9 are compiled once and *executed twice*
in the CLI (V_H matrix, then kin_ion) — the CLI's second sweep reuses the same modules.

**What is NOT there, which is the point:**

* **no collective anywhere in the ρ sweep, the Poisson solve, the matrix-element sweep
  or the ψ loads** — 144 k-points, 3 sweeps, and not one per-k collective;
* **no `all-to-all`, no `reduce-scatter`, no `collective-permute`** in any of the 250
  modules;
* **nothing that scales as O(nk·N_r) or O(N_r) repeated** — the only N_r-sized
  collective in the program is the single ρ all-reduce, and 1.4 MB is exactly
  `36·36·135·8`, i.e. the density itself, once;
* the 8.29 MB gather is the (nk, nb, nb) result, once, as designed and documented.

Contract met. There is no excess communication left to eliminate.

### X.7 — GATE 4: QSGW-IN-LOOP READINESS

**(a) Cost per invocation at P=80.** From the measured ladder, the kernel is
k-partitioned and at P=16 each rank owns exactly 9 of 144 k. At P=80 the k partition
gives 2 k to 64 ranks and 1 k to 16, so the critical path is 2/9 of the P=16 work:

> **kernel ≈ 14.28 s × (2/9) ≈ 3.2 s per invocation at P=80**, and **≈1.6 s at P=144**
> (one k per rank, 100 % partition efficiency).

Communication per invocation is unchanged with P in *size* (1.4 MB + 8.29 MB/rank) and
is microseconds of wire time; the one-time ≈12 s Gloo bootstrap is paid on iteration 0
only. S's flop-based estimate was ≈0.5 s; the measured effective rate on Frontera CPU
(≈7.5 GFLOP/s/socket for this FFT-heavy mix) puts it at ≈3 s. **Either way it is
affordable against a multi-minute SC iteration** — the blocker S identified was
engineering, and it is gone.

**(b) Does the density build consume the CURRENT wavefunctions?** Verified and, where
it did not, closed:

* **The existing seam composes.** `sigma_dispatch`'s `hartree_basis_rotation=U_qp`
  takes the DFT-basis operator this kernel returns and rotates it into the QP basis
  (`O_QP = U†·O_DFT·U`). The kernel still builds ⟨mk|V_H|nk⟩ from the file's DFT
  orbitals, so it returns exactly the operand that seam expects — **unchanged and
  correct** for QSGW at fixed density, which is what the driver runs today.
* **The density side is now openable too.**
  `build_valence_density_distributed(..., psi_rotation=U)` with `U` of shape
  `(nk, nmix, nocc)` builds ρ from the *current* occupied orbitals
  `ψ^cur_n = Σ_m U[k,m,n] ψ^DFT_m`, applied on the **G-flat** coefficients before the
  box scatter (`nmix·nocc·ns·ngkmax` flops instead of `…·N_r` — 20× cheaper at 12×12,
  and the FFT box is only ever materialised at `nocc` bands). `compute_hartree_matrix`
  forwards it. Unit-gated: U = 1 reproduces the plain load **bit-identically**, and a
  band swap leaves ρ invariant. With a rotation supplied the band axis is not chunked
  (the mixing couples bands); the sweep stays k-partitioned, full rate for P ≤ nk.
  The two knobs are orthogonal by construction: `psi_rotation` changes *which density*
  V_H is generated by, `hartree_basis_rotation` changes *which basis* the operator is
  expressed in.
* **One trap, now documented and closed.** `sigma_dispatch._hartree_cache` memoises
  V_H across SC iterations — correct at fixed density, silently wrong the moment the
  density updates. `invalidate_hartree_cache()` exists and the requirement is stated
  at the cache definition.

**(c) `hartree_source=gspace` now has its cluster gate** (S's open item). Fixture,
P=4, same run dir and same `kin_ion.h5` as S's `stored` gate, only the key changed:

```
  hartree_source: requested=gspace → resolved=gspace
  V_H: rebuilding the exact FFT-grid matrix on the fly — DISTRIBUTED over the run's
       own mesh (rho: one psum; Poisson: replicated; <mk|V_H|nk>: k-partitioned + one gather)
  eqp0 vs S's stored run: n=496  max|delta| = 0.000000e+00   IDENTICAL
```

`stored` ≡ `gspace` on QP energies, distributed, end to end. The `gspace` array is
also now published to the driver as a genuinely **replicated global array**
(`replicate_to_mesh`) rather than the single-device array `jnp.asarray` produced —
indistinguishable at P=1, an operand-sharding mismatch at P>1.

### X.8 — 2D truncation, and cleanliness

`truncation_2d` still comes from the deck's `sys_dim` through `validate_operator_inputs`
(CLI) and `int(config.sys_dim) == 2` (driver) — S's de-hardwiring is reused verbatim,
not re-derived, and the same convention reaches V_loc and V_H inside one routine.
One implementation (`compute_hartree_matrix`), three consumers (generation CLI,
`hartree_source=gspace`, the QSGW loop), layout and communication contracts in the
docstrings, no duplicated density math.

### X.9 — Files, artifacts, jobs

**Changed (worktree only, NOT committed)** — `wt-G`, branch `gspace-vh-scaling` @ 8841d5e:
`src/gw/kin_ion_io.py` (the distribution layer + distributed CLI),
`src/file_io/wfn_loader.py` (`load_process_local`),
`src/common/wfn_transforms.py` (`load_kpoint_fftbox_local`, `process_local_mesh`,
`load_kpoint_fftbox` delegation), `src/psp/get_DFT_mtxels.py` (`report_devices`,
`valence_density_from_kpoint(nocc=None)`), `src/gw/sigma_dispatch.py` (mesh + replicated
publish + `invalidate_hartree_cache`), `src/file_io/kin_ion.py` (doc),
`tests/test_sanity_gates_jax.py` (+9 tests, 34 total).

**Artifacts** `/scratch2/08271/jackmc/lorrax_setup/wk_X/`: `run.sh` (holder runner with
socket pinning), `all_gates.sh`, `cmp_kin_ion.py`, `cmp_eqp.py`, `hlo_collectives.py`,
`hlo_p4/` (the full P=4 HLO dump, 250 optimized modules), `logs/` (`m12_P1`,
`m12_P1_t56`, `s_P4`, `s_P8`, `s_P16`, `g1_unit`, `g3_collectives`, `g4_gspace`),
`m12_P{1,4,8,16}.h5`, `m12_P1_t56.h5`, `cx_P{1,4}.h5`, `g_gspace/`.
Holder job **7875571** (8 nodes × 2, development).

**Reusable gotchas for the next agent.**
1. A `jax.devices()` call at module import time silently forecloses
   `jax.distributed.initialize()` for every importer. Grep for it before distributing
   any CLI.
2. `jax.devices()[0]` is **not** this rank's device. Any 1×1 mesh built for
   "process-local" work must use `jax.local_devices()`.
3. XLA:CPU results depend on the **thread count**: the same serial code at 28 vs 56
   threads differs by 3.6e-14 on this deck. Pin threads before declaring a bit-parity
   failure — that is what a 3.55e-14 "regression" turned out to be.
4. The first collective of a Gloo/CPU run costs ~12 s; bootstrap the exact reduction
   you are about to time, not a generic one.

## Z — recompile-hazard audit of the S–X change wave + the FFI-linalg PLAN API (wt-J, branch `linalg-api-audit`, base b1efa0f, 2026-07-26 — NOT committed)

**One line: the `_reshard_z` bug class is NOT present anywhere in the 4,420
new Python lines since 1ee52b2 — every new in-body jit already sits in a
module-level kernel cache — and the FFI-linalg call sites now take a
RESOLVED PLAN (`p = linalg.plan(op, mesh, backend=…, n=…)`, then `p(A)` /
`p.batched(A)`) that owns the resharding, the batch loop and the
per-backend output conventions, so the resolve→branch→reshard→loop→stack
boilerplate exists ONCE instead of five times.  Every gate green, and the
`JAX_LOG_COMPILES` per-kernel compile counts are IDENTICAL before and
after.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_Z/` (`log_compiles.py`,
`tally.py`, `compiles.sbatch`, `gates.sbatch`, `dist_tier.sbatch`,
`before_gspace_P{1,4}/`, `after_gspace_P{1,4}/`, `gwP{4,8}/`, `htP{4,8}/`,
`tier_{auto,distributed}_P4/`).  Jobs: `wkZcomp.7875683.out` (before),
`wkZgate.7875700.out` (all gates + after), `wkZdist.7875702.out` /
`.7875705.out` (the ζ `distributed` tier under the wk_V host lib).

### Z.1 — AUDIT 1: recompile hazards + jax idiom across S/T/U/V/W/X

Scope: `git diff --name-only 1ee52b2..b1efa0f` = 36 files / **5,021
insertions**, of which 29 `.py` files / 4,420 insertions.  Method: an AST
sweep for jit definitions/calls inside function bodies, **cross-referenced
against the diff's ADDED line ranges** so pre-existing code is not
re-litigated; then a read of every new hit plus targeted greps for the
other classes; then `JAX_LOG_COMPILES=1` on the fixture, before and after.

| class | new-code hits | verdict |
|---|---|---|
| **(a)** `@jax.jit` / `jax.jit(...)` in a function body → fresh callable ⇒ fresh key for JAX's identity-keyed trace/lower/compile caches | **8** (`ffi/scalapack/eigh.py:140`, `gw/kin_ion_io.py:296`, `isdf/core.py:1632, 1722, 2373, 2414`, +2 in the same factories) | **0 defects.** Every one is inside a module-level dict-cache factory (`_JIT_CACHE`, `_PSUM_KERNELS`, `_dist_factor_cache`, `_dist_solve_cache`, `_solve_cache`). `_reshard_z` is confirmed still hoisted into `_solve_cache` (T.4) with the explanatory comment intact. |
| **(b)** closure / identity captured per call; resolve-vs-call disagreement | **1 real defect** | **FIXED** — Z.2 |
| **(c)** traced-vs-static ⇒ per-shape recompiles | **1** | **FLAGGED + measured** — Z.3 |
| **(d)** `donate_argnums` on a still-referenced buffer | 3 new donations | **0 defects.** `_block`'s `zeta_acc`, `_solve_one_q_and_update`'s `zeta_acc`, `_reshard_z`'s `z` are all rebound at the call site — and all three run under the fused r-chunk jit, where donation is inert anyway. |
| **(e)** missing `out_shardings` on a jit producing a big array | 0 | `_factor_c_q_distributed_rank_truncate._fn` carries an explicit `out_shardings`; every other new jit takes its layout from a `shard_map` `out_specs` or from a donated accumulator. |
| **(f)** `jnp.zeros/ones` unsharded in a multi-device path | 0 | `jnp.zeros_like(Z_q)` inherits its operand's layout; `kin_ion_io`'s `rho_local = jnp.zeros(nx,ny,nz)` is deliberately process-local (JAX's default device is `local_devices()[0]`; the whole ρ sweep is single-device land by design and documented as such). |
| **(g)** shape-unstable branches | 2 | one IS (c); the other is the eager q-block remainder in `_distributed_pinv_apply` — bounded at TWO compiled shapes, now documented in place. |
| **"safe linalg"** — raw `jnp.linalg.*` on ill-conditioned physics objects without the rcond/ridge/truncation guards | 0 | The only new `linalg` on a physics object is the `distributed` tier's eigh, truncated at `rcond·λ_max` with V.4's pad-aware `λ_max`. `density_symmetry_check.py:513`'s `np.linalg.inv` is an INTEGER rotation matrix, guarded by a 1e-8 integrality check that returns `None`. No new `jnp.linalg.solve/inv/cholesky` on a CCT/χ/W-class object. |

Repo-wide (outside the diff) 80 in-body jit definitions have no inline
cache guard.  Every one is a `make_*`/`build_*` factory or a once-per-run
driver (`compute_wfns_fi`, `h_transform`, `prepare_coarse`, `solve_bse`),
**except** `common/wfn_transforms.py`'s ten, which ARE guarded — through
`_cached_jit(name, key, build)` rather than an inline dict, so a naive
scanner reports them as unguarded.  One genuine unguarded case,
`wfn_transforms.load_centroids_band_chunked._reshard_all`, is once per
centroid load per channel: **named, benign, left alone.**

### Z.2 — the one real defect: the ζ `distributed` tier resolved one backend and called another

`isdf/core._factor_c_q_distributed_rank_truncate` called
`backend_module('scalapack').batched_distributed_eigh(...)` — a HARD-CODED
backend — while its own admission gate `_resolve_zeta_gather` approves the
route with `resolve_backend('eigh', 'distributed', mesh, n=n_rmu)`, and
`'distributed'` means **cuSOLVERMp on a CUDA mesh**
(`resolve._DISTRIBUTED_DEFAULT`).  So on a GPU mesh the tier passes its own
guard and then dies inside `ffi.scalapack`'s host-only check — exactly the
broken-promise class V.2 added the resolve-time refusal to eliminate.
Invisible on CPU, which is why the V gates could not see it.

Fixed by the migration: the tier now says
`linalg_plan('eigh', mesh_xy, backend='distributed', n=n_pad).batched(C_q)`,
so the platform default is chosen in ONE place.  On CPU the resolution and
the call are byte-identical to before, pinned by a new test cell
(`test_plan_batched_matches_the_backend_call_cpu`) that asserts bit-equality
of `plan.batched` against the raw wrapper the tier used to call.

### Z.3 — the one measured recompile finding: per-k `ngk` in X's new G-space V_H sweep

`gw/kin_ion_io.compute_hartree_matrix`'s k-partitioned ⟨mk|V_H|nk⟩ sweep
calls `psp.get_DFT_mtxels.compute_local_V_k` with `Gk_crys` at **each
k-point's own `ngk`**, so `_compute_local_V_k_jit` compiles once per
DISTINCT `ngk`:

    fixture WFNsmall.h5, IBZ ngk = [749, 754, 754, 780]  ->  3 distinct
    MEASURED (P=1, hartree_source=gspace):  jit(_compute_local_V_k_jit)  3 compiles

Bounded by the **IBZ** k-count, not by nk (the full-BZ unfold inherits IBZ
values) — so ~19 for an MoS₂ 12×12 IBZ.  A handful of compiles of a small
kernel, not a storm.  **Flagged, not fixed.**  `compute_local_V_k` already
carries a `g_mask` hook that would collapse it to one compile by padding to
`ngkmax`, and no call site uses it: the pad columns contribute exact zeros,
but they **change the summation order** of the ⟨m|V|n⟩ reduction, and this
matrix element sits inside the ~500 eV H₀ cancellation that X exists to get
right.  Recorded as a comment at the call site with that reasoning, so the
next reader does not "fix" it.

### Z.4 — compile-count evidence (`JAX_LOG_COMPILES=1`, cohsex fixture, `hartree_source=gspace`)

`wk_Z/log_compiles.py` sets the env before any jax import and `runpy`s the
CLI; `wk_Z/tally.py` counts `Finished XLA compilation of …`.

| | before (b1efa0f) | after (this branch) |
|---|---|---|
| P=1, total compiles | **423** | **423** |
| P=4, total over 4 ranks | **1877** (469/rank) | **1877** |
| per-kernel diff | — | **per-kernel compile counts IDENTICAL** (both P) |

So the refactor is exactly compile-neutral, which is the point: it moved
call-site plumbing, not kernels.  Nothing in the new code dominates the
tally — the largest NAMED kernel is `jit(_compute_local_V_k_jit)` at 3
(Z.3), then `jit(_kernel)` (the fused r-chunk body) at 2, its documented
"full chunks + remainder" pair.  The top of the list is JAX's own
primitive-level jits (`multiply` 44, `broadcast_in_dim` 44,
`convert_element_type` 32 at P=1), which are per-shape and not addressable
from this code.

### Z.5 — AUDIT 2: the FFI-linalg call-site API

**What the survey found.**  Regions with >2 nearby linalg-FFI calls:
`isdf/core` (charge factor + the new distributed tier + transverse LU +
three back-solve branches), `bandstructure/bse_setup` (fH_q eigh),
`bse/vq_interp.prepare_coarse` (coarse C_q eigh), `gw/w_isdf` (the fused
cuBLASMp W-solve), `common/*bench*` (four scripts).  Every one had grown the
same five lines around the call that mattered:

    resolved = resolve_backend(op, requested, mesh, n=…)   # 1. resolve
    if resolved == NATIVE: …                               # 2. branch
    grid = NamedSharding(mesh, P('x','y'))                 # 3. re-derive the contract
    A = jax.device_put(A, grid)                            # 4. reshard (3 spellings)
    pairs = [call(A[i]) …]; lam = jnp.stack(…)             # 5. batch by hand

with the **in/out sharding contract implicit** (a `P('x','y')` literal at
each site, nowhere stated as the API), the reshard spelled three different
ways (`device_put` in vq_interp, `with_sharding_constraint` in bse_setup and
isdf/core, nothing at all in the ζ tier), the batch loop written out wherever
the backend lacks a batched entry point, and the cuSOLVERMp
conj-transpose known only INSIDE `dispatch_eigh` — so anything reaching
`backend_module` directly (the ζ tier, four benches) had to remember it
independently.  Error handling was the one consistent part: `resolve_backend`
raises at resolve time with the failed guard named, and every site calls it.

**The design: a resolved plan.**  New `src/ffi/linalg/plan.py`:

```python
p = linalg.plan("eigh", mesh, backend=cfg.eigh_backend, n=rank)   # ONCE
log(p.describe())
if p.is_native:
    lam, R = jnp.linalg.eigh(A_batch)    # caller owns the fused fast path
else:
    lam, R = p(A_tile)                   # or p.batched(A_stack)
```

`LinalgPlan` (frozen dataclass) carries `op / requested / backend`,
`is_native`, `module`, `describe()`, and the LAYOUT CONTRACT AS DATA:
`in_sharding` = `P('x','y')`, `batch_in_sharding` = `P(None,'x','y')`, both
`None` on a native plan.  `__call__` moves operands into the contract via
the single `ensure_sharding` helper (traced ⇒ `with_sharding_constraint`;
already there (mesh+spec) ⇒ untouched; else `device_put`), calls the
backend, and normalises the output convention.  `batched` uses the backend's
own batched entry point when it has one (ScaLAPACK eigh: one descriptor, one
workspace for the whole stack) and otherwise loops + stacks — so **the call
site no longer encodes which backends are batched**.  A one-row-per-`(op,
backend)` `_IMPL` table is the only thing a new backend adds.

Three deliberate non-goals, stated in the module docstring:
1. **It does not change resolution.**  `plan()` is `resolve_backend()` plus
   storage; route strings and `auto` policy untouched (pinned).
2. **`is_native` means the caller owns it.**  Native cholesky/solve_lu are
   the channel-policy routes in `isdf/core` (replicated dense factor / 2-D
   blocked `shard_map` / per-q ridged solve), not one call — `plan(...)`
   raises there naming what owns them rather than pretending.  `eigh` is the
   one op whose native path IS one call, so the plan runs it, preserving
   `dispatch_eigh` exactly.
3. It does not absorb the ζ-fit route strings or the cuSOLVERMp cholesky
   HANDLE; those stay where they are pinned.

`dispatch_eigh` survives as a five-line shim over a plan, so nothing outside
had to move at once.

**Migrated (4 clusters):**

| site | change |
|---|---|
| `bandstructure/bse_setup.compute_wfns_fi` | ONE plan for the whole q loop; `dispatch_eigh` per q → `eigh_plan(...)` — which also stops re-running the whole guard ladder on every q. The native branch's `dispatch_eigh(…, "off")` inside `_q_batch` becomes the `jnp.linalg.eigh` it always was, removing a facade call from inside a jit. |
| `bse/vq_interp.prepare_coarse` | the hand-rolled per-q `device_put` + `dispatch_eigh` + two `jnp.stack`s → `eigh_plan.batched(C_herm(sl))`; two dead `NamedSharding` bindings dropped |
| `isdf/core._factor_c_q_distributed_rank_truncate` | `backend_module('scalapack').batched_distributed_eigh` → `plan(backend='distributed').batched` — the **Z.2 fix** |
| `common/eigh_benchmark` (both modes) | plan, and the plan HOISTED OUT of the timing loop — it was resolving inside the median, against scorecard L §6's own advice |

**Plus one de-duplication inside `isdf/core`.**  The three distributed
back-solve branches (`cusolvermp_cholesky` potrs, `cusolvermp_lu` /
`scalapack_lu` getrf+getrs, the new `distributed` C⁺Z GEMM) differ ONLY in
the call; the NRHS pad → solve → `_reshard_zeta_mu_X_r_Y_to_mu_XY` → trim
frame around them was written out three times.  Now
`_distributed_backsolve(Z_q, mesh, run)`, one copy — which matters more
than de-duplication, because FFI-adjacent resharding is precisely where this
code base has lost the most time (J.9's silent NaNs from a Z re-layout,
T.4's per-r-chunk recompile of one, V.4's deleted `_reshard_z`).
`solve_zeta` lost 45 lines and gained no branch.

**Left, with a migration note** (new "Call sites: migrated, and not" table in
`docs/dev/linalg_ffi.md`): the `isdf/core` cholesky/LU **backend calls**
(pinned route strings + the cuSOLVERMp handle whose block-cyclic geometry
`solve_zeta` rebuilds — migrate WITH the route strings, never separately);
`gw/w_isdf`'s cuBLASMp fused W-solve (not a selectable op, no native twin to
dispatch against); and `common/slate_*_test.py`,
`slate_vs_cusolvermp_bench`, `eigh_block_sweep`, `cusolvermp_*_test.py`,
which exercise backend INTERNALS (raw buffer layouts, `block_size`,
`compute_evecs`) that the plan normalises away — testing through the
abstraction would stop them testing the thing.

### Z.6 — GATES (all PASS)

| gate | result |
|---|---|
| `pytest test_ffi_linalg_contract.py -k plan` (4 new cells, no host FFI) | **3 passed, 1 skipped** (the host-FFI cell) |
| same, **WITH** the wk_V host FFI lib (`LORRAX_FFI_HOST_SO=build_host_V`, full MKL/SLATE/IMPI env) | **4 passed** — including the `plan.batched` ↔ raw-`batched_distributed_eigh` bit-equality pin on the Z.2 migration, against real `pzheevd` |
| `pytest -k 'plan or resolver or cap or tier or route'` (contract + zeta invariance) | **5 passed, 3 skipped, 0 failed** |
| `pytest test_zeta_mesh_invariance.py -k 'cap or tier or route'` (route/tier pins) | **2 passed** |
| GW cohsex fixture **P=4** vs `eqp_ref.dat` @1e-3 | **PASS, max\|Δ\| = 1.0e-06 eV**, 0/1888 over tol; route still `replicated_rank_truncate` / tier `replicated` |
| GW cohsex fixture **P=8** vs ref @1e-3 | **PASS, max\|Δ\| = 1.0e-06 eV**, 0/1888 |
| htransform **P=4** vs `bs_groundtruth_meshless.dat` | **PASS, max\|ΔE\| = 0.000e+00 — BIT-EXACT** |
| htransform **P=8** vs ground truth | **PASS, max\|ΔE\| = 0.000e+00 — BIT-EXACT** |
| ζ `distributed_zeta_solve=distributed` **P=4** (wk_V host lib, real `pzheevd`) vs ref | **PASS, 1.0e-06 eV**, 0/1888 (jobs 7875702 and 7875705, the latter on the final tree); route `distributed_rank_truncate`, telemetry `[zeta rank_truncate/distributed] n_pad=60 rcond=1.0e-08 n_keep/q=[60 …]` — identical to V.5 |
| `distributed` vs `auto` eqp, directly | **max\|Δ\| = 0.00e+00** (both jobs) — and `auto` still prints `replicated_rank_truncate` / tier `replicated`, i.e. the existing routes are byte-identical |
| compile counts before/after | **IDENTICAL** (Z.4) |
| `py_compile` on all 11 touched/new files | PASS |

New cells in `tests/test_ffi_linalg_contract.py`:
`test_plan_resolution_is_identical_to_resolve_backend` (every
`(op, requested)` pair through both spellings — **including the ones that
RAISE, asserting the same exception type**),
`test_plan_native_contract` (no sharding contract on a native plan; eigh
runs and `batched` batches; cholesky/solve_lu refuse naming `isdf/core`),
`test_plan_describe_and_module_are_honest`, and
`test_plan_batched_matches_the_backend_call_cpu` (bit-equality against the
raw `batched_distributed_eigh` — the pin on the Z.2 migration).

### Z.7 — named, not done

* The `id(mesh)` cache key used by six kernel caches (`isdf/core` ×5,
  `kin_ion_io._psum_kernel`) LOOKS like a use-after-GC hazard and is not:
  the cached VALUE closes over the mesh through its `shard_map`, so the Mesh
  cannot be collected while its entry lives and its id cannot be recycled.
  Failure mode is a redundant compile on an equal-but-distinct Mesh, never a
  stale hit.  Now documented at `kin_ion_io.py`, with the rule: use
  `ffi.slate.batched._mesh_key` wherever the cached value does NOT retain
  the mesh.  (`sigma_dispatch._hartree_cache` is the one place it doesn't —
  and there the cached value is mesh-independent, so it is still safe.)
* `_distributed_pinv_apply` slices its q-blocks EAGERLY while the sibling
  `per_q` tier slices inside the jit off a traced q: two compiled shapes
  instead of one.  Deliberate (the q-batch comes from a byte budget, so a
  uniform block shape would mean padding nq and factoring q-blocks that do
  not exist).  Noted in place.
* `plan` covers `eigh` end to end; `cholesky` / `solve_lu` plans resolve and
  expose `.module` / `.batched` but no call site takes one yet.  That is the
  next migration and it must move **with** the ζ route strings, not before
  them.
* `_IMPL` records that SLATE's per-tile `potrf` returns a HANDLE, so
  `plan.batched('cholesky','slate')` refuses with a message pointing at
  `isdf/core.factor_c_q`'s `slate_cholesky` branch rather than failing
  inside `jnp.stack`.

---

## Y — the standing HLO/timing/memory probe, and the measured method-tradeoff matrix (wk_Y, 2026-07-26; NO source edits)

> **⚠ CLAIM-DECAY (AL, 2026-07-27): em1-scoped collective walls.** Every multi-node JAX CPU-collective WALL TIME in this section was measured while Gloo was bound to Frontera's 1 GbE management NIC (`em1`, 129.114.x.x) — the campaign-wide default before `runtime.pin_gloo_interface()` landed (AK.10/AL). Byte counts, HLO collective inventories and residency ledgers remain valid; the wall-time consequences of collective-bound stages re-price on ib0 (measured 3.3x whole-pipeline at 785c/P=16, and see AL for 606c/P=80). Single-node numbers are unaffected.

**One line: HLO tracing is now a two-command standing practice rather than a
one-off, and its first 13-cell run says (i) the `distributed` ζ tier beats
`replicated` on wall at *every* point measured, (ii) `per_q` is 12–40× slower
*and* does not actually do the per-q gather its docstring promises, (iii)
`hartree_source=gspace` costs +3.5 % of a run for exactly three collectives
totalling 16 MB, and (iv) the biggest collective in a ζ-fit is not in the ζ tier
at all.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_Y/` — `probe.sbatch`,
`submit_probe.sh`, `analyze_probe.py`, `runs/<tag>/` (13 run dirs, each with its
generated `gw.in`, `probe_meta.txt`, `probe.<jid>.out`, `mem_node0.<jid>.log`,
`hlo/` and `probe_summary.txt`), and `runs/MATRIX.csv` (every cell as one row —
regenerate with `python3 analyze_probe.py runs/*/ --csv`).
Jobs **7875685–7875695** (11 cells) and **7875703/7875704** (the μ=1998 follow-up).
Source: main checkout `/work2/08271/jackmc/frontera/lorrax` @ **b1efa0f**, unmodified.

### Y.0 — the harness (this is the deliverable the directive actually asked for)

    ./submit_probe.sh <tag> <P> <wall_min> KEY=VAL ...      # one cell
    python3 analyze_probe.py runs/<tag> [...] [--csv]       # one summary block

`probe.sbatch` builds a run dir under `wk_Y/runs/<tag>`, **generates** `gw.in`
from its parameters (so a cell is reproducible from its `probe_meta.txt` alone),
links the MoS₂ 12×12 deck, and runs it with all four instrument families on:

| knob | what it turns on |
|---|---|
| `PROBE_HLO=1` (default) | rank-0 `XLA_FLAGS=--xla_dump_to … --xla_dump_hlo_as_text --xla_debug_buffer_assignment_show_max=400` + `JAX_LOG_COMPILES` — **K.0's verified flag set, unchanged** (this jaxlib F-aborts on an unknown flag; do not add to it) |
| always | `LORRAX_RCHUNK_DEBUG=1` + `LORRAX_ZETA_RANK_LOG=1` — T.1's per-chunk RSS / `live_arrays` / `d_zq`/`d_solve` split, and the n_keep telemetry |
| always | node-0 `free` sampler at 15 s |
| `PROBE_MAXCHUNK=N` | `LORRAX_MAX_RCHUNKS` — the r-chunk loop stops after N. **This is what makes the directive's "kill it after 10 min" mode work properly**: a clean python break, so `write_g_flat` still runs and the stage table still prints |
| `PROBE_EXITZETA=1` | `LORRAX_EXIT_AFTER_ZETA` — skip V_q/W/Σ |
| `PROBE_KILLMIN=N` | hard stop: a `timeout` wrapper **around srun, not `scancel`**, so the batch script survives to run `sacct` and the analyser, and a partial fit still yields the dump and every chunk timing printed so far |
| `PROBE_KEEP=lean` (default) | post-run cleanup, added after the first matrix cost **67 GB of Lustre**: deletes the dump's `.ll`/`.mlir`/`.o` files (99 % of its bytes and useless to any analysis — 5 GB → 5 MB per cell) and the throwaway `tmp/` ζ tensors + `sigma_mnk.h5` (a `MAXCHUNK`-truncated run's ζ is meaningless by construction). 67 GB → **235 MB** for the whole matrix. `PROBE_KEEP=all` opts out |

Parameters: `CENTROIDS NBAND NCOND TIER HARTREE RCHUNK MAXCHUNK EXITZETA RESTART
KILLMIN HLO KINION SRC`. Both files carry their own documentation, including the
list of things that bite (XLA flag F-abort, the P>1 compile-cache deadlock, the
`GLOO_SOCKET_IFNAME=ib0` stall, the `FI_PROVIDER_PATH` requirement for the
ScaLAPACK host FFI, and `build_host_V` vs `build_host`).

`analyze_probe.py` is plain python3 (no jax/h5py — runs on a login node), reads
only what is on disk, and prints six blocks: **CELL** (parameters + resolved
mesh/route/tier), **STAGES**, **CADENCE** (per-chunk wall + the z_q/back-solve
split), **MEMORY** (planner HWM vs measured, RSS slope, `live_arrays`),
**COLLECTIVES** (by op **and by module** — see Y.6), **ANOMALIES**. `--csv`
emits one matrix row per run dir. Self-tested against wk_X's P=4 dump: it
reproduces X.6's table exactly (10 collectives, the 1 399 680 B ρ psum and the
33 177 600 B assembly gather).

Two harness bugs found and fixed while building it, both worth knowing:
the in-job analyser used to append its summary to the `.out` it parses (doubling
every line on a re-parse — it now writes `probe_summary.txt` and the parser stops
at its own marker); and the benign `cuInit 303` CUDA-plugin traceback that every
CPU rank prints had to be whitelisted or it swamps the anomaly list.

### Y.1 — the matrix (13 cells, MoS₂ 12×12, nband=160, r_chunk 6480, 3 of 27–28 chunks)

See `SESSION_REPORT_2026-07-25.md` § "Method tradeoffs (measured)" for the full
tables. Headlines:

| μ | P (mesh) | tier | chunk s | z_q s | back-solve s | ζ-tier coll bytes | coll instrs |
|---|---|---|---|---|---|---|---|
| 276 | 16 (4×4) | replicated | 77.6 | 38.4 | 38.6 | 1270 MB | 43 |
| 276 | 16 (4×4) | **per_q** | **510.4** | 39.6 | **470.4** | 1270 MB × **144 execs** | 42 |
| 276 | 16 (4×4) | **distributed** | 59.4 | 39.3 | **19.6** | 1182 MB | **30** |
| 276 | 64 (8×8) | replicated | 34.1 | 18.6 | 14.9 | 704 MB | 43 |
| 276 | 64 (8×8) | **per_q** | **624.4** | 18.9 | **604.9** | 704 MB × **144** | 42 |
| 276 | 64 (8×8) | **distributed** | 31.8 | 18.8 | **12.3** | 658 MB | **30** |
| 606 | 64 (8×8) | replicated | 59.4 | 21.3 | 37.6 | 2487 MB | 43 |
| 606 | 64 (8×8) | **distributed** | 45.2 | 21.7 | **23.0** | **1442 MB (−42 %)** | **30** |

`_reshard_z` is absent from every `distributed` dump (V.6 reproduced on the
production deck at two P and two μ). `_solve_all_at_once`'s gather is
`nq·μ_pad²·16` to the byte at every size.

**The distributed tier beats replicated on wall everywhere here (1.22–1.97× on the
back-solve), which V.3's library ladder did not predict.** The reconciliation is
V.4's own design note: the explicit `C⁺` makes the per-r-chunk back-solve one GEMM
instead of two and deletes `_reshard_z`, and the r-chunk loop pays that 27–28×
while `pzheevd` is paid once. V.3 measured the factorisation in isolation; this
measures the object the production loop actually spends its time in.

`hartree_source` at P=16, full pipeline: Σ stage 317.2 s (`stored`) / 318.5 s
(`isdf`, ≈free — V_q[0] is already built for W) / 337.9 s (`gspace`, **+20.7 s**,
attributed by its own sub-timers to ρ 13.40 s of which the psum is 10.85 s, the
assembly gather 0.20 s, and ~11 s of un-timed matrix sweep in the stage self).
`stored` and `gspace` give an **identical** implied Vxc, [−24.262, −4.455] eV.

### Y.2 — **`per_q` does not do the per-q gather** (the finding that justifies the practice)

`jit(_solve_one_q_and_update)`'s optimized HLO contains the *same* two all-gathers
as `_solve_all_at_once`, terminating in `c128[144,288,288]` — the **whole**
`(nq, μ_pad, μ_pad)` stack — and only then a `dynamic_slice` for the single q.
The buffer assignment names it in the peak stack-trace breakdown:

    _solve_one_q_and_update  Total bytes used: 790 088 272 (753.49 MiB)
      ├── Z_col                                       268 738 560  (34.0 %)
      ├── zeta_acc                                    268 738 560  (34.0 %)
      ├── .../dynamic_slice                           238 878 720  (30.2 %)   <-- the FULL-stack gather
    _solve_all_at_once       Total bytes used: 1 240 012 816 (1.15 GiB)
      ├── .../dot_general                             515 082 240  (53.0 %)
      ├── Z_col                                       268 738 560  (27.7 %)
      ├── .../slice                                   175 509 504  (18.1 %)

So at 276c the tier buys a **1.57× module-peak reduction (1.240 → 0.790 GB), not
144×**, and because the module runs once per q it moves 144 × 238.9 MB = **34.4 GB
per r-chunk** where `replicated` moves 238.9 MB. That is the 12–40× wall, explained.
XLA did not sink the q-axis `dynamic_slice` through the μ-axis `all-gather` even
though the two commute — a missed optimisation, not a code bug, but the code's
"one `(μ,μ)` tile at a time" contract does not survive SPMD partitioning on
XLA:CPU.

**Settled at production μ — jobs 7875703 / 7875704** (1998 centroids, μ_pad = 2048,
P = 64, 8×8, `MAXCHUNK=1`), peak stack-trace breakdown of each solve module:

    _solve_one_q_and_update   Total bytes used: 11 979 015 312 (11.16 GiB)
      ├── .../dynamic_slice                        10 871 635 968  (90.8 %)  <-- ITS gather
      ├── Z_col                                       476 577 792  ( 4.0 %)
      ├── zeta_acc                                    476 577 792  ( 4.0 %)
      └── L_q_sharded                                 150 994 944  ( 1.3 %)
    _solve_all_at_once        Total bytes used: 29 162 981 392 (27.16 GiB)
      ├── .../slice                                 9 662 519 808  (48.4 %)  <-- the factor gather
      ├── .../complex                               9 197 577 216  (46.1 %)  <-- a logical-extent copy
      ├── Z_col                                       476 577 792  ( 2.4 %)
      └── .../dot_general                             464 942 592  ( 2.3 %)

Every number is a closed form, to the byte:
`nq·μ_pad·(μ_pad + μ_pad/P_x)·16` = 144·2048·2304·16 = 10 871 635 968 (per_q);
`nq·μ_pad²·16` = 144·2048²·16 = 9 663 676 416 ≈ the replicated gather; and
`nq·μ_log²·16` = 144·1998²·16 = 9 197 577 216 = T.3's logical-extent copy exactly.
**So T.3's arena model is confirmed component by component, and T.5's headline —
"9.36 GB → 0.065 GB, 144×" — is REFUTED**: `per_q`'s own gather buffer (10.87 GB)
is *larger* than the replicated gather it exists to avoid (9.66 GB). The tier's
real benefit is **1.67× on module peak (19.95 → 11.98 GB) and 2.43× on total
allocated (27.16 → 11.16 GiB)**, and it comes entirely from `per_q` never forming
T.3's two logical-extent copies — not from gathering less.

**Planning consequence: T.7's residency table needs revising.** Its "c2406 with
per_q → back-solve arena **0.28 GB**" becomes, by the measured law at μ_pad = 2448
and P_x = 12, `144·2448·(2448+204)·16` ≈ **14.96 GB/rank** — so that config's
19.8 GB/rank total is really ≈34.5 GB/rank and the headroom T.7 reported is not
there. `distributed` (which genuinely gathers no `(μ,μ)` object — V.4, and 30-vs-43
instructions in Y.1) is the tier that delivers what `per_q` promised, wherever the
mesh is square.

The tier remains **numerically exact** — T.6's bit-identity gate is untouched by
any of this. What was wrong is only the memory and latency claim.

**One production-μ cadence point, from the cell that died** (`d_1998_P64_rep`,
1998c, P=64): its single completed r-chunk was **267.1 s (z_q 23.0 + back-solve
242.6 s)** — the replicated back-solve grows 14.9 → 242.6 s for μ 276 → 1998 at
fixed P, i.e. ≈μ^2.2, against a μ² arithmetic expectation. The `per_q` cell at the
same size had **still not completed one r-chunk** more than 20 minutes into its
chunk loop, so the 12–40× penalty measured at 276c does not shrink with μ — which
is what the Y.2 mechanism predicts, since the 144 serial executions do not get
cheaper relative to one batched call.

### Y.3 — the biggest collective in a ζ-fit is not in the ζ tier

`F_tensor_write`'s unsharded G-flat gather, `c128[nq, μ_pad, ngkmax]`, measured:

| μ (μ_pad) | bytes | law |
|---|---|---|
| 276 (288) | 5 708 537 856 | `nq·μ_pad·ngkmax·16` |
| 276 (320) | 6 342 819 840 | same, P=64 |
| **606 (640)** | **12 685 639 680** | same |
| **1998 (2048)** | **40 594 046 976 — MEASURED, as the OOM that killed job 7875704** | same |
| 2406 (2448) | 48.54 GB (extrapolated) | same |

**P-independent** — adding nodes does not shrink it.

**This stopped being a projection.** `d_1998_P64_rep` (job 7875704, 1998
centroids, P=64, 32 nodes) completed r-chunk 1 in 266 s and then died `rc=1`:

    jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED:
        Out of memory allocating 40 594 046 976 bytes
      File ".../jax/experimental/multihost_utils.py", line 108,
        in _handle_array_process_allgather
    *** LORRAX FAIL-FAST: rank 56/64 died with ... Exiting rc=1 without teardown

and `nq·μ_pad·ngkmax·16 = 144·2048·8603·16 = 40 594 046 976` **to the byte**.
**The planner named `F_tensor_write` as the binder and sized it at 19.97 GB/dev —
2.03× too small**: it identified the right term and under-predicted the number
that killed the job by a factor of two.

The margin is quantified by the probe's own instrumentation: the chunk-1
`[rchunk_dbg]` line reports **`live=60.685 GB`/rank** at this size (8.3 GB at
276c), so the 40.59 GB request lands on 60.7 GB of already-live arrays —
**101 GB against the 96 GB/rank a 2-rank Frontera node affords, a ~5 % miss.**
So at 1998 centroids the ceiling is *this collective*, not anything in the ζ
solve, and **no tier choice avoids it** (per_q and distributed change the
back-solve; the writer is outside both). At the flagship's P=144 the live term
shards 2.25× further while the writer only falls to 39.96 GB (μ_pad=2016) — it
fits, but this term is what eats the headroom.
J.10 / V.7's named wall now has an exact coefficient, a three-point measured curve
and a death certificate. Fix unchanged: mpi4py + `HDF5_MPI=ON` h5py so
`PHDF5_HOST` becomes reachable — and this is now the **single highest-priority**
item for the centroid ladder, ahead of any further ζ-tier work.

(Side note, an unplanned validation: O's fail-fast hardening did exactly its job —
named the dying rank, refused teardown, and put the failure in the job's exit
code instead of letting peers hang in a collective.)

Two loader `process_allgather`s are **P-LINEAR and unmodelled by the ISDF memory
model** (K.1 #3/#8's family, now with a scaling law from two points):
`s32[P·nk, 36,36,135]` = `P·nk·n_rtot·4` → 1.61 GB at P=16, **6.45 GB at P=64**,
projected **14.5 GB at P=144**; and `c128[P·nk, ngkmax]` = `P·nk·ngkmax·16` →
0.32 / 1.27 / **2.85 GB**. Together **17.4 GB/rank projected at P=144** — the
dominant memory term exactly where the campaign is heading.

### Y.4 — K.2's rematerialisation is a RECTANGULAR-MESH artifact

**Zero `[SPMD] Involuntary full rematerialization` warnings in any of the 13 run
dirs** (4×4 and 8×8, μ = 276/606/1998, all three tiers, all three V_H sources). K.2 measured exactly 1/rank (80 at P=80) on the **8×10** mesh, for
`{devices=[1,10,1,1,8]<=[8,10]T(1,0)} → {devices=[1,8,1,1,10]<=[80]}`. When
p_x = p_y that pair is the same sharding and the transposition is free. Combined
with the `distributed` tier's square-mesh requirement (V.7): **prefer 144 ranks
(12×12) over 80 (8×10) for two independent reasons.** Caveat: 4×4/8×8 vs 8×10 is
square-vs-rectangular *and* smaller; a 10×10 point would isolate it.

### Y.5 — planner accuracy, and the memory cure still holding

`HWM estimate × 2 ranks` vs measured node peak: **1.54×** (conservative) at P=16,
**1.11×** at P=64, **0.98× (under-predicts)** at 606c/P=64 — and at 1998c/P=64 the
planner's own binder, `F_tensor_write`, is under-sized **2.03×** against the
allocation that OOM'd the job (19.97 GB/dev modelled vs 40.59 GB requested, Y.3).
**The trend is monotone toward under-prediction as μ grows, and it crosses at
about 600 centroids.** The planner is no longer a safety margin at production μ. Binder is
`A_centroid_load` in all 13 cells rather than `C_fit_one_rchunk` — the smaller
r_chunk moved the binding stage. T.2's glibc cure holds at both P and both μ: RSS
slope ≤ 0 in every multi-chunk cell (−0.08…−0.09 GB/rank/chunk = chunk 1's
one-off transient being returned) and `jax.live_arrays()` byte-constant. T.4 holds
too: 195–208 rank-0 compiles for a ζ-only run, 475 for the full pipeline, with no
per-chunk multiple.

Side result: running with a deliberately truncated ζ (`MAXCHUNK=2` of 27) made
O's `_warn_on_unphysical_h0` guard fire correctly on the `isdf` V_H route
(implied Vxc [−162.85, +578.33] eV, 11 434/11 520 flagged) while staying silent on
`stored`/`gspace` — an unplanned but welcome live test of the silent-failure
hardening.

### Y.6 — the caveat every future reader of these tables needs

The HLO gives bytes **per instruction per execution**; it does **not** give
execution counts. `replicated` runs its solve module once per r-chunk, `per_q`
runs it `nq` = 144 times, and a naive byte total makes them look identical (that
is exactly how the per_q defect could hide). `analyze_probe.py` therefore prints
a **by-module** breakdown alongside the by-op one, and the module's loop structure
in the source is what supplies the multiplier. Do not read a single "total
collective bytes" number as a method comparison.

### Y.7 — cost of the practice

13 cells, 8–32 nodes each, 7–15 min wall apiece (~75 node-hours in total), each
with its own rank-0 HLO dump. The two μ=1998 cells needed only `MAXCHUNK=1` —
their whole purpose was one buffer-assignment file, and they produced it ~13 min
in, which is exactly the "start a run with a trace and kill it after 10 min" mode
the directive asked for. The recommended
standing cadence is one `MAXCHUNK=1 EXITZETA=1 KILLMIN=12` cell per significant
change — startup + compile + one r-chunk is already enough for the collective
table, the buffer assignment, the tier/route banners and the planner comparison;
only the RSS slope needs ≥3 chunks.

---

## AA — the three communication offenders Y's traces exposed, closed (wt-J, branch `gather-fixes`, base 823d7ca, 2026-07-26 — NOT committed)

> **⚠ CLAIM-DECAY (AL, 2026-07-27): em1-scoped collective walls.** Every multi-node JAX CPU-collective WALL TIME in this section was measured while Gloo was bound to Frontera's 1 GbE management NIC (`em1`, 129.114.x.x) — the campaign-wide default before `runtime.pin_gloo_interface()` landed (AK.10/AL). Byte counts, HLO collective inventories and residency ledgers remain valid; the wall-time consequences of collective-bound stages re-price on ib0 (measured 3.3x whole-pipeline at 785c/P=16, and see AL for 606c/P=80). Single-node numbers are unaffected.

**One line: the two "P-LINEAR loader `process_allgather`s" turned out not to be
loader code at all — they are `jax.device_put(numpy, <multi-process
NamedSharding>)` firing JAX's own hidden `multihost_utils.assert_equal`, a debug
assertion that all-gathers `P ×` the array; removing it deletes 1.93 GB/rank at
P=16, 7.73 GB/rank at P=64 and 17.4 GB/rank projected at P=144, for zero
numerical change.  `per_q`'s defeated gather is cured by moving the q-slice
INSIDE a `shard_map`: 238.9 MB → 1.66 MB per execution (**144×, exactly nq**),
and the tier goes from 12.8× the replicated back-solve wall to **1.02×**.  And
the planner's `F_tensor_write` term was sizing the wrong tensor — the object that
actually killed the 1998-centroid run is `nq·μ_pad·ngkmax·16`, not `nq·μ²·16`.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AA/` (`gate.sbatch`,
`gate.7875720.out`, `gwP4/ gwP8/ gwP4perq/ gwP4dist/ htP4/ htP8/`) and three new
probe cells in Y's harness, `wk_Y/runs/AA_P16_rep_after`, `AA_P16_per_after`,
`AA_P64_606_rep_after` (jobs 7875721 / 7875722 / 7875725).  `wk_Y/runs/MATRIX.csv`
regenerated to include them.  **No new instrumentation was written** — the whole
measurement is Y's `submit_probe.sh` + `analyze_probe.py` with `SRC=` pointed at
the worktree, which is what the standing practice is for.

### AA.1 — ROOT CAUSE: the two loader all-gathers are `jax.device_put`'s hidden `assert_equal`

Y.3 named them by shape — `s32[P·nk, nx,ny,nz]` and `c128[P·nk, ngkmax]`, both
in `jit(_identity_fn)` modules whose mesh is `#sdy.mesh<["processes"=P,
"local_devices"=1]>`.  That mesh is constructed in exactly one place in JAX,
`multihost_utils._handle_array_process_allgather`, and `grep process_allgather
src/` finds **nothing** in the loader.  The caller is JAX itself:

```python
# jax/_src/dispatch.py::_device_put_sharding_impl, the `not s_is_fully_addressable`
# branch, for a numpy array or an UNCOMMITTED jax.Array:
if xla_bridge.process_count() == len(s._internal_device_list.process_indices):
    multihost_utils.assert_equal(x, fail_message="... is not the same on each process.")
```

`assert_equal` is `process_allgather(x, tiled=True)`.  So **every**
`jax.device_put(host_table, NamedSharding(mesh, spec))` in a multi-process run
silently pays a P-linear all-gather — on device *and* on the host, since
`process_allgather` ends in `np.asarray(out.addressable_data(0))` — purely to
assert that the table each rank computed from the same file is the same table.
Eight such calls sit in `WfnLoader` (`box_index_dev`, the five
`_ensure_phdf5_static` tables, the union-read `offsets`/`counts`), one in
`PsiGStore`, two in `isdf_fitting`, one in `htransform`.  Two of them are big:

| table | law | P=16 | P=64 | P=144 (proj.) |
|---|---|---|---|---|
| `box_index` (sparse-G→FFT-box index, s32) | `P·nk·n_rtot·4` | 1.612 GB | 6.450 GB | 14.5 GB |
| `phase_per_full` (τ-phase row, c128) | `P·nk·ngkmax·16` | 0.317 GB | 1.269 GB | 2.85 GB |
| the six small tables (offsets/counts/ibz/sym/trs/U) | mixed; `counts` is **O(P²)** | 0.003 GB | 0.039 GB | 0.19 GB |
| **total** (P=16 / P=64 measured, P=144 projected) | | **1.932 GB** | **7.758 GB** | **17.5 GB** |

**Fix — `common/collectives.py::device_put_process_local(host_array, sharding)`.**
Each process slices out only the shard(s) its own devices own — the whole array
for a replicated spec, its own hyperslab for a sharded one, via
`sharding.addressable_devices_indices_map(shape)` — and declares them with
`jax.make_array_from_single_device_arrays`.  Zero collectives.  It is the idiom
the codebase already had in three private copies (`WfnLoader._assemble_process_local`,
`kin_ion_io.replicate_to_mesh`, the slab-io process-local writers), now written
once with the JAX behaviour it works around documented at the call site.
`LORRAX_CHECK_REPLICA=1` restores the assertion for a debugging run.

The precondition (bit-identical input on every rank) is the same one
`device_put` was spending 17 GB/rank to check, and every call site satisfies it
by construction: the tables are pure functions of the WFN file and the mesh
shape.  `counts_global` is the interesting one — it is genuinely *sharded*
`P(('x','y'), None)`, so the replacement is literally the per-rank hyperslab the
FFI reader wants, and its old gather was **O(P²)**.

**MEASURED (probe cells `a_P16_rep` → `AA_P16_rep_after`, 276c, P=16, identical
parameters, 3 r-chunks):**

| | before (`a_P16_rep`) | after (`AA_P16_rep_after`) |
|---|---|---|
| collective instructions | 43 | **32** |
| total collective bytes | 11.380 GB | **9.448 GB** (−17.0 %) |
| `jit(_identity_fn)` module total | 18 instrs / 8 447 058 752 B | **7 instrs / 6 514 756 160 B** |
| the `s32[2304,36,36,135]` gather | 1 612 431 360 B | **gone** |
| the `c128[2304,8603]` gather | 317 140 992 B | **gone** |
| biggest collective in the run | the loader index gather | now `F_tensor_write`'s ζ writer gather (5 708 537 856 B) |

Delta `8 447 058 752 − 6 514 756 160 = 1 932 302 592 B` — **to the byte** the sum
of the table above.  The 6.51 GB that remains in `_identity_fn` is real data
movement (the ζ-writer gather + the ψ reshards), not assertion traffic.
Steady-state r-chunk wall is unchanged (78.5 s after vs 79.7 s before on chunks
2–3), as it must be: this removes memory and bytes, not arithmetic.

**And the same thing at P=64** (`c_606_P64_rep` → `AA_P64_606_rep_after`, 606c):
`jit(_identity_fn)` 18 instrs / 20 889 990 400 B → **7 instrs / 13 132 468 480 B**,
a delta of **7 757 521 920 B = 7.76 GB/rank** — again to the byte the P=64 column
of the table (6.450 + 1.269 + 0.039).  Run-wide: 43 → **32** collective
instructions, 24.398 → **16.640 GB** (−31.8 %).  Extrapolating the two measured
points to the flagship's 12×12 mesh gives **17.5 GB/rank at P=144** returned.
Wall is again a null result, as intended: 59.39 → **59.15 s/chunk**, back-solve
37.60 → **37.11 s**.

**Honest note on the node-level peak.** It does **not** move at 606c/P=64: 43
GB/node before and after.  That is expected once you look at *when* each transient
happens — the loader gather is paid during `A_centroid_load`, the ζ-writer gather
(12.69 GB/rank = 25.4 GB/node, AA.3) is paid at `write_g_flat`, and the latter is
the larger of the two, so it, not the loader, sets `free`'s high-water mark.  What
AA.1 buys at 606c is **headroom underneath the peak**, and the peak *itself* stops
being reachable-by-accident: the two no longer stack in a run where they overlap
(they do at higher μ, where the loader term is unchanged at 7.76 GB/rank but the
writer term grows as μ).  The bytes-on-the-wire result is unambiguous and is the
deliverable; the `free`-sampler peak is a coarser instrument (15 s cadence, and it
also sees the page cache separately).

Two `_identity_fn` gathers on the `processes` mesh survive and are left alone:
`f64[64,4]` and `f64[64]` (2 KB total, pre-existing, present identically in Y's
dumps) and the `u32[P]` scalar of `sync_global_devices`, which is the barrier
itself.

### AA.2 — `per_q` now does the per-q gather (Y.2's headline, closed)

Y.2 proved the tier's contract did not survive SPMD partitioning: the traced-`q`
`dynamic_slice` + `with_sharding_constraint(replicated)` compiled to the WHOLE
`(nq, μ, μ)` all-gather followed by the slice, so the module moved *more* than
`replicated` and ran `nq` times.  XLA will not sink a q-axis slice through a
μ-axis all-gather.

**Fix: do the selection inside a `shard_map` (`isdf/core.py::_per_q_block`),
where the gather is structurally per-tile.**  The rank's own `(nq, μ/Px, μ/Py)`
block is sliced locally, then two `lax.all_gather(..., tiled=True)` calls — over
`'x'` on axis 1, then `'y'` on axis 2 — rebuild exactly the `(1, μ, μ)` replicated
tile the batched kernel would have seen.  There is nothing left to hoist.  The
three back-solve bodies are the same functions `_sharded_cho_solve_batch` calls,
at batch 1, on identical shapes and identical operand values.

Optimized HLO of `jit(_solve_one_q_and_update)`, 276c / P=16 / μ_pad=288:

| | before (`a_P16_per`) | after (`AA_P16_per_after`) |
|---|---|---|
| collectives in the module | `c128[144,72,288]` 47 775 744 B + `c128[144,288,288]` 191 102 976 B | `c128[1,288,72]` **331 776 B** + `c128[1,288,288]` **1 327 104 B** |
| **gathered bytes / execution** | **238 878 720** | **1 658 880** — **144.0×, exactly nq** |
| module `Total bytes used` | 790 088 272 (753.49 MiB) | **555 435 664 (529.70 MiB)** — 1.42× |
| gather traffic per r-chunk (×144 execs) | 34.4 GB | **239 MB** (≈ `replicated`'s single 238.9 MB) |

and the wall follows exactly:

| 276c, P=16, post-compile chunks only | back-solve [s] | chunk [s] |
|---|---|---|
| `replicated` before (`a_P16_rep`, ch. 2–3) | 38.85 | 79.71 |
| `per_q` **before** (`a_P16_per`, ch. 2 — it never reached 3) | **470.4** | **510.9** |
| `replicated` after (`AA_P16_rep_after`, ch. 2–3) | 36.14 | 78.48 |
| **`per_q` after** (`AA_P16_per_after`, ch. 2–3) | **36.88** | **78.75** |

(Chunk 1 is excluded on both sides — it carries the r-chunk kernel compile, and it
landed differently on the two allocations: 71.9 s fit in Y's run against 108–111 s
in these.  Chunks 2+ are the steady state and the `replicated` rows are the
control that says the machine and the code agree to 1.5 %.)

**12.8× on the back-solve, 6.5× on the r-chunk, and the tier is now within 2 % of
`replicated` on wall while keeping the `nq`× smaller live gather.**  Projected to
the production point Y.2 settled (μ_pad = 2048, nq = 144, 8×8): per-execution
gather `μ²·16·(1 + 1/p_y)` = 67.11 MB + 8.39 MB = **75.5 MB**, against the
**10.87 GB** Y measured — the same 144×, and the number the directive asked the
buffer assignment to show.

Consequences for the guidance Y wrote:
* **Y.1's "per_q is 12–40× slower" and Y.2's "its gather is larger than
  replicated's" are now historical**, true of 823d7ca and earlier only.  The
  `HISTORY` block above `_per_q_block` records the defeated form verbatim so it
  is not re-introduced.
* **T.7's residency table is restored to something close to its original claim.**
  Y.2 revised c2406/P=144's back-solve arena from 0.28 GB to 14.96 GB using the
  measured `nq·μ_pad·(μ_pad + μ_pad/P_x)·16` law.  That law is gone; the arena is
  now `μ_pad²·16·(1 + 1/p_y)` = **0.104 GB** at μ_pad = 2448, p_y = 12.  (Still
  not T.5's 0.28 GB figure, which counted differently, but the same order.)
* `distributed` still wins on wall where the mesh is square (19.6 s at P=16 /
  276c vs 36–37 s for the other two), so the "prefer `distributed` on a square
  mesh" advice is unchanged.  What changed is the fallback: on a **rectangular**
  mesh `per_q` is now a free memory win instead of a 12–40× penalty.

### AA.3 — planner: `F_tensor_write` was sizing the wrong tensor (2.10× under)

Two different objects cross the `H5PY_ALLGATHER` SlabIO seam, and the model only
carried one of them:

    V_qmunu / W0_qmunu   (n_q_ibz, μ, μ)        μ²      <- modelled
    the G-flat ζ tensor  (n_q_disk, μ, ngkmax)  μ·ngkmax <- NOT modelled

Whenever `ngkmax > μ` — true at every centroid count the campaign has run — the
ζ write is the binder.  Y's death certificate (`runs/d_1998_P64_rep`, job
7875704) is the exact arithmetic:

    measured fatal allocation        = 40 594 046 976 B
    nq·μ_pad·ngkmax·16 = 144·2048·8603·16 = 40 594 046 976   ✓ to the byte
    old planner F_t = 2·nq·μ_pad²·16      = 19 327 352 832   → 2.10× too small

`F_t` is now `2 · max(V_tensor, gflat_tensor)`, the `2 ×` being the gathered
device buffer plus `process_allgather`'s closing host `np.asarray` copy.  At the
1998c/P=64 point that is **81.19 GB/dev** — which is the honest number: it does
not fit, and the planner now says so instead of reporting 19.97 GB and letting
the job die.  Verified live at 276c/P=16: `F_tensor_write` moved from absent-from-
the-top-4 to **14.09 GB/dev**, and `2·144·288·8603·16 + E_base` reproduces it.

Second planner change: **`loader_tables` is now a persistent term** —
`nk·n_rtot·4 + nk·ngkmax·16`, the two REPLICATED loader tables AA.1 leaves
resident (121 MB at MoS2 12×12).  Small, but **P-independent**, so it belongs in
the floor rather than nowhere.  The `p_min` search short-circuits when that
P-independent term alone busts the budget, instead of stepping to 2²⁰.

Neither change touches `band_chunk` / `r_chunk` / `q_chunk` selection — verified
identical (`band_chunk=160 r_chunk=6480 (27 chunks) q_chunk=144`) before and
after at 276c/P=16.  `persistent` 0.77 → 0.89 GB is exactly the new term.

**On Y.5's "0.98× under-prediction at 606c/P=64" (anomaly 5).**  The live
`analyze_probe` figure for that cell is **0.48×** (20.7 GB/node predicted vs 43
measured), and both halves of the gap are now accounted for and fixed: the
unmodelled `assert_equal` gather is **15.5 GB/node** at P=64 (AA.1) and the
unmodelled ζ-writer gather is **25.4 GB/node** at 606c (AA.3) — a
`MAXCHUNK`-truncated probe still runs `write_g_flat`, so that cell paid it.

**Re-measured, cell `AA_P64_606_rep_after` (job 7875725), same parameters as
`c_606_P64_rep`:**

| 606c, P=64, 8×8 | before (`c_606_P64_rep`) | after (`AA_P64_606_rep_after`) |
|---|---|---|
| planner HWM / binder | 10.33 GB/dev, `A_centroid_load` | **31.16 GB/dev, `F_tensor_write`** |
| planner × 2 ranks | 20.7 GB/node | 62.3 GB/node |
| measured node peak | 43 GB | 43 GB (unchanged — set by the ζ writer, see AA.1) |
| **planner / measured** | **0.48× (UNDER-predicts)** | **1.45× (conservative)** |
| collective instrs / bytes | 43 / 24.398 GB | **32 / 16.640 GB** |
| `band_chunk` / `r_chunk` / `q_chunk` | 192 / 6464 (28) / 144 | **192 / 6464 (28) / 144 — unchanged** |
| chunk wall / back-solve | 59.39 / 37.60 s | 59.15 / 37.11 s (control) |
| RSS slope, `live_arrays` | −0.093 GB/rank/chunk, 16.978 GB const | −0.091, 16.978 GB const (T.2 holds) |

The cure is on the estimate side, and it is the right side: the model now covers
the term that actually sets this cell's peak.  **Every point in the matrix is now
conservative** — 1.54× at 276c/P=16, 1.45× at 606c/P=64 — where Y measured the
sequence drifting 1.54 → 1.11 → 0.98 → (2.03× under at 1998c).  Chunk selection
and wall are unchanged, which is what "honest, not cautious" has to mean.

### AA.4 — GATES (all PASS; `wk_AA/gate.7875720.out`, 4 nodes, 8 ranks)

| gate | result |
|---|---|
| `tests/test_wfn_loader_eager.py` (loader parity, 1 proc) | **15 passed, 1 skipped** |
| GW cohsex fixture **P=4**, `auto`, vs `eqp_ref.dat` @1e-3 | **PASS, max\|Δ\| = 1.0e-06 eV**, 0/1888 over tol |
| GW cohsex fixture **P=8**, `auto`, vs ref @1e-3 | **PASS, max\|Δ\| = 1.0e-06 eV**, 0/1888 |
| **P=8 vs P=4 directly** (mesh invariance) | **max\|Δ\| = 0.00e+00 — BIT-IDENTICAL** |
| GW fixture **P=4, `distributed_zeta_solve=per_q`** vs ref | **PASS, 1.0e-06 eV** |
| **`per_q` vs `auto` at P=4** — the gate on `_per_q_block` | **max\|Δ\| = 0.00e+00 — BIT-IDENTICAL** |
| htransform bandstructure **P=4** vs meshless ground truth | **max\|ΔE\| = 0.000e+00, bit-exact=True** |
| htransform bandstructure **P=8** vs meshless ground truth | **max\|ΔE\| = 0.000e+00, bit-exact=True** |
| `py_compile` on all eight touched files | OK |

The `distributed_zeta_solve=distributed` regression cell **refused at resolve
time** in this environment — `liblorrax_ffi_host.so` is not on the default search
path from a plain sbatch (it needs `LORRAX_FFI_HOST_SO=$WORK/lorrax_ffi_unified/
build_host_V`, V.7c).  That is the designed clean refusal, not a failure, and the
tier is unmodified by this workstream; V.5's gates remain its authority.

Comparisons use `multidev_compare.py` (the eqp `.dat` files are `k-point N:`
blocks, not `loadtxt`-able — the gate script's own inline comparator choked on
that and the numbers above were produced by re-running the established tool on
the same files).

### AA.5 — files touched (worktree only, NOT committed)

`src/common/collectives.py` (+`device_put_process_local`), `src/file_io/wfn_loader.py`
(8 call sites), `src/common/psi_G_store.py` (1), `src/gw/isdf_fitting.py` (2 + the
tier banner now prints the real per-execution tile size and says `×nq executions`),
`src/bandstructure/htransform.py` (the `G` accumulator was
`device_put(jnp.zeros(...), sharding)` — same hidden assertion, `P·rank²·16`),
`src/isdf/core.py` (`_per_q_block` + docstrings), `src/gw/gflat_memory_model.py`
(F term + `loader_tables` + `p_min` guard), `src/gw/gw_config.py` (key help text).

**Named, deliberately NOT done:** `centroid/kmeans_isdf.py::shard` and
`centroid/pivoted_cholesky.py:418` have the same `device_put` antipattern, but
they run in the centroid-*generation* driver, not the GW path, and nothing in
this workstream's gate ladder exercises them.  One-line change each when someone
is next in those files.  Likewise the `PHDF5_HOST` writer fix that would delete
the `F_tensor_write` gather outright is a separate workstream (mpi4py +
`HDF5_MPI=ON` h5py); AA only makes the planner tell the truth about it.

## AB — `PHDF5_HOST` made REACHABLE: mpi4py + `HDF5_MPI=ON` h5py, and the writer all-gather deleted (wk_AB, 2026-07-26; ENVIRONMENT workstream, **zero source edits**)

**One line: the environmental fix J.2/J.10/V.7/Y.3/AA.3 have all been pointing at
is done and gated — `mpi4py` 4.1.2 + `h5py` 3.16.0 built `HDF5_MPI=ON` against
Frontera's Intel-MPI parallel HDF5 1.14.6 now live in a separate prefix, `gw_config`
routes `slab_io=auto` to `PHDF5_HOST`, and at 606 centroids / P=16 the
12.05 GB `F_tensor_write` all-gather is GONE from the optimized HLO, the ζ file is
bit-identical, the fixture eqp gate is unchanged at max|Δ| = 1.0e-6 eV, and the
shared production venv was never touched while two 72-node flagships ran against it.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AB/` — `build_overlay.sbatch`,
`mpiio_smoke.py`, `routing_smoke.sbatch`, `probe_ab.sbatch`, `h5_bitcmp.py`,
**`CUTOVER.md`** (activation lines / fold-in criteria / three-level rollback),
`runs/c606P16_{base,ov}/`, and the job logs `build.7875713.out`,
`smoke.7875724.out`, `probe606.7875719.out`, `zcmp606.7875735.out`.
The overlay itself: `/work2/08271/jackmc/frontera/lorrax_env_mpi_overlay/`
(`pkgs/` + `MANIFEST.txt` with sha256s, `wheels/` with both built wheels,
`site/` = the 30 MB two-package overlay).
Source: main checkout `/work2/08271/jackmc/frontera/lorrax` @ **823d7ca**, unmodified.

### AB.0 — the build (reproducible; `CUTOVER.md` §5 is the executable version)

    sbatch wk_AB/build_overlay.sbatch      # 1 node, qsmall, ~6 min, job 7875713

Five things make it work and none of them are obvious:

1. **The venv has no `pip` and no `setuptools`** — it was created by `uv`. The build
   runs pip *out of its own wheel* (`python pip-26.1.2-py3-none-any.whl/pip`, a wheel
   is a zip with `pip/__main__.py`), so nothing has to be installed to make the
   installer exist.
2. **Compute nodes have no outbound network; login nodes have no apptainer.** So the
   7 sdists/wheels are `curl`'d on the login node with sha256 verification
   (`pkgs/MANIFEST.txt`) and every install is `--no-index --find-links --no-build-isolation`.
3. **`CC=mpicc` fixes the link as well as the compile** — setuptools' `customize_compiler`
   rewrites `LDSHARED` when `CC` is overridden, so no separate `LDSHARED=` is needed.
4. **`I_MPI_CC=gcc`** — the container has gcc 12.2/g++ but no icc; same deviation
   `config/frontera/build_ffi_host.sh` already documents.
5. **Build on node-local `/tmp`, publish at the end** (Lustre under the concurrent
   72-node flagships), then one `cp -a` + `mv` swap.

Build-time deps (setuptools / wheel / Cython / pkgconfig) go to a throwaway
`/tmp/.../tools` on `PYTHONPATH` and **never enter the shipped `site/`**, which
contains exactly two packages. `numpy` 2.4.3 comes from the base venv.

Verified in-job: `h5py.get_config().mpi = True`, `h5py.version.hdf5_version = 1.14.6`,
and `ldd` on `h5py/*.so` resolves `libhdf5.so.310` → the phdf5 install and
`libmpi.so.12` → Intel MPI 2020.4. Then a **4-rank collective MPI-IO write+readback
on Lustre under `srun --mpi=pmi2`** (`mpiio_smoke.py`): `MPIIO_SMOKE_OK`, 4/4 ranks exact.

**The base venv was not modified.** A size+path manifest of all 11 928 non-`__pycache__`
files is identical before and after; the only delta in the whole workstream was 12
auto-generated `packaging/*.pyc` (CPython's normal atomic byte-compilation, which
the script now suppresses with `PYTHONDONTWRITEBYTECODE=1`). Jobs **7875551** and
**7875552** were executing against that venv for the entire session.

### AB.1 — the router fires, and the writer is numerically inert (job 7875724, P=4 fixture)

Two `cohsex_debug` runs, same tree, differing only in the writer:

| cell | `PYTHONPATH` | input | resolved backend | eqp vs `eqp_ref.dat` |
|---|---|---|---|---|
| base | `SRC` | `use_ffi_io=false`, `slab_io=h5py_allgather` | `H5PY_ALLGATHER` | **PASS**, 1888 values, max\|Δ\| = 1.00e-6 eV, max rel 5.29e-9 |
| overlay | `OVERLAY/site:SRC` | `use_ffi_io=true`, `slab_io=auto` | **`PHDF5_HOST`** | **PASS**, 1888 values, **identical** max\|Δ\| = 1.00e-6 eV |

The router's own line, verbatim from the overlay log — this is the print that has
never fired on Frontera before:

    [config] use_ffi_io=true on CPU backend; phdf5 FFI is CUDA-only.  Routing
    SlabIO through PHDF5_HOST (mpi4py + h5py-parallel) — same per-rank
    collective MPI-IO write semantics.

`tmp/zeta_q.h5` bit-compare (`h5_bitcmp.py`, sha256 per dataset): **47 of 48
datasets BIT-IDENTICAL**, including `zeta_q_G` itself. The 48th is
`isdf_header/fit_provenance`, a JSON blob that embeds the absolute `wfn_file`
path — `|S523` vs `|S526`, the 3-byte difference between `smoke_base` and
`smoke_overlay` in the run-dir name. Nothing else differs; 0 attributes differ.

### AB.2 — **THE GATE**: 606 centroids, P=16, the writer all-gather is gone (job 7875719)

Both cells in ONE 8-node allocation, same nodes, same compile storm, `MAXCHUNK=3
EXITZETA=1 TIER=replicated`, differing only in `slab_io`
(`wk_AB/runs/c606P16_{base,ov}/probe_summary.txt`):

| | base — `H5PY_ALLGATHER` | overlay — `PHDF5_HOST` |
|---|---|---|
| collective instructions | 39 | **38** |
| total collective RESULT bytes | 21.374 GB | **9.322 GB (−56.4 %)** |
| all-gather bytes | 17.419 GB | **5.368 GB (−12.051 GB, exactly the writer)** |
| `jit__identity_fn` module | 18 instrs / 15.686 GB | **17 instrs / 3.634 GB** |
| **biggest collective** | **12 051 357 696 B `c128[144,608,8603]` all-gather** | **1 612 431 360 B — the loader index gather** |
| wall | 796 s | **544 s** |
| r-chunk loop (code's own timer) | 403.8–406.9 s | 403.6–409.9 s (**control: unchanged**) |
| planner (unchanged @823d7ca) | HWM 33.89 GB/dev, binder `A_centroid_load`, `F_tensor_write` 2.04 | identical |
| RSS slope / `live_arrays` | −0.081 GB/rank/chunk, 15.733 GB const | −0.077, 15.733 GB const (T.2 holds) |

The deleted instruction, verbatim from the base dump and **absent from every one of
the overlay's 194 modules**:

    %all-gather = c128[144,608,8603]{2,0,1} all-gather(%copy), channel_id=1,
                  replica_groups=[1,16]<=[16], dimensions={1}, use_global_device_ids=true

`nq·μ_pad·ngkmax·16 = 144·608·8603·16 = 12 051 357 696` **to the byte**, and
`μ_pad = 608 = ⌈606/16⌉·16` — the P=16 instance of Y.3's law, whose P=64 instance
is the 12.686 GB (`μ_pad=640`) figure in `c_606_P64_rep`. Per-rank ζ shard is
`n_mu_local = 38` rows, i.e. **12.051 GB → 0.753 GB/rank**, and it is written as 16
hyperslabs which by construction are not XLA collectives at all.

This also exercises the padded-tail clip, which the P=4 fixture cannot: μ_pad = 608
against a logical `n_rmu = 606`, so rank 15's 38-row shard runs off the end of
`valid_shape` and `_clip_shard_to_valid` must drop 2 of its rows. **Job 7875735**
re-runs the pair at `MAXCHUNK=1` for exactly that compare:

    zeta_q_G   (144, 606, 8603)  complex128  BIT-IDENTICAL  sha256[:16]=641b5d104a85fc0f
    H5_BITCMP_OK        (0 mismatches, 0 attributes differing)

— the full production-shape ζ tensor, written as 16 clipped MPI-IO hyperslabs,
byte-for-byte equal to the rank-0 serial allgather write. (The 7875719 compare was
lost to a harness bug of mine: `h5_bitcmp.py` ran without `LD_LIBRARY_PATH`, so the
overlay's h5py died `ImportError: libimf.so`, and the lean cleanup then deleted
`tmp/`. `probe_ab.sbatch` now sets the runtime and refuses to clean up unless the
compare returns 0.)

### AB.3 — the writer's WALL-CLOCK cost, measured for the first time

The chunk-loop timer is identical between cells, so the 252 s wall difference is
entirely outside the fit loop — and node-0's 15 s `free` sampler shows exactly where:

    7875719 base    (allgather)  16:10:06 → 16:12:21  used_GB 10→16→19→22→25→28→31→40→44 → exit
    7875719 overlay (MPI-IO)     same phase           used_GB 15, 15, 11, 11 → exit  (no ramp)
    7875735 base    (allgather)  16:27:36 → 16:29:37  used_GB 10→16→19→23→26→29→32→44→44 → exit
    7875735 overlay (MPI-IO)     same phase           used_GB 15 flat → exit          (no ramp)

**≈120–135 s of monotone allocate-and-write on both allgather runs, ≈0 s on both
MPI-IO runs** (7875735's wall: 418 s base vs 236 s overlay, same ordering). The ramp's amplitude also confirms J.2's parenthetical "×2 (device+host)":
`_slab_io_allgather._to_host` holds the `process_allgather` result *and* its closing
`np.asarray` host copy simultaneously, so the node (2 ranks) climbs ~34 GB against a
12.05 GB logical tensor.

Honest caveat on the remaining ~117 s: the base cell ran first, with a cold page
cache for `WFN.h5`, and startup was 256 s vs the overlay's 139 s. That part is **not
attributed to the writer**. The 135 s ramp is.

### AB.4 — the writer ceiling, recomputed honestly (and why "4650 → 50k" needs two footnotes)

Two different tensors cross the SlabIO seam (AA.3's decomposition), and the two
famous ceiling numbers are for *different ones*:

| object | per-rank bytes, `H5PY_ALLGATHER` | per-rank bytes, `PHDF5_HOST` |
|---|---|---|
| **B** the G-flat ζ tensor `zeta_q_G` `(n_q_disk, μ_pad, ngkmax)` | `nq·μ_pad·ngkmax·16` ×(1 or 2) | **/P** |
| **A** the restart tensors `V_qmunu`/`W0_qmunu` `(n_q, μ, μ)` | `2·nq·μ²·16` | **/P** |

μ at which the writer alone exceeds the per-rank envelope E, MoS₂ 12×12
(`nq=144`, `ngkmax=8603`):

| configuration | E = 72.2 GB/dev | E = 92 GB/dev |
|---|---|---|
| allgather, **1** buffer (Y.3's measured fatal allocation) | **3 643** (B binds) | **4 641** (B binds) — this is J.2's "~4650" |
| allgather, **2** buffers (device+host, AA.3's corrected planner, and what AB.3 measured) | **1 821** (B) | **2 321** (B) |
| `PHDF5_HOST`, P=64 | 44 783 (**A** binds; B is 233 k) | 50 553 |
| `PHDF5_HOST`, P=80 | **50 069** (A) | 56 519 |
| `PHDF5_HOST`, P=144 | 67 175 (A) | 75 829 |

**So J.10's "μ ≈ 3,960 → μ ≈ 50,100" survives — but only as a statement about
object A at P=80, and the "before" side of it is optimistic**: the thing that
actually binds under the allgather writer is object B, whose ceiling is
**3 643 (1-buffer) or 1 821 (2-buffer)** at the same envelope, not 3 960. The
environment fix is therefore worth *more* than J.10 claimed, and J.2's 4 650 and
J.10's 3 960 are the same wall measured with two different conventions on two
different tensors. Under `PHDF5_HOST` object B stops being a wall in any
convention (it needs μ > 233 000 at P=64).

**Effective ceiling after this fix is NOT 50 k.** It is set by the next two walls in
J.10's table, both untouched by AB: the replicated `rank_truncate` eigh, which is a
**TIME** wall at μ ≈ 4 000 (O(nq·μ³), zero P-scaling, ~5.5 h at 4 k), and then
`B_cct_chol` at μ ≈ 16 700. AB removes a memory wall that sat *below* both of them
and hands the ladder to workstream V/L's distributed `pzheevd`.

Measured confirmations of the law, all to the byte: 12 051 357 696 (606c/P=16, AB.2),
12 685 639 680 (606c/P=64, Y.3), 40 594 046 976 (1998c/P=64 — the OOM that killed
job 7875704).

### AB.5 — the two P-linear loader all-gathers are UNCHANGED by the overlay (as expected)

At 606c/P=16 the overlay dump still carries both of Y.3's loader gathers, byte-identical
to the base cell:

    s32[2304,36,36,135]  all-gather = 1 612 431 360 B   (P·nk·n_rtot·4,  P·nk = 16·144)
    c128[2304,8603]      all-gather =   317 140 992 B   (P·nk·ngkmax·16)

They are `WfnLoader` collectives, not SlabIO ones, so nothing in this workstream
could have touched them — and with the writer gone the first of them is now **the
biggest collective in the run**. Their cure is **AA.1**'s
`device_put_process_local` (measured in `wk_AA`), which lands in `wt-J`; AB's traces
are a clean "before" for it at a P the AA cells did not run. **AA and AB are
independent and compose**: AA deletes the two loader gathers, AB deletes the writer
gather, and the 606c/P=16 dump says the intersection is empty.

### AB.6 — cutover, and what is deliberately NOT done

Full detail in `wk_AB/CUTOVER.md`. The three points that matter here:

* **Activation is one line** for any sbatch that already sets the ScaLAPACK/SLATE
  host-FFI runtime (`wk_Y/probe.sbatch`, `wk_L/bench2.sbatch`, the production
  templates): `export PYTHONPATH="$OVERLAY/site:$SRC:$PYTHONPATH"`. The
  `LD_LIBRARY_PATH` / `I_MPI_PMI_LIBRARY` / `FI_PROVIDER_PATH` blocks those scripts
  need are already **exactly** the blocks the overlay's h5py needs — nothing new.
  Launch must be `srun --mpi=pmi2` (already true).
* **Do not fold into the base venv until the flagships drain.** `use_ffi_io` already
  defaults to `True`, so folding it in silently flips **every** `slab_io=auto` CPU
  run from `H5PY_ALLGATHER` to `PHDF5_HOST`. That is the intended end state, but it
  should be a decision, not a side effect. The fold is then a 10-second
  `pip install` of the two already-built wheels in `overlay/wheels/` — no rebuild.
* **Rollback is three levels deep and all cheap**: drop the `PYTHONPATH` line
  (per job) / set `slab_io = h5py_allgather` explicitly (per deck, and the way to
  A/B with mpi4py still importable) / `mv site site.disabled` (global).

Deliberately not done, and named for whoever is next:

1. **`mpi4py` initialises MPI with `thread_level = "multiple"` by default** (its
   `rc` default, checked in the shipped source), which matches what
   `ffi/phdf5/cpp/context.cc` and `ffi/slate/cpp/context.cc` request, and both guard
   on `MPI_Initialized`. So mpi4py-first and FFI-first orderings should both be safe
   — but **no cell in this workstream ran `PHDF5_HOST` together with
   `distributed_zeta_solve=distributed` / `distributed_lu=scalapack`**, i.e. the
   double-`MPI_Init_thread` path is reasoned-about, not tested. Test it before the
   first production run that wants both.
2. `_slab_io_mpi_host` uses **independent** MPI-IO (`dset[slc] = arr`), not
   `collective` mode. That is the FFI's own default and it is what was gated here;
   whether `dxpl.set_dxpl_mpio(COLLECTIVE)` is faster on Frontera's Lustre is an
   open (and now cheap) measurement.
3. Replicated axes make every rank write identical bytes to the same hyperslab under
   independent MPI-IO — semantically correct, documented in the module, but it is
   redundant I/O for the small metadata datasets.
4. **No source edits were needed.** The router, `_slab_io_mpi_host.py`, and the
   planner's `slab_io_replicates` wiring were all already correct and were exercised
   end-to-end for the first time by this workstream.

## AE — the phdf5 WRITE core ported to the host lib: the CPU writer is now bare-MPI C++, and the mpi4py overlay is optional (wt-F, branch `host-write-ffi`, base f484265, 2026-07-26 — NOT committed)

**One line: the owner's preferred permanent answer — "isn't the phdf5 slab io
written in C++ so it could use bare MPI?" — is now true. `write_ffi.cc`
compiles into `liblorrax_ffi_host.so` from the SAME translation unit as the
CUDA lib under `LORRAX_FFI_NO_CUDA`, the host build's device-staging seam
collapses to nothing (H5Dwrite reads the XLA buffer in place), and
`slab_io=auto` on CPU is now capability-probed PHDF5_FFI → PHDF5_HOST →
H5PY_ALLGATHER — so AB's mpi4py + h5py-parallel overlay is no longer on the
critical path for the writer fix.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AE/` — `build.sbatch`,
`gate_p4.sbatch`, `gate_dist.sbatch`, `probe_ae.sbatch`, and the job logs
`build.7876099.out`, `gate.7876100.out`, `gatedist.7876103.out`,
`probe.7876102.out`, plus `run_{base,ffi,ffidist,distbase}/` and
`runs/ae606_{base,ffi}/`.
Library: **`$WORK/lorrax_ffi_unified/build_host_W/liblorrax_ffi_host.so`**
(11 targets). `build_host_V` (10) and `build_host` (9) are untouched.

### AE.1 — the port: one TU, three seams, and one seam that DISAPPEARS

`src/ffi/phdf5/cpp/platform_seam.h` is new and now owns the seams that
workstream A had written inline in `read_ffi.cc`; `read_ffi.cc` and
`write_ffi.cc` both include it, so the two TUs cannot drift:

1. **handler binding** — `LRX_PHDF_HANDLER(PhdfWrite)` → `PhdfWriteFfi`
   (CUDA, with `.Ctx<PlatformStream<cudaStream_t>>()`) or
   `PhdfWriteHostFfi` (cpu, no stream Ctx). Same jax.ffi target string
   `lorrax_phdf5_write` on both, so every `ffi_call` site stays
   platform-agnostic.
2. **index copy-in** — the `(ndim × int64)` offset / valid_shape control
   buffers: `cudaMemcpy` D2H vs a plain host read (`copy_index_to_host`,
   now shared instead of duplicated).
3. **payload staging** — and this is the interesting one. The CUDA path
   `ensure_pinned`s, `cudaMemcpyAsync`es the shard D2H and has the writer
   thread wait on `ctx->d2h_event`. **The host path does none of it**:
   XLA's CPU buffer already IS host memory, so `H5Dwrite` reads the local
   ζ shard *in place* — no staging allocation, no copy, no event. At 606
   centroids / P=16 that is 0.75 GB/rank of host RAM and one full memcpy
   per chunk simply not spent. (Lifetime is safe for the same reason the
   CUDA path's async D2H source is: the Python dispatcher holds `A` and
   blocks on the returned token, and XLA does not donate the input, so the
   buffer outlives the Future.)

Everything else in the file — per-rank hyperslab derivation, `valid_shape`
clipping, the empty-selection path, the FIFO writer thread, the
`H5Dwrite` itself — is byte-identical on both platforms and was not touched.

The writer thread is kept on the host build too. It has nothing to
synchronise with any more, but one thread per ctx draining a FIFO is what
guarantees every rank enters the MPI-IO collectives in the same program
order, which is the collective correctness requirement, not a CUDA detail.

**Both platform builds are verified.** The host lib builds clean (job
7876099, zero warnings from LORRAX code — all warnings are jaxlib header
noise), all four phdf5 TUs pass `g++ -std=c++17 -fsyntax-only -Wall -Wextra`
in **both** modes against real CUDA headers, and the **CUDA library itself
was rebuilt from this worktree and its symbol table checked** — see AE.4b.

### AE.2 — the router: capability probe, not env presence

`gw.gw_config._route_cpu_slab_io(print_fn)` replaces the inline
`import mpi4py` test. Three tiers, each loud:

| tier | condition | line it prints |
|---|---|---|
| **PHDF5_FFI** | `probe_target('lorrax_phdf5_write','cpu')` usable | "host FFI exports the collective phdf5 write handler … no mpi4py needed" |
| PHDF5_HOST | else, and `mpi4py` + `h5py.get_config().mpi` | "the host FFI write handler is unavailable (**reason**) … routing through PHDF5_HOST" |
| H5PY_ALLGATHER | else | "neither writer is available (host FFI: **reason**; venv lacks mpi4py…)" |

The probe is `probe_target`, not `has_target`, precisely so the demotion
message carries the three-state diagnosis: *unknown target* vs *library
could not be dlopen'd* (an `LD_LIBRARY_PATH` problem, nothing wrong with
the build) vs *loaded but does not export the symbol* (the only case that
means "rebuild"). A host lib built before this port lands in the third
state and demotes silently-but-loudly to the old behaviour — the same
graceful demotion A's read path uses. `ffi_loader._HOST_TARGET_SYMBOLS`
gains `lorrax_phdf5_write → PhdfWriteHostFfi`, and `has_phdf5_write()`
mirrors `has_phdf5_read()`.  The router runs **only for `slab_io=auto`** —
a deck that names its writer explicitly is honoured verbatim and does not
even pay the `dlopen`.

**The environment-presence-dependent writer routing is gone.** Which
writer runs is now a property of the deployed `.so`, and it is printed.

### AE.3 — GATES

All cells ran with **`PYTHONPATH = <worktree>/src` only** — every one of
them printed `MPI4PY_ABSENT` / `H5PY_MPI False`, so a `PHDF5_FFI` decision
provably is the C++ writer and cannot be the Python one.

**(a) unit — bit-identical to the allgather reference** (job 7876100, P=4,
2×2 mesh, `cohsex_debug` fixture). Both files SlabIO drives, every dataset:

| file | datasets | result |
|---|---|---|
| `tmp/zeta_q.h5` (incl. **`zeta_q_G`** `(9,60,780)` c128) | 48 | **`H5_BITCMP_OK`** — 0 mismatches, 0 attrs differing |
| `tmp/isdf_tensors_60.h5` (**`V_qmunu`**, `G0_mu_nu`, `W0_qmunu`, `psi_full_y`, `enk_full`, `band_window`, `kgrid`, `vhead`, `whead`, …) | 11 | **`H5_BITCMP_OK`** |

Not just ζ: the restart tensors go through the same `SlabIO.write_slab`
and are covered. (AB's fixture compare had one legitimate mismatch,
`isdf_header/fit_provenance`, because its two cells had different run-dir
names; AE's cells differ only in the writer, so even that is identical.)

**(b) e2e — fixture eqp** (job 7876100): all three cells
**PASS, 1888 values, max|Δ| = 1.00e-6 eV, max rel 5.29e-9** vs
`tests/regression/cohsex_debug/eqp_ref.dat` — identical to the
`H5PY_ALLGATHER` baseline to the digit.

**(c) combo — FFI writer + `distributed_zeta_solve=distributed`** (jobs
7876100 `ffidist`, 7876103 `distbase`). Two independent MPI users in one
process (the phdf5 ctx's `MPI_Init_thread(THREAD_MULTIPLE)` and the
ScaLAPACK/BLACS grid behind `pzheevd`), no mpi4py anywhere: runs clean,
tier log confirms `Zeta back-solve tier: distributed`, eqp
max|Δ| = 1.00e-6 eV. AB.6's deferred item 1 ("no cell ran PHDF5_HOST
together with `distributed_zeta_solve=distributed`; reasoned-about, not
tested") is now tested for the FFI writer.
The 2×2 was completed deliberately, because `ffidist` vs `base` differs in
the *solver* as well as the writer and so proves nothing about the writer:

| | replicated tier | distributed tier |
|---|---|---|
| allgather writer | `run_base` | `run_distbase` |
| **FFI writer** | `run_ffi` | `run_ffidist` |

Along the **writer** axis at the distributed tier (`distbase` vs
`ffidist`): `zeta_q.h5` **`H5_BITCMP_OK`** (`zeta_q_G` sha256
`c71f34f19b94ada8` on both) and `isdf_tensors_60.h5` **`H5_BITCMP_OK`**.
The writer is inert under both ζ solvers.

**(d) readelf — no CUDA in the host lib** (job 7876099). DT_NEEDED is
exactly `libhdf5.so.310, libmpi.so.12, libslate.so.2,
libmkl_{scalapack_lp64,blacs_intelmpi_lp64,intel_lp64,gnu_thread,core,gf_lp64}.so,
libgomp.so.1, libfabric.so.1, lib{lapackpp,blaspp}.so.2, libstdc++.so.6,
libm/libgcc_s/libc` — `AE_CUDA_FREE_OK`. All 11 handler symbols present
(`AE_SYMBOLS_OK`), i.e. V's 10 **plus** `PhdfWriteHostFfi`; `build_host_V`
byte-for-byte untouched.

### AE.4 — **THE GATE**: 606 centroids, P=16, the writer all-gather is gone — with NO mpi4py

Job **7876102**, 8 nodes × 2 ranks, both cells in ONE allocation, same nodes,
same compile storm, `MAXCHUNK=3 EXITZETA=1 TIER=replicated`, differing only in
`slab_io`.  This is AB.2's gate re-run against the FFI writer; the difference
from AB is that **nothing was prepended to `PYTHONPATH`** — both cells printed
`MPI4PY_ABSENT`, so `PHDF5_HOST` was not reachable and the router's PHDF5_FFI
decision is provably the C++ writer.

| | base — `H5PY_ALLGATHER` | ffi — **`PHDF5_FFI` (host lib)** |
|---|---|---|
| collective instructions | 28 | 28 |
| total collective RESULT bytes | 19.441 GB | **7.390 GB (−62.0 %)** |
| all-gather bytes | 15,486,716,480 | **3,435,359,168 (−12,051,357,312)** |
| `jit__identity_fn` module | 7 instrs / 13.753 GB | **7 instrs / 1.702 GB** |
| **biggest collective** | **12 051 357 696 B `c128[144,608,8603]` all-gather** | **1 194 393 600 B — the loader gather** |
| wall | 682 s | **550 s (−19 %)** |
| optimized HLO modules scanned | 184 | 185 |

The deleted instruction, verbatim from the base dump and **absent from every
one of the ffi cell's 185 modules**:

    %all-gather = c128[144,608,8603]{2,0,1} all-gather(%copy), channel_id=1,
                  replica_groups=[1,16]<=[16], dimensions={1}, use_global_device_ids=true

— byte-for-byte the same instruction AB.2 deleted with the mpi4py overlay,
now deleted by a `.so`.  `nq·μ_pad·ngkmax·16 = 144·608·8603·16`, and the
all-gather total drops by 12 051 357 312 B, i.e. that instruction to within
384 B (one small unrelated gather differs between the cells).

**`zeta_q.h5` bit-compare: `H5_BITCMP_OK`, 0 mismatches, 0 attributes
differing** — including the full production-shape

    zeta_q_G   (144, 606, 8603)  complex128  BIT-IDENTICAL  sha256[:16]=cb4e67bafbff21c3

written as **16 clipped MPI-IO hyperslabs**, byte-for-byte equal to the rank-0
serial allgather write.  μ_pad = 608 against a logical `n_rmu` = 606, so
rank 15's 38-row shard runs off the end of `valid_shape` and the C++
`file_count` clip must drop 2 of its rows — the padded-tail path the P=4
fixture cannot exercise.  (This is the C++ clip in `WriteDispatch`, the
counterpart of `_slab_io_mpi_host._clip_shard_to_valid` that AB gated.)

**Node-0 memory, 15 s sampler** — both cells reach the *same* ~51 GB transient
compute peak mid-fit, so the writer's signature is entirely in the closing
phase:

    base (allgather)  21:40:25 → 21:42:10  used_GB 15→19→22→25→28→31→38→44 → exit
    ffi  (MPI-IO)     last 90 s            used_GB 15,15,14,15,15,15,15,11 → exit  (no ramp)

≈105 s of monotone allocate-and-write on the allgather cell, **zero on the FFI
cell** — which instead reports its cost honestly and cheaply in the close log:

    [SlabIO.close] draining 1 pending writes for zeta_q.h5 …
    [SlabIO.close] Python dispatch drained in 14.7 s; joining writer thread
    [SlabIO.close] writer thread joined in 0.0 s; calling H5Fclose collectively
    [SlabIO.close] H5Fclose returned in 1.1 s

i.e. 15.8 s of real per-rank collective I/O in place of ~105 s of gather-and-
copy.  Same conclusion as AB.3, reached with no Python package involved.

### AE.4b — the CUDA build is not broken by the shared-header refactor (rtx-dev)

Hoisting the seams out of `read_ffi.cc` into `platform_seam.h` touches the
GPU library too, and Frontera cannot link-test it as a side effect of the
CPU work — so it was gated explicitly (jobs **7876117 / 7876214 / 7876349**,
artifacts in `wk_AE_gpu/`):

* `liblorrax_ffi.so` (CUDA, `LORRAX_FFI_PHDF5=1`) **builds clean from wt-F**
  into a private stage (`$WORK/lorrax_ffi_wtF_cuda`; the shared
  `$WORK/lorrax_ffi/{build,build_phdf5}` were not touched). **Zero warnings
  naming any file under `src/ffi/phdf5/`** — all 28 are jaxlib-header noise
  that also appears on `cusolvermp/cpp/*` TUs AE never touched.
* `nm -D` exports exactly the CUDA names — `PhdfWriteFfi`, `PhdfReadFfi`,
  `PhdfReadKchunkFfi`, `PhdfReadKchunkUnionFfi` — and **no `*HostFfi` name
  leaked in**, which is the thing the `LRX_PHDF_HANDLER` macro exists to get
  right.
* Off-machine detail worth recording: `config/frontera/build_ffi.sh`
  compiles these TUs with **g++ against the CUDA headers, not nvcc**
  (deliberate — the pip toolchain has ptxas but no nvcc driver). "Compiles
  and links into the real CUDA .so" is proven; "compiles under nvcc" is not
  a state this build system reaches, for AE or for anything before it.

The GPU **end-to-end** writer gate (fixture gw with `slab_io=phdf5_ffi` on
4 RTX GPUs, bit-compare vs the allgather cell) is **unproven, not failed**:
every cell froze in `gw_init`'s `load_centroid_wfns` →
`load_centroids_band_chunked`, before any SlabIO / `H5Fcreate` / phdf5
`MPI_Init` — one rank spinning at 100% CPU with a constant IP and RSS, three
parked, all four GPUs idle. **The control settles it**: a pristine
`git archive f484265` tree (no AE changes, no `platform_seam.h`) run against
the *pre-existing shared* production lib stalls identically (`rc=124` at
1500 s, zero `Zeta fitting` lines). The hang is **pre-existing at f484265 on
rtx-dev and independent of this port** — it belongs to whoever owns the GPU
path, and it blocks any GPU writer gate, AE's or otherwise. (Diagnosis was
capped by there being no `py-spy` in the venv and host `gdb` being unable to
attach into the container.)

### AE.5 — files touched (worktree only, NOT committed)

New: `src/ffi/phdf5/cpp/platform_seam.h`.
Modified: `src/ffi/phdf5/cpp/write_ffi.cc` (the port),
`src/ffi/phdf5/cpp/read_ffi.cc` (seams hoisted into the shared header —
no behaviour change), `src/ffi/common/cpp/host/CMakeLists.txt` (+write TU),
`config/frontera/build_ffi_host.sh` (+`PhdfWriteHostFfi` in the symbol
gate), `src/ffi/common/ffi_loader.py` (`_HOST_TARGET_SYMBOLS` +
`has_phdf5_write`), `src/gw/gw_config.py` (`_route_cpu_slab_io` + the auto-only guard),
`src/file_io/{slab_io,_slab_io_ffi,_slab_io_mpi_host}.py` (docs only),
`docs/dev/HANDOFF_cpu_frontera_2026-07.md` (open item 3 → DONE),
`docs/architecture/codebase.md`, `src/ffi/phdf5/ARCHITECTURE.md`.
Outside the tree: `wk_AB/CUTOVER.md` gains the "overlay is now optional"
banner and a pointer at the top of its fold-in section.

### AE.6 — what this changes for whoever is next

* **Deploying the writer fix is now `cp` of a `.so` + `LORRAX_FFI_HOST_SO`.**
  No venv surgery, no `PYTHONPATH` prepend, nothing to fold in while
  flagships are running against the shared venv. AB's three-level rollback
  is replaced by "point at `build_host_V` instead of `build_host_W`".
* **`PHDF5_HOST` is not dead** — it is tier 2 and the writer A/B control,
  and it is the only option on a site that cannot build the host lib.
* **Not done, named:** the host write path uses **independent** MPI-IO
  (`ctx->use_collective_write` defaults false), same as
  `_slab_io_mpi_host`; whether `H5FD_MPIO_COLLECTIVE` is faster on
  Frontera's Lustre is still the open (and now one-env-var) measurement
  AB.6 named. Replicated axes still make every rank write identical bytes
  to the same hyperslab — correct, redundant, small.

## AC — flagship A recovered on the AA+AB stack, and the post-ζ stages gated by restart (wk_AC, 2026-07-26; **zero source edits**)

> **⚠ CLAIM-DECAY (AL, 2026-07-27): em1-scoped collective walls.** Every multi-node JAX CPU-collective WALL TIME in this section was measured while Gloo was bound to Frontera's 1 GbE management NIC (`em1`, 129.114.x.x) — the campaign-wide default before `runtime.pin_gloo_interface()` landed (AK.10/AL). Byte counts, HLO collective inventories and residency ledgers remain valid; the wall-time consequences of collective-bound stages re-price on ib0 (measured 3.3x whole-pipeline at 785c/P=16, and see AL for 606c/P=80). Single-node numbers are unaffected.

**One line: flagship A's 7875551 never finished a single r-chunk (it predates
AA and ran the defeated `per_q`), the `n_keep` datum it DID produce is salvaged
in full, the relaunch on PHDF5_HOST cut the planner's HWM 28.58 -> 14.35 GB/dev
and its writer term 27.6 GB -> 1.37 GB/dev, `distributed_zeta_solve=distributed`
turned out NOT to survive P=144 (a Gloo failure in the C+ formation, plus a
1.75x-SLOWER eigh than the replicated route it replaces), and the AA-fixed
`per_q` relaunch ran a flat 283 s/chunk -- a 71 min projected zeta-fit against
T.7's 2.5 h -- until the owner stopped it at 3/15 for cluster priority.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AC/` — `runAC.sbatch`
(the three-mode production template), `stage_table.py` (login-node, stdlib-only
log reducer), `POSTMORTEM_7875551.md`.
Also `watch.sh` (the 2-min health poll), `SUBMIT_NOTES.md`, and
`runs/{dist_P144_FAILED,perq_P144_KILLED}/` (both logs + samplers).
Run dir: `/scratch2/08271/jackmc/lorrax_mos2_12x12/run_A_c2406_b400_AC/`
(`gw.in`, `gw_restart.in`, `gw.<jobid>.out`, `mem_node0.<jobid>.log`).
Source: main checkout `/work2/08271/jackmc/frontera/lorrax` @ **f484265**
(AA merged), unmodified. Overlay: `lorrax_env_mpi_overlay/site` (AB).
Host FFI: `lorrax_ffi_unified/build_host_V` (V.7c, the 10-target lib).

### AC.0 — the correction: job 7875551's ζ-fit did NOT complete

See `POSTMORTEM_7875551.md`. Summary: **zero** `LoopProgress` milestones in
7 h 26 m, so r-chunk 1 of 15 never finished; the last content line is the
pre-loop `G-flat ζ accumulator` print and `tmp/zeta_q.h5` froze at 16 717 956 B
(the header write) at 12:37. The job was submitted at 12:11, **before** AA
(`f484265`, 16:20) merged, so `auto` resolved to the *defeated* `per_q` — the
form Y.2 refuted, which gathers the whole `(nq, μ, μ)` = 13.81 GB/rank and
slices afterwards, `nq = 144` times per r-chunk ⇒ ≈2 TB of gather traffic per
rank per chunk. The 48.5 GB/rank writer all-gather was the *second*,
never-reached death. The banner in that log lacks AA's
`(×nq executions/r-chunk)` clause, which dates the module exactly.

**Record datum salvaged: the `n_keep` distribution at nband = 400.**

| nband (b_pad) | `n_keep`/q | mean | / n_log = 2406 |
|---|---|---|---|
| 160 | 1050–1056 | 1051.83 | 43.7 % |
| **400 (432)** | **1675–1680** | **1676.31** | **69.7 %** |

2.5× the band window buys **1.594×** the kept rank — the ceiling *does* track
the window rather than μ, but sub-linearly, and it did not approach the ~2600
the ">= 8× bands" rule predicted. λ_max ∈ [2.5251e-2, 2.5291e-2],
λ_min(kept) ∈ [2.5255e-10, 2.5507e-10]; the retained condition number is
1.0e+8 = 1/rcond, so the cut is **rcond-bound, not spectrum-bound**.
Identical to the bin in job 7875070, which died in chunk 2/15 — two independent
runs, same histogram.

### AC.1 — the relaunch: what changed, and the gates it cleared at startup

`run_A_c2406_b400_AC/gw.in` is `run_A_c2406_b400/gw.in` with **three plumbing
keys** changed and no physics touched:

| key | 7875551 | 7876062 |
|---|---|---|
| `use_ffi_io` | `false` | **`true`** |
| `slab_io` | `h5py_allgather` | **`auto`** ⇒ router picks `PHDF5_HOST` |
| `distributed_zeta_solve` | (unset ⇒ `auto` ⇒ `per_q`) | **`distributed`** |

plus the AB activation block in the sbatch (`PYTHONPATH` overlay prepend,
`LD_LIBRARY_PATH`, `I_MPI_PMI_LIBRARY`/`FI_PROVIDER_PATH`,
`srun --mpi=pmi2`) and `LORRAX_FFI_HOST_SO → build_host_V`.

Live gates from `gw.7876062.out`:

    [config] use_ffi_io=true on CPU backend; phdf5 FFI is CUDA-only.  Routing
    SlabIO through PHDF5_HOST (mpi4py + h5py-parallel) …
    Computing L_q = distributed rank-truncated pinv (2D-sharded C+)
      [PSD, charge channel, path=distributed_rank_truncate]
    Zeta back-solve tier: distributed (distributed_zeta_solve=distributed)
      replicated (nq,μ,μ) gather would be 13.81 GB/rank; per-q tile 0.104 GB
      (×nq executions/r-chunk); distributed tier gathers NO (μ,μ) object
    [scalapack.eigh] n=2448 g=204 grid=12x12 loc=204x204 lwork=144432
      (0.0022 GiB) lrwork=151057 (0.0011 GiB) liwork=17234

**This is the first time all three of AA, AB and V/W have run together at
production scale** — AB.6 explicitly flagged the mpi4py-`MPI_Init_thread` /
FFI-`MPI_Init_thread` ordering as reasoned-about-but-untested, and wk_AB job
7875943 tested it only at 606c/P=16. It holds at P=144.

**The planner's verdict moved, and that is the whole workstream in one table:**

| planner line | 7875551 (allgather + per_q) | 7876062 (PHDF5_HOST + distributed) |
|---|---|---|
| `band_chunk` / `r_chunk` / `q_chunk` | 432 / 11664 (15) / 144 | **432 / 11664 (15) / 144 — unchanged** |
| `persistent` | — | 2.56 GB/dev |
| `F_tensor_write` stage peak | (the binder) | **1.37 GB/dev** |
| `A_centroid_load` | — | 12.23 GB/dev |
| `C_fit_one_rchunk` | 14.23 GB/dev | 14.35 GB/dev |
| **HWM / binder** | **28.58 GB/dev, `F_tensor_write`** | **14.35 GB/dev (17 % of budget), `C_fit_one_rchunk`** |

The writer term fell from *the* binder to 1.37 GB/dev — `slab_io_replicates=False`
divides it by P — and the binder reverted to the honest one, the fit kernel.
Chunk selection is byte-identical, so nothing about the numerics moved.

### AC.2 — **NEW FINDING: `distributed_zeta_solve = distributed` does not survive P=144** (job 7876062)

The tier resolved, the ScaLAPACK handler loaded, `pzheevd` ran to completion on
all 144 q, the truncation telemetry printed — and then the run died in the very
next step, forming `C⁺`:

    File ".../src/isdf/core.py", line 1677, in _factor_c_q_distributed_rank_truncate
      return _dist_factor_cache[key](W, V)
    jax.errors.JaxRuntimeError: UNKNOWN: Gloo ReduceScatter failed:
      [gloo/transport/tcp/pair.cc:547] Connection closed by peer

**It is not memory.** `sacct` MaxRSS = **10 693 036 K = 10.69 GB/rank** against
an 85 GB budget and a 14.35 GB planner HWM; node-0's sampler sat flat at 10 GB
for the whole 45 min; there is no `bad_alloc`, no `RESOURCE_EXHAUSTED`, no
`Killed` anywhere in the log. The error census is pure transport: 306 ×
`Socket closed`, 76 × `Connection closed by peer`, 68 × `Gloo AllGather failed`,
12 × `Gloo ReduceScatter failed`, 4 × `Connection reset by peer`. The first
ranks to go were 119–132 (nodes `c182-03x/04x/05x`); everything after that is
the documented cascade.

The collective that broke is `_pinv_local`'s pair —
`lax.all_gather(V_loc,'x',axis=1,tiled=True)` then
`lax.psum_scatter(...,'y',...)` — q-batched over all 144 q at once. Per rank
that is `nq·μ_pad·(μ_pad/Py)·16 = 144·2448·204·16` = **1.15 GB gathered plus a
~1 GB reduce-scatter**, on Gloo/TCP-over-IPoIB across 144 ranks. **V/W only ever
exercised this tier at P ≤ 16** (V.5's gates are P=4 and P=16; V.3's ladder tops
out at a 4×4 grid), and V.3 already recorded three `DEADLINE_EXCEEDED` /
`Socket closed` deaths in its own harness. This is that failure mode arriving on
the production path.

**And `pzheevd` got *slower* going 16 → 144 ranks at fixed n.** The batched FFI
call is 144 sequential `pzheevd`s with no progress output; it began ≈19:58 and
the truncation telemetry landed ≈20:29, so **≈30 min for 144 matrices ⇒
≈12 s/matrix at n=2448 on a 12×12 grid**, against V.3's **2.89 s/matrix** on a
4×4 grid — a **4× regression for 9× the ranks**. At `g = N/max(Px,Py) = 204` the
12×12 grid holds exactly one 204×204 block per rank, i.e. a pure block (not
block-cyclic) layout, and the tridiagonal reduction is latency-bound.
Workspace was a non-issue (`lwork=144432` ⇒ 2.2 MiB + 1.1 MiB rwork per rank).

**And the control ran the next hour, on the same 72 nodes, so this is a
same-machine A/B**: job 7876086's `replicated_rank_truncate` did the *identical*
144 × n=2448 eigh in **≈17 min** (`Computing L_q` at 21:10:26; the four
`[zeta rank_truncate]` q-batch blocks landed ≈21:15 / 21:19 / 21:25 / 21:27),
i.e. **≈7 s/matrix on 28 cores, P-independent** against `pzheevd`'s
**≈12 s/matrix on 144 ranks**.

| eigh route, 144 × n=2448, same nodes | wall | per matrix | outcome |
|---|---|---|---|
| `pzheevd` on the 12×12 grid (7876062) | ≈30 min | ≈12 s | followed by a fatal Gloo collective |
| replicated `jnp.linalg.eigh` (7876086) | **≈17 min** | **≈7 s** | **completed** |

V.3's honest note — "`pzheevd` does not beat the replicated native eigh on WALL
TIME at these sizes and rank counts" — therefore extends past its own ladder:
at P=144 the distributed eigh is **1.75× SLOWER** than the replicated one it was
built to replace, and its downstream `C⁺` step does not survive the transport.
The tier's remaining virtue is memory (13.81 GB/rank of replicated factor
avoided), which at this μ the machine did not need.

**Operational hazard worth naming:** a 30-minute FFI call that prints nothing is
indistinguishable from a hang. `wk_AC/watch.sh` therefore gives the setup/eigh
phase a 2400 s stall threshold and every other phase 180–600 s.

**Free consistency check the failure handed us.** The distributed route reports
`n_keep` against `n_pad`, the replicated route against `n_log`:

| route | banner | `n_keep`/q | mean |
|---|---|---|---|
| `replicated_rank_truncate` (7875551) | `n_log=2406` | 1675–1680 | 1676.3125 |
| `distributed_rank_truncate` (7876062) | `n_pad=2448` | 1717–1722 | 1718.3125 |

The difference is **42.0000 = n_pad − n_log**, i.e. exactly the identity-pad
block's 42 unit eigenvalues, which are always above the cut. Histograms match
bin for bin (30/60/39/12/3). The two routes agree on the physical kept rank to
the last count — V.4's pad handling is correct, and nobody should read the
larger number as a discrepancy.

### AC.2b — the flagship went out on `per_q` instead (job 7876086)

`per_q` is what `auto` picks at this size anyway (T.5's ladder: `nq·μ_pad²·16 =
13.81 GB` > the 4 GiB `LORRAX_ZETA_GATHER_CAP_GIB`), it is the tier **AA.2
structurally fixed** (1.02× `replicated`'s back-solve wall, `nq`× smaller live
gather, bit-identical to `auto` at P=4 in AA.4), its per-execution gather is
**0.104 GB** rather than 1.15 GB, and its charge factorization is the
`replicated_rank_truncate` eigh that **completed at exactly this size** in both
7875551 and 7875070. So it is simultaneously the safer and — per AC.2's 12 s vs
2.5 s/matrix — the *faster* choice here.

`runAC.sbatch` gained `AC_TIER=<tier>`, which rewrites
`distributed_zeta_solve` into a generated `gw.run.<jobid>.in` kept beside the
log. Varying it across reruns in one directory is safe **by design**: T.5
deliberately keeps the tier out of the ζ fit-provenance hash because it is
numerically neutral, so a tier change does not invalidate an on-disk ζ and the
`vq` gating mode still re-enters at `compute_V_q`.

### AC.3 — the first real ζ-fit chunk timings at 2406c / b432 / P=144 (job 7876086, killed by the owner at 3/15 chunks for cluster-priority reasons — **not** a failure)

Three chunks completed and the cadence is flat, so this is the datum the
campaign has been trying to get since the c2406 ladder started.
`LORRAX_RCHUNK_DEBUG=1`, rank 0:

| chunk | fit [s] | of which `z_q_build` | of which back-solve | `h5_write` [s] | rank RSS [GB] | `live_arrays` [GB] |
|---|---|---|---|---|---|---|
| 1/15 | 272.9 | 46.0 | 226.8 (83.1 %) | 0.76 | 4.562 | 85.597 |
| 2/15 | 282.0 | 54.7 | 227.3 (80.6 %) | 0.006 | 4.375 | 85.597 |
| 3/15 | 284.2 | 55.3 | 228.9 (80.5 %) | 0.006 | 4.375 | 85.597 |
| **steady state (2+)** | **283.1** | **55.0** | **228.1** | **0.006** | — | — |

**Projected full ζ-fit = 273.7 + 14 × 283.1 ≈ 4237 s ≈ 71 min**, against T.7's
prediction of "~600 s/chunk × 15 ≈ 2.5 h" — **2.1× better than planned**.
The code's own `LoopProgress` agreed live: `ETA 1833 s` after chunk 2.
Add the ≈17 min replicated eigh and ≈2 min of startup and the whole
pre-writer stage is **≈90 min**. The owner's "post-ζ ≤ 50 % of the ζ-fit wall"
criterion therefore sets the budget at **≈35 min for V_q + screening + Σ**.

Two controls inside the table:

* **`live_arrays` is constant to the byte** (85.597 GB) across all three
  chunks, and rank RSS *falls* 4.562 → 4.375 GB with `d_fit = −0.591` on
  chunk 2. **T.2's glibc-trim cure holds at 2406 centroids** — the ramp that
  killed 7874803 and 7875071 is absent at the largest μ yet run.
* **`h5_write` collapses to 6 ms** after chunk 1. Under `H5PY_ALLGATHER` this
  is the per-chunk accumulate into `gflat_acc`; the number simply confirms the
  accumulate is not where the writer cost lives — that is all in the single
  post-loop `write_g_flat`, which is what AB moved to per-rank hyperslabs.

Node-0 memory (15 s sampler) over the whole run: **11–35 GB used of 192**,
peak 35 GB at a chunk boundary, `buff_cache` 46 GB. Against the 120 GB/node the
allgather stack predicted for this configuration, and against the planner's
14.35 GB/dev × 2 = 28.7 GB/node, so the planner is **1.22× conservative** here.

**Stages NOT reached** (owner kill at 21:40:59, 3/15 chunks in): the post-loop
`write_g_flat`, `V_q`, the restart-tensor write, screening, Σ, eqp. **The
post-ζ half of the mission is therefore still open**, and the gating harness in
AC.4 is what the next wave should use to close it — it does not need another
8 h run, only a `full` run to the end of the fit, then two ≤2 h cells.

### AC.3b — **the `zeta_q.h5` size test is a TRAP under `PHDF5_HOST`, in both directions**

The kill was ordered partly on the reading that `zeta_q.h5` "was still 16.7 MB".
That file is in the **old** run dir:

    run_A_c2406_b400/tmp/zeta_q.h5        16,717,956 B   mtime 12:37  (job 7875551)
    run_A_c2406_b400_AC/tmp/zeta_q.h5 47,706,807,444 B   mtime 21:27  (job 7876086)

and the AC file is **47.71 GB — the full logical `zeta_q_G` extent**:
`nq·n_rmu·ngkmax·16 = 144·2406·8603·16 = 47 690 076 672` plus the 16.7 MB of
header/metadata, to 1.0004×. Parallel HDF5 **allocates the dataset eagerly at
create time** (MPI-IO needs the extent fixed before any rank writes its
hyperslab), so under `PHDF5_HOST` the file reaches full size **before the first
r-chunk runs**. Hence:

* **16.7 MB is the `H5PY_ALLGATHER` signature, not a hang signature.** On that
  backend `zeta_q_G` only materialises inside `write_g_flat`, so a file stuck at
  16.7 MB means "the loop has not finished", which is true of every healthy run
  for its first N hours.
* **47.7 MB→GB is not evidence the ζ landed either.** The AC file was full-size
  at 21:27, five minutes *before* chunk 1 even completed.

**The only liveness signal is `[rchunk_dbg]` / `LoopProgress`; the only
completeness signal is `isdf_header/zeta_is_done`** (False here by construction —
`mark_zeta_done` runs after the loop). `wk_AC/watch.sh` keys on exactly those and
never on file size.

### AC.3c — progressing vs wedged: the discriminator, stated once

For the pattern record, the three c2406 deaths are now cleanly separated, and
**file size distinguishes none of them**:

| job | tier / source | chunks done | cadence | verdict |
|---|---|---|---|---|
| 7875551 | pre-AA `per_q` (defeated) | **0** in 7 h 26 m | no `Started zeta fitting`, no milestone, ever | **WEDGED** (≈2 TB/rank/chunk of gather) |
| 7876062 | `distributed` | 0 — died before the loop | eigh finished, `C⁺` collective died | **CRASHED** (Gloo at P=144) |
| **7876086** | **AA-fixed `per_q`** | **3 of 15** | **273.7 / 282.0 / 284.2 s, σ < 2 %** | **HEALTHY — killed by the owner, mid-chunk-4** |

The tell for "wedged" is the *absence of the first* `LoopProgress` line, because
`LoopProgress._start` is set on the first `step()` call: a run that never
finishes chunk 1 prints nothing at all, not even a "Started". The tell for
"slow" is a cadence that exists and is flat. 7876086's was flat to 1.9 %.

### AC.4 — the post-ζ gating harness

`runAC.sbatch` carries three modes on one deck, so the post-ζ stages can be
timed in isolation and a hang can be bisected in minutes rather than hours:

| `AC_MODE` | deck | tmp/ | starts at | isolates |
|---|---|---|---|---|
| `full` | `gw.in` (`restart=false`) | **wiped** | ζ-fit | the flagship |
| `vq` | `gw.in` (`restart=false`) | **kept** | `compute_V_q` — `gw_init._zeta_reuse_ok` sees a complete, provenance-matching `tmp/zeta_q.h5` and skips the fit | **V_q + screening + Σ** |
| `restart` | `gw_restart.in` (`restart=true`) | **kept** | `load_restart_state_from_h5` | **screening + Σ** |

The `vq` mode is the load-bearing one and it is *not* `restart=true`:
`restart=true` skips the V_q build as well (it loads `V_qmunu` from
`tmp/isdf_tensors*.h5`), so it cannot time V_q at all. The ζ-reuse cache —
`zeta_is_done` **and** byte-identical `fit_provenance` **and** matching centroid
table — is what makes "restart after the ζ stage" mean "re-enter at
`compute_V_q`". `restart=true` then subtracts V_q, and the difference of the
two runs *is* the V_q wall, cross-checked against the code's own
`gw_jax.*` timers.

Both gating modes also carry `AC_HLO=1` (rank-0 `--xla_dump_to` +
`JAX_LOG_COMPILES`, K.0's verified flag set) and `AC_MEMDBG=1` (the
`mem_probe` stage peaks) as opt-ins; neither is on for the 8 h flagship.

---

## AD — the SHARDED W solve, ONE eqp assembly, and the GN-PPM rung that had no coverage (wt-G, branch `w-shard-eqp-unify`, base f484265, 2026-07-26 — NOT committed)

> **⚠ CLAIM-DECAY (AL, 2026-07-27): em1-scoped collective walls.** Every multi-node JAX CPU-collective WALL TIME in this section was measured while Gloo was bound to Frontera's 1 GbE management NIC (`em1`, 129.114.x.x) — the campaign-wide default before `runtime.pin_gloo_interface()` landed (AK.10/AL). Byte counts, HLO collective inventories and residency ledgers remain valid; the wall-time consequences of collective-bound stages re-price on ib0 (measured 3.3x whole-pipeline at 785c/P=16, and see AL for 606c/P=80). Single-node numbers are unaffected.

**One line: W now leaves the Dyson solve 2-D sharded as `W_q(μ_X, ν_Y)` and stays
that way into every consumer — the last replicated O(nq·μ²) object in production
is gone, for ZERO numerical change (fixture eqp0/eqp1 BYTE-IDENTICAL to base at
P=4 and P=8, in COHSEX and GN-PPM, on CPU **and on a real 2×2 CUDA mesh**); the
two eqp entry points collapsed into ONE assembly that applies the V_H seam and
the mean-field gate exactly once (byte-identical from both entry points on all
three `kin_ion` shapes); the multi-device GN-PPM rung J proved had ZERO coverage
is in the suite and green at 2×2 and 2×4; the eager PPM fit is one jitted kernel
whose fitted Ω/B are bit-identical; and `w_dyson_solver = distributed` runs the
whole solve through the `linalg.plan` seam against real ScaLAPACK, reproducing
the per-q LU to 0.00e+00 eV.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AD/` (`gate_{a,b,c,e,g,h}.sbatch`,
`ppm_fit_bitcmp.py`, `w_collectives.py`, `wsolve_table.sh`),
`wk_AD_gpu/` (the CUDA cell), `wk_AD_scalapack/` (the ScaLAPACK cell).
Jobs **7876098** (items 2+3), **7876105** (items 1+4), **7876106** (V_H-seam
branches), **7876108**, **7876110** (rtx-dev, 4× Quadro RTX 5000),
**7876111** (w_dyson_solver controls), **7876112** (ScaLAPACK `distributed`),
**7876115** (HLO A/B), **7876116** (the staged-reshard fix).

### AD.1 — item 2: the dead band-window duplicate, deleted

`Meta.band_ranges` (a `SimpleNamespace` of seven `(lo,hi)` pairs) and its
`Meta.band_range(name)` accessor were read by **nothing** in `src/` or `tests/`,
and its `sigma=(b1,b3)` entry CONTRADICTED the real Σ window
(`BandSlices.sigma = slice(0, b3-b0)`, i.e. `[b0,b3)` — every occupied band).
Reading the projector against the dead convention is what cost workstream N a
wrong root-cause attribution; the warning comment bolted on afterwards was
longer than the code it guarded.

Deleted, with the now-unused `SimpleNamespace` import. `BandSlices`'s docstring
names itself the single source of truth in one line; the stale cross-references
in `gw/cohsex_sigma.build_Gij` and `docs/architecture/codebase.md` now point
there. `Meta` carries the five band EDGES and nothing else.

### AD.2 — item 3: ONE eqp assembly; two entry points that cannot drift

**What was actually duplicated.** The *math* was already shared. What was not
was **the V_H seam** — written once in `gw_output.write_results` (`folded` →
zeros) and again, differently and more completely, in `eqp_bgw.make_eqp_bgw`
(`folded` → zeros, `stored` → substitute the exact array) — plus **two
different mean-field gates in two wordings**. That is precisely the seam
production job 7874840 fell through: −453 eV QP gap at rc=0.

```
        load artifacts                    in-memory arrays
              |                                  |
      make_eqp_bgw                     gw_output.write_results
              \                                  /
               +-------- assemble_eqp() --------+   seam → gate → Z → Newton
                              |
                    EqpAssembly.write()             the ONE BGW formatter
```

* `resolve_hartree_diag_ev(...)` — the seam, one copy, returning
  `(hartree_used, rule ∈ {suppressed, substituted, as-given})`. Its
  `exact_hartree_diag_ev` operand is optional **on purpose**: the live driver
  passes `None` because `sigma_dispatch` already substituted the exact matrix at
  the single point V_H enters `SigmaResult`; the CLI passes `kin_ion.h5`'s
  `v_hartree` because nothing upstream did. The *decision* is one place either way.
* `assemble_eqp(...) → EqpAssembly` — seam, mean-field gate, Z-factor, Newton
  update, each exactly once. `EqpAssembly.write()` is the only formatter call.
* `write_results` ALWAYS emits `eqp0.dat`/`eqp1.dat`; `make_eqp_bgw` is now
  load-artifacts → `assemble_eqp` → `.write`, with **zero independent
  semantics** — its only remaining private step is reading `v_hartree` off disk
  to hand the seam its operand.
* **One gate, not two.** `_warn_on_unphysical_h0` (the source-aware one, which
  says *"this is NOT an ISDF convergence problem"* on the exact routes) now runs
  for both paths from inside `assemble_eqp`, and emits its failure through
  `common.sanity.warn` — so it gained the `*** LORRAX SANITY FAILURE` grep token
  and the `LORRAX_SANITY=strict` raise that used to exist only on the CLI, and
  the live driver gained both for the first time. `compute_eqp_diag` stays
  silent (pinned by its existing test).

| gate | result |
|---|---|
| fixture COHSEX **P=4**, `eqp0_test.dat`/`eqp1_test.dat`, base f484265 vs branch | **BYTE-IDENTICAL** |
| fixture COHSEX **P=8**, same | **BYTE-IDENTICAL** |
| fixture **GN-PPM P=4 and P=8**, both files | **BYTE-IDENTICAL** |
| **GPU**, 2×2 CUDA mesh, COHSEX and GN-PPM | **BYTE-IDENTICAL** (job 7876110) |
| post-hoc `make_eqp_bgw` on the fixture artifacts, base vs branch — **legacy** kin_ion | **BYTE-IDENTICAL** |
| same — **folded** (`has_hartree=True`, no `v_hartree`) | **BYTE-IDENTICAL** |
| same — **stored** (pristine kin_ion + a `v_hartree` deliberately ≠ sigma_mnk's ISDF column) | **BYTE-IDENTICAL** |
| the seam is LIVE, not a no-op: branch legacy vs folded, folded vs stored | **DIFFER** (both), as they must |
| new cell `test_both_eqp_entry_points_are_byte_identical` — CLI vs the live in-memory entry point on the same artifacts, all three kin_ion shapes | **PASS** |
| `tests/test_sanity_gates.py` / `_jax.py` / `test_eqp_bgw.py` / `test_ffi_linalg_contract.py` | **23 / 35 / 4 / 9 passed, 0 failed** |

Byte-comparison is over everything past the `#` provenance line, which carries a
UTC timestamp and differs on every write by construction.

**Which tree each cell ran on** (the staged output reshard of AD.4a landed
mid-campaign, so this matters): job **7876116** re-ran COHSEX P=4, COHSEX P=8
and GN-PPM P=8 against the base on the FINAL tree — all BYTE-IDENTICAL, and
`Involuntary full rematerialization` = 0 in every cell. The GN-PPM P=4, the
`make_eqp_bgw` seam cells, the GPU cells, the ScaLAPACK cell and the PPM-fit
bit-compare ran on the pre-staging tree; the staging is a pure data-movement
constraint on one array, and the three cells re-run after it are byte-identical,
so those results carry — but they were not individually re-executed and that is
stated rather than glossed.

### AD.3 — item 4: the GN-PPM rung, and the fit as one kernel

**(a) The rung.** J found the fixture's default `compute_mode` resolves to
COHSEX, so the dynamic path — the static+probe W pair, the PPM fit, the
reduce-scatter Σ_c(ω) assembly — had **zero multi-device coverage**.
`compute_mode = gn_ppm` is now a standard cell at P=4 (2×2) and P=8 (2×4), on
CPU and GPU. Proof it resolves rather than silently falling back: the sigma
dump's column labels switch from `sigSX/sigCOH/sigTOT` to `sigX/sigC/sigXC`.

| GN-PPM gate | result |
|---|---|
| P=4 branch vs base, full sigma dump (**2428** values) | **max\|Δ\| = 0.00e+00** |
| P=8 branch vs base, same | **max\|Δ\| = 0.00e+00** |
| GPU 2×2 branch vs base, same | **max\|Δ\| = 0.00e+00**, eqp0/eqp1 byte-identical |
| P=4 vs P=8 on the branch (mesh invariance) | eqp0/eqp1 **BYTE-IDENTICAL** |

**(b) The fit.** `minimax_screening.fit_gn_ppm_from_wc_pair` ran EAGERLY: ~15
concurrent `(nq,μ,μ)` complex128 temporaries (`denom`, `safe`, `ratio`,
`omega_sq`, its real part, four masks, two `where` results, `B_vals`, two
reduction operands), each its own device allocation with zero XLA buffer reuse,
fed by the (then replicated) W pair — and invisible to the ISDF memory model,
which stops at Stage E. At μ_pad = 2048 that is 15 × 4.8 GB of unmodelled arena.

Now one **module-level** `@partial(jax.jit, static_argnums=(4,))` kernel
`_gn_ppm_fit_kernel`, with the single host sync (`_scalar_to_host_float`, which
gathers and therefore cannot be traced) left outside. `n_log` is static, so it
compiles once per (shape, logical extent) — the same key the eager path retraced
on — and module-level, so it is not a Z.1-class in-body-jit hazard.

Bit-exactness is structural, not hoped for: every op is elementwise, so fusion
cannot reassociate anything, and the two reductions count booleans (exact
integers in float64).

| PPM-fit gate | result |
|---|---|
| `ppm_fit_bitcmp.py`, base-src vs branch-src, fixed pseudo-random `(9,24,24)` Wc pair **with a μ pad** and the exact-equal / negative-Ω² / near-tiny-denominator branches all exercised | `omega`, `B`, `good`, `unfulfilled` — **all four BIT-IDENTICAL, max\|Δ\| = 0.000e+00** |

### AD.4 — item 1: the sharded W solve

`w_isdf._get_w_solve_fn` ended in `with_sharding_constraint(W_flat, rep_3d)`
— `P(None,None,None)`, an all-gather of the whole `(nq,μ,μ)` stack onto **every**
rank. It now lands on `nat_3d = P(None,'x','y')`. The `shard_map` above computes
the same numbers either way; only data movement changes, so bit-exactness is by
construction — and measured, in two compute modes, at two mesh shapes, on two
platforms.

**Nothing ever wanted the replication.** Every consumer is either
layout-agnostic (`sigma_dispatch`, `sc_iteration`, `gw_jax` — they put W in a
dict and forward it) or **immediately re-imposes exactly this layout**:

| consumer | use of W | before | now |
|---|---|---|---|
| `symmetry_maps.unfold_v_q` (IBZ→full BZ, `screening.py:223`) | jit `out_shardings=P(None,'x','y')` over a `shard_map` with the same `in_specs` | replicated → sharded scatter | **free** |
| `cohsex_sigma._convolve` (Σ_SX, Σ_COH) | reshapes to 5-D, constrains to `V_FFT5D_SPEC = P(None,None,None,'x','y')` | replicated → sharded scatter | **free** |
| `ppm_sigma.fit_ppm` (`Wc = W − V`, then the PPM fit) | pins `q_shard = P(None,'x','y')` on everything derived | replicated → sharded scatter | **free** |
| `experimental/head_wing_schur` | a hand-written `_reshard_W_to_flatq` that UNDID the replication, with a comment doubting XLA would honour `rep_3d` | explicit undo | **no-op** |
| `gw_output.persist_w0_and_head` → `SlabIO.write_slab` | infers layout from `A.sharding` on the FFI backend; gathers on the allgather backend | — | unchanged; strictly better on `PHDF5_HOST` |
| `screening._gate_w` | `check_hermitian(W[0])` — the ONLY q-index anywhere | — | fine: `gw_init.py:811` already runs the identical check on `V_q_raw[0]`, always `P(None,'x','y')` |
| BSE | never sees this array — reloads `W0_qmunu` from HDF5 in another layout | — | unaffected |

And the `CUBLASMP_FFI` screening solver has **always** returned `P(None,'x','y')`
from the same driver code, so the whole downstream chain was already a shipped,
proven configuration on this layout.

**Per-rank residency:** `nq·μ²·16` → `nq·μ²·16/P`, ×2 for the static+probe pair,
re-paid every SC iteration. This was J.2 offender #3, break-μ ≈ 4.4 k; it now
P-scales.

#### AD.4a — the trap: a DIRECT constraint is replicate-then-partition

Landing on `P(None,'x','y')` in ONE step does **not** remove the gather. XLA
refuses the composite and falls back to replicate-then-partition, which the
compiler announces:

```
[SPMD] Involuntary full rematerialization. The compiler cannot go from sharding
{devices=[4,1,1]<=[4]} to {devices=[1,2,2]<=[4]} ... %copy = c128[3,60,60] ...
op_name="jit(_solve_w)/shard_map"
```

Measured, 1 per rank, on **both** platforms (job 7876115 CPU: base 0, branch 4;
job 7876110 CUDA: base 0, branch 4-of-4-ranks) — and the optimized HLO confirms
it: the `jit__solve_w` module's collective table was **identical** before and
after, still carrying the full `all-gather c128[12,60,60]` (0.691 MB at the
fixture). The transient being gathered is exactly the `nq·μ²` object this change
exists to stop materialising, so the direct constraint bought the residency win
and gave the transient straight back.

**Fix: stage the reshard through `P('x',None,'y')`** — one mesh axis per step,
the exact reverse of the input path whose staging is already documented in this
function with a measured 62 % peak reduction:

```
q-parallel [px·py,1,1] -> P('x',None,'y') [px,1,py] -> P(None,'x','y') [1,px,py]
```

**The W-solve HLO, before and after** (fixture, P=4, 2×2, rank-0
`--xla_dump_to`, module `jit__solve_w`, job 7876116):

| collective | BASE (replicated W) | BRANCH (sharded W, staged) |
|---|---|---|
| **all-gather** | **1 × `c128[12,60,60]` = 0.691 MB** — the WHOLE `(nq_pad, μ, μ)` stack, unsharded, P-INDEPENDENT | **0 — gone** |
| all-to-all | 4 (`c128[1,5,30,30]` ×2, `c128[1,1,3,60,30]` ×2) = 0.317 MB | 6 (the same 4, plus `c128[1,3,60,1,30]` + `c128[5,1,30,30]`) = 0.475 MB |
| collective-permute | 2 × `c128[1,60,30]` = 0.058 MB | 3 × `c128[1,60,30]` = 0.086 MB |
| **total instrs / bytes** | **7 / 1.066 MB** | **9 / 0.561 MB  (−47 %)** |
| **largest single collective** | **0.691 MB — and it is the term with no `/P`** | **0.086 MB — everything now scales `nq·μ²/P`** |

The staged reshard **deletes the all-gather outright**: the base's single
largest collective is the entire `nq·μ²` stack — the object with no `/P` in it,
J.2 offender #3 — and the branch's largest is 0.086 MB with every term now
`nq·μ²/P`. The price is 2 extra all-to-alls and 1 extra collective-permute,
which is the staging, and total collective bytes still fall 47 %.
`[SPMD] Involuntary full rematerialization` count in `jit(_solve_w)`:
**base 0, branch-with-a-DIRECT-constraint 4, branch-with-STAGING 0.**

#### AD.4b — the plan seam: "a solve with v and (1−vχ) that naturally shards it"

A second route, `w_dyson_solver = distributed`, forms `A = (1 − pref·V·χ₀)` and
calls

```python
linalg.plan("solve_lu", mesh, backend="distributed", n=n_rmu).batched(A, B)
```

so the μ axes **never leave `P(None,'x','y')`** — no q-parallel gather of (μ,μ)
tiles at all. `distributed` became legal vocabulary for `solve_lu` (scalapack on
cpu, cusolvermp on CUDA — the two backends `_IMPL` already listed); this is the
second call site to take a resolved plan and the first for an op other than
`eigh`. Two load-bearing details:

* **A is built at the LOGICAL μ extent, then padded with an identity block** (V
  with zeros). Forming `V @ χ` at the padded extent regroups the reduction per
  pad extent — the 1e-8-rel wobble GN-PPM amplifies to eV near a pole
  (ROOT_CAUSE 2026-07-08). The padded system is block diagonal `[[A_log,0],[0,I]]`
  with RHS `[[V_log],[0]]`, so the solution is exactly `[[W_log],[0]]` and
  partial pivoting cannot mix the blocks.
* **A and B are always freshly built buffers.** `scalapack.batched_distributed_solve_lu`
  DONATES both operands, and V is still needed by Σ_SX/Σ_COH/Σ_X and by `Wc = W − V`.

`auto` on a CPU mesh resolves to native, so the production route is untouched; a
resolve-time refusal is caught, printed **with the resolver's own message**, and
falls back to the per-q LU (announced once, not once per SC iteration):

```
[W solve] w_dyson_solver=distributed refused at resolve time: solve_lu backend
'scalapack' requested but its FFI handler (lorrax_scalapack_batched_solve_lu)
is not usable: ... Could not locate liblorrax_ffi_host.so — falling back to the per-q LU.
```

**And with the host lib present it RUNS** (job 7876112, P=4, 2×2, `build_host_V`):

| cell | vs `eqp_ref.dat` | vs the per-q LU control |
|---|---|---|
| control (per-q LU) | PASS, max\|Δ\| = 1.00e-06 eV | — |
| **`distributed` (ScaLAPACK `pXgetrf`/`pXgetrs`)** | **PASS, 1.00e-06 eV** | **max\|Δ\| = 0.00e+00; eqp0/eqp1 BYTE-IDENTICAL** |
| `lstsq` | PASS, 0.00e+00 | **0.00e+00; byte-identical** |

Three independent confirmations the distributed route really ran: no fallback
line (and the print channel is demonstrably live — the `lstsq` cell emitted its
banner into the same capture); `W.compile` dropped 20× (0.242 → 0.012 s: no
shard_map/LU to compile, one custom call) while `W.exec` rose (real BLACS/MPI
setup on a 60×60 problem — distribution is pure overhead at fixture size); and
the native route's `jit(_solve_w)/shard_map` SPMD signature is present in the
control and **absent** in the distributed cell, exactly as the plan promises.
The identity-block padding contract therefore holds empirically.

#### AD.4c — lstsq, and why `solve` is the right default

The owner asked for lstsq/solve. `A = (1 − Vχ₀)` is **square**, and it is `I`
minus a term whose spectral radius is below 1 wherever the RPA screening is
physical — an eigenvalue of `Vχ₀` reaching 1 is a plasmon instability, not a
conditioning failure. So the default is a **solve** (pivoted LU, unchanged), and
`w_dyson_solver = lstsq` is wired through the same seam as the explicit
**rank-deficient fallback**: `jnp.linalg.lstsq`'s SVD returns the minimum-norm W
when A is singular to working precision, at the cost of an SVD instead of an LU.

**Measured eqp delta, lstsq vs lu** (job 7876111, COHSEX P=4, both runs in the
same job): sigma dump **max\|Δ\| = 0.00e+00 over 1888 values at atol 1e-9**;
`eqp0` QP column **max\|Δ\| = 0.000000e+00 eV, 0 of 120 states differ**. The two
algorithms agree below the last printed digit on this fixture — the change of
algorithm is invisible in the observables, which is what a well-conditioned A
predicts. Positive controls that the key is real, not ignored:
`[W solve] Dyson inner solve = lstsq (logical mu extent 60 of 60)` is printed,
and `w_dyson_solver = nonsense` raises
`ValueError: w_dyson_solver='nonsense' invalid; expected auto / lu / lstsq / distributed`.

### AD.5 — files touched (worktree only, NOT committed)

`src/common/meta.py` (−30 lines), `src/gw/wavefunction_bundle.py`,
`src/gw/cohsex_sigma.py` (comments) · `src/gw/eqp_bgw.py` (+the seam, the
assembly, `EqpAssembly`), `src/gw/gw_output.py` · `src/gw/minimax_screening.py`
(the jitted fit) · `src/gw/w_isdf.py` (sharded output + the staged reshard + the
plan route + the lstsq inner), `src/gw/screening.py`, `src/gw/gw_config.py`
(`w_dyson_solver`), `src/ffi/linalg/resolve.py` (`distributed` for `solve_lu`) ·
`tests/test_sanity_gates_jax.py` (+1 cell) · `docs/dev/linalg_ffi.md`,
`docs/architecture/codebase.md`.

### AD.6 — named, not done

* The GPU lattice covers **2×2 only**. No 1×4/4×1/2×4 CUDA mesh, no CUDA FFI
  route (`use_ffi_io=false`, no cuSOLVERMp/cuBLASMp), and only the 60-centroid
  fixture — so the *performance* claim for the shard is unverified at the μ where
  it matters, even though the *correctness* claim is now solid on both platforms.
* The `distributed` W route is gated at P=4 on a square CPU mesh only. Its
  guards (non-square mesh, `n` not divisible by the mesh axes) are exercised only
  as refusals.
* `w_dyson_solver` is a `BackendConfig` field and is not threaded into the
  BSE/htransform CLIs, which take their linalg backends from flags rather than
  the input file — the known asymmetry in the context brief.
* `_get_w_solve_fn_low_mem` (the cuBLASMp fused solver) stays outside the plan
  seam: not a selectable op, no native twin, exactly as `docs/dev/linalg_ffi.md`
  says.

---

## AF — the `distributed` ζ tier survives P=144: a TRANSPORT-AGNOSTIC collective-payload bound, not a memory bound and not a Gloo workaround (wt-J, branch `gloo-robust-distributed`, base a290f5b, 2026-07-26 — NOT committed)

> **⚠ CLAIM-DECAY (AL, 2026-07-27): em1-scoped collective walls.** Every multi-node JAX CPU-collective WALL TIME in this section was measured while Gloo was bound to Frontera's 1 GbE management NIC (`em1`, 129.114.x.x) — the campaign-wide default before `runtime.pin_gloo_interface()` landed (AK.10/AL). Byte counts, HLO collective inventories and residency ledgers remain valid; the wall-time consequences of collective-bound stages re-price on ib0 (measured 3.3x whole-pipeline at 785c/P=16, and see AL for 606c/P=80). Single-node numbers are unaffected.

**One line: AC.2's death was a single-shot 1.15 GB Gloo AllGather (plus a
~1 GB ReduceScatter) in the C⁺ formation at 144 ranks — with MaxRSS at 12 %
of budget, i.e. every existing cap was satisfied and the job still died;
the tier now carries a SECOND, orthogonal bound — `LORRAX_COLLECTIVE_CHUNK_MB`
(default 128 MB) — enforced by a host-level q-block loop that XLA cannot
fuse back, proven bounded in the optimized HLO (the C⁺ all-gather goes
147,456 → 16,384 B and its reduce-scatter 36,864 → 4,096 B, exactly nq×),
and numerically invisible: the chunked distributed tier reproduces the
replicated route to 0.00e+00 over 1888 sigma values at P=4 and P=16.
Job 7876346 walks straight through the step that killed 7876062 (127.8 MB ×
9 executions instead of 1.15 GB × 1) and through a back-solve whose
single-shot payload would have been 5.48 GB, and then runs the ζ-fit
**1.62× FASTER than the `per_q` control** (steady state 174.2 s/chunk vs
283.1 s, back-solve 1.91×) — so the accepted "1.75× slower" tradeoff turns
out to be eigh-only and is repaid by the back-solve within the same run.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AF/` — `runAF.sbatch`,
`gate.sh`, `run.sh`, `holder.sbatch`, `audit_hlo.py` (wk_V's, unmodified),
`gloo_probe.py`, `out_*.txt`, `hlo_nochunk/ hlo_chunk/ hlo_chunk1/`,
cells `P4rep/ P4dist/ P4dist_chunk/ P16dist_chunk/ P16hlo/`.
Run dir: `/scratch2/08271/jackmc/lorrax_mos2_12x12/run_A_c2406_b400_AF/`
(a COPY of AC's deck — AC's flagship dir and its 47.7 GB `tmp/zeta_q.h5`
are untouched).

### AF.0 — the diagnosis AC.2 stopped one step short of: a memory cap is not a transport cap

The tier already had a cap on gathered data: `_ZETA_GATHER_MAX_BYTES`
(`LORRAX_ZETA_GATHER_CAP_GIB`, 4 GiB). It was **satisfied** in job 7876062.
It bounds how much gathered data may be **live**; it says nothing about how
many bytes ONE collective instruction hands to the transport in a single
shot. Those are different numbers and only the second one killed the job:

| evidence (job 7876062, 72 nodes × 2 ranks, 12×12) | value |
|---|---|
| MaxRSS (`sacct`) | 10.69 GB/rank vs an 85 GB budget — **not memory** |
| planner HWM | 14.35 GB/dev — satisfied |
| the C⁺ all-gather, ONE instruction | `nq·μ·(μ/Py)·16` = 144·2448·204·16 = **1.15 GB** |
| the C⁺ reduce-scatter, ONE instruction | `nq·(μ/Px)·μ·16` ≈ **1.15 GB** |
| error census | 306× `Socket closed`, 76× `Connection closed by peer`, 68× `Gloo AllGather failed`, 12× `Gloo ReduceScatter failed` |
| the ScaLAPACK eigh in the same job | **completed all 144 matrices** — its collectives go through Intel MPI, not Gloo |

And the counter-evidence that *fixes the bound* rather than merely lowering
it: job **7876086**, the AA-fixed `per_q` tier, ran HEALTHY on the identical
144 ranks (3 r-chunks, σ < 2 %) while issuing `nq` all-gathers of
**0.104 GB** each per r-chunk. So at this rank count and on this transport,
~100 MB per collective is a **measured-good** payload and ~1.15 GB is a
**measured-fatal** one. The default cap is set just above the measured-good
point, and every collective in the tier is held under it.

The back-solve was the *unreached* second death, and it was bigger: at
`qb = 93` (what the 4 GiB memory cap alone allows at `μ_pad = 2448`,
`r_chunk = 11664`) the GEMM's Z-column gather is `93·2448·972·16` =
**3.54 GB** in one instruction. Fixing only the C⁺ step would have moved the
crash one stage later.

### AF.1 — the fix: APPROACH 1 (payload chunking) — and it is ARCHITECTURE-NEUTRAL

**Read this first, because it is the point.** The fix contains **no
Gloo-specific and no Frontera-specific behaviour anywhere load-bearing.**
The tier still issues plain `lax.all_gather` / `lax.psum_scatter`; the only
change is that it issues them in bounded per-instruction payloads. There is
no transport probe, no fabric detection, no environment sniffing and no
per-backend branch. The identical code path runs unchanged on NCCL/CUDA and
on any other XLA backend, and `LORRAX_COLLECTIVE_CHUNK_MB = 128` is a
**transport-agnostic default, not a cluster tuning**.

That is not a rationalisation of what was convenient — it is the correct
model of the failure. Bounded collectives are the robust regime on *every*
fabric and oversized single-shot ones are the fragile regime on *every*
fabric; they differ only in HOW they degrade (a slow tail, a retry storm, an
outright transport error). **Gloo simply fails loudest**, which is why it was
the fabric that exposed a latent property of the emitted program. A 5.5 GB
single-shot all-gather was never a good instruction to emit on any
interconnect; it happened to be survivable on quieter ones.

**Approaches 2 and 3 were NEVER INVOKED.** No Gloo env tuning is part of the
fix (AF.5 shows there is essentially nothing to tune, which is a finding, not
a lever we pulled). No MPI rerouting is part of the fix — `jax_cpu_
collectives_implementation='mpi'` was priced and left unbuilt. Approach 2
(structural payload reduction) is reachable by turning one knob to its floor
and was exercised only as a *gate* at P=4/P=16, never as the production
route. The delivered capability is approach 1 alone.

### AF.1a — what the knob does, and where

`isdf/core.py` gains one knob and two helpers:

```
LORRAX_COLLECTIVE_CHUNK_MB   default 128 MB   (0 / negative = unbounded,
                                               i.e. exactly the pre-AF code)
_collective_chunk_bytes()    the cap in bytes
_chunk_q(nq, per_q_collective_bytes)   the largest q-block that fits
_chunk_log(...)              one line per call site, on by default
```

applied at BOTH ζ-tier sites, with the block sized from the **largest single
collective** the block emits per q — not the sum, and not the live
footprint, because the interconnect sees instructions, not live sets:

| site | collectives per q | per-q bytes @ c2406/P=144 | q_block | per-execution payload |
|---|---|---|---|---|
| `_factor_c_q_distributed_rank_truncate` (C⁺) | `all_gather('x')` μ²/Py, `psum_scatter('y')` μ²/Px | 7.99 MB | **16** (9 executions) | **128 MB** (was 1.15 GB) |
| `_distributed_pinv_apply` (C⁺Z GEMM) | `all_gather('y')` μ²/Px, `all_gather('x')` μ·r/Py | 38.07 MB | **3** (48 executions) | **114.2 MB** (was **5.482 GB**) |

**The chunking is a HOST-LEVEL loop over q-blocks — one XLA execution per
block — not a loop inside one jit.** That is the load-bearing design choice
and it is the direct answer to the brief's AD warning: XLA carries
collective-combiner passes, so a loop *inside* the jit is a
chunked-in-Python / fused-in-HLO non-fix. Separate executions cannot be
combined by construction, and AF.2 proves the emitted instruction actually
shrank. The pattern is also not new here: `_distributed_pinv_apply` already
looped over eager q-slices for its memory cap, so the C⁺ site was brought
into line with its sibling rather than given a new idiom.

**AD's staged-sharding trap does not apply and was checked, not assumed.**
The trap is about a *direct* `with_sharding_constraint` across two mesh axes
being lowered as replicate-then-partition. Neither site has one: both are
`shard_map` bodies whose collectives are already written one mesh axis at a
time (`all_gather` on `'x'`, then `psum_scatter` on `'y'`), and the q-block
slices are on the **unsharded** q axis, so nothing reshards. The
`[SPMD] Involuntary full rematerialization` count in the ζ-tier modules is
**0** in every dump below.

**Approach 2 (structural payload reduction) was NOT INVOKED** — but note
that the chunking *converges to it*: at a small enough cap, `q_block = 1` is
exactly AA's per-q form, and the P=4/P=16 gates below were deliberately run
in that limit. So approach 2 is reachable by turning one knob rather than by
writing new code, and it was exercised only as a gate. At production scale
the block is preferred because one q's collective is 38 MB — already well
under the cap — so forcing 144 executions per r-chunk would pay 3× the
dispatch latency for no payload benefit.

**Approach 3 (rerouting this collective through MPI) was NOT INVOKED.** It is
recorded in AF.5 with its exact price purely so the next person does not have
to re-derive it. Nothing in the shipped fix depends on the transport layer,
so nothing in it needed rerouting.

### AF.2 — HLO GATE: the emitted collectives really are bounded (P=16, 4×4)

Rank-0 `--xla_dump_to`, optimized HLO, `wk_AF/audit_hlo.py` (wk_V's tool,
unmodified). Same fixture, same 16 ranks, three caps. The ζ-tier's two
`shard_map`s both live in modules named `jit__block`:

| module / instruction | cap 128 MB (does not bite, = pre-AF) | cap 0.5 MB (back-solve only) | cap 1e-5 MB (both, `q_block=1`) |
|---|---|---|---|
| **C⁺ back-solve**, Z-column `all-gather` | `c128[9,64,3372]` = **31,076,352 B** | `c128[1,64,3372]` = **3,452,928 B** | `c128[1,64,3372]` = **3,452,928 B** |
| **C⁺ formation**, V-row `all-gather('x')` | `c128[9,64,16]` = **147,456 B** | `c128[9,64,16]` = 147,456 B | `c128[1,64,16]` = **16,384 B** |
| **C⁺ formation**, `reduce-scatter('y')` | `c128[9,16,16]` = **36,864 B** | `c128[9,16,16]` = 36,864 B | `c128[1,16,16]` = **4,096 B** |
| ζ-tier module total | 31.55 MB | 3.79 MB | 3.51 MB |
| **largest collective anywhere in the run** | **31,076,352 B** (the ζ tier) | — | **15,538,176 B** (`jit_fn`, the V_q path — NOT the ζ tier) |
| run-wide collectives / bytes | 78 / 0.0787 GB | — | 78 / 0.0506 GB |

Read the leading dimension: it is the q-block, and it goes 9 → 1 exactly as
the knob asks, in **both** the all-gather and the reduce-scatter. The
instruction count is unchanged (78 run-wide) — the loop replaced one big
instruction with N small ones *across executions*, so no combiner had
anything to merge. **A chunked-in-Python, fused-in-HLO loop would have left
the `[9,...]` shapes in place; it did not.**

At the 128 MB default the fixture's payloads are already under the cap, so
`q_block = nq` and the emitted HLO is **byte-identical to the pre-AF tier** —
which is why every existing P≤16 gate is unaffected by construction, and why
the forced-cap cells below are the ones that actually test the new path.

### AF.3 — CORRECTNESS GATE: chunking is numerically invisible (P=4 and P=16)

cohsex fixture (`tests/regression/cohsex_debug`), sigma dump = 1888 values,
compared with the established `multidev_compare.py`:

| gate | result |
|---|---|
| **P=4 distributed, cap 128 MB** (`q_block=9`, unchunked) vs `eqp_ref.dat` @1e-3 | **PASS, max\|Δ\| = 1.00e-06 eV**, 0/1888 over tol |
| **P=4 distributed, cap forced → `q_block=1`** vs `eqp_ref.dat` @1e-3 | **PASS, 1.00e-06 eV**, 0/1888 |
| **chunked vs unchunked at P=4** (the gate on the loop itself) | **max\|Δ\| = 0.00e+00 @ atol 1e-12** |
| **P=16 distributed, `q_block=1`** vs `eqp_ref.dat` @1e-3 | **PASS, 1.00e-06 eV**, 0/1888 |
| **P=16 chunked vs P=4 chunked** (mesh invariance) | **max\|Δ\| = 0.00e+00** |
| **P=4 `auto` → `replicated_rank_truncate`** vs ref | **PASS, 1.00e-06 eV** |
| **DISTRIBUTED(chunked, P=4) vs REPLICATED(P=4)** — the brief's gate 1 | **max\|Δ\| = 0.00e+00** |
| **DISTRIBUTED(chunked, P=16) vs REPLICATED(P=4)** — the brief's gate 1 | **max\|Δ\| = 0.00e+00** |
| `[collective chunk]` telemetry present and honest at both sites | yes (`out_P4dist_chunk.txt`, `out_P16dist_chunk.txt`) |

New unit cell `tests/test_zeta_mesh_invariance.py::
test_distributed_tier_collective_payload_is_bounded` — **PASSES** — pins the
default (128 MB), the production arithmetic (`q_block = 16` for C⁺ and `3`
for the back-solve at nq=144 / μ_pad=2448 / r_chunk=11664 / 12×12), the
`q_block ≥ 1` floor, the *non*-biting at fixture scale, and the
`CHUNK_MB=0` reproduction escape hatch. `tests/test_zeta_mesh_invariance.py`
+ `tests/test_charge_zeta_route.py`: **6 passed, 7 skipped, 0 failed**.
`tests/test_ffi_linalg_contract.py` + `tests/test_invariance_gates.py`:
**26 passed, 25 skipped, 1 error** — the error is a 900 s *harness* timeout
in the `bispinor_session` fixture SETUP (`tests/conftest.py:150` drives a
whole GW run single-rank through `subprocess`, on a shared dev node), not an
assertion, and it is on the transverse route, which this workstream does not
touch. Stated rather than filtered out.

### AF.4 — THE GATE: P=144 (job 7876346, 72 nodes × 2 ranks, 12×12, c2406/b432)

Same deck as AC's flagship, same nodes class, `AC_TIER=distributed`,
`AC_CHUNK_MB=128`, run dir `run_A_c2406_b400_AF/` (a copy — AC's dir and its
47.7 GB `tmp/zeta_q.h5` untouched). **The tier passed the step that killed
7876062, and both bounds are visible in the run's own telemetry:**

    [collective chunk] C+ formation (pinv): nq=144 q_block=16 (9 executions)
      max collective/exec = 127.8 MB (cap 134.2 MB, unchunked would be 1.151 GB)
    [collective chunk] C+ back-solve (GEMM): nq=144 q_block=3 (48 executions)
      max collective/exec = 114.2 MB (cap 134.2 MB, unchunked would be 5.482 GB)

Note the second number: the back-solve's single-shot payload at this size is
**5.48 GB**, i.e. **4.8× larger than the one that actually crashed**. Fixing
only the C⁺ step would have moved the death one stage later, into the r-chunk
loop — which is why AF.1 bounds both sites from one knob.

**`n_keep` matches the replicated route exactly, mod the known pad offset.**

| route | banner | `n_keep`/q | mean |
|---|---|---|---|
| `replicated_rank_truncate` (7875551, `n_log=2406`) | AC's record | 1675–1680 | 1676.3125 |
| `distributed_rank_truncate` (7876062, `n_pad=2448`) | AC's record | 1717–1722 | 1718.3125 |
| **`distributed_rank_truncate` (7876346, this run)** | `n_pad=2448` | **1717–1722** | — |

Identical to 7876062's histogram — which is the right control, because 7876062
*did* get its eigh and truncation out before dying. The chunking changed the
data movement and nothing else.

**And the tier is FASTER than the `per_q` control it was supposed to trade
wall time against.** Same nodes class, same deck, `LORRAX_RCHUNK_DEBUG=1`,
rank 0:

| r-chunk | AC `per_q` (7876086) | **AF `distributed` (7876346)** | ratio |
|---|---|---|---|
| chunk 1 `fit` | 272.9 s | **167.8 s** | **1.63× faster** |
| chunk 1 `z_q_build` | 46.0 s | 46.9 s | 1.02× (shared code — the control) |
| chunk 1 **back-solve** | 226.8 s | **120.8 s** | **1.88× faster** |
| chunk 2 `fit` | 282.0 s | **174.7 s** | **1.61× faster** |
| chunk 2 `z_q_build` | 54.7 s | **54.9 s** | **1.004× — the control lands** |
| chunk 2 **back-solve** | 227.3 s | **119.7 s** | **1.90× faster** |
| chunk 3 `fit` / `z_q_build` / back-solve | 284.2 / 55.3 / 228.9 s | **173.8 / 54.7 / 119.2 s** | 1.64× / 1.01× / **1.92×** |
| **steady state (chunks 2–3)** | **283.1 / 55.0 / 228.1 s** | **174.2 / 54.8 / 119.4 s** | **1.62× / 1.00× / 1.91×** |
| cadence spread | σ < 2 % | **σ < 0.3 %** | — |
| `h5_write` (chunk 2+) | 0.006 s | 0.005 s | — |
| `live_arrays` | 85.597 GB | **85.597 GB** | identical to the byte |
| rank RSS, chunk 1 → 2 | 4.562 → 4.375 GB | 4.661 → 4.489 GB | both FALL (T.2 trim holds) |

`z_q_build` agreeing to **0.4 %** on chunk 2 is the load-bearing control: it
is the same code on both tiers, so the two runs are on comparable hardware
and the 1.90× back-solve difference is real, not an allocation artefact.
`live_arrays` constant to the byte and RSS *falling* chunk-to-chunk are the
same two health signals AC.3 recorded for `per_q` — the chunked loop does not
leak per execution, which is the thing 48 executions/r-chunk could plausibly
have broken.

**Timing honesty, the whole ledger.** The owner accepted "1.75× slower". The
measured trade is narrower than that and it lives entirely in the eigh:

| stage, c2406 / P=144 | replicated route | distributed route | verdict |
|---|---|---|---|
| charge factorisation (eigh) | ≈17 min (`jnp.linalg.eigh`, AC 7876086) | ≈31 min (7876062) / **≈41 min** (this run, 22:41:57 → 23:23:20) | **1.8–2.4× slower, ONE TIME** |
| ζ back-solve, per r-chunk | 226.8 s (`per_q`, AC 7876086) | **120.8 s** | **1.88× faster, ×15 chunks** |
| ζ-fit r-chunk, per chunk | 272.9 s | **167.8 s** | **1.63× faster, ×15** |

**The whole-stage ledger, projected from the measured steady states:**

| c2406 / P=144, pre-writer | `per_q` (AC 7876086) | `distributed` (AF 7876346) |
|---|---|---|
| charge eigh | ≈17 min | ≈41 min |
| ζ-fit loop, 15 chunks | 273.7 + 14×283.1 = **≈71 min** (projected; AC stopped at 3/15) | **2605.8 s = 43.4 min — MEASURED, all 15/15** |
| **eigh + fit** | **≈88 min** | **≈84 min** |

All fifteen chunks, `fit` in ms: 167832 / 174671 / 173810 / 173975 / 174847 /
173155 / 173658 / 174369 / 175021 / 174865 / 173218 / 174718 / 172733 /
174819 / 174118. Steady-state σ = **0.6 s on 174.0 s (0.4 %)** across chunks
2–15 — flat to the end, with no drift, no leak and no late-run degradation
from the 48-executions-per-chunk loop. **This is the first ζ-fit ever
completed at 2406 centroids / P=144 by any tier.**

So the back-solve win (109 s/chunk × 14 chunks ≈ 25 min) **more than pays for
the eigh's extra 24 min**: the distributed tier is wall-neutral-to-slightly-
better end to end at this size, while never replicating the 13.81 GB/rank
factor. The owner's accepted "1.75× slower" turns out to be an eigh-only
figure that the tier earns back downstream — the honest headline is *not* a
tradeoff at c2406/P=144. (The eigh regression itself is a ScaLAPACK latency
property at `g = N/max(Px,Py) = 204`, one pure block per rank, unchanged by
this workstream and named in AF.7. It also varies run to run: 31 min in
7876062, 41 min here, same call and same size.)

### AF.4b — what the run found NEXT: a 6.99 TB `process_allgather` in the PHDF5_HOST **reader** (job 7876346, `compute_V_q`)

Job 7876346 completed the ζ-fit and then died — **downstream of this
workstream, in a different subsystem, and only because the fix let a run
reach that stage at this size for the first time ever.** Recorded here
because it is now the campaign's blocker.

    jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED:
      Out of memory allocating 6987250335744 bytes

The arithmetic identifies it exactly, to the byte:

    nq · μ_pad · ngkmax · 16 = 144·2448·8603·16 = 48 522 571 776 B/rank
    × P = 144                                   = 6 987 250 335 744 B  ✓

Call path (144/144 ranks, identical): `gw_jax.main` → `gw_init.
prepare_isdf_and_wavefunctions` → `gw_init.compute_V_q` → `compute_vcoul.
compute_all_V_q` → `v_q_g_flat.compute_all_V_q_g_flat` →
`_compute_V_q_g_flat_one_tile` → `v_q_g_flat.read_all_ibz` →
`zeta_loader.read_zeta_G_slab` → `slab_io.read_slab` →
**`_slab_io_mpi_host.read_slab`** → `jax.device_put` →
`multihost_utils.process_allgather`.

**The `144 x 144` in that number is `nq x P`, NOT `nq x nk` — and the
difference is the whole diagnosis.** The factorisation
`nq*nk*mu_pad*ngkmax*16` is numerically identical here because the run has
`nq = 144`, `nk = 144` (12x12 k-grid) **and** `P = 144` (12x12 mesh) — three
different 144s. It is the wrong one physically, and taking it at face value
sends the search into chi_0 / W / a missing k-chunk loop, none of which are
involved. The decisive evidence:

* the traceback's innermost frames are `jax/_src/api.py::device_put` ->
  `dispatch._device_put_sharding_impl` -> `multihost_utils.process_allgather`
  — JAX's **hidden `assert_equal`**, whose multiplier is `process_count()`,
  i.e. **P**, by construction;
* the object being gathered is **rank 3**, `(q_count, mu_count, n_G_sph)`.
  There is no k axis in it at all. `read_all_ibz` requests exactly
  `(n_q_ibz, n_rmu_padded, ngkmax)` and nothing else;
* no einsum, no contraction and no polarizability appears anywhere in the
  144 identical stacks — the deepest LORRAX frame is a **file reader**.

So there is no unsummed k axis, no XLA-materialised einsum, and no missing
k-chunk loop. There is also no new planner term to add: the object is the
ζ tensor the planner already models (AA.3's `F_tensor` term); what was
unmodelled was a `P x` multiplier on a **layout probe**, which is a defect
to delete rather than a cost to budget.

**Root cause: AA.1's antipattern, in a place AA.1 did not look.** To ask
"which hyperslab does this rank own?", the PHDF5_HOST reader built a zero
array of the **whole global shape** and `device_put` it onto the
multi-process `NamedSharding` purely to read
`.addressable_shards[0].index`. That materialises the full tensor on every
rank *and* trips JAX's hidden `multihost_utils.assert_equal`, i.e. a
`process_allgather` of `P ×` the global array. AA.1 fixed eight such call
sites in the loader, `PsiGStore`, `isdf_fitting` and `htransform`; this one
lives in the wk_AB reader, was written after that sweep, and is
**O(P × global tensor)** — so it is invisible at gate scale and lethal at
production scale. **Why 606c and 276c "passed": they never executed this code.**  Not a
threshold, not a different scaling — zero coverage.  `grep` for a `V_q`
line in every campaign log that also mentions `PHDF5_HOST`:

| log | PHDF5_HOST | reached `compute_V_q` |
|---|---|---|
| `wk_AB/slurm-7875943.out` (606c/P=16) | yes | **no** — writer + `zeta_q_G` bit-compare only |
| `wk_AB/probe606.7875719.out`, `zcmp606.7875735.out` | yes | **no** |
| `wk_AC/runs/dist_P144_FAILED` (7876062) | yes | **no** — died in the C⁺ Gloo collective |
| `wk_AC/runs/perq_P144_KILLED` (7876086) | yes | **no** — owner-killed at 3/15 chunks |
| **7876346 (this workstream)** | yes | **reached it, and died in it** |
| **7876423 (this workstream, fixed)** | yes | **passed it** |

Every earlier run either used the `H5PY_ALLGATHER` backend — a *different*
`read_slab` implementation that never had this probe — or died upstream of
`compute_V_q`. The PHDF5_HOST **reader** had **no production coverage at
any centroid count** until the ζ-fit started completing. That is the honest
reconciliation: the defect is `O(P × global tensor)` and would have been
fatal at 606c/P=16 too (12.05 GB × 16 = 193 GB in one allocation) — nothing
protected the smaller runs except never getting there.

**Fixed** (`src/file_io/_slab_io_mpi_host.py`): the probe is now
`sharding.addressable_devices_indices_map(read_shape)` — pure metadata, zero
allocation, zero collectives, and the same idiom
`common.collectives.device_put_process_local` already uses. Gate
`wk_AF/probe_equiv.py` asserts the new probe reproduces the old one's
`local_shape` and `shard_offset` **exactly** on a real 2×2 mesh across
sharded / replicated / flat-tuple `('x','y')` specs: **ALL AGREE**.
Re-run in `AC_MODE=vq` (the ζ is complete on disk, so the job re-enters at
`compute_V_q`) as job **7876423** — **and it clears the stage**:

    REUSING the existing ζ at .../tmp/zeta_q.h5 — FIT SKIPPED.
    isdf_header/zeta_is_done is True, the centroid table …
    V_q g-flat [CC]: n_q_ibz=144, ngkmax=8603, g_chunk=8603 (1/q),
      n_rmu_L=2406→2448, n_rmu_R=2406→2448, unfold=IBZ→full
    [CC] q=1/144 … q=144/144: kernel ≈ 0.87 s each
    V_q computed:  Shape: (144, 2448, 2448)   V_q=0 trace: 237075709.4403

144 q-points at ≈0.87 s/kernel, node-0 memory **flat at 9 GB**, zero
`RESOURCE_EXHAUSTED`, zero `FAIL-FAST`. Two independent confirmations the
ζ from 7876346 is sound and complete: `zeta_is_done is True` with a
provenance match, and `compute_V_q` consumed all 144 q-slabs off it without
error. **This is the first time the campaign has ever got past `compute_V_q`
at 2406 centroids / P=144.**

**Gates on the reader fix:**

| gate | result |
|---|---|
| `wk_AF/probe_equiv.py` — new probe vs old probe, real 2×2 mesh, sharded / replicated / flat-`('x','y')` specs | **ALL AGREE** on `local_shape` and `shard_offset` |
| fixture eqp, P=4, `auto` route, vs `eqp_ref.dat` @1e-3 | **PASS, max\|Δ\| = 1.00e-06 eV**, 0/1888 |
| fixture eqp, P=4, **before vs after the reader change** (cell `P4rep` vs `P4rdr`, identical deck and route) | **max\|Δ\| = 0.00e+00** — no collateral change |
| **P=144 `AC_MODE=vq` (job 7876423) past the death point** | **PASS** — `compute_V_q` completed, 144/144 q at ≈0.87 s, node memory flat at 8–9 GB, zero `RESOURCE_EXHAUSTED`, zero `FAIL-FAST` |

### AF.4c — 7876423's OUTCOME: TIMEOUT, and a THIRD wall — the restart-tensor write is silent for ~3 h

**The job did not finish.** `sacct`: `State=TIMEOUT, Elapsed=03:00:04`
(the 3 h wall I set), `MaxRSS = 4 197 060 K = 4.2 GB/rank`. It was killed by
Slurm, not by an error: **zero** `RESOURCE_EXHAUSTED`, **zero** `FAIL-FAST`,
**zero** `Gloo`/`JaxRuntimeError` across 144 ranks, node-0 memory flat at
8–11 GB the whole time. So the AF gates stand and nothing regressed — but
the run stopped short of the stage table.

**Where the 3 h went — resolved exactly, by the instrument in AF.6.** The
last content line is `V_q=0 trace: 237075709.4403` at ≈00:20. From then until
the SIGTERM at 03:15 — **2 h 55 m** — the run emitted **not one line**. But
the new per-dataset log, run on the fixture, gives the write ORDER and the
shape of every dataset, and that turns `isdf_tensors_2406.h5`'s three size
jumps into an exact identification (agreement to <1 KB on all three):

| file jump | bytes | identified as | law | data? |
|---|---|---|---|---|
| →13.338 GB (≈00:21) | 13 338 016 352 | `V_qmunu` | `nq·n_rmu²·16` = 13 337 478 144 | **written** |
| →26.675 GB (≈01:21) | +13 337 480 272 | `W0_qmunu` **placeholder** | same law | **NO — `create_dataset` only** |
| →31.465 GB (03:16) | +4 789 517 912 | `psi_full_y` | `nk·nb·2·n_rmu·16` = 4 789 518 336 | **written** |

Three consequences, and the middle one corrects what I wrote an hour ago:

1. **The 26.68 GB reading was NOT "two tensors landed".** The second jump is
   `init_W0=True`'s zero-data pre-allocation — parallel HDF5 sizing the
   dataset at create time. That is AC.3b's trap wearing a different hat, and
   I walked into it in this very entry before the instrument existed. The new
   log now prints `W0_qmunu placeholder ALLOCATED … (no data written)` so
   nobody reads it as progress again.
2. **`G0_mu_nu` is tiny, not a 13 GB tensor** — the fixture log shows
   `G0_mu_nu (60,)`, i.e. shape `(n_rmu,)`. The "partway through writing G0"
   hypothesis was wrong.
3. **The run had progressed FURTHER than the first restart write.**
   `psi_full_y` is written by a *later* `write_restart_state_to_h5(mode="a")`
   call, so the job was past `write_restart_state_to_h5`'s first invocation
   and inside the wavefunction-bundle region when the wall hit.

**The actual rate: 18.13 GB of real data (13.34 + 4.79) in 2 h 55 m ≈
1.7 MB/s aggregate across 144 ranks.** That is the defect signature. It is
not a memory problem (MaxRSS 4.2 GB/rank) and not a transport error (zero
Gloo markers); it is a write path that is ~3 orders of magnitude off what
the same job's ζ writer achieved (47.7 GB with `h5_write` at 5 ms/chunk).
Which of `create_dataset`'s eager allocation, the per-rank hyperslab write,
or the file-open/close cycle owns that is the one thing still open — and the
instrument now measures each of them separately, per dataset, in MB/s.

**The operational hazard, for the third time in this campaign.** AC.2 named
it for the 30-minute silent `pzheevd`; this is the same failure of
observability at 6× the duration. A stage that runs for hours and prints
nothing is indistinguishable from a hang, and AC.3c's discriminator
("progressing = a cadence exists; wedged = no first milestone") **cannot be
applied here at all**, because this stage has no milestone to emit. The
file-size signal that did work is exactly the one AC.3b warns is a trap in
the other direction. **The first thing the next workstream should add is a
progress line per dataset in `write_restart_state_to_h5`** — it is cheaper
than another 72-node hour and it is what would have made this run
diagnosable while it was alive.

**Handoff status.** Workstream **AI** (job 7876530,
`wk_AI/runAI_close.sbatch`) picked this up and is running in the same run
dir; as of 04:11 it has produced 144 per-rank
`tmp/isdf_tensors.rank<N>.x<i>.y<j>.h5` files (≈502 MB each), i.e. a
file-per-rank write in place of the single collective dataset. Its outcome
is AI's to report, not AF's. **The complete ζ survived and is what AI is
building on** — `tmp/zeta_q.h5`, 47 712 355 497 B, `zeta_is_done=True`,
produced by 7876346's 15/15 fit. That artifact is the durable output of this
workstream.

**Recommendation (still standing): do NOT simply resubmit with a longer wall.** A stage the
owner budgeted at ≈35 min (V_q + screening + Σ together) did not finish in
2 h 55 m; that is a defect signature, not a slow-but-healthy one, and
throwing 8 h at it repeats the mistake AC.3c documents. `AC_MODE=restart`
is also not an escape — it loads `tmp/isdf_tensors*.h5`, which this run left
**incomplete**. The ζ (`tmp/zeta_q.h5`, `zeta_is_done=True`) IS complete and
reusable, so `AC_MODE=vq` remains the right re-entry once the writer is
instrumented.

This is flagged as **out of AF's original scope and inside it by
consequence**: AF's mandate was the ζ tier's collectives, and the ζ tier's
gate passed. The reader defect is reported rather than absorbed into AF's
headline.

### AF.5 — BONUS (NOT part of the fix): what is actually tunable about JAX's Gloo transport (probe `gloo_probe.py`)

**None of this is load-bearing.** The shipped fix touches no transport
setting and works identically with every knob below left alone; this section
exists because the brief asked for the probe, and because its answer explains
*why* an architecture-neutral fix was the only real option. Ran against the
production jaxlib (0.9.1):

* **`GLOO_SOCKET_IFNAME` does not appear ANYWHERE in the shipped jax /
  jaxlib** (`strings` over every `.so` in `site-packages/jaxlib/` and
  `jax_cuda12_plugin/`, plus a grep over the Python tree: **zero hits**).
  The interface is instead an *argument*:
  `make_gloo_tcp_collectives(distributed_client, hostname=None,
  interface=None)` — and `xla_bridge.make_cpu_client` passes **neither**,
  with no config flag to reach them. So the `GLOO_SOCKET_IFNAME=ib0` line
  every sbatch in this campaign carries is, for JAX's CPU collectives in
  this build, **inert**. (It is not harmful, and other components may read
  it; it is simply not the knob it is believed to be.)
* The only `GLOO_*` env strings present in the binary are
  `GLOO_DISABLE_CONNECTION_RETRIES`, `GLOO_ENABLE_RANK_AS_SEQUENCE_NUMBER`
  and `GLOO_ENABLE_STORE_V2_API`. **None of them changes payload handling**,
  and the first would make the observed failure *worse* (it disables the
  connection retries that are currently masking some of it).
* There is **no timeout, no buffer-size, no chunk-size and no algorithm
  knob** exposed for the Gloo path — no env var, no `jax.config` entry, no
  keyword. `gloo::Context::getTimeout` and `tcp::Socket::{send,recv}Timeout`
  exist in the library but nothing reachable from JAX sets them.

> **⚠ REFUTED (AK.4/AK.10/AL, 2026-07-27): "the transport offers no dial" is WRONG — the dial exists and is now first-class runtime code.** `make_gloo_tcp_collectives(..., interface=)` takes the NIC as a constructor argument; jax 0.9.1 simply never passes it, and re-registering the CPU backend factory reaches it (`runtime.pin_gloo_interface()`, override `LORRAX_GLOO_IFNAME`, landed by AL). The factual substrate of this section stands (GLOO_SOCKET_IFNAME truly is absent from the binaries and inert; no env/config knob exists) — the *conclusion* below does not. Corollary: AF.1's 128 MB payload bound was measured against a saturated 1 GbE link and should be re-priced on ib0 before being treated as a permanent constraint (AK.10 consequence 2).

⇒ **the transport offers no dial; the only variable under our control is the
payload.** Which is the happy outcome, not a constraint to regret: the one
lever that exists is also the portable one. A fix expressed as "emit smaller
collectives" is a property of the program and survives any change of fabric,
backend or machine; a fix expressed as "set this env var" would not have.

**Approach 3, priced exactly.** `jax_cpu_collectives_implementation` accepts
`"mpi"` and `jaxlib._jax.make_mpi_collectives` **is present and
constructible** — but `Init()` fails with

    MPItrampoline: ERROR: The environment variable MPITRAMPOLINE_LIB is not set.

i.e. jaxlib's MPI backend is built against **MPItrampoline**, so making it
live needs an MPIwrapper shim built against Frontera's Intel MPI and
`MPITRAMPOLINE_LIB` pointed at it — a build task, not a config flip. It is
attractive on paper (it would route **every** JAX CPU collective through the
same Intel MPI that `pzheevd` already drives cleanly at P=144) and it is the
right thing to reach for if a payload-bounded collective is ever still seen
to fail. **It was not needed, not built, and forms no part of the delivered
fix** — recorded only so the next person does not repeat the investigation.

### AF.6 — files touched (worktree only, NOT committed)

**Branch state:** the five files below were reviewed and committed by the
orchestrator as **`c62c898`** on `gloo-robust-distributed` (I did not commit;
verified intact — `_DEFAULT_COLLECTIVE_CHUNK_MB`/`_chunk_q` and
`addressable_devices_indices_map` both present in the commit).

**Provenance boundary, as of 04:15.** wt-J is now being edited concurrently
by workstream **AI**, which is closing the AF.4c write wall and has a live
144-node job (7876530) against this tree. Uncommitted in wt-J right now:
`src/common/timing.py`, `src/file_io/_slab_io_ffi.py`,
`src/file_io/_slab_io_mpi_host.py` and `src/file_io/tagged_arrays.py`
(298 insertions total) — of which **only the ~37-line `restart_write`
instrumentation in `tagged_arrays.py` is AF's**; the rest is AI's. AF's
committed work was re-verified intact after those edits (`_chunk_q` ×8 in
`isdf/core.py`, `addressable_devices_indices_map` ×2 in
`_slab_io_mpi_host.py`). Reviewers should split that diff by author, not
treat it as one change.

`src/file_io/tagged_arrays.py` — **NEW, AF.4c**: one rank-0 progress line per
dataset in `write_restart_state_to_h5` (`name`, shape, GB, seconds, MB/s),
plus an explicit `W0_qmunu placeholder ALLOCATED … (no data written)` line so
the AC.3b file-size trap cannot be misread as progress. ON by default
(`LORRAX_RESTART_WRITE_LOG=0` silences); print-only, no semantic change.
This is the instrument whose absence made 7876423's 2 h 55 m undiagnosable;
running it on the fixture is what produced AF.4c's exact identification.
Gated: fixture eqp **PASS 1.00e-06 eV** vs `eqp_ref.dat` and **0.00e+00** vs
the pre-instrumentation run on the identical deck — print-only, inert.

`src/file_io/_slab_io_mpi_host.py` — the shard-layout probe in `read_slab`
(AF.4b): full-global-shape `jnp.zeros` + `device_put` onto a multi-process
sharding → `sharding.addressable_devices_indices_map`, pure metadata.
`src/isdf/core.py` — `_DEFAULT_COLLECTIVE_CHUNK_MB` / `_collective_chunk_bytes`
/ `_chunk_q` / `_chunk_log` + the AF rationale block; the C⁺ formation
restructured into `_masks` (spectrum + truncation + telemetry, unchanged
maths) and a q-blocked `_block` loop; `_distributed_pinv_apply`'s q-batch
now the `min` of the memory cap and the collective cap.
`tests/test_zeta_mesh_invariance.py` — +1 cell.
No behaviour change at any scale where the cap does not bite, by
construction and confirmed in HLO.

### AF.7 — named, deliberately NOT done

* **The ζ output reshard is not chunked.** `_reshard_zeta_mu_X_r_Y_to_mu_XY`
  moves the whole `(nq, μ_pad, r_chunk)` tensor in one all-to-all — 456 MB
  per rank at c2406/P=144, above the 128 MB cap. Left alone on evidence,
  not oversight: (a) `all-to-all` does **not** appear in 7876062's error
  census at all, and (b) the healthy `per_q` run 7876086 paid a strictly
  larger version of the same movement (two staged all-to-alls of the same
  tensor) on the same 144 ranks without incident. Chunking it would mean
  touching the one reshard this codebase has lost the most time to (J.9,
  T.4), for no measured benefit. If an all-to-all ever does fail at scale,
  the same `_chunk_q` helper applies to it unchanged.
* **`pzheevd`'s 1.75× is untouched.** AC.2 measured ≈12 s/matrix on the
  12×12 grid against the replicated route's ≈7 s. That is a ScaLAPACK
  latency property at `g = N/max(Px,Py) = 204` (one pure block per rank,
  latency-bound tridiagonal reduction), not a transport one, and the owner
  has explicitly accepted it. A block-cyclic descriptor with more than one
  block per rank is the lever if it is ever wanted; it is a descriptor
  change in `eigh_ffi.cc`, not a JAX-side one.
* **No capability probe for the cap.** `LORRAX_COLLECTIVE_CHUNK_MB` is a
  static default, not a measured one. The honest statement is that 128 MB
  is bracketed by two production measurements (0.104 GB good, 1.15 GB
  fatal) at P=144 — it is not a curve.

## AG — the rtx `load_centroid_wfns` hang, root-caused and killed: the JAX persistent compile cache is a P>1 GPU DEADLOCK (wt-F, branch `rtx-centroid-hang`, base 6431782, 2026-07-27 — NOT committed)

**One line: the hang is not in the WFN loader, the phdf5 FFI, MPI, NCCL or
anything AE touched — it is the JAX persistent compile cache, which JAX
writes from process 0 ONLY while `common/jax_compile_cache.py` partitioned
it per rank, so `np4/rank0` holds 882 entries and `np4/rank{1,2,3}` hold
ZERO, forever; process 0 then hits the cache and skips compilation while its
peers block in `xla::gpu::AutotunerPass`' cross-process key-value exchange
until the job is reaped. AE's exact hanging cell now COMPLETES in 165 s with
eqp `max|Δ| = 1.00e-06 eV`, and the guard that fixes it prints the reason.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AG/` — `probe_load.py`,
`hangdump.py`, `ag2.sbatch`, `ag3.sbatch`, `ag4_gate.sbatch`, the stack dump
`stacks_ae_exact.txt`, and job logs `ag2.7876372.out`,
`ag3.7876375.out.keep`, `ag4.7876422.out` (+ `run_*/rank{0..3}.log`,
`g_*/rank*.log`).  Jobs **7876369** (first probe), **7876372** (loader
probe), **7876375** (the bisect + the stack), **7876422** (the gate).

### AG.1 — THE STACK (job 7876375, cell `ae_exact`, 4× Quadro RTX 5000)

AE's recipe reproduced first try: `rc=124` at 901 s, **zero** `Zeta fitting`
lines, last output the ISDF planner — byte-for-byte AE's symptom.  `ps` at
t=300 s: **one rank at 70.3 % CPU, three at 2.5 %**; `nvidia-smi`: all four
GPUs **0 % utilisation**.

Python (faulthandler, all threads, per rank):

    rank 0   wfn_transforms.py:2108  load_centroids_band_chunked   <- _reshard_all
             pxla.py:1361 MeshExecutable.__call__                  (EXECUTING)
    rank 1   wfn_transforms.py:1663  load_psi_gflat_padded         <- loader.load
    rank 2   pxla.py:2844 _cached_compilation                      (COMPILING)
    rank 3   ...same...

Process 0 is a whole pipeline stage AHEAD of its peers.  The C stack of a
parked rank (host `eu-stack`, ptrace_scope=0) says why — verbatim, frames
#5-#13:

    #5  absl::Notification::WaitForNotificationWithTimeout
    #6  xla::CoordinationServiceAgent::GetKeyValue
    #7  xla::DistributedRuntimeCoordinationServiceClient::BlockingKeyValueGet
    #8  xla::DistributedKeyValueStore::Get
    #11 pjrt::CApiKeyValueStore::Get
    #12 xla::Autotuner::Autotune(xla::HloModule*, ..., xla::MultiProcessKeyValueStore&)
    #13 xla::gpu::AutotunerPass::RunImpl
    ... GpuCompiler::RunHloPasses -> PjRtStreamExecutorClient::Compile

**XLA:GPU compilation is a COLLECTIVE.** `AutotunerPass` shards the
autotuning across processes and exchanges the results through the JAX
coordination service.  A process that does not compile never publishes its
share, and every peer blocks in `GetKeyValue` forever.

### AG.2 — why process 0 didn't compile: JAX writes the cache from rank 0 only

`jax/_src/compiler.py::_cache_write`, unconditional, not a LORRAX choice:

    # Only write cache entries from the first process. Otherwise we create
    # problems with contention for writes on some filesystems, e.g., GCS.
    if distributed.global_state.process_id != 0:
      return

`common/jax_compile_cache.py` nests entries under `{base}/np{P}/rank{i}/`
"so each rank reads/writes its own dir" — and its docstring claimed "the
per-rank caches converge within one warm run".  **They do not and cannot.**
Measured at the top of job 7876375, three days after those dirs were created:

| dir | entries |
|---|---|
| `~/.cache/isdf_jax_compilation/np4/rank0` | **882** |
| `np4/rank1` | **0** |
| `np4/rank2` | **0** |
| `np4/rank3` | **0** |

The same shape holds at every world size the campaign has ever used — from
the same job's exit dump: `np8/rank0` = **107**, `np8/rank{1..7}` = **0**.

Every P>1 GPU run is therefore one cached module away from a permanent hang:
the hit/miss pattern must be IDENTICAL on every rank for the autotune
exchange to close, and this layout guarantees it never is.

### AG.3 — the config delta AD-vs-AE, resolved

The scoping hypotheses (`use_ffi_io`, the phdf5 CUDA READ path, `--mpi=pmi2`,
the shared-vs-unified `.so`, `CUDA_VISIBLE_DEVICES`) are **all innocent** —
each was tested and each is in the *passing* column below.

| axis | AD GPU cells (7876110) PASSED | AE GPU cells (7876117/…/7876349) HUNG | verdict |
|---|---|---|---|
| **`ISDF_JAX_CACHE_DIR`** | **`""` (in AD's ENVBLOCK)** | **unset → `~/.cache/isdf_jax_compilation`** | **THE TRIGGER** |
| geometry | 2 nodes × 2 ranks | 1 node × 4 ranks | innocent (the §10 July-23 campaign ran 1×4 fine) |
| MPI bring-up | plain `srun` | `srun --mpi=pmi2` | innocent |
| `ffi_env.sh` | not sourced | sourced | innocent |
| `LORRAX_FFI_SO*` / `LORRAX_FFI_PHDF5` | unset | set | innocent — but it silently flips the loader backend (AG.5) |
| loader backend (auto) | `phdf5_host` | `phdf5` (CUDA FFI) | innocent — see AG.4 |
| `use_ffi_io` / `slab_io` | `false` / `h5py_allgather` | `true` / `phdf5_ffi` | innocent |
| `NCCL_IB_DISABLE` / `NCCL_SOCKET_IFNAME` | set | unset | innocent |

The early-campaign recipes (§10, `gw_ffi_dist.sbatch`) are AE's recipe
exactly — 1 node, `-n 4`, `--mpi=pmi2`, `ffi_env.sh`, `use_ffi_io=true` — and
they worked because the cache dirs were **created that same night** (mtime
03:57, first runs 03:5x–05:13) so every rank missed symmetrically.  The
asymmetry is what accumulates, not the recipe.

**The causal cell, one variable, back-to-back on the same node** (job 7876375):

| cell | ISDF_JAX_CACHE_DIR | rc | wall | `Zeta fitting` | eqp |
|---|---|---|---|---|---|
| `ae_exact` | *unset* | **124** | **901 s (killed)** | **0 / 4 ranks** | **none** |
| `ae_nocache` | `""` | **0** | **70 s** | **4 / 4 ranks** | **PASS 1888 values, max\|Δ\| = 1.00e-06 eV** |

Everything else identical: same tree, same `.so`, same `--mpi=pmi2`, same
`use_ffi_io=true slab_io=phdf5_ffi`, same node.

### AG.4 — the phdf5 CUDA FFI read is NOT the bug (it was the prime suspect)

A minimal probe (`wk_AG/probe_load.py`, job 7876372) drives ONLY the read
path — `WfnLoader(mesh=2×2)` → `box_index_dev` → `_ensure_phdf5_static`
(collective `H5Fopen` + `MPI_Init_thread`) → `loader.load(bands=(0,48),
k='full_bz')` — under AE's exact launch, backend pinned to `phdf5`:

    _ensure_phdf5_static  0.80 s      loader.load  2.37 s      checksum OK
    PROBE_DONE_OK

The collective MPI-IO read, the FFI's `MPI_Init` under `--mpi=pmi2`, and the
CUDA staging tail are all **fine**.  So is the loader auto-pick: it resolves
to `phdf5` correctly and the FFI is genuinely usable.  No refusal was needed
and none was added — the mission's fallback branch is not the answer here.

**Second, real, latent trap found on the way** (job 7876369): `ffi_loader.get_lib`
dlopens the FFI `RTLD_GLOBAL`, which publishes the site's Intel parallel
HDF5 (`libhdf5.so.310` = 1.14.6) into the global namespace.  h5py ships its
OWN ABI-incompatible HDF5 (`h5py.libs/libhdf5-*.so.320` = 2.0.0).  Probe the
FFI **before** `import h5py` and h5py dies at import with

    ValueError: Not a datatype (not a datatype)

Production imports h5py first by luck of ordering.  `get_lib` now imports
h5py before the `CDLL`, making the safe order unconditional.

### AG.5 — THE FIX

1. **`src/common/jax_compile_cache.py`** — `ensure_jax_compile_cache()`
   REFUSES the persistent cache at `jax.process_count() > 1` and prints, on
   process 0, the mechanism, the cost ("compiles from scratch on every rank,
   which is CORRECT and ~1 min slower"), and both escape hatches
   (`ISDF_JAX_CACHE_DIR=""` to silence, `LORRAX_JAX_CACHE_MULTIPROCESS=1` to
   re-arm).  The module docstring carries the full derivation with the JAX
   source quoted, so nobody re-derives it.  **P=1 is untouched** — the cache
   is safe and useful there and still arms.
   This makes the campaign's out-of-band convention (every CPU launcher
   exports `ISDF_JAX_CACHE_DIR=""`; no GPU launcher did) an in-tree,
   process-count-aware property.
2. **`src/file_io/wfn_loader.py`** — the backend auto-pick is announced
   once per (backend, world) on rank 0.  It is a real fork in behaviour
   selected by whether an `.so` happens to be on `LD_LIBRARY_PATH`, AD and
   AE differed on it, and **neither log said so**.
3. **`src/ffi/common/ffi_loader.py`** — `import h5py` before the
   `RTLD_GLOBAL` `CDLL` (AG.4).
4. **`config/frontera/ffi_env.sh`** — a comment saying why it deliberately
   does NOT export `ISDF_JAX_CACHE_DIR` (the in-tree guard is
   process-count-aware; exporting `""` here would kill the P=1 win).
5. **`docs/dev/env_vars.md`** — both rows rewritten with the real mechanism.

### AG.6 — GATES (job 7876422, rtx-dev, wt-F @ `rtx-centroid-hang`)

| gate | cell | result |
|---|---|---|
| **AE's previously-hanging cell completes** (1 node × 4, `--mpi=pmi2`, `ffi_env.sh`, `use_ffi_io=true slab_io=phdf5_ffi`, **no** cache opt-out) | `fixed_ae_exact` | **rc=0, 165 s** (was rc=124 @ 901 s) — refusal line printed, backend announced |
| fixture eqp on GPU ≤ 1e-6 | `fixed_ae_exact` | **PASS 1888 values, max\|Δ\| = 1.00e-06 eV, max rel 5.29e-09, 0 over tol at atol 1e-06** |
| same, allgather-writer twin | `fixed_ae_base` | **rc=0, 54 s, PASS 1.00e-06 eV** |
| **AD's passing cells still pass** (2 nodes × 2, AD's ENVBLOCK verbatim, no `ffi_env.sh`, no `pmi2`) | `fixed_ad_like` | **rc=0, 119 s, PASS 1.00e-06 eV**; announce confirms AD really ran `phdf5_host` |
| **P=1 cache still arms** (must NOT refuse) | `p1_cache_on` | **rc=0, 34 s**, `np1/rank0` **475 → 726 entries**, refusal-line count **0**, eqp **PASS 1.00e-06 eV** |
| **POSITIVE CONTROL — re-arm the cache and the hang MUST come back** (`LORRAX_JAX_CACHE_MULTIPROCESS=1`, otherwise identical to `fixed_ae_exact`) | `poscontrol` | **rc=124, STALLED at 781 s**, no refusal line printed — the guard is what is doing the work, not luck |
| **BONUS — AE.4b's blocked GPU e2e writer gate** — `zeta_q.h5` from the CUDA phdf5_ffi writer vs the allgather writer, 4 RTX GPUs | `fixed_ae_exact` vs `fixed_ae_base` | **`H5_BITCMP_OK`** |

No C++ was touched, so AE's CPU-side unit bit-compares are unaffected by
construction; a CPU-mode unit cell (`test_wfn_loader_eager.py`,
`test_sanity_gates.py`) runs in the same job.  The only CPU-visible change is
that a P>1 CPU run which forgot `ISDF_JAX_CACHE_DIR=""` now prints the
refusal instead of silently arming a cache it must not have — every campaign
CPU launcher already sets it, so behaviour is unchanged there.

### AG.7 — named, not done

* **The cache is refused at P>1, not repaired.** Repairing it needs process-0's
  entries to be *readable* by every rank AND the hit/miss pattern to be
  *identical* on every rank.  A single shared dir does not achieve the second:
  JAX's cache key hashes the per-rank device assignment, so the peers still
  miss (that mismatch is exactly why the per-rank partitioning was introduced).
  `jax_share_binary_between_hosts` does not help either — it makes the peers
  wait on process 0 to publish, which is the same deadlock when process 0 hit
  the cache.  A real fix is a LORRAX-side barrier that agrees on hit-or-miss
  before compiling.  Until then P>1 pays the compile — and the compile storm
  (context brief, ~138 modules/rank) remains the standing lever it always was;
  this workstream explains why the "obvious" cure was never safe.
* **The stale `np4/rank0` cache (882 entries) is still on disk.** Harmless now
  (refused), but it is 5 MB of a trap; whoever repairs the cache should clear it.
* `jax.process_count()`-aware refusal is evaluated inside
  `ensure_jax_compile_cache`, i.e. after `jax.distributed.initialize`. Every
  caller already satisfies that (`runtime.bootstrap()` runs first); a driver
  that called it earlier would see `n_proc == 1` and arm the cache.
* Only the 60-centroid fixture at P=4 was gated on GPU. The fix is a launch-time
  policy switch and is scale-independent, but that is reasoning, not a measurement.

## AH — the P>1 persistent compile cache REPAIRED: a process-invariant key + a coordination-service hit/miss agreement (wt-AH, branch `mp-compile-cache`, base 7648fd4, 2026-07-27 — NOT committed)

**One line: the campaign's oldest deferred lever is closed. AG's refusal is
replaced by a working shared cache — the key is made process-invariant (the
per-rank component is `accelerator_config`, NOT the device assignment AG.7
guessed), the ranks agree on the usable entry set over the coordination-service
KV store before anything compiles, writes are made atomic, and JAX's own
rank-asymmetric XLA sub-caches are turned off. AG's exact stall reproducer now
runs in 20 s against 73 s cache-off (3.6x); CPU P=8 htransform goes from 152
compiles/rank to ZERO; every cell rc=0 with eqp max|Δ| = 1.00e-06 eV or
bit-exact bandstructure. The honest counterweight, also measured: at 606
centroids / P=16 the entire compile storm is ~5 s of XLA per rank, so on CPU
the repaired cache is worth ~1 % of a 431 s run — and reading its 876 kB of
tiny entries serially costs 29 s, which is why a threaded prefetch and a
single-stripe layout ship with it. The lever is real on GPU and small on CPU;
it is now CORRECT everywhere, which it was not before.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AH/` — `keyprobe.py`
(the cache-key forensics), `ah1_cpu.sbatch`, `ah2_gpu.sbatch`,
`ah3_cpu_prize.sbatch`, `ah4_gpu22.sbatch`, `ah5_prefetch.sbatch`,
`probe_ah.sh` (a copy of wk_Y's `probe.sbatch` patched only to honour
`PROBE_CACHE` / `PROBE_RUNDIR` / `PROBE_PORTOFF` / `PROBE_EXTRA_ENV`), and the
job logs `ah{1..5}.<jid>.out`. Jobs **7876434** (CPU matrix), **7876437**
(rtx-dev: key forensics + 1x4 GPU gate), **7876440** (the prize, interleaved +
606c/P=16), **7876443** (2x2 GPU + the node-local A/B), **7876461** (prefetch A/B +
unit suite).

### AH.1 — WHAT WAS ACTUALLY WRONG WITH A SHARED DIRECTORY (three things, all measured)

AG.7 named one suspect ("JAX's cache key hashes the per-rank device
assignment"). The direction is right and the component is wrong, and there are
two more failure modes underneath it. `wk_AH/keyprobe.py` runs 4 ranks, records
every key JAX computes, and turns on `jax._src.cache_key`'s DEBUG logging so the
per-COMPONENT cumulative hashes can be diffed:

| probe (job 7876437) | backend | keys identical on all 4 ranks? |
|---|---|---|
| `kcpu_raw` | CPU | **NO** — first diverging component is **#5 `accelerator_config`**; #4 `compile_options` MATCHES |
| `kgpu_raw` | CUDA | yes |

So on GPU jax already strips what varies (`strip_device_assignment=(backend.
platform == "gpu")`, with the comment "In case of GPU multi-process tasks we
need to strip device assignment to use cache key as invariant between
processes"), but on **CPU** the divergence is in
`_hash_accelerator_config`, which hashes
`xla_client.get_topology_for_devices(devices).serialize()` — a blob carrying
process-local content. The device assignment is innocent on both platforms.

Consequence: with a shared dir and no fix, **process 0 — the only rank JAX lets
write — is the only rank that can ever hit.** Measured directly (`ah1`, cell
`k4_warm`): rank 0 `hits=7 vetoed=0`, ranks 1-3 `hits=0 vetoed=7 compiles=7`.
That is not merely "no win": it is exactly the divergent hit/miss pattern that
deadlocks XLA:GPU's collective autotune.

Two more, from reading jax 0.9.1:

* **`LRUCache.put` is NOT atomic.** `jax/_src/lru_cache.py` does
  `cache_path.write_bytes(val)` with no tmp+rename (the mission brief assumed
  otherwise). A concurrent reader can get a truncated entry; `_cache_read`
  swallows the exception into a MISS, and the ranks diverge.
* **JAX auto-enables a SECOND, rank-asymmetric cache.** With the persistent
  cache on, `compiler.py::get_compile_options` points XLA at
  `{cache_dir}/xla_gpu_per_fusion_autotune_cache_dir` with
  `AutotuneCacheMode.UPDATE` on process 0 and `READ` on the peers
  (`jax_persistent_cache_enable_xla_caches` defaults to exactly that one). That
  changes the set of fusions each process still has to autotune — i.e. the
  input to `AutotunerPass`' modulo-P work split — so it can desynchronise the
  same exchange one level down.

And `jax_share_binary_between_hosts` is not an alternative: `compile_or_get_
cached` returns early on a hit, so the share path is only reached on a MISS —
when process 0 hits and its peers miss, the peers block forever on a
publication that never comes (AG.7's reading, confirmed by the source). It also
uses the cache key as the KV key, so on CPU, where the keys differ per rank, it
would deadlock even on a symmetric cold miss.  (Both readings are from the jax
0.9.1 sources; neither was run to a hang here, because the agreement makes the
option moot.)

### AH.2 — THE DESIGN (`src/common/jax_compile_cache.py`)

1. **ONE shared directory per world size**, `{base}/np{P}/`. The `rank{i}/`
   partitioning that made AG's asymmetry permanent is deleted.
2. **`_install_invariant_key_patch()`** — do for every platform what jax does
   for GPU: force `strip_device_assignment=True`, and replace the serialized-
   topology hash with a canonical `platform:count:device_kinds:host_target`
   string. `host_target` is `platform.machine()` + the `/proc/cpuinfo` model
   name, so entries built for a different CPU model stay apart (that is the
   ORIGINAL "poisoned cache" failure of ADVICE §4, which the topology blob used
   to cover). **Load-bearing for safety, not just speed** — if it cannot be
   installed, the cache is switched off rather than used.
3. **A snapshot hit/miss agreement**, O(1) RPCs per rank per RUN, taken inside
   `ensure_jax_compile_cache()` after `jax.distributed.initialize`:
   `p0 set {ns}/keylist` (the sorted keys it can see) → all get it → each rank
   publishes a presence BITMASK over that list (882 entries = 111 bytes) →
   `p0` ANDs them and publishes the result → `wait_at_barrier` commits. Every
   later cache probe is answered from that frozen intersection, so hit/miss is
   identical on every rank **by construction**, and entries process 0 writes
   *during* the run stay invisible until the next one — which closes the
   torn-read race in (1) above without needing a per-module protocol.
4. **A key-environment fingerprint** rides along with each mask (`XLA_FLAGS`
   minus jax's own excluded-from-key list, jaxlib version, cache config, host
   target). Any mismatch → the cache is turned off on every rank, loudly. This
   catches the trap wk_Y's own probe harness sets: a rank-0-only `XLA_FLAGS`
   for an HLO dump makes rank 0 compute different keys, so rank 0 MISSES while
   its peers HIT — AG's deadlock with the roles reversed.
5. **Atomic writes** — `get_file_cache` is replaced so `put` goes through a
   temp file + `os.replace`. Writes stay process-0-only (jax's rule; under SPMD
   process 0 compiles the same module set as everyone, so its writes cover it).
6. **`jax_persistent_cache_enable_xla_caches=""` at P>1**, killing (3) above.
7. **Graceful degradation, never a hang.** Missing client, KV/barrier failure,
   timeout, non-invariant key, env mismatch → agreed set is EMPTY, every rank
   misses and compiles (always correct), reason printed on every rank. An entry
   that WAS agreed but then fails to load would re-introduce divergence, so
   that path aborts the process loudly (`LORRAX_JAX_CACHE_STRICT=0` downgrades).
8. Instrumentation: an `atexit` one-liner per rank —
   `xla_compiles=N (Ts) cache_probes=P hits=H (Ts) vetoed=V agreed=A/S` — armed
   on EVERY path including cache-off, so "with" and "without" are the same
   measurement. `compile_cache_stats()` exposes it programmatically.

`LORRAX_JAX_CACHE_MULTIPROCESS` is reconciled: it was `0`/off (the escape hatch
that "re-armed the deadlock"); it is now `1`/on, the default path, and `=0`
restores AG's refusal verbatim for bisecting.

### AH.3 — GATE 1: NO DEADLOCK, COLD AND WARM, CPU AND GPU

Every cell rc=0.  eqp gate = `multidev_compare.py` vs `eqp_ref.dat` at 1e-6;
htransform gate = last column vs `bs_groundtruth_meshless.dat`.
"compiles/rank" is this workstream's `atexit` counter (real
`backend_compile_and_load` calls), armed identically with the cache on and off.

| gate | cell (job) | geometry | rc | wall | compiles/rank | correctness |
|---|---|---|---|---|---|---|
| CPU P=8 cold | `gw8_cold` (7876440) | 2 nodes x 4 | 0 | 27 s | 373 (9.5 s) | eqp **1.00e-06 eV** |
| **CPU P=8 warm** | `gw8_warm2` | 2 nodes x 4 | 0 | 19 s | **5** (0.7 s), 368 hits | eqp **1.00e-06 eV** |
| CPU P=4 cold | `gw4_cold` | 1 node x 4 | 0 | 65 s | 357 (9.1 s) | eqp **1.00e-06 eV** |
| **CPU P=4 warm** | `gw4_warm` | 1 node x 4 | 0 | 14 s | **3** (0.4 s), 354 hits | eqp **1.00e-06 eV** |
| CPU P=8 htransform cold | `ht8_cold` | 2 nodes x 4 | 0 | 18 s | 152 (4.6 s) | bandstructure **bit-exact** |
| **CPU P=8 htransform warm** | `ht8_warm3` | 2 nodes x 4 | 0 | 9 s | **0**, 152/152 hits | bandstructure **bit-exact** |
| GPU cold | `g4_cold` (7876437) | 1 node x 4 RTX | 0 | 55 s | 371 (37.1 s) | eqp **1.00e-06 eV** |
| **GPU warm = AG's EXACT STALL REPRODUCER** | `g4_warm` | 1 node x 4 RTX, `--mpi=pmi2`, `ffi_env.sh`, `use_ffi_io=true slab_io=phdf5_ffi`, cache ON and populated | **0** | **20 s** | **14** (2.5 s), 357 hits | eqp **1.00e-06 eV** |
| GPU 2x2 cold | `g22_cold` (7876443) | 2 nodes x 2 RTX | 0 | 49 s | 359 (30 s) | eqp **1.00e-06 eV** |
| **GPU 2x2 warm** | `g22_warm2` | 2 nodes x 2 RTX | 0 | **18 s** | **3** (2.1 s), 356 hits | eqp **1.00e-06 eV** |

(AG's identical cell was **rc=124, stalled at 901 s with 0/4 ranks reaching
`Zeta fitting`**.  The job-2 2-node cells with the CUDA phdf5 FFI died in
Intel-MPI/OFI bring-up — `I_MPI_FABRICS=shm, but multi node launch is detected
... OFI addrinfo() failed` — which is AE/AG's known multi-node FFI limitation,
not the cache; job 4 reran them on AD's `h5py_allgather` recipe, as AG did.)

### AH.4 — GATE 2: THE PRIZE, MEASURED

Compile counts are exact and confound-free; wall is not, so the fixture walls
below are the MEDIAN of three cache-off and three warm cells run INTERLEAVED
inside one job (job 7876440), after a throwaway warm-up cell — the first
2-node cell of a job costs 120 s to the container/page cache and 26 s
thereafter, which is exactly the fake-win trap ADVICE §4 warns about.

| workload | metric | cache OFF | WARM | change |
|---|---|---|---|---|
| **CPU P=8 htransform (the storm case)** | compiles/rank | **152** | **0** | **-100 %** |
| | XLA compile s/rank | 4.6 | 0.0 | |
| | wall (median of 3) | 14 s | **10 s** | **-29 %** |
| **CPU P=8 gw_jax** | compiles/rank | **373** | **5** | **-98.7 %** |
| | XLA compile s/rank | 9.5 | 0.7 | |
| | wall (median of 3) | 26 s | **20 s** | **-23 %** |
| CPU P=4 gw_jax | compiles/rank | 357 | 3 | -99.2 % |
| | wall | 22 s | 14 s | -36 % |
| **GPU 4x RTX gw_jax** | compiles/rank | 371 | 14 | -96.2 % |
| | XLA compile s/rank | 38.6 | 2.5 | |
| | wall | 73 s | **20 s** | **-73 % (3.6x)** |
| GPU 2x2 gw_jax | wall | 50 s | 18 s | -64 % (2.8x) |
| P=1 CPU gw_jax | compiles | 253 | 2 | -99.2 % |
| P=1 GPU gw_jax | compiles / wall | 261 / 38 s | 10 / 12 s | -96 % / 3.2x |

Two honest caveats the numbers make visible:

* **The cache removes XLA compilation, not tracing.**  373 compiles/rank cost
  only 9.5 s of XLA at CPU fixture scale (H measured the same, ~7 s); the rest
  of the fixed floor is jaxpr tracing, lowering and dispatch, which the
  persistent cache cannot touch.  The GPU win is 3.6x because GPU compilation
  costs 38.6 s/rank (autotuning), not because more modules are cached.
* **On CPU the first warm run of a job can be a wash.**  `gw8_warm1` read its
  301 entries in 2-3 s on the node that wrote them and **12 s** on the other
  node (cold Lustre client) — against the 9.5 s of compile it replaced, hence
  29 s vs 26 s.  The second and third warm cells, with the entries in page
  cache, read in 0.3-0.7 s and land at 19-20 s.  `LORRAX_JAX_CACHE_PREFETCH=1`
  issues those reads from a 16-thread pool right after the agreement; job 5
  A/Bs it (fresh copy of the cache dir per cell so the read is genuinely cold).

### AH.4b — GATE 2 AT PRODUCTION SCALE, and the number that resizes the lever

Driven through a copy of wk_Y's probe harness: MoS2 12x12, **606 centroids,
P=16** (8 nodes x 2 ranks x 28 threads), `TIER=replicated MAXCHUNK=1
EXITZETA=1 HLO=0`, i.e. startup + the full compile set + one r-chunk of the
zeta fit.

| cell (job 7876440) | cache | wall | compiles/rank | XLA compile s/rank | cache-read s/rank |
|---|---|---|---|---|---|
| `b_off1` (first cell of the job — cold container) | OFF | 436 s | **174** | 4.4-5.2 | — |
| `b_cold` | populate | 355 s | 174 | 4.4 | 0 |
| `b_warm1` | WARM | 350 s | **5-6** | **0.46-0.49** | **8.8-29.0** |
| `b_off2` (the fair cache-off comparison) | OFF | **346 s** | 174 | 4.4-5.2 | — |
| `b_warm2` (prefetch ON, same nodes as `b_warm1`) | WARM | **345 s** | **5-6** | **0.46-0.49** | **0.38** (+0.05 s prefetch) |
| `b_warm_pf` (job **7876461**, prefetch ON, DIFFERENT nodes → cold client cache) | WARM | 425 s (first cell of its job — cold container, cf. `b_off1` 436 s) | **5-6** | **0.46-0.93** | **0.38-0.42** (+0.05 s prefetch) |

Two things fall out.

**(1) The compile saving at production scale is ~4.5 s per rank.** The ~174
modules/rank cost 4.4-5.2 s of XLA compilation, and every rank compiles them
concurrently, so that is ~5 s of WALL — about **1 %** of a 431 s run — not
174 x P of anything.

**(2) Reading the cache costs MORE than compiling it, if the reads are
serial.** 169 agreed entries, **876 kB total**, took **29.0 s on rank 0 and
8.8 s on rank 8** — pure per-file Lustre latency with 16 ranks opening the same
directory. Net effect at this scale: **warm 350 s vs cache-off 346 s — a
wash**, the ~4.5 s of compile saved handed straight back to the filesystem.
Two fixes shipped for it: `LORRAX_JAX_CACHE_PREFETCH` (default **on**)
pulls the agreed entries through a 16-thread pool right after the agreement,
and a fresh cache directory gets `lfs setstripe -c 1` (these files are 6 kB
each; a striped layout multiplies the RPCs per open for no benefit).

`b_warm2` ran with the prefetch on (the default was flipped between cells) and
its reads are **0.38 s instead of 29.0 s**, for 0.05 s of prefetch — but it ran
6 minutes after `b_warm1` on the same nodes, so the page cache is a confound
and that ratio is an upper bound, not a clean A/B. Job 5 (**7876461**) does the clean one — fresh copy of the cache directory
per cell, so the second node's client cache is genuinely cold. CPU P=8 fixture,
node-1 ranks: **prefetch OFF 2.07-2.26 s of reads; prefetch ON 0.66-0.71 s of
reads for 0.29 s of prefetch** — 2.3x less filesystem time. Walls in that cell
trio: cache-off 26 s, warm/prefetch-off 19 s, warm/prefetch-on **18 s**.
htransform, same protocol, is cleaner still: node-1 reads **3.87-3.94 s → 0.39-
0.42 s** (a 5.7x cut) for 0.29 s of prefetch, wall **13 s → 10 s**. And at
606c/P=16 on nodes that had never touched the cache (`b_warm_pf`, job 7876461,
c209-011..018 vs job 3's c209-021..028): **0.38-0.42 s of reads for 0.05 s of
prefetch, against `b_warm1`'s 8.8-29.0 s serial** — the same 142 entries. So
the prefetch default of ON is measured, not argued. (Caveat: those entries had
been read minutes earlier from other clients, so Lustre's server-side caches
were warm even though the client caches were not; treat 0.05 s as the good
case, not the worst case.) What is NOT confounded: the
compile counts (174 → 5-6 on every rank, cold and warm, both runs) and the
fact that neither warm run moves the 346 s wall, because ~4.5 s is all there
was to win.

**This is the honest resizing of the lever the campaign has been chasing since
day one.** At production scale on CPU the ~174 modules/rank cost about **5
seconds** of XLA compilation per rank — and every rank compiles them
concurrently, so it is ~5 s of WALL, not 174 x P of anything. A perfect
compile cache therefore removes ~1 % of a 346 s production CPU run. The
"compile storm" is real as a COUNT and small as a COST: what the count is
buying is jaxpr tracing, lowering and eager dispatch, which the persistent
cache cannot touch (it is consulted after lowering). ADVICE §4's old CAUTION
box was right and D's "the cache is the path to killing 158 s" was not.

Where the cache IS large is exactly where compilation is expensive:
**GPU, where autotuning makes compilation cost 38.6 s/rank and the warm run is
3.6x faster end to end.** That is also the platform where getting this wrong
deadlocks, which is why it had to be built correctly rather than left off.

### AH.5 — GATES 3 and 4: the controls

| control | cell | what must happen | result |
|---|---|---|---|
| **forced divergence, CPU P=8** | `gw8_div` (7876440), `LORRAX_JAX_CACHE_FORCE_DIVERGE=9` on every rank != 0 | detect, drop, say so, DO NOT hang | 605 advertised → **596 agreed, 9 DROPPED** with the reason printed; every rank compiles 9-10; **rc=0, 18 s**, eqp 1.00e-06 eV |
| **forced divergence, GPU 1x4** (where divergence really deadlocks) | `g4_div_forced` (7876437), `=7` | same | 297 → **290, 7 DROPPED**; 22 compiles/rank; **rc=0, 28 s**, eqp 1.00e-06 eV |
| **forced divergence, GPU 2x2** | `g22_div` (7876443), `=9` | same | 288 → **279, 9 DROPPED**; 15 compiles/rank; **rc=0, 19 s**, eqp 1.00e-06 eV |
| **rank-0-only `XLA_FLAGS`** (wk_Y's HLO-dump pattern) — rank 0 would MISS while its peers HIT | `gw8_envmix` (7876440) | detect the key-environment mismatch and turn the cache off on EVERY rank | all 8 ranks print `DEGRADED TO CACHE-OFF — ... DIFFERENT key-affecting environment (XLA_FLAGS / jaxlib version / jax cache config) ...`; 373 compiles/rank; **rc=0**, eqp 1.00e-06 eV |
| **AG's refusal still reachable** | `ref8` (7876434), `LORRAX_JAX_CACHE_MULTIPROCESS=0` | print AG's line, no cache | `DISABLED at 8 processes by LORRAX_JAX_CACHE_MULTIPROCESS=0 (scorecard-AG refusal)`; 373 compiles/rank; rc=0, eqp 1.00e-06 eV |
| **P=1 unchanged** | `p1_cold`/`p1_warm` (CPU, 7876434), `gp1_a`/`gp1_b` (GPU, 7876437) | cache still arms, dir grows, no refusal line | CPU 253 → **2** compiles, 0 → 251 entries; GPU 261 → **10** compiles, 38 s → 12 s; eqp 1.00e-06 eV throughout |
| **unit suite** | `tests/test_compile_cache_agreement.py` | the agreement's safety invariants, with a fake in-memory KV client and one thread per "rank" | **8/8 assertions pass** driving the real `_agree_on_entries` (shared dir → all agree; one rank blind to 2 entries → all 4 ranks drop exactly those 2; `FORCE_DIVERGE` really drops; cold cache agrees on nothing; O(1) RPCs — 9 sets + 21 gets at P=8 regardless of 200 entries; bitmask round-trip; the directory scan ignores empty and `.tmp` files; no client → raise, never hang). **`8 passed` under pytest in-container** (job 7876461, 10.1 s). |

`LORRAX_JAX_CACHE_MULTIPROCESS` semantics, reconciled: default flipped `0` →
`1`. The variable that used to mean "re-arm the deadlock" now means "use the
repaired cache", and `=0` is the bisecting escape hatch back to AG's refusal.
Nothing in-tree or in any launcher sets it, so the transition is a pure default
change; every launcher that exports `ISDF_JAX_CACHE_DIR=""` keeps working
unchanged (and now leaves the win on the table — see the ADVICE §4 update).

### AH.6 — the design space, and what was rejected

| candidate | verdict |
|---|---|
| **per-rank dirs** (the status quo ante) | permanently asymmetric, because jax writes from process 0 only. This IS the AG deadlock. Deleted. |
| **naive shared dir** | insufficient on its own: the key is not process-invariant on CPU, writes are not atomic, and jax auto-enables a rank-asymmetric XLA sub-cache (AH.1). All three fixed. |
| **shared dir + process-invariant key, no agreement** | works in the happy path — `g4_noagree_warm` (7876437) completed in 19 s — and fails exactly when the ranks' VIEWS differ, which is the realistic mistake (a node-local `/tmp` cache dir; see the A/B in AH.7). The agreement is the insurance, the invariant key is the mechanism of the win. |
| **`jax_share_binary_between_hosts=True`** | rejected. `compile_or_get_cached` returns on a hit BEFORE the share path, so process-0-hits/peers-miss is the same deadlock; and it keys the KV broadcast on the cache key, which on CPU differs per rank, so it would hang even on a symmetric cold miss. (It also has exactly one process compile, which reads as incompatible with a sharded autotuner — inference from the sources, not measured here.) |
| **all-rank writes** | rejected as unnecessary: under SPMD process 0 compiles the same module set as everyone, so its writes already cover it, and P-way writes to one Lustre directory buy nothing. Writes were made atomic instead, which is what the concurrency argument actually needed. |
| **`--xla_gpu_shard_autotuning=false`** | exists in this jaxlib (`strings` on `xla_cuda_plugin.so` confirms the DebugOptions field). It would remove the coordination-service exchange from GPU compilation entirely and so retire the whole deadlock class, at the cost of every process autotuning everything. NOT exercised here — the agreement makes it unnecessary — but it is a real belt-and-braces knob for anyone who wants one. It must be set identically on every rank, which the key-environment fingerprint now enforces. |

### AH.7 — IS THE AGREEMENT LAYER LOAD-BEARING?  The A/B (job 7876443, rtx-dev)

With the key made process-invariant and a fully-shared directory, the naive
design happens to work — `g4_noagree_warm` completed in 19 s.  The agreement
earns its place when the ranks' VIEWS of the cache differ, and the most
plausible way a user creates that is a **node-local cache dir**.  JAX writes
from process 0 only, so node 0's `/tmp` fills and every other node's stays
empty.  Same populated state, agreement OFF then ON, 2 nodes x 2 GPUs:

| step | result |
|---|---|
| `nl_populate` — `ISDF_JAX_CACHE_DIR=/tmp/ah_nodelocal`, cold | rc=0, 49 s, 359 compiles/rank. Node A: **286 entries**. Node B: **0 entries**. |
| **(A) `nl_noagree`** — same, warm, `LORRAX_JAX_CACHE_NO_AGREE=1` | **rc=124, STALLED at 600 s**, no `eqp_test.dat` — AG's deadlock, reproduced by construction |
| **(B) `nl_agree`** — same, warm, agreement ON (the shipped default) | **rc=0, 48 s**; `287 entries advertised, 0 agreed by all ranks` + `*** 287 entries DROPPED ***`; all 4 ranks compile 359; eqp **1.00e-06 eV** |

That is the whole argument in two cells: the same populated cache, the same
launch, one env var apart — one hangs the job, the other says out loud that the
cache is unusable here and runs correctly without it.

### AH.8 — a second stale ADVICE §4 claim, also corrected

Every warm CPU run emits a wall of

    E cpu_aot_loader.cc:220] Loading XLA:CPU AOT result. Target machine
    feature +prefer-no-gather is not supported on the host machine ...
    This could lead to execution errors such as SIGILL.

— 738 of them in one rank log. ADVICE §4 says these are the "poisoned cache"
and that they force a recompile each. **They do not.** `prefer-no-gather` /
`prefer-no-scatter` are LLVM *cost-model pseudo-features*: present at compile
time, never present in a runtime CPU feature list, so `cpu_aot_loader.cc`'s
comparison mismatches on every load on every machine. It is a `LOG(ERROR)`,
not a rejection. Measured on the very run that printed those 738 lines:
**369/373 hits, 4 compiles**; on `ht8_warm3`, 304 lines with **152/152 hits
and ZERO compiles**. Both ADVICE §4 and the module docstring now say so, so
nobody disables the cache again because of them.

### AH.9 — named, not done

* **The prefetch ships ON and is measured** (job 7876461: htransform reads
  3.9 s → 0.39 s, wall 13 s → 10 s; 606c/P=16 reads 8.8-29.0 s → 0.38-0.42 s).
  The `lfs setstripe -c 1` that ships alongside it is NOT measured: it only
  applies to entries written into a FRESH cache directory, and every cache in
  this workstream predates it. It is a 20-line best-effort call justified by
  the file sizes (6 kB), not by an A/B — someone should delete a cache,
  repopulate, and check whether it moves the cold-read number at all.
* **The cache dir default is still `~/.cache`.** One entry is one small file
  and a populated world size is several hundred, so the default spends `/home1`
  inodes — the quota that locks you out of Frontera (ADVICE §2). Launchers
  should export `ISDF_JAX_CACHE_DIR=$SCRATCH/...`; changing the in-tree default
  is an owner call. The stale `~/.cache/isdf_jax_compilation/np*/rank*` dirs
  AG.7 flagged (14 MB, ~1700 inodes) were deleted.
* **The shared test infra still opts out.** `lorrax_setup/alloc_run.sh` and
  `cpumn_a.sbatch` export `ISDF_JAX_CACHE_DIR=""`, as does every campaign CPU
  launcher. Those are other agents' running harnesses, so they were left alone:
  behaviour is unchanged for them, they just leave the (small, on CPU) win on
  the table. Anyone who wants it should export
  `ISDF_JAX_CACHE_DIR=$SCRATCH/lorrax_jaxcache` instead of `""`.
* **`wk_Y/probe.sbatch` still hard-sets `ISDF_JAX_CACHE_DIR=""`** and, when
  `PROBE_HLO=1`, sets `XLA_FLAGS` on rank 0 only. The second is now detected
  and handled (the run degrades to cache-off, loudly, instead of diverging),
  but if the standing probe is ever meant to measure the cache, both lines
  need attention. `wk_AH/probe_ah.sh` is the patched copy used here.
* **Only `gw.gw_jax` and `bandstructure.htransform` were driven end to end.**
  The other five callers (`run_nscf`, `run_sternheimer`, `kmeans_cli`,
  `bse_feast`, plus the library-internal `w_isdf`/`ppm_tau_kernel` calls) share
  the same one-line entry point and the same code path, but that is reasoning,
  not a measurement.
* **`_fatal` uses `os._exit(70)`** when an agreed entry turns out to be
  unreadable. That kills the rank hard so srun tears the step down; it is the
  "should never happen" branch and it has never fired in any cell here, so it
  is untested in anger.
* **Counters are not locked.** `_STATE.probes/hits/compiles` are incremented
  from whatever thread JAX compiles on; the failure mode is a lost update in a
  diagnostic, never in the agreed set (which is immutable after startup).


---

## AJ — the MoS₂ 4×4 / 30 Ry FAST TEST DECK: built, gated, baselined (wk_AJ, 2026-07-27; **zero source edits**, main checkout @ eab0dd3)

**One line: there is now a complete, reproducible LORRAX deck whose full GN-PPM
G0W0 runs in ~3 minutes on ONE node — 98× smaller WFN, 9× fewer q-points, 4.4×
smaller ngkmax than the 12×12 — and it closes the H0-vs-`kih.dat` identity to
7.8e-5 eV, the first independent re-validation of the Q (loader) + S (stored
V_H) stack on a SECOND deck with a DIFFERENT symmetry configuration.**

Everything lives in `/scratch2/08271/jackmc/mos2_4x4_test/`; `REGENERATE.sh`
prints the whole recipe and every step is an sbatch script beside it.
Jobs: `qe_all.7876519`, `deck.7876520`, `gw400.7876523`, `gw800.7876526`.

### AJ.0 — the ecut ambiguity in the directive, resolved (flagging it)
The directive said "~30 Ry **ecutrho**". Taken literally that is
`ecutwfc = 7.5 Ry`, which no MoS₂ deck can use. Both existing decks set
`ecutrho = 4 × ecutwfc` (the QE norm-conserving default: 80/320), and the
`gnppm_debug` fixture — the one deck whose ionic terms N independently verified
as physical — is **ecutwfc = 30**. So: **ecutwfc = 30 Ry, ecutrho = 120 Ry**.
The two readings differ by 16× in basis size, hence the note.

### AJ.1 — THE DECK (dims, and what changed from the 12×12)

| | **mos2_4x4_test (new)** | mos2_80ry_12x12 (ref) | ratio |
|---|---|---|---|
| ecutwfc / ecutrho | **30 / 120 Ry** | 80 / 320 Ry | |
| k-grid (Γ-centred, no shift) | **4×4×1** | 12×12×1 | |
| nk full BZ / nk in file (IBZ) | **16 / 10** | 144 / 144 | 9× / 14.4× |
| `ntran` (WFN symmetry) | **2** (E + σ_h, symmorphic, `tnp=0`) | 1 (`nosym`) | |
| ngkmax (min ngk) | **1964** (1933) | 8603 | 4.38× |
| FFT grid (n_r) | **(24,24,80) = 46 080** | (36,36,135) = 174 960 | 3.80× |
| ρ-sphere `ng` | **15 631** | 67 737 | 4.33× |
| nelec (spinor bands) | **26** | 26 | — |
| mnband in WFN.h5 | **256** | 400 | |
| σ window `nval/ncond/nband` | **26 / 102 / 128** | 26 / 44 / 120 | |
| cell volume / alat | 702.2012 bohr³ / 5.979645 bohr | identical | — |
| `WFN.h5` | **153 MiB** | 15.65 GB | **98×** |
| `kin_ion.h5` / `dipole.h5` | **8.1 / 15 MiB** | 33 / 116 MiB | |
| DFT gap on the grid | **2.2121 eV** (VBM −5.3602, CBM −3.1481) | 1.70 eV | |

Everything else is **verbatim** from the 12×12: cell, atoms, both ONCV FR
pseudos (md5-identical), `noncolin`+`lspinorb`, `no_t_rev`,
`assume_isolated='2D'`, `conv_thr=1e-10`, `diago_full_acc`. The only edits are
ecut, kgrid, `nbnd`, and **removing `nosym`/`noinv` from the nscf**.

`nband = 128` is chosen so the σ window is divisible by **both** P=4 and P=16
(and 8/32/64) — no mesh pad anywhere. `nval = 26 = nelec` (the only
configuration N/S proved safe for the Hartree ρ).

**Two deck facts worth knowing before you reuse this cell.**

1. **The cell as written is ~2.8e-6 off ideal hexagonal**
   (`|a2| = 3.164 30` vs `|a1| = 3.164 292`; `a2_y` vs `a1·√3/2` differ by
   7.6e-6 Å). QE's lattice-symmetry test uses `eps1 = 1e-6` on the rotated
   axes, so **it rejects the 3-fold rotations and finds only 2 ops**
   ("2 Sym. Ops. (no inversion) found") — in *both* this deck and the 12×12's
   own scf. So "symmetry ON" here means `ntran = 2` + TRS, i.e. the **gnppm
   fixture's configuration**, and the 4×4 IBZ is 10 points (4 TRIM + 6 TRS
   pairs), not the 4 the full D₃ₕ would give. That still satisfies `ntran > 1`
   and still exercises the general `SymMaps` branch and TRS unfolding, which is
   the deliberate contrast with the `nosym` 12×12. Tightening the cell to exact
   hexagonal would recover 12 ops and a 4-point IBZ — a cheap future win, at
   the cost of "verbatim". Note `centroid/kmeans_cli` **recovers the full 12-op
   group from the charge density anyway** and closes the centroid sets under it.
2. **The scf k-grid MUST equal the nscf k-grid** (both 4×4×1 here). LORRAX
   rebuilds ρ from the WFN's own occupied bands while QE's `kih.dat` carries
   V_H from the *scf* density; a denser scf grid would be better physics but
   would break the H0 identity gate by construction. Consequence: the 4×4 mesh
   does not contain K = (1/3,1/3), so the deck's DFT gap (2.21 eV) is a
   grid-restricted gap, not MoS₂'s physical one. This is a **numerics** test
   deck, not a physics deck.

### AJ.2 — GENERATION IS ~5 MINUTES, END TO END (all on `development`)

| step | script | resources | wall |
|---|---|---|---|
| scf | `qe_all.sbatch` | 1 node × 56 ranks, `-npools 4 -ndiag 1` | **9 s** |
| nscf (256 bands, 10 IBZ k) | ″ | ″ | **56 s** |
| pw2bgw (a) WFN+vxc+kih | ″ | `-n 24` | **5 s** |
| pw2bgw (b) RHO | ″ | `-n 24` | **3 s** |
| `wfn2hdf.x` | ″ | serial | **4 s** |
| centroids 108 / 402 / 785 | `deck_complete.sbatch` | 1 node, 1 proc | **102 / 19 / 26 s** |
| `kin_ion.h5` (`--hartree`) | ″ | ″ | **18 s** |
| `dipole.h5` | ″ | ″ | **23 s** |
| | | **total** | **~4.5 min** |

(`-ndiag 1` per ADVICE §7 — kept even though this nscf is small.)

Centroids are orbit-quantised under the density-recovered 12-op group:
requested 100/400/800 → **108 / 402 / 785**. 785 is the "≈800" point and it
**pads to exactly 800** on the 4×4 mesh (`V_q` shape `(16, 800, 800)`), so the
owner's 800/16 = 50 arithmetic lands. The CLI flag is `--out-suffix` (the old
`--out` abbreviation is dead now that `allow_abbrev=False`).

`kin_ion.h5` carries the **new S format**: `kin_ion (16,128,128)` pristine
T+V_loc+V_NL **plus** `v_hartree (16,128,128)`, `has_hartree=False`,
provenance `nk=16, nb=128, nval=26, ncond=102, nelec_bands=26, sys_dim=2,
truncation_2d=True, bispinor=False, nspinor=2, fft_grid=[24,24,80],
input_file=deck.in, wfn_file=WFN.h5, pseudopotentials=['Mo','S']`. Both GW runs
print `hartree_source: requested=auto → resolved=stored`.

### AJ.3 — GATES (all green)

**(a) TRS + spatial density check (workstream U), measured at every load:**
`TRS=HOLDS (‖m‖/‖ρ‖ = 4.29e-14, coverage 25 %, mesh-implies-TRS=True) |
spatial 2/2 ops tested, max resid 5.36e-13 | ∫ρ = 26.000000 (rel 4.1e-16)`.
The 25 % coverage is exactly the TRS-folded-IBZ situation U predicted (only the
4 TRIM points are self-paired), and the verdict is unambiguous.

**(b) THE H0 IDENTITY GATE — `gate_h0.py`, all 16 full-BZ k × 128 bands
(2048 states) vs `out/kih.dat`:**

| quantity | result |
|---|---|
| **rms(H0 − kih)** | **7.8e-5 eV** (mean −6.3e-5, max\|Δ\| **2.4e-4**) |
| per-k rms | flat, 7.66e-5 … 8.06e-5 across all 16 k |
| per-band rms | 1.85e-5 … 2.39e-4 (worst = band 0, the Mo semicore) |
| implied Vxc range | [−24.2546, −3.4392] eV vs QE `vxc.dat` [−24.2548, −3.4391] |
| **rms(implied Vxc − Vxc_QE)** | **7.8e-5 eV** |
| `_warn_on_unphysical_h0` | **SILENT** in both GW runs |

This is a **fresh, independent** confirmation of Q+S: a different ecut (30 vs
80 Ry), a different symmetry configuration (`ntran=2` + TRS-unfolded vs
`nosym`), a different band count, and the identity still closes at 1e-4 eV.
It also gates the **unfold**: 6 of the 16 full-BZ k are TRS images of their
representative and they reproduce its `kih` value to the same 1e-4 eV.

`gate_h0.py` differs from `wk_Q/gate_rms.py` in two necessary ways — it sums
`kin_ion + v_hartree` (the new format; the old script silently reads a
~500 eV-short H0), and it maps full BZ → IBZ (`nk_full=16 ≠ nk_ibz=10`, which
could not arise on the `nosym` 12×12). It uses `SymMaps.irr_idx_k` when jax is
importable and an equivalent pure-numpy k-star search otherwise, **so it runs
on a login node**; both paths were exercised here and agree exactly (same map,
same 7.8e-5 eV).

**(c) Cross-deck physical consistency (free, and a strong one).** Same
material, same cell, k=0 band 0, two decks 2.7× apart in ecutwfc and 9× in
k-sampling:

| | V_H | H0 |
|---|---|---|
| 4×4 / 30 Ry (this deck) | **602.399 eV** | **−41.941 eV** |
| 12×12 / 80 Ry (`wk_Q/kin_ion_vh120_FIXED.h5`) | 602.402 eV | −41.927 eV |

V_H agrees to **3 meV**, H0 to **14 meV**.

**(d) eqp sanity.** Positive, physical, and — usefully for a regression
fixture — **already converged in μ**:

| | DFT gap | eqp0 (G0W0) | eqp1 |
|---|---|---|---|
| 402 centroids, P=4 | 2.2121 | **3.5819** | 3.2646 |
| 785 centroids, P=16 | 2.2121 | **3.5867** | 3.2633 |
| Δ(785 − 402) | — | **+4.8 meV** | **−1.3 meV** |

QP shift at 402c: VBM +0.228 eV, CBM +1.598 eV (gap opens 1.370 eV).
`sigma_diag.dat`'s VH column reads **602.399 eV**, not 0.000 — the S contract
(`stored` ⇒ *substitute*, not suppress) working as designed.

### AJ.4 — **BASELINE A: 402 centroids, P=4** (1 node × 4 ranks × 14 threads, mesh 2×2)
`run_400c/`, job 7876523. Step wall **183 s**; `Total recorded` **104.6 s**;
431 XLA compiles/rank (13.4 s), `ISDF_JAX_CACHE_DIR=""`.

| stage | s | % |
|---|---|---|
| `load_centroid_wfns` | 2.20 | 2.1 |
| **`zeta_fit_chunked`** | **49.27** | **47.1** |
|  ⤷ `chunk_loop` (2 r-chunks) | 45.62 | 43.6 |
|  ⤷⤷ `chunk.z_q_build` | **38.98** | 37.3 |
|  ⤷⤷ `chunk.solve` (back-solve) | 5.66 | 5.4 |
|  ⤷ `cholesky` / `write_g_flat` | 1.03 / 1.17 | 1.0 / 1.1 |
| `V_q_compute` | 1.02 | 1.0 |
| `wavefunction_setup` | 0.31 | 0.3 |
| `chi0_W` (χ 1.22 + W 0.72 + fold 0.38) | 3.18 | 3.0 |
| **`sigma`** (`ppm_sigma.exec` 45.94) | **48.66** | **46.5** |

### AJ.5 — **BASELINE B: 785 centroids, P=16** (8 nodes × 2 ranks × 28 threads, mesh 4×4)
`run_800c/`, job 7876526. Step wall **384 s**; `Total recorded` **258.9 s**;
419 XLA compiles/rank (12.7 s), same cache-off setting.

| stage | s | % |
|---|---|---|
| `load_centroid_wfns` | 2.46 | 0.9 |
| **`zeta_fit_chunked`** | **87.18** | **33.7** |
|  ⤷ `chunk_loop` (1 r-chunk) | 75.85 | 29.3 |
|  ⤷⤷ `chunk.z_q_build` | 24.91 | 9.6 |
|  ⤷⤷ **`chunk.solve` (back-solve)** | **50.25** | **19.4** |
|  ⤷ `cholesky` / `write_g_flat` | 4.64 / 5.34 | 1.8 / 2.1 |
| `V_q_compute` | 3.39 | 1.3 |
| `wavefunction_setup` | 0.36 | 0.1 |
| `chi0_W` (χ 0.78 + W 4.91 + fold 0.96) | 7.72 | 3.0 |
| **`sigma`** (`ppm_sigma.exec` 153.36) | **157.84** | **61.0** |

Σ sub-walls: `ω≥E_F cond` 60 s, `ω≥E_F val` 12 s, `ω<E_F cond` 11 s (+ val).

### AJ.6 — WHAT THE TWO BASELINES SAY (the point of the 800-centroid ask)

**The ζ back-solve crosses over `z_q_build` in the borderline-large-μ regime.**
It is the μ³ term and it is the one that runs away:

| | 402c / P=4 | 785c / P=16 | |
|---|---|---|---|
| `chunk.z_q_build` | 38.98 s | 24.91 s | ↓ 1.6× (4× devices, ~2× work) |
| `chunk.solve` | 5.66 s | 50.25 s | **↑ 8.9×** |
| solve / z_q_build | **0.15** | **2.02** | **13× swing** |

`z_q_build` scales ~μ·n_r and shards cleanly (more devices actually helped);
`solve` scales ~μ³ and at 785c it is already 66 % of the chunk loop *on 4× the
hardware*. Anyone hunting the ζ-fit wall at large μ should point at
`zeta_fit.chunk.solve`, not `z_q_build` — the 12×12/276c numbers (where
`z_q_build` dominates) are misleading about where large-μ time goes.

**Σ becomes the majority stage.** 46.5 % → 61.0 %; `ppm_sigma.exec` alone grew
45.9 → 153.4 s (3.3×) for 1.95× the centroids, i.e. **~μ²** and *not* helped by
4× the devices. At P=16 this deck is Σ-bound, not ζ-bound.

**Deck-quality numbers that improve with μ:** GN-PPM "unfulfilled" modes
7.30 % → **2.73 %**; ζ rank-truncation keeps all μ at every q
(`n_keep/q = 402` of 402, λ_max/λ_min_kept ≈ 1.1e7) — the fit is well
conditioned at both counts. `QSGW: 1538 clipped (75.1 %)` is higher than the
12×12's 54.8 % but does not enter `qp_solver=one_shot_dft`.

### AJ.7 — HOW TO USE THIS DECK
* **One GW run per directory** (ADVICE §6d): `run_400c/` and `run_800c/` each
  own their `gw.in` and `tmp/zeta_q.h5`, and symlink the shared deck files.
* `restart = true` + the existing `tmp/isdf_tensors_{402,785}.h5` skips straight
  to Σ — the right way to iterate on the post-ζ stages.
* The two timing scripts set `ISDF_JAX_CACHE_DIR=""` **on purpose**, so these
  baselines are directly comparable to every 12×12 number on this scorecard and
  carry no cold/warm-cache ambiguity. Per ADVICE §4 the cache is fixed and
  worth ~1 % on CPU; turn it on for real work, not for regressions against
  these tables.
* `qp_wfn_rotations.h5` / QP-WFN dump is **skipped** on this deck: "Σ on 16
  k-points but WFN carries 10 (IBZ); the one-shot dump only supports IBZ-Σ."
  A real gap for symmetry-reduced decks — the 12×12 never hit it because
  full BZ == IBZ there. Set `debug.write_wfn_h5=false` to silence, or fix the
  dump if a QSGW/htransform run needs it here.

### AJ.8 — LOOSE ENDS / SMALL DEFECTS SEEN (not fixed — no source edits)
* `gw/kin_ion_io.py:760` prints `Hartree folded in: {args.hartree}`. That label
  is wrong in the post-S world: it reports `--hartree` (stored array), not
  `--fold-hartree`. It read "Hartree folded in: True" for a file that is
  correctly **not** folded.
* `kin_ion` carries `hartree_truncation_2d=False` while the `v_hartree`
  dataset it ships with carries `truncation_2d=True`. Only the latter is
  meaningful for the stored route; the former looks like a folded-path leftover
  and is an invitation to misread the file.
* `centroid/kmeans_cli` cannot use `out/MoS2.save` on either MoS₂ deck: this QE
  7.2 build writes `charge-density.dat`, and `rho_from_qe_save` requires
  `charge-density.hdf5`. It falls back to the WFN IBZ sum, loudly and
  correctly — but every `--qe-save` in the 12×12 scripts has been a no-op too.

## AI — the restart-tensor writer's 1.7 MB/s, root-caused and fixed: it is a STRIDED-TILE + INDEPENDENT-MPI-IO defect, not a transport one — and `lfs` does not exist inside the container (wt-J, branch `tensor-writer-fix`, base eab0dd3, 2026-07-27 — NOT committed)

**One line: `V_qmunu` is a 2-D `(x,y)` TILE of a CONTIGUOUS HDF5 dataset, so
its innermost contiguous file run is `(μ/Py)·16` = **3.2 kB** and there are
`nq·(μ/Px)` = **28 800 of them per rank** — under INDEPENDENT MPI-IO that is
one `MPI_File_write_at` each, 4.1 M of them across 144 ranks; the ζ writer is
fast on the SAME transport in the SAME job only because `zeta_q_G` is sharded
on one axis AND created with `chunks=(1, μ, ngkmax)`, making a rank's tile one
contiguous 2.3 MB region. `H5FD_MPIO_COLLECTIVE` lets ROMIO's two-phase
exchange aggregate the strided tiles: **74.2 → 2066 MB/s, 27.9×**, measured on
the production per-rank geometry — and the second half of that is a Lustre
finding the owner called for: **`lfs` IS NOT PRESENT INSIDE THE APPTAINER
IMAGE**, so `_lustre_prestripe` has been a silent no-op on Frontera for the
whole campaign (both of job 7876423's output files came back
`lmm_stripe_count: 1` — 13.3 GB through ONE OST). The layout now comes from
MPI-IO's `striping_factor`/`striping_unit` hints, which ROMIO applies through
`llapi` with no binary on PATH.**

**And the counter-intuitive half of the matrix, which is the part worth
remembering: striping ALONE, without collective, makes it WORSE — 74.2 →
12.0 MB/s at stripe 16.** More OSTs under 3.2 kB independent strided writes
means more round-trips per byte, not more bandwidth. Striping is a
multiplier on aggregation, not a substitute for it.

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AI/` — `wbench.py`
(the JAX-free writer microbench), `run_bench.sh`, `run.sh`, `bitcmp_all.py`,
`b1.log b2.log b4.log` (the matrices), `out_g{A_indep,B_coll,C_allg}.txt`,
`out_bitcmp.txt`, `runAI_close.sbatch`, cells `g{A_indep,B_coll,C_allg}/`.

### AI.1 — the diagnosis: two writers, one transport, three orders of magnitude

AF.4c measured 18.13 GB of restart tensors in 2 h 55 m (~1.7 MB/s aggregate
across 144 ranks) against the same job's ζ writer moving 47.7 GB in minutes,
and named the open question: *which* of dataset creation, the per-rank
hyperslab write, or the file-open cycle owns it. It is the hyperslab write,
and the reason is a **shape** property, not a transport property:

| | `zeta_q_G` (fast) | `V_qmunu` (1.7 MB/s) |
|---|---|---|
| shape | `(nq, μ, ngkmax)` | `(nq, μ, μ)` |
| sharding | μ over the FLAT mesh — **1-D** | `P(None,'x','y')` — **2-D TILE** |
| HDF5 layout | `chunks=(1, μ, ngkmax)` (`isdf_fitting.py:737`) | `chunks=None` → **CONTIGUOUS** |
| per-rank contiguous run | `(μ/P)·ngkmax·16` = **2.3 MB** | `(μ/Py)·16` = **3.2 kB** |
| runs per rank | 144 | `nq·(μ/Px)` = **28 800** |
| runs, whole job | 20 736 | **4 147 200** |

Under `H5FD_MPIO_INDEPENDENT` (what `_slab_io_mpi_host` used, and what the
C++ writer still defaults to) HDF5 decomposes a hyperslab selection into its
contiguous file runs and issues one `MPI_File_write_at` per run. 3.2 kB per
call against a Lustre OST is the classic ~MB/s regime. `psi_full_y`
`(nk, nb, ns, μ)` is the same defect one worse: 3.2 kB runs, **124 416 of
them per rank**, and it is replicated over `'x'`, so all 12 ranks of each mesh
column wrote identical bytes to the same hyperslab — 12× redundant traffic.

**Reproduced with no JAX in the loop** (`wk_AI/wbench.py`, pure
mpi4py + h5py, production per-rank tile): P=8, 0.164 GB → **5.9 MB/s**.

### AI.1b — the second root cause, and it is the owner's: the stripe was never set

`_slab_io_ffi._lustre_prestripe` shells out to `lfs setstripe`. Every
production run executes inside `py312.sif`, which **does not ship `lfs`**, so
`shutil.which("lfs")` returns `None` and the function returns — silently, by
design ("we don't want to pollute stdout on non-Lustre targets"). Verified on
the artifacts of job 7876423:

    lfs getstripe .../tmp/isdf_tensors_2406.h5  ->  lmm_stripe_count: 1
    lfs getstripe .../tmp/zeta_q.h5             ->  lmm_stripe_count: 1

i.e. 13.3 GB of `V_qmunu` and 47.7 GB of ζ both funnelled through a **single
OST**, for the entire campaign. The C++ writer has always set
`striping_factor`/`striping_unit` in its `MPI_Info` (`context.cc:234`); the
h5py backend set **no info at all** — the single largest divergence between
the two "equivalent" writers. It now sets the same keys, and MPI-IO applies
them via `llapi` from inside the container with no binary on PATH.

Two further traps found while fixing it, both now closed in source:

* **A Lustre layout is fixed at CREATE time.** `H5Fcreate(TRUNC)` reuses the
  inode, so a rerun in a directory that already holds a 1-stripe
  `isdf_tensors_*.h5` keeps 1 stripe for ever and throws the lever away.
  `mode='w'` now **unlinks** before creating.
* `_lustre_prestripe`'s no-op is now announced once instead of silent.

### AI.2 — THE MATRIX (wk_AI/wbench.py, P=16 on 8 nodes, mesh 4×4)

`μ = 800` so the per-rank tile is `(144, 200, 200)` — **the production
per-rank shape and run count** (2448/12 = 204 at 12×12), 1/9 the total bytes.
`V_qmunu` geometry, 1.475 GB, `write_s` is the H5Dwrite window between
barriers:

| case | write s | MB/s | × base |
|---|---|---|---|
| **independent, no hints — THE SHIPPED WRITER** | 19.88 | **74.2** | 1.0 |
| independent + stripe 16×4 MiB | 122.86 | **12.0** | **0.2 — WORSE** |
| independent + stripe 32×4 MiB | 73.28 | 20.1 | 0.3 — worse |
| **COLLECTIVE**, no stripe hints | 1.12 | **1313.4** | **17.7** |
| **COLLECTIVE + stripe 16** | 0.90 | 1638.0 | 22.1 |
| **COLLECTIVE + stripe 32** | **0.71** | **2066.2** | **27.9** |
| collective + stripe 32 + `romio_cb_write=enable` | 0.81 | 1825.8 | 24.6 |
| HDF5 `chunks=(1,μ/Px,μ/Py)`, independent, stripe 16 | 2.61 | 565.3 | 7.6 |
| HDF5 `chunks=(1,μ/Px,μ/Py)`, collective, stripe 16 | 2.24 | 658.8 | 8.9 |
| HDF5 `chunks=(1,μ,μ)`, collective, stripe 16 | 1.30 | 1131.8 | 15.3 |
| reshard → row-slab, independent, stripe 16 | 1.15 | 1285.1 | 17.3 |
| reshard → row-slab, collective, stripe 16 | 0.77 | 1919.5 | 25.9 |
| reshard → q-slab, independent, stripe 16 | 1.42 | 1038.8 | 14.0 |
| reshard → q-slab, collective, stripe 16 | 0.96 | 1541.1 | 20.8 |

`psi_full_y` geometry (`(144, 108, 2, 800)`, 0.398 GB, 31 104 runs/rank,
replicated ×4 over `'x'`):

| case | write s | MB/s | × base |
|---|---|---|---|
| **independent, no hints — THE SHIPPED WRITER** | 24.40 | **16.3** | 1.0 |
| independent + stripe 16 | 20.56 | 19.4 | 1.2 |
| **collective + stripe 16** | 0.85 | **469.2** | **28.8** |

(The bench's *dedup* rows are absent on purpose: `wbench.py` spells the null
participant as h5py's high-level `ds[0:0,…] = empty`, which does not enter
`H5Dwrite` and therefore deadlocks a collective — a bench defect, and a
useful demonstration of exactly the hazard the production code avoids by
using the low-level `select_none()` path. Dedup is gated in AI.3 instead,
where the fixture's `psi_full_y` IS replicated over `'x'` so half the ranks
take the null-selection branch, and the file still comes back bit-identical
to the serial oracle.)

**Read the matrix as three separate statements:**

1. **Collective is the lever** (17.7× on its own). Everything else is a
   multiplier on it.
2. **Striping is real but CONDITIONAL** — +57 % on top of collective
   (1313 → 2066), and **−84 % without it** (74 → 12). It is worth having as a
   first-class knob and it would have been actively harmful as the only fix.
3. **Resharding before the write is NOT needed.** A row-slab or q-slab
   all-to-all (the "cheap all-to-all vs a 3-order IO win" candidate) lands at
   1285–1920 MB/s — *below* collective-on-the-native-tile, while adding a
   13.8 GB all-to-all and touching the reshard this codebase has lost the most
   time to (J.9, T.4). **Not done, on measurement.** Likewise HDF5 chunking:
   it rescues the independent path (74 → 565) but is strictly worse than
   collective and would make the on-disk layout device-count dependent.
   Both were priced, neither was shipped.

### AI.3 — CORRECTNESS GATE: bit-identical to BOTH reference writers

`cohsex_debug` fixture, P=4, 2×2 mesh, three cells differing only in the
writer (`wk_AI/g*/`, `out_bitcmp.txt`):

| cell | writer |
|---|---|
| `gA_indep` | PHDF5_HOST, `COLLECTIVE_WRITES=0 DEDUP_REPLICAS=0` (the pre-AI path) |
| `gB_coll` | PHDF5_HOST, defaults: collective + dedup + stripe 32 (**the fix**) |
| `gC_allg` | `H5PY_ALLGATHER` — rank-0 serial `h5py` (the independent oracle) |

| gate | result |
|---|---|
| `gA_indep` vs `gB_coll`, `isdf_tensors_60.h5` (11 datasets incl. `V_qmunu`, `W0_qmunu`, `psi_full_y`, `G0_mu_nu`, `enk_full`) | **`H5_BITCMP_OK`** |
| `gA_indep` vs `gB_coll`, `zeta_q.h5` (48 datasets incl. `zeta_q_G`) | **`H5_BITCMP_OK`** |
| **`gC_allg` vs `gB_coll`, `isdf_tensors_60.h5`** | **`H5_BITCMP_OK`** |
| **`gC_allg` vs `gB_coll`, `zeta_q.h5`** | **`H5_BITCMP_OK`** |
| eqp vs `eqp_ref.dat` @1e-3, all three cells | **PASS, 1888 values, max\|Δ\| = 1.00e-06 eV, max rel 5.29e-09, 0 over tol** |

The `gC` row is the load-bearing one: the collective per-rank hyperslab
writer with replica dedup produces **byte-for-byte** what a single rank
writing the whole gathered array serially produces.

### AI.4 — THE PRODUCTION MEASUREMENT: 18.13 GB in **2.2 s**, at 2406c / P=144 (job 7876530)

Same run dir, same 72 nodes × 2 ranks, same 12×12 mesh, `AC_MODE=vq` (ζ
reused off disk, `zeta_is_done=True`). AF.4c's instrument, now on the fixed
writer, verbatim from the log:

    [SlabIO.phdf5_host] isdf_tensors_2406.h5 mode=w collective_write=True
        dedup_replicas=True stripe_count=32 stripe_unit=4194304 B
    [restart_write] V_qmunu   (144, 2406, 2406)  13.34 GB in 1.6 s  (8182 MB/s)
    [restart_write] G0_mu_nu  (2406,)             0.00 GB in 0.0 s
    [restart_write] enk_full  (144, 432)          0.00 GB in 0.0 s
    [restart_write] W0_qmunu placeholder ALLOCATED (144,2406,2406) 13.34 GB
        in 0.0 s (no data written)
    [restart_write] psi_full_y (144, 432, 2, 2406) 4.79 GB in 0.6 s (7428 MB/s)

| | AF job 7876423 | **AI job 7876530** |
|---|---|---|
| the whole restart-tensor write | **2 h 55 m, and did not finish** | **2.2 s** |
| `V_qmunu`, 13.34 GB | ~1.4–3.7 MB/s | **8182 MB/s** |
| `psi_full_y`, 4.79 GB | **never written** (see below) | **7428 MB/s** |
| `V_q` stage | — | 168.6 s (144 q @ ≈0.85 s) |
| `wavefunction_setup` | — | 0.5 s |
| file stripe | `lmm_stripe_count: 1` | **`lmm_stripe_count: 16`, 4 MiB** |

Two honesty notes:

* **AF.4c's third file jump was misattributed, and this run proves it.**
  `psi_full_y` was **not written** by 7876423 — `h5ls` on that file lists
  `V_qmunu, W0_qmunu, G0_mu_nu, enk_full, band_window, kgrid, n_rmu_logical,
  restart_format_version` and **no `psi_full_y`**, and job 7876508 died on
  exactly that (`Restart file … is missing canonical psi_full_y dataset`).
  The +4.79 GB at 03:16 was its *space allocation*, whose object-header
  metadata never reached disk because the file was never closed. So the
  ledger is 13.34 GB written in ~2 h 55 m, not 18.13 GB — the pathology was
  ~2× worse than AF recorded, and the run was ~1 h further from finishing
  than the file size suggested. (This is AC.3b's trap for the third time, in
  the direction AF.4c itself warned about.)
* **The stripe request was clamped.** The job asked for
  `striping_factor=32`; Lustre/ROMIO granted **16**. Reported as granted,
  not as requested — the banner prints the request, `lfs getstripe` prints
  the truth, and the two differ.

### AI.4b — THE CLOSING RUN: the FIRST full 2406c / b400 end-to-end (job 7876530, **COMPLETED rc=0, 45 min 35 s**)

`AC_MODE=vq` on `run_A_c2406_b400_AF/`, 72 nodes × 2 ranks, 12×12 mesh, the
ζ from 7876346 reused off disk (`zeta_is_done=True`, provenance matched).
**The campaign has never before produced a complete GW at 2406 centroids.**

| stage | wall | note |
|---|---|---|
| startup + `jax.distributed` + compile | ≈45 s | 04:00:54 → 04:01:31 |
| `load_centroid_wfns` | **29.6 s** | loader 10.5 / gflat→rμ 4.1 / reshard 14.3 |
| ζ fit | **SKIPPED** | reused; the fit itself was AF's 43.4 min |
| `V_q_compute` | **168.6 s** | 144 q @ ≈0.85 s |
| **restart-tensor write** (`V_qmunu` + `G0` + `enk` + W0 placeholder) | **1.6 s** | **was ~2 h 55 m** |
| `wavefunction_setup` | **0.5 s** | |
| **`psi_full_y` write** | **0.6 s** | 4.79 GB @ 7428 MB/s |
| `save_restart_state_per_proc` | **≈4 min 43 s** | 144 dead files, 72 GB — see AI.6 |
| **`chi0_W` (screening)** | **27.7 s** | χ₀ 12.6 s + Dyson solve 13.9 s + compile 0.9 s |
| `W0_qmunu` write (13.34 GB, post-screening) | **< 15 s** | same fixed path |
| `sigma` (GN-PPM) | **1957.4 s** | `sigma.exec` 1896.4 s; two τ windows (73 + 72 nodes) |
| eqp / output | ≈12 s | |
| **TOTAL** | **2731 s = 45 m 35 s**, `rc=0` | `Total recorded` 2183.8 s |

**Deliverables:**

| | |
|---|---|
| **QP gap** | **2.7271 eV** indirect (VBM at K = (⅔,⅔,0), CBM at Λ = (⅙,⅙,0)) |
| QP min *direct* gap | 2.8428 eV at K |
| DFT gap | 1.7010 eV → **GW correction +1.026 eV** |
| cross-check | S's 12×12 reference at fewer centroids gave **2.6475 eV** (+0.947) and a K→K direct QP gap of 2.8765 eV. 2406c moves the indirect gap **+80 meV** and keeps the same K/Λ topology — the expected direction for a better-converged ISDF basis, not a new number to be surprised by. |
| **implied-Vxc guard** | `implied Vxc = E_DFT − kin_ion[exact V_H folded] ∈ [−24.262, −4.455] eV over 11520 (k,n)` — **SILENT / PASS** (compare the gnppm fixture's passing `[−28.281, −6.027]`; contrast O.9's broken `[−144.98, +88.42]` and S's `[−626.96, −87.61]`) |
| `n_keep` | **not re-measured** — the ζ was reused, so it is AF 7876346's **1717–1722 per q, mean 1718.3** at `n_pad = 2448`, which is the provenance-matched producer of this exact `zeta_q.h5` |
| writer route | `PHDF5_HOST`, `collective_write=True dedup_replicas=True`, granted stripe **16 × 4 MiB** |
| health | zero `FAIL-FAST`, zero `RESOURCE_EXHAUSTED`, zero Gloo errors, `MaxRSS 17.5 GB/rank` vs an 85 GB budget, node-0 RSS ≤ 37 GB |
| **owner's flop-budget invariant** | whole GW < 0.5 × ζ-fit wall. ζ-fit was 43.4 min ⇒ budget ≈21 min. **Screening is 27.7 s — 0.01× the fit, three orders inside budget.** Σ at 32.6 min is the stage that exceeds it (1.25× the fit); it is `sigma.exec`, one monolithic dispatch, and it is NOT this workstream's. |

**The restart artifact AF's run failed to produce now exists.**
`tmp/isdf_tensors_2406.h5` (31.465 GB) carries all 11 datasets —
`V_qmunu`, **`psi_full_y`**, `W0_qmunu` with **`W0_ready = TRUE`**,
`G0_mu_nu`, `enk_full`, `band_window`, `kgrid`, `n_rmu_logical`,
`restart_format_version`, `vhead`, `whead` — so `AC_MODE=restart` on this
directory is live for the first time (job 7876508 died precisely on the
missing `psi_full_y`), and screening + Σ can now be re-entered in ~35 min
without recomputing ζ or V_q.

**Caveat, from AK, recorded because it bounds these numbers:** this job ran the
pre-AK Gloo configuration, i.e. **JAX's CPU collectives went over the 1 GbE
`em1` management NIC, not IB**. Every JAX-collective-bearing stage above
(`sigma.exec` especially, and `load_centroids.reshard`) is therefore an
**upper bound**; AK measured 3.3× on the total pipeline at 606c/P=16 from the
ib0 pin alone. **The writer numbers are unaffected** — MPI-IO goes through
Intel MPI/OFI, which was already on IB, and the 8182 MB/s is a Lustre
measurement, not a Gloo one. The run was left to finish rather than
relaunched: it was 26 min in, past every collective wall, and it is the
campaign's first end-to-end.

### AI.5 — files touched (worktree wt-J, NOT committed)

`src/file_io/_slab_io_mpi_host.py` — `_mpi_io_hints()` (`striping_factor` /
`striping_unit`, plus `romio_cb_write` / `romio_ds_write` / `cb_nodes` left at
ROMIO's automatic policy unless asked — forcing `cb_write=enable` measured
*slower*, 1826 vs 2066); `info=` on the `h5py.File` open; a cached
`H5FD_MPIO_COLLECTIVE` dxpl; `write_slab` now routes through a new
`_write_hyperslab` that does one `H5Dwrite` per rank per dataset with either a
hyperslab or a **null** selection, so every rank enters every collective
write; replica dedup via one small `allgather` of `(offset, shape)`; the
`mode='w'` unlink; one banner line naming the mode actually in force.
Knobs, all defaulting to the measured-best: `LORRAX_PHDF5_COLLECTIVE_WRITES`
(default **1**; `0` restores the pre-AI behaviour exactly),
`LORRAX_PHDF5_DEDUP_REPLICAS` (default 1), `LORRAX_PHDF5_STRIPE_COUNT` (16),
`LORRAX_PHDF5_STRIPE_SIZE_FS` (4M).

`src/file_io/_slab_io_ffi.py` — `_lustre_prestripe` says once, on rank 0,
that it is skipping because `lfs` is absent, instead of returning silently.

`src/common/timing.py` — `LORRAX_TIMING_TRACE=1` makes every
`timing.section` announce entry/exit with a wall-clock stamp on rank 0
(`LORRAX_TIMING_TRACE_DEPTH`, default 3). Print-only; the accumulated tree
and the final report are unchanged. **This is the answer to the observability
gap AC.2, AC.3c and AF.4c each hit separately**: it is a milestone cadence for
the whole code, including stages that are one monolithic `jit` call
(`chi.exec`, `W.exec`) and therefore cannot carry a `LoopProgress`.

`src/file_io/tagged_arrays.py` — AF.4c's per-dataset restart-write instrument,
folded in as part of this diff (unchanged from AF).

### AI.6 — named, deliberately NOT done

* **The C++ (`PHDF5_FFI`) writer still defaults to independent.**
  `LORRAX_PHDF5_COLLECTIVE_WRITES=1` already reaches it
  (`context.cc:159`), and AI.2 says it should be its default too — but that
  library is the GPU path's writer as well, its comment records an OpenMPI/Cray
  crash history at large collective writes, and no cell in this workstream
  exercised it. Flagged, not flipped.
* **Reads are untouched.** `read_slab` still issues independent reads; they
  were never the bottleneck and changing them was out of scope.
* **`save_restart_state_per_proc` is the NEXT writer wall, and it is pure
  waste.** Immediately after the (now 2.2 s) canonical write, `gw_init.py:
  1013` calls it unconditionally; it `device_get`s
  `V_qmunu[..., vx0:vx1, vy0:vy1]` and `psi_full_y[..., py0:py1]` with
  **process-dependent bounds** and writes 144 files
  `tmp/isdf_tensors.rank{r}.x{i}.y{j}.h5` — ≈0.5 GB each, **≈72 GB** at
  c2406/P=144. Measured in job 7876530: node-0 RSS climbs monotonically
  ~5–7 GB/min from the moment the canonical write ends, consistent with
  materialising the full `(nq,μ,μ)` and `(nk,nb,ns,μ)` tensors per rank
  (13.34 + 4.87 GB × 2 ranks/node ≈ 36 GB/node of pure replication) before a
  single byte is written. **A `grep` over the whole tree finds NO reader of
  `V_local` / `psi_full_local` / `isdf_tensors.rank*.h5`** — the canonical
  restart is `isdf_tensors_2406.h5` and `load_restart_state_from_h5` reads
  only that. This is a dead 72 GB write behind a 36 GB/node gather, and it
  is also the one remaining place in the flow that builds a
  *process-dependent* slice of a globally-sharded array. It should be
  deleted or env-gated off by default. **Not touched in this workstream** —
  it is a different subsystem from the writer and doing it mid-flagship
  would have invalidated the one measurement this workstream exists to
  produce.
* **The stripe count is a static default, not a curve.** 32 was best of
  {0, 16, 32} at P=16 by 26 %; it is not swept against OST count or P.

---

## AK — the GW part is not the regression: it got FASTER. What the flop invariant is really detecting is that JAX's CPU collectives run on the 1 GbE management NIC (wt-G, branch `gw-part-forensics`, base eab0dd3, 2026-07-27 — NOT committed)

**One line: the razor says HEAD's GW half at 606c/P=80 is 3 % FASTER than the
July-25 tree with every physics observable BIT-IDENTICAL (sigX/sigC/sigXC/Eo
over 10 080 values; VBM/CBM/E_F to the last digit) and AD's sharded W measured
5.5× faster on the Dyson solve — so there is no introduced GW-part regression;
what the owner's invariant is detecting is that Σ's τ loop, which is
BYTE-IDENTICAL between the two trees, moves 9.2 GB/rank of reduce-scatter per
run and gets 55 MB/s/node for it, because Gloo is bound to Frontera's 1 GbE
`em1` (129.114.x.x) and not to `ib0` — the same wire that killed the end-to-end
razor twice. AK also lands −40.8 s (12.4 %) on Σ, bit-exact at P=80, and gives
the two previously-untimed GW stages their first stage rows.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AK/` — `gwpart_ab.sbatch`
(the razor), `gate_ak.sbatch` / `gate_cohsex.sbatch`, `ib_ab.sbatch` +
`ib_gloo_run.py`, `stage_diff.py`, `sigxc_cmp.py`.
Jobs **7876516** / **7876527** (end-to-end razor — both died, AK.5),
**7876528** (the razor in restart form, cells A/B/C), **7876522** / **7876531**
(fixture GN-PPM gate, pre- and post-freeze), **7876529** (COHSEX arm),
**7876536** (the transport A/B).

### AK.0 — the razor had to change shape, and the substitution is verified

The directive's razor was "rerun sweep_c606 end-to-end on HEAD and diff the
stage table". That run does not complete on HEAD (AK.5). So the razor was
sharpened rather than abandoned: all three cells **restart from the July-25
reference run's own `tmp/isdf_tensors_606.h5`** (written by job 7874609), which
makes V_q, ψ and E_nk bit-identical inputs and leaves the code that runs
screening → Σ → eqp as the only variable. 40 nodes × 2 ranks, mesh 8×10 — the
same shape as 7874609.

| cell | tree | what it is |
|---|---|---|
| **A** | `c3008f0` | the July-25 reference tree. NOT `b2abc0c`: 7874609 ran with the sigma-window pad still uncommitted in its working tree (without it an 8×10 mesh trips the reduce-scatter divisibility assert at window 70 — which is how we know it was there); `c3008f0` is the merge that captured it. |
| **B** | `eab0dd3` | current HEAD, unmodified |
| **C** | `eab0dd3` + AK | HEAD plus this workstream's changes |

**The substitution is measured, not assumed.** Cell A's `sigma_diag.dat`
reproduces the original in-line July-25 artifact **bit-for-bit on all 10 080
rows and all seven columns** (max\|Δ\| = 0.00e+00 on sigX / sigC.re / sigC.im /
sigXC.re / sigXC.im / VH / Eo).

### AK.1 — THE STAGE DIFF: no introduced GW-part regression, and one 5.5× win

| stage | A (July-25) | B (HEAD) | Δ | ratio |
|---|---|---|---|---|
| `gw_jax.restart_load` | 10.77 | 13.77 | +3.01 | 1.28 |
| `gw_jax.chi0_W` | 14.96 | **4.66** | −10.30 | **0.31** |
| — `W.exec` (the Dyson solve) | 12.42 | **2.24** | −10.18 | **0.18 — 5.5× faster** |
| — `chi.exec` | 1.32 | 1.33 | +0.01 | 1.01 |
| `gw_jax.sigma` | 356.99 | 353.16 | −3.83 | 0.99 |
| — `sigma.exec` | 331.44 | 329.01 | −2.43 | 0.99 |
| **Total recorded** | **382.72** | **371.60** | **−11.12** | **0.97** |

**Not one GW stage is slower on HEAD.** The 5.5× on `W.exec` is **AD.4's
sharded W solve, measured at production μ for the first time** — AD.6 listed
exactly this as named-not-done ("the *performance* claim for the shard is
unverified at the μ where it matters"). It is verified: 12.42 s → 2.24 s at
μ = 606, nq = 144, P = 80. `restart_load` +3.0 s is the O/AD sanity gates on
that path (`check_finite` on V_q / ψ / E_nk plus tr V_q[0] > 0).

**Physics gate (the directive's), A vs B over 10 080 (k,n):**

| column | max\|Δ\| | verdict |
|---|---|---|
| sigX, sigC.re, sigC.im, **sigXC.re, sigXC.im**, Eo | **0.00e+00** | **IDENTICAL** |
| VH | 6.027e+02 | differs *by design* — see below |
| E_F / VBM / CBM | — | −4.402191 / −5.252679 / −3.551704 eV on **both** |

ΔSigXC matches July-25 not to 1e-6 but to **zero**, on every value. The VH
*column* moves because HEAD resolves `hartree_source: auto → folded` on this
deck's `kin_ion_vh120_FIXED.h5` and prints the (correctly) suppressed ISDF
`sig_h` where July-25 printed the un-suppressed 602 eV quadrature. H₀ itself
suppressed it on both trees — which is why VBM/CBM/E_F agree to the last digit.
That is the S/AD seam reporting correctly, not a physics change.

### AK.2 — the invariant, measured on three decks

Owner's invariant: `V_q + tensors + screening + Σ  <  0.5 × ζ-fit`.

| deck | μ | P (mesh, nodes) | ζ-fit | V_q | χ₀→W | Σ | GW total | GW/ζ |
|---|---|---|---|---|---|---|---|---|
| MoS₂ 12×12 (Jul-25, 7874609) | 606 | 80 (8×10, 40) | 2023.4 | 66.3 | 18.2 | 341.6 | 426.4 | **0.211 PASS** |
| MoS₂ 4×4 (AJ, HEAD) | 402 | 4 (2×2, **1**) | 49.3 | 1.0 | 3.2 | 48.7 | 52.9 | **1.07 FAIL** |
| MoS₂ 4×4 (AJ, HEAD) | 785 | 16 (4×4, **8**) | 87.2 | 3.4 | 7.7 | 157.8 | 168.9 | **1.94 FAIL** |

Two corrections to that first row, both from AK's new instrumentation: the
July-25 GW total **excluded** the probe-ω W and the W0 restart write because
neither was timed (AK.6b). With them it is ≈455 s, i.e. **GW/ζ ≈ 0.225**, still
PASS. The invariant is not failing because of μ and not because of anything
HEAD introduced — it fails wherever the ζ-fit is *small*, because Σ has a floor
that shrinks with neither μ nor P.

### AK.3 — Σ does not strong-scale, and its τ loop is BYTE-IDENTICAL to July-25

`git diff c3008f0..eab0dd3` over `ppm_tau_kernel.py`, `ppm_accumulators.py`,
`ppm_windows.py`, `greens_function_kernel.py`, `fft_helpers.py`: **no changes at
all**. `wavefunction_bundle.py` gained five lines of docstring. The Σ compute
path on HEAD *is* the Σ compute path of July-25, so everything below is
**structural, not introduced** — the distinction the owner asked for.

Per-rank τ-kernel arithmetic is `nk·nb·μ_pad²/P` plus the two projector
einsums. Across the three measured cells that quantity is nearly constant
(402²/4 = 40 804 vs 800²/16 = 40 000), so the cells isolate everything that is
*not* arithmetic:

| cell | nodes | thr/rank | s per τ node | GFLOP/rank/τ | GFLOP/s/rank | **MFLOP/s/core** |
|---|---|---|---|---|---|---|
| 4×4 402c P=4 | **1** | 14 | 0.275 | 1.76 | 6.40 | **457** |
| 4×4 785c P=16 | 8 | 28 | 0.902 | 1.73 | 1.92 | **68.5** |
| 12×12 606c P=80 | 40 | 28 | 1.973 | 3.77 | 1.91 | **68.3** |

**(i) Per-rank throughput collapses 3.3× the moment the mesh leaves one node,
then stops changing** — 8 nodes and 40 nodes land on the *same* 1.9 GFLOP/s.
That is a flat communication floor, not a scaling law. **(ii) 68 MFLOP/s/core
against a ~35 GFLOP/s/core machine is 0.2 % of peak**, so "by flops" cannot
predict Σ's wall in any regime that spans nodes.

Collective payload, from the shard shapes (4 `psum_scatter`s per τ node — re
and im × the `'x'` and `'y'` axes):

| cell | MB/rank/τ | GB/rank per Σ stage | achieved |
|---|---|---|---|
| 4×4 402c P=4 (1 node) | 8.7 | 1.46 | 31.7 MB/s/rank |
| 4×4 785c P=16 (8 nodes) | 11.4 | 1.94 | 12.6 MB/s/rank |
| **12×12 606c P=80 (40 nodes)** | **54.6** | **9.17** | **27.7 MB/s/rank = 55 MB/s/node** |

### AK.4 — WHY: the collectives are on the 1 GbE management NIC, not InfiniBand

Job 7876527's failure text prints Gloo's peer addresses:

```
Gloo collective permute failed: [gloo/transport/tcp/pair.cc:547]
    Connection closed by peer [129.114.71.49]:5863
```

and on a Frontera compute node:

```
em1 : 129.114.x.x    1 GbE, MTU 1500      <-- every Gloo peer address is here
ib0 : 192.168.x.x    InfiniBand, MTU 4092
```

Every peer in every failure message (129.114.71.{29,31,34,39,41,42,48,49,50,
51,52,54,57,59,61}) is on **em1**. 55 MB/s/node is half of a 1 Gb link — Σ at
606c/P=80 is *saturating the management network*.

AF.5 found the mechanism and stopped one step short of the consequence: it
established that `GLOO_SOCKET_IFNAME` appears nowhere in the shipped
jax/jaxlib, so the line every sbatch in this campaign carries is inert, and
concluded "the transport offers no dial". **It offers one; it is not wired.**
`make_gloo_tcp_collectives(distributed_client, hostname=None, interface=None)`
takes the interface as a constructor argument, and
`jax/_src/xla_bridge.py:340` calls it with `distributed_client` only.
`register_backend_factory` overwrites `_backend_factories[name]` and refuses
only once a backend is already *initialized* — so re-registering `"cpu"` at
import time installs a factory that passes `interface=`.

`wk_AK/ib_gloo_run.py` is that wrapper. It is **deliberately NOT in the LORRAX
tree**: it changes the wire under every collective in the code and belongs in
`runtime/` only after a gate at 40+ nodes and an owner decision.

### AK.5 — the end-to-end razor does not run on HEAD (blocking, and not AK's)

Two independent 40-node runs of the *identical July-25 deck* on HEAD (jobs
**7876516**, **7876527**) both died in ζ **r-chunk 1**, ~4 min into the chunk
loop, where the July-25 tree completes that chunk in 205 s and the whole run in
53 min. Same failure both times:

```
Gloo context initialization failed: DEADLINE_EXCEEDED: GetKeyValue() timed out
  with key: cpu:gloo/81920,83968,86016,88064,90112,92160,94208,96256/0
  and duration: 29.999999929s
```

**Eight distinct 8-device subgroup contexts** — one per `'x'`-axis group of the
8×10 mesh — time out at exactly 30 s simultaneously; 62 of 80 ranks then die.
Before the timeout every rank is alive and burning ~50 % of one core with 27 of
28 threads idle (checked live over ssh: load average 0.13, RSS 3.0 GB, node
memory back down from the 20 GB chunk transient). A rendezvous stall, not a
compute hang and not OOM (MaxRSS 9.4 GB/rank of 96).

Handed over rather than absorbed. Bisection range **`c3008f0..eab0dd3`, 22
first-parent steps**; the ζ-chunk-loop changes inside it are `a1969c3`/`5a231b8`
(psi-gather all-to-all), `81ff092` (memory-ramp cure + per-q solve tier),
`c62c898` (AF's collective payload bounding) and `f484265` (the device_put
assert-collective). AJ's 4×4 deck runs a full pipeline in ~4 min and is the
right bisection vehicle. Note the razor's restart cells skip the ζ fit entirely
and run clean, which independently localises the fault to ζ.

### AK.6 — what AK changed, and what it bought

**(a) The screening stage is no longer silent.** Everything between the ζ-fit
and the first `Started sigma[...]` printed nothing, and for the dynamic modes
the *entire probe-ω W* — a second full χ₀ build and a second full-BZ Dyson
solve — had **neither a `timing.section` nor a print**: invisible in the stage
table *and* invisible while it ran. `screening.py` now carries a
`_ScreeningCadence`; every phase announces itself with a timestamp *before* it
starts (so the last line on screen names what the run is inside) and reports
its wall after, with a `LoopProgress` bar over the W evaluations in the same
format as the ζ chunk loop and the Σ τ sweep.

```
Started screening (chi0 -> W) at 03:44:41.
  [ 03:44:42 ] screening: W[static] chi0 build ...  (9 tau nodes, 9 q, mu=60)
  [ 03:44:42 ] screening: W[static] chi0 build done in 0.0 s
  [ 03:44:42 ] screening: W[static] Dyson solve ...  (9 q, mu=60, full BZ)
  ...
Finished screening (chi0 -> W) at 03:44:43.  Elapsed: 0 s.
```

`common/progress.py` gained `LoopProgress.start()` — `step()`'s lazy banner
prints only *after* the first iteration, which for a stage whose whole problem
is silence is exactly when it is no longer useful. Idempotent; a no-op for
every existing caller.

**(b) Two stages that had no row now have one**, and at 606c/P=80 they are not
small:

| new stage row | 606c / P=80 | note |
|---|---|---|
| `gw_jax.chi0_W_probe` | **3.62 s** (chi 0.80 + W 2.56 + compiles) | the probe-ω W; now split compile/exec like the static role, including the AOT precompiles it never had (different τ-node count and a full-BZ q extent mean neither kernel was a cache hit, so its compile was being charged to its exec) |
| `gw_jax.persist_w0` | **25.66 s = 7.1 % of recorded** | the W0 restart write — AF.4c's 1.7 MB/s, 2 h 55 m-of-silence stage, which sat between two timed stages with a row in neither. The write path is AE/AF's; this is the instrument, not the fix. |

**(c) `pad_sigma_window` over-padded both axes.** The precondition is
`m % p_x == 0` **and** `n % p_y == 0` — two independent one-axis constraints,
because m is reduce-scattered over `'x'` only and n over `'y'` only. It was
rounding *both* up to a multiple of the **product** `p_x·p_y`. Σ_c(ω,k,m,n) and
every per-τ tile, D2H copy and host accumulate feeding it scale as
`m_pad·n_pad`:

| mesh | P | window | old (product rule) | new (per-axis) | tile ratio |
|---|---|---|---|---|---|
| 2×2 | 4 | 8 (fixture) | 8×8 | 8×8 | 1.00 |
| 2×3 | 6 | 8 (fixture) | 12×12 | 8×9 | **2.00** |
| 8×10 | 80 | 70 (sweep_c606) | 80×80 | 72×70 | **1.27** |
| 8×8 | 64 | 70 | 128×128 | 72×72 | **3.16** |
| **12×12** | **144** | **70** | **144×144** | **72×72** | **4.00** |
| 12×12 | 144 | 80 (AF flagship) | 144×144 | 84×84 | **2.94** |

The product rule was on a collision course with the mesh the campaign is moving
to: Y.4 recommends square meshes for two independent reasons, and on a square
mesh the rule is at its worst. `strip_sigma_window` now tests **both** trailing
extents — with independent pads one axis can sit at the real extent while the
other is padded (8×10, window 70 → m = 72, n = 70), and testing only the last
axis would have returned an m-padded Σ untouched.

**MEASURED, cell B → cell C at 606c/P=80:**

| stage | B (HEAD) | C (HEAD+AK) | Δ | ratio |
|---|---|---|---|---|
| `sigma.exec` | 329.01 | **288.25** | **−40.77** | **0.88** |
| `gw_jax.sigma` | 353.16 | 312.58 | −40.57 | 0.89 |
| Total recorded | 371.60 | 359.54 | −12.06 | 0.97 |

(Total falls by only 12 s because C also *adds* the 29.3 s of work that B ran
untimed — 3.62 probe + 25.66 persist_w0. The Σ win is the full 40.8 s.)

### AK.7 — gates

| gate | result |
|---|---|
| **razor cell B vs C at P=80**: `eqp0` / `eqp1` / `eqp_g0w0` / `sigma_diag` | **BYTE-IDENTICAL** (×4) |
| same, `sigma_diag` column-by-column over 10 080 values (all 7 columns) | **max\|Δ\| = 0.00e+00** |
| fixture **GN-PPM P=6** (2×3 — where the pad change is LIVE, 12×12 → 8×9): `eqp0`/`eqp1`/sigma dump | **BYTE-IDENTICAL** (×3) |
| fixture **GN-PPM P=4** (2×2 — pad identical; isolates the cadence/timing diff) | **BYTE-IDENTICAL** (×3) |
| fixture **COHSEX P=4** (the single-request arm of the cadence, which GN-PPM never exercises) | **BYTE-IDENTICAL** (×3) |
| fixture COHSEX vs `eqp_ref.dat`, base and branch | **PASS**, max\|Δ\| = 1.00e-06 eV both |
| re-gated after the source freeze (the probe AOT precompiles landed mid-campaign) | **identical md5s to the pre-freeze run** |

Every AK change is either print-only or a pure shape change on an axis that is
a *batch* index of every contraction it enters, so bit-exactness is structural
— and it is also measured, at four mesh shapes and in both compute modes.

### AK.8 — files touched (worktree only, NOT committed)

`src/gw/screening.py` (`_ScreeningCadence`; the probe-W timing sections and its
AOT precompiles), `src/common/progress.py` (`LoopProgress.start()`),
`src/gw/gw_jax.py` (`timing.section("gw_jax.persist_w0")`),
`src/gw/ppm_sigma.py` (`pad_sigma_window` per-axis, `strip_sigma_window` both
extents, `_run_sigma_branch` `m_pad`/`n_pad`). 231 insertions, 39 deletions.

### AK.9 — named, not done

* **The transport fix is a wrapper, not a landed change.** `ib_gloo_run.py`
  re-registers the CPU backend factory; putting it in `runtime/` needs a gate
  at 40+ nodes and an owner decision.
* **Σ's 4 collectives per τ node could be 2.** The re/im channels each run two
  `psum_scatter`s; stacking the two `left_partial`s (and the two
  `result_partial`s) before the reduce halves the message count at identical
  bytes and is bit-exact by construction (a rank-wise elementwise sum cannot
  care that two independent arrays were concatenated). Not landed: at 6.5 MB
  per message the payload looks bandwidth-bound, so this is worth doing *after*
  the transport question is settled, not before.
* **`_to_host_np(sigma_kij, tiled=False)` is a P-INDEPENDENT gather** — once per
  Σ branch, `n_ω_branch·nk·nb²·16` bytes onto **every** rank (≈237 MB/branch at
  606c/b160, growing as nb²). Same family as Y.3's `F_tensor_write`. Untouched:
  it feeds both the h5 write and the eqp path and deserves its own workstream.
* **`check_hermitian(W[0])` now transposes a 2-D-sharded tile** (AD made W
  `P(None,'x','y')`), i.e. an all-to-all where it used to be a device-local
  transpose. 0.086 MB at 606c, 96 MB at c2406 — measured negligible, recorded
  so nobody rediscovers it.
* **V_q was not re-measured.** The restart razor loads V_q from the tensors, so
  `V_q_compute` (66.3 s on July-25) is the one GW stage without an A/B. The
  end-to-end razor would have covered it; AK.5 is why it did not.

### AK.10 — THE MEASUREMENT: pinning Gloo to `ib0` is 3.3× on the whole pipeline, bit-identical (job 7876536)

AK.4 was an inference from failure text. This is the experiment. AJ's 4×4
785c deck, P=16 (4×4 mesh), 8 nodes × 2 ranks × 28 threads, HEAD source,
**both cells launched through the same `ib_gloo_run.py` wrapper** so the only
difference between them is the `interface=` argument:

| stage | STOCK (em1, 1 GbE) | **ib0** | ratio |
|---|---|---|---|
| **Total recorded** | **247.7** | **75.2** | **0.30 — 3.3× faster** |
| `gw_jax.zeta_fit_chunked` | 83.09 | **22.25** | **0.27** |
| — `zeta_fit.chunk.solve` | 48.90 | **3.57** | **0.07 — 13.7×** |
| — `zeta_fit.chunk.z_q_build` | 24.64 | 12.93 | 0.52 |
| — `zeta_fit.write_g_flat` | 2.86 | 0.98 | 0.34 |
| `gw_jax.V_q_compute` | 3.45 | 0.91 | 0.26 |
| `gw_jax.chi0_W` | 5.57 | 4.64 | 0.83 |
| `gw_jax.sigma` | 153.19 | **45.55** | **0.30** |
| — `sigma.exec` | 148.82 | **42.71** | **0.29** |
| compile-only rows (`chi.compile`, `W.compile`, `sigma.compile`, `CCT`, `slice_halves`) | — | — | **1.00 ± 0.05** |

**Physics gate: `eqp0.dat`, `eqp1.dat`, `sigma_diag.dat`, `eqp_g0w0.dat` —
BYTE-IDENTICAL.** Which NIC carries a reduction cannot change the reduction,
and it does not.

The control is honest twice over: the stock cell reproduces AJ's independently
measured baseline (ζ 83.1 vs 87.2 s, `sigma.exec` 148.8 vs 153.4 s in job
7876526), and **every compile-only row is unchanged at ratio 1.00** — exactly
the rows that do no collective. Only communication moved.

**Consequences.**

1. **Every number this campaign has measured was taken on a 1 Gb Ethernet
   link.** Every scaling wall, every "the collective is too big" failure, AF's
   1.15 GB AllGather death at P=144, AC.2's 30-minute `pzheevd`, and AK.5's
   rendezvous storm are all observations of `em1`, not of the algorithm. The
   J.2/T.7/Y.3 residency-and-collective ledgers stay valid as *byte* counts;
   their *wall-time* consequences were all measured against a link ~100×
   slower than the machine's real fabric.
2. **AF.1's payload chunking was the right fix for the wrong reason.** Bounding
   collectives to 128 MB made P=144 survivable on a saturated 1 Gb link. It
   should be re-priced on `ib0` before it is treated as a permanent constraint.
3. **It does NOT by itself restore the owner's invariant.** On the 4×4 deck
   GW/ζ goes 1.96 → 2.31, because ζ speeds up *more* than Σ (0.27 vs 0.30) —
   the ζ back-solve is the most collective-bound stage in the code (13.7×).
   So the invariant question and the transport question are genuinely separate,
   and AK.2/AK.3's structural finding stands: Σ has a floor that neither μ nor
   P shrinks. The transport fix moves the floor down 3.4×; it does not remove
   it.
4. **The 12×12 flagship is the one that matters and is NOT yet measured on
   `ib0`.** Everything above is the 4×4 deck at P=16. The 606c/P=80 and
   c2406/P=144 cells are the obvious next runs, and AK.5's ζ wedge — itself a
   Gloo *rendezvous* failure — is a prime candidate to simply evaporate on a
   fabric that is not saturated.

**Status: NOT landed.** `wk_AK/ib_gloo_run.py` is a wrapper, not a tree change.
Landing it means calling
`xla_bridge.register_backend_factory("cpu", <factory passing interface=…>)`
from `runtime/bootstrap()` before any backend touches `jax.devices()`, with the
interface name resolved per-machine (it is `ib0` on Frontera; it must not be
hard-coded) and a clean fallback when the interface does not exist. That is a
small change with a very large blast radius and it wants an owner decision plus
a gate at 40+ nodes, which is why AK measured it and stopped.

## AL — the Gloo→ib0 pin is LANDED runtime code and GATED at the flagship: the AK.5 ζ wedge was an em1 artifact, the 606c/P=80 pipeline now runs end-to-end in 544 s, and the invariant's em1 "PASS" was itself a transport artifact (wt-G, branch `al-gloo-ib-pin` @ 6ed2414, base 5c0b006, 2026-07-27 — COMMITTED on the branch, not merged)

**One line: `runtime.pin_gloo_interface()` (bootstrap-integrated, auto-detect
`ib*`/`hsn*`, `LORRAX_GLOO_IFNAME` override, loud rank-0 announcement, degrade-
to-stock on every failure path) is in the tree and gated at 40 nodes: the
end-to-end 606c/P=80 run that em1 could not complete AT ALL (AK.5's twin
rendezvous deaths) finishes in **544 s wall** (July-25 needed 3193 s), the ζ
chunk that killed jobs 7876516/7876527 passes in **16.6 s** (was 205 s on
July-25 em1), and all four output artifacts are BYTE-IDENTICAL across
em1-restart / ib0-restart / ib0-full-recompute / ib0-full-with-compile-cache —
five-way, at P=80.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AL/` (`smoke_pin.sbatch`,
`gate40_ib0.sbatch`, `smoke.7876540.out`, `gate40.7876541.out`); run dirs
`$SCRATCH/lorrax_mos2_12x12/al_{e2e,rs,cache}_ib0/`. Jobs **7876540** (1-node
smoke), **7876541** (40-node gate, three cells, dev queue, 17m33s total).

### AL.1 — what landed (commit 6ed2414, 198 insertions)

`src/runtime/__init__.py::pin_gloo_interface()`, called from `bootstrap()`
between `set_default_env()` and `init_jax_distributed()` (i.e. before any
backend exists). Mechanism = AK.10's PoC made defensive: re-register the
`"cpu"` backend factory with a wrapper that builds
`make_gloo_tcp_collectives(distributed_client=..., interface=<iface>)` and
forwards everything else to the stock `make_cpu_client`. Scope guards, each
verified by a smoke cell:

* single-process → silent no-op; `JAX_PLATFORMS != "cpu"` → silent no-op
  (GPU collectives are NCCL's; NCCL_SOCKET_IFNAME still applies there);
* iface resolution: `LORRAX_GLOO_IFNAME` override (name, or `off|none|0`),
  else auto-detect first UP `ib*` then `hsn*` (Perlmutter) with an assigned
  IPv4 (`/sys/class/net` + SIOCGIFADDR, no deps);
* EVERY failure path — no fabric NIC, name not UP/no IPv4, jax internals
  moved, factory already initialized, collectives constructor raises —
  prints a loud `[runtime]` reason and falls back to stock. Never crashes,
  never hangs. An unroutable pinned iface fails at the first collective
  with Gloo's 30 s timeout → exception → failfast hook kills the job (not a
  hang); `LORRAX_GLOO_IFNAME=off` is the escape hatch.
* The decision is ALWAYS announced on rank 0 (quality-pattern #8: env may
  grant capability, but the flip is loud):
  `[runtime] Gloo collectives pinned to ib0 (192.168.x.x; auto-detected
  high-speed fabric). The jax default binds the coordinator-route NIC — em1,
  1 GbE, on Frontera compute nodes.`

Docs: `docs/dev/env_vars.md` (§2 `LORRAX_GLOO_IFNAME`, §6 the
GLOO_SOCKET_IFNAME-is-inert row), `LORRAX_FRONTERA_ADVICE.md` §10b (the
whole story) + a correction box in §6 (whose "GLOO_SOCKET_IFNAME is THE
important one" was wrong).

### AL.2 — smoke (job 7876540, 1 node × 2 ranks, GN-PPM fixture)

Three cells through the LANDED code, differing only in `LORRAX_GLOO_IFNAME`:
auto (pinned ib0, announced), `off` (announced disable, stock), `zz9`
(announced skip, degraded to stock). All rc=0, `eqp_test.dat`
**BYTE-IDENTICAL ×3**.

### AL.3 — GATE (a): the transport-only A/B at P=80 — restart razor, same src, same tensors

`al_rs_ib0` = AK's razor cell C bit-for-bit (HEAD+AK src, restart from job
7874609's own `isdf_tensors_606.h5`, 40 nodes, mesh 8×10) with ONLY the
transport changed. Vs `gwab_C` (em1, job 7876528):

| stage | em1 (gwab_C) | ib0 (AL) | ratio |
|---|---|---|---|
| `sigma.exec` | 288.25 | **62.04** | **0.215 — 4.6×** |
| `gw_jax.sigma` | 312.58 | 69.26 | 0.222 |
| `gw_jax.persist_w0` | 25.66 | 6.01 | 0.23 (h5py_allgather writer: its gather leg was em1 too) |
| `chi0_W` / `W.exec` | 4.84 / 2.42 | 3.16 / 0.77 | 0.65 / 0.32 |
| `chi0_W_probe` | 3.62 | 2.40 | 0.66 |
| `restart_load` | 12.83 | 10.99 | 0.86 |
| compile rows (chi/W/sigma.compile) | — | — | **0.85–1.05 ≈ 1.00** |
| **Total recorded** | **359.54** | **91.83** | **0.255 — 3.9×** |

Vs `gwab_B` (em1 HEAD without AK's pad fix): sigma.exec 329.01 → 62.04 =
**5.3×**. Cell wall 141 s. Only communication moved — AK.10's P=16 control
structure reproduces at P=80.

**Bit-identity (the AK cell-A standard):** `eqp0` / `eqp1` / `sigma_diag` /
`eqp_g0w0` **BYTE-IDENTICAL to gwab_B AND gwab_C** (md5 e5602fbc90… etc.),
and vs job 7874609's original artifact `sigma_diag` matches on **sigX /
sigC.re / sigC.im / sigXC.re / sigXC.im / Eo at max|Δ| = 0.00e+00 over all
10 080 rows** (VH column moves by the documented `hartree_source` seam,
AK.1 — byte-identity vs the July-25 file is impossible BY DESIGN on HEAD for
that one column, which is why the gate is stated per column).

### AL.4 — GATE (b): the ζ-fit blocker is CLOSED — it was an em1 saturation artifact

`al_e2e_ib0` = the FULL sweep_c606 deck (restart=false), env byte-identical
to the two em1 runs that died (7876516/7876527), the landed pin the only live
change. **rc=0, wall 544 s.** The eight 8-device `'x'`-axis Gloo subgroup
contexts that timed out at exactly 30 s simultaneously on em1 all
established instantly (160/80/80 `[Gloo] … connected` lines, zero
`DEADLINE_EXCEEDED`), and r-chunk 1 — where both em1 jobs died ~4 min in —
completed in ~20 s. **No bisection of c3008f0..eab0dd3 was needed: there is
no code regression.** What changed between "July-25 completes in 205 s/chunk"
and "HEAD dies in chunk 1" is plausibly HEAD's extra subgroup communicators
meeting an em1 already at saturation; on ib0 the question is moot at this
scale. Full stage table vs the July-25 reference (7874609, em1 — note this
comparison bundles transport with the merged HEAD improvements; AL.3 is the
transport-only isolation):

| stage | Jul-25 em1 | ib0 AL | ratio |
|---|---|---|---|
| `load_centroid_wfns` | 295.0 | 96.3 | 0.33 (phdf5_host + §5b; 3.9 s warm — see AL.5) |
| `zeta_fit_chunked` | 2023.4 | **211.4** | **0.104 — 9.6×** |
| — chunk loop (9 chunks) | 1847.2 (205 s/chunk) | **147.5 (16.6 s/chunk)** | **0.080 — 12.5×** |
| — z_q_build | 1216.2 | 109.8 | 0.090 |
| — back-solve | 630.0 | **36.5** | **0.058 — 17.3×** |
| `V_q_compute` | 66.3 | 19.2 | 0.29 |
| `chi0_W` (+probe, ib0) | 18.2 | 6.8 (+5.5) | — |
| `gw_jax.sigma` | 341.6 | 73.8 | **0.216** |
| — `sigma.exec` | 311.8 | 62.0 | **0.199 — 5.0×** |
| **Total recorded / wall** | **2744.8 / 3193** | **419.8 / 544** | **0.153 / 0.170** |

Physics: this cell RECOMPUTES the whole ζ → V_q → W → Σ chain from WFN.h5,
and its `sigma_diag` is **byte-identical to the restart cell's** and
column-zero vs July-25 — the recomputed-tensor chain reproduces the
July-25-tensor chain exactly, on a different tree and a different wire.

**V_q A/B closed (the one GW stage AK could not measure — AK.9):**
`V_q_compute` = 19.2 s on HEAD/ib0 (15.0 s warm) vs 66.3 s on July-25/em1.
Scope: cross-tree AND cross-transport (an em1 HEAD number cannot exist —
that configuration dies in ζ before reaching V_q); no pure-transport V_q
ratio at P=80 is claimable from this campaign.

### AL.5 — GATE (c): pin × AH compile-cache repair — no interaction

`al_cache_ib0` = the same full deck with the repaired P>1 cache ON (fresh
`$SCRATCH` dir). rc=0, wall 365 s, outputs **byte-identical** to both other
cells. Banner sequence healthy under the pin: `[compile-cache] ARMED at 80
processes, shared dir …/np80 (0 entries advertised, 0 agreed…; agree+prefetch
0.06s)` → `cold cache: nothing to reuse…` → per-rank summaries
(`enabled=True`, 433 probes, 433 vetoed — correct for a cold dir) → 347
entries populated by process 0 for the next run. The KV agreement (runs on
the coordination service, not Gloo) is indifferent to the pin, as predicted;
verified on the cold path only (a warm-hit run under the pin was not part of
this gate). Bonus reproducibility datum: `zeta_fit_chunked` 211.39 vs 211.41 s
and `sigma.exec` 62.10 vs 62.03 s across the two full cells — the ib0 numbers
are stable to ~0.1%, page cache being the only meaningful cell-to-cell delta
(`load_centroid_wfns` 96.3 → 3.9 s).

### AL.6 — the invariant and the Σ floor, priced on a real fabric (the AK.2/AK.3 sequel)

Owner's invariant `GW ≤ 0.5 × ζ-fit`, AK's corrected accounting (GW = V_q +
χ₀→W + probe-W + persist_w0 + Σ):

| | ζ-fit | GW total | GW/ζ |
|---|---|---|---|
| Jul-25 em1 (AK.2 corrected) | 2023.4 | ≈455 | **0.225 "PASS"** |
| **ib0 HEAD (AL, e2e cell)** | **211.4** | **111.7** | **0.53 FAIL** |

**ζ sped up 2.3× more than GW (9.6× vs 4.1×; back-solve 17.3× vs Σ 5.0×) —
exactly AK.10-consequence-3's direction — so the pin does not restore the
invariant; it REVEALS that the em1 "PASS" was an artifact: ζ was even more
transport-crippled than Σ, and fixing the wire re-weights the ratio toward
Σ's structural floor.** (Under the directive's narrower GW = V_q+W+Σ the
ratio is 0.47 — the verdict straddles 0.5 depending on whether the two
stages July-25 never timed are charged; either way the margin is gone.)

What ib0 closes and what remains, in Σ τ-loop terms (AK.3's metric,
`sigma.exec`/168 τ nodes, 3.77 GFLOP/rank/τ, 28 cores/rank):

| | s/τ node | MFLOP/s/core | % of 1-node roofline (457) |
|---|---|---|---|
| em1, P=80 (AK.3) | 1.973 | **68.3** | 15% |
| **ib0, P=80 (AL)** | **0.369** | **365** | **80%** |

The em1 flatline ("8 and 40 nodes land on the same 1.9 GFLOP/s") is GONE —
achieved collective bandwidth went 55 → ~296 MB/s/node (54.6 MB/rank/τ ÷
0.369 s). **The remaining invariant gap is structural, not transport:**
~20% residual τ-loop comm/imbalance, plus the fixed GW stages (V_q 19 s,
persist_w0 6 s, probe 5 s), against a ζ-fit that keeps shrinking. Next
levers stay AK.9's: halve Σ's 4 psum_scatters/τ, the P-independent
`_to_host_np` gather, and re-pricing AF.1's 128 MB payload bound on ib0.

### AL.7 — claim-decay housekeeping (pattern #9), applied

* **AF.5's "the transport offers no dial" — ⚠ REFUTED banner added in place**
  (the factual substrate stands; the conclusion doesn't; the dial is now
  first-class code).
* **⚠ em1-scope banners added to sections R, overnight-A/B, T, X, Y, AA, AC,
  AD, AF** (multi-node Gloo collective walls) and a fabric-unverified note on
  V/W (`pzheevd` is Intel-MPI; its provider was never measured). Byte counts
  and HLO inventories in those sections remain valid; wall-times of
  collective-bound stages re-price 4–17× where re-measured here.
* `LORRAX_FRONTERA_ADVICE.md`: §6 correction box (inert GLOO_SOCKET_IFNAME),
  new §10b (the story + tables + the blanket pre-2026-07-27 em1 scope
  statement).

### AL.8 — named, not done

* **The pin is committed on `al-gloo-ib-pin` (wt-G), NOT merged** — orchestrator
  merge + a Perlmutter `hsn*` smoke (the pattern is coded and unit-exercised,
  never run on a real hsn machine) remain.
* **A warm-cache run under the pin** (AL.5 verified the cold path only).
* **c2406/P=144 on ib0** — the next scale jump; every P=144 number in this
  ledger is em1-scoped, and AF.1's payload bound + AC's flagship economics
  should be re-priced there first.
* **em1-side V_q on HEAD** is unmeasurable (dies in ζ); the V_q A/B is
  cross-tree by necessity (AL.4).
* The e2e cell's `load_centroids.reshard` 92.6 s (cold) is now the dominant
  load cost at this shape — untouched here, noted for the I/O owner.

## AM — defaults alignment: slab_io=auto unconditional, FFI I/O default, C++ collective-write flip (2026-07-27, wt-H @ defaults-alignment 2176127)

Scope: owner directive "default to all of it on an arbitrary input file"; no
performance claims beyond parity gates.  Branch: `defaults-alignment` off
9f20cb6 (4 commits + docs).  NOT merged — orchestrator merges.

1. **`slab_io=auto` now routes unconditionally** (fixes the merged-tree
   finding: auto was inert unless `use_ffi_io=true`, gw_config.py:1544,
   pattern #8).  CPU ladder unchanged (FFI → HOST → ALLGATHER); NEW GPU
   ladder: PHDF5_FFI iff CUDA lib exports the write handler AND single-node
   (cross-node GPU FFI = known MPI bring-up failure → announced demotion),
   then PHDF5_HOST (probe requires real MPI_Init + 1 device/proc), then
   ALLGATHER.  The PHDF5_HOST tier probe now runs `from mpi4py import MPI`
   for real, so a PMI-mismatched harness (MPI_Init error 16) demotes with
   the srun --mpi=pmi2 / I_MPI_PMI_LIBRARY hint instead of dying — a probe
   never kills the run; explicit `slab_io=phdf5_host` still fails loudly.
2. **`use_ffi_io` demoted to deprecated tri-state**: unset (None, the new
   default) → router; `false` → forced h5py_allgather (DeprecationWarning);
   `true` → no-op notice; ignored (with notice) when `slab_io` explicit.
   Parse matrix (6 cells: none/false/true/explicit/both/host) PASS in-container.
3. **C++ writer default flip** (context.cc): `LORRAX_PHDF5_COLLECTIVE_WRITES`
   default 0 → 1, matching the Python phdf5_host writer, so the env var means
   the same thing in every writer.  Added rank-local replica dedup to
   write_ffi.cc (canonical writer = coord 0 on every unconsumed mesh axis; no
   allgather, unlike the Python dedup) because overlapping selections are UB
   under collective MPI-IO.  `LORRAX_PHDF5_DEDUP_REPLICAS=0` disables.
   Host lib rebuilt from wt-H → `lorrax_ffi_unified/build_host_AM` (11
   targets).  **CUDA lib also rebuilt from wt-H** (`lorrax_ffi_wtH_cuda/
   build_phdf5`) and gated on rtx — the CUDA gate is NOT pending.

Gates (all PASS):
- **Write bit-compare** (P=4, holder 7876938): new defaults vs
  collective=0/dedup=0 vs serial-h5py oracle — 15/15 datasets BITCMP_OK,
  including P('x',None) replica groups, fully-replicated, valid_shape-padded.
- **CPU P=16 minimal gw.in** (785c, 8 nodes, NO slab_io/use_ffi_io keys):
  with host lib → router announces PHDF5_FFI; without → PHDF5_HOST +
  `collective_write=True dedup_replicas=True stripe_count=16` banner.  Both
  rc=0; **eqp0 data-identical to run_800c AND run_800c_merged**; ζ-fit 21.9 s
  (FFI) / 22.5 s (HOST) vs 22.6 s baseline — within noise.  V_qmunu restart
  write 25.9 GB/s through the flipped C++ collective default.
- **CPU P=1** (108c, mesh 1×1): rc=0, router → PHDF5_FFI, eager read backend,
  streamed paths fine (66.8 s total).
- **GPU rtx-dev smoke, BARE input** (job 7876949, 1 node × 4 RTX 5000):
  g108 rc=0 (182 s), g402 rc=0 (78 s).  Router: "slab_io=auto on GPU
  backend: CUDA FFI exports ... (single-node) → PHDF5_FFI".  Gloo pin
  silent no-op on GPU confirmed (0 `[runtime] Gloo` lines).  **eqp0 GPU vs
  CPU: 402c max|Δ| = 1.0e-9 eV; 108c max|Δ| = 0.0** (5160 values each,
  tol 2e-5).  Multi-process GPU compile cache worked (283 hits, agreed
  350/350) — AH's fix holding on rtx.
- Verified already-landed defaults engage with zero input keys on both
  platforms: collective writes ON, dedup ON, 16×4 MiB stripe hints ON
  (advisory-only confirmed), ib0 pin ON w/ announced fallback (CPU) /
  silent no-op (GPU), per-proc restart dump OFF.

Portability audit + deprecation inventory: see workstream AM final report.
Artifacts: /scratch2/08271/jackmc/lorrax_setup/wk_AM/ (write_gate.py,
config_matrix.py, run.sh, rtx_smoke.sbatch, logs), runs in
/scratch2/08271/jackmc/mos2_4x4_test/run_am_p16{,b}, run_am_p1, run_gpu_g108,
run_gpu_g402.

## AO — centroid-load collective hygiene: the two ledgered device_put one-liners closed, a src/-wide assert-collective sweep, and the reshard-92.6s verdict (wt-F, branch `centroid-load-collectives` @ 749d9c7, base 19aeece, 2026-07-27 — COMMITTED on the branch, not merged)

**One line: the AA.5 "named, not done" pair (`centroid/kmeans_isdf.py::shard`,
`centroid/pivoted_cholesky.py` orbit_id) plus ~24 more
`jax.device_put(<host array>, <multi-process sharding>)` call sites across
gw/, file_io/, ffi/linalg/, bse/ and the eigh benches are converted to
`device_put_process_local` (AA.1 mechanism; `LORRAX_CHECK_REPLICA=1` re-arms
the assertion); the 92.6 s `load_centroids.reshard` ledger item is NOT data
movement — it is cold-start process SKEW absorbed by the pipeline's FIRST
collective, now charged to its own `load_centroids.pre_reshard_sync` row; and
the real μ·nb-scaling cost underneath (K.2's involuntary-remat full all-gather,
471.9 MB/rank at 606c/8×10) is REMOVED by splitting `_reshard_all` into two
per-axis staged chains — verified by a compile-level A/B/C at the production
8×10 mesh: remat warning 1 → 1 → **0**, biggest collective 471.9 → **59 MB**.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AO/` (cell scripts,
gw16.log, gw80.log, probe.log + reshard_probe.py, hlo_800c/ + hlo_606c/ rank-0
dumps, km4_*.log, pytest logs, master.log); runs in
`/scratch2/08271/jackmc/mos2_4x4_test/run_800c_AO` and
`/scratch2/08271/jackmc/lorrax_mos2_12x12/sweep_c606_AO{,2}`.

### AO.1 — the two one-liners (ledger closed) + the sweep

Converted (all with the AA.1 comment + escape hatch at the call site):
`centroid/kmeans_isdf.py::shard` (positions/ρ/metric/centroids — the largest
host arrays the centroid driver touches), `centroid/pivoted_cholesky.py`
orbit_id (P×M×4 B), `gw/v_q_g_flat.py` v_q table (P·nq·ngkmax·8),
`gw/degen_average.py` (six post-Σ (nk,nb,nb) c128 matrices),
`gw/sc_iteration.py` H0/E_qp/U_qp, `gw/sigma_dispatch.py` sig_x,
`file_io/_slab_io_ffi.py` `_replicated_i64_vector` (a per-write blocking
collective on a control buffer) + the write-path replicated fallback (P × the
FULL written tensor when it fires), `file_io/_slab_io_allgather.py` read
fallback, `ffi/linalg/plan.py::ensure_sharding` host-operand branch,
`bse/bse_w_exact.py` (rolled ψ_c ×2 + eps_c per finite-q point + probe seed
via a zero-copy `np.broadcast_to` view), `bse/w_omega_chain.py` probe seed,
`bse/vq_interp.py` (build_cq q-chunk loop ×2 + EqR + prepare_coarse chunk
loop ×5 + V_SRc host mirror), `bse/exciton_bands.py` refit V_pad,
`bse/bse_io.py` g0_X/g0_Y, `common/eigh_block_sweep.py` + `eigh_benchmark.py`
(bench operands — the assert distorted the thing being benchmarked).

**Audited and deliberately left**: `isdf/core.py:1520` (committed global
array → the reshard branch, no assert), `exciton_bands` V_q0/V_q_full/
V_stack/W_R and `bse_io` bse_k_grid puts (same — global/committed operands),
`runtime/__init__.py` psum warmup (tiny, and warming the wire is its job),
`gw/kin_ion_io.py` local-device puts + its process_allgather (a REAL gather
of rank-local values, not the antipattern).

### AO.2 — gates

| gate | result |
|---|---|
| 785c/P=16 e2e (8 nodes, PHDF5_HOST route, wt-F): eqp0/eqp1/eqp_g0w0/sigma_diag data rows | **BIT-IDENTICAL to run_800c (AJ baseline) AND run_800c_merged** (md5 201b31e9…/af09d549…/4236fe49…/4b4b5716…) |
| centroid strict contract: `kmeans_cli 400 --orbit` on the 4×4 deck, 4-device mesh, base vs wt-F | **centroids_frac_402 BYTE-IDENTICAL (md5 59158341…) — and identical to the deck's original single-device file** (tree AND device-count invariance) |
| fixture GW P=1 (the process-local degradation path) vs eqp_ref, COMMITTED tree | **PASS, max\|Δ\| = 1.0e-06 eV** (0/1888 over tol — the established ref tolerance) |
| fixture GW P=4 vs eqp_ref, COMMITTED tree | **PASS, max\|Δ\| = 1.0e-06 eV** (0/1888) |
| **P=4 vs P=1 directly** (multi-process placement vs delegation branch of the helper) | **data rows BYTE-IDENTICAL** (md5 0826b167…) |
| **606c/P=80/8×10 full e2e** (fresh ζ→V_q→W→Σ, sweep_c606_AO, job-step under holder 7876986): eqp0/eqp1/sigma_diag data rows | **BYTE-IDENTICAL to AL's al_e2e_ib0 AND al_cache_ib0** (sigma_diag md5 e5602fbc90… — AL.3's exact artifact) |
| touched-module pytest (loader/kmeans/ffi-contract/file_io/vq_interp/w_omega_chain) | **41 passed, 44 skipped; all 14 failures reproduced bit-for-bit on untouched 19aeece** — pre-existing: 13× test_file_io zeta readers vs the NEW `zeta_is_done` provenance gate (stale fixtures never stamp it), 2× test_bse_w_omega_chain shard_map varying-manual-axes carry mismatch in `bse_ring_comm.apply_V_ring` (`lax.fori_loop` carry `A0` unvarying vs varying output; JAX-version strictness).  Both ledgered below. |

### AO.3 — the reshard-92.6s verdict: SKEW, not movement

Job 7876541's own cells are the controlled experiment nobody had read
closely: cell 1 (`al_e2e_ib0`, the allocation's FIRST srun) vs cell 3
(`al_cache_ib0`) are identical code / identical 40 nodes / 433 XLA compiles
each / compile-cache 0 hits in BOTH — and `load_centroids.reshard` = 92.609
vs **0.550 s**.  Compile is excluded (13 s total, both cells), collectives
identical, data identical.  The only cell-to-cell difference is cold node
caches, and the unrecorded startup wall (srun→first timing row) is 124 s
(cell 1) vs 46 s (cell 3).  On the PHDF5_HOST / §5b process-local loader
routes NOTHING synchronizes before `_reshard_all` (independent h5py reads,
local FFTs) — so the first collective's wall on rank 0 IS the max startup
skew of 80 ranks importing python/jax from cold Lustre.  AL.5's "page cache"
attribution was right about the cause but the row it lands in is the reshard.
Every other recorded reshard row in the campaign is 0.13–3.3 s (grep across
wk_AK/wk_AG/deck/validation logs), and only first-srun-of-allocation cells
show the big number.  **Landed: a named timed `collectives.barrier` row,
`load_centroids.pre_reshard_sync`, immediately before the reshard section**
— skew now has its own line and `.reshard` reports the collective itself
(pattern #5: the observable now discriminates).

### AO.4 — the structural cost underneath, REMOVED: `_reshard_all` per-axis chains

K.2's involuntary remat (the μ:'y'→μ:'x' move on the transposed ψ_rμ —
x-major↔y-major device order, XLA b/433785288) is real and μ·nb-scaling:
471.9 MB/rank full-tensor all-gather at 606c/b160/8×10, 4.9 GB/rank projected
at 2406c/b432.  Fix: two independent staged chains from the band-sharded
input — `b:(x,y)→b:'y'→μ:'y'` (psi_rmu) and `b:(x,y)→b:'x'→transpose→μ:'x'`
(psi_rmuT) — each a partial gather over the OTHER axis + one same-axis
band→μ all-to-all.  **Trap found on the way (pattern #4)**: with the μ-pad
applied BEFORE the chains, CSE hands ONE pad output to both chains, the
partitioner pins it to b:'y' and chain 2's b:'x' demand re-creates the exact
same remat on the padded tensor (measured live in gw80 attempt 1 AND in the
compile probe).  The committed form pads per-chain AFTER the stage
constraint.  Compile-level A/B/C at forced 8×10 (wk_AO/reshard_probe.py,
production shape 144×160×2×606→640):

| variant | SPMD remat warning | biggest collective |
|---|---|---|
| A: old single-chain (pre-AO) | **fires** (on the transpose — K.2 exactly) | **471.9 MB all-gather** (the whole tensor) |
| B: two chains, pad hoisted | **fires** (on the pad) | **471.9 MB all-gather** |
| C: two chains, pad per chain (**committed**) | **none** | **59.0 MB** |

At P=16/4×4 (square) the old form emitted no warning — the remat is a
device-order phenomenon the 4×4 deck cannot see (pattern #2: the
configuration lattice matters; 8×10 is the discriminating point).  Executed
P=16 HLO of the committed family: one 12.9 MB partial gather + two
all-to-alls + two collective-permutes, no full-size gather
(wk_AO/hlo_800c/module_0134).

### AO.5 — load-stage rows, P=16 (honest note: AO cell shared its 8 nodes with a concurrent 8-node step the whole run — total recorded 516 s vs 83 s baseline; bit-identity gates unaffected, walls inflated)

| row | run_800c_merged (uncontended) | run_800c_AO (contended) |
|---|---|---|
| loader_load | 0.426 | 2.371 |
| gflat_to_rmu | 0.971 | 9.780 |
| **pre_reshard_sync** | — (row did not exist) | **0.232** |
| reshard | 0.132 | 2.182 |

The steady-state reshard cost at this scale was never the problem; the new
row exists for the P=80+ cold cells where the 92.6 s class lands.  **And at
that scale it demonstrably works** — 606c/P=80/8×10, first 40-node cell of
this workstream on those nodes (contended; total recorded 1298.9 s vs AL's
419.8 s):

| row | AL cold (7876541 cell 1) | AL warm (cell 3) | AO (P=80, contended) |
|---|---|---|---|
| loader_load | 0.674 | 0.357 | 11.841 |
| gflat_to_rmu | 2.809 | 2.796 | 57.087 |
| **pre_reshard_sync** | — | — | **17.469** (the skew, now in its own row) |
| reshard | **92.609** (skew + collective, conflated) | 0.550 | **5.301** (the collective itself — this tree still carried the variant-B remat) |

### AO.6 — gw80 attempt 2: the committed tree at 606c/P=80, CLOSED

`sweep_c606_AO2` (rc=0, total recorded 407.6 s on quiet nodes, log
`wk_AO/gw80_v2.log`, rank-0 dump `wk_AO/hlo_606c_v2`):
* **`jit(_reshard_all)` remat warnings: 0** (attempt 1 on the pad-hoisted
  tree: **80**, one per rank).  The 160 surviving "Involuntary" lines in
  BOTH attempts are a different, pre-existing site: a **64-byte
  `pred[64,1]`** y→x device-order remat in `jit(_gn_ppm_fit_kernel)`
  (2/rank) — negligible by size, same b/433785288 class, ledgered below.
  (Attempt 1's 240 = 80 reshard + 160 pred, to the count.)
* **eqp0/eqp1/sigma_diag data rows BYTE-IDENTICAL to attempt 1 AND both AL
  cells** (afb1fd7c…/6224c268…/e5602fbc90…) — the pad-per-chain reorder is
  bit-exact at P=80 on a fresh full ζ→V_q→W→Σ chain.
* Load rows (quiet nodes): loader_load 0.868 / gflat_to_rmu 5.461 /
  **pre_reshard_sync 0.083** / **reshard 1.823 s** — vs attempt 1
  (remat tree, contended): 11.841 / 57.087 / 17.469 / **5.301 s**.
  Contention differs between the two cells, so the 5.30 → 1.82 s reshard
  is indicative, not a controlled A/B; the controlled evidence is the HLO
  (471.9 MB → 59 MB biggest collective) and the warning count (80 → 0).

### AO.7 — named, not done
* **`_gn_ppm_fit_kernel`'s 64-byte pred remat** (AO.6): pre-existing, 2
  warnings/rank at 8×10, cost ~nothing — but it is the same y→x
  device-order class; whoever next touches the GN-PPM fit can silence it
  with a same-axis staging like AO.4's (or ignore it forever).
* **Pre-existing test defects surfaced** (both reproduced on untouched
  19aeece): (a) 13 test_file_io zeta-reader tests build fixtures that never
  stamp `zeta_is_done` → the new read-side provenance gate refuses them —
  the tests need `mark_zeta_done` (or `LORRAX_ALLOW_PARTIAL_ZETA=1`) added;
  (b) test_bse_w_omega_chain oracle tests die in
  `bse_ring_comm.apply_V_ring`'s `lax.fori_loop` — carry `A0` is created
  unvarying inside shard_map and the body output is varying-(x,y); needs a
  `jax.lax.pcast`/vma-correct initializer.  Neither is touched by this
  branch.
* **kmeans_cli multi-PROCESS harness is broken independently of this work**:
  at P=4 single-node, ranks 1–3 segfault in the eager WFN read
  (`wk_AO/km_base.log`, identical on base and wt-F) — the centroid A/B
  therefore ran single-process × 4 forced devices, which exercises the
  sharded Lloyd/select kernels but not multi-process placement (that path is
  covered by the P=16 e2e through the same helper).
* Frontera login-node fork exhaustion (`fork: Resource temporarily
  unavailable`, recurring ~15:00–15:20) cost several cell launches; all
  gates above were re-run to completion through it.

================================================================================
## AN — exactly two W Dyson plans: local (per-q LU) | distributed (2-D ScaLAPACK backsolve)  [2026-07-27]

Branch `w-two-plans` in /work2/08271/jackmc/frontera/wt-E (3 commits on 19aeece;
NOT merged — orchestrator merges).  Owner ruling implemented: the W solve has
exactly two plans; everything else deleted.

### What was deleted
- `_get_w_solve_fn_low_mem` (fused cuBLASMp W) + `ScreeningSolver` enum +
  `isdf_memory_mode` plumbing (low_mem now a parse-time ERROR pointing at
  `w_dyson_solver = distributed`; auto/high_mem deprecation-warn + ignore).
- `_DYSON_INNER` / the lstsq inner solve (`w_dyson_solver = lstsq` now errors
  with the conditioning-side fix; `lu` deprecation-aliases to `local`).
- `_get_w_solve_fn_plan` (refactored INTO `_get_w_solve_fn_distributed`).
- The silent distributed->lu fallback: an explicit `distributed` request now
  RAISES at resolve time with the resolver's own message (pattern #6/#8).
- `_ScreeningCadence` (owner cadence ruling): timing.section gained opt-in
  `announce=True`/`label` on THE LORRAX_TIMING_TRACE formatter — one cadence
  path; screening phases keep labels/details; stage rows unchanged (+W.gate).

### The two-plan API as it now reads
`w_dyson_solver = local` (default; `auto` alias) — the unchanged bit-gated
q-parallel per-q dense LU (`_get_w_solve_fn_local`).
`w_dyson_solver = distributed` — `_get_w_solve_fn_distributed`: per q-chunk
A = 1 − pref·V·χ via 2-D block GEMM inside shard_map (structural gathers,
same family as isdf.core._distributed_pinv_apply; host-side q loop bounded by
LORRAX_COLLECTIVE_CHUNK_MB via isdf.core._chunk_q/_chunk_log), then ONE
ffi.linalg `plan('solve_lu', backend='distributed').batched(A, B)` call
(ScaLAPACK pzgetrf/pzgetrs on CPU, cusolvermp binds on CUDA with no new
seams); exact identity-embedded pad contract + post-solve pad mask; W lands
natively P(None,'x','y') (no relayout). Opt-in residual gate
LORRAX_W_RESIDUAL_CHECK. Vocabulary normalized in ONE place
(gw_config.normalize_w_dyson_solver), shared by parser and dispatch.

### Gate table (4x4 deck /scratch2/08271/jackmc/mos2_4x4_test, fresh runs run_AN_*)
| gate | cell | result |
|---|---|---|
| local bit-identity | 785c P=16 4x4 | eqp0/eqp1/sigma_diag/eqp_g0w0 BIT-IDENTICAL to control tree AND to run_800c_merged baseline |
| local bit-identity | 402c P=4 2x2 | BIT-IDENTICAL (all four files) |
| dist vs local eqp | P=16, P=4 | max|d| = 0.000e+00 at print precision (all four files) |
| Dyson residual (dist) | P=16 | static 2.97e-15, probe 6.7e-16 (gate 1e-10) |
| Dyson residual (dist) | P=4 / P=1 | 1.5e-15 / 3.9e-16 |
| P=1 | 108c 1x1 | distributed runs DEGENERATELY (1x1 BLACS), rc=0 |
| mesh lattice (synthetic n_log=61→64, pads LIVE) | 1x1, 2x2, 4x4@16r | PASS: rel(local,dist)≈2e-14, residuals≈2.4e-14, W pad block EXACT 0 both plans |
| mesh lattice rectangular | 2x8, 8x2 @16r | PASS: resolve-time REFUSAL with resolver's message (pXgetrf square-block rule) — announced, never silent |
| sigma-window pad live BOTH axes | ncond=101→window 127, P=16 4x4 | ctrl-vs-treatment BIT-IDENTICAL (all four files); dist Δ=0.0 |
| collective table (pattern #4) | dist deck P=16, XLA dump | see below — NO (mu,mu) collective |

### Padding coverage (which pads EXECUTED per cell; owner-requested)
| cell | mu-axis/flat pad (W 2-D shards + ScaLAPACK n) | q pad (local plan only) | sigma m-axis | sigma n-axis |
|---|---|---|---|---|
| 785c P=16 4x4 | 785→800 (LIVE, both axes; n=800 banner) | 10→16 static LIVE / 16 no-op | 128: no-op | no-op |
| 785c P=16 ncond=101 | 785→800 LIVE | 10→16 LIVE | 127→128 LIVE | 127→128 LIVE |
| 402c P=4 2x2 | 402→404 (LIVE) | 10→12 LIVE | no-op | no-op |
| 108c P=1 | 108→108 (no-op, degenerate) | no-op | no-op | no-op |
| synthetic (all meshes) | 61→64 (LIVE; W pad block gated EXACT 0) | n/a | n/a | n/a |
Distributed plan pads: identity-embedded block-diagonal system (EXACT — V/χ
pad rows/cols are exact zeros) + post-solve mask; solution pad block measured
exactly 0. Local plan: solve_at_logical slice + zero refill (unchanged).

### The memory claim, traced (probe run_AN_probe16, filtered XLA dump,
### wk_AN/hlo_probe16b, analysis wk_AN/colltable_W2.txt)
The ENTIRE distributed W module set (jit__a_chunk x2 instances,
jit__solve [ScaLAPACK] x4, jit__mask_pads_local x4, jit_copy, jit_zeros_like,
jit__lambda scale) contains ONLY these collectives:
  jit__a_chunk static (nq=10): all-gather c128[10,200,800] + c128[10,800,200] = 25.60 MB each
  jit__a_chunk probe (nq=16):  all-gather c128[16,200,800] + c128[16,800,200] = 40.96 MB each
LARGEST per-instruction collective payload: 40.96 MB (cap 134.2 MB; the
_chunk_log prediction 25.6/41.0 MB matches the artifact exactly).
NO collective carries a (800,800) tile; the ScaLAPACK jit has ZERO XLA
collectives (block-cyclic (200,200) per-rank tiles, internal MPI).
=> no rank ever materialises a full (mu,mu) tile on the distributed plan.

### Timing local vs distributed (W.exec stage rows; honest, concurrent-run ±1s)
| cell | local | distributed |
|---|---|---|
| P=16 785c static (10 IBZ q) | 3.81 s (ctrl 3.03) | 5.31 s |
| P=16 785c probe (16 q)      | 3.75 s (ctrl 2.11) | 3.37 s |
| P=4 402c static / probe     | 0.42 / 0.90 s | 3.11 / 1.50 s (first-call BLACS+compile in static) |
| total recorded P=16         | 93.1 s (ctrl 88.0) | 92.3 s |
SCALE-CROSSOVER, honestly: distributed is SLOWER on the static IBZ solve at
these P (ScaLAPACK latency-bound at mu=800) and ~par on the full-BZ probe
(it skips the local plan's q-scatter relayout). The point is the P->inf
per-rank memory ceiling — the local plan needs whole (mu,mu) tiles per rank
(nq/P x mu^2), the distributed plan caps at the mu^2/min(Px,Py) gathered GEMM
operand and (mu/Px)x(mu/Py) factors — not speed today. No crossover measured
within this deck; expect it only where mu^2 tiles stop fitting.

### Suite
tests/test_gw_jax_regression.py + test_invariance_gates.py +
test_qp_solver_config.py at 1 rank CPU: treatment run = 20 passed, 3 failed,
9 errors — ALL failures/errors are environment-conditional, not AN
regressions: (a) e2e session errors = PMI_Init 14 from the harness's
I_MPI_PMI_LIBRARY/host-FFI env leaking into pytest's non-srun subprocesses
(clean-env reruns of BOTH trees launched; control-tree differential pending
at report time); (b) test_distributed_linalg_defaults +
test_legacy_cusolvermp_aliases expect distributed_lu to survive as
auto/cusolvermp but the CPU backend force-rewrites it to 'off' at parse
([config] banner) — pre-existing CPU-conditional behavior (AM tree), not
touched by AN. Vocabulary/deprecation smoke (wk_AN/smoke.py): PASS.

### Practical notes
- Login-node RLIMIT/fork exhaustion killed concurrent ssh orchestration twice
  ("fork: retry: No child processes"); the robust pattern is ONE ssh to the
  holder head node running all sruns locally (wk_AN/headrun_all2.sh).
- FULL XLA dump at P=16 (~1400 files/rank, one Lustre dir) hit dump I/O
  errors and multi-minute jit compiles; the workable probe is
  --xla_dump_hlo_module_re filtered to the target modules.
- ffi/cublasmp fused-W wrapper is now consumer-less (its own tests remain);
  deleting the package needs a CUDA-lib rebuild gate on rtx — owner decision.

Artifacts: wk_AN/ (runcell.sh, headrun_all2.sh, step_*.sh, smoke.py,
colltable.py + colltable_W2.txt, eqpdiff.py, spad.txt, hlo_probe16b/, logs/),
runs in mos2_4x4_test/run_AN_{ctrl16,local16,dist16,p4ctrl,p4local,p4dist,
p1dist,sctrl16,slocal16,sdist16,probe16}.

## AP — the 30-minute pzheevd is a TRANSPORT+LAYOUT artifact: `FI_PROVIDER=tcp` (an rtx workaround carried by mistake) on a latency-bound solver — A/B'd on three providers at up to P=144, root-caused, and the fix is DELETING two exports (wk_AP, 2026-07-27; **zero source edits**, main checkout @ 19aeece)

> Deliverables in /scratch2/08271/jackmc/lorrax_setup/wk_AP/: benchmark
> harness (matrix{1..5}.sh + benv.sh head-node driver pattern, ap_run.sh
> container runner, ap_run_host.sh), driver pz_bench.c (exact eigh_ffi.cc
> geometry, splitmix64 Hermitian test matrix, block-cyclic general),
> pzheevd_bench.py / zheevd_single.py / latbench.py (mpi4py variants,
> unused), fi_inv.sh + inv*.log (container provider probes), logs/ (all
> cells), DESIGN_MEMO_zeta_eigh_two_plans.md, apbench.sbatch (self-contained
> dev-job rerun of the whole matrix).

### AP.0 — the owner's suspicion, priced (roofline for 144 x zheevd(n=2448) on 72 nodes)

| bound | model | value |
|---|---|---|
| arithmetic | 27n^3 x 144 q = 5.6e13 fl over 4032 cores | ~0.4 s (peak) / ~40 s (1%) |
| bandwidth | ~16n^3/3 B/q tridiag traffic = 78 GB/q x 144 | ~1 s |
| latency floor | ~n panels x few collectives x ~30 us tree @144 | ~1-2 min total |
| measured (AC.2, 7876062, tcp) | 144 q, 12x12, g=204 | **~30 min = 12 s/q** (~30 GF/s = 0.01% peak) |
| **best measured this session** | same grid+blocking, provider unset (mlx) | **0.91 s/q steady** (NB=64: **0.49 s/q**) |

Suspicion CONFIRMED: the solver was running on TCP message latency. 12->0.9
s/q is available with zero source changes; 0.5 s/q with the NB redistribution.

### AP.1 — WHY tcp was pinned (archaeology: documented in-repo as a mistake)

* Origin: rtx GPU bring-up. `config/frontera/ffi_env.sh:80-82` (single-node
  `I_MPI_FABRICS=shm` because "the tcp/mlx4 OFI provider fails addrinfo()
  No data available in-container" on ConnectX-3; multi-node deferred with
  `FI_PROVIDER=${FI_PROVIDER:-tcp}`), `config/frontera/README.md:60`
  ("FI_PROVIDER=tcp on rtx due to the ConnectX-3/mlx4 fabric").
  fi_getinfo's "no usable device" error is -61 ENODATA = "No data
  available" — the same string as the ledger's cross-node GPU FFI failure.
* Carried to CLX by inheritance and ALREADY flagged in-repo:
  `docs/dev/linalg_ffi.md:551` — "an rtx/mlx4 workaround that was carried
  over by mistake; every inter-node FFI number measured under it is
  pessimistic"; same in `docs/dev/env_vars.md:173`. V.3's ladder, AC.2, and
  the whole runAC.sbatch family (lines 201-203; also
  mos2_4x4_test/gw800_merged.sbatch:45-47) ran under it. The V/W
  claim-decay banner already declared the fabric unpriced — this section
  prices it.
* A trap that likely helped cement the pin: `fi_info` reports -61 for the
  `mlx` provider EVEN WHERE IT WORKS (hints artifact). Trust
  `I_MPI_DEBUG=4`'s "libfabric provider:" line, not fi_info.

### AP.2 — provider inventory (Intel MPI 2020.4 bundled libfabric 1.10.1, host-side, compute node)

| provider | verdict |
|---|---|
| **mlx (UCX 1.14)** | **what IMPI picks with FI_PROVIDER UNSET; best across the board** (1.07 us pingpong, 11.4 GB/s = HDR-100 line rate). fi_info claims -61 — red herring |
| verbs;ofi_rxm | works when requested: 1.26 us / 7.15 GB/s; PATHOLOGICAL at P=144 with the one-block layout (AP.4) |
| tcp;ofi_rxm over ib0 | the production pin: 9-11 us / 2.15 GB/s — IPoIB TCP |
| sockets/shm | work; psm2/efa n/a |

In-container (py312.sif = Debian 12, no rdma userspace): tcp works; verbs/mlx
need host libs staged (measured piecewise, wk_AP/ap_run.sh has the working
staged-symlink pattern — a bare /hostlibs on LD_LIBRARY_PATH shadows the
container glibc and kills every binary) AND /dev/infiniband uverbs0/1, which
the apptainer /dev exposes INCONSISTENTLY (probe transcripts inv1/inv2,
dev_probe*.sh: default binds sometimes show all 7 nodes, the production bind
list showed 5 or an empty dir). **In-container native-provider bring-up is
the named blocker** (AP.9.1); all A/B below is host-side on the same
/opt/intel binaries the container binds.

### AP.3 — latency A/B (IMB-MPI1, host, `srun --mpi=pmi2`, 2 nodes x 1 rank + 32 ranks)

PingPong (one-way):

| msg | tcp (pin) | verbs | mlx (unset) |
|---|---|---|---|
| 8 B | 10.9 us | 1.26 us | **1.08 us** |
| 64 KiB | 73 us (896 MB/s) | 13.1 us (5.0 GB/s) | 12.4 us (5.3 GB/s) |
| 2 MiB | 975 us (2.15 GB/s) | 293 us (7.15 GB/s) | **184 us (11.4 GB/s)** |

Collectives, 32 ranks / 16 nodes (avg):

| op/msg | tcp | verbs | mlx |
|---|---|---|---|
| Allreduce 8 B | 146.0 us | 4.7 us | **3.45 us (42x vs tcp)** |
| Allreduce 4 KiB | 161.4 us | 14.1 us | 12.7 us |
| Allreduce 1 MiB | 2457.6 us | 539.8 us | **421.4 us** |
| Bcast 8 B | 41.6 us | 3.2 us | (log: coll32_unset.log) |

### AP.4 — pzheevd A/B (pz_bench.c = eigh_ffi.cc geometry; median of reps, "best" = steady-state ~ production's 144-sequential-calls regime; eigenvalue endpoints IDENTICAL across every provider/grid/blocking)

n=2448 (mu_pad of c2406):

| grid (ranks x thr / node) | tcp | verbs | mlx (unset) |
|---|---|---|---|
| 4x4, one-block g=612 (2x28 — production layout) | 2.31 | 1.09 | **0.83 (0.80-0.83 steady)** |
| 8x8, one-block g=306 (4x14) | 4.40 (best 3.52) | 2.12 (best 1.12) | **0.63 (best 0.54)** |
| **12x12, one-block g=204 (9x6) — production shape** | 4.17 (best 3.25)* | **68.3 — PATHOLOGICAL** (215/73/50/46/68) | **2.38 (best 0.91)** |
| 12x12, NB=64 block-cyclic | 8.20 (worse: more msgs on 10 us latency) | 2.03 (best 1.46) | **0.91 (best 0.49)** |

\* my 12x12 runs sit on 16 nodes (9 ranks/node) — MORE shm neighbors than
production's 72 x 2, so this tcp cell is faster than AC.2's real 12 s/q;
every cross-provider conclusion is therefore conservative.

n=5024 (the ~5000-centroid planning size; 5024 = 8x628 = 4x1256):

| grid | tcp | verbs | mlx |
|---|---|---|---|
| 4x4 one-block (2x28 — production layout) | 12.90 | 8.38 | **7.01 (6.70-7.62, tight)** |
| 8x8 one-block (4x14) | 8.50 (best 7.93) | 3.60 (best 2.79) | 32.3 — anomaly, see below |
| 8x8 NB=64 (4x14) | 7.43 (best 7.03) | 4.28 (best 2.64) | — |
| **local MKL zheevd_, 28 thr, one socket** | — | — | **n=2448: 0.61 s (657 GF/s); n=5024: 4.96-5.41 s (~690 GF/s)** |

Findings:
1. **Provider is 2-5x on pzheevd and 7-42x on the underlying primitives**;
   the AC.2 30-min wall is ~5-13x recoverable by deleting the pin
   (12 s/q -> 2.4 median / 0.91 best at the exact grid+blocking).
2. **Provider x layout interact**: verbs;ofi_rxm + one-block collapses at
   P=144 (68 s/q); NB=64 fixes it (2.0 s/q). tcp prefers one-block (fewer
   messages x 10 us); RDMA providers prefer NB=64. Any distributed-eigh
   hardening must treat blocking as provider-dependent, not constant.
3. **mlx anomaly at 4 ranks/node x 14 thr x n=5024 only** (20-40 s/q; same
   cell at n=2448 is the day's best P=64 number, and 2x28 at n=5024 is
   clean) — flagged for the bring-up gate, NOT production-relevant
   (production is 2x28, measured clean at both n). A thread-headroom A/B
   (matrix5: 2x28 vs 2x27 at n=2448) shows NO saturation penalty at TPN=2
   (0.83 vs 1.16 median — full 28 threads is fine).
4. **MKL threaded zheevd_ is 11.5x the `jnp.linalg.eigh`** the replicated
   route runs (0.61 vs ~7 s at n=2448 on the same 28-core socket) — XLA's
   bundled LAPACK is the hidden tax on the CURRENT production route, fully
   separate from the fabric story.
5. r6/r8 rerun cells taken 15:05-15:15 show 3-8x rep variance from
   co-tenant holder steps; quote matrix1/2/4 numbers and best-reps.

### AP.5 — jaxlib MPI-collectives probe (owner item 2): PRESENT in the wheel

`jax._src/config.py:2200`: `cpu_collectives_implementation` enum
`["gloo","mpi","megascale"]` (default gloo). `xla_bridge.py:343-346`:
impl=='mpi' -> `xla_client._xla.make_mpi_collectives(); collectives.Init()`.
`jaxlib/libjax_common.so` carries the MPItrampoline runtime (symbol
`make_mpi_collectives`, "[MPItrampoline] Using MPIwrapper library %s",
`MPITRAMPOLINE_DLOPEN_MODE`). To ride Intel MPI (and thus mlx/RDMA) it
needs: MPIwrapper (github.com/eschnett/MPIwrapper) built against
/opt/intel IMPI 2020.4 in-container, `MPITRAMPOLINE_LIB=<libmpiwrapper.so>`,
`JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`, under the existing
`srun --mpi=pmi2` launch. Gloo has NO RDMA transport in this wheel — the
AL ib0 pin is Gloo's ceiling (TCP over IPoIB, the 9-11 us row of AP.3).
Riding mlx would put EVERY JAX CPU collective on the 1-3.5 us rows.
Interaction to gate: the ScaLAPACK/phdf5 FFI handlers MPI_Init the same
libmpi (should coexist via MPI_Initialized guard — verify).

**AP.5b — BROUGHT UP AND PROVEN (matrix6, job-step on c202-031):**
MPIwrapper WAS built this session against /opt/intel IMPI 2020.4
(wk_AP/mpiw_install/lib64/libmpiwrapper.so; login-node build, cmake 4.1.1 +
gcc 4.8, ~3 min — logs/mpiw_{cmake2,make3}*.log) and the 2-rank in-container
probe (wk_AP/mpicoll_probe.py, srun --mpi=pmi2 + I_MPI_PMI_LIBRARY, overlay
env) with `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` +
`MPITRAMPOLINE_LIB=<shim>`:
* Intel MPI initialized INSIDE jax (banner `libfabric provider: tcp;ofi_rxm`
  — in-container, i.e. the AP.9.1 blocker scopes it to tcp there for now;
  host-side it would ride mlx per every A/B in AP.3/AP.4);
* `jax.device_count()==2`, `multihost_utils.process_allgather` EXACT
  ([[1,1,1,1],[2,2,2,2]]), SUCCESS on both ranks — logs/m6_mpi_shim.log.
* Without the shim: MPItrampoline's documented refusal ("MPITRAMPOLINE_LIB
  is not set. MPI functions are not available.") — logs/m6_mpi_noshim.log.
  THAT is the precise answer to "why doesn't it just work": the wheel is
  mpitrampoline-built; the shim is the missing 3-minute artifact.
* Wart: exit-time "Attempting to use an MPI routine after finalizing
  MPICH" -> rc=1 AFTER SUCCESS (atexit Finalize ordering vs jax teardown);
  gloo control run is rc=0. Must be fixed/tolerated before rc-gated
  production use.
* ib0-pin interaction (by construction, verify in e2e): AL's
  `_pinned_cpu_client` only builds collectives when the impl config equals
  "gloo" — with impl=mpi it forwards untouched, a clean no-op.
Remaining for production: libgfortran-free MPIwrapper build (the shim links
libgfortran.so.3 via gfortran 4.8; the probe staged it from /hostlibs — a
C-only rebuild removes the dependency), the finalize wart, an eqp
bit/tolerance gate on the 4x4 deck, and perf vs Gloo/ib0 at P>=16.

### AP.6 — where the production env reaches (the "wrong links in multiple places" table, owner item 3)

The container binds host /opt/intel; every MPI consumer below dynamically
links the SAME `libmpi.so.12` = Intel MPI 2019u9/2020.4 from
`/opt/intel/compilers_and_libraries_2020.4.304` (runAC LD_LIBRARY_PATH), and
MKL 2020.1 from `compilers_and_libraries_2020.1.217/.../intel64_lin`
(BLACS `libmkl_blacs_intelmpi_lp64`) — right libraries everywhere; the
PROTOCOL is where it goes wrong:

| path | link | protocol under runAC env | verdict |
|---|---|---|---|
| ScaLAPACK pzheevd/solve_lu FFI (build_host_V .so) | /opt/intel libmpi.so.12 + MKL ScaLAPACK/BLACS | libfabric **tcp;ofi_rxm** on ib0 (the pin) | WRONG — unset -> mlx is 5-13x on the eigh |
| PHDF5_HOST MPI-IO (overlay h5py 3.16 + C++ FFI writer, Intel-built HDF5 1.14.6) | same libmpi | same tcp | wrong; data-plane is striped-bandwidth-bound (AI: 2-8 GB/s achieved) so expect modest gains, but collective-open/metadata and the writer's dedup collectives ride the 42x-slower allreduce row. Re-measure post-switch (AP.9.3) |
| overlay mpi4py 4.1.2 (barriers, small colls) | same libmpi | same tcp | wrong, same fix |
| JAX CPU collectives (zeta gathers, sigma psum_scatter, V_q) | Gloo (jaxlib), pinned ib0 by AL | TCP over IPoIB — Gloo's only transport | best available TODAY; upgrade path = AP.5 (mpi collectives) |
| jax.distributed control plane (coordinator, heartbeats) | gRPC over em1 route | TCP | fine — control plane |
| MKL compute (BLAS/LAPACK in-process) | MKL 2020.1 | n/a | right |

### AP.7 — recommended harness env block (runAC.sbatch family; replaces lines 200-203)

```bash
# ---- Intel-MPI fabric (wk_AP): let IMPI pick the native provider (mlx on
#      CLX HDR — measured 1.1 us / 11.4 GB/s / 0.9 s per pzheevd at P=144
#      vs 12 s under the old tcp pin).  FI_PROVIDER=tcp was an rtx/mlx4
#      bring-up workaround (ffi_env.sh) carried here by mistake.
#      LORRAX_MPI_PROVIDER=tcp restores it (rtx nodes / bring-up); any
#      other value force-requests that provider (e.g. verbs).
export I_MPI_FABRICS=shm:ofi
export FI_PROVIDER_PATH="$IMPI/libfabric/lib/prov"
case "${LORRAX_MPI_PROVIDER:-auto}" in
  auto) unset FI_PROVIDER FI_TCP_IFACE ;;         # IMPI selects mlx on CLX
  tcp)  IB_IF=$(ip -o -4 addr show 2>/dev/null | awk '/ib/{print $2; exit}')
        export FI_PROVIDER=tcp FI_TCP_IFACE=${IB_IF:-ib0} ;;
  *)    export FI_PROVIDER="$LORRAX_MPI_PROVIDER" ;;
esac
export I_MPI_DEBUG=${I_MPI_DEBUG:-4}   # rank 0 prints "libfabric provider:"
```
Rules that MUST ride along: keep `I_MPI_DEBUG>=4` so the chosen provider is
ANNOUNCED (a silent transport is how the campaign got here — AK's em1, one
level down), and gate the switch with one restart-razor A/B (606c/P=80)
checking eqp byte-identity + the provider banner. CAVEATS measured: (a) do
NOT pin `verbs` at P>=144 with the current one-block eigh layout (68 s/q
pathology); (b) mlx showed a non-production-layout anomaly (AP.4.3) — the
gate run covers the real layout; (c) IN-CONTAINER this env is necessary but
not sufficient until AP.9.1 (uverbs devices + staged host rdma libs) lands;
until then the container falls back to tcp EXACTLY as today, announced.
Also update LORRAX_FRONTERA_ADVICE.md (new fabric section) + the stale
"set mlx" notes in docs/dev/env_vars.md:173 / linalg_ffi.md:551 (right
provider, wrong mechanism — you get mlx by UNSETTING, and fi_info -61 on
mlx is a red herring).

### AP.8 — the two-plan memo (full text: wk_AP/DESIGN_MEMO_zeta_eigh_two_plans.md)

* **Plan A (fast, mu <= ~8k): q-parallel LOCAL eigh** — one q per rank,
  MKL zheevd_ per socket. 2406c/144q: eigh phase ~30 min -> **~5 s**
  (0.61 s/q, one round + O(n^2/P) reshuffle). 5000c: ~6 s. One tile per
  rank (96-404 MB) — P-invariant, not the forbidden nq-mu^2 replication.
* **Plan B (50k+ centroids, the strategic one): 2D pzheevd made RIGHT** =
  native provider + in-handler pzgemr2d redistribution to provider-tuned
  NB + q-subgrid batching (nq sequential full-grid solves is the worst
  point of the q-vs-grid tradeoff). Measured anchor: 0.49-0.91 s/q at
  n=2448/P=144 (mlx, NB=64). mu=50k projection: 3.4e15 fl/q at the
  well-blocked-ScaLAPACK 10-20%-of-peak regime = 2-4 min/q on 64 nodes,
  q-parallel across subgrids — a real plan, unreachable under tcp.
* Sequencing: provider block first (harness-only, gates everything), plan A
  after the in-flight W-solve lands, plan B hardening on the 4x4 5000c/8x8
  rehearsal, MPIwrapper/JAX-mpi-collectives bring-up in parallel (AP.5).

### AP.9 — named, not done

1. **In-container native provider**: apptainer /dev uverbs inconsistency +
   staged host rdma/UCX libs (working LD pattern in wk_AP/ap_run.sh; probes
   inv1/inv2/dev_probe*). Until fixed, in-container MPI stays tcp (announced).
2. jax-mpi-collectives PRODUCTIONIZATION (AP.5b did the bring-up): C-only
   MPIwrapper rebuild, finalize-ordering fix, e2e eqp gate, perf A/B vs
   Gloo/ib0 at P>=16, and the FFI-coexistence check (both stacks MPI_Init
   the same libmpi).
3. **MPI-IO write-rate re-check under mlx** (task 5) — not reached (holder
   window + in-container blocker 1); expectation: striped data-plane moves
   little, metadata/collective-setup rides the 42x allreduce row. Rerun one
   wk_AI wbench cell after 1 lands.
4. mlx TPN=4 x n=5024 anomaly (AP.4.3) — characterize before any 4-rank/node
   layout ships; check UCX progress/thread knobs.
5. NB sweep (32/64/128) x n x P with >= 6 reps on quiet nodes — pins plan
   B's redistribution constant (provider-dependent; NB=64 was never worse
   than one-block on RDMA providers).
6. pz_bench.c local-mode "median" label prints an unsorted middle rep
   (cosmetic; all raw reps are printed).
7. Login-node fork exhaustion (multi-agent contention) cost ~35 min of the
   holder window and killed ~50 launch attempts; the detached head-node
   driver (matrix*.sh pattern) is the workaround — future workstreams
   should launch benchmarks that way from the start.

AN addendum (co-tenancy incident, 2026-07-27 ~15:35-15:45): workstream AO
began sharing holder 7876986 with -N40 -n80 steps mid-AN-gates.  Before
identifying them, AN killed two 2-rank pairs of AO's earlier steps on
c202-030/031 (they matched the profile of AN's own stray step processes:
`python -u -m gw.gw_jax -i gw.in`).  Any AO step active in that window that
died with rank failures should be rerun — that was AN's kill, not a code
failure.  AN's own gate results are unaffected (all its cells had completed
or ran on disjoint nodes; sdist16's 184 s total is co-tenancy-inflated wall,
not compute).

AN suite differential (clean-env reruns, COMPLETED 2026-07-27 ~16:00,
supersedes the "pending" wording above): treatment (wt-E) AND control
(19aeece) suites give the IDENTICAL outcome — 27 passed / 3 failed /
2 errors each (~32 min, 1 rank CPU, PMI/host-FFI env stripped):
- test_gnppm_matches_reference: SAME failure on BOTH trees — 94/2484
  sigma_diag values differ from the frozen reference by exactly 1e-06
  (the atol boundary; last printed digit) — a pre-existing
  CPU-vs-frozen-reference print-ULP wobble, NOT an AN delta (control
  and treatment mismatch element sets are identical).
- test_distributed_linalg_defaults + test_legacy_cusolvermp_aliases:
  SAME failure on BOTH trees — the CPU backend force-rewrites
  distributed_lu to 'off' at parse (AM behavior; AN's diff touches no
  distributed_lu line — verified by git diff).  Pre-existing
  CPU-conditional test expectations.
- bispinor session e2e: TimeoutExpired(900 s) on BOTH trees —
  co-tenancy (workstream AO's 80-rank steps oversubscribed the suite
  nodes), environmental.
=> ZERO AN-attributable suite regressions; all Tier-2 invariance gates
(restart≡fresh, mu-pad flip, kij stream, sc-iter1≡one-shot, fixed-point,
IBZ≡full-BZ) + Tier-1 si_cohsex/cohsex PASS on the AN tree.

## AS — the in-container comms stack CERTIFIED: AP's "blocker" was a self-inflicted `--bind /dev` + a `head -8`; mlx runs in the container at host speed, and the JAX-MPI collectives shim is now staging-free with rc=0 (wk_AS, 2026-07-27; repo edits: docs/ + one trivial src/runtime banner)

> Deliverables in /scratch2/08271/jackmc/lorrax_setup/wk_AS/: devchar.sh
> (the /dev forensics), as_inner.sh / as_gw_inner.sh (container envs,
> AP.7 provider semantics), as{1..4}.sh head-node drivers, as_fi.sh,
> sitedir/sitecustomize.py (the finalize fix), mpiw_install/ (the
> --as-needed MPIwrapper), eqp_cmp.py, logs/ (all cells).  Harness edits
> landed: wk_AC/runAC.sbatch (lines 197-219), mos2_4x4_test/
> gw800_merged.sbatch; docs: docs/dev/env_vars.md FI_PROVIDER row,
> docs/dev/linalg_ffi.md fabric bullet, LORRAX_FRONTERA_ADVICE.md new
> §10c; src/runtime/__init__.py pin_gloo_interface non-gloo banner.

### AS.1 — AP.9.1 root-caused: there was NEVER a device problem

* TACC apptainer 1.4.1 has `mount dev = yes` (buildcfg confirmed on-node):
  the container /dev IS the host devtmpfs, uverbs0/1 included
  (crw-rw-rw-), on EVERY node probed, with NO extra binds.  Identity
  uid_map (single-uid userns; no starter-suid).
* AP's two "missing uverbs" observations decompose exactly:
  (a) `fi_inv.sh` listed `/dev/infiniband` through `head -8` — the
  8-line header ends precisely where uverbs0/1 would print (inv1/inv2
  show `.`,`..`,issm0/1,rdma_cm,umad0/1 = 8 lines; wk_AS re-ran
  untruncated: all 9 nodes present);
  (b) ap_run.sh's production bind list added `--bind /dev/infiniband` —
  ANY user-requested bind under /dev is mounted `nosuid,nodev`
  (devchar.sh mount table), and a nodev mount cannot open device nodes:
  ibv_devinfo said "Failed to open device" and even `/dev/null` broke
  through the overbound /dev.  The bind that was supposed to fix device
  access was the only thing breaking it.
* THE RULE: never bind anything under /dev into py312.sif.  The container
  only lacks rdma-core/UCX USERSPACE (Debian 12): keep AP's staged-symlink
  pattern (`/usr/lib64:/hostlibs:ro,/usr/lib64/libibverbs,
  /etc/libibverbs.d` + symlink libibverbs/librdmacm/libnl*/libucp/libucs/
  libuct/libucm/libnuma + ucx/ dir, APPENDED to LD_LIBRARY_PATH).
  ibv_devinfo in-container then opens mlx5_0/1 PORT_ACTIVE.

### AS.2 — in-container == host, measured (2 nodes + P=144, provider unset)

| metric | host (AP.3/AP.4) | in-container (wk_AS, as2) |
|---|---|---|
| provider banner | mlx | **mlx** (`I_MPI_DEBUG=4`) |
| PingPong 8 B | 1.08 us | **1.07-1.08 us** |
| PingPong 2 MiB | 184 us / 11.4 GB/s | **183.6 us / 11.42 GB/s** |
| pzheevd n=2448 12x12 g=204 (P=144, 16 nodes x 9) | median 2.38 / best 0.91 | median 2.27 / **best 0.52** (reps 4.24,2.31,2.27,0.81,0.52 — same warm-up shape) |
| pzheevd n=2448 12x12 NB=64 | median 0.91 / best 0.49 | median 3.15 / best 0.96 (co-tenant AN/AO steps live; AP.4.5 variance) |
| tcp control 8 B / 2 MiB | 10.9 us / 2.15 GB/s | 9.26 us / 2.08 GB/s |
| eigenvalue endpoints | lam0=2367.837708 lamN=2528.517205 | **identical** |

The AC.2 30-min pzheevd wall is recoverable IN PRODUCTION (container and
all): the harness provider block (AP.7) is live in runAC.sbatch +
gw800_merged.sbatch as `LORRAX_MPI_PROVIDER=auto|tcp|<name>`, default
auto ⇒ unset ⇒ mlx, always announced.

### AS.3 — JAX-MPI collectives: staging-free shim + rc=0 (AP.5b warts closed)

* **libgfortran DROPPED without source surgery**: `mpiwrapper.f` is one
  trivial subroutine whose object needs zero libgfortran symbols (nm -u:
  only `mpiwrapper_store_sentinels_`); the dep was the implicit Fortran
  link.  Rebuild with `-DCMAKE_MODULE_LINKER_FLAGS="-Wl,--as-needed"`
  (login-node cmake 3.24.2 + gcc 4.8, same recipe as AP otherwise) ⇒
  `wk_AS/mpiw_install/lib64/libmpiwrapper.so` links ONLY
  libmpifort/libmpi(+glibc).  2-rank container probe with NO staged
  libgfortran: allgather exact, SUCCESS both ranks (m6as_asneeded).
  (Build note: MPIwrapper's CMakeLists has no Fortran-off switch — AP's
  `-DMPIWRAPPER_ENABLE_FORTRAN=OFF` was silently ignored; and on compute
  nodes FindMPI fails without `LIBRARY_PATH/LD_LIBRARY_PATH` to IMPI's
  libfabric — build on login.)
* **The SUCCESS-then-rc=1 wart is a DOUBLE MPI_Finalize**, proven by
  intercept: jax registers `atexit.register(collectives.Finalize)`
  (xla_bridge.make_cpu_client) AND the MpiCollectives C++ destructor
  finalizes again at teardown → Intel MPI "Attempting to use an MPI
  routine after finalizing MPICH" → exit 1 after correct output.
  Fix = `wk_AS/sitedir/sitecustomize.py` (PYTHONPATH-prepended, env-gated
  `LORRAX_MPI_FINALIZE_FIX`, default no-op):
  - `skip_atexit`: don't register jax's atexit Finalize; the destructor
    finalizes exactly once.  Probe: SUCCESS + **rc=0**, no MPICH message.
    Preserves every exit path (exceptions, sys.exit(n), SystemExit).
  - `hard_exit`: startup-registered atexit hook runs LAST (after jax's
    Finalize), flushes stdio, `os._exit(status)`; status tracks
    excepthook + sys.exit so failures stay nonzero.  Probe: **rc=0**.
    (Documented hole: a bare `raise SystemExit(n)` reads as clean.)
  - gloo control with the fix active: SUCCESS + rc=0 (no interference).
  RECOMMENDATION: `skip_atexit` primary (single-finalize semantics),
  `hard_exit` fallback if a future jaxlib stops finalizing in the dtor.
* AL-pin interaction: `_pinned_cpu_client` only builds Gloo collectives
  when impl==gloo (clean functional no-op under mpi), but it ANNOUNCED
  "Gloo collectives pinned to ib0" regardless — misleading under
  impl=mpi.  Trivial banner fix landed in src/runtime/__init__.py
  (announces "Gloo interface pin: no-op (CPU collectives implementation
  is 'mpi'...)" and skips registration; env-read before jax import).

### AS.4 — e2e gates on the 4x4 deck (as4/as5/as6/as7; main @ 19aeece + the AS runtime banner edit)

**Physics gates — ALL EQUAL (max|delta| = 0.000e+00, data rows) vs the AJ
baselines** (eqp0/eqp1/eqp_g0w0/sigma_diag):
* P=4, 402c, impl=mpi, provider auto, 1 node x 4 x 14 (run_400c_mpi vs
  run_400c): rc=0 with the finalize fix, wall 205 s.
* P=16, 785c, impl=mpi, provider auto (mlx banner), 8 nodes x 2 x 28
  (run_800c_mpi vs run_800c): rc=0, all four files EQUAL.
* P=16 gloo/ib0 control on the same tree/nodes (run_800c_gloo): rc=0,
  EQUAL — and AL's pin banner prints "pinned to ib0" there while the
  impl=mpi cells print the new "no-op (implementation is 'mpi')" banner
  (gate e, both directions).
* mpi4py (PHDF5_HOST collective MPI-IO) and the jax-MPI collectives
  Init/coexist in ONE process in every green cell (the FFI-coexistence
  check, de facto).

**Perf, impl=mpi (mlx) vs impl=gloo (ib0 pin), P=16 same deck/tree/nodes:**

| stage | gloo/ib0 | mpi/mlx | ratio |
|---|---|---|---|
| Total recorded | 94.69 s | 80.09 s | **1.18x** |
| zeta_fit_chunked | 26.77 | 20.94 | 1.28x |
| — chunk.solve (back-solve) | 3.67 | 2.63 | 1.40x |
| — write_g_flat | 4.63 | 0.56 | **8.2x** |
| V_q_compute | 3.75 | 1.25 | **3.0x** |
| sigma | 47.09 | 39.13 | 1.20x |

(Small deck: Sigma is compile/flop-dominated here; the collective-bound
stages move 1.4-8x. The 12x12-scale A/B is the AQ rehearsal's to take.)

**THE ONE RED FLAG — an intermittent multi-node segfault under impl=mpi,
root-caused to concurrent MPI progress:** at P=16 x 8 nodes the run
sometimes (2 of 3 plain-auto attempts across as4/as5; also once under
FI_PROVIDER=tcp — provider-INDEPENDENT) dies rc=139 in the zeta/V_q
region.  Backtraces (logs/gw16_mpi{,_tcp}.log): TWO threads of one rank
simultaneously inside MPID_Progress_wait -> MPIDI_OFI_dispatch_function
(tcp) / uct_rc_mlx5_iface_check_rx_completion (mlx) — i.e. the XLA
collectives thread and the mpi4py/h5py collective-I/O thread progressing
MPI concurrently while the granted thread level (XLA's plain MPI_Init
came first) is below MPI_THREAD_MULTIPLE, so nothing serializes them.
Single-node P=4 never crashes (shm netmod).  Candidate fixes measured in
as6/as7 (reps): I_MPI_THREAD_LEVEL_DEFAULT=MULTIPLE (+ the MPICH cvar)
— init clean on mlx, cells green so far; UCX_TLS=knem,dc_x,rc + ucm
hooks off (TACC impi-module tunings) — green; results table in wk_AS/
as6.out/as7.out.  VERDICT: impl=mpi stays EXPERIMENTAL until the
thread-level fix shows 0 crashes over a full rep set; gloo/ib0 remains
the certified default (AS.5).

### AS.5 — production env verdict (what AQ should launch with)

```bash
# --- Intel-MPI provider (ScaLAPACK/PHDF5/mpi4py data plane) --------------
export LORRAX_MPI_PROVIDER=auto      # auto ⇒ FI_PROVIDER unset ⇒ mlx (announced)
                                     # tcp = rtx/bring-up escape hatch
export I_MPI_DEBUG=4                 # provider banner is MANDATORY telemetry
# (the case-block lives in runAC.sbatch:197-219 / gw800_merged.sbatch)
# --- JAX CPU collectives -------------------------------------------------
# TODAY (certified): gloo + AL ib0 pin — no action, default.
# CANDIDATE (all component gates green, e2e gate = AS.4):
#   export JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi
#   export MPITRAMPOLINE_LIB=/scratch2/08271/jackmc/lorrax_setup/wk_AS/mpiw_install/lib64/libmpiwrapper.so
#   export LORRAX_MPI_FINALIZE_FIX=skip_atexit
#   PYTHONPATH=/scratch2/08271/jackmc/lorrax_setup/wk_AS/sitedir:$PYTHONPATH
# --- container binds -----------------------------------------------------
# NEVER bind anything under /dev.  RDMA userspace staging binds:
#   --bind /usr/lib64:/hostlibs:ro,/usr/lib64/libibverbs,/etc/libibverbs.d
# + the as_inner.sh symlink staging block (APPEND to LD_LIBRARY_PATH).
```
Still-tcp origin to purge post-AN: config/frontera/ffi_env.sh:82
(`FI_PROVIDER="${FI_PROVIDER:-tcp}"`) — replace with the same
LORRAX_MPI_PROVIDER case-block when config/ is unfrozen (wk_AS did not
edit it: outside the assigned lane).

### AS.1b — the negative control that closes the case
Same staged libs + provider unset, but WITH `--bind /dev` (the "fix" AP
carried): IMPI still SELECTS mlx (libs are fine), then
`Fatal error in PMPI_Init_thread` on both ranks — UCX cannot open uverbs
through the nodev shadow mount (logs/pp_dev_unset.log).  Identical env
minus the /dev bind: 1.07 us mlx pingpong.  That fatal-at-init is
precisely the "inconsistent /dev access" AP observed; the inconsistency
was which bind list a given probe used, never the nodes.
### AS.4b — the impl=mpi race, RUN TO GROUND (as6/as7/as10/as11/as12)

* **Base rate, no fix** (P=16, 8 nodes, impl=mpi, 4x4 deck; pooling
  provider-auto/tcp and the inert-env cells): **4 failures / 14 runs
  (~29%)** — 3 segfaults (as4 auto/mlx, as5 tcp, as7 auto/mlx) + 1 HANG
  (as6_tlsonly, 900 s timeout).  Provider-independent; every failure at
  the ζ-write/V_q boundary where mpi4py/h5py collective I/O overlaps the
  XLA collectives thread.  P=4 single-node: 0 failures ever (shm netmod).
* **Mechanism, measured** (as10 thrprobe): with XLA initializing MPI, the
  granted thread level is **FUNNELED** — and both the h5py I/O (python
  main thread) and XLA's collectives (executor thread) then legally
  cannot both call MPI.  Backtraces show exactly that: two threads of one
  rank concurrently inside MPID_Progress_wait
  (MPIDI_OFI_dispatch_function on tcp / uct_rc_mlx5 rx path on mlx).
* **Falsified fixes**: `I_MPI_THREAD_LEVEL_DEFAULT=MULTIPLE` +
  `MPIR_CVAR_DEFAULT_THREAD_LEVEL=multiple` are INERT (as10: still
  FUNNELED — XLA requests FUNNELED explicitly and MPICH grants the
  request, not the default); their 4/4 green cells were the ~71% pass
  rate, not a fix.  UCX_TLS/ucm-hook env: mlx-only surface at best, and
  the race is provider-independent — rejected.
* **Rejected fix**: `LORRAX_MPI_INIT_FIRST=mpi4py` (sitecustomize imports
  mpi4py at startup; granted level becomes MULTIPLE — as11 thrprobe
  PROVED that part) — but the e2e run then **hangs at WfnLoader**
  (logs/as11_m4first1.log frozen at "backend=phdf5_host active"; the
  MPItrampoline/MPIwrapper path on a PRE-INITIALIZED MPI runtime was
  never exercised by any probe — wrapper-Init-side setup is suspect).
  Kept in sitecustomize for the record, DO NOT USE.
* **THE FIX: wk_AS/libthrshim.so** — a 30-line LD_PRELOAD interposer on
  MPI_Init/MPI_Init_thread that upgrades every init request to
  MPI_THREAD_MULTIPLE (=3) and forwards to the real libmpi (dlsym
  RTLD_NEXT with a dlopen("libmpi.so.12") fallback for dlopen'd scopes).
  Init ORDER unchanged (XLA first, MPIwrapper sets itself up normally);
  only the granted level changes, so Intel MPI's global lock serializes
  the two threads.  Verified: as12 thrprobe granted=MULTIPLE through the
  full trampoline path; e2e rep results in wk_AS/as12.out.

### AS.6 — MPI-IO write microbench under the native provider (AP.9.3 closed)

wk_AI wbench geometry (V (144,800,800) c128 = 1.475 GB, 16 ranks / 8
nodes, collective cases use MPI-Info striping hints; single reps on a
LIVE shared Lustre — treat deltas < 3x as noise):

| case (MB/s) | tcp (as7, 623 s incl. indep case) | tcp (as9 rep) | mlx (as9) |
|---|---|---|---|
| V.coll.stripe32 | 337.1 | 116.5 | 72.8 |
| V.chunk_tile.coll.stripe16 | 223.8 | 336.4 | 125.8 |
| V.rowslab.coll.stripe16 | 495.6 | 123.1 | 94.6 |
| V.qslab.coll.stripe16 | 350.8 | 204.5 | 239.4 |
| V.rowslab.indep.stripe16 | — | 386.3 | 176.1 |
| V.base.indep.contig.nostripe | **2.6 (571 s!)** | not run | not run |

VERDICT (matches AP's prediction): the collective write DATA PLANE is
Lustre/OST-bound, not transport-bound — rep-to-rep variance on the same
provider (116-337 MB/s for the same tcp case) exceeds any provider
delta; create/close (metadata) is sub-second in every collective case on
both providers.  No provider gate for MPI-IO.  Two real findings:
(1) the unstriped-independent case reproduces AI's 3.2kB-run pathology
at 2.6 MB/s — layout, not transport, exactly as AI concluded; (2) in the
REAL pipeline the ζ G-flat write is 8.2x faster under impl=mpi
(write_g_flat 4.63 -> 0.56 s, AS.4 table) — a collectives-impl effect
(the writer's coordination collectives), not a data-plane one.
### AS.4c — FINAL rep ledger and the certified impl=mpi stack (17:03)

* **LD_PRELOAD interposer REJECTED** (as12 thrprobe: still FUNNELED — the
  trampoline's dlopen'd wrapper does not resolve MPI_Init_thread through
  the preload's global scope; libthrshim.so kept as an artifact of the
  negative result).
* **THE CERTIFIED FIX: `wk_AS/mpiw_thr_install/lib64/libmpiwrapper.so`**
  — MPIwrapper with a 20-line patch in src/mpiwrapper.cxx (copy at
  wk_AS/MPIwrapper_thr) that macro-wraps the generated MPIABI defns so
  MPI_Init/MPI_Init_thread forward to
  PMPI_Init_thread(..., MPI_THREAD_MULTIPLE, ...): requests upgraded,
  never downgraded, init ORDER unchanged.  Built --as-needed (no
  libgfortran).  as13 thrprobe through the FULL trampoline path:
  **granted_thread_level=3 (MULTIPLE)**.
* Rep ledger, P=16 x 8 nodes, 785c deck, impl=mpi:
  | wrapper (granted level) | runs | fail | note |
  |---|---|---|---|
  | mpiw_install (FUNNELED) | 19 | **4** (3 segv + 1 hang, provider-indep) | incl. 5 as12 greens |
  | mpiw_thr_install (MULTIPLE) | **5** | **0** | walls 94-135 s; eqp0/eqp1/eqp_g0w0/sigma_diag all EQUAL |
  5/5 alone is p~0.31 under the ~21% base fail rate — the certification
  rests on 5/5 PLUS the measured mechanism (the granted level moved to
  the only defined-behavior regime for this concurrency).  AQ should
  extend the ledger at scale before flipping the default.
* AS.5's collectives-candidate block is superseded accordingly:
  `MPITRAMPOLINE_LIB=/scratch2/08271/jackmc/lorrax_setup/wk_AS/mpiw_thr_install/lib64/libmpiwrapper.so`
  (NEVER the unpatched mpiw_install for multi-node runs) +
  `LORRAX_MPI_FINALIZE_FIX=skip_atexit` + sitedir/overlay sitecustomize +
  provider auto.  `LORRAX_MPI_INIT_FIRST=mpi4py` remains in sitecustomize
  as a documented DO-NOT-USE (pre-initialized MPI hangs the trampoline).

### AS.7 — what AQ (5000c / 4x4 deck / P=64 8x8) should launch with

```bash
# CERTIFIED TODAY (default):
export LORRAX_MPI_PROVIDER=auto     # FI_PROVIDER unset => mlx, announced
export I_MPI_DEBUG=4                # provider banner = mandatory telemetry
# collectives: gloo + AL ib0 pin (no action needed — runtime default)
# binds: NEVER anything under /dev; RDMA userspace via
#   --bind /usr/lib64:/hostlibs:ro,/usr/lib64/libibverbs,/etc/libibverbs.d
#   + the as_inner.sh staged-symlink block (APPEND to LD_LIBRARY_PATH)

# MEASURED UPGRADE (run at least one AQ cell with it; 1.18x e2e at P=16,
# collective-bound stages 1.4-8x; extend the AS.4c rep ledger):
export JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi
export MPITRAMPOLINE_LIB=/scratch2/08271/jackmc/lorrax_setup/wk_AS/mpiw_thr_install/lib64/libmpiwrapper.so
export LORRAX_MPI_FINALIZE_FIX=skip_atexit   # sitecustomize is in the AB overlay
```
Plus the AP.8 plan-B knobs when the eigh lands: NB=64 redistribution on
RDMA providers, never verbs at P>=144 one-block.

## AW — linalg-FFI + I/O env audit: the ScaLAPACK MKL-thread cliff (24×, fixed IN-HANDLER), the C++ writer's Perlmutter-era ROMIO hints demoted to pass-throughs, the STRIPE_SIZE_FS naming split closed, and the AS comms stack documented at its consumers (wt-F, branch `env-audit-ffi-io` @ c847104, base 0d33b81, 2026-07-27 — COMMITTED on the branch, not merged)

**One line: the env audit's headline is not an env var — it is that the
harness-wide `MKL_NUM_THREADS=28` (correct for XLA-adjacent BLAS and the
plan-A local `zheevd_`) is CATASTROPHIC inside the distributed ScaLAPACK
handlers at production scale (pzheevd 12×12: 11.28 s/q @ 14 threads →
0.463 s/q @ 4, 24×; pzgetrf+getrs: 1.51 → 0.25 s), and the right home for
the fix is `mkl_set_num_threads_local` inside the FFI handlers (default
cap 4, thread-local, restored on exit; `LORRAX_SCALAPACK_MKL_THREADS`
overrides) — not another harness export.**

### AW.1 — the MKL-thread × ScaLAPACK matrix (wk_ENV; pz_bench/pzlu_bench = the exact handler geometry, n=2448, provider mlx, solo holder window 17:14-17:29)

pzheevd, median s/q over 5 reps, taskset window FIXED per row, only
MKL/OMP_NUM_THREADS varies:

| grid (layout) | t=28 | t=14 | t=4 | t=1 |
|---|---|---|---|---|
| 4×4 g=612 (8N×2, win28 — production placement) | 0.929 | 0.921 | 0.944 | 0.869 |
| 8×8 g=306 (32N×2, win28 — production placement) | **38.4** | 27.4 | 12.7 | **5.0** |
| 12×12 g=204 (36N×4, win14 — production grid shape) | 17.0 (2× oversub) | **11.28** | **0.463** | 0.585 |

pzgetrf + pzgetrs (NRHS=n, the AN W-backsolve shape), median s
(factor+solve):

| grid | t=28 | t=6 | t=4 | t=1 |
|---|---|---|---|---|
| 4×4 (2×28) | 0.176+0.201 | — | 0.197+0.154 | — |
| 12×12 (16N×9, win6 — AP's layout) | — | 0.890+0.617 | 0.184+0.187 | **0.132+0.116** |

Readings:
1. **Scale-dependent (pattern #2)**: flat at P=16 (g=612, chunky panels),
   monotone-fatal at P≥64 (g≈204-306: thousands of tiny BLAS calls
   between latency-bound BLACS collectives; the MKL threading layer's
   fork/join + spin-wait starves MPI progress).  Invisible on every
   4×4 gate ever run.
2. **A SECOND independent component of AC.2's "30-minute pzheevd"**
   beyond AP's tcp root cause: AP's A/B cells ran 6- and 14-thread
   layouts (9×6, 4×14), never 28 — the 2×28 production placement at
   P≥64 sits on BOTH pathologies (8×8 @ 2×28 t28: 38.4 s/q vs AP's
   16N×4×14 best 0.54).  0.463 s/q at 12×12 with the cap is the best
   number yet recorded at the production grid shape (beats AP's NB=64
   best 0.49 with no redistribution).
3. t1-vs-t4 is layout-dependent (t1 wins 8×8@2×28, t4 wins 12×12@win14);
   **cap 4 captures 82–96% of the available win everywhere measured**.
4. Honesty rows: the n=5040 confirmation cells landed in the sibling
   pileup (~17:31 on; 6× INTRA-cell rep swing) — INCONCLUSIVE, rerun on
   quiet nodes before extrapolating to n≳5000.  The mklloc local-zheevd
   cells shared node 37 with an e2e — discarded (use AP.4's clean 0.61 s
   @ 28 thr).  Eigenvalue endpoints identical across every thread count.

### AW.2 — the fix (in-handler, not in-harness)

`scalapack/cpp/blacs_grid.h::MklThreadScope` — pins the CALLING thread's
MKL team via `MKL_Set_Num_Threads_Local` (dlsym RTLD_DEFAULT: zero MKL
link dependency, clean no-op on a non-MKL ScaLAPACK), restores the
previous local value on scope exit.  Applied around the whole batched
loop in `eigh_ffi.cc` + `solve_lu_ffi.cc`.  Default `auto` =
min(mkl_get_max_threads(), 4), i.e. caps only when it would shrink;
`LORRAX_SCALAPACK_MKL_THREADS=off|<N>` overrides.  The harness
`MKL_NUM_THREADS=28` is UNTOUCHED and must stay (plan-A local route:
657 GF/s at 28 thr).  Scope conditions recorded in linalg_ffi.md
(n=2448, P=16–144; re-verify if plan-B NB redistribution lands).

### AW.3 — I/O env: forced ROMIO hints → pass-throughs; the stripe-size naming split; the AS.4b guard

* `phdf5/cpp/context.cc` FORCED `romio_cb_write=enable`,
  `romio_ds_write=disable`, `cb_buffer_size=64M`, `cb_nodes=world_size`
  — Perlmutter/OpenMPI-era tunings (0.85→4.4 GB/s THERE), never
  revalidated on Frontera, where AI measured forcing `cb_write=enable`
  *slower* than ROMIO auto (1826 vs 2066 MB/s).  All four are now set
  only when the env is non-empty — identical semantics to
  `_slab_io_mpi_host._mpi_io_hints`, so unset-env = "ROMIO decides" in
  EVERY writer and a set knob reaches both (AM's "same knob, same
  meaning").  Python writer gained the `CB_BUFFER_SIZE` passthrough.
* **Registry bug closed**: the C++ writer read only the undocumented
  byte-valued `LORRAX_PHDF5_STRIPE_SIZE`; the documented knob is
  `LORRAX_PHDF5_STRIPE_SIZE_FS` ("4M").  Since AM made PHDF5_FFI the
  default CPU writer, the documented knob did not reach the default
  writer.  context.cc now parses `_FS` first (suffix form), legacy
  bytes as fallback.  VERIFIED live: `isdf_tensors_785.h5` written by
  the new lib comes back `lfs getstripe` = 16 × 4 MiB.
* **Known gap found, not fixed (out of files-scope): `zeta_q.h5` under
  the FFI route is STILL 1-stripe** — its inode is pre-created by
  h5py (isdf_fitting) where `_lustre_prestripe` is the in-container
  no-op, and the FFI opens it mode='a', where striping hints are inert
  by design.  Fix belongs at the isdf_fitting creation site (route the
  create through a SlabIO 'w' open or unlink-first) — flagged for the
  orchestrator; it predates AW (AI.1b measured the same 1-stripe on
  job 7876423's zeta_q.h5).
* `open_ctx` now runs `MPI_Query_thread` and WARNS on rank 0 when the
  granted level is below MPI_THREAD_MULTIPLE — the AS.4b race signature
  (jax impl=mpi + unpatched MPIwrapper grants FUNNELED; the phdf5
  writer thread's concurrent MPI-IO is then UB, measured ~29%
  multi-node crash rate).  The hazardous config self-announces at open
  instead of segfaulting mid-ζ.

### AW.4 — per-variable verdicts

| item | verdict |
|---|---|
| `MKL_NUM_THREADS=28` harness-wide | KEEP; fixed in-handler (AW.1/2) |
| `LORRAX_SCALAPACK_MKL_THREADS` | NEW; default auto(=cap 4), `off` restores pre-AW |
| `LORRAX_PHDF5_CB_WRITE/_DS_WRITE/_CB_NODES/_CB_BUFFER_SIZE` | KEEP as pass-through A/B levers; defaults now ROMIO-auto in BOTH writers |
| `LORRAX_PHDF5_STRIPE_COUNT/_SIZE_FS` | KEEP (the measured lever, AI); `_FS` now reaches C++; legacy byte var honoured |
| `LORRAX_FFI_SO` / `LORRAX_FFI_HOST_SO` | KEEP as-is: machine facts with sane in-tree defaults; `probe_target` already separates the 3 failure modes |
| LD_LIBRARY_PATH "SLATE first" | NOT required — host-lib RUNPATH covers SLATE; the dir holds only slate/blaspp/lapackpp so its position shadows nothing; REQUIRED-presence = HDF5 lib + libfabric + ICC runtime (no RUNPATH coverage).  Only measured ordering hazard remains AS.1's append-hostlibs rule.  Documented in ffi_loader docstring |
| h5py-before-RTLD_GLOBAL-dlopen (AG trap) | STRUCTURAL, not env-dependent: `get_lib` imports h5py before every dlopen; no env bypasses it |
| `MPITRAMPOLINE_LIB` auto-default inside LORRAX | REJECTED: names an out-of-repo scratch artifact; the certified(thr)-vs-hazardous(unpatched) choice must stay visible in the harness; MPItrampoline already refuses loudly when unset.  Documented at consumers (ffi_loader + _slab_io_mpi_host docstrings, env_vars.md §6 rows incl. the thr-build-only rule) |
| `LORRAX_MPI_FINALIZE_FIX` / sitecustomize | overlay-level, not read by src/; documented in env_vars.md §6 + module docstrings |
| `HDF5_USE_FILE_LOCKING=FALSE` | **Not load-bearing on /scratch2** (mount has real `flock`; full e2e with it UNSET: rc=0, all four output files bit-identical — run_800c_awlock).  MPIO VFD takes no locks; only serial-h5py side paths are governed.  KEEP the single harness export as a machine fact: /work2 is `localflock` (cross-node locks silently incoherent) and h5py wheels differ |
| linalg facade (`ffi/linalg/*`) | env-CLEAN by design (zero os.environ reads) — resolution is input/CLI-driven, pattern #8 compliant; no change |

### AW.5 — gates (ALL PASS)

| gate | result |
|---|---|
| host lib rebuild | `lorrax_ffi_unified/build_host_AW` from wt-F @ c847104, rc=0, all 11 target symbols exported |
| writer bit-compare (AM write_gate.py, P=4, new lib) | **15/15 BITCMP_OK** (new defaults vs collective=0/dedup=0 vs serial-h5py oracle; unique-tile, axis-0-replicated, replica-group, fully-replicated, valid_shape-padded classes), `[write_gate] PASS`, rc=0 |
| 4×4 e2e P=16, PHDF5_FFI route (build_host_AW), clean 8 nodes (job 7877394) | rc=0; **eqp0/eqp1/eqp_g0w0/sigma_diag ALL max\|Δ\| = 0.000e+00 vs run_800c** (1290/1290/2080/2080 rows) |
| e2e wall | Total recorded **84.36 s** vs 82.79 s (run_800c_merged, the current-gen baseline — within noise) vs 258.9 s (run_800c, the named AJ baseline, pre-AK/AL em1 era).  NOT WORSE |
| HDF5_USE_FILE_LOCKING unset e2e | rc=0, Total recorded 82.95 s, all four files EQUAL |
| pzheevd/pzgetrf thread-matrix | AW.1 tables (the required cells incl. {1,4,14,28} at the production shape) |
| CUDA lib rebuild | NOT trivially available (no rtx window this session) — SPEC'd: context.cc is shared-core; rebuild liblorrax_ffi.so from a tree carrying this diff + rerun the AM rtx smoke before the next GPU campaign |

Files: `src/ffi/phdf5/cpp/context.cc`, `src/ffi/scalapack/cpp/{blacs_grid.h,
eigh_ffi.cc,solve_lu_ffi.cc}`, `src/ffi/common/cpp/host/CMakeLists.txt`
(+CMAKE_DL_LIBS), `src/ffi/common/ffi_loader.py`,
`src/file_io/_slab_io_mpi_host.py`, `src/ffi/PORTING.md`,
`src/ffi/phdf5/ARCHITECTURE.md`, `docs/dev/env_vars.md`,
`docs/dev/linalg_ffi.md`.  Harness-line changes are SPEC ONLY (AU owns):
`wk_ENV/AW_harness_spec.md`.  Artifacts:
`/scratch2/08271/jackmc/lorrax_setup/wk_ENV/` (aw_benv.sh,
aw_mkl_matrix{,2}.sh, pzlu_bench.c + binary, aw_e2e*.sh, aw_gates.sbatch,
logs/, wgate/); run dirs `mos2_4x4_test/run_800c_awffi2` (the gate),
`run_800c_awlock` (locking cell), `run_800c_awffi` (holder co-tenant
casualty — log frozen mid-ζ under 2× oversubscription with AV2, killed by
AW at 17:44, disregard).  Co-tenancy note: AW scancel'd ONLY its own steps
(7877328.86/.112, both `--job-name=AW_e2e`).

## AV — the LORRAX application-knob family audited: two env twins deprecated loudly, the 128 MB collective cap RE-PRICED on the post-em1 transports (verdict: keep), and env_vars.md rebuilt around capability-vs-policy (wt-E, branch `env-audit-lorrax-knobs` @ 58a75aa, base 0d33b81, 2026-07-27 — COMMITTED on the branch, not merged)

**One line: none of the LORRAX_* application knobs slows communication at
its default — the one knob that touches comms (`LORRAX_COLLECTIVE_CHUNK_MB`,
AF's 128 MB payload cap) measures FREE at P=16 (impl=mpi/mlx, five caps flat
596-714 ms, z_q control flat) and costs only ~2-3 % of the ζ-fit at P=64 on
ib0-Gloo (both directions bounded by clean same-window cells), so the
default stays 128 = protective at the only measured fatality point
(P=144/em1, 1.15 GB single-shot) and near-free everywhere measured;
the real defects found were POLICY leaks, not slow paths: the ζ conditioning
twins (`LORRAX_ZETA_RCOND`/`_RIDGE`) silently beat the input keys — now
deprecation-warned with the key as source of truth — and the doc registry was
18 knobs stale.**

Artifacts: `/scratch2/08271/jackmc/lorrax_setup/wk_AV/` — `step_zeta.sh`,
`step_gate16.sh`, `mkcell.sh`, `headrun_AV{1,2,3,4,5}.sh` (head-node
drivers, `--overlap` steps labeled `AV_*`), `logs/` (all cells).  Run dirs:
`mos2_4x4_test/run_AV_*`.  Harness spec for AU:
`/scratch2/08271/jackmc/lorrax_setup/wk_ENV/AV_harness_spec.md`.

### AV.1 — the chunk-knob measurement matrix (THE headline item)

Deck: 4×4/785c (μ_pad=800), `distributed_zeta_solve=distributed`,
`r_chunk_size=11520` (4 r-chunks), `LORRAX_EXIT_AFTER_ZETA=1` +
`LORRAX_RCHUNK_DEBUG=1`.  nq=10 (IBZ).  Per-q largest collective =
Z-column all-gather('x') μ·(r_chunk/Py)·16 = **36.9 MB** (P=16, 4×4 mesh)
/ **18.4 MB** (P=64, 8×8).  The C⁺-formation site (2.56 MB/q) never
chunks at these sizes — the back-solve is the discriminating site, as at
production (AF).  Signal = steady-state `chunk.solve` (chunks 2-4);
control = `z_q_build` (cap-independent code, exposes co-tenancy).

**P=16, impl=mpi (provider mlx, AS.4c stack), SAME-window sequential,
quiet reps only (z_q control 3.43-3.55 s flat):**

| cap (MB) | q_block | execs | largest collective/exec | solve, chunks 2-4 (ms) | mean |
|---|---|---|---|---|---|
| 64 | 1 | 10 | 36.9 MB | 630/560/649 | **613** |
| 128 (default) | 3 | 4 | 110.6 MB | 681/630/594 | **635** |
| 256 | 7 | 2 | 258.0 MB | 690/706/747 | **714** |
| 512 | 10 | 1 | 368.6 MB | 570/655/668 | **631** |
| 0 = unbounded | 10 | 1 | 368.6 MB | 607/779/673 + 624/569/594 (rep 2) | **686 / 596** |

**FLAT: 596-714 ms with no monotone trend.**  10 executions of 36.9 MB
≈ 1 execution of 369 MB on mlx at P=16 — the host-loop dispatch overhead
is below the chunk-to-chunk noise (±8 %).  Chunk-1 (compile-inclusive)
walls are equally flat (4.5-5.0 s across caps): the extra compiled shape
per q-block size is immaterial.

**P=16, impl=gloo + ib0 pin:** quiet-window points only exist at
q_block=10 — `solve` 750/775/679 (pass 1) and 710/667 (pass 2) ≈ **0.7 s,
statistically identical to mpi/mlx**.  Every chunked gloo cell landed on
windows co-tenanted by AU/AW/AT steps (z_q control 5-33 s vs 3.5 s quiet
— up to 10×) and is REJECTED by the control rather than reported.  The
honest statement: on ib0 at P=16 the unbounded 369 MB single-shot is
demonstrably fine, and no quiet chunked-gloo cell contradicts the mpi/mlx
flatness; a dedicated-window gloo A/B is the one cell this matrix lacks.

**P=64 (8×8, 32 nodes), impl=gloo/ib0 (19.2 MB/q; quiet cells only —
z_q control 0.82-0.94 s flat; the one flooded cell, first cap-128
attempt with AU's 16-node step co-resident, showed z_q 2.2→6.1 s and is
rejected):**

| cap (MB) | q_block | execs | largest collective/exec | solve, chunks 2-4 (ms) | mean |
|---|---|---|---|---|---|
| 64 | 3 | 4 | 57.5 MB | 379/363/362 | **368** |
| 128 (default) | 7 | 2 | 134.2 MB | 397/381/385 | **388** |
| 256 | 10 | 1 | 191.7 MB | 313/306/341 | **320** |
| 512 | 10 | 1 | 191.7 MB | 321/315/322 | **319** |
| 0 = unbounded | 10 | 1 | 191.7 MB | 336/327/337 | **333** |

(256/512 do not bite at 19.2 MB/q — they are unbounded-class replicas
and they bound the noise at ±7 ms.)  The unbounded class sits at
319-333 ms and the chunked cells at 368-388 ms: the host-loop dispatch
overhead is REAL at P=64 — **+35-70 ms per r-chunk, i.e. +11-21 % of a
0.33 s back-solve stage and ~2-3 % of the whole ζ-fit** — and it is NOT
monotone in execution count (qb=7/2-exec ≥ qb=3/4-exec), so it is
per-execution dispatch+sync, not payload.  At production scale the same
loop measured 1.9× FASTER than its per_q control with 48 exec/chunk
(AF.4), so the overhead shrinks into the win as the matrices grow.  The
emitted q_blocks match the arithmetic exactly at every cap — plumbing
verified at P=64.

**VERDICT — default stays `128`, and the scope conditions are now
recorded (pattern #9):** the cap's protective claim is scoped to
P=144/em1-Gloo (1.15 GB single-shot fatal, 0.104 GB good — AF); its cost
is now measured ≈ 0 at P ∈ {16, 64} on both certified transports at
payloads ≤ 369 MB.  Changing the default buys nothing measurable anywhere
and spends the only margin that exists at the one measured fatality
point.  What WOULD justify raising it: a P≥144 ib0/mlx cell showing GB-
class single-shots healthy — unreachable on a 40-node holder (P=80 max,
deck maxes at 8×8).  Both opt-in consumers (`distributed_zeta_solve`,
`w_dyson_solver=distributed`) are off the default path, so the knob has
ZERO default-path exposure at any scale.

### AV.2 — the per-knob verdict table (grep-complete for the family)

| knob (default) | why it exists | comms/perf at default | verdict |
|---|---|---|---|
| `LORRAX_COLLECTIVE_CHUNK_MB` (128) | AF payload bound | none on default path; ≈0 cost measured on opt-in tiers at P≤64 | **KEEP 128** (AV.1); harness must stop shadowing it (AV.4) |
| `LORRAX_COLLECTIVE_CHUNK_LOG` (1) | AF observability | rank-0 print, dedup'd | KEEP on |
| `LORRAX_ZETA_RCOND` (env twin) | pre-key conditioning dial | none | **DEPRECATED — implemented** (AV.3) |
| `LORRAX_ZETA_RIDGE` (env twin) | same | none | **DEPRECATED — implemented** |
| `LORRAX_ZETA_GATHER_CAP_GIB` (4) | live-bytes gather budget (T) | granularity only, measured neutral | KEEP |
| `LORRAX_ZETA_REPLICATE_CAP_GIB` (4) | replicated-route reachability gate | route-affecting | KEEP env; **#1 promotion candidate** (doc §1b) |
| `LORRAX_ZETA_RANK_LOG` (1) | K's n_keep conditioning telemetry | jax.debug.print, ~0 | KEEP on |
| `LORRAX_SANITY` (warn) | O's stage-boundary invariant gates | ms/stage | KEEP (strict in CI) |
| `LORRAX_MEM_DEBUG` (off) | HBM probes | 0 off | KEEP |
| `LORRAX_FAILFAST` (1) | CLI failure propagation (#7) | none | KEEP on |
| `LORRAX_TIMING_TRACE`/`_DEPTH` (0/3) | section tracing | 0 off | KEEP |
| `LORRAX_RESTART_WRITE_LOG` (1) | AF.4c liveness instrument | rank-0 print | KEEP on |
| `LORRAX_PER_PROC_RESTART` (0) | AI forensic per-rank dump | 4m43s + 72 GB when ON at c2406/P=144 | KEEP OFF; **flag for deletion** (no in-tree reader; file lives in file_io/ — I/O sibling's lane) |
| `LORRAX_W_RESIDUAL_CHECK` (0) | AN's distributed-W numerical contract | +1 diagnostic jit when on | KEEP off |
| `LORRAX_FORCE_REFIT` (off) | ζ-reuse escape hatch | none | KEEP |
| `LORRAX_FORCE_FULL_BZ` (0 ×5) | IBZ-cascade debug bypass | more work when on | KEEP (defaults consistent) |
| `LORRAX_EXIT_AFTER_ZETA` (off) | profiling short-circuit | none | KEEP (this audit ran on it) |
| `LORRAX_MAX_RCHUNKS` (unset) | profiling/sweep ceiling | perf when set | KEEP |
| `LORRAX_RCHUNK_DEBUG` (off) | per-chunk telemetry | 0 off | KEEP |
| `LORRAX_SKIP_VQ_GATES` (0) | skip V_Q self-checks | saves check time when on | KEEP off |
| `LORRAX_GALERKIN_CHUNK_GIB` (6) | htransform accum budget | perf | KEEP |
| `LORRAX_CHECK_REPLICA` (0) | re-arm AA's replica assert | O(P×tensor) when on | KEEP off |
| `LORRAX_ALLOW_PARTIAL_ZETA` (0) | forensic read of incomplete ζ | none | KEEP off |
| `LORRAX_EXTRA_MU_PAD` (0) | pad-invariance gate | none | KEEP (test-only) |
| `LORRAX_MALLOC_TRIM` (1, isdf_fitting) | T.2 RSS cure | positive | KEEP on |
| `ISDF_CHUNK_TARGET_UTILIZATION` / `ISDF_ZCT_STAGE_CAP_GB` / `_FRAC` | planner dials | memory/perf | KEEP; promotion candidates #2 |

(`LORRAX_TRS_*`, `LORRAX_MALLOC_{TUNE,MMAP_MB,TRIM_MB}`, `LORRAX_FAILFAST`
implementation files live in `runtime/` / `file_io/` / `common/` shared
lanes — verdicts recorded here and in the doc; no cross-lane edits made.)

**Direct answers to the owner's three questions:** (a) *too many
variables?* — 27 in the family; 2 deprecated (implemented), 1 flagged for
deletion, 6 ranked for promotion to input keys, the rest are justified
machine-caps or debug switches now classified in the doc; (b) *any that
slow communication?* — none at defaults; the print knobs are rank-0-only
and dedup'd; the chunk cap measured free (AV.1); (c) *any suboptimal?* —
the chunk default was the candidate and survives re-pricing with its
scope now honest; the two silent env-beats-key sites were the real rot
and are fixed.

### AV.3 — implemented changes (branch `env-audit-lorrax-knobs`)

* `isdf/core.py::_deprecated_env_float` — input key is the source of
  truth; a non-empty env twin still overrides but prints ONE rank-0
  deprecation notice.  Applied at both `rank_truncate` factor sites
  (replicated + distributed).  Empty env now counts as unset instead of
  crashing `float('')`.  No live harness sets the twins (audited: the one
  historical user is the completed `run_B_c1998_rcond10/run72.sbatch`).
* `gw_init.py` ζ-provenance echo mirrors the empty-is-unset semantics;
  recorded strings byte-identical for every case that ever produced a
  reusable ζ (no spurious refits).
* `docs/dev/env_vars.md` REBUILT: input-file keys (env twins deprecated)
  / machine-capability env / debug env, defaults + measured-scope
  columns; 18 post-07-25 knobs added (SANITY, FAILFAST, TIMING_TRACE,
  RESTART_WRITE_LOG, PER_PROC_RESTART, COLLECTIVE_CHUNK*, CHECK_REPLICA,
  MALLOC_*, ALLOW_PARTIAL_ZETA, the AH cache family, …); consistency
  audit re-run (no default drift; ZETA_RCOND now via one shared helper).
  Siblings' rows (PHDF5_*/FFI/cache/runtime) carried forward for the
  orchestrator to fold their updates into.

### AV.4 — gates

| gate | result |
|---|---|
| 4×4 e2e P=16, final defaults (deck unchanged, gloo+ib0): eqp0 vs `run_800c` | **PASS — data rows bit-identical** (only the `# Generated at` timestamp line differs); rc=0 |
| wall not worse | **PASS** — same-session A/B on twin quiet windows: treatment (this tree) `Total recorded` **83.2 s** vs control (0d33b81) **86.1 s**; both ≪ the AJ baseline (384 s step / 258.9 s recorded, em1-era).  An earlier gate cell under sibling flood read 352.7 s — rejected by the co-tenancy control and superseded by the A/B. |
| deprecation banner fires + numerics inert (`LORRAX_ZETA_RCOND=1e-8` = key default, EXIT_AFTER_ZETA) | **PASS** — banner printed once on rank 0, fit completed, clean SystemExit(0), rc=0 |
| chunk-default change gate | **N/A — default unchanged** (AV.1 verdict) |
| `tests/test_zeta_mesh_invariance.py::test_distributed_tier_collective_payload_is_bounded` | untouched and still pins default 128 — consistent with the verdict |

### AV.5 — named, not done

* **Quiet-window gloo chunked A/B at P=16** — every gloo chunked cell was
  co-tenancy-flooded (z_q control 5-33 s); the mpi/mlx matrix + gloo
  unbounded points carry the verdict.  One dedicated 8-node window ×
  ~15 min closes it.
* **P≥144 ib0/mlx re-pricing of the fatality point** — the only
  measurement that could justify RAISING the default; needs a 72-node
  window on the 12×12 deck (AC_CHUNK_MB={512,2048,0} × the C⁺ step).
* **`LORRAX_PER_PROC_RESTART` deletion** — file_io sibling's lane.
* **Promotions** (replicate-cap key, planner-dial keys, wfn-backend key)
  — ranked in the doc; each is a gw_config + one-site change.

## AT — jax-init env audit: the ib0 pin had a second silent-em1 hole (forgot-JAX_PLATFORMS + cuda-plugin discovery failure — MEASURED, closed), the compile-cache "" opt-out is now announced and its harness rationale refuted, and the CPU harness sheds its inert *_SOCKET_IFNAME exports (wk_ENV, 2026-07-27; branch env-audit-jaxinit @ 366a938 in wt-G, from 0d33b81 — committed, NOT merged)

> Deliverables: branch env-audit-jaxinit (src/runtime/__init__.py,
> src/common/jax_compile_cache.py, src/common/collectives.py,
> docs/dev/env_vars.md; one commit 366a938); per-variable verdict table for
> workstream AU at wk_ENV/AT_harness_spec.md; probes + cells in
> wk_ENV/{AT_head.sh,AT_inner.sh,AT_repin.py,AT_cmp.py,logs/}; run dirs
> mos2_4x4_test/run_AT_{gate16,gate16b,cache16}; cache artifact
> wk_ENV/jax_cache_AT/np16 (338 entries).

### AT.1 — the finding that upgrades AL: the pin's platform gate leaked (measured, then closed)

pin_gloo_interface engaged only when JAX_PLATFORMS == "cpu" exactly.  The
2-rank repin2 probe launched WITHOUT the export (i.e. bootstrap()'s
platform="gpu" default, "cuda,cpu"):
* the cuda PLUGIN fails at DISCOVERY time on CPU nodes ("Jax plugin
  configuration error", logged, never raised), so jax.devices() SUCCEEDS on
  cpu — the RuntimeError path fallback_to_cpu_if_no_gpu_backend catches
  never fires (its docstring's 'Unable to initialize backend' strings are
  the plugin-INSTALLED-but-GPU-absent shapes; a discovery failure raises
  nothing);
* the run continued multi-process on CPU with STOCK Gloo transport — the
  silent em1 regime of AK.4/AK.10 (3.3x whole-pipeline), no banner, nothing
  to grep for.  A forgotten export in any future harness = silent 3.3x.
FIX (branch, two layers): (a) the pin now engages whenever "cpu" is in the
JAX_PLATFORMS list AND no GPU is physically present (_gpu_is_present:
CUDA_VISIBLE_DEVICES=="" or /dev/nvidia*), banner quotes the platform;
(b) the CPU fallback's raise-then-downgrade path re-arms the pin sentinel
and re-runs the pin after forcing JAX_PLATFORMS=cpu (covers "gpu"/"cuda"
values without "cpu").  GPU-node behavior unchanged by construction (gate
(a) requires no GPU present) — rtx re-verification named in AT.5.
Probe after fix (repin2b): "[runtime] Gloo collectives pinned to ib0 ...
[JAX_PLATFORMS='cuda,cpu' with no GPU present: this run lands on the CPU
backend]", process_allgather EXACT on both ranks, rc=0.

### AT.2 — the other two repo changes

* jax_compile_cache: ISDF_JAX_CACHE_DIR="" opt-out is now ANNOUNCED on
  rank 0 (+ whitespace-only == opt-out).  The production harnesses carry
  `export ISDF_JAX_CACHE_DIR=""` with a comment ("MUST stay empty —
  deadlocks at P>1") that stopped being true when AH landed; the silent
  opt-out is why nobody noticed (quality-pattern #8).  Unit suite
  tests/test_compile_cache_agreement.py on the AT tree: 8 passed.
* collectives.device_put_process_local: LORRAX_CHECK_REPLICA parse
  hardened to the standard falsy vocabulary — previously "off"/"no"/"OFF"
  silently ENABLED the opt-in P-linear debug all-gather (7.8 GB/rank at
  P=64, Y.5) while reading as "disabled".
* docs/dev/env_vars.md: LORRAX_CHECK_REPLICA / LORRAX_FAILFAST /
  LORRAX_MALLOC_* registered; GLOO_IFNAME + cache rows updated.

### AT.3 — per-variable harness verdicts (full table: wk_ENV/AT_harness_spec.md; harness edits are AU's lane)

* KEEP (load-bearing): JAX_PLATFORMS=cpu (backend + pin contract — see
  AT.1 for what its absence used to cost), JAX_ENABLE_X64=1,
  CUDA_VISIBLE_DEVICES="" (feeds _gpu_is_present), JAX_COORDINATOR_ADDRESS
  (per-LAUNCH port under shared holders; control plane on em1 is fine —
  AP.6), OMP/MKL/OPENBLAS_NUM_THREADS=28 (BLAS subset only: MKL pzheevd
  team, SLATE OpenMP tasks, numpy's bundled scipy_openblas64 — NOT the XLA
  pool, which sizes from taskset affinity, ADVICE §10),
  OMP_MAX_ACTIVE_LEVELS=2 + MKL_DYNAMIC=FALSE only in SLATE-capable
  harnesses (runAC), flagged as a 28x28 nesting hazard if SLATE-over-MKL
  ever nests — do not propagate to other harnesses.
* REMOVE: GLOO_SOCKET_IFNAME=ib0 (INERT — string absent from shipped
  jax/jaxlib, AF.5/AK.4; runAC's "stalls ~280 s without it" comment is
  refuted archaeology — that stall was em1 transport, fixed by AL's
  interface= pin) and NCCL_SOCKET_IFNAME=ib0 from CPU harnesses (no NCCL
  in a CPU run; keep on GPU harnesses).  ~30 carrier files inventoried in
  the spec; live templates: runAC.sbatch, alloc_run.sh, cpumn_a.sbatch,
  wk_AD/gate.sbatch, wk_AL/gate40_ib0.sbatch, wk_R/band/run_bands.sbatch.
* XLA_FLAGS: none steady-state; NEVER --intra_op_parallelism_threads
  (nonexistent in this jaxlib, F-abort — ADVICE §10 / job 7874158);
  rank-0-only HLO dumps forfeit the compile cache LOUDLY via the AH
  key-env fingerprint (by design) — export dump flags on ALL ranks with
  --xla_dump_hlo_module_re when a cacheable probe is wanted.
* CHANGE: ISDF_JAX_CACHE_DIR="" -> $SCRATCH/lorrax_jax_cache (AT.4).
* alloc_run.sh flagged (AU): OMP=14 hardcoded regardless of TPN and NO
  taskset — XLA:CPU sizes its pool to all 56 visible cores per rank on
  multi-rank steps.

### AT.4 — gates (785c 4x4 deck, P=16 8x2x28, gloo/ib0, provider auto, spec env; holder 7877328 under 3-sibling co-tenancy)

| cell | rc | Total recorded | physics (vs run_800c_gloo == run_800c) |
|---|---|---|---|
| gate16 (cache "") | 0 | 742.4 s — co-tenancy artifact: AV's 512-rank step shared all 8 nodes; excess sits ENTIRELY in zeta_fit.cholesky (522.7 vs 2.7) + write_headers (118.5 vs 0.1); rows outside the window at/below baseline (chunk.solve 3.75 vs 3.67, write_g_flat 0.73 vs 4.63, V_q 2.47 vs 3.75, sigma 50.8 vs 47.1) | eqp0/eqp1/eqp_g0w0/sigma_diag data rows ALL BYTE-IDENTICAL |
| **gate16b (rerun, quieter)** | 0 | **90.86 s vs 94.69 s baseline (AS.4) — NOT WORSE (better)** | ALL BYTE-IDENTICAL |
| cache_cold ($SCRATCH dir) | 0 | 354.6 s (co-tenant throughout) | ARMED cold; 410 compiles/rank; 338 entries written (atomic, p0) |
| cache_warm | 0 | **98.46 s** | ALL BYTE-IDENTICAL; 337/337 agreed, rank 0: 3 compiles (0.56 s) vs 410 (13.9 s), 407 hits (1.18 s reads), agree+prefetch 0.87 s |
| p1fix (P=1 cohsex fixture) | 0 | 35 s step wall | completed; opt-out banner + compile counter live at P=1 |
| repin2 / repin2b (2-rank, no JAX_PLATFORMS) | 0 | 4 s | AT.1: hole measured pre-fix, pin banner + exact allgather post-fix |
| pytest test_compile_cache_agreement.py | 0 | 1.2 s | 8 passed |

Cache verdict for the harness (AU): ENABLE — at this deck wall-neutral
(98.5 vs 94.7 within the AS.4c co-tenancy band; compile is only ~14 s/rank
here and overlapped), physics identical, and the win scales with the
compile storm (per-rank compile count is problem-size-invariant; the
2208-compile decks are the target).  LORRAX_JAX_CACHE_MULTIPROCESS stays
default-on; =0 remains the bisect hatch.

### AT.5 — named, not done
1. rtx verification of the AT.1 pin-gate change (no GPU window this
   session; change is no-GPU-guarded by construction, and GPU-node
   behavior is untouched by inspection).
2. Harness edits themselves — AU's lane (spec at wk_ENV/AT_harness_spec.md).
3. alloc_run.sh taskset/OMP defects (flagged, AU's lane).
4. The fallback docstring's two 'Unable to initialize backend' strings are
   now known to be the plugin-installed-only shapes; if a future venv
   drops the cuda plugin entirely, re-probe (repin pattern in
   wk_ENV/AT_repin.py).

## AX — bispinor propagation: the campaign's charge-path rewrites audited against the transverse/bispinor path, two real defects fixed, and the first bispinor runs on the 4×4 deck (wt-BC, branch `bispinor-propagation` @ 683cdb6, base 4c8d143, 2026-07-27/28 — COMMITTED on the branch, not merged)

**One line: every campaign change propagates to the bispinor path by construction
(shared code) except two — the transverse distributed-LU silent fallback and the
bispinor restart hole (Σ^B silently dropped at rc=0) — both fixed and gated; the
4×4 deck now has bispinor baselines at 402c+143T/P=4 and 785c+275T/P=16 with the
charge-unchanged gate at bit-zero.**

### AX.1 — propagation verdict table (item × verdict)

| # | campaign change | bispinor/transverse verdict |
|---|---|---|
| 1 | AO centroid device_put fixes (kmeans_isdf.py::shard, pivoted_cholesky orbit_id) | **propagated by construction** — the `--density-mode current` path shares the same kmeans/prune/shard code; only the weight array differs. AO's `_reshard_all` split + pre_reshard_sync live in `load_centroids_band_chunked`, which the transverse ψ load (gw_init) calls verbatim. |
| 2a | AN two-plan W doctrine (`w_dyson_solver=local\|distributed`) | **propagated / n-a by design** — a bispinor run's screening+W runs the identical two-plan machinery on the charge channel (CC tile read back as V_qmunu); the TT tiles are BARE (Phase-1 DHF+Breit, no transverse W Dyson solve exists). Zero stale plumbing: no ScreeningSolver/isdf_memory_mode/low_mem refs anywhere in the bispinor files. |
| 2b | transverse replicated-LU silent fallback (ledger 6 / W.3.1) | **FIXED (d15f716)** — divisibility moved to RESOLVE time in `_resolve_solver_kind_transverse(n_rmu_logical=)`: explicit `distributed_lu=cusolvermp\|on\|scalapack` + non-dividing n_rmu_T now **raises** naming the fix; `auto` announces the demotion (rank-0 print). solve_zeta's in-body guard kept as announced defense-in-depth (print, not warnings.warn). Subprocess test pins refuse(135@2×2)/honor(136)/auto→lu. |
| 3 | AI/AM/AW collective writer + provenance | **propagated** — all transverse tensors (ζ_T ×3, all 7 V^{μν} tiles, ψ_T) route through SlabIO (same backend enum, collective writes, stripe hints); ζ_T files carry the same isdf_header (vertex_mu_L, r_mu_fft_idx, zeta_is_done) and ZetaLoader's fail-loud gates. Restart gap **FIXED (3d89885)**: the gw_init claim "consumers fail loud" was FALSE — the Σ^B fold-in is a silent no-op on None, so a bispinor restart ran rc=0 with Σ^B dropped. Contained fix as designed: per-channel `psi_full_y_transverse` dataset (own logical μ_T clip + attr), loader re-pad, restart rebuild of the σ^B bundle, loud refusals (pre-fix file, μ_T-vs-centroids_file_current mismatch, missing v_q_bispinor.h5, bispinor+None invariant in gw_jax). |
| 4 | AO device_put sweep in bispinor gw files | **clean + one completion (683cdb6)** — v_q_bispinor.py / sigma_x_bispinor.py / wavefunction_bundle.py have NO raw multi-process device_put (all I/O via SlabIO). One shared-path payer AO missed: `cohsex_sigma.build_Gij` (np.eye operand, deterministic) → device_put_process_local. |
| 5 | AM use_ffi_io deprecation | **propagated** — the bispinor entry points' `use_ffi_io: bool\|None = None` kwargs are inert tri-state defaults normalized centrally by `_normalize_slab_backend`; all bispinor call sites pass `backend=cfg.backend.slab_io`. |
| 6 | μ padding | **shared machinery** — meta_curr refresh (gw_init), BispinorVqReader `_padded_shape_LR`, sigma_x_bispinor `_pad_V_to_padded`, solve_zeta logical-extent slicing all route through `runtime.padding.padded_mu_extent`. Both new transverse sets chosen pads-live: 143 % 4 = 3 (P=4), 275 % 16 = 3 (P=16). |
| 7 | cusolvermp_charge / cusolvermp_lu legacy keys | **CONSISTENT — removal unblocked.** Both channels resolve through ONE shared ladder (`_resolve_channel_ladder`); input keys `distributed_cholesky`/`distributed_lu` are the only things any consumer reads (gw_config → gw_init → isdf_fitting, both channels); the legacy keys exist ONLY in gw_config's parse-time alias block (honored iff the portable key is 'auto', deprecation-warned). No src/ consumer reads them. |

### AX.2 — transverse centroid sets (provenance = charge recipe + `--density-mode current`)
`kmeans_cli N --orbit --density-mode current --qe-save out/MoS2.save` on the deck
(same orbit-closure treatment; weight = Σ(j^Gordon)² with n_occ=26; prune v×(v+c)
window (0,26)×(0,52); header comments name density/weight/channels exactly like the
charge sets). Requested 134 → 140 (divisible by 4, REJECTED) → retried 135 → **143**
(`centroids_frac_143_current.txt`, = `centroids_T_t134.txt`); requested 262 → **275**
(`centroids_T_t262.txt`). Ratios 143/402 = 0.356, 275/785 = 0.350.

### AX.3 — runs (all on the merged env of gw800_merged.sbatch, PYTHONPATH→wt-BC; holder 7877628)
- **charge gate**: 785c/P=16 fresh run → eqp0 AND eqp1 data rows **BIT-IDENTICAL to
  run_800c** (and to run_800c_merged). The three fixes are charge-invisible.
- **bi4** (402c+143T, P=4/2×2, mesh pads live): rc=0, all sanity gates silent, wall
  ~7.2 min (total recorded 433 s; ζ×4 fits, 7 V tiles, Σ^B 9 tiles, GN-PPM Σ).
  QP gap 3.8309 eV (charge 402c: 3.5819 → Σ^B fold-in +0.249 eV).
- **bi16** (785c+275T, P=16/4×4): rc=0, gates silent. QP gap 3.8156 eV (charge 785c:
  3.5867 → Σ^B +0.229 eV; 20 meV vs the small cell).
- **bi4 restart** (restart=true, same dir): takes the restart path ("σ^B-side Wfns
  rebuilt from restart (n_rmu_T=143)"), eqp0 **BIT-IDENTICAL** to the fresh bi4 run —
  the new per-channel round-trip is exact.
- **pytest**: 14 passed (new transverse-LU resolve test, extended restart pad
  round-trip incl. transverse case, existing sigma_x_bispinor + V_q_bispinor units).

### AX.4 — bispinor eqp plausibility (same signals as the charge audit; eqp_stats.py in wk_AX)
| signal | charge 402c | bi4 (402c+143T) | charge 785c | bi16 (785c+275T) |
|---|---|---|---|---|
| deep-valence std-over-k (bands 1-6) | 3–115 meV | 30–136 meV | 3–32 meV | 3–36 meV |
| Kramers pair degeneracy of corrections (mean) | 5.7 meV | 6.9 meV | 6.2 meV | 7.1 meV |
| cond-4 band means | +1.89..+1.97 eV | +2.18..+2.26 eV | +1.85..+1.92 eV | +2.09..+2.17 eV |
| cond-4 spread of means | 77 meV | 83 meV | 78 meV | 89 meV |
| QP gap | 3.5819 | 3.8309 | 3.5867 | 3.8156 |

Σ^B acts as a nearly k-rigid shift (cond std-over-k pattern preserved: 302→291,
136→138 meV etc.); TRS/Kramers structure intact at the same meV level as charge;
Σ^B tile traces Hermitian-consistent (V^{ij}/V^{ji} pairs equal to 6 digits).

### AX.5 — the 1/3-ratio judgment
No kept-rank exists for the transverse factor (indefinite CCT → LU + ridge; W.3),
so the conditioning signals are: (a) the CHARGE basis at 402 is itself un-truncated
(n_keep/q = 402/402 at rcond 1e-8) — the deck is not rank-limited at these counts;
(b) Σ^B convergence: per-tile traces move −0.468→−0.514 eV (diag) and the eqp-level
Σ^B gap shift moves only 249→229 meV between (402c,143T) and (785c,275T) — the
transverse quadrature at ~1/3 is stable at the tens-of-meV level on this deck, NOT
starved. Verdict: ~1/3 is sufficient here; re-verify the ratio at production μ
(claim scoped to the 4×4 numerics deck, quality pattern #9).

### AX.6 — open
- bi1 (P=1, 402c+143T) vs bi4 P=4 consistency: launched; verdict in wk_AX/logs/bi1.log
  (compare data rows vs wk_AX/bi4_fresh_eqp0.dat).
- W.3 items 2/3 (distributed_lu key unification; ragged-block descriptor) unchanged.
- ζ reuse remains charge-only (bispinor always refits — documented, unchanged).

### AX.6-resolution — P=1 vs P=4 (402c+143T, identical inputs)
bi1 rc=0, gates silent, wall ~11 min. eqp0 P=1 vs P=4: 4/1280 rows differ, all
in the LAST printed digit — max |Δ| = 1e-9 eV (the same 1e-9 threshold AM's
GPU-vs-CPU eqp gate passed at). Device-count-invariant to printed precision.
Artifacts: wk_AX/{bi4_fresh,bi16,bi1,charge16}_eqp0.dat, logs/, eqp_stats.py.

## AU — MPI/libfabric transport env audited per-variable and the harness env blocks consolidated: FI_PROVIDER_PATH is REQUIRED (not hygiene), the "UCX defaults" were TACC module tunings riding in by env inheritance (load-bearing: 2x at 1 MiB allreduce), and the production harnesses' `auto` was silently still tcp for want of the staging binds — all fixed (wk_AU, 2026-07-27; repo branch `env-audit-transport` @ wt-H, NOT merged)

> Deliverables in /scratch2/08271/jackmc/lorrax_setup/wk_AU/: au_inner.sh
> (per-variable-strippable final transport env), au1.sh / au2.sh (head-node
> drivers, --job-name=AU_*), au_gw_inner.sh (e2e gate env = post-AU
> gw800_merged block), au4.sbatch (final-state gate job), logs/ (all
> cells).  Repo: branch env-audit-transport in /work2/08271/jackmc/
> frontera/wt-H, REBASED onto post-merge main 4c8d143 (single commit
> 975f958 — ffi_env.sh case-block + env_vars.md transport rows; the
> env_vars.md conflict resolved as a deduplicated superset: kept the
> AT/AV-rebuilt detailed rows, added AU's six measured rows, dropped the
> stale duplicate FI_PROVIDER/short-form rows the earlier merge left).
> Harnesses edited in place with .bak_AU_20260727 backups.

### AU.1 — the per-variable verdict table (cells: wk_AU/logs, quiet-window
### unless noted; baseline reproduced AS.2 exactly)

| variable | why it was set | measured | verdict / action |
|---|---|---|---|
| `FI_PROVIDER` seed in ffi_env.sh:82 (`:-tcp`) | rtx/mlx4 bring-up (AP.1) | AP/AS: tcp costs 5-13x on pzheevd, 42x on 8B allreduce | **DELETED** — replaced with the AS.5 `LORRAX_MPI_PROVIDER` case-block (commit 3dba3b6).  Bonus archaeology: the `:-tcp` fallback rarely applied anyway — TACC's impi module exports `FI_PROVIDER=mlx` into every shell and sbatch inherits it |
| `I_MPI_FABRICS=shm:ofi` | AP.7 block | unset ⇒ identical (pp 1.08 us / 11.37 GB/s vs 1.07 / 11.38; it IS the IMPI 2019+ default) | **KEEP** — zero-cost declaration + guard against stray inherited values |
| `I_MPI_PMI_LIBRARY` | pmi2 bootstrap | login env carries `/usr/lib64/libpmi.so` = PMI-**1**, wrong protocol AND absent in-container | **KEEP, scope documented**: needed only under `srun --mpi=pmi2`; must be set UNCONDITIONALLY there (inheritance poisons it); harmless elsewhere |
| `I_MPI_DEBUG=4` | provider banner | =0 ⇒ latency/bandwidth identical (1.07 us / 11.7 GB/s) — banner is init-time only | **KEEP =4** — the banner is the only trustworthy provider observable (fi_info false-negatives mlx); mandatory telemetry |
| `FI_PROVIDER_PATH` | AP.7 block | **unset ⇒ PMPI_Init FATAL** `MPIDI_OFI_mpi_init_hook ... addrinfo() No data available` (libfabric finds NO providers; mpivars.sh not sourced in-container).  Same error string as the rtx-era archaeology — some of that history was likely THIS, not the fabric | **REQUIRED — keep everywhere** (upgraded from "belt-and-braces") |
| `UCX_TLS` + 5 `UCX_*MLX5*` timeout/retry | never set by us — inherited from TACC's default impi module in every sbatch/ssh env | strip-all ⇒ 8B rows unchanged (pp 1.07 us, allreduce@32 3.38 vs 3.41 us) but **1 MiB Allreduce@32 419 -> 799 us (1.9x)**.  Every AP/AS mlx number was measured UNDER these tunings — QUALITY_PATTERNS #8 in the wild | **SETDEFAULT the six module values** in the MPI harness blocks (inherited wins; stripped launch envs no longer silently lose 2x).  Never hard-pin (rtx/mlx4 has no dc_x); no other UCX knobs.  (TLS-vs-timeouts isolation cell was co-tenancy-destroyed — moot: setdefault reproduces all six) |
| `FI_TCP_IFACE` logic | tcp branch of the case-block | tcp cell: 9.26 us / 2.97 GB/s on ib0, announced | **KEEP** — correct and scoped to the escape hatch |
| `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` | pre-AK cargo cult | inert with jax (AF.5/AK.4) / GPU-only | **REMOVED from every live CPU harness** (runAC, gw800_merged*, gw800_p16, gw400_p4, gate40_ib0, cpumn_a, alloc_run, wk_AD_scalapack/gate, wk_R/band/run_bands), each with a pointer to the real dial (`LORRAX_GLOO_IFNAME`) |
| LD_LIBRARY_PATH (transport slice) | libfabric/UCX resolution | staged RDMA/UCX symlink dir must be APPENDED (bare /hostlibs shadows container glibc — AS.1) | **KEPT/landed as append-only**; deeper LD ordering left to AW |

### AU.2 — the implementation gap closed: `auto` was still tcp in production

runAC.sbatch and gw800_merged.sbatch carried the AP.7 case-block but NOT
the AS.1 RDMA-userspace staging (binds or symlink block) — so
`LORRAX_MPI_PROVIDER=auto` in the PRODUCTION harnesses still degraded to
tcp in-container (announced, but the 30-min pzheevd wall would have been
back at the next 12x12 flagship).  Landed in both: the
`/usr/lib64:/hostlibs:ro,/usr/lib64/libibverbs,/etc/libibverbs.d` binds +
the appended symlink-staging block + the UCX setdefaults.  ffi_env.sh
(repo) now carries the same case-block (its `shm`-only fabric default kept
for the rtx single-node context).

### AU.3 — gates (in-container, holder 7877328 + dedicated job 7877617; sibling co-tenancy noted per cell)

| gate | result | vs reference |
|---|---|---|
| pingpong (2-node, provider auto, final env) | **1.07 us / 11.38 GB/s, banner mlx** | AS.2: 1.07-1.08 us / 11.42 GB/s — EQUAL |
| Allreduce@32/16 nodes 8 B / 1 MiB | **3.41 us / 419 us** | AP.3: 3.45 us / 421 us — EQUAL |
| tcp escape hatch | 9.26 us / 2.97 GB/s, banner tcp;ofi_rxm, FI_TCP_IFACE=ib0 | AS.2 control: 9.26 us / 2.08 GB/s — consistent |
| pzheevd n=2448 12x12 g=204 P=144 (production shape) | quiet cell (au3): steady reps **0.90 / 0.92 / 0.96 / 1.14 s/q** (one 49 s co-tenant blip), lam0/lamN IDENTICAL to AP/AS | AS.2: median 2.27 / best 0.52 (same warm-up shape) — within the certified band, better than median |
| 4x4 e2e P=16 gloo (au3, transport-final env, cache "") | rc=0, **eqp0/eqp1/eqp_g0w0/sigma_diag ALL BYTE-IDENTICAL to run_800c**; mlx banner + "[runtime] Gloo collectives pinned to ib0" both fired IN-CONTAINER | Total recorded 120.4 s vs 94.7 baseline — the delta is ONE 27.4 s zeta_fit.cholesky call (2.68 s in AS.4) concurrent with a demonstrated co-tenant burst (the pz rep2=49 s blip, same fabric window; AT's same-deck same-window cell recorded 742 s).  Transport-sensitive rows: write_g_flat 4.63 -> **1.38 s** (mpi4py coordination now rides mlx), chunk.solve/z_q_build/sigma within noise.  Wall verdict: not-worse net of co-tenancy, and the collective-bound writer row is 3.4x BETTER |
| **4x4 e2e P=16 FINAL harness state** (au4, dedicated job 7877617: transport env + cache ON per AT verdict, cold cache, quiet nodes) | rc=0; ALL THREE banners fired (`libfabric provider: mlx`, `[runtime] Gloo collectives pinned to ib0`, `[compile-cache] ARMED at 16 processes`); **eqp0/eqp1/eqp_g0w0/sigma_diag ALL BYTE-IDENTICAL to run_800c**; **Total recorded 93.09 s vs 94.69 s baseline — NOT WORSE (better), cold-cache** | closes the wall gate with no caveat and retroactively confirms the co-tenancy reading of the 120/139 s holder-window cells |

The mlx auto-selection banner fired in-container in EVERY auto cell,
including both e2e gates under the production harness env — the staging
gap (AU.2) is closed and announced.

### AU.4 — sibling harness specs: all three found and APPLIED

- **AT** (wk_ENV/AT_harness_spec.md): *_SOCKET_IFNAME purge everywhere
  (incl. AT's inventory additions wk_AD_scalapack/gate.sbatch +
  wk_R/band/run_bands.sbatch; step_*.sh one-offs left as dead records),
  refuted runAC stall-comment rewritten, alloc_run.sh taskset by
  SLURM_LOCALID + threads=56/TPN (their flagged oversubscription defect),
  and — after AT's final scorecard verdict landed ("ENABLE — wall-neutral
  98.5 vs 94.7, byte-identical, win scales with the compile storm") — the
  compile cache FLIPPED ON in runAC.sbatch + gw800_merged.sbatch
  (`ISDF_JAX_CACHE_DIR=${ISDF_JAX_CACHE_DIR-$SCRATCH/lorrax_jax_cache}`,
  explicit-"" opt-out preserved via the colon-less expansion).
- **AV** (AV_harness_spec.md): the `LORRAX_COLLECTIVE_CHUNK_MB=${AC_CHUNK_MB:-128}`
  code-default shadow replaced by a conditional export (only when the
  operator dials AC_CHUNK_MB) in the three live carriers wk_AF/runAF.sbatch,
  wk_AF/runAF_close.sbatch, wk_AI/runAI_close.sbatch (each .bak_AU'd).
  ZETA env-twin items: no live harness exports them — no action, confirmed.
- **AW** (AW_harness_spec.md): all verdicts are keep-as-is and match AU's
  edits (MKL_NUM_THREADS=28 kept, HDF5_USE_FILE_LOCKING kept, staged-libs
  APPEND rule already the implementation); the MPITRAMPOLINE_LIB
  thr-build-only warning is satisfied in documentation (no live harness
  carries the line; ADVICE §10c + AS.7 carry the warning).

### AU.5 — named, not done (repro attached, per the convergence directive)

1. The UCX TLS-vs-timeouts decomposition of the 419->799 us finding
   (nice-to-know; production is covered by the setdefault of all six
   module values).  Repro: wk_AU/au_inner.sh AU_UCX=strip|tlsonly + the
   au1.sh ar32 cells on a QUIET 16-node set — the one attempt
   (logs/ar32_tls.log) was co-tenancy-destroyed (24.8 ms at 1 MiB).
2. ffi_env.sh multi-node rtx path: I_MPI_FABRICS still defaults `shm`
   there and the mlx-over-mlx4 story is unmeasured; the tcp escape hatch
   is the documented rtx answer.  Repro: source ffi_env.sh with
   LORRAX_MPI_FABRICS=shm:ofi on 2 rtx nodes, watch the I_MPI_DEBUG
   banner.
(An earlier item — a zero-co-tenant e2e wall — was RESOLVED by au4/job
7877617: 93.09 s on dedicated nodes, better than baseline.)

## AY — pre-release audit (61 findings, 57 fixed) + the AQ 4962c/P=64 rehearsal: mpi collectives is the ONLY stack that survives the distributed tiers at P=64 (gloo/ib0 RESCOPED to P≤16), check_hermitian was the run's largest collective (fixed twice, HLO-pinned), the monolithic fused τ kernel is tried-and-REFUTED, and the size ladder is green through rung 2 with the memory model exact to 3.5%/0.6% (wk_REL + mos2_4x4_test, 2026-07-28; main checkout branch `fix/zq-band-gather-device-invariance`, commits b3bd130..dc30af4 on base 4ebe19e — COMMITTED on the branch, NOT pushed; +1 working-tree file NOT committed)

**One line: the release wave was audited (8 area auditors, 61 findings, 57 fixed
across 5 commits, every fix re-gated), the AQ P=64 rehearsal certified the mpi
collective stack (2/2 green, 514/554 s, Dyson ≤7.5e-15) while gloo/ib0 died
reproducibly in the distributed tiers (0/2, ReduceScatter 30 s timeout — a
transport-INDEPENDENT member of the AC.3 failure class), the Σ-perf round landed
instrumentation + AK.9's collective halving at exact-0 parity but refuted the
fused-τ-kernel hypothesis with HLO evidence, and the b256/b512 size ladder ran
rungs 0-2 green with a provenance-clean rung-3 deck staged (L3 job queued).**

### AY.0 — context

Same-day owner session on the 4ebe19e handoff state: (1) pre-release audit +
fix pass over the campaign's handoff diff; (2) the staged AQ rehearsal
(4962c = the `aq_c5000_driver.sh` ACCEPTED count, P=64 8×8, 32 nodes dev,
BOTH distributed tiers forced: `distributed_zeta_solve=distributed` +
`w_dyson_solver=distributed`); then the two owner-directed campaigns in order:
(3) Σ-stage perf (ppm_sigma 274.450 s vs zeta_fit_chunked 65.121 s at the AQ
shape — `run_AQ_c4962_p64_mpi/gw.log`, Total recorded 400.983 s — is the
imbalance AK.2's owner invariant flags) and (4) the size-escalation ladder
(SIZE_CAMPAIGN_BRIEF.md: 4×4 deck at 30 Ry, P=64 fixed, escalate until death).
Workstream dirs: `wk_REL/` (audit + Σ-perf + ladder notes), run dirs + harnesses
in `mos2_4x4_test/`. Every number below was read from the on-disk job log named
next to it (a notification-fabrication storm during the ladder — documented in
`wk_REL/ladder_rung1_notes.md` R1.3/R1.4/R2.2 — made this non-optional).

### AY.1 — pre-release audit: 61 findings, 57 fixed, every fix re-gated

8 fable/high area auditors over the handoff diff (findings JSON
`wk_REL/audit_findings.json`; fix groups `wk_REL/fixgroups/G1_fileio(12)/
G2_runtime(12)/G3_gwcore(22)/G4_bispinor_env(15).json`): **61 findings =
5 high / 27 medium / 29 low; 57 fixed in commits b3bd130…fe58318, 4 deferred**
(cublasmp fused-W package + `_chunk_q` promotion — owner ledger, annotated
in-tree). The 5 highs: FFI-writer `mode='w'` unlink never ran in-container
(Lustre striping silently lost — layout is fixed at inode create); Gloo-pin
warnings rank-0-gated (per-node NIC failures silent); resolve.py 1-D-mesh
SILENT demote on explicit `cusolvermp`; gw_config cusolvermp-on-CPU
demote-to-off; cohsex `x_head` bare `device_put`. Commit list (git log
4ebe19e..dc30af4, all on `fix/zq-band-gather-device-invariance`):

| commit | area |
|---|---|
| b3bd130 | sanity: fuse check_hermitian stats into one jitted module (AY.3 fix v1) |
| b955b42 | file_io+timing: mode='w' inode replace (both writers, loud on failure), announced allgather fallback, shared stripe parsing, restart-stamp reader |
| d23aeb1 | runtime+cache+device_put: rank-tagged Gloo-pin diagnostics, SCRATCH-first compile cache, x_head/qsgw `device_put_process_local` |
| 935ed14 | gw core+linalg+phdf5: explicit-request refusals at resolve/parse, announced auto demotes, probe-W dedup, C++/Python env grammar parity |
| 8487ff8 | bispinor+isdf+env: preflight transverse refusals, restart centroid-table hash guard, env registry completed, UCX setdefaults tracked |
| fe58318 | sanity: `with_sharding_constraint` in check_hermitian + HLO regression test (AY.3 fix v2) |
| 6805729 | handoff addendum (rehearsal verdicts, cache-cold rule, audit record) |
| dc30af4 | sigma: τ-loop instrumentation + branch-tail single gather + AK.9 stacked/reordered reduce-scatters (AY.4) |

Verification cells (`wk_REL/verify.7877788.out`, job 7877788): host lib
rebuilds from the fixed C++ (`build_host_AUDIT`, ScaLAPACK+phdf5 exports in
DT_NEEDED); 785c P=16 bare-input rc=0 173 s, **eqp0 vs run_800c max|Δ| =
0.00e+00 over 5160 values** (md5 differs by header timestamp only).
`test_slate_cholesky_trsm_cpu` hangs — bisected (job 7877804,
`wk_REL/bisect.7877804.out`): rc=143 IDENTICALLY on `build_host_AUDIT` and the
pre-audit `build_host_AW` lib → **PRE-EXISTING, not audit-introduced** (open
ledger; the full-pytest cell `wk_REL/pytest_v.7877798.out` hits the same test).

### AY.2 — AQ 4962c/P=64 rehearsal: mpi 2/2 GREEN, gloo/ib0 0/2

Harness `mos2_4x4_test/aq_rehearsal.sbatch` (`LORRAX_AQ_COLL=gloo|mpi`; needs
`LORRAX_FFI_HOST_SO` — the first gloo cell, job 7877749, rc=1 at 82 s, was
exactly that harness defect: no host lib ⇒ "Available eigh backends on this
mesh: native" ⇒ designed fail-fast; NOT a transport datum).

| cell | job | verdict |
|---|---|---|
| mpi collectives (AS.7 upgrade), warm cache | 7877754 | **rc=0 wall=554 s**, src@4ebe19e |
| mpi collectives, CACHE-COLD | 7877789 | **rc=0 wall=514 s**, src@8487ff8 |
| gloo/ib0, warm | 7877753 | **rc=1 wall=211 s** — `Gloo ReduceScatter failed: [gloo/transport/tcp/buffer.cc:72] Read timeout`, ranks die mid-distributed-tier |
| gloo/ib0, cold | 7877761 | **rc=1 wall=227 s** — same signature (FAIL-FAST ranks 35/52 in the run's gw.log) |

mpi cells: provider banner `libfabric provider: mlx`, route
`Computing L_q = distributed rank-truncated pinv … path=distributed_rank_truncate`
(ADVICE §6a satisfied), **W Dyson residual max |(1−Vχ)W−V|/|V| = 7.502e-15**
(4 q sampled, `run_AQ_c4962_p64_mpi/gw.log`). Suspected gloo mechanism:
ScaLAPACK-FFI stalls desync ranks past Gloo's HARD 30 s timeout; jax exposes
no knob. Dedicated 32-node dev jobs — no co-tenancy caveat.

> **⚠ CLAIM-DECAY (AY, 2026-07-28): gloo/ib0 is RESCOPED to P≤16.** The AL/AS/AU
> certifications of Gloo-on-ib0 as the CPU collective stack (P=80 e2e at 606c,
> P=16 byte-identical gates) all ran WITHOUT the ScaLAPACK-backed distributed
> tiers forced. At P=64 with both tiers forced it fails reproducibly, warm AND
> cold cache — a SECOND, transport-independent member of the AC.3/AF Gloo
> ReduceScatter failure class (AL correctly re-scoped the first to em1; this one
> survives ib0). The ib0 pin itself remains correct and load-bearing; the decayed
> claim is "gloo/ib0 suffices at scale", not "the pin works". **Distributed tiers
> at P≥64 require the mpi collectives stack**; certified-stack doc edit is
> owner-gated (AY.9). Older sections were deliberately not edited.

> **⚠ CLAIM-DECAY (AY, 2026-07-28): collective-table gates are only valid
> CACHE-COLD.** Cache-hit modules never re-dump HLO, so a warm-cache table
> silently under-reports (false "no violation"). Measured here: the cold mpi
> cell dumped 1356 after_optimizations modules vs 1130 warm
> (aq_rehearsal.7877789/7877754.out). Applies to any collective-table gate taken
> with `ISDF_JAX_CACHE_DIR` set since AH flipped the cache ON; physics gates are
> unaffected. All AY tables below are cache-cold.

### AY.3 — check_hermitian: the run's largest collective, fixed twice, HLO-pinned

1. **Defect** (AQ P=64 table at 4ebe19e, aq_rehearsal.7877754.out):
   `common/sanity.py::check_hermitian` did an EAGER transpose+subtract on the
   sharded (μ,μ) W tile → XLA all-gathered BOTH operands —
   `all-gather [4992,4992] 398.72 MB` ×2 in EACH of modules jit_subtract
   0730/0973, per gate call.
2. **Fix v1 (b3bd130) — NOT ENOUGH**: fusing into one cached jit
   (`_herm_stats`) still gathered — GSPMD re-introduced both full-tile
   gathers (jit_fn 0536/0698, same 398.72 MB, in the COLD 7877789 table).
3. **Fix v2 (fe58318)**: `with_sharding_constraint` pins the transposed
   operand's sharding. Pinned by
   `tests/test_sanity_gates_jax.py::test_check_hermitian_sharded_no_full_gather`
   — a compiled-HLO assertion (colltable predicate), PASSED in the bisect job's
   herm cell (7877804, 36 passed).
4. **Production confirmation**: the rung-1/rung-2 cold tables (src@6805729,
   AY.5) FLAG only the known ζ-apply gather — no (μ,μ) tile collective on any
   rank. AQ-table FLAG count 5 → 1.

### AY.4 — Σ-stage perf: decomposition measured, AK.9 halving landed at exact-0 parity, fusion REFUTED

Baseline for every A/B: `run_AQ_c4962_p64_mpi` (AY.2 cold cell) —
**sigma.exec 272.040 s over 176 τ**. Full log: `wk_REL/sigma_perf_results.md`;
ranked candidates `wk_REL/sigma_perf_candidates.json`. All parity gates below
are max|diff| on sigma_diag.dat/eqp0.dat/eqp1.dat at tol 1e-12.

**Round 1 (job 7878038, sigma_perf_ab.7878038.out, two passes, cache-cold):
parity exact 0.0 BOTH passes.** Landed: always-on τ-loop instrumentation
(`sigma.tau.dispatch/host_accum`, `sigma.finalize`, `sigma.host_gather`,
`LORRAX_SIGMA_TAU_TIMING=1` staged mode) + the AK.9 finalize-tail fix (Σ branch
tiles stay per-rank host numpy; ONE end-of-stage gather replaces 4× device
re-upload + full-slab 64-process allgather; bit-identical by construction).
Pass A (production) 273.278 s vs 272.040 (+0.5%, node noise). Pass B (staged,
NOT comparable, 313.503 s) — the first per-τ decomposition of the Σ wall
(shares of the 295.0 s dispatch, 176 τ):

| τ-stage row | s | note |
|---|---|---|
| w_phase | 5.70 | candidate #3 (exp specialization) is low-value |
| G_build | 11.17 | |
| G_ifft | 79.63 | FFT-adjacent layout churn … |
| V_ifft | 16.54 | … totals 191.9 s = 65% of staged τ time |
| GW_mult_fft | 95.74 | |
| project_rs | 84.75 | collectives + dots |
host numpy ω-projection: 0.71 s total.

> **⚠ CLAIM-DECAY (AY, vs sigma_perf_candidates.json): the analysts'
> "4-5 s/branch finalize tail" was the async-D2H deque drain (by design); the
> baseline's own Finished→Started seams were 0-1 s, so the eliminated gather
> cost only ~1-4 s at nb=128. The fix's value is AK.9's nb²-growth term
> (~237 MB/branch at b160), not this deck's seconds.**

**Round 2 (job 7878092): AK.9 stacking verified.** Both re/im channels ride ONE
psum_scatter per mesh axis: HLO reduce-scatter **4 → 2**/τ, payloads
2×5.11→1×10.22 MB ('x') + 2×0.07→1×0.13 MB ('y'), parity exact 0.0 (pass B).
Pass A was an INFRA failure (c208-020 apptainer squashfuse mount, rc=255, node
excluded thereafter), not code. Staged project_rs 84.7→47.6 s, BUT
collective-free rows moved ±25% between runs with staged total unchanged — at
~10 MB messages the stacking mostly relocates latency/skew wait at this shape
(claim recorded with measured domain).

**Round 3 (job 7878110, AC.4 restart-gated iteration harness `sigma_iter.sbatch`
— restart=true + tmp/ symlinked from the parity-verified isdf_tensors_4962.h5;
4 passes, ALL rc=0 and parity exact 0.0, validating the harness itself):**
- a (stacking + owner-approved axis-order swap, prod): **278.049 s** — the swap
  is parity-clean but shows no measurable win (Δ within the ±5 s cross-run band).
- c (monolithic FUSED τ kernel, prod): **278.959 s vs 272.040 baseline —
  performance-NEUTRAL** (d's single fused row 265.057 s/176 τ = 1.506 s/τ ≡ the
  decomposed rate). **Fused-module HLO REGRESSION: transposes 20 (18 large) vs 6
  decomposed** — XLA re-anchored layouts WORSE inside the merged shard_map; the
  flat-k helper-boundary transposes were NOT the binding constraint. Both owner
  gates (neutrality + near-zero large transposes) FAILED → **reverted; re-apply
  patch archived `wk_REL/fused_tau_refuted_2026-07-28.patch`**. Scope: measured
  ONLY at 4×4/nb=128/μ_pad=4992/8×8/XLA:CPU; re-open only on evidence from a
  different shape/backend.

**host_accum re-attribution (working tree, post-dc30af4):** `sigma.tau.host_accum`
split into `sigma.tau.d2h_wait` + `sigma.tau.omega_project`;
`_project_tau_onto_omega_np` pref-fold + astype-drop (3 full-size passes → 1).
Gate job 7878233 (sigma_haccum.7878233.out, 4 restart-gated passes): the nb=128
half is GREEN — k128a/b parity exact 0.0, k128a production split:
host_accum 245.78 s = **d2h_wait 244.97 s + omega_project 0.669 s** — at nb=128
host_accum IS the device-compute wait, not host math (the "72.7 s / 84% at
nb=256" alarm was a row-semantics artifact, as predicted in the notes). The
nb=256 half (l1a/l1b vs run_L1_b256) FAILED rc=1 at 16-17 s with no outputs —
the nb≥256 attribution is **named, not done** (AY.9).

### AY.5 — size ladder: rungs 0-2 GREEN, rung-3 deck provenance-clean, L3 queued

Campaign brief `wk_REL/SIZE_CAMPAIGN_BRIEF.md`; full agent notes (incl. two
monitor-integrity incidents and their re-verification rule)
`wk_REL/ladder_rung1_notes.md`. Fixed arch 32 nodes / P=64 8×8, coll=mpi,
cache-cold, both distributed tiers forced, budget 90 GB/dev.

| rung | job | shape | wall | planner HWM → measured VmHWM | ratio |
|---|---|---|---|---|---|
| 0 | 7877789 | nb=128, μ=4962 | rc=0 **514 s** | (40 GB budget era) | — |
| 1 | 7878104 | nb=256 (26v/230c), μ=2475 | rc=0 **434 s** | 8.61 GB/dev → 9,340,064 kB = 8.91 GiB | **1.035** |
| 2 | 7878225 | nb=256, μ=3491 | rc=0 **495 s** | 11.98 GB/dev → 12,641,028 kB = 12.06 GiB | **1.006** |

The memory model is essentially EXACT at these sizes (07-25 calibration was
1.22×); rung-1 sacct MaxRSS 9,353,324 kB agrees with the /proc sampler within
0.15%; per-rank spread 8.86-8.91 GiB, remarkably flat; 10-13% of budget —
nowhere near the memory wall yet. Artifact regen gates (deck_b256 job 7878101,
3 m 18 s): kin_ion_b256/dipole_b256 exactly 4× the 128-window files; kmeans
c2500→**2475 orbit-closed** (mod8=3, pads live, TRS 4.29e-14); rung-2 c3500→
**3491 orbit-closed** (job 7878132; independent login-side awk re-verification
matches the in-job gate exactly).

Physics gates (from run_L*/eqp files, band 26/27): rung-1 eqp0 gap **3.5819 eV**
vs rung-0 3.5788 (+3.1 meV), eqp1 3.2516 vs 3.2526 (−1.0 meV) across a DOUBLED
band window and a different centroid set — PASS (J.7 restart-window guard held);
rung-2 eqp0 **3.6290** / eqp1 **3.2895** = +47/+38 meV vs rung 1, the expected
μ-convergence from below (matches the AQ 12×12 pattern). Stage rows (disk):
rung-1 sigma.exec 86.99 s (host_accum 72.71 = 84%), zeta_fit 36.2 s; rung-2
sigma.exec 157.98 s (host_accum 136.02 = 86%), zeta_fit 53.4 s.

Scaling hazards measured (cold collective tables, AN_MU set per rung):
- **ω-cube all-gather 687.87 MB/rank, c128[41,16,256,256]** (`_to_host_np`
  class, module jit__identity_fn) — byte-IDENTICAL between μ=2475 and μ=3491 ⇒
  **nb²-not-μ scaling**: ~2.75 GB/rank at nb=512, ~11 GB at nb=1024. TOP
  σ-side size hazard; removal needs sharded/rank-0 ω-cube consumers (AK.9's
  "own workstream").
- **ζ-apply full-μ gather [1,μ_pad,5760]**, linear in μ: 230.03 MB (pad 2496)
  → 324.40 MB (3520) → 460.06 MB (4992). The colltable FLAGs it by design
  (exit 1 after printing — not a run failure). Otherwise NO (μ,μ) tile
  collective on any rank — scaling doctrine holds.
- **host_accum grows SUPERLINEARLY in μ**: 72.7 s (μ=2475) → 136.0 s (μ=3491),
  ×1.87 for ×1.41 μ, now 86% of sigma.exec — handed to the Σ-perf workstream
  (AY.4's split shows the nb=128 analogue is pure d2h wait; nb=256 split failed,
  AY.9).
- isdf_tensors restart scratch: 3.47 → 6.70 GB, ×1.93 ≈ (3491/2475)² — μ²
  confirmed. sigma_mnk.h5 byte-equal 1.413 GB both rungs (nb/k/ω-determined).

Rung 3 (nb=512) — TRUE NSCF regen with the provenance chain intact:
- qe_b512 job 7878241: NSCF nbnd=512 from the STAGED EXISTING SCF density
  (original out/ untouched); the `cmp RHO` gate diff is exactly the BGW header
  date stamps (payload byte-identical, sizes equal) — **staged-density
  provenance HOLDS**.
- deck_b512 job 7878246: **el_compare gate max |Δ eigenvalue| bands 1..256 =
  1.45e-11 eV** (tol 1e-5) — the b512 NSCF reproduces the 256-band spectrum to
  diagonalization precision; gate_h0 at nb=512 vs kih_b512/vxc_b512 PASS;
  kin_ion_b512/dipole_b512 4× sizes sane.
- kmeans c5000 @ 0:512 (job 7878254): **4951 orbit-closed** (mod8=7) — nearly
  shape-matched to AQ's 4962 at 4× the band window.
- **Rung-3 GW: job 7878263** (`l3_b512_c5000.sbatch` → run_L3_b512_c5000,
  window 26/486/512) submitted; **PENDING behind the dev-queue per-user node
  cap at write time** — verdict belongs to the next section.

### AY.6 — GPU release gate: merged-tree CUDA lib GREEN

rtx-dev, wk_REL/rtx.7877756.out: new stage cell (`stage_ffi_deps.sh` CANNOT run
on login nodes — no containers there; job 7877751 was the pre-staging cell that
fell back to the wt-H lib) → **merged-tree CUDA lib builds OK**
(`lorrax_ffi_merged_cuda/build_phdf5/liblorrax_ffi.so`, cusolverMp present);
4×4 deck: **eqp0 GPU-vs-CPU max|Δ| = 1.00e-09 eV (5160 values), GPU P>1 vs P=1
max|Δ| = 0.00e+00 — bit-exact**; slab_io=auto routes PHDF5_FFI on GPU.
Re-gated GREEN post-audit at src@8487ff8 (rtx.7877790.out, same two PASSes).

### AY.7 — GATES (all cache-cold where a collective table is claimed)

| gate | job(s) | result |
|---|---|---|
| AQ mpi rehearsal ×2 (both tiers forced) | 7877754, 7877789 | **rc=0 554/514 s**; mlx banner; `path=distributed_rank_truncate`; Dyson residual ≤7.502e-15 |
| 785c P=16 audit-verify eqp0 vs run_800c | 7877788 | **PASS — max|Δ|=0.00e+00 eV**, 5160 values (md5 header-only) |
| check_hermitian HLO regression (no full-tile all-gather) | 7877804 | **PASSED** (herm cell, 36 passed) |
| slate-hang bisect (AUDIT vs pre-audit AW lib) | 7877804 | rc=143 on BOTH → pre-existing, not introduced |
| Σ round-1 parity (prod + staged) | 7878038 | **exact 0.0** ×2 on sigma_diag/eqp0/eqp1 |
| Σ round-2 AK.9 stacking parity + HLO rs 4→2 | 7878092 | **exact 0.0**; rs counts/payloads as designed |
| Σ iter-1 parity ×4 (incl. FUSED) | 7878110 | **exact 0.0** ×4 — restart-gated deck reproduces full-run outputs bitwise (AC.4 harness validated) |
| Σ haccum split parity, nb=128 | 7878233 | k128a/b **exact 0.0**; l1a/l1b **FAIL rc=1** (AY.9) |
| rung-1 physics vs rung-0 | 7878104 | **PASS** — gap Δ +3.1/−1.0 meV (eqp0/eqp1) |
| rung-2 physics vs rung-1 | 7878225 | **PASS** — +47/+38 meV μ-convergence from below |
| b512 provenance (el_compare / gate_h0 / RHO) | 7878246, 7878241 | **PASS** — 1.45e-11 eV; GATE PASS; payload byte-identical |
| GPU merged-tree CUDA lib | 7877756, 7877790 | **PASS** — 1.0e-09 vs CPU, 0.0 vs P=1 |
| gloo/ib0 distributed tiers | 7877753, 7877761 | **FAIL rc=1** ×2 (the AY.2 finding, not a regression) |

### AY.8 — files touched

Unusually for the house rule, this session's source edits are **COMMITTED on the
branch** (owner session on the main checkout `/work2/08271/jackmc/frontera/lorrax`,
`fix/zq-band-gather-device-invariance`, 8 commits b3bd130..dc30af4 — see AY.1
table for the per-commit file areas; dc30af4 = `src/gw/ppm_accumulators.py` +
`ppm_sigma.py` + `ppm_tau_kernel.py`, 478 insertions). Branch NOT pushed, NOT
merged. Working tree, NOT committed: `src/gw/ppm_accumulators.py` (the
d2h_wait/omega_project row split + `_project_tau_onto_omega_np` pref-fold
[value-identical, flagged in-code] + astype-drop [bit-exact]).
(`manual/05_isdf/5.1_pair_density_factorization.md` was already modified at
4ebe19e — pre-session, not AY's.) Reverted, archived:
`wk_REL/fused_tau_refuted_2026-07-28.patch`. New artifacts: `wk_REL/`
{audit_findings.json, fixgroups/, sigma_perf_candidates.json,
sigma_perf_results.md, ladder_rung1_notes.md, b256_verify.py,
el_compare_b512.py, verify/bisect/rtx/pytest sbatch+outs};
`mos2_4x4_test/` {aq_rehearsal, sigma_perf_ab, sigma_iter, sigma_haccum,
l1/l2/l3, deck_b256*, qe_b512, deck_b512*}.sbatch + run dirs.

### AY.9 — named, not done

1. **nb≥256 d2h_wait/ω-project attribution**: job 7878233's l1a/l1b passes
   (vs run_L1_b256) died rc=1 in 16-17 s with no sigma_diag.dat and no Python
   frame in the .out — cause undiagnosed. The on-record prediction (d2h_wait
   ≈70 s, omega_project ≈2-4 s at nb=256 ⇒ the nb≥256 σ wall is the DEVICE μ²
   tile, not host math) is UNVERIFIED. Repro: sigma_haccum harness, l1 passes.
2. **Rung-3 verdict**: job 7878263 PENDING at write time; everything upstream
   is green — record its result as a new appended section, not an edit here.
3. **ζ-apply full-μ gather** [1,μ_pad,5760] (linear in μ; 460 MB at μ=4962) —
   known and FLAGged per-rung; judge against the distributed-tier contract at
   12×12 scale (open from the handoff).
4. **ω-cube nb²/rank replication removal** (sharded/rank-0 consumers for head
   injection, diag interpolation, sigma_mnk write) — AK.9 calls it its own
   workstream; ~2.75 GB/rank at nb=512 makes it a rung-3+ blocker candidate.
5. **Owner-gated**: branch push; gloo P≤16 rescope in the certified-stack docs
   (the AY.2 banner is the ledger record); deprecation removals; cublasmp
   deletion; the 4 deferred audit findings.
6. **test_slate_cholesky_trsm_cpu hang** — pre-existing (AY.1), open ledger.
7. *(appended 2026-07-28, wk_REL)* — item 1 is now RESOLVED-VERIFIED (jobs 7878233/7878276: harness link bug diagnosed; d2h_wait 78.5 s vs omega_project 1.18 s at nb=256 — device μ² tile confirmed, host exonerated; commits dc30af4+9e6f7d0) and item 4 now has its handoff design: `wk_REL/DESIGN_MEMO_omega_cube_sharding.md` (two-plan sharded-consumer proposal, consumer inventory, bit-parity gates, cost estimate; L3 evidence: 2751 MB/rank single gather, module_0962, job 7878263).

### AY.addendum — size ladder CLOSED through rung 4; memory model certified; centroid Gram build is the first real wall (2026-07-28, wk_REL ladder agent)

> ⚠ CLAIM-DECAY on the AY headline: "green through rung 2 with the memory
> model exact to 3.5%/0.6%" — now superseded by the FULL ladder: green
> through RUNG 4, model ratios 1.035/1.006/1.063/0.969 (rungs 1-4).

Measured (all disk-verified; jobids inline; runs in mos2_4x4_test/run_L*):
- Rung 3 (512b, mu=4951; job 7878263): rc=0 wall 795 s; predicted 17.51 vs
  measured 18.61 GiB; omega-cube gather 2751.46 MB/rank = exactly 4x the
  256b value (nb^2 confirmed); at matched mu vs AQ (job 7877789, 128b):
  sigma.exec x1.48 for bands x4, zeta cholesky band-independent
  (37.7 -> 36.9 s).
- Rung 4 (512b, mu=6947; job 7878363): rc=0 wall 1217 s; predicted 24.36
  vs measured 23.60 GiB (26% of budget); colltable (mu,mu)+ FLAG clean;
  sigma.tau.host_accum 610.5 s = 89% of sigma.exec — host-accum series
  73/136/329/610 s is SUPERLINEAR in mu (the ladder's emerging time wall;
  handoff = wk_REL/DESIGN_MEMO_omega_cube_sharding.md).
- Deck regen: b256 family (jobs 7878101/7878132), b512 true-NSCF family
  from STAGED SCF density (jobs 7878241/7878246): el_compare gate
  max|d eig| = 1.45e-11 eV over bands 1..256; RHO payload byte-identical
  (7 header-timestamp bytes differ); gate_h0 nb=512 PASS.
- NEW WALL (first genuine ladder death): kmeans c7000 pivoted-Cholesky
  Gram build — pair_density materializes 2x (nk,2,2,M,M) = 2x98 GB at
  M=9786 on a 192 GB node (job 7878309, all 6 N-nudges RESOURCE_EXHAUSTED
  at isdf/core.pair_density via pivoted_cholesky:977). Bridged
  methodology-clean on nvdimm (job 7878358: 6947 orbit-closed, mod8=3).
- GATES: per-rung W-Dyson residual <= 1.7e-14; H0/implied-Vxc in range;
  path=distributed_rank_truncate every run; mlx provider banner every run;
  0 real tracebacks; QP gaps per rung in the frontier ledger
  (wk_REL/SIZE_CAMPAIGN_BRIEF.md). Convergence flag: mu-convergence at
  fixed window is not monotone-from-below on this deck (256b family
  +47 meV 2475->3491; 512b family -240 meV 4951->6947).
- Honesty notes: cross-rung timing deltas span src 8487ff8 -> 9e6f7d0
  (sigma-perf merges landed mid-ladder); sacct MaxRSS undersamples brief
  peaks (rung 3: 5.6 GiB vs 18.6 GiB real) — per-rank /proc VmHWM sampler
  (new in the L* harnesses) is the authoritative instrument; the
  notification channel fabricated hundreds of future-dated job events all
  session — every number here was re-read from disk under an audit-log
  discipline (see ladder notes).

### AY.addendum — files touched (scratch harnesses only; repo untouched by the ladder runs)
mos2_4x4_test/{deck_b256,deck_b256_c3500,deck_b512,deck_b512_c5000,
deck_b512_c7000,qe_b512,l1_b256,l2_b256_c3491,l3_b512_c5000,l4_b512_c7000}
.sbatch + nscf_b512.in + pw2bgw_b512*.in + deck_b256/b512.in;
wk_REL/{b256_verify.py,el_compare_b512.py,ladder_rung1_notes.md}.

### AY.addendum — named, not done
Column-blocked Gram accumulation for pivoted_cholesky (single-device path)
— in progress as R6 in the ladder notes at time of writing; omega-cube
sharded consumers (memo); NSCF b1024 deck; restart-scratch GC policy.

> ⚠ CLAIM-DECAY on "AY.addendum — named, not done": the column-blocked
> Gram fix is now DONE and gate-VERIFIED (2026-07-28 same session): wt-REL
> @ 9e6f7d0 branch wsREL-gramfix, +72 lines in centroid/pivoted_cholesky.py
> (single-device column-blocked build, env LORRAX_GRAM_COL_BLOCK, auto
> block from device budget) + isdf/core.py (gram_q0_from_pair grows a
> default-preserving symmetrize=True kwarg; rectangular blocks skip the
> square-only Hermitian step, the caller applies the identical 0.5(G+G^H)
> once on the assembled square). Gate job 7878488: forced 4-block c2475
> regen is BYTE-IDENTICAL (data rows) to the same-node unblocked control
> and to the original rung-1 set; py_compile PASS. First gate iteration
> (7878470) caught the square-only symmetrization crash; 7878483 was an
> apptainer squashfuse infra failure on c201-030 (excluded). NOT committed
> — orchestrator merges.

## BB — 2026-07-31 dev-queue 4x4 VALIDATION SMOKE of the layering campaign (tree @ 3d98e98 [+3a49da3 template fix]; jobs 7884599, 7884602): densifier probe GREEN, in-tree ONE-cpp-tree FFI build GREEN, template/transport/staging GREEN — and a RELEASE-BLOCKING code defect found: 656abdf deleted _run_sigma_branch's omega_global_idx param but left the caller passing it, so EVERY GW sigma run on today's tree dies (TypeError on all ranks) at the first sigma branch

**One line: the runtime smoke did exactly its job — everything the login
gates could see is green, and the one thing they cannot see (a call-site
kwarg for a deleted parameter) kills every GW run at the top of the sigma
stage; report-and-stop per doctrine, one-line fix named below.**

### BB.1 setup (all paths are evidence)
- Deck: 800c merged geometry, fresh run dirs under `mos2_4x4_test/run_800c_valsmoke_*`,
  inputs copied from the pinned baseline
  `mos2_4x4_test/_archive/2026-07-30/runs/run_800c_merged` (E_F(midgap)=-4.254195 eV,
  Total recorded 82.792 s).
- Launch: **through the vendored template** `config/frontera/templates/gw_dev.sbatch`
  (validates ef7cfa2 vendoring), P=16 / 8 nodes / dev queue.
- Runtime: fresh `--no-compile` bundle `/scratch2/08271/jackmc/lorrax_bundle_3d98e98/lorrax_cpu_bundle.tar`
  (src @ 3d98e98, stripe 12; built because the existing bundle's src snapshot is
  49877c0 = YESTERDAY's tip — running it would have validated nothing).
  Verified post-hoc: bundle `src/gw/ppm_sigma.py` is byte-identical to HEAD.
- FFI host .so: **in-tree rebuild of the ONE C++ tree** (34c5c46) via
  `config/frontera/build_ffi_host.sh` inside the job (37 s, rc=0) →
  `$WORK/lorrax_ffi_unified/build_host_ONE/liblorrax_ffi_host.so`,
  sha256 42242d3348323b4c1a821b24c66b07be7f93bef245e24f1647a27f5e91d74297.
  Exported handler table (23 T/D `*Ffi`/`lrx_*` symbols) **identical** to the
  build_host_DIVAUDIT baseline (f9088e8a…).  The staged
  `$WORK/lorrax_ffi_wtA/build_host` copy (27f95975…, Jul 25) was REJECTED: no
  mklfft handlers, cannot serve LORRAX_FFT_FFI=1.  (The task-named
  `src/ffi/cpp/build_host` did not exist on disk; building it in-repo would
  have dirtied the tree — .gitignore does not cover `build_host/` — so the
  build went to lorrax_ffi_unified like every predecessor.)

### BB.2 job 7884599 (submitted 19:50, ran 19:51–19:54)
| step | verdict | evidence |
|---|---|---|
| pre-A in-tree FFI build | **GREEN** rc=0 | `mos2_4x4_test/valsmoke_ffibuild.7884599.log`; symbol diff vs DIVAUDIT: IDENTICAL |
| pre-B W-densifier HLO probe (`tools/probe_w_densifier_hlo.py`, 2x2 mesh, 4 host devices) | **GREEN** rc=0 | `valsmoke_probe.7884599.log`: output='k' AND 'R': gather-class HLO ops **0**, max\|Δ\| vs eager reference **0.000e+00**, out sharding pinned True → audit P0-4 semantic claim holds compiled |
| main GW, certified weapons (LORRAX_FFT_FFI=1 + _FUSED=1) | **rc=1** | `run_800c_valsmoke_tmpl/gw_dev.7884599.log`: all 16 ranks "Exited with exit code 1", NO traceback |
| transport/stack banners | **GREEN** | libfabric provider: **mlx**; impl=mpi + 3 warmed cliques (112 ms); staged=1 in **1.18 s** (node-local bundle); slab_io=auto → **PHDF5_FFI**; zeta fit + W stage + restart writes all completed (`tmp/isdf_tensors_785.h5` 374 MB, `zeta_q.h5` 251 MB written) |

Death point: immediately after rank-0's "GN invalid modes → static COHSEX"
line, BEFORE the first `_build_windows_for_branch` window print (baseline
prints "ω≥E_F cond window …" next).

### BB.3 job 7884602 A/B/C (ran 19:56–20:10) — defect is dial-independent
| cell | config | rc | log |
|---|---|---|---|
| A | FFT-FFI fully OFF | 1 (same point) | `run_800c_valsmoke_fftoff/gw_dev.7884602.log` |
| B | LORRAX_FFT_FFI=1, FUSED off | 1 (same point) | `run_800c_valsmoke_fusedoff/gw_dev.7884602.log` |
| C | fused repro, per-rank `--output=rank.%t.out` | 1 (same point) | `run_800c_valsmoke_fused_repro/rank.*.out` — **zero error text in any of the 16 per-rank files** |

### BB.4 ROOT CAUSE (static, conclusive)
Commit **656abdf** ("Deprecation deletions … kij_stream") deleted the
keyword-only parameter `omega_global_idx` from `_run_sigma_branch`
(diff hunk @ old line 589) **but left the call site passing it**:

    src/gw/ppm_sigma.py:1030   (HEAD == the bundle the jobs ran)
        branch_tiles, _ = _run_sigma_branch(
            omega_nonneg_ry=br.omega_abs, omega_global_idx=br.omega_idx,  ← TypeError
            ...

→ `TypeError: _run_sigma_branch() got an unexpected keyword argument
'omega_global_idx'` on EVERY rank at the first branch — exactly the observed
death point, in all four runs, independent of every FFT/GEMM dial.  The only
in-tree reference to `omega_global_idx` is this call site.  Fix is one line
(drop the kwarg); NOT hotfixed here per report-and-stop doctrine — the sigma
stage of 656abdf..HEAD is runtime-dead until it lands, and no eqp/h5 parity
numbers exist for the campaign tree.
Why the gates missed it: py_compile/AST layering gates don't check call-site
kwargs against signatures; the commit's own verification note says exactly
that ("py_compile clean … AST suites …").
**Secondary finding (observability):** the fail-fast excepthook's traceback +
"LORRAX FAIL-FAST" banner (runtime/__init__.py:203-227, stderr-only, then
os._exit(1)) did not survive to srun-captured output in ANY of 4 runs — even
with per-task `--output` files — so a guaranteed-TypeError run is
indistinguishable from a silent native exit in the logs.  The hook needs a
stdout echo (stdout lines from the same ranks DID survive) and/or an fsync
before `os._exit`.

### BB.5 login-side verification (no allocation; post-deletion tree)
- Gates: `tests/test_layering.py` **68/68**, `tests/test_crossfile_requests.py`
  **32/32**, `tests/test_env_registry.py` **9/9**.
- `python3 -m compileall` clean over src/{gw,file_io,bse,ffi,common,isdf,runtime}.
- AST scan: zero dangling references to any SYMBOL deleted by 656abdf
  (`_AccumMode`, `_H5Sink`, `_accumulate_kij_stream`,
  `copy_sigma_kij_h5_to_omega_h5`, `_lustre_prestripe`,
  `save_restart_state_per_proc`, `_select_accum_mode`, …); `kij_stream`
  survives only in comments + the intended gw_config refusal.  (Note the
  class of defect that DID slip through is a kwarg name, which no symbol
  scan sees.)
- Template reconciliation vs the archived certified harness
  (`_archive/2026-07-30/harness/gw800_merged.sbatch`): env content equivalent
  (transport/UCX/PMI2/LD_LIBRARY_PATH/taskset/cache); ONE defect found and
  fixed in commit **3a49da3**: the STAGE TABLE capture grep'd
  `/Timing summary/` but gw_jax prints `--- Timing ---` — the anchor never
  matched (inherited verbatim from the harness, same dead line there).

### BB.6 verdicts vs the four caveats
| caveat | verdict |
|---|---|
| (a) densifier zero-all-gather + parity | **CLOSED GREEN** (compiled-HLO probe, BB.2) |
| (b) kij_stream deletion leaves sigma intact | **CLOSED RED** — sigma is BROKEN by the deletion (BB.4); one-line fix named |
| (c) certified stack e2e from vendored template | **PARTIAL GREEN**: template/bundle/transport/FFI-.so/W-stage all green through the template; e2e blocked by (b) |
| (d) eqp parity vs pinned 4x4 baselines | **BLOCKED** by (b) — no numbers; compare harness is staged (`mos2_4x4_test/compare_valsmoke.py`, tol 1e-8 eV vs run_800c_merged) and runs in-job once (b) lands |

Harvest-after-fix: resubmit `mos2_4x4_test/valsmoke_tmpl.sbatch` (it rebuilds
nothing it doesn't need, reuses the bundle+.so above; rebuild the bundle first
if the fix commits, since the bundle pins src) and read
`[VALSMOKE SUMMARY]`/`[compare]` lines from `valsmoke_tmpl.%j.out`.
NOT committed to the repo beyond 3a49da3 (template anchor fix, pathspec-only);
scorecard append only.  Jobs 7884599 + 7884602 (both mine, dev queue, ~4 + ~14 min).

### BB closure (2026-07-31 late) — validation smoke fully green
- Fix a077a6c (drop stale omega_global_idx kwarg; fail-fast echoes to stdout) + rebuilt bundle lorrax_bundle_a077a6c.
- Job 7884609: GW rc=0 in 2:37 on the vendored template; densifier probe 0 gather ops / 0.000e+00 both modes.
- Parity vs run_800c_merged baseline (jobs 7884609 + compare-only 7884612; harness parser fixed for key=value rows):
  eqp0/eqp1/eqp_g0w0/sigma_diag all max|delta| = 0.000e+00; sigma_mnk.h5 max|delta| = 1.137e-13 eV
  (worst dataset sigma_total_kij_ev), tol 1e-8. VERDICT: PASS.
- Notes: login h5tools (1.8) cannot open HDF5-1.14 files — h5 comparisons must run in-container.
  compare_only.sbatch kept beside valsmoke_tmpl.sbatch for reuse.

### BC (2026-07-31 night) — b300 pipeline: 300-band 4x4 deck end-to-end + QP bandstructure
- Chain: NSCF nbnd=300 (7884642, 18s) -> dipole_b300 14s / kin_ion_b300 40s (7884648) -> centroids 2979/3000, rank gate 270/270 (7884654, 67s) -> GW 300b chi/W + 26v/18c sigma window (7884656, GW leg 323s recorded, rc=0) -> htransform DFT+QP + PNG (7884861, 57s/41s, rc=0).
- Defects found+fixed en route: kmeans rank-gate NameError (f96c180), kmeans writer jax.process_index (4829656), htransform silent eqp-file skip -> refusal (f96c180), post-leg SLURM_NTASKS leak into jax world size, missing libfabric in post LD_LIBRARY_PATH, eqp converter 100x ambiguity gate -> 20x.
- Physics: DFT gap 1.72 eV -> G0W0 2.43 eV (VBM at K); QP valence widened [-5.37,0.86] Ry vs DFT [-4.48,0.80].
- Timing flags: zeta-fit = 64% of GW wall at 2979c (cholesky 105.1s + z_q_build 76.6s); chi0_W_probe re-runs chi+W (~9.6s, duplicates the 9.1s real pass); sigma.exec 71.2s is 88% d2h_wait.
- PNG: /scratch2/08271/jackmc/mos2_4x4_test/mos2_4x4_b300_bandstructure.png; per-band files run_b300_ht_{dft,qp}/bandstructure.dat.

## BD (2026-07-31 late night) — non-GW drivers at P=16 (certified 8x2x28 GW geometry) + many-centroid-limit audit: 3/4 drivers speed up, ALL FOUR byte-exact or 1e-14 vs P=1; htransform+kmeans need LORRAX_MPI_FORCE_THREAD_MAIN=1 (missing warm_mesh_cliques call sites — code finding, not patched); kmeans P>1 fixes (2cbd824+d58bad5+4829656) VERIFIED on a real deck for the first time — accepted set sha256-IDENTICAL to P=1

### BD.1 setup
- Jobs: 7884867 (all four drivers, dipole/kin-ion GREEN, htransform/kmeans FAILED), 7884870 (retry of the two failed legs with `LORRAX_MPI_FORCE_THREAD_MAIN=1`, both GREEN + compare). 8 nodes x 2 ranks x 28 threads, srun --mpi=pmi2, impl=mpi collectives, patched MPIwrapper, provider mlx (banner confirmed each step), cache OFF (baseline parity). Harness: /scratch2/08271/jackmc/nongw_p16/{nongw_p16.sbatch,nongw_p16_retry.sbatch,compare_p16.py}; per-step logs step_*.788487{0,..}.log.
- Src = committed-HEAD snapshot @ 273bcbd (/scratch2/08271/jackmc/nongw_p16/tree), NOT the f96c180 bundle src: the kmeans writer-gate fix 4829656 postdates the bundle. Bundle still staged for venv/overlay.
- P=1 baselines reused from tonight's BC chain (jobs 7884648 dipole/kin-ion, 7884654 kmeans, 7884861 htransform). No baseline file was overwritten: fresh dirs p16_ops/, run_b300_ht_dft_p16/, b300_kmeans_p16/, outputs *_p16.*. P=1 not rerun — P=16 walls carry ~4-6 s more startup (srun+stage+distributed init), which only biases AGAINST P=16, and the drivers' own timing tables are quoted alongside.

### BD.2 verdict table (wall = harness step timer; recorded = driver "--- Timing ---" TOTAL/total)
| driver (b300 deck)              | P=1 wall | P=16 wall | speedup (wall) | recorded P=1 -> P=16 | correctness vs P=1                                                | multi-proc status |
|---------------------------------|----------|-----------|----------------|----------------------|-------------------------------------------------------------------|-------------------|
| psp.get_dipole_mtxels           | 14 s     | 18 s      | 0.8x           | n/a -> 5.7 s         | both datasets EXACT (max delta 0)                                  | WORKS unmodified (prepare_mesh warm-up in-driver) |
| gw.kin_ion_io (-n 300 --hartree)| 40 s     | 14 s      | 2.9x           | 24.5 -> 6.7 s (3.7x) | kin_ion EXACT; v_hartree max delta 2.13e-14 (rel ~5e-16, psum order)| WORKS unmodified |
| bandstructure.htransform (dft)  | 57 s     | 27 s      | 2.1x           | 33.7 -> 18.4 s (1.8x)| bandstructure.dat byte-IDENTICAL (all 7 cols exact)                | WORKS only with LORRAX_MPI_FORCE_THREAD_MAIN=1 (BD.3) |
| centroid.kmeans_cli 3000 --orbit| 67 s     | 31 s      | 2.2x           | 55.6 -> 25.1 s (2.2x)| accepted set 2979c sha256-IDENTICAL; rank gate 270/270 PASS        | WORKS only with LORRAX_MPI_FORCE_THREAD_MAIN=1 (BD.3) |
- Scaling anatomy: the k-sweeps are near-ideal (kin-ion vh_matrix_k 8.45->0.80 s, kin_ion_k 8.67->0.67 s at 16 k / 16 ranks); what doesn't scale is replicated setup (kmeans setup.weight 7.1->7.0 s, init 3.3->3.4 s; htransform initialize_wfns 9.8->8.1 s — replicated SVD inside). dipole is too small to amortize the ~8 s distributed startup: its whole P=1 compute is ~10 s. The 56- vs 28-thread baseline asymmetry (P=1 ran 1x56) additionally understates P=16 on the thread-parallel sections.
- kmeans P>1 verdict (supersedes the stale "kmeans is RED at P>1"): first real-deck P>1 evidence for 2cbd824 (process-local centroid-tree mesh) + d58bad5 (padded-Gram active-mask) + 4829656 (writer gate) — all three hold at P=16; Lloyd converged identically (16 steps, movement 0.0), same 375 reps -> 4173 candidates -> same 270 orbits -> byte-identical file. No seeded-reduction-order divergence to explain: outputs are identical.
- htransform at P=16 auto-selected the phdf5 collective MPI-IO read backend; dipole/kin-ion/kmeans stayed on "eager ... host h5py read per rank" even at 16 processes (see BD.4 blockers).

### BD.3 failure + root cause (job 7884867, htransform & kmeans, 16/16 ranks each)
- Traceback (identical class both drivers): `jax.errors.JaxRuntimeError: UNKNOWN: Buffer Definition Event: MPI: Communicator requested from a thread that is not the one MPI was initialized from.` htransform dies at wfn_transforms.py:2138 (block_until_ready in load_centroids_band_chunked, via initialize_wfns -> streaming_galerkin_solve); kmeans at kmeans_isdf.py:741 (first Lloyd collective). Full logs: nongw_p16/step_{htdft,kmeans}.7884867.log.
- ROOT CAUSE (code-class, NOT patched per task contract): neither driver warms its mesh cliques from the main thread. htransform's `_build_mesh_xy` (src/bandstructure/htransform.py:41-62) documents the choice: "NOT prepare_mesh ... Whether this driver should warm is the open warm-up-contract question (numbered request)". centroid.distribution.build_mesh (src/centroid/distribution.py:94) likewise calls bare resolve_mesh. First collective then fires under XLA's PARALLEL ThunkExecutor from a pool worker, and jaxlib's MpiCollectives::CreateCommunicators refuses (MPI_Is_thread_main false) — the exact BSE-class failure warm_mesh_cliques was built for (collectives.py prepare_mesh docstring; 32 refusals at P=16, gate 7881216). kin-ion/dipole survive because they call prepare_mesh (gw/kin_ion_io.py:684; psp/get_dipole_mtxels.py:40 import + call) — their tables show the collective_warmup section.
- Retry lever: `LORRAX_MPI_FORCE_THREAD_MAIN=1`, the documented fallback ("retained only as a fallback and as the positive control in the gates", env_vars.md L357; legal with the patched THREAD_MULTIPLE wrapper). Both legs then rc=0 with byte-identical outputs. The DURABLE fix is one line per driver: route `_build_mesh_xy` and `build_mesh` through `common.collectives.prepare_mesh` — answers htransform's open warm-up-contract question with BD's measurement.

### BD.4 many-centroid-limit audit (read-only; file:line @ 273bcbd) — objects that must fit ONE rank today, smallest structural fix each
| driver | one-rank object (scaling) | where | smallest structural fix |
|---|---|---|---|
| htransform | A = psi@centroids gathered REPLICATED, (nk*nb, ns*N_mu) c128 | htransform.py:262 (device_put rep) | keep A sharded P(None,'y'); feed the Gram-eigh below |
| htransform | dense SVD of A on EVERY rank | htransform.py:265-270 (_svd_replicated; seam noted :251-253) | SVD -> Gram-eigh: eigh of A A^H (nk*nb square, N_mu-free) via ffi.linalg plan; V from A^H U s^-1 sharded |
| htransform | Vh (rank, ns*N_mu) replicated through _trim_and_pad | htransform.py:356-377 | shard Vh on 'y' (only consumer is B = L^-1 Vh, row-parallel) |
| htransform | B_at_mu (rank, ns, N_mu) re-replicated (and ctilde) | htransform.py:626-627 | owner-shard B_at_mu on mu ('y'); ctilde is N_mu-free (keep rep) |
| htransform | bandstructure.dat written by EVERY rank, no gate (race, shared FS) | htransform.py:1437/:1570 | process_rank()==0 writer gate (same idiom as kmeans_cli:588) |
| htransform | (non-mu) L=cholesky(G) all-gathers the (rank,rank) face to each rank | htransform.py:604 (seam comment :597-603) | swap to factor_c_q / distributed FFI cholesky when rank grows |
| kmeans/prune | dense Gram G (M_pad, M_pad) c128 — sharded P('x','y') at P>1 but WHOLE on one device at P=1: the measured ~61.5k-point single-node ceiling owner (60.5 GB at 61.5k) post-active-mask | pivoted_cholesky.py:681 (build), :284-286 (row reshard) | already-distributed at P>1 — the P=1 ceiling retires by running prune at P>1 (now proven, BD.2); for P=1 decks: tile the select the way the col-blocked build already tiles G |
| kmeans/prune | pair tensors P_l/P_r (nk, ns, ns, M, M) — sharded, but per-rank M^2/P; single-device path col-blocked only | pivoted_cholesky.py:800-806 (98 GB EACH at M~9.8k), :877-899 | extend the col-block loop to the multi-device path (bound per-rank transients by budget, not by P) |
| kmeans/prune + htransform | psi-at-centroids (nk, nb, ns, M) sharded on ONE mesh axis only — per-rank ~ nk*nb*ns*M/sqrt(P) | wfn_transforms.py:1933-1934, :2002-2003 (out_Y 'y'-only, out_X 'x'-only) | shard mu over BOTH axes (('x','y') product) with the sharding_fit/pad idiom |
| kmeans/weight | band-range weight + charge density built REPLICATED on the full r-grid (N_r host arrays per rank) | kmeans_cli.py (setup.weight section; 7.0 s unscaled at P=16) | grid-shard the weight sum like the Lloyd loop already shards positions |
| dipole | gathered (nk, 3, nb, nb) replicated on every rank | get_dipole_mtxels.py:912 -> collectives.py:705/:724-726 (gather_indexed_blocks) | owner-sharded write: rank 0 assembles per-k blocks straight into the h5 (or MPI-IO per-k writes); nobody consumes the replica |
| kin-ion | gathered (nk, nb, nb) x2 (kin_ion + v_hartree) replicated | kin_ion_io.py:453, :719-720 -> collectives.py:705 | same owner-sharded write |
| dipole/kin-ion/kmeans | WFN eager read: FULL file h5py-read PER RANK even at P=16 ("eager (auto, 16 processes)") | WfnLoader backend banner, jobs 7884867/7884870 | these drivers build/hold a mesh — let auto pick phdf5 collective read as htransform already does |
| BSE (bse_jax ring/preview path) | _load_ring_subset h5py-reads FULL V_qmunu (nq, mu, mu), W0_qmunu, psi_full on every rank before sharding | bse_io.py:1453,:1455,:1470 | use load_bse_data_from_restart_sharded (exists, imported at bse_ring_comm.py:28) on this path too |
| BSE (matvec) | W sharded P('x','y') — GOOD; psi_c/psi_v mu-sharded on ONE axis ('x' or 'y') — sqrt(P) class | bse_ring_comm.py:156-161 | same both-axes mu-shard as above |
| exciton_bands | per-Q conduction stacks x5/y5: mu on one axis each (sqrt(P)); loader is the sharded one — GOOD | exciton_bands.py:241-243 | same both-axes mu-shard |
- Common thread: nothing except htransform's SVD family is N_mu^2-replicated anymore — the survivors are (a) three replicated SVD-basis objects in htransform, (b) sqrt(P)-only mu-sharding of psi-at-centroids everywhere the loader is used, (c) replicated nb^2 k-gathers in dipole/kin-ion (write-only consumers), (d) per-rank eager WFN/restart reads. All four have named, mechanical fixes; none needs a new algorithm.

### BD.5 named, not done
- The one-line prepare_mesh routing for htransform/_build_mesh_xy and centroid.distribution.build_mesh (BD.3) — src change, owner call, this task was measurement-only.
- htransform Gram-eigh replacement for _svd_replicated + mu-sharded Vh/B_at_mu (BD.4) — the whole many-centroid story for this driver hangs on it.
- gather_k_blocks owner-sharded write mode (BD.4) — retires the last replicated nb^2 object in dipole/kin-ion.
- dipole P=16 anti-scaling: not worth chasing until the deck is big enough that dipole_k > startup (~10x this size); re-measure there before engineering anything.

## BE — htransform XLA compile-count attribution (2026-07-31; jobs 7884866 / 7884869 / 7884871, MoS2 4x4 b300, P=1, 1 node dev)

**The hypothesis under test was jax flow-control inefficiency — Python loops retracing per iteration or per shape. It is REFUTED, by measurement, and the honest headline is that the compile storm was never the wall.**

Method: a wrapper on `jax._src.compiler.backend_compile_and_load` recording each module's MLIR `sym_name`, its operand types, its wall time and the LORRAX stack frames beneath it (`/scratch2/08271/jackmc/mos2_4x4_test/compile_probe.py`; per-compile TSVs at `probe_*.{7884866,7884871}.compiles.tsv`). The probe is itself validated: on unmodified code it reproduces the 7884861 baseline count (137) and its `bandstructure.dat` byte-for-byte.

### BE.1 Attribution of the 137 — 22 real kernels, 115 single-primitive eager modules

| source | kind | n | compile s |
|---|---|---:|---:|
| `bandstructure/htransform.py` | eager 1-op | 38 | 0.76 |
| `common/sanity.py::_finite_stats` (3 `check_finite` gates) | eager 1-op | 35 | 1.04 |
| `common/wfn_transforms.py::gflat_to_rmu` `build()` phase/index precompute | eager 1-op | 24 | 0.52 |
| `psp/get_DFT_mtxels.py::valence_density_from_kpoint` (density-symmetry gate) | eager 1-op | 17 | 0.46 |
| `bandstructure/htransform.py` named jits (`_accum`, `_build`, `_kpath_batch`, …) | REAL KERNEL | 17 | 0.83 |
| `common/wfn_transforms.py` loader kernels (`_kernel`, `to_rchunk fn`, `_reshard_all`, …) | REAL KERNEL | 5 | 0.44 |
| `common/wfn_transforms.py::_slice_bands_gflat` int cast | eager 1-op | 1 | 0.01 |
| **total** | | **137** | **4.07** |

- **Distinct-shape retraces: ZERO. Per-iteration loop traces: ZERO.** The 121-point k-path loop runs four 32-wide batches and compiles `_kpath_batch` ONCE and its `dynamic_slice` ONCE; the band-chunk loop compiles `_accum` ONCE. 7c99337's `band_pad_to` / `nq`-pad work already closed that class — there is nothing left of it to fix.
- Only 14 of 137 are the same `(sym_name, signature)` twice, and most of those are two different call sites with different shardings, not a cache miss.
- The real mechanism is that **every eager `jnp` op is its own XLA module.** `jnp.eye` is six; `_finite_stats`' 13-op reduction is thirteen; `phx[:, r_mu[:,0]]` advanced indexing is nine.

### BE.2 What was fixed (commit ccab276, `src/bandstructure/htransform.py` + `src/common/sanity.py`)

| site | before | after |
|---|---:|---:|
| `sanity.py:150` `_finite_stats_fn` — module-level cached jit, the pattern `_herm_stats` (12 lines below) already used and this sibling had not | 35 | 3 |
| `htransform.py:628` `S = jnp.eye(rank)` -> jitted with explicit `rep` out_sharding | 6 | 1 |
| `htransform.py:733/755` `_f_params_from_energies` -> `_f_params_jit` (static band indices) | 6 | 1 |
| `htransform.py:1247` `_diag_eig_at_gamma` absorbs `f_eps[:,0]` + the four printed 5-element windows | 4 | 0 |
| `htransform.py:1338` `_gamma_rt` absorbs the round-trip residual reduction | 4 | 0 |
| `htransform.py:1331` `q0` as a numpy constant | 2 | 0 |
| `htransform.py:1363` `_prep_kpath` — wrap + pad in one jit | 5 | 1 |
| `htransform.py:870/967` `f_eps.T` moved inside `_build` | 1 | 0 |

### BE.3 Before / after (cold = persistent cache off, both legs identical)

| leg | compiles | compile s | wall s | bandstructure.dat vs 7884861 baseline |
|---|---:|---:|---:|---|
| BEFORE dft (7884861 / probe 7884866) | 137 | 4.07 | 33.7 | (baseline) |
| BEFORE qp  (7884861 / probe 7884866) | 137 | 3.63 | 36.0 | (baseline) |
| AFTER dft cold, cache OFF (7884871) | **82** | 2.78 | 36.1* | **byte-IDENTICAL, max\|delta\| 0.0 over all 7 cols** |
| AFTER qp  cold, cache OFF (7884871) | **82** | 2.98 | 34.4* | **byte-IDENTICAL, max\|delta\| 0.0** |
| AFTER dft cold, cache ON, fresh dir  | 81 | 2.46 | 31.4 | byte-IDENTICAL |
| AFTER dft **WARM**, cache ON         | **0** | 0.00 | **29.4** | byte-IDENTICAL |
| AFTER qp  **WARM**, cache ON         | **2** | 0.13 | **28.7** | byte-IDENTICAL |

\* the cache-OFF legs ran under the attribution probe, which pays a Python traceback per compile; the cache-ON legs ran the driver directly. **Do not read a wall win out of this table.** 55 fewer modules is ~1.3 s of a ~33 s wall, inside run-to-run variance. `h_transform` is 22.6 s of that wall and it is 128 `eigvalsh` of (704, 704) plus the IFFT — not compile.

The QP warm leg's 2 compiles are correct and irreducible: `_fun_jit`/`_dfun_jit` are `static_argnames=('a','n','shift')`, so the QP f-transform parameters are genuinely different kernels.

### BE.4 Parity is proven finer than `bandstructure.dat` can show

`write_bands_to_file` formats at `%.8f`, so file equality only proves 1e-8 Ry. The stronger gate is the log diagnostics, which resolve to 1e-15 and match the 7884861 baseline **exactly**: `ctilde[0]` ortho `1.129e-14`, `max|C C^H - I|` `8.215e-13`, `fH_k` real range `[-4.886e+00, 5.679e-01]`, Gamma round-trip `1.332e-15`, `fH eig last 5` tail `-4.09826606e-15`.

### BE.5 Taken to the measurement and then REVERTED — `wfn_transforms.gflat_to_rmu` (24 -> 1)

Folding that phase/index precompute into one jit works and saves 23 modules, **but it moved the diagnostics at the 1e-15 level** (job 7884869: ctilde ortho `1.129e-14` -> `1.109e-14`, round-trip `1.332e-15` -> `1.776e-15`, `fH_k` max `5.679e-01` -> `5.662e-01`) — fusing `exp(2*pi*i*k*(j/n))` shifts it by a ULP. `bandstructure.dat` still compared equal in that job, but ONLY because of the `%.8f` formatting above; that is not evidence of safety for ISDF/GW/BSE, which share this loader through psi-at-centroids. Reverted. 23 cold modules do not buy an unverifiable numerics change on a shared hot path.

### BE.6 Named, not done
- `psp/get_DFT_mtxels.py::valence_density_from_kpoint` (lines 205-212) — **17 modules, 0.46 s, one `jax.jit` away** (the body is already a pure array expression; `nocc`/`weight`/`spin_degeneracy` are the static args). Left alone only because it is outside this task's stated domain and has three call paths including the distributed sweep. This is the single largest remaining item.
- 12 eager 1-op modules remain in `htransform.py`, all cheap and each with a reason to leave it: `fH_k[0]` (2 — the slice is what keeps the eigvalsh off an all-gather), the `gamma_positions` `norm`+`argmin` fallback (2 — that line is fragile: it runs only because `_clean_label` writes a real `Γ` while the comparison at `initialize_kpath` tests the literal `'Γ'` string, and it happens to return 0 because the pad rows and Γ both have norm 0), `enk_sigma[:,0]` (2), the k-path loop slice (1), three `reshape`s, one `transpose`, one `subtract`.
- Harness: `post_b300.sbatch` and `gw_ht_b300.sbatch`'s `run_ht` block now leave the compile cache **ON** (the `ISDF_JAX_CACHE_DIR=""` opt-out is removed; the default resolves to `$SCRATCH/lorrax_jax_cache/np{P}`). Six other `.sbatch` files in `mos2_4x4_test/` still carry the opt-out and were out of scope: `ops_b300`, `deck_b300`, `deck_complete`, `qe_deck_b300`, `gw400_p4`, `gw800_p16`.
