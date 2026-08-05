# LORRAX shared context brief (read me first — don't re-derive this)

LORRAX = JAX GW-BSE electronic-structure code (ISDF ζ-fit → GN-PPM sigma →
htransform band interpolation → BSE/exciton). Runs multi-process via
`jax.distributed` + a 2D `('x','y')` device mesh (`shard_map`,
`NamedSharding`, `check_rep=False`). FP64/complex128 throughout.

## Machine facts (measured — trust these)
- Frontera CPU nodes (2×28-core Xeon 8280, 192 GB) BEAT the RTX 5000 GPUs ~3×
  for this FP64 workload (RTX FP64 is 1:32). CPU is the production target here.
- Rank/thread: 2 ranks/node × 28 threads (1/NUMA socket), `taskset` pinned.
  XLA:CPU threadpool obeys `XLA_FLAGS=--intra_op_parallelism_threads`, NOT OMP.
- Compile storm anatomy (MEASURED): per-rank compile count is problem-size
  invariant (~138/rank); the big-system "2208 compiles" ≈ 16 ranks × 138.
  Dominant cost = every SPMD rank recompiling the same modules. The shared
  persistent compile cache would fix it but is DISABLED (`ISDF_JAX_CACHE_DIR=""`)
  due to a P>1 cache deadlock (per-rank hit/miss divergence hangs the compile
  barrier). Don't re-diagnose this; it's established.
- `jax.distributed` startup is slow (~1 min+); steps under a shared allocation
  can die `DEADLINE_EXCEEDED` under contention — retry, it's infra.

## Repo map (main checkout: /work2/08271/jackmc/frontera/lorrax, src/ layout)
- `file_io/wfn_loader.py` — THE WFN reader (class WfnLoader). Backends:
  `eager` (h5py + numpy `unfold_psi` per k), `phdf5` (collective MPI-IO FFI,
  GPU + now CPU host lib), `phdf5_host` (h5py union read → same on-device
  unfold kernel as FFI; zero-build CPU fallback). All unfold paths single-source
  the symmetry algebra (`common/symmetry_maps.py`: `unfold_psi`, `trs_augment_U`,
  `tau_phase_row`) — proven bit-identical. Output: `(n_k, nb_padded, ns, ngkmax)`
  c128, band-sharded `P(None,('x','y'),None,None)` typical.
  NOTE: it now carries THREE host-side build paths (`_eager_build`,
  `_eager_build_process_local` [§5b per-rank band shard], `_phdf5_host_build`)
  plus the FFI path — known consolidation candidate.
- `common/psi_G_store.py` — gwjax's ψ streaming: host-resident G-flat tiles,
  `io_callback` slicer inside `lax.scan` + per-iter `all_gather`; FFT box never
  persistently materialized. `common/async_io.py` — single-worker daemon that
  overlaps `WfnLoader.load` with device compute (`AsyncWfnReader`).
- `common/wfn_transforms.py` — `to_rchunk` (G-flat→r-slab, LOCAL FFTs inside one
  shard_map, shape-cached jit), `iter_psi_rchunk_bandwise` (~:1572, htransform's
  SYNCHRONOUS per-band-chunk loader.load→to_rchunk path; now takes `band_pad_to`).
- `bandstructure/htransform.py` — streaming Galerkin (`streaming_galerkin_solve`,
  G-accum loop ~:289 with shape-cached `_accum`), CLI `main` (~:1030).
  Loader is now mesh-aware (ψ band-sharded).
- `gw/gw_jax.py` + `gw/isdf_fitting.py` + `isdf/core.py` — GW driver, ISDF ζ-fit
  (consumes PsiGStore), solver-kind resolvers (`_resolve_solver_kind{,_charge,
  _transverse}` — input keys `distributed_cholesky|distributed_lu`, `auto`
  CPU-safe). `ffi/common/dispatch.py` — eigh/cholesky backend dispatch
  (`auto|off|cusolvermp|slate`).
- `bse/` — Lanczos/Haydock/exciton paths are single-jit and CPU-portable
  (audited). `bse/vq_interp.py` — V_Q interpolation; `build_cq` now returns
  face-sharded `P(None,'x','y')`; `minibz_head_vlr` rank-parallel, single-sourced
  from `gw/coulomb/base.py`. `bse/exciton_bands.py` — finite-Q driver (one
  lax.scan over the Q path).
- `psp/` — DFT/pseudopotential side (run_nscf generates WFN.h5 via
  `file_io/wfn_writer.py`; `qe_save_reader.py` reads QE output).
- `runtime/__init__.py` — `set_default_env()` (before `import jax`) +
  `init_jax_distributed()` + `fallback_to_cpu_if_no_gpu_backend()` (gated on
  `_gpu_is_present()`). 7 CLIs now carry this same header (duplication candidate).
- `ffi/phdf5/cpp/` — collective MPI-IO reader; SHARED-CORE design: same TUs
  compile into CUDA lib and host lib; `LORRAX_FFI_NO_CUDA` switches 3 seams
  (handler binding / index copy-in / staging tail). Host build recipe:
  `config/frontera/build_ffi_host.sh`. Host lib is read-only.

## Branch state — fix/zq-band-gather-device-invariance @ 419f57e
All merged & verified bit-exact: z_q invariance fix (a549471), V_q remat
(b9406cd), §5b process-local load (d45950f), phdf5_host + mesh-aware htransform
+ BSE bootstrap (a7b332b), band-chunk uniform pad −23% (bc58cc1), CPU-safe
linalg dispatch + bootstrap unify (cd96495), phdf5 FFI shared-core (eb2e369),
exciton minibz+sharded C_q (48cbb5e). Full handoff:
`docs/dev/HANDOFF_cpu_frontera_2026-07.md`. Ops playbook:
`/work2/08271/jackmc/frontera/LORRAX_FRONTERA_ADVICE.md` (esp §11).

## Test infra (established — use as-is)
- Login-node apptainer is BLOCKED. Use the shared holder:
  `/scratch2/08271/jackmc/lorrax_setup/alloc_run.sh <N_nodes> <tasks/node>
  <PYTHONPATH_src> <workdir> python -u <script.py | -m module> [args]`
  (never inline `python -c`). It reads the holder job-id from
  `current_holder_jid`, ssh's to the head node, `srun --overlap --jobid` under
  apptainer with the CPU env (JAX_PLATFORMS=cpu, x64, ib0, ISDF_JAX_CACHE_DIR="",
  venv on PATH, coordinator auto). Holder = job in `current_holder_jid` (40 dev
  nodes, 2h). Overflow: self-submit sbatch to `qnormal` (mirror
  `cpumn_a.sbatch`). Dev QOS = 2 jobs/40 nodes total — don't submit a 2nd dev job.
- Fixture: `tests/regression/cohsex_debug/` (WFNsmall.h5, nk=9 full-BZ/4 IBZ,
  nb=150-file/8-window, ngkmax=780, centroids_frac_60.txt, cohsex_test.in).
  htransform ground truth: `/scratch2/08271/jackmc/lorrax_setup/
  bs_groundtruth_meshless.dat` (last col = energy; gate max|Δ|<1e-8).
- Suites: `tests/test_wfn_loader_eager.py` (loader parity),
  `tests/test_zeta_mesh_invariance.py` (multiprocess workers — SLOW, run solo),
  `src/common/wfn_loader_backend_parity_test.py`.
- venv: `/work2/08271/jackmc/frontera/lorrax_env/.venv` (editable install →
  main checkout src; agents override with PYTHONPATH=<worktree>/src).

## Phase-2 outcomes (merged or merging — don't redo)
- F (wfn-read): one shared `load_psi_gflat_padded` helper (wfn_transforms) feeds
  htransform iter / centroid load / PsiGStore populate; htransform does ONE ψ
  window load per run (was n_bc+1). AsyncWfnReader measured overlap=0.000 →
  production-dead (owner call on deletion). PsiGStore pattern NOT applicable to
  htransform (single r-sweep). get_DFT_mtxels reuses the passed loader.
- G (infra): wfn_loader host builds share one `_kplan`+`_assemble_process_local`
  scaffold; `runtime.bootstrap()` = the canonical 2-line CLI header (9 CLIs);
  ffi_loader has public `has_target(target, platform)`/`has_phdf5_read()`;
  isdf/core charge/transverse resolvers consolidated into `_resolve_channel_ladder`.

## The distributed-linalg stack (for workstream I — current state)
Two parallel dispatch mechanisms with one shared vocabulary (auto|off|cusolvermp|
slate[,scalapack]; `auto` CPU-safe everywhere):
1. GW/ISDF side: `isdf/core.py` `_resolve_solver_kind` (+ `_resolve_channel_ladder`
   post-G) → route strings ('cusolvermp_cholesky', 'sharded_cholesky',
   'replicated_rank_truncate', 'lu', 'cusolvermp_lu', slate/scalapack variants) →
   FFI imports inside core. Selected by INPUT-FILE keys `distributed_cholesky`,
   `distributed_lu` (gw_config.py validates → gw_init → isdf_fitting → core).
2. BSE/htransform/exciton side: `ffi/common/dispatch.py::dispatch_eigh` (and
   friends) selected by the `--eigh-backend` CLI FLAG (htransform, exciton_bands
   → vq_interp, bse_setup) — NOT an input-file key (a known gap).
Backends/impls: `ffi/cusolvermp/` (CUDA-only, square-mesh, eigh+cholesky+LU),
SLATE + ScaLAPACK host handlers (CPU, via _HOST_TARGET_SYMBOLS/host lib),
in-tree native fallbacks (sharded_cholesky shard_map impl, replicated_rank_truncate,
jnp.linalg.eigh, per-q LU). Guards: `_mesh_is_cpu`, square-mesh checks
(cusolverMpSyevd DEADLOCKS on rectangular blocks), charge-channel replication cap
(LORRAX_ZETA_REPLICATE_CAP_GIB, default 4 GiB). Capability probing exists for
phdf5 (`has_target`) but NOT yet for linalg (slate-less build + slate request =
call-time failure). Route-pinning test: tests/test_zeta_mesh_invariance.py
worker_cap. Physics warning: `distributed_cholesky=off` silently destroys physics
(ADVICE §6a) — never change default routes.

## Rules
- Physics/numerics must stay bit-exact (or rank-count-invariant where sampling).
- Edit ONLY in your assigned worktree. Do NOT git commit/push — leave changes;
  the orchestrator reviews and merges.
- Log measured wins to /scratch2/08271/jackmc/lorrax_setup/SPEEDUP_SCORECARD.md
  (append under your workstream letter).
- JAX_LOG_COMPILES=1 via a wrapper .py (set os.environ before jax import).
