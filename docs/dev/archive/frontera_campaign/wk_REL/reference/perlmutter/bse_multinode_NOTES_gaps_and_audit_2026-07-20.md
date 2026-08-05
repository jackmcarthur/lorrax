# Multi-node exciton pipeline — per-gap fixes + linalg audit (working notes)

Branch `agent/bse-multinode` off `agent/bse-integration`, worktree
`sources/worktrees/lorrax_A_bse_integration`.

## Files changed (per gap)

### Gap 1 — distributed init (single-sourced)
`src/bse/exciton_bands.py` top-of-module: added the SAME three-call LORRAX
bootstrap gw.gw_jax uses — `runtime.set_default_env()` BEFORE `import jax`,
then `runtime.init_jax_distributed()` + `runtime.fallback_to_cpu_if_no_gpu_backend()`
after. `runtime/__init__.py` is the canonical single source (its docstring:
"Previously five different modules had their own copies ... This module owns
all three"). NOTE: the brief pointed at `cusolvermp_eigh_test._maybe_init_jax_distributed`,
but that is itself one of those older copies; reusing `runtime` is the stronger
single-source (no new/duplicated helper). Idempotent + no-op at proc_count<=1,
so the 1-GPU path is byte-unchanged.

### Gap 2 — multi-node launch harness
`runs/.../11_.../run11.sh` (new): `-N4 -n16 --gres=gpu:4 select_gpu.sh
<shifter> in_container.sh` = 16 procs, 1 GPU each (CUDA_VISIBLE_DEVICES=
$SLURM_LOCALID). This is the PROVEN gw.gw_jax 16-GPU layout
(runs/VI3/04_gw_6x6_600b_2026-06-17/run_vi3_lorrax.sh line 25), NOT
`--gpus-per-task=1` (which ffi/common/cpp/run_shifter.sh documents as breaking
JAX distributed topology sync). No `--mpi` flag: JAX uses SLURM env, not PMI.
`run_exciton_16gpu.sh` (new) drives OFF/ON with `--eigh-backend cusolvermp`.

### Gap 3 — rank-0 I/O guards
`src/bse/exciton_bands.py`: added `_rank0`/`log()` (rank-0 print), wrapped the
`.dat` write + matplotlib + savefig + summary prints in `if _rank0:`, threaded
`log` as `log_fn` into initialize_wfns / compute_wfns_fi / gate. Non-I/O host
numpy (Q path, k-roll, mini-BZ head QMC) runs redundantly on all procs
(deterministic). Added a `[dist]` device_count/process_count/mesh banner.

### Gap 4 — host<->device sharded placement (the subtle correctness point)
Probe (`probe_dist.py`) validated on 16 GPU: `jax.device_put(host_numpy,
sharded NamedSharding)` IS multi-process correct in JAX 25.04 (each proc places
only its addressable shard) — so the loader (already `make_array_from_process_local_data`),
`prepare_coarse` numpy->sharded device_puts, `compute_wfns_fi`, and the V_stack
build need NO placement change. The REVERSE (device->host gather) was the gap:
  * `src/bse/exciton_bands.py` `_gather_host(x)` (new) — sharding-aware: fully
    addressable -> `device_get`; process-spanning -> `process_allgather`. Used
    for the on-grid gate (`psi_cQ_X`/`psi_c_X` are μ-sharded P(...,'x')) and the
    evs gather. (Plain `process_allgather(tiled=True)` DUPLICATES a
    fully-addressable/replicated array's leading axis 16x — must branch.)
  * `src/bse/vq_interp.py` `_to_host(x)` (new, same logic) — `prepare_coarse`
    gathers the q-sharded `qb3` S_b/V_b/F_b tiles (out_shardings P(('x','y'),..))
    that span processes.
  * valence pad-ε guard: done ON DEVICE (jnp.where) instead of
    device_get->jnp.asarray (which both failed on a process-spanning shard and
    dropped sharding).

### Gap 5 — head correction serial (per-Q, cheap)
`minibz_head_vlr` is pure host numpy with DETERMINISTIC Sobol QMC
(seed_offset=0) → identical `(gstar, head_val)` on every process. Computed
redundantly on all procs (no change needed). Confirmed.

### Bonus fix (blocking, distributed) — band divisibility
`src/common/wfn_transforms.py` `gflat_to_rmu`: the htransform galerkin entry
(streaming_galerkin_solve, restart=false) passes an un-rounded band window
(nb=40) that the GW path would pre-round via Meta._round_up(world_size); on a
16-device mesh 40 % 16 != 0 aborted `load_centroids_band_chunked`. Fix: pad the
band axis up to a multiple of mesh.size with ZERO bands (ψ=0 → zero centroid
samples, trimmed from the output). Byte-identical when nb % mesh.size == 0
(single-node / 4-GPU). Only `gflat_to_rmu` needed it (the loader returns
replicated psi here, so `to_rchunk` in the G-accum stage is fine).

## Item 6 — GEMM audit (cublasmp vs native dot_general)
RECOMMENDATION: keep the block-Lanczos matvec GEMMs as native sharded
`dot_general`; FFI eigh only (as done). Reasons:
  * `build_bse_stack_matvec` GEMMs are STRUCTURED multi-index einsums over the
    pair basis (`kvsN,cvk->cksN`; `kctM,cksN->MNtsk`; exchange `kcvN,bcvk->bN`
    etc.), batched over the Lanczos block dim `b`, with `lax.psum_scatter`
    reduce-scatter for the k/μ/ν reductions — already the right distributed
    primitive.
  * `ffi/cublasmp/batched.py:batched_distributed_gemm` handles STACKS of plain
    2-D dense matmuls C[q]=A[q]@B[q], each `P(None,'x','y')` — the W-solve V@χ
    shape, NOT the matvec's fused contractions. Routing the matvec through it
    would require reshaping to 2-D GEMMs (losing fusion) and one host-dispatched
    FFI call per Lanczos iteration — breaking the single-compile `lax.scan`
    (the whole point of build_path_solver). The dense eigh in `prepare_coarse`
    (C_q, n_μ²=640²) is the correct FFI seam and is where cusolverMp is applied.
