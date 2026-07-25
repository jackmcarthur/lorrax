# Handoff — CPU adaptation & compile-storm work (branch `fix/zq-band-gather-device-invariance`)

*Last updated 2026-07-25. Companion to the operational playbook
`$WORK/LORRAX_FRONTERA_ADVICE.md` (outside the repo). This file lives in
the repo so it travels with the branch.*

## Goal of this branch
Make LORRAX's GPU-targeted paths run correctly and fast on multi-process
**CPU** nodes (Frontera Xeon, `jax.distributed` + a 2D `('x','y')` device
mesh), kill XLA "compile storms", and unify GPU/CPU device selection —
without changing any physics. Every change below is verified **bit-exact**
(or rank-count-invariant) against the pre-change result.

## Commit map (oldest → newest)
| commit | what |
|---|---|
| `a549471` | **z_q band-gather device-count invariance fix** + hardening (the original bug: band `all_gather` is `bpd_max`-strided but the g-axis mask assumed contiguous → wrong at the short remainder chunk for P>1). Static Y-compaction gather table. |
| `b9406cd` | **V_q remat**: drop the degenerate size-1 q-axis staging in `_make_per_q_kernel`; reshard the real (μ,G) on x/y instead of full rematerialization. |
| `d45950f` | **§5b process-local eager WFN load**: each rank builds only its band shard via `make_array_from_single_device_arrays`; unlocks 40–80 node runs (no rank materializes the full `(n_k,nb,ns,ngk)` host array). |
| `a7b332b` | **phdf5_host CPU WFN backend** + **mesh-aware htransform loader** + **BSE CPU bootstrap**. |
| `bc58cc1` | **htransform band-chunk uniform pad** (compile fix) + latent `UH_bc`/ψ band-dim bug fix. |
| `cd96495` | **CPU-safe switchable linalg dispatch** + **unified jax gpu/cpu bootstrap** across CLIs. |
| `eb2e369` | **phdf5 FFI shared-core CPU enable** (one read path, switch only device-staging). |
| `48cbb5e` | **exciton**: rank-parallel mini-BZ head + sharded `C_q`. |
| `0c25ae7`,`50ba500`,`971baef` | merge commits (linalg, phdf5-ffi, exciton branches). |

## Initiatives, in detail

### 1. WFN read on CPU — two layers, shared unfold
- **`phdf5_host` backend** (`file_io/wfn_loader.py`): host `h5py` union read →
  the **same** on-device vectorised unfold kernel (`_phdf5_unfold_and_shard`)
  the GPU phdf5 FFI uses. Band-sharded; unfold compiles **once** (lru-cached).
  Zero build. Symmetry algebra is single-sourced through `unfold_psi`'s
  primitives (`trs_augment_U`, `tau_phase_row`), so it is *mathematically
  identical* to the eager path — not a parallel reimplementation.
- **phdf5 FFI shared-core CPU** (`ffi/phdf5/cpp/*`, `ffi/common/*`): the
  collective MPI-IO read core compiles from the **same** translation units
  into both the CUDA and a new **host** lib; a single `LORRAX_FFI_NO_CUDA`
  flag switches exactly 3 seams (handler binding; index copy-in; the
  device-staging tail — `cudaMemcpyAsync` H2D vs `std::memcpy`, `cudaMallocHost`
  vs `aligned_alloc`). This is the "share the core, switch the device
  functions" design — **not** a separate CPU reader.
- **Backend auto-pick** (`_auto_pick_backend`): GPU CUDA FFI → else CPU **host
  FFI** (probes the host lib for the phdf5 read symbol via `hasattr`) → else
  the **h5py twin** `phdf5_host` (now a documented no-FFI fallback). Graceful:
  the currently-deployed host lib lacks the FFI handlers, so CPU runs fall back
  to the twin today with **zero regression**; rebuild to enable the FFI read.
- **Mesh-aware htransform loader**: `htransform.setup_wfn_and_sym` now builds
  the mesh up front and passes it to the loader, so ψ is band-sharded per rank
  (phdf5_host on CPU / phdf5 FFI on GPU) instead of replicated on every rank.

**To enable the CPU FFI read** (optional; the twin works without it): build the
host lib with `config/frontera/build_ffi_host.sh --fresh` inside the py312
apptainer container (stages into `$WORK/lorrax_ffi_host` or a dir of your
choice), then point `LORRAX_FFI_SO` at it. The host lib is **read-only**
(`write_ffi.cc` not ported — CPU only needs reads).

### 2. htransform compile storm — anatomy & fix
Measured, not assumed (see `SPEEDUP_SCORECARD.md`):
- Per-k `ngk` is **already** uniform (loader pads to `ngkmax`, sentinel
  `box_index`, zero G-slot) — every heavy kernel compiles once/rank.
- The `~2208` compile figure is **≈16 ranks × ~138 compiles/rank**; per-rank
  count is problem-size-invariant. **The dominant factor is SPMD rank
  replication, not shape variation.**
- The one real residual was the **band-chunk remainder** width (a non-uniform
  last chunk forced `to_rchunk` + `_accum` to recompile). Fixed with a
  **band-axis uniform zero-pad** (`band_pad_to`, mirroring the `ngkmax` pad),
  which also fixed a latent `UH_bc` vs ψ band-dim mismatch. Measured **−23%**
  Galerkin wall on non-uniform chunks; bit-exact.
- **THE big lever (open):** kill the rank replication with a **shared
  persistent compile cache** so ranks 1–15 hit rank-0's modules. Blocked by
  the P>1 cache **deadlock** (see Open Items).

### 3. Switchable linalg + unified bootstrap
- The switchable input-file keys already existed (`distributed_cholesky`,
  `distributed_lu`, `eigh_backend`; all default `auto`, CPU-safe native).
  Fixed the one real CPU hole: `isdf/core._resolve_solver_kind_transverse`
  LU `auto` no longer auto-picks cuSOLVERMp on a CPU mesh.
- Unified the jax bootstrap (`set_default_env()` **before** `import jax`, then
  `init_jax_distributed()` + `fallback_to_cpu_if_no_gpu_backend()`) across
  `bse_jax`, `bse_kpm`, `bse_feast`, `bse_w_exact`, `davidson_absorption`,
  `absorption_haydock`, `htransform`.
- Broadened `fallback_to_cpu_if_no_gpu_backend` to match
  `Unable to initialize backend 'cuda|gpu|rocm'`, **gated on `_gpu_is_present()`**
  so a real GPU-node init failure is re-raised, never masked by a slow CPU run.

### 4. Exciton finite-Q / BSE on CPU
- **Mini-BZ head average** (`bse/vq_interp.minibz_head_vlr`): now single-sources
  GW's physics (`gw/coulomb/base.py`, `gw.vcoul`) and is **rank-parallel** —
  `jax.random.fold_in` on the *global* sample slot makes the result
  **rank-count-invariant** (bit-identical at nproc 1/2/4), each rank doing 1/P
  of the QMC work, combined by `process_allgather`.
- **`build_cq`**: returns the `(μ,ν)`-face-sharded `P(None,'x','y')` device
  array directly instead of a `_to_host` gather that materialized the full
  `(nq,nμ,nμ)` c128 stack (**~13.4 GB/proc** at nq=144, nμ=2412) on every
  process. Consumers (`run_gates`, `prepare_coarse`) reshard/gather on device.

## Verification (all bit-exact / rank-invariant)
- z_q invariance test (P=1,2,3,4,6 incl 2×3 remainder): passes < 1e-9.
- **Merged** htransform end-to-end (4-rank 2×2 CPU, real-symmetry WFNsmall):
  `max|Δ energy| = 0.000e+00` vs the pre-change baseline.
- phdf5 host FFI: CPU multi-rank read bit-exact vs eager (k=ibz & full_bz).
- exciton: minibz head bit-identical nproc 1/2/4; `build_cq` sharded spec +
  value vs numpy 1.7e-16; consumer path vs numpy eigh 5.7e-14.
- linalg dispatch `verify.py`: PASS (LU auto→lu on CPU, cusolvermp cleanly
  rejected, native eigh err 1.3e-15, 9 CLIs import clean on CPU).

## Open items (ranked by leverage)
1. **The compile-cache deadlock** — the highest-value remaining storm work.
   Enabling a shared `jax_compilation_cache_dir` would cut ~138/rank × 16 down
   toward 138 total, but the P>1 cache deadlock (per-rank cache diverges on
   hit/miss → cross-process compile barrier hangs; see FRONTERA_ADVICE §4)
   forces `ISDF_JAX_CACHE_DIR=""`. Fixing that unlocks the real 158s win.
2. **Rebuild & deploy the host FFI lib** so CPU uses the collective read
   (currently falls back to the h5py twin).
3. **Port `write_ffi.cc`** to the host lib if CPU writes are ever needed.
4. Full `tests/test_zeta_mesh_invariance.py` green-run on CPU (logic confirmed
   by `verify.py`; the multiprocess-worker pytest is slow under shared-alloc
   contention).

The distributed-linalg stack (dispatch, backends, config keys, guards) is now
unified behind the `ffi.linalg` facade — see `docs/dev/linalg_ffi.md` for the
architecture, the per-backend constraints, and the sharp edges (square-mesh
deadlock, the `distributed_cholesky=off` physics warning, replication cap).

## How this branch was built (for the next agent)
Parallel work used **git worktrees** under `/work2/.../frontera/wt-{A,BC,D,E}`
(one branch each off `a7b332b`), a **shared 40-node dev holder** + the
`alloc_run.sh` runner (see FRONTERA_ADVICE §11), then a controlled 4-way merge.
</content>
