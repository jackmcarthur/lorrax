# LORRAX environment variables — the registry

*Every environment variable LORRAX reads, where it is read, what it
defaults to, and whether it should still be an env var at all.*

First generated 2026-07-25 from an audit of every `os.environ` /
`os.getenv` / `getenv()` read under `src/` (Python **and** C++);
reorganized 2026-07-27 (workstream AV) around the rule the campaign keeps
re-learning (QUALITY_PATTERNS #8):

> **Environment may grant capability; it must not silently select
> policy.**  Physics- and routing-relevant choices change only via
> declared inputs (the input file), where they are parsed, validated,
> echoed into the run log, and captured by provenance.  Env vars are for
> *machine facts* (where a library lives, which fabric, how many ranks)
> and for *debug switches*.  Where an env var still overrides an input
> key, that override is DEPRECATED and prints loudly.

## How to read this page

Three classes, in three sections:

| class | meaning | section |
|---|---|---|
| **input-file keys (env twins deprecated)** | policy: physics / numerics / routing. The key is the record; any env twin still honoured prints a deprecation notice. | §1 |
| **machine-capability env** | runtime machine/library facts and resource caps. Legitimately env vars. May affect perf, never physics. | §2 |
| **debug / diagnostic env** | printing, dumping, timing, test hooks. Never changes numerics. | §3 |

Build-time-only variables (§4) and external variables LORRAX sets or
depends on (§5) close the page.  The **Measured scope** column records
the conditions under which a default was chosen (QUALITY_PATTERNS #9:
every performance claim carries its measured domain) — an empty cell
means the default is a design constant, not a measurement.

---

## 1. Input-file keys — and the deprecated env twins

These change the numbers a run produces.  Their home is `cohsex.in`.

### 1a. Env twins that still work (deprecated, loud)

The SC family was promoted first (`gw_config.py` `_sc_env` pattern); the
ζ conditioning pair was promoted by workstream AV (2026-07-27): the key
is the source of truth, a non-empty env var still overrides, and the
override prints a rank-0 deprecation notice (once per process,
`isdf/core.py::_deprecated_env_float`).  ζ-fit provenance records the
EFFECTIVE (post-override) value, so a rerun that drops the env cannot
silently reuse a ζ fit at a different conditioning cutoff.

| env twin (deprecated) | input key | default | effect |
|---|---|---|---|
| `LORRAX_ZETA_RCOND` | `zeta_rcond` | `1e-8` | Relative eigenvalue cutoff of the rank-revealing ζ pseudo-inverse (both the replicated and the distributed `rank_truncate` factor, `isdf/core.py`). THE conditioning knob of the μ ladder: at μ=1998 it alone decides `n_keep ≈ 1015/1998` with no spectral gap (scorecard K). |
| `LORRAX_ZETA_RIDGE` | `zeta_ridge` | `0` | Additive Tikhonov ridge on `C_q` before the ζ factorization. Superseded by `rank_truncate` as the conditioning cure; kept for the cholesky route. |
| `LORRAX_SC_MAX_ITER` | `sc_max_iter` | — | self-consistency loop cap |
| `LORRAX_SC_TOL_EV` | `sc_tol_ev` | — | SC convergence tol |
| `LORRAX_SC_ACCEL` | `sc_accelerator` | — | SC accelerator choice |
| `LORRAX_SC_DEPTH` | `sc_history_depth` | — | SC history depth |
| `LORRAX_SC_MIXING` | `sc_mixing` | — | SC mixing |
| `LORRAX_SC_DUMP_DIR` | `sc_dump_dir` | — | SC dump dir |

No live harness depends on the ζ env twins (audited 2026-07-27: the one
historical user is the completed one-off
`run_B_c1998_rcond10/run72.sbatch`).  This deprecation pattern is the one
to copy for any future promotion.

**Stale doc reference:** `docs/theory/isdf-zeta-vq.md:479` mentions
`LORRAX_GFLAT_CHUNK_SIZE`.  Nothing reads it — the knob is the input key
`gflat_chunk_size`.

### 1b. Promotion candidates (physics/routing knobs still living in env)

Ranked.  These escape input-file validation, the run log, and provenance.

1. **`LORRAX_ZETA_REPLICATE_CAP_GIB` → a memory-section key.**  Byte cap
   (GiB, default `4`; production raises to 16) on the `(nq, μ, μ)` c128
   stack below which the charge factor is fully **replicated** and
   therefore mesh-invariant (`isdf/core.py`).  Above the cap
   `rank_truncate` REFUSES rather than downgrading.  It gates *which
   route carried the conditioning cure*, so "which route did this run
   take" currently depends on an env var.  Raising it is already a
   deliberate, logged act — which is why it ranks below the (now-done)
   rcond/ridge promotion.
2. `ISDF_CHUNK_TARGET_UTILIZATION` / `ISDF_ZCT_STAGE_CAP_GB` /
   `ISDF_ZCT_STAGE_CAP_FRAC` → memory-section keys, for symmetry with
   `band_chunk_size` / `r_chunk_size` / `gflat_chunk_size`, which are
   already keys (planner dials in `gw_config.py`: utilization target
   clamped to [0.85, 1.0]; absolute/fractional caps on the ζ-contraction
   stage transient).
3. `LORRAX_WFN_BACKEND` (`""` → config/auto; forces `eager` \| `phdf5` \|
   `phdf5_host`, `file_io/wfn_loader.py`) → the `slab_io`/backend config
   section, so the read path is recorded alongside the write path.

**Do NOT promote:** anything in §3 (debug), §4 (build), or the compile
cache (§2 — a machine fact; its mandatory-`""` status during regression
timing belongs in the job script, not the physics input).

---

## 2. Machine-capability env — runtime

Machine and library facts, transport/placement, and resource caps.
Legitimately env vars.  None of these may change physics; several change
wall time, and those carry their measured scope.

### 2a. Transport / placement / distributed

| var | default | effect | measured scope |
|---|---|---|---|
| `LORRAX_GLOO_IFNAME` | unset (→ auto-detect) | Which NIC carries JAX's **Gloo CPU collectives** (`runtime/__init__.py:pin_gloo_interface`, from `bootstrap()`).  Unset: auto-detect the first UP `ib*`/`hsn*` IPv4 interface and re-register the CPU backend factory; jax's own default binds the coordinator-route NIC — on Frontera the **1 GbE `em1`**.  A name forces that interface (skipped loudly if not UP with an IPv4); `off`/`none`/`0` disables.  Decision announced on rank 0; every failure path degrades to stock transport with a printed reason.  No-op for P=1, for `JAX_PLATFORMS != cpu`, and under `cpu_collectives_implementation=mpi` (announced as a no-op there). | ib0 vs em1: 3.3× whole-pipeline at 785c/P=16, bit-identical outputs (AK.10/AL); 606c/P=80 e2e 544 s (AL). |
| `LORRAX_COLLECTIVE_CHUNK_MB` | `128` | Upper bound on ONE emitted collective's payload in the ζ `distributed` tier (C⁺ formation + back-solve GEMM, `isdf/core.py`) and the distributed W Dyson A-build (`gw/w_isdf.py`), enforced as a host-level q-block loop XLA cannot fuse back (AF).  `0`/negative = unbounded — reproduction escape hatch only.  A per-instruction TRANSPORT cap, deliberately orthogonal to the LIVE-bytes memory cap `LORRAX_ZETA_GATHER_CAP_GIB`. | Bracketed at P=144 on **em1-era Gloo**: 1.15 GB single-shot AllGather fatal, 0.104 GB good (AF).  Re-priced 2026-07-27 on ib0-Gloo and impl=mpi at 785c/P=16, caps {64,128,256,512,∞} × payloads ≤369 MB: cap indistinguishable from unbounded (chunk-loop overhead ≤ noise floor; scorecard AV matrix).  Default stays 128 = safe everywhere measured; on ib0/mlx it is not known to be *necessary* below ~370 MB payloads, and the em1 fatality point no longer applies. |
| `LORRAX_ZETA_GATHER_CAP_GIB` | `4` | Byte budget for the ζ back-solve's replicated-factor all-gather transient (LIVE bytes; also bounds the distributed tier's eager q-batch, `isdf/core.py`). | |
| `LORRAX_ZETA_REPLICATE_CAP_GIB` | `4` | See §1b(1) — promotion candidate. | production 12×12 runs raise to 16 (overnight A/B) |
| `LORRAX_MAX_RCHUNKS` | unset | Ceiling on the r-chunk count the planner may pick (`gw/isdf_fitting.py`).  Memory/perf, not numerics — but chunking is load-bearing at large μ (scale ladder). | |
| `LORRAX_GALERKIN_CHUNK_GIB` | `6` | htransform's Galerkin accumulation chunk budget (`bandstructure/htransform.py`).  Perf only. | |
| `LORRAX_FAILFAST` | `1` (on) | CLI failure propagation (`runtime/__init__.py::bootstrap`): an uncaught exception aborts the whole step instead of leaving P−1 ranks hanging (QUALITY_PATTERNS #7).  `0` disables.  `SystemExit(0)` (e.g. `LORRAX_EXIT_AFTER_ZETA`) stays a clean exit. | |
| `LORRAX_MALLOC_TUNE` / `LORRAX_MALLOC_MMAP_MB` (`1`) / `LORRAX_MALLOC_TRIM_MB` (`128`) | on | glibc malloc tuning at bootstrap — the arena-retention cure (scorecard T).  `LORRAX_MALLOC_TUNE=0` disables. | RSS ramp root-caused + cured at 12×12/P=80 (T.2) |
| `LORRAX_MALLOC_TRIM` | `1` (on) | Per-r-chunk `malloc_trim` in the ζ-fit loop (`gw/isdf_fitting.py`); glibc-only, no-op elsewhere. | same (T.2) |

### 2b. Libraries, backends, caches, I/O placement

| var | default | effect |
|---|---|---|
| `LORRAX_FFI_SO` | in-tree `src/ffi/common/cpp/build/liblorrax_ffi.so` | Path to the **CUDA** FFI library (`ffi/common/ffi_loader.py`). |
| `LORRAX_FFI_HOST_SO` | in-tree `.../host/build/liblorrax_ffi_host.so` | Path to the **host** FFI library.  Point at `$WORK/lorrax_ffi_unified/build_host*/…` for the SLATE+ScaLAPACK build. |
| `LORRAX_WFN_BACKEND` | `""` (→ config/auto) | See §1b(3) — promotion candidate. |
| `ISDF_JAX_CACHE_DIR` | `$XDG_CACHE_HOME/isdf_jax_compilation` | JAX persistent compile-cache dir (`common/jax_compile_cache.py`); `""` opts out entirely.  One shared `{base}/np{P}/` per world size; ON at every P since AH (process-invariant key + coordination-service hit/miss agreement replaced the AG refusal — the old per-rank layout + jax's process-0-only write guaranteed divergence and, on GPU, a collective-compile deadlock). |
| `LORRAX_JAX_CACHE_MULTIPROCESS` | `1` (on) | `0` restores the scorecard-AG refusal (no persistent cache at P>1) for bisection. |
| `LORRAX_JAX_CACHE_INVARIANT_KEY` | `1` (on, P>1) | Process-invariant persistent-cache key.  `0` does not merely lose the win — it switches the cache OFF at P>1 rather than let rank 0 hit alone (measured 7/7 vs 0/7 divergence). |
| `LORRAX_JAX_CACHE_AGREE_TIMEOUT_S` | `300` | P>1 hit/miss agreement timeout; on expiry degrades to cache-off with a printed reason, never hangs. |
| `LORRAX_JAX_CACHE_STRICT` | `1` | Abort loudly when an agreed entry cannot load on this rank (divergence hazard).  `0` = warn (**unsafe on GPU: can hang**). |
| `LORRAX_JAX_CACHE_PREFETCH` / `_THREADS` | `1` (P>1) / `16` | Page-cache prefetch of agreed entries (measured 29 s serial Lustre latency at 606c/P=16 without it — cold-read cache is a net loss without prefetch). |
| `LORRAX_MINIMAX_CACHE_DIR` | package dir | Minimax grid table cache (`gw/minimax_screening.py`). |
| `LORRAX_DISABLE_MINIMAX_DISK_CACHE` | `""` (off) | Disables that disk cache. |
| `LORRAX_PHDF5_STRIPE_COUNT` / `LORRAX_PHDF5_STRIPE_SIZE_FS` | `16` / `4M` | Lustre striping applied before `H5Fcreate` (3 sites each, consistent). |
| `LORRAX_PHDF5_MPI_STACK` | `mpich` | Build+launch: which MPI the phdf5 FFI links/loads (`run_shifter.sh`). |
| `LORRAX_PHDF5_ALIGN_MB` | `4` | C++: `H5Pset_alignment` threshold, MiB; `0` disables. |
| `LORRAX_PHDF5_COLL_META` | `0` | C++: `1` re-enables collective metadata ops (default off is faster). |
| `LORRAX_PHDF5_CB_NODES` / `_CB_PER_NODE` / `_CB_BUFFER_SIZE` / `_CB_WRITE` / `_DS_WRITE` | MPI-IO defaults | C++: ROMIO collective-buffering hints. |
| `LORRAX_PHDF5_INDEPENDENT` | off | Independent instead of collective MPI-IO (host slab writer). |
| `LORRAX_FORCE_REFIT` | `""` (off) | `1` forces the ζ fit even when `tmp/zeta_q.h5` is complete and its provenance matches (`gw_init.py`).  Ops escape hatch for the ζ-reuse cache (the reuse gate itself is provenance-checked). |
| `LORRAX_ALLOW_PARTIAL_ZETA` | `0` | Permits reading a ζ file whose `zeta_is_done` flag is unset (`file_io/zeta_loader.py`).  Forensics-only: a half-written ζ is otherwise indistinguishable from a complete one (QUALITY_PATTERNS #7). |

## 3. Debug / diagnostic env

None of these change results.  The print knobs that are ON by default
are on because their absence has already cost 72-node hours (the AC/AF.4c
observability failures) — cheap keeps, not clutter.

### 3a. Always-on telemetry (set to `0` to silence)

| var | default | effect |
|---|---|---|
| `LORRAX_ZETA_RANK_LOG` | `1` (on) | Per-q `n_keep / λ_max / λ_min(kept)` from the rank truncation (`isdf/core.py`, both tiers).  **The conditioning signal of the μ ladder** — leave it on. |
| `LORRAX_COLLECTIVE_CHUNK_LOG` | `1` (on) | One line per ζ-tier/W call site naming the emitted per-collective payload, q-block and cap (`isdf/core.py::_chunk_log`).  A tier that silently stopped chunking would otherwise be invisible until it took a 72-node job down again. |
| `LORRAX_RESTART_WRITE_LOG` | `1` (on) | One rank-0 line per dataset in `write_restart_state_to_h5` (name, shape, GB, s, MB/s) + an explicit `W0_qmunu placeholder ALLOCATED` line (`file_io/tagged_arrays.py`).  The instrument whose absence made AF.4c's silent 2 h 55 m undiagnosable. |
| `LORRAX_PHDF5_CLOSE_VERBOSE` | `1` | Chatter on phdf5 close (`file_io/_slab_io_ffi.py`). |
| `LORRAX_SANITY` | unset = warn | Stage-boundary invariant checks (`common/sanity.py`): `0`/`off` skips all (escape hatch); default checks and **warns loudly but keeps running** (a false positive must never kill a 40-node job); `strict` raises `SanityError` — set in CI / regression gates. |

### 3b. Opt-in probes, dumps, test hooks

| var | default | effect |
|---|---|---|
| `LORRAX_TIMING_TRACE` / `_DEPTH` | `0` / `3` | Every `timing.section` announces entry/exit on rank 0, nesting capped at `_DEPTH` (`common/timing.py`). |
| `LORRAX_MEM_DEBUG` | unset | Memory high-water probes at named sites (`gw_init.py`, `isdf_fitting.py` ×3; presence-test at all 4). |
| `LORRAX_RCHUNK_DEBUG` | unset | Per-r-chunk shape/timing/RSS lines (`isdf_fitting.py`, `isdf/core.py`) — the instrument behind every per-chunk table on the scorecard. |
| `LORRAX_EXIT_AFTER_ZETA` | unset | Clean `SystemExit(0)` right after the ζ fit (`gw_init.py`).  Combine with `LORRAX_MAX_RCHUNKS` + `LORRAX_RCHUNK_DEBUG` for fast fit-only sweeps (the AV chunk matrix ran exactly this way). |
| `LORRAX_W_RESIDUAL_CHECK` | `0` | Prints the direct Dyson residual `‖(1−Vχ)W − V‖/‖V‖` on the first few q after a `w_dyson_solver = distributed` W solve (`gw/w_isdf.py`) — the strict numerical contract of the distributed plan (block-cyclic LU is not bit-comparable to the local per-q LU).  Adds one diagnostic jit; leave OFF when taking collective-table probes. |
| `LORRAX_CHECK_REPLICA` | `0` | Re-arms the replica-consistency `assert_equal` that `common/collectives.py::device_put_process_local` suppresses (JAX's hidden `process_allgather`, AA.1).  Debug-only: O(P × tensor). |
| `LORRAX_SKIP_VQ_GATES` | `0` | Skips the V_Q interpolation self-checks (`bse/vq_interp.py`).  The gates exist because V_Q interpolation errors are silent. |
| `LORRAX_TRS_CHECK` | `1` (on) | Load-time MEASUREMENT of the WFN's symmetries against the density built from its own occupied ψ (`common/density_symmetry_check.py`, called from `wfn_loader`).  `0` restores pre-2026-07 flags-only behaviour — and re-arms the scorecard-Q class of silent time-reversal; `strict` raises instead of warning.  A diagnostic whose measured verdict gates `SymMaps` TRS augmentation — default on. |
| `LORRAX_TRS_TOL` / `LORRAX_TRS_SPATIAL_TOL` / `LORRAX_TRS_MAX_K` | `1e-6` / `1e-4` / `32` | Tolerances / k-sample cap for that check.  Fixture residuals ≤ 1e-12; real magnetization is O(1e-2…1); the sample is ±-closed, so `MAX_K` is a sufficiency choice, not an approximation (`0` = all k). |
| `LORRAX_FORCE_FULL_BZ` | `0` (5 sites, consistent) | `1` disables IBZ-only ζ writes / bypasses the IBZ cascade (`gw_init.py` ×3, `screening.py`, `v_q_g_flat.py`).  Debug/test bypass: changes work done and bytes written, not physics. |
| `LORRAX_EXTRA_MU_PAD` | `""` (→ 0) | **Test-only** extra μ-pad rows to prove pad-extent invariance (`runtime/padding.py`).  Any result that moves under this at fixed P is a defect.  NEVER set in production. |
| `LORRAX_PER_PROC_RESTART` | `0` (off) | Per-rank restart shard dump (`file_io/tagged_arrays.py::save_restart_state_per_proc`).  No in-tree reader; measured 4 m 43 s + 72 GB at 12×12/c2406/P=144 immediately after the canonical collective write of the same data finished in 2.2 s.  Forensics-only; **candidate for deletion** next time `file_io/` is open (owner: I/O sibling). |
| `LORRAX_FFI_DEBUG_SHARDS` / `LORRAX_WRITE_NO_JIT` | unset | Slab-writer shard-descriptor dumps / un-jitted write path (`_slab_io_ffi.py`). |
| `LORRAX_FFI_PROFILE` | off | C++: per-call FFI timing. |
| `LORRAX_PHDF5_TIME` / `_WRITE_DEBUG` / `_DUMP_HINTS` / `_SKIP_DESTROY` / `_NO_COLL_META` | off | C++-side phdf5 diagnostics. |
| `LORRAX_LU_DEBUG` / `LORRAX_LU_NO_PIVOT` / `LORRAX_LU_DEBUG_DUMP` | off | cuSOLVERMp LU diagnostics. |
| `LORRAX_JAX_CACHE_EXPLAIN` | `0` | Turns on `jax_explain_cache_misses` logging. |
| `LORRAX_JAX_CACHE_FORCE_DIVERGE` / `LORRAX_JAX_CACHE_NO_AGREE` | `0` | Compile-cache TEST HOOKS: positive control / the deadlock reproducer.  Never in production. |
| `KP2_DEBUG` / `STERN_DEBUG` | unset / `0` | Sternheimer debug (`psp/run_sternheimer.py`, `solvers/sternheimer_solve.py`). |
| `PF_ARTIFACTS_DIR` / `ISDF_JAX_PROFILE_DIR` | `profile` / unset | Trace output dirs (`common/jax_profile.py`, `bse/test_bse.py`). |
| `ISDF_COHSEX_TEST_PLATFORM` | `auto` | Test harness: force `cpu`/`gpu` for the e2e gates (`tests/harness.py`). |

## 4. Build-time only

Read by `config/**/*.sh`, `src/ffi/common/cpp/**/build*.sh` and CMake.
Never by the running Python; setting them in a job script does nothing.

`LORRAX_ROOT`, `LORRAX_SRC`, `LORRAX_VENV`, `LORRAX_SITE`,
`LORRAX_SITE_PACKAGES`, `LORRAX_INSTALL_ROOT`, `LORRAX_DEPS`,
`LORRAX_IMAGE`, `LORRAX_SIF`, `LORRAX_SHIFTER*`, `LORRAX_MODULE*`,
`LORRAX_FFI_STAGE*`, `LORRAX_FFI_BUILD_DIR`, `LORRAX_FFI_SOURCES`,
`LORRAX_FFI_HOST_*`, `LORRAX_FFI_IMAGE`, `LORRAX_FFI_PYTHON`,
`LORRAX_FFI_NO_CUDA`, `LORRAX_FFI_HAVE_{PHDF5,CAL,CUBLASMP}`,
`LORRAX_FFI_ALLOW_DEFAULT_MPI`, `LORRAX_FFI_{PHDF5,SLATE,NVHPC}_DIR*`,
`LORRAX_XLA_FFI_INCLUDE_DIR`, `LORRAX_XLA_FFI_HEADERS_DIR`,
`LORRAX_HDF5_ROOT`, `LORRAX_MPI_INCLUDE_DIR`, `LORRAX_MPICH_LIB_DIR`,
`LORRAX_MPI_LIBRARY`, `LORRAX_MPI_TYPE*`, `LORRAX_MPI_FABRICS`,
`LORRAX_IMPI_ROOT`, `LORRAX_PMI2_LIB`, `LORRAX_ICC_RUNTIME`,
`LORRAX_MKL_ROOT`, `LORRAX_SCALAPACK_LIBRARIES`, `LORRAX_NVHPC_*`,
`LORRAX_CUSOLVERMP_{STAGE,PIN,CHECK}`, `LORRAX_CUBLASMP_{PIN,CHECK}`,
`LORRAX_SLATE_*` (`REPO`, `COMMIT`, `BUILDS_DIR`, `MAKE_J`, `STACK`,
`INSTALL_DIR*`, `HOST_INSTALL_DIR`, `CUDATOOLKIT`),
`LORRAX_HAVE_SLATE`, `LORRAX_HOST_HAVE_SLATE`, `LORRAX_DARSHAN_LIB_DIR`,
`LORRAX_LUSTRE_STRIPE_*`, `LORRAX_NO_PRESTRIPE`, `LORRAX_XLA_CMDBUF`,
`LORRAX_SLURM_{ACCOUNT,QOS,CONSTRAINT}`, `LORRAX_NNODES`,
`LORRAX_NTASKS`, `LORRAX_PARTITION`, `LORRAX_NGPU`,
`LORRAX_GPUS_PER_NODE`, `LORRAX_SELECT_GPU`, `LORRAX_TIER2_WORKDIR`,
`LORRAX_FRONTERA_ADVICE`, `LORRAX_INPUT`.

`LORRAX_CUDA_CHECK` and `LORRAX_LIB_CHECK` are **C macros**, not env
vars — the earlier grep-based counts were misleading.

## 5. External variables LORRAX sets, reads, or that its harnesses dial

| var | LORRAX's handling |
|---|---|
| `JAX_ENABLE_X64` | `setdefault "1"` at 30 sites; canonical one is `runtime.set_default_env`.  Consistent everywhere. |
| `JAX_PLATFORMS` | `setdefault "cuda,cpu"`, or hard-set to `"cpu"` by `set_default_env(platform="cpu")`.  `ffi_loader.platform_from_env` READS it (default `""`) to pick the FFI library **without** initializing the JAX backend. |
| `JAX_PROCESS_COUNT` → `JAX_NUM_PROCESSES` → `SLURM_NTASKS` → `1` | process-count resolution chain, `runtime/__init__.py`. |
| `JAX_PROCESS_INDEX` → `SLURM_PROCID` → `0` | process-index chain. |
| `JAX_COORDINATOR_ADDRESS` | overrides the `SLURM_NODELIST`-derived coordinator. |
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION` | jax's own dial (`gloo` default \| `mpi` \| megascale).  `gloo`: the `LORRAX_GLOO_IFNAME` pin applies (certified default).  `mpi`: the AS.4c-certified experimental stack — REQUIRES the thread-MULTIPLE MPIwrapper (`MPITRAMPOLINE_LIB=…/mpiw_thr_install/lib64/libmpiwrapper.so`) and `LORRAX_MPI_FINALIZE_FIX=skip_atexit`; the unpatched wrapper multi-node has a measured ~21–29 % segv/hang rate (FUNNELED thread level vs concurrent mpi4py/h5py progress). |
| `MPITRAMPOLINE_LIB` / `LORRAX_MPI_FINALIZE_FIX` / `LORRAX_MPI_INIT_FIRST` | Harness-level (wk_AS `sitecustomize.py`, NOT read by `src/`): the MPIwrapper shim path; the double-`MPI_Finalize` fix (`skip_atexit` recommended); mpi4py-first init — documented DO-NOT-USE (hangs the trampoline on a pre-initialized runtime). |
| `LORRAX_MPI_PROVIDER` | Harness-level dial (runAC.sbatch / gw800_merged.sbatch case-block, NOT read by `src/`): `auto` = `FI_PROVIDER` unset ⇒ Intel MPI picks the native `mlx` (UCX) provider on CLX; `tcp` = the rtx/mlx4 escape hatch; any other value force-requests that provider.  Never `verbs` at P≥144 with the one-block eigh layout (68 s/q pathology, AP.4). |
| `FI_PROVIDER` | not read by LORRAX.  On Frontera CLX **leave it UNSET** — Intel MPI then auto-selects `mlx`: 1.07 µs / 11.4 GB/s vs the old `FI_PROVIDER=tcp` pin's 10.9 µs / 2.15 GB/s, and pzheevd n=2448 at P=144 goes 12 s/q → 0.5–0.9 s/q (AP.3/AP.4; reproduced in-container by AS.2).  Keep `I_MPI_DEBUG≥4` so rank 0 announces `libfabric provider:`; do NOT trust `fi_info`, which reports −61 for `mlx` even where it works.  In-container: apptainer's default mount already exposes the host `/dev` (uverbs included) — **never `--bind /dev[/...]`** (a nosuid,nodev shadow copy breaks every device open, AS.1); stage the RDMA userspace via the `/hostlibs` symlink pattern (AS.1 / wk_AS `as_inner.sh`), or the provider falls back to tcp, announced. |
| `CUDA_VISIBLE_DEVICES` | read to derive `local_device_ids`; `tests/conftest.py` rewrites it per xdist worker. |
| `_LORRAX_JAX_DISTRIBUTED_DONE` / `_LORRAX_GLOO_PIN_DONE` | LORRAX's own idempotency sentinels (for `jax.distributed.initialize` and `pin_gloo_interface`) — env-scoped on purpose, so they survive module re-imports.  Self-set, not user knobs. |
| `XLA_PYTHON_CLIENT_ALLOCATOR`, `TF_GPU_ALLOCATOR`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_MEM_FRACTION` | `setdefault` in the psp CLIs; `gw_init`/`gw_output` only *read* them, to caveat the high-water-mark report (cuda_async under-reports).  Not an inconsistency. |
| `XDG_CACHE_HOME` | base for the JAX compile cache. |
| `HDF5_USE_FILE_LOCKING` | `setdefault "FALSE"` in one psp test. |
| `OMP_NUM_THREADS` | `setdefault "32"` in `psp/orbital_magnetization.py` only.  On Frontera the XLA:CPU threadpool does **not** obey OMP — `taskset` pinning + `XLA_FLAGS=--intra_op_parallelism_threads` are the real mechanisms (FRONTERA_ADVICE §10). |
| `MPLBACKEND` | `setdefault "Agg"` for headless plotting. |
| `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` | `GLOO_SOCKET_IFNAME` is **INERT with jax** — the string appears nowhere in the shipped jax/jaxlib (AF.5/AK.4); every job script that exports it is exporting a no-op.  The working dial is `LORRAX_GLOO_IFNAME` (§2a).  `NCCL_SOCKET_IFNAME` *is* read by NCCL and still matters on GPU runs. |

---

## Consistency audit

Every LORRAX-owned variable read at more than one site was checked for
default drift (re-checked 2026-07-27).  **No inconsistencies found:**

| var | sites | defaults |
|---|---|---|
| `LORRAX_FORCE_FULL_BZ` | 5 | all `'0'` ✓ |
| `LORRAX_PHDF5_STRIPE_COUNT` | 3 | all `'16'` ✓ |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | 3 | all `'4M'` ✓ |
| `LORRAX_MEM_DEBUG` | 4 | all presence-test ✓ |
| `LORRAX_RCHUNK_DEBUG` | 2 | all presence-test ✓ |
| `LORRAX_ZETA_RCOND` | 2 factor sites + 1 provenance echo | one shared helper (`isdf/core._deprecated_env_float`); provenance mirrors its empty-is-unset semantics ✓ |
| `LORRAX_ZETA_RANK_LOG` | 2 | same off-set `('0','','false')` ✓ |

The scanner also flags `JAX_PLATFORMS`, `CUDA_VISIBLE_DEVICES`,
`XLA_PYTHON_CLIENT_*` and `TF_GPU_ALLOCATOR` as having "multiple
defaults".  Each is a `setdefault` (writer) paired with a plain `get`
(reader) — the intended pattern, not drift.

Re-run the audit with:

```bash
python3 tools/env_audit.py src        # AST walk; flags "MULTIPLE DEFAULTS"
grep -rn 'getenv(' src/ffi            # the C++ side, which the tool can't see
```
