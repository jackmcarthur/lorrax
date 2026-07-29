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
| `LORRAX_COLLECTIVE_CHUNK_MB` | `128` | Upper bound on ONE emitted collective's payload in the ζ `distributed` tier (C⁺ formation + back-solve GEMM, `isdf/core.py`) and the distributed W Dyson A-build (`gw/w_isdf.py`), enforced as a host-level q-block loop XLA cannot fuse back (AF).  `0`/negative = unbounded — reproduction escape hatch only.  A per-instruction TRANSPORT cap, deliberately orthogonal to the LIVE-bytes memory cap `LORRAX_ZETA_GATHER_CAP_GIB`. | Bracketed at P=144 on **em1-era Gloo**: 1.15 GB single-shot AllGather fatal, 0.104 GB good (AF).  Re-priced 2026-07-27 (scorecard AV matrix), caps {64,128,256,512,∞}: at 785c/P=16 impl=mpi/mlx the cap is indistinguishable from unbounded (596-714 ms back-solve, no trend); at P=64 ib0-Gloo the chunk loop costs +35-70 ms/r-chunk (~2-3 % of the ζ-fit) vs a healthy 192 MB single-shot, and at P=144 the chunked tier measured 1.9× FASTER than its per_q control (AF.4).  Default stays 128 = protective at the one measured fatality point, near-free everywhere else; on ib0/mlx it is not known to be *necessary* below ~370 MB payloads. |
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
| `LORRAX_GLOO_IFNAME` | unset (→ auto-detect) | Which NIC carries JAX's **Gloo CPU collectives** (`runtime/__init__.py:pin_gloo_interface`, called from `bootstrap()`). Unset: auto-detect the first UP `ib*`/`hsn*` interface with an IPv4 and re-register the CPU backend factory so `make_gloo_tcp_collectives` gets `interface=` — on Frontera that is `ib0` (InfiniBand); jax's default binds the NIC that routes to the coordinator, which is the **1 GbE management NIC `em1`** (measured 3.3× whole-pipeline on the 4×4 deck, bit-identical outputs — scorecard AK.10/AL). A name forces that interface (skipped loudly if it is not UP with an IPv4); `off`/`none`/`0` disables the pin. Machine capability, not policy — but the decision is **always announced on rank 0**, and every failure path degrades to stock jax transport with a printed reason rather than crashing. No-op for single-process runs and for `JAX_PLATFORMS != cpu` (GPU collectives are NCCL's business) — but a run that then *downgrades* to CPU (`fallback_to_cpu_if_no_gpu_backend`) re-runs the pin after forcing `JAX_PLATFORMS=cpu`, so a GPU-less node can no longer silently land its collectives on em1 (workstream AT). |
| `LORRAX_FAILFAST` | `1` (on, P>1 only) | Uncaught per-rank exception → rank-tagged banner + `os._exit(1)` so the *job* fails instead of the peers hanging in a collective the dead rank never joins (`runtime/__init__.py:install_failfast_excepthook`). `0`/`off` disables. No-op single-process. |
| `LORRAX_MALLOC_TUNE` | `1` (on) | Pins glibc `M_MMAP_THRESHOLD`/`M_TRIM_THRESHOLD` so freed XLA:CPU transients return to the OS (`runtime/__init__.py:tune_glibc_malloc`; workstream T: cures the RSS-∝-FLOPs ramp that OOM'd the 1998c fits, ≤4 % wall cost). `LORRAX_MALLOC_MMAP_MB` (`1`) / `LORRAX_MALLOC_TRIM_MB` (`128`) set the thresholds. `0` disables. |
| `LORRAX_FFI_SO` | in-tree `src/ffi/common/cpp/build/liblorrax_ffi.so` | Path to the **CUDA** FFI library (`ffi/common/ffi_loader.py:96`). |
| `LORRAX_FFI_HOST_SO` | in-tree `src/ffi/common/cpp/host/build/liblorrax_ffi_host.so` | Path to the **host** FFI library (`ffi_loader.py:103`). Point this at `$WORK/lorrax_ffi_unified/build_host/…` for the SLATE+ScaLAPACK build. FFI dependency note: the linalg handlers link MKL ScaLAPACK+BLACS on Frontera (Cray LibSci via `-DLORRAX_SCALAPACK_LIBRARIES` elsewhere); the GEMM handler below builds against either vendor's standard CBLAS — batched `cblas_?gemm_batch` entry when the BLAS has it (probed at configure time, `check_symbol_exists`), plain-GEMM loop otherwise. Works in principle with Intel MKL or Cray LibSci; **tested with Intel only so far**. |
| `LORRAX_BANDS_GEMM_FFI` | unset (→ `auto`) | The `contract_bands_block_reshard` FFI GEMM dial (`common/contract_bands.py`, read at kernel-FACTORY time — consumers key their kernel caches on it). PERFORMANCE PURPOSE: XLA:CPU lowers the primitive's LARGE right contraction through Eigen dots that saturate **1.6–1.9× below vendor BLAS at full threads** (bare-dot probe; the in-module rate is a further ~2× below that) — routing ONLY that contraction through the host GEMM handler (`lorrax_mklblas_gemm_batch`) measured project_rs 29.4→19.6 s and sigma.exec 58.3→49.2 at nb=128/P=64 (jobs 7879008/7879010). `auto`/unset = **ON when the platform is CPU and the handler resolves in the host .so** (announced once — capability detection, not policy, doctrine #8); on CUDA auto is OFF **silently by design** (XLA:GPU's dot lowering already dispatches cuBLAS, optimal — the target is not even in `ffi_loader`'s CUDA table, so nothing to unset on a GPU run); auto also quietly keeps the XLA plan for `extra="minor"` and non-f64/c128 dtypes. Platform is read from `JAX_PLATFORMS` via `ffi_loader.platform_from_env` (first entry wins; never initializes the backend, so factory-time cache-key calls stay safe before `jax.distributed.initialize`). Production CPU runs are covered: the harnesses export `JAX_PLATFORMS=cpu`, and `runtime.bootstrap()`'s GPU-less downgrade (`fallback_to_cpu_if_no_gpu_backend`) **forces** it to `cpu`. But the resolution is *lexical*, so two cases give auto-OFF on a machine that is in fact CPU — safe direction, no speedup, no error, and both fixed the same way (export `JAX_PLATFORMS=cpu`, or set this dial `=1`): a bare driver that leaves `JAX_PLATFORMS` **unset** (resolves CUDA-first by the `platform_from_env` default), and any driver that reaches a CPU mesh while `JAX_PLATFORMS` still reads `cuda,cpu`. Explicit `0` disables; explicit `1` announces-or-REFUSES (non-CPU mesh, missing handler, minor order, unsupported dtype — never a silent downgrade). Perf only, value-level identical (1e-12 gate class, not bit-exact). |
| `LORRAX_MKLBLAS_THREADS` | unset (→ `auto`) | C++ (`mklblas/cpp/gemm_batch_ffi.cc`): BLAS team size pinned (thread-locally, dlsym'd `MKL_Set_Num_Threads_Local` — no-op on non-MKL BLAS) for the GEMM handler call. `auto` = ambient `omp_get_max_threads()` (the production 28/rank under taskset); `off` = 1; integer pins exactly. Strict full-string grammar; unrecognized values announce on stderr and fall back to `auto`. |
| `LORRAX_FFT_FFI` | `0` (off) | Routes the `make_flat_k_*` FFT helpers through the platform FFI handlers (`common/fft_helpers.py`, factory-time read, kernel caches key on it): MKL FFT (DFTI API) on cpu meshes, cuFFT strided on CUDA meshes — same target names, resolved per lowering platform (2026-07-29 CUDA mirror). Explicit request: refuses (with the probe reason) when the platform's library lacks the handler; and refuses any dtype but `complex128` on **both** platforms (the handlers are c128-only; the XLA path accepts the rest). **SCOPE — read this before assuming a subsystem is covered:** the flag reaches ONLY `make_flat_k_*` call sites, i.e. `gw/` + `isdf/` + `bandstructure/htransform`. **BSE has no `make_flat_k_*` call site and is NOT affected by this flag on either platform.** The 2026-07-29 sweep routed BSE's remaining raw `jnp.fft` calls through `fft_helpers.local_ifftn3`/`local_fftn3`, which are *aliases* of `jnp.fft.ifftn`/`fftn` (bit-identical, dtype-agnostic — this is what keeps BSE's complex64 fp32-GMRES FFTs working); it centralizes the call sites for a future switch, it does not turn one on. |
| `LORRAX_FFT_FFI_FUSED` | `0` (off) | The fused IFFT·(G·W)·FFT τ-kernel entry (`gw/ppm_tau_kernel.py` → `make_flat_k_gw_conv`); independent of `LORRAX_FFT_FFI`; same refuse semantics. |
| `LORRAX_WFN_BACKEND` | `""` (→ config/auto) | Forces the WFN read backend: `eager` \| `phdf5` \| `phdf5_host` (`file_io/wfn_loader.py:277`). |
| `ISDF_JAX_CACHE_DIR` | `$XDG_CACHE_HOME/isdf_jax_compilation` (→ `~/.cache/...`) | JAX persistent compile-cache dir (`common/jax_compile_cache.py`). `""` (or whitespace-only) opts out entirely — **announced on rank 0 since AT** (the silent opt-out let harnesses keep exporting `""` for the pre-AH deadlock reason long after AH removed the deadlock). **Entries live in ONE shared `{base}/np{P}/` per world size and the cache is ON at every P** (scorecard AH). It used to be refused at `process_count() > 1` because the old per-rank `rank{i}/` layout combined with JAX's process-0-only write (`jax/_src/compiler.py::_cache_write`) guaranteed a divergent hit/miss pattern, and XLA:GPU compilation is a collective (`xla::gpu::AutotunerPass` → `MultiProcessKeyValueStore` → `CoordinationServiceAgent::GetKeyValue`) → permanent silent hang (scorecard AG). AH replaces the refusal with a real fix: a process-invariant cache key, a coordination-service agreement on the usable entry set taken before any compile, atomic writes, and JAX's rank-asymmetric XLA sub-caches disabled. |
| `LORRAX_JAX_CACHE_MULTIPROCESS` | `1` (**on** — was `0`) | Historically the escape hatch that "re-armed the deadlock"; now the DEFAULT path. Set it to `0` to restore the scorecard-AG refusal (no persistent cache at all when P > 1) if you ever need to bisect against it (`common/jax_compile_cache.py`). |
| `LORRAX_JAX_CACHE_INVARIANT_KEY` | `1` (on, P>1 only) | Makes jax's persistent-cache key process-invariant (forces the GPU-only device-assignment strip on every platform and canonicalises the accelerator-config hash). MEASURED without it (4 CPU ranks): rank 0 hits 7/7 while ranks 1-3 hit 0/7 and compile — which is the AG divergence, so `0` does not merely lose the win, it switches the cache OFF at P > 1 rather than let process 0 hit alone. The per-rank component is `accelerator_config` (the serialized topology), not the device assignment. |
| `LORRAX_JAX_CACHE_AGREE_TIMEOUT_S` | `300` | Timeout for the P>1 hit/miss agreement. On expiry the run degrades to cache-off with a printed reason; it never hangs. |
| `LORRAX_JAX_CACHE_STRICT` | `1` | An entry all ranks agreed on but this rank cannot load is a divergence hazard → abort loudly. `0` downgrades to a warning (**unsafe on GPU: can hang**). |
| `LORRAX_JAX_CACHE_FORCE_DIVERGE` | `0` | TEST HOOK / positive control: every rank != 0 pretends its N alphabetically-last cache entries are missing. The agreement must drop them and say so; the run must NOT hang. |
| `LORRAX_JAX_CACHE_NO_AGREE` | `0` | TEST HOOK: shared dir with the agreement layer OFF — the naive design, i.e. the deadlock reproducer. Never use in production. |
| `LORRAX_JAX_CACHE_EXPLAIN` | `0` | Turns on `jax_explain_cache_misses` (JAX logs why each entry was not written/read). |
| `LORRAX_JAX_CACHE_PREFETCH` | `1` (on, P>1) | After the agreement, pulls the agreed entries into the page cache from a thread pool. Cache entries are tiny (876 kB for 140 of them) so reading them is pure per-file Lustre latency: measured **29 s serial** at 606c/P=16, against ~4.5 s of XLA compile saved — without this the cache is a net loss on a cold-read CPU run. `LORRAX_JAX_CACHE_PREFETCH_THREADS` (16) sets the pool size. |
| `LORRAX_MINIMAX_CACHE_DIR` | package dir | Where minimax grid tables are cached (`gw/minimax_screening.py:69`). |
| `LORRAX_DISABLE_MINIMAX_DISK_CACHE` | `""` (off) | Disables that disk cache (`minimax_screening.py:67`). |
| `LORRAX_PHDF5_STRIPE_COUNT` | `16` | Lustre stripe count. Read at 4 sites (`_slab_io_ffi.py`, `_slab_io_mpi_host.py`, `isdf_fitting.py`, C++ `context.cc`) — all `'16'`, consistent. |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | `4M` | Lustre stripe size in the `lfs setstripe -S` spelling. Read at the same 4 sites since workstream AW — previously the C++ writer read only the undocumented byte-valued `LORRAX_PHDF5_STRIPE_SIZE`, so THE documented knob silently did not reach the `PHDF5_FFI` writer (the default CPU route since AM). The byte spelling still works as a C++-side legacy fallback. |
| `LORRAX_PHDF5_MPI_STACK` | `mpich` | Build+launch: which MPI the phdf5 FFI links/loads (`run_shifter.sh:42`). |
| `LORRAX_PHDF5_ALIGN_MB` | `4` | C++: `H5Pset_alignment` threshold, MiB; `0` disables. |
| `LORRAX_PHDF5_COLL_META` | `0` | C++: `1` re-enables collective metadata ops (default off is faster). |
| `LORRAX_PHDF5_CB_NODES` / `_CB_PER_NODE` / `_CB_BUFFER_SIZE` / `_CB_WRITE` / `_DS_WRITE` | unset (→ ROMIO auto) | ROMIO collective-buffering pass-throughs, forwarded VERBATIM by **both** writers when non-empty; unset means ROMIO's automatic policy in both (unified by AW). The C++ writer used to FORCE Perlmutter-era defaults (`romio_cb_write=enable`, `romio_ds_write=disable`, `cb_buffer_size=64M`, `cb_nodes=world_size`); on Frontera forcing `romio_cb_write=enable` measured *slower* than ROMIO auto (wk_AI: 1826 vs 2066 MB/s) and the rest were never revalidated off Perlmutter. Keep them unset; they exist as A/B levers. |
| `LORRAX_PHDF5_INDEPENDENT` | off | Independent instead of collective MPI-IO **reads** (C++). |
| `LORRAX_PHDF5_COLLECTIVE_WRITES` | `1` (on) | Collective (two-phase) MPI-IO for the bulk **writes**, in BOTH writers (C++ `ffi/phdf5/cpp/context.cc:env_flag` and the Python `phdf5_host` writer `file_io/_slab_io_mpi_host.py:_env_flag` — one shared boolean grammar since the fix/zq audit; word spellings like `false` used to flip only the Python writer).  Default flipped 0→1 on this branch, MEASURED (wk_AI microbench, production tile geometry): `V_qmunu`'s strided 2-D tile decomposes into 4.1 M × 3.2 kB independent writes at P=144 (scorecard AF.4c's 1.7 MB/s), vs a few large aggregated writes per ROMIO aggregator under `H5FD_MPIO_COLLECTIVE` — ~3 orders of magnitude on the same transport.  `0` restores independent writes (pre-AI behaviour).  Cray caution: the `ad_cray_write_coll.c` OOM at ≥1 GB/rank aggregates predates this default — recorded in context.cc; revalidate before flipping it on on Perlmutter. |
| `LORRAX_PHDF5_DEDUP_REPLICAS` | `1` (on) | One canonical writer per distinct hyperslab when a mesh axis is replicated (both writers).  `0` lets every replica rank write its identical copy — merely wasteful under independent MPI-IO, **undefined behaviour** (overlapping selections) under collective MPI-IO.  Debug-only off. |
| `LORRAX_SCALAPACK_MKL_THREADS` | unset (→ `auto` = cap 4) | MKL team size pinned (thread-locally) inside the ScaLAPACK FFI handlers (`eigh`, `solve_lu`). MEASURED (wk_ENV AW, pz_bench n=2448, mlx): at the production 12×12 grid pzheevd is **11.28 s/q at 14 MKL threads vs 0.463 s/q at 4** (24×); 17.0 s/q oversubscribed at 28; 8×8 at the production 2×28 placement shows the same cliff; 4×4 (P=16) is flat. `auto`/unset caps the calling thread's MKL team at 4 (only when the global setting is larger); `off`/`0` restores the pre-AW behaviour (handler inherits `MKL_NUM_THREADS`, i.e. 28 in production — do not do this at P ≥ 64); an integer pins exactly that. The global `MKL_NUM_THREADS=28` stays right for the local `zheevd_` plan-A route and is untouched.  Values are case-insensitive since the fix/zq audit, and an UNRECOGNIZED value announces loudly on stderr and falls back to `auto` — it used to fall through `atoi()` to `off`, silently restoring the 24× configuration. |
| `LORRAX_FORCE_REFIT` | `""` (off) | `1` (or any standard truthy value: `true`/`yes`/`on`, case-insensitive — `isdf/core._env_bool` since the fix/zq audit) forces the ζ fit even when `tmp/zeta_q.h5` is complete and its provenance matches (`gw_init.py`).  Ops escape hatch for the ζ-reuse cache (the reuse gate itself is provenance-checked). |
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
| `LORRAX_PHDF5_LOG` | `1` (on) | One rank-0 banner per `phdf5_host` file open naming the EFFECTIVE collective-write / dedup / striping configuration (`file_io/_slab_io_mpi_host.py`) — the router-prints-its-decision rule (#8). |
| `LORRAX_SANITY` | unset = warn | Stage-boundary invariant checks (`common/sanity.py`): `0`/`off` skips all (escape hatch); default checks and **warns loudly but keeps running** (a false positive must never kill a 40-node job); `strict` raises `SanityError` — set in CI / regression gates. |

### 3b. Opt-in probes, dumps, test hooks

| var | default | effect |
|---|---|---|
| `LORRAX_TIMING_TRACE` / `_DEPTH` | `0` / `3` | Every `timing.section` announces entry/exit on rank 0, nesting capped at `_DEPTH` (`common/timing.py`). |
| `LORRAX_MEM_DEBUG` | unset | Memory high-water probes at named sites (`gw_init.py`, `isdf_fitting.py` ×3; presence-test at all 4). |
| `LORRAX_RCHUNK_DEBUG` | unset | Per-r-chunk shape/timing/RSS lines (`isdf_fitting.py`, `isdf/core.py`) — the instrument behind every per-chunk table on the scorecard. |
| `LORRAX_EXIT_AFTER_ZETA` | unset | Clean `SystemExit(0)` right after the ζ fit (`gw_init.py`).  Combine with `LORRAX_MAX_RCHUNKS` + `LORRAX_RCHUNK_DEBUG` for fast fit-only sweeps (the AV chunk matrix ran exactly this way). |
| `LORRAX_W_RESIDUAL_CHECK` | `0` | Prints the direct Dyson residual `‖(1−Vχ)W − V‖/‖V‖` on the first few q after a `w_dyson_solver = distributed` W solve (`gw/w_isdf.py`) — the strict numerical contract of the distributed plan (block-cyclic LU is not bit-comparable to the local per-q LU).  Adds one diagnostic jit; leave OFF when taking collective-table probes. |
| `LORRAX_CHECK_REPLICA` | `0` (off) | Re-enables `jax.device_put`'s cross-process equality assertion inside `common/collectives.py:device_put_process_local` — i.e. deliberately pays the hidden P-linear all-gather (7.8 GB/rank at P=64, scorecard Y.5/AO) the helper exists to avoid, to verify a host table really is replica-identical. Debug only; standard falsy vocabulary (`""`/`0`/`false`/`no`/`off`, case-insensitive, since AT). |
| `LORRAX_SKIP_VQ_GATES` | `0` | Skips the V_Q interpolation self-checks (`bse/vq_interp.py`).  The gates exist because V_Q interpolation errors are silent. |
| `LORRAX_TRS_CHECK` | `1` (on) | Load-time MEASUREMENT of the WFN's symmetries against the density built from its own occupied ψ (`common/density_symmetry_check.py`, called from `wfn_loader`).  `0` restores pre-2026-07 flags-only behaviour — and re-arms the scorecard-Q class of silent time-reversal; `strict` raises instead of warning.  A diagnostic whose measured verdict gates `SymMaps` TRS augmentation — default on. |
| `LORRAX_TRS_TOL` / `LORRAX_TRS_SPATIAL_TOL` / `LORRAX_TRS_MAX_K` | `1e-6` / `1e-4` / `32` | Tolerances / k-sample cap for that check.  Fixture residuals ≤ 1e-12; real magnetization is O(1e-2…1); the sample is ±-closed, so `MAX_K` is a sufficiency choice, not an approximation (`0` = all k). |
| `LORRAX_FORCE_FULL_BZ` | `0` (5 sites, consistent) | `1` disables IBZ-only ζ writes / bypasses the IBZ cascade (`gw_init.py` ×3, `screening.py`, `v_q_g_flat.py`).  Debug/test bypass: changes work done and bytes written, not physics. |
| `LORRAX_EXTRA_MU_PAD` | `""` (→ 0) | **Test-only** extra μ-pad rows to prove pad-extent invariance (`runtime/padding.py`).  Any result that moves under this at fixed P is a defect.  NEVER set in production. |
| `LORRAX_PER_PROC_RESTART` | `0` (off) | Per-rank restart shard dump (`file_io/tagged_arrays.py::save_restart_state_per_proc`).  No in-tree reader; measured 4 m 43 s + 72 GB at 12×12/c2406/P=144 immediately after the canonical collective write of the same data finished in 2.2 s.  Forensics-only; **candidate for deletion** next time `file_io/` is open (owner: I/O sibling). |
| `LORRAX_FFI_DEBUG_SHARDS` / `LORRAX_WRITE_NO_JIT` | unset | Slab-writer shard-descriptor dumps / un-jitted write path (`_slab_io_ffi.py`). |
| `LORRAX_FFI_PROFILE` | off | C++: per-call FFI timing. |
| `LORRAX_SCALAPACK_EIGH_LOG` | unset (presence-test) | C++ `scalapack/cpp/eigh_ffi.cc`: prints the pzheevd workspace sizes (lwork/lrwork/liwork, GiB) once per call — the workspace is malloc'd outside XLA and invisible to the JAX-side memory planner. |
| `LORRAX_MKLBLAS_LOG` | unset (presence-test) | C++ `mklblas/cpp/gemm_batch_ffi.cc`: one stderr line on the first GEMM-handler call (dtype, batch/M/N/K, threads, batched-entry vs plain-loop path). |
| `LORRAX_PHDF5_TIME` / `_WRITE_DEBUG` / `_DUMP_HINTS` | off | C++-side phdf5 diagnostics.  (`_SKIP_DESTROY` and `_NO_COLL_META` were listed here historically but NOTHING reads them — `_NO_COLL_META`'s live replacement is `LORRAX_PHDF5_COLL_META` in §2b, `_SKIP_DESTROY` survives only as ARCHITECTURE.md prose; dropped by the fix/zq audit.) |
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
| `FI_PROVIDER` | not read by LORRAX.  On Frontera CLX **leave it UNSET** — Intel MPI then auto-selects `mlx`: 1.07 µs / 11.4 GB/s vs the old `FI_PROVIDER=tcp` pin's 10.9 µs / 2.15 GB/s, and pzheevd n=2448 at P=144 goes 12 s/q → 0.5–0.9 s/q (AP.3/AP.4; reproduced in-container by AS.2).  Keep `I_MPI_DEBUG≥4` so rank 0 announces `libfabric provider:`; do NOT trust `fi_info`, which reports −61 for `mlx` even where it works.  In-container: apptainer's default mount already exposes the host `/dev` (uverbs included) — **never `--bind /dev[/...]`** (a nosuid,nodev shadow copy breaks every device open, AS.1); stage the RDMA userspace via the `/hostlibs` symlink pattern (AS.1 / wk_AS `as_inner.sh`), or the provider falls back to tcp, announced. |
| `CUDA_VISIBLE_DEVICES` | read to derive `local_device_ids`; `tests/conftest.py` rewrites it per xdist worker. |
| `_LORRAX_JAX_DISTRIBUTED_DONE` / `_LORRAX_GLOO_PIN_DONE` | LORRAX's own idempotency sentinels (for `jax.distributed.initialize` and `pin_gloo_interface`) — env-scoped on purpose, so they survive module re-imports.  Self-set, not user knobs. |
| `XLA_PYTHON_CLIENT_ALLOCATOR`, `TF_GPU_ALLOCATOR`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_MEM_FRACTION` | `setdefault` in the psp CLIs; `gw_init`/`gw_output` only *read* them, to caveat the high-water-mark report (cuda_async under-reports).  Not an inconsistency. |
| `XDG_CACHE_HOME` | base for the JAX compile cache. |
| `HDF5_USE_FILE_LOCKING` | `setdefault "FALSE"` in one psp test; exported `FALSE` by every production harness. AUDITED (AW, 2026-07-27): **not load-bearing on Frontera `/scratch2`** — the mount has real `flock`, and a full 785c/P=16 e2e with the variable UNSET (HDF5 default locking) ran rc=0 with all four eqp/sigma files bit-identical (`run_800c_awlock`). The MPI-IO VFD takes no POSIX locks at all, so the variable only ever governs the serial-h5py side paths (eager reads, deferred attrs, `_introspect_dataset`). KEEP the harness export anyway: `/work2` mounts `localflock` (locks are node-LOCAL — cross-node "locking" there is silently incoherent, so honest intent is to disable), and h5py wheel HDF5s differ in lock default. Machine fact, one export per harness, never per-tool. |
| `OMP_NUM_THREADS` | `setdefault "32"` in `psp/orbital_magnetization.py` only. NOTE: on Frontera the XLA:CPU threadpool does **not** obey OMP — `taskset` pinning is the real mechanism (FRONTERA_ADVICE §10). |
| `MPLBACKEND` | `setdefault "Agg"` for headless plotting. |
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION` | Not read by LORRAX; jax's own config (`gloo` default \| `mpi` \| `megascale`). `mpi` puts EVERY jax CPU collective on the MPItrampoline runtime baked into the jaxlib wheel → Intel MPI → mlx/RDMA (measured 1.18× e2e at P=16, collective-bound stages 1.4–8×, scorecard AS.4). EXPERIMENTAL until the AS.4c rep ledger extends at scale; requires the two rows below plus the overlay `sitecustomize`. The FFI side coexists by construction: `ffi/phdf5/cpp/context.cc` and `slate/cpp/context.cc` only `MPI_Init_thread(MULTIPLE)` when nothing initialized MPI first, and the phdf5 open now WARNS when the granted level is below MULTIPLE (the AS.4b race signature). |
| `MPITRAMPOLINE_LIB` | Not read by LORRAX — MPItrampoline's own dial. Must point at the **THREAD_MULTIPLE-patched** MPIwrapper (`wk_AS/mpiw_thr_install/lib64/libmpiwrapper.so`); the unpatched build grants FUNNELED and measured a ~29% multi-node crash/hang rate (AS.4b: phdf5/mpi4py I/O thread vs XLA collectives thread, both inside `MPID_Progress_wait`). Deliberately NOT auto-defaulted by LORRAX (audit AW): it is a machine fact naming a build artifact outside the repo, the hazardous-vs-certified choice must stay visible in the harness, and MPItrampoline already refuses loudly when it is missing. Harness block: scorecard AS.7. |
| `LORRAX_MPI_FINALIZE_FIX` | Read by the OVERLAY `sitecustomize.py` (wk_AS), not by `src/`. `skip_atexit` (recommended) suppresses jax's atexit `collectives.Finalize` so the C++ destructor finalizes exactly once — without it every impl=mpi run exits rc=1 after SUCCESS (double MPI_Finalize). `hard_exit` is the fallback; `LORRAX_MPI_INIT_FIRST=mpi4py` remains a documented DO-NOT-USE (hangs the trampoline on a pre-initialized MPI). No-op under gloo. |
| `LORRAX_MPI_PROVIDER` | The harness/`ffi_env.sh` dial over `FI_PROVIDER` (scorecard AP.7/AS.5). `auto` (default) **unsets** `FI_PROVIDER`+`FI_TCP_IFACE` so Intel MPI picks the native provider (`mlx` on CLX); `tcp` restores the IPoIB pin with `FI_TCP_IFACE=ib0` (the rtx/mlx4 escape hatch); any other value force-requests that provider (never `verbs` at P≥144 one-block — 68 s/q pathology, AP.4). Read by the sbatch env blocks and by `config/frontera/ffi_env.sh`; not read by Python. The `auto` unset is load-bearing, not hygiene: TACC's default impi module exports `FI_PROVIDER=mlx` into every login/compute shell and sbatch/ssh inherit it, so "leave it unset" requires actively unsetting. |
| `I_MPI_FABRICS` | Harnesses export `shm:ofi` — which is Intel MPI 2019+'s own default, so this is *documentation of intent* plus a guard against a stray inherited value, not a behavior change (AU: pingpong identical with it unset). `ffi_env.sh` defaults it to `shm` (single-node rtx bring-up: skips OFI init entirely); override with `LORRAX_MPI_FABRICS=shm:ofi` for multi-node. |
| `I_MPI_PMI_LIBRARY` | Needed ONLY under `srun --mpi=pmi2` (the Intel-MPI-under-slurm bootstrap; harness cells that run no MPI code don't need it, and nothing dlopens it unless MPI inits). MUST be set unconditionally where used: TACC's login env exports `/usr/lib64/libpmi.so` — a PMI-**1** library, wrong protocol for `--mpi=pmi2` AND absent inside the container. The staged PMI2 lib is `$WORK/host_pmi/libpmi2.so.0` (`LORRAX_PMI2_LIB` overrides in `ffi_env.sh`). |
| `I_MPI_DEBUG` | Default 4 in every harness + `ffi_env.sh`. Init-time-only output (provider banner + pinning table); AU measured steady-state pingpong identical at `I_MPI_DEBUG=0`, so the banner is free. It is MANDATORY telemetry — the `libfabric provider:` line is the only trustworthy provider observable (`fi_info` false-negatives on mlx), and a silent transport is how the em1/tcp era happened. Keep ≥4. |
| `FI_PROVIDER_PATH` | Harnesses + `ffi_env.sh` pin it to `$IMPI/libfabric/lib/prov` (the bundled provider .so dir). **REQUIRED in-container, not belt-and-braces**: `mpivars.sh` (which normally sets it) is not sourced there, and AU measured that with it unset PMPI_Init aborts outright — `MPIDI_OFI_mpi_init_hook ... addrinfo() failed ... No data available`, i.e. libfabric finds NO providers at all. (Note this is the exact error string of the rtx-era "tcp/mlx4 fails in-container" archaeology — some of that history may have been a missing FI_PROVIDER_PATH, not the fabric.) |
| `UCX_*` (`UCX_TLS`, retry/timeout tunings) | TACC's default impi module exports `UCX_TLS=knem,dc_x,rc` + `UCX_{RC,DC,UD}_MLX5_{TIMEOUT,RETRY_COUNT}` bumps into every shell, and sbatch/ssh launches inherit them — so every AP/AS `mlx` number (1.07 µs / 11.4 GB/s, pzheevd 0.52 s/q) was measured UNDER those tunings, not under bare UCX defaults. AU A/B (in-container, provider auto): stripping every `UCX_*` leaves 8 B pingpong/allreduce unchanged (1.07 µs / 3.38 µs) but **doubles the 1 MiB 32-rank Allreduce (419 → 799 µs)** — the tunings are load-bearing for large-message collectives. The harness blocks therefore SETDEFAULT the six module values (`${UCX_TLS:-knem,dc_x,rc}` etc.): inherited values always win, but a stripped launch environment (cron, clean ssh) no longer silently loses 2×.  Since the fix/zq audit the TRACKED `config/frontera/ffi_env.sh` carries the same six setdefaults (guarded to the non-`tcp` provider cases), so the mitigation is auditable in-repo rather than only in the /scratch harnesses. Do not hard-pin them (rtx/mlx4 has no `dc_x`), and do not add other UCX knobs — the mlx TPN=4 × n=5024 anomaly (AP.9.4) is the only open UCX question and is not production-relevant (2×28 layout is clean). |
| `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` | `GLOO_SOCKET_IFNAME` is **INERT with jax** — the string appears nowhere in the shipped jax/jaxlib (scorecard AF.5/AK.4); every job script that exports it is exporting a no-op. The working dial is `LORRAX_GLOO_IFNAME` above (§2). `NCCL_SOCKET_IFNAME` *is* read by NCCL and still matters on GPU runs. |

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
| `LORRAX_ZETA_RCOND` | 2 factor sites + 1 provenance echo | ONE shared non-empty-env-wins rule (`isdf/core._env_override_raw`), used by both the factor sites (`_deprecated_env_float`) and the provenance record (`deprecated_env_record`) — the inline mirror in `gw_init` was deleted by the fix/zq audit ✓ |
| `LORRAX_ZETA_RANK_LOG` | 2 | one shared helper (`isdf/core._env_bool`, canonical `1/true/yes/on` grammar — the old `not in ('0','','false')` off-set left the log on under `=False`/`=off`; fixed by the fix/zq audit) ✓ |

The scanner also flags `JAX_PLATFORMS`, `CUDA_VISIBLE_DEVICES`,
`XLA_PYTHON_CLIENT_*` and `TF_GPU_ALLOCATOR` as having "multiple
defaults".  Each is a `setdefault` (writer) paired with a plain `get`
(reader) — the intended pattern, not drift.

Re-run the audit with:

```bash
python3 tools/env_audit.py src        # AST walk; flags "MULTIPLE DEFAULTS"
grep -rn 'getenv(' src/ffi            # the C++ side, which the tool can't see
```
