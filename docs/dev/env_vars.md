# LORRAX environment variables — the registry

This page owns four facts about every environment variable LORRAX reads: its
**spelling**, its **default**, its **class**, and its **parse grammar**. Each
row adds one sentence on what the variable controls and what it refuses; the
explanation belongs to the page the row links to. `tests/test_env_registry.py`
fails when a variable read under `src/` or `services/*/src` (Python or C++)
has no row here.

**Policy.** The environment grants machine capability (library paths,
transport, thread counts, resource caps) and debug switches. It never selects
physics or routing: those are input-file keys, which are validated, echoed and
recorded in provenance. An env twin of an input key that still works is
deprecated and prints a notice whenever it is set (§1a); the routing-affecting
variables that remain are §1b.

**Grammars used below.** *bool* is `runtime/env_flags.py::env_bool`: unset or
blank gives the default, `1 true yes on` (any case) is on, `0 false no off` is
off, and any other value is off and announced once with `*** LORRAX SANITY`.
*falsy-set* means `"" 0 false no off` (any case) is off and every other value
is on. A row with neither word states its own grammar.

What a run resolved is printed in its rank-0 [startup report](../environment/overview.md#startup-block).
Several variables are read once, before backend creation, so `os.environ`
after startup is not evidence of what applied.

---

## 1. Input keys: env twins and routing-affecting variables

### 1a. Deprecated env twins

Unset or blank uses the input key. Any other value wins, is cast as shown (a
value the cast rejects raises), and prints a deprecation line naming the key.
The ζ pair announces on rank 0 once per process, and ζ-fit provenance records
the effective value, so a rerun without the variable cannot silently reuse a ζ
fit made at a different cutoff.

| var | input key | class | cast | controls |
|---|---|---|---|---|
| `LORRAX_ZETA_RCOND` | `zeta_rcond` (`1e-8`) | twin | float | Relative eigenvalue cutoff of the rank-revealing ζ pseudo-inverse (`rank_truncate`, `isdf/core.py`). |
| `LORRAX_SC_MAX_ITER` | `sc_max_iter` | twin | int | Self-consistency iteration cap. |
| `LORRAX_SC_TOL_EV` | `sc_tol_ev` | twin | float | Self-consistency convergence tolerance, eV. |
| `LORRAX_SC_MIXING` | `sc_mixing` | twin | float | Self-consistency mixing weight. |
| `LORRAX_SC_DUMP_DIR` | `sc_dump_dir` | twin | string | Directory for per-iteration self-consistency dumps. |

### 1b. Routing-affecting variables

Each of these chooses a route or a tolerance outside the input file, so it
escapes input validation and provenance. Set one deliberately and record it
with the run.

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_ZETA_REPLICATE_CAP_GIB` | `4` | routing-affecting | Float GiB, read at import of `isdf/core.py` (a malformed value fails the import). Cap on the `(nq, μ, μ)` c128 charge stack below which the ζ factor is replicated, and therefore mesh-invariant; above it `rank_truncate` refuses and names the value to set. |
| `ISDF_CHUNK_TARGET_UTILIZATION` | `0` (planner default) | routing-affecting | `gw_config.env_float`: a malformed value is announced and the default used; a positive value is clamped to `[0.85, 1.0]`. Fraction of the device memory budget the chunk planner fills, which sets the chunk shapes (`gw/gw_config.py`). |
| `LORRAX_WFN_BACKEND` | unset (auto) | routing-affecting | Stripped, lower-cased: `eager` or `phdf5` forces the WFN read backend (`services/wfn_loader/src/wfn_loader/loader.py::_auto_pick_backend`); `phdf5_host` refuses; any other value takes the auto pick without comment. A mesh-less loader always reads eager. At P>1 with no loadable FFI the auto pick refuses, and `eager` is the way through. |
| `LORRAX_SIGMA_PLAN` | `box` | routing-affecting | Stripped, lower-cased enum `box` or `panes`; blank is `box`; any other value refuses naming both (`gw/sigma_plan.py`). `box` is the shared denominator-box planner for MPA and GN/HL-PPM (`gw/sigma_box_plan.py`); `panes` keeps the per-method quadrature controls as the comparison route. |

---

## 2. Machine capability and resource caps

None of these may change physics. Those that change wall time carry the one
number that sets the cost of choosing a value.

### 2a. Startup: network transport and GPU pool

`runtime/network_env.py` applies the network decision once, before backend
creation, and only on a CUDA run; a CPU or ROCm run is untouched. What the
decision is and why: [Perlmutter: network transport at startup](../environment/machines/perlmutter.md#network-transport-at-startup).

| var | default | class | grammar and effect |
|---|---|---|---|
| `SLURM_STEP_NUM_NODES` | unset | launch | Positive integer node count of the step, the first topology evidence; an invalid value falls through to the host list. |
| `SLURM_STEP_NODELIST` | unset | launch | Slurm host-list expression, expanded by `scontrol show hostnames`. |
| `OMPI_COMM_WORLD_SIZE`, `OMPI_COMM_WORLD_LOCAL_SIZE` | unset | launch | Positive integers; equal means one node, local below world means several (no uniform placement assumed). |
| `NERSC_HOST` | unset | machine | Exactly `perlmutter` enables the Perlmutter profile on a multi-node step. |
| `SLURM_NETWORK` | `lx run` exports `no_vni` on every multi-node GPU step | launch | Comma-separated Slurm network options; `no_vni` must be present when `srun` creates the step, since setting it inside Python is too late. A multi-node Perlmutter CUDA step without it refuses at startup. |
| `NCCL_NET`, `NCCL_NET_PLUGIN` | Perlmutter profile: `AWS Libfabric` and the site OFI plugin by absolute path | transport | Presence of either, even empty, bypasses site selection and keeps the caller's configuration; `NCCL_NET=Socket` (any case) refuses on a multi-node Perlmutter CUDA step. The profile refuses when the site plugin file is missing. |
| `NCCL_NET_GDR_LEVEL`, `FI_CXI_DISABLE_HOST_REGISTER`, `FI_CXI_RDZV_THRESHOLD`, `NCCL_CROSS_NIC`, `NCCL_SOCKET_IFNAME` | Perlmutter profile: `PHB`, `1`, `0`, `2`, `hsn` | transport | Filled in only when the profile is selected and only where unset; exported values are kept verbatim. |
| `XLA_PYTHON_CLIENT_ALLOCATOR`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_CLIENT_MEM_FRACTION` (deprecated spelling `XLA_PYTHON_CLIENT_MEM_FRACTION`) | set by `runtime.set_default_gpu_pool` | GPU pool | On a CUDA run with an NVIDIA device, when neither `ALLOCATOR` nor `PREALLOCATE` is exported, the runtime sets all three: `cuda_async`, `true`, and `XLA_CLIENT_MEM_FRACTION` = `runtime.GPU_POOL_FRACTION` (`0.89`) unless a fraction is already exported; a CPU run, a GPU-less node and ROCm get nothing, and an exported `ALLOCATOR` or `PREALLOCATE` makes the caller own the pool. Refused: `cuda_async` with preallocation off, preallocation on with no allocator, both fraction spellings at once, and an allocator spelling jaxlib rejects (a blank allocator is removed). `tests/conftest.py` pins `bfc` with preallocation off for test workers, which share GPUs. What the pool does: [environment/overview.md#gpu-pool](../environment/overview.md#gpu-pool). |
| `XLA_FLAGS` (`--xla_gpu_autotune_level`) | GPU: `--xla_gpu_autotune_level=0` appended | machine | `runtime.set_default_xla_gpu_autotune` appends the flag unless `--xla_gpu_autotune_level=N` or `--xla_gpu_autotune_level N` is already present; other flags are untouched and a caller value wins. Not added on a CPU run. Level 0 cuts cold compile 13–16 % at P=4 with execution unchanged. |
| `LORRAX_CPU_SKIP_GPU_PLUGINS` | `1` | machine | *bool*. On a CPU-only run (`JAX_PLATFORMS=cpu`, or no NVIDIA device on the node) skips jax's CUDA plugin discovery (cost on a cold Frontera node: [frontera.md §3](../environment/machines/frontera.md#3-cold-start)); `0` re-enables it and records a demotion. A run with a GPU present is unaffected. |
| `LORRAX_MATMUL_PRECISION` | unset (`highest`) | numerics | `highest` or `float32`; any other token refuses naming the variable, `high` included (it selects a 3-pass TF32 decomposition on XLA:GPU). Sets `jax_default_matmul_precision` at `bootstrap()`; left at XLA's default every f32 and c64 dot runs at TF32 (forward error 1.9e-4 against 3.2e-7 pinned on the BSE ladder matvec). Pinning is free at block width 1 and costs 8 % at 2 and 46 % at 4. |
| `LORRAX_FAILFAST` | `1` | machine | *bool*. At P>1 an uncaught exception on one rank prints a rank-tagged banner and calls `os._exit(1)`, so the job fails instead of its peers hanging in a collective; the CLI bootstrap likewise aborts the step. `0` disables both; `SystemExit(0)` stays a clean exit. |
| `LORRAX_MALLOC_TUNE` / `LORRAX_MALLOC_MMAP_MB` / `LORRAX_MALLOC_TRIM_MB` | on / `1` / `128` | machine | *bool* / integer MiB / integer MiB. At bootstrap `runtime.tune_glibc_malloc` sets glibc `M_MMAP_THRESHOLD` and `M_TRIM_THRESHOLD` so freed XLA:CPU transients return to the OS (≤ 4 % wall); a tuning that does not arm is a demotion. |
| `LORRAX_BLAS_TUNE` | `1` | machine | *bool*. At import of `runtime`, before numpy, `setdefault`s `OPENBLAS_THREAD_TIMEOUT=1` so idle OpenBLAS workers stop spinning; an exported value wins, and numpy imported first records a `blas_tune` demotion (`tests/test_runtime_blas_env.py` enforces the import order). Spinning made a 140×140 `cho_factor` between Python work 28.7 ms against 0.067 ms standalone; the setting costs 22 % only in a tight back-to-back BLAS loop, which no driver is. |

### 2b. Memory caps, collective chunking, schedules

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_COLLECTIVE_CHUNK_MB` | `128` | resource cap | Float MiB; a malformed value uses the default; `≤ 0` is unbounded (reproduction only). Upper bound on one emitted collective's payload, enforced as a host-level loop XLA cannot fuse back: ζ-fit collectives (`isdf/core.py`), the distributed W Dyson A-build (`gw/w_isdf.py`), the rank-0 owner gather of k-partitioned sweeps (`common/collectives.py::gather_indexed_blocks_to_owner`), and the V_q G-panel width (`gw/v_q_g_flat.py`). A single 1.15 GB all-gather was fatal on Gloo at P=144; on MPI at P=16 the cap costs nothing measurable. Orthogonal to the live-bytes cap below. |
| `LORRAX_ZETA_QPARALLEL` | unset (`auto`) | schedule | Blank or `auto`: fold when P>1, nq ≥ 2 and nq·μ³ ≥ 5e9; otherwise *bool* (`1` forces the fold, `0` the all-ranks execution). Schedule of the replicated charge ζ factor: the fold scatters q over all devices, each factoring whole per-q tiles, and reshards back; same plan, same bits (`tests/test_zeta_mesh_invariance.py`). Unfolded, P=16 at nq=10, μ=2979 spent 105 s in one dense eigh per q on every rank. |
| `LORRAX_PPM_FIT_ARENA_GIB` | unset | resource cap | Float GiB. Overrides the free bytes the GN-PPM fit's q-block sizer (`gw/minimax_screening.py::_gn_ppm_fit_q_block`) prices against; unset, the sizer takes 0.8 of the device pool's free bytes (min over processes) and one q's footprint from the kernel compiled at q = 1, and refuses (`GATE gn_ppm_fit_capacity`) below one q. Bit-exact at any q block. |
| `LORRAX_KIN_ION_LOOKAHEAD` | `2` | schedule | Positive integer; a malformed value refuses naming the variable. Host-ahead-of-device depth of `common/collectives.py::sweep_local_k`; `1` serialises, the control for measuring the overlap. |
| `LORRAX_GRAM_COL_BLOCK` | unset (auto) | resource cap | Falsy tokens (`""`, `0`, `false`, `no`, `off`) select auto; a positive integer pins the width (floor 256, aligned to both mesh axes); anything else refuses. Tile width of the pivoted-Cholesky Gram build (`centroid/pivoted_cholesky.py::build_gram_q0_via_loadwfns`); auto picks the largest width whose AOT-measured live set fits, full width when the whole Gram fits. Workspace only; the contraction order and the `P('x','y')` Gram are unchanged. |
| `LORRAX_FACE_TO_BATCH_ROUTE` | unset (`staged_reshard.DEFAULT_ROUTE`) | schedule | A `common.staged_reshard` route name for the fH_q face→batch move (`bandstructure/bse_setup.py::resolve_reshard_route`); a caller's `reshard_route` argument wins. An unknown token is announced (`*** LORRAX SANITY`) and the default runs. Movement only, value-identical. |
| `LORRAX_LANCZOS_REORTH` | unset (`cgs2`) | schedule | `cgs2` or `mgs`; an unknown token refuses. Read by `bse/bse_lanczos.py::reorth_route` and passed as a token to `solvers/lanczos.py`. `cgs2` is batched classical Gram–Schmidt twice, 2·max_iter all-reduces of an `(m,)` vector; `mgs` is the per-vector sweep, max_iter(max_iter+1)/2 scalar all-reduces, kept for bisects. Same reorthogonalisation window on both (max\|Δλ\| 1e-14 eV). |
| `LORRAX_MALLOC_TRIM` | `1` | machine | *bool*. One `malloc_trim(0)` per real-space tile of the current-channel ζ-fit loop (`gw/isdf_fitting.py`); glibc only, a no-op elsewhere. |

### 2c. Numerical guards

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_RANK_POLICY` | `refuse` | guard | `refuse`, `warn` or `off`; anything else refuses naming the variable. Whether a rank truncation that bound and left the certified regime stops the run (`common/rank_criterion.py`), at the ζ charge and transverse fits and the indefinite transverse ridge path. Never changes a number. Site register: [`rank_truncation_policy.md`](rank_truncation_policy.md). |
| `LORRAX_SPECTRAL_CLOSURE` | `snap` | guard | `snap`, `strict` or `off`; a misspelling raises. Degeneracy closure of a spectral rank cut (`common/spectral_closure.py`; the ζ sites in `isdf/core.py`, `centroid/pivoted_cholesky.py`, `gw/downfold.py`): `snap` drops the whole straddled block, `strict` refuses, `off` does not look. |
| `LORRAX_BAND_DEGENERACY` | unset (`strict` for a named edge, `snap` for `number_bands`) | guard | `strict`, `snap` or `off` (`common/band_degeneracy.py::MODES`, the `--band-degeneracy` vocabulary); an unrecognised value refuses. Band-window degeneracy guard read in `gw/gw_init.py`: a window rounds outward or refuses. Never set it to make a gate pass. |
| `LORRAX_CENTROID_PC_TOL` | unset (`sqrt(eps)` ≈ 1.49e-8) | guard | Float. Stopping tolerance of the pivoted-Cholesky centroid selector (`common/pivoted_cholesky.py`), relative to the largest Gram diagonal: certification stops when the largest residual Schur diagonal falls to `tol·max(diag G)`. Raising it certifies fewer directions; `LORRAX_CENTROID_SELECT` decides whether that refuses. |
| `LORRAX_CENTROID_SELECT` | `deliver` | guard | `deliver` or `strict`; anything else refuses. When the candidate pool is numerically flat but not empty (`centroid/pivoted_cholesky.py`), `deliver` returns the requested set and prints the certified rank, the delivered count and the ζ truncation it implies; `strict` refuses. |
| `LORRAX_CENTROID_POINT_RANK_CAP` | `4096` | resource cap | Integer. Size cap on the dense point-rank diagnostic of the explicit representative-group path (`centroid/pivoted_cholesky.py::point_granularity_rank`); above it the diagnostic reports NOT MEASURED instead of paying O(n³) host work. Never changes the selection. |
| `LORRAX_FI_FSHOULDER_TOL` | `0.0` | guard | `gw_config.env_float` in refuse mode. Floor of the f-shoulder gate on the fine-grid interpolation window (`bandstructure/bse_setup.py::resolve_fi_fshoulder_tol`): a band whose occupation shoulder is at or below it is refused from the f-transform window. A negative value disables the gate, announced; any non-default value is announced. |

### 2d. FFI libraries and native threads

The C++ side of the registry test sees only `getenv`, `log_here` and
`env_flag` literals; the alias rows read through `mklpin::knob_value` are
kept by hand.

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_FFI_SO` | in-tree `src/ffi/cpp/build/liblorrax_ffi.so` | machine | Path of the CUDA FFI library. A sealed deployment needs no pin: the loader finds `lorrax_ffi_bundle.json` beside `lib/`. Pinning either sealed leg requires this and `LORRAX_FFI_HOST_SO` to name the exact pair in that manifest; a missing, partial, mixed, wrong-hash, wrong-origin or wrong-ABI selection refuses. An ordinary run sets no `LD_LIBRARY_PATH`. Build legs: [`architecture/ffi_layout.md`](../architecture/ffi_layout.md). |
| `LORRAX_FFI_HOST_SO` | in-tree `src/ffi/cpp/build_host/liblorrax_ffi_host.so` | machine | Path of the host FFI library, under the same sealed-pair rule. `config/frontera/build_ffi_host.sh` writes its site build elsewhere, so an unsealed developer run may point here. |
| `LORRAX_FFI_ABI_STRICT` | unset (announce) | machine | Exactly `1` refuses an unsealed library that exports no `lorrax_ffi_{host,cuda}_abi_version`; unset prints `LEGACY-UNSEALED` with the hash and continues. A sealed bundle is always strict, and a stamped ABI disagreement always refuses. The artifact-level twin is `LORRAX_FFI_VERIFY_STRICT` in `scripts/verify_ffi_build.sh` ([`building_ffi.md`](../building_ffi.md)). |
| `LORRAX_BANDS_GEMM_FFI` | unset (`on`, required) | machine | `ffi/gate.py` grammar: unset or `1` is the vendor-BLAS host handler for `contract_bands_block_reshard` on a CPU mesh; `0` is an announced, uncertified opt-out onto the XLA einsum; a stale `auto` resolves to the default with a note. A missing or unloadable handler refuses at startup and at the kernel factory; kernel caches key on the value. Serves f64/f32/c128/c64 and refuses other dtypes. Not read on CUDA, where XLA's dot lowering already calls cuBLAS. XLA:CPU's Eigen dots run 1.6–1.9× below vendor BLAS at full threads. |
| `LORRAX_FFT_FFI` | unset (`on`, required) | machine | `0` refuses on both platforms: there is no XLA flat-k twin. On CPU the host flat-k FFT serves every `make_flat_k_*` call site (`gw/`, `bandstructure/htransform`); on CUDA the flat-k transform is the k-convolution router's nvidia-mathdx mode and this gate is silent. c128 only; another dtype refuses at trace time. BSE has no `make_flat_k_*` call site. Which engine answers: [`ffi_layout.md` §3](../architecture/ffi_layout.md). |
| `LORRAX_FFTW3_SO` | unset (the candidate ladder) | machine | Path of the file the host FFT engine is `dlopen`ed from (`ffi/cpp/fftw/fft_flat_k_ffi.cc::fftw3_candidates`, at the first FFT), tried ahead of the build's compile-time hint and of `libfftw3.so.3`, `libfftw3.so.mpi31.3`, `libmkl_rt.so`, `libfftw3.so`. A bad path is skipped silently; no engine at all refuses at the first host FFT, naming every candidate (the startup gate checks only the handler symbol). Nothing at run time checks that the named file is a CPU FFTW3 (a GPU FFTW-API shim exports the same three symbols); the stage-time check is `ffi/cpp/gate_one_fftw.sh`. |
| `LORRAX_MKLBLAS_THREADS` | unset (`auto`) | machine | `auto` = ambient `omp_get_max_threads()`, `off` = 1, an integer pins; strict full-string match, an unrecognised value is announced on stderr and becomes `auto`. Thread-local BLAS team of the GEMM handler call (`cpp/cblas/gemm_batch_ffi.cc`; a no-op on non-MKL BLAS). |
| `LORRAX_FFT_FFI_THREADS` | unset (`auto`) | machine | Same grammar, integer 1–4096. OpenMP team of the flat-k FFT chunk loop (`fft_flat_k_ffi.cc::team_threads`). `LORRAX_MKLFFT_THREADS` is a deprecated alias, announced once; this spelling wins when both are set. |
| `LORRAX_FFT_FFI_CHUNK` | unset (auto) | machine | Positive integer or auto; an unrecognised value is announced and auto runs. Trail elements per FFT chunk; auto sizes the per-thread buffer (`nk·chunk·16 B`) to ~512 KiB so it stays in L2 (the strided form runs 2.8× slower single-thread). `LORRAX_MKLFFT_CHUNK` is a deprecated alias, announced once. |
| `LORRAX_SCALAPACK_MKL_THREADS` | unset (`auto`, cap 4) | machine | Case-insensitive `auto`, `off`/`0`, or an integer; an unrecognised value is announced and becomes `auto`. MKL team size pinned inside the ScaLAPACK `eigh` and `solve_lu` handlers: `auto` caps it at 4 when the global setting is larger, `off` inherits `MKL_NUM_THREADS`. pzheevd at n=2448 on a 12×12 grid runs 11.3 s/q at 14 threads against 0.46 s/q at 4. |
| `LORRAX_SCALAPACK_ALLOW_SLATE_API` | off | machine | Standard boolean spellings, case-insensitive; a malformed value is announced and stays refused. Waives the refusal of SLATE's `libslate_scalapack_api` overlay answering the pzheevd/pzgetrf symbols (`cpp/scalapack/blacs_grid.h`); see `SLATE_SCALAPACK_TARGET` (§5). |

### 2e. Compile cache

The `LORRAX_JAX_CACHE_` rows use the *falsy-set* grammar, so a blank value
turns a default-on switch off. Read in `common/jax_compile_cache.py`.

| var | default | class | grammar and effect |
|---|---|---|---|
| `ISDF_JAX_CACHE_DIR` | unset: the runtime default, `$SCRATCH/.cache/lorrax/jax_compile/<source>_<jax/jaxlib>_<ffi bundle>/np{P}` | machine | Unset, `common.jax_compile_cache` arms the cache in one namespace per source release or commit, jax/jaxlib and FFI bundle, and rank 0 prunes whole namespaces (never one used in the last 5 days; after 7 days unused, or LRU past 2 GiB or 200k files). A non-empty value is used as-is, never namespaced or pruned; blank or whitespace opts out. Launchers and modules do not set it. |
| `JAX_COMPILATION_CACHE_MAX_SIZE` | `-1` (unlimited) | external | JAX's own control; `0` disables the cache. A positive cap (JAX's LRU eviction) is supported only at P=1 and refuses at P>1, where LORRAX freezes an agreed all-rank entry set at startup. |
| `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` | unset: LORRAX sets `0` whenever the cache is on | external | JAX's write threshold. JAX's own `1.0` default persisted 2 of 666 MoS2 bispinor executables; an exported value wins. |
| `LORRAX_JAX_CACHE_MULTIPROCESS` | `1` | machine | *falsy-set*. `0` disables the persistent cache at P>1. |
| `LORRAX_JAX_CACHE_INVARIANT_KEY` | `1` (P>1) | machine | *falsy-set*. Makes the persistent-cache key process-invariant (strips the device assignment, canonicalises the accelerator-config hash). Off, ranks key differently and the cache is switched off at P>1. |
| `LORRAX_JAX_CACHE_SHARD_SLICE` | `1` (P>1) | machine | *falsy-set*. Patches `ArrayImpl._multi_slice` so every rank compiles one program (static shard sizes, dynamic offsets; bit-identical). Reached on every P>1 GPU run through `runtime.nccl_warmup`. `0` is the red-twin test hook that restores per-rank programs. |
| `LORRAX_JAX_CACHE_AGREE_TIMEOUT_S` | `300` | machine | Integer seconds (a malformed value uses the default) for the P>1 hit/miss agreement; expiry degrades to cache-off with a printed reason. |
| `LORRAX_JAX_CACHE_STRICT` | `1` | machine | *falsy-set*. An agreed entry this rank cannot load aborts; `0` warns instead (unsafe on GPU: can hang). |
| `LORRAX_JAX_CACHE_PREFETCH` / `LORRAX_JAX_CACHE_PREFETCH_THREADS` | `1` (P>1) / `16` | machine | *falsy-set* / integer. After the agreement, reads the agreed entries into the page cache from a thread pool: serial reads cost 29 s at P=16 against ~4.5 s of compile saved. |
| `LORRAX_JAX_COMPILE_AGREEMENT` | `1` (P>1 with a coordination client) | machine | *bool*. Before every backend compile, hashes the location-free module and exchanges it over the coordination service in one global compile order; a different key, a different module in the same slot, or a missing rank refuses before execution, naming every rank's key. Process-local handle literals are canonicalised. `0` is an unsafe bisect-only opt-out. |
| `LORRAX_JAX_COMPILE_AGREE_TIMEOUT_S` | `0` | machine | Seconds; `0` waits without bound and prints a heartbeat every 60 s naming the missing rank. A late rank is skew, not disagreement, so set a finite value only in tests. |
| `LORRAX_MINIMAX_CACHE_DIR` | `~/.cache/lorrax/minimax_quadratures` | machine | Where the minimax service caches rules solved at run time (`services/minimax/src/minimax/cache.py`); cached rules are never certified, and the key records the solver version and numerics backend. |
| `LORRAX_DISABLE_MINIMAX_DISK_CACHE` | unset | machine | Stripped, case-insensitive `1`, `true` or `yes` disables that disk cache. |

### 2f. HDF5 and slab I/O

Why each default was chosen: [`architecture/slab_io.md#tuning`](../architecture/slab_io.md#tuning).
The C++ writer is `ffi/cpp/phdf5/context.cc`; the Python side is
`file_io/_slab_io_ffi.py`.

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_PHDF5_STRIPE_COUNT` | `clamp(P, 4, 128)` | machine | Integer; a non-integer or negative value refuses (both sides). Lustre stripe count, which is also the ROMIO aggregator count. |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | power of two nearest `P/16` MiB, clamped to `[1, 4]` MiB | machine | `lfs setstripe -S` spelling with at most one suffix character; an unknown suffix refuses, and `4MiB` refuses. `LORRAX_PHDF5_STRIPE_SIZE` is the byte-valued legacy spelling, read when this one is unset; a non-integer refuses. |
| `LORRAX_PHDF5_ALIGN_MB` | `4` | machine | Integer MiB `H5Pset_alignment` threshold; `0` disables. `4`, `1` and `0` measure within repeat noise. |
| `LORRAX_PHDF5_COLL_META` | `0` | machine | `1` enables collective metadata operations (off is faster). |
| `LORRAX_PHDF5_CB_NODES` / `LORRAX_PHDF5_CB_PER_NODE` / `LORRAX_PHDF5_CB_BUFFER_SIZE` / `LORRAX_PHDF5_CB_WRITE` / `LORRAX_PHDF5_DS_WRITE` | unset (ROMIO auto) | machine | ROMIO collective-buffering hints: four are forwarded verbatim; `CB_PER_NODE` becomes `cb_config_list="*:N"`. A/B levers; forcing `romio_cb_write=enable` measured slower than auto on Frontera. |
| `LORRAX_PHDF5_INDEPENDENT` | off | machine | Independent instead of collective MPI-IO reads. |
| `LORRAX_PHDF5_COLLECTIVE_WRITES` | `1` | machine | *bool* (C++ `env_flag`, the same table). Transfer mode of writes that leave as they are (`lorrax_phdf5_write`); `0` writes them independently. File-order row-block writes (`lorrax_phdf5_write_independent`, `docs/architecture/slab_io.md`) are always independent and do not read it. A strided `V_qmunu` tile at P=144 decomposes into 4.1 M × 3.2 kB independent writes, and independent writes at 16×4M striping fell to 0.068 GiB/s at 4 nodes, while collective stayed within ~10 % of the best at every geometry. |
| `LORRAX_PHDF5_DEDUP_REPLICAS` | `1` | machine | One canonical writer per distinct hyperslab when a mesh axis is replicated. `0` lets every replica write, which is undefined behaviour under collective MPI-IO; debug only. |
| `LORRAX_PHDF5_REQUIRE_MPI_WORLD` | `1` | machine | Stripped, case-insensitive `0 false no off` is off; any other value, blank included, is on. At the first collective open, compares `MPI_COMM_WORLD` size to `jax.process_count()`; a mismatch always refuses, and this knob makes an undeterminable world refuse (`1`) or warn (`0`). Without the probe a PMI-flavour mismatch gives every rank a private singleton world and unsynchronised writers on one file, with rc=0. |
| `LORRAX_PHDF5_SKIP_MPI_WORLD_CHECK` | off | debug | *bool*. Disables the world check above entirely; a debugging escape, never a remedy. |
| `LORRAX_HDF5_ONE_OWNER` | `measure` | guard | Trimmed, lower-cased enum `measure` or `strict`; any other value refuses naming both (`file_io/hdf5_owner.py::policy`). `measure` counts sequential cross-stack alternation on one file and reports it per path; `strict` refuses it. A live overlap with a writer refuses under both. Why: [`slab_io.md#one-owner`](../architecture/slab_io.md#one-owner). |
| `LORRAX_FORCE_REFIT` | unset | machine | *bool*. Forces the ζ fit even when `tmp/zeta_q.h5` is complete and its provenance matches (`gw/gw_init.py`). |

---

## 3. Debug and diagnostics

These leave production results unchanged unless the row says **A/B**; an A/B
switch marks the run as debug in its record. Stage-boundary telemetry and
refusals stay on by default; per-file and per-operation instruments are opt-in.

### 3a. Telemetry, logging and named overrides

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_DEBUG_PRINT` | `0` (off) | debug | *bool*; the one print-verbosity switch a production driver honours, read by Python (`runtime`) and the native layer alike. `1` enables rank-0 timing-section entry/exit (depth 3), cache-miss explanations, kmeans helper detail, GW memory probes, healthy HDF5 inventories, PHDF5 open/close/timing/shard/hint detail, per-dataset restart-write receipts, and native FFT/convolution/BLAS provider diagnostics. I/O errors, conditioning receipts and collective-payload receipts print regardless; forensic sidecars and diagnostics that do extra numerical work have their own rows. |
| `LORRAX_SLAB_IO_TIMING` | `0` | debug | *bool* (Python and C++ `env_flag`). Records every public SlabIO call's wall time in `slab_io_timing.rank<R>.log` plus a rank-0 per-file line on stderr; `write_enqueue` and `read_union_dispatch` time dispatch, not completion. Native per-file read/write counts, bytes and collective wall totals are captured after the close drain; no collectives are added. An enabled FFI run requires a provider exporting `lrx_phdf5_close_timed`. |
| `LORRAX_H5_JOURNAL` | `0` | debug | Three states: `1 true yes on` or empty is on, `0 false no off` is off, `sync`/`fsync` fsyncs every line; any other token refuses. Per-rank HDF5 operation journal (`file_io/h5_journal.py`), one line per open/close/create/read/write/attribute touch, written before the call at `file_io/hdf5_owner`, the `SlabIO` methods and `_slab_io_ffi`'s lifecycle calls. |
| `LORRAX_H5_JOURNAL_DIR` | unset (working directory) | debug | Where the journal and crash-ring files land. A directory that cannot be created disables the journal with one warning; the run continues. |
| `LORRAX_SANITY` | unset (`warn`) | guard | `0`/`off` skips the stage-boundary invariant checks (`common/sanity.py`); unset checks and warns without stopping; `strict` raises `SanityError` (CI and regression gates). Does not govern `sanity.refuse_nonfinite`. |
| `LORRAX_ALLOW_NONFINITE_RESULT` | unset (refuse) | debug | `1 true yes on` downgrades `common/sanity.py::refuse_nonfinite` (the refusal of a non-finite `kin_ion`, `Σ_total`, `E_qp` or `eqp` column about to be written) to a loud warning so the NaN artifact lands for forensics; anything else refuses. |
| `LORRAX_ALLOW_X64_OFF` | unset (refuse) | debug | *bool*. With 64-bit values resolved off the run refuses, at `set_default_env` (an explicit `JAX_ENABLE_X64=0`) and again on the live jax during startup; `1` continues as an announced uncertified run in which every result is f32/c64. |
| `LORRAX_ALLOW_TRS_VELOCITY_PARITY_BREAK` | `0` | debug | *bool*. Overrides the QSGW head-velocity time-reversal parity refusal (`gw/qsgw_head.py`). The gate is inactive when the two-component DFT-reference check finds broken TRS or gives no verdict. |
| `LORRAX_PPM_MEM_DIAG` | `0` | debug | Stripped, exactly `1`, `true` or `on`. Prints rank 0's device allocator state (`[ppm mem]`) at the GN-PPM fit stages, blocking on the named arrays so an async OOM is attributed to its stage, and host VmRSS/VmHWM (`[host rss]`) after the Σ(ω) executor, the head, the band extrapolation and the finalize sub-stages (`gw/ppm_sigma.py`); stderr. |
| `LORRAX_PPM_HERM_DIAG` | `0` | debug | Stripped, case-insensitive `1 true yes on`. Measures the PPM amplitude's inherited hermiticity residual over B_q and Ω_q at every q (`gw/ppm_sigma.py`); a report, not a gate. |
| `LORRAX_SIGMA_TAU_TIMING` | `0` | debug | *bool*. Per-stage blocking timing rows of the staged τ kernel (`gw/ppm_tau_kernel.py`: W/pole synthesis, G build, k convolution, band projection, accumulator, progress wait) on the host-driven shared-pole and sector W builders. Refuses on the resident pole route (GN/HL-PPM, elementwise MPA), which runs each window as one executable; profile that with `jax.profiler`. Refuses on bracketed face carriers. Numerics identical; wall time not comparable to the fused path. |
| `LORRAX_UNIFORM_RULE_TRACE` | unset | debug | Presence test: any non-empty value, `0` included, turns it on. Prints each MPA box plan's raw real support and final padded denominator box (`gw/sigma_box_plan.py`); changes no support, rule, cache key or value. |
| `LORRAX_DAV_MVSCAN` | unset | debug | Comma list of block widths (e.g. `1,2,4,8`). Times `apply_H` at each width inside the Davidson route (`bse/bse_lanczos.py`); changes no result. |
| `LORRAX_DAV_TRACE` | unset | debug | Path to an `.npz`. Process 0 writes the Davidson per-iteration history (iteration, matvec count, subspace size, eigenvalues, residuals, wall) and the distinct-program count (`bse/bse_lanczos.py`). |
| `LORRAX_W_RESIDUAL_CHECK` | `0` | debug | *falsy-set*. Prints the Dyson residual `‖(1−Vχ)W − V‖/‖V‖` on the first few q after a `w_dyson_solver = distributed` W solve (`gw/w_isdf.py`); adds one jit, so leave it off when taking collective-table probes. |
| `LORRAX_FFI_PROFILE` | off | debug | C++, exactly a leading `1`. Per-call timing split inside the cuSOLVERMp eigh handler (`cpp/cusolvermp/eigh_ffi.cc`). |
| `ISDF_JAX_PROFILE_DIR` / `ISDF_JAX_PROFILE_SECTIONS` | unset / unset (every section) | debug | Directory for `jax.profiler` traces of timed sections (`common/jax_profile.py`) / comma list of substrings selecting which sections open a trace. A session live across a multi-node PHDF5 collective write segfaults rank 0, so name the sections above P=4. |

### 3b. A/B switches, probes and test hooks

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_DEBUG_GN_ODD_RESIDUE_OFF` | `0` | debug, **A/B** | *bool*. On a measured-broken-TR GN-PPM or MPA fit, discards the anti-Hermitian frequency-odd residue (`D=0`, so `R+ = R- = B`); GN applies it at `fit_gn_ppm_from_wc_pair`, MPA when the ordered-residue fit is consumed. Prints `WARNING -- DEBUG`; refuses on a TRS or single-residue fit. |
| `LORRAX_DEBUG_SHARED_POLE_EVEN_PART` | unset | debug, **A/B** | `all` or `exclude_q0`; unset, `0`, `off`, `false`, `no` keep the production route; any other value refuses, as does a TRS store. On an ordered shared-pole store, Σ consumes W^even_q = [W_q + W_-q^T]/2 (`gw.mpa.sigma.shared_pole_even_part_kernel`); `exclude_q0` keeps the ordered W at q=0. Σ[W] − Σ[W^even] is the odd part. Prints `WARNING -- DEBUG`. |
| `LORRAX_DEBUG_SIGMA_MAX_TAU_DISPATCHES` | unset | debug | Positive integer; malformed or non-positive refuses. Stops the MPA Σ executor after that many post-prewarm τ node evaluations, synchronises for an honest wall time, prints seconds per node (and the phase table with `LORRAX_SIGMA_TAU_TIMING=1`), then exits rc=0 before any Σ/QP output. |
| `LORRAX_VQ_LR_GZ_TRIM` | `0` | debug, **A/B** | Exactly `1`. Trims the structurally dead G_z columns from the long-range v(q) design basis (`bse/vq_interp.py::lr_gset`; 337 → 161 columns on the MoS2 slab). Off by default because with it on the coverage null reads 5.8e-3 against a 1e-6 tolerance. |
| `LORRAX_SKIP_VQ_GATES` | `0` | debug | Exactly `1`. Skips the V_Q interpolation self-checks (`bse/vq_interp.py`), which exist because interpolation errors are silent. |
| `LORRAX_TRS_CHECK` | `1` | guard | Stripped, case-insensitive: `strict` also refuses a broken or inconclusive reference; `0 false off no` refuse (`GATE retired_LORRAX_TRS_CHECK_off`), since skipping the measurement would assert time reversal. Automatic two-component DFT-reference measurement before any TRS-dependent consumer (`symmetry_maps/density_symmetry_check.py`); consumers read only `WfnLoader.trs_holds` → `SymMaps.trs_allowed`. |
| `LORRAX_TRS_TOL` / `LORRAX_TRS_MAX_K` | `1e-6` / `12` | guard | Float / integer (`0` = all). Occupied-density residual tolerance and the maximum number of independent comparisons of that check. |
| `LORRAX_RHO_SYMMETRISE` | `1` | debug, **A/B** | Stripped, case-insensitive `0 off false no` is off; anything else on. Projects the valence density onto the WFN space group's invariant subspace before V_H is built (`psp/get_DFT_mtxels.py::symmetrize_valence_density`); the projector preserves ∫ρ. Off restores the raw accumulated ρ. |
| `LORRAX_EXTRA_MU_PAD` / `LORRAX_EXTRA_RANK_PAD` | unset (`0`) | test hook | Non-negative integer; negative or malformed refuses. Extra null rows on the μ axis (`runtime/padding.py`) / on the htransform Galerkin rank axis (`bandstructure/htransform.py::resolve_extra_rank_pad`) to prove pad-extent invariance: any result that moves at fixed P is a defect (`tests/test_pad_parity_gates.py`). Never in production. |
| `LORRAX_EXIT_AFTER_ZETA` | unset | debug | *bool*. Clean `SystemExit(0)` right after the ζ fit (`gw/gw_init.py`). |
| `LORRAX_MAX_RCHUNKS` | unset | debug | Integer ≥ 1; malformed or `< 1` refuses. Stops the ζ fit after N μ-batches (charge channel) or N real-space tiles (current channels) (`gw/isdf_fitting.py`), for profiling. A truncated ζ is still written but not marked complete and gets no `fit_provenance`, so it is never reused (`gw_config.ZETA_TRUNCATING_ENV_KNOBS`). |
| `LORRAX_ALLOW_PARTIAL_ZETA` | `0` | debug | Integer; non-zero permits reading a ζ file whose `zeta_is_done` flag is unset (`services/zeta_loader`). Forensics only. |
| `LORRAX_CHECK_REPLICA` | `0` | debug | *falsy-set*. Re-enables `jax.device_put`'s cross-process equality assertion in `lxkit.placement.device_put_process_local`, paying the P-linear all-gather the helper avoids (7.8 GB/rank at P=64) to verify a host table is replica-identical. |
| `LORRAX_WRITE_NO_JIT` | unset | debug | Dispatches the SlabIO writer's `shard_map` eagerly on the calling thread instead of through the jit wrapper (`file_io/_slab_io_ffi.py`); a probe for jit-argument buffer retention. |
| `LORRAX_LU_NO_PIVOT` | unset | debug, **A/B** | *bool*, parsed in C++ once per process; an unparseable value keeps pivoting and is announced. On disables pivoting in the cuSOLVERMp batched LU (`cpp/cusolvermp/batched_solve_lu_ffi.cc`). |
| `LORRAX_LU_DEBUG_DUMP` | off | test hook | Array sidecar written by `tests/bench/cusolvermp_solve_lu_test.py`; nothing else reads it. |
| `LORRAX_KFFT_CPU_TEST_XLA` | unset (`tests/conftest.py` sets `1`) | test hook | Exactly `1`. On a CPU mesh the k-convolution router (`ffi/fft.py`) takes an announced XLA FFT arm instead of the host plan handlers, because in-process pytest CPU meshes have no host FFI library; never read on CUDA. The router itself has no dial and keeps compiled mathdx images in `$SCRATCH/.cache/lorrax/kconv_mathdx` (else `~/.cache/lorrax/kconv_mathdx`). |
| `LORRAX_JAX_CACHE_FORCE_DIVERGE` | `0` | test hook | Integer N: every rank ≠ 0 pretends its N alphabetically last cache entries are missing; the agreement must drop them and say so, never hang. |
| `LORRAX_JAX_CACHE_NO_AGREE` | `0` | test hook | *falsy-set*. Shared cache directory with the agreement off: the deadlock reproducer. |
| `LORRAX_JAX_CACHE_KEYDUMP` | unset | test hook | Directory; at exit every rank writes the set of persistent-cache keys it asked about to `<dir>/rank{i:03d}_of{N:03d}.json` (atomic rename). Only the key set separates shared from private programs; read by `tests/test_jax_cache_contract.py`. |
| `LX_MESH4_MODE` | unset (auto) | test hook | `srun`, `local-cpu` or `none`: how `tests/mesh_launch.py` launches a four-process leg; auto picks `srun` whenever it is usable. |
| `LX_MESH4_DECKS` | `0` | test hook | `1` runs the cache contract's driver-deck arms even when only `local-cpu` is available. |
| `ISDF_COHSEX_TEST_PLATFORM` | `auto` | test hook | `cpu` or `gpu` forces the platform of the end-to-end gates (`tests/harness.py`). |

---

## 4. Build and launch scripts

Read by shell scripts under `config/` and `src/ffi/cpp/`, and by CMake, never
by the running Python; exporting one in a job script after the build does
nothing. Build legs: [`architecture/ffi_layout.md`](../architecture/ffi_layout.md).

`LORRAX_ROOT`, `LORRAX_SRC`, `LORRAX_VENV`, `LORRAX_SITE`,
`LORRAX_SITE_PACKAGES`, `LORRAX_INSTALL_ROOT`, `LORRAX_IMAGE`, `LORRAX_SIF`,
`LORRAX_SHIFTER*`, `LORRAX_MODULE*`, `LORRAX_FFI_STAGE*`,
`LORRAX_FFI_BUILD_DIR`, `LORRAX_FFI_SOURCES`, `LORRAX_FFI_HOST_*`,
`LORRAX_FFI_IMAGE`, `LORRAX_FFI_PYTHON`, `LORRAX_FFI_NO_CUDA`,
`LORRAX_FFI_HAVE_{PHDF5,CAL,CUBLASMP,CUFFT}`, `LORRAX_FFI_PLATFORM`,
`LORRAX_HOST_HAVE_{FFTW3,SCALAPACK}`, `LORRAX_CBLAS_{DIR,INCLUDE_DIR,LIBRARY}`,
`LORRAX_FFTW3_{INCLUDE_DIR,LIBRARY}`,
`LORRAX_MKL_{BLACS,THREAD_LIB,SCALAPACK_LIBRARY}`,
`LORRAX_FFI_ALLOW_DEFAULT_MPI`, `LORRAX_FFI_PHDF5_DIR*`,
`LORRAX_FFI_SLATE_DIR*`, `LORRAX_FFI_NVHPC_DIR*`, `LORRAX_FFI_FFTW_DIR*`,
`LORRAX_FFI_PHDF5_HOST`, `LORRAX_FFI_SLATE_HOST`, `LORRAX_FFI_NVHPC_HOST`,
`LORRAX_FFI_FFTW_HOST`, `LORRAX_FFTW_STAGE_CLOBBER`, `LORRAX_FFTW3_STAGE`,
`LORRAX_GATE_ONE_FFTW`, `LORRAX_GATE_FFTW_PY`, `CRAY_FFTW_PATH`,
`LORRAX_XLA_FFI_INCLUDE_DIR`, `LORRAX_XLA_FFI_HEADERS_DIR`, `LORRAX_HDF5_ROOT`,
`LORRAX_MPI_INCLUDE_DIR`, `LORRAX_MPICH_LIB_DIR`, `LORRAX_MPI_LIBRARY`,
`LORRAX_MPI_TYPE*`, `LORRAX_IMPI_ROOT`, `LORRAX_ICC_RUNTIME`, `LORRAX_MKL_ROOT`,
`LORRAX_SCALAPACK_LIBRARIES`, `LORRAX_NVHPC_*`,
`LORRAX_CUSOLVERMP_{STAGE,PIN}`, `LORRAX_CUBLASMP_PIN`, `LORRAX_SLATE_*`,
`LORRAX_HAVE_SLATE`, `LORRAX_HOST_HAVE_SLATE`, `LORRAX_DARSHAN_LIB_DIR`,
`LORRAX_XLA_CMDBUF`, `LORRAX_NNODES`, `LORRAX_NTASKS`, `LORRAX_NGPU`,
`LORRAX_SELECT_GPU`, `LORRAX_TIER2_WORKDIR`, `LORRAX_INPUT`,
`LORRAX_FFI_VERIFY_STRICT`, `LORRAX_BUILD_JOBS` (`8`),
`LORRAX_MPICH_CONTAINER_DIR` (`/opt/udiImage/modules/mpich`),
`LORRAX_SAPI_EXTRA_INCLUDES`, `LORRAX_{PY,VENV_DIR,SRC_DIR}`,
`LORRAX_OVERLAY_BUILD_DIR`, `LORRAX_MKL_LIB`.

`LORRAX_CUDA_CHECK`, `LORRAX_LIB_CHECK`, `LORRAX_CUSOLVERMP_CHECK`,
`LORRAX_CUBLASMP_CHECK` and the `LORRAX_CFG_` family are C macros or CMake
variables, not environment variables.

### 4a. Perlmutter launch and site configuration

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_NVHPC_SUBPATH` | `0.7.2_cuda12.9/math_libs/12.9/lib64` (`config/perlmutter/site_config.sh`) | launch | The cuSOLVERMp stage a run loads. It selects a communication path, not only a version, and every stage exports the same SONAME, so a mismatch links cleanly and warns about nothing: read [`ffi_layout.md` §4](../architecture/ffi_layout.md) before changing it. |
| `LORRAX_NVHPC_ROOT` / `LORRAX_NVHPC_MOUNT` | none; `build.sh` refuses when unset | build | The stage the `.so` is compiled against, and its bind-mount point (`/lorrax_nvhpc`). |
| `LORRAX_PLATFORM` | `gpu` (`run_shifter.sh`) | launch | `gpu`, or `cpu`/`host`. Decides `MPICH_GPU_SUPPORT_ENABLED` for the Shifter launch; on the CPU leg it must be 0 or Cray MPICH aborts in `MPI_Init_thread`. |
| `LORRAX_MPICH_GPU_SUPPORT` | derived from `LORRAX_PLATFORM` | launch | `0` or `1`; anything else refuses. Carries the value across Shifter's mpich module, which unsets `MPICH_GPU_SUPPORT_ENABLED`. |
| `LORRAX_PHDF5_MPI_STACK` | `mpich` | build | Which MPI the PHDF5 FFI links and loads (`run_shifter.sh`, CMake). |
| `LORRAX_GATE_ONE_MPI` | `on` (`cpp/gate_one_mpi.sh`) | launch | The one-MPI-implementation-per-address-space gate; `off` disables it and says so on every run. |
| `LORRAX_RUN_DIR` | none; mandatory in the Frontera template | launch | The run directory holding the input deck, beside `LORRAX_ROOT` and `LORRAX_INPUT`. It does not select the compile-cache location. |
| `LORRAX_PM_{PRGENV,MPICH,CMAKE,FFTW,HDF5,HDF5_DIR,LIBSCI,LIBSCI_DIR,LIBSCI_FLAVOUR,MPICH_DIR}` | `config/perlmutter/site_config.sh` | build | Perlmutter module versions and prefixes for PrgEnv, cray-mpich, cmake, cray-fftw, cray-hdf5 and cray-libsci, consumed by the host builders. |
| `LORRAX_MPIWRAPPER_ROOT_DEFAULT` / `LORRAX_MPIWRAPPER_PREFIX_DEFAULT` / `LORRAX_MPIWRAPPER_COMMIT_DEFAULT` / `LORRAX_MPIWRAPPER_ABI_DEFAULT` | `$HOME/software/lorrax_mpiwrapper_cray` / `…/current` / the pinned v2.11.1 commit / `2.10.0` | build | Immutable, content-addressed Perlmutter releases and the atomic active symlink of the unmodified Cray-MPICH MPIwrapper adapter; used by the builder and the CPU-MPI prelude. |
| `LORRAX_SLATE_{SRC,SCALAPACK_API_DIR}` | `cpp/stage/slate_build_scalapack_api.sh` | build | SLATE ScaLAPACK-API overlay build: source tree (mandatory) and output prefix. |
| `LORRAX_MPIWRAPPER_SO` | `$LORRAX_MPIWRAPPER_PREFIX/lib64/libmpiwrapper.so` on Perlmutter | launch | The Perlmutter prelude verifies the adjacent pinned-source/MPI-ABI/SHA256 manifest before exporting it as `MPITRAMPOLINE_LIB`. |
| `LORRAX_SLATE_HOST_LIB` | staging default | build | Host SLATE library path for the link line. |

### 4b. Frontera staging and MPI build scripts

Read by `config/frontera/stage_runtime.sh`, `build_cpu_runtime_bundle.sh`,
`build_mpiwrapper.sh`, `build_mpi_overlay.sh` and `ffi_env.sh` at job launch.

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_BUNDLE` | unset | launch | Path to `lorrax_cpu_bundle.tar` for node-local staging; unset or missing is an announced fallback to the shared filesystem. |
| `LORRAX_BUNDLE_OUT` / `LORRAX_BUNDLE_STRIPE` | `$SCRATCH/lorrax_bundle` / script default | build | Where the bundle is written, and its Lustre striping. |
| `LORRAX_STAGE` | `1` | launch | `0` disables node-local staging, announced. |
| `LORRAX_STAGE_ROOT` | `/tmp/lorrax_stage.$UID` | launch | Node-local extraction directory. |
| `LORRAX_STAGED` / `LORRAX_STAGE_S` / `LORRAX_OVERLAY_DIR` | outputs | launch | Set by `stage_runtime.sh`: staged or not, seconds spent, the resolved overlay directory. |
| `LORRAX_VENV_FALLBACK` / `LORRAX_OVERLAY_FALLBACK` / `LORRAX_SRC_FALLBACK` | shared-filesystem paths | launch | Where to run from when staging is off or fails. |
| `LORRAX_OVERLAY` / `LORRAX_OVERLAY_PREFIX` | `$WORK/lorrax_env_mpi_overlay/site` / `$WORK/lorrax_env_mpi_overlay` | build | The MPI overlay site the bundle packs, and `build_mpi_overlay.sh`'s install prefix. |
| `LORRAX_MPIWRAPPER_REPO` / `LORRAX_MPIWRAPPER_COMMIT` | upstream MPIwrapper, pinned v2.11.1 commit | build | What `build_mpiwrapper.sh` fetches. |
| `LORRAX_MPIWRAPPER_STAGE` / `LORRAX_MPIWRAPPER_SRC` / `LORRAX_MPIWRAPPER_BUILD` / `LORRAX_MPIWRAPPER_PREFIX` | Frontera `$WORK/lorrax_mpiwrapper/…` | build | Frontera stage, source, build and install paths. Perlmutter's builder owns `$LORRAX_MPIWRAPPER_ROOT/{stage,releases,current}` and does not accept these redirections. |
| `LORRAX_MPIWRAPPER_ROOT` | `$LORRAX_MPIWRAPPER_ROOT_DEFAULT` (Perlmutter) | build | Root holding the builder sentinel, fresh candidates, releases and the `current` symlink. |
| `LORRAX_MPIWRAPPER_REFERENCE_SO` | unset | build | Optional reference `.so` for the build-note comparison. |
| `LORRAX_CMAKE` | `command -v cmake` | build | The cmake `build_mpiwrapper.sh` uses. |
| `LORRAX_FFI_SO_PHDF5` | `$LORRAX_FFI_STAGE/build_phdf5/liblorrax_ffi.so` | launch | The PHDF5-enabled CUDA FFI `.so` that `ffi_env.sh` exports as `LORRAX_FFI_SO`. |
| `LORRAX_MPI_PROVIDER` | `auto` | launch | `auto` unsets `FI_PROVIDER` and `FI_TCP_IFACE` so Intel MPI picks the native provider (`mlx` on CLX); `tcp` pins IPoIB with `FI_TCP_IFACE=ib0` (the rtx/mlx4 escape); any other value requests that provider (never `verbs` at P ≥ 144). The unset is load-bearing: TACC's default impi module exports `FI_PROVIDER=mlx` into every shell. |
| `LORRAX_MPI_FABRICS` / `LORRAX_PMI2_LIB` | `shm` / `$WORK/host_pmi/libpmi2.so.0` | launch | `ffi_env.sh` overrides for `I_MPI_FABRICS` (use `shm:ofi` for multi-node) and `I_MPI_PMI_LIBRARY`. |

---

## 5. External variables LORRAX sets or reads

| var | LORRAX's handling |
|---|---|
| `JAX_ENABLE_X64` | `runtime.set_default_env` `setdefault`s `1`, and `runtime.set_x64_on_imported_jax` pushes the resolved value onto a jax imported earlier. A resolved `False` refuses unless `LORRAX_ALLOW_X64_OFF` (§3a). |
| `JAX_PLATFORMS` | `setdefault` `cuda,cpu`, or hard-set to `cpu` by `set_default_env(platform="cpu")`; `ffi_loader.platform_from_env` reads it to pick the FFI library without initialising a backend. |
| `JAX_PLATFORM_NAME` | jax's deprecated spelling; `runtime` pops it at both CPU-downgrade sites, and `psp/get_DFT_mtxels.py` counts either spelling as the caller choosing a platform. |
| `JAX_PROCESS_COUNT` → `JAX_NUM_PROCESSES` → `SLURM_NTASKS` → `1` | Process-count resolution chain (`runtime`). |
| `JAX_PROCESS_INDEX` → `SLURM_PROCID` → `0` | Process-index chain. |
| `JAX_COORDINATOR_ADDRESS` | Overrides the coordinator derived from `SLURM_STEP_NODELIST` / `SLURM_NODELIST`, then `SLURMD_NODENAME` / `HOSTNAME`. |
| `_LORRAX_JAX_DISTRIBUTED_DONE` | LORRAX's idempotency sentinel for `jax.distributed.initialize`; self-set, not a knob. |
| `CUDA_VISIBLE_DEVICES` | Read to derive local device ids; `tests/conftest.py` rewrites it per xdist worker. |
| `SLURM_NNODES` | Read by `gw/gw_init.py` for the ranks-per-node estimate of a local-backend capacity check (malformed → 1). |
| `SLURM_JOB_ID`, `SLURM_STEP_ID` | Recorded in response-bank and Galerkin receipts. |
| `JAX_EXPLAIN_CACHE_MISSES` | `0` (off): JAX's own switch; `1` prints JAX's per-trace cache-miss explanations. `LORRAX_DEBUG_PRINT` does not turn it on. |
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION` | Multi-process CPU runs require `mpi`; `runtime.announce_cpu_collectives()` refuses any other backend. Why: [`environment/transports.md`](../environment/transports.md). |
| `MPITRAMPOLINE_LIB` | MPItrampoline's adapter, absolute, set before JAX import: Frontera's patched Intel-MPI build, Perlmutter's unmodified upstream build from `config/perlmutter/build_mpiwrapper.sh`. A vendor `libmpi.so` is not an adapter. Contract: [`mpi_collectives.md`](mpi_collectives.md). |
| `LD_PRELOAD` | Perlmutter CPU-MPI: `config/perlmutter/cpu_mpi_env.sh` prepends `/opt/cray/pe/lib64/libpmi.so.0` and verifies it resolves under the Cray tree; foreign MPI, MPItrampoline/MPIwrapper and `libpmi2` entries refuse. The startup report records whether `libpmi.so.0` is preloaded. |
| `MPICH_ASYNC_PROGRESS` | The Perlmutter CPU-MPI prelude sets `1`, so Cray MPICH grants `MPI_THREAD_MULTIPLE` to XLA's FUNNELED request with one progress thread per rank; the prelude unsets `MPIR_CVAR_ASYNC_PROGRESS`. |
| `LORRAX_MPI_FORCE_THREAD_MAIN`, `LORRAX_MPI_FINALIZE_FIX` | Frontera MPIwrapper/overlay controls, unset in production; `runtime` reads both only to print them in the startup report. `common.collectives.warm_mesh_cliques()` creates every communicator from the main thread, and drivers using `runtime.finalize_process()` need no finalize overlay. |
| `SLATE_SCALAPACK_TARGET` | SLATE's own dial, read by `blacs_grid.h` to announce the demotion it controls: unset means `HostTask`, so a CUDA-built SLATE runs on the CPU unless it is `devices`. Relevant only with `LORRAX_SCALAPACK_ALLOW_SLATE_API`. |
| `OPENBLAS_THREAD_TIMEOUT` | `setdefault` `1` by `LORRAX_BLAS_TUNE` (§2a) at import of `runtime`; read once by OpenBLAS at the first `import numpy`, which is why every entry point imports `runtime` first. |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | Read for the startup report's thread table and oversubscription warning; LORRAX sets none of them. On Frontera the XLA:CPU thread pool ignores OMP; `taskset` is the pinning mechanism. |
| `XDG_CACHE_HOME` | Last-resort base for the JAX compile cache (§2e). |
| `SCRATCH` | Base of the k-convolution mathdx image cache (§3b); read, never set. |
| `HDF5_USE_FILE_LOCKING` | `runtime.set_default_env` `setdefault`s `FALSE` before any store opens, and `file_io/hdf5_owner` reports the value. It governs only the serial h5py paths (the MPI-IO VFD takes no POSIX locks); Frontera `/work2` mounts node-local `localflock`, where cross-node locking is incoherent. |
| `MPLBACKEND` | `setdefault` `Agg` for headless plotting. |
| `FI_PROVIDER` | Not read by LORRAX. On Frontera CLX leave it unset (`LORRAX_MPI_PROVIDER=auto`) so Intel MPI picks `mlx` (provider costs: [transports §3](../environment/transports.md#3-the-intel-mpi-provider-layer-frontera)). `fi_info` falsely reports `mlx` unavailable; trust the `libfabric provider:` line instead. In apptainer never `--bind /dev`. |
| `FI_PROVIDER_PATH` | Harnesses and `ffi_env.sh` pin `$IMPI/libfabric/lib/prov`; required in-container, where `mpivars.sh` is not sourced and `PMPI_Init` otherwise finds no provider. |
| `I_MPI_FABRICS` | Harnesses export `shm:ofi` (Intel MPI's own default, guarding against an inherited value). |
| `I_MPI_PMI_LIBRARY` | Required under `srun --mpi=pmi2` and set unconditionally there: TACC's login environment exports a PMI-1 library that is wrong for pmi2 and absent in the container. |
| `I_MPI_DEBUG` | `4` in every harness; the rank-0 `libfabric provider:` line is the only trustworthy provider observable, and it costs nothing after init. |
| `UCX_*` (`UCX_TLS` and the RC/DC/UD MLX5 timeout and retry settings) | Harnesses and `ffi_env.sh` `setdefault` the six TACC impi-module values (`UCX_TLS=knem,dc_x,rc` and the retry/timeout bumps); stripping them doubles a 1 MiB 32-rank allreduce (419 → 799 µs). Inherited values win; do not hard-pin (rtx has no `dc_x`). |
