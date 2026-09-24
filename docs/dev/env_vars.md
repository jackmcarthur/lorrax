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
| `LORRAX_ZETA_RIDGE` | `zeta_ridge` (`0`) | twin | float | Additive Tikhonov ridge on `C_q` before the Cholesky ζ factorization. |
| `LORRAX_SC_MAX_ITER` | `sc_max_iter` | twin | int | Self-consistency iteration cap. |
| `LORRAX_SC_TOL_EV` | `sc_tol_ev` | twin | float | Self-consistency convergence tolerance, eV. |
| `LORRAX_SC_ACCEL` | `sc_accelerator` (`anderson`) | twin | stripped, lower-cased string | Self-consistency accelerator; `SCConfig` refuses every value but `anderson`. |
| `LORRAX_SC_DEPTH` | `sc_history_depth` (`20`) | twin | int | Accelerator history depth; `< 1` refuses. |
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
| `LORRAX_MIXEDPREC_ALLOW_TF32` | unset (guard on) | routing-affecting | Exactly `1` (no strip, no case folding); every other value leaves the guard on. Bypasses the complex64 BSE solver's refusal of an unpinned fp32 matmul precision (`bse/w_ladder_mixedprec.py::_refuse_unpinned_matmul_precision`), so the solve may run at TF32 with a higher refinement-residual floor; A/B measurement only. |

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
| `LORRAX_CPU_SKIP_GPU_PLUGINS` | `1` | machine | *bool*. On a CPU-only run (`JAX_PLATFORMS=cpu`, or no NVIDIA device on the node) skips jax's CUDA plugin discovery, which costs ~77 s on a cold Frontera node; `0` re-enables it and records a demotion. A run with a GPU present is unaffected. |
| `LORRAX_MATMUL_PRECISION` | unset (`highest`) | numerics | `highest` or `float32`; any other token refuses naming the variable, `high` included (it selects a 3-pass TF32 decomposition on XLA:GPU). Sets `jax_default_matmul_precision` at `bootstrap()`; left at XLA's default every f32 and c64 dot runs at TF32 (forward error 1.9e-4 against 3.2e-7 pinned on the BSE ladder matvec). Pinning is free at block width 1 and costs 8 % at 2 and 46 % at 4. |
| `LORRAX_FAILFAST` | `1` | machine | *bool*. At P>1 an uncaught exception on one rank prints a rank-tagged banner and calls `os._exit(1)`, so the job fails instead of its peers hanging in a collective; the CLI bootstrap likewise aborts the step. `0` disables both; `SystemExit(0)` stays a clean exit. |
| `LORRAX_MALLOC_TUNE` / `LORRAX_MALLOC_MMAP_MB` / `LORRAX_MALLOC_TRIM_MB` | on / `1` / `128` | machine | *bool* / integer MiB / integer MiB. At bootstrap `runtime.tune_glibc_malloc` sets glibc `M_MMAP_THRESHOLD` and `M_TRIM_THRESHOLD` so freed XLA:CPU transients return to the OS (≤ 4 % wall); a tuning that does not arm is a demotion. |
| `LORRAX_BLAS_TUNE` | `1` | machine | *bool*. At import of `runtime`, before numpy, `setdefault`s `OPENBLAS_THREAD_TIMEOUT=1` so idle OpenBLAS workers stop spinning; an exported value wins, and numpy imported first records a `blas_tune` demotion (`tests/test_runtime_blas_env.py` enforces the import order). Spinning made a 140×140 `cho_factor` between Python work 28.7 ms against 0.067 ms standalone; the setting costs 22 % only in a tight back-to-back BLAS loop, which no driver is. |

### 2b. Memory caps, collective chunking, schedules

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_COLLECTIVE_CHUNK_MB` | `128` | resource cap | Float MiB; a malformed value uses the default; `≤ 0` is unbounded (reproduction only). Upper bound on one emitted collective's payload, enforced as a host-level loop XLA cannot fuse back: ζ-fit collectives (`isdf/core.py`), the distributed W Dyson A-build (`gw/w_isdf.py`), the rank-0 owner gather of k-partitioned sweeps (`common/collectives.py::gather_indexed_blocks_to_owner`), and the V_q G-panel width (`gw/v_q_g_flat.py`). A single 1.15 GB all-gather was fatal on Gloo at P=144; on MPI at P=16 the cap costs nothing measurable. Orthogonal to the live-bytes cap below. |
| `LORRAX_ZETA_GATHER_CAP_GIB` | `4` | resource cap | Float GiB, read at import of `isdf/core.py`. Budget for the ζ back-solve's gathered-factor transient, and the `auto` boundary between the two whole-tile back-solve tiers: `local` (each factor stays on its q owners and only the right-hand side moves) whenever `ceil(nq/P)·P ≤ 2·nq` or the stack exceeds the cap, otherwise `replicated`. |
| `LORRAX_ZETA_QPARALLEL` | unset (`auto`) | schedule | Blank or `auto`: fold when P>1, nq ≥ 2 and nq·μ³ ≥ 5e9; otherwise *bool* (`1` forces the fold, `0` the all-ranks execution). Schedule of the replicated charge ζ factor: the fold scatters q over all devices, each factoring whole per-q tiles, and reshards back; same plan, same bits (`tests/test_zeta_mesh_invariance.py`). Unfolded, P=16 at nq=10, μ=2979 spent 105 s in one dense eigh per q on every rank. |
| `LORRAX_PPM_FIT_ARENA_GIB` | `8` | resource cap | Float GiB, read at import. Temp-arena budget of the GN-PPM fit's q-chunk loop (`gw/minimax_screening.py::_gn_ppm_fit_q_block`): sizes `q_block`, and `q_block ≥ nq` takes the single-shot path. Bit-exact by construction; arena 74.3 → 4.6 GiB at μ=24,933, P=64. |
| `LORRAX_KIN_ION_LOOKAHEAD` | `2` | schedule | Positive integer; a malformed value refuses naming the variable. Host-ahead-of-device depth of `common/collectives.py::sweep_local_k`; `1` serialises, the control for measuring the overlap. |
| `LORRAX_GRAM_COL_BLOCK` | unset (auto) | resource cap | Falsy tokens (`""`, `0`, `false`, `no`, `off`) select auto; a positive integer pins the width (floor 256, aligned to both mesh axes); anything else refuses. Tile width of the pivoted-Cholesky Gram build (`centroid/pivoted_cholesky.py::build_gram_q0_via_loadwfns`); auto picks the largest width whose AOT-measured live set fits, full width when the whole Gram fits. Workspace only; the contraction order and the `P('x','y')` Gram are unchanged. |
| `LORRAX_FACE_TO_BATCH_ROUTE` | unset (`staged_reshard.DEFAULT_ROUTE`) | schedule | A `common.staged_reshard` route name for the fH_q face→batch move (`bandstructure/bse_setup.py::resolve_reshard_route`); a caller's `reshard_route` argument wins. An unknown token is announced (`*** LORRAX SANITY`) and the default runs. Movement only, value-identical. |
| `LORRAX_BSE_MATVEC_OPT` | unset | schedule | Comma set; the only token is `gspmd`, and an unknown token refuses (`bse/bse_stack_matvec.py::matvec_opts`). `gspmd` builds the BSE W term with no manual `shard_map`, so XLA's SPMD partitioner picks the collectives: an audit route, not a default. |
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
| `LORRAX_FFTW3_SO` | unset (the candidate ladder) | machine | Path of the file the host FFT engine is `dlopen`ed from (`ffi/cpp/mklfft/fft_flat_k_ffi.cc::fftw3_candidates`, at the first FFT), tried ahead of the build's compile-time hint and of `libfftw3.so.3`, `libfftw3.so.mpi31.3`, `libmkl_rt.so`, `libfftw3.so`. A bad path is skipped silently; no engine at all refuses at startup naming every candidate. Nothing at run time checks that the named file is a CPU FFTW3 (a GPU FFTW-API shim exports the same three symbols); the stage-time check is `ffi/cpp/gate_one_fftw.sh`. |
| `LORRAX_MKLBLAS_THREADS` | unset (`auto`) | machine | `auto` = ambient `omp_get_max_threads()`, `off` = 1, an integer pins; strict full-string match, an unrecognised value is announced on stderr and becomes `auto`. Thread-local BLAS team of the GEMM handler call (`cpp/mklblas/gemm_batch_ffi.cc`; a no-op on non-MKL BLAS). |
| `LORRAX_FFT_FFI_THREADS` | unset (`auto`) | machine | Same grammar, integer 1–4096. OpenMP team of the flat-k FFT chunk loop (`fft_flat_k_ffi.cc::team_threads`). `LORRAX_MKLFFT_THREADS` is a deprecated alias, announced once; this spelling wins when both are set. |
| `LORRAX_FFT_FFI_CHUNK` | unset (auto) | machine | Positive integer or auto; an unrecognised value is announced and auto runs. Trail elements per FFT chunk; auto sizes the per-thread buffer (`nk·chunk·16 B`) to ~512 KiB so it stays in L2 (the strided form runs 2.8× slower single-thread). `LORRAX_MKLFFT_CHUNK` is a deprecated alias, announced once. |
| `LORRAX_SCALAPACK_MKL_THREADS` | unset (`auto`, cap 4) | machine | Case-insensitive `auto`, `off`/`0`, or an integer; an unrecognised value is announced and becomes `auto`. MKL team size pinned inside the ScaLAPACK `eigh` and `solve_lu` handlers: `auto` caps it at 4 when the global setting is larger, `off` inherits `MKL_NUM_THREADS`. pzheevd at n=2448 on a 12×12 grid runs 11.3 s/q at 14 threads against 0.46 s/q at 4. |
| `LORRAX_SCALAPACK_ALLOW_SLATE_API` | off | machine | Standard boolean spellings, case-insensitive; a malformed value is announced and stays refused. Waives the refusal of SLATE's `libslate_scalapack_api` overlay answering the pzheevd/pzgetrf symbols (`cpp/scalapack/blacs_grid.h`); see `SLATE_SCALAPACK_TARGET` (§5). |

### 2e. Compile cache

The `LORRAX_JAX_CACHE_` rows use the *falsy-set* grammar, so a blank value
turns a default-on switch off. Read in `common/jax_compile_cache.py`.

| var | default | class | grammar and effect |
|---|---|---|---|
| `ISDF_JAX_CACHE_DIR` | unset (`$LORRAX_RUN_DIR/.lorrax_jax_cache`; without a run directory `$SCRATCH/lorrax_jax_cache`, then `$XDG_CACHE_HOME/isdf_jax_compilation`) | machine | A non-empty value overrides every derived location; blank or whitespace opts out. Sequential drivers share one `{base}/np{P}` directory. Launchers should set `LORRAX_RUN_DIR`: the startup agreement over a large shared cache costs 7–15 s. |
| `JAX_COMPILATION_CACHE_MAX_SIZE` | `-1` (unlimited) | external | JAX's own control; `0` disables the cache. A positive cap (JAX's LRU eviction) is supported only at P=1 and refuses at P>1, where LORRAX freezes an agreed all-rank entry set at startup. |
| `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` | `1.0` | external | JAX's default and user override, honoured; `0` only in a cache-contract experiment. |
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
| `LORRAX_PHDF5_COLLECTIVE_WRITES` | `1` | machine | *bool* (C++ `env_flag`, the same table). Collective two-phase writes; `0` writes independently. A strided `V_qmunu` tile at P=144 decomposes into 4.1 M × 3.2 kB independent writes, and independent writes at 16×4M striping fell to 0.068 GiB/s at 4 nodes, while collective stayed within ~10 % of the best at every geometry. |
| `LORRAX_PHDF5_DEDUP_REPLICAS` | `1` | machine | One canonical writer per distinct hyperslab when a mesh axis is replicated. `0` lets every replica write, which is undefined behaviour under collective MPI-IO; debug only. |
| `LORRAX_PHDF5_REQUIRE_MPI_WORLD` | `1` | machine | Stripped, case-insensitive `0 false no off` is off; any other value, blank included, is on. At the first collective open, compares `MPI_COMM_WORLD` size to `jax.process_count()`; a mismatch always refuses, and this knob makes an undeterminable world refuse (`1`) or warn (`0`). Without the probe a PMI-flavour mismatch gives every rank a private singleton world and unsynchronised writers on one file, with rc=0. |
| `LORRAX_PHDF5_SKIP_MPI_WORLD_CHECK` | off | debug | *bool*. Disables the world check above entirely; a debugging escape, never a remedy. |
| `LORRAX_HDF5_ONE_OWNER` | `measure` | guard | Trimmed, lower-cased enum `measure` or `strict`; any other value refuses naming both (`file_io/hdf5_owner.py::policy`). `measure` counts sequential cross-stack alternation on one file and reports it per path; `strict` refuses it. A live overlap with a writer refuses under both. Why: [`slab_io.md#one-owner`](../architecture/slab_io.md#one-owner). |
| `LORRAX_FORCE_REFIT` | unset | machine | *bool*. Forces the ζ fit even when `tmp/zeta_q.h5` is complete and its provenance matches (`gw/gw_init.py`). |

---

## 3. Debug / diagnostic env

These do not change production defaults.  The one entry explicitly labelled
**physics A/B switch** changes results only when set and loudly marks the run
as debug-only.  Stage-boundary telemetry stays on where its
absence has already cost 72-node hours (the AC/AF.4c observability failures).
Per-file and per-operation HDF5 diagnostics are opt-in because they are
incident instruments, not production results.

### Rows moving to §3 (being rewritten)

| var | default | effect |
|---|---|---|
| `LORRAX_MAX_RCHUNKS` | unset | Ceiling on the r-chunk count the planner may pick (`gw/isdf_fitting.py`).  Memory/perf, not numerics — but chunking is load-bearing at large μ (scale ladder). | |
| `LORRAX_PPM_MEM_DIAG` | `0` | Diagnostic only. `1` prints rank 0's device allocator state (`[ppm mem]`) at the GN-PPM fit stages (`gw/ppm_sigma.py::fit_ppm`, blocking on the named arrays so an async OOM is attributed to its stage) and host VmRSS/VmHWM (`[host rss]`, `gw/ppm_sigma.py::host_rss_diag`) after the Sigma(omega) executor, the head, the band extrapolation and the two finalize sub-stages. Both go to stderr. | CrI3 16x16 GN-PPM SC (2026-09-22/23): attributed the fit OOM and the post-Sigma host OOM. |
| `LORRAX_DAV_MVSCAN` | unset (→ off) | Comma list of block widths (e.g. `1,2,4,8`).  DIAGNOSTIC, off by default: times `apply_H` at each width inside the Davidson route so the block-width choice can be measured rather than assumed (`bse/bse_lanczos.py`).  Landed with the 2026-08-08 Davidson-competitiveness lane; it only times, it changes no result. |
| `LORRAX_DAV_TRACE` | unset (→ off) | Path to an `.npz`.  DIAGNOSTIC, off by default: dumps the Davidson per-iteration history (iter, matvec count, subspace size, eigenvalues, residuals, wall) plus the distinct-program count, written on process 0 only (`bse/bse_lanczos.py`).  Landed with the 2026-08-08 Davidson-competitiveness lane; read-only instrumentation. |
| `LORRAX_VQ_LR_GZ_TRIM` | unset (→ off, `=1` opts in) | Trim the structurally dead G_z columns out of the long-range v(q) design basis before the fit (`bse/vq_interp.py::lr_gset`).  On the MoS2 slab 176 of 337 columns are dead by construction (`max|A[:, dead]| = 0.000e+00`) and nG drops 337 → 161.  **DEFAULT OFF, and it must stay off until the null is understood**: with the trim on, `run_nulls` reads `F_own_rebuild_vs_cleaned_LR_tile_max = 5.793e-03` against a `1e-6` tolerance (FAIL), where the default reads `6.004e-11` (OK).  At the default the code reproduces pre-trim `013aad92` exactly.  Landed by the 2026-08-08 vq-interp lane; see FIX_vq_interp.md.  **Since 2026-08-10** the trim follows `lr_fit_degrees` — the energy cutoff with a two-shell floor (`docs/architecture/decisions.md`) — rather than `DEG_B26P`'s keys, so on a thick slab it now keeps every channel the model actually fits instead of a fixed four.  That widens what the trim retains; it does not change this knob's default, and the coverage-null question above is untouched by it. |
| `LORRAX_KFFT_CPU_TEST_XLA` | unset (tests/conftest.py sets `1`) | TEST-ONLY.  On a **cpu** mesh the k-convolution router (`ffi/fft.py`, decisions.md 2026-09-24) takes an announced `jnp.fft` arm instead of the host plan handlers, because in-process pytest cpu meshes on Perlmutter have no host FFI library.  Never read on CUDA; never a production route.  The router itself has **no dial**: it picks nvidia-mathdx on CUDA and the plan route on cpu from the mesh platform, and keeps its compiled images in `$SCRATCH/.cache/lorrax/kconv_mathdx` (else `~/.cache/lorrax/kconv_mathdx`; `SCRATCH` is read, never set).  `LORRAX_FFT_FFI_FUSED`, `LORRAX_CONV_KMINOR_FFI` and `LORRAX_CONV_KLEAD_FFI` were deleted with the engines they chose between (2026-09-24). |
| `LORRAX_JAX_CACHE_FORCE_DIVERGE` | `0` | TEST HOOK / positive control: every rank != 0 pretends its N alphabetically-last cache entries are missing. The agreement must drop them and say so; the run must NOT hang. |
| `LORRAX_JAX_CACHE_NO_AGREE` | `0` | TEST HOOK: shared dir with the agreement layer OFF — the naive design, i.e. the deadlock reproducer. Never use in production. |
| `LORRAX_JAX_CACHE_KEYDUMP` | unset | A directory into which **every rank** writes the SET of persistent-cache keys it asked about, `<dir>/rank{i:03d}_of{N:03d}.json`, at exit (tmp+rename, so a reader never sees a short file). This is what makes the cache contract's SYMMETRY arm falsifiable: `xla_compiles` and `vetoed` are per-rank COUNTS, and four ranks that each compiled a private program report exactly the same counts as four ranks that shared one — only the key set separates them. Cache-miss logging under `LORRAX_DEBUG_PRINT` names only keys that MISSED and therefore prints nothing when every rank hits its own private entries. Consumed by `tests/test_jax_cache_contract.py` via `tests/mesh_launch.py::read_keydumps`; the same list is on `compile_cache_stats()["keys"]` for in-process probes. Diagnostic file output only — it changes no cache decision. |
| `LX_MESH4_MODE` | unset | Test-harness knob (`tests/mesh_launch.py`): pins how a four-PROCESS leg is launched — `srun` (inside a Slurm allocation), `local-cpu` (four local processes wired by an explicit jax coordinator, CPU backend), or `none`. Unset auto-detects, and `srun` always wins when it is usable so the CPU emulation can never quietly stand in for the real P=4 leg. |
| `LX_MESH4_DECKS` | `0` | Test-harness knob: run the cache contract's full DRIVER-DECK arms even when the only available launch is `local-cpu`. Off by default because a CPU mesh is fine for device-count logic and never substitutes for the P=4 leg on a GPU path (`AGENT_PREAMBLE.md`, the four-GPU rule), so off-cluster the deck arms skip with that reason while the red twins still run. |
| `LORRAX_PHDF5_MPI_STACK` | `mpich` | Build+launch: which MPI the phdf5 FFI links/loads (`run_shifter.sh:42`). |
| `LORRAX_ALLOW_PARTIAL_ZETA` | `0` | Permits reading a ζ file whose `zeta_is_done` flag is unset (`services/zeta_loader/src/zeta_loader/loader.py`).  Forensics-only: a half-written ζ is otherwise indistinguishable from a complete one (QUALITY_PATTERNS #7). |

### 3a. Driver telemetry and diagnostic logging

| var | default | effect |
|---|---|---|
| `LORRAX_DEBUG_PRINT` | `0` (off) | **The only print-verbosity switch honored by a production driver.** `1` enables one driver debug stream: rank-0 timing-section entry/exit (fixed depth 3), cache-miss explanations, kmeans helper detail, GW memory/r-chunk probes, healthy HDF5 inventories, PHDF5 open/close/timing/shard/hint detail, per-dataset restart-write receipts, and native FFT/convolution/BLAS provider diagnostics. `0` leaves a quiet production stream; storage-library success chatter is absent while actual I/O errors remain unconditional. Numerical conditioning and collective-payload receipts also remain unconditional because they report stability/scaling invariants rather than verbosity. This switch does not enable forensic sidecars/dumps or diagnostics that execute extra numerical work; those remain separately named below. Canonical `runtime.env_flags.env_bool` grammar. |
| `LORRAX_SLAB_IO_TIMING` | `0` (off) | Optional SlabIO service timing. Set `1` before the run to record every public SlabIO call's API wall time in `slab_io_timing.rank<R>.log`, plus a compact rank-0 per-file line on stderr. `write_enqueue` and `read_union_dispatch` measure dispatch, not completion; native per-file H5Dread/H5Dwrite counts, selected bytes and collective wall totals are captured after the existing close drain. Compare rank logs for skew; no timing collectives are added. Enabled FFI runs require a provider exporting `lrx_phdf5_close_timed`; ordinary runs work with older providers. The serial emulated tier has API times only. Off creates no sidecar and does not time calls. |
| `LORRAX_H5_JOURNAL` | `0` (off) | Opt-in per-rank HDF5 operation journal (`file_io/h5_journal.py`): one line per open/close/create/read/write/attr touch, written BEFORE the call and line-buffered at the three existing choke points (`file_io/hdf5_owner.note_open`/`note_close`, the `SlabIO` methods, and `_slab_io_ffi`'s lifecycle calls). A normal production run creates no `h5_journal.rank<R>.log` sidecars. **Grammar: three states, not a boolean** — `1`/`true`/`yes`/`on`/empty = on, `0`/`false`/`no`/`off` = off, `sync`/`fsync` = fsync after every line. An unrecognised token REFUSES naming the variable. Set `1` while diagnosing native HDF5 failures; set `sync` for segfault-grade capture. It writes forensic evidence only; healthy stdout inventory follows `LORRAX_DEBUG_PRINT`. |
| `LORRAX_H5_JOURNAL_DIR` | unset (→ `os.getcwd()`) | Where the journal and crash-ring files land (`file_io/h5_journal.py`). The default is the process's working directory, which is the run directory for every LORRAX driver launch. A directory that cannot be created disables the journal with ONE warning and lets the run continue — an instrument, never a gate. |
| `LORRAX_HDF5_ONE_OWNER` | `measure` | One-HDF5-library-instance-per-open-file policy (`file_io/hdf5_owner.py`, audit A1 / sandbox claims/0110): `measure` counts sequential cross-stack alternation; `probe()` is silent while safe and always reports `UNSAFE-BY-A1`. `strict` also turns that measured unsafe condition (two mapped libhdf5 objects AND a file written through both) into a refusal. The LIVE-overlap-with-a-writer refusal is unconditional under both. Any other token REFUSES naming the variable. **Added to this page 2026-08-15** — it was read through a module constant, which `tools/env_audit.py` and `tests/test_env_registry.py` cannot resolve, so it shipped unregistered and the gate stayed green; the read is now a literal and the gate covers it. |
| `LORRAX_SANITY` | unset = warn | Stage-boundary invariant checks (`common/sanity.py`): `0`/`off` skips all (escape hatch); default checks and **warns loudly but keeps running** (a false positive must never kill a 40-node job); `strict` raises `SanityError` — set in CI / regression gates. **It does not govern `sanity.refuse_nonfinite`** — see the row below. |
| `LORRAX_ALLOW_TRS_VELOCITY_PARITY_BREAK` | `0` (off) | Debug override for the QSGW head-velocity TRS parity refusal. The gate is inactive when the 2c DFT reference check finds broken TRS or provides no usable verdict. |
| `LORRAX_ALLOW_X64_OFF` | unset (→ refuse) | The named override for the runtime's x64 refusal (`runtime.enforce_x64`).  With 64-bit values resolved OFF the run refuses at `set_default_env` (an explicit `JAX_ENABLE_X64=0` request) or at step 8a of `initialize_communicator_stack` (the flag read off the live jax — fires whatever turned it off).  `1` continues as an announced UNCERTIFIED run: LORRAX physics is complex128 throughout, so every result is f32/c64, printed on every startup.  Standard `env_bool` grammar. |
| `LORRAX_ALLOW_NONFINITE_RESULT` | unset (→ refuse) | The named forensic escape for `common/sanity.py::refuse_nonfinite`, the REFUSAL on a non-finite object the run is about to ship (`gw/gw_jax.py`'s `kin_ion` / `Σ_total` / `E_qp` seam and `gw/eqp_bgw.py::write_bgw_eqp`'s two columns). `1`/`true`/`yes`/`on` downgrades it to the loud `*** LORRAX SANITY FAILURE` warning so the NaN artifact lands on disk for forensics; anything else refuses. **Deliberately NOT `LORRAX_SANITY`**: that switch buys back the COST of the reductions, and cost is not what is at stake here — on bcc Fe every one of 7176 E_QP entries was NaN and the driver exited **rc=0** in 883 s at the default level (JID 57051742, CLAIMS 204). |

### 3b. Opt-in probes, dumps, test hooks

| var | default | effect |
|---|---|---|
| `LORRAX_DEBUG_GN_ODD_RESIDUE_OFF` | `0` (off) | **DEBUG-ONLY physics A/B switch; never production.** On a measured-broken-TR GN-PPM or MPA fit, `1`/`true`/`yes`/`on` discards the anti-Hermitian frequency-odd residue, setting `D=0` so `_residue_for_space` gives `R+=R-=B`; the even fit, screening, and every other input stay unchanged. GN applies the switch at `fit_gn_ppm_from_wc_pair`; MPA applies it when the stamped ordered-residue fit is consumed. The run record prints `WARNING -- DEBUG` plus a method-specific odd-Sigma line. It REFUSES on a TRS/single-residue fit because there is no TR-odd residue to discard. Canonical `runtime.env_flags.env_bool` grammar. |
| `LORRAX_DEBUG_SHARED_POLE_EVEN_PART` | unset | **DEBUG-ONLY odd-channel diagnostic; never production.** On an ordered (time-reversal-broken) shared-pole store, `all` makes Sigma consume the even part W^even_q = [W_q + W_-q^T]/2: the full-q tile [W_+(q) + W_+(-q)^T]/2 serves both causal branches (`gw.mpa.sigma.shared_pole_even_part_kernel`). This is exactly Sigma of the union store {b(q)/sqrt2 at Omega(q)} and {conj b(-q)/sqrt2 at Omega(-q)}. `exclude_q0` keeps the ordered W at q = 0. Sigma[W] - Sigma[W^even] is the odd part. The run record prints `WARNING -- DEBUG`. It REFUSES unknown values and TRS stores. Unset, `0`, `off`, `false` and `no` preserve the production route exactly. |
| `LORRAX_UNIFORM_RULE_BACKEND` | unset (→ `numpy`) | Execution backend for the minimax reduction inside `build_uniform_rule`: `numpy`, `jax`, or `auto`, case-insensitive after surrounding whitespace is stripped; any other token REFUSES naming the variable. `auto` selects JAX only when an accelerator is present and the start rank is at least 40 with a fit cloud of at least 2000 points, otherwise NumPy. A non-empty explicit `backend=` API argument takes precedence; requesting `jax` still falls back to NumPy if JAX cannot be imported or enumerate devices. This changes reduction performance and floating-point detail, not the acceptance criterion. Backend is not a cache-compatibility dimension, although backend-dependent result detail can change a newly written rule's content digest. |
| `LORRAX_UNIFORM_RULE_TRACE` | unset | Print each MPA box plan's raw real support and final padded denominator box. Diagnostic only: it changes no support, rule, cache key, or executor value. Accuracy, reduction time, and cache location are deck keys. |
| `LORRAX_SIGMA_TAU_TIMING` | `0` | Per-stage blocking timing rows for the staged τ kernel (`gw/ppm_tau_kernel.py`; rows split W/pole synthesis, G build, k convolution, band projection, accumulator, and progress wait) on the routes that still dispatch node by node: the host-driven shared-pole and sector W builders. **Refused on the resident pole route** (GN/HL-PPM, elementwise MPA), which runs each quadrature window as one executable (`DeviceOmegaAccumulator.integrate_window`) and has no per-node host boundary; profile that route with `jax.profiler`. Numerics identical (same primitives, same order, separate XLA modules); walltime NOT comparable to the fused path. Bracketed face carriers refuse because that diagnostic has not been ported. |
| `LORRAX_DEBUG_SIGMA_MAX_TAU_DISPATCHES` | unset | **DEBUG-ONLY bounded MPA Σ performance probe; never production.** A positive integer stops after that many real-shape post-prewarm τ node evaluations (the variable retains its historical spelling), synchronizes the partial device accumulator for an honest wall time, prints seconds/node (and the phase table when `LORRAX_SIGMA_TAU_TIMING=1`), then exits intentionally with rc=0 before any Sigma/QP output can consume the incomplete quadrature. Malformed or nonpositive values refuse. Unset/blank preserves the full executor exactly. |
| `LORRAX_PPM_HERM_DIAG` | `0` | Deck-level ε_H measurement of the PPM amplitude's inherited hermiticity residual — `check_hermitian` over B_q and Ω_q, all q, rtol 1.0 (`gw/ppm_sigma.py:275`).  Diagnostic, not a gate (the channel merge needs no hermiticity). |
| `LORRAX_EXTRA_RANK_PAD` | `""` (→ 0) | **Test-only** extra null directions on the htransform Galerkin rank axis, on top of the mesh-lcm round-up (`bandstructure/htransform.py::resolve_extra_rank_pad`) — the pad-extent-invariance knob for this axis, exactly the role `LORRAX_EXTRA_MU_PAD` plays for μ.  Any result that moves under it at fixed P is a defect.  Negative or malformed REFUSES.  NEVER set in production.  There is deliberately no `LORRAX_EXTRA_BAND_PAD` counterpart — ruling and precondition in [`architecture/decisions.md` 2026-08-06](../architecture/decisions.md).  Exercised by `tests/test_pad_parity_gates.py` — until 2026-08-06 the ONLY test-suite mention was `test_layering.py`'s `_L1_LIBRARY_ENV_READS` registry, whose two consumers are `ast.parse` static analysis and cannot tell a working resolver from a dead one. |
| `LORRAX_EXIT_AFTER_ZETA` | unset | Clean `SystemExit(0)` right after the ζ fit (`gw_init.py`). Combine with `LORRAX_MAX_RCHUNKS` for fast fit-only sweeps; add `LORRAX_DEBUG_PRINT=1` when the sweep needs per-chunk detail. |
| `LORRAX_W_RESIDUAL_CHECK` | `0` | Prints the direct Dyson residual `‖(1−Vχ)W − V‖/‖V‖` on the first few q after a `w_dyson_solver = distributed` W solve (`gw/w_isdf.py`) — the strict numerical contract of the distributed plan (block-cyclic LU is not bit-comparable to the local per-q LU).  Adds one diagnostic jit; leave OFF when taking collective-table probes. |
| `LORRAX_CHECK_REPLICA` | `0` (off) | Re-enables `jax.device_put`'s cross-process equality assertion inside `lxkit.placement.device_put_process_local` (re-exported by `common.collectives`, `distrib_la` and `wfn_loader`) — i.e. deliberately pays the hidden P-linear all-gather (7.8 GB/rank at P=64, scorecard Y.5/AO) the helper exists to avoid, to verify a host table really is replica-identical. Debug only; standard falsy vocabulary (`""`/`0`/`false`/`no`/`off`, case-insensitive, since AT). |
| `LORRAX_SKIP_VQ_GATES` | `0` | Skips the V_Q interpolation self-checks (`bse/vq_interp.py`).  The gates exist because V_Q interpolation errors are silent. |
| `LORRAX_TRS_CHECK` | `1` (on) | Automatic two-component DFT-reference measurement before any TRS-dependent consumer runs. `strict` additionally refuses a broken or inconclusive reference. Historical `0`/`off` values are **retired and refused by name** (`GATE retired_LORRAX_TRS_CHECK_off`) because skipping the measurement defaulted global TR to true and therefore asserted a symmetry rather than disabling a diagnostic. The consumer verdict is only `WfnLoader.trs_holds` → `SymMaps.trs_allowed`; `SymMaps(..., allow_trs=...)` is likewise retired. |
| `LORRAX_TRS_TOL` / `LORRAX_TRS_MAX_K` | `1e-6` / `12` | Occupied-density residual tolerance and maximum independent comparisons (`0` = all). |
| `LORRAX_RHO_SYMMETRISE` | `1` (on) | Projects the valence density onto the subspace invariant under the WFN file's space group before V_H is built (`psp/get_DFT_mtxels.py::symmetrize_valence_density`, applied in `build_hartree_potential` and `compute_valence_density`).  The star average is a projector: it preserves ∫ρ exactly and moves ρ by no more than the asymmetry it removes, so it is on by default — an unsymmetrised ρ is simply wrong on a reduced k-set, and on a full-BZ sum it leaves V_H only as star-invariant as the ψ unfold that built it while every other term of H₀ is exactly star-invariant.  `0`/`off` restores the raw accumulated ρ; A/B and suspect-symmetry-block use only. |
| `LORRAX_FORCE_FULL_BZ` | retired | Removed in the bispinor parent route; production follows the WFN symmetry. No runtime reader remains. |
| `LORRAX_EXTRA_MU_PAD` | `""` (→ 0) | **Test-only** extra μ-pad rows to prove pad-extent invariance (`runtime/padding.py`).  Any result that moves under this at fixed P is a defect.  NEVER set in production. |
| `LORRAX_WRITE_NO_JIT` | unset | Un-jitted Slab-writer path (`_slab_io_ffi.py`); changes execution and is not print verbosity. |
| `LORRAX_FFI_PROFILE` | off | C++: per-call FFI timing. |
| `LORRAX_LU_NO_PIVOT` | off | Experimental cuSOLVERMp math-path override that disables pivoting; changes the solve and is not print verbosity. |
| `LORRAX_LU_DEBUG_DUMP` | off | Benchmark-only array sidecar written by `tests/bench/cusolvermp_solve_lu_test.py`; no production source reads it. |
| `PF_ARTIFACTS_DIR` / `ISDF_JAX_PROFILE_DIR` | `profile` / unset | Trace output dirs (`common/jax_profile.py`, `tests/bench/test_bse.py`). |
| `ISDF_COHSEX_TEST_PLATFORM` | `auto` | Test harness: force `cpu`/`gpu` for the e2e gates (`tests/harness.py`). |

## 4. Build-time only

Read by `config/**/*.sh`, `src/ffi/cpp/common/**/build*.sh` and CMake.
Never by the running Python; setting them in a job script does nothing.

> **This section is 120 of the 234 `LORRAX` env names the tree actually
> reads — an `arch.mk` expressed as environment.** Why that is the shape
> to move away from, and which knobs defer a decision the build already
> made, is [`architecture/ffi_layout.md` §3c](../architecture/ffi_layout.md).

`LORRAX_ROOT`, `LORRAX_SRC`, `LORRAX_VENV`, `LORRAX_SITE`,
`LORRAX_SITE_PACKAGES`, `LORRAX_INSTALL_ROOT`, `LORRAX_DEPS`,
`LORRAX_IMAGE`, `LORRAX_SIF`, `LORRAX_SHIFTER*`, `LORRAX_MODULE*`,
`LORRAX_FFI_STAGE*`, `LORRAX_FFI_BUILD_DIR`, `LORRAX_FFI_SOURCES`,
`LORRAX_FFI_HOST_*`, `LORRAX_FFI_IMAGE`, `LORRAX_FFI_PYTHON`,
`LORRAX_FFI_NO_CUDA`, `LORRAX_FFI_HAVE_{PHDF5,CAL,CUBLASMP,CUFFT}`,
`LORRAX_FFI_PLATFORM`, `LORRAX_HOST_HAVE_{FFTW3,SCALAPACK}`,
`LORRAX_CBLAS_{DIR,INCLUDE_DIR,LIBRARY}`,
`LORRAX_FFTW3_{INCLUDE_DIR,LIBRARY}`,
`LORRAX_MKL_{BLACS,THREAD_LIB,SCALAPACK_LIBRARY}`,
`LORRAX_FFI_ALLOW_DEFAULT_MPI`, `LORRAX_FFI_{PHDF5,SLATE,NVHPC,FFTW}_DIR*`,
`LORRAX_FFTW_STAGE_CLOBBER`, `LORRAX_FFTW3_STAGE`,
`LORRAX_GATE_ONE_FFTW`, `LORRAX_GATE_FFTW_PY`, `CRAY_FFTW_PATH`,
`LORRAX_XLA_FFI_INCLUDE_DIR`, `LORRAX_XLA_FFI_HEADERS_DIR`,
`LORRAX_HDF5_ROOT`, `LORRAX_MPI_INCLUDE_DIR`, `LORRAX_MPICH_LIB_DIR`,
`LORRAX_MPI_LIBRARY`, `LORRAX_MPI_TYPE*`, `LORRAX_MPI_FABRICS`,
`LORRAX_IMPI_ROOT`, `LORRAX_PMI2_LIB`, `LORRAX_ICC_RUNTIME`,
`LORRAX_MKL_ROOT`, `LORRAX_SCALAPACK_LIBRARIES`, `LORRAX_NVHPC_*`,
`LORRAX_CUSOLVERMP_{STAGE,PIN}`, `LORRAX_CUBLASMP_PIN`,
`LORRAX_SLATE_*` (`REPO`, `COMMIT`, `BUILDS_DIR`, `MAKE_J`, `STACK`,
`INSTALL_DIR*`, `HOST_INSTALL_DIR`, `CUDATOOLKIT`),
`LORRAX_HAVE_SLATE`, `LORRAX_HOST_HAVE_SLATE`, `LORRAX_DARSHAN_LIB_DIR`,
`LORRAX_LUSTRE_STRIPE_*`, `LORRAX_NO_PRESTRIPE`, `LORRAX_XLA_CMDBUF`,
`LORRAX_SLURM_{ACCOUNT,QOS,CONSTRAINT}`, `LORRAX_NNODES`,
`LORRAX_NTASKS`, `LORRAX_NGPU`,
`LORRAX_GPUS_PER_NODE`, `LORRAX_SELECT_GPU`, `LORRAX_TIER2_WORKDIR`,
`LORRAX_INPUT`.

### 4a. Perlmutter launch + site config (added 2026-08-06)

Everything above §4 is Frontera-shaped. These were in source and in **no**
section of this page. Audited against `lorrax_P` @ `886139f`; the Frontera
tree at `b61c1df` does not have the first four at all.

| var | default | meaning |
|---|---|---|
| `LORRAX_NVHPC_SUBPATH` | `0.7.2_cuda12.9/math_libs/12.9/lib64` (`run_shifter.sh:171`) | The single source of truth for which cuSOLVERMp stage a run loads. ⚠ **It selects a communication path, not just a version, and every stage exports the same SONAME so a mismatch links cleanly and warns about nothing.** That is the whole of what this registry says about it; the stage/comm-path table, the CMake-default skew and the measured evidence are owned by **`docs/architecture/ffi_layout.md` §4** — read it before touching the CUDA leg. |
| `LORRAX_NVHPC_ROOT` / `LORRAX_NVHPC_MOUNT` | no default — **`build.sh:54-91` REFUSES** | The stage the `.so` is COMPILED against, and the bind-mount point (`/lorrax_nvhpc`). Refuses rather than guessing, because there is no safe default: a guess picks a comm path silently. |
| `LORRAX_PLATFORM` | `gpu` (`run_shifter.sh:55-63`) | `gpu` \| `cpu`/`host`. Decides `MPICH_GPU_SUPPORT_ENABLED` for the Shifter launch — it is per platform, not a constant; on the CPU leg it must be 0 or Cray MPICH aborts in `MPI_Init_thread`. |
| `LORRAX_MPICH_GPU_SUPPORT` | derived from the above (`run_shifter.sh:70`) | Explicit `0\|1` override; anything else refuses. Carried under a `LORRAX_` name because shifter's mpich module **unsets** `MPICH_GPU_SUPPORT_ENABLED` itself, so `in_container.sh` re-derives on the far side of that boundary. |
| `LORRAX_GATE_ONE_MPI` | `on` (`cpp/gate_one_mpi.sh:41`) | The "exactly one MPI implementation in this address space" gate (hazard S3). `=off` disables it and says so loudly on every run. |
| `LORRAX_RUN_DIR` | no default; mandatory in the Frontera template | The run directory holding the input deck, a peer of `LORRAX_ROOT`/`LORRAX_INPUT`. When `ISDF_JAX_CACHE_DIR` is absent it also scopes the persistent JAX cache to this workflow; it never overrides an explicit cache path or opt-out. |
| `LORRAX_PM_{PRGENV,MPICH,CMAKE,FFTW,HDF5,HDF5_DIR,LIBSCI,LIBSCI_DIR,LIBSCI_FLAVOUR,MPICH_DIR}` | per `config/perlmutter/site_config.sh` | The Perlmutter module/prefix family: versioned PrgEnv/cray-mpich/cmake, cray-fftw, cray-hdf5 (+ dir), cray-libsci (+ dir, + flavour) and cray-mpich dir. Consumed by the host builders. |
| `LORRAX_MPIWRAPPER_ROOT_DEFAULT` / `LORRAX_MPIWRAPPER_PREFIX_DEFAULT` / `LORRAX_MPIWRAPPER_COMMIT_DEFAULT` / `LORRAX_MPIWRAPPER_ABI_DEFAULT` | `$HOME/software/lorrax_mpiwrapper_cray` / `…/current` / pinned v2.11.1 commit / `2.10.0` | Immutable, content-addressed Perlmutter releases and atomic active symlink for the unmodified Cray-MPICH MPIwrapper adapter; consumed by the builder and CPU-MPI prelude. |
| `LORRAX_FFI_{NVHPC,PHDF5,SLATE}_HOST` | `config/modulefiles/lorrax/0.1.0.lua:210-212` | Host-side counterparts of the `_DIR*` knobs above; not matched by this page's `_DIR*` globs. |
| `LORRAX_BUILD_JOBS` | `8` (`config/perlmutter/build_ffi_host.sh:246`) | `cmake --build --parallel N`. |
| `LORRAX_MPICH_CONTAINER_DIR` | `/opt/udiImage/modules/mpich` | In-container MPICH module dir, substituted into the modulefile. |
| `LORRAX_SLATE_{SRC,SCALAPACK_API_DIR}`, `LORRAX_SAPI_EXTRA_INCLUDES` | see `cpp/stage/slate_build_scalapack_api.sh:46,57,90` | SLATE ScaLAPACK-API overlay build: source tree (mandatory, `:?`), output prefix, extra `-I` flags when `nvcc` is not in the expected layout. |
| `LORRAX_{PY,VENV_DIR,SRC_DIR}`, `LORRAX_OVERLAY_BUILD_DIR`, `LORRAX_MPIWRAPPER_SO`, `LORRAX_SLATE_HOST_LIB`, `LORRAX_MKL_LIB` | staging defaults; `LORRAX_MPIWRAPPER_SO` defaults to `$LORRAX_MPIWRAPPER_PREFIX/lib64/libmpiwrapper.so` on Perlmutter | Staging outputs and link-line paths. The Perlmutter prelude resolves `LORRAX_MPIWRAPPER_SO`, then verifies the adjacent pinned-source/MPI-ABI/SHA256 manifest before exporting it as `MPITRAMPOLINE_LIB`. |

`LORRAX_CUDA_CHECK` and `LORRAX_LIB_CHECK` are **C macros**, not env
vars — the earlier grep-based counts were misleading.  So are
`LORRAX_CFG_STR` / `LORRAX_CFG_STR2`
(`cpp/common/build_config.cc`, the two-level stringification macro
pair) and the whole `LORRAX_CFG_*` family, which are CMake vars configured
into `lorrax_config.h` rather than anything the environment can set; and so
are `LORRAX_CUSOLVERMP_CHECK` (`cpp/cusolvermp` error-check macro) and
`LORRAX_CUBLASMP_CHECK` (`cpp/cublasmp/batched_gemm_ffi.cc:44`), which
this list used to carry as if they were the STAGE/PIN staging knobs'
siblings (P1 audit, 2026-07-31).

Likewise **not variables**, and excluded on purpose so a future grep-based
diff does not re-add them: `LORRAX_ISSUES` (a substring of the filename
`KNOWN_LORRAX_ISSUES.md`), `LORRAX_SC_` / `LORRAX_JAX_CACHE_` /
`LORRAX_SLURM_` / `LORRAX_PHDF5_` / `LORRAX_FFI_` (prefix fragments from
prose globs and `startswith()` scans, whose real members are all listed
elsewhere on this page), and the `LORRAX_TEST_*` / `LORRAX_FAKE_*` /
`LORRAX_PAIR_*` / `LORRAX_T_*` family, which are string literals inside
`tests/test_env_registry.py` and `tests/test_env_grammar.py`.

Two entries removed from the list above, with reasons (P1 audit):

* **`LORRAX_FRONTERA_ADVICE` is a FILENAME, not an env var** —
  `$WORK/LORRAX_FRONTERA_ADVICE.md`, the out-of-repo machine-advice doc
  that the 2026-07 Frontera CPU handoff pointed at.  Nothing
  reads it from the environment.
* **`LORRAX_PARTITION` has no in-repo reader** — no `config/**` or
  `src/**` file consumes it.  It is an EXTERNAL-OVERLAY variable: the
  /scratch harness generation reads it when composing sbatch headers.
  Recorded here so the name stays reserved; do not add an in-repo reader
  without moving the row to a live section.

### 4b. Launch/staging scripts (read at job-launch time, not by Python)

Read by `config/frontera/stage_runtime.sh` /
`build_cpu_runtime_bundle.sh` / `build_mpiwrapper.sh` /
`build_mpi_overlay.sh` / `ffi_env.sh`.  Same build-time rule: setting
them inside the running Python does nothing.

| var | default | effect |
|---|---|---|
| `LORRAX_BUNDLE` | unset | Path to `lorrax_cpu_bundle.tar` (required for node-local staging; unset/missing → announced fallback to the shared filesystem). |
| `LORRAX_BUNDLE_OUT` | `$SCRATCH/lorrax_bundle` | Where `build_cpu_runtime_bundle.sh` writes the bundle. |
| `LORRAX_BUNDLE_STRIPE` | script default | Lustre striping applied to the bundle output. |
| `LORRAX_STAGE` | `1` | `0` disables node-local staging entirely (announced). |
| `LORRAX_STAGE_ROOT` | `/tmp/lorrax_stage.$UID` | Node-local extraction dir. |
| `LORRAX_STAGED` / `LORRAX_STAGE_S` | outputs | Set BY `stage_runtime.sh` (staged? / seconds spent), not user knobs. |
| `LORRAX_VENV_FALLBACK` / `LORRAX_OVERLAY_FALLBACK` / `LORRAX_SRC_FALLBACK` | shared-FS paths | Where to run from when staging is off or fails. |
| `LORRAX_OVERLAY` | `$WORK/lorrax_env_mpi_overlay/site` | MPI overlay site the bundle build packs. |
| `LORRAX_OVERLAY_PREFIX` | `$WORK/lorrax_env_mpi_overlay` | `build_mpi_overlay.sh` install prefix (`…/site` is the product). |
| `LORRAX_OVERLAY_DIR` | output | Set BY `stage_runtime.sh` to the resolved (staged or fallback) overlay dir. |
| `LORRAX_MPIWRAPPER_REPO` / `LORRAX_MPIWRAPPER_COMMIT` | upstream MPIwrapper, pinned v2.11.1 commit | What `build_mpiwrapper.sh` fetches. |
| `LORRAX_MPIWRAPPER_STAGE` / `LORRAX_MPIWRAPPER_SRC` / `LORRAX_MPIWRAPPER_BUILD` / `LORRAX_MPIWRAPPER_PREFIX` | Frontera: `$WORK/lorrax_mpiwrapper/…`; Perlmutter exposes only the atomic active prefix | Frontera stage/source/build/install paths. Perlmutter deliberately does not accept independently redirected source/build/install paths: its builder owns `$LORRAX_MPIWRAPPER_ROOT/{stage,releases,current}` so `--fresh` cannot delete an unrelated or active tree. |
| `LORRAX_MPIWRAPPER_ROOT` | `$LORRAX_MPIWRAPPER_ROOT_DEFAULT` (Perlmutter only) | Root containing the builder sentinel, fresh candidates, immutable content-addressed releases, and atomic `current` symlink. |
| `LORRAX_MPIWRAPPER_REFERENCE_SO` | unset | Optional reference `.so` for the build-note section comparison; skipped when absent. |
| `LORRAX_CMAKE` | `command -v cmake` | Which cmake `build_mpiwrapper.sh` uses. |
| `LORRAX_FFI_SO_PHDF5` | `$LORRAX_FFI_STAGE/build_phdf5/liblorrax_ffi.so` | `ffi_env.sh:41`: the phdf5-enabled CUDA FFI `.so` that `LORRAX_FFI_SO` is exported from. |

## 5. External variables LORRAX sets, reads, or that its harnesses dial

| var | LORRAX's handling |
|---|---|
| `JAX_ENABLE_X64` | Owned by the runtime since 2026-08-27. `runtime.set_default_env` does the canonical `setdefault "1"` (library backstops remain for bare imports), and `runtime.set_x64_on_imported_jax` pushes the resolved value onto a jax that was imported before the runtime ran — the order the per-driver `config.update` lines never covered. A resolved `False` REFUSES, at `set_default_env` (the request) and again at step 8a of `initialize_communicator_stack` (the flag read off the live jax); the named opt-out is `LORRAX_ALLOW_X64_OFF` (§3b). |
| `JAX_PLATFORMS` | `setdefault "cuda,cpu"`, or hard-set to `"cpu"` by `set_default_env(platform="cpu")`.  `ffi_loader.platform_from_env` READS it (default `""`) to pick the FFI library **without** initializing the JAX backend. |
| `JAX_PLATFORM_NAME` | jax's DEPRECATED spelling of `JAX_PLATFORMS`.  LORRAX never sets it for a run: `runtime` **pops** it at both CPU-downgrade sites (`runtime/__init__.py:565,967`) so a stale value cannot fight the downgrade; `psp/get_DFT_mtxels.py:24` presence-tests it (either spelling counts as "the caller chose a platform"); the one `setdefault "cpu"` writer is the exempt bench driver `tests/bench/benchmark_synthetic.py:22`. |
| `JAX_PROCESS_COUNT` → `JAX_NUM_PROCESSES` → `SLURM_NTASKS` → `1` | process-count resolution chain, `runtime/__init__.py`. |
| `JAX_PROCESS_INDEX` → `SLURM_PROCID` → `0` | process-index chain. |
| `JAX_COORDINATOR_ADDRESS` | overrides the `SLURM_NODELIST`-derived coordinator. |
| `FI_PROVIDER` | not read by LORRAX.  On Frontera CLX **leave it UNSET** — Intel MPI then auto-selects `mlx`: 1.07 µs / 11.4 GB/s vs the old `FI_PROVIDER=tcp` pin's 10.9 µs / 2.15 GB/s, and pzheevd n=2448 at P=144 goes 12 s/q → 0.5–0.9 s/q (AP.3/AP.4; reproduced in-container by AS.2).  Keep `I_MPI_DEBUG≥4` so rank 0 announces `libfabric provider:`; do NOT trust `fi_info`, which reports −61 for `mlx` even where it works.  In-container: apptainer's default mount already exposes the host `/dev` (uverbs included) — **never `--bind /dev[/...]`** (a nosuid,nodev shadow copy breaks every device open, AS.1); stage the RDMA userspace via the `/hostlibs` symlink pattern (AS.1 / wk_AS `as_inner.sh`), or the provider falls back to tcp, announced. |
| `CUDA_VISIBLE_DEVICES` | read to derive `local_device_ids`; `tests/conftest.py` rewrites it per xdist worker. |
| `_LORRAX_JAX_DISTRIBUTED_DONE` | LORRAX's own idempotency sentinel for `jax.distributed.initialize` — env-scoped on purpose, so it survives module re-imports.  Self-set, not a user knob.  (`_LORRAX_GLOO_PIN_DONE` went with the Gloo interface pin.) |
| `XLA_PYTHON_CLIENT_ALLOCATOR`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_CLIENT_MEM_FRACTION` (current spelling) / `XLA_PYTHON_CLIENT_MEM_FRACTION` (deprecated) | Registry facts only; **what the allocators do, and why LORRAX reserves the async pool, is owned by `docs/environment/overview.md` §2.1.** All three are set together, all or nothing, in ONE function, `runtime.set_default_gpu_pool()` (called by `set_default_env()`, so every driver inherits it; `psp/get_DFT_mtxels.py` calls it directly only because it is a standalone CLI that never calls `bootstrap()`): on CUDA `cuda_async`, `PREALLOCATE=true`, and `XLA_CLIENT_MEM_FRACTION=runtime.GPU_POOL_FRACTION` (0.89 on every CUDA node, 40 and 80 GB alike — owner ruling 2026-09-24) unless either fraction spelling is already set; untouched on a CPU run, on a node without an NVIDIA device, and on ROCm (jaxlib defaults; no ROCm deployment). No module file or launcher sets any of them (`tests/test_perlmutter_environment_descriptor.py`). `runtime._check_allocator_env()` *validates* a caller-supplied allocator and *removes* a blank one, which is a WRITE, not a read. Both fraction spellings at once REFUSE in `set_default_gpu_pool()` (jaxlib refuses the pair too, but inside plugin discovery, where it reads as a missing backend); either alone is read by `runtime/xla_memory.py`, the deprecated one flagged in the startup report. The policy is ALL OR NOTHING: it is applied whole only when neither `ALLOCATOR` nor `PREALLOCATE` is exported, and `cuda_async` with `PREALLOCATE` off refuses. `tests/conftest.py` pins the one stated test-only exception (`bfc`, `PREALLOCATE=false`: pytest workers share GPUs with the mesh-cell child). |
| `TF_GPU_ALLOCATOR` | **Not a LORRAX variable and inert for JAX.** Listed only so a future grep-based diff does not re-add it: it is a TensorFlow knob, measured byte-identical with and without (job 7882442), and has no writer anywhere in `src/`. |
| `SLATE_SCALAPACK_TARGET` | SLATE's OWN dial, read by its ScaLAPACK-compat shim (`scalapack_slate.hh:170-188`), surfaced here because `blacs_grid.h:305` reads it to ANNOUNCE the demotion it controls: unset defaults to `HostTask`, so a SLATE built `gpu_backend=cuda` still runs on the **CPU** unless it is set to `devices`.  Only meaningful with `LORRAX_SCALAPACK_ALLOW_SLATE_API` (§2b). |
| `OPENBLAS_THREAD_TIMEOUT` | `setdefault "1"` by `runtime.tune_blas_threading` at import of `runtime` (§2a `LORRAX_BLAS_TUNE`); an exported value wins. log2 of OpenBLAS's idle spin count. It is read ONCE, in the library constructor that runs at the first `import numpy` — which is why this one setting cannot wait for the startup call, and why the entry-point import order is a tested invariant rather than a convention. |
| `OMP_WAIT_POLICY`, `KMP_BLOCKTIME`, `GOMP_SPINCOUNT` | **Not set, and not because they were forgotten.** The numpy/scipy BLAS here is scipy-openblas built on pthreads, so `OMP_WAIT_POLICY=PASSIVE` provably does nothing to it (measured: 17.44 s vs 15.02 s unset, noise). They WOULD reach the FFI's own OpenMP teams (`ffi/cpp/mklfft/fft_flat_k_ffi.cc`, `mklblas/gemm_batch_ffi.cc`, `#pragma omp parallel num_threads(nthr)`), a genuinely separate threading layer where Intel OpenMP's `KMP_BLOCKTIME` default is 200 ms of spin. Untested there, and the sign is not obvious: the FFT handler is a `#pragma omp for` chunk loop, the one shape where spinning HELPS. Measure before setting. |
| `XDG_CACHE_HOME` | last-resort legacy base for the JAX compile cache (`ISDF_JAX_CACHE_DIR` / `LORRAX_RUN_DIR` come first — §2b). |
| `HDF5_USE_FILE_LOCKING` | `setdefault "FALSE"` in one psp test; exported `FALSE` by every production harness. AUDITED (AW, 2026-07-27): **not load-bearing on Frontera `/scratch2`** — the mount has real `flock`, and a full 785c/P=16 e2e with the variable UNSET (HDF5 default locking) ran rc=0 with all four eqp/sigma files bit-identical (`run_800c_awlock`). The MPI-IO VFD takes no POSIX locks at all, so the variable only ever governs the serial-h5py side paths (eager reads, deferred attrs, `_introspect_dataset`). KEEP the harness export anyway: `/work2` mounts `localflock` (locks are node-LOCAL — cross-node "locking" there is silently incoherent, so honest intent is to disable), and h5py wheel HDF5s differ in lock default. Machine fact, one export per harness, never per-tool. |
| `OMP_NUM_THREADS` | Never set by LORRAX (the `psp/orbital_magnetization.py` `--cpu` setdefault was deleted 2026-09-24); an exported value wins over the runtime's BLAS thread tuning. NOTE: on Frontera the XLA:CPU threadpool does **not** obey OMP — `taskset` pinning is the real mechanism (FRONTERA_ADVICE §10). |
| `OPENBLAS_NUM_THREADS` | Read-only, for the startup report's thread-count/oversubscription table (`runtime/__init__.py:1287`); LORRAX never sets it. |
| `SLURMD_NODENAME` / `HOSTNAME` | Coordinator-address fallback chain after `SLURM_NODELIST` (`runtime/__init__.py:768`) — machine facts, read only. |
| `MPLBACKEND` | `setdefault "Agg"` for headless plotting. |
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION` | JAX's own config (`gloo` default \| `mpi` \| `megascale`); LORRAX production multi-process CPU runs require `mpi`. `runtime.announce_cpu_collectives()` refuses any other backend because the measured gloo failure is silent corruption. The evidence and site transports are owned by `docs/environment/transports.md`. |
| `MPITRAMPOLINE_LIB` | MPItrampoline's adapter path. Frontera points it at the patched Intel-MPI build; Perlmutter points it at the exact unmodified upstream build from `config/perlmutter/build_mpiwrapper.sh`. It must be absolute and set before JAX import; vendor `libmpi.so` is not an adapter. Full contract: `docs/dev/mpi_collectives.md`. |
| `LD_PRELOAD` | On Perlmutter CPU-MPI, `config/perlmutter/cpu_mpi_env.sh` prepends `/opt/cray/pe/lib64/libpmi.so.0` before Python and verifies it resolves under the Cray tree. Non-MPI entries are preserved; foreign MPI, MPItrampoline/MPIwrapper and `libpmi2` entries are refused. This is the Cray PMI initialization-order workaround; `libpmi2.so.0` is a measured negative. |
| `MPICH_ASYNC_PROGRESS` | Perlmutter CPU-MPI prelude hard-sets HPE's supported public control to `1`; unset elsewhere. Cray MPICH then grants `MPI_THREAD_MULTIPLE` to XLA's explicit FUNNELED request and creates one progress thread per rank. |
| `MPIR_CVAR_ASYNC_PROGRESS` | Internal MPICH CVAR used during diagnosis; the Perlmutter prelude now unsets it in favor of public `MPICH_ASYNC_PROGRESS`. |
| `LORRAX_MPI_FORCE_THREAD_MAIN` | Historical Frontera MPIwrapper gate; **SUPERSEDED and unset in production.** `common.collectives.warm_mesh_cliques()` creates every mesh-axis/world communicator from the Python main thread. Perlmutter's unmodified wrapper does not implement this override. |
| `LORRAX_MPI_FINALIZE_FIX` | Frontera overlay control (`skip_atexit` / `hard_exit`), not read by `src/`. Drivers using `runtime.finalize_process()` do not need the overlay; bare interpreter teardown can still exit nonzero after otherwise successful MPI work. Lifecycle scope: `docs/dev/mpi_collectives.md`. |
| `LORRAX_MPI_PROVIDER` | The harness/`ffi_env.sh` dial over `FI_PROVIDER` (scorecard AP.7/AS.5). `auto` (default) **unsets** `FI_PROVIDER`+`FI_TCP_IFACE` so Intel MPI picks the native provider (`mlx` on CLX); `tcp` restores the IPoIB pin with `FI_TCP_IFACE=ib0` (the rtx/mlx4 escape hatch); any other value force-requests that provider (never `verbs` at P≥144 one-block — 68 s/q pathology, AP.4). Read by the sbatch env blocks and by `config/frontera/ffi_env.sh`; not read by Python. The `auto` unset is load-bearing, not hygiene: TACC's default impi module exports `FI_PROVIDER=mlx` into every login/compute shell and sbatch/ssh inherit it, so "leave it unset" requires actively unsetting. |
| `I_MPI_FABRICS` | Harnesses export `shm:ofi` — which is Intel MPI 2019+'s own default, so this is *documentation of intent* plus a guard against a stray inherited value, not a behavior change (AU: pingpong identical with it unset). `ffi_env.sh` defaults it to `shm` (single-node rtx bring-up: skips OFI init entirely); override with `LORRAX_MPI_FABRICS=shm:ofi` for multi-node. |
| `I_MPI_PMI_LIBRARY` | Needed ONLY under `srun --mpi=pmi2` (the Intel-MPI-under-slurm bootstrap; harness cells that run no MPI code don't need it, and nothing loads it unless MPI inits). MUST be set unconditionally where used: TACC's login env exports `/usr/lib64/libpmi.so` — a PMI-**1** library, wrong protocol for `--mpi=pmi2` AND absent inside the container. The staged PMI2 lib is `$WORK/host_pmi/libpmi2.so.0` (`LORRAX_PMI2_LIB` overrides in `ffi_env.sh`). |
| `I_MPI_DEBUG` | Default 4 in every harness + `ffi_env.sh`. Init-time-only output (provider banner + pinning table); AU measured steady-state pingpong identical at `I_MPI_DEBUG=0`, so the banner is free. It is MANDATORY telemetry — the `libfabric provider:` line is the only trustworthy provider observable (`fi_info` false-negatives on mlx), and a silent transport is how the em1/tcp era happened. Keep ≥4. |
| `FI_PROVIDER_PATH` | Harnesses + `ffi_env.sh` pin it to `$IMPI/libfabric/lib/prov` (the bundled provider .so dir). **REQUIRED in-container, not belt-and-braces**: `mpivars.sh` (which normally sets it) is not sourced there, and AU measured that with it unset PMPI_Init aborts outright — `MPIDI_OFI_mpi_init_hook ... addrinfo() failed ... No data available`, i.e. libfabric finds NO providers at all. (Note this is the exact error string of the rtx-era "tcp/mlx4 fails in-container" archaeology — some of that history may have been a missing FI_PROVIDER_PATH, not the fabric.) |
| `UCX_*` (`UCX_TLS`, retry/timeout tunings) | TACC's default impi module exports `UCX_TLS=knem,dc_x,rc` + `UCX_{RC,DC,UD}_MLX5_{TIMEOUT,RETRY_COUNT}` bumps into every shell, and sbatch/ssh launches inherit them — so every AP/AS `mlx` number (1.07 µs / 11.4 GB/s, pzheevd 0.52 s/q) was measured UNDER those tunings, not under bare UCX defaults. AU A/B (in-container, provider auto): stripping every `UCX_*` leaves 8 B pingpong/allreduce unchanged (1.07 µs / 3.38 µs) but **doubles the 1 MiB 32-rank Allreduce (419 → 799 µs)** — the tunings are load-bearing for large-message collectives. The harness blocks therefore SETDEFAULT the six module values (`${UCX_TLS:-knem,dc_x,rc}` etc.): inherited values always win, but a stripped launch environment (cron, clean ssh) no longer silently loses 2×.  Since the fix/zq audit the TRACKED `config/frontera/ffi_env.sh` carries the same six setdefaults (guarded to the non-`tcp` provider cases), so the mitigation is auditable in-repo rather than only in the /scratch harnesses. Do not hard-pin them (rtx/mlx4 has no `dc_x`), and do not add other UCX knobs — the mlx TPN=4 × n=5024 anomaly (AP.9.4) is the only open UCX question and is not production-relevant (2×28 layout is clean). |
| `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` | `GLOO_SOCKET_IFNAME` is **INERT with jax** — the string appears nowhere in the shipped jax/jaxlib (scorecard AF.5/AK.4); every job script that exports it is exporting a no-op. There is no LORRAX Gloo dial any more either — LORRAX's CPU collectives run on MPI, whose transport is selected by `LORRAX_MPI_PROVIDER` / `FI_PROVIDER`.  (`LORRAX_GLOO_IFNAME`, the AL-era interface pin for `runtime::pin_gloo_interface`, is **HISTORICAL**: the pin was removed with the mpi migration and the name survives only in a comment at `runtime/__init__.py:56`.  Exporting it does nothing.) `NCCL_SOCKET_IFNAME` *is* read by NCCL and still matters on GPU runs. |

---

## Consistency audit

Every LORRAX-owned variable read at more than one site was checked for
default drift (re-checked 2026-07-27; bounded Python roots re-checked
2026-09-02).

> **WHAT THIS AUDIT DOES NOT CHECK, and what that cost (2026-07-30).**  It
> compares the **default** each site falls back to.  It does not look at the
> PARSE.  So a knob read the same wrong way at every site scores a ✓: the
> a retired memory-debug row used to read `4 | all presence-test ✓`,
> where "presence-test" means `if os.environ.get(...)` — under which
> `=0` turned the probes **on**.  Four sites agreeing on a
> broken parse is not consistency; it is one defect copied four times, and
> the ✓ is what stopped anyone looking.  A green "consistency" column
> means only *"these sites agree"*, never *"these sites are right"*.
> `tests/test_env_grammar.py` is the check that speaks to correctness.

| var | sites | defaults | parse |
|---|---|---|---|
| **every boolean knob** | — | — | **ONE PARSER since 2026-08-22: `runtime/env_flags.py::env_bool`.** `gw.gw_config.env_bool`, `file_io._slab_io_ffi._env_flag` and `runtime._env_falsy` are re-exports of it — checked by IDENTITY, not equality, in `tests/test_env_grammar.py::test_defect3_vocabulary_has_not_drifted`, and `test_the_substrate_parsers_import_the_grammar_rather_than_copying_it` refuses a re-grown literal. The two that were converted both SWALLOWED an unrecognised token in silence, **in opposite directions** — `_env_flag` resolved it off, `_env_falsy` left the knob on — so a typo in a knob's VALUE moved a default in whichever direction the reader happened to use. Why the grammar is at L3 and not in `gw_config`: those two parsers are L3 and may not import an L1 module, so a grammar owned at L1 is one the substrate re-invents. `ffi/gate.py::MODE_SPELLINGS` keeps its own resolver (its `auto` is load-bearing) and only its on/off token sets are checked set-equal. |
| `LORRAX_FORCE_FULL_BZ` | retired | Removed in the bispinor parent route; production follows the WFN symmetry. No runtime reader remains. |
| `LORRAX_PHDF5_STRIPE_COUNT` | **2** (was miscounted as 3 here and 4 in §2b; re-counted 2026-08-06) | **✗ SPLIT since `e5c9618`** — Python `clamp(nranks, 4, 128)`, C++ literal `16`.  This row read "both `16` ✓" until 2026-08-06 | **✓ agreed (2026-08-06)** — both refuse a non-integer and both refuse a negative count.  Was SPLIT: Python refused loudly, C++ forwarded the raw string to `MPI_Info_set` with no validation at all. |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | 2 | **✗ SPLIT since `e5c9618`** — Python ramps 1 → 4 MiB with the rank count, C++ is flat `1M`.  This row read "both `1M` ✓" until 2026-08-06 | **✓ agreed (2026-08-06)** — both refuse an unknown suffix, and both refuse `4MiB`.  Was SPLIT: C++ computed `mult=0` for an unknown suffix and **silently kept 1 MiB**, which is what the warning box above says the audit cannot catch — identical defaults, different parses. |
| `LORRAX_ZETA_RCOND` | 2 factor sites + 1 provenance echo | ONE shared non-empty-env-wins rule (`isdf/core._env_override_raw`) ✓ | that one rule serves both the factor sites (`_deprecated_env_float`) and the provenance record (`deprecated_env_record`); the inline mirror in `gw_init` was deleted by the fix/zq audit ✓ |

The scanner also flags `JAX_PLATFORMS`, `CUDA_VISIBLE_DEVICES`,
`XLA_PYTHON_CLIENT_*` and `TF_GPU_ALLOCATOR` as having "multiple
defaults".  Each is a `setdefault` (writer) paired with a plain `get`
(reader) — the intended pattern, not drift.  (`TF_GPU_ALLOCATOR` no longer
has a writer at all: it is inert for JAX and was deleted, not corrected.
See the `XLA_PYTHON_CLIENT_ALLOCATOR` row above.)

Re-run the audit with:

```bash
python3 tools/env_audit.py src        # AST walk; flags "MULTIPLE DEFAULTS".
                                      # Sees helper-mediated reads (env_bool /
                                      # env_float / _env_falsy / Gate(env=...))
                                      # and carries a --selftest that fails
                                      # loudly on interpreters whose AST it
                                      # cannot read (py3.7 ast.Str), instead of
                                      # printing a FALSE-CLEAN empty report.
grep -rn 'getenv(' src/ffi            # the C++ side, which the AST walk can't see
python3 tests/test_env_registry.py    # ENFORCEMENT: every LORRAX read site
                                      # under src/ (py AND C++) must have a row
                                      # on this page, or the gate fails.
```

The registry gate is what stops this page decaying again (it had gone
stale twice before 2026-07-31, both times because the audit tool was a
silent no-op on the login python). C++ tuning spellings registered here are
the current `LORRAX_FFT_FFI_{THREADS,CHUNK}` forms; all native print detail
uses `LORRAX_DEBUG_PRINT`. Known gate gap: the C++ scan
matches `getenv`/`log_here`/`env_flag` literals only, so reads funneled
through `mklpin::knob_value(...)` are invisible to it — those rows are
maintained by hand.

## Why CPU collectives run on `impl=mpi`

Not on this page. The mechanism, the gloo silent-corruption evidence, the
`MPI_Is_thread_main` gate and the launch recipe are owned by
**`docs/dev/mpi_collectives.md`**, with the measured transport verdicts in
**`docs/environment/transports.md`**.

A ~35-line retelling stood here until 2026-08-06 and is deleted rather than
trimmed. It had already gone wrong twice in ways the owner page had not — it
carried a superseded remedy (`LORRAX_MPI_FORCE_THREAD_MAIN=1`) as the current
one, and before that a mechanism ("collectives inside a `lax.scan` inside a
`shard_map`") that a clean-room probe refuted. Both errors are exactly what a
second copy is for. The registry rows for
`JAX_CPU_COLLECTIVES_IMPLEMENTATION`, `MPITRAMPOLINE_LIB`,
`LORRAX_MPI_FORCE_THREAD_MAIN` and `LORRAX_MPI_FINALIZE_FIX` stay in §5,
where they belong; their *explanations* now live in one place.
