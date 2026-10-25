# LORRAX environment variables — the registry

This page owns four facts about every environment variable LORRAX reads: its
**spelling**, its **default**, its **class**, and its **parse grammar**. Each
row adds one sentence on what the variable controls and what it refuses; the
explanation belongs to the page the row links to. `tools/lint/test_env_registry.py`
fails when a variable read under `src/` or `services/*/src` (Python or C++)
has no row here.

**Policy.** The environment grants machine capability (library paths,
transport, thread counts, resource caps) and debug switches. It never selects
physics or routing: those are input-file keys, which are validated, echoed and
recorded in provenance. An env twin of an input key is
removed, so the input key is the only route; the routing-affecting variables
that remain are §1b.

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

### 1b. Routing-affecting variables

Each of these chooses a route or a tolerance outside the input file, so it
escapes input validation and provenance. Set one deliberately and record it
with the run.

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_WFN_BACKEND` | unset (auto) | routing-affecting | Stripped, lower-cased: `eager` or `phdf5` forces the WFN read backend (`services/wfn_loader/src/wfn_loader/loader.py::_auto_pick_backend`); `phdf5_host` refuses; any other value takes the auto pick without comment. A mesh-less loader always reads eager. At P>1 with no loadable FFI the auto pick refuses, and `eager` is the way through. |

---

## 2. Machine capability and resource caps

None of these may change physics. Those that change wall time carry the one
number that sets the cost of choosing a value.

### 2a. Startup: network transport and GPU pool

`runtime/network_env.py` applies the network decision once, before backend
creation, and only on a CUDA run; a CPU or ROCm run is untouched. The site's
values come from its profile, `config/<site>/site.toml`. What the decision is
and why: [Perlmutter: network transport at startup](../environment/machines/perlmutter.md#network-transport-at-startup).

| var | default | class | grammar and effect |
|---|---|---|---|
| `SLURM_STEP_NUM_NODES` | unset | launch | Positive integer node count of the step, the first topology evidence; an invalid value falls through to the host list. |
| `SLURM_STEP_NODELIST` | unset | launch | Slurm host-list expression (`nid[001000-001003,001010]`), expanded in Python (`runtime.network_env.expand_nodelist`); its first host is the coordinator's. |
| `OMPI_COMM_WORLD_SIZE`, `OMPI_COMM_WORLD_LOCAL_SIZE` | unset | launch | Positive integers; equal means one node, local below world means several (no uniform placement assumed). |
| `LORRAX_SITE_PROFILE` | unset | machine | Names the site profile, `config/<LORRAX_SITE_PROFILE>/site.toml`, ahead of `NERSC_HOST`; a name with no such file refuses at startup. |
| `NERSC_HOST` | unset | machine | Selects `config/<NERSC_HOST>/site.toml` when that file exists (`perlmutter` does); any other value selects no profile. |
| `SLURM_NETWORK` | Perlmutter's GPU setup/module defaults to `no_vni` before `srun`; `lx run` also exports it | launch | Comma-separated Slurm network options. Setup preserves every explicit value, including empty, and does not fill it inside a numeric `SLURM_STEP_ID`/`SLURM_STEPID`. The profile's `step_network` must be present when `srun` creates the step; setting it inside Python is too late. A multi-node profiled CUDA step without it refuses at startup. |
| `NCCL_NET`, `NCCL_NET_PLUGIN` | Perlmutter profile: `AWS Libfabric` and the site OFI plugin by absolute path | transport | Presence of either, even empty, bypasses site selection and keeps the caller's configuration; `NCCL_NET=Socket` (any case) refuses on a multi-node profiled CUDA step. The profile refuses when its plugin file is missing. A multi-node CUDA step with neither, and no `libnccl-net.so` on `LD_LIBRARY_PATH`, warns. |
| `NCCL_NET_GDR_LEVEL`, `FI_CXI_DISABLE_HOST_REGISTER`, `FI_CXI_RDZV_THRESHOLD`, `NCCL_SOCKET_IFNAME` | Perlmutter profile: `PHB`, `1`, `0`, `hsn` | transport | Filled in from the profile's `[nccl.env]` only where unset; exported values are kept verbatim. A multi-node CUDA step on a Slingshot node (`/sys/class/cxi`) with no profile gets `FI_CXI_DISABLE_HOST_REGISTER=1` and `FI_CXI_RDZV_THRESHOLD=0` and a warning naming the missing profile. |
| `XLA_PYTHON_CLIENT_ALLOCATOR`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_CLIENT_MEM_FRACTION` (deprecated spelling `XLA_PYTHON_CLIENT_MEM_FRACTION`) | set by `runtime.set_default_gpu_pool` | GPU pool | On a GPU run whose leg declares an allocator (`PlatformLeg.xla_allocator`; CUDA: `cuda_async`), when neither `ALLOCATOR` nor `PREALLOCATE` is exported, the runtime sets all three: that allocator, `true`, and `XLA_CLIENT_MEM_FRACTION` = `runtime.pool_fraction` of the card (the memory rule, [memory model](../architecture/memory-model.md#budget)) unless a fraction is already exported; a CPU run, a GPU-less node and ROCm (no allocator declared; jaxlib's BFC at 0.75) get nothing, and an exported `ALLOCATOR` or `PREALLOCATE` makes the caller own the pool. Refused: `cuda_async` with preallocation off, preallocation on with no allocator, both fraction spellings at once, an allocator spelling jaxlib rejects (a blank allocator is removed), and an allocator the leg's plugin cannot build (`PlatformLeg.xla_allocators`; `cuda_async` on ROCm: GATE xla_allocator_leg). What the pool does: [environment/overview.md#gpu-pool](../environment/overview.md#gpu-pool). |
| `XLA_FLAGS` (`--xla_disable_hlo_passes`) | `rematerialization` added to the pass list | machine | `runtime.disable_xla_rematerialization` merges `rematerialization` into a caller's `--xla_disable_hlo_passes` list (or appends the flag), on GPU and CPU; it is never removed. A compiled module larger than the device then refuses (`GATE xla_rematerialization`, [memory model](../architecture/memory-model.md#module-does-not-fit)). |
| `XLA_FLAGS` (`--xla_gpu_autotune_level`) | GPU: the leg's `PlatformLeg.xla_flag_defaults` appended (CUDA: `--xla_gpu_autotune_level=0` and `--xla_gpu_enable_triton_gemm=false`; ROCm: `--xla_gpu_autotune_level=0` and `--xla_gpu_disable_async_collectives=ALLCOLLECTIVES`) | machine | `runtime.set_default_xla_gpu_autotune` appends each flag unless it is already present as `--name=V` or `--name V`; other flags are untouched and a caller value wins, flag by flag. Neither is added on a CPU run. What each flag does and what it measured: [compilation §3](../architecture/compilation.md#3-xla-programs). |

### 2b. Memory caps, collective chunking, schedules

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_COLLECTIVE_CHUNK_MB` | `128` | resource cap | Float MiB; a malformed value uses the default; `≤ 0` is unbounded (reproduction only). Upper bound on one emitted collective's payload, enforced as a host-level loop XLA cannot fuse back: ζ-fit collectives (`isdf/core.py`), the distributed W Dyson A-build (`gw/w_isdf.py`), the rank-0 owner gather of k-partitioned sweeps (`common/collectives.py::gather_indexed_blocks_to_owner`), and the V_q G-panel width (`gw/v_q_g_flat.py`). A single 1.15 GB all-gather was fatal on Gloo at P=144; on MPI at P=16 the cap costs nothing measurable. Orthogonal to the live-bytes cap below. |
| `LORRAX_PPM_FIT_ARENA_GIB` | unset | resource cap | Float GiB. Caps the bytes the GN-PPM fit's q-block sizer (`gw/minimax_screening.py::_gn_ppm_fit_q_block`) prices against (min over processes); below one q it warns and runs one q; unset, the q block is the fixed tile (`runtime.tiles`), at least one q, with one q's footprint from the kernel compiled at q = 1. Bit-exact at any q block. |

### 2c. Numerical guards

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_RANK_POLICY` | `refuse` | guard | `refuse`, `warn` or `off`; anything else refuses naming the variable. Whether a rank truncation that bound and left the certified regime stops the run (`common/rank_criterion.py`), at the ζ charge and transverse fits and the indefinite transverse ridge path. Never changes a number. Site register: [`rank_truncation_policy.md`](../architecture/rank_truncation_policy.md). |

### 2d. FFI libraries and native threads

The C++ side of the registry test sees only `getenv`, `log_here` and
`env_flag` literals; the alias rows read through `mklpin::knob_value` are
kept by hand.

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_FFI_SO` | in-tree `src/ffi/cpp/build/liblorrax_ffi.so` | machine | Path of the CUDA FFI library. A sealed deployment needs no pin: the loader finds `lorrax_ffi_bundle.json` beside `lib/`. Pinning either sealed leg requires this and `LORRAX_FFI_HOST_SO` to name the exact pair in that manifest; a missing, partial, mixed, wrong-hash, wrong-origin or wrong-ABI selection refuses. An ordinary run sets no `LD_LIBRARY_PATH`. Build legs: [`architecture/ffi_layout.md`](../architecture/ffi_layout.md). |
| `LORRAX_FFI_HOST_SO` | in-tree `src/ffi/cpp/build_host/liblorrax_ffi_host.so` | machine | Path of the host FFI library, under the same sealed-pair rule. `config/frontera/build_ffi_host.sh` writes its site build elsewhere, so an unsealed developer run may point here. |
| `LORRAX_FFI_ROCM_SO` | in-tree `src/ffi/cpp/build_rocm/liblorrax_ffi_rocm.so` | machine | Path of the ROCm FFI library, under the same sealed-pair rule (a bundle binds the host leg and one device leg). The ROCM leg is declared ([platform legs](../architecture/ffi_layout.md#platform-legs)) but no library is built yet, so a ROCm mesh asking for a native target refuses by naming this file. |
| `LORRAX_BANDS_GEMM_FFI` | unset (`on`, required) | machine | `ffi/gate.py` grammar: unset or `1` is the vendor-BLAS host handler for `contract_bands_block_reshard` on a CPU mesh; `0` is an announced, uncertified opt-out onto the XLA einsum; a stale `auto` resolves to the default with a note. A missing or unloadable handler refuses at startup and at the kernel factory; kernel caches key on the value. Serves f64/f32/c128/c64 and refuses other dtypes. Not read on CUDA, where XLA's dot lowering already calls cuBLAS. XLA:CPU's Eigen dots run 1.6–1.9× below vendor BLAS at full threads. |
| `LORRAX_FFTW3_SO` | unset (the candidate ladder) | machine | Path of the file the host FFT engine is `dlopen`ed from (`ffi/cpp/fftw/fft_flat_k_ffi.cc::fftw3_candidates`, at the first FFT), tried ahead of the build's compile-time hint and of `libfftw3.so.3`, `libfftw3.so.mpi31.3`, `libmkl_rt.so`, `libfftw3.so`. A bad path is skipped silently; no engine at all refuses at the first host FFT, naming every candidate (the startup gate checks only the handler symbol). Nothing at run time checks that the named file is a CPU FFTW3 (a GPU FFTW-API shim exports the same three symbols); the stage-time check is `ffi/cpp/gate_one_fftw.sh`. |
| `LORRAX_MKLBLAS_THREADS` | unset (`auto`) | machine | `auto` = ambient `omp_get_max_threads()`, `off` = 1, an integer pins; strict full-string match, an unrecognised value is announced on stderr and becomes `auto`. Thread-local BLAS team of the GEMM handler call (`cpp/cblas/gemm_batch_ffi.cc`; a no-op on non-MKL BLAS). |
| `LORRAX_FFT_FFI_THREADS` | unset (`auto`) | machine | Same grammar, integer 1–4096. OpenMP team of the flat-k FFT chunk loop (`fft_flat_k_ffi.cc::team_threads`). |

### 2e. Compile cache

Read in `common/jax_compile_cache.py`, which arms JAX's own persistent cache
and adds only the namespace, the compile agreement and the receipt. What the
cache keys, what makes it cold and how it is pruned:
[compilation §3](../architecture/compilation.md#3-xla-programs); what it costs:
[§4](../architecture/compilation.md#4-where-the-time-goes).

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_CACHE_ROOT` | unset: `$SCRATCH/.cache/lorrax`, else `${XDG_CACHE_HOME:-~/.cache}/lorrax` | machine | The per-user cache tree (`lxkit.user_cache_root`): the JAX compile cache's default base, the NVRTC image cache and kernel store (`kconv_mathdx/images`), the self-check results and stamp (`perfcheck`), the `tests/hsuite` run tree, and `CUDA_CACHE_PATH` in `gpu_env.sh`. A non-blank value replaces the `SCRATCH`/XDG rule. Set it where that default is slow per file on compute nodes (the self-check fails "slow cache root" there), then rerun `lorrax-postinstall`. |
| `ISDF_JAX_CACHE_DIR` | unset: the runtime default, `$SCRATCH/.cache/lorrax/jax_compile/<jax/jaxlib>_<ffi bundle>_<key schema>/np{P}` | machine | The only location control besides `LORRAX_CACHE_ROOT`. Unset, the default namespace, pruned by rank 0. A non-empty value is the base directory, never namespaced or pruned; `np{P}` is still appended. Blank or whitespace opts out and clears JAX's `jax_compilation_cache_dir`, so an exported `JAX_COMPILATION_CACHE_DIR` does not apply either. Launchers and modules do not set it. |
| `JAX_COMPILATION_CACHE_MAX_SIZE` | `-1` (unlimited) | external | JAX's own control: an integer of at least `-1`; `0` turns the cache off, as the blank opt-out does. A positive cap (JAX's LRU eviction) runs at P=1 and refuses at P>1 with `UnsafeCachePolicy`, because the eviction rewrites an access-time file under a lock on every hit on every rank. |
| `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` | unset: LORRAX sets `0` whenever the cache is on | external | JAX's write threshold. JAX's own `1.0` default persisted 2 of 666 MoS2 bispinor executables; an exported value wins. |

### 2f. HDF5 and slab I/O

Why each default was chosen: [`architecture/slab_io.md#tuning`](../architecture/slab_io.md#tuning).
The C++ writer is `ffi/cpp/phdf5/context.cc`; the Python side is
`file_io/_slab_io_ffi.py`.

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_PHDF5_STRIPE_COUNT` | `clamp(P, 4, 128)` | machine | Integer; a non-integer or negative value refuses (both sides). Lustre stripe count, which is also the ROMIO aggregator count. Read by both sides; LORRAX no longer re-exports it (GATE env-frozen, `runtime.env_guard`). |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | power of two nearest `P/16` MiB, clamped to `[1, 4]` MiB | machine | `lfs setstripe -S` spelling with at most one suffix character; an unknown suffix refuses, and `4MiB` refuses. |

---

## 3. Debug and diagnostics

These leave production results unchanged unless the row says **A/B**; an A/B
switch marks the run as debug in its record. Stage-boundary telemetry and
refusals stay on by default; per-file and per-operation instruments are opt-in.

### 3a. Telemetry, logging and named overrides

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_MEMORY_RECEIPTS` | unset (production receipts) | debug | `calibrate`: every priced stage waits on its work at entry and exit, every section reads the allocator, and a BFC arena takes its mark's first raise (exact) or a 10 ms poll (`>=`). Production reads the allocator at section boundaries only (`lxkit.memory`, [memory model](../architecture/memory-model.md#receipts)). |
| `LORRAX_MEMORY_STRICT` | unset | debug | `1`: the census -- an unpriced figure is GATE memory_unpriced, and a price without provenance is a finding (`lxkit.memory.findings`); implies calibrate receipts. |
| `LORRAX_MEMORY_TRACE` | unset | debug | `<path>`: one JSON Line per closed section (live arrays by shape and creating section, the allocator, the executables that ran there), `<path>.p<i>` on process i > 0; no wait of its own (`lxkit.memory_residuals` reads it). |
| `LORRAX_NMU2_BACKUP` | unset | debug | `report`: GATE nmu2_no_backup (a whole N_mu-scale matrix on a rank with no mesh route) becomes a finding instead of a refusal, for a census. |
| `LORRAX_DEBUG_PRINT` | `0` (off) | debug | *bool*; the one print-verbosity switch a production driver honours, read by Python (`runtime`) and the native layer alike. `1` enables rank-0 timing-section entry/exit (depth 3), cache-miss explanations, kmeans helper detail, GW memory probes, healthy HDF5 inventories, PHDF5 open/close/timing/shard/hint detail, per-dataset restart-write receipts, and native FFT/convolution/BLAS provider diagnostics, plus the per-rank slab I/O timing logs (`slab_io_timing.rank*.log`), the cuSOLVERMp eigh per-phase profile and the ELPA pool report. I/O errors, conditioning receipts and collective-payload receipts print regardless; forensic sidecars and diagnostics that do extra numerical work have their own rows. |
| `LORRAX_STAGE_SYNC` | `0` | debug | *bool*. `1` restores every timing-only device wait (`common.timing.stage_sync`): section watchers (`timed(watch=True)`, `TimingSection.watch`), the `common.timing.stage_wait` boundaries (`sigma.input_wait`, `sigma.finalize_input_wait`, Σ_x, the moment batches, the ζ factor) and the Σ τ progress milestones, so every stage row is its own device time (profiling). Off, no hot path synchronizes for a clock: the total wall is exact and a stage row is host-attributed (device work lands in the first later stage that reads it). Changes no result. |
| `LORRAX_H5_JOURNAL` | `0` | debug | Three states: `1 true yes on` or empty is on, `0 false no off` is off, `sync`/`fsync` fsyncs every line; any other token refuses. Per-rank HDF5 operation journal (`file_io/h5_journal.py`), one line per open/close/create/read/write/attribute touch, written before the call at `file_io/hdf5_owner`, the `SlabIO` methods and `_slab_io_ffi`'s lifecycle calls. |
| `LORRAX_SANITY` | unset (`warn`) | guard | `0`/`off` skips the stage-boundary invariant checks (`common/sanity.py`); unset checks and warns without stopping; `strict` raises `SanityError` (CI and regression gates). Does not govern `sanity.refuse_nonfinite`. |
| `LORRAX_ALLOW_NONFINITE_RESULT` | unset (refuse) | debug | `1 true yes on` downgrades `common/sanity.py::refuse_nonfinite` (the refusal of a non-finite `kin_ion`, `Σ_total`, `E_qp` or `eqp` column about to be written) to a loud warning so the NaN artifact lands for forensics; anything else refuses. |
| `LORRAX_REDUCTION_GUARD` | unset | debug | `log` or `raise` (anything else refuses at start-up). Wraps `jax.lax.psum`/`psum_scatter`/`pmean` before the drivers import: a raw call on a floating operand over more than two ranks without an `# exact-reduce:` comment is recorded (`log`) or refused (`raise`), because its sum order follows the device topology ([collectives](../environment/transports.md#rank-order-reductions)). Changes no result. |
| `ISDF_JAX_PROFILE_DIR` / `ISDF_JAX_PROFILE_SECTIONS` | unset / unset (every section) | debug | Directory for `jax.profiler` traces of timed sections (`common/jax_profile.py`) / comma list of substrings selecting which sections open a trace. A session live across a multi-node PHDF5 collective write segfaults rank 0, so name the sections above P=4. |

### 3b. A/B switches, probes and test hooks

| var | default | class | grammar and effect |
|---|---|---|---|
| `LX_MACHINE` | unset (then `NERSC_HOST`) | test hook | Lower-cased machine name that `lxkit.testing.detect_machine` selects a test profile by, ahead of `NERSC_HOST`; an unknown name selects the `unknown` profile, which asserts nothing. |
| `LORRAX_TRS_CHECK` | `1` | guard | Stripped, case-insensitive: `strict` also refuses a broken or inconclusive reference; `0 false off no` refuse (`GATE retired_LORRAX_TRS_CHECK_off`), since skipping the measurement would assert time reversal. Automatic two-component DFT-reference measurement before any TRS-dependent consumer (`symmetry_maps/density_symmetry_check.py`); consumers read only `WfnLoader.trs_holds` → `SymMaps.trs_allowed`. |
| `LORRAX_CHECK_REPLICA` | `0` | debug | *falsy-set*. Re-enables `jax.device_put`'s cross-process equality assertion in `lxkit.placement.device_put_process_local`, paying the P-linear all-gather the helper avoids (7.8 GB/rank at P=64) to verify a host table is replica-identical. |
| `LORRAX_CD_STAR_CERTIFICATE` | `0` (off) | Symmetry certificate of the contour-deformation Sigma consumer (`gw.cd.sigma`): N > 0 appends one non-identity member of each of the first N multi-member k stars to the directly computed rows and compares them with the wedge-unfolded cube over (omega, m, n); the max deviation is printed and stored as `symmetry_certificate` in `cd_sigma_receipt.json`. Costs N/nk_red more contraction work; ignored on a full-BZ (identity-group) run. Debug only. |
| `LORRAX_CD_STOP_AFTER` | unset (off) | Instrument of the contour-deformation Sigma consumer (`gw.cd.sigma.parse_stop_after`): `imag:N,real:M,real_from:K`, each part optional. `imag:N` runs N imaginary chunks and leaves the axis, `real_from:K` starts the real axis at chunk K (negative from the end, ghost tail rebuilt as on a resume), `real:M` stops after M real chunks executed by this process. The run writes `cd_stop_receipt.json` (stage seconds, chunk counts and sizes, reach, checkpoint writes) and the leak ledger, then raises `CdStopAfter` on every rank; no Sigma is finalized. `imag` and `real_from` enter the `cd_checkpoint` identity (a probe checkpoint never continues a production run). Only under this variable may `cd_chi0_rules_file` be a partial (`--windows` census) build; a point of an unbuilt window refuses (`GATE cd_chi0_rule_unbuilt`). A malformed value refuses (`GATE cd_stop_after`). Timing probes only. |
| `LORRAX_LU_DEBUG_DUMP` | off | test hook | Array sidecar written by `services/distrib_la/bench/cusolvermp_solve_lu_test.py`; nothing else reads it. |
| `LORRAX_KDFT_KBOX_ARM` | unset (the rule) | debug, **A/B** | `single`, `split` (the rule's passes), `split2` or `split3`; anything else is the rule. Forces the kbox r-space kernel's arm where it fits (`ffi/fft.py`, read by Python and the kdft handler alike): the gate's and the bench's override. Changes no result beyond round-off. |
| `LORRAX_KDFT_OUTER` | `measured` | debug, **A/B** | Stripped, case-insensitive `measured` or `all`; anything else refuses. On a ROCm mesh, `all` routes the BSE outer load and its fused decode (`ffi/fft.py::make_local_kconv_klead_outer`, `make_local_kconv_klead_outer_decode`; `cpp/kdft/kconv_outer_rocm_ffi.cc`) to the kdft targets wherever the kernels fit, past the measured rule (`GATE kdft-kconv-outer-box`, `GATE kdft-kconv-outer-decode-slower`; `docs/architecture/kconv.md#bse-outer`); the gate and the bench (`tests/test_kdft_outer_gate.py`) set it. |
| `LORRAX_KDFT_KBOX_ARM` | unset (the arm rule) | debug, **A/B** | `single`, `split` (the rule's passes), `split2` or `split3` forces that k-box r-space arm where it fits (`ffi/fft.py` arm rule, mirrored by `cpp/kdft/kbox_rspace_ffi.cc` `arm_override`); an arm that cannot hold the grid falls to the rule and the build log names the arm taken; any other value is the rule. It is part of the prebuilt image key. The gate's and the bench's override ([k-convolution](../architecture/kconv.md)). |

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
`LORRAX_FFI_CRAY_DIR*`, `LORRAX_FFI_NVHPC_DIR*`, `LORRAX_FFI_FFTW_DIR*`,
`LORRAX_FFI_PHDF5_HOST`, `LORRAX_FFI_CRAY_HOST`, `LORRAX_FFI_NVHPC_HOST`,
`LORRAX_FFI_FFTW_HOST`, `LORRAX_FFTW_STAGE_CLOBBER`, `LORRAX_FFTW3_STAGE`,
`LORRAX_GATE_ONE_FFTW`, `LORRAX_GATE_FFTW_PY`, `CRAY_FFTW_PATH`,
`LORRAX_XLA_FFI_INCLUDE_DIR`, `LORRAX_XLA_FFI_HEADERS_DIR`, `LORRAX_HDF5_ROOT`,
`LORRAX_MPI_INCLUDE_DIR`, `LORRAX_MPICH_LIB_DIR`, `LORRAX_MPI_LIBRARY`,
`LORRAX_MPI_TYPE*`, `LORRAX_IMPI_ROOT`, `LORRAX_ICC_RUNTIME`, `LORRAX_MKL_ROOT`,
`LORRAX_SCALAPACK_LIBRARIES`, `LORRAX_NVHPC_*`,
`LORRAX_CUSOLVERMP_{STAGE,PIN}`, `LORRAX_CUBLASMP_PIN`, `LORRAX_DARSHAN_LIB_DIR`,
`LORRAX_XLA_CMDBUF`, `LORRAX_NNODES`, `LORRAX_NTASKS`, `LORRAX_NGPU`,
`LORRAX_SELECT_GPU`, `LORRAX_TIER2_WORKDIR`, `LORRAX_INPUT`,
`LORRAX_FFI_VERIFY_STRICT`, `LORRAX_BUILD_JOBS` (`8`; `16` for the CUDA leg),
`LORRAX_BUILD_PREFIX` (`<checkout>/.build`: the
cuSOLVERMp/cuBLASMp stage of the Perlmutter build scripts),
`LORRAX_MPICH_CONTAINER_DIR` (`/opt/udiImage/modules/mpich`),
`LORRAX_SAPI_EXTRA_INCLUDES`, `LORRAX_{PY,VENV_DIR,SRC_DIR}`,
`LORRAX_OVERLAY_BUILD_DIR`, `LORRAX_MKL_LIB`.

`LORRAX_CUDA_CHECK`, `LORRAX_LIB_CHECK`, `LORRAX_CUSOLVERMP_CHECK`,
`LORRAX_CUBLASMP_CHECK` and the `LORRAX_CFG_` family are C macros or CMake
variables, not environment variables.

### 4a. Perlmutter launch and site configuration

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_NVHPC_SUBPATH` | `0.9.1_cuda12.9/math_libs/12.9/lib64` (`config/perlmutter/site_config.sh`) | launch | The cuSOLVERMp stage a run loads. It selects a communication path, not only a version, and every stage exports the same SONAME, so a mismatch links cleanly and warns about nothing: read [`ffi_layout.md` §4](../architecture/ffi_layout.md) before changing it. |
| `LORRAX_NVHPC_ROOT` / `LORRAX_NVHPC_MOUNT` | none; `build.sh` refuses when unset | build | The stage the `.so` is compiled against, and its bind-mount point (`/lorrax_nvhpc`). |
| `LORRAX_PLATFORM` | `gpu` (`run_shifter.sh`) | launch | `gpu`, or `cpu`/`host`. Decides `MPICH_GPU_SUPPORT_ENABLED` for the Shifter launch; on the CPU leg it must be 0 or Cray MPICH aborts in `MPI_Init_thread`. |
| `LORRAX_MPICH_GPU_SUPPORT` | derived from `LORRAX_PLATFORM` | launch | `0` or `1`; anything else refuses. Carries the value across Shifter's mpich module, which unsets `MPICH_GPU_SUPPORT_ENABLED`. |
| `LORRAX_PHDF5_MPI_STACK` | `mpich` | build | Which MPI the PHDF5 FFI links and loads (`run_shifter.sh`, CMake). |
| `LORRAX_GATE_ONE_MPI` | `on` (`cpp/gate_one_mpi.sh`) | launch | The one-MPI-implementation-per-address-space gate; `off` disables it and says so on every run. |
| `LORRAX_CHECKOUT` | unset (the checkout containing cwd, else the module's snapshot) | launch | Path of a LORRAX checkout. `lx` runs its `src/` (`lxkit.launcher_policy.select_source_root`), and `runtime.source_closure` refuses a run whose imported runtime is not that checkout's. A path with no `src/runtime/__init__.py` refuses. |
| `LORRAX_RANK_THREADS` | set by `src/ffi/cpp/select_gpu.sh`, never by hand | launch | What the per-rank wrapper did to the rank's CPU threads, as one phrase: the affinity mask cut to one hardware thread per physical core (or why it was left alone: no SMT siblings in the mask, no sysfs topology, no `taskset`) and `OMP_NUM_THREADS` = cores − 2 with OpenBLAS and MKL at 1; or that the caller exported `OMP_NUM_THREADS` and nothing changed. The runtime only reads it into the startup CPU row. |
| `LORRAX_CUDART_SHIM` | set by `config/perlmutter/gpu_env.sh` outside a one-node allocation | launch | `<link>=<target>`: the node-local `/tmp/lorrax-$USER/cudart-hmem/<id>/libcudart.so` link to the venv's CUDA runtime that `gpu_env.sh` puts on `LD_LIBRARY_PATH` for libfabric; `select_gpu.sh` creates it on each node. |
| `LORRAX_CUBIN_STORE_WORKER` | set by `scripts/build_cubin_store.py --jobs`, never by hand | launch | `<w>/<n>`: this worker takes every n-th image task, starting at w. |
| `LORRAX_PREFIX` / `LORRAX_RUNTIME` | set by `<prefix>/lorrax-env.sh` (`install_deps.sh`) | launch | The install prefix, and `image` (the default install: every job runs through `<prefix>/bin/lorrax-run`) or `in-place`. `gpu_env.sh` makes no cudart link for `image` (lorrax-run makes one inside each node's restored image); `lorrax-postinstall` and `scripts/perf_check.py` read both to pick and to diagnose the launch. |
| `LORRAX_IMAGE_ROOT` | set by `lorrax-run`, never by hand | launch | The restored runtime image tree (`/tmp/lorrax-$USER/rt-<id>`); `lorrax-run` also sets `LORRAX_ROOT` to its `lorrax/` source and `VIRTUAL_ENV` to its venv. |
| `LORRAX_IMAGE_ROOTS` | `/tmp /dev/shm $TMPDIR /local/scratch /lscratch` | launch | Node-local directories `lorrax-run` restores the image into, in order, when the canonical parent (`/tmp/lorrax-$USER`) lacks room (about 9 GB during a restore); the tree is then linked at the canonical path. |
| `LORRAX_IMAGE_FALLBACK` | `<prefix>/unpacked` | launch | Where a node with no node-local room runs from: one extracted copy, made by the first such node and shared; it reads the install filesystem per file. |
| `LORRAX_IMAGE` / `LORRAX_IMAGE_STREAMS` / `LORRAX_IMAGE_CHDIR` | `<prefix>/current` / `16` / unset | launch | The archive `lorrax-run` restores; parallel 16 MiB read streams of the fetch; `source` runs the command from the image's source tree. |
| `LORRAX_RUN_DIR` | none; mandatory in the Frontera template | launch | The run directory holding the input deck, beside `LORRAX_ROOT` and `LORRAX_INPUT`. It does not select the compile-cache location. |
| `LORRAX_PM_{PRGENV,MPICH,CMAKE,FFTW,HDF5,HDF5_DIR,LIBSCI,LIBSCI_DIR,LIBSCI_FLAVOUR,MPICH_DIR}` | `config/perlmutter/site_config.sh` | build | Perlmutter module versions and prefixes for PrgEnv, cray-mpich, cmake, cray-fftw, cray-hdf5 and cray-libsci, consumed by the host builders. |
| `LORRAX_MPIWRAPPER_ROOT_DEFAULT` / `LORRAX_MPIWRAPPER_PREFIX_DEFAULT` / `LORRAX_MPIWRAPPER_COMMIT_DEFAULT` / `LORRAX_MPIWRAPPER_ABI_DEFAULT` | `$HOME/software/lorrax_mpiwrapper_cray` / `…/current` / the pinned v2.11.1 commit / `2.10.0` | build | Immutable, content-addressed Perlmutter releases and the atomic active symlink of the unmodified Cray-MPICH MPIwrapper adapter; used by the builder and the CPU-MPI prelude. |
| `LORRAX_FFI_CRAY_DIR` | `$HOME/software/lorrax_cray_runtime/stage` | build, launch | The Shifter route's Cray runtime stage (LibSci, GTL): written by `src/ffi/cpp/stage/cray_runtime_stage.sh`, bind-mounted at `/lorrax_cray` by `run_shifter.sh` and the lorrax modulefile. |
| `LORRAX_MPIWRAPPER_SO` | `$LORRAX_MPIWRAPPER_PREFIX/lib64/libmpiwrapper.so` on Perlmutter | launch | The Perlmutter prelude verifies the adjacent pinned-source/MPI-ABI/SHA256 manifest before exporting it as `MPITRAMPOLINE_LIB`. |

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
| `LORRAX_HOST_SCALAPACK` | `0` | build | `config/frontera/build_ffi_host.sh`: `1` builds and verifies the MKL ScaLAPACK group; `0` builds a parallel-HDF5-only host leg and says so. |

### 4c. Frontier OpenBLAS and ELPA builds (`config/frontier/build_openblas_seq.sh`, `build_elpa_hip.sh`)

| var | default | class | grammar and effect |
|---|---|---|---|
| `LORRAX_OPENBLAS_ROOT` | `$HOME/software/openblas_seq` | build | `build_openblas_seq.sh` installs the sequential, locking OpenBLAS into `$LORRAX_OPENBLAS_ROOT/install`. |
| `LORRAX_ROCM_BLAS_DIR` | `$HOME/software/openblas_seq/install` | build | `build_elpa_hip.sh`: the OpenBLAS prefix ELPA and its netlib ScaLAPACK link. |
| `LORRAX_ELPA_ROOT` | `$HOME/software/elpa_builds` | build | Directory under which the script installs the immutable, content-keyed prefix `elpa-2026.02.001-<key>` (read-only once complete; verified on rerun). |
| `LORRAX_ELPA_GFX` | `gfx90a` | build | GPU target of ELPA's AMD kernels; part of the prefix key. |
| `LORRAX_ELPA_MAKE_J` | `16` | build | Parallel jobs of the ELPA `make`. |
| `LORRAX_ELPA_OPENMP` | `1` | build | `1` builds ELPA with OpenMP (`libelpa_openmp`, threaded host stages); part of the prefix key. |
| `LORRAX_ELPA_ROCM_DIR` | unset (`build_elpa_hip.sh --print-prefix`; built when incomplete) | build | `config/frontier/build_ffi_rocm.sh`: the ELPA prefix the ROCM leg links (passed as CMake `-DLORRAX_ELPA_ROCM_DIR`). |

---

## 5. External variables LORRAX sets or reads

| var | LORRAX's handling |
|---|---|
| `JAX_ENABLE_X64` | `runtime.set_default_env` `setdefault`s `1`, and `runtime.set_x64_on_imported_jax` pushes the resolved value onto a jax imported earlier. A resolved `False` refuses; there is no override (FP64 cannot be switched off). |
| `JAX_PLATFORMS` | `setdefault` `<gpu>,cpu` with `<gpu>` the installed jaxlib GPU plugin's platform (`cuda` or `rocm`, `runtime._gpu_plugin_platform`), or hard-set to `cpu` by `set_default_env(platform="cpu")`; `lxkit.platform_from_env` reads it to pick the FFI library without initialising a backend. |
| `JAX_PLATFORM_NAME` | jax's deprecated spelling; `runtime` pops it at both CPU-downgrade sites, and `psp/get_DFT_mtxels.py` counts either spelling as the caller choosing a platform. |
| `JAX_PROCESS_COUNT` → `JAX_NUM_PROCESSES` → `OMPI_COMM_WORLD_SIZE` → `MV2_COMM_WORLD_SIZE` → `PMI_SIZE` → `SLURM_NTASKS` → `1` | Process-count chain (`runtime._WORLD_SIZE_ENVS`), first set wins. When it resolves 1 while any of them says more, startup refuses (`GATE split_launch`). |
| `JAX_PROCESS_INDEX` → `OMPI_COMM_WORLD_RANK` → `MV2_COMM_WORLD_RANK` → `PMI_RANK` → `SLURM_PROCID` → `0` | Process-index chain (`runtime._WORLD_RANK_ENVS`); an MPI launcher's own rank beats the `SLURM_PROCID` its ranks inherit under hydra/`ibrun`. |
| `JAX_COORDINATOR_ADDRESS` | Overrides the coordinator. Otherwise, in a Slurm step, the first host of `SLURM_STEP_NODELIST` on port 61440 + (`SLURM_JOB_ID` + 257 `SLURM_STEP_ID`) mod 4096; outside a step, jax's cluster detection, else startup refuses (`GATE coordinator_address`). |
| `CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` | The first one set gives the local device ids passed to `jax.distributed.initialize` (`runtime._visible_device_count`); each one set empty masks its vendor's GPUs for `runtime._gpu_is_present`. |
| `SLURM_NNODES` | Read by `gw/gw_init.py` for the ranks-per-node estimate of a local-backend capacity check (malformed → 1). |
| `SLURM_JOB_ID`, `SLURM_STEP_ID` | Key the coordinator port; recorded in response-bank and Galerkin receipts. |
| `JAX_EXPLAIN_CACHE_MISSES` | `0` (off): JAX's own switch; `1` prints JAX's per-trace cache-miss explanations. `LORRAX_DEBUG_PRINT` does not turn it on. |
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION` | Multi-process CPU runs require `mpi`; `runtime.announce_cpu_collectives()` refuses any other backend. Why: [`environment/transports.md`](../environment/transports.md). |
| `MPITRAMPOLINE_LIB` | MPItrampoline's adapter, absolute, set before JAX import: Frontera's patched Intel-MPI build, Perlmutter's unmodified upstream build from `config/perlmutter/build_mpiwrapper.sh`. A vendor `libmpi.so` is not an adapter. Contract: [the adapter](../environment/transports.md#adapter). |
| `LD_PRELOAD` | Perlmutter CPU-MPI: `config/perlmutter/cpu_mpi_env.sh` prepends `/opt/cray/pe/lib64/libpmi.so.0` and verifies it resolves under the Cray tree; foreign MPI, MPItrampoline/MPIwrapper and `libpmi2` entries refuse. The startup report records whether `libpmi.so.0` is preloaded. |
| `MPICH_GPU_SUPPORT_ENABLED` | Perlmutter machine setting, never set by `src/`: `0`, from `config/perlmutter/gpu_env.sh` (a clone's GPU steps source it), the `lorrax_A` module, or `cpu_mpi_env.sh` (CPU steps). The site's default `craype-accel-nvidia80` exports `1`, which aborts `MPI_Init` because neither FFI leg links Cray's GTL ([Perlmutter](../environment/machines/perlmutter.md#mpich-gpu-support)). |
| `CUDA_CACHE_PATH` | Perlmutter machine setting, never set by `src/`: `config/perlmutter/gpu_env.sh` defaults it to `cuda_jit` in the per-user cache tree, `$SCRATCH/.cache/lorrax/cuda_jit` (see `SCRATCH`; a caller's value is kept). CUDA's own default is `~/.nv/ComputeCache`; one fresh suite plus the P1 chain wrote 208 MB there. |
| `MPICH_ASYNC_PROGRESS` | The Perlmutter CPU-MPI prelude sets `1`, so Cray MPICH grants `MPI_THREAD_MULTIPLE` to XLA's FUNNELED request with one progress thread per rank; the prelude unsets `MPIR_CVAR_ASYNC_PROGRESS`. |
| `LORRAX_MPI_FORCE_THREAD_MAIN`, `LORRAX_MPI_FINALIZE_FIX` | Frontera MPIwrapper/overlay controls, unset in production; `runtime` reads both only to print them in the startup report. `common.collectives.warm_mesh_cliques()` creates every communicator from the main thread, and drivers using `runtime.finalize_process()` need no finalize overlay. |
| `OPENBLAS_THREAD_TIMEOUT` | `setdefault` `1` as the first statement of `runtime`; read once by each OpenBLAS copy when it is mapped, which is why every entry point (and every driver package `__init__`) imports `runtime` first. |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | Read for the startup report's thread table and oversubscription warning; LORRAX sets none of them. On Frontera the XLA:CPU thread pool ignores OMP; `taskset` is the pinning mechanism. |
| `SCRATCH` | Base of the NVRTC cubin cache ([compilation §2](../architecture/compilation.md#2-native-kernels-and-nvrtc-images)), of the JAX compile cache (§2e) and of the `tests/hsuite` run tree (`.cache/lorrax/hsuite`), and of `CUDA_CACHE_PATH` in `config/perlmutter/gpu_env.sh` (one rule, `lxkit.user_cache_root`), unless `LORRAX_CACHE_ROOT` is set. Where it is unset, missing or read-only the tree is `${XDG_CACHE_HOME:-~/.cache}/lorrax` instead. Read, never set. No quadrature rule is stored there or anywhere else. |
| `HDF5_USE_FILE_LOCKING` | `runtime.set_default_env` `setdefault`s `BEST_EFFORT` before any store opens, and `file_io/hdf5_owner` reports the value: h5py's opens lock where the filesystem supports `flock` and skip the lock where it does not. It governs only the serial h5py paths (the MPI-IO VFD takes no POSIX locks), so the FFI and h5py stacks cannot refuse each other through it. On Perlmutter NERSC's Slurm task prolog exports `FALSE` into every task without a value, so the default applies only where nothing exports one. Frontera `/work2` mounts node-local `localflock`: locks there hold within a node and are incoherent across nodes, which refuses nothing that `FALSE` would allow. |
| `MPLBACKEND` | `setdefault` `Agg` for headless plotting. |
| `FI_PROVIDER` | Not read by LORRAX. On Frontera CLX leave it unset (`LORRAX_MPI_PROVIDER=auto`) so Intel MPI picks `mlx` (provider costs: [transports §3](../environment/transports.md#intel-mpi-provider)). `fi_info` falsely reports `mlx` unavailable; trust the `libfabric provider:` line instead. In apptainer never `--bind /dev`. |
| `FI_PROVIDER_PATH` | Harnesses and `ffi_env.sh` pin `$IMPI/libfabric/lib/prov`; required in-container, where `mpivars.sh` is not sourced and `PMPI_Init` otherwise finds no provider. |
| `I_MPI_FABRICS` | Harnesses export `shm:ofi` (Intel MPI's own default, guarding against an inherited value). |
| `I_MPI_PMI_LIBRARY` | Required under `srun --mpi=pmi2` and set unconditionally there: TACC's login environment exports a PMI-1 library that is wrong for pmi2 and absent in the container. |
| `I_MPI_DEBUG` | `4` in every harness; the rank-0 `libfabric provider:` line is the only trustworthy provider observable, and it costs nothing after init. |
| `UCX_*` (`UCX_TLS` and the RC/DC/UD MLX5 timeout and retry settings) | Harnesses and `ffi_env.sh` `setdefault` the six TACC impi-module values (`UCX_TLS=knem,dc_x,rc` and the retry/timeout bumps); stripping them doubles a 1 MiB 32-rank allreduce (419 → 799 µs). Inherited values win; do not hard-pin (rtx has no `dc_x`). |
