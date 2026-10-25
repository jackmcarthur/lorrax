# Environment variables

`LORRAX_*` The few environment variables a user sets, and the developer switches.
{: .page-summary }

The environment grants machine capability: where the native libraries are, which site profile applies, how much memory a collective may take, where the compile cache lives, and how strict a check is. Physics and routing are input keys, validated, echoed and recorded with the run; set an input key before reaching for a variable. What a run resolved is printed in the rank-0 startup report, since several variables are read once before the JAX backend starts.
{: .lead }

A `bool` variable follows `runtime.env_flags.env_bool`: unset or blank gives the default; `1 true yes on` (any case) is on; `0 false no off` is off; any other value is off and announced once with `*** LORRAX SANITY`. Every other type is named in its entry, which says what a malformed value does. Defaults are read from the registry `docs/reference/env_vars.md` of the LORRAX tree and checked against the read sites; each entry carries its source line.

## Site and libraries {: #ffi }

A sealed deployment (the Perlmutter module, a cloud bundle) needs none of these: the loader finds the bundle manifest beside its `lib/` directory. The legs of the native library are described on [Installation](installation.md#ffi).

### `LORRAX_SITE_PROFILE` {: .kw data-type="env · string" data-default="unset" }
Names the site profile `config/<name>/site.toml`, ahead of `NERSC_HOST` and `LMOD_SYSTEM_NAME`.

The profile sets the network transport of a multi-node CUDA step ([Installation](installation.md#perlmutter)). A name with no such file refuses at startup.
{: .kw-details }

### `LORRAX_FFI_SO` {: .kw data-type="env · path" data-default="in-tree build" }
Path of the CUDA leg, `liblorrax_ffi.so`.

Unset, `src/ffi/cpp/build/liblorrax_ffi.so`. Pinning a leg of a sealed bundle requires this and `LORRAX_FFI_HOST_SO` to name the exact pair of its manifest; a partial, mixed, wrong-hash or wrong-ABI selection refuses.
{: .kw-details }

### `LORRAX_FFI_HOST_SO` {: .kw data-type="env · path" data-default="in-tree build" }
Path of the host leg, `liblorrax_ffi_host.so`.

Unset, `src/ffi/cpp/build_host/liblorrax_ffi_host.so`. Same sealed-pair rule as `LORRAX_FFI_SO`.
{: .kw-details }

### `LORRAX_FFI_ROCM_SO` {: .kw data-type="env · path" data-default="in-tree build" }
Path of the ROCm leg, `liblorrax_ffi_rocm.so`.

Unset, `src/ffi/cpp/build_rocm/liblorrax_ffi_rocm.so`. It takes the place of `LORRAX_FFI_SO` on an AMD GPU, and a sealed bundle pairs it with `LORRAX_FFI_HOST_SO` under the same rule.
{: .kw-details }

### `LORRAX_FFTW3_SO` {: .kw data-type="env · path" data-default="unset" }
File the host FFT engine is loaded from, tried before the built-in candidate list.

A bad path is skipped silently; no engine at all refuses at the first host FFT, naming every candidate. Nothing at run time checks that the file is a CPU FFTW3.
{: .kw-details }

### `ISDF_JAX_CACHE_DIR` {: .kw data-type="env · path" data-default="namespaced under $SCRATCH" }
Location of JAX's persistent compile cache, which LORRAX arms with a namespace, a cross-rank compile agreement and a receipt.

Unset, `$SCRATCH/.cache/lorrax/jax_compile/<jax>_<ffi bundle>_<key schema>/np<P>`, pruned by rank 0. A non-empty value is the base directory (`np<P>` still appended, never pruned); blank turns the cache off.
{: .kw-details }

## Memory {: #memory }

Caps that depend on the machine's memory. None changes a result beyond round-off.

### `LORRAX_COLLECTIVE_CHUNK_MB` {: .kw data-type="env · float MiB" data-default="128" }
Upper bound on the payload of one emitted collective.

Applies to the ζ-fit collectives, the distributed W Dyson build, the rank-0 gather of k-partitioned sweeps and the $V_\bq$ G-panel width. A malformed value uses the default; `≤ 0` is unbounded.
{: .kw-details }

### `LORRAX_PPM_FIT_ARENA_GIB` {: .kw data-type="env · float GiB" data-default="unset" }
Caps the bytes the GN-PPM fit prices its q block against.

Unset, the q block is the fixed tile. Below one q it warns and runs one q. Bit-exact at any block.
{: .kw-details }

## Wavefunction reads {: #io }

### `LORRAX_WFN_BACKEND` {: .kw data-type="env · enum" data-default="unset (auto)" }
Forces the transport that reads `WFN.h5`.

`eager` reads with h5py on each process; `phdf5` reads collectively through the FFI. `phdf5_host` refuses; any other value takes the automatic pick, which refuses at more than one process when no FFI library loads (`eager` is the way through).
{: .kw-details }

`eager` `phdf5`
{: .kw-values }

## Checks {: #checks }

Each sets how hard a check stops the run. Neither changes a number.

### `LORRAX_SANITY` {: .kw data-type="env · enum" data-default="unset (warn)" }
Stage-boundary invariant checks.

Unset, they check and warn; `strict` refuses (CI and regression gates); `0` or `off` skips them. Does not govern the non-finite-result refusal.
{: .kw-details }

`strict` `off`
{: .kw-values }

### `LORRAX_TRS_CHECK` {: .kw data-type="env · enum" data-default="1" }
The two-component time-reversal measurement on the occupied states, run before any consumer uses time reversal.

`strict` also refuses a broken or inconclusive reference. `0 false off no` refuse: skipping the measurement would assert time reversal ([Required DFT inputs](inputs-from-dft.md#spinors)).
{: .kw-details }

`1` `strict`
{: .kw-values }

## Frontera job template {: #launch }

Read by `config/frontera/templates/gw_dev.sbatch`, not by Python ([Quickstart](quickstart.md)).

### `LORRAX_ROOT` {: .kw data-type="env · path" data-default="none" }
The LORRAX checkout. Required by the Frontera template; set by the Perlmutter module.

### `LORRAX_RUN_DIR` {: .kw data-type="env · path" data-default="none" }
The run directory holding the input file. Required by the Frontera template.

### `LORRAX_INPUT` {: .kw data-type="env · path" data-default="gw.in" }
The input file the Frontera template passes to `-i`.

## Developer variables {: #developer }

For debugging, profiling and tuning a new machine; never set in a production run.

??? note "Developer variables"

    | Variable | Default | Effect |
    |---|---|---|
    | `LORRAX_CHECKOUT` | unset | The checkout whose `src/` the launcher runs. A path without `src/runtime/__init__.py`, or a runtime imported from elsewhere, refuses. |
    | `LORRAX_DEBUG_PRINT`{: #lorrax_debug_print } | `0` | The one verbosity switch, read by Python and the native layer: timing-section entry and exit, cache-miss explanations, memory probes, HDF5 inventories and timings, the per-rank slab-I/O timing logs, native-provider diagnostics, the cuSOLVERMp eigensolver phase profile, the ELPA pool report, the per-rank GPU binding of `select_gpu.sh`. |
    | `LORRAX_STAGE_SYNC` | `0` | Restores every timing-only device wait, so each stage row of the timing table is its own device time. |
    | `ISDF_JAX_PROFILE_DIR` | unset | Directory of `jax.profiler` traces of the timed sections. |
    | `ISDF_JAX_PROFILE_SECTIONS` | every section | Comma list of substrings selecting the sections that open a trace. Name them above four processes: a trace live across a multi-node parallel-HDF5 write crashes rank 0. |
    | `LORRAX_H5_JOURNAL` | `0` | Per-rank journal of every HDF5 open, close, read, write and attribute access, written before the call. `1 true yes on` or blank is on; `sync` also fsyncs each line; any other value refuses. |
    | `LORRAX_REDUCTION_GUARD` | unset | `log` records and `raise` refuses unordered floating-point `psum`, `psum_scatter` and `pmean` over more than two ranks; any other value refuses at startup. |
    | `LORRAX_CHECK_REPLICA` | `0` | Re-enables JAX's cross-process equality assertion when a host table is placed on devices, at the cost of an all-gather. |
    | `LORRAX_ALLOW_NONFINITE_RESULT` | off | Writes a non-finite `kin_ion`, Σ, $E^{\mathrm{QP}}$ or `eqp` column with a warning instead of refusing, so the file lands for forensics. |
    | `LORRAX_RANK_POLICY`{: #lorrax_rank_policy } | `refuse` | `warn` or `off` lets a ζ rank truncation that left the certified regime continue, to inspect it; any other value refuses. |
    | `LORRAX_MKLBLAS_THREADS` | `auto` | BLAS team inside the GEMM handler (MKL only): `auto` is the ambient OpenMP maximum, `off` is 1, an integer pins it. |
    | `LORRAX_FFT_FFI_THREADS` | `auto` | OpenMP team of the host FFT chunk loop; same grammar, integers 1–4096. |
    | `LORRAX_PHDF5_STRIPE_COUNT` | clamp(P, 4, 128) | Lustre stripe count of new files, which is also the MPI-IO aggregator count; a non-integer or negative value refuses. |
    | `LORRAX_PHDF5_STRIPE_SIZE_FS` | P/16 MiB, 1–4 | Lustre stripe size in `lfs setstripe -S` spelling (`4M`); an unknown suffix refuses. |
    | `LORRAX_BANDS_GEMM_FFI` | `1` | On a CPU mesh the band contractions run through the vendor-BLAS batched-GEMM handler of the host leg, and a missing handler refuses at startup. `0` is an announced debug opt-out onto XLA's own dot lowering, which runs 1.6 to 1.9 times slower. An unrecognised value is announced and taken as the default. A GPU run ignores it, since XLA already calls cuBLAS there. |
    | `LORRAX_KDFT_OUTER` | `measured` | On a ROCm mesh, `all` sends the BSE outer load and its fused decode to the kdft kernels wherever they fit, past the measured routing rule. The ROCm gate tests and benchmarks set it. Any value other than `measured` or `all` refuses. |
    | `LORRAX_KDFT_KBOX_ARM` | unset (the rule) | `single`, `split`, `split2` or `split3` forces that arm of the ROCm k-box r-space kernel where it fits, for the gate tests and benchmarks. An arm that cannot hold the grid falls back to the rule, and any other value is the rule. |
    | `LORRAX_GPU_DEVICE` | unset | Comma list of device ordinals, one per local rank, that `select_gpu.sh` uses ahead of `CUDA_VISIBLE_DEVICES` and the Slurm GPU lists. A launcher sets it to place several one-GPU steps on different devices of a node, which Slurm does not do. A list with no entry for some rank refuses. |

## Variables of other software {: #external }

LORRAX reads or sets these; the vendor's documentation owns their meaning.

| Variable | LORRAX's handling |
|---|---|
| `JAX_PLATFORMS` | Set to `<gpu>,cpu` unless exported, with `<gpu>` the installed plugin's platform (`cuda` or `rocm`); set to `cpu` by a CPU-only entry point. Also selects the FFI leg. |
| `JAX_ENABLE_X64` | Set to `1` unless exported, and pushed onto a JAX imported earlier. A resolved off refuses. |
| `CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` | The first one set gives the local device ids; an empty one masks that vendor's GPUs. `select_gpu.sh` sets `CUDA_VISIBLE_DEVICES` per rank. |
| `XLA_PYTHON_CLIENT_ALLOCATOR`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_CLIENT_MEM_FRACTION` | On a GPU run where neither of the first two is exported, LORRAX sets the leg's allocator (CUDA: `cuda_async`), preallocation and its pool fraction. Exporting either makes the caller own the pool; inconsistent combinations and both fraction spellings at once refuse. |
| `XLA_FLAGS` | `rematerialization` is added to `--xla_disable_hlo_passes`; on an NVIDIA GPU, `--xla_gpu_autotune_level=0` and `--xla_gpu_enable_triton_gemm=false` are appended unless present (on AMD, the autotune flag and `--xla_gpu_disable_async_collectives=ALLCOLLECTIVES`). A caller's value wins flag by flag. |
| `JAX_COORDINATOR_ADDRESS` | Overrides the coordinator. Otherwise, in a Slurm step, the first host of `SLURM_STEP_NODELIST` on a port derived from `SLURM_JOB_ID` and `SLURM_STEP_ID`; with neither, startup refuses. |
| `JAX_PROCESS_COUNT`, `SLURM_NTASKS` and the MPI equivalents | Process count, first set wins. A count of 1 while any of them says more refuses (`GATE split_launch`). |
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION` | A multi-process CPU run requires `mpi`; any other value refuses. |
| `JAX_COMPILATION_CACHE_MAX_SIZE` | `0` turns the persistent cache off; a positive cap refuses at more than one process. |
| `NCCL_NET`, `NCCL_NET_PLUGIN` | Either one, even empty, keeps the caller's transport; `NCCL_NET=Socket` refuses on a multi-node profiled CUDA step. Otherwise the site profile sets them and the other NCCL and libfabric `FI_*` variables it names (Perlmutter: `FI_CXI_*` and `FI_MR_CACHE_MONITOR = kdreg2`). |
| `SLURM_NETWORK` | Must contain the profile's step network (`no_vni` on Perlmutter) on a multi-node CUDA step, or startup refuses. |
| `NERSC_HOST`, `LMOD_SYSTEM_NAME` | Select `config/<name>/site.toml` when that file exists, after `LORRAX_SITE_PROFILE`. |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | Printed in the startup report with an oversubscription warning; never set. |
| `HDF5_USE_FILE_LOCKING` | Set to `BEST_EFFORT` unless exported; NERSC's task prolog exports `FALSE`. |
| `MPICH_GPU_SUPPORT_ENABLED` | Must be `0`: neither FFI leg links Cray's GTL. Set by `config/perlmutter/gpu_env.sh` and the module, never by Python. |
| `SCRATCH` | Base of the compile cache; `~` when unset. |

