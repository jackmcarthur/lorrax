# Compilation: what is built, when, and at what cost

A LORRAX run executes three kinds of compiled code, each built when its key
is new to the cache that holds it. This page gives each key, where the result
is kept, what makes it cold, what it costs, and how to read a run. The
variables are owned by [the registry](../reference/env_vars.md#2e-compile-cache).
Code is cited as `file:line`; every measurement names its deck, $P$ and claim.

## Terms

| term | meaning |
|---|---|
| $P$ | ranks, one GPU each, on the square mesh |
| program | one XLA executable: one jitted function at one key (§3) |
| lowering | tracing a function to a jaxpr, then lowering the jaxpr to a StableHLO module |
| request | one call of JAX's `compile_or_get_cached` on one rank: a persistent-cache read, or a compile |
| real compile | one XLA backend compile (`backend_compile_and_load`) |
| cubin cache | the per-user, writable image directory, `ffi.fft.cubin_cache_dir()` (`src/ffi/fft.py:424`): `$SCRATCH/.cache/lorrax/kconv_mathdx`, or `~/.cache/lorrax/kconv_mathdx` where the site defines no `SCRATCH` |
| store | a release's read-only image directory, `ffi.fft.CUBIN_STORE`: `cubin_store/` at the source root. A checkout has none unless one is built into it |
| fully cold, release cold, warm | no store and both caches empty; the release's store present and the JAX cache empty (a release's first run); both caches hold every key the run needs |
| hsuite | `tests/hsuite`, its 16 default stages at P4 on one A100-80 node, with the run directory, `TMPDIR`, `SCRATCH` and both caches node-local |

## 1 The three kinds

| kind | built by | key: what forces a new build | kept in | cost |
|---|---|---|---|---|
| native kernels | nvcc and the host compiler, at install | the bundle | the sealed FFI bundle | none on a listed architecture; else a driver PTX JIT (§2) |
| NVRTC images (cubins) | NVRTC, inside the FFI, on first use | source, headers, deciding options, toolchain and device (§2); never $N_\mu$ or $P$ | the cubin cache, seeded from the store | about 7 s each (§2) |
| XLA programs | JAX and XLA, in every process | function, abstract values, shardings, static arguments and trace context (§3) | process memory; the persistent cache, one directory per namespace and $P$ | §4 |

## 2 Native kernels and NVRTC images

**Native kernels.** The CUDA leg is compiled once per bundle, for the
architectures [Building the FFI libraries](../installation/ffi-build.md#with-cmake-directly)
lists. On any other device the driver JIT-compiles the PTX at first load into
`CUDA_CACHE_PATH` ([registry §5](../reference/env_vars.md#5-external-variables-lorrax-sets-or-reads)),
at a cost not measured.

**NVRTC images.** One service, `src/ffi/cpp/common/rtc_build.h`, builds
every kernel compiled at run time: the k-convolution family, its BSE outer
kernels, and the Fourier plan's fused pair. The same source is the ROCM leg's
service with the vendor spelled by the leg (`common/leg.h`): hipRTC, resolved
beside the HIP runtime the process mapped, for the device's full target id
(`--offload-arch=gfx90a:sramecc+:xnack-`), keyed `hiprtc<maj>.<min>@<libhiprtc
real path>`, and stored as `kdft_m<mode>_…_gfx90a_<key>.hsaco` in the same cache
directory ([FFI layer](ffi_layout.md), the kdft family). The rest of this
paragraph is the CUDA leg's. It compiles for the device's own
`sm_<cc>`, keeps each image in process, stores it in the cubin cache and
loads it through the driver API. The cubin cache is always on, has no knob,
and serves every world size.

| part of the key | what it is |
|---|---|
| program | FNV-1a over the embedded source, the text of each embedded header and the NVRTC options that decide the image (`rtc_build.cc`, `key`). Editing a source or header, comments included, invalidates its images |
| toolchain | `nvrtc::mathdx_toolchain`: the cuFFTDx or cuBLASDx, commonDx, CUTLASS and CCCL version headers, the wheel's dist-info name, and the NVRTC version with the loaded `libnvrtc`'s real path. A version header that reads empty disables the disk cache for that build rather than leaving the key |
| kernel constants | the k-convolution modes and BSE outer kernels: [k-convolution §14](kconv.md#build-and-cache); the Fourier pair: its two GEMM extents and supports (`fourier_plan_cuda_ffi.cc:249`) |
| not keyed | file names, include paths, the host code, $N_\mu$, $P$ |

| step | rule |
|---|---|
| file | `kconv_m<mode>[w<variant>]_<nkx>x<nky>x<nkz>_ns<ns>[x<n_r>][_c64]_sm<XY>_<key>.cubin`, `kconv_outer[_dec]_…` or `plan_pair_…`, framed by a `LRXKCONV1` header with the key and a payload hash. A k-convolution image is about 8.5 MB on sm_80, 6.1 MB of it line info (claim 4127) |
| write | to a unique temporary, renamed into place (`rtc_build.cc`, `disk_store`), so concurrent ranks each publish a whole file |
| read | a torn, foreign or non-ELF file is recompiled over the entry; one the driver refuses to load is unlinked first (`:251`) |
| cost | about 7 s per image (claim 4129), on every rank that misses: nothing is shared until a file lands |
| store | a process's first `cubin_cache_dir()` call links every store image the cache lacks into it (`ffi.fft.seed_cubin_cache`, `src/ffi/fft.py:451`) under a temporary name, renamed into place; a present entry is kept, a dangling link replaced. A bad store image is rebuilt into the cache over the link; the store file is never written. With the hsuite's 25-image store a cold run builds no image (claim 4126) |
| building a store | After a successful P4 hsuite, module acceptance snapshots its native-identity cache into the candidate's `cubin_store/`; publication verifies and copies the regular files and hashes into the clean source archive. The cache may also contain images from earlier runs; each native request still checks its own image key. No extra grids are compiled. For a larger store, `scripts/build_cubin_store.py` builds the 330 k-grid-keyed images of its `GRIDS` table with a fresh `SCRATCH`; smoke decks add the system-keyed ones: 392 images, 3.33 GB (claim 4127) |
| another architecture | the `sm<XY>` in each name makes a store per-architecture: an A100 store gives an H100 run nothing. A device the installed cuFFTDx cannot target fails the startup probe, and every factory takes the XLA backend, announced on rank 0 and recorded as a demotion in every compile receipt and in the startup facts (`GATE mathdx-probe`, [k-convolution §13](kconv.md#refusals)) |

**How the two caches meet.** Every mathdx custom call (the k-convolution
family and the Fourier pair) carries two string attributes: `cubin_dir`, and
`mathdx_root`, the wheel's install directory (`_mathdx_common`,
`src/ffi/fft.py:418`). Both are in the HLO, so both enter the JAX key and the
agreement fingerprint (§3) of every program holding such a call: a new
`SCRATCH` or a moved venv misses every such program, and ranks whose strings
differ are refused. A store changes neither string, so it changes no JAX key
(claim 4126).

## 3 XLA programs

**One call.** A jitted function's in-process key is the jit object, the
abstract value (shape, dtype) and sharding of every argument, the static
arguments, and the trace context. The first call at a new key takes this
path; later calls in the process reuse the executable.

```text
call ─► in-process hit? ─yes─► execute
            │ no
            ▼
          trace ─► lower ─► request: fingerprint, agreement slot, persistent key
                                │
                     hit ◄──────┴──────► miss: agreement check ─► XLA compile ─► rank 0 writes
                      │                                                │
                      └──────────────► load ─► execute ◄───────────────┘
```

A jit built inside a function is a new object on every call, so it lowers on
every call; `jnp.zeros(..., device=sharding)` lowers once per (shard shape,
dtype) instead (claim 4153). A `with mesh:` block would put its mesh into the
trace context and re-trace held programs, so the runtime sets
`jax_disallow_mesh_context_manager` (`src/runtime/__init__.py:262`), so a
`with mesh:` block raises (claim 4194).

**Every process lowers.** In production each driver is its own process, and
the in-process cache dies with it. A warm persistent cache saves the XLA
compile only: JAX needs the module to compute its key. The mesh warm-up
compiles before the cache is armed and is never stored. A planner's candidate
compiles (`runtime.aot_memory.check_chunk`: once, and once more over the room;
`step_up`: one per step) are ordinary requests.

**The persistent cache.** `common.jax_compile_cache` arms JAX's own cache once
per process (`ensure_jax_compile_cache`, step 7 of
`runtime.initialize_communicator_stack`).

| aspect | rule |
|---|---|
| key | JAX's: the module, the compile options, `XLA_FLAGS`, jaxlib and the backend |
| location | `$SCRATCH/.cache/lorrax/jax_compile/<namespace>/np{P}`. The namespace names jax, jaxlib, the FFI bundle and the key schema (`cache_namespace`), never the LORRAX commit, which reaches a program only through its HLO |
| writes | process 0 only, every program it compiles (threshold 0 s). Every rank reads, so at $P > 1$ the directory must be on a filesystem every node sees |
| ranks | on a miss every rank compiles the module itself (claim 683). Every rank asks for rank 0's key. JAX strips the device assignment from a GPU key. A CPU key also carries the device assignment and process-local device records. At $P > 1$ `_install_rank_invariant_cpu_key` removes both while retaining platform and target-machine bytes. On XLA 0.12 the CPU records are inside the authenticated `PjRtTopologyDescriptionProto`/`Any` envelope; an unknown or malformed serialization keeps JAX's original identity. JAX's per-fusion XLA caches are off at $P > 1$ |
| host callbacks | stored like any other program since JAX 0.12 (`compiler._cache_write` refused them through 0.9). A checked dense solve stays callback-free inside [`distrib_la.checked_program`](../services/distrib_la/api.md#dense-factorizations), which spares it a host round trip per call |
| pruning | default location only. Rank 0 touches every entry it reads and, in a daemon thread at most every 6 h, removes entries and other namespaces unused for 7 days, and the least recently used namespaces earlier past 2 GiB or 200 000 files, never one used in 5 days |

| change | what goes cold |
|---|---|
| jax or jaxlib | a new namespace |
| the FFI bundle; for an unsealed library, any rebuild (its SHA-256, `_library_identity`) | a new namespace |
| `_KEY_SCHEMA` in `common/jax_compile_cache.py` | a new namespace |
| `XLA_FLAGS`, except the `--xla_dump_*` family and the few others JAX excludes | every key |
| $P$ | a different `np{P}` directory |
| system size, k grid, band counts | every key whose shapes change |
| `SCRATCH` | the default location, so every key; under an explicit `ISDF_JAX_CACHE_DIR`, every mathdx program (§2) |
| the mathdx wheel's path | every mathdx program (§2) |
| LORRAX source | only the programs whose HLO changes |

**The cross-rank compile agreement.** At $P > 1$ with a coordination client,
each request publishes its module's fingerprint (SHA-256 of the MLIR without
debug locations, process-local `…handle` literals canonicalised) under the
rank's request number (`_requested`). Numbers are taken in program order on
the calling thread, under one lock that covers only the numbering and the
publish, so ranks that request the same programs in the same order agree.
Only a real compile checks (`_check_compile_record`): rank 0 reads every
rank's record, a peer reads rank 0's, and a mismatch refuses before the
compile, naming every rank's key, instead of hanging in the next collective.
A cache hit checks and waits for nothing. A check waits without bound by
default ([registry](../reference/env_vars.md#2e-compile-cache)). The agreement
costs 6.2 s on the warm hsuite (161.9 against 155.7 s, claim 4130) and stays on.

**The runtime's XLA flags.** The runtime merges the GPU leg's declared
compile flags (`PlatformLeg.xla_flag_defaults` in `lxkit.LEGS`, merged by
`runtime.set_default_xla_gpu_autotune`) and one HLO-pass flag into
`XLA_FLAGS`; a caller's value wins, flag by flag. The leg is the one
`JAX_PLATFORMS` names, else the installed plugin's. The leg flags are
GPU-only; a CPU run gets none.

| flag | leg | effect | claim |
|---|---|---|---|
| `--xla_gpu_autotune_level=0` | CUDA | cold compile −12.9 % (Na) and −16.3 % (Si) at P4, execution within noise | 683 |
| `--xla_gpu_enable_llvm_module_compilation_parallelism=true` | CUDA, no longer set | measured on JAX 0.9.1: XLA split each module's LLVM IR into parts compiled in parallel; hsuite release-cold XLA compile 219.5 → 195.2 s. XLA removed the flag before jaxlib 0.12 (openxla/xla 194f1fe5d9) and aborts on it, so the leg no longer declares it; the 0.12 cold compile is unmeasured | 4179 |
| `--xla_gpu_autotune_level=0` | ROCM | MI250X P4 seeded cold: XLA compile gnppm 21.8 → 18.3 s, exciton_bands 15.1 → 13.3 s against level 4; warm walls within noise; d/zgemm n = 1024–8192 steady state identical (zgemm 8192: 39.6 TF/s at both) | runtime lane 2026-10-14 (pool logs runtime_a2_flags, runtime_a3) |
| `--xla_gpu_disable_async_collectives=ALLCOLLECTIVES` | ROCM | every collective runs on the compute stream, in program order. With async collectives an all-reduce on the collective stream and a collective of another communicator on the compute stream are in flight together, and RCCL deadlocks across ranks (all GCDs at 100 %, every rank waiting on the same executable): `qsgw_head._drude_tensor_kernel` (psum over x,y beside psum over y) hung bisp_sc 3/3, the Coulomb-root program hung sp_export 1/1; a 25-line repro of that kernel hangs at iteration 1 and runs 300/300 with the flag. bisp_sc PASS 189.2 s with it (A100 173.7 s) | disteigh lane 2026-10-15 (pool logs disteigh_bisp1-3, disteigh_repro_async/sync) |
| `rematerialization` added to `--xla_disable_hlo_passes` (`runtime.disable_xla_rematerialization`) | all | On a GPU leg with a readable card total (CUDA, ROCm) a module that then does not fit the device refuses before it runs ([memory model](memory-model.md#module-does-not-fit)); on CPU nothing checks | — |

Not declared for ROCM, each measured on MI250X at P4 (gnppm XLA compile,
278 programs): `--xla_gpu_force_compilation_parallelism=7` 18.4 s against
18.3 s, and the LLVM-module flag (since removed from XLA), which XLA's AMDGPU
compiler could not act on (it does not link split modules). Where a ROCm compile goes (gnppm, XLA
VLOG timers): HLO passes 4.0 s, backend 12.1 s, of which LLVM AMDGPU codegen
and HSACO link (`CompileToHsaco`) 6.1 s. Per program that is about 66 ms,
against 88 ms per program on the A100-80 release-cold hsuite (claim 4212), so
compile is not a ROCm-specific cost; the persistent cache removes it there
as on CUDA (keys are rank-invariant on ROCm too: JAX reports
`backend.platform == "gpu"` and strips the device assignment).

**Vendor run-time kernels (ROCm).** Two ROCm libraries compile or load
device code at run time, outside XLA and outside JAX's cache; the leg declares
both and `runtime.vendor_kernels` implements them (CUDA and the host declare
none). rocFFT builds each new FFT kernel with hipRTC (1.9–7.6 s per length per
process on MI250X; the pip wheel ships no system kernel cache). Its kernels are
keyed by name, arch, HIP version and generator checksum, so, like the kconv
image store, they live in `lxkit.user_cache_dir("rocfft")` and outlive the
compile-cache namespace (`PlatformLeg.runtime_kernel_stores`): rocFFT reads a
read-only snapshot (`ROCFFT_RTC_SYS_CACHE_PATH`), writes new kernels to one
SQLite file per process (`ROCFFT_RTC_CACHE_PATH`; shared writers on Lustre
spin), and each process folds its file into the snapshot at exit (copy, insert,
rename). From the store the first FFT of a length costs 0.02–0.04 s. hsuite
kmeans→dipole builds 16 kernels per process, 10.0 s of the cold kmeans wall.
A "cold" hsuite run (empty compile cache) therefore reads both vendor-kernel
stores, kconv and rocFFT, as a CUDA cold run reads the kernels cuFFT ships.
rocBLAS loads its Tensile tables on a process's first GEMM: 2.84 s for an 8x8
dgemm, then 0.055 s for the first zgemm (P1 probe, autotune 0, either level).
`PlatformLeg.blas_host_warmup` compiles one 8x8 GEMM per dtype at start-up
(main thread, so the compile agreement sees the same programs on every rank)
and runs them on a helper thread, where the load overlaps the driver's
imports. Before that thread starts the process environment freezes
(`runtime.env_guard`, `GATE env-frozen`): the thread's `hipModuleLoadData`
calls `getenv` inside HSA, and glibc's `getenv` takes no lock, so a main-thread
`os.environ` write that adds a key (and so reallocates `environ`) is a
use-after-free there. That was the intermittent rc 139 start-up SIGSEGV
(Frontier, 2026-10-16: 6 of 72 concurrent Fe start-ups, from the SlabIO
striping export at the first WFN.h5 open; evidence in `runtime.env_guard`). Every LORRAX
environment write therefore happens before step 7b; a later LORRAX write that
adds a key refuses, any other code's is announced. With the SlabIO export removed and the
freeze armed, the same stress (6 concurrent 4-rank start-ups over 3 nodes, hot
and cold caches) gave 0 of 48 crashes (40 of them through the crash window; the other 8
sat on frontier00749, whose GCDs 4-7 stall in the same window and is now excluded from
the pool), and the Fe deck's eqp0/eqp1 were
unchanged bit for bit (branch `crash/w8`). AMD comgr's code-object cache stays per namespace
(`runtime_kernel_caches`, `<cache>/vendor/comgr`). Runtime start-up itself
(`initialize_communicator_stack`, 5.3–8.7 s per process: jax import ~2 s,
distributed and backend init ~1.5 s, clique warm-up 2.6 s) is not cached.

## 4 Where the time goes

**The hsuite.** One A100-80 node, P4, node-local:

| state | wall | compile | claim |
|---|---|---|---|
| release cold | 323.2 s | 2261 real XLA compiles, 199.2 s | 4212 |
| warm | 138.5 s | 3037 requests per rank | 4212 |
| fully cold | release cold plus the NVRTC builds | 25 NVRTC images, 180.8 s | 4129 |

**A warm run still pays the compile path.** At 4030 requests per rank the warm
hsuite spent 76.3 s on it: trace 18.7 s, lower 34.3 s, cache key and read
23.5 s (claim 4129). Fewer lowerings, not a different cache, move it. The
census (claim 4155) finds three classes: a later driver re-lowering an earlier
one's program (957; the hsuite clears JAX's in-process caches between stages,
as production's one process per driver does); a stage re-lowering its own key
(per-call jits, eager glue, nested mesh contexts, mostly removed: claims 4152,
4154, 4194); and first lowerings, of 2576 distinct keys from 1136 functions.
The top 10 of 2655 call sites hold 29 % of the compile-path seconds and the
top 100 hold 65 % (claim 4129).

**Production decks.** CrI3 6×6 and Fe 4³ bispinor SC, forced face route, P4,
one A100-80 node, cold JAX cache, maps 0–2, without and with the LLVM flag;
no leg builds an NVRTC image, and eqp is identical in 6 of 6 files
(claim 4179):

| deck | XLA compile, map 0 | later map | rank-0 wall | device peak |
|---|---|---|---|---|
| CrI3 6×6 | 156.6 → 134.3 s | map 1: 52.7 → 43.1 s | 590.2 → 573.9 s | 44.76 GB, unchanged |
| Fe 4³ | 155.0 → 133.6 s | map 2: 53.0 → 42.4 s | 473.9 → 430.3 s | 21.70 GB, unchanged |

## 5 Compile threads

A jitted function compiles on the thread that makes its first call. XLA's
backend compile releases the GIL: the hsuite's 75 largest programs, with all
four ranks compiling and 16 cores per rank, take 104.0 s serial and 55.8,
29.1, 16.4 and 12.4 s on 2, 4, 8 and 16 threads, each compile stretching
×1.23 at 8 and ×1.69 at 16. 73.2 of their 105.5 cold seconds have shapes fixed
before the first call: once the inputs load, or at a named boundary (the
tangential frequency rule, the pole model, the Σ plan). The 1566 real
compiles under 50 ms (45.5 s) no thread can take; only fewer programs moves
them (claim 4178).

**`compile_ahead`.** `common.jax_compile_cache.compile_ahead(program, *args)`
lowers on the calling thread in program order and reserves the agreement's
request number there. XLA then compiles on one of min(8, physical cores)
helper threads. Whichever of the helper and the live call reaches the compile
first compiles under that number, and the other waits, so a program compiles
once and takes one number on every rank, whatever the timing. Two owners
submit: the scalar tangential W-model setup, before the ζ fit (V^(1/2) of
the bare V's parents under a distributed `linalg`, and the MPA scalar head
fit when a head correction is on), and a streamed bank (all its row passes at
once). On the P4 hsuite this takes release cold from 336.7 to 326.3 s, with
requests and compiles unchanged and eqp byte-identical; the CrI3 6×6
bispinor deck moves by less than 1 s (claim 4210). The LLVM flag shortens
each large compile; threads overlap whole compiles.

## 6 Levers

| lever | state | claim |
|---|---|---|
| the three host-callback programs | closed by JAX 0.12, which stores them; they compiled in every process on 0.9 (claim 4139). Unmeasured on GPU | 4139 |
| more `compile_ahead` sites | open, none without reordering its owner: the static and direct bank streams wait on the frequency rule, which the moments build; the distrib_la stacks take the call's batch; the ζ kernels compile inside the memory check's loop | — |
| merging the largest families | not carried: they are distinct by physics (χ integrate's 11 programs are the four-current family blocks and node sets, the MPA window runner's bisp_sc programs the sectors' pole-panel widths, the tangential rounds one per sector), and merging their shapes adds masked work | 4155 |
| keep the in-memory caches between hsuite stages | rejected: warm 161.5 s against 161.9 s | 4130 |
| drop traceback locations | rejected: warm 161.8 s against 161.9 s | 4130 |
| the relaxed tangential tier | rejected: moves eqp by up to 0.64 eV | 4130 |
| `--xla_gpu_force_compilation_parallelism=16` | rejected: Si cold compile +3.0 % | 684 |
| `--xla_gpu_enable_libnvptxcompiler`, `--xla_gpu_libnvjitlink_mode` | rejected: abort at flag parsing in jaxlib 0.9.1 | 4130, 4179 |
| defer compiles behind async dispatch | rejected: hides at most 57.6 of 221.6 s (a sync follows every one or two dispatches) and needs a fork of JAX's pjit path | 4178 |

Release cold cannot reach two minutes on the hsuite while its XLA compile
alone is 199.2 s (claim 4212).

## 7 Reading a run

| line | printed | read it as |
|---|---|---|
| `<stage> compile: real R (S s), cache hits H` | rank 0, at the end of each driver and SC map (`compile_receipt`) | from SC map 2 on, R > 0 means the map recompiled a program it had |
| `disk-cache hit`, `NVRTC built` | rank 0, one per image, under `[kconv_mathdx]`, `[kconv_outer]` or `[fourier_plan]` | the k-convolution line also prints `NVRTC rebuilt (cached image refused)`; it and the Fourier pair print `store FAILED` where the cubin could not be written |
| the `Environment` startup line | always | the leg's XLA flags with their provenance, and the agreement's state |
| the cache directory; `[kconv]` | with `LORRAX_DEBUG_PRINT=1` | the `[kconv]` line names the wheel root and the cubin cache's image count, including the images linked from the store |
| every compile with its seconds | with `JAX_LOG_COMPILES=1` | — |

The hsuite's `summary.json` splits each stage's compile path
([README](../../tests/hsuite/README.md#wall-time-and-caches)).
`tests/test_glue{,2,3}_program_counts.py` count lowerings on a CPU mesh and
fail on a program built per call, per value or per context.

| message | when | what to do |
|---|---|---|
| `GATE cross_rank_compile_agreement` | ranks lowered different modules at one request number, or a rank missed a finite deadline | remove the rank-conditional shape or jit; `LORRAX_JAX_COMPILE_AGREEMENT=0` is a bisect-only opt-out |
| `UnsafeCachePolicy` | `JAX_COMPILATION_CACHE_MAX_SIZE` above 0 at $P > 1$, or below −1 | leave it unset |
| `GATE heterogeneous_gpu_targets` | the ranks hold different GPU models | one GPU model per job |
| `GATE xla_rematerialization` | a module larger than the device, on CUDA | [memory model](memory-model.md#module-does-not-fit) |
| `persistent compile cache OFF (…)`, `DISABLED: cannot arm …` | the opt-out, `MAX_SIZE=0`, or a directory that cannot be made | [registry §2e](../reference/env_vars.md#2e-compile-cache) |
| `compile-storm telemetry OFF`, `jax-compat: …` | at $P = 1$, a `jax._src` surface the counter or the fit gate wraps is absent or renamed | the [JAX contract](../installation/index.md#jax); the startup check refuses an unsupported JAX first |
| `GATE compile-agreement`, `REFUSED: jax-support.private-arity` | at $P > 1$, the compile agreement or the device-fit gate cannot install: without them a rank-divergent compile hangs the mesh | the [JAX contract](../installation/index.md#jax) |
| `distrib_la: HOST CALLBACK` | a checked solve traced outside `checked_program`, so its program keeps a host callback and waits on the host at every call | build that program with `distrib_la.checked_program` |
