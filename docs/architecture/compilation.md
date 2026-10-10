# Compilation: what is built, when, and at what cost

A LORRAX run executes three kinds of compiled code: native kernels built
with the FFI libraries, NVRTC images, and XLA programs. Each kind has its own
key. A build happens when a key is new to the cache that holds it. This page
derives each key, says where its result is kept and what makes it cold,
measures the cost, and lists what is left to cut. The variables that move a
cache are owned by [the registry](../reference/env_vars.md#2e-compile-cache).
Code is cited as `file:line`; read the file rather than the number. Every
measurement names its deck, $P$ and sandbox claim.

## Terms

| term | meaning |
|---|---|
| $P$ | ranks, one GPU each, on the square mesh |
| program | one XLA executable: one jitted function at one key (§3) |
| lowering | tracing a function to a jaxpr, then lowering the jaxpr to a StableHLO module |
| request | one call of JAX's `compile_or_get_cached` on one rank (a program lookup): a persistent-cache read, or a compile |
| real compile | one XLA backend compile (`backend_compile_and_load`) |
| image | one NVRTC cubin |
| cubin cache | the per-user, writable image directory, `ffi.fft.cubin_cache_dir()` (`src/ffi/fft.py:424`): `$SCRATCH/.cache/lorrax/kconv_mathdx`, or `~/.cache/lorrax/kconv_mathdx` where the site defines no `SCRATCH` |
| store | a release's read-only image directory, `ffi.fft.CUBIN_STORE`: `cubin_store/` at the source root. A checkout has none unless one is built into it |
| fully cold | no store, empty cubin cache, empty JAX cache |
| release cold | the release's store present, empty JAX cache: a release's first run |
| warm | both caches hold every key the run needs |
| hsuite | `tests/hsuite`, its 16 default stages at P4 on one A100-80 node, with the run directory, `TMPDIR`, `SCRATCH` and both caches node-local |

## 1 The three kinds

| kind | built by | key: what forces a new build | kept in | cost |
|---|---|---|---|---|
| native kernels | nvcc and the host compiler, at install | the bundle | the sealed FFI bundle | none on a listed architecture; else a driver PTX JIT (§2) |
| NVRTC images | NVRTC, inside the FFI, on first use | source, headers, deciding options, toolchain and device (§2). Never $N_\mu$ or $P$ | the cubin cache, seeded from the store | about 7 s each (§2) |
| XLA programs | JAX and XLA, in every process | function, abstract values, shardings, static arguments and trace context (§3) | process memory; the persistent cache, one directory per namespace and $P$ | §4 |

## 2 Native kernels and NVRTC images

**Native kernels.** The CUDA leg of the FFI is compiled once per bundle, for
the architectures [Building the FFI libraries](../installation/ffi-build.md#with-cmake-directly)
lists; a run on one of them compiles nothing here. Elsewhere the driver
JIT-compiles the PTX at first load into CUDA's own cache, `CUDA_CACHE_PATH`
([registry §5](../reference/env_vars.md#5-external-variables-lorrax-sets-or-reads)),
at a cost not measured.

**NVRTC images.** One service, `src/ffi/cpp/common/nvrtc_build.h`, builds
every kernel compiled at run time: the k-convolution family with its BSE outer
kernels, and the Fourier plan's fused pair. It compiles for the device's own
`sm_<cc>`, keeps each image in process, stores it in the cubin cache and loads
it through the driver API. The cubin cache is always on, has no knob, and
serves every world size.

| part of the key | what it is |
|---|---|
| program | FNV-1a over the embedded source, the text of each embedded header and the NVRTC options that decide the image. Editing a source or header, comments included, invalidates its images |
| toolchain | `nvrtc::mathdx_toolchain`: the cuFFTDx or cuBLASDx, commonDx, CUTLASS and CCCL version headers, the wheel's dist-info name, and the NVRTC version with the loaded `libnvrtc`'s real path. A version header that reads empty disables the disk cache for that build rather than leaving the key |
| kernel constants | the k-convolution modes and BSE outer kernels: [k-convolution §14](kconv.md#build-and-cache); the Fourier pair: the FFT extents and supports (`fourier_plan_cuda_ffi.cc:249`) |
| not keyed | file names, include paths, the host code, $N_\mu$, $P$ |

- **File.** `kconv_m<mode>[w<variant>]_<nkx>x<nky>x<nkz>_ns<ns>[x<n_r>][_c64]_sm<XY>_<key>.cubin`,
  `kconv_outer[_dec]_…` or `plan_pair_…`, framed by a `LRXKCONV1` header with
  the key and a payload hash; about 8.5 MB on sm_80, 6.1 MB of it line info
  (claim 4127). A write goes to a unique temporary and is renamed into place
  (`nvrtc_build.cc:180`), so concurrent ranks each publish a whole file. A
  torn, foreign or non-ELF file is recompiled over the entry; one the driver
  refuses to load is unlinked first (`:251`).
- **Cost.** About 7 s per image, on every rank that misses: nothing is shared
  until a file lands. The fully cold hsuite builds 25 images in 180.8 s
  (claim 4129).
- **Store.** A process's first `cubin_cache_dir()` call links every store
  image the cache lacks into it (`ffi.fft.seed_cubin_cache`, `src/ffi/fft.py:451`)
  under a temporary name, renamed into place; a present entry is kept, a
  dangling link replaced. A bad store image is rebuilt into the cache over the
  link; the store file is never written. With the hsuite's 25-image store a
  cold run builds no image (claim 4126).
- **Building a store.** `scripts/build_cubin_store.py`, one four-rank `lx run`
  on the target architecture with a fresh `SCRATCH`, builds its `GRIDS` table
  (the hsuite fixtures, the smoke decks and the production grids); running the
  smoke decks against the same cache adds the system-keyed images, and the
  release copies the images it ships into `cubin_store/`. The full table is
  392 images, 3.33 GB (claim 4127).
- **Another architecture.** The `sm<XY>` in each name makes a store
  per-architecture: an A100 store gives an H100 run nothing, so build one on
  the new architecture. A device the installed cuFFTDx cannot target fails the
  startup probe, and every factory takes the XLA backend with one warning
  (`GATE mathdx-probe`, [k-convolution §13](kconv.md#refusals)).

**How the two caches meet.** Every mathdx custom call carries two string
attributes, `cubin_dir` and `mathdx_root`, the wheel's install directory
(`_mathdx_common`, `src/ffi/fft.py:418`). Both are in the HLO, so both enter
the JAX key and the agreement fingerprint (§3) of every program holding a
k-convolution: a new `SCRATCH` or a moved venv misses every such program, and
ranks whose strings differ are refused. A store changes neither string, so it
changes no JAX key (claim 4126).

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
dtype) instead (claim 4153). `src/` opens no `with mesh:` block, whose mesh
stack would enter the trace context and re-trace programs a driver already
holds; `tests/test_glue3_program_counts.py` fails on a new one (claim 4194).

**Every process lowers.** The in-process cache dies with the process, and in
production each driver is its own process. A warm persistent cache saves the
XLA compile only: JAX needs the module to compute its persistent key. The mesh
warm-up compiles before the cache is armed and is never stored. A planner that
sizes a chunk from its compiled figure compiles candidates
(`runtime.aot_memory.check_chunk`: once, and once more over the room;
`step_up`: one per step); they are ordinary requests.

**The persistent cache.** `common.jax_compile_cache` arms JAX's own cache once
per process (`ensure_jax_compile_cache`, step 7 of
`runtime.initialize_communicator_stack`). JAX's key hashes the module, the
compile options, `XLA_FLAGS`, jaxlib and the backend. LORRAX adds the location,
`$SCRATCH/.cache/lorrax/jax_compile/<namespace>/np{P}`; the namespace names
jax, jaxlib, the FFI bundle and the key schema (`cache_namespace`), never the
LORRAX commit, which reaches a program only through its HLO. The write
threshold is 0 s. Rank 0, in a daemon thread at most every 6 h, removes
entries and other namespaces unused for 7 days, and the least recently used
namespaces earlier past 2 GiB or 200 000 files, never one used in 5 days.

**Several ranks and nodes.** Process 0 is the only writer; every rank reads,
so at $P > 1$ the directory must be on a filesystem every node sees. Each $P$
has its own `np{P}`. JAX's per-fusion XLA caches are off at $P > 1$. On a
miss every rank compiles the module itself (claim 683). JAX strips the device
assignment from the key on GPU, so every rank asks for the same key; on CPU
it does not, and the CPU hit rate at $P > 1$ is unmeasured.

| change | what goes cold |
|---|---|
| jax or jaxlib | a new namespace |
| the FFI bundle; for an unsealed library, any rebuild (its SHA-256, `_library_identity`) | a new namespace |
| `_KEY_SCHEMA` in `common/jax_compile_cache.py` | a new namespace |
| `XLA_FLAGS`, except the `--xla_dump_*` family and the few others JAX excludes | every key |
| $P$ | a different `np{P}` directory |
| system size, k grid, band counts | every key whose shapes change |
| `SCRATCH`, or the mathdx wheel's path | every k-convolution program (§2) |
| LORRAX source | only the programs whose HLO changes |

**Never cached.** JAX never stores a program with a host callback, so it
compiles again in every process; the counter prints `UNCACHEABLE <module>` the
first time. A checked dense solve stays callback-free inside
[`distrib_la.checked_program`](../services/distrib_la/api.md#dense-factorizations).
The hsuite has three such programs (claim 4139).

**The cross-rank compile agreement.** At $P > 1$ with a coordination client,
each request fingerprints its module (SHA-256 of the MLIR without debug
locations, process-local `…handle` literals canonicalised) and publishes it
under the rank's request number (`_requested`). Numbers are taken in program
order on the calling thread, under one lock that covers only the numbering and
the publish, so they agree across ranks that request the same programs in the
same order. Only a real compile checks (`_check_compile_record`): rank 0
reads every rank's record, a peer reads rank 0's, and a mismatch refuses
before the compile, naming every rank's key, instead of hanging in the next
collective. A cache hit checks and waits for nothing. At the default timeout,
0, a check waits without bound and rank 0 names a missing rank every 60 s.
The agreement costs 6.2 s on the warm hsuite (161.9 against 155.7 s, claim
4130). It stays on.

**The runtime's XLA flags.** On GPU, `runtime.set_default_xla_gpu_autotune`
(`src/runtime/__init__.py:497`) appends two flags to `XLA_FLAGS`; a caller's
value wins, flag by flag. A CPU run gets neither.

| flag | effect | claim |
|---|---|---|
| `--xla_gpu_autotune_level=0` | cold compile −12.9 % (Na) and −16.3 % (Si) at P4, execution within noise | 683 |
| `--xla_gpu_enable_llvm_module_compilation_parallelism=true` | XLA splits each module's LLVM IR into parts compiled in parallel; hsuite release-cold XLA compile 219.5 → 195.2 s, programs of 0.5 s or more −22 %, programs under 50 ms +10 %; results bitwise, device peak unchanged | 4179 |

`runtime.disable_xla_rematerialization` (`:561`) adds `rematerialization` to
`--xla_disable_hlo_passes` on GPU and CPU. On CUDA, a module that then does not
fit the device refuses before it runs
([memory model](memory-model.md#module-does-not-fit)); on CPU nothing checks.

## 4 Where the time goes

**The hsuite.** One A100-80 node, P4, node-local:

| state | wall | compile | claim |
|---|---|---|---|
| release cold | 326.3 s | 2261 real XLA compiles | 4210 |
| warm | 139.9 s | 3037 requests per rank | 4210 |
| fully cold | release cold plus the NVRTC builds | 25 NVRTC images, 180.8 s | 4129 |

**A warm run still pays the compile path.** The census run spent 76.3 s of
its warm wall on it: trace 18.7 s, lower 34.3 s, cache key and read 23.5 s
(claim 4129). Fewer lowerings, not a different cache, move it. The census
(claim 4155) finds three classes: a later driver re-lowering an earlier one's
program (957; the hsuite clears JAX's in-process caches between stages, as
production's one process per driver does); a stage re-lowering its own key
(per-call jits, eager glue, nested mesh contexts, mostly removed: claims 4152,
4154, 4194); and first lowerings (2576 keys from 1136 function definitions).
The tail is long: the top 10 of 2655 call sites hold 29 % of the compile-path
seconds and the top 100 hold 65 % (claim 4129).

**Production decks.** CrI3 6×6 and Fe 4³ bispinor SC, forced face route, P4,
one A100-80 node, cold JAX cache, maps 0–2, without and with the LLVM flag
(claim 4179):

| deck | XLA compile, map 0 | later map | rank-0 wall |
|---|---|---|---|
| CrI3 6×6 | 156.6 → 134.3 s | map 1: 52.7 → 43.1 s | 590.2 → 573.9 s |
| Fe 4³ | 155.0 → 133.6 s | map 2: 53.0 → 42.4 s | 473.9 → 430.3 s |

No leg builds an NVRTC image. Device peaks are unchanged (44.76 and
21.70 GB) and eqp is identical in 6 of 6 files.

## 5 What threads can and cannot do

A jitted function traces, lowers and compiles on the thread that makes its
first call, before it runs. Claim 4178 measures the hsuite's 75 largest
programs, recompiled with all four ranks at once and 16 cores per rank.

- **XLA's backend compile releases the GIL.** The 75 programs take 104.0 s
  serial, 55.8 s on 2 threads, 29.1 s on 4, 16.4 s on 8 and 12.4 s on 16; each
  program's own compile stretches ×1.23 at 8 threads and ×1.69 at 16.
- **Most large programs are knowable ahead.** 73.2 of their 105.5 cold seconds
  have shapes fixed before the first call: once the inputs load, or at a named
  boundary (the shared-pole frequency rule, the pole model, the Σ plan).
- **The small ones are the floor.** 1566 real compiles under 50 ms hold
  45.5 s. Each compiles at its first call, so no thread can take it; only
  fewer programs, or a cheaper pipeline per module, moves them.

**`compile_ahead` takes the large programs whose shapes are known.**
`common.jax_compile_cache.compile_ahead(program, *args)` lowers on the calling
thread in program order and reserves the agreement's request number there.
XLA then compiles on one of min(8, physical cores) helper threads. Whichever of
the helper and the live call reaches the compile first compiles under that
number, and the other waits. A program therefore compiles once and takes one
number on every rank, whatever the timing. Two owners submit:

- the W-model setup, before the ζ fit: V^(1/2) of the bare V's parents
  (scalar shared pole, distributed `linalg`) and the MPA scalar head fit
  (from `n_poles`);
- a streamed bank: all its row passes at once.

On the P4 hsuite this takes release cold from 336.7 to 326.3 s, with requests
and compiles unchanged and eqp byte-identical. The CrI3 6×6 bispinor deck
moves by less than 1 s (claim 4210).

The LLVM flag (claim 4179) shortens each large compile from inside; threads
overlap whole compiles. The two are complements.

## 6 What remains

| lever | state | expected | claim |
|---|---|---|---|
| the three uncacheable programs | compile in every process | removing their host callbacks | 4139 |
| the largest families | distinct by physics: χ integrate's 11 programs (the four-current family blocks and node sets), the MPA window runner's bisp_sc programs (the sectors' pole-panel widths), the shared-pole rounds (one per sector). Merging their shapes adds masked work | none | 4155 |
| more `compile_ahead` sites | the static and direct bank streams wait on the frequency rule, which the moments build; the distrib_la stacks take the call's batch; the ζ kernels compile inside the memory check's loop | none without reordering their owners | 4210 |

Release cold cannot reach two minutes on the hsuite while its XLA compile
alone is 195.2 s (claim 4179).

**Measured and rejected:**

| lever | result | claim |
|---|---|---|
| keep the in-memory caches between hsuite stages | warm 161.5 s against 161.9 s | 4130 |
| drop traceback locations | warm 161.8 s against 161.9 s | 4130 |
| the relaxed shared-pole tier | moves eqp by up to 0.64 eV | 4130 |
| `--xla_gpu_force_compilation_parallelism=16` | Si cold compile +3.0 % | 684 |
| `--xla_gpu_enable_libnvptxcompiler`, `--xla_gpu_libnvjitlink_mode` | abort at flag parsing in jaxlib 0.9.1 | 4130, 4179 |
| defer compiles behind async dispatch | hides at most 57.6 of 221.6 s (a sync follows every one or two dispatches) and needs a fork of JAX's pjit path | 4178 |

## 7 Reading a run

- `<stage> compile: real R (S s), cache hits H, uncacheable U [names]` on rank 0
  at the end of each driver and SC map (`compile_receipt`). From SC map 2 on,
  R > 0 means the map recompiled a program it had; U > 0 compiles in every process.
- Every image prints `disk-cache hit`, `NVRTC built` or `NVRTC rebuilt (cached
  image refused)` on rank 0, under `[kconv_mathdx]`, `[kconv_outer]` or
  `[fourier_plan]`, with `store FAILED` where the cubin could not be written.
- The always-on `Environment |` startup line names both XLA flags with their
  provenance and the agreement's state. `LORRAX_DEBUG_PRINT=1` adds the cache
  directory and a `[kconv]` line with the wheel root and the cubin cache's
  image count, including the images linked from the store.
- `JAX_LOG_COMPILES=1` logs every compile with its seconds. The hsuite's
  `summary.json` splits each stage's wall into trace, lower, cache reads, XLA
  compiles, the agreement and NVRTC builds
  ([hsuite README](../../tests/hsuite/README.md#wall-time-and-caches)).
- `tests/test_glue{,2,3}_program_counts.py` count lowerings on a CPU mesh and
  fail on a program built per call, per value or per context.

| message | when | what to do |
|---|---|---|
| `GATE cross_rank_compile_agreement` | ranks lowered different modules at one request number, or a rank missed a finite deadline | remove the rank-conditional shape or jit; `LORRAX_JAX_COMPILE_AGREEMENT=0` is a bisect-only opt-out |
| `UnsafeCachePolicy` | `JAX_COMPILATION_CACHE_MAX_SIZE` above 0 at $P > 1$ | leave it unset |
| `GATE heterogeneous_gpu_targets` | the ranks hold different GPU models | one GPU model per job |
| `GATE xla_rematerialization` | a module larger than the device, on CUDA | [memory model](memory-model.md#module-does-not-fit) |
| `persistent compile cache OFF (…)`, `DISABLED: cannot arm …` | the opt-out, `MAX_SIZE=0`, or a directory that cannot be made | [registry §2e](../reference/env_vars.md#2e-compile-cache) |
| `compile-storm telemetry OFF`, `jax-compat: …` | a `jax._src` surface the counter or the fit gate wraps is absent | the [JAX contract](../installation/index.md#jax); the startup check refuses an unsupported JAX first |
| `distrib_la: UNCACHEABLE` | a checked solve traced outside `checked_program`, so its program keeps a host callback | build that program with `distrib_la.checked_program` |
