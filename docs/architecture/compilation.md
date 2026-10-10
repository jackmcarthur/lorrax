# Compilation: what is built, when, and at what cost

A LORRAX run executes three kinds of compiled code: native kernels built
with the FFI libraries, NVRTC images of the k-convolution kernels, and XLA
programs. Each kind has its own key. A build happens when a key is new
to the cache that holds it. This page derives each key, says where its
result is kept and when it is built, measures the cost, and lists what is
left to cut. The NVRTC build and the cubin store are owned by
[k-convolution §14](kconv.md#build-and-cache); the variables that move a
cache are owned by [the registry](../reference/env_vars.md#2e-compile-cache).
Code is cited as `file:line` at `b3147d82c`; read the file rather than the
number. Every measurement names its deck, $P$ and sandbox claim.

## Terms

| term | meaning |
|---|---|
| $P$ | ranks, one GPU each, on the square mesh |
| program | one XLA executable: one jitted function at one key (§3) |
| lowering | tracing a function to a jaxpr, then lowering the jaxpr to a StableHLO module |
| request | one call of JAX's `compile_or_get_cached` on one rank: a persistent-cache read, or a compile |
| real compile | one XLA backend compile (`backend_compile_and_load`) |
| image | one NVRTC cubin |
| fully cold | no cubin store, empty cubin cache, empty JAX cache |
| release cold | the release's cubin store present, empty JAX cache: a release's first run |
| warm | both caches hold every key the run needs |
| hsuite | `tests/hsuite`, its 16 default stages at P4 on one A100-80 node, with the run directory, `TMPDIR`, `SCRATCH` and both caches node-local |

## 1 The three kinds

| kind | built by | key: what forces a new build | kept in | cost |
|---|---|---|---|---|
| native kernels | nvcc and the host compiler, at install | the bundle | the sealed FFI bundle | none at run time |
| NVRTC images | NVRTC, inside the FFI, on first use | mode, k grid, $n_s$, variant and device; the FFT plane for mode 10 and the Fourier pair; the band rank for the BSE outer kernels. Never $N_\mu$ or $P$ | the per-user cubin cache, seeded from the release's store | about 7 s each (§4) |
| XLA programs | JAX and XLA, in every process | function, abstract values, shardings, static arguments and trace context (§3) | process memory; the persistent cache, one directory per namespace and $P$ | §4 |

## 2 Native kernels and NVRTC images

**Native kernels.** The CUDA leg of the FFI is compiled once per bundle
([Building the FFI libraries](../installation/ffi-build.md)). It carries SASS
for sm_80, 86, 89, 90, 100 and 120 and PTX for compute_80 and compute_120
(`CMAKE_CUDA_ARCHITECTURES`, `src/ffi/cpp/CMakeLists.txt:89`). The driver
JIT-compiles the PTX only on a device with no matching SASS. A run on a
listed architecture compiles nothing here.

**NVRTC images.** The k-convolution kernels (modes 0–11), the BSE outer-product
kernels and the Fourier plan's fused pair are compiled at run time for the
device's own `sm_<cc>`. The key is set by the kernel's compile-time
constants. For modes 0–9 and 11 these are the k grid, $n_s$, the variant, the
`live` bit and the device. Mode 10 and the Fourier pair take the FFT plane. The BSE outer
kernels take $K = \min(n_c, n_v)$. An image is built once per user: the
per-user cache (`ffi.fft.cubin_cache_dir`, `src/ffi/fft.py:422`) keeps it, and
a release's store (`ffi.fft.CUBIN_STORE`, `:444`) is linked into that cache by
`ffi.fft.seed_cubin_cache` (`:448`) once per process. Key, file format and
store build: [k-convolution §14](kconv.md#build-and-cache).

**How the two caches meet.** Every mathdx custom call carries the cache path
as its string attribute `cubin_dir` (`_mathdx_common`, `src/ffi/fft.py:416`).
The path is part of the HLO, so it is part of the JAX key of every program
that holds a k-convolution. A warm JAX cache is warm only for a process that
resolves the same cubin path, which on Perlmutter means the same `SCRATCH`.
The store leaves the path unchanged, so adding or removing a store changes
no JAX key (claim 4126).

## 3 XLA programs

**One call.** A jitted function is compiled at the granularity of its
in-process key: the jit object, the abstract value (shape, dtype) and
sharding of every argument, the static arguments, and the trace context,
which includes the mesh stack of `with mesh:` blocks. The first call at a new
key takes this path; later calls in the same process reuse the executable.

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

Three consequences follow from the key. A jit built inside a function is a
new object on every call, so it lowers on every call; `jnp.zeros(..., device=sharding)`
lowers once per (shard shape, dtype) instead (claim 4153). A nested `with mesh:`
inside a driver's own puts a second mesh on the trace context and re-traces
programs the driver already holds (claim 4169). A new system size changes the
shapes and so misses every entry of a warm cache.

**Every process lowers.** The in-process cache dies with the process, and in
production each driver is its own process. A warm persistent cache saves the
XLA compile only: every program is still traced and lowered, because JAX needs
the module to compute its persistent key.

**The persistent cache.** `common.jax_compile_cache` arms JAX's own cache once
per process (`ensure_jax_compile_cache`, `src/common/jax_compile_cache.py:1079`,
called at step 7 of `runtime.initialize_communicator_stack`). JAX's key hashes
the module, the compile options, `XLA_FLAGS`, jaxlib and the backend; on GPU it
is the same on every rank. LORRAX adds the location,
`$SCRATCH/.cache/lorrax/jax_compile/<namespace>/np{P}`, where the namespace
names jax, jaxlib, the FFI bundle and the key schema (`cache_namespace`, `:933`).
The LORRAX commit is not in it: source reaches a program only through its HLO,
which the key already hashes. A new `XLA_FLAGS` value changes every key, and a
new FFI bundle opens a new namespace, so either makes the next run release
cold. Process 0 writes, every rank reads, and the write threshold is 0 s. On a
miss every rank
compiles the module itself; a cold compile is not shared across ranks (claim
683). Location, pruning and opt-out: [registry §2e](../reference/env_vars.md#2e-compile-cache).

**Never cached.** JAX never stores a program that carries a host callback, so
such a program compiles again in every process. The compile counter prints
`UNCACHEABLE <module>` the first time it meets one (`_uncacheable`, `:764`). A
checked dense solve stays callback-free inside
[`distrib_la.checked_program`](../services/distrib_la/api.md#dense-factorizations).
The hsuite has three such programs per run (claim 4139).

**The cross-rank compile agreement.** At $P > 1$, each request fingerprints
the location-free module (SHA-256) and publishes it under the rank's request
number (`_requested`, `:748`). Each real compile first reads every rank's
record for that number (`_counting`, `:781`). A rank that lowered a different
module refuses before the compile, naming every rank's key, instead of hanging
in its next collective. Requests run under one process lock (`:754`), so the
numbers agree across ranks that request the same programs in the same order.
That lock also makes a process compile one program at a time (§5). On the warm
hsuite the agreement costs 6.2 s: 161.9 s with it, 155.7 s without (claim
4130). It stays on.

**The runtime's XLA flags.** On GPU, `runtime.set_default_xla_gpu_autotune`
(`src/runtime/__init__.py:494`) appends two flags to `XLA_FLAGS`; a caller's
value wins, flag by flag.

| flag | effect | claim |
|---|---|---|
| `--xla_gpu_autotune_level=0` | cold compile −12.9 % (Na) and −16.3 % (Si) at P4, execution within noise | 683 |
| `--xla_gpu_enable_llvm_module_compilation_parallelism=true` | XLA splits each module's LLVM IR into parts compiled in parallel; hsuite release-cold XLA compile 219.5 → 195.2 s, programs of 0.5 s or more −22 %, programs under 50 ms +10 %; results bitwise, device peak unchanged | 4179 |

`runtime.disable_xla_rematerialization` (`:558`) also adds `rematerialization`
to `--xla_disable_hlo_passes` on GPU and CPU; a module that then does not fit
the device refuses before it runs ([memory model](memory-model.md#module-does-not-fit)).
`--xla_gpu_force_compilation_parallelism=16` is a different flag: it forces a
thread count and slowed Si's cold compile by 3.0 % (claim 684).

## 4 Where the time goes

**The hsuite.** One A100-80 node, P4, node-local:

| state | tree | wall | compile | claim |
|---|---|---|---|---|
| fully cold | b6e679d3b | 583.9 s | 25 NVRTC images (180.8 s), then every XLA compile | 4129 |
| release cold | fe3ac4cad | 366.8 s | 2295 real compiles, 219.5 s | 4160, 4179 |
| release cold, LLVM flag | fe3ac4cad + `compile_ahead` + flag | 343.8 s | 195.2 s of XLA compile | 4179 |
| warm | fe3ac4cad | 144.5 s | 3207 requests per rank, 11 real compiles; trace and lower 49.5 s | 4160 |

Main `4799c5d7e` takes the requests to 3167 per rank, with walls 365.7 s cold
and 144.0 s warm (claim 4168). Main `b3147d82c` adds the flag as a default; no
hsuite pair has been measured on it. Per driver at fe3ac4cad, cold and warm:
bisp_sc 130.1 s and 52.6 s; na_dipole 14.5 s and 6.4 s; gnppm 28.7 s cold
(claim 4160).

**The warm wall is mostly the compile path.** On main c0f20111d the warm hsuite
made 4080 lowerings per rank, costing 70.1 s: trace 18.3 s, lower 36.3 s,
cache key and read 15.5 s (claim 4155). Fewer lowerings, not a different cache,
is what moves the warm wall.

**Where the requests come from** (claim 4155, the same census):
- 957 re-lower a program an earlier driver already lowered. The hsuite clears
  JAX's in-process caches between stages, which stands in for production's one
  process per driver, so production pays these too.
- 497 re-lower a key already lowered in the same stage. Most of them were
  per-call jits and eager glue that main now builds once per shape (claims
  4152, 4154).
- The rest are first lowerings: 2576 distinct persistent keys from 1136
  function definitions.

The tail is long. The requests come from 2655 call sites; the top 10 sites
hold 29 % of the compile-path seconds and the top 100 hold 65 % (claim 4129).

**Production decks.** CrI3 6×6 bispinor SC and Fe 4³ bispinor SC, forced face
route, P4, one A100-80 node, cold JAX cache, maps 0–2 (claim 4179). Without
and with the LLVM flag:

| deck | XLA compile, map 0 | later map | rank-0 wall |
|---|---|---|---|
| CrI3 6×6 | 156.6 → 134.3 s | map 1: 52.7 → 43.1 s | 590.2 → 573.9 s |
| Fe 4³ | 155.0 → 133.6 s | map 2: 53.0 → 42.4 s | 473.9 → 430.3 s |

The walls are clean pairs, with no NVRTC build in either leg. Device peaks
are unchanged (44.76 and 21.70 GB) and eqp is identical in 6 of 6 files.

## 5 What threads can and cannot do

Claim 4178 measures the hsuite's 75 largest programs, recompiled with all four
ranks at once and 16 cores per rank.

- **XLA's backend compile releases the GIL.** The 75 programs take 104.0 s
  serial, 55.8 s on 2 threads, 29.1 s on 4, 16.4 s on 8 and 12.4 s on 16. Each
  program's own compile stretches ×1.23 at 8 threads and ×1.69 at 16.
- **One lock serializes them in each process** (§3, the agreement).
- **Most large programs are knowable ahead.** 73.2 of the 105.5 cold seconds
  of the top 75 have their shapes fixed before the first call: once the inputs
  load, or at a named boundary (the shared-pole frequency rule, the pole
  model, the Σ plan).
- **The small ones are the floor.** 1566 real compiles under 50 ms hold
  45.5 s. Each is compiled at its first call, so no thread can take it. Only
  fewer programs, or a cheaper pipeline per module, moves them.
- **Deferring compiles behind async dispatch** would hide at most 57.6 of
  221.6 s, because a sync follows every one or two dispatches (median over
  1714 host syncs). It would also need a fork of JAX's pjit path. It is not
  carried.

The LLVM flag and threads are complements: the flag shortens each large
compile from inside, and threads overlap whole compiles.

## 6 What remains

| lever | state | expected | claim |
|---|---|---|---|
| `compile_ahead`: lower on the calling thread in program order, take the agreement slot there, compile on a pool of min(8, physical cores) threads | branch `perf/parcomp-20261010` `0478adf95`, owner decision. Bitwise; one site (the bank's $V^{1/2}$): 368.2 / 146.0 s against 370.8 / 146.1 s, cold / warm | seven more sites, about −25 to −35 s cold, 120–150 lines | 4180 |
| nested `with mesh:` blocks | 138 of the 729 same-stage re-lowerings differ only in the mesh stack, from 30 sites; they re-trace 1412 calls | one `with mesh` per driver removes them | 4169 |
| the three uncacheable programs | compile in every process | removing their host callbacks | 4139 |
| the largest families | distinct by physics: χ integrate's 11 programs (the four-current family blocks and node sets), the MPA window runner's 4 bisp_sc programs (the sectors' pole-panel widths), the shared-pole rounds (one per sector). Merging their shapes adds masked work | none | 4155 |

Release cold cannot reach two minutes on the hsuite while its XLA compile
alone is 195.2 s (claim 4179).

**Measured and rejected** (warm hsuite, 161.9 s baseline, claim 4130): keeping
the in-memory caches between stages (161.5 s) and dropping traceback locations
(161.8 s) move nothing past noise. The relaxed shared-pole tier moves eqp by up
to 0.64 eV. `--xla_gpu_enable_libnvptxcompiler` and
`--xla_gpu_libnvjitlink_mode` abort at flag parsing in jaxlib 0.9.1 (claims
4130, 4179).

## 7 Reading a run

- `<stage> compile: real R (S s), cache hits H, uncacheable U [names]` on rank 0
  at the end of each driver and of each SC map (`compile_receipt`,
  `src/common/jax_compile_cache.py:800`). From SC map 2 on, R > 0 means the map
  recompiled a program it already had; U > 0 compiles again in every process.
- `[kconv_mathdx] disk-cache hit` or `NVRTC built …` on rank 0 for every image
  ([k-convolution §14](kconv.md#build-and-cache)).
- Under `LORRAX_DEBUG_PRINT=1`, the startup report names the cache directory,
  the agreement state and both XLA flags with their provenance.
- `JAX_LOG_COMPILES=1` logs every compile with its seconds.
- The hsuite's `summary.json` splits each stage's wall into trace, lower,
  cache reads, XLA compiles, the agreement and NVRTC builds
  ([hsuite README](../../tests/hsuite/README.md#wall-time-and-caches)).
