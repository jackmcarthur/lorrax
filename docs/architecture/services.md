# Substrate services

*Which capabilities are services, which of them expose a backend choice to
the caller, and the boundary a caller must not cross. Each service's caller
contract lives on its own page, linked from the inventory. Levels and import
rules are on [the three levels](layers.md); the vendor library behind each
routine, and the gate that proves it built, is on [the FFI layer](ffi_layout.md).*

## What a service is {#what-is-a-service}

A capability is a service when it has all four of:

1. **An interface a caller can use without knowing the backend.** The caller
   states *what* in its own vocabulary (a path, a mesh, a spec, an operand),
   never *how*.
2. **A stated guarantee strong enough to design against.** SlabIO: nothing
   larger than one rank's tile is materialised. `distrib_la`: a resolved
   backend name is a promise, so the call cannot fail for an availability
   reason.
3. **Something that varies underneath, per machine**: a vendor library, a
   transport, a driver generation. Where nothing varies, the module is a
   helper, not a service.
4. **A gate proving it built right**, or a stated admission that there is
   none.

## The service inventory {#inventory}

### Backend services

| service | the call a caller makes | varies underneath | caller picks the backend? | contract |
|---|---|---|---|---|
| **`file_io.slab_io`** | `SlabIO(path, *, mode, mesh)` → `create_dataset` / `write_slab` / `read_slab` / `read_slabs` | the phdf5 handler, its MPI and HDF5, Lustre striping, collective buffering | **no**, by design ([below](#choice)) | [SlabIO](slab_io.md) |
| **`ffi.io`** | `open_file(path, *, mesh, mode)` → `write_sharded_slab` / `read_sharded_slab` / `read_kchunk_union_sharded` | CUDA or host library, from the mesh's devices | no | [FFI layer §5](ffi_layout.md); SlabIO is its only transport consumer |
| **`ffi.fft`** (entered through `common.fft_helpers`) | the k-convolution router's doors (`make_fused_conv_kpair`, `make_kconv_klead`, `make_kfft_klead`, …) and `make_flat_k_fft` | nvidia-mathdx on CUDA; FFTW3-ABI plans on cpu | no: the mesh platform decides, and no variable or deck key selects a route | [k-convolution router](ffi_layout.md#k-convolution-router-and-the-mathdx-family) |
| **`ffi.gemm`** | `gemm_batch(a3, b3)` inside the caller's own `shard_map` | the CBLAS provider, and whether it has a batched entry | no: `LORRAX_BANDS_GEMM_FFI` is on/off, cpu only | `src/ffi/gemm.py` |
| **`distrib_la`** | `plan(op, mesh_xy, *, backend=…)` → `plan(A_tile)` / `plan.batched(A_stack)`, plus `matmul`, `gemm_plan` | ScaLAPACK, SLATE, cuSOLVERMp, cuBLASMp or native, per op, machine and mesh geometry | **yes**, by design, through deck keys ([below](#choice)) | [`distrib_la`](../services/distrib_la.md) |
| **`wfn_loader`** | `WfnLoader(path, *, mesh=None, backend='auto')` | `eager` (h5py) or `phdf5` (one collective read through `SlabIO.read_slabs`) | escape hatch only: `LORRAX_WFN_BACKEND` | [`wfn_loader`](../services/wfn_loader.md) |
| **`ffi.common.ffi_loader`** over `lxkit.native_provider` | `get_lib(platform)`, `probe_target(target, platform)` | which library pair: a sealed bundle or a build tree | no: `LORRAX_FFI_SO` / `LORRAX_FFI_HOST_SO` pin a path, and a pin that is not a file refuses | [FFI layer §2c–§2d](ffi_layout.md) |
| **`common.collectives`** | `prepare_mesh()`, `gather_k_blocks()` | NCCL on CUDA; MPI on CPU | no: the CPU transport is a deployment fact | [transports](../environment/transports.md) |
| **`runtime`** | `initialize_communicator_stack()` | allocator, plugin discovery, CPU or GPU backend, network transport | no | [below](#runtime) |
| **`common.jax_compile_cache`** | called once by `runtime` | filesystem, world size, jaxlib generation | no | `tests/test_compile_cache_agreement.py` |

### Single-owner services

These packages under `services/` hold one source of truth two or more drivers
need. Nothing varies underneath them per machine, so they carry no backend
choice; they are services for property 1 and for the door rule below.

| package | owns | contract |
|---|---|---|
| `symmetry_maps` | IBZ↔full-BZ tables, star maps, unfolds, the 2c TRS check | [`symmetry_maps`](../services/symmetry_maps.md) |
| `minimax` | certified quadrature tables and their refusals | [`minimax`](../services/minimax.md) |
| `vcoul` | bare and truncated Coulomb kernels | [`vcoul`](../services/vcoul.md) |
| `zeta_loader` | the `zeta_q.h5` format and its readers; collective reads go through SlabIO | [`zeta_loader`](../services/zeta_loader.md) |
| `lxkit` | native-provider attestation, the gate and probe vocabulary, launcher and placement policy | `services/lxkit/` |

### The runtime entry {#runtime}

`runtime.initialize_communicator_stack()` is called once, at the top of a
driver, above the driver's own `import jax`. In order it installs the
fail-fast excepthook, seals the source closure, sets the environment before
jax reads it (including the one GPU pool policy), selects the CPU-collectives
transport, runs `jax.distributed` (auto-detected when `SLURM_NTASKS > 1`),
picks GPU or CPU, enforces the JAX generation, builds the run's mesh with
every communicator created, enforces the required FFI backends, enables the
compile cache and prints the rank-0 startup report. It returns a `RuntimeStack` whose `mesh` is the run's
mesh; never build a second one. `runtime.finalize_process(rc)` is the
sanctioned driver exit. The startup report and its debug form are described
in the [environment overview](../environment/overview.md#startup-block);
knob spellings are in [`env_vars.md`](../dev/env_vars.md). A dial missing from
the report fails `tests/test_runtime_startup_report.py`.

### Backends are not services {#ffilinalg}

`distrib_la._scalapack`, `._slate` and `._cusolvermp` are backends of the
`distrib_la` door. Call sites reach them through `distrib_la.backend_module()`
or a plan; a `src/` import of one fails `tests/test_layering.py`.
`ffi.cusolvermp` is a re-export shim over `distrib_la._cusolvermp` for the
bench drivers, and `ffi.cublasmp` is reached only from the bench drivers.

`ffi.gate` is the mechanism, not a service: the resolver behind the two
environment dials, `LORRAX_FFT_FFI` (cpu flat-k FFT) and
`LORRAX_BANDS_GEMM_FFI`. `distrib_la` does not use it, because its choice
comes from the deck and it resolves once from arguments.

Not services, although callers find them beside the services:
`common.contract_bands`, `common.staged_reshard` and `common.sharding_fit`
(movement patterns with no vendor library; `contract_bands` consumes
`ffi.gemm`), `common.timing` (instrumentation), `runtime.xla_memory` (a
read-only mirror of jaxlib's environment parse).

## Which services expose a choice, and which hide one {#choice}

**SlabIO hides the choice.** A caller states a path, a mode, a mesh and
logical shapes, and gets tiles. There is no `backend=` argument and no deck
key, because every alternative moved the same bytes to the same place at
equal or worse cost: a rank-0 gather is an out-of-memory at the design size,
not a slow tier ([`decisions.md`](decisions.md), 2026-08-05). A deployment that
cannot serve the tile path refuses at open.

**`distrib_la` exposes the choice.** The deck keys `eigh_backend`,
`distributed_cholesky` and `distributed_lu` choose the backend per operation;
the package reads no environment. The environment grants capability (which
`.so`); the deck makes the choice. The backends have different costs for
different shapes: the native q-batched path solves many small matrices at
once, the distributed backends spread one large matrix over the mesh, and
only the caller knows which it holds. An explicit request never demotes; only
`auto` demotes, with a rank-0 announcement naming geometry or capability as
the cause. The vocabularies and the resolution policy are on the
[`distrib_la`](../services/distrib_la.md) page.

**`wfn_loader` exposes an escape hatch.** `LORRAX_WFN_BACKEND` (`eager` or
`phdf5`) is defensible only because a parity test holds the two backends
byte-identical: the choice cannot change an answer.

**The FFT and GEMM paths expose no backend choice.** The k-convolution router
picks its backend from the mesh platform alone. `LORRAX_FFT_FFI` and
`LORRAX_BANDS_GEMM_FFI` are capability switches: the FFI layer is required
([`decisions.md`](decisions.md), 2026-08-01), so `LORRAX_FFT_FFI=0` refuses,
and `LORRAX_BANDS_GEMM_FFI=0` selects an announced debug fallback.

**The rule.** Expose a choice only where the alternatives have different
costs that the call site, not the deployment, can judge, and then either hold
the alternatives to a parity contract or document the cost difference.
Deployment facts (which MPI, which HDF5, which FFTW3 `.so`) belong to the
environment and to [the FFI layer](ffi_layout.md); call-site facts (many small
tiles or one large one) belong in the deck.

## The boundary {#boundary}

[The three levels](layers.md) assigns each service its level (`lxkit` and
`distrib_la` are L3; the single-owner packages take the L1 default), and
imports run downhill only. Three rules hold at every call site:

* **The package is the door.** A `src/` module imports a service's top-level
  package, never a submodule (`tests/test_layering.py` rule 6).
* **Drivers never name a backend.** The backend is chosen inside a service
  door, from a deck key or the mesh platform.
* **Announce or refuse.** A demotion is announced from the rank it happened
  on; an explicit request that cannot be honoured raises with the reason and
  the fix. Resolve-time checks (platform, geometry, handler probe, grammar)
  fire when a factory or plan is built; trace-time checks (dtype, rank,
  extents) fire inside the returned callable.

## Adding a service

One module owns the pattern. Use a typed `Gate` or explicit parameters, never
loose environment reads. Refuse at resolve time where the fact is known
there, and at trace time where it is not. Make every announcement greppable
and rank-disciplined. A factory-time dial goes into `ffi.FFI_DIAL_ENV` and
`ffi.ffi_dial_key()`, so it enters kernel cache keys and the compile cache's
cross-rank fingerprint (`tests/cache_key_lint.py`, rule `env-dial`).
`tests/test_layering.py` and `tests/test_runtime_startup_report.py` enforce
the level and the announcement.

Decide explicitly whether the caller picks the backend, and record why.
Default to *no*: a dial whose settings are "correct" and "worse in every
measured respect" is not a choice, and each one costs a router, a vocabulary,
a deck key and a refusal per door. Add the row to the
[inventory](#inventory), the operation and its gate to the
[kernel operations](ffi_layout.md#kernel-operations) table, and each
machine's library to [the FFI layer §3a](ffi_layout.md#3a-the-dependency-matrix).
