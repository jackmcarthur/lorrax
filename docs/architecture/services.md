# Substrate services

*The L3 substrate ([layers](layers.md)) as a set of internal **services**.
Part 1 says what a service is here, which capabilities are services, and
which of them deliberately expose a choice to the caller. Part 2 is the
API reference: per service, its purpose, its **public API as it exists in
the code today**, its contract (what it refuses, what it announces, and
**when**), its level, and its dependencies.*

*Signatures are verified against `src/` on `integration/2026-08-06`; where
a service is changing on an unmerged branch, the section says so and names
the branch. Signatures here win over any older doc prose.*

## What a service is, here {#what-is-a-service}

The name is not decoration. A capability on this page is a service when it
has all four of:

1. **An interface a caller can use without knowing the backend.** The
   caller states *what*, in its own vocabulary — a path, a mesh, a spec, an
   operand — and never *how*.
2. **A stated guarantee**, strong enough to design against. SlabIO's is
   "nothing larger than one rank's tile is ever materialised".
   `ffi.linalg`'s is "a returned backend name is a promise: its handler is
   compiled in and every mesh guard passed, so the call cannot fail for an
   availability reason".
3. **Something that genuinely varies underneath, per machine** — a vendor
   library, a transport, a driver generation. Where nothing varies, a
   module is a helper, not a service, and does not need this page.
4. **A gate proving it built right**, or a named admission that there is
   none.

The owner's statement of purpose for (1), which is the test to apply when
designing a new one: *agents wiring in parallel I/O should not have to
think about the nitty-gritty so much.*

**Property 3 and 4 are recorded elsewhere and are not repeated here.**
[`ffi_layout.md` §3](ffi_layout.md) is the routine × machine ×
alternatives × gate matrix: for each numerical or I/O routine LORRAX does
not implement itself, who serves it on Perlmutter and on Frontera, what
else could, how you would know it built right, and whether that check
passes. **This page is the other half of the same material — what a
*caller* sees. Read them together; do not restate one in the other.**

## The service inventory {#inventory}

| service | the call a caller actually makes | guarantee | varies underneath | caller picks the backend? | gate |
|---|---|---|---|---|---|
| **`file_io.slab_io`** | `SlabIO(path, mode=, mesh=)` → `create_dataset` / `write_slab` / `read_slab` | no object larger than one rank's tile is materialised | Lustre striping, ROMIO collective buffering, the phdf5 handler, the MPI it was built against | **no — by design** (see [below](#choice)) | GATE 1, GATE 7 (`ffi_layout` §3); `tests/test_slab_io_routing.py` |
| **`ffi.linalg`** | `plan(op, mesh_xy, backend=…)` → `plan(A_tile)` / `plan.batched(A_stack)` | a resolved backend name is a *promise*: handler compiled in and every geometry guard passed, so the call cannot fail for an availability reason | ScaLAPACK / SLATE / cuSOLVERMp / native, per op, machine and mesh geometry | **yes — by design**, via three deck keys (see [below](#choice)) | **none at build time**; `tests/test_ffi_linalg_contract.py` at run time |
| **`ffi.fft`** (entered through `common.fft_helpers`) | `make_flat_k_fft(mesh, kgrid, spec, kind=…)` | one flat-k batched 3-D FFT under the same jax.ffi target names on cpu and CUDA | cuFFT / MKL-DFTI / an FFTW3-ABI `dlopen` ladder | no — capability only (`LORRAX_FFT_FFI` is on/off, and off *refuses*) | GATE 5b (load time only). **Which engine answered is not gated** — GATE 8 exists on `fix/host-ffi-fftw-container-stage-2026-08-06`, unmerged |
| **`ffi.gemm`** | `gemm_batch(a3, b3)`, inside the caller's own `shard_map` | batched host GEMM at vendor-BLAS rate, or an announced XLA fallback | Cray LibSci / MKL / netlib / AOCL / OpenBLAS CBLAS; batched entry point present or not | no — capability only | GATE 2 (one LibSci flavour). Which *entry point* was chosen is announced, not gated |
| **`ffi.io`** — parallel HDF5 | `open_file(path, mesh=, mode=)` → `write_sharded_slab` / `read_sharded_slab` | collective MPI-IO straight to a hyperslab; no rank-0 gather | cray-hdf5-parallel + Cray MPICH / phdf5 + Intel MPI; CUDA vs host `.so` | no — the platform comes from the mesh's devices | GATE 1, GATE 7 |
| **`ffi.common.ffi_loader`** — the meta-service | `get_lib(platform)`, `probe_target(target, platform)` | a three-way reason for every failure: unknown target / library would not load / library loaded but exports no such handler | `liblorrax_ffi.so` vs `liblorrax_ffi_host.so`; per-site RPATH | no; `LORRAX_FFI_SO` / `_HOST_SO` pin a path, and since 2026-08-06 a **set-but-missing pin refuses** rather than falling through | every other service's refusal quotes it |
| **`common.collectives`** | `prepare_mesh()`, `gather_k_blocks()` | the run's mesh with every communicator it will need already created | NCCL on CUDA; MPI — **not gloo** — on CPU ([transports](../environment/transports.md)) | no — `JAX_CPU_COLLECTIVES_IMPLEMENTATION` is a deployment fact, not a call-site choice | run-time, not build-time: `psum_scatter_checked` returns a Freivalds residual, because gloo's `reduce_scatter` is measured to return **wrong values** |
| **`file_io.wfn_loader`** | `WfnLoader(...).load(...)` | ψ(G) in the G-flat layout, byte-identical between backends | `eager` (h5py + numpy) vs `phdf5` (collective FFI + on-device unfold) | **escape hatch only** — `LORRAX_WFN_BACKEND`; the parity contract is what makes the choice safe to expose | `tests/bench/wfn_loader_backend_parity_test.py` (byte-identical) |
| **`common.jax_compile_cache`** | `ensure_jax_compile_cache()`, once, right after `runtime.bootstrap()` | a persistent compile cache that is safe across processes | filesystem (Lustre `$SCRATCH` vs `~/.cache`), world size, jaxlib generation | no | `tests/test_compile_cache_agreement.py`, with a `LORRAX_JAX_CACHE_FORCE_DIVERGE` positive control |
| **`runtime`** | `initialize_communicator_stack()` | one ordered bootstrap; every choice with more than one possible outcome is printed on rank 0 | allocator, plugin discovery, CPU-vs-GPU backend, glibc malloc | no | `tests/test_runtime_startup_report.py` — a dial absent from the report fails |

**The vendor packages are backends, not services.** `ffi.cusolvermp`,
`ffi.slate` and `ffi.scalapack` have **zero** direct-import consumers in
`src/`; every call site goes through `ffi.linalg.resolve.backend_module()`.
Listing them here as peers of `ffi.linalg` would invite someone to import
one. (`ffi.cublasmp` has no consumer at all — it is bench- and test-only.)

**Not services, kept nearby because callers reach for them from the same
place:** `common.timing` (pure instrumentation — nothing varies);
`common.sharding_fit`, `common.staged_reshard` and `common.contract_bands`
(movement patterns with no vendor library — `contract_bands` *consumes*
the `ffi.gemm` service rather than being one; `staged_reshard` is an
XLA-partitioner workaround with a single consumer); `runtime.xla_memory`
(a read-only mirror of jaxlib's own env parse — a fact collector).

**`ffi.gate` is the mechanism, not a service.** It is instantiated exactly
three times — `ffi/gemm.py` (`LORRAX_BANDS_GEMM_FFI`) and `ffi/fft.py`
twice (`LORRAX_FFT_FFI`, `LORRAX_FFT_FFI_FUSED`). `ffi.linalg`
deliberately does **not** use it: it resolves once from arguments, because
its choice comes from the deck rather than the environment.

A candidate service is arriving on a branch: **`common.vma`**
(`agent/vma-pvary-marking-2026-08-06`, unmerged) — one `mark_varying(init,
axes)` helper hiding *which jax spelling* marks a `shard_map` loop carry
as device-varying (`lax.pvary` on 0.7–0.8, `lax.pcast` from 0.9, identity
below 0.7.0, and a **refusal at import** on a jax that tracks varying
manual axes but offers neither). It is the same pattern applied to a
library *version* rather than a vendor library, and it is what unblocks
leaving jax 0.5.3. **Read `agent/jax-070-land-2026-08-06` instead** — it
contains that branch plus the container move and the supported-version
window, and is the branch the [JAX straddle
note](../environment/overview.md) tracks. Both are unmerged.

## Which services expose a choice, and which hide one {#choice}

These two look inconsistent and are not. The difference is worth stating
because harmonising them would be a regression in one direction or the
other.

**SlabIO hides the choice.** As of
`feat/slabio-one-backend-2026-08-06` (unmerged) there is no `backend=`
argument, no `use_ffi_io=`, no `SlabIOBackend`, and no `slab_io` deck key:
a caller states a path, a mode, a mesh and *logical* shapes, and gets
tiles. That is correct because **every alternative moved the same bytes
to the same place, at equal or worse cost**. `H5PY_ALLGATHER` was a
rank-0 gather — not a slow tier but an out-of-memory at the design size,
and it had to be refused at seven separate doors before an eighth
ungated route was found. `PHDF5_HOST` drove the *same* collective MPI-IO
through two extra Python packages, selected only by a stale `.so`. A dial
whose settings are "correct" and "worse in every measured respect" is not
a choice; it is a way to get it wrong.

**`ffi.linalg` exposes the choice, and should.** The choice is a deck
key, in three parallel spellings — `eigh_backend`
(`auto | off | distributed | cusolvermp | slate | scalapack`),
`distributed_cholesky` (`auto | off | cusolvermp | slate`) and
`distributed_lu` (`auto | off | distributed | cusolvermp | scalapack`) —
and **`ffi.linalg` reads no environment at all**. The environment grants
*capability* (which `.so`); the deck makes the *choice*.

That is correct because the backends are not interchangeable
implementations of one cost. `auto` for `eigh` resolves to **native** on
every platform, because the q-batched `jnp` path solves ndev matrices
concurrently while the FFI path solves one matrix ndev-ways and walks the
batch serially — **measured 100–600× slower per fit-size matrix**
(`common.eigh_benchmark --mode dispatch`). `distributed` is the name for
the opposite shape, one tile spread over the mesh, and is the only way
out of the replicated ζ-factor's O(nq·μ³)-with-no-P-scaling wall. Which is
right depends on what the caller is solving — knowledge SlabIO's callers
have no analogue of, because a byte transport has no such shape parameter.

The exposure is disciplined in two ways worth copying. An **explicit**
request never demotes: only `auto` demotes, and only with a rank-0
announcement naming whether the cause was geometry or capability. And
`eigh_backend`'s accepted vocabulary is *imported from the resolver*
(`eigh_backend_choices()` reads `ffi.linalg.resolve.BACKEND_CHOICES`), so
the deck parser cannot drift from the thing it configures. **The other two
keys hardcode their vocabularies** and can drift; that is a defect in
them, not a second design.

**A third position: `file_io.wfn_loader`** exposes `LORRAX_WFN_BACKEND`
(`eager` vs `phdf5`) as an *escape hatch*, not a tuning dial. That is only
defensible because the two backends are held **byte-identical** by a
parity test — the choice cannot change an answer, only how it was
obtained. An escape hatch without a parity contract is just an
undocumented backend selector.

**The FFT and GEMM dials are neither.** `LORRAX_FFT_FFI` and
`LORRAX_BANDS_GEMM_FFI` are two-valued capability switches, not backend
selectors — you cannot ask for cuFFT-instead-of-MKL, only for the service
or (for GEMM) an announced, uncertified debug fallback. Since the
2026-08-01 ruling that [FFI backends are required](decisions.md),
`LORRAX_FFT_FFI=0` **refuses** rather than falling back, because the XLA
twin it would have fallen back to is deleted.

**The rule this suggests**, stated so a new service has something to argue
against: *expose a choice only where the alternatives have genuinely
different costs that the call site — not the deployment — is in a position
to judge; and where you expose one, either hold the alternatives to a
parity contract or make the cost difference the documented reason.*
Deployment facts (which MPI, which HDF5, which FFTW3 `.so`) belong to the
environment and to [`ffi_layout.md`](ffi_layout.md); call-site facts (a
batch of small tiles versus one large one) belong in the deck.

---

# Part 2 — the API reference

Conventions used throughout:

* **Level / imports.** Imports run downhill only, L1 → L2 → L3
  ([layers](layers.md)). Every service on this page is L3 unless marked;
  an L3 module imports nothing above L3.
* **Resolve time vs trace time.** *Resolve-time* checks fire when a
  factory/plan is built (platform, mesh geometry, handler probe, env
  grammar); *trace-time* checks fire inside the returned callable on the
  operand (dtype, rank, extents) — a factory cannot know them earlier.
* **Announce-or-refuse.** A demotion must say so from the rank it
  happened on; an explicit request that cannot be honored raises with the
  reason and the fix. Silence is legal only where declared.

## The dependency tree

ASCII (mkdocs here has no mermaid fence). Downhill edges only, except the
sanctioned exceptions marked `!`:

```
L1  physics drivers & kernels
    gw/  bse/  psp/  isdf/  bandstructure/  centroid/  file_io readers
     │ calls
     ▼
L2  solvers/  mixing/  common.minimax  common.cholesky_2d
    common.rank_criterion  centroid.kmeans_isdf
     │ calls
     ▼
L3  substrate services (this page)
    ┌──────────────────────────────────────────────────────────────┐
    │ runtime ─────────── process bootstrap, xla_memory            │
    │   └─ uses → common.collectives (mesh + warm-up)              │
    │ common.collectives ── THE cross-process layer                │
    │   ├─ common.contract_bands ──┐ movement primitives           │
    │   ├─ common.staged_reshard ──┤  (use collectives specs +     │
    │   └─ common.sharding_fit ────┘   ffi.gemm via its Gate)      │
    │ common.fft_helpers ─ delegates gated bodies to ffi.fft       │
    │ ffi.gate ─ the one env-gate resolver                         │
    │   ├─ ffi.fft   (flat-k MKL-DFTI / cuFFT)                     │
    │   ├─ ffi.gemm  (vendor-CBLAS batched GEMM, host)             │
    │   └─ ffi.linalg (plan → resolve → dispatch;                  │
    │        backends: scalapack / slate / cusolvermp / native)    │
    │ ffi.io ─ parallel-HDF5 MPI-IO handlers (ffi.phdf5 = shim)    │
    │ file_io.slab_io ─ sharded-slab HDF5 transport                │
    │   └─ tiers: _slab_io_ffi (ffi.io) → _slab_io_mpi_host        │
    │             (mpi4py+h5py) → _slab_io_allgather               │
    │             [collapsing to ONE tier — see slab_io below]     │
    │ common.timing ─ instrumentation (leaf; no deps)              │
    └──────────────────────────────────────────────────────────────┘

sanctioned upward edges (tests/test_layering.py, argued in layers.md §5):
 !R1  file_io.slab_io ──▶ gw.gw_config          (lazy; SlabIOBackend enum)
      — closed by deletion on feat/slabio-one-backend-2026-08-06
 !R2  solvers.sternheimer_solve ──▶ psp.dft_operators
 !R3  centroid.kmeans_isdf ──▶ centroid.orbit_syms   (lazy)
 !R4  mixing.acceleration writes JAX_ENABLE_X64 at import (env touch)
```

---

## runtime {#runtime}

**Purpose.** Process bootstrap: everything that must happen before and
around the first backend init, in the one order that is correct.

**Public API** (`src/runtime/__init__.py`, `src/runtime/xla_memory.py`):

```python
def initialize_communicator_stack(*, platform: str = "gpu",
                                  axis_names=("x", "y"),
                                  print_fn=None) -> RuntimeStack
def finalize_process(rc: int = 0)                        # does NOT return
def bootstrap(*, platform: str = "gpu") -> None          # steps 2–5 only
def set_default_env(*, platform: str = "gpu") -> None
def install_failfast_excepthook() -> None
def skip_gpu_plugin_discovery(*, announce: bool = True) -> bool
def init_jax_distributed() -> None
def nccl_warmup(mesh_xy) -> None
def fallback_to_cpu_if_no_gpu_backend() -> None
def announce_cpu_collectives() -> None
def tune_glibc_malloc() -> bool
def collect_startup_facts(mesh, *, cache_error: str | None = None) -> dict
def format_startup_report(f: dict) -> list

# runtime/xla_memory.py (moved out of gw.gw_config 2026-07-31; re-exported there)
def resolve_xla_gpu_memory_env() -> XlaGpuMemoryEnv
def classify_xla_pool(stats, *, backend: str = "gpu",
                      env: XlaGpuMemoryEnv | None = None) -> XlaPoolReading
```

`RuntimeStack` (slots): `mesh` (the run's clique-warmed `('x','y')` mesh —
never build a second one), `platform`, `device_kind`, `n_devices`,
`n_local_devices`, `process_index`, `process_count`, `facts`, `report`.

**Contract.** Call `initialize_communicator_stack()` ONCE at the top of a
driver, **above the driver's own `import jax`** (the module imports jax
only inside function bodies); all seven chain drivers do (2026-08-01).
`finalize_process(rc)` is the sanctioned driver exit: explicit ordered
teardown (effects barrier → unregister jax's `clean_up` atexit →
`jax.distributed.shutdown()` → run remaining atexit hooks → announced
`os._exit(rc)`) — it exists because jax's own interpreter-exit client
destruction deadlocks after fully-cold in-process compile storms (jobs
7884928/7884989); `gw.gw_jax` is the adopter. Order: failfast hook → env defaults →
collectives announcement → plugin skip → `jax.distributed` → backend init
(the CPU demotion point) → mesh + warm-up → compile cache → the rank-0
startup report. Refuses: an unrecognized `XLA_PYTHON_CLIENT_ALLOCATOR`
spelling (resolve time, before jaxlib turns it into a fake missing-backend
error); a cuda-init failure **on a GPU node** is re-raised, never masked.
Announces: every demotion tagged `DEMOTION:`, the resolved collectives
implementation (with the gloo corruption warning), every FFI dial's
resolved mode *including off*. Adding a dial without adding it to the
report fails `tests/test_runtime_startup_report.py`.

**Level / deps.** L3. Uses `common.collectives` (mesh, warm-up),
`common.jax_compile_cache`, `ffi` gates (facts only).
`runtime.nccl_warmup` and `collectives.warm_mesh_cliques` are deliberately
**not merged** — only the call site (`prepare_mesh`) is shared
([layers "What NOT to unify"](layers.md#what-not-to-unify)).

**Deep dive.** `docs/dev/env_vars.md` §0 (the startup report contract),
[environment/overview](../environment/overview.md).

---

## common.collectives {#commoncollectives}

**Purpose.** *The* cross-process layer: mesh ownership, warm-up, and every
collective a driver should need — callers never touch `Mesh`,
`NamedSharding`, `shard_map` or `multihost_utils` directly.

**Public API** (`src/common/collectives.py`, selected — drivers get their
warmed mesh from `runtime.initialize_communicator_stack` (all seven do,
2026-08-01); `prepare_mesh` is the library-level door, `gather_k_blocks`
the workhorse collective):

```python
def prepare_mesh(mesh=None, *, axis_names=("x", "y"), print_fn=print)
def resolve_mesh(mesh=None, *, axis_names=("x", "y"))
def single_device_mesh()
def warm_mesh_cliques(mesh, *, print_fn=print) -> float

def gather_k_blocks(n_k: int, per_k, *, item_shape, label: str,
                    lookahead: int | None = None, print_fn=print)
def sweep_local_k(ks_local, blk: int, item_shape, per_k, *,
                  lookahead=None, print_fn=print, label="k")
def psum_scatter_checked(x, axis, scatter_dimension, tiled=True, *, name)
def report_collective_residual(name, residual, *, rtol=COLLECTIVE_RTOL,
                               print_fn=print) -> None

def device_put_process_local(host_array, sharding, *, check: bool | None = None)
def replicate_to_mesh(host_array, mesh)
def shard_over_k(arr, mesh, *, axes=None)
def all_gather_processes(x, *, tiled: bool = False)
def gather_to_host(x)
def gather_indexed_blocks(local_vals, local_idx, n_total: int)
def process_count() -> int;  def process_rank() -> int
def device_count() -> int;   def barrier(name, *, print_fn=print) -> bool
```

**Contract.**

* `prepare_mesh` = `resolve_mesh` + warm every communicator the run will
  need (`warm_mesh_cliques` on CPU/`impl=mpi`; the NCCL warm-up stays in
  `runtime`). Call once, before any jit; both warm-ups are no-ops off
  their platform, at P=1, and on an already-warmed mesh. It **is** a
  collective — call synchronously on every rank.
* `resolve_mesh` refuses (resolve time) a mesh this process cannot compute
  on (`_require_addressable`) instead of a bare `StopIteration` inside a
  jit; with no mesh given it returns the **canonical mesh** — the square
  s×s mesh over `jax.devices()` (square-only ruling, decisions.md
  2026-08-01: a non-perfect-square device count REFUSES, naming the square
  counts to request — idle-rank truncation would deadlock under impl=mpi),
  built ONCE per process per axis-name tuple and cached (`_CANONICAL_MESHES`, 2026-08-01): a library
  that re-resolves "the run's mesh" after `initialize_communicator_stack`
  gets THE object back, not an equal-but-distinct twin that would double
  every shape-keyed jit cache (the identity contract of
  `single_device_mesh`, extended to the run mesh).
* `single_device_mesh` — the ONE 1×1 process-local mesh per process
  (identity is load-bearing: shape-keyed jit caches embed the mesh
  object). `wfn_transforms.process_local_mesh` is an alias, not a copy.
* `warm_mesh_cliques` creates each mesh-axis MPI communicator plus the
  world clique from the **main thread** in a jit small enough to take
  XLA's sequential thunk executor — the mechanism that satisfies jaxlib's
  `MPI_Is_thread_main` guard. Skipping it kills any clique first created
  inside a real jit under `impl=mpi`
  ([transports](../environment/transports.md)).
* `psum_scatter_checked` returns a **pair** `(out, residual)` — a
  Freivalds checksum of its own output, born from gloo's silent
  reduce-scatter corruption; pass the residual to
  `report_collective_residual`.
* `device_put_process_local` places a host array **without a collective**;
  precondition (trace-time, unchecked by default): the array is
  bit-identical on every process. `LORRAX_CHECK_REPLICA=1` re-enables
  JAX's assertion at the cost of the all-gather.
* `Mesh(` construction outside `common.collectives` / `runtime` /
  `bse.bse_ring_comm` fails `tests/test_layering.py` rule 4.

**Level / deps.** L3. Leaf over jax; `centroid/distribution.py` is the one
policy client above it.

---

## common.contract_bands {#commoncontract_bands}

**Purpose.** THE multi-stage band projection + reshard: two-stage
`psum_scatter` chain on the `('x','y')` mesh so no (m,n,k)-, (m,μ)- or
(μ,μ)-sized object is ever materialized on one rank.

**Public API** (`src/common/contract_bands.py`):

```python
def contract_bands_block_reshard(
    mesh_xy: Mesh, *,
    channels: str = "none",        # "none" | stacked-channel plans
    extra: str = "none",
    axes: tuple[str, str] = ("x", "y"),
    divisibility_hint: str = "",
) -> Callable
def bands_gemm_ffi_mode() -> str       # re-exports of the ffi.gemm dial
def bands_gemm_ffi_enabled() -> bool   # (factory-time, cache-key safe)
```

**Contract.** Resolve time: refuses a mesh whose minor (consecutive-rank)
axis is not `axes[1]` — the large payload must reduce-scatter over the
node-local axis; refuses non-divisible extents, naming
`divisibility_hint`. Reads **no environment** — its GEMM backend comes
from `ffi.gemm.GATE`, a typed capability object (the target state for L2/L3
dials). The dial is factory-time: consumers fold `ffi.ffi_dial_key()` into
kernel cache keys. Encodes (measured, cited in the deep dive):
large-payload-on-'y' ordering, stacked collectives (one per mesh axis, flat
in channel count), de-promoted f64-split lowering for real operands.

**Level / deps.** L3 despite the noun ([layers §4.1](layers.md)). Depends
on `common.collectives` idioms and `ffi.gemm`.

**Deep dive.** `docs/dev/staged_reshard_primitive.md` (the API contract +
evidence, jobs 7879008/7879010).

---

## common.staged_reshard {#commonstaged_reshard}

**Purpose.** The movement-only sibling: a staged `(B, M, N)` face→batch
reshard (`P(None,'x','y')` → `P(('x','y'),None,None)`) as two
single-axis `all_to_all`s, avoiding GSPMD's "involuntary full
rematerialization" (a measured 64× per-rank residency blow-up, job
7882974).

**Public API** (`src/common/staged_reshard.py`):

```python
def face_to_batch_reshard(mesh: Mesh, *,
                          axes: tuple[str, str] = ("x", "y"),
                          route: str = DEFAULT_ROUTE,
                          divisibility_hint: str = "",
                          log_fn=None) -> Callable
def face_to_batch_reshard_supported(mesh, shape, *,
                                    axes=("x", "y"),
                                    route: str = DEFAULT_ROUTE) -> bool
```

**Contract.** Resolve time: `axes[1]` must be the mesh's minor axis;
divisibility refusals name the hint. `*_supported` is the no-raise probe
for callers with a fallback. Trace time: operand rank/extent checks in the
returned callable. Announces its route once per site via `log_fn`.
**The factory must be invoked synchronously on every rank** — it warms
the mesh cliques (a no-op off `impl=mpi` and on an already-warm mesh);
under `impl=mpi` a rank that skips it deadlocks the others in the clique
warm-up.

**Routes.** Two schedules, selected by `route=`, bit-exact against each
other: `split_b_first` (default: all_to_all over `x` then `y`; needs
`B % (p_x·p_y)` plus the input layout's `M % p_x`, `N % p_y`) and
`flatten_m_first` (all_to_all over `y` then over the flattened
`(x, y)`; additionally needs `M % (p_x·p_y)`, and pads M locally when it
does not divide — e.g. M=672 → 704 at P=64, +4.76%).

**Level / deps.** L3. Sole consumer:
`bandstructure/bse_setup.py:428` (the fH_q face→batch move in
`_q_batch`). One consumer is why this is a helper and not a
[service](#inventory).

**Deep dive.** The module docstring of `src/common/staged_reshard.py` is
the real contract (routes, divisibility, residency proof, measured
domain). `docs/dev/staged_reshard_primitive.md` documents the
*contraction* sibling `common.contract_bands`, not this module.

---

## common.sharding_fit {#commonsharding_fit}

**Purpose.** A `PartitionSpec` that is *legal* for the extents in hand —
kills the `IndivisibleError`-inside-jit class (785 = 5×157 divides
nothing; job 7882476).

**Public API** (`src/common/sharding_fit.py`):

```python
def fit_sharding(mesh, spec, shape, what: str = "", print_fn=None,
                 itemsize: int = 16)
def legal_spec(mesh, spec, shape, what: str = "", print_fn=None,
               itemsize: int = 16)
def shard_factor(mesh, entry) -> int
def padded_extent(mesh, entry, n: int) -> int
```

**Contract.** Resolve time, pure host arithmetic (no jax objects
created). Demotes an axis that does not divide to replicated and
**announces** the demotion with the per-rank memory consequence
(`itemsize`); `what` names the tensor in the message. Never silently
changes a legal spec.

**Level / deps.** L3. `runtime.padding` is the padding-side counterpart
(pad to divisibility instead of demoting).

---

## common.rank_criterion (L2) {#commonrank_criterion}

**Purpose.** The pseudo-inverse truncation criterion: how many spectral
directions survive an amplification cap — physics-free, liftable.

**Public API** (`src/common/rank_criterion.py`):

```python
def select_rank(spectrum, rtol)
def noise_floor_rtol(n_rows, n_cols=None)     # reference line ONLY, never a cut
def rank_report(spectrum, rtol, *, label="truncation",
                quantity="singular values", rank_used=None,
                n_rows=None, n_cols=None)     # -> RankReport
```

**Contract.** **L2**: reads no environment, imports no L1 — dials are
parameters. `rank_report` returns the announcement object (the caller
prints); cutting at `noise_floor_rtol` is documented-catastrophic and the
function's docstring says so.

---

## ffi.gate {#ffigate}

**Purpose.** ONE resolver for every env-gated, rank-local FFI capability:
grammar, platform, probe, announce-or-refuse — in one place instead of
per-dial drifting copies.

**Public API** (`src/ffi/gate.py` — the only `gate.py` in the tree; the
`ffi.common.gate` re-export shim was **deleted** 2026-08-06 in `ad71053`
— on `integration/2026-08-06`, **not** on `origin/main` — with zero
importers left):

```python
@dataclass(frozen=True)
class Gate:
    env: str; target: str; platforms: tuple[str, ...]
    modes: tuple[str, ...]; default: str; off_label: str
    # + per-service message fields (label, auto_on_msg, resolved_msg, …)

    def mode(self) -> str                    # tier 0: strict grammar
    def enabled(self) -> bool                # tier 1: LEXICAL, cache-key safe
    def platform_ok(self, mesh) -> bool
    def require(self, mesh, *, target=None) -> str      # announce-or-REFUSE
    def resolve(self, mesh, *, target=None) -> str | None  # tier 2, mode-aware

def rank_id() -> Optional[int];  def rank0() -> bool
def announce_once(key, msg, *, scope: str = "rank0") -> bool
def mesh_ffi_platform(mesh) -> str
def reset_gate_state() -> None               # tests only
MODE_SPELLINGS = {"off": ("0","off","false","no"),      # two-valued: the
                  "on":  ("1","on","true","yes")}       # auto tier is gone

# src/ffi/__init__.py — the ONE aggregate of factory-time dial state;
# consumers fold it into kernel cache keys (ppm_tau_kernel, cohsex_sigma,
# w_isdf).  Tier-1 lexical reads only: safe in any cache-lookup path.
def ffi_dial_key() -> tuple    # (("fft_ffi", bool), ("fft_ffi_fused", bool),
                               #  ("bands_gemm_ffi", bool))
```

**Contract.** Two tiers, deliberately: `enabled()` never initializes the
JAX backend (it is the kernel-cache key — `ffi.ffi_dial_key()` aggregates
every dial); `resolve()`/`require()` are mesh-aware and exact. Grammar
errors announce once (rank-locally) and fall to **off**. `require()` is
mode-independent and refuses (resolve time) on out-of-scope platform or a
failed probe, quoting `probe_target`'s three-way reason. Operand
dtype/rank/extent refusals are trace-time and live in the consumer wrapper
bodies — a single-phase `plan()`-shaped API would have to lie about when
it checked, which is why these dials do not fold into `ffi.linalg`.
`rank_id()` reads launcher env vars **before** `jax.process_index()`
(which would initialize the backend).

**Level / deps.** L3. Depends only on `ffi.common.ffi_loader`.

**Deep dive.** `docs/dev/ffi_gate_contract.md`.

---

## ffi.fft {#ffifft}

**Purpose.** The flat-k batched 3-D FFT service: MKL FFT (DFTI API) on cpu
meshes, cuFFT advanced layout on CUDA, under the same target names —
stride descriptors read the dot-layout tile in place, deleting XLA's
layout transposes (sigma.exec 272 → 71.9 s at nb=128/P=64).

**Public API** (`src/ffi/fft.py`; `ffi.mklfft` is a shim. Physics code
enters via `common.fft_helpers`, which delegates its gated bodies here —
`fft_helpers.make_flat_k_fft` is the single door):

```python
GATE       = Gate(env="LORRAX_FFT_FFI",       modes=("off","on"), default="on",
                  off_policy="refuse",    # the XLA twin is DELETED
                  platforms=("cpu","CUDA"), target=FLAT_K_TARGET)
FUSED_GATE = Gate(env="LORRAX_FFT_FFI_FUSED", modes=("off","on"), default="on",
                  off_policy="fallback",  # decomposed chain, still FFI
                  platforms=("cpu","CUDA"), target=GW_CONV_TARGET)

def fft_ffi_enabled() -> bool;   def fft_ffi_mode() -> str
def fused_fft_ffi_enabled() -> bool;  def fused_fft_ffi_mode() -> str
def require_fft_ffi(mesh: Mesh, target: str = FLAT_K_TARGET) -> str
def make_flat_k_fft_ffi(mesh, kgrid, spec, *, kind, norm, out_spec) -> Callable
def make_gw_conv_ffi(mesh, kgrid, g_spec, v_spec, *,
                     norm='ortho', mult=1.0) -> Callable
def ffi_fft_scale(kind, norm, nk) -> float
def validate_flat_spec(spec: P, what: str) -> P

# the single door (common/fft_helpers.py; unconditionally FFI since the
# 2026-08-01 ruling — the gated XLA twin is deleted, =0 refuses):
def make_flat_k_fft(mesh, kgrid, spec, *, kind, norm='ortho', out_spec=None)
def make_flat_k_ifftn(mesh, kgrid, spec, *, norm='ortho', out_spec=None)
def make_flat_k_fftn(mesh, kgrid, spec, *, norm='ortho', out_spec=None)
# shard_map-interior XLA layer, KEPT (no FFI route exists; BSE/isdf/psi):
def make_sharded_ifftn_3d(mesh, in_spec, out_spec, *, norm=None,
                          axes=(-3, -2, -1))
def make_sharded_fftn_3d(mesh, in_spec, out_spec, *, norm=None,
                         axes=(-3, -2, -1))
```

**Contract.** Resolve time (factory): `validate_flat_spec` refuses a
sharded k-axis; `require_fft_ffi` refuses an unusable backend with the
probe reason — never a silent XLA fallback for an explicit request.
`LORRAX_FFT_FFI_FUSED` selects **which entry point** the τ kernel builds,
not which backend serves it (one refusal wording, owned by `GATE`). Both
dials are factory-time reads and MUST be in every consumer cache key
(`ffi.ffi_dial_key()`). Parity gates: unit 1e-16, h5 ≤ 2.5e-14 eV.

**The `make_w_densifier` consumer pattern** (`bse/bse_io.py`): the
coarse→fine W densifier composes the `shard_map`-interior kernels —
`make_sharded_ifftn_3d` → zero-pad in R → `make_sharded_fftn_3d` — inside
ONE `jax.jit` whose `out_shardings` pins the (μ,ν) spec, so per-rank peak
stays at the local tile and no replicated N_μ²-class array exists
(audit P0-4/P2-7; the eager `local_ifftn3` + `device_put` form is banned
by `tests/test_fft_shardmap_context.py`). New sharded FFT consumers should
copy this shape.

```python
def make_w_densifier(mesh_xy: Mesh, w_spec: P,
                     fine_grid: tuple[int, int, int], *,
                     output: str = "k") -> Callable   # 'k' | 'R'
```

**Level / deps.** L3. Uses `ffi.gate`, `ffi.common.ffi_loader`; handlers
in `src/ffi/cpp/mklfft/` and `src/ffi/cpp/cufft/`.

**Deep dive.** `docs/dev/flat_k_fft_service.md`; portability end-state
[ffi_layout §7](ffi_layout.md).

---

## ffi.gemm {#ffigemm}

**Purpose.** Vendor-CBLAS batched GEMM on the host — `contract_bands`'
right-GEMMs at BLAS rate (1.6–1.9× over XLA:CPU Eigen dots, jobs
7879008/7879010). Not a general GEMM service.

**Public API** (`src/ffi/gemm.py`; `ffi.mklblas` is a shim keeping the
historical `bands_gemm_*` names):

```python
GATE = Gate(env="LORRAX_BANDS_GEMM_FFI", modes=("off","on"),
            default="on", off_policy="fallback",  # XLA arm kept for minor
            platforms=("cpu",), target=GEMM_TARGET)

def gemm_ffi_enabled() -> bool;  def gemm_ffi_mode() -> str
def require_gemm_ffi(mesh) -> str
def gemm_batch(a3, b3)     # A (BA,M,K) @ B (BB,K,N) -> C (BA,M,N), C[i]=A[i]@B[i%BB]
```

**Contract.** `platforms=("cpu",)`, so on a GPU mesh the demotion is
**declared-silent** — this is the only `Gate` in the tree that sets
`silent_platform_demote`, and it is justified: XLA:GPU already dispatches
cuBLAS, so there is nothing for a reader to act on. *(This section used to
call it "the only default-`auto` dial". That is false since `2a73b4b`
(on `integration/2026-08-06`, not `origin/main`) deleted the `auto`
vocabulary: `MODE_SPELLINGS` is
two-valued, all three `Gate`s declare `modes=("off","on"), default="on"`,
and `Gate.__post_init__` would now refuse a gate declaring `auto`.)*
Trace time:
`gemm_batch` refuses mismatched K, `BA % BB != 0`, and mixed dtypes;
dtype dispatch (f64/f32/c128/c64) happens inside the `.so`. No
`shard_map` here **by design**: it is a rank-local handler called from
inside the caller's own `shard_map` body — the structural difference from
the `slate/scalapack/cusolvermp/cublasmp` families whose handlers hold an
MPI/NCCL context.

**Level / deps.** L3. `ffi.gate` + host handler
`src/ffi/cpp/mklblas/gemm_batch_ffi.cc` (runtime dlsym with announced
refusal — no MKL present means slower-and-loud, never broken).

**Deep dive.** `docs/dev/vendor_gemm_service.md`.

---

## ffi.linalg {#ffilinalg}

**Purpose.** Distributed dense linalg (`eigh`, `cholesky`, `solve_lu`)
over the `('x','y')` mesh: resolve a backend once, get a plan that owns
the layout contract and the call.

**Public API** (`src/ffi/linalg/{plan,resolve,dispatch}.py`):

```python
def plan(op: str, mesh_xy: Mesh, *, backend: str = "auto",
         n: int | None = None) -> LinalgPlan

@dataclass
class LinalgPlan:
    op: str; requested: str; backend: str; mesh: Mesh; n: int | None
    in_sharding: NamedSharding | None        # P('x','y') tile
    batch_in_sharding: NamedSharding | None  # P(None,'x','y') stack
    # plan.is_native; plan(A_tile); plan.batched(A_stack)

def resolve_backend(op, requested, mesh_xy, *, n=None) -> str
def list_backends(op, mesh_xy) -> dict[str, str]  # name -> "available (…)" |
                                                  #   "unavailable: <first failed guard>"
def mesh_platform(mesh_xy) -> str;  def mesh_is_cpu(mesh_xy) -> bool
def dispatch_eigh(A, mesh_xy, backend: str)
def ensure_sharding(x, sharding: NamedSharding)
```

**Contract.** All policy at resolve time: `plan()` runs
`resolve_backend` (`auto` policy, platform/geometry guards, the `n`
divisibility guard when pinned) and stores the answer — route strings are
byte-identical to calling the resolver directly. The plan owns the layout
contract (`P('x','y')` single tile, `P(None,'x','y')` stack, eigenvalues
replicated) and reshards operands into it; `plan.batched` is uniform
whether or not the library has a batched entry point. `plan()` on a
native Cholesky/LU **raises** — those fast paths belong to the caller's
jit (`plan.is_native` says so); native `eigh` is the one runnable
exception.

Backend **choice** is an input-deck key and **this package reads no
environment at all**; `LORRAX_FFI_SO` / `_HOST_SO` grant capability via
`ffi_loader`, and the platform comes from the mesh's devices, never from
`JAX_PLATFORMS`. Vocabulary per op, and the deck key that sets it:

| op | deck key | accepted | notes |
|---|---|---|---|
| `eigh` | `eigh_backend` | `auto \| off \| distributed \| cusolvermp \| slate \| scalapack` | vocabulary is **imported** from `resolve.BACKEND_CHOICES`, so parser and resolver cannot drift. `use_low_mem_eigh = true` rewrites `auto`/`native` to `distributed` |
| `cholesky` | `distributed_cholesky` | `auto \| off \| cusolvermp \| slate` | no `distributed` spelling, deliberately. Vocabulary is a **hardcoded duplicate** — it can drift |
| `solve_lu` | `distributed_lu` | `auto \| off \| distributed \| cusolvermp \| scalapack` | vocabulary is likewise a hardcoded duplicate |

Six guards run in one fixed order (vocabulary → platform → known-broken →
capability probe → process coverage → geometry → `n` divisibility), all at
resolve time. Two are worth knowing at the call site: **SLATE `heev` on a
CPU mesh is a hard `RuntimeError`**, not a demotion (bug L-2, deterministic
SIGSEGV), and cuSOLVERMp `syevd` on a rectangular mesh **deadlocks inside a
collective** rather than returning an error, so the square-mesh guard is
load-bearing rather than tidy. `off` is an override, not a guard: it is
honoured unconditionally, before every check.

Backends today: ScaLAPACK `pzheevd` (the permanent CPU distributed eigh),
SLATE, cuSOLVERMp (deletion candidate, [ffi_layout §5](ffi_layout.md)),
`native`.

**Level / deps.** L3. `ffi.common.ffi_loader` + the backend packages;
consumed by `isdf/core`, `gw/w_isdf`, `bandstructure/bse_setup`,
`bse/vq_interp`.

**Deep dive.** `docs/dev/linalg_ffi.md`.

---

## ffi.io — parallel HDF5 {#ffiio}

**Purpose.** The MPI-IO handlers themselves: each process reads and writes
its own hyperslab of a shared HDF5 file, with no gather through rank 0.
`file_io.slab_io` is the physics-facing service over this one; **`ffi.io`
has a second, independent consumer in `file_io.wfn_loader`**, which is why
it is listed separately rather than folded into SlabIO.

**Public API** (`src/ffi/io.py`):

```python
def open_file(path: str, *, mesh: Mesh, mode: str = "w") -> int   # opaque PhdfCtx*
def close_file(path_or_handle) -> None
def write_sharded_slab(fh, ds_name, A, *, mesh,
                       global_shape=None, valid_shape=None) -> None
def read_sharded_slab(fh, ds_name, *, global_shape, dtype, mesh)
def read_kchunk_sharded(...);  def read_kchunk_union_sharded(...)
```

**Contract.** Backend is hidden: `_platform_for_mesh` picks `CUDA` vs
`cpu` from the mesh's devices, and the platform is recorded per handle so
`close_file` routes back through the *opening* library. Refuses at resolve
time: a mode outside `{w, a, r}`; a mesh without both `x` and `y` axes;
`p*q != jax.process_count()`; a non-2-D operand to `write_sharded_slab`;
and — the orphaned-inode guard — **re-opening an already-open path with a
different mode**. No `Gate`, no env dial.

**Level / deps.** L3. `ffi.common.ffi_loader` plus the C++ handlers in
`src/ffi/cpp/phdf5/`.

**Migration note.** `ffi.phdf5` is four re-export shims over this module,
and their own docstring says deleting the shim is the gate that the
migration is complete. **That gate is not met, and no production module
has migrated**: every importer under `src/` still reaches `ffi.phdf5` —
`file_io/_slab_io_ffi.py` (4 sites) and `file_io/wfn_loader.py` (3) —
and `src/` contains **zero** direct imports of `ffi.io`. Only tests import
the new name.

---

## file_io.slab_io {#file_ioslab_io}

**Purpose.** The sharded-slab HDF5 transport (the L3 half of `file_io`;
the format readers above it are L1): each rank writes/reads its own
hyperslab, replacing ad-hoc allgather-to-rank-0 patterns.

**Public API** (`src/file_io/slab_io.py`):

```python
class SlabIO:   # context manager
    def __init__(self, path, *, mode: str = "w", mesh=None,
                 backend=None,               # SlabIOBackend enum
                 use_ffi_io: bool | None = None) -> None   # legacy, coerced
    def create_dataset(self, name, *, shape, dtype, chunks=None,
                       attrs=None) -> None
    def write_slab(self, name, A, *, offset=None, global_shape=None,
                   valid_shape=None, dtype=None, chunks=None,
                   k_chunk_size=None) -> None
    def read_slab(self, name, *, shape=None, dtype=None, offset=None,
                  valid_shape=None, mesh=None, partition_spec=None,
                  as_numpy: bool = False) -> jax.Array
    def write_attr(self, name, value) -> None
    def close(self) -> None
```

**Contract on this branch.** Three backend tiers, selected by
`SlabIOBackend` (defined in `gw.gw_config` — sanctioned exception R1;
`slab_io = auto` in the deck resolves via the capability router in
`LorraxConfig.from_input_file`):

| tier | module | mechanism | requires |
|---|---|---|---|
| `PHDF5_FFI` | `_slab_io_ffi` | collective MPI-IO via `ffi.io` (CUDA lib or CUDA-free host lib) | mesh; the lib exports the handler |
| `PHDF5_HOST` | `_slab_io_mpi_host` | same MPI-IO, driven by mpi4py + parallel h5py | mesh; the overlay |
| `H5PY_ALLGATHER` | `_slab_io_allgather` | gather to rank 0, serial h5py | reachable at **exactly one process** |

Resolve time: `PHDF5_FFI`/`PHDF5_HOST` without a mesh raise; a non-enum
`backend` raises `TypeError`. Padding contract: files always store the
**logical** shape — producers pass `valid_shape` for the un-padded prefix,
consumers request a mesh-divisible physical `shape` and get a zero-filled
tail (driver-side padding is `runtime.padding`'s job). The host-MPI path
writes synchronously by design (the threaded FFI deadlocks at `H5Fclose`
under `MPI_THREAD_SINGLE` — `_slab_io_mpi_host.py` docstring).

`H5PY_ALLGATHER` is a **refusal, not a fallback** above one process
([decisions.md](decisions.md) 2026-08-05): it materialises the whole array
on rank 0, which is the memory wall the per-rank-tile contract exists to
avoid.

### The interface is collapsing to one transport {#slab-io-one-transport}

**On `feat/slabio-one-backend-2026-08-06` (unmerged — not an ancestor of
`integration/2026-08-06` or of `origin/main`)** the three tiers, the enum,
the router and the two deck keys are **deleted**, and the constructor
becomes `SlabIO(path, *, mode="w", mesh)` — nothing else. This is the
flagship worked example of the [hide-the-choice](#choice) position, and
new services should be read against it:

* **A caller no longer computes a mesh-divisible extent.** `read_slab`
  with no `shape` returns the dataset rounded *up* to the mesh-divisible
  extent under `partition_spec`, zero-filled past the dataset — which is
  the padded consumer buffer every sharded consumer actually wanted. It
  used to **refuse** that call, so every caller computed the round-up
  itself and two of them computed it differently. The easy call is now the
  correct call.
* **`write_slab` clips the pad to the dataset**: the extent written is
  `min(A.shape, dataset - offset)` per dimension, derived from the dataset
  SlabIO already knows about, so a buffer padded for mesh divisibility
  needs no extra argument. `valid_shape` survives only as the ragged-chunk
  override.
* **The deletion closed [layers.md](layers.md) request R1** — "the enum
  needs a home neither package owns". The answer was *nowhere*: with one
  transport there is no enum, no `backend` parameter, and no uphill
  L3 → L1 import.
* **Seven refusals were not enough.** `H5PY_ALLGATHER` had been closed at
  seven separate doors across three sessions, each closure reported
  complete, and an eighth ungated route survived all of them — a direct
  `from file_io._slab_io_allgather import _to_host` in `gw/gw_init.py`,
  bypassing the enum entirely. **A capability that must be refused at
  seven doors is dead code wearing a safety label**; the branch deletes
  the module instead, and the guarantee then holds by construction rather
  than by a check. That lesson generalises past SlabIO and is the reason
  this page has a "hide the choice" position at all.
* **`restart = true` works at P>1 again**, ported to per-rank tiles and
  bit-exact per shard at nspinor 1/2/4 including bispinor. The previous
  full-file reader was guarded off above one process with no replacement.

**Level / deps.** L3. `ffi.io` (lazy), mpi4py/h5py (lazy), the R1 lazy
enum import — the last two go away with the tiers.

**See also.** [`slab_io.md`](slab_io.md) — the transport in full: the
per-rank-tile contract, the launcher requirement and the singleton-MPI
trap it avoids, the measured striping policy, the certification, and the
failure modes. That page is rewritten on the same branch.

---

## common.timing {#commontiming}

**Purpose.** The run's hierarchical timing tree — how every stage lands in
the end-of-run table without drivers hand-rolling timers.

**Public API** (`src/common/timing.py`, module-level convenience over
`TimingCollector`):

```python
def section(name: str, *, announce: bool = False,
            label: str | None = None) -> TimingSection   # with-block
def timed(name: str | None = None, *, watch: bool = False) -> Callable  # decorator
def record(name: str, seconds: float, *, count: int = 1) -> None  # top-level row
def report(*, print_fn=print, title="--- Timing (seconds) ---",
           min_percent=None, max_depth=None, wall=None) -> None
def get_collector() -> TimingCollector;  def reset() -> None
def process_elapsed_s() -> float | None
```

**Contract.** Pure instrumentation: changes nothing but the log, safe at
import (no jax). Sections nest via a thread-local stack;
`record` is the escape hatch for a prologue that cannot take a `with`
block. Services on this page open their own sections
(`collective_warmup`, `{label}_k` / `{label}_gather` in
`gather_k_blocks`) so drivers get them for free.

**Level / deps.** L3 leaf (stdlib only).

---

## Adding a service

The mechanical bar, condensed from the pages above: one module owns the
pattern; a typed `Gate` (or explicit parameters) instead of loose env
reads; refusals at resolve time where the fact is known there, trace time
where it is not; every announcement grep-able and rank-disciplined; the
dial in `ffi.ffi_dial_key()` if factory-time; a deep-dive doc when the
design encodes measurements. `tests/test_layering.py` and
`tests/test_runtime_startup_report.py` enforce the level and the
announcement halves mechanically.

The design bar is the four properties in
[§What a service is](#what-is-a-service), and one question they do not
answer on their own: **does the caller pick the backend?** Answer it
explicitly and record the reason, because both answers are in the tree and
both are right ([above](#choice)). Default to *no*: the burden is on the
dial to justify itself. A dial whose settings are "correct" and "worse in
every measured respect" is not a choice, and each one costs a router, a
vocabulary, a deck key, a refusal per door, and — as SlabIO measured — the
possibility that an eighth door exists.

Add the row to the [inventory](#inventory), and the vendor-and-gate facts
to [`ffi_layout.md` §3](ffi_layout.md), not here.
