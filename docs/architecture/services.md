# Substrate services — the API reference

*The L3 substrate ([layers](layers.md)) as a set of internal services: for
each one, its purpose, its **public API as it exists in the code today**,
its contract (what it refuses, what it announces, and **when** — resolve
time vs trace time), its level and allowed import direction, and its
dependencies. Deep-dive documents are linked; signatures here are verified
against `src/` (2026-08-01) and win over any older doc prose.*

Conventions used throughout:

* **Level / imports.** Imports run downhill only, L1 → L2 → L3
  ([layers](layers.md)). Every service on this page is L3 unless marked;
  an L3 module imports nothing above L3.
* **Resolve time vs trace time.** *Resolve-time* checks fire when a
  factory/plan is built (platform, mesh geometry, handler probe, env
  grammar); *trace-time* checks fire inside the returned callable on the
  operand (dtype, rank, extents) — a factory cannot know them earlier.
* **Announce-or-refuse.** `auto` may demote but must say so from the rank
  it happened on; an explicit request that cannot be honored raises with
  the reason and the fix. Silence is legal only where declared.

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
    │ file_io.slab_io ─ sharded-slab HDF5 transport                │
    │   └─ tiers: _slab_io_ffi (ffi.phdf5) → _slab_io_mpi_host     │
    │             (mpi4py+h5py) → _slab_io_allgather               │
    │ common.timing ─ instrumentation (leaf; no deps)              │
    └──────────────────────────────────────────────────────────────┘

sanctioned upward edges (tests/test_layering.py, argued in layers.md §5):
 !R1  file_io.slab_io ──▶ gw.gw_config          (lazy; SlabIOBackend enum)
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
  jit; with no mesh given it returns the **canonical mesh** — the
  most-square factorization of `jax.devices()`, built ONCE per process per
  axis-name tuple and cached (`_CANONICAL_MESHES`, 2026-08-01): a library
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
`_q_batch`). *(SUPERSEDED 2026-07-31: this section used to name
`common.zeta_projection`'s basis change as a consumer — false;
`zeta_projection` only cites the doctrine, it never calls the factory.)*

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

**Public API** (`src/ffi/gate.py`; `ffi.common.gate` is a shim):

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
MODE_SPELLINGS = {"auto": ("auto",), "off": ("0","off","false","no"),
                  "on": ("1","on","true","yes")}

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
GATE       = Gate(env="LORRAX_FFT_FFI",       modes=("off","on"), default="off",
                  platforms=("cpu","CUDA"), target=FLAT_K_TARGET)
FUSED_GATE = Gate(env="LORRAX_FFT_FFI_FUSED", modes=("off","on"), default="off",
                  platforms=("cpu","CUDA"), target=GW_CONV_TARGET)

def fft_ffi_enabled() -> bool;   def fft_ffi_mode() -> str
def fused_fft_ffi_enabled() -> bool;  def fused_fft_ffi_mode() -> str
def require_fft_ffi(mesh: Mesh, target: str = FLAT_K_TARGET) -> str
def make_flat_k_fft_ffi(mesh, kgrid, spec, *, kind, norm, out_spec) -> Callable
def make_gw_conv_ffi(mesh, kgrid, g_spec, v_spec, *,
                     norm='ortho', mult=1.0) -> Callable
def ffi_fft_scale(kind, norm, nk) -> float
def validate_flat_spec(spec: P, what: str) -> P

# the un-gated XLA entry points, common/fft_helpers.py:
def make_flat_k_fft(mesh, kgrid, spec, *, kind, norm='ortho', out_spec=None)
def make_flat_k_ifftn(mesh, kgrid, spec, *, norm='ortho', out_spec=None)
def make_flat_k_fftn(mesh, kgrid, spec, *, norm='ortho', out_spec=None)
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
GATE = Gate(env="LORRAX_BANDS_GEMM_FFI", modes=("auto","off","on"),
            default="auto", platforms=("cpu",), target=GEMM_TARGET)

def gemm_ffi_enabled() -> bool;  def gemm_ffi_mode() -> str
def require_gemm_ffi(mesh) -> str
def gemm_batch(a3, b3)     # A (BA,M,K) @ B (BB,K,N) -> C (BA,M,N), C[i]=A[i]@B[i%BB]
```

**Contract.** The only default-`auto` dial: AUTO-ON is announced
(`[bands_gemm] AUTO-ON …`); on GPU meshes the demotion is declared-silent
(XLA:GPU already dispatches cuBLAS — nothing to act on). Trace time:
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
exception. Backend **choice** is an input-deck key; the env grants
capability only. Backends today: ScaLAPACK `pzheevd` (the permanent CPU
distributed eigh), SLATE, cuSOLVERMp (deletion candidate,
[ffi_layout §5](ffi_layout.md)), `native`.

**Level / deps.** L3. `ffi.common.ffi_loader` + the backend packages;
consumed by `isdf/core`, `gw/w_isdf`, `bandstructure/bse_setup`,
`bse/vq_interp`.

**Deep dive.** `docs/dev/linalg_ffi.md`.

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

**Contract.** Three backend tiers, selected by
`SlabIOBackend` (defined in `gw.gw_config` — sanctioned exception R1;
`slab_io = auto` in the deck resolves via the capability router in
`LorraxConfig.from_input_file`):

| tier | module | mechanism | requires |
|---|---|---|---|
| `PHDF5_FFI` | `_slab_io_ffi` | collective MPI-IO via `ffi.phdf5` (CUDA lib or CUDA-free host lib) | mesh; the lib exports the handler |
| `PHDF5_HOST` | `_slab_io_mpi_host` | same MPI-IO, driven by mpi4py + parallel h5py | mesh; the overlay |
| `H5PY_ALLGATHER` | `_slab_io_allgather` | gather to rank 0, serial h5py | nothing (last resort; rank-0 bandwidth limit) |

Resolve time: `PHDF5_FFI`/`PHDF5_HOST` without a mesh raise; a non-enum
`backend` raises `TypeError`. Padding contract: files always store the
**logical** shape — producers pass `valid_shape` for the un-padded prefix,
consumers request a mesh-divisible physical `shape` and get a zero-filled
tail (driver-side padding is `runtime.padding`'s job). The host-MPI path
writes synchronously by design (the threaded FFI deadlocks at `H5Fclose`
under `MPI_THREAD_SINGLE` — `_slab_io_mpi_host.py` docstring).

**Level / deps.** L3. `ffi.phdf5` (lazy), mpi4py/h5py (lazy), the R1 lazy
enum import.

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

The bar, condensed from the pages above: one module owns the pattern; a
typed `Gate` (or explicit parameters) instead of loose env reads; refusals
at resolve time where the fact is known there, trace time where it is not;
every announcement grep-able and rank-disciplined; the dial in
`ffi.ffi_dial_key()` if factory-time; a deep-dive doc when the design
encodes measurements. `tests/test_layering.py` and
`tests/test_runtime_startup_report.py` enforce the level and the
announcement halves mechanically.
