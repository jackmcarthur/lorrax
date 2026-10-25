# lxkit

`import lxkit` The shared foundation under every LORRAX service: device-platform detection, environment-variable dials, library probes, process-local placement, rank-order reductions, the device-memory model and the persistent-cache root.
{: .page-summary }

`lxkit` holds the rules that every part of a LORRAX workflow must apply identically, so that no driver or service keeps its own copy. It decides which native library a process belongs to (CPU, CUDA or ROCm) from the device itself, resolves the few environment-variable dials that switch a native kernel on or off, and reports why a native library is unusable in terms that say what to fix. It places replicated host tables onto the mesh without a hidden all-gather, sums floating-point partials over mesh axes in a fixed rank order so that GW and BSE results do not depend on which devices a job was given, and prices, admits and audits the device memory of every stage, from the ζ fit through $W$, Σ and the BSE kernel. [`distrib_la`](distrib_la.md) and the other services stand on it, and so does `src/`. A developer adding a stage uses it for three things in particular: `ordered_psum` in place of `lax.psum` for any floating sum across ranks, `lxkit.memory.admit` to choose between a whole-matrix route and a mesh route by free memory, and `device_put_process_local` to put a host table on the mesh.
{: .lead }

<div class="glance" markdown>

Import
:   `import lxkit`; `from lxkit import memory, legs, native_provider`

Depends on
:   the Python standard library only; JAX is imported lazily inside functions

Devices
:   CPU, CUDA, ROCm (platform policy is data in `lxkit.LEGS`)

Used by
:   every service and driver: [`distrib_la`](distrib_la.md), the native-library loaders, GW, BSE, ISDF, I/O

</div>

## The import contract {: #import }

`import lxkit` loads the standard library and nothing else. Every JAX import sits inside a function body, and the package declares zero runtime dependencies. This is enforced by a subprocess test, for one reason: `Gate.enabled()` is read as a kernel-cache key at factory time, before `jax.distributed.initialize`, and anything that touched the JAX backend at import would initialize it too early. Keep any new code in `lxkit` to the same rule.

`lxkit` carries policy, never tables. Symbol tables, library search paths and backend preference rows belong to the service that owns the library; `lxkit` holds the decision procedure they share.

## Platform key and legs {: #legs }

A *leg* is one native library that registers LORRAX's FFI targets under one XLA platform. The platform key is `"cpu"`, `"CUDA"` or `"ROCM"`, and every gate, router and preference row in LORRAX is written in it.

| name | signature | returns |
|---|---|---|
| `device_vendor` | `(device) -> str` | `"cpu"`, `"cuda"`, `"rocm"`, or the device's own platform string for an unknown vendor. Read from the device client's platform, version and device kind, never from `device.platform`, which is `"gpu"` for every GPU vendor |
| `mesh_ffi_platform` | `(mesh, platform_map=FFI_PLATFORM_MAP) -> str` | the platform key of the mesh's first device; an unknown vendor passes through unmapped and has no native target |
| `platform_from_env` | `(default="CUDA") -> str` | the platform key from the first entry of `JAX_PLATFORMS`, without touching the backend; for code that runs before `jax.distributed.initialize` |
| `FFI_PLATFORM_MAP` | `Mapping[str, str]` | vendor to platform key, `{"cpu": "cpu", "cuda": "CUDA", "rocm": "ROCM"}` |
| `LEGS` | `Mapping[str, PlatformLeg]` | every leg, by platform key |
| `PlatformLeg` | frozen dataclass | one leg's facts: library file and its path variable (`so_name`, `so_env`), build directory, ABI symbol, the `JAX_PLATFORMS` names and device nodes that select it, XLA flag and allocator defaults, the memory model's `outside_allowance`, `memory_reserve`, `node_devices` and `distributed_context`, collective library and measured vendor workspace rows |
| `leg_for_vendor` | `(vendor) -> PlatformLeg | None` | the leg of a `device_vendor` string |
| `leg_for_platform` | `(platform) -> PlatformLeg` | the leg of a platform key, else the host leg |
| `leg_for_jax_platform` | `(name) -> PlatformLeg | None` | the device leg a `JAX_PLATFORMS` entry selects |
| `accelerator_legs` | `() -> tuple` | the device legs, in `LEGS` order |
| `leg_device_present` | `(leg, *, environ=None, exists=None) -> bool` | a device of the leg is visible (a device node exists and no visibility variable such as `CUDA_VISIBLE_DEVICES` is set empty); never imports JAX |
| `process_accelerator_leg` | `(*, environ=None, exists=None) -> PlatformLeg | None` | the device leg this process can host, decided before any JAX backend exists |
| `accelerator_leg_to_open_first` | `(platform, opened, tried, *, environ=None, exists=None) -> str | None` | the device leg a loader must open before the host leg, because device and host builds can share library sonames and the first one mapped wins |

A per-platform fact that more than one service needs is a field of `PlatformLeg`, not a branch in code. `src/ffi/fft.py` reads its leg as:

```python
leg = leg_for_vendor(device_vendor(jax.devices()[0]))
```

## Gates: environment-variable dials {: #gate }

A `Gate` is one environment-gated, rank-local native capability: a handler called inside someone else's `shard_map` that holds no communicator. Distributed solvers that hold an MPI or NCCL communicator are chosen by argument instead ([`distrib_la` backends](distrib_la.md#backends)). LORRAX's one gate today is `LORRAX_BANDS_GEMM_FFI` (vendor batched GEMM on CPU), declared in `src/ffi/gemm.py`; `src/ffi/gate.py` subclasses `lxkit.Gate` to bind LORRAX's own library probe.

```python
@dataclass(frozen=True)
class Gate:
    env: str                    # the variable, e.g. "LORRAX_BANDS_GEMM_FFI"
    target: str                 # default FFI target to probe
    platforms: tuple[str, ...]  # platform keys the dial exists on
    modes: tuple[str, ...]      # vocabulary, a subset of ("off", "on")
    default: str                # what unset or empty means
    off_label: str              # what =0 selects, for announcements
    probe: Callable[[str, str], tuple[bool, str]] | None = None
    off_policy: str = "fallback"        # or "refuse"
    silent_platform_demote: str = ""    # non-empty: the reason an out-of-scope platform is silent
    # plus message fields: label, resolved_msg, off_announce_msg, off_refuse_msg,
    # platform_demote_msg, refuse_platform_msg, refuse_probe_msg
```

| method | tier | returns | notes |
|---|---|---|---|
| `mode()` | 0, grammar | `"on"` or `"off"` | unset or empty gives `default`; an unrecognized value is announced on the rank that read it and resolves to `default`, never to `off` |
| `enabled()` | 1, environment only | `bool` | never initializes the backend; safe before `jax.distributed.initialize`; use as a kernel-cache key |
| `platform_ok(mesh)` | 2 | `bool` | the mesh platform is in `platforms` |
| `require(mesh, *, target=None, announce=True)` | 2 | platform key | mode-independent; `RuntimeError` when the platform is out of scope or the probe fails, quoting the probe's reason |
| `resolve(mesh, *, target=None)` | 2 | platform key or `None` | `None` when off or out of scope; an in-scope handler that cannot be served raises |
| `enforce(mesh, *, announce=True)` | 2, startup | platform key or `None` | called once per gate after the mesh exists: `on` requires the handler; `off` announces the fallback or raises per `off_policy` |

The spelling table is `MODE_SPELLINGS`: `off` accepts `0 off false no`, `on` accepts `1 on true yes`; `MODE_HELP` holds the help strings. There is no `auto`. Construction raises `ValueError` for a mode or default outside this vocabulary. `dial_key(*gates) -> tuple` is `((env, enabled), ...)`, the cache-key component of every factory-time dial.

Announcements follow one rule: a decision that cannot differ per rank (platform out of scope, handler resolved) prints on rank 0; one that can (a grammar error, a failed probe) prints on the rank where it happened, prefixed `[rank N]`.

| function | purpose |
|---|---|
| `rank_id() -> int | None` | the launcher rank from `SLURM_PROCID`, `PMI_RANK` or `OMPI_COMM_WORLD_RANK`, else `jax.process_index()`; `None` for a single process |
| `rank0() -> bool` | true on the rank that speaks for rank-invariant facts |
| `announce_once(key, msg, *, scope="rank0", emit=True) -> bool` | print `msg` at most once per process for `key`; `scope="local"` prints on the calling rank; `emit=False` burns the key silently. `ValueError` for an empty message or a bad scope |
| `reset_gate_state()` | forget every announcement (tests only) |

To add a dial: declare the `Gate` next to the handler's `ffi_call`, never at a consumer; give it the narrowest vocabulary; write refusal messages that name the fix; export `<dial>_enabled()` and add it to every consumer's cache key; call `enforce` at startup if the layer is required.

## Probe vocabulary {: #probe }

Every library probe in LORRAX answers in one shape so that a harness can act on it without knowing whose library it was. Two exception types separate the question every skip decision turns on:

| name | meaning |
|---|---|
| `LibraryNotBuilt(FileNotFoundError)` | no library exists for this platform on the search path. An environment fact; a test may skip |
| `LibraryUnusable(OSError)` | a library was found but cannot be used. A defect; a test must fail |
| `ProbeResult(ok, reason)` | `NamedTuple`; `reason` is never empty. Unpacks like a plain `(bool, str)` pair |
| `AVAILABLE` | `ProbeResult(True, "available")` |
| `unknown_target(target, platform, known=())` | not a target of this platform's library: a typo or a wrong-platform request; nothing to rebuild |
| `not_loadable(platform, exc, *, target="", hint="")` | loading failed (a missing dependency, a wrong `LD_LIBRARY_PATH`); says nothing about whether the handler is compiled |
| `missing_symbol(target, symbol, path, *, build_hint="")` | loaded, but the handler is absent: the only reason that means rebuild |

`distrib_la.probe_target` and `ffi.common.ffi_loader.probe_target` both return these.

## Process-local placement {: #placement }

`jax.device_put(host_array, sharding)` onto a sharding that spans several processes first all-gathers the value to assert it is equal on every process, a transient that grows with $P$. When every process already holds the same host table, these functions place only each process's own shards.

| function | purpose |
|---|---|
| `device_put_process_local(host_array, sharding, *, check=None)` | place a process-replicated host array. `host_array` must be bit-identical on every process. A tracer or an already globally sharded array takes a real `jax.device_put`. `check=True` or `LORRAX_CHECK_REPLICA=1` restores JAX's equality assertion |
| `device_put_process_tiles(shape, sharding, tile)` | the same placement without the whole host array: `tile(index)` returns the host value at one shard's index tuple; it must be a pure function of rank-invariant inputs |
| `device_put_process_local_cached(host_array, sharding)` | for small control values (at most 64 KiB each, 8 MiB in total per device): returns the array placed the first time the same value was asked for. The result is shared; never donate it |
| `placement_cache_stats() -> dict` | `{hits, misses, bypass, bytes, peak_bytes}` of the cached placement |

`src/file_io/qp_wfn.py` places a replicated band rotation for each k point this way:

```python
U = device_put_process_local(np.asarray(U_kmn[ik], dtype=np.complex128), replicated)
```

## Rank-order reductions {: #collectives }

A collective library's all-reduce adds the $P$ per-rank terms in an order taken from the physical device topology, so the same job on another device set differs in the last bit, and the ISDF fit can amplify that into visible changes in $\Sigma^c$ and quasiparticle energies. These functions move the terms with `all_to_all` / `all_gather` (bitwise, whatever the route) and add them on each device in rank order, so the result depends only on the rank count. Two ranks, one rank, and integer or boolean operands are exact anyway and go straight to the library collective.

| function | use |
|---|---|
| `ordered_psum(x, axis_name, *, slab_bytes=None)` | `lax.psum` in rank order, any pytree; inside `shard_map` |
| `ordered_psum_scatter(x, axis_name, *, scatter_dimension=0, tiled=False, slab_bytes=None)` | `lax.psum_scatter` in rank order; same arguments and result. `ValueError` when the scattered dimension does not match the axis size |
| `ordered_reduce(local_fn, *arrays)` | a global sum over sharded concrete arrays outside `shard_map`: `local_fn(*shards)` returns each rank's partial, which must sum away every sharded dimension; pass a module-level function (it is a cache key). `TypeError` on a tracer |
| `ordered_sum(x, axis=None, *, keepdims=False, mesh=None, spec=None)` | `jnp.sum` with the cross-rank part in rank order; inside a traced program pass `mesh` and the operand's `spec`, or it raises `TypeError` |
| `ordered_norm(x, axis=None, *, keepdims=False, mesh=None, spec=None)` | the 2-norm through `ordered_sum` |
| `ordered_transient_bytes() -> int` | the fixed extra device memory these reductions may hold (two slabs); the memory model reserves it |
| `install_reduction_guard(mode="log", dump_path=None)` | wrap `lax.psum`, `psum_scatter` and `pmean` so a raw call on a floating operand over more than two ranks is recorded (`"log"`) or refused (`"raise"`) |
| `install_reduction_guard_from_env()` | install it from `LORRAX_REDUCTION_GUARD`; the runtime does this before the drivers import |
| `reduction_trace() -> list` | the records the guard collected |

Operands above the fixed slab size of 8 MiB are reduced in slabs inside a `fori_loop`; slabbed and one-shot results are bitwise equal. Every floating sum over a mesh axis in LORRAX must use these; a lint enforces it. A raw `lax.psum` is allowed only where the sum is exact at any rank count, marked on the same line with `# exact-reduce: <reason>`. The BSE ring exchange in `src/bse/bse_ring_comm.py` finishes its contraction this way:

```python
U = ordered_psum(U_partial, axis_name="y")   # (batch, mu_local_x, nk), sum over y in rank order
```

`mark_varying(x, axes)` declares a loop carry built from zeros as device-varying over the mesh `axes` inside `shard_map` (`lax.pcast(x, axes, to="varying")`), with no communication. Mark only the axes the loop body introduces; a marked carry that leaves through a replicated `out_specs` is an error. `src/bse/bse_ring_comm.py`:

```python
R0 = mark_varying(jnp.zeros((X.shape[0], X.shape[1], nk, nspinor, nu_local), dtype=X.dtype),
                  ("x", "y"))
```

## The memory model: `lxkit.memory` {: #memory }

One implementation prices, admits and measures the device memory of every stage. A stage states its per-rank cost as a `Price`; `admit` picks the fastest route whose price fits the room; the allocator's high-water mark over the stage's timing section is then compared with the price and printed as a receipt (`memory <stage>: priced P | measured M`). Platform differences are fields of `lxkit.LEGS`.

| name | signature | meaning |
|---|---|---|
| `Term` | `Term(fixed=0.0, spread=0.0, source="", grid=None)` | bytes per rank $=$ `fixed` $+$ `spread`$/P$ ($+$ `grid(P)`), with its provenance; `.at(ranks)` evaluates it |
| `Price` | `Price(operands, xla, vendor, outside, ranks=1, label="", increment=False, whole=0)` | a stage's per-rank price as four `Term`s: the stage's own arrays, XLA temporaries, vendor-library workspace, and memory outside the allocator. `whole` is the largest matrix side a rank holds whole. `.bytes(ranks)`, `.terms(ranks)`, `.describe(ranks)`; prices add |
| `priced` | `(nbytes, source, *, term="xla", spread=0.0, ranks=1, increment=False) -> Price` | a one-term price |
| `admit` | `(stage, routes, room_bytes, *, probe=None, runs_in=None, local=False, device=True, detail=False) -> str` | the admission policy. `routes` is `((name, Price or callable[, room]), ...)`, fastest first. Returns the first route that fits; if none fits, the cheapest one runs with a warning, or with a banner and the minimum mesh size when it is over the physical limit |
| `book` | `(price, stage=None)` | book a price as part of a stage (the innermost open section, or the next to open under that name) |
| `room` | `(*, bucketed=True) -> int` | free planning memory per device: budget minus bytes in use, the minimum over processes. Every process must call it. Bucketed to whole GiB above 1 GiB |
| `resolve` | `(deck_bytes=0.0, linalg="local", *, total=None, outside=None, agree=None) -> float` | sets the run budget per device once at startup; every process enters |
| `run` | `() -> dict` | the run's figures: limit, budget, reserve, `linalg`, devices per node, platform |
| `vendor` | `(leg, lib, op, n, ranks, itemsize, unit=None) -> Term` | a vendor workspace from the leg's measured row, for a library with no size query |
| `unpriced` | `(what, then="priced 0")` | declare a figure nothing can price; `GATE memory_unpriced` under `LORRAX_MEMORY_STRICT=1`, else one warning |
| `record_route`, `routes` | `(site, route, reason, *, warning=None)`; `() -> list` | record and list route decisions for the run log |
| `sizing` | `(stages=None, *, limit_bytes=None, per_node=None, platform=None) -> dict` | the smallest square $P$ at which every priced stage fits, and the binding stage |
| `bucket`, `reset` | | GiB bucketing; forget the run's records |

A stage with a whole-matrix route and a mesh route prices both and lets `admit` choose. `src/isdf/core.py` does this for the transverse ζ factor, adding the measured LU workspace from [`distrib_la`](distrib_la.md#workspace) as a vendor term:

```python
return memory.admit(f"transverse zeta factor {int(nq)} x {int(n_rmu)}^2", (
    ("local", local), ("mesh", mesh_price)),
    int(shared_device_room("transverse zeta factor")))
```

where `mesh_price()` returns `compiled_price(...) + memory.Price(vendor=memory.Term(float(vendor), source=...), ranks=ndev)`. A route whose price holds a matrix of side at least $N_\mu/2$ whole on one rank must come with a distributed alternative, or `admit` raises `GATE nmu2_no_backup`.

## Native-provider policy {: #native-provider }

`lxkit.native_provider` is the decision procedure that both LORRAX's FFI loader and `distrib_la.loader` use to select, open and attest a native library. Each leg's build records its full dependency closure (every library file by path, size and SHA-256). At startup, `open_and_attest(path, *, platform, expected_abi, abi_symbols, build_hint, strict_unstamped=False, ...)` checks that record against the files, opens the recorded dependencies by exact path, opens the leg, checks its ABI stamp, and then verifies from `/proc/self/maps` that each recorded library family is mapped exactly once. A missing or stale record, a wrong file or a second copy of a family refuses with `GATE native-closure`, naming the file and the remedy. `locate_library(platform, *, specs, candidates, expected_abi, ...)` selects the leg from its path variable or the build tree before any `dlopen`. `native_closure_identity(platform=None)` is the identity of the enforced closure, a component of every persistent cache key that depends on native code. `python -m lxkit.closure verify <leg.so>` checks a recorded closure without loading anything. A new service that ships its own native library should call these rather than `ctypes.CDLL` directly.

## Persistent-cache root {: #cache }

| function | purpose |
|---|---|
| `user_cache_dir(name) -> Path` | `$SCRATCH/.cache/lorrax/<name>`, or `~/.cache/lorrax/<name>` where `SCRATCH` is unset. `name` must be one path component (`ValueError` otherwise) |
| `atomic_write_bytes(path, data) -> Path` | publish `data` whole or not at all: write a temporary unique per host, process and call, fsync, rename over the target. Readers see the old file or the new one, never a partial one, on a shared parallel file system |

The XLA compile cache, the k-convolution kernel cache (`src/ffi/fft.py`: `user_cache_dir("kconv_mathdx")`) and the vendor-kernel stores each take one subdirectory and keep their own key and pruning rules.

## Other modules {: #other-modules }

These are not re-exported from `lxkit` and are imported by path when needed.

| module | purpose |
|---|---|
| `lxkit.testing` | the shared pytest harness, installed as a `pytest11` plugin: an autouse fixture that resets announcement state, `absent_or_broken` (skip on `LibraryNotBuilt`, fail on `LibraryUnusable`), device and machine requirements. The one module that imports pytest |
| `lxkit.deck_doctor` | a login-node preflight of one GW input file, run by the `lx` launcher in a one-rank step |
| `lxkit.launcher_policy` | the `lx` launcher's pure policy: allocation pins, per-node GPU geometry, checkout selection |
| `lxkit.memory_residuals` | `python -m lxkit.memory_residuals trace.json`: the per-section residual table of a `LORRAX_MEMORY_TRACE` |
| `lxkit.closure` | `python -m lxkit.closure verify|identity <leg.so>` |

## Environment variables {: #env }

| variable | read by | effect |
|---|---|---|
| `JAX_PLATFORMS` | `platform_from_env`, `process_accelerator_leg` | first entry selects the platform before the backend exists |
| `SLURM_PROCID`, `PMI_RANK`, `OMPI_COMM_WORLD_RANK` | `rank_id` | the launcher rank, read before `jax.process_index()` |
| `LORRAX_BANDS_GEMM_FFI` | the CPU GEMM `Gate` | `0` runs the XLA dot fallback, announced |
| `LORRAX_CHECK_REPLICA=1` | `device_put_process_local` | restore JAX's cross-process equality assertion (and its all-gather) |
| `LORRAX_REDUCTION_GUARD=log|raise` | `install_reduction_guard_from_env` | record or refuse raw floating `psum`s (the records are read with `reduction_trace()`) |

| `LORRAX_MEMORY_RECEIPTS=calibrate` | `lxkit.memory` | wait on priced stages and read the allocator in every section, for exact receipts |
| `LORRAX_MEMORY_STRICT=1` | `lxkit.memory` | an unpriced figure raises `GATE memory_unpriced`; a price without provenance is a finding |
| `LORRAX_MEMORY_TRACE=<path>` | `lxkit.memory` | one JSON line per closed section: live arrays, allocator readings, executables |
| `SCRATCH` | `user_cache_dir` | root of the persistent caches |

## Invariants and gotchas {: #gotchas }

- **Never import JAX at module scope in `lxkit`.** The stdlib-only import is a tested property that `Gate.enabled()` depends on.
- **Platform from the device, not the string.** `device.platform` is `"gpu"` on both NVIDIA and AMD; use `device_vendor` or `mesh_ffi_platform`.
- **A grammar error never turns a dial off.** A misspelled value resolves to the gate's default, so a typo cannot disable a required path.
- **Explicit requests are never downgraded.** `Gate.require` raises; only `resolve` returns `None` for an out-of-scope platform, and it says so unless the gate records why it is silent.
- **Absent is not broken.** Catch `LibraryNotBuilt` to skip; let `LibraryUnusable` fail.
- **Replicated means bit-identical.** `device_put_process_local` trusts that every process passes the same bytes; with the check off, a mismatch is not detected.
- **Use ordered reductions for floating sums.** A raw `lax.psum` over more than two ranks needs an `# exact-reduce:` justification, and `jnp.sum` over a sharded axis hides a topology-ordered all-reduce; use `ordered_sum` or `ordered_reduce`.
- **`memory.room()` and `memory.resolve()` are collective.** Every process must call them, in the same order.
- **Price with provenance.** Every nonzero `Term` should carry a `source`; under `LORRAX_MEMORY_STRICT=1` one without is a finding.
- **Two `dial_key`s.** `lxkit.dial_key(*gates)` keys on the gates passed to it; `distrib_la.dial_key()` takes no arguments and covers `distrib_la`'s own factory-time settings. They are different functions.

## Where LORRAX uses it {: #consumers }

| file | purpose |
|---|---|
| `services/distrib_la/src/distrib_la/loader.py`, `src/ffi/common/ffi_loader.py` | `native_provider.locate_library`, `open_and_attest`, `legs`, probe vocabulary |
| `services/distrib_la/src/distrib_la/resolve.py`, `subspace.py` | `device_vendor`, `leg_for_vendor`, `FFI_PLATFORM_MAP` for the backend platform key |
| `src/ffi/gate.py`, `src/ffi/gemm.py` | the LORRAX `Gate` subclass and the `LORRAX_BANDS_GEMM_FFI` dial |
| `src/ffi/fft.py` | leg lookup and `user_cache_dir` for the k-convolution kernels |
| `src/common/gpu_utils.py` | `memory.resolve` and `memory.room`: the run budget and the free room |
| `src/runtime/aot_memory.py` | `memory.admit` of compiled program chunks |
| `src/isdf/core.py`, `src/isdf/zeta_mubatch.py` | `memory.admit` and `memory.book` for the ζ factor and the ζ fit G tiles; `ordered_psum` in the fit |
| `src/bandstructure/fh_interp.py` | `memory.admit` for H-transform interpolation stages |
| `src/bse/bse_ring_comm.py`, `src/bse/bse_stack_matvec.py` | `ordered_psum`, `mark_varying` in the BSE kernel matrix-vector products |
| `src/gw/qsgw_density.py`, `src/gw/head_correction.py`, `src/gw/cd/sigma.py` | `ordered_psum` in QSGW density, head and contour-deformation Σ sums |
| `src/file_io/qp_wfn.py`, `src/isdf/galerkin.py`, `src/bse/bse_nontda.py` | `device_put_process_local` for replicated host tables |
| `src/file_io/parallel_transport.py` | `ordered_sum` in link-quality norms |
| `src/runtime/vendor_kernels.py` | `user_cache_dir` and `LEGS` for vendor kernel stores |
