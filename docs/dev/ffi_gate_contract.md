# The FFI env-gate contract (`ffi.gate.Gate`)

One resolver for every env-gated, rank-local FFI capability: a handler called
inside somebody else's `shard_map`, holding no communicator. Source:
`src/ffi/gate.py`. The FFI layer is required
([decisions](../architecture/decisions.md)): required-layer gates default on,
and a missing library refuses at startup.

| gate | declared in | platforms | default | `=0` (`off_policy`) |
|---|---|---|---|---|
| `LORRAX_BANDS_GEMM_FFI` | `ffi/gemm.py` | cpu | `on` | `fallback`: announced, uncertified XLA einsum arm, retained because `extra="minor"` cannot ride a batched GEMM ([vendor GEMM](vendor_gemm_service.md)) |
| `LORRAX_FFT_FFI` | `ffi/fft.py` | cpu | `on` | `refuse`: the XLA flat-k arm is deleted |

On CUDA neither dial exists (`silent_platform_demote`): XLA's dot already
calls cuBLAS, and the flat-k transform and k-convolutions belong to the
k-convolution router. The router is not a `Gate`: `ffi.fft.kconv_backend(mesh)`
returns `mathdx` on CUDA and `plan` on cpu and refuses any other platform, and
`require_kconv(mesh)` checks the nvidia-mathdx wheel and its targets (CUDA) or
the flat-k plan target (cpu) at startup ([FFI layout](../architecture/ffi_layout.md)).
Every operation's engines, gate and code: the
[kernel operations](../architecture/ffi_layout.md#kernel-operations) table;
every target: the [kernel catalog](../architecture/ffi_layout.md#kernel-catalog).

## API

```python
GATE.mode()          -> "on" | "off" | "auto"   tier 0: grammar
GATE.enabled()       -> bool                    tier 1: lexical, no backend init
GATE.platform_ok(m)  -> bool                    does this mesh's platform have a backend
GATE.require(m)      -> platform                tier 2: announce or refuse
GATE.resolve(m)      -> platform | None         tier 2, mode-aware
GATE.enforce(m)      -> platform | None         tier 2, startup
```

### Grammar

| mode | spellings |
|---|---|
| `off` | `0` `off` `false` `no` |
| `on` | `1` `on` `true` `yes` |
| `auto` | `auto` |

Unset or empty maps to the gate's `default`. Each gate declares its own
vocabulary; both current gates accept only `off`/`on`, so `=auto` is a grammar
error there. A value outside the vocabulary is announced once, on the rank
that read it, and resolves to the **default**: with `off` able to refuse, a
typo resolving to `off` would kill a run. `Gate.__post_init__` refuses a
declared mode with no resolver branch, and refuses `auto` without an
`auto_capability` string naming the capability test. `auto` is admitted only
for an optional accelerator whose off path is a certified reference.

### Two tiers

| | tier 1 `enabled()` | tier 2 `require` / `resolve` / `enforce` |
|---|---|---|
| reads | the env var only | the live `mesh.devices` |
| initialises the JAX backend | no | already initialised (it has a mesh) |
| callable before `jax.distributed.initialize` | yes, and must be | no |
| used for | kernel-cache keys at factory time | the lowering decision and refusals |

Consumers key kernel caches on `enabled()` (`gw.ppm_tau_kernel`'s
`pipeline_key`/`cache_key`), and those factories can run before the
distributed runtime exists, so tier 1 may not probe or read a platform.
`enforce(mesh)` runs once per gate from `runtime.initialize_communicator_stack`
right after the mesh exists, so a missing library refuses at startup, not at
the first kernel factory mid-run.

### Refusals

`require(mesh)` raises `RuntimeError` when the mesh platform is not in
`platforms` (naming the platform and why the dial is scoped that way), or when
`ffi_loader.probe_target` finds the handler unusable. The probe's reason is
quoted verbatim: *unknown target*, *library could not be loaded*, and *loaded
but does not export* have three different fixes, and only the last means
rebuild. `require()` is mode-independent: it answers whether this mesh can
serve this handler, which is also what a directly constructed wrapper asks.

Refusal is two-phase. Platform and handler are checked at resolve time;
operand dtype, rank and extent are trace-time facts and are refused in the
wrapper body (`ffi.fft.make_flat_k_fft_ffi`'s traced wrapper,
`contract_bands._ffi_dtypes_ok`).

### Who announces

One question decides: can the decision differ per rank?

| decision | per-rank? | speaks |
|---|---|---|
| platform out of scope, handler resolved | no | rank 0 (`scope="rank0"`) |
| env grammar error, failed probe | yes (env and `LD_LIBRARY_PATH` are per-process) | the rank it happened on (`scope="local"`), `[rank N]`-prefixed on ranks ≥ 1 |

Silence is allowed only when declared: `silent_platform_demote` is the recorded
reason, not a boolean.

`gate.rank_id()` reads the launcher's rank, `SLURM_PROCID` → `PMI_RANK` →
`OMPI_COMM_WORLD_RANK` (the same order as the C++ `announce_here()`), and
falls back to `jax.process_index()` only when none is set. It never calls
`jax.process_index()` first: that initialises the XLA backend and would break
tier 1.

## Adding a dial

1. Declare the `Gate` in the FFI subpackage that owns the handler, next to its
   `ffi_call`, never at a consumer.
2. Give it the narrowest honest vocabulary. `auto` only with a named
   capability test and a certified off path.
3. Write the messages: harnesses grep the announce lines, and a refusal must
   name the fix ([quality patterns](QUALITY_PATTERNS.md) #6).
4. Export `<dial>_enabled()` and add it to every consumer's kernel cache key.
5. Wire `enforce` into startup if the layer is required.
6. Gate it: grammar, announce strings, refusal texts, and that an off dial
   loads no library, each with a deliberately broken twin that must fail; a
   cell whose twin passes is void, not green. Run on both platforms.

## What stays outside `Gate`

`distrib_la.resolve` ([distrib_la](../services/distrib_la.md)) serves
distributed solvers that hold an MPI/BLACS/NCCL communicator. Its selection
input is a function argument (`plan(op, mesh, backend=, n=)`), it resolves in
one phase because it receives `n`, and its process-coverage and mesh-geometry
guards would be permanent no-ops for a rank-local handler. The split is by
what the backend is: `plan` for communicator-holding solvers, `Gate` for
rank-local dials. The C++ handlers share no base class; the shared resolver is
Python policy only.
