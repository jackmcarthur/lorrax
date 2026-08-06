# JAX ahead-of-time lowering: agent reference

JAX has a real staged compiler pipeline — Python → jaxpr → StableHLO/HLO → compiled executable → execution — and you can interrogate each stage independently, before running anything, using only abstract argument descriptions. Most performance questions that feel like they need a profiler have cleaner answers earlier in the pipeline.

## The pipeline and its objects

`jax.jit(f)` exposes `.trace(*args)`, `.lower(*args)` (equivalent to `.trace(*args).lower()`), and `.lower(...)` exposes `.compile()`:

```python
wrapped  = jax.jit(f, **jit_kwargs)
traced   = wrapped.trace(*specs)       # → jax.stages.Traced
lowered  = traced.lower()              # → jax.stages.Lowered
compiled = lowered.compile()           # → jax.stages.Compiled
```

**Match questions to the right object.** Jaxpr answers "what program did JAX stage?" Lowered IR answers "what does the compiler see — ops, shardings, collectives?" Compiled analyses answer "what does the executable estimate for memory and cost?" Runtime profiling answers "what actually happened on-device?" Asking the wrong stage is the most common failure mode — reading jaxpr to learn allocator behavior, or staring at runtime traces to figure out whether the compiler inserted collectives.

## Abstract arguments: `ShapeDtypeStruct`

`jax.ShapeDtypeStruct(shape, dtype)` carries shape and dtype without allocating storage, so you can lower and compile for shapes that would be impossible to instantiate. Sharding can be attached: `jax.ShapeDtypeStruct(shape, dtype, sharding=s)`.

```python
spec = jax.ShapeDtypeStruct((N, M), jnp.complex64)
lowered = jax.jit(f).lower(spec)
```

**Static arguments are the exception.** If `chunk_size` is in `static_argnums`, you must pass the actual integer — it selects *which version of the program* JAX stages, not the tensor signature. Each distinct static value is a separate trace + lower + compile. If chunk size only changes array shapes (not a static arg), re-tracing is skipped and only lowering + compilation recur.

## Stage 1: Jaxpr

Jaxpr is the cheapest readable summary of staged program structure. It shows primitives, loop/conditional outlines, how transformation combinators (`vmap`, `scan`, `cond`, `checkpoint`) rewrote the body, and the survival or disappearance of Python structure. It will *not* tell you allocator lifetimes, communication overlap, or the final buffer plan.

```python
# Quick one-off, no jit wrapper needed:
jaxpr = jax.make_jaxpr(f)(*args)

# Or from the staged pipeline:
print(traced.jaxpr)
```

`jax.make_jaxpr` is often more convenient for quick structural checks when you don't plan to lower. It accepts the same abstract specs.

**When jaxpr is the answer:** a stray static value caused unexpected specialization; a captured closure constant changed the program; a `scan` carry size exploded; a `vmap` distributed wrong; a `cond` didn't remain local. These show up more clearly in jaxpr than anywhere else.

## Stage 2: Lowered IR

```python
stablehlo_text = lowered.as_text("stablehlo", debug_info=True)
hlo_text       = lowered.as_text("hlo", debug_info=True)
```

StableHLO is generally more legible; HLO is more backend-specific. Always pass `debug_info=True`.

The primary use of lowered IR is *structural discrimination*: does this function trigger sharding-induced collectives? Is an intermediate replicated when it should be partitioned? Does a rewrite actually lower differently?

**What to grep for in StableHLO.** The collective ops are `stablehlo.all_gather`, `stablehlo.all_reduce`, `stablehlo.reduce_scatter`, `stablehlo.all_to_all`, and `stablehlo.collective_permute`. Vendor library calls (cuBLAS GEMMs, cuFFT, cuDNN) hide inside `stablehlo.custom_call` with target-specific names like `__cublas$gemm` or `__cudnn$fused` — these are often responsible for mysterious scratch memory that `memory_analysis()` doesn't fully account for.

### Sharding debugging ladder

**1. Declared contract.** Read `in_shardings`, `out_shardings`, and any `jax.lax.with_sharding_constraint` calls inside the function. This tells you what you *asked* for.

**2. Intermediate sharding probes.** `jax.debug.inspect_array_sharding` fires its callback at compile time, not runtime, and works from abstract specs alone:

```python
def f(x):
    y = heavy_subcomputation(x)
    jax.debug.inspect_array_sharding(y, callback=lambda sh: print("y:", sh))
    return next_stage(y)

jax.jit(f, in_shardings=..., out_shardings=...).lower(spec_x).compile()
```

For visual inspection of concrete sharded arrays outside JIT, `jax.debug.visualize_array_sharding(arr)` prints an ASCII device grid (≤2D only). For higher-rank arrays, `.sharding` on any `jax.Array` gives the `Sharding` object directly.

**3. IR inspection.** Now grep for the collective ops listed above, suspicious transposes/reshapes around partition boundaries, and evidence that a local-looking operation induced cross-device traffic.

## Stage 3: Compiled analyses

```python
compiled = lowered.compile()
m = compiled.memory_analysis()      # may return None on some backends
c = compiled.cost_analysis()        # dict-like, e.g. c['flops']
```

The standard memory estimate:

```python
total = (
    m.temp_size_in_bytes
    + m.argument_size_in_bytes
    + m.output_size_in_bytes
    - m.alias_size_in_bytes
)
```

This is a compiler-side estimate, not allocator truth. It does not account for vendor library workspaces, runtime communication buffers, or allocator fragmentation. It is excellent for comparisons between candidates and for rejecting impossible shapes. If both candidates are within a few percent of the memory limit, runtime validation is warranted.

`cost_analysis()` returns a dict whose structure is not stable across versions/backends but typically includes `'flops'`.

### Buffer donation

`donate_argnums` / `donate_argnames` on `jax.jit` tells the compiler it may reuse input buffers for outputs, which directly increases `alias_size_in_bytes` and reduces the effective total. This matters when pushing memory limits:

```python
compiled = jax.jit(f, donate_argnums=(0,)).lower(spec).compile()
m = compiled.memory_analysis()
# m.alias_size_in_bytes will reflect the donated input buffer
```

If donation is rejected (e.g. the input is still live after the call, or shapes don't match an output), JAX silently falls back — the memory analysis will show the difference. Donation also implies that the donated array becomes invalid after the call.

### `compiler_options`

Per-function XLA flags without environment variables:

```python
jax.jit(f, compiler_options={"xla_gpu_memory_limit_slop_factor": 150})
```

Useful for per-function memory tuning, forcing or disabling specific XLA passes, etc. The available keys are backend-specific and not formally documented — check XLA source or `--xla_dump_to` output for flag names.

## Shape sweeps for chunk-size tuning

A sweep maps candidate chunk sizes to abstract specs, lowers, compiles, and records `memory_analysis()`. This includes the compiler's view of temporaries and buffer reuse, which raw tensor-volume arithmetic misses.

```python
wrapped = jax.jit(f, **jit_kw)
rows = []
for c in candidates:
    compiled = wrapped.lower(*make_specs(c)).compile()
    m = compiled.memory_analysis()
    rows.append((c, m.temp_size_in_bytes, m.argument_size_in_bytes,
                  m.output_size_in_bytes, m.alias_size_in_bytes))
```

Note: wrapping once outside the loop is correct when only shapes change (not static args). Each shape variant still requires a separate lower + compile, but re-tracing is avoided.

**Compilation is not free.** For large programs, XLA compilation can take minutes per variant. A sweep of 50 chunk sizes can easily take hours. Use `jax.eval_shape(f, *specs)` and raw shape arithmetic to narrow the interval first, compile a coarse grid, then refine near the frontier. Persistent compilation caching (`jax_compilation_cache_dir`) helps across runs but not within a single sweep of novel shapes.

**`jax.eval_shape`** does shape inference with no FLOPs and returns `ShapeDtypeStruct`s. Perfect for "does this return the right shape?" Not useful for temporaries, collectives, or rematerialization.

## Rematerialization / checkpointing

Two levels. At the AD level, `jax.ad_checkpoint.print_saved_residuals(f, *args)` shows what residuals are saved vs. recomputed. At the compiler level, lower and compile variants with different `jax.checkpoint` / `jax.remat` policies and compare `memory_analysis()` and `cost_analysis()`.

## Compilation debugging

Repeated recompilation and cache misses are questions about the specialization structure of the staged program.

`jax.log_compiles(True)` logs every compilation event. `jax.config.update("jax_explain_cache_misses", True)` (or `JAX_EXPLAIN_CACHE_MISSES=1`) explains why. Persistent compilation caching: `jax.config.update("jax_compilation_cache_dir", "/path")`.

## Multi-host / multi-process

On multi-node setups (e.g. Perlmutter), `jax.distributed.initialize()` must be called before any JAX operations. After initialization, `jax.devices()` returns the full global device list across hosts, and shardings operate on global shapes. The local vs. global shape distinction matters: a `jax.Array` sharded across 4 nodes has a global shape, but each host holds only its local shards, accessible via `arr.addressable_shards`.

AOT lowering works the same way in multi-process — `ShapeDtypeStruct` specs use global shapes, and the lowered IR reflects the full distributed program including inter-host collectives. `inspect_array_sharding` callbacks will fire on every host, showing the global sharding.

For debugging multi-host collective hangs: `NCCL_DEBUG=INFO` (or `NCCL_DEBUG=TRACE`) surfaces NCCL-level communication events. `NCCL_ASYNC_ERROR_HANDLING=1` converts hangs into errors.

## Shape polymorphism via `jax.export`

`jax.export.export(jax.jit(f))(*symbolic_specs)` produces a single lowered artifact parameterized by symbolic dimension variables, avoiding repeated tracing/lowering across a shape family:

```python
from jax import export
spec = jax.ShapeDtypeStruct(export.symbolic_shape("batch, N"), jnp.float32)
exported = export.export(jax.jit(f))(spec)
print(exported.mlir_module())
```

Useful for structural reasoning across shape families. Does not abolish concrete compilation — exact memory summaries still require concrete shapes.

## GPU memory allocation

JAX preallocates 75% of GPU memory on first use. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` changes the fraction. `XLA_PYTHON_CLIENT_PREALLOCATE=false` disables it. `XLA_PYTHON_CLIENT_ALLOCATOR=platform` gives minimal on-demand allocation (slow, useful for debugging exact consumption).

## Deep compiler debugging: HLO dumps

When `memory_analysis()` says a candidate is large and you need to know *why*:

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_dumps" python my_script.py
```

Produces `after_optimizations-buffer-assignment.txt` with compile-time buffer reuse, live ranges, and peak-memory structure. Heavy artillery — don't reach for it if earlier evidence settles the issue.

## Runtime tools (escalation only)

`jax.profiler.save_device_memory_profile("snapshot.prof")` captures a device memory snapshot; diff two with pprof's `--diff_base` for leak tracking. XProf / TensorBoard `memory_viewer` tab shows device memory over program order (static compiler info, not wall time). Vendor profilers (nsys, ncu, rocprof) for hardware-level questions.

If compile-time and runtime memory disagree, the diagnostic question is "what runtime phenomenon explains the gap?" — vendor scratch, communication buffers, host transfers — not "which view is fake."

## Decomposition and metadata

If a pipeline is too large, lower subfunctions independently with their own abstract specs. Tag major regions with `jax.named_scope("pair_density_build")` — named regions make lowered text searchable, diffable, and connectable to profiler output.

## Evidence hierarchy

From cheapest/highest-signal to most expensive:

1. `memory_analysis()` + `cost_analysis()` on the relevant subfunction
2. Intermediate sharding callbacks via `inspect_array_sharding`
3. Jaxpr of the relevant subfunction
4. StableHLO of the relevant subfunction
5. HLO pass dumps / buffer-assignment files
6. Runtime memory profiles, XProf, vendor profilers

## Compact idioms

```python
spec = jax.ShapeDtypeStruct(shape, dtype)

traced   = jax.jit(f, **kw).trace(*specs)
lowered  = traced.lower()
compiled = lowered.compile()

m = compiled.memory_analysis()
total = m.temp_size_in_bytes + m.argument_size_in_bytes + m.output_size_in_bytes - m.alias_size_in_bytes

c = compiled.cost_analysis()                           # e.g. c['flops']
lowered.as_text("stablehlo", debug_info=True)
lowered.as_text("hlo", debug_info=True)
jax.make_jaxpr(f)(*specs)                              # quick jaxpr, no jit needed
jax.eval_shape(f, *specs)                              # shape inference only
jax.debug.inspect_array_sharding(y, callback=print)    # fires at compile time
jax.ad_checkpoint.print_saved_residuals(f, *args)       # AD residual inspection
```

## Reusable inspection helper

```python
def inspect_bundle(f, *specs, **jit_kwargs):
    wrapped  = jax.jit(f, **jit_kwargs)
    traced   = wrapped.trace(*specs)
    lowered  = traced.lower()
    compiled = lowered.compile()
    return {
        "jaxpr":     traced.jaxpr,
        "stablehlo": lowered.as_text("stablehlo", debug_info=True),
        "hlo":       lowered.as_text("hlo", debug_info=True),
        "memory":    compiled.memory_analysis(),
        "cost":      compiled.cost_analysis(),
    }
```

## Mistakes to watch for

- The compiled object is specialized to the exact signature used at lowering. Different static value or different shape → different program.
- `memory_analysis()` and `cost_analysis()` are debugging interfaces, not stable API contracts.
- Reading anonymous HLO without `debug_info=True` or sharding probes is a waste of time.
- Demanding allocator-level evidence from jaxpr, or tracing-level evidence from compiled analyses.
- Forgetting that `custom_call` ops in lowered IR represent vendor library calls with their own scratch memory outside the compiler's accounting.