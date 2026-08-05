# Is each LORRAX reach beneath JAX's public API necessary?

*Audit 2026-07-30. Read-only; nothing under `/work2/08271/jackmc/frontera/lorrax`
was modified. Every claim below is either (a) read out of the installed
jax/jaxlib 0.9.1 in `/work2/08271/jackmc/frontera/lorrax_env/.venv`, (b) read out
of upstream source at the exact XLA commit our jaxlib pins, or (c) carried from a
cited doc/issue URL. Where a claim rests on a **negative** search result that is
said so explicitly.*

---

## 0. The pin

| | |
|---|---|
| `jax` | **0.9.1** (`jax/version.py`: `_release_version = '0.9.1'`, `_git_hash = 58cb6e556c996bf0361bca9e64890a551e513280`) |
| `jaxlib` | **0.9.1** |
| `jax-cuda12-plugin` / `jax-cuda12-pjrt` | 0.9.1 |
| XLA commit pinned by jax 0.9.1 | **`3cc8846c10052cc1c32c4db87866eac4e4cdbccd`** ([`jax-v0.9.1:third_party/xla/revision.bzl`](https://github.com/jax-ml/jax/blob/jax-v0.9.1/third_party/xla/revision.bzl)) |
| Python | 3.12.13 |
| released | 0.9.1 on 2026-03-02; latest at audit time is 0.11.0 (2026-07-16) |

**Method caveat.** The venv's interpreter (`home = /usr/local/bin` in
`pyvenv.cfg`) lives inside the container, so nothing here was *executed*. Every
"jax 0.9.1 does X" claim is from reading the installed `.py` sources or from
`strings`/`nm` on the installed `.so`s. Where I use a binary result I use it only
**positively** (a symbol/string that IS present). This audit makes no claim of
the form "`strings` found nothing, therefore absent" — that error was made
earlier in this project by a `strings` census that missed a header-only C++
template (`gloo::ReduceScatterHalvingDoubling`), and the correction is recorded
in `wk_REL/docs/upstream_prior_art.md` §0.

---

## 1. Verdict table

| # | Site | span | total lines | executable lines | verdict | one line |
|---|---|---|---|---|---|---|
| 1 | `runtime/aot_memory.py` | whole file | 519 | 188 | **NO_SUPPORTED_PATH** | cuFFT plan workspace is the *only* GPU library workspace XLA leaves out of buffer assignment, and `memory_analysis()` is buffer-assignment-only by construction |
| 2 | `collectives.py` `psum_scatter_checked` &co. | 302-465 | 164 | 45 | **NO_SUPPORTED_PATH** (and mis-tiered — it is 100 % public API) | XLA:CPU exposes three collective flags, all timeouts; algorithm is hardcoded; no numeric-check knob exists off GPU |
| 3 | `collectives.py` `warm_mesh_cliques` | 466-552 | 87 | 30 | **NO_SUPPORTED_PATH** | guard confirmed in *our* binary; no public API pre-creates CPU communicators; upstream TODO b/380457503 says they want one and don't have one |
| 4 | `runtime/__init__.py` `announce_cpu_collectives` | 265-324 | 60 | 20 | **REPLACEABLE_NOW (1 line of 20)** | `jax.config.jax_cpu_collectives_implementation` is public and documented; the rest of the function has no JAX equivalent |
| 5 | `common/jax_compile_cache.py` | whole file | 1005 | 423 | **NO_SUPPORTED_PATH** for ~140 core lines; rest is LORRAX ops, not a JAX workaround | upstream supplies exactly two things (rank-0 writes, GPU-only key stripping); CPU keys are process-dependent *by design* and unchanged on `main` |
| 6 | `collectives.py` `device_put_process_local` | 147-227 | 81 | 24 | **REPLACEABLE_NOW** | `jax.make_array_from_process_local_data(sharding, arr, arr.shape)` is public, documented, in our pin, and its docstring states LORRAX's precondition verbatim |
| 7 | `collectives.py` `barrier` | 66-146 | 81 | 14 | **NO_SUPPORTED_PATH** (for the policy; the primitive is already the supported one) | `sync_global_devices` is the only barrier JAX has and LORRAX already calls it; the 14 lines are error policy JAX does not offer |

**Totals.** Executable lines removable **today**: ≈ **15** (14 at site 6, 1 at
site 4). Removable **on upgrade**: **0** — see §9, nothing in this audit is
fixed by any released jax/jaxlib through 0.11.0. Genuinely **necessary**:
≈ **417** executable lines (sites 1, 2, 3, 5-core, 7), plus ≈ 280 lines of site-5
Lustre/inode/reporting engineering that was never a JAX workaround at all.

**Two corrections to the brief's tiering.** Sites 2 and 4 do not touch any
private API — site 2 uses only `lax.psum_scatter` / `lax.psum` /
`lax.axis_index` / `lax.dynamic_slice` / `jnp` / `jax.debug.callback`, site 4
only `os.environ` and `/dev/nvidia*`. Meanwhile **site 5 is the deepest private
reach in the whole audit**: `jax_compile_cache.py` is the only file of the seven
that imports `jax._src` — six modules (`cache_key`, `compilation_cache`,
`lru_cache`, `compiler`, `config`, `distributed`, plus `jax._src.lib`) — and it
monkeypatches four upstream functions and subclasses a private class.

---

## 2. Site 6 — `device_put_process_local` → `REPLACEABLE_NOW`

**This is the one to do.**

### The premise is still true in 0.9.1

`jax/_src/dispatch.py::_device_put_sharding_impl`, lines 483-499 of the
installed source (identical to
[`jax-v0.9.1:jax/_src/dispatch.py#L483-L499`](https://github.com/jax-ml/jax/blob/jax-v0.9.1/jax/_src/dispatch.py#L483-L499)
and still on `main`):

```python
if ((x_is_jax_array and not x._committed) or
    type(x) in array_types or type(x) in dtypes.python_scalar_types):
  if xla_bridge.process_count() == len(s._internal_device_list.process_indices):
    multihost_utils.assert_equal(
        x, fail_message=(... "is not the same on each process" ...))
  return _DeferredShardArg(x, s, aval, True, copy)
```

`np.ndarray` is in `array_types`. `assert_equal` is `process_allgather(x,
tiled=True)` — a real, P-linear allgather. Landed in jax **0.4.31**, commit
[`061f4df82a`](https://github.com/jax-ml/jax/commit/061f4df82a07cb0a347be14d9f2960b1ac2c224f),
never removed (verified present at 0.4.31 → 0.11.0 → `main`), carrying an
in-code `# TODO(yashkatariya,mattjj): Move this check to jit.` that is still
there. **No config flag disables it**; **no CHANGELOG entry ever announced it**;
**it is documented nowhere** (`multi_process.md`, `distributed_data_loading.md`,
`faq.rst`, `sharded-computation.md` at 0.9.1: zero hits for `assert_equal` or
"same on each process"). No GitHub issue about it was found — *negative search
result, not proof of absence.*

So the 40-line WHY comment at `collectives.py:106-146` stays regardless. It
documents something upstream still does not.

### The replacement

```python
jax.make_array_from_process_local_data(sharding, arr, arr.shape)
```

- **Public and documented.** https://docs.jax.dev/en/latest/_autosummary/jax.make_array_from_process_local_data.html — listed in `docs/jax.rst` under the `jax.Array` section next to `make_array_from_callback` and `make_array_from_single_device_arrays`. Not experimental.
- **Present in our pin.** `jax/_src/array.py:846`, exported at `jax/__init__.py:162`. Added in **jax 0.4.29**, commit [`8f045cafd2`](https://github.com/jax-ml/jax/commit/8f045cafd2) (2024-05-15) — version established by *tag bisection*, because there is **no CHANGELOG entry** for this function anywhere in the 0.1.58→0.11.0 changelog.
- **Its docstring states LORRAX's precondition verbatim:**
  > "If global_shape is the same as local_data.shape, then the data must be the same across all hosts."
  > "if `global_shape == local_data.shape`, the local data is assumed to be the actual target array that will be sharded into device."
- **No collective.** I read `_array_from_process_local_data` (`array.py:986-1044`). With `global_shape == local_data.shape` every `full_dim[i]` is True, so the callback returns `local_data[global_index]` and it hands off to `make_array_from_callback` → per-addressable-device `device_put`. No `multihost_utils`, no `process_allgather`, no `assert_equal`, no barrier. Fully-replicated specs additionally take `make_array_from_callback`'s `is_fully_replicated` fast path (`array.py:791-794`), evaluating the callback exactly once.
- **Single-process** short-circuits to `jax.device_put` after checking `global_shape in (None, local_data.shape)` (`array.py:970-980`) — which is exactly LORRAX's own `process_count() <= 1` branch.

**`global_shape` must be passed explicitly.** With `global_shape=None` it calls
`local_to_global_shape`, which for a sharded spec infers a *larger* global shape
and then rejects the data. This is the one way to get it wrong.

### Two guards that must stay

1. **An operand that is already a non-fully-addressable `jax.Array`** — that is a genuine reshard, not host staging. Keep the existing passthrough to `jax.device_put`.
2. **A process with zero addressable devices in the target sharding.** `array.py:1016-1017`:
   ```python
   addressable_shards = sharding.addressable_devices_indices_map(global_shape)
   shard = next(iter(addressable_shards.values()))
   ```
   No default → bare `StopIteration`. Unguarded on `main` today; no test covers it. **This case is reachable in LORRAX**: `centroid/kmeans_cli.py:201` and `:216`, `centroid/charge_density.py:108,245` and `centroid/pivoted_cholesky.py:119` all build `Mesh(np.asarray(jax.devices()[:1]).reshape(1,1), ("x","y"))`, and the `--no-shard` branch at `kmeans_cli.py:200-201` has no multi-host guard (the *other* two fallbacks at `:216` do, with a comment saying exactly why). `pivoted_cholesky.py:427` then calls `device_put_process_local` on a `NamedSharding` over whatever mesh it was handed. Today LORRAX's `if not idx_map: return jax.device_put(...)` handles this correctly — and `device_put` is collective-free there too, because `process_count() != len(process_indices)` so JAX skips its own assertion.

### Accounting

The function body is **24 executable lines**; the replacement is ~10. **≈ 14
lines removed.** All ~50 call sites are untouched — this is exactly the
all-or-nothing shape the owner asked for, one wrapper, no carve-outs. The win is
not the 14 lines: it is that a hand-rolled `addressable_devices_indices_map` →
`ascontiguousarray` → per-device `device_put` →
`make_array_from_single_device_arrays` loop is replaced by the upstream-tested
path whose documented contract *is* LORRAX's invariant.

> **Note on the brief's span.** Site 6 was given as lines 147-301 (155 lines).
> The function is 147-227 (81 lines, 24 executable). Lines 230-301 are the
> 72-line WHY comment for **site 2**, not site 6.

---

## 3. Site 1 — `aot_memory.py` → `NO_SUPPORTED_PATH`

### The premise is correct, and narrower and sharper than the docstring says

Three independent confirmations, at the pinned XLA commit `3cc8846c`:

**(a) `memory_analysis()` is documented as a lower bound, computed only from
buffer assignment.** `xla/pjrt/pjrt_executable.h` L321-326:

```cpp
// Static memory usage for a compiled program.
// The on-device memory needed to run an executable is at least
//   generated_code_size_in_bytes
//   + argument_size_in_bytes + output_size_in_bytes - alias_size_in_bytes
//   + temp_size_in_bytes.
```

"Static", "at least". The producer (`pjrt_stream_executor_client.h`
L627-641) derives every field from `buffer_assignment_proto()` +
`GetAllocations()` + `SizeOfGeneratedCodeInBytes()` and nothing else — including
`peak_memory_in_bytes`. JAX's own docstring (`jax/_src/stages.py`, and
https://docs.jax.dev/en/latest/jax.stages.html) is even weaker: *"Intended for
visualization and debugging purposes… its structure can be arbitrary: it need
not be consistent across versions of JAX and jaxlib, or even across
invocations."*

**(b) `FftThunk` declares no scratch slice.**
[`xla/backends/gpu/runtime/fft_thunk.h`](https://github.com/openxla/xla/blob/3cc8846c10052cc1c32c4db87866eac4e4cdbccd/xla/backends/gpu/runtime/fft_thunk.h):

```cpp
BufferUses buffer_uses() const override {
  return {BufferUse::Read(input_buffer_, input_shape_),
          BufferUse::Write(output_buffer_, output_shape_)};
}
```

Confirmed positively in **our** wheel — demangled from
`jax_plugins/xla_cuda12/xla_cuda_plugin.so`:

```
xla::gpu::FftThunk::FftThunk(xla::gpu::Thunk::ThunkInfo, xla::FftType,
    absl::Span<long const>, xla::BufferAllocation::Slice const&,
    xla::BufferAllocation::Slice const&, xla::Shape const&, xla::Shape const&)
```

Exactly two `BufferAllocation::Slice`s: input and output. No third.

**(c) The workspace comes from the runtime allocator.** `fft_thunk.cc` builds
`se::OwningScratchAllocator<2> scratch_allocator(device_ordinal,
memory_allocator)` where `memory_allocator` is
`buffer_allocations.memory_allocator()` — the live BFC allocator, not the buffer
plan — and threads it into `CreateBatchedPlanWithScratchAllocator` /
`UpdatePlanWithScratchAllocator`. `xla/stream_executor/cuda/cuda_fft.cc` then
does `cufftSetAutoAllocation(plan_, 0)` → `cufftMakePlanMany64(...,
&scratch_size_bytes_)` → `UpdateScratchAllocator` → `AllocateBytes` →
`cufftSetWorkArea`. Confirmed positively in our wheel:

```
stream_executor::gpu::CUDAFftPlan::Initialize(StreamExecutor*, Stream*, int,
    size_t*, size_t*, size_t, size_t, size_t*, size_t, size_t,
    fft::Type, int, ScratchAllocator*)
stream_executor::gpu::CUDAFftPlan::UpdateScratchAllocator(Stream*, ScratchAllocator*)
stream_executor::OwningScratchAllocator<2ul>
```
plus the imported symbols `cufftSetAutoAllocation`, `cufftMakePlanMany64`,
`cufftGetSizeMany64`, and the source path
`external/xla/xla/backends/gpu/runtime/fft_thunk.cc`.

### The finding the docstring should add: cuFFT is the *only* exception

cuDNN convolution scratch **is** in buffer assignment
([`convolution_thunk.h`](https://github.com/openxla/xla/blob/3cc8846c10052cc1c32c4db87866eac4e4cdbccd/xla/backends/gpu/runtime/convolution_thunk.h): `BufferUse::Scratch(scratch_buffer_, ...)`),
and so is the cuBLAS workspace
([`gemm_thunk.h`](https://github.com/openxla/xla/blob/3cc8846c10052cc1c32c4db87866eac4e4cdbccd/xla/backends/gpu/runtime/gemm_thunk.h): `std::optional<const BufferAllocation::Slice> workspace_`).
So the correct statement is not "`memory_analysis()` misses library workspaces"
— it misses **exactly one**, FFT. That makes the upstream ask small and
precedented: give `FftThunk` a scratch slice like conv and gemm already have,
and it lands in `temp_size_in_bytes` for free.

Also worth recording in the module docstring: the workspace is allocated on
**every** execution, not just the first. `OwningScratchAllocator` is a stack
local in `RunFft` and frees on destruction, so each later call re-enters
`UpdatePlanWithScratchAllocator` → fresh `AllocateBytes`. And
`OwningScratchAllocator::AllocateBytes` passes `retry_on_failure=false`, so on a
tight pool it fails immediately rather than triggering the allocator's retry —
which is consistent with the `aot_cufft_sanity.py` observation that CrI3 Q=13
died at *cuFFT plan creation* rather than at a JAX OOM.

### Is there a supported alternative?

**No, for the workspace half.** `memory_analysis()` cannot see it, by
construction. `device.memory_stats()['peak_bytes_in_use']` is the only thing
that ever will, and it is post-hoc, not AOT — which defeats the purpose (the
choosers exist to avoid running the thing that OOMs). `grep -i fft
xla/xla.proto` at the pin returns **zero** hits: there is no FFT flag of any
kind, and nothing in `xla.proto` bounds library workspace
(`xla_gpu_redzone_padding_bytes` is autotuning-only;
`xla_gpu_temp_buffer_use_separate_color` relocates XLA's own arena;
`xla_gpu_redzone_scratch_max_megabytes` is `reserved`/removed). No upstream
issue asks for FFT workspace accounting — *weak negative; GitHub issue search
does not index code and was unauthenticated.*

**Simplifications that are available but do not remove the reach:**

1. **Resolve libcufft from the wheel instead of `/proc/self/maps`.** The plugin's RPATH is `$ORIGIN/../../nvidia/cufft/lib:…` and `nvidia_cufft_cu12-11.4.1.4` ships `libcufft.so.11` there, so the path is deterministic. Saves ~25 lines but is *strictly less faithful* — `/proc/self/maps` is correct even when `LD_LIBRARY_PATH` supplies a system libcufft. **Recommend keeping `/proc/self/maps`.**
2. **Replace the regex with a structured IR walk.** `jax.stages.Lowered.compiler_ir()` is public; matching `stablehlo.fft` and reading its `fft_length` attribute removes the whole "XLA changed its HLO text format" failure class that `HloFftParseError` exists to catch. Still IR-walking, so still a reach — but a typed one.
3. **Have the callers pass `FftSpec` directly.** The V_q / chi0 / σ / bispinor choosers are LORRAX's own kernels and know their FFT shapes in Python. This deletes the parser, `HloFftParseError`, and 3 of the 5 tests — ≈ 100 of the 188 executable lines. It is a LORRAX design change, not a JAX one, and it trades genericity for size.

### Separate finding — and it is the most consequential thing in this audit

**`aot_kernel_peak_bytes` has zero callers in `src/`.** The only references
outside the module are `tests/test_aot_memory.py` and
`scripts/profiling/aot_cufft_sanity.py`. The module docstring says "Caller (V_q
chooser, chi0 chooser, σ chooser, bispinor pickers) compares `total` to its
budget" — that is not true of the tree as it stands.

What the production planner actually uses is
`common/fft_helpers.py::query_fft_peak_bytes`, called from
`gw/gflat_memory_model.py:141-144`. That function computes

```python
m = compiled.memory_analysis()
total = temp + argument + output - alias
```

and **nothing else**, while its docstring claims the result is *"the full
per-rank HBM footprint of a standalone FFT jit (input buffer + output buffer +
**cuFFT scratch**, minus donated-alias savings)"*.

Per (a)-(c) above that clause is structurally false. `query_fft_peak_bytes`
compiles `make_jittable_local_fftn_3d`, which is `jnp.fft.fft` per axis inside
`custom_partitioning` (`fft_helpers.py::_make_jittable_local_fft`) — i.e. an XLA
`fft` HLO op, i.e. `FftThunk`, i.e. workspace outside buffer assignment. So the
shipped GPU memory model under-predicts the FFT-box stages by exactly the cuFFT
plan workspace, in the one place a user meets it.

**Magnitude is unmeasured** and I am not going to guess it: per-axis 1-D
transforms have much smaller plans than a fused 3-D plan, and cuFFT often needs
no workspace at all for small 1-D sizes. But the numbers already recorded in
`scripts/profiling/aot_cufft_sanity.py` for the 3-D V_q box are not small — CrI3
6×6×1 80 Ry at Q=13: `compiled_peak 66.32 GB`, and **cuFFT plan creation failed**
inside an 80 GB budget, implying > 13.7 GB of workspace that
`memory_analysis()` never reported. Note also that `gflat_memory_model.py`'s own
*analytic fallback* multiplies by `_FFT_CUFFT_FACTOR = 4.0` with the comment
"cuFFT out-of-place plan holds ~2 box-sized scratch slots on top of the in/out
boxes" — so the fallback path counts scratch and the "exact" path does not,
which is the wrong way round.

**This needs no new JAX capability.** LORRAX already owns the code that measures
it. The fix is to route `query_fft_peak_bytes` through
`aot_kernel_peak_bytes` (or to add its cuFFT term), and to correct the
docstring either way. Flagged here because the audit uncovered it; it is a
correctness bug in shipped code, not an API question.

---

## 4. Site 2 — `psum_scatter_checked` &co. → `NO_SUPPORTED_PATH`

### Two corrections first

**It is not a Tier-1 reach.** Reading all 45 executable lines: `jnp.arange`,
`jnp.exp`, `jnp.mod`, `jnp.tensordot`, `jnp.stack`, `jnp.abs`, `jnp.maximum`,
`jnp.finfo`, `lax.dynamic_slice`, `lax.psum_scatter`, `lax.psum`,
`lax.axis_index`, `jax.debug.callback`. Every one is public. There is no
`jax._src`, no ctypes, no binary introspection anywhere in 302-465.

**Nothing upstream can replace it.** An exhaustive census of `xla/xla.proto`
(2237 lines) and `xla/debug_options_flags.cc` (3875 lines) at the pinned commit
yields exactly three `xla_cpu_*` collective flags, **all timeouts**:
`xla_cpu_collective_call_warn_stuck_timeout_seconds`,
`xla_cpu_collective_call_terminate_timeout_seconds`,
`xla_cpu_collective_timeout_seconds`. The algorithm is hardcoded — AllReduce is
`options.setAlgorithm(gloo::AllreduceOptions::Algorithm::RING)`, ReduceScatter is
`gloo::ReduceScatterHalvingDoubling<T>` with no alternative. Every numeric-check
flag in XLA is GPU-only (`xla_gpu_experimental_enable_checksum_tracing_on_thunks`,
`xla_gpu_pgle_accuracy_checker`, `xla_gpu_shape_checks`); the only `xla_cpu_*`
verification knob is `xla_cpu_emitter_verification_level`, which validates
emitted IR, not collective numerics.

### Corroborating structural evidence for the defect

Independently of LORRAX's own 604/604-vs-gloo measurements
(`wk_REL/docs/UPSTREAM_gloo_psum_scatter_corruption.md`):
`xla/backends/cpu/collectives/gloo_collectives_test.cc` contains **exactly one
test** — `TEST(GlooCollectives, AllReduce)` with `kNumParticipants = 2`. There is
no `ReduceScatter`, `AllGather`, `AllToAll` or `CollectivePermute` test, and
nothing at a non-power-of-two world size. Combined with
`gloo/reduce_scatter.h` being functionally untouched since 2018-02-09 and gloo
itself being in "maintenance-only mode", the default CPU reduce-scatter path in
JAX has zero upstream test coverage. That is not proof of the bug, but it is
exactly the condition under which the bug is plausible and would go unfound.

One hypothesis worth recording for whoever files the issue: gloo's
non-power-of-two path in `ReduceScatterHalvingDoubling` uses hand-rolled binary
blocks plus a bit-reversal reorder ("*instead of ranks 0..7 having blocks
A,B,C,D,E,F,G,H … what you get is A,E,C,G,B,F,D,H*"). LORRAX's reproducer is
P=4, a power of two, so this is *not* our failure — but it is the least-exercised
code in the stack and would sharpen any upstream report.

### The honest question is not "can JAX do this"

It is "should this ship". `psum_scatter_checked` has **zero call sites** — its
own docstring says so ("WIRING STATUS: no call site uses this yet"). LORRAX now
runs `impl=mpi` (clean 604/604) and site 4 shouts on stderr if a run ever lands
on gloo. So today this is 164 lines of unwired insurance in a public release.

That is not a reason to delete it — the defect is real, measured, and unfixed
upstream — but it is a reason to decide deliberately. Note that wiring it does
**not** violate the BSE constraint: the added traffic is one 2-element `psum`,
O(1) in `N_mu`, `N_k·N_q` and `P`, and the Freivalds contraction is a local
memory sweep with no large intermediate. The blocker is cost, not comm:
+0.7 % at the 0.41 MB BSE payload but **+107 % at 41 MB and +56 % at 253 MB**
under mpi. If it is wired, wire it where it is free and say so; if it is not,
say in the module docstring that it is a diagnostic entry point, not a live
guard.

---

## 5. Site 3 — `warm_mesh_cliques` → `NO_SUPPORTED_PATH`. Must stay.

### The guard, confirmed three ways

Source, at the pinned XLA commit
[`3cc8846c:xla/backends/cpu/collectives/mpi_collectives.cc`](https://github.com/openxla/xla/blob/3cc8846c10052cc1c32c4db87866eac4e4cdbccd/xla/backends/cpu/collectives/mpi_collectives.cc):

```cpp
void MpiCollectives::Init() {
  int provided;
  MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
  ...   // `provided` is never read again
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
MpiCollectives::CreateCommunicators(...) {
  int flag;
  MPI_Is_thread_main(&flag);
  if (!flag) {
    return absl::UnknownError(
        "MPI: Communicator requested from a thread that is not "
        "the one MPI was initialized from. ...");
  }
```

Positively confirmed in **our** `jaxlib/libjax_common.so`: the 147-byte message
string, the symbol `MPI_Is_thread_main`, the mangled
`xla::cpu::MpiCollectives::CreateCommunicators(...)`, and the source path
`external/xla/xla/backends/cpu/collectives/mpi_collectives.cc`. The project's own
disassembly (`wk_REL/docs/jax_threadmain_alternatives.md` §0) locates the `cmpl
$0x0,-0x54(%rbp)` and the `absl::UnknownError` call site, and confirms that none
of `AllReduce`/`ReduceScatter`/`AllGather`/`AllToAll`/`CollectivePermute`/
`Broadcast`/`Send`/`Recv` carries a comparable check.

The whole 77-line file was read: the check is **unconditional** — no
`DebugOptions` field, no env var, no `Config` member. There is no way to turn it
off.

### No supported pre-warm exists

- `jax/distributed.py` exports only `initialize`, `is_initialized`, `shutdown`.
- The collectives constructors are `xla_client._xla.make_gloo_tcp_collectives` / `make_mpi_collectives` — private, and they must be handed to `make_cpu_client` *before* backend init, so they cannot warm a live client. `make_cpu_client` itself is `jax._src.xla_bridge.make_cpu_client`; `jax.lib.xla_bridge` is gone in 0.9.1.
- Upstream says so itself. [`xla/backends/cpu/collectives/cpu_cliques.cc`](https://github.com/openxla/xla/blob/main/xla/backends/cpu/collectives/cpu_cliques.cc):
  ```cpp
  // Container for initialized and ready to use CPU cliques. In contrast to GPU
  // cliques, CPU cliques are not lockable, and we create communicators lazily
  // when needed.
  ...
  // TODO(b/380457503): Consider switching to a lockable CPU clique model similar
  // to GPU cliques, and creating all communicators upfront.
  ```
  i.e. "create all communicators upfront" is an open upstream wish, not an API.
- The config-knob route was independently falsified twice by this project: `ThunkExecutor::Options` are stack literals (`execute_sequential_buffer_threshold = 512`, `execute_sequential_num_thunks_threshold = 8`) constructed only in `CpuExecutable::Create`, with no `DebugOptions` read; and measured, `taskset` to one core (`NumSchedulableCPUs() == 1`) still refuses, as does `--xla_cpu_use_thunk_runtime=false`.
- The only prior art is [openxla/xla#16430](https://github.com/openxla/xla/issues/16430) ("Segfault when using CPU collectives plus `--xla_force_host_platform_device_count=2`"), which is the only other public appearance of this error string. Closed 2024-09-03 by fixing the *gloo* path; the MPI half was explicitly deferred and the guard is verbatim in `main` today.

### One hazard to add to the docstring

`AcquireCommunicator` memoizes the **failure**, not just the success:

```cpp
absl::call_once(*create_comm_once, [&]() {
  auto comm = CreateCommunicator(collectives, clique_key, rank);
  absl::MutexLock lock(thread_safe_clique.mu);
  if (!comm.ok()) { thread_safe_clique.create_comm_status[rank] = comm.status(); return; }
  ...
});
absl::MutexLock lock(thread_safe_clique.mu);
RETURN_IF_ERROR(thread_safe_clique.create_comm_status[rank]);
```

Once a `(collectives, clique_key, rank)` triple fails the thread-main check it
**never retries** for the process lifetime — every later collective on that
clique returns the same cached `UnknownError`, even from the main thread. So the
warm-up must be the *first* touch of each device set. LORRAX already satisfies
this and its `_WARMED_MESHES.add(key)`-after-success ordering is right, but the
consequence deserves a line: a mesh whose cliques were touched by any earlier
jit cannot be rescued by warming it afterwards.

### Verdict

Necessary. It is the smallest known fix (30 executable lines, three 8-byte
`psum`s, ~150 ms once per process, O(1) in `N_mu`/`N_k`/`N_q`/`P`), it changes no
compiled HLO, it removes a patched-dependency override rather than adding one,
and it is strictly safer than the alternative (`MPI_Comm_split` is collective
over `MPI_COMM_WORLD`; creating cliques from one thread in a program-defined
order removes a cross-rank ordering deadlock the wrapper override leaves open).
It does rest on an internal detail with no stability promise — the clique cache
keyed on the device set alone — which the docstring already says. Keep the
`TF_CPP_VMODULE=cpu_cliques=3` check in the release gates.

---

## 6. Site 4 — `announce_cpu_collectives` → `REPLACEABLE_NOW`, one line

`jax.config.jax_cpu_collectives_implementation` is a supported public read:

- Documented at https://docs.jax.dev/en/latest/config_options.html.
- `jax/_src/config.py:2198-2207` in our pin: `optional_enum_state(name='jax_cpu_collectives_implementation', enum_values=["gloo","mpi","megascale"], default=DEFAULT_CPU_COLLECTIVES_IMPL)` with `DEFAULT_CPU_COLLECTIVES_IMPL = "gloo"`.
- `optional_enum_state` folds the env var into the default at import: `default = os.getenv(name.upper(), default)`, then `setattr(Config, name, property(...))`. `jax.config` is public (`jax/__init__.py`). So `jax.config.jax_cpu_collectives_implementation` returns precisely what LORRAX computes by hand at `runtime/__init__.py:300-301`.
- History: flag added in **jax 0.4.27** (CHANGELOG, 2024-05-07), with `'mpi'` available from the same release; default flipped `none` → `gloo` in **0.5.1** (2025-02-24, "*multi-process CPU communication works out-of-the-box*"); `jax_cpu_enable_gloo_collectives` removed in **0.8.0**.

**Take it**, for two reasons that are not line count: it validates against the
enum (a mis-cased or stale value can no longer be announced as fine while JAX
does something else), and it tracks upstream if the enum changes. Note `'none'`
was legal in 0.4.27 and is **not** legal now — it raises `ValueError` at `import
jax`. So an operator's stale `JAX_CPU_COLLECTIVES_IMPLEMENTATION=none` will kill
the process before `announce_cpu_collectives` can speak. That is arguably better
(loud), but it is a changed failure mode and should be noted where the harness
sets the variable.

**The rest of the function must stay.** The flag reads *intent*, not the resolved
backend: `make_cpu_client` only builds collectives when
`distributed.global_state.client is not None`, it is overridable by an explicit
`collectives=` argument, and **there is no public accessor for what the client
actually got** (searched; negative result). Nor is there any JAX equivalent for
the `MPITRAMPOLINE_LIB` check, the `JAX_PLATFORMS` parse, the
`_gpu_is_present()` false-positive suppression, or the rank-0 gating. Honest
accounting: **1 of 20 executable lines.**

---

## 7. Site 5 — `jax_compile_cache.py` → `NO_SUPPORTED_PATH` for the core

### Reclassification

This is the deepest reach in the audit, not Tier 2. It imports `jax._src.cache_key`,
`jax._src.compilation_cache`, `jax._src.lru_cache`, `jax._src.compiler`,
`jax._src.config`, `jax._src.distributed` and `jax._src.lib`, and it replaces
four upstream functions plus subclasses `LRUCache`. If the release is going to
carry a "we reach into jaxlib here" note anywhere, it is here.

### Every premise verified against the pinned source

| claim in the module docstring | verified |
|---|---|
| JAX writes cache entries from process 0 only, unconditionally | ✅ `jax/_src/compiler.py::_cache_write`: `if distributed.global_state.process_id != 0: ... return`. Also CHANGELOG jax 0.4.19 |
| the key is process-invariant only on GPU | ✅ `jax/_src/cache_key.py:129` `strip_device_assignment=(backend.platform == "gpu")`; `_hash_accelerator_config` hashes `get_topology_for_devices(devices).serialize()` |
| `LRUCache.put` is a bare `write_bytes`, not tmp+rename | ✅ `jax/_src/lru_cache.py`: `cache_path.write_bytes(val)`. Worse than stated: the `filelock` is only taken when `eviction_enabled`, i.e. `max_size != -1` — and `-1` is the default — so at LORRAX's settings JAX takes **no lock on `get` or `put`** |
| `jax_share_binary_between_hosts` deadlocks on an asymmetric hit | ✅ derived independently: in `compile_or_get_cached` the persistent-cache read returns *early* on a hit, before the share path, so a rank-0 hit + peer miss blocks the peers on `blocking_key_value_get_bytes` for `jax_share_binary_between_hosts_timeout_ms` — **default 20 minutes** |

### What upstream actually provides

Exactly two things: rank-0-only writes (0.4.19) and GPU-only device-assignment
stripping (0.4.26, commit
[`0b28a4b168`](https://github.com/jax-ml/jax/commit/0b28a4b168)).

The official page
(https://docs.jax.dev/en/latest/persistent_compilation_cache.html, source
`docs/persistent_compilation_cache.md`, freshness banner **`reviewed:
'2024-11-07'`**) has one paragraph on multi-node. It says a shared FS is required
for *reuse*. It never claims concurrent access is safe, and — the key gap —
**it contains no warning that rank-divergent hit/miss can hang a job.** Its only
statement about divergence is the benign one (non-rank-0 recompiles).

No CPU process-invariant key exists, on `main` today: all 48 commits ever
touching `cache_key.py` (latest 2026-07-29) leave the platform gate alone. No
public issue requests one — *negative search result; this area is dominated by
Copybara exports with no public PR, so treat it as "not found".* An aggravating
fact: jaxlib **0.4.21** gave CPU devices globally unique IDs and real
`process_index`es, which is what made the CPU device assignment process-dependent
in the first place; the 0.4.26 fix deliberately excluded CPU.

The hazard LORRAX's design targets is filed and open upstream:
[openxla/xla#7716](https://github.com/openxla/xla/issues/7716) (hawkinsp,
2023-12-12) — *"Nondeterminism in triton-autotuner and others leads to hangs in
multi-process gpu execution"* — whose proposed remedy is exactly
compile-once-and-broadcast. And the alternative LORRAX considered and rejected
is undocumented, has no public PR
(`PiperOrigin-RevId: 597754861`), and the one PR that tried to fix it
([jax#22272](https://github.com/jax-ml/jax/pull/22272)) was closed unmerged;
[jax#18819](https://github.com/jax-ml/jax/issues/18819) (allow non-rank-0 writes)
is closed unimplemented.

### The honest split of 423 executable lines

- **≈ 140 lines with no public equivalent, at all** — `_install_invariant_key_patch`, `_install_agreement_patch`, `_install_atomic_put_patch`, `_agree_on_entries`. **NO_SUPPORTED_PATH.** JAX offers no cache-key hook, no cache-probe hook, no atomic put, and no public coordination-KV client.
- **≈ 280 lines that were never a JAX workaround** — `$SCRATCH` default + inode reasoning, `_single_stripe` (Lustre), `_prefetch_agreed` (Lustre page-cache latency), the compile/hit counters and atexit report, the five test hooks, the announcements. This is LORRAX's own HPC engineering; JAX will never supply it. If the file is to shrink, this is where — in particular `LORRAX_JAX_CACHE_FORCE_DIVERGE` and `LORRAX_JAX_CACHE_NO_AGREE` are test scaffolding (one of them self-described as "the DEADLOCK REPRODUCER, not a supported mode") shipping in `src/`.

### One trade explicitly NOT to take

The agreement's transport could be moved off the private
`jax._src.distributed.global_state.client` onto
`jax.experimental.multihost_utils.process_allgather` of the 111-byte bitmask —
public API. **Don't.** `process_allgather` runs a jit, which (a) is itself a
persistent-cache probe, i.e. exactly the divergence the agreement exists to
prevent, in the one window where the agreement is not yet installed; and (b) on
CPU requires the mesh cliques to already be warm (site 3) before the cache is
armed. The private KV client is collective-free and compile-free. This is a case
where the private API is the correct engineering choice.

### Two cheap additions

1. `jax_compilation_cache_check_contents` landed in **exactly 0.9.1** (CHANGELOG: *"If set, we miss when `get()` is called on a value that has not been `put()` by the current process… When a value is `put()`, we verify that its contents match."*). It forces a miss on every rank that has it set — the newest and most divergence-prone knob in the family — and it is **not** in `_key_env_fingerprint`'s knob list (which hashes four). Add it.
2. `jax_share_binary_between_hosts` is "considered and rejected" in prose only. Assert it OFF. If a user's environment sets it, JAX's early-return-on-hit plus LORRAX's agreement is precisely the 20-minute stall.

---

## 8. Site 7 — `barrier` → `NO_SUPPORTED_PATH` for the policy. Keep.

`jax.experimental.multihost_utils.sync_global_devices` is the only barrier JAX
has, it is documented
(https://docs.jax.dev/en/latest/jax.experimental.multihost_utils.html), and
LORRAX already calls it. `jax.distributed` documents only `initialize` and
`shutdown` — no barrier, no allgather, no broadcast. `multihost_utils` carries no
deprecation or "moving to stable" notice in 0.9.1, so its experimental status is
stable-in-practice but formally unguaranteed. There is no reach beneath the API
here at all.

The 14 executable lines are pure policy that JAX does not offer and has no
config for: single-process → skip silently; `ImportError` → skip; **anything
else → fatal**. That policy is the whole point (it is what stops a broken
barrier from becoming hang-then-rc-0), and it is correct as written.

Two accuracy notes for the docstring:

- `sync_global_devices` is **not** a cheap coordinator barrier. Its implementation is `assert_equal(np.uint32(zlib.crc32(name)))` → `process_allgather(..., tiled=True)` → a jit + allgather. Four bytes per rank, so ~4 kB at P=1000 — fine — but it needs a live backend and, on CPU, warm cliques. One more reason site 3 must run before any barrier.
- "JAX uses [`name`] as the collective key, so a typo on one rank is itself a hang" is right in effect but wrong in mechanism: the CRC32 goes into the compared *value*, not a key, so mismatched names produce an `AssertionError` on every rank rather than a hang. It is a rank that *skips* the barrier that hangs its peers — which is exactly the failure this function exists to prevent.

---

## 9. Would upgrading help? No.

Zero of the seven sites is fixed by any released jax/jaxlib through 0.11.0
(2026-07-16). Verified per site:

| site | evidence |
|---|---|
| 1 FFT workspace | `fft_thunk.h` byte-identical pin-vs-`main`; `fft_thunk.cc` and `cuda_fft.cc` differ only by `TF_ASSIGN_OR_RETURN` → `ASSIGN_OR_RETURN`. 9 commits since the file moved in Jan 2025, all NFC. No FFT flag exists (`grep -i fft xla/xla.proto` → 0) |
| 2 gloo reduce-scatter | jax 0.9.1 and 0.11.0 pin the **identical** gloo revision `54cbae0d3a67fa890b4c3d9ee162b7860315e341`; `ReduceScatterHelper` byte-identical; `GlooCommunicator::ReduceScatter` differs only by an ABSL-macro NFC |
| 3 MPI guard | verbatim in `main` today; the file's entire history is three commits, none touching the guard |
| 4 config read | already available |
| 5 cache key | none of the 48 commits ever made to `cache_key.py` (latest 2026-07-29) touches `strip_device_assignment=(backend.platform == "gpu")` |
| 6 `device_put` allgather | present at 0.4.31 → 0.11.0 → `main`, with its original TODO intact |
| 7 barrier | `multihost_utils` unchanged in status |

The jax CHANGELOG for 0.9.2, 0.10.0, 0.10.1, 0.10.2 and 0.11.0 contains no
mention of gloo, MPI, CPU collectives, reduce-scatter, or multi-process CPU.
Upgrade for other reasons if you like; **do not present it as a mitigation for
anything in this audit.**

---

## 10. If we only did one thing

**Within the audit's scope: site 6.** Replace the body of
`device_put_process_local` with
`jax.make_array_from_process_local_data(sharding, arr, arr.shape)`, keeping the
two guards (already-global `jax.Array`; zero addressable devices) and the
`LORRAX_CHECK_REPLICA` knob. It is the only site where a public, documented,
already-installed API does the same job, its docstring states LORRAX's
precondition in so many words, it touches ~50 call sites without changing any of
them, and it removes hand-rolled sharding arithmetic in favour of upstream's
tested path. ~14 executable lines, zero behavioural change, zero risk to the
release.

**Uncovered by the audit, and arguably more valuable: the
`query_fft_peak_bytes` correction (§3).** The shipped GPU planner's docstring
claims its number includes cuFFT scratch; XLA's `FftThunk` proves it cannot. The
fix requires no new JAX capability — LORRAX already wrote the module that
measures it, and that module currently has no callers in `src/`. If the release
is about not shipping silent wrong numbers, this outranks a 14-line cleanup.
