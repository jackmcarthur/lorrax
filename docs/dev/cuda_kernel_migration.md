# Porting a native-JAX expression to a CUDA FFI kernel

*A checklist and a trap list, written for the next agent doing this somewhere
else in the tree. It is generalised from ONE worked example — the fused-conv
family's k-minor member, `src/ffi/cpp/cufft/conv_kminor_cuda_ffi.cc`, which
replaced the BSE ladder-W rung's ifft·multiply·fft chain and measured 4.66x on
that chain — 1.84x from the hardware-optimal traffic bound. That file's header owns the NUMBERS (the "WHAT MOVED THE NUMBER"
catalog); this page owns the METHOD. Where they overlap, the numbers live
there and the reasoning lives here.*

*Related: `ffi_gate_contract.md` (the dial), `flat_k_fft_service.md` (the FFT
service contract), `vendor_gemm_service.md` (a mechanism page in this shape),
`docs/architecture/ffi_layout.md` (where handlers live and what the build
does), `QUALITY_PATTERNS.md` §6 (refusals name their fix).*

---

## 0. First decide whether to do it at all

XLA is a good compiler. It will not be beaten on a single elementwise op, a
single GEMM, or a single FFT — it dispatches the same vendor libraries you
would. **The win is never the operation; it is the SEAM BETWEEN operations.**

Ask, in this order:

1. **How many HBM round trips does the expression make, and how many does it
   need?** Count them in the optimized HLO, not in the source. The worked
   example: XLA emitted 8.1 passes over a 91.7 MB buffer where the fused form
   needs 1.12. That ratio *is* the available speedup, and nothing else in the
   port matters as much.
2. **Is there a layout change on either side?** A transpose that exists only to
   feed the next op can be absorbed into a kernel's store or load map and
   deleted rather than moved. In the worked example that one op was 30% of the
   whole matvec.
3. **Are there normalisation / scaling passes?** Vendor FFT libraries implement
   no norm convention, so the compiler emits separate scale kernels. Every
   scalar that commutes with the linear parts folds into one constant applied
   once, inside.
4. **Is a handler for this shape already in the tree?** Check
   `ffi_loader._CUDA_TARGET_SYMBOLS` and the family docstrings in
   `src/ffi/fft.py` before writing anything. Two handlers computing the same
   expression at different layouts is a *family* (below); two handlers
   computing the same expression at the same layout is a defect.

If the answer to 1 is "it already makes one pass", stop. You will spend a week
to lose to `nvcc`'s own scheduler.

---

## 1. The build path and runtime fallback

The device leg (`src/ffi/cpp/`, `LORRAX_FFI_PLATFORM=cuda`) keeps each direct
kernel in one **NVRTC source string**. That string is the authored source for
both routes: on the Perlmutter build, CMake extracts the literal into a
generated `.cu`, compiles an sm_80 cubin with the nvcc already required by the
cuBLASMp leg, and embeds it in `liblorrax_ffi.so`; at runtime, a matching A100
loads the image through checked driver-API calls and never invokes NVRTC.

When no matching image is embedded, the same literal is compiled for the
queried device compute capability with NVRTC and loaded through driver-API
entry points resolved by `dlsym`. A cuFFT-only/eigh-only build with
`LORRAX_FFI_HAVE_CUBLASMP=OFF` still requires no nvcc and retains that route;
do not move the AOT step outside that existing CUDA-language gate.

What you get for free by following it:

* **The fallback serves every GPU generation.** The architecture string is
  built at run time (`--gpu-architecture=sm_%d%d`). The embedded fast path is
  currently narrower; see the installation gap below.
* **No link-time `libcuda`.** The driver is absent on login and build nodes;
  `dlsym` against the already-loaded library (JAX loaded it) keeps the artifact
  linkable everywhere.
* **A negative cache.** An NVRTC failure is deterministic for a
  process+context, and the handler is called once per dispatch, so record the
  failure and refuse from the record instead of recompiling on every call.

Reusable glue, currently duplicated on purpose between
`fft_flat_k_cuda_ffi.cc` and `conv_kminor_cuda_ffi.cc`: the `DriverApi` dlsym
struct, `ensure_kernels` (per-context module cache + negative cache), and the
launch helper. **They are duplicated because an NVRTC failure is cached per
module: one source string serving two handlers means a compile error in the
new, uncertified one disables the certified one in the same process.** Fold
them into a shared header once the second handler is certified — that is a
stated debt, not an oversight.

Build into an **isolated** directory (`LORRAX_FFI_BUILD_DIR=.../build_<lane>`),
never the tree's `build/`, and select it per-run with `LORRAX_FFI_SO`. The
precedent is the `distrib_la` `build_matmul_cuda` lane. `src/ffi/cpp/build.sh`
runs the whole build contract (`scripts/verify_ffi_build.sh`) unchanged.

### AOT images and the sm_80 installation gap

!!! danger "TODO — implement a real multi-architecture installation build"
    Only **sm_80** is prebuilt today. That is deliberate for Perlmutter's
    A100s and removes a measured roughly 30-second NVRTC cold-start cost, but
    it is not a general portability design. Any other architecture logs
    `AOT_ARCH_MISS` and correctly uses the NVRTC fallback, paying the compile
    cost per process.

    Replace this with either a fatbin over an explicit, maintained
    architecture list or an install-time compile for the detected GPU. Keep
    the kernel text single-sourced: generate AOT inputs from `kKernelSrc` (as
    the current CMake does), never maintain a parallel hand-written `.cu`.
    Installers should read the prominent warnings in
    [Perlmutter installation](../installation/perlmutter.md#native-ffi-stack-on-perlmutter)
    and [FFI native libraries](../installation/ffi-native-libs.md#4-build-liblorrax_ffiso-non-shifter).

---

## 2. The traps, in the order they bite

Each of these was found by measurement on a kernel that was already CORRECT.
None of them shows up in a parity test. Check `-Xptxas -v` output and the
optimized HLO, not just the residual.

| # | Trap | Symptom | Fix |
|---|---|---|---|
| 1 | 64-bit integer div/mod in a per-element loop | kernel ~4x off roofline, profile blames "integer" | hoist to a per-ROW table; carry loop indices incrementally |
| 2 | complex128 as two `double` loads | 2 instructions, half a sector each | one 16-byte `__align__(16)` struct |
| 3 | register array with a runtime loop bound | `-Xptxas -v` shows a stack frame / spill stores | template the body on the count; ONE ENTRY POINT PER VALUE |
| 4 | one kernel switching over N instantiations | every launch pays the heaviest arm's registers; big blocks fail to launch | separate `extern "C"` entry points + `__launch_bounds__` |
| 5 | even shared-memory row stride | 8-way bank conflict when threads walk rows | `SP = n \| 1` |
| 6 | `__restrict__` on a pointer XLA may alias | UB under `input_output_aliases` | drop it on the aliased pair; keep it on the others |
| 7 | assuming the original roofline still holds | you optimise the term that stopped binding | re-derive after every change |

**Trap 3 is the expensive one** and it is worth restating: a register array
indexed by anything the compiler cannot resolve at compile time is not in
registers — it is in *local* memory, which is DRAM behind L1. That is strictly
worse than the shared memory you were trying to avoid, and it is silent. The
tell is `0 bytes stack frame` in `-Xptxas -v`; treat anything else as a defect.

---

## 3. Keep the extents runtime, and the counts compile-time

The tension in every one of these kernels: performance wants compile-time
sizes, and generality forbids a per-size compiled family.

The resolution that worked, and that generalises: **template on the per-thread
CAPACITY, never on the problem's extents.** The problem's extents arrive as
call attributes and are used as runtime loop bounds; the *number of elements a
thread holds* is a small bounded set you instantiate. Then one code path serves
every size, primes included, and the register arrays are still register arrays.

Where you need an intra-thread reduction over a register array, choose the
ownership so the reduction runs over the WHOLE array — then both the output and
the source index are compile-time loop counters of unrolled loops, and the
extents can stay runtime values. In the worked example that is why a thread
owns a *line* along one axis and not a *plane*: the plane needs
`acc[(e/n2)*n2 + j]`, whose index depends on a runtime extent, and it spills.

If a shape does not fit the fast structure, keep the general structure as a
**plan-time fallback arm** and gate both against the same reference. A shape
that misses the fast path is then slow, not wrong. Report which arm ran in the
log line, so a measurement can never be attributed to the wrong one.

---

## 4. Bounds are derived; refusals are named

Query the device for anything that limits you (`cudaDeviceGetAttribute`;
`cudaDevAttrMaxSharedMemoryPerBlockOptin` plus `cuFuncSetAttribute` if you want
more than the 48 KB default) and compute the envelope from it. Then, for any
operand you cannot serve:

* refuse with `ffi::ErrorCode::kInvalidArgument`,
* quote the arithmetic (what it needed, what the device has),
* state the largest thing you CAN serve, and
* **name the handler to use instead.**

A kernel that quietly mis-handles one shape is worse than one that refuses it,
and a refusal without the fix in it is a broken promise (`QUALITY_PATTERNS`
§6). The same rule covers dtype: refuse `complex64` by name rather than
up-casting, because a caller measuring an fp32 arm must not silently get fp64
arithmetic.

---

## 5. Rank-locality is what "works at any GPU count" means

These handlers run *inside somebody else's* `shard_map`. They hold no
communicator and see only the local tile, so multi-GPU generality is two
checkable properties, not an assertion:

1. **Every extent comes from the runtime buffer dimensions.** No shape is
   assumed; the local tile changes with the mesh and must be data.
2. **The compiled module contains no collective.** Census the optimized HLO for
   `all-gather` / `collective-permute` / `reduce-scatter`. A handler that
   quietly needed a gather passes every value test and still will not scale.

Test both on a real multi-device mesh against the single-device reference, and
say which you tested (one process driving N devices is not the same as N
processes).

---

## 6. Wiring: the seam, the dial, the gates

**Placement.** The handler goes in the shared vendor layer beside its
relatives, not next to its first caller. Two entries computing the same
expression at different resident layouts are a FAMILY: give them one contract
docstring, one seam in `common/fft_helpers.py`, and a table saying which
layout picks which. Never transpose to reach one — that spends exactly what
the fusion saves.

**Naming.** Share a target string across platforms only if both platforms
implement it. A CUDA-only handler takes a CUDA-named target, so a CPU mesh
REFUSES instead of resolving to nothing.

**The dial** (`ffi.gate.Gate`, contract in `ffi_gate_contract.md` §3). An
accelerator dial is not a required-layer dial: its OFF state is the production
implementation, so it carries three modes — `off`, `auto` (the default once
certified) and `on`. `auto` uses the kernel where the capability is present and
falls through to the caller's own path where it is not, SILENTLY, because the
fallthrough is the certified reference rather than a degraded twin; `on`
refuses by name, which is the mode a certification run uses so it cannot
silently measure the other arm. Declaring `auto` obliges you to fill
`auto_capability` — the capability test, named, printed in the startup line —
and the `Gate` constructor refuses an `auto` without one. Consequences to wire,
all of which have tests that will catch you:

* add it to `ffi.FFI_DIAL_ENV` **and** `common.jax_compile_cache.RANK_FINGERPRINT_ENV`
  (a dial that changes the emitted op set must be in the cross-rank
  fingerprint), and to `ffi.ffi_dial_key()`;
* add it to `runtime._ffi_dial_facts` so the startup report states it;
* enforce it at startup only when it is ON, so a default-off dial does not
  announce an opt-out on every run.

**Do not edit the consumer.** Put the whole behaviour in a new module and hand
the integration lane a patch that is an import plus a guarded rebind. The
consumer then carries one spelling of the expression plus a hook.

---

## 7. The certification gates that made it believable

Structure them so each cell can fail for exactly one reason:

| gate | what it catches |
|---|---|
| numerics vs the native-JAX expression at the real shapes | the port is wrong |
| the two output-layout arms are **bit-identical** | the arms forked; a tolerance here would test nothing |
| a `complex64` operand RAISES | silent demotion |
| optimized-HLO census | the chain did not actually collapse; a temp buffer survived |
| a size sweep across the whole accepted range, primes included | a per-size assumption crept in |
| a multi-device mesh + collective census | it does not scale |
| the shape sweep the *previous* approach died on | you reproduced the defect you exist to remove |

Two notes on the numerics gate. First, quote the same metric the prior lane
used, or the numbers are not comparable. Second, **both arms are
approximations**: measure each against a higher-precision reference before
concluding a 1e-15 difference is your error budget. In the worked example the
CUDA kernel was *closer to the exact answer* than the FFT chain it replaced,
and their mutual difference was the c128 ULP floor.

Benchmark with `BENCH_ALLOC=bfc`; the platform allocator inflates a microbench
by up to 11x. Take the minimum over blocks, not the mean, on a shared node —
and re-run before believing a regression, because a co-tenant moves the
*reference* arm too.

---

## 7b. The two BSE adoption paths

The hook lives in `bse_ring_comm`'s two ring-matvec builders, which is why the
adoption story splits cleanly in two.

**Ring-matvec consumers inherit it with ZERO extra work.** Anything that builds
its operator through `build_bse_ring_matvec` / `build_bse_ring_matvec_full` —
the non-TDA optical solver, FEAST, the pseudopole route, the ladder resolvent
and its preconditioner stack — gets the fused rung the moment the dial resolves
`auto → on`, because they all share the one `_apply_W_from_T` the hook
replaces. There is nothing per-consumer to port, and nothing per-consumer to
gate: the routing decision is `rung_uses_conv_kminor`, called once per operator
build.

**The TDA stack matvec is a real port, and it is smaller than it looks.**
`bse_stack_matvec._w_stack → _conv_decode` sits INSIDE one big `shard_map` and
calls `local_fftn3` directly. The historical reason it had no FFI route was
that the flat-k entry (`make_flat_k_*`) wraps its OWN `shard_map`, and
`shard_map` cannot nest — **that was a property of that wrapper, not of custom
calls.** This handler is per-rank local and introduces no collective (§5,
measured), so calling it on the local shards from inside an existing
`shard_map` composes. The port is therefore:

1. call the handler on the local `(d0,d1,d2,d3,d4,nk)` shard directly, without
   the `shard_map`-wrapping factory — i.e. the `ffi_call` from
   `ffi.fft.make_conv_kminor_ffi`'s inner body, not the factory;
2. a **layout check on the TDA tile**: the stack's chain buffer must present k
   minor-most and the stored kernel as `(d1,d2,nk)`, or the port needs an
   `out_layout` of its own rather than a transpose (§0.2);
3. **one gate cell** against the stack's existing XLA spelling, at the same
   1e-15 class as the ring gate.

Do not port it by lifting the ring hook: the ring's operand names and its
decode layout are the ring's, and the stack's `_conv_decode` consumes a
different one.

**Follow-on for the integration lane, recorded here because it is not the
kernel's to fix.** The non-TDA matvec applies the rung TWICE on independent
operands and those two applications are measured STRICTLY SERIAL (ratio
1.97–2.00, unchanged by `--xla_gpu_enable_latency_hiding_scheduler`). With the
fused kernel at ~47% of achievable HBM there is real headroom in running them
concurrently, but a handler cannot take it: the FFI contract hands it XLA's
stream, and launching onto a private one would have to synchronise back
(serialising anyway) or race the consumer. It is a scheduler question.

---

## 8. Handoff: porting the Σ pipeline's `gw_conv` consumer onto this backend

*Written for an agent with no context from the k-minor lane. Execute against
this section; it names every file you need and every fact that is already
measured, so nothing here should be re-derived.*

### 8.1 What you are porting, and what you are NOT

The Σ τ kernel (`src/gw/ppm_tau_kernel.py`) already calls a **fused conv**:
`common.fft_helpers.make_flat_k_gw_conv` → `ffi.fft.make_gw_conv_ffi` → target
`lorrax_mklfft_gw_conv`, implemented for CUDA in
`src/ffi/cpp/cufft/fft_flat_k_cuda_ffi.cc::GwConvDispatch`. That handler is
**already fused** in the sense that matters most (§0.1): three transforms and
the broadcast multiply in one call, with the R-space intermediate never
materialised. So this is **not** a "add fusion" port. It is an "adopt the
kernel-level techniques and the certification structure" port.

Do not attempt to make Σ call the k-minor entry. Its operand is k-LEADING and
transposing to reach a k-minor kernel spends exactly what the fusion saves
(§6). The two members stay distinct — see 8.4.

### 8.2 What is shared as-is

| asset | where | note |
|---|---|---|
| NVRTC + driver-API glue | `conv_kminor_cuda_ffi.cc` (`DriverApi`, `ensure_kernels`) | currently duplicated on purpose (§1); **folding it into a shared header is the natural first commit of this port**, and it is the change that makes the duplication stop being debt |
| the twiddle-ring axis pass | `lrx_pass` / `lrx_ctg_cross` | direct per-axis DFT against an `len`-entry ring, extents runtime; drop-in for any separable transform |
| the launch-plan discipline | `plan_launch` | device-derived bounds, odd shared stride, divisor thread counts, arm selection in one place |
| the gate harness shape | `tests/bench/bench_conv_kminor.py` | the six cells of §7; copy the structure, not the shapes |
| the dial wiring checklist | §6 above | every list a new dial must join |

### 8.3 What differs, and it is all layout

1. **Stride handling is the whole difference.** The k-minor handler reads a
   contiguous k-row. The Σ operand is `(nk, a, mx, b, my)` with k LEADING, so a
   transform element's neighbours are a full `mx·b·my` tile apart. The current
   CUDA implementation delegates that to `cufftPlanMany64`'s advanced data
   layout. A hand-written replacement would have to make each thread's `len`
   gathers stride by the batch extent — coalesced ACROSS threads (consecutive
   threads take consecutive trail elements), which is the opposite mapping from
   the k-minor arm's.
2. **The layout anchor points the other way, and this is measured** (O7,
   `reports/screening_diagrams_wbse/evidence/opt_fftffi/`): Σ's `dot` wants
   k-major and XLA's `fft` wants the transform axes minor-most, so every
   boundary paid a full transpose of the μ² tile — 65% of the staged τ dispatch
   before the FFI existed. The FFI removes a transpose there. In the BSE rung
   there was no such transpose to remove, which is why the same handler LOST
   there (1.61× at nk=64, 4.00× at nk=216).
3. **There is no residency bound on the Σ side.** The k-minor arm must hold a
   whole k-row in shared memory (§4); a plan-based strided engine does not, so
   Σ keeps access to large `nk` that the k-minor member refuses by name.
4. **Amortisation differs, so the kernel operand differs.** The k-minor member
   takes `W` already in R space because its caller (`ensure_W_R`) builds it
   once per solve and reuses it across hundreds of matvecs. Σ has no such
   cache, so its handler transforms `W` itself, every call. Do not "unify"
   this: it is a real difference in the callers, not an inconsistency.

### 8.4 Why the two members must remain distinct entries of one family

Because the choice between them is a **measured property of the caller's
resident layout**, not a preference — and because each is a loss in the other's
position:

* k-strided in the k-minor caller: 1.61× the XLA chain at nk=64, 4.00× at
  nk=216 (O7 table C). The batch stride *is* the μ·ν·s² tile there, and
  `cufftPlanMany64(istride=T, idist=1)` degrades with it.
* k-minor in the k-strided caller: it would need a transpose to reach the
  contiguous row, which costs what the fusion saves, plus the residency bound.

One contract, two resident layouts, one docstring stating both — that is the
family shape (`ffi/fft.py`, "THE FUSED-CONV FAMILY"). A third member is welcome
under the same contract; a merged member is not.

### 8.5 The concrete first steps

1. Extract the NVRTC/driver glue into `src/ffi/cpp/cufft/nvrtc_module.h` and
   have BOTH handlers use it. Zero behaviour change; gate with the existing
   `test_fft_flat_k_numerics.py`.
2. Stand up the gate harness for Σ's conv *before* touching the kernel: the six
   cells of §7 against the current handler, so the port has a reference that
   already passes.
3. Only then measure whether a hand-written strided kernel beats
   `cufftPlanMany64` on Σ's shapes. It may not — Σ's transforms are large and
   plan-based O(N log N) is the right engine there. **The techniques worth
   porting regardless of that answer are the scale fold, the store-pattern
   layout deletion, and the derived-bound refusals**, all of which are engine
   independent.
