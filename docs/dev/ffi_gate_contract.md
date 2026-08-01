# The FFI env-gate contract (`ffi.gate.Gate`)

*One resolver for every env-gated, rank-local FFI capability in the tree.
Written for the agent who adds the fourth dial, and for the reviewer who has
to believe the third one.  Sources: `src/ffi/gate.py`,
`src/ffi/gemm.py`, `src/ffi/fft.py`.  Consolidation
2026-07-30; the assessment that motivated it is
`wk_REL/FFI_MICROSERVICE_ASSESSMENT.md`.*

> **REQUIRED-LAYER REVISION, 2026-08-01** (`docs/architecture/decisions.md`
> — "FFI backends are required, not optional").  Every gate now defaults
> **ON**; the `auto` capability-detection tier (`_auto_lexical`, the
> memoized verdict, the AUTO-ON/auto-unavailable announcements) is
> **deleted** — auto-demotion to a duplicate compute path is what the
> ruling forbids.  Three consequences rewire this page:
>
> 1. **`Gate.enforce(mesh)`** is the new startup tier: called per gate by
>    `runtime.initialize_communicator_stack` (step 6b) right after the
>    mesh exists, so a missing/unloadable library refuses AT STARTUP,
>    quoting `probe_target`'s three-way reason (which names the `.so`)
>    and pointing at `docs/environment/overview.md`.
> 2. **`=0` is per-gate policy** (`Gate.off_policy`): `"fallback"` runs a
>    structurally-retained native path, announced once as an uncertified
>    debug opt-out (`LORRAX_BANDS_GEMM_FFI` — the XLA einsum arm must
>    exist for `extra='minor'`; `LORRAX_FFT_FFI_FUSED` — the decomposed
>    chain is itself FFI-served); `"refuse"` names a DELETED duplicate
>    (`LORRAX_FFT_FFI` — the XLA flat-k arm is gone).
> 3. **The grammar fallback is the gate's default, not `off`**: with
>    `off` able to refuse, a typo resolving to `off` would kill a run;
>    resolving to the certified default (announced) cannot.
>
> Prose below describing `auto`, the two-tier lexical probe, or "default
> OFF" is kept as design history; where it conflicts with this box, this
> box wins.

Before this existed there were **four** independent "resolve a backend"
idioms sharing only `ffi_loader.probe_target`, and one of them
(`gw/ppm_tau_kernel.py:81`, `LORRAX_FFT_FFI_FUSED`) had drifted far enough
to be a live correctness trap: no grammar, no announcement, so `=yes`
worked and `=Y` silently did nothing.  There are now **two**, and they are
different for a reason (§4).

## 1. The contract

```python
from ffi.gate import Gate

GATE = Gate(env="LORRAX_FFT_FFI", target="lorrax_mklfft_flat_k",
            platforms=("cpu", "CUDA"), modes=("off", "on"),
            default="off", off_label="default XLA FFT path",
            label={...}, resolved_msg={...},
            refuse_platform_msg="...", refuse_probe_msg="...")

GATE.mode()          -> "on" | "off" | "auto"     strict grammar
GATE.enabled()       -> bool        TIER 1, lexical — cache-key safe
GATE.platform_ok(m)  -> bool        does this mesh's platform have a backend
GATE.require(m)      -> platform    TIER 2, announce-or-REFUSE
GATE.resolve(m)      -> platform|None                mode-aware tier 2
```

### 1.1 Grammar — a NAMED vocabulary, per gate

| mode | spellings | meaning |
|---|---|---|
| `auto` | `auto`, unset | capability detection: on where known-optimal AND available |
| `off` | `0` `off` `false` `no` | never |
| `on` | `1` `on` `true` `yes` | explicit request; announce-or-refuse |

Unset/empty always maps to the gate's declared `default`, never to a
spelling.  A gate declares WHICH modes it accepts: `LORRAX_FFT_FFI` is
two-valued, so `LORRAX_FFT_FFI=auto` is a *grammar error* there rather than
a silently different policy from the GEMM dial.

Anything outside the vocabulary → announce once, resolve **`off`** — even
for a gate whose default is `auto`.  An unparseable request is not evidence
that the user wanted the capability, so take the direction that cannot break
a run.  Message shape (byte-identical to the two pre-consolidation copies):

```
*** LORRAX_FFT_FFI='Y' is not a recognized value (accepted: 0/off/false/no,
1/on/true/yes).  Treating as OFF (default XLA FFT path). ***
```

### 1.2 Two tiers, and why one resolver cannot be one tier

| | tier 1 `enabled()` | tier 2 `require()` / `resolve(mesh)` |
|---|---|---|
| platform from | `JAX_PLATFORMS`, lexically (`ffi_loader.platform_from_env`) | the live `mesh.devices` |
| initializes the JAX backend? | **NO** | yes (it has a Mesh, so the backend is already up) |
| exact? | no — lexical | yes |
| may be called before `jax.distributed.initialize`? | **yes, and it must be** | no |
| used for | KERNEL-CACHE KEYS at factory time | the actual lowering decision, and refusals |

Tier 1 exists because consumers key kernel caches on the gate
(`gw/ppm_tau_kernel.py`'s `pipeline_key` / `cache_key`), and those factories
can run before `jax.distributed.initialize`.  `ffi_loader.py:178-180`
records why `get_lib(None)` is forbidden there.

The lexical read is a known, bounded inaccuracy and resolves in the SAFE
direction: an unset `JAX_PLATFORMS` resolves CUDA-first, and so does a CPU
run whose `JAX_PLATFORMS` still reads `cuda,cpu`; both give auto-OFF — the
XLA lowering, no error, just no speedup.  Both are fixed the same way
(`export JAX_PLATFORMS=cpu`, or set the dial explicitly).  Production CPU
harnesses export it, and `runtime.bootstrap()`'s GPU-less downgrade
*forces* it to `cpu`.

### 1.3 `auto` may demote — but must announce, from the rank it happens on

The rule is one question: **can this decision differ per rank?**

| decision | differs per rank? | who speaks |
|---|---|---|
| platform out of scope | no | rank 0 (`scope="rank0"`) — or nobody, see below |
| `auto` engaged (probe OK) | no in practice | rank 0 |
| env-var grammar error | **yes** (env is per-process) | the rank it happened on (`scope="local"`) |
| probe FAILED | **yes** — usually one rank's `LD_LIBRARY_PATH` | the rank it happened on |

`scope="local"` prints unprefixed on rank 0 (so the strings the gate
harnesses grep are byte-identical) and `[rank N] `-prefixed elsewhere, so a
one-rank misconfiguration is both visible and attributable.  A rank-0 guard
on these two is exactly how such a misconfiguration hides; that guard is
what the pre-consolidation copies had, or rather did not have — they printed
the grammar warning on *every* rank with no dedupe and no attribution.

**Silence is allowed only when declared.**  `Gate.silent_platform_demote` is
a string, not a bool: to demote silently you must write down why.  The one
current use is the GEMM dial on a GPU mesh, where the target is not in
`ffi_loader`'s CUDA table at all — the dial does not merely resolve off, it
does not exist, so an announcement would report a non-decision on every GPU
run and name a variable the user has no reason to have set.

`auto` also does not print the first-use receipt: the tier-1 `AUTO-ON` line
already is that receipt.  The receipt belongs to an EXPLICIT request.

### 1.4 Explicit requests REFUSE — with the reason and the fix

Under `mode == "on"`, `require(mesh)` raises `RuntimeError` when:

1. the mesh's platform is not in `platforms` — message names the platform it
   saw and why the dial is scoped that way;
2. `probe_target` says the handler is unusable — message quotes the reason
   **verbatim**, because `probe_target`'s three-way taxonomy (*unknown
   target* / *library could not be loaded* / *loaded but does not export*)
   has three different fixes and a bare `False` sends people to rebuild a
   library that is fine (`ffi_loader.py:384-441`; the failure that motivated
   it is wk_P G4, 2026-07-25).

`require()` is deliberately **mode-independent**: it answers "can this mesh
serve this handler", which is what an explicit request asks and also what a
wrapper constructed directly by a consumer asks
(`ffi.mklfft.make_gw_conv_ffi` is built only by a caller that already
decided; its refusals must not depend on re-reading a flag).

### 1.5 Refusal is TWO-PHASE, and that is not a defect

Only platform and handler can be checked at resolve time.  Operand **dtype,
rank and extent are trace-time facts**, so those refusals live in the
wrapper body (`ffi/fft.py`'s `_flat_k_fft_ffi`,
`contract_bands._ffi_dtypes_ok`).  Say so in every service doc; a
`plan()`-shaped single-phase API would have to lie about when it checked.

## 2. Rank identity: one rule, Python and C++

`gate.rank_id()` reads `SLURM_PROCID` → `PMI_RANK` → `OMPI_COMM_WORLD_RANK`,
the same list in the same order as the C++ `announce_here()`
(`cpp/mklblas/gemm_batch_ffi.cc:229-235`).  It falls back to
`jax.process_index()` only when none is set (single process, or an unknown
launcher), where that is both correct and harmless.

It does **not** try `jax.process_index()` first, and that is load-bearing:
`jax.process_index()` goes through `get_backend()`
(`jax/_src/xla_bridge.py:1119`) and **initializes the XLA backend**.  The
pre-consolidation `contract_bands._rank0()` did exactly that, two lines
after a docstring promising the auto resolution "never initializes the JAX
backend" — true of `platform_from_env`, false of the announce that follows
it.  Consolidating the three rank-0 helpers is what surfaced it.  The gate
job pins this with a call counter (`wk_REL/gatecheck.py` cell C(4)).

## 3. Adding a fourth dial

1. Declare the `Gate` **in the FFI subpackage that owns the handler**, next
   to its `ffi_call` — never at a consumer.  `fft_helpers.py:306-307` states
   the rule ("the backend switch happens here and nowhere else"); the one
   flag that broke it is the one that lost grammar and announcements.
2. Give it the narrowest vocabulary that is honest.  No `auto` unless you
   can name the capability test AND the measured reason.
3. Write the messages.  They are the product: harnesses grep them
   (`gbp_ab.sbatch:17`, `gbp_gate.sbatch:144`), and a refusal without the
   fix in it is a broken promise (QUALITY_PATTERNS §6).
4. Export `<dial>_enabled()` from the subpackage `__init__` and make every
   consumer's kernel cache key include it (grep `pipeline_key`).
5. Gate it: extend `wk_REL/gatecheck.py`.  Every cell there runs a
   deliberately BROKEN twin that must fail; a cell whose twin passes is
   reported VOID, not green — that is not a formality, it is how the first
   run of this very gate discovered that its two bit-parity cells were
   perturbing below the c128 ULP and testing nothing (job 7882060).
   Run it on **both** platforms: `wk_REL/ffisvc_gate.sbatch` (CPU, host
   MKL) and `wk_REL/ffisvc_gpu.sbatch` (rtx-dev, cuFFT).  Reference passes:
   **7882116/7882124** (CPU, 9/9) and **7882123** (GPU, 9/9).

## 4. What this does NOT absorb, and why

**`ffi/linalg/resolve.py` stays separate.**  Its guards 4 (one JAX process
per device, `resolve.py:399-407`) and 5 (mesh geometry, `:410-430`) exist
because those backends hold an MPI/BLACS/NCCL communicator.  FFT and GEMM
are rank-local handlers *inside* a `shard_map` with no communicator: both
guards would be permanent no-ops, and `resolve_backend`'s promise — "a
returned FFI name is a *promise*" (`:261-264`) — would become weaker for two
of its ops than for the others.  Its selection input is also a function
ARGUMENT (`plan.py:301`), not an env var, so it has no grammar tier at all
and resolves everything in one phase because it receives `n`.

Two idioms, then, and the split is by what the backend IS:

* **`plan(op, mesh, backend=, n=)`** — distributed solvers that hold a
  communicator.  Argument-driven, single-phase, every guard at resolve time.
* **`Gate`** — rank-local dials inside somebody else's `shard_map`.
  Env-driven, two-tier, two-phase.

**No shared C++ handler base either.**  cuFFT plan workspace, MKL OpenMP
chunking and a scratch-free BLAS call are three genuinely different measured
designs (`src/ffi/cufft/__init__.py`, `TEMPLATE.md:188-195`).  The shared
resolver is Python policy; the handlers stay unshared.
