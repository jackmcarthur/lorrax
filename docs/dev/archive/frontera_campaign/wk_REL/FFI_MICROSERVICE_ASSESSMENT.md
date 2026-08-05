# Can the FFT and GEMM work be wrapped as microservices like `contract_bands`?

*Assessment only — nothing under `/work2/08271/jackmc/frontera/lorrax` was modified.
Repo state: branch `fix/zq-band-gather-device-invariance` @ `49877c0`, dirty working
tree (14 modified files incl. `docs/dev/staged_reshard_primitive.md`,
`src/common/contract_bands.py`). Every claim below carries a `file:line` I read.
Where a claim rests on a negative search, the search is named.*

---

## 0. Verdict up front

**Yes for GEMM. Half-yes for FFT. And there is a third service hiding inside the FFT
work — the memory-model query — that is currently making a false promise.**

Three separate findings drive everything else:

1. **The house already has the shape, written down, and these two services do not
   follow it.** `src/ffi/AGENTS.md:149-150` and `src/ffi/TEMPLATE.md:10-20` prescribe
   `src/ffi/<lib>/<feature>.py` (a `shard_map` → `ffi_call` wrapper) re-exported from
   `src/ffi/<lib>/__init__.py`. `src/ffi/mklfft/`, `src/ffi/cufft/` and
   `src/ffi/mklblas/` contain **`cpp/` and nothing else** — no `__init__.py`, no
   Python module. Their Python halves live in `src/common/fft_helpers.py` and
   `src/common/contract_bands.py`. These are the **only two** of the 21 `ffi_call`
   sites in `src/` whose call lives outside `src/ffi/` (full inventory below, §A.6).

2. **There are four independent "resolve a backend" idioms in the tree**, sharing only
   `ffi_loader.probe_target`. A fifth would be a mistake; consolidating to two is
   cheap and safe.

3. **`query_fft_peak_bytes` is the FFT service's third public entry point, is
   contractually false, and feeds the GPU chunk planner.** It is the only item here
   that can produce a *wrong run* rather than untidy code. It is my
   single highest-value change.

Two premises in the brief need correcting before anything else, because they change
what is provable:

* **Premise "zero GPU execution" is wrong for the FFT service.** There is a real
  single-GPU run of the cuFFT handler:
  `wk_REL/results/logs/audit_gpu_fft.log:3` (`jax 0.9.1 platform=gpu devices=[CudaDevice(id=0)]`,
  Quadro RTX 5000 sm_75, `wk_REL/results/logs/audit_gpu.7879378.out:9`), with the handler's own
  log lines proving it executed (`audit_gpu_fft.log:11-14`, `:63-64`), plus a 15-case
  correctness gate at `wk_REL/results/logs/cufft_unit.log:11-33` and a compiled-HLO census at
  `wk_REL/results/logs/audit_gpu_hlo.log`. The repo records this at
  `docs/dev/HANDOFF_2026-07-29.md:26-28`. It **is** still true for the GEMM service and
  for `contract_bands` itself (`audit_gpu_gemm.log:3` runs on a GPU node but every row
  is `XLA:CPU dot_general` / bare MKL — no GPU GEMM was measured), which is what
  `docs/dev/staged_reshard_primitive.md:280-282` claims. Details and what it changes:
  §F.
* **Premise on `query_fft_peak_bytes` is correct, and the repo already contradicts
  itself in writing.** `src/common/fft_helpers.py:48-52` promises "input buffer +
  output buffer + **cuFFT scratch**"; `:104-110` computes only
  `memory_analysis()`. `src/runtime/aot_memory.py:3-6` states the opposite as a
  fact: "cuFFT plan workspace is allocated lazily by jaxlib's FFT thunk at first
  kernel invocation — **outside XLA's accounting**." Both files are in-tree; no
  external XLA source is needed to establish the contradiction. §F.2.

---

## A. Is there a common shape? — the divergence table

Read the rows as: everything left of the double line is what a *caller* has to know.

### A.1 The reference: `contract_bands` (GEMM dial) vs FFT vs the five contrast families

| | **contract_bands / mklblas GEMM** | **fft_helpers / mklfft+cufft (plain)** | **fft_helpers (fused conv)** | **ffi/linalg (eigh/chol/lu)** | **ffi/phdf5** | **ffi/{slate,scalapack,cusolvermp,cublasmp}** |
|---|---|---|---|---|---|---|
| Python module home | `src/common/contract_bands.py` (**not** `src/ffi/`) | `src/common/fft_helpers.py` (**not** `src/ffi/`) | same | `src/ffi/linalg/{resolve,plan,dispatch}.py` | `src/ffi/phdf5/{read,write,context}.py` | `src/ffi/<lib>/*.py` |
| Selection input | env `LORRAX_BANDS_GEMM_FFI`, read `contract_bands.py:192` | env `LORRAX_FFT_FFI`, read `fft_helpers.py:430` | env `LORRAX_FFT_FFI_FUSED`, read **`gw/ppm_tau_kernel.py:81`** — *outside the service* | **function argument** `backend=` (`ffi/linalg/plan.py:301`); **no `os.environ` anywhere in `src/ffi/linalg/`** | env `LORRAX_WFN_BACKEND` (`file_io/wfn_loader.py:371`) + capability router (`gw/gw_config.py:191`, `:255`) | none — selected by `resolve.backend_module` (`resolve.py:475`) |
| Grammar | strict 3-way `on/off/auto`, unknown → announce + OFF (`:192-204`) | strict 2-way `0/1`, unknown → announce + OFF (`:430-440`) | **none** — `in ("1","true","yes","on")`, anything else silently False (`ppm_tau_kernel.py:81-82`) | vocabulary guard raises (`resolve.py:310-316`) | `eager\|phdf5\|phdf5_host`, unknown falls through to auto | n/a |
| Has `auto` | **yes** (default) `:207-250` | **no** — default is hard OFF (`:430`) | **no** | yes (`resolve.py:335-349`) | yes (`wfn_loader.py:369-397`) | no |
| Where the platform comes from | `ffi_loader.platform_from_env` (**lexical, JAX_PLATFORMS**, `:230`) for auto **+** `mesh.devices.flat[0].platform` (`:270`, `:439`) for the refusal | `mesh.devices.flat[0].platform` only (`:452`) | same | `mesh_platform(mesh)` (`resolve.py:141-145`) | `_platform_for_mesh(mesh)` (`phdf5/context.py:44`) — **except `write.py:113` uses bare `get_lib()`** | slate/scalapack: mesh (`slate/context.py:70`); **cusolvermp/cublasmp: bare `get_lib()`** → `jax.default_backend()` (`ffi_loader.py:181`) |
| **Resolves before JAX backend init?** | **YES for the auto verdict** (that is the whole point — it is a kernel-cache key, `ppm_tau_kernel.py:233`, `:406`); rationale `contract_bands.py:214-224` | **NO** — needs a `Mesh` | NO | NO | NO | NO (slate/scalapack need a Mesh; cusolvermp/cublasmp need the default backend) |
| Explicit request that cannot be honored | **REFUSES**: non-CPU mesh `:272-278`; handler missing `:282-285` (quotes probe reason); `extra="minor"` `:423-430`; bad dtype `:467-475` | **REFUSES**: unsupported platform `:457-461`; handler missing `:480-483`; `out_spec` reshard `:547-550`; non-c128 `:572-576` | same refusals via `_require_fft_ffi(mesh, _FFT_FFI_CONV_TARGET)` `:616` | **REFUSES**, 6 guards, `resolve.py:356-437` | **honored verbatim, no capability check** — `wfn_loader.py:374-375` `return forced` | REFUSES (shape/geometry/coverage) |
| `auto` demotes… | silently on non-CPU **by design** (`:230-233`) and on `extra="minor"` (`:439-441`); **announces** the ON/OFF verdict once on rank 0 (`:236-249`) | n/a | n/a | announces the geometry demote on rank 0 (`resolve.py:239-245`) | announces on rank 0 (`wfn_loader.py:319`, `:333-335`) | n/a |
| Announce (Python) — rank-guarded? | verdict lines **yes** (`:238`, `:288`, via `_rank0()` `:175-179`); **grammar warning NO** (`:201`, no rank check in `bands_gemm_ffi_mode`) | backend line **yes** (`:487-491`); **grammar warning NO** (`:437`) | inherits fft_helpers | yes (`resolve.py:239`) | yes / caller-supplied `print_fn` (`gw_config.py:191`) | none printed at all |
| Announce (C++) — rank-guarded? | **yes**, `announce_here()` reads `SLURM_PROCID`/`PMI_RANK`/`OMPI_COMM_WORLD_RANK` (`gemm_batch_ffi.cc:229-235`); announcement is **unconditional** (not behind the log flag) `:237-271` | **NO rank guard**: `mklfft` prints per-descriptor-commit **per thread** (`fft_flat_k_ffi.cc:255-262`) and first-call (`:396-406`) under `LORRAX_MKLFFT_LOG`; `cufft` likewise (`fft_flat_k_cuda_ffi.cc:352-358`, `:420-423`, `:452-458`, `:520-529`, `:599-611`) | same | n/a | rank-guarded (`phdf5/cpp/context.cc:204`, `:408`) | `blacs_grid.h`/`eigh_ffi.cc:227` behind `LORRAX_SCALAPACK_EIGH_LOG` |
| Library located/loaded via | `ffi_loader.probe_target` (`:234`, `:280`) | `ffi_loader.probe_target` (`:478`) | same | `probe_target`/`has_target` (`resolve.py:390`, `:339`) | `ffi_loader.get_lib(platform)` | `ffi_loader.get_lib` (some bare) |
| Fallback ("two plans") | XLA `einsum` always built (`:480`, `:491`, `:502`) ✓ | XLA `custom_partitioning`/`shard_map` FFT path always built (`:703-730`) ✓ | XLA 3-FFT composition in the `else` branch (`ppm_tau_kernel.py:249-254`) ✓ | `native` pure-JAX (`resolve.py:117-121`) ✓ | `phdf5_host` → `h5py_allgather` ✓ | none (resolver owns it) |
| In-place / aliasing | **none by design** — GEMM output can't alias an operand (`:305-306`, `gemm_batch_ffi.cc:67-69`) | `input_output_aliases={0:0}` (`:564`) | `{0:0}` (`:640`) | n/a | n/a | n/a |
| Hidden (XLA-invisible) scratch | **none** — pure BLAS call, no workspace | host: process-global `Arena` under a mutex (`fft_flat_k_ffi.cc:418-426`, `:471`); CUDA: two grow-only `cudaMalloc` arenas outside the XLA allocator (`fft_flat_k_cuda_ffi.cc:401-404`, `:409-425`), cuFFT auto-alloc **off** (`:435`) | same + `V_R` arena **measured at 99.7 MB** (`wk_REL/results/logs/audit_gpu_fft.log:64`) | ScaLAPACK workspace | MPI-IO buffers | — |
| Unit gate in `tests/` | `tests/test_contract_bands.py` (19 call sites), `tests/test_projection_lgemm.py` | **NONE** | **NONE** | `tests/test_ffi_linalg_contract.py` | `tests/test_slab_io_routing.py` | via linalg contract test |
| C++-side env vars documented in `docs/dev/env_vars.md` | ✓ `LORRAX_MKLBLAS_THREADS` (:128), `LORRAX_MKLBLAS_LOG` (:192) | **✗ none of `LORRAX_MKLFFT_THREADS`, `LORRAX_MKLFFT_CHUNK`, `LORRAX_MKLFFT_LOG`, `LORRAX_CUFFT_LOG`** | ✗ | ✓ | ✓ (`PORTING.md:146-177`) | ✓ |

Negative-search provenance for the last row: `for v in ...; do grep -rln "$v" docs/ config/ scripts/ tests/; done` returned nothing for the four FFT variables and hits for both MKLBLAS variables. These are literal, unique ASCII identifiers, so grep is a sound instrument here (unlike the `strings`-on-a-header-only-template failure the brief warns about).

### A.2 The divergences that actually matter

Ordered by consequence, not by size.

1. **Resolution timing is genuinely different and must stay that way.**
   `bands_gemm_ffi_enabled()` resolves *lexically* from `JAX_PLATFORMS`
   (`contract_bands.py:230` → `ffi_loader.platform_from_env`, `ffi_loader.py:191-199`)
   because it is consumed as a **kernel-cache key** at `ppm_tau_kernel.py:233` and
   `:406`, which may run before `jax.distributed.initialize`. `ffi_loader.py:178-180`
   documents exactly why `get_lib(None)` is forbidden there. The FFT service resolves
   from a live `Mesh` (`fft_helpers.py:452`) and *also* is a cache key at the same two
   lines — but `fft_ffi_enabled()` (`:423-440`) reads only the env, never the mesh, so
   it is safe by accident, not by design. **Any unified resolver must keep the
   env-only/lexical tier separate from the mesh tier.** This is the single reason
   `ffi/linalg/resolve.py` cannot simply absorb these two.

2. **The FFT service's policy is split across two modules.**
   `LORRAX_FFT_FFI_FUSED` is read at `gw/ppm_tau_kernel.py:81` — a *consumer* — while
   `LORRAX_FFT_FFI` is read at `fft_helpers.py:430`. `fft_helpers.py:400-402` states
   the contract: "these helpers stay THE single FFT entry point (owner rule) — the
   backend switch happens here and nowhere else." That rule is violated by the fused
   flag, and the violation is not cosmetic: the fused flag has **no grammar
   validation and no announcement** (`ppm_tau_kernel.py:81-82`), so
   `LORRAX_FFT_FFI_FUSED=yes` works, `=Y` silently does nothing, and nothing is
   printed either way. Contrast the two flags that live in the service
   (`fft_helpers.py:435-440`, `contract_bands.py:199-204`).

3. **`auto` exists for GEMM, not for FFT — and nothing in the tree says why.**
   `env_vars.md:129` records the FFT default as `0 (off)` without justification, while
   `:127` gives a long rationale for the GEMM `auto`. On GPU the FFI FFT is *measured*
   2.1–2.4× faster than the XLA path at production shapes
   (`audit_gpu_fft.log:104-105`: S2 flat-k XLA 22.28 ms vs FFI 9.18 ms;
   `:119-120`: S4 21.85 vs 10.53; fused conv `:117-118`: 33.89 vs 15.62). That is a
   larger margin than the one that justified GEMM's `auto`. Whether to turn it on is
   an owner call; the *absence of a recorded reason* is the defect.

4. **Rank discipline diverges three ways within one service.** The GEMM handler
   determines rank-0 from launcher env in C++ (`gemm_batch_ffi.cc:229-235`) and from
   `jax.process_index()` in Python (`contract_bands.py:175-179`); the FFT handlers
   have **no** rank guard in C++ at all. At P=64 with `LORRAX_MKLFFT_LOG` set,
   `fft_flat_k_ffi.cc:255-262` prints once **per thread per descriptor geometry per
   rank**.

5. **Explicit-refusal timing is two-phase for both services and one-phase for
   linalg.** `plan(op, mesh, backend=, n=)` gets `n` as an argument
   (`ffi/linalg/plan.py:301`), so *every* guard fires at resolve time. FFT and GEMM
   cannot: dtype is a trace-time fact. The GEMM dtype refusal fires inside the
   `shard_map` body (`contract_bands.py:451-475`); the FFT dtype refusal fires inside
   the returned callable (`fft_helpers.py:571-576`). **This is correct and must stay**
   — but it means "refuses at resolve time" is only true of the platform/handler
   guards, and no doc says so.

---

## B. What is genuinely shared vs. duplicated

### B.1 What `src/ffi/common/ffi_loader.py` already centralizes (732 lines)

* The **target-name → C++-symbol tables**, per platform (`:92-116` CUDA, `:132-151`
  host). All three of our services are registered there:
  `lorrax_mklfft_flat_k` / `lorrax_mklfft_gw_conv` appear in **both** tables under
  different symbols (`:106-107`, `:140-142`) — the same-target/different-symbol split
  that lets `fft_helpers` issue one platform-agnostic `ffi_call`;
  `lorrax_mklblas_gemm_batch` appears in the **host table only** (`:146`), which is
  what makes a forced CUDA probe report *unknown target*.
* **`.so` location + `dlopen`** with the h5py-ordering fix (`:481-519`).
* **`probe_target(name, platform) -> (bool, reason)`** (`:384-441`) with the
  three-way reason taxonomy — this is the one genuinely shared, genuinely valuable
  primitive, and all three idioms use it.
* **`platform_from_env(default)`** (`:191-199`) — backend-init-free platform read.
  **Exactly one consumer in `src/`:** `contract_bands.py:230`. (Confirmed by
  `grep -rn "platform_from_env" src/`; the other hits are in `src/common/slate_*_test.py`
  standalone drivers.)

### B.2 What each service still hand-rolls — quantified

| Duplicated thing | Copies | Lines | Load-bearing? |
|---|---|---|---|
| rank-0 helper | 3: `contract_bands.py:175-179`, `fft_helpers.py:487-491` (inline), `ffi/linalg/resolve.py:248-254` | 5+5+7 | **No** — incidental |
| "announced once" dedupe set | 2: `contract_bands.py:171`, `fft_helpers.py:420` (+ `resolve.py:227`) | 3 | **No** |
| env-grammar parser + unknown-value warning | 3: `contract_bands.py:192-204`, `fft_helpers.py:430-440`, `ppm_tau_kernel.py:81-82` | 13+11+2 | **No** — and the 3rd copy is *defective* (no grammar, no warning) |
| probe → announce-or-refuse block | 2: `contract_bands.py:268-294`, `fft_helpers.py:470-503` | 27+34 | **No** |
| mesh → FFI platform string | 3 distinct implementations: `fft_helpers.py:443-461` (raises), `resolve.py:141-145` (passthrough), `phdf5/context.py:44`, plus `contract_bands.py:270`/`:439` inline | ~19+5+~8 | **Partly** — the FFT one *raises* on an unsupported platform, which is a policy the others don't have |
| **Python resolution total** | | **~225 lines across 3 modules** | |
| C++ MKL thread pin (`mkl_set_local_fn` + `MklLocalPin`/`MklThreadScope` + `str_ieq` + `team_threads`) | **3**: `scalapack/cpp/blacs_grid.h:157-233`, `mklfft/cpp/fft_flat_k_ffi.cc:96-159`, `mklblas/cpp/gemm_batch_ffi.cc:277-338` | 64 + 62 + ~76 | **The duplication itself is now load-bearing in a bad way** — see below |
| C++ `log_enabled()` | 3, three different env names | 4×3 | No |

**The C++ thread-pin duplication has already drifted, and the drift is behavioral.**
Diffing the two 62/64-line blocks (`sed -n '96,159p' mklfft/... ` vs
`sed -n '277,338p' mklblas/...`) gives **16 differing lines, 10 of them comment text**.
The functional differences are exactly two:

* `mklfft/cpp/fft_flat_k_ffi.cc:100` resolves the pin with
  `dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads_Local")` only.
* `mklblas/cpp/gemm_batch_ffi.cc:280` uses `resolve_sym(...)`, which falls back to
  `RTLD_NEXT` (`:139-145`).

So under a local-scope `dlopen` the GEMM handler still pins MKL and the FFT handler
silently does not. That is a real capability divergence introduced by copy-paste, and
nothing in the tree would surface it.

**The stated reason for the duplication is real, not laziness.**
`mklfft/cpp/fft_flat_k_ffi.cc:50-53` and `mklblas/cpp/gemm_batch_ffi.cc:76-80` both say
the copy exists "so this TU stays comms-free / MPI-free". Confirmed:
`scalapack/cpp/blacs_grid.h:40` includes `<mpi.h>` and `:42` includes the SLATE ctx
header. The fix is therefore **not** "include blacs_grid.h" — it is a new MPI-free
header. And the repo's own threshold has been crossed:
`src/ffi/TEMPLATE.md:194-195` — *"Extract into `common/` only when a **third** library
would copy the same code a **third** time."* There are now three copies.

### B.3 Duplication that must stay

* **cuFFT plan workspace vs. no BLAS scratch.** `fft_flat_k_cuda_ffi.cc:435` disables
  cuFFT auto-allocation and shares one grow-only arena (`:401`, `:409-425`), with a
  `cudaDeviceSynchronize()` before every growth (`:412`). The GEMM handler has no
  workspace at all (`gemm_batch_ffi.cc:67-69`). Any "unified handler base class" that
  tried to abstract this would be inventing a concept one of the two does not have.
* **The concurrency model.** cuFFT serializes plan-cache/arena/enqueue under one
  process mutex and states the single-compute-stream assumption honestly
  (`fft_flat_k_cuda_ffi.cc:72-78`, `:519`, `:598`); the MKL FFT parallelizes its chunk
  loop with OpenMP and pins MKL to 1 thread inside (`fft_flat_k_ffi.cc:332-337`); the
  GEMM makes one internally-threaded BLAS call (`gemm_batch_ffi.cc:407`). Three
  different, each measured. Do not unify.
* **The MKL compact-chunk staging** (`fft_flat_k_ffi.cc:299-315`, measured 2.8×) is
  explicitly *not* mirrored on CUDA (`fft_flat_k_cuda_ffi.cc:20-23`: "the host
  engine's per-thread L2 buffer was a CLX cache artifact, not part of the contract").
  That is exactly the right call and the right way to say it.

---

## C. Can they share one service shape?

**Yes — one shared *resolver*, two shared *wrapper modules*, and NO shared handler
abstraction.** Concretely:

### C.1 What to build: `src/ffi/common/gate.py` (~120 lines)

```python
# src/ffi/common/gate.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Gate:
    """One env-gated FFI capability.  Owns grammar, probe, announce, refuse."""
    env: str                      # "LORRAX_FFT_FFI"
    target: str                   # "lorrax_mklfft_flat_k"
    platforms: tuple[str, ...]    # ("cpu", "CUDA") | ("cpu",)
    default: str                  # "off" | "auto"
    label: dict[str, str]         # platform -> human backend name
    why_on: str                   # one line, measured, for the AUTO-ON announce

    def mode(self) -> str:                       # "on"|"off"|"auto"; strict grammar,
        ...                                      # unknown -> announce once + "off"

    def enabled_lexical(self) -> bool:           # NEVER touches the JAX backend.
        """Cache-key safe.  auto := mode=='auto' and platform_from_env() in
        self.platforms and probe_target(...) ok.  Announces once on rank 0."""

    def resolve(self, mesh, *, extra_refusals=()) -> str | None:
        """Mesh-aware resolution.  Returns the platform string to lower on, or
        None if the gate is OFF.  Under mode=='on' RAISES with the probe reason;
        under 'auto' returns None silently for a platform outside self.platforms
        and announces any other demote on the rank it happens on."""
```

Backing helpers (also shared, replacing the 3 rank-0 copies and 2 dedupe sets):
`gate.rank0()`, `gate.announce_once(key, msg)`.

**Adoption, exactly:**

| Site | Change |
|---|---|
| `contract_bands.py:166-294` | delete 125 lines; keep `bands_gemm_ffi_mode/enabled` as 3-line delegations (they are public API, `__all__` at `:159-163`, and `tests/test_contract_bands.py:56-57` imports them) |
| `fft_helpers.py:418-503` | delete 86 lines; keep `fft_ffi_enabled()` as a delegation |
| `ppm_tau_kernel.py:70-83` | delete; replace with a `Gate("LORRAX_FFT_FFI_FUSED", "lorrax_mklfft_gw_conv", ...)` **declared in the FFT service module**, restoring the `fft_helpers.py:400-402` "switch happens here and nowhere else" rule |
| `ffi/linalg/resolve.py:227-254` | reuse `gate.rank0()` / `announce_once` (optional, −20 lines) |

**Net Python: about −225 +120 = −105 lines, and the number of resolution idioms goes
4 → 2** (arg-driven `plan()` for distributed solvers; env-driven `Gate` for
rank-local dials). Risk: low — the behavior is currently identical modulo the three
defects the consolidation *fixes* (fused-flag grammar, unguarded grammar warnings,
drifted rank-0 helpers). `tests/test_contract_bands.py` (19 primitive call sites,
13 `LORRAX_BANDS_GEMM_FFI` mutations) already pins the GEMM half.

### C.2 What to move: the two missing wrapper modules

Per `src/ffi/AGENTS.md:149-150` and `TEMPLATE.md:10-20`:

```
src/ffi/mklfft/__init__.py       re-export flat_k_fft, gw_conv
src/ffi/mklfft/flat_k.py         the shard_map(ffi_call) bodies  ← fft_helpers.py:530-656
src/ffi/mklblas/__init__.py      re-export gemm_batch
src/ffi/mklblas/gemm.py          the shard_map-free ffi_call     ← contract_bands.py:296-319
```

`src/common/fft_helpers.py` and `src/common/contract_bands.py` then keep *only* their
math and their public factories, and import the FFI bodies like every other family
does. Pure move (~150 lines relocated, 0 net), and it is the change that makes the
word "microservice" literally true: after it, all 21 `ffi_call` sites in `src/` live
under `src/ffi/`.

`src/ffi/cufft/` needs no Python module — it deliberately registers the *same target
strings* as `mklfft` (`ffi_loader.py:99-107`, `fft_flat_k_cuda_ffi.cc:654-656`), so
one `mklfft/flat_k.py` serves both platforms. That design is correct and should be
called out in the doc rather than mirrored in Python.

### C.3 What to build in C++: `src/ffi/common/cpp/mkl_thread_pin.h` (~55 lines)

`resolve_sym` (the `RTLD_DEFAULT → RTLD_NEXT` form from
`gemm_batch_ffi.cc:139-145`, i.e. the *better* of the two), `str_ieq`, `MklLocalPin`,
and `team_threads(const char* env_name)`. MPI-free by construction, so all three
consumers can include it — `blacs_grid.h:157-233` included. Net **≈ −135 lines** of
C++, and it closes the `RTLD_NEXT` drift.

### C.4 What CANNOT be unified — with named reasons

| Proposal | Verdict | Reason |
|---|---|---|
| Fold FFT/GEMM into `ffi/linalg/resolve.py`'s `OPS` | **NO** | Guards 4 (one JAX process per device, `resolve.py:399-407`) and 5 (mesh geometry, `:410-430`) exist because those backends hold an MPI/BLACS/NCCL communicator. FFT and GEMM are rank-local handlers *inside* a `shard_map` with no communicator; both guards would be permanent no-ops, and `resolve_backend`'s promise ("a returned FFI name is a *promise*", `:261-264`) would become weaker for two of its ops than the others. |
| One `plan()`-style object for all five | **NO** | `plan()` takes `n` and resolves everything at once (`plan.py:301`). FFT/GEMM dtype and shape are trace-time facts (`fft_helpers.py:571-586`, `contract_bands.py:451-475`). A single-phase API would have to lie. Keep the two-phase contract and *document* it. |
| A shared C++ "FFI handler base" | **NO** | The three handlers' scratch models are structurally different (§B.3) and each was measured. `TEMPLATE.md:188-195` is right about this. |
| Give the FFT service a `local_*` FFI entry (callable inside a `shard_map`) | **Not now, but this is the real gap** | See §D.1. It is the only change here that would need new measurement rather than refactoring. |

---

## D. Are the call-site interfaces clean?

### D.1 **Defect 1 (worst): the FFT service has two entry layers and the gate only reaches one.**

`fft_helpers.py` exposes two families:

* `make_flat_k_*` — wraps its own `shard_map` (`:567`, `:643`, `:715-719`), and **is**
  FFI-gated.
* `local_fftn3` / `local_ifftn3` (`:315`, `:324`) — bare `jnp.fft` aliases, for code
  *already inside* a `shard_map` (shard_map cannot nest), documented at `:306-314` as
  "ONE source for the local FFT". **These have no FFI route at all.**

Consequence, measured by grep:

* `src/isdf/core.py:35-39` **imports `make_flat_k_ifftn`/`make_flat_k_fftn` and never
  calls them** (only hits are the import lines and a comment at `:268`). It makes six
  raw `jnp.fft` calls instead: `:355`, `:360`, `:373`, `:845`, `:850`, `:861`.
  The reason is stated and is legitimate — `:262-272`: the pair-density pipeline is
  one monolithic `shard_map`, and a decomposed chain costs 21 GiB vs 13 GiB of
  BufferAssignment slots (`:273-280`).
* `src/common/wfn_transforms.py` likewise: raw `jnp.fft` at `:536`, `:709`, `:975`,
  `:1266`.
* `docs/dev/env_vars.md:129` already documents the *symptom* ("the flag reaches ONLY
  `make_flat_k_*` call sites") without naming the *cause* (the layer split).

So the "ONE source for the local FFT" contract holds only for the shard_map-wrapping
layer, and `LORRAX_FFT_FFI` structurally cannot reach ISDF's ζ-fit or the wavefunction
transforms. **This must stay as it is until someone adds a shard_map-free FFI entry**
— and doing that requires deciding whether a stride-descriptor FFT is even the right
thing at those layouts, which is a measurement, not a refactor. Naming it in the doc
is the immediate fix.

### D.2 Defect 2: an env var read at a consumer

`gw/ppm_tau_kernel.py:81` reads `LORRAX_FFT_FFI_FUSED`. Already covered in §A.2.2; it
is a direct violation of the service's own stated rule at `fft_helpers.py:400-402`,
and it is where the grammar/announce discipline was lost.

### D.3 Defect 3: `query_fft_peak_bytes` — a false contract with a silent fallback

Three problems in one 72-line function:

1. **The docstring promises what the code cannot deliver.** `:48-52` says the return
   includes "cuFFT scratch"; `:104-110` sums `temp + argument + output − alias` from
   `memory_analysis()`. `src/runtime/aot_memory.py:3-6` states in-tree that cuFFT plan
   workspace is outside XLA's accounting. Under `LORRAX_FFT_FFI=1` it is *doubly*
   invisible: `fft_flat_k_cuda_ffi.cc:435` turns cuFFT auto-allocation off and
   `cudaMalloc`s its own arena (`:418`), plus a second `V_R` arena
   **measured at 99.7 MB** at production shape (`wk_REL/results/logs/audit_gpu_fft.log:64`).
2. **On failure it silently returns a guess.** `:84-102`: any exception → `3 ×
   data_size / n_devices`, cached, returned. The comment says "Logged so the caller
   notices" — **there is no log statement**. The caller then swallows it again:
   `gw/gflat_memory_model.py:148-152` wraps the call in `except Exception: pass` and
   falls back to `_c128(...) * _FFT_CUFFT_FACTOR` (`:79`, `= 4.0`). So a GPU chunk
   planner can silently size on a 3× or 4× fudge factor while the docstring at
   `gflat_memory_model.py:133-137` says "Queries XLA **exactly** for the cuFFT plan
   scratch".
3. **The function that actually does the job has zero production callers.**
   `src/runtime/aot_memory.py:472` `aot_kernel_peak_bytes` — which parses the HLO for
   `fft` ops and calls `cufftGetSize` on jaxlib's own libcufft (`:26-34`) — is
   referenced only by `tests/test_aot_memory.py:160,172`, `scripts/profiling/
   aot_cufft_sanity.py:50` (a private helper), and prose. Verified by
   `grep -rn "aot_kernel_peak_bytes" .` — zero hits in `src/`. Meanwhile
   `docs/architecture/memory-model.md:687-689` states as fact: *"That single real
   number is measured live by `src/runtime/aot_memory.py` (`aot_kernel_peak_bytes`, a
   `cufftGetSize*` query over the compiled HLO) and added to the V_q tile chooser's
   estimate."* **It is not.**

**What this means for the FFT service's interface contract:** the service currently
publishes *three* entry points — `make_flat_k_*`, `make_flat_k_gw_conv`, and
`query_fft_peak_bytes` — and only the first two have an honest contract. A memory
query is a legitimate part of a microservice interface (the caller needs to size
around it), but it must return a *breakdown*, not one number, because the FFT service
has three distinct memory classes: XLA-visible buffers, XLA-invisible cuFFT plan
workspace, and (under the FFI gate) XLA-invisible LORRAX arenas. Exactly the shape
`AotPeakBreakdown` (`aot_memory.py:76`) already has.

### D.4 Other call-site observations

* **No boolean traps, no leaked backend identity.** `make_flat_k_fft` returns the same
  `(nk,*trail) -> (nk,*trail)` callable either way (`:696-730`); `contract_bands`
  returns the same `project(ψ_left, O, ψ_right)` either way. Callers never branch on
  which backend they got. Verified across all `make_flat_k_*` call sites
  (`gw/cohsex_sigma.py:109-111`, `gw/w_isdf.py:95-97`, `gw/ppm_tau_kernel.py:248-254`,
  `bandstructure/htransform.py:572`) — **none passes `out_spec`**, so the FFI
  reshard refusal (`fft_helpers.py:546-550`) is currently unreachable in production.
  Good.
* **Cache keys are correct.** Both `ppm_tau_kernel.py:232-233` and `:405-406` include
  all three gate reads, per `staged_reshard_primitive.md:86-88`.
* **Dead public API in the FFT service.** `make_jittable_local_ifftn_3d` (`:267`) has
  **zero** call sites anywhere; `make_jittable_local_fftn_3d` (`:287`) has exactly one
  — inside `query_fft_peak_bytes` (`:78`). A public factory whose only consumer is the
  module's own memory probe should be private or deleted before release.
* **`compute_block_size_for_2d_cholesky` (`fft_helpers.py:113`) is not an FFT
  function** and lives in the middle of the FFT module. Cosmetic, but it is the kind
  of thing a release reviewer will flag.

---

## E. Documentation

`docs/dev/staged_reshard_primitive.md` (512 lines) is the standard: it has a §1 API
contract with the operand/sharding table and a numbered refusal list, a §2 raison
d'être stated as a *structural* invariant, a §3 of techniques each with its evidence
pointer and its measured domain, a §4 "how to gate any change", a §5 "next consumer",
and an honest amendment recording a withdrawn model. That last property — recording
what was *falsified* — is the hardest part to reproduce and the most valuable.

Two things to fix in the standard itself before cloning it:

* **§3.4 is stale on the mechanism it describes at greatest length.** Lines 194-254
  describe a CMake `check_symbol_exists` probe, its `--allow-shlib-undefined` fix, and
  tell the reader to check a configure log line. That probe was **deliberately
  removed**: `src/ffi/common/cpp/host/CMakeLists.txt:333-345` — *"NO FEATURE PROBE
  HERE — deliberate (owner order 2026-07-29)"* — and the choice is now runtime `dlsym`
  per precision (`gemm_batch_ffi.cc:7-40`, `:188-202`). The configure line the doc
  tells you to read no longer exists; the current one is at `CMakeLists.txt:366-370`.
  (Note the doc file is uncommitted-modified in the working tree, so this may be
  mid-edit.)
* **`docs/dev/` is excluded from the shipped site** (`mkdocs.yml:29-31`,
  `exclude_docs: | dev/`). For a public release, "its own small documentation" needs a
  decision about whether these live in `docs/dev/` (agent-facing) or under
  `docs/architecture/` (user-facing). The FFI-native-libs page already exists in the
  nav (`mkdocs.yml`, Installation section).

### E.1 `docs/dev/flat_k_fft_service.md` — table of contents (sketch)

1. **API contract** — the three factories (`make_flat_k_ifftn/fftn`,
   `make_flat_k_gw_conv`) with their `(nk,*trail)` shape contract, the `spec` rule
   (leading three k-axes replicated), norm conventions and where they are computed
   (`fft_helpers.py:505-515`), and the numbered refusal list split into
   **factory-time** (platform, handler) vs **trace-time** (dtype, rank, extent) —
   the two-phase contract §A.2.5 says must be written down.
2. **Raison d'être, per platform, honestly** — the CPU layout-anchor argument with
   its measurement; and the *separate* GPU argument (§F.1), because the mechanism
   claim currently asserted in `fft_flat_k_cuda_ffi.cc:7-11` is not what the GPU HLO
   census shows.
3. **The two entry layers and why the gate reaches only one** (§D.1) — with the ISDF
   and `wfn_transforms` sites named, and the shard_map-nesting reason stated as
   structural, not as a TODO.
4. **Platform mirror** — same target strings, different symbols
   (`ffi_loader.py:99-107` vs `:140-142`), MKL DFTI strides vs `cufftPlanMany64`
   advanced layout, NVRTC-without-nvcc and why (`fft_flat_k_cuda_ffi.cc:46-55`).
5. **Memory contract** — the three memory classes; what `query_fft_peak_bytes` does
   and does not include; the arena sizes with the measured 99.7 MB; the
   single-compute-stream assumption (`fft_flat_k_cuda_ffi.cc:72-78`).
6. **Threading / chunking** — `LORRAX_MKLFFT_THREADS`, `LORRAX_MKLFFT_CHUNK`,
   `LORRAX_MKLFFT_LOG`, `LORRAX_CUFFT_LOG` (none currently documented anywhere), the
   L2-chunk rationale and the deliberate non-mirroring on CUDA.
7. **In-place / donation** — `{0:0}`, when XLA grants it, both paths.
8. **How to gate a change** — parity class (value-level ~1e-15, never bit-exact),
   the unit-gate cells that exist in `wk_REL/` but **not in `tests/`**, HLO pins,
   cache-cold rule.
9. **Evidence index** — CPU jobs; GPU: `audit_gpu_fft.log`, `cufft_unit.log`,
   `audit_gpu_hlo.log`, job 7879378, with the measured domain (1× RTX 5000, sm_75,
   single device, standalone bench — **not** the production Σ driver, **not** multi-GPU).
10. **Open questions / what is not claimed.**

### E.2 `docs/dev/vendor_gemm_service.md` — table of contents (sketch)

1. **API contract** — `C[i] = A[i] @ B[i % BB]`, row-major NN, `BA % BB == 0`, the
   four precisions, no `input_output_aliases` and why.
2. **Why it is not a general GEMM service** — it is the body of one contraction in one
   primitive (`contract_bands.py:477-497`); everything else is untouched. This is the
   section that keeps the next agent from "adopting" it everywhere.
3. **The `auto` contract** — the lexical `JAX_PLATFORMS` read, why it must not
   initialize the backend, the two known lexical-miss cases and their identical fix
   (currently only in `env_vars.md:127` — it belongs here).
4. **Refusals** — the four explicit-mode refusals with their reasons.
5. **Vendor portability** — runtime `dlsym`, **per precision**, why the build-time
   probe was removed (jobs 7879278/7879281), what the unconditional first-use
   announcement guarantees. *This is the section that replaces the stale
   `staged_reshard_primitive.md:194-254`.*
6. **Threading** — `LORRAX_MKLBLAS_THREADS`, the `MklLocalPin`, why the AW cliff does
   not apply here (`gemm_batch_ffi.cc:76-80`).
7. **Platform reach** — host table only; what a GPU user sees (nothing).
8. **Gating** — 1e-12 value-parity class, the `tests/test_contract_bands.py` HLO pins,
   jobs 7879008/7879010.

Both docs should be ≤ 200 lines. `staged_reshard_primitive.md` is 512 because it
documents a *primitive with physics in it*; these two document *mechanisms*.

---

## F. The two premises, corrected

### F.1 GPU execution exists for FFT — and one GPU claim in the source is not supported by the project's own HLO census

**What was actually run** (job 7879378, 2026-07-29 02:05–02:12, node c196-012,
1× Quadro RTX 5000 sm_75, `src@0dd94a8`, `audit_gpu.7879378.out:1-13`):

* `wk_REL/results/logs/cufft_unit.log:11-33` — 15 correctness cells, all PASS, max rel err
  0–3.7e-16, incl. the fused `gw_conv` vs decomposed composition (2.87e-16), both
  donation modes, and all four norm conventions on an odd (3,2,1)×(5,7,3) grid. NVRTC
  compiled a real cubin for sm_75 (`:10`). This is a genuine GPU gate.
* `wk_REL/results/logs/audit_gpu_fft.log` — timing at production shapes (S2 = the 398.7 MB Σ τ
  G-tile). FFI beats XLA on flat-k by 2.1–2.4× and on the fused conv by 2.2×
  (`:104-105`, `:117-120`).
* `wk_REL/results/logs/audit_gpu_hlo.log` — a compiled-HLO census on GPU.

**What the HLO census actually says, and the caveat.** At every flat-k site it reports
`transpose ops: 0 / copy ops: 0` and the verdict line *"NO LARGE LAYOUT TRANSPOSE"*
(`audit_gpu_hlo.log:9-55`). Taken at face value that would refute the premise stated in
`src/ffi/cufft/cpp/fft_flat_k_cuda_ffi.cc:7-11` ("same anchor as the host handler …
so XLA transposes the tile before/after EVERY fft"). **But the census is a defective
instrument for this question**: the same rows report `fusion ops: 1` and
`XLA temp bytes: 398.7 MB` for the flat-k site (`:50-53`) against
`fusion ops: 0` / `XLA temp bytes: 0.0 MB` for the minor-most twin at identical
element count (`:66-71`). A 398.7 MB temp produced by a fusion **is** the layout
materialization; the census only counts `opcode == transpose|copy`, so its verdict
line is wrong by construction.

The defensible statement is therefore: **on GPU the layout cost is real (398.7 MB temp
+ the measured 22.28 vs 8.90 ms flat-k-vs-minor gap, `audit_gpu_fft.log:104`, `:108`)
but it is emitted as a fusion, not as a `transpose`.** The C++ header's mechanism
sentence should be reworded and the census script's verdict line fixed — it is exactly
the "void instrument" class already known to this project.

**Bonus, from the same census:** Part 4 (`:143-180`) confirms the de-promotion policy
matters on GPU too — the real×complex case shows `XLA temp bytes: 402.9 MB` with one
fusion, against 4.2 MB for both the pure-c128 and the f64-split forms, all three
dispatching `__cublas$gemm`. That is GPU evidence *for* `contract_bands` policy 3
(`contract_bands.py:507-517`) that `staged_reshard_primitive.md:280-282` does not
know it has.

**What still needs a real GPU run** (mark these clearly in any doc):

| Conclusion | Status |
|---|---|
| cuFFT handler is correct on GPU | **Measured**, 1 device (`cufft_unit.log`) |
| cuFFT handler is ~2× the XLA path at production shapes | **Measured**, 1 device, standalone bench |
| cuFFT plan workspace is 0 for these strided Z2Z plans | **Measured** — `workspace=0.0 MB` at every plan (`audit_gpu_fft.log:12,29,44,72,83`). Note this is the *FFI handler's* plan, **not** XLA's `FftThunk` plan, which uses a different layout. Says nothing about XLA's scratch. |
| `V_R` arena is ~100 MB and invisible to `memory_analysis()` | **Measured** (`audit_gpu_fft.log:64`) |
| The service behaves under **multi-GPU / sharded meshes** | **Unmeasured** — every GPU log is `[CudaDevice(id=0)]` |
| The service behaves inside the **production Σ driver** on GPU | **Unmeasured** — all GPU runs are standalone benches |
| `contract_bands` primitive on a CUDA mesh | **Unmeasured** — `staged_reshard_primitive.md:280-282` is still accurate |
| GEMM dial on GPU | **Not applicable and never will be** — host table only (`ffi_loader.py:146`); `audit_gpu_gemm.log` measured host BLAS on a GPU node |

### F.2 The memory-model claim — see §D.3. Summary of what is provable in-tree without XLA sources:

* `fft_helpers.py:48-52` promises cuFFT scratch; `:104-110` computes only
  `memory_analysis()`.
* `aot_memory.py:3-6` asserts, in the same repo, that cuFFT workspace is outside XLA's
  accounting.
* Under the FFI gate the workspace is provably outside it anyway
  (`fft_flat_k_cuda_ffi.cc:435` + `:418`).
* `aot_kernel_peak_bytes` (`aot_memory.py:472`) — the function that does query
  `cufftGetSize` — has **zero** callers in `src/`, while
  `docs/architecture/memory-model.md:687-689` says it is wired into the V_q chooser.

---

## G. Recommendations, ranked by value / risk

| # | Change | Value | Risk | Files |
|---|---|---|---|---|
| **1** | **Fix the FFT memory contract.** Correct the `query_fft_peak_bytes` docstring; make the `except Exception` path actually log (or raise under an explicit flag) instead of silently returning 3×; remove the `except Exception: pass` at `gflat_memory_model.py:149`; return/propagate an `AotPeakBreakdown`-shaped result with the three memory classes separated; wire `aot_kernel_peak_bytes` in or delete the claim at `memory-model.md:687-689`. | **Highest** — this is the only item that can produce a wrong run (GPU OOM or a needlessly small chunk), and it is on the least-evidenced path | **Low** — 3 files, no lowering change, no gate needed beyond the existing `tests/test_aot_memory.py` | `common/fft_helpers.py:39-110`, `gw/gflat_memory_model.py:130-152`, `runtime/aot_memory.py`, `docs/architecture/memory-model.md` |
| 2 | **Move `LORRAX_FFT_FFI_FUSED` into `fft_helpers`** with a strict grammar and a rank-0 announce, matching the other two flags. | High — restores the service's own stated rule; fixes a silently-ignored value class | Very low — one flag, one cache key already keyed on it | `common/fft_helpers.py`, `gw/ppm_tau_kernel.py:70-83` |
| 3 | **`src/ffi/common/cpp/mkl_thread_pin.h`** — one MPI-free header, three consumers. Closes the `RTLD_DEFAULT`-vs-`RTLD_NEXT` drift. The repo's own threshold (`TEMPLATE.md:194-195`) is met. | High | Low — mechanical; C++ rebuild + existing gates | `ffi/common/cpp/`, `ffi/mklfft/cpp`, `ffi/mklblas/cpp`, `ffi/scalapack/cpp/blacs_grid.h` |
| 4 | **`src/ffi/common/gate.py`** (§C.1). −105 lines, 4 idioms → 2. | High | Low-medium — touches the two cache-key functions; `tests/test_contract_bands.py` covers the GEMM half, the FFT half has **no test at all** (write one first) | `ffi/common/gate.py` (new), `common/contract_bands.py`, `common/fft_helpers.py` |
| 5 | **Move the FFI bodies under `src/ffi/{mklfft,mklblas}/`** per `AGENTS.md:149-150`. Makes "microservice" literally true. | Medium-high (release hygiene; a reviewer *will* notice the empty `cpp`-only dirs) | Low, but a wide import diff | ~4 new files, 2 edited |
| 6 | **Write the two docs** (§E.1, §E.2) and fix `staged_reshard_primitive.md:194-254`. | Medium-high | None | `docs/dev/` |
| 7 | **Add a `tests/` unit gate for the FFT service.** Today `tests/` never imports `common.fft_helpers` (verified: `grep -rn "fft_helpers" tests` → nothing). The gate exists only in `wk_REL/`, which does not ship. | Medium-high for a public release | Low | `tests/test_fft_flat_k.py` (new) |
| 8 | Rank-guard the C++ FFT log output (`announce_here()` from `gemm_batch_ffi.cc:229-235`); document the four undocumented `LORRAX_MKLFFT_*` / `LORRAX_CUFFT_LOG` vars in `env_vars.md`. | Medium | None | 2 `.cc`, 1 `.md` |
| 9 | Reword `fft_flat_k_cuda_ffi.cc:7-11` (fusion, not `transpose`) and fix the HLO census verdict line to count fusion-produced temps. | Medium (a wrong claim in a source header) | None | 1 `.cc`, 1 `wk_REL` script |
| 10 | Decide and **record** whether `LORRAX_FFT_FFI` should get an `auto` like the GEMM dial, given the 2.1–2.4× measured GPU margin. Owner call. | Owner-dependent | Medium (changes what runs by default) | — |
| 11 | Make `make_jittable_local_ifftn_3d` (0 callers) / `make_jittable_local_fftn_3d` (1 internal caller) private or delete; move `compute_block_size_for_2d_cholesky` out of the FFT module. | Low-medium (release tidiness) | Very low | `common/fft_helpers.py` |
| — | **Do NOT** fold FFT/GEMM into `ffi/linalg/resolve.py`; **do NOT** build a shared C++ handler base; **do NOT** unify the three scratch/concurrency models. Reasons in §C.4 and §B.3. | | | |

**Do #1 first.** Everything else on this list improves the code's shape; #1 is the only
one that changes whether a GPU run is correctly planned, and it is the item the brief
already half-identified. It is three files and needs no new measurement — the
contradiction is provable from two files already in the tree.
