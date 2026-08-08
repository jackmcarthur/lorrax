# Distributed dense linear algebra — internals reference (historical `ffi.linalg`)

> **START SOMEWHERE ELSE.** If you are reading about distributed linalg for
> the first time, read
> [`docs/distributed_linalg.md`](../distributed_linalg.md) — why JAX cannot
> do these operations, what the layer is, how backends are chosen, and what
> the measurements say. The contract is
> [`docs/services/distrib_la.md`](../services/distrib_la.md). **This page is
> the internals reference and the recorded campaign reasoning**, kept for the
> C++ side and the design history.

> **RENAME MAP (2026-08-07). This page is history below "Adding a
> backend"; the live reference is [`docs/services/distrib_la.md`](../services/distrib_la.md).**
> The stack moved into `services/distrib_la/` and the four packages this
> page is written against were **deleted** (commit `b3f3675`, 17 files).
> Everywhere below, read:
>
> | this page says | it is now |
> |---|---|
> | `ffi.linalg` / `from ffi import linalg` | `distrib_la` (the package IS the door) |
> | `ffi/linalg/plan.py` (`plan`, `Plan`, `_IMPL`, `ensure_sharding`) | `distrib_la.plan` |
> | `ffi/linalg/resolve.py` (`resolve_backend`, `list_backends`, `backend_module`, `BACKEND_CHOICES`, `_SPEC`) | `distrib_la.resolve` |
> | `ffi/linalg/dispatch.py` (`dispatch_batched_eigh`) | `distrib_la.dispatch` |
> | `ffi/linalg/_slate.py` / `_scalapack.py` | `distrib_la._slate` / `._scalapack` |
> | `ffi/linalg/_slate._mesh_key` | `distrib_la.mesh_key` — **public now**, and there is one of it |
> | `ffi.slate` / `ffi.scalapack` (the wrapper packages) | `distrib_la._slate` / `._scalapack`; the README is `services/distrib_la/docs/slate_backend.md` |
> | `common.cholesky_2d` (`cholesky_2d_batched`, `dense_to_tiles`, `tiles_to_dense`) | `distrib_la._native2d`, reached as `plan('cholesky', mesh, backend='native2d')` |
> | `ffi/common/ffi_loader.py`'s linalg half | `distrib_la.loader` (lorrax's loader keeps the FFT/GEMM/phdf5 half) |
> | `tests/test_ffi_linalg_contract.py` | `services/distrib_la/tests/test_distrib_la_contract.py`, marker `distrib_la` |
>
> Prose and campaign history below has NOT been rewritten path-by-path —
> rewriting 500 lines of recorded reasoning to change a module name loses
> more than it fixes. This table is the translation. Anything that
> **instructs** you to do something has been rewritten in place.

*A guide to LORRAX's distributed-linalg stack, written for a JAX power
user — the facade is deliberately small and self-contained enough to
lift into another SPMD JAX codebase.*

Convention used across the dev docs: `wk_REL/` = `/scratch2/08271/jackmc/lorrax_setup/wk_REL/` (campaign evidence, machine-local, not shipped).

## The problem

LORRAX runs multi-process SPMD JAX: one process per device, a 2-D
`('x','y')` `Mesh`, `shard_map` + `NamedSharding` (`check_rep=False`),
FP64/complex128 throughout. Three dense-linalg operations sit on the
hot path with matrices that range from "trivially per-device" to "does
not fit on any single device":

| op | where it is used | matrix |
|---|---|---|
| `eigh` | BSE/htransform: fH_q band interpolation (`bandstructure/bse_setup.py`), coarse exchange tiles C_q (`bse/vq_interp.py`) | (rank, rank), rank ≈ nspinor·n_μ (10²–10⁴) |
| `cholesky` | GW ζ-fit, charge channel (`isdf/core.py`) | (nq, n_μ, n_μ) batched HPD |
| `solve_lu` | GW ζ-fit, transverse channels (`isdf/core.py`); the W Dyson solve (`gw/w_isdf.py`, `w_dyson_solver = distributed`) | (nq, n_μ, n_μ) batched Hermitian-indefinite |

XLA gives you excellent *batched per-device* factorizations but no
distributed single-tile ones. Vendor/HPC libraries (cuSOLVERMp, SLATE,
ScaLAPACK) give you distributed block-cyclic factorizations but each
with its own platform, mesh-geometry, and build constraints — and a
constraint violation can mean a silent *deadlock inside a collective*,
not an error. The facade exists so that every one of those constraints
is checked in ONE place, at **resolve time**, before any collective is
entered.

## Architecture

```
input file (cohsex.in)         CLI flags            env
  distributed_cholesky =…   --eigh-backend …   LORRAX_FFI_HOST_SO…
  distributed_lu       =…         │                  │
  eigh_backend         =…         │                  │
        └────────────┬────────────┘                  │
                     ▼                               ▼
   ffi.linalg.resolve_backend(op, requested, mesh, n=…)   ← ALL guards
                     │             ▲
                     │             │ capability probe:
                     │             │ ffi_loader.has_target(target, platform)
                     ▼             │ (dlopens the .so, checks the symbol)
        "native" | "cusolvermp" | "slate" | "scalapack"
                     │
                     ▼
   ffi.linalg.plan(op, mesh, backend=…, n=…) -> LinalgPlan
        plan.is_native            ← "you own this route"
        plan(A)                   ← ONE tile, resharded inside
        plan.batched(A)           ← a stack, uniform across backends
        plan.describe()           ← one line for the run banner
                     │
                     ▼
   ffi.linalg.backend_module(name).<distributed op>   ← the ONE import seam
```

Four layers, one module each in `src/ffi/linalg/`:

* **`resolve.py`** — vocabulary, guard ladder, `resolve_backend`,
  `list_backends`, `backend_module`, `mesh_is_cpu`/`mesh_platform`.
* **`plan.py`** — `plan()` → `LinalgPlan`: **the call-site interface**.
  Resolves once and carries the answer together with the layout
  contract, the operand reshard, the batch behaviour and the per-backend
  output conventions.  See "The plan API" below.
* **`dispatch.py`** — `dispatch_eigh`, the older single-call entry point,
  now a thin shim over a plan.  Kept for back-compat; new code takes a
  plan.  Plus `dispatch_batched_eigh`, which is *not* back-compat: it is
  the one place the **batching asymmetry** is handled.  Only ScaLAPACK
  exposes `batched_distributed_eigh`, so a caller that names it directly
  works on a host mesh and `AttributeError`s on CUDA.  The capability
  test is a `getattr` on the resolved module, not a platform test, so a
  backend that gains a stacked entry later is picked up with no edit;
  the fallback walks the batch through `plan(A[q])`, which is what
  applies the per-backend eigenvector-layout normaliser (calling
  `mod.distributed_eigh` directly returns cuSOLVERMp's RAW buffer —
  rows, not columns).  Gated by
  `tests/multi_device/batched_eigh_dispatch_gate.py`; see that function's
  docstring for the donation and `lax.scan` rulings and their job ids.
* **backends** — `ffi/cusolvermp/` (CUDA), `ffi/slate/` (CUDA + host),
  `ffi/scalapack/` (host), each a thin `shard_map`+`jax.ffi` wrapper
  over one C++ handler in `liblorrax_ffi.so` / `liblorrax_ffi_host.so`;
  plus the **native** in-tree implementations (pure JAX), which are
  first-class backends, not fallbacks of last resort — they are the
  *measured default* at production tile sizes.

The GW ζ-fit keeps a channel-specific policy layer ON TOP of the facade
(`isdf/core.py::_resolve_channel_ladder`, `_resolve_solver_kind_charge`,
`_resolve_solver_kind_transverse`): the replication cap, the
rank-truncation refusal, and the charge/transverse route strings live
there, while platform/capability/coverage/geometry guarding is delegated
to `ffi.linalg.resolve_backend` (the explicit `slate`/`scalapack`
handlers call it directly).

## The guard ladder (resolve time, fixed order)

1. **vocabulary** — is the name a backend of this op at all?
2. **platform** — cusolvermp is CUDA-only; scalapack is host-only;
   slate has handlers for both platforms.
3. **capability** — `ffi_loader.probe_target(target, platform)`: is the
   handler actually *usable*? Requesting a missing one fails **here**,
   with a message listing what IS available — not minutes later at the
   first distributed call.

   > `probe_target` returns `(usable, reason)` and separates three cases
   > that have three different fixes: **unknown target** (typo / wrong
   > platform), **the library would not load** (missing `.so`, a library it
   > needs not found, glibc/GLIBCXX mismatch, wrong `LD_LIBRARY_PATH` — the
   > handler may be perfectly well compiled), and **loaded but does not
   > export the symbol** (the genuine partial build — the only one that
   > means "rebuild"). `has_target` is the bool version, for auto-pick
   > fallback logic; anything that reports a refusal to a human must use
   > `probe_target` and quote the reason.
   >
   > This distinction was added after wk_P G4 (2026-07-25) refused
   > `cholesky`+`slate` on a **legal 8×1 mesh** with "not compiled into
   > the cpu FFI library", while `nm -D` showed `SlatePotrfHostFfi`
   > present and workstream L had run that exact handler at 0.111 s. The
   > real cause was an incomplete `LD_LIBRARY_PATH`; the old message sent
   > you to rebuild a library that was fine. **The unified host lib's own
   > built-in search path covers MKL, SLATE's `lib64` and Intel MPI's
   > `lib/release`,
   > but NOT `libhdf5.so.310`, `libfabric`, or the Intel compiler runtime
   > (`libimf`/`libsvml`/`libintlc`/`libirng`) — those must be on
   > `LD_LIBRARY_PATH`.** Check with `ldd <so> | grep 'not found'`.
4. **process coverage** — the FFI backends run ONE JAX process per
   device (their MPI/NCCL context is per-process). A single-process
   multi-device mesh cannot drive them.
5. **geometry** — square mesh for `eigh` (see "Sharp edges"); for SLATE,
   **square or N×1** (`px > 1 and py > 1 and px != py` is rejected — one
   square tile per rank is impossible on both axes otherwise) plus the
   1×q stride-assert guard; ScaLAPACK's square-or-1-D descriptor
   requirement. Explicit `cusolvermp` for cholesky/solve_lu on a 1-D
   mesh resolves to `native` (documented legacy ladder semantics) rather
   than raising.

   > The SLATE rules in `_check_geometry` are a deliberate MIRROR of
   > `ffi/slate/context.validate_tile_layout` (the call-time guard).
   > **Keep them in sync** — a rule enforced only at call time makes a
   > returned `'slate'` a broken promise. That was bug L-1 (2026-07-25):
   > `resolve_backend('cholesky','slate', 2×4)` returned `'slate'` and
   > the next call raised. Fixed by adding the `px != py` rule here.
6. **divisibility** — `n % px == 0 and n % py == 0`, checked when the
   caller passes `n` (block-cyclic one-tile-per-rank layouts have no
   ragged tiles).

Errors are `ValueError` (bad name / platform / coverage / geometry /
divisibility) or `RuntimeError` (not compiled), and always name the
failed guard, the mesh, and the available alternatives.

## Backends at a glance

| backend | ops | platform | mesh geometry | needs | notes |
|---|---|---|---|---|---|
| `native` | all | any | any | nothing (pure JAX) | eigh: `jnp.linalg.eigh`, q-batched (every device solves its own shard of the batch). cholesky: replicated dense factor (mesh-invariant) or in-tree `sharded_cholesky` shard_map kernel. solve_lu: per-q `jnp.linalg.solve` + ridge. |
| `cusolvermp` | eigh, cholesky, solve_lu | CUDA only | eigh: **square** (deadlock otherwise); cholesky/lu: true-2D (px,py ≥ 2), else falls back to native | `liblorrax_ffi.so` + NCCL | Block-cyclic; the factor is grid-dependent (see replication cap). eigh returns a RAW buffer whose conj-transpose is the eigenvector matrix — `dispatch_eigh` normalizes this. |
| `slate` | eigh (CUDA only), cholesky | CUDA or host | **square or N×1** (`px,py > 1` needs `px == py`); never 1×q (stride assert) | SLATE in the FFI build | The portability path (Frontier/Aurora). Returns TRUE column eigenvectors. Host `heev` is **broken and now REFUSED at resolve time** — see "Sharp edges". |
| `scalapack` | eigh, solve_lu | host only | square or 1-D | `liblorrax_ffi_host.so` linked against MKL ScaLAPACK+BLACS (Cray LibSci elsewhere) | eigh = `pzheevd`/`pdsyevd`, **the permanent CPU distributed eigh** (what `distributed` resolves to there); solve_lu = `pXgetrf`+`pXgetrs`. Returns TRUE column eigenvectors. Both are explicit-request only — neither is ever auto-picked. |
| `distributed` (alias) | eigh | any | that of the backend it names | — | Not a library: "spread ONE tile over the mesh with this platform's distributed eigh". Resolves to `scalapack` on cpu (**permanently**) and `cusolvermp` on CUDA, then runs the full guard ladder. This is the name the ζ-fit's `distributed_zeta_solve = distributed` tier uses. |

> **Same-.so neighbours (not linalg backends):** `liblorrax_ffi_host.so`
> also carries the flat-k MKL FFT (DFTI API) handlers
> (`common.fft_helpers`, `LORRAX_FFT_FFI` — Intel-specific by
> construction) and the `contract_bands` GEMM handler
> (`lorrax_mklblas_gemm_batch`, `LORRAX_BANDS_GEMM_FFI`, auto default on
> CPU).  The GEMM handler is vendor-portable: it builds against any
> standard CBLAS on the resolved linalg link line and prefers the
> batched `cblas_?gemm_batch` entry when the BLAS has it, falling back
> to a plain `cblas_?gemm` loop otherwise — **works in principle with
> Intel MKL or Cray LibSci (batched entry when available, plain-GEMM
> loop otherwise); tested with Intel only so far.**  *(SUPERSEDED
> detail, 2026-07-31: this box used to say the batched entry was found
> by a CMake `check_symbol_exists` probe.  That build-time probe was
> DELETED 2026-07-29 after two false-negative gate cycles; the decision
> is run-time `dlsym`, per precision, announced at first use —
> `vendor_gemm_service.md` §4 is the current contract.)*

**When does an FFI backend actually win?** It depends on the *regime* —
and the two measured regimes point in opposite directions.

*Batched eigh on GPU: native wins by a mile.* Measured
(`tests/bench/eigh_benchmark.py --mode dispatch`, complex128, 2×2 A100-80GB
mesh, native batch 32): the FFI eigh is 640×/249×/281×/94× slower *per
matrix* at n = 512/1024/2048/4096 (cusolvermp) — fixed-cost dominated.
The native path solves ndev matrices concurrently; the FFI path solves
one matrix ndev-ways and walks the batch serially. Hence `auto` →
`native` for eigh, always.

> ⚠️ **That number is batched-GPU-eigh only.** An earlier version of this
> page stated "FFI is 100–600× slower" without qualification; for a
> *single tile on a multi-node CPU mesh* it is **backwards**.

*Single tile on a Frontera CPU mesh: the FFI wins* (workstream L,
2026-07-25; 8 nodes × 2 ranks, 4×4 mesh, c128, all arrays face-sharded
`P('x','y')`, medians over 10 reps, relerr ≤ 1e-15, zero hangs):

| op | backend | FFI median | native median | FFI win |
|---|---|---|---|---|
| `cholesky` n=400 | slate | **0.111 s** | 0.204 s | **1.8×** |
| `solve_lu` n=400, nrhs=200 | scalapack | **0.256 s** | 1.563 s | **6.1×** |
| `solve_lu` n=400, nrhs=200 (in the μ=3000 downfold chain) | scalapack | **0.240 s** | 2.766 s | **11.5×** |
| `cholesky` n=400 (in the μ=3000 chain) | slate | 0.116 s | 0.071 s | 0.6× |
| `eigh` n=1200 / 3000 | — (slate blocked, L-2) | — | 0.93 s / 10.2–11.4 s | — |

The FFI advantage is a **multi-node effect**: it grows with rank spread,
because native gathers the tile onto every process while the FFI keeps
it distributed. On 1 node × 16 ranks (pure shm MPI) the same cholesky is
0.142 s FFI vs 0.020 s native — native wins.

**Cost model to design around** (host lib, Frontera CPU):

* **first call ≈ 1.4–2.7 s** — SLATE/ScaLAPACK context creation + XLA
  compile of the wrapper. Amortized from rep 1.
* **per-call floor ≈ 0.10–0.26 s** — below this the FFI cannot win, no
  matter the size. Anything smaller belongs on `native`.
* ⇒ **hoist `resolve_backend` and the first call out of any loop**, and
  don't route matrices whose native solve is already < ~0.25 s.

*Distributed eigh on a Frontera CPU mesh: ScaLAPACK `pzheevd`*
(workstream V, 2026-07-26; c128 Hermitian PD, `P(None,'x','y')`, medians
over 3 reps, `FI_PROVIDER=tcp` so the inter-node numbers are pessimistic):

| mesh (nodes) | n | ‖AZ−ZΛ‖/‖A‖ | ‖ZᴴZ−I‖ | max\|Δλ\|/λ_max | ‖C⁺−C⁺_native‖/‖C⁺‖ | t/matrix |
|---|---|---|---|---|---|---|
| 1×1 (1) | 64 | 1.1e-15 | 1.3e-14 | 7.2e-16 | 2.3e-15 | 0.009 s |
| 1×1 (1) | 512 | 1.3e-15 | 6.5e-14 | 1.8e-15 | 4.8e-15 | 0.28 s |
| 2×2 (2) | 64 | 1.2e-15 | 1.5e-14 | 1.1e-15 | 2.6e-15 | 0.012 s |
| 2×2 (2) | 512 | 2.1e-15 | 1.1e-13 | 2.5e-15 | 6.4e-15 | 0.54 s |
| 2×2 (2) | 2048 | 4.0e-15 | 4.0e-13 | 5.3e-15 | 1.2e-14 | 2.33 s |
| 4×4 (8) | 2016 | timing only | | | | 1.99 s |
| 4×4 (8) | 2448 | timing only | | | | 2.89 s |

Against the REPLICATED `jnp.linalg.eigh` it replaces (measured at 2×2,
P-independent by construction): 0.076 s at n=512, 1.44 s at n=2048. So
**`pzheevd` does not win on wall time at these sizes** — 7× slower at
n=512/4 ranks, roughly parity at n≈2000 on 16 ranks. It wins on the two
things wall time does not show: the `(nq, μ, μ)` factor is never
replicated (65 MB/rank vs 9.36 GB/rank at MoS2 12×12), and the cost
divides by P where the native path is flat. Route a tile here when it
does not fit replicated, or when P is large — not to make a small eigh
faster.

Reruns are bit-deterministic on a fixed grid. First call is 8–29 s (BLACS
grid + XLA compile), amortized from rep 1 — hoist it out of loops. The
4×4 rows are latency only (the harness's replicated reference path kept
hitting Gloo `DEADLINE_EXCEEDED` under cluster contention); correctness at
that shape is covered by the P=16 fixture gate. Note the **gauge-invariant**
column: `C⁺ = Z diag(1/λ) Zᴴ` — the quantity the ζ-fit consumes — agrees
with the native eigh to 1e-14 even though the eigenVECTORS do not (and
must not be compared across meshes).

**Bottom line:** `cholesky` (slate), `solve_lu` (scalapack) and now
`eigh` (scalapack `pzheevd`) are production-ready on Frontera CPU, as is
the sharded `SᴴWS` GEMM chain at μ_big = 1200/3000. SLATE's host `heev`
is not (bug L-2) and is refused.

## The config surface

Input-file keys are the source of truth; CLI flags override them.

| key (cohsex.in `[cohsex]`) | values | consumer |
|---|---|---|
| `distributed_cholesky` | `auto \| off \| cusolvermp \| slate` | GW ζ-fit charge channel (`isdf/core`) |
| `distributed_lu` | `auto \| off \| cusolvermp \| scalapack` | GW ζ-fit transverse channels (`isdf/core`) |
| `eigh_backend` | `auto \| off \| cusolvermp \| slate` | BSE/htransform eigh sites (`bse_setup`, `vq_interp`) via `htransform` / `exciton_bands` |
| `distributed_zeta_solve` | `auto \| replicated \| per_q \| distributed` | GW ζ-fit back-solve tier (`isdf/core`) |

`ffi.linalg.resolve_backend('eigh', …)` additionally accepts `distributed`
(and `scalapack`) beyond the `eigh_backend` key's vocabulary; the ζ-fit tier
is the caller that uses them.

### `distributed_zeta_solve` — the ζ back-solve tier

| value | factor | back-solve | O(μ²)-per-q object ever replicated? |
|---|---|---|---|
| `replicated` | dense `eigh` per q, **redundantly on every rank** | gather the whole `(nq, μ, μ)` factor, then matmul | yes — `nq·μ²·16` B/rank, on EVERY r-chunk |
| `per_q` | same | gather ONE `(μ, μ)` tile at a time | yes, but one tile (`μ²·16` B peak) |
| `distributed` | ScaLAPACK `pzheevd` over the whole mesh, truncation on the replicated spectrum, `C⁺` kept 2D-sharded | stacked GEMM `C⁺ @ Z` with BOTH operands `P(None,'x','y')` | **no** |
| `auto` (default) | — | `replicated` under `LORRAX_ZETA_GATHER_CAP_GIB` (4 GiB), `per_q` above | — |

`distributed` is the only tier whose **factorisation** cost scales with P:
the other two run one dense `eigh` per q on every rank, O(nq·μ³) with no
P-scaling at all (~5.5 h at μ=4k, ~86 h at μ=10k on 28 cores — the wall
that caps the centroid ladder at ~4k). It is EXPLICIT opt-in because a
block-cyclic eigh picks a different (equally valid) eigenvector gauge, so
ζ agrees with the other tiers to ~κ·ε rather than bit-exactly. It requires
the charge channel, `charge_zeta_solve = 'rank_truncate'`, and a mesh the
ScaLAPACK eigh accepts (host, one process per device, **square or 1-D**,
μ_pad divisible by both axes) — all checked at resolve time.

Layout contract of the `distributed` tier (nothing here is ever
replicated):

```
C_q, C⁺   (nq, μ, μ)  P(None,'x','y')   rows on 'x', cols on 'y'
V         (nq, μ, μ)  P(None,'x','y')   eigenvectors as COLUMNS
λ         (nq, μ)     replicated        ascending, identical per rank
Z, ζ      (nq, μ, r)  P(None,'x','y')   μ on 'x', r on 'y'
```

Z staying on `'x'/'y'` (instead of columns-on-the-FLAT-mesh) is what makes
it work — see "Sharp edges".

**Collective payload bound (`LORRAX_COLLECTIVE_CHUNK_MB`, default 128 MB) —
transport-agnostic.**
A memory cap is not a transport cap. `LORRAX_ZETA_GATHER_CAP_GIB` bounds how
much gathered data may be *live*; this bounds how many bytes ONE
`all_gather`/`psum_scatter` instruction hands to the interconnect in a single
shot. Both ζ-tier `shard_map`s (forming `C⁺`, and applying it) are driven by
a **host-level loop over q-blocks** — one XLA execution per block, so the
emitted collective cannot be re-combined by a compiler pass — with the block
sized from the *largest single* collective per q.

This is a property of the emitted program, not of any backend: the tier still
issues plain `lax` collectives, and the identical code path runs unchanged on
NCCL/CUDA and on any other XLA backend. There is no transport detection and
no per-fabric branch. Bounded collectives are the robust regime on every
fabric; oversized single-shot ones are the fragile regime on every fabric and
differ only in how they degrade. **Treat 128 MB as a portable default, not a
cluster tuning.**

At MoS2 12×12/c2406 the bound gives `q_block = 16` for the C⁺ formation
(127.8 MB/execution against 1.151 GB unchunked) and `q_block = 3` for the
back-solve GEMM (114.2 MB against 5.482 GB). It was calibrated against the
loudest available failure: job 7876062 died at P=144 inside the unchunked
1.15 GB collective with MaxRSS at 12 % of budget, while the healthy `per_q`
control on the same 144 ranks was issuing 104 MB collectives. Setting the
knob to `0` restores the single-shot behaviour (for reproducing that failure
only); below the cap it is a no-op, so at fixture scale the emitted HLO is
unchanged. `LORRAX_COLLECTIVE_CHUNK_LOG=0` silences the per-site line.

* `--eigh-backend` (htransform, exciton_bands CLIs) **overrides** the
  `eigh_backend` key; unset, the key (default `auto`) applies.
* The legacy keys `cusolvermp_charge` / `cusolvermp_lu` were REMOVED
  (2026-07-31); use `distributed_cholesky` / `distributed_lu`.
* Env: `LORRAX_FFI_SO` / `LORRAX_FFI_HOST_SO` point at the CUDA/host
  `.so`; `LORRAX_ZETA_REPLICATE_CAP_GIB` moves the replication cap
  (default 4 GiB); `LORRAX_ZETA_RIDGE` / `LORRAX_ZETA_RCOND` condition
  the ζ-solve.

`gw_config.py` validates all three keys at parse time. On a CPU JAX
backend it rewrites `distributed_cholesky`/`distributed_lu` values that
cannot work (with a printed notice), but deliberately does NOT rewrite
`auto` (see "Sharp edges") and does not touch `eigh_backend` — an
explicit FFI eigh request keeps fails-loudly semantics at resolve time.

## The plan API

`linalg.plan(op, mesh, backend=…, n=…)` returns a **`LinalgPlan`**: the
resolution, plus everything a call site used to re-derive around it.

| attribute / method | what it is |
|---|---|
| `.backend` | the resolved name (`native` / `cusolvermp` / `slate` / `scalapack`) |
| `.is_native` | `True` ⇒ **the caller owns this route** (see below) |
| `.in_sharding` / `.batch_in_sharding` | the operand contract: `P('x','y')` for one tile, `P(None,'x','y')` for a stack; `None` on a native plan |
| `.module` | the backend package, via `backend_module` |
| `plan(A, …)` | run on ONE tile; operands moved to `.in_sharding` first |
| `plan.batched(A, …)` | run on a stack — the backend's own batched entry point if it has one, otherwise a loop + `jnp.stack` |
| `.describe()` | one line for a run banner |

Why it exists: every FFI-linalg call site had grown the same five lines
around the one that mattered — `resolve_backend(...)`, compare against
`NATIVE`, build a `NamedSharding(mesh, P('x','y'))`, `device_put` /
`with_sharding_constraint` the operand into it, then loop the batch axis
and `jnp.stack` because that particular backend has no batched entry
point.  Five copies is five places for **FFI-adjacent resharding** to
drift, and that is where this code base has lost the most time (J.9's
silent NaNs from a Z re-layout, T.4's per-r-chunk recompile of one, V.4's
deleted `_reshard_z`).  `ensure_sharding` — traced ⇒
`with_sharding_constraint`, already-there ⇒ untouched, otherwise
`device_put` — is now the single copy.

**The plan does not change resolution.**  `plan()` calls
`resolve_backend` with the caller's arguments and stores the answer;
route strings, `auto` policy and every guard are byte-identical to
calling the resolver directly, and the pinned route tests cover both
spellings.

**`is_native` means the caller owns it.**  `native` is a real backend
whose fast paths are *batched, and fused into the caller's jit* — and
for cholesky / solve_lu it is not one call at all but the channel-policy
route in `isdf/core` (replicated dense factor / 2-D blocked `shard_map` /
per-q ridged solve).  So `plan(...)` runs the native path only for
`eigh`, where it genuinely is one call, and raises for the other two
naming what owns them rather than pretending.

**Hoist the plan out of loops.**  Resolution `dlopen`s (cached
afterwards) and the first FFI call builds a BLACS / cuSOLVERMp context
and compiles an XLA module — 1.4–2.7 s measured, scorecard L §5.

## Code examples

Resolve + call for eigh on a 2×2 mesh (the `bse_setup` pattern):

```python
import jax.numpy as jnp
from jax.sharding import Mesh
from ffi import linalg

mesh = Mesh(devices.reshape(2, 2), ('x', 'y'))          # one process/device

# Resolve ONCE, before the q loop; every guard fires here.
eigh_plan = linalg.plan("eigh", mesh, backend=requested, n=n)
log(eigh_plan.describe())

if eigh_plan.is_native:
    lam, R = jnp.linalg.eigh(A_qbatch)      # q-sharded batch, fused in caller
else:
    lam, R = eigh_plan(A_one)               # ONE distributed tile
# Either way: A @ R == R @ diag(lam) (the plan normalizes conventions).
```

A stack, without caring whether the backend is batched (the
`vq_interp.prepare_coarse` and ζ-tier pattern):

```python
lam, R = eigh_plan.batched(C_chunk)         # (nb, n, n) -> (nb, n), (nb, n, n)
```

ScaLAPACK factors the whole stack in one FFI call (one descriptor, one
workspace); cuSOLVERMp and SLATE have no batched eigh, so the plan loops
and stacks.  The call site does not encode which is which.

Cholesky through the ζ-fit policy layer (route strings) and the facade
import seam:

```python
from isdf.core import _resolve_solver_kind_charge
kind = _resolve_solver_kind_charge(mesh, override, n_rmu=n_mu, nq=nq,
                                   charge_zeta_solve="rank_truncate")
# 'replicated_rank_truncate' | 'replicated_cholesky' | 'sharded_cholesky'
# | 'cusolvermp_cholesky' | 'slate_cholesky'  (or a loud refusal)

if kind == 'cusolvermp_cholesky':
    mp = linalg.backend_module('cusolvermp')
    L = mp.batched_distributed_cholesky(C_q, mesh=mesh)
```

What can run here?

```python
>>> linalg.list_backends("cholesky", mesh)      # CPU 2x2, slate-less build
{'native':     'available (replicated dense / in-tree sharded_cholesky (isdf/core))',
 'cusolvermp': "unavailable: cholesky backend 'cusolvermp' is CUDA-only but …",
 'slate':      "unavailable: cholesky backend 'slate' requested but its FFI "
               "handler (lorrax_slate_potrf) is not compiled into the cpu FFI library. …"}
```

## Adding a backend

**Everything in this section is inside `services/distrib_la/`.** No step
touches `src/`. The three files the previous version of this section told
you to edit (`src/ffi/<name>/`, `ffi/linalg/resolve.py`,
`ffi/linalg/plan.py`) no longer exist.

1. Write the wrapper module
   `services/distrib_la/src/distrib_la/_<name>.py`. Copy
   `distrib_la/_cusolvermp.py` — it is written flat as the template and
   its docstring calls out the three per-routine decisions (donation,
   handle-vs-array return, output normalisation).
2. Register the C++ handler symbol in `distrib_la/loader.py`
   (`_CUDA_TARGET_SYMBOLS` / `_HOST_TARGET_SYMBOLS`). That makes
   `probe_target()` / `has_target()` — and therefore the capability guard
   — work for free, with the ABSENT-vs-BROKEN split already correct. If
   the same handler is also reachable from lorrax's own loader
   (`src/ffi/common/ffi_loader.py` keeps the FFT/GEMM/phdf5 half), add the
   row there too: both loaders open the same `.so`.
3. Add the backend to `distrib_la/resolve.py`: one entry per op in
   `BACKEND_CHOICES`, one `(op, backend) → (target, platforms)` row in
   `_SPEC`, any geometry rule in `_check_geometry`, a `_DISTRIBUTED_DEFAULT`
   row per platform if it is a platform's `distributed` answer, and a
   branch in `backend_module`. A declared-untested tier still gets its
   rows — that is what makes the routing question have an answer.
4. Add ONE row to `distrib_la/plan.py`'s `_IMPL`: the single-tile entry
   point, the stacked one (either may be `None` — the plan fills the
   missing side in), and an output normaliser if the library's convention
   differs. Every call site picks the backend up from that row; do NOT
   normalize conventions at call sites. Declare donation in `DONATES` if
   the op's is different.
5. If it returns an opaque factor, add its branch to `distrib_la/factor.py`
   so it arrives as a `FactorToken` and leaves through `solve()`. Do not
   hand a raw handle across a `jit` boundary — the token is deliberately
   not a pytree.
6. Extend the deck vocabulary: `gw_config.py`'s validation reads
   `eigh_backend_choices()`, which reads `distrib_la.BACKEND_CHOICES`, so
   for `eigh` there is nothing to do but check the key comment. The
   `distributed_cholesky` / `distributed_lu` / `w_dyson_solver` lists in
   `gw_config` are separate deck vocabularies and are edited by hand.
7. Write the tests **first or with it, never after**: an L-a algebra cell,
   an L-b emulated-mesh cell, an L-c body in `_CLI_CELLS`, a contract cell
   for each refusal it can emit, and — for every one of them — the case
   where the check returns FALSE. Add a machine-profile row if the backend
   is expected present somewhere, so an unexpected skip is a failure.
8. Update `docs/services/distrib_la.md` (the Backends table) and, if it
   moves numbers, record a baseline row rather than a slow test.

### Call sites: migrated, and not

| call site | state |
|---|---|
| `bandstructure/bse_setup.compute_wfns_fi` (fH_q) | **plan** — one plan for the whole q loop |
| `bse/vq_interp.prepare_coarse` (coarse C_q) | **plan** — `plan.batched` replaced the hand-rolled per-q loop + `jnp.stack` |
| `isdf/core._factor_c_q_distributed_rank_truncate` (ζ `distributed` tier) | **plan** — `backend='distributed'`, so the platform default is chosen in ONE place (it used to hard-code `scalapack` while the tier's own guard approved `distributed`, which on a CUDA mesh approved cuSOLVERMp and then called the host-only backend) |
| `tests/bench/eigh_benchmark.py` | **plan** — and the plan is now hoisted out of the timing loop |
| `isdf/core.solve_zeta` cusolvermp-potrs / getrf+getrs branches | route strings + a cuSOLVERMp *handle* whose block-cyclic geometry `solve_zeta` rebuilds; the surrounding pad → solve → reshard → trim frame is de-duplicated into `_distributed_backsolve`, the backend call itself is left on `backend_module`. Migrate together with the ζ route strings, never separately — they are pinned. |
| `isdf/core.factor_c_q` cusolvermp / slate cholesky branches | same: pinned route strings, and the two backends return different objects (handle vs `to_jax_lower()` L). `_IMPL` records the asymmetry; the branches can move once a route-pin test covers the handle path. |
| `gw/w_isdf._get_w_solve_fn_distributed` (the W Dyson solve, plan 2 of 2) | **plan** — `plan('solve_lu', mesh, backend='distributed').batched(A, B)` with `A = (1 − pref·V·χ₀)` formed by a chunked 2-D block GEMM inside `shard_map` (house style: `isdf/core._distributed_pinv_apply`; per-instruction payloads bounded by `LORRAX_COLLECTIVE_CHUNK_MB`), reached by the input key `w_dyson_solver = distributed`.  The μ axes never leave `P(None,'x','y')`, so W_q(μ_X, ν_Y) is solved where it already lives — no rank ever materialises a full (μ, μ) tile. `distributed` is legal vocabulary for `solve_lu` (scalapack on cpu, cusolvermp on CUDA — the two backends `_IMPL` lists).  A resolve-time refusal RAISES with the resolver's own message — an explicitly requested distributed W solve never silently downgrades to the local per-q LU (the only other plan; `w_dyson_solver = local`, the default). |
| `gw/w_isdf` cuBLASMp fused W-solve | **REMOVED** (two-plan W cleanup, 2026-07-27): the fused gemm+potrf+trsm kernel and its `isdf_memory_mode = low_mem` key are gone, superseded by `w_dyson_solver = distributed` (which binds cusolvermp on CUDA meshes through this facade — no new seams).  The `ffi/cublasmp` wrapper itself still exists for its own tests; it has no production consumer. |
| `tests/bench/slate_*_test.py`, `slate_vs_cusolvermp_bench`, `eigh_block_sweep`, `cusolvermp_*_test.py` | deliberately NOT migrated: they exercise backend *internals* (raw buffer layouts, `block_size`, `compute_evecs`) that the plan normalizes away. Testing through the abstraction would stop them testing the thing. |

## Sharp edges (read before touching defaults)

* **Square-mesh deadlock.** `cusolverMpSyevd` requires square ScaLAPACK
  blocks; on a non-square mesh it DEADLOCKS inside a collective instead
  of returning an error (observed 4×1/1×4, 2026-07-10). This is why the
  square-mesh check is a resolve-time guard and not a call-time
  courtesy.
* **`distributed_cholesky = off` silently destroys physics.** `off` is
  an *override* that short-circuits the whole route policy to
  `sharded_cholesky` — skipping the replicated route, which is the ONLY
  one carrying the rank-truncation conditioning
  (`charge_zeta_solve = 'rank_truncate'`, the production default). A
  full MoS2 12×12 G0W0 once ran to rc=0 with a QP gap of −161 eV this
  way (`tests/test_charge_zeta_route.py` pins the routes;
  FRONTERA_ADVICE §6a). Never "clean up" the default routes.
* **The replication cap.** The block-cyclic distributed factors are
  grid-DEPENDENT (partial-sum regrouping differs per process grid), and
  GN-PPM amplifies a ~0.3% factor drift into tens of eV. Below
  `LORRAX_ZETA_REPLICATE_CAP_GIB` (default 4 GiB for the whole
  (nq, n_μ, n_μ) c128 stack) the charge factor is therefore fully
  replicated and mesh-invariant. Above the cap, `rank_truncate`
  REFUSES rather than silently downgrading
  (`tests/test_zeta_mesh_invariance.py::test_rank_truncate_refuses_above_the_replication_cap`).
* **One process per device.** All FFI backends assume the LORRAX
  process model. Forced multi-device single-process meshes
  (`--xla_force_host_platform_device_count`) get `native` only; the
  coverage guard enforces this.
* **`auto` is CPU-safe everywhere.** No `auto` path ever selects a
  CUDA-only backend on a CPU mesh (`mesh_is_cpu` guard), and gw_config
  must keep letting `auto` THROUGH on CPU (rewriting it to `off` was the
  −161 eV bug above).
* **The production mesh shape decides whether FFI linalg is reachable
  at all.** SLATE and ScaLAPACK both need a **square or N×1** mesh. A
  production 8×10 (P=80) therefore gets `native` for *everything* — the
  guards reject cleanly at resolve time (no hang), but the FFI is simply
  not in play. If FFI cholesky/LU matters for a run, choose **8×8 (64
  ranks)** or **10×10 (100 ranks)** and keep `n` divisible by both axes.
* **The flat-mesh column-sharding trap (scorecard J.9).** A block-sharded
  `(μ,μ)` operator can only be applied by ranks that share a column block
  cooperating on the μ contraction. ζ's `Z` is built at
  `P(None,'x','y')` but the replicated/per_q back-solve reshards it to
  `P(None, None, ('x','y'))` — columns over the **flat** mesh — after
  which ranks sharing a `y` index hold UNRELATED column blocks. A 2-D
  SUMMA on that layout `psum`s partial products built from different
  columns: NaNs, `rc=0`, no crash (the gate caught them only as
  float-count deficits in eqp). The `distributed` tier's answer is not to
  fix the SUMMA but to **never do the reshard**: it consumes Z in the
  layout `z_q_from_psi_sm` builds it in. If you add another sharded-operator
  path, check `Z.sharding.spec` first.
* **SLATE host `heev` SIGSEGVs (bug L-2, open → now REFUSED).** `lorrax_slate_eigh` →
  `slate::heev(…, Target::HostTask)` segfaults rank 0 deterministically
  at *every* configuration tried: n = 64/512/1200, mesh 1×1/2×2/4×4,
  intra- and inter-node, `compute_evecs` True and False, SLATE built
  both `blas_threaded=true` and `false`. It reproduces on a **1×1 mesh,
  single rank, single process**, so it is not MPI, not the comm remap,
  not the LORRAX layout contract; ruled out: `MPI_Query_thread` = 3
  (MULTIPLE), `ldd -r` clean, MKL 2020.1 exports the full 2-stage set.
  ~~Prime suspect: SLATE 2025.05's host `heev` against MKL's LAPACK 3.8
  (the upstream validation was Cray LibSci).~~
  > **REFUTED 2026-07-31 (job 7883880).** `slate::heev` is not the fault.
  > Reached through SLATE's own ScaLAPACK-compatibility shim
  > (`scalapack_heevd.cc` → `slate::heev`, `LD_PRELOAD`ed over LORRAX's
  > ScaLAPACK handler) on a **1×1 CPU mesh**, the very same `libslate.so.2`
  > and MKL returned **CORRECT** at n = 32 / 64 / 128 / 512
  > (`max|Δλ|` 1.24e-14 → 5.76e-13, `‖AZ−ZΛ‖/‖A‖` ≤ 3.2e-15,
  > `‖ZᴴZ−I‖` ≤ 8.0e-14) — while in the **same job, same mesh, same sizes**,
  > `ffi.slate`'s own host handler died with **SIGSEGV (rc 139) at n=32 and
  > n=64**. Same library, same MKL, same process image, opposite outcomes ⇒
  > the fault is in **`cpp/slate/host_ffi.cc`'s call path**, not upstream.
  > Enumerable differences at that call site: LORRAX builds `Z` as
  > SLATE-managed tiles and copies out via `tileGetForReading`, whereas the
  > shim wraps the caller's buffer with `Matrix::fromScaLAPACK`; LORRAX uses
  > the no-`grid_order` `HermitianMatrix::fromScaLAPACK` overload; and
  > LORRAX passes only `{Target, HostTask}` where the shim also passes
  > `MaxPanelThreads` and `InnerBlocking` (`MethodEig::DC` is ruled out —
  > `src/heev.cc:92` shows DC is already SLATE's default).
  > Reproducer, with the direct handler as its built-in red twin:
  > `wk_REL/harness/slalias_l2.sbatch`. **The resolve-time refusal below
  > stays** until someone fixes and gates the handler — this note changes
  > where to look, not what is currently safe to run.

  On the **same** library and
  context, `potrf`, `trsm` and ScaLAPACK `getrf/getrs` are all clean
  (potrf n=512 residual 1.47e-16). Since the handler IS compiled and the
  capability probe passes, nothing else would catch it and
  `resolve_backend` would hand back a name whose first call kills the job
  with no Python traceback — so **`('eigh','slate')` on a CPU mesh is now
  a resolve-time refusal** naming ScaLAPACK as the replacement. Use
  `distributed` (= `scalapack`) for a distributed CPU eigh, or
  `auto`/`off` for the native one. Repro:
  `wk_L/diag.py --px 1 --py 1 --ops mpi,ctx,potrf512,eigh512`.
* **`pXheevd` can return `INFO = 0` with correct eigenvalues and a
  silently garbage `Z`.** The eigenvector back-transform
  (`pXunmtr`/`pXormtr`) is a *separate* workspace requirement that
  `pXheevd`'s published `LWORK` formula does not always cover — nor does
  MKL's query. Measured on this stack: real symmetric n=32, 1×1 grid,
  PDSYEVD asks 2305 doubles, PDORMTR needs 3232 ⇒ `INFO = 0`,
  `max|Δλ| = 7e-15`, `‖ZᴴZ−I‖ = 6e-15`, and `‖AZ−ZΛ‖/‖A‖ = 1.40`. The only
  symptom is one `PDORMTR parameter number 16 had an illegal value` line
  on stderr. `eigh_ffi.cc` now floors `LWORK` with
  `max((NB(NB−1))/2, (NP0+MQ0)·NB) + NB² + 8N`. **Corollary for anyone
  adding an eigensolver: an eigenvalue-only test does not test an
  eigensolver.** Always assert `A @ Z == Z diag(W)`.
* **MKL's `pzheevd` workspace query is MANDATORY, not advisory.** MKL
  returns an `LWORK` far above the netlib `LWMIN` on multi-rank grids
  (measured 368× at N=20000 on a 2×2 grid) and **rejects the netlib
  minimum with `INFO = -16`**. `eigh_ffi.cc` therefore treats a failed
  query as fatal and uses `max(query, reference formula)`. The workspace
  is `malloc`'d inside the handler, so it is invisible to the JAX memory
  planner — `LORRAX_SCALAPACK_EIGH_LOG=1` prints it per call (measured
  11 MB WORK + 6 MB RWORK at n=2016 on a 4×4 grid; it grows as the grid
  shrinks). Also: netlib and MKL both implement `JOBZ='V'` ONLY for
  `pzheevd` (`'N'` returns `INFO=-1`), and `IA=JA=1` with `MB_A == NB_A`
  is enforced (`INFO = -706` / `-4` otherwise). `pzheevr` (MRRR) is the
  documented fallback: it supports `JOBZ='N'` and eigenvalue subsets and
  needs ~n²/p instead of ~3n²/p workspace, at slightly worse
  orthogonality on clustered spectra.
* **SLATE's ScaLAPACK overlay can hijack the `scalapack` backend
  invisibly, and is now REFUSED (2026-07-31).** SLATE ships an optional
  `libslate_scalapack_api.so` that re-defines the ScaLAPACK entry points
  and forwards them to `slate::`; its README documents `LD_PRELOAD`
  interception. Measured against LORRAX's eleven names, it **DEFINES 6**
  (`pzheevd_` `pdsyevd_` `pzgetrf_` `pdgetrf_` `pzgetrs_` `pdgetrs_` —
  i.e. every operation this backend performs) and supplies **none** of
  the five grid/descriptor names (it *calls* `numroc_` and
  `Cblacs_gridinfo` itself). So one `LD_PRELOAD` replaces every solve
  while `resolve_backend('eigh','scalapack')` still returns `scalapack`
  and still promises a callable backend — `ffi_loader` keys only on
  LORRAX's own handler symbols and cannot see it (measured:
  `has_target('lorrax_scalapack_eigh')` still answers `True` with the
  overlay live). **And it works on 1×1** — both ops, machine precision, n up
  to 512 — which is the hazard, not a reassurance, because 1×1 is the one
  geometry that cannot expose the defect. On a **2×2 mesh** (job 7883978,
  n=64) the overlay returns `‖AZ−ZΛ‖/‖A‖ = 1.52e-01` and `‖ZᴴZ−I‖ = 6.98`
  for eigh and `1.55e-01` for LU, where MKL is correct to 1e-15 on the same
  mesh. **The whole defect is one permutation**: the shims hard-code
  `MPI_COMM_WORLD` and want shard `(mx,my)` on rank `mx+my*p`, LORRAX puts
  it on `mx*q+my`. Flip the mesh's device order to Fortran and the two swap
  places exactly — overlay correct to 1.4e-15, MKL wrong by the same 1.52e-01
  to four digits. So the fix is LORRAX-side and needs no patched dependency,
  but it is a *swap*: the two providers cannot share a mesh until
  `blacs_ctxt_for` derives its permutation from the mesh's actual device
  order instead of assuming C. Separately, every shim hard-wires `info = 0`
  ("todo: extract the real info") — the ζ-fit's LU ridge covers the singular
  case, nothing covers a non-converged eigh — and `slate::heev` requires a
  square process grid (`heev.cc:102`) and `GridOrder::Col` (asserted in five
  places), while `getrf`/`getrs`/`potrf`/`trsm` carry no such constraint.
  `cpp/scalapack/blacs_grid.h` resolves the provider of the routine it is
  about to call (`dlsym` + `dladdr`) and refuses by default, naming all of
  it. Reproducer: `wk_REL/harness/slalias_mesh.sbatch`.
  `LORRAX_SCALAPACK_ALLOW_SLATE_API=1` downgrades to one loud line.
  Full measurement + the build script for the red twin:
  [`src/ffi/PORTING.md` §0b](../../src/ffi/PORTING.md).
* **`scalapack.batched_distributed_solve_lu` DONATES both A and B.**
  Rebuild them if you need to compute a residual afterwards.
* **MKL threading is a scale-dependent cliff inside the ScaLAPACK
  handlers — the handlers now pin their own team.**  The harness-wide
  `MKL_NUM_THREADS=28` (right for the local `zheevd_` route and BLAS at
  large tiles) is catastrophic for `pzheevd`/`pzgetrf` at the production
  grid, where g≈204 panels mean thousands of tiny BLAS calls interleaved
  with latency-bound BLACS collectives and the threading layer's
  fork/join + spin-wait starves MPI progress.  Measured (wk_ENV AW,
  pz_bench/pzlu_bench = the exact handler geometry, n=2448, mlx):
  12×12 pzheevd **11.28 s/q @ 14 thr vs 0.463 s/q @ 4 thr (24×)**,
  17.0 s/q @ 28 oversubscribed; 8×8 at the production 2 ranks/node × 28
  placement is monotone — 38.4 / 27.4 / 12.7 / 5.0 s/q @ 28/14/4/1 thr;
  pzgetrf+pzgetrs 12×12: 1.51 s @ 6 thr vs 0.37 @ 4 / 0.25 @ 1.  At 4×4
  (P=16, g=612) every thread count is flat — which is why this was
  invisible on the small deck.  `eigh_ffi.cc`/`solve_lu_ffi.cc` now cap
  the calling thread's MKL team via `mkl_set_num_threads_local`
  (default min(current, 4); `LORRAX_SCALAPACK_MKL_THREADS` overrides,
  `off` restores the old behaviour; dlsym-resolved, no-op on non-MKL
  ScaLAPACK).  Scope conditions: measured at n=2448, P=16–144, in a
  co-tenant-free holder window; the n=5040 confirmation cells landed in
  sibling-workstream co-tenancy (6× intra-cell rep swing) and are
  INCONCLUSIVE — rerun on quiet nodes before extrapolating the cap to
  n ≳ 5000, and re-verify if plan-B NB redistribution lands (blocking
  changes the panel size this pathology rides on).
* **Fabric: leave `FI_PROVIDER` UNSET on Frontera CLX** (do not pin `mlx`
  explicitly — you get it by unsetting; `fi_info` falsely reports −61 for
  `mlx` even where it works, so trust only `I_MPI_DEBUG=4`'s
  "libfabric provider:" banner). Measured (scorecard AP.3/AP.4, reproduced
  in-container by AS.2): unset ⇒ `mlx` at 1.07 µs / 11.4 GB/s; the old
  `FI_PROVIDER=tcp` (IPoIB) pin is 10.9 µs / 2.15 GB/s and was an
  rtx/mlx4 workaround carried over by mistake — every inter-node FFI
  number measured under it is **pessimistic** (pzheevd n=2448 P=144:
  12 s/q → 0.5–0.9 s/q after the fix). Never request `verbs` at P≥144
  with the one-block layout (68 s/q pathology). The harness dial is
  `LORRAX_MPI_PROVIDER` (`auto`|`tcp`|explicit). TACC also documents
  `ibrun`, not `srun --mpi=pmi2`, as the supported container-MPI
  launcher.

## Verification

* `wk_I/verify.py` (lorrax_setup scratch) — resolution policy, guard
  ladder, route pinning, config threading; CPU, no FFI build needed.
* `tests/test_charge_zeta_route.py`, `tests/test_zeta_mesh_invariance.py`
  — route-string pins (run with 4+ host devices).
* `tests/test_ffi_linalg_contract.py` — wrapper shape/layout contracts
  against the real FFI builds.
