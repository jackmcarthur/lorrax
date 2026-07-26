# Distributed dense linear algebra over a JAX device mesh (`ffi.linalg`)

*A guide to LORRAX's distributed-linalg stack, written for a JAX power
user — the facade is deliberately small and self-contained enough to
lift into another SPMD JAX codebase.*

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
| `solve_lu` | GW ζ-fit, transverse channels (`isdf/core.py`) | (nq, n_μ, n_μ) batched Hermitian-indefinite |

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
   ffi.linalg.dispatch_eigh(A, mesh, resolved)        ← call-time routing
   ffi.linalg.backend_module(name).<distributed op>   ← the ONE import seam
```

Three layers, one module each in `src/ffi/linalg/`:

* **`resolve.py`** — vocabulary, guard ladder, `resolve_backend`,
  `list_backends`, `backend_module`, `mesh_is_cpu`/`mesh_platform`.
* **`dispatch.py`** — `dispatch_eigh` (per-op call-time dispatch and
  output-convention normalization).
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
   > platform), **the library would not load** (missing `.so`, unresolved
   > `DT_NEEDED`, glibc/GLIBCXX mismatch, wrong `LD_LIBRARY_PATH` — the
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
   > you to rebuild a library that was fine. **The unified host lib's
   > RUNPATH covers MKL, SLATE's `lib64` and Intel MPI's `lib/release`,
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
| `slate` | eigh, cholesky | CUDA or host | **square or N×1** (`px,py > 1` needs `px == py`); never 1×q (stride assert) | SLATE in the FFI build | The portability path (Frontier/Aurora). Returns TRUE column eigenvectors. Host `heev` is **broken** — see "Sharp edges". |
| `scalapack` | solve_lu | host only | square or 1-D | `liblorrax_ffi_host.so` linked against MKL ScaLAPACK+BLACS (Cray LibSci elsewhere) | The host twin of cusolvermp's LU; explicit request only, never auto-picked. |

**When does an FFI backend actually win?** It depends on the *regime* —
and the two measured regimes point in opposite directions.

*Batched eigh on GPU: native wins by a mile.* Measured
(`common/eigh_benchmark.py --mode dispatch`, complex128, 2×2 A100-80GB
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
| `eigh` n=1200 / 3000 | — (blocked, L-2) | — | 0.93 s / 10.2–11.4 s | — |

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

**Bottom line:** `cholesky` (slate) and `solve_lu` (scalapack) at
n = 400–1200, and the sharded `SᴴWS` GEMM chain at μ_big = 1200/3000,
are production-ready on Frontera CPU. Distributed `eigh` is not
(bug L-2).

## The config surface

Input-file keys are the source of truth; CLI flags override them.

| key (cohsex.in `[cohsex]`) | values | consumer |
|---|---|---|
| `distributed_cholesky` | `auto \| off \| cusolvermp \| slate` | GW ζ-fit charge channel (`isdf/core`) |
| `distributed_lu` | `auto \| off \| cusolvermp \| scalapack` | GW ζ-fit transverse channels (`isdf/core`) |
| `eigh_backend` | `auto \| off \| cusolvermp \| slate` | BSE/htransform eigh sites (`bse_setup`, `vq_interp`) via `htransform` / `exciton_bands` |

* `--eigh-backend` (htransform, exciton_bands CLIs) **overrides** the
  `eigh_backend` key; unset, the key (default `auto`) applies.
* Legacy keys `cusolvermp_charge` / `cusolvermp_lu` (`auto|on|off`) are
  still honored with a deprecation warning when the portable key is left
  at `auto`.
* Env: `LORRAX_FFI_SO` / `LORRAX_FFI_HOST_SO` point at the CUDA/host
  `.so`; `LORRAX_ZETA_REPLICATE_CAP_GIB` moves the replication cap
  (default 4 GiB); `LORRAX_ZETA_RIDGE` / `LORRAX_ZETA_RCOND` condition
  the ζ-solve.

`gw_config.py` validates all three keys at parse time. On a CPU JAX
backend it rewrites `distributed_cholesky`/`distributed_lu` values that
cannot work (with a printed notice), but deliberately does NOT rewrite
`auto` (see "Sharp edges") and does not touch `eigh_backend` — an
explicit FFI eigh request keeps fails-loudly semantics at resolve time.

## Code examples

Resolve + call for eigh on a 2×2 mesh (the `bse_setup` pattern):

```python
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ffi import linalg

mesh = Mesh(devices.reshape(2, 2), ('x', 'y'))          # one process/device

# Resolve ONCE, before the q loop; every guard fires here.
resolved = linalg.resolve_backend("eigh", requested, mesh, n=n)

if resolved == linalg.NATIVE:
    lam, R = jnp.linalg.eigh(A_qbatch)                  # q-sharded batch
else:
    A = jax.device_put(A_one, NamedSharding(mesh, P('x', 'y')))
    lam, R = linalg.dispatch_eigh(A, mesh, resolved)    # ONE distributed tile
# Either way: A @ R == R @ diag(lam) (dispatch normalizes conventions).
```

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

1. Write the wrapper package `src/ffi/<name>/` (copy
   `ffi/cusolvermp/eigh.py` — it is written flat as the template; the
   three per-routine decisions are called out in its docstring).
2. Register the C++ handler symbol in `ffi/common/ffi_loader.py`
   (`_CUDA_TARGET_SYMBOLS` / `_HOST_TARGET_SYMBOLS`). That makes
   `has_target()` — and therefore the capability guard — work for free.
3. Add the backend to `ffi/linalg/resolve.py`: one entry per op in
   `BACKEND_CHOICES`, one `(op, backend) → (target, platforms)` row in
   `_SPEC`, any geometry rule in `_check_geometry`, and a branch in
   `backend_module`.
4. Route the call in `ffi/linalg/dispatch.py` (for eigh) or the
   consumer's route-string branch (`isdf/core.py` for the ζ-fit ops),
   normalizing the output convention in the dispatcher, not at call
   sites.
5. Extend the config vocabulary (`gw_config.py` validation + the key
   comment) and this page's table.

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
* **SLATE host `heev` SIGSEGVs (bug L-2, open).** `lorrax_slate_eigh` →
  `slate::heev(…, Target::HostTask)` segfaults rank 0 deterministically
  at *every* configuration tried: n = 64/512/1200, mesh 1×1/2×2/4×4,
  intra- and inter-node, `compute_evecs` True and False, SLATE built
  both `blas_threaded=true` and `false`. It reproduces on a **1×1 mesh,
  single rank, single process**, so it is not MPI, not the comm remap,
  not the LORRAX layout contract; ruled out: `MPI_Query_thread` = 3
  (MULTIPLE), `ldd -r` clean, MKL 2020.1 exports the full 2-stage set.
  Prime suspect: SLATE 2025.05's host `heev` against MKL's LAPACK 3.8
  (the upstream validation was Cray LibSci). On the **same** library and
  context, `potrf`, `trsm` and ScaLAPACK `getrf/getrs` are all clean
  (potrf n=512 residual 1.47e-16). **Use `eigh_backend = auto`
  (native).** Repro: `wk_L/diag.py --px 1 --py 1 --ops mpi,ctx,potrf512,eigh512`.
* **`scalapack.batched_distributed_solve_lu` DONATES both A and B.**
  Rebuild them if you need to compute a residual afterwards.
* **Fabric: use `FI_PROVIDER=mlx`, not `tcp`, on Frontera.** Frontera CLX
  is ConnectX-6 and TACC's own `impi` modulefile sets `FI_PROVIDER=mlx`
  plus UCX tunings. `FI_PROVIDER=tcp` (IPoIB) is an rtx/mlx4 workaround
  that was carried over by mistake; every inter-node FFI number measured
  under it is **pessimistic**. TACC also documents `ibrun`, not
  `srun --mpi=pmi2`, as the supported container-MPI launcher.

## Verification

* `wk_I/verify.py` (lorrax_setup scratch) — resolution policy, guard
  ladder, route pinning, config threading; CPU, no FFI build needed.
* `tests/test_charge_zeta_route.py`, `tests/test_zeta_mesh_invariance.py`
  — route-string pins (run with 4+ host devices).
* `tests/test_ffi_linalg_contract.py` — wrapper shape/layout contracts
  against the real FFI builds.
