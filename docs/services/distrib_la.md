# distrib_la — distributed dense linear algebra over a JAX device mesh

`services/distrib_la/`. Independently installable (`pyproject.toml`,
src-layout); depends on `lxkit` + jax and nothing else in LORRAX. Every
vendor library it can reach — ScaLAPACK, SLATE, cuSOLVERMp — appears in
exactly one dependency edge (`distrib_la.loader`, which `dlopen`s a `.so`
by path) and in zero declared dependencies.

## Known limitation — `distributed_eigh` hangs at a 3×3 mesh for n ≥ 3072

**Read this before you plan a large deck.** `distributed_eigh` is not
currently safe on a 3×3 device mesh once the matrix reaches n = 3072. It does
not fail and it does not raise; it hangs silently and forever. The cuSOLVERMp
banner prints, nothing follows it, and the job sits there until the scheduler
or a human kills it. Runs were killed at 420 s and, in one case, at 900 s
without the call ever returning.

The break is bracketed tightly. At a 3×3 mesh, n = 2049 completes in 6.9 s
with a maximum eigenvalue error of 1.5e−11, and n = 3072 never returns. Every
larger size tested at that mesh — 3072, 4098, 6144, 8190 — hangs the same way.
A 2×2 mesh at n = 8192 is fine, finishing in 9.3 s, so this is specific to the
3×3 geometry rather than a general size ceiling.

**The allocator is exonerated.** The hang reproduces identically under the
`platform` allocator, which is what the fleet runs today, and under the
recommended BFC settings. Whatever is wrong at 3×3 was wrong before the
allocator question was ever asked, and it will still be wrong if the allocator
recommendation is rejected. The card reported 36.4 GiB free at the moment of
the call, so this is not memory starvation either.

What this means in practice is that **large-deck users must not assume
`distributed_eigh` works past roughly n ≈ 2k until this is closed.** If your
stage calls `eigh` on a big matrix, run it on a 2×2 mesh, or keep the matrix
below the bracket, or use a non-distributed route. A 3×3 mesh with a large
matrix will not give you a wrong answer — it will give you no answer at all,
which in a batch queue is an expensive way to find out.

Two things are still owed here. The first is a control: a 4×4 leg at n = 8192
is what separates "3×3 is odd because it is not a power of two" from "anything
past 2×2 is broken". That leg never got a placement before the measuring
window closed, and the log it did leave carries no cuSOLVERMp line at all, so
its non-zero exit is a queue artifact and must not be read as a hang. The
second is the root cause itself, which needs a multi-node GPU allocation to
chase and therefore has not been attempted. One unconfirmed lead, offered as a
lead: at a 3×3 grid, n = 2049 gives a local block of 683, below the usual 1024
tile, while n = 3072 gives exactly 1024 per rank and so more than one block
column per rank for the first time. A hang appearing exactly where the 2-D
block-cyclic distribution stops being trivial on a non-power-of-two grid is a
plausible library-side story, but nobody has confirmed it.

The owner's disposition is **revisit soon**. This is a `distrib_la` solver
defect and it belongs to whoever owns `distrib_la`.

The seven-leg evidence table, the exact reproduction and the probe live in the
2026-08-09 amendment to `tests/KNOWN_FAILURES.md` (§1, "`distributed_eigh`
hangs at a 3×3 mesh for every n ≥ 3072"). The run artifacts are under
`/pscratch/sd/j/jackm/sigma_scaling_0809/_reports/` in legs `la_p4_bfc85`,
`la_p9_bfc85`, `la_p9_platform`, `la_p9_small_bfc85`, `la_p9_3072`,
`la_p9_mid` and `la_p9_6144`; the probe is
`sigma_scaling_0809/probe_pressure_sq.py`, and the prose write-up is
`SIGMA_SCALING.md` §8. `la_p16_8192` is the leg that did not run.

## Purpose

One door for `eigh`, `cholesky` and `solve_lu` on an `('x','y')` device
mesh, so a driver says *what* to compute and *where*, never *which
library*. Before this service the same three ops were dispatched from ten
`src/` modules through four packages (`ffi.linalg`, `ffi.slate`,
`ffi.scalapack`, `common.cholesky_2d`), with per-call-site guard ladders
that disagreed; the worst measured consequence of that drift was a silent
route change that ran to completion with `rc=0` and a QP gap of **−161 eV**.
Backend selection is declarative data, backend *capability* is probed once,
and a resolved backend name is a **promise** that every guard passed.

The service is the door. `import distrib_la` and use top-level names;
`from distrib_la.plan import …` from outside the package is a layering
failure that `tests/test_layering.py` fails on (with a red twin).

## API

| name | what it is |
|---|---|
| `plan(op, mesh, *, backend='auto', n=None, closure=None) -> Plan` | Resolve once, then call. **Eager** — dlopens, probes, reads `jax.process_count()`. |
| `Plan(A)` / `Plan.batched(A_stack)` | One tile at `P('x','y')` / a stack at `P(None,'x','y')`. **Trace-safe** — no dlopen, no `device_put`, no process count inside. |
| `Plan.is_native`, `.backend`, `.describe()` | The resolved fact, readable; never something a caller must branch on to be correct. |
| `Plan.batched_route`, `BATCHED_ROUTES`, `BATCHED_SCAN_UNROLL` | HOW a stack runs: a `lax.scan` over the single-matrix op, or the backend's own stacked entry where the library has one, with a batch-axis-reshard route reserved. **The one place that decides.** Introspection — a caller reads it, never branches on it. |
| `Plan.native_fn` | A pure closure for a fusion-critical site that needs the math inside its own `jit`. Native backends only. |
| `factor(op, A, mesh, *, backend, ...) -> FactorToken` | Factor once. |
| `solve(token, B) -> jax.Array` | Back-solve many. The token carries the handle (ScaLAPACK `ipiv`, cuSOLVERMp raw buffer, SLATE `SlateLowerL`). |
| `resolve_backend(op, requested, mesh, *, n=None) -> str` | The **raising** probe. `n` is decoupled from the operand so a caller that will pad can ask before it has built anything. |
| `list_backends(op, mesh) -> dict` | The **never-raising** report, for startup banners. |
| `BACKEND_CHOICES`, `EIGH_BACKENDS`, `CHOLESKY_BACKENDS`, `LU_BACKENDS`, `OPS`, `NATIVE` | The vocabulary. Importable with **no `.so` anywhere on the machine** — a deck parser must not need the FFI layer to read a deck. |
| `mesh_key(mesh)`, `mesh_platform(mesh)`, `mesh_is_cpu(mesh)` | Stable hashable mesh identity (axes, extents, platform, device ids) and its two predicates. `mesh_key` is for any cache whose stored value does **not** retain the mesh; `id(mesh)` there is the documented drift. |
| `dial_key()` | Factory-time dials folded into one tuple, for kernel cache keys. |
| `probe_target`, `has_target` | Capability, with the ABSENT / BROKEN split (`lxkit.probe`). |
| `dispatch_batched_eigh(A, mesh, backend)` | The one legacy entry point kept for `gw.qsgw_density`. |
| `rank_cut(op, values, *, rcond \| n_keep, closure=None) -> RankCut` | **The rank-revealing surface.** Cut a spectrum with the degeneracy-closure guard. Also `Plan.rank_cut` / `FactorToken.rank_cut`, which bind the op and the mode. |
| `cholesky_pivot_spectrum(L)`, `lu_rank_spectrum(LU)` | What a factorization's rank-revealing spectrum *is*: `\|diag L\|²` and `\|diag U\|`. Pure `jnp` on the last two axes, so one call serves a tile and a stack. |
| `CLOSURE_MODES`, `CLOSURE_DEFAULT_MODE`, `CLOSURE_RTOL`, `CLOSURE_ENV`, `RANK_SPECTRA`, `SpectralClusterError` | The guard's vocabulary. Importable with **no jax and no `.so`**, like `BACKEND_CHOICES`. |

Two phases and they stay two: only platform and handler guards can fire at
resolve time — operand dtype, rank and extent are trace-time facts — so a
single-phase API would have to lie about when it checked.

## The degeneracy-closure guard on rank cuts

**A rank cut that stops inside a degenerate cluster keeps a
symmetry-arbitrary slice of an eigenspace.** A symmetry that commutes with
the factored operator maps each of its eigenspaces onto itself and mixes
the members of a degenerate block freely, so cutting *between* whole blocks
leaves the retained span invariant, and cutting *through* one leaves a span
that differs between `q` and `Sq`, chosen by round-off. Everything built on
it loses the k-star identity. The criterion, its tolerance and the whole
saga are `common/spectral_closure` and
`tests/known_failures/2026-08-10-spectral-cut-closure.md`; what this
service adds is the wiring, on the two ops the owner named.

**The surface is new, because there was no rank decision here to guard.**
`eigh`, `cholesky` and `solve_lu` factor and solve; not one of them
returned a rank before this. So `rank_cut` is where a rank cut is made,
and `cholesky_pivot_spectrum` / `lu_rank_spectrum` say what a
factorization's rank-revealing spectrum is:

```python
p  = distrib_la.plan("cholesky", mesh, backend="native2d", closure="snap")
L  = p.batched(A)
rc = p.rank_cut(distrib_la.cholesky_pivot_spectrum(L[q]), rcond=1e-8)
keep = rc.n_keep        # >= the threshold's own answer, always
```

For Cholesky the spectrum is `|diag(L)|²` — with the pivot order fixed,
that is exactly the Schur-complement diagonal a pivoted Cholesky compares
against its tolerance. It is **squared**, and that is load-bearing: a
relative-gap criterion is not invariant under a square root, so two pivots
agreeing to 1e-6 have diagonals agreeing to ~5e-7, and reading the
diagonal instead of the pivot invents degeneracies. For LU the spectrum is
`|diag(U)|`, unsquared, because `A = P L U` already lives on the
operator's scale.

**Opt-in, and OFF by default this round.** `closure='snap'|'strict'|'off'`
on `plan()` / `factor()`, or `LORRAX_DISTRIB_LA_CLOSURE` in the
environment; the kwarg wins, and the mode resolves **once**, at plan time,
so two calls through one plan cannot disagree. Nothing about resolution,
routing, sharding, donation or the bytes a factorization returns depends
on it — `test_arming_the_closure_changes_no_resolution_fact` compares all
of those across the three modes, and `np.array_equal` on the factor itself
is an L-b cell.

Why off, when `common/spectral_closure` defaults to `snap`: the arithmetic
argument for snapping transfers unchanged, but *who may change a route*
does not. This service's route semantics are certified surface, and a
guard that arrived switched on would change the rank a shipped operation
returns without its caller asking — the shape of the silent route change
this package's own Purpose section prices at a QP gap of −161 eV.
**Owner row: flipping the default to `snap` is the owner's call; it is one
constant, `distrib_la.closure.DEFAULT_MODE`.** The dial name is
deliberately *not* `LORRAX_SPECTRAL_CLOSURE`, so that arming the ζ fit's
guard cannot silently arm this one.

**The kernel is a COPY of `common/spectral_closure`'s host face, not an
import**, because services are import-isolated from `src/` by charter and
by gate, and ~90 lines of stdlib arithmetic is cheaper than an import
edge. The copy is held by a consistency cell in `tests/test_spectral_closure.py`
— the only side that can see both — which runs thirteen hostile synthetic
spectra through both `cluster_at_cut`s at every cut and asserts the dicts
agree **field for field**, plus a cell that pins the one deliberate
divergence (the default) and one that asserts the copy imports nothing
from `common`.

**Known limits, named rather than implied.**

* `FactorToken` has **no `rank_spectrum()`**. A token's factor is opaque by
  design — block-cyclic on a specific grid — and pulling `diag` out of one
  is a distributed gather with a per-backend layout rule each. A caller
  that wants a closed rank off an FFI factor materialises it through that
  backend's own route (`cholesky_handle_to_natural_L`,
  `SlateLowerL.to_jax_lower`) and hands the spectrum in. **Owed.**
* There is **no device face**. `common/spectral_closure` has a pure-`jnp`
  `snap_keep_outward` because the ζ cut lives inside a jitted kernel; no
  rank decision in this service does, so there is one host face and
  `strict` raises where it is called.
* `|diag(U)|` is a **weaker rank revealer** than a rank-revealing QR or an
  SVD and can miss a near-null space a column-pivoted factorization would
  expose. The guard bounds where a cut may land, not whether the spectrum
  is the right one to read.

## Contract

* **Promise semantics.** A returned backend name means every guard passed:
  platform, **known-broken combinations**, compiled handler,
  one-process-per-device coverage, mesh geometry, divisibility. The call
  cannot then fail for an availability or geometry reason.
* **Explicit requests REFUSE**, with the failed guard named and a fix.
  Only `auto` demotes, and it announces once, from the rank it happened on.
  A grammar error (an unknown spelling) falls to the gate DEFAULT.
* **A COST NOTICE is not a demote.** An explicit `distributed`/`cusolvermp`
  eigh below n=16384 prints what that route costs (rank 0, once per mesh
  geometry) and then returns it unchanged. The measurement is in
  § Performance; the reason it is a notice and not a demote is the line
  above this one.
* **Exception types are part of the API**: `ValueError` / `RuntimeError`,
  each constructible from one string. `bandstructure/bse_setup.py:386-403`
  re-raises `type(exc)(_why)`; a service-specific exception class would
  escape that handler and delete the `use_low_mem_eigh` refusal message.
* **Layout.** Eigenvalues come back replicated; eigenvectors as **COLUMNS**,
  on every backend. Batched inputs are `P(None,'x','y')`, single tiles
  `P('x','y')`. `ensure_sharding` is the one place that is spelled.
* **Donation** is declared per op, not per call site:
  `eigh` donates nothing, `cholesky` donates argument 0, `solve_lu` donates
  0 and 1. A donated operand must be a fresh value at the call site.
* **Tokens are opaque.** `FactorToken` exposes `op`, `backend`, `mesh`, `n`,
  `nbatch` and no factor. "Never reshard it, feed it back verbatim" is the
  type rather than a comment. It is deliberately **not** a JAX pytree, so a
  `jit` boundary refuses it by name instead of tracing a block-cyclic
  handle (pinned by `test_a_factor_token_cannot_enter_the_composed_kernel`).
* **`solve()` checks `B` against the token's `n` and `nbatch`**, so a
  mismatched RHS refuses instead of corrupting a solve or hanging in a
  collective.
* **`factor`/`solve` is not the fused route.** `factor('solve_lu', …)`
  raises `NotImplementedError` for any backend but ScaLAPACK; one factor per
  solve is `plan('solve_lu', mesh, backend=…).batched(A, B)`.
* **Env grants capability, never selects a backend.** `LORRAX_FFI_SO` /
  `LORRAX_FFI_HOST_SO` pin which `.so` to open. `distrib_la.resolve` reads
  no environment at all. An explicit pin that cannot be honoured is a
  refusal, never a fall-through.
  `LORRAX_DISTRIB_LA_CLOSURE` is the one other variable this package reads,
  and it is not an exception to that rule: it selects no backend and
  changes no route — it arms a rank-cut guard on a surface that returns no
  matrix. `resolve.py` still reads nothing.
* **A rank cut never shrinks.** Whatever `rank_cut` returns is at least
  the proposal it was given; the closure guard only ever moves a cut
  outward. Sizing the retained set stays the caller's `rcond`, and this
  guard only refuses to let that size land mid-block.

## Backends

Preference is declarative data with every supported platform represented,
including the declared-untested tier.

| op | vocabulary | `auto` | `distributed` resolves to |
|---|---|---|---|
| `eigh` | `auto off distributed cusolvermp slate scalapack` | `native` | cpu → **scalapack**, CUDA → **cusolvermp**, ROCm → **slate** |
| `cholesky` | `auto off native2d cusolvermp slate` | `native` | — (its CPU story is a channel-policy ladder in the caller, not one library) |
| `solve_lu` | `auto off distributed cusolvermp scalapack` | `native` | cpu → **scalapack**, CUDA → **cusolvermp** |

* `native` is the floor everywhere and the measured default at every
  production tile size. `auto` never picks an FFI backend.
* `native2d` is the 2-D block-distributed tiled Cholesky (`_native2d`, the
  old `common/cholesky_2d`). Pure JAX, every platform, but a *different
  algorithm*: at n=10k on P=128 the replicated-reshard route costs 1.6
  GB/device and this one 5 MB/device. `auto` never picks it — a cost model
  that differs by three orders of magnitude in **both** directions is the
  caller's decision.
* **ROCm is DECLARED-UNTESTED.** The preference rows exist so the routing
  question has an answer; LORRAX builds no ROCm `.so`. Known measurement
  gap, recorded in `resolve.py`: on the jaxes this tree runs,
  `Device.platform` is `'gpu'` for both vendors, so a real ROCm mesh lands
  on the CUDA row. Disambiguating is **one row** and needs a machine.
* **SLATE `heev` on a host mesh is refused outright** — bug L-2,
  deterministic SIGSEGV against MKL/LibSci LAPACK. Two contract cells carry
  the skip by design.
* **SLATE `eigh` on a MULTI-RANK CUDA mesh is refused at n ≥ 4096** — bug
  L-4, the CUDA sibling of L-2: SIGSEGV, `srun` rc=139, every rank of the
  job down (jobid 56457930). Size-scoped, because it *returns* at
  n ≤ 2048 on the same mesh. A 1×1 CUDA mesh at that size is unmeasured
  and therefore not refused.
* **cuSOLVERMp `eigh` with `compute_evecs=False` is refused** — bug L-3,
  `cusolverMpSyevd` status=7 at every n. A library defect (`bufferSize`
  succeeds; the handler only forwards the flag). Permanent, not a
  stopgap: LORRAX wants `compute_evecs=True` everywhere (owner,
  2026-08-07).
* **ELPA — REGISTERED CANDIDATE, not built, not wired.** Neither shipped
  distributed eigensolver is good in the regime that matters: cuSOLVERMp
  has a ~1.55 s per-matrix floor, and SLATE has a *cheaper* floor
  (0.40 s at n=64) but worse asymptotics (5.44 s at n=2048, already 2.8×
  cuSOLVERMp) and it **crashes at 4096** — i.e. it dies exactly where a
  distributed eigh becomes necessary. Building ELPA as the "dream stack"
  eigensolver is under consideration by the owner. Nothing in this tree
  depends on it; the row exists so the question has a written answer.
* **There is no ScaLAPACK `potrf` handler anywhere in the tree**, and no
  SLATE `getrf`. The vocabularies above are the whole truth.
* **The two platform `.so`s share their SLATE.** Both `liblorrax_ffi.so` and
  `liblorrax_ffi_host.so` carry `NEEDED libslate.so.2` and `NEEDED
  libblaspp.so.2`, resolved out of different builds; `ld.so` keys a loaded
  object by SONAME, so the first one opened decides which
  `blas::get_device_count()` the other calls, and the host build's is a
  compiled-in 0. Both loaders therefore open **CUDA before cpu**
  (`_open_cuda_before_host`). See Antipatterns.

## Tests

Four tiers, markers `services` + `distrib_la`, applied by a collection hook
(a `pytestmark` in a conftest is silent — `tests/test_service_selection.py`
measures that the marks arrived).

| tier | file | needs |
|---|---|---|
| L-a shape/contract algebra | `test_distrib_la_shape_algebra.py` | nothing — a laptop, milliseconds |
| L-a rank-closure guard | `test_distrib_la_closure.py` | nothing — the criterion is stdlib arithmetic and the plan wiring resolves to `native` on a fake mesh |
| L-b emulated multi-device | `test_distrib_la_emulated_mesh.py` | `XLA_FLAGS` set by the SERVICE conftest; **skips**, never asserts, below 4 devices |
| L-c real multi-process | `test_distrib_la_multiproc.py` | `srun -n 4`; shared `check_*(mesh, …)` bodies + a `__main__` CLI (`_CLI_CELLS`) — same functions, no duplicated logic |
| contract + wiring | `test_distrib_la_contract.py` | the `.so` pins; every refusal constructibly fires |
| C++ / ELF acceptance | `test_so_acceptance.py` | binutils + a pinned `.so`; reads the ELF, never dlopens |
| import isolation | `test_distrib_la_import_isolation.py` | `python -S` subprocess; `sys.modules` AND `sys.path` asserted, plus a red twin and a with-lorrax-still-passes |
| skip honesty | `test_distrib_la_skip_honesty.py` | a machine profile; ABSENT = skip, BUILT-AND-BROKEN = **FAIL** |

* **Hostile geometry is mandatory**: a real 2×2 with non-dividing extents
  and padding round-trips, with the anti-tautology self-assertion (the pad
  divisor must be provably non-vacuous).
* **Every check ships with the case where it returns FALSE.** No exceptions.
* Run it standalone: `pytest services/distrib_la/tests` (never loads
  `tests/conftest.py`). Run it from the monorepo: `pytest -m distrib_la`.
  Deselect: `--no-services` / `--only-service=NAME`, never a second `-m`
  (`pyproject` sets `addopts = "-m 'not extra'"` and an explicit `-m`
  REPLACES it, silently re-enabling 26 deselected suites).
* Perlmutter floor, 2026-08-07, HEAD `eeece71`, BUILD_NOTES pins: full-suite
  `-m distrib_la` **130 cells / 0 failed / 22 skipped** (was 124 / 8 failed
  before the SONAME fix); service-only by path **250 / 0 / 3**, of which
  lxkit is 120 and unchanged.

## Performance

Recorded baselines (never slow tests) in
`services/distrib_la/bench/baselines/{cpu,gpu}{1x1,2x2}.json`: one row per
(op × backend × shape), with `seconds`/`min`/`max`, `compile_seconds`, the
resolved backend, and — for a combination that cannot run — the refusal
text instead of a number. All rows below: Perlmutter, **jobid 56447670**,
1 node, jax `0.7.0.dev20260807`, complex128, BUILD_NOTES `.so` pins.
Regression detection = diffing baseline files across branches.

> The jobid above read **56444350** until 2026-08-07. That was stale — it
> is the step-2 L-c multiproc leg, not the leg the committed baselines
> came from. Every row in all four baseline files carries `"jobid":
> "56447670"`, which is the number now printed here.

> **The eight `cholesky`/`slate` rows in the baseline files are now
> PESSIMISTIC and have not been regenerated.** They were measured against
> the per-q Python loops in `factor`/`solve`, which re-compiled SLATE's
> kernel once per matrix; those loops are scans as of
> `feat/batched-canonical-2026-08-08` and § "The batched surface is a
> scan" measures the same route 35× faster on gpu 2×2. Regenerating the
> sweep is a perf leg on a shared GPU pool and is deliberately NOT folded
> into that change — read the A/B table for this route, not
> `baselines/*.json`, until a regeneration leg lands.

**`auto` resolves to `native` everywhere, and the numbers say why.**

| op | backend | mesh | shape (nq, n) | s |
|---|---|---|---|---|
| eigh | native | cpu 1×1 | (2, 1024) | 1.446 |
| eigh | scalapack | cpu 1×1 | (2, 1024) | 1.451 |
| eigh | scalapack | cpu 2×2 | (2, 1024) | 1.099 |
| eigh | native | gpu 1×1 | (2, 1024) | 0.044 (via `plan`, `cusolvermp` 0.0435) |
| cholesky | native2d | gpu 2×2 | (8, 256) | 0.022 |
| cholesky | native2d | cpu 2×2 | (2, 1024) | 0.375 |

**cuSOLVERMp `eigh` on a 4-process 2×2 costs a flat ~1.55 s PER MATRIX,
independent of size.** That is the collective, not the factorization:

| shape (nq, n) | total s | s / matrix |
|---|---|---|
| (2, 64) | 3.117 | 1.56 |
| (2, 256) | 3.201 | 1.60 |
| (2, 1024) | 3.525 | 1.76 |
| (8, 256) | 12.713 | **1.59** |

64×64 and 1024×1024 cost the same. A per-matrix cost that does not move
with n⁴× the work is a fixed collective/context charge.

### The crossover, and where the ~1.55 s actually goes

Step 6's eigh investigation (jobid **56447670**, 4 processes on a real
2×2 A100 mesh, complex128, warm medians, one matrix per call). Rows are in
`baselines/{cpu,gpu}2x2.json` tagged `"leg": "step6.cross_size"`.

| n | native replicated | cuSOLVERMp | SLATE | cuSOLVERMp / native |
|---|---|---|---|---|
| 64 | **0.00149** | 1.586 | 0.401 | 1064× |
| 256 | **0.00412** | 1.561 | 0.546 | 378× |
| 1024 | **0.02662** | 1.754 | 1.387 | 66× |
| 2048 | **0.07275** | 1.932 | 5.444 | 27× |
| 4096 | **0.39550** | 2.739 | **SIGSEGV** | 6.9× |

**Is SLATE the reasonable one?** No. It has the cheaper floor — 0.40 s vs
1.59 s at n=64, and it beats cuSOLVERMp up to n≈1024 — but its scaling is
worse (5.44 s at 2048, already 2.8× cuSOLVERMp) and it **dies at 4096**,
which is where distributed eigh starts to be the thing you actually need.
Faster floor, worse asymptotics, crashes in the capacity regime. That is
the whole reason ELPA is a registered candidate (§ Backends).

**The ~1.55 s is not ours and is not reachable.** `LORRAX_FFI_PROFILE=1`
splits the warm n=64 call as: `cusolverMpSyevd` **99.998%**, `plan()` +
`resolve_backend` 9.6 µs, cuSOLVERMp context-cache HIT 0.72 µs,
descriptors + `bufferSize` ≤ 0.02 ms. Cold-only extras, outside the warm
number: one-time context bootstrap 0.66–0.81 s, XLA compile ~0.79 s.
There is no in-tree edit that moves this. `block_size` does not either
(64/128/256/default at n=4096: 2.44/2.41/2.44/2.70 s).

**CPU tells the same story without needing an extrapolation.** ScaLAPACK
`pzheevd` never beat native replicated on one node, and the gap widens:

| n | native replicated | scalapack | ratio |
|---|---|---|---|
| 64 | **0.00063** | 0.00255 | 4.0× |
| 256 | **0.01626** | 0.01717 | 1.06× |
| 1024 | **0.27120** | 0.55437 | 2.04× |
| 2048 | **1.48904** | 4.12157 | 2.77× |

`auto` → `native` (`resolve.py`) is therefore **vindicated on both
platforms**, at every size anyone has measured. Do not change it.

### `distributed` eigh is for CAPACITY, not speed — and that regime is UNMEASURED

Every row above **fits on one device**, so every row above is a regime
where a distributed library has no reason to win. Extrapolating two fitted
power-law exponents puts break-even near **n ≈ 1.9 × 10⁴**, which is
**4.6× past the largest n measured (4096)** and is sensitive to the fit
window (last-three-points ≈ 1.4 × 10⁴; all-five ≈ 2.8 × 10⁴). Read the
**decade**, never the digits. It lands in the same decade as the
single-device capacity wall: an n×n complex128 matrix plus its
eigenvector copy and workspace stops fitting in 40 GB around n ≈ 2.7 × 10⁴.

So the honest summary is: *below the capacity wall, native wins by
between 1064× and 6.9×; at and above it, native cannot run at all and
nobody has measured what the alternatives cost.* An explicit
`distributed`/`cusolvermp` eigh below n=16384 now prints that on rank 0,
once per mesh geometry, and **still runs exactly what was asked for** —
explicit requests are never demoted.

### Open measurement gaps (inherit this map; do not re-derive it)

1. **MULTI-NODE — the biggest gap by a distance.** Every number in this
   document is one node, 4 ranks. The case `distributed` exists for is a
   mesh spanning nodes, and there is not one measurement of it.
2. **The capacity regime itself** (n ≳ 2.7 × 10⁴, matrices too large for
   one device). Unmeasured, so the crossover stays an extrapolation.
3. **float64 rows.** Everything here is complex128. The real-symmetric
   path is exercised by contract cells but has no baseline row.
4. **`compute_evecs=False`.** Refused on cuSOLVERMp (bug L-3), so its
   cost is unknown on every backend — including the ones where it works.
5. **A genuine CPU partition.** The "cpu" legs ran on the GPU pool's nodes
   with `JAX_PLATFORMS=cpu`; the Milan CPU partition has one census leg
   (jobid 56446562) and no perf leg.
6. **Non-square 1-D meshes.** ScaLAPACK accepts them (square *blocks*, not
   a square grid) and nothing has been timed on one.
7. **Batched vs serial, split by backend.** ~~Never isolated.~~ Partly
   closed by the A/B in § "The batched surface is a scan": on cpu 2×2 at
   nq=8/n=64 the scan route costs 0.0115 s against the ScaLAPACK stacked
   entry's 0.01125 s — 2% for a route with no C++ batching at all — so the
   stacked entry's advantage at THIS shape is small. What stays open is
   whether that holds at large `nq`, where one descriptor and one
   workspace should start to tell.
8. **The (2048, 4096] window for bug L-4.** SLATE CUDA eigh's true crash
   threshold is somewhere in there; 4096 is just the smallest size anyone
   watched it die at.
9. **The 4×4 control at n = 8192.** Owed to the 3×3 hang written up at the
   top of this document. Until someone runs it, we cannot say whether the
   hang is a non-power-of-two-grid problem or a "anything past 2×2"
   problem, and those two have very different blast radii. Root-causing
   the hang itself needs a multi-node GPU allocation.

### The batched surface is a scan, and the route toggle is one place

**BUILT.** `Plan.batched` is a `lax.scan` over this package's own
single-matrix operation, at `BATCHED_SCAN_UNROLL = 1` — a named constant,
deliberately changeable, changed only with a measurement behind it.
Backend-native stacked entry points (ScaLAPACK's eigh `many`, cuSOLVERMp's
`batched_potrf`, both `batched_solve_lu`) are backend-internal
optimizations **behind that same interface**, never a second public
surface. `dispatch_batched_eigh` used to be that second surface — it
carried its own `getattr` capability probe and its own serial loop — and
is now `plan(...).batched(A)`.

The point is not the loop. A Python loop over `nb` matrices is `nb`
separate calls the compiler never sees together, and there is nowhere in
it to put "run this batch some other way". A scan is one node, so the
choice collapses to **`Plan.batched_route`, the one place that decides**:

* **(a)** `ROUTE_SCAN` — a scan of the *distributed* single-matrix op.
  The default, and the definition of the surface.
* **(b)** `ROUTE_BACKEND_BATCHED` — the backend's stacked entry where the
  library has one. Its saving is in C++, around ONE descriptor and ONE
  workspace, and no scan can recover it.
* **(c)** `ROUTE_BATCH_RESHARD` — **RESERVED. Not built; asking for it
  raises and says so.** Reshard `(q, μ_x, ν_y) → (q_xy, μ, ν)`, run the op
  locally with the native jax kernel per matrix, reshard back. This serves
  the very-large-`N_μ` doctrine right up to single-matrix capacity
  **without paying any distributed-library fixed cost** — which the tables
  above show is the entire cost below n ≈ 10⁴.

(a)/(b)/(c) behind one toggle point is how small-system, non-distributed
linalg happens **without a parallel API**.

**Route (c) needs no new movement primitive.**
`common.staged_reshard.face_to_batch_reshard` already implements exactly
that exchange: `P(None,'x','y') → P(('x','y'),None,None)` as two
single-mesh-axis `all_to_all`s, because the one-step move is not a tile
permutation and GSPMD silently degrades it to replicate-then-partition
(measured 64× per-rank residency blow-up, job 7882974). Divisibility
rules and both schedules are documented in that module. The remaining work
is **wiring distrib_la's batched ops to it, not inventing movement.**

#### What the restructure cost and bought

A/B on the same nodes with the BUILD_NOTES pins, `origin/main` (`21d68e0`)
against `feat/batched-canonical-2026-08-08`, real 4-process 2×2 meshes,
nq=8, n=64, complex128. `compiles` is XLA compilations on the cold call,
counted off jax's own compile log; `warm` is the median of three
subsequent calls.

| leg | case | route | compiles | cold s | warm s |
|---|---|---|---|---|---|
| cpu 2×2 | eigh, `main` | Python loop | 12 | 0.307 | 0.0237 |
| cpu 2×2 | eigh, branch | **(a) scan** | **1** | 0.109 | **0.0115** |
| cpu 2×2 | eigh, `main` | (b) stacked | 1 | 0.063 | 0.01130 |
| cpu 2×2 | eigh, branch | (b) stacked | 1 | 0.063 | 0.01125 |
| cpu 2×2 | SLATE cholesky factor+solve, `main` | 2 Python loops | 165 | 3.370 | 3.2578 |
| cpu 2×2 | SLATE cholesky factor+solve, branch | **2 scans** | **3** | 0.177 | **0.0087** |
| gpu 2×2 | eigh/cuSOLVERMp, `main` | Python loop | 11 | 14.250 | 12.518 |
| gpu 2×2 | eigh/cuSOLVERMp, branch | **(a) scan** | **1** | 14.177 | 12.702 |
| gpu 2×2 | SLATE cholesky factor+solve, `main` | 2 Python loops | 165 | 6.234 | 5.887 |
| gpu 2×2 | SLATE cholesky factor+solve, branch | **2 scans** | **3** | 0.841 | **0.166** |

**Every result is BIT-IDENTICAL across the two trees** — W and Z for both
eigh legs and X for the cholesky legs, `np.array_equal` true, on cpu and
on gpu. This is an interior restructure and the arrays say so.

The eigh/cuSOLVERMp row is the only one where the branch is not faster on
a single run, and repeating it says it is not slower either. Warm seconds
over three alternating runs on the same allocation:

| run | `main` | branch |
|---|---|---|
| 1 | 12.518 | 12.702 |
| 2 | 12.613 | 12.459 |
| 3 | 12.537 | 12.503 |
| mean | **12.556** | **12.555** |

The ordering lands on the wrong side of the difference as often as the
right one. That is what a **flat per-matrix collective charge** looks
like: § "The crossover" measures cuSOLVERMp at 1.56/1.60/1.76/1.59 s per
matrix across four shapes with **99.998% of it inside
`cusolverMpSyevd`**. Nothing in that leg is ours to move, and the scan
did not move it.

**The compile count is the result to read.** The old serial route compiled
per iteration on any wrapper without a jit cache; the scan compiles once,
whatever `nq` is. SLATE is where that mattered most —
`_slate.distributed_cholesky` and `_slate.distributed_trsm` build their
`shard_map` at eager top level with no `jax.jit` around it and no
per-signature cache, the shape every other wrapper here has and those two
do not, so an `nbatch`-long Python loop re-traced and re-compiled the
kernel `nbatch` times. 165 compiles → 3, and 5.9 s → 0.17 s of GPU wall.

**An eager scan is not enough, and the first measurement said so.**
Without a `jax.jit` cache per signature the scan is correct and *slower*:
the cpu eigh row came out at 0.080 s warm against 0.024 s for the Python
loop, because `lax.scan` called eagerly re-traces and re-lowers the whole
loop every call while the Python loop's `nq` backend calls each landed in
the wrapper's own `_JIT_CACHE`. The same scan as a pre-compiled executable
timed 0.0114 s, so ~69 ms of that 80 was pure retrace. `plan._SCAN_CACHE`
is the fix and it is keyed on `mesh_key`, which is strictly finer than the
`(Px, Py)` the backends key their MPI/NCCL contexts on — a cache that
cannot hand back an executable whose baked-in context handle has moved on.

**`lax.scan` over a distributed FFI call is viable on BOTH platforms, and
CUDA was the open question.** The host answer was already recorded (job
7889132, ScaLAPACK, P=4). The route that actually *ships* is cuSOLVERMp
and SLATE over NCCL + `cal_comm`, whose failure mode is a HANG with no
traceback, and no CUDA mesh had ever been pointed at it. It traces,
compiles and runs: `eigh/cusolvermp` above is that route, on a real
4-process 2×2 A100 mesh, residual 2.296e-15.

*Experiment note, not a plan:* a possible **fourth** route, for the
capacity regime only (matrices too large for one device **and**
latency-bound) — overlap `k` independent distributed ops from a C++-side
context pool. Feasible on CUDA (a second cuSOLVERMp context costs
**0.065 s**, measured) but NCCL needs one communicator per concurrent
slot; off-limits on the host leg (Cray MPICH's default
`MPI_THREAD_SINGLE`, with documented deadlock history at `H5Fclose` —
`src/ffi/phdf5/ARCHITECTURE.md`); contention-bound at compute-heavy sizes
anyway. ELPA targets the same slice properly.

**Any such restructuring must keep the batched-vs-serial bit-identity
gate green** (`tests/multi_device/batched_eigh_dispatch_gate.py`, and its
adopted twin `check_batched_eigh_dispatch` in the L-c suite: the two
routes agree to **0 ulp** in both W and Z — job 7889132 at nq=6/n=32, and
again at nq=8/n=64 with the scan on both sides of the comparison). That
gate is the guard on this whole design, and it is why route (c) is safe
to add later: it can prove a new route did not change an answer. Its
`_force_serial` argument is `Plan.batched`'s private `_route` override —
the toggle is what makes the gate able to run two routes over one set of
operands at all.

**SLATE `factor`+`solve` (potrf + two trsm passes), the route H3 moved
onto**, cpu 2×2: 0.727–2.945 s; gpu 2×2: 1.397–7.237 s. Its
**correctness** is the load-bearing result, and it is first-execution
evidence — the SLATE trsm back-solve had never run before step 2:

| leg | residual vs the native reference | bar |
|---|---|---|
| SLATE trsm, CPU c128, per-q | 3.8e-16 / 4.5e-16 | rtol 1e-12, NOT relaxed |
| SLATE trsm, GPU c128, per-q | 1.0e-15 / 6.9e-16 | " |
| SLATE token round trip, GPU 2×2 | 4.4e-16 | " |
| SLATE token round trip, CPU 2×2 | 4.3e-16 | " |
| cuSOLVERMp token round trip, GPU 2×2 | 5.6e-16 | " |
| ScaLAPACK LU + ipiv, CPU 2×2 | 3.855e-13 | native LU control 3.856e-13 |
| batched-vs-serial eigh | **bit-identical** (0.0 in W and Z) | — |

## Antipatterns

* **Editing `src/ffi/<name>/`.** `src/ffi/linalg/`, `src/ffi/slate/`,
  `src/ffi/scalapack/` and `src/common/cholesky_2d.py` are **deleted**
  (17 files, commit `b3f3675`). A backend lives in
  `services/distrib_la/src/distrib_la/_<name>.py` plus one row in each of
  `loader._*_TARGET_SYMBOLS`, `resolve._SPEC`/`BACKEND_CHOICES` and
  `plan._IMPL`. `docs/dev/linalg_ffi.md` § "Adding a backend" is the
  procedure; the C++ under `src/ffi/cpp/` did not move and its target
  strings are frozen.
* **Writing a Python loop over the batch axis.** That is what
  `Plan.batched` is for, and looping outside it hands the compiler `nb`
  separate calls, re-traces any wrapper that builds its `shard_map`
  eagerly, and — the real cost — puts the batch somewhere no route toggle
  can reach. Measured on the two SLATE wrappers that have no jit cache:
  165 compiles and 5.9 s of GPU wall for an 8-matrix factor+solve, against
  3 compiles and 0.17 s through the scan.
* **Adding a second batched entry point for a backend that has one.** A
  stacked FFI entry is one `plan._IMPL` row and it is then taken
  automatically; exposing it as its own public function recreates the
  `dispatch_batched_eigh` split, where two places decided the same thing
  and one of them forgot the eigenvector normaliser.
* **Calling `lax.scan` over an FFI wrapper eagerly, uncached.** It is
  correct and it is slower than the loop: ~69 ms per call of retrace and
  relower at nq=8/n=64 on cpu 2×2. Route it through `Plan.batched`, which
  owns `plan._SCAN_CACHE`.
* **Selecting a backend from the environment.** There is no env var that
  does it and adding one is the antipattern. Deck keys choose
  (`eigh_backend`, `distributed_cholesky`, `distributed_lu`,
  `w_dyson_solver`); env only says which `.so` exists.
* **Comparing Z across meshes.** Eigenvectors are gauge-dependent: a
  degenerate subspace has no canonical basis and two meshes will return
  different (equally correct) columns. Compare **eigenvalues**, or a
  gauge-invariant contraction (`Z diag(W) Zᴴ`, subspace projectors,
  `|Zᴴ Z'|`). A test that diffs Z across meshes will pass on Si and fail on
  the first degenerate system.
* **Reading token internals.** `token._factor` is private and its layout is
  block-cyclic on that mesh's specific grid. Reaching for `.shape` on a
  token is the same error one level up — it has `.n` and `.nbatch`.
  Anything that needs the factor's *bytes* wants a different API, not this
  one.
* **Branching on `token.backend` / `plan.backend` outside this package.**
  They are introspection (a banner, a test's message). A caller that
  branches on them has re-implemented the resolver, and the two will drift.
* **Re-deriving a mesh identity by hand.** Use `mesh_key(mesh)`. `id(mesh)`
  as a cache key is only safe when the cached value retains the mesh, and
  the case where it does not is exactly where somebody re-spells it.
* **Bypassing the `sys.path` bootstrap.** `services/*/src` is on no
  `PYTHONPATH` any launcher sets: `lx` rewrites the container `PYTHONPATH`
  to exactly `<checkout>/src` and the Shifter image pip-installs nothing.
  A module that imports `distrib_la` must call `ffi._services.ensure_on_path()`
  first (`tests/test_service_path_bootstrap.py` runs the bare-`sys.path`
  import in a subprocess). This is transitional plumbing with an owner
  decision behind it — do not paper over it with a `sys.path.insert` of
  your own, and do not assume a green pytest leg proves it: the service
  conftest puts services on the path during collection, so a broken
  bootstrap is a **green-suite / red-cluster** failure.
* **Assuming the two platform `.so`s are independent.** They share
  `libslate.so.2` / `libblaspp.so.2` by SONAME. Opening the host library
  first gives every CUDA SLATE handler a `blas::get_device_count()` of 0
  and a `FAILED_PRECONDITION` naming a device count nobody set. Do not
  "fix" that by unsetting `CUDA_VISIBLE_DEVICES` — measured, it is not a
  visible-device problem; the failing leg had exactly one visible GPU.
* **Making the refusal path a `try/except` at the call site.** An explicit
  backend that cannot be honoured must raise out of the driver. Six probe
  calls in `isdf/core.py` exist only for their raise; wrapping any of them
  turns a loud refusal into a silent different-backend run
  (`tests/test_charge_zeta_route.py` pins all six).
