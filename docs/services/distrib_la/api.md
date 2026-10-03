# distrib_la: the caller's API

`distrib_la` is LORRAX's one entry point for dense linear algebra on a 2-D
device mesh: Hermitian eigensolves, Cholesky and LU, GEMM, the polar factor,
and the fixed-capacity subspace algebra of iterative eigensolvers. This page
is for anyone writing a stage that calls it: the layouts it accepts, what each
call computes, and what it promises. Which vendor library runs underneath, and
how the deck chooses, is [Backends](backends.md); the subspace algebra is
[Active subspace](subspace.md). Read [the three levels](../../architecture/layers.md)
first if you have not met the service rule.

## The mesh and the layouts

Every call takes a `jax.sharding.Mesh` with axes `('x', 'y')` of extents
`P_x × P_y`, `P = P_x·P_y` devices. The FFI routes also need one JAX process
per device, with process `x·P_y + y` on cell `(x, y)` (y-minor order), because
each library builds its own MPI or NCCL grid from the process ranks. LORRAX
builds only square meshes, `P_x = P_y = p`.

A matrix never lives whole on one rank unless the caller asks for it. The
service names five layouts:

| layout | shape | `PartitionSpec` | per-rank piece |
|---|---|---|---|
| face, one matrix | `(n, n)` | `P('x','y')` | `(n/P_x, n/P_y)` |
| face, a stack | `(B, n, n)` | `P(None,'x','y')` | `(B, n/P_x, n/P_y)` |
| batch | `(B_p, n, n)` | `P(('x','y'),None,None)` | `B_p/P` whole matrices |
| axis (GEMM operands) | `A (q, m, k)`, `B (q, k, n)` | `P(None,'x',None)`, `P(None,None,'y')` | rows of A, columns of B, complete `k` |
| row face (`contract_faces`) | `(b, m, K)` | `P(None,'x',None)` / `P(None,'y',None)` | rows on one axis, complete `K` |

`B_p = ⌈B/P⌉·P` is the batch padded to the device count; pad rows are zero.
Face extents must tile the mesh (`n % P_x == n % P_y == 0`); the service never
pads a matrix face, because a padded face changes the factor the caller gets.
Pad the matrix yourself and slice afterwards. Operands are `float64` or
`complex128`.

## Installing and importing

`services/distrib_la/` is an installable package (src layout, Python ≥ 3.12)
whose runtime dependencies are `lxkit`, JAX and NumPy; it imports nothing from
LORRAX's `src/`.

```bash
cd services/distrib_la
python -m pip install -e ../lxkit -e .
```

Import top-level names only. `from distrib_la.<submodule> import …` from
outside the package fails `tests/test_layering.py`, because the package is its
own facade: ScaLAPACK, SLATE, cuSOLVERMp and cuBLASMp are reached through one
module, `distrib_la.loader`, and nowhere else. Native JAX routes, the
vocabularies and the capability report work with no native library present;
the FFI routes need the LORRAX native pair ([Backends § where the libraries
come from](backends.md#where-the-libraries-come-from)).

## Two phases and the promise {#contract}

Every planned entry point (`plan`, `gemm_plan`, `plan_polar_factor`,
`plan_subspace`) runs in two phases. The **eager** phase resolves the backend:
it opens the native library, probes the handler, reads `jax.process_count()`
and checks the mesh. What it returns is **trace-safe**: it can be called inside
a caller's `jit` or `lax.scan`, and checks only operand dtype, rank and extent.
The split exists because availability is a host fact and operand shape a trace
fact; one phase could not check both at the right time. Hoist plans out of
loops: the first FFI call also builds a BLACS or cuSOLVERMp context and
compiles a module, a cost paid once per plan that a plan built inside a loop
would pay on every iteration.

**A returned backend is a promise.** When `resolve_backend` returns a name,
every guard has passed (vocabulary, platform, known-broken combination,
compiled handler, one process per device, mesh geometry, divisibility of `n`),
so the call cannot later fail for an availability or geometry reason. An
explicit request that cannot be honoured refuses at resolve time, naming the
failed guard and the fix. Only `auto` may demote, and it announces the demotion
once on rank 0. A silent route change is treated as the worst failure the
service can have, because it runs to completion with plausible numbers.

Refusals are `ValueError` or `RuntimeError` (`TypeError` /
`NotImplementedError` for misuse), each constructible from one string: callers
such as `src/bandstructure/bse_setup.py` re-raise `type(exc)(why)` with added
context, which a service-specific exception class would break.

## Dense factorizations: `plan`, `factor`, `solve`

```python
import distrib_la as dla

eigh = dla.plan('eigh', mesh, backend='auto', n=n)   # eager: resolve once
w, z = eigh.batched(a_stack)                          # trace-safe
```

`plan(op, mesh, *, backend='auto', n=None, batched_route=None,
budget_bytes=None) -> Plan` resolves `op ∈ {eigh, cholesky, solve_lu}`.
Without `budget_bytes` the route defaults to `'batch_reshard'`; with it and no
route, the service decides each eigh stack ([the eigh-stack API](#eigh-stack)).
Passing `n` (independent of any operand, so a caller that will pad can ask
first) runs the divisibility guard at resolve time.

| call | operands | result |
|---|---|---|
| `Plan(A)` | one matrix at `P('x','y')` | `eigh`: `(w, Z)` |
| `Plan.batched(A_stack, …)` | a stack at `P(None,'x','y')` | the same per matrix |
| `Plan.native_fn` | — | a pure trace-safe closure, native backends only, for a caller that needs the math inside its own `jit` |
| `Plan.backend`, `.is_native`, `.batched_route`, `.route_for(shape, dtype)`, `.stack_route(shape, dtype)`, `.describe()`, `.donates` | — | the resolved facts, for reports; a caller never branches on them |

**Conventions, identical on every backend.** Eigenvalues return ascending and
replicated; eigenvectors return as columns, `A Z = Z diag(w)`, in the input's
face layout. `ensure_sharding` is the only reshard on entry: a tracer gets a
sharding constraint, an array already in layout passes untouched, anything
else is placed process-locally.

**No distributed result is returned unchecked** (`_result_check`). The
distributed libraries fail silently. cuSOLVERMp 0.9.1's syevd returned wrong
eigenvectors with status 0 and info 0 (2×2 grid, reproduced outside LORRAX) on
a Hermitian matrix with 680 exact-zero rows (n 2688, block 224) and on
rank-deficient PSD responses with no zero row (n 432, most block sizes). So:

- Every eigh route replaces exact-zero rows by distinct diagonal sentinels
  below the Gershgorin bound and returns them as exact zero eigenpairs, in
  ascending order (`deflate_zero_rows`; a matrix with no zero row passes
  through unchanged).
- Every distributed eigh (cuSOLVERMp, SLATE, ScaLAPACK; single, scanned or
  stacked) is checked against 8 fixed-seed probe columns X, the same on every
  rank: ‖(AZ − Z diag(w))X‖/(‖A‖‖X‖) and ‖(ZᴴZ − I)X‖/‖X‖ within
  `roundoff_tol(n)` = 64·n·eps, at O(n²k) beside the O(n³) solve. A failed check is noted on rank 0's
  stderr and solved again: shifted (A + ‖A‖_F I, which moves a large
  near-zero cluster off the origin), then in cuSOLVERMp's other layout with
  and without the sentinels, then gathered on every rank when its compiled
  program and the solver's workspace fit 1 GiB (`GATHERED_EIGH_BYTES`,
  n ≤ 3096; agreed over ranks). XLA reserves a retry branch's temporaries
  in every program that holds it, so a caller's room does not raise this.
- Every route-(c) eigh is checked the same way on its face-layout result,
  against the Hermitian part of the input (the local solver symmetrizes). A
  failed check solves again shifted, then gathered when it fits (the rule
  above), and refuses by name. Measured cost at P4: 0.3–4.1 ms per eigh at
  n = 3328–18304 (at most 0.2 % of the solve).
- Every distributed LU solve (`plan('solve_lu').batched`, `factor`/`solve`)
  is checked on its actual solution through sketches taken before the library
  consumes A and B: ‖Wᴴ(AX − B)‖/(√k (‖A‖‖X‖ + ‖B‖)) within `roundoff_tol(n)`. Its
  operands are gone, so there is no retry.
- A result that still fails prints `GATE distrib_la_result_check` (op, n, the
  call site, the errors) on rank 0's stderr and comes back NaN-poisoned on
  every rank, so a caller's finite-result gate refuses; an eager call raises
  the GATE itself. The verdict is one replicated scalar, so every rank takes
  the same branch. (A raise inside the host callback is not used: it is an
  unordered effect, which each rank meets at a different point.)

The bounds are backward errors, which a stable solver keeps near n·eps
whatever the condition number. `services/distrib_la/bench/eigh_zero_block_check.py`
and `eigh_conformance_check.py` (n/p ∈ {257, 389, 1144, 4576}, P4 and P16) are
the regression checks. The cuSOLVERMp handlers do not yet read `info`; that
is a native fix for the next bundle.

**Donation is declared per operation** (`DONATES`), because a caller must know
whether its buffers survive before it knows which library runs: `eigh` donates
nothing, `cholesky` operand 0, `solve_lu` operands 0 and 1. A donated operand
must be a fresh value at the call site.

**Factor once, solve many.** `factor(op, A, mesh, *, backend, …) ->
FactorToken` and `solve(token, B)` split an LU or Cholesky so one factor serves
many right-hand sides (getrf once, getrs per call). The token exposes `op`,
`backend`, `mesh`, `n` and `nbatch` and hides the factor (ScaLAPACK's `ipiv`,
cuSOLVERMp's raw buffer, SLATE's lower factor), because that factor is
block-cyclic on that mesh and only its own library can read it. A token is not
a pytree, so passing it through `jit` refuses by name; `solve` checks `B`
against `n` and `nbatch`. `plan('solve_lu').batched(A, B)` is one complete
factor and solve per call.

**Native Cholesky and LU belong to the caller.** Under `backend='native'`,
`Plan(A)` implements `eigh` only; `cholesky` and `solve_lu` raise
`NotImplementedError`, because their native form (a replicated factor, a
ridged per-q solve) is a physics channel policy the service does not own.
Reach them through `batched_route='batch_reshard'`, `backend='native2d'`
(Cholesky), or an FFI backend.

### How a stack runs: the batched routes {#batched-routes}

`Plan.batched_route` is the one place that decides how a stack of `B`
matrices runs:

- **(a) `scan`:** `lax.scan` over the backend's single-matrix call, compiled
  once per (op, backend, mesh, signature).
- **(b) `backend_batched`:** the library's own stacked entry (ScaLAPACK eigh,
  cuSOLVERMp Cholesky and LU, ScaLAPACK LU, `native2d`, `jnp.linalg.eigh`),
  one descriptor and one workspace for the whole stack.
- **(c) `batch_reshard`:** move the batch axis onto the mesh, run the native
  JAX kernel on whole local matrices, move the outputs back. No
  distributed-library call.

The caller selects `batched_route ∈ {'batch_reshard', 'auto'}`;
`'batch_reshard'` is the default for `plan`, `dispatch_batched_eigh` and
`matmul`. `'auto'` takes (b) when the backend has a stacked entry and (a)
otherwise. Route (c) is the default because for every matrix that fits one
device a distributed library's cost is its fixed per-call charge, which the
local kernel does not pay ([Backends § performance](backends.md#distributed-is-a-capacity-route)).
Route and backend are orthogonal: an explicit backend is still resolved and
probed before (c) runs; `backend='off'` is the provider-free spelling.

**Route (c)'s exchanges.** The forward moves and their literal inverse run in
one `shard_map`, two single-axis `all_to_all`s each way, because the one-step
face-to-batch move is not a tile permutation and GSPMD would lower it as
replicate-then-partition:

| step | collective (`tiled=True`) | local shape |
|---|---|---|
| input face | — | `(B_p, n/P_x, n/P_y)` |
| forward x | `all_to_all('x', split_axis=0, concat_axis=1)` | `(B_p/P_x, n, n/P_y)` |
| forward y | `all_to_all('y', split_axis=0, concat_axis=2)` | `(B_p/P, n, n)` |
| inverse y | `all_to_all('y', split_axis=2, concat_axis=0)` | `(B_p/P_x, n, n/P_y)` |
| inverse x | `all_to_all('x', split_axis=1, concat_axis=0)` | `(B_p, n/P_x, n/P_y)` |

The inverse runs y then x; x then y is shape-correct on a square mesh and
scrambles the data. Eigenvalues come back replicated through one device
`all_gather`; eigenvectors, factors and solutions return at `P(None,'x','y')`.
Nothing crosses the host.

- **Ragged batches** are zero-padded before the first exchange. Pad slots
  never enter a Cholesky or LU (a scalar `lax.cond` on the global index) and
  are dropped after the inverse. Eigh solves each rank's local stack in one
  batched call; a pad slot is an exact zero matrix and returns zeros.
- **Capacity is the boundary.** Each device holds `⌈B/P⌉` whole `n × n`
  matrices, their outputs and the local solver workspace. When one matrix
  does not fit one device, use `batched_route='auto'` with a distributed
  backend.
- **The eigh-stack API** {#eigh-stack}. A caller that holds a face stack
  `(B, n, n)` passes its room per rank beside its own live set,
  `plan('eigh', mesh, n=n, backend='distributed', budget_bytes=room)`, and
  the service decides each stack (`Plan.stack_route`, a `StackRoute`):
  - route (c) when the compiled program that runs the stack fits the room:
    its slices' exchanges, local eighs, inverse exchanges and result check
    (outputs and temporaries; the caller's operand is already in its live
    set) plus cuSOLVER's reported workspace per whole matrix. The most whole
    matrices per rank that fit win, and the stack runs in that many slices,
    one `lax.scan` over slice starts writing the face outputs in place;
  - the whole-mesh provider otherwise, including room 0.
  The decision is compiled, never a formula. The room is a caller value every
  rank shares, and the choice is agreed over ranks through the runtime's KV
  store (the fewest whole matrices per rank any rank chose), so every rank
  runs the same route, rounds and collectives. It is printed once per
  (op, B, n, dtype, room) by `new_stack_routes()`, which a driver prints
  through its reporter, and listed by `describe()`. Per whole matrix at
  complex128 a rank needs 7 n² × 16 B: 3 n² compiled (input, vectors, one
  copy) and 4 n² of cuSOLVER syevd workspace (one A100, n = 3328–18304),
  so n = 9152 needs 9.4 GB and n = 18304 needs 37.6 GB.
  There are no sub-meshes or rank groups: whole mesh or whole matrices per
  rank. `_route=` on `batched` is the test-only override.
- **Batch-layout input.** An eigh stack already at `P(('x','y'),None,None)`
  is solved rank-locally with no movement.
- **CPU collectives.** Route (c) calls `warm_mesh_cliques(mesh)` first, so on
  a multi-process CPU run with MPI collectives every communicator is created
  from the main thread ([collective transports](../../environment/transports.md)).
- **Keywords.** `block_size` and `compute_evecs` are dropped at the route
  boundary; any other keyword raises `TypeError`.

`dispatch_batched_eigh(A, mesh, backend='distributed', *,
batched_route='batch_reshard')` is `plan('eigh', …).batched(A)` for
`gw.qsgw_density`.

### Resident operands: `batch_layout` and `local_batch`

A factor used against many right-hand sides is placed in the batch layout
once, and only the right-hand sides move per call:

```python
F_b = dla.batch_layout(F_face, mesh)                       # once: (B_p, n, n), q-local
run = dla.local_batch(lambda F, Z: F @ Z, mesh, resident=(0,))
X = run(F_b, Z_face)                                       # per call: Z moves, F does not
```

`batch_layout` takes a face stack (moved by the route (c) exchanges) or a
replicated array (sliced locally) and returns `(B_p, …)` at
`P(('x','y'),None,…)`. Rank `x·P_y + y` owns rows `[r·B_p/P, (r+1)·B_p/P)`,
the order the exchanges produce, so a resident row meets the right-hand side
of the same global index. Resident operands may have any trailing shape (a
`(B_p, n)` pivot table, for example); a `resident` position whose operand is
not in batch layout refuses before tracing. Per call a right-hand side costs
`2·⌈B/P⌉` whole blocks of exchange; `out_layout='batch'` leaves the outputs in
the batch layout and halves that, for a consumer that reads whole matrices per
rank. `is_batch_layout` tests the layout.

## Matrix products

Three entry points cover GEMM, chosen by how often the shape repeats and where
the call sits:

| entry | for | phases |
|---|---|---|
| `matmul` | an eager product of any shape | resolves and probes on every call |
| `gemm_plan` / `local_gemm_plan` | one fixed shape called many times inside a `jit` or `lax.scan` (Green build, Σ projection) | eager plan, trace-safe call |
| `panel_matmul`, `batch_gram`, `contract_faces` | face products whose contraction axis must never be gathered whole on a rank | trace-safe, no provider |

### `matmul`

`matmul(A, B, C=None, *, mesh, alpha=1, beta=0, transa='N', transb='N',
backend='auto', batched_route='batch_reshard', budget_bytes=None)` computes
$D = \alpha\,\mathrm{op}(A)\,\mathrm{op}(B) + \beta C$ with
$\mathrm{op} \in \{N, T, C\}$ (none, transpose, adjoint). Rank-2 operands are
at `P('x','y')`, rank-3 stacks at `P(None,'x','y')` with one shared leading
batch; the output keeps the input rank. `C` may be omitted only when
`beta = 0`; real operands refuse complex `alpha` or `beta`. Every input face
and the output face must tile the mesh.

- **Staged route** (`batched_route='batch_reshard'`, the default): the route
  (c) exchanges for A, B (and C when `beta ≠ 0`), a local `jnp.matmul`, and
  the inverse exchanges for D. A rank-2 call is lifted to batch 1 and padded
  to `P`. Each device holds `⌈B/P⌉` whole A, B and D matrices plus exchange
  buffers, so use it only when those fit.
- **Provider route** (`batched_route='auto'`): the distributed GEMM of the
  platform ([Backends § GEMM](backends.md#gemm-providers-and-active-range-kernels)).
  With a shared `budget_bytes`, the staged route runs when one rank's whole
  matrices fit and the provider otherwise.

### `gemm_plan`

`gemm_plan(mesh, *, m, k, n, nq, dtype, backend='auto', alpha=1, beta=0,
layout='face', reduction_axis=None, out_spec=None, enable_active_range=False,
warmup=True) -> GemmPlan` fixes one N,N shape, $D_q = \alpha A_q B_q + \beta
C_q$ for `q < nq`, with `A (nq, m, k)`, `B (nq, k, n)`, `C`, `D (nq, m, n)`.
`nq` holds k-points; a spinor axis is flattened into `m`, `k` or `n`, or the
plan is called once per spin. `GemmPlan(A, B, C=None, *, out=None)` is
trace-safe: no library load, no probe and no new `jit` inside the caller's
trace.

- **N,N only.** There is no transpose argument; a caller with a transposed
  operand stores it once in the complementary face layout.
- **`alpha`, `beta` are compile-time** FFI attributes. With `beta = 0` the plan
  also compiles a variant that creates the zero addend inside the same
  program; `out=` donates an existing buffer instead (legal only with
  `beta = 0`, since a `beta ≠ 0` plan would scale stale content). Pass `C` or
  `out`, not both.
- **`warmup=True`** compiles and runs the plan once on dummy operands, so the
  first real call inside a timed loop pays nothing; `warmup=False` keeps the
  eager validation and probe but defers compilation to the first call, for a
  one-shot outer `jit` that compiles the GEMM with its neighbours.
- **`GemmPlan.local_call`** runs the same GEMM from inside the caller's own
  `shard_map` over the plan's mesh, on bare local tiles.
- **`backend='off'` refuses**, because the staged route would materialize
  whole operands on every device, which a planned GEMM exists to avoid.

| `layout` | operands | contraction |
|---|---|---|
| `face` | all at `P(None,'x','y')`; `m % P_x`, `k % P_x`, `k % P_y`, `n % P_y` | the platform's distributed GEMM ([Backends](backends.md#gemm-providers-and-active-range-kernels)) |
| `axis` | `A (q, m_X, k)`, `B (q, k, n_Y)`, `D (q, m_X, n_Y)`; complete local `k` | a local product, no collective (`local_gemm_plan`) |

For `layout='axis'`, `out_spec=P(None,'x',None)` or `P(None,None,'y')` keeps
one centroid axis sharded, and `reduction_axis='x'` or `'y'` contracts a tiled
axis with a local GEMM and a `psum_scatter` (two-axis output only).

### Active ranges {#active-ranges}

A Green's function $G_k = \sum_n \psi_{nk} w_n \psi_{nk}^\dagger$ restricted to
an energy window contracts only a band interval $[lo, hi)$, but its operands
keep their full band extent `K` so the compiled shapes never change. With
`enable_active_range=True`,

```python
D = plan.active_range(A, B, lo, hi, C=None, *, out=None, weights=None)
```

computes $D_q = \alpha\,(A_q \,\mathrm{diag}(w_q))_{:,lo:hi}\, B_{q,lo:hi,:} +
\beta C_q$ without allocating anything of the interval's size. `lo`, `hi` are
integer scalars or `(nq,)` arrays, replicated; `weights` is `(nq, K)` and needs
a complex plan when complex. `0 ≤ lo ≤ hi ≤ K` is required: invalid Python
integers raise eagerly, invalid traced bounds raise in the distributed
provider, and the local kernels return an all-NaN output rather than clamp. An
empty interval applies the `beta` semantics without reading A or B; a full
interval calls the dense GEMM, so its evaluation order is unchanged. Interior
holes in an interval still cost arithmetic.

When the interval is known before tracing, `prepared =
plan.prepare_active_range(lo, hi)` validates and captures the bounds as
compile-time metadata, so no bounds operand exists and no device-to-host copy
or stream synchronization happens per call. The caller owns each prepared
callable; each distinct interval compiles its own executable on first use, so
keep one per recurring interval and use `active_range` for intervals that
change. Preparation probes the provider eagerly even for a full interval, so a
missing handler cannot hide behind the dense path. Local active plans need
replicated `K` and a two-axis output; reduction-axis and single-axis-output
plans refuse the option. How each route implements the interval is in
[Backends](backends.md#gemm-providers-and-active-range-kernels).

### Bounded face products: `panel_matmul` {#bounded-face-products}

The Green build multiplies face matrices whose contraction axis is the band
index, and the layout rule forbids any rank to hold a band-complete row or
column panel. `panel_matmul(A, B, *, mesh, panel_bytes, bounds=None,
weights=None, partner=False)` forms $A B$ as a batched 2-D SUMMA inside one
`shard_map`:

- `A` is `(q, m, k)` at `P(None,'x','y')`; `B` is `(q, k, n)` in the same
  layout, or `(q, s, k, n)` at `P(None,None,'x','y')`, a sample axis over
  which each A panel is broadcast once.
- **Panels.** On a square `p × p` mesh each panel takes `w` local columns of
  every owner block: one `all_gather` per operand carries `p·w ≤ k/p`
  columns, with every batch row in the same exchange and the same local GEMM.
  The next panel is gathered before the current one is multiplied, so two
  panels of at most one owner block are live. `w` is the even split of `k/p`
  into the fewest panels with
  `itemsize·q·2·p·w·(m/P_x + n/P_y) ≤ panel_bytes`, at most `k/p²` local
  columns each; a remainder ends in one narrower panel.
- **`bounds`** `(q, 2)`, replicated, names each row's live contraction
  interval (the caller has zeroed the rest); each panel's local product runs
  over that interval's columns only, so dead bands cost no flops.
- **Accumulation.** With `bounds`, or with three or more panels, every panel
  after the first adds into the output tile in place (on CUDA the local
  `beta = 1` GEMM, without bounds through its prepared target over every
  column). XLA folds one `c + a @ b` into its GEMM but leaves one of two
  adjacent ones as a separate add that holds two extra output tiles; a
  two-panel product without bounds stays on XLA.
- **`weights`** `(q, k)` scale each panel slice of `A` on its way into the
  gather, so no weighted copy of `A` exists. **`partner=True`** also returns
  $\bar A\,\mathrm{diag}(w)\,\bar B$ from the same exchange (the Green's
  function and its antiunitary partner), each gathered panel conjugated before
  its own GEMM.
- Rectangular meshes and the sample axis stream owner panels by a masked
  `psum` and contract every column.
- `panel_bytes` bounds the two live operand panels only; the caller admits
  the input and output faces, compiled temporaries and provider workspace.

Measured cost (A100-40GB, warm, ms per Green-sized complex product; CrI3 8×8
`q=10, m=n=2904, k=144`; Fe 8³ `q=59, m=n=2560, k=120`; valence windows):

| route | P4 CrI3 | P4 Fe 8³ | P16 CrI3 | P16 Fe 8³ |
|---|---|---|---|---|
| full-k gather (band-complete on every rank; forbidden by the layout rule) | 2.68 | 6.59 | 1.86 | 5.14 |
| batched SUMMA, panels ≤ k/p (this route) | 3.22 | 6.79 | 2.57 | 7.37 |
| two k/2 panels (every band live on a rank; forbidden) | 3.22 | 6.79 | 1.85 | 5.95 |
| cuBLASMp SUMMA, one call per q | 6.68 | 10.95 | 6.88 | 5.11 |

cuBLASMp runs one SUMMA per q and joins the XLA stream by events at entry and
exit, so it cannot overlap neighbouring work; this route exchanges every q in
one collective per panel. Under XLA's default scheduler the prefetched gather
still runs on the compute stream, so it does not overlap the GEMM; at P16 the
exchange is most of a build (1.2 of 1.85 ms on CrI3).

**`batch_gram(b, weights, bounds, *, mesh, nbatch, right=None,
partner=False)`** forms $W_q = b_q\,\mathrm{diag}(w_q)\,c_q^\dagger$ over each
row's interval when the factors `b (B_p, m, K)` and `c` already sit in the
batch layout with whole rows per rank: each rank contracts its own rows with
the same local active-range GEMM, and only `W` moves, batch to face.

**`contract_faces(b_X, b_Y, weights, start, stop, *, mesh,
return_transpose=False)`** forms $(b_X\,\mathrm{diag}(w))\,b_Y^\dagger$ for
row faces `(b, m, K)` at `P(None,'x',None)` and `P(None,'y',None)` (or
`(b, μ, s, K)`, spin merged into μ) with replicated weights `(b, K)` and
interval `[start, stop)`. The output is `(b, m, m)` at `P(None,'x','y')`;
every product is local, with no collective and no provider.

## The polar factor and spectral directions

The parallel-transport links of the dipole step need the unitary closest to an
overlap matrix $A$, $L = U V^\dagger$ for $A = U \Sigma V^\dagger$.
`polar_factor(A, mesh, *, backend='distributed', rcond=None) -> (L, s)` and
`plan_polar_factor(mesh, *, n, backend='distributed', rcond=None,
batched_route='auto') -> PolarPlan` diagonalize the Hermitian dilation

$$
H = \begin{pmatrix} 0 & A \\ A^\dagger & 0 \end{pmatrix}, \qquad
H \begin{pmatrix} u_i \\ \pm v_i \end{pmatrix}
  = \pm\sigma_i \begin{pmatrix} u_i \\ \pm v_i \end{pmatrix},
$$

with one planned eigensolve of extent $2n$, read $(u_i, v_i)$ as $\sqrt2$ times
the halves of the $n$ positive-eigenvalue eigenvectors, and form
$L = \sum_{\sigma_i > \mathrm{rcond}\,\sigma_{\max}} u_i v_i^\dagger$. It never
forms $A^\dagger A$, which would square the condition number and lose the small
singular values the link quality check reads.

- **Input:** one square `float64`/`complex128` matrix at `P('x','y')` with `n`
  divisible by both mesh axes; anything else refuses, with no implicit
  reshard. For a non-divisible extent, zero-pad to the next common multiple
  and slice the leading block of `L`; the pad's null directions fall below the
  cutoff, and `plan_polar_factor` reports the minimum pad.
- **Output:** `L` with A's shape and sharding; `s`, length `n`, real,
  descending, replicated. `rcond` is relative to $\sigma_{\max}$ (`None`:
  $n\varepsilon$), so a rank-deficient A returns its polar partial isometry.
- **Cost:** one $2n$ eigensolve and one $n^3$ GEMM, $O(n^2/P)$ memory per
  process (the dilation holds $4n^2$ elements). Hoist `plan_polar_factor` out
  of k-point loops; `polar_factor` caches plans for eager calls and refuses a
  tracer.
- **Stacks:** `PolarPlan.batched(A)` takes `(B, n, n)` at `P(None,'x','y')`.
  With `batched_route='batch_reshard'` each rank solves its `⌈B/P⌉` whole
  matrices locally; with `'auto'` each matrix takes the distributed solve. The
  dipole step picks the route from the deck's `linalg` dial (`local` →
  batch route, `distributed` → mesh solve), never from free memory.
- Compare `L` and `s` across meshes, never dilation eigenvectors, which are
  gauge-dependent.

`right_singular_vectors(W, tau, *, eigh_plan, column_extent, multiplet_tol=1e-6,
real_rows=None, max_rank=None)` returns the right singular directions with
$\sigma/\sigma_{\max} > \tau$; `leading_eigenvectors(W, r, *, eigh_plan,
column_extent, …)` returns the leading `r` eigenvectors (optionally bounded by
`rcond·max|λ|`); `retain_leading_eigenvectors(Q, values, r, *, mesh, column_extent, …)`
narrows a `leading_eigenvectors` result to a smaller width with no new
eigensolve. All three close a whole multiplet at the
cut (relative gap ≤ `multiplet_tol`), so a degenerate subspace is never split.
`W` is a square face, a face stack or a batch-layout stack; `eigh_plan` is the
caller's resolved plan of extent `m`, and its route decides local or
distributed. Singular directions are eigenvectors of the normal matrix
$W^\dagger W$ with $\sigma = \sqrt{\max(\lambda, 0)}$, one eighth of the
dilation's eigensolver flops; the cut is $\tau^2$ on $\lambda$, so
`right_singular_vectors` refuses $\tau^2 < 10^4 m\varepsilon$. On a cuSOLVERMp
plan the normal matrix $G$ is solved as $I + G/\lVert G\rVert_F$ and the
eigenvalues mapped back, because cuSOLVERMp's divide-and-conquer can fail on
the exact-zero eigenvalue cluster of $G$; this moves $\sigma$ at the cut by about
$\varepsilon\sqrt r/\tau^2$ ($3\cdot10^{-8}$ at $r = 2\cdot10^4$,
$\tau = 10^{-3}$), below the multiplet tolerance. Only the $O(m)$ spectra cross
the host, broadcast from one process as bit patterns so every rank makes the
same cut.

## Face-pinned block glue

Eager slicing, concatenation and $a + a^\dagger$ of face-sharded arrays come
out replicated, so a `(b, R, R)` block would cost $16R^2$ bytes per rank
instead of $16R^2/P$. These helpers run the same elementwise program with the
operand's face as the output sharding (one executable per function, layout and
statics), bitwise equal to the eager form:

| call | result |
|---|---|
| `hermitian_part(a)` | $(a + a^\dagger)/2$ for `(b, R, R)` |
| `hermitian_block(block, off, corner)` | $\begin{pmatrix} \text{block} & \text{off}^\dagger \\ \text{off} & \text{corner} \end{pmatrix}$, `(b, R+r, R+r)` |
| `join_columns(a, b)` | column panels `(b, n, R)` and `(b, n, r)` concatenated |
| `diagonal_like(values, like)` | $\mathrm{diag}(\text{values})$ in `like`'s dtype and face |
| `on_face(fn, out, *operands, **static)` | a module-level `fn` with outputs placed on `out` |

`face_sharding(array)` returns the `NamedSharding` of a concrete array sharded on every axis, else `None`. Traced operands and
unsharded host arrays take the plain function; no helper gathers, pads or
reshards.

## Workspace queries {#workspace}

The native libraries allocate workspace outside XLA's memory planner, so a
caller sizing a stage asks the service first. All three queries allocate
nothing and accept `float64` or `complex128` only.

- `workspace_bytes_per_rank(plan, op, shapes, dtype)`: the device workspace
  per rank of a resolved CUDA `Plan` or `GemmPlan`. For a distributed `eigh`
  it is the cuSOLVERMp workspace rounded to 256 bytes plus one private operand
  tile `(n/P_x)(n/P_y)·itemsize`, which keeps eigh non-donating despite the
  destructive tridiagonalization; do not add that tile again. A local eigh
  returns the cuSolverDn workspace plus a 4-byte info word. For `gemm`, one
  vendor workspace per context persists across calls, so budget
  `max(GEMM workspace) + max(concurrent eigh scratch)`. Query collectively on
  the real mesh; non-CUDA meshes refuse.
- `matmul_workspace_bytes_per_rank(mesh, shapes, dtype, *, backend,
  batched_route)`: the same query for a `matmul` route.
- `fits_local(plan, op, shapes, dtype, budget_bytes)`: whether
  $\sum \mathrm{prod(shape)}\cdot\mathrm{itemsize} + \text{workspace} \le$
  `budget_bytes` for the caller's whole per-rank set of complete matrices. The
  workspace is the cuSolverDn query on CUDA, the LAPACK `?heevd` optimum on a
  host (complex: `lwork = 2n + n²`, `lrwork = 1 + 5n + 2n²`,
  `liwork = 3 + 5n`), or the compiled local GEMM temporary. The budget must be
  the same on every rank, because the answer selects collectives.

Operand carriers, transpose staging, communication buffers and persistent
context resources are the caller's to admit.

## Round-off tolerance {#roundoff}

`roundoff_tol(n, dtype=complex128)` is `ROUNDOFF_MARGIN · n · eps(dtype)` with
`ROUNDOFF_MARGIN = 64`. It is the relative bar of every Hermiticity check: the
checked local eigh, `leading_eigenvectors` on a distributed solver and its
spectrum diagnostic, the bare response-moment gate
(`gw.response_bank._check_bare_hermitian`) and the dense DFT Hamiltonian
(`psp.run_dense_h`). Each check compares `max|A − A^H| / max|A|` against it, with
`n` the matrix side when the inner length of the products is not known. A
fixed `1e-12` is below the round-off of a GEMM-built Hermitian once `n·eps`
passes it (n ≈ 4500 in complex128). A refusal therefore names a wrong input,
such as a missing conjugate or an unsymmetrized product, and never round-off.

## Vocabulary and introspection

| name | purpose |
|---|---|
| `BACKEND_CHOICES`, `EIGH_BACKENDS`, `CHOLESKY_BACKENDS`, `LU_BACKENDS`, `OPS`, `NATIVE`, `MATMUL_BACKEND_CHOICES`, `BATCHED_ROUTES`, `BATCHED_ROUTE_CHOICES`, `BATCHED_ROUTE_DEFAULT` | the vocabularies, importable with no native library so a deck parser needs no FFI |
| `resolve_backend(op, requested, mesh, *, n=None, compute_evecs=True)` | the raising probe |
| `list_backends(op, mesh)` | the never-raising report, for startup banners |
| `resolve_matmul_backend(requested, mesh, *, batched_route)` | the GEMM probe |
| `mesh_key(mesh)`, `mesh_platform(mesh)`, `mesh_is_cpu(mesh)` | a hashable mesh identity (axes, extents, platform, device ids) and its predicates; key any cache whose value does not retain the mesh on `mesh_key`, because two same-shaped meshes over different devices lower to different handlers |
| `dial_key()`, `probe_target`, `has_target`, `backend_module` | the factory cache key and the capability probes (absent vs broken) |

## Rules for callers

- **No `jnp.linalg.svd` or `eigh(A.H @ A)` at a consumer.** Use
  `polar_factor`, or hoist `plan_polar_factor` for a streamed loop.
- **No Python loop over the batch axis.** Use `Plan.batched`: a loop hands
  the compiler `B` separate calls, and with SLATE recompiles its eager
  `shard_map` wrappers per matrix.
- **No backend from the environment.** The `linalg` deck dial chooses; the
  environment only says which library exists.
- **No eigenvector comparison across meshes.** Degenerate subspaces have no
  canonical basis; compare eigenvalues or gauge-invariant contractions
  ($Z\,\mathrm{diag}(w)\,Z^\dagger$, projectors, $|Z^\dagger Z'|$).
- **No branching on `plan.backend` or token internals**, and no `try/except`
  around a refusal: an explicit backend that cannot be honoured must stop the
  driver.
- **No hand-built mesh identities** (`id(mesh)` is safe only when the cached
  value retains the mesh) and no `sys.path` edits: an installed consumer
  imports `distrib_la` directly.
