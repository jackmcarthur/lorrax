# Memory model

A LORRAX calculation needs its per-rank live set to fit available device
memory and its processes' combined host memory to fit their node. The budget
is a planning target with the warn-and-run policy below. The leading
device faces are tiled over the `x × y` mesh; their leading spread terms fall
as `1/P` at fixed legal carriers. Padding, minimum tiles and replicated data
set floors, and band-complete ψ copies scale as `1/sqrt(P)`. This page states, stage by stage,
what is resident per rank, how it scales, and which planner prices it.
Stages share the pricing, admission and measurement API `lxkit.memory`
(`services/lxkit/src/lxkit/memory.py`). A quote can combine compiled-program
statistics (`memory_analysis`), provider size queries, measured vendor rows
and shape-based proposals; its coverage matters
([§ Prices from measurement](#measured), [known gaps](#pricing-coverage)). No planner refuses on a price (owner
2026-10-19, `memory.admit`): the fastest route that fits runs; when none fits,
the cheapest runs with one `RuntimeWarning`, or, when it is over the allocator's
physical limit, with a loud banner naming the shortfall and an estimated
minimum process count from that quote. Separate gates can refuse an unavailable
workspace query, a compiled module quoted above the device
([`GATE xla_rematerialization`](#module-does-not-fit)), a measured vendor row used
outside its range under `memory.STRICT` (`GATE memory_unpriced`), and the
correctness gates.

| symbol | meaning |
|---|---|
| `N_k` | full-zone k points (= full-zone q points) |
| `Q` | q rows the ζ fit keeps (the IBZ wedge when the grid is reduced) |
| `n_s` | spinor components (1, 2; 4 for the bispinor fit) |
| `μ` | centroid count, mesh-padded |
| `N_b` | bands in the fit window |
| `N_G`, `N_Gψ` | ζ sphere and ψ sphere sizes (`ngkmax`) |
| `N_r` | FFT grid points |
| `P = p_x·p_y` | ranks on the square mesh |

All arrays are complex128 (16 B) unless stated.

## Budget

Every planner prices its stage against one number per device, the budget `B`
(`common.gpu_utils.device_budget_bytes`, held by `lxkit.memory`). `B` comes from
the allocator's limit `L`, the bytes outside it, and the leg's reserve; every
platform difference is a row of `lxkit.LEGS`, not code.

### The pool {#pool}

On CUDA, XLA takes every device buffer of every program from the CUDA
driver's default memory pool, through the stream-ordered allocator
(`cuda_async`): the arrays a program returns, and one buffer per executable
that holds all its temporaries. FFI handlers that call `cudaMallocAsync`
draw from the same pool. The variables that select this allocator, and what
jaxlib builds from them, are in
[environment § GPU pool](../environment/overview.md#gpu-pool). Two counts
describe the pool at a time `t`:

```text
U(t)   bytes in use: allocated and not yet freed        (CU_MEMPOOL_ATTR_USED_MEM_CURRENT)
V(t)   bytes reserved: device memory the pool maps      (CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)
```

`V − U` is free space inside the pool. At start-up XLA allocates and frees
one block of `R` bytes, so `V = R` before the first program runs. It also sets
the pool's release threshold to `R`: at a stream synchronize the driver
unmaps free pool memory only above `R`. Everything else the process holds
is outside the pool, `O(t)`: the CUDA context and modules, the NCCL
communicators, cuFFT plans, and the cuSOLVERMp contexts and their workspace.
The card's free bytes are `F(t) = M − O(t) − V(t)`.

### The rule {#rule}

```text
L = bytes_limit                    where the allocator enforces it (BFC: ROCm)
L = M − max(O_now, O_leg)          otherwise (cuda_async), M the card total
B = (L − O_pend)·(1 − r) − T_ord   lxkit.memory.resolve, the minimum over processes
R = M − O_leg − O_dist             the pool's reservation (memory.pool_reservation; runtime.pool_fraction = R/M)
```

`O_leg` is the leg's `outside_allowance` (CUDA 3.6 GB, ROCm 4.92 GB: the measured
outside bytes, 3.0 and 4.1 GB, with the 1.2 margin), `O_dist` the distributed
providers' contexts (`lxkit.legs.DISTRIBUTED_CONTEXT_BYTES`, 3.6 GB), `O_pend` =
`O_dist` under `linalg = distributed` where the allocator does not enforce its
limit (the contexts open later, outside it), `r` the leg's `memory_reserve`, and
`T_ord` the rank-order reductions' fixed transient, `lxkit.ORDERED_TRANSIENT_SLABS`
(2) slabs of `LORRAX_ORDERED_SLAB_MB` (8 MiB) = 16 MiB
([rank-order reductions](../environment/transports.md#rank-order-reductions)).
**`r` = 0.18567 is provisional**: the 2026-10-08 rule's overshoot φ = 0.19 with its
1.2 margin, `1.2φ/(1 + 1.2φ)`, so `B` equals that rule's budget (A100-80GB: 66.4 GB
at linalg local, 63.4 GB distributed; MI250X GCD: 48.99 GB of a 60.18 GB arena)
until the gate legs measure `r` as the largest `(allocator peak − priced)` over
passing stages ([§ what φ covers](#what-phi-covers) is its present evidence).

A positive `memory_per_device_gb` can only lower `B` (above it, one warning and
the run keeps `B`). On CPU `L` is the node's `MemTotal` over the processes on the
node and the process's devices. The room beside what is allocated now is
`memory.room() = B − bytes_in_use` (the allocator's own figure; CPU: live arrays),
the minimum over processes, floored to a whole GiB.
On CPU the live count is the per-device union of local storage intervals:
global/local aliases and overlapping views spend the budget once. Reading
the count neither synchronizes pending kernels nor retains local arrays.

At eager stage boundaries, `distrib_la.ensure_sharding` preserves a concrete
operand already in the required layout: an eager `with_sharding_constraint`
can copy the entire operand even when its sharding matches. Inside a `jit`,
the helper retains the constraint; a genuinely different layout still reshards.
The PPM wrapper likewise adopts matching pole frequencies, even/odd residues,
validity masks and Wc(0) directly from the producer on its existing face.

CPU compiler heaps are separate from that live-array count. A cold P121
1 MiB all-to-all compile left 30.39 GiB of node PSS, although glibc had
already freed most of the compiler scratch. Releasing those free pages
with input and executable live reduced it to 14.32 GiB, with exact output
and unchanged warmed collective latency. The shared real-compile boundary
(`common.jax_compile_cache`) now reclaims free glibc pages only on CPU, when
both RSS growth and free heap reach the existing malloc trim threshold
(128 MiB by default). Its RSS watermark spans smaller compiles and follows
natural deallocation. The existing `LORRAX_MALLOC_TUNE` opt-out and threshold
win; GPU compiles, persistent-cache hits, and tiny follow-up compiles do not
trim. The integrated P121 probe used one trim per rank (15.8 ms median,
4.35 s cold compile); this reduces retained host memory after compilation,
not the compiler's transient peak (evidence:
`/pscratch/sd/j/jackm/cs_claude/htransform_followup_20261022/winterp_evidence/collective_p121_*`).

### Routes and admission {#routes}

`lxkit.memory.admit(stage, routes, room)` is the one admission point. `routes`
are `(name, Price)` pairs, fastest first; a `Price` has four per-rank terms,
`operands`, `xla`, `vendor` and `outside`, each `fixed + spread/P (+ grid(P))` with
its provenance. The fastest route that fits its room runs. When none fits, the
cheapest runs: within the physical limit (room + reserve) with one warning; over
it with a banner at plan time and again when the stage opens, giving the terms,
the shortfall and `memory.sizing()`'s `estimated minimum from this memory quote: P = N`,
the smallest square mesh at which the current modeled stage quotes fit. This
is a planning estimate, not a measured capacity limit: a quote may cover a
checked retry that the run does not execute, and its scaling model is not a
fresh compilation at every candidate mesh. If no square mesh up to P = 4096
fits the model, the report says so and names the quoted binding stage. A route that opens
a provider context under `linalg = local` carries it as its `outside` term
(`memory.contexts()`). Every decision prints once on rank 0 (`route <site>: <route>
(<prices>)`) and the run report's Routes table lists each with its count.

| site | routes |
|---|---|
| W Dyson (`gw.w_isdf._dyson_route`) | local: faces + $16\mu^2(3\lceil Q/P\rceil + 2)$; distributed: faces + one LU pair + contexts |
| response-bank Coulomb roots (`gw.response_bank._select_coulomb_program`) | complete compiled checked program for both roots plus native GEMM/eigh workspace and contexts; local first, existing distributed algebra when it does not fit |
| charge Cplus factor (`isdf.core._factor_c_q_face`) | complete compiled first and whole-chain programs plus matching native EIGH query and contexts; smaller local shares follow the service's doubling schedule before the whole-mesh program, each including its checked backup |
| SC eigh, density scan (`gw.sc_iteration`, `gw.qsgw_density`) | one $(n_b, n_b)$ tile within 1 % of `B`; cuSOLVERMp eigh / `band_2d` |
| tangential whole parents, constructor, staged eigh stacks (`gw.tangential_execution`) | the local route's compiled programs; the face / whole mesh + contexts |
| head Γ GEMM (`gw.tangential_head._gamma_gemm_route`), ζ face GEMMs | staged where its program compiled fits the room (`distrib_la.staged_fits`); `panel_matmul` faces |
| transverse ζ factor, Galerkin factor, coarse V(q) chunk | each route's own program compiled (`runtime.aot_memory.compiled_price`) |
| htransform node-pair stages: fine-q top block, path and active window, orbital operators, moment grid (`bandstructure.fh_interp.admit_node_route`) | `q-batched`: whole $(N_B, N_B)$ matrices of the rank's q rows after one face→q exchange; `face`: one q tiled over the mesh (`linalg = distributed` pins it); each route's program compiled, the chosen executable then run |
| nodal fit: quadrature metric, band/node Grams (`isdf.galerkin`) | orbit normal equations and every nonnegative principal solve on `P(x,y)`; point-pair scratch streamed in bounded blocks; band Gram, Loewdin and coefficients on band/rank faces through `distrib_la`; only linear geometry, orbit weights and eigenvalue vectors reach hosts |

Provider contexts are an accelerator allowance. The host leg charges zero
through the same `lxkit.memory.contexts()` owner; importing the compiled-price
helper does not initialize a CUDA provider on CPU.
LU principal solves and ring-lift closes ask `distrib_la.provider_contexts(plan)`
for the resolved provider's allowance. Only vendor MPI grids add that term;
`one_device` and `native2d` reuse the ordinary leg context. This run-dependent
allowance is refreshed outside the cached executable price.

The response bank prices the complete requested q stack of `V^(1/2)` and
its supported inverse before dispatch. One parent fitting the eigensolver
does not imply that both reconstructions and the stack's native workspace
fit. Admission uses `compiled_new_bytes` beside the already-live input,
including the existing checked retry, and the same provider queries used
by dispatch. The selected program carries its dense layout into that query;
its full checked EIGH is queried on the whole mesh even when the subsequent
root reconstructions use local GEMMs.
Both outputs remain `H_q(mu_X,nu_Y)` faces. The programs and executable
caches retain geometry and plans, never the Coulomb input.

The response bank and tangential screening owner share bare-V parent rows
and both Coulomb roots across their bank, moment and constructor calls.
Accepted roots are shared across local and distributed execution requests
when the authenticated operator, packed basis, numerical cut and mesh agree;
both return the same `H_q(mu_X,nu_Y)` face contract. A constructor's sub-span
slices the existing roots rather than retaining another pair per parent.
The numerical backend remains part of the compiled-program identity.
One-shot GW releases these cache references after W0 persistence and before
Sigma. QSGW retains reuse between maps and releases them after the final
W0 persistence. Clearing these holds does not delete arrays still owned by
callers or clear compiled programs.

The nodal fit keeps sampled states on centroid faces. Whole-orbit packing
allows each rank to accumulate its normal-equation tile without gathering
node samples or a whole orbit matrix. The shared nonnegative quadratic solver
uses the existing CUDA capacity route when its complete checked principal
program plus native GETRF scratch fits the current agreed room. A lazily
priced whole-mesh LU is its backup; other device legs retain their provider.
Only the selected executable runs, and inactive weights return exactly zero.
The one-system program cache keys its concrete provider, not the live room. Loewdin eigenvectors and inverse roots remain band faces; explicit
all-to-all and transpose-partner exchanges produce the compact product-rank
coefficient and target carriers. Basis artifacts continue to store physical
band/rank extents through SlabIO, which reads directly into mesh-padded faces.
The shared full-grid refit also keeps its coefficient chunks on node/state
faces, forms its node Gram and sampled overlaps through bounded SUMMA panels,
and exchanges the final rows directly into product-r slabs. Its retained-band
residual uses those same faces; it never recreates the complete coefficient
window on each rank.
The remaining replicated fit metadata grows linearly with node/orbit count or
with `N_k N_b` eigenvalues; there is no host `N_orbit²` or `N_k N_b²` matrix.

### Prices from measurement {#measured}

- **Programs.** At compiled-price sites, a route or dispatch is priced by the program that runs it,
  lowered and compiled at its abstract operands: `temp + outputs − alias` plus
  the largest dynamic FFT workspace (`runtime.aot_memory.compiled_new_bytes`):
  cuFFT/hipFFT plans or ROCm's split k-minor provider query, whichever is larger,
  taking the largest figure any rank read. These are new-buffer figures beside
  the live operands, not measurements of total process peak. Removing an unused
  output can also change temporary storage, so its byte size alone is not the
  net saving. Where the program then runs, the priced executable is the one
  executed (the response bank, the ζ tile passes, the tangential rounds), so a
  price costs no extra compile. `check_chunk` corrects a chunk once from that
  figure ([§ The compiled check](#the-compiled-check)).
- **Proposals, then the run's check.** Where the program that will run is not
  known when the stage is admitted (its cut and room are the call's), a closed form
  proposes: a local constructor's and the line selection's eigh stacks take
  `distrib_la.eigh_stack_proposal` (five matrices per rank's share of the batch, plus
  the solver workspace outside the program, `distrib_la.local_eigh_workspace`). The run then admits its own executable before it
  executes: an eager route-(c) attempt, first or retry (`Plan._checked_reshard`),
  checks its `memory_analysis` plus any separately priced solver workspace
  against the agreed room (`memory.room`,
  the minimum over ranks); over it, twice the rounds while a rank's share still
  shrinks, each compiled only when the one before is over, and when none fits the
  smallest runs under `admit`'s warning. Every rank runs the largest choice any rank made. A local reduction round is admitted
  at its compiled round program (`tangential_execution.price_round`). The receipt
  shows the proposal; the run's check prints when it is over.
- **Room classes.** A room enters a program or a price only through tests of
  compiled figures against it (`distrib_la.room_test`: `place()`'s fit and
  step, the gathered retry's admission, `staged_fits`, the size route).
  `distrib_la.room_class` caches a program or price per class of rooms whose
  tests come out alike, so a round that repeats a shape in another room reuses
  its compile; a hit replays its tests to any enclosing build.
- **Vendors.** Workspaces outside XLA come from the provider's size query
  (cuSOLVERMp eigh; the Dyson LU at the getrs cap it posts,
  `distrib_la.lu_workspace_query`) or a measured row (`lxkit.LEGS`
  `vendor_rows`) within its stated range; outside it, a warning (`GATE
  memory_unpriced` under `LORRAX_MEMORY_STRICT`).
  The response bank resolves its Dyson plan at the actual operand batch:
  stored parents or a streamed moment span, while the physical prefactor
  still uses the full q count. The agreed allocator room already includes
  materialized operands; planning subtracts only future abstract operands.
  The same resolved checked program and LU plan own compilation, native
  workspace queries and completion. Its getrs cap is withdrawn after either
  dispatch or a failed admission; charge moments have no LU to admit.
- **Route completeness.** No whole N_μ-scale matrix (side ≥ N_μ/2) is held on a
  rank without a mesh route beside it: a stage whose only candidates do so
  refuses (`GATE nmu2_no_backup`; `LORRAX_NMU2_BACKUP=report` makes it a
  finding for a census).

### Pricing coverage limits {#pricing-coverage}

ELPA's device pool and per-handle rocBLAS regions are explicit byte-buffer
results in the compiled eigensolve. They are counted once there, without a
second measured pool allowance. A stack uses local ROCm eigensolver scratch
only when its resolved route runs matrices locally; a leading batch dimension
alone does not select that scratch. The qualified native ROCm batched EIGH
target owns an exact queried arena in its executable. Legacy JAX scratch outside
its windows and the scalar complex128 Jacobi retry retain a separate measured
workspace price. ELPA's small retained allocations, context resources
and any fallback beyond its empirical pool remain outside the compiled arena.
A checked EIGH includes local gathered-retry scratch on CUDA and ROCm only when
that retry is admitted; sequential primary and retry scratch use their maximum.
The platform outside allowance and native allocation counters are separate from
the executable quote. Neither quote nor allowance is a general capacity proof.

`ZetaG.contract_v` records its actual compiled G-tile step or face chain,
including the prefetched and written tiles, beside already-live factors and
carries. The grouped live-V path (`contract_v_group`, including its face
variant) still uses the μ-batch proposal without a complete compiled
group-pass admission. Its individual factors and GEMMs can have compiled
prices; those do not price their combined live set or admit the initial V/S
allocations. The grouped estimate is not a complete capacity certificate.

Wide ROCm local LU/triangular applications use the blocked native handler when
one logical matrix's A or RHS span exceeds 2³¹ bytes. Its shape query and
execution share the same blocked rocSOLVER/rocBLAS call sequence. An explicit
byte-buffer result holds the queried vendor workspace and, for factor-and-solve,
the pivot vector. The actual executable's price therefore includes this scratch
beside its layout copies and outputs; callers do not add a separate estimate.
The single-device fast path retains its whole-matrix capacity limit. When
that path does not fit, the shared `native2d` LU keeps coefficient and RHS
faces distributed. Column panels have at most 128 columns; each pivot
exchanges two narrow panel rows, then the completed panel's permutation
exchanges at most 256 full row pieces once, including prior L and the RHS.
Its compiled program prices the factor copies and triangular solves. Input
consumption does not imply that A aliases an output: only the actual compiler
alias bytes are subtracted. This route removes the whole-matrix memory floor,
but pivot reductions still have communication latency; neither a fixed
four-matrix bound nor an unconditional speedup is promised.
The [matched 144-GCD qualification](../services/distrib_la/backends.md#size-route)
reports the solve's executable quote separately from timing and process peak.

The opt-in ROCm fused outer decode is also outside complete workspace coverage.
The default measured rule keeps the XLA decode because the fused version was
slower. With `LORRAX_KDFT_OUTER=all`, its phase-slot scratch is not included in
the compiled quote or a provider query ([kernel owner](kconv.md#bse-outer)).
An executable price under that override is not a complete capacity certificate.

### Why the pool's reserved bytes run past R {#pool-growth}

XLA's allocator issues `cuMemAllocFromPoolAsync` and `cuMemFreeAsync` on one
compute stream, so a free takes effect when the device reaches it in stream
order. JAX dispatches asynchronously: the host enqueues program `k+1`, and
with it `k+1`'s allocations, while the device still runs program `k`, so
`k`'s frees are still pending when `k+1` allocates. Without mapping memory,
the driver serves a request from

1. free space whose frees have completed, coalesced across adjacent blocks; or
2. a block, or adjacent blocks, whose frees are pending on the same stream.

It does not join a pending block to completed free space. A request larger
than every extent of either kind maps new memory, and `V` rises past `R`:

- if the card has room, `V` stays above `R` until the next synchronize trims
  the pool back to its threshold;
- if not, the attempt maps what is free, so `F` falls to zero, and then fails.
  XLA's allocator synchronizes its stream and retries once
  (`xla/stream_executor/gpu/gpu_cudamallocasync_allocator.cc`, `AllocateRaw`).
  By then the frees have completed and coalesced, and the retry is served
  inside `R`.

Either way the excess lasts until the next synchronize. A driver-level test
of the default pool on one A100-80GB shows each rule (reservation
`T` = 38.29 GB filled and freed once, as XLA does; claim 4090):

| case | frees before the request | request | `V` after | outcome |
|---|---|---|---|---|
| A | one 0.6·T block, completed | 0.5·T | T | served from the freed block |
| B | two adjacent 0.3·T blocks, pending | 0.58·T | T | served from the two pending blocks |
| C | one 0.6·T block pending, beside the never-used 0.4·T | 0.8·T | 1.4·T (53.62 GB) | the pool grows by 0.4·T |
| C, synchronized first | the same, completed | 0.8·T | T | served from the coalesced 1.0·T |
| D | C with 2 GB free on the card | 0.8·T | high-water T + 1.98 GB, then T | fails; served inside T after a synchronize and retry |
| E | 12 programs, each one kept 0.04·T output and one 0.5·T temporary, host ahead | — | high-water T + 1.54 GB | live set at most 0.62·T |
| E, synchronized each step | the same | — | T | no growth |

An A100-40GB gives the same cases at `T` = 19.08 GB on every rank (C grows
by 0.4·T; D reaches T + 1.98 GB, fails, and is served inside T on the retry).

The production traces show the same signature (sampler readings every
0.25–0.5 s; `CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH` catches what happens between
readings):

- CrI3 6×6 bispinor SC, P4, A100-40GB, budget 31.6 GB (claim 4091): at
  the end of the ζ stage, `U` = 24.99 GB in `V` = 35.20 GB. Before the next
  reading, the reserved high-water rose by 5.13 GB, which was all the free
  memory on the card (`F` = 5.14 GB). `U` fell to 1.92 GB as the next
  stage began, and `V` read 35.20 GB again at the next sample.
- The same deck on A100-80GB, budget 66.4 GB: the high-water rose by 5.10 GB,
  again all the card's free memory, with 44.55 GB in use and 33 GB free
  inside the pool.
- CrI3 24×24 bispinor, P64, A100-80GB, deck 72 GB (claim 4084): the reserved
  high-water reached 82.88 GB against 63.46 GB in use, with 4 MB free on the
  card, 7.1–7.25 GB past `R` on every rank.

So the reserved high-water minus the bytes in use is not a property of the
program. Whenever one such request occurs it reads "the card was full", at any
budget.

### What φ covers {#what-phi-covers}

The transient growth needs no room in the budget for XLA's own allocations:
the request that triggers it is served inside `R` after the synchronize, or
from the card's free memory until the next one. An allocation outside the
pool made while the pool is grown can find the card full. The driver gives
such an allocation idle pool memory on demand, but not a block whose free
has not completed ([environment § GPU pool](../environment/overview.md#gpu-pool)),
which is why a library workspace belongs inside the pool rather than in a
`cudaMalloc` of its own. Two things do need room in the budget, and `φ`
covers them:

- **in-use bytes above the budget**, `U_peak / B − 1`: peaks the planners
  under-price. Measured: Ni 20³ bispinor P64, A100-80GB, deck 72 GB:
  76.15 GB in the Σ τ sweep (+5.8 %, claim 4091); CrI3 24×24 bispinor
  P64, deck 72 GB: 69.33 GB (−3.7 %, claim 4084); CrI3 6×6 bispinor SC P4:
  25.21 GB of 31.6 GB on A100-40GB and 44.76 GB of 66.4 GB on A100-80GB
  (claim 4091).
- **free space between live buffers** that no request can use even after a
  synchronize. One case is measured. A CrI3 6×6 bispinor one-shot with 2634
  current points (P4, A100-80GB, deck budget 68.16 GB) was refused a 1.82 GB
  eigh scratch request while 4.65 GiB of its 70.53 GiB pool was free and
  18.5 MB was free on the card (claim 3315): 6.6 % of the pool.

With φ = 0.19 the rule leaves `1.2·φ·B` = 7.2 GB above the budget on A100-40GB
and 15.1 GB on A100-80GB for these two terms.

### The allocator {#allocator}

`cuda_async` with its pool reserved is the one allocator configuration on
CUDA ([environment § GPU pool](../environment/overview.md#gpu-pool)); the
allocator is the device leg's declaration (`PlatformLeg.xla_allocator`), and
the ROCm leg declares BFC (`cuda_async` does not exist there), preallocated at
`R` like the CUDA pool: 60.18 of 68.70 GB (61.50 GB before the leg's O of 4.1 GB) on an MI250X GCD instead of jaxlib's
75 % (51.53 GB), so the planner budget (41.96 -> 50.08 GB on hsuite) fits in
the arena. Outside it, after RCCL start-up, 512 MiB collectives, a 3D FFT and a
GEMM, each P4 rank held 0.99-1.08 GB of the 7.20 GB left (hipMemGetInfo minus
the arena's `pool_bytes`, `runtime.xla_memory.outside_pool_bytes`). With `HSA_NO_SCRATCH_RECLAIM=1` (the leg's runtime env) ROCr keeps each
queue's largest scratch allocation: the bytes outside the arena, sampled
every 0.25 s through whole runs, peaked at 4.07 GB per P4 rank in hsuite kmeans
(2.71 GB with reclaim) and 3.30 GB on Fe V2 (2.13 GB), so the leg declares
`outside_pool_bytes` = 4.1 GB. The hipFFT
plan workspace is inside the arena and priced like cuFFT's
(`PlatformLeg.fft_plan_scratch`, `runtime.aot_memory`). The
alternatives jaxlib offers were measured on CrI3 6×6 bispinor SC maps 0–2
at P4, cold, at the rule's budget (claim 4092). "Peak in use" is
XLA's `peak_bytes_in_use`; "high-water" is the pool's reserved high-water
under `cuda_async` and the arena's under BFC.

| card | allocator | peak in use | high-water | W per map (s) | run (s) |
|---|---|---|---|---|---|
| A100-40GB | `cuda_async`, reserved at `R` = 35.20 GB | 25.21 GB | 40.40 GB, the card full for an instant | 199.3 / 112.3 / 63.2 | 670.3 |
| A100-40GB | BFC, preallocated at `R` | 25.22 GB | 35.20 GB, 5.12 GB always free | 200.3 / 112.5 / 63.2 | 670.8 |
| A100-40GB | `platform` (`cudaMalloc` per buffer, no pool) | not reported | none; the card's own high-water 27.29 GB | 209.7 / 122.6 / 72.8 | 712.9 |
| A100-80GB | `cuda_async`, reserved at `R` = 77.89 GB | 44.76 GB | 83.01 GB, the card full for an instant | 197.7 / 124.1 / 60.4 | 688.8 |
| A100-80GB | BFC, preallocated at `R` | 44.77 GB | 77.89 GB, 5.12 GB always free | 204.7 / 110.6 / 64.5 | 672.1 |
| A100-80GB | BFC, growing (no preallocation) | 44.78 GB | 63.82 GB in map 0 | 195.4 (map 0) | — |

eqp is identical under every allocator (1288 of 1288 rows, maps 0–2).
`platform` holds nothing beyond what is in use (27.29 GB is the bytes in use
plus `O`), but it costs 6 % of the run and up to 10 % of a warm map, and XLA
reports no memory under it, so no receipt has a figure. BFC preallocated at
`R` never grows past `R`, so it removes the transient at no measured cost in
time. It still loses three things the policy keeps:

- the [per-stage receipt](#the-per-stage-receipt): BFC keeps no resettable
  in-use mark, so every stage reads the run's high-water (25.22 GB at every
  stage of the A100-40GB run, against stage peaks of 0.22–25.21 GB under
  `cuda_async`);
- one pool for XLA and the FFI: an FFI `cudaMallocAsync` draws from the CUDA
  pool outside BFC's arena, so it adds to `O`;
- room for a request that does not fit: a fixed arena refuses it, where the
  CUDA pool borrows free card memory until the next synchronize.

A growing BFC arena holds 19.0 GB (42 %) more than its peak in use, and an
unreserved `cuda_async` pool stalls 25–110 ms at every launch
([environment § GPU pool](../environment/overview.md#gpu-pool)).

### Streamed loops take a fixed tile {#fixed-tile}

**Streamed loops take a fixed tile, not the budget.** A loop that streams over
k, q, bands, centroids, samples or rows takes the most units whose per-rank
scaling bytes fit `runtime.tiles.TILE_BYTES` (1 GiB; `tile_units`). The tile
comes from the loop's shapes alone, so every rank computes it without a
collective, and no result depends on how much memory a run was given. Some
planners still size from `memory_per_device_gb` (the same value on every
rank, so they cannot deadlock); [the fixed-tile ruling](decisions.md#fixed-tile)
lists them. A 4 GiB tile is not used: at the Fe/Ni 20³ P64-local tile it cut
one χ₀ dispatch by 16 % (four-current) and 9 % (scalar) for four times the
tile's device memory.

<a id="streamed-chi-bank"></a>
**The χ bank streams instead of splitting into sample groups.**
The response stream's carry is the bank: the value and slope of every
response sample on the response rows. When every sample's carry does not fit
one group on the devices, the stream runs once with every sample, and each
finished row pass goes to SlabIO's per-rank streamed tier
([SlabIO](slab_io.md#streamed-tier); `response_bank.response_bank_residence`,
the tangential bank's device / host / file rule): host memory when it takes
at most half the host budget, else one file per rank. The devices hold two
pass carries, then three sample reads. Its size per map, over all ranks, is

```text
χ bank = 2 · N_samples · N_rows · 16 · d²       bytes
```

with `N_rows` the parent q rows the stream carries (their −q partner rows
too on an ordered bank without a unitary inversion) and `d` the carrier: `μ`
on the charge bank, the packed extent `μ + 3μ_T` on the four-current bank.
Ni 20³ charge (22 samples, 1062 parents, μ 1782): 2.4 TB per map, 34.9 GiB
per rank at P64. Fe 20³ bispinor (1062 parents, μ 1800, μ_T 900): 15.3 TB
per map, which needs that much free scratch and quota. The W bank comes on
top while the samples are solved
([tangential model §10](tangential_model.md#10-byte-model)), and both are
released within the map.

When the disk or quota cannot hold the bank on some rank, the samples
stream in disk groups, each try half the last, one store reused group
after group; once a disk group would be no larger than the device group,
the samples run in groups on the devices. A group re-runs every Green pair,
so this costs time: the Fe 20³ bispinor χ build at the P64-local shape is
about 420 s per map in one streamed group, 1,300–1,400 s in four disk
groups and about 5,700 s in device groups. Without the row-pass engine (the
full-k Green route) the bank stays in device groups.

## The unit: one Green's-function tile

```text
G_tile = 16 · N_k · n_s² · μ² / P          bytes per rank
```

`G_tile` is one k-resolved spinor `(μ, ν)` operator (`G`, `W`, `V`) sharded
`P(None, 'x', 'y')`. The screening and Σ stages hold a few of them; the
tangential capacity ledger warns above `3U` with `U = G_tile`. The ζ-fit
receipt prints every term as a `G_tile` ratio against a ceiling of
`4·G_tile`, so a reader sees at once whether the fit, rather than the GW
stages, sets the node count.

## ψ carriers: faces and band panels {#psi-carriers}

The raw-parent carrier holds two orientations of `ψ_{n k̄ s}(r_μ)` on the
`n_par` parents, stored as faces: bands on one mesh axis, centroids on the
other. A Green build can use local GEMMs after gathering the summed bands,
or retain these faces and exchange bounded SUMMA panels. The response and
scalar q-wedge Σ streams choose between them before execution; the small
Σ projector gathers only its projected bands for the length of its call:

```text
M_face  = 2·16·n_par·n_s·μ·N_b / P                      resident; μ and bands both tiled
M_axis  = 16·n_par·n_s·μ·N_b·(1/p_x + 1/p_y)
        = 2·16·n_par·n_s·μ·N_b / √P                     transient axis orientations at the requested band count
M_panel = 2·16·n_par·n_s·μ·(N_b/p_x)·(1/p_x + 1/p_y)    one Green build's two live SUMMA panels
```

The shared centroid sampler's band-complete X/Y outputs use `M_axis` with
the requested k rows. Its persistent streamed outputs have exactly the
requested k and band extents; only transfer/FFT tiles carry terminal pad
rows. The terminal tile is trimmed after its face reshard and before its
donated insertion, so no final padded-output crop overlaps two complete
face pairs. Transfer faces and the local full-grid FFT scan remain separate
workspaces; the fixed tile is a workspace target, not a total-process cap.

Centroid padding stays in the sampler's band-all-ranks layout before the
two independent face reshards. One padded transfer tile feeds both chains;
padding an already centroid-sharded face is avoided. In a CPU JAX 0.9.1
virtual-P4 abstract compile with `N_k=25`, `N_b=8191`, `n_s=2` and
`μ=2049→2052`, this reduced the insertion executable's temporary allocation
from 1,075,052,676 to 806,879,360 bytes (about 25%). This is a compiler
workspace comparison, not a measured process peak or a general capacity
bound; the actual executable and native workspace queries still set the
stage's admission price.

With `s = n_par/N_k` (the symmetry reduction) and `r = N_b/μ`, in units of
the Green tile:

```text
M_face / G_tile  = 2·s·r / n_s          independent of P
M_axis / G_tile  = 2·s·r·√P / n_s       grows as √P
M_panel / G_tile = 4·s·r / n_s          independent of P
```

A resident axis copy overtakes one full-centroid Green tile at
`√P = n_s/(2·s·r)`. These ratios compare layouts; current orbit-row streams
do not require four whole Green tiles. For
`μ = 10·N_b` (`r = 0.1`):

| P | n_s=1, s=1 | n_s=2, s=1 | n_s=4, s=1 | n_s=2, s=1/6 |
|---|---|---|---|---|
| 16 | 0.8 | 0.4 | 0.2 | 0.07 |
| 64 | 1.6 | 0.8 | 0.4 | 0.13 |
| 256 | 3.2 | 1.6 | 0.8 | 0.27 |
| 1024 | 6.4 | 3.2 | 1.6 | 0.53 |

(`M_axis` in `G_tile`; `M_face = 0.2·s/n_s` and `M_panel = 0.4·s/n_s` at every P.)

A Green build `G = ψ·diag(f)·ψ†` on faces is a batched 2-D SUMMA
(`distrib_la.panel_matmul`): band panels of at most `N_b/p` columns are
all-gathered (`A` over `y`, `B` over `x`), every k in each exchange, and
multiplied into the rank's own tile, so no reduction follows and no rank holds
a band-complete panel. Two panels are live, the next prefetched. Orbit-row
streams apply this product to their live row window and price its panels at
that geometry.
The Σ projector `ψ†·O·ψ` reshards the two
projected-band ψ faces to the band-complete orientation for the call and
contracts each `(μ_x, ν_y)` slab locally, so the μ-sized operator never
moves; its transient is `M_axis` at the projected band count.

## The Green-side stages

χ₀ and Σ(τ) read parent Greens through the fused k-convolution, which
unfolds them on its load, so no full-k Green exists. Both use orbit-closed
centroid-row windows; one window's parent Green is
`T_R = 16·n_par·n_s²·R·ν`, with `ν = μ/P_y` and `R` local rows.
A Green of real weights (real node times) reads its antiunitary
partner as `conj(G)` on the load; complex times build a conjugate-face
partner tile (`partner = 1`). Each Green build adds its two live SUMMA
panels `M_panel` ([§ ψ carriers](#psi-carriers)). New bytes per
rank beside what is live:

```text
χ₀ carry  = 16·n_out·N_k·μ²/P
χ₀ window = 2·(1 + partner)·T_R + 2·16·n_out·N_k·R·ν + ψ rows/panels + S_11
Σ(τ) row  = 16·[(2 + partner)·n_par·n_s²·ν + N_k·ν + 2·n_q,irr·ν + n_par·n_s·(N_b + 2·N_bΣ)]
Σ(τ) pass = R·(Σ(τ) row) + ψ + S_7
```

The ordinary static/GN minimax response keeps its original τ scan inside
each row window, then performs one forward χ transform after that τ sum.
It retains the same nodes, weights, edge references, spin trace and q
completion. The full χ carry remains distributed over every rank; it does
not retain whole-parent Gv/Gc matrices. The right ψ operand is prepared once
per call. The executable's actual additional XLA bytes and queried provider
scratch are admitted beside the live operands and placed unfold tables.
Row sizes follow the fixed 1 GiB tile, including for small one-window cases.

The bounded charge k-convolution cache is the single owner of placed charge
tables. Its live row variants share the same mesh/plan's eight fields, counted
once; full-span calls use the same window entry. There is no separate placement
cache retaining a plan after its last operator retires. That eviction
retires table-bound minimax and
response-bank programs, including cached layout variants and executables,
through the same owner. A caller or an unfinished background compile can
still retain its own operands until its work finishes; eviction does not
cancel work or delete borrowed device buffers. Recompiling an explicitly
held retired program returns its future to the caller without restoring
global cache ownership of the retired tables. Retirement follows table
liveness, not a rank-local future's completion state.

`S_11` is mode 11's split-arm scratch, at most one `T_R`
(`greens_function_kernel.chi0_kconv_scratch`); it is 0 on the single pass.
`S_7` is mode 7's, `min(R·ν, ⌊1 GiB / s⌋)` pairs of `s` B, at most 1 GiB, with
`s = 16·N_k·n_s²` plus `16·N_k` (the chunk's pair-major `W_R`) on the m7s arm (n_s = 2
where one padded column fits a block: 14³–21³ on an A100)
(`greens_function_kernel.sigma_kconv_scratch` → `ffi.fft.klead_unfold_scratch_bytes`; the
handler rounds an m7s chunk down to whole waves, so the price is its upper bound there); it is 0
on the single arm (bulk grids below about 14³ at n_s = 1 or 2).

The scalar Σ(τ) kernel (`ppm_tau_kernel._sigma_subtile_kernel`) runs each
rank's tile in row passes of whole centroid orbits (`gw.subtile_stream`):
`Σ(τ) row` is one local μ row's live set (`ν = μ/P_y`: the parent Green and
its partner, mode 7's output, `W_prep`, the pass's rows of `W` and its
partner on the q parents, the pass's ψ rows), and a pass holds the most rows
whose bytes fit the fixed tile. The ordinary minimax χ and scalar Σ share
`subtile_stream.green_operand_program` and `prepare_green_operands`. They price
operand placement and the complete consumer before allocating band-complete ψ.
If that live set fits, the axis layout gathers each sum band once and keeps
local Green GEMMs inside every row/node scan. Otherwise the summed-band ψ stays
on its canonical 2D faces and each row pass uses the shared bounded SUMMA Green
product. The axis layout forms its conjugated, merged right operand once
outside all row and node loops. The face layout borrows both canonical
Green faces directly; its compiled consumer includes any right
conjugation/layout workspace, without retaining a separately prepared
whole-wavefunction copy beside every convolution. The two axis banks occupy
`32·n_par·n_s·N_b·μ/sqrt(P)` bytes beside the original ψ owners. FACE retains
no additional full Green banks. Σ additionally places its small QP projector on
axis carriers, `16·n_par·n_s·N_bΣ·(μ/P_x + μ/P_y)`.

Admission uses the largest compiled new-byte figure over ranks: the maximum
of placement's new bytes and its retained outputs plus the consumer's new
bytes and queried native convolution scratch, against current room. Placement
finishes before the consumer starts, so their scratch does not overlap.
Compile-only calls compile and price these same programs without allocating
prepared ψ banks; the caches retain programs/geometry, not ψ payloads. Static
SX/COH uses the same choice and scalar node. Full-zone scalar pole sources
use the identity q-load of this row owner; their source storage, read and
canonical-to-packed conversion stay unchanged. Photon Σ sectors retain their
existing contracts.

These layout selectors use the exact minimum room over ranks. Neither
program cache keys on room, so rounding it to whole GiB can reject a fitting
axis layout without saving compilations. The global bucket policy remains
with the planners whose programs do depend on their bucketed room. Each
room query already performs its process agreement; compiled prices still
use a separate maximum agreement.

The charge direct and four-current response row streams use the same choice
inside `response_bank._compiled`: compile the complete incumbent stream
first, including its inline band gathers, and compare its new bytes plus native
convolution scratch with the minimum current room over ranks. An over-budget
axis body is never dispatched. Its fallback runs the same node and row scans
with `psi_k,s(mu_X,n_Y)` and `psi_k(n_X,s,mu_Y)` still on their 2D faces,
using the shared Green SUMMA and complex antiunitary partner. The merged right
operand is formed once on its face, so the summed-band resident and panels
scale as `N_b/P`, rather than a band-complete `N_b/sqrt(P)` copy. Each sample
group and streamed-bank segment admits its own actual executable; carry
donation and device/host/file bank residence are unchanged. The minimum face
route retains `memory.admit`'s existing warn-and-run policy if even it is over
room. This is a layout choice, not a quadrature or band-window approximation.
Full-zone scalar pole sources use an identity q-load table and the same row
owner; their file format does not select a whole-Green fallback. An
indivisible centroid orbit can require the whole local input window. Native
CUDA and ROCm k-box mode 7 use the fixed tile geometry to store and project
large outputs in smaller x blocks, retaining the input Green and `W_prep`
once and reducing the accumulated projection once. CPU, reference and ROCm
fallback convolutions form the full output before cropping, so they keep one
output call. Static (COHSEX), resident-pole and shared-pole scalar Σ retain
every spin component in each input window and stored block.
The spatial callable owns its pass-price metadata. Resident-pole τ wrappers
reuse that callable's geometry/provider identity; caller-supplied spatial
programs are not cached, and no raw-object-id price registry remains.

**Row passes run as one scan.** The passes of the Σ τ node, the static
τ = 0 node, the sector node and the charge and four-current χ streams are
equal windows of `R` rows (`subtile_stream.plan_windows`, through
`plan_passes` on the χ streams): `R` is the orbit-aligned row count nearest
the fixed tile, and every window ends on an admissible orbit cut. The
windows run as one `lax.scan` (`scan_passes`, `stream_passes`). The tiny
three-int32 pass descriptors are
barriered as a complete table before their indexed reads. This keeps the GPU
small-loop compiler from trying to evaluate a scalar optimization barrier on
a constant schedule; the protected read and bounded row-window lifetime stay
the same. Each window slices
its rows at a traced offset (`window_rows`); its unfold tables are cut from
the tables placed once per run, on the device (`window_load`, the one
table-cut owner); and the k-convolutions see only its live rows `[lo, hi)`
(the `live` operand, [k-convolution family](kconv.md#live-rows)),
so a padded window costs only its local Green GEMM rows. One program serves
every pass, so compile time does not grow with the pass count: the static
Σ program at 223 passes compiles in 0.8 s, and the streamed χ bank compiles one
segment program on the charge stream and at most four on the four-current
stream (claims 3114, 3122). A rank whose rows fit one
tile runs one whole pass.
The ordinary minimax χ response uses the same window and placement owners;
its τ sum precedes the final window transform, while the direct response
bank keeps individual node rows for its value/slope projections. On the
packed bispinor route the static photon
response (`V_packed`, `W_packed`, `2·16·Q·(μ + 3μ_T)²/P`) is deleted after
the static Σ channels read it, before Hartree and the τ sweep.

## Stage inventory

| stage | resident per rank (leading terms) | priced by | over budget |
|---|---|---|---|
| ψ(G) and centroid faces | charge fit: conj ψ(G) on the rank's G slots, `16·N_k·N_b·n_s·N_Gψ/P`. Centroid carrier `ψ(r_μ)`: `M_face` resident; each band contraction adds its transient panels, at most one `G_tile` ([§ ψ carriers](#psi-carriers)) | `plan_zeta_route_g` | with the fit |
| ζ fit, charge channel (route G) | C factor, one μ-batch working set, the ψ(G) slice; the Z store `16·Q·μ·N_G/P` lives on host or disk | `plan_zeta_route_g` ([§ route G](#route-g-zeta-fit)) | warns; smallest μ-batch |
| ζ fit, current channels (bispinor) | the same, at μ_T, with three factors, accumulators and Z stores | `plan_zeta_route_g(n_vertex=3)` ([§ route G](#route-g-zeta-fit)) | warns; smallest μ-batch |
| V_q | `V_acc` `16·Q·μ_L·μ_R/P`, one q-tile of ζ rows, G panels | `vq_tile_bytes` ([§ V_q](#vq-g-panels-and-q-tiles)) | warns; one q |
| V_q unfold | `16·N_k·μ²/P`, sharded `P(None,'x','y')` | — | — |
| tangential screening and Σ | response-bank faces, pencils, eigh workspace, then G and W tiles | the capacity ledger ([tangential model](tangential_model.md), byte model) | warns and admits, when a stage and its named concurrent stages exceed the budget |
| static / GN-PPM screening | distributed χ₀ carry and one orbit-closed parent-Green row window ([§ Green-side](#the-green-side-stages)); the GN fit's q block and output carry | `w_isdf._run_minimax_chi`: actual compiled additional bytes plus provider scratch; `fit_gn_ppm_from_wc_pair`: q-block proposal followed by the complete compiled scan | warns and admits |
| Σ(τ) sweep | resident pole fields, selected AXIS or borrowed FACE ψ, the small QP projector, then one row pass ([§ Green-side](#the-green-side-stages)) | `subtile_stream.plan_windows`: equal orbit-aligned windows, the pass count from the fixed tile and the shortest window that keeps it | — |
| matrix-element sweep (V_H, four-current) | the step's slabs, and FFT boxes `(2 + 2·n_comp)·n_s·N_r·16` per band of a band-layout operator | `mtxel_sweep.plan_sweep`: bands in the fewest chunks whose boxes fit the fixed tile | — |
| ψ loader off the fit plan (ζ reuse, current faces) | one band tile of G-flat rows, samples and faces | `gflat_memory_model.loader_band_chunk`: the fixed tile, at least the automatic 16 | warns; one scan row |
| kmeans candidate Gram | one square Gram face `16·M²/P`, the k batch's distinct X/Y WFN face sets, and a fixed square projector tile | `centroid.pivoted_cholesky`: k batches price each distinct band interval once; equal or nested windows share one face set and slice it inside the tiled Gram executable. Pruning releases its padded Gram after diagnostics and returns only points and linear-size receipts; slicing a ragged logical Gram for return would replicate it | warns and admits |
| moment bank, sample and line q spans | `(per_q·w + 16)` faces for a batch of `w` q parents; a streamed sample's span holds its value and slope rows, unpack and prefetch, `6·w` faces | `response_bank.parent_span`: under `linalg = local` (one whole matrix per rank) at least P parents per round, else the most within the fixed tile; rounds balanced | the ledger warns |
| sector Σ(τ) sweep (bispinor) | band-complete ψ, then one row pass: the four-spinor parent Green and partner, the kconv call's Σ rows, W(t)'s pass rows | `subtile_stream.plan_windows` (`mpa.sector_sigma.sector_tau_factory`): equal windows within the fixed tile | — |
| direct Γ head (bulk metals) | the compiled per-sample footprint × samples per call, split over every rank | `photon_direct_head.direct_gamma_chunk_plan`: the fixed tile, at least 2¹⁰ samples per rank, at most one 2¹⁷ replicate per call | — |
| head wings | `n_ends` gathered endpoint blocks `16·N_k·n_s·block·N_b` | `qsgw_head.head_wing_mu_block`: the fixed tile, at least 16 centroids | — |
| exciton_bands C_q | P_R and its update, one ψ chunk and its Pk | `vq_interp.build_cq_q_chunk`: q rows within the fixed tile | — |
| htransform R images | f(H)_R and the back-map L_R, `16·N_k·N_B·(N_B + N_T)/P` (`N_B` = spin × nodes, `N_T` = spin × centroids, 0 when they are the nodes), and the int16 WS code tiles `2·N_k·N_B·(N_B + N_T)/P` | `fh_interp.build_fH_R` compiled (one program); `node_pair_table` exact code extents; WS tie-mask tile's own compiled executable | warns and admits |
| BSE coarse-to-fine W | resident coarse W plus the fine output on the centroid face, and one fixed tile of embedding/FFT scratch | `bse_densify.make_w_densifier` scans centroid-face blocks within `TILE_BYTES`; shared `densify_w` admits the exact executable's output and scratch beside resident arrays | warns and admits |
| restart write | one sharded tile, `max(16·Q·μ²/P, 16·Q·μ·N_G/P)` (SlabIO writes per-rank hyperslabs) | — | — |

Replicated per-process metadata includes the TRS-augmented centroid permutation
and lattice-wrap tables, `O(n_sym·μ)`, and q-representative tables, `O(N_k)`; it does
not shrink with the mesh.

`SymMaps` constructs its dense `kq_map`, `kqfull_map` and unwrapped
`all_unfolded_qpt_ids` arrays independently at first access, then retains
the same writable, assignable NumPy arrays as before. WFN unfolding and
centroid placement use the linear-size k/q representatives and do not
construct these dense tables. The reduced-q lookup costs
`4·N_k·N_q,reduced` bytes; the full-q lookup and pair-label table each cost
`4·N_k²`. The small unwrapped q-vector list remains eager. `trivial_view()`
materializes these arrays before copying to preserve its historical shared
full-q/pair-label arrays and independent reduced-q copy. Q-consuming stages
therefore retain their previous table costs. No WFN/file handle is kept by
the lazy builders.

Centroid sampling (`common.wfn_transforms.gflat_to_rmu`) constructs dense
sphere-to-box inverse rows and separable Bloch-phase tables inside the local
FFT scan. For `c` scan rows and `b` local bands, a chunk touches at most
`w = min(N_k, ceil((c + b - gcd(c,b))/b))` distinct k rows. The per-k tables
therefore occupy at most `4·w·N_r` inverse bytes and, with Bloch phases,
`16·w·min(3·μ, nx+ny+nz)` phase bytes: shape alone chooses between the
three axis exponent vectors and their gathered centroid values. This is
never larger than the former `3·16·w·μ` phase bound. No table is retained
for every k row. The gathered per-FFT-row intermediates remain part of
the compiled scan tile. G-flat coefficients stay band-sharded, while the canonical sparse
sphere-index table remains replicated metadata. One local FFT still needs
its complete spatial grid; increasing P does not reduce that one-row floor.

The streaming centroid loader reuses geometry-only zero, reshard and donated
insertion programs through its existing signature cache. The unmasked keys
contain extents and shardings; centroid coordinates, WFN/Meta objects and
device data are not retained. Packed-mask reshards and insertions stay local
to the request. This avoids repeated tracing, lowering and persistent-cache
reads on warm calls without retaining sampled faces or request-index tables.

The centroid feature metric donates its overwritten density accumulator,
symmetry-row field bank and final scalar grid to their update programs.
Only the old accumulator is donated; the incoming band tile and the two
density operands remain valid. The first band tile becomes the running
density directly. Its temporary Python name is released after every window
update, so the final tile does not stay live across the next WFN load or
the symmetry fold. These updates preserve the existing band, k and symmetry
sum order and the replicated density layout.

When `0 < c < b` and `c` divides `b`, chunks stay within one k row. An
outer k scan retains that row's inverse and phase data while an inner band
scan reuses it; crossing-k chunks keep the flat scan. Both scans have
`unroll=1`. This trades repeated table construction for one retained k row,
not an all-k cache. Retention can prevent aliasing with FFT temporaries:
the P4 probe added about 0.19 MB of compiled scratch at 24×24×80 and 8.40 MB
at 128³, independent of whether the request had one or three k rows. The
actual compiled price includes this bounded storage; identical peak memory
is not assumed. Full-band chunks keep their previous flat-scan route.

<a id="ws-geometry-memory"></a>
The htransform WS constructor computes only addressable node×node and
target×node code tiles. It retains no all-site pair matrix or target×target
table. For local int16 code bytes `C`, host construction holds `O(C)` data
(up to three copies during remapping), plus one displacement block. Its
width `m` satisfies `m·(64·N_k + 32·n_image + 48) ≤ TILE_BYTES` (1 GiB,
at least one pair); this bounds the mask and sort buffers by a fixed tile,
not the process's total RSS. Repeated spin-site rows share one geometry
decision. The actual device tie-mask program is compiled and admitted before
execution; host copies and NumPy allocator overhead remain outside that
device price. Only geometry and the small agreed tie-pattern dictionary are
replicated. A Perlmutter P4 probe with 4,000 targets, 264 nodes and a 4×4 grid
reduced process peak RSS from 1,925 to 469 MiB and the retained replicated
host table from 149 MiB to 1.8 KiB (evidence:
`/pscratch/sd/j/jackm/cs_claude/bse_memory_prerelease_20261022/memory/`).

Fine W's resident size alone does not determine preprocessing capacity. The
shared densifier scans independent centroid-face blocks inside `shard_map`,
so coarse inverse FFTs, pair-dependent WS weights, and fine transforms use
one fixed tile instead of whole additional W carriers. Both centroid axes
can be blocked; no whole k axis is split. The final five-dimensional output
is the scan carry, avoiding a whole-output layout copy on exit. The exact
compiled price separates the output's face scaling from per-rank scratch,
and is admitted beside the already resident coarse input. Both the fine-grid
bundle and the exciton coarse-W path use that same execution owner; the
consumed coarse input is released before the Q sweep.

A Perlmutter CPU P4 probe with 10,000 centroids, coarse 3×3 and fine 6×6 has
3.6 GB coarse input and 14.4 GB fine output per rank. Centroid tiling reduces
compiled temporaries from 18.40 GB to 0.99 GB and measured peak RSS from
36.63 GB to 19.43 GB; execution changes from 23.06 s to 24.99 s. These are
W-only memory proxies, not physical 100-atom calculations (evidence:
`/pscratch/sd/j/jackm/cs_claude/htransform_followup_20261022/winterp_evidence/`).

The χ₀ node cannot be chunked over q, because the k-convolution needs the
whole k axis on every rank. A completed W of an earlier screening role is
spilled to host (`common.collectives.spill_to_host`) while a later role runs,
and restored afterwards.

## Route G: every ζ fit {#route-g-zeta-fit}

Each ζ fit forms `Z_q(μ, G)` a batch of centroids at a time and solves
`ζ = C⁻¹ Z` once, in G tiles
([ζ fit by μ-batches](zeta_fit_mubatch.md) owns the algorithm and the
per-object byte table). The charge channel is one fit; the three bispinor
current channels are one fit with `n_v = 3` channels at μ_T. Per rank:

```text
fixed    = n_v · C factor (16·⌈Q/P⌉·μ², whole tiles on their q owners)
         + centroid faces + sphere and cylinder index tables
Ψ        = 16·N_k·N_b·n_s·⌈N_Gψ/P⌉                        ψ(G), device-resident
work(b)  = max over the batch's three stages (GEMM + all-to-all;
           D cylinder; plane groups) of their live sets     b centroids, b = multiple of P;
                                                            plane stages at c_out ≤ b/P rows;
                                                            Z rows, Z/k-conv output rows and
                                                            the ζ-cylinder accumulator × n_v
store    = n_v · 16·Q·μ·N_G/P                               host or disk, never the device
```

The rule: with `M_f = budget − fixed`, ψ(G) is resident and the batch is
`b = (M_f − Ψ)/c_μ` in multiples of `P`, where `c_μ` is the per-centroid slope
of `work` at one plane per group. The stage-0 parent chunk and the
plane-group width take the most units whose bytes fit the fixed tile: a
parent's X_B rows, and a plane's share of the owner plane stage for one row;
the rows per chunk then fill the budget. No timing model picks a shape. The Z store stays on host when
it fits 0.8 of the node's `MemAvailable` over the processes on the node
(minimum over processes; `MemAvailable` and `MemTotal` are capped by the job's
memory cgroup, its limit less its usage net of reclaimable page cache,
`common.gpu_utils._meminfo_gb`); otherwise it is a slab_io scratch dataset. The
finalize streams it in G tiles sized to a quarter of the budget.

- **Over the budget**: when the smallest configuration (ψ(G) resident,
  `b = P`, one plane per group) exceeds the budget, it runs with one warning
  line. ψ(G) streaming is not implemented. Fix: more ranks or more memory
  per device.

The receipt (`ISDF μ-batch plan`) prints the route, batch, collectives per
batch against the minimum efficient payload, the Z-store placement, every
term in `G_tile` units, and the HWM estimate.

## Vq G panels and q-tiles

`V_q[μ, ν] = Σ_G conj ζ̃_q(μ, G) · v_q(G) · ζ̃_q(ν, G)`. ζ̃ stays
`P(('x','y'), None)`; each G step all-gathers one `(μ/p_x, g)` panel over
`'y'` and one `(μ/p_y, g)` panel over `'x'` inside a `shard_map`, so no rank
holds a whole `(μ, N_G)` face. One kernel launch contracts a q-tile, a scan
over q around the G scan. `gw.v_q_g_flat.vq_tile_bytes` owns the terms (μ
extents mesh-padded; `[+μ_R]` only for an off-diagonal tile):

```text
resident = 16·Q·μ_L·μ_R/P + 16·Q·μ_L/p_x + 16·Q·μ_L·n_sub/P    V_acc, g0, one-leg columns
per_q    = 16·(μ_L [+ μ_R])·N_G/P + 16·N_G                    ζ rows + replicated v row
work     = 16·(μ_L + μ_R [+ μ_R])·N_G/P                       sliced faces, R permute
         + 2·16·μ_L·μ_R/P                                     V_q carry
         + 16·g·(2·μ_L/p_x + μ_R/p_y)                         L, L·v, R panels
peak     = resident + work + q_tile·per_q
host     = 16·(μ_L [+ μ_R])·N_G/P per q                       phdf5 read staging
```

`_plan_vq_tiles` picks the panel width `g` (`vq_g_chunk_size` if positive,
otherwise the largest width ≤ 4096 whose panel all-gather fits
`LORRAX_COLLECTIVE_CHUNK_MB` and whose panels take at most half of what one q
leaves), then `q_tile` as every q whose rows fit the fixed tile and the
host staging budget, balanced across tiles. Both come from the shapes, so every
rank issues the same collective reads; the accumulators are priced, not capped.
When `resident + work + per_q` alone exceeds an explicit `budget_bytes`, or
one q's host staging does not fit, the planner warns and runs one q per tile.

For the charge channel, `ZetaG.contract_v` accumulates `V_q` tile by tile as
it forms ζ from the Z store, keeping ζ only at the columns the head consumers
name.

Live-fit callers adopt that returned V carry without retaining an unused
outer zero matrix. Grouped shell carries are allocated directly; their
constructor does not create a discarded V. All-face groups dispatch before
q-owner-only weight padding. The ζ-file-only pass calls
`contract_v(None, with_v=False)`: it applies the same factors, G masks and
writer layout without constructing or uploading a full zero Coulomb-weight
table or accumulating V. These lifetime changes do not close the grouped
[pricing gap](#pricing-coverage).

Its Z-store tile enters as `Z_q_XY(mu,G)` and moves to `Z_q(mu_X,G_Y)`
through the existing explicit `_to_mu_owner` exchange, cropping padded q
rows afterwards. A direct sharding constraint here replicated the whole
`Q_pad × μ × g` tile on every rank. The input payload is instead
`16·ceil(Q/P)·μ·g` bytes per rank and the output is `16·Q·μ·g/P`;
exchange packing adds shard-sized copies, included in the compiled G-tile
chain price. In particular, `Q < P` retains the input's one-parent floor,
without a P-parent replica. The same transport serves the face inverse
channel and the ζ-file-only pass.

On a reducing q grid, fresh scalar and photon-charge initialization retain
canonical `V_qirr(mu_X,nu_Y)` (`16·N_q,irr·μ_canonical²/P`). The GW restart
reader makes the same opt-in choice; the public readers and generic V producer
still return full-q arrays by default. The exact all-q finite census and
q/−q reciprocity maximum stream fixed-width panels through the same operator
unfold, with runtime q-index operands and a masked final tail. Gamma is
admitted and completed once before pricing the statistics-only program against
current room. Panel output/scratch is its actual compiled increment; a closed
form for the two transported panels is `2·16·B·μ_canonical²/P`. Restart keeps
its finite/trace gates without computing the unused reverse panel, and sanity
off transports Gamma only.

The fresh writer persists its unchanged authenticated canonical pre-unfold
capture, including the existing table/centroid/provenance checks. The late
`_select_coulomb_q_rows` seam authenticates an already-selected operator against
the current canonical wedge, then `_coulomb_on_wedge` packs both centroid axes
and attaches the packed tables. These canonical-row selection and packing
programs are compiled, priced and admitted before they run. Screening and Σ retain only
`V_qirr(mu_X,nu_Y)` (`16·N_q,irr·μ_packed²/P`). A nonreducing grid retains
its full-q carrier because every row is required; a full-storage fallback uses
the existing full unfold, admitted before materialization. Those cases retain
the `16·N_k·μ_canonical²/P` resident. No reduced-grid fresh initialization or
q-IBZ restart needs that full-zone resident.

## What the closed forms cannot see

### The compiled check

At sites using `runtime.aot_memory.check_chunk` (listed below), a planner's
closed form proposes a chunk and the check reads its compiled executable
before it runs. The
stage is priced as `fixed + chunk·per_unit` new bytes per rank beside what is
live. The executable's figure is `temp + outputs − alias` plus cuFFT plan
scratch (`compiled_new_bytes`), plus what the stage holds beside it that buffer
assignment cannot see: a donated carry the caller allocated, a native
handler's queried run-time scratch, a lookahead copy of the output. At these
sites the compiled figure corrects the chunk proposal:
within the room the chunk runs unchanged, whatever the closed form said. Over
the room, the slope is corrected from that one point,
`per_unit = max(per_unit, (compiled − fixed)/chunk)`, the chunk solved
directly, and compiled once more. There is no bisection; a chunk already at
the stage's minimum, or a second figure still over the room, warns and runs.
Each figure is the largest any rank read, so every rank recompiles, or not,
together; a figure compared per rank would split the ranks between a compile
and an all-gather, a silent hang (INVARIANTS 21).
The direct stream's check compiles through the dispatch's own executable cache
(`gw.response_bank._compiled`), so the checked executable is the one that runs.
Reading a figure costs 0.01–0.07 s for the direct stream, 0.02–0.12 s for the
Σ τ window per map, and 0.02–0.04 s once for the ζ batch (Fe 4³ and Fe 8³, P4).
When the samples split into groups, the one-group executable that prices the
temporaries is compiled once at map 0 and never runs (Fe 8³: 2.3 s cold,
0.15 s warm).

| stage | chunk | compiled figure available | what it misses (priced elsewhere) |
|---|---|---|---|
| response direct stream (`gw.response_bank`) | samples per group | yes: temporaries | the donated carry (an argument), mode 11's split-arm scratch (`chi0_kconv_scratch`) |
| Σ τ window (`gw.mpa.sigma.SynthesisTau.admit`) | row pass | yes: the first window executable | the synthesis GEMM's native workspace and mode 7's split-arm scratch `S_7` (added); the passes are not re-solved |
| ζ μ batch (`gw.isdf_fitting`, route G) | centroids per owner | yes: the batch executable, which the loop then runs | the lookahead batch's rows (its output, added) |

The direct stream's group is the one tangential size that follows the
budget: it fits the capacity ledger, then the compiled check above. The ledger
does not own the ψ carriers and other residents live when W is built; the
rule's `φ` ([§ Budget](#budget)) is all they have.
On a device the stream and the line selection are two phases, so a group
costs its carry plus the larger phase.

### A module larger than the device {#module-does-not-fit}

XLA's HLO rematerialization pass is off for every LORRAX program
(owner 2026-10-02). `runtime.set_default_env` adds
`--xla_disable_hlo_passes=rematerialization` to `XLA_FLAGS`, merged into any
pass list the caller already gives. The pass runs only when a module's peak
is above XLA's limit, where it can take hours to compile and frees little; on a
module that fits it changes nothing, so results are bitwise.

With the pass off, every compiled module is checked once, when jax builds
its executor for the module's first execution and before any of its buffers
exist (`runtime.aot_memory.refuse_over_device`, installed by
`common.jax_compile_cache`). When the module's buffers,
`temp + arguments + outputs − aliased`, exceed the card's total memory
(`runtime.xla_memory.device_total_bytes`), the run stops with

```text
GATE xla_rematerialization: REFUSED module '<name>': its compiled buffers
need 137.04 GB per device (temp 125.49 + arguments 11.55 + outputs 0.22
- aliased 0.22), and the device has 85.09 GB.  …
```

This gate compares the compiled buffer estimate with the card total, not
the budget; it is not a measurement of the process's full live set or of
unqueried native allocations. A module
between the budget and the card is not refused; the budget only warns. The
check runs on every GPU leg whose card total is readable (CUDA and ROCm). It does not run at compile, because planners compile
larger candidates on purpose to read their figures and then shrink
([the compiled check](#the-compiled-check), the kmeans Gram width); a
candidate that never runs is never refused.

### What compiled statistics miss

Measured on Fe 8³ charge, map 0, P4, A100-40GB, `memory_per_device_gb = 36`
(claim 2970). Peaks are the pool high-water of
the stage's section; prices are what the stage-memory table prints.

| stage | closed form | compiled | measured peak |
|---|---|---|---|
| response direct stream (group 16 of 22) | 34.01 (ledger only) | 35.58 (temporaries 9.71 = priced; residents 1.96) | 35.58 |
| Σ τ sweep (`d = 2`) | 13.30 | 14.02 | 13.96 |
| ζ μ batch (320 centroids) | 29.38 new, 29.97 total | 22.68 new | 23.73 total |

The latency-hiding scheduler grows the direct stream's temporaries. The check
then works as designed: the group of 16 compiles 34.08 GB of new bytes against
a 34.04 GB room, the slope is corrected, and the group of 15 runs, recompiled
once, priced and peaking at 34.55 GB.

The direct stream's peak was 0.51 MB above its price (the dispatch's small
arguments). Outside the pool, the CUDA context, NCCL communicators, library
handles and the mathdx modules held 2.65 GB per rank, and 4.7 GB with a
cuSOLVERMp context. Neither has a reserve of its own: both are in `φ` and `O`
of the rule ([§ Budget](#budget)).

When a minimum or recompiled chunk is still over its room, `check_chunk` prints one
warning line and the stage runs at that chunk; if the room is really
missing, it fails at allocation.

### Native handlers

The nvidia-mathdx k-convolution kernels allocate no device workspace beyond
shared memory, except the split-arm intermediates of modes 7, 8 and 11, which
XLA's scratch allocator grants ([k-convolution family](kconv.md#tiles)).
ROCm's native mode 4 k-minor convolution also takes one local input-sized
scratch buffer when its provider selects a split plan with output layout 1.
The provider's `lrx_kminor_workspace_bytes_rocm` query uses that same plan;
`runtime.aot_memory` parses the actual compiled local shape, grid and layout
and includes the exact bytes. Layout 0 and native transform-only mode 5 need
no extra arena allocation. For these native FFT families, missing query symbols or unparseable calls
refuse by name instead of admitting a guessed zero. The total adds the
maximum of this term and the XLA FFT plan workspace, because the calls in the
executable run serially. The [wide ROCm LU and general Fourier arenas](#pricing-coverage)
are explicit results owned by their providers; callers do not add them again
as FFT workspace.
On CPU the host `gw_conv` handler keeps a reused host arena of
`16·N_k·m_x·m_y` bytes and per-thread compact chunks, outside XLA.

## Communication cost model

`gw/comm_model.py` prices one collective that moves `V` bytes per rank among
`n_peers` other ranks as `T = α₀ + α_peer · n_peers + V / β`, with one
machine's constants (`gw.comm_model.MACHINES`, measured with
`tools/comm_model_bench.py`). It is a diagnostic: no planner picks a size or
a route from it. The route-G receipt prints each batch's collectives against
`min_efficient_payload(n_peers) = 4·β·(α₀ + α_peer·n_peers)`, the payload at
which latency is 20 % of the call, and the all-to-all floor at `β`.

## Planning a run

0. **Before the run, without a device.** `gw.stage_inventory` prices every GN-PPM
   stage per rank from the deck's sizes (and the ISDF, ζ, V(q) and Σ stages
   tangential shares): the residents each stage opens beside, plus its transient
   from the runtime's own price function (`plan_zeta_route_g`, `dyson_route_prices`,
   `ppm_tau_kernel.sigma_row_plan`/`sigma_pass_bytes`, `density_band_chunk`), else the
   arrays `LORRAX_MEMORY_TRACE` shows the stage allocating. `minimum_p(deck)` gives
   the smallest square mesh and its binding stage in seconds;
   `tools/scale_dryrun/dryrun.py` prints the table. Every stage of three one-node
   runs (MoS₂/hBN N_μ 8160 and 16320, Al₂O₃ 4³ N_μ 6000) prices within 0.86–1.22 of
   its measured peak (`tests/test_stage_inventory_calibration.py`). At large N_μ the
   first wall on a 2×2×1 slab is the ζ fit's C_q assembly, `(N_k + n_par)·n_s²·16μ²/P`
   of pair products beside C (1.5 G/P), not Σ(τ): the GN-PPM Σ(τ) node streams rows
   within the fixed tile, its band-complete ψ falling as `1/sqrt(P)`.
1. **Budget.** Leave `memory_per_device_gb = 0` for the model's budget, or
   set it lower to cap it (a value above it warns).
2. **Mesh.** Use a square mesh and price its actual padded carriers.
   Leading face terms fall as `1/P` in the divisible regime; a q-owned
   factor's `ceil(Q/P)` stops at one parent. Fixed scratch, replicated
   metadata, indivisible orbits and minimum tiles set further floors;
   band-complete ψ copies instead scale as `1/sqrt(P)`.
3. **Read the receipts** in `gwjax.out`: `ISDF μ-batch plan` for each ζ
   fit (the charge channel, and the current channels with `channels = 3`),
   which names its binder, and `Resident ψ` for the GW carriers.
4. **On a `memory over budget` warning or a `!! MEMORY: <stage> is PRICED OVER the
   allocator limit` banner**, run at
   the banner's estimated minimum from the memory quote or with more memory per device. For the ζ fit the smallest batch keeps ψ(G)
   resident; for V_q, more ranks, fewer centroids, or a smaller ζ sphere;
   `vq_g_chunk_size` shrinks only the panel workspace.
5. **Compare with the run.** `gwjax.out` prints MAJOR-STAGE DEVICE AND HOST
   MEMORY: each stage's device peak (max and min over ranks), the planner's
   price and `γ = peak / price`, or "no planner", the host columns, and the
   section that set the device peak. `γ > 1` is an under-estimate to
   investigate. The run's planner does not price host memory: the `rise` column
   names the stage that grew the process (`gw.stage_inventory.host_bytes`
   estimates it before the run: the Z store on the host plus a measured 9 GB/rank).

## The per-stage receipt

Every `common.timing` section is a tracker of `lxkit.memory`. A planner books its
price to the section that allocates it, `memory.book(price, stage=...)` (the
tangential ledger's `reserve(..., runs_in=...)`); that section waits on the work
dispatched before it, meters the allocator's high-water from the booking, waits on
its own work, and prints one rank-0 line:

```text
memory <stage>: priced P (operands a, xla b, vendor c, outside d) | measured M | γ g
```

with `γ = measured / priced` (gate: `1/1.3 ≤ γ ≤ 1`). Production reads the
allocator at section boundaries only and never waits.
`LORRAX_MEMORY_RECEIPTS=calibrate` waits on each priced stage at entry and exit
and reads every section; `LORRAX_MEMORY_STRICT=1` adds the census;
`LORRAX_MEMORY_TRACE=<path>` writes one JSON Line per closed section (live
arrays, allocator, the executables that ran) in either mode, without waits
(`lxkit.memory_residuals`). An increment price
(`priced(..., increment=True)`) adds the allocator's bytes in use at entry as its
`operands` as a part of its own (`resident at stage entry`), read once per stage, so
rows booked under one label cannot drop it. The tangential store's admissions are booked so, beside what the caller
holds whether or not a live ledger row names it. The reader is the leg's `peak_reader`: CUDA's pool
`CU_MEMPOOL_ATTR_USED_MEM_HIGH`, reset at each boundary (exact); BFC's
`peak_bytes_in_use`, which never resets, exact where the stage raised the mark
(first raise) and otherwise a sampler thread's polled `bytes_in_use` maximum,
printed `>=M` (a lower bound); CPU's live arrays, `live M`. Outside every reader:
NCCL and library workspaces outside the allocator, which the `vendor` and
`outside` terms price.

Host memory is read at the same boundaries from
`resource.getrusage(RUSAGE_SELF).ru_maxrss`, the process's resident
high-water mark. It never falls and cannot be reset, so the table prints two
numbers per stage, each the max over ranks: `host GB`, the mark at the
stage's last exit, and `rise`, the sum of what the stage's own intervals
added to the mark. A stage that stays below an earlier host peak shows
`rise +0.00`.

## Large-band matrix-element sweeps

The shared kinetic, ionic and dipole sweep distributes the G contraction
over all ranks and returns an x/y band-block matrix. Its local G-slab GEMM
streams interleaved rows from each x block through the existing band-block
reduce-scatter. For k tile `K`, `c` components and local row-panel width
`t`, the partial and its reorder cost `2 * K * c * p_x * t * nb * 16`
bytes. Their width is balanced within one fixed tile, with one row as the
floor. A zero-filled gather builds only the current bra panel, including
its tail; it never pads a second whole G-slab WFN. A padded output shard
can coexist with its final crop; executable memory analysis includes it.
The conservative k planner retains its earlier whole-partial price, so the
row-panel change does not raise k residency or change k grouping.

At 10,003 logical bands (10,048 carrier), one k and 64 CPU ranks, the
dipole probe's compiled temporary fell from 9,692,382,280 to 1,049,575,176
bytes/rank; output remained 75,721,728 bytes/rank. This is a nonphysical
small-G memory/layout fixture, with independent analytic values and a
4,802,880,432-byte three-component output file. It establishes bounded
partials, not a material's FFT throughput or total capacity. The full
requested `nk * c * nb² / P` output and the loaded WFN sphere remain
resident. Streaming those k outputs to existing SlabIO hyperslabs is a
separate future change. See the matrix-element probe in
`services/wfn_loader/bench/AXIS_STRESS.md`.
