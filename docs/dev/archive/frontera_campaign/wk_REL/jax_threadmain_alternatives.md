# Removing the patched MPIwrapper from the `impl=mpi` path

*Can LORRAX run BSE under `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` without a
customised MPIwrapper?  Assessment of four routes, with the evidence for each.*

Author context: repo `/work2/08271/jackmc/frontera/lorrax` @ `49877c0`;
jaxlib 0.9.1 at
`/work2/08271/jackmc/frontera/lorrax_env/.venv/lib/python3.12/site-packages/jaxlib`.
Everything below about XLA's behaviour is read out of that installed binary
(`libjax_common.so`, full `.symtab` present, 696 852 symbols) or measured in a
job, never taken from documentation or memory.

---

## HEADLINE

**A fix exists that needs no patched dependency, no XLA flag, and no change to
BSE's communication pattern or its sharding.**  It is ~15 lines of Python and
one extra *O(1)* collective per mesh, executed once at start-up.

> Create each mesh-axis MPI communicator ONCE from the Python main thread, at
> mesh-construction time, by running a tiny `psum` over each axis and over the
> whole mesh.  XLA caches communicators in a process-global clique map; every
> later collective — including the ones a pool worker issues from inside the
> BSE Lanczos jit — then hits the cache and never reaches the
> `MPI_Is_thread_main` guard.

Measured at P=4 (job **7881047**, `sacct` COMPLETED), gate **unset**, on the
program whose gate-off cell is the established negative control:

| cell | `LORRAX_MPI_FORCE_THREAD_MAIN` | warm-up | refusals | result |
|---|---|---|---|---|
| `ctlOFF` | unset | none | **1** | FAIL — refusal reproduced |
| `warmOFF_psum` | unset | psum over x, y, world | **0** | **PASS** |
| `warmOFF_all`  | unset | + psum_scatter/all_gather per axis | **0** | **PASS** |
| `posON` | **1** | none | 0 | PASS (positive control) |

All three passing cells print byte-identical values
(`|ox-cx|=2.842171e-14  |oy-cy|=2.009718e-14`).  Warm-up cost: **149.5 ms**,
once per process.

Confirmed on the **real 785c MoS2 4×4 TDA Lanczos deck** at P=4 (job
**7881054**, COMPLETED 0:0), gate **unset**, driver unmodified:

| cell | gate | warm-up | refusals | eigenvalues (eV) |
|---|---|---|---|---|
| `bse_OFF_nohook` | unset | none | **8** | — (dies) |
| `bse_OFF_hook` | **unset** | 3 cliques, 153.6 ms | **0** | `[1.30537661 1.3504201 1.42411254 1.50449023]` |
| `bse_ON_nohook` | 1 | none | 0 | `[1.30537661 1.3504201 1.42411254 1.50449023]` |

— character-identical to job 7879697.2 (wrapper override) and job 7879463
(gloo), i.e. a four-way agreement.

And the discriminating control that explains everything (job **7881053**):
warming **only the world clique** — which is exactly what LORRAX's existing
`ensure_grouped_collectives_ready` barrier does — **still fails**.  It is the
per-axis sub-cliques that were missing all along.

This is **not** route 2.  It adds no collective to any Lanczos iteration, does
not touch `src/bse`, does not move a single byte of BSE payload, and cannot
change any intermediate's sharding — the warmed collectives are three 8-byte
`psum`s issued before the solver exists.  Against the owner's two-axis test it
is neutral by construction: the compiled BSE HLO is *unchanged*, so the
collective table and the allocation table are unchanged.

Ranking, revised for the owner's constraint:

1. **Route 1b — main-thread clique warm-up** (new; the recommendation).
2. **Route 3 — fix the guard upstream** (the real fix; draft in §4).  It is a
   genuine jaxlib bug, and it also removes the *other* override.
3. **Route 4 — keep the wrapper as an interim** while (2) lands upstream.
4. ~~Route 1 — an XLA/jax config knob~~ — **falsified**, twice (§1).
5. ~~Route 2 — restructure BSE~~ — **rejected without further work** (§2).

---

## 0. Ground truth, re-derived from the installed jaxlib

Every claim the brief asked me to verify rather than accept:

**(a) The refusal is XLA's, at communicator CREATION only.**
`xla::cpu::MpiCollectives::CreateCommunicators` @ `0xcc7a730`:

```
cc7a752:  callq  cc7cb00 <MPI_Is_thread_main>
cc7a757:  cmpl   $0x0,-0x54(%rbp)          ; if (!flag) ...
cc7a868:  callq  <absl::UnknownError>      ; esi = 0x93 = 147-byte message
cc7a7f1:  callq  cc7b140 <xla::cpu::MpiCommunicator::MpiCommunicator(int,int)>
```
Source path in the binary:
`external/xla/xla/backends/cpu/collectives/mpi_collectives.cc`.
**CONFIRMED.**

**(b) The operations carry no such guard.**
`objdump` over `MpiCommunicator::AllReduce` (`0xcc7b200`–`0xcc7b530`) contains
**0** references to `MPI_Is_thread_main`; likewise `ReduceScatter`,
`AllGather`, `AllToAll`, `CollectivePermute`, `Broadcast`, `Send`, `Recv`.
**CONFIRMED** — and this is the crux of the upstream argument (§4).

**(c) `Init()` requests FUNNELED and ignores `provided`.**
`MpiCollectives::Init()` @ `0xcc7a600`:

```
cc7a610:  lea    0xbac8dd(%rip),%rax   # <MPI_THREAD_FUNNELED>
cc7a617:  mov    (%rax),%edx           ; required = MPI_THREAD_FUNNELED
cc7a619:  lea    -0x20(%rbp),%rcx      ; &provided
cc7a61d:  xor    %edi,%edi             ; argc = NULL
cc7a61f:  xor    %esi,%esi             ; argv = NULL
cc7a621:  callq  cc7caf0 <MPI_Init_thread>
cc7a626:  ...    MPI_COMM_WORLD -> MPI_Comm_rank / MPI_Comm_size, then VLOG(1)
```
`-0x20(%rbp)` is never read again. **CONFIRMED.**

**(d) Both MPI entry points are MPItrampoline indirect stubs.**
```
MPI_Init_thread:     jmpq *0xc5b1cd(%rip)   # <MPIABI_Init_thread>
MPI_Is_thread_main:  jmpq *0xc5b1cd(%rip)   # <MPIABI_Is_thread_main>
```
i.e. the pip jaxlib wheel *cannot* do `impl=mpi` at all without some
MPIwrapper on `MPITRAMPOLINE_LIB`.  An *unpatched* upstream MPIwrapper is an
ordinary external dependency; only the patch is the objection. **CONFIRMED.**

**(e) NEW — the process-global clique cache.  This is what makes route 1b
work, and it is not mentioned in any of the campaign's existing notes.**

```
xla::cpu::AcquireCommunicator(CpuCollectives*, const CpuCliqueKey&, RankId)
   absl::Mutex::lock()                                   # 8641b80
   ... hash/probe node_hash_map< pair<CpuCollectives*, CpuCliqueKey>,
                                 ThreadSafeClique >      # GetProcessCpuCliques()::cliques
   on MISS: CpuClique::CpuClique(key)                    # 864211f
            absl::call_once -> xla::cpu::CreateCommunicator(...)
                            -> MpiCollectives::CreateCommunicators()   <-- the guard
   absl::Mutex::unlock()                                 # 86421f3
```

and the key is only the device list:

```
xla::cpu::CpuCliqueKey::ToString()   -> "devices=[" + StrJoin(CliqueKey::devices()) + "]"
xla::cpu::CpuCliqueKey::HashValue()  -> combines CliqueKey::devices() only
xla::cpu::operator==(CpuCliqueKey,CpuCliqueKey) -> CliqueKey::devices() only
```

So: **on a cache hit the guard is never evaluated, and the cache key is the
participating-device set alone** — not the op kind, not the payload, not the
rendezvous. A clique created by a tiny main-thread `psum` over mesh axis `x`
is the *same key* a `psum_scatter`/`all_gather` over `x` asks for later from a
pool worker.

Direct confirmation, `TF_CPP_VMODULE=cpu_cliques=3`, job 7881047 cell
`vlogOFF_warm` (P=4, device ids 0/2048/4096/6144, 2×2 mesh):

```
cpu_cliques.cc:121] Acquire communicator for clique key devices=[0, 4096] and rank 0
cpu_cliques.cc:135] Created a new clique for clique key devices=[0, 4096]
cpu_cliques.cc:97 ] Create a new communicator for clique key devices=[0, 4096] and rank 0
... devices=[0, 2048] ...                     (the y-group)
... devices=[0, 2048, 4096, 6144] ...         (the world clique)
```
three cliques per process, created during the warm-up, and **none created
again** when the 253.6 MB program ran.  In the paired `vlogOFF_nowarm` cell the
*only* clique that ever appears is the world one and the run dies on the
refusal.

**(f) The `is_sequential` thresholds are compile-time constants — this kills
route 1.**  See §1.

---

## 1. ROUTE 1 — a jax/XLA configuration knob.  FALSIFIED (two independent ways)

### 1.1 The sequential-vs-parallel decision is not configurable

`xla::cpu::ThunkExecutor`'s constructor (`0xc8abd40`) computes:

```
c8ac056:  cmp   %r14,%r12          ; small_buffers = (find_if_not(...) == end())
c8ac059:  sete  %al
c8ac063:  or    0xb8(%rdx),%al     ; |= execution_graph.is_sequential()
c8ac069:  mov   0x8(%rdx),%rcx
c8ac06d:  sub   (%rdx),%rcx
c8ac070:  sar   $0x3,%rcx          ; num_thunks
c8ac074:  cmp   0x8(%rbx),%rcx     ; <= options.execute_sequential_num_thunks_threshold
c8ac078:  setbe %cl
c8ac07b:  or    %al,%cl
c8ac07d:  mov   %cl,0xb8(%rdx)     ; is_sequential_
```

i.e. `is_sequential_ = graph_is_sequential || all_buffers_small ||
num_thunks <= threshold`.  The `Options` come from exactly one place —
`xla::cpu::CpuExecutable::Create` (`0xc8a5190`) — and they are a **stack
literal**:

```
c8a53de:  movq  $0x200,-0x108(%rbp)   ; execute_sequential_buffer_threshold = 512
c8a53e9:  movq  $0x8,  -0x100(%rbp)   ; execute_sequential_num_thunks_threshold = 8
c8a53f4:  movl  $0x0,  -0xf8(%rbp)
c8a53fe:  movb  $0x0,  -0xf4(%rbp)
c8a5449:  callq c8ad2d0 <ThunkExecutor::Create(ThunkSequence, Options const&)>
```

No `DebugOptions` read, no flag, no env var.  The only other callers of
`ThunkExecutor::Create` are `CallThunk::Create`, `ConditionalThunk::Create`
and `WhileThunk::Create` (found by disassembling the whole `.so` and grepping
the call targets) — same story.

The complete `xla_cpu_*` census in this binary is:
`collective_call_terminate_timeout_seconds`, `collective_call_warn_stuck_seconds`,
`collective_call_warn_stuck_timeout_seconds`, `collective_timeout_seconds`,
`copy_insertion_use_region_analysis`, `disable_loop_unrolling`,
`disable_new_fusion_emitters`, `disable_platform_dependent_math`,
`disable_slp_vectorizer`, `dump_unoptimized_hlo_snapshots`,
`emitter_verification_level`, `enable_concurrency_optimized_scheduler`,
`enable_custom_matmul_tiling`, `enable_experimental_deallocation`,
`enable_fast_math`, `enable_fast_min_max`, `enable_mlir_fusion_outlining`,
`enable_mlir_lowering`, `enable_mlir_tiling_and_fusion`,
`enable_platform_dependent_math`, `enable_xprof_traceme`,
`experimental_onednn_*`, `experimental_xnn_*`, `fast_math_honor_*`,
`generate_unique_c_style_kernel_entry_points`, `matmul_tiling_{k,m,n}_dim`,
`max_isa`, `memory_region_name`, `multi_thread_eigen`,
`parallel_codegen_split_count`, `prefer_vector_width`, `scheduler_type`,
`small_while_loop_byte_threshold`, `sparse_cuda_threads`,
`strict_dot_conv_math`, `use_acl`, `use_fusion_emitters`, `use_onednn`,
`use_thunk_runtime`, `use_xla_runtime`, `use_xnnpack`.
**Nothing forces sequential thunk execution and nothing sizes the intra-op
pool.**

### 1.2 Measured: neither remaining candidate works

Job **7881047**, gate unset, no warm-up:

| cell | knob | refusals | result |
|---|---|---|---|
| `ctlOFF_1core` | `taskset -c <one core>` → `tsl::port::NumSchedulableCPUs()` = **1** (verified from the driver: `schedulable cpus: 1`) | 1 | **still refused** |
| `ctlOFF_nothunk` | `XLA_FLAGS=--xla_cpu_use_thunk_runtime=false` | 1 | **still refused** |

`tsl::port::NumSchedulableCPUs()` (`0x67a6cf0`) really is `sched_getaffinity`
+ `__sched_cpucount`, so `taskset` is the genuine intra-op-pool-size lever —
and shrinking the pool to one thread does **not** move communicator creation
back to the main thread.  That is the interesting negative: the collective
thunk runs on a *pool* thread, and there is always at least one pool thread.

**Pricing, as asked.**  Because the 1-core cell still fails, route 1 costs
infinity for zero benefit and no perf number is needed.  For the record the
price would have been catastrophic anyway: LORRAX runs 2 ranks × 28 threads
per Frontera CLX node, so the knob would have been a 28× reduction in intra-op
parallelism on every GEMM and FFT in Σ and in the BSE matvec.

**Verdict: no configuration knob exists.**  This confirms the claim in
`docs/dev/mpi_collectives.md`, but now from the `Options` construction site
and from two measurements rather than from a flag-name census alone.

---

## 2. ROUTE 2 — restructure BSE.  REJECTED, and it would not have worked anyway

Per the owner's constraint this needed to be shown neutral on comm *and*
memory before being proposed.  I am rejecting it for a stronger reason: **it
does not fix the problem.**

The collective sites in the TDA Lanczos path
(`src/bse/bse_stack_matvec.py:110-131`, inside `lax.scan` inside `shard_map`):

| line | op | axis | role |
|---|---|---|---|
| 110 | `lax.all_gather(X_b, "y", axis=1, tiled=True)` | y | encode |
| 112 | `lax.all_gather(R, "x", axis=0, tiled=True)` | x | encode |
| 126 | `lax.psum_scatter(..., "x", scatter_dimension=0)` | x | decode, μ-sum + c→x scatter |
| 129 | `lax.psum_scatter(..., "y", scatter_dimension=1)` | y | decode, ν-sum + v→y scatter |

plus, in the SAME jit but **outside** `shard_map`: the V-term contraction
`jnp.einsum("kcvN,bcvk->bN", M_Y, X)` (`:157`) reduces over `c`, which is
sharded on `x`, so the SPMD partitioner emits an all-reduce over the x-groups;
and `solve_bse_sharded`'s `_full_run` (`src/bse/bse_lanczos.py:246-315`) wraps
the entire Lanczos/Davidson loop, its dot products (world reductions) and
`out_shardings=rep_eig` (world all-gather) in one more jit.

Two consequences:

1. **"Get the collective out of the `shard_map`/`scan`" does not help.**  The
   V-term collective is *already* outside `shard_map`, at the top level of the
   same jit, and it lands on the same parallel `ThunkExecutor`.  The
   discriminator is executable size, not lexical nesting — which is exactly
   what §3.1 of `eager_and_psumscatter_notes.md` established and what job
   7881047's `ctlOFF` cell (a single `shard_map`, no `scan` at all, 253.6 MB
   payload) reproduces: **no scan, no nesting, still refused.**
2. The only restructure that *would* move the collectives to the main thread
   is splitting `_full_run` so that every collective sits in its own jit small
   enough to trip `execute_sequential_buffer_threshold = 512 B` or
   `execute_sequential_num_thunks_threshold = 8`.  For BSE that means one jit
   per collective per Lanczos iteration, with the `T`-family
   `(μ_loc, ν_loc, ns, ns, nk)` tensor crossing every boundary as a
   materialised jit output.  That destroys the scan's whole reason for
   existing (one live `T` regardless of `n_trials`,
   `bse_stack_matvec.py:20-29`), and 512 B is not a threshold any real BSE
   intermediate can meet.

**REJECTED.**  Not "rejected with a caveat" — it fails the comm/memory test
*and* it fails the correctness test.  I did not run `wk_AN/colltable.py`
before/after because there is no "after" worth building.

---

## 3. ROUTE 1b — main-thread clique warm-up.  **RECOMMENDED**

### 3.1 The idea

At mesh-construction time, on the Python main thread, run one tiny `psum` per
mesh axis and one over all axes.  Each is its own jit with an 8-byte buffer,
so `all_buffers_small` and `num_thunks <= 8` both hold and XLA takes
`ThunkExecutor::ExecuteSequential`, which runs the thunk inline on the calling
thread.  `MPI_Is_thread_main` is therefore true, `MPI_Comm_split` succeeds,
and the clique lands in `GetProcessCpuCliques()::cliques`.  Every later
acquisition of that device set is a cache hit.

For a `(p_x, p_y)` mesh each rank needs exactly three cliques: its x-group,
its y-group, and the world — confirmed by the `cpu_cliques=3` VLOG in §0(e).

### 3.2 Evidence

**Job 7881047** (`sacct` COMPLETED) — mechanism, on the standalone 253.6 MB
reduce-scatter program that is the campaign's established negative control:
see the table in HEADLINE.

**Job 7881053** (`sacct` COMPLETED 0:0, 00:02:34) — discriminating controls,
to separate "per-clique caching" from the weaker "any collective first is
enough".  Scored on the driver's own `TM RESULT` line (4 ranks) and the
refusal count, not on `rc`:

| cell | warm-up | ranks PASS | refusals | verdict | expectation |
|---|---|---|---|---|---|
| `warmOFF_world` | **world clique ONLY** | 0 | 1 | FAIL | must fail |
| `warmOFF_axes` | x + y, **no world** | 0 | 1 | FAIL | must fail |
| `warmOFF_xonly` | x only | 0 | 1 | FAIL | must fail |
| `warmOFF_psum` | x + y + world | **4** | **0** | **PASS** | must pass |
| `ctlOFF` | none | 0 | 1 | FAIL | negative control |
| `posON` (gate ON) | none | 4 | 0 | PASS | positive control |

Every prediction held.  The mechanism is **per-clique** caching: warming any
proper subset of the cliques the program touches is fatal, and warming all of
them is sufficient.

**This retro-explains the whole "world-collective-first contract" story.**
`common/contract_bands.py:169`'s `ensure_grouped_collectives_ready` issues a
world-spanning barrier — which is exactly the `warmOFF_world` cell, and it
FAILS.  The old contract was not wrong about warm-up mattering; it warmed the
wrong cliques.  §3.1 of `eager_and_psumscatter_notes.md` was right to
falsify "ordering", and right that the executor path is the discriminator —
but the *actionable* variable is which device sets have communicators, and
nobody had tested that because the earlier clean-room probes (job 7879684)
were all small enough to run sequentially and so passed with no warm-up at
all, leaving the negative control void.

**Job 7881054** — the production test: the real 785c MoS2 4×4 TDA Lanczos deck
at P=4, gate unset, driver unmodified, warm-up injected by an out-of-tree
`runpy` wrapper (`wk_REL/thrmain_alt/tm_warmhook.py`) that patches
`bse.bse_ring_comm.create_mesh_2d` in memory.  Source: the all-MPI
workstream's manifest-verified snapshot at `6c7feb0`, `sha256sum -c` clean at
job start **and** at job end.  `sacct` COMPLETED 0:0, 00:02:09:

| cell | gate | hook | rc | wall | refusals | verdict |
|---|---|---|---|---|---|---|
| `bse_OFF_nohook` | unset | no | 1 | 67 s | **8** | FAIL — refusal on the real deck |
| `bse_OFF_hook` | **unset** | **yes** | **0** | 33 s | **0** | **PASS** |
| `bse_ON_nohook` | 1 | no | 0 | 28 s | 0 | PASS (positive control) |

```
[warmhook] 3 cliques warmed on the main thread for mesh (2, 2) axes=['x','y'] in 153.6 ms
```

Both passing cells print

```
Lowest 4 eigenvalues (eV): [1.30537661 1.3504201  1.42411254 1.50449023]
Lowest 4 eigenvalues (Ry): [0.09594338 0.09925401 0.1046703  0.11057795]
```

which is **character-identical** to job 7879697.2 (the wrapper-override fix
cell) and to job 7879463 (the gloo reference) — a four-way agreement:
gloo, `impl=mpi` + override, and `impl=mpi` + warm-up-with-no-override all
give the same spectrum.

The 33 s vs 28 s wall difference is single-shot and 30× larger than the
153.6 ms warm-up, so it is cell ordering / page-cache noise, not a cost
measurement.  Do not quote it as one.

### 3.3 Cost

* One-time, **149.5 ms** measured at P=4 for the three-clique warm-up
  (`warm=psum`); 358–401 ms for the redundant `warm=all` variant, which the
  discriminating controls show is unnecessary.
* Payload: three 8-byte `psum`s.  Independent of `N_mu`, `N_k`, `N_q`.
  Scales as `O(log P)` in latency, once per process, before the solver exists.
* Zero change to the compiled BSE HLO — therefore zero change to the
  collective table and zero change to the allocation table, by construction
  rather than by measurement.

### 3.4 It is also *safer* than the wrapper override

`MPI_Comm_split` is collective over `MPI_COMM_WORLD`.  The override lets XLA
call it from arbitrary intra-op pool workers; correctness then rests on every
rank creating cliques in the same order.  `AcquireCommunicator` serialises
creation *within* a process (one global `absl::Mutex`), but nothing serialises
it *across* ranks, so two independent cliques becoming ready in different
orders on different ranks is a latent `MPI_Comm_split` deadlock.  The warm-up
removes that exposure entirely: all cliques are created from one thread, in a
program-defined order, before any jit runs.

### 3.5 Proposed diff (NOT applied — `src/` is owned by the all-MPI workstream)

Add to `src/common/collectives.py` (it already owns the collectives helpers)
and call it from `bse.bse_ring_comm.create_mesh_2d`, from
`gw`'s mesh factory, and from `common.contract_bands` in place of the
falsified `ensure_grouped_collectives_ready`:

Note `src/common/collectives.py` currently imports only
`typing`/`__future__` at module scope and does its jax imports inside
functions; the sketch below keeps that convention.

```python
_WARMED_MESHES: set = set()


def warm_mesh_cliques(mesh, *, print_fn=print) -> float:
    """Create every mesh-axis MPI communicator on the calling (main) thread.

    jaxlib 0.9.1's ``xla::cpu::MpiCollectives::CreateCommunicators`` refuses
    with "MPI: Communicator requested from a thread that is not the one MPI
    was initialized from" unless ``MPI_Is_thread_main`` is true, and XLA:CPU's
    parallel ThunkExecutor issues collective thunks from intra-op pool
    workers.  The guard fires only on communicator CREATION, and
    ``xla::cpu::AcquireCommunicator`` caches communicators in a process-global
    map keyed *only* by the participating-device set.  So creating each clique
    once here — from the main thread, in a jit small enough that XLA takes
    ``ThunkExecutor::ExecuteSequential`` — makes every later acquisition a
    cache hit and the guard is never evaluated again.

    O(1): three 8-byte psums for a 2-D mesh, ~150 ms once per process at P=4.
    No-op off the mpi implementation or in a single-process run.
    """
    import os
    if os.environ.get(
            "JAX_CPU_COLLECTIVES_IMPLEMENTATION", "").strip().lower() != "mpi":
        return 0.0
    import time
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from jax.experimental.shard_map import shard_map
    if jax.process_count() <= 1:
        return 0.0
    key = (tuple(int(d.id) for d in mesh.devices.ravel()),
           tuple(mesh.axis_names))
    if key in _WARMED_MESHES:
        return 0.0
    t0 = time.perf_counter()
    tiny = jnp.zeros(1)
    axes = list(mesh.axis_names)
    groups = list(axes) + ([tuple(axes)] if len(axes) > 1 else [])
    for ax in groups:
        f = jax.jit(shard_map(lambda a, ax=ax: lax.psum(a, ax), mesh=mesh,
                              in_specs=(P(None),), out_specs=P(None),
                              check_rep=False))
        jax.block_until_ready(f(tiny))
    dt = time.perf_counter() - t0
    _WARMED_MESHES.add(key)
    if jax.process_index() == 0:
        print_fn(f"[collectives] warmed {len(groups)} MPI cliques for mesh "
                 f"{tuple(mesh.devices.shape)} in {dt*1e3:.0f} ms")
    return dt
```

Callers must invoke it **synchronously on every rank** (it is a collective) —
the same contract `ensure_grouped_collectives_ready` already documents.

### 3.6 What this does NOT remove

Override **1** of the wrapper — the `MPI_THREAD_MULTIPLE` upgrade — is still
required, and for an independent reason: XLA's collective *operations* still
run on pool threads concurrently with h5py/mpi4py collective MPI-IO on the
main thread, which is the AS.4b 4-failures-in-14-runs class.  Route 1b removes
override **2** only.  Route 3 is what removes both.

---

## 4. ROUTE 3 — the jaxlib guard is a bug.  Upstream issue/PR draft

### 4.1 Is it a bug, or deliberate conservatism?  My judgement: **a bug.**

The honest reading of the code is that it is *internally inconsistent*, and
that is what makes it a bug rather than a conservative choice:

* Under the level XLA actually requests — `MPI_THREAD_FUNNELED` — the guard is
  *correct*: FUNNELED means only the initialising thread may call MPI, so
  refusing `MPI_Comm_split` from another thread is right.
* But XLA then **violates that same contract everywhere else**.
  `MpiCommunicator::AllReduce/ReduceScatter/AllGather/AllToAll/CollectivePermute/
  Broadcast/Send/Recv` carry no guard (verified: 0 hits in each body), and
  XLA:CPU's own parallel `ThunkExecutor` demonstrably issues them from
  intra-op pool workers.  Under a FUNNELED grant those calls are undefined
  behaviour — and we have measured it as such: 4 failures in 14 runs at
  P=16 × 8 nodes, 3 segfaults + 1 hang, backtraces showing two threads of one
  rank inside `MPID_Progress_wait`.
* And `Init()` never reads `provided`, so there is no check at the one place a
  check would be sound.

So the guard is simultaneously **too strict** (it blocks a `MPI_Comm_split`
that is perfectly legal under `MPI_THREAD_MULTIPLE`) and **too lax** (it
polices creation but not the thousands of subsequent operations that have the
same requirement).  A deliberate conservatism would have guarded the
operations too, or refused at `Init()` when `provided < MPI_THREAD_FUNNELED`.

### 4.2 Draft issue for `jax-ml/jax` (component: XLA:CPU MPI collectives)

> **Title:** XLA:CPU MPI collectives request `MPI_THREAD_FUNNELED`, ignore
> `provided`, and gate communicator creation on `MPI_Is_thread_main` — which
> both rejects legal `MPI_THREAD_MULTIPLE` usage and leaves the actual
> collective calls unguarded
>
> **Environment:** jax/jaxlib 0.9.1, CPU backend,
> `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`, Intel MPI 2019 U9 via
> MPItrampoline/MPIwrapper 2.11.1, 4–64 processes, 1 device per process.
>
> **What happens.** Any program whose first use of a given
> participating-device set happens inside a jitted computation big enough for
> XLA:CPU to take the *parallel* `ThunkExecutor` path dies on every rank with
>
> ```
> UNKNOWN: MPI: Communicator requested from a thread that is not the one MPI
> was initialized from. Multiple threads/devices per process are not yet
> supported.
> ```
>
> Small programs pass, because `ThunkExecutor::is_sequential_` is true for
> them (`all_buffers_small` at the hard-coded 512-byte threshold, or
> `num_thunks <= 8`) and `ExecuteSequential` runs thunks inline on the calling
> thread.  Real programs take `Execute<ReadyQueue>`, a preceding thunk
> completes on an intra-op pool worker, and the collective thunk's continuation
> — including `AcquireCommunicator` — runs there.  Minimal reproducer: a single
> `shard_map` with `psum_scatter` over each axis of a 2×2 mesh with a ~250 MB
> local partial; the same program with a ~100-byte partial passes.
>
> **Why the guard is wrong.**
> `xla/backends/cpu/collectives/mpi_collectives.cc`:
>
> 1. `MpiCollectives::Init()` calls
>    `MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided)` and
>    never reads `provided`.
> 2. `MpiCollectives::CreateCommunicators()` opens with
>    `MPI_Is_thread_main(&flag); if (!flag) return UnknownError(...)`.
> 3. `xla/backends/cpu/collectives/mpi_communicator.cc`'s `AllReduce`,
>    `ReduceScatter`, `AllGather`, `AllToAll`, `CollectivePermute`,
>    `Broadcast`, `Send` and `Recv` have **no** thread check at all.
>
> `MPI_Is_thread_main` is a *thread-identity* test.  It is false on any
> non-initialising thread at every level, including `MPI_THREAD_MULTIPLE`,
> where calls from that thread are explicitly legal.  Meanwhile, if FUNNELED
> really were the contract XLA intends, then (3) is already undefined
> behaviour on every multi-threaded execution — which is what we observe in
> production: at 16 processes over 8 nodes we measured 4 failures in 14 runs
> (3 segfaults, 1 hang) with two threads of one rank simultaneously inside the
> MPI progress engine, and upgrading the granted level to
> `MPI_THREAD_MULTIPLE` (out of band, via the MPI wrapper) removed them.
>
> **Proposed fix.**
>
> ```c++
> // mpi_collectives.h
> int provided_thread_level_ = MPI_THREAD_SINGLE;
>
> // mpi_collectives.cc
> absl::Status MpiCollectives::Init() {
>   int initialized = 0;
>   MPI_Initialized(&initialized);
>   if (initialized) {
>     // Coexist with an embedding application (mpi4py, h5py-parallel, a
>     // Fortran driver) that has already initialised MPI.
>     MPI_Query_thread(&provided_thread_level_);
>   } else {
>     MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE,
>                     &provided_thread_level_);
>     mpi_initialized_by_xla_ = true;
>   }
>   if (provided_thread_level_ < MPI_THREAD_SERIALIZED) {
>     LOG(WARNING) << "MPI granted thread level " << provided_thread_level_
>                  << " (< MPI_THREAD_SERIALIZED). XLA:CPU issues collectives "
>                     "from intra-op pool threads; behaviour is undefined.";
>   }
>   ...
> }
>
> absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
> MpiCollectives::CreateCommunicators(...) {
>   if (provided_thread_level_ < MPI_THREAD_MULTIPLE) {
>     int is_main = 0;
>     MPI_Is_thread_main(&is_main);
>     if (!is_main) {
>       return absl::FailedPreconditionError(absl::StrCat(
>           "MPI: communicator creation requested from a non-initialising "
>           "thread, but MPI granted only thread level ",
>           provided_thread_level_, " (< MPI_THREAD_MULTIPLE). Rebuild or "
>           "configure the MPI implementation for MPI_THREAD_MULTIPLE."));
>     }
>   }
>   ...
> }
> ```
>
> i.e. **test the capability, not the thread's identity**, and request the
> level XLA actually needs.
>
> **Two related improvements, separable from the above.**
>
> * `MPI_Comm_split` is collective over `MPI_COMM_WORLD`, so allowing it from
>   pool threads makes correctness depend on all ranks creating cliques in the
>   same order.  `xla::cpu::AcquireCommunicator` serialises creation within a
>   process but nothing serialises it across ranks.
>   **`MPI_Comm_create_group` is collective only over the given group** and
>   would remove that ordering requirement entirely — a strictly better
>   implementation for the sub-clique case that XLA actually uses.
> * The `Finalize()` path registers an `atexit` handler; on Intel MPI the
>   post-`atexit` C++ teardown then makes one more MPI call and the process
>   exits non-zero after a successful run ("Attempting to use an MPI routine
>   after finalizing MPICH").  XLA should only finalize MPI if it initialised
>   it (`mpi_initialized_by_xla_` above).
>
> **Workaround for anyone hitting this today** (no jaxlib rebuild): create each
> clique once from the main thread before entering the big jit — a tiny
> `jax.jit(shard_map(lambda a: lax.psum(a, axis), ...))` per mesh axis and one
> over all axes.  `AcquireCommunicator`'s cache is keyed only by the
> participating-device set, so every later acquisition from a pool thread is a
> hit and the guard is never reached.  Measured: 149 ms once per process at
> P=4, and it makes a previously 100 %-fatal program pass with byte-identical
> results.

### 4.3 Honest assessment of route 3 as a *plan*

It is the right fix and it deletes both overrides, but it is not a schedule
LORRAX can depend on: a jax/XLA PR from an outside contributor to the MPI CPU
backend, then a jaxlib release, then an environment bump.  File it, and ship
route 1b now.

---

## 5. ROUTE 4 — if the wrapper must stay: the minimum viable form

With route 1b in place, override 2 (`MPI_Is_thread_main`) can be **deleted**.
What must remain is override 1 only:

* **What is overridden:** `MPI_Init` / `MPI_Init_thread` forward to
  `PMPI_Init_thread(..., MPI_THREAD_MULTIPLE, ...)`.  Requests are upgraded,
  never downgraded; init order unchanged.  That is one hunk against upstream
  MPIwrapper v2.11.1 and it is required by XLA's *operations*, not by its
  communicator creation.
* **Versioning / runtime detection.**  Do not rely on a build recipe being
  re-run.  Have LORRAX detect the granted level at run time rather than
  assuming the wrapper is on the path: `mpi4py` with `rc.initialize=False`
  then `MPI.Query_thread()` reports it in-process (this already works — job
  7879684 measured `granted_thread_level=3 (MULTIPLE)` this way), and
  `ffi/phdf5/cpp/context.cc` already warns when the level is below MULTIPLE.
  Promote that from a warning to a startup banner line printed by
  `runtime.announce_cpu_collectives()`, and stamp the wrapper's `.text`
  sha256 next to it so a log alone identifies which binary ran.
* **How a user on another machine builds it:**
  `config/frontera/build_mpiwrapper.sh --fresh` on a login node (MPIwrapper
  compiles Fortran bindings; the py312 container has no gfortran).  The script
  pins the upstream commit, applies the patch, and verifies the override in
  the *machine code* — it disassembles `MPIABI_Init_thread` and asserts
  `required` is hard-set to 3.  Keep that check; a wrapper that silently
  grants FUNNELED loads exactly like a good one.
* **Failure mode if they do not build it:** the run *succeeds* most of the
  time and segfaults or hangs ~29 % of the time at P≥16, inside
  `MPID_Progress_wait`, at whatever boundary happens to overlap h5py
  collective MPI-IO with an XLA collective.  That is the worst possible
  failure mode — intermittent, scale-dependent, and it looks like a fabric
  problem.  It must be a **hard refusal at start-up**, not a warning: if
  `MPI.Query_thread() < MPI_THREAD_MULTIPLE` and `process_count() > 1` and the
  implementation is `mpi`, LORRAX should fail fast and name
  `build_mpiwrapper.sh`.
* **Best interim of all:** offer the upgrade *without* the patch by shipping
  the one-hunk change upstream to `eschnett/MPIwrapper` as an env-gated
  feature (`MPITRAMPOLINE_THREAD_LEVEL=multiple`).  MPIwrapper is a small,
  actively maintained project and this is a generically useful knob — it would
  turn a customised dependency into a stock one with an environment variable,
  which is exactly the outcome the owner asked for.

---

## 6. What I could NOT test, and why

* **Route 1b at scale.**  Everything here is P=4 / 2 nodes (three jobs,
  all `sacct` COMPLETED 0:0, total 7 min 19 s of wall).  The warm-up does
  *not* introduce the AS.4b concurrent-progress exposure (it strictly reduces
  MPI-from-pool-thread activity relative to the override), so I expect it to
  be safer at scale rather than riskier — but "expect" is not "measured", and
  the P=16 × 8-node rep ledger that Phase 2 of the all-MPI workstream is
  running for the override should be repeated for the warm-up before any
  default flips.  I deliberately did not launch anything at that size.
* **Route 1b for GW / Σ meshes.**  I verified the clique set for a 2-D
  `(x, y)` mesh.  Any code path that builds a *different* mesh (prefix-sliced
  meshes, the `contract_bands` sub-meshes, a 1-D mesh) needs its own warm-up
  call; `warm_mesh_cliques` is written to be mesh-generic and idempotent, but
  I have not enumerated LORRAX's mesh factories.  `TF_CPP_VMODULE=cpu_cliques=3`
  on one production run would settle it in a single job — that is the cheapest
  next step.
* **Whether XLA ever needs a clique that is not a mesh-axis subgroup.**
  Collective-permute over irregular source/target pairs, or an all-to-all over
  a reshaped mesh, would create a device set the axis-wise warm-up misses.  I
  saw none in the BSE or repro HLO (the `cpu_cliques=3` dump shows exactly
  three per rank), but I have not swept the whole code base.  The mitigation
  is cheap and general: if the refusal ever reappears, the VLOG names the
  missing device set exactly.
* **The upstream PR itself.**  Drafted, not filed — filing against
  `jax-ml/jax` is the owner's call, and the text above is written to be
  pasted.
* **`--xla_cpu_multi_thread_eigen=false`.**  Not measured.  It was not worth a
  cell once `taskset`-to-one-core (a strictly stronger restriction on
  intra-op parallelism) was shown to still refuse.

---

## 7. Artifacts

| path | what |
|---|---|
| `wk_REL/thrmain_alt/tm_repro.py` | mechanism driver (253.6 MB reduce-scatter + optional main-thread clique warm-up, 6 warm modes incl. discriminating controls) |
| `wk_REL/thrmain_alt/tm_probe.sbatch` | job 7881047 — mechanism + route-1 knobs + `cpu_cliques=3` dumps |
| `wk_REL/thrmain_alt/tm_probe2.sbatch` | job 7881053 — discriminating controls, corrected scoring |
| `wk_REL/thrmain_alt/tm_warmhook.py` | out-of-tree `runpy` wrapper; warms cliques for the **unmodified** BSE driver |
| `wk_REL/thrmain_alt/tm_bse.sbatch` | job 7881054 — real 785c BSE deck, gate unset, hook vs no-hook vs gate-on |
| `wk_REL/thrmain_alt/logs_<jobid>/`, `bselogs_<jobid>/`, `*.tsv` | per-cell logs and ledgers |

**Instrument defect found and recorded:** job 7881047's `verdict` column is
wrong for every cell.  That harness does not export
`LORRAX_MPI_FINALIZE_FIX=skip_atexit` or the overlay `sitecustomize`, so every
run — *including the gate-ON positive control* — exits `rc=1` after
succeeding, exactly as `docs/dev/mpi_collectives.md` documents.  The `refusal`
column and the driver's own `TM RESULT` line are the load-bearing ones, and
they agree with each other on all eight cells.  Job 7881053 scores on those
instead.  Nothing in the conclusions rests on `rc` from job 7881047.

Nothing was committed.  Nothing under `/work2/.../lorrax` was modified.
