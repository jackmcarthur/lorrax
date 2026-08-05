# AS.7 scope banner — DRAFT TEXT FOR THE COORDINATOR TO LAND

wk_REL scale10k, 2026-07-29.  Item 3 of the coordinator's follow-up list.
**I have not edited either target file.**  Both blocks below are drop-in.

## The finding being documented

`JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` (the AS.7 "MEASURED UPGRADE" block) is
certified on GW and **fails on BSE**.  Evidence:

| job | stack | deck | result |
|---|---|---|---|
| 7879295 | impl=mpi, P=64 | GW, nb=1024, mu=10015 | rc=0, 1811 s |
| **7879458** | impl=mpi, P=4 | **BSE**, 785c | **rc=1, all 4 ranks** |
| 7879463 | gloo/ib0, P=4 | BSE, 785c | rc=0, 112 s |
| 7879470 | gloo/ib0, P=64 | BSE, mu=10015 | rc=0 (2 of 3 legs; 3rd was the writer bug) |

Failure signature — the loader completes, the q=0 head is injected,
`BSE problem (sharded 2x2): 4 cond x 4 val x 16 k = 256 dim` prints, and then
every rank dies at the first materialisation of the Lanczos result
(`bse/bse_jax.py:179  n_done = int(n_iter_done)`, i.e. inside the compiled
`bse_lanczos.solve_bse_sharded._full_run`):

    jax.errors.JaxRuntimeError: UNKNOWN: Buffer Definition Event: MPI:
    Communicator requested from a thread that is not the one MPI was
    initialized from. Multiple threads/devices per process are not yet
    supported.

VmHWM 0.60 GiB — nowhere near a memory limit.  Structural reason BSE differs
from GW: BSE's collectives sit inside a `lax.scan` inside a `shard_map` inside
the Lanczos loop, all under ONE jit, where XLA:CPU's thunk executor may run the
collective thunk on an intra-op pool worker; jax's MPI collectives backend caches
its communicator against the initialising thread and refuses.  GW's collectives
sit at module top level and run on the main thread.

Measured domain: P=4, MoS2 4x4, 785 centroids, TDA Lanczos.  NOT re-tested at
other P for impl=mpi (the failure is a thread-affinity refusal, not a scale
effect, so it is expected to be P-independent — but that is inference, not
measurement, and the banner says so).

---

## (1) For `SPEEDUP_SCORECARD.md`, section AS.7

Insert immediately **after** the ```` ``` ```` closing the AS.7 launch block
(house style: `> ⚠ CLAIM-DECAY` blockquote, correction in place, nothing
silently edited).

```markdown
> ⚠ **CLAIM-DECAY (wk_REL scale10k, 2026-07-29) — the MEASURED UPGRADE block is
> GW-ONLY.  `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` KILLS BSE.**
> Job 7879458 (BSE, 785c, P=4, this exact env cell): all ranks die at the first
> materialisation of the Lanczos result with
> `JaxRuntimeError: UNKNOWN: Buffer Definition Event: MPI: Communicator
> requested from a thread that is not the one MPI was initialized from.
> Multiple threads/devices per process are not yet supported.`
> — raised from `bse/bse_jax.py:179`, inside the compiled
> `bse_lanczos.solve_bse_sharded._full_run`.  VmHWM 0.60 GiB, so not memory.
> The same cell runs GW fine at P=64/mu=10015 (job 7879295, rc=0).
> **Mechanism:** BSE's collectives live inside a `lax.scan` inside a `shard_map`
> inside the Lanczos loop, all under ONE jit, so XLA:CPU's thunk executor is free
> to issue the collective from an intra-op pool thread; jax's MPI collectives
> backend binds its communicator to the thread that initialised MPI and refuses
> any other.  GW's collectives are top-level and run on the main thread.
> **Rule:** BSE (and anything else whose collectives sit under a scan/while_loop
> inside a single jit) MUST run on the certified gloo/ib0 default until the
> thread-affinity issue is resolved upstream.  Verified working on gloo/ib0:
> job 7879463 (P=4, rc=0) and job 7879470 (P=64, mu=10015, rc=0).
> Measured domain: P=4, MoS2 4x4, 785c, TDA Lanczos; the refusal is a thread
> binding, not a scale effect, so it is expected — not measured — at other P.
```

## (2) For `docs/dev/env_vars.md`, the `JAX_CPU_COLLECTIVES_IMPLEMENTATION` row

Append to the END of that row's cell (line ~250), after
"…the phdf5 open now WARNS when the granted level is below MULTIPLE (the AS.4b
race signature)." — keep it inside the same table cell:

```markdown
 **GW-ONLY — `mpi` BREAKS BSE (wk_REL, 2026-07-29).** Any kernel whose
collectives sit inside a `lax.scan`/`while_loop` inside a `shard_map` inside a
single jit can have XLA:CPU issue them from an intra-op pool thread, and jax's
MPI backend binds its communicator to the MPI-initialising thread: the BSE TDA
Lanczos dies on every rank with `MPI: Communicator requested from a thread that
is not the one MPI was initialized from` (job 7879458, P=4, 785c; raised at
`bse/bse_jax.py:179` inside `solve_bse_sharded._full_run`; VmHWM 0.60 GiB, not
memory). GW is unaffected (top-level collectives, main thread; job 7879295 rc=0
at P=64/mu=10015). Use the gloo/ib0 default for BSE — verified rc=0 at P=4 (job
7879463) and P=64/mu=10015 (job 7879470).
```

## (3) Optional third landing site

`LORRAX_FRONTERA_ADVICE.md` §10c ends with "Gloo/ib0 stays the certified
default." — that sentence is now load-bearing rather than conservative, and one
clause makes it actionable:

```markdown
Gloo/ib0 stays the certified default — and for BSE it is MANDATORY: `impl=mpi`
kills the TDA Lanczos on every rank with a communicator/thread refusal
(CORRECTED 2026-07-29, wk_REL, job 7879458; GW is unaffected).
```
