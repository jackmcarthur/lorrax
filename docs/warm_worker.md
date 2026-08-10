# The warm worker, and what the per-leg floor actually costs

This note records what a four-GPU verification leg spends before it reaches
any physics, why the usual explanation for that cost is wrong, and what a
persistent worker does about it. Everything below was measured on Perlmutter
on 2026-08-10 at P=4 under BFC at `MEM_FRACTION=0.85`, with one leg alone on
its node. The evidence lives in `/pscratch/sd/j/jackm/warm_worker_2026-08-10/`
and the harness change lives on the sandbox branch
`feat/lx-warm-worker-2026-08-10`.

## The floor is not the queue, and it is not the container

The working assumption behind this lane was that a fresh cluster leg pays two
to four minutes of allocation, container start, jax import and device
initialisation before any physics runs. That assumption is wrong, and it is
wrong in a way that matters, because it points at the two components you
cannot do much about and away from the one you can.

An instrumented cold leg decomposes as follows. The timestamps come from a
wrapper that stamps the clock outside the container, a second that stamps it
inside, and the runtime stack's own phase accounting; the leg is otherwise
byte-identical to an ordinary `lx run -N 1 -G 4 -n 4`, because the launcher
takes `lx`'s own `--dry-run` line and splices the two stamps into it rather
than hand-writing a geometry.

| phase | seconds |
|---|---|
| fresh two-node `salloc`, interactive QOS, uncontended | 11.0 (0 against a live pool) |
| `lx` pool probe and Slurm step creation | 0.81 |
| Shifter container start | 1.80 |
| container to first line of python | 0.16 |
| `import runtime` | 0.03 |
| `initialize_communicator_stack` | 13.11 |
| first sharded `device_put` | 0.19 |
| first jit dispatch | 0.09 |
| second dispatch, same shapes | 0.00 |

The whole srun wall was 18.35 s, of which 16.26 s elapsed before the probe's
first real dispatch. So the floor is about sixteen seconds, not two to four
minutes, and allocation is either eleven seconds or — the normal case, since
the fleet keeps a pool — nothing at all. Container start, the other component
the assumption named, is under two seconds.

What actually dominates is `runtime.initialize_communicator_stack`, and it
breaks down further into 7.24 s for the environment, the jax import,
`jax.distributed.initialize` and backend creation; 3.56 s to build the 2×2
mesh and warm its NCCL cliques; and 2.24 s to arm the persistent compile
cache, which agreed 4513 entries across the four ranks and prefetched for
about a second. None of those three is something a launcher can shorten. All
three are things a process pays once and could then stop paying, which is the
entire argument for a warm worker.

It is worth being concrete about how large a share this is of a real leg. The
standard Si BSE verification leg — `bse_si_test.in` at `--n-val 4 --n-cond 4
--n-eig 20 --px 2 --py 2 --write-eigs` — took 22.74, 20.75 and 19.33 seconds
across three cold runs. Against a ~16 s floor, that leg is roughly three
quarters bring-up and one quarter physics. The GW stage on the same deck took
32 s cold, so it is about half bring-up.

## Where this floor sits among all the legs the fleet actually ran

The decomposition above is one instrumented leg, so it was checked against the
corpus: 2,183 `[lx] step … exit … in N s` lines and 1,904 runtime startup
banners across roughly 2,198 logs under `/pscratch/sd/j/jackm` dated 2026-08-06
to 2026-08-10. Three corrections come out of that, and the first is a
correction to this note's own headline.

**Sixteen seconds is the P=4 cold-cache floor, not the typical leg.** The
median step in the corpus is 31 s end to end with a median bring-up of 4.5 s,
because most legs are P=1 or hit a populated compile cache; the median
bring-up share is 19.5%. It is the P=4 legs that look like the measurement
above — the paired arms in `/pscratch/sd/j/jackm/import_audit_0809/_reports/`
spend 9.3–12.0 s of a 14.4–17.5 s wall on bring-up, so 65–70%, against 44–46%
for the P=1 arms of the same battery. Since the four-GPU rule makes every
verification leg P=4, the floor this note attacks is the right one to attack —
but it is the verification-leg floor, not a fleet-wide average, and quoting it
as the latter would be the same overreach the brief made.

**The "two to four minutes" figure appears to be inherited from two true
statements about other things.** `bin/lx`'s own header records 8–21 minutes of
`sbatch` queue wait for 30-second jobs, measured 2026-08-05 — before the pool
existed, which is the mechanism `lx` was built to delete. And
`tests/test_import_time_gate.py` says roughly four-fifths of a warm driver's
wall is bring-up rather than physics, which is a correct *ratio*; the wall it
is a ratio of is about fifteen seconds. A ratio and a queue-wait measurement
combined into an absolute, and the absolute survived.

**The claim is true for exactly one leg class, and it is rare.** Cold
compile-cache arming can genuinely cost minutes:
`/pscratch/sd/j/jackm/xd_parent/downfold_f.log` spent 292.4 s of a 313 s step
in bring-up, 284.5 s of it arming the cache. That is the only banner above 40 s
in 1,904, so about one leg in two thousand. The runner-up class is first-touch
mesh and NCCL setup at 25–36 s. Neither P=16 nor the pytest suite is an
exception — no P=16 banner exists in the corpus, and the largest
`env_and_distributed` row anywhere is 11.1 s.

### The larger prize is not bring-up at all

Worth recording alongside, because it dwarfs what this worker recovers. Of the
152 evidence directories with three or more steps, 115 — three quarters — are
strictly serial, never running two steps at once. Across them the inter-step
gaps total **32.4 hours against 17.5 hours of actual step wall**: the fleet
spends nearly twice as long waiting for an agent to read a result and submit
the next leg as it does computing. Median duty cycle is 0.41.

The examples are unambiguous. `/pscratch/sd/j/jackm/import_audit_0809/_reports`
ran ten independent A/B legs strictly back to back — `before_1`, `after_1`,
`before_2`, `after_2`, `before_3` at 02:36:43, 02:37:06, 02:37:25, 02:37:47,
02:38:05 — arms interleaved in order but never in time, with about 1,000 s
recoverable. `/pscratch/sd/j/jackm/perf_bse_0808/ab_final/_reports` ran 55
whole-node legs one at a time on a four-node pool. And
`/pscratch/sd/j/jackm/mpa_gamma_0809/_reports` fired 41 one-GPU legs that never
used more than two GPUs at once. That lanes *can* do better is settled by
`/pscratch/sd/j/jackm/mpa_farm16_0810/_reports`, which fired 16 one-GPU legs
three seconds apart for an occupancy of 5.97.

So a warm worker is the right fix for the bring-up term and it is not the
biggest lever available. Since co-tenancy became free with GPU placement on
2026-08-10, the larger recovery is fan-out, and the two compose: a worker that
answers in seconds makes a batched submit worth building, because the round
trip stops being amortised by a twenty-second leg.

## What the worker is

`lx warm start` launches an ordinary `lx run -N 1 -G 4 -n 4` whose payload
never exits. Four srun-pinned processes import jax once, build the 2×2 mesh
once, and then consume a work-queue directory. A leg is a JSON spec naming a
driver entry point, an argv, a working directory, and optionally the
environment the leg declares it needs. `lx warm submit` writes such a spec and
waits for the completion marker that rank 0 writes when the leg is done.

Two design choices are worth stating because the obvious alternatives do not
work.

The queue is sequence-numbered rather than scanned. Four independent processes
have to agree on which leg is next without talking to each other, and a
directory scan cannot give that agreement — two ranks scanning a shared
filesystem a millisecond apart can legitimately see different sets. A
monotonic sequence removes the question: every rank waits for `<seq>.spec.json`
by name, and the name is the agreement. The submitter allocates the number
with an `O_EXCL` create, which is atomic on Lustre, so concurrent submitters
cannot collide.

The rendezvous after each leg is a file barrier, not a collective. A jax
collective would be the natural choice and is certainly faster, but it cannot
time out: one rank that dies inside a leg would leave its three peers blocked
forever with nothing to report. The file barrier costs a few hundred
milliseconds and can give up, which is what lets a wedged leg become a
completion marker that says so.

Entry points are named as `module:function` or `module:__main__`. The second
form exists because `bse.bse_jax` has no importable `main()` at all — its
entire CLI, sixty-odd `add_argument` calls and every exit path, lives inside
its `if __name__ == "__main__":` block. `runpy.run_module(..., run_name="__main__")`
re-executes that module body, which re-runs its top-level
`initialize_communicator_stack()` call and gets the cached stack back. Warm,
by construction, with no change to the driver.

## Inherited state, which is the real hazard

Running successive legs in one process means each leg starts in whatever state
the last one left. This round mitigates that three ways and measures the
result rather than asserting it.

Each leg gets its own working directory, its own `sys.argv`, and an
environment that is applied before the leg and restored after it. Standard
output and standard error are redirected at the file-descriptor level rather
than through Python's stream objects, because the drivers reach C — phdf5,
NCCL, the FFI libraries and jax's own runtime all write on the descriptor, and
a `redirect_stdout` would capture half the output and let the other half leak
into the next leg's log.

Every completion marker reports the environment drift the leg left behind,
computed against the worker's launch environment. This is not decoration: it
caught the BSE driver setting `LORRAX_PHDF5_STRIPE_SIZE_FS` and
`LORRAX_PHDF5_STRIPE_COUNT` on itself, which persist into the following leg.
They are identical on every leg and the outputs are unaffected, so the drift
is benign here — but it is exactly the class of thing that would otherwise be
invisible, and the marker now names it.

The taint rule governs declared environment. A spec may declare the
environment variables it needs; if any of them differs from the value the
worker was launched with, the worker refuses the leg by name rather than
running it under its own environment. Marking the spec `--recyclable` instead
re-execs the quartet into the requested environment. The re-exec scrubs
`_LORRAX_JAX_DISTRIBUTED_DONE` on the way, which matters: that sentinel is set
by `init_jax_distributed` and cleared nowhere in the tree, so an inherited copy
would make the new generation skip `jax.distributed.initialize()` and come up
with no coordination client. Every marker records the generation it ran under.

A leg that reaches `finalize_process` is a special case, because that function
shuts down `jax.distributed`, drains the atexit registry and then calls
`os._exit` — it would take the worker with it and leave no trace. The worker
traps `os._exit` for the duration of a leg, records the leg as a hard exit,
and recycles. The quartet is genuinely poisoned at that point; the trap does
not save it, it just makes the death legible.

### The red twin

Three cold control legs produced a byte-identical `eigenvectors.h5`
(`ec41a980d3e756edf5a55ac69e691477`). Three successive warm legs of the same
deck produced that same digest, as did the leg submitted through the `lx warm`
verb and the leg run by the recycled generation. The printed eigenvalue vector
agrees too, at `1f100810eb6f31284896357d671e6b05` across the warm legs and the
cold control. So warm legs match each other and match a cold-leg control.

The env-sensitive twin behaves as designed. A leg declaring
`LORRAX_TIMING_TRACE=1` without `--recyclable` was refused in 0.10 s with the
offending key named; the same leg with `--recyclable` caused the quartet to
re-exec into generation 1, come back ready in 7.2 s, and run — still
byte-identical. Neither path contaminated anything silently.

## The numbers

| | cold | warm |
|---|---|---|
| standard Si BSE verification leg | 20.94 s (mean of 22.74 / 20.75 / 19.33) | 2.40 s (second leg) |
| three-leg light-lane sequence | 63.17 s | 10.71 s |
| GW stage on the same deck | 32 s | 7.35 s |

The warm worker itself costs 12.75 s of in-process boot, or 20.1 s of wall
including the srun step and container start — one floor, once. A recycled
generation comes back in 7.2 s, cheaper than a cold leg because the container
is already up.

So the three-leg sequence is about six times faster end to end, and if the
worker is already up when the lane starts — which is the point of a persistent
worker — the same three legs cost 10.7 s against 63.2 s. The owner's blocker
inventory asked for something that "would halve light-lane wall time"; this
is closer to a sixth.

## Honest limits

The crude worker cannot host a leg that needs a different mesh or a different
device count. This is not an implementation gap that a later patch closes
cheaply: `initialize_communicator_stack` raises rather than build a second
mesh for a different axis set, `RuntimeStack.reshape` is the only sanctioned
way to change shape, and `bse.bse_jax` never calls it. One worker means one
geometry, and since every verification leg is P=4 under the four-GPU rule,
that happens to be the geometry we want.

It also cannot host a leg from a different source tree. `PYTHONPATH` is fixed
when the worker launches, so a lane on another branch needs its own worker.

Environment-flip A/Bs are possible but not free. The refusal and the recycle
both work, and a recycle costs 7.2 s, but a worker that has recycled carries
the new environment as its launch environment from then on, and the taint rule
only guards variables a spec *declares*. An undeclared leg submitted to a
recycled worker inherits that environment silently. The generation is recorded
in every marker, so the evidence is attributable after the fact, but a lane
running an env A/B should use cold legs or two workers rather than one worker
that flips.

Legs that write through SlabIO and phdf5 collectively do work — this was the
open question about a shape-fixed communicator, and the answer is that it is
fine at fixed P. The GW stage, which persists `isdf_tensors_480.h5` and
`zeta_q.h5` through the collective writer, ran warm at 7.35 s with
`W0_qmunu` persisted, entered as `gw.gw_jax:main`. Note the entry point:
`gw.gw_jax:__main__` would have reached `finalize_process` and poisoned the
quartet, whereas its importable `main()` returns normally. Which spelling a
driver gets is therefore not cosmetic.

Two hazards are known and unmeasured this round. The tree carries several
dozen module-level jit and kernel caches keyed on shapes and mesh objects,
with no reset hook anywhere; a long-lived worker will accumulate them, and
nobody has yet watched a worker's memory over a few hundred legs. And a leg
that diverges across ranks — one rank raising while its peers sit in a
collective — wedges the quartet; the file barrier times out and reports, but
the collective itself is not interruptible from outside.

## What production-grade would need

None of this round is that, and the gap is worth naming rather than
discovering later. A production worker wants an importable `main(argv)` on
`bse.bse_jax` so the `runpy` path is a fallback rather than the norm; reset
hooks for the jit caches and the mesh so one worker can serve more than one
geometry; visibility in `lx status` and `lx gpuclaims`, so other lanes can see
that four GPUs are held by a worker rather than by a wedged step; crash
isolation, since today a leg that segfaults takes the quartet with it; and a
memory watchdog that recycles on growth instead of waiting for an OOM. Until
those exist, the warm worker is for the iteration between landings, and
landing evidence still comes from a cold leg.
