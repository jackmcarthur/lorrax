# Silent data corruption in XLA:CPU gloo `reduce-scatter` (`jax.lax.psum_scatter`)

*Report prepared 2026-07-29. Everything below is measured on the cluster
named in §2; claims are separated from non-claims in §5 and §6.*

## 1. Summary

Under the **XLA:CPU gloo collectives backend**
(`JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo`), a `jax.lax.psum_scatter`
over a 2-D device mesh **intermittently returns wrong data with no error,
no warning, and a zero exit code**. Approximately **5 % of executions**
are affected (5 of 100 in the controlled run of §4.1; 6 of 100 in an
earlier run), and **80 % of process lifetimes** see at least one bad
execution. The corrupted result is of the same order as the correct one —
not a NaN, not an obviously broken value — so a computation consuming it
produces a plausible wrong answer. In our case it silently destroyed the
Hermiticity of a matrix that is Hermitian by construction, which is the
only reason we noticed.

The same program is bit-exact when the identical reduction is expressed
with `psum` (all-reduce) instead of `psum_scatter` (§4.4), which is what
localizes the fault to reduce-scatter.

Later results narrow it much further. The **mpi** collectives backend runs
the identical program clean in **604/604 executions** (§4.6), so this is not
XLA's collective lowering in general. The corrupted data is **not any
missing, doubled, zeroed, substituted, mis-offset or wrong-clique combination
of the correct contributions** — every such model is excluded arithmetically
against the exact operands (§4.6b, §4.6c, §4.6e) — and it is **not
uninitialised or freed heap** either, which we measured rather than assumed
(§4.6d). What is left is sharp and unresolved: the wrong value is
**deterministic and machine-independent** — seven values recurring bit-exactly
across three node pairs and three jobs — yet it is not any function of the
correct operands constructible from outside the process.

**Where the suspect code is — REVISED 2026-07-29, superseding an earlier claim
in this report.** Earlier revisions of this summary asserted that this jaxlib's
gloo contains *no reduce-scatter algorithm at all* and that
`xla::cpu::GlooCommunicator::ReduceScatter` was therefore XLA's own
allgather-then-local-reduce — concluding that the bug was XLA's rather than
gloo's. **That was wrong, and it was wrong in the evidence, not just the
conclusion.** See the corrected §4.6c. gloo's reduce-scatter *is* compiled into
this build, and the suspect is
`gloo::ReduceScatterHalvingDoubling<std::complex<double>>::run()` — an
**in-place recursive-halving algorithm that works out of internal scratch
buffers**, last functionally modified **2018-02-09** — as selected and driven
by XLA's `GlooCommunicator::ReduceScatter` wrapper. That relocation matters for
where this is filed (§4.6c) and it supplies, for the first time, a mechanism
that can reach the out-of-range value in §4.6d.

## 2. Environment

| | |
|---|---|
| jax / jaxlib | **0.9.1 / 0.9.1** |
| numpy | 2.4.3 |
| Python | 3.12 (Apptainer image `py312.sif`) |
| backend | **CPU** (`JAX_PLATFORMS=cpu`, x64 enabled) |
| collectives | **gloo** (`JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo`), socket iface pinned to `ib0` |
| mesh | 2x2, **4 processes x 1 device** (`jax.distributed.initialize`, MPI-launched) |
| hardware | TACC Frontera CLX (Intel Xeon Platinum 8280, 56 cores/node), 2 ranks/node, 28 threads/rank, Mellanox IB (ib0) |
| OS | CentOS 7, kernel 3.10.0-1160.90.1.el7 |
| threading | `OMP_NUM_THREADS=28`, ranks pinned with `taskset` |

The mesh axes map to replica groups as: `'y'` = consecutive ranks
(intra-node pairs `{0,1}`, `{2,3}`), `'x'` = stride-2 (inter-node
`{0,2}`, `{1,3}`).

**The mpi comparison has now been run** — see §4.6. The blocker described
in earlier revisions of this report (the mpi backend refusing to create
grouped communicators) was jaxlib's own `MPI_Is_thread_main` guard in
`xla::cpu::MpiCollectives::CreateCommunicators`, which we worked around in
our MPI wrapper. Under `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` the
identical program is clean in **604/604 executions** across two
allocations; gloo, interleaved with it in the same allocations and on the
same nodes, is not.

**Correction to the `ib0` row above, added with §4.6.** `GLOO_SOCKET_IFNAME`
is **inert** with this jaxlib — the string is absent from every `.so` in
jaxlib 0.9.1 (verified by `strings`), and jax constructs gloo with no
`interface=` argument. The **in-project** harness (Run A, §4.1) *was* pinned
to `ib0`, because our runtime re-registers the CPU backend factory and passes
`interface="ib0"` explicitly; its log records
`Gloo collectives pinned to ib0 (192.168.46.242)`. The **standalone
reproducer** (§3) exports `GLOO_SOCKET_IFNAME=ib0` but does not do that, so
it ran on jax's default interface selection, not on an explicit `ib0` pin.
The corruption therefore reproduces under **both** interface configurations.
See §6.5.

## 3. Reproducer

`gloo_psum_scatter_repro.py` (accompanying, **no project imports** — pure
`jax` + `numpy`).

Each rank builds a deterministic local partial `loc` of shape
`(n_q=16, m=99, n=10008)` complex128 (**253 MB**, the reduce-scatter
input payload) from its own `jax.lax.axis_index` values — no host arrays,
no `device_put`, no RNG, so **every rank's operand is bit-identical on
every repetition**. Inside one `shard_map` it then computes the same
mathematical sum three ways:

```python
ox   = psum(psum_scatter(loc, 'x', scatter_dimension=2, tiled=True), 'y')
oy   = psum(psum_scatter(loc, 'y', scatter_dimension=2, tiled=True), 'x')
full = psum(loc, ('x', 'y'))                       # CONTROL: all-reduce only
cx   = dynamic_slice(full, axis_index('x') * n//2, n//2, axis=2)   # local
cy   = dynamic_slice(full, axis_index('y') * n//2, n//2, axis=2)   # local
```

Two independent detectors:

* **DRIFT** — each repetition against repetition 0, at **exact zero**.
  The operands never change, so any nonzero difference is corruption, not
  arithmetic. This is the decisive detector.
* **CONTROL** — chain against all-reduce-plus-local-slice. These differ
  only by reduction association (measured floor ~1e-16 relative), so a
  gap of order the answer is corruption.

Run: `srun -N2 --ntasks-per-node=2 -n4 python3 gloo_psum_scatter_repro.py --reps 20`,
repeated across 5 fresh processes.

**It reproduces standalone.** Job 7879558, first process, repetition 18
of 20 (jax 0.9.1 / jaxlib 0.9.1, python 3.12.13, backend cpu, 4
processes, collectives=gloo):

```
rep 0:  |ox-cx|=2.842171e-14   |oy-cy|=2.009718e-14   scale=1.119664e+02
*** rep 18: CONTROL gap  |ox-cx|=2.667173e+01   |oy-cy|=2.009718e-14 ***
*** rep 18: DRIFT vs rep0 (must be exactly 0) ->
        chainX=2.667173e+01   chainY=0.000000e+00
        ctrlX =0.000000e+00   ctrlY =0.000000e+00 ***
REPRO RESULT: 1/19 reps drifted, 1/19 disagreed with the all-reduce control
```

Read that block carefully — it is the whole bug in five lines:

* `scale = 1.119664e+02` is the magnitude of the correct answer, and the
  chain X error is `2.667e+01`. **That is a ~24 % error**, not a
  numerical wobble.
* The association floor is `2.8e-14` (rep 0, same comparison), so the
  error is ~15 orders of magnitude above noise.
* `ctrlX` and `ctrlY` drift by **exactly `0.000000e+00`** — the
  all-reduce-only formulation of the identical sum is bit-stable across
  all repetitions. Only the `psum_scatter` chain moves.
* `chainY` is likewise exactly `0.000000e+00` on this repetition: one
  chain is corrupted while the other, in the *same executable, same
  repetition*, is bit-perfect.

The reproducer has **no project imports** — `jax`, `jax.numpy`, `numpy`
and nothing else — and generates its operands inside the `shard_map`
from `jax.lax.axis_index`, so there is no host staging, no RNG and no
file I/O anywhere in the failing path.

*Rate in the standalone form:* the first of five processes gave 1/19.
The full 5x20 standalone sweep is job 7879558; its aggregate is appended
at the end of this file when it completes. **The authoritative rate
quoted in §1 and §4.1 is from the in-project harness (200 executions),
not from this smaller standalone sweep** — the reproducer's job is to
show a maintainer the failure without our code, which it does.

## 4. Evidence

### 4.1 Rate

Both runs: 5 fresh processes x 20 repetitions = 100 executions of each
chain. "Wrong" = disagrees with a structurally independent computation of
the same quantity by many orders of magnitude above the ~1e-16 floor.

**Run A — controlled** (frozen source snapshot, byte-verified before,
during and after; see §5.4):

| process | chain X wrong | chain Y wrong |
|---|---|---|
| 0 | 0/20 | **2/20** |
| 1 | 0/20 | **1/20** |
| 2 | 0/20 | **1/20** |
| 3 | 0/20 | 0/20 |
| 4 | 0/20 | **1/20** |
| **total** | **0/100** | **5/100** |

**Run B — earlier, same configuration**: chain X 3/100, chain Y 3/100;
3 of 5 processes affected.

Combined: **11 corrupted executions in 200**, 7 of 10 process lifetimes.

### 4.2 The corrupted output is always segment 0

The output of chain X is sharded over `'x'`, so rank `(x,y)` holds block
`x`; chain Y's output is sharded over `'y'`, so rank `(x,y)` holds block
`y`. Recording per-rank `||shard||^2` and diffing against the known-good
value:

```
Run A (chain Y): ranks [0,2] = (x=0,y=0),(x=1,y=0)  ->  block y=0
Run B (chain X): ranks [0,1] = (x=0,y=0),(x=0,y=1)  ->  block x=0
```

**All 11 events, both runs, both axes, corrupt output segment 0** — the
first segment of the scatter, i.e. the one that would carry a rank's own
contribution. Never segment 1.

### 4.3 The wrong value is identical on both holders, and differs every time

Both ranks that hold the corrupted block report the *same* wrong
`||shard||^2`, and the value differs between occurrences:

```
correct  2.090585055562361e+10
observed 2.096495112140704e+10   2.097017772924163e+10
         2.201642275979959e+10   2.235211142759555e+10   2.288963577449787e+10
```

Two consequences:

* **identical on both holders excludes the following all-reduce.** In
  chain Y the `psum_scatter('y')` is followed by `psum('x')`; if the
  all-reduce were at fault the two holders would generally disagree.
  They never do. The corruption is already present in the reduce-scatter
  output and the all-reduce faithfully propagates it.
* **a different wrong value each time indicates a race**, not a
  deterministic offset or a systematic mis-association.

### 4.4 Controls

* **The other chain is EXACTLY bit-stable.** In every one of the five
  Run-A events the drift report reads
  `chainX 0.000000e+00 ... chainY <1e+02 class>`. Exactly `0.0`, not
  "small" — so this is not accumulated round-off, and the two chains
  differ only in which mesh axis carries the reduce-scatter.
* **An all-reduce-only formulation is clean in 200/200 executions.** In
  our application code the same overlap matrix is computed by a second,
  structurally different plan that shards both operand axes and needs
  only `psum` — no `psum_scatter`, no chunking. It has never been
  corrupted.
* Magnitudes of the discrepancy are 3.7e+01 to 1.4e+02 against a
  floating-point floor of 7.2e-14 for the same comparison — 15 orders of
  magnitude above noise.


### 4.5 Timing/clustering — an observation, explicitly NOT a periodicity

Occurrences are not uniformly spread over a process's repetitions, but
the pattern does **not** reproduce across jobs, so we report it as an
observation only:

| run | repetition indices of the events | candidates |
|---|---|---|
| in-project, frozen (7879540) | 11, 14, 15, 15, 17 | 1..19 |
| in-project, earlier (7879526) | 0, 1, 2, 4, 5, 9 | 0..19 |
| standalone (7879558) | 18, 18 (first two processes) | 1..19 |

Taken alone each run looks clustered — P(all 5 of 7879540 in the last 9
of 19) = 0.024; P(all 6 of 7879526 in the first 10 of 20) = 0.016 — but
**the two clusters sit in opposite halves**, which kills any fixed-phase
"fires at repetition N" explanation. The two standalone processes both
firing at repetition 18 has probability 1/19 = 0.053 of happening by
chance and was noticed *after* the fact, so it carries no evidential
weight on its own.

What this is consistent with, and all we are willing to say: the trigger
is **timing/environment sensitive within a job** (whatever else is
contending for the node and fabric during that window), not periodic in
the program. A maintainer should not expect a fixed repetition index to
reproduce it.

### 4.6 The mpi collectives backend is clean on the same program

Earlier revisions listed "no comparison against the mpi backend" as
NOT-established, because `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` refused to
create grouped communicators for a standalone driver. That refusal is
**jaxlib's, not MPI's**: `xla::cpu::MpiCollectives::CreateCommunicators`
calls `MPI_Is_thread_main()` and returns
`absl::UnknownError("MPI: Communicator requested from a thread that is not
the one MPI was initialized from")` when it is false. XLA:CPU issues the
collective from an intra-op pool worker, so the test fails for any clique
whose first use is inside a jitted program. It is a `MPI_Is_thread_main`
test, **not** a thread-*level* test — no `MPI_THREAD_*` level changes it.
Because that symbol resolves through MPItrampoline to an MPIwrapper we
build, we overrode the stub and ran the comparison.

**Result — the same frozen driver, same shapes, same 2x2 mesh, same 20
repetitions per process, only the collectives backend differing:**

| job | nodes | backend | fresh processes | executions of each chain | processes corrupted |
|---|---|---|---|---|---|
| 7880756 | A | **mpi** (`MPI_Is_thread_main` override) | 10 + 1 short | **204** | **0** |
| 7880756 | A | gloo (positive control, same allocation, interleaved) | 5 | 100 | 0 |
| 7880763 | B | **mpi** | 20 | **400** | **0** |
| 7880763 | B | gloo | 12 | 240 | **8 of 12** |
| 7880838 | A | gloo (poison test, §4.6d) | 12 | 240 | **6 of 12** |

*Job 7880756* (`COMPLETED`, 25 m 34 s, source manifest verified at job start
**and** end) interleaved the mpi and gloo processes so both legs sample the
same window of node/fabric conditions. *Job 7880763* is a deeper run on a
different node pair; the counts above are those complete at the time of
writing (it was still accumulating when this revision was written), read
from the job log on disk. The mpi denominator only grows from here.

Combined: **604 mpi executions, zero corrupted**, against a gloo leg that
corrupted **8 of its 12 process lifetimes on node pair B**. Stated rather
than over-read: 0 in 604 bounds the mpi rate at **< 0.5 %** (95 %, rule of
three); under the 5 % gloo rate the probability of zero in 604 is
`0.95^604 = 3.5e-14`. This is evidence that the mpi backend does not carry
the fault at this shape and process count. It is **not** proof of absence,
and it says nothing about other shapes, payloads or process counts (§6.3
still stands).

### 4.6a The wrong value is drawn from a SMALL DISCRETE SET, not a continuum

This revises §4.3's "a different wrong value each time indicates a race".
Node pair B corrupted **8 of its 12 gloo process lifetimes** (job 7880763,
`COMPLETED`, manifest verified start to end), and the eight events take
exactly **five** distinct values:

| deviation `|ox-cx|` | occurrences | repetition | chain |
|---|---|---|---|
| `8.475121e+01` | 3 | 0, 0, 0 | X |
| `6.381745e+01` | 2 | 5, 19 | X |
| `7.012239e+01` | 1 | — | X |
| `3.257096e+01` | 1 | 0 | X |
| `2.667173e+01` | 1 | 10 | X |

against a `2.842171e-14` floor. In *this job* every event is on chain X and
every `chainY` reading is exactly `0.000000e+00`. **Do not generalise that:**
job 7880838 on a different pair later fired chain Y twice (`2.360697e+01`,
`1.939797e+01`), so both chains are susceptible and the apparent asymmetry
here is a property of this sample, not of the bug. The two controls are
exactly `0.0` in all eight events. Three *independent processes* — fresh
allocations of the communicator, fresh gloo connections — produced the
**bit-identical** wrong deviation, and the repeats of the other two values
occurred many repetitions apart in different processes.

**The set is reproducible ACROSS JOBS AND ACROSS NODE PAIRS.** The third
value, `2.667173e+01`, is bit-identical to the deviation quoted in §3 from
job 7879558 — a different job, run fifteen hours earlier on a different
pair of compute nodes. Two independent occurrences, separated by everything
that could plausibly vary except the program and the data, produced the same
seven printed digits.

A free-running partial-sum race does not do this. A small set of reachable
wrong values, each exactly reproducible and each recurring across
independent runs, is the signature of a **buffer or ordering fault with a
few discrete outcomes** — segment 0 receiving a stale, dropped, or
un-reduced contribution — rather than of arithmetic interleaving. Each
distinct deviation should correspond to a specific "which contribution went
missing" case, which makes this checkable arithmetically against the known
operands and is, in our view, the most tractable handle in this report.

Node pair A produced **zero** events in 100 gloo executions in the same hour,
which looked like strong node-pair dependence. **That reading did not hold.**
Later the same night, job 7880838 on that same pair A fired **6 events in 240
executions**. Susceptibility therefore varies with time, not (or not only)
with the node: a null run is weak evidence of a healthy pair. A maintainer
should budget several hundred executions before concluding a configuration is
clean.

**Negative control — the grouped MPI communicators really are on the
critical path** (job 7880767, `COMPLETED`). Same program, same allocation,
one environment variable apart:

| cell | grouped-communicator override | refusal lines | result |
|---|---|---|---|
| `gateOFF` | not set | **4** (one per rank) | dies with the refusal above |
| `gateON` | set | **0** | runs, clean |

So the mpi leg is not somehow bypassing the reduce-scatter: without the
override the program cannot even build the communicator the `psum_scatter`
needs. The measured floating-point association floor is *identical* on both
backends (`|ox-cx| = 2.842171e-14`, `|oy-cy| = 2.009718e-14`, bit-for-bit),
which is what one expects when the same HLO runs and only the transport
implementation differs.

**Performance, same payload, incidental but relevant** (job 7880767): at a
1.12 GB `all_reduce` / 2.24 GB `all_gather` / 1.12 GB `reduce_scatter`, the
mpi backend took 0.83 s / 1.05 s / 0.63 s where gloo took 14.99 s / 31.11 s /
11.98 s. Most of that gap is the fabric — our mpi path is Intel MPI over
`mlx` verbs while gloo is TCP — but gloo in this jaxlib has no non-TCP
transport, so the gap is structural rather than a tuning matter.

### 4.6b The corrupted data is not a missing, doubled or partially-transferred contribution

§4.6a establishes a small set of bit-reproducible wrong values. Because the
reproducer generates every operand deterministically inside the `shard_map`
from `lax.axis_index`, each rank's exact contribution to segment 0 is
computable offline, so the natural question — *which* contribution went
missing — is answerable arithmetically with no cluster communication at all.
We did that. **No simple missing-contribution model reproduces any measured
value.**

Notation: `L_r` is rank *r*'s contribution to segment 0, with rank *r* at mesh
position `(ix, iy) = (r // 2, r % 2)` owning global k-range
`[r·K_LOC, (r+1)·K_LOC)`. Chain X's reduce-scatter has replica groups
`{0,2}` (iy=0) and `{1,3}` (iy=1); segment 0 is produced by ranks 0 and 1.
Correct output is `L0 + L1 + L2 + L3`.

**Pass 1 — whole-segment models** (job 7880798, `COMPLETED`,
`decompose_deviations.py`). Every non-empty subset of contributions missing;
own- or peer-doubled in either group; segment zeroed; segment holding the
other segment's reduced data. The calculation validates itself: our numpy
`max|control|` reproduces the printed `scale = 1.119664e+02` exactly, so the
operand reconstruction is correct and the exclusions are real. Nearest
approach to any measured value was **3 %**, most were 8-40 % away.

**Pass 2 — partial transfers** (job 7880803, `COMPLETED`,
`decompose_partial.py`). The output buffer is `(16, 99, 5004)` complex128 in
C order, so a transfer that stopped early or started late corrupts a
contiguous **prefix or suffix of the flat buffer**, not whole columns. For a
model `D` applied to a flat suffix from cut *i*, the reported number is
`max|D|` over `flat[i:]`, a non-increasing step function whose distinct
values are the running records from the end — about 15-23 per model per
direction. So each model admits only a few tens of achievable deviations,
which makes a 7-significant-digit match a sharp test rather than a
coincidence. (This is why the test is done at *record* granularity: with
7.9e6 elements, some individual element sits within 1e-6 of almost any value
in range, so an element-level test would prove nothing.) 24 models × both
directions = **823 achievable values**, checked against all six measured
deviations.

**Result: no match, at any cut, for any model.** Nearest approaches:

| measured | nearest achievable | rel. | model |
|---|---|---|---|
| `6.381745e+01` | `6.380460e+01` | 2.0e-4 | SUFFIX, `L0` missing |
| `7.012239e+01` | `7.015186e+01` | 4.2e-4 | SUFFIX, `L0` missing |
| `3.257096e+01` | `3.258106e+01` | 3.1e-4 | SUFFIX, `L0+L1` missing |
| `2.667173e+01` | `2.662042e+01` | 1.9e-3 | PREFIX, `L1` missing |
| `8.475121e+01` | `8.442564e+01` | 3.8e-3 | SUFFIX, all four missing |
| `2.360697e+01` | `2.351789e+01` | 3.8e-3 | SUFFIX, `L1` missing |

These near-misses carry **no evidential weight**: 823 candidate values spread
over a range of order `1e2` give a mean spacing near `6e1` of roughly
`1.4e-1`, i.e. ~2e-3 relative, so nearest approaches of 2e-4 to 4e-3 are what
chance produces. None comes within four orders of magnitude of the 1e-6
criterion.

**What this excludes, and what survives.** Excluded: a peer's contribution
never arriving; a rank's own contribution dropped; either counted twice; the
segment zeroed; the segment holding the other segment's data — each of these
whole, *and* each applied to any prefix or suffix of the buffer. Also
excluded a priori: a stale buffer from the previous repetition, because the
operands are bit-identical every repetition, so last repetition's buffer
holds the *correct* value and would give a deviation of exactly `0.0`.

What survives is the class in which the corrupted region does not hold *any*
linear combination of the four correct contributions **at the same indices** —
byte-level garbage, or correct data written at a wrong offset (a misaligned
or shifted copy). Combined with §4.6a's finding that the wrong values are
few, exactly reproducible, and recur across jobs and node pairs, the natural
remaining hypothesis is **a deterministic offset/alignment fault in segment
0's receive path**, not a lost or mis-ordered reduction. The obvious next
test, which we have not run, is to scan candidate shifts *s* and ask whether
`max|C0_shifted(s) − C0|` reproduces a measured value; a full scan is
7.9e6 × 7.9e6 and needs a targeted shift set (gloo chunk sizes, `K_LOC`,
`M`, `N/2`, powers of two) rather than brute force.

### 4.6c The offset scan is also negative — and the suspect code is gloo's own halving-doubling algorithm

> **THIS SECTION WAS REWRITTEN 2026-07-29. The two "facts read out of the
> binary" that opened the previous revision were both false, and the conclusion
> they supported — "the fault is XLA's, not gloo's; file against XLA/jax rather
> than gloo" — is withdrawn.** The offset scan below is unaffected and still
> stands as an exclusion; only the attribution changes. The retracted text is
> reproduced in the method-lesson box at the end of this section, because the
> way it went wrong is instructive and the record should show it.

**1. This gloo build DOES contain a reduce-scatter algorithm.** The previous
revision inferred its absence from a `strings` census, which found only
`external/gloo/gloo/allgather.cc` and `external/gloo/gloo/allreduce.cc` among
the gloo algorithm sources. **That inference was invalid.**
`gloo/reduce_scatter.h` is a **header-only template** — the whole algorithm is
defined in the header, there is no `reduce_scatter.cc`, and so the compiler
emits **no `external/gloo/gloo/*.cc` path string for it at all**. `strings`
could not have seen it under any circumstances.

The symbol table settles it directly:

```
$ nm -C .../site-packages/jaxlib/libjax_common.so | grep -c ReduceScatterHalvingDoubling
112

0000000000cc6c5a0 t gloo::ReduceScatterHalvingDoubling<std::complex<double> >::
                       ReduceScatterHalvingDoubling(std::shared_ptr<gloo::Context> const&,
                       std::vector<std::complex<double>*>, int, std::vector<int>,
                       gloo::ReductionFunction<std::complex<double> > const*)
0000000000cc6dab0 t gloo::ReduceScatterHalvingDoubling<std::complex<double> >::run()
0000000000cc6e350 t gloo::ReduceScatterHalvingDoubling<std::complex<double> >::getDistributionMap(...)
```

112 symbols, and the instantiation at `0xcc6dab0` is for
`std::complex<double>` — **exactly the dtype this report's reproducer uses**.
The algorithm is not merely present; the specific instantiation our failing
call needs is present and is what runs.

**2. The anonymous-namespace reducer belongs to a DIFFERENT communicator.**
The symbol the previous revision pointed at,

```
xla::cpu::(anonymous)::ReduceScatter<PrimitiveType>(
    ReductionKind, absl::Span<const void* const> inputs, void* output, long count)
```

is defined at **line 198 of
`xla/backends/cpu/collectives/in_process_communicator.cc`** — the
`InProcessCommunicator`, which serves single-process multi-device runs
(`--xla_force_host_platform_device_count`). It is **not on the gloo path**, it
is not reached by our 4-process run at all, and the allgather-then-local-reduce
story built on it never existed.
[Source](https://github.com/openxla/xla/blob/main/xla/backends/cpu/collectives/in_process_communicator.cc)

**3. What `GlooCommunicator::ReduceScatter` actually does.** From
[`xla/backends/cpu/collectives/gloo_communicator.cc`](https://github.com/openxla/xla/blob/main/xla/backends/cpu/collectives/gloo_communicator.cc)
(verbatim, and byte-equivalent to the code in our jaxlib — see §7b):

```cpp
Future<> GlooCommunicator::ReduceScatter(send_buffer, recv_buffer, dtype,
                                         count, reduction_kind, executor) {
  size_t chunk_bytes = count * primitive_util::ByteWidth(dtype);
  std::unique_ptr<char[]> temp(new char[chunk_bytes * context_->size]);   // 253 MB for us
  std::memcpy(temp.get(), send_buffer.opaque(), chunk_bytes * context_->size);
  switch (dtype) { ...
    case C128:
      RETURN_IF_ERROR(ReduceScatterHelper<std::complex<double>>(
          context_, reduction_kind, temp.get(), count));
  ... }
  std::memcpy(recv_buffer.opaque(), temp.get(), chunk_bytes);   // unconditionally offset 0
}

// ReduceScatterHelper:
std::vector<int> recv_elems(context->size, chunk_elems);
gloo::ReduceScatterHalvingDoubling<T> algorithm(
    context, std::vector<T*>{reinterpret_cast<T*>(buffer)},
    chunk_elems * context->size, recv_elems, reduction_function);
algorithm.run();
```

XLA's contribution is a bounce buffer and two `memcpy`s. **All of the
reduction and all of the communication is `gloo::ReduceScatterHalvingDoubling`.**
Two details of that shape are worth carrying into the filing:

* the algorithm reduces **in place** in `temp`, and allocates a further
  internal `recvBuf_` of the full payload (`gloo/reduce_scatter.h:128`), so the
  operation runs entirely out of scratch memory that no longer holds only our
  operands. This is the mechanism §4.6d needed and did not have;
* XLA `memcpy`s the result from **offset 0 of `temp`, unconditionally, on every
  rank**. That is the algorithm's own output convention, but it is also exactly
  consistent with §4.2's finding that the corruption is *always* segment 0.

**4. Where this should now be filed.** The honest answer is **both, with the
primary against XLA/jax**:

* **jax-ml/jax (primary).** That is where the reproducer runs, where the
  failing call site (`GlooCommunicator::ReduceScatter`) lives, and — decisively
  — where the *choice* to use this algorithm is made. XLA made gloo the
  **default** CPU collectives backend in February 2025
  ([jax#26264](https://github.com/jax-ml/jax/pull/26264), landing with the note
  that "multi-process CPU communication works out-of-the-box") and deleted the
  legacy opt-in flag in September 2025
  ([jax#31884](https://github.com/jax-ml/jax/pull/31884)). So every JAX user
  doing multi-process CPU today is routed, by default and without any
  documentation naming it, into an algorithm that has not been functionally
  touched since **2018-02-09**.
* **pytorch/gloo (cross-reference).** The algorithm is theirs. Its complete
  history is three commits — `ReduceScatter CPU Implementation` (2018-02-09),
  `Remove PATENTS clause` (2018-12-12), `Applying CLANGFORMAT formatting`
  (2024-10-02) — i.e. **no functional change in eight years and no bug fix
  ever**. Their README declares gloo *"feature complete and in
  maintenance-only mode"* and lists its primitives as *"a barrier, broadcast,
  and allreduce"* — **reduce-scatter is not in the list**.
  [README](https://github.com/pytorch/gloo) ·
  [docs/algorithms.md](https://github.com/pytorch/gloo/blob/main/docs/algorithms.md)

A maintainer reading only the old §4.6c would have been sent to audit the
wrong file. That is why this correction is at the front of the report as well
as here.

> **METHOD LESSON — a negative from `strings` is not evidence of absence for
> header-only C++.** The retracted inference chain was: *"the only gloo
> algorithm `.cc` paths in the binary are `allgather.cc` and `allreduce.cc`,
> therefore this gloo build has no reduce-scatter, therefore
> `GlooCommunicator::ReduceScatter` must be XLA's own code, therefore this is
> an XLA bug and not a gloo bug."* Every link followed from the one before and
> the whole thing read as airtight. It was wrong at step one, because C++
> templates defined in headers leave **no source-path string and no `.cc`
> entry** — they are emitted as instantiated symbols in the translation unit
> that uses them, and only the symbol table shows them. The check that would
> have caught it costs one command (`nm -C … | grep`) and we did not run it
> before drawing a conclusion about where a bug lived. Generalisation for the
> rest of this campaign: **`strings` can confirm presence, never absence.**
> Any claim of the form "X is not in this binary" must come from `nm`/`objdump`
> over the symbol table, not from a string census — and a claim that reassigns
> blame between two projects deserves two independent confirmations before it
> is relayed as settled.

**The scan.** *(Motivation restated after the rewrite above. This scan was
originally designed against the withdrawn pointer-arithmetic model
`inputs[j] = base + j*full_stride + my_segment*segment_size`. Its **value as an
exclusion is unchanged** — it tests, arithmetically and against the exact
operands, whether the corrupted region holds any operand read at any wrong
offset, and that question is backend-agnostic. It is if anything more relevant
now: recursive halving/doubling is precisely an algorithm whose steps are
offset-and-stride arithmetic over a shrinking buffer, so an offset slip inside
`run()` would land in this scan's search space.)* A wrong segment index or a
wrong stride yields a contribution read from the wrong column offset — the
surviving hypothesis from §4.6b, and not a model that §4.6b excluded. Job
**7880821** (`COMPLETED`, `decompose_shift.py`) tested it. Operands: each rank
individually (`L0..L3`), each replica-group sum (`L0+L2`, `L1+L3`, `L0+L1`,
`L2+L3`) and the full reduction; column offsets `d` from a targeted set of 24
(structural — 1, `M`=99, `K_LOC`=1008, `N/2`=5004 — and byte/alignment
motivated — 4 = one 64 B cache line, 256 = one 4 KiB page, powers of two);
plus flat shifts inside the `(16, 99, 5004)` buffer including 131072 = one
2 MiB huge page. The specific slip worth naming — loop index used where the
rank's own segment index belongs, so peer *j* contributes its segment *j* —
is the `d = 5004` row, and it is tested on every operand. Reconstruction
validated for the third time: `max|C0| = 1.119664e+02`, the printed `scale`.

**Statistical bar, computed rather than asserted.** 261 whole-buffer
candidates spanning `[8.2e-15, 1.566e+02]`, plus 7360 partial prefix/suffix
record values. At `rtol = 1e-6` the expected number of spurious hits is
**2.1e-4 (whole)** and **6.0e-3 (partial)** near `V = 6.4e+01`. So a hit at
that tolerance would be signal, not coincidence.

**Nothing cleared the bar. Nothing came close.** Best whole-buffer approach
over all six measured values was `6.07e-4` relative — three orders of
magnitude short — and that one is a chain-Y model matched against a chain-X
measurement, i.e. not even the right replica group. Zero partial matches.

**Combined verdict of §4.6b and §4.6c.** The corrupted region is not:
a missing contribution, a doubled contribution, a zeroed segment, the other
segment's data, a stale buffer from the previous repetition, or any operand
read at any tested wrong offset — each of these whole, *and* each applied to
any prefix or suffix. **The corrupted data is not derivable from the correct
operands by any omission or re-indexing we can construct.**

What that leaves is **memory that does not belong to this reduction**:
uninitialised, or foreign, bytes. That is a more serious class than a
misplaced offset, and we say so plainly — a consumer of this collective may
receive not merely a wrong number but the contents of some other buffer.

> **RETRACTED IN PART (§4.6d).** Two of the four rows below were falsified
> by our own poison test, and the mechanism the table argues for was itself
> excluded. Job 7880838 fired on node pair A — the pair that had given 0 in
> 100 executions — which kills the "node-pair dependence" row; and it fired
> **chain Y** twice (`2.360697e+01`, `1.939797e+01`), which kills the "chain Y
> never deviates" row. The table is kept, struck through in effect, because
> the reasoning was load-bearing for a conclusion we no longer hold and the
> record should show that. **What survives, and is strengthened, is row 2** —
> see the note after the table.

**~~The strongest argument in this report~~ (see retraction above): one
mechanism explains four observations that had looked unrelated.**
Uninitialised or recycled memory is
*deterministic given the same allocation history* — the same program makes the
same allocations in the same order, so it sees the same stale bytes. That
single property accounts, with no further assumptions, for all four of:

| observation | why uninitialised/recycled memory predicts it |
|---|---|
| the wrong value comes from a **small discrete set**, not a continuum (§4.6a) | only a few distinct stale contents are reachable at that allocation site |
| the same value is **bit-exact across independent processes**, and `2.667173e+01` recurs across jobs **fifteen hours apart on different nodes** (§4.6a) | identical allocation sequence ⇒ identical stale bytes; nothing about it is timing-dependent once the region is reached |
| **strong node-pair dependence** — pair B 8 of 12, pair A 0 of 100 executions in the same hour (§4.6) | memory state is a property of the node, not of the program |
| **chain X hit in all eight pair-B events; chain Y never deviated in twelve processes** — every `chainY` reading in the run is exactly `0.000000e+00` | the two chains allocate at different points in the sequence; only one lands on the poisoned region |

**What actually survives — and it is cleaner than what it replaces.** Row 2
is now much stronger than when written. `2.667173e+01` has been observed in
**three different jobs on three different node pairs** — 7879558 (c207-026/027),
7880763 (c209-003/004) and 7880838 (c208-009/010) — and `6.381745e+01` and
`2.360697e+01` likewise recur across pairs. Seven distinct values are now on
record (`8.475121e+01`, `7.012239e+01`, `6.381745e+01`, `3.257096e+01`,
`2.667173e+01`, `2.360697e+01`, `1.939797e+01`, `7.691618e+01`,
`7.731059e+01`, `1.021892e+04`) — ten in all, and both chains are affected. So:

> **The wrong value is a small, discrete, bit-reproducible set that is
> independent of the machine, the job and the repetition index.** It is a
> deterministic function of the program and its data, not of the node.

That is a sharper claim than the node-dependence story it replaces, and it
remains hostile to the "free-running race" reading of §4.3. It is also
what makes the residual mystery acute: the value is deterministic, yet
§4.6b/§4.6c exclude every function of the correct operands we can construct,
and §4.6d excludes uninitialised memory.

**Recommendation.** Black-box arithmetic has taken this as far as it goes; we
can exclude mechanisms but we cannot identify memory we do not own. §4.6d
runs the cheap half of the remaining work — a process-wide heap poison, which
needs no rebuild — and states its decision rule in advance. The other half is
an ASAN or debug build exercising
`xla::cpu::GlooCommunicator::ReduceScatter` and its local reducer, which we
cannot do from outside the wheel and which we are asking a maintainer for.

### 4.6d The poison test — no rebuild required, decision rule pre-registered

The obvious way to settle §4.6c's hypothesis is to poison memory and see
whether the wrong values follow. **This does not need a jaxlib rebuild, an
LD_PRELOAD shim, or a debug build**: glibc's own `MALLOC_PERTURB_` already
does precisely this poison, process-wide, from the environment.

```
MALLOC_PERTURB_=128   freshly malloc'd bytes -> 0x7F
                      0x7F7F7F7F7F7F7F7F as float64 = 1.382417e+306
                      freed bytes            -> 0x80
                      0x8080808080808080 as float64 = -2.937447e-306
```

`1.38e+306` against a legitimate scale of `1.119664e+02` is unmistakable.
Zeros were deliberately **not** used: a zero-filled region is indistinguishable
from a legitimately empty buffer, and §4.6b already excluded a near-zero
region (that is the `MISSING L0+L1+L2+L3` model, whose prefix/suffix records
match no measured value).

**Decision rule, fixed before the run:**

| observed under `MALLOC_PERTURB_=128` | conclusion |
|---|---|
| deviations become `~1e306`, `inf` or `nan` | **uninitialised heap confirmed** — the corrupted region is freshly-allocated, never-written memory |
| deviations become `\|C0\|`-shaped (`~1e2`) | **freed heap** — the freed-poison byte `0x80` is `≈2.9e-306`, i.e. numerically zero, so the region reads as empty |
| deviations unchanged (`8.475121e+01`, …) | glibc-heap uninitialised/freed memory **excluded**; the foreign data is live, written memory or a non-glibc arena |
| neither leg corrupts at all | **inconclusive** — the node pair was not susceptible that night; the interleaved CONTROL cells decide this, not the poison cells |

**Scope limit, stated up front.** `MALLOC_PERTURB_` only touches the glibc
heap. A negative result excludes *uninitialised or freed glibc-heap memory*;
it does **not** exclude foreign memory that has been legitimately written
(another live buffer in the same process), nor a pool or arena that does not
round-trip through `malloc` per use. Those would need the ASAN route.

Harness `poison_test.sbatch`: twelve fresh processes, six with the poison and
six control, **interleaved in one allocation on one node pair** so
susceptibility is established alongside the measurement. 20 repetitions each,
which covers the repetitions that fired before (0, 5, 10, 19).

**RESULT (job 7880838, `COMPLETED 0:0`, 44 m 54 s, node pair A): the
uninitialised-memory hypothesis is REFUTED.**

| | cells fired | poison-magnitude / non-finite values |
|---|---|---|
| poison (`MALLOC_PERTURB_=128`) | **3 of 6** | **0** |
| control (unset) | **3 of 6** | **0** |

Identical fire rates, and the wrong values under poison are drawn from
*exactly* the same known set as without it — `8.475121e+01` (poison_3),
`2.667173e+01` (poison_4), `3.257096e+01` (poison_5) — with not one value of
poison magnitude and not one `inf` or `nan` anywhere in 240 executions.

By the rule fixed in advance, this lands squarely on the third row:
**uninitialised or freed glibc-heap memory is excluded.** Whatever the
corrupted region contains, it is not fresh or recently-freed heap; it is
either live, written memory belonging to something else in the process, or a
buffer from an arena that does not round-trip through `malloc`. Confirming
*which* needs the inside of the process — the ASAN/debug route we are asking
a maintainer for. **Corroborated on the susceptible pair.** Job **7880837** (`COMPLETED 0:0`,
44 m 56 s, node pair B) gives the same answer: poison cells fired **4 of 6**,
control cells **5 of 6**, and again **zero** poison-magnitude or non-finite
values. Across both jobs that is **0 poison hits in 24 cells / 480
executions**, with poison and control firing at statistically
indistinguishable rates (7/12 vs 8/12). The exclusion is not marginal.

**One event on pair B is independently decisive — and it does not need any
statistics.** Control cell `ctrl_3` produced a chain-Y deviation of
**`1.021892e+04`**, which is **91× the legitimate scale** of `1.119664e+02`.
No combination of the four contributions can reach it: with
`max|L_r|` of `9.18e+01`, `2.78e+01`, `1.84e+01` and `9.75e+00`, even the
worst case `2·Σ_r max|L_r|` is about `3.0e+02` — the observed value is **34×
beyond the maximum the operands can produce**. A single measurement therefore
refutes the entire linear-combination family that §4.6b, §4.6c and §4.6e
excluded pass by pass, and it does so without appealing to tolerances,
candidate counts or spurious-hit rates.

It also says something positive: the corrupted region holds data of a
magnitude this computation never generates. That is foreign content with real
structure — not zeros, not fresh heap, not a re-indexing of our own operands.

**A mechanism that can reach it — added 2026-07-29 with the §4.6c correction.**
Until now `1.021892e+04` was the single hardest fact in this report: every
mechanism we could name was excluded, and a value 34× beyond what the operands
can produce was not merely unexplained but *unreachable* under the reading we
then held. Under the corrected §4.6c it becomes **reachable for the first
time**, and by the plainest possible route.

`gloo::ReduceScatterHalvingDoubling` is a **recursive-halving, distance-
doubling** algorithm that reduces **in place** in the caller's buffer and
allocates its own internal `recvBuf_` sized to the full payload
(`gloo/reduce_scatter.h:128`). Across `lg(P)` steps it repeatedly receives a
peer's half into scratch, reduces it against a region of the working buffer,
and halves the active extent; gloo's own documentation adds a final reordering
phase in which processes exchange data with their bit-reversed partner
*because "the blocks are not ordered in correct order"*
([docs/algorithms.md](https://github.com/pytorch/gloo/blob/main/docs/algorithms.md)).
Three consequences follow directly:

* **Partially-reduced intermediates are not bounded by `max|L_r|`.** Our
  operand bound `2·Σ_r max|L_r| ≈ 3.0e+02` constrains the *final* answer and
  any linear combination of the four contributions. It does **not** constrain
  the contents of gloo's internal scratch at an intermediate step, nor anything
  else sharing that arena. A value 34× outside the operand bound is exactly
  what "bytes from a buffer that is not this reduction's output" looks like,
  and §4.6d already proved those bytes are *live written memory*, not fresh or
  freed heap.
* **It explains determinism without appealing to node state.** The step
  schedule, the buffer sizes and the allocation order are fixed functions of
  `count` and `P`, which are identical in every repetition and every job. A
  small set of bit-reproducible wrong values recurring across three node pairs
  and three jobs (§4.6a) is what a *fixed* schedule with an occasional ordering
  or completion slip produces — and it is what defeated the "free-running
  race" reading in §4.3.
* **It is consistent with segment 0.** XLA `memcpy`s the result from offset 0
  of the working buffer unconditionally (§4.6c), and offset 0 is where the
  halving recursion converges. §4.2's "all 11 events, both runs, both axes,
  segment 0" needs no separate explanation.

**Held as a hypothesis, not a finding.** We have **not** read
`ReduceScatterHalvingDoubling::run()`'s buffer management, we have not
instrumented it, and we have not identified which buffer the foreign bytes come
from. The four exclusion passes (§4.6b, §4.6c, §4.6e) and the poison test
(§4.6d) stand exactly as measured and are what they always were: exclusions.
What has changed is that the residue they left — *live foreign memory of a
magnitude our computation never generates* — now has a named, plausible home
instead of none. Confirming it still needs the ASAN/debug build we are asking a
maintainer for; the difference is that we can now name the function to point it
at.

### 4.6e Cross-clique substitution is negative too — the exclusion is now complete

§4.6d left "a real buffer belonging to the wrong participant" as the natural
residue: deterministic, machine-independent, but not fresh heap. The specific
form that predicts is a wrong rank index in
`inputs[j] = base + j*full_stride + my_segment*segment_size`, i.e. a peer's
contribution taken from the **wrong clique**. *(That pointer expression comes
from the reading withdrawn in §4.6c and no longer describes any code we
believe runs. The **exclusion below is unaffected**: it asks whether the
corrupted region holds contribution `L_a` where `L_b` belongs, which is a
question about the data, not about which source file computes it. Under the
corrected reading the same question is still worth having answered, because a
rank mix-up in the halving/doubling step schedule would produce exactly this
signature.)* Passes 1-3 never covered it:
pass 1's "counted twice" models used only the eight WITHIN-group ordered
pairs, leaving `L0-L3`, `L3-L0`, `L1-L2`, `L2-L1` — exactly the cross-group
substitutions — untested.

Job **7880946** (`decompose_crossgroup.py`) tested all twelve ordered pairs,
whole and applied to any flat prefix or suffix. 12 whole candidates over
`[2.746e+01, 9.683e+01]` plus 402 partial records; expected spurious hits at
`rtol = 1e-6` are `2.2e-5` (whole) and `7.4e-4` (partial), so a hit would be
unambiguous.

**No match, whole or partial, for any of the seven measured values.** Nearest
whole approach `2.97e-2` relative. Reconstruction validated again
(`max|C0| = 1.119664e+02`).

**The exclusion is therefore complete across four independent passes.** The
corrupted region is not: any missing, doubled, zeroed or substituted
contribution (§4.6b); any of those applied to a prefix or suffix (§4.6b); any
column offset or byte/alignment shift of any operand (§4.6c); a stale buffer
from the previous repetition (a priori); uninitialised or freed glibc heap
(§4.6d, measured); or a contribution taken from the wrong clique (§4.6e).

What remains is genuinely narrow and, we think, the most useful thing we can
hand over: **the wrong value is deterministic and machine-independent — the
same seven values recur bit-exactly across three node pairs and three jobs —
yet it is not any function of the correct operands that can be constructed
from outside the process.** Those two facts together are hard to reconcile
without reading the implementation, which is why the remaining work is a
debug/ASAN build over `xla::cpu::GlooCommunicator::ReduceScatter` and its
local reducer rather than more black-box statistics from us.

### 4.6f Aside: no >2 GiB payload cliff in either CPU backend at P=4

We separately suspected a 32-bit byte count somewhere in the CPU collectives
path, because an unrelated 64-process run of ours hung for 3 h 21 m inside a
two-stage all-gather whose first stage moved 1,442,840,576 B and completed
and whose second stage moved 11,005,853,696 B and did not — with `2^31 =
2,147,483,648` between them. A payload sweep (job 7880767, `COMPLETED`;
job 7880769, `COMPLETED`) does not support that reading at `P=4`:

| primitive | largest buffer | mpi | gloo |
|---|---|---|---|
| all_reduce | 0.894 GiB | 0.78 s | — |
| all_gather | 1.788 GiB | 0.90 s | 15.0 s |
| all_reduce | 1.043 GiB (`>= 2^31`) | 0.96 s | 14.99 s |
| all_gather | 2.086 GiB (`>= 2^31`) | 1.10 s | 31.11 s |
| reduce_scatter | 1.043 GiB (`>= 2^31`) | 0.72 s | 11.98 s |
| all_reduce / reduce_scatter | 2.086 / 2.086 GiB | 1.83 / 1.31 s | 30.0 / 24.0 s |
| all_gather | 4.172 GiB | 2.07 s | 60.1 s |
| all_reduce / reduce_scatter | **5.125 GiB** | 4.11 / 3.14 s | — |
| all_gather | **10.250 GiB = 11,005,853,696 B**, the exact hung size | *(not measured — our probe process was OOM-killed before the call)* | **completed, 105.09 s** |

The decisive row is the last one: **gloo moved exactly the byte count that
hung, and completed.** So the size alone is not the mechanism, and a plain
32-bit **byte** count in the shared XLA:CPU collectives path is refuted. The mpi
backend is verified past `2^31` in all three primitives up to 5.125 GiB.
What we could *not* test is the mpi backend at 10.25 GiB — our probe needed
more memory than the node allowed and was killed before the collective
started, so that one square is blank rather than green. The original hang
also differed in process count (64 vs 4) and replica-group width (8 vs 2).

**A DIFFERENT 32-bit hazard that this section does NOT cover — found
2026-07-29, and it is an element count, not a byte count.**
`gloo::ReduceScatterHalvingDoubling`'s constructor takes the element count as
**`const int`** (`gloo/reduce_scatter.h:115`), and the per-rank receive counts
as `std::vector<int>`. XLA passes it `chunk_elems * context_->size` and
`std::vector<int> recv_elems(context_->size, chunk_elems)` (§4.6c). So the
reduce-scatter *element* count is 32-bit on the gloo path regardless of what
the byte counts do.

This is **orthogonal to the sweep above**: §4.6f refutes a 32-bit *byte* count,
and does so with an `all_gather` — a primitive that does not go through this
constructor at all. Nothing in the sweep tests a 32-bit element count in
reduce-scatter, so §4.6f must not be read as covering it.

**Where it would bind for us, and why we are currently safe.** Our failing call
is `count = 16 × 99 × 5004 = 7,926,336` elements per chunk with `P_group = 2`,
so the constructor receives `15,852,672` — three orders of magnitude inside
`INT_MAX = 2,147,483,647`. **This cannot be our corruption.** It would first
bind at ~`2.1e9` elements in a group, i.e. about **32 GiB** of complex128 in a
single reduce-scatter input, or **8 GiB** of float32 — reachable for a large
`P_group` at scale but far outside anything measured here. Worth one line in
the upstream issue as an adjacent defect, and worth remembering if LORRAX ever
pushes the reduce-scatter chain to those sizes; **not** offered as a candidate
explanation for anything in this report.

### 4.7 Methodology note: a corruption on repetition 0 inverts the counter

One of the two gloo events in job 7880763 landed on **repetition 0**, the
run's own reference. The DRIFT detector then reports `19/19 reps drifted`
(every later, correct repetition differs from the bad reference) for what is
**one** corrupted execution, and the CONTROL counter reports `0/19` because
it skips the reference repetition. The event is still visible — the printed
`rep 0:` line reads `|ox-cx| = 8.475121e+01` against a `2.842171e-14` floor —
but anyone re-running this reproducer should read the `rep 0` line and not
only the summary counters. Rates quoted elsewhere in this report are
unaffected: no rep-0 event occurred in the runs they are computed from.

## 5. Ruled out

1. **The consuming kernel.** A gate that drives the identical downstream
   contraction with a transfer matrix that never touches the collective
   returns `3.037e-16` bit-identically at 4, 16, 64 and 144 processes.
2. **Loop/chunking logic in our code.** Sweeping the chunk size over
   {1,7,11,16,25} chunks against a dense NumPy reference on 4 *emulated*
   devices in one process gives bit-identical results every time
   (`5.111e-15`). Single-process, no real collectives: clean.
3. **Our algebra.** Two structurally independent implementations of the
   same matrix agree to `3.8e-16 ... 1.0e-15` on the executions that are
   not corrupted.
4. **Source provenance.** Run A executed from a frozen source snapshot
   whose manifest was verified at job start, mid-run and after the run.
   Every `.pyc` in the snapshot was parsed (PEP 552) and its embedded
   source mtime+size checked against the snapshot's own `.py`: 11/11
   match, all timestamp-validated (`flags=0`), none hash-unchecked. The
   bytecode executed was compiled from the verified sources.
5. **Round-off.** The unaffected chain drifts by exactly `0.0`, and the
   affected chain by ~1e+02 on a quantity of order 1e+05. There is no
   round-off interpretation of either number.
6. **Determinism/repeatability of the operands.** Operands are generated
   from mesh coordinates by integer hashing; no RNG, no host staging, no
   file I/O. Repetition N and repetition 0 have bit-identical inputs by
   construction.

## 6. NOT established

1. **No axis preference is claimed.** Run A is chain Y 5 / chain X 0;
   Run B is 3 / 3. Combined 8 vs 3 of 11 events; under a fair split
   P(>=8 of 11 on one side) is ~23 % two-tailed, so the split is not
   statistically distinguishable from chance. The two runs also used
   different physical nodes. What *is* consistent across all 11 events is
   the plan (reduce-scatter), and the segment (0).
2. **No root cause, but the search space is now much smaller (§4.6b, §4.6c).**
   We have not instrumented the code, and we have no debug build. What we
   have done is exclude, arithmetically and against the exact operands,
   every missing/doubled/zeroed/substituted/mis-offset contribution model —
   whole and partial. We therefore do offer a hypothesis where earlier
   revisions offered none: **live foreign memory reaching the output** from
   inside `gloo::ReduceScatterHalvingDoubling<std::complex<double>>::run()`,
   the in-place recursive-halving algorithm that XLA's
   `GlooCommunicator::ReduceScatter` selects and drives (§4.6c, §4.6d).
   It is a hypothesis, not a finding: we have not read that function's buffer
   management and we cannot identify memory we do not own from the outside.
   **Corrected 2026-07-29** — earlier revisions of this item asserted the
   opposite attribution ("XLA's own CPU reduce-scatter; this gloo build has no
   reduce-scatter algorithm"), which was false; see the rewritten §4.6c and its
   method-lesson box.
3. **Only one shape, one payload, one mesh.** Everything here is
   `P=4` on a `2x2` mesh with a 253 MB complex128 reduce-scatter input.
   We have not swept payload size, mesh shape, process count, or dtype.
   Our larger meshes (16/64/144) ran a *different* code path that uses
   the same primitive but was not instrumented with the exact-zero
   detector, so **we cannot say whether the bug is present at other P** —
   only that we did not detect it there.
4. ~~**Not compared against the mpi collectives backend.**~~ **RESOLVED —
   see §4.6.** The comparison has been run. `mpi` is clean in 204/204
   executions of the identical program (bound: rate < 1.5 % at 95 %), with
   a gloo positive control in the same allocation and a negative control
   proving the grouped MPI communicators are genuinely on the critical
   path. What remains NOT established is anything beyond this shape: the
   mpi leg was measured only at `P=4`, `2x2`, 253 MB, so it inherits
   item 3's caveats in full.
5. **Only partially tested on other gloo transports.** Two interface
   configurations have now produced the corruption: the in-project harness
   with an explicit `interface="ib0"` (§4.1) and the standalone reproducer
   on jax's default interface selection (§3, §4.6). We have **not**
   positively identified which NIC the default selection binds; setting
   `JAX_COORDINATOR_ADDRESS` to the head node's InfiniBand IPv4 did not
   change the standalone leg's throughput (206-303 s per 20-repetition
   process, against 203 s with the coordinator on the management-network
   address), which suggests gloo picks its device from the local hostname
   rather than the coordinator route. `GLOO_SOCKET_IFNAME` is inert here
   (§2). We have not tried non-TCP fabrics; this jaxlib's gloo has no
   non-TCP transport to try.

## 7. Impact and workaround

The failure is silent and produces plausible values, so any consumer
without an independent invariant will accept it. Our mitigations, in
order of value:

0. **Move off gloo.** With §4.6 in hand this is our primary mitigation and
   supersedes the three below for any consumer that can use
   `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`. It requires working around the
   `MPI_Is_thread_main` refusal described in §4.6 for any program whose
   grouped collectives are first issued from inside a jitted region — either
   by patching the MPI wrapper, or (better, and needing no patched dependency)
   by creating each mesh-axis communicator once from the main thread at
   start-up, per `jax_threadmain_alternatives.md` route 1b. See §7b.3 for that
   pattern's upstream standing.
   **Do not substitute a jax upgrade for this** — 0.11.0 ships the identical
   gloo commit and byte-equivalent reduce-scatter code (§7b.1).
1. **Prefer an all-reduce formulation** where the algorithm permits it.
   Ours did — resharding both operand axes replaced a 4-collective
   reduce-scatter chain with one `psum` per output, and is both faster
   and clean in 200/200.
2. **Carry an invariant the collective cannot fake.** For a congruence
   `T W T^H` with Hermitian `W`, the result is Hermitian for *any* `T`,
   so a Hermiticity check tests only the machinery and costs one reduction.
3. **Compare two structurally different collective patterns** computing
   the same quantity, at a tolerance well above the association floor.


## 7b. Upstream status — three facts that change the available options

*Added 2026-07-29 from a full prior-art sweep of jax-ml/jax, openxla/xla and
pytorch/gloo (issues and PRs, open and closed). The complete sweep, including
every candidate report judged match / partial / unrelated, is
`wk_REL/docs/upstream_prior_art.md`. Headline of that sweep: **no upstream report
matches this corruption. It is novel and should be filed.***

**1. Upgrading jax fixes nothing, and must NOT be presented as a mitigation.**
jax/jaxlib 0.9.1 shipped 2026-03-02; the current release is **0.11.0
(2026-07-16)**, five releases later. It makes no difference here:

| | jax 0.9.1 | jax 0.11.0 |
|---|---|---|
| XLA pin (`third_party/xla/revision.bzl`) | `3cc8846c1005…` | `131bf41acb46…` |
| gloo pin (`third_party/gloo/workspace.bzl`) | `54cbae0d3a67…` | **`54cbae0d3a67…` — identical** |
| `ReduceScatterHelper` | — | **byte-identical** |
| `GlooCommunicator::ReduceScatter` | — | differs only by `TF_RETURN_IF_ERROR` → `RETURN_IF_ERROR` (NFC, 2026-05-21) |

Both releases pin the **same gloo commit**, so
`ReduceScatterHalvingDoubling::run()` is literally the same code. The
[jax CHANGELOG](https://github.com/jax-ml/jax/blob/main/CHANGELOG.md) entries
for 0.9.2, 0.10.0, 0.10.1, 0.10.2 and 0.11.0 mention gloo, MPI, CPU collectives
and reduce-scatter **zero times**. Upgrade for other reasons if you like; it is
not a fix, and a maintainer cannot close this as "already fixed upstream".

For the same reason, XLA's CPU collectives have **not** been rewritten or
deprecated: `xla/backends/cpu/collectives/` has taken seven commits in all of
2026, every one of them mechanical (license headers, an ABSL-macro NFC pass,
two automated code changes, a `cpu_cliques_test` disable/re-enable pair, and a
rendezvous change).

**2. The `MPI_Is_thread_main` refusal (§4.6, and `jax_threadmain_alternatives.md`)
is a real, live, unfixed upstream gap — file it.**
[openxla/xla#16430](https://github.com/openxla/xla/issues/16430) quotes our
error string verbatim and is the **only** occurrence of that string anywhere on
GitHub. But it is *not* our bug and it did **not** fix ours: its trigger is
multiple devices in one process (`--xla_force_host_platform_device_count=2`),
whereas ours is one device per process with the collective issued from an
intra-op pool worker. It was closed 2024-09-03 as `completed` by fixing the
**gloo** path only
([openxla/xla#16640](https://github.com/openxla/xla/pull/16640)); on the MPI
side the maintainer wrote *"to make it work with MPI we'd need to implement
some sort of hierarchical collectives … that's more work"*, and never returned
to it. Confirmed against `openxla/xla@main` today: the guard, the
`MPI_THREAD_FUNNELED` request and the unread `provided` are all still there
verbatim
([mpi_collectives.cc](https://github.com/openxla/xla/blob/main/xla/backends/cpu/collectives/mpi_collectives.cc)),
and `git log` on that file returns **no commits at all since 2025-06-01**. The
operations still carry no thread checks. A **new** issue is warranted,
cross-referencing #16430 as related-but-distinct; the draft in
`jax_threadmain_alternatives.md` §4.2 is accurate against current `main`.

**3. Our clique warm-up is unsanctioned, but it is what upstream itself says it
wants.** There is no public API, documentation or example anywhere in jax or
XLA for pre-creating CPU communicators — the pattern works only because
`AcquireCommunicator`'s process-global cache is keyed on the participating
device set alone, an internal detail with no stability promise. However,
[`cpu_cliques.cc`](https://github.com/openxla/xla/blob/main/xla/backends/cpu/collectives/cpu_cliques.cc)
carries this immediately above `AcquireCommunicator` (line 118):

```cpp
// TODO(b/380457503): Consider switching to a lockable CPU clique model similar
// to GPU cliques, and creating all communicators upfront.
```

So route 1b is aligned with the maintainers' own stated direction, and
**b/380457503 is the reference to cite** when asking upstream either to fix the
guard or to expose a supported upfront-creation entry point. Until one of those
lands, keep the `TF_CPP_VMODULE=cpu_cliques=3` check in the belt: if the
lockable model is adopted, the cache-key behaviour route 1b depends on could
change.

## 8. Files accompanying this report

| file | what |
|---|---|
| `gloo_psum_scatter_repro.py` | standalone reproducer, no project imports |
| `zproj_repro.sbatch` | the SLURM harness that runs it 5x20 on 2 nodes |
| `zproj_provoke.sbatch` | the fuller in-project harness that first found it |
| `zproj_prov_<jobid>_<n>/out.log` | raw per-process logs incl. per-rank shard checksums |
| `mpi_psum_scatter_repro.sbatch` | the mpi-vs-gloo harness of §4.6 (interleaved legs, one allocation) |
| `mpi_psum_scatter_negctrl.sbatch` | the grouped-communicator negative control of §4.6 + the payload sweep of §4.6f |
| `xla_cpu_collective_payload_probe.py` | standalone payload probe (§4.6f), one collective per jitted function |
| `payload_exactsize.sbatch` | the 5.125 / 10.250 GiB cells of §4.6f |
| `decompose_deviations.py` | §4.6b pass 1, whole-segment models (offline numpy) |
| `decompose_partial.py` | §4.6b pass 2, partial prefix/suffix transfers (offline numpy) |
| `decompose_shift.py` | §4.6c pass 3, column-offset and byte/alignment shift scan (offline numpy) |
| `poison_test.sbatch` | §4.6d `MALLOC_PERTURB_` poison test, 6 poison + 6 control interleaved |
| `poison_verdict.sh` | applies §4.6d's pre-registered decision rule to the poison logs |
| `decompose_crossgroup.py` | §4.6e pass 4, all 12 ordered cross-clique substitutions |
| `upstream_prior_art.md` | §7b — full prior-art sweep of jax-ml/jax, openxla/xla and pytorch/gloo; every candidate report judged; verdict on what to file |

Job ids on our system, for our own traceability: 7879491 (first
sighting), 7879496 (chunk-loop exoneration), 7879526 (Run B), 7879540
(Run A, frozen snapshot), 7879558 (standalone reproducer), 7880756 (§4.6
mpi-vs-gloo, interleaved, node pair A), 7880763 (§4.6 deeper run, node pair
B), 7880767 (§4.6 negative control + §4.6f payload sweep), 7880769 (§4.6f
exact-size cells), 7880798 / 7880803 (§4.6b decomposition, offline), 7880821 (§4.6c offset
scan, offline), 7880838 / 7880837 (§4.6d poison test, node pairs A / B),
7880946 (§4.6e cross-clique, offline).

## 9. Contact / caveat

We are a downstream user, not gloo or XLA developers. Everything in §4
is measured; §5 lists what we excluded and how; §6 lists what we did not
establish. If any of §6 matters for triage we are able to run further
experiments on this cluster.

---

## 10. Standalone reproducer aggregate (job 7879558, appended automatically)

```
  REPRO RESULT: 1/19 reps drifted, 1/19 disagreed with the all-reduce control
  REPRO RESULT: 1/19 reps drifted, 1/19 disagreed with the all-reduce control
  REPRO RESULT: 0/19 reps drifted, 0/19 disagreed with the all-reduce control
  REPRO RESULT: 0/19 reps drifted, 0/19 disagreed with the all-reduce control
  REPRO RESULT: 0/19 reps drifted, 0/19 disagreed with the all-reduce control
TOTAL: 2/95 reps drifted (2.1%), 2 control disagreements; 2 of 5 processes affected

repetition-index histogram (count, rep):
      2 18
```

      *** rep 18: DRIFT vs rep0 (must be exactly 0) -> chainX=2.667173e+01  chainY=0.000000e+00  ctrlX=0.000000e+00  ctrlY=0.000000e+00 ***
      *** rep 18: DRIFT vs rep0 (must be exactly 0) -> chainX=0.000000e+00  chainY=2.360697e+01  ctrlX=0.000000e+00  ctrlY=0.000000e+00 ***

Events span 1 distinct repetition indices across 2 occurrences.
This is consistent with §4.5: the rep-18 coincidence in the first two
processes was chance, and the repetition index is NOT predictive.
§4.5 stands as written.
