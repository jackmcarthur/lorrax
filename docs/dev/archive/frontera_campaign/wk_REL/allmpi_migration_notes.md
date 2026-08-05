# All-MPI migration — retiring gloo as a LORRAX collectives transport

2026-07-29, owner-directed. Repo `/work2/08271/jackmc/frontera/lorrax` at
`6c7feb0` (branch `fix/zq-band-gather-device-invariance`). **Nothing is
committed.** Working dir for evidence: `wk_REL/allmpi/`.

Goal, in the owner's framing: *as few references to gloo as possible without
losing significant performance; all-or-nothing over a jumbled carve-out.*

Three phases, strictly ordered. Phase 3 (the deletions) is gated on 1 and 2.

---

## 0. Housekeeping finding — the live worktree is SHARED and was being edited

While this workstream was running, another workstream wrote
`src/common/sanity.py` (20:05:35) and `src/solvers/lanczos.py` (20:07:15) into
`/work2/08271/jackmc/frontera/lorrax`. The first snapshot attempt
(`srcpin_snapshot_v2`, which tars the worktree) captured those in-flight edits
and attributed them to `6c7feb0` — its own PROVENANCE line reads
`uncommitted: ... M src/common/sanity.py; M src/solvers/lanczos.py`.

That snapshot was **discarded**. The Phase 2 snapshot is built with
`git archive 6c7feb0 src`, which cannot race a concurrent editor:

```
SRCSNAP = wk_REL/srcsnap_allmpi_git_20260729_200809_6c7feb0
          345 files hashed (343 src/ + PROVENANCE.txt + srcpin_resolve.sh)
          file list md5-equal to `git ls-tree -r --name-only 6c7feb0 src`
          both concurrently-edited files verified == the commit, != the worktree
```

**Recommendation for `srcpin_resolve.sh`:** `srcpin_snapshot`/`_v2` should
either refuse when `git status --porcelain src/` is non-empty, or take their
content from `git archive <rev>` rather than from the worktree. As written they
faithfully record the contamination in PROVENANCE.txt and then proceed, which
depends on a human reading that line.

---

## PHASE 1 — the MPI wrapper gets a home in version control. **COMPLETE, and reframed as an INTERIM STOPGAP.**

### Owner objection, accepted (mid-task): a customized dependency is a liability

> *"you say it's one wrapper symbol... but a customized dependency is really
> something we should want to avoid."*

Correct, and it changes the standing of this work rather than its content. A
locally-patched MPI ABI shim is something every user on every machine must
build, version and understand; shipping a released scientific package that
depends on one is a liability, not a detail.

What survives the objection untouched: **gloo must go** (its reduce-scatter
silently corrupts ~5% of executions — that case rests on gloo's own record),
and **`impl=mpi` is the destination**. What the objection puts back in play is
only *how BSE gets there*. Three routes are under assessment by a jax-internals
workstream, any of which deletes override (2):

| route | status |
|---|---|
| **(A) patch the MPIwrapper** — this phase | works today; now measured at scale (Phase 2); **an unwanted dependency** |
| **(B) a jax/XLA knob** forcing the sequential thunk executor | under assessment. **One negative already measured by this workstream and handed over:** this jaxlib's complete `set_xla_cpu_*` DebugOptions list contains nothing that forces sequential thunk execution, and `jax_cpu_enable_async_dispatch=0` is *not* the lever (a probe passed with it both on and off) |
| **(C) restructure BSE** so its grouped collectives are issued from the main thread | under assessment. Its matvec payload is only 0.41 MB, which permits options Σ (5-41 MB/instruction) cannot afford |
| **(D) upstream fix** — `MPI_Is_thread_main` is STRICTER than `MPI_THREAD_MULTIPLE` requires, so jaxlib's guard is arguably just wrong | under assessment; this is the outcome that lets everyone delete the workaround |

**The two overrides retire SEPARATELY — this matters and is easy to get
wrong.** B/C/D remove override (2) (`MPI_Is_thread_main`, communicator
*creation*) only. Override (1) (the `MPI_THREAD_MULTIPLE` upgrade) is about
concurrent MPI *progress during execution* — h5py collective I/O on the Python
main thread against XLA's collective thunks on pool workers — which none of
B, C or D changes. It retires only if jaxlib stops requesting
`MPI_THREAD_FUNNELED`, i.e. a second, independent upstream ask. Until then an
**unpatched** wrapper on a multi-node `impl=mpi` run is the ~29% crash regime
regardless of what happens to override (2).

The exit condition is now written into the artifacts themselves, not just here:
a STATUS section at the top of `docs/dev/mpi_collectives.md`, a STATUS block in
the `build_mpiwrapper.sh` header, and the `LORRAX_MPI_FORCE_THREAD_MAIN` row in
`env_vars.md`.

The claim being made for Phase 1 is now the narrow one: **as long as the
wrapper is needed, a pinned, reproducible, machine-code-verified in-tree recipe
is strictly better than an untracked binary in a scratch directory.** If B, C
or D lands, this phase becomes documentation of a solved problem — which is a
good outcome, and the artifact is written so that deleting it is a small,
obvious edit.


`LORRAX_MPI_FORCE_THREAD_MAIN` previously existed only as
`wk_REL/MPIwrapper_thrmain/src/mpiwrapper.cxx` (an untracked copy of upstream)
and `wk_REL/mpiw_thrmain_install/lib64/libmpiwrapper.so`
(sha256 `3e8ee475…`). All-MPI could not be a releasable state while it depended
on an untracked binary.

### What landed (all new files, nothing modified)

| file | what |
|---|---|
| `config/frontera/build_mpiwrapper.sh` | build recipe, `build_ffi*.sh` conventions |
| `config/frontera/mpiwrapper/lorrax_thread.patch` | the ENTIRE LORRAX delta from upstream, 81 lines |
| `docs/dev/mpi_collectives.md` | what the overrides do, why, and the launch recipe |
| `docs/dev/env_vars.md` | new `LORRAX_MPI_FORCE_THREAD_MAIN` row; corrected `MPITRAMPOLINE_LIB` and `LORRAX_MPI_FINALIZE_FIX` rows |

### The delta from upstream is exactly one hunk

Upstream is `eschnett/MPIwrapper` v2.11.1 at
`966f4231c96153a08295fc7d0bcbd65e916a73fd`. The pristine checkout at
`wk_AP/MPIwrapper` is byte-identical to `wk_REL/MPIwrapper_thrmain` **except**
`src/mpiwrapper.cxx`, and that difference is a single 67-line addition. So the
right thing to version-control is a patch, not a vendored fork; MPIwrapper is
external source under its own licence and the script fetches the pinned commit.

Verified: applying `lorrax_thread.patch` to a fresh pristine tree reproduces
`wk_REL/MPIwrapper_thrmain/src/mpiwrapper.cxx` **byte-for-byte**.

### The recipe was RUN, not just written

```
LORRAX_ROOT=… LORRAX_MPIWRAPPER_STAGE=$WORK/lorrax_mpiwrapper \
  config/frontera/build_mpiwrapper.sh --fresh
```

clones upstream at the pin, applies the patch, cmake-builds against Intel MPI
2020.4 on a login node, installs, and verifies. Result:

* artifact `$WORK/lorrax_mpiwrapper/install/lib64/libmpiwrapper.so`,
  sha256 `d0113a78…`
* **`.text` is BIT-IDENTICAL to the reference `3e8ee475…` binary.**
  Section-by-section, the only differences in the whole file are
  `.note.gnu.build-id`, `.note.xalt.info` (TACC's job-tracking note, which
  embeds the build CWD — 1052 vs 1056 bytes) and the `.dynamic` offsets that
  4-byte shift induces. `.text`, `.rodata`, `.data`, `.data.rel.ro`,
  `.dynsym`, `.dynstr`, `.eh_frame` all match exactly.

So the in-tree recipe demonstrably produces the same code as the out-of-tree
binary every measurement to date was made with. Phase 2 runs against the
recipe's artifact, not the untracked one.

### Verification is at the MACHINE-CODE level, deliberately

A wrapper that silently grants `MPI_THREAD_FUNNELED` builds, loads and runs
exactly like a good one, and puts you back in the ~29% AS.4b crash regime with
no signal. Source-level or symbol-level checks do not catch that (both
overrides are file-static and gcc inlines them, so `nm` finds no function
symbol on a perfectly good build). The script therefore disassembles the two
MPIABI entry points:

```
00000000000252f0 <MPIABI_Init_thread>:
   252f0:  ba 03 00 00 00   mov  $0x3,%edx        ; required := MPI_THREAD_MULTIPLE
   252f5:  e9 76 a2 ff ff   jmpq <PMPI_Init_thread@plt>

0000000000025310 <MPIABI_Is_thread_main>:
   ... getenv("LORRAX_MPI_FORCE_THREAD_MAIN") ... movl $0x1,(%rbx)   ; gate ON
   ... jmpq <PMPI_Is_thread_main@plt>                                ; gate OFF
```

and asserts both shapes. `LORRAX_MPIWRAPPER_REFERENCE_SO=<path>` additionally
compares `.text` against a known-good build.

### Instrument defect hit and fixed while writing this

The first verification block used `strings "$SO" | grep -q PATTERN` under
`set -o pipefail`. `grep -q` exits at the first match, `strings` dies of
SIGPIPE, pipefail propagates 141 — and the script declared a **good** binary
broken. This is the identical trap already documented in
`config/frontera/build_ffi_host.sh`'s symbol-check block. Every such pipeline
in the new script now captures its producer's output into a variable first, and
the hazard is called out in a comment so the next author does not re-add it.

---

## PHASE 2 — AS.4c-style rep ledger AT SCALE. **IN FLIGHT.**

### What has to be shown

The `MPI_Is_thread_main` override is verified at **P=4 / 2 nodes only**
(jobs 7879697 / 7879702, 5/5 green). It deliberately lets `MPI_Comm_split` run
on an intra-op pool worker **concurrently with main-thread MPI** — a new
concurrency exposure layered on the one AS.4b characterised:

> AS.4b base rate, no fix: **4 failures / 14 runs (~29%)** at P=16 x 8 nodes —
> 3 segfaults + 1 hang, provider-independent, every failure at the ζ-write /
> `V_q` boundary where h5py collective MPI-IO on the Python main thread
> overlaps XLA's collectives thread. P=4 single-node: 0 failures ever (shm
> netmod). AS.4c's THREAD_MULTIPLE wrapper took it to 5/5 — but 5/5 alone is
> p≈0.31 under a 21-29% base rate, so that certification rested on the
> mechanism, not on the count.

So P=4 evidence is structurally incapable of clearing this class: the class has
never fired at P=4. The ledger must be at P=16 x 8 nodes (where it *was*
measured) and at P=64.

### Harness

`wk_REL/allmpi/` — `ledger_p16.sbatch`, `ledger_p64.sbatch`, `inner_gw.sh`,
`inner_bse.sh`. Inner scripts are derived from `wk_AS/as_gw_inner.sh` (the
harness that measured AS.4b) and `wk_REL/harness/bse_as7_inner.sh`, changed only to
read the frozen snapshot, export `PYTHONDONTWRITEBYTECODE=1`, and pass
`LORRAX_MPI_FORCE_THREAD_MAIN` through.

| | P=16 job 7880801 | P=64 job 7880802 |
|---|---|---|
| layout | 8 nodes x 2 ranks, `normal`, 3 h | 32 nodes x 2 ranks, `normal`, 4 h |
| GW deck | 785c `run_800c`, `restart=false`, `slab_io=h5py_allgather` — the AS.4b deck, node count and stage boundary | 4962c `run_AQ_c4962_p64_gloo` (~554 s measured) |
| BSE deck | 785c TDA Lanczos (the 7879458/7879463 deck) | N_mu=10015, 40 Lanczos iters |
| control | one gloo/ib0 GW + one gloo/ib0 BSE, same allocation | one gloo/ib0 GW |
| cells | `mpi_gw_<n>`, `mpi_bse_<n>` interleaved, up to 12 reps each | `mpi_gw_<n>a/b`, `mpi_bse_<n>` cycled until wall headroom |

A cell FAILS on rc≠0 (segfault → 139), on timeout (rc=124 — AS.4b counted its
one hang as a failure), or if the communicator refusal string appears. Each row
records rc, wall, segv count, hang flag, refusal count. GW cells also emit
comment-stripped md5s of `eqp0/eqp1/eqp_g0w0/sigma_diag` and BSE cells emit the
eigenvalues, so a **silent behaviour change** is visible alongside the crash
ledger rather than needing a separate gate.

Both jobs verified their source pin at START. Wrapper on the path is the
Phase-1 recipe artifact `d0113a78…`.

### Harness defects found and fixed BEFORE the ledger was believed

Three, all of which would have produced a wrong ledger. Recorded because two
of them are re-treads of hazards this campaign has already documented once.

1. **`grep -c` prints 0 and exits 1.** `segv=$(grep -ac PAT $LOG || echo 0)`
   therefore yields the two-line value `"0\n0"`, which breaks the TSV row
   *and* makes `[ "$refusal" = 0 ]` false — so the scorer marked cells FAIL at
   `rc=0`. Fixed to `|| true`. (Cousin of the `strings | grep -q` + `pipefail`
   SIGPIPE trap in Phase 1, and of the same trap already documented in
   `build_ffi_host.sh`.)
2. **A missing `LORRAX_FFI_HOST_SO` export**, not a transport result. The
   first P=64 attempt (job 7880802) failed every cell at 22-74 s in
   `fit_zeta` → `_resolve_linalg_backend('eigh','distributed')`. **The gloo
   control failed identically**, which is what identified it as a harness
   omission rather than evidence about the override — the value of keeping a
   same-allocation control even on a leg where you "know" the answer.
3. **The harness was not frozen.** `srun` re-reads the inner script from disk
   on every cell, so editing `inner_gw.sh` while job 7880801 was in flight
   would have changed the experiment mid-ledger, after the job had already
   logged the script's sha256. Job 7880801 was cancelled and both sbatch
   scripts now copy the inner scripts into a job-private `harness_<jobid>/`
   directory at start and run from there. Same failure mode as an unpinned
   source tree, one layer up. (The same mistake was made and corrected twice
   more: a `pytest` job whose tree I edited mid-run, and a Phase 3 gate job
   whose snapshot I invalidated by a later edit. Both were re-run.)

4. **Two concurrent ledger jobs shared one run directory — the worst of the
   four, because it manufactured a plausible FAILURE.** `run_gw`/`run_bse`
   used fixed deck paths (`$D/run_AMP16_gw`, `.../bse`) and each cell begins
   with `rm -rf $RUN`. Submitting a second P=16 ledger (7880832) to extend the
   rep count while the first (7880810) was still running meant the two jobs
   deleted each other's deck mid-cell and each copied the other's `gw.log`.

   Two cells then scored **FAIL** (`mpi_gw_7` rc=1/41 s, `mpi_gw_8` rc=1/42 s)
   with no segfault, no hang and no MPI refusal — exactly the shape one might
   report as "an unexplained intermittent failure at P=16", i.e. the AS.4b
   class reappearing. It was not. The evidence that settles it:

   * `sacct`: 7880810 started **20:19:05** on `c134-*`; 7880832 started
     **20:29:15** on `c106-*`.
   * 7880810's `mpi_gw_4.log` is **truncated to 1432 B at 20:29:39** — the
     minute 7880832 came up.
   * 7880810's `mpi_gw_7.log` records `host=c106-111`, a node **7880832**
     owns and 7880810 does not.
   * 7880832's `ctl_gloo_gw.log` — a *gloo* cell — contains
     `Gloo interface pin: no-op (CPU collectives implementation is 'mpi')`,
     i.e. it captured the other job's *mpi* cell.
   * Four of the swapped logs are byte-for-byte the same size (31453 B).

   **Both jobs were cancelled and every cell they produced is VOID**
   (moved to `wk_REL/allmpi/voided/`), including the eight that had passed
   before the collision — a ledger that needs an mtime argument to say which
   of its own rows to believe is not a ledger. Deck paths are now
   job-scoped (`run_AMP16_${SLURM_JOB_ID}_gw`) in all three sbatch scripts,
   and the ledger was re-run as a SINGLE job.

   The general lesson, which is the same one as (3) one level further out:
   **anything a rep ledger writes must be keyed by job id.** A frozen source
   snapshot and a frozen harness do not help if two jobs share a scratch
   directory.

### RESULT — P=16 x 8 nodes: the class is CLEARED. **90 cells, 0 failures.**

Job **7880843**, `sacct` COMPLETED, elapsed 1 h 32 m 39 s, source pin
**verified immutable at job START and END**.

| group | runs | fail | segv | hang | refusal | 95% upper bound | P(0 fails \| p=0.29) |
|---|---|---|---|---|---|---|---|
| mpi+force, GW 785c  | 45 | **0** | 0 | 0 | 0 | 6.4% | |
| mpi+force, BSE 785c | 45 | **0** | 0 | 0 | 0 | 6.4% | |
| **pooled** | **90** | **0** | 0 | 0 | 0 | **3.3%** | **4.1e-14** |
| gloo control | 2 | 0 | 0 | 0 | 0 | | |

Read the two right-hand columns together. The 95% one-sided Clopper-Pearson
upper bound on the failure rate is **3.3%** — an order of magnitude below the
29% AS.4b base rate. And if that class were still present at its measured
rate, the probability of seeing zero failures in 90 runs would be **4e-14**.
The concurrent-MPI-progress class does not reappear at P=16 x 8 nodes with
`MPI_Comm_split` running on an intra-op pool worker.

**The ledger doubles as a 46-fold determinism check, and that is arguably the
stronger result.** Every GW cell emitted comment-stripped md5s of its four
artifacts. Across **46 GW runs — 45 under `impl=mpi` + the gloo control —
there is exactly ONE md5 per artifact**:

```
eqp0.dat        201b31e9baf62966   x46   (1290 data rows)
eqp1.dat        af09d549ed579677   x46   (1290 data rows)
eqp_g0w0.dat    4236fe492ec9b605   x46   (2096 data rows)
sigma_diag.dat  4b4b5716ab5bfe3f   x46   (2096 data rows)
```

and across 46 BSE runs there is exactly **one** distinct eigenvalue vector:

```
Lowest 4 eigenvalues (eV): [1.30537661 1.3504201  1.42411254 1.50449023]
```

which is also the value the gloo reference and the original P=4 runs produced.
Given that gloo's corruption signature is *a different wrong value each time*,
46 byte-identical results spanning both transports is direct evidence that
neither transport corrupted this workload in this window — and it is the
invariant the campaign previously lacked on these paths (§2.6 of the
psum_scatter archaeology).

Timings, incidentally: GW median **92 s under mpi vs 155-201 s under gloo/ib0**
on the same allocation, i.e. **1.7-2.2x**, larger than the 1.18x AS.4 recorded.

### RESULT — P=64 x 32 nodes

Job **7880811** (32 nodes, 4 h). *(final numbers appended when it lands; at
last read: 17 mpi cells, 0 failures, bound 16.2%, P(0 fails | 29%) = 3.0e-03,
GW 4962c median 403 s and BSE N_mu=10015 median 245 s, gloo control 549 s.)*

## PHASE 3 — the cleanup. **PREPARED AND GATED; applied only on Phase 2 green.**

Prepared in an ISOLATED tree (`wk_REL/allmpi/tree_PHASE3_6c7feb0`, built from
`git archive 6c7feb0`), never in the shared live worktree, so nothing was
deleted from the repo before the replacement was proven and so a concurrent
workstream could not be disturbed.

### Scope actually found at `6c7feb0`

Confirms the owner's scoping: **four** transport-conditional branch points in
all of `src/` (`runtime/__init__.py:460`, `:510`; `contract_bands.py:198`;
`zeta_projection.py:224`), zero gloo in `config/`, and **zero transport
references anywhere under `src/bse/`** — BSE was forced onto gloo
operationally in job scripts, never in code. LORRAX never *sets* the
implementation, it only reads it.

### The change set (7 files under src/, net -437 lines)

| file | Δ | what |
|---|---|---|
| `src/runtime/__init__.py` | **-228** | the whole 282-line Gloo pin block (`_GLOO_PIN_SENTINEL`, `_GLOO_DISABLE_TOKENS`, `_FABRIC_PREFIXES`, `_SIOCGIFADDR`, `_iface_ipv4`, `_iface_is_up`, `_detect_fabric_iface`, `pin_gloo_interface`), the `__all__` entry, the `bootstrap()` call, and the re-arm inside `fallback_to_cpu_if_no_gpu_backend` — replaced by a 45-line `announce_cpu_collectives()` |
| `src/common/contract_bands.py` | -56 | `ensure_grouped_collectives_ready` + its `__all__` entry + the factory call site; module-docstring item 4 rewritten as *withdrawn* |
| `src/common/zeta_projection.py` | -145 | `ensure_world_clique_ready` (including the advice string that pointed users AT the corrupting backend) + `__all__` + import + all five call sites + a now-unused `import os` |
| `src/common/zeta_projection_test.py` | -8 | drops both helpers and the `pin_gloo_interface()` call; the "this driver runs on gloo BY MEASUREMENT" comment corrected |
| `src/file_io/_slab_io_mpi_host.py` | -1 | "the certified default (gloo collectives)" → the MPI stack; wrapper path now names the in-tree recipe |
| `src/gw/kin_ion_io.py` | -1 | drops `GLOO_SOCKET_IFNAME`/`LORRAX_GLOO_IFNAME` from a launch-failure message |
| `src/ffi/phdf5/cpp/context.cc` | +1 | the thread-level warning stops offering "or run the gloo collectives default" and names `build_mpiwrapper.sh` |

Docs: `docs/dev/env_vars.md` (both `LORRAX_GLOO_IFNAME` rows removed, the
`_LORRAX_GLOO_PIN_DONE` and `GLOO_SOCKET_IFNAME` rows corrected, the
`JAX_CPU_COLLECTIVES_IMPLEMENTATION` row rewritten, the **AS.7 block
replaced**), `docs/dev/HANDOFF_2026-07-29.md:99-101` (marked SUPERSEDED),
`docs/dev/staged_reshard_primitive.md` (§3.5 rewritten, the two stale
amendments replaced), `config/frontera/README.md` (+ the launch recipe).

### The gloo census in `src/` after the change

Six lines, down from a 282-line block plus four branch points:

```
runtime/__init__.py:49    historical comment (the falsy-parse consolidation)
runtime/__init__.py:272   comment explaining WHY the announcement exists
runtime/__init__.py:284   docstring
runtime/__init__.py:294   the default when the env var is unset
runtime/__init__.py:309   the warning text itself
zeta_projection_test.py:73  comment recording the corrected claim
```

Nothing in `src/` branches on the transport in a way that changes **what is
computed**. The single remaining `impl == "mpi"` test decides only which of
two log lines is printed.

### The one judgement call, flagged for the owner

Deleting the pin outright would leave a forgotten
`JAX_CPU_COLLECTIVES_IMPLEMENTATION` export **silent** — the run would ride
gloo (corrupting *and* 14-30x slower on the collective-bound stages) with
nothing printed anywhere. Today the pin at least prints a banner. Since the
owner's instruction is that the recipe be the *documented* default rather than
a code default, `announce_cpu_collectives()` is the minimum that keeps the
codebase's own standing doctrine #3 ("a demotion may happen, but it must
announce itself"): one rank-0 line, one place, no behaviour change.

**It is a separable commit.** If the owner prefers literally zero gloo
references in `src/`, drop commit 2 below and `bootstrap()` simply loses a
call; nothing else in the change set depends on it.

Unit-checked over eight environment corners (P=1, non-rank-0, impl=mpi with
and without `MPITRAMPOLINE_LIB`, impl unset, impl=`GLOO` upper-case, pure-GPU
platform, `cuda,cpu` with and without a GPU present): silent where it should
be silent, warns where it should warn.

### Gate results

**(a) Production GW + BSE A/B, job 7880854 — `sacct` COMPLETED 0:0, 5 m 21 s.**
BASE source then PHASE3 source, `impl=mpi` + `LORRAX_MPI_FORCE_THREAD_MAIN=1`,
P=16 x 8 nodes, back to back in ONE allocation on the SAME nodes. Both
snapshots' manifests verified at job start **and** end.

| leg | rc | wall |
|---|---|---|
| GW  BASE   | 0 | 175 s |
| GW  PHASE3 | 0 | 88 s |
| BSE BASE   | 0 | 24 s |
| BSE PHASE3 | 0 | 24 s |

Artifact parity, comment lines stripped (the header stamps its own generation
time and a raw byte-diff over-discriminates on it):

```
IDENTICAL  eqp0.dat        (1290 data rows)
IDENTICAL  eqp1.dat        (1290 data rows)
IDENTICAL  eqp_g0w0.dat    (2096 data rows)
IDENTICAL  sigma_diag.dat  (2096 data rows)
IDENTICAL  BSE eigenvalues
```

6772 data rows, byte-identical. An earlier run of the same gate against the
previous snapshot (job 7880836, before an unused `import os` was dropped) also
returned PASS with the same digests.

Do **not** read the 175 s → 88 s GW wall as a speedup from the change: the
PHASE3 leg runs second and benefits from page cache on the deck. The A/B was
built to compare *values*, not walls; the transport speedup is measured
separately by the gloo-vs-mpi cells in the ledger.

**(b) `common.zeta_projection` A/B — a SEPARATE gate, because the production
gate cannot reach it.** `src/common/zeta_projection.py` has **no importer
anywhere in `src/`** (one mention, in a comment in `common/collectives.py`).
Its only consumer is its own driver, `common/zeta_projection_test.py`. So the
GW/BSE gate above proves nothing about the 145 lines deleted there — including
the advice string that pointed users at the corrupting backend — and that is
precisely the module where the gloo corruption was first caught.

`wk_REL/allmpi/zproj_gate.sbatch` runs three cells (`dense`, `roundtrip`,
`selection`) on BOTH trees, back to back, P=4 / 2 nodes, under `impl=mpi` +
`LORRAX_MPI_FORCE_THREAD_MAIN=1` — the configuration the deleted helper's own
docstring said could not work.

**It immediately failed, and the failure was mine, not the transport's**
(job 7880908, all three PHASE3 cells rc=1 against three BASE cells rc=0):

```
File ".../src/common/zeta_projection.py", line 742, in build_zeta_transfer
    p_x, p_y = _check_mesh(mesh_xy, axes)
NameError: name '_check_mesh' is not defined
```

The first deletion was written as "remove everything from the warm-up section
header to the next section header". Three unrelated helpers lived in that
range — `_MU_DIV_MSG`, `_check_mesh`, `_require_div` — and went with it. The
module still imported, still `py_compile`d clean, and still passed the
production gate (which never calls it); only executing the module caught it.

Fixed by rebuilding the file from BASE with a deletion bounded by the two
*definitions* rather than by the two *comment banners*, and then checked
properly rather than by eye:

* **pyflakes** — see the correction below: my first two pyflakes runs were
  VACUOUS (wrong interpreter). Re-run properly inside the container (job
  7881211), the finding set is **identical** across BASE, the Phase 3 tree and
  the live worktree: 275 findings, 16 undefined names, all pre-existing at
  `6c7feb0`, none introduced here. A `py_compile`/import check cannot see an
  F821 inside a function body; pyflakes can — when it actually runs.

**Instrument defect in the gate itself, also fixed.** Its first version
diffed raw driver stdout, so it reported `*** DIFFERS` on (i) the runtime
banner, which changes **by design** in Phase 3, and (ii) Intel MPI's startup
table, which prints PIDs. The real regression was buried under that noise and
the summary line said only "FAIL". It now checks exit codes explicitly,
compares only the driver's own `rel = ` / `PASS` result lines, and prints the
banner change separately and labelled as expected.


**Re-run on the corrected change set — job 7880917, `sacct` COMPLETED 0:0,
1 m 51 s: ZPROJ GATE PASS.**

```
cell=dense      BASE rc=0  PHASE3 rc=0
cell=roundtrip  BASE rc=0  PHASE3 rc=0
cell=selection  BASE rc=0  PHASE3 rc=0

IDENTICAL  dense       W_L device-vs-host   rel = 1.522e-16  (gate 1e-14)
                       W_L hermiticity      rel = 1.522e-16
                       W_S vs dense numpy   rel = 5.292e-15  (gate 1e-12)
                       W_S hermiticity      rel = 3.473e-16
                       cond(G_S) max over q =  2.422e+01
IDENTICAL  roundtrip   PASS
IDENTICAL  selection   W_S vs W_L[idx,idx]  rel = 7.641e-17  (gate 1e-12)
```

The single intended difference, printed separately and labelled:

```
BASE   [runtime] Gloo interface pin: no-op (CPU collectives implementation is 'mpi'…)
PHASE3 [runtime] CPU collectives: mpi (MPItrampoline -> …/libmpiwrapper.so).
```

Note what this cell also demonstrates in passing: the module's grouped
`psum_scatter` chain runs clean under `impl=mpi` with **no warm-up call of any
kind**, which is the direct experimental refutation of the contract the
deleted helper existed to satisfy.

---

## RETARGET (2026-07-29, ~22:30) — the jax-internals agent found a fix with no patched dependency

`wk_REL/docs/jax_threadmain_alternatives.md`. **The wrapper's `MPI_Is_thread_main`
override is superseded.** `common.collectives.warm_mesh_cliques(mesh)` creates
one MPI communicator per mesh axis plus one over all axes, from the Python main
thread, at mesh-construction time — three 8-byte `psum`s, ~150 ms once per
process. XLA caches communicators per participating-device set and calls
`CreateCommunicator` only on a miss, so every later collective (including the
ones a pool worker issues inside the BSE Lanczos jit) is a cache hit and never
reaches the guard.

### What this does to each phase

* **Phase 1** — the wrapper does not disappear, its scope shrinks. Override 2
  is superseded; **override 1 (`MPI_THREAD_MULTIPLE`) is still required** for
  the AS.4b concurrent-progress class and retires only upstream. Docs rewritten
  accordingly (`mpi_collectives.md` STATUS, the `build_mpiwrapper.sh` header,
  the `env_vars.md` row).
* **Phase 2** — retargeted. The override-path ledger is complete and recorded
  below; the *shipping* path is now the warm-up, so new ledgers run with
  `LORRAX_MPI_FORCE_THREAD_MAIN` **unset** and carry a negative control.
* **Phase 3 — MATERIALLY REVISED.** My two "obsolete" helpers are **partially
  vindicated and are NOT deleted.** The old model was wrong about *ordering*
  but right that warm-up matters; it warmed the WORLD clique only, which the
  controls (job 7881053) show fails, as do x-only and x+y-without-world. Only
  x+y+world passes — the caching is **per-clique**. So the correct move was to
  fix the warm-up, not remove it.
* **Phase 3 gloo deletion — unchanged.** It stands on gloo's own record.

### Revised Phase 3 change set

`common/collectives.py` gains `warm_mesh_cliques` (the module that already owns
the collectives helpers). `contract_bands` and `zeta_projection` call it where
the old helpers stood — five sites in `zeta_projection`, one in the
`contract_bands` factory — and the two wrongly-scoped helpers are retired in
its favour rather than simply dropped. One helper, in one place, warming the
right device sets, instead of two helpers warming the wrong ones.

I did NOT touch `src/bse/` or `src/solvers/` (the jax agent's lane); the
`bse_ring_comm.create_mesh_2d` call site is theirs to add.

### Two instrument defects found while gating this, both worth keeping

1. **My refusal counter counted the documentation of the error as the error.**
   The BASE `zeta_projection` cells scored `refusals=1` while returning rc=0.
   The "refusal" was the module's own advice string, which embeds the literal
   text `MPI: Communicator requested from a thread that is not the one MPI was
   initialized from` inside a paragraph recommending gloo. A grep for the error
   matches the paragraph that describes it. (This is also an independent reason
   to delete that string: it is both wrong advice and a detector poisoner.)
2. **The `zeta_projection` negative control is VOID at that size.** `dense_neg`
   — BASE source, gate unset, no warm-up — returned **rc=0**. The driver's
   cells are small enough that XLA takes the sequential thunk executor and the
   guard is never reached, which is precisely the trap that voided the earlier
   clean-room probes (job 7879684) and made warm-up look unnecessary. So the
   zproj gate proves *no regression*, and proves the warm-up runs (banner:
   `warmed 3 MPI cliques for mesh (2, 2) axes=['x', 'y'] in 116 ms`), but it
   does **not** prove the warm-up is load-bearing. That evidence has to come
   from a real deck: the jax agent's job 7881054 at P=4 (8 refusals without,
   0 with) and the negative-control cells in the at-scale ledgers below.

### Clique census — the warm-up covers EVERY clique GW touches (job 7881073)

`TF_CPP_VMODULE=cpu_cliques=3` on production GW 785c at **P=16, gate UNSET**:

```
rc=0, wall 174 s, refusals 0
clique CREATIONS (cache misses) : 48   = 16 ranks x 3
distinct device sets created    :  9
    4 row-sets    [0,2048,4096,6144] [8192..14336] [16384..22528] [24576..30720]
    4 column-sets [0,8192,16384,24576] [2048,10240,18432,26624] ... (stride 8192)
    1 world-set   (all 16)
```

For a 4x4 mesh that is **exactly** `p_y` rows + `p_x` columns + world, and
exactly three creations per rank. There is no tenth device set — **no code
path on the GW production path asks for a non-mesh-axis clique**, which is the
question the owner posed. Every other acquisition in the log (thousands) is a
cache hit and never reaches the guard.

*Instrument defect, caught and fixed:* the first version of this census counted
`Acquire communicator` lines and reported "6624 creations after warm-up" for a
run whose real answer is 48. An acquire is a cache **hit** and proves nothing;
only `Create a new communicator` reaches the guard. The census now counts
creates and enumerates their device sets, and says what the expected set is.

### Census with BSE covered — job 7881102, the decisive pair

Re-run with the BSE cell routed through the out-of-tree hook. **Both
workloads, P=16, wrapper gate UNSET:**

| cell | rc | wall | creations | distinct sets | ranks | refusals |
|---|---|---|---|---|---|---|
| GW 785c  | **0** | 168 s | 48 | **9** | 16 | **0** |
| BSE 785c | **0** | 21 s  | 48 | **9** | 16 | **0** |

48 = 16 ranks x 3 cliques; 9 = 4 rows + 4 columns + 1 world for a 4x4 mesh.
**Neither GW nor BSE asks for a non-mesh-axis clique**, so the mesh-axis
warm-up is not merely sufficient here by luck — it is exhaustive.

And this is the load-bearing negative control the zproj gate could not give,
on a real deck at real scale: the **same** BSE cell, same job script, same
node count, differing only in whether the warm-up call site exists —

```
7881096  bse_warm  rc=1  32 refusals  0 warm-up banners   (no call site)
7881102  bse_warm  rc=0   0 refusals  warm-up ran         (call site via hook)
```

### The warm-up needs a call site at EVERY mesh factory — BSE does not have one yet

Same job, BSE cell at P=16, gate unset: **rc=1, 32 refusals, 0 warm-up
banners.** Not a concurrency failure and not a scale failure — BSE's mesh is
built by `bse.bse_ring_comm.create_mesh_2d`, and **this workstream does not own
`src/bse/`** (the jax-internals workstream is reading there). GW is covered for
free because its grouped kernels go through `contract_bands_block_reshard`,
which now warms; BSE's Lanczos path does not go through that factory.

**Handoff, one line:** `src/bse/bse_ring_comm.py::create_mesh_2d` must call
`common.collectives.warm_mesh_cliques(mesh)` before returning. Until it does,
BSE under `impl=mpi` needs either that call site or the wrapper gate.

For the at-scale ledger I did **not** work around this by editing `src/bse/`.
The BSE cells instead run through the jax workstream's out-of-tree
`wk_REL/thrmain_alt/tm_warmhook.py`, which patches `create_mesh_2d` in memory
and re-runs the unmodified driver — so the ledger measures the warm-up
mechanism at scale without either workstream editing the other's files.

### RESULT — WARM-UP route, P=16 x 8 nodes: **60 cells, 0 failures**

Job **7881094**, `sacct` COMPLETED, 1 h 21 m 42 s, `LORRAX_MPI_FORCE_THREAD_MAIN`
**UNSET** throughout, source pin verified immutable at START and END.

| group | runs | fail | segv | hang | refusal | 95% upper | P(0 fails \| 29%) |
|---|---|---|---|---|---|---|---|
| warm-up, GW 785c  | 30 | **0** | 0 | 0 | 0 | 9.5% | |
| warm-up, BSE 785c | 30 | **0** | 0 | 0 | 0 | 9.5% | |
| **pooled** | **60** | **0** | 0 | 0 | 0 | **4.9%** | **1.2e-09** |

**Cross-route determinism now spans three transports/routes.** The 32 GW cells
in this job emit the same four md5s as the 46 GW cells of the override-path
ledger — 78 production GW runs across **gloo**, **mpi + wrapper override**, and
**mpi + warm-up (no patched dependency)**, one md5 per artifact:

```
eqp0.dat 201b31e9baf62966   eqp1.dat af09d549ed579677
eqp_g0w0.dat 4236fe492ec9b605   sigma_diag.dat 4b4b5716ab5bfe3f
```

and one distinct BSE eigenvalue vector across all of them,
`[1.30537661 1.3504201 1.42411254 1.50449023]` — which is also the P=4 value
from the original gloo run. Given gloo's corruption signature is a *different*
wrong value each occurrence, this is the invariant these paths never had.

### RESULT — WARM-UP route, P=64 x 32 nodes: IN FLIGHT

Job **7881095** (32 nodes, 4 h wall). At last read: **7 rep cells, 0
failures** (5 GW 4962c at ~405 s, 2 BSE N_mu=10015 at ~331 s), bound 34.8%.
Not yet decisive at P=64 on this route; the override route reached 20 cells /
bound 13.9% there. Finish this job before flipping any default at P>=64.

### Summary of the two routes, at scale

| route | P=16 x 8 nodes | P=64 x 32 nodes | pooled |
|---|---|---|---|
| mpi + wrapper override (route A) | **90 / 0**, bound 3.3%, P=4.1e-14 | **20 / 0**, bound 13.9% | **110 / 0**, bound **2.7%**, P(0\|29%) = 4.4e-17 |
| mpi + main-thread warm-up (route 1b — what we ship) | **60 / 0**, bound 4.9%, P=1.2e-09 | **11 / 0**, bound 23.8% *(job still running)* | **71 / 0**, bound **4.1%**, P(0\|29%) = 2.8e-11 |

(`n / f` = rep cells / failures; bound = 95% one-sided Clopper-Pearson upper
bound on the failure rate; `P(0|29%)` = probability of zero failures if the
AS.4b class were still present at its measured 29%.)

At P=16 x 8 nodes — the configuration where AS.4b measured the
concurrent-MPI-progress class at 4/14 (~29%) — **both routes clear it**, and
route A's 90/0 is an upper bound on route 1b's exposure in that dimension
(route 1b makes strictly fewer MPI calls from pool threads: none at all for
communicator creation).

### Warm-up ledger controls — one void, one genuine, and a cost number

Jobs 7881094 (P=16) / 7881095 (P=64), warm-up route, gate unset.

**`neg_nowarm_bse` is VOID — a fourth control defect, same family as the
other three.** It was meant to run BSE with no warm-up at all, and its log
says `[warmhook] 3 cliques warmed on the main thread for mesh (4, 4) ...`.
The negative-control block switched the *source snapshot* but not the
*out-of-tree hook*, so the cell received exactly the thing it was supposed to
withhold. Fixed (the block now unsets `AM_BSE_HOOK` around those cells).

**`neg_nowarm_gw` PASSING is a genuine result, not a defect.** BASE source,
gate unset, P=16: **0 warm-up banners, 0 refusals, rc=0**. GW does not need
the warm-up and never did — consistent with the AQ 4962c/P=64 and every AY
round having run `impl=mpi` before any of this existed. GW's cliques happen to
be first touched during a phase XLA runs sequentially. **So the GW arm of the
ledger cannot discriminate; only the BSE arm can.** Worth stating plainly
because it means "GW is green on the warm-up path" is a weaker statement than
it looks.

**The definitive negative/positive pair is therefore the clique census, not
the ledger control** — same deck, same P=16, same job script, differing only
in whether the warm-up call site exists:

```
7881096  bse_warm  rc=1  32 refusals  0 warm-up banners
7881102  bse_warm  rc=0   0 refusals  warm-up ran, 48 creations / 9 device sets
```

**Cost at scale — and a correction to my own first reading of it.** There are
TWO numbers in the logs and they differ by ~45x, so it matters which is quoted:

| what | P=16 | P=64 |
|---|---|---|
| in-tree `warm_mesh_cliques` (from `contract_bands`, GW path) | **54-64 ms** | **63 ms** |
| out-of-tree `tm_warmhook` (from `create_mesh_2d`, BSE path) | 2834 ms | 3001 ms |

I initially quoted the 2834 ms as "the warm-up cost at P=16". That is wrong as
a statement about what we would ship. The difference is *when* in the process
the warm-up runs, not what it does: the hook fires at mesh construction, which
is the first MPI traffic in the process and therefore pays connection
establishment; the in-tree call fires at kernel-build time, after the run has
already done collectives, and costs ~60 ms.

Either way it is **once per process, flat in P (54-64 ms at P=16, 63 ms at
P=64), and independent of `N_mu`/`N_k`/`N_q`**. The honest summary is that the
warm-up itself is ~60 ms and effectively free; the ~3 s seen on the BSE path is
first-MPI-touch cost that the process pays sooner rather than later.

*Not* explained, and deliberately not quoted as a cost: the first P=64 GW
warm-up cell walled 922 s against ~403 s for the override-path cells on a
different allocation. That is 500 s against a 63 ms warm-up, so it is not the
warm-up; it is node/co-tenancy variation or first-cell compile. It needs an
A/B in one allocation before anyone reads anything into it.

### The fifth void check — and the pattern all five share

**My pyflakes runs were not running.** `…/.venv/bin/pyflakes` has a shebang
pointing at `…/.venv/bin/python`, which symlinks to `/usr/local/bin/python3.12`
— a path that exists only *inside* the apptainer container. On a login node it
dies with `bad interpreter: No such file or directory`, printing **one line to
stderr**. My check was

```
$PF $TREE/src 2>&1 | grep -cE "undefined name|invalid syntax"     ->  0
```

and 0 was the error message not matching, not a clean tree. The follow-up
"identical to BASE, 1 finding each" was that same error line, identical in both
because it was the same error. Two separate reassuring results, both content-free.

Re-run inside the container (job **7881211**, `sacct` COMPLETED):

| tree | findings | undefined names | syntax errors |
|---|---|---|---|
| BASE `6c7feb0` | 275 | 16 | 0 |
| Phase 3 tree | 275 | 16 | 0 |
| live worktree (mine + the other workstream's) | 275 | 16 | 0 |

Stripped of line numbers the three sets are **byte-identical**, so the real
conclusion survives — the change set introduces no new lint finding, and the
16 undefined names (`PartitionSpec` in `wfn_loader`, `jax` in
`build_projectors_qe`, and the cross-cell locals in `zeta_projection_test`) are
all pre-existing.

*One consequence to flag rather than paper over:* the Phase 3 snapshot's
`PROVENANCE.txt` carries the line `lint: pyflakes over src/ IDENTICAL to the
BASE tree`, written when the check was still vacuous. It happens to be true —
job 7881211 established it independently — but it was **not** evidence when it
was written. The snapshot is manifest-frozen and a running job depends on it,
so it has deliberately not been edited; this note is the correction.

**The pattern in all five defects of this campaign is one thing:** a check
whose failure mode is *silence* was read as a pass.

1. `grep -c` prints 0 and exits 1 -> `|| echo 0` made a two-line value -> every
   cell scored FAIL at rc=0.
2. `strings | grep -q` under `pipefail` -> SIGPIPE -> a good binary reported
   broken.
3. Two ledger jobs sharing a run directory -> two cells scored FAIL that were
   collisions, and eight that had already passed became unattributable.
4. A negative control that received the very hook it was meant to withhold.
5. A linter that never ran.

Four of the five produced *reassuring* output. The counter-measure that
actually worked, every time, was the same: **make the instrument prove it can
fail.** The zproj gate found a real `NameError` only because it executed the
module; the clique census became meaningful only when it counted CREATEs
(which can be zero) instead of ACQUIREs (which never are); and the BSE warm-up
claim is only load-bearing because 7881096 (no call site) and 7881102 (call
site) differ on the same deck at the same scale.

## COMMIT-READY SPLIT (nothing is committed, by instruction)

Five commits. Commit 1 stands alone. Commits 2-4 are the gloo deletion and the
warm-up fix; commit 5 is the doc correction. **Commit 6 is NOT mine** — it is
the one-line BSE call site the jax-internals workstream owns.

### 1. `build: version-control the interim MPIwrapper, and document its exit`

```
config/frontera/build_mpiwrapper.sh              (new)
config/frontera/mpiwrapper/lorrax_thread.patch   (new)
docs/dev/mpi_collectives.md                      (new)
config/frontera/README.md                        (+ the multi-process CPU launch recipe)
docs/dev/env_vars.md                             (+ LORRAX_MPI_FORCE_THREAD_MAIN row,
                                                  marked SUPERSEDED; MPITRAMPOLINE_LIB
                                                  and LORRAX_MPI_FINALIZE_FIX corrected)
```

The docs carry the STATUS framing: override 2 is **superseded** by
`warm_mesh_cliques`; override 1 (`MPI_THREAD_MULTIPLE`) is **still required**
and retires only upstream. Evidence: the recipe was run end to end and its
`.text` is bit-identical to the out-of-tree reference `3e8ee475…`; both
overrides are asserted in the machine code.

### 2. `collectives: warm every mesh-axis MPI clique from the main thread`

```
src/common/collectives.py                        (+ warm_mesh_cliques)
```

The mechanism that removes the patched-dependency requirement for
communicator creation. Self-contained and independently reviewable.

### 3. `collectives: fix the mesh-clique warm-up call sites`

```
src/common/contract_bands.py       (ensure_grouped_collectives_ready -> warm_mesh_cliques)
src/common/zeta_projection.py      (ensure_world_clique_ready -> warm_mesh_cliques, 5 sites)
src/common/zeta_projection_test.py
```

**This is a fix, not a deletion.** The old helpers warmed the WORLD clique
only, which the controls show fails; the corrected warm-up covers every mesh
axis plus the world. Also removes the advice string that pointed users at the
corrupting backend (and which poisoned our own refusal detector).

### 4. `collectives: delete the Gloo interface pin; announce the implementation instead`

```
src/runtime/__init__.py                          (-228)
```

Justified on gloo's own record and independent of everything above. The
45-line `announce_cpu_collectives()` that replaces it is separable — drop it
and `bootstrap()` just loses a call (see "the one judgement call").

### 5. `docs: correct the impl=mpi discriminator; drop the gloo advice`

```
docs/dev/env_vars.md                 (AS.7 block replaced; the gloo rows)
docs/dev/HANDOFF_2026-07-29.md       (:99-101 marked SUPERSEDED)
docs/dev/staged_reshard_primitive.md (§3.5 + the stale amendments)
src/file_io/_slab_io_mpi_host.py     (-1)
src/gw/kin_ion_io.py                 (-1)
src/ffi/phdf5/cpp/context.cc         (+1)
```

`docs/dev/env_vars.md` is touched by commits 1 and 5 in different hunks — stage
it by hunk.

### 6. NOT MINE — the BSE call site (jax-internals workstream)

```
src/bse/bse_ring_comm.py::create_mesh_2d   -> call warm_mesh_cliques(mesh)
```

Without it BSE under `impl=mpi` still needs the wrapper gate. Measured at
P=16: 32 refusals without, 0 with.

### Deliberately NOT changed

* `docs/dev/HANDOFF_2026-07-28.md:30` — a dated handoff recording that the AL
  workstream landed the pin. History, true when written.
* The `b4c7bca` **commit message** still states the wrong discriminator.
  Immutable; the correction lives in the docs it wrote.
* `config/frontera/ffi_env.sh` `FI_PROVIDER="${FI_PROVIDER:-tcp}"` — transport-
  adjacent but not gloo, outside this brief.
* Anything under `src/bse/` or `src/solvers/` — the other workstream's lane.

