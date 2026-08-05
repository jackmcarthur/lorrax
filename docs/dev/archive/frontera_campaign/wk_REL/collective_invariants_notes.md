# Collective invariants — closing the detection gap

2026-07-29, owner-directed.  Repo `/work2/08271/jackmc/frontera/lorrax` at
`6c7feb0` (branch `fix/zq-band-gather-device-invariance`).  **Nothing is
committed.**  All source edits are in the WORKING TREE only.

Background this builds on: `wk_REL/docs/eager_and_psumscatter_notes.md` §2.6 (the
structural finding) and `wk_REL/docs/UPSTREAM_gloo_psum_scatter_corruption.md` (the
bug).  The one-line statement of the gap:

> `check_hermitian` runs at exactly five sites — `common/sanity.py:53`,
> `gw/ppm_sigma.py:279,281`, `gw/screening.py:472`, `gw/gw_init.py:878` — and
> **none of them is downstream of a `psum_scatter`.**  Every invariant in this
> codebase sits on a CONSTRUCTION tile.  None sits on COLLECTIVE OUTPUT.  That
> is why a silent corruption firing at ~5 % per execution left no trace in 1913
> job logs.

## TL;DR

* **TASK 1 is done and free.**  All five Lanczos α sites now carry the
  Hermitian-form invariant `max|Im α| / max|α|`.  Zero extra collectives
  (HLO-verified), no measurable time (3 x P=1 + a warm P=4 A/B on the real
  BSE), byte-identical eigenvalues.  Always on, not gated.
* **The detector fires, demonstrated on the real BSE matvec at P=4.**  A
  segment-0 fault of the measured shape injected at
  `bse_stack_matvec.py:126` gives `residual = 3.7e-3 x rel`, linear over eight
  decades.  Detection threshold `rel = 2.7e-7` against a real corruption of
  `rel ~ 0.24`: a margin of `~1e6`.  A single bad execution in an otherwise
  clean 20-iteration solve is caught with 4-5 orders to spare.
* **It did not catch a real gloo event, because there is none to catch at this
  payload — and that is the new finding.**  On a node pair that corrupted 2 of
  2 processes at 253 MB, there were **0 events in 876 reduce-scatter
  executions at ≤ 41 MB** in the same allocation.  BSE-payload corruption rate
  **< 0.34 % (95 %)**.  The standing "BSE runs on a transport with measured
  silent corruption" is too strong for the payload BSE actually issues.
* **TASK 2 is designed, implemented and priced — and it is NOT free.**  Free at
  the BSE payload (+0.7 % mpi), but +107 % at 41 MB on mpi, because the cost is
  the Freivalds contraction (a memory-bandwidth sweep), **not** the scalar
  all-reduce the task assumed.  No call site is wired; that is the owner's
  call and 10 of the 16 sites belong to the concurrent all-MPI workstream.

---

# TASK 1 — the Lanczos α-Hermiticity detector

## 1.1 What was changed

Two files, working tree only:

| file | change |
|---|---|
| `src/solvers/lanczos.py` | the detector, at all five α sites, + the derivation |
| `src/common/sanity.py`   | `report_hermitian_residual` factored out of `check_hermitian` so both paths share ONE verdict, tolerance and message |

`src/solvers/lanczos.py:267` computed `alpha_j = jnp.vdot(q_prev, z).real` and
discarded the imaginary part.  For a Hermitian `H`, `⟨q, Hq⟩` is real for ANY
`q` — converged or not, at every iteration — so `Im α` is exactly the residual
of "the matvec returned `H·q`".  It was already being computed and thrown away.
The two BLOCK variants had the same disease one level up: `α_j = Q_jᴴ H Q_j` is
a Hermitian Gram block, and `_build_block_tridiag`'s closing `(T + Tᴴ)/2`
**silently absorbs** any violation.

All five sites now carry it:

| site (line at 6c7feb0) | function | quantity |
|---|---|---|
| `:83`  | `block_lanczos_eig` (eager)             | `max\|α − αᴴ\| / max\|α\|` via `sanity.check_hermitian`, verbatim, on the tiny `(n_blocks, bs, bs)` stack |
| `:185` | `simple_lanczos_eig` (eager)            | `max_j\|Im α_j\| / max_j\|α_j\|` |
| `:267` | `lanczos_eig_jit` (**the BSE path**)    | same, carried in the `fori_loop` carry |
| `:418` | `block_lanczos_eig_jit`                 | per-block `max\|α − αᴴ\|`, before the symmetrisation |
| `:521` | `block_lanczos_eig_jit_converged`       | same |

No signature changed anywhere; no caller was touched.  The residual leaves the
traced region through ONE unordered `jax.debug.callback` per solve carrying
three float64 scalars, because these solvers run inside
`bse_lanczos._full_run`'s outer jit with fixed `out_shardings` and you cannot
raise from inside a `fori_loop`.  Reducing to a *scalar* inside the jit before
the callback is load-bearing: a rank-0 array has no axis to shard, so XLA must
hand the callback the global value; passing the `(n_iter,)` array instead would
have risked each process seeing only its own shard and under-reporting.

**The quantity is check_hermitian's.**  T is Hermitian by construction, so
`(T − Tᴴ)_jj = 2i·Im α_j`; `max_j|Im α_j| / max_j|α_j|` is literally
`max|A − Aᴴ| / max|A|` restricted to T's diagonal, scaled against the tile's own
scale exactly as `check_hermitian` does.  Normalising by `max_j|α_j|` rather
than the per-iteration `|α_j|` is deliberate — a single α passing near zero
must not manufacture a false positive — and it is `check_hermitian`'s own
convention, not a new one.

## 1.2 The tolerance — derived, not tuned

`ALPHA_HERM_RTOL = 1e-9`.  Derivation (also in the module header):

With `‖q‖₂ = 1` and `u = 2⁻⁵³ = 1.11e-16`, `Im α` has exactly two sources and
both are a CONTRACTION LENGTH times u:

1. **the dot product.**  `α = Σᵢ conj(qᵢ) zᵢ` over `n` terms;
   `|fl(Σ) − Σ| ≤ γ_n Σ|qᵢzᵢ| ≤ γ_n‖z‖`, `γ_n ≈ n·u`.  (Pairwise summation,
   which XLA/BLAS actually use, is `O(log n · u)`; we keep the pessimistic
   `n·u`.)
2. **the matvec.**  `z = fl(Hq) = Hq + δz`, `‖δz‖ ≲ c_H·u·‖H‖`, with `c_H` the
   matvec's accumulation depth.  `⟨q, δz⟩` has no reason to be real.  For the
   BSE matvec the deep chains are the two reduce-scatter contractions over μ
   and ν (length `N_mu` each) plus the k-FFT and the c/v einsums, so
   `c_H ≈ 2·N_mu + nk·log₂nk + n_c + n_v = O(N_mu)`.

With `scale = max_j|α_j| = θ‖H‖`, `θ = O(1)`:

```
    rel  ≲  (n + c_H)·u / θ
```

At the LARGEST production BSE shape on this stack (`N_mu = 10015`,
`n = n_c·n_v·nk ≈ 4·10³`, θ ≥ 0.1 generous):
`(4·10³ + 2·10⁴)·1.11e-16 / 0.1 ≈ 2.7e-11`.

`1e-9` is that bound rounded up by ~40×.  There is no tuning freedom in the
gap it leaves: **any tolerance in `[1e-11, 1e-4]` gives the same verdict on
every case measured below.**

Measured floors (job 7880806/7880813, dense Hermitian matvec, single process):

| n | derived budget `2n·u` | measured `max\|Im α\|/max\|α\|` |
|---|---|---|
| 64   | 1.42e-14 | 5.2e-17 |
| 256  | 5.68e-14 | 4.2e-17 |
| 1024 | 2.27e-13 | 5.9e-17 |

and on the REAL BSE at P=4 (785c MoS₂ 4×4, `impl=mpi`): **7.185e-16**.  The
bound is pessimistic by 2–3 orders, as expected from pairwise summation.

## 1.3 Cost — free, hence ALWAYS ON

The scalar variants: `jnp.vdot` already produces a complex scalar; `.imag` is
the half that was being discarded — at worst two extra length-n multiply-
accumulate passes against a matvec orders of magnitude more expensive, in a
loop body that already does `n_reorth` dot products of the same length.  The
block variants: `α_j` is already fully materialised (the recurrence subtracts
`Q_j @ α_j`), so the residual is `bs² = O(10)` flops on a resident tile.  No
extra collective, no device sync, no full-tile pass.

Measured (three-way attribution, interleaved A/B/C):

| shape (dense matvec) | ARITH − BASE, three runs | FULL − BASE, three runs | measurement floor (median − min of BASE) |
|---|---|---|---|
| n=1024, 40 iters | **+0.21 % / +0.06 % / +0.26 %** | −1.5 % / +4.1 % / +0.07 % | 2.9 % / 1.6 % / — |
| n=4096, 40 iters | −4.3 % / +5.0 % / −1.1 % | −10.3 % / −8.9 % / −9.9 % | 13 % / 14 % |
| n=4096, 20 iters | +3.9 % / −0.1 % / −0.6 % | +4.1 % / +6.0 % / −5.1 % | 18 % / 6 % |

(three independent runs — jobs 7880813, 7880826 and 7880915 — interleaved
A/B/C, 15 reps, median.  `BASE` = the pre-edit function copied verbatim from
`6c7feb0`; `ARITH` = instrumented with `_emit_alpha_herm` stubbed, isolating
the per-iteration arithmetic; `FULL` = as shipped.  Absolute BASE times:
59-61 ms, 1.25-1.34 s, 0.58-0.67 s.)

**The signed deltas straddle zero on every shape and in both directions**, and
every one of them is smaller than that shape's own median-minus-minimum spread.
The honest statement is not "+0.2 %" — it is **"below the measurement floor of a
shared node"**, and at the tightest measurement available (n=1024, floor 1.6 %)
`ARITH − BASE` is +0.06 % and −1.53 % in the two runs.  `FULL ≈ ARITH`
everywhere, so the one callback + one print per solve is not the cost either
(worst observed: +2.4 ms on a 59 ms solve; on a real BSE solve that is a
constant of a few ms against tens of seconds).

`max|Δ eigenvalue| = 0.000e+00` on every shape in both runs.

**"Zero extra collectives" is not an argument, it is measured.**  Job **7880831**
`COLLINV_hlo` (COMPLETED 0:0) dumps the rank-0 *optimized* HLO for the P=4 BSE
solve on both snapshots and counts collective instructions:

```
                       BASE   FIX
    all-reduce            7     7
    reduce-scatter        2     2
    all-gather            0     0
    collective-permute    0     0
    all-to-all            0     0
    ZERO EXTRA COLLECTIVES: PASS - BASE and FIX collective counts identical
```

with both legs producing `[0.09594338 0.09925401 0.1046703 0.11057795]` Ry.
The reason there is nothing to add: `α_j` comes out of `jnp.vdot` on a
mesh-sharded vector, so the all-reduce that replicates it is **already** in the
recurrence — `Im α` rides that same reduction, and the post-loop `max()` is
over an array that is already replicated.

Per the owner's rule ("gate behind `sanity_enabled()` ONLY if it costs anything
measurable"), it is **not** gated: it reports through
`report_hermitian_residual(..., always=True)`, which bypasses the
`LORRAX_SANITY` *cost* escape hatch while still honouring `strict`.  A free
invariant that a stray environment variable can silence is not an invariant.

One informational line is printed per solve from **process 0 only**, verified
as exactly one line per solve at P=4 (`grep -c` = 1).  That line is the point:
the previous state of the world was 1913 job logs in which the invariant was
invisible whether it passed or not.  **Grep token in the delivered tree
(FIX-B):**

```
  lanczos[lanczos_eig_jit]: alpha non-Hermitian part / max|alpha| = 7.185e-16 (tol 1e-09, worst j=10)  OK
```

The P=4 gates below ran FIX-A, whose line reads
`max|Im alpha|/max|alpha| = ...` — same number, same place, older wording; the
FIX-A → FIX-B diff is message text only and is itemised under Provenance.  The
failure token is `*** LORRAX SANITY FAILURE:` on both, which is the one to
grep a campaign for.

## 1.4 GATE — clean run

**Job 7880807** `COLLINV_mpi`, `sacct` **COMPLETED** 0:0, 00:05:48, partition
`small`, PHY25006, 2 nodes x 2 ranks = **P=4**.  Deck: MoS2 4x4, 785 centroids,
TDA Lanczos, `--n-val 4 --n-cond 4 --n-eig 4 --max-lanczos-iter 20 --n-reorth -1`
— the same deck as jobs 7879458/7879463/7879697.  Transport `impl=mpi` with
`LORRAX_MPI_FORCE_THREAD_MAIN=1`, i.e. the transport measured clean in 584/584,
so every event in this job is attributable to the fault injected and not to the
wire.  Three frozen snapshots, each manifest-verified at job START **and** END
(351/351 files hashed, zero cached bytecode, `PYTHONDONTWRITEBYTECODE=1`).

**Detector reads at round-off on a clean run:**

```
  lanczos[lanczos_eig_jit]: max|Im alpha|/max|alpha| = 7.185e-16 (tol 1e-09, worst j=10)  OK
```

exactly ONE such line per solve (process 0 only; verified `grep -c` = 1), and
zero `LORRAX SANITY FAILURE` lines.  `7.185e-16` against a derived budget of
`~1e-12` for this shape: the detector's floor on the production BSE matvec, with
its two real reduce-scatters per iteration, is at unit roundoff.

**VALUE PARITY — the instrumentation is numerically inert.**  BASE (the two
files at `git HEAD`) and FIX (with the detector), same allocation, same nodes,
back to back:

```
  BASE  Lowest 4 eigenvalues (eV): [1.30537661 1.3504201  1.42411254 1.50449023]
  FIX   Lowest 4 eigenvalues (eV): [1.30537661 1.3504201  1.42411254 1.50449023]
  VALUE PARITY PASS - byte-identical
```

and that vector is character-identical to the one job 7879697 recorded for both
its `impl=mpi` and its gloo cells, and to job 7879463's original gloo run — so
this is now a **five-way** agreement, not a two-way one.

## 1.5 GATE — the detector actually fires

The owner's gate asks for a demonstration that the detector fires.  The premise
"the corruption reproduces at ~5 % per execution in the BSE configuration" is
**not established** and this workstream could not establish it — see §1.6 — so
the demonstration was built to be decisive independently of whether a real gloo
event lands: a segment-0 fault of the *measured shape* is injected into the
**real BSE matvec at the real `psum_scatter`** (`bse_stack_matvec.py:126`) and
swept in magnitude.

Fault model, taken from the upstream report §4.2/§4.3: only OUTPUT SEGMENT 0 of
the scatter is perturbed (`lax.axis_index('x') == 0`), by a deterministic
unit-modulus pattern scaled to `rel x RMS(A)`; everything else exact.  It lives
ONLY in the `FAULT` snapshot, gated on `LORRAX_FAULT_PSUMSCATTER` read at TRACE
time, so `rel=0` produces HLO identical to the shipped module.  Nothing of this
is in the working tree.

**E1 — magnitude curve, fault on every execution (job 7880807):**

| fault `rel` | residual `max\|Im α\|/max\|α\|` | detector | lowest eigenvalue (eV) |
|---|---|---|---|
| 0 (null cell) | 7.185e-16 | — | 1.30537661 |
| 1e-12 | 4.198e-15 | — | 1.30537661 |
| 1e-10 | 3.730e-13 | — | 1.30537661 |
| 1e-8  | 3.725e-11 | — | 1.30537661 |
| **1e-6**  | **3.725e-09** | **FIRES** | 1.3053766 |
| 1e-4  | 3.725e-07 | FIRES | 1.30537635 |
| 1e-2  | 3.725e-05 | FIRES | 1.30535028 |
| 1e-1  | 3.723e-04 | FIRES | 1.30506952 |

The response is **exactly linear** over eight decades: `residual = 3.725e-3 x
rel`, with the null cell four decades below the smallest injected fault.  The
detection threshold is therefore

```
    rel_min = ALPHA_HERM_RTOL / 3.725e-3 = 2.7e-7
```

— the detector catches a reduce-scatter whose output segment 0 is wrong by
**0.3 parts per million**.

**The margin against the real bug.**  The measured gloo corruption is a `2.667e+01`
error on a `1.119664e+02` answer (upstream §3), i.e. `rel ~ 0.24`.  Extrapolating
the fitted line: residual `~ 9e-4`, which is **~10⁶ x the tolerance**.  There is
no plausible instance of this fault that this detector misses.

**The reason it matters, in one row of the table.**  At `rel = 1e-1` the BSE
eigenvalues move by 0.3 meV — the 4th decimal place, entirely plausible, exactly
the "rc=0 but garbage" signature the sanity module was written for.  A human
reading `1.30506952` instead of `1.30537661` has no way to know.  The invariant
residual for the same run is `3.7e-04`, **five orders of magnitude** above its
tolerance. The physics error is invisible; the structural error is unmissable.

**E2 — SPORADIC fault, which is the real bug's actual shape.**  The upstream
report's fault hits a *minority* of executions.  A Python call counter cannot
express that here (the matvec body is traced ONCE inside `lax.scan` inside
`shard_map` inside the Lanczos `fori_loop`), so the trigger is a hash of the
collective's own output — deterministic, therefore reproducible, yet effectively
random across iterations. `every=20` over a 20-iteration solve is the
"ONE bad execution in an otherwise clean solve" case:

| cell | residual | detector | lowest eigenvalue (eV) |
|---|---|---|---|
| `rel=1e-1, every=8`  | 1.190e-04 | FIRES | 1.3055572 |
| `rel=1e-1, every=20` | 1.198e-04 | FIRES | 1.30529822 |
| `rel=1e-2, every=20` | 1.017e-05 | FIRES | 1.30537989 |

A single corrupted reduce-scatter inside an otherwise bit-perfect solve is
caught with **four to five orders of margin**.  Note the last row: `rel=1e-2`
once in 20 executions moves the lowest eigenvalue by 3e-6 eV — utterly
undetectable as physics — and still fires the invariant at 1e4 x tolerance.

**All FIVE variants, not just the one BSE uses** (job 7880915, P=1, dense
Hermitian matvec, n=256, 12 iterations, FIX-B).  The P=4 cells above exercise
`lanczos_eig_jit` only; this covers the other four, including the two block
paths whose `(T + Tᴴ)/2` would otherwise absorb the violation:

```
    variant                              clean    1e-12    1e-10    1e-08    1e-06    1e-04    1e-02    1e-01
    lanczos_eig_jit                        -        -        -       FIRE     FIRE     FIRE     FIRE     FIRE
    simple_lanczos_eig                     -        -        -       FIRE     FIRE     FIRE     FIRE     FIRE
    block_lanczos_eig_jit                  -        -        -       FIRE     FIRE     FIRE     FIRE     FIRE
    block_lanczos_eig_jit_converged        -        -        -       FIRE     FIRE     FIRE     FIRE     FIRE
    block_lanczos_eig                      -        -        -       FIRE     FIRE     FIRE     FIRE     FIRE

  SPORADIC (hash trigger, ~1/N of executions):
    rel=1e-01 every= 4: all five FIRE      rel=1e-02 every= 4: all five FIRE
    rel=1e-01 every=12: all five FIRE      rel=1e-02 every=12: all five FIRE
```

**Zero false positives on the clean column, all five variants, every run.**
That column is the one that decides whether a firing is believed.

**The one call shape the unit gate did not cover** is
`block_lanczos_eig_jit` invoked from *inside* a `lax.scan` body — what
`bse/exciton_bands.py:182` does, one Lanczos solve per exciton Q point, so the
debug callback fires once per scan step.  Job **7880939** `COLLINV_scan`
(COMPLETED 0:0) runs the identical problem through `lax.scan` and through a
Python loop:

```
  lanczos[block_lanczos_eig_jit]: ... = 1.089e-16 (tol 1e-09, worst j=4)  OK   (x3, scan path)
  lanczos[block_lanczos_eig_jit]: ... = 1.089e-16 (tol 1e-09, worst j=4)  OK   (x3, loop path)
  [scanprobe] scan-vs-loop max|d eigenvalue| = 0.000e+00
  [scanprobe] PASS - detector inside lax.scan neither raises nor perturbs
```

One report per Q point, correct residuals, bit-identical eigenvalues.

Note also that the sporadic cells' eigenvalues are *further* from the truth in
places than the persistent ones (`1.35093583` vs `1.34981105` for the second
root): a single bad iteration breaks the Krylov recurrence's orthogonality
rather than applying a coherent perturbation, so "rarer" is not "milder".

## 1.6a Regression suite

`pytest -q -x` on the delivered tree (job **7880935**, in the container,
`PYTHONPATH` = the working tree):

```
1 failed, 13 passed, 25 deselected  (164.97 s)
FAILED tests/test_bse_dense_reference.py::test_w_positive_control[ring]
  TypeError at src/bse/bse_ring_comm.py:245 -- lax.fori_loop carry inside
  shard_map: "the varying manual axes do not match ... complex128[1,2,399,9]
  vs complex128[1,2,399,9]{V:(x,y)}"
```

`bse_ring_comm.py` is not a file this workstream touches, so the question is
whether the failure is pre-existing.  **A whole-suite pass/fail is not the
discriminator** — job 7880941's BASE leg (the tree at `git HEAD`, my two files
reverted) is already red in ~30 places well before it reaches 40 % of the
suite, so `6c7feb0` does not have a green suite to begin with.  The
discriminator is the **failure SET**, BASE vs FIX, on the eight test files that
can reach `lanczos.py` / `sanity.py` / `collectives.py` (job **7880945**):

**Job 7880945** `COLLINV_pab`, COMPLETED.  Same eight test files, same
`tests/` directory, only `src/` swapped between the BASE snapshot (`git HEAD`)
and the delivered tree:

```
  BASE  15 failed, 72 passed, 3 skipped, 14 deselected   (209.51 s)  PYTEST_RC_base=1
  FIX   15 failed, 72 passed, 3 skipped, 14 deselected   (105.84 s)  PYTEST_RC_fix=1

  === FAILURE-SET DIFF (must be empty for 'pre-existing') ===
  IDENTICAL FAILURE SETS: the failure is PRE-EXISTING, not introduced here
```

Same 15 names, same counts, `diff` of the sorted `FAILED` lines is empty.
**This change introduces no test failure and fixes none.**  The 15, for the
record:

* 10 x `test_sanity_gates_jax.py` — `kin_ion_io`, `rho_work_items`, hartree
  mesh, `process_local` load.  None involves `check_hermitian`.
* 3 x `test_bse_dense_reference.py::*[ring]` — the `bse_ring_comm.py:245`
  shard_map VMA `TypeError`.
* 2 x `test_bse_dense_reference.py::test_nontda_*`.

The three tests that DO assert on `check_hermitian`
(`test_check_hermitian_true_and_false`, `test_device_check_hermitian`,
`test_check_hermitian_sharded_no_full_gather`) **pass on both legs** — which is
the direct coverage for the one function this change refactors.

Two things worth recording regardless of the verdict:

* The failure is a **jax 0.9 `shard_map` VMA (varying-manual-axes) typing
  change** in the LEGACY ring matvec (`build_bse_ring_matvec(low_mem=True)`),
  which `bse_stack_matvec`'s retirement note already lists as superseded.  JAX
  even prints the fix (`jax.lax.pcast(..., ('x','y'), to='varying')` on the
  initial carry).  It is worth someone's fifteen minutes; it is not this
  workstream's.
* **Gate-instrument defect:** the first pytest job reported `PYTEST_RC=0` next
  to `1 failed`, because the rc came from the `tail` at the end of the pipe,
  not from pytest.  Same class as `OWNER_DECISIONS.md`'s "l7 discards its rc
  through a pipe".  The A/B job above captures the rc without a pipe.

## 1.6 GATE — gloo hunt

The owner's gate says: "run the same BSE configuration on gloo, where the
corruption reproduces at ~5 % per execution — if the detector is real it will
catch a corrupted solve."  **That premise does not hold, and establishing why
is the most useful new fact in this workstream.**  The ~5 % is a measurement of
a **253 MB** reduce-scatter.  The BSE matvec's reduce-scatter
(`bse_stack_matvec.py:126`) moves **0.41 MB** — 600x smaller — and nothing in
the campaign had ever tested in between.  So the hunt was run with a
susceptibility calibration and a payload ladder in the SAME allocation, which
makes a null result mean something instead of nothing.

### The node pairs

Susceptibility is node-pair dependent (upstream §4.6a: pair A 0/100, pair B
5/5 processes, same hour).  Three pairs were drawn:

| job | pair | 253 MB reproducer | verdict |
|---|---|---|---|
| 7880808 | c211-\* | 0/19, 0/19, 0/19 (0 of 60) | **not susceptible** — its ladder and BSE cells carry no information |
| 7880829 | c209-003/004 | 1/19; **rep-0 hit** at `3.257096e+01` | susceptible (job died on a harness bug before its BSE cell) |
| **7880909** | c208-\* | **1/19 at `2.667173e+01`; rep-0 hit at `3.257096e+01`** — 2 of 2 processes | **SUSCEPTIBLE** — this is the pair everything below runs on |

Two independent notes for the upstream report, both free:

* The `2.667173e+01` value is **bit-identical** to job 7879558's rep-18 event
  on different nodes weeks apart, and `3.257096e+01` is a **new** member of the
  set (which now reads {8.475121e+01, 6.381745e+01, 3.257096e+01,
  2.667173e+01}).  §4.6a's "small discrete set of exactly-reproducible wrong
  values, not a continuum" is reinforced, and the set is not per-node-pair.
* Both rep-0 hits are §4.7's counter inversion in the wild.  Read the `rep 0:`
  line, not the summary.

### CELL B — the payload ladder, on the susceptible pair (job 7880909)

| payload | scattered-axis width | executions | corrupted |
|---|---|---|---|
| **253 MB** (the reference)          | 10008 | 2 processes | **BOTH** |
| 0.41 MB (BSE class), run a          | 400   | 19 | 0 |
| 0.41 MB (BSE class), run b          | 400   | 19 | 0 |
| 0.41 MB, **scattered axis = 4** (the BSE matvec's actual width) | 4 | 19 | 0 |
| 4 MB                                | 400   | 19 | 0 |
| 41 MB (Sigma class)                 | 400   | 19 | 0 |
| **total ≤ 41 MB**                   |       | **76** | **0** |

Same nodes, same hour, same executable, same gloo build.  If the rate were
payload-independent at 5 %, `P(0 in 76) = 0.95^76 = 0.021`.  **Payload
independence is disfavoured at p ≈ 0.02.**

### CELL C — 20 BSE solves on gloo, on the susceptible pair, detector live

```
CELL C SUMMARY: detector fired in 0 of 20 gloo BSE solves
                (= 800 reduce-scatter executions)
CELL C eigenvalue spread:
     80 Lowest 4 eigenvalues (eV): [1.30537661 1.3504201  1.42411254 1.50449023]
CELL C residual spread:
     20 = 7.185e-16 (tol
```

Three things at once, and the third is the one that matters:

1. **Zero detector firings** in 20 fresh processes = 800 reduce-scatter
   executions at the BSE payload.
2. **Every eigenvalue bit-identical** across all 20 (80 lines = 20 solves x 4
   ranks).  By the upstream report's own §4.3 logic — the wrong value differs
   between occurrences — bit-identical re-runs are independent evidence that
   none of the 20 was corrupted.  The detector and the A/B agree.
3. **Every residual bit-identical at `7.185e-16`.**  Not "all below tolerance"
   — all *the same number*, to every digit, across 20 independent processes on
   a transport that is demonstrably corrupting other payloads on those nodes in
   that hour.  That is a much stronger statement than a threshold pass, and it
   is a second detector for free.

### What the null means, stated precisely

Combining CELL B and CELL C: **0 events in 876 gloo reduce-scatter executions
at ≤ 41 MB, on nodes that corrupted 2 of 2 processes at 253 MB in the same
allocation.**  Rule of three: the corruption rate at the BSE matvec's payload
is **< 0.34 % (95 %)**, against ~5 % at 253 MB on the same hardware.

So:

* **The detector did not catch a real gloo corruption, because there was no
  real gloo corruption to catch at this payload.**  That is a finding about the
  BUG, not about the detector — and it is the finding that was missing: the
  campaign had assumed BSE was exposed (`OWNER_DECISIONS.md`: "BSE therefore
  runs on a transport with measured silent corruption") on the strength of a
  253 MB measurement.  On this evidence BSE's exposure is bounded at < 0.34 %,
  and the "payload size is the leading candidate discriminator" inference in
  `eager_and_psumscatter_notes.md` §2.5 now has direct support instead of being
  an inference.
* **The detector's ability to fire is not in question** — it is demonstrated in
  §1.5 on the real BSE matvec at P=4, with a linear response over eight decades
  and a threshold 10⁶ below the corruption's own magnitude.  A detector is
  validated by injecting the fault, not by waiting for the weather.
* **Not established:** whether the payload floor is sharp, where it sits
  between 41 MB and 253 MB, and whether it holds at other P / mesh shapes /
  replica-group widths.  All of §6.3 of the upstream report still stands; this
  narrows it, it does not close it.

### CELL F — warm wall-time A/B on the real BSE (job 7880909)

Job 7880807's BASE/FIX pair was not a cost measurement (BASE ran first and paid
the 60 s XLA compile).  Repeated warm, alternating, same transport, same nodes:

```
  BASE pass1  rc=0 wall=21s        FIX  pass1  rc=0 wall=21s
  BASE pass2  rc=0 wall=21s        FIX  pass2  rc=0 wall=21s
  all four:   [0.09594338 0.09925401 0.1046703  0.11057795] Ry
```

Identical to the second, and byte-identical eigenvalues on all four.  Together
with the HLO gate (§1.3) that is the cost claim closed on the real workload:
**no measurable time, no extra collective, no changed bit.**

---

# TASK 2 — `psum_scatter_checked`

## 2.1 The design

`common/collectives.py` now carries `psum_scatter_checked(x, axis,
scatter_dimension, tiled=True, *, name) -> (out, residual)` plus
`report_collective_residual(name, residual)` and `COLLECTIVE_RTOL`.
**No call site is wired.**  See §2.5 for why that is a decision and not an
omission.

**The invariant.**  A reduce-scatter over a replica group of P ranks satisfies,
by definition, `concat_r out_r == Σ_p x_p`, so for ANY weight tensor `w`

```
    Σ_{global j} w_j (Σ_p x_p)_j   ==   Σ_r Σ_{local j} w_{rL+j} out_r[j]
```

Both sides are built from `psum`, which the corruption study established as the
clean path (the all-reduce-only formulation of the same reduction was 200/200
clean in the harness that corrupted the reduce-scatter).  **The checker does not
rest on the primitive it is checking.**

**Freivalds, not a plain sum** — as instructed, and the reason is load-bearing:
a plain sum is invariant under moving mass between segments and cancels for a
zero-mean perturbation, which is exactly the observed failure shape (a stale /
dropped / duplicated SEGMENT, upstream §4.6a).

**`w` is rank-1 factorised**, `w[i0,i1,…] = Π_d v_d[i_d]`, with each `v_d` a
deterministic unit-modulus sequence — a pure function of `(length, axis)`, so
every rank agrees with no communication and a failure reproduces exactly.  It is
never materialised (it would be as large as the 253 MB payload); the contraction
is a sequence of axis-wise matrix-vector products: ONE pass, O(1) extra memory,
no large intermediate.

**ONE scalar all-reduce, not two.**  Only `|psum(s_in − s_out)|` is needed, so
the difference is formed locally and a single 2-element `psum` carries it with
the scale.

**One implementation trap, recorded** (it cost a job): the weight vector must be
generated at the GLOBAL axis length and then `dynamic_slice`d at this rank's
offset.  Generating it at `local_len + offset` puts `lax.axis_index()` — a
tracer — inside `jnp.arange` and raises `ConcretizationTypeError` (job 7880819).

## 2.2 It works — measured detection

Two formulations were measured.  The FIRST used a separate weighted-L1 pass for
the scale; the SHIPPED one uses ``|s_in|`` (the checksum of the correct
reduction) instead, because that pass turned out to be the entire cost (§2.3)
**and** the shipped form is both more sensitive and better-conditioned.

**Clean floors** (job 7880840, P=4, complex128) — identical to 6 digits on mpi
and gloo, as they must be (same HLO, different transport):

| payload | shipped floor | first-form floor |
|---|---|---|
| 0.41 MB (BSE matvec) | **3.6e-14** | 3.2e-14 |
| 41 MB (Sigma class)  | **2.0e-13** | 1.4e-13 |
| 253 MB (the corrupter) | **4.7e-13** | 2.3e-11 |

The floor grows like ``sqrt(payload)``, which is what pairwise summation of a
length-N random-walk sum predicts.

**Sensitivity to a segment-0 fault** (job 7880905, `impl=mpi`), shipped form:

| injected `rel` | residual @ 0.41 MB | residual @ 41 MB |
|---|---|---|
| 1e-14 | 2.26e-14 | 1.84e-13 |
| 1e-12 | 5.29e-12 | 1.23e-12 |
| 1e-10 | 5.32e-10 | 1.12e-10 |
| 1e-8  | 5.32e-08 | 1.12e-08 |
| 1e-6  | 5.32e-06 | 1.12e-06 |
| 1e-4  | —        | 1.12e-04 |
| 1e-2  | —        | 1.12e-02 |
| 1e-1  | —        | 1.12e-01 |

`residual = 0.53 x rel` and `1.12 x rel` — **essentially 1:1**, because the
fault and the checksum live on the same scale.  (The first formulation was
`7.8e-3 x rel`, i.e. 100x less sensitive, for 100x the cost.)

`COLLECTIVE_RTOL = 1e-10` therefore sits 200-2800x above the measured floor and
catches a fault of relative size ``~2e-10``, against an observed corruption of
``rel ~ 0.24`` — a margin of ``~1e9``.

## 2.3 COST — measured, and it depends entirely on the formulation

Jobs **7880825** and **7880840**, one allocation each, both transports,
`plain` = bare `lax.psum_scatter` in the same `shard_map`, medians of 12:

| payload | transport | plain | +L1-scale pass (first form) | **SHIPPED (no scale pass)** | sub-sampled 1/16 |
|---|---|---|---|---|---|
| 0.41 MB (BSE) | mpi  | 0.77 ms | 1.27 ms (**+65 %**) | **0.775 ms (+0.7 %)** | 0.72 ms (−6.8 %) |
| 41 MB         | mpi  | 21.9 ms | 56.3 ms (**+157 %**) | **45.4 ms (+107 %)** | 32.4 ms (+48 %) |
| 253 MB        | mpi  | 168.8 ms | 282.4 ms (**+67 %**) | **262.8 ms (+56 %)** | 223.1 ms (+32 %) |
| 0.41 MB (BSE) | gloo | 5.02 ms | 5.39 ms (+7.5 %) | **5.08 ms (+1.3 %)** | 4.78 ms (−4.8 %) |
| 41 MB         | gloo | 386 ms  | 363 ms (−6.0 %) | **368 ms (−4.8 %)** | 362 ms (−6.4 %) |
| 253 MB        | gloo | 2589 ms | 2686 ms (+3.7 %) | **2651 ms (+2.4 %)** | 2712 ms (+4.8 %) |

**Three findings the owner should have.**

1. **The task's premise is wrong on this stack, and it matters.**  "Two SCALAR
   all-reduces against a 253 MB reduce-scatter is microseconds."  The scalar
   all-reduce is *not* where the money goes — the design needs only ONE of
   them anyway, and it costs ~0.5 ms (milliseconds, not microseconds, on this
   fabric).  The cost is the **Freivalds contraction**, a memory-bandwidth
   sweep over the operand: at 41 MB the absolute overhead is +34 ms on mpi and
   +38 ms on gloo — nearly identical, which is what pins it on compute rather
   than communication.

2. **The verdict inverts by transport, against intuition.**  On **gloo** (slow,
   TCP) the check is within noise at every payload (−6 % to +5 %) and is
   shippable as-is.  On **mpi** (fast, verbs) the reduce-scatter is cheap
   enough that a pass over its operand doubles it.  Since the direction of
   travel is BSE and GW moving to mpi, "always on, everywhere" buys a ~2x tax
   on the reduce-scatter chain **precisely on the transport that does not have
   the bug**.

3. **At the payload BSE actually issues, the shipped form is free** (+0.7 % mpi,
   +1.3 % gloo).  The +107 % row is a 41 MB collective the BSE matvec does not
   issue.  What it costs in a real run:

   | path | executions / run | overhead | share of run |
   |---|---|---|---|
   | BSE solve (2/matvec x 20 iters) | 40 | ~0.2 ms total | < 0.01 % of a 21 s solve |
   | Sigma tau-projection (2-4 per tau node, O(100) nodes) | ~200-400 | ~9-19 s at the 41 MB row | **real — needs the levers or a decision** |

## 2.4 Levers, if the Sigma path wants it cheaper

Measured in job **7880840**:

1. **Drop the L1-scale pass** — DONE, it is what the shipped form does.  This
   alone took the BSE payload from +65 % to +0.7 % and 41 MB from +157 % to
   +107 %.  The `jnp.abs(x)` in the first form materialised a second full-size
   complex array; that was the whole overhead at small payload.
2. **Sub-sample each segment** (`factor=16`: contract the first 1/16 of every
   segment).  41 MB: +107 % → **+48 %**.  253 MB: +56 % → **+32 %**.  The
   identity holds on any index subset, so this is still an EXACT check — of a
   subset — and it is legitimate here specifically because the measured fault
   is a whole-SEGMENT fault, so it lands inside the sampled block by
   construction.  It would miss a fault confined to the unsampled tail; that is
   the trade being priced.  (It does not scale as 1/16 because the strided
   reshape+slice materialises a copy.)
3. **Check every N-th execution** — NOT measured, but free to implement (a
   counter in the scan carry).  With ~5 % per-execution corruption and hundreds
   of executions per run, checking 1 in 10 still gives ~99 % per-run detection
   at 1/10 the cost.  Best cost/coverage ratio of the three, and it composes
   with the other two.  If the Sigma path is wired at all, this is the lever to
   use there.

**Recommendation to the owner:** wire the BSE sites (`bse_stack_matvec.py`,
`bse_ring_comm.py`) unconditionally — measured free at their payload — and put
the Sigma/ISDF sites behind lever 3 at a rate the owner picks, or leave them
unwired and rely on the α-invariant of TASK 1, which is genuinely free and
already covers the BSE path end-to-end.

## 2.5 What is NOT done, and why

**No call site is wired.**  Three reasons, in order:

1. The owner asked for "the design with a measured cost **before** wiring all
   15", and the measured cost is not free (§2.3).  This is the decision point.
2. **Ten of the fifteen sites are in files another workstream owns.**
   `contract_bands.py` (8 sites: 584, 597, 609, 614, 621, 626, 632, 637) and
   `zeta_projection.py` (2 sites: 502, 505) are being edited by the concurrent
   all-MPI migration.  Wiring them now would conflict.  The five I *could*
   have touched — `isdf/core.py:1947`, `bse_stack_matvec.py:126,129`,
   `bse_ring_comm.py:274,836,966` (that is six; the site list in the task
   totals 16, not 15) — are exactly the "per-call-site carve-out" the owner
   rejected, so wiring a subset would be worse than wiring none.
3. Wiring is **not** a one-line substitution.  `psum_scatter_checked` returns a
   pair, and every one of the sites is inside a `shard_map` body (most inside a
   `lax.scan` inside a `shard_map`), so the residual has to be threaded out
   through the body's carry and the `out_specs`.  That is ~3 mechanical lines
   per site plus one `report_collective_residual` per coarse boundary.  There
   is no zero-plumbing alternative: a value cannot escape a `shard_map` body
   any other way.  (A `jax.debug.callback` *inside* the body would avoid the
   plumbing, but it puts a host round trip in the hottest scan in the codebase,
   which is a worse trade than 3 lines per site.)

**Also not established:** the checker has only been exercised on a
2-rank replica group (`P=4`, `2x2`) with `scatter_dimension=2` and
`tiled=True`.  `tiled=False` is refused explicitly rather than silently
mis-indexed.  Wider replica groups (`p_y=8` at P=64) are untested.

---

# Provenance

## Working-tree state — NOT COMMITTED

`/work2/08271/jackmc/frontera/lorrax` on `fix/zq-band-gather-device-invariance`,
`git HEAD = 6c7feb0`:

```
M src/solvers/lanczos.py     TASK 1 — the alpha-Hermiticity invariant, 5 sites
M src/common/sanity.py       report_hermitian_residual factored out of check_hermitian
M src/common/collectives.py  TASK 2 — psum_scatter_checked (NO call site wired)
```

Everything else in `git status` belongs to other workstreams and was not
touched: `manual/05_isdf/5.1_pair_density_factorization.md` (pre-existing),
`config/frontera/README.md`, `docs/dev/env_vars.md`,
`docs/dev/mpi_collectives.md`, `config/frontera/build_mpiwrapper.sh`,
`config/frontera/mpiwrapper/` (the all-MPI migration workstream, which appeared
in the tree during this session).

**Coordination with the concurrent all-MPI workstream: no collision.**  It owns
`runtime/__init__.py`, `contract_bands.py` and `zeta_projection.py`; none of the
three is touched here.  That is also the main reason TASK 2's fifteen call sites
are NOT wired — ten of them live in `contract_bands.py` and
`zeta_projection.py` (§2.5).

## Snapshots (frozen, manifest-verified; do not delete)

| id | role |
|---|---|
| `srcsnap_collinv_20260729_200919_6c7feb0` | **FIX-A** — the tree as gated by jobs 7880807/7880808/7880813/7880826/7880831 |
| `srcsnap_collinv_20260729_200919_BASE_6c7feb0` | **BASE** — identical to FIX-A except `sanity.py` and `lanczos.py` reverted to `git HEAD` |
| `srcsnap_collinv_20260729_200919_FAULT_6c7feb0` | **FAULT** — FIX-A + the gated segment-0 injection in `bse_stack_matvec.py`.  Detector-proof cell ONLY; never shipped |
| `srcsnap_collinvB_20260729_204749_6c7feb0` | **FIX-B** — the delivered tree |

All 351/351 files hashed, zero cached bytecode, `PYTHONDONTWRITEBYTECODE=1`
exported by `srcpin_resolve`, and every job re-verified its manifest at END as
well as START ("no file changed AND none appeared").

**FIX-A → FIX-B is a message-text change plus the TASK 2 primitive.**  In
`lanczos.py` the diff is the module docstring, the `_ALPHA_FORMS` table and the
`form` argument that threads it — so the block variants' failure message says
"α_j = Q_jᴴ H Q_j is HERMITIAN … antihermitian part" instead of the scalar
variants' "⟨q,Hq⟩ is REAL … imaginary part".  No arithmetic, no tolerance, no
control flow.  `collectives.py` changed because the measured cost (§2.3) showed
the L1-scale pass WAS the overhead; the primitive is not on any import path a
gate exercises.  FIX-B was re-gated by job 7880915 (all five variants, floor +
sensitivity + no-false-positive + cost).

## Jobs

| job | what | state |
|---|---|---|
| 7880807 | `COLLINV_mpi` — clean run, value parity, in-situ fault sweep E1/E2 | COMPLETED 0:0, 5:48 |
| 7880808 | `COLLINV_gloo` — susceptibility + payload ladder + 20 BSE, node pair NOT susceptible | COMPLETED 0:0, 20:10 |
| 7880813 / 7880826 | `COLLINV_probe` — P=1 floor / sensitivity / cost, FIX-A | 7880826 COMPLETED 0:0 |
| 7880825 / 7880840 | `COLLINV_task2` / `_t2b` — TASK 2 cost, both transports, three formulations | COMPLETED 0:0 |
| 7880831 | `COLLINV_hlo` — BASE vs FIX optimized-HLO collective table | COMPLETED 0:0, 1:39 |
| 7880905 | `COLLINV_t2c` — TASK 2 detection sweep, shipped formulation | COMPLETED 0:0, 3:04 |
| 7880909 | `COLLINV_g5` — susceptibility + payload ladder + 20 BSE + warm A/B on a SUSCEPTIBLE pair | COMPLETED 0:0, 18:19 |
| 7880915 | `COLLINV_probe` — re-gate of FIX-B, all 5 variants | COMPLETED 0:0, 3:39 |
| 7880939 | `COLLINV_scan` — detector inside `lax.scan` (the `exciton_bands` shape) | COMPLETED 0:0 |
| 7880935 / 7880941 | `COLLINV_pyt` / `_pab` — regression suite, and its BASE-vs-FIX failure-set A/B | see §1.6a |
| 7880829 | `COLLINV_gloo2` — VOID (harness bug killed the BSE cell), but its susceptibility cells found the c209-003/004 pair | FAILED 1:0 |
| 7880819 | `COLLINV_task2` — VOID (`ConcretizationTypeError` in the first checksum implementation) | — |
| 7880806 / 7880800 / 7880799 | earlier probe iterations, superseded | — |

Gate-instrument defects found and recorded so the next reader is not misled:

1. **A Python call counter cannot inject a fault into a traced solver.**
   `lax.fori_loop` traces the matvec ONCE, so the counter never advances and
   the injection silently does nothing.  Job 7880806's sensitivity table showed
   three blank rows for exactly this reason and it looked like a detector
   failure.  Use a hash of the data.
2. **`int(|Re Σ out| · 1e13)` is a badly-conditioned hash.**  Past `2**53` the
   float64 spacing exceeds 1 and `% every` becomes biased; it blanked one
   variant's whole row in jobs 7880806/7880813.  Take the fractional part
   first.
3. **A 1/8 trigger over 12 iterations misses 20 % of the time**, and once the
   first trigger misses the trajectory stays clean, so the ENTIRE row goes
   blank rather than just one cell.  Sweep magnitude at `every=1` and test
   sporadicity separately.
4. **The first cell of a job pays the XLA compile.**  Job 7880807's BASE leg
   ran first (82 s) and its FIX leg second (21 s); that pair is NOT a cost
   measurement and must not be quoted as one.  §1.6's warm A/B exists because
   of this.
5. **`set -u` + a stray positional in an `echo`** killed job 7880829's BSE cell
   after its susceptibility cells had already found two real corruptions.
6. The standalone bench exits `rc=1` on `Attempting to use an MPI routine after
   finalizing MPICH` — an atexit artefact of not having LORRAX's
   `LORRAX_MPI_FINALIZE_FIX`.  The measurements above it are complete and
   valid; the rc is not.
7. **`pytest | tail` reports `tail`'s exit code.**  Job 7880935 printed
   `PYTEST_RC=0` directly under `1 failed`.  Same class as
   `OWNER_DECISIONS.md`'s "l7 discards its rc through a pipe"; job 7880941
   redirects to a file and captures the rc without a pipe.
8. **`pytest -x` hides the failure SET.**  With `-x` the run stops at the first
   failure, so it cannot answer "is this failure pre-existing" — that needs the
   whole set on both legs.  Job 7880941 drops `-x`.

## Artifacts

| file | what |
|---|---|
| `wk_REL/probes/lanczos_herm_probe.py` | P=1 unit gate: floor / sensitivity / false-positive / cost, all 5 variants |
| `wk_REL/probes/psum_scatter_checked_bench.py` | TASK 2 standalone prototype + cost + detection bench |
| `wk_REL/harness/collinv_probe.sbatch` | the P=1 gate harness |
| `wk_REL/harness/collinv_mpi.sbatch` | clean run + value parity + in-situ fault sweep |
| `wk_REL/harness/collinv_gloo.sbatch`, `collinv_g4.sbatch`, `collinv_g5.sbatch` | susceptibility + payload ladder + BSE hunt + warm A/B |
| `wk_REL/harness/collinv_hlo.sbatch` | the zero-extra-collectives gate |
| `wk_REL/harness/collinv_task2.sbatch`, `collinv_task2b.sbatch`, `collinv_t2c.sbatch` | TASK 2 cost (3 formulations) and detection sweep |
| `wk_REL/probes/lanczos_scan_probe.py`, `collinv_scan.sbatch` | the `exciton_bands` `lax.scan` call-shape regression probe |
| `wk_REL/harness/collinv_pytest.sbatch`, `collinv_pytest_ab.sbatch` | regression suite, and the BASE-vs-FIX failure-set A/B |
| `wk_REL/harness/collinv_gloo2.sbatch`, `collinv_gloo3.sbatch` | superseded harness iterations, kept for the job trail |

---

# SUMMARY — what is established, what is not, what the owner must decide

## Established

1. **TASK 1 landed and is free.**  All five Lanczos α sites carry the
   Hermitian-form invariant.  Zero extra collectives (HLO-verified, job
   7880831), no measurable time (three P=1 runs + a warm P=4 A/B on the real
   BSE), byte-identical eigenvalues everywhere.  It is therefore always-on, not
   gated.
2. **The detector fires, demonstrated on the real BSE matvec at P=4** (job
   7880807).  Linear response over eight decades, `residual = 3.7e-3 x rel`,
   detection threshold `rel = 2.7e-7`, against a real corruption of
   `rel ~ 0.24` — a margin of `~1e6`.  A single corrupted execution inside an
   otherwise clean 20-iteration solve is caught with 4-5 orders of margin.
3. **The tolerance is derived from the arithmetic, not fitted.**
   `(n + c_H)·u/θ ≈ 2.7e-11` at the largest production shape; `1e-9` is that
   rounded up.  Measured floors are 4e-17 (P=1) to 7.2e-16 (real BSE at P=4).
   Any tolerance in `[1e-11, 1e-4]` gives the same verdict on every case
   measured.
4. **The BSE payload does not corrupt.**  0 events in 876 gloo reduce-scatter
   executions at ≤ 41 MB on nodes that corrupted 2 of 2 processes at 253 MB in
   the same allocation.  Rate at the BSE payload **< 0.34 % (95 %)**.  This
   downgrades the standing assumption that BSE is exposed.
5. **No regression.**  BASE-vs-FIX failure-set A/B over the eight test files
   that can reach the changed code: **identical**, 15 failed / 72 passed on
   both legs (job 7880945).  The tree at `6c7feb0` is already red in those 15
   places; this change moves none of them.  The three direct `check_hermitian`
   assertions pass on both legs.
6. **TASK 2's primitive works** and its cost is measured on both transports at
   three payloads.  In the shipped formulation it is **free at the BSE payload**
   (+0.7 % mpi, +1.3 % gloo) and **not free at Sigma-class payloads**
   (+107 % at 41 MB on mpi).  The cost is the Freivalds contraction — a
   memory-bandwidth sweep — not the scalar all-reduce.

## Not established

* Where between 41 MB and 253 MB the corruption's payload floor sits, whether
  it is sharp, and whether it holds at other P / mesh shape / replica-group
  width.  Upstream §6.3 stands; this narrows it.
* Whether `psum_scatter_checked` behaves at replica groups wider than 2
  (`p_y = 8` at P=64) — untested.  `tiled=False` is refused rather than
  silently mis-indexed.
* Whether the α-invariant's floor holds at the 10015-centroid BSE shape.  The
  derivation says `~1e-12`; only the 785c shape (`7.2e-16`) was measured.

## Owner decisions

1. **TASK 2 wiring.**  Recommended: wire the BSE sites unconditionally (free at
   their payload); leave the Sigma/ISDF sites unwired, or wire them behind the
   every-N-th-execution lever (§2.4).  Ten of the fifteen sites are in the
   all-MPI workstream's files, so wiring must be sequenced with it and should
   land as ONE change.
2. **Whether the < 0.34 % BSE-payload bound changes the release posture.**
   `OWNER_DECISIONS.md` currently records "BSE therefore runs on a transport
   with measured silent corruption" as part of the release blocker.  That
   sentence is now too strong for the payload BSE actually issues.
3. **Whether the same α-style free invariant should be hunted elsewhere.**  The
   Σ path has no equivalent — but `ppm_sigma.py` already builds Hermitian
   `B_q`/`Ω_q`, and `check_hermitian` is called on them at CONSTRUCTION
   (`:279,281`) rather than after the τ-projection's reduce-scatter chain.
   Moving/duplicating that one call downstream is the same trick, and it is the
   obvious next free invariant.
