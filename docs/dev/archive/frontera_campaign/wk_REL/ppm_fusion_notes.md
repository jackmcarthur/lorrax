# PPM fit — the fusion barrier (R37), 2026-07-29

Workstream: sigma-memory, resumed after the R34/R36 stand-down.
Repo `/work2/08271/jackmc/frontera/lorrax`, worktree **wt-RELC**, branch
`wsREL-isdf-window` @ **88010ac**, working tree clean at start.  NOT COMMITTED.

---

## 0. ONE-LINE RESULT

**There is no fusion barrier.  The guard chain fuses completely, and always
did.  The 74.27 GiB was never 32 temporaries — it was ONE buffer: the
replicated, full-GLOBAL-shape `f64[nq, mu_pad, mu_pad]` materialisation of the
constant mode-count mask, built only to add up its own ones.  Deleting that one
line (the value is an exact compile-time integer) takes the module from 35.563
to 4.063 live tiles at BOTH sizes, and removes 74.27 GiB at production.**

---

## 1. WHAT THE PREVIOUS SESSION'S PREMISE GOT WRONG

R32/R33/R34/R36 hand this workstream three claims, all now falsified from the
same HLO those sections cite:

| claim (R32.2 / R33.2 / R36) | truth (measured) |
|---|---|
| "111 top-level full-tile instructions, only 13-14 fusions" | the ENTRY computation has **7** full-tile instructions: 2 parameters + **5 kLoop fusions**.  The other 66 are INSIDE `%fused_computation` bodies. |
| "42 guard-related full-tile ops sitting OUTSIDE any fusion" | **ZERO** full-tile elementwise ops sit outside a fusion.  Every `and`/`compare`/`select`/`real`/`abs`/`is-finite` is inside a fusion body and never materialises. |
| "the arena is 32 live tiles; ~6 genuine, ~26 unfused artefacts" | the arena is **1 buffer**.  There are ~4 live tile-equivalents total, and they are all genuine (2 params + 2 outputs' worth). |

**Root cause of the error: the census counted instructions in fusion bodies as
"top level" and as "outside any fusion".**  An op inside `%fused_computation`
is exactly the op that does NOT get a buffer — it is the fused case.  Counting
those as evidence of non-fusion inverted the diagnosis, and the whole
"attack the fusion barrier" framing (including this task's own brief) descends
from it.

This also explains R34's zero-byte negative result honestly: cutting 46
instructions moved nothing because **none of those instructions had a buffer to
begin with.**  R34's conclusion ("op-count is the wrong lever") was right; its
reason ("the guard chain will not fuse") was wrong.

## 2. WHAT THE OBJECT ACTUALLY IS

Straight from the reference run's own dump
(`run_PPMFIT2_ref_default/hlo_dump/module_0873.jit__gn_ppm_fit_kernel.
cpu_after_optimizations{,-memory-usage-report}.txt`):

    allocation 33: size 95.06MiB, preallocated-temp:
       95.06MiB(100%);  95.06MiB;  offset 0;  3 values;
       f64[2,2496,2496], f64[2,312,312], f64[]
    ...everything else in that arena: 800 B + 312 B + 72 B.

    %fused_computation () -> f64[2,2496,2496] {
      %iota.7   = s64[2496] iota()
      %lt.12    = pred[2496] compare(%iota.7, 2475), direction=LT
      %and.67   = pred[2496,2496] and(broadcast dim0, broadcast dim1)
      %convert_element_type.10 = f64[2496,2496] convert(%and.67)
      ROOT %broadcast.7 = f64[2,2496,2496] broadcast(...), dimensions={1,2}
    }
    %wrapped_reduce-window.1 -> ... -> %wrapped_reduce.1 = f64[]   (NO all-reduce)

Note the shapes: `2496`, not `312`.  **That branch is UNSHARDED.**  `mode_mask`
is built from `jnp.arange(n_mu)`, which carries no sharding, and its consumer
is a reduce to a scalar, so GSPMD kept the whole thing REPLICATED — every one
of the 64 ranks materialises the entire global mask.  (The *other* consumer of
`mode_mask`, inside `good`, DID get sharded: `%fused_computation.4` rebuilds it
locally from `partition-id`.  So the sharded lowering exists; this branch just
didn't take it.)

Source line, `src/gw/minimax_screening.py:571-572` @ 88010ac:

    m = jnp.broadcast_to(mode_mask, good.shape)
    n_modes = jnp.sum(m.astype(jnp.float64))

### 2.1 the arithmetic, which closes it

    reference : 2 * 2496 * 2496 * 8      =         99,680,256 B = 95.06 MiB
    production: 16 * 24960 * 24960 * 8   =     79,744,204,800 B = 74.27 GiB

**79,744,204,800 is byte-for-byte the allocation in
`Out of memory allocating 79744204800 bytes`** (jobs 7879469 / 7879487).
The arena was never a 32-tile scratch pool; it was this single array.

**And the famous "exactly 32.0x one tile, at two very different sizes" is an
identity, not a measurement of temporaries:**

    arena / tile = (mu_pad / mu_local)^2 * (8 bytes f64 / 16 bytes c128)
                 = p_x^2 / 2 = 8^2 / 2 = 32

It read 32.0 twice because both measurements were at **P=64 on an 8x8 mesh**.
At P=256 (16x16) the same code would have read exactly 128.0.  The constancy
that was taken as proof of a structural instruction-chain property was proof of
a fixed mesh.

## 3. THE FIX (occam's razor: the value is a CONSTANT)

`mode_mask` is an outer AND of `arange(n_mu) < n_log` with itself, so it has
exactly `n_log**2` true entries; broadcast over the leading axes the sum is
exactly `prod(lead) * n_log**2`.  Production is ~1e10, far below 2**53, and
summing 0.0/1.0 in float64 is EXACT while every partial sum stays under 2**53 —
so the reduction's result IS that integer, and emitting the integer is
**bit-identical**, not merely mathematically equal.  (The `>= 2**53` branch is
kept, unreachable, so the exactness argument can never be silently violated.)

    _n_lead = prod(good.shape[:-2])
    _n_modes_exact = _n_lead * n_log * n_log
    n_modes = jnp.asarray(float(_n_modes_exact), dtype=jnp.float64)

**No guard is touched.**  `mode_mask` still gates `good`, still zeroes pad modes
at birth, and `safe`/`isfinite`/`omega_sq_re>0` are untouched.  The deleted code
computed no guard — it counted how many modes a guard *could* apply to.

## 4. MEASUREMENT — compile-only A/B, job 7880762 (1 node, small, 3 min)

`wk_REL/probes/ppmfus_probe.py` AOT-compiles the kernel on 64 fake CPU devices with the
production 8x8 mesh and reads `compiled.memory_analysis()`.  Nothing runs and
nothing is allocated, so the 74 GiB object is measured on one node in seconds
instead of reproduced by OOMing 32.

**Instrument validated against the real runs:** probe OLD reference temp =
99,680,256 B vs the run's `allocation 33: size 95.06MiB` (exact); probe OLD
production temp = 79,744,204,800 B vs the OOM byte count (exact); module totals
110,777,552 / 88,621,977,680 vs the dumps' 110,777,984 / 88,621,978,120 (432 /
440 B of bookkeeping).

### live tiles (the owner's metric: bytes / one local c128 tile)

| | reference mu=2,475 | | production mu=24,933 | |
|---|---|---|---|---|
| | **OLD** | **NEW** | **OLD** | **NEW** |
| TEMP (the "arena") | **32.000** | **0.500** | **32.000** | **0.500** |
| args | 2.000 | 2.000 | 2.000 | 2.000 |
| outputs | 1.563 | 1.563 | 1.563 | 1.563 |
| **TOTAL module** | **35.563** | **4.063** | **35.563** | **4.063** |

### absolute bytes

| | reference | production |
|---|---|---|
| local c128 tile | 3,115,008 B (2.97 MiB) | 2,492,006,400 B (2.32 GiB) |
| TEMP  OLD -> NEW | 99,680,256 -> 1,558,304 B | **79,744,204,800 -> 1,246,080,032 B** |
| TOTAL OLD -> NEW | 110,777,552 -> 12,655,600 B | **88,621,977,680 -> 10,123,852,912 B** |
| | 105.65 -> 12.07 MiB | **82.54 -> 9.43 GiB** |

**Temp arena 64x smaller; whole module 8.75x smaller; 35.563 -> 4.063 live
tiles — the owner's "fewer than 10" target is met with margin, at BOTH sizes.**
The ratio is identical at the two sizes because, as in section 2.1, it is
structural.

### fusion census (unchanged where it matters)

| | ref OLD | ref NEW | prod OLD | prod NEW |
|---|---|---|---|---|
| kLoop fusions | 12 | 8 | 13 | 9 |
| full-tile ops at ENTRY | 7 (2 param + 5 fusion) | 7 | 7 | 7 |
| full-tile ops in fusion BODIES | 66 | 66 | 66 | 66 |

The four fusions that disappear are the mask broadcast plus its three reduce
wrappers — a whole subgraph deleted, not fusion quality changing.  The guard
chain's fusion structure is **byte-identical before and after**: XLA:CPU
duplicates it into three fusion bodies (one per consumer: the `good` pred
output, the `good`->f64 count operand, and the `omega_vals` select), which is
the memory-optimal choice and is why no guard intermediate ever gets a buffer.

## 5. BIT-EXACTNESS — 200 randomised trials, both directions

`wk_REL/probes/ppmfus_equiv.py`, legs A and B of job 7880762.  All FIVE outputs
(`omega_vals`, `B_vals`, `good`, `n_good`, `n_modes`) compared by **bit
pattern** (`view(uint64)`, NaN payloads included), not numeric equality — the
standard R36 §4 insists on.

Edge cases actually hit per leg: `denom` exactly 0 (200), 1e-20 unsafe (200),
1e-13 safe (400), **exactly 1e-14 — ON the `> 1e-14` boundary** (400), sign
flips forcing `omega_sq_re < 0` (200), NaN in (29), +/-Inf in (29), padded
extents (140), unpadded `n_log == n_mu` (40), `n_log == 1` (9).

    [equiv OLD] BIT-EXACT MISMATCHES: 0  -> EQUIVALENT
    [equiv NEW] BIT-EXACT MISMATCHES: 0  -> EQUIVALENT

Leg A is the anti-strawman control: run under the OLD snapshot, the imported
kernel must equal the inline transcription used as reference.  It does.  Leg B
is the claim.  A separate assertion inside every trial checked the reduction's
own value against `nq * n_log**2` directly; it never fired.

## 6. PRODUCTION GATES

`wk_REL/harness/ppmfus_gate.sbatch` (see §8 for why it is a copy, not the stock file),
both against the pinned reference eqp0 3.5819 / eqp1 3.2516, tol 1e-3 eV:

| job | path | gw | eqp0 | eqp1 | verdict |
|---|---|---|---|---|---|
| **7880764** | default (deck takes q_block = nq: single-shot) | rc=0, 236 s | 3.5819, **\|d\| = 0.00e+00 eV** | 3.2516, **\|d\| = 0.00e+00 eV** | **PASS** |
| **7880765** | `LORRAX_PPM_FIT_ARENA_GIB=0.01` -> **forced q_block=1** | rc=0, 238 s | 3.5819, **\|d\| = 0.00e+00 eV** | 3.2516, **\|d\| = 0.00e+00 eV** | **PASS** |

Both `COMPLETED 0:0`, `gate_rc=0`, `MANIFEST VERIFIED at END`, VmHWM 8.91 GiB
(unchanged vs rung 1).  (`sacct` step `.1` FAILED 1:0 on 7880764 is the known
colltable-exits-nonzero-on-FLAG behaviour, not a run failure.)

**The forced-chunk leg is proven to have actually chunked**, not silently taken
the default: its dumped module is `c128[1,312,312]` — one q per call.

### 6.1 value parity is FULL BYTE PARITY, not just the gap

Data bytes of every output file, compared across the pre-change reference run
`run_PPMFIT2_ref_default` (OLD), and both new runs.  Only the
`# Generated by LORRAX ... at <timestamp>` header line differs anywhere:

| file | OLD | NEW single-shot | NEW q_block=1 |
|---|---|---|---|
| `eqp0.dat` | `e40759b8b34c164e` | = | = |
| `eqp1.dat` | `3626ca1306dd0b6d` | = | = |
| `eqp_g0w0.dat` | `fbfddd34bdb3e414` | = | = |
| `sigma_diag.dat` | `259f7144999d5cf4` | = | = |

(md5, header line dropped.)  This is stronger than the gate's 1e-3 eV
tolerance: **every digit of every value in all four files is identical.**

### 6.2 the arena removal, confirmed END-TO-END in the production runs

Not the compile-only probe — the gate runs' own rank-0 dumps,
`run_PPMFUS_ref_*/hlo_dump/module_0873.jit__gn_ppm_fit_kernel.
cpu_after_optimizations-memory-usage-report.txt`:

| run | Total bytes used | largest preallocated-temp |
|---|---|---|
| OLD `run_PPMFIT2_ref_default` | 110,777,984 (105.65 MiB) | **95.06 MiB** |
| NEW `run_PPMFUS_ref_default` | **12,655,968 (12.07 MiB)** | **1.49 MiB** |
| NEW `run_PPMFUS_ref_forcedchunk` | 6,328,608 (6.04 MiB) | 761.3 KiB |

12,655,968 measured vs 12,655,600 predicted by the compile-only probe — 368 B.
**4.063 live tiles, exactly as predicted.**

### 6.3 snapshot discipline held

Gated snapshot carries **0 `.pyc`** after two 32-node runs and its manifest
still verifies post-hoc — contrast the 114 `.pyc` in the predecessor's
snapshot (§8).

## 7. WHAT WAS NOT CHANGED, AND WHY

- **Every guard.**  `safe`, `isfinite`, `omega_sq_re > 0`, `mode_mask`, `good`
  are byte-identical in the HLO before and after.
- **`|denom| > 1e-14`.**  Still REJECTED as R36 §4 says: `re^2+im^2 > 1e-28` is
  mathematically equal but can move elements across the threshold.  Held.
- **`_GN_PPM_FIT_LIVE_TILES = 32`.**  Now measured to be ~4, i.e. the sizer
  over-chunks by ~8x.  Left at 32 ON PURPOSE: campaign doctrine (R5, R30.3) is
  that a sizer reading HIGH is safe and one reading LOW is not, and lowering it
  changes the production chunk plan, which is a separately-gated change rather
  than a comment edit.  Comment updated to say so.  It does not touch the
  reference deck, which takes the single-shot path either way.
- **q-chunking.**  Kept.  It is landed and gated, it is now nearly always a
  no-op (the object it was shrinking is gone), and removing it would be an
  unforced ungated change.  Open item O1 (its silent floor) still stands.
- **The `good`->f64 count tile** (`%and_convert_fusion`, f64[nq,mu,mu]) is now
  the largest temp: 0.5 tile, 1.19 GiB at production.  XLA:CPU wraps reduces in
  their own kernels and will not fuse a producer into them, so an f64 operand
  gets materialised.  Halving it (reduce a narrower integer) is worth ~0.25
  tile and was NOT attempted — 4.063 is already inside the target and the
  change would need its own exactness argument.

## 8. INSTRUMENT DEFECT FOUND (and avoided here)

The stock `mos2_4x4_test/gate_sigma_reference.sbatch` hardcodes
`SRCPIN_COMMIT=4f77842` **in its provenance banner** while letting `SRCDIR`
point anywhere, and its `inner.sh` does not set `PYTHONDONTWRITEBYTECODE`.
Consequence, verified on disk: `srcsnap_ppmfit2_20260729_090947_ec96ba9`
contains **114 `.pyc` files** written by the run that used it — the snapshot
mutated at run time and `srcpin_verify_end` would have passed vacuously over
executable bytecode.  This is precisely the failure mode `srcpin_resolve.sh`
documents.

`wk_REL/harness/ppmfus_gate.sbatch` is a copy with three fixes: EXPLICIT `$SRCSNAP` with
manifest verification at START **and END**, banner derived from the snapshot
itself, and `PYTHONDONTWRITEBYTECODE=1` in the inner script.  Both snapshots
used here verified clean at both ends of job 7880762 and carry 0 `.pyc`.

## 9. ARTIFACTS

    wk_REL/probes/ppmfus_probe.py            compile-only arena + census instrument
    wk_REL/probes/ppmfus_equiv.py            200-trial bit-exact A/B
    wk_REL/harness/ppmfus_probe.sbatch        job 7880762 (1 node, small, 3 min)
    wk_REL/harness/ppmfus_gate.sbatch         provenance-fixed reference gate
    wk_REL/results/logs/ppmfus_probe.7880762.out   all four legs
    wk_REL/srcsnap_ppmfus_OLD_20260729_192257_88010ac   clean 88010ac
    wk_REL/srcsnap_ppmfus_NEW_20260729_192449_88010ac   + the fix (probed)
    wk_REL/srcsnap_ppmfus_NEW2_20260729_192726_88010ac  + comments (gated;
        executable AST verified IDENTICAL to NEW, so the probe numbers carry)

Change: `src/gw/minimax_screening.py` only, +54/-4, UNCOMMITTED in wt-RELC.

## 10. CONSEQUENCE FOR THE FRONTIER

The certified wall "usable mu < 24,933 at P=64" was set by this buffer.  At
mu=24,933 the kernel's per-rank footprint goes **82.54 GiB -> 9.43 GiB**
against 93.0 GiB/rank available; the measured VmHWM at the OOM was 99.05 GiB.
The fit kernel is no longer the binder.  **This is a prediction, not a result:
the mu=24,933 demonstration (open item O3) has still never run** — R36's O3 was
never executed and nothing here changes that.  The next rung should re-measure
the wall rather than assume it moved by exactly 73 GiB.
