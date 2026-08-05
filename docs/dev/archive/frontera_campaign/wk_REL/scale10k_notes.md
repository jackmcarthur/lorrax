# wk_REL — FULL-PIPELINE AT 10k CENTROIDS: end-to-end, timing, antipattern audit

Owner-commissioned workstream, 2026-07-29.  Tree `/work2/08271/jackmc/frontera/lorrax`
@ **1a52d51**; source pin `wk_REL/snapshots/srcpin_1a52d51/` (read-only `git archive`, used as
PYTHONPATH by every job here).  Agent worktree for edits:
`/work2/08271/jackmc/frontera/wt-REL10k` (branch `wt-REL10k-scale`), **NOT committed**.

Size point: MoS2 4x4 / 30 Ry / **nband = 1024**, **N_mu = 10015**, P = 64 (8x8 mesh,
32 nodes x 2 ranks), account PHY25006, queues `normal` / `small`.

All numbers below are read off on-disk artifacts with a clock cross-check against
`sacct`.  Job IDs are inline.  Nothing here is taken from a notification.

---

## S0 — THE THREE VERDICTS (headline)

| path | verdict at (nb=1024, mu=10015, P=64) | evidence |
|---|---|---|
| **kmeans / centroids** | **GREEN single-process; RED at EVERY P>1.**  The accepted 10015-point set was produced on ONE rank (mesh 1x1) in 308 s, VmHWM 63.15 GiB.  The same invocation SIGSEGVs (rc=139) at P=64 **and at P=4**, in an eager unsharded `jnp.fft.ifftn` in the charge-density build (`psp/get_DFT_mtxels.py:196`) — never reaching the prune, its Gram, or its divisibility refusal.  S4.1 / S4.1a. | jobs 7879286 (1 node, accepted), 7879470 (P=64 rc=139), 7879492 (P=4 rc=139), 7879495 (faulthandler + `--no-orbit`, both rc=139, same frame) |
| **gw_jax** | **GREEN** (already proven).  1811 s wall, 1485.9 s recorded, VmHWM 36.03 GiB/rank. | job 7879295 |
| **BSE** | **GREEN at 10k — first BSE run of this campaign.**  Blocked first by the AS.7 MPI-collectives cell (S3.1); runs to completion on the certified gloo/ib0 default, and escalates 96x in BSE dimension (1 024 -> 98 304) without a memory or refusal wall.  One real bug found and fixed on the way (the eigenvector writer, S4.8/S6.3). | jobs 7879458 (mpi, RED), 7879463 (gloo P=4, rc=0), 7879470 (10k P=64 ladder: rc 0/1/0) |

BSE had **never been executed** anywhere in this campaign before today: a grep of every
`*.sbatch`/`*.sh` under `/scratch2/08271/jackmc/{lorrax_setup,mos2_4x4_test,lorrax_mos2_12x12}`
returns zero BSE driver invocations.  Prior "BSE" jobs are pytest gates on synthetic
fixtures (`wk_REL/probes/gbp_bse_parity.py`, `gbp_bse_fp32.py`, 4 emulated CPU devices) and the
`wk_BC` `vq_interp` unit runs.  The largest BSE in the project record is a Perlmutter run
(`src/bse/EXCITON_BANDS.md`: 16 x A100, px=py=4, n_mu = 640) on a filesystem that does not
exist here.  So every BSE number below is a first measurement, not a regression check.

---

## S1 — HARNESS (reusable)

- `mos2_4x4_test/bse_inner.sh` — container-side per-rank cell for `bse.bse_jax`.  Env block
  copied VERBATIM from `run_L5_b1024_c10000/inner.sh` (the certified rung-5 GW cell) except
  for `BSE_COLL`; adds VmHWM sampling and rank-0 `--xla_dump_to`.
- `mos2_4x4_test/km_inner.sh` — same cell for `centroid.kmeans_cli` at P>1.
- `mos2_4x4_test/bse_smoke_p4.sbatch` — 2-node P=4 shakeout, `--export=ALL,COLL=mpi|gloo`.
- `mos2_4x4_test/bse_ladder_10k_p64.sbatch` — the 32-node ladder (3 BSE legs + the kmeans
  P=64 leg).
- `mos2_4x4_test/rel10k_gate.sbatch` — A/B gates for the two fixes in S6.

Invocation that works (the `--bse --lanczos --tda` triple is load-bearing — without
`--lanczos` the CLI goes to FEAST, without `--bse` it runs RPA and skips W entirely,
without `--tda` it runs non-TDA):

    python -u -m bse.bse_jax -i gw.in --lanczos --tda --bse \
        --n-val N --n-cond N --n-occ 26 --n-eig K --max-lanczos-iter I --n-reorth -1

Artifacts consumed: `<input_dir>/tmp/isdf_tensors_<mu>.h5` (found by newest mtime) and the
`wfn_file` named in the input.  At the 10k point that is
`run_L5_b1024_c10000/tmp/isdf_tensors_10015.h5`, 56.61 GB, containing
`V_qmunu {16,10015,10015}`, `W0_qmunu {16,10015,10015}` (W0_ready), `psi_full_y
{16,1024,2,10015}`, `enk_full {16,1024}`, `G0_mu_nu`, `vhead`, `whead`, `kgrid`.
The `--matvec-kind` flag documented in `STATUS.md` / `BGW_COMPARE.md` is **stale**:
`solve_bse_sharded` retired the selector and always builds
`bse_stack_matvec.build_bse_stack_matvec`.

---

## S2 — TIMING TABLE (deliverable, task 3)

Wall/stage numbers are from the runs' own `common.timing` reports and `[stage HH:MM:SS]`
lines; memory is `/proc/<pid>/VmHWM` sampled every 10-20 s per rank by the harness
(sacct MaxRSS is NOT used).  **GiB = 2^30 B** throughout.

### 2.1 kmeans / centroid selection — job 7879286, 1 node, **P = 1**, 56 threads

| stage (`timing` row) | s | % of recorded | note |
|---|---:|---:|---|
| setup.wfn_io | 14.27 | 7.4 | |
| setup.charge_density | 1.53 | 0.8 | |
| setup.weight (band_range 0:1024) | 25.81 | 13.5 | 1024-band, 10 k, symmetrized over 12 ops |
| kmeans (Lloyd, 8 steps, 1250 reps) | 20.95 | 10.9 | init 0.95 / lloyd 17.26 / assign 2.26 |
| snap_unfold | 0.12 | 0.1 | 1250 reps -> 13872 centroids |
| **prune** (pivoted Cholesky) | **129.13** | **67.3** | |
| &nbsp;&nbsp;prune.gram | 98.01 | 51.1 | left.load 1.82 + right.load 1.97 + **q0_sum 92.81** |
| &nbsp;&nbsp;prune.select | 26.71 | 13.9 | sharded-select kernel on a 1x1 mesh |
| **recorded total** | **191.81** | 100 | |
| **process wall** | **308** | | **116 s (38 %) UNATTRIBUTED** |
| VmHWM | **63.15 GiB** (66 219 996 kB) | | of 192 GB node |

> **Instrumentation gap (named, not guessed).** 116 s of the 308 s wall carries no
> `timing` row.  `timing.reset()` runs after `import jax`/`bootstrap()`, and nothing wraps
> the interpreter+JAX import, the `--qe-save` charge-density read path setup, the final
> `np.savetxt`, or the 173 XLA compiles (17.58 s of which IS inside the sections).  The
> fix is three `timing.section` scopes in `centroid/kmeans_cli.py::main`:
> `kmeans_cli.bootstrap` (around the module import + `bootstrap()` — needs the section to
> be opened by the module, so realistically a wall-clock stamp at first line of `main`),
> `kmeans_cli.write_output` (around `np.savetxt`), and one `timing.section("prune.select")`
> subdivision splitting compile from execute.  Until then any statement about where those
> 116 s go is a guess and is not made here.

### 2.2 gw_jax — job 7879295, 32 nodes, **P = 64** (8x8), 28 threads/rank, cache-COLD

| stage | s | % recorded | VmHWM checkpoint |
|---|---:|---:|---|
| gw_jax.load_centroid_wfns | 51.19 | 3.4 | 9.50 GiB reached here |
| &nbsp;&nbsp;loader_load / gflat_to_rmu / reshard | 46.87 / 1.95 / 2.25 | | |
| **gw_jax.zeta_fit_chunked** | **358.26** | **24.1** | **36.03 GiB — the run's peak** |
| &nbsp;&nbsp;zeta_fit.cholesky | 240.61 | 16.2 | +25.05 GiB step lands in this window |
| &nbsp;&nbsp;zeta_fit.chunk_loop (z_q_build 67.80 / solve 20.13) | 89.17 | 6.0 | |
| &nbsp;&nbsp;zeta_fit.CCT / close_io / other | 8.58 / 14.48 / 5.4 | | |
| gw_jax.V_q_compute | 5.54 | 0.4 | |
| gw_jax.wavefunction_setup | 0.88 | 0.1 | |
| gw_jax.chi0_W (static) | 75.26 | 5.1 | chi.exec 26.05 / **W.exec 46.26** |
| gw_jax.chi0_W_probe (PPM 2nd frequency) | 89.04 | 6.0 | chi.exec 20.06 / **W.exec 68.56** |
| W.gate (finiteness+hermiticity) | 1.68 | 0.1 | |
| gw_jax.persist_w0 | 27.25 | 1.8 | writes W0_qmunu into the restart |
| **gw_jax.sigma** | **876.79** | **59.0** | |
| &nbsp;&nbsp;sigma.exec | 852.61 | 57.4 | 176 tau, 4.84 s/tau |
| &nbsp;&nbsp;&nbsp;&nbsp;sigma.tau.d2h_wait | 747.70 + 64.50 | 50.3+4.3 | **absorbs device compute — see gap note** |
| &nbsp;&nbsp;&nbsp;&nbsp;sigma.tau.omega_project | 18.41 + 1.43 | 1.2 | pure numpy, small (as at nb=128/256) |
| **recorded total** | **1485.87** | | |
| **job wall (sacct)** | **1811** (00:30:11) | | 325 s = startup + 438 compiles/rank (36.4 s rank 0) + teardown |
| **VmHWM** | **36.03 GiB/rank** (72.1 GiB/node of 192) | | reached in zeta_fit, NEVER exceeded later |

> **Instrumentation gap (named).** `sigma.tau.d2h_wait` = 812.2 s is 95 % of `sigma.exec`
> and is a *waiting* row, not a work row — it absorbs the device kernel through the lag-2
> `np.asarray` drain (this is documented behaviour, `wk_REL/docs/sigma_perf_results.md`).  At
> nb=128 and nb=256 the split was resolved by re-running with **`LORRAX_SIGMA_TAU_TIMING=1`**,
> which turns on the blocking stage rows `sigma.tau.{w_phase,G_build,G_ifft,V_ifft,
> GW_mult_fft,GW_conv_ffi,project_rs}` in `gw/ppm_tau_kernel.py`.  That is the exact
> `trace_section` set that would attribute this 812 s row at nb=1024; it was not re-run
> here (a staged pass is a second 30-min 32-node allocation and the BSE path was the
> owner's first priority).  **Do not attribute the 812 s from the nb=128/256 shares** —
> the shares move with nb (project_rs went 16 % -> 39 % from nb=128 to 256).

> **Contamination in this row set (found, quantified below).** The rung-5 GW inner script
> sets `LORRAX_W_RESIDUAL_CHECK=1`.  That is a *diagnostic* whose own docstring says
> "never on in the traced production path".  Its cost is inside `W.exec` — see S5.1.

### 2.3 BSE — job 7879470, 32 nodes, P = 64 (8x8), gloo/ib0, cache-COLD

`bse.bse_jax` carries **no `timing` instrumentation at all** (S7.6), so the phase split
below is reconstructed from the harness's per-rank `/proc` sampler (10 s cadence, rank 0)
cross-checked against the log's `BSE problem` / `Lowest N eigenvalues` markers.  It is a
bounded attribution, not a `timing` table — stated as such.

**Leg 1 — n_val 8 / n_cond 8 (the smallest legal window at an 8x8 mesh), N_mu = 10015,
n_eig 4, 20 Lanczos iterations, HLO dumped:**

| phase (from the VmHWM/VmRSS trace) | s | % | evidence |
|---|---:|---:|---|
| restart load (`load_bse_data_from_restart_sharded`: W_q, V_q0, psi via **serial h5py**) | **~305** | **67** | RSS climbs 0 -> 0.79 GiB monotonically over t+0..t+305 s, then the staging buffer is released (RSS 0.79 -> 0.44 at t+216 while HWM holds) |
| q=0 head injection + M_X/M_Y pair amplitudes | ~40 | 9 | HWM 0.79 -> 1.19 -> 1.41 GiB at t+307..327 |
| Lanczos `_full_run` (compile + 20 iterations) | ~110 | 24 | HWM steps 1.41 -> **6.08 GiB** at t+347 (the T-tensor chain), flat to t+438 |
| **leg wall** | **458** | | rc=0 |
| **VmHWM** | **6.28 GiB/rank** (12.6 GiB/node of 192) | | |

Physics produced (first exciton spectrum of this campaign at 10k):
`Lowest 4 eigenvalues (eV): [1.32291308 1.4328756 1.62516053 1.84614601]`.
NOTE this rides the rung-5 DFT energies from `enk_full` (no `--eqp`); the rung-5 QP result
is itself RED (R12 prune-window defect), so these are a **scale/perf datum, not physics**.

Memory model check: `T_b (mu_loc, nu_loc, ns, ns, nk)` = 1256 x 1256 x 4 x 16 x 16 B
= 1.615 GiB; ~3 live copies through `ifft -> W_R* -> fft` + W_R (404 MB) + the loaded
tensors (~0.8 GiB) ~= 6.2 GiB.  Matches the measured 6.28 GiB to ~1 %.

**Leg 2 — n_val 24 / n_cond 64 (`64 cond x 24 val x 16 k = 24576 dim`), n_eig 8,
40 iterations, `--write-eigs 8`: rc = 1, 315 s, VmHWM 5.92 GiB/rank.**
The *solve* completed and printed
`Lowest 8 eigenvalues (Ry): [0.1029597 0.12620309 0.15902403 0.19859046 0.26078785
0.33421876 ...]` / `(eV): [1.40083853 1.71708109 2.16363287 2.70196189 3.54820071
4.54727956 ...]` — the failure is in the **writer**, see S4.8.  A structurally complete
`eigenvectors.h5` (3.16 MB, `exciton_data/eigenvectors {1,8,16,64,24,1,2}`) was left on
disk by whichever rank won the race, which is precisely why this class is dangerous.

Note leg 2 is FASTER than leg 1 (315 s vs 458 s) at a 24x larger BSE dimension: the
restart read dominates and is band-window-independent, and leg 2's read hit warm page
cache from leg 1.  Do not read a speed-up into that number.

**Leg 3 — n_val 24 / n_cond 256 (`256 cond x 24 val x 16 k = 98304 dim`), n_eig 8,
40 iterations: rc = 0, 274 s, VmHWM 9.98 GiB/rank.**
`Lowest 8 eigenvalues (eV): [1.44270799 1.91285996 2.60472177 3.64560545 4.94538439
6.56299827 ...]`

### Ladder summary (job 7879470, all at N_mu = 10015, P = 64, gloo/ib0)

| leg | n_val x n_cond | BSE dim | rc | wall (s) | VmHWM (GiB/rank) |
|---|---|---:|---:|---:|---:|
| s8x8 (smallest legal at 8x8) | 8 x 8 | 1 024 | **0** | 458 | 6.28 |
| m24x64 (+ `--write-eigs 8`) | 24 x 64 | 24 576 | **1** (writer race, S4.8) | 315 | 5.92 |
| l24x256 | 24 x 256 | 98 304 | **0** | 274 | 9.98 |

**Verdict: BSE reaches 10k and escalates 96x in BSE dimension without a memory or
refusal wall.**  Nothing in the escalation was memory-bound; the only rc!=0 was the
output-writer race, which is fixed in S6.3.  No leg came within 10x of the node budget.

### 2.3.1 CLOSED-FORM BSE per-rank memory model (reconciled to the byte, 3 points)

    per-rank bytes  ~  [ ~3 x T_b ]                    band-window INDEPENDENT
                     + [ 2 x M    ]                    ∝ nc * nv * mu / p
                     + [ W_R (+ W_q, before FIX B) ]   ∝ mu^2 / p^2
                     + psi / host

with `T_b = (mu_pad/p_x)(mu_pad/p_y) ns^2 nk x 16 B` = 1.615 GiB here and
`M = nk * nc * nv * (mu_pad/p) x 16 B`.

| leg | nc x nv | predicted 2M | measured VmHWM | delta vs leg 1 |
|---|---|---:|---:|---:|
| s8x8 | 8 x 8 | 0.04 GB | 6.28 GiB | — |
| l24x256 | 256 x 24 | **3.95 GB** | **9.97 GiB** | **+3.69 GiB = +3.96 GB** |

The measured growth matches the hoisted exchange pair amplitudes `M_X`, `M_Y`
(`bse_io.py:958-963`, built EAGERLY at load time, audit P3) to **0.3 %**.  So the BSE
memory wall at fixed mu is the **band window**, not mu: at P=64 / mu=10015 the deck's full
available window (`nv=26`, `nc=998`) costs `2M` ~ 15.4 GB/rank on top of a
band-window-independent ~5.5 GiB — i.e. ~21 GiB/rank, ~42 GiB/node of 192.  **The full
window is reachable at this size.**  mu enters only through `T_b ∝ mu^2/P` and
`M ∝ mu/p`, so a 32k-centroid BSE at P=64 would need ~16 x 1.6 = 26 GiB of T-chain —
which is where the next wall is.

Further legs appended below.

---

## S3 — END-TO-END: WHAT IT TOOK

### 3.1 FINDING (blocking, transport): the certified AS.7 MPI-collectives cell does NOT survive the BSE solver

Job **7879458** (P=4, 785 centroids, `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` +
THREAD_MULTIPLE MPIwrapper + `LORRAX_MPI_FINALIZE_FIX=skip_atexit`, i.e. the identical
env block that runs GW at P=64 in job 7879295): the loader completes, the q=0 head is
injected, `BSE problem (sharded 2x2): 4 cond x 4 val x 16 k = 256 dim` prints, and then
**every rank** dies at the first materialisation of the Lanczos result:

    jax.errors.JaxRuntimeError: UNKNOWN: Buffer Definition Event: MPI: Communicator
    requested from a thread that is not the one MPI was initialized from.
    Multiple threads/devices per process are not yet supported.

surfaced at `bse/bse_jax.py:179  n_done = int(n_iter_done)` — i.e. inside the compiled
`bse_lanczos.solve_bse_sharded._full_run`.  rc=1 on all 4 ranks, VmHWM 0.60 GiB (nowhere
near a memory limit).

This is **not** a BSE-code or memory failure.  The structural difference from GW is that
BSE's collectives sit inside a `lax.scan` inside a `shard_map` inside the Lanczos loop,
all under ONE jit, where XLA:CPU's thunk executor is free to run the collective thunk on
an intra-op pool worker; jax's MPI collectives backend caches its communicator against
the initialising thread and refuses.  GW's collectives sit at module top level and run on
the main thread.

**Consequence for the campaign:** the AS.7 "measured upgrade" cell (1.18x vs gloo/ib0 at
P=16) is **GW-only**.  Any BSE work must run on the certified gloo/ib0 default, and the
two-plan/queue notes that assume `impl=mpi` everywhere need a scope banner.

Job **7879463** — same deck, same everything, `BSE_COLL=gloo`: **rc=0, 112 s wall**,
`Lowest 4 eigenvalues (eV): [1.30537661 1.3504201 1.42411254 1.50449023]`, VmHWM 0.6 GiB.
Gloo/ib0 is the route.

### 3.2 The 10k ladder (job 7879470)

Legs, all at the FULL mu=10015 basis; only the band window moves.  "Smallest legal" at an
8x8 mesh is n_val 8 / n_cond 8: the loader pads valence to a multiple of `grid_y` and
conduction to a multiple of `grid_x`, and the trial vector is `P(None,'x','y',None)`.

Results are in S2.3 (per-leg wall/VmHWM/eigenvalues + the ladder summary table).  Headline:
**rc = 0 / 1 / 0 across a 96x escalation in BSE dimension, the single rc=1 being the
output-writer race (S4.8), and no leg within 10x of the node budget.**

Where the NEXT BSE wall is, from the S2.3.1 model rather than from a guess:
* on the band axis at fixed mu, `2M ∝ nc*nv*mu/p`.  The deck's full available window
  (nv=26 -> pad 32, nc=998 -> pad 1000) costs `2M` ~ 15.4 GB/rank on top of the
  band-window-independent ~5.5 GiB, i.e. ~21 GiB/rank = ~42 GiB/node of 192.  **Reachable
  today.**
* on the mu axis at fixed window, `T_b ∝ mu^2/P` dominates: 1.615 GiB at mu=10015/P=64
  becomes ~16.5 GiB at mu=32k/P=64, and the chain holds ~3 of them -> ~50 GiB/rank =
  100 GiB/node.  That is the wall to pre-register for a rung-8 BSE, and it is a
  `T`-tensor wall, not a W or a psi wall.

---

## S4 — ANTIPATTERN AUDIT (task 2)

### 4.1 **DOCTRINE-1 VIOLATION — the centroid path requires an O(N_mu^2) object on ONE rank**

`centroid/pivoted_cholesky.py::build_gram_q0_via_loadwfns` + `prune_candidates_by_pivoted_cholesky`.

The candidate Gram `G` is `(M, M)` complex128 with `M = ceil(N_c * oversample)` after orbit
unfolding — at the rung-5 point `M = 13872` for `N_c = 10000`, i.e. `M ~ 1.386 * N_mu`.
`G` bytes `= 16 * M^2 ~ 30.7 * N_mu^2`:

| N_mu | M | one (M,M) copy | peak as shipped (4 live copies, S4.2) |
|---:|---:|---:|---:|
| 10 015 | 13 872 | 3.08 GB | 12.3 GB |
| 32 059 | 44 500 | 31.7 GB | 127 GB |
| 100 000 | ~139 000 | 309 GB | impossible |

A sharded route EXISTS in the same file (`make_sharded_gram_q0`,
`make_sharded_pivoted_cholesky_select`, and `prune_candidates_by_pivoted_cholesky` already
reshards `G` to `P(('x','y'), None)` before the select).

**Static prediction (mine) and what actually happened — the prediction was WRONG, and the
truth is worse.**  From source I predicted a clean refusal at `pivoted_cholesky.py:371`:
`prune_candidates_by_pivoted_cholesky` raises when `M % (p_x*p_y) != 0`, and
`M = 13872 = 2^4 * 3 * 17^2` gives `13872 % 64 = 48`, so only `P in {1,2,4,8,16}` divide.
**That refusal is never reached.**  MEASURED (job 7879470, kmeans leg, 32 nodes / P=64,
the identical invocation the accepted single-process run used):

    [LEG kmeans_p64 rc=139 wall=115s]        # 139 = SIGSEGV, all 64 ranks
    MAX VmHWM = 9.15 GiB

The last application output on every rank is
`Sharded mesh: ('x'=8, 'y'=8) over 64 devices`; the very next banner in
`kmeans_cli.main` (`[orbit] WFN stores N sym op(s); recovered ...`) never prints, and no
Python traceback is produced — the process dies on a signal, not an exception, and the
jax coordination service then reports "The tasks have crashed."  So the multi-process
centroid path fails **upstream of the prune**, somewhere between `_build_mesh` and the
orbit-symmetry recovery, with a hard segfault rather than the designed refusal.
(`[charge_density] source='auto' -> no QE .save found, falling back to IBZ wavefunction
sum from WFN.h5` prints identically in the accepted single-process run, so the density
source is not the difference.)

**The P=4 discriminator settles it (job 7879492): SIGSEGV there too** — rc=139, 115 s,
VmHWM 9.28 GiB (nowhere near the 192 GB node), and `13872 % 4 == 0` so the divisibility
gate is not even in play.  P=4 also gets FURTHER than P=64 and localises the crash to a
single function.  Ordered markers, last-to-first:

    ✓ JAX initialized: 4 device(s) (local: 1, proc 0..3/4)
    [charge_density] source='auto' -> ... IBZ wavefunction sum from WFN.h5
    P/4 = 11520 < 100000; sharding anyway (multi-host: single-device fallback would deadlock)
    Sharded mesh: ('x'=2, 'y'=2) over 4 devices
    [orbit] WFN stores 2 sym op(s); recovered 12-op symmorphic point group ...
    k-means weight: band_range Sigma_{n in [0,1024)} ... (symmetrized, 12 ops)
    Grid: 24x24x80 = 46080 points; N_c = 1250          <-- LAST LINE
    <SIGSEGV>                                           "Lloyd: N steps" never prints

### 4.1a THE FAULTING FRAME (job 7879495, `PYTHONFAULTHANDLER=1`, P=4) — exact

`Grid: ...`/`Lloyd: ...` bracket `weighted_kmeans_jax`, so I expected the sharded Lloyd.
**Wrong again — the faulthandler names a different function, and every captured dump
names the SAME one:**

    Fatal Python error: Segmentation fault
    Current thread ... (most recent call first):
      jax/_src/numpy/ufunc_api.py:182       in __call__
      jax/_src/numpy/array_methods.py:610   in deferring_binary_op
      jax/_src/numpy/fft.py:99              in _fft_core
      jax/_src/numpy/fft.py:271             in ifftn
      src/psp/get_DFT_mtxels.py:196         in valence_density_from_kpoint
      src/psp/get_DFT_mtxels.py:298         in compute_valence_density
      src/centroid/charge_density.py:153    in rho_from_wfn_ibz
      src/centroid/charge_density.py:366    in get_charge_density
      src/centroid/kmeans_cli.py:277        in main

The faulting op is `get_DFT_mtxels.py:196`

    psi_r = jnp.fft.ifftn(psi_occ, axes=(-3, -2, -1), norm='ortho') * scale

— an **eager, unsharded `jnp.fft.ifftn`** over the all-k-resident occupied ψ FFT box
(10 IBZ k x 26 bands x 2 spinor x 46080 grid = 383 MB), executed on a MULTI-PROCESS JAX
client.  It is the "single source of truth for the density quadrature" (its own docstring),
shared with `compute_valence_density`, the per-k CLI and
`gw.kin_ion_io.build_valence_density_distributed`.

So the multi-process centroid failure is **NOT** in the Lloyd kernel, **NOT** in the
pivoted-Cholesky prune, and **NOT** the divisibility refusal: it is in
`kmeans_cli.main:277`'s charge-density construction, before the mesh is even used, in a
plain eager FFT that has simply never been executed with `jax.process_count() > 1`.
Honest caveat on ordering: at least one rank printed the `Sharded mesh` / `[orbit]` /
`k-means weight` banners (which live AFTER line 277) before the step was torn down, so
ranks reach the fault at different times; every faulthandler dump that was captured,
however, is the frame above.

Mitigation that does NOT apply on this deck: `get_charge_density(source='auto')` prefers
QE's stored density and would never enter this code — but it requires
`<prefix>.save/charge-density.**hdf5**` (`charge_density.py:350`) and the b1024 deck's
`b1024_out/MoS2.save/` holds `charge-density.**dat**`, so the auto-detect correctly
declines and falls back to the IBZ sum.  The accepted single-process run took the same
fallback (its log carries the identical banner), which is why P=1 is green and P>1 is not.

Either way the operational conclusion is the same and is the headline for this path:
**at 10k centroids the centroid selector is a single-rank program, its Gram is an
O(N_mu^2) object on that one rank, and its multi-process route does not run.**
S7.1 sizes the repair.

Everything else in the audit's scope is clean on doctrine 1: `AN_MU=10048 colltable.py`
over the rung-5 GW dump (1344 modules, 448 with collectives) reports
**"NO collective carries a full (mu,mu) tile"**, and the BSE inner loop's largest per-rank
object is `T_b (mu/p_x, nu/p_y, ns, ns, nk)` = `N_mu^2/64` — correct 2-D scaling.

### 4.2 Involuntary rematerialization — the (M,M) Gram assembly (kmeans)

`pivoted_cholesky.py:1014-1031`, the column-blocked path (the ladder-wall fix, commit
b436e47).  The blocking bounds the *pair-density* tensors correctly, but the assembly that
follows is four EAGER full-size ops outside any jit:

    g_blocks.append(G_b)  x n_blocks     # list holds a full (M,M) worth
    G = jnp.concatenate(g_blocks, axis=1)   # + a second (M,M); g_blocks still live
    G = 0.5 * (G + jnp.conj(G.T))           # + conj(G.T) temp + the sum

so **four live (M,M) copies at the peak** — 12.3 GB at M=13872, and 127 GB at the M of the
c32000 set, which is the entire node.  This is the same class as the AO `_reshard_all`
471.9 -> 59 MB fix (an intent that the executed form does not honour), except here the
mechanism is eager-op materialisation rather than partitioner hoisting.
**FIXED — see S6.1.**

Memory model for the measured 63.15 GiB VmHWM at 10k, reconciled to the byte class:
`col_block` is sized so the two pair tensors take 25 % of the auto-detected budget
(`0.25 * 168.211 GB`), i.e. `2 * nk * ns^2 * M * col_block * 16 B` with
`col_block = 1480` -> 2 x 21.0 GB = 42.0 GB; + 12.3 GB of Gram copies; + psi/loader/host
~9 GB = 63 GB.  Matches.

### 4.3 Missing buffer donation at a real top-level boundary (BSE)

`bse/bse_lanczos.py::solve_bse_sharded`.  `W_R = ifft_q(W_q)` was computed **inside**
`_full_run`, the single jit that contains the whole Lanczos.  There, `W_q` is a jit
*parameter*: its buffer is owned by the caller for the entire call and cannot be freed,
and XLA has no same-shape *output* to alias it onto — which is exactly what the in-code
note at the decorator recorded ("the donation was always declined").  Net: **both `W_q`
and `W_R` resident for the whole solve**, `2 * (mu_pad/p_x) * (nu_pad/p_y) * nk * 16 B`
per rank = 2 x 404 MB at mu=10015/P=64, growing as `mu^2/P` (2 x 4.1 GB at mu=32k).
`src/bse/` has exactly ONE `donate_argnums` in the whole package (`vq_interp.py:369`).
**FIXED — see S6.2.**

House fact confirmed in place: donation inside `bse_stack_matvec`'s `jax.jit(_matvec)` is
INERT, because that jit is only ever traced into `_full_run`.  The only real dispatch
boundaries in the Lanczos path are `_full_run` itself and (in the Davidson path)
`apply_H`.

### 4.4 Collectives on the wrong mesh axis / unstacked (BSE inner loop) — CONFIRMED at HLO level

`bse_stack_matvec.py::_w_stack` (`:110-134`).  Per **trial**, inside the `lax.scan`:

    all_gather(X_b, 'y')                                  small
    all_gather(R, 'x')          (c_full, nk, ns, nu_loc)  LARGE   <- stride-8 'x' groups
    psum_scatter(..., 'x', dim=c) (c_full, nu_loc, ns, nk) LARGE  <- stride-8 'x' groups
    psum_scatter(..., 'y', dim=v) (c_loc, v_full, nk)      small  <- node-local 'y' groups

The 8x8 mesh maps device (x,y) -> process 8x+y, so 'y' groups are 8 consecutive ranks
(4 nodes, node-local pairs) and 'x' groups are stride-8 (8 distinct nodes, zero locality).
**The large payloads ride 'x'** — the exact inversion the sigma kernel abandoned in the
AK.9 axis-order swap.  And nothing is stacked: the trial axis is the natural stack axis,
but the collectives are inside the scan body, so a block-b solve fires `4b` grouped
collectives per matvec where 4 would move the same bytes.  Both were already predicted
statically in `wk_REL/docs/RESHARD_OVERHEAD_MEMO.md` §6.1 (#1); this workstream confirms them
on the compiled artifact at the 10k shape (S3.2 collective table).
**Sized as a recommendation in S7.2 — the sigma precedent measured NO win from the same
swap at its shape, so this is not a cheap-and-clearly-safe fix.**

### 4.5 BSE does not inherit LORRAX_FFT_FFI — and must not, cheaply

`common/fft_helpers.py:306-324`: `local_ifftn3`/`local_fftn3` are literal `jnp.fft.ifftn`/
`jnp.fft.fftn` aliases.  `_w_stack` calls them on
`(mu_loc, nu_loc, ns, ns, nkx, nky, nkz)` with `axes=(4,5,6)` — the **minor-most** layout,
which commit 068286c measured at **0.00 MB moved** vs 124.60 MB for the k-major flat-k
layout on the same tile.  So BSE is already in the *good* FFT layout and the FFI (whose
entry points are the flat-k helpers) cannot be bolted on without first relabelling BSE
into the costly layout.  This is the "route (a) must ship atomically" conclusion of
068286c, and it holds at 10k: **do not adopt LORRAX_FFT_FFI in BSE as a movement-only
change.**

### 4.6 The committed machinery is NOT adopted in `src/bse/`

`grep -rn "contract_bands\|bands_gemm_ffi_enabled" src/` at 1a52d51 hits only
`common/zeta_projection.py`, `gw/ppm_tau_kernel.py`, `ffi/common/ffi_loader.py`.
**Zero hits in `src/bse/`** — `contract_bands_block_reshard` is not adopted anywhere in
the BSE tree, so the GEMM dial (`LORRAX_BANDS_GEMM_FFI`, all four precisions as of
1a52d51) has no BSE consumer either.  The commit message for 1a52d51 ("closes the BSE c64
gap") refers to the *handler* being able to take c64, gated by `wk_REL/probes/gbp_bse_fp32.py`
on a synthetic fixture — not to any BSE production call site.  Sized in S7.3.

### 4.8 **BUG FOUND AT SCALE — all 64 ranks race one `eigenvectors.h5`** (rc=1 class)

`bse/bse_io.py::write_eigenvectors_stream:84` — `with h5py.File(output_file, "w") as f:`
with **no rank gate anywhere in the function or at its call site**
(`bse_jax.py:225`).  At P=64 on Lustre, with `HDF5_USE_FILE_LOCKING=FALSE` (which the whole
campaign sets), the 64 writers truncate each other:

    OSError: Unable to synchronously create file (file signature not found)
    OSError: Unable to synchronously create file (truncated file: eof = 96,
             sblock->base_addr = 0, stored_eof = 2048)

Measured: job **7879470 leg m24x64**, rc=1 on ~60 of 64 ranks, **after** the physics
completed and the eigenvalues printed.  The surviving file is structurally complete and
passes `h5ls`, so nothing downstream would notice — QUALITY_PATTERNS #7's exact scenario
("a half-written artifact is indistinguishable from a complete one"; "P ranks overwrote one
output file cleanly") realized in production the first time the writer ever ran at P>1.
**FIXED — see S6.3.**

### 4.7 kmeans writes its output from EVERY rank (rc=0 class)

`centroid/kmeans_cli.py:495` — `np.savetxt(out_file, ...)` is **not** rank-gated, while the
`timing.report()` six lines later IS (`if jax.process_index() == 0`).  At P>1 every rank
opens and writes the same path concurrently.  This is QUALITY_PATTERNS #7's own example
("P ranks overwrote one output file cleanly").  It has never fired because the path has
only ever run at P=1.  Fix is one `if jax.process_index() == 0:` — but it is only reachable
after S7.1, so it is filed with it.

---

## S5 — COST-RANKED FINDINGS FROM THE COMPILED ARTIFACT (10k, cache-cold)

`AN_MU=10048 wk_AN/colltable.py run_L5_b1024_c10000/hlo_dump` ->
`wk_REL/results/colltable_L5_10k.txt`.  448 modules carry collectives.  Ranked by bytes:

| # | module | collectives | per-rank bytes | verdict |
|---|---|---|---:|---|
| 1 | `module_0390.jit__identity_fn` | all-to-all x1 (8 buffers `c128[10,1,1,157,5760]`) | **1157.53 MB** | zeta-side reshard; largest single collective in the run |
| 2 | `module_0388.jit__block` | all-gather `c128[1,10048,5760]` + `c128[1,1256,10048]` | 926.02 + 201.92 MB | full-mu gather against the local r block (5760 = 46080/8) |
| 3 | `module_0656` / `0732` `.jit__res` | 2x all-gather `c128[4,10048,1256]` + 3x `c128[4,1256,10048]` | **5 x 807.70 MB = 4.04 GB, TWICE** | **the W Dyson residual DIAGNOSTIC — see S5.1** |
| 4 | `module_0864 jit_sigma_sx`, `0924 jit_sigma_coh` | all-reduce `c128[16,1024,2,1256]` + `c128[16,1024,1024]` | 658.51 + 268.44 MB **each** | fully-replicated results; grow as `nb*mu/p` and `nb^2` |
| 5 | `module_0463`/`0658 .jit__do_unfold` | 2x2 all-to-all, 8 buffers `c128[16,1256,157]` | 807.70 MB per module | W IBZ->full-BZ unfold |
| 6 | `module_0134.jit__reshard_all` | 2x all-gather `c128[16,128,2,10015]` + 2x all-to-all | 656.34 + 656.41 MB | the AO fix WORKING AS DESIGNED (full/Px + full/Py); costs 2.25 s |
| 7 | `module_0862 jit_hartree`, `0870 jit__psum`, `0980 jit__kernel` | all-reduce/all-gather `c128[16,1024,1024]` | 268.44 MB each | replicated (nk,nb,nb); `nb^2` growth |

### 5.1 **The single cheapest win found: `LORRAX_W_RESIDUAL_CHECK=1` is ON in the ladder harness**

`run_L5_b1024_c10000/inner.sh` exports it.  `gw/w_isdf.py:546-569`'s own docstring:
"Diagnostic-only, opt-in via `LORRAX_W_RESIDUAL_CHECK=1`; **never on in the traced
production path, so the collective-table gate is taken with it OFF**."  Its `_res` jit
(modules 0656 / 0732) forms `I - V@chi` and `A@W - V` on the first 4 q and pays
**5 all-gathers of 807.70 MB per call, once per W solve, twice per run = ~8.1 GB/rank of
purely diagnostic collective traffic**, inside `W.exec` (46.26 s static + 68.56 s probe =
114.8 s, 7.7 % of recorded time).  It also *invalidates the collective table* taken from
this dump for modules 0656/0732 — which is why they are labelled diagnostic above rather
than counted as a production antipattern.

This is a **harness** defect, not a code defect: it is one line in another workstream's
inner script.  Recommendation: drop it from `l5/l7_*.sbatch` for every timing/colltable run
and keep it only for correctness gates.  Expected recovery: bounded above by 114.8 s minus
the true Dyson solve time; an A/B is one restart-gated 32-node leg (S7.5).

---

## S6 — WHAT I FIXED (movement-only, gated)

Three changes were attempted; **one was refuted and reverted, two are delivered**.  All
live in `/work2/08271/jackmc/frontera/wt-REL10k` (branch `wt-REL10k-scale`), **NOT
committed**; `py_compile` clean.  Final delivered diff: `src/bse/bse_lanczos.py` (+39/-11)
and `src/bse/bse_io.py` (+22); `src/centroid/pivoted_cholesky.py` back at HEAD.

| fix | what | parity gate | value gate | status |
|---|---|---|---|---|
| A | blocked-Gram donated accumulator (kmeans) | **byte-identical** centroid file (7879484) | **no VmHWM win** (63.60 vs 63.15 GiB) | **REFUTED, REVERTED** |
| B | `W_R = ifft(W_q)` hoisted to a donated top-level jit (BSE) | **exact 0.0** at P=4 (7879476) AND **max rel 0.000e+00** at P=64/10k (7879500) | HLO `input_output_alias` honoured at P=4 and P=64; -33 MB at P=4, **VmHWM-NEUTRAL at P=64** | **DELIVERED** (value claim scoped: buffer-set, not RSS, at 10k) |
| C | one writer for `eigenvectors.h5` (BSE) | eigenvalues identical to base at P=64 | **rc 1 -> 0, write races 18 -> 0**, valid eigenvectors.h5 (7879500) | **DELIVERED** |

### 6.1 `centroid/pivoted_cholesky.py` — **TRIED, GATED, REFUTED, REVERTED**

The change (assemble the blocked Gram into ONE donated accumulator via
`jax.lax.dynamic_update_slice` + a donated `_hermitize`, taking live `(M,M)` copies from
4 to 2) is **correct** — job 7879484 ran it at the 10k point, rc=0, and its centroid file
is **BYTE-IDENTICAL** to the accepted `centroids_frac_10015_b1024_c10000.txt` (the base
leg in job 7879476 reproduced the same file, so the harness itself is validated).

**But it buys nothing: VmHWM 63.604 GiB (fixed) vs 63.153 GiB (base) — no win, inside
run-to-run noise.**  My S4.2 premise ("four live copies set the peak") is REFUTED by the
centroid ladder itself.  Measured VmHWM across five accepted sets on the SAME node class
(jobs 7879286, 7879344, 7879354):

| N_c | M | M^2 | VmHWM (GiB) | VmHWM (GB) | fit |
|---:|---:|---:|---:|---:|---:|
| 10 000 | 13 872 | 1.924e8 | 63.15 | 67.81 | 67.81 |
| 15 000 | 20 287 | 4.116e8 | 67.17 | 72.12 | 71.65 |
| 20 000 | 25 932 | 6.725e8 | 71.06 | 76.30 | 76.22 |
| 25 000 | 30 919 | 9.560e8 | 75.77 | 81.36 | 81.19 |
| 30 000 | 35 005 | 1.225e9 | 80.01 | 85.91 | 85.91 |

    VmHWM(GB) = 64.43 + 17.53 bytes * M^2        (max error 0.66 % over a 6.4x span in M^2)

**17.53 B/M^2 is ~1.1 copies of a c128 (M,M), not 4.**  The reason is the reservoir: the
two pair-density tensors are auto-sized to a CONSTANT 25 % of the memory budget
(`col_block` is chosen so `2 * nk * ns^2 * M * col_block * 16 B` = 0.25 * budget = 42 GB at
every M), so the assembly's extra copies are allocated inside memory the allocator has
already reclaimed from them and never push the high-water.  A 4 -> 2 reduction cannot
lower a peak that measures at ~1.  The change is therefore recorded as **measured-neutral
and reverted**; the patch is preserved at
`wk_REL/docs/patches/gram_donated_accum_REFUTED_2026-07-29.patch` (same treatment as the refuted fused
tau kernel).  Scope of the refutation: MoS2 4x4 / 30 Ry / nb=1024, single-process,
`memory_per_device_gb` auto-detected at 168.211 GB on a 192 GB node.

**The useful product of that measurement is a predictive wall for the single-process
centroid path:** at 192 GB/node the fit gives `M <= 85 300`, i.e.
**N_c <~ 61 500 centroids** (N_c=33000 -> 101 GB, 40000 -> 118 GB, 50000 -> 149 GB,
61000 -> 190 GB).  This is a memory wall of the SINGLE-RANK path only — S7.1 removes it.

### 6.2 `bse/bse_lanczos.py` — hoist `W_R = ifft(W_q)` to a DONATED top-level jit

`W_R` is now built by `jax.jit(_W_local_ifftn, donate_argnums=(0,))` outside `_full_run`,
the caller-side `data["W_q"]` reference is released, and `_full_run` takes `W_R` directly
(the Davidson branch already consumed `W_R`, so it just stops recomputing it).  Releases
one `(mu_pad/p_x, nu_pad/p_y, nk)` c128 buffer for the whole solve: **404 MB/rank at
mu=10015/P=64, `mu^2/P` growth**.  Value-identical (same helper, axes, norm, operand);
gated at 1e-12 on the P=4/785c eigenvalues against job 7879463, plus VmHWM A/B and the
HLO `input_output_alias` line.

### 6.2b **THE CONTROLLED A/B (job 7879500) — BOTH FIXES VALIDATED AT P=64**

Base then fixed, back-to-back, **same 32 nodes, same allocation, same order of gloo group
creation**, n_val 24 / n_cond 64 / N_mu 10015 / `--write-eigs 8`:

| leg | rc | wall | h5py write races | gloo ctx timeouts | VmHWM/rank | eigenvectors.h5 |
|---|---:|---:|---:|---:|---:|---|
| **base** (srcpin 1a52d51) | **1** | 585 s | **18** | 0 | 5.911 GiB | 3 159 592 B (race product) |
| **fixed** (wt-REL10k) | **0** | 224 s | **0** | 0 | 5.932 GiB | 3 159 592 B (rank 0) |

**Eigenvalue parity: `max rel = 0.000e+00` -> PASS (1e-12)** — exact, at production scale.

Readings:
1. **FIX C is proven.** The writer race reproduces on these nodes (18 ranks hit
   `Unable to synchronously create file`) and the one-writer gate removes it completely:
   rc 1 -> 0, races 18 -> 0, a valid `eigenvectors.h5` of the same size, no hang (the
   ungated `device_get` loop kept multi-process program agreement, as designed).
2. **The two earlier gloo deaths (7879486, 7879493) were transport luck, NOT the fix.**
   The fixed tree runs with **0 gloo context timeouts** on the same node class where
   7879493's fixed leg died with them, and the base leg in this very job also saw 0.  The
   S6.2a worry that the extra cold compile widens startup skew past the 30 s KV deadline is
   **not supported**: 1 clean run and 2 dirty runs of the same binary, with a clean base on
   the dirty nodes, is the documented P>=64 gloo flakiness (retry cost ~1 leg).
3. **FIX B does not regress anything and its donation is honoured at scale**
   (`HloModule jit__wrap ... input_output_alias={ {}: (0, {}, may-alias) }` on
   `c128[1256,1256,4,4,1]`, `num_partitions=64`), but it is **VmHWM-NEUTRAL here**
   (5.932 vs 5.911 GiB — noise).  Same reservoir mechanism that refuted FIX A: the loader's
   401 MB/rank numpy staging buffer is freed before the solve, so the 404 MB the donation
   releases is re-used inside already-touched memory instead of lowering the process
   high-water.  Its value is the XLA live-buffer set (OOM headroom at larger mu, where
   `W ∝ mu²/P` outgrows the staging reservoir), NOT measured RSS at 10k.  Recorded with
   that domain.
4. **Do NOT read 224 s vs 585 s as a speed-up.**  The base leg took the restart read
   COLD; the fixed leg hit warm page cache for the same 27.4 GB.  This is the same
   confound as leg 2 vs leg 1 in the ladder, and it is the strongest single piece of
   evidence for S7.4 (the read, not the solve, is the wall).

### 6.2a Gate status of FIX B at P=64 (superseded by 6.2b, kept for the record)

At P=4 / 785c (job 7879476) FIX B is fully gated: eigenvalue parity **exact 0.0**, VmHWM
1.029 -> 0.996 GiB (-33 MB, vs the 39.7 MB one W buffer at that shape), and the HLO gains
`input_output_alias={ {}: (0, {}, may-alias) }` on `c128[394,394,4,4,1]` where the base
module had none.

At P=64 / 10015 the donation **compiles and is honoured** — the fixed tree's HLO carries
`HloModule jit__wrap ... input_output_alias={ {}: (0, {}, may-alias) },
entry_computation_layout={(c128[1256,1256,4,4,1])->c128[1256,1256,4,4,1]},
num_partitions=64`, i.e. exactly the 404 MB/rank W buffer aliased — but **both attempts
(jobs 7879486 and 7879493) died before the solve** in

    Gloo context initialization failed: DEADLINE_EXCEEDED: GetKeyValue() timed out
    with key: cpu:gloo/0,2048,...,14336/1        (7879486, consecutive-rank 'y' groups)
    with key: cpu:gloo/0,16384,...,114688/1      (7879493, stride 'x' groups)

both at the 30 s deadline.  That is the campaign's documented gloo-at-P>=64 fragility
(AC.2 / §10b), it is a transport failure with no relation to buffers, and the same code
is green at P=4 — **but two failures with the fix and one success without it, on three
DIFFERENT allocations, cannot separate "the fix perturbs collective startup" (it does add
one extra cold XLA compile + dispatch before the first collective, which is a real way to
widen inter-rank skew past a 30 s KV deadline) from node luck.**  Job 7879500 runs BASE
then FIXED back-to-back inside ONE allocation to settle it.  **Until that lands, FIX B is
NOT validated at P=64 and must not be merged on the P=4 evidence alone.**

### 6.3 `bse/bse_io.py` — ONE writer for `eigenvectors.h5`

Early-returns every non-zero rank before the `h5py.File(..., "w")`.  The
`jax.device_get(eigenvectors[i])` loop is deliberately kept on BOTH branches: those slice a
GLOBAL replicated array, so they dispatch an XLA computation every process must enter, and
a naive rank-0-only body would convert an rc=1 into a hang.  `solve_bse_sharded` pins
`out_shardings=(rep_eig, rep_eig, rep_eig)` with `rep_eig = NamedSharding(mesh, P())`, so
the eigenvectors are replicated and rank 0's file is byte-for-byte what any rank would have
written.  Gate: the same leg re-run at P=64 must go rc=1 -> rc=0 with identical eigenvalues
and a valid `eigenvectors.h5`.  Jobs 7879486 / 7879493 could not deliver that verdict
(both died in gloo context init before the solve, S6.2a); job 7879500's controlled
back-to-back A/B is the one that settles it.  **FIX C is therefore not yet gated at P=64
either** — its correctness argument is airtight (the writer is pure host I/O on replicated
data, and the ungated `device_get` loop preserves the multi-process program agreement),
but the campaign's rule is that a claim needs a run, and this one does not have one yet.

---

## S7 — RECOMMENDATIONS (sized, NOT applied)

Ordered by cost recovered at the 10k point and by what they unblock at 32k+.

### 7.1 (STRUCTURAL, highest value) Make the centroid path P>1-capable at arbitrary M
Closes the only doctrine-1 violation found (S4.1), removes the measured single-node wall
(`N_c <~ 61 500`, S6.1), and is a hard prerequisite for any centroid set beyond that.
* **STEP 0 — fix the P>1 SIGSEGV; it is now localised to ONE line (S4.1a):**
  `psp/get_DFT_mtxels.py:196`, the eager unsharded
  `jnp.fft.ifftn(psi_occ, axes=(-3,-2,-1), norm='ortho')` in
  `valence_density_from_kpoint`, called from `centroid/charge_density.py:153
  rho_from_wfn_ibz` via `kmeans_cli.py:277`.  Two candidate repairs, both cheap:
  (a) run the density quadrature through the same `shard_map`'d local-FFT helper the rest
  of LORRAX uses (`common.fft_helpers.make_sharded_ifftn_3d`) with the k/band axis
  sharded, which is also the right answer for memory as `nk*nb` grows; or (b) compute the
  density on rank 0 (or under `jax.default_device(local_device)`) and broadcast with
  `device_put_process_local` — the density is a pure function of the WFN, identical on
  every rank, exactly the situation that idiom exists for.  (a) is preferred because the
  same function is the documented single source for `compute_valence_density`, the per-k
  CLI, and `gw.kin_ion_io.build_valence_density_distributed`.  Nothing below is reachable
  until this is fixed.
* Pad the candidate pool to `M_pad = ceil(M/n_dev)*n_dev` with sentinel rows whose Gram
  diagonal is exactly 0, and give `make_sharded_pivoted_cholesky_select` an `active_init`
  operand instead of its all-True initializer, so pads can never be picked.  Zero-pad +
  mask is the same contract `runtime.padding.padded_mu_extent` already applies to mu
  everywhere else in LORRAX; it removes the `M % n_dev` refusal at
  `pivoted_cholesky.py:371` without touching which physical candidates compete.
* Build the Gram row-sharded (`build_gram_q0_via_loadwfns` already returns `P('x','y')`
  and the caller already reshards to `P(('x','y'), None)`), and hermitize **in the sharded
  layout**: `0.5*(G + conj(G^T))` where the transpose is an all-to-all reshard, not an
  all-gather.  **NOTE a landmine on the way**: the legacy
  `make_sharded_gram_q0(enforce_hermitian=True)` (`pivoted_cholesky.py:566-579`)
  all-gathers the full `(M,M)` onto EVERY device to hermitize, and says so in its own
  comment.  It is not on the live path today (the live path is
  `build_gram_q0_via_loadwfns`), but it must not become the P>1 route as written.
* Rank-gate `centroid/kmeans_cli.py:495`'s `np.savetxt` (S4.7) and the
  `[pivoted_cholesky]` prints in the same pass — they are only latent because the path has
  never run at P>1.
* Size: ~1 day + a configuration-lattice gate (P in {1, 4, 16, 64}; `keep_idx` must be
  byte-identical at P=1 and P=64 on the same candidate set, since the pivoted Cholesky is
  deterministic).  Payoff: at mu=32k the Gram goes from 31.7 GB on one rank to ~0.5 GB/rank.

### 7.2 (BSE, movement-only, LOW expected value — do it as a measurement, not a fix)
Swap the `_w_stack` contraction order so the LARGE payload rides the node-local 'y' groups
(contract mu first, `psum_scatter('y', dim=v)` on `(mu_loc, v_full, ns, nk)`, then
`psum_scatter('x', dim=c)` on the small block) — the exact mirror of the sigma AK.9 swap.
Value-identical, NOT bit-exact (per-channel contraction order changes), so it needs the
1e-12 gate.  **Scope warning from the precedent**: the same swap in `ppm_tau_kernel` was
parity-clean and HLO-clean and showed *no measurable win* at its shape
(`sigma_perf_results.md`, job 7878110: 278.049 vs 272.040 s, inside the +-5 s band).  At
leg-1 shape the BSE payload is only 5.14 MB; it becomes interesting at leg-3 shape
(~164 MB), so gate it there or not at all.  Same site, higher value: **stack the trial
axis** — the scan body fires 2 all_gathers + 2 psum_scatters PER TRIAL, so a block-b solve
pays 4b grouped collectives where 4 would move the same bytes.  That is a real
restructuring (the scan exists to bound `T` at one copy), and `RESHARD_OVERHEAD_MEMO` §6.1
already flags it; it is the "two-phase primitive variant for BSE trial-stacking" that the
handoff records as an **OPEN owner API decision** — so it is owner-gated, not agent work.

### 7.3 (BSE) Adopt `contract_bands_block_reshard` at the `_w_stack` decode
`src/bse/` has **zero** uses of the primitive at 1a52d51 (S4.6).  The decode
`conj(psi_c)·U·psi_v -> (c_X, v_Y)` is exactly the primitive's mathematics with
`(m, n) = (c, v)` and `O = U_b`, so adoption is a layout change on `U_b`
(`(mu_loc, nu_loc, ns, ns, nk)` -> the primitive's `(nk, s, mu, s', nu)`) plus
`extra="leading"` for the trial axis.  Adoption buys: the primitive's stacked/ordered
collectives (7.2 for free), the divisibility refusal contract, and the GEMM dial
(`LORRAX_BANDS_GEMM_FFI`, all four precisions as of 1a52d51 — whose "closes the BSE c64
gap" refers to the *handler*, gated on `wk_REL/probes/gbp_bse_fp32.py`'s synthetic fixture, not to
any BSE call site).  Size: ~half a day + the 1e-12 parity gate + a collective-table A/B.
Note the primitive requires the minor mesh axis to be `axes[1]` and `c % p_x == 0`,
`v % p_y == 0` — the BSE loader already pads exactly that way.

### 7.4 (BSE, the biggest MEASURED BSE cost) Route the restart read through `SlabIO`
Measured at 10k (S3.2): ~305 s of the 458 s leg-1 wall is the restart load, ~67 %.
`bse_io._read_wq_sharded` / `_read_vq0_sharded` / `_read_psi_mu_sharded` use **serial
h5py** hyperslabs: for the 3-D flat-q layout `_resolve_munu_reader` issues
`dset[:, mu0:mu1, nu0:nu1]`, i.e. per rank `nk * mu_loc = 16 * 1252 = 20 032` discrete
20 KB runs out of a 56.6 GB file, x64 ranks = ~1.28 M small reads, ~90 MB/s aggregate.
Byte accounting for leg 1: `W0_qmunu` (16, 10015, 10015) c128 = **25.68 GB** + the q=0
`V_qmunu` tile 1.61 GB + the `psi_full_y` band slices 0.08 GB = **27.4 GB** moved in
~305 s = **90 MB/s aggregate, 1.4 MB/s per rank** on Lustre.
`file_io/slab_io.py::SlabIO.read_slab` (collective MPI-IO / host-FFI; the wk_AI writer on
the same class of object went 1.7 MB/s -> 8182 MB/s) is committed and **unused by
`src/bse/`** (`grep -rn "slab_io\|SlabIO" src/bse/` returns one unrelated comment).
Size: ~half a day; the read is byte-exact so the gate is a checksum of `W_q`/`V_q0`/`psi`
plus the eigenvalue parity.  **This is the single highest-value BSE change found** — it
targets ~2/3 of the wall of every BSE run at this size, and unlike everything else in this
list its payoff grows with mu^2 (the W tile) rather than staying flat.

### 7.5 (harness, free) Drop `LORRAX_W_RESIDUAL_CHECK=1` from timing/colltable runs
See S5.1: ~8.1 GB/rank of diagnostic collectives inside `W.exec`, and it invalidates the
collective table for modules 0656/0732.  One line in `l5_*/l7_*.sbatch`.  A/B = one
restart-gated 32-node leg.

### 7.6 (instrumentation, cheap) Give `bse.bse_jax` a `timing` report
`bse_feast`, `bse_kpm` and `bse_w_exact` all carry `timing.section` + `timing.report`;
`bse_jax._preview_lanczos` carries **none**, which is why S3.2's phase split had to be
reconstructed from the VmHWM sampler.  Three sections would fix it permanently:
`bse.restart_load` (around `load_bse_data_from_restart_sharded`), `bse.solve` (around
`solve_bse_sharded`), `bse.write_eigs` — plus, inside `solve_bse_sharded`, `bse.w_ifft`
and a split of `_full_run` into compile vs execute.  Purely additive; no numerical effect.

### 7.7 (GW, closes the report's biggest unattributed row) Staged sigma pass at nb=1024
`sigma.tau.d2h_wait` = 812 s (55 % of recorded GW time) is a wait row.  One restart-gated
32-node leg with `LORRAX_SIGMA_TAU_TIMING=1` produces the
`w_phase / G_build / G_ifft / V_ifft / GW_mult_fft / GW_conv_ffi / project_rs` split at
nb=1024, which is the only honest way to say where sigma's time goes at this size.  The
nb=128 and nb=256 shares must NOT be extrapolated: `project_rs` moved 16 % -> 39 % between
those two points.

### 7.9 (kmeans, latent) Rank-gate `centroid/kmeans_cli.py:495`'s `np.savetxt`
Same class as the eigenvector-writer bug that DID fire (S4.8, fixed in S6.3) — it just has
not fired yet because the path has never completed at P>1.  One line; do it in the same
pass as 7.1 so it is exercised.

### 7.8 (GW, watch item) The replicated `(nk, nb, nb)` / `(nk, nb, ns, mu_loc)` assemblies
`jit_sigma_sx`, `jit_sigma_coh`, `jit_hartree`, `jit__psum`, `jit__kernel` all-reduce or
all-gather fully-replicated results of 268.44 MB (`c128[16,1024,1024]`, `nb^2` growth) and
658.51 MB (`c128[16,1024,2,1256]`, `nb*mu/p` growth) per rank.  At nb=2048 the first is
1.07 GB/rank.  AK.9 already named the `_to_host_np(sigma_kij, tiled=False)` member of this
family; this workstream confirms four more sites at 10k.  Not touched here — the
downstream consumers (head injection, diag interpolation, `sigma_mnk` write) would all
have to accept a sharded object, which `sigma_perf_results.md` correctly calls "its own
workstream".


---

## S8 — JOB / ARTIFACT INDEX (everything above is traceable to one of these)

| job | what | verdict |
|---|---|---|
| 7879286 | kmeans 10k, 1 node, P=1 (pre-existing; the ACCEPTED centroid set) | rc=0, 308 s, VmHWM 63.15 GiB |
| 7879295 | GW rung 5, 32 nodes, P=64 (pre-existing) | rc=0, 1811 s, VmHWM 36.03 GiB/rank |
| 7879344 / 7879354 | kmeans 15k..33k, 1 node (pre-existing) | source of the S6.1 memory-wall fit |
| **7879458** | BSE P=4 / 785c, `impl=mpi` | **rc=1** — MPI communicator/thread refusal (S3.1) |
| **7879463** | BSE P=4 / 785c, gloo/ib0 | **rc=0**, 112 s — the route |
| **7879470** | 10k ladder: 3 BSE legs + kmeans P=64 | BSE 0/1/0; kmeans **rc=139 SIGSEGV** |
| **7879476** | fix gates round 1 (`small`) | FIX B parity 0.0 + HLO alias; FIX A base leg byte-identical; FIX A leg died on a dtype slip |
| **7879484** | FIX A re-gate | rc=0, byte-identical, **no memory win -> REFUTED** |
| **7879486** | FIX B + FIX C gate at P=64/10k, attempt 1 | **rc=1 — gloo ctx init DEADLINE_EXCEEDED before the solve**; HLO alias present (S6.2a) |
| **7879492** | kmeans P=4 discriminator | **rc=139 SIGSEGV** — same failure at P=4 (S4.1) |
| **7879493** | FIX B + FIX C gate at P=64/10k, attempt 2 | **rc=1 — gloo ctx init DEADLINE_EXCEEDED again**, different replica-group key |
| **7879495** | kmeans P=4 `PYTHONFAULTHANDLER` + `--no-orbit` bisect | both legs rc=139, **identical frame** `get_DFT_mtxels.py:196` (S4.1a) |
| **7879500** | controlled base-vs-fixed A/B at P=64, ONE allocation | **base rc=1 (18 write races) / fixed rc=0 (0 races), parity 0.000e+00 -> FIX B + FIX C VALIDATED AT SCALE** |

Files written by this workstream:
- `wk_REL/docs/scale10k_notes.md` (this file)
- `wk_REL/results/colltable_L5_10k.txt` — GW collective table at 10k (448 modules)
- `wk_REL/results/colltable_BSE10k_s8x8.txt` — BSE collective table at 10k (42 modules)
- `wk_REL/docs/patches/gram_donated_accum_REFUTED_2026-07-29.patch` — the reverted FIX A
- `wk_REL/snapshots/srcpin_1a52d51/` — read-only source pin
- `mos2_4x4_test/{bse_inner.sh, km_inner.sh, bse_smoke_p4.sbatch,
  bse_ladder_10k_p64.sbatch, rel10k_gate.sbatch, rel10k_gate_km.sbatch,
  bse10k_fixgate_p64.sbatch, km_p4_probe.sbatch, km_p4_fault.sbatch,
  bse10k_ab_p64.sbatch}`
- run dirs `mos2_4x4_test/run_BSE10k_{s8x8,m24x64,l24x256}`,
  `run_BSE_smoke_785_p4{,_gloo}`, `run_GATE_{bse_base,bse_fixed,km_base,km_fixed,km_fixed2}`,
  `run_KM10k_p64`, `run_KM10k_p4`, `run_BSE10k_m24x64_fixed`

Code delta (worktree `/work2/08271/jackmc/frontera/wt-REL10k`, branch
`wt-REL10k-scale`, **NOT committed**): `src/bse/bse_lanczos.py` (+39/-11),
`src/bse/bse_io.py` (+22).  `src/centroid/pivoted_cholesky.py` reverted.

---

# PART II — COORDINATOR FOLLOW-UPS (2026-07-29, after e63bc8a)

## S9 — ITEM 1: the centroid path at P>1 is FIXED

### 9.1 Root cause (one class, three call sites)

Not the Lloyd kernel and not the prune: three centroid call sites paired a
**GLOBAL band-sharded `loader.load(...)`** with a **1x1 mesh built from
`jax.devices()[:1]`**.  `jax.devices()` is the GLOBAL device list, so that mesh
is *process 0's device on every rank*.  Rank 0 computes happily; every other
rank SIGSEGVs the moment the boxed FFT is issued on a device it cannot address.
That is why the logs showed exactly ONE rank's worth of progress banners and why
`Sharded mesh:` appeared exactly once at P=64.

| file | site | what it built |
|---|---|---|
| `centroid/charge_density.py` | `_load_wfn_k_fftbox_ibz` (:106-111) | ρ from the IBZ ψ sum — **the faulting frame** |
| `centroid/charge_density.py` | `rho_from_band_range` (:254, :272) | the band-range k-means weight |
| `centroid/pivoted_cholesky.py` | `gather_wfn_at_candidates` (:117-122) | ψ at the candidate points |
| `centroid/kmeans_cli.py` | `_build_mesh` (:201, :216) | the single-device fallbacks |

**The repo already contains the fix and names this exact hazard**:
`common/wfn_transforms.process_local_mesh()` — *"NOTE the difference from
`jax.devices()[:1]` ... `jax.devices()[0]` is process 0's device on every rank —
a mesh no rank but 0 can compute on."*  It was applied to `load_kpoint_fftbox`
and never to the centroid tree.  The fix is `loader.load_process_local(...)` +
`process_local_mesh()` at each site — the established process-local contract,
byte-identical at P=1 (`jax.local_devices()[0] is jax.devices()[0]` there).

### 9.2 Gate result — job 7879525

| leg | P | result | VmHWM/rank |
|---|---:|---|---:|
| p4 | 4 | died in the Gram build, **memory** (S9.3) | 73.01 GiB |
| **p16** | **16** | **COMPLETED, 10015 centroids** | **35.53 GiB** |
| p64 | 64 | (below) | |

**The SIGSEGV is gone at every P.**  Every stage that previously never executed
off rank 0 now runs on all ranks, and at P=16 the whole path completes.

**P=16 vs the accepted P=1 set — the gate, stated correctly:**

    sort(P=16 file)  cmp  sort(P=1 file)   ->  BYTE-IDENTICAL
    symmetric difference: 0 added, 0 removed, 10015 common

and the prune's own discriminating diagnostics are bit-identical to P=1:

| diagnostic | P=1 (job 7879286) | P=16 (job 7879525) |
|---|---|---|
| G diag range | [7.632e-17, 1.158e-04] | [7.632e-17, 1.158e-04] |
| picked-pivot residuals first/mid/last | 1.158e-04 / 2.116e-11 / 0.000e+00 | identical |
| tr(R_k)/tr(G) first/mid/last | 9.330e-01 / 7.653e-07 / 1.874e-10 | identical |
| orbits picked -> centroids | 897 -> 10015 | 897 -> 10015 |
| rank | 630 | 630 |
| Σ_r w (band_range weight) | 67197.1544 | 67197.1544 |
| Lloyd | 8 steps, max movement 0.000000 | 8 steps, max movement 0.000000 |

> **GATE VERDICT, honestly stated.** A literal `diff` of the two files reports 96
> differing lines — and that is the *gate being wrong*, not the result.  The 48
> affected rows are a permutation WITHIN symmetry orbits (e.g. row 19 holds
> `0.458333 0.000000 0.987500` at P=1 and `0.541667 0.541667 0.987500` at P=16 —
> both points are present in BOTH files).  The Lloyd's float reductions have a
> P-dependent order, which changes which member of an orbit ends up as the
> stored representative, and `unfold_orbit_unique_with_id` emits an orbit
> starting from its representative.  The centroid file is an UNORDERED point set,
> so a byte-compare over-discriminates: it would reject a correct result.
> **The discriminating gate is the SORTED compare plus the prune-diagnostic
> triple above, and on that gate P=16 PASSES EXACTLY.**  (QUALITY_PATTERNS
> addendum: before acting on an observable, ask what it looks like in the healthy
> state — here the healthy state is a permuted file.)
> Consequence to record: centroid μ-ordering is P-dependent, so downstream
> artifacts (ζ, W, Σ) are covariant-but-not-bitwise-comparable across P.
> Observables (eqp, gaps) are invariant under the permutation.

### 9.3 The P=4 death is MEMORY, and it exposes a second, separate defect

At P>1 the column-blocked Gram build is **disabled** — `pivoted_cholesky.py:987`
gates it on `if n_dev_total == 1`.  So a multi-device run materialises the full
open-spin pair tensors `(nk, ns, ns, M/p_x, M/p_y)`, i.e.
`1024 * M^2 / P` bytes each, TWO of them:

| P | per tensor | 2 tensors GB/rank | GB/node (2 ranks) | fits 192 GB? |
|---:|---:|---:|---:|---|
| 1 | 197.1 GB | 394.1 | 788.2 | NO (hence the blocking) |
| **4** | **49.3 GB** | **98.5** | **197.1** | **NO** |
| 8 | 24.6 GB | 49.3 | 98.5 | yes |
| 16 | 12.3 GB | 24.6 | 49.3 | yes (measured 35.53 GiB) |
| 64 | 3.1 GB | 6.2 | 12.3 | yes |

Observed at P=4: VmHWM **73.01 GiB and still climbing** when a peer's socket
closed —
`Gloo AllGather failed: Connection closed by peer` at the
`load_centroids_pre_reshard` barrier, i.e. the classic signature of a peer
being OOM-killed, not a transport fault.  Node total was heading for
2 x 98.5 = 197 GB against a 192 GB node.
**So the multi-device path has a MINIMUM P (>= 8 on this deck at M=13872), which
is the inverse of the usual scaling story and is worth a guard.**  Fix: extend
the existing column-blocking to the multi-device path (block the LOCAL column
extent `M/p_y` instead of refusing to block whenever `n_dev_total > 1`), or
refuse at resolve time with the arithmetic above.  Sized as S10.3.

## S10 — ITEM 4: why the sharded Gram route is unreachable, precisely

**It is neither dead code nor flag-gated, and there is no second "sharded
route" to switch on.**  `prune_candidates_by_pivoted_cholesky` — the live entry,
called from `kmeans_cli.py:475` — is ALREADY fully mesh-parameterised, and it is
the only route:

| step | line | already sharded? |
|---|---|---|
| requires a 2-D ('x','y') mesh, refuses otherwise | :358-363 | — |
| Gram build `build_gram_q0_via_loadwfns` | :389 | **yes** — returns G `P('x','y')` (`isdf/core.gram_q0_from_pair` return contract) |
| reshard for the pivot scan | :399-401 | **yes** — `P(('x','y'), None)` |
| select kernel `make_sharded_pivoted_cholesky_select` | :415 | **yes** — the sharded kernel is what runs at every P |

So the O(N_mu^2)-on-one-rank object is **not a design choice — it is what this
sharded route DEGENERATES TO at P=1**, where the 1x1 mesh makes every shard the
whole matrix.  At P=64 the identical code gives, per rank:

| object | P=1 | P=64 (8x8) |
|---|---:|---:|
| G as built, `P('x','y')` | 3.08 GB | (M/8)^2 x 16 = **48.1 MB** |
| G after reshard, `P(('x','y'),None)` | 3.08 GB | (M/64) x M x 16 = **48.2 MB** |

— a 64x reduction, and **doctrine 1 is satisfied outright**.  The wall
`VmHWM = 64.4 + 17.5 M^2 bytes` (ceiling `N_c ~ 61 500`) is a **P=1 wall only**.

### What actually blocked reaching it — three things, in hit order

1. **[FIXED, S9]** the `jax.devices()[:1]` process-0-device mesh: every rank but
   0 SIGSEGVed long before the prune.  Fixed via `process_local_mesh()`;
   P=16 now completes with a set byte-identical (sorted) to P=1.
2. **[OPEN — memory, and it is an INVERSE-scaling trap]** the column-blocked
   Gram build is gated `if n_dev_total == 1` (`pivoted_cholesky.py:987`), so a
   multi-device run never blocks and materialises `2 x 1024 M^2 / P` bytes/rank
   of pair densities.  That is 98.5 GB/rank at P=4 (measured: the P=4 leg died
   at VmHWM 73.01 GiB and climbing, with a peer's socket closing — an OOM kill
   wearing a transport error's clothes).  **Fits only for P >= 8** on this deck.
3. **[OPEN — THE remaining blocker]** the hard divisibility refusal at
   `pivoted_cholesky.py:371`: `M % (p_x*p_y) != 0`.  `M = 13872 = 2^4 * 3 * 17^2`,
   so only `P in {1,2,4,8,16}` divide it; **P=32 and P=64 refuse**.  M is an
   orbit-unfold count (special-position orbits are not all of size n_sym), so it
   is not controllable from the CLI.

Also present but NOT on the live path: `make_sharded_gram_q0` is dead code with
respect to `kmeans_cli` (nothing calls it), and it carries a landmine —
`enforce_hermitian=True` (:566-579) **all-gathers the full (M,M) onto every
device** to hermitize, and says so in its own comment.  It must not be adopted
as the P>1 builder.

### What it would take (sized)

* **Divisibility** — pad the candidate pool to `M_pad = ceil(M/n_dev)*n_dev`
  with sentinel rows whose Gram diagonal is exactly 0, and give
  `make_sharded_pivoted_cholesky_select` an `active_init` operand in place of
  its all-True `active` initializer so a pad can never be picked.  Zero-pad +
  mask is the contract `runtime.padding.padded_mu_extent` already applies to mu
  everywhere else in LORRAX.  Does not change which physical candidates compete.
* **Small-P memory** — block the LOCAL column extent (`M/p_y`) instead of
  refusing to block whenever `n_dev_total > 1`; or refuse at resolve time with
  the `2 x 1024 M^2 / P` arithmetic in the message (broken-promise pattern).
* **Output** — rank-gate `kmeans_cli.py:495`'s `np.savetxt` (still ungated; it
  only survived P=16 because every rank wrote identical bytes, which is exactly
  the eigenvector-writer bug that DID bite at P=64).
* **Gate** — the SORTED-set compare + prune-diagnostic triple from S9.2, **not**
  a byte-compare of the centroid file.

Estimate ~1 day plus a P in {1,4,16,64} lattice run.  Payoff: centroid
generation scales with P instead of with one node's memory, which removes the
`N_c ~ 61 500` ceiling and the last doctrine-1 violation in the pipeline.

### 9.4 P=64 with the fix — the segfault is gone; the DESIGNED refusal is reached

Job **7879533** (fix ON, rank-assert gate OFF — see 9.5), 32 nodes, P=64:

    [kmeans p64-nocheck rc=1 wall=150s]
    --- any segfault? (must be NONE) ---
    0
    MAX VmHWM = 10.83 GiB
    Unfolded 1250 reps -> 13872 distinct centroids (n_sym=12)
    Pivoted-Cholesky prune: 13872 -> 10000 (target 897 orbits)
    ValueError: M=13872 (number of candidates) must be divisible by the product
    of mesh axes 'x' and 'y' (= 64).  The sharded pivoted-Cholesky select kernel
    splits M evenly across shards.  Either drop the last 48 candidate(s) before
    calling this function, run on a mesh size that divides M, or pass
    ``--no-shard`` to use a single-device 1x1 mesh.

**Zero segfaults at P=64.**  The whole path — charge density, orbit recovery,
band-range weight, Lloyd, orbit unfold, into the prune — now executes on all 64
ranks, at 10.83 GiB/rank.  What stops it is the DESIGNED divisibility refusal,
which announces itself and names its own fix.  That is exactly the state item 4
predicted from source, and it is the correct behaviour: a loud refusal, not a
silent crash.

### 9.5 FINDING: the rank-assertion gate is NOT usable at P=64 on gloo

Arming `LORRAX_CHECK_REPLICA=1` (job 7879525, p64 leg) put the run's death at

    centroid/kmeans_isdf.py:870  shard() -> common/collectives.py:206
      multihost_utils.assert_equal(...)
    -> Gloo context initialization failed: DEADLINE_EXCEEDED: GetKeyValue()
       timed out with key cpu:gloo/<all 64 ranks>/16 ... 30.0 s

`device_put_process_local` exists **precisely to delete** that hidden P-linear
`assert_equal` all-gather (scorecard AA.1); `LORRAX_CHECK_REPLICA=1` puts it
back, and creating a WORLD-group gloo context at P=64 is the same 30 s KV
timeout that killed BSE jobs 7879486/7879493.  So:
**the rank-assertion gate is valid at P<=16 (it passed there, no divergence) and
is itself the failure at P=64 on gloo.**  Use it at P<=16; at P=64 the
discriminating gate is the sorted-set + prune-diagnostic compare of S9.2.

## S11 — ITEM 2: LORRAX_W_RESIDUAL_CHECK contamination footprint

Sweep of `/scratch2/08271/jackmc` (all `*.sbatch`, `*.sh`, `*.py`, `inner.sh`):

| quantity | value |
|---|---|
| `export LORRAX_W_RESIDUAL_CHECK=` lines found | **103** |
| ... of which set it to `1` | **103 (all)** |
| ... set to `0`/`off`/unset anywhere | **0** |
| distinct `*.sbatch` harnesses that set it | **24** (18 in `mos2_4x4_test/`) |
| run dirs where it actually FIRED (`[W solve] Dyson residual` in gw.log) | **65** |

The 18 deck-level harnesses: `aq_rehearsal`, `diag_b512_weap`,
`gate_sigma_reference`, `l1_b256`, `l2_b256_c3491`, `l2_b256_c3500`,
`l3_b512_c5000`, `l4_b512_c7000`, `l5_b1024_c10000`, `l6_r45_b2048`,
`l7_b1024_bigmu`, `omega_512cell`, `omega_ab`, `sigma_haccum2`,
`sigma_hostaccum_gate`, `sigma_iter`, `sigma_perf_ab` (+ this workstream's own
`wres_ab_p64`).  **That is the entire size ladder (l1..l7) and the whole sigma
perf A/B family** — i.e. every `W.exec` / `chi0_W` wall this campaign has quoted
from a ladder or sigma-perf run was measured with the diagnostic ON.

`gw/w_isdf.py:546-569` docstring: *"Diagnostic-only, opt-in via
`LORRAX_W_RESIDUAL_CHECK=1`; **never on in the traced production path, so the
collective-table gate is taken with it OFF**."*  The HLO cost
(`wk_REL/results/colltable_L5_10k.txt`, modules 0656/0732): 5 all-gathers of 807.70 MB
per W solve, twice per run = **~8.1 GB/rank** of purely diagnostic traffic
inside `W.exec`.

---

## S12 — SOURCE-PROVENANCE AUDIT (cross-campaign hazard, coordinator-propagated)

The ladder workstream submitted jobs whose `SRCDIR` pointed at a worktree it was
editing, and its own verdict was "module semantics plus luck, not discipline."
**The same applies to seven of my jobs.**  Assessed honestly, from file mtimes
against `sacct` windows — not asserted.

### 12.1 Which jobs were exposed

Frozen (`wk_REL/snapshots/srcpin_1a52d51/`, a read-only `git archive`) and therefore never
at risk: 7879458, 7879463, 7879470, 7879492, 7879495, 7879529, and the `base`
legs of 7879476 / 7879500.

Pointed at the LIVE worktree `wt-REL10k/src`: **7879476, 7879484, 7879486,
7879493, 7879500 (fixed leg), 7879525, 7879533.**

### 12.2 Did an edit land during a run?

Worktree mtimes: `bse_lanczos.py` 03:04:10, `bse_io.py` 03:16:27,
`charge_density.py` 03:52:28, `pivoted_cholesky.py` 03:52:48,
`kmeans_cli.py` 03:53:16.

| job | window | edits inside | what it ran | exposure |
|---|---|---|---|---|
| 7879476 | 03:06:14–03:12:34 | none | bse + kmeans | clean |
| **7879484** | 03:14:39–03:18:42 | **`bse_io.py` @03:16:27** | `centroid.kmeans_cli` ONLY | edited module is in a subtree this program never imports |
| 7879486 | 03:18:28–03:24:36 | none | bse only | clean |
| 7879493 | 03:26:19–03:31:48 | none | bse only | clean |
| 7879500 | 03:33:17–03:46:57 | none | bse only | clean |
| **7879525** | 03:54:53–04:01:37 | none (last edit 03:53:16, **97 s before start**) | kmeans only | clean |
| 7879533 | 04:02:45–04:05:21 | none | kmeans only | clean |

**Exactly one edit landed inside a run window** — `bse_io.py` during the
kmeans-only job 7879484 — and `centroid.kmeans_cli` never imports `bse.*`.
The `pivoted_cholesky.py` Fix-A revert (`git checkout`) fell between runs or
during BSE-only jobs, which likewise never import `centroid.*`.

**So no run ever loaded a module that was edited during its own window, and the
two gates that carry the headline claims — 7879525 (byte-identity) and 7879533
(P=64 refusal) — are clean.  Their results stand.**

### 12.3 Why that is luck and not discipline — and worse here than for the ladder

`centroid/kmeans_cli.py` imports the hot gate file **LAZILY**:

    :447   from .pivoted_cholesky import prune_candidates_by_pivoted_cholesky

inside `main()`, at the point of use — roughly **150 s into a run**, after the
weight build and the Lloyd.  `charge_density.py:114-115`, `:233-235` and
`pivoted_cholesky.py:101-102`, `:870-872` are function-level too.  So the
"Python's module cache held" argument that protected the ladder agent **does not
protect me**: a save landing in that 150 s window would have been picked up, and
the run would have been built from two source states with no symptom.  The only
thing that saved the byte-identity gate was a 97-second margin.

### 12.4 Discipline adopted (from this point, all jobs)

    wk_REL/srcsnap_rel10k_20260729_041117_e63bc8a/   (frozen copy)
      src/ MANIFEST.sha256 (228 .py) PROVENANCE.txt VERIFY.diff (EMPTY = equal)
    wk_REL/snapshots/pointers/CURRENT_SRCSNAP_REL10K                    (pointer)

Jobs now read `SRC=$(cat $WK/snapshots/pointers/CURRENT_SRCSNAP_REL10K)/src`.  Separate pointer
from the ladder agent's `CURRENT_SRCSNAP` so neither workstream clobbers the
other.  Re-snapshotting is an explicit timestamped act.
Job **7879537** re-runs the P=16 byte-identity gate from the snapshot and
`sha256sum -c`'s the manifest **at job start AND at job end**, so the run
carries its own proof the source was immutable throughout — a stronger
statement than any after-the-fact mtime audit.

## S13 — ITEM 3: AS.7 scope banner (drafted, NOT landed)

Full drop-in text: **`wk_REL/docs/AS7_SCOPE_BANNER_DRAFT.md`**.  Three targets, all
left untouched for the coordinator to land:

1. `SPEEDUP_SCORECARD.md` §AS.7 — a `> ⚠ CLAIM-DECAY` blockquote immediately
   after the launch block, stating the MEASURED-UPGRADE cell
   (`JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` + `MPITRAMPOLINE_LIB` +
   `LORRAX_MPI_FINALIZE_FIX`) is **GW-only** and kills BSE.
2. `docs/dev/env_vars.md`, the `JAX_CPU_COLLECTIVES_IMPLEMENTATION` row (~:250)
   — an appended "GW-ONLY — `mpi` BREAKS BSE" clause naming the rule in terms of
   the *structure* that triggers it (collectives under a `scan`/`while_loop`
   inside a `shard_map` inside one jit), not just "BSE".
3. `LORRAX_FRONTERA_ADVICE.md` §10c — one clause turning "Gloo/ib0 stays the
   certified default" into "and for BSE it is MANDATORY".

Evidence table in the draft: 7879295 (GW, impl=mpi, rc=0) vs 7879458 (BSE,
impl=mpi, rc=1, all ranks, VmHWM 0.60 GiB) vs 7879463 / 7879470 (BSE, gloo/ib0,
rc=0).  Measured domain stated explicitly: P=4 / 785c / TDA Lanczos; the refusal
is a thread binding rather than a scale effect, so P-independence is *inferred,
not measured*, and the banner says so.

### 12.5 Re-gate from the frozen snapshot — job 7879537, CLEAN

    [kmeans p16-snapshot rc=0 wall=170s]
    MANIFEST OK: snapshot unchanged since capture          (at job START)
    MANIFEST OK at end: source was IMMUTABLE for the whole run
    SORTED SETS BYTE-IDENTICAL to the accepted P=1 set -> PASS
    (row-order diff, informational): 96 lines
    MAX VmHWM = 35.53 GiB

Every discriminating diagnostic is character-for-character equal to the P=1
reference (job 7879286), printed side by side in the job output:

    Σ_r w = 67197.1544 over 1024 bands
    Lloyd: 8 steps, max movement = 0.000000
    Unfolded 1250 reps → 13872 distinct centroids (n_sym=12)
    G built, shape=(13872, 13872), diag range [7.632e-17, 1.158e-04]
    picked-pivot residuals: first=1.158e-04 mid=2.116e-11 last=0.000e+00
    tr(R_k)/tr(G): first=9.330e-01 mid=7.653e-07 last=1.874e-10
    orbit-aware: 897 orbits picked → 10015 unfolded centroids (orbit-closed)
    After pruning: 10015 centroids (rank=630)

**The centroid P>1 fix is now gated on an immutable tree whose integrity is
proved by the run itself, at both ends.**  This supersedes 7879525's p16 leg as
the citable result (that leg agreed, and its window was clean, but it read a
live worktree).

### S11.1 — MEASURED cost of the diagnostic (job 7879529, mu=10015, P=64)

Restart-gated A/B, both legs in ONE allocation, identical except the flag:

| stage | ON (=1) | OFF (=0) | delta |
|---|---:|---:|---:|
| `W[static] Dyson solve` (10 q, IBZ wedge) | 55.8 s | **50.2 s** | **-5.6 s (-10.0 %)** |
| `W[probe] Dyson solve` (16 q, full BZ) | 70.7 s | **66.2 s** | **-4.5 s (-6.4 %)** |
| `Finished screening (chi0 -> W)` | 179 s | **168 s** | **-11 s (-6.1 %)** |
| `[W solve] Dyson residual` lines emitted | 2 | **0** | flag verified effective |

Residual values themselves (ON leg): max 1.887e-14 (static), 4.680e-15 (probe)
— the distributed Dyson solve is numerically excellent, which is the one thing
the diagnostic is genuinely for.

> ⚠ **CLAIM-DECAY on my own S5.1.** I called this "the single cheapest win
> found" on the strength of its BYTES (~8.1 GB/rank).  **Measured, it is worth
> ~10 s per run** — 6 % of screening, **0.6 % of the 1811 s GW wall.**  Large
> collective volume, small wall.  Correcting in place per house rule.
>
> And the honest caveat on even that: cross-allocation variance on this stage is
> the SAME size as the effect.  The rung-5 run (job 7879295) recorded
> `W[static] 46.3 s` **with the check ON** — faster than this A/B's 50.2 s with
> it OFF — and its screening total was 168 s, equal to my OFF leg.  So the
> -5.6/-4.5 s deltas are trustworthy only because both legs ran back-to-back in
> one allocation and both stages moved the same way; the number must be quoted
> as "~10 s/run, same-allocation, +/- the node band", never as an absolute.

**What the contamination actually costs the campaign** is therefore not wall
time but *evidence quality*:
1. every absolute `W.exec` / `chi0_W` number quoted from any of the 65 affected
   run dirs is ~6 % high on the W stage;
2. the collective table taken from those dumps is WRONG for modules 0656/0732 —
   it reports ~4 GB/rank of all-gathers that the production path does not issue,
   which is exactly what `w_isdf.py`'s docstring warns about;
3. A/B comparisons between two contaminated runs are unaffected (common-mode).

Recipe to fix the harnesses: `wk_REL/docs/WRESIDUAL_HARNESS_FIX_RECIPE.md`
(default-OFF via `${WRES:-0}` riding the existing `--export=ALL,...` convention;
ON only for correctness gates; plus an announce-at-resolve line and a colltable
guard so it cannot silently recur).  **No production harness was edited.**

## S14 — PART II JOB INDEX

| job | what | verdict |
|---|---|---|
| **7879525** | centroid P>1 gate, legs P=4/16/64, live worktree src | SIGSEGV GONE at all P; p16 completed; p4 OOM (S9.3); p64 died in the *gate* (S9.5) |
| **7879529** | `LORRAX_W_RESIDUAL_CHECK` A/B, mu=10015, P=64 | ON 55.8/70.7/179 s vs OFF 50.2/66.2/168 s -> **~10 s/run** (S11.1); cancelled after both legs' screening |
| **7879533** | centroid P=64, fix ON, rank-assert OFF | **0 segfaults**, reaches the DESIGNED divisibility refusal (S9.4) |
| **7879537** | **P=16 re-gate from the FROZEN SNAPSHOT** | **rc=0; MANIFEST OK at start AND end; SORTED SETS BYTE-IDENTICAL to P=1 -> PASS** (S12.5) |

New files: `wk_REL/docs/AS7_SCOPE_BANNER_DRAFT.md` (item 3),
`wk_REL/docs/WRESIDUAL_HARNESS_FIX_RECIPE.md` (item 2),
`wk_REL/srcsnap_rel10k_20260729_041117_e63bc8a/` + `wk_REL/snapshots/pointers/CURRENT_SRCSNAP_REL10K`,
`mos2_4x4_test/{km_pgt1_gate,km_p64_nocheck,km_p16_snap,wres_ab_p64}.sbatch`.

Uncommitted code delta (item 1), worktree `wt-REL10k` on top of e63bc8a:
`src/centroid/charge_density.py` (+26/-11), `src/centroid/pivoted_cholesky.py`
(+10/-5), `src/centroid/kmeans_cli.py` (+8/-2).  `py_compile` clean.

---

## S15 — SHARED-POINTER DEFECT: audit + hardening (coordinator, 2026-07-29)

The zeta workstream found that a pointer file in the SHARED `wk_REL` is itself a
shared mutable: a peer can repoint it between snapshot creation and job launch,
and the job then runs someone else's source **while looking disciplined**.  It
observed a foreign snapshot appearing at 04:11:17 inside its own
snapshot(04:10:40)→submit(04:15) window.  **That 04:11:17 snapshot was mine**
(`srcsnap_rel10k_20260729_041117_e63bc8a`) — so this workstream is the peer that
made the hazard visible.

### 15.1 Confirmations requested

**(a) No job of mine resolves the generic `wk_REL/snapshots/pointers/CURRENT_SRCSNAP`.**
`grep -ln "CURRENT_SRCSNAP[^_]" *.sbatch` over all 12 of my sbatch files returns
NOTHING.  Full source-resolution audit:

| resolution | files | pointer? |
|---|---|---|
| `srcpin_1a52d51/src` (frozen `git archive`) | bse_smoke_p4, bse_ladder_10k_p64, km_p4_probe, km_p4_fault, wres_ab_p64, + `base` legs of rel10k_gate / bse10k_ab | no |
| `wt-REL10k/src` (LIVE worktree — the S12 hazard) | rel10k_gate, rel10k_gate_km, bse10k_fixgate_p64, bse10k_ab_p64, km_pgt1_gate, km_p64_nocheck | no |
| `$(cat CURRENT_SRCSNAP_REL10K)` (my PER-WORKSTREAM pointer) | km_p16_snap only | mine only |

The container-side inner scripts hold no pointer reads either — `km_inner.sh:7`
and `bse_inner.sh:18` consume `${KM_SRC}`/`${BSE_SRC}` passed in from the job.

**(b) The one pointer-reading job resolved ONCE and verified the manifest.**
`km_p16_snap.sbatch` read the pointer at a single line and derived everything
from `$SNAP`; it `sha256sum -c`'d at job start (:40) and job end (:55).
Job 7879537's own log records the RESOLVED ABSOLUTE PATH and both verifications:

    SNAP = .../srcsnap_rel10k_20260729_041117_e63bc8a
    MANIFEST OK: snapshot unchanged since capture        (START)
    MANIFEST OK at end: source was IMMUTABLE for the whole run   (END)

and the pointer still resolves to that same path today.  **So 7879537's
provenance is verified rather than asserted, and it would have survived the
shared-pointer failure mode**: a repoint would have printed a different `SNAP =`
line, and a mutation would have failed the hash.  Recording the resolved path
*plus* a content hash is what makes the hazard detectable after the fact.

**No job of mine is queued or running**, so nothing is currently exposed.

### 15.2 Hardening adopted — explicit per-job pin, `wk_REL/srcpin_resolve.sh`

    source $WK/srcpin_resolve.sh
    srcpin_resolve || exit 90      # at job start
    ...
    srcpin_verify_end              # at job end

* `SRCSNAP` must be passed **explicitly**:
  `sbatch --export=ALL,SRCSNAP=$(cat $WK/snapshots/pointers/CURRENT_SRCSNAP_REL10K) job.sbatch`.
  The pointer is dereferenced by the human ON THE LOGIN NODE AT SUBMIT TIME; the
  job receives a frozen absolute path and reads no pointer at all.
* **Unset `SRCSNAP` is a hard failure (rc=90), never a fallback** — a fallback is
  exactly how a plausible wrong number gets produced.
* Refuses a missing `src/` (91), a missing manifest (92), and anything that
  looks like a live worktree (93).
* Manifest verified at START (94 on mismatch) and at END (95), and the resolved
  path + provenance are echoed into the log.

Tested before use (not asserted):

| test | expected | got |
|---|---|---|
| `SRCSNAP` unset | refuse | **rc=90** |
| `SRCSNAP` = live worktree | refuse | **rc=92** (no manifest) |
| real snapshot | pass | **rc=0**, `LORRAX_SRC` exported |
| **copy with ONE comment line appended to `pivoted_cholesky.py`** | **detect** | **rc=94 at start, rc=95 at end** — `sha256sum: WARNING: 1 computed checksum did NOT match` |

The mutation test is the load-bearing one: verification is what converts a
wrong-source run into a hard failure instead of a plausible number.

`km_p16_snap.sbatch` is converted to this template and re-run as job **7879547**
with the explicit pin, to prove the template executes (a template that aborts
correctly but cannot run is no template).

### 15.3 SECOND GAP FOUND while hardening: the manifest did not cover the bytecode

The first snapshot recipe (mine, and — if the same `find -name '*.py'` idiom was
used — the other workstreams') hashed **only `*.py`**.  But `cp -a` copies
`src/**/__pycache__/*.pyc` into the snapshot AND preserves mtimes, so CPython's
timestamp invalidation **accepts that cached bytecode and executes it**.

Verified on the live snapshot: for all three files I fixed, the `.pyc` header's
embedded source mtime equals the `.py`'s actual mtime —

    charge_density  : pyc_embedded_mtime=1785315148 == py_mtime -> pyc WILL BE USED
    pivoted_cholesky: pyc_embedded_mtime=1785315168 == py_mtime -> pyc WILL BE USED
    kmeans_cli      : pyc_embedded_mtime=1785315196 == py_mtime -> pyc WILL BE USED

So a mutated `.pyc` would have passed `sha256sum -c` (which checked 228 `.py`
only) while changing what actually ran — **the same "looks disciplined, runs
something else" class as the pointer defect, one level below source.**

**No result is affected.** Those `.pyc` were compiled from the fixed sources
during job 7879525 (03:54, after the 03:52–03:53 edits), so bytecode and source
agree, and the P=16 gate did exercise the fixed code (it completed instead of
segfaulting — the old code cannot do that).

Closed two ways in `wk_REL/srcpin_resolve.sh`:
* `srcpin_resolve` now **detects and warns** on any snapshot shipping `.pyc`
  (retrofit check for snapshots already made — including peers');
* `srcpin_snapshot` builds them correctly: **`__pycache__` excluded** and
  **every file hashed, not just `*.py`**.  New snapshot
  `srcsnap_rel10k_20260729_042647_2cbd824`: VERIFY.diff 0 bytes, **341 hashed
  entries** (was 228), **0 `.pyc`**, resolver reports *"no cached bytecode in
  snapshot — manifest covers everything executable"*.
  Taken from a CLEAN tree (coordinator committed the centroid fix as 2cbd824),
  so it carries no uncommitted delta.

### 15.4 Live proof that the explicit-pin model actually removes the hazard

With job **7879547 still RUNNING**, I repointed `CURRENT_SRCSNAP_REL10K` from the
old snapshot to the new one — i.e. reproduced the exact peer-repoint event the
zeta workstream feared.  The running job is provably unaffected: its own log
holds

    [srcpin] SRCSNAP  = .../srcsnap_rel10k_20260729_041117_e63bc8a

the absolute path it was pinned to at submit, and it never consults the pointer.
Under the pointer-read model that repoint would have silently changed the source
of any job launched in the following seconds.  **The pointer is now a
submit-time convenience for a human, not an input to a job.**

### 15.5 Recommendation to the other workstreams (via the coordinator)

1. Check whether your snapshot ships `__pycache__` — `find <snap>/src -name '*.pyc' | wc -l`.
   If non-zero, your manifest almost certainly does not cover it.
2. Rebuild with `source wk_REL/srcpin_resolve.sh; srcpin_snapshot <worktree> <wk_dir> <tag>`.
3. Switch jobs to `--export=ALL,SRCSNAP=<abs path>` + `srcpin_resolve` /
   `srcpin_verify_end`; delete pointer reads from job bodies.
4. Any run whose log does not record the RESOLVED ABSOLUTE PATH plus a manifest
   result should be treated as provenance-unverified, regardless of discipline
   claimed at submit time.

### 15.6 Hardened template validated — job 7879547

    [kmeans p16-snapshot rc=0 wall=191s]
    [srcpin] SRCSNAP  = .../srcsnap_rel10k_20260729_041117_e63bc8a   (EXPLICIT, --export)
    [srcpin] MANIFEST VERIFIED at job START — source matches capture
    [srcpin] MANIFEST VERIFIED at job END — source was IMMUTABLE for the whole run
    SORTED SETS BYTE-IDENTICAL to the accepted P=1 set -> PASS
    MAX VmHWM = 35.53 GiB

The centroid P>1 gate is now established **twice** (7879537, 7879547), the second
time under explicit per-job pinning with manifest verification at both ends —
**and with a pointer repoint deliberately performed mid-flight** (15.4).  A
template that aborts correctly but cannot run is no template; this one runs.

### 15.7 Residual, named not fixed: the helper itself is a shared mutable

`wk_REL/srcpin_resolve.sh` lives in the shared dir and is `source`d at job start,
so a job launching while a peer edits it could source a half-written file.
(Once sourced the functions are in memory, so a mid-run edit is harmless — job
7879547 sourced it at 04:24:53 and was unaffected by my 04:26 edit.)
Clean closure would be to copy the resolver INTO each snapshot so the manifest
covers it, sourcing `$SRCSNAP/srcpin_resolve.sh` after a one-line bootstrap
verify.  Low priority (the file is ~90 lines and rarely changes) but it is the
last shared mutable on the path, and it should be closed before the pattern is
adopted campaign-wide.

---

## S16 — PROVENANCE, FINAL: the deeper mechanism (zeta) + my residual, closed

### 16.1 The zeta finding, reproduced here

Excluding `__pycache__` at BUILD time is necessary but **not sufficient**: the
`.pyc` are written INTO the snapshot at RUN time by the job's own imports.
Reproduced directly:

    PYTHONPATH=<snap>/src python -c "import demo"
      .pyc before 0 -> after 1     <-- the "frozen" snapshot mutated at run time
    PYTHONDONTWRITEBYTECODE=1
      .pyc before 0 -> after 0     <-- stays frozen

**My own snapshots show the complementary half of the mechanism.** The first
one (`cp -a`, kept `__pycache__`) had **0 of 88 `.pyc` written after capture** —
because Python found valid cache and reused it. The zeta workstream excluded the
cache, so every import wrote one. **So my "fix" (excluding `__pycache__`) moved
me INTO their failure mode**; the two findings are one mechanism seen from both
sides, and `PYTHONDONTWRITEBYTECODE=1` is what closes it either way.

Why it matters for my verify-at-END specifically: `sha256sum -c` only checks
files LISTED in the manifest, so runtime-written `.pyc` are not a spurious
failure — they are **unchecked**, and the end verification passes **vacuously**
over executable bytecode that did not exist at capture. The export is what makes
that check meaningful rather than decorative.

### 16.2 Landed in `wk_REL/srcpin_resolve.sh` (inherited by every workstream)

1. **`export PYTHONDONTWRITEBYTECODE=1`** in `srcpin_resolve`, announced in the
   job log. Also added to `mos2_4x4_test/km_inner.sh` so it survives the
   srun/apptainer hop.
2. **New-file detection at END** (`rc=96`), because `sha256sum -c` is blind to
   files that APPEAR. This is the belt-and-braces for a job that forgets the
   env var — verified: a planted `__pycache__/kmeans_cli.cpython-312.pyc` gives
   `rc=96`, a mutated hashed file still gives `rc=95`, both give `rc=96`, clean
   gives `rc=0`.
3. **Retrofit warning now names the asset exposure**, per the coordinator: a
   `*.py`-only manifest is NOT "we hash the source". On this tree it leaves
   **113 of 341 files unchecked, including 31 `.npz` minimax QUADRATURE
   TABLES** — a swapped quadrature table is a **silent physics change** a
   `.py`-only manifest cannot see. My all-files manifest covers all 31.
4. **Container-interpreter trap documented**: any `.pyc` inspection must use the
   container's 3.12, not the login node's 3.7, or the magic number mismatches on
   every file (an artefact of the audit script, not tampering). Noted that the
   16-byte pyc header layout `magic|flags|src-mtime|src-size` is the same in 3.7
   and 3.12, so reading the embedded source mtime — which is what my earlier
   analysis did — is safe from either interpreter.

### 16.3 My named residual, closed: `srcpin_snapshot_v2`

The resolver was the last shared mutable on the path. `srcpin_snapshot_v2`
roots the manifest at the SNAPSHOT rather than at `src/`, so it covers
`src/**` (all file types, `__pycache__` excluded) **plus `PROVENANCE.txt` plus a
copy of `srcpin_resolve.sh` itself** — 343 hashed entries. `srcpin_resolve`
auto-detects the manifest root, so v1 snapshots still verify (backward
compatible, tested both ways). Carrying a hashed copy of the resolver is for
**auditability** — an auditor can see exactly which logic ran — not a security
boundary; the shared copy still bootstraps the verify.

### 16.4 One more trap found while testing the helper

`srcpin_verify_end` silently misbehaved when I piped `srcpin_resolve` in a test:
a pipeline runs it in a subshell, so its `export`s never reach the caller and
the end-check would verify a bogus root. That is the vacuous pass again, from a
third direction. Now a hard **`rc=97`** with the cause named ("Do NOT pipe
srcpin_resolve"). Found because I checked the function's actual return code
instead of trusting the printed FATAL — an earlier test had reported `rc=0`
while printing FATAL, which was my harness capturing `sed`'s status.

---

## S17 — ITEM: UNLOCK P=64 CENTROID GENERATION (zero-pad + active-mask)

### 17.1 The fix is SMALLER than I scoped, because the padding already existed

I scoped "pad the candidate pool to a multiple of n_dev".  Reading the builder
showed **LORRAX already does that**: `build_gram_q0_via_loadwfns` goes through
`Meta.from_system(n_rmu=M)`, and `common/meta.py:117` sets

    n_rmu_padded = padded_mu_extent(M, world_size) = round_up(M, n_dev)

with ψ pad rows zero, so **G already comes back `(M_pad, M_pad)` with the
trailing rows/cols exactly zero** — 13888 at P=64 for M=13872.  The select
simply never knew which rows were padding, so it refused instead.

**The whole fix is therefore the ACTIVE MASK**, not new padding:
* `make_sharded_pivoted_cholesky_select` gains an optional `active_init`
  operand (row-sharded bool) replacing its all-True `active` initializer.
  Pad rows start inactive → `masked_d = -inf` forever → unpickable.
* `prune_candidates_by_pivoted_cholesky` reads `M_pad = G.shape[0]`, builds the
  mask (True on `[0,M)`, False on `[M,M_pad)`), pads `orbit_id` with `-1`,
  trims `d_final`/`G` back to the logical `M`, and drops the divisibility
  refusal.
* When `n_pad == 0` (P=1, and P=16 where 16 | 13872) **no operand is passed at
  all**, so those paths take the byte-identical old code — which the p16
  control leg confirms.

Deliberately NOT relying on the tie-break: pads sit at the highest global
indices and the pivot rule takes the lowest index among ties, so they would
lose today — but that is an accident of index ordering, not a contract.

### 17.2 A real bug in my first attempt, caught by the gate (job 7879553)

My first version computed `M_pad = round_up(M, n_dev)` **from the logical M and
padded G again on top of the builder's pad** → 13872 + 16 = 13888 from the
builder, then +16 from me = **13904**, which 64 does not divide:

    jax._src.sharding.IndivisibleError: ... array axis 0 is partitioned 64
    times, but the dimension size is 13904 (full shape: (13904, 13904))

Fixed by reading `M_pad` off `G.shape[0]` instead of recomputing it, plus a
guard that refuses if `M_pad % n_dev != 0` (naming the expected value).
The p16 control leg in that same job still PASSED (`SORTED SETS
BYTE-IDENTICAL`), which is exactly the designed behaviour — at P=16 `n_pad==0`
so the pad path is not taken — and is why a single-P gate would have shipped
the bug.  **The configuration lattice caught it, not the code review.**

### 17.3 Rank-gated centroid writer

`centroid/kmeans_cli.py:501` `np.savetxt` is now `jax.process_index() == 0`.
It survived P=16 only because all ranks wrote identical bytes — the latent form
of the bug that DID bite at P=64 in `bse_io.write_eigenvectors_stream`.  Safe to
gate: no collective follows (the very next statement is the existing
rank-0-gated `timing.report`).

### 17.4 Scope call I made, and why

The coordinator also asked for column-blocking on the multi-device path (the
P=4 OOM, S9.3).  **I did not do it**, deliberately: blocking the column axis
there means slicing an axis that is SHARDED on 'y', which cannot be done
locally without a reshard — the block loop would have to move inside the
`shard_map` in `pair_density`/`gram_q0_from_pair`, i.e. a change to the physics
data path, which is not a movement-only edit and not something to land under
time pressure alongside the fix that actually unlocks capacity.  It is also not
on the path to the goal: the pair tensors are `2 x 1024 M^2 / P` bytes/rank, so
**P >= 8 already fits** and P=64 (6.2 GB/rank) fits easily.  Recommended instead
as a resolve-time refusal carrying that arithmetic (broken-promise pattern), and
sized as a follow-up.

### 17.5 RESULT — P=64 centroid generation UNLOCKED (job 7879557)

    [LEG p64 rc=0 wall=202s]
    [pivoted_cholesky] zero-pad for the sharded select: M 13872 -> 13888
        (+16 inactive rows) so M_pad % 64 == 0; pads carry d=0 and start INACTIVE
    GATE 1: SORTED SETS BYTE-IDENTICAL to the accepted P=1 set -> PASS
    MAX VmHWM = 10.80 GiB
    output files written = 1                      (rank gate holding)

GATE 2 — every prune diagnostic **bit-identical to the P=1 reference**:

| diagnostic | P=1 (7879286) | P=64 (7879557) |
|---|---|---|
| Σ_r w | 67197.1544 | 67197.1544 |
| Lloyd | 8 steps, max movement 0.000000 | identical |
| unfold | 1250 reps → 13872 (n_sym=12) | identical |
| G diag range | [7.632e-17, 1.158e-04] | identical |
| picked-pivot residuals | 1.158e-04 / 2.116e-11 / 0.000e+00 | identical |
| tr(R_k)/tr(G) | 9.330e-01 / 7.653e-07 / 1.874e-10 | identical |
| orbits → centroids | 897 → 10015 | identical |
| rank | 630 | 630 |

The `RuntimeError: picked N PAD row(s)` guard did not fire — no pad was ever
selected, which is the mask doing its job rather than the tie-break.

**Capacity delivered.**  The centroid path at N_mu = 10015:

| | P=1 (was the only route) | P=64 (now) |
|---|---:|---:|
| wall | 308 s | **202 s** |
| VmHWM/rank | **63.15 GiB** | **10.80 GiB** |
| Gram residency | 3.08 GB on ONE rank | **48 MB/rank** |
| scaling | `VmHWM = 64.4 + 17.5 M^2 bytes`, ceiling **N_c ~ 61 500** | `M^2/P` — the one-node ceiling is GONE |

So the O(N_mu^2)-on-one-rank object is now avoidable in practice, not just in
principle: it was a P=1 degeneration of a route that was already sharded, and
the only thing standing between the campaign and P-scaled centroid generation
was a refusal where an active-mask belonged.

Per the coordinator's framing, that leaves the terminal OOM — the unsharded
`(n_tau, mu_pad, mu_pad)` object in the W probe — as the only remaining
whole-object materialisation known in the pipeline.

### 17.6 P=16 control leg + job-level source verification (same job 7879557)

    [LEG p16 rc=0 wall=62s]   CHECK_REPLICA=1   MAX VmHWM = 35.53 GiB
    --- pad path taken? ---   (EMPTY — n_pad==0 at P=16, old code path)
    GATE 1: SORTED SETS BYTE-IDENTICAL to the accepted P=1 set -> PASS
    GATE 3: rank-assertion gate ON and clean (no divergence)
    output files written = 1  (rank gate holding)

    --- SOURCE IMMUTABLE FOR THE WHOLE JOB? ---
    [srcpin] MANIFEST VERIFIED at job END — no file changed AND none appeared:
    [srcpin]   source was IMMUTABLE for the whole run

So the P lattice is now {1, 16, 64} all producing the SAME centroid set, with
P=16 exercising the no-pad path and P=64 the pad+mask path, under an explicitly
pinned snapshot verified at both ends with the new-file detector armed.

### 17.7 Named, not done (small, deliberate)

* `G = G[:M, :M]` at the end of `prune_candidates_by_pivoted_cholesky` trims the
  padded Gram before returning it.  The ONLY caller (`kmeans_cli.py:481`)
  discards it via `keep_idx, rank, *_ = ...`, so this slice is pure work and on
  a row-sharded array it can force a reshard.  It ran fine at P=64 (job
  7879557), so I left it rather than churn another 32-node gate; the cleanup is
  to drop the slice and document that the returned G carries the builder's zero
  padding.
* Multi-device column-blocking (the P=4 OOM) — see 17.4 for why it was not
  attempted and what to do instead.
