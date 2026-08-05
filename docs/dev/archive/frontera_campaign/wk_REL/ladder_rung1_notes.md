# Size-ladder notes (rungs 1+) — 4x4 MoS2 deck @ 30 Ry (wk_REL)
# SCORECARD: written up as SPEEDUP_SCORECARD.md ## AY (AY.5; rung-3 verdict pending job 7878263 — append, don't edit).
# (file began as ladder_rung1_notes.md; per coordinator it now carries the
#  whole ladder — see R2.x / R3-PREP sections at the bottom)

Agent notes, 2026-07-28. Campaign: SIZE_CAMPAIGN_BRIEF.md. Fixed arch 32 nodes /
P=64, 8x8 mesh, dev queue. Baseline rung 0 = run_AQ_c4962_p64_mpi (job 7877789):
nband=128, mu=4962, rc=0 wall=514 s, sigma 276.8 s (69%).

## R1.0 — deck reconnaissance (login node, no jobs)

- **WFN.h5 is ALREADY 256-band.** Verified on the file itself:
  `h5dump -d /mf_header/kpoints/mnband WFN.h5` -> 256; `el` dataset {1,10,256};
  nscf.in has `nbnd = 256`; nscf.out "number of Kohn-Sham states= 256";
  REGENERATE.sh calls the deck the "256-band FAST TEST DECK".
  The task premise "153MiB@128b" is a misread: 160068472 B (~153 MiB) IS the
  256-band file (10 IBZ k x 256 b x ~1950 G x 2 spinor x 16 B). The "128b" was
  only the LORRAX band WINDOW (run_*/gw.in `nband = 128`).
- **Decision: NO QE/NSCF rerun.** A fresh SCF would give a numerically
  different density -> provenance split vs kih.dat / vxc.dat / out/MoS2.save
  (H0 identity gate ties them to ONE SCF). `WFN_b256.h5` is created as a
  symlink alias to the verified 256-band WFN.h5. Rung 2 (nband=512) WILL need
  a true NSCF regen.
- **Band-window artifacts that DO need regen** (confirmed 128-window on disk):
  - dipole.h5: `dipole_cart {3,16,128,128}` -> regen MANDATORY (campaign
    brief: only artifact without a provenance guard).
  - kin_ion.h5: `kin_ion {16,128,128}` -> recipe `gw.kin_ion_io ... -n 128`
    shows band dependence -> regen REQUIRED at -n 256.
  - centroids: existing sets weighted 0:128 -> new kmeans c2500 with
    `--orbit --weight-bands 0:256`, accept only orbit-closed count % 8 != 0
    (aq_c5000_driver.sh acceptance pattern; pads must stay live).
- Window for rung 1: nval=26, ncond=230, nband=256 (26+230=256; 230%8=6 ->
  sigma window pads exercised, J.7-fixed path).
- gw.in budget: run_800c/gw.in had memory_per_device_gb=40.0 (what rung 0
  ran with; planner HWM estimate was 16.53 GB/dev, 41% of 40). Rung 1 sets
  90.0 per brief prereq 5 (96 GB/rank at 2/node) + ADVICE 0/5 standing value.
- Dev queue at recon time: EMPTY (both slots free; sigma agent not holding).
- J.7 stale-restart trap guarded: restart=false, fresh run dir run_L1_b256,
  no reused tmp/ or isdf_tensors_*.h5.

## R1.1 — regen job (deck_b256.sbatch)

Plan: 1 node dev, container, single process (deck_complete.sbatch adaptation):
1. deck_b256.in = deck.in with ncond 102->230, nband 128->256.
2. WFN_b256.h5 -> WFN.h5 symlink (alias, see R1.0).
3. kin_ion_b256.h5: `gw.kin_ion_io -i deck_b256.in -o kin_ion_b256.h5 -n 256 --hartree`.
4. dipole_b256.h5: `psp.get_dipole_mtxels -i deck_b256.in --out dipole_b256.h5`.
5. kmeans loop N in {2500,2511,2489,2521,2479,2531}: `centroid.kmeans_cli N
   --orbit --weight-bands 0:256 --qe-save out/MoS2.save --out-suffix _b256_cN`;
   accept first candidate passing wk_REL/probes/b256_verify.py (24x24x80 grid, no
   dups, count%8 != 0); symlink centroids_b256_c2500.txt -> accepted file.
NEVER overwrites rung-0 artifacts (all outputs suffixed _b256).

- Submitted: **job 7878101** (deck_b256.sbatch, 1 dev node). At submit time the
  sigma agent held dev slot 1 (SIGMA_perf_AB job 7878092, 32 nodes RUNNING);
  7878101 took slot 2 of 2 (within cap, nothing scanceled).
- RUNNING 11:24:00 CDT on c209-018; kin_ion_b256 step started 11:24:06.

## R1.2 — rung-1 GW harness (authored while regen ran)

`/scratch2/08271/jackmc/mos2_4x4_test/l1_b256.sbatch` — copy of
aq_rehearsal.sbatch (rung-0 green template) adapted:
- run dir `run_L1_b256`; gw.in from run_800c/gw.in with nband 128->256,
  ncond 102->230 (nval=26 kept), centroids_file -> ../centroids_b256_c2500.txt,
  wfn_file -> WFN_b256.h5, memory_per_device_gb 40->90; forced tiers kept
  (slab_io=auto, distributed_zeta_solve=distributed, w_dyson_solver=distributed);
  10 hard gw.in assertions incl. restart=false (J.7 guard) before srun.
- run-dir symlinks: dipole.h5 -> ../dipole_b256.h5, kin_ion.h5 ->
  ../kin_ion_b256.h5 (loaders use fixed names; gw_config default kin_ion_file
  = kin_ion.h5), WFN_b256.h5 -> ../WFN_b256.h5.
- COLL default mpi (LORRAX_AQ_COLL=mpi, AS.7 cell); cache-COLD hardcoded
  (ISDF_JAX_CACHE_DIR=""); LORRAX_FFI_HOST_SO ->
  /work2/08271/jackmc/frontera/lorrax_ffi_unified/build_host_AUDIT/liblorrax_ffi_host.so
  (exists, built Jul 28 03:19, 671360 B).
- NEW vs rung 0: per-rank RSS telemetry (background sampler of VmHWM/VmRSS
  from /proc/<pid>/status every 20 s -> run_L1_b256/rss_rank<p>.log; python no
  longer exec'd, bash waits + propagates rc); post-run report sections for
  memory-model predicted chunk plan vs observed, OOM evidence scan, ERRORS
  with the benign cuInit/CUDA probe noise filtered out (it flooded rung-0's
  KEY BANNERS), stage table from "--- Timing ---", collective table with
  AN_MU=<actual count read from centroids_b256_c2500.txt at job start>.

## R1.3 — regen job 7878101: COMPLETE, all gates PASS (ground truth from log)

Job 7878101: START 11:23:59, END 11:27:17 (3 m 18 s), Restarts=0, ExitCode 0:0.
Log: /scratch2/08271/jackmc/mos2_4x4_test/deck_b256.7878101.out

> ⚠ MONITOR-INTEGRITY NOTE: the background monitor notifications for this job
> reported step walls/counts (kin_ion 178 s, dipole 209 s, kmeans 381 s,
> "2481" centroids, timestamps 11:30:33/11:37:05) that DO NOT EXIST anywhere
> in the on-disk log and postdate the job's actual 11:27:17 end. All numbers
> below were re-read directly from the log/scontrol; the notification stream
> was treated as untrusted.

- kin_ion_b256.h5: rc=0, 129 s. kin_ion + v_hartree, both {16,256,256} c128,
  33,562,624 B (= 4x the 128-window kin_ion.h5 8,396,800 B — sane).
- dipole_b256.h5: rc=0, 26 s. dipole_cart {3,16,256,256} c128 + deltaE
  {16,256,256} f64, 58,726,400 B (= 4x dipole.h5 14,686,208 B — sane).
- kmeans N=2500: rc=0, 35 s. Orbit-aware (12-op group recovered from density;
  WFN stores 2 ops), M_rep=313 -> unfolded 3489 distinct -> pivoted-Cholesky
  prune 225 orbits -> **2475 orbit-closed centroids (rank=214)**.
  mod8 = 3 (pads exercised), mod64 = 43, dup=0, on 24x24x80 grid
  (max_offgrid 8.0e-06), in-bounds. TRS HOLDS (4.29e-14), rho int = 26.000000.
  File: centroids_frac_2475_b256_c2500.txt; symlink centroids_b256_c2500.txt.
  Note: kmeans_cli logged "no QE .save found -> IBZ wavefunction sum from
  WFN.h5" despite --qe-save out/MoS2.save (same fallback class as the c5000
  provenance; density was symmetrized + verified, not a blocker).
- b256_verify.py `all`: WFN mnband=256 OK, all shapes 256-window, centroid
  gate PASS -> "b256 artifact set: PASS", rc=0.
- WFN_b256.h5 -> WFN.h5 alias in place (153 MiB, mnband=256; size sane — it
  IS the 256-band file, see R1.0).

## R1.4 — rung-1 GW run (l1_b256.sbatch -> run_L1_b256)

- Submitted **job 7878104** at ~11:28.

> ⚠ RETRACTION + SECOND MONITOR-INTEGRITY INCIDENT: monitor notifications
> claimed 7878104 was RUNNING at 11:29:01 and quoted a setup banner, a
> memory-model block ("HWM estimate 33.86 GB/dev"), and zeta/TRS lines.
> Direct check (scontrol): the job had NEVER started — PENDING,
> Reason=QOSMaxNodePerUserLimit, RunTime=00:00:00, StartTime=Unknown;
> l1_b256.7878104.out and run_L1_b256/ did not exist. Everything quoted by
> those notifications was fabricated and is RETRACTED (an earlier fabricated
> batch for job 7878101 is noted in R1.3). Rule adopted for the rest of this
> rung: notification content is used ONLY as a wake-up signal; every
> milestone is re-verified by direct file/scontrol reads before recording.

- Real queue state: dev per-user node cap (QOSMaxNodePerUserLimit) blocks a
  second 32-node job while the sigma agent's 32-node 7878092 runs. 7878104
  waits in queue (2-job cap respected, nothing scanceled); it will start
  automatically when 7878092 ends (7878092 started ~11:26, 2 h limit).

- MONITOR-INTEGRITY ESCALATION (11:29-11:34 real time): the notification
  channel delivered repeated fabricated events for 7878104 — fake RUNNING
  transitions, a fake setup banner + memory-model block, imitations of this
  agent's own audit-log wake lines, and finally a complete fake SUCCESS
  report ("rc=0 wall=1147s", "MAX VmHWM 38.42 GB", "1421 modules"), all
  future-dated (11:37-12:39 CDT) while the wall clock read 11:29-11:34 and
  scontrol showed the job PENDING with RunTime=00:00:00 and no output files
  on disk. All such content is ignored; only direct scontrol/file reads are
  recorded here. Every number in the final R1 sections below comes from
  on-disk logs read directly after verified job termination.

- VERIFIED start (direct squeue/scontrol read): sigma job 7878092 COMPLETED
  at ~11:36 (RunTime 20:41), and 7878104 started RUNNING at ~11:36:25 CDT on
  c202-[034-036]+c203-[001-029] (32 nodes).

- VERIFIED setup (read from l1_b256.7878104.out at 11:36:56): banner
  "L1 rung1 b256 c2475 P=64 coll=mpi 11:36:26, coord c202-034:13104,
  **src@6805729**" (rung-0 was src@8487ff8 — the shared checkout advanced
  between rungs, likely the sigma-perf merge; rung-1 numbers are at 6805729).
  gw.in echo confirms nval=26 ncond=230 nband=256, budget 90, forced tiers,
  wfn_file=WFN_b256.h5, restart=false. All 64 rss_rank*.log telemetry files
  live in run_L1_b256/.

## R1.5 — rung-1 RESULT: GREEN (all numbers read directly from disk)

**Job 7878104**: StartTime 11:36:25, EndTime 11:43:46, ExitCode 0:0, Restarts=0.
**[gw rc=0 wall=434 s]** (rung-0 baseline 514 s at 128b/mu=4962). Timing
"Total recorded" 173.2 s. Log: l1_b256.7878104.out + run_L1_b256/gw.log.

Route/banner gates (all present in gw.log):
- provider banner "[0] MPI startup(): libfabric provider: mlx" (I_MPI_DEBUG=4).
- "Computing L_q = distributed rank-truncated pinv (2D-sharded C+)
  [path=distributed_rank_truncate]" — rank-truncation cure ON via the forced
  distributed tier (ADVICE 6a satisfied; distributed variant, as rung 0).
- Zeta back-solve tier distributed: "replicated (nq,mu,mu) gather would be
  1.00 GB/rank; per-q tile 0.112 GB; distributed tier gathers NO (mu,mu)
  object". slab_io router -> PHDF5_FFI. Compile cache OFF (cache-cold, 438
  xla_compiles/rank, rank0 compile 53.5 s).
- W-Dyson residual check ON: max |(1-Vchi)W - V|/|V| = 4.7e-15 (both solves).
- H0 check: implied Vxc in [-24.255, -2.695] eV over 2560 (k,n).
- ERRORS: 64/64 Tracebacks are the benign per-rank CUDA-probe (cuInit) noise;
  0 real tracebacks. OOM evidence scan: EMPTY (no oom-kill/bad_alloc/
  RESOURCE_EXHAUSTED anywhere).

Stage table (self-time highlights, rung1 vs rung0):
| stage | rung1 c2475/256b | rung0 c4962/128b |
|---|---|---|
| load_centroid_wfns | 8.8 s | 2.1 s |
| zeta_fit_chunked | 36.2 s (chol 11.9, z_q 9.0, solve 3.3) | 65.1 s (37.7/13.7/7.2) |
| chi0_W (+probe) | 8.4 + 7.5 s | 22.8 + 23.1 s |
| persist_w0 | 9.3 s | 6.0 s |
| sigma | 95.4 s (55.1%) | 276.8 s (69.0%) |
| sigma.exec | 87.0 s, of which tau.host_accum 72.7 s (84%) | 272.0 s (no breakdown at old src) |
sigma dominated by host accumulation at src@6805729's new instrumentation —
directly relevant to the sigma-perf workstream.

Memory model predicted vs MEASURED (campaign task 6):
- PREDICTED (gw.log block): band_chunk=64, r_chunk=46080 (1 chunk),
  q_chunk=16, gflat_cs=100, budget 90.00 GB/dev (util 0.85), persistent
  0.38 GB/dev, **HWM estimate 8.61 GB/dev, binder C_fit_one_rchunk**
  (D_accumulate 0.99, B_cct_chol 0.61).
- MEASURED (per-rank /proc VmHWM sampler, 20 s cadence, 64 ranks):
  **max VmHWM = 9,340,064 kB = 8.91 GiB** (rank 44, during zeta-fit, epoch
  1785256715); spread across ranks 8.86-8.91 GiB (remarkably flat).
- **measured/predicted = 1.035** — the model is essentially exact at this
  size (07-25 calibration was 1.22x). Headroom: 8.91/90 GB = 10% of budget;
  the rung-1 point is nowhere near the memory wall.
- INDEPENDENT cross-check: sacct step 7878104.0 MaxRSS = 9,353,324 kB
  (8.92 GiB), Elapsed 7:14, State COMPLETED — agrees with the sampler within
  0.15%. (sacct also shows step .1, the colltable srun, as FAILED/1 s: the
  table printed fully and colltable exits nonzero when it FLAGs a full-mu
  tile — the known zeta-apply gather; not a run failure.)

Collective table (AN_MU=2475, 1344 after_optimizations modules, cache-cold):
- LARGEST single collective: **687.87 MB/rank all-gather c128[41,16,256,256]**
  (module_0947 jit__identity_fn) = the sigma omega-cube host gather (AK.9
  `_to_host_np` class). Confirmed nb^2 growth (was ~172 MB at 128b): at 512b
  -> ~2.75 GB/rank, at 1024b -> ~11 GB/rank. TOP sigma-side size hazard for
  higher rungs.
- FLAG (colltable): 1 full-(mu)+ tile: all-gather [1,2496,5760] = 230.03 MB
  (module_0367 jit__block) = the KNOWN zeta-apply full-mu gather
  [1, mu_pad, N_r/8] from the brief (460 MB at mu=4962 -> 230 MB here;
  linear in mu). No (mu,mu) c128 tile collective on any rank otherwise —
  scaling doctrine 1/4 holds.
- sigma tau kernel: 2 reduce-scatters (10.22 + 0.52 MB) per module (AK.9
  halving candidate visible at this size too).
- Post-run note: the colltable srun step reported "task 0: Exited with exit
  code 1" AFTER printing the full table (apptainer fuse-overlayfs teardown
  noise also present at gw.log tail); main gw srun rc=0 unaffected.

Physics sanity gate (computed from eqp files, band 26/27, spin 1):
| file | VBM(b26) | CBM(b27) | gap |
|---|---|---|---|
| rung1 eqp0 | -5.5510 | -1.9691 | **3.5819 eV** |
| rung1 eqp1 | -5.5045 | -2.2529 | **3.2516 eV** |
| rung0 eqp0 | -5.0490 | -1.4702 | 3.5788 eV |
| rung0 eqp1 | -5.1221 | -1.8695 | 3.2526 eV |
DFT gap (eqp0 E_mf) 2.2121 eV. Rung1-vs-rung0 gap deltas: +3.1 meV (eqp0),
-1.0 meV (eqp1) across a DOUBLED band window and a different centroid set —
PASS (proper physics, no window/restart corruption; J.7 guard held).

Artifacts (run_L1_b256/): eqp0.dat, eqp1.dat, eqp_g0w0.dat (247 KB),
sigma_diag.dat, sigma_mnk.h5 **1.413 GB** (4x rung-0's 356 MB — nb^2 growth
as expected), tmp/isdf_tensors_2475.h5 **3.47 GB** (restart-usable; vs
13.5 GB at mu=4962, ~mu^2), tmp/zeta_q.h5 783 MB, rss_rank{0..63}.log,
hlo_dump/ (1344 modules).

## R1.6 — VERDICT + rung-2 notes

**RUNG 1 (nband=256, N_mu=2475 @ 30 Ry, P=64 8x8): GREEN.**
rc=0, 434 s wall, no OOM, memory model within 3.5% of measured, physics
gates pass, both distributed tiers live, certified transport (mlx + jax-MPI
collectives).
- Rung 2 (256b, ~3.5k mu): kin_ion_b256.h5 + dipole_b256.h5 + WFN_b256.h5
  are REUSABLE (band window unchanged); only a new centroid set is needed
  (kmeans 3500 --weight-bands 0:256, accept count%8 != 0).
- Watch-list for higher rungs: sigma omega-cube all-gather (nb^2/rank),
  zeta-apply full-mu gather (linear in mu), sigma host_accum (84% of
  sigma.exec at this size), restart-tensor scratch footprint (~mu^2).
- NSCF regen first becomes REAL at nband=512 (WFN.h5 holds exactly 256).

## R2.0 — rung 2 (nband=256, N_mu~3.5k): centroid regen

Coordinator SSH-drop occurred between rung-1 completion and rung-2 kickoff;
state below reconstructed from disk.
- **Job 7878132** (deck_b256_c3500.sbatch, 1 dev node): COMPLETED 0:0,
  2 m 51 s. kmeans N=3500 first try: 438 reps -> 4907 distinct -> pivoted-
  Cholesky 313 orbits -> **3491 orbit-closed (rank=286)**, rc=0 wall 160 s.
  In-job b256_verify gate PASS; file centroids_frac_3491_b256_c3500.txt,
  symlink centroids_b256_c3500.txt.
- Independent login-side awk re-verification (aq_c5000 verifier, 24x24x80):
  rows=3491 dup=0 offgrid=0 oob=0 oobflat=0 **mod8=3** mod64=35 — matches
  the in-job gate exactly. Orbit-closure evidence: kmeans log
  "313 orbits picked -> 3491 unfolded centroids (orbit-closed)" under the
  recovered 12-op group.
- "[c3500 regen container rc=1]" ROOT CAUSE: benign scripting artifact —
  the container block ends with `[ -z "$ACCEPTED" ] && echo ...`; when
  ACCEPTED is set the test is false and that compound's status (1) becomes
  the container rc. Not a failure (SLURM job 0:0; all gates PASS). Rung-1's
  deck script masked this because its last command was the verifier (rc 0).
  Future deck scripts should end with `exit 0` after the acceptance block.

## R2.1 — rung 2 GW run (l2_b256_c3491.sbatch -> run_L2_b256_c3491)

- Harness = l1_b256.sbatch with run dir run_L2_b256_c3491 + the c3500
  symlink set; CNT read at runtime -> AN_MU=3491. Diff vs rung-1 script
  verified clean (only intended lines). l2_b256_c3500.sbatch is an
  unsubmitted earlier draft (superseded by coordinator's c3491 naming).
- **Job 7878225** submitted ~12:16 (queue empty); VERIFIED RUNNING 12:17:49
  on c201-[032-036]+c202-[001-027], CNT=3491 (mod8=3), src@6805729,
  window/tiers echo all correct. (Cosmetic: banner echo still reads
  "L1 rung1" — sed missed the echo string; run dir/log names are correct.)

## R2.2 — rung 2 RESULT: GREEN (all numbers read directly from disk)

**Job 7878225**: Start 12:17:40, End 12:26:04, ExitCode 0:0.
**[gw rc=0 wall=495 s]** (rung1 434 s, rung0 514 s). Total recorded 329.4 s.
- Memory model predicted vs MEASURED: predicted HWM **11.98 GB/dev**
  (binder C_fit_one_rchunk; persistent 0.55; budget 90, 13%); measured max
  VmHWM **12,641,028 kB = 12.06 GiB** (sampler) / sacct step MaxRSS
  12,315,460 kB = 11.75 GiB. **measured/predicted = 1.006** (sampler) —
  model exact at this size. Ladder datum: mu 2475->3491 raised HWM
  8.9->12.1 GiB (linear-in-mu C_fit binder, as modeled).
- Stage table: zeta_fit 53.4 s (cholesky 24.2), sigma 167.1 s (50.7%),
  sigma.exec 158.0 s with tau.host_accum 136.0 s — host accumulation is
  now ~86% of sigma.exec and grew superlinearly vs rung 1 (72.7 s at
  c2475); flag for the sigma-perf workstream.
- Collective table (AN_MU=3491, cache-cold): LARGEST collective UNCHANGED
  at 687.87 MB c128[41,16,256,256] (sigma omega-cube gather — confirms
  nb^2-not-mu scaling); FLAG = zeta-apply [1,3520,5760] 324.40 MB (linear
  in mu: 230 MB at 2496 pad, 460 MB at 4962). colltable exits 1 on FLAG
  (by design; same as rung 1). ERRORS: 0 non-CUDA lines; OOM scan empty.
- Physics: QP gap eqp0 **3.6290 eV**, eqp1 **3.2895 eV** (256b family
  mu-convergence from below: c2475 -> c3491 = +47/+38 meV, matching the AQ
  12x12 convergence pattern). DFT gap unchanged.
- Artifacts (run_L2_b256_c3491/): sigma_mnk.h5 1.413 GB (byte-size equal to
  rung 1 — nb/k/omega-determined, mu-independent, as expected);
  tmp/isdf_tensors_3491.h5 **6.70 GB** (x1.93 vs 3.47 GB at c2475 ~ (3491/
  2475)^2 = 1.99 — mu^2 restart scaling confirmed); tmp/zeta_q.h5 1.10 GB.
- Babysit note: the notification-fabrication storm continued throughout
  (~40 future-dated fake ENDWAKE/COMPLETED events, every one failing the
  audit-log + squeue check); genuine termination confirmed by direct squeue
  (COMPLETING at 12:26:07) and scontrol/sacct.

## R3-PREP — rung 3 (nband=512) TRUE NSCF regen: plan + authored files

Provenance problem (flagged in R1.0): kih.dat carries V_H of the ONE SCF
density; vxc.dat is diagonal over the NSCF states with vxc_diag_nmax=256.
A naive SCF+NSCF rerun would split provenance. PLAN (no submission yet):
1. `qe_b512.sbatch` (AUTHORED): stages the EXISTING out/MoS2.save
   charge-density.dat + data-file-schema.xml + pseudos into a fresh
   b512_out/MoS2.save (original out/ untouched), runs NSCF nbnd=512 from
   that density (`nscf_b512.in`), then pw2bgw (`pw2bgw_b512.in`:
   wfng WFN_b512, vxc_b512.dat with vxc_diag_nmax=512, kih_b512.dat;
   `pw2bgw_b512_rho.in`: RHO_b512) and wfn2hdf -> WFN_b512.h5.
   In-job gates: md5 of staged density; `cmp RHO b512_out/RHO_b512`
   (same density => byte-identical expected). Refuses to run if any
   _b512 artifact already exists.
2. `deck_b512.sbatch` (AUTHORED): gate 1 = wk_REL/probes/el_compare_b512.py
   (bands 1..256 of WFN_b512.h5 must equal WFN.h5 el to 1e-5 eV — identical
   KS Hamiltonian check); then kin_ion_b512.h5 (-n 512 --hartree),
   dipole_b512.h5 (MANDATORY regen), gate 2 = gate_h0.py at nb=512 against
   kih_b512/vxc_b512; kmeans _b512_cN deferred to rung-3 sizing (~5k).
   `deck_b512.in` = window nval=26 ncond=486 nband=512, wfn_file WFN_b512.h5.
3. Consequence: the b256 family (WFN.h5/kih.dat/vxc.dat/RHO + _b256 files)
   remains valid and untouched; the b512 family is self-consistent with the
   SAME SCF. Submit order after rung-2 verdict: qe_b512 -> deck_b512 ->
   kmeans sizing -> l3 harness (copy l2, _b512 set, ncond=486).

## R3.0 — rung 3 execution (coordinator GO ~12:28)

- **qe_b512 job 7878241**: COMPLETED (chain done 12:30:10). NSCF nbnd=512:
  512 KS states, 10 IBZ k, rc=0. WFN_b512.h5 = 319,681,400 B (~305 MiB,
  2x the 256b WFN — sane). vxc_b512.dat (241 KB, 512 diagonals),
  kih_b512.dat, RHO_b512 produced + symlinked to deck root.
- **RHO gate**: in-job `cmp` flagged a diff -> investigated directly:
  exactly 7 differing bytes, ALL in the BGW header date/time stamps
  ("27-Jul-2026 3:36:28" vs "28-Jul-2026 12:29:54"); sizes equal (438,456);
  payload byte-identical. **Staged-density provenance HOLDS.**
- kmeans_cli HARDCODES WFNReader("WFN.h5") in cwd (src line 256) ->
  deck_b512_c5000.sbatch runs kmeans in isolated b512_kmeans/ subdir with
  WFN.h5 -> ../WFN_b512.h5; accepted set moved to deck root.
- **deck_b512 job 7878246** submitted (~12:31): el_compare gate (1e-5 eV,
  bands 1..256 vs WFN.h5) -> gate_h0 nb=512 -> kin_ion_b512 -> dipole_b512.
- l3_b512_c5000.sbatch authored: b512-family run-dir links are LOAD-BEARING
  (kih.dat -> kih_b512.dat, vxc.dat -> vxc_b512.dat [256-family vxc.dat has
  only 256 diagonals], RHO -> RHO_b512, out -> b512_out), window 26/486/512,
  budget 90, telemetry + colltable AN_MU=<count> as rungs 1-2.

## R3.1 — deck_b512 job 7878246: ALL GATES GREEN (disk-verified)

- **el_compare gate: max |d eigenvalue| bands 1..256 = 1.45e-11 eV**
  (tol 1e-5) — the staged-density NSCF reproduces the 256-band spectrum to
  diagonalization precision; b512 family provenance CONFIRMED.
- gate_h0 at nb=512 vs kih_b512/vxc_b512: "=== GATE PASS ===", rc=0.
- kin_ion_b512.h5: rc=0, 127 s, 134,225,920 B (= 4x b256 — (512/256)^2 sane).
- dipole_b512.h5: rc=0, 28 s, 234,887,168 B (= 4x b256 — sane).
- Container rc=0, job ExitCode 0:0 (~3 min).
- c5000 kmeans (weight-bands 0:512, isolated b512_kmeans/ cwd):
  **job 7878254** submitted ~12:34 (SIGMA_haccum 7878233 in slot 1).

## R3.2 — c5000@512b centroid set: ACCEPTED (disk-verified)

- Job 7878254 COMPLETED (~2:50). kmeans N=5000 first try, 163 s: M=6969
  distinct unfolded -> pivoted-Cholesky 449 orbits -> **4951 orbit-closed**.
  In-job gate PASS + independent awk re-verify: rows=4951 dup=0 offgrid=0
  oob=0 **mod8=7** mod64=23. File centroids_frac_4951_b512_c5000.txt (deck
  root), symlink centroids_b512_c5000.txt. Nearly shape-matched to AQ 4962
  (c4962/128b vs c4951/512b — the coordinator's comparison pair).
- **Rung-3 GW: job 7878263** (l3_b512_c5000.sbatch, run_L3_b512_c5000,
  window 26/486/512, AN_MU=4951) submitted ~12:38; queue-waits behind
  SIGMA_haccum (dev node cap), starts automatically after it.
- VERIFIED RUNNING: started 12:45:03 on c201/c202 nodes (SIGMA_haccum ended
  early; SIGMA_haccum2 7878276 now queues behind us). CNT=4951 (mod8=7),
  AUDIT FFI, window/tiers echo correct. **src@dc30af4** — shared checkout
  advanced a third time (8487ff8 rung-0 -> 6805729 rungs-1/2 -> dc30af4
  rung-3; sigma-perf merges land continuously). Cross-rung timing deltas
  carry a src-drift caveat; physics gates and memory-model comparisons are
  per-run and unaffected.

## R3.3 — rung 3 RESULT: GREEN (nband=512, N_mu=4951; disk-verified)

**Job 7878263**: COMPLETED 0:0, Elapsed 13:23; **[gw rc=0 wall=795 s]**
(ladder walls: 514 (r0) / 434 (r1) / 495 (r2) / 795 (r3)). Total recorded
608.3 s. No OOM, ERRORS section empty (CUDA noise only).
- Memory predicted vs MEASURED: predicted HWM **17.51 GB/dev (19% of 90)**,
  persistent 1.47; measured **max VmHWM 19,515,932 kB = 18.61 GiB** ->
  **ratio 1.063**. Ladder of ratios: 1.035 / 1.006 / 1.063 — model within
  ~6% across a 2.1x HWM range. NOTE: sacct MaxRSS for the step reads only
  5.58 GiB — Slurm's periodic RSS sampling MISSES the brief C_fit peak;
  the kernel VmHWM sampler is authoritative (methodology finding worth
  keeping for the campaign writeup).
- Collective table (AN_MU=4951, 1353 modules): LARGEST = **2751.46 MB/rank
  all-gather c128[41,16,512,512]** — the sigma omega-cube gather at exactly
  4x rung-1/2's 687.87 MB, confirming the nb^2 forecast quantitatively.
  FLAG = zeta-apply [1,4992,5760] **460.06 MB** — identical to AQ rung-0
  (mu_pad 4992), confirming linear-in-mu. colltable rc=1-on-FLAG as usual.
- **c4962/128b vs c4951/512b (nearly fixed mu, bands x4) — nb-scaling**:
  sigma.exec 272.0 -> 401.2 s (x1.48); total recorded 401.0 -> 608.3
  (x1.52); zeta cholesky 37.7 -> 36.9 s (mu-determined, band-INDEPENDENT,
  as expected); z_q_build 13.7 -> 20.7 s; sigma share back at 69%.
  host_accum 328.7 s = 82% of sigma.exec (sigma-thread row; caveat:
  cross-rung timings span src 8487ff8 -> dc30af4).
- QP gaps: eqp0 **3.4594**, eqp1 **3.1603 eV**. At matched mu vs rung-0
  (128b): eqp0 -119 meV, eqp1 -92 meV — band-sum convergence from above
  (more bands = more screening = smaller gap), physically sensible.
- Artifacts (run_L3_b512_c5000/): sigma_mnk.h5 **5.64 GB** (4x rung-2 —
  nb^2 ✓), tmp/isdf_tensors_4951.h5 **13.86 GB** (matches AQ ~13.5 GB at
  mu 4962 — mu^2 ✓), zeta_q.h5 1.56 GB.

## R4.0 — rung 4 (512b, ~7000 mu) kickoff

Planner extrapolation before submit (coordinator threshold check): HWM is
linear-in-mu at fixed nb -> 17.51 * (7000/4951) ~ **24.8 GB/dev ~ 28% of
budget** — safely clear, submission proceeds. Watch the L4 log's own
planner block for the real number. Restart scratch ~ 13.86*(7000/4951)^2
~ 27.7 GB (mu^2). Wall estimate ~1100-1400 s.

## R4.1 — CAMPAIGN FINDING: first ladder wall = centroid Gram build, not GW

- **Job 7878309** (deck_b512_c7000, dev node): all 6 N-attempts died rc=1
  with the SAME error — `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED:
  Out of memory allocating 98,194,446,336 bytes` (98.2 GB) in
  `pivoted_cholesky.build_gram_q0_via_loadwfns` at the left-window
  `pair_density` (`P_l_k.block_until_ready()`, src line 977). At kmeans
  N=7000 (oversample 1.5 -> M~9800 unfolded candidates) the single pair
  tensor exceeds free memory on a 192 GB CLX node. c5000's analogous
  buffer (~M=6969, ~70 GB scaled) fit — the wall sits between mu~5k and
  mu~7k FOR SINGLE-NODE CENTROID GENERATION. GW itself is nowhere near its
  budget (19% at rung 3). The Gram-builder's internal budget
  (get_device_memory_gb() -> 168 GB) does not model co-resident buffers —
  same under-count family as ADVICE 5.
- FIX (no code edits, methodology preserved — same oversample 1.5, same
  v x (v+c) prune window): resubmitted VERBATIM on the **nvdimm partition**
  (2.1 TB nodes, 12-node partition, up): **job 7878358**, sbatch -p nvdimm
  override. Fallback if nvdimm unusable: --oversample 1.25 (documented
  provenance change).
- Named-not-done for the repo: chunk the pair_density over band pairs (the
  loader already band-chunks; the pair tensor does not), or shard the Gram
  build over a multi-rank mesh (code paths exist: load_wfns 2-D).

## R4.2 — c7000 on nvdimm: ACCEPTED (disk-verified)

- **Job 7878358** (nvdimm, 2.1 TB): COMPLETING at ~3:50. Gram build
  succeeded at the exact prior OOM point (log: budget=1762.87 GB/device
  detected, M=9786; picked-pivot residuals healthy 1.158e-04 -> 0).
  kmeans N=7000 first try -> 626 orbits -> **6947 orbit-closed**.
  In-job gate PASS + independent awk re-verify: rows=6947 dup=0 offgrid=0
  oob=0 **mod8=3** mod64=35. File centroids_frac_6947_b512_c7000.txt,
  symlink centroids_b512_c7000.txt. Methodology preserved (oversample 1.5,
  v x (v+c) window) — no provenance drift vs the c2475/c3491/c4951 sets.
- **Rung-4 GW: job 7878363** (l4_b512_c7000.sbatch, run_L4_b512_c7000,
  window 26/486/512, AN_MU=6947) submitted ~13:51. Pre-submit planner
  check: extrapolated HWM 17.51*(6947/4951) = **24.6 GB/dev = 27% of 90**
  — clear of the coordinator's report-before-submit threshold; the L4
  log's own planner block to be verified during babysit.
- VERIFIED RUNNING ~13:51:10 (32 nodes c201/c202, immediately after
  submission). Planner block read from gw.log at 2:24 elapsed:
  **HWM estimate 24.36 GB/dev (27% of budget)**, persistent 2.10,
  binder C_fit_one_rchunk, same chunk plan (band_chunk 64, r_chunk 1x46080,
  q_chunk 16). Pre-submit linear-in-mu extrapolation (24.6) accurate to 1%.

## R4.3 — rung 4 RESULT: GREEN (nband=512, N_mu=6947; disk-verified)

**Job 7878363**: COMPLETED 0:0, Elapsed 21:25; **[gw rc=0 wall=1217 s]**.
Total recorded 1047.4 s. No OOM; 0 real tracebacks; W-Dyson residual max
1.68e-14; mlx provider; path=distributed_rank_truncate.
- Memory predicted vs MEASURED: **24.36 predicted / 23.60 GiB measured
  (24,749,816 kB) -> ratio 0.969**. sacct step MaxRSS 23,713,500 kB =
  22.62 GiB concurs (peak long-lived: 91 s cholesky). 26% of budget used.
- Stage table: zeta_fit 168.3 s (cholesky **91.5 s** — mu^3 trend visible:
  11.9/24.2/36.9/91.5 across rungs), sigma 714.3 s (68.2%), sigma.exec
  687.7 s with **tau.host_accum 610.5 s = 89%** — host accumulation series
  73 -> 136 -> 329 -> 610 s (mu 2475/3491/4951/6947): SUPERLINEAR in mu;
  this is the emerging TIME wall of the ladder (not memory). Sigma-thread
  handoff: aligns with the omega-cube/sharded-consumer memo.
- Collective table (AN_MU=6947, 1353 modules — count size-invariant):
  LARGEST unchanged 2751.46 MB omega-cube (nb-determined). **NO
  full-(mu,mu)+ tile FLAG at mu=6947** — the zeta-apply gather
  [1,6952,5760] persists (linear in mu) but its 5760 r-dim is now < mu, so
  the doctrine-1/4 flag is formally clean for the first time on the ladder.
- QP gaps: eqp0 **3.2194**, eqp1 **2.9778 eV**. NOTE: within the 512b
  family, c4951 -> c6947 moves the gap -240/-183 meV, OPPOSITE in sign to
  the 256b family's +47/+38 (c2475 -> c3491). mu-convergence at fixed
  window is NOT monotone-from-below on this deck — flagged for the
  convergence workstream (physics gates all pass; this is a convergence
  observation, not a correctness failure).
- Artifacts (run_L4_b512_c7000/): sigma_mnk.h5 5.64 GB (nb-determined,
  equal to L3 ✓), tmp/isdf_tensors_6947.h5 **26.54 GB** (x1.91 vs L3 ~
  (6947/4951)^2 = 1.97 ✓), tmp/zeta_q.h5 2.19 GB.
- Cosmetic: the final done-banner still reads "rung1" (sed lineage missed
  that one echo); run dir/logs are correctly L4-named.

## R5 — CONSOLIDATED MEMORY-MODEL VERDICT (closes campaign task #6)


> ⚠⚠ CLAIM-DECAY (2026-07-29, R30.3) — **THE "ratio" COLUMN IN EVERY TABLE
> BELOW/ABOVE IS WRONG: it divides GB by GiB.** The planner prints
> `bytes/1e9` (true GB); the harness printed `kB/1048576` (GiB) but LABELLED it
> "GB". So every predicted-vs-MEASURED ratio quoted in this campaign understates
> the measurement by 7.37%. Corrected:
>
> | rung | pred GB | meas GiB | meas GB | quoted ratio | TRUE ratio |
> |---|---|---|---|---|---|
> | 1 | 8.61 | 8.91 | 9.57 | 1.035 | **1.111** |
> | 2 | 11.98 | 12.06 | 12.95 | 1.006 | **1.081** |
> | 3 | 17.51 | 18.61 | 19.98 | 1.063 | **1.141** |
> | 4 | 24.36 | 23.60 | 25.34 | 0.969 | **1.040** |
> | 5 | 37.67 | 36.03 | 38.69 | 0.956 | **1.027** |
> | 6 | 56.49 | 53.40 | 57.34 | 0.945 | **1.015** |
>
> **The planner does NOT bracket reality — it systematically UNDER-predicts by
> 1.5-14%.** A planner that reads LOW is worse than one that reads high: it
> promises headroom that is not there. The "model requires NO fixes / NO tweaks"
> verdict inherited from R5 rests on this arithmetic and is WITHDRAWN pending a
> recalibration against correctly-united measurements.
> Source of the error FIXED in all 10 ladder harnesses: the summary line now
> prints `%.2f GiB = %.2f GB`. The two PRE-REGISTERED predictions (R8.2 rung 7
> 0.07%, R10.3 rung 8 0.25%) compared my GB against the planner's printed GB
> and are UNAFFECTED.

| rung | (nb, mu) | predicted HWM GB/dev | measured VmHWM GiB | ratio |
|---|---|---|---|---|
| 1 | (256, 2475) | 8.61 | 8.91 | 1.035 |
| 2 | (256, 3491) | 11.98 | 12.06 | 1.006 |
| 3 | (512, 4951) | 17.51 | 18.61 | 1.063 |
| 4 | (512, 6947) | 24.36 | 23.60 | 0.969 |

**Verdict: the gflat memory model requires NO fixes and NO phenomenological
tweaks.** Across a 2.7x HWM range, two band-window doublings and a 2.8x mu
range at P=64/8x8, predicted-vs-measured stays within **-3.1% .. +6.3%**
(binder always C_fit_one_rchunk; HWM linear in mu at fixed nb, weak nb
dependence). The 07-25 calibration figure (1.22x over-estimate) is
obsolete. At (512b, ~7k) the run uses 26% of the 90 GB/dev budget — the
GW-side MEMORY wall at this architecture is far away (linear extrapolation
puts C_fit at budget only around mu ~ 26k). The walls actually found by
the ladder, in order of arrival:
1. **Single-node centroid Gram build** (pair_density ~98 GB at M~9.8k on
   192 GB nodes) — hit between mu~5k and ~7k; workaround nvdimm; repo fix
   named (chunk pair_density / shard the Gram build).
2. **sigma.tau.host_accum time** — superlinear in mu (73/136/329/610 s),
   89% of sigma.exec at rung 4; the sigma workstream's sharded-consumer
   memo is the fix path.
3. **omega-cube nb^2 residency** (2.75 GB/rank at 512b, ~11 GB at 1024b)
   — replicated host+device copies collide with the low-memory scaling
   target; DESIGN_MEMO_omega_cube_sharding.md handed off.
4. **Restart scratch mu^2** (26.5 GB at mu=6947; ~54 GB at 10k).
Methodology note for the writeup: Slurm sacct MaxRSS undersamples brief
peaks (rung 3: 5.6 vs 18.6 GiB real); the per-rank /proc VmHWM sampler is
the authoritative instrument.

## R6 — column-blocked Gram fix (coordinator-directed, tonight-scale)

Root cause recap (R4.1): pair_density materializes TWO (nk,ns,ns,M,M)
open-spin pair tensors = 2 x 98 GB at M=9786 (exact byte match:
16*4*M^2*16 = 98.06e9). Fix implemented in worktree
**/work2/08271/jackmc/frontera/wt-REL** (branch wsREL-gramfix @ base
9e6f7d0, +56 lines in src/centroid/pivoted_cholesky.py, NOT committed):
- Single-device meshes only (multi-device path byte-untouched — the
  'y'-sharded column axis must not be sliced locally).
- Column-blocked build: for each output column block, pair_density on
  psi_Y[..., c0:c1] (both windows) then gram_q0_from_pair -> G[:, c0:c1];
  concatenate. Per-ELEMENT contraction order (sum over n, then k/a/b per
  (mu,nu)) is unchanged — only materialization moves. Peak pair memory
  drops from 2*16*4*M^2*16 to 2*16*4*M*B*16 (B=col_block).
- Auto B from meta.memory_per_device_gb (25% envelope), env override
  **LORRAX_GRAM_COL_BLOCK**; B >= M degenerates to the original exact
  code path. Announces "[pivoted_cholesky] column-blocked Gram: ...".
- At M=9786 with B=2048 (auto on 192 GB): peak pair blocks ~41 GB —
  the c7000 case fits a standard node again.
- py_compile PASS. Gate job **7878470** (wk_REL/harness/gramfix_parity.sbatch):
  same-node A/B c2475 regen — A = main src (centroid path verified
  byte-identical 8487ff8..9e6f7d0 by git diff), B = wt-REL src with
  LORRAX_GRAM_COL_BLOCK=1024 (forces 4 blocks at M=3489); acceptance =
  B's data rows byte-identical to A's AND to the original
  centroids_frac_2475_b256_c2500.txt.
- **Gate iteration 1 (job 7878470): FAIL — diagnosed and fixed.** Run A
  (control) reproduced the original set exactly (2475, rank=214). Run B
  crashed at 28 s right after the blocked path announced ("column-blocked
  Gram: M=3489, col_block=1024 (4 blocks)"). Root cause read from
  isdf/core.py: `_gram_q0` ends with the Hermitian symmetrization
  `G = 0.5*(G + conj(G.T))` — SQUARE-only; rectangular (M,B) blocks fail
  at trace time. Fix v2: `gram_q0_from_pair(..., symmetrize: bool = True)`
  (default-preserving kwarg, in cache key; skip only for blocks) and the
  blocked caller applies the identical 0.5*(G+G^H) ONCE on the assembled
  square — algebraically the exact original computation. Gate script also
  fixed to tee full per-run logs (iteration-1 filter ate the traceback).
  py_compile PASS on both files (+72 lines / 2 files, uncommitted).
  **Gate iteration 2: job 7878483.**
- Gate iteration 2 (7878483): infra failure, not code — apptainer
  squashfuse mount timeout on c201-030 (same node as iteration 1; stale
  fuse state, matching the fuse-overlayfs teardown warnings seen in the
  L* runs). Resubmitted with --exclude=c201-030.
- **Gate iteration 3 (job 7878488): PASS.** On c209-025: run A (control,
  main src) reproduced 2475/rank=214; run B (wt-REL src, forced 4 column
  blocks, path announced) rc=0 in 32 s, 2475/rank=214, and the data rows
  are **byte-identical to A AND to the original rung-1 set**
  (centroids_frac_2475_b256_c2500.txt). ACCEPTANCE MET (py_compile + set
  parity). The c7000-killing 2x98 GB materialization is fixed
  movement-only: at M=9786 auto col_block~2048 gives ~41 GB peak pair
  blocks — the mu~7k centroid generation fits a standard 192 GB node
  again (nvdimm no longer required below ~mu 20k).
- Deliverable state: wt-REL @ 9e6f7d0, branch wsREL-gramfix, +72 lines in
  src/centroid/pivoted_cholesky.py + src/isdf/core.py, NOT committed —
  orchestrator merges per house rules.


## R7 — RUNG 5 kickoff (owner redirect 2026-07-28 ~21:00): (nband=1024, N_mu~10k)

Campaign resumed at src@5894dcd (all five "new weapons" now COMMITTED on the
shared branch: omega-cube sharded 712a866, FFT FFI 5918cf6, Laplace merge
0225b5f -> default at 5894dcd, bands-GEMM FFI 5894dcd, blocked Gram
b436e47/92aad84).  Queue empty at kickoff; dev 2-job cap respected throughout.

### R7.0 — harness + family authored (login, no jobs)

- b1024 family scripts adapted 1:1 from the gate-proven b512 recipe:
  qe_b1024.sbatch (staged density from out/MoS2.save — the ONE SCF),
  nscf_b1024.in (nbnd=1024), pw2bgw_b1024{,_rho}.in (vxc_diag_nmax=1024,
  WFN_b1024/kih_b1024/RHO_b1024), deck_b1024.sbatch (el_compare gate ->
  kin_ion -n 1024 --hartree -> dipole regen MANDATORY -> gate_h0 nb=1024),
  deck_b1024_c10000.sbatch (kmeans N=10000 --weight-bands 0:1024, isolated
  b1024_kmeans/ cwd, STANDARD 192 GB node on purpose = committed blocked-Gram
  acceptance test; nvdimm only as fallback), wk_REL/probes/el_compare_b1024.py
  (bands 1..512 vs WFN_b512.h5, tol 1e-5 eV).
- l5_b1024_c10000.sbatch = l4 template + rung-5 weapons: gw.in gains
  sigma_omega_layout = sharded (resolve gate needs nb_sigma%8==0 on both
  mesh axes: nval=26 + ncond=998 = 1024 ✓; ncond%8=6 keeps window pads
  live); env gains LORRAX_FFT_FFI=1, LORRAX_FFT_FFI_FUSED=1,
  LORRAX_BANDS_GEMM_FFI=1; HOSTSO -> build_host_BANDS (782992 B, has
  MklFftFlatKHostFfi + MklBlasGemmBatchHostFfi, superset of build_host_AUDIT
  — nm-preflight added).  Composition risk checked in src: all four flags
  are in ppm_tau_kernel's pipeline_key; sharded+streaming cannot co-occur at
  P=64; laplace merge is config-independent.  Banner strings actually say L5
  (the rung1-echo lineage bug is fixed, not inherited).
- Pre-submit planner check (coordinator threshold rule): fixed-nb law from
  rungs 3-4, HWM ~ 0.52 + 3.43e-3*mu GB/dev -> ~35 GB at mu~10k, plus
  persistent growth at 1024b (~4-5 GB; the ~11 GB/rank omega-cube residency
  is REMOVED by sigma_omega_layout=sharded) => expected ~40 GB vs budget 90
  (~45%) — CLEAR to submit; in-log planner block to be verified at babysit.

### R7.1 — CAMPAIGN FINDING: the band axis has a BASIS ceiling on this deck

- **Job 7879033** (qe_b1024, first attempt): NSCF died in seconds —
  `Error in routine memory_report (1): more bands than PWs!` (nscf_b1024.out;
  rc=1, downstream pw2bgw/wfn2hdf cascade-failed, no artifacts written).
- Root cause (arithmetic, not config noise): at 30 Ry the spinor basis is
  npwx*npol ~ 2x1950 ~ 3900; QE's Davidson workspace check needs
  nbnd*diago_david_ndim <= npwx*npol.  Default ndim=4: 1024*4 = 4096 > 3900
  FAIL (b512 passed at 2048 < 3900 without ever surfacing the constraint).
- FIX: nscf_b1024.in sets diago_david_ndim = 2 (2048 < 3900; diago_full_acc
  kept).  Resubmitted as **job 7879036**.
- FRONTIER NOTE (band axis, 30 Ry FIXED per brief): nbnd=1024 is close to
  the LAST clean band doubling on this deck — nbnd=2048 would need
  2048*2 = 4096 > 3900 (refused even at ndim=2; only cg-class solvers could
  go higher, into basis-exhaustion territory ~3900).  The (nband, N_mu)
  escalation beyond rung 5 must therefore ride the mu axis (and the brief's
  N_mu ~ 6-14x nband band keeps mu-room: 1024b supports mu up to ~14k).

> ⚠ CLAIM-DECAY (same session, minutes later): R7.1's first diagnosis
> ("default ndim=4; ndim=2 fixes it") was WRONG — job 7879036 (ndim=2
> explicit) died identically because QE 7.2's diago_david_ndim default IS
> already 2.  Source-verified arithmetic (PW/src/memory_report.f90:483,
> setup.f90:423-425): the gate is npwx_g < nbndx with npwx_g a G-sphere
> ESTIMATE ~ NINT(4pi/3*sqrt(30)^3/(2pi^3/V)) = ~1948 per SPINOR COMPONENT
> (npol never credited, though the true noncolin basis is ~3900), and
> nbndx = 2*nbnd (david, ndim=2 minimum) / nbnd (cg, isolve=1) /
> 2*nbnd (ppcg, rmm-davidson).  CORRECTED CEILINGS on this deck at 30 Ry:
> Davidson nbnd <= 974; CG nbnd <= 1948.  nband=1024 therefore REQUIRES
> diagonalization='cg' (job 7879040, submitted 21:19); nband=2048 is
> refused by EVERY solver (2048 > 1948) — the band axis of the ladder
> TERMINATES at 1024 with the cutoff fixed by the brief.  This is a
> certified wall (QE-side, arithmetic, not memory).

### R7.2 — the nbnd=1024 NSCF gauntlet (three failure modes, one patch)

Attempt ledger (all disk-verified from nscf_b1024.out / qe_b1024.*.out):
1. **7879033** david (QE default ndim=2): `memory_report: more bands than
   PWs!` — npwx_g estimate ~1948 < nbndx=2048.
2. **7879036** david ndim=2 explicit: identical (the edit was a no-op —
   2 IS the QE 7.2 default; first R7.1 diagnosis corrected above).
3. **7879040** cg (nbndx=nbnd=1024, passes the check): dies at k-point 1 of
   c_bands_nscf with `cdiaghg (1025): S matrix not positive definite`.
4. **7879041** cg + startingwfc='random': IDENTICAL death — kills the
   atomic-trial hypothesis; random 1024-trials in the ~3900-dim spinor
   space are far from dependent (MP aspect 0.26 -> Gram cond ~10).
Localization of failure 3/4 (source read, not guessed): the "Computing
kpt #: 1" banner proves wfcinit's OWN rotate_wfc at n_starting=1024
SUCCEEDED for every pool k; the dying call is the CG-ONLY pre-rotate in
diag_bands_nscf (c_bands.f90:768, evc aliased in/out) whose zhegv reports
S(1,1)<=0 — a CG-path defect at this size, not a physics/rank problem.
Davidson's cegterg path has NO such pre-rotate (its subspaces are built
iteratively with reorthogonalization) and is blocked ONLY by the
npwx_g-estimate check, which is provably npol-BLIND for noncolin runs
(true state dim = npwx_g*npol ~ 3896 >= nbndx = 2048).
- **ACTION (tool patch, announced): PW/src/memory_report.f90:483 patched
  to `npwx_g*npol < nbndx`** (comment block marks it LOCAL PATCH, LORRAX
  size campaign 2026-07-28); pw.x rebuilt 21:24 (make pw, mpiifort,
  bin/pw.x symlink -> PW/src/pw.x picks it up).  Zero numerical surface:
  memory_report computes NOTHING — the patch only corrects an estimate
  comparison that is wrong for spinors.  b256/b512 artifacts predate the
  rebuild and are untouched.
- **7879046** david ndim=2 on patched pw.x: submitted 21:25.  Fallback if
  cegterg itself misbehaves at 1024: retreat to a b960 family (nval=26,
  ncond=934, nband=960 — 960%8=0 for the sharded omega gate, 934%8=6 pads
  live, nbndx=1920 <= 1948 legal even UNPATCHED) — a documented 6%
  concession on the band axis.

### R7.3 — SUCCESSOR HANDOVER (agent #2, 2026-07-29 01:04)

Predecessor terminated 21:17 (quota) with job 7879046 in flight; it SUCCEEDED
after the death.  Everything below is disk-verified by the successor (sacct +
file mtimes + log text), not inherited.

**Hazard found and neutralised at handover: a MOVING SOURCE TREE.**  The
concurrent GEMM/BSE workstream had uncommitted edits in
`/work2/08271/jackmc/frontera/lorrax` at 00:39 — including
`src/common/contract_bands.py`, which is on the sigma `project_rs` hot path of
every ladder GW run.  A 32-node/2 h run must not import a tree another agent is
editing.  Fix: `git archive 4f77842` ->
`wk_REL/snapshots/srcpin_4f77842/` (verified: 228 py files, identical to HEAD; only the
workstream's uncommitted edits and `lorrax.egg-info` differ).  All rung-5
harnesses now set `PYTHONPATH` to the PIN, and l5's banner prints
`src@4f77842 (PINNED snapshot ...)` instead of `git rev-parse` on the live repo.
This is read-only w.r.t. the repo — nothing committed, nothing checked out.

**BANDS_GEMM dial**: the AUTO-default gate has NOT landed at 4f77842 (it is the
workstream's uncommitted WIP), so per the owner directive's fallback clause the
rung-5 run sets `LORRAX_BANDS_GEMM_FFI=1` EXPLICITLY.  Recorded here so the
rung-5 row is not misread later as evidence about the AUTO default.

#### R7.3a — b1024 NSCF harvest + provenance gate: PASS

Job 7879046 (david ndim=2 on the npol-patched pw.x), nscf_b1024.out:
- `number of Kohn-Sham states = 1024`, `ethr = 3.85E-13, avg # of iterations
  = 30.7`, `End of band structure calculation`, `JOB DONE`, PWSCF 6m29.79s WALL.
- highest occupied / lowest unoccupied = -5.3602 / -3.1481 eV.
- cost profile for the 45 Ry planning (below): cegterg 375.31 s of which
  **cdiaghg 346.58 s over 104 calls** — the dense subspace diagonalisation is
  92% of the NSCF, and it scales as nbndx^3, essentially independent of ecut.
- The one `Error reading attribute index` line is PRE-EXISTING (identical count
  in nscf_b512.out) — pseudopotential XML noise, not a failure.

**el_compare gate (job 7879287, wk_REL/probes/el_compare_b1024.py, tol 1e-5 eV):**
`max |d eigenvalue| over bands 1..512, all k = 1.859e-11 eV`, mean 7.276e-14,
worst (k=6, band=512).  **PASS.**  This is the load-bearing verification of the
`memory_report.f90:483` npol patch: the patched pw.x + cegterg reproduces, to
2e-11 eV, the spectrum the UNPATCHED pw.x produced at b512.  The patch's "zero
numerical surface" claim is now measured, not just argued.

#### R7.3b — b1024 deck completion (job 7879287, 4m01s, 1 node): ALL GREEN

- `kin_ion_b1024.h5` 536,879,104 B — rc=0, wall 188 s.
- `dipole_b1024.h5` 939,530,240 B — rc=0, wall 35 s.  MANDATORY regen done
  (the one artifact with no provenance guard; open-ledger rule honoured).
- density-symmetry on WFN_b1024.h5: TRS HOLDS (||m||/||rho|| = 1.27e-14),
  spatial 2/2 ops max resid 5.88e-14, integral rho = 26.000000 (rel 4.1e-16).
- **gate_h0 at nb=1024: PASS** — rms(H0 - kih) = 3.9e-5 eV, mean -1.7e-5,
  max|d| 2.39e-4; per-band rms 5.6e-6 .. 2.4e-4 (worst band 0);
  implied Vxc [-24.2546, -2.3610] eV vs QE Vxc [-24.2548, -2.3609],
  rms(impliedVxc - Vxc_QE) = 3.9e-5 eV.

#### R7.3c — c10000 centroid set = the blocked-Gram ACCEPTANCE TEST: **PASS on a
#### STANDARD 192 GB NODE, no nvdimm** (job 7879286, 1 node, 308 s)

Ladder wall #1 (R4.1/R6) is CLOSED at the committed code.  Evidence:
- node `free -g` total 186 GB (standard, c207-027 — NOT nvdimm).
- `[pivoted_cholesky] column-blocked Gram: M=13872, col_block=1480
  (10 blocks; single-device path)` — the committed path announced and took.
- auto block sized from `budget=168.211 GB/device` (host RAM auto-detect),
  25% envelope -> 1480 columns.
- `G built, shape=(13872, 13872), diag range [7.632e-17, 1.158e-04]`.
- orbit-aware: 897 orbits -> **10015 unfolded centroids (orbit-closed)**,
  rank=630, `mod 8 = 7` (window pads exercised).  ACCEPTED first try (N=10000,
  no nudge needed).
- **VmHWM = 66,219,996 kB = 66.2 GB** on a 186 GB node — 36% of the node.
  The OLD code would have needed 2 x 2048 x M^2 = 2 x 401 GB here.  This is the
  quantitative acceptance evidence for 92aad84/b436e47.

SCALING NOTE (corrects an assumption worth recording): centroid generation does
NOT scale with nband.  The Gram is built on the PRUNE WINDOW
`left=(0,26) right=(0,52)` (v x (v+c)), so the psi tensors are (nk, <=52, ns, M)
— sub-GB.  Only the k-means *weight* uses all 1024 bands.  The single-node cost
is therefore ~ (Gram M^2 x 16) + (pair blocks 2 x nk x ns^2 x M x B x 16), both
mu-only.  At the 25% auto envelope the pair blocks are budget-capped, so the
binder becomes the M^2 Gram itself: M~40k -> 26 GB, M~80k -> 102 GB.
**The centroid axis is NOT the terminal wall** — the mu axis is open on this
node class well past anything the GW side will survive.

#### R7.3d — rung-5 GW launch (job 7879295, submitted 01:13)

Pre-submit planner-budget check (coordinator threshold rule) — done from the
gflat model source, not just the predecessor's linear fit.  Reading
`gw/gflat_memory_model.py`:
    HWM = persistent + r_chunk * C_slope
    persistent = L_q(nq mu^2 /P) + gflat_acc(nq_disk mu ngkmax /P)
               + psi_copies(2 psi/p_x + 2 psi/p_y, psi = nk ns mu nb 16)
               + loader_tables (P-INDEPENDENT)
    C_slope(bytes per r-point) = slots*nk*ns^2*mu*16/p_xy   [pair carry]
                               + nq*mu*16/p_xy              [Z_q]
                               + 2*nk*band_chunk*ns*16/p_y  [gathered psi(r)]
                               + nk*(band_chunk/p_xy)*ns*16 [all-to-all source]
The pair-carry term dominates C_slope and is LINEAR in mu; with r_chunk pinned
at n_rtot = 46080 (24x24x80) it reproduces the measured ladder slope
2.5-3.1e-3 GB/mu to within the fit scatter.  Rung-5 prediction:
46080 x 64 x 10015 ~ 29.5 GB transient + ~3-4 GB persistent = **~33 GB/dev vs
90 GB budget (~37%)** — comfortably CLEAR, submitted.  (Predecessor's
independent linear extrapolation said ~40 GB; same verdict either way.)

## R8 — WHERE THE TERMINAL OOM IS: the planner's r-chunk PERFORMANCE FLOOR

Derived from `gw/gflat_memory_model.py` (read, not guessed) and calibrated on
the two most recent planner blocks (rung 4 measured, rung 5 in-log).

**Structure.**  `HWM = persistent + r_chunk * C_slope`, and the two knobs the
planner has are `band_chunk` and `r_chunk`.

    persistent = L_q (nq*mu^2 /P)                  ... /P
               + gflat_acc (nq_disk*mu*ngkmax /P)  ... /P
               + psi_copies (2*psi/p_x + 2*psi/p_y, psi = nk*ns*mu*nb*16)
                                                    ... only /sqrt(P) !
               + loader_tables                      ... P-INDEPENDENT
    C_slope [bytes per r-point]
               = slots*nk*ns^2*mu*16/p_xy    (pair-density carry — DOMINANT)
               + nq*mu*16/p_xy               (Z_q)
               + 2*nk*band_chunk*ns*16/p_y   (gathered psi(r))
               + nk*(band_chunk/p_xy)*ns*16  (all-to-all source)

**Calibration (both points at band_chunk=64, P=64, budget 90 GB, util 0.85):**

| rung | (nb, mu) | persistent GB | transient GB | r_chunk | C_slope B/r-pt |
|---|---|---|---|---|---|
| 4 | (512, 6947)  | 2.10 | 22.26 | 46080 | 483,073 |
| 5 | (1024, 10015)| 5.79 | 31.88 | 46080 | 691,840 |

Slope difference 208,767 B over Δmu = 3068 gives **68.05 B per unit mu**, and
the intercept 10,330 B matches the analytic psi-slab terms (8,704 B) to 16% —
i.e. `C_slope ≈ 68.05*mu + 10,330` B/r-point at band_chunk=64.  This is the
same 2.5–3.1e-3 GB/mu the ladder measured empirically, now with a mechanism.

**The escape that exists, and the one that does not.**  Stage C is chunkable in
r, so growth in mu is normally absorbed by shrinking `r_chunk` — HWM simply
pins to the 76.5 GB target and the run gets slower, not fatter.  BUT model
line 366 imposes a PERFORMANCE FLOOR:

        r_lo = min(mu, n_rtot)
        r_chunk = max(r_lo, min(n_rtot, headroom_C / C_slope))

Once `r_lo` exceeds the budget-derived chunk — i.e. once `68.05*mu^2 >
headroom_C` — the floor WINS and HWM stops tracking the budget.  From there
HWM grows **quadratically in mu** with no knob left in the automatic path.

**Predicted wall, 30 Ry / nb=1024** (persistent fitted as psi_copies =
256*mu*nb exactly, plus a mu^2 remainder calibrated at 3.16 GB @ mu=10015):

    r_lo binds near      mu ~ 25,000
    budget (90 GB) blown near mu ~ 29,000-30,000

The ladder's N list 15000/20000/25000/30000 (job 7879344) brackets this.
A hard ceiling also exists on this axis: centroids are grid points, so
mu <= n_rtot = 24*24*80 = 46080 at 30 Ry.  The predicted OOM sits below it —
**the wall is reachable on the existing b1024 deck.**

**The ONE named escape lever** (to be spent per campaign doctrine, once, when
the OOM arrives): `r_chunk_size` in gw.in (`gw_config.py:644`,
`r_chunk_override`), which overrides r_lo downward.  It trades wall time for
memory and it is the ONLY thing that moves this wall without changing P.  If
the run survives with it and dies again, the next object up is `persistent`,
whose leading term `psi_copies` is divided by **sqrt(P) only** (2 copies on
'x', 2 on 'y' — never /P), which is the true architectural floor at fixed
P=64 and the thing the thousands-of-low-memory-ranks target ultimately cares
about.

**45 Ry lineage** (n_rtot = 25*25*100 = 62500, nb=2048): same algebra, with
psi_copies doubled by nb and n_rtot x1.356.  r_lo binds near mu ~ 25,200 and
budget near mu ~ 27,400 — which is 12.3-13.4x nband, i.e. INSIDE the
campaign's N_mu ~ 6-14x band.  The 45 Ry ladder therefore hits the same wall
at a physically legitimate ratio rather than a degenerate one.

## R9 — the 45 Ry DECK LINEAGE (owner directive 2026-07-29)

Owner: *"if you can't do 1024 bands then do like a 45 Ry calculation."*  1024
bands DID succeed at 30 Ry (R7.3a), so the ceiling binds BEYOND 1024 and the
directive's real content is: raise the plane-wave cutoff to lift the band
ceiling AND increase genuine per-rank work.  This is a **distinct deck
lineage**, not a rung of the 30 Ry ladder: every path carries an `_r45`/`r45_`
tag and NO 30 Ry artifact is touched or reused.

### R9.0 — provenance chain (FULL regen; the staged-density trick is only
### valid at FIXED cutoff, so the SCF itself is recomputed)

    scf_r45.in  (ecutwfc 45 / ecutrho 180)  -> out_r45/MoS2.save   [the ONE 45 Ry SCF]
      |-- stage density --> r45_b1024_{a,b,c}_out/  (NSCF 1024, the probe)
      |                       winner promoted: r45_b1024_out -> WFN_r45_b1024.h5
      |                       + vxc_r45_b1024.dat / kih_r45_b1024.dat / RHO_r45_b1024
      `-- stage density --> r45_b2048_out/          (NSCF 2048, the escalation rung)
                              -> WFN_r45_b2048.h5 + vxc/kih/RHO_r45_b2048
    deck_r45_b2048.in  -> kin_ion_r45_b2048.h5, dipole_r45_b2048.h5 (MANDATORY)
    centroids_r45_b2048_c<N>.txt   (NEW 25x25x100 grid — 30 Ry sets are INVALID)

Gates are INTERNAL (owner ruling): SCF/NSCF convergence, el_compare 2048-vs-1024
WITHIN the 45 Ry family, gate_h0/implied-Vxc at the new cutoff, W-Dyson
residual, Kramers/degeneracy, collective-table doctrine.  **NO cross-cutoff
eqp parity against 30 Ry.**  The 30-vs-45 Ry QP gap difference is recorded as a
cutoff-convergence PHYSICS OBSERVATION, explicitly labelled as such.

### R9.1 — 45 Ry basis facts (job 7879300, MEASURED)

    SCF converged in 11 iterations, 12 s WALL (nbnd=40).  Total E = -178.22961446 Ry
    Dense grid  28615 G-vectors    (30 Ry: 15631; ratio 1.831 vs (45/30)^1.5 = 1.837)
    FFT dimensions (25, 25, 100) = 62500 points   (30 Ry: 24x24x80 = 46080; x1.356)
    2 Sym. Ops., 10 k-points in the IBZ

**n_rtot x1.356 is the GW-side content of the cutoff raise**: Stage C's
transient is `r_chunk * C_slope` and r_chunk is capped by n_rtot, so the same
mu costs 1.36x more memory and work per rank at 45 Ry.

### R9.2 — the deck has TEN k-points, not sixteen (corrects a planning premise)

Disk-verified from `WFN_b1024.h5 /mf_header/kpoints/ngk`: **10 entries**
[1947 1947 1964 1947 1933 1933 1947 1964 1933 1964], `ngkmax = 1964`,
`/mf_header/gspace/ng = 15631`.  The 4x4 grid has 16 points in the full BZ
(nk_tot=16, which is what the GW/centroid code uses) but only 10 in the IBZ,
which is what QE pools over.  So "one k-point per pool" is **-nk 10**, and the
coordinator's suggested `-nk 16` is not achievable.  112 ranks is not divisible
by 10, hence `-n 110` (11 ranks/pool) for the one-k-per-pool configs.

### R9.3 — BAND CEILING at 45 Ry: the npol patch is STILL load-bearing

`memory_report.f90:91`: `npwx_g = NINT( 4pi/3 * sqrt(ecutwfc)^3 / (2pi^3/omega)
/ g_fact )` — an ESTIMATE exactly proportional to ecutwfc^1.5.  30 Ry gives
~1948 (true ngkmax 1964, so the estimate is good to 1%).  45 Ry gives
1948 x 1.83712 = **~3578** per spinor component.

| gate | form | ceiling at 30 Ry | ceiling at 45 Ry |
|---|---|---|---|
| STOCK   | `npwx_g < nbndx`        | nbnd <= 974  | nbnd <= 1789 |
| PATCHED | `npwx_g*npol < nbndx`   | nbnd <= 1948 | nbnd <= 3578 |

(david, ndim=2 => nbndx = 2*nbnd.)  **nbnd=2048 at 45 Ry needs nbndx=4096 >
3578, so a stock pw.x still REFUSES it** — the coordinator's hypothesis that
the new lineage might not need the local patch is FALSE at 2048 bands.  It
would hold only for nbnd <= 1789.  The patch's numerical innocence is now
MEASURED, not argued (R7.3a: 1.86e-11 eV el_compare across the rebuild).

### R9.4 — PARALLELIZATION PROBE (owner: measure, do not assume)

QE build is `-D__DFTI -D__MPI` — **no `-D__SCALAPACK`**, so `-ndiag>1` exercises
LAXlib's own MPI ortho ("custom distributed-memory algorithm").  All configs
below run the SAME physical NSCF (nbnd=1024 @ 45 Ry) from the same staged
density, on `small`.

| probe | layout | ranks/pool | wall | node-h | cdiaghg | cegterg |
|---|---|---|---|---|---|---|
| a | 2 nodes, -n 110 -nk 10 -ndiag 1 | 11 | 234 s | 0.130 | 149.82 s | 172.46 s |
| b | 2 nodes, -n 110 -nk 10 -ndiag 9 | 11 | 252 s | 0.140 | 202.13 s | 225.81 s |
| c | 2 nodes, -n 112 -nk  4 -ndiag 16 | 28 | 350 s | 0.194 | | |
| d | 1 node,  -n  50 -nk 10 -ndiag 1 |  5 | **228 s** | **0.063** | | |
| e | 1 node,  -n  56 -nk  8 -ndiag 1 |  7 | 421 s | 0.117 | | |

Probe e is the sharpest confirmation of the one-k-per-pool rule: `-nk 8` on a
10-k IBZ leaves two pools carrying TWO k-points each, so the wall is set by
those pools and nearly doubles (421 s vs 228 s, 1.85x) even though the node
count and rank count are the same or larger.  **The pool count must DIVIDE the
k-point count, and equal it if possible.**

**MEASURED VERDICT — optimal is `1 node, -n 50 -nk 10 -ndiag 1` (probe d):
228 s at 0.063 node-hours, i.e. the same wall time as two nodes for HALF the
cost, and 1.53x faster than the b1024-style `-nk 4` layout.**  Three findings,
each a measurement rather than folklore:

1. **One k-point per pool wins.**  -nk 10 (the IBZ count) beats -nk 4 by 50%
   in wall and 3.1x in node-hours (234 s / 0.130 vs 350 s / 0.194).  This is
   the owner's instinct, confirmed.
2. **`-ndiag > 1` LOSES here.**  cdiaghg 202.13 s distributed (3x3) vs
   149.82 s serial — 35% worse on the diagonalization itself, 8% worse
   overall.  Cause: this QE is built `-D__DFTI -D__MPI` with **no
   `-D__SCALAPACK`**, so -ndiag>1 falls to LAXlib's own MPI ortho
   ("custom distributed-memory algorithm"), which is communication-bound at
   nbndx=2048 on 9 ranks.  Do NOT assume ScaLAPACK-era guidance here.
3. **Extra ranks/pool are nearly free of value, so use FEWER NODES.**  With
   -ndiag 1 the subspace diagonalization is serial on ONE rank per pool and is
   64-87% of the NSCF wall (149.82/172.46 at 45 Ry; 346.58/375.31 at 30 Ry).
   The other ranks idle through it; they buy only h_psi+FFT (~20 s).  Halving
   the node count therefore costs ~nothing in wall (234 -> 228 s, within
   noise) and halves the bill.  The node-hour lever is fewer nodes, not more
   ranks — the opposite of the usual reflex.

### R9.4a — measured 45 Ry basis, and the ceiling arithmetic CONFIRMED

`WFN_r45_b1024.h5` (job 7879300, 1,176,032,244 B) reports, authoritatively:

    /mf_header/kpoints/ngk    = [3577 3597 3586 3597 3578 3578 3597 3586 3578 3586]
    /mf_header/kpoints/ngkmax = 3597      (30 Ry: 1964)
    /mf_header/gspace/ng      = 28615     (30 Ry: 15631)

ngkmax ratio 3597/1964 = 1.831 vs the analytic (45/30)^1.5 = 1.837 — the
ecut^1.5 scaling used for the ceiling holds to 0.3%.  The memory_report
ESTIMATE npwx_g = 1948 x 1.83712 = 3578 sits 0.5% below the true 3597, the
same sign and size of error as at 30 Ry (1948 vs 1964, 0.8% low).  So the R9.3
ceilings stand as computed: **stock pw.x refuses nbnd=2048 at 45 Ry
(nbndx=4096 > 3578); the local npol patch is still load-bearing.**

### R9.5 — memory fit, stated explicitly (coordinator's requirement)

45 Ry: npwx ~ 3608/spinor component => noncolin state length npwx*npol ~ 7216.
`small` 2 nodes = 112 cores / 384 GB; 10 pools over 2 nodes => 38.4 GB/pool.

| nbnd | nbndx | evc | psi+hpsi | subspace hc/sc/vc | total/pool | of 38.4 GB |
|---|---|---|---|---|---|---|
| 1024 | 2048 | 118 MB | 473 MB | 201 MB | ~0.8 GB | 2% |
| 2048 | 4096 | 237 MB | 946 MB | 805 MB | ~2.0 GB | 5% |
| 3578 (ceiling) | 7156 | 413 MB | 1.65 GB | 2.46 GB | ~4.5 GB | 12% |

`small` is ample at every band count this lineage can legally reach; `normal`
is NOT needed for QE.  (NC pseudos => no spsi term.)

### R9.6 — cost verdict for the expensive stage

cdiaghg is O(nbndx^3) and ~independent of ecut, so nbnd 1024 -> 2048 is ~8x.
From probe a (149.82 s at nbndx=2048): NSCF at nbnd=2048/45 Ry ~ 20-25 min,
plus h_psi/FFT and pw2bgw/wfn2hdf on a ~2.4 GB WFN => **well under an hour**.
With `small`'s 48 h wall there is no staging problem and no timeout risk;
the earlier "may exceed one dev job" concern is retired.

### R9.7 — DEATH #1 (45 Ry, nbnd=2048): a SECOND npol-blind QE guard

**Job 7879353's predecessor, job 7879348** — NSCF at 45 Ry / nbnd=2048 died in
18 s, rc=1:

    Error in routine diag_bands (1): too many bands, or too few plane waves
    Abort(1) on node 5 ... MPI_Abort(MPI_COMM_WORLD, 1)

CLASSIFICATION: **QE/tool FAIL-FAST** (not OOM-killer, not XLA allocator, not
a LORRAX refusal).  It cleared the patched `memory_report` gate — the log shows
`number of Kohn-Sham states = 2048` and the grid banner — then aborted inside
`diag_bands` at the first k-point.

ROOT CAUSE (source-read, `PW/src/c_bands.f90:306-309`):

    ipw = npwx                       ! LOCAL max-over-k plane-wave count
    CALL mp_sum(ipw, intra_bgrp_comm)  ! -> ~the GLOBAL spatial PW count
    IF ( nbndx > ipw ) CALL errore('diag_bands','too many bands...',1)

This is **the same npol-blindness already corrected at memory_report.f90:486**,
in a second place: it compares `nbndx = david*nbnd = 4096` against the SPATIAL
plane-wave count (~3600 here) instead of the spinor trial-vector dimension
`ipw*npol` (~7200).  Note the guard is also mildly PARALLELIZATION-DEPENDENT —
`ipw` is a sum of per-rank max-over-k values, so it grows with ranks/pool
through padding/imbalance; that is why the 30 Ry nbnd=1024 run (14 ranks/pool,
nbndx=2048 vs global npwx 1964) squeaked past it while this one (5 ranks/pool)
did not.  Nobody should rely on that margin.

ESCAPE ATTEMPTED (the ONE lever for this death): extend the SAME, already
numerically-validated correction to this guard.

    IF ( nbndx > ipw*npol ) CALL errore(...)

Justification, in order of strength:
1. **Empirical**: the spatial comparison is already demonstrably the wrong
   invariant.  The 30 Ry nbnd=1024 run operated at `nbndx = 2048 > npwx_global
   = 1964` and its spectrum matched the independent b512 calculation to
   **1.86e-11 eV** (job 7879287).  Running "past" the spatial PW count is
   already known-good for spinors.
2. **Zero numerical surface**: `ipw` appears at exactly three lines in the
   whole file (declaration 255, and 306-308).  Nothing is computed from it;
   it exists solely for this guard.
3. **Consistency**: identical correction, identical reasoning, to the
   2026-07-28 memory_report patch.
Backup kept: `PW/src/c_bands.f90.orig_prepatch_20260729`.  pw.x rebuilt
01:40 (make pw, rc=0).  Resubmitted as **job 7879353**.
GATE FOR THE PATCH: `el_compare_r45_b2048.py` — bands 1..1024 of the new
WFN must reproduce the r45_b1024 family (built on the UNPATCHED-for-this-guard
binary) to 1e-5 eV.  Same validation shape that certified the first patch.
FALLBACK if the gate fails or cegterg misbehaves: nbnd=1536 at 45 Ry
(nbndx=3072 < 3578) needs NEITHER patch — a 50% band-axis gain over the 30 Ry
ceiling instead of 100%.

TOOL-STATE LEDGER (both patches are LOCAL, announced, and outside the LORRAX
repo — nothing was committed to lorrax):
| file | line | change | rebuilt |
|---|---|---|---|
| PW/src/memory_report.f90 | 486 | `npwx_g*npol < nbndx` | 2026-07-28 21:24 |
| PW/src/c_bands.f90 | 308 | `nbndx > ipw*npol` | 2026-07-29 01:40 |
All 30 Ry artifacts (b256/b512/b1024) predate the 01:40 rebuild and are
untouched; the 45 Ry b1024 family was built at 01:36, i.e. BEFORE it, which is
precisely what makes it a valid independent reference for the patch gate.

## R7.4 — RUNG 5 RESULT (nband=1024, N_mu=10015): **MEMORY GREEN, PHYSICS RED**

**Job 7879295**: `[gw rc=0 wall=1770s]`, no OOM, no tracebacks, W-Dyson residual
max 1.887e-14, mlx provider, path=distributed_rank_truncate, all four weapons
confirmed live in-log (`[fft_ffi] flat-k 3-D FFTs -> MKL FFT (DFTI API) host
FFI handler`, `sigma_omega_layout = sharded: ... no full-cube replication`).

### Memory — the campaign's primary instrument, and it is GREEN

    predicted HWM 37.67 GB/dev (42% of budget) [binder C_fit_one_rchunk]
    MEASURED  MAX VmHWM = 36.03 GB   ->  ratio 0.956
    persistent 5.79 GB, r_chunk 46080 (1 chunk), band_chunk 64, q_chunk 16
    restart V_qmunu (16, 10015, 10015) = 25.68 GB  (= nq*mu^2*16 exactly)
    zeta_fit 358.3 s of which cholesky 240.6 s (mu^3: 11.9/24.2/36.9/91.5/240.6
    across rungs 1-5)

The gflat model now holds to -4.4%..+6.3% across FIVE rungs, 4.3x HWM range,
nb 256->1024 and mu 2475->10015.

### Physics — NOT green.  The QP gap is wrong by ~3 eV and eqp1 is NEGATIVE

Extraction convention VERIFIED by reproducing rung 4 exactly from its own
eqp files (indirect gap = min_k eqp(band 27) - max_k eqp(band 26), column 4;
gives 3.2194 / 2.9778, matching R4.3 to 4 decimals):

| rung | (nb, mu) | eqp0 gap | eqp1 gap |
|---|---|---|---|
| 4 | (512, 6947)  |  3.2194 |  2.9778 |
| 5 | (1024,10015) | **0.3645** | **-0.3639** |

Per-k evidence (eqp0, k1): DFT VBM -5.3602 eV -> QP **-13.4801** eV, i.e. a
**-8.12 eV** self-energy correction where rung 4 had -0.77 eV.  A negative
eqp1 fundamental gap is unphysical.  This is not a convergence wobble; it is
a correctness failure.

### Why this is confounded, and the experiment that separates it

Rung 5 changed FOUR things at once relative to the last known-good rung:

| | rung 4 (green) | rung 5 (red) |
|---|---|---|
| size | (512, 6947) | (1024, 10015) |
| src | pre-weapons commit | 4f77842 (pinned) |
| host FFI | build_host_AUDIT | build_host_BANDS |
| FFT FFI | OFF | LORRAX_FFT_FFI=1 + _FUSED=1 |
| bands GEMM FFI | OFF | =1 |
| omega cube | replicated (default) | **sigma_omega_layout = sharded** |
| Laplace merge | OFF (pre-default) | **ON (default at 5894dcd)** |

Disk-verified: `run_L4_b512_c7000/inner.sh` exports ONLY
`LORRAX_FFI_HOST_SO=.../build_host_AUDIT/...` — rung 4 ran with NO weapons and
no `sigma_omega_layout` line in its gw.in.  So the entire rungs 1-4 ladder is
a NO-WEAPONS baseline, and rung 5 is the first weaponised run at any size.
The per-weapon parity gates cited in the commits (712a866, 5918cf6, 0225b5f,
5894dcd) were taken at nband=128/256 on the AQ deck, NOT at 512/1024.

Two 32-node runs submitted to separate size from weapons (the 2x2):
- **job 7879357** — (1024, 10015) with **WEAPONS=off** (build_host_AUDIT, no
  FFT/GEMM FFI, replicated omega cube, `LORRAX_SIGMA_LAPLACE_MERGE=0`).
  If the gap returns to ~3 eV, the WEAPONS are guilty.
- **job 7879359** — (512, 6947), i.e. rung 4's EXACT size and artifacts, with
  **WEAPONS=on**.  If the gap leaves 3.2194/2.9778, a weapon is guilty and the
  b1024 deck is exonerated.
Suspicion ordering (to be tested, not assumed): `sigma_omega_layout=sharded`
first — it is the only weapon that changes how Sigma(omega) is CONSUMED by the
QP solve, which is exactly downstream of every gate that passed (H0, W-Dyson
residual, Kramers) and upstream of eqp.  `ncond = 998` leaves `998 % 8 = 6`, so
the sharded consumer's window pads are live.

**Ladder consequence**: the rung-5 MEMORY row is trustworthy and is retained
(the planner-vs-VmHWM instrument does not depend on the QP values).  The
rung-5 PHYSICS row is withdrawn pending the 2x2.  Rung 6 (job 7879349, mu
15007) is running with weapons ON and will inherit the same defect; its memory
row is still wanted, its gap is not.

### R7.5 — localising the rung-5 defect BEFORE the 2x2 lands: it is in Sigma_c

Cheap discriminator read straight from the two gw.logs — the bare exchange
self-energy, which uses the SAME ISDF centroid basis and the SAME WFN as the
correlation part but none of the omega machinery:

    rung 4 (512, 6947, no weapons)  Bare Σ_X (eV), k=0:
        -40.5483 -40.5483 -33.2791 -33.2791 -32.6922 -32.6922 -32.6761 -32.6761
    rung 5 (1024, 10015, weapons)   Bare Σ_X (eV), k=0:
        -40.5358 -40.5358 -33.2733 -33.2733 -32.6882 -32.6882 -32.6668 -32.6668

Agreement is 12.5 meV on -40.5 eV (0.03%) — exactly the size one expects from
a different mu and a different centroid set, and FAR from the ~8 eV error in
the QP energies.  Consequences:

- The **b1024 WFN is healthy** (Σ_X is built from it).
- The **c10015 centroid set is healthy** (Σ_X is built in that ISDF basis) —
  the blocked-Gram output is exonerated as a source of the QP defect.
- `gate_h0` already exonerated kin_ion/kih/vxc (rms 3.9e-5 eV).
- Therefore the defect lives in the **omega-dependent correlation self-energy
  Sigma_c** — precisely the region the four rung-5 weapons touch
  (`sigma_omega_layout=sharded` consumer, the Laplace channel merge, the FFT
  FFI + fused G*W multiply), plus the q->0 head correction that reads
  `dipole_b1024.h5`.
- Remaining suspects, in order: (1) sharded omega-cube consumption,
  (2) Laplace merge default, (3) dipole/head correction, (4) FFT FFI.
  Jobs 7879357/7879359/7879361 discriminate (1),(2),(4) from (3) and from size.

### R7.6 — dipole / head correction ALSO exonerated; defect isolated to the Σ_c body

The q->0 head is the only place `dipole_b1024.h5` enters Σ, and it is the one
deck artifact with no provenance guard — so it was suspect (3).  It is clean:

| quantity | rung 3 (512,4951) | rung 4 (512,6947) | rung 5 (1024,10015) |
|---|---|---|---|
| v(q->0) a.u.        | 2443.340 | 2443.340 | 2443.340 |
| W^c(q->0, w=0)      | -1784.025 | -1784.025 | -1784.035 |
| Omega_h (eV)        | 13.980207 | 13.980207 | 13.983724 |
| R_h Ry·a.u.         | 916.565830 | 916.565830 | **916.801714** |
| max abs Σ^head_diag | 2.9691 eV | 2.9691 eV | **2.9678 eV** |
| on-shell occ band   | -1.0802 eV | -1.0802 eV | **-1.0802 eV** |

R_h agrees to 0.026% and the on-shell occupied-band head shift is IDENTICAL to
four decimals.  `dipole_b1024.h5` is therefore correct, and so is the head
machinery under the sharded layout's rank-local injection.

**Elimination table for the rung-5 defect** (every row disk-verified):

| component | evidence | verdict |
|---|---|---|
| WFN_b1024.h5 | el_compare vs b512 = 1.86e-11 eV | CLEAN |
| kin_ion / kih / vxc | gate_h0 rms 3.9e-5 eV, implied-Vxc 3.9e-5 eV | CLEAN |
| centroids c10015 | bare Σ_X matches rung 4 to 0.03% (same ISDF basis) | CLEAN |
| dipole_b1024.h5 / head | R_h 0.026%, on-shell head shift identical | CLEAN |
| W (screening) | Dyson residual max 1.887e-14 | CLEAN |
| Σ_X (exchange) | -40.5358 vs -40.5483 eV | CLEAN |
| **Σ_c (correlation body)** | QP error ~ -8 eV at the VBM | **DEFECTIVE** |

So the failure is confined to the omega-dependent correlation self-energy —
i.e. to some subset of {sharded omega-cube consumption, Laplace channel merge,
FFT FFI + fused G*W}.  Nothing about the deck, the basis, or the screening is
implicated, which is why every provenance and physics gate the campaign runs
still passed while the answer was wrong.  **Lesson for the gate suite: the
ladder has no gate that would have caught this** — the eqp values themselves
are the only witness, and no rung compares them against an independent
baseline.  A cheap fix is to carry a fixed (nb, mu) reference point and assert
its gap on every rung.

### R8.1 — MODEL GAP FOUND: the gflat planner does NOT model the Σ omega cube

Job 7879357 (rung-5 size, WEAPONS=off) prints the IDENTICAL planner block to
job 7879295 (weapons ON): `HWM estimate = 37.67 GB/dev, persistent 5.79`.
But the two runs cannot have the same peak — `sigma_omega_layout=sharded` is
precisely what REMOVES the replicated `(nω, nk, nb, nb)` omega cube
(2751.46 MB/rank at nb=512, ~11 GB/rank projected at nb=1024, ladder wall #3).

`gflat_memory_model.py` covers only the ISDF chunk plan (stages
A_centroid_load / B_cct_chol / C_fit_one_rchunk / D_accumulate / E_v_q /
F_tensor_write).  The Σ stage's omega cube is outside it entirely.  Two
consequences the campaign must carry:

1. **The 0.956-1.063 predicted/measured agreement across rungs 1-5 is a
   statement about the ISDF stage only** — it held because C_fit_one_rchunk
   was the global binder in every one of those runs (all had nb <= 1024 and,
   for rungs 1-4, an omega cube of at most 2.75 GB, far below C_fit).
2. **At larger nb the omega cube can overtake C_fit and the planner will not
   see it coming.**  Replicated cube ~ nω·nk·nb²·16 B / rank: at nb=2048
   (the 45 Ry rung) that is ~4x the nb=1024 figure, i.e. ~44 GB/rank — which
   would BIND before C_fit does.  With `sharded` it is ~zero.
   So on the 45 Ry lineage the sharded layout is not a nicety, it is load
   bearing — which makes the rung-5 physics regression in exactly that code
   path the campaign's critical-path problem, not a side quest.

Prediction to be checked against job 7879357's measured VmHWM when its Σ stage
runs: weapons-OFF at (1024, 10015) should peak ~11 GB ABOVE the weapons-ON
36.03 GB, i.e. ~47 GB, while the planner still says 37.67.

### R7.7 — BISECT RESULT 1: the weapons are INNOCENT at rung-4 size

**Job 7879359** — (512, 6947), rung 4's EXACT deck and centroid set, with ALL
FOUR rung-5 weapons ON (sharded omega layout, FFT_FFI+FUSED, BANDS_GEMM_FFI,
merged Laplace default, build_host_BANDS), at the pinned commit 4f77842:

    eqp0  VBM= -6.1259  CBM= -2.9065  QP gap = 3.2194 eV
    eqp1  VBM= -5.9453  CBM= -2.9675  QP gap = 2.9778 eV
    VmHWM 23.60 GB  (rung 4 measured: 23.60 GB)

**Identical to four decimals to the no-weapons rung-4 baseline, and the peak
memory matches to the reported precision.**  So the weapon bundle does NOT
break the physics per se — the per-weapon parity gates in 712a866 / 5918cf6 /
0225b5f / 5894dcd hold up here at 4x the band window and 1.4x the mu of the
AQ deck they were originally taken on.

This REFUTES the leading hypothesis of R7.4/R7.6 and re-opens the question.
The remaining possibilities are:
  (a) a **size** effect at nb=1024 / mu=10015 that is independent of weapons;
  (b) a **weapon x size INTERACTION** — the weapons are fine at nb=512 and
      break at nb=1024.
Job 7879357 — (1024, 10015) with WEAPONS=off — discriminates them:
GREEN => (b) interaction;  RED => (a) size, weapons fully exonerated.
Note (a) cannot be a deck-artifact problem: R7.6's elimination table already
cleared the WFN, the centroid set, kin_ion/kih/vxc and the dipole/head using
Σ_X and the head fit, which are computed from exactly those objects.
Structural suspicion for (a)/(b): nb_sigma = 1024 = 2^10 with band_chunk = 64
and an 8x8 mesh (mb = nbl = 128), vs 512 -> mb = nbl = 64 = band_chunk.

### R9.8 — 45 Ry nbnd=2048 NSCF: SUCCESS on the patched binary (job 7879353)

The c_bands.f90 npol patch (R9.7) is validated operationally:

    [nscf rc=0 wall=995s node-hours=0.2764]     (1 node, -n 50 -nk 10 -ndiag 1)
    number of Kohn-Sham states = 2048
    ethr = 3.85E-13,  avg # of iterations = 22.4
    highest occupied, lowest unoccupied level (ev):  -5.3575  -3.1471
    PWSCF : 16m18.76s CPU  16m29.28s WALL

- The predicted cost (R9.6: "~20-25 min from the nbndx^3 scaling") came in at
  16.5 min — the estimate was conservative by ~30%.  0.28 node-hours for a
  2048-band noncollinear NSCF at 45 Ry is cheap; `small` handles it trivially.
- **CUTOFF-CONVERGENCE PHYSICS OBSERVATION** (labelled as such per the owner's
  ruling — this is NOT a gate): the DFT fundamental gap is
      30 Ry, nbnd=1024:  -5.3602 -> -3.1481  =  2.2121 eV
      45 Ry, nbnd=2048:  -5.3575 -> -3.1471  =  2.2104 eV
  i.e. **1.7 meV** of change for a 50% cutoff raise (1.83x the plane waves).
  The 30 Ry deck is essentially converged at the DFT level for this quantity;
  the cutoff raise buys BAND-AXIS HEADROOM and per-rank work (n_rtot x1.356),
  not a different answer.  The QP-level comparison must wait for the GW run
  and will likewise be reported as an observation, never as a parity gate.
- Still pending for the family: el_compare 2048-vs-1024 (the patch's numerical
  gate), pw2bgw/wfn2hdf, kin_ion/dipole/gate_h0, centroids on the 62500 grid.

### R7.8 — BISECT RESULT 2: weapons fully exonerated at nb=512; +1.84x for free

**Job 7879361** — (512, 6947), all weapons ON **except**
`sigma_omega_layout=sharded` (the prime suspect, dialled out on its own):

    eqp0 3.2194 eV   eqp1 2.9778 eV

**Identical to job 7879359 (all weapons ON) and to rung 4 (no weapons).**
Three configurations, three identical answers at rung-4 size:

| config | job | eqp0 | eqp1 | wall | VmHWM |
|---|---|---|---|---|---|
| no weapons (rung 4) | 7878363 | 3.2194 | 2.9778 | 1217 s | 23.60 GB |
| ALL weapons | 7879359 | 3.2194 | 2.9778 | **661 s** | 23.60 GB |
| all weapons minus sharded | 7879361 | 3.2194 | 2.9778 | — | — |

**Bonus result worth its own row: the weapon bundle is a measured 1.84x
end-to-end speedup (1217 -> 661 s) at BIT-IDENTICAL physics and unchanged peak
memory, at 512 bands / mu=6947** — 4x the band window and 1.4x the mu of the
AQ deck the individual weapons were originally gated on.  That is a real,
independently reproduced validation of 712a866 / 5918cf6 / 0225b5f / 5894dcd
at ladder scale, and it is the strongest evidence the campaign has that those
commits are sound.

CONSEQUENCE FOR THE RUNG-5 DEFECT: it is NOT "the weapons are broken".  It is
either a pure SIZE effect at nb=1024 / mu=10015, or a WEAPON x SIZE
interaction that only switches on above nb=512.  Job 7879357 (weapons OFF at
the rung-5 size) is the single remaining discriminator.  Because R7.6 already
cleared every deck artifact via Σ_X and the head fit, a RED there would point
at the Σ_c band-window machinery itself at nb_sigma=1024 rather than at any
input.

### R9.9 — the c_bands.f90 npol patch is NUMERICALLY VALIDATED (job 7879367)

`el_compare_r45_b2048.py`, bands 1..1024 of `WFN_r45_b2048.h5` (built on the
PATCHED pw.x, 01:57) against `WFN_r45_b1024.h5` (built at 01:36, i.e. BEFORE
the 01:40 rebuild — a genuinely independent reference):

    max |d eigenvalue| over bands 1..1024, all k: 1.405e-10 eV
    mean |d|: 1.425e-13 eV      worst (k=4, band=1023)
    ngk identical between the two WFNs (same basis, checked in the gate)
    PASS (tol 1e-5 eV)

Both LOCAL QE patches are now measured, not argued:

| patch | gate | result |
|---|---|---|
| memory_report.f90:486 `npwx_g*npol < nbndx` | 30 Ry b1024 vs b512 | 1.859e-11 eV |
| c_bands.f90:308 `nbndx > ipw*npol` | 45 Ry b2048 vs b1024 | 1.405e-10 eV |

Both correct the same npol-blindness; both are pure guards that compute
nothing; both reproduce an independently-built spectrum to ~1e-10 eV.  The
escape spent on DEATH #1 is therefore certified, and the 45 Ry band axis is
open to nbnd <= 3578 (Davidson) instead of 1789.

**RHO provenance inside the 45 Ry family also holds**: `RHO_r45_b2048` vs
`RHO_r45_b1024` differ in exactly **4 bytes**, at offsets 72/73/75/76, all
ASCII digits in the BerkeleyGW header timestamp (`cmp -l` values 63/65/63/66
vs 65/67/64/60).  The density payload is byte-identical — the b512 precedent
was 7 such bytes.

### R8.2 — the wall model is now PREDICTIVE to <1% (rung 7 pre-registered)

The rung-7 planner numbers were computed and WRITTEN DOWN BEFORE the job ran
(submission turn, job 7879369), then compared to the in-log block:

| quantity | predicted | actual | error |
|---|---|---|---|
| persistent | 15.96 GB | 15.86 GB | 0.6% |
| C_slope | 1,707,021 B/r-pt | — | — |
| r_chunk | 35,467 | 35,520 | 0.15% |
| n_r_chunks | 2 | 2 | exact |
| **HWM** | **76.50 GB** | **76.45 GB** | **0.07%** |
| r_lo binding? | No | No (35520 > 24933) | exact |

This is a pre-registered prediction, not a fit — the constants
(`C_slope = 68.05*mu + 10,330`, `persistent = 540,244*mu + 4*mu^2` at nb=1024)
were fixed from rungs 4/5/6 and applied unchanged.  **The extrapolated wall is
therefore load-bearing**, not a hand-wave.

Rung 7 also sits exactly where the model says the regime changes: HWM has
stopped rising with mu and PINNED to the 76.5 GB target (85% of budget) by
splitting r into 2 chunks.  That is the planner absorbing growth, and it will
keep working until `r_lo = min(mu, n_rtot)` overtakes the budget-derived chunk.

**A SECOND instrument appeared: `P_min (floor)` is climbing.**
    rung 5 (mu 10015): P_min = 1
    rung 6 (mu 15007): P_min = 3
    rung 7 (mu 24933): P_min = 6
`P_min` is the smallest rank count whose un-chunkable `persistent` still fits
the target — i.e. a direct readout of the architectural floor the campaign
cares about.  It is rising superlinearly in mu (persistent carries a mu^2 term
÷P and a mu·nb term ÷sqrt(P)).  If a rung ever reports `P_min > 64` the run
will FAIL FAST at resolve time rather than OOM — a distinct and equally
certifiable wall, and the one that speaks directly to the
thousands-of-low-memory-ranks target.

### Pre-registered predictions for the remaining rungs (30 Ry, nb=1024)

| rung | mu | r_lo binds? | predicted HWM | expectation |
|---|---|---|---|---|
| 8 | ~29,900 (c30000) | **YES** (30000 > 27634) | **81.4 GB** (90% budget) | survives, regime change visible |
| 9 | ~31,900 (c32000) | YES | **90.9 GB** | at/over budget — OOM likely |
| 10 | ~32,900 (c33000) | YES | **96.1 GB** | **OOM** (96 GB/rank physical) |

Actual/predicted has run 0.945-0.956 on the last three rungs, so the true peak
lands ~5% below these; the physical ceiling is 96 GB/rank (192 GB node / 2
ranks) less OS and page cache.  Rung 9-10 is where the terminal OOM should
arrive.  Hard ceiling on the axis: mu <= n_rtot = 46080.

## R10 — **DEATH #2: a FAIL-FAST WALL AT mu = 16,384 EXACTLY** (job 7879369)

Rung 7 (1024, 24933) died in 2m51s.  This is the first wall the ladder has hit
on the mu axis, and it is NOT the memory wall R8 predicted — it arrives well
before it.

CLASSIFICATION: **LORRAX FAIL-FAST** (not OOM-killer, not XLA allocator, not
QE).  All 64 ranks raised the same ValueError at RESOLVE time; peak
`VmHWM = 39.64 GB` against a planner estimate of 76.45 GB — memory was never
the issue.  sacct: step FAILED 1:0, MaxRSS 42,225,676 kB (concurs).

    ValueError: charge_zeta_solve='rank_truncate' needs the replicated route,
    but the CCT stack (nq=10, n_mu=24933) is 92.63 GiB > the 4.00 GiB cap.

THE OBJECT AND ITS SCALING LAW (source-read, `isdf/core.py`):

    _replicated_factor_q_chunk(nq, mu) = max(1, min(nq, 4GiB // (mu^2 * 16)))
    _replicate_rank_truncate_ok  <=>  batch * mu^2 * 16
                                      <= max(LORRAX_ZETA_REPLICATE_CAP_GIB,
                                             _REPLICATED_FACTOR_MAX_BATCH_BYTES = 4 GiB)

At every ladder size `batch` has already collapsed to 1, so the criterion is
just `mu^2 * 16 <= 4 GiB`, i.e.

    **mu_max = sqrt(4 GiB / 16 B) = 16,384 EXACTLY, independent of nq.**

The object is ONE q-slice of the replicated rank-truncating charge factor —
`mu^2` complex128, QUADRATIC in mu and REPLICATED on every rank (no P scaling
whatsoever).  Verified against the ladder:

| mu | q-slice | default cap | observed |
|---|---|---|---|
| 10015 (rung 5) | 1.49 GiB | OK | ran, factor 240.6 s |
| 15007 (rung 6) | 3.36 GiB | OK | running |
| 19991 | 5.96 GiB | REFUSED | (not attempted) |
| 24933 (rung 7) | 9.26 GiB | **REFUSED** | **died 2m51s** |

Why it refuses instead of downgrading is deliberate and correct: the comment
at core.py:1157-1164 records that on 2026-07-21 a silent fallback here
returned zeta 4.5x too large and rebuilt V_q to relF 16-32 instead of 1.8e-15.
`rank_truncate` is the only route carrying the rank-truncation physics cure,
so above the cap the code refuses rather than silently lose it.

### R10.1 — MESSAGE DEFECT worth fixing (minor, but it misdirects the operator)

The error quotes the WHOLE STACK, `nq*mu^2*16 = 92.63 GiB`, and therefore
advises `LORRAX_ZETA_REPLICATE_CAP_GIB=94`.  But the branch that actually
refused tests ONE q-batch (`mu^2*16 = 9.26 GiB`) — that is the entire point of
`_replicate_rank_truncate_ok` being a DIFFERENT criterion from
`_replicate_charge_ok`.  Following the message would have the operator reserve
94 GiB where 10 suffices, i.e. it turns a workable escape into an apparently
impossible one.  Suggested fix: report `batch*mu^2*16` and suggest that.

### R10.2 — ESCAPE SPENT (the ONE lever, and it plausibly applies)

`LORRAX_ZETA_REPLICATE_CAP_GIB=16` — chosen from the CORRECT quantity
(9.26 GiB q-slice), not the message's 94.  Cost accounting done BEFORE
submitting:
- memory: +9.26 GiB/rank replicated transient on top of the 76.45 GB planner
  estimate is affordable only because the planner peak and this transient are
  in different stages; worst case ~86 GB vs 96 GB/rank physical.  TIGHT — this
  is itself a thing to measure.
- time: the factor is a dense eigh per q, O(mu^3), run REDUNDANTLY on every
  rank (core.py:1019-1024 documents this as "the real mu ceiling on this
  route ... needs a genuinely distributed eigh (SLATE/ScaLAPACK), not a bigger
  cap").  Extrapolating rung 5's measured `zeta_fit.cholesky = 240.6 s` at
  mu=10015 by (24933/10015)^3 = 15.4x gives **~3712 s (62 min)** for the
  factor alone — affordable inside the 8 h `normal` job, and the run is
  wired with ZCAP as a first-class knob.
Resubmitted as **job 7879380**.

**THIS IS A CERTIFIED ARCHITECTURAL WALL even though the escape exists.**  The
lever buys headroom, not scalability: the object is replicated `mu^2` and the
work is `O(nq*mu^3)` redundant per rank with ZERO P-scaling, so at fixed P=64
it cannot be escaped by buffer donation, comm reduction, or sharding — only by
implementing a genuinely distributed eigh.  That is precisely the
"thousands of low-memory ranks" collision the campaign exists to find, and it
is the first wall on this ladder that is architectural rather than budgetary.

### R9.10 — 45 Ry b2048 DECK COMPLETE, all gates green (job 7879367, 8m47s)

| artifact | size | wall | gate |
|---|---|---|---|
| WFN_r45_b2048.h5 | 2,351,256,564 B | — | el_compare vs r45_b1024 **1.405e-10 eV PASS** |
| kin_ion_r45_b2048.h5 | 2,147,491,840 B | 292 s | rc=0 |
| dipole_r45_b2048.h5 | 3,758,102,528 B | 117 s | rc=0 (MANDATORY regen: cutoff AND window both changed) |
| gate_h0 @ nb=2048 | — | — | **rms(H0-kih) = 3.1e-5 eV**, max abs 3.19e-4 |
| implied Vxc vs QE | — | — | **rms 3.1e-5 eV**; ranges [-24.2579,-2.5808] vs [-24.2583,-2.5807] |
| density symmetry | — | — | TRS HOLDS 3.51e-15, spatial 2/2 resid 1.78e-14, integral rho = 26.000000 (rel 0.0e+00) |

The 45 Ry lineage is now established end-to-end from its own SCF through a
fully gated 2048-band LORRAX deck.  Centroids on the new 25x25x100 grid are
building (job 7879368, targets 13000/18000/23000/28000).  Note the artifact
sizes: dipole is 3.76 GB and kin_ion 2.15 GB at nb=2048 — both scale as nb^2,
so the 45 Ry lineage's deck footprint is 4x the 30 Ry b1024 family's.

**IMPORTANT for the 45 Ry GW rungs**: R10's wall applies here too and is
CUTOFF-INDEPENDENT — `mu_max = 16,384` at the default cap, since the criterion
is `mu^2*16 <= 4 GiB` with no reference to the grid or the cutoff.  The 45 Ry
ladder targets 13000/18000/23000/28000, so **only the 13000 rung clears the
default cap**; 18000 and up require `ZCAP` exactly as rung 7 did.  Budget the
eigh accordingly: O(mu^3) redundant per rank, calibrated at 240.6 s for
mu=10015 (rung 5), so mu=23000 costs ~(23000/10015)^3 x 240.6 = ~2900 s and
mu=28000 ~5250 s for the factor alone.

### R10.3 — RUNG 8 (1024, 32059): the planner has RUN OUT OF ROOM, as predicted

Second pre-registered prediction, again written down before submitting
(job 7879381, `ZCAP=24`):

| quantity | pre-registered | actual | error |
|---|---|---|---|
| persistent | 21.43 GB | 21.28 GB | 0.7% |
| r_from_budget | 25,123 | — | — |
| r_lo BINDS? | **YES** (32059 > 25123) | **YES** (r_chunk = r_lo, not the budget value) | exact |
| r_chunk | 32,059 | 32,064 (= r_lo rounded up to p_xy=64) | 0.02% |
| **HWM** | **91.70 GB** | **91.47 GB** | **0.25%** |
| % of budget | 102% | **102%** | exact |

**The planner is now emitting a plan it KNOWS exceeds the budget** — 91.47 GB
against the 90 GB it was given, because `r_chunk` cannot go below
`r_lo = min(mu, n_rtot) = 32059` and there is no other knob in the automatic
path.  This is precisely the R8 mechanism, arriving at the predicted mu.

`P_min (floor)` continues to climb — **1 -> 3 -> 6 -> 10** at
mu = 10015 / 15007 / 24933 / 32059.  Extrapolating the persistent formula,
`P_min` reaches 64 (i.e. the resolver would refuse outright at this
architecture) somewhere near mu ~ 90-100k, far beyond the n_rtot = 46080
ceiling on this deck — so on THIS deck the r_lo/budget wall arrives first and
`P_min` stays a diagnostic rather than a wall.

Outcome to be recorded when the job lands: predicted 91.47 GB x the observed
0.945-0.956 ratio gives ~87 GB actual against **96 GB/rank physical**
(192 GB node / 2 ranks), so rung 8 sits in the band where it may survive
narrowly.  The zeta replicated q-slice (15.32 GiB at this mu, admitted by
ZCAP=24) lands in a DIFFERENT stage (B_cct_chol, planner 58.29 GB) so the two
peaks should not add.  If rung 8 survives, mu ~ 34,600 is where the C_fit peak
reaches 96 GB physical on the same law:
    HWM(r_lo-bound) = 72.05*mu^2 + 550,574*mu  bytes
and mu <= n_rtot = 46080 remains the hard ceiling of the axis.

### R10.4 — the TIME wall is overtaking the memory wall on the mu axis

Measured this session, all with the weapons ON (so the FFT FFI's 3.78x on
sigma.exec is ALREADY applied):

    rung 5 (mu 10015): whole run 1770 s; zeta_fit 358.3 s (cholesky 240.6 s)
    rung 6 (mu 15007): FIRST sigma window alone = 714 s (4 windows to go)
    rung 7 (mu 24933): replicated rank-truncate eigh projected ~3712 s
    rung 8 (mu 32059): same eigh projected ~7890 s (2.2 h) BEFORE sigma

Two superlinear terms are now compounding:
1. `zeta_fit.cholesky` — the mu^3 series 11.9 / 24.2 / 36.9 / 91.5 / 240.6 s
   across rungs 1-5, and above mu=16,384 it becomes the REPLICATED eigh of
   R10, which is O(nq*mu^3) redundant on every rank with no P scaling.
2. `sigma.tau.host_accum` — the pre-existing superlinear series
   73 / 136 / 329 / 610 s at mu 2475/3491/4951/6947 (ladder wall #2).

**Consequence for the campaign**: on the mu axis at fixed P=64 the ladder now
runs out of TIME before it runs out of MEMORY.  The dev queue's 2 h wall
would already have killed rungs 7-8 (hence the move to `normal`), and the
projected rung-8 wall is >3 h.  The memory frontier at mu ~ 34,600 is
reachable in principle but each rung costs hours, and every hour of it is
spent in code that does not scale with P.  This reinforces R10's verdict: the
binding constraint on this axis is architectural (redundant O(mu^3) work and
replicated mu^2 objects), not the per-rank memory budget.

## R11 — SESSION INDEX (successor agent #2, 2026-07-29 01:04-02:15)

These notes are append-only, so the sections below are interleaved
chronologically.  Logical reading order for this session:

**Handover and hygiene**
  R7.3   handover, the MOVING-TREE hazard and the srcpin_4f77842 fix

**Rung 5 (30 Ry, 1024 bands, mu 10015) — prerequisites all green**
  R7.3a  NSCF harvest + el_compare 1.859e-11 eV (validates the 1st QE patch)
  R7.3b  deck: kin_ion, dipole (MANDATORY), gate_h0 3.9e-5 eV
  R7.3c  **blocked-Gram ACCEPTANCE TEST: PASS on a standard 192 GB node**
  R7.3d  pre-submit planner check

**Rung 5 result and the physics investigation**
  R7.4   MEMORY GREEN (36.03 vs 37.67 GB), **PHYSICS RED** (gap 0.36 eV)
  R7.5   Σ_X exonerates WFN + centroids
  R7.6   head/dipole exonerated; defect isolated to the Σ_c body
  R7.7   bisect 1: ALL weapons at (512,6947) reproduce rung 4 EXACTLY
  R7.8   bisect 2: minus-sharded also exact; **weapons are a 1.84x speedup**
         => the defect is a SIZE effect or a weapon x size interaction >512b

**The memory model and its walls**
  R8     mechanism: the r-chunk PERFORMANCE FLOOR r_lo = min(mu, n_rtot)
  R8.1   MODEL GAP: the planner does not model the Σ omega cube
  R8.2   model is PREDICTIVE to <1% (rung 7 pre-registered)
  R10.3  rung 8 pre-registered to 0.25%; planner now at 102% of budget
  R10.4  the TIME wall is overtaking the memory wall

**The certified wall**
  R10    **DEATH #2: FAIL-FAST at mu = 16,384 EXACTLY**, scaling law, evidence
  R10.1  the error message misdirects (quotes the stack, tests the q-batch)
  R10.2  escape spent (ZCAP) and CONFIRMED working

**The 45 Ry deck lineage (owner directive)**
  R9     provenance chain and the INTERNAL-gates ruling
  R9.1   measured 45 Ry basis (FFT 25x25x100 = 62500, n_rtot x1.356)
  R9.2   **the deck has TEN k-points in the IBZ, not sixteen**
  R9.3   band ceilings; the npol patch is STILL load-bearing at 2048
  R9.4   parallelization probe, five configs — winner 1 node -nk 10 -ndiag 1
  R9.4a  measured ngkmax 3597; ecut^1.5 scaling confirmed to 0.3%
  R9.5   explicit memory fit per pool
  R9.6   cost verdict
  R9.7   **DEATH #1** and the c_bands.f90 npol patch
  R9.8   NSCF 2048 SUCCESS; the 1.7 meV cutoff-convergence observation
  R9.9   the 2nd QE patch NUMERICALLY VALIDATED (1.405e-10 eV)
  R9.10  45 Ry b2048 deck COMPLETE, all gates green

**Still in flight at the end of the session** (harvest these first):
  job 7879349 rung 6 (1024, 15007) — memory + gap
  job 7879380 rung 7 (1024, 24933, ZCAP=16) — first run past the mu=16384 wall
  job 7879381 rung 8 (1024, 32059, ZCAP=24) — planner 91.47 GB = 102% of budget
  job 7879357 (1024, 10015) WEAPONS=off  } the two runs that decide whether
  job 7879382 (1024, 10015) SHARDED=off  } the rung-5 defect is size or sharded
  job 7879368 45 Ry centroids (13000/18000/23000/28000 on the 62500 grid)
  jobs 7879344 / 7879354 30 Ry centroids c30000 / c33000
Harvest recipe: `wk_REL/harness/qpgap.sh <run_dir>` for the gap (convention certified
against rungs 3 and 4), and the `MAX VmHWM across ranks` line in the job .out.

### R11.1 — the source pin was VINDICATED, quantitatively

At handover the live repo was at 4f77842 with uncommitted edits (mtime 00:39).
By 02:16 it had advanced to **068286c** — the concurrent GEMM/BSE workstream
committed during this session.  `git diff --stat 4f77842 068286c -- src`:

    10 files changed, 560 insertions(+), 114 deletions(-)
      src/common/contract_bands.py    | 194 +++++---   <-- sigma project_rs HOT PATH
      src/ffi/mklblas/cpp/gemm_batch_ffi.cc | 238 +++++---
      src/ffi/common/cpp/host/CMakeLists.txt | 133 +++---
      src/bse/vq_interp.py, ...

Every 32-node run this session imported
`wk_REL/snapshots/srcpin_4f77842/src` instead, verified at the end to still contain
exactly the 228 .py files of commit 4f77842.  Had the runs used the live tree,
the ladder's rungs would have been executed against a moving 560-line diff on
sigma's hot path — and the rung-5 physics investigation (R7.4-R7.8), which
turns entirely on comparing configurations, would have been uninterpretable.
**Recommendation: make the source pin standard for all multi-hour batch work
whenever another workstream shares the tree.**  Cost: one `git archive`, ~850 KB,
zero repo mutation (no commit, no checkout, no worktree metadata).

### R10.5 — the mu=16,384 wall is now BRACKETED ON BOTH SIDES, three instances

An accidental but useful control: the c30000 auto-chain submitted mu=30011
WITHOUT `ZCAP` (job 7879389).  It died the same way, at the predicted size:

    predicted q-slice 30011^2*16 = 13.42 GiB  vs the 4 GiB default cap
    ValueError: ... the CCT stack (nq=10, n_mu=30011) is 134.21 GiB > 4.00 GiB

(and note the message again quotes the STACK, 134.21 GiB, advising CAP=136,
where the branch actually tested the 13.42 GiB q-slice — R10.1 confirmed a
second time.)

**Empirical bracket of the wall, all at nb=1024 / P=64 / 30 Ry:**

| mu | q-slice mu^2*16 | default cap (4 GiB) | job | outcome |
|---|---|---|---|---|
| 10015 | 1.49 GiB | under | 7879295 | ran (factor 240.6 s) |
| 15007 | 3.36 GiB | under | 7879349 | ran |
| **16,384** | **4.00 GiB** | **= cap** | — | **the wall** |
| 24933 | 9.26 GiB | OVER | 7879369 | **REFUSED** (2m51s) |
| 30011 | 13.42 GiB | OVER | 7879389 | **REFUSED** |
| 24933 + ZCAP=16 | 9.26 GiB | admitted | 7879380 | ran past the gate |
| 32059 + ZCAP=24 | 15.32 GiB | admitted | 7879381 | ran past the gate |

Two instances below, two above, two escaped — the threshold
`mu_max = sqrt(4 GiB / 16 B) = 16,384` is confirmed from both directions and
the escape is confirmed effective at two different mu.  **This is the
campaign's certified frontier on the mu axis at fixed P=64.**

## R12 — **NAMED MECHANISM for the rung-5 RED: the ISDF prune window never
## tracks the deck's conduction window** (coordinator priority, 2026-07-29)

### R12.0 — the size axis, with rung 6 harvested

| rung | (nb, mu) | mu/nb | eqp0 | eqp1 | verdict |
|---|---|---|---|---|---|
| 1 | (256, 2475) | 9.7 | 3.5819 | 3.2516 | GREEN |
| 2 | (256, 3491) | 13.6 | 3.6290 | 3.2895 | GREEN |
| 3 | (512, 4951) | 9.7 | 3.4594 | 3.1603 | GREEN |
| 4 | (512, 6947) | 13.6 | 3.2194 | 2.9778 | GREEN (weapons and no-weapons IDENTICAL) |
| 5 | (1024, 10015) | 9.8 | **0.3645** | **-0.3639** | **RED** |
| 6 | (1024, 15007) | 14.7 | **1.4296** | **1.8522** | **RED, but LESS wrong** |

Two facts fix the shape of the defect:
1. It appears between nb=512 and nb=1024 at essentially CONSTANT mu/nb ratio
   (13.6 green at 512, 9.8 red at 1024) — so it is not a mu/nb budget effect.
2. At FIXED nb=1024 it IMPROVES strongly with mu (0.36 -> 1.43 eqp0 for
   mu 10015 -> 15007).  A logic/indexing bug would not heal with more mu.
Together these say: **the ISDF basis is under-resolving something whose
difficulty grows with nb, while the thing that selects the basis does not.**

### R12.1 — the defect, named

`src/centroid/kmeans_cli.py`, `_resolve_sigma_window()`:

    n_val  = int(args.prune_n_val)  if given else int(wfn.nelec)      # 26
    nb_total = int(wfn.nbands)                                         # 1024
    n_cond = int(args.prune_n_cond) if given else min(n_val, nb_total - n_val)
                                                     # -> min(26, 998) = 26

`n_cond` **defaults to n_val**, so the pivoted-Cholesky prune window is
`left=(0,26) right=(0,52)` for EVERY deck this campaign has ever built.
Disk-verified identical in all four centroid logs:

    b256  c2500 (job 7878101): prune window: v x (v+c) left=(0,26) right=(0,52)
    b512  c7000 (job 7878358): prune window: v x (v+c) left=(0,26) right=(0,52)
    b1024 c10000(job 7879286): prune window: v x (v+c) left=(0,26) right=(0,52)
    b1024 c15000(job 7879344): prune window: v x (v+c) left=(0,26) right=(0,52)

and the ladder harness passes no `--prune-n-cond`, so the default always won.
Meanwhile the deck's ACTUAL sigma window is `nval=26 + ncond=998 = 1024`.

**So the ISDF centroids are selected to resolve a 26x52 pair-density block
while Sigma_c consumes a 1024x1024 one.**  The mis-targeting has been present
since rung 1; it is harmless while the basis is incidentally rich enough
(nb<=512) and becomes fatal at nb=1024, which is exactly the observed
threshold, and it is partially rescued by raising mu, which is exactly the
observed mu-dependence.

TWO aggravating factors in the same place:
- The docstring of `_resolve_sigma_window` says it returns *"(n_val, n_cond)
  of the sigma window — the bands the ISDF must span"* and calls itself the
  *"single source of truth for both consumers"*.  That is FALSE whenever
  `ncond != nval`.  The `--prune-n-cond` help string ("default = n_val") is
  literally correct, so the CLI is honest and the docstring is not.
- The default prune MODE is `v_x_vc` = `left (0, n_val)`, so **cond x cond
  pair densities never enter the selection criterion at all**, at any n_cond.
  For Sigma_c of a conduction QP state summed over 998 conduction bands, cxc
  pairs are the bulk of the sum.  `--prune-window vc_x_vc` exists for exactly
  this ("full sigma-window square Gram, also includes cxc") and is not used.

### R12.2 — the decisive test (submitted, dose-response by design)

`deck_b1024_prunefix.sbatch`: SAME N=10000, SAME WFN, SAME orbit closure as
the rung-5 centroid file; the ONLY change is the prune window.
  job 7879432  `--prune-n-cond 230`  -> right=(0,256),  ~4.9x Gram band work
  job 7879433  `--prune-n-cond 998`  -> right=(0,1024), ~19.7x (FULL, correct)
Then rerun rung 5 on each.  PREDICTION: the QP gap moves back toward
3.2-3.6 eV monotonically with the prune width.  If it does not, this mechanism
is exonerated and the search moves to the omega/window construction
(`ppm_windows.py` `_SigmaBranch` E_ref_A/E_ref_B are `np.min` over
band-range-masked E_A, i.e. genuinely band-range-derived) — which is the next
suspect and has NOT yet been cleared.

### R12.3 — status of this claim

MECHANISM IDENTIFIED AND CONSISTENT WITH EVERY OBSERVATION, **NOT YET PROVEN**.
What is proven: the clamp exists, it applied to every centroid set the
campaign owns, and the deck's sigma window is 20x wider than the prune window
at rung 5.  What is not yet proven: that closing the clamp restores the gap.
Jobs 7879432/7879433 decide it.  I am NOT calling it a release blocker until
that returns — but if it lands, the release-blocking statement is:
**"ISDF centroid selection silently ignores the deck's conduction window, so
every GW result with ncond >> nval rests on a basis chosen for a different,
much smaller problem."**

### R12.4 — QUANTITATIVE CONFIRMATION, before the GW rerun even starts

`deck_b1024_prunefix.sbatch` job **7879432** rebuilt the rung-5 centroid set
with the ONLY change being `--prune-n-cond 230` (prune window
`right=(0,52)` -> `right=(0,256)`).  Same N=10000, same WFN, same candidate
pool M=13872, same orbit closure, same 897-orbit target:

| quantity | DEFAULT window (0,52) | CORRECTED window (0,256) |
|---|---|---|
| job | 7879286 | 7879432 |
| Gram diag range | [**7.632e-17**, 1.158e-04] | [**7.189e-13**, 1.784e-04] |
| orbits requested | 897 | 897 |
| **achieved rank** | **630** | **897** |
| accepted centroids | 10015 | 10015 |
| VmHWM | 66.2 GB | 69.4 GB |
| wall | 308 s | 263 s |

**The default prune window is RANK-DEFICIENT BY 30%.**  Pivoted Cholesky was
asked for 897 independent interpolation directions and could only find 630,
because the 26x52 pair-density block does not excite most of the candidate
centroids — their Gram diagonals sit at 7.6e-17, i.e. numerical zero.  Widen
the window to 26x256 and the SAME candidate pool delivers the full 897.

So the rung-5 centroid file contains 10015 points of which the selection
criterion could only certify 630 directions' worth of independence, and those
630 were chosen to span a 52-band problem.  Sigma_c then used them for a
1024-band problem.  This is the mechanism, measured, and it required no GW run
to see — **the rank number was printed in every centroid log the campaign has
ever produced and nobody was reading it.**

Recommended cheap gate #3 (free, no extra job): assert
`achieved rank == requested orbits` in the centroid acceptance step
(`b256_verify.py`).  A 30% shortfall should have been a hard refusal.

GW confirmation in flight: job **7879438** reruns rung 5 on the corrected set
(`centroids_b1024_pn230_c10000.txt`, 10015 centroids); job 7879433 is building
the full `--prune-n-cond 998` (right=(0,1024)) dose behind it.

## R13 — OWNER HYPOTHESIS (zeta-fit conditioning at large nb): **STRONGLY
## SUPPORTED by existing artifacts, no new jobs required**

The owner proposed that the rung-5 defect is numerical instability in the zeta
fit at large nb — continuum pair densities making the CCT Gram ill-conditioned
and pushing rank truncation into a regime where it truncates real content.
`isdf/core.py:1645-1660` already prints exactly the needed diagnostic on every
run (`[zeta rank_truncate/distributed]`), and nobody had read it.  Extracted
from the four existing rung logs:

| rung | nb | mu | n_pad | n_keep(q0) | keep % | lam_max | lam_min_kept | kappa |
|---|---|---|---|---|---|---|---|---|
| 3 | 512 | 4951 | 4992 | 4183 | **83.8%** | 0.17610 | 1.760e-09 | 1.00e8 |
| 4 | 512 | 6947 | 6976 | 4570 | **65.5%** | 0.21562 | 2.157e-09 | 1.00e8 |
| 5 | 1024 | 10015 | 10048 | 6700 | **66.7%** | 0.35824 | 3.584e-09 | 1.00e8 |
| 6 | 1024 | 15007 | 15040 | 7108 | **47.3%** | 0.51229 | 5.120e-09 | 1.00e8 |

(kappa is 1/rcond by construction — rank_truncate cuts at `lam_max*rcond`, so
the retained span is always exactly 1e8.  The informative numbers are n_keep
and the truncation floor.)

### R13.1 — the retained rank SATURATES: exactly the plateau the owner predicted

    nb=512 : mu 4951 -> 6947  (+40.3%)  =>  rank 4183 -> 4570  (+9.3%)
    nb=1024: mu 10015 -> 15007 (+49.8%) =>  rank 6700 -> 7108  (+6.1%)

Adding 50% more centroids buys 6% more retained rank.  **The ISDF fit cannot
be repaired by adding mu** — which is why rung 6 is still RED (1.4296) despite
15007 centroids, and why the mu-axis escalation was chasing something it could
never catch.

### R13.2 — the truncation is discarding REAL CONTENT, by six decades

At rung 6 the cut is at `lam_max * rcond = 0.51229 * 1e-8 = 5.12e-9`, and it
throws away **7,932 of 15,040 modes (53%)**.  The f64 noise floor of a Gram
with `lam_max = 0.512` and n = 15040 is `~1e-16 * lam_max * sqrt(n)
= 6.3e-15`.  So the discarded modes sit **~10^6 above numerical noise** — they
are physical content, not roundoff.  `zeta_rcond = 1e-8` is a default chosen
for much smaller fits and it is far too aggressive here.

Note the trend is already visible WITHIN the green rungs: 83.8% retained at
rung 3, 65.5% at rung 4 — the ladder was eating into real content well before
nb=1024, which is consistent with the green rungs' own mu-convergence being
non-monotone (+47 meV at 256b, -240 meV at 512b, R4.3).  The defect did not
switch on at nb=1024; it crossed a severity threshold there.

### R13.3 — TEST SUBMITTED (owner check #2)

Rung-5 configuration (nb=1024, mu=10015, the RED point), SAME centroid set,
only `zeta_rcond` changed:
    job 7879450  zeta_rcond = 1e-10   (retains ~2 more decades of spectrum)
    job 7879451  zeta_rcond = 1e-12   (~4 more decades; kappa 1e12 leaves ~4
                                       f64 digits, so this may over-retain)
`zeta_ridge` is also now wired (`ZRIDGE=`) for the Tikhonov variant.
PREDICTION: if conditioning is the mechanism the QP gap moves monotonically
toward the 3.2-3.6 eV family as rcond drops, and there is an optimum before
kappa eats the f64 mantissa.

### R13.4 — how this relates to R12 (the prune-window clamp)

They are DIFFERENT objects and are not in competition:
- **R12** is the *selection* Gram — pivoted Cholesky over candidate centroids,
  a preprocessing step.  Measured rank-deficiency 630/897 with the default
  26x52 window, cured to 897/897 by widening it.
- **R13** is the *fit* Gram — the runtime CCT whose spectrum is truncated at
  `zeta_rcond`.  Measured 47-67% retention.
Both are ISDF conditioning and both plausibly contribute.  The two prune-fix
reruns already in flight (jobs 7879438 pn230, 7879441 pn998) hold rcond fixed
and change only the centroid POSITIONS, so together with 7879450/7879451
(which hold positions fixed and change rcond) the 2x2 separates them cleanly.

### R13.5 — what this would mean if confirmed

Not a code bug but a METHOD LIMIT, and a publishable one: on a 4x4 30 Ry cell
the upper bands are deep in the continuum, their pair densities are nearly
degenerate, and the ISDF pair-density manifold's numerical rank saturates near
~7,100 at rcond 1e-8.  Beyond that, additional bands enter Sigma_c through a
basis that cannot represent them.  **The ladder must therefore publish a
maximum usable nb per cutoff**, and the campaign's `N_mu ~ 6-14x nband`
heuristic is invalid above the saturation point — the binding quantity is the
retained rank, not mu.

## R14 — OWNER-APPROVED FIXES, GATES AND WRITE-UP (2026-07-29)

Code lives in worktree **/work2/08271/jackmc/frontera/wt-RELC**, branch
**wsREL-isdf-window**, base 4f77842, **NOT COMMITTED** (house rule: the
orchestrator merges).  `python3 -m py_compile` PASS.  +80/-2 lines, one file.

### R14.1 — APPROVED FIX 1: the prune window now matches what Sigma_c consumes

`src/centroid/kmeans_cli.py::_resolve_sigma_window`

    -   else min(n_val, nb_total - n_val)     # clamped conduction extent to n_val
    +   else max(0, nb_total - n_val)         # FULL conduction window in the WFN

Framed as a **BUG, not a default change**, in the docstring: it states
explicitly that every pre-fix centroid set — b256 c2500, b512 c7000, b1024
c10000, b1024 c15000 — was selected on a 26x52 block, carries the measured
rank table (630/897 -> 897/897), and records the measured cost of the fix
(+13% wall, +15 GB peak).  The `--prune-n-cond` help text, which still said
"default = n_val", was corrected too.

Choice of default: the FULL WFN conduction window rather than the deck's ncond,
because kmeans_cli does not read the deck and a superset is always safe.
Explicit narrowing remains available and is now guarded by the rank gate.

### R14.2 — APPROVED FIX 2 (gate 3): rank assertion as a HARD REFUSAL

Same file, after the `After pruning: N centroids (rank=R)` print.  Refuses when
`rank < ceil((1 - tol) * n_orbit_keep)`, default `tol = 0.01`, override
`LORRAX_CENTROID_RANK_TOL`.  The message is actionable: it names the requested
vs achieved counts and the percentage, echoes the prune window that produced
them, explains that the file would be padded with numerically-null directions
and can be wrong by electron-volts without failing any other gate, and lists
the three fixes (`--prune-n-cond`, `--prune-window vc_x_vc`, `--oversample`/N).
On success it prints `[rank gate] R/K directions certified ... PASS`.

Harness-side twin: **`wk_REL/harness/centroid_rank_gate.sh`** (parses an existing
kmeans log, so it also guards runs made against pinned/older source).  Wired
into `deck_b1024_cbig`, `deck_r45_cent`, `deck_b1024_prunefix`,
`deck_b1024_c10000`.  SELF-TESTED both ways:

    defective b1024 c10000 (job 7879286): requested 897 achieved 630 -> FAIL (exit 2)
    fixed     pn230        (job 7879432): requested 897 achieved 897 -> PASS (exit 0)

### R14.3 — GATE 2 (Sigma reference) is live and passing

`gate_sigma_reference.sbatch`, job 7879439: `eqp0 3.5819 / eqp1 3.2516`,
**|d| = 0.00e+00 eV** against the rung-1 pin, at tol 1e-3.  Bit-exact across a
commit range AND weapons on/off — the current build reproduces the campaign's
oldest Sigma_c result exactly, so the codebase is not broken in general.

### R14.4 — ITEM 3: proposed rcond rule (NOT adopted; measurement in flight)

`zeta_rcond = 1e-8` is scale-dependent in the wrong direction: the absolute
truncation floor is `lambda_max * rcond`, which RISES with fit size
(1.76e-9 -> 5.12e-9 across the ladder), while the true noise floor
`~eps * lambda_max * sqrt(n_pad)` stays ~6e-15.  So bigger fits truncate more
aggressively in absolute terms exactly when they can least afford it.
Proposed rule, to be adopted only if the 2x2 supports it:

    rcond_eff = max(rcond_floor, C * eps * sqrt(n_pad))       # C ~ 10-100

which lands near 1e-14..1e-13 at these sizes instead of 1e-8.  **Not applied** —
a physics-affecting default is not changed silently; jobs 7879450 (1e-10) and
7879451 (1e-12) are measuring the response first.

### R14.5 — ITEM 5: two NAMED DEFECTS recorded

**DEFECT D1 — `sigma_omega_layout=replicated` HANGS at nb>=1024.**
Jobs **7879357** (WEAPONS=off) and **7879382** (SHARDED=off), both
(nb=1024, mu=10015), P=64.  Evidence:
- both reached `Finished sigma[w<E_F val]` cleanly in gw.log, elapsed 622 s and
  338 s respectively; log then STOPS with no error, no traceback, no warning;
- 20+ minutes later: no `sigma_mnk.h5`, no `eqp0.dat`/`eqp1.dat`, no
  `sigma_diag.dat`; gw.log mtime frozen at the sigma-finish timestamp;
- per-rank /proc sampling showed **VmRSS = 0.00 GB on ranks 0, 1, 17, 33, 63**
  with VmHWM retained at ~36 GB — i.e. **every rank's process had exited** while
  `srun` never returned and Slurm still reported the step RUNNING;
- rungs 5 and 6 at the SAME size with `sharded` ON completed normally.
Both cancelled.  **Release-note-level consequence: `sigma_omega_layout=sharded`
is MANDATORY at nb>=1024.**  Practical consequence for this campaign: the
"weapons-off at nb=1024" control is unobtainable by that route (moot — the
b512 bisect showed the weapons bit-identical and R14.3 reproduces rung 1
exactly with weapons ON).

**DEFECT D2 — the chunk planner is blind to the sigma omega cube.**
`gflat_memory_model.py` models only the ISDF chunk plan (A_centroid_load /
B_cct_chol / C_fit_one_rchunk / D_accumulate / E_v_q / F_tensor_write).  The
Sigma-stage omega cube is outside it entirely, so replicated and sharded runs at
(1024, 10015) print an IDENTICAL `HWM estimate = 37.67 GB/dev` although the
replicated layout additionally holds `n_omega*nk*nb^2*16 B` per rank — 2.75 GB
at nb=512, ~11 GB at nb=1024, ~44 GB projected at nb=2048, where it would BIND
before C_fit and the planner would not see it coming.  Consequence for the
record: the 0.945-1.063 predicted/measured agreement claimed across rungs 1-8
is a statement about the ISDF stage ONLY, valid while C_fit is the global
binder.  It should be quoted that way from now on.

### R14.6 — ITEM 4: standalone write-up for the repo docs

**`docs/dev/isdf_rank_saturation_and_max_usable_nb.md`** (in wt-RELC, new file):
why the gate suite missed it; the saturation table and the two readings
(rank saturates; truncation is six decades above noise); the selection-window
bug and its rank cost; **the practical rule — a MAX USABLE nb PER CUTOFF, with
a four-step recipe using the diagnostic the code already prints**; the explicit
**retirement of the `N_mu ~ 6-14x nband` heuristic** (all three failing points
sit INSIDE the band, so membership tells you nothing); both new gates; the open
`zeta_rcond` item; and D1/D2.  Written for users of the released code, not for
this campaign.

> ⚠ CLAIM-DECAY on R14.4 (same session, 2026-07-29): my proposed `zeta_rcond`
> rule was WRONG and is WITHDRAWN.  I proposed sizing rcond against the f64
> Gram noise floor (`~C*eps*sqrt(n_pad)` -> ~1e-14).  `gw_config.py:617-628`
> documents that `zeta_rcond = 1e-8` was chosen from a MEASURED sweep and is
> the LOW end of an over-complete recovery PLATEAU spanning 1e-8..1e-4:
>   * MoS2 4x4 / 1204c: **1e-10 only PARTIALLY recovers, MAE 1.4 eV vs BGW**,
>     while the whole 1e-8..1e-4 plateau collapses to ~0.04 eV;
>   * bulk Si 4x4x4 / 960c (BGW-anchored gate) genuinely has eigenvalues below
>     the cut, and 1e-6 drifts sigTOT by 1.021 meV where 1e-8 costs 0.054 meV.
> Evidence: reports/gw_rank_truncation_2026-07-20,
> gw_bandrange_centroids_2026-07-21, sweep table in docs_gwjax/COHSEX_INPUT.md.
> So LOWERING rcond is documented to be WORSE, not better — truncation is a
> CURE for over-completeness here, not merely a numerical hygiene knob.  My
> "truncation discards real content six decades above noise" arithmetic stands
> as arithmetic, but the inference that retaining those modes would HELP does
> not: they are the over-complete directions the cure exists to remove.
> The owner's ruling (handle numerical instability in a separate pass) is the
> right call and this is another reason for it.  The 2x2 axis-B cells
> (jobs 7879450 rcond=1e-10, 7879451 rcond=1e-12) are now expected to be
> WORSE than baseline; they are recorded as data, not as a proposed fix.
> **NO physics-affecting default was changed.**

## R15 — CAPACITY PASS (owner redirect 2026-07-29): parking physics, removing caps

### R15.0 — 2x2 recorded and PARKED (retained rank per cell, from the logs)

All four cells are (nb=1024, mu~10015, n_pad=10048), same WFN:

| cell | job | prune window | rcond | n_keep | retained |
|---|---|---|---|---|---|
| baseline (rung 5) | 7879295 | (0,52) | 1e-8 | 6700 | 66.7% |
| A1 prune fix | 7879438 | (0,256) | 1e-8 | 6793 | 67.6% |
| A2 prune fix full | 7879441 | (0,1024) | 1e-8 | 6788 | 67.6% |
| B1 | 7879450 | (0,52) | 1e-10 | **8290** | 82.5% |
| B2 | 7879451 | (0,52) | 1e-12 | **9461** | 94.2% |

**Structural result, independent of the QP outcome**: widening the prune window
moves the FIT Gram's retained rank almost not at all (6700 -> 6793, +1.4%),
while rcond moves it a lot (6700 -> 9461).  So the SELECTION defect (R12) and
the FIT truncation (R13) are largely INDEPENDENT knobs — the selection fix buys
a better-conditioned basis at identical mu (rank 630 -> 897 in the SELECTION
Gram) but does not by itself change how much of the fit spectrum survives.
That is why the selection fix is kept as a capacity enabler while the rcond
question is parked for the numerical-stability pass.

### R15.1 — **THE mu=16,384 WALL IS A PREMATURE REFUSAL, NOT A CAPACITY LIMIT**

Scoped against wk_AP's two-plan memo as directed — and the memo turned out not
to be needed for THIS deck's wall, because the wall is not where R10 placed it.
Reading the actual call order in `gw/isdf_fitting.py:434-465`:

    1. _resolved_solver_kind = _resolve_solver_kind(...)      # <-- RAISES here
    2. _resolved_zeta_gather = _resolve_zeta_gather(...)
    3. if _resolved_zeta_gather == 'distributed':
           _resolved_solver_kind = 'distributed_rank_truncate'   # <-- DISCARDS (1)

Step 1 runs `_resolve_solver_kind_charge`, whose `_auto_pre` refuses when the
REPLICATED q-slice `mu^2*16` exceeds `max(ZCAP, 4 GiB)`.  Step 3 then throws
that resolution away and substitutes `distributed_rank_truncate`, whose layout
contract (core.py, "LAYOUT CONTRACT" block) is explicit that **nothing in that
section ever replicates an O(mu^2) object** — C_q, C+, V are all
`P(None,'x','y')`; only `lambda` (nq, mu) is replicated.

**So the capacity check refuses a run on the size of a buffer that route never
allocates.**  Direct proof from this campaign's own logs: rung 7 (mu=24933,
ZCAP=16, job 7879380) and rung 8 (mu=32059, ZCAP=24, job 7879381) both print

    Computing L_q = distributed rank-truncated pinv (2D-sharded C+)
    [PSD, charge channel, path=distributed_rank_truncate]

i.e. **the replicated factor was never used in either run.**  The `ZCAP` lever
"worked" not by making a replicated buffer affordable, but by letting the
resolver get past a check for a route that was then discarded.  R10's
"architectural wall / needs a distributed eigh" framing was WRONG on this
point: the distributed eigh already exists, is already the default at
`distributed_zeta_solve=distributed`, and was already running.

> ⚠ CLAIM-DECAY on R10: the mu=16,384 threshold and its arithmetic are correct,
> and it is correct that the run REFUSES there by default.  But the conclusion
> that it is "architectural ... cannot be escaped by donation, comm reduction
> or sharding, only by implementing a genuinely distributed eigh" is WRONG.
> It is a resolution-ordering defect.  The O(nq*mu^3) redundant-eigh cost I
> attributed to it belongs to the REPLICATED route, which production does not
> take when `distributed_zeta_solve=distributed`.  wk_AP's plan A/B remain the
> right answer for decks that DO use the replicated route (and for mu where one
> (mu,mu) tile stops fitting a rank, ~50k), but they are not needed to move
> THIS frontier.

### R15.2 — FIX (worktree wt-RELC, NOT committed), +3 files, py_compile PASS

- `gw/isdf_fitting.py`: resolve the back-solve TIER FIRST (it has no
  dependency on the factor kind — checked its signature), then pass
  `replicated_factor_used=(tier != 'distributed')` into `_resolve_solver_kind`.
- `isdf/core.py`: thread `replicated_factor_used` through
  `_resolve_solver_kind` -> `_resolve_solver_kind_charge`; when it is False,
  return the nominal `'replicated_rank_truncate'` WITHOUT enforcing the
  replication capacity, so the caller's override at step 3 takes effect.
- The refusal is PRESERVED whenever the replicated factor really is used —
  that is the 2026-07-21 protection (silent fallback returned zeta 4.5x too
  large, V_q relF 16-32 instead of 1.8e-15) and it must not be weakened.

**Expected capacity effect: the mu ceiling for `distributed_zeta_solve=
distributed` moves off 16,384 entirely** — the binding constraint becomes the
2D-sharded objects (mu^2/P per rank) and the r-chunk floor of R8, i.e. hardware.

GATE (job **7879469**): mu=24933, **no ZCAP**, patched source via the new
`SRCDIR` harness override.  Acceptance: (a) it must NOT refuse — the identical
configuration refused in 2m51s as job 7879369; (b) it must print
`path=distributed_rank_truncate`; (c) its eqp must match job 7879380 (same mu,
same everything, ZCAP=16) — same route, so this is a bit-parity expectation.

### R15.3 — ITEM 2: the restart tensor IS written unconditionally, and it is mu^2

MEASURED from this campaign's own run directories:

| mu | `tmp/isdf_tensors_<mu>.h5` | check vs mu^2 |
|---|---|---|
| 6947 | **26.5 GB** | — |
| 10015 | **56.6 GB** | 26.5*(10015/6947)^2 = 55.1 ✓ |
| 15007 | **123.2 GB** | 26.5*(15007/6947)^2 = 123.7 ✓ |

Clean mu^2. Projections on the SAME deck: mu=24933 -> **341 GB**,
mu=32059 -> **564 GB**, and at the axis ceiling mu = n_rtot = 46080 -> **1.17 TB
per run**.  With several ladder rungs in flight concurrently that is multi-TB of
scratch for artifacts nobody reads.

**Answer to the question asked**: it is not optional today.  `gw_init.py:926`
branches on `if not cfg.restart:` — `restart = false` means *"do not READ a
restart; compute fresh AND WRITE the tensors"*.  There is **no config key that
computes fresh without writing** — grep for `write_restart|save_restart|
skip_restart` in `gw_config.py` returns nothing.  (The separate
`save_restart_state_per_proc` path IS already opt-in via
`LORRAX_PER_PROC_RESTART=1` and is not what produces these files.)

RECOMMENDED FIX (not implemented — flagged for the orchestrator as it is a new
input key and touches the restart contract): add `write_restart` (default
`true`, so existing decks are unchanged) and skip the
`write_restart_state_to_h5` call when it is false.  Pure capacity win for
throwaway/scaling runs: at mu=32059 it saves ~564 GB of scratch and the whole
collective write.  The V_qmunu component alone is `nq*mu^2*16` = 263 GB there.
A sharded/streamed variant is a bigger design question; simply not writing it
is the 90% win and is trivially safe because the file is only ever an input to
a LATER run.

### R15.4 — ITEM 3 scoped: teaching the planner about the omega cube

`gflat_memory_model.py` takes `sys = dict(nk, ns, nq, nq_disk, mu, nb, ngkmax,
n_rtot)` — it has neither `n_omega` nor the sigma layout, so it cannot see the
cube at all.  Minimal correct change:
1. plumb `n_omega` and `sigma_omega_layout` into `plan()`;
2. add a stage term `G_sigma_omega = n_omega*nk*nb^2*16` for `replicated`
   (and `/P` for `sharded`), entering the per-stage peak table like the others;
3. because it is a Sigma-stage residency and not an ISDF chunk, it must be
   compared against the budget as its OWN stage rather than folded into
   `C_fit_one_rchunk`;
4. when the replicated term alone exceeds the budget, REFUSE at resolve time
   with a message naming `sigma_omega_layout=sharded` — which is exactly the
   case that HANGS today (D1), so the refusal converts a 20-minute silent hang
   into an instant, actionable error.
Sizes for the refusal threshold, this deck (n_omega=41, nk=16): nb=512 ->
2.75 GB, nb=1024 -> **11.0 GB**, nb=2048 -> **44.0 GB** (would bind before
C_fit at the 45 Ry rung).  NOT implemented in this pass — flagged with the
arithmetic so it can be done deliberately; D1's hang makes it the highest-value
of the remaining planner work.

### R15.5 — CAPACITY-FIX GATE: acceptance (a) and (b) MET (job 7879469)

Disk-verified at 03:05 (NOT from a task notification — notifications in this
environment have been observed fabricated, so every field below is a fresh
read of the log / sacct):

    grep -c ValueError run_CAPFIX_noZCAP_c25000/gw.log   ->  0
    Computing L_q = distributed rank-truncated pinv (2D-sharded C+)
      [PSD, charge channel, path=distributed_rank_truncate]
    sacct 7879469 -> RUNNING 00:04:45, ExitCode 0:0
    ISDF memory model -> HWM estimate 76.45 GB/dev, 2 r-chunks x 35520 r-points

- **(a) it does NOT refuse.**  The byte-identical configuration (mu=24933, no
  ZCAP) refused at **2m51s** as job 7879369.  The patched run passed that mark
  and is 4m45s in and still going.  This is the wall removed.
- **(b) route confirmed** `distributed_rank_truncate` — i.e. it is running the
  route whose capacity was never in question, exactly as the diagnosis said.
- **(c) eqp parity vs job 7879380 (same mu, ZCAP=16)** — pending both runs.
  Expectation is bit-parity, since the two take the identical route and differ
  only in whether a check on an unused buffer fired.
- Planner numbers are identical to the ZCAP run (76.45 GB, r_chunk 35520,
  2 chunks), which independently confirms nothing about the executed plan
  changed — only the refusal.

**CAPACITY DELTA from a 3-file ordering fix**: the mu ceiling for
`distributed_zeta_solve=distributed` moves from the spurious **16,384** to the
next real constraint, which per R8 is the r-chunk performance floor at
**mu ~ 34,600** (predicted HWM 96 GB/rank), with the hard axis ceiling
`mu <= n_rtot = 46,080` behind it.  **~2.1x usable mu, no new numerics.**

NEXT PROBE (queued behind the current fleet): mu=32059 with NO ZCAP on the
patched source, then the c33000 set — that walks the frontier up to the R8
memory wall, which is now the true binding constraint on this axis.

## R16 — ITEM 4: SOFTWARE-CAP SWEEP, with my own disk verification

Systematic sweep of the pinned tree for caps that bind before hardware.  I
VERIFIED the two most decision-relevant findings against this campaign's own
logs rather than taking them on trust; both confirmed, one with an important
nuance the static reading could not have known.

### CAP A — collective-chunk floor: crossed at mu~8k, **13x exceeded, SILENT**

`isdf/core.py:1790,1819`: `_DEFAULT_COLLECTIVE_CHUNK_MB = 128.0` and
`return max(1, min(nq, chunk_bytes // per_q_collective_bytes))`.  Once ONE q's
collective exceeds the bound the `max(1, ...)` floor means the chunker has no
recourse; payload then grows as mu^2 unchecked.  Predicted first crossing
`mu^2*16/p_x > 128 MiB` -> mu > 8,192.

**MEASURED from my own runs** (`max collective/exec` vs `cap 134.2 MB`):

| rung | mu | max collective/exec | vs cap |
|---|---|---|---|
| 4 | 6947 | 97.3 MB | **under** ✓ |
| 5 | 10015 | 926.0 MB | 6.9x over |
| 6 | 15007 | 452.4 MB | 3.4x over |
| 7 | 24933 | **1773.2 MB** | **13.2x over** |

First crossing between mu=6947 and 10015 — exactly the predicted ~8,192.
**Nothing raises; the `[collective chunk]` line prints the violation and the
run continues.**

**NUANCE THE STATIC READ COULD NOT KNOW — and it matters.**  core.py:1783-1786
records 1.15 GB as *measured-fatal* (job 7876062, MaxRSS 10.69 GB against an
85 GB budget).  My rung 7 emits **1.77 GB and survives**.  The difference is
TRANSPORT: 7876062 was Gloo; this campaign runs
`JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` on mlx/RDMA.  So the 128 MB bound is
**stale** — calibrated for a transport production no longer uses, and now
exceeded 13x with no ill effect.  Verdict: a latent hazard and a
mis-calibration, NOT a current wall.  Right fix is to re-derive the bound for
the MPI/mlx transport and make the floor LOUD when it cannot honour it, rather
than to lower mu.

### CAP B — HDF5 4 GiB chunk limit on Sigma(omega): **already violated 2.6x**

`file_io/sigma_output.py:310`: `om_chunks = (n_omega, min(k_chunk, nk), nb, nb2)`
with `k_chunk_size: int = 16` never overridden by either call site
(`gw/ppm_pipeline.py:353,364`).  Chunk bytes = `n_omega*min(16,nk)*nb^2*16`.

**MEASURED**: at my deck's `n_omega=41, nk=16, nb=1024` that is **10.25 GiB**
against HDF5's hard 4 GiB `H5Pset_chunk` refusal — i.e. **already 2.6x over at
nb=1024**.  My runs only survived because the FFI backend NO-OPS `chunks`:
`_slab_io_ffi.py:620-628` warns and ignores it, and rung 5's log carries **128**
of those warnings while writing `sigma_mnk.h5` = 22.6 GB successfully.
`_slab_io_mpi_host.py:377` and `_slab_io_allgather.py:116` DO honour `chunks`.

**Consequence, and it is release-relevant: the same run at the same shape
SUCCEEDS under PHDF5_FFI and REFUSES under PHDF5_HOST / allgather.**  Binds at
nb=1024 for n_omega>=16, and at nb=2048 for n_omega>=4.

### CAP C — full Sigma_c(omega,k,i,j) host buffer: 90 GB/rank at nb~2,929

`gw/ppm_sigma.py:1023-1026` allocates `n_omega*nk*nb^2*16` on EVERY rank with no
rank guard, unless `streaming or sharded_layout`.  The `small_grid` escape at
`ppm_accumulators.py:67` does not help because `_select_accum_mode` returns
KIJ_HOST again whenever `n_proc != 1`.  Same expression as D2's omega cube, now
with a precise bind point: 11 GB at nb=1024, **44 GB at nb=2048**, 90 GB at
**nb ~ 2,929**.  Mirror at `qsgw_utils.py:298-306` forces a full-cube allgather
per device.  Elided only by `sigma_omega_layout=sharded` — which is the same
conclusion D1 reached from the hang, arrived at independently.

### CAP D — H5PY_ALLGATHER replicates the whole tensor: 90 GB/rank at mu~13,258

`_slab_io_allgather.py:72-73` keeps TWO full copies (device gather + host numpy)
of `V_qmunu` on every rank: `2*nq*mu^2*16`.  Auto-selected without any size
check on multi-node runs when mpi4py/h5py-parallel is missing
(`gw_config.py:288-292` unconditionally declines PHDF5_FFI cross-node).
**Silent OOM.**  Not hit here (this campaign runs slab_io=auto -> FFI), but it
would bite any site without parallel h5py at mu>13k.

### CAP E — >2 GB single H5Dwrite for V_qmunu: mu ~ 23,177

`gw/v_q_bispinor.py:336-341` + `_slab_io_mpi_host.py:472-501` issue ONE
`dset.id.write()` per rank for the whole block, no payload chunking.  Per-rank
`nq*mu^2*16/P` = 2 GiB (the classic ROMIO per-call limit) at **mu = 23,177**;
4.9 GB at mu=35,000.  The C++ FFI writer is size_t-clean, so this is a
ROMIO/HDF5 limit that surfaces deep inside H5Dwrite, not a resolve-time refusal.

### CAP F — pivoted-Cholesky 50%-of-basis guard **INTERACTS WITH MY R15.2 FIX**

`centroid/pivoted_cholesky.py:348-357`: refuses when
`max_band > 0.5 * ngk_max * nspinor`.  With the REAL ngkmax (not the 0.06*n_rtot
proxy): 30 Ry ngkmax=1964, nspinor=2 -> ceiling **1964**; 45 Ry ngkmax=3597 ->
ceiling **3597**.  My decks (nb=1024 and 2048) clear it.

**BUT**: my R15.2 fix sets the prune window to the FULL WFN window, so
`max_band = nb_total` where the old clamp gave 52.  A deck with
`nb_total > 0.5*ngkmax*nspinor` — e.g. nb_total=2000 at 30 Ry — would now REFUSE
where it previously (wrongly) succeeded.  **That refusal is CORRECT** (you
cannot ISDF-resolve pair densities of bands beyond half the basis) and the
message is actionable, but it IS a behaviour change my fix introduces and it
must be called out in the merge note.  Recorded here so it is not a surprise.

### Checked and harmless (on the record)

`gflat_memory_model.py:366` r_lo floor (escapable via `r_chunk_size`);
`:327` P_min (warning only, loader_tables ~5 MB); `:83` GFLAT_CHUNK_SIZE_CAP
(throughput only); `_ZETA_GATHER_MAX_BYTES` (tier switch, not refusal);
`htransform.py` Galerkin chunk (bc_cap ~402); `aot_memory.py:398` cufft 2^31
batch (batch ~4,096); `pivoted_cholesky.py:684` int32 pivot sentinel (M <=
n_rtot ~62.5k << 2^30); phdf5 C++ size_t-clean; `zeta_q_G` chunk (binds only at
mu>=71,600 with this deck's ngkmax); `_DENSE_N_MAX=4096` (BSE non-TDA debug
path only).  Divisibility refusals (loud, size-independent) enumerated in the
sweep output.

### Revised capacity picture on the mu axis, after R15.2

    mu  8,192   collective-chunk floor crossed        SILENT, non-fatal on mpi/mlx (stale bound)
    mu 13,258   H5PY_ALLGATHER 90 GB/rank             silent OOM — only if that backend is selected
    mu 16,384   replicated-factor refusal             **REMOVED by R15.2**
    mu 23,177   >2 GB single H5Dwrite (V_qmunu)       deep failure, mpi_host writer
    mu ~34,600  r-chunk performance floor (R8)        the real memory wall
    mu  46,080  n_rtot — hard axis ceiling

## R17 — **THE PRUNE-WINDOW FIX RESTORES THE QP GAP. Defect (A) WAS the cause.**

**Job 7879438 COMPLETED 27:51.**  (nb=1024, mu=10015) — the exact rung-5 point
that produced 0.3645 / -0.3639 eV — rebuilt with ONLY the ISDF prune window
widened (`--prune-n-cond 230`, i.e. `right=(0,256)` instead of `(0,52)`):

| | eqp0 | eqp1 |
|---|---|---|
| rung 5 baseline, prune (0,52) | **0.3645** | **-0.3639** |
| **prune (0,256), everything else identical** | **3.1350** | **3.0710** |
| healthy family (rungs 1-4) | 3.22 - 3.63 | 2.98 - 3.29 |

**The gap is back in the healthy family.**  Controls confirming only the window
changed: same WFN, same N=10000 -> 10015 centroids, same weapons, same
`rcond = 1.0e-08` (read from the run's own `[zeta rank_truncate]` line), same
`n_pad = 10048`; `gw.in` differs only in `centroids_file`.  Bare Sigma_X is
unchanged at -40.5368 vs the baseline's -40.5358 (0.002%), as expected since
exchange was never the problem.

### R17.1 — this CORRECTS the emphasis of R13, and the correction matters

I reported the rank-saturation table (R13) as the mechanism and the owner's
conditioning hypothesis as "strongly supported".  The saturation is REAL and
the arithmetic stands — but it was **not what broke rung 5**.  The decisive
evidence is the pair of numbers I already had and read the wrong way round:

    prune fix moves the FIT-Gram retained rank by +1.4% only (6700 -> 6793)
    prune fix moves the QP gap from 0.3645 eV to 3.1350 eV

**So the operative quantity is WHICH centroids were selected, not HOW MANY
directions the fit Gram retains.**  A +1.4% change in retained rank cannot
explain a 2.8 eV swing; the same 10,015 points, re-chosen against a
representative pair-density block, can and did.  Rank COUNT is not basis
QUALITY — my R13.4 framing ("different objects, both plausibly contribute")
was right that they are independent, and wrong to lead with the fit Gram.

Consequences for what has been written:
- **R12 is the mechanism.  R13 is a real but SEPARATE method limit.**
- `docs/dev/isdf_rank_saturation_and_max_usable_nb.md` must be re-pitched: the
  rank-saturation curve and the "max usable nb per cutoff" rule remain valid
  and worth publishing, but the document currently implies saturation caused
  the failure.  It did not.  §2.1 and §2.2 need their roles swapped, and the
  claim that the ladder "must publish a max usable nb" should be softened to a
  recommendation pending a re-measurement with a CORRECT prune window — the
  saturation numbers in that table were all taken on rank-deficient bases.
- The owner's instruction to park the conditioning physics for a separate pass
  is vindicated for a second reason: the conditioning data on record is
  contaminated by the selection bug and should be re-taken after the fix.

### R17.2 — release-blocker statement, now EARNED

> **ISDF centroid selection silently ignored the deck's conduction window, so
> every GW result with `ncond >> nval` rests on a basis chosen for a different,
> much smaller problem.  On this deck at nb=1024 that produced a QP gap of
> 0.36 eV against a true ~3.1 eV — an error of ~2.8 eV — while passing every
> gate in the suite.**

Fix, gates and the rank refusal that makes it un-repeatable are in R14/R15
(worktree wt-RELC, `wk_REL/docs/patches/wsREL-isdf-window.patch`, NOT committed).
Remaining confirmation in flight: job 7879441 (`--prune-n-cond 998`, the FULL
window) should land at or above 3.1350; if it does, the dose-response is
monotone across (0,52) -> (0,256) -> (0,1024) and the case is closed.

## R18 — **CORRECTION to R16 CAP A: the collective bound is NOT benign.**
## Rungs 7 and 8 HUNG, and D1 is broader than "replicated".

> ⚠ CLAIM-DECAY on R16 CAP A ("a latent hazard and a mis-calibration, NOT a
> current wall") and on R15.5's implied clean run at mu=24933.  I called the
> 128 MB bound stale-and-survivable on the strength of rung 7 PRINTING
> 1773.2 MB and continuing.  It printed that line and then **hung on it**.

Disk-verified 03:12 (fresh reads, not notifications):

| job | mu | sharded? | max collective/exec | outcome | log frozen |
|---|---|---|---|---|---|
| 7878363 | 6947 | no | 642.9 MB | COMPLETED | — |
| 7879295 | 10015 | **yes** | 926.0 MB | COMPLETED | — |
| 7879349 | 15007 | **yes** | 1386.1 MB | COMPLETED | — |
| 7879380 | 24933 | **yes** | **1773.2 MB** | **HUNG** | 10 min, at `W Dyson A-build (GEMM)` |
| 7879381 | 32059 | **yes** | (never printed) | **HUNG** | **59 min**, at `Zeta back-solve tier` |

Both hung runs: `VmRSS = 0.00 GB` on ranks 0 and 33 (processes exited), VmHWM
frozen (62.4 GB rung 8, 71.5 GB rung 7), gw.log mtime frozen, Slurm still
RUNNING.  Both cancelled.

**Two corrections follow.**

1. **D1 is NOT specific to `sigma_omega_layout=replicated`.**  Rungs 7 and 8
   both have `sigma_omega_layout = sharded` in their gw.in (grep-confirmed) and
   hung anyway.  So the earlier release-note statement "sharded is mandatory at
   nb>=1024" is necessary but NOT sufficient — sharded runs hang too, at larger
   mu.  What all four hangs share is `distributed_zeta_solve=distributed` +
   `w_dyson_solver=distributed` at nb=1024, i.e. the DISTRIBUTED collective
   plane, not the omega layout.
2. **There IS a fatal collective-payload ceiling on MPI/mlx** — it is simply
   higher than the Gloo-era 1.15 GB datum in core.py:1783.  Bracketed by this
   campaign: **1386.1 MB completes, 1773.2 MB hangs.**  So the 128 MB
   `_DEFAULT_COLLECTIVE_CHUNK_MB` is indeed mis-calibrated (10x too low), but
   the `max(1, ...)` floor is a REAL wall once one q exceeds whatever the
   transport tolerates, and it fails by HANGING, not by raising.

**FRONTIER CONSEQUENCE — this, not the memory wall, is the certified mu wall.**
    mu 15,007  max collective 1386 MB  -> completes
    mu 24,933  max collective 1773 MB  -> HANGS
so the usable ceiling on this axis at P=64 is **between mu = 15,007 and
24,933**, well below R8's memory wall (~34,600) and below the collective-floor
crossing (~8,192, which is merely where the bound stops being honoured).
Narrowing it needs one run at mu ~ 20,000 (the c20000 set, 19,991, already
built and accepted) — NOT queued tonight; 64 nodes were just returned and the
consolidation is the priority.

CAVEAT, stated plainly: the correlation between payload and hang is strong and
the bracket is clean, but the MECHANISM is not proven — rung 8 hung before
printing any collective line.  What is certain: (a) both runs hung, (b) both
had sharded on, (c) the mu bracket above.  Attributing it specifically to the
collective payload remains a hypothesis until a targeted test.

### R17.3 — DOSE-RESPONSE COMPLETE AND MONOTONE (jobs 7879295/7879438/7879441)

All at nb=1024, mu 10015-10037, same WFN, same N, weapons ON, rcond 1e-8.
**Only the ISDF prune window differs:**

| prune window | selection rank | eqp0 | eqp1 |
|---|---|---|---|
| `(0,52)` old default | 630 / 897 | **0.3645** | **-0.3639** |
| `(0,256)` | 897 / 897 | **3.1350** | **3.0710** |
| `(0,1024)` full | 897 / 897 | **3.7227** | **3.4551** |

**Monotone in window width, and the case is closed**: the QP gap moves from
unphysical (negative eqp1) into the physical 3.1-3.7 eV range purely by
re-selecting the SAME number of centroids against a representative
pair-density block.

HONEST RESIDUAL, stated so it is not over-claimed: the two fixed windows do not
agree with each other (3.1350 vs 3.7227 eqp0, a 0.59 eV spread), and the full
window sits slightly ABOVE the healthy family's trend (nb=256 -> 3.58-3.63,
nb=512 -> 3.22-3.46, so nb=1024 would be expected near 3.0-3.2 by
extrapolation).  The c998 set also has 10037 centroids vs 10015, so it is not a
perfectly controlled pair.  **That residual is a CONVERGENCE question for the
numerical-stability pass, not a correctness one** — the correctness claim needs
only the move off 0.3645/-0.3639, which is unambiguous and monotone.  Which
window is best-converged (and whether `--prune-window vc_x_vc`, which adds cxc
pair densities, is better still) should be settled in that pass.

## R19 — **2x2 COMPLETE.** Axis A fixes it; axis B is CATASTROPHIC.

All cells: nb=1024, mu 10015-10037, n_pad=10048, same WFN, weapons ON.

| cell | prune window | rcond | fit n_keep | eqp0 | eqp1 |
|---|---|---|---|---|---|
| baseline | (0,52) | 1e-8 | 6700 | 0.3645 | -0.3639 |
| **A1** | **(0,256)** | 1e-8 | 6793 | **3.1350** | **3.0710** |
| **A2** | **(0,1024)** | 1e-8 | 6788 | **3.7227** | **3.4551** |
| B1 | (0,52) | **1e-10** | 8290 | **-206.83** | **-1039.84** |
| B2 | (0,52) | **1e-12** | 9461 | **-5049.59** | **-304.20** |

### R19.1 — axis B is not "worse", it is DESTROYED

`zeta_rcond = 1e-10` gives a QP gap of **-206.8 eV**; `1e-12` gives
**-5049.6 eV**.  These are not degraded answers, they are numerical wreckage —
hundreds to thousands of eV on a 2.2 eV DFT gap.  This confirms the
`gw_config.py:617-628` plateau far more violently than its own cited datum
(MoS2 4x4/1204c, MAE 1.4 eV at 1e-10): at THIS size 1e-10 is off by 200 eV.

**My R13/R14.4 reasoning is now refuted from both ends and I want that on the
record without hedging.**  I argued the truncation was "discarding real content
six decades above the f64 noise floor" and proposed sizing rcond against that
floor.  The arithmetic was right; the inference was exactly backwards.
Retaining those modes — 6700 -> 8290 -> 9461, i.e. **+41% more retained
rank** — moves the answer from wrong-by-2.8-eV to wrong-by-5000-eV.  They are
not "real content": they are the over-complete, near-null directions whose
pseudo-inverse amplifies noise by 1/lambda, and truncating them is the CURE.
`zeta_rcond=1e-8` is doing exactly the job it was measured into.

**Corollary, and it is the sharpest statement this campaign produced:**

> Retained rank is not basis quality — in EITHER direction.
> Axis A: +1.4% rank  ->  +2.8 eV of correctness (0.3645 -> 3.1350).
> Axis B: +41% rank   ->  -5000 eV of correctness.
> What matters is WHICH directions the basis spans, never how many.

### R19.2 — the owner's ruling is vindicated twice

Parking the conditioning physics and refusing to let a physics-affecting default
be changed on my noise-floor argument was correct.  Had `zeta_rcond` been
lowered on the strength of R13's arithmetic, the result would have been a
5000 eV error shipped behind a plausible-sounding rationale.  The two guards
that stopped it were (a) the standing rule not to change physics defaults
silently, and (b) reading the config's own recorded measurement history before
proposing a rule.  Both are cheap; both were decisive.

### R19.3 — what is settled and what is not

SETTLED: the rung-5 defect is the prune-window clamp (R12), the fix restores
the gap monotonically (0.3645 -> 3.1350 -> 3.7227), and `zeta_rcond=1e-8` must
not be lowered.
NOT SETTLED, for the numerical-stability pass: the 0.59 eV spread between the
two corrected windows; whether `--prune-window vc_x_vc` (adds cxc pair
densities) is better still; and the rank-saturation curve of R13, every point of
which was taken on a rank-deficient basis and which must be re-measured before
any "max usable nb" rule is published.

## R20 — ⚠⚠ **I WAS WRONG ABOUT THE HANGS. THERE IS NO COLLECTIVE-PAYLOAD WALL.**
## D1, R18 and the mu bracket are ALL VOID — a parsing bug in my own monitoring.

**Root cause of my error, stated plainly.**  My liveness check parsed the
per-rank sampler log with the wrong field index.  The sampler writes

    1785312831 VmHWM: 74961688 kB VmRSS: 25533656 kB
    $1         $2     $3       $4 $5     $6       $7

I read `$5` as the VmRSS VALUE.  `$5` is the literal string `"VmRSS:"`, which
awk evaluates numerically as **0**.  So every run I inspected reported
"VmRSS = 0.00 GB", and I concluded the processes had exited.  **They had not.**

Re-read with the correct field `$6`:

| run | mu | last sample | VmHWM | **VmRSS** |
|---|---|---|---|---|
| DIAG noweap | 10015 | 02:44:57 | 35.9 GB | **16.8 GB** |
| DIAG noshard | 10015 | 02:44:47 | 36.0 GB | **25.3 GB** |
| rung 7 | 24933 | 03:13:51 | 71.5 GB | **24.4 GB** |
| rung 8 | 32059 | 03:14:01 | 62.4 GB | **28.1 GB** |

All four were ALIVE with 17-28 GB resident.  **I cancelled four healthy 32-node
runs.**

**POSITIVE PROOF for rung 7** — its own log, after my scancel landed at
03:14:04:

    [stage 03:15:01]   <- W[probe] Dyson solve (16 q, mu=24933, full BZ)  735.6 s
    [stage 03:15:01] -> W[probe] finiteness + hermiticity gate

It completed a **735.6 s** Dyson solve and advanced to the next stage while
being torn down.  It was never hung — that stage simply emits nothing for
12 minutes, and I read silence as death.

### R20.1 — what this VOIDS

> ⚠ CLAIM-DECAY, total, on:
> * **D1** ("`sigma_omega_layout=replicated` HANGS at nb>=1024") — VOID.  No
>   evidence of a hang.  The two runs were in the post-sigma phase (a
>   ~22 GB `sigma_mnk.h5` write plus QSGW) for 9-20 minutes, which is slow but
>   not pathological, and both were resident and alive when I killed them.
>   **"sharded is mandatory at nb>=1024" is WITHDRAWN** — it was never
>   demonstrated.
>
>   > **RE-ESTABLISHED ON NEW EVIDENCE (DLM campaign, 2026-07-29, agent #3).**
>   > D1's *conclusion* turns out to be right; D1's *reasoning* stays void, and
>   > nothing below inherits from it.  Job **7879688** (mu=10015, nb=1024, P=64,
>   > `sigma_omega_layout=replicated`) finished all four sigma branches at
>   > 09:45:18 and then did nothing for **3 h 21 m 22 s** until the wall clock
>   > killed it (rc=124).  The object is
>   > `module_0978.jit__identity_fn`, an 11,005.85 MB all-gather
>   > `c128[41,16,128,128] -> c128[41,16,1024,1024]` = `n_omega*nk*nb^2*16`.
>   > What makes this a hang and the old runs not: **no `sigma_mnk.h5` was ever
>   > created** (so it is not a slow write — the distinguishing fact D1 lacked),
>   > `module_0978` is the LAST module XLA compiled with zero HLO files written
>   > afterwards, all 64 ranks kept sampling `/proc` every 20 s to within 8 s of
>   > the kill, and VmHWM was frozen at 36.00 GiB throughout (so not an OOM).
>   > The `sharded` twin 7879687 completed the whole run in 1520 s.
>   > "Use sharded at nb>=1024" is therefore reinstated — **as a new claim with
>   > its own evidence**, not as a restoration of D1.  Detail:
>   > `distributed_linalg_largemu_notes.md` §6.3.
> * **R18** ("there IS a fatal collective-payload ceiling on MPI/mlx,
>   bracketed 1386-1773 MB") — VOID.  Built entirely on the false hangs.
> * **The "usable mu bracketed 15,007-24,933" frontier claim** — VOID.
> * My R16 CAP A "correction".  The ORIGINAL R16 reading was closer to right:
>   the 128 MB bound is exceeded (13x) with no observed ill effect.  What is
>   still true is that the bound is exceeded and the `max(1,...)` floor is
>   silent; what is NOT true is that anything died of it.

### R20.2 — what SURVIVES (independent of the parsing bug)

- **The prune-window bug and its fix.**  Evidence is QP gaps and selection
  ranks from COMPLETED runs: 0.3645 -> 3.1350 -> 3.7227 eV, rank 630/897 ->
  897/897.  Untouched.
- **The rcond 2x2** (-206 eV at 1e-10, -5049 eV at 1e-12).  COMPLETED runs.
- **The capacity/premature-refusal fix.**  The refusal is real and observed
  (jobs 7879369 and 7879389 both died in ~3 min with an explicit ValueError);
  the patched runs demonstrably do not refuse.  Untouched.
- **The Sigma reference gate** (|d| = 0.00e+00 eV).  Untouched.
- **The cap sweep's static code findings** (HDF5 4 GiB chunk, H5PY_ALLGATHER
  replication, >2 GB H5Dwrite, planner blindness, restart mu^2).  These are
  code reading plus arithmetic and do not depend on any hang.

### R20.3 — the actual state of the mu frontier

There is **no demonstrated wall between mu=15,007 and mu=32,059.**  Rung 7
(24933) was mid-flight and healthy at 71.5 GB VmHWM; rung 8 (32059) was at
62.4 GB and progressing through the zeta back-solve.  Both were killed by me,
not by the code.  The frontier is therefore **UNDETERMINED above 15,007** and
must be re-established by letting a run finish.  Note rung 7's VmHWM of
71.5 GB against its 76.45 GB planner estimate (ratio 0.935) — consistent with
the model and with NO impending memory wall either.

### R20.4 — process lesson, recorded because it cost four jobs

The rung-5 "all ranks exited" discriminator that I have leaned on twice is only
as good as its parser.  `VmRSS` printed as exactly `0.00 GB` for every rank of
every run should itself have been the tell — a real mass exit does not produce
a perfectly uniform zero across 64 ranks and four independent jobs.  **A
diagnostic that reports the same suspiciously clean value everywhere is more
likely broken than the system it is measuring.**  The sampler script and the
reader are now known-good; `$6` is VmRSS, `$3` is VmHWM.

## R21 — ITEMS 1-3 ANSWERED: there is no wall to localize, but the bound IS
## structurally unenforceable. Precise finding.

**Item 2 (the coordinator's first hypothesis: "are the distributed zeta path's
collectives going through the chunking at all?").  ANSWER: YES, they are.**
`isdf/core.py` has exactly two `_chunk_q(...)` call sites (lines 1918 and 2052),
both on the distributed zeta path, and every log line the campaign produced
carries the `[collective chunk]` announcement from them.  The chunking is
applied and it is working — it turns a 17.7 GB unchunked collective into 10
executions of 1.77 GB.  So the hypothesis is refuted: this is not a
missing-chunking bug.

**What IS true — the bound is structurally unenforceable above a size.**

    def _chunk_q(nq, per_q_collective_bytes) -> int:
        """Largest q-block whose LARGEST single collective fits the budget."""
        return max(1, min(int(nq), _collective_chunk_bytes()
                                   // max(1, int(per_q_collective_bytes))))

The chunker splits **only along q**.  Once ONE q's collective exceeds the
budget, `max(1, ...)` returns 1 and there is no remaining granularity: the
emitted payload is then whatever one q costs, and it grows as `mu^2` with the
bound simply ignored.  Threshold: `mu^2*16/p_x > 128 MiB` -> **mu > 8,192**.

Observed in every run, at every call site, at every size from mu=6947 upward:
**`q_block = 1` already**.  Measured max payloads: 642.9 MB (6947) -> 926.0 MB
(10015) -> 1386.1 MB (15007) -> 1773.2 MB (24933), i.e. up to **13.2x the
134.2 MB cap**.

**Item 1 (localize the wall) and item 3 (name the real limit): THERE IS NO
WALL.**  Per R20 the four "hangs" were my parsing bug; two runs at 1773 MB and
one at 1386 MB were healthy when I killed them, and two more (7879469, 7879487)
are running past that payload right now.  So nothing has been shown to break at
any payload this campaign has produced.  The honest statement is a silent bound
violation and a mis-calibration (128 MB predates the MPI/mlx transport), not a
capacity limit.

**RECOMMENDED (small, safe, doctrine-consistent — not implemented, not asked
for yet):** make the floor LOUD.  The project's standing rule is "announce, and
never silently downgrade"; here the code silently fails to honour a bound it
advertises.  A one-line warning when `_chunk_q` returns 1 AND
`per_q_collective_bytes > budget` would have made this visible from the first
rung instead of being reverse-engineered from a log field.  If a payload wall is
ever actually demonstrated, the structural fix is finer-than-q chunking (split
the mu or G axis inside a q), which is movement-only — but there is no evidence
today that it is needed, and I am not proposing to build it on speculation.

## R22 — HARNESS BUG (mine): the 45 Ry centroid gate used the 30 Ry grid

Job 7879368 ran **1 h 35 min** and rejected **every** centroid set it built
(c13000, c18000, c23000, and three nudge retries).  Rejection signature was
identical each time:

    centroids ...: rows=12979 mod8=3 mod64=51 max_offgrid=4.80e-01 dup=1139
    centroid gate: REJECT

Root cause, disk-verified both sides:

    kmeans (r45 job 7879368):  "Grid: 25x25x100 = 62500 points"
    kmeans (30 Ry job 7879286): "Grid: 24x24x80 = 46080 points"
    wk_REL/probes/b256_verify.py:18    GRID = (24, 24, 80)      # HARD-CODED

The verifier measured 45 Ry centroids against the 30 Ry lattice, so every point
looked ~0.48 cells off-grid and 1139 collided into duplicates.  **The centroid
sets were almost certainly fine; the gate was wrong.**  I reused
`b256_verify.py` in `deck_r45_cent.sbatch` without noticing the grid is not a
campaign constant — the same class of mistake as the VmRSS field index (R20):
a helper carried across a context change without re-checking its assumptions.

FIX: `b256_verify.py` now takes an optional 3rd argument `NX,NY,NZ`
(`centroids <file> 25,25,100`), defaulting to the 30 Ry grid so every existing
call site is unchanged.  Self-tested: the 30 Ry c10000 set still PASSES against
the default (`max_offgrid=8.00e-06, dup=0`).  `deck_r45_cent.sbatch` now passes
`25,25,100`.  Job 7879368 cancelled — justified, since its gate rejected 100% of
outputs by construction — and resubmitted as **7879510**.

Cost of the bug: ~1.6 node-hours of centroid builds discarded, and the 45 Ry GW
lineage blocked for the duration.  Cheap to have caught earlier: the verifier
prints the grid it uses only since this fix; before, it printed only the verdict.

## R23 — APPROVED FIX: the LOUD FLOOR (wt-RELC, +35 lines, py_compile PASS)

`isdf/core.py::_chunk_log`, placed **before** the
`LORRAX_COLLECTIVE_CHUNK_LOG` early-return on purpose — a routine-logging knob
must not be able to silence a bound violation — with its own dedup key so it
cannot be swallowed by the normal per-(where,nq,qb,bytes) dedup:

    if qb <= 1 and per_q_bytes > budget and process_index() == 0:
        print("  [collective chunk] WARNING {where}: cannot honour the payload
               bound — one q alone emits {X} MB against a {B} MB budget ({r}x).
               q is the only split axis, so q_block is already 1 and the bound
               is ABANDONED, not enforced. Payload grows as mu^2 from here
               (unhonourable above mu ~ sqrt(budget*p_x/16), i.e. mu > 8192 at
               128 MiB / p_x=8). No failure has been attributed to this; raise
               LORRAX_COLLECTIVE_CHUNK_MB to silence it honestly, or accept
               the larger payload.")

The threshold arithmetic is in both the message and the comment, along with the
measured history (926 MB at mu=10015, 1386 at 15007, 1773 at 24933, up to
13.2x) and the explicit statement that **no failure has been attributed to the
violation** — so a future reader cannot mistake this warning for a known wall.

### R23.1 — two-sided gate (jobs 7879512 / 7879513), b256 reference, mu=2475

Chosen because it is the cheapest config that exercises the distributed zeta
path (434 s), and because at mu=2475 the per-q collective is
`2475^2*16/8 = 12.25 MB` — comfortably on the SILENT side of the 134.2 MB
default, and forced onto the FIRING side by `LORRAX_COLLECTIVE_CHUNK_MB=1`
(1.0 MB budget) without changing mu, nb, the deck, or any physics input.

    A  7879512  default budget 134.2 MB  vs 12.25 MB  ->  must be SILENT
    B  7879513  forced  budget   1.0 MB  vs 12.25 MB  ->  must FIRE

Both also re-assert the pinned Sigma reference (eqp0 3.5819 / eqp1 3.2516,
tol 1e-3), so the gate proves the warning changes no numbers.
`SRCDIR` and `CCMB` were wired into `gate_sigma_reference.sbatch` for this.

## R24 — HARNESS AUDIT, first finding is my own (category (d), exit masking)

The rank-gate wiring I added in R14.2 was broken in the exact way this audit is
hunting:

    /scratch2/.../centroid_rank_gate.sh "$0.$SLURM_JOB_ID.out" 2>/dev/null || \
      /scratch2/.../centroid_rank_gate.sh "$(ls -t .../*.${SLURM_JOB_ID}.out | head -1)"
    RANKGATE=$?
    echo "[rank gate rc=$RANKGATE]"

Three defects in five lines:
1. `$0` is the SCRIPT path, so the primary argument is
   `deck_b1024_cbig.sbatch.<jobid>.out` — a file that **never exists**. The
   primary path failed on every invocation, silenced by `2>/dev/null`, and the
   gate ran only via its unchecked fallback.
2. If the glob found nothing the helper got an empty argument; the resulting
   non-zero rc was captured but only **echoed**.
3. The script ends `exit 0`, so a FAILING rank gate could not fail the job.
   **An echoed rc that a trailing `exit 0` discards is not a gate.**

FIXED in all four centroid harnesses: resolve the log by jobid (unique) and
REFUSE to guess (`INDETERMINATE`, rc=3, when it cannot be located), keep the rc,
announce loudly on non-pass, and `exit ${RANKGATE:-0}` so the gate can actually
fail the job. `bash -n` PASS on all four.

### R24.1 — the parser that was NOT wrong, and why that matters

The l5/l7 harnesses' own VmHWM summary is **label-keyed**, not positional:

    awk '{for(i=1;i<=NF;i++) if($i=="VmHWM:"){v=$(i+1)+0; ...}}'

It scans for the literal `"VmHWM:"` token and takes the NEXT field, so it is
immune to the column shift that broke my ad-hoc check.  Two consequences:
(a) **every VmHWM number reported from a job's .out this campaign is sound** —
the reported memory series is not in doubt; (b) the fragile parser was the one
I hand-rolled inline, standing next to a correct one I could have reused.  The
lesson is narrower and more useful than "check your fields": **when a harness
already parses a log correctly, call that parser instead of writing a second
one.**

### R23.2 — the gate CONTRADICTED MY PREDICTION, and the gate was right

The first two-sided attempt (jobs 7879512 / 7879513) was designed on my
assumption that the per-q payload is `mu^2*16/Px` at BOTH call sites, giving
12.25 MB at mu=2475 and therefore a silent default side.  The "silent" run
FIRED once.  Reading the emitted line rather than trusting the design:

    C+ formation (pinv):    q_block=10 (1 execution)  124.6 MB   <- under cap
    C+ back-solve (GEMM):   q_block=1  (10 executions) 230.0 MB  <- 1.7x OVER

The two sites have DIFFERENT payload laws.  From core.py's own communication
accounting for the back-solve tier:

    per rank, per q:  (mu^2/Px + mu*r_chunk/Py) * 16 B

so the back-solve is dominated by `mu*r_chunk/Py` — **LINEAR in mu**, not
quadratic.  At r_chunk=46,080 and Py=8 it breaches a 134.2 MB budget above

    mu > budget*Py/(16*r_chunk) = **1,456**

Checked numerically: mu=2475 gives pinv 12.3 MB/q and back-solve 240.3 MB/q
against the measured 230.0 MB — the model matches.

**CONSEQUENCE — a correction to R21/R23 that makes the finding bigger, not
smaller.**  I reported the bound as first violated at mu > 8,192 (i.e. from
rung 5 onward).  That is true only of the pinv site.  The back-solve site
breaches above mu ~ 1,456, which is **below every production mu this project
has ever run**, including AQ rung 0 (mu=4962) and the b256 reference gate
(mu=2475) that is now the pinned Sigma reference.  So the advertised
collective-payload bound has been silently abandoned at that site by
essentially EVERY run in the project's history.  Still: **no failure has ever
been attributed to it** — this remains an honesty fix, not a wall.

FIXES APPLIED to the warning: the message no longer quotes a single global mu
threshold (it was right for one site and wrong by 5.6x for the other); it now
names the per-site growth law and says the back-solve bound is already
unhonourable at the smallest production mu.  The comment carries both formulas
and both thresholds with the measured 2475 numbers as the worked example.

RE-GATED with a design that is correct for the site that actually fires:
    A  job 7879520  CCMB=512 -> 537 MB budget > 230 MB/q  => must be SILENT
    B  job 7879521  default  -> 134 MB budget < 230 MB/q  => must FIRE
Same deck, same mu, same physics; only the budget moves. Both re-assert the
pinned Sigma reference so the warning is proven to change no numbers.
The mis-designed first pair was cancelled — **my prediction was wrong, not the
code**, and this is the third time this session that a measurement assumption,
not the system under test, was the defect.

### R22.1 — grid fix CONFIRMED by the strongest possible evidence

Job 7879510, first attempt, with the corrected verifier:

    centroid gate: verifying against grid 25x25x100
    ACCEPTED target=13000 N=13000 file=centroids_frac_12979_r45_b2048_c13000.txt

**The accepted file has the identical name and count as the one job 7879368
REJECTED** (`centroids_frac_12979_r45_b2048_c13000.txt`, rows=12979).  Same N,
same seed, same WFN, same code — only the verifier's grid changed.  That is
definitive: the centroid sets were always correct and the GATE was the defect.
The 45 Ry lineage is unblocked; 1.6 node-hours were lost to a gate that could
reject for a reason unrelated to what it claimed to test (audit category (c)).

The verifier now PRINTS the grid it used on every invocation, so this class of
mismatch is visible in the log rather than inferred from a rejection pattern.

## R25 — TWO findings from running the rank gate on a live 45 Ry log

### R25.1 — the 45 Ry deck is WORSE than 30 Ry: rank 597 / 1166 = **51.2%**

Job 7879510, `c13000` on the 45 Ry b2048 deck, built with the PINNED source
(i.e. the OLD clamped prune window, since the fix lives only in wt-RELC):

    Pivoted-Cholesky prune: 18062 -> 13000 (target 1166 orbits)
      prune window: v x (v+c)  left=(0,26) right=(0,52)
    After pruning: 12979 centroids (rank=597)

**51.2% of the requested independence**, against 70.2% (630/897) on the 30 Ry
b1024 deck.  The deficiency deepens with nband exactly as R12 predicts: the
clamped window is `26 x 52` on BOTH decks, but the 45 Ry deck's sigma window is
`nval 26 + ncond 2022 = 2048` — twice as wide as b1024's — so the selection
block is half as representative and certifies half as many directions.

This is an INDEPENDENT DECK confirming the release-blocker on a different
cutoff, band count and FFT grid.  It also means **every 45 Ry centroid set this
campaign has built so far is rank-deficient** and must be rebuilt with the
fixed `kmeans_cli` before any 45 Ry GW rung is run.  Nothing has been run on
them, so nothing is invalidated — the fix landed before the lineage did.

### R25.2 — MY RANK GATE HAD A PAIRING BUG (audit category (b)/(c))

Running it on a live multi-attempt log exposed it immediately:

    [rank gate] requested=1630 achieved=597    <- WRONG PAIR

`1630` is the target of the c18000 attempt still IN PROGRESS; `597` is the rank
of the c13000 attempt that COMPLETED.  The first version took `tail -1` of
`"target N orbits"` and `tail -1` of `"After pruning ... rank=N"`
**independently**, and these harnesses retry with nudged N, so a log routinely
holds several attempts — and an in-flight attempt prints its target before the
previous attempt's rank is the last one seen.  The gate then compared two
unrelated numbers.  It happened to still say FAIL here, but it could equally
have said PASS by pairing a small target with a large rank.

**A gate that pairs numbers from different experiments is exactly the "can
PASS or REJECT for a reason unrelated to what it tests" class.**

FIXED: the gate now walks the log in order and commits `target`/`rank`/`window`
together, only on an attempt that actually COMPLETED (i.e. emitted an
`After pruning` line).  Verified three ways after the fix:
    r45 multi-attempt log     -> requested=1166 achieved=597  (51.2%)  FAIL  [correct pair]
    30 Ry defective (7879286) -> requested=897  achieved=630  (70.2%)  FAIL
    30 Ry fixed     (7879432) -> requested=897  achieved=897           PASS

### R23.3 — LOUD-FLOOR GATE: **PASS, two-sided, with physics invariance**

Jobs 7879520 / 7879521, b256 reference deck, mu=2475, identical in every
respect except `LORRAX_COLLECTIVE_CHUNK_MB`:

| side | budget | back-solve per-q | expect | warnings | Sigma reference |
|---|---|---|---|---|---|
| A 7879520 | 537 MB (CCMB=512) | 230.0 MB | SILENT | **0** ✓ | eqp0 3.5819 / eqp1 3.2516, **\|d\| = 0.00e+00 eV** PASS |
| B 7879521 | 134.2 MB (default) | 230.0 MB | FIRE | **1** ✓ | eqp0 3.5819 / eqp1 3.2516, **\|d\| = 0.00e+00 eV** PASS |

So the warning (a) fires exactly when the bound is breached, (b) stays silent
when it is not, and (c) **changes no numbers** — both sides reproduce the pinned
Sigma reference bit-exactly.  An announcement that perturbed the answer would be
worse than the silence it replaces; this proves it does not.

Emitted text verified against the correction (R23.2):

    WARNING C+ back-solve (GEMM): cannot honour the payload bound — one q alone
    emits 230.0 MB against a 134.2 MB budget (1.7x). q is the only split axis,
    so q_block is already 1 and the bound is ABANDONED, not enforced. The per-q
    payload grows with mu at this site (pinv ~ mu^2/Px; back-solve ~
    mu*r_chunk/Py, so the back-solve bound is already unhonourable at the
    smallest production mu). No failure has been attributed to this; raise
    LORRAX_COLLECTIVE_CHUNK_MB to silence it honestly, or accept the larger
    payload.

Mechanically checked: contains no `mu > 8192` (the wrong global threshold, 0
occurrences) and does contain the per-site law `r_chunk/Py` (1 occurrence).

## R26 — THE DEFECT IN ONE LINE: the clamped window CAPS the achievable rank

Rewriting the rank gate to check EVERY attempt (audit finding #4) immediately
produced the cleanest statement of the whole rung-5 story.  Gating the full
`deck_b1024_cbig` log (30 Ry, nb=1024, four targets):

| set | requested | achieved | % of request | % of the 1352 ceiling |
|---|---|---|---|---|
| c10000 | 897 | 630 | 70.2% | 47% |
| c15000 | 1357 | 860 | 63.4% | 64% |
| c20000 | 1824 | 1009 | 55.3% | 75% |
| c25000 | 2289 | 1089 | 47.6% | 81% |
| c30000 | 2769 | 1138 | 41.1% | 84% |

**A `left=(0,26) x right=(0,52)` prune window contains at most `26 x 52 = 1352`
independent pair densities.  The achievable selection rank is therefore
MATHEMATICALLY CAPPED at ~1352, no matter how many centroids are requested.**
The measured ranks asymptote to it (630 -> 860 -> 1009 -> 1089 -> 1138) and the
fractional deficiency deepens monotonically with mu (70% -> 41%), because the
request grows while the ceiling does not.

That single sentence explains every observation the campaign fought over:
- **nb-dependence**: the ceiling is fixed at 1352 while the sigma window grows
  with nb, so the mismatch worsens — and it is worse still at 45 Ry
  (rank 597/1166) where the sigma window is 2048 wide.
- **mu-dependence** (the thing that made it look like a conditioning problem):
  more centroids cannot buy rank once the ceiling binds; they only dilute it.
  Rung 6's partial recovery (0.36 -> 1.43 eV at mu 10015 -> 15007) was the
  *centroid count* improving coverage of a basis that was still capped, not the
  fit conditioning improving.
- **why the fix works instantly**: `(0,256)` offers `26*256 = 6,656` and
  `(0,1024)` offers `26,624` — both far above any request this campaign makes,
  so the full 897 is certified and the QP gap returns (3.1350 / 3.7227 eV).

**Scope consequence, stated plainly: every big-mu set on the 30 Ry b1024 deck
is rank-deficient** (c15000 63%, c20000 55%, c25000 48%, c30000 41%).  Rungs 6-8
and the two in-flight capacity runs all use them.  That does NOT invalidate the
capacity/memory results — VmHWM, planner agreement and the premature-refusal fix
do not depend on basis quality — but **no PHYSICS number from any big-mu rung is
usable**, and none should be quoted.  Rung 6's 1.4296 eV is hereby withdrawn as
a physics datum; it is a symptom, not a measurement.

## R27 — RANK GATE REWRITTEN (audit findings #3, #4, #17)

The first version had three defects, one of them fail-open:

- **#3 FAILED OPEN.** It extracted the target with a hard-coded character
  offset (`substr($0, RSTART+7)` for the literal `"target "`).  Had the wording
  become `target: N orbits`, REQ would have gone non-numeric, FLOOR would have
  collapsed to 0, and `[ "$GOT" -lt 0 ]` is false — **PASS, unconditionally,
  forever**, on the one gate standing between this campaign and another 0.36 eV
  result.  The GOT side happened to fail closed, which is why the asymmetry was
  invisible in testing.
- **#4 GATED ONLY THE LAST ATTEMPT.**  The harnesses ship EVERY accepted set
  from an NLIST; 3 of 4 were ungated while one PASS line was read as covering
  all of them.
- **#17 IN-FLIGHT LOG.**  Callers point it at the job's own still-flushing
  stdout, so the last visible attempt may not be the last real one.

REWRITTEN: no character offsets anywhere (digits are matched, not counted);
non-integers **FAIL CLOSED (exit 4)**; every completed attempt is gated on its
own line with the exit code the worst verdict; the attempt COUNT is printed so
a caller can spot a truncated log; POSIX awk only (no gawk 3-arg `match`),
since it runs both outside and inside the container.

Regression-tested four ways:

    R1 defective 30 Ry (630/897)      -> FAIL  rc=2   ✓
    R2 fixed 30 Ry (897/897)          -> PASS  rc=0   ✓
    R3 multi-attempt (4 targets)      -> all 4 gated, all FAIL, rc=2   ✓
                                         (old gate reported on ONE)
    R4 corrupted target "target: NNN" -> FAIL CLOSED rc=4  ✓
                                         (old gate would have said PASS)

## R28 — I RECREATED THE MOVING-TREE HAZARD I OPENED THE SESSION BY FIXING

At 01:04 I pinned the source (`srcpin_4f77842`) precisely because a shared tree
was being edited under long jobs.  Then, from 03:01, I ran gate and capacity
jobs with `SRCDIR=/work2/.../wt-RELC/src` — a **live worktree I was actively
editing**.  Timeline, disk-verified:

    7879469 started 03:01:08     capacity run, mu=24933
    7879487 started 03:18:28     capacity run, mu=24933
    wt-RELC/src/gw/isdf_fitting.py   mtime 03:14:25   (rebase stash/reset/pop)
    wt-RELC/src/centroid/kmeans_cli.py mtime 03:14:25 (same)
    wt-RELC/src/isdf/core.py         mtime 03:50:39   (the loud floor)

So both capacity runs had their source files rewritten underneath them mid-run.
Same failure class as everything else this audit found: a rule I established,
then violated the moment it was inconvenient.

**DECISIVE TEST that the runs are nonetheless uncontaminated.**  The loud floor
went into `core.py` at 03:50, after both started.  Both runs emit collective
payloads far over budget (1246 MB and 1773 MB), so if they had picked up the
edit they would necessarily have warned:

    7879469  loud-floor warnings = 0      <- started 03:01, pre-edit code
    7879487  loud-floor warnings = 0      <- started 03:18, pre-edit code
    7879521  loud-floor warnings = 1      <- started 03:52, post-edit code

Python's module cache held: each process imported `isdf.core` at startup and
kept it.  The 03:14 rewrite was a clean rebase replay (identical content), and
the 03:50 edit demonstrably did not reach either run.  **The capacity results
stand.**  But that is luck plus module semantics, not discipline — a late or
lazy import would have produced a run built from two different source states,
undetectably.

FIX for everything from here: a FROZEN snapshot, never the live worktree.

    /scratch2/.../wk_REL/srcsnap_isdf_window_0408   (228 py files, == wt-RELC)
    /scratch2/.../wk_REL/snapshots/pointers/CURRENT_SRCSNAP            (pointer file)

Future gate runs use `SRCDIR=$(cat wk_REL/snapshots/pointers/CURRENT_SRCSNAP)/src`.  Editing the
worktree then cannot touch a running job, and re-snapshotting is an explicit,
timestamped act that appears in the record.

**Caveat this forces on the capacity parity claim**: 7879469 and 7879487 were
also intended as a rebase-parity pair, but they read the worktree 17 minutes
apart across a rebase, so they are not a clean bit-parity comparison either.
What they DO establish — and it is what was asked — is that mu=24933 runs
end-to-end with no ZCAP on the patched source, taking
`path=distributed_rank_truncate`, where the identical config refused in 2m51s.
A true bit-parity pair would need two runs from the same frozen snapshot; that
is now possible and was not before.

## R29 — FIX CONFIRMED ON THE SECOND DECK: 45 Ry rank 597/1166 -> 1166/1166

Job 7879527, `c13000` on the 45 Ry b2048 deck, rebuilt with the fixed
`kmeans_cli` (via the new `SRCDIR` hook).  Disk-verified:

    prune window: v x (v+c)  left=(0,26) right=(0,2048)
    G built, shape=(18062, 18062), diag range [1.083e-12, 3.704e-04]
    After pruning: 13027 centroids (rank=1166)
    [rank gate] requested=1166 achieved=1166 (100.0%, floor 1155) -> PASS
    ACCEPTED -> centroids_r45_b2048_c13000.txt

The release blocker is now closed across TWO INDEPENDENT DECKS — different
cutoff, band count, FFT grid, candidate pool and centroid count:

| deck | sigma window | clamped (0,52) | fixed | window ceiling |
|---|---|---|---|---|
| 30 Ry b1024 | 1024 | 630 / 897 = 70.2% | **897 / 897** | 26x52 = 1352 |
| 45 Ry b2048 | 2048 | 597 / 1166 = 51.2% | **1166 / 1166** | 26x52 = 1352 |

Both were capped by the SAME fixed 1352-direction ceiling (R26), and the deeper
deficiency at 45 Ry is exactly what a 2x wider sigma window against an unchanged
ceiling predicts — the mechanism is quantitative, not just directional.

The Gram diagonal minimum moves with it, same signature both decks:

    30 Ry: 7.632e-17 (clamped) -> 4.996e-12 ((0,1024))
    45 Ry: (clamped, near-null) -> 1.083e-12 ((0,2048))

i.e. under the clamped window most candidate centroids sit at numerical zero and
cannot be certified; widen it and they carry real weight.

STATUS OF THE 45 Ry LINEAGE: `centroids_r45_b2048_c13000.txt` is the first
CORRECT centroid set this campaign has produced for that deck (13027 points,
full rank).  The job continues through 18000/23000/28000.  No 45 Ry GW has ever
run, so nothing there needs re-doing — the fix landed before the lineage did.

## R30 — ***THE TERMINAL OOM. CERTIFIED.*** (job 7879469, mu=24933, nb=1024, P=64)

The capacity run died. This is the failure the campaign existed to find, and it
is a genuine one — not a refusal, not a harness artifact, not a hang.

    [gw rc=134 wall=4387s]                 134 = 128+6 = SIGABRT
    sacct 7879469.0: FAILED  ExitCode 6:0  MaxRSS 103,894,092 kB
    MAX VmHWM across ranks   = 99.08 GiB = 106.39 GB
    node usable 186 GiB, 2 ranks/node     =  93.0 GiB/rank available

**CLASSIFICATION: XLA ALLOCATOR OOM.** Not the OOM-killer (no cgroup/slurmstepd
kill line), not a LORRAX refusal (the FAIL-FAST lines are LORRAX *reporting* the
JAX error, which is the designed behaviour), not QE. The allocator said so
verbatim, on 60+ of 64 ranks:

    jax.errors.JaxRuntimeError: INTERNAL: Buffer Definition Event:
    Error dispatching computation: Out of memory allocating 79744204800 bytes.

### R30.1 — THE OBJECT, exactly

    79,744,204,800 B  =  74.27 GiB  =  79.74 GB   in ONE allocation
    /16 (c128)        =  4,984,012,800 elements
    solved:  8 x 24,960^2 x 16  =  79,744,204,800   EXACT MATCH

so the failing object is **`(8, mu_pad, mu_pad)` complex128** — a **q-batch of
EIGHT** of the `(mu, mu)` matrices, with `mu = 24933` padded to
`24960 = 390 x 64`. One rank tried to place a 79.74 GB buffer against ~93 GiB
of headroom while already carrying the rest of the working set.

**SCALING LAW: `q_block * mu_pad^2 * 16` bytes, per rank, NOT divided by P.**
At q_block=8 that is 128 bytes per mu^2. Solving against the ~93 GiB/rank
ceiling gives a wall at **mu ~ 26,900** for this q_block — consistent with
mu=24933 dying once the rest of the working set is added.

Died between `[stage 04:09:55] <- W[probe] finiteness + hermiticity gate` and
04:14:08, i.e. entering the stage after the W probe.

### R30.2 — this is what the removed refusal was blind to

The refusal I removed (R15) gated `nq * mu^2 * 16` on the REPLICATED route.
The object that actually killed the run is a **q-batch of 8 on the DISTRIBUTED
route** — a different quantity the old check never looked at, and one the fix
correctly did not protect (it was never the replicated buffer). So:
- the capacity fix is still right: it removed a check on a buffer that route
  never allocates, and the run got ~35 minutes further into the physics;
- but "the ceiling moves to the R8 memory wall at mu ~ 34,600" was **wrong**.
  The real ceiling on this route is **mu ~ 25,000**, set by a `q_block * mu^2`
  batch that neither the planner nor any refusal models.

> ⚠ CLAIM-DECAY on R15.5's "~2.1x usable mu": the REFUSAL was removed and that
> stands, but the usable-mu claim does not. Measured: mu=24933 now runs 74 min
> and reaches the W stage, then OOMs. Usable mu on this path at P=64 is
> **below 24,933**, not 34,600.

### R30.3 — ⚠⚠ CAMPAIGN-WIDE UNIT ERROR in every predicted/measured ratio

Found while checking the 99.05 figure against the node size. The two reporters
disagree on units and nobody noticed:

    planner:  bytes / 1e9        -> true GB
    harness:  kB / 1048576       -> GiB, but LABELLED "GB"

So every predicted-vs-measured ratio this campaign has quoted compares GB
against GiB and understates the measurement by 7.37%. Corrected:

| rung | (nb, mu) | pred GB | meas GiB | meas GB | TRUE ratio |
|---|---|---|---|---|---|
| 1 | (256, 2475) | 8.61 | 8.91 | 9.57 | **1.111** |
| 2 | (256, 3491) | 11.98 | 12.06 | 12.95 | **1.081** |
| 3 | (512, 4951) | 17.51 | 18.61 | 19.98 | **1.141** |
| 4 | (512, 6947) | 24.36 | 23.60 | 25.34 | **1.040** |
| 5 | (1024, 10015) | 37.67 | 36.03 | 38.69 | **1.027** |
| 6 | (1024, 15007) | 56.49 | 53.40 | 57.34 | **1.015** |

    quoted all campaign (wrong): 0.945 - 1.063
    TRUE:                        1.015 - 1.141

**The model does not bracket reality; it systematically UNDER-predicts by
1.5-14%.** That is the dangerous direction for capacity planning, and it
inherits from the predecessor's rungs 1-4 table, so it is not a regression I
introduced — but I repeated it in every report. Both the R8.2 and R10.3
"pre-registered to 0.07% / 0.25%" claims are also affected: those compared my
predicted GB against the planner's printed GB (both 1e9), so THOSE two remain
valid; it is the predicted-vs-MEASURED comparisons that are wrong.

FIX: the harness's summary line must say GiB, or divide by 1e9/1024 to print
GB. Not changed yet — it is a reporting string in a live harness and I will not
edit it under running jobs (R28).

## R31 — .pyc SNAPSHOT AUDIT (coordinator-propagated) + clean re-snapshot

**MY SNAPSHOTS DID SHIP BYTECODE.** Counts:

    srcsnap_isdf_window_0408/src   121 .pyc
    wt-RELC/src                    121 .pyc
    srcpin_4f77842/src             123 .pyc   <- even the git-archive pin!

The git-archive pin is the notable one: `git archive` emits no bytecode, so
those 123 files were written INTO the "frozen" pin by the container on every
run. **A pin that the runs write back into is not frozen.** 236 of the files
carry the `cpython-312` tag — the container's own interpreter, i.e. exactly the
ones a production run would load.

**BUT: no stale bytecode was executed, and the gate runs stand.** Two checks:
1. All .pyc are TIMESTAMP-mode (flags=0), not `hash-unchecked`. CPython
   validates the embedded source mtime+size before use, so a stale .pyc is
   recompiled rather than run. (The dangerous PEP-552 `unchecked-hash` mode,
   which executes without validating, is absent — checked explicitly.)
2. Content-verified for all three files I edited: each `cpython-312.pyc`
   contains the distinctive constant from its edit —
   `cannot honour the payload bound` (core), `LORRAX_CENTROID_RANK_TOL`
   (kmeans_cli), `replicated_factor_used` (isdf_fitting). Bytecode matches
   source.
   (`grep` on core.py itself returns 0 for that string only because it is split
   across f-string lines — the same artifact that made my earlier verification
   report a false 0. Noted so it is not mistaken for a discrepancy.)

The hazard is nonetheless real for anything built with `cp -a`, which preserves
mtimes on BOTH the .py and the .pyc and therefore keeps a copied .pyc VALID.

### R31.1 — adopted the 10k workstream's tooling rather than writing my own

`wk_REL/srcpin_resolve.sh` provides `srcpin_resolve` (explicit `SRCSNAP`,
rc=90 on unset, manifest verify at start, .pyc coverage warning),
`srcpin_verify_end` (proves immutability across the whole run) and
`srcpin_snapshot` (excludes `__pycache__`, hashes every file).

New snapshot: **`srcsnap_isdfwin_20260729_043303_ec96ba9`**
    .pyc shipped : 0
    files hashed : 341 (all of src/, not just *.py)
    VERIFY.diff  : empty (sha256 of zero bytes) — matches wt-RELC exactly
    resolver     : PASSES, and reports "no cached bytecode in snapshot"

### R31.2 — the residual: I did NOT close it the way it was proposed, and why

The proposal was to copy the resolver into the snapshot "so the manifest covers
it". I tried that by regenerating `MANIFEST.sha256` from the snapshot ROOT — and
it **broke the shared contract**: `srcpin_resolve` does `cd "$SRCSNAP/src"` and
verifies paths relative to `src/`, so a root-rooted manifest made every entry
fail to open. I restored their convention immediately.

Closing it properly requires either putting the resolver inside `src/`
(pollutes the source tree) or changing `cd "$SRCSNAP/src"` to `cd "$SRCSNAP"` in
the SHARED resolver — which other workstreams `source` live, and editing a
shared file that jobs read at startup is precisely the residual being closed.
**I will not unilaterally edit it mid-flight.**

What I did instead, additively and without touching the shared contract:
- the resolver IS copied into the snapshot (`$SNAP/srcpin_resolve.sh`);
- a second, separate `MANIFEST.root.sha256` covers it plus `PROVENANCE.txt` and
  `VERIFY.diff` (3 entries, hashes recorded);
- `MANIFEST.sha256` keeps the 10k convention so every existing consumer works.
A job that wants the fully-closed form sources `$SRCSNAP/srcpin_resolve.sh`
after checking it against `$SRCSNAP/MANIFEST.root.sha256`. Making that the
default needs a coordinated one-line change to the shared resolver — flagged
for the orchestrator, not done here.

## R32 — ***OBJECT LOCATED.*** It is a TEMP ARENA in the GN-PPM fit kernel,
## it IS sharded, and it DOES scale with P. My frontier entry was wrong.

HLO mined from the failing run's own rank-0 dump (389 after_optimizations
modules).  The largest module by far:

    module_0914.jit__gn_ppm_fit_kernel   Total bytes used: 88,621,978,120 (82.54 GiB)

and its allocation table names the killer outright:

    74.27GiB( 90%);  allocation 35: size 74.27GiB, **preallocated-temp**
     2.32GiB;        allocation 1: parameter 1, shape |c128[16,3120,3120]|
     2.32GiB;        allocation 2: parameter 0, shape |c128[16,3120,3120]|
                     outputs: (f64[16,3120,3120], c128[16,3120,3120],
                               pred[16,3120,3120], f64[])

74.27 GiB = **79,744,204,800 bytes exactly** — the failing allocation.

### R32.1 — what this overturns

**`3120 = 24960 / 8 = mu_pad / p_x`.** The parameters and outputs are
**PROPERLY SHARDED 2-D tiles**, 2.32 GiB each. The global unsharded object
would be `c128[16,24960,24960]` = 159.5 GB; nothing of that size is allocated.

> ⚠⚠ CLAIM-DECAY, total, on R30/R30.1/R30.2 and the FRONTIER LEDGER entry:
> I wrote *"79.74 GB is the FULL object, not a shard"*, *"shape
> (n_tau, mu_pad, mu_pad)"*, *"bytes_per_rank = n_tau * mu_pad^2 * 16, NOT
> divided by P"*, and *"adding RANKS will not move it"*. **All four are
> WRONG.** The `(8, mu, mu)` reading was one of three shapes the degenerate
> byte count admitted, and it is not the right one. What is actually there:
>
>     arena = 32 x [one sharded tile]  =  32 * nq * (mu/sqrt(P))^2 * 16
>           =  32 * nq * mu^2 * 16 / P          <-- DOES fall with P
>
>     P=64  -> 74.3 GiB      P=128 -> 37.1 GiB      P=256 -> 18.6 GiB
>
> The measured 32.0x ratio is exact: 79,744,204,800 / (16*3120*3120*16) = 32.0.

### R32.2 — WHY 32 tiles are live: an UNFUSED elementwise chain

Instruction census of the entry computation, counting only ops whose output is
a full `[16,3120,3120]` tile:

    28 broadcast   22 parameter   11 and    9 compare   8 select
     6 subtract     6 multiply     6 fusion  5 real     5 abs
     4 is-finite    1 convert

~111 full-tile-shaped instructions and only **6 fusions**. This is a long
elementwise/select/compare chain (the PPM pole fit with its finiteness and
branch guards) that XLA:CPU has largely NOT fused, so it materialises dozens of
full-tile temporaries and the scratch arena grows to 32 tiles.

**That is a code-generation/placement problem, not an algorithmic requirement.**
An elementwise fit over tiles needs O(few) live temporaries, not 32.

### R32.3 — consequence for the frontier, stated honestly

The wall at mu=24,933 / P=64 is REAL and REPRODUCIBLE (two runs, byte-identical
allocation) — that stands. But it is **a placement artifact, not a hard
architectural limit**:
- it scales as 1/P, so more ranks DO move it (37.1 GiB at P=128);
- it is a temp arena, so fusion or q-chunking of the fit kernel moves it at
  fixed P;
- nothing in the algorithm requires 32 live full-tile temporaries.

So "usable mu < 24,933 at P=64" remains the measured fact, but my
characterisation of WHY was wrong, and the fix space is much larger than I
reported.

### R32.4 — candidate fix, NOT yet proposed as a design

The obvious movement-only candidate is to chunk `_gn_ppm_fit_kernel` over q
(nq=16 -> blocks), which divides the arena proportionally and is the same shape
as the blocked-Gram fix (R6) and the omega-cube fix. Whether the chain can
instead simply be fused is an XLA question I have not investigated. **Per the
standing rule I am reporting the located object first and not building
anything.**

## R33 — q-CHUNKING THE GN-PPM FIT (authorized; movement-only)

### R33.1 — why chunking is bit-exact HERE, by construction

`minimax_screening._gn_ppm_fit_kernel` is, in full:

    denom      = Wc0 - Wc_probe
    safe       = |denom| > 1e-14
    ratio      = where(safe, Wc_probe/denom, 0)
    omega_sq   = -(z^2) * ratio
    good       = safe & isfinite(Re omega_sq) & (Re omega_sq > 0) & mode_mask
    omega_vals = where(mode_mask, where(good, sqrt(Re omega_sq), fallback), 0)
    B_vals     = -0.5 * Wc0 * omega_vals
    -> (omega_vals, B_vals, good, n_good/max(n_modes,1))

**Every operation is elementwise in the leading (q) axis**; `mode_mask` is a
(mu,nu) constant broadcast against it. The only cross-q operations are the two
reductions, and both are sums of BOOLEANS cast to f64 — exact integers
(max ~1e10 << 2^53) — so splitting and re-summing them is associativity-safe.
Chunking therefore changes evaluation ORDER and PLACEMENT only. Single caller
(`ppm_sigma.py:255` -> `fit_gn_ppm_from_wc_pair`), so the kernel's return
signature could be changed safely.

CHANGE: kernel now returns RAW COUNTS `(n_good, n_modes)` instead of the ratio,
and the wrapper does the single division — identical arithmetic on identical
exact integers. Wrapper loops q-blocks, concatenates the three tile outputs.
`q_block >= nq` takes the historical single-shot call **untouched**.

Sizer (`_gn_ppm_fit_q_block`, budget `LORRAX_PPM_FIT_ARENA_GIB`, default 8):

| case | local tile/q | q_block | arena after | arena before |
|---|---|---|---|---|
| mu=24933 P=64 (the OOM) | 148.5 MiB | **1** | **4.64 GiB** | 74.27 GiB |
| mu=10015 P=64 (rung 5) | 25.0 MiB | 10 | 7.81 GiB | 12.50 GiB |
| mu=2475 P=64 (ref gate) | 1.6 MiB | **16 = nq** | 0.78 GiB | 0.78 GiB (single-shot) |

**16x arena reduction at the size that died.**

### R33.2 — ITEM 4: the guards ARE most of the unfused instructions. NOT touched.

From the failing run's own HLO (`module_0914`, after_optimizations):

    top-level full-tile [16,3120,3120] instructions : 111
    fusions                                         :  14 (all kind=kLoop)
    guard-related full-tile ops OUTSIDE any fusion  :  42
        11 and · 9 compare · 8 select · 5 real · 5 abs · 4 is-finite

So **42 of 111 full-tile instructions are the finiteness/branch guards, and
they sit at top level rather than inside the kLoop fusions** — each one
materialising a full tile (a `pred[16,3120,3120]` mask alone is 148.5 MiB).
That is a large part of why the arena is 32 tiles instead of the ~3 the
docstring predicts.

**Every guard is load-bearing and NONE was weakened or removed:**
- `safe` guards a division by a near-zero denominator;
- `isfinite` guards NaN/Inf out of that division;
- `omega_sq_re > 0` selects physically valid poles;
- `mode_mask` kills pad modes at birth — ROOT_CAUSE.md 2026-07-08 records that
  handing pads the live-looking fallback inflated the mode census by a
  device-count-dependent amount;
- `good` is itself a RETURNED output (`valid_qmunu`), so at least one boolean
  tile is mandatory regardless.

**Reported, not restructured**: a guard chain that costs ~30 GiB of scratch is a
design question for the owner (fuse it, compute masks in a lower precision, or
accept the cost), not something to quietly rewrite for memory. Chunking gets the
memory back without touching any of it.

## R34 — INTERMEDIATE REDUCTION in the GN-PPM fit (owner directive, occam's razor)

### R34.0 — the 32x arena multiple is STRUCTURAL, not size-specific

Measured at two very different shapes, from two different runs' rank-0 HLO:

    mu=24933, tile c128[16,3120,3120] = 2.32 GiB -> arena 74.27 GiB = 32.0x
    mu= 2475, tile c128[2,312,312]    = 2.97 MiB -> arena 95.06 MiB = 32.0x

**Exactly 32.0x both times.** The live-temporary count is a property of the
instruction chain, not of the problem size — which is why chunking works
(it shrinks the tile) and why reducing the count helps at every size.

### R34.1 — two reductions, each provably equivalent, guards UNTOUCHED

**Reduction 1 — drop the defensive mask on `ratio`.**
`ratio = where(safe, Wc_probe/denom, 0)` was a full-tile c128 SELECT (2.32 GiB
at production size) doing nothing but masking. It is redundant: `safe` stays
ANDed into `good`, and the only consumer of `ratio` is
`omega_sq -> omega_sq_re -> sqrt`, discarded by `where(good, ...)` on exactly
the lanes where `safe` is false. Case check on a `safe == False` lane:

    old: ratio=0        -> omega_sq_re=0        -> isfinite=T, (0>0)=F -> good=F
    new: ratio=inf/nan  -> omega_sq_re=inf/nan  -> isfinite=F          -> good=F

Both give `good=False`, so `omega_vals`/`B_vals` take the same branch.
**Guard semantics unchanged** — `safe` still gates `good`; only a masked COPY
is no longer materialised.

**Reduction 2 — fold the nested selects and hoist the fallback to 2-D.**
`where(mode_mask, where(good, sqrt, fallback), 0.0)` was a NESTED pair of
full-tile selects. Since `good` already contains `& mode_mask`, the outer
select folds into the inner one's FALSE operand, and that operand then depends
only on `mode_mask` — a `(mu, nu)` array with **no q axis**:

    good=T (=> mode_mask=T) : old sqrt     ; new sqrt                ✓
    good=F, mode_mask=T     : old fallback ; new where(T,fallback,0) ✓
    good=F, mode_mask=F     : old 0.0      ; new where(F,fallback,0) ✓

Saves one full-tile f64 select (1.16 GiB) plus a full-tile broadcast; the
surviving fallback operand is nq times smaller.

### R34.2 — offline equivalence proof BEFORE spending a gate

200 randomised trials with edge cases deliberately injected — `denom` exactly
zero, `denom` 1e-20 (unsafe), `denom` 1e-13 (just safe), sign flips producing
`omega_sq_re < 0`, and pad modes — comparing all FIVE outputs
(`omega_vals`, `B_vals`, `good`, `n_good`, `n_modes`) of old vs new:

    BIT-EXACT MISMATCHES: 0  -> EQUIVALENT

### R34.3 — candidates CONSIDERED AND REJECTED (honest accounting)

- **`|denom| > 1e-14` -> `re^2+im^2 > 1e-28`** (avoids a sqrt). REJECTED:
  mathematically equivalent but NOT bit-equivalent at the boundary — rounding
  can move which elements pass the guard. That is a guard-semantics change,
  which is forbidden. Not worth a different answer to save one op.
- **Merging `isfinite(x) & (x > 0)` into one compare.** REJECTED as not
  actually fewer materialisations: both predicates are needed (`x>0` is already
  False for NaN, but +Inf passes `>0` and must be caught by `isfinite`), and
  rewriting as `(x>0) & (x<inf)` is still two compares plus an and.
- **Computing `Re(-(z^2)·ratio)` without materialising the complex
  `omega_sq`.** REJECTED as byte-neutral: it needs `Re(ratio)` and `Im(ratio)`
  as two f64 tiles (2 x 1.16 GiB) in place of one c128 tile (2.32 GiB), and
  adds ops. No win.
- **Applying a guard to a REDUCED quantity.** NOT AVAILABLE: the guards are
  elementwise per (q, mu, nu) mode and their results are consumed elementwise;
  the only reductions are the two mode counts, which are downstream of `good`.

### R34.4 — CENSUS RESULT: parity PASSES, but the ARENA DID NOT MOVE

Gates 7879674 (single-shot) and 7879675 (forced q_block=1), both against the
pinned reference:

    eqp0 3.5819 vs pinned 3.5819   |d| = 0.00e+00 eV   PASS
    eqp1 3.2516 vs pinned 3.2516   |d| = 0.00e+00 eV   PASS

So the two reductions are **bit-exact in production**, as the offline proof said.

Census at the identical reference shape (`c128[2,312,312]`, tile 2.97 MiB):

| | c128 | f64 | pred | fusions | **arena** | **live tiles** |
|---|---|---|---|---|---|---|
| BEFORE (chunk only) | 61 | 41 | 36 | 13 | **95.06 MiB** | **32.0** |
| AFTER (+2 reductions) | 39 | 27 | 26 | 12 | **95.06 MiB** | **32.0** |

**HONEST NEGATIVE RESULT.** The reductions removed ~46 full-tile instruction
occurrences (c128 -22, f64 -14, pred -10) and the module's total bytes are
byte-identical (110,777,984 both) — **the arena did not shrink at all, and the
live-tile count is still exactly 32.0.** Fusions went 13 -> 12, i.e. XLA did
NOT fuse more; it fused one fewer.

Interpretation: XLA:CPU's buffer assignment peak here is not driven by the
instruction COUNT I reduced. Removing two selects removed work but not any
buffer that was live at the peak. **The `< 10` tile target is NOT met**, and
these two changes should be understood as correctness-neutral tidying that buys
nothing in memory by themselves. They are worth keeping (fewer ops, less work,
provably identical) but they are NOT the fix.

**What DID buy memory is the chunking, and only the chunking** — it shrinks the
UNIT (148.5 MiB -> 148.5 MiB per q at q_block=1, i.e. arena 74.27 -> 4.64 GiB at
mu=24933) while leaving the COUNT at 32. The owner's framing is exactly right:
unit, not count.

## R35 — VERDICTS on the owner's three questions

### Q1: is anything BATCHING that need not be? **NO — confirmed from code.**

`ppm_sigma.py:255` calls the fit ONCE:

    Wc0_q = W0_q - V_q
    Wci_q = Wprobe_q - V_q
    omega_qmunu, b_qmunu, valid_qmunu, unfulfilled = fit_gn_ppm_from_wc_pair(
        Wc0_q, Wci_q, z, fallback_omega=..., n_mu_logical=...)

The kernel's axes are **(q, mu_x, mu_y) and nothing else**. There is NO tau axis:
the GN-PPM fit consumes exactly TWO frequency points — `W(0)` and `W(z_probe)` —
not a quadrature grid. tau lives in the sigma tau-loop, which already processes
one node at a time. **So the leading axis is q, the only batched axis, and it is
now chunkable.** Recording this plainly because it is the natural thing for the
next reader to suspect and it is a dead end.

### Q2: is there a "one unit at a time" option, and does the DEFAULT fit the budget?

Partly. `_gn_ppm_fit_q_block` has floor 1 (one q at a time) and its default
budget (`LORRAX_PPM_FIT_ARENA_GIB`, 8 GiB) does hold at production size:
q_block=1 gives 4.64 GiB <= 8 GiB. **But my sizer has the SAME silent-floor
defect the loud floor was built to announce**: if even ONE q exceeds the budget,
`max(1, ...)` returns 1 and the budget is abandoned with no announcement. At
mu = 24,933 one q costs 32 x 148.5 MiB = 4.64 GiB, so the crossover is near
mu ~ 33,000 — i.e. just beyond the current frontier and reachable on the next
rung. **NOT FIXED** (a code change at wrap-up would ship ungated); logged as
open item O1 with its exact next step.

### Q3: which of the 32 tiles are irreducible? Per-value accounting.

Named full-tile values in the kernel after the reductions:

    INPUTS (2)   Wc0 [c128]              live to the end (B_vals uses it)
                 Wc_probe [c128]         dead after `denom`
    OUTPUTS (3)  omega_vals [f64], B_vals [c128], good [pred]
    INTERMEDIATE denom [c128], abs(denom) [f64], safe [pred], ratio [c128],
                 omega_sq [c128], omega_sq_re [f64], isfinite [pred],
                 (osr>0) [pred], mode_mask-broadcast [pred], 3x and [pred],
                 sqrt(osr) [f64], omega_vals->c128 convert [c128],
                 2x reduction operand [f64]

That is ~20 named values, of which **at most ~6 need to be simultaneously live**
(Wc0, the current intermediate, the running predicate, and the three outputs) —
every other value is consumed by its single successor and could be fused away.
**So of the 32 live tiles, roughly 6 are genuine and ~26 are unfused-op
artefacts.** The 42 guard ops sitting outside the kLoop fusions (11 and,
9 compare, 8 select, 5 real, 5 abs, 4 is-finite) are the concentration to
attack: each materialises a full tile that a fused loop would never write.
Getting under 10 means making XLA:CPU fuse that predicate chain — which the two
source-level reductions demonstrably did NOT achieve, so the next attempt should
target the fusion barrier itself, not the op count.

# =====================================================================
# R36 — SESSION HANDOFF (successor agent #2, 2026-07-29 01:04 - 09:30)
# =====================================================================

## 1. CERTIFIED FRONTIER, as it now stands

**30 Ry MoS2 4x4, nb=1024, P=64 (8x8, 32 nodes): usable mu < 24,933.**
mu=15,007 completes; mu=24,933 OOMs, reproduced BYTE-IDENTICALLY twice
(jobs 7879469 and 7879487, both `Out of memory allocating 79744204800 bytes`,
rc=134/SIGABRT, VmHWM 99.05 and 98.7 GiB against 93.0 GiB/rank available).

**The object** (located in the run's own rank-0 HLO): `module_0914.
jit__gn_ppm_fit_kernel`, `allocation 35: size 74.27GiB, preallocated-temp`.
It is a **temp ARENA**, not an array. The kernel's parameters and outputs are
correctly sharded `c128[16,3120,3120]` tiles (3120 = mu_pad/p_x), 8.27 GiB
total. The arena is **exactly 32.0x one tile**, measured at BOTH mu=24,933
(74.27 GiB / 2.32 GiB) and mu=2,475 (95.06 MiB / 2.97 MiB) — the multiple is
structural, not size-dependent.

    arena = 32 * nq * (mu/sqrt(P))^2 * 16  =  32 * nq * mu^2 * 16 / P

**It DOES scale with P** (74.3 GiB at P=64 -> 37.1 at P=128 -> 18.6 at P=256).
Cause: ~111 full-tile instructions with only 14 kLoop fusions; 42 of them are
guard ops (and/compare/select/real/abs/is-finite) sitting OUTSIDE any fusion.

> This corrects my own earlier entry, which called it a whole-object
> `(n_tau, mu, mu)` materialisation with NO P scaling. All four of those claims
> were wrong; the byte count was degenerate across three shapes and I chose the
> wrong one. The measured wall is real; my explanation of it was not.

## 2. WHAT LANDED (branch wsREL-isdf-window)

| change | gate | result |
|---|---|---|
| ISDF prune window = deck's real sigma window (RELEASE BLOCKER) | QP gap monotone in prune width | 0.3645 -> 3.1350 -> 3.7227 eV; selection rank 630/897 -> 897/897 |
| centroid rank gate, HARD REFUSAL | 4-way regression | FAIL(630/897) / PASS(897/897) / multi-attempt / **FAIL-CLOSED on corrupt input** |
| premature replicated-capacity refusal removed (CAPACITY) | mu=24933 no ZCAP | ran past the 2m51s refusal point, `path=distributed_rank_truncate` |
| loud floor on the collective-payload bound | two-sided | silent at 537 MB budget, fires at 134 MB, both bit-exact on the reference |
| GN-PPM fit q-chunking | reference, single-shot AND forced q_block=1 | **both \|d\| = 0.00e+00 eV**; arena 74.27 -> 4.64 GiB at mu=24,933 |
| 2 intermediate reductions | reference, both paths | **both \|d\| = 0.00e+00 eV**; instructions -46, **arena UNCHANGED** |
| Sigma reference gate (new, permanent) | — | bit-exact across a commit range and weapons on/off |

## 3. THE LIVE-TILE TARGET (< 10), and where it stands

    BEFORE chunking      32.0 tiles x 2.32 GiB (mu=24933)  = 74.27 GiB
    AFTER chunking       32.0 tiles x 148.5 MiB (q_block=1) =  4.64 GiB
    AFTER +2 reductions  32.0 tiles                          = UNCHANGED

**Chunking shrank the UNIT, not the COUNT. The count is still exactly 32 and
the < 10 target is NOT met.** ~6 of the 32 are genuine live values; ~26 are
unfused-op artefacts. The next attempt must target the FUSION BARRIER (the
42 out-of-fusion guard ops), not the instruction count — cutting 46 instructions
moved the arena by zero bytes, which is the strongest evidence available that
op-count is the wrong lever.

## 4. REJECTED CANDIDATES — and the distinction that must survive

- **`|denom| > 1e-14` -> `re^2+im^2 > 1e-28`.** REJECTED. It is
  **mathematically equivalent but NOT bit-equivalent**: rounding can move which
  elements sit either side of the threshold, so the SET of modes marked `safe`
  can change. That is a guard-semantics change. **This distinction is the single
  most important piece of reasoning to carry forward**: for a guard, "same in
  exact arithmetic" is not good enough — only "same set of elements selected, in
  IEEE double" is. Every reduction that DID land was proven on that stricter
  standard (200 randomised trials with denom = 0, 1e-20, 1e-13, sign flips and
  pads; 0 mismatches across all five outputs).
- **Merging `isfinite(x) & (x>0)`.** REJECTED: both are needed (+Inf passes
  `>0`), and the rewrite is not fewer materialisations.
- **Avoiding the complex `omega_sq`.** REJECTED: byte-neutral (2 f64 tiles
  replace 1 c128) and adds ops.
- **Guarding a reduced quantity.** NOT AVAILABLE: guards are elementwise per
  (q, mu, nu); the only reductions are downstream of `good`.

## 5. CAVEAT THAT GOVERNS EVERY BIG-MU NUMBER

**No physics number from any big-mu rung is usable.** Every big-mu centroid set
on the b1024 deck is rank-deficient (c15000 63%, c20000 55%, c25000 48%,
c30000 41%) because the prune window was clamped — fixed, but not re-run at
those sizes. Rung 6's 1.4296 eV is WITHDRAWN as a datum. The OOM and all
memory numbers are unaffected (memory does not depend on basis quality).
Also: **the planner UNDER-predicts by 1.5-14%** (GB-vs-GiB error, corrected;
a planner that reads low is worse than one that reads high).

## 6. OPEN ITEMS, each with its next concrete step

- **O1 — my `_gn_ppm_fit_q_block` has the same silent-floor defect the loud
  floor announces.** If one q exceeds the budget, `max(1,...)` abandons it
  silently. Crossover ~mu 33,000, i.e. the NEXT rung. NEXT STEP: add the same
  announcement `_chunk_log` now carries; gate on the reference (cheap).
- **O2 — get under 10 live tiles.** NEXT STEP: find why the guard chain does not
  fuse into the kLoop (inspect one fusion's boundary in
  `module_0873...after_optimizations.txt`), rather than removing more ops.
- ~~**O3 — the demonstration did not run.** Job 7879672 (mu=24,933, chunked
  path) never left PENDING.~~ **SUPERSEDED — IT RAN, AND IT WORKED**
  (DLM campaign, 2026-07-29, agent #3). 7879672 was scheduled ~2.5 h after this
  handoff was written: 12:07:36 -> 13:21:47, rc=134, run dir
  `run_PPMFIT_demo_c25000`, snapshot `srcsnap_ppmfit_20260729_044438_ec96ba9`
  exactly as prescribed. In its own rank-0 HLO:

      module_0914.jit__gn_ppm_fit_kernel  allocation 36: size 4.64 GiB
                                          (was allocation 35: size 74.27 GiB)

  **Predicted 4.64 GiB, measured 4.64 GiB**, and the 74.27 GiB allocation is
  absent from every dump. Parameters `c128[1,3120,3120]` confirm q_block=1.
  R37's re-reading is confirmed arithmetically too: 74.27/4.64 = 16.00 = nq,
  and `1 * 24960^2 * 8 B = 4.6417 GiB` is exactly the residue of the replicated
  global mask at q_block=1. **The chunking fix is now GATED *and*
  DEMONSTRATED.**
  > Degeneracy trap for the next reader: that same module still contains a
  > `74.27 **MiB**` live-out. Do not match on the number without the unit.

  **The wall moved to a NEW object** — the run got from the zeta fit all the way
  into the fourth sigma branch before dying on
  `module_0958.jit__tau_kernel allocation 21: 25,699,614,720 B = 23.93 GiB`,
  reproduced byte-identically by job 7879690 on a *different* snapshot. That
  object is `sigma (nk, s, mu_X, s', mu_Y)` = `c128[16,2,3120,2,3120]`, measured
  at three mu (4.92 / 15.90 / 23.93 GiB at mu_pad 10048 / 20032 / 24960), and it
  scales as `nk * mu_pad^2 / P`. Full characterisation:
  `distributed_linalg_largemu_notes.md` §6.2.
  > **It is NOT nk-chunkable** — see O7. The PPM analogue does not transfer.
- **O4 — 45 Ry lineage** has one correct centroid set (c13000, rank 1166/1166);
  18000/23000/28000 still need rebuilding with the fixed kmeans_cli. No 45 Ry
  GW has ever run.
- **O5 — 12 sbatch files still use `SRCDIR=`, not the in-snapshot resolver.**
  NEXT STEP: `source "$SRCSNAP/srcpin_resolve.sh"; srcpin_resolve` +
  `srcpin_verify_end`. Never landed — the HLO work took priority.
- **O6 — audit remainder**: `audit_pin.sbatch`/`audit_cpu_gemm.sbatch` still
  carry the backgrounded-srun defect with NO retraction marker;
  `h5_sigma_compare.py` uses `or` not `and`; `colltable.py` treats an empty
  dump dir as PASS. Ranked list in SIZE_CAMPAIGN_BRIEF.md.

- **O7 (NEW, 2026-07-29 agent #3) — the tau-kernel arena is NOT nk-chunkable.
  Do not attempt the PPM analogue here; it is mathematically invalid.**
  The obvious next move after O3 is "do to `jit__tau_kernel` what
  `_gn_ppm_fit_q_block` did to `jit__gn_ppm_fit_kernel`": host-loop the
  leading axis of the 23.93 GiB object. **That axis is `nk`, and it is an
  FFT axis, not a batch axis.** From source:

  * `ppm_tau_kernel._sigma_kij_kernel` docstring:
    `Sigma_kij = project_rs[ FFT[ G(R) . W(R) / sqrt(Nk) ] ]`, *all flat-k*.
  * `common/fft_helpers.make_flat_k_gw_conv` computes
    `sigma = fftn( ifftn(G) * ifftn(W)[:, None, :, None, :] * mult )`
    with the transform taken over the k-grid `(nkx, nky, nkz)`, nk = their
    product = 16 for this 4x4x1 deck.
  * `greens_function_kernel.build_G_tau` returns `(nk, s, mu_X, s, mu_Y)`.

  So the leading 16 is the k index, the kernel is a **convolution over k**, and
  every output k depends on every input k. Splitting it into blocks does not
  reassociate a sum — there is no sum to reassociate — it computes a different
  transform. (This is precisely the property the PPM kernel HAD and this one
  lacks: there the patch note says "every operation in the kernel is elementwise
  in the leading (q) axis". Here it is not.)

  **Also note the intended mitigation is already in and already on.** The fused
  path `make_flat_k_gw_conv` exists specifically to keep the R-space tile from
  materialising ("chunked so the R-space G tile never materializes — the Sigma
  tau kernel's big intermediate"), and `LORRAX_FFT_FFI=1 LORRAX_FFT_FFI_FUSED=1`
  were **active in all three big-mu runs** (handler `lorrax_mklfft_gw_conv`
  announced in each gw.log). What remains in the arena is therefore not FFT
  scratch: it is G(k) and sigma(k) themselves, 9.28 GiB each, plus the f64 Re/Im
  split channels. Those are irreducible under this formulation.

  **The correct lever, if one is wanted:** chunk `(s, mu_X)` — a mu-strip loop.
  The k-FFT is pointwise in that axis, so the convolution stage stays bit-exact;
  but `project_rs` CONTRACTS mu_X, so strip-wise partial sums **reassociate a
  float reduction and are NOT bit-exact**. It would need value-parity gating at
  a stated tolerance, and it touches the psum_scatter payloads that are
  colltable-gated. This is exactly the kernel's own deferred follow-up (a)
  ("m-chunking at add-tau"), and it is a substantially deeper change than the
  PPM one — not a transfer of it. Cheaper first: raise P (the arena is
  `~160 * nk * mu_pad^2 / P` bytes; P=256 gives ~6.5 GiB and clears mu=24,933).

  Secondary observation, not costed: `s = s' = 2` doubles BOTH mu axes, i.e. the
  object is 4x what an `s=1` deck would carry. Whether s=2 is structural here or
  an artifact of how this deck's spinor axis is stored is a physics/deck
  question, deliberately not touched under the capacity-only caveat.

# =====================================================================
# R37 — ***THE ARENA IS ONE BUFFER, AND IT IS GONE.***
#       (successor agent #3, 2026-07-29 19:15-19:40).  Full write-up:
#       wk_REL/docs/ppm_fusion_notes.md
# =====================================================================

> ⚠⚠ CLAIM-DECAY, total, on R32.2 / R33.2 / R34.4 / R35-Q3 / R36 §1 and §3.
> **There is no fusion barrier.  The guard chain fuses completely, and always
> did.**  The census that produced "111 top-level full-tile instructions ...
> 42 guard ops sitting OUTSIDE any fusion" counted instructions INSIDE
> `%fused_computation` bodies as top-level and as unfused.  An op inside a
> fusion body is exactly the op that gets NO buffer.  Measured on the same
> module: the ENTRY computation holds **7** full-tile instructions
> (2 parameters + **5 kLoop fusions**) and **ZERO** unfused full-tile
> elementwise ops; the other 66 are all inside fusion bodies.
>
> R34's negative result was right and its explanation was wrong: cutting 46
> instructions moved zero bytes because **none of them had a buffer.**

## R37.1 — what the 74.27 GiB actually was

ONE allocation, read from the reference run's own memory-usage report:

    allocation 33: size 95.06MiB, preallocated-temp
        3 values; f64[2,2496,2496], f64[2,312,312], f64[]
    (everything else in that arena: 800 B + 312 B + 72 B)

`2496`, not `312`: **UNSHARDED.**  It is the REPLICATED, full-GLOBAL-shape
`f64[nq, mu_pad, mu_pad]` materialisation of the constant mode-count mask —
`minimax_screening.py:571-572`, `jnp.sum(broadcast_to(mode_mask, good.shape)
.astype(f64))`.  `mode_mask` comes from `jnp.arange(n_mu)`, which carries no
sharding, and its consumer is a scalar reduce, so GSPMD kept the branch
replicated: every one of 64 ranks built the entire global mask to add up its
own ones.  (The mask's OTHER consumer, inside `good`, IS sharded — GSPMD
rebuilds it locally from `partition-id`.  Only this branch escaped.)

    reference : 2*2496*2496*8    =        99,680,256 B = 95.06 MiB
    production: 16*24960*24960*8 =    79,744,204,800 B = 74.27 GiB

**79,744,204,800 is byte-for-byte `Out of memory allocating 79744204800
bytes`** (jobs 7879469/7879487).  And the "exactly 32.0x one tile at two very
different sizes" is an IDENTITY, not a temporary count:

    arena/tile = (mu_pad/mu_local)^2 * (8 f64 / 16 c128) = p_x^2/2 = 8^2/2 = 32

It read 32.0 twice because both runs were **P=64 on an 8x8 mesh**.  At P=256 it
would have read 128.0.  The constancy taken as proof of a structural
instruction-chain property was proof of a fixed mesh.

## R37.2 — the fix: the value is a compile-time CONSTANT

`mode_mask` has exactly `n_log**2` true entries, so the sum is exactly
`prod(lead)*n_log**2` (~1e10 at production, << 2**53).  Summing 0.0/1.0 in
float64 is EXACT below 2**53, so emitting the integer is **BIT-IDENTICAL**, not
merely mathematically equal.  **No guard touched** — the deleted code computed
no guard, it counted how many modes a guard could apply to.
+61/-12 in `src/gw/minimax_screening.py`, wt-RELC @ 88010ac, **UNCOMMITTED**.

## R37.3 — LIVE TILES (the owner's metric), both sizes

Compile-only A/B, job 7880762 (1 node, `small`, 3 min): AOT-compile on 64 fake
CPU devices with the production 8x8 mesh and read `memory_analysis()`.
Instrument validated — it reproduces the real dumps' 95.06 MiB and the exact
79,744,204,800 B OOM without running or allocating anything.

| | ref mu=2,475 OLD | NEW | prod mu=24,933 OLD | NEW |
|---|---|---|---|---|
| TEMP (the "arena") | **32.000** | **0.500** | **32.000** | **0.500** |
| TOTAL module | **35.563** | **4.063** | **35.563** | **4.063** |
| absolute TOTAL | 105.65 MiB | 12.07 MiB | **82.54 GiB** | **9.43 GiB** |

kLoop fusions 12->8 (ref) / 13->9 (prod): the four that vanish are the mask
broadcast plus its three reduce wrappers — a subgraph deleted, not fusion
quality changing.  Guard-chain fusion structure is byte-identical before/after.

**The owner's "< 10 such tiles" target is MET: 35.563 -> 4.063, at both sizes.**

## R37.4 — GATES: bit-exact on BOTH paths, and byte-parity beyond the gate

200 randomised offline trials each way (edge cases hit: denom exactly 0,
1e-20, 1e-13, **exactly 1e-14 on the `>` boundary**, sign flips, NaN, +/-Inf,
pads, `n_log==n_mu`, `n_log==1`), all five outputs compared by BIT PATTERN:
**0 mismatches**, in both directions (leg A is the anti-strawman control).

| job | path | eqp0 | eqp1 | verdict |
|---|---|---|---|---|
| 7880764 | default (single-shot) | 3.5819, \|d\|=0.00e+00 | 3.2516, \|d\|=0.00e+00 | **PASS** |
| 7880765 | forced **q_block=1** | 3.5819, \|d\|=0.00e+00 | 3.2516, \|d\|=0.00e+00 | **PASS** |

Forced leg proven to have chunked (its module is `c128[1,312,312]`).  Beyond
the 1e-3 eV gate: `eqp0.dat`, `eqp1.dat`, `eqp_g0w0.dat` and `sigma_diag.dat`
are **BYTE-IDENTICAL** (data md5, timestamp header excluded) across OLD, NEW
single-shot and NEW q_block=1.  End-to-end in the runs' own dumps:
110,777,984 -> **12,655,968** bytes (temp 95.06 MiB -> 1.49 MiB), within 368 B
of the compile-only prediction.

## R37.5 — consequence, stated as a PREDICTION not a result

At mu=24,933 the fit kernel's per-rank footprint goes **82.54 -> 9.43 GiB**
against 93.0 GiB/rank; measured VmHWM at the OOM was 99.05 GiB.  The fit kernel
is no longer the binder.  **But O3 still stands — the mu=24,933 demonstration
has never run.**  Re-measure the wall; do not assume it moved by exactly 73 GiB.
A scan of all 454 modules in the reference dump found **no other instance of
this defect class** (the next-largest temp, `module_0315.jit_fn` 7.05 GiB, is
four properly-SHARDED working buffers).

## R37.6 — instrument defect found (avoided here, still live for others)

`mos2_4x4_test/gate_sigma_reference.sbatch` hardcodes `SRCPIN_COMMIT=4f77842`
in its provenance BANNER while letting `SRCDIR` point anywhere, and its
`inner.sh` never sets `PYTHONDONTWRITEBYTECODE`.  Verified consequence:
`srcsnap_ppmfit2_20260729_090947_ec96ba9` holds **114 `.pyc`** written by the
run that used it — the snapshot mutated at run time and `srcpin_verify_end`
would have passed vacuously over executable bytecode.  `wk_REL/ppmfus_gate.
sbatch` is a fixed copy (explicit `$SRCSNAP`, manifest verified at START and
END, banner derived from the snapshot, `PYTHONDONTWRITEBYTECODE=1`); its
snapshot carries **0 `.pyc`** after two 32-node runs.  Add to the O6 list.

## R37.7 — open, unchanged

O1 (silent floor in `_gn_ppm_fit_q_block`), O3 (the mu=24,933 demonstration),
O4, O5, O6 all stand.  NEW: `_GN_PPM_FIT_LIVE_TILES = 32` is now measured to be
~4, i.e. the sizer over-chunks ~8x.  Left at 32 DELIBERATELY (R5/R30.3: a sizer
reading HIGH is safe, LOW is not); lowering it changes the production chunk
plan and needs its own gate.  The BSE FFT wall-share measurement was NOT taken
— it was the fallback for this task stalling, and it did not stall.
