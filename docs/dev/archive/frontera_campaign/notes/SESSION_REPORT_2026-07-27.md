# LORRAX Frontera — session report 2026-07-27

Branch `fix/zq-band-gather-device-invariance` @ **9f20cb6** (local, not pushed).
Seven commits today: merges of workstreams AK (GW forensics fixes), AI
(collective MPI-IO writer), AL (ib0 Gloo pin), plus the per-proc-restart-dump
gate. Scorecard sections AJ / AK / AI / AL appended by their agents;
11 ⚠ claim-decay banners inserted (every pre-07-27 collective-wall number is
em1-scoped; AF.5's "no dial exists" refuted with mechanism preserved).

## The three headline results

### 1. First complete end-to-end GW at 2406 centroids / P=144
Job **7876530**, 45 m 35 s, rc=0, zero Gloo/OOM events, MaxRSS 17.5 GB/rank
(85 GB budget). **QP gap 2.7271 eV indirect** (VBM K, CBM Λ), min direct
2.8428 eV at K; DFT 1.7010 → +1.026 eV. Same topology as 606c (2.6475 eV),
+80 meV — μ-convergence from below. implied-Vxc guard silent
([−24.262, −4.455] eV over 11 520 (k,n)). `isdf_tensors_2406.h5` complete and
restart-usable (`W0_ready=TRUE`) — Σ-only / BSE work at 2406c can start from
disk. n_keep provenance: AF job 7876346's 1717–1722/q (ζ reused, not re-fit).

### 2. Gloo has been on the 1 GbE management NIC all campaign (AK → AL)
jax 0.9.1 never passes `interface=` to `make_gloo_tcp_collectives`, so CPU
collectives bind the coordinator-route NIC: **em1, 1 GbE, MTU 1500** — not
ib0. `GLOO_SOCKET_IFNAME` is inert (absent from shipped jax). Landed:
`runtime.pin_gloo_interface()` (auto-detect ib*/hsn*, `LORRAX_GLOO_IFNAME`
override/off, announced always, degrades loudly to stock on any failure).

Measured at 606c/P=80 (40 nodes, transport the only variable):
- total recorded **359.5 → 91.8 s (3.9×)**; sigma.exec 288.25 → 62.04 s
  (4.6×); ζ-fit 2023 → 211 s vs July-25 (9.6×); back-solve 17.3×; V_q
  66.3 → 19.2 s. Compile rows ≈ 1.00. eqp/sigma_diag **byte-identical five
  ways** (em1/ib0 × restart/full/cached).
- **The ζ-fit blocker at 606c/P=80 was an em1 saturation artifact** — same
  env as the two failed jobs, pin the only change, rc=0 in 544 s (July-25:
  3193 s). **No c3008f0..eab0dd3 code regression exists.** No bisection.
- Σ τ-loop per-core throughput: 68.3 → **365 MFLOP/s/core = 80% of the
  single-node roofline** (457). The em1 flatline is gone.

### 3. The restart-tensor writer: 1.7 MB/s → 8182 MB/s (AI)
Two root causes, both measured:
- **Shape**: `V_qmunu` (nq,μ,μ) @ P(None,'x','y') is a 2-D tile of a
  contiguous dataset — 3.2 kB innermost runs × 28 800/rank = 4.1 M
  independent-mode writes at P=144. Fix: `H5FD_MPIO_COLLECTIVE` default-on
  (null selections for empty ranks, replica dedup), `LORRAX_PHDF5_COLLECTIVE_WRITES=0`
  reverts.
- **Striping**: `lfs` does not exist in the apptainer image — `_lustre_prestripe`
  was a silent no-op all campaign (production files at stripe count 1, 13 GB
  through one OST). Fix: MPI-IO `striping_factor`/`striping_unit` hints
  (ROMIO applies via llapi, no binary needed); `mode='w'` unlinks first
  (Lustre layout is fixed at inode create; TRUNC reuses the inode).
- Microbench (P=16, production tile): independent 74 MB/s; independent+stripe
  **12 MB/s (striping alone is harmful)**; collective 1313; collective+stripe32
  **2066**. Production: 18.13 GB in 2.2 s. Gates: H5_BITCMP_OK vs independent
  path AND serial oracle; eqp max|Δ| = 1e-6 eV.
- Correction to AF.4c: `psi_full_y` was never written by 7876423 (allocation
  metadata only) — the pathology was ~2× worse than recorded.

## Also landed
- **AK's fixes** (all byte-identical gated P=80/P=6/P=4/COHSEX): screening
  cadence (`_ScreeningCadence` — probe-ω W had neither timer nor print),
  stage rows `chi0_W_probe` + `persist_w0` (7% of GW half, was invisible),
  **per-axis `pad_sigma_window`** (−12.4% Σ at 8×10; avoids 3.16–4.0× latent
  waste on square meshes), `LoopProgress.start()`.
- **AI's telemetry**: `LORRAX_TIMING_TRACE` (every timing.section announces
  enter/exit — covers monolithic-jit stages), per-dataset restart-write rates,
  W0 placeholder labeled "ALLOCATED … no data written" (AC.3b trap).
- **Per-proc restart dump gated OFF** (`88992a9`): `save_restart_state_per_proc`
  has no in-tree reader and cost 4 m 43 s + 72 GB at c2406/P=144 right after
  the canonical 2.2 s write. `LORRAX_PER_PROC_RESTART=1` re-enables.
- **AJ's 4×4 test deck** `/scratch2/08271/jackmc/mos2_4x4_test/`: 30/120 Ry,
  sym on (2 ops + TRS, 10 IBZ), nband=128, WFN 153 MiB (98× smaller),
  centroids 108/402/785, full regen ~4.5 min. H0 vs QE rms 7.8e-5 eV —
  independent re-validation of the Q+S stack on a second ecut and the sym-on
  path. eqp 3.5819 eV (402c) → 3.5867 (785c). Numerics deck (K unsampled).

## The owner's flop invariant (GW ≤ 0.5× ζ-fit)
- **No introduced GW regression** — AK's razor: HEAD GW half is 3% *faster*
  than July-25; AD's sharded W is **5.5× faster** on the Dyson solve;
  ΔSigXC = 0.00e+00 over 10 080 values.
- Screening at 2406c: 27.7 s ≈ 0.01× ζ-fit — inside budget by three orders.
- **ib0-native GW:ζ at 606c/P=80 = 0.53 — marginal FAIL** (em1's 0.225 "pass"
  was itself a transport artifact: ζ speeds up more than Σ on ib0).
  Remaining gap is structural Σ communication (~20% residual τ-loop comm +
  fixed stages). Next levers (AK.9): halve Σ's 4 psum_scatters/τ,
  `_to_host_np` gather, re-price the 128 MB payload bound on ib0.

## Open ledger (carried + new)
1. **c2406/P=144 on ib0**: every P=144 number is em1-scoped; rerun AF/AC
   decks with the pinned tree (`wk_AL/gate40_ib0.sbatch` as template).
2. Perlmutter `hsn*` path coded but never run on real hardware — smoke
   `bootstrap()` once there.
3. Warm-cache run under the pin (`wk_AL/jaxcache_al.7876541`, 347 entries).
4. e2e cold `load_centroids.reshard` = 92.6 s — now the dominant load cost.
5. Σ structural levers (AK.9, above) — the path to GW:ζ ≤ 0.5.
6. U's spinor validation gap + nspin=2; transverse replicated-LU silent
   fallback on non-dividing n_log; dipole.h5 120-band head vs 160-band
   window; cohsex fixture regeneration; cross-node GPU phdf5 FFI
   (Intel-MPI/OFI bring-up); two centroid-driver device_put one-liners.
7. AJ minor defects: kin_ion_io "Hartree folded in" print on stored path;
   `hartree_truncation_2d` attr contradiction; `--qe-save` no-op (QE 7.2
   writes .dat, reader wants .hdf5); QP-WFN dump skipped on sym-reduced decks.

## Owner decisions outstanding
- Push the branch (7 new commits today; ~52 total, all local).
- mpi4py overlay fold-in vs deploying the unified FFI `build_host_W` .so.
- Whether the ib0 pin's default-on behavior is acceptable fleet-wide
  (announced, `LORRAX_GLOO_IFNAME=off` escape hatch; GPU/NCCL untouched).

## Merged-tree verification (post-merge, jobs 7876889 + 7876899, 4×4 deck 785c/P=16)
First full pipelines on the merged tree (9f20cb6), fresh ζ-fits both runs:
- **eqp0 IDENTICAL to AJ's pre-merge baseline** in both runs (allgather route
  and PHDF5_HOST collective route) — physics gate at zero across the full
  merge stack (pin + writer + Σ pad + cadence + per-proc gate).
- All banners live in production: ib0 pin (auto-detected 192.168.47.57),
  `[SlabIO.phdf5_host] collective_write=True dedup_replicas=True
  stripe_count=16`, restart-write telemetry (W0 3010 MB/s), per-proc skip.
- Wall 2 m 52 s on 8 nodes vs AJ's em1 baseline: ζ-fit 87.2→22.6 s, sigma.exec
  157.8→42.7 s, total recorded ≈3.1×.
- **New ledger items found by the exercise**: (a) `slab_io=auto` is inert
  unless `use_ffi_io=true` (gw_config.py:1544) — an "auto" silently gated on a
  second key, pattern-#8 violation; should route unconditionally.
  (b) PHDF5_HOST needs `/opt/intel` bound into apptainer + `srun --mpi=pmi2` +
  `I_MPI_PMI_LIBRARY` (documented in runAC.sbatch, easy to drop in ad-hoc
  harnesses — MPI_Init error 16 is the signature).

## Workstream AM — defaults alignment (merged 19aeece)
Bare input file now gets everything: slab_io=auto routes unconditionally on
both backends (CPU: FFI→HOST→ALLGATHER probed+announced; NEW GPU router with
single-node FFI gate + announced demotion), use_ffi_io deprecated to a
tri-state override (false → forced allgather + DeprecationWarning), C++ writer
collective default flipped to match Python (same env var, same meaning) with
rank-local replica dedup added (collective-mode UB fix), PMI-mismatch now
demotes the auto probe with an actionable message instead of dying.
Gates: 15/15 write bit-compare; 6-cell parse matrix; CPU P=16 minimal-input
eqp0 identical + P=1 sanity; GPU rtx bare-input smoke rc=0 with eqp
GPU-vs-CPU max|Δ|=1e-9 eV; both FFI libs rebuilt (host + CUDA, CUDA gated on
rtx). Portability audit: 6 fixed, rest benign/none-found; Cray collective-
write caution documented with escape hatch (Perlmutter ledger item).
Deprecation inventory (19 items, LIST ONLY) delivered for owner rulings.

## Env-audit round (AT/AU/AV/AW, merged) + bispinor stage (AX, merged) — closed 07-28
- AT (jaxinit): pin engaged on cpu-in-list+no-GPU (closed a silent-em1 hole when
  JAX_PLATFORMS unset); CHECK_REPLICA falsy-parse fixed; cache verdict ENABLE.
- AU (transport): FI_PROVIDER=tcp seed deleted (LORRAX_MPI_PROVIDER case-block);
  FI_PROVIDER_PATH upgraded to REQUIRED in-container; UCX module vars
  setdefault'd (stripping them silently cost 2x on 1MiB allreduce); inert
  GLOO/NCCL ifname exports purged from 9 harnesses; staging binds added to
  production harnesses (auto was still degrading to tcp in-container).
- AV (knobs): COLLECTIVE_CHUNK_MB=128 kept on measurement; ZETA env twins
  deprecated; env_vars.md rebuilt.
- AW (ffi/io): MklThreadScope (28-thread nested MKL was a second independent
  cause of the 30-min pzheevd: 17 s/q -> 0.46 s/q at production shape); ROMIO
  hints to pass-throughs; STRIPE_SIZE_FS registry bug fixed; MULTIPLE-level
  warning guard.
- AX (bispinor): transverse LU silent fallback -> resolve-time refusal;
  bispinor restart round-trip implemented (Sigma^B was silently dropped);
  transverse sets 143/275 (~1/3); bi4/bi16 green, Sigma^B a near-k-rigid
  +0.23-0.25 eV shift, Kramers ~7 meV, restart bit-identical;
  cusolvermp_charge/lu removal UNBLOCKED (channels share one resolver ladder).
Orchestrator assessment of the audit branches: stable and maintainable —
every default change is measured-and-cited, every failure path announces,
env semantics unified across writers, and the doc registry now carries
measured-scope columns. Two watch items: UCX setdefaults are TACC-module
-coupled (revisit on Perlmutter), and the compile-cache flip's first
large-P production outing should be watched once.
