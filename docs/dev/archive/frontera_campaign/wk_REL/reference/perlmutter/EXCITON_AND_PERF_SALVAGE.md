# Exciton bandstructure + performance salvage — Perlmutter archive

**Scope.** Read-only reconnaissance of
`/work2/08271/jackmc/frontera/lorrax_perlmutter_salvage/` (927 dated report files,
Apr–Jul 2026, plus `environment/` and `home_uncommitted/`), for (1) prior working
runs of LORRAX's exciton-bandstructure path, (2) htransform cost, (3) BSE-matvec
cost, (4) measured FFI-vs-native margins, (5) recorded failures.

**Written** 2026-07-30. Nothing in the archive was modified.

**Evidence tags used throughout**

| tag | meaning |
|---|---|
| **[MEASURED]** | a number that appears in a run log, or in a report table sourced from one |
| **[PROSE]** | asserted in narrative with no log line or number preserved in the archive |
| **[PROPOSED]** | design/plan text only; no evidence it was ever run |
| **[RECONSTRUCTED]** | derived arithmetically here from logged quantities; not lifted verbatim |

**Hardware caveat, applies everywhere.** Every timing below is Perlmutter **A100**
(40 GB or 80 GB — stated per run, and it matters), CUDA 12.9, JAX container
`nvcr.io/nvidia/jax:25.04-py3`, one JAX process per GPU. Frontera's Quadro
RTX 5000 is **sm_75, 16 GB, no fast fp64**; this entire path is complex128. Treat
every wall-clock number as a *ratio* and *shape* reference, never an absolute, and
expect fp64-heavy stages (SVD, eigh, the matvec GEMMs) to degrade far more on
sm_75 than bandwidth-bound stages do.

**What the archive physically contains.** Only `.md` / `.log` / `.txt` / `.png`
(+ 13 `.sh`, 6 `.py`, 4 `.patch`, 1 `.slurm`). There are **no `.in` decks, no
`.dat` outputs, no `.npz`, no `.h5` restarts, and no LORRAX source tree**. Every
parameter below is recovered from driver banners and report tables; no exciton
eigenvalue array survives numerically.

---

## 1. Does a working exciton-bandstructure run exist?

**Yes — several, on three different data sets.** `src/bse/exciton_bands.py` ran to
completion and wrote `.dat` + `.png` on at least eight occasions. The module is not
*in* the archive, but Python DeprecationWarnings in the run logs quote it by
absolute path and line number — `.../src/bse/exciton_bands.py:367` in
`reports/bse_multinode_2026-07-20/logs/run16_on.log`, `:386` in
`reports/bse_cublasmp_recon_2026-07-20/logs/exciton_drecon.log`, `:472` cited as a
fix site in `reports/bse_figures_2026-07-20/WORKLOG.md:96`. That is runtime
evidence the file existed and executed, not a design-doc claim.

### 1.1 Inventory of completed runs

| # | report dir | data | Q path | window | n_μ | GPUs | wall | evidence |
|---|---|---|---|---|---|---|---|---|
| **A** | `bse_exciton_converged_2026-07-21` | converged 80 Ry / 12×12 G0W0, **QP energies**, exact exchange | 13-pt **on-grid** Γ-M-K-Γ | 8v8c, fH nb=28 | 2412 | 16 × A100-**80G** | **630.3 s** | report.md:258-266 |
| B | `bse_exciton_smooth_2026-07-21` | same data, `--vq-mode interp` | 39-pt M-Γ-K (off-grid) | 8v8c, fH nb=28 | 2412 | 16 × A100-80G | **1998.8 s** | logs/run_exciton_smooth.log |
| C | `bse_exciton_smooth_2026-07-21` | same, coarse twin | 11-pt M-Γ-K | 8v8c, fH nb=28 | 2412 | 16 × A100-80G | **749.2 s** | logs/run_exciton_interp.log |
| D | `bse_multinode_2026-07-20` | MoS2 12×12, DFT energies | 40-pt Γ-M-K-Γ | 8v8c, fH nb=40 | 640 | 16 × A100-40G | **758.0 / 778.1 s** (head off/on) | WORKLOG.md:142-150 |
| E | `bse_cublasmp_recon_2026-07-20` | same as D, `--distributed-recon on` | 40-pt Γ-M-K-Γ | 8v8c, fH nb=40 | 640 | 16 × A100-40G | **773.7 s** | logs/exciton_drecon.log:508-515 |
| F | "dir10" baseline, quoted by D and E | same as D | 40-pt Γ-M-K-Γ | 8v8c, fH nb=40 | 640 | 4 × A100-40G | **636 s** | WORKLOG.md:142-150 |
| G | `bse_coarse_w_pad_2026-07-20` | MoS2 6×6 smoke, native vs coarse-W | 7-pt G-M-K | 4v4c, fH nb=80 | 640 | 1 × A100 | **58.6 / 63.3 s** | smoke_driver_nb44.log |
| H | `bse_refactor_map_2026-07-15` PHASE2_LOG | MoS2 3×3, first deliverable | 32-pt Γ-M-K-Γ | 4v4c | 640 | 1 × A100 | — | PHASE2_LOG.md:1531-1545 |

Two further exciton runs were **set up but produced no numbers**:
`bse_figures_2026-07-20` (56-pt path, "block-Lanczos solve running",
WORKLOG.md:129) was **HELD** pending a scissor decision; and
`gw_converged_12x12_80ry_2026-07-21` §5 records eight configurations tried and
abandoned before run A fixed the blocker (§1.6).

### 1.2 THE run to reproduce — run A

The only exciton bandstructure in the archive built on a **converged** G0W0, on
**quasiparticle** energies, with **exact uninterpolated** exchange. Full report
copied here as `bse_exciton_converged_2026-07-21.md`.

**Provenance** (report.md:3-10): run dir
`runs/MoS2/08_mos2_exciton_converged_2026-07-21/`; branch
`agent/bse-exciton-converged` off `agent/gw-converged-campaign` @ `5e50b8e`;
Perlmutter jobs `56288029` and `56288782`; 2 × (4 nodes / 16 × A100-**80GB**,
`--constraint="gpu&hbm80g"`). Input = the 80 Ry / 12×12 / n_μ=2412 G0W0 of
`reports/gw_converged_12x12_80ry_2026-07-21` (`00b_lorrax_gw_2400c_ranktrunc`),
**reused unchanged** — no new QE, no new centroids, no new self-energy.

**Parameters that worked** — all [MEASURED], from report.md and from
`bse_exciton_smooth_2026-07-21/logs/run_exciton_smooth.log`, which shares the
identical setup and prints the banner lines run A's own log does not preserve:

| parameter | value | source |
|---|---|---|
| BSE window | **8v8c**, absolute bands 18–34 | report.md:189-191 |
| htransform fH window | **nb = 28** (`nval = ncond = 14`), absolute bands **[12, 40)** | report.md:189-191 |
| guard bands | **6 valence + 6 conduction** | report.md:24-26 |
| k-grid / nk | 12×12×1, **nk = 144** | run_exciton_smooth.log:35 |
| centroids n_μ | **2412** = 208 orbit reps × 12-op recovered D3h, band-range weighted | run_exciton_smooth.log:35 |
| real-space grid nr | **174 960** | run_exciton_smooth.log:35 |
| SVD | `(4032, 4824)` → **rank 4032** (full row rank), σ_max 1.764 | run_exciton_smooth.log:37 |
| ctilde orthogonality | **9.77e-15** | run_exciton_smooth.log:43 |
| band chunk | **7** (5.26 GB/chunk) | run_exciton_smooth.log:35 |
| mesh | **4×4** (px=4, py=4), 16 processes, 1 GPU each | run_exciton_smooth.log:1 |
| Q path | on-grid Γ-M-K-Γ, **6 + 2 + 4 intervals = 13 Q**; Γ, M, K **and Λ=(⅙,⅙)** all land on it | report.md:221-223 |
| exchange | `--vq-mode ongrid` — the stored production tile `V_qmunu[wrap(−Q)]`; **no interpolation error, no mini-BZ head model** | report.md:210-220 |
| energies | `--eqp` on **both** legs; eqp1.dat, n_occ = 26, QP shifts **−9.3220 … +4.3561 eV** | run_exciton_smooth.log:33 |
| scissor | **none** | report.md:31-33 |
| f-transform | a = 0.434282 Ry, n = 3.00, shift = 0.393183 Ry | run_exciton_smooth.log:47 |
| q-list built | `(12,12,1)` → **1872 q-pts** (13 Q) / 5616 (39 Q), **batch = 32** | run_exciton_smooth.log:46 |
| solver | block-Lanczos, one compile for the whole path | report.md:264 |
| device high-water | **≈ 17 GiB** across the whole nb 16→40 sweep | report.md:28-30 |

**Physics** (report.md:226-238): E₁(Γ) = **2.092053 eV**; binding **543.5 meV**
against the converged GW direct gap 2.6356 eV @ K; A–B splitting at Γ 116.4 meV;
global minimum **momentum-indirect at M, 1.955182 eV**, Λ 4.6 meV above; indirect
exciton **136.9 meV** below the bright one; Γ path closure exactly 0.000000 meV on
all 8 branches; Γ A-doublet splitting 0.741 meV (TRS-required degeneracy — the one
genuine symmetry number, since the Γ closure is a determinism check, not an
independent test).

**Two caveats the report states plainly** (report.md:269-286): (i) only the
valley-commensurate Q (Γ, Λ, K, M) are converged sampling points on a 12×12 mesh —
the ~2.27 eV points between them are a **mesh artefact**, not dispersion; (ii) the
543 meV binding is **not k-converged** — monolayer MoS2 keeps rising past 24×24.
Converging it needs a finer BSE grid, *not* a wider fH window.

### 1.3 The band-window capacity rule — the single most reusable result

`bse_exciton_converged_2026-07-21/report.md:107-115` (copied here as
`bse_exciton_window_sweep_table_2026-07-21.md`): a 7-point sweep at fixed
n_μ = 2412 / nk = 144, with the driver's **own** gate
(`bse.exciton_bands.gate_htransform_vs_stored`) imported verbatim rather than
re-implemented. [MEASURED], jobs 56288029/56288782, 16 × A100-80G:

| nb | window | nk·nb | SVD rank | ortho | QP min-sval | recon (QP) | verdict |
|---|---|---|---|---|---|---|---|
| 16 | [18,34) | 2304 | 2304 | 1.2e-14 | 0.1855 | 0.000 | FAIL (no guards) |
| 20 | [16,36) | 2880 | 2880 | 1.2e-14 | 0.9544 | 0.000 | **PASS** |
| 24 | [14,38) | 3456 | 3456 | 1.4e-14 | 0.9544 | 0.000 | **PASS** |
| **28** | **[12,40)** | 4032 | 4032 | 9.8e-15 | 0.9544 | 0.000 | **PASS ← used** |
| 32 | [10,42) | 4608 | **4568** | 8.2e-02 | 0.9542 | 48.824 | FAIL |
| 36 | [8,44) | 5184 | 4716 | 2.3e-01 | 0.9551 | 674.963 | FAIL |
| 40 | [6,46) | 5760 | 4804 | 4.5e-01 | 0.9172 | 925.727 | FAIL |

Three transferable rules:

1. **The capacity bound is `nb < rank(ψ_μ)/nk`, not `nb < nspinor·n_μ/nk`.** The
   nominal rule predicts nb < 33.5, so nb = 32 should pass; **it does not**. The
   measured rank of ψ-at-centroids at rtol 1e-8 is **4570**, not 4824 — the
   D3h-orbit-closed centroid set spans only ~95 % of its nominal column space.
   True ceiling **nb ≤ 31** (report.md:14-22, 164-183). *Orbit closure costs ~5 %
   of capacity; do not spend that margin.* The driver now warns:
   `[warn] ψ-at-centroids is RANK-DEFICIENT: 5184 states vs numerical rank 4716
   (nspinor·n_μ = 4824)` (`logs/window_sweep_main.log:328`).
2. **`ctilde` orthogonality is the discriminator, and `min-sval` is blind to it.**
   min-sval is 0.954 at both nb = 28 (pass) and nb = 32 (fail) while the energy
   metric moves 0.000 → 60.5 meV. Gate on the **energy**, not the subspace overlap
   (report.md:137-151).
3. **Two guard bands minimum.** nb = 16 has exact energies but min-sval 0.0917 —
   and **mesh-dependent** (0.2999 on 2×2, 0.0917 on 4×4). The f-transform sets
   `shift = max_k ε_top`, so the topmost selected state is numerically degenerate
   with fH's exact zero eigenvalues and its eigenvector is arbitrary; the per-band
   overlaps show it (`0.746 0.746 0.641 0.641 0.689 0.689 0.199 0.097`)
   (report.md:153-161).

An **independent** capacity curve at n_μ = 640, printed by the driver itself
(`bse_coarse_w_pad_2026-07-20/smoke_driver_nb44.log:28`): *"MoS2/640c on-grid
|Δε_c|: ~1 meV at nband ≤ 48, 7 meV at 64, ~955 meV at 80. min-sval does not see
this; the on-grid gate does."* A finer scan of the same cliff at
`PHASE2_LOG.md:1886-1889`: nband 34/40/48/64/80 → **0.16 / 1.03 / 1.91 / 7.36 /
955 meV**, with per-band Gram error ‖C_kC_kᴴ−I‖ going 1–2 % (bands 0–33) to **~40 %**
(bands 56–79) even though `rank(α) = 1280 = ns·n_μ` — *"the DOF COUNT is fine …
the sampling/orthonormalization is the wall"*, and *"Owner premise INVERTED: a
larger interp basis is WORSE, not better"* (PHASE2_LOG.md:1893-1900).

> **One superseded claim to be aware of.** `PHASE2_LOG.md:1799-1803` (2026-07-18)
> concluded "the `nb ≤ ns·n_μ/nk` cap was an artifact" on the evidence that
> min-sval was 0.8853 at rank-limited SVD. That is exactly the blind metric of
> rule 2. The 2026-07-21 sweep, which measures **energies**, supersedes it.

### 1.4 Traps that cost prior sessions

* **`nband` in the input is an ABSOLUTE band index.** A gap-centred window
  `[26−nval, 26+ncond)` needs `nband ≥ 26+ncond`. Setting it to `nval+ncond` (the
  natural reading, and what `gwbands.in` does) puts it *below* `b_start`;
  `common/wfn_transforms.load_centroids_band_chunked` zeroes the whole window and
  the SVD of a zero matrix returns rank 0. The parent campaign recorded this as
  "a non-zero `b_start` is separately broken" — **it is not**
  (report.md:78-95). With `nband` correct, every gap-centred window returned
  ctilde orthogonality **1.0–1.4e-14**. [MEASURED]
* **The q=0 head can be silently absent.**
  `bse_coarse_w_pad_2026-07-20/smoke_driver_nb44.log` carries
  `RuntimeWarning: BSE q=0 head NOT injected though G0_mu_nu is present and
  inject_head=True: vhead/whead both unresolved` — "exciton binding energies will
  be under-bound at the zone centre". A run that hits this still completes and
  still plots. Grep the log for `q=0 head injected (rank-1, dual-sharded G0…)`.
* **`initialize_wfns(eqp_file=…)` silently returns DFT energies.** Its
  `htransform.read_eqp_energies` expects a different text format and swallows the
  parse failure with a log line. Run A deliberately routes through
  `bse_io.read_bgw_eqp` instead (report.md:204-208).
* **The QP gate number is a permutation, not an error.** With `--eqp` on,
  `max|Δε_c|` is **57.902 meV — bit-identical at nb = 16, 20, 24 and 28**. A fixed
  number that does not move with the window is a *sorting* mismatch (htransform
  returns ascending QP order, stored `eps_c` is DFT-index order), not interpolation
  error. Read `recon` (which sorts both sides, and is 0.000) beside it
  (report.md:125-134).

### 1.5 What the archive is silent on

* **No input deck survives.** The Q path is a QE-style `K_POINTS crystal_b` block
  inside `cohsex.in`, parsed by `read_lorrax_input` into
  `params["kpoints_crystal_b"]` as `nseg` lines of `kx ky kz [n] [# label]`
  (`gw_refactor_map_2026-07-01/archive/files/gw_config.md:22`);
  `bse.exciton_bands` reads the same block the htransform bandstructure driver
  consumes (`PHASE2_LOG.md:1431`). The block never affects the GW k-grid, which
  comes from WFN.h5 (`FLAGS.md:227,393`). I have [RECONSTRUCTED] the 13-, 40-, 39-
  and 139-point paths from logged node indices plus per-Q |Q| columns, in
  `qpath_mos2_GMKG_crystal_b.txt`. The 32-pt and 56-pt paths are not
  reconstructable.
* **No launcher script survives.** `run11.sh`, `run_exciton_16gpu.sh`,
  `run_exciton.sh`, `run_sweep.sh`, `window_sweep.py`, `plot_exciton_bands.py` are
  all *named* in the reports; none is in the archive. The launch model exists in
  prose only (§2.4).
* **No numeric exciton output survives** — no `.dat`, no `.npz`. Only PNGs.
* **Runs B and C were never adjudicated.** `bse_exciton_smooth_2026-07-21/` has
  **no `report.md`** — logs and one diagnostic PNG only. Timing evidence, not
  physics evidence.

### 1.6 A contradiction between two same-day reports — resolved

`gw_converged_12x12_80ry_2026-07-21/report.md:33-38` states stage 5 "**produced no
numbers**", that the BSE would need ~5680 centroids whose replicated `fH_R` is
69 GiB/device, and that "accuracy and memory are mutually exclusive at 80 Ry".
`bse_exciton_converged_2026-07-21/report.md:27-30`, same day, on a branch derived
from it, states this "**no longer binds**" and runs the whole nb 16→40 sweep at a
~17 GiB high-water mark.

**The later report supersedes**, and its three fixes are named and each is
memory/layout only, physics-identical (report.md:44-76) — see §2.5. The nb = 16
gate reproduced 0.000 meV / min-sval 0.0917 **bit-for-bit** before and after.
**Quote the later report.**

---

## 2. htransform — the dominant cost at production n_μ

### 2.1 The measured cost, by regime

| run | nk | nb | n_μ | nr | nQ | GPUs | `htransform_psi_cQ` | **s / Q** | share |
|---|---|---|---|---|---|---|---|---|---|
| A | 144 | 28 | 2412 | 174 960 | 13 | 16 × A100-80G | **323.8 s** | **24.9** | 51 % of 630 s |
| B | 144 | 28 | 2412 | 174 960 | 39 | 16 × A100-80G | **957.4 s** | **24.5** | 48 % of 1999 s |
| C | 144 | 28 | 2412 | 174 960 | 11 | 16 × A100-80G | **278.1 s** | **25.3** | 37 % of 749 s |
| D | 144 | 40 | 640 | 46 080 | 40 | 16 × A100-40G | **15.5 s** | **0.39** | 2 % of 758 s |
| F | 144 | 40 | 640 | 46 080 | 40 | 4 × A100-40G | **155 s** | **3.9** | 24 % of 636 s |

Sources: A `bse_exciton_converged_2026-07-21/report.md:258-266`; B, C
`bse_exciton_smooth_2026-07-21/logs/run_exciton_{smooth,interp}.log` timing blocks;
D, F `bse_multinode_2026-07-20/WORKLOG.md:142-150` and `logs/run16_off.log:457-464`.
All [MEASURED].

**Correction to the "htransform is the dominant cost" premise — it dominates, but
only by ~1.16×, and only at large n_μ.** The three converged runs give
htransform : solve = 323.8 : 279.6, 957.4 : 833.3, 278.1 : 238.0 — a consistent
**1.15–1.17×**. It is the largest single stage, not a landslide. And the ordering
**reverses** at n_μ = 640, where neither is dominant and `vq_prepare` eats 53 % of
wall (398 of 758 s, §4.1 row 1).

**Scaling in n_μ is steep and the archive does not model it.** Per-Q htransform
cost goes **0.39 → 24.7 s ≈ 63×** for a 3.8× rise in n_μ (640 → 2412) at identical
nk and nQ and a *smaller* band window (40 → 28). nr rises 3.8× too, so the naive
`n_μ · nr` product rises only 14×. Confounds: the two runs also differ in card
(40 GB vs 80 GB) and in nb. **Plan on super-quadratic, do not fit an exponent to
two points.** The dominant bundle scales cleanly though —
`psi_rmu_Y = (5616, 8, 2, 2412)` vs `(5760, 8, 2, 640)` is a 3.77× byte increase
(`run_exciton_smooth.log:78`, `run16_off.log:190`).

**Consistency check:** A, B and C are the same setup at three path lengths and give
24.9 / 24.5 / 25.3 s per Q — cleanly linear in nQ with negligible fixed cost.
Per-Q is the right unit.

### 2.2 The Galerkin setup is cheap — the cost is all per-Q

Cross-run table, all [MEASURED] from the driver's own sub-timers:

| source | GPUs / mesh | nk, nb, n_μ | load_centroids | SVD (shape→rank) | G accum | Chol | **Total Galerkin** |
|---|---|---|---|---|---|---|---|
| `run_exciton_smooth.log:35-44` | 16 / 4×4 (80G) | 144, 28, 2412 | 4.86 s | (4032,4824)→4032, 4.22 s | 4 chunks, 2.80 s | 0.50 | **13.55 s** |
| `window_sweep_main.log` nb=16 | 16 / 4×4 (80G) | 144, 16, 2412 | 5.40 s | (2304,4824)→2304, 1.98 s | 3 chunks, 2.31 s | 0.31 | **11.17 s** |
| `window_sweep_main.log` nb=36 | 16 / 4×4 (80G) | 144, 36, 2412 | 6.80 s | (5184,4824)→4716, 6.22 s | 6 chunks, 4.37 s | 0.59 | **20.04 s** |
| `run16_off.log:114-184` | 16 / 4×4 (40G) | 144, 40, 640 | 4.04 s | (5760,1280)→1280, 1.37 s | 1 chunk, 1.36 s | 0.36 | **8.10 s** |
| `ht_lean.log:1-13` | 1 / 1×1 | 36, 90, 1496 | 2.46 s | (3240,2992)→2852, 2.32 s | 2 chunks, 2.36 s | 0.28 | **8.34 s** |
| `smoke_driver_nb44.log:17-25` | 1 / 1×1 | 36, 80, 640 | 2.39 s | (2880,1280)→1280, 1.23 s | — | 0.30 | **6.28 s** |

The build is **6–20 s everywhere** and `htransform_setup` (which contains it) is
6–17 s in every driver run — i.e. ~2 % of the stage at production scale. Note
`load_centroids_band_chunked` is **co-equal with the SVD**, not a minor prologue.

> **Variance warning.** `gw_conduction_postfix_2026-07-21/post/ht_bmax.log` runs the
> *identical* shape as `ht_lean.log` (nk=36, nb=90, n_μ=1496, 1 GPU) and reports
> **17.93 s vs 8.34 s** — a **2.15× run-to-run spread** on host/FS variance alone.
> Treat any single Galerkin wall as ±2×.

### 2.3 What made htransform faster — the 2026-07-18 perf pass [MEASURED]

`PHASE2_LOG.md:1571-1666` (branch `agent/bse-bands-perf`, 1-node 4 × A100-40G
interactive alloc **56095430**, 12×12 nw config, 3-pt smoke, warm FS cache). Owner
trigger: *"it does to my intuition seem to be really slow"*.

| stage | before | after |
|---|---:|---:|
| vq_prepare | 52.9 s | **9.7 s** |
| htransform_psi_cQ | 13.0 s | **6.0 s** |
| solve_scan warm | 13.9 s | 13.9 s (unchanged — matvec-bound by design) |
| **TOTAL (3-pt)** | **124 s** | **63 s** |

Named commits:

* **`18f5cfb` — the single biggest htransform win.** `compute_wfns_fi`'s q-batch
  was **replicated on all 4 GPUs** (`PHASE2_LOG.md:1605`: "nQ·nk rank-1152 eighs,
  28 ms each, batch 32 — replicated"). Applying the `_kpath_batch` batch-sharding
  idiom `P(('x','y'))`: **911 → 346 ms per 32-q batch**; ψ_cQ **12.8 → 4.8 s per
  3 Q** (PHASE2_LOG.md:1633).
* **`b352f64`** — driver conduction stacks as ONE jitted pad+reshard, killing a
  ~1.7 GB host round-trip; Γ-gate slices before `device_get`. `htransform_psi_cQ`
  tick **13.0 → 6.0 s** (PHASE2_LOG.md:1636).
* **`809b0fb`** — vq trainer q-batched on device, ngkmax zero-pad (**≤2 compiles,
  was 15**), q axis sharded in chunks of 48, one host round-trip per chunk, and
  `eigh_backend=auto` = **batched native**, which "kills the single-process
  cusolverMp trap". `prepare_coarse` **17.5 → 4.3 s** (PHASE2_LOG.md:1616). *Caveat:
  bundled commit — the BEFORE split at :1586 shows "eigh only 3.3" of the 17.5, so
  the backend swap is not the dominant term of that 4×.*
* **`58e140e`** — XHX gate on device, k-chunk-accumulated Gram; `run_gates`
  **18.9 → 2.4 s**, zero remat warnings.
* Measured 40-pt end-to-end after: **432 s** on 4 × A100. Non-solve stages
  **253 → 66 s (3.8×)**, values bit-identical at every printed digit.

**Two levers measured, wired, and deliberately left off:**
`--skip-rerun-check` — the diagnostic warm re-scan is **42 % of production wall**;
and `--max-iter 20` — Ritz drift at bs=8, n_flat=2304 is
`max|evs(80)−evs(40)| = 0.000000 meV`, `max|evs(40)−evs(20)| ≤ 0.0004 meV`, so it
halves the solve **below print precision**. Default left at 40 by owner call.
Together: 40-pt ≈ **155 s** vs ≈ 435 s pre-fix (PHASE2_LOG.md:1620-1662). The
2026-07-21 runs did adopt `--skip-rerun-check` ("warm re-run check SKIPPED"); they
did **not** adopt `--max-iter 20`.

### 2.4 What made htransform faster — multi-node [MEASURED, confounded]

`bse_multinode_2026-07-20/WORKLOG.md:142-152`, MoS2 12×12 / n_μ = 640 / 40 Q:
`htransform_psi_cQ` **155 s (4 GPU) → 15.5 s (16 GPU)** = **10× on 4× the
devices**. ⚠ The 4-GPU number predates the §2.3 fixes landing in that lineage, so
device count and the perf pass are **conflated** in this table. Do not quote 10× as
pure scaling.

Launch model (WORKLOG.md:13-21, NOTES:19-26):
`srun -N4 -n16 --gres=gpu:4 select_gpu.sh <shifter> in_container.sh python3 -u -m
bse.exciton_bands … --px 4 --py 4 --eigh-backend cusolvermp`, with `select_gpu.sh`
setting `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`. Explicitly **NOT**
`--gpus-per-task=1` (breaks JAX distributed topology sync). Square mesh required
because cuSolverMp requires p == q.

### 2.5 Known htransform bottlenecks and hard limits

* **`fH_R` replication was the wall.** `bandstructure/bse_setup.py:156` did
  `jax.device_put(fH_R, NamedSharding(mesh_xy, P()))` — `(nk, ns·n_μ, ns·n_μ)`
  complex128 on **every** device, and because the source is sharded JAX routed it
  through `x._value`, a host gather of the same size *per process*
  (report.md:50-56; `gw_converged…/report.md:432-441`). Quadratic in n_μ and
  **mesh-independent**, so no device count resolves it: 102 GiB/device at
  nk = 144, nb = 48, n_μ > 3456. The pre-fix OOMs, exact
  (`bse_exciton_converged_2026-07-21/logs/hi_wrap.log`, 16 × A100-80G):
  nb=32 `fH_R(replicated) = 44.77 GiB/device` → OOM **96 153 404 288 bytes**;
  nb=36 47.72 GiB/dev → OOM 102 484 943 744; nb=40 49.52 GiB/dev → OOM
  106 345 341 824. `passA_wrap.log` OOM'd at exactly **37 456 183 296 bytes**
  (34.88 GiB = the nb=28 `fH_R` column). **Fixed** — `fH_R` now stays
  `P(None,'x','y')`; the q-Fourier sum contracts over the unsharded R axis so each
  device builds its own (i,j) tile with zero communication, and one all-to-all onto
  the q axis precedes the eigh. **Verify this fix is in your tree before trusting
  any wide-window run.**
* **Unannotated einsum outputs.** XLA will not partition a contraction whose
  *output* is unannotated even when its operands are sharded. `build_fH_R`
  materialised the whole `(nk, rank, rank)` product on every device — **57.84 GiB
  at nb = 20**, with an explicit `Can't reduce memory use below … by
  rematerialization` — and `_q_batch` did the same for `(bs, rank, rank)`
  (9 × 11.4 GiB at nb = 36). Directly observable in
  `logs/diag_wrap.log` (`compute_wfns_fi raised XlaRuntimeError:
  RESOURCE_EXHAUSTED … 23887872256 bytes`) and
  `bse_exciton_smooth_2026-07-21/logs/run01_OOM_PR_unsharded_alloc.log`
  (`only reduced to 49.93GiB`).
* **A live, unquantified SPMD remat in the production log.**
  `bse_multinode_2026-07-20/logs/run16_off.log` carries, **32 times**:
  `[spmd] Involuntary full rematerialization. The compiler was not able to go from
  sharding {devices=[16,1,1,1]<=[16]} to {devices=[1,1,1,4,4]<=[4,4]T(1,0)
  last_tile_dim_replicate} … for HLO operation: %param = c128[360,8,2,640] …
  metadata={op_name="psi"}. You probably want to enrich the sharding annotations.`
  That is the conduction ψ bundle (360 = 5760/16 q-rows/device) being resharded by
  rematerialization between the q-sharded and μ-sharded layouts. **No report in the
  archive quantifies or fixes it.** [MEASURED — the warning; NEVER-QUANTIFIED — its
  cost.] The same run also shows the rematerializer achieving essentially nothing:
  `Can't reduce memory use below 22.16GiB … only reduced to 23.73GiB, down from
  23.73GiB originally`.
* **SVD rank must be mesh-aligned.** A rank-deficient window returns an arbitrary
  integer rank and the first `device_put` dies with `ValueError: … should be
  divisible by 4, but it is equal to 4570`. Now rounded down to
  `lcm(mesh.x, mesh.y)` (report.md:68-75).
* **Band divisibility on a 16-device mesh.** `wfn_transforms.gflat_to_rmu`
  band-flat-shards `P(None,('x','y'),…)` and so required `nb % mesh.size == 0`; the
  Galerkin entry passes an un-rounded window (nb = 40, divides 4 but not 16) →
  hard error. Fixed by zero-band padding + trim (WORKLOG.md:67-76).
* **n_μ padding vs the SVD reshape.** `load_centroids_band_chunked` pads n_μ to a
  multiple of the device count (**1496 → 1504 on 16 GPU**) but
  `streaming_galerkin_solve` reshaped with the true n_μ → crash for any centroid
  count not divisible by the device count. Fixed at
  `src/bandstructure/htransform.py:108` (`psi_rmu_Y[..., :n_mu]`),
  `bse_figures_2026-07-20/WORKLOG.md:101-105`, CONFIRMED empirically.
* **`gflat_to_rmu` chunk inflation.** `cs = chunk_size or N` was not clamped to the
  row count, then zero-padded N → ⌈N/cs⌉·cs. When the HBM-budget cs exceeds N —
  every small-problem Galerkin — the FFT box inflates by cs/N: at the 3×3 nb = 80
  refit, **N = 720 padded to 6103 (8.5×), an 8.4 GB box / 16.76 GiB fused alloc for
  ~1 GB of data**. One-line fix `cs = min(cs, N)`, commit `2e90edb`
  (PHASE2_LOG.md:1670-1679).
* **Galerkin band chunk was hard-coded 64.** Caused a **51.6 GB single allocation**
  OOM at n_μ = 2412 / nb = 80. Sizing it to the ψ box instead (`5e50b8e`) "is what
  made stage 5 runnable at all" (`gw_converged…/report.md:292-307`). Visible after
  as `band_chunk=7 (5.26 GB/chunk)`.
* **Window-boundary degeneracy — the "Λ-valley dip" was an artefact.** At 12×12
  Kramers pairs (even, odd) are exactly degenerate and an fH window boundary that
  **cuts a pair** fails at the eV scale off-grid. Between-pair boundary min-gaps
  range 2194 meV down to **5.9 meV** (the 31|32 boundary). D_min A/B across all
  5760 k+Q, driver window (24,32) vs a clean window: on-grid rows exact, median
  9.7 meV, **max 316.6 meV at iQ 9** — present **only** in the driver-window curve.
  The delivered exciton E₁(Q) tracked the artefact
  (PHASE2_LOG.md:1683-1706). [MEASURED]
* **Basis size does not smooth the bands.** 640c vs 1000c on the identical 40-pt
  path: per-state |ΔE| median 9.7 / mean 11.0 / **max 46.5 meV**, and on the
  artefact rows the shift is the *same* ~10 meV — not ISDF error
  (PHASE2_LOG.md:1727-1740).
* **`run_gates` OOM at 1496 centroids** — 58 GB replicated alloc; env opt-out
  `LORRAX_SKIP_VQ_GATES=1` added (`bse_figures_2026-07-20/WORKLOG.md:114-116`).
* **Declined buffer donations** appear throughout
  (`complex128[144,40,2,11520]`, `[36,64,2,46080]`, …). The analogous BSE audit
  proved the class is **cosmetic — no copy emitted** (JOINT_FINDINGS.md:107); the
  htransform path was never separately checked.

### 2.6 htransform path cost is set by nQ×nk, NOT by window width [MEASURED]

`scissor_farband_htransform_2026-07-20/ht_lean.log` (copied here). 1 GPU, mesh
(1×1), MoS2 6×6, nk = 36, nb = 90, nr = 46 080, n_μ = 1496, 139-point Γ-M-K-Γ path
(nodes [0, 50, 80, 138]). Per-window walls across the sweep:

| window | [0,2) | [0,12) | [0,14) | [0,26) | [0,40) | [0,50) | [0,90) |
|---|---|---|---|---|---|---|---|
| wall | 80.7 s | 76.8 s | 76.8 s | 83.1 s | 86.7 s | 77.5 s | 86.2 s |

**Essentially flat.** Widening the interp window is nearly free in time and
catastrophic in accuracy: on-grid QP recon goes 3.3 meV at [0,2) → **325 038 meV**
at [0,40) → **348 818 meV** at [0,50). `ctilde[0] orthogonality error: 2.440e-01`
at build time. This is a *far-band QP* study, and its lesson is that far-from-gap
QP corrections destroy the interpolation long before the DFT metrics notice —
`DFTrec` at [0,40) is 508.5 meV while `QPrec` is 325 038 meV. Total sweep 802.4 s.

---

## 3. BSE matvec — the second cost centre

### 3.1 Measured solve cost

| run | n_μ | nk | window | nQ | GPUs | `solve_scan` cold | ms / Q |
|---|---|---|---|---|---|---|---|
| A | 2412 | 144 | 8v8c | 13 | 16 × A100-80G | **279.6 s** | 21 500 |
| B | 2412 | 144 | 8v8c | 39 | 16 × A100-80G | **833.3 s** | 21 366 |
| C | 2412 | 144 | 8v8c | 11 | 16 × A100-80G | **238.0 s** | 21 637 |
| D | 640 | 144 | 8v8c | 40 | 16 × A100-40G | **163.1 s** cold / 162.9 warm | 4 073 |
| E | 640 | 144 | 8v8c | 40 | 16 × A100-40G | **160.9 s** / 161.1 | 4 027 |
| F | 640 | 144 | 8v8c | 40 | 4 × A100-40G | **202 s** / 199 | 5 050 |
| — | 1000 | 144 | — | 40 | 4 × A100-40G | 825.2 s | 20 700 |
| G | 640 | 36 | 4v4c | 7 | 1 × A100 | 32.6 s | 4 663 |

Sources: A report.md:258-266; B, C `run_exciton_{smooth,interp}.log`; D
`run16_on.log:451`; E `exciton_drecon.log:502`; F, 1000c PHASE2_LOG.md:1770,1901;
G `smoke_driver_nb44.log:50`. All [MEASURED].

Two conclusions. **(i) The solve scales badly with device count**: 4 → 16 GPUs buys
only 202 → 163 s (**1.24×**), against the htransform's nominal 10×. **(ii) It
scales ~5.3× with n_μ 640 → 2412** — steep, but far gentler than the htransform's
~63×. So the htransform overtakes the matvec at production n_μ; the matvec (and
`vq_prepare`) dominate at fixture n_μ.

Solve temp memory (driver's own `memory_analysis`) is **~flat in nQ but ~7× in
n_μ** — 985.6 MiB (n_μ=640, 40 Q) → 7173.4 MiB (2412, 11 Q) → 8362.8 MiB (2412,
39 Q). Consistent with the μ_loc·ν_loc T-tensor being the driver.

> **Do not use the pre-merge 4-GPU walls.** A 4v4c run at the same grid gave
> 14.08 s/Q vs 5.05 s/Q for a *larger* 8v8c window on the same hardware; the report
> attributes it to environment — "container host-BLAS", srun without
> `--cpus-per-task` (PHASE2_LOG.md:1837-1840). The container numpy BLAS is
> effectively **serial, ~3 GFLOPS**; host stages are thread-starved without
> `--cpus-per-task` + `OMP_NUM_THREADS`.

### 3.2 Attribution — the solve is 100 % matvec, and the matvec is bandwidth-bound

`PHASE2_LOG.md:1584-1624` (`probe_solve.py`, 4 GPU, 12×12) [MEASURED]:

* solve_scan warm 4.65 s/Q = 118 ms fixed + **115 ms/iteration × 40**;
* **bare stack matvec (bs = 8) warm = 123 ms/call ≈ the whole per-iteration cost**
  → the scan solve is 100 % matvec; QR / full-reorth / α overhead unmeasurable;
* the T tensor `(μ_loc, ν_loc, ns, ns, nk)` is **944 MB/device/trial** at 12×12,
  ~7 HBM round-trips ≈ **4.3 ms/trial floor**; measured **15.4 ms/trial ≈ 3.6×
  floor** including 2×2 collectives;
* recorded verdict: **"No solver-side lever left (c64 stays owner-vetoed)"**; "the
  solve's 115 ms/iteration is structural (bandwidth + ns²·nk T-tensor)".

### 3.3 The three-agent matvec efficiency audit — the deepest profiling in the archive

`bse_refactor_map_2026-07-15/archive/matvec_efficiency_audit/JOINT_FINDINGS.md`
(copied here as `bse_matvec_efficiency_audit_JOINT_FINDINGS_2026-07-16.md`, with
`trace_dossier.md` + 26 raw xprof logs beside it in the archive). Perlmutter
4 × A100, job 56010372, source HEAD `6ca714b`. All [MEASURED].

**The thesis** (:20-24): *"The BSE matvec is **never compute-bound** — it is
HBM-bandwidth-bound at production size (stack inflated 1×1: arithmetic intensity
**0.039 FLOP/byte**, warm 15.1 ms ≈ 2.5× the HBM-traffic floor) and
collective-latency-bound at fixture size. So the levers rank by **bytes moved and
collectives launched**, not FLOPs."* `bytes_accessed = 8.96 GB` vs
`flops = 349 MFLOP`.

**Device category split** (:88-95), stack inflated 1×1 nt1, 14.07 ms/call:

| category | % device wall | ms | note |
|---|---:|---:|---|
| gemm | 35.3 % | 4.97 | **bandwidth**-bound (349 MFLOP total) — reads/writes 655 MB operands |
| fft | 28.4 % | 4.00 | IFFT+FFT pair, algorithmically required |
| elementwise/fusion | 19.3 % | 2.71 | W_R×T_R multiply + norms + V-term |
| **copy/transpose** | **17.0 %** | **2.39** | the two T-transposes = 2.14 ms = **15.2 % of wall** |

At fixture size copy/transpose is the **#1 category at 36.7 %**. The transposes are
real ops inside kInput fusions, **not free bitcasts**, and are *structurally
irreducible*: the encode/decode are batched GEMMs over k while **cuFFT requires k
as the three minor axes**.

**Collectives are count-bound, not bandwidth-bound** (:148-160): overlap is already
on — measured XLA flag A/B (stack inflated 2×2 nt8) **default 33.93 ms, `=true`
38.70 ms (no-op), `=false` 43.90 ms (+29 %)**. Yet collectives still eat 32–96 % of
device time because the cost model is *(collective COUNT) × (per-barrier
sync-wait)*: the same all-gather measures **GPU:0 = 621 µs vs GPU:3 = 3081 µs**
(5× straggler spread on a ~6 KB message) — the duration is barrier-sync surfacing
**upstream compute imbalance**, not data movement.

**Ring-vs-stack crossover** (:166-171, warm min-of-15 ms):

| matvec | regime | nt | 1×1 | 2×2 | mem 1×1/2×2 |
|---|---|---|---|---|---|
| stack | inflated | 1 | 15.13 | 18.80 | 1350 / 819 MB |
| ring | inflated | 1 | 14.09 | **12.35** | 1350 / 347 MB |
| stack | inflated | 8 | 90.93 | **33.93** | 1359 / 578 MB |
| ring | inflated | 8 | — | 55.13 | — / 2647 MB |

**Crossover ≈ nt 2–3 on both axes → empirical dispatch threshold nt ≤ 2 → ring,
nt ≥ 3 → stack.** Ring memory is linear in nt (10.5 GB @1×1 nt8) vs stack flat
(1.36 GB) — that is why the stack exists. Trial-stack temp is **flat** in
n_trials (183.9 → 183.4 → 183.4 MB for 1/4/8) while ring is strictly linear
(183.9 → 734.1 → 1467.7 MB) (PHASE2_LOG.md:186-196).

### 3.4 Matvec optimizations: kept, regressed, vetoed

| id | change | measured | disposition |
|---|---|---|---|
| **P3** | hoist M_X/M_Y pair-amplitudes out of the iteration (2 GEMMs, 471 MB each) | nt1 **15.11 → 13.67 ms (+9.6 %)**; nt4 48.59 → 47.17 (+2.9 %); relerr 7.6e-17 / 4.1e-16 = bit-identical | **LANDED** (PHASE2_LOG.md:720-728). Caught a real staleness bug: the hoisted M went stale in `build_finite_q_data`, closure rel_err **2.66** |
| **P2** | exchange collectives 40 → 12 on the resolvent (32 collective-permute + 4 all-reduce + 4 reduce-scatter → 8 all-reduce + 4 all-gather) | b=1 **14.7 → 19.6 ms (~30 % SLOWER)**; b=8 **1.94 → 1.51 ms/col**; e2e 148.25 → 146.31 s | **NOT MERGED** — "the value is architectural (scale-out) + code (−62 net lines)", branch `agent/bse-comms-opt` deliberately excluded (PHASE2_LOG.md:949-990, 2076-2078) |
| **P-NT** | nt-aware dispatch replacing the `matvec_kind` flag | routing bs==1 through the stack was a measured **~1.5× single-vector regression**; threshold **bs ≤ 2 → ring, bs ≥ 3 → stack** | **LANDED**; `matvec_kind` **deleted** (PHASE2_LOG.md:209-210, 960-966) |
| **P1** | **c64 mixed-precision matvec** | **~2× on the whole HBM-bound W-term** — halves T and all ~7 round-trips. *"Biggest single lever."* | **OWNER-VETOED 2026-07-16, NEVER MEASURED.** "gain is inferred from the measured AI=0.039 profile, **not a c64 A/B wall (not run)**" (JOINT_FINDINGS.md:198, 231-232) |
| **P4** | eliminate one W-term T-transpose | **15.2 % of device wall** @inflated, 36.7 % @fixture; one → ~7-8 % | **[PROPOSED], never run.** "It is **not free elimination**" — trades a transpose for a larger encode collective; net win regime-dependent |
| **P5** | drop cosmetic `donate_argnums` | confirmed **no copy emitted** at any site (HLO `copy(` grep) | dropped as a perf item |
| **P7** | XLA latency-hiding-scheduler flag | **confirmed non-lever** — already default-on, off costs +29 % | rejected |

### 3.5 Compile behaviour — a genuine strength, keep it

[MEASURED] PHASE2_LOG.md:1517-1530 and the AFTER census at :1637-1643 (32-pt path,
1 GPU A100, `JAX_LOG_COMPILES`, cold cache):

| engine | compiles | note |
|---|---:|---|
| `solve_path` (scan of per-Q block-Lanczos) | **1** (26.3 s XLA) | whole path + refit rows in ONE compile |
| `eval_vq` | **1** | every Q dispatch-only |
| `_q_batch` (htransform ψ(k+Q)) | **1** | batched q-list |
| `_clean_split` | 15 → **1** | was one per distinct ngk; fixed by ngkmax zero-pad |
| one-time small eager ops | ~135–150 | O(1) in nQ |

**Per-Q marginal compiles: 0**, re-verified at nQ = 37 and again at 16-GPU scale
(run A: 13 Q, one compile).

### 3.6 A solver bug that invalidates older small-window results

PHASE2_LOG.md:1458-1474. With a 4v4c window (n_flat = 144) and Krylov 320
(bs 8 × 40 iter), the **fixed-iteration block Lanczos ran past Krylov exhaustion**:
the residual block collapses, QR of a ~zero block returns junk directions, and the
manufactured α/β blocks put Ritz values anywhere — measured **60–100 meV BELOW the
dense ground state**, with different garbage per code path (production
`solve_bse_sharded`, the driver scan, and the htransform-ψ variant all disagreed
below 0.179 eV while dense eigh said the true minimum **is** 0.179359 eV). Fix:
clamp `max_iter` at `floor(n/bs)`; full reorthogonalisation by default. Post-fix
`solve_bse_sharded` == 144-dim dense eigh to 0.0000 meV. A related fix allocated
**M+1** Krylov slots at all three jit sites (PHASE2_LOG.md:1281-1287).

> *"Every earlier small-window BSE Lanczos run on this lineage is suspect below its
> true ground state."* — PHASE2_LOG.md:1473-1474

Cross-mesh ghost signature worth knowing: bs=1 gives max|1×1 − 2×2| = **1.5e-16**
(bit-level); at bs=4 the *true* Ritz values are bit-identical across meshes while
the **ghost** entries differ by ~7e-5 (PHASE2_LOG.md:257-263).

### 3.7 Routing the matvec through cuBLASMp — considered TWICE, rejected BOTH times

The clearest "abandoned with a recorded reason" in the archive; both records are
independent, and **neither is a measurement**.

**Record 1** — `bse_multinode_2026-07-20/WORKLOG.md:105-110`:

> **Recommendation: keep the matvec as sharded `dot_general`.** These are BATCHED
> multi-index pair-basis contractions (k,c,v,μ,ν), not single large dense GEMMs;
> cublasmp targets large 2D dense `C=A@B`. Routing them through cublasmp would
> force 2D reshapes + per-batch FFI calls and lose XLA fusion of the ring-comm
> reduce-scatter pattern.

**Record 2** — `bse_multinode_NOTES_gaps_and_audit_2026-07-20.md:68-82`, with the
actual index strings: `kvsN,cvk->cksN`, `kctM,cksN->MNtsk`, exchange
`kcvN,bcvk->bN`, batched over the Lanczos block dim `b`, with `lax.psum_scatter`
for the k/μ/ν reductions — "already the right distributed primitive".
`ffi/cublasmp/batched.py:batched_distributed_gemm` handles stacks of *plain 2-D
dense* `C[q]=A[q]@B[q]` — the W-solve V@χ shape, **not** the matvec's fused
contractions. Routing it there would need one host-dispatched FFI call per Lanczos
iteration, **breaking the single-compile `lax.scan`**.

Also audited and left native: `block_lanczos_eig_jit`'s projected Rayleigh-Ritz
`eigh(T)` is ≤ 320² and replicated; the block-QR is tall-skinny and cheap
(WORKLOG.md:97-100).

Tag: **[PROSE + structural argument]. No cuBLASMp matvec was ever timed.** That is
a legitimate reason but not a measured refutation, and it does not transfer to a
regime where the matvec *is* a large dense GEMM.

### 3.8 HLO collective audit of the interpolation — clean [MEASURED]

`bse_multinode_HLO_AUDIT_2026-07-20.md` (already in this directory), n_μ = 640,
nG = 337, nq = 144, 4×4 mesh. **Note this audits `_clean_split` and `eval_vq` —
the V_Q interpolation — NOT `bse_stack_matvec`.**

* `eval_vq` (`V = V_SR + conj(A_x) @ A_y.T`): only **3 `collective-permute` on
  `c128[160,337]`** — the two small A reshards. The outer product is a **local
  dot**. No collective carries the n_μ = 640 tile dim.
* `_clean_split` (`R g Rᴴ`, `Sc@V_δ@Sc`, batched over q): **zero collectives** —
  a raw grep of `hlo_clean_split_16gpu.txt` for
  `all-reduce|all-gather|collective-permute|reduce-scatter|all-to-all` returns 0.
* The cuSolverMp seam **does** gather: R goes `P('x','y')` (160×160/dev) →
  qb3-replicated (640×640/dev), 6.2 MiB/tile, ~18.8 MiB/dev per 48-q chunk. Cheap
  at n_μ = 640; this is the gather §4.4's cuBLASMp work eliminates.

**There is no xprof or HLO dump of `bse_stack_matvec` at production 12×12 scale
anywhere in the archive.** The only matvec traces are the audit's fixture+inflated
1×1/2×2 set on 4 × A100.

---

## 4. FFI vs native — measured margins, with baselines

### 4.0 Two decisive negatives on the claims you asked me to chase

**(a) There is no FFT FFI of any kind in the Perlmutter archive, so there is no
Perlmutter evidence for the ~3.78× MKL-FFT figure.** Four independent
verifications:

1. The FFI source tree is exactly three subpackages —
   `environment/lorrax_setup/src/ffi/{common,cusolvermp,phdf5,slate}`.
   `PORTING.md:1-4` opens: *"Covers `ffi.cusolvermp`, `ffi.phdf5`, and
   `ffi.slate`. All three link into one `liblorrax_ffi.so`."* Three, not four.
2. The linked binary confirms it.
   `environment/manifests/runtime_elf_dependencies.txt` lists `liblorrax_ffi.so`'s
   `NEEDED` entries as `libcusolverMp`, `libcublasmp`, `libcal`, `libnccl`,
   `libcudart`, `libcusolver`, `libcusolverMg`, `libhdf5_parallel_gnu_123`,
   `libmpi_gnu_91`, `libslate`, `libblaspp`, plus libc/libstdc++/libgomp. **No
   `libcufft`, no `libmkl_rt`, no DFTI.**
3. The complete FFI target table —
   `gw_refactor_map_2026-07-01/archive/files/ffi_common.md:21` — has **16
   entries**: `lorrax_cusolvermp_{eigh,batched_potrf,potrs,solve_lu}`,
   `lorrax_cublasmp_{batched_gemm,w_solve}`, `lorrax_cusolvermg_eigh_f64`,
   `lorrax_phdf5_{write,read,read_kchunk,read_kchunk_union}`,
   `lorrax_slate_{eigh,potrf,trsm,batched_potrf,batched_trsm}`. **No FFT/DFT
   target.**
4. `DFTI` occurs **zero** times in 927 files. `MKL` occurs 13 times, never as an
   FFT backend — and the one substantive sentence is a *rejection*:
   `ffi_host_platform_2026-07-10/report.md:142-144` — *"MKL's ScaLAPACK exists via
   `intel-oneapi` modules but targets Intel MPI — wrong ABI for Cray MPICH; not
   needed."*

"flat-k FFT" here is **pure JAX/XLA**, not FFI:
`gw_refactor_map_2026-07-01/archive/files/common_fft.md:30` describes
`make_flat_k_fft` as a reshape → `with_sharding_constraint` → local 3D FFT
wrapper dispatching to `make_sharded_{i,}fftn_3d` (a single 3D **cuFFT plan via
XLA**, worth ~1 s on a Si 4×4×4 BSE 200-iter Lanczos vs the per-axis
custom_partitioning form — an XLA-vs-XLA number). cuFFT is only ever reached
through XLA's `fft_thunk`.

There is **no `sigma.exec` of 272 s or 71.9 s anywhere**; the largest `sigma.exec`
in the archive is 165.6 s.

For contrast, the **current** Frontera tree
(`/work2/08271/jackmc/frontera/lorrax/src/ffi/`) contains `cufft`, `mklfft`,
`mklblas`, `scalapack` and `linalg` alongside the three Perlmutter ones, with
`src/ffi/mklfft/cpp/fft_flat_k_ffi.cc` and
`src/ffi/cufft/cpp/fft_flat_k_cuda_ffi.cc` present. **Those backends are
post-Perlmutter work and must be validated against the Frontera runs that produced
their numbers.**

> The nearest thing to an FFT experiment in the archive went the **opposite** way:
> `fft_helper_unification_2026-05-11/report.md:48-59` — shape
> `(9,80,2,24,24,80)`, 1 A100, c128: plain local 3D IFFT **0.052 s / 2.123 GB peak**
> vs a manual 8-chunk FFT loop **10.596 s / 2.389 GB** — a **200× slowdown** — with
> cuFFT scratch measured at **0.0 GB** for every production shape probed. Decision
> (:96, :101): *"do not land chunking for now … I did not find such a case."*

**(b) No ~7.3× GEMM margin exists in this archive.** `grep -niE "7\.3\s*x"` over
the whole tree returns **zero** hits. The only GEMM-FFI A/B preserved is §4.1
row 4, and it is **−5 %**. **No evidence found.**

### 4.1 The margins table — every row states its baseline

All [MEASURED] unless tagged. ⚠ marks rows where more than one variable changes.

| # | wrapped kernel | baseline (native) | platform | shape regime | margin | source |
|---|---|---|---|---|---|---|
| 1 | **cuSolverMp** charge Cholesky | `sharded_cholesky` (JAX) | 4 × A100, 2×2, 1 node | MoS2 3×3 bispinor, 256 charge / 208 transverse centroids | **0.78× (28 % SLOWER)** — 3.70 vs 2.89 s | `ffi_host_platform_2026-07-10/report.md:198-199` |
| 2 | **SLATE** charge Cholesky | `sharded_cholesky` | 4 × A100, 2×2 | same | **0.65× (55 % SLOWER)** — 4.48 vs 2.89 s | same, :198,:200 |
| 3 | **SLATE** charge Cholesky | `sharded_cholesky` | **1 Milan CPU node**, 4 ranks × 32 c | same | **0.47×** — 1.56 vs 0.73 s | same, :201-202 |
| 4 | **cuSolverMp** transverse LU | per-q JAX LU | 4 × A100, 2×2 | same | **0.5× (2× SLOWER)** — ~3.0 vs ~1.5 s/chan | same, :198-199 |
| 5 | **ScaLAPACK (libsci)** LU | per-q JAX LU | **1 Milan CPU node** | same | **1.42× — the ONLY throughput WIN in the archive** — 3.1 vs 4.4 s | same, :201,:203 |
| 6 | cuSolverMp chol + LU, **full GW wall** | `sharded_cholesky` + JAX LU | 4 × A100, JID 52886424 | MoS2 3×3 COHSEX | **0.999× (tie)** — 101.40 vs 101.25 s; charge ζ fit 10.870 → 11.290 s (slower) | `ffi_boundary_profile_a_2026-05-13/report.md:64-70` |
| 7 | **SLATE** potrf, ζ-fit solve step | `sharded_cholesky` | 1 × A100, 1×1 | 399 centroids, nq = 5 | **1.02× (dead heat)** — 0.243 vs 0.247 s; *total recorded worse* 8.939 vs 8.760 s | `slate_linalg_ffi_2026-07-10/p4_e2e/gnppm_{slate,default}/run.log:225,243` |
| 8 | **cuSolverMp eigh**, `vq_prepare` | native batched eigh **on 4 GPUs** ⚠ device count also changes | 4 → 16 × A100, 4 nodes | 144 sequential Hermitian **640×640** c128 | **0.15× (6.7× SLOWER)** — 59 → 398 s | `bse_multinode_2026-07-20/WORKLOG.md:142-158` |
| 9 | cuSolverMp eigh — **accuracy** | numpy reference | 16 × A100, 4×4 | 640×640 c128 + f64 sym | max\|eval−ref\| **3.865e-12** / **5.230e-12** PASS | same, :80-85 |
| 10 | cuSolverMp eigh — **physics invariance** | dir10 4-GPU native `.dat` | 16 vs 4 × A100 | 40 Q × 8 eigenvalues | **max\|ΔE\| = 0.0000e+00 meV** | same, :127-132 |
| 11 | **cuBLASMp** 2-D distributed V_Q recon | replicated `_clean_split`, **same 16 GPUs, same eigh** — a clean A/B | 16 × A100-40G, 4×4 | n_μ = 640, nq = 144 | **0.95× (5 % SLOWER)** — vq_prepare 398 → 418 s | `bse_cublasmp_recon_2026-07-20/WORKLOG.md:126-143` |
| 12 | cuBLASMp recon — **parity** | replicated recon | 16 × A100 | n_μ = 640 | relF S **1.70e-16**, V_SRc **2.34e-13**, Fch **1.92e-15** | same, :105-118 |
| 13 | cuBLASMp recon — **capability** | replicated recon | 16 × A100-40G | synthetic Hermitian C_q | replicated **OOMs at n_μ = 32768** (48.00 GiB request); distributed runs at **1024 MiB/proc**. Crossover ≈ **n_μ 24k** | same, :69-92 |
| 14 | **cuSolverMp Cholesky** ζ charge-solve | `replicated_rank_truncate` (JAX) — *different algorithm* | 16 × A100-80G | n_μ = 2412, nq = 144, 18 r-chunks | **0.52× (1.93× SLOWER)** — 814.8 vs 421.2 s — **and produced a ζ that FAILED the disk gate** | §4.4 |
| 15 | **PHDF5** slab write/read | `process_allgather` + rank-0 h5py | 4 GPU, 1 node | Si 4×4×4 / MoS2 4×4, ~300 MB | **1.02× / 0.96× (tie)** — 53.2 vs 54.1 s; 51.3 vs 49.3 s | `C_unified_slab_io_2026-04-17/report.md:19-26` |
| 16 | **PHDF5**, restart workload | allgather | 4 GPU | Si 4×4×4, 64 q reads | **0.95× total; 0.25× on the V_q read leg** — 56.1 vs 53.2 s | `C_zeta_restart_slab_io_2026-04-17/report.md:37-45` |
| 17 | PHDF5 collective writes | gather | 4 nodes / 16 GPU | 4 GB writes | **"8×"** | **[PROSE] — the `phdf5_vs_gather_bench` output is ABSENT from the archive** (`C_unified_slab_io_2026-04-17/report.md:30`) |
| 18 | PHDF5_FFI generally | *(unnamed)* | *(unnamed)* | *(unnamed)* | *"~5× faster w/ Lustre striping"* | **[PROSE] — UNBASELINED** (`gw_config.md:16`) |
| 19 | SLATE first-call cost | test reordering | 1 × A100 | trsm[48] | ~**11 s** one-time lib load + CUDA/SLATE setup + first FFI compile (test itself ~2 s) | `suite_speedup_2026-07-15.md:96-100` |
| — | **MKL FFT (DFTI) flat-k** | — | — | — | claimed 3.78× | **NOT PRESENT IN ARCHIVE** (§4.0a) |
| — | **GEMM path** | — | — | — | claimed 7.3× | **NOT PRESENT IN ARCHIVE** (§4.0b) |

Row 8's confound must be stated whenever you quote 6.7×: it compares 4 GPU/native
against 16 GPU/FFI. Row 10 shows the physics is bit-identical, so only the *timing*
attribution is confounded, not correctness.

### 4.2 The pattern, stated plainly

**Across every FFI-vs-native comparison the Perlmutter archive actually measured,
the wrapped kernels tie or lose on speed at the scales run.** The single positive
throughput margin is ScaLAPACK LU on CPU (1.42×, row 5); the only other positive
number (row 17's 8×) has no preserved measurement.

The FFI's recorded justifications, in the authors' own words, are **capability**
(row 13 — the only path once one n_μ×n_μ tile no longer fits, which is exactly the
LORRAX scaling target), **portability/equivalence** (bit-identical σ across
backends, rows 9/10/12), and **memory headroom** — not throughput.

The recorded mechanism for the losses is consistent: *"the 144 per-q cusolverMp
eighs on small 640×640 tiles are NCCL-latency-bound across 4 nodes (the expected
'cross-node collectives don't help a small case' regime)"*
(`bse_multinode_2026-07-20/WORKLOG.md:152-158`), and independently *"this FFI is
behaving like a serial queue of independent distributed solves, not like a batched
library call"* (`cusolvermp_ffi_profile_2026-05-12/report.md:63`, from a clean nq
sweep: 1 → 1.951 ms, 3 → 3.821, 9 → 9.447, 18 → 17.862 ms — near-perfectly
linear). The suggested remedy — a **batched-across-q FFI eigh** — is
**[PROPOSED], never implemented.**

The FFI's own overhead is measurable: enabling cuSolverMp took GPU events
**12 338 → 241 980**, GPU compute streams **8 → 654**, D2D copies **846 → 28 173**
(8.55 GiB/11.71 ms → 29.07 GiB/103.09 ms) in the same run, with the report
concluding *"This is not a convincing speedup path yet"*
(`ffi_boundary_profile_a_2026-05-13/report.md:168-179, 239-242`).

**Transfer warning to Frontera.** Rows 1–5 and 8/11 are 1-node-4-GPU or
4-node-16-GPU Perlmutter results. The small-tile penalty is a *cross-node NCCL
latency* effect and will differ on a single node or a different interconnect. Carry
the **mechanism** (small tiles + per-q sequential FFI calls + cross-node
collectives = latency-bound), not the numbers. Also note the honest caveat the
backend-matrix report puts on itself (`ffi_host_platform_2026-07-10/report.md:206-212`):
*"at this fixture scale the solver axes are noise … **Backend performance
discrimination needs production scale**"* — and the production-scale FFI-vs-native
comparison it names as the prerequisite **was never run**.

### 4.3 Numbers in this archive that are NOT FFI margins — do not let these leak in

| number | what it actually is | source |
|---|---|---|
| **3050× / 1069×** | `jax.jit` vs a bare eager `shard_map` (which re-traces and re-lowers to HLO **every call**). `gen` 2536 → 0.83 ms; `snapshot` 2773 → 2.60 ms. Single call, 1×1 mesh. | `PHASE2_LOG.md:481-491` |
| **6.3× / ~38×** | G-flat vs r-space **algorithm** change. `V_q_compute` 4.4 → 0.7 s, `close_io` 3.8 → 0.1 s. **Both arms use the same FFI I/O path.** | `gflat_e2e_mos2_3x3_2026-05-11/report.md:32-39` |
| **~18×** | `cholesky_2d_batched` docstring, one XLA dispatch vs a Python loop — **native-vs-native**, and the code auditors themselves label it *"unverified perf folklore baked into docs"* | `gw_refactor_map_2026-07-01/archive/files/cholesky_2d.md:88, 177-179` |
| **2.96× (812 → 274 s)** | IBZ-cascade **physics/symmetry** reduction (q-count 36 → 8) | `zeta_rchunk_memory_model_2026-05-13/cri3_ibz_cascade_validation.md:10` |
| **2000×** | async-FFI *dispatch* latency vs the **synchronous FFI**, not vs native — and it cost **+10 s end-to-end** until the root cause was found | `ffi_e2e_profile_2026-04-18/report.md:50-53` |
| **~2×** | runtime-arg vs closure-baked symmetry tables — native-only, and flagged as a docstring claim | `gw_refactor_map_2026-07-01/SHARDING_RULES.md:162` |
| **4.3× / 2.28× / 19.5×** | pure numpy/Python algorithmic changes to a crossing-quadrature solver; no FFI content | `minimax_solver_speed_2026-07-21/report.md:50-52` |

**Allocator confound.** Any FFI-vs-native number measured under a different
allocator is not comparable: *"**BFC allocator is ~10 % faster than `platform` on
pure-JAX workloads**. Use `platform` when FFI is hot (Sigma stage, cuSOLVERMp,
phdf5); use BFC for ζ-fit-only benchmarks"*
(`memory_model_refit_2026-05-17/MEMORY_MODEL_SYNTHESIS.md:162`; write is also ~2×
faster on BFC per `agent_s_round11_liveverify.md:216-218`).

### 4.4 Row 14 in full — where the FFI was slower *and* wrong

Three ζ full-BZ regeneration attempts, all n_μ = 2412 / nk = nq = 144 /
16 × A100-80G / 18 r-chunks, in `bse_exciton_smooth_2026-07-21/logs/`:

| # | log | route | wall | outcome |
|---|---|---|---|---|
| 1 | `zeta_fullbz_BAD_cholesky_fallback.log` | `L_q = chol(C_q) [path=cusolvermp_cholesky]` | **814.8 s**, HWM 56.01/65 GB | gate `makeVq_vs_disk max 3.189e+01` vs tol 5e-06 → **FAIL** |
| 2 | `zeta_fullbz_OOM_replicated_eigh.log` | `path=replicated_rank_truncate` after raising `_REPLICATED_CHOL_MAX_STACK_BYTES` 4 → 16 GiB | — | **OOM**, `RESOURCE_EXHAUSTED … 42550052712 bytes` (39.63 GiB) |
| 3 | `zeta_fullbz.log` | `L_q = rank-truncated pinv [path=replicated_rank_truncate]` | **421.2 s** (fit 98.7 %), HWM 56.11/65 GB | gate `makeVq_vs_disk max 1.166e-08` → **PASS** |

`zeta_probe_ibz_vs_fullbz.log` diagnoses attempt 1 precisely: the *IBZ* ζ
reproduces the disk V_qmunu to relF 1e-9…1e-15, while the *full-BZ* ζ from the
Cholesky route is off by relF **16–28** at every probed q, with best-fit scale
exactly 1.0 — structural error, not normalisation.

**Caveat on the 1.93×:** these are different *algorithms* (Cholesky vs
rank-truncated pinv), not one algorithm on two backends, so it is a combined
algorithm+backend margin. What is unambiguous is the *choice*: the native
replicated route was both faster and the only correct one at this size. This
corroborates `gw_converged_12x12_80ry_2026-07-21/report.md:99-150` independently —
*"only the replicated route carries the rank-truncation cure"*, whose sanity gate
failed on exactly `path=replicated_rank_truncate appears 0x in gw.out`.

### 4.5 What the cuBLASMp recon bought — and it is missing from the current tree

`bse_cublasmp_recon_WORKLOG_2026-07-20.md` + `…_NMU2_DISTRIBUTION_2026-07-20.md`
(both copied here). The point was to kill §3.8's gather: R stays 2-D sharded and
the whole reconstruction runs through batched distributed GEMM, so every n_μ²
intermediate is `P(None,'x','y')`.

| n_μ | replicated tile / proc | distributed shard / proc | ‖RΛRᴴ−C‖/‖C‖ | eigh (s) | recon GEMMs (s) |
|----:|---:|---:|---:|---:|---:|
| 640 | 0.01 GiB | 160×160 = 0.4 MiB | 3.18e-15 | 5.9 | 2.5* |
| 4096 | 0.25 GiB | 1024×1024 = 16 MiB | 5.88e-15 | 4.5 | 1.7 |
| 16384 | 4.00 GiB | 4096×4096 = 256 MiB | 1.15e-14 | 22.2 | 20.3 |
| 32768 | 16.00 GiB → **OOM** | 8192×8192 = 1024 MiB | 1.69e-14 | 109.7 | 85.5 |

`*` includes one-time cuBLASMp descriptor/context warmup. An **N_μ² distribution
audit** verified all 9 O(n_μ²) intermediates (Qraw, S, Sc, A_ref, A_lr, V_δ, T1,
V_SRc, zt) at `P(None,'x','y')`, per-proc shard exactly (n_μ/Px, n_μ/Py) =
(160,160) — never (640,640), never half-distributed. **9/9 full-mesh.** Compiled
HLO shows `sharding={devices=[1,4,4]<=[16]}` on
`custom_call_target="lorrax_cublasmp_batched_gemm"`.

Kept but **default OFF**: `prepare_coarse(..., distributed_recon=False)`; CLI
`--distributed-recon off|on|auto`.

**This capability is NOT in the current Frontera tree.**
`grep -rn "distributed_recon\|_distributed_prims\|_recon_distributed_chunk"` over
`/work2/08271/jackmc/frontera/lorrax/src/` returns nothing, and
`src/bse/exciton_bands.py` has no `--distributed-recon` flag (its argparse block at
lines 292–370 carries every *other* archive flag). The underlying primitive
`ffi/cublasmp/batched.py:batched_distributed_gemm` **did** survive. Branch
`agent/bse-cublasmp-recon` was never merged; recovering it is a git operation and
the copied WORKLOG is the spec.

### 4.6 FFI optimizations abandoned, with reasons

| what | outcome | reason | source |
|---|---|---|---|
| CUDA-Graph replay of the cuSolverMp q-loop | compiled, **failed at runtime on all 4 ranks** | `INTERNAL: cuda graph capture cusolverMpPotrf failed at q=0: status=7` — "`cusolverMpPotrf` … **does not appear stream-capture-safe**". Prototype **removed from source**, library rebuilt | `ffi_boundary_profile_a_2026-05-13/report.md:279-297` |
| Descriptor/workspace caching for potrf/potrs | deprioritized | setup is "**single-digit microseconds** per FFI call"; expected win **below 0.1 %** at nq=9 | `cusolvermp_ffi_profile_2026-05-12/report.md:7, 112` |
| Async prefetch of G-flat slab reads over PHDF5 FFI | **deadlocked**, disabled via env | "Tighter NCCL ↔ MPI interleaving needed" | `gflat_e2e_mos2_3x3_2026-05-11/report.md:143-147` |
| Async D2H in the phdf5 write handler | reverted, then partly reinstated | `cudaEventDestroy` blocked **750–775 ms on ranks 1/2/3** but ~8 µs on rank 0. Portable lesson: **pool your events** | `ffi_cudaevent_mystery_solved_2026-04-18/report.md:9-11, 60-65, 162-165` |
| `use_ffi_io` + `use_phdf5_gspace` multi-chunk together | fails | "PhdfWfnReader + SlabIO-FFI on the same ranks **race on MPI-IO state**" | `aot_memory_model_poc_2026-04-20/report.md:296-303` |
| `cusolvermg` | never productionized | "no live consumer — bench-only target" | `slate_linalg_ffi_2026-07-10/report.md:326-327` |
| cuBLASMp itself | **dead for two months** (2026-05-10 → 07-10) | stage drift → wrong CAL-ABI dispatch (cuBLASMp 0.4.0 CAL vs cuSolverMp 0.7.2 NCCL) → `status=6` everywhere; `screening_solver=cublasmp_ffi` unusable. **Consequence: the CUBLASMP_FFI fused W-solve has NO timing measurement vs JAX_NATIVE LU anywhere in the archive** | `slate_linalg_ffi_2026-07-10/report.md:11-13, 262-271` |
| cuSolverMp eigh on 4 GPU | **hung** → fell back to native | recorded once, no follow-up | `PHASE2_LOG.md:83` |

### 4.7 The other big lever is not an FFI at all

The largest measured speedups in the whole exciton pipeline came from **XLA
sharding hygiene, not native kernels**: replicated → sharded q-batch (911 → 346 ms,
§2.3), trainer q-batching (vq_prepare 52.9 → 9.7 s), `with_sharding_constraint` on
einsum outputs (§2.5), killing per-shape recompiles (`_clean_split` 15 → 1), and
removing host round-trips. Non-solve stages **253 → 66 s = 3.8×**, values
bit-identical. If you are hunting margin on Frontera, the archive says look here
first.

---

## 5. Failures, with their recorded reasons

| # | what was tried | outcome | recorded reason | source |
|---|---|---|---|---|
| 1 | Exciton bands on the 80 Ry / n_μ=2412 restart, **8 configurations** | abandoned, "produced no numbers" | `fH_R` **replicated**: 49.93 GiB/dev at n_μ=2412; 102 GiB/dev at the capacity-satisfying n_μ; quadratic in n_μ, mesh-independent | `gw_converged…/report.md:33-38, 425-470` |
| 2 | ↳ retried after the sharding fixes | **SUCCEEDED** (run A) | three memory/layout fixes; ~17 GiB HWM | supersedes #1, §1.6 |
| 3 | fH window nb = 32, 36, 40 at n_μ=2412 | FAIL 60.5 / 817.8 / 1314.1 meV | ψ-at-centroids rank-deficient **before** the nominal bound (4570 of 4824) | §1.3 |
| 4 | fH window nb = 16 (zero guards) | FAIL, min-sval 0.0917, **mesh-dependent** | f-transform `shift = max_k ε_top` → top state degenerate with fH's exact zeros | §1.3 |
| 5 | Routing the BSE matvec through cuBLASMp | rejected twice, **never benchmarked** | batched multi-index einsums ≠ dense 2-D GEMM; breaks the single-compile `lax.scan` | §3.7 |
| 6 | cuSolverMp Cholesky for the full-BZ ζ | wrong ζ (gate 31.89 vs tol 5e-6) **and** 1.9× slower | only the replicated route carries the rank-truncation cure | §4.4 |
| 7 | Distributed cuSolverMp eigh for speed | 6.7× slower | 144 sequential 640² tiles, cross-node NCCL latency-bound; the FFI is "a serial queue of independent distributed solves" | §4.1-4.2 |
| 8 | Dense off-grid Q on converged data, host side | **host OOM**, SLURM `oom_kill`, all 16 tasks killed | vq prep host mirrors | `bse_exciton_smooth…/logs/run01_hostOOM_prep_mirrors.log` |
| 9 | ↳ device side | `Can't reduce memory use below 24.08GiB … only reduced to 49.93GiB` | unsharded `fH_R`-class allocation | `logs/run01_OOM_PR_unsharded_alloc.log` |
| 10 | 1000-centroid basis to smooth the bands | **did not work** | the iQ 6/9/16-17 dips are window-cache artefacts, not ISDF error | §2.5 |
| 11 | Fixed-iteration block Lanczos past Krylov exhaustion | **ghost Ritz values 60–100 meV below the true ground state** | QR of a collapsed residual block returns junk directions | §3.6 |
| 12 | c64 matvec — "biggest single lever, ~2×" | **owner-vetoed 2026-07-16, never measured** | gain inferred from AI = 0.039, not from a c64 A/B wall | §3.4 |
| 13 | Exchange collectives 40 → 12 | **~30 % regression at b=1**, win at b≥8 | latency-bound fixture; value is architectural | §3.4 |
| 14 | `bse_k_grid` + exciton_bands | crash in `build_fH_R` ifft-reshape | driver passed the **densified fine** grid as `kgrid_co` while `ctilde` is coarse | `bse_figures…/WORKLOG.md:73-78, 96-100` |
| 15 | vq_interp on an orbit-closed (D3h) centroid set | `ValueError: vq_interp needs FULL-BZ zeta storage` | GW writes IBZ-only ζ whenever orbit closure passes — **the D3h centroid cure and the BSE consumer are in tension by default**, joined only by the undocumented `LORRAX_FORCE_FULL_BZ=1` | `gw_converged…/report.md:472-489` |
| 16 | `run_gates` at 1496 centroids | OOM, 58 GB replicated alloc | opt-out `LORRAX_SKIP_VQ_GATES=1` added | `bse_figures…/WORKLOG.md:113-116` |
| 17 | vq_interp C_q eigh, nq=36 on a 16-device mesh | ran on 4 GPU instead | 36 not divisible by 16; needs q-axis padding to 48 — **deferred, not a physics issue** | `bse_figures…/WORKLOG.md:117-121` |
| 18 | `initialize_wfns(eqp_file=…)` for QP energies | **silently returns DFT energies** | wrong text format, parse failure swallowed by a log line | §1.4 |
| 19 | Centroid pruning at 1600 centroids | OOM on the pivoted-Cholesky Gram, 1 GPU | `--oversample 1.5` default; fix `--oversample 1.0` | `bse_figures…/WORKLOG.md:31-32` |

---

## 6. Reproduction checklist

1. **Confirm the `fH_R` sharding fixes are in your tree first** (§1.6, §2.5).
   Without them a converged-scale window OOMs, and the archive's own parent
   campaign concluded — wrongly — that the problem was unsolvable.
2. **Set `nband` as an ABSOLUTE index** ≥ `26 + ncond` for a gap-centred window.
   Check `ctilde orthogonality` is ~1e-14, not 1e-1 (§1.4).
3. **Choose nb from the measured rank**, not the nominal bound:
   `nb ≤ rank(ψ_μ)/nk`. At n_μ=2412 / nk=144 that is **nb ≤ 31**, and **nb = 28**
   is the validated setting. At n_μ = 640, keep `nband ≤ 48` (§1.3).
4. **Keep ≥ 2 guard bands** each side; run A used 6 + 6.
5. **Prefer an on-grid Q path + `--vq-mode ongrid`** — no full-BZ ζ, no mini-BZ
   head model, zero interpolation error. Q-path blocks in
   `qpath_mos2_GMKG_crystal_b.txt`. Going off-grid costs a ζ regeneration (§4.4)
   and ~3× the wall (630 s / 13 Q vs 1999 s / 39 Q).
6. **Use `--eqp` with a BGW-format `eqp1.dat`**, not `initialize_wfns(eqp_file=)`.
   Expect a fixed ~58 meV in the gate's `max|Δε_c|` — that is a permutation; read
   `recon` beside it (§1.4).
7. **Verify the q=0 head is injected** — its absence is only a `RuntimeWarning`
   and the run still plots (§1.4).
8. **Take the two free speedups the archive left on the table**:
   `--skip-rerun-check` (the warm re-scan is **42 %** of production wall) and
   `--max-iter 20` (values-preserving below print precision, halves the solve).
   Together ≈ 2.8× on a 40-pt path. Both flags exist in the current tree; only the
   first was ever adopted (§2.3).
9. **Give host stages threads.** Use `--cpus-per-task` and `OMP_NUM_THREADS` — the
   container numpy BLAS is effectively serial (~3 GFLOPS) and several "slow" runs
   in the archive are that, not physics (§3.1).
10. **Budget htransform first at production n_μ** (~25 s/Q at n_μ=2412 on
    16 × A100-80G) and the matvec/`vq_prepare` first at fixture n_μ (§2.1, §3.1).
11. **On Frontera, re-measure every FFI margin.** §4.2. And expect the fp64-heavy
    stages (SVD, eigh, matvec GEMMs) to shift the balance further on sm_75.

---

## 7. Artifacts copied alongside this file

| file | what it is | why reusable |
|---|---|---|
| `bse_exciton_converged_2026-07-21.md` | full report for run A | the only converged / QP-energy / exact-exchange exciton bandstructure in the archive |
| `bse_exciton_window_sweep_table_2026-07-21.md` | the 7-point nb sweep | the band-window parameter table (§1.3) |
| `qpath_mos2_GMKG_crystal_b.txt` | **[RECONSTRUCTED]** `K_POINTS crystal_b` blocks, 13/40/39/139-point paths | no deck survives; arithmetically consistent with every logged banner |
| `centroids_frac_640_mos2_fixture.txt` | 640 ISDF centroids, fractional, snapped to FFT grid [24 24 80] | the fixture shared 3×3↔12×12 *deliberately* to isolate k-grid from basis effects; trained runs D–H |
| `bse_multinode_WORKLOG_2026-07-20.md` | 16-GPU distributed exciton pipeline | launch model, 5 distributed gaps + fixes, gate-4 timing table |
| `bse_multinode_NOTES_gaps_and_audit_2026-07-20.md` | per-gap working notes | the matvec-vs-cuBLASMp rejection with actual einsum index strings |
| `bse_matvec_efficiency_audit_JOINT_FINDINGS_2026-07-16.md` | 3-agent matvec profiling | AI = 0.039, category split, transpose cost, ring/stack crossover, the P1–P7 lever ranking (§3.3–3.4) |
| `ffi_host_platform_backend_matrix_2026-07-10.md` | the 7-row GPU+CPU backend timing/equivalence matrix | **the** FFI-vs-native table; rows 1–5 of §4.1 come from it |
| `bse_cublasmp_recon_WORKLOG_2026-07-20.md` | 2-D distributed V_Q reconstruction | spec for a capability **absent from the current tree** (§4.5) |
| `bse_cublasmp_recon_NMU2_DISTRIBUTION_2026-07-20.md` | n_μ² sharding audit, 9/9 | directly serves the "no N_μ² tile on one rank" scaling target |
| `bse_kgrid_2026-07-20.md` | `bse_k_grid` coarse→fine interpolation | the economy route: coarse GW → fine BSE without a second NSCF |
| `bse_HANDOFF_ARBITRARY_Q_2026-07-18.md` | arbitrary-Q program handoff | settled rulings not to relitigate; branch/worktree map; trap list |
| `htransform_window_sweep_ht_lean_2026-07-20.log` | 1-GPU 6×6 / 1496c window sweep | independent htransform timing + the far-band QP blow-up (§2.6) |

Pre-existing here and relevant: `bse_multinode_HLO_AUDIT_2026-07-20.md` (§3.8),
`suite_speedup_2026-07-15.md` (§4.1 row 19),
`memory_model_hlo_calibration_2026-05-17.md`, `XPROF_TRACE_GUIDE.md`.

---

## 8. Where the archive is silent — stated, not inferred

* **MKL / DFTI / cuFFT FFI margins.** Nothing. The backend did not exist on
  Perlmutter (§4.0a). The 3.78× figure is unverifiable from here.
* **A ~7.3× GEMM margin.** Zero hits. The only GEMM-FFI A/B is −5 % (§4.1 row 11).
* **`slate_vs_cusolvermp_bench.py` output.** The script is referenced six times
  across the reports; **no results, log or table survives.** The single most
  conspicuous missing measurement.
* **`phdf5_vs_gather_bench` output.** The "8×" that seeded the FFI-wins folklore
  (§4.1 row 17) has no preserved run.
* **Any production-scale (CrI3-class n_rmu) FFI-vs-native linalg comparison** —
  which `ffi_host_platform_2026-07-10/report.md:211-212` itself names as the
  prerequisite for real discrimination.
* **Any measured cuBLASMp fused-W-solve vs JAX-native-LU timing** — the path was
  ABI-dead for two months (§4.6).
* **Any single-node FFI-vs-native margin on the BSE path.** All BSE FFI
  comparisons are 4-node/16-GPU; the cross-node latency confound is unremovable.
* **Any Frontera / sm_75 / fp64-limited measurement.** None — the archive predates
  the move.
* **A c64 matvec A/B.** The "biggest single lever, ~2×" was never run (§3.4).
* **Any xprof or HLO of `bse_stack_matvec` at production 12×12 scale.** Only
  fixture/inflated 1×1 and 2×2 on 4 GPU (§3.8).
* **Cost attribution *inside* `htransform_psi_cQ` at n_μ=2412.** The 24.5 s/Q is a
  single driver tick; how it divides between the `_q_batch` eigh, `build_fH_R` and
  the f-transform is unknown. The only sub-split ever published is the 4-GPU
  n_μ=640 one.
* **The cost of the 32× SPMD involuntary rematerialization** on the production
  ψ bundle (§2.5).
* **Any sweep of the htransform q-batch size** — `batch=32` is a constant in every
  log from 2026-07-18 on.
* **A scaling model for htransform per-Q cost in n_μ.** Two points only (§2.1).
* **A benchmarked cuBLASMp BSE matvec.** Rejected on structure, never timed (§3.7).
* **A batched-across-q FFI eigh.** [PROPOSED] as the fix for the 6.7×; never built.
* **Verdicts on runs B and C.** They completed and wrote output; no report
  adjudicated the numbers (§1.5).
* **Exciton eigenvalue data.** No `.dat`/`.npz` anywhere. To get the old curves
  numerically you must re-run.
