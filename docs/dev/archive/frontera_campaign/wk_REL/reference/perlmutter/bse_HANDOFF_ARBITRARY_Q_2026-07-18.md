# HANDOFF — arbitrary-Q V_Q interpolation + exciton bandstructure program

_2026-07-18. Fresh-agent entry point for the BSE interpolation / exciton-bands
work. Read this, then the read-order below. Everything cited is committed to the
sandbox repo unless marked otherwise. The GW-side program context lives in
`report.md`/`MAP.md`/`PLAN.md` in this directory; this file covers the
arbitrary-Q arc specifically._

## 0. One-paragraph state

The bare-exchange tile V_Q[μν] can be produced at arbitrary Q with NO r_tot
object and no solve at the target: Tikhonov-cleaned SR tiles (Fourier stencil)
+ a global n_μ×26-coefficient LR model ("b26p") evaluated closed-form. Validated
0.4–0.6% on physical matrix elements, ≪0.1 meV against per-Q refit ground truth
at off-grid Q. On top of it: a production exciton-bandstructure pipeline
(single-compile lax.scan over a high-sym Q path, both kernels: on-grid W direct
+ interpolated-V_Q exchange) delivered MoS2 Γ-M-K-Γ bandstructures at 3×3-grid
and 12×12-grid training. Production default for accuracy-critical work remains
the per-Q ζ refit; the interp scheme is the fast/amortized candidate.

## 1. Read order (physics/method)

1. `archive/designs/ARBITRARY_Q_PRIMER.md` — SELF-CONTAINED problem+method
   primer (ISDF fit, nearsightedness spine, all rulings). Start here.
2. `archive/designs/F_SCHEME_NOTE.html` — the presentable PR-style walkthrough
   with formatted equations (also published as a claude.ai artifact).
3. `archive/designs/arbitrary_q_bse.md` — the living technical record:
   §§1-2 htransform contract + finite-Q dataflow; §3 ζ-structure refutations
   (+§3.5 falloff study); §9 literature survey + owner rulings (§9.8 g0
   winding → finite-α split); §10-11 campaign + off-grid; §12 tile schemes +
   operator theory + Wannier spine; §13 the b26p compact-LR result (+§13.6
   consolidation record); §14 stress tests (if the stress agent landed it —
   check).
4. `archive/designs/ARBITRARY_Q_PRIMER_RESPONSE.md` — external counterproposals
   (adjudicated: frames dead §12.3/12.5, fit-based extraction became b26p).
5. `archive/designs/"Decay-Properties-Spectral-Projectors-2013.pdf (1).md"` —
   Benzi–Boito–Razouk, the operator-theory backing.
6. `PHASE2_LOG.md` — chronological log of everything (search the § titles).
7. `CLEANUP_LOG.md`, `BSE_DELETION_LEDGER.md` — what was deleted/kept and why.

Key settled rulings a new agent must NOT relitigate: no r_tot objects in
interpolation machinery; no eigenvector-frame transport (gapless spectrum —
theorem); no literal real-space multipole moments; no |Q|²-multiply of summed
tiles; the exchange is DENSE in (k,k′) (adjudication in
`archive/adjudication/VERDICT.md`); ridge-ζ is NOT default-safe for GW Σ
(`reports/zeta_ridge_ab_2026-07-17/`).

## 2. Code — branches and worktrees (lorrax_A repo)

| branch | worktree | contents | state |
|---|---|---|---|
| `agent/bse-phase2` | `sources/lorrax_A` (main tree) @ c4c349f | B1 dense exchange, stack matvec, W-resolvent+finite-q+W(ω) chain, pair-amp hoist, per-q recompile fix, non-TDA solvers (bse_nontda.py), P1 lanczos fixes | the integration base |
| `agent/bse-exciton-bands` | `sources/worktrees/lorrax_A_exciton_bands` | **src/bse/vq_interp.py** (production b26p backend) + **src/bse/exciton_bands.py** (Q-path driver) + compute_wfns_fi(q_list) | the pipeline; production runs use THIS worktree |
| `agent/bse-bands-perf` | `sources/worktrees/lorrax_A_bands_perf` @ 2e90edb | perf pass on the pipeline (trainer 5.4×, sharded htransform batch, chunk clamp) — bit-identical outputs | UNMERGED into exciton-bands; merge point 9fca293 |
| `agent/screening-degeneracy-fix` | (no worktree) | degeneracy-rounded screening window + degeneracy gate | atop f19136e, unmerged |
| `agent/bse-comms-opt` | `sources/worktrees/lorrax_A_comms_opt` | exchange single-sourcing (40→12 collectives) + nt-dispatch | unmerged; P-NT premise needs post-P2 re-measurement |
| `agent/bse-phase2-zeta-ridge` | `sources/worktrees/lorrax_A_ridge_wt` | zeta_ridge_eps knob (default OFF) | unmerged; NOT default-safe |

Merge topology decision pending the owner. Never commit to main; lorrax_D is
another session's — untouchable.

Reference (pre-production) implementation + its acceptance test:
`runs/MoS2/A_bse_w0_resolvent_2026-07-16/primer_response_study/REFERENCE_arbitrary_q_vq.py`
(+ `test_reference_e2e.py`, `README.md` = the scratch-script ledger for that
whole study dir). vq_interp.py must keep reproducing its pinned numbers.

## 3. Run data (what trained/validated what)

| run dir | contents |
|---|---|
| `runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/` | 3×3 fixture restart (640 centroids) — trained the ORIGINAL delivered bandstructure; also the campaign fixtures |
| `runs/MoS2/A_bse_w0_resolvent_2026-07-16/` | the whole resolvent/interp study: fixture_run (gnppm restart), full_basis, interp_study (falloff), primer_response_study (campaign, tile schemes, b26p, reference impl), arbitrary_q_recon |
| `runs/MoS2/B_exciton_bands_2026-07-17/` | FIRST bandstructure deliverable (3×3-trained, 32 pts) — exciton_bands_GMKG.{dat,png} + WORKLOG |
| `runs/MoS2/04_mos2_12x12_bands_2026-07-18/` | the 12×12 production run: qe/ (144-k NSCF), 00_lorrax_cohsex (144-q restart, 640 centroids, nval26/ncond54/nband80), 01_lorrax_exciton_bands (40-pt scan + endgame diagnostics incl. the D_min(Q) Λ-valley dip investigation), 05_htransform_spbands + 02/03_*_1000c variants (in flight at handoff) |
| `runs/MoS2/A_bse_bands_perf_2026-07-18/` | perf-pass session record (WORKLOG, probes, xprof) |
| `runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/` | symmetry/degeneracy investigations (diag/FINDINGS.md, diag2/FINDINGS2.md) + Si restarts used by falloff/3D tests |
| `runs/MoS2/A_bse_nontda_2026-07-17/`, `runs/Si/B_zeta_ridge_covariance_2026-07-17/`, `reports/zeta_ridge_ab_2026-07-17/` | non-TDA prototype; ridge A/B artifacts |

Restart h5s are on scratch, uncommitted (repo convention). The fixture
`centroids_frac_640.txt` is shared 3×3↔12×12 deliberately (isolates k-grid from
basis effects).

## 4. In flight at handoff (check task state before touching)

1. 12×12 640c endgame: dip window-guard A/B + deliverable assembly (overlay
   PNG, timing table) — agent on `01_lorrax_exciton_bands`.
2. SP-bandstructure + D_min(Q) panel + 1000-centroid variant (02/03/05 dirs).
3. Stress tests §14 (α budget, 3D/Si, Q→Γ edges, robustness) — may have landed;
   check arbitrary_q_bse.md for §14.
Owner's open wishes on record: smoothness question (basis vs physics — the
D_min panel + 1000c run answer it); W(ω) physics workshop (w_omega_chain.py is
a frozen MVP pending owner sign-off); merge decisions.

## 5. Traps (all in KNOWN_SANDBOX_ERRORS.md — read the 2026-07-1x entries)

rk-unwrap q-labeling (worth 155×); sphere-center wrap at half-boundary q;
fresh worktrees lack liblorrax_ffi.so (copy from lorrax_A build dir); Lmod
modules broken in non-interactive shells (use the module-free salloc+srun+
shifter pattern in `cleanup_verify/` scripts); container numpy BLAS is
single-threaded (~3 GFLOPS — use --cpus-per-task + OMP_NUM_THREADS on host
stages); interactive per-user allocation limits (attach with srun --overlap to
the lx-alloc-jackm job instead); never bare python3 on login nodes; foreground
GPU ≤10 min, longer runs as tracked background tasks; never idle-wait.

## 6. House rules that bind this work

Feature branches only; suite green before commit (golden gates:
test_gw_jax_regression cohsex/si_cohsex_3d/gnppm + test_ibz_equals_full_bz);
per-element math for new contractions; single-source (no parallel old/new
paths); no new classes; physical metrics (M†VM, exciton shifts) are the verdict
variables — tile Frobenius is a diagnostic only; disk-log provenance for every
number (phantom-table lesson); report deletions in a ledger the owner can read.
