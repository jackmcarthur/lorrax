# Driver reference

**Eight** drivers have sections on this page. Seven of them form the chain, in order:
input generation (centroids, dipole, kin-ion) → GW self-energy → QP bandstructure
interpolation → BSE → exciton bands. The eighth, [downfold](#downfold-gwdownfold_cli),
sits beside the chain rather than in it: it compresses a finished GW run for the BSE
stages and nothing upstream of them.

Two user-facing drivers in the chain's orbit have **no section here**, which is worth
knowing before you go looking for them. `gw.eqp_bgw` is the fastloop chain's
`eqp-convert` stage, named in the smoke-test line below and documented nowhere on this
page; `postprocess.rotate_wfn_to_qp` is what turns a GW run's `qp_wfn_rotations.h5` into
the `WFN_qp.h5` that the htransform section tells you to feed it. Deck keys and defaults
are verified against `gw_config._DEFAULTS` as of 2026-08-01; the complete key list is in
[input_reference.md](input_reference.md).

All eight start through `runtime.initialize_communicator_stack` (one module-top
call: mesh + clique warm-up + startup report), so every driver is launchable under
the certified P=16 geometry — `config/frontera/templates/gw_dev.sbatch`
(srun --mpi=pmi2 + apptainer + `impl=mpi`, 8 nodes × 2 ranks × 28 threads) with the
module swapped in. `--help` and bad argv are answered before that call in
`gw_jax`, `kin_ion_io`, `downfold_cli` and `kmeans_cli` (`runtime.cli_seam`);
the other four still bring the runtime up first. Smoke test for any change to the first five drivers: the
fastloop mini-deck chain (kmeans→dipole→kin-ion→gw→eqp-convert→htransform dft+qp)
checks every stage against pinned references at P=1 AND on a 2×2 host-device mesh
in ~2–5 min: `cd /scratch2/08271/jackmc/lorrax_sandbox && sbatch
fastloop/run_fastloop.sbatch` (or `bash …` inside any allocation; it reads the
LIVE repo src by default, `LORRAX_SRC` overrides). Exit 0 = pass, 1 = numeric
drift (per-file deltas printed), 2 = a stage failed (the log names the driver).
BSE and exciton bands are NOT in the chain yet.

> **THE FOUR-GPU RULE — every GPU verification leg for any driver on this page
> runs at P=4.** A P=1-only verification is never sufficient for landing; unit
> and CPU cells are exempt. The owner's rationale, verbatim: *"use four gpus
> for 100% of all testing so that never ever do we run something on one GPU and
> then learn it doesn't generalize later"*. The fastloop chain above checks P=1
> and a 2×2 host-device mesh, which is a portability check and not the P=4 leg.
> On Perlmutter ask for the whole node with `-G 4`; see `AGENT_PREAMBLE.md` at
> the repository root.

## centroids — `centroid.kmeans_cli`

Selects the N_c real-space ISDF interpolation points: a density-weighted k-means over the WFN FFT grid
(optionally on symmetry-orbit representatives so the set is closed under the crystal point group, recovered
from the density rather than the possibly-truncated WFN group), snapped to the grid, then an oversampled
candidate pool is pruned to N_c by pivoted Cholesky on the pair-density Gram over the sigma band window.
It does NOT read the deck: it opens `WFN.h5` (fixed name, cwd) and optionally the QE `<prefix>.save` density;
the sigma window is resolved from the WFN (`n_val = wfn.nelec`, `n_cond = wfn.nbands - n_val`) or the
`--prune-n-*` flags. Output: `centroids_frac_<n>[<suffix>].txt`, consumed by the GW run via deck keys
`centroids_file` (default `centroids_frac.txt`) and, for the bispinor current channel, `centroids_file_current`.

Invoke: `python3 -m centroid.kmeans_cli N --orbit --qe-save <path>.save [--out-suffix _x]`; single node
(`/scratch2/08271/jackmc/mos2_4x4_test/deck_b300.sbatch` step 3) or multi-process — certified at P=16:
67→31 s wall (2.2×), accepted 2979c set sha256-IDENTICAL to P=1, rank gate 270/270 (scorecard BD.2, jobs
7884867/7884870; the thread-main refusal seen there is closed — mesh + warm-up now come from
`initialize_communicator_stack`). What does not scale: the replicated full-r-grid weight build (~7 s) and
single-mesh-axis ψ-at-centroids sharding (√P class) — see BD.4.
Reuse: the `.txt` carries no stamp; content is guarded downstream by the GW run (centroid-table md5 on the
restart tensors + element-wise compare against `zeta_q.h5`), so regenerating centroids invalidates ζ and
tensor reuse there. Fastloop stage: `kmeans` (~26 s of the chain).

| flag | default | meaning |
|---|---|---|
| `N_c` (positional) | 400 | number of centroids after pruning |
| `--oversample` | 1.5 | k-means runs at ceil(N_c*x), pruned back; 1.0 disables pruning |
| `--prune-n-cond` | `nbands - n_val` (full WFN conduction window) | Cholesky window conduction extent; the pre-2026-07-29 `min(n_val, nb-n_val)` default was a 30%-rank-deficiency bug |
| `--prune-window` | `v_x_vc` | Gram band pair: `v_x_c` (legacy) / `v_x_vc` (adds v×v for V_H) / `vc_x_vc` (full sigma square, use when ncond >> nval) |
| `--centroid-weight` | `band_range` (scalar mode) | k-means weight: sigma-window Sum|psi|^2 vs occupied-only `charge_density` (slabs: occupied-only gives zero vacuum support, V_H sign-wrong) |
| `--density-mode` | `scalar` | `scalar` = charge channel; `current` = Gordon-current weight for bispinor transverse channels (suffix `_current`) |
| `--orbit` / `--no-orbit` | auto (on if n_sym>1) | symmetry-closed centroid set from orbit representatives |
| `--rho-power` | 1.0 | weight^alpha; centroid density ~ w^(0.6*alpha) |
| `LORRAX_CENTROID_RANK_TOL` (env) | 0.01 | rank-gate tolerance; lowering it is a deliberate override |
| `LORRAX_CENTROID_SELECT` (env) | `deliver` | what the select does when the pool is numerically flat but still has candidates; `strict` restores the 2026-08-07 refusal |

Invariant: the selection window must span the sigma window `[0, nelec+ncond)` the GW run consumes; the default
is a superset of any deck's `ncond`, so a shortfall means it was narrowed explicitly. Main refusal: "FATAL:
pivoted-Cholesky rank deficiency" (certified rank < requested orbits) — check `--prune-n-cond` against the
deck's sigma window first, since that gate is a PROXY for a prune-window mismatch and not a general accuracy
statement; `vc_x_vc`, `--oversample`, or a lower N are the other levers, and `LORRAX_CENTROID_RANK_TOL` is the
named override for a set you have MEASURED. Also fatal: FFT-grid mismatch rho vs WFN (needs ecutrho = 4*ecutwfc).

Read the `[point rank]` lines, not only the `[rank gate]` PASS: in orbit mode the gate counts ORBITS and says
nothing about the points in the file it blesses. Rank deficiency in the delivered POINT set is reported and
does not refuse — it is measured anti-correlated with BerkeleyGW agreement on the Si anchor deck.
Policy: [`docs/dev/rank_truncation_policy.md`](dev/rank_truncation_policy.md) §7.

## dipole — `psp.get_dipole_mtxels`

Computes velocity matrix elements `<mk|v|nk> = p + i[r, V_NL]` per Cartesian component (momentum
`sum_G (k+G) c*c` plus the nonlocal-pseudopotential commutator, sign-flipped to the BGW convention) and the
`deltaE = E_b - E_b'` table, for all full-BZ k. Reads the deck (`-i`, default `cohsex.in`) for
`wfn_file`/`nval`/`ncond`/`nband`/`bispinor`; writes `dipole.h5` (`dipole_cart` (3,nk,nb,nb), `deltaE`),
consumed by `gw.head_correction` for the q->0 head S(omega) in Sigma_SX/Sigma_COH. Reuse contract: root
attrs `prov_wfn_sha256` (SHA-256 over the WFN eigenvalue table + k-list — the DFT solution's identity, not
the file bytes) plus `prov_{nval,ncond,nband,nb_written,wfn_file}`; `check_dipole_provenance` warns/refuses
when a regenerated WFN left a stale `dipole.h5` of the right shape but wrong contents (an unstamped
pre-guard file also fails).

Invoke: `python3 -m psp.get_dipole_mtxels -i deck.in [--out dipole.h5]`; multi-process capable (k-partitioned
sweep, rank-0 write); certified 1-proc in `deck_b300.sbatch` step 5 and at P=16 (both datasets EXACT vs P=1,
job 7884867). P-scaling: 0.8× wall at P=16 on the b300 deck — the k-sweep itself scales but the whole P=1
compute (~10 s) cannot amortize distributed startup; run small decks at P=1 (BD.2, wontfix). The replicated
`(nk,3,nb,nb)` gather before the rank-0 write is the open large-nb bound (BD.4). Fastloop stage: `dipole`
(~16 s of the chain).

| flag | default | meaning |
|---|---|---|
| `-i` / `--input` | `cohsex.in` | deck; band count = min(WFN bands, max(nelec+ncond, deck `nband`)) |
| `--vnl-mode` | `analytic` | nonlocal velocity via analytic dZ/dK vs `numeric` central FD on V_NL(k) |
| `--vnl-h` / `--vnl-h-rel` | 1e-5 / 0.0 | FD step for numeric mode (absolute; relative to median |K| if larger) |
| `--vnl-num-scheme` | `naive` | FD scheme; `richardson` extrapolates two steps |
| `--skip-vnl` | off | write p-hat only — matches BGW `use_momentum` for apples-to-apples absorption |
| `--with-finite-q` | off | also write `finite_q/` group: `rho_cvkq`, symmetrized `v_cvkq` = (v_R+v_L)/2 with umklapp G-lookup, `kminq_idx` — for the SOS chi head/wing/S/w pipeline |
| `--iq-list` | all 0..nk-1 | reduced-BZ q indices for `--with-finite-q` |
| `--out` | `dipole.h5` | output path |

Main new-user failure: the downstream provenance check — "generated from a DIFFERENT DFT solution or band
window than this run" after any WFN regeneration (fix: rerun this driver); an unstamped pre-guard file also
fails the check. Missing pseudopotentials in the deck dir / `../qe/{scf,nscf}` fallback is the other stopper.

## kin-ion — `gw.kin_ion_io`

Computes the ionic one-body Hamiltonian matrix `<mk|T + V_loc + V_NL|nk>` for every full-BZ k and, by default,
the exact FFT-grid mean-field Hartree `<mk|V_H|nk>` — the pieces of H0 = kin_ion + V_H, a ~500 eV cancellation
(MoS2: -502 + 461 eV), from which the run reconstructs Vxc = E_dft - kin_ion - V_H. Reads the deck (`-i`) as
the single source of truth (`sys_dim` -> Coulomb truncation, `nval`/`ncond`/`nband`, `bispinor`), plus `WFN.h5`
and the `*.upf` pseudopotentials; writes `kin_ion.h5` with two datasets: `kin_ion` (pristine T+V_loc+V_NL) and
`v_hartree` (separate, so one file serves both exact-V_H and ISDF-V_H runs). The V_H build is distributed:
rho partitioned over (k, band-chunk) with exactly one psum, the Poisson solve replicated by design, and
`<mk|V_H|nk>` k-partitioned with one gather; the exact route is pinned to QE's `kih.dat` at rms 1e-4 eV,
while the ISDF `V_q[0]` route plateaus at ~0.1-0.5% relative — up to eV-scale on the gap (4.15 eV measured).

Invoke: `python3 -m gw.kin_ion_io -i deck.in -o kin_ion.h5 -n NB [--no-hartree]`; multi-rank capable
(rank-0 write after gather); certified 1-proc in `deck_b300.sbatch` step 4 (`-n 300 --hartree`) and at P=16:
40→14 s wall (2.9×; driver-recorded 24.5→6.7 s), `kin_ion` EXACT vs P=1, `v_hartree` max Δ 2.13e-14 (psum
order), k-sweeps near-ideal (vh_matrix_k 8.45→0.80 s at 16 k / 16 ranks) — BD.2, job 7884867.
Reuse contract: dataset attrs `nk`/`nb`/`sys_dim`/`truncation_2d`/`pseudopotentials`/`has_hartree`/
`hartree_truncation_2d`/`input_file`/`wfn_file`. NOTE no WFN content hash (unlike dipole): a regenerated WFN
with a stale `kin_ion.h5` is NOT auto-detected — rerun this driver after any WFN regeneration. Fastloop
stage: `kin_ion` (~6 s of the chain).

| key / flag | default | meaning |
|---|---|---|
| `-n` / `--nb` | max(deck `nband`, nelec+ncond) | bands written; HARD FLOOR nelec+ncond — the run's `load_kin_ion_submatrix` reads `[0, nelec+ncond)` |
| `--no-hartree` | off | skip V_H entirely; the GW run must then use `hartree_source = isdf` |
| `--fold-hartree` | off | LEGACY: add V_H into `kin_ion` values, stamp `has_hartree=True` (old-artifact reproduction only) |
| `--sys_dim` | from deck (default 2) | may only CONFIRM the deck's `sys_dim`; contradiction is fatal |
| `--pseudo_dir` | deck dir (falls back `../qe/{scf,nscf}`) | where `*.upf` live |
| `hartree_source` (deck, read by GW run) | `auto` | the G-space vs ISDF V_H switch: `stored` (require `v_hartree`) / `isdf` (V_q[0] tile) / `gspace` (rebuild exact in-run) / `auto` (stored if present, else legacy fold, else isdf) |
| `kin_ion_file` (deck) | `kin_ion.h5` | file the GW run reads back |

New-user refusals: `--sys_dim contradicts sys_dim=... in deck` (fix the deck); `Requested N bands but the
deck's sigma window needs nelec+ncond = ...` (raise `-n` or the WFN band count); "launcher advertises P tasks
but jax.distributed joined a world of 1" (broken distributed launch would make every rank clobber the output).

## gw — `gw.gw_jax`

Computes the GW quasiparticle correction: ISDF ζ-fit + bare Coulomb V_q(μ,ν) → minimax-quadrature χ₀ → per-q Dyson solve W = (1−Vχ₀)⁻¹V → Σ_x ⊕ Σ_c (static COHSEX or GN/HL plasmon-pole with 4-branch τ-integration and analytic q→0 head) → eigh of H_QP = kin_ion + V_H + Σ_xc, with BGW-style degenerate-set averaging.
Consumes `WFN.h5`, `centroids_frac.txt`, `kin_ion.h5` (+ `centroids_file_current`/`tmp/v_q_bispinor.h5` when bispinor); produces `tmp/zeta_q.h5` (+`zeta_q_mu{1,2,3}.h5`), the restart tensor file `tmp/isdf_tensors_{n_rmu}.h5` (V_qmunu, G0, enk_full, psi_full_y[_transverse], W0 + head scalars for BSE — plus `psi_full_y_mun` under `low_mem_bands = true`, additive to `psi_full_y` and read only by that same key on restart; the schema BSE/downfold read, `psi_full_y`, is unchanged either way), and `eqp0.dat`, `eqp1.dat`, `sigma_diag.dat` and `qp_wfn_rotations.h5` (`eqp_g0w0.dat` in PPM one-shot). `write_eqp2=true` additionally writes `eqp2.dat`: a fixed-Sigma eigenvalue-self-consistent ladder obtained by rotating the one-shot full Sigma(omega) table into each updated QP basis and rediagonalizing until max-|dE| over protected/non-scissored states is at most 1 meV by default. It does not rebuild screening, W, or Sigma diagrams. Protected states must remain covered by the omega grid; optional out-of-grid semicore states preserve `E-E_F`, and optional high conduction states use the in-grid affine scissor. A mandatory final map reapplies both after the putative final QP rotation before the file is accepted.

**k-basis of the energy files — every block states its own k, so join on the coordinate, never on the block position.** `eqp0.dat`/`eqp1.dat` are on the **IBZ wedge** (one block per `wfn.kpoints` entry; the coordinate is the BGW `(3f13.9,i8)` block header, as BerkeleyGW writes it). `sigma_diag.dat` and `eqp_g0w0.dat` are on the **full BZ** (one block per `sym.unfolded_kpts` entry), and each block carries a `# kcrys kx ky kz` line. The bases are deliberate and not interchangeable. Htransform consumes the columnar IBZ `eqp1.dat` directly and routes its unfold through the symmetry service as specified below; it does not require a pre-unfolded positional converter. Pairing LORRAX blocks against BerkeleyGW's IBZ blocks **by position** has produced a wrong answer three times — on Si 4×4×4 the map is `[0,1,2,5,6,7,10,27]`, so positions 0–2 agree and only diverge from position 3, and the worst instance reported 291 meV where the truth was 28 meV. `tests/test_eqp_kpoint_basis.py` pins all of this.
Two files this list used to name are not written by this driver, measured 2026-08-10 on a COHSEX run of `tests/regression/si_bse_debug`: `sigma_mnk.h5` has its only producer in `ppm_pipeline.py`, so a `compute_mode = cohsex` run writes no such file however `sigma_omega_h5_file` is set; and `WFN_qp.h5` is written by `postprocess.rotate_wfn_to_qp`, a separate driver with no section on this page, which reads the `qp_wfn_rotations.h5` this one does write. The htransform section's `--wfn-file WFN_qp.h5` therefore names a file the documented chain never produces on its own — run `python -m postprocess.rotate_wfn_to_qp WFN.h5 qp_wfn_rotations.h5` between the two stages.
Band windows (`wavefunction_bundle.BandSlices` from `Meta.from_system`): the σ/QP window is always [0, nelec+ncond) — nval moves only the interior edge b1 = nelec−nval, never the window bottom; `nband` sets the χ₀/Σ band-sum top b4 = round_up(nband, world_size) (pads are zeroed).

Invoke: `python -m gw.gw_jax -i gw.in` under the certified Frontera CLX launch block `config/frontera/templates/gw_dev.sbatch` (srun --mpi=pmi2 + apptainer + impl=mpi CPU collectives; typical 8 nodes × 2 ranks × 28 threads = P=16; full-pipeline parity jobs 7884609/7884612). Demonstrated end-to-end by `/scratch2/08271/jackmc/mos2_4x4_test/valsmoke_tmpl.sbatch` and `gw_ht_b300.sbatch` (their explicit `LORRAX_FFT_FFI=1` + `LORRAX_FFT_FFI_FUSED=1` exports are redundant since 2026-08-01 — the FFI stack is the required default).
P-scaling at P=16 (b300 deck, 2979c/300b): the ζ charge factor executes q-parallel above nq·mu³ ≥ 5e9 — `zeta_fit.cholesky` 104.4 → 11.8 s (8.9×), GW wall 335.9 → 214.6 s, bit-identical (job 7885024); `distributed_zeta_solve = distributed` runs wall 222.1 s with no O(mu²) replica, parity ≤ 4.9e-5 eV gauge-class (job 7885077) and wins outright at large mu (mu=10015: factor 4712 → 236 s on 64 ranks); the Σ τ-loop is device-bound (its d2h_wait is the kernel wall surfacing in the drain), boundary flushes now once per branch (jobs 7885105/7885109).

| key (deck unless noted) | default | meaning |
|---|---|---|
| `nband` / `nval` / `ncond` | 100 / 5 / 5 | χ₀ band-sum top / interior valence edge / σ conduction count; σ diagonals for [0, nelec+ncond) |
| `compute_mode` | auto | Σ ansatz: `x_only` \| `cohsex` \| `gn_ppm` \| `hl_ppm` \| `mpa`; auto never infers `mpa`. `mpa` is declared but REFUSES AT DRIVER ENTRY until one real-material chi/W/fixed-head/Sigma/QSGW disk calculation passes end to end — see [Multipole frequency integration](theory/THEORY_mpa_implementation.md). |
| `qp_solver` | auto→one_shot_dft | `one_shot_dft` (one-shot full-matrix effective H, Sigma at E_DFT with QSGW Hermitian symmetrization; distinct from fixed-state diagonal G0W0) \| `fixed_point` (on-shell solve) \| `self_consistent` (QSGW loop — **static Σ only**: refused at driver entry beside a dynamic `compute_mode`, because the SC finalize leaves the full Σ_c(ω) cube in the QP basis by design while H/Σ_x are rotated to DFT, so their diagonals cannot be combined. Pair it with `cohsex`) |
| `write_eqp2` | false | On a dynamic one-shot run, additionally iterate the fixed full-matrix Sigma(omega) in the evolving QP basis and write BGW-format `eqp2.dat`; screening/W/Sigma diagrams remain fixed, and repeated eigensolves use `distrib_la` (native batch-reshard while tiles fit, distributed above the `sc_eigh` memory threshold) |
| `eqp2_tol_ev` / `eqp2_max_iter` | 0.001 / 20 | max-|dE| cutoff over protected/non-scissored states (1 meV) / iteration cap; convergence must survive one final post-rotation map, and nonconvergence or an uncovered required state refuses before writing |
| `eqp2_accelerator` / `eqp2_history_depth` | rcrop / 5 | rCROP on the Hermitian Hamiltonian carried in the original DFT basis (so eigenvector gauge is absent from the residual); `linear` selects unaccelerated Picard iteration |
| `restart` | true | reuse `tmp/isdf_tensors_{n_rmu}.h5` (skip ζ-fit/V_q); stamp contract below |
| `ppm_probe_chi_reuse` | off | opt-in `auto`: probe χ₀ folds into ONE fused τ sweep on static+k augmented nodes; pays only where per-node χ cost dominates (nets +1.2 s at b300 — planner cost; job 7885109) |
| `hartree_source` | auto | THE G-space vs ISDF V_H switch: `stored` (v_hartree array in kin_ion.h5) \| `gspace` (exact FFT-grid rebuild) \| `isdf` (V_q[0] quadrature) \| legacy `folded` = V_H inside kin_ion values; auto resolves stored→folded→isdf |
| `charge_zeta_solve` | rank_truncate | charge ζ CCT conditioner: rank-revealing eigh pseudo-inverse dropping λ < `zeta_rcond`·λmax (default 1e-8) vs bit-identical historical `cholesky`. The cut is GATED, not merely announced: when it discards anything and the achieved κ_eff exceeds the certified 1e8, the run REFUSES — [`docs/dev/rank_truncation_policy.md`](dev/rank_truncation_policy.md), override `LORRAX_RANK_POLICY` |
| `distributed_zeta_solve` | auto | ζ back-solve tier: `replicated` \| `per_q` \| `distributed` (nothing O(μ²) replicated; needs rank_truncate + square/1-D mesh); auto = replicated under 4 GiB gather cap, else per_q |
| `w_dyson_solver` | auto=local | W Dyson plan: `local` per-q pivoted LU in the q-sharded map vs `distributed` 2-D block-cyclic backsolve (ScaLAPACK/cuSOLVERMp); refuses loudly, never downgrades |
| `sigma_omega_layout` | replicated | Σ_c(ω,k,m,n) cube: `sharded` keeps (m,n) mesh-tiled end-to-end, for every `qp_solver`; refuses at resolve time when the σ window does not divide both mesh axes, or h5py_allgather at P>1. The ω cube exists only under a dynamic `compute_mode`, which `self_consistent` refuses at driver entry (see `qp_solver`), so the layout choice is live for `one_shot_dft` and `fixed_point` only |
| `low_mem_bands` | false | `gw.wavefunction_bundle` `layout="face"`: two 2-D-sharded ψ copies (`psi_nmu`/`psi_mun`, `2*S/(Px*Py)`/rank) in place of the legacy four single-axis copies (`2*S/Px + 2*S/Py`/rank). Default `false` (`layout="legacy"`) is bit-identical to every deck written before this key existed — measured bit-identical against unmodified `origin/main` on real 4-rank CUDA for the supported envelope (`claims/0429.md`). Ported: the carrier + restart; the G builder (`greens_function_kernel.build_G`/`build_G_tau`), the band projector (`common.contract_bands.contract_bands_block_reshard`) and the STATIC Σ channels only (`cohsex_sigma`'s `x_only`/`sigma_sx`/`sigma_coh`/`hartree`) via a planned `distrib_la.gemm_plan`, and the ordinary χ₀ minimax kernel in `w_isdf` — G/projection/Hartree algebra parity measured on real 4-rank CUDA to ~1e-16 (`claims/0428.md`); a real end-to-end `compute_mode=cohsex` run confirmed the same (`claims/0429.md`). Also ported: `head_correction=full`'s q→0 head/body wings and the static metallic wings (`qsgw_head.head_wings_sharded`/`static_head_wings_sharded`, `layout='face'`), via a mu-blocked band-pair gather (`_head_wing_kernel_face`) rather than the legacy full-band-replicated-psi ring — parity vs legacy + an independent NumPy oracle, and a real-CUDA memory gate isolating the kernel's OWN marginal contribution to the allocator peak (measured zero, separately from the psi bundle's own O(mu) construction cost) across a 16x local-mu-block increase, both on real 4-rank CUDA. The ISDF ζ fit itself (`gw.isdf_fitting.fit_zeta_to_h5`) does not yet hold psi in the face layout during Stage A-D (it converts only after the fit returns); this session (2026-08-22, `claims/0432.md`) freed the fit's own Y-form single-axis centroid copies right after CCT — real, layout-independent, halves that stage's psi floor — but did not give the fit's surviving X-form operand true all-P product sharding (named remaining work, needs a band-sharded/all-reduce redesign of `c_q_from_psi_sm`/`z_q_from_psi_sm`). **`compute_mode = gn_ppm \| hl_ppm \| mpa` (any dynamic Σ_c(ω) mode, insulating MPA included) is NOT ported and refuses by name** (`low_mem_bands_dynamic_ppm_unported`) — discovered on real hardware after an earlier version of this row (following the audit guide's own landing-order text) implied GN-PPM was supported. Narrow envelope while QSGW rotation/exact-response/bispinor consumers are ported one at a time; an unsupported combination refuses by name — see `reports/gwjax_low_mem_bands_audit_2026-08-22/report.md`'s §6 envelope, `KNOWN_LORRAX_ISSUES.md`, and `docs/input_reference.md`'s fuller row |

Restart/reuse contract. Two independent caches. (1) `tmp/isdf_tensors_{n_rmu}.h5` (V_qmunu, G0, enk_full, psi_full_y[_transverse], W0 + `W0_ready`): reused only when the band-window/n_rmu/kgrid attrs match AND the `centroids_charge_md5`/`centroids_transverse_md5` content hashes match — the quadrature BASIS, not just its size ("same count, different points" refuses). Its small `qp_state_source_provenance` record carries the canonical WFN fingerprint scheme/hash for the matched `psi_full_y`/`enk_full` carrier plus that WFN's positive QP stamp, if any. `file_io.qp_wfn` alone makes, serializes, parses and compares this record; the restart writer only transports opaque bytes. A current restart refuses a changed WFN at every state-join seam; a legacy unstamped restart remains usable on the incumbent no-external-QP path. (2) `tmp/zeta_q.h5`: reused only when `zeta_is_done` is set and the `fit_provenance` JSON is byte-identical — n_rmu, band ranges, the charge pair-training domain (`ordered_lr_plus_rl` for asymmetric serving windows), bispinor, both cutoffs, EFFECTIVE `zeta_rcond`/`zeta_ridge` (env overrides applied), `charge_zeta_solve`, `gamma_contract_mode`, `write_ibz_only`, band_norms, fft_grid, ecutwfc/ecutrho, wfn_file+wfn_bytes, and (2026-08-01) the `distributed_zeta_solve` GAUGE tier: 'replicated' | 'distributed' (per_q collapses to replicated — same factor bits). A stamp missing the tier key is legacy replicated (announced, reusable); a pre-change asymmetric charge stamp lacks the non-legacy pair-domain key and refits, while equal-window charge and transverse stamps remain unchanged. Mesh/P are deliberately excluded: a ζ fit at P=4 is reusable at P=80. `LORRAX_FORCE_REFIT=1` forces a refit; every mismatch costs compute, never correctness.
FFI gates are env vars, not deck keys, and the FFI layer is REQUIRED (`docs/architecture/decisions.md` 2026-08-01): `LORRAX_FFT_FFI` (MKL/cuFFT flat-k FFT backend), `LORRAX_FFT_FFI_FUSED` (fused IFFT·(G·W)·FFT τ kernel), `LORRAX_BANDS_GEMM_FFI` — all default ON; grammar 0/off/false/no, 1/on/true/yes (a stale `auto` resolves to the default with a grammar note); a missing/unloadable handler refuses at startup naming the `.so` (`Gate.enforce` via `runtime.initialize_communicator_stack`); `=0` refuses for the FFT dial (the XLA duplicate is deleted) and is an announced uncertified debug opt-out for the other two (src/ffi/gate.py; per-knob table in `docs/dev/env_vars.md`).
Large-N_mu / fully distributed operation — every distributed key per stage, per-rank memory scalings, what is still replicated, the auto-thresholds and the certified example jobs: `docs/dev/large_nmu_operation.md`.
New-user failure modes: restart refusals naming a changed band window or centroid table ("same count, different points") → rerun `restart = false`; bispinor runs refuse on missing `psi_full_y_transverse`/`v_q_bispinor.h5` (Σ^B would silently drop); sanity gates kill runs with non-negative Σ_x diagonals or NaN kin_ion/Σ before eigh; sharded-layout ValueErrors quote the divisibility/backend fix.
Fastloop stage: `gw` (~40 s cold) — the mini-deck exercises the standing bare-launch path for SlabIO's availability probe (the `slab_io` deck key was removed 2026-08-06; there is one transport).

## downfold — `gw.downfold_cli`

Compresses a FINISHED GW calculation onto a smaller ISDF centroid basis, so that BSE and exciton-band work on a few-dozen-band window runs in a basis sized for that window rather than for Σ. It reads the parent run's `isdf_tensors_<mu_L>.h5`, selects μ_S of the parent's centroids by pivoted Cholesky against the RETAINED band window, solves the least-squares transfer `T = S_SS⁻¹ S_cross` in the PAIR-DENSITY metric (the three Grams are `isdf.core.c_q_from_psi_sm` with its ψ operands sliced to the window — no new kernel, no second ζ fit, no WFN read), and writes `V_qmunu`, `W0_qmunu`, the `_nohead` twins where the parent had them, `G0_mu_nu` and a sliced `psi_full_y` through the congruence `A_S = T A_L T†` into a restart bundle **in the unchanged format at the smaller μ**. The child inherits the parent's canonical QP source-state record verbatim; it never infers a new state from the compressed carrier. `bse.bse_jax` is therefore a zero-change drop-in: point it at the output directory and nothing else moves.

**`bse.exciton_bands` is not a plain reader**, and it took three fixes to make it a drop-in. Measured 2026-08-10 on a 936→189 downfold of a Si 4×4×4 parent, it broke three independent ways while `bse.bse_jax` read the same bundle fine; the asymmetry has one cause, which is that `bse_jax` only READS the stored tensors while this driver REBUILDS objects in the same ISDF basis. (1) It reads a `centroids_file`, which the small bundle only has if the downfold was given `parent_centroids_file` — so it now takes the parent's table off the bundle's own `downfold_provenance`, with the deck as a fallback, and uses `keep_idx` to identify the child's coordinate rows. (2) The one published whole-state randomized-QRCP basis is selected from full Bloch states by `isdf.galerkin` and evaluated directly at those child rows; centroid count is not a second state-space rank authority, and there is no exact parent-fit alternate. (3) Off-grid exchange interpolates a stored `zeta_q.h5`, which the downfold now transports as `zeta_S = conj(T) zeta_L` — the head vector's map at every G rather than only at G=0, which is exactly the map under which V rebuilt from ζ is the congruence the bundle already stores. All three fixes carry a red twin in `tests/test_exciton_bands_downfold_dropin.py`. `enk_full`, the head scalars and the parent's Coulomb-policy and band-window stamps ride through verbatim; the band axis is NOT truncated (the window is the FIT window, not a band cut — truncating it would renumber every band index and move the stamp `assert_restart_window_matches` refuses on).

Takes its OWN input file (`[downfold]`), not a GW deck — a `[cohsex]` section is refused by name, and unknown keys are REFUSED rather than warned about. Full key reference and the reasoning behind each default: `docs/downfold.md`. `--print-schema` lists them.

Three numbers on every run, and they are not interchangeable — and **none of the three is an accuracy gate**, which is why the run now ends by printing the observable comparison that is (parent and small bundle, same deck, same flags, compare the lowest eigenvalue; production BSE wants better than 1 meV). The three: the EIGENVALUE rank of the retained window's Gram (what μ_S is validated against — the driver REFUSES when you ask for more directions than the window holds, and `mu_small = auto` sets it to exactly that, sizing by rank rather than by accuracy), pivoted Cholesky's SELECTION certificate (necessary, not sufficient; ~3× larger at the same nominal tolerance — `DOWNFOLD_RANK_PROBE.md` §7, a campaign report not carried in this repository), and `eps_W(q)`, the per-q Pythagorean error bar. Note that the "~3×" is a statement about the two ranks *at one tolerance*, and the defaults do not put them there: `downfold_select_tol` defaults to the kernel's √ε ≈ 1.49e-8 while `downfold_rcond` defaults to 1.1e-6, so on a default run the printed pair can agree exactly — measured 2026-08-10, both read 189 — without anything being wrong. The last one is exact rather than an estimate: the fit is an orthogonal projection, so `‖𝒲‖² − ‖𝒲_S‖² = ‖𝒲 − 𝒲_S‖²` and `eps_W = sqrt(1 − ‖𝒲_S‖²/‖𝒲‖²)` is computable from μ×μ traces with no reference calculation and without ever forming the N×N observable. A ridge would destroy the orthogonality that makes it exact, which is why no ridge is applied anywhere on this path.

Invoke: `python3 -u -m gw.downfold_cli -i downfold.in`. Mesh = the run's square startup mesh, or `--px/--py` (square only, `RUNTIME.reshape`).

| key / flag | default | meaning |
|---|---|---|
| `source_restart` | **required** | the parent GW run directory, or its `isdf_tensors_<mu>.h5`. REFUSES on more than one bundle in the directory rather than taking the newest — this driver is the one most likely to create that ambiguity |
| `output_restart` | **required** | output DIRECTORY; writes `<dir>/tmp/isdf_tensors_<mu_S>.h5`. May not equal the source |
| `band_range_left` / `band_range_right` | **required** (or `n_val`/`n_cond`) | the retained window, `lo:hi`, half-open, ABSOLUTE band indices. Equal for BSE work. Asymmetric = the Σ-serving shape: accepted, announced as unvalidated, ~2× the μ_S |
| `mu_small` | **required** | an explicit integer IN POINTS, validated by comparing the observable against the parent's. On a parent that carries a centroid source map the selection runs in whole symmetry orbits and this request is FLOORED to the largest union of whole orbits that does not exceed it (`mu_S requested 185 points -> REALIZED 168`, printed loudly, both numbers also stamped in `downfold_provenance`); the floor spends less and never rounds up, and it is taken against the rank ceiling as resolved, so the realized count never exceeds either your request or the ceiling. `auto` (= the eigenvalue rank at `downfold_rcond`) is a rank CEILING and is NOT recommended: it sized the 2026-08-10 walk to 189 and the lowest BSE eigenvalue came out 2.09 eV wrong with `eps_W` at 1.3e-2 and nothing refusing, so it now prints a loud warning at selection time and again at the end of the run |
| `downfold_rcond` | `1.1e-6` | relative eigenvalue cut on S_SS = a cap on pseudo-inverse amplification, NOT a gap-finder. The measured 20-band ceiling is ~190 directions on two decks; at 1e-8 and below the "rank" tracks the pool size rather than the physics |
| `downfold_select_tol` | `sqrt(eps)` | pivoted-Cholesky stopping tolerance. NOT the same knob as `downfold_rcond` |
| `mode` | `cur` | subset selection from the parent's centroids. `refit` (fresh narrow-window k-means + a second ζ fit) is REFUSED, not demoted |
| `plan` | `auto` | `auto`/`local` = the local plan. `distributed` (μ over a 2-D `distrib_la` grid) is later work and REFUSES rather than demoting — a block-cyclic factorisation is a different numerical gauge |
| `report_residual` | `true` | compute `eps_W`. Two GEMMs at μ_L per q; leave it on |
| `residual_refuse_above` | none | refuse to WRITE when the worst-q `eps_W` exceeds this |
| `parent_centroids_file` | none | when given, writes the kept rows as a sibling centroid table and stamps its md5; without it the bundle carries no `centroids_charge_md5` (and says so) rather than inheriting the parent's, which would name the wrong points |

Failure modes, all loud: `mu_small` above the window's eigenvalue rank refuses and prints the measured ceiling with three named fixes; a parent whose `W0_ready` is False refuses (that dataset is the all-zeros placeholder, and downfolding it would pass every shape check); a parent with no `kgrid` stamp refuses (the Gram build is a convolution over k and cannot guess the split). The output bundle is deliberately indistinguishable by SHAPE from a natively fitted one, which is what makes it a drop-in; what it carries in `downfold_provenance` so that a reader can still tell is listed on [the downfold page](downfold.md).

Does NOT apply to plasmon-pole or multipole reductions: `B_q` is a residue and would transform, but `Omega_q` is a pole POSITION per matrix element and no change of basis maps a table of pole frequencies. Downfold the linear objects and re-fit the pole model in the small basis.

## htransform — `bandstructure.htransform`

Hamiltonian-transformation bandstructure interpolation to an arbitrary k-path from coarse-grid data. `isdf.galerkin` selects the published shared whole-state basis from stacked full-Bloch states with deterministic randomized QRCP, factors the selected physical states exactly, and projects all states into that one alpha gauge. The canonical `PsiGStore` and WFN transforms stream the G-flat→real-space work; centroids are only registered evaluation points for the fitted basis. The driver forms `fH_k = Σ_n f(ε_nk)c_nk c_nkᴴ`, applies the canonical flat-k IFFT to obtain `fH_R`, and recovers path energies with the existing batched eigensolver plus the archived Newton inverse. `--qp-rotations qp_wfn_rotations.h5` is the full quasiparticle-Hamiltonian path: it consumes the matched `U_mnk,E_qp` artifact through `file_io.qp_wfn` and rotates only the compact Galerkin state rows, so the unchanged builder represents `f(H_QP)=U f(E_QP) Uᴴ`. `--eqp-file` is deliberately the cheaper diagonal approximation: it changes energies in the current WFN band labels and cannot represent off-diagonal QP mixing.

When the fitted window contains outer DFT guards, the authenticated QP block
must extend strictly above the returned window: `n_qp_corrected > n_return`.
The exact corrected range comes from the canonical rotation artifact's
`band_range`, never from a deck count or an array-shape guess.  From those
corrected compact rows and the same flat-k transform, the driver forms
`P_A(k)=Σ_{n<n_qp_corrected}|c_nk><c_nk|` and its Fourier interpolant
`P_A(q)`.  For `fH(q) u_j(q)=λ_j(q)u_j(q)`, it selects the complete corrected
block by active character `p_j=<u_j|P_A(q)|u_j>`, energy-orders that block,
and publishes only its lowest `n_return` interior states.  Every path point
must have a positive physical-energy separation, above the canonical
multiplet tolerance, between the returned top and every nonreturned fitted
state.  Otherwise the run refuses: a merely positive character gap makes a
pointwise assignment deterministic but does not make active/guard switches
path-continuous.  If the artifact corrects the complete fitted window there is
no active/DFT boundary and the ordinary full-H energy ordering remains the
exact path.  Eigenvector phase cannot reach the file; the character operator
is face-sharded like `fH_R`, and no rank-squared object is gathered or
replicated.
At the host publication seam the driver asks the symmetry service to identify
every path point that is periodically identical to a coarse-grid row.  It
reports max/RMS returned-spectrum differences and the worst path/coarse/band
indices over the complete intersection.  This is an accuracy receipt, not a
replacement tolerance: the shared whole-state projection is approximate and
the independent fine-QE oracle still decides acceptance.  A plotted “Exact
Gamma” marker is emitted only when Gamma is actually on both grids; neither
coarse row zero nor a nearest path point is assumed to be Gamma.
Consumes the same deck as gw_jax ([cohsex] keys + `K_POINTS {crystal_b}` path), `WFN.h5` (or `--wfn-file WFN_qp.h5`), `centroids_file`; writes `bandstructure.dat` (VBM shifted to 0, energies in eV; rank 0 is the only writer). Mutually exclusive `--basis-output` and `--basis-input` publish or require the reusable mesh-independent fit artifact; `isdf.galerkin` owns both the basis and its stamped SlabIO lifecycle. Distributed-key coverage (`eigh_backend`, `use_low_mem_eigh`) and the per-rank memory story: `docs/dev/large_nmu_operation.md`.
The returned band window is `(nelec−nval, nelec+ncond)`. Standalone output
requires `nval=nelec`, hence it begins at absolute band zero; internal BSE
consumers retain explicit partial-window contracts. The VBM/Fermi index is
resolved relative to the absolute window start. Standalone htransform fits
`--guard-bands` additional conduction
bands above the returned window, because the top of the fit window lies on the
f-transform's zero shoulder and is not a valid output band.  The eqp override
must supply every returned and guard band at every k.  `bandstructure.dat`
records both the returned absolute window and the wider fit window in its
header.

Invoke: `python -m bandstructure.htransform -i ht.in [--qp-rotations qp_wfn_rotations.h5 | --eqp-file eqp1.dat]`; the production scientific report is `htransform.out` and the interpolated table is `bandstructure.dat` (override with `--report-file` / `--output-file`). Kernel diagnostics use the driver-wide `LORRAX_DEBUG_PRINT=1` switch. The whole-state memory ledger prints and checks the exact mesh-dependent live set before compilation; do not reuse performance or rank expectations from the deleted centroid-Gram implementation.

**Whole-state memory gate.** `isdf.galerkin` owns one bounded, zeta-style outer-r/inner-band stream for the randomized sketch, exact selected-state Gram, and physical projection. Its canonical `PsiGStore` supplies full-Bloch G-flat→r chunks; no full-r basis or second WFN/FFT route exists. The precompile ledger prices each alternative stage, including compiled WFN transform workspace, candidate/sketch faces, selected-state rows, factor, and coefficients; it reduces the WFN band carrier and then the r carrier before refusing against the worst-rank allocator budget. `LORRAX_GALERKIN_CHUNK_GIB` controls the stream tile only and never changes candidates, pivots, or delivered rank.

The optional restart stores only logical-rank fitted payloads and reconstructs exact-null carrier padding for the reader mesh; stream chunks and padding are scheduling metadata, not persisted physics.

`K_POINTS {crystal_b}` block format, required by this driver and by `bse.exciton_bands` and specified in no other page: the header line, then a count of path corners, then one line per corner giving three fractional (crystal) reciprocal coordinates and the number of points to the **next** corner, with `#label` comments optional. The last corner takes a count of 1. A Γ→X→M path on a 4×4×4 grid, used for the 2026-08-10 exciton-band run:

```text
K_POINTS {crystal_b}
3
  0.0000 0.0000 0.0000 2  #gG
  0.5000 0.0000 0.0000 2  #X
  0.5000 0.5000 0.0000 1  #M
```

Choose the per-segment counts with the consuming driver in mind: `bse.exciton_bands --vq-mode ongrid` refuses any path Q that is not on the coarse grid, so on a 4×4×4 grid the corner spacing has to divide into steps of 0.25 — the two-interval segments above land on 0.0, 0.25, 0.5 exactly. `--vq-mode interp` accepts arbitrary Q but needs full-BZ `zeta_q.h5` **and a slab cell** — on a 3-D deck the arbitrary-Q mode is `--vq-mode refit`. Note that `bse.exciton_bands` no longer takes these counts as the last word: it applies `--q-per-segment` (default 16) as a floor, so a deck like the one above draws 16 Q per segment unless you ask for fewer. The single-particle driver on this page is unaffected and still reads the counts verbatim.

`--eqp-file` format (`read_eqp_energies`): **LORRAX's own `eqp1.dat`, passed directly.** BerkeleyGW columns on the **irreducible wedge** — one `(3f13.9,i8)` crystal-coordinate header per `wfn.kpoints` entry (10 blocks, not 16), each followed by that many `(2i8,2f15.9)` band rows in eV. The block coordinates are checked against the deck's own wedge (a file from another deck, or in another k-order, is refused rather than landing its energies on the wrong k), the absolute band labels place the columns in the sigma window, and the **unfold to the full BZ happens inside the driver through the symmetry service** (`symmetry_maps.star_broadcast`, via its single adapter `file_io.kin_ion.broadcast_ibz_to_full_bz`). eV→Ry conversion is the reader's. A missing or unparseable `--eqp-file` is a FATAL SystemExit, never a silent DFT fallback.

`--qp-rotations` format: the canonical `qp_wfn_rotations.h5` written by GW, including full-BZ (or symmetry-service-unfoldable wedge) `U_mnk`, matched `E_qp_nk_rydberg`, absolute `band_range`, `kgrid`, and full-BZ crystal coordinates. The artifact's complete QP block must lie inside the fitted htransform window and, when outer DFT guards remain, must extend above the returned window so the returned-interior gate can be evaluated. Wider fitted rows remain DFT states/energies, matching the canonical QP-WFN writer's block-identity convention; a partial overlap is refused because slicing `U_mnk` is not unitary. `--qp-rotations`, `--eqp-file`, and an already-stamped `WFN_qp.h5` are alternate state descriptions, never layers to stack. Association is explicit: the presence of a similarly named artifact beside an eqp file is not proof that they share a source. **Remaining provenance gap:** the rotation artifact currently has no source-WFN fingerprint. The consumer validates its absolute band range, k-grid, and full-BZ coordinates against the selected WFN, but cannot prove exact source-WFN identity; adding that stamp belongs to the canonical writer format, not to htransform.

Before 2026-08-15 this required a *pre-unfolded* full-BZ text file (`nk == sym.nk_tot`) whose `k-point <K>:` blocks were paired to k **by position**, produced by an out-of-tree `make_eqp_htformat.py` that did the IBZ→full unfold by hand. Both the positional pairing and that converter are gone — the converter has no job left, and hand-rolled unfolding is not permitted outside the symmetry service.

| key / flag | default | meaning |
|---|---|---|
| `nval` / `ncond` | 5 / 5 | returned window `(nelec−nval, nelec+ncond)`. Standalone bandstructure output refuses unless `nval=nelec`, because an omitted occupied lower boundary can recover at sampled k yet ring between them. Internal BSE interpolation retains its explicit partial-window contract. |
| `htransform_rank_multiplier` | 20 | search ceiling `ceil(multiplier*N_band)` for the sole published whole-state randomized-QRCP basis. `htransform_qr_eps` selects delivered rank; this is not an `rtol` or requested final rank. `0` is only an archived spelling of 20. No per-k gauge repair or exact-span alternate exists. |
| `--basis-output` / `--basis-input` | empty | mutually exclusive immutable write/read lifecycle for the gauge-coupled `ctilde`, node basis and compact selected-state factor; input never refits, output never reuses, exact numerical identity is validated, and logical extents are repadded for the current mesh |
| `eigh_backend` (key) / `--eigh-backend` (CLI overrides) | auto | fH_q eigensolver for the get_centroids_fi handoff: auto\|off = q-batched native eigh; `distributed`/`cusolvermp`/`slate`/`scalapack` spread one `(rank,rank)` tile over the mesh (wide windows; square mesh, one process/device). The whole-state QRCP basis has no Gram eigensolve. |
| `use_low_mem_eigh` | false | same axis by intent: true + auto ⇒ distributed; true + off refused at parse |
| `get_centroids_fi` | false | BSE handoff: also compute fine-grid ψ at coarse centroids via `bse_setup.compute_wfns_fi` |
| `kgrid_fi` / `wfn_fi_min` / `wfn_fi_max` | "" / 0 / 0 | fine k-grid "nx ny nz"; sub-window on the htransform band axis (0 = full window) |
| `wfn_fi_q_chunk` | 0 = N_q_coarse | fine-q chunk per f(H(q)) build; floor, rounded to device count |
| `--a-band` | top band | band whose bandwidth sets the f-transform scale a |
| `--guard-bands` | 4 | fit this many extra conduction bands above the returned `nval+ncond` window; the returned bands must pass the shared f-shoulder gate, while the guards absorb the transform's exact-zero top edge |
| env `LORRAX_GALERKIN_CHUNK_GIB` | 6 | bounded whole-state r-stream tile budget; changes chunk count, never fitted basis semantics |

Failure modes: a standalone occupied-band refusal means `nval` omitted lower
occupied states; include them rather than publishing a Hamiltonian with an
uncontrolled lower spectral boundary. A QRCP search-saturation refusal means
the rank criterion reached the configured search ceiling; inspect its
projection receipts before increasing `htransform_rank_multiplier`. An
`f-shoulder` refusal means a requested output band is absent from fH at some
coarse k; add guard bands, do not disable the gate. Newton inversion stops as
soon as the archived global residual contract is met and refuses if the
50-step cap leaves `max|f(x)-y| > 1e-12 Ry`. The all-coarse FFT round-trip is
an enforced transform invariant; the spectrum/energy values are necessary
approximate-projection diagnostics. Neither is a positive locality
certificate: a Fourier interpolant can reproduce every sample and still ring
between samples. FATAL on `--eqp-file` not found/parsed;
sanity refusals on NaN ctilde/E_nk or E_nk spread > 272.11 eV.

## bse — `bse.bse_jax`

Solves the Bethe-Salpeter equation for neutral excitations: the electron-hole Hamiltonian H = D + V − W (D = quasiparticle transition energies ε_c−ε_v, V = bare exchange, W = statically screened direct term) in the pair basis |vk,ck⟩, in the ISDF μ-basis on a 2D (μ,ν) device mesh. It consumes the GW restart `isdf_tensors_<N_mu>.h5` (newest in the run dir wins; datasets `psi_full_y`, `enk_full`, `V_qmunu`, `W0_qmunu` with `W0_ready` attr) plus optionally `eqp1.dat` for QP energies, and produces exciton eigenvalues and, with `--write-eigs`, a BGW-layout `eigenvectors.h5` (eV; valence axis reversed on write; rank 0 is the ONLY writer — every-rank writes raced and truncated the file at P=64, job 7879470). All sharded solves (Lanczos/Davidson/FEAST) apply H through the one trial-stack matvec (`bse_stack_matvec`): a shard_map whose body scans the trial axis so exactly one T-tensor is alive regardless of block width; the `--matvec-kind ring|gather|simple` flag survives in the CLI but the sharded eigensolve path ignores it (retired — see the "legacy `matvec_kind` selector is retired here" note in `bse_lanczos.py::solve_bse_sharded`, and the retirement plan in `bse_stack_matvec`'s module docstring). The `bse_k_grid` input key densifies the bundle (ψ/ε via one htransform fH, W by zero-pad in R = exact trig interpolation, through `bse_io.make_w_densifier` — the ONE sharded densifier, per-rank peak = the local (μ,ν) tile) from the coarse restart grid to a fine grid before any solve. The q=0 exchange tile `V_q0` is *not* densified by default: its body is built from the centroids and the G-sphere alone and is therefore k-grid-invariant, and only its rank-1 head scalar `<v>_mBZ` depends on the grid (the cell is BZ/N_k). Setting `head_minibz_average = true` opts into rebuilding the tile through `vq_interp` with the fine mini-BZ head, which replaces the deck's disk body and its injected `vhead`. Before 2026-08-09 that rebuild ran unconditionally on this path, ignoring the key.
Reuse contract: consumption-side only — the newest-mtime `isdf_tensors_*.h5` wins (log line names the losers) and screening is gated solely by the `W0_ready` attr; NONE of the GW band-window/centroid stamps are re-verified here, so point the driver at the run dir whose physics you mean. One exception is representation provenance: an external `--eqp` is accepted only when the restart's canonical source-state record proves it was built from the same selected, unrotated WFN. A legacy restart with no record still runs without an external QP state, but `--eqp` on it refuses instead of guessing. Distributed keys it shares with the chain (`eigh_backend`, transport): `docs/dev/large_nmu_operation.md`.

Which bundle you point it at is a cost decision as well as a physics one, and once a GW reference exists there are two ways to spend it. The first is the one everything above describes: consume the parent restart in the basis Σ was sized for — hundreds of bands' worth of centroids — and pay that size again on every solve, every parameter change and every Q path. The second is to compress the parent once with the downfold driver above, whose input file is documented in [downfold.md](downfold.md), and then run the cheap studies against the smaller bundle. The downfold selects μ_S of the parent's centroids against the band window the BSE will actually consume and writes a restart in the unchanged format at the smaller μ, so this driver reads it with no flag, no code change and no knowledge that a compression happened, while every (μ,ν) tensor it stores and multiplies shrinks quadratically. Measured on an over-complete silicon parent: 960 centroids down to 191 on a 20-band retained window — 5× in μ, 25× per tensor — through this driver unmodified.

What that costs belongs in the same breath, because it is bought rather than free. The machinery itself is exact: a downfold that keeps every centroid over the full band window reproduces the parent's own lowest twenty excitons to **0.010 meV** MAE end to end, which is round-off through the pseudo-inverse and nothing else, so every meV past that floor is spent deliberately, by shrinking μ_S. The 5× demo above is an aggressive cut and it moves those twenty eigenvalues by **37.4 meV** MAE. Production BSE work wants better than 1 meV, and only a full-frequency MPA-class study has any business tolerating ~10 meV — so size μ_S for the accuracy you need rather than for the compression you want, and settle it the way this codebase settles every cut, by sweeping it and taking the plateau in the observable. The per-q `eps_W` the downfold prints is a tripwire for a compression that has gone badly wrong, not a transferable statement about meV: one and the same one-per-cent error bar covered that 37 meV drift on the over-complete parent and a 1.7 eV drift on the shipped one.

One condition on all of this is settled before the GW run rather than after it. Downfolding wants an OVER-COMPLETE parent, because redundancy on the retained window is the only thing there is to throw away — so run the GW stage at a generous μ_L if you intend to compress it afterwards; the shipped `si_bse_debug` 480-centroid production set holds no redundancy at all on a 20-band window, and downfolding that destroys the spectrum instead of shrinking it. The other is enforced at the downfold itself, and from the opposite side: μ_S is validated against the EIGENVALUE rank of the retained window's Gram, the driver refuses above the ceiling it measured and quotes the number, and it prints that rank on every run beside pivoted Cholesky's selection certificate — roughly three times larger at the same nominal tolerance — so the two cannot be read as one.

What the downfold does not yet buy is a cheaper Σ. The compression is faithful to the window it was fitted on, and that window is the BSE's shape: the direct and exchange kernels both contract ψ legs lying inside the retained window, so a symmetric fit covers them exactly. Σ is the other shape — its internal band sum runs over the full window while its outer projection does not — and the asymmetric window that would express it is accepted by the schema and announced as unvalidated, with no end-to-end Σ gate behind it. Re-fitting Σ in a downfolded basis is real, buildable, later work; today the small bundle serves the retained-window BSE and the exciton bands built on it, and nothing upstream of them.

Invoke: `python -u -m bse.bse_jax -i cohsex.in --lanczos ...` in the GW run directory. Single node = 1 process, mesh the square s×s over `jax.device_count()` (non-square counts refuse — decisions.md 2026-08-01); multi-node uses the certified `config/frontera/templates/gw_dev.sbatch` geometry (srun --mpi=pmi2, 2 ranks/node, taskset 28 cores/rank, apptainer + `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`) with the module swapped in for `gw.gw_jax`.

| key / flag | default | meaning |
|---|---|---|
| `--lanczos` | off | REQUIRED for the Lanczos/Davidson eigensolve; without it the driver dispatches FEAST (`bse_feast`) |
| `--bse` / `--rpa` | RPA | kernel: default (neither flag) is RPA D+V; `--bse` enables the screened −W direct term |
| `--tda` | off | Tamm-Dancoff (resonant only); default is full non-TDA via `bse_nontda` (definite-pencil solve) |
| `--n-val` / `--n-cond` | 4 / 4 | pair-basis band window; loader clamps over-requests; valence auto-detected by mean ε<E_F (`--n-occ` overrides) |
| `--band-degeneracy` | `strict` | what happens when a window boundary lands inside a degenerate multiplet (a Kramers pair under SOC+TRS, any irrep of dimension > 1 — half a multiplet is not a subspace of anything). `strict` refuses and names the counts that would work; `snap` widens the window OUTWARD to the multiplet boundary and says so loudly; `off` proceeds on the cut multiplet. The default has been `strict` since the owner's ruling of 2026-08-10: `snap` shipped as the default with `824032b7` and within a day had silently turned the `si_bse_debug` BerkeleyGW anchor's requested 4v4c into 4v8c, which the gate reported as an 0.0906 eV code regression that no branch had caused (`tests/KNOWN_FAILURES.md`). A widened window is a different calculation rather than a repair, so widening is something you now ask for by name. The "same multiplet" tolerance is `--degeneracy-tol-ry`, 1 meV by default |
| `--solver` | `lanczos` | `davidson` = diagonal-precond per-state convergence (oscillator strengths); `lanczos` = spectrum shape; `trlan` = thick-restart Lanczos (`solvers/thick_restart_lanczos.py`), a bounded-memory restart of the same Krylov space |
| `--block-size` / `--max-lanczos-iter` | 1 / auto | block width / TOTAL Krylov dim bound (block iters = max/bs) |
| `--n-reorth` | −1 | reorth window; −1 = full reorth (needed for degenerate spinor spectra; small windows breed ghosts). Since the CGS2 default (2026-08-08) a narrow window is **not** faster — the batched route costs `2·max_iter` collectives whatever the window — so ghosts are the only remaining reason to widen it and there is no speed reason to narrow it |
| `--davidson-precond` | `bare` | Davidson preconditioner diagonal. `bare` divides by `dE = E_c − E_v`; `exact` divides by the assembled `diag(H)` (what BerkeleyGW hands PRIMME), assembled once per payload at identical per-iteration cost; `auto` picks on transition-space dimension at `bse_davidson_helpers.EXACT_PRECOND_AUTO_MIN_DIM`. **That crossover is a placeholder awaiting a large-deck measurement, not a measured value** — see `PRECOND_BUILD_FREE.md` "what would settle it", a campaign report not carried in this repository |
| `--davidson-olsen` | off | Olsen-correct the preconditioned direction (project out its component along the current Ritz vector). Two batched inner products per iteration; this is what makes a small `--davidson-eps-shift` safe |
| `--davidson-eps-shift` | 1e-3 Ry | Davidson preconditioner regularisation (= 13.6 meV) |
| `--davidson-m-max` | 10·n_eig | max Davidson subspace dimension before restart (measured knee). Memory is `2·m_max` trial vectors — Davidson stores both `V` and `HV` |
| `--trlan-m-max` | max(3·n_eig, 60) | thick-restart Lanczos basis slots — the memory cap that makes the solve fit at production `bse_dim`, paid for in extra matvecs. Useful range 3–5× n_eig |
| `--trlan-n-keep` | n_eig+10 | Ritz vectors retained across a thick-restart. Must satisfy `n_eig ≤ n_keep < m_max` |
| `--matvec-kind` | `ring` | **INERT on the sharded eigensolve path** — every sharded solve builds `bse_stack_matvec` regardless (`bse_lanczos.py`, "the legacy `matvec_kind` selector is retired here"). Still parsed, and still steers `bse.absorption_haydock`. Whether it should refuse or be deleted is an open owner decision |
| `--ring-timing` | off | parsed and inert. Kept parseable on purpose so archived launch scripts do not `SystemExit`; for per-term timings use `timing.section` on the jitted matvec |
| `--write-eigs [N]` | off | write `eigenvectors.h5` (rank-0 single-writer; non-TDA adds `eigenvectors_coupling` Y) |
| `--eqp FILE` | none | **Diagonal QP approximation**, not a full `U f(E) U†` state: IBZ `eqp1.dat` (BGW columnar, as LORRAX GW writes it — one block per `wfn.kpoints` entry, k-coordinates in the block header): re-slice bands on QP energies (n_occ re-resolved). Unfolded onto this driver's full-BZ k-axis by `bse_io.apply_eqp_corrections` **through the symmetry service** (`symmetry_maps.star_broadcast` via `file_io.kin_ion.broadcast_ibz_to_full_bz`); `input_file` is required, because the star tables come from the deck's `SymMaps`. The restart fingerprint must match that deck WFN and both must be unrotated; legacy/unproved, mismatched, or already-QP carriers refuse. Until 2026-08-15 this row said "full-BZ" (wrong), and passing `input_file=None` selected a mean-field-energy nearest-match at 0.01 eV — bespoke unfolding, now deleted. |
| `bse_k_grid` (input key) | `""` | "NX NY NZ" fine grid; each axis at least the coarse extent, with no integer-nesting requirement (8x8x1→12x12x1 evaluates the coarse Fourier polynomial directly). Empty = coarse bundle byte-identical. W's nonnested Γ-cell head overlap is constructed on the exact LCM common grid rather than by floor-divided cosets. |
| `LORRAX_BSE_MATVEC_OPT` (env) | unset | only token is `gspmd`, a default-off audit route that drops the manual `shard_map`; unknown tokens REFUSE.  `yhoist` was made **permanent** and `krep` **removed** on 2026-08-08 — setting either now refuses |

Failure modes: a loud banner "W0_qmunu not ready — falling back to BARE COULOMB V" means GW screening never ran — the spectrum has no excitonic screening; `FileNotFoundError: isdf_tensors_*.h5` means you are not in (or did not point `-i` at) a GW run dir; multiple restarts are resolved newest-mtime; forgetting `--bse` silently gives RPA.
Not in the fastloop chain: the fastest smoke is this driver on a mini GW run dir (e.g. a fastloop `check.<jobid>/<leg>` dir) single-process with `XLA_FLAGS=--xla_force_host_platform_device_count=4` — a real 2×2 mesh, no MPI.

## exciton bands — `bse.exciton_bands`

Computes the finite-momentum exciton dispersion E_S(Q) — the TDA BSE H_Q = D_Q + V_Q − W in the pair basis |vk, c k+Q⟩ — along the `K_POINTS crystal_b` path in the input (same block/parser as the htransform bandstructure driver; refuses without it). Q-shifted conduction eigenpairs ψ_c(k+Q), ε_c(k+Q) come from ONE interpolated htransform fH (`bandstructure.bse_setup.compute_wfns_fi` with q-list {k+Q}; q-axis chunking via the `wfn_fi_q_chunk` input key, default 0 = one chunk sized like fH_R); W is the unchanged coarse-grid FFT convolution (all k-differences stay on-grid); exchange V_Q is interpolated (`vq_interp`, tile momentum wrap(−Q); G=0 KEPT at finite Q, BGW energy_loss convention; production q=0 tile at Γ). The whole path is ONE jitted `lax.scan` of block-Lanczos over the stack matvec (one compile for all Q). Consumes the same `isdf_tensors_*.h5` (+ optional diagonal `eqp1.dat` on BOTH legs); writes `<prefix>.dat` and `<prefix>.png` (rank 0 only, followed by a collective barrier — without it ranks exiting early got the job SIGABRT rc=134 after correct outputs, job 7882507). Before the htransform leg, its WFN fingerprint must match the stored restart state. There is no `--qp-rotations` path: full-QP use requires a restart generated from the same stamped `WFN_qp.h5` the deck selects; otherwise the driver refuses rather than mixing DFT and rotated legs.

Why that head handling is what it is — the nonanalytic G=0 exchange term, the longitudinal–transverse splitting it produces along the Q path, and the measured domain in which `head_minibz_average` is the right object rather than an approximation to the exact pointwise value — is [The long-range exchange head and LT splitting](theory/lt-exchange-head.md).

Invoke: `python -u -m bse.exciton_bands -i cohsex.in --n-val 4 --n-cond 4 --n-eig 6 [--vq-mode both --refit-points 0,8,15]`; never on a login node. Same single-node (1 process; omitted `--px/--py` = the run's mesh, an explicit shape must consume the whole device count) vs multi-node story as bse_jax: the certified geometry/transport is `config/frontera/templates/gw_dev.sbatch` (module swapped in); `/scratch2/08271/jackmc/mos2_4x4_test/valsmoke_tmpl.sbatch` certifies the shared W-densifier (pre-B HLO probe: no all-gather + parity).

| key / flag | default | meaning |
|---|---|---|
| `--vq-mode` | `interp` | Where the finite-Q exchange tile comes from. `interp` — the b26p model, fast (one jitted evaluator, per-Q dispatch-only) and **slab-only**: its per-\|G_z\| channels exist only on a q_z=0 slab, so a `sys_dim = 3` deck is refused by name at the model build. `refit` — a per-Q ζ refit from the htransform ψ, contracted with the kernel the **producer** used (`gw.compute_vcoul` + the deck's mini-BZ head slot on a bulk deck). Nothing in it is 2-D, so this is the arbitrary-Q exchange a 3-D crystal runs, not merely a checking mode; it is the expensive one by design. `both` — interp on the whole path plus refit at `--refit-points` spot checks, solved in the SAME compiled scan so the ΔE_S table is apples-to-apples. `ongrid` — no model at all, the stored `V_qmunu[wrap(−Q)]` tile: exact, needs no ζ, works with IBZ-only ζ, but only at Q on the coarse exchange-tile grid |
| `--refit-window` | `zeta` | Which band window the `refit` re-fits ζ' on, and therefore **which gate certifies the run**. `zeta` — the producer's own ζ-fit window, so the refit reproduces the stored `V_qmunu` tiles and is certified BY that tile identity (`vq_interp.refit_ongrid_null`). It needs `n_mu,parent · n_s ≥ nk · nb_ζ` for the htransform Galerkin leg, which a wide GW window can exceed (3840 against a 1920 basis on the Si 4×4×4 / 960-centroid / 60-band lineage, where `build_fH_R` correctly refuses). `bse` — fit ζ' on the DECK's window instead (the BSE window plus its conduction guards), which carries every pair density the exchange will ask for and drops the bound with it. The price: ζ' ≠ ζ, the stored tiles do NOT come back, and the tile null is unavailable — so the certification moves up to the CONTRACTED object, every path Q that lands on the coarse tile grid solved twice in one scan (refit exchange vs producer's tile) and compared eigenvalue by eigenvalue. Only meaningful under `--vq-mode refit`; anything else refuses |
| `--cert-grade` | `reference` | Which NAMED tolerance the `--refit-window=bse` contracted certification is held to. `reference` = 0.01 meV, the gate a published NUMBER must clear. `visualization` = 1.0 meV, for a deliverable that is a PICTURE — an exciton band structure whose features live at tens of meV — on a route once thought to have a 0.858 meV representability floor on the Si μ=960 lineage — a number since REFUTED as a four-corner sample. **Two grades, both module constants (`CERT_TOL_BY_GRADE`), and no flag, env var or deck key produces a third number.** The certification still RUNS and still REFUSES above the grade it was given, and on a pass the grade and the certified worst \|ΔE_S\| are stamped into the provenance line, the `.dat` header and the `.png`, so a figure cannot be separated from the tolerance it was drawn under. 1.0 meV is a statement about what a picture needs, never a prediction that a route clears it — on a real `--q-per-segment 16` path the interior on-grid Q reached 22.952 meV where the segment corners read 0.841 (`tests/known_failures/2026-08-11-refit-vq-sharded-fetch-and-cert-grades.md` §4) |
| `--refit-r-chunk` | `2048` | r-grid chunk of the refit's Z build; the per-chunk pair-density temp is `(nk, nb, nb, r_chunk)` c128 — shrink on dense k-grids (e.g. 512 at 12×12) to fit device memory |
| `--q-per-segment` | `16` | MINIMUM Q per `K_POINTS crystal_b` segment. A **floor** on the deck's own per-segment counts, not an override: a segment the deck asks for more points on keeps its own count, and `--q-per-segment 1` is the identity and the only way to get the deck's counts verbatim. The default is 16 because E_S(Q) along a high-symmetry path is a bandstructure and the decks in tree carried counts of 1 and 2 — eight diagonalisations for Γ–X–W–L–Γ–Σ, i.e. straight lines between corners. Each extra Q is one more row of the same compiled scan (no extra compile); what grows linearly is the htransform ψ_c(k+Q) cache, which is the thing to watch on a `bse_k_grid`-densified bundle. A floor above ~2 puts most Q **off** the coarse grid, so it needs `--vq-mode interp` and therefore full-BZ `zeta_q.h5` — on an IBZ-ζ lineage the run refuses and names the re-fit |
| `--n-eig` / `--block-size` / `--max-iter` | 6 / 8 / 40 | states per Q / block width / block-Lanczos iters (full reorth by default) |
| `--band-degeneracy` | `strict` | the same multiplet guard as `bse.bse_jax` above, on the same flag with the same default, and it is checked twice here: once where the loader resolves `--n-val`/`--n-cond`, and again on the htransform conduction window, which is cut in window-relative rather than absolute band indices and so is not made safe by the first. The second seam is report-only when you ask for `snap` — its shape is already committed and widening it would desynchronise the conduction caches from the window the loader sized — so on a deck that cuts a multiplet there, `strict` is the mode that actually stops you |
| `--eqp FILE` | none | run on QP energies; corrects stored valence AND htransform enk_sigma (one parser, `bse_io.read_bgw_eqp`) |
| `--eigh-backend` | input key (`auto`) | both distributed-eigh sites (vq_interp C_q, bse_setup fH_q): `auto`/`off` = q-batched native; `cusolvermp`/`slate` = one-tile-at-a-time distributed FFI |
| `head_minibz_average` (input key) | `False` | finite-Q exchange head = mini-BZ cell average instead of point value (4-13% near-Γ error); CLI `--head-minibz-average` overrides. Also gates the `bse_k_grid` coarse→fine rebuild of the q=0 tile (off = the coarse tile and its `vhead` are carried through unchanged). Since `e06fc7de` the average is the moment tensor `<v q_a q_b>` contracted with the transition dipoles (the correct object; see `docs/theory/lt-exchange-head.md` for its validity domain — cell-averaged contexts near Γ, not a substitute for pointwise evaluation) |
| `--w-coarse-grid NX,NY,NZ` | unset | sample a native fine-grid W on a nested coarse BZ sub-grid, then zero-pad W_R back to fine (exact trig interp). The native fine extent must be an integer multiple of the requested coarse extent because this mode begins by decimating existing samples. For a native coarse restart and a nonnested target such as 8x8x1→12x12x1, use `bse_k_grid` instead. |
| `--w-head-densify` | `c1` | How W's Γ head crosses either densifier. `c1` splits it off BEFORE interpolation and re-attaches it analytically over the coarse Γ cell (`gw.head_densify`), so the trigonometric interpolant only ever sees the body — which is what BerkeleyGW's `kernel.x`/`intkernel` split exists to guarantee, because the interpolant of a Kronecker-delta head is a Dirichlet kernel and cannot produce either the bulk 1/q² rise or the slab 1/|q| cusp. Γ averages and finite-q bare factors dispatch through the dimension's `vcoul` kernel; nonnested grids use fractional fine-cell overlaps from their LCM common refinement. `legacy` lets the delta ride through: the documented defect, kept ONLY as the A/B control arm that prices the repair, and it says so when it fires. No effect unless `--w-coarse-grid` or `bse_k_grid` requests densification. |
| `--w-head-gamma-cell` | `fine` | Which mini-BZ cell the re-attached Γ head is averaged over. `fine` is correct; `coarse` is the design's RED TWIN — invisible when the grids are equal and it breaks the head sum rule everywhere else. Gate work only |
| `--rerun-check` | **off** | RUN the diagnostic warm re-solve of the whole scan (reproducibility assert + per-Q dispatch timing). Off by default since 2026-08-08: measured 37.7% of wall at P=4/41Q and 38.1% at P=64/91Q (job 7882533) |
| `--skip-rerun-check` | off | pre-flip spelling of the same thing. Now a no-op naming the default, kept so existing harnesses keep parsing; wins over `--rerun-check` if both are given |
| `--a-band` | none | window-relative band whose bandwidth sets the f-transform width (only if default `a` collapses off-grid ε_c). On a `bse_k_grid` run the same resolved index is passed to both the whole-bundle densifier and the later shifted-Q cache; the two f-transform shoulders may not differ. |
| `nband`/`nval`/`ncond` (input keys) | 100/5/5 | htransform fH window; BSE conduction window [nval, nval+n_cond) must sit interior with ~4-16 guard bands — TWO-SIDED: too few guards rings off-grid 100-1000 meV, too many (nband=80/640c) corrupts on-grid ε_c by ~955 meV |
| `htransform_rank_multiplier` (input key) | `20` | Search ceiling for the same whole-state randomized-QRCP basis used by standalone htransform; `htransform_qr_eps` chooses delivered rank and `0` is an archived alias for 20. The compact selected-state factor preserves one global alpha gauge for refit without a full-r projector. |

Failure modes: `ValueError: no K_POINTS crystal_b block` (add the Q path); the Γ gate assert "htransform conduction cache grossly inconsistent" (max|Δε_c|>0.05 Ry or min-sval<0.5) means the interp window is over-packed — reduce `nband` toward the BSE window; `--vq-mode=interp` refuses on IBZ-only ζ storage and on any deck that is not a slab (both by name — use `ongrid` for the first, `refit` or `ongrid` for the second); `ongrid` refuses off-grid path Q, and `--q-per-segment`'s default floor of 16 puts most path Q off any coarse grid, so check that flag first if a deck that used to run stops; `--refit-window=bse` refuses if its contracted certification misses the `--cert-grade` tolerance, and that is a number to report, not to widen.
Reuse contract: same as bse_jax — newest-mtime `isdf_tensors_*.h5`, `W0_ready` gate, no GW-stamp re-verification. A downfolded bundle is read here too, but by a different route than `bse.bse_jax` uses and for a reason worth knowing: `bse_jax` only READS the stored tensors, while this driver REBUILDS objects in the same ISDF basis (ψ at finite Q, and the exchange tile off the grid). The one whole-state QRCP fit resolves the PARENT centroid table only to map `keep_idx`, then evaluates its basis directly on the child μ_S rows; no parent-width B or projected wavefunction exists. The restart's μ and `keep_idx` remain the authority; invalid or duplicate rows refuse. Off-grid exchange still needs a `zeta_q.h5` beside the bundle; `gw.downfold_cli` transports one (`zeta_S = conj(T) zeta_L`), and a bundle without one gets a refusal that says whether the cause is the downfold or an IBZ-ζ parent, plus the note that `--vq-mode=ongrid` needs no ζ at all. Distributed-eigh key coverage (`eigh_backend` at both sites): `docs/dev/large_nmu_operation.md`. Not in the fastloop chain — smoke like bse_jax (mini GW dir + 4 host devices).
