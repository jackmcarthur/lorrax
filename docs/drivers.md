# Driver reference

The seven pipeline drivers in chain order: input generation (centroids, dipole,
kin-ion) → GW self-energy → QP bandstructure interpolation → BSE → exciton bands.
Deck keys and defaults are verified against `gw_config._DEFAULTS` as of 2026-08-01;
the complete key list is in [input_reference.md](input_reference.md).

All seven start through `runtime.initialize_communicator_stack` (one module-top
call: mesh + clique warm-up + startup report), so every driver is launchable under
the certified P=16 geometry — `config/frontera/templates/gw_dev.sbatch`
(srun --mpi=pmi2 + apptainer + `impl=mpi`, 8 nodes × 2 ranks × 28 threads) with the
module swapped in. Smoke test for any change to the first five drivers: the
fastloop mini-deck chain (kmeans→dipole→kin-ion→gw→eqp-convert→htransform dft+qp)
checks every stage against pinned references at P=1 AND on a 2×2 host-device mesh
in ~2–5 min: `cd /scratch2/08271/jackmc/lorrax_sandbox && sbatch
fastloop/run_fastloop.sbatch` (or `bash …` inside any allocation; it reads the
LIVE repo src by default, `LORRAX_SRC` overrides). Exit 0 = pass, 1 = numeric
drift (per-file deltas printed), 2 = a stage failed (the log names the driver).
BSE and exciton bands are NOT in the chain yet.

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
| `--use-phdf5` | off | parallel-HDF5 loader for WFN.h5 too big for host RAM |
| `LORRAX_CENTROID_RANK_TOL` (env) | 0.01 | rank-gate tolerance; lowering it is a deliberate override |

Invariant: the selection window must span the sigma window `[0, nelec+ncond)` the GW run consumes; the default
is a superset of any deck's `ncond`, so a shortfall means it was narrowed explicitly. Main refusal: "FATAL:
pivoted-Cholesky rank deficiency" (certified rank < requested orbits) — widen `--prune-n-cond`, use
`vc_x_vc`, or raise `--oversample`. Also fatal: FFT-grid mismatch rho vs WFN (needs ecutrho = 4*ecutwfc).

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
Consumes `WFN.h5`, `centroids_frac.txt`, `kin_ion.h5` (+ `centroids_file_current`/`tmp/v_q_bispinor.h5` when bispinor); produces `tmp/zeta_q.h5` (+`zeta_q_mu{1,2,3}.h5`), the restart tensor file `tmp/isdf_tensors_{n_rmu}.h5` (V_qmunu, G0, enk_full, psi_full_y[_transverse], W0 + head scalars for BSE), and `eqp0.dat`, `eqp1.dat`, `sigma_diag.dat`, `sigma_mnk.h5`, `WFN_qp.h5` (`eqp_g0w0.dat` in PPM one-shot).
Band windows (`wavefunction_bundle.BandSlices` from `Meta.from_system`): the σ/QP window is always [0, nelec+ncond) — nval moves only the interior edge b1 = nelec−nval, never the window bottom; `nband` sets the χ₀/Σ band-sum top b4 = round_up(nband, world_size) (pads are zeroed).

Invoke: `python -m gw.gw_jax -i gw.in` under the certified Frontera CLX launch block `config/frontera/templates/gw_dev.sbatch` (srun --mpi=pmi2 + apptainer + impl=mpi CPU collectives; typical 8 nodes × 2 ranks × 28 threads = P=16; full-pipeline parity jobs 7884609/7884612). Demonstrated end-to-end by `/scratch2/08271/jackmc/mos2_4x4_test/valsmoke_tmpl.sbatch` and `gw_ht_b300.sbatch` (their explicit `LORRAX_FFT_FFI=1` + `LORRAX_FFT_FFI_FUSED=1` exports are redundant since 2026-08-01 — the FFI stack is the required default).
P-scaling at P=16 (b300 deck, 2979c/300b): the ζ charge factor executes q-parallel above nq·mu³ ≥ 5e9 — `zeta_fit.cholesky` 104.4 → 11.8 s (8.9×), GW wall 335.9 → 214.6 s, bit-identical (job 7885024); `distributed_zeta_solve = distributed` runs wall 222.1 s with no O(mu²) replica, parity ≤ 4.9e-5 eV gauge-class (job 7885077) and wins outright at large mu (mu=10015: factor 4712 → 236 s on 64 ranks); the Σ τ-loop is device-bound (its d2h_wait is the kernel wall surfacing in the drain), boundary flushes now once per branch (jobs 7885105/7885109).

| key (deck unless noted) | default | meaning |
|---|---|---|
| `nband` / `nval` / `ncond` | 100 / 5 / 5 | χ₀ band-sum top / interior valence edge / σ conduction count; σ diagonals for [0, nelec+ncond) |
| `compute_mode` | auto | Σ ansatz: `x_only` \| `cohsex` \| `gn_ppm` \| `hl_ppm` \| `mpa`; auto infers from `do_screened`/`use_ppm_sigma`/`ppm_model` and never infers `mpa`. `mpa` (multipole-W, the owner's "FF") is declared on the axis but REFUSES AT DRIVER ENTRY until its Σ stage lands — see [THEORY_mpa_implementation.md](theory/THEORY_mpa_implementation.md) |
| `qp_solver` | auto→one_shot_dft | `one_shot_dft` (G0W0 at E_DFT) \| `fixed_point` (on-shell solve) \| `self_consistent` (QSGW loop) |
| `restart` | true | reuse `tmp/isdf_tensors_{n_rmu}.h5` (skip ζ-fit/V_q); stamp contract below |
| `ppm_probe_chi_reuse` | off | opt-in `auto`: probe χ₀ folds into ONE fused τ sweep on static+k augmented nodes; pays only where per-node χ cost dominates (nets +1.2 s at b300 — planner cost; job 7885109) |
| `hartree_source` | auto | THE G-space vs ISDF V_H switch: `stored` (v_hartree array in kin_ion.h5) \| `gspace` (exact FFT-grid rebuild) \| `isdf` (V_q[0] quadrature) \| legacy `folded` = V_H inside kin_ion values; auto resolves stored→folded→isdf |
| `charge_zeta_solve` | rank_truncate | charge ζ CCT conditioner: rank-revealing eigh pseudo-inverse dropping λ < `zeta_rcond`·λmax (default 1e-8) vs bit-identical historical `cholesky` |
| `distributed_zeta_solve` | auto | ζ back-solve tier: `replicated` \| `per_q` \| `distributed` (nothing O(μ²) replicated; needs rank_truncate + square/1-D mesh); auto = replicated under 4 GiB gather cap, else per_q |
| `w_dyson_solver` | auto=local | W Dyson plan: `local` per-q pivoted LU in the q-sharded map vs `distributed` 2-D block-cyclic backsolve (ScaLAPACK/cuSOLVERMp); refuses loudly, never downgrades |
| `sigma_omega_layout` | replicated | Σ_c(ω,k,m,n) cube: `sharded` keeps (m,n) mesh-tiled end-to-end, for every `qp_solver`; refuses at resolve time when the σ window does not divide both mesh axes, or h5py_allgather at P>1. Under `self_consistent` the full-cube gather runs once per Σ evaluation, so that is the solver with the most to gain |

Restart/reuse contract. Two independent caches. (1) `tmp/isdf_tensors_{n_rmu}.h5` (V_qmunu, G0, enk_full, psi_full_y[_transverse], W0 + `W0_ready`): reused only when the band-window/n_rmu/kgrid attrs match AND the `centroids_charge_md5`/`centroids_transverse_md5` content hashes match — the quadrature BASIS, not just its size ("same count, different points" refuses). (2) `tmp/zeta_q.h5`: reused only when `zeta_is_done` is set and the `fit_provenance` JSON is byte-identical — n_rmu, band ranges, bispinor, gspace_mode, both cutoffs, EFFECTIVE `zeta_rcond`/`zeta_ridge` (env overrides applied), `charge_zeta_solve`, `gamma_contract_mode`, `write_ibz_only`, band_norms, fft_grid, ecutwfc/ecutrho, wfn_file+wfn_bytes, and (2026-08-01) the `distributed_zeta_solve` GAUGE tier: 'replicated' | 'distributed' (per_q collapses to replicated — same factor bits); a stamp MISSING the key is legacy replicated (announced, reusable), a real tier mismatch refits. Mesh/P are deliberately excluded: a ζ fit at P=4 is reusable at P=80. `LORRAX_FORCE_REFIT=1` forces a refit; every mismatch costs compute, never correctness.
FFI gates are env vars, not deck keys, and the FFI layer is REQUIRED (`docs/architecture/decisions.md` 2026-08-01): `LORRAX_FFT_FFI` (MKL/cuFFT flat-k FFT backend), `LORRAX_FFT_FFI_FUSED` (fused IFFT·(G·W)·FFT τ kernel), `LORRAX_BANDS_GEMM_FFI` — all default ON; grammar 0/off/false/no, 1/on/true/yes (a stale `auto` resolves to the default with a grammar note); a missing/unloadable handler refuses at startup naming the `.so` (`Gate.enforce` via `runtime.initialize_communicator_stack`); `=0` refuses for the FFT dial (the XLA duplicate is deleted) and is an announced uncertified debug opt-out for the other two (src/ffi/gate.py; per-knob table in `docs/dev/env_vars.md`).
Large-N_mu / fully distributed operation — every distributed key per stage, per-rank memory scalings, what is still replicated, the auto-thresholds and the certified example jobs: `docs/dev/large_nmu_operation.md`.
New-user failure modes: restart refusals naming a changed band window or centroid table ("same count, different points") → rerun `restart = false`; bispinor runs refuse on missing `psi_full_y_transverse`/`v_q_bispinor.h5` (Σ^B would silently drop); sanity gates kill runs with non-negative Σ_x diagonals or NaN kin_ion/Σ before eigh; sharded-layout ValueErrors quote the divisibility/backend fix.
Fastloop stage: `gw` (~40 s cold) — the mini-deck exercises the standing bare-launch path for SlabIO's availability probe (the `slab_io` deck key was removed 2026-08-06; there is one transport).

## htransform — `bandstructure.htransform`

Galerkin QP bandstructure interpolation to an arbitrary k-path from coarse-grid data: Galerkin-projects ψ onto the ISDF centroid basis (since 62ba395 a Gram-eigh of A Aᴴ — nk·nb square, N_μ-free, through the `ffi.linalg` eigh plan under the same `eigh_backend` deck key as fH_q — with Vᴴ/`B_at_mu` μ-sharded; rank ≤ nk·nb), builds the f-transformed Hamiltonian fH_k = Σ_n f(ε_nk) c_nk c_nkᴴ, IFFTs to lattice fH_R, then per path-q eigvalsh + Newton inversion of f recovers ε_n(q); with `--eqp-file` the anchoring ε are GW QP energies, giving the QP bandstructure.
Consumes the same deck as gw_jax ([cohsex] keys + `K_POINTS {crystal_b}` path), `WFN.h5` (or `--wfn-file WFN_qp.h5`), `centroids_file`; writes `bandstructure.dat` (VBM shifted to 0, energies in Ry; rank 0 is the only writer). No restart file of its own — reuse means feeding it the same WFN/centroids/eqp inputs. Distributed-key coverage (`eigh_backend`, `use_low_mem_eigh`) and the per-rank memory story: `docs/dev/large_nmu_operation.md`.
Band window is (nelec−nval, nelec+ncond) (`load_wfns_and_enk_for_sigma`) and must contain ALL valence bands — set `nval = nelec` (b300 deck: nval=26=nelec): the VBM/Fermi is taken from band index nelec−1 of the window-sorted path energies, correct only when the window starts at band 0, and the eqp override must supply every window band at every k.

Invoke: `python -m bandstructure.htransform -i ht.in --verbose [--eqp-file eqp_ht.dat]`; single-process CPU is ample at production size (rank ≤ nk·nb·nspinor ~1400), demonstrated by `/scratch2/08271/jackmc/mos2_4x4_test/gw_ht_b300.sbatch` (GW at P=16 via gw_dev.sbatch, then htransform twice at P=1/56 threads: DFT bands, then QP bands from the converted eqp file). Multi-process certified: P=16 wall 57→27 s (2.1×), `bandstructure.dat` byte-IDENTICAL to P=1 (BD.2, job 7884870; the thread-main refusal there is closed); post-Gram-eigh P=4 srun exact-0 vs baseline, one writer (job 7885093). Memory: the Gram-eigh removed the replicated A+SVD term — 1 proc × 4 host devices, μ 4962→18084: VmHWM slope +1.9 → +1.2 GB, wall 67 → 44 s; the remaining μ-bound is single-axis ψ sharding (large_nmu doc, "still replicated" list). Fastloop stages: `ht_dft`/`ht_qp` (~8+6 s), including the "Using EQP energies" log gate.

`--eqp-file` format (`read_eqp_energies`): `k-point <K>:` block per FULL-BZ k-point — nk must equal sym.nk_tot (16, not the 10-point IBZ) — with per-band `n=<B> ... EQP=<value>` (or `sigX=`) lines, energies in Ry (the DFT path returns wfn.energies in Ry). No GW output satisfies this directly: `eqp0/eqp1.dat` are BGW columns on the IBZ, `eqp_g0w0.dat` spells `Re=` not `EQP=`. Convert with `make_eqp_htformat.py` (deck dir), which joins eqp_g0w0 + eqp1 on E_DFT/Eqp0 and emits Ry; a missing or unparseable `--eqp-file` is a FATAL SystemExit, never a silent DFT fallback.

| key / flag | default | meaning |
|---|---|---|
| `nval` / `ncond` | 5 / 5 | window (nelec−nval, nelec+ncond); nval MUST equal nelec (all valence bands) |
| `eigh_backend` (key) / `--eigh-backend` (CLI overrides) | auto | fH_q eigensolver for the get_centroids_fi handoff AND the Galerkin Gram-eigh (there `auto` = native replicated is fine — the tile is N_μ-free): auto\|off = q-batched native eigh; `distributed`/`cusolvermp`/`slate`/`scalapack` spread ONE (rank,rank) tile over the mesh (wide windows; square mesh, 1 proc/device) |
| `use_low_mem_eigh` | false | same axis by intent: true + auto ⇒ distributed; true + off refused at parse |
| `get_centroids_fi` | false | BSE handoff: also compute fine-grid ψ at coarse centroids via `bse_setup.compute_wfns_fi` |
| `kgrid_fi` / `wfn_fi_min` / `wfn_fi_max` | "" / 0 / 0 | fine k-grid "nx ny nz"; sub-window on the htransform band axis (0 = full window) |
| `wfn_fi_q_chunk` | 0 = N_q_coarse | fine-q chunk per f(H(q)) build; floor, rounded to device count |
| `--a-band` | top band | band whose bandwidth sets the f-transform scale a |
| `--fh-diagnostics` | `auto` (follows `--verbose`) | `on`/`off` force the fH_k range stats, the Γ eigenvalue check against f(ε) and the Γ round-trip. They are 33% of the cache-cold h_transform stage (1.442 s, 7 XLA programs) and hold fH_k alive across the whole solve (576 MiB at the reference shape) — hence off unless asked for |
| env `LORRAX_FH_ORTHO_TOL` / `LORRAX_GALERKIN_CHUNK_GIB` | 1e-6 / 6 | orthonormality gate cap (0 disables — never in production) / streamed ψ chunk budget |

Failure modes: `build_fH_R` refusal "Galerkin coefficients are NOT orthonormal" = centroids cannot span the window — fix with more centroids or a narrower window, NEVER rtol (tightening rtol makes it worse, measured); FATAL on `--eqp-file` not found/parsed (use the converter); sanity refusals on NaN S/ctilde/E_nk or E_nk spread > 20 Ry (a garbage GW eqp file fed forward).

## bse — `bse.bse_jax`

Solves the Bethe-Salpeter equation for neutral excitations: the electron-hole Hamiltonian H = D + V − W (D = quasiparticle transition energies ε_c−ε_v, V = bare exchange, W = statically screened direct term) in the pair basis |vk,ck⟩, in the ISDF μ-basis on a 2D (μ,ν) device mesh. It consumes the GW restart `isdf_tensors_<N_mu>.h5` (newest in the run dir wins; datasets `psi_full_y`, `enk_full`, `V_qmunu`, `W0_qmunu` with `W0_ready` attr) plus optionally `eqp1.dat` for QP energies, and produces exciton eigenvalues and, with `--write-eigs`, a BGW-layout `eigenvectors.h5` (eV; valence axis reversed on write; rank 0 is the ONLY writer — every-rank writes raced and truncated the file at P=64, job 7879470). All sharded solves (Lanczos/Davidson/FEAST) apply H through the one trial-stack matvec (`bse_stack_matvec`): a shard_map whose body scans the trial axis so exactly one T-tensor is alive regardless of block width; the `--matvec-kind ring|gather|simple` flag survives in the CLI but the sharded eigensolve path ignores it (retired — see the "legacy `matvec_kind` selector is retired here" note in `bse_lanczos.py::solve_bse_sharded`, and the retirement plan in `bse_stack_matvec`'s module docstring). The `bse_k_grid` input key densifies the whole bundle (ψ/ε via one htransform fH, V_q0 via `vq_interp`, W by zero-pad in R = exact trig interpolation, through `bse_io.make_w_densifier` — the ONE sharded densifier, per-rank peak = the local (μ,ν) tile) from the coarse restart grid to a fine grid before any solve.
Reuse contract: consumption-side only — the newest-mtime `isdf_tensors_*.h5` wins (log line names the losers) and screening is gated solely by the `W0_ready` attr; NONE of the GW band-window/centroid stamps are re-verified here, so point the driver at the run dir whose physics you mean. Distributed keys it shares with the chain (`eigh_backend`, transport): `docs/dev/large_nmu_operation.md`.

Invoke: `python -u -m bse.bse_jax -i cohsex.in --lanczos ...` in the GW run directory. Single node = 1 process, mesh the square s×s over `jax.device_count()` (non-square counts refuse — decisions.md 2026-08-01); multi-node uses the certified `config/frontera/templates/gw_dev.sbatch` geometry (srun --mpi=pmi2, 2 ranks/node, taskset 28 cores/rank, apptainer + `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`) with the module swapped in for `gw.gw_jax`.

| key / flag | default | meaning |
|---|---|---|
| `--lanczos` | off | REQUIRED for the Lanczos/Davidson eigensolve; without it the driver dispatches FEAST (`bse_feast`) |
| `--bse` / `--rpa` | RPA | kernel: default (neither flag) is RPA D+V; `--bse` enables the screened −W direct term |
| `--tda` | off | Tamm-Dancoff (resonant only); default is full non-TDA via `bse_nontda` (definite-pencil solve) |
| `--n-val` / `--n-cond` | 4 / 4 | pair-basis band window; loader clamps over-requests; valence auto-detected by mean ε<E_F (`--n-occ` overrides) |
| `--solver` | `lanczos` | `davidson` = diagonal-precond per-state convergence (oscillator strengths); `lanczos` = spectrum shape; `trlan` = thick-restart Lanczos (`solvers/thick_restart_lanczos.py`), a bounded-memory restart of the same Krylov space |
| `--block-size` / `--max-lanczos-iter` | 1 / auto | block width / TOTAL Krylov dim bound (block iters = max/bs) |
| `--n-reorth` | −1 | reorth window; −1 = full reorth (needed for degenerate spinor spectra; small windows breed ghosts). Since the CGS2 default (2026-08-08) a narrow window is **not** faster — the batched route costs `2·max_iter` collectives whatever the window — so ghosts are the only remaining reason to widen it and there is no speed reason to narrow it |
| `--davidson-precond` | `bare` | Davidson preconditioner diagonal. `bare` divides by `dE = E_c − E_v`; `exact` divides by the assembled `diag(H)` (what BerkeleyGW hands PRIMME), assembled once per payload at identical per-iteration cost; `auto` picks on transition-space dimension at `bse_davidson_helpers.EXACT_PRECOND_AUTO_MIN_DIM`. **That crossover is a placeholder awaiting a large-deck measurement, not a measured value** — see `PRECOND_BUILD_FREE.md` "what would settle it" |
| `--davidson-olsen` | off | Olsen-correct the preconditioned direction (project out its component along the current Ritz vector). Two batched inner products per iteration; this is what makes a small `--davidson-eps-shift` safe |
| `--davidson-eps-shift` | 1e-3 Ry | Davidson preconditioner regularisation (= 13.6 meV) |
| `--davidson-m-max` | 10·n_eig | max Davidson subspace dimension before restart (measured knee). Memory is `2·m_max` trial vectors — Davidson stores both `V` and `HV` |
| `--trlan-m-max` | max(3·n_eig, 60) | thick-restart Lanczos basis slots — the memory cap that makes the solve fit at production `bse_dim`, paid for in extra matvecs. Useful range 3–5× n_eig |
| `--trlan-n-keep` | n_eig+10 | Ritz vectors retained across a thick-restart. Must satisfy `n_eig ≤ n_keep < m_max` |
| `--matvec-kind` | `ring` | **INERT on the sharded eigensolve path** — every sharded solve builds `bse_stack_matvec` regardless (`bse_lanczos.py`, "the legacy `matvec_kind` selector is retired here"). Still parsed, and still steers `bse.absorption_haydock`. Whether it should refuse or be deleted is an open owner decision |
| `--ring-timing` | off | parsed and inert. Kept parseable on purpose so archived launch scripts do not `SystemExit`; for per-term timings use `timing.section` on the jitted matvec |
| `--write-eigs [N]` | off | write `eigenvectors.h5` (rank-0 single-writer; non-TDA adds `eigenvectors_coupling` Y) |
| `--eqp FILE` | none | full-BZ `eqp1.dat` (LORRAX GW format): re-slice bands on QP energies (n_occ re-resolved) |
| `bse_k_grid` (input key) | `""` | "NX NY NZ" fine grid; each axis a multiple of the coarse grid; empty = coarse bundle byte-identical |
| `LORRAX_BSE_MATVEC_OPT` (env) | unset | only token is `gspmd`, a default-off audit route that drops the manual `shard_map`; unknown tokens REFUSE.  `yhoist` was made **permanent** and `krep` **removed** on 2026-08-08 — setting either now refuses |

Failure modes: a loud banner "W0_qmunu not ready — falling back to BARE COULOMB V" means GW screening never ran — the spectrum has no excitonic screening; `FileNotFoundError: isdf_tensors_*.h5` means you are not in (or did not point `-i` at) a GW run dir; multiple restarts are resolved newest-mtime; forgetting `--bse` silently gives RPA.
Not in the fastloop chain: the fastest smoke is this driver on a mini GW run dir (e.g. a fastloop `check.<jobid>/<leg>` dir) single-process with `XLA_FLAGS=--xla_force_host_platform_device_count=4` — a real 2×2 mesh, no MPI.

## exciton bands — `bse.exciton_bands`

Computes the finite-momentum exciton dispersion E_S(Q) — the TDA BSE H_Q = D_Q + V_Q − W in the pair basis |vk, c k+Q⟩ — along the `K_POINTS crystal_b` path in the input (same block/parser as the htransform bandstructure driver; refuses without it). Q-shifted conduction eigenpairs ψ_c(k+Q), ε_c(k+Q) come from ONE interpolated htransform fH (`bandstructure.bse_setup.compute_wfns_fi` with q-list {k+Q}; q-axis chunking via the `wfn_fi_q_chunk` input key, default 0 = one chunk sized like fH_R); W is the unchanged coarse-grid FFT convolution (all k-differences stay on-grid); exchange V_Q is interpolated (`vq_interp`, tile momentum wrap(−Q); G=0 KEPT at finite Q, BGW energy_loss convention; production q=0 tile at Γ). The whole path is ONE jitted `lax.scan` of block-Lanczos over the stack matvec (one compile for all Q). Consumes the same `isdf_tensors_*.h5` (+ optional `eqp1.dat` on BOTH legs); writes `<prefix>.dat` and `<prefix>.png` (rank 0 only, followed by a collective barrier — without it ranks exiting early got the job SIGABRT rc=134 after correct outputs, job 7882507).

Invoke: `python -u -m bse.exciton_bands -i cohsex.in --n-val 4 --n-cond 4 --n-eig 6 [--vq-mode both --refit-points 0,8,15]`; never on a login node. Same single-node (1 process, --px/--py mesh) vs multi-node story as bse_jax: the certified geometry/transport is `config/frontera/templates/gw_dev.sbatch` (module swapped in); `/scratch2/08271/jackmc/mos2_4x4_test/valsmoke_tmpl.sbatch` certifies the shared W-densifier (pre-B HLO probe: no all-gather + parity).

| key / flag | default | meaning |
|---|---|---|
| `--vq-mode` | `interp` | `interp` (b26p model) \| `both` (+per-Q ζ-refit ground-truth spot checks) \| `ongrid` (exact stored tiles, on-grid Q only, works with IBZ-only ζ); `refit` alone refuses |
| `--n-eig` / `--block-size` / `--max-iter` | 6 / 8 / 40 | states per Q / block width / block-Lanczos iters (full reorth by default) |
| `--eqp FILE` | none | run on QP energies; corrects stored valence AND htransform enk_sigma (one parser, `bse_io.read_bgw_eqp`) |
| `--eigh-backend` | input key (`auto`) | both distributed-eigh sites (vq_interp C_q, bse_setup fH_q): `auto`/`off` = q-batched native; `cusolvermp`/`slate` = one-tile-at-a-time distributed FFI |
| `head_minibz_average` (input key) | `False` | finite-Q exchange head = mini-BZ cell average instead of point value (4-13% near-Γ error); CLI `--head-minibz-average` overrides |
| `--w-coarse-grid NX,NY,NZ` | unset | sample W on a coarse BZ sub-grid, zero-pad W_R to fine (exact trig interp); subsumed by the general `bse_k_grid` key |
| `--rerun-check` | **off** | RUN the diagnostic warm re-solve of the whole scan (reproducibility assert + per-Q dispatch timing). Off by default since 2026-08-08: measured 37.7% of wall at P=4/41Q and 38.1% at P=64/91Q (job 7882533) |
| `--skip-rerun-check` | off | pre-flip spelling of the same thing. Now a no-op naming the default, kept so existing harnesses keep parsing; wins over `--rerun-check` if both are given |
| `--a-band` | none | window-relative band whose bandwidth sets the f-transform width (only if default `a` collapses off-grid ε_c) |
| `nband`/`nval`/`ncond` (input keys) | 100/5/5 | htransform fH window; BSE conduction window [nval, nval+n_cond) must sit interior with ~4-16 guard bands — TWO-SIDED: too few guards rings off-grid 100-1000 meV, too many (nband=80/640c) corrupts on-grid ε_c by ~955 meV |

Failure modes: `ValueError: no K_POINTS crystal_b block` (add the Q path); the Γ gate assert "htransform conduction cache grossly inconsistent" (max|Δε_c|>0.05 Ry or min-sval<0.5) means the interp window is over-packed — reduce `nband` toward the BSE window; `--vq-mode=interp` refuses on IBZ-only ζ storage (use `ongrid`); `ongrid` refuses off-grid path Q.
Reuse contract: same as bse_jax — newest-mtime `isdf_tensors_*.h5`, `W0_ready` gate, no GW-stamp re-verification. Distributed-eigh key coverage (`eigh_backend` at both sites): `docs/dev/large_nmu_operation.md`. Not in the fastloop chain — smoke like bse_jax (mini GW dir + 4 host devices).
