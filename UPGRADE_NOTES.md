# Upgrade notes

User-visible changes, newest first. Each entry says what changed, which
results move, and what a user must change in decks, environment or files.
The binding rulings behind breaking changes are in
`docs/architecture/decisions.md`; older history is in git.

## 2026-10-22 — GW: `compute_mode = tangential`; BSE: BerkeleyGW output files, srlr only

- **Decks.** The tangential W model is `compute_mode = tangential`. The
  `sigma_w_model` key refuses: "`sigma_w_model` is now `compute_mode =
  tangential` (or `compute_mode = mpa` for the elementwise model)". Write
  `compute_mode = tangential` for `compute_mode = mpa` + `sigma_w_model =
  tangential`; results are bit-identical. `sigma_w_accuracy`,
  `sigma_w_support_sites_ev`, `write_w`, `write_poles` and
  `bispinor_gw = full_tangential` require `compute_mode = tangential`.
- **BSE W interpolation.** `w_head_densify` is `srlr` only; `c1` and
  `legacy` refuse. A `kgrid_fine` BSE needs a restart with `W0_lr` (a static
  GW: `cohsex`, `gn_ppm`, `hl_ppm`). A BSE on a `compute_mode = mpa` restart
  refuses (`GATE bse_mpa_w0`).
- **BSE files.** `eigenvectors.h5`, `eigenvalues*.dat` and `absorption_*.dat`
  are BerkeleyGW's exact layout (`versionnumber`, int32 scalars, `kpts`
  (nk, 3), the four eigenvector datasets, `spin_kernel` per kernel, the WFN's
  `mf_header`; `eigenvalues.dat` along a deck `polarization`; absorption files
  with BerkeleyGW's four header lines only). Readers of the old `version`,
  (3, nk) `kpts` or `eigenvectors_coupling` must follow.
- **FEAST** is deprecated, and refuses under the TDA (`GATE feast_tda_complex`).

## 2026-10-21 — BSE: BerkeleyGW key names in the deck, TDA and the screened kernel by default

The BSE drivers read their settings from the deck passed with `-i`, under
BerkeleyGW's absorption.x and kernel.x names. Old command lines refuse with
the deck key to write ("`--n-val` is now deck key `number_val_bands_fine`").

- **Results that move.** TDA is the default (`full_bse = true` for the full
  BSE; was full BSE unless `--tda`). The screened kernel is the default (was
  RPA unless `--bse`); `local_fields = true` gives the RPA. On a non-spinor
  WFN `spin_singlet` (default) or `spin_triplet`; on a spinor WFN those keys
  refuse. A deck that relied on either old default changes its answer.
- **bse.bse_jax:** keep `-i`, `--px/--py`, `--report-file`, `--out-prefix`.
  `--n-val`, `--n-cond`, `--n-occ` are `number_val_bands_fine`,
  `number_cond_bands_fine`, `highest_occupied_band`; `--eqp` is
  `eqp_corrections` (or `eqp_co_corrections` with `kgrid_fine`; a bare key
  reads `eqp.dat` / `eqp_co.dat`); `--dipole` is `dipole_file`; `--n-eig`,
  `--write-eigs` are `number_eigenvalues`, `write_eigenvectors` (0 none,
  < 0 every computed state); `--solver lanczos` is `solver = block_lanczos`;
  `--max-lanczos-iter` is `solver_max_iter`; `--haydock-iter N` is
  `haydock = true` with `number_iterations`; the other solver, KPM and FEAST
  flags are the deck keys of the same name in snake case. `--lanczos` is gone.
- **bse.absorption_haydock:** `--eta-eV`, `--n-iter`, `--n-omega` are
  `energy_resolution`, `number_iterations`, `delta_frequency`; `--V-cell`,
  `--n-spinor`, `--n-spin` are gone (read from the WFN); `--pol` is
  `polarization`, a vector along b1 b2 b3.
- **bse.exciton_bands:** the window, solver, eqp and dipole settings are the
  same deck keys; its own flags stay. `--max-iter` is `solver_max_iter`.
- **bse.bsemat_cli:** `--n-val`, `--n-cond`, `--kernel`/`--extended`,
  `--exciton-Q` are kernel.x's deck keys `number_val_bands`,
  `number_cond_bands`, `extended_kernel`, `exciton_Q_shift Qflag Qx Qy Qz`
  (kernel.x's sign, Q_shift = -Q). One kernel form per run.
- **Fine grid:** `bse_k_grid` is `kgrid_fine` (also the htransform utility's
  `kgrid_fi`); new `kgrid_fine_shift` (kgrid.x's dk) and
  `number_cond_bands_coarse`. The htransform utility's `wfn_fi_min`/`max`
  are `number_val_bands_fine`/`number_cond_bands_fine`, counted from the
  occupied boundary, and its fine bands are written to `<stem>_fine.h5`.
- **New output:** with `dipole_file` the eigensolve writes
  `absorption_eh.dat` and `absorption_noeh.dat` (Gaussian by default).
- **Deck syntax:** BerkeleyGW's `key value` and bare-flag lines are read.

## 2026-10-21 — Old deck keys, values and flags refuse instead of being aliased

A renamed deck key, deck value or command-line flag is no longer accepted
under its old spelling. It refuses with one line, "`old` is now `new`" (or
"`old` was removed: ..."), before the run starts; edit the deck or the
command line. HDF5 names inside existing restart bundles, stores and banks are
still read under their old spellings, with one NOTE.

## 2026-10-21 — The shared-pole W model is the tangential W model

The production full-frequency W model is renamed after the tangential
interpolation that builds it, in the code, the decks, the files and the
logs. The entries below this one keep the old name. Nothing is computed
differently.

- **Decks:** `sigma_w_model = tangential` (was `shared_pole`),
  `bispinor_gw = full_tangential` (was `full_shared_pole`), and in TD-aGW
  decks `collision_screening = tangential` (was `shared_pole`). An old value
  refuses with one line naming the new one: edit the deck.
- **Files:** the per-map store is `<label>_tangential/` (was
  `<label>_shared_pole/`); the model, bank, sector-manifest and receipt schemas
  are `lorrax.tangential.v1`, `lorrax.tangential-bank.v3`,
  `lorrax.tangential-sectors.v1`, `lorrax.tangential.receipt.v1`; the restart
  bundle's member dataset is `tangential_member`. Stores, banks and restart
  bundles written before the rename are read where they are, under their old
  names. The recipe and gate tags (`shared_real_pole_v1_r4`,
  `shared_real_pole_gates_*`) and their hashes do not change, so a stored model
  still authenticates.
- **Logs and refusals:** `GATE shared_pole_*` ids are `GATE tangential_*`,
  and the log lines say "tangential" where they said "shared-pole". A script
  that greps a log or a GATE id must use the new spelling.
- **Code:** `gw.shared_pole_*` modules are `gw.tangential_*`,
  `file_io.shared_pole_store` is `file_io.tangential_store`, and every
  `shared_pole` identifier is `tangential` (`screen_shared_poles` is
  `screen_tangential_poles`). Every old spelling is listed in one table,
  `src/runtime/renames.py`; `docs/dev/renaming.md` has the procedure.

## 2026-10-21 — Centroid and kin-ion flag cleanup

`centroid.kmeans_cli`: the positional count is `n_centroids` (was `N_c`;
same default, 400). `--orbit` is gone and `--no-orbit` is `--no-symmetries`;
symmetric selection stays the default whenever the decorated atoms have more
than one operation. `--rho-power`, `--prune-window`, `--prune-n-val`,
`--prune-n-cond`, `--plot-zoom`, `--no-shard` and `--force-shard` are
removed. Their defaults are now fixed: unit power, left window = occupied +
the deck's `ncond` (every band without `-i`), right window = every WFN band,
and the device mesh's latency floor (which only applies to one process
driving several devices). `--fit-window` is the one window override.
`gw.kin_ion_io`: `-n`/`--nb` is removed; the file always covers
`[0, nelec + ncond)`, its old default.

- **Results do not move** for a command line that used only defaults, and the
  centroid table's header keeps its layout (`rho_power=1` included).
- **Command lines** that name a removed flag now refuse: drop it, or replace
  a window flag with `--fit-window L0:L1,R0:R1`.
- **W-av stencil removed.** The `w_av_first_neighbors` and `w_av_second_neighbors` deck keys and the dipole driver's `--w-av-only` are gone, with the unused W-av reader and writer; a deck that sets either key now refuses as an unknown key.

## 2026-10-21 — Davidson for every lowest-N BSE solve: the `bse_jax` default route, exciton bands per Q, margin roots, convergence gates on FEAST and the non-TDA matrix-free route

`bse.exciton_bands` now solves each Q's `--n-eig` lowest with `bse_lanczos`'
Davidson (`--solver davidson`, the default), the route `bse_jax` uses: the
exact diagonal E_c(k+Q) - E_v(k) plus the kernel diagonal as preconditioner,
soft locking, and a stop when every root's residual is below
`--davidson-tol-mev` (0.01 meV). One planned solver serves the path. A Q that
stops on the cap refuses (`GATE davidson_converged`) before anything is
written. `--solver lanczos` keeps the fixed block-Lanczos scan, which has no
convergence test.

- **`bse_jax` route.** With no route flag `bse_jax` now runs the eigensolve
  (Davidson); it used to run FEAST. `--lanczos` is kept as a no-op alias, and
  FEAST needs `--feast`. A FEAST flag without `--feast` refuses, as do two route
  flags. Without `--tda` the default is now the dense structure-preserving
  non-TDA solve (`bse_nontda`, N <= 4096); it used to be non-TDA FEAST. A
  larger non-TDA window refuses before the restart load and the W transform
  (`GATE nontda_dense_size`) and names the alternatives (`--tda`, `--feast`).
- **FEAST refuses unconverged pairs** (`GATE feast_converged`): every in-window
  Ritz pair's residual must be below `--feast-tol-mev` (0.01 meV, the Davidson
  default); out-of-window pairs are leakage and are not tested. It used to log
  the residuals and report the values regardless. A saturated window (every
  Ritz pair in-window, so eigenvalues may be missing) refuses too; raise
  `--feast-ritz-count`. `--feast-iter` (2) is now a `bse_jax` flag.
- **Margin roots.** BSE Davidson (`bse_jax`, `exciton_bands`) solves 8 roots
  above `--n-eig` (fewer when the transition space is small, at most half of
  it), stops when the lowest `--n-eig` converge, and reports only those. NSCF
  Davidson is unchanged. `bse_jax --davidson-margin N` sets the count and
  `--davidson-margin-tol-mev` a looser residual at which margin roots lock.
- **Non-TDA matrix-free** (`bse_nontda`, `solver='matrixfree'`, opt-in): restart
  cycles now run until every root's omega residual is below 0.01 meV
  (`mf_tol_mev`; cap `mf_n_restarts`, now 50) and a capped stop refuses
  (`GATE nontda_converged`). It used to run 3 cycles and return the result
  whatever the residual. The default dense non-TDA route is unchanged.
- **Results move** by the solvers' residuals: predicted at most 0.01 meV on
  hsuite `exciton_bands`, `bse` and `bse_bisp`.
- **Command lines.** `exciton_bands --block-size` is read by `--solver lanczos`
  only and refuses otherwise. `--max-iter` is the Davidson iteration cap,
  default max(30, min(200, n/2)) at transition dimension n; under
  `--solver lanczos` its default stays 40. `bse_lanczos.solve_bse_sharded`'s
  Python defaults are now `solver_kind="davidson"`,
  `davidson_precond="exact"` (were `lanczos` / `bare`), as on the CLI.
- **Open (audit S3).** The NSCF Davidson program's HLO differs once from
  main's (the new opt-in parameters are traced through); its results are
  bitwise equal, and the first NSCF run after the update recompiles.
- **Open (audit S4).** No test covers a multiplet straddling the `--n-eig`
  boundary, the case the margin roots exist for.

## 2026-10-21 — BSE lowest-N default is Davidson with a residual stop; Haydock runs in the eigensolve's process

`bse_jax --lanczos --tda` now solves the `--n-eig` lowest roots with block
Davidson (`--solver davidson`): the exact assembled diagonal preconditioner
(`--davidson-precond exact`), soft locking (a root under the tolerance adds no
correction, `solvers.plan_davidson(lock_converged=True)`), and a stop when
every root's residual `||Hx - Ex||` is below `--davidson-tol-mev` (0.01 meV;
an exact eigenvalue lies within the residual). The old default, single-vector
Lanczos at a fixed Krylov dimension, is `--solver lanczos`. On the Si 8^3 SOC
deck (8v x 8c, 512 k, dimension 32768, N_mu 1100, lowest 40), counted on the
dense operator assembled from the loader's own operands (whose tight
eigenvalues equal the GPU 1600-step Lanczos to 6e-12 meV): 19 iterations,
265 H applications against 1600, eigenvalues within 8.5e-8 meV of the
reference. Without locking the same solve takes 640-800.

- **Results move** by the solver's residual where the old solve converged:
  8.5e-8 meV on Si 8^3, 0.00 meV on hsuite `bse` (GPU gate, P=4). hsuite
  `bse_bisp` moves by -0.16 meV (S001) and -22.38 meV (S002; 31 % of the
  fixture's 72 meV DFT gap), and its reference is regenerated (-1.01590,
  -0.86783 eV): the old one came from main's explicit `--solver davidson`
  stopping unconverged at its 40-iteration cap (40/40 iterations, residuals
  16.1 / 106.4 meV), which main wrote with exit 0. The new values equal
  full-Krylov Lanczos (K = 50, the whole space) on both the old and the new
  restart.
- **Warning.** Results from explicit `--solver davidson` runs on main before
  this landing may be unconverged: check their logs (iterations against the
  cap, residuals) before reusing them.
- **Command lines.** A flag only another solver reads now refuses
  (`--lanczos-rtol`, `--lanczos-check-every`, `--n-reorth`, `--block-size` are
  Lanczos's; `--davidson-*` Davidson's; `--trlan-*` TRLan's): a line that tuned
  the Lanczos recurrence must say `--solver lanczos`. `--max-lanczos-iter` is
  the Davidson iteration cap. `--davidson-precond exact` also works under
  `--rpa` (diagonal `dE + V_x/nk`).
- **One process for eigensolve + absorption.** `--haydock-iter N` (with `--tda
  --dipole`) runs `bse.absorption_haydock`'s recursion after the eigensolve on
  the same payload and the solve's `W_R` (`solve_bse_sharded(keep_W_R=True)`),
  writing the same `absorption_haydock.h5` and `absorption_haydock_b*_eh.dat`;
  `--haydock-pol`, `--haydock-eta-ev`, `--haydock-omega-*`, `--haydock-n-omega`,
  `--haydock-n-spin`, `--haydock-no-eps1`, `--haydock-out-prefix` mirror the
  standalone flags, and the cell volume is the deck WFN's. This drops the second
  process's startup, restart load (25 s with `bse_k_grid` htransform + densify
  on Si 8^3), W transform and compiles.
- **Haydock.** `absorption_haydock --pol` (default `xyz`) recurses a subset of
  polarizations, one matvec each per step (`x` alone for cubic crystals); the h5
  arrays carry one column per polarization run and a `polarizations` attribute.
  The standalone route transforms `W_q` with it donated (one `(mu, nu, k)` tile
  per rank instead of two), places the seed block on the trial layout, and
  keeps the Lanczos vectors on it (main's step all-gathered the residual block
  for its norm). The IP JDOS builds one Lorentzian for all polarizations in
  bounded blocks (`absorption_common.LORENTZ_BLOCK_ELEMENTS`): Si 8^3, 4001
  omegas, 6.35 s -> 2.09 s on four cores, and no `n_omega x n_transitions`
  host array.
- **Explicit `--solver davidson` tolerance.** It was `atol` or 1e-8 Ry; it is
  now `--davidson-tol-mev` (0.01 meV = 7.35e-7 Ry), 73x looser in the residual
  and still a 0.01 meV bound per root (`--davidson-tol-mev 1.36e-4` is the old
  1e-8 Ry stop).
- **An unconverged Davidson refuses.** Any stop but CONVERGED (the cap is
  `--max-lanczos-iter`, default `max(30, min(200, dim // 2))`; a stall)
  raises `GATE davidson_converged` with n_conv/n_eig, the iterations against
  the cap and the largest residual in meV, before eigenvectors, spectrum or
  Haydock output are written. It used to warn and exit 0.
- **`--dipole-position-fix` on `--lanczos`.** It was missing from the
  eigensolve route's flags (since 7999900e3), so `bse_jax --lanczos
  --dipole-position-fix` was refused at parse time; it now reaches the loader.
- **Memory and CPU.** The Davidson basis and its images (2 m_max x N x 16 B / P,
  m_max = 10 n_eig: 128 GB / P at N = 1e7, n_eig 40) are priced and admitted as
  `bse.davidson_basis`. `GATE cpu_davidson_threads` refuses more than one
  emulated CPU device per process below 4 host threads per device, where the
  CPU subspace provider's callbacks used to hang until XLA's rendezvous timeout.
- **Test.** `tests/test_bse_default_solver.py` (script; 1x1 on a login node,
  2x2 with at least 16 host threads).

## 2026-10-21 — `bse_k_grid` `--eqp` dipoles: Rayleigh-quotient default, exact position opt-in (`--dipole-position-fix`)

Owner decision. Under `--eqp` the densified dipoles are again
`v/ΔE` of the fine states' Rayleigh quotients (what they were before
219ee8275). The exact-position fix of 219ee8275 (r from the DFT f(H)
eigenbasis of the window, rotated back) is now opt-in:
`bse_jax --dipole-position-fix`, `absorption_haydock --dipole-position-fix`,
`dump_window_dipoles --dipole-position-fix`, TD-aGW deck key
`[lorrax] dipole_position_fix = true` (bundle system), loader
`velocity_position_fix`.

- **Why.** The fix is exact only when the interpolated DFT f(H) is more
  accurate than the QP f(H) that defines ψ. Si 4³→8³ (near-scissor ladder):
  the DFT block mixes valence and conduction by 3.8 % and the fix puts the
  window-sum |d|² 12.1 % rms off the explicit 8³ dipoles (max 36 %) against
  2.4 % (max 6.5 %) for the default; IP ε₂ 2.2 % against 0.86 % rms of peak.
- **CrI3-type runs must opt in** (strongly band-dependent QP shifts). CrI3
  8×8→12×12 against the native 12×12: the default loses 1.4 % of Σ|d|²
  (0.986), 13 % below 1.6 eV (0.87), the lowest transition's quotient 60 meV
  high; the fix matches to 0.05 %. 8×8→30×30: 0.985 / 0.81.
- **Receipt.** Without the fix the log prints the fix's own mixing numbers
  (max valence↔conduction mixing of the DFT f(H) eigenbasis, max relative
  f(H) off-diagonal; `result["qp_mixing"]`). Large values say the run needs
  the fix. There is no automatic switch.
- **Results move.** `bse_k_grid` runs with `--eqp`: dipoles, oscillator
  strengths and absorption (not the exciton energies). TD-aGW bundles
  (`tdagw.td_system`) on a `bse_k_grid` deck with `eqp_file` now take the
  pair-rule position of `v/ΔE_RQ` unless `dipole_position_fix = true`; the
  CrI3 TD lane, for which the fix was built, must set it.
- **Code.** `densified_dipoles(..., qp_states=, mixing_receipt=)`:
  `qp_states` is the fix; `dump_window_dipoles` writes `d_alpha_rq` only with
  it. Bitwise: the opt-in path equals the old r-fix and the default equals the
  old `d_alpha_rq` (toy of `tests/test_bse_kgrid_dipoles_rfix_cpu.py`) for an
  isolated fit-window top. With an entangled top the opt-in also receives the
  weighted velocity (the entry below), so it is not bitwise the old r-fix
  there.

## 2026-10-21 — `bse_k_grid` dipoles: weighted velocity at an entangled fit-window top; `dipole.h5` band range

The densifier carries the coarse velocity through the htransform as
`g v g` when the fit window's top band meets the first unfitted band, and
divides `g` back in the fine window (`bse_densify.velocity_weight`,
`_velocity_from_weighted`). `g = 1 − exp(−x²)`, `x = (E_cut − ε)/w`, zero
above `E_cut` = the first unfitted band's lowest energy; `w` = half the
distance from the BSE window's top to `E_cut`. There the window projector
jumps where bands cross, and the unweighted velocity is not smooth in k (Si
4³→8³, 36 of 100 bands: image outer shell 26 % against 1.9 % for f(H)). A
window top that stays below every unfitted band keeps the unweighted route
bit for bit, and the on-grid identity stays exact either way.

- **Results move.** `bse_k_grid` dipoles when the velocity file holds bands
  above the fit window and they are entangled with it (every production
  deck so far). Si 4³→8³ kg_srlr, momentum operator, against the explicit 8³
  dipoles (448 off-grid k): velocity over exact ΔE, window sum, rms 6.1 % →
  2.7 %; with the default Rayleigh-quotient dipoles |d|² 6.5 % → 2.4 % rms
  (max 20 % → 6.5 %, BZ total +3.5 % → −0.3 %) and IP ε₂ at identical QP
  energies 1.37 % → 0.86 % rms of peak on 2.5–4.5 eV (peak +2.7 % → +0.3 %).
  kg_srlr as run on main (unweighted, r-fix): 13.7 % rms, IP ε₂ 3.0 %.
  Exciton energies do not move.
- **Gate.** `GATE densified_velocity_weight`: an unfitted band at or below the
  BSE window's top, or a weight below 1e-3 in the fine window. Fix: raise
  `ncond`.
- **`dipole.h5`.** Every writer stamps `band_min = 0`, `band_max = nb` (the
  absolute band range of the stored axis; `nbands` stays the WFN's count).
  No regeneration: an unstamped file reads `[0, nb)`. `GATE
  dipole_band_window` (`file_io.dipole.require_dipole_bands`) refuses a file
  that does not cover the reader's window from band 0: the BSE slicer
  (`absorption_common.slice_dipole_to_bse_window`, so `window_dipoles`,
  `exciton_bands` and `w_ladder`) and `head_dense_file`. A short file used to
  slice to fewer conduction bands without a word.
- **Test.** `tests/test_bse_kgrid_dipoles_cpu.py`
  (`test_entangled_window_top_weights_the_velocity`, band-window refusals).

## 2026-10-20 — BSE q=0 screened head uses the exact band overlaps (opt-in loader mode)

At q = 0, G = 0 the direct-kernel pair densities are band overlaps,
`<nk|n'k> = delta_nn'`, so the screened head is the transition-diagonal
constant `whead/(Omega N_k)`. The ISDF rank-1 `conj(g0) g0` on the W tile
evaluates the overlaps through the centroid fit and is not orthonormal. The
loader now has a mode, `q0_head`: `"rank1"` (default, unchanged: head on
W_q) or `"diagonal"` (whead off W, `data["q0_head_ry"]` set). The BSE drivers
(`bse_jax`, `absorption_haydock`, `exciton_bands`, `bse_feast`, `bse_kpm`)
request `"diagonal"` and every operator takes the scalar off its conduction
diagonal (`bse_head.head_shifted_eps_c`): TDA and non-TDA A block, Haydock,
FEAST/KPM, the preconditioner diagonal, `exciton_bands` at every Q. The
coupling block gets no head (`<ck|v'k> = 0`). On a diagonal bundle the
`bse_k_grid` densifier and `exciton_bands --w-coarse-grid` keep their q ≠ 0
heads and move fine Γ to the diagonal. `vhead` on the exchange tile is
unchanged. TD-aGW, `bse_w_exact`, `w_ladder` and `w_omega_chain` load
`"rank1"` and are bitwise unchanged.

- **Results move (BSE only).** Each exciton moves by
  `(whead/(Omega N_k)) (c_S - 1)`, `c_S` the ISDF head content of the state.
  MoS2 12x12, N_mu 1198: A/B up by about +4.3/+4.5 meV (c = 1.0076/1.0081;
  0.14 % of the 3.108 eV QP gap), which removes the BerkeleyGW–LORRAX rigid
  residual; the window admits -8.1 to +14.0 meV. Si 8^3, N_mu 1100: -0.04 to
  +0.09 meV.
- **Decks.** No change. `vhead`/`whead_0freq` overrides act as before.
- **Code.** `_inject_q0_head(..., nk=, q0_head=)` returns
  `(V_q0, W_q, q0_head_ry, log)`; bundles carry `q0_head` and `q0_head_ry`.
  `head_shifted_eps_c` refuses a rank-1 bundle with a nonzero scalar and a
  diagonal bundle without one. A new driver that requests `"diagonal"` must
  build its operands through the helper and be added to the test's list.
- **Test.** `tests/test_bse_q0_head_exact_overlap.py`.

## 2026-10-21 — the scalar head pins its Z moment; metals: generalized Drude replaces the cell Lindhard

The scalar head's pole model now carries both on-shell moments exactly:
W^c_h(0), as before, and M_1 = sum_p B_p/Omega_p^2 = (1/pi) int_0^inf dxi
[W^c_h(i xi) - W^c_h(0+)]/xi^2, the slope d Sigma_h/d omega = -M_1/(Omega N_k).
M_1 comes from 24 dedicated imaginary-axis head samples (Gauss-Legendre in t,
xi = 0.5 Ry t/(1-t); `gw.qsgw_head.head_quadrature_rule`), each a full head
build folded through W_body(Gamma, i xi): exact on shared-pole stores, the
fitted elementwise body on `sigma_w_model = mpa`.  `_pin_static_head_sample`
adds the constraint sum B/Omega^2 = M_1; M_1 is stamped in the head store
(`fit_moment_m1`, beside the fit's own `fit_moment_m1_unpinned`;
`read_head_fit_collective` returns `moment_m1`); the shared-pole route prints
both values in the rank-0 log.  Heads with no rule keep the
W(0)-only pin and build no extra rows: `off`, the BGW finite-q0 head, the
w_bse ladder head, the `full_shared_pole` four-current sector head.

On a metal, every dynamic head row is now the generalized Drude term
D r(z)/z^2 with a Fermi-liquid memory function (deck `head_drude_gamma0_ev`,
default 0 = the undamped D/z^2, and `head_drude_gamma_width_ev`); the
per-sample anisotropic Lindhard function leaves the dynamic head (its M_1 is
log-divergent in the cell's smallest q.u).  The static row is Thomas-Fermi on
every route, folded or not.  M_1 subtracts the Drude model's own limit
W^c(0+) = -<v>, not the Thomas-Fermi row.

- **Results.** eqp0 is unchanged (W(0) pinned as before): 0 at print
  precision on Si 4^3, Si 6^3 and Fe 4^3.  The head's slope moves by the same
  Delta(d Sigma_h/d omega) = -(M_1,quad - M_1,fit)/(Omega N_k) for every state;
  Z and eqp1 move through it state by state.  Si (dense 16^3 head, 8 poles):
  M_1 +0.66 % (4^3) and +0.32 % (6^3); eqp1 QP gap -0.19 meV (-0.016 % of
  1165.0 meV) and -0.06 meV (-0.005 % of 1133.3 meV).  Fe 4^3: M_1 +11 %
  (the fit misses the Drude head's small-xi weight), eqp1 -15.4 meV mean
  within 1 eV of mu (spread 12.4 meV), |Delta Z| <= 0.030.  N = 24 and 48
  agree to 3e-10 on Fe.  The hsuite `sp_export` (also the `no_local_fields`
  deletion above) and `na_sc` references are regenerated.
- **Off-shell cost.** The poles stay fixed, so the second constraint is
  carried by the residues alone and the fit's residual at its other samples
  grows: Si 4^3 0.89 -> 25.3, Si 6^3 3e-7 -> 43.3, Fe 4^3 6.6 -> 31.7
  Ry bohr^3.  The off-shell head Sigma moves by at most 3.5 meV within 3.4 eV
  of the shell and 22 meV within 6.8 eV (Si 4^3, eta = 0.25 eV); readers of
  the real-axis model (`sigma_omega_grid_debug`, uncovered states, the SC
  census) see it.  A refit of the poles under both constraints is not done.
- **Cost.** 24 more head rows per screening build: +2.4 s on the Si
  shared-pole head stage (1.7 -> 4.1 s); on `sigma_w_model = mpa` each row
  also evaluates the fitted Gamma body.
- **Users.** No deck change.  Test: `tests/test_head_quadrature_moment.py`.

## 2026-10-20 — `head_correction = no_local_fields` is deleted; `full` is the one head

The deck value `no_local_fields` is gone and refuses as an unknown policy; use
`full` (the default) or `off` (debug). Ruling:
[decisions, 2026-10-20](docs/architecture/decisions.md#one-head).

- Scalar decks that set `no_local_fields` (shared-pole one-shot and SC) now fold
  the head's local fields once through W(Γ). Every state moves by the same
  `ΔΣ_h = (1/2 − f) ΔW^c_h(0)/(Ω N_k)` on shell, so the gap moves by
  `ΔW^c_h(0)/(Ω N_k) = v·Δ(1/ε_M)/(Ω N_k)`; eqp1 also moves through the head's
  frequency slope. The hsuite `sp_export` reference is regenerated: on its
  fixture the folded W^c_h(0) is -923.325 Ry bohr^3 against the direct
  -1382.833, so every eqp0 moves by -173.665 meV (occupied) or +173.665 meV
  (empty), the gap by +347.33 meV, as predicted; its eqp1 moves also through
  the pinned M_1 (entry above).
- `bispinor_gw = full_shared_pole`: `full` (now the default and the only head
  value) is the existing direct sector head; numbers are unchanged
  (`bisp_sc`, `bisp_sc3`, `bse_bisp`). The run record states "no local-field
  fold on the four-current bank". `GATE full_shared_pole_head` and
  `GATE bispinor_head_correction_no_local_fields_unavailable` are deleted.
- Shared-pole `bare_transverse` (four-spinor charge carrier): `full` now folds
  the head like the scalar store instead of refusing
  (`GATE shared_pole_head_nspinor` is deleted); its charge operator, wing
  vertices and velocity are spin-traced on the one carrier. Test:
  `tests/test_shared_pole_head_ordered_fold.py::test_four_component_carrier_wings_are_spin_traced`.

## 2026-10-20 — BSE `--eqp`: a full-zone file is read per k; star-asymmetric QP shifts refuse

`file_io.restart_bundle.apply_eqp_corrections`, the eqp reader of every BSE
frontend (`bse_jax`, Haydock, Davidson, the direct restart loader, exciton
bands), now reads a file with exactly one block per full-BZ k block by
block, matched by crystal coordinate mod G to 1e-6, as BerkeleyGW does. It
used to unfold such a file from its star representatives: on the Si 8³
full-zone WFN 29 of 512 k kept their own block and the other 483 blocks were
discarded. A file-wedge eqp file, what LORRAX's GW writes on a reduced WFN,
is unfolded through the symmetry service as before.

- **Results move only for star-asymmetric full-zone files, and those now
  refuse** (`GATE eqp_star_spread`). The gated quantity is the QP shift
  E_QP − E_DFT averaged over each DFT multiplet (bands within the 1 meV
  `DEGENERACY_TOL_RY`), which may differ across a symmetry star by at most
  1 meV; the per-band values are applied as given. Each read prints the
  multiplet-mean and band-by-band star spreads and the largest spread inside
  one multiplet (0.396 meV, bands 17–20, on both Si 8³ files below: 0.041 %
  of eqpA_main's 965.03 meV QP fundamental gap, direct 3072.79 meV). A
  star-symmetric file gives the old energies to its print precision: Si 8³
  `sidirect/d8/eqp1.dat` bit for bit, `si8v2` `eqpA_main_si.dat`
  (htransform at 6446c17d3) within 1.0e-6 meV (1.0e-7 % of its 965.03 meV
  gap). Wedge files are bit for bit (Si 4³ `eqp1.dat`, `eqp1_a.dat`). The old
  Si 8³ `sidirect/eqpA_si.dat` (36 bands; QP column of bands 1–16 from the
  QRCP htransform) refuses at 476.025 meV (bands 15–16, k = (3/8, 3/8, 3/8)
  against (3/8, 0, 0); 50.9 % of its 935.54 meV QP fundamental gap, direct
  3072.76 meV). LORRAX BSE runs on it applied star-representative energies
  that BerkeleyGW did not; rerun them on a symmetric file.
- **htransform.** `--eqp-file` (`read_eqp_energies`) uses the same reader
  and the same gate, with E_DFT from the file's own column: a full-zone
  coarse eqp file keeps its per-k blocks, a star-asymmetric one refuses, and
  a wedge file reads bit for bit as before.
- **Bands gated.** The BSE reader gates the bands that both the file and the
  deck's spectrum carry; htransform gates every band of the file, before its
  band window is cut. A multiplet that the file's top band cuts (visible on
  the BSE path, whose spectrum has more bands) is reported, not gated.
- **Newly accepted:** a full-zone file on a reduced WFN, which used to
  refuse on its block count.
- **Newly refused:** a full-zone-sized file in which some full-BZ k matches
  no block or two; on a full-zone WFN such a file used to be applied by
  position. A file with more blocks than the full BZ refuses by name
  (`GATE eqp_full_zone_count`; e.g. Si 8³'s 512-block file on a Si 4³ deck),
  as it refused before on its block count.
- **Decks.** No change. The 1 meV tolerance is a constant
  (`EQP_STAR_SPREAD_TOL_EV`), not a key.
- **Tests.** `tests/test_eqp_fullzone_cpu.py`: wedge file bit for bit;
  symmetric full-zone file equal to the unfold; per-k blocks, with the
  pre-fix unfold as control variant, in both readers; asymmetric refusal in
  both readers; off-grid, duplicate-block and superset refusals; a gauge
  split inside a multiplet that passes and is applied per band; a
  multiplet cut by the file's top band, reported and not gated.

## 2026-10-20 — metal head: thermal near-degenerate pairs leave the static interband head

On a smeared metal, a pair split by less than the cell scale but off the
tetrahedron Fermi surface (`w = 0`) kept its whole interband weight, which grows
as `f'/Delta^2` at `z = 0`: on Fe 4^3 two pairs 1.9 eV below mu, split by
0.19 meV, gave the static interband head `8 pi S_xx = -1.5e4` (z: -146).  The
pair split (`gw.fermi_surface.intraband_pair_fraction`) now also moves pairs
weighted by the smeared occupation difference; the moved share enters the
intraband term with its tetrahedron weight, zero there.  S, the wings and the
four-current head take the one rule; D and the atoms are unchanged.  Metals
with such pairs move (Fe 4^3 static head isotropic, `-74`); insulators are
bitwise unchanged.  Test: `tests/test_shared_pole_head_ordered_metal.py::test_d`.

## 2026-10-20 — ordered metals under the full head; `GATE shared_pole_head_ordered` is gone

An ordered (time-reversal-broken) metal with `head_correction = full` now folds its head
instead of refusing (`GATE shared_pole_head_ordered` is deleted). The wings
split each pair like S and D (`MetalPairSplit`): the interband wing keeps
1 − φ and the intraband wing −pref·φ·w̄/z takes the rest, a trace over each
degenerate multiplet. The former intraband wing used only the diagonal of each
group, had the opposite sign, and was cancelled only by time reversal, so its
value depended on the gauge inside the multiplet. The metal's shifted origin
(z = i·`mpa_metal_origin_shift_ry`) is now the static Thomas–Fermi row: S, the wings and the
shared-pole Γ body are evaluated at z = 0 there (`metal_static_rows`). The
metal S kernel clips only the resonance; it no longer drops pairs with
ΔE < 4.6 × 10⁻⁶ Ry at z = 0.

- **Results.** Insulators are bitwise unchanged. Time-reversal-even SC metals
  move from the second map on: hsuite `na_sc` map 1 moves occupied states by
  −27.6 meV and empty states by +28.2 meV (semicore 0). The wing split alone or
  the static row alone gives this result; reverting both reproduces the former
  value bit for bit. The `na_sc` reference (`tests/hsuite/reference/na_sc.npz`)
  is regenerated. Fe 4³, head full, with the thermal-pair rule above: static
  interband head 8π q̂·S·q̂ = −74.64 / −73.98 / −73.98 (principal axes; −15060 /
  −15060 / −146 before), W^c + ⟨v⟩ = 8.214 Ry bohr³ at the origin, rigid head
  shift W^c/(2ΩN_k) = −1.935 eV, spread of states within 1 eV of μ 6.7 meV rms,
  10.2 meV max.
- **Decks.** No change. Metal decks that refused with
  `GATE shared_pole_head_ordered` now run.
- **Test.** `tests/test_shared_pole_head_ordered_metal.py`.

## 2026-10-20 — htransform nodal basis (replaces the randomized-QRCP Galerkin basis)

The htransform basis (`isdf.galerkin`) is now nodal: the fitted states'
values at an orbit-closed node set, an orbit-tied quadrature metric and a
per-k Löwdin frame. f(H) is a node-pair kernel continued off the coarse grid
with per-pair Wigner–Seitz images (`common.ws_images`), and every fine-k,
path, active-window, orbital and moment-grid solve is a top-block subspace
solve whose route (`q-batched` or `face`) the memory model admits
([how-to](docs/how-to/htransform-and-exciton-bands.md) §2–3).

- **Results move.** Every htransform band structure, every `bse_k_grid`
  (fine-k BSE) result, `exciton_bands` and the `vq_interp` refit inputs are
  new numbers; recompute them. Off the coarse grid the bands are
  star-covariant: Si 4³→8³ (default 780 nodes) star spread 239.3 meV under
  the QRCP basis, 2.1e-7 meV now; off-grid DFT MAE 41.6 → 6.78 meV (max 204
  → 28.0; 0.63 % / 2.6 % of the 1080.4 meV QP fundamental gap of the deck's
  eqp1_a.dat, direct 3072.8 meV). MoS₂ 9×9→12×12 (the 1198 centroids): star spread
  9.3 → 6.5e-6 meV, MAE 2.30 → 2.31 meV (0.07 % of 3230.8 meV, direct
  3552.9). On the grid they reproduce the fitted energies.
- **hsuite.** `reference/htransform.npz` and `reference/exciton_bands.npz` are
  rebaselined (other references unchanged). Every hsuite path k and exciton Q
  lies on the 5×5×1 coarse grid, where the nodal basis reproduces the
  fixture's DFT energies to 1.8e-3 meV; the old reference carried the QRCP
  basis's on-grid error (up to 0.584 meV in `htransform.dat`, 0.10 % of the
  fixture's 577.2 meV QP fundamental gap, direct 841.3 meV; and −1.72 /
  +1.65 / −4.06 meV, up to 0.70 %, in the exciton energies at Γ / X / M).
- **Decks.** `htransform_rank_multiplier` and `htransform_qrcp_seed` refuse
  by name (`GATE htransform_qrcp_retired`): delete them. `htransform_qr_eps`
  is kept, but it is now only the relative pseudo-inverse cutoff of the
  refit's node Gram; the fit has no tolerance.
- **Centroids and nodes.** Orbit-closed sets are mandatory: a centroid or
  node set that is not closed under the WFN's operations refuses (regenerate
  it with `centroid.kmeans_cli --orbit`). The nodes are the ISDF centroids
  up to 1.25 × 22 × fitted bands; above that, an orbit-closed k-means
  selection of 22 × fitted bands is made once and cached beside
  `galerkin_dft.h5` as `galerkin_nodes_<N>_b<b0>-<b1>_<wfn>_<params>.txt`
  (`params`: a hash of the selector, its version, the k-means version
  `centroid.kmeans_isdf.KMEANS_VERSION` and the arguments). Rank 0 decides
  the cache hit and its node set reaches every rank
  (`common.collectives.rank0_read`), on a hit and after a fresh selection
  alike. Zero-weight nodes
  (W = 0) are dropped from the rank axis, which is exact. The states at the
  centroids, which the BSE consumers read, are back-map targets whenever the
  nodes differ from them. The fit warns when the metric eps exceeds 0.1
  (too few nodes).
- **Files.** `galerkin_dft.h5` is format 6 (node indices, the kept-node
  mask and target states added). An older file is left untouched and the basis is refitted in memory
  on every run; delete it to publish the format-6 basis once.
- **Memory.** The fine-k stage holds f(H)_R and the back-map L_R,
  `16·N_k·N_B·(N_B + N_T)/P` bytes per rank (`N_B` = spin × nodes, `N_T` =
  spin × centroids, 0 when the nodes are the centroids), plus int16 WS code
  tiles `2·N_k·N_B·(N_B + N_T)/P`; the lattice-box R tensor it replaces was
  `16·N_k·rank²/P` without the back-map. Both are admitted (`lxkit.memory`).
  Si 4³ at P = 4 (768 nodes after dropping 12 zero-weight ones, 1100
  targets): 1.65 GB per device against 0.10 GB for the QRCP box (rank 636);
  device peak of the fine stage 4.11 against 1.01 GiB. MoS₂ 9×9 (878 of the
  1198 centroids kept, 1198 targets): 2.66 GB per device, peak 6.79 GiB.
- **Tests.** `tests/test_nodal_htransform_cpu.py`: star covariance, on-grid
  exactness, off-grid accuracy against the control variants (sign-flipped
  pair displacement, one image set per class, the box), the orbit-tied
  metric, the top-block solve at block ≪ rank with its `max_iter = power`
  control, and the back-map through a non-uniform metric.

## 2026-10-20 — `head_dense_file`: a dense-k direct head for insulators

New deck key, off by default (`""`): a deck without it, or naming the GW mesh's
own `dipole.h5`, builds today's head bit for bit.  It names a `dipole.h5` of a
denser NSCF of the same crystal and Hamiltonian; its full-zone S_d(z) replaces
the GW mesh's at every response node, while the wings, W(Γ) and the mini-BZ
cell stay the GW mesh's.  The fold is the GW mesh's local-field fold scaled by
M = S_d,dense S_d,mesh⁻¹ (`gw.head_correction.scaled_head_fold`; +1.2 % in
ε_M on Si 4³ with an 8³ set, docs/theory/s-tensor-convention.md#scaled-fold).
The persisted `S_cart_head`, and so the BSE, inherit it.  Metals refuse
(`GATE head_dense_metal`), as do another operator, crystal or band window
(`GATE head_dense_operator` / `head_dense_lattice` / `head_dense_window`).
Recipe: QE NSCF on the dense mesh from the GW deck's SCF (nbnd ≥ occupied + 28),
pw2bgw, then `python -m psp.get_dipole_mtxels -i <dense deck>`.
Test: `tests/test_head_dense_source.py`.

## 2026-10-20 — one velocity operator: the V_NL sign is gone; regenerate old velocity files

The velocity is $\partial_k H = i[H, r] = p + i[V_\mathrm{NL}, r]$, a unique
operator, so the knob that could subtract the nonlocal term is deleted: the
deck key `vnl_velocity_sign`, the dipole step's `--vnl-velocity-sign`, the
`prov_vnl_velocity_sign` stamp, and the sign argument of
`common.mtxel_sweep.dipole_operator` / `uniform_gauge_operator` and
`psp.get_dipole_mtxels.compute_finite_q_mtxels` (whose keyword default was
the subtraction).  A deck that still names the key reports it as unrecognized.

- **Regenerate every `dipole.h5`.** Every `dipole.h5` written before this
  commit must be regenerated with `python3 -m psp.get_dipole_mtxels -i <deck>`.
  Readers refuse a file without `format_version = 2`
  (`GATE dipole_format_version`); there is no migration reader. The same
  command rewrites the parallel-transport artifact (now schema 5) and the Hall
  artifact (now schema 3), whose readers refuse older schemas (the velocity
  reader of `sc_head_update = dft_velocity` used to accept schemas 2–4 without
  checking the operator).
- **`--skip-vnl` is unchanged.** The bare momentum $p$ (BerkeleyGW
  `use_momentum`) stays a separate operator, not a sign; its files are
  regenerated the same way.
- **Results.** Values built at the former default (+1) are unchanged; the Hall
  operator fingerprint of a +1 artifact is byte-identical.

## 2026-10-20 — c1 head channel: the q ≠ 0 projector carries the gauge phase

`w_head_densify = c1` re-attaches W's Γ head at every fine q inside the coarse
Γ cell as `S(q)/Ω · conj(g₀(q)) ⊗ g₀(q)` with `g₀(q)_μ = g₀_μ e^{−2πi q·r_μ}`
(`gw.head_densify.attach_head_channel`; `r_μ = idx/N` from
`bse.bse_densify.w_tile_centroid_frac`, the rows the W densifier embeds; q the
shortest representative). It used `g₀(q) = g₀`, the Γ vector without the
full-Bloch gauge phase of the W tiles, which broke the star covariance of the
c1 tile at the coarse Γ cell's boundary q (Si 4³→8³ c1: 0.17 relative at the
first shell; the body is covariant to 2 × 10⁻⁸).

- **Results.** c1 runs densified across grids move at the fine q of the coarse
  Γ cell other than Γ: `bse_k_grid` with a fine grid other than the coarse one
  and `exciton_bands --w-coarse-grid`. `srlr`'s long-range model rows
  (`_srlr_terms`) now take their Bloch phase on the same `idx/N` positions
  instead of the centroid file's six-digit text (at most 3.3 × 10⁻⁷ in
  fractional units on the shipped Si table), so `srlr` tiles move at that
  rounding level. `fine == coarse` and undensified runs are unchanged bit for
  bit. No deck change is needed.
- **API.** `attach_head_channel(..., *, centroid_frac, bvec)`: both keywords are
  required. A run reads and receipt-checks the tile positions once
  (`w_tile_centroid_frac`) and passes them on:
  `w_pair_embedding(coarse_grid, wfn, centroid_frac, n_mu_carrier, sys_dim)`
  takes the positions instead of the centroid file, and `_srlr_terms` takes
  `centroid_frac=`. `build_w_head_channel(..., geometry=)` takes the
  `CoulombGeometry` whose `bvec` the caller passes to the attach.

## 2026-10-20 — BerkeleyGW exports: `epsmat_ecut` and the finite-Q `bsemat`

`[epsmat]` export: `epsmat_ecut` (Ry, default 0 = the full sphere) replaces
`epsilon_cutoff`, which now refuses by name (`GATE epsmat_retired_key`). It sets the
written G cutoff with or without `bgw_reference`; a positive value writes the leading
sub-block of the converged ε⁻¹ in BerkeleyGW's G order, not a re-inversion
([theory §3](docs/theory/bgw_epsmat_export.md#ecut)). `bse.bsemat_cli --exciton-Q Q1 Q2 Q3`
(crystal, default 0) writes `kernel.x`'s finite-momentum TDA kernel: row k holds the
electron at k and the hole at k − Q, `exciton_Q_shift` = −Q, `qflag` 2 on the restart
grid, and `qflag` 0 off it with `--wfnq-co`. The exchange tile is V at wrap(+Q) with every
G ([theory](docs/theory/bgw-bsemat-export.md#finite-q)).

- **Results.** Unchanged at the same cutoff and at Q = 0: a cut file equals the full
  file's leading block (1.0 × 10⁻¹⁵ on Si 4³), and the Q = 0 `bsemat.h5` equals the
  previous writer's, all four matrices. New outputs: on Si 4³ the 12 Ry file moves the
  lowest 40 absorption energies by at most 0.078 meV (0.007 % of the 1.080 eV QP gap;
  direct gap 3.072 eV) against the 25 Ry file. On MoS₂ 9×9 at Q = (1/9, 0, 0) the file
  agrees with BerkeleyGW's own kernel to 0.42 meV MAE, 0.88 meV max (0.024 % and
  0.051 % of the 1.732 eV gap of the DFT energies used; the fundamental gap is the
  direct gap at K).
- **Decks.** Rename `epsilon_cutoff` to `epsmat_ecut` in `[epsmat]` files. An export
  without `bgw_reference` no longer needs a cutoff: 0 writes the producer's
  `bare_coulomb_cutoff` sphere (`ecutwfc` when unset).
- **Receipts.** The `gauge` value of a `bsemat` file's `<file>.json` for the identity gauge
  now reads `identity (control variant)`.
- **Fixed.** `--mu-small` exports, which refused on main since the downfold API change
  (6ec872560), run again.
- **Open.** Off-grid Q (`qflag` 0) is implemented but not validated, and it refuses
  unless the restart's ζ stores the full zone (`GATE bsemat_finite_q_offgrid`; gw_init
  stores the IBZ by default). `bse.exciton_bands` still contracts its finite-Q exchange
  at wrap(−Q): bright states at that MoS₂ Q are off by up to 6.6 meV (0.38 %); Q = 0 is
  unaffected. BerkeleyGW 4.0's `qflag` 2 route has two defects of its own; the theory
  page gives the workarounds.

## 2026-10-20 — `band_extrapolation_checkpoints`: several Σ band counts from one sweep

New deck key, off by default (`""`): a run without it is unchanged (same plan,
same sweep, same reduce).  `band_extrapolation_checkpoints = 627, 882, 1251`
gives each count its own three-count plan (the plan of a run with
`number_bands_sigma` = that count), sums the disjoint brackets between the
union of all their cuts and the run's own once, and writes
`sigma_band_checkpoints.h5`/`.dat` beside `sigma_diag_file`: raw S(N) and
spectral-shell Ŝ(N) band-diagonal Σ_c on shell, dΣ/dω and Z at every count, the
fit's Ω/β/tail ratios, and the raw diagonal of every cut.  W, the windows and
the evaluation energies are the full run's.  The run's own count reads its three
counts from the union sweep (`BandCountCube.restrict`; its N1/N3 matrices are
kept, `_on_shell_reduce(..., keep)`), equal to the plain run to round-off.
Scalar on-shell MPA one-shot only (`GATE band_extrapolation_checkpoints`).
Tests: `tests/test_band_checkpoints_cpu.py` (each checkpoint equals the
production extrapolation of its restricted plan to rtol 1e-11).

## 2026-10-20 — memory model: prices from measurement

`lxkit.memory` prices, admits and receipts every stage
([memory model](docs/architecture/memory-model.md)). Stage prices come from the
stages' own programs compiled and from provider size queries; distributed and
local routes are chosen by them (`distrib_la.place`, `staged_fits`, the ζ, V(q)
and Galerkin choosers), and every whole N_μ-scale site has a mesh route. The
cuSOLVERMp LU chunks `getrs` to the posted room.

- **Results.** Unchanged within the solvers' tolerances; routes can move where
  the compiled figure differs from the old closed form.
- **API.** `common.gpu_utils.warn_over_budget`, `record_stage_price`,
  `set_device_budget_gb` and `stage_prices` are gone: use `lxkit.memory.admit`,
  `book` and `set_budget`.
- **Environment.** New diagnostics: `LORRAX_MEMORY_RECEIPTS=calibrate`,
  `LORRAX_MEMORY_STRICT=1`, `LORRAX_MEMORY_TRACE=<path>`, `LORRAX_NMU2_BACKUP=report`
  ([env vars](docs/reference/env_vars.md)). No deck change.
- **Open.** Sixteen route-chooser compiles still price a program that does not run
  (CPU hsuite: `matmul.route_bytes` 10, the head Γ and ζ staged GEMMs; `place()`'s
  `mesh_bytes` 4 and `_gathered_admitted` 2): the chooser should compile the caller's
  program once at the proposed route and run it. Six of them, on Frontier Fe, are not
  kept by the persistent cache: the Dyson LU decision (`gw.response_bank._Decided.resolve`
  -> `Plan.decision` -> `place`) compiles route-(c) `solve_lu` candidates outside
  `distrib_la.checked_program`, so each carries a host callback. `spole.gram_reduction`
  and `spole.bank` price above the 1.3 band (the face round's row holds its program's eigh
  twice); `spole.direction_selection` prices 1.9x its peak (run30k 46.86 vs 24.44 GB:
  the conservative side and the selection eigh). A kernel-priced selection was tried and
  dropped (its key broke the receipt rows and it compiled in the width search). The run30k
  minimum-P binder is `zeta_fit.mubatch.write_zeta`, 53.44 GB against 51.46 measured.
  After the landing gate (run30k, REPORT6) `spole.bank` 2.0x, `spole.moment_read` 2.2x and
  `spole.sample_batch_read` 1.6x are over too. (`bank.moment_batches` was 54.73 GB, a
  closed-form batch tile booked beside its compiled dispatches; it books none now and prices
  24.67 GB against an 11.23 GB trace peak. Its store write rows still price each field as
  output + temporary + the live input, 4 x 4.5 GB for M0..M3 at run30k against about 2.7 GB
  of measured staging. The batch's read-back, split and photon canonical copy run as priced:
  `bank.moment_correlations` 10.36 GB priced and measured.) The
  selection's `phase_dense_temporaries` (24 n^2/P per parent, 21.6 GB at run30k) duplicates
  the direction-SVD program, which should price it at the stack it runs. The
  native ncclMin agreement in the LU workspace query (B3) and the eigh memory change
  (ABI 11) are not in this landing.

## 2026-10-20 — W densifier: per-centroid-pair Wigner–Seitz embedding

`bse_k_grid` and `exciton_bands --w-coarse-grid` now embed each coarse R class
of the W tile W_μν(R) at the fine-lattice images R + L·T of minimal
|R + L·T + r_μ − r_ν|, weight 1/N_deg over ties
(`bse.bse_densify.pad_W_R_to_grid` through `common.ws_images`). This entry
supersedes the 2026-10-16 W-densifier entry: the lattice-coordinate box
(`_zeropad_R_axis`, with its Nyquist split) is gone, because it is not
covariant under point operations that fold the centroids into the home cell
non-uniformly (Si 4³→8³ W body: star covariance 0.24–0.28 with the box,
1.3–1.5 × 10⁻⁷ now). The densifier refuses (`GATE w_pair_centroids`) a
centroid table that does not match the restart's `centroids_charge_md5`
receipt or the tile's μ row count.

- **Results.** Every densified run moves and must be recomputed:
  `bse_k_grid` with a fine grid other than the coarse one and
  `exciton_bands --w-coarse-grid`, `c1` and `srlr` alike, odd coarse axes
  included (Si 4³→8³: the W body changes by 19–27 % at the first three
  off-grid shells). Undensified runs are unchanged. No deck change is needed.

## 2026-10-19 — retarded-only equilibrium-reference GW snapshots

`collision_statistics = "retarded_reference"` selects an explicitly
equilibrium-like approximation: full instantaneous density and Hamiltonian
build the retarded polarization and screened interaction; their positive
and negative frequency pole branches supply the zero-temperature GW
self-energy reference. All electronic density entries and all complex
self-energy entries are retained, including off-diagonal coherent and
dissipative terms at the instantaneous column energies.

This choice fits only the retarded response and does not require an
independent lesser/noise model or its positivity/FDT gates. Signed residues
are retained; fit accuracy, q/$-q$ pairing and runtime electronic
diagnostics still apply. Growing retarded poles are permitted only when
the explicit collision evaluation height exceeds their growth bound, so
the one-sided self-energy integral converges; the margin is reported and
no pole is shifted. Response lag decay and collision
regularization remain separate controls. The branch rule is an explicit
approximation, not an inference of exact nonequilibrium fluctuations from
retarded screening. Frozen and updated runs use the same rule.

The Hamiltonian inner solve now holds its numerical polarization $A,B$
fitting space fixed and refits $C$ from current full-density samples on
each sweep. Dyson screening and the complete complex self-energy still
update; no physical collective pole or screened interaction is held fixed.
The space is rebuilt at each physical refresh. This removes fresh-fit
rank/pole selection from the inner convergence map without loosening the
Hamiltonian or response-fit tolerances. A fixed-basis check failure refuses
instead of silently rebuilding inside Anderson history, and failures now
retain scalar residual histories. Updated-screening trajectories and
iterative absolute-HF initialization can change; one-shot reference
equilibrium fits and their frozen-screening control retain their prior
fit path. No deck change is required.

## 2026-10-19 — explicit lossless transition-bath screening statistics

Joint collision modes accept the opt-in
`collision_statistics = "lossless_bath"` prescription described in
[theory](docs/theory/td_agw_dynamical.md#lossless-transition-bath).
It constructs real-pole RPA screening with transition-specific colored-bath
occupations, keeping response-free fractional-population noise. Thermal
populations give screened collective-pole KMS weights, including zero
positive-frequency lesser weight at zero temperature. The earlier
`finite_lag` model remains a separately labelled choice.

This approximation excludes coherences between distinct instantaneous
energies from **screening only**. The density propagated by the electronic
equation, coherent Hamiltonian and column-sampled electronic collision
lines retain the full density matrix. It must not be interpreted as
full-coherence screening. `joint_frozen` freezes the same bath construction
at equilibrium for matched comparisons. The lossless pole bank does not use
`collision_adiabatic_ev`; collision regularization remains a separate
control. No restart is interchangeable across statistical selections.

Screening transition gaps at or below `1e-12` Ry are resolved as degenerate;
the receipt reports the gap adjustments, while Hamiltonian and electronic
self-energy energies are unchanged. The final response is checked against
the original gaps. Unequal-population degenerate channels use an explicit
weak shot-noise reservoir; equal-population degenerate channels retain
exactly static Wick noise without finite-frequency collective tails.
Their response is aggregated before screening, and a kernel Schur solve
retains bare zero-frequency noise after response cancellation. Thermal KMS
applies to occupations thermal on this resolved screening model, including
equal populations at degenerate levels. It is not a claim for arbitrary
original tiny-gap occupation mismatches.

The reduction preserves signed Hermitian transition structure and thermal
residue normalization; it does not clamp complex poles or infer noise from
the retarded residues alone. An optional preliminary energy-bin Galerkin
projection reduces transition-space cost while projecting the same colored
statistics. Distinct q/$-q$ partners are derived algebraically. Self-inverse
q banks use an explicit convex average with their crossing partner at the
screened-interaction level; this final average need not equal Dyson screening
of the original compressed polarization. Independent microscopic response
checks bound the final approximation. Unstable, unresolved or singular pole banks
refuse. Full-matrix bath checks do not establish an electronic Pauli theorem:
number, gain/loss boundary and propagated-density checks remain necessary.
The construction changes collision results and supplies a declared
approximation for testing damping, not a completed material validation.

## 2026-10-19 — screening statistical preparation: CPU references and scope

New small CPU references distinguish correlated Gibbs, isolated adiabatic,
and quench/local-noise RPA statistics. They verify a thermal target and a
finite two-endpoint correlated coherent-kick construction, and reject both
the stationary arbitrary-coherence spectrum and an equilibrium-KMS plus
old delta-bubble shortcut. Equations, assumptions and limits are in the
[statistical preparation section](docs/theory/td_agw_dynamical.md#keldysh-screening).

There is **no propagation change**: no new statistical mode, deck setting or
restart format. The `finite_lag` guard remains; these references do not repair
the general mixed-density runtime. The coherent-kick reference is a declared
quasiboson preparation for isospectral pure projectors, not an exact fermionic
covariance or a general collision-positivity result. No damping claim follows
from its CPU tests.

## 2026-10-19 — joint collision statistics now require explicit acknowledgement

Active `collision_screening = "joint"` or `"joint_frozen"` decks now refuse
at `td_collision_statistics` before GW setup unless they explicitly set
`collision_statistics = "finite_lag"`. The default is `"unspecified"`.
Collision-off and legacy frozen defaults are unchanged. Restart fingerprints
include the selection, and joint refresh receipts record the known limitation.

This is a breaking safety guard, not a statistical repair. The implemented
finite-lag model does not generally satisfy screened equilibrium FDT; an
exact scalar counterexample retains positive-frequency lesser collective
weight even as `collision_adiabatic_ev` tends to zero. That historical key
specifies finite lag decay, not a validated physical adiabatic limit.
Physical lossless collective statistics/adiabatic switch-on remain
unimplemented. Select `finite_lag` only for explicitly labelled model
experiments; do not infer physical thermal GW relaxation from convergence.
See the [analytic counterexample](docs/theory/td_agw_dynamical.md#damping).

## 2026-10-19 — TD-aGW collision: column-sampled one-sided MGKBA operator

Collision-enabled propagation replaces the arithmetic average of the row- and
column-shell self-energy tables with the one-sided Markov operator of
[TD-aGW dynamical §1.5](docs/theory/td_agw_dynamical.md#damping). The particle
retarded table is sampled at the column energy; the retarded hole table is
the adjoint of the advanced row table. Their complex principal-value terms
enter the coherent update once, alongside the held collision coefficients.
This changes collision results, including off-diagonal dynamics. The defaults
of `collision_refresh_sweep`, `collision_in_scattering`,
`collision_rate_intermediates`, `collision_basis` and
`collision_internal_pairing` are now respectively `active`, `full`, `active`,
`band` and `vertex`; decks that need an earlier comparison must set
all of its choices explicitly while the collision is off. Collision-enabled
propagation hard-refuses populations-only lines, all-band rate intermediates,
and symmetric internal pairing: after column sampling only the closed active,
full-ρ vertex construction passes the reciprocal-pole number gate (the synthetic
residual/flux is 1.39×10⁻⁸, versus 4.74×10⁻² for symmetric). Earlier collision
measurements below describe the previous arithmetic-average operator and do not
certify the replacement.

The opt-in `collision_basis = "instantaneous"` now uses the full current
density matrix and converges the material Hamiltonian and coherent M_H
correction together. `collision_snapshot_steps`, `collision_snapshot_tol_ev`
and `collision_snapshot_mixing` bound the solve; an unconverged snapshot
refuses. Accepted eigenpairs and collision tables come from the same final
sweep. Joint screening replaces the screened-exchange increment by bare HF.
`collision_hamiltonian = "hf"` reconstructs its absolute baseline from the
source's authenticated kinetic/ionic, Hartree and Fock matrices, then includes
M_H once. `reference` instead anchors the bare-HF/dynamical increments to the
existing reference energies; this is a distinct effective reference model,
not a claim that DFT energies solve the absolute QSGW-like closure.

The equilibrium collision counterterm defines `subtracted`. The legacy frozen
screened-exchange kernel additionally needs its static-W subtraction. Joint
modes use a bare-HF kernel and no static reduced-W subtraction, avoiding a
full-kernel/reduced-screening mismatch.
`collision_screening = "refreshed"` and `"shared_pole"` now refuse at
`td_collision_keldysh_screening`. Their retained machinery is a
retarded/population snapshot; it does not implement the joint response/noise
realization or general nonequilibrium W< required by the new operator.

`collision_screening = "joint"` adds independent full-rho lesser/greater
bubbles and their general rational Keldysh screening. `joint_frozen` holds the
same reduced equilibrium interaction for matched comparisons. Joint modes
require the instantaneous basis. Their Coulomb rank, rational support/order,
held-out response/noise tolerance, adiabatic regulator and fitting height are
explicit deck settings. The passive equilibrium GW constructor is unchanged.
Density bounds and both positivity boundary matrices are reported without
clipping; this nonsecular closure has no automatic Pauli-bound guarantee.

## 2026-10-19 — TD-aGW collision: instantaneous basis, vertex pairing, G = L diag(w) R^H (opt-in)

Two `[lorrax]` keys of the full-ρ internal line (`collision_in_scattering = "full"`);
with both unset results are unchanged and the run fingerprint is unchanged.

- **`collision_basis = "instantaneous"`** (default `"band"`; frozen screening): the
  lines' basis and energies are the eigenpairs of the propagator's slow
  Hamiltonian diag(e) + ΔΣ[ρ_eq + U diag(f − f_eq) U^†] (e its window energies;
  a fixed point per trial and refresh, its held ΔΣ^h included; [TD-aGW dynamical
  §1.5, Eq. 8b](docs/theory/td_agw_dynamical.md#damping)) instead of the band
  basis at the bundle's ε^G. Frozen bands sit at ε^G + Δ_edge(k), Δ_edge = e − ε^G
  of the window's edge band on their side (0 when the propagator runs on ε^G), and
  their part is swept at every refresh.
- **`collision_internal_pairing = "vertex"`** (default `"symmetric"`): Σ_ij(ε_i)
  and Σ_ij(ε_j) take the internal energy of the index on their own vertex (two
  sweeps); `symmetric` keeps ½ {g(ε), ρ} in both (one sweep).
- **Code.** `gw.wavefunction_bundle.ParentGreenCarrier` takes optional
  `green_left` / `green_right` panels (G = L diag(w) R^H; None is the faces,
  bitwise the previous operands). `gw.ppm_sigma.window_sigma_c` takes `pairing`
  and `window["e_frozen"]`; the full-ρ line runs on these panels instead of the
  2 n_A pseudo-bands (same tables to 9×10⁻¹⁶). With `rho` its
  `static_limit["sx"]` holds the head's static part (`head` zero).
  `tdagw.dynsig.window_collision` takes `basis`, `pairing` and `model`.
  Tests: `tests/test_tdagw_collision_instbasis.py`.

## 2026-10-19 — TD-aGW collision: full-ρ in-scattering, active rate intermediates (opt-in)

Two `[lorrax]` keys of the split refresh (`collision_refresh_sweep = "active"`);
with both unset results are unchanged and the run fingerprint is unchanged.

- **`collision_in_scattering = "full"`** (default `"populations"`): the active
  part's internal line is the window's whole ρ (hole ρ_mn, particle (1 − ρ)_mn,
  symmetric ½ g(ε_m) + ½ g(ε_n); [TD-aGW dynamical §1.5, Eq. 8a](docs/theory/td_agw_dynamical.md#damping)),
  swept on 2 n_A pseudo-bands after the window. `gw.ppm_sigma.window_sigma_c`
  takes an optional `window["rho"]` (instantaneous basis, `internal = "active"`
  only); `_population_branches`, `compute_sigma_c_ppm_omega_grid` and the
  shared-pole window run take explicit particle weights.
- **`collision_rate_intermediates = "active"`** (default `"all"`): the rates
  G~> and G~< use the window's own intermediate states; the frozen part keeps
  only its Hermitian contribution.
- With both on, Σ_k Tr C[ρ] = 0 for any Hermitian ρ up to the τ-quadrature
  tolerance when W is reciprocal (`tests/test_tdagw_collision_fullrho.py`).

## 2026-10-18 — TD-aGW collision term on main: split refresh, refreshed GN-PPM screening

The on-shell collision term of TD-aGW (`[lorrax] collision`, [TD-aGW dynamical
§1.5](docs/theory/td_agw_dynamical.md#damping)) is merged onto this main
(6f7b2ca26) and gains two refresh options; the GW driver gains one opt-in
observer field. Nothing in a deck has to change.

- **Default path unchanged.** With the new keys unset, results are those of the
  collision branch. The GW driver on the CrI3 516 GN-PPM deck is bitwise
  main's (15/15 `sigma_mnk.h5` datasets and attributes, `sigma_diag.dat`); the
  516 collision drives agree with the branch's to ≤ 2.4×10⁻¹² in P(t) (main's
  driver moves Σ_c by ≤ 2.1×10⁻¹¹ eV against the branch's 07 files).
- **`collision_refresh_sweep = "active"`** (default `"full"`). A refresh sweeps
  the window's own intermediate bands and adds the frozen bands' part, swept
  once at t0, each part on its own held quadrature session: 0.95 s against
  3.3 s per refresh on 516 2v2c, equal to the full sweep to the rule tolerance
  (rates 2×10⁻³ of their maximum). Unlike the full sweep, whose one-shot plan
  re-plans when a state passes the occupation-weight floor (the held ΔΣ^h then
  carries 0.38 meV of rule change), it keeps its rules.
- **`collision_screening = "refreshed"`** (default `"frozen"`), with
  `collision_frozen_bands` and `collision_screening_gamma_ev` (1 meV). Each
  refresh re-screens W at 0 and iω_p from the window populations and refits the
  run's poles: 5.0 s (2v2c) and 11.5 s (10v14c) per refresh on 516. A measured
  prototype: the subtracted form then needs its C[ρ_eq] on the refreshed W
  (`gnppm_collision(rebase=True)`, no deck key yet), and the invalid-mode flips
  make ΔΣ^h jump between refreshes at strong fields (theory page §1.5).
- **GW seam.** `gw.ppm_pipeline.observe_window_sigma_source(screening=True)`
  also leaves the run's `fit_plan` and copies of its two W samples; with no
  observer, or without the opt-in, the driver does nothing new.
  `gw.ppm_sigma.window_sigma_c` takes `static_limit_wc0=`.

## 2026-10-19 — Contour-deformation Σ_c (`sigma_frequency_integration = contour_deformation`)

A second Σ frequency integration evaluates Σ_c by contour deformation: an imaginary-axis
Gauss–Legendre sum plus the real-axis residues of G, read from Hermite knots at Im z = η, on the ISDF
W(z) built from the shared-pole bank's response-quadrature χ⁰ and the bank's Dyson pair equations
(`gw.cd`; [dev note](docs/dev/cd_frequency_integration.md)).  It is the only contour-deformation route
in LORRAX.

- **Existing decks.** The default `sigma_frequency_integration = pole_model` leaves every deck on its
  pole-model path; the `cd_*` keys refuse by name outside the CD route (`GATE cd_applicability`).
- **Deck keys (CD only).** `cd_imaginary_nodes` (96), `cd_knot_step_ev` (0.0625, at most η/4),
  `cd_residue_reach_ev` (`auto`), `cd_knot_chunk` (0 = planner), `cd_band_tile` (0 = planner),
  `cd_chi0_rule_tolerance` (1e-8), `cd_chi0_rule_capacity` (1088), `cd_chi0_rules_file` (empty = build
  in the job), `cd_own_energy_band_min` (1), `cd_checkpoint` (`off` | `write` | `resume`).
  [Input reference](docs/input_reference.md#sigma).
- **Limits (named refusals).** `compute_mode = mpa`, `screening_diagrams = w_rpa`,
  `head_correction = off`, `linalg = local`, `use_band_extrapolation = false`, one η; the pole-model
  keys (`sigma_w_model`, `sigma_w_accuracy`, `sigma_w_support_sites_ev`, `write_w`, `write_poles`,
  `mpa_*`) refuse beside it.
- **Instruments.** `LORRAX_CD_STAR_CERTIFICATE=N` (symmetry certificate of the unfolded cube) and
  `LORRAX_CD_STOP_AFTER=imag:N,real:M,real_from:K` (timing probe; `cd_stop_receipt.json`).
  [Environment registry](docs/reference/env_vars.md).
- **χ⁰ rule CLI.** `python -m gw.cd.chi0_rules ... --out FILE` precomputes the rule file on a CPU
  node (the run's log prints the geometry arguments); `cd_chi0_rules_file` reads it and refuses any
  identity mismatch (`GATE cd_chi0_rules_identity`).
- **Outputs.** `cd_sigma_own_energy.npz`, `cd_sigma_receipt.json`, `cd_leak_ledger.csv`,
  `cd_chi0_rules.pkl` (rules built in the job), `tmp/cd_checkpoint.h5` (`cd_checkpoint`).
- **Service API.** `minimax.RESPONSE_MAX_KAPPA` is public (value unchanged);
  `response_bank.response_quadrature` takes optional `requests=` and `rule_options=` (the bank passes
  neither, so its rules are unchanged).
- **Retired, not supported.** The CD tools that did not land: the ordered-Lehmann CD of
  `wip/cd-production`, the fe_cd/CDMETAL harness, cd_modern (spectral-density χ⁰), the cdref plane-wave
  reference factory, the cd_expansion and CD replay scripts, and the `internal_ff_cd` branches.  Their
  decks and caches do not carry over; write a CD deck with the keys above.

## 2026-10-18 — On-shell Sigma sweep checkpointed (one-shot restart)

A one-shot shared-pole run on the default on-shell path now writes its swept
Σ_c, the pair and diagonal slots, to `tmp/sigma_checkpoint_oneshot_on_shell.h5`
(`file_io.sigma_checkpoint`, the grid path's mechanism and payoff rule). A
rerun with `restart = true` whose checkpoint authenticates skips the τ sweep
and goes straight to finalize. Before this, the on-shell sweep was recomputed
on every restart. The identity is the grid path's plus the evaluation and the
on-shell request tables (E_eval, E_DFT, coverage). A run whose energies moved
recomputes, and the old file is removed with a named WARNING. The grid and
on-shell sweeps never read each other's file. SC runs are unchanged: they
continue at map granularity from `sc_seed/sc_checkpoint.h5`.

- **Results unchanged.** Measured on Fe 8^3 (MPA shared pole, one-shot,
  4 A100 ranks) against `6f7b2ca26`: the eqp0/eqp1/eqp_g0w0/sigma_diag
  bodies, every `sigma_mnk.h5` and `qp_wfn_rotations.h5` dataset and
  attribute, and the per-rank device peak are bitwise equal. The restart
  that reads the checkpoint reproduces them bitwise.
- **Cost.** 27.3 MiB, written in 0.3 s after a 105 s sweep. The restart
  reads it in 0.5 s, and its wall falls from 144.3 s to 49.8 s.
- **Grid debug path** (`sigma_omega_grid_debug = true`): outputs and its
  checkpoint file are byte-identical to `6f7b2ca26`, and its restart still
  reads it.
- **Files.** A new `tmp/` file per one-shot run. It is written only when
  `write_restart_tensors` is on and the sweep took at least 5× its predicted
  write.

## 2026-10-18 — Sigma(omega) grid debug restored

The ω-grid Σ_c(ω) path is back as an opt-in debug path,
`sigma_omega_grid_debug = true` (default false; [On-shell Σ_c §grid
debug](docs/theory/sigma-on-shell.md#grid-debug)). With the key set, every
output is the one `c2975e450` wrote with the same key. That covers the
`(n_ω, nk, nb, nb)` cube and `omega_ev` in `sigma_mnk.h5`, eqp0/eqp1 from the
interpolated Σ_c and the stencil Z, the C(ω) eqp receipt (and
`python -m gw.eqp_bgw --finite-difference-spacing`), `sigma_freq_debug.dat`,
the SC `sigC`/`sigXC` columns, and the one-shot sweep checkpoint
(`tmp/sigma_checkpoint_oneshot.h5`, written and read on restart).
`qp_omega0_ev` (`write_qsgw_datasets`) is back as `c2975e450` had it: written
for an unsharded cube only. Under `sigma_omega_layout = sharded` it is
omitted by name, as before; that held on every validation leg (4 ranks and
1), so it was not compared bitwise.

- **Default path unchanged.** With the key unset, results are bitwise those
  of `663c27863` (the on-shell entry below). Measured on Fe 8^3 (MPA shared
  pole, 4 A100 ranks, SC map and one-shot): the eqp0/eqp1/sigma_diag bodies,
  every `sigma_mnk.h5` dataset and attribute, the Response-quadrature lines
  and the per-rank device peak.
- **Grid outputs as before.** With the key, on the same deck, the same files
  plus the cube, `omega_ev`, `sigma_freq_debug.dat`, the C(ω) receipt and the
  sweep checkpoint (written, and read on restart) are bitwise those of
  `c2975e450`.
- **Cost under the key.** On that deck (n_ω = 271) `sigma_mnk.h5` grows
  from 7.9 MiB to 714 MiB and the τ sweep from 105.3 s to 122.3 s; the
  per-rank device peak does not move (the W response sets it). A one-shot
  run also writes a 2.99 GiB sweep checkpoint.
- **eqp2 stays retired.** `write_eqp2`, `eqp2_tol_ev`, `eqp2_max_iter`,
  `eqp2_accelerator`, `eqp2_history_depth` and `eqp2_file` still refuse by
  name at parse time, now as `GATE eqp2_retired`; `sigma_omega_grid_debug`
  no longer refuses. `eqp2.dat` is not written.
- **Not restored:** the `omega_ev`/`sigc_omega_ev_{re,im}` curve of
  `python -m gw.plane_wave_pipeline --deck` (it stays on shell).
- **API:** `DebugConfig.sigma_omega_grid_debug` is back;
  `DeviceOmegaAccumulator.grid(...)` builds the grid accumulator.
  `LorraxConfig.eqp2`, `EQP2Config` and `FilePaths.eqp2_file` stay removed.
- **What you must change:** nothing for default decks. A deck that dropped
  `sigma_omega_grid_debug = true` because of the retirement may set it again.

## 2026-10-17 — ROCm: ELPA is the distributed eigensolver

On Frontier (ROCm), `eigh` with `linalg = "distributed"` above the one-device
bound (`ONE_DEVICE_MAX_N`, unchanged) now runs ELPA 2026.02.001's one-stage GPU
solver (`backend elpa`), in place of SLATE's `heev`. SLATE stays selectable
(`backend = "slate"`) and still serves `potrf` and `solve_lu`. Measured on 2x2
MI250X GCDs (`ELPA_ROCM_REPORT.md`): complex128 n = 20480 in 26 s through
`distributed` (SLATE: 122 s), n = 51200 in 211 s (SLATE aborts at 30720).
Residual and orthogonality ~1e-14.

- **Which results move.** Only runs that diagonalise a matrix above the
  one-device bound on ROCm: eigenvalues and eigenvectors change at rounding
  level (a different algorithm). Runs at or below the bound (Fe P=4 among
  them) are bitwise unchanged.
- **Memory.** During the solve, ELPA's device buffers and the workspaces of
  all of its rocBLAS/rocSOLVER handles come from one XLA workspace (inside the
  arena, sized from measured high-waters). ELPA's input is the operand's
  transposed temporary, aliased, so there is no copy. Between calls the
  handles hold no device memory. Measured at n = 51200 on 2x2, the process's
  device memory outside the arena peaked at 2.30 GB (allowance
  `DISTRIBUTED_LINALG_OUTSIDE_BYTES` = 3 GB). If a rocBLAS workspace region is
  too small, the solve **refuses** with
  `rocBLAS/rocSOLVER call(s) failed during the solve`; it never returns a
  result.
- **Two-stage ELPA is not available.** It was slower everywhere measured and
  returns wrong eigenvectors on 4x4 grids at n >= 47104. There is no option
  for it, and an `ELPA_FORCE_*` environment variable makes the solve refuse
  (ELPA would let it override the pinned settings).

**What a user must change.** Rebuild the ROCm leg with
`config/frontier/build_ffi_rocm.sh`. On first use it builds ELPA with
`config/frontier/build_elpa_hip.sh` (two LORRAX source patches) into a
content-keyed prefix. The leg's closure record gains `libelpa_openmp.so.19`
(private). Unset any `ELPA_FORCE_*` variable.

## 2026-10-17 — Sigma_c on shell only: the ω-grid path and eqp2 are retired

Dynamic Σ_c is evaluated on shell only
([On-shell Σ_c](docs/theory/sigma-on-shell.md)). The ω-grid debug path
(`sigma_omega_grid_debug = true`) and everything only it fed are removed.
That includes the grid fold, the ω-cube unfold and writer, the interpolating
readers, the finite-difference Z, the streamed ω combination, the C(ω) eqp
receipt, the sweep checkpoint and fixed-Σ evSC (`eqp2.dat`).

- **Results do not move** on the remaining paths. Measured on Fe 8^3 P4
  (one SC map, A100), against the previous commit: the eqp0/eqp1 bodies are
  bitwise identical. The sigma_diag body is identical on every column the
  two files share; the SC `sigC`/`sigXC` columns are omitted (below). The
  P4 hsuite passes against its references.
- **Deck keys removed.** Each one refuses by name at parse time, with any
  value (`GATE omega_grid_retired`): `sigma_omega_grid_debug`, `write_eqp2`,
  `eqp2_tol_ev`, `eqp2_max_iter`, `eqp2_accelerator`, `eqp2_history_depth`,
  `eqp2_file`. Delete the lines; nothing replaces them.
- **Outputs removed:**
  - `eqp2.dat` and its report section.
  - The `(n_ω, nk, nb, nb)` Σ_c(ω) cube and `omega_ev` in `sigma_mnk.h5`.
  - `qp_omega0_ev` (`write_qsgw_datasets`): named as omitted, permanently.
  - The C(ω) curve in the eqp receipt (`eqp_assembly_conditioned_sigma_c_omega_diag_ev`).
  - The `omega_ev` and `sigc_omega_ev_{re,im}` keys of
    `python -m gw.plane_wave_pipeline --deck`. Its `sigc_at_e_ev` is now the
    exact on-shell Σ_c,nn(E_DFT): the `dft0` table, Σ(0) outside the covered
    range, where the old value was a clamped interpolation.
- **`sigma_diag.dat` under SC** no longer carries a column of zeros for the
  DFT-basis Σ_c(E_DFT), which on shell does not exist. The `sigC` and
  `sigXC` columns are omitted and named in a `# omitted:` header line.
- **`sigma_freq_debug.dat`** (one-shot) reads the on-shell operand: Σ_c at
  E_DFT and E_eval, dΣ/dω, Z, and the q→0 head at E_DFT. Under SC it is
  omitted by name.
- **`eqp_bgw.make_eqp_bgw`** reassembles from the on-shell receipt only. An
  old ω-grid `sigma_mnk.h5` refuses (`GATE omega_grid_receipt_retired`). The
  CLI option `--finite-difference-spacing` is gone.
- **Z guard.** eqp1 and the SC tail weights share one Z
  (`eqp_bgw.on_shell_z_factor`). It prints one line counting states with Z
  outside (0, 1] or non-finite, and a non-finite Z refuses
  (`GATE z_factor_finite`).
- **SC grid-edge census** (`sigma_out_of_grid = static`) runs on shell again,
  from two diagonal-only tables, Σ(0) and Σ at the nearer range edge, that
  the request adds in that mode only.
- **Restart:** a `sigma_checkpoint_oneshot.h5` from the grid path is no
  longer read. The on-shell sweep is recomputed on restart, as before.
- **API:** `mixing.acceleration.rcrop_nojit`, `band_partition.
  apply_band_partition`/`build_omega_band_partition` and `scissor.
  qsgw_out_of_range_energies`/`classify_bands_in_grid` are removed (their one
  caller was eqp2). `LorraxConfig.eqp2`, `EQP2Config`,
  `DebugConfig.sigma_omega_grid_debug` and `FilePaths.eqp2_file` are gone.
- **What you must change:** delete the removed keys from decks, and stop
  reading `eqp2.dat`, `qp_omega0_ev` or an ω axis from `sigma_mnk.h5`.

## 2026-10-17 — k-convolution kernels: four sub-1.15× arms removed (CUDA)

A simplification pass on the mathdx modes 7, 9 and 11 (`src/ffi/cpp/cufft/kconv_mathdx_cuda_ffi.cc`,
`kbox_stage.cuh`). Removed: mode 7's per-call k-fastest index-table kernel (M7KF; the factory now
forms the k-fastest tables in XLA and passes them to a new target), the m7s split arm at
n_s = 1, mode 9's four-block W2 rule below N_k = 256 and its fixed-column exception (W2 there now
runs two blocks a SM), and mode 9's
persistent split plane pass (m9b); the dead tuning bits of the kernel text; and the duplicated
pair-order, table-load and axis-pass helpers (now in `kbox_stage.cuh`). The arms that remain are
listed per mode in [k-convolution §4](docs/architecture/kconv.md#arms).

- **Results do not move:** every remaining arm computes the same lines in the same order (bitwise
  on the 45-case modes 7/9 sweep, the 21 mode-11 comparisons and the Fe 8³ P4 deck bodies).
- **Speed:** per call (A100, 45 k-grid cases x modes 7 and 9 against the previous kernels): mode 7
  up to 1.12× at n_s = 1 on 14³/15³ (the m7s rule), mode 9 1.20–1.22× at 6³ (0.30 -> 0.36 ms) and
  1.11–1.23× at ≥ 14³; every other case within 1.02× (M7KF's gain is kept: Fe 8³ mode 7 4.06 ms).
  Mode 11 is unchanged.
- **Memory:** `ffi.fft.klead_unfold_scratch_bytes` now prices the m7s arm's pair-major W_R (16·N_k B
  a pair at n_s = 2 on 14³–21³ A100 grids), which the Σ τ window checks already add; before, that
  arm took up to 25 % more scratch than priced. M7KF's unpriced handler workspace is gone; the
  k-fastest tables are XLA buffers of `4·N_k·(μ_x + ν_y)·n_s` B a call (2.95 MB on a Fe 8³ P4
  rank), counted by `memory_analysis`.
- **New target:** mode 7 calls `lorrax_mathdx_kconv_klead_unfold_kf` (`_xblock` plus the k-fastest
  tables; a new handler, so no ABI bump). A CUDA leg without it refuses at startup
  (`GATE kconv-target`); `_xblock` stays for older trees and runs the chain arm there.
- **Startup:** `require_kconv` also probes `lorrax_mathdx_kconv_chi_unfold_q`, so a CUDA leg older than
  that target refuses at startup (`GATE kconv-target`), not at the first W stream.
- **`make_kconv_chi_unfold(rows=...)`:** a static `rows` outside `[0, N_k)` raises `ValueError`.
- **Build log:** the `[kconv_mathdx]` line names `variant=` and `arm=`.
- **What you must change:** nothing in decks or environment, and no handler signature moved (ABI
  unchanged). The kernel text changed, so the cubin store recompiles each (mode, grid) once
  (about 8 s each); the module is republished with a bundle from this commit.

## 2026-10-17 — Dyson response programs: one compile per LU decision on the CUDA room leg

Under `linalg = distributed` on CUDA, the W response's Dyson programs keyed on a figure
(`one_device`'s bound) that moves with almost any change of the ledger room, so a moved
room recompiled them. They now key on what the program runs, `(backend, route, rounds)`
(`distrib_la` `Plan.decision`), and the measured size route's agreed costs are memoised
without the room, so a new room exchanges, reads and compiles nothing to decide.

- **Results do not move:** the same routes and rounds are chosen as before.
- **New refusal:** a measured size route whose ranks hold different rooms now refuses by
  name ("has a room that differs across processes") instead of routing divergently.
- **What you must change:** nothing.

## 2026-10-17 — W response: warm rule nodes cost 0 in the planner; the q-span row window gathers through its cached program

Two schedule-only changes in `src/gw/response_bank.py`. No rule, energy or output byte moves.

- **Warm rule-tree nodes cost 0 in the round planner** (`_attempt_cost`). A node that holds an
  earlier map's rule on the same member set is tried first on that rule's nodes: one projection
  pass and no pencil when it fits. It was priced as a cold attempt (2.4–3.4 s on the device
  route), so on Fe 8³ SC maps 1–3 the planner deferred warm needed nodes to a round of their own
  and filled it with cold speculation. They now ride in the round of the failing shared fit.
  Fe 8³ P4, `sc_max_iter 3`: rule construction −7.6 s and W response −6.7 s summed over the four
  maps; quadrature lines, `eqp0_iter*.dat` and per-rank peak memory identical.
- **`_RowWindow.__getitem__` takes rows through the cached `_MemberRows` gather program**
  instead of eager indexing (1.64 → 0.12 ms dispatch per take; bitwise, same sharding). Fe 8³ P4,
  `sc_max_iter 1`, two A/B pairs: W response 158.8 / 159.0 s → 157.2 / 157.0 s; `eqp0.dat`,
  `eqp1.dat` and `sigma_diag.dat` identical.
- **What you must change:** nothing.

## 2026-10-17 — k-convolution kernels: faster mathdx modes 7, 9 and 11 (CUDA); a new native target

The NVRTC-built k-convolution kernels (`src/ffi/cpp/cufft/kconv_mathdx_cuda_ffi.cc`) gain
faster arms; every output is bitwise identical to the previous kernels.

- **Mode 7** (W_R pass): a fused single-arm tile for n_s <= 2 (cp.async gather,
  orbit-blocked tiles, k-fastest index tables) and a split arm (persistent plane pass,
  pair-major W_R, overlapped column pass) with an n_s = 1 rule. **Mode 9**: batched W2
  gather and a fixed-column store. **Mode 11** (chi0 from the raw-parent Greens, n_s = 2):
  a fused single-arm tile; `make_kconv_chi_unfold(acc=None)` writes without an
  accumulator, and `acc=None, rows=q` stores only the q rows of the flat-k transform
  through the new target `lorrax_mathdx_kconv_chi_unfold_q`, which `gw.w_isdf`'s direct
  stream uses wherever `fn.q_store` holds.
- **Results do not move:** bitwise on 45 k-grid cases x modes 7 and 9 and 21 mode-11
  comparisons (A100), and Fe 8^3 P4 eqp0/eqp1/sigma_diag bodies. Fe 8^3 P4 one map, warm
  (A100, BFC default pool): W response 198.6 -> 159.6 s, Sigma tau sweep 141.4 -> 107.0 s,
  map wall 346.9 -> 273.4 s.
- **What you must change:** nothing in decks. The Python needs a `liblorrax_ffi.so` built
  from this source: with an older CUDA leg, an n_s = 2 mode-11 deck fails loudly
  (`lorrax_mathdx_kconv_chi_unfold_q` unregistered; the old handler refuses `acc=None`).
  Rebuild a clone's pair (`config/perlmutter/build_ffi_cuda.sh`); the Perlmutter module is
  republished with a bundle from this commit. The kernel source changed, so the cubin
  store recompiles each (mode, grid) once (about 8 s each).

## 2026-10-16 — undensified BSE: the `--dipole` file must be stamped for the deck WFN

`absorption_common.window_dipoles` now checks the file's `prov_wfn_sha256`
against the deck WFN's fingerprint on undensified runs too (AUDIT B12), with
the gate the `bse_k_grid` path already had (`GATE velocity_wfn_fingerprint`):
another WFN's `dipole.h5` on the same k list (another NSCF, cutoff or
structure) used to pass the nk check silently. A file with no stamp refuses;
regenerate it with `python -m psp.get_dipole_mtxels -i <deck>`.

## 2026-10-16 — `bse_k_grid` + `--eqp`: the dipoles are the exact position of the fine QP states

**Changes numbers.** With `--eqp` the densified fine states diagonalize the
interpolated QP f(H), which mixes the velocity's DFT bands off the coarse
grid, and `d = v/ΔE` of their Rayleigh-quotient energies was not their
position: CrI3 8×8→12×12 against the native 12×12 lost 1.4 % of Σ|d|² and
13 % below 1.6 eV (Σ|v|² kept to 4e-4; the lowest transition's quotient
60 meV high), 8×8→30×30 1.5 % and 19 %. `densified_dipoles(qp_states=True)`
(the loader passes `eqp_file is not None`) forms r in the DFT eigenbasis of
the fine states' own f(H) block and rotates it back; without `--eqp` the
dipoles are bitwise unchanged. Oscillator strengths, ε₂ and helicity tables
of `bse_k_grid` runs with `--eqp` move (the 30×30 CrI3 bright states are
the ones that showed 14–23 % below native 8×8); rerun them, or re-contract a
saved `eigenvectors.h5` with `python -m bse.dump_window_dipoles` (same
deterministic gauge; `--check-eigenvectors --check-key d_alpha_rq` proves it
on the old run first). `full_window` adds `r_window` (pair rule `mask` by
default, `velocity_pair_rule` / `velocity_pair_gap_ev` on the loader) next to
`v_window`: TD-aGW should take r from it rather than `v_window/ΔE`. The route
label of fixed runs ends with "exact r in the fine QP states (DFT eigenbasis
of their f(H) block, v1)".

## 2026-10-16 — `bse_jax --helicity-table`: per-state σ± strengths in the report

With `--lanczos --tda --write-eigs --dipole`, `--helicity-table` adds each
written state's in-plane circular strengths to `bse.out`:
I± = |D_x ∓ i D_y|²/(2 N_k) in bohr² per unit cell for D = ⟨0|r|S⟩, and
η = (I+ − I−)/(I+ + I−). σ+ is the field E₀(cos ωt, sin ωt, 0), which rotates
x → y, counter-clockwise seen from +z
([helicity](docs/architecture/bse.md#dipoles-and-absorption)).

- **Results.** Report-only and off by default: `eigenvectors.h5` and every
  other output are unchanged. CrI3 8×8→30×30 (c1, provisional W, 42v × 14c):
  A at 1.5777 eV is σ+ (η = +1.0000), B at 2.0223 eV σ− (−0.9997), and the
  strongest σ+ state is at 2.3121 eV (I+ = 1.06 bohr² per cell, η = +0.9988).

## 2026-10-16 — `bse_k_grid` dipoles: the deck WFN's velocity through the densifying htransform

`bse_jax --write-eigs --dipole` and `absorption_haydock` sliced a `dipole.h5`
onto a `bse_k_grid` bundle. The deck WFN's file has the coarse k list and
failed on its shape (Haydock at trace, `bse_jax` after its eigensolve); a fine
WFN's `dipole.h5` has the bundle's nk in that loader's own band gauge and
contracted silently wrong (MoS₂ 9×9→12×12 SR/LR: ε₂ 18 % rms of the direct
12×12 peak, bright A partners 29.7 and 25.4 bohr² against 45.6). The densifier
now carries the deck WFN's velocity through the same htransform and contracts
it with the same fine eigenvectors that build ψ
(`bse.bse_densify.densified_dipoles`); the transition energies are the
velocity's own DFT energies there.

- **Commands.** Under `bse_k_grid`, `--dipole` names the deck (coarse) WFN's
  `dipole.h5`, with at least the htransform window's bands. No fine NSCF is
  needed for dipoles. Any other WFN's file refuses
  (`GATE velocity_wfn_fingerprint`); on an undensified run a file from
  another k grid refuses (`GATE dipole_k_grid`).
- **Results.** MoS₂ 9×9→12×12 against the direct 12×12 run: ε₂ rms 2.01 %
  (SR/LR) and 3.92 % (c1) of its peak; bright A pair |d|² 47.4 and 47.3,
  B pair 53.0 and 52.7 bohr², against 45.6 and 53.1 (partners within 0.6 %;
  the densified pairs are split by up to 0.6 meV). Without `--eqp` the window
  dipoles match the fine NSCF's own to 6e-6 at shared k and 0.8 % rms
  (Frobenius norm per k) off them. Densified eigenvalues are unchanged, and
  runs without `bse_k_grid` are bit-identical.
- **Outputs.** A densified run's `exciton_data/dipoles` (`eigenvectors.h5`)
  and Haydock h5 metadata name the dipole route, its source file and the
  on-grid identity.


## 2026-10-16 — `file_io.epsreader`: ε⁻¹(G, G') in row-G order; 3-D `eps0mat` export refuses

`EPSReader.get_eps_matrix` returns ε⁻¹_q(G, G') with rows G and columns G'
— the transpose of the stored block, whose `[a, b]` is ε⁻¹(G_b, G_a)
(BerkeleyGW's Fortran column order; `file_io.bgw_epsmat`). It returned the
stored block before. `get_eps_diagonal` reads `mats/matrix-diagonal` as
`(nq, nmtx, 2)`. `tests/test_bgw_epsmat_cpu.py` pins both on a corner of a
BerkeleyGW-written file. `gw.epsmat_cli` export now refuses a `sys_dim = 3`
`eps0mat` (`GATE epsmat_q0_body_3d`) unless the deck sets
`q0_body_3d = omit_head_wing`: the 3-D q → 0 body lacks the O(1) head–wing
term. Its μ-space algebra runs on `P(None,'x','y')` faces with the
`distrib_la` face eigensolver (cuSOLVERMp when each device has its own
process).

- **Results.** Main's one reader of the file, `gw/head_correction.py`, reads
  `epshead` and does not move. Code that read `get_eps_matrix` as the stored
  block must drop its transpose. Exported blocks move at the conditioning of
  V, ≤ 3.1e-9 relative on CrI3 6×6 (κ(V) 5.7–7.6e6); projections at 1.5e-14.
- **Decks.** A bulk export of `eps0mat` adds `q0_body_3d = omit_head_wing`
  or exports `files = epsmat`.

## 2026-10-16 — Native dependencies: recorded at build, enforced at start-up, on every route

Every FFI leg's build records the exact library set it resolved, and every
start-up enforces it, on every route (a clone, the Perlmutter module, a sealed
bundle) and for every leg (CUDA, ROCm, host)
([the dependency closure](docs/installation/ffi-build.md#closure)). This closes
the defect where a plain clone launch on Perlmutter mapped the HPC SDK's
cuSOLVERMp 0.8 in place of the staged 0.9.1 (the first distributed LU/eigh
failed with `status=7`), and the site-default LibSci/HDF5 mapped cray-mpich
9.1.0 beside 9.0.1.

- **Recorded.** `<leg>.closure.json` beside each leg (schema 2): the leg's
  SHA-256 and every library of its `DT_NEEDED` closure (path, size, mtime,
  SHA-256, SONAME, family). That covers the cuSOLVERMp/cuBLASMp stage, the
  GPU math and collective libraries the venv's JAX plugin maps (cuBLAS,
  cuSOLVER, cuFFT, NCCL; rocBLAS, rocSOLVER, HIP; RCCL attested once JAX maps
  it), MPI, HDF5, LibSci, SLATE/BLAS++/LAPACK++ and the HIP SLATE's OpenBLAS.
  `PROVENANCE` gains `closure_sha256`. Sealed bundles (manifest schema 2)
  carry both records.
- **Enforced.** The loader opens the recorded files by exact path before the
  leg. After the leg's `dlopen` it requires every recorded family to be
  mapped exactly once, as the recorded file (or byte-identical). Otherwise:
  `GATE native-closure: got <wrong>; want <expected>; fix: <remedy>`. A missing
  or stale record refuses too. There is no fallback and no warning mode.
  `LD_LIBRARY_PATH` no longer decides a leg's libraries, so a clone needs no
  hand-built library path. Each leg logs one `closure … enforced` line and one
  `closure … libraries:` line naming every family's file and hash: the
  identity lines for HIP, rocSOLVER, rocBLAS, RCCL, HDF5, MPI, cuSOLVERMp,
  NCCL and the rest.
- **Caches.** `lxkit.native_provider.native_closure_identity(platform)` keys
  every cache that depends on native libraries:
  - the JAX compile-cache namespace, schema `k2`, so every `k1` namespace is
    retired by the age pruner. An explicit `ISDF_JAX_CACHE_DIR` is split by
    the same `ffi-<12 hex>` tag;
  - the kconv image store and the rocFFT kernel store.

  The first run after upgrading recompiles once.

- **NCCL: one pin, every route.** The `nvidia-nccl-cu13` 2.30.7 wheel (the
  `uv.lock` pin). The CUDA leg now compiles and links against the wheel, not
  the HPC SDK's `comm_libs` 2.29.3. The module venv installs the same wheel,
  its modulefile drops `comm_libs` NCCL, and its sealed bundle records the
  wheel. A run that maps any other `libnccl` refuses. The NCCL net plugin is
  unchanged (`config/perlmutter/site.toml`, AWS OFI).

**What a user must change.**
- **Rebuild every FFI leg** with this tree's build scripts, and build a new
  module prefix (the module venv gains the NCCL wheel; `setup_env.sh` never
  refreshes a venv in place). Perlmutter:
  `config/perlmutter/build_ffi_host.sh`, `build_ffi_cuda.sh`. Frontier:
  `config/frontier/build_ffi_host.sh`, `build_ffi_rocm.sh`. A leg without a
  schema-2 record refuses at start-up (`no closure record`). Reseal bundles
  with this tree's `seal_bundle.py`; a schema-1 bundle refuses.
- **Rebuild after changing a recorded library**: a `uv sync` that changes the
  CUDA or ROCm wheels, a SLATE or OpenBLAS rebuild, a moved or purged `.build`,
  or a site module upgrade in place. The next run refuses with
  `stale closure record` until the legs are rebuilt.
- **Drop hand-built library paths.** Launch scripts that prepend the stage or
  the pinned directories to `LD_LIBRARY_PATH` are no longer needed; they
  change nothing a leg maps.
- **No second copy may be mapped.** An `LD_PRELOAD`, or a Python package that
  loads another MPI, HDF5, cuBLAS, NCCL, rocBLAS, … beside the recorded one
  (for example an mpi4py built against the site-default MPI), now refuses at
  start-up instead of running.
- **A host-leg SLATE op refuses in a device process.** There the device leg's
  SLATE (Frontier: SLATE-HIP with OpenBLAS) answers the SLATE SONAMEs, so the
  host leg's `lorrax_slate_*` handlers would run against a library they were
  not built against. They are not registered, and calling one raises
  `GATE native-closure: host-leg <target> needs SLATE-CPU but the device leg's
  SLATE serves this process`. Run SLATE ops on the device mesh, or in a
  CPU-only process. Closures now merge in `dlopen` order, shared by every
  loader; before this, a process whose two loaders finished their checks in
  the other order refused at its first batched eigh (Frontier P=16).
- **Source the site launch script before a multi-node `srun`.**
  `config/perlmutter/gpu_env.sh` now links `libcudart.so`, which libfabric
  dlopens by that name, to the venv's CUDA runtime, first on
  `LD_LIBRARY_PATH` (Frontier's `rocm_env.sh` already did this for
  `libhsa-runtime64.so`). A multi-node step whose `libcudart.so` names another
  file refuses at start-up. Before this, a 4-node clone step ran the
  toolkit's cudart 13.2.75 for the plugin and the leg, unrecorded.
- Results: unchanged on the clone route. The enforced libraries are the ones
  the documented routes already ran. Measured against aff42ce37 (DEP_ENFORCE.md):
  Fe V2 P=4 bitwise on both machines; Fe V2 P=16 on 4 Perlmutter nodes bitwise,
  NCCL 2.30.7 over AWS Libfabric with GDRDMA on all 16 ranks and no `NCCL WARN`.
  Per-rank device peaks are unchanged.
- **The Perlmutter module carries a clone's CUDA wheels.** Its venv is
  `uv export --extra cuda13` of the lock (cuBLAS 13.6.1, cuFFT 12.3, the CUDA
  runtime 13.3, cuDNN 9.24, …), no longer `jax[cuda13-local]` on the toolkit's
  libraries (cuBLAS 13.4.1). Build a new module prefix: `setup_env.sh` never
  refreshes a venv. The modulefile lists no toolkit library directory
  ([Perlmutter](docs/installation/perlmutter.md)). Measured: module Fe V2 P=4
  is bitwise identical to the clone run of the same tree.

## 2026-10-16 — rocm/integration8: follow-ups on the release

Integration 8 lands the follow-up lanes below on the release `aff42ce37`. Fe V2, MI250X,
P=4, one node (`/lustre/orion/mat280/scratch/jackmc/int8/<run>/run.r0.log`):
- Total 89.33 s (release, `int7/feR1`) → 84.92 s (`h5fe1`). Rule construction
  32.91 → 27.40 s, constructor 14.27 → 12.85 s, Σ τ other 6.13 → 3.73 s.
- eqp0 / eqp1 are bitwise equal to the release after every group. From syn/w12 on,
  `sigma_mnk.h5` moves by ≤ 2.2e-14 relative (per dataset, max |Δ| / max |ref|) and
  `qp_wfn_rotations.h5` `U_mnk` by ≤ 2.6e-11; before it every file is bitwise.
- Device peak per rank 6.77 GB, as the release, in every stage row. On ROCm a stage row
  reads the allocator's `peak_bytes_in_use`, which never resets, so every row shows the
  run's high-water mark (set in ζ), not the stage's own peak.
- Rank-0 transfer census, release → integration 8 (`int7/cenR` → `int8/h5fe2`): H2D
  `device_put` 737 → 238, D2H 433 → 348, explicit D2H 74 → 70, `jax.block_until_ready`
  18 → 5.
- hsuite P=4: 16/16. Na SC completes (the release refused it, `GATE shared_pole_zero_ritz`
  at q=2) and matches its reference: eqp max |Δ| 4.0e-7 eV.

**What a user must change.**
- **Rebuild the FFI libraries** from this tree (`config/frontier/build_ffi_rocm.sh`; host
  and CUDA likewise). The handler ABI goes 6 → 8: SlabIO's read, read_multi, k-chunk union
  and write handlers take no control operand ([SlabIO](docs/architecture/slab_io.md#stream-race)),
  and `active_gemm_device` takes `conj_b`. A library of another ABI refuses at load.
- New refusals: `GATE phdf5_posted_control`, `GATE slab_io_write_finite`,
  `GATE shared_pole_round_tables_device`, `GATE shared_pole_ritz_eigh_finite` (an
  eigensolver failure left after the Jacobi re-solve; such a round used to refuse as
  `shared_pole_zero_ritz`).

| lane (commit) | what | measured effect (Fe V2 P=4 unless noted) | gate |
|---|---|---|---|
| lsync/w5 polar (7e5898496, 005699355), slabmr/w8 (473f3bde9), comm7 T1 (9265e080a), ctor5/w12 (49ada1eaf, 3a7876794) | polar cuts from one device word; SlabIO posted control (ABI 8), device order fence, k-chunk union reader, staging budget; transpose partner without self pairs; constructor round tables, directions recipe and pole counts on the device | constructor 14.27 → 12.77 s; census as above | Fe bitwise (2 runs); phdf5 read_multi, active, kdft gates at P=4 |
| crash/w9 (b697bce69; int8 bf8041aee: public import) | distinct inert sentinels; Jacobi re-solve of a non-finite local round; `shared_pole_ritz_eigh_finite` | Na SC completes; no re-solve fired in hsuite (the sentinels suffice) | hsuite 16/16; Fe bitwise |
| syn/w12-i8 (dfce55663) | active GEMM reads B conjugated on load; identity little-group projector; per-branch W orientation | Σ τ other 5.84 → 3.70 s | eqp bitwise, Σ ≤ 2.2e-14 relative |
| rcgpu (6b8629482) | the rule tree abandons a geometry on its error forecast's prediction bound | rule construction 32.42 → 27.40 s | Fe bitwise |

## 2026-10-16 — rocm/integration7: the ROCm leg's host syncs, transfers and kernels

Integration 7 lands the lane branches listed below on `rocm/integration6` 4e6b28dfc. The rules
and tools are in [Host–device sync](docs/architecture/host_device_sync.md),
[Determinism](docs/architecture/determinism.md),
[Start-up, environment and caches](docs/architecture/startup_environment.md) and
[ROCm leg](docs/architecture/rocm_leg.md).
- Fe V2, MI250X, P=4, one node: total 112.18 s (int6 control, frontier00605)
  → 91.83 / 90.61 s (frontier00620; there was no int6 control on that node).
  Rule construction 48.82 → 33.70 / 33.02 s, constructor 17.75 → 13.96 /
  14.09 s, Σ τ other 7.89 → 6.72 / 6.70 s.
- Rank-0 transfer census, int6 → int7 tip (`xsync/cen2` → `int7/cen5`): H2D
  `device_put` 7545 → 713, D2H 2089 → 425, explicit D2H 461 → 71,
  `block_until_ready` (array + `jax.`) 907 + 117 → 119 + 6. The `jax.` count
  depends on timing (SlabIO waits only on reads still in flight), so other runs
  give 6–17 (the lsync/w3 run below: 119 + 7 = 126).
- eqp against int6/e5 (int7 tip, off-shell Σ_c): 2.3e-8 / 5.0e-8 eV (eqp0 / eqp1); the Fe gate is
  ≤ 1e-7 eV.

Every wall above is from the integration release notes (rank-0 stage timing;
`/lustre/orion/mat280/scratch/jackmc/int7/<run>/run.r0.log`); per-lane numbers
are from the lane's report.

**With main's on-shell Σ_c (this merge).** main makes on-shell dynamic Σ_c the
default (entry below). Merging it with dput/w2's device Σ window table needed one
semantic resolution (`ppm_accumulators.integrate_window`/`run_window`,
`sigma._sigma_window_table`): the shared-pole sweep keeps the window table for its
operands (nodes, masks, reference energies, intervals), and on shell each window's
coefficient is main's `_on_shell_pack`, placed once with the table
(`DeviceOmegaAccumulator.on_shell_coefficients`), so a window still transfers nothing
from the host. Fe V2 P=4 on the release branch (frontier03277): total 89.33–95.02 s,
Σ τ other 6.13 s (6.67 s for the off-shell int7 tip on the same node). Against the
off-shell int7 tip, eqp0 moves ≤ 0.032 eV; eqp1 moves up to 10.4 eV on states on Σ
pole structure, because Z is now the analytic derivative (Z ∈ [−1.44, 3.20]; 241 of
1664 states with |ΔZ| > 0.1). The Fe reference is re-established on this branch.
Follow-up: an on-shell `window_pick` (today the table's dense coefficient is still
computed on the device and discarded on shell; transient capacity × n_ω complex).

**What a user must change.**
- **Per-stage walls need `LORRAX_STAGE_SYNC=1`.** Timing-only device waits no
  longer run by default. Totals are exact, but a stage row is host-attributed:
  device work is charged to the first later stage that reads it. The stage table
  prints which attribution is in force.
- **The environment freezes at start-up step 7b.** A LORRAX write that adds an
  environment key after that refuses (`GATE env-frozen`); another library's is
  announced. A warning of either kind in a log is a bug to report.
- **Rebuild the ROCm FFI library** from this tree (`config/frontier/build_ffi_rocm.sh`).
  New handlers: `kdft_kconv_parent`, `kdft_plane_fft_gather`, the read_multi
  phdf5 handler, and the MFMA tiles of `active_gemm_device`.
- **`wfn_trs` stamps** move to schema `wfn-trs-stamp-v2`: an old stamp is a miss
  and is measured again once.
- **Multi-rank runs no longer depend on the GPU placement.** The four-rank
  reductions on the Fe path are rank-ordered, so two runs with different
  rank→GPU maps give the same bits.
- New registry rows: `LORRAX_STAGE_SYNC`, `LORRAX_CHECK_COUNTS`,
  `LORRAX_KDFT_KBOX_ARM`.

| lane (commit) | what | measured effect (Fe V2 P=4 unless noted) | gate |
|---|---|---|---|
| ctorfix/w8 (b601b2801), mfma/w8 (43af0af0b), dput/w1 (6259b21ef), census tool (8f6a2c8fe) | stacked partner eighs and device pack tables; FP64 MFMA tiles in the device-bounds active-range GEMM; content-keyed cached placement and the device-agreed polar spectrum | merged together: total 112.18 → 104.87 s, constructor 17.75 → 14.34 s, Σ τ other 7.89 → 6.82 s, line_select 1.48 → 1.03 s; census H2D `device_put` 7545 → 1528 | active gate 64/64 at P=4; eqp 2.1e-8 / 6.2e-8 eV |
| dput/w2 (8229a52e3) | Σ window table on the device, device zeros and row slices, pack offsets on the device, SlabIO commit zeros | bank and Σ stages pass `transfer_guard("disallow")` (lane, g5b) | eqp 1.4e-8 / 2.1e-8 eV |
| kconv/w1 (446148490) | kdft modes 1 and 10 on ROCm | ζ 3.14 → 3.00 s; ζ device peak per rank 10.70 → 6.77 GB; mode 1 2.45×, mode 10 1.32× per call (lane) | `test_kdft_parent_gate.py` at P=4, 1e-12 |
| crash/w8 (c885977cc) | the environment freeze (rc 139 start-up SIGSEGV); atomic, validated `wfn_trs` stamps | 0 crashes in 48 stressed start-ups (lane) | CPU +12 tests (`test_env_guard`, cache safety) |
| lsync/final (42c9647a7) | χ₀ contour accumulate with a static-trip inner loop | χ₀ direct dispatch 12 930 → 5 798 syncs at P=1; wall neutral (lane) | eqp bitwise |
| slabmr/w8 (60c9049af) | `SlabIO.read_multi` / `PhdfReadMulti` on every leg | `sample_batch_read` 2.80–2.87 → 2.12–2.16 s; constructor host syncs 836 → 292 (lane) | phdf5 read_multi gate 10/10 at P=4 |
| lsync/w3 (8e3d3eec0) | timing-only waits under `LORRAX_STAGE_SYNC=1`; coordination-service barrier; read heartbeat off the caller | explicit waits 1024 → 126 per rank; walls 106.98 → 106.94 s same node (lane) | eqp byte-identical |
| rcgpu/w1 (ee0e014ff) | rule-construction shift pencils on the GPU | rule construction 49.23 → 33.76 s (lane) | eqp 6.2e-7 / 1.54e-6 eV (gate ≤ 5e-6) |
| mfma2/w9 (9f46a8690) | active GEMM 32×32 tile and register K prefetch | Σ τ other 7.00 → 6.67 s (lane); W full-K call 10.4 → 13.9 TF/s | active gate 64/64 |
| dput/w3 (29a402fab) | polar counts cut on the device; Coulomb sub-span start and Σ ω-slab starts on the device | rank-0 H2D 1531 → 1446 (lane) | eqp 2.6e-8 / 5.4e-8 eV vs the previous tip |
| lsync/w4 (2caadcb95) | rank-uniform status word per checked solve; deferred reciprocity census; batched digest verdicts and reduction | census D2H 455 → 425, explicit D2H 235 → 71 (merges 11–12 together) | eqp bitwise, same node |
| rcgpu/w2 (5252601a4) | host re-decision of device acceptances the dilation cannot resolve | eqp vs e5 1.5e-6 → 2.3e-8 / 5.0e-8 eV; +0.2–0.3 s (lane) | Fe gate ≤ 1e-7 eV restored |
| det/w1 (92385e7fc) | rank-order four-rank reductions (`lxkit.ordered_psum`, `ordered_psum_scatter`) | bitwise across three placements on two nodes; wall cost not measurable (lane) | hsuite scalar chain; `test_ordered_collectives.py` |

**Known issues** (release notes):
- hsuite na_sc: on the det tip a deterministic `GATE shared_pole_zero_ritz` (q=2,
  SC W map 0) on every placement; before det/w1 it was flaky (a tolerance miss of
  1.09–1.12e-1 eV against 4.3e-2, or an intermittent V_q NaN). Root cause: a
  marginal SC path sensitive to 1-ulp reduction order; owned by the crash lane.
- bisp_sc: before det/w1 it intermittently refused with an all-zero CT Gram
  (`GATE shared_pole_sector_retained_metric_positive`); after det/w1 it passed on
  every placement tested (three placements, two nodes) and on the release branch.
- On shell, the Σ window table's dense coefficient is computed and discarded
  (follow-up: an on-shell `window_pick`).

## 2026-10-16 — `bse_k_grid` with `--eqp`: the QP energies enter the htransform

`bse_jax` and `absorption_haydock` re-sliced the coarse restart's QP energies
after `bse_k_grid` had densified the bundle, onto a k axis they do not have.
`--eqp` now goes into the densifying htransform (`initialize_wfns`, its one
reader on that path), so the fine ε are quasiparticle and nothing is
re-sliced after.

- **Results.** `bse_k_grid` runs with `--eqp` run on QP energies. Runs
  without `--eqp`, or without `bse_k_grid`, are unchanged.

## 2026-10-16 — W densifier: an even coarse axis splits its Nyquist coefficient

`bse_k_grid` and `exciton_bands --w-coarse-grid` interpolate W by zero-padding
it in R (`bse.bse_densify._zeropad_R_axis`). On an even coarse axis the
Nyquist coefficient R = −n_c/2 was kept single-sided at weight 1; it is now
split, weight ½ at R = ±n_c/2. Coarse tiles are unchanged, and every fine
tile is Hermitian; single-sided, MoS₂ 6×6→12×12 off-grid tiles were
20–63 % non-Hermitian.

- **Results.** Odd coarse axes (MoS₂ 9×9, z = 1) are bit-identical. Runs that
  pad an even coarse axis move, `c1` and `srlr` alike (Si 4³→8³, MoS₂ 6×6).
  No deck change is needed.

## 2026-10-15 — `w_head_densify = srlr`: SR/LR W densification

A second coarse-to-fine W mode beside `c1` (the default, unchanged). `srlr`
splits W exactly at every coarse q into a short-range part and the screened
long-range channels (bulk: the WS head slot; slab: that slot with its G_z
shells), subtracts a fitted model of the long-range part before the
trigonometric densifier and adds it back at every fine q
([SR/LR densification](docs/theory/w-srlr-densification.md)). Coarse tiles
stay exact.

- **GW.** The static Dyson solve (`compute_mode = cohsex`, and every other
  route through `gw.screening.compute_static_w`'s static role) now also
  writes the restart group `W0_lr`: the dressed form factors F̃ and channel
  polarizability P per irreducible q, n_L·N_μ numbers per q. Cost: three
  skinny products against χ₀ and W. A χ₀/W mismatch refuses
  (`GATE lr_channels_negative_P`). The shared-pole route does not write it
  yet.
- **BSE.** `w_head_densify = srlr` needs that group and W0 on its q wedge;
  otherwise it refuses by name (`GATE srlr_restart_channels`,
  `srlr_wedge_tables`). Restarts from earlier trees, and shared-pole
  restarts, keep `c1`. `exciton_bands --w-coarse-grid` accepts `c1` only.
- **Results.** `c1` runs are unchanged. No deck change is needed.
## 2026-10-16 — dynamic Σ_c is evaluated on shell; the ω grid is `sigma_omega_grid_debug`

The τ sweep of every dynamic Σ (GN/HL-PPM, MPA, shared-pole, four-current
sectors) now folds each σ(τ) into what its readers use, not into an ω grid:
the QSGW pair Σ_mn(E_m), Σ_mn(E_n) and the band diagonals at E_eval and E_DFT
with the exact dΣ/dω ([On-shell Σ_c](docs/theory/sigma-on-shell.md)). The
Σ cube is depth 2 instead of n_ω, and the windows are planned on the states'
own energies.

Results that move:

- The QSGW Σ and eqp0 lose the ω-grid interpolation error, because they read
  Σ at the state's own energy.
- eqp1's Z is the analytic derivative, not the ±0.5 eV central difference.
- The band-extrapolation fit reads each state's Σ where H reads it.

Off-grid states keep the `sigma_out_of_grid` and Σ(ω = 0) rules.

Measured on the P4 hsuite (on shell vs the old 0.25 eV grid, and vs the
same grid route at `sigma_omega_step_ev = 0.025`; max|ΔE| / median, meV):

| file | grid 0.25 − on shell | grid 0.025 − on shell |
|---|---|---|
| gnppm eqp0 | 4.7 / 0.7 | 0.2 / 0.0 |
| spx eqp0 (spinor shared-pole) | 83.9 / 3.9 | 0.6 / 0.1 |
| na_sc map 0 QP eigenvalues | 385.3 / 0.5 | 1.4 / 0.0 |
| bisp_sc eqp0, eqp1 | 23.2 / 2.6 | 0.6 / 0.1 |
| cohsex eqp0, eqp1 | 0 | — |

The deltas are the 0.25 eV grid's interpolation error: refining the grid
converges onto the on-shell values. On spx, the 0.025 eV grid's spline matches
the on-shell Re Σ_c to 0.8 meV and the exact dΣ/dω to 0.006. The large
outliers are states on η-scale pole structure (η = 0.25 eV); for example, Na
k0 b6–8 at E − E_F = 11.3 eV, where Re Σ_c moves ~1 eV between adjacent
0.25 eV nodes.

eqp1 moves more (spx: up to 594 meV, median 15 meV): the ±0.5 eV stencil
smooths dΣ/dω over 1 eV of that structure (spx k0 n0: Z = 0.355 exact vs
0.690 stencil). The stencil error is independent of grid step: at 0.025 eV
the grid route's eqp1 still differs by 695 meV.

na_sc: map 1 now reads max|dE| = 1.95 eV, above the fixture's
`sc_tol_ev = 1.5`. The fixture therefore runs 3 maps instead of 2, and its
final eqp/Hartree references change at the eV level. This is a different
stopping map, not a different fixed point.

The extrapolation tail (inf − last, β) is unchanged. The Σ cube is 2 frequency
slots instead of n_ω = 48 (spx), 41 (bisp_sc) and 87 (na_sc). References were
regenerated at P4 from this branch; the evidence directories are
`/pscratch/sd/j/jackm/cs_claude/sigma_depth2/runs/{grid,fine,onshell2,regen}`.

`sigma_mnk.h5` keeps `sigma_c_kij_ev`/`sigma_total_kij_ev`, now `(2, nk, nb,
nb)`. It adds per-table diagonals and `sigma_evaluation = on_shell`, and has
no `omega_ev`. The eqp receipt carries the on-shell operand in place of the
C(ω) curve.

Deck changes:

- `write_eqp2 = true` needs `sigma_omega_grid_debug = true`
  (`GATE eqp2_needs_omega_grid`).
- On shell, `sigma_freq_debug.dat`, `qp_omega0_ev`, an SC run's DFT-basis
  `sigC(Edft)` and the MPA sweep checkpoint are omitted by name.
- GN/HL-PPM's q→0 head is now the closed form MPA's is; its grid values are
  unchanged.

## 2026-10-14 — one platform-leg table; ROCM is a declared leg

The per-platform facts of the native libraries (file, pin, build directory,
C-ABI suffix, ABI symbol, and how a process knows it can host a device leg)
were restated in `lxkit.gate`, `lxkit.native_provider`, `ffi.common.ffi_loader`,
`distrib_la.loader` and `runtime`. They are now one table, `lxkit.LEGS`
([platform legs](docs/architecture/ffi_layout.md#platform-legs)), and the two
loaders' copies of the CUDA-first open logic are one function
(`lxkit.accelerator_leg_to_open_first`).

No result moves on CUDA or cpu, and no deck or environment change is needed
there: the CUDA and host legs keep their file names, pins and ABI symbols, and
a sealed CUDA bundle loads as before (`tests/test_platform_legs.py` pins the
facts). On a ROCm mesh the FFI platform key is now `ROCM`, a declared leg
whose library (`liblorrax_ffi_rocm.so`, pin `LORRAX_FFI_ROCM_SO`) is not built
yet: a request for a native target refuses by naming that file, where it
used to refuse with `unknown platform 'rocm'`. `distrib_la.resolve` keys its
ROCm rows `ROCM`. On WSL2 the loaders now see the mapped libcuda as a CUDA
device, as `runtime` already did, so the CUDA leg opens before the host leg
there too.

## 2026-10-13 — FFI bundle: the LU factors are a declared output; both legs carry RUNPATH $ORIGIN

The fused distributed LU solves behind `linalg = distributed` (the W Dyson
solve: cuSOLVERMp on CUDA, ScaLAPACK on CPU) now call new targets,
`lorrax_cusolvermp_batched_solve_lu_factors` and
`lorrax_scalapack_batched_solve_lu_factors`. They return the LU factors as a
second output aliased to A. The old targets wrote the factors into A without
declaring it, so a reader of A later in the same program, or the caller's
own array, could see the factors. XLA now copies A only while something still
reads it. On the P4 hsuite eqp is byte-identical in 10/10 files, and the
release-cold wall is unchanged (327.8 s against 328.2 s on main; claim TBD).
The ABI-6 targets stay in the libraries for older trees, and the handler ABI
stays 6.

A tree at or after this commit needs a bundle built from it. On an older
bundle its distributed LU refuses at resolve, naming the missing target. On
CPU meshes the ScaLAPACK backend imports again: since 8deacaeae
`distrib_la._scalapack` imported a name `_slate` no longer defines, so every
CPU distributed eigh and LU failed at import.

Both legs now carry one search path, `$ORIGIN`, as a `DT_RUNPATH`, so the
run environment's `LD_LIBRARY_PATH` decides every machine library.
`config/perlmutter/ffi_mpi.sh` lists the pinned HDF5, cray-mpich and LibSci
directories, and the module puts them ahead of `/opt/cray/pe/lib64`. With
them, a GPU and a CPU process map the same files as with the previous bundle
(claim TBD). Without the LibSci directory, a run refuses at load because it
would map two MPI runtimes (claim TBD). A site module that loads this bundle
must name these directories. `seal_bundle.py` refuses a leg whose search path
is not one `$ORIGIN`-first `DT_RUNPATH`, and any bundled library that
searches a relative, scratch or build directory.

kconv_outer's build log now prints `store FAILED` when a cubin could not be
written to the disk cache. No deck change.

## 2026-10-13 — the device budget can only lower; memory-driven routes print once and fall back

- **Budget.** A `memory_per_device_gb` above the memory rule no longer raises
  the budget: the run warns and keeps the rule (72 on an A100-40GB runs at
  31.60 GB). A caller-owned BFC allocator's `bytes_limit` also caps it. Under
  the runtime's own pool nothing changes.
- **W Dyson under `linalg = local`.** The charge solve, like the photon solve
  before it, takes the distributed plan when one rank's whole matrices do not
  fit the budget, instead of stopping at `GATE xla_rematerialization` or running
  out of memory. The photon price now also counts the face-tiled V, χ and W.
- **Provider contexts.** A `local` run that falls back to cuSOLVERMp is
  charged the 3 GB of contexts the local rule did not withhold; when neither
  plan fits, the run takes the one less over and warns, naming
  `linalg = distributed`.
- **One record.** Every memory-driven route (W Dyson, SC eigh, density scan,
  shared-pole constructor, line selection, staged and face eigh stacks, head Γ
  GEMM) prints once as `route <site>: <route> (<prices>)` and is listed in the
  report's Routes table. The old per-site lines (`[W solve] linalg =
  distributed -> …`, `Shared-pole sector constructor: route …`,
  `distrib_la eigh stack …`, the photon and SC-eigh fallback notes, the
  staged-eigh warning) are gone.
- **cuFFT pricing.** An HLO FFT parse miss or an unavailable cuFFT query no
  longer refuses (the Σ τ admission, the canonical WFN r-slab preflight): it
  warns and prices the scratch conservatively.

No deck or environment change. claim TBD: P4 hsuite eqp identity, route
equivalence and the A100-80 budget simulation.

## 2026-10-13 — Perlmutter site profile; launcher-agnostic distributed start

Perlmutter's NCCL network values (the OFI plugin path, the site NCCL module's
settings, the `no_vni` step requirement) and the collective cost model's
constants moved from `runtime/network_env.py` and `gw/comm_model.py` into
`config/perlmutter/site.toml`, selected by `NERSC_HOST` or
`LORRAX_SITE_PROFILE`. On Perlmutter the same settings apply and the same
refusals fire. Startup also changes in these ways:

- World size and rank come from the `JAX_*` overrides, then
  `OMPI_COMM_WORLD_*`, `MV2_COMM_WORLD_*`, `PMI_SIZE`/`PMI_RANK`, then
  `SLURM_NTASKS`/`SLURM_PROCID`. A launch whose tasks would each form a
  one-process world refuses in every driver (`GATE split_launch`); before,
  only drivers on `RunSession` checked, after the fact.
- The coordinator is the Slurm step's first host, from a Python host-list
  parse instead of `scontrol`, on port 61440 + (`SLURM_JOB_ID` + 257
  `SLURM_STEP_ID`) mod 4096 instead of 22000 + `SLURM_STEP_ID`. At P > 1
  outside a Slurm step, with no `JAX_COORDINATOR_ADDRESS` and nothing jax's
  cluster detection recognizes, startup refuses (`GATE coordinator_address`)
  instead of using each rank's own host.
- At P > 1, a GPU process that addresses more than one device refuses at
  startup (`GATE one_gpu_per_process`). One process over several GPUs is
  unchanged: kmeans runs that way, and the other drivers refuse it at their
  first SlabIO open.
- On a Slingshot node with no site profile, a multi-node CUDA step gets
  `FI_CXI_DISABLE_HOST_REGISTER=1` and `FI_CXI_RDZV_THRESHOLD=0` and a warning
  naming the missing profile. A multi-node CUDA step with neither `NCCL_NET`
  nor a net plugin warns.
- `jax.distributed.is_initialized()` replaces the inherited
  `_LORRAX_JAX_DISTRIBUTED_DONE` sentinel, so a driver started by a P > 1
  parent initializes its own world.

On the P4 hsuite eqp is byte-identical in 10/10 files, cold and warm, and the
cold wall is 324.8 s against 324.3 s on main on the same node. On a 2-node P8
step the profile applies the same settings as before and all_to_all runs at
31.3 GB/s per rank against 32.0 on main (claim TBD). No deck change; an
install without the `config/` directory has no site profile.

## 2026-10-13 — I/O and environment safety checks

- **Shared output directory.** A SlabIO `"w"` or `"a"` open at P > 1 now checks,
  once per directory, that every rank sees one filesystem there (rank 0 writes a
  nonce file; every rank reads it back). A node-local directory (`/tmp`, local
  NVMe, a per-node bind) refuses on every rank with `GATE io_global_commit`,
  stage `SlabIO.shared_directory`, naming the directory and the failing host.
  Before, each node wrote its own partial file and rank 0 committed node 0's.
  Single-node runs in `/tmp` are unaffected. Put a multi-node run directory on
  `$SCRATCH` or CFS.
- **HDF5 file locking.** The runtime default of `HDF5_USE_FILE_LOCKING` is now
  `BEST_EFFORT` (was `FALSE`): h5py's opens take `flock` locks where the
  filesystem supports them, so another process's open of a file held for
  writing refuses ("unable to lock file") instead of racing. The FFI's MPI-IO
  driver takes no locks, so the two HDF5 stacks do not refuse each other.
  An exported value still wins; `config/frontera/ffi_env.sh` no longer exports
  `FALSE`. On Perlmutter NERSC's Slurm task prolog exports `FALSE` into every
  task that has no value, so Perlmutter runs keep `FALSE` unless the launch
  exports `BEST_EFFORT`; with it exported, the P4 hsuite is unchanged (claim TBD).
- **Host memory.** `MemTotal` and `MemAvailable` are capped by the job's memory
  cgroup (v2 `memory.max`/`memory.current`, v1 `limit_in_bytes`/`usage_in_bytes`,
  page cache it can reclaim not counted as used), so the host-residence choices
  and the CPU device budget see a batch job's limit, not the node's.
- **Startup checks.** A CUDA device below compute capability 8.0 refuses
  (`GATE cuda-sm80`). On a CPU mesh the host FFT and batched GEMM engines run
  once at startup (`GATE plan-engine`). A failed mathdx probe still sends every
  k-convolution to XLA, but rank 0 now prints the demotion with its cause and
  cost, and every compile receipt ends `demoted kconv` (claim TBD for the cost
  figures, which are `docs/architecture/kconv.md#why-fused`'s).
- **JAX private surface.** `runtime.jax_support` checks the parameter names, not
  only the counts, of every `jax._src` callable LORRAX patches, now including
  `compile_or_get_cached`, `backend_compile_and_load` and
  `ExecuteReplicated.__init__`. At P > 1 a compile-agreement or device-fit-gate
  patch that cannot install refuses by name (`GATE compile-agreement`); at P = 1
  it warns.
- **ROCm.** SlabIO keys its library on the device vendor, so a ROCm mesh
  refuses by name (`unknown platform 'rocm'`) instead of loading the CUDA
  library; `distrib_la` warns when a non-CPU device takes the host LAPACK
  subspace plan.

On the P4 hsuite (one A100 node, against main 94741c5ae) eqp is byte-identical
in 10/10 files cold, warm and with `BEST_EFFORT` in force; the release-cold
wall is 324.1 s against 324.3 s, with the same 3036 lookups and 2260 compiles
(claim TBD). No deck change.


## 2026-10-13 — dipole: the kinetic velocity rides the velocity sweep

With `parallel_transport_out` set, the dipole stage now computes the p-only
kinetic velocity as the second operator of the velocity sweep, not in a
second sweep over the same ψ: one program fewer per dipole stage (P4 hsuite
lookups 3039 → 3036, cold compiles 2263 → 2260). `dipole.h5` is identical in
every dataset; `parallel_transport.h5` `velocity_kinetic_cart` moves by at
most one ulp (2.2e-16). eqp is byte-identical in 10/10 files. No deck or
environment change.

## 2026-10-13 — SC final diagnostics with sigma_freq_debug_output on a static Sigma

A self-consistent COHSEX or exchange-only run with `sigma_freq_debug_output =
true` no longer dies in its final diagnostics (`ValueError: Size of label
'n'`, the head rotation) when the mesh does not divide the SC band count (5
bands at 2×2). A static Σ has no band carrier, so the final rotation pads U to
the mesh; the head diagonals now rotate with U's logical block. Σ, H and eqp
are unchanged: on the P4 hsuite eqp is byte-identical in 15/15 files. No deck
or environment change.

## 2026-10-13 — CPU ranks at P > 1 hit the persistent compile cache

On CPU at P > 1, ranks 1..P-1 now find the entries rank 0 writes. JAX's CPU
key carried each process's device assignment and the head of its serialized
topology, which lists the process's own devices. No peer key matched rank
0's, so the peers compiled every program on every run. At P > 1
`common.jax_compile_cache` now strips both, as JAX already strips the device
assignment on GPU. The key keeps the CPU target machine (triple, CPU name,
features). On a warm CPU P4 run of six hsuite stages each rank makes 633
requests. Before, ranks 1–3 hit 0 and compiled 642 programs each; now every
rank hits 630 and compiles 12, and the wall goes from 46.7 to 23.9 s. eqp is
byte-identical. Every CPU key at P > 1 changes, so such a cache is cold once;
P = 1 keys are unchanged. GPU keys are unchanged: on the P4 hsuite all 2250
GPU programs keep their key. The CPU-device programs inside a GPU run (4 on
the hsuite) take the new CPU key and compile once. No deck or environment
change.

## 2026-10-12 — SC final diagnostics with sigma_freq_debug_output on a symmetry-reduced grid

A self-consistent run with `sigma_freq_debug_output = true` on a
symmetry-reduced k-grid no longer dies in its final diagnostics (`ValueError:
Size of label 'k'`, the head rotation). It died at any `sc_max_iter` once it
reached them; a run that ends in `GATE sc_fixed_point_not_converged` never
did. The map's retained static head terms are now on the loop k-set, like Σ
and U beside them. Σ, H and eqp are unchanged: on the P4 hsuite eqp is
byte-identical in 15/15 files, and each SC stage compiles one more small
program (the row gather). No deck or environment change.

## 2026-10-10 — programs whose shapes are known ahead compile on helper threads

`common.jax_compile_cache.compile_ahead(program, *args)` lowers a program on
the calling thread, takes its cross-rank compile-agreement number there, and
hands XLA's backend compile to one of min(8, physical cores) helper threads.
The live call finds the executable or waits for it. If the live call reaches
the compile before the helper does, it compiles under the reserved number and
the helper takes its executable, so every rank numbers its requests alike.
Two owners use it. The scalar shared-pole W-model setup submits H = V^(1/2)
of the bare V's parents (decks with a distributed `linalg`) and, with a head
correction, the MPA scalar head fit (from the deck's `n_poles`) before the
zeta fit. A streamed bank submits all its row
passes at once. On the P4 hsuite, release cold goes from 336.7 to 326.3 s and
warm stays at 140 s. Program requests (3037 per rank) and XLA compiles (2261)
do not change, and eqp is byte-identical in 10/10 files. The runtime now sets
`jax_disallow_mesh_context_manager`, so a new `with mesh:` raises. When the
persistent cache is off (`ISDF_JAX_CACHE_DIR=""`,
`JAX_COMPILATION_CACHE_MAX_SIZE=0`, or a directory that cannot be armed), the
runtime now also clears `jax_compilation_cache_dir`. Before this, an exported
`JAX_COMPILATION_CACHE_DIR` stayed live behind the OFF line. No deck or
environment change, and no opt-out: results are bitwise.

## 2026-10-10 — a release carries a cubin store that seeds the per-user cubin cache

A release now carries `cubin_store/` at its source root: the NVRTC images of
its target architecture, built once at install by `scripts/build_cubin_store.py`.
A process's first `ffi.fft.cubin_cache_dir()` call links every store image the
per-user cubin cache lacks into it; an image already in the cache is kept, and
a bad store image is rebuilt into the cache while the store is left as it was.
The cache path, and so every JAX compile key, is the same with or without a
store, so a warm JAX cache stays warm. A release-cold P4 hsuite builds no NVRTC
image and takes 389.7 s against 575.7 s without the store (sandbox claim 4126).
A checkout has no store and builds each image once per user, as before.
Results are unchanged. No deck or environment change
([Compilation §2](docs/architecture/compilation.md#2-native-kernels-and-nvrtc-images)).

## 2026-10-10 — `src/` enters no legacy mesh context; both W branches share one window program

Nothing in `src/` or the services opens `with mesh:` any more. Every sharding
there is explicit, so the context was never read; it only put the mesh stack
into every jit's trace and lowering key, and a nested block re-traced and
re-lowered programs identical to earlier ones. `tests/test_glue3_program_counts.py`
fails on a new one. In `gw.mpa`, `WSynthesis` takes no `ordered` and its
`w_kernel` no `hole`: `window_operands(space, ...)` passes the branch's tables,
`SynthesisTau.window_kernel()` takes no argument, `kconv_tables` may be a
function of the window's `space`, `sector_sigma.ParentW` is `(W, partner)` and a
sector node's `loads(space)` picks the branch. Host (k, band) head tables stay
NumPy. Results are bitwise; the P4 hsuite's warm program lookups per rank go
from 3167 to 3037. No deck or environment change.

## 2026-10-10 — `--xla_gpu_enable_llvm_module_compilation_parallelism=true` is a runtime default

`runtime.set_default_xla_gpu_autotune` now appends a second GPU compile flag
to `XLA_FLAGS` beside `--xla_gpu_autotune_level=0`, and XLA compiles each
module's LLVM IR in parallel parts. A caller's value of either flag wins, flag
by flag; a CPU startup is untouched. Results are bitwise and the device peak is
unchanged. The P4 hsuite's release-cold XLA compile goes from 219.5 to 195.2 s
(sandbox claim 4179); the production decks are on
[Compilation](docs/architecture/compilation.md#4-where-the-time-goes). JAX's
persistent-cache key hashes `XLA_FLAGS`, so the first GPU run after this change
compiles every program once. The startup report names both flags and their
provenance. To opt out, export
`XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=false`. No deck
change.

## 2026-10-09 — `tests/hsuite` uses the runtime's compile cache and reports a per-stage compile census

With `HSUITE_CACHE_DIR` unset, the suite no longer points `ISDF_JAX_CACHE_DIR`
at `$SCRATCH/.cache/lorrax/hsuite/jax-cache`, a directory that was never
namespaced by FFI bundle or pruned. It uses the runtime's cache instead: an
exported `ISDF_JAX_CACHE_DIR`, else the default namespace. The first run after
this change is therefore JAX-cold once. The old `hsuite/jax-cache` directory is
no longer read and may be deleted. `HSUITE_CACHE_DIR` still moves the cache,
and `python -m tests.hsuite.chain` still defaults to a cold `<out>/jax-cache`.

`summary.json` gains `compile_s`, which splits each stage's wall into trace,
lowering, cache reads and XLA compiles, the compile agreement, and NVRTC
cubin builds (`tests/hsuite/README.md`, "Wall time and caches"). Each stage
log now ends with a compile receipt. An SC driver's map-0 receipt therefore
counts that driver only, not every stage since the process started. No result
moves and no deck changes.

## 2026-10-09 — the four-current χ₀ stream applies the Dirac parity on k rows on the XLA and cpu routes

`ffi.fft.make_kconv_chi_vertex` (mode 11 with channel vertices) on the XLA backend and on the
cpu plan backend multiplied the inverse-transformed (R-space) Gc operand by `sign_c`, the
per-k Dirac parity of the four-current stream's mixed quadrants
(`gw.w_isdf._photon_chi_kconvs`). It now multiplies Gc's unfolded k rows before the transform,
as the mathdx kernel does on its load. Results move on those two routes for every bispinor
four-current response (`bispinor_gw = full_shared_pole`, and the photon χ₀ streams generally)
whose symmetry group has an operation with parity −1. On those routes the bare moments were
not Hermitian (CrI3 6×6 bispinor SC at P4: `GATE response_moment_hermiticity`, 1.8e-6 against
3.3e-11), so such runs refused in map 0. The mathdx route does not move. No deck change.

## 2026-10-09 — the memory rule's φ is stated as what it covers; `XlaGpuMemoryEnv.peak_note` goes

`φ` (`runtime.POOL_OVERSHOOT`, 0.19) is the fraction by which the pool's in-use
bytes may run above the planner budget: peaks the planners under-price, and free
space trapped between live buffers. The pool's reserved bytes can also rise past its
reservation for an instant, when a request meets only frees still pending on XLA's
stream; that excess takes the card's free memory until the next stream synchronize
and is not in `φ` (`docs/architecture/memory-model.md#pool-growth`). The start-up
warning for a `memory_per_device_gb` above the rule now says what the margin covers
instead of quoting reserved high-water marks. `runtime.xla_memory.XlaGpuMemoryEnv`
loses its `peak_note` field, which nothing read. No result moves; no deck change.

## 2026-10-09 — the one-shot reads deep semicore states on the coarse windows, not at Σ(ω = 0)

A dynamic one-shot now builds the coarse (semicore) class as an SC run does
(`gw_init.coarse_class_for_deck`). A coarse state outside the requested set,
that is deeper than the W model's 15 eV active depth, reads its own Σ(E) on
SC map 0's coarse windows (η_semi = 5 eV) instead of Σ(ω = 0). Requested
states, including a requested coarse state, keep the near grid, which is
unchanged (`qp_support.oneshot_support_ev`). On AgI 6³ with
`number_bands_protected = 64` (P4) the Ag 4s, Ag 4p and I 4d eqp0 move by
+4.1 to +8.0 eV, to within 0.5–2.2 eV of a converged deck-η read (Σ(0) was
4.6–10.2 eV from it), and eqp within 10 eV of midgap moves by at most
0.28 meV. The Σ sweep takes 98 more τ pairs there (485 → 583). The one-shot no
longer warns that those states use Σ(ω = 0). A `sigma_omega_patches_ev`
`lo:hi:eta` triple now serves a one-shot's coarse states outside the requested
set; a one-shot with none warns once that the triples have no state to serve
and ignores them, as it ignored them before. No deck change. The physics is in
[self-consistency §2](docs/self_consistency.md#2-band-treatment).

## 2026-10-09 — `linalg` is the only dense-layout dial; the ζ stamp drops `distributed_lu`

`exciton_bands` and `htransform` no longer take `--eigh-backend` or
`--distrib-la-batched-route`: the deck's `linalg = local | distributed`
selects the eigensolver and the batch route, and a command line that passes
either flag fails at argument parsing. Remove the flag.

The ζ fit provenance no longer records `distributed_lu`; no stage read it, and
the transverse factor is the whole-tile local LU on every layout
(`transverse_solver_kind` records it). A ζ file stamped by an earlier build
still reuses, because the retired key is dropped before the stamps are
compared. The charge-ζ identity digest drops it too, so a restart bundle or
MPA store that bound the identity before this change no longer matches it and
is treated as stale by its consumer. The resolved backend config has no
`distributed_lu` or `distributed_cholesky` field, and the hand-built-config
gates of `head.bispinor_tt_head_correction` (a field no deck could set) are
gone with the field. No result moves.

## 2026-10-09 — log and error text names the `linalg` dial, not retired deck keys

Messages that named a retired deck key now name `linalg`. The W solve line
reads `[W solve] linalg = distributed -> …` (was `w_dyson_solver=distributed`);
the SC eigh line ends `route from linalg: …` (was `sc_eigh=…`); the ScaLAPACK
host-only refusals say that `linalg = distributed` resolves to cuSOLVERMp on a
CUDA mesh; the fH_q eigh log tags `[low-memory eigh, linalg = distributed]`.
`exciton_bands --eigh-backend` and `--distrib-la-batched-route` are debug
overrides of what `linalg` resolves, as htransform's are; their help says so.
Scripts that grep the old strings must read the new ones. No result moves.

## 2026-10-08 — one memory rule sets the device budget and the pool; route G takes its chunk and plane groups from the fixed tile

With `M` the card total and `O` the bytes outside the XLA pool (3 GB, plus 3 GB under
`linalg = distributed`), the planner budget is `B = (M − 1.2·O)/(1 + 1.2·φ)` with `φ = 0.19`
and the pool reserves `M − 1.2·O_max` (`runtime.planner_budget_bytes`, `runtime.pool_fraction`).
A100-40GB: pool fraction 0.89 → 0.830, budget 33.97 → 31.60 GB (28.66 distributed). A100-80GB:
fraction 0.89 → 0.915, budget 68.16 → 66.36 GB (63.43 distributed). A positive `memory_per_device_gb` is used as given; above the rule it warns at start-up
with both numbers. On CPU the rule takes the node's `MemTotal` over its processes, shared by a
process's devices (it was host RAM over the global device count). Every GPU run prints the bytes outside the pool after warm-up (max over
ranks) and warns above 0.75 of `M − R`. Deleted: `runtime.GPU_POOL_FRACTION`, the 0.90 × `bytes_limit`
budget, the nvidia-smi fallback, the 4 GB defaults, the BFC fragmentation table (planners now fill
the budget itself), `RUNTIME_RESERVE_BYTES`, `ISDF_CHUNK_TARGET_UTILIZATION` (unset it).
The route-G ζ fit takes its stage-0 parent chunk and its plane-group width from the fixed tile,
not from A100 timings; `gw.comm_model` is a receipt diagnostic only. Budget-sized stages may
choose other sizes, which moves eqp at round-off. At P4 on A100-40GB against main 92cb1fe0e:
CrI3 6×6 bispinor SC maps 0–2 within 0.001 µeV. CrI3 8×8 GN-PPM charge within 1.1 µeV on every
state above −15 eV; the semicore states near −17 eV, which that deck's QP solve moves by more
than 10 eV, differ by up to 20 meV in map 1. ζ fit: 56.4 → 49.8 s (charge) and 40.9 → 40.1 s
(bispinor). The final rule was not re-gated on A100-80GB (claim 3978). No deck change.

## 2026-10-08 — the response streams' block accumulate no longer materializes its terms

The XLA contour block accumulate (`gw.contour_accumulator.contour_block_accumulate_local`,
which replaced the CUDA kernel on 2026-10-08) summed a chunk's terms as one unrolled chain,
and XLA materialized every term's `[output, q, rows, cols]` product at once. On the Fe 4³
forced-face bispinor SC deck at P4 (40 GB) the direct stream's peak went from 9.87 to
45.23 GB and its compile from 11.1 to 66.8 s. It now adds one term per `fori_loop` step,
in place on the accumulator. The peak is back to 9.87 GB (the CUDA kernel's); the direct
dispatch is 13.8 s against the kernel's 9.0 s on Fe (0.97 against 0.69 s on CrI3 6×6), and
W per map is 2-3 % slower. eqp is bitwise against both the chain and the kernel. No deck change.

## 2026-10-08 — the staged sector selection runs each round at once; held checks read one sample at a time again

The staged route's CC/TT selection ran in sub-batches of the 1 GiB tile (3 parents for TT at
CrI3 24×24 P64), each with its own reads and eigh stacks: TT.all took 251 s per warm map
against 186 s before the staged route. The selection's eighs run one whole matrix per rank, so
its unit is the round's layer of R ≤ P parents (decisions.md#fixed-tile), and it now runs the
whole round at once; its eigh stacks take route (c) or the mesh by the same rule as every other
stack. The held checks read one held sample at a time again: reading them all in one call kept
the tiles of every sample live and raised the held section's peak from 25.8 to 40.5 GB at P64,
which no row priced. Results move at round-off only. No deck change.

## 2026-10-08 — a k grid mathdx cannot hold runs on the XLA backend instead of refusing

On CUDA the k-convolution factories now take the XLA backend (`jnp.fft` along the k axes),
with one `RuntimeWarning` and one router line per grid, where `ffi.fft.mathdx_refusal` gives
a reason: a k axis above 40, a k-box split-arm plane tile beyond the device's opt-in shared
memory (A100 to 25³, sm_86/89 to 19³), a k-box column beyond it for the BSE k-minor modes, or
a startup probe compile that failed on any process. These used to refuse
(`GATE mathdx-kconv-axis`, the k-box residency gates at kernel build, `GATE mathdx-probe`,
`GATE response_vertex_grid`). The decision uses only the grid and the device attribute, so
every rank makes the same one. Grids mathdx holds keep it, and their results do not move.
No deck change.

## 2026-10-08 — the screened packed photon Dyson solve follows `linalg`; `full_static_cohsex` runs under `linalg = local`

`bispinor_gw = full_static_cohsex` no longer requires `linalg = distributed`. Its packed Dyson
solve takes the deck's W plan: under `local` it is route (c), whole matrices per rank, unless
one rank's V, χ and W for its ⌈n_q/P⌉ parents plus one LU pair exceed the device budget,
when it solves on the distributed plan with one warning. CrI3 6×6 40 Ry COHSEX at P4: the
two plans give eqp equal to the printed digit (1288/1288 rows), and the screening stage takes
11.7 s on route (c) against 15.3 s on cuSOLVERMp. `compute_static_photon_response` defaults to
`dyson_solver = 'local'`. No deck change; a deck that set `linalg = distributed` only to pass
the old refusal can drop it.

## 2026-10-08 — the cuBLASMp GEMM is gone; every distributed GEMM is panel_matmul

`distrib_la.matmul(batched_route='auto')` and `gemm_plan(layout='face')` run
`distrib_la.panel_matmul`, the batched 2-D SUMMA in XLA, on every platform and for every
backend name except `off`; `cublasmp`, `cusolvermp`, `scalapack` and `slate` are still
accepted and run it too (the last two used to refuse). At P4 on A100-80 it is 1.4-1.5x
faster than the cuBLASMp face on the ζ-projector and SC-rotation shapes, to 2e-15. The face
plan needs no native context and no process-per-cell layout, so it also runs on a one-process
CPU mesh; it compiles at its first call (`warmup` reaches only axis plans), and
`GemmPlan.local_call` is gone (no caller). The GEMM workspace query returns 0 for a
distributed GEMM. Deleted: `src/ffi/cpp/cublasmp/` (the batched GEMM, its active-range
variants, the uncalled fused W-solve and its kernels) and their loader rows, with the
loader rows of `lorrax_slate_trsm` and the SLATE/ScaLAPACK batched GEMMs; the B11 bundle
keeps the handlers. `LORRAX_FFI_EXPECT_BACKENDS` no longer lists `cublasmp`. Results move at
round-off: at P4, eqp within 0.002 µeV on the CrI3 6×6 charge one-shot, the CrI3 bispinor SC
maps 0-2 and the Na SC deck (`linalg = distributed`). Na's Σ_c(ω) moves by up to 6 meV at the
frequency points where |Σ_c| is 130-435 eV (relative 1.4e-5). No deck change.

The distributed LU's result check (`distrib_la.plan('solve_lu').batched` inside a jitted
program) could report a correct solve as failed and return NaN: the cuSOLVERMp handler writes
its factors into A without declaring the alias, and XLA could read A for the check after the
call. The check's sketch is now fenced before the call.

## 2026-10-08 — the contour accumulator and the spin rotation run in XLA

The two small CUDA kernels with no measured gain are deleted: the contour accumulator
(`A[o] += p[o]·c` on the response-bank streams; `gw.contour_accumulator` now holds the XLA
sum and its block form) and the spin rotation of `symmetry_maps.unfold_spin_centroid_operator`
(the two JAX contractions it already had for other widths). Startup no longer checks their
handlers (`ffi_loader.require_cuda_handlers` is gone). The B11 bundle keeps the handlers;
the next bundle drops `response/contour_accumulate*` and `symmetry/spin_rotate*`. eqp is
bitwise on the CrI3 6×6 charge shared-pole one-shot and bispinor SC maps 0-2 at P4. The XLA
sum costs time on the χ₀ direct stream: dispatch 0.69 → 1.10 s and compile 0.79 → 1.64 s,
and the one-shot run goes 106.6 → 114.3 s (+7 %); the kernel's 1.6× on its stage is under the
ruling's 2× bar. No deck or environment change.

## 2026-10-08 — the k-convolution router has an XLA backend; LORRAX_FFT_FFI is gone

`ffi.fft.kconv_backend` returns `mathdx` on CUDA, `plan` on cpu (the host FFTW3-ABI route,
kept because it is 3.8-7.4x faster than `jnp.fft` at a CrI3 6x6 rank tile, claim 3979) and
`xla` on every other platform (ROCm). The XLA backend is the reference composition each
mathdx mode replaced: the same unfolds, products and spin sums with `jnp.fft` along the k
axes. `ffi.fft.xla_reference()` builds factories on it on any platform;
`tests/test_kconv_xla_gate.py` holds modes 0, 2-5, 7, 9, 10 and 11 to 1e-12 of it on the same
GPUs at P4. `LORRAX_FFT_FFI`, `make_flat_k_fft_ffi` and the `LORRAX_KFFT_CPU_TEST_XLA`
hook are deleted (neither variable is read). Results do not move on CUDA or cpu. Drivers
import their k-axis factories from `common.fft_helpers`.

## 2026-10-08 — distrib_la's budgeted eigh-stack decision is priced from the shapes

A plan built with `budget_bytes` used to compile each eager route-(c) candidate to measure it
and to agree the size over ranks through the KV store. It now prices every candidate from the
shapes (`BATCH_EIGH_TILES` = 8 n² per whole matrix a rank holds, the measured first attempt,
plus the checked chain's temporaries), eager and traced alike: no sizing compile and no
exchange. `distrib_la.SIZING_FAILED` and `StackRoute.sizing_seconds` are removed. The scalar
face constructor, the remaining budgeted caller, admits route (c) at that price where the
traced price was about 6 n². No result moves; a stack near its room's edge may run on the
whole mesh instead of route (c). No deck change.

## 2026-10-08 — dead distrib_la surfaces removed: factor/solve, dispatch_batched_eigh, SLATE trsm, two GEMM wrappers

`distrib_la.factor`, `distrib_la.solve` and `FactorToken` had no caller and are gone; so is
`distrib_la.dispatch_batched_eigh` (its one caller, the QSGW density eigh, calls
`distrib_la.plan('eigh', ...).batched` directly, the same plan and route). The SLATE
`distributed_trsm` wrapper and the SLATE and ScaLAPACK batched GEMM wrappers (whose handlers are
not built, so a request refused at the probe) are removed, and `panel_matmul` refuses a 4-D
operand or a non-square mesh by name instead of falling back to its masked-psum kernel. No
result moves. Out-of-tree code that called the removed functions calls `distrib_la.plan`.

## 2026-10-08 — the bispinor sector route is decided from the shapes: q-local, or staged rounds of min(nq, P) parents

The sector constructor's priced route tree is replaced. The route is decided once per map from
the recipe shapes and printed on one line (`Shared-pole sector constructor: route`): q-local rounds of P parents
when every sector's local round fits, as before; otherwise rounds of R = min(nq, P) parents on
the face, for every deck. In each round TT, CC and then CT reduce through the staged programs:
the selection in sub-batches of the 1 GiB tile; the GEMM stages over the widest halving of R
whose shape-priced program fits beside the round's stacks; each eigh once over the round's
stack, on route (c) when its shape price fits the budget (`distrib_la.eigh_stack_bytes`,
8 n² per matrix per rank), else on the whole mesh with one warning. A second line per round
(`Shared-pole sector constructor: round of parents a..b:`) names its sides, stage widths and eigh routes.
Deleted: the sector face batch and its step-down, the decoupled admission and its face-round
fallback, the selection room and width, the eigh rooms and sizing compiles of the sector path,
the local CT rerun on the face, the held-versus-live width fallbacks of the CT spans and the
writer carrier, the free-memory warning beside the eighs, and the cuBLASMp workspace quote of
the face constructor. The stage rows now price the eigenvector stacks the stages read (2.6 GB
per rank for TT and 4.6 GB for CT at CrI3 24×24 P64), and a face CT price no longer counts its
program twice. The held checks read every held sample of a sector in one call. The sector
W(τ) right operand is again formed at each node (forming it once moved nothing at P4); CC's one
reduce-scatter stays. Results move at round-off only: P4, maps 0–2, CrI3 6×6 forced onto the face
max |ΔE_qp| 0.001 µeV against main (W per map 214.7 / 123.5 / 65.3 s against 248.8 / 189.0 / 66.4,
peak 25.8 against 26.9 GB); Fe 4³ forced onto the face 0.041 µeV (four rounds of 4 where main ran
one face round of 13: W 180.0 / 92.2 / 124.7 s against 141.2 / 88.6 / 99.5, peak 9.9 against 15.1 GB);
Fe 4³ at 4 GB, q-local on both, bitwise. No deck change.

## 2026-10-08 — the platform key is the device vendor; one gate implementation

`lxkit.device_vendor` reads the vendor from the device client (`client.platform`,
`platform_version`, `device_kind`); `device.platform` is `gpu` for every GPU vendor and
is no longer read as CUDA. `mesh_ffi_platform` maps `cuda` to `CUDA` and `cpu` to `cpu`;
`rocm` passes through, so no gate, loader or startup report opens `liblorrax_ffi.so` on
a ROCm mesh. `ffi.gate` is now a binding of `lxkit.gate` (one announcement set; the unused
`auto` mode is gone). Before the backend exists, `runtime._gpu_is_present` also sees AMD
devices (`/dev/kfd`, masked by an empty `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`),
the default `JAX_PLATFORMS` names the installed jaxlib plugin's platform (`cuda,cpu` or
`rocm,cpu`), and `local_device_ids` come from the first visibility variable set. No
result moves on CUDA or cpu; no deck or environment change. ROCm still refuses at the
k-convolution router until it has an XLA backend.

## 2026-10-08 — two sector route crashes fixed (relaxed tier on the face route; CT alone on the face)

`sigma_w_accuracy = relaxed` with `bispinor_gw = full_shared_pole` stopped at bank build with a
`TypeError` whenever a sector took the face route: the relaxed tier has no pole budget, so the
face Ritz carrier is None, and the decoupled admission multiplied by it. It now prices the whole
H'_vv side, as the stage price does. A deck with CC and TT q-local, CT on the face, nq ≥ P and
decoupling refused stopped with `ValueError: max() iterable argument is empty` in the face-rounds
warning; the warning now names CT and its q-local need. No result moves. No deck change.

## 2026-10-08 — XLA is the reference path; vendor routes must earn their place

One owner ruling replaces two (`docs/architecture/decisions.md#xla-reference`): the
2026-08-01 "FFI backends are required" ruling and the 2026-09-24 "mathdx is the only
NVIDIA k-convolution route" ruling are deleted. XLA is the reference path for every
operation on every platform. A vendor kernel or library stays only when it is ≥ 2×
faster on a production stage or decisive on memory, behind one service, gated against
XLA on the same device; cuSOLVERMp stays the capacity route. The entry lists what does
not conform yet. Docs only: no result moves, no deck or environment change.

## 2026-10-07 — the decoupled sector rounds price CT's outputs and every array the sectors and rounds hold

The round loop's rows on the decoupled route (`constructor.live`, the held checks and the writes)
ran beside the CC/TT kept rows and the resident models, but not beside CT's all-parent outputs.
At CrI3 24×24 P64 that left 4.55 GB unpriced, and the held and write sections read γ 1.74 / 1.55.
`construct_cross_sector_all` now reserves `sector.decoupled.held.CT` itself and puts it in the CT
budget's upstream. The held and kept rows count every device array a sector holds, not only its
models and signed factors. A round's live row counts all its sector and CT slices. CrI3 6×6 P4:
held and write γ 1.31 / 1.26 → 1.14 / 1.10, eqp unchanged (claim 3764). The rest is the held
section's own reads and error temporaries, about 1.2 GB at P4. No deck change.

## 2026-10-07 — the sector W(τ) right operand is formed once per Σ call; one reduce-scatter for CC's band brackets

On the axis route (replicated pole columns), the W(τ) synthesis transposed and conjugated
the τ-invariant right factor at every τ node. That was a copy of 2.1 / 4.9 / 16.5 GB per node
for CC / TT / CT at CrI3 24×24 P64. `sector_synthesis` now forms the GEMM's right operand
once (`shared_pole_right_operand`). The mixed sector's partner conj(b_X) d b_Yᵀ is built as
conj(b_X conj(d) b_Y†), from the same operands at conjugate weights, as the Green's partner
already is (`right_formed=True`; memory-neutral, since it replaces the factor). The CC
sector's band brackets now reduce in one reduce-scatter (`BandProjector.finish(stacked=True)`),
not one per bracket. eqp is unchanged on CrI3 6×6 and Fe 4³ (claim 3755). No deck change.

## 2026-10-07 — the TT, CT and TC sector Greens contract only each window's live bands

The CC sector's band brackets already built each Green through the active-range GEMM over
their own bands. TT, CT and TC built theirs densely over all bands in every window: 752 at
CrI3 24×24, where a window holds 130–622 live ones. `sector_node` now builds every Green with
`build_G_tau(trim_zero_bands=True)` through the active-range GEMM. On CrI3 6×6 P4 the Σ τ
sweep fell from 34.6 / 45.1 / 42.0 s to 28.5 / 36.4 / 32.0 s; Fe 4³ moved +5 %. eqp moved at
round-off (claim 3739). No deck change.

## 2026-10-07 — the decoupled selection gets its own eigh room and the widest sub-batch that keeps route (c)

The decoupled CC/TT selection capped its eigh room by the face round's room. The
whole-price face step (f6ae798cb) widened CrI3 24×24 P64 from 3 to 10 parents and
left that room near zero, so on main 56a24169c every n = 5184 TT selection eigh ran
on the whole mesh. cuSOLVERMp failed its check silently 37 times and each failure
reran in its own program: TT.all took 1322.9 s against 308.7 s (claim 3706). The
selection now gets the room beside its own row at its last sub-batch, priced from
the shapes with every other parent's panels at the conservative side
(`decoupled_selection_room`). Its sub-batch is the widest at or below the face batch
whose stacks run one whole matrix per rank in that room (`decoupled_selection_width`).
CrI3 P64 replay: 51 GiB room, 10 parents, route (c). The stage widths stay bounded
by the face batch. P4 eqp is unchanged (CrI3 6×6 and Fe 4³). No deck change.

## 2026-10-07 — the decoupled sector route is admitted by its price; the decoupled CT needs route-(c) eighs and runs after the selection panels are released

060697565's rule (decoupled only when parents < ranks) is replaced by a price, in this order:
q-local when every sector is local; else decoupled CC and TT when their stacks rows fit beside
the live set (`decoupled_admission`, priced from the shapes at the conservative sides before
any read, TT first and CC beside TT's held outputs); else face rounds at the stepped width,
warned at nq >= P. CT is priced after CC and TT, at its actual joint side, before its reads.
It decouples only when its stacks row fits and both its metric and Ritz stacks run one whole
matrix per rank (route (c)). Otherwise it runs lever 1's rounds. On main 060697565 the CrI3
24×24 P64 CT stacks (61 × 17408²) ran the whole-mesh scan: CT.all took 1251.5 s at map 0 and
831.4 s at map 1, against 440 s in rounds. After its pencils, the CC and TT selection panels
and spans are released (`release_selection_panels`), since the rounds read only models and
signed factors. The CT eighs then run beside the kept models, so route (c) fits more often
(CrI3 6×6 P4: CT eigh peak 26.90 GB, against 30.29). P4 decks with nq >= 4 take the decoupled
route again wherever it is priced to fit. eqp is unchanged on CrI3 6×6 and Fe 4³ (claim 3682).
No deck change.

## 2026-10-07 — the face batch steps down by its whole price; decoupled stages run at the width that fits beside their stacks

`face_batch_width` stepped down by the room left for the program, which is below
zero whenever the width-proportional bytes outgrow the budget at the first width.
It then jumped to one parent. It now steps by `width * available // aggregate`.
Ni 20³ bispinor at P64, 36 GB, 32 line sites: 17 parents per round, where it ran
1 (CPU replay). CrI3 24×24 at P64, 72 GB: 10 of 61, where main ran 3. On the
decoupled route (parents < ranks) the selection still runs in sub-batches of the
face batch. The CC, TT and CT stage programs, and CT's pencil rounds, run at the
largest width whose program fits beside the sector's stacks row
(`decoupled_width`): CrI3 P64 replay TT 5, CC 10. The receipt prints both widths.
P4 gates against main (claim 3651): CrI3 6×6 and Fe 4³ bispinor eqp within
0.04 µeV over three maps, where both decks keep their batch. A forced-decoupled
CrI3 6×6 matches lever 2 to 0.001 µeV. No deck change.

## 2026-10-07 — the decoupled sector route takes fewer parents than ranks; at nq >= P the face rounds are warned with the q-local need

The decoupled CC/TT/CT route (37baf6124, fdebbbd63, 302a519fe) holds every parent's
selection panels and stage stacks at once, so it now engages only when the parents
are fewer than the ranks (CrI3 24×24 at P64: 61 < 64). At nq >= P the face rounds of
main before 37baf6124 run, as they did. Ni 20³ bispinor at P64, 36 GB, 32 line sites
(641 parents) had taken the decoupled route and run out of memory in
`spole.sector.TT.all`: its TT stacks are 116.6 GB/rank at any sub-batch. The 32 line
sites, not the decoupled code, put TT on the face: TT's local reduction prices
63.9 GB at side 23520 (14 sites: 14880, local). When the sectors run face rounds at
nq >= P, one RuntimeWarning now names the q-local need and the
`memory_per_device_gb` that runs every sector local (Ni: 64, an 80 GB card). The warning
comes from the bank-residence admission, before the bank is built. P4 decks with
nq >= 4 (CrI3 6×6, Fe 4³) take face rounds again. No result moves on a deck whose
route is unchanged. No deck change.

## 2026-10-07 — the CT sector reduces every parent at once; decoupled eigh rows carry their own programs; the free-memory check only warns

Beside the decoupled CC and TT, CT now assembles each face round's joint
pencil (metric, value and the two output panels) from that round's samples at
one compacted span for every parent, writes it in place into one stack, and
runs the joint reduction's keep and output stages over sub-batches with its
two eighs (the metric's keep cut and the Ritz step) once over the stack; the
metric and value stacks are released after the keep stage. At CrI3 24×24
P64 that replaces 21 rounds of two eighs each. Each decoupled eigh row now
reserves its boundary plus its own program (the service's route-(c) program,
or on the whole mesh its vectors stack and workspace) instead of its whole
room, so the P64 replay bounds the measured peaks with priced rows (TT stacks
row 67.08 against 64.91 GB, CC's 61 × 12288 eigh row 35.79 against 35.77).
fdebbbd63's runtime guard could move an eigh stack to the whole mesh when the
measured free pool was short; following the fixed-tile ruling (a route read
from free memory moves results with allocator state) it now gives one
warning naming the shortfall and never changes the route. Results move at
round-off (CT eigh grouping): CrI3 6×6 and Fe 4³ bispinor eqp within
0.06 µeV of main over three SC maps (claim 3618). No deck change.

## 2026-10-07 — decoupled sector eighs priced beside their own stacks; the sector ledger bounds the peak

Each decoupled CC/TT eigh stack now takes its room beside its own boundary
stack instead of the largest one, the derivative panels dW Q are released
once the pencil is formed, and TT reduces before CC, so TT's largest stacks
run beside no other sector's outputs. Each sector's held all-parent outputs
are a ledger row through the CT rounds, each eigh runs under a row of its
boundary and its whole room, and the stacks row carries the stage program and
the output stage, so the constructor's section prices bound the measured pool
peaks (replay of the CrI3 24×24 P64 leg, claims 3545 and 3555: TT.all 64.91,
CC.all 35.77 GB, `tests/test_shared_pole_decoupled_price.py`). A runtime guard
runs a stack the ledger admits one whole matrix per rank on the whole mesh,
with one warning, when the measured free pool (the minimum over processes) is
short of its program. At CrI3 24×24 P64 the 61 × 18432 Y^H G_r Y stack, which
ran as 61 whole-mesh solves (about 376 s per map), gets a 46.2 GB room against
its 43.3 GB route-(c) program (estimate from the shapes). The report now has a
timing and memory row per decoupled sub-stage (`decoupled.selection`,
`.eigh_hvv`, ...). Results are unchanged: CrI3 6×6 and Fe 4³ bispinor eqp
equal to main within 0.05 µeV through map 1 (claim 3574). No deck change.

## 2026-10-06 — the bispinor sector constructor reduces every CC/TT parent at once; its face GEMMs are panel_matmul

The face route of the sector constructor ran CC, TT and CT in rounds of a
few parents (CrI3 24×24 at P64: 21 rounds of 3), one program per round with
its eighs inside, so every round paid its eighs serially. When the face
route has more parents than its batch, CC and TT now reduce every parent at
once: the selection runs in sub-batches, the reduction's stage programs run
over sub-batches and write one stack per array in place, and each eigh runs
once over the stack, one whole matrix per rank where it fits the room.
Stacks that do not fit beside the live set reduce in face rounds with a
warning. CT still runs in rounds, on slices of the held outputs. Every
constructor face GEMM is `distrib_la.panel_matmul` (transposed operands by
one grid-transpose exchange, compiled with XLA's latency-hiding scheduler):
at the P64 sector shapes 12.0–13.4 TF/s per A100 against 3.1–5.4 for the
cuBLASMp face (claim 3425). The metric corrections stay Newton–Schulz (the
paired metric needs one iteration). The constructor receipt names the
decoupled stacks, the eigh room and the largest ‖ZAZ−I‖/√R per sector.
CrI3 24×24 bispinor at P64 (16 × A100-80GB, claim 3545), seconds:

| | map 0 cold, main → now | map 1 warm, main → now |
|---|---|---|
| map wall | 3083.6 → 2249.8 | 3328.3 → 1986.9 |
| W response | 2576.9 → 1763.2 | 2746.1 → 1452.7 |
| CC | 469.3 → 168.0 | 461.5 → 107.1 |
| TT | 1038.5 → 673.3 | 1136.5 → 605.1 |
| CT (21 rounds) | 621.9 → 495.8 | 586.4 → 440.3 |

Against the 176.3 s charge map that is 12.8× cold and 11.3× warm (was
18.9×). There the 61 × 18432 TT eigh does not fit one matrix per rank
beside the stacks (43.3 GB against a 37.6 GB room) and runs as 61
whole-mesh solves, about 376 s of each map. Results move at round-off
(product order): Fe 4³ and CrI3 6×6 bispinor eqp within 0.16 µeV of main
over three SC maps (claims 3477, 3470). No deck change.

## 2026-10-06 — shared-pole W line sites cover every requested state; far-state energies change by design

The shared-pole recipe (`shared_real_pole_v1_r4`, new recipe hash) places W's
line sites over every requested state at E_in ± 5 eV
(`gw.qp_support.requested_reads_ev`, `support_read_pad_ev`; the Σ plan pad
stays 2 eV), so the line reaches every state the run asks for and never less
far than the former fixed ±5 eV window did. Their count is closed form,
⌈(Ω_R − h)·ln(4/ε)/(πh)⌉, held between 18 − n_imag and a 40-sample cap:
Fe 4³ 22, CrI3 6×6 17, NiPS3 12×7 17, Si 4³ / Ni 20³ / Fe 20³ 32 (the cap),
CrI3 24×24 16, CrSBr 20×15 15. A requested state beyond the old window read
an extrapolated W (Si 0.5–0.7 eV, Ni 20³ up to 0.92 eV). The model header
records `support_top_ev` and `support_reads_ev`; the Σ planner warns, never
refuses, when a sample group reads outside them (the coarse semicore windows
always do). `support_delivery_window_ev` is gone; ω_p + 3.5 eV still tops the
imaginary ladder only. Against each deck's dense line ladder (max / RMS meV,
main → this, cold P4; ±2 eV | 2–10 eV | far):
Fe 4³ charge 2.56/1.72 → 0.95/0.54 | 43.1/7.66 → 15.6/3.22 | 274/56.4 → 61.0/9.96;
Fe 4³ bispinor 8.03/4.26 → 2.87/1.44 | 39.3/7.89 → 31.1/6.45 | 850/143 → 82.5/18.4;
CrI3 6×6 (eqp1) 1.67/0.53 → 1.83/0.67 | 10.4/1.81 → 10.3/1.45 | 209/34.5 → 68.2/11.8;
Si 4³ (eqp0) 0.02/0.02 → 0.01/0.01 | 20.0/3.48 → 9.18/1.35 | 514/142 → 56.8/13.8.
The hsuite fixtures, against the same kind of dense ladder: H2⁻ shared-pole
one-shot (eqp0) 27.26/9.01 → 29.46/9.52 | RMS 12.45 → 12.92 (the same reach,
9.99 against 9.93 eV, with the sites moved 0.02–0.5 eV on a 6-centroid toy);
H2⁻ bispinor SC 21.59/5.62 → 21.42/6.06; bcc Na SC 0.57/0.36 → 0.31/0.18 |
5.12 → 4.12 | 35.88/14.26 → 18.05/7.17 meV. The hsuite references of sp_export,
bisp_sc, bse_bisp and na_sc are regenerated. Cold W response: Fe charge
43.4 → 48.9 s, Fe bispinor 145.9 → 140.8 s, CrI3 6×6 140.4 → 155.0 s, Si
37.6 → 58.7 s (14 → 32 line sites, with the response-rule fixes of e08a5c135
and 4662eac7d). A deck that requests protected states to ~40 eV (Fe/Ni 20³)
takes 32 line sites; projected, cold W rises about 7 % on Ni 20³ bispinor and
10–30 % on Fe 20³ charge (map 0). A restart bundle written by the old recipe
refuses by name (`recipe_hash` mismatch; rebuild).

## 2026-10-06 — a response sample near the top of its interval builds instead of refusing

The shared-pole χ response rule refused a line sample high in its own
transition interval (Si 4³: 44.8 + 2.6i eV in 0.69–52.5 eV, `response
exponential fit failed`). Its pencil proposes decaying times. A pole at Re z
then needs coefficients ~e^{Re t·Re z}, and the coefficient-mass gate refuses
every geometry. A group whose pencil fails now takes Gauss–Legendre times on
the imaginary axis, which carry any pole with O(1) coefficient mass. This
happens only when they fit the pencil's capacity (192 nodes; 384 for one
sample), with the same projection and gates; otherwise the group is halved
as before. A group whose pencil succeeds is unchanged: eqp is bitwise on main's
Si 4³, Fe 4³ charge and CrI3 6×6 decks (cold P4).

Measured on the WSUPPORT ±5 eV Si deck (top site 44.8 eV, 40 samples): it now
builds. The rule takes 9.8 s with 52 pencil nodes plus 140 closed-form nodes
(sampled error 3e-13), and W takes 59.7 s. Against the same-reach reference with
twice the line sites, eqp0 within ±2 eV of μ agrees to 0.02 meV. Within ±10 eV
the RMS is 2.2 meV (max 16 meV), less than the ±2 eV placement's 3.5 meV RMS
against the same reference; the remainder is line-site density. No deck change.

## 2026-10-06 — the cuSOLVERMp eigh runs at a block of 128–256 and reads its info

A cuSOLVERMp eigh ran at the largest divisor of n/p up to 256, which is
1–9 when n/p is prime or a small multiple of one: the CrI3 6×6 TT side at
2634 current points, n 7908 on 2×2, ran at block 6. Such a tile is now
padded to the smallest edge with a divisor in [128, 256] (7908 → 7912,
block 172; at most 127 rows per rank), with sentinel rows that leave the
result. cuSOLVERMp 0.9.1 returns vectors that are not orthonormal inside a
near-zero cluster with status 0 and info 0 at some (matrix, block, shift):
at n 7908 on rank-deficient PSD matrices it failed at block 6 shifted and
at one tile per rank, never at blocks 86–247, and the retry at one tile per
rank needed a 2.0 GB workspace against 0.25 GB at block 6. That retry is
gone; the chain is unshifted, a smaller block, shifted and
re-orthonormalized, shifted at the smaller block. Measured at P4: one eigh
at n 7908 10.8 → 1.5 s (n 7894, n/p prime: 290 → 1.5 s); the block-table
sizes 778–16832 are 1.3–37× faster and the sizes with a block of 128 or more
are unchanged (claims 3314, 3315). Results move at round-off only where the
layout changes; the 1316-point CrI3 6×6 deck has none (eqp identical). The
cuSOLVERMp handlers now zero and read info after every syevd, potrf,
potrs, getrf and getrs and return a nonzero value as a named error, and the
workspace query prices the block the solve runs at; both take effect with
the next bundle (B10 prices one tile per rank at the solve side, an
over-price). No deck change.

## 2026-10-06 — the response rule skips the shared fit that cannot pass

The shared-pole χ response rule (`minimax.response_group_rules`) sizes its
Hankel pencil by the narrowest sample height and 8× the farthest line site.
When the 1 eV imaginary sample and line sites far above the gap fall in
different halves of the sample set, the whole set's pencil is wider than
either half's (Si 4³ with line sites to 41.78 eV: 1540 against 800). Every
such set we measured failed its shared fit after all six geometries, because
the coefficient-mass gate refuses the far line poles. The set was then halved.
That wasted attempt was most of the cliff seen above 22 samples. The halves
are now built directly. Rules, eqp and `sigma_diag.dat` are bitwise on every
leg (cold P4, Si 4³, main against this change). Rule build and cold W, in s:

| samples, top line site | rule build | W |
|---|---|---|
| 14 or 22 to 9.44 eV (main) | 0.8 / 0.9, unchanged | 34.3 / 37.7–38.9 |
| 40 to 9.44 eV | 1.4, unchanged | 48.4–48.6 |
| 32 to 23.9 eV | 13.4 → 3.8 | 56.1 → 46.6 |
| 40 to 41.78 eV (the WSUPPORT Si placement) | 26.1 → 8.5 | 74.7 → 57.2 |

The sample count does not set the build time; the reach does. No deck change.

## 2026-10-06 — the face batch is priced from the shapes, never compiled to be measured

A face-route shared-pole construction (the CC/TT/CT sectors and the scalar
constructor) admitted its parent batch by compiling each round's whole-chain
program at the conservative side, reading its size, and then compiling the
program that runs. On CrI3 24×24 at P64 that was six programs that never ran,
187 s of every cold map 0 (CC 71 s, TT 80 s, CT 37 s); on the Fe 4³ and CrI3
6×6 face decks at P4, 90–200 s per process. The batch is now admitted by the
byte model that admits every local round (`shared_pole_byte_terms`, `b/P`
copies per rank), with the eigh and matmul workspace quoted once beside it;
the constructor line reads `face program N GB/rank priced from the shapes`
and the receipt key is `program_bytes_per_rank`. The price is an upper bound
on the compiled figure: two tiled copies of the local round's term, where the
CrI3 24×24 P64 receipt's compiled programs held 1.57–1.63 of one, so the
admission lands on the same batch as before (3 of 61 parents at P64, replayed
by `tests/test_shared_pole_face_price.py`; 1 on the P4 face decks). The
programs that run and their shapes are unchanged. Measured cold at P4
(every leg on a fresh cache): CrI3 6×6 forced-face SC, map-0 compile 399.4 →
235.7 s, W 497.2 → 324.0 s, eqp bitwise at maps 0–2 and maps 1–2 unchanged;
Fe 4³ face deck at a 2 GB budget, map-0 compile 325.5 → 257.1 s, W 405.5 →
330.6 s, eqp bitwise at map 0. Known cost on a memory-starved deck: the price
is 0.25–0.4 GB above the compiled figure, so the eigh room beside the admitted
batch is that much smaller; on the 2 GB Fe deck map 1 compiled the per-parent
CT face-rerun programs two to three times at the same shapes (29 more
compiles, 42.4 vs 13.7 s; W 71.9 → 104.0 s) and maps 1–2 moved by ≤ 0.09 µeV;
the held widths, growth events and program shapes are identical on both arms.
No deck change.

## 2026-10-06 — kmeans names the pool rank and its stop; a budget-stopped selection warns

The centroid header's `achieved numerical rank=N` is now `pool rank=N of W
kept pivots on a M-point pool` with the stop: `pool spent` (no unpicked
candidate's residual is above the floor, so N is the pool's rank, a lower
bound on the pair-set rank) or `point budget` (the pool still held
directions, so the rank is not measured). The second case warns
`CentroidRankNotEstablished` in `kmeans.out`: the Fe 4³ deck's 312 points
read "rank 312" while its pair set ranks above 1304. A charge selection also
prints the rank law's N_μ for 1 meV RMS Σ_x and written/estimate. Centroid
coordinates do not change. Scripts that parse `achieved numerical rank=`
must read `pool rank=`.

## 2026-10-06 — the bispinor Σ runs the mixed pair CT + TC as one fused window

`bispinor_gw = full_shared_pole` Σ made four τ sweeps per map, one per
sector pair; the two mixed sweeps each synthesized a W pair that is the
other's transposed pair. The mixed pair now runs as one window: CT's pair
synthesized once per τ node, TC's pair by one exchange, both contractions in
the same loop trip. Results move at round-off only: Fe 4³ bispinor SC maps
0–2 max |Δeqp| 0.13 µeV (map 0 bitwise), CrI3 6×6 one-shot bitwise (claim
3289). Measured cold on Ni 20³ P64 (claim 3294): map 0 3126 → 3065 s, Σ τ
1803 → 1731 s, the mixed sweeps 829 → 760 s, −69 s per map (−2 % of the
map, −4 % of Σ τ). The stage log shows three sector sweeps (CC, TT, CT+TC)
instead of four. No deck change.

## 2026-10-05 — the SC stop test reads each sorted pair on its own; `sc_exact_degeneracy_tol_ev` is retired

The SC criterion is max over k and trusted labels of |E_out − E_in| at the
label's own sorted column (the spectral distance on the trusted columns). It
no longer averages input and output over a label block first. Inside an exact
multiplet the sorted energies are equal, so nothing changes there; over a
block that Σ splits, the mean hid the change of the splitting (a pair whose
splitting moved 0 → 10 meV at a fixed centre read 0 and CONVERGED). The state
identity now matches single projector overlaps, with no multiplet grouping,
so the eqp `SC_identity` comments carry one row per label (same keys). That
also removes the spurious blocks formed when the multiplet means made the
reference spectrum non-monotonic. Replayed on nine existing SC histories
(CrI3 24×24, CrSBr 16×12 and 20×15, AgI, Ni 20³, Fe 20³ and 4³), the stop map
does not move; the largest within-block motion the mean removed at a stop map
was 0.023 meV. A deck that sets `sc_exact_degeneracy_tol_ev` refuses by name:
delete the line.
## 2026-10-05 — shared-pole SC rounds write their model files directly

A model round whose K extent is already held (SC map 2 on; map 1 records
the extent at its finalize) writes its batch into the final
`factor`/`poles2_ry2` datasets at once; map 0, map 1 and a one-shot still
stage. The file holds the same bytes, census, digest and commit (Fe 4³
bispinor P4, file route: eqp bitwise over maps 0–2; a job killed mid-write
leaves `finalized = false` and no commit, refused by name), and the scalar
constructor's one-batch write is the same direct write. The header carries
`direct_extent`; `staging_payload_bytes` and `peak_payload_bytes` are gone.
No deck change.

## 2026-10-05 — `full_shared_pole` on a WFN with time reversal refuses at driver entry

`bispinor_gw = full_shared_pole` on a WFN with time reversal now refuses at
driver entry, on the measured symmetry and before any basis, ζ or W work
(`GATE full_shared_pole_trs`), instead of in the sector store after the
screening stage. With time reversal the static charge–current coupling
vanishes and the mixed CT/TC terms are purely dynamical; use
`bispinor_gw = bare_transverse` (docs/theory/bispinor-gw.md §10).

## 2026-10-05 — the one-shot names every band it reads at Σ(ω = 0)

The one-shot's incomplete-grid warning now lists each band read at
Σ(ω = 0), with its DFT energy range about E_F and its k count, instead of
only a cell count. Those are the requested states the sampled grid does not
reach: deep states the W model treats as inactive. No number moves. To
sample them, widen `sigma_omega_min_ev` or add a `sigma_omega_patches_ev`
window.

## 2026-10-05 — kmeans reads the deck's `wfn_file`

`centroid.kmeans_cli -i DECK` opens the deck's `wfn_file`, relative to the
deck as in `kin_ion_io` and `gw_jax`, instead of `./WFN.h5` in the working
directory. Without `-i` it still reads `./WFN.h5`. The centroid header's
`source wavefunctions` line names the file read. Centroids do not move.
A deck whose `wfn_file` is unset now reads `WFN.h5` beside the deck, not in
the working directory.

## 2026-10-05 — the V_NL spin-orbit mode is read from QE's `<spinorbit>`

`kin_ion_io`, the dipole driver and every other V_NL user read j-resolved
versus j-averaged projectors from `<spinorbit>` in the QE
`data-file-schema.xml` that authenticates the WFN (`WfnLoader.spinorbit`).
The measurement against degenerate multiplets is gone. It refused every
centrosymmetric crystal with time reversal whose double group has no irrep
larger than two (D3d: Bi, 2H-PbI2); those now run. Fully relativistic
pseudopotentials on an nspinor = 2 WFN with no authenticating schema now
refuse by name: put the NSCF `data-file-schema.xml` (or its `.save`) beside
`WFN.h5`, in its directory or the two above it. Results that ran before do
not move. No deck change.

## 2026-10-05 — the compile cache is JAX's own, plus the namespace, the compile agreement and the receipt

`common/jax_compile_cache.py` no longer freezes an all-rank agreed entry
set at startup, vetoes other lookups, makes the key process-invariant,
canonicalizes `jit__multi_slice`, writes entries atomically or prefetches
them. Those layers guarded XLA:GPU's collective autotuner, which hung when
one rank hit the cache and skipped the exchange (2026-07-27, jax 0.7.0);
the runtime has run at `xla_gpu_autotune_level=0` since 2026-09-03, so a
divergent hit/miss pattern now costs the missing rank one compile. What
remains: the default location and namespace, threshold 0, JAX's per-fusion
XLA caches off at P > 1, rank 0's age pruner, the cross-rank compile
agreement (a rank lowering a different program is still refused by name),
the device-fit gate and the compile receipt. 2560 → about 1100 lines.
- The compile agreement's protocol changed with it. Every rank now
  publishes its module fingerprint on every compile request, a cache hit
  included, and only a rank that actually compiles reads the others' (rank
  0 reads all, a peer reads rank 0's). Before, only compiling ranks
  published and rank 0 collected a verdict, so a rank that hit an entry
  its peers were compiling left them waiting forever (measured on CPU at
  P2: 300 s, named every 60 s); the frozen agreed set existed to make
  that impossible. An asymmetric hit now costs nothing; a warm run pays
  one KV set per program per rank.
- Environment: `LORRAX_JAX_CACHE_MULTIPROCESS`, `_INVARIANT_KEY`,
  `_SHARD_SLICE`, `_AGREE_TIMEOUT_S`, `_STRICT`, `_PREFETCH`,
  `_PREFETCH_THREADS`, `_FORCE_DIVERGE`, `_NO_AGREE` and `_KEYDUMP` are
  gone and ignored. `ISDF_JAX_CACHE_DIR`, `LORRAX_JAX_COMPILE_AGREEMENT`
  and `LORRAX_JAX_COMPILE_AGREE_TIMEOUT_S` stay.
- Log: the `ARMED` line and the per-rank summary keep their shape without
  the agreed/vetoed/prefetch fields. On CPU runs peers may compile a few
  rank-local programs process 0 has cached (the key is process-invariant
  on GPU only); that is a compile, not a hang.
- Results do not move. A run that sets `xla_gpu_autotune_level` above 0
  with a shared cache is back in the 2026-07 regime and must set
  `ISDF_JAX_CACHE_DIR=""`.

## 2026-10-05 — every SC map and driver prints its compile receipt

Rank 0 prints one line per SC map and one at the end of each driver, in
production, without a debug flag and before a run can be killed:
`  [compile-cache] SC map N compile: real R (S s), cache hits H, uncacheable U [names]`
(`stage <driver module> compile: ...` at a driver's end). The counts are since
the previous receipt: R real XLA compiles and their S seconds, H persistent
cache hits, U compiles of programs with host callbacks, which JAX's cache
never stores, named. Each such program also prints
`UNCACHEABLE <module>: N host callbacks (S s)` once. A map past the second
with R > 0, or any U > 0, compiles work it already did; the sandbox's
`tools/parse_compile_receipts.py` flags both. The SC map line is in the
report and stdout; the driver line, the UNCACHEABLE warning and every other
`[compile-cache]` notice go to stderr (production stdout used to swallow
them). No result moves and nothing needs to change. Fe 4³ bispinor at P4,
cold: SC map 2 still compiles 23 programs (4.9 s); a warm second process
compiles 12 (1.5 s) at map 0 and none after, one of them the uncacheable ζ
factor (`isdf/core.py`).
## 2026-10-05 — htransform band operators come from the WFN's own k-points

`--color` and `--velocity` no longer load the full-BZ ψ of the fitted
window. σ and the atomic-character projectors are formed at the WFN's own
k-points in one `common.mtxel_sweep` pass, as the dipole driver forms the
velocity, and `symmetry_maps` unfolds them: σ as an axial time-odd vector,
a projector as an invariant scalar. Per rank ψ costs N_k,irr·nb·n_s·N_G·16/P
bytes. CrI3 24×24 with 208 bands refused at 155 GB per device at P4; it now
runs on one A100-80GB node at a 23 GiB peak. The orbital total sums every
stored band in one mapped call. Its value is unchanged: CrI3 bispinor and
charge to 1e-12 μB, Fe 4³ to 12 digits, path operators to 5e-11. The
`htransform.out` orbital block loses the `1/E_ceiling extrapolation` and
`spin + orbital extrapolated` lines; the intercept moved with the fit range.
No deck change.

## 2026-10-05 — the compile-cache namespace no longer names the source commit

The default persistent-cache directory is now
`$SCRATCH/.cache/lorrax/jax_compile/jax<v>-jaxlib<v>_<ffi bundle>_k1/np{P}`.
The LORRAX commit or release left the namespace, so a commit or a `lorrax_A`
republish that leaves a program's HLO unchanged reuses its entry instead of
starting cold. JAX's key still covers the module, compile options, XLA flags
and backend; the FFI bundle and a hand-bumped key schema (`_KEY_SCHEMA`) cover
the rest. Rank 0 touches the entries it uses and prunes, in a background
thread, every entry unused for 7 days. No results move. The first run after
this change starts cold once; the old per-commit namespaces are retired by
the existing pruner after 7 days unused.

## 2026-10-05 — distrib_la programs are keyed on what enters the HLO; sqrt_v and the Dyson programs are checked programs; a traced stack route is priced, not compiled

Compile audit rows 4, 5 and 6. A distrib_la program key (`_program_key`,
`_stack_bytes`, `_reshard_stack_program`, `Plan._program`,
`_scan_over_single`) held the call site and the plan's byte budget, so one
eigh stack compiled once per calling line and per room (CrI3 24×24 P64:
8 extra compiles, 22 s cold per process). The keys now hold op, mesh,
shape, dtype, rounds and phase only; the refusal still names its site, on
the host. `checked()` and `checked_eigh()` no longer take `site=`.
response_bank's `sqrt_v` and its Dyson, slope and moment programs are
`distrib_la.checked_program`s (`checked_program` takes `in_shardings`, and
an AOT caller hands its executable's `(out, status)` to `call.finish`), so
they hold no host callback and the persistent cache stores them; a checked
solve traced in a bare jit outside any `checked_program` still prints its
notices through one callback, and now warns once per site (`UNCACHEABLE`).
A traced route-(c) decision no longer compiles a stack program to measure
it (that program was thrown away: 14 compiles, about 40 s cold on CrI3
P64); it prices each candidate from the shapes with the planners' formula
(`_stack_price`), and the route line says "priced from the shapes". An
eager call still compiles its first-attempt program once and runs it. A
plan's room (`budget_bytes`) is a decision input and no longer part of a
plan's identity: the face plans and the programs built on them are shared
across rooms, so a room that crosses a GiB between SC maps no longer
recompiles the face programs; a shape's face program keeps the stack route
it was traced with. No deck or environment change; results do not move.

## 2026-10-05 — Shared-pole rounds have one shape per run: fixed-width rounds and a pencil at capacity

- **One round schedule.** The local, face, scalar and CT-rerun routes all
  take fixed-width rounds from `shared_pole_local.parent_rounds`: a short
  last round repeats its last real parent and every consumer reads the
  leading `real` slots. Before, the face route cut a ragged last batch
  (CrI3 24×24: 61 parents at width 3 left a width-1 tail that recompiled
  every CC, TT, CT and held program, about 120 s cold and 30 s warm per
  leg). A padded tail costs no measurable time: the width-1 round already
  took as long as a width-3 round.
- **Round inputs on recipe carriers; the pencil extent grows in map 0 and
  holds from map 1.** Every state panel and the infinity block are padded
  to their recipe carriers (`recipe_panel_widths`, `recipe_infinity_width`:
  the imaginary width, the line cap, Q's carrier for a partner, the infinity
  width; a selection closed over a multiplet past its width keeps its own,
  wider carrier, as a metal's M1 directions do), so a round program's inputs
  have one shape. The pencil extent (`round_tables`) is the carrier of the
  largest selection, grow-only across the model's rounds and SC maps: the
  partner (TRS-odd) counts have no bound below the panels' capacity, and a
  pencil at that capacity costs the eigh (capacity / high water)^3 on every
  round of every map (CrI3 24×24 P64: CC 20800 against 17472, TT 32000
  against 24832, +616 s per map, measured), so the extent is discovered in
  map 0 (CrI3: eight growths, about 250 s of recompiles once per chain) and
  held from map 1. The local ordered round's kept span sits on its pole
  budget, as the face round's does; the grown-then-rerun Ritz rung is gone.
- **Face batch sized from the held sides (BISPPERF lever 1): measured and
  declined.** Sizing the bispinor face batch at the sides held after map 0
  took CrI3 24×24 P64 from batch 3 to 5, but a round's cost scales with its
  parents (per parent CC 11.6 vs 12.3, TT 22.8 vs 23.7, CT 7.4 vs 10.3 s), so
  the sector stage broke even (about 2890 against 2931 s) while every process
  paid a second sizing pass (+30–36 s on P4 face decks). The batch is sized
  once per map at the recipe bound, as before, and the decision now prints
  when it is made: `Shared-pole face batch: N parent(s) of nq per round;
  sized in X s` (the constructor's summary line came only after the stage).
- **Deleted.** `grow_round`, the carrier/extent/Ritz histories (the session's
  `shared_pole_carriers` now holds only the writer widths), the reduction
  preview/admit closures and the "selection exceeds its held carrier" log
  lines. Results: eqp moves at most at the eigh's bit level where the map-0
  side changed; later maps are bitwise.

## 2026-10-05 — kmeans sizes from the shapes; a CT face fallback; the checkpoint digest splits over ranks

- **kmeans.** The candidate Gram's k batches and square tiles, and the
  feature metric's band and k chunks, are sized from the shapes and the
  fixed 1 GiB tile (`runtime.tiles`), never from `memory_per_device_gb` or
  the card. Before, the Gram used one full block when it fit a quarter of
  the budget, else budget-sized tiles from a compiled-peak ladder, and the k
  batches took half the budget. So the summation order, and through a
  near-tie pivot the centroid set, could change with the budget. On Fe 4³
  and CrI3 6×6 the selected sets equal main's bit for bit at two budgets
  each, so Σ and eqp do not move there. CrI3 6×6 selection wall: +2 %. The
  Gram build is 17 % faster; the feature metric is 0.7 s slower, because it
  now takes two band chunks. `LORRAX_GRAM_COL_BLOCK` still pins a tile
  width. A price over the budget warns and runs.
- **Bispinor sectors.** With at least as many parents as ranks the sector
  rounds run local. A local round whose CT pencil does not fit at its actual
  spans now warns and reruns its parents as face batches, CC and TT
  included. Before, it ran over budget and failed in compile or OOM after CC
  and TT. Rounds that fit are unchanged. A forced fallback on Fe 4³
  bispinor gives eqp bitwise equal to the local and the face routes.
- **SC and one-shot Σ checkpoints.** Every rank hashes 1/P of the cube
  slices from the closed file, and the slice digests combine by XOR. Before,
  rank 0 re-read and hashed the whole file after every write and on resume.
  At CrI3 24×24 carry size (1.77 GB, P4) the digest takes 0.5 s instead of
  1.8 s. The digest is the same at any P, so a P1 continuation of a P4
  checkpoint authenticates. A checkpoint written before this change does not
  (`cube digest of an older format`): the SC trajectory restarts from its
  warm seed with a warning, and a one-shot Σ sweep is recomputed once.
- Decks do not change.

## 2026-10-05 — the bispinor carrier is named the normalized RKB lift

The four-component carrier $[I;X](I+X^\dagger X)^{-1/2}\psi_L$ was called
the "isometric" lift. It is now the normalized restricted-kinetic-balance
(RKB) lift, after the RKB and X2C/NESC literature
(docs/theory/bispinor-gw.md §1). The construction and every number are
unchanged.
- Code: `NORMALIZED_RKB_LIFT = "normalized_rkb"` and
  `NORMALIZED_RKB_FOUR_CURRENT_REPRESENTATION` replace the `ISOMETRIC_*`
  names. No deck key changes.
- Files. The charge-carrier stamp is now `normalized_rkb_four_current_v1`.
  A bispinor restart bundle stamped `isometric_kinetic_balance_four_current_v1`
  refuses with `GATE restart_bispinor_charge_carrier`; set `restart = false`
  once. A bispinor SC checkpoint with the old stamp does not continue: the
  trajectory restarts from the seed, with a warning. The lift provenance
  that ζ and Hall artifacts carry is the formula string, which is unchanged.

## 2026-10-05 — checked-solve programs are stored by the persistent compile cache

distrib_la's checked eigh and LU solves printed their retry notices and their
refusal through `jax.debug.callback`, and JAX never writes a program with a
host callback to its persistent cache, so every program holding a checked
solve (the shared-pole round programs, `checked_program`'s `jit(run)`, and
their sizing compiles) compiled again in each process: 274.6 s of rank-0
compile per CrI3 24×24 P64 chain leg. The program now returns every attempt's
check errors beside its failure flag, and the host prints the same notices and
the named `GATE distrib_la_result_check` refusal after the call. Fe 4³
bispinor on the face route at P4, second process on a warm cache: 32 → 12
compiles, 209 → 1.5 s of compile, W response 306 → 101 s; eqp bitwise. A
checked solve traced in a caller's own jit, scan or cond (outside any
`checked_program`) still names its refusal, through one host callback in that
program. The shifted retry's Newton–Schulz step count is no longer printed.

## 2026-10-05 — bispinor charge and current on one normalized four-component carrier

`bispinor = true` now evaluates every vertex on the normalized RKB lift
$\Psi=[I;X](I+X^\dagger X)^{-1/2}\psi_L$, $X=(\alpha_{\rm FS}/2)\boldsymbol\sigma\cdot\mathbf p$
(owner ruling; docs/theory/bispinor-gw.md §1, decisions.md). The direct
field ($V_H[\rho]+\boldsymbol\alpha\cdot\mathbf A[J]$), the charge ζ,
$\Pi_{00}$, $W_C$ and the CC Σ use it, and so do the current vertices
(α·A, the current ζ, CT/TT). The lift is an isometry, so $\int\rho$ is the
electron count. Before, every vertex took the raw lift $[\psi_L;X\psi_L]$,
whose norm exceeds one by $(\alpha^2/4)\langle p^2\rangle$; its excess
small-component charge added to the direct field a V_H-gauge term that grows
with the vacuum. A large-block charge $(\psi_L,0)$ was on main for one day
(2026-10-04) and is gone.
- Results move against the raw lift (sandbox claims 3228, 3238, 3251–3253,
  P4). CrI3 6×6 slab one-shot: the direct field drops by 110–320 meV per
  state. It now differs from the two-spinor V_H by −0.2 to −7.9 meV per state
  (group means Cr 3s −5.3, VBM −2.3, CBM −2.5). The CC Σ differs from the
  charge Σ by +0.55 meV mean (≤ 2.1). The Γ gap, bispinor − charge, is
  +0.26 meV (bare_transverse) and +0.92 meV (full_shared_pole, head off),
  against +22.1 and +23.0 meV on the raw lift. Fe 4³, one map from the
  charge fixed point: the direct field within 2 eV of E_F drops by about
  32 meV mean; m_orb (0.0303 μB at 35 bands) is within 0.2 % of the raw-lift
  value.
- Files. Charge and current ζ fitted on another lift refit. A bispinor
  restart bundle written before this change refuses with
  `GATE restart_bispinor_charge_carrier`; set `restart = false` once. A
  bispinor SC checkpoint (`sc_seed/sc_checkpoint.h5`) from before this
  change does not continue: the trajectory restarts from the seed, with a
  warning. A static-gauge Hall artifact from before this change refuses on
  its lift stamp; rebuild it with `--static-gauge-hall-only`.
  kin_ion.h5 and dipole.h5 are reused unchanged; the scalar Γ-head dipole
  stays on the raw lift (decisions.md, "Not yet conforming").
- Decks: no change.

## 2026-10-05 — the ordered partner state keeps one carrier, so constructor rounds recompile less

At an imaginary (or Re z = 0) support of an ordered shared-pole round, the
conjugate state holds the part of O = W Q outside span(Q). It used to be
dropped when no parent kept a partner direction, and otherwise to carry its
own count's width, so a round's state list moved parent by parent and the TT
and CT round programs compiled again in nearly every round
(`shared_pole_directions._round_partner_directions`). The state is now always
present on Q's own carrier (count 0 when empty); its live columns are
unchanged. Fe 4³ bispinor SC maps 0–2 on the face route at P4 (1.5 GB budget,
compile cache off): 169 → 101 round-program compiles, W construction 610 →
347 s. eqp: bitwise at map 0, ≤ 0.13 µeV at maps 1–2 (inert padding columns
in the pencil); the CrI3 6×6 bispinor one-shot is bitwise. No deck change.

## 2026-10-05 — W-bank file tier warns instead of refusing; the pole-budget cut keeps whole multiplets

Robustness-review fixes (reports A1–A3, #7, #9, #12, #14, #15):
- A W-bank field created after the bank's initialization (line panels, photon
  contact fields) that the disk or quota refuses is held in host memory with
  a warning in the report's WARNINGS block, so the map runs on. Before, its first write refused the map on
  every rank (`GATE streamed_bank_capacity … refused at creation`), after the
  whole χ stream.
- The per-rank streamed tier pads records to the page size, not the
  filesystem block (16 MiB on CFS/GPFS). Where `fallocate` works, reserved
  bytes are no longer promised a second time, so a bank no longer needs
  twice its size free. The Lustre quota room is taken under the hard limit,
  so the soft-limit grace period no longer refuses every file-tier bank.
- `sigma_w_accuracy = relaxed` on a bispinor deck whose sectors take the face
  route ran into a TypeError at batch admission; it now runs.
- A complete shared-pole bank built against another bare-V digest (another
  P, or code before this change) is rebuilt instead of refused with
  `GATE shared_pole_output: … built with another bare V`.
- The pole-budget keep cut no longer splits a degenerate multiplet at its
  edge: a member tied to its neighbour below the cut leaves with it, so K can
  be a few below the budget at a high-symmetry parent. Results move only
  where the budget binds at a multiplet; W(q) then keeps its little-group
  symmetry and no longer depends on the eigh route (P, card memory).
  Fe 4³ scalar and bispinor SC (maps 0–2, local and forced face) and the
  CrI3 6×6 one-shots (charge, charge with band extrapolation, bispinor) at
  P4: eqp bitwise to main.
- The one-shot Σ checkpoint identity includes the band-bracket plan, so a
  restart after a change of `band_extrapolation_bracket_scheme` recomputes.
- Test harness: `tests/conftest.py` seals the source closure once per test
  process. Under `lx test` a test that imported `gw.*` before `file_io` took
  the release's services, and a later import in the same xdist worker raised
  `SourceClosureError` (6 of 16 focused tests failed, other lanes' included).
  Tests need no bootstrap import of their own.
- Decks do not change.

## 2026-10-05 — review fixes: CPU row passes, the DFT+U static head, 2c TRS for WFN_qp, mode-7 scratch

- CPU (host mesh) runs whose direct response stream needs two or more row
  windows no longer die with a TypeError under x64 in the contour block
  accumulate. GPU runs are unchanged.
- The one-shot static head refuses a dipole.h5 without i[r, V_U] on a DFT+U
  deck (`GATE static_head_dipole_operator`), whatever `LORRAX_SANITY` says;
  it had warned, or with `LORRAX_SANITY=0` skipped the check. Regenerate such
  a dipole.h5 with `python -m psp.get_dipole_mtxels -i <deck>`.
- 2c TRS: a `WFN_qp.h5` also finds the QE schema from its source WFN
  (`qp_wfn_source`), so it gets its source's verdict; a nonmagnetic SOC
  `WFN_qp.h5` read by BSE, htransform or a restart had taken TRS off.
- 2c TRS: this narrows the 2026-10-04 guard. The wavefunction guard refuses
  (`GATE trs_qe_nonmagnetic_wfn_consistent`) only at a QE moment of exactly
  0; at 0 < m < 1e-4 μB/cell, or on a spatial-pair failure, TRS goes off
  with a warning. The SCF moment is read from the selected schema only (an
  alias schema only when its density file matches byte for byte), and not at
  all when a t_rev row decides.
- Files. A nonmagnetic SOC `WFN_qp.h5` written before 09e4bcd81 (2026-09-30)
  that stores both k and −k now refuses with `GATE
  trs_qe_nonmagnetic_wfn_consistent`: its −k rows were rotated in the wrong
  gauge (raw-pair residual 3.2e-2 on a MoS2 3×3 file). Before, it took TRS
  off without saying why. Regenerate it with the current gwjax.
- The Σ τ window prices mode 7's run-time split-arm scratch (up to 1 GiB per
  call on bulk grids from about 14³) and counts it in its overlap check and
  ledger reservation, which warns when over budget. Grids below about 14³
  are unchanged.
- Fe 4³ bispinor SC and Si 4³ SOC SC (maps 0–2) and a MoS2 3×3 SOC one-shot,
  P4: eqp bitwise against main 88785c2e8.
- Decks do not change.

## 2026-10-04 — compiled sizes agree over ranks; PT magnets stream their −q rows

- `check_chunk` (the ζ μ batch, the response direct stream) and the Σ τ
  window's overlap test and reservation compare the largest compiled figure
  any rank read. A figure that differed across ranks could split the ranks
  between a recompile and an all-gather and hang. No result moves.
- `step_up` clamps a step to its top before the caller's snap. A route-G
  plane step-up whose implied block count passed the plane-group count raised
  ZeroDivisionError; it now steps to one plane group per block and warns.
- The shared-pole −q mirror uses inversion only where SymMaps authorizes it
  as a unitary row. A magnet whose inversion holds only with time reversal
  (QE t_rev = 1, e.g. a PT-symmetric antiferromagnet) now streams its −q rows;
  before, W at the line samples was built from a non-symmetry. Nonmagnetic
  decks and magnets with unitary inversion (Fe, Ni, CrI3 FM) are unchanged.
- The sector CT reduction's final eigh reads the Hermitian part of Y^H V Y.
  On the face route the whole-mesh eigh's result check could refuse it near
  the keep cut. Fe 4³ bispinor P4 maps 0–2, local and forced face: map 0
  eqp bitwise, CC/TT models bitwise, CT W(z) within 2e-11; through the SC
  maps eqp moves at most 0.14 µeV.
- The bare-moment Hermiticity gate and `distrib_la.leading_eigenvectors`'
  face check no longer replicate their stacks on every rank.
- Decks do not change.

## 2026-10-04 — the response bank's dense stages fill the mesh under `linalg = local`

The sample Dyson, line selection and moment Dyson of the shared-pole bank run
one whole matrix per rank under `linalg = local`, but their q span was sized
as faces in the fixed tile, so a span of w < P parents left P − w ranks idle.
The span is now at least P parents per round, balanced over the rounds
(`response_bank.parent_span`): CrI3 24×24 at P64 goes from spans of 10 and
moment batches of 2 to one round of 61; Ni 20³ from spans of 72 and moment
batches of 21 to 11 rounds of 59. A streamed sample span holds about
6·16·n² bytes per rank (6.6 GB at CrI3 24×24); the ledger warns if that is
over budget. `linalg = distributed` keeps the face-tile span.
`subtile_stream.plan_windows` keeps its pass count and takes the shortest
orbit-aligned window that keeps it, so the streamed χ bank stores fewer dead
rows. No deck change. eqp is bitwise against main on Fe 4³ bispinor SC maps 0–2
and the CrI3 6×6 bispinor one-shot at P4, also with a test tile that makes
both changes active.

## 2026-10-04 — robustness fixes: HL head, mini-BZ shares, SC resume, orbital totals

- `hl_ppm`: the head plasma frequency counts the WFN's electrons
  ($\omega_p^2 = 16\pi N_e/V$). It counted occupied bands, half of $N_e$ on a
  scalar WFN (Si: 11.7 eV instead of 16.6 eV), so scalar HL-PPM head Σ moves.
- The S-tensor q→0 head average splits its draw evenly over ranks. At
  P = 9, 25, 36, 49, 100 it refused (`GATE cross_rank_compile_agreement`);
  P = 4, 16, 64 are unchanged.
- `sc_mixing` other than 1.0 refuses (`GATE sc_mixing_retired`): nothing has
  read it since the linear path was deleted. Delete the key from decks that
  set it. `LORRAX_SC_MIXING` is gone.
- The SC checkpoint's deck digest reads lines as the deck parser does (`=`
  or `:`, case, `#` comments). A checkpoint written before this change does
  not match it: the run warns and starts again from the seed.
- htransform `--velocity` totals: a T = 0 total (no `occ_smearing_width_ry`)
  with overlapping bands or a fractional count refuses
  (`GATE orbital_totals_t0_gap`), and the band ceilings start above every
  band within 10 kT of μ, so the 1/E_ceiling extrapolation moves wherever
  0.6·nb lay inside the occupied set (CrI3 charge, nb 208). With no width,
  the coarse and grid E_F use the rounded electron count, and an occupied top
  returned band refuses (`GATE htransform_moments_window`) instead of
  warning.
- `kin_ion` stamps the `nval`/`ncond` GW resolves (`number_bands_protected`);
  its datasets are unchanged.

## 2026-10-04 — htransform's orbital moment is the modern-theory one

`--color orbital` colors the path bands by the wavepacket moment
$\hat n\cdot\mathbf m_n$ of a stored velocity of the WFN's own states, and
`--velocity FILE` prints the coarse per-cell orbital moment (stored band
ceiling and 1/E_ceiling extrapolation), the spin moment with g_e = 2.00232
and their sums to `htransform.out` (docs/how-to/htransform-and-exciton-bands.md
§6). CrI3 24×24 at the 750-band ceiling: DFT −0.012136, bispinor QSGW
−0.030802 μB/cell along QE's magnetization axis, parallel to the spin moment.
- The atomic-sphere ⟨L⟩ (per-atom L rows, its `moments.txt` lines and the
  `L_*` names in `band_operators_path.npz`) is gone; `--moments-grid` sums
  spin only. `orbital:[EL:]l` character colors are unchanged.
- QSGW: run on the SC run's `WFN_qp.h5` with `--wfn-file` and
  `--velocity dipole_qsgw.h5`. A DFT `dipole.h5` on a QP WFN, a QP velocity
  with no Σ term, and `--velocity` with `--qp-rotations`/`--eqp-file` refuse.
- `psp.orbital_magnetization`: the Sternheimer branch (`--method`,
  `--truncation-2d`) is gone; the CLI refuses a QP WFN and projects on the
  spin-moment vector; its `--out` npz has `colA_n`/`colB_n`, `m_spin` (vector)
  and `spin_axis` in place of `colA_z`/`colB_z`/`m_spin_z`.
  `orbital_pieces_at_k` moved to `psp.orbital_response`.

## 2026-10-04 — the static head reads the deck's Hubbard input

`HeadResolver` now passes `hubbard_input` and `hubbard_occupations` to the
`dipole.h5` provenance check. Before, every DFT+U deck with the default
`wcoul0_source = s_tensor` head refused with `GATE dftu_velocity_input`
before map 0. No result moves; plain-DFT decks are unchanged.

## 2026-10-04 — QE's data decides two-component time reversal

For a two-component WFN, TRS now holds only if QE types no operation
t_rev = 1 and the SCF absolute magnetization is below 1e-4 μB/cell. A
t_rev = 1 row proves Θ broken (Θ⁻¹(Θg) = g would be a unitary symmetry), and
in noncollinear DFT the only TR-odd Kohn-Sham term is σ·B_xc[m], so TRS holds
iff m(r) = 0. The tolerance splits Kramers pairs by about 0.1 meV at a Stoner
I of 1 eV/μB. The magnetization comes from the bound schema when it is
self-consistent, is 0 when `do_magnetization` is false, and otherwise comes
from the SCF schema whose `charge-density` file the NSCF `.save` holds byte
for byte (an NSCF schema writes `absolute` = 0).
- No authenticated schema, or no matching SCF schema, means TRS off. Co-stage
  the SCF `*.save` (QE's `scf/` beside `nscf/`) with the NSCF one.
- The wavefunction check runs only when QE says nonmagnetic, as a guard: a
  residual above tolerance refuses (`GATE trs_qe_nonmagnetic_wfn_consistent`).
  A TRIM-only or empty guard no longer turns TRS off for a nonmagnetic,
  non-centrosymmetric crystal.
- Sandbox decks keep their verdicts: CrSBr, CrI3, NiPS3, Fe, Co, Ni off; Si,
  Na, Bi, TaAs, MoS2 on; the H2⁻ hsuite fixture off. The stamp algorithm is
  v3, so no cached verdict is reused.
- Decks do not change.

## 2026-10-04 — bispinor sectors build whole q-local pole models per rank when parents ≥ ranks

With at least as many irreducible q-parents as ranks, the bispinor CC/TT/CT
sector constructor now runs local rounds: each rank builds its own parent's
three models, ceil(nq/P) rounds. Before, the CT joint pencil's conservative
price (both retained spans at twice their pole budgets) sent all three sectors
to the face route. There the batch sizing admitted one parent per round, and
every parent ran on the whole mesh in turn.
- Ni 20³ bispinor SC, P64 on A100-40GB: the map-0 sector constructor takes
  643 s in 11 rounds of 64 parents, against about 10,300 s (641 face rounds
  of 16.1 s). The CT joint side is 3968–4352, against the conservative 10760.
- Each round admits the CT pencil at its actual spans; a price over the budget
  warns and runs. With fewer parents than ranks the old rule applies.
- The models match main's local route bit for bit. Decks that move from face
  to local change by the face/local difference: on Fe 4³, CT W(z) within
  9.6e-7 relative at map 0 and eqp within 0.1 µeV.
- A local round writes each model once (no 4-parent store span cap).
- Decks do not change.

## 2026-10-04 — the two-component TRS check pairs k with −k only through QE-unitary rows

The DFT-reference time-reversal check now takes the spatial operation that maps
a stored k to −k only from the rows QE types unitary (t_rev = 0). It used to
take the first matching row, including the ones QE composes with time reversal.
On a magnet where such a row maps k to −k before a unitary one does, the test
compared Θg with g and passed by construction. Monolayer CrSBr (M ∥ b; QE's
ΘC2z precedes inversion) is the case found: it reported TRS consistent at 6.6e-8
on a 6 μB/cell ferromagnet. It then ran on the even shared-pole route, with −k
and (kx, −ky) unfolded through false unitary C2z and C2x. Its q = 0 Gram refused
(GATE shared_pole_gram_valid, −1.86e-6 at 16×12).
- Now the check reads 0.29 and reports TRS broken. CrSBr takes the ordered
  route: all 63 parents construct, and the q = 0 Gram minimum is −6.1e-12.
- Any deck whose verdict flips moves from the even to the ordered route. No
  other production deck in the sandbox flips: non-magnetic and schema-less
  WFNs, CrI3 and the H2⁻ fixture keep their verdicts. Without a QE schema every
  row is still presumed unitary, and SymMaps warns as before.
- The cached verdict stamps carry the QE typing, so an old stamp is not reused.
- Decks do not change.

## 2026-10-04 — half first Anderson step; the SC criterion pairs by sorted index

The SC Anderson loop's first step is now half a plain step, x1 = x0 + f0/2.
Later steps are unchanged (β = 1). The step index is the global map index,
so a run resumed from a map-0 checkpoint takes the same half step. Along a
mode with plain-step eigenvalue λ, the map-1 residual is |1 + λ|/2 of map 0's
instead of |λ|.
- CrI3 24×24 bispinor (λ ≈ −2.5): 0.75 instead of 2.5.
- Si 4³ (λ ≈ +0.15): map 1 is worse (1117 against 343 meV), and the run
  still converges in 8 maps.
- On Si 4³, eqp0 at |E − VBM| ≤ 10 eV moves by at most 1.0 meV. The deepest
  valence pair, 12 eV below the VBM, moves 6.2 meV. That is within that
  state's dependence on the held Σ plan, which is 12.8 meV on the previous
  code.

The convergence criterion now pairs each input column with the output column
of the same sorted index. Before, input and output were matched to the map-0
labels by two separate assignments. On CrI3 24×24 charge QSGW, hybridized
states then read 1.3 eV where the sorted pair moved 6 meV, so the 1 meV
criterion fired at map 10 instead of map 9. The label now only names a pair.
The `SC map gain` line pairs adjacent inputs by sorted index.

Log lines:
- `SC identity: max |dE| = ... eV, input and output paired by sorted index`
  replaces `SC identity: max |dE| by overlap = ... (sorted-index value ...)`.
- `SC matrix residual` ends with `FLAG` when max|dE| exceeds max_k‖f_k‖₂.

Decks do not change. The in-process linear-mixing path is deleted. A checkpoint
written before this change still resumes. Its first `SC map gain` line after
the resume compares a label-paired table with a sorted one.

## 2026-10-04 — an SC run continues across processes from sc_seed/sc_checkpoint.h5

Every unconverged SC map except the budget's last now writes
`sc_seed/sc_checkpoint.h5` beside the warm seed. It holds:
- the Anderson history stacks, newest pair and residual history;
- the previous map's Z and the map-0 identity labels;
- the held plan without fitted rules: ω grid, windows, supports, K/CT/carrier
  extents, and each Σ window's boxes, including the box its rule was built on.

When a deck's `sc_initial_qp_rotations_file` names a warm seed with an
authenticated checkpoint beside it, the run continues at the next map. It
authenticates when the deck matches, apart from the `restart`, seed,
`sc_max_iter` and `sc_tol_ev` lines, and when the WFN, carry and
`sc_history_depth` match. The continuation:
- numbers maps globally (`eqp0_iterNNNN.dat`, `tmp/mpa/sc_NNNN_shared_pole`);
- rebuilds each Σ rule cold on its held box;
- plans the χ rule cold.

Before, every process took one unmixed step: on CrI3 24×24 the gap went
4.154 → 4.283 → 4.007 eV over two such steps.

Results:
- A one-map-per-process Fe 4³ bispinor SC follows the one-process run within
  3 µeV per map. A seed alone is off by 485 meV at map 2.
- The write costs 0.1 s per Fe 4³ map. On CrI3 24×24 the file is 1.8 GB.
- A converged run deletes its checkpoint. One that does not authenticate is
  left in place with a WARNING. To start fresh from the same seed, remove the
  file.

Log lines: `SC checkpoint: map N written` and `SC resume: continuing the
Anderson trajectory at map n`. `file_io.sigma_checkpoint` now publishes through
`collective_atomic_file_transaction`, with a commit over the cube bytes. A
one-shot Σ checkpoint written before this change is recomputed once.

Decks do not change. A leg chain copies the whole `sc_seed/` directory.

## 2026-10-03 — route-G stage 0 streams over parent chunks; step_up steps by the compiled figure

Stage 0 of a route-G ζ batch (X_B, the pair GEMM and the all-to-all) runs
one chunk of raw parents at a time and writes into the owner's D̃, which is
allocated once with its zero column. The all-to-all no longer holds a second
copy of D̃ (28.7 GB on CrI3 24×24 bispinor, at every P), and X_B is one
chunk's instead of growing with P. A chunk is the fewest parents whose X_B
psum and all-to-all reach the comm model's efficient payload, at most one
1 GiB tile of X_B, balanced over the parents (CrI3 24×24: 61 chunks of 1;
small decks: one chunk). The X_B band chunks (`x_chunks`) are removed.
`runtime.aot_memory.step_up` now steps to the block count the compiled figure
implies (a secant through the last two figures), rounded up to the next
count with fewer plane groups per block, instead of doubling; a batch 0.3 %
over the room at 50 blocks now runs at 63, not 100. Results: Fe 4³ bispinor
V_q bitwise on the 0.25 GB deck and within rel 1.4e-10 on the roomy deck
(claim 3177). CrI3 24×24 bispinor at P16 now fits the 56.16 GB target:
the charge batch compiles to 38.18 GB beside 16.3 GB resident, where main
modelled 75.6 GB. The plane stage then binds at 16 blocks (charge) and
50–63 blocks (currents), with t_u = 0.44 s per (block, row). Projected P16 ζ
fits: charge 18 × 110 s ≈ 33 min, currents ≈ 62 min (91 min measured with the
old doubling to 100 blocks). Decks do not change.

## 2026-10-03 — sector face rounds solve on the Ritz carrier; CT span widths are held from map 0

The CC/TT face reduction solves its kept span on `face_ritz_carrier` (the
pole budget, per-rank tile on the extent ladder), as the scalar face route
already did, and the sector batch is sized with it: the Schur and final eighs
run at the carrier and twice it instead of at H'_vv's side and the pencil
side (CrI3 24×24 bispinor P64: TT 9216/18432 instead of 12416/24832, whose
24832 stack ran on the whole mesh; CC 6144/12288 instead of 8736/17472). The
CT round's retained CC/TT span widths are held from SC map 0, so later
rounds and map 1 no longer recompile the CT reduction as the widths regrow.
Results: Fe 4^3 forced-face bispinor SC maps 0-2, eqp0 bitwise at map 0 and
within 0.077 µeV at maps 1-2; W response 197.0 / 62.9 / 13.5 s against
196.1 / 66.9 / 14.6 s (carrier). The CT binding on top is bitwise at maps
0-1 and within 0.032 µeV at map 2; map 1's CT stage drops from 20.0 to
14.2 s (W response 55.7 s against 62.9 s). Decks do not change.

## 2026-10-03 — the CC/TT/CT face batch is admitted by its compiled size

The bispinor sector constructor on the face route admits its parent batch by
the largest compiled size of the round's three whole-chain programs at the
conservative shapes (CC's and TT's reduction, on the Ritz carrier since the
entry above, and CT's cross reduction), beside
the resident sector models, as the scalar face batch already was. Every CC,
TT and CT eigh stack is decided against the room beside that batch row, which
bounds every round's retry. Each face model check of the scalar constructor
takes its eigh room beside its own whole-chain program. The analytic face
prices of the sector reduction ($14R^2+12nR$) and of the CT cross reduction
($10CT+14R^2+12n(C+T)$) are gone. Both face batches now start at every parent
and step down by room, so a deck that fits runs one round. The
one-parent-per-rank start of the previous entry is withdrawn; it split Fe 4^3
at P4 into four rounds whose different sides recompiled every program. The
sector constructor line names each sector's compiled program, its sizing
wall and the eigh rooms used. Results: Fe 4^3 forced face, SC maps 0-2.
Bispinor eqp is bitwise to main, and the W response is 197.3 / 66.7 / 14.6 s
against 150.2 / 67.2 / 14.6 s. Map 0 adds the one-time batch sizing (46 s).
The scalar returns to its pre-2026-10-03 batch of 13 and is bitwise to it; it
differs from current main by 0.071 ueV or less. Decks do not change.

## 2026-10-03 — face programs size on their first attempts; the scalar face batch is admitted by its compiled size

A whole-mesh (face) shared-pole program now holds only the first attempt and
check of every checked eigh inside it and returns their mesh-reduced failure
flag (`distrib_la.checked_program`); a failed first attempt returns zeros so
the program stays finite. Only when the flag is set does the round run again,
with every eigh's whole chain in-graph and on the whole mesh; a solve that
fails every attempt refuses by name. Route-(c) stack
decisions inside these programs are sized on the first attempt, not the
in-graph retry chain. The scalar face constructor admits its parent batch only
by the compiled size of its round program at the conservative side
(`face_batch_width`, starting at one parent per rank), and that figure
replaces the `14R^2+12nR` price in its reduction rows. Each round is admitted
by the compiled size of its whole-chain, whole-mesh program on the arrays it
runs, which is also its retry; its eigh stacks take route (c) in the room
that leaves. The constructor
line names the compiled size and its sizing wall. The CC/TT/CT sector batch
keeps its analytic price. Results move by round-off: Fe 4^3 forced face SC
maps 0-2 within 0.071 ueV of main (scalar; bispinor bitwise), CrI3 24x24
charge P16 eqp0 within 0.063 ueV. On CrI3 P16 80 GB the face batch is 8
parents (was 7), every reduction eigh stack runs one whole matrix per rank,
and the map-0 constructor drops from 902 s to 644 s. A face batch now holds at
most one parent per rank: Fe 4^3 forced face at P4 runs 4 rounds instead of 1
(+0.5 s per later map). Decks do not change.

## 2026-10-02 — distrib_la decides where a face eigh stack runs

The whole-mesh (face) shared-pole constructors, scalar and CC/TT/CT, hand
every eigh stack to `distrib_la` with the room per rank beside their admitted
batch. The service runs the stack as whole matrices per rank (route (c), the
local solver) when the compiled program that runs it fits that room, in as
many slices as it needs, and on the whole mesh otherwise. The n x n eighs
(directions, infinity, passivity) take route (c) almost always; a side-sized
Gram eigh takes it only where 7 n^2 x 16 bytes per matrix (3 n^2 compiled,
4 n^2 cuSOLVER workspace) fits beside the batch (n = 9152: 9.4 GB; n = 18304:
37.6 GB). Each decision is printed once
after the constructor line, `distrib_la eigh stack B x n^2 complex128:
...`, and the constructor line now names the face batch and its eigh rooms.
Every route-(c) eigh is checked against the service's probes with the same
retry chain as a distributed eigh, and the route is decided on the compiled
size of what the program reserves. The choice is agreed over ranks. Results
move by eigenvector gauge only: Fe 4^3 forced-face
SC maps 0-2, scalar and bispinor, agree with local main to 0.15 ueV. On the
same deck the face constructor's maps 1-2 drop from about 15 s to 2 s on 4
A100s.

The ordered (time-reversal-broken) face reduction now solves its kept span
on the pole budget's carrier, as a local round does: its Schur and final
eighs run at the carrier and twice it instead of at H'_vv's side and the
pencil side (CrI3 24×24 charge on 4×4: 6144 and 12288 instead of 9152 and
18304). The dropped columns are exact zeros: CrI3 P16 eqp0 and eqp1 are
bitwise to the run without it, and its map-0 constructor drops from 1354 s to
about 1030 s (W response 1470 to 1017 s). Decks do not change.

## 2026-10-02 — distributed eigh and LU solves are checked before they are returned

cuSOLVERMp 0.9.1's distributed eigh returns wrong eigenvectors with status 0
and info 0 on valid Hermitian inputs: exact-zero blocks, large near-zero
clusters of rank-deficient PSD matrices, sentinel-padded pencils. This was
reproduced outside LORRAX. On main the face (whole-mesh) route used some of
these silently: 10 eighs per constructor on CrI3 24×24 charge at P16, 2 per
map on Fe 4³ bispinor with the sector route on the face.

- **Every distributed eigh is checked** with fixed-seed probes (backward
  error within `distrib_la.roundoff_tol(n, dtype)`). A failure is solved
  again: shifted, with the vectors re-orthonormalized; then in the other
  cuSOLVERMp layout; then gathered when it fits 64 MiB.
  Each retry prints one `distrib_la:` line on rank 0's stderr.
- **Every distributed LU solve and Cholesky/LU `factor`/`solve`** is checked
  on its solution.
- **A result that no retry repairs refuses** with `GATE
  distrib_la_result_check` (op, n, call site, errors). Inside a program the
  result is NaN, so the caller's finite-result gate refuses too.
- **Every eigh route deflates exact-zero rows** with sentinels relative to
  the matrix. gw's own copies (`zero_row_safe_eigh`) are gone.
- **Which results move:**
  - Runs whose face eighs failed silently on main move to the correct
    decomposition.
  - Local-route runs move at round-off only: Fe 4³ scalar and bispinor maps
    0–2 ≤ 0.17 µeV, from the relative sentinels and the deflation of the
    normal and polar kernels.
  - Forced-face Fe 4³ bispinor agrees with local to ≤ 1.3 µeV.
- **Memory:** a checked distributed eigh reserves 7 n²/P elements per rank
  inside a program, priced by `distrib_la.workspace_bytes_per_rank`. An eager
  call reserves only its first attempt, about 3 n²/P, and runs any retries
  as a separate program.
- **Nothing to change in decks.** The check has no dial. The native
  cuSOLVERMp handlers do not yet read `info` (next bundle).

## 2026-10-02 — planners admit by compiled size; kin_ion writes the window GW reads

No result moves: Fe 4³ scalar and bispinor and Na 8³ SC maps 0–2 are bitwise.
What changes is which runs warn, the files kin_ion writes, and a few receipts.

- **`gw.kin_ion_io` writes b3 bands by default.** b3 = nelec + ncond, the
  window GW reads, with `number_bands_protected` resolved as GW resolves it.
  The old default, max(`number_bands`, nelec + ncond), loaded ψ(G) for bands
  no consumer reads; CrI3 24×24 went from 750 bands to 208. A `kin_ion.h5`
  written before still works, and `-n NB` still writes more.
- **Compiled chunks are admitted only when they fit.** A compiled chunk is
  admitted only when it fits the room; the analytic price no longer admits it.
  A chunk at its minimum that does not fit now prints the over-budget warning
  instead of passing silently. Every rank runs the smallest chunk any rank
  chose (`runtime.aot_memory.agreed_chunk`).
- **The ζ μ-batch plan states its real batch.** Route G keeps whole centroid
  orbits per owner, so the plan floors the batch at P × the widest orbit.
  Its HWM is that of the batch that runs, and it warns before the compile
  when that HWM is over the target. CrI3 24×24 bispinor at P36 now plans
  432 centroids at 70 GB/rank, where it used to print 36. It does not fit
  40 GB cards at any P ([zeta fit](docs/architecture/zeta_fit_mubatch.md)).
- **Hermiticity checks scale with n.** Every Hermiticity check compares
  against `distrib_la.roundoff_tol(n) = 64·n·eps`
  ([api](docs/services/distrib_la/api.md#roundoff)), not a fixed 1e-12. A
  large GEMM-built matrix no longer refuses on round-off. A matrix of side
  n < 71 meets a bar below 1e-12.
- **Receipt changes.**
  - `constructor_receipt.json` no longer carries `spectral_moment_cauchy`;
    the report never passed its support test. A non-finite sector M1 still
    refuses (`GATE shared_pole_sector_nonfinite`).
  - On a deck with broken time reversal, the parallel-transport head no
    longer computes the velocity-parity diagnostic; it prints one line
    instead.

## 2026-10-02 — XLA rematerialization is off; a module larger than the device stops by name

LORRAX turns off XLA's HLO rematerialization pass for every program: the
runtime adds `--xla_disable_hlo_passes=rematerialization` to `XLA_FLAGS`,
merged into any pass list you already set. The pass ran only on a module
whose peak was above XLA's memory limit. It could add hours of compile, and on
the decks that printed `LORRAX GATE xla_rematerialization: a module does not
fit` it freed nothing. Compile time drops where the pass ran long (on the
CrI3 8×8 kmeans Gram it gave up in under a second, so nothing changes there).
Results do not move: Fe 4³ scalar and bispinor SC maps 0–3 and kmeans
centroids are bitwise.

Every compiled module is now checked against the card before it first runs.
When its buffers (temp + arguments + outputs − aliased) exceed the device's
total memory, the run stops with
`GATE xla_rematerialization: REFUSED module '<name>': its compiled buffers
need X GB per device (…), and the device has Y GB`, instead of an
out-of-memory error deep in the allocator. Raise the rank count or lower the
module's size ([memory model](docs/architecture/memory-model.md#module-does-not-fit)). The
`memory_per_device_gb` budget still only warns. The old stderr banner is gone.

Expect one cold compile: `XLA_FLAGS` is part of the compile-cache key, so the
first run of every deck after this change recompiles every program (the P4
hsuite took 445 s cold), and later runs hit the cache again. If a future jax
changes the private executor this check hooks, the run prints one
`jax-compat: … GATE xla_rematerialization is NOT armed` line and continues
unchecked.

## 2026-10-02 — band extrapolation on the four-current Σ (`full_shared_pole`)

`bispinor_gw = full_shared_pole` now uses `use_band_extrapolation` (on by
default), as the scalar shared pole and `bare_transverse` already do. The CC
class's Green band sum is split into the three brackets and extrapolated by
the pooled `spectral_shell` fit. TT, CT and TC, which are c⁻² of CC, are
summed to N as before ([four-current Σ](docs/theory/band-extrapolation.md#four-current)).
Every `full_shared_pole` run that does not set `use_band_extrapolation = false` moves
once. A deck with `number_bands_sigma` < 2·n_occ now refuses at startup:
raise the band count or set the key to false. Decks that set it to false are
bitwise. On Fe 4³ bispinor `full_shared_pole` SC at 36 bands, map 0 eqp0 within E_F ± 10 eV
moves by a median of 342 meV (at most 891 meV, mean −385 meV). Σ τ costs 13–20 %
more per map, and the map wall 3–6 % more; device peak is unchanged.

## 2026-10-02 — "door" is gone from the code and the logs

The term "door" is replaced everywhere. A mathdx k-convolution call site and its
symmetry tables are now a "kconv call" and "kconv tables". A service's top-level
package is its "public API". A single function that serves one operation is an
"entry point". The `ffi.fft` router's functions are "factories". Three output
names change: the stage `response.door_tables` is now `response.kconv_tables`,
the four-current line reads "four-current mode-11 kconv calls: N kconv calls",
and the capacity-ledger row `door_tables` is now `kconv_tables`. A parser that
matched the old names must match the new ones. `minimax.door` is now
`minimax.serving`; `import minimax` is unchanged. No number moves.

## 2026-10-01 — the shared-pole χ and W banks stream through scratch; keep terabytes free

When a shared-pole map's χ bank (the value and slope of every response sample)
does not fit the devices, the response stream now runs once with every sample
and writes its carry to SlabIO's per-rank streamed tier, instead of re-running
every Green pair once per sample group. The W bank goes to the same tier when
it does not fit the devices. Each bank is held in host memory when it fits half
the host budget, else in one file per rank under
`<run dir>/<label>_shared_pole/streamed_bank/`. gwjax.out names the choice, for
example `Response quadrature: chi bank file, 34.94 GiB/rank; streamed; exceeds
half the host budget`. Results are bitwise (Fe 4³, Na 8³ and Fe 4³ bispinor
SC maps 0–2). The Fe 8³ P4 χ build takes 35 → 19 s per map.

- **Scratch.** A large deck needs the χ bank's bytes free on scratch, and
  under the quota, during every map: 2.4 TB for Ni 20³ charge and 15.3 TB for
  Fe 20³ bispinor at 1062 parents
  ([memory model](docs/architecture/memory-model.md#streamed-chi-bank)). When
  the disk or quota cannot hold it, the samples run in disk groups (half the
  samples per try), then in device groups. Each group re-runs every Green
  pair: the Fe 20³ bispinor χ build is about 420 s per map in one group and
  1,300–1,400 s in four.
- **Lifetime.** A file is unlinked as soon as it is opened, so `ls` shows an
  empty directory while the run holds the bytes, and every exit, SIGKILL
  included, frees them. Nothing is kept across SC maps, and an interrupted
  map cannot resume from the tier.
- **Cleanup.** Code before 2ee02941a left named files after an abnormal exit.
  The first shared-pole map of a process now removes every `streamed_bank/`
  directory in its run directory and prints `WARNING shared-pole output:
  removed streamed stores left by an earlier process`. Delete those
  directories of finished runs by hand
  ([SlabIO](docs/architecture/slab_io.md#streamed-tier)).
- **Map-1 failure, fixed in 6bc09936c.** With the W bank on per-rank files,
  a store created after others had been written counted their bytes twice
  against the quota, was refused, and was then written: the map failed with
  `GATE io_global_commit … streamed_bank.commit` and a `KeyError` (Ni 20³ P64,
  map 1). A source tree before 6bc09936c, and a module built from one, still
  fails this way.

## 2026-10-01 — row passes run as one scan; compile no longer grows with the pass count

The Σ τ node, the static τ = 0 node (Σ_x, SX, COH), the bispinor sector node
and the charge and four-current χ streams now run their row passes as equal
orbit-aligned windows in one `lax.scan`. The unfold tables are cut on the
device and the k-convolutions skip each window's padded rows
([memory model](docs/architecture/memory-model.md#the-green-side-stages)), so
one program serves every pass. Compile: the static Σ program at 223 passes
25.7 → 0.8 s; the streamed χ bank's segment programs 33 (21.6 s) → 1 (0.7 s)
on the charge stream and 203 (234.6 s) → 4 (4.9 s) on the four-current
stream, whose peak falls by 3.6 GB per rank. Walls are within ±1 %. A rank whose rows fit one
tile runs one pass and is bitwise (Fe 4³ scalar, Na 8³ and Fe 4³ bispinor
`full_shared_pole`, maps 0–2). With several Σ passes per rank the sums group
differently and move at round-off, the size of a 4e-16 control (≤ 1e-7 eV on
Fe 4³ forced to many passes); the χ streams stay bitwise. The
k-convolutions' `live` operand needs a native bundle built from 935bcede8 or
later.

## 2026-10-01 — the photon bank solves a sample's Dyson value and slope in one program

The four-current bank now forms each sample's W − W∞ and its slope in one program, as
the charge bank does; the contact and W∞ − V enter in the old order. Fe 4³ bispinor
Dyson dispatch per SC map is 18 % faster. Results move at round-off: Wc is bitwise,
the slope moves at 1.3e-15 relative on the local backend (bitwise on the distributed
one), and Fe 4³ bispinor SC maps move ≤ 0.08 µeV, inside a 4e-16 control.

## 2026-10-01 — the bispinor sector constructor solves on the Ritz carrier

A bispinor SC map's sector constructor now solves each round's kept span on the
ladder rung of the largest kept count that sector has had, as the scalar model
does, and reruns wider when a round keeps more. Map 0 is bitwise; later maps move
at round-off (Fe 4³ bispinor ≤ 0.09 µeV).

## 2026-10-01 — the bispinor sector W(τ) runs on the scalar model's synthesis

Each photon sector's W(τ) = B_A d(τ) B_B† is now formed by the scalar shared-pole
synthesis (`gw.mpa.sigma.synthesize_shared_pole_parents`) and placed by the scalar
rule: replicated pole columns when they fit, else whole parents per rank on a
`linalg = local` deck, else the face SUMMA. Only each parent's live pole columns
are contracted. The SUMMA panels and the W∞ − V constant's q panels are sized from
the shapes (one tile), no longer from `memory_per_device_gb`. Fe 4³ bispinor `full_shared_pole`
Σ τ is 19 % faster per SC map. Bispinor results move at round-off: map 0 is bitwise,
maps 1–2 move ≤ 0.19 µeV (a ±4e-16 control moves them ≤ 0.11 µeV). Scalar decks
are bitwise.

## 2026-10-01 — the shared-pole Σ τ window overlaps one node's W exchange with the next node's compute

On GPU, the scalar shared-pole Σ τ window now evaluates two τ nodes per loop
trip when the compiled paired window fits the device budget beside the live
stages. Otherwise it runs one node per trip, as before. When it pairs, the Σ τ
window line of gwjax.out's memory table says "two nodes per trip". Only
that program is compiled with XLA's latency-hiding scheduler
(`gw.ppm_accumulators.WINDOW_OVERLAP`); the global flag stays off. One node's
W(τ) synthesis and all_to_all now run beside the other node's k-convolutions.
On the Ni 20³ SC map 0 at P64 (band extrapolation on), the window executables
take 351.7 → 322.3 s (−8.4 %). The window holds a second node's live set:
compiled 9.76 → 16.95 GB per rank, and a run peak of 28.5 GB instead of 23.9 GB,
now set by Σ τ. The Σ τ stage prices this and admission reads it from the
compiled executable. On a cold cache the first window compiles 13 → 34 s.
Results move at round-off: Fe 4³ and Na 8³ SC maps 1–2 stay inside a 4e-16
control, map 0 is bitwise, and the Ni map is bitwise at printed precision.

## 2026-10-01 — QP seeds are projected on each k's little group; the four-current χ bank carries no −q rows

An external SC seed (`sc_initial_qp_rotations_file`) is now averaged over each
kept k's little group at import (H ← |G_k|⁻¹ Σ_L A_L(H), with the band
representations of the little-group operations from the WFN), so a seed written
by another run or code version cannot break this run's symmetry; gwjax.out's
"SC initial Hamiltonian" line prints the largest change (the Fe 4³ bispinor
2026-09-19 seed: 7.2e-5 eV). A seed whose Σ window cuts a multiplet refuses
(`GATE little_group_band_representation`). Seeded SC runs move once (Fe 4³
bispinor ≤ 0.16 meV); runs without a seed are unchanged. The four-current bank
on inversion-symmetric magnets now also forms χ_{−q} from the parent rows by
the inversion (Lorentz blocks mixed by the inversion's action), halving it
(Fe/Ni 20³: 21.4 → ~10.7 GB per sample per rank); bispinor SC eqp moves by
≤ 32 µeV.

## 2026-10-01 — distributed eigh runs block-cyclic

A `linalg = distributed` eigenproblem (cuSOLVERMp, `distrib_la`) now describes
each rank's tile with a square block, the largest divisor of n/p at most 256,
so no rank idles as the trailing matrix shrinks; one all_to_all over `x` puts
the eigenvalue index back in order. At n = 16000: 17.50 → 10.59 s on 16 GPUs
(1.65×), 20.05 → 15.65 s on 64 (1.28×); neutral at n/p ≤ 1000. LU and
Cholesky keep one tile per rank (block-cyclic solves measured 2–3× slower).
Eigenpairs move at round-off (eigenvalues ≤ 1e-15 relative). Fe 4³
`linalg = distributed` eqp0 at maps 0–2 is bitwise at its natural block and
moves ≤ 1.8e-4 meV at a forced block of 72, the size of a 4e-16 control
(1.4e-4 meV).

## 2026-10-01 — the static Σ runs the Σ τ kernel at τ = 0; the map-0 exchange compile is gone

Σ_x (every SC map) and static COHSEX on the parent route now run the Σ τ
sub-tile kernel at τ = 0 with the static interaction in place of W(τ), its
unfold tables read as device operands. Before, the Σ_x program baked the
Green's global unfold tables into its HLO: at Ni 20³ on 64 GPUs (1782
centroids) that compile took 93 s of the 105–120 s map-0 "Sigma exchange",
which now takes about 10 s. At the P64-local tile a cold call takes 28.7 →
6.7 s (compile 25.0 → 2.4 s), a warm call 0.720 → 0.672 s, and the peak
14.0 → 8.9 GB per rank. Results move at round-off: Σ_x by 4.1e-16 relative
at most; map 0 is bitwise, and Fe 4³ and Na 8³ SC maps 1–2 move by the size
of a 4e-16 Σ_x control.

## 2026-10-01 — no `coulomb.h5`; an older bank's constructor resume refuses

Every scalar shared-pole map wrote the bare V to `coulomb.h5` and read it
back to build the Coulomb roots (54.6 GB and 41 s at Ni 20³ map 0, plus a
content hash). The roots are now formed from V on the devices, bit for bit
the file route's, and V is named by a device digest that is the same on
every rank and at every P. Results are bitwise, and a map directory holds
no `coulomb.h5`. Resuming a constructor from a bank built before this change
refuses with `GATE shared_pole_output: the bank at … was built with another
bare V; use a fresh run directory`: delete that map directory or use a fresh
run directory. The photon V file is unchanged.

## 2026-10-01 — the χ bank carries no −q rows on inversion-symmetric magnets

On the ordered (time-reversal-broken) scalar route, every response sample's
bank carried the parent q rows and their −q partners (2120 rows at Fe/Ni 20³),
although only the line samples' partner solve reads the −q rows. When the
magnetic group holds a unitary inversion (Fe, Co, Ni, CrI3), the partner
χ_{−q} = U_I χ_q U_Iᴴ is now formed from the parent row by the inversion unfold
(`symmetry_maps.unfold_isdf_operator`) at that solve, so the stream and the bank
carry only the parent rows: half the bank per sample (Fe/Ni 20³ P64: 3.4 → 1.7
GB/rank). Scalar SC results on inversion-symmetric magnets move once, by up
to 0.75 meV (Fe 4³, maps 0–2). The unfolded partner differs from the streamed
one by ≤ 4e-9 relative, and the line selection amplifies it (a 1+4e-9 scaling
of the partner moves Fe 4³ by 1.77 meV); this is inside the 2 meV gate for the
ill-conditioned fit and selection. Time-reversal-symmetric decks (Na), groups
without a unitary inversion, and the four-current route are unchanged (bitwise).

## 2026-10-01 — a price over the memory budget warns, never refuses

No planner stops a run because a priced or compiled memory figure exceeds
`memory_per_device_gb` (or its tile). The shared-pole capacity ledger, the
compiled chunk check, the shared-pole sector batch and local pencil rounds,
the ζ μ-batch, V_q, pair-convolution, GN-PPM fit, Galerkin, kmeans Gram,
centroid-load, W-av and dense-H planners each raise one `RuntimeWarning:
memory over budget at <stage>: needs X GB/rank, budget Y GB/rank, over by Z GB;
continuing (an OOM is possible)` (gwjax.out lists it under WARNINGS), take
their smallest size and run; a device that truly lacks the room OOMs. The gates `shared_pole_capacity`
(budget), `compiled_chunk_capacity`, `shared_pole_round_capacity`,
`zeta-mubatch-capacity`, `zeta-mubatch-orbit-capacity`, `vq_tile_budget`,
`pairconv-capacity`, `gn_ppm_fit_capacity`, `bispinor-v-host-park`,
`pw-screening-budget` and the Γ-projection logical bound are gone. Decks that
fit are bitwise. Kernel shape limits (`GATE response_vertex_grid`) and
correctness gates still refuse.

## 2026-09-30 — shared-pole V staging: q tiles, one sync, a chunked digest

At map 0 the shared-pole W staged the bare V wedge one parent at a time, each
write synced, and then rank 0 alone SHA256-ed the whole file: 141.6 s (Fe 20³)
and 147.2 s (Ni 20³) on 64 GPUs, for a 54.6 GB `coulomb.h5`. The wedge now
streams in `runtime.tiles` q tiles into one write transaction that is synced
once, and `response_bank.resource_digest` (the one owner; it also authenticates
`v_q_bispinor.h5`) is SHA256 over the SHA256s of 256 MiB chunks that the ranks
read round robin, so each rank reads size/P. On one node (P4) with that 54.6 GB
file the rank-0 hash took 54.5 s and the chunked one 24.5 s (2.2 GB/s per node;
about 1.6 s over 16 nodes); the per-parent write and sync took 39.4 s and the
tiled write 36.7 s (one node is bandwidth-bound; at P64 the 1062 collective
syncs set the remaining ~87 s). The digest does not depend on the process
count. It differs from the old flat SHA256, so resuming a shared-pole
constructor from a directory staged before this release refuses with
`GATE response_coulomb_identity: content hash differs`; delete that map
directory and rerun. V and every result are unchanged.

## 2026-09-30 — every response sample group runs one program

When the response sample group is smaller than the sample count (memory-bound
decks: Fe/Ni 20³ on 64 40 GB GPUs take groups of 4 for 22 samples), the last
group was short (2 samples), a new carry shape, so `bank.compile.direct`
compiled a third stream program in the middle of map 0 (47.8 s on the Ni 20³
P64 run, which then died of host memory in that dispatch). A short group now
fills its empty slots with zero weights and runs the planned group's program;
it reserves the slots it allocates. Results are unchanged: with the group
forced to 5 on Fe 4³ (groups 5,5,5,5,2), eqp0 at maps 0–2 is bitwise to the
previous code at group 5 and at the default single group, and map 0 compiles two direct
programs instead of three. Decks whose samples fit one group are unaffected.

## 2026-09-30 — the scalar χ₀ mode-11 kconv call reads its tables as operands

The charge response stream on a raw-parent plan (the shared-pole direct
stream, the moment correlations and the retarded stream) baked its plan's
global unfold tables (row, trs, lsrc, rsrc, mph, nph, spin) into every
program as HLO constants. The tables now enter as device operands, placed
once per run and plan (`ffi.fft.make_kconv_chi_unfold(load=)`, as the
four-current route does), and the stream binds them as a trailing argument.
At the Fe 20³ table size (8000 k, 1792 centroids, one node): compile
7.08 → 0.57 s per program, generated code 115 → 0.1 MB, and each held
executable no longer adds host memory (+1.54 GB → +0.00 GB RSS per rank for
the second program). The Fe/Ni 20³ P64 runs compiled this program three times
at map 0 (34–48 s each), and the Ni run died of host memory during the
third. At the P64-local shape the stream's temporaries are 13.74 → 13.60 GB
and its dispatch is unchanged (0.439 s per Green pair). Results are bitwise
(Fe 4³ scalar and Na 8³ SC, three maps). The Σ τ mode-7 and mode-8 kconv calls
still bake their tables.

## 2026-09-30 — one MPI per process; CPU runs no longer hang in MPI_Init

Before it loads a sealed bundle's private SLATE closure, the native loader
(`lxkit.native_provider`) now loads the machine libraries that closure needs
from the directories the FFI leg's own DT_RPATH names: on Perlmutter, LibSci
25.09 and cray-mpich 9.0.1. Before, the private `libblaspp.so.2`, opened by
path, found the site-default LibSci 26.03 through `/opt/cray/pe/lib64`, and that
LibSci links cray-mpich 9.1.0. Every process then mapped two MPIs. A CPU run,
whose JAX MPI collectives had already started 9.0.1, hung in phdf5's
`MPI_Init`. GPU runs did all FFI and HDF5 MPI on 9.1.0; they now use 9.0.1, the
MPI the legs were built against, and results are bitwise (hsuite P4, Fe 4³
scalar SC maps 0–2). The one-MPI check now also sees `libmpi_gnu.so`
(cray-mpich ≥ 9.1) and refuses two MPIs by name. Each process prints one line,
`[lorrax native] rank=<r> mpi=<path>`. No deck or environment change is needed.
Remove any `LD_LIBRARY_PATH` LibSci 25.09 workaround.

## 2026-09-30 — SUMMA panel loops accumulate in place

When its loop has three or more band panels, `distrib_la.panel_matmul` now adds
each panel after the first into the output tile in place, through the local
beta = 1 GEMM. XLA folds one `c + a @ b` into its GEMM. Of two adjacent ones it
left one as an add, which holds two more output tiles. That happened in a
3-panel loop (XLA inlines its one-trip scan) and when a narrower tail panel
follows the full panels. Those Green builds now hold two fewer output tiles. On
the Fe 20³ P36-local CC stream (P4 proxy), a 3-panel loop compiles at 61.3 GB
instead of 73.1 GB, and the 360-band loop at 90.3 GB instead of 99.5 GB.
Loops of four or more full panels with no tail were already folded and compile
the same; P36 runs six. Two-panel loops (P4) keep XLA's GEMM. Results are
bitwise: with the panel loop forced to six panels, Fe 4³ scalar and bispinor
`full_shared_pole`, Na 8³ and MoS2 SC + BSE match the previous code, beside a 4e-16
control that moves.

`distrib_la.panel_matmul_extra_tiles` is deleted, and the photon row-pass count
no longer adds two Green tiles at p_x ≥ 3. Fe 20³ P36 at M_T 900 now prices
1,1,1,1 row passes at 70 or 75 GB (was 2,1,1,1), counted at 64.33 GB. M_T 1800
prices 3,7,7,14 at 75 GB (was 4,7,7,15). The 2,1,1,1 that the counted-passes
entry below gave P36 was an over-count. P36's 6-panel loop never held the two
extra tiles: its CC stream compiles at 60.68 GB before and after this change.

## 2026-09-30 — four-current row passes are counted, not fitted

`photon_response_passes` used a fitted price (2·parents + 1.5·planes). It now
counts each pass's live buffers from their shapes (`greens_function_kernel.price_photon_pass`).
The terms are the ones the XLA buffer assignments show:
- the channel planes, counted twice at 2 or more passes;
- the quadrant Greens and their partners;
- the SUMMA panels;
- both Dirac halves of the operand faces;
- the operands that other family pairs keep live (`photon_held_faces`);
- the placed kconv tables and mode 11's run-time scratch.
The ledger budget (`memory_per_device_gb`) and `GATE response_photon_passes` are
unchanged. The stage-memory table gets one row, "photon direct stream, row passes
(…)". Across 20 AOT programs at the Fe 20³ P36-local shape, the count is within
−3.1 % to +4.9 % of the compiled peak.

Fe 20³ P36 at M_T 900 and 70 or 75 GB moves from 2,2,2,2 passes (CC, CT, TC, TT)
to 1,1,1,1 with the in-place SUMMA accumulate above. The 2×2 proxy of the P36
tile compiles one pass on every pair at 64.49 + 1.07 GB scratch = 65.56 GB.
M_T 1800 has no AOT (production is M_T 900). Fe 4³ prices 1 pass before and
after, and its results are bitwise.

## 2026-09-30 — the four-current response compiles without baked symmetry tables

The bispinor four-current χ₀ (mathdx mode 11) now reads its unfold load
tables as device operands. Each kconv call used to bake its plan's global tables
into the program as constants. Each distinct table array is placed once per
run (stage `response.kconv_tables`), and every q batch and SC map reads it.
No kernel or bundle change.

Fe 20³ P36-local AOT (M_T 900, 2 row passes per family pair, cold cache, 4 ranks per node):

| | before | after |
|---|---|---|
| compile per rank | 136.7 s | 5.7 s |
| host max RSS per rank | 48.1 GB | 3.1 GB |
| node host peak | 175 of 251 GiB | 42 GiB |
| device args / temp / code | 24.63 / 40.03 GB / 644 MB | 25.44 / 39.80 GB / 0.45 MB |

What moves: nothing. Fe 4³ bispinor `full_shared_pole` SC is bitwise, and its warm
per-map W response wall is unchanged. Two concurrent compiles at M_T 1800 no
longer run the host out of memory on table copies.

## 2026-09-30 — streamed loops take a fixed 1 GiB tile; `device_room_bytes` is gone

Every planner that streams over k, q, bands, centroids, samples or rows now
takes the most units whose per-rank bytes fit one fixed tile,
`runtime.tiles.TILE_BYTES` (1 GiB). The tile comes from the loop's shapes
alone. It never reads free device memory or `memory_per_device_gb`, so every
rank computes it without a collective, and no result depends on the budget.
`common.gpu_utils.device_room_bytes` (the allocator read gathered over
processes) is deleted.

Two sizes still follow `memory_per_device_gb`, through a ledger: the
shared-pole response sample group (more samples per group is more than 10 %
faster per map, and the group moves no number since the all-sample response
rule) and the Galerkin whole-state fit in htransform.

What moves: nothing on decks whose loops already fit one tile (Fe 4³ scalar
and bispinor SC, Na 8³ SC, MoS2 SC + BSE and the hsuite are bitwise). On
larger decks a loop that used to take more than 1 GiB per rank now runs in
more passes of at most 1 GiB. Most of these loops are independent per unit,
so their numbers are unchanged. Three split a sum: the exciton_bands C_q q
chunk, the V_q G panel and the Σ output spin block. A deck whose tile shrinks
there moves at round-off. Per-map wall can move either way.

## 2026-09-30 — a held SC map refits the response rule warm when line sites move

When the shared-pole line sites move at a held SC map, the χ response rule is
refitted. The refit now tries the held rule's complex times first
(`minimax.response_group_rules` looks up the previous rule by its member
set; before, it looked them up by member order, which changes when sites
move, so every refit started cold). If the held times pass the same sampled
tolerance at the new samples, they are kept. Fe 4³ charge SC: rule build
3.4 → 0.7 s at maps 1 and 3, the W stage 7.2 → 4.6 s, 79 nodes as before.

What moves: SC runs whose line sites move. From the first warm refit on, they
move once, within the rule tolerance (Fe 4³ ≤ 0.002 meV within E_F ± 10 eV).
Runs whose sample order did not change, which includes Na 8³, are bitwise.

## 2026-09-30 — the shared-pole response rule no longer depends on the memory budget

The response bank fits its complex-time rule on all samples at once
(`response_bank.response_quadrature`). The response group that
`memory_per_device_gb` picks now only batches the evaluation: each group streams
the whole rule for its own samples. Before, each group got its own rule, so eqp
depended on the budget. Smaller groups were also less accurate: against a rule
tightened 100×, the error reached 14.6 meV on Na 8³ at group 1 and 7.6 meV on
Fe 4³ at group 8.

What moves: runs whose ledger picked a response group smaller than the sample
count (gwjax.out: "Response quadrature: N samples in M shared-node groups" with
M > 1). They move once, to the default's accuracy. Runs with every sample in one
group (all default-budget symmetric decks) are bitwise. A smaller group now
costs about 1.5–2× more Green pairs than before at the same group size; its
memory is unchanged.

## 2026-09-30 — the restart W0 is formed only on the q parents

GW no longer forms a full-q static W0 when it writes `W0_qmunu` for BSE. The
shared-pole evaluators (`shared_pole_static_wc`, and `sector_static_wc` for
`bispinor_gw = full_shared_pole`) compute V + W_c(0) at V's q parents only,
and the writer stores the producer's parents with their unfold tables. The file layout is
unchanged: a deck whose q axis reduces already stored W0 on the q parents, and
one whose q axis does not reduce stores every q. Both layouts still read; BSE
unfolds the parents on load in bounded q tiles, into the full-q layout its
kernels use. Nothing to regenerate; results are bitwise.

## 2026-09-30 — the Hall current takes the dipole's V_NL operator

`get_dipole_mtxels --static-gauge-hall-only` builds the Hall current with the
same nonlocal operator as `dipole.h5`. Superseded by the one-operator entry of
2026-10-20, which refuses every Hall artifact older than schema 3.

## 2026-09-30 — `parallel_transport`: the Σ term is served whenever the links are usable

The `parallel_transport` head no longer sets its Σ term D_kΔH to zero on a map
whose link bound exceeds 1 %. Complete links serve it on every map. The link
error is a k-convergence measure: the dipole step warns above
`--parallel-transport-validation-rtol` and still writes the artifact, each SC
map logs the error and its bound on the Σ term, and neither gates anything.
Only links that are not usable (incomplete, or a stencil or
window-hybridization gate fails) zero the term, on every map.

SC decks whose links were above the bound move once: MoS2 3×3 SOC (link error
9.2 %) converges to a 4.48 eV gap with the term served (5.17 eV with it
zeroed). Decks that stayed below 1 % are bitwise (Fe 4³ scalar)
([self-consistency](docs/self_consistency.md#metals-direct-drude-head)).

## 2026-09-30 — parallel-transport links: schema 4 on the link shell; rerun the dipole step

`parallel_transport` now differentiates on the point-group-closed
Marzari–Vanderbilt link shell (`common.parallel_transport.link_stencil`).
A link artifact written before this (schema 3, three reduced axes) refuses
under `parallel_transport`; rerun the dipole step (`psp.get_dipole_mtxels`).
`dft_velocity` still reads the old file. The bcc link stage takes about 2×
longer. Orthogonal lattices keep the three axes. Fe 4³ bispinor eqp moves
≤ 1.6 meV, Si SOC ≤ 16 µeV.

`bispinor_gw = full_shared_pole` SC decks that do not name `sc_head_update`
now run `parallel_transport` when a link artifact exists, as scalar decks
do, and move once (Fe 4³ eqp0 within E_F ± 10 eV ≤ 9.7 meV). On a metal,
`bare_transverse` refuses `parallel_transport`
([self-consistency](docs/self_consistency.md#metals-direct-drude-head)).

## 2026-09-30 — regenerate `WFN_qp.h5` from WFNs that store both k and −k

`WFN_qp.h5` now keeps time reversal on a WFN that stores two k of one orbit
(e.g. MoS2 3×3). A file written before this from such a WFN has broken rows,
and the BSE and GW runs that read it used them. Regenerate it with
`python -m postprocess.rotate_wfn_to_qp WFN.h5 qp_wfn_rotations.h5`; WFNs
without such rows (Si) give the same file
([self-consistency](docs/self_consistency.md#8-seeding-restart-and-outputs)).
An SC run that writes `WFN_qp.h5` now binds `dipole_qsgw.h5` to it, so a GW
run on `WFN_qp.h5` can take that file as its `dipole.h5`.

## 2026-09-30 — SC W line sites held within max(3 meV, 0.1 × max|dE|)

Held shared-pole W line sites are re-placed only when they would move by more
than max(3 meV, a tenth of the previous map's max|dE|). SC runs that re-plan
their sites move once; Fe 4³ scalar `parallel_transport` SC now converges
(28 maps; it stalled at map 16)
([self-consistency](docs/self_consistency.md#shared-pole-w-with-retained-quadrature)).

## 2026-09-29 — `qp_solver = fixed_point` and `eqp_root.dat` are retired

`qp_solver = fixed_point` refuses by name; set `one_shot_dft` (Σ at E_DFT)
or `self_consistent` (Σ at each map's own energies). No QP equation
E = h₀ + ReΣ(E) is solved on any route. The dynamic one-shot run no longer
writes `eqp_root.dat`, and `sigma_diag.dat` drops its `QP_status` column
(`Z` stays). `eqp0.dat` and `eqp1.dat` are unchanged.

## 2026-09-29 — the semicore class on the bispinor (sector) route

Every dynamic self-consistent bispinor deck with a coarse class
(`bispinor_gw = full_shared_pole` or any bispinor `compute_mode = mpa`)
moves once. The coarse (semicore) class was already built once from the DFT
ladder, above the Σ route; the sector Σ now reads it as the scalar Σ does:
the coarse states are read on held windows at η_semi = 5 eV, certified at
max(`sigma_quadrature_eps`, 3e-3), instead of at Σ(ω = 0), and
`sc_semicore = dft` (the default) pins their DFT block. `sc_semicore` and
`sigma_omega_patches_ev` `lo:hi:eta` triples behave the same on both routes.
`sc_semicore = dft` named on a run without a coarse class no longer refuses
(`GATE sc_semicore` is gone); it logs that there is nothing to pin.
Scalar decks are bitwise. A `full_shared_pole` SC run's head is its sector
model, so a converged run no longer refuses at the end
(`GATE sc_final_map_requires_iteration_head`), and a one-map
`dft_velocity` run is admitted (`GATE full_shared_pole_dft_velocity_one_map`
is gone).

## 2026-09-29 — bulk bispinor V carries the mini-BZ head average; bispinor refusals move to setup

Every bulk (`sys_dim = 3`) bispinor deck with `mc_average_vcoul_body = true`
(the default) moves once: the CC and TT tiles now take the scalar V's mini-BZ
average at the q ≠ 0 head slot (`v_q_g_flat.v_head_fn_in_V`, one owner); a TT
slot takes ⟨v⟩ P^T(K̂). Slab decks are unchanged. `full_shared_pole` with
`head_correction` unset resolves to `no_local_fields` (logged); an explicit
`full` still refuses. `w_bse` and `hl_ppm` on a WFN without
measured time reversal refuse before the basis, not after ζ and V (HL-PPM used
to keep one residue silently). Headless shared-pole SC warns on bispinor FD
metals too.

## 2026-09-29 — `sc_semicore = dft`: semicore pinned at its DFT block, mixing kept

New SC key, default `dft`; `qp` is the previous behaviour. Every dynamic SC
deck with a coarse class moves once.
Under `dft` the coarse (semicore) class keeps its DFT block of H in the DFT
basis and its end of every protected–semicore element reads Σ at E_DFT on the
held coarse windows ([self-consistency](docs/self_consistency.md#2-band-treatment)).
Fe 4³ and MoS2 3×3 decks requested with `number_bands_protected`, with the
semicore read at η_semi = 5 eV and, in comparison builds with
`qp_support.SEMICORE_ETA_EV` set to 8 eV, at 8 eV: same maps to converge (14, 8),
equal or fewer τ pairs (MoS2 333 → 321), semicore QP within 27 meV of DFT
(qp: 0.1–6 eV deeper), and the protected states' η_semi 8 − 5 spread falls
3–6× (E_F ± 10 eV std Fe 5.1 → 1.4, MoS2 5.0 → 0.8 meV). The default is a
no-op on a run without a coarse class (static modes, an `nval`
that covers every occupied band).

## 2026-09-29 — coarse (semicore) windows certified at max(ε, 3e-3)

Self-consistent decks with a coarse class move once. The coarse windows are
certified at max(`sigma_quadrature_eps`, 3e-3) (`qp_support.SEMICORE_EPS`),
so at the default ε 1e-4 they take 3e-3; every other Σ
window keeps `sigma_quadrature_eps`. At
η_semi 1 eV against ε 1e-4: map-2 τ pairs MoS2 3×3 530 → 465, Fe 4³ charge
1087 → 982; states within E_F ± 10 eV move ≤ 0.06 meV at maps 0–1 (≤ 1.9 meV
at map 2 of the unconverged Fe run); semicore QP ≤ 4.3 meV at map 0. The
planner's node law for grouping coarse windows is now evaluated on the box
each run is built on, so it equals the certified count.

## 2026-09-29 — the production QSGW partition: counted b3, semicore Σ read class

Every dynamic self-consistent deck with a coarse class (`qp_solver =
self_consistent`, scalar MPA/shared-pole route) moves once. See
[self-consistency](docs/self_consistency.md#2-band-treatment).

- **b3 counts bands, as before.** b3 = nelec + `ncond`. The QP matrix
  [b0, b3) rotates among itself;
  [b3, number_bands) is the scissored tail (DFT ψ, rigid shift, no Σ, no
  mixing). The ζ fit is unchanged. The classes below change only where
  Σ_c(ω) is read.
- **One request key, `number_bands_protected`** (the documented form): every
  occupied band plus conduction bands up to that total. Its semicore (coarse)
  class is every occupied band below a ≥ 4 eV band gap. The `nval` / `ncond`
  form stays: there the coarse class is every occupied state below the
  lowest requested valence band (a smaller `nval` moves more valence states
  onto the coarse windows). Giving both forms refuses
  (`GATE band_request_forms`). A dipole artifact is stamped with the request
  window, so a deck switched to `number_bands_protected` needs a dipole
  written with `nval` = the occupied count.
- **Semicore moves to coarse windows.** On the scalar MPA/shared-pole route the
  coarse states are read at their own energy on held windows at η_semi = 5 eV
  (one per coarse manifold; the Σ plan groups them to the least closed-form
  node count) instead of at Σ(ω = 0) (below E_F − 15 eV) or on the near grid
  at the deck η, certified at max(`sigma_quadrature_eps`, 3e-3).
  `sigma_omega_patches_ev` accepts `lo:hi:eta` triples as user coarse windows
  (`GATE sigma_coarse_window`).
- **Continuous tail weights.** The scissored tail's rigid shift Δ_c weights
  each conduction state by min(Z, 1/Z) (0 for Z ≤ 0) instead of Z inside a
  hard Z ∈ (0, 1] cut, so the tail law has no jump where a state's Z crosses 1.
- **A new refusal.** `zeta_nband` below b3 now refuses on every run,
  one-shot included (`GATE qp_matrix_zeta_left`; it was a warning).

## 2026-09-29 — the planners check their compiled executables

- Each chosen chunk's executable is checked against its planner's price
  before it runs (`runtime.aot_memory.check_chunk`; the response direct
  stream's sample group, the ζ μ batch, and the Σ τ window's price). When a
  chunk is over its room, it is recompiled once at a corrected size. If it is
  still over, the run warns and continues (since 2026-10-01; it refused before).
- The shared-pole response group must also fit the device room, which counts
  resident bytes the capacity ledger does not own. A deck whose group was
  sized into that gap gets a smaller group and moves once. Fe 8³ charge P4 at
  36 GB keeps its group of 16 and is bitwise. With the latency-hiding flag it
  runs 15 and peaks at 34.55 GB instead of 36.04 GB.
- Leave at least 5 GB between `memory_per_device_gb` and the card for NCCL,
  the CUDA context and cuSOLVERMp. See
  `docs/architecture/memory-model.md` for what compiled statistics miss.

## 2026-09-29 — shared-pole χ₀ direct stream through mathdx mode 11

- The shared-pole bank's direct stream (charge, metal or insulator, on a
  raw-parent plan) forms each node's correlation with mathdx mode 11 from the
  parent Greens; no full-k Green is built. Decks whose response groups keep
  their size move at round-off (Fe 4³ charge SC ≤ 0.2 µeV).
- The freed memory lets the response group grow, and the group size still
  follows `memory_per_device_gb`. A deck whose group grows gets a different
  shared-node rule, with the same certified accuracy, and moves once: Fe 8³
  charge SC groups go from 2 to 16 samples, Green pairs per map from 344 to
  100, the held map from 335 to 276 s, and eqp0 within E_F ± 10 eV by at most
  1.28 meV (median ≤ 13 µeV).

## 2026-09-28 — no quadrature rule is stored across runs; `sigma_quadrature_cache_dir` is retired

Every run places its own Σ(ω) and χ quadrature rules and reuses them only
inside the run. The deck key `sigma_quadrature_cache_dir` refuses by name:
remove it. `LORRAX_MINIMAX_CACHE_DIR` and `LORRAX_DISABLE_MINIMAX_DISK_CACHE`
are no longer read: unset them. The directories
`$SCRATCH/.cache/lorrax/sigma_box_rules` and
`~/.cache/lorrax/minimax_quadratures` are no longer used and can be deleted.
Results equal those of a run that started with an empty cache.

## 2026-09-28 — band extrapolation on the shared-pole Σ

- Scalar `compute_mode = mpa` (shared pole or MPA fit) now extrapolates the Σ_c
  band sum with the same brackets and pooled (β, Ω) fit as GN/HL-PPM.
  `use_band_extrapolation` defaults on, so every scalar shared-pole run moves
  once, and a deck with `number_bands_sigma` < 2·n_occ now refuses at startup,
  as GN/HL-PPM already does: for example Fe 4³ scalar at 35 bands (needs 36)
  and MoS2 at 44 bands (needs 52). Raise the band count or set
  `use_band_extrapolation = false` there. Bispinor `mpa` is unchanged.

## 2026-09-28 — band extrapolation: pooled denominator shell, cuts at 70/85/100 %

- `spectral_shell` now fits one (β, Ω) over the QP window's states: band A adds
  a_i·Σ_k w_k (E_Ak − E_i + Ω)^−β to state i, with a per-state amplitude from
  the widest shell. The per-state exponent is gone. Every GN/HL-PPM run with
  `use_band_extrapolation` on moves once. On Si 4³ at 78 bands against the
  complete basis the std over the ±10 eV states drops from 109 meV (per-state,
  cuts 64/72/78) to 9.2 meV (pooled, cuts 50/64/78); see
  [Band extrapolation](docs/theory/band-extrapolation.md).
- `total_fractions` cuts are 70 % and 85 % of `number_bands_sigma` (were 80 % and
  90 %), so the three bracket counts change and the Σ τ-loop recompiles once.
- `sigma_mnk.h5`: `sigma_c_extrap_beta_kn` holds the pooled β (NaN on states
  without a tail); new attributes `pooled_beta`, `pooled_omega_ev`,
  `pooled_residual_rms_ev`, `pooled_state_count`.

## 2026-09-26 — `occupation_window_threshold` is retired; one occupation support for χ and Σ

The deck key `occupation_window_threshold` refuses by name: remove it. A band
belongs to a Green's-function branch of χ or Σ if and only if its weight
(f on the occupied branch, 1 − f on the empty one) is at least 1e-5 in
magnitude, which for Fermi–Dirac occupations is 11.5 k_BT from μ
(`gw.efermi.band_in_occupation_window`). The one-shot and every SC map use
this one support; the previous floor was 0.005 (5.3 k_BT), which dropped
states that still carry weight. Metal decks move once; insulators, whose
weights are 0 or 1, are unchanged.

## 2026-09-26 — `band_extrapolation_estimator = band_index_only` is retired

The value `band_index_only` refuses by name. Remove the key, or set
`band_extrapolation_estimator = spectral_shell`, the default and only accepted
value ([band extrapolation](docs/theory/band-extrapolation.md)). The
band-index fit S(N) = S_∞ + A/N ignores where the omitted bands lie in
energy. Decks that did not name the value are unchanged.

## 2026-09-24 — FFI handler ABI 6; a native library of another ABI refuses

The Python tree and its native FFI handlers agree on handler ABI 6
(`src/ffi/cpp/common/lorrax_ffi_abi.h`, mirrored by
`ffi.common.ffi_loader.LORRAX_FFI_ABI_VERSION`). A sealed native bundle whose
manifest records another `ffi_abi`, or a loose `liblorrax_ffi*.so` that stamps
another ABI, refuses at startup before any handler is called ("native bundle
ABI mismatch" or "HANDLER ABI MISMATCH"), because a mismatched handler
otherwise fails later with an argument-count error that names neither
library. A bundle's CUDA and host legs are used together:
`LORRAX_FFI_SO` and `LORRAX_FFI_HOST_SO` must both be unset or both select
one bundle, and a partial or mixed override refuses. What to change: on
Perlmutter, members of m4598 load the current `lorrax_A` module
([installation](docs/installation/perlmutter.md#module)); elsewhere rebuild and reseal
both legs from this source tree ([installation](docs/installation/index.md)).
Do not point either variable at a library built from older source.

## 2026-09-24 — k-axis convolutions on nvidia-mathdx

- **NVIDIA GPUs now require the `nvidia-mathdx` wheel** (pinned in the
  `cuda12`/`cuda13` extras; header-only).  Without it a CUDA run refuses at
  startup with `GATE mathdx-headers` and the fix `pip install nvidia-mathdx`.
- `LORRAX_FFT_FFI_FUSED`, `LORRAX_CONV_KMINOR_FFI` and `LORRAX_CONV_KLEAD_FFI`
  are gone: Σ, COHSEX and the BSE ladder/stack convolutions have one route
  per platform (nvidia-mathdx on CUDA, the host FFTW plans on cpu).  A leftover
  setting is ignored.
- The kernels compile on first use (about 6 s per k-grid) and are kept in
  `$SCRATCH/.cache/lorrax/kconv_mathdx` (else `~/.cache/lorrax/kconv_mathdx`);
  the second run of a deck loads them in ~10 ms.
- The flat-k transform (χ₀, head, htransform) also runs on nvidia-mathdx on
  CUDA.

## 2026-08-28 — startup ownership, BSE mesh flags, emulated CPU meshes

- The runtime owns `JAX_ENABLE_X64`: it applies the resolved value even when
  jax was imported before the driver, and a resolved `False` refuses at
  startup. `LORRAX_ALLOW_X64_OFF=1` continues as an announced uncertified
  run. The per-driver `jax.config.update` lines are gone.
- Drivers no longer arm the persistent compile cache; step 7 of
  `runtime.initialize_communicator_stack` owns it. With `ISDF_JAX_CACHE_DIR`
  unset it is on, in one namespace per source release under
  `$SCRATCH/.cache/lorrax/jax_compile`, with JAX's write threshold at 0;
  `ISDF_JAX_CACHE_DIR=""` turns it off.
- BSE-family drivers: omitted `--px/--py` now means the run's canonical
  square mesh (it used to mean 1×1). An explicit shape must consume the
  job's device count exactly — under- and over-requests both refuse.
- `gw_jax`, `kin_ion_io`, `downfold_cli` and `kmeans_cli` answer `--help`
  and bad argv before any runtime exists (`runtime.cli_seam`); the other
  four drivers still pay full bring-up first.
- Single-process multi-device CPU meshes
  (`XLA_FLAGS=--xla_force_host_platform_device_count=N`) now run end to
  end: `SlabIO` serves them through an announced serial tier
  (`file_io._slab_io_serial`, CPU only). The `p*q == process_count`
  refusals stand everywhere else.

## 2026-08-18 — retired HDF5 controls now refuse or are absent

- GW decks containing `slab_io` or `use_ffi_io` now refuse with a targeted
  removal message. Remove either line; SlabIO has one collective transport.
- kmeans no longer accepts `--use-phdf5`; `WfnLoader` selects its one valid
  scalable read path from runtime capability.
- SlabIO no longer accepts `chunks=`. The argument was ignored and every
  collective dataset was already contiguous. The sigma and zeta writers no
  longer request a layout the native create cannot produce.
