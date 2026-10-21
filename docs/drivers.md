# Driver reference

LORRAX has six core drivers in seven modules: GW preprocessing is two modules
(`psp.get_dipole_mtxels` and `gw.kin_ion_io`). In chain order:

| stage | module | produces |
|---|---|---|
| ISDF points | `centroid.kmeans_cli` | `centroids_frac_<N>.txt` |
| velocity operator | `psp.get_dipole_mtxels` | `dipole.h5` |
| one-body Hamiltonian | `gw.kin_ion_io` | `kin_ion.h5` |
| GW | `gw.gw_jax` | `eqp0.dat`, `eqp1.dat`, `qp_wfn_rotations.h5`, the restart bundle |
| band interpolation | `bandstructure.htransform` | `bandstructure.dat` ([how-to](how-to/htransform-and-exciton-bands.md)) |
| optical BSE | `bse.bse_jax` | exciton energies, `eigenvectors.h5` |
| exciton dispersion | `bse.exciton_bands` | `E_S(Q)` table and plot |

[`gw.downfold_cli`](#downfold-gwdownfold_cli) sits beside the chain: it
compresses a finished GW restart for the two BSE drivers.
[`tdagw.tdgw_jax`](#td-agw-tdagwtdgw_jax) also sits beside it: it propagates
real-time TD-aGW on the header of a TDGW run.
[`tdagw.kernel_export` and `tdagw.tdgw_bridge`](#tdgw-kernel-and-bridge-tdagwkernel_export-tdagwtdgw_bridge)
sit beside it too: they hand the BSE interaction to TDGW's real-time propagation.
[`gw.epsmat_cli`](#epsmat-gwepsmat_cli) also sits beside it: it converts a
finished bundle's static W to and from BerkeleyGW's ε⁻¹ files.
So does [`bse.bsemat_cli`](#bsemat-bsebsemat_cli): it writes a
finished bundle's static W as BerkeleyGW 4.0's `bsemat.h5`.

Every driver brings up the runtime with one module-top call,
`runtime.initialize_communicator_stack` (mesh, communicators, startup report).
`gw_jax`, `kin_ion_io`, `kmeans_cli`, `downfold_cli`, `tdgw_jax`, `epsmat_cli` and
`bsemat_cli` answer `--help` and bad arguments before that call
(`runtime.cli_seam`). Launch geometry is owned by
the machine pages ([Perlmutter](environment/machines/perlmutter.md),
[Frontera](environment/machines/frontera.md)); every deck key is in the
[input reference](input_reference.md). The tables below list only the keys
and flags that change what a driver computes.

## centroids — `centroid.kmeans_cli`

Selects the $N_\mu$ real-space interpolation points $r_\mu$ of the ISDF
factorization $\psi^*_m(r)\psi_n(r) \approx \sum_\mu \zeta_\mu(r)\,
\psi^*_m(r_\mu)\psi_n(r_\mu)$ ([theory](theory/physics.md)). Weighted k-means
runs on the real-space FFT grid with the importance

$$w(r) = \Big[\sum_k w_k \sum_{m\in L,\,n\in R} |\psi^\dagger_{mk}(r)\psi_{nk}(r)|^2\Big]^{1/2},$$

unit band weights over the left/right fit windows $L, R$ (occupations do not
enter), for $\lceil N_\mu \cdot \text{oversample}\rceil$ candidates. Pivoted
Cholesky on the pair-density Gram over the same windows then prunes the pool
to $N_\mu$; every emitted point updates the Schur residual. In `current` mode
the feature rows are the three Dirac-current components and the Gram is the
transverse one. With orbit mode the candidates are closed under the decorated
atoms' spatial Seitz group (improper and nonsymmorphic operations included) and
pruning admits only whole orbits that fit the budget. Numerical flatness of the
pool is reported, not refused, because an over-complete interpolation set can
be accurate; non-PSD input, pool exhaustion and invalid pivots refuse
([rank policy](architecture/rank_truncation_policy.md)). The header's `pool
rank=` line names the stop: `pool spent` (no unpicked candidate's residual
is above the floor, so the count is the pool's rank, a lower bound on the
pair-set rank) or `point budget` (the pool still held directions; the rank is
not measured and kmeans warns `CentroidRankNotEstablished`). For the
charge channel a `rank law:` line gives the
[rank law](theory/isdf-exchange-accuracy.md#rank-law)'s $N_\mu$ for 1 meV RMS
Σ_x beside the written count.

Reads the deck's `wfn_file` (`-i`; relative to the deck), else `WFN.h5` in the
working directory. Writes `centroids_frac_<n>[<suffix>].txt`
(fractional coordinates, with a provenance header) and `kmeans[<suffix>].out`.
The GW deck names the table with `centroids_file`, and the current-channel table
with `centroids_file_current`. The GW run guards reuse by the table's content
hash, so regenerating centroids invalidates its ζ and restart tensors.

Invoke: `python3 -m centroid.kmeans_cli N [--density-mode current] [--orbit]`.
The projector sweep partitions IBZ parents across ranks, so useful parallelism
is capped by the parent count; the full-grid projector carriers are priced and
refused before loading if they do not fit.

| flag | default | meaning |
|---|---|---|
| `N_c` (positional) | 400 | centroids after pruning |
| `--oversample` | 1.5 | k-means candidate factor; 1.0 disables pruning |
| `-i` / `--input` | unset | GW deck; its `ncond` sets the Σ conduction window on the `v_x_vc` left leg |
| `--prune-n-val` / `--prune-n-cond` | `nelec` / `nbands − n_val` | pruning window extents; `--prune-n-val` below `nelec` refuses |
| `--prune-window` | `v_x_vc` | Gram band pair: `v_x_c`, `v_x_vc` (left = occupied + Σ conduction from `-i`, right = all bands; without `-i` it falls back to `vc_x_vc` and says so), `vc_x_vc` (all bands on both legs) |
| `--fit-window L0:L1,R0:R1` | unset | explicit left/right windows for candidates and pruning |
| `--density-mode` | `scalar` | `scalar` charge Gram; `current` transverse three-current Gram (needs orbit closure, `--rho-power 1`; suffix `_current`) |
| `--orbit` / `--no-orbit` | on if the atom group has more than one operation | orbit-closed selection |
| `--rho-power` | 1.0 | sampling weight $w^\alpha$; point density $\propto w^{3\alpha/5}$ |
| `LORRAX_CENTROID_SELECT` (env) | `deliver` | `strict` refuses a numerically flat pool |

Both windows must start at band 0 and hold every occupied band; a window that
drops one refuses (`CentroidWindowDropsOccupiedError`), here and when `gw_jax`
reads the table's header. A header whose left window stops below the Σ
conduction window (the old `v_x_vc` default) only warns.

## dipole — `psp.get_dipole_mtxels`

Computes the velocity matrix elements
$\langle mk|\hat v_a|nk\rangle = \langle mk|2(k+G)_a + \partial V_\mathrm{NL}/\partial k_a|nk\rangle = \langle mk|i[H, r_a]|nk\rangle$
(Ry atomic units) for every full-BZ $k$ and Cartesian $a$, with the nonlocal
term from the analytic $k$-derivative of the projectors. Consumed by
`gw.head_correction` for the $q\to0$ head $S(\omega)$ of Σ and by the BSE. The
operator, the links and the artifact are owned by
[the velocity operator](theory/qp-velocity.md).

Reads the deck (`wfn_file`, `nval`, `ncond`, `nband`, `bispinor`) and the
`*.upf` files (deck directory, then `../qe/scf`, `../qe/nscf`). Writes
`dipole.h5` (`file_io.dipole`): `dipole_cart` `(3, nk, nb, nb)`,
`band_energies` `(nk, nb)` (readers derive $E_b - E_{b'}$), and root provenance attributes (`prov_wfn_sha256` and its
fingerprint scheme, `prov_{nval,ncond,nband,nb_written,wfn_file}`, the
representation, V_NL and $q\to0$-operator schemes) and `format_version` (2).
Every reader refuses a file without format version 2 (`GATE
dipole_format_version`), and `check_dipole_provenance` refuses a file whose
WFN, band extent, representation or V_NL inclusion does not match the run, or
whose operator scheme is unstamped; the fix is to rerun this driver. The sweep shards bands over the mesh and SlabIO writes the
velocity from those shards; no rank gathers the table.

Invoke: `python3 -m psp.get_dipole_mtxels -i deck.in [--out dipole.h5]`.

| flag | default | meaning |
|---|---|---|
| `--vnl-mode` | `analytic` | nonlocal velocity by analytic $dZ/dK$, or `numeric` finite differences (`--vnl-h`, `--vnl-h-rel`, `--vnl-num-scheme naive\|richardson`) |
| `--skip-vnl` | off | write $\hat p$ only (BerkeleyGW `use_momentum`) |
| `--pseudo-dir` | deck directory | where the `*.upf` live |
| `--with-finite-q` / `--iq-list` | off / all | also write the `finite_q/` group (`rho_cvkq`, symmetrized `v_cvkq`, `kminq_idx`); its conduction axis is sized by the producer's `ncond` |
| `--parallel-transport-out` | `parallel_transport.h5` beside `--out` | the link and velocity artifact read by `sc_head_update` ([its schema](theory/qp-velocity.md#9-the-artifact-parallel_transporth5)). `--no-parallel-transport` skips it; `--parallel-transport-velocity-only` writes only the velocity stage |
| `--parallel-transport-bands` | `0`: min(WFN bands, ⌈1.25 × deck bands⌉) | outer band set of the links ([why](theory/qp-velocity.md#4-links-and-the-covariant-derivative)) |
| `--parallel-transport-validation-rtol`, `--parallel-transport-rcond` | 5e-3, 1e-10 | the link-error warning threshold and the polar-factor cutoff ([link error](theory/qp-velocity.md#6-the-link-error-and-what-it-means)) |

## kin-ion — `gw.kin_ion_io`

Writes $\langle mk|T + V_\mathrm{loc} + V_\mathrm{NL}|nk\rangle$ to `kin_ion.h5`.
Hartree is not in it: GW builds $V_H$ live ([theory](theory/hartree.md)). The
operator commutes with the space group and time reversal, so it is computed on
the orbit parents and unfolded through the symmetry service.

Invoke: `python3 -m gw.kin_ion_io -i deck.in [-o kin_ion.h5] [-n NB]`. The deck
owns the system, band window and spinor settings; `WFN.h5` and the
pseudopotentials are stamped as provenance, and a GW run refuses a file whose
k storage, spinor representation, system dimension, band extent, WFN, input or
pseudopotential stamp differs from its own. The sweep shards bands over the
mesh and `file_io.kin_ion.write_kin_ion` writes the star-wedge slab from those
shards through SlabIO; no rank gathers it.

| key / flag | default | meaning |
|---|---|---|
| `-n` / `--nb` | b3 = nelec + ncond, the window GW reads (`number_bands_protected` resolved as GW resolves it) | bands written (ψ(G) is loaded for each); refuses below b3 |
| `--sys_dim` | the deck's | may only confirm the deck; a contradiction refuses |
| `--pseudo_dir` | deck directory, then `../qe/{scf,nscf}` | `*.upf` location |
| `kin_ion_file` (deck) | `kin_ion.h5` | the file GW reads |

A launcher that starts P tasks while `jax.distributed` joins a world of one
refuses, because every rank would compute and write the whole file.

## gw — `gw.gw_jax`

Computes quasiparticle energies from the ISDF-compressed GW self-energy. The
stages, in `main()` order:

1. ζ fit and bare Coulomb: $\zeta_{q\mu}(r)$ by least squares from the
   pair-density Gram, then $V_{q,\mu\nu}$ in G-space
   ([ISDF and $V_q$](theory/isdf-zeta-vq.md), [face-ψ fit](architecture/zeta_fit_face_psi_cct.md)).
2. Screening: $\chi_0$ on a minimax imaginary-time grid built on G's spectral
   range, then the per-$q$ Dyson solve $W = (1 - V\chi_0)^{-1} V$ at the
   frequencies the Σ scheme requests ([minimax quadrature](theory/minimax-quadrature.md)).
3. Self-energy: $\Sigma_x \oplus \Sigma_c$ in the ansatz the deck selects,
   with the $q\to0$ head carried as a scalar channel through every stage
   ([Σ(ω) quadrature](theory/sigma-quadrature-problem.md),
   [multipole W](theory/THEORY_mpa_implementation.md),
   [metals](theory/metallic-mpa-screening.md)).
4. QP extraction per `qp_solver`, then $\mathrm{eigh}(T + V_\mathrm{ion} + V_H + \Sigma)$.

G(τ) is never materialized; it exists only as $\psi\psi^*$ phases inside the
χ₀ and Σ kernels.

**Inputs.** The deck, `WFN.h5`, `centroids_file`, `kin_ion.h5`, `dipole.h5`
(for the head); on bispinor runs also `centroids_file_current`.

**Outputs.** Formats, columns, units and band conventions of every file:
[outputs](how-to/berkeleygw-users.md#outputs).

| file | content |
|---|---|
| `eqp0.dat`, `eqp1.dat` | BerkeleyGW-format QP energies on the WFN wedge; in SC runs, the converged SC spectrum |
| `sigma_diag.dat` | the Σ diagonal decomposition per k and band |
| `eqp_g0w0.dat` | dynamic one-shot runs: $H_0 + \Sigma_{xc}(E_\mathrm{DFT})$ |
| `sigma_mnk.h5` | dynamic modes: the on-shell Σ band matrices Σ_mn(E_m), Σ_mn(E_n) and diagonals |
| `qp_wfn_rotations.h5` | the QP eigensystem $U$, $E_\mathrm{QP}$ with the source-WFN fingerprint, read by htransform, BSE and SC seeding |
| `WFN_qp.h5` | ψ rotated by U with QP energies (`write_wfn_h5`); a one-shot run on a symmetry-reduced WFN writes none |
| `tmp/zeta_q.h5` | ζ, plus `zeta_q_mu{1,2,3}.h5` on bispinor runs |
| `tmp/isdf_tensors_<N_mu>.h5` | the restart bundle with the static W0; the BSE input |
| `gwjax.out` | the run report |

`qp_wfn_rotations.h5` converts to a QP WFN through the same writer:
`python -m postprocess.rotate_wfn_to_qp WFN.h5 qp_wfn_rotations.h5`.

**Band windows** (`wavefunction_bundle.BandSlices` from `Meta.from_system`). The
Σ/QP window is $[0, n_\mathrm{elec} + n_\mathrm{cond})$. `nval` sets only the
interior edge $b_1 = n_\mathrm{elec} - n_\mathrm{val}$. The band-sum top is
`number_bands` (alias `nband`), rounded up to the world size with zero pads.

Invoke: `python -m gw.gw_jax -i gw.in`.

| key | default | meaning |
|---|---|---|
| `number_bands` / `nval` / `ncond` | 100 / 5 / 5 | χ₀ and Σ band-sum top / interior valence edge / Σ conduction count. `number_bands_chi` and `number_bands_sigma` split the two sums |
| `compute_mode` | `auto` | Σ ansatz: `x_only` \| `cohsex` \| `gn_ppm` \| `hl_ppm` \| `mpa`. `auto` infers from the legacy flags and never selects `mpa`. `gn_ppm` refuses metallic occupations ([input reference](input_reference.md)). Production: `mpa` with `sigma_w_model = shared_pole` ([production QSGW](how-to/production-qsgw.md)) |
| `qp_solver` | `auto` → `one_shot_dft` | `one_shot_dft`: full-matrix effective H with Σ at $E_\mathrm{DFT}$, Hermitian-symmetrized; `self_consistent`: the QSGW loop, Σ read at each map's own energies. No QP equation is solved; `fixed_point` is retired and refuses by name ([self-consistency](self_consistency.md)) |
| `sigma_omega_grid_debug` | false | debug: Σ_c on the whole ω grid instead of on shell (the `sigma_mnk.h5` Σ(ω) cube, the grid freq-debug table, `qp_omega0_ev`, the ±0.5 eV Z); see [on-shell Σ_c](theory/sigma-on-shell.md#grid-debug). `write_eqp2` and the `eqp2_*` keys are retired and refuse by name |
| `restart` | false | `true` reuses `tmp/isdf_tensors_<N_mu>.h5` after authentication |
| `linalg` | `local` | the one layout dial for dense solves ([input reference](input_reference.md)) |

**Reuse.** Two caches, each refusing or refitting on mismatch, never reusing
wrongly.
(1) `tmp/isdf_tensors_<N_mu>.h5` is reused under `restart = true` only when
the band window, $N_\mu$ and k-grid attributes match and the centroid content
hashes (`centroids_charge_md5`, `centroids_transverse_md5`) match: same count,
different points refuses. Its QP-state record carries the WFN fingerprint, so a
changed WFN refuses at every state join.
(2) `tmp/zeta_q.h5` is reused when `zeta_is_done` is set, the stored centroid
table equals this run's, and the `fit_provenance` JSON is byte-identical (band
ranges, pair-training domain, cutoffs, effective `zeta_rcond`/`zeta_ridge`, FFT
grid, cutoffs, WFN identity). Mesh and P are excluded: a fit at P=4 is reusable
at P=80. `LORRAX_FORCE_REFIT=1` forces a refit.

Operating at thousands of centroids (the distributed plan, per-rank scalings):
[dense solves on P devices](architecture/dense_linear_algebra.md#stage-plans). Environment variables:
[registry](reference/env_vars.md).

### Contour-deformation frequency integration (`sigma_frequency_integration = contour_deformation`)

The one dynamic-Σ dispatch (`gw.sigma_dispatch._compute_mpa_sigma`) and the one finalizer
(`finalize_dynamic_sigma`) are shared with the pole models; only the body kernel changes.
`gw.cd.sigma.compute_sigma_c_cd_omega_grid` streams the ISDF W(z) at the CD nodes and knots, built
from the shared-pole bank's response-quadrature χ⁰ (`gw.cd.sigma.bank_chi0`), and accumulates Σ_c on
the file k wedge inside one `shard_map` over q; W(z) is never persisted.  Besides `sigma_mnk.h5` the
run writes `cd_sigma_own_energy.npz`, `cd_sigma_receipt.json` and `cd_leak_ledger.csv`; with
`cd_checkpoint = write | resume` also `tmp/cd_checkpoint.h5`, and under `LORRAX_CD_STOP_AFTER`
`cd_stop_receipt.json`.  The χ⁰ rules are built in the job (written to `cd_chi0_rules.pkl`) or
precomputed on a CPU node with `python -m gw.cd.chi0_rules ... --out FILE` and read through
`cd_chi0_rules_file`.  Keys: [input reference](input_reference.md#sigma); equations, owners,
memory, rules and refusals: [dev note](dev/cd_frequency_integration.md).

## downfold — `gw.downfold_cli`

Compresses a finished GW restart onto $\mu_S$ of its $\mu_L$ centroids for the
retained BSE band window and writes a restart bundle in the unchanged format,
which both BSE drivers read unmodified. Invoke:
`python3 -u -m gw.downfold_cli -i downfold.in`.
[The downfold page](downfold.md) owns the equations, the procedure, the
outputs and the refusals; the `[downfold]` keys are in the
[input reference](input_reference.md#downfold-the-downfold-input-file).

## epsmat — `gw.epsmat_cli`

Writes a finished GW bundle's static W as BerkeleyGW 4.0 `eps0mat.h5` /
`epsmat.h5` (`mode = export`), or projects BerkeleyGW's files onto the
bundle's ζ and writes a copy of the bundle whose `W0_qmunu` is theirs
(`mode = project`). Invoke: `python3 -u -m gw.epsmat_cli -i epsmat.in`
(`--print-schema` lists the keys). [The theory page](theory/bgw_epsmat_export.md)
owns the equations, the outputs and the refusals; the `[epsmat]` keys are in the
[input reference](input_reference.md#epsmat-the-epsmat-input-file).
`epsmat_ecut` (Ry, default 0 = the full sphere) sets the G cutoff of the
written files; a smaller value writes the leading sub-block of the converged
ε⁻¹, not a re-inversion ([theory §3](theory/bgw_epsmat_export.md#ecut)).


## bsemat — `bse.bsemat_cli`

Writes BerkeleyGW 4.0's kernel file `bsemat.h5` (TDA, or `extended_kernel` with
`--extended`) from a finished static-Dyson GW bundle. The head, wing and body at
every q come from the stored V, W and one ζ column per q, so no plane-wave pair
density is formed. Invoke:
`python3 -u -m bse.bsemat_cli -i cohsex.in --wfn-co WFN_co.h5 --n-val NV --n-cond NC --bgw-eps epsmat.h5`
on a square mesh. `--wfn-co` is the explicit full-grid WFN that BerkeleyGW's
absorption reads (`no_symmetries_coarse_grid`); the file is written in its
coefficient gauge and k order. `--bgw-eps` supplies the q list that fixes
BerkeleyGW's head slot at zone-boundary ties. The run prints BerkeleyGW's
`kernel.out` Frobenius lines and writes `<output>.json` with every gate's
measured value. `--mu-small` (default `off`) first downfolds the bundle onto the
file's band window (an integer, `kx` = k·N_b of the parent's centroids with
N_b = NV + NC, or `auto` for the eigenvalue-rank ceiling; `--downfold-rcond`) with
the [downfold](downfold.md) driver's selection and transfer, and builds the rows
by GEMMs over the small W; the theory page gives its measured accuracy. `--q0-body dipole` adds the
3-D q0 body term from the bundle's stored q → 0 wings. `--kernel`
(`--extended` = `--kernel extended`), `--gauge` (`--no-gauge-map` = `--gauge identity`),
`--q0-body` and `--mu-small` each take several values; one run then writes one file
per combination from a single load of the bundle, head split and gauge map, named
`<stem>_<tag><ext>` after `-o` (for example `--kernel tda extended` writes
`bsemat_tda.h5` and `bsemat_ext.h5`). Each `<file>.json` carries `stages`, the wall
seconds of every stage, and rank 0 prints the `common.timing` stage table at the end.
`--exciton-Q Q1 Q2 Q3` (crystal, default 0) writes the finite-momentum TDA kernel
(BerkeleyGW `exciton_Q_shift` = −Q): on the restart grid from the restart's own
states (`qflag` 2); off it, `--wfn-co` is the restart grid shifted by +Q, the
electron leg comes from the deck's htransform and `--wfnq-co` is the restart
grid's WFN (`qflag` 0; not yet validated, so it refuses unless the restart's ζ
covers the full zone). BerkeleyGW 4.0's absorption needs a compensating
`eqp.dat` at `qflag` 2 ([theory](theory/bgw-bsemat-export.md#finite-q)).
[The theory page](theory/bgw-bsemat-export.md) owns the equations, the conventions and the
refusals.

## htransform — `bandstructure.htransform`

Interpolates band energies from the coarse grid to the deck's
`K_POINTS {crystal_b}` path: the window's states are expanded in one
k-independent Galerkin basis, $f(H)_k = \sum_n f(\varepsilon_{nk})c_{nk}c_{nk}^\dagger$
is Fourier-interpolated, and $f$ is inverted at each path point. Optional
routes put QP energies on the path (`--qp-rotations` or `--eqp-file`), color
the bands by spin, orbital character or the modern-theory orbital moment
(`--color`, the last with `--velocity`), print the coarse per-cell orbital and
spin moments (`--velocity`) and sum the spin on a uniform grid (`--moments-grid`).

Reads the deck, `WFN.h5` (or `--wfn-file WFN_qp.h5`) and `centroids_file`.
Writes `bandstructure.dat` (eV, VBM at 0), `htransform.out` and the reusable
basis `galerkin_dft.h5`; rank 0 writes.

Invoke: `python -m bandstructure.htransform -i ht.in [--qp-rotations qp_wfn_rotations.h5 | --eqp-file eqp1.dat] [--color spin|orbital] [--velocity dipole.h5] [--moments-grid 40 40 40]`.

[Band interpolation and exciton bands](how-to/htransform-and-exciton-bands.md)
owns the method, every flag and key, the output formats and the refusals.

## bse — `bse.bse_jax`

Solves the Bethe–Salpeter equation for $Q = 0$ excitons in the transition
basis $|vk \to ck\rangle$: the full (non-TDA) problem by default, the resonant
block $A$ with `--tda`. `--rpa` (the default kernel) drops $W$; `--bse`
includes it. The Hamiltonian, the inputs, the screened W(0) handoff, the
matvec and its kernels, the solvers, the dipoles, the outputs and the refusals
are on [the BSE page](architecture/bse.md).

Invoke: `python -u -m bse.bse_jax -i cohsex.in --tda --bse ...` in the GW
run directory. The mesh is the run's square startup mesh; `--px`/`--py` must
be square and use every device. The CLI is strict: an unknown flag refuses,
and so does a flag the chosen route (the default eigensolve, `--feast` or
`--kpm-dos`) does not read — `--eqp`, `--qp-rotations`, `--n-eig`, `--n-occ` and the
solver flags are eigensolve-only (marked *Lanczos* below, after its old flag).

| flag | default | meaning |
|---|---|---|
| `--lanczos` / `--feast` | eigensolve | the eigensolve (`--solver`, Davidson by default) is the default route since 2026-10-21; `--lanczos` is a no-op alias. `--feast` runs FEAST windows (`bse_feast`, `--feast-iter` 2) and refuses unless every in-window Ritz pair's residual is below `--feast-tol-mev` (0.01 meV; `GATE feast_converged`), and when a window is saturated (every Ritz pair in-window: raise `--feast-ritz-count`) |
| `--bse` / `--rpa` | RPA | `--bse` adds $-W$ |
| `--tda` | off | resonant block only; without it every route solves the full BSE, and `--lanczos` goes to the dense `bse_nontda` solver, ignoring `--solver`, `--block-size` and `--n-reorth` |
| `--n-val` / `--n-cond` / `--n-occ` | 4 / 4 / the WFN's `ifmax` | transition window; the occupied-band count is `--n-occ` (*Lanczos*), else the deck WFN's `ifmax` |
| `--band-degeneracy` | `strict` | *Lanczos*. A window edge inside a multiplet: `strict` refuses and names working counts, `snap` widens outward, `off` proceeds; tolerance `--degeneracy-tol-ry` (1 meV) |
| `--solver` | `davidson` | *Lanczos*. `davidson` (default since 2026-10-20): block Davidson on the `--n-eig` lowest roots with the exact diagonal preconditioner and soft locking, stopped by its residual test; `lanczos` (fixed Krylov dimension, or `--lanczos-rtol`); `trlan` (thick restart, bounded memory, fixed matvec budget, `--trlan-*`). A flag only another solver reads refuses (`--block-size`, `--lanczos-rtol`, `--lanczos-check-every`, `--n-reorth` are Lanczos's; `--davidson-*` Davidson's; `--trlan-*` TRLan's) |
| `--davidson-margin` / `--davidson-margin-tol-mev` | 8 / `--davidson-tol-mev` | *Lanczos*, Davidson. Roots solved above `--n-eig` and not reported (at most half the transition space); the residual at which a margin root locks (stays in the basis, no further correction) |
| `--davidson-tol-mev` / `--davidson-precond` / `--davidson-m-max` | 0.01 / `exact` / 10·n-eig | *Lanczos*, Davidson. Stop when every root's residual $\lVert Hx - Ex\rVert$ is below the tolerance (meV; an exact eigenvalue lies within the residual, the error is typically residual²/gap); diagonal `exact` = assembled diag(H), `bare` = $E_c - E_v$; basis capacity before a restart |
| `--block-size` / `--max-lanczos-iter` / `--n-reorth` | 1 / auto / −1 | *Lanczos*. Block width / iteration budget: Lanczos's total Krylov dimension, Davidson's cap on block iterations, TRLan's matvec budget / reorthogonalization window (−1 = full, needed for degenerate spinor spectra) |
| `--n-eig` / `--write-eigs [N]` | 5 / off | *Lanczos*. Eigenpairs; write `eigenvectors.h5` |
| `--dipole FILE` | none | *Lanczos*, with `--write-eigs` and `--tda`: store each written state's dipole $\langle 0\lvert\hat r\rvert S\rangle$ from `FILE` (a `dipole.h5`) in `eigenvectors.h5`. Under `bse_k_grid`, `FILE` is the deck (coarse) WFN's, carried through the densifying htransform ([dipoles](architecture/bse.md#dipoles-and-absorption)); with `--eqp` the dipoles are $v/\Delta E$ of the fine states' Rayleigh quotients, and `--dipole-position-fix` (opt-in, CrI3-type QP ladders; the log's QP/DFT mixing receipt says when) makes them the exact position of the fine QP states; `absorption_haydock --dipole [--dipole-position-fix]` reads it the same way, and `python -m bse.dump_window_dipoles -i deck.in --n-val NV --n-cond NC [--n-occ N] --dipole FILE [--eqp EQP] [--dipole-position-fix] [--full-window] --out d.h5 [--check-eigenvectors eigenvectors.h5 --check-key d_alpha_rq]` writes them (and, with the fix, `d_alpha_rq`) in the run's gauge for re-contracting a saved `eigenvectors.h5` |
| `--haydock-iter N` | 0 (off) | *Lanczos*, with `--tda` and `--dipole`: after the eigensolve, N steps of the Haydock recursion of `bse.absorption_haydock` on the same payload and transformed kernel, in the same process; writes `--haydock-out-prefix` (`absorption_haydock`) `.h5` and `absorption_haydock_b<1-3>_eh.dat`. `--haydock-pol` (`xyz`) picks the Cartesian polarizations (one matvec each per step; `x` alone for a cubic crystal), `--haydock-eta-ev` (0.1), `--haydock-omega-min-ev`/`-max-ev`/`--haydock-n-omega` (0/15/1500), `--haydock-n-spin` (1), `--haydock-no-eps1` as the standalone driver's flags; the cell volume is the deck WFN's |
| `--helicity-table` | off | *Lanczos*, with `--dipole`: add each written state's in-plane circular strengths to `bse.out`, $I_\pm = \lvert D_x \mp i D_y\rvert^2/(2N_k)$ in bohr² per unit cell for $D = \langle 0\lvert\hat r\rvert S\rangle$, and $\eta = (I_+ - I_-)/(I_+ + I_-)$. σ+ is the field $E_0(\cos\omega t, \sin\omega t, 0)$: x → y, counter-clockwise seen from +z ([helicity](architecture/bse.md#dipoles-and-absorption)) |
| `--eqp FILE` | none | *Lanczos*. Diagonal QP energies from the wedge `eqp1.dat`, unfolded through the symmetry service; the restart must be proved to come from the same unrotated WFN |
| `--qp-rotations FILE` | none | *Lanczos*, native grid. Solve in the eigenbasis of the GW run's $H_{QP}$ from its `qp_wfn_rotations.h5`: rotated ψ at the centroids, $E_{QP}$, QP-basis dipoles; `--n-val`/`--n-cond` count QP states. Exclusive with `--eqp`; refuses `bse_k_grid` ([QP-rotated basis](architecture/bse.md#the-qp-rotated-basis)) |
| `bse_k_grid` (deck) | `""` | fine grid "NX NY NZ", each axis at least the coarse extent |
| `head_minibz_average` (deck) | false | read only under `bse_k_grid`: rebuilds the q = 0 exchange tile with the fine grid's mini-BZ head and takes the W head's Γ-cell reference from the analytic sphere ([LT head](theory/lt-exchange-head.md)); must match the GW run |

Forgetting `--bse` gives RPA. Absorption comparisons with BerkeleyGW:
[BerkeleyGW users](how-to/berkeleygw-users.md#bse).

## exciton bands — `bse.exciton_bands`

Computes the finite-momentum exciton dispersion $E_S(Q)$ of the TDA
Hamiltonian $H_Q = D_Q + V_Q - W$ in the basis $|vk \to c\,k{+}Q\rangle$ along
the deck's `K_POINTS {crystal_b}` path. The shifted conduction states come
from htransform, W is the coarse-grid convolution of the GW restart, and the
exchange tile at off-grid $Q$ comes from `bse.vq_interp`. Each $Q$'s lowest
roots come from `bse_lanczos`' Davidson (the default: one planned solver for
the path, residual stop, a $Q$ that does not converge refuses) or, with
`--solver lanczos`, from block Lanczos in one compiled `lax.scan`. Writes `<prefix>.dat` and `<prefix>.png` (rank 0).

Invoke: `python -u -m bse.exciton_bands -i cohsex.in --n-val 4 --n-cond 4 --n-eig 6`.

[Band interpolation and exciton bands](how-to/htransform-and-exciton-bands.md#7-exciton-bands)
owns the method, the exchange routes (`--vq-mode`) and their certification,
the flags, the output format and the refusals.

## td-agw — `tdagw.tdgw_jax`

Propagates the real-time TD-aGW equation of motion of TDGW (Xiaoxun Gong's
code) for one or several drives: $\dot\rho = -i[H(t), \rho] - i\mathbf
E\cdot[\mathbf r^\mathrm{intra}, \rho] - \Gamma\circ(\rho - \rho^\mathrm{eq})$
with $H = \mathrm{diag}(E^\mathrm{QP}) + \mathbf E\cdot\mathbf r +
\Delta\Sigma[\rho]$, in TDGW's units and conventions. It reads a TDGW deck,
takes energies, k grid and dipoles from the header of a TDGW run, and writes
`tdgw.h5` + `tdgw.toml` that `tdgw-pp` reads unchanged, and one run log,
`tdagw.out`, in `outdir` ([reading it](how-to/td-agw-run.md#log)). The equation, the
integrators, the layout and the output are on
[TD-aGW propagation](theory/td_agw_propagation.md); the whole chain, from the GW
run to a drive, is [TD-aGW end to end](how-to/td-agw-workflow.md).

Invoke: `python -u -m tdagw.tdgw_jax -i tdgw.toml`. The only flag is
`-i/--input`; everything else is a deck key
([input reference](input_reference.md#td-agw)). The mesh is the run's
startup mesh; the k axis is sharded over every device.

| deck key | default | what it changes |
|---|---|---|
| `control.files.load_header_from` | required | the TDGW run whose header (band window, k grid, energies, dipole) is propagated |
| `lorrax.method` | `dp54` | `dp54` takes TDGW's RK45 steps (the parity mode); `etdrk4`, `cfm4`, `magnus2`, `rk4` take a fixed `lorrax.dt` |
| `lorrax.model` | by `calc_type` | ΔΣ[ρ]: `none`, TDGW's `kernel.h5` (`tdgw_kernel`) or the ISDF model of a GW restart bundle (`isdf`, with `lorrax.isdf_input`) |
| `lorrax.trials` | one drive | several `{E_x, E_y, E_z}` drives propagated together; one `trial-<i>/` output each |
| `lorrax.intraband` | `true` | the intraband commutator $\mathbf E\cdot[\mathbf r^\mathrm{intra}, \rho]$ |

## TDGW kernel and bridge — `tdagw.kernel_export`, `tdagw.tdgw_bridge`

Both read a GW restart bundle through the BSE loader and build the TD-aGW
interaction $\Delta\Sigma = K\,\delta\rho$ (screened exchange plus Hartree,
$G = 0$ excluded) on a band window. `tdagw.kernel_export` writes $K$ as a
TDGW `kernel.h5` (Rydberg, TDGW's band order and the k order of its `WFN.h5`),
which an unmodified TDGW run reads in place of a BerkeleyGW kernel. At TDGW
41a5c34 only the TDA file (`--tda`) propagates natively: TDGW's
extended-kernel reader drops the run's parallel k axis. `tdagw.tdgw_bridge`
runs TDGW with its self-energy update replaced by the matrix-free operator,
the full kernel by default and TDGW's TDA semantics with `--tda`, so no dense
kernel is formed. Both refuse a `WFN.h5`
whose k-points are not the full bundle grid. On more than one rank the bridge
needs an h5py built against parallel HDF5, because TDGW writes its output
collectively, and refuses otherwise.

Invoke: `python -m tdagw.kernel_export --input cohsex.in --wfn WFN.h5 --ibndmin 129 --ibndmax 132 --ivbm 130 --out kernel.h5 [--terms both|sex|hartree] [--tda] [--receipt r.json]`
and `python -m tdagw.tdgw_bridge tdgw.toml --input cohsex.in [--terms both|sex|hartree] [--tda] [--receipt r.json]`.
[The TD-aGW interaction model](theory/td_agw.md) owns the equations, the
units, the kernel-file convention and the cost.
