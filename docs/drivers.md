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

Every driver brings up the runtime with one module-top call,
`runtime.initialize_communicator_stack` (mesh, communicators, startup report).
`gw_jax`, `kin_ion_io`, `kmeans_cli` and `downfold_cli` answer `--help` and bad
arguments before that call (`runtime.cli_seam`). Launch geometry is owned by
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
$\langle mk|\hat v_a|nk\rangle = \langle mk|2(k+G)_a + s\,\partial V_\mathrm{NL}/\partial k_a|nk\rangle$
(Ry atomic units) for every full-BZ $k$ and Cartesian $a$, with the nonlocal
term from the analytic $k$-derivative of the projectors. The sign $s$ is
`vnl_velocity_sign` (deck key or `--vnl-velocity-sign`; unset resolves to
`+1`, stamped as `prov_vnl_velocity_sign`). Consumed by `gw.head_correction`
for the $q\to0$ head $S(\omega)$ of Σ and by the BSE. The operator, its sign,
the links and the artifact are owned by
[the velocity operator](theory/qp-velocity.md).

Reads the deck (`wfn_file`, `nval`, `ncond`, `nband`, `bispinor`) and the
`*.upf` files (deck directory, then `../qe/scf`, `../qe/nscf`). Writes
`dipole.h5` (`file_io.dipole`): `dipole_cart` `(3, nk, nb, nb)`,
`band_energies` `(nk, nb)` (readers derive $E_b - E_{b'}$), and root provenance attributes (`prov_wfn_sha256` and its
fingerprint scheme, `prov_{nval,ncond,nband,nb_written,wfn_file}`, the
representation, V_NL and $q\to0$-operator schemes). `check_dipole_provenance`
refuses a file whose WFN, band extent, representation or V_NL convention does
not match the run, or whose operator scheme is unstamped; the fix is to rerun
this driver. The sweep shards bands over the mesh and SlabIO writes the
velocity from those shards; no rank gathers the table.

Invoke: `python3 -m psp.get_dipole_mtxels -i deck.in [--out dipole.h5]`.

| flag | default | meaning |
|---|---|---|
| `--vnl-mode` | `analytic` | nonlocal velocity by analytic $dZ/dK$, or `numeric` finite differences (`--vnl-h`, `--vnl-h-rel`, `--vnl-num-scheme naive\|richardson`) |
| `--skip-vnl` | off | write $\hat p$ only (BerkeleyGW `use_momentum`) |
| `--vnl-velocity-sign` | deck, else `+1` | relative sign of $i[r,V_\mathrm{NL}]$ |
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
| `sigma_mnk.h5` | dynamic modes: the Σ(ω) band matrices |
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
| `write_eqp2` | false | dynamic one-shot: iterate the fixed Σ(ω) matrix in the evolving QP basis and write `eqp2.dat`; screening and Σ are not rebuilt. Keys `eqp2_*` in the [input reference](input_reference.md) |
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

## downfold — `gw.downfold_cli`

Compresses a finished GW restart onto $\mu_S$ of its $\mu_L$ centroids for the
retained BSE band window and writes a restart bundle in the unchanged format,
which both BSE drivers read unmodified. Invoke:
`python3 -u -m gw.downfold_cli -i downfold.in`.
[The downfold page](downfold.md) owns the equations, the procedure, the
outputs and the refusals; the `[downfold]` keys are in the
[input reference](input_reference.md#downfold-the-downfold-input-file).

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

Invoke: `python -u -m bse.bse_jax -i cohsex.in --lanczos --tda --bse ...` in the GW
run directory. The mesh is the run's square startup mesh; `--px`/`--py` must
be square and use every device. The CLI is strict: an unknown flag refuses,
and so does a flag the chosen route (Lanczos, `--kpm-dos`, or the default
FEAST) does not read — `--eqp`, `--n-eig`, `--n-occ` and the solver flags are
Lanczos-only.

| flag | default | meaning |
|---|---|---|
| `--lanczos` | off | Krylov eigensolve; without it the driver runs FEAST (`bse_feast`) |
| `--bse` / `--rpa` | RPA | `--bse` adds $-W$ |
| `--tda` | off | resonant block only; without it every route solves the full BSE, and `--lanczos` goes to the dense `bse_nontda` solver, ignoring `--solver`, `--block-size` and `--n-reorth` |
| `--n-val` / `--n-cond` / `--n-occ` | 4 / 4 / the WFN's `ifmax` | transition window; the occupied-band count is `--n-occ` (*Lanczos*), else the deck WFN's `ifmax` |
| `--band-degeneracy` | `strict` | *Lanczos*. A window edge inside a multiplet: `strict` refuses and names working counts, `snap` widens outward, `off` proceeds; tolerance `--degeneracy-tol-ry` (1 meV) |
| `--solver` | `lanczos` | *Lanczos*. `lanczos` (spectrum shape), `davidson` (per-state convergence, `--davidson-*`), `trlan` (thick restart, bounded memory, `--trlan-*`) |
| `--block-size` / `--max-lanczos-iter` / `--n-reorth` | 1 / auto / −1 | *Lanczos*. Block width / total Krylov dimension / reorthogonalization window (−1 = full, needed for degenerate spinor spectra) |
| `--n-eig` / `--write-eigs [N]` | 5 / off | *Lanczos*. Eigenpairs; write `eigenvectors.h5` |
| `--dipole FILE` | none | *Lanczos*, with `--write-eigs` and `--tda`: store each written state's dipole $\langle 0\lvert\hat r\rvert S\rangle$ from `FILE` (a `dipole.h5`) in `eigenvectors.h5` |
| `--eqp FILE` | none | *Lanczos*. Diagonal QP energies from the wedge `eqp1.dat`, unfolded through the symmetry service; the restart must be proved to come from the same unrotated WFN |
| `bse_k_grid` (deck) | `""` | fine grid "NX NY NZ", each axis at least the coarse extent |
| `head_minibz_average` (deck) | false | read only under `bse_k_grid`: rebuilds the q = 0 exchange tile with the fine grid's mini-BZ head and takes the W head's Γ-cell reference from the analytic sphere ([LT head](theory/lt-exchange-head.md)); must match the GW run |

Forgetting `--bse` gives RPA. Absorption comparisons with BerkeleyGW:
[BerkeleyGW users](how-to/berkeleygw-users.md#bse).

## exciton bands — `bse.exciton_bands`

Computes the finite-momentum exciton dispersion $E_S(Q)$ of the TDA
Hamiltonian $H_Q = D_Q + V_Q - W$ in the basis $|vk \to c\,k{+}Q\rangle$ along
the deck's `K_POINTS {crystal_b}` path. The shifted conduction states come
from htransform, W is the coarse-grid convolution of the GW restart, and the
exchange tile at off-grid $Q$ comes from `bse.vq_interp`. The whole path is
one compiled `lax.scan`. Writes `<prefix>.dat` and `<prefix>.png` (rank 0).

Invoke: `python -u -m bse.exciton_bands -i cohsex.in --n-val 4 --n-cond 4 --n-eig 6`.

[Band interpolation and exciton bands](how-to/htransform-and-exciton-bands.md#7-exciton-bands)
owns the method, the exchange routes (`--vq-mode`) and their certification,
the flags, the output format and the refusals.
