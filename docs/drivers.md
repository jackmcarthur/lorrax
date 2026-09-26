# Driver reference

LORRAX has six core drivers in seven modules: GW preprocessing is two modules
(`psp.get_dipole_mtxels` and `gw.kin_ion_io`). In chain order:

| stage | module | produces |
|---|---|---|
| ISDF points | `centroid.kmeans_cli` | `centroids_frac_<N>.txt` |
| velocity operator | `psp.get_dipole_mtxels` | `dipole.h5` |
| one-body Hamiltonian | `gw.kin_ion_io` | `kin_ion.h5` |
| GW | `gw.gw_jax` | `eqp0.dat`, `eqp1.dat`, `qp_wfn_rotations.h5`, the restart bundle |
| band interpolation | `bandstructure.htransform` | `bandstructure.dat` |
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
([rank policy](dev/rank_truncation_policy.md)).

Reads `WFN.h5` in the working directory. Writes `centroids_frac_<n>[<suffix>].txt`
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
| `--prune-n-val` / `--prune-n-cond` | `nelec` / `nbands − n_val` | pruning window extents |
| `--prune-window` | `v_x_vc` | Gram band pair: `v_x_c`, `v_x_vc` (adds v×v, needed by V_H), `vc_x_vc` (full Σ square, for ncond ≫ nval) |
| `--fit-window L0:L1,R0:R1` | unset | explicit left/right windows for candidates and pruning |
| `--density-mode` | `scalar` | `scalar` charge Gram; `current` transverse three-current Gram (needs orbit closure, `--rho-power 1`; suffix `_current`) |
| `--orbit` / `--no-orbit` | on if the atom group has more than one operation | orbit-closed selection |
| `--rho-power` | 1.0 | sampling weight $w^\alpha$; point density $\propto w^{3\alpha/5}$ |
| `LORRAX_CENTROID_SELECT` (env) | `deliver` | `strict` refuses a numerically flat pool |

The fit windows must cover the Σ window the GW run consumes.

## dipole — `psp.get_dipole_mtxels`

Computes the velocity matrix elements
$\langle mk|\hat v_a|nk\rangle = \langle mk|\hat p_a + s\,i[\hat r_a, V_\mathrm{NL}]|nk\rangle$
for every full-BZ $k$ and Cartesian $a$, with $\hat p = \sum_G (k+G) c^*c$
and the nonlocal commutator from the analytic $k$-derivative of the
projectors. The sign $s$ is `vnl_velocity_sign` (deck key or
`--vnl-velocity-sign`; unset resolves to `+1`, stamped as
`prov_vnl_velocity_sign`). Consumed by `gw.head_correction` for the $q\to0$
head $S(\omega)$ of Σ and by the BSE head term ([theory](theory/physics.md)).

Reads the deck (`wfn_file`, `nval`, `ncond`, `nband`, `bispinor`) and the
`*.upf` files (deck directory, then `../qe/scf`, `../qe/nscf`). Writes
`dipole.h5`: `dipole_cart` `(3, nk, nb, nb)`, `deltaE` `(nk, nb, nb)` =
$E_b - E_{b'}$, and root provenance attributes (`prov_wfn_sha256` and its
fingerprint scheme, `prov_{nval,ncond,nband,nb_written,wfn_file}`, the
representation, V_NL and $q\to0$-operator schemes). `check_dipole_provenance`
refuses a file whose WFN, band extent, representation or V_NL convention does
not match the run, or whose operator scheme is unstamped; the fix is to rerun
this driver. The k sweep is partitioned across ranks; rank 0 writes.

Invoke: `python3 -m psp.get_dipole_mtxels -i deck.in [--out dipole.h5]`.

| flag | default | meaning |
|---|---|---|
| `--vnl-mode` | `analytic` | nonlocal velocity by analytic $dZ/dK$, or `numeric` finite differences (`--vnl-h`, `--vnl-h-rel`, `--vnl-num-scheme naive\|richardson`) |
| `--skip-vnl` | off | write $\hat p$ only (BerkeleyGW `use_momentum`) |
| `--vnl-velocity-sign` | deck, else `+1` | relative sign of $i[r,V_\mathrm{NL}]$ |
| `--pseudo-dir` | deck directory | where the `*.upf` live |
| `--with-finite-q` / `--iq-list` | off / all | also write the `finite_q/` group (`rho_cvkq`, symmetrized `v_cvkq`, `kminq_idx`); its conduction axis is sized by the producer's `ncond` |
| `--parallel-transport-out` | unset | write the SlabIO parallel-transport artifact read by `sc_head_update`; `--parallel-transport-velocity-only` writes only the DFT-velocity stage `dft_velocity` needs |

## kin-ion — `gw.kin_ion_io`

Writes $\langle mk|T + V_\mathrm{loc} + V_\mathrm{NL}|nk\rangle$ to `kin_ion.h5`.
Hartree is not in it: GW builds $V_H$ live ([theory](theory/hartree.md)). The
operator commutes with the space group and time reversal, so it is computed on
the orbit parents and unfolded through the symmetry service.

Invoke: `python3 -m gw.kin_ion_io -i deck.in [-o kin_ion.h5] [-n NB]`. The deck
owns the system, band window and spinor settings; `WFN.h5` and the
pseudopotentials are stamped as provenance, and a GW run refuses a file whose
k storage, spinor representation, system dimension, band extent, WFN, input or
pseudopotential stamp differs from its own. Multi-rank sweeps are distributed;
rank 0 writes after the gather.

| key / flag | default | meaning |
|---|---|---|
| `-n` / `--nb` | max(`nband`, nelec + ncond) | bands written; refuses below nelec + ncond, the window GW reads |
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

**Outputs.**

| file | content |
|---|---|
| `eqp0.dat` | BerkeleyGW format. $E_\mathrm{DFT} + \Delta(E_\mathrm{DFT})$, $\Delta = \langle T + V_\mathrm{ion} + V_H + \Sigma_{xc}\rangle - E_\mathrm{DFT}$ |
| `eqp1.dat` | BerkeleyGW format. Linearized $E + Z\,\Delta(E)$, $Z = (1 - \partial_\omega \mathrm{Re}\,\Sigma_c)^{-1}$ |
| `sigma_diag.dat` | Σ diagonals in eV. Bispinor runs add `sigCC`, `sigTT`, `sigCT` (= CT + TC); ordered broken-TR GN runs add `sigC_odd` |
| `eqp_g0w0.dat` | PPM one-shot only: Re/Im of $H_0 + \Sigma_{xc}(E_\mathrm{DFT})$ |
| `qp_wfn_rotations.h5` | the QP eigensystem $U_{mnk}$, $E_\mathrm{QP}$ with the source-WFN fingerprint, read by htransform, BSE and SC seeding |
| `WFN_qp.h5` | ψ rotated by U with QP energies (`write_wfn_h5`, default true) |
| `sigma_mnk.h5` | the dynamic Σ cube (PPM/MPA only) |
| `tmp/zeta_q.h5` | ζ, plus `zeta_q_mu{1,2,3}.h5` on bispinor runs |
| `tmp/isdf_tensors_<N_mu>.h5` | the restart bundle: `V_qmunu`, `W0_qmunu` (`W0_ready`), `psi_parent_y`, `enk_full`, head scalars, band window; the BSE input |
| `gwjax.out` | the run report: every resolved pathway |

All energy files are on the irreducible wedge. `eqp0.dat`/`eqp1.dat` carry
the BerkeleyGW `(3f13.9,i8)` block header, `sigma_diag.dat` and `eqp_g0w0.dat` a
`# kcrys` line per block. Join blocks to another code's by that coordinate,
never by block position: the two codes' wedges order k differently.

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
| `compute_mode` | `auto` | Σ ansatz: `x_only` \| `cohsex` \| `gn_ppm` \| `hl_ppm` \| `mpa`. `auto` infers from the legacy flags and never selects `mpa`. `gn_ppm` refuses metallic occupations ([input reference](input_reference.md)) |
| `qp_solver` | `auto` → `one_shot_dft` | `one_shot_dft`: full-matrix effective H with Σ at $E_\mathrm{DFT}$, Hermitian-symmetrized; `fixed_point`: on-shell diagonal solve (dynamic modes); `self_consistent`: the QSGW loop ([self-consistency](self_consistency.md)) |
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
[large-μ operation](dev/large_nmu_operation.md). Environment variables:
[registry](dev/env_vars.md).

## downfold — `gw.downfold_cli`

Compresses a finished GW restart onto a subset of $\mu_S$ of its $\mu_L$
centroids for the retained BSE band window. Pivoted Cholesky selects the rows;
the transfer $T = S_{SS}^{-1} S_{SL}$ is the least-squares fit in the
pair-density metric (Grams from `isdf.core.c_q_from_psi_sm` on the window), and
every $(\mu,\nu)$ tensor transforms by the congruence $A_S = T A_L T^\dagger$
(ζ by $\zeta_S = \bar T \zeta_L$). The output is a restart bundle in the
unchanged format at the smaller μ, so both BSE drivers read it unmodified;
every stored tensor shrinks by $(\mu_S/\mu_L)^2$.

The fit is exact only on redundancy: run the GW stage at a generous $\mu_L$ if
you intend to compress it. The driver prints the window Gram's eigenvalue rank
(the ceiling for $\mu_S$, refused above), the Cholesky selection certificate,
and the per-q projection error $\epsilon_W$; none of them is an accuracy gate.
Size $\mu_S$ by comparing the lowest BSE eigenvalues of parent and child. Pole
models (PPM, MPA) do not transform: refit them in the small basis.

Invoke: `python3 -u -m gw.downfold_cli -i downfold.in`. It takes its own
`[downfold]` input; [the downfold page](downfold.md) owns its keys and
refusals.

## htransform — `bandstructure.htransform`

Interpolates band energies to an arbitrary k-path. On the coarse grid it forms

$$f(H)_k = \sum_n f(\varepsilon_{nk})\, c_{nk} c_{nk}^\dagger,$$

where $c_{nk}$ are the states' coefficients in one shared whole-state Galerkin
basis and $f$ is a smooth monotone map, linear below the window and flat to
zero at its top. $f(H)_R$ follows by the flat-k inverse FFT; at each path point
the driver Fourier-interpolates $f(H)_q$, diagonalizes it, and inverts $f$ by
Newton iteration. States above the window map to exactly zero, so no band
crosses the window edge: $f(H)_k$ is smooth in k and $f(H)_R$ short-ranged. `isdf.galerkin` selects the basis from stacked full-Bloch
states by deterministic randomized QRCP and projects every state into that one
gauge; the WFN transforms stream the G→r work.

The returned window is $(n_\mathrm{elec} - n_\mathrm{val}, n_\mathrm{elec} + n_\mathrm{cond})$.
Standalone output requires `nval = nelec`: an omitted lower occupied boundary
can reproduce the samples and still ring between them. `--guard-bands` extra
conduction bands are fitted above the window, because the top of the fit window
sits on $f$'s zero shoulder.

Two QP routes, mutually exclusive, never stacked:

- `--qp-rotations qp_wfn_rotations.h5` is the full QP Hamiltonian: it rotates
  the compact Galerkin rows by the authenticated $U$, so the builder represents
  $f(H_\mathrm{QP}) = U f(E_\mathrm{QP}) U^\dagger$. When outer DFT guards
  remain, the QP block must extend above the returned window; the returned
  states are selected by their character in the QP projector
  $P_A(q)$, and a path point without a physical-energy gap between the
  returned top and the rest refuses. Stale, unstamped or foreign-WFN artifacts
  refuse before $U$ is applied.
- `--eqp-file eqp1.dat` is the diagonal approximation: it replaces energies in
  the DFT band labels and cannot represent QP mixing. The file is LORRAX's own
  wedge `eqp1.dat`; its block coordinates are checked against the deck's wedge
  and unfolded through the symmetry service. A missing or unparseable file is
  fatal.

Consumes the GW deck (with a `K_POINTS {crystal_b}` path), `WFN.h5` (or
`--wfn-file WFN_qp.h5`) and `centroids_file`. Writes `bandstructure.dat` (eV,
VBM at 0, returned and fitted windows in the header) and `htransform.out`;
rank 0 writes. `galerkin_dft.h5` beside the deck caches the basis: reused on an
exact provenance match, refit in memory otherwise.

Invoke: `python -m bandstructure.htransform -i ht.in [--qp-rotations qp_wfn_rotations.h5 | --eqp-file eqp1.dat]`.

| key / flag | default | meaning |
|---|---|---|
| `nval` / `ncond` | 5 / 5 | returned window |
| `htransform_rank_multiplier` | 20 | QRCP search ceiling $\lceil 20\, N_\mathrm{band}\rceil$; `htransform_qr_eps` (1e-3) selects the delivered rank |
| `--guard-bands` | 4 | extra fitted conduction bands |
| `--a-band` | top band | band whose bandwidth sets $f$'s scale |
| `linalg` / `--eigh-backend` | `local` | layout of the $f(H)_q$ eigensolve |
| `get_centroids_fi`, `kgrid_fi`, `wfn_fi_min`/`wfn_fi_max`, `wfn_fi_q_chunk` | off | BSE handoff: fine-grid ψ at the coarse centroids (`bandstructure.bse_setup.compute_wfns_fi`) |

Refusals: a QRCP search that saturates the ceiling (inspect the projection
receipts before raising the multiplier); an `f-shoulder` refusal when a
returned band is absent from $f(H)$ at some coarse k (add guard bands); Newton
not reaching $\max|f(x)-y| \le 10^{-12}$ Ry in 50 steps. The reproduction of
coarse-grid energies is reported as a receipt; it is not a locality
certificate, and the independent fine-grid QE comparison decides acceptance.

`K_POINTS {crystal_b}` format (shared with `bse.exciton_bands`): a count of
corners, then one line per corner with three crystal coordinates and the number
of points to the next corner, `#label` optional; the last corner takes 1.

```text
K_POINTS {crystal_b}
3
  0.0000 0.0000 0.0000 2  #G
  0.5000 0.0000 0.0000 2  #X
  0.5000 0.5000 0.0000 1  #M
```

## bse — `bse.bse_jax`

Solves the Bethe–Salpeter equation for $Q = 0$ excitons in the transition
basis $|vk \to ck\rangle$. The resonant block is

$$A = D + 2V - W \quad(\text{scalar, singlet}),\qquad A = D + V - W \quad(\text{spinor}),$$

with $D = \varepsilon_c - \varepsilon_v$, $V$ the bare exchange and $W$ the
statically screened direct term, both applied in the ISDF μ basis without
forming $A$. The default is the full (non-TDA) problem with the coupling block
$B$; `--tda` keeps $A$ only. `--rpa` (the default kernel) drops $W$; `--bse`
includes it. The matvec, shardings and output layout are in the
[BSE README](../src/bse/context/README.md).

Every sharded solver (Lanczos, block Lanczos, Davidson, thick-restart Lanczos,
FEAST) applies $H$ through the one trial-stack matvec
(`bse_stack_matvec`), whose trial axis is scanned so one direct-term tensor is
alive regardless of block width. `bse_k_grid` densifies the bundle before any
solve: ψ and ε through one htransform $f(H)$, W by zero-padding in R (exact
trigonometric interpolation, `bse.bse_densify.make_w_densifier`).

Consumes the run directory's single `isdf_tensors_*.h5` (more than one refuses,
`GATE bse_restart_ambiguous`), plus optionally `eqp1.dat`. If `W0_qmunu` lacks
`W0_ready`, the direct term is the bare V: the GW run never screened. The GW
band-window and centroid stamps are not re-verified here; point `-i` at the run
whose physics you mean. Writes `bse.out` and, with `--write-eigs`,
BerkeleyGW-layout `eigenvectors.h5` (rank 0).

Invoke: `python -u -m bse.bse_jax -i cohsex.in --lanczos --bse ...` in the GW
run directory. The mesh is the run's square startup mesh; `--px`/`--py` must
be square and use every device.

Without `--lanczos` the driver hands the solve to FEAST and forwards only
`-i`, `--n-val`, `--n-cond`, `--px`/`--py`, `--bse`/`--rpa`/`--tda` and the
`--feast-*`, `--gmres-*` and `--kpm-*` knobs. The rows marked *Lanczos* below
apply to the `--lanczos` route only; a FEAST run with `--eqp` gets DFT-energy
excitons. The driver parses with `parse_known_args`, so an unknown flag is
ignored, not refused.

| flag | default | meaning |
|---|---|---|
| `--lanczos` | off | Krylov eigensolve; without it the driver runs FEAST (`bse_feast`) |
| `--bse` / `--rpa` | RPA | `--bse` adds $-W$ |
| `--tda` | off | resonant block only |
| `--n-val` / `--n-cond` / `--n-occ` | 4 / 4 / auto | transition window; valence resolved from $\varepsilon < E_F$ (`--n-occ`: *Lanczos*) |
| `--band-degeneracy` | `strict` | *Lanczos*. A window edge inside a multiplet: `strict` refuses and names working counts, `snap` widens outward, `off` proceeds; tolerance `--degeneracy-tol-ry` (1 meV) |
| `--solver` | `lanczos` | *Lanczos*. `lanczos` (spectrum shape), `davidson` (per-state convergence, `--davidson-*`), `trlan` (thick restart, bounded memory, `--trlan-*`) |
| `--block-size` / `--max-lanczos-iter` / `--n-reorth` | 1 / auto / −1 | *Lanczos*. Block width / total Krylov dimension / reorthogonalization window (−1 = full, needed for degenerate spinor spectra) |
| `--n-eig` / `--write-eigs [N]` | 5 / off | *Lanczos*. Eigenpairs; write `eigenvectors.h5` |
| `--eqp FILE` | none | *Lanczos*. Diagonal QP energies from the wedge `eqp1.dat`, unfolded through the symmetry service; the restart must be proved to come from the same unrotated WFN |
| `bse_k_grid` (deck) | `""` | fine grid "NX NY NZ", each axis at least the coarse extent |
| `head_minibz_average` (deck) | false | mini-BZ cell average of the exchange head ([LT head](theory/lt-exchange-head.md)); also rebuilds the q = 0 tile on `bse_k_grid` |

Forgetting `--bse` gives RPA. Absorption comparisons with BerkeleyGW:
`src/bse/BGW_COMPARE.md`; module status: `src/bse/STATUS.md`.

## exciton bands — `bse.exciton_bands`

Computes the finite-momentum exciton dispersion $E_S(Q)$ from the TDA
Hamiltonian $H_Q = D_Q + V_Q - W$ (exchange weighted as in `bse_jax`) in the
basis $|vk \to c\,k{+}Q\rangle$, along the deck's `K_POINTS {crystal_b}` path.
The shifted conduction states $\psi_c(k+Q)$, $\varepsilon_c(k+Q)$ are
eigenpairs of one interpolated htransform $f(H)$ (`compute_wfns_fi` on the
q-list $\{k+Q\}$). W is the unchanged coarse-grid convolution, because every
$k - k'$ stays on the grid when all conduction legs shift by the same Q. The
exchange tile at tile momentum $\mathrm{wrap}(-Q)$ comes from `bse.vq_interp`
(G = 0 kept at finite Q). The whole path is one jitted `lax.scan` of block
Lanczos over the stack matvec: one compile for all Q.

Writes `<prefix>.dat` and `<prefix>.png` (rank 0). With `--eqp` both legs, the
stored energies and the htransform's, are corrected. There is no
`--qp-rotations` route: a full-QP run needs a restart generated from the same
stamped `WFN_qp.h5` the deck selects. Why the head is handled as it is:
[the long-range exchange head](theory/lt-exchange-head.md); the traps and
sizing rules: `src/bse/EXCITON_BANDS.md`.

Invoke: `python -u -m bse.exciton_bands -i cohsex.in --n-val 4 --n-cond 4 --n-eig 6`.

| flag | default | meaning |
|---|---|---|
| `--vq-mode` | `interp` | `interp`: interpolated tile, slab decks only (refuses `sys_dim = 3`), needs full-BZ `zeta_q.h5`. `refit`: per-Q ζ refit with the producer's Coulomb kernel, the arbitrary-Q route for bulk. `both`: interp plus refit at `--refit-points` in the same scan. `ongrid`: the stored tile, exact, coarse-grid Q only |
| `--refit-window` | `zeta` | `zeta` refits on the producer's window, certified by reproducing the stored tiles (needs $N_\mu n_s \ge N_k n_b$); `bse` fits the deck's window and certifies the contracted eigenvalues at on-grid Q |
| `--cert-grade` | `reference` | tolerance of the `bse`-window certification: `reference` 0.01 meV, `visualization` 1.0 meV; stamped into the outputs |
| `--q-per-segment` | 16 | floor on the deck's per-segment counts (1 = the deck's counts) |
| `--n-eig` / `--block-size` / `--max-iter` | 6 / 8 / 40 | per-Q block Lanczos |
| `--band-degeneracy` | `strict` | as in `bse_jax`, checked on both the loader window and the htransform conduction window |
| `--w-coarse-grid` / `--w-head-densify` / `--w-head-gamma-cell` | unset / `c1` / `fine` | densify a native fine W from a nested coarse sub-grid; `c1` splits the Γ head off before interpolation and re-attaches it analytically |
| `nband` / `nval` / `ncond` (deck) | | htransform window; the BSE conduction window must sit interior with guard bands above it |

Refusals: no `K_POINTS crystal_b` block; the Γ gate ("htransform conduction
cache grossly inconsistent") when the interpolation window is over-packed;
`interp` on IBZ-only ζ or a bulk deck; `ongrid` at an off-grid Q; a failed
`--refit-window=bse` certification. A downfolded bundle works here too: the
driver takes the parent centroid table from the bundle's `downfold_provenance`
and the transported `zeta_q.h5` for off-grid exchange.
