---
driver: bse
page: keywords
---

# Input keywords

`bse.bse_jax` Every command-line flag and deck key the BSE driver and `bse.absorption_haydock` read.
{: .page-summary }

<div class="chain-map" data-driver="bse" data-page="keywords"></div>

`bse.bse_jax` takes command-line flags; deck keys are written `key = value` in the GW deck passed with `-i`. Defaults are read from the keyword schema (`src/bse/bse_jax.py` and `src/bse/absorption_haydock.py` argument parsers, `src/gw/gw_config.py` `_DEFAULTS`). The CLI is strict: an unknown flag refuses, and so does a flag set away from its default on a route that does not read it. Where a flag has a BerkeleyGW counterpart in `kernel.inp` or `absorption.inp`, the details name it.

## `bse.bse_jax` {: #bse-jax }

Required
{: .kw-group }

**Command line**

### `-i`, `--input` {: .kw data-type="path" data-default="required" }
The GW deck. Locates the restart bundle, `wfn_file`, the centroid table and the deck keys below.

### `--n-val` {: .kw data-type="int" data-default="required" }
Valence bands $n_v$ below the occupied boundary $n_{\mathrm{occ}}$. BerkeleyGW: `number_val_bands`.

Clamped to $n_{\mathrm{occ}}$. Counts bands, spinor bands included.
{: .kw-details }

### `--n-cond` {: .kw data-type="int" data-default="required" }
Conduction bands $n_c$ from $n_{\mathrm{occ}}$ up. BerkeleyGW: `number_cond_bands`.

Clamped to the bands stored in the bundle.
{: .kw-details }

**Deck keys**

### `nval`, `ncond`, `number_bands` {: .kw data-type="int" data-default="required" }
Deck keys. The GW band window the bundle was written with; under `bse_k_grid`, also the window of the band interpolation.

The bundle authenticates against them, so they must be the GW run's values. Under `bse_k_grid`, fewer than four conduction guard bands above the BSE window draws a warning.
{: .kw-details }

### `centroids_file` {: .kw data-type="path" data-default="required" }
Deck key. Centroid table; its content hash authenticates the bundle, and under `bse_k_grid` it is the interpolation's point set.

On a downfolded bundle the parent table comes from the bundle's own provenance.
{: .kw-details }

Optional
{: .kw-group }

**Route and kernel**

### `--lanczos` {: .kw data-type="flag" data-default="no-op" }
Selects the eigensolve, which is the default route, so the flag changes nothing. It is kept so that older command lines still parse.

The eigensolve runs the solver chosen by `--solver` with `--tda` and the dense full-BSE solver without. Only this route reads `--n-eig`, `--write-eigs`, `--dipole`, `--eqp`, `--n-occ` and the solver flags. Two route flags together, such as `--lanczos` with `--kpm-dos`, refuse.
{: .kw-details }

### `--tda` {: .kw data-type="flag" data-default="off" }
Tamm–Dancoff: solve the resonant block $A$ only. Off: the full $\begin{pmatrix}A&B\\-B^*&-A^*\end{pmatrix}$ problem on every route.

BerkeleyGW is TDA by default; LORRAX is not. Required by `--dipole`.
{: .kw-details }

### `--bse` {: .kw data-type="flag" data-default="off" }
Includes the screened direct term: kernel $D + w_x V - W$. Off: the RPA kernel $D + w_x V$, and $W$ is never built.

With it, the BerkeleyGW counterpart is `spin_singlet` (scalar) or the spinor kernel.
{: .kw-details }

### `--rpa` {: .kw data-type="flag" data-default="off" }
Forces the RPA kernel $D + w_x V$ even with `--bse`. BerkeleyGW: `local_fields`.

**Bands and energies**

### `--eqp` {: .kw data-type="path" data-default="—" }
`eqp1.dat` with diagonal QP energies on the irreducible wedge; replaces $E$ by $E^{\mathrm{QP}}$ in the diagonal, ψ unchanged. Eigensolve route only. BerkeleyGW: `eqp_corrections` (`eqp_co_corrections` with `bse_k_grid`).

Without `bse_k_grid` the window is re-sliced on the corrected spectrum; with it the QP energies enter the band interpolation.
{: .kw-details }

### `--n-occ` {: .kw data-type="int" data-default="WFN ifmax" }
Occupied-band count that places the window $[n_{\mathrm{occ}} - n_v,\, n_{\mathrm{occ}} + n_c)$. Eigensolve route only.

`--kpm-dos` always takes the count from the deck's WFN.
{: .kw-details }

### `wfn_file` {: .kw data-type="path" data-default="WFN.h5" }
Deck key. WFN for the occupied count, the cell volume, bundle authentication, the W0 rebuild and the band interpolation; relative to the deck.

### `--band-degeneracy` {: .kw data-type="enum" data-default="strict" }
What happens when a window edge cuts a degenerate multiplet (a Kramers pair under spin–orbit coupling). Eigensolve route only; the other routes use `strict`.

`strict` refuses and names counts that work; `snap` widens the window outward to the multiplet boundary; `off` proceeds. BerkeleyGW: `degeneracy_check_override`.
{: .kw-details }

`strict` `snap` `off`
{: .kw-values }

### `--degeneracy-tol-ry` {: .kw data-type="float" data-default="7.3499e-05 (1 meV)" }
Energy tolerance in Ry for "same multiplet". Eigensolve route only.

**Solver**

### `--n-eig` {: .kw data-type="int" data-default="5" }
Number of lowest eigenpairs $\Omega_S$, $A^S$. Eigensolve route only. BerkeleyGW: `number_eigenvalues`.

### `--solver` {: .kw data-type="enum" data-default="davidson" }
Eigensolver for the lowest `--n-eig` roots. Ignored without `--tda`.

`davidson`: block Davidson with the exact diagonal preconditioner, stopped when every root's residual is below `--davidson-tol-mev`. `lanczos`: block Lanczos with CGS2 reorthogonalization, at a fixed Krylov dimension or to `--lanczos-rtol`. `trlan`: thick-restart Lanczos, the convergence of `lanczos` from a basis capped at `--trlan-m-max`. A flag that only another solver reads refuses.
{: .kw-details }

`davidson` `lanczos` `trlan`
{: .kw-values }

### `--davidson-tol-mev` {: .kw data-type="float" data-default="0.01" }
Davidson stopping rule in meV. The solve stops when every root's residual $\lVert Hx - \Omega x\rVert$ is below it, relative above $|\Omega| = 1$ Ry.

A residual $r$ puts an exact eigenvalue within $\lVert r\rVert$ of $\Omega$, so the tolerance bounds each root's error, which is typically $\lVert r\rVert^2$ divided by the gap to the next root. A solve that reaches the iteration cap first refuses (`GATE davidson_converged`).
{: .kw-details }

### `--max-lanczos-iter` {: .kw data-type="int" data-default="max(30, min(200, N/2))" }
Iteration budget, with $N = n_{c,\mathrm{pad}}\,n_{v,\mathrm{pad}}\,N_k$. For Davidson it caps the block iterations, and the solve normally stops earlier on `--davidson-tol-mev`.

For `lanczos` it is the Krylov dimension, run as `max_lanczos_iter // block_size` block steps. For `trlan` it is the matvec budget.
{: .kw-details }

### `--davidson-precond` {: .kw data-type="enum" data-default="exact" }
Davidson preconditioner diagonal.

`exact`: the assembled $\mathrm{diag}(H) = E_c - E_v + (V_x - W_d)/N_k$, or $E_c - E_v + V_x/N_k$ under `--rpa`, which is what BerkeleyGW hands PRIMME. It is assembled once, and each iteration costs the same as with `bare`. `bare`: the transition energy $E_c - E_v$. `auto`: chooses by the transition-space dimension.
{: .kw-details }

`exact` `bare` `auto`
{: .kw-values }

### `--davidson-margin` {: .kw data-type="int" data-default="8" }
Extra roots solved above `--n-eig` and not reported, at most half the transition space. The stop reads only the lowest `--n-eig`.

### `--davidson-margin-tol-mev` {: .kw data-type="float" data-default="--davidson-tol-mev" }
Looser residual, in meV, at which a margin root locks. A locked root stays in the basis and gets no further correction. It must be at least `--davidson-tol-mev`.

### `--davidson-m-max` {: .kw data-type="int" data-default="10 × n_eig" }
Davidson subspace size before restart; memory is $2m_{\max}$ vectors.

### `--davidson-olsen` {: .kw data-type="flag" data-default="off" }
Olsen correction of the preconditioned direction; makes a small `--davidson-eps-shift` safe.

### `--davidson-eps-shift` {: .kw data-type="float" data-default="1e-3" }
Preconditioner regularization in Ry. An explicit `0` is replaced by the default.

### `--block-size` {: .kw data-type="int" data-default="1" }
Lanczos block width: larger matrix products, fewer dispatches.

### `--lanczos-rtol` {: .kw data-type="float" data-default="0.0" }
`0`: fixed iteration count. `> 0`: exits when the lowest `--n-eig` Ritz values change by less than this between checks.

Lanczos solver only, at any block size; absolute floor $10^{-8}$.
{: .kw-details }

### `--lanczos-check-every` {: .kw data-type="int" data-default="4" }
Convergence-check cadence in block iterations under `--lanczos-rtol > 0`.

### `--n-reorth` {: .kw data-type="int" data-default="-1" }
Reorthogonalization window; `-1` is the full Krylov depth.

Narrower windows give ghost eigenvalues and cost the same; keep full for spinor spectra.
{: .kw-details }

### `--trlan-m-max` {: .kw data-type="int" data-default="max(3 n_eig, 60)" }
Thick-restart basis slots, the memory cap.

### `--trlan-n-keep` {: .kw data-type="int" data-default="n_eig + 10" }
Ritz vectors kept across a restart; $n_{\mathrm{eig}} \le n_{\mathrm{keep}} < m_{\max}$.

**Output files**

### `--write-eigs` {: .kw data-type="int" data-default="off" }
Writes the lowest $N \le$ `--n-eig` states to `eigenvectors.h5`; the bare flag writes `--n-eig`. BerkeleyGW: `write_eigenvectors`.

### `--dipole` {: .kw data-type="path" data-default="—" }
With `--write-eigs` and `--tda`: contracts each written state with this `dipole.h5` and stores $d^\alpha_S$ as `exciton_data/dipoles`.

A `psp.get_dipole_mtxels --skip-vnl` file corresponds to BerkeleyGW's `use_momentum`.
{: .kw-details }

### `--report-file` {: .kw data-type="path" data-default="bse.out" }
Run report of the eigensolve route.

**Fine grid and head**

### `bse_k_grid` {: .kw data-type="str" data-default="\"\"" }
Deck key. The fine BSE grid $\bk_{\mathrm{fi}}$ as `NX NY NZ` (or comma-separated). Replaces BerkeleyGW's `WFN_fi` with coarse-to-fine kernel interpolation.

Each axis must be at least the coarse one, and the two grids need not nest. ψ, $E^{\mathrm{QP}}$, $W$ and the dipoles move to $\bk_{\mathrm{fi}}$ in the loader, so `--dipole` and `bse.absorption_haydock` run on the fine grid.
{: .kw-details }

### `w_head_densify` {: .kw data-type="enum" data-default="c1" }
Deck key. Long-range model split off $W$ before the densification under `bse_k_grid`.

`srlr`, the intended route: the long range is carried by the plane-wave channels $\bq+\bG_a\to0$ (bulk: the Wigner–Seitz slot; slab: plus the $G_z$ shells), whose dressed form factors and polarizability come from the bundle's `W0_lr`, are fitted in $\bq$, screened, switched off smoothly by a window inside the zone, and evaluated at every fine $\bq$; Γ keeps the `c1` head. `c1`: the $\bq \to 0$ head alone, re-attached at each fine $\bq$ in the coarse Γ cell. Today only a static-Dyson GW run writes `W0_lr`, so after a shared-pole run `srlr` refuses (`GATE srlr_restart_channels`) and `c1` is what runs. See [Theory §1.4](theory.md#grid-w) and [Short- and long-range W](../../theory/w-range-splitting.md).
{: .kw-details }

`c1` `srlr`
{: .kw-values }

### `head_minibz_average` {: .kw data-type="bool" data-default="false" }
Deck key, read only under `bse_k_grid`. `true` rebuilds the $V_{\bq=0}$ tile with the fine grid's mini-BZ head and takes the $W$ head's Γ-cell reference from the analytic sphere.

Must match the GW run ([q → 0 head](../../theory/q0-head.md#minibz)). `false` carries the coarse tile through unchanged.
{: .kw-details }

### `vhead` {: .kw data-type="float" data-default="—" }
Deck key. Overrides the bundle's $V^{\mathrm{head}}$ (Ry·bohr³) in the rank-one $V_{\bq=0}$ head. Shared with the GW side.

A cross-code validation knob (BerkeleyGW's head). A non-numeric value refuses.
{: .kw-details }

### `whead_0freq` {: .kw data-type="float" data-default="—" }
Deck key. Overrides $W^{\mathrm{head}}(\omega = 0)$ on $W_{\bq=0}$. Added only to a screened $W$.

### GW screening and head keys {: .kw data-type="deck" data-default="GW run" }
`head_correction`, `wcoul0_source` and the other screening keys are read only when the bundle has no stored W0 and the driver rebuilds $W(0)$.

They must describe the GW run; see the [GW keywords](../gw/keywords.md).
{: .kw-details }

### `wfn_fi_q_chunk` {: .kw data-type="int" data-default="0" }
Deck key. Fine-grid points per batch of the band interpolation under `bse_k_grid`; `0` uses the coarse $N_k$. Memory only.

### `htransform_rank_multiplier`, `htransform_qr_eps`, `htransform_qrcp_seed` {: .kw data-type="float, float, int" data-default="20.0, 1e-3, 0" }
Deck keys. Basis controls of the band interpolation under `bse_k_grid`. Defined on the [htransform](../../utilities/htransform.md#htransform_qr_eps) page.

### `sigma_freq_debug_output` {: .kw data-type="bool" data-default="false" }
Deck key. Runs the `vq_interp` self-checks when `head_minibz_average` rebuilds the exchange tile. Large host memory.

**Exciton density of states**

### `--kpm-dos` {: .kw data-type="flag" data-default="off" }
Computes the KPM density of states of $H$ and exits. It cannot be combined with `--lanczos`.

### `--kpm-window-count` {: .kw data-type="int" data-default="4" }
`--kpm-dos` only. Number of windows of equal state count marked on the density of states.

### `--kpm-n-moments`, `--kpm-n-random`, `--kpm-n-lanczos` {: .kw data-type="int" data-default="100, 4, 100" }
`--kpm-dos` only. Chebyshev moments $M$, stochastic trace vectors $R$ ($RM$ matvecs), Lanczos steps for the bounds.

### `--kpm-emin-ev`, `--kpm-emax-ev` {: .kw data-type="float" data-default="Lanczos bounds" }
`--kpm-dos` only. Override the spectral bounds, in eV.

### `--kpm-plot-file` {: .kw data-type="path" data-default="bse_dos_kpm.png" }
`--kpm-dos` only. DOS plot; the moments go beside it as `.npz`.

**Device layout**

### `--px`, `--py` {: .kw data-type="int" data-default="startup mesh" }
Device mesh over the $(c, v)$ transition axes. Given together, square, and using every device.

## Absorption (`bse.absorption_haydock`)

A separate entry point. It always uses the TDA kernel $D + w_x V - W$, needs at least two devices, and checks band degeneracy `strict`. Under `bse_k_grid` the loader densifies the bundle and the dipoles with it.

Required
{: .kw-group }

**Bands and cell**

### `-i`, `--n-val`, `--n-cond` {: .kw data-type="path, int, int" data-default="required" }
As for `bse.bse_jax`.

### `--n-occ` {: .kw data-type="int" data-default="required" }
Occupied-band count; re-resolved on the QP spectrum under `--eqp`.

### `--V-cell` {: .kw data-type="float" data-default="required" }
Cell volume $\Omega_{\mathrm{cell}}$ in bohr³ for the prefactor $\kappa$. Not read from the WFN.

Optional
{: .kw-group }

**Inputs**

### `--dipole` {: .kw data-type="path" data-default="dipole.h5" }
Velocity matrix elements; the seed $|d_\alpha\rangle$ in position form, with the file's own band energies.

### `--eqp` {: .kw data-type="path" data-default="—" }
As for `bse.bse_jax`. BerkeleyGW: `eqp_corrections`.

**Spectrum**

### `--eta-eV` {: .kw data-type="float" data-default="0.1" }
Lorentzian half-width $\eta$ in eV, also of the JDOS column. BerkeleyGW: `energy_resolution` with `lorentzian_broadening`.

### `--omega-min-eV`, `--omega-max-eV`, `--n-omega` {: .kw data-type="float, float, int" data-default="0.0, 15.0, 1500" }
Uniform frequency grid in eV, endpoints included. The Kramers–Kronig integral for $\varepsilon_1$ is truncated to it.

### `--n-iter` {: .kw data-type="int" data-default="200" }
Haydock steps per polarization, without reorthogonalization. BerkeleyGW: `number_iterations`.

### `--no-eps1` {: .kw data-type="flag" data-default="off" }
Skips Kramers–Kronig; the $\varepsilon_1$ column is `1`.

**Spin factors and output**

### `--n-spinor` {: .kw data-type="int" data-default="2" }
$n_{\mathrm{spinor}}$ in $\kappa$. The default is the spinor case: set `1` for a scalar run.

### `--n-spin` {: .kw data-type="int" data-default="1" }
$n_{\mathrm{spin}}$ in $\kappa$; `2` only for a collinear spin-polarized run.

### `--out-prefix` {: .kw data-type="path" data-default="absorption_haydock" }
Names `<prefix>.h5`. The text files are always `absorption_haydock_b{1,2,3}_eh.dat`, in the prefix's directory.

## Retired

### `--ring-timing` {: .kw data-type="flag" data-default="refuses" }
Retired; refuses by name. Time the matvec with `common.timing`.

### `w_head_densify = legacy` {: .kw data-type="value" data-default="refuses" }
Retired. Use `c1` or `srlr`.

### `LORRAX_LANCZOS_REORTH` {: .kw data-type="env" data-default="refuses" }
Retired; CGS2 is the only reorthogonalization. Unset it.

