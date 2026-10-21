---
driver: exciton-bands
page: keywords
---

# Input keywords

`bse.exciton_bands` Every command-line flag and deck key exciton_bands reads, grouped by what it controls.
{: .page-summary }

<div class="chain-map" data-driver="exciton-bands" data-page="keywords"></div>

The driver takes command-line flags and reads deck keys, written `key = value` one per line, from the deck passed with `-i`; the deck also carries the `K_POINTS` path block. Defaults are read from the driver's argparse parser (`src/bse/exciton_bands.py`, `build_parser`) and the keyword schema (`src/gw/gw_config.py`). An unknown flag refuses; a prefix of a flag is not accepted.

Required
{: .kw-group }

**Files**

### `-i`, `--input` {: .kw data-type="path" data-default="required" }
The GW deck with a `K_POINTS` block. Its directory holds the restart bundle.

### `centroids_file` {: .kw data-type="path" data-default="required" }
ISDF point table the restart was built on; the driver evaluates $\psi_{c\,\bk+\bQ}$ at these points. Shared with [`bse`](../bse/keywords.md).

On a downfolded bundle the parent table recorded in the bundle is used first.
{: .kw-details }

**Bands**

### `--n-val` {: .kw data-type="int" data-default="required" }
Valence bands in the BSE window, counted down from the highest occupied band. Shared with [`bse`](../bse/keywords.md).

Clamped to the occupied count, which is the WFN's `ifmax`; this driver has no `--n-occ`.
{: .kw-details }

### `--n-cond` {: .kw data-type="int" data-default="required" }
Conduction bands in the BSE window. Shared with [`bse`](../bse/keywords.md).

Must fit inside the deck's H-transform window below its guard bands.
{: .kw-details }

### `nval` {: .kw data-type="int" data-default="required" }
Valence bands of the H-transform window, $[n_\mathrm{elec}-n_\mathrm{val},\, n_\mathrm{elec}+n_\mathrm{cond})$.

Under `--refit-window zeta` the window must be the GW run's ζ-fit window exactly; it is also the window `--refit-window bse` fits on.
{: .kw-details }

### `ncond` {: .kw data-type="int" data-default="required" }
Conduction bands of the H-transform window.

Keep at least `--n-cond` + 4: fewer than four guard bands above the BSE conduction window warns, and more than 16 warns too.
{: .kw-details }

**Q path**

### `K_POINTS` {: .kw data-type="block" data-default="required" }
The path corners: a line starting `K_POINTS`, the corner count, then one line per corner with three crystal coordinates, a point count and an optional `#label`.

The count on a corner's line is the number of points on the segment that ends there; the first corner's count is unused.
{: .kw-details }

**Restart and layout**

### `sys_dim` {: .kw data-type="int" data-default="required" }
Dimension of the Coulomb truncation, `3` (bulk) or `2` (slab). Required by the deck parser; this driver does not read it otherwise.

Whether `interp` and `both` run is decided by the restart bundle's Coulomb-policy stamp (`sys_dim` 2 or unstamped) and a slab-separable cell with a planar $\bq$ grid; the `refit` kernel is the bulk one only for a stamped `sys_dim = 3` bundle. See [Dependencies](dependencies.md).
{: .kw-details }

Optional
{: .kw-group }

**Q path**

### `--q-per-segment` {: .kw data-type="int" data-default="16" }
Minimum number of $\bQ$ on each path segment.

A floor, not an override: denser deck segments keep their counts, and `1` reproduces the deck's counts exactly.
{: .kw-details }

### `--extra-q` {: .kw data-type="str" data-default="—" }
Extra fractional $\bQ$, `"x,y,z;x,y,z"`, solved in the same scan and written with mode `extra`, not plotted.

Pass point-group images of off-grid path points: their levels must agree, so any difference is interpolation error.
{: .kw-details }

**Exchange at Q**

On a bulk system set `--vq-mode refit`, because `interp` and `both` are slab-only.

### `--vq-mode` {: .kw data-type="enum" data-default="interp" }
Source of the off-grid exchange tile $V_\bQ$.

`interp` and `both` are slab-only (judged from the bundle's Coulomb stamp and the cell, not the deck); `refit` is the bulk route; `ongrid` is exact but refuses off-grid $\bQ$.
{: .kw-details }

`interp` `refit` `both` `ongrid`
{: .kw-values }

### `--head-minibz-average` {: .kw data-type="bool" data-default="head_minibz_average" }
`interp`: replace the finite-$\bQ$ exchange head by its mini-BZ cell average, carried as a rank-three tensor term.

Overrides the deck key, but can only switch it on. Reads `dipole.h5`. Refuses with `refit`, also when set by the deck; ignored by `ongrid`.
{: .kw-details }

### `head_minibz_average` {: .kw data-type="bool" data-default="false" }
Deck form of `--head-minibz-average`. Shared with [`bse`](../bse/keywords.md), which reads it under `bse_k_grid`; it must match the GW run.

### `--dipole` {: .kw data-type="path" data-default="dipole.h5 beside the deck" }
`dipole.h5` for the mini-BZ head tensor. Read only when the head average is on.

Unlike [`bse`](../bse/keywords.md), where `--dipole` stores exciton dipoles, here it supplies the head's $\bq$-linear coefficient.
{: .kw-details }

### `--refit-window` {: .kw data-type="enum" data-default="zeta" }
`refit`: band window the new ζ is fitted on.

`zeta` is the GW run's ζ-fit window, certified by the stored tiles. `bse` is the deck's window, certified on the eigenvalues at on-grid path points; it refuses with any other `--vq-mode`.
{: .kw-details }

`zeta` `bse`
{: .kw-values }

### `--cert-grade` {: .kw data-type="enum" data-default="reference" }
Tolerance of the eigenvalue certification: `reference` 0.01 meV, `visualization` 1.0 meV.

The grade and the worst difference are stamped into the report, the `.dat` header and the plot. There is no third tolerance.
{: .kw-details }

`reference` `visualization`
{: .kw-values }

### `--refit-points` {: .kw data-type="str" data-default="5 evenly spaced" }
`both`: comma-separated path indices to refit.

### `--alpha` {: .kw data-type="float" data-default="0.30" }
`interp`: width of the Gaussian short-range/long-range split of $v$, in bohr⁻¹.

### `--eps-tik` {: .kw data-type="float" data-default="1e-4" }
`interp`: relative Tikhonov filter width applied to the coarse tiles.

### `--refit-guard-bands` {: .kw data-type="int" data-default="4" }
`refit`: guard bands of the refit's H-transform window above the window ζ is fitted on.

The WFN must carry them. `0` must fail the tile check.
{: .kw-details }

### `--refit-r-chunk` {: .kw data-type="int" data-default="2048" }
`refit`: real-space chunk of the pair-density build. Lower it on dense $\bk$ grids to fit device memory.

**Bands**

### `--eqp` {: .kw data-type="path" data-default="—" }
BerkeleyGW-format `eqp1.dat`: run on quasiparticle energies. Shared with [`bse`](../bse/keywords.md).

Both legs move: the stored valence and conduction energies and the H-transform energies at $\bk+\bQ$. Refuses on a QP wavefunction and with `bse_k_grid`.
{: .kw-details }

### `--band-degeneracy` {: .kw data-type="enum" data-default="strict" }
A BSE window edge inside a degenerate multiplet. Shared with [`bse`](../bse/keywords.md).

`strict` refuses and names working counts, `snap` widens the loaded window outward, `off` proceeds. Checked on the loaded window and on the H-transform conduction window, where `snap` only warns.
{: .kw-details }

`strict` `snap` `off`
{: .kw-values }

### `--degeneracy-tol-ry` {: .kw data-type="float" data-default="7.3499e-5" }
"Same multiplet" tolerance in Ry (1 meV). Shared with [`bse`](../bse/keywords.md).

**Solver**

### `--n-eig` {: .kw data-type="int" data-default="6" }
Exciton levels per $\bQ$. Shared with [`bse`](../bse/keywords.md), where the default is 5.

### `--block-size` {: .kw data-type="int" data-default="8" }
Block width of the per-$\bQ$ block Lanczos. Shared with [`bse`](../bse/keywords.md), where the default is 1.

### `--max-iter` {: .kw data-type="int" data-default="40" }
Block-Lanczos iterations; the Krylov dimension is block × iterations, capped at the physical transition count.

Reorthogonalization is always full.
{: .kw-details }

**Files**

### `wfn_file` {: .kw data-type="path" data-default="WFN.h5" }
Wavefunctions for the H-transform, relative to the deck.

### `--out-prefix` {: .kw data-type="str" data-default="exciton_bands" }
Prefix of `<prefix>.dat`, `<prefix>.png` and the default report name.

### `--report-file` {: .kw data-type="path" data-default="<prefix>.out" }
Run report.

**Screened interaction**

### `bse_k_grid` {: .kw data-type="str" data-default="&quot;&quot;" }
Fine BSE grid `"NX NY NZ"`: the loader moves the whole problem onto it before the solve. Shared with [`bse`](../bse/keywords.md).

The exchange-tile grid stays the coarse restart grid. Refuses with `--eqp`.
{: .kw-details }

### `w_head_densify` {: .kw data-type="enum" data-default="c1" }
Treatment of the long-range part of $W$ when it is densified. Shared with [`bse`](../bse/keywords.md).

`--w-coarse-grid` implements `c1` only; `srlr` runs on the `bse_k_grid` path. `legacy` refuses by name.
{: .kw-details }

`c1` `srlr`
{: .kw-values }

### `--w-coarse-grid` {: .kw data-type="str" data-default="—" }
`NX,NY,NZ`: sample $W$ on this sub-grid of the BSE grid and interpolate it back for the direct term.

Each axis must divide the BSE grid. Equal to the BSE grid, it changes nothing.
{: .kw-details }

### `--w-head-densify` {: .kw data-type="enum" data-default="w_head_densify" }
Overrides the deck's `w_head_densify` for `--w-coarse-grid`. Only `c1` is accepted.

### `--w-head-gamma-cell` {: .kw data-type="enum" data-default="fine" }
Mini-BZ cell over which the re-attached $\Gamma$ head is averaged. `coarse` is a deliberately wrong control for tests.

`fine` `coarse`
{: .kw-values }

**Restart and layout**

### `--px`, `--py` {: .kw data-type="int" data-default="startup mesh" }
Device mesh over the transition axes. Must be square and use every device. Shared with [`bse`](../bse/keywords.md).

### `linalg` {: .kw data-type="enum" data-default="local" }
Layout of the H-transform eigensolves; `distributed` selects the low-memory eigensolver.

`local` `distributed`
{: .kw-values }

### `wfn_fi_q_chunk` {: .kw data-type="int" data-default="0" }
How many $\bk+\bQ$ points the H-transform diagonalizes at once. `0` means the coarse $\bk$ count.

**H-transform basis**

These keys tune the H-transform that supplies the conduction states at $\bk+\bQ$.

### `--a-band` {: .kw data-type="int" data-default="top band" }
Window-relative band whose bandwidth $W_a$ sets the H-transform shoulder width $a = 4W_a$.

Set it to a flat conduction band only when the conduction states fall in the compressed shoulder of $f$.
{: .kw-details }

### `htransform_qr_eps` {: .kw data-type="float" data-default="1e-3" }
Rank tolerance of the whole-state Galerkin basis. See [htransform](../../utilities/htransform.md#htransform_qr_eps).

### `htransform_rank_multiplier` {: .kw data-type="float" data-default="20" }
Search ceiling of the Galerkin basis, in units of the fitted band count. `0` reads as 20.

### `htransform_qrcp_seed` {: .kw data-type="int" data-default="0" }
Seed of the basis candidate shuffle and sketch.

**Checks and debugging**

### `--rerun-check` {: .kw data-type="bool" data-default="false" }
Solve the whole scan a second time and refuse unless the two tables agree to $10^{-10}$ Ry.

### `--skip-rerun-check` {: .kw data-type="bool" data-default="false" }
Names the default; kept so existing launch scripts parse. Wins over `--rerun-check`.

### `sigma_freq_debug_output` {: .kw data-type="bool" data-default="false" }
Also runs the `interp` model's self-check battery. Replicated tensors; tens of GB at large $N_\mu$.

