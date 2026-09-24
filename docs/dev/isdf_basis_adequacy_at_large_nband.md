# ISDF basis adequacy at large band counts

**Rule: the ISDF basis must be selected against the band window that Σ_c and
χ₀ actually consume.** A basis selected for a smaller pair-density block and
used for a larger one is wrong by electron-volts while every upstream check
(NSCF provenance, H₀ identity, W Dyson residual, TRS, bare Σ_X, the q→0 head
fit) passes, because none of them looks at Σ_c.

## Which centroids, not how many

On MoS₂ 4×4×1 at 30 Ry, `nb = 1024`, the same 897-point centroid count
re-selected against three pivoted-Cholesky prune windows (same WFN, candidate
pool and `zeta_rcond = 1e-8`):

| prune window | selection rank | eqp0 gap (eV) | eqp1 gap (eV) |
|---|---|---|---|
| `(0, 52)` | 630 / 897 | 0.364 | −0.364 |
| `(0, 256)` | 897 / 897 | 3.135 | 3.071 |
| `(0, 1024)` | 897 / 897 | 3.723 | 3.455 |

Bare Σ_X is unchanged throughout. Widening the window moved the ζ fit Gram's
retained rank by 1.4 % and the gap by 2.8 eV: the operative quantity is which
centroids are selected, not how many directions the fit retains. The full
window costs +13 % wall and +15 GB peak for the selection at this size.

## The selection window (`centroid/kmeans_cli.py`)

`_resolve_sigma_window` defaults the prune window to the full conduction
window in the WFN, `n_cond = nbands − n_val`, a superset of any deck's band
counts. `--prune-n-cond` narrows it; narrow only to at least
`max(number_bands_chi, number_bands_sigma)` of every deck the centroid file
will serve. `--prune-window {v_x_c, v_x_vc, vc_x_vc}` chooses which pair
densities enter the prune Gram.

`kmeans.out` prints `After pruning: N centroids (rank=R)`, both in point
units. Numerical rank deficiency of an over-complete pool is reported, not
refused (the downstream ζ solve truncates unresolved modes;
[rank truncation](rank_truncation_policy.md#where-rank-deficiency-must-not-refuse));
`LORRAX_CENTROID_SELECT=strict` makes it a refusal for diagnosis. Structural
failures (non-PSD Gram, exhausted pool, invalid pivot) always refuse.

**Refusal:** `centroid/pivoted_cholesky.py` refuses when the window's top band
exceeds half the plane-wave basis, `max_band > 0.5·ngk_max·n_spinor`: pair
densities of bands beyond that cannot be ISDF-resolved. Fix: raise the cutoff
or lower the band count.

## Two band counts: the fit window is the max

With `number_bands_chi` and `number_bands_sigma` set independently
([input reference](../input_reference.md)), the ζ fit must span the pair
densities of whichever consumer reaches higher:

    ISDF window top = max(number_bands_chi, number_bands_sigma)

ψ is loaded once up to the padded top of the larger count, ζ is fitted on that
window, and the smaller consumer takes a slice inside it.

| what | where |
|---|---|
| the `max` | `gw_config.BandCounts.isdf` (one property) |
| the invariant at the consuming seam | `gw_init.assert_isdf_window_is_the_max`, called from `fit_zeta`; refuses otherwise |
| which count won, and what the fit was built for | logged every run by `BandCounts.describe()` |
| tests | `tests/test_band_count_split.py` |

`zeta_nband` may narrow the fit below a band sum's top (the BSE Galerkin
capacity bound is why it exists); the consumer left above the fit edge then
runs on an extrapolated ζ basis, and the run reports that per consumer by name.

## Sizing observables

* `[zeta rank_truncate]` prints `n_keep / n_pad` per q. Report it; it is an
  observable, not a limit, and it does not measure basis quality.
* `N_μ ≈ 6–14 × nband` is a starting point, not an adequacy check: the failing
  selections above sat inside that band.
* The restart tensor scales as `N_μ²` (≈ 57 GB at N_μ = 10⁴, 123 GB at
  1.5·10⁴). `write_restart_tensors = false` computes fresh without writing it;
  `restart_q_storage` chooses the q-set when it is kept.

## Gate: pinned Σ reference

A Σ_c-only corruption is caught only by checking Σ_c. Before accepting a new
size point, and on any change to the Σ or ISDF path, re-run MoS₂ 4×4×1 30 Ry,
`nval 26 / ncond 230 / nband 256`, 2475 orbit-closed centroids, P = 64, and
assert the indirect QP gaps `eqp0 = 3.5819 eV`, `eqp1 = 3.2516 eV` to
1e-3 eV (the configuration is deterministic to ~1e-6; the failure above is
2800× the tolerance). Cost: 3.9 node-hours.
