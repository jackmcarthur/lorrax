# ISDF basis adequacy at large band counts

**Status**: measured on the MoS2 4x4x1 deck, 2026-07-29 (LORRAX size campaign).
**Audience**: anyone choosing `nband` / `ncond` / `N_mu` for a production GW run.

**The rule this document exists for**: *the ISDF basis must be SELECTED against
the band window `Sigma_c` actually consumes.* If it is not, results can be wrong
by electron-volts while every gate in the suite passes. That was a real bug
(Section 2); it is fixed, and Section 5 adds the refusal that makes it
un-repeatable.

**A second, SEPARATE observation whose interpretation is still open**: the ISDF
fit's retained rank saturates as `N_mu` grows (Section 3). It is **not** what
caused the failure above, and every number in that table was measured on a basis
built with the clamped window — i.e. on a rank-deficient selection. The curve is
real; what it implies for choosing `nband` is pending a re-measurement.

---

## 1. Why this document exists

A GW run at `nband = 1024` produced a QP gap of **0.36 eV** where the answer is
**~3.1-3.7 eV**, with a **negative** `eqp1` fundamental gap. It passed every
gate the project runs:

| gate | result | why it did not catch this |
|---|---|---|
| `el_compare` (NSCF provenance) | 1.86e-11 eV | upstream of the ISDF entirely |
| `gate_h0` (H0 identity / implied Vxc) | 3.9e-5 eV | tests the one-body Hamiltonian |
| W Dyson residual | 1.9e-14 | tests the screening solve, not Sigma |
| density symmetry / TRS | 1.3e-14 | tests the wavefunctions |
| bare `Sigma_X` diagonal | 0.03% vs a good run | exchange needs far less ISDF rank |
| q->0 head correction fit | 0.026%, identical on-shell shift | analytic, ISDF-independent |
| route / restart / rc checks | all clean | structural, not numerical |

Every one is upstream of, or orthogonal to, the correlation self-energy.
**Nothing in the suite looked at the quantity the code exists to produce.** Two
gates have since been added (Sections 5 and 6).

## 2. The bug: the selection window never tracked the deck's conduction window

`centroid/kmeans_cli.py::_resolve_sigma_window` defaulted to

    n_cond = min(n_val, nbands - n_val)

which clamps the conduction extent of the pivoted-Cholesky **prune window** to
`n_val`. On this deck (`nelec = 26`) that produced a `26 x 52` prune window for
*every* centroid set ever built, while the `nb=1024` deck's sigma window is
`nval 26 + ncond 998 = 1024`. **The basis was selected to resolve a `26 x 52`
pair-density block and then used for a `1024 x 1024` one.**

Rebuilt at identical N, WFN and candidate pool (`M = 13872`), changing only the
prune window:

| prune window | Gram diagonal minimum | selection rank achieved / requested |
|---|---|---|
| `(0,52)` (old default) | 7.632e-17 | **630 / 897** |
| `(0,256)` | 7.189e-13 | **897 / 897** |
| `(0,1024)` (full) | 4.996e-12 | **897 / 897** |

The old default is **rank-deficient by 30%**: pivoted Cholesky was asked for 897
independent directions and could certify only 630, because with a narrow window
most candidate centroids sit at numerically-zero Gram diagonal. The shortfall
was printed in every centroid log the project ever produced.

### 2.1 The decisive experiment

Same deck point (`nb=1024`, `N_mu` 10015-10037), same WFN, same weapons, same
`zeta_rcond = 1e-8`. The **only** difference is the prune window used to select
the centroids:

| prune window | selection rank | eqp0 | eqp1 |
|---|---|---|---|
| `(0,52)` old default | 630 / 897 | **0.3645** | **-0.3639** |
| `(0,256)` | 897 / 897 | **3.1350** | **3.0710** |
| `(0,1024)` full | 897 / 897 | **3.7227** | **3.4551** |
| healthy family, nb = 256 / 512 | — | 3.22 - 3.63 | 2.98 - 3.29 |

**Monotone in window width**, and the gap moves from unphysical (negative
`eqp1`) into the physical range purely by re-selecting the *same number* of
centroids against a representative pair-density block. Bare `Sigma_X` is
unchanged throughout (-40.5368 vs -40.5358) — exchange was never the problem.

Stated honestly: the two corrected windows do not agree with each other
(0.59 eV apart in `eqp0`), and the full window sits slightly above what the
`nb = 256 -> 512` trend would extrapolate. **That residual is a convergence
question, not a correctness one** — the correctness claim needs only the
unambiguous, monotone move off `0.3645 / -0.3639`. Which window is
best-converged (and whether `--prune-window vc_x_vc`, which adds `c x c` pair
densities, is better still) is left open.

### 2.2 What this does NOT show, and why that matters

Widening the prune window moved the *fit* Gram's retained rank by only **+1.4%**
(6700 -> 6793) while moving the QP gap by **2.8 eV**. So:

> **The operative quantity is WHICH centroids are selected, not HOW MANY
> directions the fit Gram retains. Rank count is not basis quality.**

Do not read Section 3 as the explanation for Section 2. They are independent.

Cost of the fix, measured rather than assumed: the full `(0,1024)` build took
349 s against 308 s for `(0,52)` at `nb=1024` — **+13% wall, +15 GB peak**
(81 GB of a 186 GB node). The correct window is cheap.

## 3. Separate observation: the fit Gram's retained rank saturates

> **CAVEAT — read this first.** Every row below was measured on a basis built
> with the CLAMPED `(0,52)` prune window, i.e. on a rank-deficient selection
> (Section 2). The saturation is real and reproducible, but its magnitude, its
> onset, and whether it constitutes a practical limit at all are **pending a
> re-measurement with a correct window**. Nothing here should be used to choose
> `nband` until that is done.

At run time the charge-channel factor truncates the `CCT` spectrum at
`lambda_max * zeta_rcond`. `isdf/core.py` prints the outcome every run as
`[zeta rank_truncate/distributed]`:

| nb | N_mu | n_pad | n_keep | retained | lambda_max | truncation floor |
|---|---|---|---|---|---|---|
| 512 | 4951 | 4992 | 4183 | 83.8% | 0.17610 | 1.76e-09 |
| 512 | 6947 | 6976 | 4570 | 65.5% | 0.21562 | 2.16e-09 |
| 1024 | 10015 | 10048 | 6700 | 66.7% | 0.35824 | 3.58e-09 |
| 1024 | 15007 | 15040 | 7108 | 47.3% | 0.51229 | 5.12e-09 |

    nb = 512 : N_mu 4951 -> 6947   (+40.3%)  gives rank 4183 -> 4570  (+9.3%)
    nb = 1024: N_mu 10015 -> 15007 (+49.8%)  gives rank 6700 -> 7108  (+6.1%)

Adding 50% more centroids buys ~6% more retained rank. Separately, at
`nb=1024, N_mu=15007` the cut discards **7,932 of 15,040 modes (53%)** while the
f64 noise floor of a Gram with `lambda_max = 0.512, n = 15040` is only
`~eps*lambda_max*sqrt(n) ~ 6.3e-15` — six orders of magnitude below the cut.

**Do not conclude from that arithmetic that a smaller `zeta_rcond` would help.**
`gw_config.py` documents a measured sweep in which `1e-8` is the *low end of an
over-complete recovery plateau* spanning `1e-8..1e-4`: at MoS2 4x4/1204c,
`1e-10` only partially recovers (**MAE 1.4 eV vs BGW**), and bulk Si 4x4x4/960c
genuinely has eigenvalues below the cut. Truncation here is a **cure for
over-completeness**, not numerical hygiene. `zeta_rcond` is a knob to check, not
a constant to distrust.

## 4. Choosing `nband` and `N_mu` in practice

1. **Select the basis on the right window.** With the fix, the default prune
   window is the full WFN conduction window — a superset of any deck's `ncond`.
   If you narrow it with `--prune-n-cond`, the rank gate (Section 5) refuses if
   that costs independence.
2. **Read the rank line.** `After pruning: N centroids (rank=R)` must have `R`
   equal to the requested orbit/point count. This is now enforced.
3. **Watch `n_keep / n_pad`** in `[zeta rank_truncate]` as an observable and
   report it — but see the Section 3 caveat before treating any value as a limit.

**On a "maximum usable `nband` per cutoff"**: an earlier draft of this document
asserted such a rule and placed it at `nb=1024` for this deck at 30 Ry. **That
assertion is withdrawn.** It was inferred from the Section 3 table, which was
taken on rank-deficient bases, and the `nb=1024` failure it rested on is now
explained by Section 2 and fixed. Whether a rank-driven `nband` ceiling exists
at all is open; re-measure with a correct prune window before publishing one.

Likewise the `N_mu ~ 6-14x nband` sizing heuristic: it did not predict the
Section 2 failure (the failing points sat inside the band, at 9.8x and 14.7x),
so **membership in that band is not evidence of adequacy**. That does not refute
it as a starting point — it simply is not a check.

## 5. Gate: centroid rank assertion

`kmeans_cli` now **refuses** to write a centroid set whose pivoted-Cholesky rank
falls short of the number of directions requested (default tolerance 1%,
override `LORRAX_CENTROID_RANK_TOL`). The message names the likely cause and the
exact flags that fix it.

Rationale: a rank shortfall does not change the number of points in the file, so
nothing downstream can notice it. It has to be caught where it happens.

**Behaviour change to be aware of**: because the default prune window is now the
full WFN window, `max_band = nbands` rather than the old clamped 52. A deck with
`nbands > 0.5 * ngkmax * nspinor` will now hit the pivoted-Cholesky
50%-of-basis refusal in `centroid/pivoted_cholesky.py` where it previously — and
wrongly — succeeded. That refusal is correct (pair densities of bands beyond
half the basis cannot be ISDF-resolved), but it is a change. On this deck the
ceiling is `nb <= 1964` at 30 Ry and `nb <= 3597` at 45 Ry, so `nb=1024` and
`nb=2048` both clear it.

## 5b. The rule under TWO band counts (`number_bands_chi` / `number_bands_sigma`)

Since 2026-08-16 a deck can size the chi0/W band sum and the Sigma band sum
independently (`docs/input_reference.md`). That splits the sentence this whole
document is built on — *"the ISDF basis must be selected against the band window
`Sigma_c` actually consumes"* — because there are now two consuming windows.

**The rule is `max`, and it is not a convenience.** The interpolation basis has
to span the pair densities of whichever consumer reaches higher:

    ISDF window top  =  max(number_bands_chi, number_bands_sigma)

The psi is loaded once over `[b0, b4)` with `b4` the padded top of the LARGER
count, the zeta fit runs on that window, and the SMALLER consumer takes a
narrower slice inside it. Sizing the basis by the smaller count would rebuild
this document's Section 2 failure exactly one index over: the larger sum would
consume pair densities the basis was never fitted to represent, and — as Section
1's table shows — every gate in the suite would still pass.

Where it is enforced, so it cannot drift back to a `min` under a refactor:

| what | where |
|---|---|
| the `max` itself | `gw_config.BandCounts.isdf`, one property, one definition |
| the invariant at the consuming seam | `gw_init.assert_isdf_window_is_the_max`, called from `fit_zeta`; refuses, quoting this document |
| "which count won, and what the fit was built for" | logged every run by `BandCounts.describe()` and by that assert; a silent `max` is not acceptable |
| tests | `tests/test_band_count_split.py` §4 and §7 |

**`zeta_nband` still narrows, and now says what it undercuts.** Narrowing the
fit below a band sum's top is a legitimate request (the BSE's Galerkin capacity
bound is why the key exists), but the consumer left above it is then running on
an extrapolated zeta basis — this document's mechanism. That is now reported per
consumer, by name, every run.

**Centroid SELECTION is a separate object and is not covered by the above.**
The prune window belongs to `centroid/kmeans_cli.py` and is chosen when the
centroid file is built, before any deck names a band count. Its default is the
full WFN conduction window (Section 5), which is a superset of any deck's
`max(chi, sigma)`, so the default composes correctly. If you narrow it by hand
with `--prune-n-cond`, narrow it to at least `max(chi, sigma)` — not to the
smaller of the two. **Not verified here**: no split-deck centroid set has been
built against a hand-narrowed prune window.

## 6. Gate: pinned Sigma reference

One fixed, cheap configuration is re-run and its QP gap asserted against pinned
values, so a `Sigma_c`-only corruption cannot again pass a full gate suite:

- **Reference**: MoS2 4x4x1 30 Ry, `nval 26 / ncond 230 / nband 256`, 2475
  orbit-closed centroids, P=64.
- **Quantity**: `eqp0` and `eqp1` indirect QP gaps.
- **Pinned**: `eqp0 = 3.5819 eV`, `eqp1 = 3.2516 eV`.
- **Tolerance**: `1e-3 eV`. The same configuration recomputed is deterministic,
  so the honest expectation is `~1e-6`; `1e-3` leaves room for compile and
  threading nondeterminism while still catching the Section 2 failure (2.8 eV,
  i.e. 2800x the tolerance) by three and a half decades.
- **Cost**: 3.9 node-hours. Run before accepting a new size point, and on every
  change to the Sigma or ISDF path.

Measured on adoption: `|d| = 0.00e+00 eV` on both quantities, bit-exact across a
commit range and with the FFI/sharded feature set on.

## 7. Related limits found alongside this (recorded, not fixed here)

- **The collective-payload bound is mis-calibrated and its floor is silent.**
  `_DEFAULT_COLLECTIVE_CHUNK_MB = 128` with a `max(1, ...)` floor: the chunker
  splits along the q axis only, so once ONE q's collective exceeds the bound it
  has no remaining granularity and the payload grows as `N_mu^2` unchecked.
  Measured `max collective/exec`: 642.9 MB (`N_mu` 6947) -> 926.0 MB (10015) ->
  1386.1 MB (15007) -> 1773.2 MB (24933), i.e. up to **13x the bound**, in every
  case with `q_block = 1` already. **No failure has been attributed to this** —
  runs at 1386 MB and 1773 MB were both healthy when last observed. It is a
  silent bound violation and a mis-calibration (the 128 MB figure predates the
  MPI/mlx transport), not a demonstrated wall. The floor should be made loud
  when it cannot honour the bound, and finer-than-q chunking is the fix if a
  wall is ever demonstrated.

- **`Sigma(omega)` requests a 10.25 GiB HDF5 chunk at `nb=1024`** —
  `n_omega * min(16, nk) * nb^2 * 16`, against HDF5's hard 4 GiB
  `H5Pset_chunk` limit. Runs survive only because the FFI slab backend
  **no-ops** `chunks`; `_slab_io_mpi_host` and `_slab_io_allgather` honour it.
  **The same run at the same shape succeeds under PHDF5_FFI and refuses under
  PHDF5_HOST.** Binds at `nb=1024` for `n_omega >= 16` and `nb=2048` for
  `n_omega >= 4`. A portability trap, not a size limit.
- **The chunk planner does not model the sigma omega cube.**
  `gflat_memory_model.py` covers only the ISDF chunk plan, so replicated and
  sharded runs print an identical HWM estimate although the replicated layout
  additionally holds `n_omega * nk * nb^2 * 16` per rank (~11 GB at `nb=1024`,
  ~44 GB at `nb=2048`, 90 GB at `nb ~ 2929`). Planner-vs-measured agreement
  quoted for this model is an ISDF-stage statement only.
- **The restart tensor** scales as `N_mu^2`: 26.5 GB (`N_mu` 6947), 56.6 GB
  (10015), 123.2 GB (15007), projected 564 GB at 32059. `restart = false` means
  *compute fresh AND write*. ~~there is no key that computes fresh without
  writing.~~ **CORRECTED 2026-08-09: there is — `write_restart_tensors = false`
  (`gw_config.py` `_DEFAULTS`, documented in `docs/input_reference.md`) skips
  every dataset with one rank-0 line, measured at 4.5 s of a ~21 s Si warm wall
  and 2.01 GB.** This paragraph is the one place that claim costs the most,
  since it is the document about the regime where the artifact reaches hundreds
  of GB. Note the two keys are independent and answer different questions:
  `write_restart_tensors` decides *whether* the file is written at all (for runs
  that DISCARD it — a BSE run against such a directory refuses on the missing
  file), while `restart_q_storage` decides *on which q-set* it is stored for runs
  that keep it.
