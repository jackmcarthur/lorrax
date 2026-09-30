# The downfold driver

`gw.downfold_cli` compresses a finished GW restart bundle from its $\mu_L$
centroids onto a subset of $\mu_S$ of them, chosen for a retained band window,
and writes a restart bundle in the unchanged format at the smaller size.
`bse.bse_jax` and `bse.exciton_bands` read the result with no flag: point them
at the output directory. Every stored $(\mu,\nu)$ tensor shrinks by
$(\mu_S/\mu_L)^2$.

The compression is exact only where the parent basis is redundant for the
retained window. The driver cannot create redundancy the parent lacks, and no
rule for sizing an over-complete parent is established.

## Equations

For each momentum transfer $q$, $S_q$ is the $\mu_L\times\mu_L$ pair-density
Gram of the retained window (left bands $m$, right bands $n$), built by
`isdf.core.c_q_from_psi_sm`. Pivoted Cholesky on it selects the kept rows
$S \subset L$. The transfer is the least-squares fit in that metric,

$$
T_q = S_{SS}(q)^{+}\, S_{SL}(q),
$$

with the pseudo-inverse truncated at `downfold_rcond` (eigenvalues
$\lambda \le \mathrm{rcond}\cdot\lambda_{\max}$ are dropped, so round-off is
amplified by at most 1/rcond). No ridge is applied. Every stored object
transforms by $T$:

$$
V_S = T V_L T^\dagger,\qquad W_S = T W_L T^\dagger,\qquad
\zeta_S = \bar T\,\zeta_L,\qquad g_{0,S} = \bar T\, g_{0,L},
$$

and $\psi$ at the kept centroids is the column slice of the parent's (`mode =
cur`). $V = \zeta^\dagger v \zeta$ then reproduces $V_S = T V_L T^\dagger$; the
writer checks the $q = 0$, $G = 0$ column of $\zeta_S$ against $g_{0,S}$ on
every run. $T_q$ depends on $q$ only through $S_q$, so a ζ stored on the
q-IBZ wedge transports row by row and the child ζ is on the parent's q set.

The downfolded $W$ is the orthogonal projection of the parent's onto the span
of the small basis on the retained window, so the relative error on that
window needs no reference:

$$
\epsilon_W(q) = \sqrt{1 - \lVert W_S\rVert^2 / \lVert W\rVert^2}.
$$

At $q = 0$ the head dominates both norms; the ratio stays meaningful, the
absolute norms do not compare across $q$.

## Input

The driver reads its own `[downfold]` file, not a GW deck. The
[input reference](input_reference.md#downfold-the-downfold-input-file) owns
the keys and defaults; `python3 -m gw.downfold_cli --print-schema` prints
them. A `[cohsex]` section refuses, and an unknown key refuses.

```ini
[downfold]
source_restart = /path/to/the/gw/run
output_restart = /path/to/the/small/bundle
band_range_left  = 0:20
band_range_right = 0:20
mu_small       = 189
downfold_rcond = 1.1e-6
```

- **The retained window** (`band_range_left`/`band_range_right`, or
  `n_val`/`n_cond`) is what the compression is faithful to. Bands outside it
  are not represented. For BSE the two legs are equal: both BSE kernels
  contract ψ legs inside the window. An asymmetric window (the Σ shape) is
  accepted with a warning; no Σ check has been run on it.
- **`mu_small` is a budget in points, spent in whole symmetry orbits** when a
  symmetry map reaches the selection: the driver keeps the largest union of
  whole orbits, in pivot order, that does not exceed it, and prints requested
  and realized counts with the parent's orbit sizes. A budget below the first
  orbit refuses and lists the legal counts. Without a symmetry map the
  selection is by points and closure is not measured.
- **`mu_small = auto`** is the eigenvalue rank of the window Gram at
  `downfold_rcond`: the largest value the driver accepts. It is sized by rank,
  not by accuracy, and the run warns. Use it to learn the ceiling, then sweep
  downward against the BSE eigenvalues. A request above the ceiling refuses
  and prints the measured rank. A 20-band window holds about 190 directions
  at rcond 1e-6 (silicon 196, hBN 185), stable as the candidate pool grows;
  at 1e-8 and 1e-10 the count (693, 1208 on silicon) grows with the pool and
  is not a ceiling.
- **`downfold_rcond` caps amplification; it does not find a gap.** ISDF
  pair-density spectra have no knee (`common/rank_criterion.py`). Change it
  only on evidence of convergence in an observable.
- **`downfold_select_tol` is a different knob.** Pivoted Cholesky stops on a
  residual Schur diagonal, the truncation on an eigenvalue; at 1e-6 the
  selection keeps 588 points of which about 195 carry an eigenvalue above the
  cut. `mu_small` is checked against the eigenvalue rank.
- **`parent_centroids_file`** overrides the parent's centroid table. By
  default the table comes from the `isdf_header` of the parent's `zeta_q.h5`
  and is checked against the bundle's `centroids_charge_md5`.
- **`parent_input_file`** defaults to `cohsex.in` in the parent run directory.
  A raw-parent GW bundle uses it to authenticate and unfold the stored
  wavefunctions.

## Procedure

1. Run the GW stage at a $\mu_L$ that is over-complete for the window you will
   retain.
2. Write `downfold.in` and run `python3 -u -m gw.downfold_cli -i downfold.in`.
3. Read the three printed numbers ([below](#what-it-prints)).
4. Validate $\mu_S$: solve the same BSE on the parent and on the child with
   the same deck and flags, and compare the lowest eigenvalues against your
   accuracy target. The driver prints this line with the run's own paths:

    ```bash
    for d in /path/to/the/gw/run /path/to/the/small/bundle; do (cd "$d" && python3 -u -m bse.bse_jax -i cohsex.in --n-val 4 --n-cond 4 --lanczos 2>&1 | tail -40); done
    ```

    Under the default `--band-degeneracy strict` a band count that cuts a
    multiplet refuses and names the counts that work; use those on both legs.
    If the eigenvalues disagree, raise `mu_small`, widen the window, or build
    a larger parent.
5. Point the BSE drivers at `output_restart`.

There is no target-accuracy mode: step 4 is the only accuracy evidence.

## What it prints {#what-it-prints}

- **The eigenvalue rank of the window Gram**: the ceiling for `mu_small`.
- **The pivoted-Cholesky selection certificate**: necessary, not sufficient,
  and about three times the eigenvalue rank at the same nominal tolerance.
- **$\epsilon_W(q)$** per momentum transfer (`report_residual`).
  `residual_refuse_above` refuses to write the bundle above a worst-q value.

$\epsilon_W$ ranks configurations within one parent and one window. It does
not transfer to meV: on a silicon 4×4×4 lineage a median $\epsilon_W$ of
1.2e-2 accompanied a 42.6 meV error in the lowest exciton. Choose the cut by
convergence of the energy.

## Output

- `<output_restart>/tmp/isdf_tensors_<μ_S>.h5`: `V_qmunu`, `W0_qmunu` and
  their readiness flags, `G0_mu_nu` transported as a vector, `psi_full_y`
  sliced to the kept centroids, `enk_full` and the head scalars unchanged, the
  parent's Coulomb-kernel policy string and band-window stamp. The band axis
  is not truncated.
- `tmp/zeta_q.h5`: the transported ζ, which off-grid exchange in
  `bse.exciton_bands` interpolates. `--vq-mode interp` still requires
  `nq == nk`; `--vq-mode ongrid` needs no ζ.
- `tmp/centroids_frac_<μ_L>_parent.txt`: the parent's table.
  `bse.exciton_bands` fits its htransform leg in the parent basis, which must
  span `nk·nb`, and slices the result to the kept rows. If no table is found
  the run says so; `bse.bse_jax` is unaffected and `bse.exciton_bands`
  refuses.
- `tmp/centroids_frac_<μ_S>_downfold.txt`: the kept rows, with its checksum
  stamped on the small bundle, so a fresh GW run can use the small basis.
- The `downfold_provenance` group: the parent file and its centroid count, the
  kept indices, the window, both tolerances, the three ranks, the retained
  rank and $\epsilon_W$ at every q, requested and realized $\mu_S$, the paths
  of the three sibling files, and the wedge-storability residual with its
  tolerance.

**The child is written on the full BZ**, also from a wedge parent (the reader
unfolds the parent first). Orbit selection makes the kept set closed under the
parent's centroid permutations, `child_unfold_tables` restricts the parent's
unfold tables to it, and every run checks them by unfolding the child's wedge
block and comparing against the full-BZ child. The wedge is still not stamped:
`symmetry_maps.qirr_store` requires a `CentroidClosureVerdict` measured over
the symmetry operations of a WFN this driver does not open.
`gw.downfold.orbit_complete_keep` is an offline tool for a kept set from
elsewhere and is not on the selection path.

## Limits

- **Pole models do not transform.** `Omega_q` is a pole position per matrix
  element and has no change of basis. Downfold the linear objects (V, W at
  each frequency or time) and refit PPM or MPA in the small basis.
- **Not validated for Σ.** The small bundle serves the retained-window BSE and
  the exciton bands built on it.
- `mode = refit` (a fresh k-means and ζ fit on the window) and
  `plan = distributed` refuse.
