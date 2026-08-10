# The downfold driver

## What it is for

You run a GW calculation once, with hundreds of bands and a large ISDF
centroid set, because that is what Σ needs. Then you want to do exciton bands
and BSE work, and the BSE only ever looks at a few dozen bands around the gap.
Carrying the full centroid basis into that work is paying Σ's price for the
BSE's problem, over and over, every time you change a BSE parameter.

The downfold is the step in between. It reads the large calculation's restart
bundle, chooses a small centroid set against the bands you actually intend to
consume, redefines the wavefunction-at-centroid coefficients on that smaller
set, and writes a restart bundle **in the same format at the smaller size**.
That last point is the whole design: the small bundle is a restart bundle like
any other, so `bse.bse_jax` reads it with no code change, no flag and no
knowledge that a downfold happened. You point that driver at a different
directory and everything else is as it was.

!!! warning "`bse.exciton_bands` does not yet read a downfolded bundle"
    This page used to promise the drop-in for *every* BSE consumer. Measured
    2026-08-10 on a 936 → 189 downfold of a silicon 4×4×4 parent, the exciton-band
    driver refuses on the small bundle for three separate reasons, and only the
    first has a workaround: it reads the deck's `centroids_file`, which the bundle
    carries only when the downfold was given `parent_centroids_file`; its
    htransform leg needs a Galerkin basis spanning `nk·nb` (1280 there against
    μ_S = 189) and refuses with `the Galerkin coefficients are NOT orthonormal`
    even after the htransform window is narrowed to the retained window; and its
    default `--vq-mode interp` needs a full-BZ `zeta_q.h5`, which the downfold
    does not write. μ_S is sized against the retained window's pair-density rank,
    which is a much smaller number than the Galerkin bound, so these are
    different criteria rather than a tuning problem. Today the small bundle
    serves `bse.bse_jax`; the exciton bands still want the parent.

Measured on silicon with a 960-centroid parent and a 20-band retained window:
191 centroids out, a five-fold reduction in μ and a twenty-five-fold one in the
storage of every (μ, μ) tensor, with the lowest twenty exciton eigenvalues
drifting 37.4 meV MAE. **How much you can compress depends entirely on whether the
parent basis was over-complete for your window**: the same deck's shipped
480-centroid set has no redundancy on a 20-band window at all, and downfolding
it destroys the spectrum rather than compressing it. The driver tells you which
situation you are in — that is what the refusal and the error bar below are
for — but it cannot create redundancy that the parent does not have.

The measurement behind that paragraph is `DOWNFOLD_S1.md` §3(c), which is a
**campaign report and is not in this repository** — it lives with the rest of the
2026-08-08 BSE campaign artifacts. Nothing on this page depends on reading it, and
where a number from it is quoted here it is quoted in full, but do not go looking
for the file in a checkout.

Nowhere does this tree tell you how to *build* an over-complete parent, and that
gap is worth naming because it is the first thing a reader of this page needs.
The guidance in [drivers.md](drivers.md) is "run the GW stage at a generous μ_L",
with no number and no invocation, and the one concrete figure anywhere — the
960-centroid parent above — is reported as a measurement rather than as a recipe.
For orientation, the parent used for the 2026-08-10 measurements on this page was
built with `python3 -m centroid.kmeans_cli 900 --seed 42 --prune-n-val 8
--prune-n-cond 52` on the `si_bse_debug` WFN, which in orbit mode delivered 936
points; that produced a window rank of 189, and, as the `mu_small` section below
records, was still not enough for a usable BSE spectrum after compression. Sizing
a parent for a downfold is unsettled work, not a documented procedure.

Two numbers put that 37.4 meV in proportion, and it is worth holding both
before you choose a cut. The first is the floor of the machinery rather than of
the compression: a downfold that keeps every centroid over the full band window
reproduces the parent's own lowest twenty excitons to **0.010 meV** MAE, end to
end through an unmodified BSE driver. That is round-off through a pseudo-inverse
and nothing else, so none of the Gram, the selection, the solve, the congruence
or the writer is costing you anything measurable; the whole of that 37.4 meV is
what a five-fold cut in μ_S bought. The second number is the bar it has to
clear. Production BSE work wants better than 1 meV, and it is only
full-frequency, MPA-class studies that have any business tolerating something
like 10 meV. So read the aggressive demo as a demonstration of the mechanism
rather than as a recommended setting: choose μ_S for the accuracy target and
confirm the choice in the observable. That is the same rule `downfold_rcond`
and `eps_W` are both held to further down this page, and for these three
questions it is the only admissible evidence there is.

```
python3 -m gw.downfold_cli -i downfold.in
```

## It takes its own input file

The downfold is not a GW run and it does not read a GW deck. It reads a
finished calculation off disk, so the only things it needs to be told are where
that calculation is, which bands must stay faithful, how small the answer
should be, how much round-off amplification you authorise, and where to put the
result. That is six facts. A GW deck carries a hundred and forty, and pointing
this driver at one would make every one of them look like an input to a
compression that none of them bears on. Handing it a `[cohsex]` section is
refused by name.

The format is the same INI-ish text every LORRAX deck uses — one section
header, `key = value` lines, `#` comments — so if you can read a GW deck you can
read this one. Unlike a GW deck, an unrecognised key is **refused rather than
ignored**: this schema has no decade of history to protect, and a misspelt
`downfold_rcond` that silently keeps its default is exactly the class of quiet
wrong answer the rest of this work exists to prevent.

A complete input file for the silicon fixture:

```ini
# Downfold the si_bse_debug GW run onto the directions a 20-band window holds.
[downfold]

source_restart = /path/to/the/gw/run
output_restart = /path/to/the/small/bundle

band_range_left  = 0:20
band_range_right = 0:20

mu_small       = auto
downfold_rcond = 1.1e-6
```

`python3 -m gw.downfold_cli --print-schema` lists every key with its default.

## Input reference

### `source_restart` — required, no default

The finished GW calculation to compress. Either the run directory, in which
case the driver looks for `tmp/isdf_tensors_*.h5` inside it, or that `.h5` file
directly.

Restart bundles are named by centroid count rather than by run, so a directory
can legitimately hold several. The BSE driver resolves that ambiguity by taking
the newest and printing a loud warning, which is the right call for a driver
whose input file already named the run. This driver **refuses** instead, on the
grounds that it is the tool most likely to create the ambiguity in the first
place: its entire job is to put a second bundle at a different μ somewhere
nearby. If there is more than one, name the file.

Relative paths resolve against the directory holding the input file, so a
downfold deck sitting beside the GW run can say `source_restart = .`.

### `output_restart` — required, no default

A **directory**. The driver creates `<dir>/tmp/isdf_tensors_<μ_S>.h5` inside
it, which is the layout every BSE consumer already looks for. It may not be the
same directory as the source, for the reason just given.

### `band_range_left`, `band_range_right` — or `n_val`, `n_cond`

The retained band window, and the most important key in the file.

This is what the compression is faithful **to**. The small basis is selected
against this window and the transfer solve preserves the observable on this
window; bands outside it are not represented and were never meant to be. A
basis selected against one window and consumed on another is a measured
failure, not a hypothetical one: a GW run at `nband = 1024` whose centroids had
been pruned against a 26 × 52 window produced a quasiparticle gap of 0.36 eV
where the answer is around 3.1 to 3.7 eV, with a negative `eqp1`, and it passed
every gate in the suite. Rebuilding at identical everything and changing only
the prune window moved the answer from 0.3645 to 3.1350 to 3.7227 eV, monotone
in window width. A downfold is that same operation performed deliberately, and
this key is where the deliberation is written down.

Two spellings, and exactly one of them may appear:

- `band_range_left = lo:hi` and `band_range_right = lo:hi` — half-open,
  **absolute** band indices into the bundle's `psi_full_y` and `enk_full`,
  counting from zero. `0:20` means bands 0 through 19.
- `n_val = N` and `n_cond = M` — the valence/conduction shorthand, meaning
  `left = (0, N)` and `right = (N, N+M)`.

The two ranges are the two legs of the pair density ρ_mn = ψ*_m ψ_n. For BSE
work they should be **equal**: the BSE's direct and exchange kernels both
contract ψ legs that lie inside the retained window, so a symmetric fit covers
them exactly.

An **asymmetric** window (the retained bands on one leg, all bands on the
other) is what Σ would need, because Σ's internal band sum runs over the full
window while its outer projection does not. The driver accepts it, and says
loudly that no end-to-end Σ gate has been run on it. Measured cost, if you want
it: about a factor of two in μ_S, not the order of magnitude that was feared.

There is no default. This is a physics choice and the driver cannot guess it.

### `mu_small` — required, no default

How many centroids the small basis should have, or the word `auto`.

`auto` means "as many as the retained window has independent pair-density
directions at `downfold_rcond`". It is the only value you can choose without
having read the measurement report, and it is the largest value the driver will
accept — but read the next paragraph before treating it as a safe default.

!!! danger "`auto` is a ceiling, not a recommendation, and it silently produced a 2.1 eV error"
    Measured 2026-08-10, following this page's own guidance end to end. A silicon
    4×4×4 parent was built at 936 centroids, downfolded on a `0:20` window at
    `mu_small = auto` (→ 189) and `downfold_rcond = 1.1e-6`, and the lowest BSE
    eigenvalue compared against the same parent solved at identical settings:

    | bundle | μ | worst-q `eps_W` | lowest BSE eigenvalue |
    |---|---|---|---|
    | parent | 936 | — | **2.3449 eV** |
    | `rcond = 1.1e-6`, `auto` | 189 | 1.33e-2 | **0.2579 eV** (−2.09 eV) |
    | `rcond = 1e-8`, `auto` | 624 | 3.26e-3 | **1.2605 eV** (−1.08 eV) |

    Nothing refused. `eps_W` reported about one per cent and the bundle was
    written, which is the tripwire behaving exactly as the section below
    describes it — and is also why that section's warning deserves to be read as
    a hard limit on what `eps_W` can tell you rather than as a caveat. The error
    does fall as μ_S rises, so this is a sizing problem and not a defect in the
    transfer solve; but it falls slowly, and at μ_S = 624 — a compression of
    only 1.5× — the spectrum was still wrong by an eV.

    The honest reading is that on this deck the `auto` ceiling and the accuracy
    a BSE needs did not overlap anywhere, and that `auto` is where a sweep
    starts rather than where it ends. Size μ_S by sweeping it against the
    observable, exactly as the `downfold_rcond` section below insists, and treat
    a downfold whose observable has not been checked against its parent as
    unvalidated.

**The driver refuses when you ask for more directions than the window
contains**, and prints the number it measured. That refusal is not a failure
mode; it is the point of the exercise arriving early and cheaply. A 20-band
window holds roughly 190 independent directions on both decks that have been
measured — 196 on silicon, 185 on hexagonal boron nitride — and the numbers are
stable to two per cent as the pool of candidate points grows by a factor of nine
to twelve. A nominal "500-centroid small basis" on such a window is a fiction in
which two thirds of the basis is truncated away by the very solve that consumes
it.

### `downfold_rcond` — default `1.1e-6`

The relative eigenvalue threshold on the small basis's Gram: directions with
λ ≤ `rcond` · λ_max are discarded, so the truncated pseudo-inverse amplifies
round-off by at most 1/`rcond` **by construction**.

It is a cap on amplification, **not** a gap-finder. ISDF pair-density spectra
are smooth and have no knee, elbow or plateau to cut at, so every criterion
phrased as "cut at the separation" is inapplicable here; `common/rank_criterion.py`
carries the derivation and the measurements that refute the discrepancy
principle, the L-curve and generalised cross-validation against this exact
failure mode.

The default is a measured number rather than a round one. At a 20-band window
silicon holds 196 independent directions at 1e-6 and hBN holds 185, and both are
real ceilings — they do not move when you add candidate points. At 1e-8 the same
silicon window reports 693 and at 1e-10 it reports 1208, and neither of those is
a ceiling at all: they keep climbing with the size of the pool you were willing
to pay for. Buying μ_S = 500 on a 20-band window means keeping directions whose
eigenvalue is 3.5e-8 of the largest, and the anchor for what that costs is a
sweep in which retaining 41 % more rank moved a 2.2 eV gap by 5000 eV.

The only admissible evidence for changing this value is **observable**
convergence: sweep it and take the plateau in the energy, not in the spectrum.
That sweep is later work. This default is where it starts.

### `downfold_select_tol` — default: the kernel's own √ε

The pivoted-Cholesky stopping tolerance, relative to the largest initial Gram
diagonal. You will not normally set it.

**It is not the same knob as `downfold_rcond`, and it does not produce the same
rank.** Pivoted Cholesky stops on a residual Schur diagonal; the truncation
stops on an eigenvalue; and the residual decays much more slowly than the
spectrum. Measured on one Gram, the selection rank runs about three times the
eigenvalue rank at the same nominal number: at 1e-6, 588 selected points of
which about 195 carry an eigenvalue above 1e-6 of the largest. Setting both
knobs to the same value is the most natural mistake available here, which is
why the driver prints the two ranks side by side, labelled differently, on
every run.

`μ_small` is validated against the **eigenvalue** rank. The selection
certificate is a necessary condition, not a sufficient one: a basis can pass the
selection and still be two thirds rank-deficient at the solve.

### `mode` — default `cur`

`cur` selects the small basis as a **subset** of the parent's centroids. Both
operands of the fit are then submatrices of a single object, no second ζ fit is
needed anywhere, the new wavefunction-at-centroid coefficients are a literal
column slice of the ones already on disk, and the exact-reproduction test
becomes an algebraic identity rather than a hopeful tolerance.

`refit` — a fresh narrow-window k-means and a fresh ζ fit — is **refused**, and
refused rather than quietly demoted so that nobody reads a CUR result as a
refit one. The measured case against it: the parent's own k-means set already
certifies 194 of the 196 directions a 20-band window contains, so a refit would
buy at most one per cent more rank for the price of a second ζ fit. The key
exists so that the door stays open and so that anyone who wants it can ask for
it by name.

### `plan` — default `auto`

`auto` and `local` both mean the local plan, which is what exists today: the
linear algebra emits no block-cyclic factorisation and the result does not
depend on the process grid. `distributed` — μ tiled over a two-dimensional
process grid — is later work and is refused rather than demoted, because a
block-cyclic factorisation is a different (equally valid) numerical gauge and
silently changing gauge under an explicit request is precisely what this
codebase's demotion doctrine forbids.

### `report_residual` — default `true`

Compute and print the per-q error bar. Leave it on. See below for what it is;
its cost is two matrix multiplications at μ_L per q, the same cost class as the
compression itself, and turning it off leaves the run with no answer to "did
this work" that does not require a second calculation to compare against.

### `residual_refuse_above` — default: report only

Refuse to write the small bundle when the worst-q error bar exceeds this. Empty
means report and always write, which is the current default because nobody has
yet measured what a good error bar looks like on a production deck. Set it once
you know.

### `parent_centroids_file` — optional

The parent run's centroid coordinate table. When you give it, the driver writes
the kept rows out as a sibling centroid file and stamps its checksum onto the
small bundle, so the small basis can later be handed to a fresh GW run.

Without it the bundle is still complete for `bse.bse_jax` — the bundle format
carries no coordinates at all, only their checksum — and the driver says so
rather than stamping the parent's checksum, which would be a lie about which
points the tensors describe.

It is **not** optional if you intend to run `bse.exciton_bands`, which reads the
deck's `centroids_file` directly. Measured 2026-08-10, a small bundle written
without this key fails there in `file_io/centroids.py` with a bare
`FileNotFoundError: …/centroids_frac_936.txt not found.` — a naked numpy
traceback carrying none of the announce-or-refuse framing the rest of the tree
uses, and no hint that the missing file is a consequence of how the downfold was
configured. When you do give the key, the kept rows are written to
`<output_restart>/tmp/centroids_frac_<μ_S>_downfold.txt`, so the consuming deck's
`centroids_file` has to be repointed at that path as well; "nothing else moves"
does not hold on this path.

## What it prints, and what to read

Three numbers matter, and they are not interchangeable.

**The eigenvalue rank of the retained window's Gram.** How many independent
pair-density directions the window actually holds. `mu_small` is validated
against this one and the run refuses when you ask for more.

**The pivoted-Cholesky selection certificate.** A necessary but not sufficient
condition, roughly three times larger at the same nominal tolerance. It is
printed beside the first so the two cannot be confused.

**`eps_W(q)`, the error bar.** The relative error of the downfolded observable
on the retained window, per momentum transfer.

If you read only one, read the third. It is worth understanding why it exists,
because it is unusual: it needs no reference calculation. Substituting the
solution back into the fit shows that the downfolded observable is the
orthogonal projection of the exact one onto the space the small basis spans.
The residual is therefore orthogonal to the fit, Pythagoras holds exactly, and

    eps_W(q) = sqrt(1 - ||W_S||^2 / ||W||^2)

is not an estimate of the error — it *is* the error, computed from traces of
μ × μ objects, without ever forming the exact observable (which has millions of
rows). So every run carries its own accuracy statement, and a downfold that
was too aggressive says so on the spot rather than three hours later in an
exciton spectrum.

One consequence worth stating out loud, because it is the reason the code
applies no ridge anywhere on this path: a ridge would destroy the orthogonality
that makes the identity exact, and `eps_W` would go on printing a plausible
number that means nothing.

And one caveat, measured: `eps_W` is a **tripwire, not a transferable gate**.
Within one parent bundle and one cut it ranks configurations monotonically, but
the same `eps_W` of about one per cent produced a 37 meV exciton drift on one
parent and a 1.7 eV drift on another (`DOWNFOLD_S1.md` §3(c), a campaign report
not carried in this repository), and a third case is tabulated under `mu_small`
above, where `eps_W` of 1.3e-2 accompanied a 2.09 eV error. Set
`residual_refuse_above` to catch a downfold that has gone badly wrong; do not
read it as a promise about meV. The only admissible evidence for choosing the
cut remains convergence in the energy itself.

At q = 0 the head divergence contaminates both norms identically, so the ratio
stays meaningful there; the absolute norms at q = 0 are head-dominated and
should not be compared across q.

## What comes out

A restart bundle in the unchanged format, at the smaller μ: `V_qmunu`,
`W0_qmunu` with their readiness flags, `G0_mu_nu` transported as a vector,
`psi_full_y` sliced to the kept centroids, `enk_full` and the head scalars
carried through verbatim, the parent's Coulomb-kernel policy string re-stamped,
and the parent's band-window stamp preserved.

The band axis is **not** truncated. The retained window decides what the
compression is faithful to; it is not a truncation of the stored bands. Cutting
the band axis would renumber every band index in the bundle and move the stamp
that guards against exactly that class of mistake, so a consumer asking for
eight occupied states would silently get different states. Band-axis truncation
is separate, later work with its own renumbering contract.

Alongside the standard datasets the bundle carries a `downfold_provenance`
group recording what it is: the parent file and its centroid count, the kept
indices, the window, both tolerances, all three ranks, the retained rank at
every q and the error bar at every q. A downfolded bundle is deliberately
indistinguishable from a natively fitted one by shape — that is what makes it a
drop-in — but it is not the same object, and a reader that wants to know can
ask.

## Where it does not apply

Plasmon-pole and multipole reductions cannot be downfolded. `B_q` is a residue
and would transform correctly, but `Omega_q` is a pole *position* per matrix
element, and there is no change of basis that maps a table of pole frequencies
from one basis to another. Downfold the linear objects — V, W(0), W(probe),
each frequency or time slice — and re-fit the pole model in the small basis,
which is cheap at that size. Anyone who transforms `Omega_q` will get numbers
that look entirely plausible and are meaningless.

It also does not yet serve Σ. What the compression is faithful to is the window
it was fitted on, and that window is the BSE's shape — both BSE kernels contract
ψ legs that lie inside the retained window, so a symmetric fit covers them
exactly. Σ is the other shape, because its internal band sum runs over the full
window while its outer projection does not, and the asymmetric window that
expresses it is the one this driver accepts and announces as unvalidated for
precisely this reason: the algebra is identical, the measured price is about
twice the μ_S, and no end-to-end Σ gate has been run on it. So nothing here is
a claim about quasiparticle energies. Re-fitting Σ in a downfolded basis is
real, buildable, later work; today the small bundle serves the retained-window
BSE and the exciton bands built on it, and nothing upstream of them.
