# Two BSE rank cuts sit outside `common/spectral_closure`, and this lane could not wire them (2026-08-11)

Filed by the BSE-solver cleanup lane
(`cleanup/bse-solvers-dedupe-2026-08-11`), whose charter was
BEHAVIOR-PRESERVING dedupe onto the services. Both findings below are real
and neither could be fixed under that charter, so they are recorded rather
than half-done. Neither is a red test today.

## What was looked for

`2026-08-10-spectral-cut-closure.md` swept `src/` and `services/` and wired
six sites; its sibling amendment
(`2026-08-10-distrib-la-rank-closure.md`, reverted whole on the owner's
ruling) established that the guard belongs at the sites that *decide* a
truncation. This lane re-ran that question over the BSE-owned solver files.
Three places in scope look at a spectrum and pick a rank:

| site | what it does | guarded? |
|---|---|---|
| `bse/bse_feast.py::_rayleigh_ritz` | **CLIPS** overlap eigenvalues at `s_floor = max(s_cutoff·s_max, 1e-30)`; the subspace dimension is never reduced | not a cut — nothing to guard, and the docstring already says why the clip was chosen over a truncation |
| `bse/bse_pseudopoles.py::_orthonormalize` | `keep = s_evals >= s_cutoff·s_max` on the eigh spectrum of the filtered-vector overlap Gram; the retained columns become the basis for every downstream pseudopole | **NO** |
| `solvers/davidson.py::_whiten_rank_revealing` | `keep = e > 1e-10·e_max` on the Gram of the CGS2-projected residual block; the sub-threshold columns are zeroed and sliced off | **NO** |

The first row is the reason the earlier sweep found nothing here: it was
looking for cuts, and the one FEAST site that looks like one is not one.

## Why neither was wired

**`_whiten_rank_revealing` is inside a `jit`.** `close_keep_mask` — the
device face — would fit, and this is exactly the shape it was built for
(`isdf/core` uses it the same way). But applying it CHANGES NUMBERS: under
the `drop_block` default a degenerate block straddling the cut is dropped,
so `rank` falls and the Davidson expansion admits fewer directions. That is
a numerics change to the shipped eigensolver, which a behavior-preserving
lane may not make. `rank_criterion.select_rank` is not an alternative here:
it is a host function over a Python list and cannot be traced.

**`_orthonormalize` is host-side, and the contract does not match.**
`select_rank(s_evals, s_cutoff)` counts `σ > σ_max·rtol`; the site keeps
`σ >= σ_max·rtol` and additionally floors the threshold at `1e-30`. The
`>=`/`>` split only bites on an exact tie and the absolute floor only binds
below `s_max ~ 1e-24`, so the two agree on every realistic spectrum — but
"agree in practice" is not "same contract", and the mandate is explicit that
a mismatch is filed, not papered over. Adding the closure guard on top is
the same numerics change as above.

## What a future lane should do

Both are one-line wirings behind a measurement, not a design question. The
work is the A/B, not the edit:

* `_whiten_rank_revealing`: run the Si BSE Davidson deck with and without
  `close_keep_mask`, and report `rank` per iteration, the matvec count, and
  the twenty exciton energies. The prior is that the guard never fires (the
  sub-threshold tail is CGS2 round-off at ~1e-30 relative, not a degenerate
  block), in which case wiring it is free and the gate becomes real.
* `_orthonormalize`: decide whether the `>=` and the `1e-30` floor are load
  bearing. If not, `select_rank` + `resolve_spectral_cut` replace four lines
  and the site joins the six already wired.

Do NOT reach for `distrib_la` for either: it has no rank-revealing
operation and deliberately no closure surface
(`2026-08-10-distrib-la-rank-closure.md`, reverted on the owner's ruling).

## Second finding, unrelated to the closure: a silent twin of the pseudopole physics

`bse/pseudopoles_sweep.py::_reconstruct_from_intermediates` re-implements
the brightness eigendecomposition, the bright Rayleigh-Ritz, the
out-of-window filter, the J-norm normalisation and the anti-resonant pole
construction that `bse/bse_pseudopoles.py::run_pseudopoles` performs — the
same algebra in the same order, on H5-loaded intermediates instead of live
ones. They are not shared code, so the sweep tool can drift from the
production path silently, and the sweep tool is what a `p_keep` choice is
made from. They differ TODAY only in reporting (the production path prints
discarded Ritz values, J-norm statistics and a non-positive-J-norm warning;
the sweep filters silently), which is precisely why factoring them together
is not a behavior-preserving edit and was not attempted here. Worth an owner
call on whether the sweep should call the production routine and swallow its
prints, or whether the duplication is accepted with a pinning test.
