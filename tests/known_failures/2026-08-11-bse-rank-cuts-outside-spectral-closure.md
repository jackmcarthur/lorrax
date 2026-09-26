# A BSE rank cut sits outside `common/spectral_closure`, and this lane could not wire it (2026-08-11)

Filed by the BSE-solver cleanup lane
(`cleanup/bse-solvers-dedupe-2026-08-11`), whose charter was
BEHAVIOR-PRESERVING dedupe onto the services. The finding below is real
and could not be fixed under that charter, so it is recorded rather than
half-done. It is not a red test today.

## What was looked for

`2026-08-10-spectral-cut-closure.md` swept `src/` and `services/` and wired
six sites; its sibling amendment
(`2026-08-10-distrib-la-rank-closure.md`, reverted whole on the owner's
ruling) established that the guard belongs at the sites that *decide* a
truncation. This lane re-ran that question over the BSE-owned solver files.
Two places in scope look at a spectrum and pick a rank:

| site | what it does | guarded? |
|---|---|---|
| `bse/bse_feast.py::_rayleigh_ritz` | **CLIPS** overlap eigenvalues at `s_floor = max(s_cutoff·s_max, 1e-30)`; the subspace dimension is never reduced | not a cut — nothing to guard, and the docstring already says why the clip was chosen over a truncation |
| `solvers/davidson.py::_whiten_rank_revealing` | `keep = e > 1e-10·e_max` on the Gram of the CGS2-projected residual block; the sub-threshold columns are zeroed and sliced off | **NO** |

The first row is the reason the earlier sweep found nothing here: it was
looking for cuts, and the one FEAST site that looks like one is not one.

## Why it was not wired

**`_whiten_rank_revealing` is inside a `jit`.** `close_keep_mask` — the
device face — would fit, and this is exactly the shape it was built for
(`isdf/core` uses it the same way). But applying it CHANGES NUMBERS: under
the `drop_block` default a degenerate block straddling the cut is dropped,
so `rank` falls and the Davidson expansion admits fewer directions. That is
a numerics change to the shipped eigensolver, which a behavior-preserving
lane may not make. `rank_criterion.select_rank` is not an alternative here:
it is a host function over a Python list and cannot be traced.

## What a future lane should do

It is a one-line wiring behind a measurement, not a design question. The
work is the A/B, not the edit:

* `_whiten_rank_revealing`: run the Si BSE Davidson deck with and without
  `close_keep_mask`, and report `rank` per iteration, the matvec count, and
  the twenty exciton energies. The prior is that the guard never fires (the
  sub-threshold tail is CGS2 round-off at ~1e-30 relative, not a degenerate
  block), in which case wiring it is free and the gate becomes real.

