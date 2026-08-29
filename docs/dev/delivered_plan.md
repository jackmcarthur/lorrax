# Delivered-error Sigma planning

The delivered pathway is an opt-in MPA correlation-Sigma planner.  It fits
the minimax service's reciprocal rule on the spectral measure each causal
branch actually delivers, instead of certifying rectangular panes that bound
support the branch may barely occupy.  The incumbent pane/window planner is
unchanged and remains the default.

Enable it with:

```bash
export LORRAX_SIGMA_PLAN=delivered
```

Unset, blank, or `panes` selects the incumbent path.  Any other spelling
refuses before planning.

For each branch, the planner forms signed internal sums `s = E + Omega` for
conduction and `s = E - Omega` for valence, with measure mass proportional to
the occupation weight and `|B|` (and `|state amplitude|` when supplied).  It
adds the requested Sigma regularization to every pole width before fitting.
Tuples within 2 eV of the branch's requested real-frequency segment remain
raw; smooth far support is reduced to an approximately 240-cell equal-mass
histogram.  Positive- and negative-frequency pole arrays are arguments per
branch, so the planner itself makes no time-reversal identification.

The result is one ordinary `SharedSigmaWindow` per branch.  It uses the same
shared tau kernel and `DeviceOmegaAccumulator` as the pane plan.  The adapter
reverses valence time nodes to match the executor's positive hole-energy
storage, folds `exp(-eta*t_exec)` into each weight exactly once, and carries
the executed G*W orientation as a global `-1` prefactor.

This is measured, not continuum-certified: `max_error` is the delivered error
on the histogram-plus-raw measure.  It does not change the MPA screening fit,
the pole-store format, or the spatial executor.  The current store exposes one
positive-pole tensor, so the driver explicitly supplies that tensor to each
branch; callers with genuinely independent branch pole sets can use the
planner API directly.  Planning currently reads the complete pole axis once
(still sharded in the spatial dimensions) before the executor returns to its
configured pole batches.
