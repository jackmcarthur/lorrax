# Known LORRAX issues

## 2026-08-31 round-two adversarial audit (AUD2)

### rCROP trial order changes the protected-band partition

**Affected tree:** `origin/fix/sc-anchor-cycle-2026-08-31` at `8b3ed208`,
`src/gw/band_partition.py:264`, `src/gw/sc_iteration.py:2505-2507` and
`:4035`.

The Schmitt decision is carried through every rCROP map evaluation, including
trials that the accelerator may reject.  A one-band call through
`build_omega_band_partition` with grid `[-1, +1]` eV and margin `0.125` eV
gave different protection for the same accepted `1.05` eV spectrum:

| evaluation order | protected at accepted 1.05 eV |
|---|---:|
| accepted directly from an unprotected start | false |
| trial at 0.99 eV, then accepted at 1.05 eV | true |

The trial first gains protection at 0.99 eV; the deadband then retains it at
1.05 eV.  A subsequent 1.20 eV spectrum did drop protection, confirming the
advertised far-edge behavior but not order independence.  Updating
`_partition[0]` unconditionally at line 4035 lets a rejected trial change the
next accepted map.  Hysteresis state must follow accepted iterates, not the
raw map-call sequence.

### Warm-started IRLS changes the fitted rule at fixed nodes

**Affected tree:** `origin/perf/plan-wall-2026-08-31` at `95c7160b`,
`services/minimax/src/minimax/roq_fit.py:320-321`, `:604-606`, and `:613`.

For six deterministic synthetic measure problems, rank 20 and identical QDEIM
times, a 45-iteration cold production solve was compared with the same solve
seeded by its 16-iteration quick fit.  All six returned different production
weights.  `max(abs(cold.weights - warm.weights))` was respectively
`0.00850, 0.0999, 0.5770, 1.3162, 0.1483, 7.06e-6`; validation residuals also
differed in all six cases (for seed 3, `1.1711663278e-9` cold versus
`1.1410965507e-9` warm).  Thus the rule is search-path dependent even before
line 606's stronger behavior, which returns an accepted quick search fit
without any production refit.  A path-independent receipt needs a canonical
final solve whose initialization and iteration schedule do not depend on
which ranks the search cached.
