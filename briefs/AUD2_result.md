# AUD2 result — round-two adversarial audit

**Measured first:** two of the three branches have reproducibility defects.
On `origin/fix/sc-anchor-cycle-2026-08-31`, the same accepted 1.05 eV band was
unprotected on a direct evaluation and protected when a 0.99 eV rejected
trial ran first.  On `origin/perf/plan-wall-2026-08-31`, six of six fixed-rank,
fixed-node fits changed weights between cold and quick-seeded production
solves; the largest `max|delta weight|` was **1.3162**.

The SC probe used the production `build_omega_band_partition` constructor with
a `[-1,+1]` eV grid and the claimed `0.125` eV margin.  Direct 1.05 eV from an
unprotected start returned `false`; 0.99 eV then 1.05 eV returned `true`; 1.20
eV returned `false`.  This tests starting-spectrum/evaluation-order dependence
and the claimed 1.20 eV release.  I did not construct a period-3 orbit.

The warm-start probe used six seeded synthetic reciprocal measures, rank 20,
96 candidate nodes, and identical selected times.  Cold/warm validation
residuals differed in all six cases.  Max weight differences were **0.00850,
0.0999, 0.5770, 1.3162, 0.1483, and 7.06e-6**.  This directly falsifies
search-path independence.  I did not rerun the full frozen ±15 eV planner or
remeasure its wall reduction.

I inspected `origin/fix/pointwise-dp-2026-08-31` and confirmed it enumerates
one constraint per supplied pointwise budget, but the funded interval ended
before a valid 12/24/48-window and P=1/4/16 timing matrix could be collected.
I therefore did **not** test its scaling claim, cross-P determinism, or budgets
at frequencies absent from the supplied enumeration.

Defects and exact affected lines are recorded in `KNOWN_LORRAX_ISSUES.md`.
The prescribed CPU gate is the verification for this documentation-only
branch; no GPU path was changed, so the four-GPU rule's unit/CPU exemption
applies.  Branch: `audit/round2-2026-08-31`.
