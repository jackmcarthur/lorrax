# AUD-P result — planner robustness audit

Heavy investigation lane.  **Three probes found three adversarial failures:**
the pointwise selector refused a feasible **3-node** plan; the advertised
three-round tightener stopped after **1** tightened round; and the SC partition
formed a stable **period-2** protected-set cycle.  One claimed invariant was
clean: the insulating static interval was bit-identical.

## Measured results

| claim probed | achieved observation | verdict |
|---|---|---|
| (7) pointwise budget, branch `eed87297` | selector refused; an explicit 3-node choice cost `[0.9, 0.1]` under `[1.0, 0.5]` | defect: scalar DP pruning loses a pointwise-feasible choice |
| (4) compounded tightening, base `5f750a77` | refused after selection call 3; call 4 was configured to accept; only one tightened allowance (`7.2e-5` from `1.6e-4`) was fitted | defect: later two tightening rounds are unreachable |
| (3) per-map partition, base `5f750a77` | masks `[1,1]`/`[1,0]`, upper energies `1.061267292017`/`0.990000000000 eV`, off-diagonal `0.2`/`0 eV`, repeated for 6 maps | defect mechanism demonstrated; no material-deck incidence claimed |
| (2) insulating Laplace floor, base `5f750a77` | plumbing passed `None`; old/new `x_min=0x1.f5c28f5c28f5cp-2`, `x_max=0x1.7ae147ae147aep+0` | clean bill: both interval endpoints bit-identical on the gapped probe |

The first reproducer executed `_select_rules` extracted verbatim from
`origin/fix/pointwise-budget-2026-08-31`.  The second called the production
`build_delivered_sigma_windows` with only fit/selection outcomes controlled.
The third used production partition construction and partition application.
The insulating probe used production `build_static_quadrature` with the solve
replaced only by an argument capture, so it measured the exact floating-point
interval delivered to the solver.

Defects, file:line loci, and complete failing cases are in
`KNOWN_LORRAX_ISSUES.md`.  Claims (1), (5), and (6), and P=1/4/16 behavior of
the newly amended branches, were not measured in this sprint.  The base already
contains an emulated P=1/4/16 determinism test; that is not evidence for the two
later branch tips.

Verification: prescribed CPU gate pending at report creation.  No GPU leg is
owed: all changes are audit documentation and all probes are scalar/CPU cells,
so the four-GPU rule's CPU-cell exemption applies.  Branch:
`audit/planner-robustness-2026-08-31`.
