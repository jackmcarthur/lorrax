# Dynamic self-consistent QSGW result

Lane weight: **heavy** — this changes the basis contract of the dynamic
self-consistent output path and requires a full P=4 Na validation.

- Branch: `feat/qsgw-dynamic-2026-08-31`
- Starting point: `41ad4dea9a42812fe9e0d2a2c41e78ff0663f785`
- Scope: rotate the converged dynamic correlation cube from the last QP
  compute basis to the DFT output basis, then remove the three fail-closed
  guards and their two documentation restrictions atomically.
- CPU evidence: pending.
- P=4 Na evidence, SC convergence, planner costs, artifact checker, and
  fixed-point diagonal comparison: pending.

