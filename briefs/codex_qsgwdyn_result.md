# Dynamic self-consistent QSGW result

Lane weight: **heavy** — this changes the basis contract of the dynamic
self-consistent output path and requires a full P=4 Na validation.

- Branch: `feat/qsgw-dynamic-2026-08-31`
- Starting point: `41ad4dea9a42812fe9e0d2a2c41e78ff0663f785`
- Scope: rotate the converged dynamic correlation cube from the last QP
  compute basis to the DFT output basis, then remove the three fail-closed
  guards and their two documentation restrictions atomically.
- CPU evidence: the exact campaign CPU environment ran
  `tests/test_fixed_sigma_evsc.py`, `tests/test_sigma_result_basis.py`,
  `tests/test_qp_solver_config.py`, and `tests/test_layering.py`: **150
  passed, 1 skipped** in 78.14 s.  A separate four-logical-device CPU
  execution of
  `test_full_sigma_cube_rotation_stays_two_axis_sharded_on_p4` passed in
  1.91 s; it matched the explicit host `U C U†` transform and retained
  `P(None,None,'x','y')`.
- P=4 Na evidence, SC convergence, planner costs, artifact checker, and
  fixed-point diagonal comparison: pending.

## Validation preconditions observed

- The supplied `-10..+10 eV` arm currently refuses in the delivered
  planner before Sigma: measured crossing span `A=125.6377502`; splitting
  down to one omega row still leaves `A=86.0280012`, beyond the widest
  certified shipped HGL table (`A=60`).  This is the required product-window
  refusal, not a reason to add direct pair evaluation.
- Under the repository's default strict band-boundary policy the same arm
  refuses before planning: its named 48-band chi edge has no spare WFN band
  with which to certify closure, and its 24-band Sigma edge splits a
  degenerate multiplet (nearby legal edges are 20, 26, and 46).  The legacy
  arm wrapper uses `LORRAX_BAND_DEGENERACY=snap`; the machine contract says
  never to use that diagnostic override to make a gate pass.
