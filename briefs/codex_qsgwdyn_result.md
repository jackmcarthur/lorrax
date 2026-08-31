# Dynamic self-consistent QSGW result

Lane weight: **heavy** — this changes the basis contract of the dynamic
self-consistent output path and requires a full P=4 Na validation.

- Branch: `feat/qsgw-dynamic-2026-08-31`
- Starting point: `41ad4dea9a42812fe9e0d2a2c41e78ff0663f785`
- Scope: rotate the converged dynamic correlation cube from the last QP
  compute basis to the DFT output basis, then remove the three fail-closed
  guards and their two documentation restrictions atomically.
- Implementation: `e7505942` performs the single row-scanned `U C U†`
  transform at SC finalize and removes the guards/docs; `b82bf168` rebuilds
  the at-DFT diagonal from that cube.  `f78fef5b` and `847f441a` discharge
  the two newly reachable head-off metallic SC seams (fixed-N MP1 seed and
  an exact-zero scalar-head record).  The campaign's five committed planner
  prerequisites are included through `0413526e`.
- CPU evidence: the exact campaign environment ran the rotation,
  basis-contract, config, metallic-SC, zero-head, delivered-window, and
  layering suites: **171 passed, 1 skipped** in 72.05 s.  A separate
  four-logical-device CPU
  execution of
  `test_full_sigma_cube_rotation_stays_two_axis_sharded_on_p4` passed in
  1.71 s; it matched the explicit host `U C U†` transform and retained
  `P(None,None,'x','y')`.
- CUDA P=4 kernel evidence: JID 57754440, step 71, nid001033, source
  `b82bf168`, four ranks/four devices.  The row-scanned output kept
  `P(None,None,'x','y')` and matched the corresponding host tiles with
  max absolute error `9.1551e-16`.  Evidence:
  `runs/Na/02_soc48b_qsgw_mpa/60_sc_delivered_20260831/codex_qsgwdyn_probe/rotation_p4.log`.
- P=4 Na reachability: step 83 reached the fresh SC MPA body fit in 195 s
  and exposed the zero-head fit defect; after its repair, step 87 passed
  that seam and refused the deliberately narrow `0..5 eV` diagnostic at
  the low-valence coverage gate in 223 s.  Step 90 used `-5..+5 eV`,
  reached the delivered planner after a fresh fit, and correctly refused
  its product-window cost budget in 205 s.  All are JID 57754440, P=4.

## End-to-end verdict

- **Blocked before the first SC Sigma execution, not PASS.**  The supplied
  `-10..+10 eV` arm on the integrated planner was measured at P=4 in JID
  57754440 step 91 (44 s, source equivalent through `0413526e`).  It
  refused `omega>=E_F cond:resonant[p1/3]` with achieved residual
  `0.00510025` and amplification p99 `7183.46`.  This is the required
  product-window refusal; the owner ruling forbids a direct-pair or relaxed
  escape hatch.
- Consequently zero SC iterations completed.  There are no achieved
  per-iteration Delta-E values, planner times, final artifact-checker PASS,
  or converged rotated-C versus fixed-point diagonal delta to report.  In
  particular, the requested U-consistency comparison must not be inferred
  from the `9.1551e-16` kernel oracle: that number validates the transform,
  not the converged Na bases.
- The independent servable `0..5 eV` fixed-point dependency control did
  complete on P=4 (JID 57754440 step 89): plan-cache miss, 6 windows,
  154 `(window,tau)` pairs, zero direct terms, Sigma 51.16 s, total 69.56 s,
  and a written `sigma_mnk.h5`.  It does not satisfy this task's SC or
  `-10..+10 eV` validation and supplies no per-SC-iteration planner timing.
- Under the repository's default strict band-boundary policy the same arm
  refuses before planning: its named 48-band chi edge has no spare WFN band
  with which to certify closure, and its 24-band Sigma edge splits a
  degenerate multiplet (nearby legal edges are 20, 26, and 46).  The legacy
  arm wrapper uses `LORRAX_BAND_DEGENERACY=snap`; the machine contract says
  never to use that diagnostic override to make a gate pass.
