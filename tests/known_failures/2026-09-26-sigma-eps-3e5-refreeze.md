# GN-PPM frozen references re-cut at the Σ quadrature default 3e-5

**Date:** 2026-09-26
**Branch:** `perf/sigma-eps-3e5-release-2026-09-26`, on main `ae320a2c`.
**Owner approval (2026-09-26):** the production `sigma_quadrature_eps` default is 3e-5; test decks and the `relaxed` tier have a 5e-4 floor. The frozen references are re-cut once, at 3e-5, together with the ω split of the non-crossing Σ branches (RC2).
**Cut tree:** `8aeb755fc`. That is the branch before this commit: RC2, the one-ε-key commit, the defaults and the deck values. A100-40GB, R44 evidence bundle `ae320a2c` (no C++ change).
**Evidence:** sandbox `runs/DEV/557_epsland_20260926/`.

## Procedure

Each reference is the output of the gate cell's own run on the cut tree:
- **Core A:** `test_a_zeta_cohsex_gnppm_match_references`, P4. It writes `gnppm_{eqp0,eqp1,sigma}.dat` and `gnppm_sigma.h5`, plus the schema-v5 box rules. The v4 rules are deleted because v5 no longer serves them.
- **Bispinor:** `test_bispinor_gnppm_matches_reference`, P4.
- **Scalar:** `test_gnppm_matches_reference`, one process.

A `RE-CUT` provenance block is added under each `sigma_diag` header. Stamps A and B are rewritten with `python -m tests.core.fixtures.stamp_references A B`.

Before freezing, each fixture was built a second time with main's rule partition at the same ε. That tree is main plus the ε, default and deck commits, without RC2.

## Moves and partition spread

All values are in meV. "Old" is the previous frozen reference. For the bispinor, old eqp comes from main's R44 run, which reproduces the old reference exactly.

| reference | old → new, max / median | partition spread at 3e-5, max | test atol |
|---|---|---|---|
| core A GN-PPM (25 rows) | eqp0 4.753 / 0.450; eqp1 6.035 / 0.997 | eqp0 0.045; eqp1 0.035 | 0.5 |
| bispinor GN-PPM (270 rows) | sigC 0.067 / 0.014; eqp0 0.066 / 0.014; eqp1 0.082 / 0.011 | sigC 0.013; eqp0 0.012; eqp1 0.014 | 0.01 |
| scalar GN-PPM (414 rows) | sigC 0.061 / 0.014; sigXC (= one-shot eqp0) 0.061 / 0.015 | sigC 0.029; eqp0 0.029; eqp1 0.023 | 0.01 |

- **Core A.** The old reference was cut at ε 1e-3. Its 4.75 meV move is that rule set's own quadrature error.
- **Bispinor and scalar.** Their pins are 1e-5 eV, the cross-machine freeze ruling of 2026-08-07. At 3e-5 two rule partitions differ by more than that, so these two pins pin rule identity, and any future rule-set change breaks them again. The tolerances are unchanged. That question is with the owner.
- **Not re-cut.**
  - `test_fixed_point_frozen_qp_rotations` is a strict xfail in the executable ledger, with an unrelated 25.7 meV cause.
  - The core B MPA references (`mpa.in`, `mpa_sc1.in` at the 5e-4 floor) belong to the owner's strict xfail `test_b_mpa_one_update_matches_references`. Only B's shipped rule cache is replaced.
