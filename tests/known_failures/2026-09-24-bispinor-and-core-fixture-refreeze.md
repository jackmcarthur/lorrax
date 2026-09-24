# Bispinor GN-PPM and core fixture A/B references re-frozen at the landing tree; every move explained

**Date:** 2026-09-24
**Branch:** `fix/physics-refs-2026-09-24`, cut from `land/overnight-2026-09-24@c52b2c42`.
**Evidence:** sandbox `runs/MoS2/46_physics_refs_bispinor_20260924/` (legs L1–L4), `runs/runtime/overnight_20260924/logs/core_p4_land_c52b2c42.rank0.log`, and sandbox claims 635 and 2696.
**Supersedes:** the "NOT re-frozen" disposition of
[2026-09-02](2026-09-02-bispinor-gnppm-fixture-runs-again-reference-not-refrozen.md).

## Bispinor GN-PPM (`tests/regression/bispinor_debug/sigma_diag_bispinor_ref.dat`)

The 2026-08-09 reference was stale by up to 1.434381 eV. Before re-freezing, every component of the drift was attributed to a named commit. Values are mean / max |Δ| in eV over the 270 rows:

| epoch | column | cause | move | measured by |
|---|---|---|---|---|
| 08-09 → 09-02 | direct | `kin_ion.h5` regeneration, forced by f80a5f70's provenance guard. V_H moves from the ISDF fallback to the stored exact V_H; Σ does not move. | +0.274625 / 1.429902 | claim 635 (JIDs 57882521, 57886833) |
| | direct | 07451900: WFN reciprocal scale applied once | +0.003042 / 0.004631 | claim 635 |
| | sigC | 4534dc79: the Dyson prefactor takes the physical nspinor (2) instead of the lift width (4). The lift width halved every χ₀ block. | +0.512756 / 0.803032 | claim 635 |
| | sigC | 27f1e5c9: orbit-safe GN pole tail | −0.001734 / 0.013729 | claim 635 |
| | sigX | 8f46b0de: Coulomb-gauge TT metric sign | +0.004243 / 0.008389 | claim 635 |
| 09-02 | sigC | 0140d997: certified box rules (837ed531 → 8576f9f9) | −0.001282 / 0.004073 | claim 635 |
| 09-02 → main dd8ee0b5 | sigC | not bisected; candidates 7571f402 (literal eta, owner ruling 09-02) and 89eaa9a3 (box-rule builder without a time budget), both on main | in-grid ≤ 0.002150, out ≤ 0.005884 | L2 vs claim-635 probe 8576f9f9; L3 |
| main → c52b2c42 | sigC | d7f556fc (owner rule 2026-09-22): a state off the [−4, +4] eV grid takes Σ(ω=0) instead of the endpoint clamp | out-of-grid only: −0.509273 / 0.860629; in-grid ≤ 1.5e-5 | L1 vs L2; L4 cross-assembly |

Controls:
- 835d9f3f (bare TT Γ average) changes nothing on this deck: kconv 61c8d018 and the landing tip agree to 0.000000.
- The H_T half of Hdir is 2e-35 eV.
- sigX and the direct column are bit-identical from 09-02 to the new cut.

The new reference is the landing-tree output (P4, JID 58826502, step `lx-Xg4-090722-712967-8498`) plus a provenance header. The strict-xfail row for `test_bispinor_gnppm_matches_reference` is removed.

## Core fixture A (`test_a_zeta_cohsex_gnppm_match_references`)

Main and the landing tree fail this test for different reasons:
- On main: 3/25 in-grid cells, ≤ 0.96 meV.
- On the landing tree: 13/25 cells, up to 0.887 eV.

New refs `gnppm_eqp0/eqp1/sigma.dat/.h5` come from the landing run (A-pieu41dy, JID 58826502, step `lx-Xg4-084936-627447-3573`). The box-rule cache is replaced by the schema-v4 rules that run wrote; the seven v3 rules were no longer served. Grid [−8, +8] eV about midgap −0.4009 eV:
- **Out-of-grid (10 cells, up to 0.887 eV):** d7f556fc, Σ(ω=0) in place of the endpoint clamp.
- **Four in-grid states within dE = 0.5 eV of the +8 eV edge (eqp1 only, −0.10..−0.18 eV):** 5335d487, whose Z probes are clipped onto the grid with the true one-sided spacing. It fixed the Z = −0.58 defect that d7f556fc exposed. Their eqp0 is unchanged to ≤ 0.97 meV.
- **Remaining in-grid cells (≤ 0.97 meV):** 89eaa9a3, which removed the box-rule time budget (the rule-cache schema migration is logged in `gnppm.out`). This is attributed by mechanism and by the green 09-05/09-09 and red 09-13 bounding runs, not by a single-commit run.

## Core fixture B (`test_b_retained_escape_grows_grid_in_the_same_map`)

- **Main** refuses with `GATE sc_fixed_quadrature_map` (b332acb5). The landing tree fixes that through 9cf542a8 and 05362d2c.
- **Landing** prints `protected=1-3 in_range=1-3`, because d7f556fc now protects every QP-window band. The assertion at `tests/core/test_driver_references.py:163` is updated. The grid-growth and receipt asserts after it are unchanged.

## Stamps A/B (`test_fixture_bytes_match_the_portable_stamp`)

This is a mechanical restamp with `python -m tests.core.fixtures.stamp_references A B`:
- **A:** the `gnppm.in` bytes (89eaa9a3 retired a deck key), `excited_state_ref.json` (01465925, the scalar-singlet BSE weight) and the references above.
- **B:** the `mpa.in` / `mpa_sc1.in` bytes (89eaa9a3; the d4214ace/a5a35701 accelerator renames).

`additional_reference_source_commits` names both regenerations. The `mpa_sc1_*` family is not regenerated: `test_b_mpa_one_update_matches_references` stays the owner's strict xfail.
