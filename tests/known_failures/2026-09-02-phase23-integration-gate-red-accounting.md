# Integration phase-23 union: all nine red test nodes accounted for

**Date:** 2026-09-02  
**Evidence:** job 57850966, step `lx-Xg1-013309-1149890-7473`,
`runs/DEV/308_integ_phase23_gate_20260902/logs/pytest_union.log`  
**Pin:** integration tree ending at `34228021042abbe871f08d0302056fa02040fe59`  
**Summary:** 3 failed, 6 errors, 416 passed, 35 skipped, 2 deselected.

This is accounting, not a repair. No assertion tolerance, fixture, or frozen
reference changed.

| Red test node | Observed disposition in the union log |
|---|---|
| `test_qsgw_parallel_transport_head.py::test_uniform_gauge_fingerprint_is_contact_capability_only` | Stale test call: `build_vnl_setup(..., soc=False)` raises `TypeError` because `soc` is no longer in the source signature. |
| `test_dft_gauge_vertices.py::test_icl_kminusq_jet_reuses_uniform_current_and_contact_exactly` | Stale exact-equality assertion: 30/48 complex entries differ only at roundoff; maximum absolute difference `9.81307787e-18`. This note does not replace exact equality with a tolerance. |
| `test_gw_jax_regression.py::test_gnppm_matches_reference` | Frozen scalar GN-PPM comparison fails: 882/2484 fields differ at `atol=1e-5`; maximum absolute difference `21.729442` eV. The mechanism is not isolated, so the reference is not re-frozen. |
| `test_gw_jax_regression.py::test_si_production_matches_frozen_reference` | Setup refuses `BandWindowDegeneracyError`: zeta left window `[0,60)` cuts a multiplet at band 60, k=3, with 0.000 meV gap against the 1.000 meV rule. |
| `test_gw_jax_regression.py::test_si_production_matches_berkeleygw` | Same session setup and same `[0,60)` `BandWindowDegeneracyError`; this is a second red node, not a second mechanism. |
| `test_gw_jax_regression.py::test_si_fast_matches_frozen_reference` | Setup instead refuses `GATE dft_head_dipole_provenance`: `head_correction=full` cannot authenticate the fixture's stale `dipole.h5`. It is not a band-window error. |
| `test_gw_jax_regression.py::test_hbn_matches_frozen_reference` | Setup refuses `BandWindowDegeneracyError`: the zeta upper boundary 80 equals the WFN extent, so closure cannot be checked without a spare band. |
| `test_gw_jax_regression.py::test_hbn_mc_average_vcoul_body_moves_sigma` | Same hBN session setup and same extent-80 `BandWindowDegeneracyError`; this is a second red node, not a second mechanism. |
| `test_gw_jax_regression.py::test_bispinor_gnppm_matches_reference` | The union does not reach the previously registered 1.43 eV reference comparison. Its one-GPU setup requests distributed cuSOLVERMp `solve_lu`, which refuses the 1×1 mesh and asks for a true 2-D mesh, `auto`, or `off`. |

Therefore the exact band-window count in this artifact is four red test nodes:
two Si consumers of one `[0,60)` setup and two hBN consumers of one extent-80
setup. The fifth regression setup is the independent dipole-provenance refusal.
The previously dated bispinor reference-drift note remains valid for the route
that runs end to end; this integration union encountered the newer 1×1 route
refusal first.
