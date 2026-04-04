COHSEX debug regression fixture
===============================

This directory contains a small end-to-end COHSEX-JAX regression case.

Input and required data:
- `cohsex_test.in`
- `WFNsmall.h5`
- `dipole.h5`
- `kin_ion.h5`
- `centroids_frac_60.txt`
- `k0_diag.txt`
- pseudopotentials (`Mo_ONCV_PBE_FR-1.0.upf`, `S_ONCV_PBE_FR-1.1.upf`)

Reference output:
- `eqp_ref.dat`

The pytest regression test executes:

```bash
python -m gw.gw_jax -i cohsex_test.in
```

in this directory and compares generated `eqp_test.dat` against `eqp_ref.dat`.
