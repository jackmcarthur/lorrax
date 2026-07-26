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

Staging this fixture for a cluster gate — READ THIS
---------------------------------------------------

**Copy, never symlink.**

```bash
mkdir -p "$RUNDIR"
cp -rL tests/regression/cohsex_debug/. "$RUNDIR"/   # -L: deref, never link
chmod -R u+w "$RUNDIR"                              # the copies must be writable
```

`ln -sf .../cohsex_debug/* $RUNDIR/` looks equivalent and is not: the driver
writes `sigma_mnk.h5`, `eqp*.dat` and `tmp/` into its run dir, and with
symlinks in place those writes land **through the link, on the fixture**. That
destroyed this directory's `sigma_mnk.h5` on 2026-07-25 with no error and no
test failure — it was caught by eye.

Two guards now exist:

* the files here are kept **read-only** (`a-w`); `tests/conftest.py`
  re-applies that at every pytest session start via
  `harness.protect_fixtures()`, so a write through a stray symlink fails
  loudly with `EACCES` instead of corrupting the fixture;
* `harness.copy_fixture()` restores owner-write on the run-dir copy, so
  in-tree gates that mutate their input file still work.

To edit a fixture on purpose: `chmod u+w <file>`, edit, commit. The next
pytest session sets it back to read-only.
