# The cohsex quickstart was red at the base for two fixture reasons, and the 337 eV `VH` move is the Hartree switch, not the regeneration

**Date:** 2026-09-01
**Branch:** `lane/bisp-e-tests-fixtures-2026-09-01`, cut from
`integ/bispinor-static-cleanup-2026-09-01` = `origin/main@8b6e3cc7`.
**Evidence:** `runs/DEV/297_bisp_e_tests_fixtures_20260901/` in the sandbox,
Perlmutter JID 57845011, one A100-SXM4-40GB per step.

## What was red at the base, and is not any more

Two cells, both for reasons that had nothing to do with physics.

**1. `tests/test_head_correction.py` — five cells, `AttributeError`.**
`HeadResolver.__init__` resolves the four-current representation from
`config.bispinor` / `config.bispinor_gw` (`src/gw/head_correction.py:1713-1716`);
the suite's two hand-built `SimpleNamespace` configs predate that and carry
neither key. Fixed by giving both namespaces `bispinor=False,
bispinor_gw="bare_transverse"` — the scalar defaults, because this suite is
about the head POLICY and not the bispinor carrier. `lx test --workers 1
tests/test_head_correction.py`: **12 passed** (step
`lx-Xg4-185245-1626511-6651`), against 7 passed / 5 failed at the base.

**2. `test_gw_jax_regression.py::test_gw_jax_matches_reference[cohsex]` — the
run never started.** The shipped `dipole.h5` and `kin_ion.h5` both predate
their provenance gates:

* under the deck's default `head_correction = full`, `GATE
  dft_head_dipole_provenance` refuses `dipole.h5`;
* under `head_correction = no_local_fields`, `file_io/kin_ion.py` refuses
  `kin_ion.h5` for having no bispinor-representation stamp.

Both regenerate from the deck in seconds with the current producers, because
`cohsex_debug` is the one regression deck that ships its own pseudopotentials.
Regenerated, committed, and the reference re-frozen from that run.

## The `VH` column moved 337 eV. It is the Hartree provider, and NOT the fixture regeneration

This is the part worth reading, because "the fixture was regenerated and the
numbers moved" is the wrong summary.

Measured against the 2026-08-15 `eqp_ref.dat`, over all 120 rows, with
`tests/harness.parse_eqp_rows` — the gate's own parser:

| column | frozen ref → HEAD | the `vnl_velocity_sign` arm alone | remainder |
|---|---|---|---|
| `sigSX`  | 0.107719 eV | 0.082991 eV (77.0 %) | 0.024728 eV |
| `sigCOH` | 0.053860 eV | 0.041496 eV (77.0 %) | 0.012364 eV |
| `sigTOT` | 0.053860 eV | 0.041496 eV (77.0 %) | 0.012364 eV |
| `VH`     | **337.142795 eV** | **0.000000 eV, 0/120 rows** | 337.142795 eV |

The middle column is a real single-variable A/B on this deck: the same
`kin_ion.h5`, the same WFN, the same head policy, `dipole.h5` rebuilt on the
other arm of `vnl_velocity_sign` and the deck moved with it (the provenance
gate correctly refuses a `-1` file under a `+1` deck, which is how the arm was
found). `VH` is **bit-identical** across that flip, so `dipole.h5` contributes
exactly nothing to it.

`VH` is `results.sig_h`, and since `62f1fa22` (2026-08-29, "make live G-space
the only Hartree path") it is built live from the run WFN —
`sigma_dispatch._compute_live_hartree` → `kin_ion_io.compute_hartree_matrix(wfn,
sym, meta, ...)`. It never reads `kin_ion.h5`. At the 2026-08-15 freeze the
shipped `kin_ion.h5` offered neither a `v_hartree` dataset nor `has_hartree`,
so `resolve_hartree_source("auto")` returned `isdf` and the frozen `VH` column
is the **ISDF V_q[0] quadrature on 60 centroids**. The move is that switch and
nothing else.

**The old pairing was internally inconsistent, and that is checkable.** LORRAX
has no `vxc.dat`: it derives the XC subtraction as `E_dft − H0` with
`H0 = kin_ion + V_H`, so the implied `Vxc` can be read back out of the written
tables. With the regenerated artifacts it is negative on **0 positive rows of
120** (min −6.273, max −1.539, mean −3.587 eV). With the 2026-08-15 pairing —
shipped `kin_ion` diagonal −402.119 eV at k=0 n=0, frozen `VH` 254.639 eV,
`Eo` −59.162 eV — it is **+88.32 eV**, which no XC potential can be. The
shipped `kin_ion.h5` diagonal also sits 232.95 eV (printed window; 242.94 eV
over all 31 shared bands) above what the current producer writes for the same
deck and the same WFN fingerprint, so that file was never the same operator.

`kin_ion.h5` does not appear in any column the gate compares — `Eo` comes
from the WFN — but it does set `H0`, so it moves `eqp0_test.dat` /
`eqp1_test.dat`, which nothing freezes.

## What stays open

* **`cohsex_test_minimax_selfcheck.in` is still ungated and its reference is
  still stale — deliberately not re-frozen.** The deck does **not** refuse
  with the regenerated artifacts (it ran to completion, step
  `lx-Xg1-191041-1841189-2344`), so the condition for re-freezing it was not met.
  Its `eqp_minimax_selfcheck.dat` is stale three independent ways: an obsolete
  writer header (`# COHSEX output: Sigma_SX ...`), 270 full-BZ rows against
  the current 120-row file wedge, and the ISDF-era `VH` family (254.863 /
  337.683 eV). Its own deck header records the ~1226 meV MAE as *evidence that
  the file is stale* and asks the owner to wire the deck to a gate or delete
  it. Re-freezing would have answered that question silently and destroyed the
  measurement the header cites. Nothing in `tests/` collects it, so nothing is
  red.
* **`docs/quickstart.md` §2's Perlmutter form cannot be run literally in-tree.**
  It says `cd tests/regression/cohsex_debug && lx run python3 -u -m gw.gw_jax
  -i cohsex_test.in`, but `gw_jax` resolves `tmp/`, `qp_wfn_rotations.h5` and
  the eqp writers against the DECK directory, and
  `harness.protect_fixtures()` sets that directory `a-w` at every pytest
  session start. On any tree where pytest has run, the documented command
  fails on `EACCES`. `tests/test_quickstart_verbatim.py` therefore runs the
  same command shape against a copy and says so in its docstring; the docs fix
  is not this lane's.
