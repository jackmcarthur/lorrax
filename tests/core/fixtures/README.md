# Core fixtures

These fixtures are intentionally smaller than the production/regression
decks.  They test that each major route still executes and satisfies a few
named numerical invariants; they are not convergence or BerkeleyGW anchors.

- `A`: scalar triclinic LiH, two atoms, a 3x3x1 k grid, seven active bands
  plus one guard band, and 21
  orbit-closed charge centroids.  Its odd extents and low symmetry own the
  ISDF, COHSEX, GN-PPM, htransform, BSE and exciton smoke references.
- `A-prime`: the two-component, magnetized, TRS-off twin of `A`, stored on
  the full k grid with 21 current-density centroids.  P1 makes its spatial
  orbit closure trivial; the non-orbit control lives on `A-cubic`.
- `A-cubic`: a scalar diamond-H2, nonsymmorphic/divisible control with a
  2x2x2 grid and eight bands.  A 24-site orbit request expands to its
  48-site closure; a separate literal 23-site file is the non-orbit control.
- `B`: one scalar closed-shell He atom in a mildly triclinic periodic box,
  Gamma only, seven active bands plus one guard band, and 13 centroids.  It
  owns the MPA one-shot and exactly-one-update QSGW references.  Periodic B
  is intentional: a fresh 0-D zeta fit currently refuses on the registered
  missing-Gflat defect, and a restart-masked fixture would be false green.

`build_fixtures.py` hashes the as-run QE inputs and the exact pseudopotential
bytes.  It builds in `.build-cache/<system>-<sha256>/` and publishes only
after the WFN conversion succeeds.  A matching committed provenance stamp
turns a later build request into a hash-verified cache hit.

Each `PROVENANCE.json` is the portable reference stamp: source commit,
measured sizes and cold walls, and the SHA-256 of every committed input and
reference.  `python -m tests.core.fixtures.stamp_references --check` is the
ordinary integrity check; omit `--check` only after deliberate regeneration.

The first build is a separate maintenance action and is not part of the
timed core suite.  On Perlmutter it is exposed as `lx test --build-fixtures`;
ordinary `lx test` only reads and authenticates committed artifacts.
