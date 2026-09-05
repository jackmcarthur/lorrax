# LORRAX test suite

## Core, core-extended, and full

The default suite is the two-minute core tier. It combines the cached tiny
fixture family under `tests/core/` with an exact roster of established service
and runtime contract cells in `tests/core/manifest.py`.

```bash
lx test                    # developer pre-push: default core, P=4 node
lx test --core-extended    # additional tiny-system and provider coverage
lx test --full             # nightly: the complete non-extra suite
```

`--census` and `-m census` remain aliases for `--full` for old automation.
The full tier is the owner of `tests/KNOWN_FAILURES.md` accounting. A named
path, `-m`, `-k`, or a service-selection option is an explicit selection and
therefore stands the default narrowing down.

The core roster is deliberately small and exact. A stale node ID is a usage
error, not a silently smaller green run. The default contains:

- dense-linalg routing, factorization, padding, reshards, and a real P4
  program;
- SlabIO host/emulated and hostile-extent cells;
- ISDF, centroid, zeta-loader, symmetry, Coulomb, and WFN-loader checks on
  the tiny fixture family;
- A-system COHSEX and GN-PPM, B-system MPA plus exactly one SC update;
- a three-point htransform pin and a tiny finite-Q TDA/exciton solve;
- strict deck/refusal, source-closure, compile-agreement, lint, and native
  provider attestations.

The old real-deck regressions and the per-defect assertion zoo remain in the
full tier. Redundant parameter sizes and retired-route duplicates are not a
reason to grow core; one hostile size and one actionable refusal are enough.

## Cached fixtures

`tests/core/fixtures/` contains four tiny mean-field members forming two
physics fixtures: the A family for ISDF/static/GN-PPM/excited-state coverage,
and single-atom B for MPA/QSGW. Every retained input and reference is
SHA-256-authenticated by `PROVENANCE.json`.

The normal suite never runs Quantum ESPRESSO. Fixture maintenance is a
separate, un-timed action:

```bash
lx test --build-fixtures
```

That command builds or hits the hash-addressed QE cache, verifies every
committed derived-reference stamp, and exits before test collection.

## Process geometry

The cached driver family supports four independent pytest ranks, which is
the landing command for the core fixture layer:

```bash
lx run -N 1 -G 4 -n 4 python3 -m pytest tests/core -p no:cacheprovider -q --tb=short --basetemp=<scratch>/core
```

`lx test` uses one task and xdist workers; it is a different launch shape.
Under plain P4 pytest, `core/rank_session.py` stages each driver directory
once on rank zero and broadcasts its absolute path. It waits for all child
exit codes and shares rank-zero stdout before the next driver starts. The
children retain their Slurm rank environment and initialize MPI/JAX themselves.
The pytest parents use bounded socket rendezvous, without initializing MPI.
The dense-linalg child uses those four existing ranks rather than spawning
another process group. Excited-state drivers use a 2x2 mesh at P4.

Shared driver work directories are retained in `.pytest_core_runs/` in the
checkout, outside pytest's temporary-directory cleanup. An explicit basetemp
is suffixed per rank for ordinary private fixtures. This supports both default
`/tmp` basetemp and a shared scratch basetemp without collective HDF5 opens
using different filenames, or one rank deleting another rank's work.
Delete `.pytest_core_runs/` only after all associated pytest jobs have ended.
