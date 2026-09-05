# LORRAX test suite

## Core, core-extended, and full

The default suite is the two-minute core tier. It combines the cached tiny
fixture family under `tests/core/` with an exact roster of established service
and runtime contract cells in `tests/core/manifest.py`.

```bash
lx test                    # developer pre-push: default core, P=4 node
lx test --core-extended    # additional tiny-system and provider coverage
lx test --full             # nightly: compact legacy/defect tier
```

`--census` and `-m census` remain aliases for `--full` for old automation.
The full tier is the owner of `tests/KNOWN_FAILURES.md` accounting. It keeps
one representative of each legacy parametrized assertion; exact core nodes,
the tiny-fixture families, and every executable registered-defect node are
protected from that compaction. A named path, `-m`, `-k`, or a
service-selection option is an explicit selection and therefore stands the
default narrowing down.

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
reason to grow core: the roster names one hostile size and one actionable
refusal by their exact parameter IDs. A base node spelling is reserved for a
test whose complete parameter family is intentionally part of core.

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

Every landing-quality GPU program is verified at P=4; P1 is only the
deterministic/timing twin. On Perlmutter, use `lx`: `lx test` reserves the
four-GPU node, mesh-marked cells run in one grouped child, and
`procs(4)` cells launch four real ranks. See `AGENT_PREAMBLE.md` for the
machine evidence contract.

Direct diagnostic invocations use the same source closure as `lx test`:

```bash
lx run -N 1 -G 1 -n 1 env PYTHONPATH=<worktree>/src:<each-service>/src \
  python3 -m pytest -q <node-id>
lx run -N 1 -G 4 -n 4 env PYTHONPATH=<worktree>/src:<each-service>/src \
  python3 <p4-program>
```

`tests/bench/` and `services/*/bench/` are standalone measurement drivers,
not pytest suites. `tests/multi_device/` holds cross-process/cross-GPU scripts
that cannot be expressed honestly as an in-process mesh cell.
