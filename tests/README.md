# LORRAX test suite

The suite is the production drivers run end to end on one tiny magnetic
system: [`hsuite/`](hsuite/README.md). Its chain is kmeans → kin_ion →
dipole → GN-PPM one-shot → shared-pole QSGW (2 maps) → BSE → htransform →
exciton bands, at P4 on one node. Each stage is checked against the stored
outputs in `hsuite/reference/`.

```bash
lx run -N 1 -G 4 -n 4 -- python3 -m pytest tests/hsuite -q -p no:cacheprovider   # P4
lx test                                   # the same cell at P1, plus the AST suites
```

Beside it are the five static AST suites that `gate0` and
`tools/release_check.sh` run as scripts: `test_layering.py`,
`test_crossfile_requests.py`, `test_env_registry.py`, `test_env_grammar.py`
and `test_fft_shardmap_context.py`. They also collect under pytest.

`bench/` holds standalone benchmark drivers. They are not tests, and pytest
does not collect them.

A driver change is judged by the chain's outputs. If a change moves the
stored numbers on purpose, regenerate them and review the diff:

```bash
lx run -N 1 -G 4 -n 4 -- python3 -m tests.hsuite.chain --out DIR --regenerate
```
