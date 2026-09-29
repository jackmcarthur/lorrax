# LORRAX tests

- `hsuite/`: the test suite, the production drivers run end to end on a tiny
  magnetic H2+ fixture at P4 ([README](hsuite/README.md)).
- `test_layering.py`, `test_crossfile_requests.py`, `test_env_registry.py`,
  `test_env_grammar.py`, `test_fft_shardmap_context.py`: static AST suites.
- `bench/`: performance tools (benchmark and profiling drivers), not tests.

How to run the suite and regenerate its references:
[Contributing](../docs/contributing.md#the-test-suite).
