# LORRAX tests

- `hsuite/`: the test suite, the production drivers run end to end on a tiny
  magnetic H2⁻ fixture at P4 ([README](hsuite/README.md)).
- `test_layering.py`, `test_crossfile_requests.py`, `test_env_registry.py`,
  `test_env_grammar.py`, `test_fft_shardmap_context.py`: static AST suites.
- `test_sc_checkpoint_resume.py`: SC continuation on toy inputs (CPU): the
  Anderson trajectory across a stop, a refused held Σ rule, the band carrier.
- `test_trs_qe_decides.py`: the 2c TRS verdict on toy loaders (CPU): QE's
  t_rev rows and SCF magnetization decide, the wavefunction guard refuses an
  inconsistent nonmagnetic WFN, and the SCF magnetization is found by density.
- `test_orbital_modern_route.py`: htransform's orbital totals and velocity
  refusals on toy inputs (CPU).
- `test_hl_head_plasma_count.py`: the HL-PPM head ω_p² counts electrons
  (CPU).
- `test_minibz_equal_shares.py`: the shared mini-BZ head average gives every
  rank one share shape (CPU, three simulated ranks).
- Benchmarks and backend checks for a standalone service live in that service's
  `services/<svc>/bench/`.

How to run the suite and regenerate its references:
[Contributing](../docs/contributing.md#the-test-suite).
