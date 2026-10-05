# LORRAX tests

- `hsuite/`: the test suite, the production drivers run end to end on a tiny
  magnetic H2⁻ fixture at P4 ([README](hsuite/README.md)).
- `test_layering.py`, `test_crossfile_requests.py`, `test_env_registry.py`,
  `test_env_grammar.py`, `test_fft_shardmap_context.py`: static AST suites.
- `test_sc_checkpoint_resume.py`: SC continuation on toy inputs (CPU): the
  Anderson trajectory across a stop, a refused held Σ rule, the band carrier.
- `test_trs_qe_decides.py`: the 2c TRS verdict on toy loaders (CPU): QE's
  t_rev rows and SCF magnetization decide, the wavefunction guard refuses an
  inconsistent WFN at m = 0 and turns TRS off at a small moment, the SCF
  magnetization is found by density, and a WFN_qp.h5 is discovered from its
  source WFN.
- `test_orbital_modern_route.py`: htransform's orbital totals and velocity
  refusals on toy inputs (CPU).
- `test_hl_head_plasma_count.py`: the HL-PPM head ω_p² counts electrons
  (CPU).
- `test_minibz_equal_shares.py`: the shared mini-BZ head average gives every
  rank one share shape (CPU, three simulated ranks).
- `test_stream_passes_cpu.py`: the scanned row-pass loop on a host mesh
  under x64 (CPU): two windows add the bytes of one.
- `test_sigma_kconv_scratch.py`: mode 7's split-arm scratch follows the
  handler's rule and is priced in the Σ pass and the τ window checks (CPU).
- `test_head_resolver_hubbard.py`: the one-shot static head carries the
  deck's DFT+U input and refuses a dipole without i[r, V_U] (CPU).
- `test_lockstep_guards.py`: run-condition guards on toy inputs (CPU):
  compiled-size checks agree over ranks, `step_up` never snaps past its top,
  the sector CT eigh operand is Hermitian, the −q mirror's inversion is an
  authorized unitary row.
- `test_streamed_bank_capacity.py`: the per-rank streamed tier's capacity probe
  on toy stores (CPU): page-aligned records, reserved bytes promised once, the
  quota room under the hard limit, a late W-bank field the disk refuses held
  in host memory.
- `test_shared_pole_guards.py`: shared-pole guards on toy inputs (CPU): the
  relaxed tier's sector face batch has no Ritz carrier, a bank of another
  bare-V digest is rebuilt, the pole-budget cut never splits a multiplet.
- Benchmarks and backend checks for a standalone service live in that service's
  `services/<svc>/bench/`.

How to run the suite and regenerate its references:
[Contributing](../docs/contributing.md#the-test-suite).
