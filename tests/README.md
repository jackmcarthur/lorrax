# LORRAX tests

`conftest.py` seals the first-party source closure
(`runtime.source_closure.ensure_source_closure`) once per test process, before
any test imports a service, as the drivers do at start-up. Without it a test
that imports `gw.*` first takes `distrib_la` and its peers from the
interpreter's install (the sealed release under `lx test`), and a later
`file_io` import in the same pytest worker refuses with `SourceClosureError`.
A new test needs no bootstrap import of its own.

- `hsuite/`: the test suite, the production drivers run end to end on a tiny
  magnetic H2⁻ fixture at P4 ([README](hsuite/README.md)).
- `test_layering.py`, `test_crossfile_requests.py`, `test_env_registry.py`,
  `test_env_grammar.py`, `test_fft_shardmap_context.py`: static AST suites.
- `test_sc_checkpoint_resume.py`: SC continuation on toy inputs (CPU): the
  Anderson trajectory across a stop, a refused held Σ rule, the band carrier,
  the checkpoint cube digest split over ranks (the same at any P).
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
- `tdagw/test_prop_*.py`: the TD-aGW propagator (CPU). `integrators`: the
  stepper of `solvers.time_integrators` against SciPy (DP54 takes SciPy RK45's
  steps; measured orders; Magnus unitarity; Strang order), each with a red
  twin. `tdgw_parity`: on a frozen TDGW 0.3.0 fixture (`fixtures/syn2v2c`,
  built by `fixtures/make_tdgw_fixture.py`) the driver's stages on a 2x2 host
  mesh take TDGW's steps and reproduce every measurement to 1e-12; red twins
  drop the intraband term and flip the kernel sign; four devices equal one.
  `deck`: every parser refusal names its gate. `isdf_adapter`: the ISDF
  adapter's dSigma equals 1/2 K (rho - rho_eq) for a synthetic TDMODEL model
  with a permuted k order and padded k; red twins drop the Ry -> Ha halving
  and pass rho for Delta rho.
- `tdagw/test_collision_cpu.py`: the one-sided collision branch operator,
  equilibrium subtraction, Hermitian update and held affine flow against
  independent NumPy/SciPy references.
- `tdagw/test_prop_collision_cpu.py` and its `_collision_case.py` child:
  collision attachment through the production propagator on a 2x2 CPU mesh;
  fixed point, integrator order, refreshes, carry/history rebasing, compile
  reuse and k-locality against exact piecewise affine solutions.
- `test_tdagw_collision_fullrho.py` and `test_tdagw_collision_instbasis.py`:
  focused script gates on a 2x2 mesh for the full density-matrix internal
  line, dense branch-table references, vertex pairing, basis covariance,
  closed-window trace checks, instantaneous-basis fixed point and head.
  Their pytest entries skip; run the scripts explicitly with four host
  devices for the CPU leg.
- `tdagw/_collision_516_case.py`: the real CrI3 GN-PPM collision diagnostic,
  including branch tables, equilibrium drift, driven refreshes, frozen/active
  splits and screening snapshots; run explicitly at P4 on compute nodes.
- `tdagw/test_validation_*_cpu.py`: the TD-aGW validation ladder
  ([theory](../docs/theory/td_agw_propagation.md#validation)), CPU on a 2x2
  host mesh through the driver's deck -> stages -> `tdgw.h5` route; each rung
  has red twins, tolerances are fixed a priori and every measured value is a
  junit testsuite property (the `record` fixture). Systems and references are
  in `tdagw/_validation_systems.py`, the response algebra in
  `tools/td_response.py`. `rabi`: a two-level system under cw and Gaussian
  drives, with and without relaxation, against the exact rotating-frame
  solution and DOP853; every integrator at its observed order; the swapped
  dipole and the 0.36 ppm Ry/Ha slip fail. `bse`: on a 3v3c window with an
  ISDF-form Hartree + SEX kernel, P(t), per-state amplitudes and frequencies
  and Im χ± equal the full BSE of the kernel; the TDA kernel fails.
  `conservation`: trace, Hermiticity and purity, the energy functional with
  its ½ and the work theorem, equilibrium stationarity, every integrator.
  `sumrules`: m1 = o.L.u, m3 and j(0+) after an instantaneous kick (IP, full,
  TDA); the TDA misses the full sum rule. `helicity`: σ± selection on a
  C3-symmetric Haldane toy down to the pulse's counter-rotating leak; the
  reversed convention and C3-broken dipoles fail. `magnetooptics`: χ_xy, ε_xy,
  σ_xy and sheet and film Kerr/Faraday rotation from TD equal the BSE;
  Onsager; no dc Hall term in the interband length gauge; the film converges
  to the sheet for an anisotropic tensor.
- `test_sigma_kconv_scratch.py`: mode 7's split-arm scratch follows the
  handler's rule and is priced in the Σ pass and the τ window checks (CPU).
- `test_qe_reported_inputs.py`: the V_NL spin-orbit mode is QE's
  `<spinorbit>` or a named refusal, and kmeans opens the deck's `wfn_file`
  (CPU).
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
  bare-V digest is rebuilt, the pole-budget cut never splits a multiplet, a
  local CT round over budget is handed back for the face rerun.
- `test_centroid_fixed_tile.py`: the kmeans candidate Gram's k batches and
  tiles and the feature metric's chunks do not read the budget (CPU, toy
  faces).
- `test_kmeans_rank_header.py`: the centroid header names the pool rank and
  its stop, warns when the point budget stopped the selection before the
  floor, and reproduces the rank law's documented Fe point (CPU).
- `test_compile_stability_cpu.py`: compile-stability guards on toy inputs
  (CPU): checked solves leave no host callback in their programs, a second
  process reuses a checked program from the persistent cache, one stack from
  two call sites compiles once, partner directions keep one carrier, face
  rounds share one width. The audit-row tests are strict xfails until their
  fix lands.
- `test_bse_kgrid_dipoles_cpu.py`: the `bse_k_grid` dipoles on a toy
  tight-binding slab (CPU, 2x2 host mesh): on-grid identities, off-grid
  dipoles against exact ones in the densified states' gauge, a planted coarse
  gauge invisible and a velocity-only gauge seen, DFT denominators under a
  scissor, and the window-dipole refusals.
- `test_bse_kgrid_dipoles_rfix_cpu.py`: `--eqp` (`qp_states`) dipoles on the same
  toy with band-dependent QP shifts: the exact position of the fine QP states
  to the interpolation floor (0.4 %) against the toy's full eigenbasis, the
  pre-fix Rayleigh-quotient `v/ΔE` red twin 12x off; the DFT path bitwise
  `v/ΔE`; `r_window` Hermitian with `d = i r[c, v]`; the pair rule equal to
  `tdagw.td_system.position_from_velocity`'s; the rotation record.
- `test_bse_helicity_cpu.py`: per-state in-plane helicity on a C3-symmetric
  toy (s, p_x, p_y with an orbital Zeeman term; CPU, 2x2 host mesh): the Γ
  transitions into p± take the field-defined σ± labels that ⟨L_z⟩ certifies;
  the swapped and the conjugated conventions fail that check; a global phase,
  a band gauge and the zone size leave the per-cell table unchanged; the
  native and `bse_k_grid` routes give one table; `bse_jax` reads
  `--helicity-table` on the Lanczos route.
- `test_bse_qp_rotation_cpu.py`: the QP-rotated BSE basis on toy inputs
  (CPU, 2x2 host mesh): U = I gives the `--eqp` dipole bit for bit and a
  permuted U does not, a cross-gap-free rotation rotates the position matrix
  exactly, the dipole equals the interband-commutator velocity over the QP gap,
  the star gauge refuses an unconjugated antiunitary row, the QP window guard
  refuses a cut edge and a window outside the block; the QP route of
  `window_dipoles` equals the DFT slice at U = I and refuses a `dipole.h5` on
  another k grid, with another WFN's stamp or none, or with `skip_vnl`.
- `test_w_pad_nyquist_cpu.py`: the W densifier's R zero-pad (CPU): an even
  coarse axis splits its Nyquist coefficient, so coarse tiles stay exact and
  every fine tile is Hermitian; the single-sided red twin is not.
- Benchmarks and backend checks for a standalone service live in that service's
  `services/<svc>/bench/`.

How to run the suite and regenerate its references:
[Contributing](../docs/contributing.md#the-test-suite).
