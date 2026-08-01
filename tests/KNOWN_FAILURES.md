# Known test failures — full-suite census (P1.1 pre-push)

Complete `python -m pytest tests/` census on Frontera CLX (in-container,
required-FFI defaults, host .so `build_host_MRG`), 2026-08-01, tree
`bbe6e56` + the fixes committed with this file.  A release ships LISTED
known-fails, never unknown ones: every non-passing test in the suite is
accounted for below.  Authoritative run: **job 7885154** (junit XMLs +
run dirs under `/scratch2/08271/jackmc/pytest_p11/`).  Job 7885150 was
the first attempt and is superseded — its 97 failures were dominated by
an instrument error (see "Instrument notes"), kept only as evidence.

## Verdicts by leg (job 7885154)

| leg | invocation | result |
|---|---|---|
| A2: full suite, bare, 1 device | `pytest tests/ --ignore=tests/test_ffi_linalg_contract.py` | 24 failed / 735 passed / 56 skipped / 26 deselected |
| B2: FFI linalg contract | `srun --mpi=pmi2 -n1` + `config/frontera/mpi_transport_env.sh`, `pytest tests/test_ffi_linalg_contract.py` | **0 failed** / 27 passed / 25 skipped |
| C: 4-device leg | `XLA_FLAGS=--xla_force_host_platform_device_count=4`, the 9 device-hungry files | 1 failed / 144 passed / 2 skipped |
| C2: nonsquare refusal | `...device_count=2`, `-k nonsquare` | 1 passed |
| D: extra tier | `-m extra`, 4 devices | 0 failed / 21 passed / 5 skipped |

Every leg-A2 failure below is triaged; after the fixes in this commit
the only remaining red is the ring-vma class.

## KNOWN FAILURES (ship-listed)

| tests | class | evidence | status |
|---|---|---|---|
| 10 ring-transport tests: `test_bse_dense_reference` `{w_positive_control,full_H,DV}[ring]` + `test_nontda_matvec_matches_dense_shao` + `test_nontda_solver_reproduces_dense`, `test_bse_stack_matvec::test_stack_memory_flat_in_n_trials`, `test_bse_w0_resolvent` (2), `test_bse_w_omega_chain` (2) | (b) pre-existing — the old handoff's "bse_ring_comm vma", verified present at HEAD and now precisely diagnosed | `TypeError: scan body ... carry ... {V:(x,y)} varying manual axes` at `src/bse/bse_ring_comm.py:382` (`_apply_V_ring_only` fori_loop carry `A0` unannotated; jax's error prescribes `lax.pvary` on the initial carry). junitA2_7885154. Serial + simple matvec arms PASS, so the dense-reference physics is still covered; only the ring transport arm is dark | OPEN — register row in sandbox `KNOWN_LORRAX_ISSUES.md` (bse section); fix needs an iterate-run cycle (audit every fori/scan carry in the ring path incl. the W arm) |
| `test_ffi_linalg_contract` under a BARE (no-srun) launch with the host .so loadable: silent interpreter death at import | (d) environment — MPI init without PMI2 glue | CLAIMS 30; reproduced 7885125 step1 (srun WITHOUT transport env also dies). **Upgrade this census adds: with `mpi_transport_env.sh` sourced under `srun --mpi=pmi2 -n1` the pytest form is fully GREEN (leg B2, 27 passed)** — the CLI matrix is no longer the only instrument | Not a code bug; invocation contract. Bare pytest must `--ignore` this file; the srun+transport leg covers it |

## Environment-limited (skips, each with its covering leg)

| tests | reason | coverage |
|---|---|---|
| 45 device-count skips (leg A2): `test_staged_reshard` (14), `test_staged_reshard_routes` (18), `test_charge_zeta_route` (7), `test_sharding_fit` (4), `test_collectives_distribution`, `test_centroid_distribution`, `test_sanity_gates_jax::test_check_hermitian_sharded` | need >=2/4 emulated devices | leg C (all green after the fix below); nonsquare-refusal cell needs a NON-square count → leg C2 (green) |
| `test_centroid_distribution::test_process_local_mesh_is_addressable` negative control | needs true multi-PROCESS (P>1), not emulated devices | P>1 srun leg (P1 scaling legs); `tests/multi_device/` is likewise srun-driven, never pytest-collected |
| `test_bse_kgrid` (7), `test_wfn_loader_eager[mos2]` (3), `test_R_proper_cri3` (1, extra tier) | fixtures pinned to `/pscratch/...` — Perlmutter, machine gone | OWNER: restage the MoS2 3×3 640c fixture + WFN.h5 on Frontera (or re-point); until then these self-skip |
| 23 CUDA cells in `test_ffi_linalg_contract`, `-m gpu`-dependent extras (3 cufft + 1 CUDA backend in leg D) | need a CUDA jax backend | P1 GPU leg (rtx) |
| `test_slate_cholesky_trsm_cpu` heev cells (2 skips in leg B2) | slate host heev SIGSEGV — documented bug L-2, `docs/dev/linalg_ffi.md` | pre-existing, pinned by the skip itself |
| 26 deselected (`-m extra` tier: sternheimer, head_wing_schur, aot_memory, R_proper_cri3, reshard_all_to_all, bse dense extras) | deselected by repo `addopts` default | leg D ran them: 21 passed / 5 skipped (rows above) |

## Fixed in this pass (committed with this file)

| tests | root cause | fix | validation |
|---|---|---|---|
| `test_file_io` (12) + `test_compute_all_V_q_g_flat::test_..._rejects_r_space_loader` | (c) stale builders: synthetic `zeta_q.h5` helpers never stamped `zeta_is_done`, and `ZetaLoader` now refuses partial files at open (completeness gate) | builders stamp done (complete synthetic payloads); the flag-behaviour tests pass `zeta_is_done=False` explicitly | GREEN in 7885154 leg A2 |
| `test_zq_from_psi_sm_bit_identity` (6) | (c) `_MockPsiGStore` missing `_bpd_per_bc` (added to `PsiGStore` by the round-6 bc-compaction, read at `isdf/core.py:534`) | mock mirrors `psi_G_store.py:147` | GREEN in 7885154 leg A2 |
| `test_sigma_ppm_gates::test_g2_branch_window_tiles_are_frozen` | (c) stale pin: G2 npz frozen 2026-07-07; `d011a36` (2026-07-23) deliberately reconditioned the Σc HGL crossing quadrature (ξ floor, `A_core` cap) — crossing-core node ladder changed 103→98 | regenerated via the module's own `_regenerate_g2_reference()` (job 7885154 step 0, CPU/f64) | GREEN in 7885154 leg A2 |
| `test_gw_jax_regression::test_gnppm_matches_reference`, `::test_bispinor_gnppm_matches_reference` | platform-migrated pins: refs frozen on Perlmutter GPU (b7654ee, 2026-07-21); on Frontera CPU/FFI the drift is EXACTLY one unit in the 6th printed decimal (max delta 1.000e-6 eV, sigC/sigXC only, sigX exact-0; 20/2484 resp. 24/1620 rows) against `atol=1e-6` | re-froze both `sigma_diag_*_ref.dat` from the job-7885154 session outputs (byte-identical copies; artifacts under `/scratch2/08271/jackmc/pytest_p11/tmpA2_7885154/`) | physics anchored independently: si_cohsex BGW anchor + cohsex Tier-1 PASS on this platform, ALL Tier-2 self-checking invariance gates PASS (restart≡fresh, μ-pad flips, SC-iter, rotations, IBZ≡full-BZ). Flagged for owner awareness (Tier-1 pin refresh) |
| `test_runtime_distributed::test_set_default_env_defaults` | (a) real gap: with the CUDA plugin already in `sys.modules` (every pytest process — jax discovery imports it during the first backend init) `skip_gpu_plugin_discovery` took the too-late branch and returned WITHOUT the arm-2 `JAX_PLATFORMS=cpu` demotion; same gap in the already-armed (`done`) branch | both branches now re-apply the demotion pinning when a GPU is requested but absent | logic driven directly on both branches (login python, stubbed plugin module): env pins to `cpu`; suite validation next leg |
| `test_charge_zeta_route::test_rank_truncate_refuses_rather_than_downgrading` (leg C) | (c) stale pin: `_replicate_rank_truncate_ok` widening (ladder R15.1) deliberately made the old `_OVER_CAP` (13.4 GiB stack, ~4 GiB per q-batch) succeed per-q-batch; refusal now binds at one (μ,μ) factor > 4 GiB (μ>16384) | refusal case moved to `_OVER_FACTOR_CAP` (μ=17000); NEW test pins the widening (stack-over-cap still runs per-q-batch) | threshold arithmetic verified against `isdf/core.py` caps; suite validation next leg |
| `test_contract_bands` (9) + `test_projection_lgemm` (2) failing at 1 device | test defect: `assert n_dev >= 4` instead of the suite-wide `pytest.skip` convention | `_mesh()` skips below 4 devices | GREEN under leg C (their real assertions, 4 devices); bare leg now skips cleanly |

## Old-handoff known-fail list, verified against today's tree

| handoff item | verdict today |
|---|---|
| file-IO fixtures | root-caused (zeta_is_done completeness gate vs stale test builders) and FIXED — see above |
| bse_ring_comm vma | STILL PRESENT — the one remaining known-fail class (10 tests); precisely diagnosed, register row added |
| kmeans multi-rank segfault | not reproducible in pytest scope; P>1 thread-main refusal FIXED repo-side (24e4dc3, subsumed e97e8ed), fastloop 2x2 shard leg certified green (CLAIMS 17); true multi-process kmeans belongs to the P>1 srun leg |
| GN-PPM pred remat | remat gates (`test_staged_reshard*`, the "Involuntary full rematerialization" pins) ALL GREEN in leg C (32 tests) — consistent with the square-mesh ruling (6da7f69) closing the K.2 rectangular-mesh remat class |

## Instrument notes (how to run this suite; the 7885150 lesson)

1. Export the environment INSIDE the container: apptainer does not forward
   the host `LD_LIBRARY_PATH`, and without it the required-FFI gate
   refuses (`libhdf5.so.310` unresolvable) — that single mistake produced
   68 failed + 29 errors in job 7885150.  Pattern: job script
   `/scratch2/08271/jackmc/pytest_p11/run_pytest2.sbatch`.
2. `tests/test_ffi_linalg_contract.py` must be `--ignore`d in the bare
   leg (import-time death without PMI2 glue) and run under
   `srun --mpi=pmi2 -n1` with `config/frontera/mpi_transport_env.sh`
   sourced — green there.
3. Per-test timeout: `pytest-timeout` staged on `PYTHONPATH`
   (`--timeout=2400 --timeout-method=signal`); do NOT also pass
   `-p pytest_timeout` (double registration).
4. e2e regression fixtures run the drivers on the CPU node via
   `ISDF_COHSEX_TEST_PLATFORM=auto` (jax native pick); compile cache under
   `$SCRATCH` keeps the whole suite ~21 min.
