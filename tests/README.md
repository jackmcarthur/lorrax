# LORRAX tests

Three kinds of checks:

- **`hsuite/`**: the production drivers run end to end on a tiny magnetic H2⁻
  fixture and a bcc Na fixture at P4 ([README](hsuite/README.md)). It is the
  verdict for a driver or physics change.
- **Lint**, `../tools/lint/`: static AST suites that read source and run no
  physics (layering, cross-file requests, environment registry and grammar,
  FFT shard-map context, nested-call arity, raw reductions, docs register,
  runtime-first import order, the local-eigensolver owner). `tools/release_check.sh`
  runs them on a login node.
- **The unit layer**, `test_*.py` here and under `tdagw/`: about two dozen tests,
  each protecting a physical or contractual invariant the hsuite cannot reach
  (an exact identity against an independent oracle, a refusal that must fire, two
  routes that must agree, an output another tool parses). Related checks share
  one test. Do not add a test that re-checks an hsuite stage or pins incidental
  implementation behaviour (a program count, a message, a byte price).

`conftest.py` seals the first-party source closure
(`runtime.source_closure.ensure_source_closure`) once per test process, before
any test imports a service, as the drivers do at start-up. Without it a test
that imports `gw.*` first takes `distrib_la` and its peers from the
interpreter's install (the sealed release under `lx test`), and a later
`file_io` import in the same pytest worker refuses with `SourceClosureError`.
A new test needs no bootstrap import of its own. Run the unit layer with
`JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4`.

## The unit layer

| file | what it holds |
|---|---|
| `test_slab_io_cpu.py` | `SlabIO` serial tier: padded block clips, reads back, a dataset that exists is reused or refused; `read_multi` / `read_slabs` equal `read_slab` |
| `test_distrib_la_eigh_layout.py` | eigh through the padded and shifted layout returns the unpadded eigensystem |
| `test_distrib_la_one_device.py` | the one-device route's eigh and LU equal NumPy |
| `test_polar_device_cut_cpu.py` | the device spectral cut equals the host sequence |
| `test_nodal_htransform_cpu.py` | the nodal htransform (the hsuite path now has off-grid points, but no oracle): WS kernel exact on the grid, star covariant, better than its controls; metric, Loewdin and its refusal; zero-weight drop and face-versus-q-batched routes; a non-orbit-closed node set is refused |
| `test_bse_kgrid_dipoles_cpu.py` | densified dipoles equal the exact ones on and off the grid |
| `test_bse_helicity_cpu.py` | σ± helicity labels on a C3 toy, with the flipped conventions failing |
| `test_bsemat_split_cpu.py` | the ISDF bsemat split equals the plane-wave head, wings and body |
| `test_bse_q0_head_exact_overlap.py` | the q = 0 head content is exact |
| `test_eqp_fullzone_cpu.py` | the eqp reader: wedge, unfold and full-zone blocks agree |
| `test_bgw_epsmat_cpu.py` | the BerkeleyGW ε⁻¹ block equals the dense RPA inverse |
| `test_bgw_w_import_cpu.py` | BerkeleyGW W import: a symmetry-image q projects to the dense Z_cᴴ(ε⁻¹_c − 1)vZ_c (phase and TR conjugation each fail in a red twin); q coverage and frequency rules; the GW reader refuses another W source, frequency or missing probe |
| `test_sigma_on_shell_cpu.py` | the on-shell Σ fold equals its closed form |
| `test_tangential_head_ordered_fold.py` | ordered wings equal the direct Lehmann sum |
| `test_tangential_decoupled.py` | the decoupled face routes equal the local round |
| `test_response_bank_blocks_cpu.py` | four-block χ₀ equals the exact ordered-pair scan |
| `test_cd_kernels.py` | contour-deformation kernels equal a dense per-q sum |
| `test_trs_qe_decides.py` | QE's data decides the 2c time-reversal verdict |
| `test_pair_ws_cpu.py` | `common.ws_images`: the table holds every minimal image |
| `test_orbital_modern_route.py` | htransform's orbital totals: sum-over-states, star weights, per-parent sum |
| `test_ordered_collectives.py` | `lxkit.ordered_psum` is the rank-order fold |
| `test_log_routing_cpu.py` | a `RunSession` routes warnings (as their message blocks with ids), progress and stage banners to the main log, the rank stream and the JSON sidecar |
| `test_native_closure.py` | the sealed library closure: each recorded family mapped once as the recorded file |
| `test_module_cubin_store.py` | accepted module image assets remain regular/self-contained; source, bundle and payload mismatches refuse (stdlib-only, also runs directly) |
| `test_network_env_multinode.py` | the multi-node network decision matrix |
| `test_perf_check.py` | `scripts/perf_check.py`'s judgement: fatbin SASS/PTX parse, binding (clone shape passes, SMT-sibling and oversubscribed teams and overlapping masks are flagged), the 15% reference rule, a second-process compile-cache miss fails |
| `test_jax_window.py` | `runtime.jax_support`'s per-leg JAX window: 0.12 everywhere, 0.9.1 on ROCm only, keyed by the startup leg (stdlib-only) |
| `tdagw/test_prop_integrators_cpu.py` | the DP54 stepper takes SciPy RK45's steps |
| `tdagw/test_validation_rabi_cpu.py` | Rabi flopping against the exact rotating-frame solution, every integrator |

## Device and script gates

These compare a native route with the XLA route on a device, or run on a 2x2 mesh
as scripts; they skip under pytest where there is no device. Run them with
`python -m tests.<name>` as each file's docstring says:
`test_rocsolver_eigh_gate`, `test_rocsolver_lu_gate`, `test_subspace_native_gate`,
`test_active_local_native_gate`, `test_distrib_la_elpa_gate`, `test_fourier_plan_xla_gate`,
`test_kconv_xla_gate`, `test_kdft_{kbox,outer,parent,plane}_gate`, `test_bse_w_blocks`, `test_bse_default_solver`,
`test_exciton_bands_davidson`, `test_bse_nontda_matrix_free`,
`test_tdagw_{collision_fullrho,collision_instbasis,dynsig,dynsig_sigma,model,kernel_export}`,
`phdf5_read_multi_gate.py`, `phdf5_sector_race_gate.py`. `tdagw/_*_516_case.py` are the
manual CrI3 diagnostics, run explicitly at P4 on compute nodes; `fixtures_cd/` holds the
Fe harness the CD laws were pinned against (`src/gw/cd/laws.py` cites it).
`multinode/` and `replays/` are launch probes and scripts, not tests.
