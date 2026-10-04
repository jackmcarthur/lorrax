# Codebase map

This page says where each part of LORRAX lives: every package under `src/`
and `services/`, what it owns, how it is entered, and who imports it, then
one line per module. It is for a developer or agent about to read or change
the source. What a module may import is a rule, owned by
[the three levels](architecture/layers.md); contracts, equations and run
policy live on the owner pages of the [documentation register](index.md#register).

## The tree

| directory | holds |
|---|---|
| `src/` | the application: 13 packages, importable with `src/` on `PYTHONPATH` |
| `services/<name>/src/<name>/` | 7 standalone packages, each with its own `pyproject.toml`; some add `bench/` drivers and `docs/` |
| `tests/` | the driver-level suite `tests/hsuite` and the static rule gates ([contributing](contributing.md)) |
| `config/` | machine recipes: environment, native builds, the Perlmutter module |
| `tools/`, `scripts/` | generators and maintenance scripts (for example `tools/gen_input_reference.py`) |
| `docs/` | this documentation |

## Packages at a glance

Each package has a level (L1 physics, L2 numerical routines, L3 substrate;
`tests/test_layering.py`). Imports run downhill only, so an L3 package may be
imported by anything, an L2 package by L1 and L2 code, and an L1 package only
by L1 code. A service is reached through its top-level package and nowhere
else.

| package | level | owns | entry points | imported by |
|---|---|---|---|---|
| `gw` | L1 | the GW and QSGW driver: ζ orchestration, $V_q$, screening (shared pole, MPA, plasmon pole, ladder), Σ, the $q\to0$ heads, SC maps, downfolding | `python -m gw.gw_jax`, `gw.kin_ion_io`, `gw.downfold_cli`, `gw.eqp_bgw`, `gw.plane_wave_pipeline` (opt-in) | `bse`, `bandstructure`, `file_io`, `psp`, `centroid`, `isdf`, `lxkit.deck_doctor` |
| `bse` | L1 | the Bethe–Salpeter equation: kernels, TDA and full solvers, absorption, exciton bands, the $W_\mathrm{BSE}$ ladder | `python -m bse.bse_jax`, `bse.exciton_bands`, `bse.absorption_haydock`, `bse.bse_feast`, `bse.bse_kpm`, `bse.bse_w_exact` | `gw.screening_bse`, `file_io.restart_bundle` (both lazily) |
| `bandstructure` | L1 | band interpolation (htransform), fine-k wavefunctions for the BSE, band operators | `python -m bandstructure.htransform` | `bse` |
| `centroid` | L1 (`kmeans_isdf` L2, `distribution` L3) | ISDF centroid selection: weighted k-means, pivoted-Cholesky pruning, the sampling metric | `python -m centroid.kmeans_cli` | `gw` |
| `isdf` | L1 | ζ-fit primitives (pair Grams, $C_q$ conditioning, the μ-batch route) and the whole-state Galerkin basis | library | `gw`, `bandstructure`, `bse`, `centroid` |
| `psp` | L1 | the plane-wave DFT layer: $T$, $V_\mathrm{loc}$, $V_\mathrm{NL}$, DFT+U and XC operators, pseudopotentials, radial tables, the preprocessing and mean-field drivers | `python -m psp.get_dipole_mtxels`, `psp.run_nscf`, `psp.run_dense_h`, `psp.run_sternheimer`, `psp.get_DFT_mtxels`, `psp.kpm_dos`, `psp.orbital_magnetization` | `gw`, `bandstructure`, `common`, `file_io`, `solvers` |
| `postprocess` | L1 | WFN → `WFN_qp.h5` rotation | `python -m postprocess.rotate_wfn_to_qp` | nothing |
| `file_io` | L1 readers and writers; L3 transport | SlabIO (the sharded HDF5 transport) and every format: the restart bundle, Σ outputs, model stores, `dipole.h5`, `kin_ion.h5`, WFN | library | `gw`, `bse`, `psp`, `bandstructure`, `centroid` |
| `common` | mixed: L1 physics helpers, L2 `rank_criterion`, `spectral_closure`, `pivoted_cholesky`, L3 collectives and glue | shared helpers: collectives, sharding and FFT glue, timing, units, WFN transforms, parallel-transport kernels | library | every package |
| `solvers` | L2 | iterative eigensolvers and spectral methods with no physics: Davidson, Lanczos, KPM, contour quadrature, pseudobands, Sternheimer CG | library | `bse`, `psp` |
| `mixing` | L2 | fixed-point acceleration: Anderson, rCROP | library | `gw.sc_iteration` (the SC loop and eqp2) |
| `runtime` | L3, except two modules (below the table) | process bootstrap, the driver session and report stream, padding receipts, compiled-memory checks, environment grammar, source closure, the fixed streaming tile | library; every driver calls `runtime.initialize_communicator_stack()` first | every driver |
| `ffi` | L3 | the bridge to native libraries: loading and attestation, capability gates, FFT and k-convolution router, GEMM, parallel HDF5, the contour accumulator; the C++ sources | library; `ffi/cpp/stage/seal_bundle.py` seals a native bundle | `gw`, `bse`, `common`, `file_io`, `psp`, `bandstructure` |

`runtime.cli_seam` and `runtime.tiles` are not named in the layer map
(`tests/test_layering.py`), so they take the L1 default; the rest of
`runtime` is L3.

## `src/gw`

| Module | Role |
|---|---|
| `__init__.py` | Package marker for GW and COHSEX drivers. |
| `band_extrapolation.py` | Plans and evaluates self-energy band-window extrapolations. |
| `band_partition.py` | Builds the three-way QSGW band partition. |
| `centroid_k_unfold.py` | Contracts raw-parent k blocks on the orbit-packed centroid basis. |
| `cohsex_sigma.py` | Orchestrates the static self-energy path. |
| `comm_model.py` | O(1) collective cost model used by the planners. |
| `compute_vcoul.py` | Dispatches Coulomb-matrix construction by dimensionality. |
| `contour_accumulator.py` | Re-export of `ffi.contour` (the contour accumulator) for `gw.w_isdf`. |
| `coulomb/__init__.py` | Compatibility package: dimension-aware adapters over the `vcoul` service. |
| `coulomb/base.py` | Compatibility import of the `vcoul` Coulomb kernels. |
| `coulomb/box_0d.py` | Compatibility import of the `vcoul` 0-D cell-box kernel. |
| `coulomb/bulk_3d.py` | Compatibility import of the `vcoul` 3-D bulk kernel. |
| `coulomb/slab_2d.py` | Compatibility import of the `vcoul` 2-D slab kernel. |
| `degen_average.py` | Averages band quantities over degeneracy blocks. |
| `downfold.py` | Builds and applies reduced interaction bases. |
| `downfold_cli.py` | Command-line entry point for interaction downfolding. |
| `downfold_config.py` | Parses and validates downfold configuration. |
| `downfold_run.py` | Orchestrates a downfold calculation. |
| `dynamic_sigma.py` | Post-processes frequency-dependent self-energy data. |
| `efermi.py` | Resolves occupations and Fermi levels. |
| `eqp_bgw.py` | Writes BerkeleyGW-compatible quasiparticle tables. |
| `experimental/__init__.py` | Package for staged GW features not yet wired into `gw_jax`. |
| `experimental/head_wing_schur.py` | Sharded head/wing/body Schur decomposition of W; the head-channel specs are live. |
| `fermi_surface.py` | Builds finite-occupation Fermi-surface quadrature. |
| `gflat_memory_model.py` | The route-G ζ-fit planner and the centroid-load tile rule. |
| `greens_function_kernel.py` | Builds the parent Green operators (`build_G_parents`, `build_G_tau`, the `face_green_product` SUMMA) and moves them with symmetry actions. |
| `gw_config.py` | Defines, parses, and validates GW runtime configuration. |
| `gw_init.py` | Loads inputs and orchestrates initialization stages. |
| `gw_jax.py` | Main GWJAX command-line driver. |
| `gw_output.py` | Serializes driver results and provenance. |
| `hartree.py` | The direct (Hartree) field of the occupied states, as band matrices. |
| `head_channel.py` | Places head-channel data on nonzero-q layouts. |
| `head_correction.py` | Constructs the Gamma-point head correction. |
| `head_densify.py` | Densifies arrays whose divergent head is stored separately. |
| `isdf_fitting.py` | Runs the route-G ζ fit and writes `zeta_q.h5` (`fit_zeta_to_h5`). |
| `kin_ion_io.py` | Produces and reads kinetic-plus-ionic matrices. |
| `minimax_config.py` | Defines shared minimax and sigma-quadrature settings. |
| `minimax_screening.py` | Adapts certified minimax rules to screening windows and fitted kernels. |
| `mixed_basis_pair_convolution.py` | Pair convolution of two plane-wave-sphere operators onto a response sphere; real-space path only. |
| `mpa/__init__.py` | Package for the MPA screening model: fit, schedule, driver and Σ. |
| `mpa/diagnostics.py` | MPA-fit instruments: conditioning, held-out residuals, perturbation refits, residue widths. |
| `mpa/evaluator.py` | MPA scalar oracle and the entry point to the minimax quadrature service. |
| `mpa/fit_driver.py` | Runs the MPA fit stage: read a column block, fit, write, finalize. |
| `mpa/model.py` | Builds one disk-bounded MPA screening model. |
| `mpa/pade_fit.py` | Fits n_p complex poles to 2·n_p complex samples of W_c. |
| `mpa/sample_plan.py` | The complex-frequency sampling plan, as data. |
| `mpa/sampling.py` | Double-parallel sample grid for the MPA fit. |
| `mpa/sector_sigma.py` | Ordered photon sectors in the common Σ frequency-quadrature executor. |
| `mpa/sigma.py` | Executes an MPA Σ plan with the GN spatial kernel. |
| `mpa/sigma_windows.py` | Derives MPA Σ frequency windows from the fitted pole geometry. |
| `mpa/small_eig.py` | Eigenvalues of small non-symmetric complex matrices in JAX. |
| `mpa/tiling.py` | Walks the fit stage over (q, ν-column) blocks under its memory rule. |
| `photon_direct_head.py` | First-order direct bulk photon head from the dipole vertex. |
| `photon_layout.py` | Defines the packed current-channel array layout. |
| `photon_sigma.py` | Evaluates self-energy contributions in the packed current layout. |
| `plane_wave_pipeline.py` | One-shot real-space (ISDF-free) GW stages, ψ(G) → χ₀ → W → Σ_x, Σ_c(ω); `python -m gw.plane_wave_pipeline` (not wired). |
| `plane_wave_screening.py` | Screened interaction W_q(G, G') on the plane-wave response sphere, per wedge q (not wired). |
| `ppm_accumulators.py` | Accumulates plasmon-pole self-energy terms. |
| `ppm_pipeline.py` | Orchestrates plasmon-pole setup and evaluation. |
| `ppm_sigma.py` | Evaluates the GN-PPM correlation self-energy. |
| `ppm_tau_kernel.py` | The shared device τ kernel of dynamic Σ (plasmon pole, MPA, shared pole). |
| `ppm_windows.py` | Dynamic-Σ branch records and the Σ broadening (η) resolver. |
| `production_report.py` | Renders the human-readable production configuration report. |
| `qgrid_symmetry.py` | Resolves q-grid symmetry policy and index tables. |
| `qp_support.py` | The sampled Σ(ω) support: where the dynamic Σ grid reaches, padded then held. |
| `qsgw_density.py` | Builds density state for QSGW iterations. |
| `qsgw_head.py` | Builds finite-link velocity and head data for QSGW. |
| `qsgw_utils.py` | The QSGW Σ_xc build, the one-shot `solve_qp` seam (Σ_xc + V_H, no QP root) and matrix I/O helpers. |
| `quadrature_log.py` | Records the quadrature rules a run used, for the production report. |
| `response_bank.py` | Response-bank algebra for the shared-pole construction. |
| `restart_q_storage.py` | Stores the producer's q parents and the record that authenticates their unfold. |
| `sc_iteration.py` | Runs one self-consistent iteration map. |
| `sc_state_identity.py` | Map-0 QP identities from multiplet-projector overlaps. |
| `scissor.py` | Applies and reports scissor corrections. |
| `screening.py` | Plans and executes screening calculations. |
| `screening_bse.py` | The GW-side stage of `screening_diagrams = w_bse` / `w_rpa_resolvent`. |
| `shared_pole_capacity.py` | Device-byte accounting for one shared-pole construction. |
| `shared_pole_constructor.py` | Constructs the physical shared real-pole W (tangential Hermite/Ritz). |
| `shared_pole_directions.py` | Selects directions and builds the per-parent state panels. |
| `shared_pole_execution.py` | Runs shared-pole equations on the full x/y mesh. |
| `shared_pole_gates.py` | Measured gates and diagnostics of a constructed shared-pole model. |
| `shared_pole_head.py` | Evaluates current Gamma shared-pole W and routes its scalar head through the common head and MPA owners. |
| `shared_pole_local.py` | Runs shared-pole parent rounds, one parent per rank. |
| `shared_pole_pencil.py` | Builds resolvent-identity pencil columns for the shared-pole construction. |
| `shared_pole_recipe.py` | Shared real-pole input recipe and gate table. |
| `shared_pole_reduction.py` | Ritz reduction of the shared-pole pencils. |
| `shared_pole_screening.py` | Deck-driven shared real-pole screening stage. |
| `shared_pole_sectors.py` | Charge/current cross pencils on parent-local stacks. |
| `sigma_box_plan.py` | Denominator-box quadrature plan for dynamic Σ(ω). |
| `sigma_dispatch.py` | Dispatches one self-energy call per resolved compute mode. |
| `sigma_x_bispinor.py` | Implements bare-current exchange routes for spinor inputs. |
| `static_gauge_response.py` | Builds packed static-gauge response inputs. |
| `static_screening.py` | Static W(0) and its q→0 head: an RPA Dyson solve for a BSE restart without W0, or the retained shared-pole model at ω = 0 for the W0 persist. |
| `subtile_stream.py` | A node rule evaluated sub-tile by sub-tile: the loop of the direct χ₀ stream and of Σ τ, with orbit-cut row passes. |
| `v_q_bispinor.py` | Builds the packed bare-current interaction operator. |
| `v_q_g_flat.py` | Builds Coulomb matrices from flattened reciprocal-space data. |
| `vcoul.py` | Compatibility shim: the deck-facing q = 0 mini-BZ averages and Voronoi point wrapping over the `vcoul` service. |
| `w_av.py` | Builds cell-averaging stencils for screened interactions. |
| `w_isdf.py` | Orchestrates independent-particle response and screened interaction stages. |
| `wavefunction_bundle.py` | Bundles wavefunction arrays and band-basis projections. |

## `src/common`

| Module | Role |
|---|---|
| `__init__.py` | Package marker for shared LORRAX utilities. |
| `async_io.py` | A single-worker host dispatcher with bounded back-pressure. |
| `band_degeneracy.py` | Finds degeneracy blocks and validates band-window boundaries. |
| `bispinor_init.py` | The kinetic-balance lift: small-component spinors from the large components. |
| `centroid_basis.py` | The in-memory centroid order: whole symmetry orbits per shard. |
| `chi_from_dipole.py` | Builds response data from dipole matrix elements. |
| `collectives.py` | Wraps process collectives and communicator warm-up. |
| `contract_bands.py` | Contracts band axes under explicit chunking. |
| `coulomb_sphere.py` | The per-q bare-Coulomb G-sphere in the padded WFN layout. |
| `fft_helpers.py` | Provides the canonical sharded real/reciprocal FFT factories. |
| `four_current_model.py` | Defines shared packed-current model vocabulary and validation. |
| `fourier_plan.py` | `LocalFourierPlan`: one local separable DFT with restricted per-axis supports. |
| `gamma_matrices.py` | Supplies spinor gamma-matrix conventions. |
| `gauss_legendre.py` | Compatibility re-export of `vcoul`'s finite-interval rule; no importers. |
| `gpu_utils.py` | The per-device budget (`memory_per_device_gb`), stage prices, `warn_over_budget` and memory detection. |
| `grouped_layout.py` | Describes grouped array layouts and transformations. |
| `gvec_fft_box.py` | Maps reciprocal-vector spheres to and from FFT boxes. |
| `jax_compile_cache.py` | Configures and audits JAX compilation caches. |
| `jax_profile.py` | Controls bounded JAX profiling captures. |
| `kq_mapping.py` | Builds k/q index mappings. |
| `meta.py` | Stores system metadata shared by calculation stages. |
| `mtxel_sweep.py` | Evaluates matrix elements in bounded sweeps. |
| `parallel_transport.py` | Constructs band-subspace parallel-transport links. |
| `pivoted_cholesky.py` | Implements shared pivoted-Cholesky selection helpers. |
| `preprocessing_output.py` | Writes preprocessing reports and provenance. |
| `progress.py` | Renders rank-aware progress output. |
| `provenance.py` | Defines provenance stamps and validation helpers. |
| `psi_G_store.py` | Maintains the host-resident reciprocal-wavefunction cache. |
| `rank_criterion.py` | Applies the shared numerical-rank criterion. |
| `sanity.py` | Runs scientific sanity checks and diagnostics. |
| `scientific_output.py` | Formats shared scientific output records. |
| `shard_map.py` | Supplies common shard-map wrappers and checks. |
| `sharding_fit.py` | A `PartitionSpec` that divides the extents in hand. |
| `spectral_closure.py` | Closes truncation cuts over spectral degeneracies. |
| `staged_reshard.py` | Implements staged collective reshards. |
| `timing.py` | Records scoped timing measurements. |
| `units.py` | Defines unit conversions. |
| `vma.py` | `mark_varying`: marks a loop carry as device-varying across JAX versions. |
| `wfn_layout.py` | Describes wavefunction shardings and layout conversions. |
| `wfn_transforms.py` | Loads and transforms wavefunctions in band chunks. |
| `zeta_projection.py` | Projects zeta data between basis layouts. |

## `src/centroid`

| Module | Role |
|---|---|
| `__init__.py` | Package marker for centroid selection. |
| `charge_density.py` | Averages a grid field over the space group (`symmetrize_on_grid`). |
| `distribution.py` | Distributes centroid work and selected points. |
| `kmeans_cli.py` | Command-line entry point for centroid generation. |
| `kmeans_isdf.py` | Implements centroid selection and refinement. |
| `kmeans_plot.py` | Produces centroid diagnostics and plots. |
| `pivoted_cholesky.py` | Selects centroid candidates by pivoted Cholesky. |
| `production_output.py` | Writes centroid-production reports and provenance. |
| `sampling_metric.py` | Resolves stored-k sampling weights and quadrature tables. |

## `src/file_io`

| Module | Role |
|---|---|
| `__init__.py` | Exposes the supported file-I/O surface. |
| `_slab_io_ffi.py` | Implements the native SlabIO transport binding. |
| `_slab_io_rank.py` | SlabIO's per-rank streamed tier: an operator written segment by segment and read back by output. |
| `_slab_io_serial.py` | Implements serial SlabIO transport. |
| `centroids.py` | Loads centroid tables through one format and symmetry entry point. |
| `commit_state.py` | Persistent completion receipt for collective artifact writes. |
| `dipole.py` | The `dipole.h5` format: the q → 0 velocity matrix, band energies, finite-q terms. |
| `epsreader.py` | Reads BerkeleyGW `eps0mat.h5` / `epsmat.h5` (the `wcoul0_source = epshead` head). |
| `h5_journal.py` | Records bounded HDF5 operation journals. |
| `hdf5_owner.py` | Enforces one process owner for an HDF5 file. |
| `host_tile_store.py` | Tile-major pinned host store for a grid of sharded tiles (not wired). |
| `io_timing.py` | Optional per-rank SlabIO wall trace. |
| `isdf_header.py` | Reads and validates ISDF HDF5 headers. |
| `kin_ion.py` | Writes `kin_ion.h5`; its readers are in `restart_bundle`. |
| `mf_header.py` | Reads mean-field metadata headers. |
| `mpa_store.py` | Frequency-resolved W restart tensors and the multipole (MPA) $B/\Omega$ fit store. |
| `parallel_transport.py` | Reads and writes parallel-transport data. |
| `paths.py` | Resolves configured input and output paths. |
| `qe_save_reader.py` | Reads bounded Quantum ESPRESSO save-directory metadata. |
| `qp_wfn.py` | Reads and writes quasiparticle wavefunction data. |
| `read_bgw_vcoul.py` | Compatibility shim over the `vcoul` service's parser of BerkeleyGW's `vcoul` file. |
| `restart_bundle.py` | Reads the GW restart bundle. |
| `shared_pole_store.py` | Shared real-pole model and construction-scratch I/O. |
| `sigma_checkpoint.py` | The swept Σ(ω) state kept between the τ sweep and finalize (`tmp/sigma_checkpoint_oneshot.h5`). |
| `sigma_output.py` | Writes self-energy and quasiparticle outputs. |
| `slab_io.py` | Exposes sharded slab reads and writes. |
| `static_gauge_head.py` | Reads and writes static-gauge head data. |
| `tagged_arrays.py` | Writes the restart bundle: restart state, W0, head scalars, parent wavefunctions. |
| `wfn_basis.py` | Provenance of wavefunctions sampled on an ordered centroid basis (`WavefunctionBasisReceipt`). |
| `wfn_writer.py` | Writes wavefunction files. |

## `src/bse`

The BSE contract is [BSE](architecture/bse.md); this table only says where each part lives.

| Module | Role |
|---|---|
| `__init__.py` | Package marker for the BSE. |
| `absorption_common.py` | Shared absorption helpers: dipole slicing to the window, the ⟨0\|r̂\|S⟩ contraction, Lorentzian, JDOS, Kramers–Kronig, `.dat`/`.h5` writers. |
| `absorption_haydock.py` | ε₂(ω) by the Haydock continued fraction on the TDA BSE; `python -m bse.absorption_haydock`. |
| `bse_davidson_helpers.py` | Start subspace and diagonal preconditioners for `solvers.davidson.davidson` on the BSE vector layout. |
| `bse_densify.py` | Coarse-to-fine densification of the BSE bundle under `bse_k_grid`. |
| `bse_feast.py` | FEAST contour eigensolver, its GMRES solves and the spectral deflation the W_BSE ladder reuses. |
| `bse_head.py` | The q = 0 Coulomb head: its scalars and their rank-one injection. |
| `bse_io.py` | Compatibility facade: re-exports names from `bse_window`, `bse_head`, `bse_densify`, `bse_loading` and the window names of `common.band_degeneracy`. |
| `bse_jax.py` | The BSE driver: `python -m bse.bse_jax`. |
| `bse_kpm.py` | KPM Chebyshev density of states of the BSE Hamiltonian. |
| `bse_lanczos.py` | `solve_bse_sharded`: TDA Lanczos, Davidson and thick-restart Lanczos; non-TDA hands off to `bse_nontda`. |
| `bse_loading.py` | Reads a GW restart into a BSE bundle: window, padding and q = 0 head; builds a missing static W(0) through `gw.static_screening` when the restart has none. |
| `bse_nontda.py` | Structure-preserving full (non-TDA) eigensolver: dense build, and an opt-in matrix-free solver. |
| `bse_preconditioner.py` | Transition energies, the exchange pair amplitude and the exchange spin weight. |
| `bse_ring_comm.py` | The BSE mesh, shardings, and the full (A, B) ring matvec. |
| `bse_stack_matvec.py` | The trial-stack TDA matvec and the non-TDA pair applier. |
| `bse_w_exact.py` | Exact W_c(ω) by shifted solves on the RPA density resolvent; the TRS pair gauge the ladder uses. |
| `bse_window.py` | The band window, its padding, the `--eqp` re-slice and the `eigenvectors.h5` writer. |
| `exchange_path.py` | Exchange tiles V_Q along an exciton momentum path. |
| `exciton_bands.py` | Finite-momentum TDA exciton bands E_S(Q); `python -m bse.exciton_bands`. |
| `head_resolvent.py` | The ladder's q = 0 macroscopic tensor Ξ_ij(z). |
| `vq_interp.py` | Arbitrary-Q bare-exchange tile V_Q. |
| `w_ladder.py` | Ladder-corrected W_BSE(z) for `screening_diagrams = w_bse`. |
| `w_omega_chain.py` | W_q(ω) from one block-Lanczos chain; called by `bse_w_exact`. |

## `src/bandstructure`

The method and the flags: [band interpolation and exciton bands](how-to/htransform-and-exciton-bands.md).

| Module | Role |
|---|---|
| `__init__.py` | Package marker. |
| `htransform.py` | The htransform driver: `python -m bandstructure.htransform`; band coloring and grid moments. |
| `fh_interp.py` | The $f(H)$ interpolation library: the $f$-transform, `build_fH_R`, the path solve, Newton inversion, the QP rotation of the compact state. |
| `bse_setup.py` | Fine-k wavefunctions at the coarse centroids from the same $f(H)$ (`compute_wfns_fi`), for BSE densification and exciton bands. |
| `orbital.py` | Band operators (spin, orbital character, the stored velocity), their interpolation, the path orbital moments and coarse orbital totals, and the grid spin moments. |
| `production_report.py` | The rank-0 report `htransform.out`. |

## `src/isdf`

| Module | Role |
|---|---|
| `__init__.py` | Exposes the ζ-fit and centroid-Gram primitives. |
| `core.py` | ISDF primitives: ψ and centroids → ζ interpolation vectors, pair Grams ([face-ψ ζ fit](architecture/zeta_fit_face_psi_cct.md)). |
| `cplus.py` | $C_q^+$ by rank truncation: the one conditioning seam of the charge fit. |
| `pair_kernels.py` | The pair-projector GEMM of the μ-batch ζ fit. |
| `zeta_mubatch.py` | The μ-batch ζ fit, route G: $Z_q(G)$ by centroid batches ([μ-batch fit](architecture/zeta_fit_mubatch.md)). |
| `galerkin.py` | The whole-state Galerkin basis of htransform: randomized-QRCP selection, factorization, projection, its artifact. |

## `src/psp`

A namespace package (no `__init__.py`).

| Module | Role |
|---|---|
| `dft_operators.py` | The plane-wave DFT Hamiltonian: build, apply, differentiate (kinetic velocity included). |
| `dft_precond.py` | Davidson preconditioner and initial guess for the DFT Hamiltonian. |
| `finite_q_head_interp.py` | Finite-q head and wing interpolation of the screened W; a proof of concept. |
| `get_DFT_mtxels.py` | DFT Hamiltonian matrix elements; spin-degeneracy helpers. |
| `get_dipole_mtxels.py` | The dipole driver: the q → 0 velocity matrix `dipole.h5` and the parallel-transport artifact. |
| `gvec_utils.py` | G-vector bookkeeping. |
| `h_dft.py` | $H\psi$ as a black box for Davidson. |
| `hubbard_ops.py` | QE's DFT+U operator $V_U(k)$ and its k derivative. |
| `ionic_gspace.py` | G-space setup of the ionic potentials and core charge. |
| `kpm_dos.py` | KPM density of states of the DFT Hamiltonian; a driver. |
| `nscf_input.py` | Parses an NSCF input file. |
| `operator_checks.py` | Preflight checks before DFT operators are built. |
| `orbital_magnetization.py` | Orbital magnetization (modern theory) from a spinor WFN; a driver. Theory: `psp/orbital_magnetization_THEORY.md`. |
| `orbital_response.py` | Orbital moments from a given velocity. |
| `pseudos.py` | Pseudopotential loading, element lookup, atom assignment. |
| `radial_tables.py` | Builds the radial Hankel tables from the species data. |
| `run_dense_h.py` | The complete-basis WFN: dense $H_k$, full eigh, every band ([complete-basis WFN](how-to/complete-basis-wfn.md)). |
| `run_nscf.py` | NSCF driver: Davidson eigenstates and pseudobands → `WFN.h5`. |
| `run_sternheimer.py` | Insulating Sternheimer driver for the $G = 0$ source column $\chi_{G'0}(q, 0)$. |
| `scf_potential.py` | Builds the self-consistent DFT potential. |
| `species.py` | Radial data from UPF files. |
| `vnl_ops.py` | The nonlocal pseudopotential operator and its analytic k derivative. |
| `xc.py` | Exchange–correlation potentials by automatic differentiation. |
| `radial/` | Radial transforms (`radial_jax`), QE-convention solid harmonics (`solid_harmonics`), spin–orbit projector algebra (`build_projectors_qe`). |
| `upf/` | UPF parsing: `load_upf`, `normalize`, and the generated data model `upf_model_2_0_1` of the UPF 2.0.1 schema (`qe_pp-2.0.1.xsd`, `UPF-v.2.0.1-format.md` beside it). |

## `src/solvers`

Level L2: none of these modules knows what a band or a k point is.

| Module | Role |
|---|---|
| `__init__.py` | Exposes the KPM and Chebyshev helpers. |
| `davidson.py` | Host API of the one planned Davidson implementation. |
| `davidson_fixed.py` | Planned block Davidson with local or distributed vector storage. |
| `lanczos.py` | Lanczos eigensolvers. |
| `thick_restart_lanczos.py` | Thick-restart Lanczos, fixed shape. |
| `bse_sp_lanczos.py` | Structure-preserving thick-restart Lanczos for the definite BSE eigenproblem. |
| `subspace_numerics.py` | Scale-relative rank discovery shared by the eigensolvers. |
| `chebyshev.py` | Chebyshev expansion and KPM utilities. |
| `dos.py` | Matrix-free density of states by KPM. |
| `quadrature.py` | Contour-integration quadrature for spectral methods (FEAST). |
| `pseudobands.py`, `pseudobands_v2.py` | Pseudobands: stochastic plus Ritz, and Galerkin–Ritz with Gauss-quadrature energies. |
| `projectors.py` | Jitted subspace projectors for the Sternheimer solve. |
| `sternheimer_precond.py` | Teter–Payne–Allan preconditioner. |
| `sternheimer_solve.py` | The level-shifted CG Sternheimer solve; it applies `psp.dft_operators` (a registered layer exception). |

## `src/mixing` and `src/postprocess`

| Module | Role |
|---|---|
| `mixing/acceleration.py` | Anderson (the SC loop) and rCROP (eqp2) acceleration. |
| `postprocess/rotate_wfn_to_qp.py` | `WFN.h5` + `qp_wfn_rotations.h5` → `WFN_qp.h5`, on the WFN's own k wedge. |

## `src/runtime`

| Module | Role |
|---|---|
| `__init__.py` | `initialize_communicator_stack()`: the one startup call every driver makes before importing JAX. |
| `aot_memory.py` | The compiled per-rank peak memory of a kernel, cuFFT scratch included. |
| `cli_seam.py` | Answers `--help` and refuses bad arguments before the runtime starts. |
| `env_flags.py` | The boolean environment-variable grammar. |
| `jax_support.py` | Refuses a JAX stack outside the supported series. |
| `network_env.py` | Site network defaults set before JAX and NCCL start. |
| `padding.py` | Mesh-padded axes: logical-to-carrier receipts, zero padding, masks ([padding](architecture/padding.md)). |
| `pjrt_log_filter.py` | Removes known benign runtime notices from stderr. |
| `production_stream.py` | One production stdout stream per driver invocation. |
| `run_session.py` | One driver invocation: report, stdout, timing, refusal, file table. |
| `source_closure.py` | Seals one coherent source tree of the application and its services. |
| `tiles.py` | The fixed 1 GiB tile every streamed loop sizes against. |
| `xla_memory.py` | XLA's GPU memory pool: its configuration and whether its numbers are real. |

## `src/ffi`

The design, every native target and the build: [the FFI layer](architecture/ffi_layout.md).

| Module | Role |
|---|---|
| `__init__.py` | The per-process FFI dials, by name. |
| `_services.py` | Puts the service packages on the import path for bare-source imports. |
| `gate.py` | `Gate`: one environment-gated, rank-local native capability; announce or refuse. |
| `fft.py` | Batched flat-k 3-D FFT and the k-convolution router. |
| `gemm.py` | Vendor-BLAS batched GEMM. |
| `contour.py` | The contour accumulator, $A[o,q,m,n] \mathrel{+}= p[o]\,c[q,m,n]$. |
| `io.py` | Parallel-HDF5 (MPI-IO) transport, behind SlabIO. |
| `common/` | Locating, loading and registering the shared libraries (`ffi_loader`); dtype helpers; a byte-exact broadcast through the JAX distributed store. |
| `cpp/` | The C++ sources by target (`cufft`, `fftw`, `cublas`, `cblas`, `cublasmp`, `cusolvermp`, `scalapack`, `slate`, `phdf5`, `response`, `symmetry`, `active_subspace`), the build scripts, the build gates, and `stage/seal_bundle.py`, which seals a two-library bundle. |
| `phdf5/` | `ARCHITECTURE.md` of the parallel-HDF5 backend. |

## `services/`

Each service is a standalone package at `services/<name>/src/<name>/`; LORRAX
reaches it through its top-level package only. The linked pages own the
caller contracts ([substrate services](architecture/services.md)).

| service | level | owns | extras |
|---|---|---|---|
| `distrib_la` | L3 | distributed dense linear algebra on the `('x','y')` mesh: plans, GEMM and SUMMA panels, eigh, Cholesky, LU, polar factor, with cuSOLVERMp/cuBLASMp, SLATE, ScaLAPACK and pure-JAX backends ([contract](services/distrib_la/api.md)) | `bench/`, `docs/` |
| `lxkit` | L3 (`deck_doctor` L1) | the foundation the services share: capability gates, the absent-versus-broken probe vocabulary, JAX version shims, process-local placement, native-provider attestation, the cache root; `python -m lxkit.deck_doctor` preflights a deck | |
| `minimax` | L1 | runtime quadrature rules: screening, Σ box rules, response-bank rules, their certificates ([contract](services/minimax.md)) | |
| `symmetry_maps` | L1 | the space group of a deck: k-grid reduction, IBZ ⇄ full-BZ tables, unfolds, q-grid time-reversal policy, the QE-schema binding ([contract](services/symmetry_maps.md)) | `bench/` |
| `vcoul` | L1 | the bare and truncated Coulomb interaction, mini-BZ averages, the Coulomb sphere, BerkeleyGW `vcoul` parsing ([contract](services/vcoul.md)) | `bench/` |
| `wfn_loader` | L1 | ψ(G) from `WFN.h5`, by host read or collective MPI-IO ([contract](services/wfn_loader.md)) | `bench/`, `docs/` |
| `zeta_loader` | L1 | the `zeta_q.h5` reader and format contract ([contract](services/zeta_loader.md)) | `bench/` |
