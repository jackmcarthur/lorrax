# Codebase map

This is the one-line source inventory of `src/gw`, `src/common`,
`src/centroid`, `src/file_io` and the service packages, regenerated
2026-09-25. It says where to start reading;
contracts, equations, shapes, and run policy remain on the owner pages in the
[documentation register](index.md#register).

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
| `contour_accumulator.py` | Accumulates a shared contour correlation into its outputs, tile by tile. |
| `coulomb/__init__.py` | Compatibility package over the `vcoul` service. |
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
| `gflat_memory_model.py` | Plans chunks for flattened reciprocal-space arrays. |
| `greens_function_kernel.py` | Constructs occupied and all-band Green-function blocks. |
| `gw_config.py` | Defines, parses, and validates GW runtime configuration. |
| `gw_init.py` | Loads inputs and orchestrates initialization stages. |
| `gw_jax.py` | Main GWJAX command-line driver. |
| `gw_output.py` | Serializes driver results and provenance. |
| `head_channel.py` | Places head-channel data on nonzero-q layouts. |
| `head_correction.py` | Constructs the Gamma-point head correction. |
| `head_densify.py` | Densifies arrays whose divergent head is stored separately. |
| `isdf_fitting.py` | Orchestrates ISDF fitting and zeta lifecycle stages. |
| `kin_ion_io.py` | Produces and reads kinetic-plus-ionic matrices. |
| `minimax_config.py` | Defines shared minimax and sigma-quadrature settings. |
| `minimax_screening.py` | Adapts certified minimax rules to screening windows and fitted kernels. |
| `mixed_basis_pair_convolution.py` | Pair convolution of two plane-wave-sphere operators onto a response sphere (not wired). |
| `mpa/__init__.py` | Package for the MPA screening model: fit, schedule, driver and Σ. |
| `mpa/diagnostics.py` | MPA-fit instruments: conditioning, held-out residuals, perturbation refits, residue widths. |
| `mpa/evaluator.py` | MPA scalar oracle and the door to the minimax quadrature service. |
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
| `plane_wave_screening.py` | Screened interaction W_q(G, G') on the plane-wave response sphere, per wedge q (not wired). |
| `ppm_accumulators.py` | Accumulates plasmon-pole self-energy terms. |
| `ppm_pipeline.py` | Orchestrates plasmon-pole setup and evaluation. |
| `ppm_sigma.py` | Evaluates the GN-PPM correlation self-energy. |
| `ppm_tau_kernel.py` | Supplies imaginary-time plasmon-pole kernels. |
| `ppm_windows.py` | Builds frequency and energy windows for plasmon-pole evaluation. |
| `production_report.py` | Renders the human-readable production configuration report. |
| `qgrid_symmetry.py` | Resolves q-grid symmetry policy and index tables. |
| `qsgw_density.py` | Builds density state for QSGW iterations. |
| `qsgw_head.py` | Builds finite-link velocity and head data for QSGW. |
| `qsgw_utils.py` | Provides QSGW fixed-point, mixing, and matrix I/O helpers. |
| `quadrature_log.py` | Records the quadrature rules a run used, for the production report. |
| `response_bank.py` | Response-bank algebra for the shared-pole construction. |
| `restart_q_storage.py` | Resolves q-axis restart storage and compatibility. |
| `sc_iteration.py` | Runs one self-consistent iteration map. |
| `sc_state_identity.py` | Map-0 QP identities from multiplet-projector overlaps. |
| `scissor.py` | Applies and reports scissor corrections. |
| `screening.py` | Plans and executes screening calculations. |
| `screening_bse.py` | Exposes screening helpers shared with BSE consumers. |
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
| `v_q_bispinor.py` | Builds the packed bare-current interaction operator. |
| `v_q_g_flat.py` | Builds Coulomb matrices from flattened reciprocal-space data. |
| `vcoul.py` | Compatibility imports for the vcoul service. |
| `w_av.py` | Builds cell-averaging stencils for screened interactions. |
| `w_isdf.py` | Orchestrates independent-particle response and screened interaction stages. |
| `wavefunction_bundle.py` | Bundles wavefunction arrays and band-basis projections. |

## `src/common`

| Module | Role |
|---|---|
| `__init__.py` | Package marker for shared LORRAX utilities. |
| `async_io.py` | Coordinates asynchronous host I/O work. |
| `band_degeneracy.py` | Finds degeneracy blocks and validates band-window boundaries. |
| `bispinor_init.py` | Initializes shared spinor-channel inputs. |
| `centroid_basis.py` | The in-memory centroid order: whole symmetry orbits per shard. |
| `chi_from_dipole.py` | Builds response data from dipole matrix elements. |
| `collectives.py` | Wraps process collectives and communicator warm-up. |
| `contract_bands.py` | Contracts band axes under explicit chunking. |
| `coulomb_sphere.py` | Supplies spherical Coulomb-cell integration helpers. |
| `fft_helpers.py` | Provides the canonical sharded real/reciprocal FFT factories. |
| `four_current_model.py` | Defines shared packed-current model vocabulary and validation. |
| `fourier_plan.py` | `LocalFourierPlan`: one local separable DFT with restricted per-axis supports. |
| `gamma_matrices.py` | Supplies spinor gamma-matrix conventions. |
| `gauss_legendre.py` | Generates Gauss-Legendre nodes and weights. |
| `gpu_utils.py` | Detects device memory and allocator state. |
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
| `sharding_fit.py` | Chooses layouts that fit device-memory constraints. |
| `spectral_closure.py` | Closes truncation cuts over spectral degeneracies. |
| `staged_reshard.py` | Implements staged collective reshards. |
| `timing.py` | Records scoped timing measurements. |
| `units.py` | Defines unit conversions. |
| `vma.py` | Reads process virtual-memory diagnostics. |
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
| `_slab_io_serial.py` | Implements serial SlabIO transport. |
| `centroids.py` | Reads and writes centroid files. |
| `commit_state.py` | Persistent completion receipt for collective artifact writes. |
| `epsreader.py` | Reads dielectric-matrix files. |
| `h5_journal.py` | Records bounded HDF5 operation journals. |
| `hdf5_owner.py` | Enforces one process owner for an HDF5 file. |
| `host_tile_store.py` | Tile-major pinned host store for a grid of sharded tiles (not wired). |
| `io_timing.py` | Optional per-rank SlabIO wall trace. |
| `isdf_header.py` | Reads and validates ISDF HDF5 headers. |
| `kin_ion.py` | Reads kinetic-plus-ionic matrices. |
| `mf_header.py` | Reads mean-field metadata headers. |
| `mpa_store.py` | Stores multipole-analysis intermediates. |
| `parallel_transport.py` | Reads and writes parallel-transport data. |
| `paths.py` | Resolves configured input and output paths. |
| `qe_save_reader.py` | Reads bounded Quantum ESPRESSO save-directory metadata. |
| `qp_wfn.py` | Reads and writes quasiparticle wavefunction data. |
| `read_bgw_vcoul.py` | Reads BerkeleyGW Coulomb data. |
| `restart_bundle.py` | Reads the GW restart bundle. |
| `shared_pole_store.py` | Shared real-pole model and construction-scratch I/O. |
| `sigma_output.py` | Writes self-energy and quasiparticle outputs. |
| `slab_io.py` | Exposes sharded slab reads and writes. |
| `static_gauge_head.py` | Reads and writes static-gauge head data. |
| `tagged_arrays.py` | Associates array payloads with metadata tags. |
| `wfn_basis.py` | Reads wavefunction basis metadata. |
| `wfn_writer.py` | Writes wavefunction files. |

## Service source packages

These rows inventory the package directories matched by `services/*/src/*/`;
the linked service pages own their caller contracts.

| Package directory | Role |
|---|---|
| `services/distrib_la/src/distrib_la/` | Distributed dense-linear-algebra service; see [contract](services/distrib_la.md). |
| `services/lxkit/src/lxkit/` | Standalone launch/runtime foundation shared by services. |
| `services/minimax/src/minimax/` | Runtime minimax quadrature solver service; see [contract](services/minimax.md). |
| `services/symmetry_maps/src/symmetry_maps/` | Crystal-symmetry mapping service; see [contract](services/symmetry_maps.md). |
| `services/vcoul/src/vcoul/` | Coulomb-kernel and cell-average service; see [contract](services/vcoul.md). |
| `services/wfn_loader/src/wfn_loader/` | Wavefunction-loading service; see [contract](services/wfn_loader.md). |
| `services/zeta_loader/src/zeta_loader/` | Zeta bundle loading service; see [contract](services/zeta_loader.md). |
