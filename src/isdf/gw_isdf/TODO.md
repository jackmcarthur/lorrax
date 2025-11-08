# TODOs for `cohsex_jax.py`

1. Profile the q-loop: identify hotspots in `compute_CCT_ZCT_for_q` / FFT path, consider JAX `vmap` and `pmap` splits.
2. Factor out repeated sharding with builders keyed by array rank (eg. `mk_shard(spec)`), limit direct `NamedSharding` usage.
3. Replace numpy fallbacks in q-loop (e.g. `np.asarray`) with JAX-friendly logic; guard with host callbacks only if unavoidable.
4. Introduce typed dataclasses for q-loop payloads instead of `SimpleNamespace`.
5. Revisit `mesh_xy` lifetime: unify creation/resuse across fresh/restart, hide within a context manager.
6. Audit `print` usage; replace with structured logging or progress callbacks.
7. Add comprehensive docstrings, including shapes/dtypes, for the major helpers (`iter_qpoint_data`, `determine_wcoul0`).
8. Convert repeated `int(meta.*)` casts to cached ints in `Meta`.
9. Implement `pytest`-style unit coverage for q-loop and V/W head injection.
10. Introduce configuration-driven toggles for diagnostics (timers, sharding prints) instead of inline `print`.
11. Move repeated band-range logic into `Meta` (e.g., add `sigma_slice` property).
12. Replace manual `einsum` and reshape patterns with `jax.numpy.tensordot` where faster / more readable.
13. Provide optional JAX profiler hooks (e.g., `jax.profiler.start_trace`) around pipeline sections.
14. Cache FFT grids (`fx/fy/fz`) per q-grid to avoid recompute per iteration.
15. Inspect `compute_v_munu_from_zeta`: consider `jax.vmap` for `vcoul_comps` slicing and pre-jitted caches keyed by shapes.
16. Ensure restart paths can resume with partial q-loop results (checkpointing).
17. Introduce constant container for global parameters (volume, nk) to avoid repeated `float(...)` conversions.
18. Adopt `argparse` subcommands for debug vs prod pipeline to reduce branching inside `main`.
19. Replace manual `save_restart_per_proc`/`write_labeled_arrays_to_h5` with context-managed checkpoint API.
20. Ensure all JITs include explicit `donate_argnums` for large buffers to reduce memory churn.
21. Review `psi` sharding: prefer `pjit`/`PartitionSpec` usage instead of manual `with_sharding_constraint`.
22. Update `compute_sigma_pipeline_jax` to accept structured inputs (dataclasses) to reduce positional args.
23. Evaluate use of `jax.experimental.mesh_utils` for mesh creation compatibility.
24. Add typed configuration object for parameters (input parsing -> dataclass) to avoid global dict access.
25. Provide end-to-end integration tests with golden eqp outputs to guard refactors.

# TODOs for `w_isdf.py`

1. Profile `get_chi0_jax`: identify expensive contractions and consider chunking / caching.
2. Consolidate repeated FFT logic (wrap in helper module).
3. Audit memory layout for `chi0` tensors; use `jax.device_put_sharded` where beneficial.
4. Add docstrings and shape/type annotations for exported functions.
5. Convert imperative loops to `jax.lax.scan`/`vmap` where possible.
6. Factor Coulomb-related helpers into `vcoul.py`, removing duplicates.
7. Implement lazy evaluation / caching for window data (currently recomputed per call).
8. Introduce structured logging for convergence metrics.
9. Add unit tests comparing `w_isdf` outputs against reference HDF5 data.
10. Replace in-function constants with module-level configuration (e.g., tolerance values).
11. Apply `pjit` for major pipeline functions to exploit multi-host setups.
12. Provide deterministic RNG seeding for Monte Carlo parts.
13. Add GPU-specific kernels or ensure compatibility with TPU/accelerator devices.
14. Replace `np` interplay with pure `jax.numpy` (avoid host-device sync).
15. Document expected memory footprint and provide utilities to estimate resource usage.

# Larger Structural Objectives

1. Split `cohsex_jax.py` into modules: input handling, q-loop builder, sigma pipeline, restart IO.
2. Create `isdf/common/mesh.py` utilities for device mesh creation and sharding namespaces.
3. Add a configuration / CLI layer using `pydantic` or similar for validation and serialization of run parameters.
4. Develop a comprehensive logging subsystem (structured logs, log levels, optional progress bars).
5. Introduce a benchmarking/profiling suite using `jax.profiler` and `tensorboard` integrations.
6. Build automated test harness with real input fixtures and CI integration (GitHub Actions).
7. Create documentation site (Sphinx/MkDocs) with developer guides and API reference.
8. Implement dependency injection / service layer for IO (WFN loading, checkpointing) to ease mocking/testing.
9. Move shared FFT/linear-algebra helpers into `isdf/common/linalg.py` with JAX-friendly abstractions.
10. Provide a plugin interface for head correction methods (epshead, S tensor, custom) to extend easily.
11. Add performance telemetry collection (timers, memory stats) and persistent reporting.
12. Establish coding standards (formatter, linter config, mypy) and enforce via pre-commit hooks.
13. Build a data validation layer for HDF5 inputs (schema definitions, sanity checks).
14. Support asynchronous checkpointing / resumption for long runs (possibly with orbax).
15. Create high-level driver scripts that orchestrate full workflows (DFT -> COHSEX -> post-processing).
16. Integrate with external visualization/reporting tools for sigma/eqp results.
17. Add reproducible container builds (Docker/Nix) with GPU support and CI tests.
18. Design a `Meta` management module that persists metadata to disk for restarts.
19. Provide declarative pipeline descriptions (YAML/JSON) to configure computation graph without modifying code.
20. Introduce robust error handling strategy with custom exception hierarchy.
