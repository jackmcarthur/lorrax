# TODOs for `cohsex_jax.py`

1. Distribute the q-point Coulomb metadata across the device mesh (per-device chunks via `device_put_sharded`) instead of the current replicated placement to enable true multi-host scaling.
2. Fold the q iteration into a `lax.scan`/`pjit` that donates CCT/ZCT work buffers and emits V/S slices, eliminating per-q recompiles and host/device churn.
3. Reuse pooled workspaces for `compute_CCT_ZCT_for_q` and the zeta Cholesky solve to avoid reallocation inside the q-loop hot path.
4. Rework wavefunction preparation to share backing storage for left/right shards and stream k-batches instead of materializing the full FFT boxes per side.
5. Stream V/S accumulation by flushing q-block slices to the restart HDF5 and rehydrating them on demand to keep device memory bounded during the loop.
6. Integrate q-block donation for `psi_*` views so sharding constraints don’t trigger extra copies each iteration.
7. Make platform/backend selection configurable instead of forcing CPU via `JAX_PLATFORM_NAME`/`JAX_PLATFORMS`.
8. Profile the q-loop: identify hotspots in `compute_CCT_ZCT_for_q` / FFT path, consider JAX `vmap` and `pmap` splits.
9. Factor out repeated sharding with builders keyed by array rank (eg. `mk_shard(spec)`), limit direct `NamedSharding` usage.
10. Replace numpy fallbacks in q-loop (e.g. `np.asarray`) with JAX-friendly logic; guard with host callbacks only if unavoidable.
11. Revisit `mesh_xy` lifetime: unify creation/resuse across fresh/restart, hide within a context manager.
12. Ensure restart paths can resume with partial q-loop results (checkpointing).
13. Implement `pytest`-style unit coverage for q-loop and V/W head injection.

# TODOs for `w_isdf.py`

1. Hoist `get_chi0_jax` helper kernels out of nested scopes and add `donate_argnums` so large chi buffers compile once and reuse memory.
2. Precompute CTSP energy masks and exponential factors per window and pass them into the chi kernel to avoid redundant `exp` evaluation.
3. Introduce rμ/rν tiling for the G-build and FFT legs so `chi0` runs in bounded memory aligned with device sharding.
4. Stream chi0 q-block slices (aligned with V/S streaming) to disk or a generator so later stages can consume without holding the full tensor.
5. Profile `get_chi0_jax`: identify expensive contractions and consider chunking / caching.
6. Consolidate repeated FFT logic (wrap in helper module).
7. Audit memory layout for `chi0` tensors; use `jax.device_put_sharded` where beneficial.
8. Convert imperative loops to `jax.lax.scan`/`vmap` where possible.
9. Factor Coulomb-related helpers into `vcoul.py`, removing duplicates.
10. Implement lazy evaluation / caching for window data (currently recomputed per call).
11. Add unit tests comparing `w_isdf` outputs against reference HDF5 data.
12. Apply `pjit` for major pipeline functions to exploit multi-host setups.
13. Provide deterministic RNG seeding for Monte Carlo parts.
14. Add GPU-specific kernels or ensure compatibility with TPU/accelerator devices.
15. Replace `np` interplay with pure `jax.numpy` (avoid host-device sync).

# Larger Structural Objectives

1. Build a hashed preprocessing artifact cache (dipoles, centroids, kin+ion) so `cohsex_jax` reuses validated inputs on restart or new runs.
2. Introduce a resumable pipeline scheduler that stages zeta, Coulomb, chi, W, and sigma as checkpointable steps.
3. Create a distributed q-block job manager to partition the q-grid across processes/device meshes for better scaling.
4. Split `cohsex_jax.py` into modules: input handling, q-loop builder, sigma pipeline, restart IO.
5. Create `isdf/common/mesh.py` utilities for device mesh creation and sharding namespaces.
6. Add a configuration / CLI layer using `pydantic` or similar for validation and serialization of run parameters.
7. Develop a comprehensive logging subsystem (structured logs, log levels, optional progress bars).
8. Introduce a benchmarking/profiling suite using `jax.profiler` and `tensorboard` integrations.
9. Build automated test harness with real input fixtures and CI integration (GitHub Actions).
10. Implement dependency injection / service layer for IO (WFN loading, checkpointing) to ease mocking/testing.
11. Move shared FFT/linear-algebra helpers into `isdf/common/linalg.py` with JAX-friendly abstractions.
12. Add performance telemetry collection (timers, memory stats) and persistent reporting.
13. Establish coding standards (formatter, linter config, mypy) and enforce via pre-commit hooks.
14. Build a data validation layer for HDF5 inputs (schema definitions, sanity checks).
15. Support asynchronous checkpointing / resumption for long runs (possibly with Orbax).
16. Create high-level driver scripts that orchestrate full workflows (DFT -> COHSEX -> post-processing).
17. Add reproducible container builds (Docker/Nix) with GPU support and CI tests.
18. Design a `Meta` management module that persists metadata to disk for restarts.
19. Introduce a robust error handling strategy with custom exception hierarchy.
