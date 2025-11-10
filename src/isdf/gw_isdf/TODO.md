# TODOs for `cohsex_jax.py`


1. Make the entire q iteration loop for interp vectors $\zeta_{q,\mu}(r)$ into a `lax.scan`/`pjit` that donates CCT/ZCT work buffers and emits V/S slices, eliminating per-q recompiles and host/device churn.
1. Reuse pooled workspaces for `compute_CCT_ZCT_for_q` and the zeta Cholesky solve to avoid reallocation inside the q-loop hot path.
1. Rework wavefunction preparation to share backing storage for left/right shards and stream k-batches instead of materializing the full FFT boxes per side. JAX allows uneven sharding inside function loops but not for function returns; we can take advantage of this.
1. Profile the q-loop: identify hotspots in `compute_CCT_ZCT_for_q` / FFT path, consider JAX `vmap` and `pmap` splits.
1. Replace numpy fallbacks in q-loop (e.g. `np.asarray`) with JAX-friendly logic; guard with host callbacks only if unavoidable.
1. Distribute the q-point Coulomb metadata across the device mesh (per-device chunks via `device_put_sharded`) instead of the current replicated placement to enable true multi-host scaling.
1. Integrate q-block donation for `psi_*` views so sharding constraints don’t trigger extra copies each iteration.
1. Make platform/backend selection configurable instead of forcing CPU via `JAX_PLATFORM_NAME`/`JAX_PLATFORMS`.
1. Factor out repeated sharding with builders keyed by array rank (eg. `mk_shard(spec)`), limit direct `NamedSharding` usage.
1. Revisit `mesh_xy` lifetime: unify creation/resuse across fresh/restart, hide within a context manager.
1. Ensure restart paths can resume with partial q-loop results (checkpointing).
1. Implement `pytest`-style unit coverage for q-loop and V/W head injection.

# TODOs for `w_isdf.py`

1. Gv and Gc Green's functions are computed by masking the (large) band index into windows and contracting along the entire now-block-sparse band index. This would be much faster if we designed a JITtable function that can multiply only the relevant blocks (which are predetermined by get_windows)
1. Chi construction should be absolutely as JITted as possible, we should remove `get_chi0_jax` helper kernels from nested scopes and add `donate_argnums` so large chi buffers compile once and reuse memory.
1. Introduce rμ/rν tiling for the G-build and FFT legs so `chi0` runs in bounded memory aligned with device sharding. (I already intended for this to be the case because the wavefunctions are sharded to make it easy; please confirm by profiling XLA comms)
1. Profile `get_chi0_jax`: identify expensive contractions and consider chunking / caching.
1. Consolidate repeated FFT logic (wrap in helper module). Same operations are used in the later sigma pipeline.
1. Audit memory layout for `chi0` tensors; use `jax.device_put_sharded` where beneficial.
1. Convert imperative loops to `jax.lax.scan`/`vmap` where possible.
1. Factor Coulomb-related helpers into `vcoul.py`, removing duplicates.
1. Implement lazy evaluation / caching for window data (currently recomputed per call).
1. Add unit tests comparing `w_isdf` outputs against reference HDF5 data.
1. Apply `pjit` for major pipeline functions to exploit multi-host setups.
1. Provide deterministic RNG seeding for Monte Carlo parts.
1. Add GPU-specific kernels or ensure compatibility with TPU/accelerator devices.
1. Replace `np` interplay with pure `jax.numpy` (avoid host-device sync).

# Larger Structural Objectives

1. Build a hashed preprocessing artifact cache (dipoles, centroids, kin+ion) so `cohsex_jax` reuses validated inputs on restart or new runs, combined in functionality with restart arrays (taggedarrays currently)
1. Automatic tests (pytest or similar). A smaller version of the cohsex_debug module's inputs should be stashed somewhere internal to check that
1. .out outputs and .log outputs. .out should probably have whatever QP energy contributions are present added together and .log should have columns for kin-ion, hartree, Sigma^SEX, Sigma^COH, etc. energies per band per kpoint
1. Introduce a benchmarking/profiling suite using `jax.profiler` and `tensorboard` integrations. Other methods to check are fine if generally more effective
1. Create `isdf/common/mesh.py` utilities for device mesh creation and sharding namespaces. Mesh scope should be essentially global to the whole COHSEX program.
1. More sophisticated timing procedures and printed output for the key functions 
1. Introduce a resumable pipeline scheduler that stages zeta, Coulomb, chi, W, and sigma as checkpointable steps.
1. Split `cohsex_jax.py` into modules: input handling, q-loop builder, sigma pipeline, restart IO.
1. Add a configuration / CLI layer using `pydantic` or similar for validation and serialization of run parameters.
1. Develop a comprehensive logging subsystem (structured logs, log levels, optional progress bars).
1. Build automated test harness with real input fixtures and CI integration (GitHub Actions).
1. Implement dependency injection / service layer for IO (WFN loading, checkpointing) to ease mocking/testing.
1. Move shared FFT/linear-algebra helpers into `isdf/common/linalg.py` with JAX-friendly abstractions.
1. Add performance telemetry collection (timers, memory stats) and persistent reporting.
1. Establish coding standards (formatter, linter config, mypy) and enforce via pre-commit hooks.
1. Design a `Meta` management module that persists metadata to disk for restarts.
1. Introduce a robust error handling strategy with custom exception hierarchy.
