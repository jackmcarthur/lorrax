# Active TODOs for `gw_jax.py`

1. Refactor restart handling into explicit stages (load tensors, load chi0 wavefunctions, resume sigma) with a single state object.
1. Support partial restart checkpoints for long runs (stage-level resume, not per-q legacy resume).
1. Add focused tests for restart integrity (`V_qmunu`, `V0_noG0_munu`, `G0_mu_nu`) and head-injection correctness.
1. Add timing/memory summaries per stage (`zeta_fit`, `V_q_compute`, `chi0_W`, `sigma`) to simplify regression triage.
1. Keep `gw_jax.py` as orchestrator and split heavy helper logic into dedicated modules once interfaces stabilize.

# Active TODOs for `w_isdf.py`

1. Reduce redundant contractions in chi construction by operating on window-local blocks where possible.
1. Profile `get_chi0_jax` communication and verify sharding choices for large `rmu/rnu` workloads.
1. Consolidate repeated FFT/convolution helpers shared with sigma pipeline code paths.
1. Add regression tests against saved HDF5 references for `chi0`, `W_q`, and dynamic-omega paths.

# Broader Structural Objectives

1. Build a cached preprocessing artifact layer (dipoles, centroids, kin+ion) keyed by validated input hashes.
1. Expand automated regression coverage around the compact `cohsex_debug` fixture.
1. Improve `.out` and `.log` outputs with explicit per-band contribution columns.
1. Add a profiling workflow (`jax.profiler`/TensorBoard or equivalent) for repeatable performance checks.
1. Evaluate a `gw_isdf.mesh` utility module to centralize mesh/sharding construction patterns.
