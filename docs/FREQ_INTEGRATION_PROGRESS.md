# Frequency Integration Rewrite Progress

Status: Stages 1-3 completed for chi path
Owner: Codex
Related plan: `docs/FREQ_INTEGRATION_REWRITE_PLAN.md`

## Scope
Track implementation progress for the separate frequency-integration pipeline under `src/gw_isdf/freqint/` while keeping legacy dynamic integration available for parity checks.

## Completed Stages

### Stage 1: Package and Data Model
Delivered:
- New package: `src/gw_isdf/freqint/`
- Core types in `types.py`:
  - `PoleBlock`, `DenomType`, `WindowExecutionPlan`, `IntegrationPlan`
- Bundle slicing helpers in `slicing.py`
- Public API surface in `api.py` and package exports in `__init__.py`

### Stage 2: Native Branch Execution (chi)
Delivered:
- Replaced stage-1 legacy scalar dispatch in `freqint.engine` with branch-native execution from planned denominator blocks.
- Added denominator dispatch for:
  - `antiresonant` (GL)
  - `resonant_below` (GL)
  - `crossing` (HGL)
  - `resonant_above` (GL reverse-edge)
- Preserved negative-frequency handling via post-accumulation conjugation.

Files:
- `src/gw_isdf/freqint/engine.py`
- `src/gw_isdf/freqint/integrands.py`

### Stage 3: Batched Omega + Tau Scan + Shared Policy
Delivered:
- Added batched denominator integration kernels using `jax.lax.scan` over tau:
  - `compute_gl_standard_batch(...)`
  - `compute_gl_reverse_batch(...)`
  - `compute_hgl_batch(...)`
- Added non-`per_omega` planning policy:
  - `shared_conservative`
  - one denominator bucket per branch per window, one shared quadrature per bucket
- Kept `per_omega` policy for exact legacy-like branch quadratures.
- Set `shared_conservative` as default in `chi_from_bundle(...)` and unified `frequency_integration(...)` API.

Files:
- `src/gw_isdf/freqint/planning.py`
- `src/gw_isdf/freqint/integrands.py`
- `src/gw_isdf/freqint/engine.py`
- `src/gw_isdf/freqint/api.py`

## Validation Completed
- Added parity tests:
  - `tests/archive/test_freqint_stage23.py`
- Results:
  - `per_omega` policy: matches legacy scalar integration to tight tolerance.
  - `shared_conservative` policy: matches within expected conservative-quadrature tolerance.
- Command run:
  - `uv run -- python -m pytest -q tests/archive/test_freqint_stage23.py`

## Current Behavior
- `chi` mode is functional through the new pipeline.
- `sigma` mode remains a placeholder (`NotImplementedError`).
- Legacy path remains intact for cross-validation and fallback.

## Remaining Work
- Implement sigma channels on the same engine path:
  - `(valence wfns, plasmons +)`
  - `(conduction wfns, plasmons -)`
- Add production benchmarks and memory/performance profiling on target hardware.
- Add broader regression coverage outside toy systems and single-window cases.
