"""Interpolation utilities (e.g., bandstructure k-path transforms)."""

# OpenBLAS spin tuning BEFORE numpy can be mapped: importing ``runtime`` sets
# OPENBLAS_THREAD_TIMEOUT as its first statement (runtime.blas_env), and
# OpenBLAS reads it once, when the library maps.  A harness whose first
# LORRAX import is ``bandstructure.<anything>`` is therefore tuned whatever it imports
# next.  ``runtime`` is jax-free at import; held by tests/test_runtime_blas_env.py.
import runtime  # noqa: F401,E402
