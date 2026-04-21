"""AOT memory model for LORRAX heavy jits.

Per-kernel empirical memory rules derived from
``jax.jit(f).lower(specs).compile().memory_analysis()`` sweeps.

Offline calibration (``sweep`` -> ``fit``) produces slope/intercept
multivariate linear fits that predict peak device bytes from
``(sys_dims, mesh shape, chunk knobs)``.  Online ``predict`` /
``choose_knobs`` evaluate the fitted formula against a budget.

See ``docs/AOT_MEMORY_MODEL.md`` for the architecture writeup and
``lowering_ahead_of_time_jax.md`` at the sandbox root for the JAX AOT
pipeline reference.
"""
from .core import (
    AotKernel,
    MeshSpec,
    SysDims,
    aot_measure,
    fit_nnls,
    load_fit,
    predict_peak,
    save_fit,
)
from .doe import build_doe_axes

__all__ = [
    "AotKernel",
    "MeshSpec",
    "SysDims",
    "aot_measure",
    "build_doe_axes",
    "fit_nnls",
    "load_fit",
    "predict_peak",
    "save_fit",
]
