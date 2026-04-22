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
    Knobs,
    aot_measure,
    fit_nnls,
    get_kernel,
    load_fit,
    predict_peak,
    save_fit,
)
from .cost import (
    CostFit,
    fit_cost_from_saved,
    fit_cost_nnls,
    load_cost_fit,
    predict_flops_per_call,
    save_cost_fit,
)
from .chooser import (
    AlphaFit,
    ChunkChoice,
    choose_chunks_analytic,
    choose_chunks_aot,
    choose_chunks_heuristic,
    describe_chunks,
)
from .doe import build_doe_axes

__all__ = [
    "AlphaFit",
    "AotKernel",
    "ChunkChoice",
    "CostFit",
    "Knobs",
    "MeshSpec",
    "SysDims",
    "aot_measure",
    "build_doe_axes",
    "choose_chunks_analytic",
    "choose_chunks_aot",
    "choose_chunks_heuristic",
    "describe_chunks",
    "fit_cost_from_saved",
    "fit_cost_nnls",
    "fit_nnls",
    "get_kernel",
    "load_cost_fit",
    "load_fit",
    "predict_flops_per_call",
    "predict_kernel_peak",
    "predict_peak",
    "save_cost_fit",
    "save_fit",
]


# ---------------------------------------------------------------------------
# Convenience — one-call prediction for gw_init.compute_optimal_chunks wiring.
# ---------------------------------------------------------------------------

_PREDICT_CACHE: dict = {}


def predict_kernel_peak(
    kernel_name: str,
    sys: SysDims,
    knobs: Knobs | None = None,
    mesh: MeshSpec | None = None,
    *,
    tag: str = "current",
) -> float:
    """Predict peak device bytes for a kernel under the given config.

    One-call wrapper around ``load_fit`` + ``predict_peak`` suitable for
    plumbing into ``gw_init.compute_optimal_chunks``.  Loads the fit
    artifact at ``src/gw/aot_memory_model/artifacts/<kernel>__<tag>__fit.json``
    (tag defaults to ``current`` — the post-reshard-audit fits) and
    evaluates the fitted formula on the (sys, knobs, mesh) point.

    ``kernel``, ``fit``, and the cached combined closure are memoised
    by ``(kernel_name, tag)`` so repeat calls are pure numpy dot-products.
    """
    knobs = knobs or Knobs.of()
    if mesh is None:
        raise ValueError("mesh (MeshSpec) is required")
    key = (kernel_name, tag)
    entry = _PREDICT_CACHE.get(key)
    if entry is None:
        kernel = get_kernel(kernel_name)
        fit = load_fit(kernel_name, tag=tag)
        entry = (kernel, fit)
        _PREDICT_CACHE[key] = entry
    kernel, fit = entry
    return predict_peak(fit, kernel, sys, knobs, mesh)
