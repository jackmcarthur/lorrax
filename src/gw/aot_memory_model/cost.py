"""Driver-level cost model: total FLOPs across the whole r-chunk loop.

The memory model (``core.fit_nnls`` on ``PRIMITIVES``) predicts one
r-chunk's peak bytes.  That tells us which knob settings FIT within a
device budget.  But two settings that both fit are NOT equally cheap:

    total_cost(chunk_r, band_chunk) =
        num_r_chunks(chunk_r) · per_call_cost(chunk_r, band_chunk, sys, mesh)

where ``num_r_chunks = ceil(n_rtot / chunk_r)``.  The per-call cost has
a chunk-independent overhead (Cholesky solve, ZCT FFT over kgrid) that
reruns every r-chunk — so cutting ``chunk_r`` in half roughly doubles
total work.  On a 46080-pt MoS2 3×3 fit the AOT FLOPs numbers are:

    chunk_r=46080,  1 call × 7.3  GFlops =  7.3 GFlops
    chunk_r=20000,  3 calls × 4.6 GFlops = 13.8 GFlops
    chunk_r= 5000, 10 calls × 3.0 GFlops = 30.0 GFlops

So bigger chunk_r → fewer total FLOPs.  The chooser's job is to pick
the biggest (chunk_r, band_chunk) that still fits in the memory budget.

The cost-fit primitives mirror the memory ones but target the
``flops`` field in ``aot_measure``'s result.  A kernel opts in by
declaring ``FLOPS_PRIMITIVES: dict[str, Callable]`` alongside its
existing ``PRIMITIVES``.  Kernels that don't declare them raise at
fit time.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .core import (
    ARTIFACTS_DIR,
    AotKernel,
    Knobs,
    MeshSpec,
    SysDims,
    load_samples,
    samples_to_fit_input,
)


@dataclass
class CostFit:
    """FLOPs-target linear fit: flops = intercept + Σ β_i · T_i.

    Parallels :class:`core.Fit` but keeps cost and memory fits in
    separate artifact files so rebuilding one doesn't invalidate the
    other.
    """
    kernel: str
    feature_names: tuple[str, ...]
    coefs: tuple[float, ...]
    intercept: float
    residual_rms: float = 0.0
    n_samples: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _kernel_flops_primitives(kernel: AotKernel) -> dict:
    prims = getattr(kernel, "FLOPS_PRIMITIVES", None)
    if not prims:
        raise ValueError(
            f"Kernel {kernel.name!r} has no FLOPS_PRIMITIVES declared — "
            "add them alongside PRIMITIVES to opt into the cost model.")
    return prims


def fit_cost_nnls(
    kernel: AotKernel,
    samples: Sequence[tuple[SysDims, Knobs, MeshSpec, dict]],
    *,
    include_intercept: bool = True,
) -> CostFit:
    """NNLS fit of per-call FLOPs against ``kernel.FLOPS_PRIMITIVES``.

    ``samples`` is the ``(sys, knobs, mesh, meas_dict)`` tuple list —
    the 4th field is the FULL measurement dict so we can read
    ``meas['flops']``.  The mem-side helper ``samples_to_fit_input``
    collapses to peak_bytes int; cost fitting needs the broader dict,
    so callers should pass the raw list-of-dicts from
    ``load_samples()``.
    """
    from scipy.optimize import nnls
    feats = tuple(_kernel_flops_primitives(kernel).keys())
    prims = _kernel_flops_primitives(kernel)

    X_rows, y = [], []
    for sys, knobs, mesh, meas in samples:
        flops = meas.get("flops")
        if flops is None:
            continue  # aot_measure couldn't get FLOPs from cost_analysis
        row = [prims[f](sys, knobs, mesh) for f in feats]
        if include_intercept:
            row = [1.0] + row
        X_rows.append(row)
        y.append(float(flops))

    if not X_rows:
        raise ValueError(
            f"No samples with flops data for kernel {kernel.name!r}. "
            "Re-run the sweep with a version that captures cost_analysis.")
    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.shape[0] < X.shape[1]:
        raise ValueError(
            f"Underdetermined cost fit for {kernel.name!r}: "
            f"{X.shape[0]} samples, {X.shape[1]} features.")
    coefs, rnorm = nnls(X, y)
    if include_intercept:
        intercept = float(coefs[0])
        slopes = tuple(float(c) for c in coefs[1:])
    else:
        intercept = 0.0
        slopes = tuple(float(c) for c in coefs)
    return CostFit(
        kernel=kernel.name,
        feature_names=feats,
        coefs=slopes,
        intercept=intercept,
        residual_rms=float(rnorm / np.sqrt(max(1, len(y)))),
        n_samples=len(X_rows),
    )


def predict_flops_per_call(
    fit: CostFit, kernel: AotKernel,
    sys: SysDims, knobs: Knobs, mesh: MeshSpec,
) -> float:
    """Evaluate the fitted formula at ``(sys, knobs, mesh)``.  Returns
    FLOPs *per one jit call* — to get the full-fit cost, multiply by
    ``num_r_chunks(chunk_r)``."""
    prims = _kernel_flops_primitives(kernel)
    total = fit.intercept
    for name, coef in zip(fit.feature_names, fit.coefs):
        total += coef * prims[name](sys, knobs, mesh)
    return float(total)


# ---------------------------------------------------------------------------
# Persistence — separate artifact file per fit so mem/cost don't clobber
# ---------------------------------------------------------------------------

def save_cost_fit(fit: CostFit, tag: str = "current") -> Path:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = ARTIFACTS_DIR / f"{fit.kernel}__{tag}__cost_fit.json"
    path.write_text(json.dumps(fit.as_dict(), indent=2))
    return path


def load_cost_fit(kernel_name: str, tag: str = "current") -> CostFit:
    path = ARTIFACTS_DIR / f"{kernel_name}__{tag}__cost_fit.json"
    raw = json.loads(path.read_text())
    raw["feature_names"] = tuple(raw["feature_names"])
    raw["coefs"] = tuple(raw["coefs"])
    return CostFit(**raw)


def fit_cost_from_saved(
    kernel_name: str, tag: str = "current", *, save: bool = True,
) -> CostFit:
    """Rank-0 helper: load the same samples.json the mem-fit uses,
    NNLS-fit against FLOPS_PRIMITIVES, write a ``*_cost_fit.json``
    artifact next to the memory fit."""
    from .core import get_kernel
    kernel = get_kernel(kernel_name)
    # samples.json entries carry meas.flops already — reconstruct the
    # (sys, knobs, mesh, meas) tuple list without dropping fields.
    raw = load_samples(kernel_name, tag=tag)
    tuples = []
    for s in raw:
        sys_kw = dict(s["sys"])
        sys_kw["kgrid"] = tuple(sys_kw["kgrid"])
        if sys_kw.get("fft_grid") is not None:
            sys_kw["fft_grid"] = tuple(sys_kw["fft_grid"])
        sys_kw.pop("n_k", None)
        sys = SysDims(**sys_kw)
        knobs = Knobs.of(**s["knobs"])
        mesh = MeshSpec(**s["mesh"])
        tuples.append((sys, knobs, mesh, s["meas"]))
    fit = fit_cost_nnls(kernel, tuples)
    if save:
        save_cost_fit(fit, tag=tag)
    return fit
