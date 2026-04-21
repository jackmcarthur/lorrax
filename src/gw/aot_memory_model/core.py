"""AotKernel registry + AOT measurement + linear fit.

A kernel subclass declares:
  * ``SYSTEM_DIMS`` — names of system dimensions it scales with
  * ``KNOBS`` — tunable chunk knobs (may be empty)
  * ``PRIMITIVES`` — dimensional volume features f(sys, knobs, mesh) -> bytes
  * ``build_specs(sys, knobs, mesh)`` — per-input ShapeDtypeStruct + Sharding
  * ``build_callable(sys, knobs, mesh)`` — the jittable to AOT-lower

``aot_measure`` runs the lower+compile pipeline and returns a dict of
``memory_analysis`` fields.  ``fit_nnls`` NNLS-fits the peak against the
kernel's declared primitives, and ``predict_peak`` evaluates the fit on a
new ``(sys, knobs, mesh)`` point.

memory_analysis() is a compiler estimate, not allocator truth: it excludes
cuFFT workspace, NCCL staging, and allocator fragmentation.  Each kernel
may carry a scalar ``gamma`` correction = actual_runtime_peak / aot_peak
calibrated against one production-scale point, applied at prediction time.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


# ============================================================================
# System / mesh / knob specs
# ============================================================================

@dataclass(frozen=True)
class SysDims:
    """Physical system dimensions driving array shapes.

    ``n_k`` is auto-derived as ``prod(kgrid)`` so it always matches the
    kernel's FFT reshape.  To vary k-count in a DoE sweep, vary ``kgrid``
    (e.g. ``(2,2,2) -> (4,4,4)``) — ``n_k`` tracks automatically.

    Not every kernel uses every field.  Each kernel's ``SYSTEM_DIMS`` tuple
    lists which fields it actually consumes, so the DoE sweeps only those.
    """
    kgrid: tuple[int, int, int]
    n_rmu: int         # ISDF centroids (μ axis)
    n_s: int = 1       # spin channels (1 or 2)
    nspinor: int = 1   # spinor dimension (1 or 2)
    n_b_v: int = 0     # valence bands in chi0 pair
    n_b_c: int = 0     # conduction bands in chi0 pair
    n_b: int = 0       # generic band count (pair-density / sigma uses)
    n_r: int = 0       # real-space FFT grid nx*ny*nz (V_q, FFT kernels)
    fft_grid: tuple[int, int, int] | None = None
    # Real-space FFT grid dims — distinct from kgrid when a kernel
    # needs BOTH the k-grid (for nq = ∏k) and the FFT box (for the
    # IFFT reshape).  Falls back to kgrid when unset (back-compat with
    # kernels that repurpose kgrid as the FFT box).

    @property
    def n_k(self) -> int:
        kx, ky, kz = self.kgrid
        return int(kx) * int(ky) * int(kz)

    def as_dict(self) -> dict:
        d = {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in asdict(self).items()}
        d["n_k"] = self.n_k
        return d

    @property
    def fft_shape(self) -> tuple[int, int, int]:
        """Real-space FFT box.  Falls back to kgrid when fft_grid is None."""
        return tuple(self.fft_grid) if self.fft_grid is not None else tuple(self.kgrid)


@dataclass(frozen=True)
class MeshSpec:
    """Device mesh shape (p_x, p_y).  Total devices = p_x * p_y."""
    p_x: int
    p_y: int

    @property
    def n_dev(self) -> int:
        return self.p_x * self.p_y

    def as_dict(self) -> dict:
        return {"p_x": self.p_x, "p_y": self.p_y}


@dataclass(frozen=True)
class Knobs:
    """Free-form knob bag. Keys are kernel-specific string names.

    Stored as a frozen mapping so it hashes cleanly for caching.  Access
    via ``knobs.get(name, default)`` or ``knobs[name]``.
    """
    values: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    def get(self, key: str, default=None):
        for k, v in self.values:
            if k == key:
                return v
        return default

    def __getitem__(self, key):
        v = self.get(key, None)
        if v is None:
            raise KeyError(key)
        return v

    def as_dict(self) -> dict:
        return dict(self.values)

    @classmethod
    def of(cls, **kwargs) -> "Knobs":
        return cls(tuple(sorted(kwargs.items())))


# ============================================================================
# AotKernel base class
# ============================================================================

class AotKernel:
    """Base class — concrete kernels override the four class-level fields
    and the two ``build_*`` methods.

    Subclasses live under ``aot_memory_model/kernels/`` and register
    themselves via ``KERNEL_REGISTRY`` at import time.
    """

    name: str = ""
    SYSTEM_DIMS: tuple[str, ...] = ()
    KNOBS: tuple[str, ...] = ()
    PRIMITIVES: dict[str, Callable] = {}
    # PRIMITIVES: name -> f(sys: SysDims, knobs: Knobs, mesh: MeshSpec) -> bytes

    def build_specs(self, sys: SysDims, knobs: Knobs, mesh) -> tuple:
        raise NotImplementedError

    def build_callable(self, sys: SysDims, knobs: Knobs, mesh) -> Callable:
        raise NotImplementedError


KERNEL_REGISTRY: dict[str, type] = {}


def register_kernel(cls):
    """Decorator: register a concrete AotKernel subclass by ``cls.name``."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"Kernel class {cls.__name__} missing ``name``.")
    KERNEL_REGISTRY[cls.name] = cls
    return cls


def get_kernel(name: str) -> AotKernel:
    # Lazy-import the kernels package so clients don't pay it at tool import.
    from . import kernels as _  # noqa: F401
    cls = KERNEL_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown kernel {name!r}. Known: {sorted(KERNEL_REGISTRY)}")
    return cls()


# ============================================================================
# AOT measurement
# ============================================================================

def aot_measure(
    kernel: AotKernel, sys: SysDims, knobs: Knobs, mesh,
    *, disable_remat: bool = True,
) -> dict:
    """Lower + compile the kernel, return a memory_analysis summary.

    ``mesh`` must be an actual ``jax.sharding.Mesh`` (for shardings).
    ``MeshSpec`` is carried alongside but the compiler takes the real mesh.

    When ``disable_remat`` is True (default), passes
    ``compiler_options={'xla_disable_hlo_passes': 'rematerialization'}``
    so XLA cannot rewrite the allocation plan to fit memory — staying in
    the pure scaling-law regime.  If memory_analysis shows a peak that
    would not have fit at runtime, that is informative (we'd pick a
    bigger mesh or smaller knob), not an error.

    Returns a dict with ``temp``, ``argument``, ``output``, ``alias``,
    ``total``, ``flops``, ``t_lower_s``, ``t_compile_s``.
    """
    import jax
    fn = kernel.build_callable(sys, knobs, mesh)
    specs = kernel.build_specs(sys, knobs, mesh)
    # build_callable returns an already-jitted function (e.g. with
    # donate_argnums set).  Do NOT double-jit: that strips the donation
    # and inflates alias_size → fit mistakes the "un-aliased output"
    # (duplicate of the donated input) as extra temp bytes.
    t0 = time.perf_counter()
    lowered = fn.lower(*specs)
    t1 = time.perf_counter()
    if disable_remat:
        compiled = lowered.compile(compiler_options={
            "xla_gpu_memory_limit_slop_factor": 10000,
        })
    else:
        compiled = lowered.compile()
    t2 = time.perf_counter()
    m = compiled.memory_analysis()
    c = compiled.cost_analysis()
    total = (
        m.temp_size_in_bytes
        + m.argument_size_in_bytes
        + m.output_size_in_bytes
        - m.alias_size_in_bytes
    )
    return {
        "temp": int(m.temp_size_in_bytes),
        "argument": int(m.argument_size_in_bytes),
        "output": int(m.output_size_in_bytes),
        "alias": int(m.alias_size_in_bytes),
        "total": int(total),
        "flops": (float(c["flops"]) if isinstance(c, dict) and "flops" in c else None),
        "t_lower_s": t1 - t0,
        "t_compile_s": t2 - t1,
    }


# ============================================================================
# NNLS fit + predict
# ============================================================================

@dataclass
class Fit:
    """Fit artifact — the slope/intercept primitive coefficients."""
    kernel: str
    feature_names: tuple[str, ...]
    coefs: tuple[float, ...]        # slope per primitive (non-negative)
    intercept: float                # constant overhead bytes
    gamma: float = 1.0              # runtime / AOT correction (kernel-specific)
    residual_rms: float = 0.0       # fit residual RMS bytes
    n_samples: int = 0
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def fit_nnls(
    kernel: AotKernel,
    samples: Sequence[tuple[SysDims, Knobs, MeshSpec, int]],
    include_intercept: bool = True,
) -> Fit:
    """NNLS fit peak_bytes = intercept + Σ_i β_i · T_i(sys, knobs, mesh).

    The intercept is also non-negative (captures cuFFT plan cache +
    JIT metadata + small replicated tails).  Any ``β_i < 0`` would indicate
    a missed or wrongly-signed primitive — NNLS clips to 0 and inflates
    residuals instead, which is easier to diagnose than a free-sign fit.
    """
    from scipy.optimize import nnls
    feats = tuple(kernel.PRIMITIVES.keys())
    if not feats:
        raise ValueError(f"Kernel {kernel.name!r} has no PRIMITIVES declared.")
    X_rows, y = [], []
    for sys, knobs, mesh, peak in samples:
        row = [kernel.PRIMITIVES[f](sys, knobs, mesh) for f in feats]
        if include_intercept:
            row = [1.0] + row
        X_rows.append(row)
        y.append(float(peak))
    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.shape[0] < X.shape[1]:
        raise ValueError(
            f"Underdetermined fit for kernel {kernel.name!r}: "
            f"{X.shape[0]} samples, {X.shape[1]} features.")
    coefs, rnorm = nnls(X, y)
    if include_intercept:
        intercept = float(coefs[0])
        slopes = tuple(float(c) for c in coefs[1:])
    else:
        intercept = 0.0
        slopes = tuple(float(c) for c in coefs)
    return Fit(
        kernel=kernel.name,
        feature_names=feats,
        coefs=slopes,
        intercept=intercept,
        residual_rms=float(rnorm / np.sqrt(max(1, len(y)))),
        n_samples=len(samples),
    )


def predict_peak(
    fit: Fit, kernel: AotKernel, sys: SysDims, knobs: Knobs, mesh: MeshSpec,
) -> float:
    """Evaluate the fitted formula at a new (sys, knobs, mesh) point.

    Applies the kernel's ``gamma`` correction (runtime/AOT discrepancy).
    Returns bytes.
    """
    total = fit.intercept
    for name, coef in zip(fit.feature_names, fit.coefs):
        total += coef * kernel.PRIMITIVES[name](sys, knobs, mesh)
    return float(total * fit.gamma)


# ============================================================================
# Persistence — JSON samples and fits under artifacts/
# ============================================================================

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def _sample_to_dict(sys: SysDims, knobs: Knobs, mesh: MeshSpec, meas: dict) -> dict:
    return {
        "sys": sys.as_dict(),
        "knobs": knobs.as_dict(),
        "mesh": mesh.as_dict(),
        "meas": meas,
    }


def save_samples(kernel_name: str, samples: list[dict], tag: str = "default") -> Path:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = ARTIFACTS_DIR / f"{kernel_name}__{tag}__samples.json"
    path.write_text(json.dumps(samples, indent=2))
    return path


def load_samples(kernel_name: str, tag: str = "default") -> list[dict]:
    path = ARTIFACTS_DIR / f"{kernel_name}__{tag}__samples.json"
    return json.loads(path.read_text())


def save_fit(fit: Fit, tag: str = "default") -> Path:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = ARTIFACTS_DIR / f"{fit.kernel}__{tag}__fit.json"
    path.write_text(json.dumps(fit.as_dict(), indent=2))
    return path


def load_fit(kernel_name: str, tag: str = "default") -> Fit:
    path = ARTIFACTS_DIR / f"{kernel_name}__{tag}__fit.json"
    raw = json.loads(path.read_text())
    raw["feature_names"] = tuple(raw["feature_names"])
    raw["coefs"] = tuple(raw["coefs"])
    return Fit(**raw)


def samples_to_fit_input(
    kernel: AotKernel, samples: list[dict],
) -> list[tuple[SysDims, Knobs, MeshSpec, int]]:
    """Convert saved-JSON samples back to the tuple form fit_nnls expects."""
    out = []
    for s in samples:
        sys_kw = dict(s["sys"])
        sys_kw["kgrid"] = tuple(sys_kw["kgrid"])
        if sys_kw.get("fft_grid") is not None:
            sys_kw["fft_grid"] = tuple(sys_kw["fft_grid"])
        sys_kw.pop("n_k", None)  # derived; not a constructor arg
        sys = SysDims(**sys_kw)
        knobs = Knobs.of(**s["knobs"])
        mesh = MeshSpec(**s["mesh"])
        out.append((sys, knobs, mesh, int(s["meas"]["total"])))
    return out
