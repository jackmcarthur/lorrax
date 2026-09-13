"""solvers — generic iterative eigensolvers and spectral methods (no physics dependencies)."""
from solvers.davidson_fixed import DavidsonPlan, DavidsonInfo, plan_local_davidson, plan_davidson
from solvers.davidson import davidson
from solvers.lanczos import block_lanczos_eig, simple_lanczos_eig, lanczos_eig_jit
from solvers.chebyshev import (
    jackson_coefficients,
    make_chebyshev_recurrence,
    chebyshev_moments,
    reconstruct_dos,
    partition_windows,
)
from solvers.dos import compute_dos, estimate_spectrum, dos_weighted_windows, geometric_windows, compute_window_partition, DOSResult, WindowPartition
from solvers.pseudobands import ritz_pseudobands, PseudobandsResult
from solvers.pseudobands_v2 import ritz_pseudobands_v2
from solvers.quadrature import feast_ellipse_quadrature

__all__ = [
    "DavidsonPlan", "DavidsonInfo", "plan_local_davidson",
    "davidson",
    "plan_davidson",
    "block_lanczos_eig",
    "simple_lanczos_eig",
    "lanczos_eig_jit",
    "jackson_coefficients",
    "make_chebyshev_recurrence",
    "chebyshev_moments",
    "reconstruct_dos",
    "partition_windows",
    "compute_dos",
    "estimate_spectrum",
    "dos_weighted_windows",
    "geometric_windows",
    "DOSResult",
    "ritz_pseudobands",
    "ritz_pseudobands_v2",
    "PseudobandsResult",
    "feast_ellipse_quadrature",
]
