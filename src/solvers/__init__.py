"""solvers — generic iterative eigensolvers and spectral methods (no physics dependencies)."""
from solvers.davidson import davidson, warmup_davidson_jit
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
from solvers.quadrature import feast_ellipse_quadrature

__all__ = [
    "davidson",
    "warmup_davidson_jit",
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
    "PseudobandsResult",
    "feast_ellipse_quadrature",
]
