"""Round-off tolerances for linear-algebra checks, scaled with the problem.

A defect relative to the matrix scale (``max|A - A^H| / max|A|``, a residual,
an orthogonality error) grows with the accumulation length of the products
that built the matrix.  A fixed bar such as ``1e-12`` is below what round-off
alone gives a GEMM-built Hermitian once ``n·eps`` passes it (``n`` about 4500
in complex128), and far above it for small ``n``.  Every such check in the
tree compares against :func:`roundoff_tol` instead.
"""
from __future__ import annotations

import numpy as np

#: How far beyond the worst-case accumulation bound ``depth·n·eps`` a defect
#: must be before a check refuses: a refusal names a wrong input (a missing
#: conjugate, a sign, an unsymmetrized product), never round-off.
ROUNDOFF_MARGIN = 64


def roundoff_tol(n: int, depth: int = 1, dtype=np.complex128) -> float:
    """``ROUNDOFF_MARGIN · depth · n · eps(dtype)``: the relative round-off bound
    of a matrix built by ``depth`` chained products of accumulation length ``n``
    (the matrix side when the inner length is not known)."""
    eps = float(np.finfo(np.dtype(dtype)).eps)
    return float(ROUNDOFF_MARGIN) * max(1, int(depth)) * max(1, int(n)) * eps
