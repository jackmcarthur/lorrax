"""Scalar-grid symmetrization for the centroid driver.

:func:`symmetrize_on_grid` averages a real field on the FFT grid over the
crystal's spatial Seitz operations through the symmetry service.
"""

from __future__ import annotations

import numpy as np

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

import symmetry_maps                                            # noqa: E402


def symmetrize_on_grid(
    field: np.ndarray,
    sym_ops: np.ndarray,
    translations_frac: np.ndarray | None = None,
) -> np.ndarray:
    """Average ``field`` over spatial Seitz operations on the FFT grid.

    With no translations, ``sym_ops`` are direct integer r-actions (the
    historical symmorphic API).  With ``translations_frac``, ``sym_ops`` are
    BGW reciprocal-space ``mtrx`` rows and the service builds the exact
    nonsymmorphic real-space pullback.  The latter is the centroid driver's
    atom-derived space-group path.
    """
    f = np.asarray(field, dtype=np.float64)
    N = np.asarray(f.shape, dtype=np.int64)
    ops = np.asarray(sym_ops, dtype=np.int64).reshape(-1, 3, 3)
    if ops.shape[0] <= 1 and translations_frac is None:
        return f
    flat = f.ravel()
    acc = np.zeros_like(flat)
    if translations_frac is None:
        for M in ops:
            acc += flat[symmetry_maps.grid_point_image_perm(N, M)]
    else:
        tau = np.asarray(translations_frac, dtype=np.float64).reshape(-1, 3)
        if tau.shape[0] != ops.shape[0]:
            raise ValueError(
                "translations_frac must have one row per symmetry; "
                f"got {tau.shape[0]} for {ops.shape[0]} operations")
        # Stream one service-owned row.  A stacked ``(n_ops, n_grid)`` int64
        # table is needless persistent state (366 MiB already at one million
        # grid points and 48 operations), while the scalar accumulator is the
        # only object this projection needs to retain.
        for op, shift in zip(ops, tau):
            row = symmetry_maps.fft_grid_pullback_perm(
                op[None, ...], shift[None, ...] * (2.0 * np.pi), N,
                validate=True)
            if row.shape != (1, flat.size):
                raise ValueError(
                    "symmetry-service FFT-grid pullback has wrong shape: "
                    f"{row.shape} != {(1, flat.size)}")
            acc += flat[row[0]]
    return (acc / ops.shape[0]).reshape(f.shape)
