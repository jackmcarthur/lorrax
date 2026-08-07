"""Re-export shim — the 2-D blocked Cholesky moved to ``distrib_la``.

It is the ``native2d`` backend of ``services/distrib_la/`` now: same
right-looking blocked algorithm, same 5-axis tile layout — except that the
tile layout is INTERNAL there, behind ``distrib_la.plan('cholesky', mesh,
backend='native2d')``, which takes and returns the dense
``P(None,'x','y')`` form.

Deletion is the replumb-complete gate.  Reachers: ``common/__init__.py:6``
(package-scope re-export) and ``isdf/core.py:31``.

Three of the module's former exports are gone rather than re-exported —
``cholesky_2d_single``, ``solve_triangular_2d`` and ``cholesky_solve_2d``
had no caller anywhere in the tree, and the last two GATHERED the factor
and solved it replicated, which is the memory behaviour the 2-D kernel
exists to avoid.  ``distrib_la.factor``/``solve`` is the distributed
factor-and-back-solve surface.
"""
from __future__ import annotations

from ffi import _services

_services.ensure_on_path()

from distrib_la._native2d import cholesky as _cholesky_2d      # noqa: E402
from distrib_la._native2d import dense_to_tiles  # noqa: E402,F401
from distrib_la._native2d import tiles_to_dense  # noqa: E402,F401


def cholesky_2d_batched(mesh, J: int, b: int):
    """Old factory signature: ``(mesh, J, b) -> f(tiles) -> tiles``.

    The service's kernel takes and returns the DENSE stack, so this wraps
    it back into the tile-in/tile-out shape ``isdf/core.factor_c_q`` still
    calls with.  The replumb deletes both this and the caller's
    ``dense_to_tiles``/``tiles_to_dense`` pair in favour of
    ``plan('cholesky', mesh, backend='native2d').batched(C_q)``.
    """
    def _chol(tiles):
        return dense_to_tiles(
            _cholesky_2d(tiles_to_dense(tiles, b), mesh=mesh, block_size=b), b)
    return _chol


__all__ = ["cholesky_2d_batched", "dense_to_tiles", "tiles_to_dense"]
