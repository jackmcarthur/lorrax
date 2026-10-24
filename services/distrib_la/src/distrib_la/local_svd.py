"""Device-local SVD and minimum-norm least squares with a per-leg driver.

ROCm explicitly selects JAX's QR-SVD (hipsolver_gesvd), rather than its
small-matrix Jacobi default. The latter stalled on a padded complex GMRES
Hessenberg on Frontier. The SVD, cutoff and minimum-norm equation stay the
same; no normal matrix is formed. CPU/CUDA retain their JAX defaults.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["svd_local", "lstsq_local"]


def _process_leg_key():
    from lxkit import device_vendor, leg_for_vendor
    leg = leg_for_vendor(device_vendor(jax.devices()[0]))
    return None if leg is None else leg.key


def _inexact_dtype(dtype):
    return (dtype if jnp.issubdtype(dtype, jnp.inexact) else
            jnp.dtype('float64' if dtype.itemsize == 8 else 'float32'))


def svd_local(a, full_matrices=True, compute_uv=True):
    """JAX SVD of a rank-local matrix/stack, using the ROCm QR driver."""
    if _process_leg_key() != "ROCM":
        return jnp.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
    a = jnp.asarray(a)
    a = a.astype(_inexact_dtype(a.dtype))
    try:
        from jax.lax.linalg import SvdAlgorithm
    except ImportError:                             # JAX 0.9.1 enum location
        from jax._src.lax.linalg import SvdAlgorithm
    return jax.lax.linalg.svd(a, full_matrices=full_matrices,
                              compute_uv=compute_uv, algorithm=SvdAlgorithm.QR)


def lstsq_local(a, b, rcond=None):
    """JAX-compatible ``(x, residuals, rank, s)`` from a genuine thin SVD.

    ``a`` is (m,n), ``b`` is (m,) or (m,nrhs). The minimum-norm solution
    retains s>0 and s>=rcond*s_max, exactly JAX's cutoff; None means
    eps*max(m,n), negative means eps. Zero padded columns retain min-norm0.
    Operands and the replicated small result retain their caller's layout.
    """
    if _process_leg_key() != "ROCM":
        return jnp.linalg.lstsq(a, b, rcond=rcond)
    a, b = jnp.asarray(a), jnp.asarray(b)
    if a.ndim != 2 or b.ndim not in (1, 2) or a.shape[0] != b.shape[0]:
        raise ValueError("lstsq_local wants a(m,n), b(m,) or b(m,nrhs)")
    if not a.size:
        return jnp.linalg.lstsq(a, b, rcond=rcond)
    dtype = _inexact_dtype(jnp.result_type(a, b))
    a, b = a.astype(dtype), b.astype(dtype)
    vector = b.ndim == 1
    if vector:
        b = b[:, None]
    u, s, vh = svd_local(a, full_matrices=False)
    eps = jnp.finfo(dtype).eps
    cutoff = eps*max(a.shape) if rcond is None else jnp.where(rcond < 0, eps, rcond)
    keep = (s > 0) & (s >= jnp.asarray(cutoff, s.dtype)*s[0])
    safe = jnp.where(keep, s, 1).astype(dtype)
    inv = jnp.where(keep, 1/safe, 0)[:, None]
    ub = jnp.matmul(u.conj().T, b, precision=jax.lax.Precision.HIGHEST)
    x = jnp.matmul(vh.conj().T, inv*ub, precision=jax.lax.Precision.HIGHEST)
    residuals = jnp.linalg.norm(b-jnp.matmul(a, x, precision=jax.lax.Precision.HIGHEST), axis=0)**2
    return (x.ravel() if vector else x), residuals, keep.sum(), s
