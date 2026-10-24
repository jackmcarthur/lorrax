"""Device-local SVD and minimum-norm least squares with a per-leg driver.

ROCm explicitly selects JAX's QR-SVD (hipsolver_gesvd), rather than its
small-matrix Jacobi default. The latter stalled on a padded complex GMRES
Hessenberg on Frontier. The SVD, cutoff and minimum-norm equation stay the
same; no normal matrix is formed. CPU/CUDA retain their JAX defaults.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["svd_local", "lstsq_local", "lstsq_hessenberg_local"]


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


def lstsq_hessenberg_local(h, g, active):
    """Minimum-norm solution of a zero-padded active Arnoldi Hessenberg.

    ``h`` is (cap+1,cap), ``g`` is (cap+1,), ``active`` is the number
    of visited columns. Only h[:active+1,:active] may be nonzero. ROCm
    selects a bounded static SVD prefix and pads y back to cap; the cutoff
    remains eps*max(FULL h.shape), including on rank-deficient breakdown.
    Other legs keep their full-size JAX least-squares calculation bitwise.
    Invalid/nonfinite carriers produce NaN rather than hiding a bad tail.
    """
    if h.ndim != 2 or h.shape[0] != h.shape[1]+1 or g.shape != (h.shape[0],):
        raise ValueError("lstsq_hessenberg_local wants h(cap+1,cap), g(cap+1)")
    if _process_leg_key() != "ROCM" or not h.shape[1]:
        return lstsq_local(h, g, rcond=None)[0]
    active = jnp.asarray(active)
    if not jnp.issubdtype(active.dtype, jnp.integer) or active.ndim:
        raise ValueError("active Hessenberg columns must be an integer scalar")
    cap = int(h.shape[1])
    dtype = _inexact_dtype(jnp.result_type(h, g))
    h, g = jnp.asarray(h, dtype), jnp.asarray(g, dtype)
    cutoff = jnp.finfo(dtype).eps*max(h.shape)
    widths, width = [], min(8, cap)
    while width < cap:
        widths.append(width)
        width *= 2
    widths.append(cap)

    def branch(width):
        def solve(args):
            a, b = args
            y = lstsq_local(a[:width+1, :width], b[:width+1], rcond=cutoff)[0]
            return jnp.pad(y, (0, cap-width))
        return solve

    inside = ((jnp.arange(cap+1)[:, None] <= active) &
              (jnp.arange(cap)[None, :] < active))
    valid = ((active >= 0) & (active <= cap) & jnp.all(jnp.isfinite(h)) &
             jnp.all(jnp.isfinite(g)) & jnp.all(jnp.where(inside, 0, h) == 0))
    index = sum(active > width for width in widths[:-1])

    def solve_valid(args):
        return jax.lax.cond(active == 0, lambda _: jnp.zeros(cap, dtype),
            lambda values: jax.lax.switch(index, tuple(branch(w) for w in widths), values), args)

    # Never send a malformed/nonfinite carrier to the vendor SVD: masking its
    # output afterward cannot recover from a driver stalled on the bad input.
    return jax.lax.cond(valid, solve_valid,
        lambda _: jnp.full(cap, jnp.nan, dtype), (h, g))
