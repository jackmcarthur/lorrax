"""Exact active-range local GEMM with DEVICE-side bounds (``ACTIVE_LOCAL_ROUTE`` ``"device"``).

The ``lorrax_active_gemm_device*`` handlers (``src/ffi/cpp/active_gemm/``: one leg-neutral,
run-time-compiled kernel) compute ``alpha * a[q][:, lo:hi] diag(w[q]) b[q][lo:hi, :] + beta *
c[q]`` reading ``bounds`` on the device.  Per call: no device-to-host copy, no stream
synchronization, no ``lax.cond``, no weighted copy of ``a`` (the weights are applied as the
kernel loads ``a``), no operand copy and no workspace.  FLOPs run over each row's active width
only, and no inactive column of ``a``, row of ``b`` or entry of ``w`` is read, so NaN there
never enters arithmetic.  Invalid bounds give an all-NaN result (as the guarded arms).

Callers import it only for a mesh whose route row is ``"device"`` and pass that platform key
to :func:`require_active_local_device`.  Both device legs build the targets (the inner product
on MFMA f64 on gfx90a / gfx94x, DMMA f64 on CUDA compute capability >= 8.0, FMA elsewhere).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from distrib_la.loader import probe_target

_TARGET = "lorrax_active_gemm_device"
#: No C operand (beta = 0): the kernel writes every row, empty intervals as zeros.
_OUT_TARGET = "lorrax_active_gemm_device_out"


def _cache_dir() -> str:
    from lxkit import user_cache_dir
    return str(user_cache_dir("active_gemm"))


def require_active_local_device(platform: str) -> None:
    """Load and capability-probe the device-bounds handlers before tracing a plan."""
    for target in (_TARGET, _OUT_TARGET):
        usable, reason = probe_target(target, platform)
        if not usable:
            raise RuntimeError("device-bounds active-range GEMM is unavailable: " + reason)


def device_call(a, b, bounds, weights=None, c=None, *, alpha, beta, conj_b=False):
    """The raw target: ``bounds`` int32 ``(1|nq, 2)``, ``weights`` ``(nq,K)`` or None.

    ``conj_b``: the kernel reads ``b`` conjugated as it loads it (``a diag(w) conj(b)``), so a
    caller holding ``b`` never forms ``conj(b)``."""
    nq, m, k = a.shape
    n = b.shape[2]
    weighted = weights is not None
    w = weights if weighted else jnp.zeros((1,), a.dtype)       # never read (weighted = 0)
    out = jax.ShapeDtypeStruct((nq, m, n), a.dtype)
    common = dict(nq=nq, m=m, k=k, n=n, weighted=int(weighted),
                  alpha_re=float(np.real(alpha)), alpha_im=float(np.imag(alpha)),
                  conj_b=int(bool(conj_b) and a.dtype.kind == "c"), cubin_dir=_cache_dir())
    if c is None:
        if beta != 0:
            raise ValueError("device active GEMM requires C when beta != 0")
        return jax.ffi.ffi_call(_OUT_TARGET, out, vmap_method="sequential")(
            a, b, w, bounds, **common)
    return jax.ffi.ffi_call(_TARGET, out, input_output_aliases={4: 0},
                            vmap_method="sequential")(
        a, b, w, bounds, c, beta_re=float(np.real(beta)), beta_im=float(np.imag(beta)), **common)


def active_local_device(a, b, bounds, weights, c=None, *, alpha, beta, conj_b=False):
    """``_active_local.active_local_matmul``'s contract on the device-bounds kernel.

    ``a`` ``(nq,m,K)``, ``b`` ``(nq,K,n)``, int ``bounds`` ``(nq,2)``, ``weights`` ``(nq,K)``.
    Equal, mixed, full and empty intervals take the same single call; invalid dynamic bounds
    give an all-NaN result.  No host callback, no device control flow.
    """
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise ValueError("device active GEMM operand geometry mismatch")
    if a.dtype != b.dtype or a.dtype not in (jnp.float64, jnp.complex128):
        raise TypeError("device active GEMM requires matching float64/complex128 operands")
    if bounds.shape != (a.shape[0], 2) or not jnp.issubdtype(bounds.dtype, jnp.integer):
        raise ValueError("device active GEMM bounds must have shape (nq,2)")
    if weights.shape != (a.shape[0], a.shape[2]) or weights.dtype != a.dtype:
        raise ValueError("device active GEMM weights must match (nq,K) and operand dtype")
    shape = (a.shape[0], a.shape[1], b.shape[2])
    if c is not None and (c.shape != shape or c.dtype != a.dtype):
        raise ValueError("device active GEMM C geometry or dtype mismatch")
    real = a.dtype.kind != "c"
    alpha = np.complex128(alpha).real if real else np.complex128(alpha)
    beta = np.complex128(beta).real if real else np.complex128(beta)
    # int64 bounds wider than int32 are invalid (0..K fits int32): map them to -1, not wrap.
    b32 = bounds if bounds.dtype == jnp.int32 else jnp.where(
        (bounds >= 0) & (bounds <= a.shape[2]), bounds, -1).astype(jnp.int32)
    if c is None and beta != 0:
        raise ValueError("device active GEMM requires C when beta != 0")
    return device_call(a, b, b32, weights, None if beta == 0 else c, alpha=alpha, beta=beta,
                       conj_b=conj_b)


__all__ = ["active_local_device", "device_call", "require_active_local_device"]
