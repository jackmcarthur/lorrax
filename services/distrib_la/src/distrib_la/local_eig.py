"""Local complex128 nonsymmetric eigenpairs with an explicit device provider.

Resolve before tracing. CUDA uses JAX's cuSOLVER ``geev`` with RIGHT vectors
only; requesting JAX's default left vectors can dispatch to host LAPACK even
for a GPU operand. CPU LAPACK is an offline/native-CPU route, never a GPU
fallback. Unsupported accelerator providers refuse.

The caller owns placement and memory admission of its complete program. Whole
matrices must fit one device; compose this operation inside ``local_batch`` for
face-layout inputs. JAX 0.9.1 queries ``cusolverDnXgeev_bufferSize`` at execution
and allocates device FFI scratch plus host scratch, reused across the local
batch. Compiled ``memory_analysis`` alone does not price that opaque workspace.
"""
from __future__ import annotations

from functools import lru_cache
import operator
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from lxkit import device_vendor
from distrib_la.tolerance import roundoff_tol

__all__ = ["EigLocalBackend", "resolve_eig_local_backend", "eig_local",
           "eig_local_diagnostics", "eig_local_workspace"]

EigLocalBackend = Literal["cusolver", "lapack"]


def _implementation(backend):
    try:
        from jax.lax.linalg import EigImplementation
    except ImportError:                         # pinned JAX versions may keep the enum private
        from jax._src.lax.linalg import EigImplementation
    return {"cusolver": EigImplementation.CUSOLVER,
            "lapack": EigImplementation.LAPACK}[backend]


def _process_backend() -> EigLocalBackend:
    vendor = device_vendor(jax.local_devices()[0])
    if vendor == "cpu":
        return "lapack"
    if vendor == "cuda":
        return "cusolver"
    raise RuntimeError(
        f"GATE local_eig_provider: no device-native general eigensolver on {vendor}; "
        "want CUDA cuSOLVER >= 11.7.1 or an actual CPU run")


def _eig(a, backend):
    w, vr = jax.lax.linalg.eig(
        a, compute_left_eigenvectors=False, compute_right_eigenvectors=True,
        implementation=_implementation(backend))
    return w, vr


@lru_cache(maxsize=1)
def resolve_eig_local_backend() -> EigLocalBackend:
    """Resolve and compile-probe this process's actual complex128 provider.

    Returns
    -------
    backend : {'cusolver', 'lapack'}
        Static argument for :func:`eig_local`. CUDA's explicit JAX lowering
        checks cuSOLVER >= 11.7.1 and the registered ``cusolver_geev_ffi``.
        LAPACK is returned only for an actual CPU process. No matrix is run
        or transferred by this probe, and it creates no communicator.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError("GATE local_eig_dtype: enable JAX x64 for complex128 eigenpairs")
    backend = _process_backend()
    device = jax.local_devices()[0]
    try:
        # Compile a tiny abstract operand on this process's local device. This
        # exercises the actual installed lowering and target, not a version
        # guess or a rank-0 availability decision.
        probe = jax.jit(lambda a: _eig(a, backend),
                        in_shardings=jax.sharding.SingleDeviceSharding(device))
        probe.lower(jax.ShapeDtypeStruct((1, 1), jnp.complex128)).compile()
    except Exception as exc:
        raise RuntimeError(
            f"GATE local_eig_provider: {backend} complex128 compile probe failed: {exc}; "
            "want a usable device-native provider (CUDA requires cuSOLVER >= 11.7.1)") from exc
    return backend


def _matrix(a, name):
    a = jnp.asarray(a)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2] or a.shape[-1] == 0:
        raise ValueError(f"local_eig: {name} must be nonempty [..., n, n], got {a.shape}")
    if np.dtype(a.dtype) != np.dtype(np.complex128):
        raise TypeError(f"local_eig: {name} must be complex128, got {a.dtype}")
    return a


def eig_local_workspace(n: int, *, backend: EigLocalBackend) -> dict:
    """Eager receipt for opaque geev scratch, beside compiled program bytes.

    CUDA queries the exact already-mapped provider with default params,
    complex128 and right vectors only. ``device_bytes`` and ``host_bytes``
    are scratch for ONE active kernel; JAX reuses it serially across the local
    batch. Concurrent kernels require a workspace each. Run on every process
    before admission; the caller owns mesh-wide agreement and its live set.

    CPU returns zero accelerator bytes and ``host_bytes=None``: CPU LAPACK's
    host workspace is not queried. The receipt is not a total-program price.
    NVIDIA's geev is a hybrid CPU/GPU algorithm, even on the explicit CUDA path.
    A failed CUDA query refuses, with no guessed-size or CPU fallback.
    """
    n = operator.index(n)
    if n < 1 or n >= 1 << 63:
        raise ValueError("eig_local_workspace: n must be a positive int64 extent")
    resolved = resolve_eig_local_backend()
    if backend != resolved:
        raise RuntimeError(
            f"GATE local_eig_provider: requested {backend}, actual provider is {resolved}")
    if backend == "lapack":
        return dict(device_bytes=0, host_bytes=None, provider="CPU LAPACK",
                    library_path=None, batch_workspace_reuse="serial")
    from distrib_la.loader import local_geev_workspace
    device, host, path = local_geev_workspace(n)
    return dict(device_bytes=device, host_bytes=host, provider="cusolverDnXgeev",
                library_path=path, batch_workspace_reuse="serial")


def eig_local(a, *, backend: EigLocalBackend):
    """Return ``(w, V)`` with ``A V = V diag(w)``, without a GPU host route.

    Parameters
    ----------
    a : complex128 array, shape (..., n, n)
        Whole local square matrices, or a batch sharded only on leading axes.
        Inside ``local_batch`` each invocation sees its own whole matrix.
    backend : {'cusolver', 'lapack'}
        Static result of eager :func:`resolve_eig_local_backend`. A provider
        inconsistent with this process's actual device refuses. Results are
        unsorted; vectors are columns. JAX returns NaNs on solver failure.

    Notes
    -----
    To obtain dual modal rows, solve ``V X = I`` through ``solve_local``;
    conjugate-transposing V is correct only for an orthonormal eigenbasis.
    This function creates no jit boundary, gathers, host callbacks or retries.
    """
    if backend not in ("cusolver", "lapack"):
        raise ValueError(f"local_eig: unknown backend {backend!r}")
    actual = _process_backend()
    if backend != actual:
        raise RuntimeError(
            f"GATE local_eig_provider: requested {backend}, actual device requires {actual}; "
            "resolve_eig_local_backend before tracing; GPU host fallback is forbidden")
    a = _matrix(a, "a")
    if not isinstance(a, jax.core.Tracer):
        if any(device_vendor(d) != device_vendor(jax.local_devices()[0]) for d in a.devices()):
            raise RuntimeError("GATE local_eig_provider: operand and resolved device differ")
        sharding = a.sharding
        if isinstance(sharding, jax.sharding.NamedSharding):
            spec = tuple(sharding.spec) + (None,) * (a.ndim - len(sharding.spec))
            if any(axis is not None for axis in spec[-2:]):
                raise ValueError("local_eig: matrix axes must be local; use local_batch on face inputs")
    return _eig(a, backend)


def eig_local_diagnostics(a, w, vr, inverse_vr):
    """Return device-only ``(status, eig_error, inverse_error, condition)``.

    ``a``, ``vr`` and ``inverse_vr`` are complex128 ``[..., n, n]`` arrays;
    ``w`` is ``[..., n]``. Each output has the leading batch shape, with no
    host transfer. Errors are Frobenius backward residuals of ``A V = V w``
    and ``V inverse_vr = I``, divided by the sum of the norms of their two
    terms (the product term uses the product of operand norms).

    Status is an int32 bitmask: 1 nonfinite operand/result, 2 eigenpair
    backward residual above ``roundoff_tol(n)``, 4 inverse-solve backward
    residual above that tolerance. ``condition = ||V||_F ||inverse_vr||_F``
    is informative only: no condition-number cutoff rejects a result.
    Callers may carry these scalars beside their own physical failure codes.
    """
    a, vr, inverse_vr = (_matrix(x, name) for x, name in
                         ((a, "a"), (vr, "vr"), (inverse_vr, "inverse_vr")))
    w = jnp.asarray(w)
    if vr.shape != a.shape or inverse_vr.shape != a.shape or w.shape != a.shape[:-1]:
        raise ValueError("local_eig_diagnostics: eigenpairs/inverse shapes must match a")
    if w.dtype != a.dtype:
        raise TypeError("local_eig_diagnostics: w must be complex128")
    norm = lambda x: jnp.linalg.norm(x, axis=(-2, -1))
    tiny = jnp.finfo(jnp.float64).tiny
    n = a.shape[-1]
    vw = vr * w[..., None, :]
    nv, ni = norm(vr), norm(inverse_vr)
    eig_error = norm(a @ vr - vw) / jnp.maximum(norm(a) * nv + norm(vw), tiny)
    inverse_error = norm(vr @ inverse_vr - jnp.eye(n, dtype=a.dtype)) / jnp.maximum(
        nv * ni + jnp.sqrt(float(n)), tiny)
    finite = (jnp.all(jnp.isfinite(a), axis=(-2, -1))
              & jnp.all(jnp.isfinite(w), axis=-1)
              & jnp.all(jnp.isfinite(vr), axis=(-2, -1))
              & jnp.all(jnp.isfinite(inverse_vr), axis=(-2, -1))
              & jnp.isfinite(eig_error) & jnp.isfinite(inverse_error))
    tol = roundoff_tol(n, a.dtype)
    status = (jnp.where(finite, 0, 1)
              | jnp.where(finite & (eig_error > tol), 2, 0)
              | jnp.where(finite & (inverse_error > tol), 4, 0)).astype(jnp.int32)
    return status, eig_error, inverse_error, nv * ni
