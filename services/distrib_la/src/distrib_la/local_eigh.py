"""The local (one-device) Hermitian eigensolver: ``jnp.linalg.eigh`` with a per-leg route.

Every LORRAX site that diagonalizes a matrix on one device calls :func:`eigh_local`
(and the native ``eigh`` plan forwards to it), so how the solver is driven is
decided here, once, per platform leg.  It is a plain function (no ``jit``
boundary): inside a caller's trace it is the same single ``eigh`` primitive as
``jnp.linalg.eigh`` (plus, on a leg with a route, O(n^2) elementwise work
around it), so it fuses as the inline call did.

The row is a backend preference of this service, so it lives here and not in
``lxkit.LEGS`` ("lxkit carries policy, never tables").  A leg without a row
calls ``jnp.linalg.eigh`` itself -- CUDA and cpu are unchanged bit for bit.

ROCM, complex dtypes: ``"reflector"``.  rocSOLVER's ``zheevd``/``cheevd``
behind jaxlib's default and QR routes (ROCm 7.14.1 pip wheels, MI250X, jax
0.9.1) report failure (info != 0, which jaxlib turns into NaN) for complex
Hermitian matrices of n >= 96 whose imaginary part is exactly zero in a row or
block: real-valued matrices stored complex (every time-reversal q = 0 / TRIM
Coulomb matrix and Gram), diagonal or block-diagonal matrices, decoupled rows,
and the sentinels of :func:`distrib_la._result_check.deflate_zero_rows`
(n <= 65 and generic complex matrices pass).  The route solves the unitarily
similar ``B = H A H`` with one fixed dense complex Householder reflector
``H = I - 2 u u^H``: ``B`` has no exact structure left, its spectrum is
``A``'s, and the vectors come back as ``H V_B``.  A scalar matrix (``alpha I``,
the zero matrix of an empty padded slot) is invariant under every unitary, so
it is answered exactly (``alpha``, ``I``).  Cost O(n^2) (two rank-1
updates and one back-transform; measured <= 0.6 % at n = 512..4096) beside the O(n^3) solve; rounding is that of
the solve (``H`` is unitary to eps).  Jacobi (``syevj``) is also exact on these
inputs but 5-28x slower than ``zheevd`` for single matrices of n = 128..4096,
so it is not the route.  Evidence: lead probes 2026-10-15 (pool logs
lead_eigh_probe, lead_eigh_probe2, lead_eigh_time, lead_eigh_probe3); the
failure made hsuite sp_export refuse with GATE shared_pole_gram_valid (NaN out
of ``response_bank._coulomb_roots``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["eigh_local", "EighResult", "COMPLEX_EIGH_ROUTE", "eigh_route"]


class EighResult(NamedTuple):
    """The fields of ``jnp.linalg.eigh``'s result (whose class is private to jax)."""
    eigenvalues: jax.Array
    eigenvectors: jax.Array


#: How the complex Hermitian eigh is driven, by ``lxkit`` leg key.  Absent = ``jnp.linalg.eigh``.
COMPLEX_EIGH_ROUTE = {"ROCM": "reflector"}


def _process_leg_key():
    from lxkit import device_vendor, leg_for_vendor
    leg = leg_for_vendor(device_vendor(jax.devices()[0]))
    return None if leg is None else leg.key


def eigh_route(dtype, leg_key=None):
    """The route this leg uses for ``dtype`` (None = ``jnp.linalg.eigh`` unchanged)."""
    if not jnp.issubdtype(dtype, jnp.complexfloating):
        return None
    return COMPLEX_EIGH_ROUTE.get(_process_leg_key() if leg_key is None else leg_key)


def _reflector(n, dtype):
    """A fixed, dense, unit complex vector u (no zero, no real-only entry): deterministic in n."""
    j = jnp.arange(n, dtype=jnp.finfo(dtype).dtype)
    u = (1.5 + jnp.cos(1.3 * j)) * jnp.exp(1j * (0.7548776662466927 * j * j + 0.5698402909980532 * j))
    u = u.astype(dtype)
    return u / jnp.linalg.norm(u)


def _reflected_eigh(a, UPLO, symmetrize_input):
    """eigh of A via B = H A H, H = I - 2 u u^H (Hermitian, unitary): same spectrum, V = H V_B."""
    if symmetrize_input:
        a = (a + jnp.conj(jnp.swapaxes(a, -1, -2))) / 2
    elif UPLO == "U":
        a = jnp.triu(a) + jnp.conj(jnp.swapaxes(jnp.triu(a, 1), -1, -2))
    else:
        a = jnp.tril(a) + jnp.conj(jnp.swapaxes(jnp.tril(a, -1), -1, -2))
    u = _reflector(a.shape[-1], a.dtype)
    au = a @ u                                    # A u           (..., n)
    uau = jnp.real(jnp.einsum("i,...i->...", jnp.conj(u), au))[..., None, None]
    # H A H = A - 2 u (A u)^H - 2 (A u) u^H + 4 (u^H A u) u u^H
    b = (a - 2 * u[:, None] * jnp.conj(au)[..., None, :] - 2 * au[..., :, None] * jnp.conj(u)[None, :]
         + 4 * uau * (u[:, None] * jnp.conj(u)[None, :]))
    w, vb = jnp.linalg.eigh(b)
    v = vb - 2 * u[:, None] * jnp.einsum("i,...ij->...j", jnp.conj(u), vb)[..., None, :]
    # A scalar matrix (alpha I, the zero matrix of an empty padded slot among them)
    # commutes with every unitary, so H A H = A keeps its structure and the solver
    # still fails on it; its eigenpairs are exact by inspection.
    n = a.shape[-1]
    eye = jnp.eye(n, dtype=a.dtype)
    alpha = a[..., :1, :1]
    scalar = jnp.all(a == alpha * eye, axis=(-2, -1))
    w = jnp.where(scalar[..., None], jnp.broadcast_to(jnp.real(alpha[..., 0]), w.shape), w)
    v = jnp.where(scalar[..., None, None], jnp.broadcast_to(eye, v.shape), v)
    return w, v


def eigh_local(a, UPLO=None, symmetrize_input=True):
    """``jnp.linalg.eigh(a, UPLO, symmetrize_input)``: ascending eigenvalues and the
    eigenvector columns, batched over leading axes, driven by the leg's route."""
    a = jnp.asarray(a)
    route = eigh_route(a.dtype)
    if route is None:
        return jnp.linalg.eigh(a, UPLO=UPLO, symmetrize_input=symmetrize_input)
    if UPLO not in (None, "L", "U"):
        raise ValueError(f"UPLO must be 'L' or 'U', got {UPLO!r}")
    if route == "reflector":
        return EighResult(*_reflected_eigh(a, UPLO, symmetrize_input))
    raise ValueError(f"unknown complex eigh route {route!r}")
