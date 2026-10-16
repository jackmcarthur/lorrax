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

BATCHED STACKS: :data:`BATCHED_EIGH_ROUTE`.  jaxlib's ROCm eigh (``hipsolver_syevd_ffi``)
loops rocSOLVER ``zheevd``/``dsyevd`` over the batch, one matrix per call (Fe line_select:
c128[16, 312, 312] at 11.2 ms per matrix, 26.4k kernels per stack).  The leg's row names a
native target that solves a whole stack in ONE rocSOLVER strided-batched call
(``src/ffi/cpp/rocsolver/batched_eigh_ffi.cc``), the driver per dtype and the stacks it wins on
(measured; the row's comment).  It replaces the solve and nothing else: a complex stack is
still reflected first, and a failed matrix is NaN.  rocSOLVER 3.35 ``zheevd`` -- jaxlib's
per-matrix loop (this leg's route without the row) and the batched call alike -- fails
SILENTLY on some inputs: NaN eigenvectors, finite eigenvalues, info = 0.  Measured (pool log
eigh_probe4, n = 312): 3 of 128 reflected degenerate matrices (multiplicity >= 16) in both
routes, every unreflected real-valued one, and 8 of the 64 unreflected Fe ISDF C_q matrices
(eigh_probe5; the reflector clears all 64); ``zheevdj`` had 0 of 1536, float64 0 of 768.
jaxlib passes such a matrix on as finite eigenvalues with NaN vectors.  This route marks every non-finite or info != 0 matrix
NaN in both outputs, on the device (no host read), and the caller's result check refuses it by
name: the q-batched plans' batch_reshard route checks every stack (GATE
distrib_la_result_check after its retries).  hsuite na_sc's exact ISDF C_q stack
c128[4, 96, 96] (a 40-fold eigenvalue) is one such input (gate case, below the c128 window).
A re-solve of the failed matrices would need the failure count on the host (one sync per
call), so the handler has none.  WORKSPACE: the handler runs a stack in chunks whose rocSOLVER
workspace fits the fixed tile (``BatchedEighRow.tile_bytes``), one XLA scratch allocation;
:func:`batched_workspace_bytes` prices it from shapes.
A routed stack whose
target is not loadable refuses by GATE ``batched_eigh_target`` (no fallback to the loop).
Gate: ``tests/test_rocsolver_eigh_gate.py`` (residual, orthogonality, eigenvalues and
eigenspaces against ``jnp.linalg.eigh`` on the same GCD).

VALID, NOT IDENTICAL.  The batched driver and jaxlib's loop are both backward stable, not
bitwise equal: on exactly degenerate eigenvalues they return different orthonormal bases of
the same eigenspaces, and elsewhere they differ at rounding level.  No caller may depend on
either.  hsuite na_sc does (wave-6 integration, pool logs eighb_*): its outcome follows the
eigensolver's rounding at ``isdf.cplus.factor``'s c128[4, 96, 96] charge C_q stack (40
decoupled rows on one constant diagonal, a 40-fold eigenvalue inside the spectrum).  C+ differs
between the routes by <= 1.9e-8 relative, as each route does from host LAPACK, and the stage
gives: jaxlib's loop, a 0.110 eV tolerance miss; the batched driver, GATE
shared_pole_zero_ritz (its ``infinite_weight_ok``, a 1e-6 bound on near-null Ritz weight);
jaxlib's loop on a second unitary reflection of the same matrix, PASS.  The stacks are gate
cases (``tests/data/rocsolver_eigh_na_sc_stacks.npz``).
"""

from __future__ import annotations

from typing import NamedTuple

import math

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["eigh_local", "EighResult", "COMPLEX_EIGH_ROUTE", "eigh_route",
           "BATCHED_EIGH_ROUTE", "BatchedEighRow", "EighWindow", "batched_eigh_row",
           "batched_workspace_bytes"]


class EighResult(NamedTuple):
    """The fields of ``jnp.linalg.eigh``'s result (whose class is private to jax)."""
    eigenvalues: jax.Array
    eigenvectors: jax.Array


#: How the complex Hermitian eigh is driven, by ``lxkit`` leg key.  Absent = ``jnp.linalg.eigh``.
COMPLEX_EIGH_ROUTE = {"ROCM": "reflector"}


class EighWindow(NamedTuple):
    """One dtype's driver and the stacks it is used for."""
    driver: str          #: rocSOLVER strided-batched driver ("heevd"/"syevd": divide and
                         #: conquer; "heevdj"/"syevdj": Jacobi subproblems, also built)
    min_batch: int       #: smallest stack (product of the leading axes) it takes
    min_n: int           #: smallest matrix side it takes
    max_n: int           #: largest matrix side it takes
    workspace_tiles: int  #: rocSOLVER workspace per matrix is at most this many n x n elements
                          #: (pool log eighc_wsq: c128 2.26-3.80, f64 3.26-4.84), the shape-only
                          #: price of :func:`batched_workspace_bytes`


class BatchedEighRow(NamedTuple):
    """A leg's native batched solver: its target and, per dtype name, the window it serves."""
    target: str          #: XLA FFI target (distrib_la.loader's table for the leg)
    windows: dict        #: dtype name -> EighWindow
    tile_bytes: int = 1 << 30  #: the fixed tile, ``runtime.tiles.TILE_BYTES`` (a service cannot
                               #: import ``src``; tests/test_rocsolver_eigh_gate.py pins it): the
                               #: handler runs a stack in chunks whose workspace fits it


#: Leg key -> its batched native eigh.  Absent (or a stack outside the row) = the solve is
#: ``jnp.linalg.eigh``.  ROCM windows, one MI250X GCD (pool logs eigh_bench1, eigh_bench2;
#: speedup = jnp.linalg.eigh / target): heevd | syevd, b = 2: 1.9x at n = 40..312, 1.36x at
#: 1960, 1.18x at 3120; b = 16: 10.7x | 12.0x at n = 40, 9.3x | 9.6x at n = 312; b = 64,
#: n = 128: 29.7x | 30.0x.  Outside: b = 1 is the same zheevd call (1.00x); n = 16 jaxlib's
#: small-n path wins (0.6-0.8x); n = 32 is mixed (c128 1.1-1.27x at b <= 16 but 0.78x at
#: b = 64; f64 0.84-1.08x), so min_n = 40, the smallest n measured to win at every batch
#: (1.86x at b = 2); n > 3120 not measured.  heevdj (eigh_bench3) only
#: wins at n <= 312 and is 2.8x slower than heevd at 16 x 312, so no window takes it.
BATCHED_EIGH_ROUTE = {"ROCM": BatchedEighRow("lorrax_rocsolver_batched_eigh", {
    "complex128": EighWindow("heevd", min_batch=2, min_n=40, max_n=3120, workspace_tiles=4),
    "float64":    EighWindow("syevd", min_batch=2, min_n=40, max_n=3120, workspace_tiles=5)})}


def _process_leg_key():
    from lxkit import device_vendor, leg_for_vendor
    leg = leg_for_vendor(device_vendor(jax.devices()[0]))
    return None if leg is None else leg.key


def eigh_route(dtype, leg_key=None):
    """The route this leg uses for ``dtype`` (None = ``jnp.linalg.eigh`` unchanged)."""
    if not jnp.issubdtype(dtype, jnp.complexfloating):
        return None
    return COMPLEX_EIGH_ROUTE.get(_process_leg_key() if leg_key is None else leg_key)


def batched_eigh_row(shape, dtype, leg_key=None):
    """The :data:`BATCHED_EIGH_ROUTE` row that solves a ``shape``/``dtype`` stack, or None."""
    row = BATCHED_EIGH_ROUTE.get(_process_leg_key() if leg_key is None else leg_key)
    win = None if row is None else row.windows.get(np.dtype(dtype).name)
    if win is None or len(shape) < 2:
        return None
    if math.prod(shape[:-2]) < win.min_batch or not win.min_n <= shape[-1] <= win.max_n:
        return None
    return row


def batched_workspace_bytes(shape, dtype, leg_key=None):
    """Per-device bytes the leg's batched target adds beside its operand and outputs for a local
    stack ``shape``, from shapes only (0 when the row does not take it): the rocSOLVER workspace,
    ``min(tile, b x workspace_tiles x n^2 x itemsize)``, plus the tridiagonal E (b n float64).
    The tile caps the first term whatever the batch (the handler chunks the stack)."""
    row = batched_eigh_row(shape, dtype, leg_key)
    if row is None:
        return 0
    win = row.windows[np.dtype(dtype).name]
    b, n, item = math.prod(shape[:-2]), int(shape[-1]), np.dtype(dtype).itemsize
    return min(int(row.tile_bytes), b * win.workspace_tiles * n * n * item) + b * n * 8


def _require_target(target):
    from distrib_la.loader import probe_target
    leg = _process_leg_key()
    usable, reason = probe_target(target, leg)        # loads the leg and registers its targets
    if not usable:
        raise RuntimeError(f"GATE batched_eigh_target: got: {target} unusable on {leg}: {reason}; "
                           f"want: the leg built with src/ffi/cpp/rocsolver (BATCHED_EIGH_ROUTE row)")


def _batched_call(target, a, driver, tile_bytes):
    """One strided-batched rocSOLVER call on a stack of FULL Hermitian matrices.

    The handler reads the row-major buffer as column-major, i.e. solves A^T = conj(A), whose
    eigenvectors are conj(V); its in-place result read row-major is conj(V)^T, so V = R^H."""
    real = jnp.finfo(a.dtype).dtype
    r, w, info = jax.ffi.ffi_call(
        target,
        (jax.ShapeDtypeStruct(a.shape, a.dtype), jax.ShapeDtypeStruct(a.shape[:-1], real),
         jax.ShapeDtypeStruct(a.shape[:-2], jnp.int32)),
        vmap_method="broadcast_all")(a, lower=np.int64(1), jacobi=np.int64(driver.endswith("j")),
                                     tile_bytes=np.int64(tile_bytes))
    return w, jnp.conj(jnp.swapaxes(r, -1, -2)), info


def _failed(w, v, info):
    """Per matrix: rocSOLVER reported failure, or (its silent mode) left a non-finite pair."""
    return (info != 0) | ~jnp.all(jnp.isfinite(v), axis=(-2, -1)) | ~jnp.all(jnp.isfinite(w), axis=-1)


def _native_batched_eigh(row, a):
    """eigh of a stack of full Hermitian matrices by ``row`` (one FFI call; no device control
    flow, no host read).  A matrix rocSOLVER reports failed, or leaves non-finite with info = 0
    (its silent mode, module docstring), is NaN in both w and v, on the device: jaxlib's failure
    contract, made explicit, for the caller's result check to refuse by name."""
    _require_target(row.target)
    win = row.windows[np.dtype(a.dtype).name]
    w, v, info = _batched_call(row.target, a, win.driver, row.tile_bytes)
    bad = _failed(w, v, info)
    w = jnp.where(bad[..., None], jnp.nan, w)
    v = jnp.where(bad[..., None, None], jnp.nan, v)
    return w, v


def _solve(b):
    """The solve of a full Hermitian stack: the leg's batched target where its row covers the
    stack, else ``jnp.linalg.eigh`` (exactly the call the reflector route always made)."""
    row = batched_eigh_row(b.shape, b.dtype)
    if row is None:
        return jnp.linalg.eigh(b)
    return _native_batched_eigh(row, b)


def _full(a, UPLO, symmetrize_input):
    """The full Hermitian matrix ``jnp.linalg.eigh(a, UPLO, symmetrize_input)`` solves."""
    if symmetrize_input:
        return (a + jnp.conj(jnp.swapaxes(a, -1, -2))) / 2
    if UPLO == "U":
        return jnp.triu(a) + jnp.conj(jnp.swapaxes(jnp.triu(a, 1), -1, -2))
    return jnp.tril(a) + jnp.conj(jnp.swapaxes(jnp.tril(a, -1), -1, -2))


def _reflector(n, dtype):
    """A fixed, dense, unit complex vector u (no zero, no real-only entry): deterministic in n."""
    j = jnp.arange(n, dtype=jnp.finfo(dtype).dtype)
    u = (1.5 + jnp.cos(1.3 * j)) * jnp.exp(1j * (0.7548776662466927 * j * j + 0.5698402909980532 * j))
    u = u.astype(dtype)
    return u / jnp.linalg.norm(u)


def _reflected_eigh(a, UPLO, symmetrize_input, solve=None):
    """eigh of A via B = H A H, H = I - 2 u u^H (Hermitian, unitary): same spectrum, V = H V_B.
    ``solve`` (default :func:`_solve`, the leg's) solves the full Hermitian B."""
    a = _full(a, UPLO, symmetrize_input)
    u = _reflector(a.shape[-1], a.dtype)
    au = a @ u                                    # A u           (..., n)
    uau = jnp.real(jnp.einsum("i,...i->...", jnp.conj(u), au))[..., None, None]
    # H A H = A - 2 u (A u)^H - 2 (A u) u^H + 4 (u^H A u) u u^H
    b = (a - 2 * u[:, None] * jnp.conj(au)[..., None, :] - 2 * au[..., :, None] * jnp.conj(u)[None, :]
         + 4 * uau * (u[:, None] * jnp.conj(u)[None, :]))
    w, vb = (_solve if solve is None else solve)(b)
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
    row = None if route is not None else batched_eigh_row(a.shape, a.dtype)
    if route is None and row is None:
        return jnp.linalg.eigh(a, UPLO=UPLO, symmetrize_input=symmetrize_input)
    if UPLO not in (None, "L", "U"):
        raise ValueError(f"UPLO must be 'L' or 'U', got {UPLO!r}")
    if route is None:                               # a real stack the batched row covers
        return EighResult(*_native_batched_eigh(row, _full(a, UPLO, symmetrize_input)))
    if route == "reflector":
        return EighResult(*_reflected_eigh(a, UPLO, symmetrize_input))
    raise ValueError(f"unknown complex eigh route {route!r}")
