"""
JAX implementations of Anderson, CROP, and rCROP acceleration methods.

These methods accelerate fixed-point iterations of the form:
    x^{(k+1)} = g(x^{(k)}) = x^{(k)} + f(x^{(k)})

where f(x) = 0 at the solution. In the context of GW self-consistency:
    x  ↔ vec(Σ^{in}_{mnk})     (flattened self-energy)
    f(x) = Σ^{out}[x] - Σ^{in}[x]  (residual)

All implementations:
- Work with complex128 arrays (required for GW)
- Use fixed-size circular history buffers (JIT-compatible)
- Support early stopping via jax.lax.while_loop
- Return residual history for convergence analysis

References:
    Wan & Międlar, "On the Convergence of CROP-Anderson Acceleration Method"
    Algorithm 2.1 (Anderson), Algorithm 2.2 (CROP), Algorithm 3.1 (CROP-Anderson)
"""

from __future__ import annotations

import os
from functools import partial
from typing import Callable, NamedTuple

# Enable 64-bit precision before importing JAX
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------


class AccelerationResult(NamedTuple):
    """Result from an acceleration method."""

    x: jnp.ndarray  # Final iterate
    residual_norms: jnp.ndarray  # Residual norm at each iteration
    iterations: int  # Number of iterations performed
    converged: bool  # Whether tolerance was reached


# -----------------------------------------------------------------------------
# Least-squares solvers
# -----------------------------------------------------------------------------


def _qr_least_squares(F: jnp.ndarray, r: jnp.ndarray, ridge: float = 1e-12) -> jnp.ndarray:
    """Solve min_γ || r - F γ ||_2 via QR factorization.

    Handles zero columns gracefully by masking out their coefficients.

    Args:
        F: Matrix of shape (n, m), complex128
        r: Vector of shape (n,), complex128
        ridge: Regularization for numerical stability

    Returns:
        γ: Solution vector of shape (m,)
    """
    m = F.shape[1]
    if m == 0:
        return jnp.zeros((0,), dtype=F.dtype)

    # Identify valid (non-zero) columns
    col_norms = jnp.linalg.norm(F, axis=0)
    valid_mask = col_norms > 1e-14

    Q, R = jnp.linalg.qr(F, mode="reduced")
    rhs = Q.conj().T @ r
    if ridge > 0:
        R = R + ridge * jnp.eye(R.shape[0], dtype=R.dtype)
    gamma = solve_triangular(R, rhs, lower=False)

    # Zero out coefficients for invalid (zero) columns
    gamma = jnp.where(valid_mask, gamma, 0.0 + 0.0j)
    return gamma


def _solve_crop_alpha(
    Fw: jnp.ndarray, filled_cols: jnp.ndarray, m: int
) -> jnp.ndarray:
    """Solve for CROP mixing coefficients with sum-to-one constraint (old version).

    Deprecated: use _solve_crop_alpha_v2 instead.
    """

    def do_zero(_):
        alpha = jnp.zeros((m + 2,), dtype=Fw.dtype)
        alpha = alpha.at[m + 1].set(1.0 + 0.0j)
        return alpha

    def do_solve(_):
        f_trial = Fw[:, m + 1]
        F_prev = Fw[:, : m + 1] - f_trial[:, None]

        hist_mask = jnp.arange(m) < filled_cols
        mask = jnp.concatenate([hist_mask, jnp.array([True])])
        mask_c = mask.astype(Fw.dtype)

        F_masked = F_prev * mask_c[None, :]
        Q, R = jnp.linalg.qr(F_masked, mode="reduced")
        rhs = Q.conj().T @ (-f_trial)
        R_reg = R + 1e-12 * jnp.eye(R.shape[0], dtype=R.dtype)
        gamma_full = solve_triangular(R_reg, rhs, lower=False)
        gamma = gamma_full * mask_c

        alpha_last = (1.0 + 0.0j) - jnp.sum(gamma)
        return jnp.concatenate([gamma, jnp.array([alpha_last])])

    return jax.lax.cond(filled_cols == 0, do_zero, do_solve, operand=None)


def _solve_crop_alpha_v2(Fw: jnp.ndarray, filled_cols: jnp.ndarray) -> jnp.ndarray:
    """Solve for CROP mixing coefficients with sum-to-one constraint.

    Solves: min_α ||F_w α||_2^2  s.t. Σα_i = 1

    The window Fw has shape (n, m+2) = (n, history_size + 1).
    Only the first `filled_cols` history columns are valid; rest are zeros.
    Last column is always the trial residual.

    We convert to affine coordinates centered on the trial vector:
        α = [γ, 1 - sum(γ)] where γ minimizes ||f_trial + F_valid @ γ||

    SOLVED FROM THE (k, k) GRAM MATRIX, NOT FROM A QR OF ``Fw``.  The row
    axis of ``Fw`` is the flattened iterate and may be SHARDED over the
    device mesh (``rcrop_nojit(row_sharding=...)``).  ``jnp.linalg.qr`` has
    no distributed lowering, so GSPMD all-gathers its whole (n, k) operand
    onto one device — 22.1 GB at nk=144, nb=2000, m=5, complex128, i.e.
    exactly the single-device object the sharding exists to remove.  Every
    row-axis contraction below reduces to at most a 6×6 complex all-reduce.

    Normal equations square the condition number, which for a general
    least squares would be the wrong trade.  Here k ≤ 6 and the columns are
    residual DIFFERENCES whose norms span orders of magnitude, so what
    actually bites is column SCALE, not column angle: each column is
    normalised before the Gram is formed and the scale is undone after.
    Equilibration also fixes the ridge's meaning — the Gram of a
    unit-column matrix has unit diagonal, so the absolute 1e-12 below IS
    relative.  (The previous form added 1e-12 to the diagonal of R, which
    for a near-null direction returns a HUGE γ rather than a damped one.)

    Args:
        Fw: Residual window matrix, shape (n, history_size + 1)
        filled_cols: Number of valid history columns

    Returns:
        α: Mixing coefficients summing to 1, shape (history_size + 1,)
    """
    ncols = Fw.shape[1]
    hist_cols = ncols - 1

    # Last column is the trial residual
    f_trial = Fw[:, -1]

    # History columns (may have zeros for unfilled slots)
    F_hist = Fw[:, :hist_cols]

    # Identify valid history columns (non-zero norm in original Fw, not the difference)
    hist_norms = jnp.linalg.norm(F_hist, axis=0)
    valid_hist = hist_norms > 1e-14  # shape (hist_cols,)

    # Compute differences for valid columns only
    # F_prev = f_hist - f_trial for valid columns, 0 for invalid
    F_prev = F_hist - f_trial[:, None]
    F_prev_masked = jnp.where(valid_hist[None, :], F_prev, 0.0 + 0.0j)

    # Column equilibration.  A null column keeps scale 1 so its Gram row
    # and column stay exactly zero and the ridge damps its γ to 0.
    col_norm = jnp.linalg.norm(F_prev_masked, axis=0)
    scale = jnp.where(col_norm > 0.0, col_norm, 1.0)
    F_scaled = F_prev_masked / scale[None, :]

    # min ||f_trial + F_scaled δ||  ⇒  (F^H F) δ = −F^H f_trial;  γ = δ/scale.
    G = F_scaled.conj().T @ F_scaled
    b = -(F_scaled.conj().T @ f_trial)
    G = G + 1e-12 * jnp.eye(hist_cols, dtype=G.dtype)
    gamma = jnp.linalg.solve(G, b) / scale

    # Zero out coefficients for invalid columns
    gamma = jnp.where(valid_hist, gamma, 0.0 + 0.0j)

    # Convert back: α = [γ; 1 - sum(γ)]
    alpha_last = (1.0 + 0.0j) - jnp.sum(gamma)
    return jnp.concatenate([gamma, jnp.array([alpha_last])])


# -----------------------------------------------------------------------------
# Anderson Acceleration (Algorithm 2.1)
# -----------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 2, 3))
def _anderson_core(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    m: int,
    maxit: int,
    tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Core Anderson acceleration loop.

    Anderson stores differences ΔX and ΔF, then solves:
        min_γ ||f^{(k)} - ΔF γ||_2

    and updates:
        x^{(k+1)} = x^{(k)} + f^{(k)} - (ΔX + ΔF) γ
    """
    n = x0.size
    dtype = x0.dtype

    x = x0
    f = residual_fn(x)

    # Circular buffers for differences
    dX = jnp.zeros((n, m), dtype=dtype)
    dF = jnp.zeros((n, m), dtype=dtype)
    head = jnp.int32(0)
    filled = jnp.int32(0)

    res0 = jnp.linalg.norm(f)
    res_buf = jnp.zeros((maxit + 1,), dtype=res0.dtype).at[0].set(res0)
    done0 = res0 <= tol
    it0 = jnp.int32(0)

    def cond(state):
        _, _, _, _, _, _, res_buf, done, it = state
        return jnp.logical_and(it < maxit, jnp.logical_not(done))

    def body(state):
        x, f, dX, dF, head, filled, res_buf, done, it = state

        # Roll buffers to get chronological order
        # oldest_pos = (head - filled) % m, roll by -oldest_pos to put oldest at column 0
        oldest_pos = (head - filled) % m
        dX_ord = jnp.roll(dX, shift=-oldest_pos, axis=1)
        dF_ord = jnp.roll(dF, shift=-oldest_pos, axis=1)

        # Mask unfilled columns (first `filled` columns are valid after rolling)
        mask_cols = (jnp.arange(m) < filled)[None, :]
        dX_use = jnp.where(mask_cols, dX_ord, 0.0 + 0.0j)
        dF_use = jnp.where(mask_cols, dF_ord, 0.0 + 0.0j)

        # Solve least-squares: min ||f - dF γ||
        gamma = _qr_least_squares(dF_use, f)

        # Anderson step: x_new = x + f - (dX + dF) γ
        step = f - (dX_use + dF_use) @ gamma
        x_new = x + step
        f_new = residual_fn(x_new)

        # Update history buffers
        dX = dX.at[:, head].set(x_new - x)
        dF = dF.at[:, head].set(f_new - f)
        head = (head + 1) % m
        filled = jnp.minimum(filled + 1, m)

        res = jnp.linalg.norm(f_new)
        it1 = it + 1
        res_buf = res_buf.at[it1].set(res)
        done1 = res <= tol

        return (x_new, f_new, dX, dF, head, filled, res_buf, done1, it1)

    x, f, dX, dF, head, filled, res_buf, done, it = jax.lax.while_loop(
        cond, body, (x, f, dX, dF, head, filled, res_buf, done0, it0)
    )
    return x, res_buf, it, done


def anderson_acceleration(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    m: int = 5,
    maxit: int = 100,
    tol: float = 1e-10,
) -> AccelerationResult:
    """Anderson Acceleration (Pulay mixing / DIIS).

    Solves x = g(x) where g(x) = x + f(x), i.e., finds f(x) = 0.

    Args:
        residual_fn: Function computing f(x) = g(x) - x
        x0: Initial guess (flattened, complex128)
        m: History depth (number of previous iterates to use)
        maxit: Maximum iterations
        tol: Convergence tolerance on ||f(x)||

    Returns:
        AccelerationResult with final x, residual history, iteration count
    """
    x, res_buf, it, done = _anderson_core(residual_fn, x0, m, maxit, tol)
    return AccelerationResult(
        x=x,
        residual_norms=res_buf[: int(it) + 1],
        iterations=int(it),
        converged=bool(done),
    )


# -----------------------------------------------------------------------------
# CROP Algorithm (Algorithm 2.2)
# -----------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 2, 3, 5))
def _crop_core(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    m: int,
    maxit: int,
    tol: float,
    use_real_residual: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Core CROP/rCROP loop.

    CROP stores iterates X and control residuals F, then builds window:
        X_w = [x^{k-m+1}, ..., x^k, x̃^{k+1}]
        F_w = [f^{k-m+1}, ..., f^k, f̃^{k+1}]

    Window size is m+1: m history slots (including current) + 1 trial.
    The current (x, f) is stored in history at position `head` BEFORE the iteration.

    Solves min ||F_w α||_2  s.t. Σα = 1, then:
        x^{k+1} = X_w α
        f^{k+1} = F_w α  (CROP) or f(x^{k+1})  (rCROP)
    """
    n = x0.size
    dtype = x0.dtype

    x = x0
    f = residual_fn(x)

    # History buffer: stores [x^{(k-m+1)}, ..., x^{(k)}] as a circular buffer
    # Size m: CROP(m) uses at most m previous iterates + 1 trial = m+1 window cols
    Xhist = jnp.zeros((n, m), dtype=dtype)
    Fhist = jnp.zeros((n, m), dtype=dtype)

    # Store initial (x0, f0) in history
    Xhist = Xhist.at[:, 0].set(x)
    Fhist = Fhist.at[:, 0].set(f)
    head = jnp.int32(1 % m)  # Next write position (handles m=1 case)
    filled = jnp.int32(1)  # One entry filled

    res0 = jnp.linalg.norm(f)
    res_buf = jnp.zeros((maxit + 1,), dtype=res0.dtype).at[0].set(res0)
    done0 = res0 <= tol
    it0 = jnp.int32(0)

    def cond(state):
        _, _, _, _, _, _, res_buf, done, it = state
        return jnp.logical_and(it < maxit, jnp.logical_not(done))

    def body(state):
        x, f, Xhist, Fhist, head, filled, res_buf, done, it = state

        # Trial step: x̃ = x + f, f̃ = f(x̃)
        x_trial = x + f
        f_trial = residual_fn(x_trial)

        # Roll history to chronological order
        # After rolling, columns 0..filled-1 contain oldest to newest
        oldest_pos = (head - filled) % m
        X_ord = jnp.roll(Xhist, shift=-oldest_pos, axis=1)
        F_ord = jnp.roll(Fhist, shift=-oldest_pos, axis=1)

        # Mask unfilled columns
        mask_cols = (jnp.arange(m) < filled)[None, :]
        X_ord = jnp.where(mask_cols, X_ord, 0.0 + 0.0j)
        F_ord = jnp.where(mask_cols, F_ord, 0.0 + 0.0j)

        # Build window: [history (m cols) | trial (1 col)] = m+1 cols max
        # Note: current (x, f) is already in history as the most recent entry
        Xw = jnp.concatenate([X_ord, x_trial[:, None]], axis=1)
        Fw = jnp.concatenate([F_ord, f_trial[:, None]], axis=1)

        # Solve for mixing coefficients with sum-to-one constraint
        alpha = _solve_crop_alpha_v2(Fw, filled)

        # Update iterate
        x_new = Xw @ alpha

        # Update residual: control (CROP) or real (rCROP)
        f_new = jax.lax.cond(
            use_real_residual,
            lambda _: residual_fn(x_new),
            lambda _: Fw @ alpha,
            operand=None,
        )

        # Store new iterate in history (circular buffer)
        Xhist = Xhist.at[:, head].set(x_new)
        Fhist = Fhist.at[:, head].set(f_new)
        head = (head + 1) % m
        filled = jnp.minimum(filled + 1, m)

        res = jnp.linalg.norm(f_new)
        it1 = it + 1
        res_buf = res_buf.at[it1].set(res)
        done1 = res <= tol

        return (x_new, f_new, Xhist, Fhist, head, filled, res_buf, done1, it1)

    x, f, Xhist, Fhist, head, filled, res_buf, done, it = jax.lax.while_loop(
        cond, body, (x, f, Xhist, Fhist, head, filled, res_buf, done0, it0)
    )
    return x, res_buf, it, done


def crop(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    m: int = 5,
    maxit: int = 100,
    tol: float = 1e-10,
) -> AccelerationResult:
    """CROP algorithm (Conjugate Residual with Optimal trial vectors).

    Uses control residuals (linear combination) which may not equal true residuals.
    More efficient than rCROP but may break down for highly nonlinear problems.

    Args:
        residual_fn: Function computing f(x) = g(x) - x
        x0: Initial guess (flattened, complex128)
        m: History depth
        maxit: Maximum iterations
        tol: Convergence tolerance

    Returns:
        AccelerationResult
    """
    x, res_buf, it, done = _crop_core(residual_fn, x0, m, maxit, tol, False)
    return AccelerationResult(
        x=x,
        residual_norms=res_buf[: int(it) + 1],
        iterations=int(it),
        converged=bool(done),
    )


def rcrop(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    m: int = 5,
    maxit: int = 100,
    tol: float = 1e-10,
) -> AccelerationResult:
    """rCROP algorithm (CROP with real residuals).

    Recomputes f(x_new) at each step instead of using linear combination.
    More robust for nonlinear problems but requires extra function evaluation.

    Args:
        residual_fn: Function computing f(x) = g(x) - x
        x0: Initial guess (flattened, complex128)
        m: History depth
        maxit: Maximum iterations
        tol: Convergence tolerance

    Returns:
        AccelerationResult
    """
    x, res_buf, it, done = _crop_core(residual_fn, x0, m, maxit, tol, True)
    return AccelerationResult(
        x=x,
        residual_norms=res_buf[: int(it) + 1],
        iterations=int(it),
        converged=bool(done),
    )


# -----------------------------------------------------------------------------
# CROP-Anderson (Algorithm 3.1)
# -----------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 2, 3, 5))
def _crop_anderson_core(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    m: int,
    maxit: int,
    tol: float,
    use_real_residual: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Core CROP-Anderson loop.

    CROP-Anderson is like CROP but checks convergence at the trial step and
    returns the trial vector x̃ when converged (equivalent to Anderson).
    """
    n = x0.size
    dtype = x0.dtype

    x = x0
    f = residual_fn(x)

    Xhist = jnp.zeros((n, m), dtype=dtype)
    Fhist = jnp.zeros((n, m), dtype=dtype)
    head = jnp.int32(0)
    filled = jnp.int32(0)

    res0 = jnp.linalg.norm(f)
    res_buf = jnp.zeros((maxit + 1,), dtype=res0.dtype).at[0].set(res0)
    done0 = res0 <= tol
    it0 = jnp.int32(0)

    # Store trial for output
    x_trial_out = x + f

    def cond(state):
        _, _, _, _, _, _, _, res_buf, done, it = state
        return jnp.logical_and(it < maxit, jnp.logical_not(done))

    def body(state):
        x, f, x_trial_out, Xhist, Fhist, head, filled, res_buf, done, it = state

        # Trial step
        x_trial = x + f
        f_trial = residual_fn(x_trial)
        res_trial = jnp.linalg.norm(f_trial)

        it1 = it + 1
        res_buf = res_buf.at[it1].set(res_trial)

        # Check if trial converged
        trial_done = res_trial <= tol

        # If trial converged, we're done; otherwise do CROP mixing
        def do_crop(_):
            oldest_pos = (head - filled) % m
            X_ord = jnp.roll(Xhist, shift=-oldest_pos, axis=1)
            F_ord = jnp.roll(Fhist, shift=-oldest_pos, axis=1)

            mask_cols = (jnp.arange(m) < filled)[None, :]
            X_ord = jnp.where(mask_cols, X_ord, 0.0 + 0.0j)
            F_ord = jnp.where(mask_cols, F_ord, 0.0 + 0.0j)

            Xw = jnp.zeros((n, m + 2), dtype=dtype)
            Fw = jnp.zeros((n, m + 2), dtype=dtype)
            Xw = Xw.at[:, :m].set(X_ord)
            Fw = Fw.at[:, :m].set(F_ord)
            Xw = Xw.at[:, m].set(x)
            Fw = Fw.at[:, m].set(f)
            Xw = Xw.at[:, m + 1].set(x_trial)
            Fw = Fw.at[:, m + 1].set(f_trial)

            alpha = _solve_crop_alpha(Fw, filled, m)
            x_new = Xw @ alpha
            f_new = jax.lax.cond(
                use_real_residual,
                lambda _: residual_fn(x_new),
                lambda _: Fw @ alpha,
                operand=None,
            )
            return x_new, f_new

        def skip_crop(_):
            return x_trial, f_trial

        x_new, f_new = jax.lax.cond(trial_done, skip_crop, do_crop, operand=None)

        # Update history
        Xhist_new = Xhist.at[:, head].set(x_new)
        Fhist_new = Fhist.at[:, head].set(f_new)
        head_new = (head + 1) % m
        filled_new = jnp.minimum(filled + 1, m)

        return (
            x_new,
            f_new,
            x_trial,
            Xhist_new,
            Fhist_new,
            head_new,
            filled_new,
            res_buf,
            trial_done,
            it1,
        )

    x, f, x_trial_out, Xhist, Fhist, head, filled, res_buf, done, it = jax.lax.while_loop(
        cond,
        body,
        (x, f, x_trial_out, Xhist, Fhist, head, filled, res_buf, done0, it0),
    )

    # CROP-Anderson returns the trial vector when converged
    x_out = jax.lax.cond(done, lambda _: x_trial_out, lambda _: x, operand=None)
    return x_out, res_buf, it, done


def crop_anderson(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    m: int = 5,
    maxit: int = 100,
    tol: float = 1e-10,
) -> AccelerationResult:
    """CROP-Anderson algorithm.

    Like CROP but checks convergence at the trial step (equivalent to Anderson
    when there's no truncation). Uses control residuals.

    Args:
        residual_fn: Function computing f(x) = g(x) - x
        x0: Initial guess
        m: History depth
        maxit: Maximum iterations
        tol: Convergence tolerance

    Returns:
        AccelerationResult
    """
    x, res_buf, it, done = _crop_anderson_core(residual_fn, x0, m, maxit, tol, False)
    return AccelerationResult(
        x=x,
        residual_norms=res_buf[: int(it) + 1],
        iterations=int(it),
        converged=bool(done),
    )


def rcrop_anderson(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    m: int = 5,
    maxit: int = 100,
    tol: float = 1e-10,
) -> AccelerationResult:
    """rCROP-Anderson algorithm.

    CROP-Anderson with real residuals (recomputes f(x) at each step).

    Args:
        residual_fn: Function computing f(x) = g(x) - x
        x0: Initial guess
        m: History depth
        maxit: Maximum iterations
        tol: Convergence tolerance

    Returns:
        AccelerationResult
    """
    x, res_buf, it, done = _crop_anderson_core(residual_fn, x0, m, maxit, tol, True)
    return AccelerationResult(
        x=x,
        residual_norms=res_buf[: int(it) + 1],
        iterations=int(it),
        converged=bool(done),
    )


# -----------------------------------------------------------------------------
# Hermitian matrix utilities for SCF
# -----------------------------------------------------------------------------


def hermitian_to_upper_flat(H: jnp.ndarray) -> jnp.ndarray:
    """Extract upper triangle of Hermitian matrix and flatten.
    
    Args:
        H: Array of shape (..., n, n) - Hermitian matrices
        
    Returns:
        Flattened upper triangle, shape (..., n*(n+1)//2) complex
        Stored as: [H[0,0], H[0,1], ..., H[0,n-1], H[1,1], H[1,2], ..., H[n-1,n-1]]
    """
    *batch, n, _ = H.shape
    # Get upper triangle indices
    i_upper, j_upper = jnp.triu_indices(n)
    # Extract upper triangle for each batch element
    H_flat = H[..., i_upper, j_upper]
    return H_flat


def upper_flat_to_hermitian(H_flat: jnp.ndarray, n: int) -> jnp.ndarray:
    """Reconstruct Hermitian matrix from flattened upper triangle.
    
    Args:
        H_flat: Flattened upper triangle, shape (..., n*(n+1)//2)
        n: Matrix dimension
        
    Returns:
        Hermitian matrix of shape (..., n, n)
    """
    batch_shape = H_flat.shape[:-1]
    i_upper, j_upper = jnp.triu_indices(n)
    
    # Initialize output
    H = jnp.zeros(batch_shape + (n, n), dtype=H_flat.dtype)
    
    # Fill upper triangle
    H = H.at[..., i_upper, j_upper].set(H_flat)
    
    # Fill lower triangle (conjugate transpose)
    # H[i,j] = conj(H[j,i]) for i > j
    i_lower, j_lower = jnp.tril_indices(n, k=-1)
    H = H.at[..., i_lower, j_lower].set(jnp.conj(H[..., j_lower, i_lower]))
    
    return H


@partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))
def _rcrop_hermitian_core(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    n_matrix: int,
    m: int,
    maxit: int,
    tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """rCROP for Hermitian matrices stored as upper triangles.
    
    The residual_fn takes and returns flattened upper triangles.
    Internally, we work with these upper-triangle vectors.
    
    Args:
        residual_fn: f(x_upper_flat) -> residual_upper_flat
        x0: Initial flattened upper triangle
        n_matrix: Matrix dimension (to reconstruct Hermitian)
        m: History depth
        maxit: Maximum iterations
        tol: Convergence tolerance
    """
    n = x0.size
    x = x0
    f = residual_fn(x)
    
    Xhist = jnp.zeros((n, m), dtype=jnp.complex128)
    Fhist = jnp.zeros((n, m), dtype=jnp.complex128)
    head = jnp.int32(0)
    filled = jnp.int32(0)
    
    res0 = jnp.linalg.norm(f)
    res_buf = jnp.zeros((maxit + 1,), dtype=res0.dtype).at[0].set(res0)
    done0 = res0 <= tol
    it0 = jnp.int32(0)
    
    def cond(state):
        _, _, _, _, _, _, res_buf, done, it = state
        return jnp.logical_and(it < maxit, jnp.logical_not(done))
    
    def body(state):
        x, f, Xhist, Fhist, head, filled, res_buf, done, it = state
        
        # Trial step
        x_trial = x + f
        f_trial = residual_fn(x_trial)
        
        # Roll history to chronological order
        oldest_pos = (head - filled) % m
        X_ord = jnp.roll(Xhist, shift=-oldest_pos, axis=1)
        F_ord = jnp.roll(Fhist, shift=-oldest_pos, axis=1)
        
        # Mask unfilled columns
        mask_cols = (jnp.arange(m) < filled)[None, :]
        X_ord = jnp.where(mask_cols, X_ord, 0.0 + 0.0j)
        F_ord = jnp.where(mask_cols, F_ord, 0.0 + 0.0j)
        
        # Build window
        Xw = jnp.concatenate([X_ord, x_trial[:, None]], axis=1)
        Fw = jnp.concatenate([F_ord, f_trial[:, None]], axis=1)
        
        # Solve for mixing coefficients
        alpha = _solve_crop_alpha_v2(Fw, filled)
        
        # Update iterate
        x_new = Xw @ alpha
        
        # Real residual (rCROP)
        f_new = residual_fn(x_new)
        
        # Store in history
        Xhist = Xhist.at[:, head].set(x_new)
        Fhist = Fhist.at[:, head].set(f_new)
        head = (head + 1) % m
        filled = jnp.minimum(filled + 1, m)
        
        res = jnp.linalg.norm(f_new)
        it1 = it + 1
        res_buf = res_buf.at[it1].set(res)
        done1 = res <= tol
        
        return (x_new, f_new, Xhist, Fhist, head, filled, res_buf, done1, it1)
    
    x, f, Xhist, Fhist, head, filled, res_buf, done, it = jax.lax.while_loop(
        cond, body, (x, f, Xhist, Fhist, head, filled, res_buf, done0, it0)
    )
    return x, res_buf, it, done


def rcrop_hermitian(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    n_matrix: int,
    m: int = 5,
    maxit: int = 100,
    tol: float = 1e-10,
) -> AccelerationResult:
    """rCROP for self-consistent Hermitian matrix problems.
    
    Optimized for Hermitian matrices by working with upper triangles only.
    The residual_fn should take/return flattened upper triangles.
    
    Args:
        residual_fn: f(x_upper_flat) -> residual_upper_flat
        x0: Initial flattened upper triangle
        n_matrix: Matrix dimension
        m: History depth
        maxit: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        AccelerationResult with final x (flattened upper triangle)
    """
    x, res_buf, it, done = _rcrop_hermitian_core(
        residual_fn, x0, n_matrix, m, maxit, tol
    )
    return AccelerationResult(
        x=x,
        residual_norms=res_buf[: int(it) + 1],
        iterations=int(it),
        converged=bool(done),
    )


def rcrop_nojit(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    m: int = 5,
    maxit: int = 100,
    tol: float = 1e-10,
    print_fn: Callable = None,
    row_sharding=None,
) -> AccelerationResult:
    """rCROP without JIT - for use when residual_fn contains JIT'd code.

    This avoids XLA constant folding issues when the residual function
    calls pre-JIT'd pipelines.

    Args:
        residual_fn: f(x) -> residual (NOT JIT'd as static)
        x0: Initial guess
        m: History depth
        maxit: Maximum iterations
        tol: Convergence tolerance
        print_fn: Optional print function for progress
        row_sharding: optional ``NamedSharding`` over the (n,) iterate's one
            axis.  THE TWO HISTORY BUFFERS ARE 2·m COPIES OF THE ITERATE
            AND ARE THE LARGEST OBJECTS IN THE SOLVER.  Left ``None``,
            ``jnp.zeros`` builds them uncommitted, i.e. on ONE device: at
            nk=144, nb=2000, m=5, complex128 that is n = 5.76e8, 9.22 GB
            per copy, 92.2 GB of history and a 22.1 GB ``concatenate``
            transient on a single device, which no rank has.  Given a
            sharding, the buffers are BORN distributed (``out_shardings``,
            not a ``device_put`` of a single-device zeros) and every
            derived array — x, f, the trial vectors, the (n, m+1) window —
            is pinned to the same layout, so per-device history is
            92.2 GB / ndev.  ``n`` must be divisible by the mesh size and
            ``x0`` must already be a ``jax.Array`` (a host array here would
            fire JAX's hidden replica ``assert_equal`` all-gather).

    Returns:
        AccelerationResult
    """
    n = x0.size
    dtype = x0.dtype

    if row_sharding is None:
        hist_sharding = None
    else:
        from jax.sharding import NamedSharding, PartitionSpec as P
        if not isinstance(x0, jax.Array):
            raise TypeError(
                "rcrop_nojit: row_sharding given but x0 is not a jax.Array "
                f"({type(x0).__name__}).  Placing a host array on a "
                "multi-process sharding fires JAX's replica assert_equal "
                "all-gather; hand in a device array.")
        if n % int(row_sharding.mesh.size):
            raise ValueError(
                f"rcrop_nojit: iterate length n={n} is not divisible by the "
                f"mesh size {int(row_sharding.mesh.size)}, so the history "
                f"buffers cannot be row-sharded.")
        hist_sharding = NamedSharding(
            row_sharding.mesh, P(row_sharding.spec[0], None))

    def _row(v):
        """Pin an (n,) array to the iterate layout (identity if unsharded)."""
        return v if row_sharding is None else jax.device_put(v, row_sharding)

    def _zeros_hist():
        if hist_sharding is None:
            return jnp.zeros((n, m), dtype=dtype)
        # out_shardings, NOT device_put(jnp.zeros(...)): the latter would
        # materialise the whole (n, m) on one device before resharding it,
        # which is the allocation this argument exists to remove.
        return jax.jit(lambda: jnp.zeros((n, m), dtype=dtype),
                       out_shardings=hist_sharding)()

    x = _row(x0)
    f = _row(residual_fn(x))

    # History buffers
    Xhist = _zeros_hist()
    Fhist = _zeros_hist()

    # Store initial
    Xhist = Xhist.at[:, 0].set(x)
    Fhist = Fhist.at[:, 0].set(f)
    head = 1 % m
    filled = 1
    
    res0 = float(jnp.linalg.norm(f))
    res_history = [res0]
    
    if res0 <= tol:
        return AccelerationResult(x=x, residual_norms=jnp.array(res_history),
                                  iterations=0, converged=True)
    
    for it in range(maxit):
        # Trial step
        x_trial = _row(x + f)
        f_trial = _row(residual_fn(x_trial))

        # Roll history to chronological order
        oldest_pos = (head - filled) % m
        X_ord = jnp.roll(Xhist, shift=-oldest_pos, axis=1)
        F_ord = jnp.roll(Fhist, shift=-oldest_pos, axis=1)
        
        # Mask unfilled columns
        mask_cols = (jnp.arange(m) < filled)[None, :]
        X_ord = jnp.where(mask_cols, X_ord, 0.0 + 0.0j)
        F_ord = jnp.where(mask_cols, F_ord, 0.0 + 0.0j)
        
        # Build window.  The (n, m+1) transient inherits the row layout
        # from its operands, so it is sharded whenever the history is.
        Xw = jnp.concatenate([X_ord, x_trial[:, None]], axis=1)
        Fw = jnp.concatenate([F_ord, f_trial[:, None]], axis=1)

        # Solve for mixing coefficients
        alpha = _solve_crop_alpha_v2(Fw, jnp.int32(filled))

        # Update iterate.  The contraction is over the (unsharded) history
        # axis, so this is row-local: no collective, layout preserved.
        x_new = _row(Xw @ alpha)

        # rCROP: compute real residual
        f_new = _row(residual_fn(x_new))

        # Store in history
        Xhist = Xhist.at[:, head].set(x_new)
        Fhist = Fhist.at[:, head].set(f_new)
        head = (head + 1) % m
        filled = min(filled + 1, m)
        
        res = float(jnp.linalg.norm(f_new))
        res_history.append(res)
        
        if print_fn is not None and it < 10:
            print_fn(f"  rCROP iter {it:02d}: residual = {res:.6e}")
        
        if res <= tol:
            if print_fn is not None:
                print_fn(f"rCROP converged in {it+1} iterations")
            return AccelerationResult(
                x=x_new, residual_norms=jnp.array(res_history),
                iterations=it+1, converged=True
            )
        
        x = x_new
        f = f_new
    
    if print_fn is not None:
        print_fn(f"rCROP did not converge after {maxit} iterations (residual = {res:.6e})")
    
    return AccelerationResult(
        x=x, residual_norms=jnp.array(res_history),
        iterations=maxit, converged=False
    )

