"""
Anderson and rCROP acceleration for the self-consistent loops.

These methods accelerate fixed-point iterations of the form:
    x^{(k+1)} = g(x^{(k)}) = x^{(k)} + f(x^{(k)})

where f(x) = 0 at the solution. In the context of GW self-consistency:
    x  ↔ Σ^{in}_{mnk}               (the iterate, kept in its own shape and layout)
    f(x) = Σ^{out}[x] - Σ^{in}[x]  (residual)

Two methods, both host-driven (not jitted), because the residual function
calls pre-jitted pipelines:
- :func:`anderson_nojit` — Anderson type II, one map evaluation per
  iteration; the SC loop (``gw.sc_iteration``) uses it.
- :func:`rcrop_nojit` — rCROP, two evaluations per iteration; the EQP2
  refinement uses it.

References:
    Wan & Międlar, "On the Convergence of CROP-Anderson Acceleration Method"
"""

from __future__ import annotations

import os
from typing import Callable, NamedTuple

# Enable 64-bit precision before importing JAX
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------


class AccelerationResult(NamedTuple):
    """Result from an acceleration method."""

    x: jnp.ndarray  # Final iterate
    residual_norms: jnp.ndarray  # Residual norm at each iteration
    iterations: int  # Number of iterations performed
    converged: bool  # Whether tolerance was reached


def _pin_entry(v, entry_sharding):
    """Pin one history entry to the operand layout (identity if unsharded).

    The one small device placement shared by rcrop_nojit and anderson_nojit:
    an entry is carry-shaped (the SC Hamiltonian), already sharded like the
    operand, so this is a layout pin, not a host payload transfer.
    """
    return v if entry_sharding is None else jax.device_put(v, entry_sharding)


# -----------------------------------------------------------------------------
# Least-squares solvers
# -----------------------------------------------------------------------------


def _solve_crop_alpha_stacked(Fw: jnp.ndarray) -> jnp.ndarray:
    """The CROP affine least-squares solve on a STACKED window, no flatten.

    ``Fw`` is ``(k+1,) + operand_shape``: history entries 0..k-1 in
    chronological order (zero for unfilled slots), trial residual last.
    It returns ``alpha`` minimising ``||sum_i alpha_i Fw_i||`` subject to
    ``sum_i alpha_i = 1``.

    Every reduction below runs over the OPERAND axes, which is the whole
    point: the entries stay in their own layout and are never reshaped, so
    a mesh-sharded operand contributes one all-reduce of a (k, k) Gram and
    two of length k.  The flat form's ``jnp.linalg.qr`` cannot do that —
    QR has no distributed lowering, so it all-gathers its (n, k) operand
    onto one device, which is the residency this avoids.

    Normal equations square the condition number.  What bites in a CROP
    window is column SCALE, not column angle — the entries are residual
    differences whose norms span decades — so each is normalised before the
    Gram is formed and the scale undone after.  That also fixes the ridge's
    meaning: a unit-column Gram has unit diagonal, so the absolute 1e-12 is
    relative.  (The flat form added 1e-12 to R's diagonal, which for a
    near-null direction returns a huge γ rather than a damped one.)
    """
    k = Fw.shape[0] - 1
    ax = tuple(range(1, Fw.ndim))          # every operand axis
    bshape = (k,) + (1,) * (Fw.ndim - 1)   # broadcast a per-entry scalar

    f_trial = Fw[-1]
    F_hist = Fw[:k]

    # Valid = nonzero norm in the ORIGINAL window, not in the difference.
    hist_norms = jnp.sqrt(jnp.sum(jnp.abs(F_hist) ** 2, axis=ax))
    valid_hist = hist_norms > 1e-14

    F_prev = jnp.where(valid_hist.reshape(bshape),
                       F_hist - f_trial[None], 0.0 + 0.0j)

    col_norm = jnp.sqrt(jnp.sum(jnp.abs(F_prev) ** 2, axis=ax))
    scale = jnp.where(col_norm > 0.0, col_norm, 1.0)
    F_scaled = F_prev / scale.reshape(bshape)

    # min ||f_trial + F_scaled δ||  ⇒  (FᴴF) δ = −Fᴴ f_trial;  γ = δ/scale.
    G = jnp.tensordot(jnp.conj(F_scaled), F_scaled, axes=(ax, ax))
    b = -jnp.tensordot(jnp.conj(F_scaled), f_trial, axes=(ax, tuple(range(f_trial.ndim))))
    G = G + 1e-12 * jnp.eye(k, dtype=G.dtype)
    gamma = jnp.linalg.solve(G, b) / scale
    gamma = jnp.where(valid_hist, gamma, 0.0 + 0.0j)

    alpha_last = (1.0 + 0.0j) - jnp.sum(gamma)
    return jnp.concatenate([gamma, jnp.array([alpha_last])])


# -----------------------------------------------------------------------------
# The two accelerators the SC loops call
# -----------------------------------------------------------------------------


def rcrop_nojit(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    m: int = 5,
    maxit: int = 100,
    tol: float = 1e-10,
    print_fn: Callable = None,
    entry_sharding=None,
    metric=None,
) -> AccelerationResult:
    """rCROP without JIT - for use when residual_fn contains JIT'd code.

    This avoids XLA constant folding issues when the residual function
    calls pre-JIT'd pipelines.

    THE ITERATE KEEPS ITS OWN SHAPE AND ITS OWN LAYOUT.  It is not
    flattened.  rCROP needs exactly two primitives — a full-array inner
    product (reduced over every operand axis, giving the (m+1, m+1) Gram)
    and an elementwise linear combination over the history — and neither
    cares about rank, so nothing here indexes a specific axis of the
    operand.  The history is a stack with a LEADING history axis,
    ``(m,) + x0.shape``; ``m`` is never a sharded axis.

    This matters because the previous form flattened to ``(n,)`` and held
    the history as ``(n, m)``.  A flat axis cannot express the operand's
    layout — there is no way to say "bra band on 'x', ket band on 'y'"
    about it — so ``jnp.zeros((n, m))`` was uncommitted and landed on ONE
    device: 2·m·nk·nb²·16 B, which is 92.2 GB at nk=144, nb=2000, m=5,
    complex128, plus a (n, m+1) window on top.  The flatten was the defect,
    not the allocation size.

    Args:
        residual_fn: f(x) -> residual (NOT JIT'd as static), shape-preserving
        x0: Initial guess, any shape
        m: History depth
        maxit: Maximum iterations
        tol: Convergence tolerance
        print_fn: Optional print function for progress
        entry_sharding: Optional ``NamedSharding`` for ONE history entry,
            i.e. for ``x0``'s shape.  The stacks are allocated at
            ``P(None, *entry_sharding.spec)`` and are born distributed
            (``out_shardings``, not a ``device_put`` of a single-device
            zeros, which would materialise the thing being avoided).  Left
            ``None``, the buffers are uncommitted exactly as before.
        metric: Optional real weights broadcastable to ``x0``'s shape.
            The Gram that fixes the mixing coefficients, and the residual
            norms, are taken over ``f * metric``; the iterate update is
            still the full elementwise combination.  A 0/1 mask restricts
            the least squares to the entries the caller trusts: the QSGW
            driver passes the non-scissored band block, so the scissored
            tail (which moves by eV per map from its own refit) cannot
            steer the coefficients.  ``None`` is the unweighted solve,
            bit for bit.

    Returns:
        AccelerationResult
    """
    shape = x0.shape
    dtype = x0.dtype

    if entry_sharding is None:
        stack_sharding = None
    else:
        from jax.sharding import NamedSharding, PartitionSpec as P
        stack_sharding = NamedSharding(
            entry_sharding.mesh, P(None, *entry_sharding.spec))

    def _entry(v):
        return _pin_entry(v, entry_sharding)

    def _zeros_hist():
        if stack_sharding is None:
            return jnp.zeros((m,) + shape, dtype=dtype)
        return jax.jit(lambda: jnp.zeros((m,) + shape, dtype=dtype),
                       out_shardings=stack_sharding)()

    x = _entry(x0)
    f = _entry(residual_fn(x))

    def _weighted(v):
        return v if metric is None else v * metric

    # History buffers
    Xhist = _zeros_hist()
    Fhist = _zeros_hist()

    # Store initial
    Xhist = Xhist.at[0].set(x)
    Fhist = Fhist.at[0].set(f)
    head = 1 % m
    filled = 1

    # Rank-agnostic 2-norm: a pure reduction over every axis.
    # ``jnp.linalg.norm(A)`` would ravel first, which is a reshape of a
    # sharded operand.
    res0 = float(jnp.sqrt(jnp.sum(jnp.abs(_weighted(f)) ** 2)))
    res_history = [res0]

    if res0 <= tol:
        return AccelerationResult(x=x, residual_norms=jnp.array(res_history),
                                  iterations=0, converged=True)

    # Broadcast an (m,) per-entry mask against a stack of any rank.
    mask_shape = (m,) + (1,) * len(shape)

    for it in range(maxit):
        # Trial step
        x_trial = _entry(x + f)
        f_trial = _entry(residual_fn(x_trial))

        # Roll history to chronological order (permutation of the leading,
        # unsharded axis: no communication).
        oldest_pos = (head - filled) % m
        X_ord = jnp.roll(Xhist, shift=-oldest_pos, axis=0)
        F_ord = jnp.roll(Fhist, shift=-oldest_pos, axis=0)

        # Mask unfilled entries
        mask_cols = (jnp.arange(m) < filled).reshape(mask_shape)
        X_ord = jnp.where(mask_cols, X_ord, 0.0 + 0.0j)
        F_ord = jnp.where(mask_cols, F_ord, 0.0 + 0.0j)

        # Build window; the transient inherits the entries' layout.
        Xw = jnp.concatenate([X_ord, x_trial[None]], axis=0)
        Fw = jnp.concatenate([F_ord, f_trial[None]], axis=0)

        # Solve for mixing coefficients.  The only collective in the
        # accelerator: one (m+1, m+1) Gram plus two length-(m+1) reductions.
        alpha = _solve_crop_alpha_stacked(_weighted(Fw))

        # Update iterate: contraction over the LEADING axis only, so it is
        # elementwise in the operand axes — no communication.
        x_new = _entry(jnp.tensordot(alpha, Xw, axes=(0, 0)))

        # rCROP: compute real residual
        f_new = _entry(residual_fn(x_new))

        # Store in history
        Xhist = Xhist.at[head].set(x_new)
        Fhist = Fhist.at[head].set(f_new)
        head = (head + 1) % m
        filled = min(filled + 1, m)

        res = float(jnp.sqrt(jnp.sum(jnp.abs(_weighted(f_new)) ** 2)))
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


def _solve_alpha_filtered(Fw: jnp.ndarray, cond_max: float = 1.0e12):
    """``_solve_crop_alpha_stacked`` with the Walker-Ni conditioning filter.

    Same affine least squares (newest entry last, zero-norm slots invalid),
    but the OLDEST valid differences are dropped until the unit-column Gram
    has condition number <= ``cond_max`` (1e12 on the Gram = 1e6 on R, the
    DFTK / Walker-Ni cap).  One (k, k) Gram and one length-k reduction, as
    before; the reduced solves are on the host.  Returns (alpha, n_used).
    """
    k = Fw.shape[0] - 1
    ax = tuple(range(1, Fw.ndim))
    bshape = (k,) + (1,) * (Fw.ndim - 1)
    f_new = Fw[-1]
    F_hist = Fw[:k]
    hist_norms = np.asarray(jnp.sqrt(jnp.sum(jnp.abs(F_hist) ** 2, axis=ax)))
    valid = hist_norms > 1e-14
    F_prev = jnp.where(jnp.asarray(valid).reshape(bshape),
                       F_hist - f_new[None], 0.0 + 0.0j)
    col = np.asarray(jnp.sqrt(jnp.sum(jnp.abs(F_prev) ** 2, axis=ax)))
    scale = np.where(col > 0.0, col, 1.0)
    F_s = F_prev / jnp.asarray(scale).reshape(bshape)
    G = np.real(np.asarray(jnp.tensordot(jnp.conj(F_s), F_s, axes=(ax, ax))))
    b = -np.real(np.asarray(jnp.tensordot(
        jnp.conj(F_s), f_new, axes=(ax, tuple(range(f_new.ndim))))))
    use = [i for i in range(k) if valid[i] and col[i] > 0.0]
    while use:
        Gu = G[np.ix_(use, use)]
        w = np.linalg.eigvalsh(Gu)
        if w[0] > 0.0 and w[-1] / w[0] <= cond_max:
            break
        use = use[1:]                       # drop the oldest difference
    gamma = np.zeros(k)
    if use:
        Gu = G[np.ix_(use, use)] + 1e-12 * np.eye(len(use))
        gamma[use] = np.linalg.solve(Gu, b[use]) / scale[use]
    alpha = np.concatenate([gamma, [1.0 - gamma.sum()]])
    return jnp.asarray(alpha, dtype=Fw.dtype), len(use)


def anderson_nojit(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    m: int = 5,
    maxit: int = 100,
    tol: float = 1e-10,
    print_fn: Callable = None,
    entry_sharding=None,
    metric=None,
) -> AccelerationResult:
    """Anderson type II (Pulay/DIIS): ONE map evaluation per iteration.

    Every evaluated pair (x_i, f_i = G(x_i) - x_i) enters a history of the
    newest ``m + 1``; the next and only evaluation is at

        x_{k+1} = sum_i alpha_i (x_i + f_i),
        alpha = argmin || sum_i alpha_i f_i ||_metric,  sum_i alpha_i = 1,

    with beta = 1 (no damping).  This is CROP with real residuals (Wan &
    Miedlar 2024, "CROP-Anderson"); rCROP (:func:`rcrop_nojit`) reaches the
    same iterates on an affine map at two evaluations per iteration, the
    second re-evaluating a residual the linear model already predicts.

    Two safeguards, neither costing a map evaluation nor carrying a tunable
    constant (the threshold is the literature's numerical default):

    * conditioning filter (Walker & Ni 2011; DFTK): the oldest differences
      are dropped until the unit-column Gram has cond <= 1e12 (cond(R) <=
      1e6), :func:`_solve_alpha_filtered`;
    * nonmonotone fallback (after Ouyang et al. 2023): an evaluation worse
      than every residual in the window means the multisecant model failed
      there, so the next point is the two-point secant between the best
      evaluated pair and the rejected one -- a free line search along the
      failed step; never twice in a row; the rejected pair stays in the
      history (it is valid secant data).

    There is deliberately NO restart on the map's discrete events: early QSGW
    maps grow the sampled Sigma grid on every call, and restarting there
    reduced the method to Picard steps, which diverge on an expansive map
    (CrI3 / Fe bispinor, 2026-09-24).  The jumps are far below the
    secant-model error at that stage, and stale pairs are down-weighted by
    the least squares itself.

    alpha is solved over the REAL numbers: the iterates are Hermitian
    matrices, a real vector space, and the Gram of Hermitian residuals is
    real.  THE ITERATE KEEPS ITS OWN SHAPE AND LAYOUT (see
    :func:`rcrop_nojit`): the history is a stack on a leading, never-sharded
    axis at ``P(None, *entry_sharding.spec)``, the only collective is one
    (m+1, m+1) Gram plus a length-(m+1) reduction, and the update is an
    elementwise combination over the history axis.  Residency: 2(m+1)
    history entries, the same as rCROP's 2m history plus its trial pair.
    """
    shape = x0.shape
    dtype = x0.dtype

    if entry_sharding is None:
        stack_sharding = None
    else:
        from jax.sharding import NamedSharding, PartitionSpec as P
        stack_sharding = NamedSharding(
            entry_sharding.mesh, P(None, *entry_sharding.spec))

    def _entry(v):
        return _pin_entry(v, entry_sharding)

    def _zeros_hist():
        if stack_sharding is None:
            return jnp.zeros((m,) + shape, dtype=dtype)
        return jax.jit(lambda: jnp.zeros((m,) + shape, dtype=dtype),
                       out_shardings=stack_sharding)()

    def _weighted(v):
        return v if metric is None else v * metric

    def _norm(v):
        return float(jnp.sqrt(jnp.sum(jnp.abs(_weighted(v)) ** 2)))

    x = _entry(x0)
    f = _entry(residual_fn(x))
    Xhist = _zeros_hist()
    Fhist = _zeros_hist()
    head, filled = 0, 0
    res_history = [_norm(f)]
    if res_history[0] <= tol:
        return AccelerationResult(x=x, residual_norms=jnp.array(res_history),
                                  iterations=0, converged=True)
    mask_shape = (m,) + (1,) * len(shape)
    best, best_res = (x, f), res_history[0]
    window_res = [res_history[0]]
    fallback = False

    for it in range(maxit):
        oldest_pos = (head - filled) % m
        X_ord = jnp.roll(Xhist, shift=-oldest_pos, axis=0)
        F_ord = jnp.roll(Fhist, shift=-oldest_pos, axis=0)
        mask_cols = (jnp.arange(m) < filled).reshape(mask_shape)
        X_ord = jnp.where(mask_cols, X_ord, 0.0 + 0.0j)
        F_ord = jnp.where(mask_cols, F_ord, 0.0 + 0.0j)
        Xw = jnp.concatenate([X_ord, x[None]], axis=0)
        Fw = jnp.concatenate([F_ord, f[None]], axis=0)
        alpha, n_used = _solve_alpha_filtered(_weighted(Fw))
        x_opt = jnp.tensordot(alpha, Xw, axes=(0, 0))
        f_opt = jnp.tensordot(alpha, Fw, axes=(0, 0))
        if fallback:
            a2, _ = _solve_alpha_filtered(_weighted(jnp.stack([best[1], f])))
            x_opt = a2[0] * best[0] + a2[1] * x
            f_opt = a2[0] * best[1] + a2[1] * f
        if print_fn is not None:
            print_fn(f"  Anderson step {it:02d}: window {n_used + 1}"
                     f"{', secant fallback to the best pair' if fallback else ''}")
        Xhist = Xhist.at[head].set(x)
        Fhist = Fhist.at[head].set(f)
        head = (head + 1) % m
        filled = min(filled + 1, m)
        Xw = Fw = X_ord = F_ord = None

        x = _entry(x_opt + f_opt)
        f = _entry(residual_fn(x))
        res = _norm(f)
        res_history.append(res)
        fallback = (not fallback) and res > max(window_res)
        window_res = (window_res + [res])[-(m + 1):]
        if res < best_res:
            best, best_res = (x, f), res
        if res <= tol:
            return AccelerationResult(x=x, residual_norms=jnp.array(res_history),
                                      iterations=it + 1, converged=True)

    return AccelerationResult(x=x, residual_norms=jnp.array(res_history),
                              iterations=maxit, converged=False)
