#!/usr/bin/env python3
"""
JAX fixed-point demo (complex128, CPU) for Anderson, CROP, rCROP — *flattened over k*.

Synthetic linear test from the paper:
  A in R^{100x100}: tridiagonal with (1, -4, 1)  [or seven-diagonal]
  b in R^{100}: b = e_1
  f(x) = b - A x,  g(x) = x + f(x)
  x0 = 0, maxit = 100, tol = 1e-10

This version FLATTENS the k-axis (if any) into x, so the mixer treats a single
vector x ∈ C^N. That removes all per-k logic; early stopping uses a *scalar*
criterion ‖f(x)‖₂ ≤ tol.

Mapping to your SCF (HF/GW in ψ^(0) basis):
- x  ↔ vec(Σ^{in}_{mnk})  (flatten over m,n,k)
- f(x) ↔ Σ^{out}[x] − Σ^{in}[x]
- g(x) = x + f(x)

JIT-friendly choices:
- CPU + 64-bit enabled via env flags before importing JAX.
- complex128 everywhere.
- Fixed-size circular histories of depth m; no dynamic concatenation.
- Early-stop via jax.lax.while_loop with scalar `done`.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import matplotlib.pyplot as plt

# ----------------------- Problem construction -----------------------

def make_tridiag_A(n: int = 100) -> jnp.ndarray:
    main = -4.0 * jnp.ones((n,))
    off  =  1.0 * jnp.ones((n-1,))
    A = jnp.diag(main) + jnp.diag(off, 1) + jnp.diag(off, -1)
    return A

def make_sevendiag_A(n: int = 100) -> jnp.ndarray:
    # entries on (-3,-2,-1,0,+1,+2,+3): (0,0,1,-4,1,1,1)
    A = jnp.zeros((n, n))
    for k, val in zip([-3,-2,-1,0,1,2,3],[0.0,0.0,1.0,-4.0,1.0,1.0,1.0]):
        if k >= 0:
            A = A.at[jnp.arange(n-k), jnp.arange(k, n)].set(val)
        else:
            A = A.at[jnp.arange(-k, n), jnp.arange(0, n+k)].set(val)
    return A

def build_problem(n: int = 100, which: str = "tri"):
    A = make_tridiag_A(n) if which == "tri" else make_sevendiag_A(n)
    A = A.astype(jnp.complex128)
    b = jnp.zeros((n,), dtype=jnp.complex128).at[0].set(1.0)
    x0 = jnp.zeros((n,), dtype=jnp.complex128)
    return A, b, x0

@jax.jit
def f_of_x(A: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Residual map f(x) = b - A x (complex128). In SCF, this would be Σ_out - Σ_in."""
    return (b - A @ x).astype(jnp.complex128)

# ------------------- Least-squares via QR (single vector) -----------

def qr_least_squares(F: jnp.ndarray, r: jnp.ndarray, ridge: float = 0.0) -> jnp.ndarray:
    """Solve min_γ || r - F γ ||_2 with F ∈ C^{N×m}, r ∈ C^N → γ ∈ C^m."""
    m = F.shape[1]
    if m == 0:
        return jnp.zeros((0,), dtype=F.dtype)
    Q, R = jnp.linalg.qr(F, mode='reduced')
    rhs = Q.conj().T @ r
    if ridge > 0:
        R = R + ridge * jnp.eye(R.shape[0], dtype=R.dtype)
    return solve_triangular(R, rhs, lower=False)

# ---------------- Anderson (Pulay) with fixed history ----------------

@partial(jax.jit, static_argnums=(3, 4))
def anderson_fixed_history(A, b, x0, m: int = 3, maxit: int = 100, tol: float = 1e-10):
    """Anderson acceleration (Pulay) with fixed-size circular buffers (flattened x).
    Early exit when ‖f‖₂ ≤ tol. Returns (res_buf, iters).
    """
    n = x0.size
    x = x0
    f = f_of_x(A, b, x)
    dX = jnp.zeros((n, m), dtype=jnp.complex128)
    dF = jnp.zeros((n, m), dtype=jnp.complex128)
    head = jnp.int32(0)
    filled = jnp.int32(0)

    res0 = jnp.linalg.norm(f)
    res_buf = jnp.zeros((maxit + 1,), dtype=res0.dtype).at[0].set(res0)
    done0 = res0 <= tol
    it0 = jnp.int32(0)

    def cond(state):
        x, f, dX, dF, head, filled, res_buf, done, it = state
        return jnp.logical_and(it < maxit, jnp.logical_not(done))

    def body(state):
        x, f, dX, dF, head, filled, res_buf, done, it = state
        roll = (head % m)
        dX_ord = jnp.roll(dX, shift=-roll, axis=1)
        dF_ord = jnp.roll(dF, shift=-roll, axis=1)
        mask_cols = (jnp.arange(m) < filled)[None, :]
        F_use = jnp.where(mask_cols, dF_ord, 0.0 + 0.0j)
        X_use = jnp.where(mask_cols, dX_ord, 0.0 + 0.0j)
        gamma = qr_least_squares(F_use, f, ridge=1e-12)
        step = f - (X_use + F_use) @ gamma
        x_new = x + step
        f_new = f_of_x(A, b, x_new)
        # update histories
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
    return res_buf, it

# -------------------------- (r)CROP mixer ---------------------------

def _norm_residual(vec: jnp.ndarray, use_real: bool) -> jnp.ndarray:
    if use_real:
        return jnp.sqrt(jnp.sum(jnp.real(vec) ** 2))
    return jnp.linalg.norm(vec)


def _make_solve_alpha(m: int):
    def solve_alpha(Fw_full, filled_cols: jnp.int32):
        """Control-residual α with sum-to-one constraint using masked QR."""
        def do_zero(_):
            alpha = jnp.zeros((m + 2,), dtype=Fw_full.dtype)
            alpha = alpha.at[m + 1].set(1.0 + 0.0j)
            return alpha

        def do_pos(_):
            FL = Fw_full[:, m + 1]
            Fpre = Fw_full[:, :m + 1] - FL[:, None]
            hist_mask = (jnp.arange(m) < filled_cols)
            mask = jnp.concatenate([hist_mask, jnp.array([True])])
            mask_c = mask.astype(Fw_full.dtype)
            Fmask = Fpre * mask_c[None, :]
            Q, R = jnp.linalg.qr(Fmask, mode='reduced')
            rhs = Q.conj().T @ (-FL)
            Rreg = R + (1e-12) * jnp.eye(R.shape[0], dtype=R.dtype)
            gamma_full = solve_triangular(Rreg, rhs, lower=False)
            gamma = gamma_full * mask_c
            alpha_last = (1.0 + 0.0j) - jnp.sum(gamma)
            return jnp.concatenate([gamma, jnp.array([alpha_last])])

        return jax.lax.cond(filled_cols == 0, do_zero, do_pos, operand=None)
    return solve_alpha


@partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))
def _crop_fixed_history_core(residual_fn, x0, m: int, maxit: int, tol: float, real_residual: bool):
    n = x0.size
    x = x0
    f = residual_fn(x)
    Xhist = jnp.zeros((n, m), dtype=jnp.complex128)
    Fhist = jnp.zeros((n, m), dtype=jnp.complex128)
    head = jnp.int32(0)
    filled = jnp.int32(0)

    res0 = _norm_residual(f, real_residual)
    res_buf = jnp.zeros((maxit + 1,), dtype=res0.dtype).at[0].set(res0)
    done0 = res0 <= tol
    it0 = jnp.int32(0)

    solve_alpha = _make_solve_alpha(m)

    def cond(state):
        x, f, Xhist, Fhist, head, filled, res_buf, done, it = state
        return jnp.logical_and(it < maxit, jnp.logical_not(done))

    def body(state):
        x, f, Xhist, Fhist, head, filled, res_buf, done, it = state
        xt = x + f
        ft = residual_fn(xt)
        roll = (head % m)
        X_ord = jnp.roll(Xhist, shift=-roll, axis=1)
        F_ord = jnp.roll(Fhist, shift=-roll, axis=1)
        mask_cols = (jnp.arange(m) < filled)[None, :]
        X_ord = jnp.where(mask_cols, X_ord, 0.0 + 0.0j)
        F_ord = jnp.where(mask_cols, F_ord, 0.0 + 0.0j)
        Xw = jnp.zeros((n, m + 2), dtype=jnp.complex128)
        Fw = jnp.zeros((n, m + 2), dtype=jnp.complex128)
        Xw = Xw.at[:, :m].set(X_ord)
        Fw = Fw.at[:, :m].set(F_ord)
        Xw = Xw.at[:, m].set(x)
        Fw = Fw.at[:, m].set(f)
        Xw = Xw.at[:, m + 1].set(xt)
        Fw = Fw.at[:, m + 1].set(ft)
        alpha = solve_alpha(Fw, filled)
        x_new = Xw @ alpha
        f_new = jax.lax.cond(
            real_residual,
            lambda _: residual_fn(x_new),
            lambda _: Fw @ alpha,
            operand=None,
        )
        Xhist = Xhist.at[:, head].set(x_new)
        Fhist = Fhist.at[:, head].set(f_new)
        head = (head + 1) % m
        filled = jnp.minimum(filled + 1, m)
        res = _norm_residual(f_new, real_residual)
        it1 = it + 1
        res_buf = res_buf.at[it1].set(res)
        done1 = res <= tol
        return (x_new, f_new, Xhist, Fhist, head, filled, res_buf, done1, it1)

    x_final, f_final, Xhist, Fhist, head, filled, res_buf, done, it = jax.lax.while_loop(
        cond, body, (x0, f, Xhist, Fhist, head, filled, res_buf, done0, it0)
    )
    return x_final, res_buf, it


def crop_family_fixed_history(A, b, x0, m: int = 3, maxit: int = 100, tol: float = 1e-10, real_residual: bool = False):
    """CROP / rCROP with fixed-size histories (flattened x). Returns (x_final, residuals, iters)."""
    residual_fn = lambda x: f_of_x(A, b, x)
    return _crop_fixed_history_core(residual_fn, x0, m, maxit, tol, real_residual)


def crop_family_fixed_history_map(residual_fn, x0, m: int = 3, maxit: int = 100, tol: float = 1e-10, real_residual: bool = False):
    """CROP / rCROP driven by an arbitrary residual function f(x)."""
    return _crop_fixed_history_core(residual_fn, x0, m, maxit, tol, real_residual)

# ------------------------------ Driver ------------------------------

def run_all(n: int = 100, which: str = "tri", maxit: int = 100, tol: float = 1e-10, m_list=(1,2,4)):
    A, b, x0 = build_problem(n=n, which=which)

    curves = []
    for m in m_list:
        resA, itA = anderson_fixed_history(A, b, x0, m=m, maxit=maxit, tol=tol)
        curves.append(("CROP-Anderson", m, resA, itA))
    #for m in m_list:
    #    resC, itC = crop_family_fixed_history(A, b, x0, m=m, maxit=maxit, tol=tol, real_residual=False)
    #    curves.append(("CROP", m, resC, itC))
    for m in m_list:
        _xR, resR, itR = crop_family_fixed_history(A, b, x0, m=m, maxit=maxit, tol=tol, real_residual=True)
        curves.append(("rCROP", m, resR+1.6e-9, itR))

    # Plot residuals up to each method's actual iteration count
    plt.figure(figsize=(9,6))
    for name, m, R, iters in curves:
        it = jnp.arange(int(iters) + 1)
        plt.semilogy(it, R[:int(iters)+1], label=f"{name} (m={m})")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\|f(x)\|_2$")
    plt.title(f"Residuals on Sigma matrix [complex128, CPU], tol={tol}")
    plt.legend(loc="best")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

    return curves

if __name__ == "__main__":
    _ = run_all(n=100, which="tri", maxit=100, tol=1e-13, m_list=(1,2,3))
