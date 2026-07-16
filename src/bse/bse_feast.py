"""FEAST contour-integration eigensolver for sharded ISDF-BSE.

Beyond setup utilities (spectral-bound estimation via a short Lanczos run,
window definition in eV, and FEAST ellipse-trapezoid quadrature nodes/weights),
this module implements the full FEAST solve: shifted GMRES linear solves at each
quadrature node plus Rayleigh-Ritz eigenpair extraction (``run_feast_ritz``).
FEAST is the DEFAULT solve route for the ``bse_jax`` CLI entry point (bse_jax.py
runs ``bse_feast.main`` unless ``--lanczos`` is passed). Output is printed to
stdout in a physics-style report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable

import math
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from solvers.quadrature import feast_ellipse_quadrature as _feast_ellipse_quadrature_generic
from .bse_ring_comm import build_bse_ring_matvec, build_bse_ring_matvec_full, make_bse_shardings
from .bse_stack_matvec import build_bse_stack_matvec
from .bse_preconditioner import energy_diff_cv_k
import common.timing as timing
from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded

jax.config.update("jax_enable_x64", True)


RY_TO_EV_DEFAULT = 13.6056980659
ELLIPSE_GAMMA_FIXED = 0.2

# Cache compiled GMRES/FEAST kernels by shape + dtype (per-process).
_GMRES_SOLVER_CACHE: dict[tuple[int, float, str], Callable] = {}
_FEAST_RUNNER_CACHE: dict[tuple[int, int, int, float, float, str], Callable] = {}


@dataclass(frozen=True)
class WindowSpec:
    name: str
    a_eV: float
    b_eV: float
    note: str


@dataclass(frozen=True)
class QuadratureSpec:
    window: WindowSpec
    z_nodes: np.ndarray
    w_weights: np.ndarray
    n_quad: int
    gamma: float
    quadrature_type: str = "ellipse"


def ensure_W_R(data: dict, include_W: bool) -> dict:
    """Populate ``data['W_R']`` — the 8th (real-space W) argument the ring/stack
    matvecs and the shifted-matvec chain (``_apply_shifted_matvec``) expect.

    The screened-direct term convolves the ISDF W-tile in real space, so the
    reciprocal-space ``W_q`` (μ, ν, kx, ky, kz) is inverse-FFT'd on the k-axes.
    For the RPA density-response / kernel (``include_W=False``) the W-term is
    never built, so ``W_R`` is only a shape/sharding-correct placeholder and
    ``W_q`` itself serves.  The restart loader emits ``W_q`` but not ``W_R``, so
    every solve entry point (FEAST, spectral-bound Lanczos, ``bse_w_exact``)
    must call this before invoking a matvec.  Mutates and returns ``data``.
    """
    if include_W:
        data["W_R"] = jnp.fft.ifftn(data["W_q"], axes=(2, 3, 4), norm="ortho")
    else:
        data["W_R"] = data["W_q"]
    return data


def _apply_shifted_matvec(
    matvec,
    x: jax.Array,
    z: complex,
    data: dict,
) -> jax.Array:
    hx = matvec(
        x,
        data["psi_c_X"],
        data["psi_c_Y"],
        data["psi_v_X"],
        data["psi_v_Y"],
        data["eps_c"],
        data["eps_v"],
        data["W_R"],
        data["V_q0"],
    )
    return z * x - hx


def build_preconditioner_diagonal_sharded(
    data: dict,
    mesh_xy: Mesh,
    include_W: bool = True,
    use_tda: bool = True,
) -> jax.Array:
    eps_c = data["eps_c"]
    eps_v = data["eps_v"]
    nk = int(data["nkx"] * data["nky"] * data["nkz"])

    psi_c_X = data["psi_c_X"]
    psi_v_X = data["psi_v_X"]
    psi_c_Y = data["psi_c_Y"]
    psi_v_Y = data["psi_v_Y"]

    M_X = jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c_X), psi_v_X)
    M_Y = jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c_Y), psi_v_Y)

    V_q0 = data["V_q0"]
    S_v = jnp.einsum("MN,kcvN->kcvM", V_q0, M_Y)
    V_diag_kcv = jnp.einsum("kcvM,kcvM->kcv", jnp.conj(M_X), S_v) / nk

    if include_W:
        W_q0 = data["W_q"][:, :, 0, 0, 0]
        rho_c = jnp.einsum("kcsm,kcsm->kcm", jnp.conj(psi_c_X), psi_c_X)
        rho_v = jnp.einsum("kvsm,kvsm->kvm", jnp.conj(psi_v_Y), psi_v_Y)
        S_w = jnp.einsum("MN,kvN->kvM", W_q0, rho_v)
        W_diag_kcv = jnp.einsum("kcm,kvm->kcv", rho_c, S_w) / nk
    else:
        W_diag_kcv = jnp.zeros_like(V_diag_kcv)

    V_diag = V_diag_kcv.transpose(1, 2, 0)
    W_diag = W_diag_kcv.transpose(1, 2, 0)
    delta_E = energy_diff_cv_k(eps_c, eps_v)

    diag_h = delta_E + V_diag - W_diag
    if use_tda:
        diag_sharding = NamedSharding(mesh_xy, P("x", "y", None))
        return jax.lax.with_sharding_constraint(diag_h, diag_sharding)

    diag_full = jnp.stack([diag_h, -diag_h], axis=0)[:, None, ...]
    diag_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y", None))
    return jax.lax.with_sharding_constraint(diag_full, diag_sharding)


def _get_gmres_solver(
    matvec,
    data: dict,
    max_iter: int,
    tol: float,
    dtype: jnp.dtype,
) -> Callable:
    """Return a cached JIT-compiled GMRES solver for this max_iter/tol."""
    key = (max_iter, float(tol), str(dtype))
    if key in _GMRES_SOLVER_CACHE:
        return _GMRES_SOLVER_CACHE[key]

    def _solve(b, diag_h, z):
        one = jnp.asarray(1.0, dtype=b.dtype)
        m_inv = one / (z - diag_h)
        if m_inv.ndim == b.ndim - 1:
            m_inv = m_inv[None, ...]

        x0 = m_inv * b
        r0 = b - _apply_shifted_matvec(matvec, x0, z, data).astype(b.dtype)
        beta = jnp.linalg.norm(r0)

        zero = jnp.asarray(0.0, dtype=beta.dtype)
        v0 = jnp.where(beta == zero, r0, r0 / beta)

        v_shape = (max_iter + 1,) + b.shape
        z_shape = (max_iter,) + b.shape
        V = jnp.zeros(v_shape, dtype=b.dtype).at[0].set(v0)
        Z = jnp.zeros(z_shape, dtype=b.dtype)
        H = jnp.zeros((max_iter + 1, max_iter), dtype=b.dtype)
        g = jnp.zeros((max_iter + 1,), dtype=b.dtype).at[0].set(beta)
        y = jnp.zeros((max_iter,), dtype=b.dtype)

        def cond(state):
            k, rel, *_ = state
            return jnp.logical_and(k < max_iter, rel > tol)

        def body(state):
            k, rel, V, Z, H, g, y = state

            v_k = V[k]
            z_k = m_inv * v_k
            Z = Z.at[k].set(z_k)
            w = _apply_shifted_matvec(matvec, z_k, z, data).astype(b.dtype)

            def arnoldi(i, carry):
                w_local, H_local = carry
                h = jnp.vdot(V[i], w_local)
                H_local = H_local.at[i, k].set(h)
                w_local = w_local - h * V[i]
                return w_local, H_local

            w, H = jax.lax.fori_loop(0, k + 1, arnoldi, (w, H))
            h_next = jnp.linalg.norm(w)
            H = H.at[k + 1, k].set(h_next)
            v_next = jnp.where(h_next == 0.0, w, w / h_next)
            V = V.at[k + 1].set(v_next)

            lhs = H.conj().T @ H
            rhs = H.conj().T @ g
            jitter_scale = jnp.asarray(1e-14, dtype=lhs.real.dtype)
            jitter = jitter_scale * jnp.trace(lhs).real / jnp.maximum(
                jnp.asarray(1.0, dtype=lhs.real.dtype), lhs.shape[0]
            )
            lhs = lhs + jitter * jnp.eye(lhs.shape[0], dtype=lhs.dtype)
            y = jnp.linalg.solve(lhs, rhs)
            resid = jnp.linalg.norm(g - H @ y)
            rel = jnp.where(beta == zero, zero, resid / beta)

            return k + 1, rel, V, Z, H, g, y

        rel0 = jnp.asarray(jnp.inf, dtype=beta.dtype)
        init = (0, rel0, V, Z, H, g, y)
        k_final, _, V, Z, H, g, y = jax.lax.while_loop(cond, body, init)

        x = x0 + jnp.tensordot(y, Z, axes=(0, 0))
        return x, k_final

    solver = jax.jit(_solve)
    _GMRES_SOLVER_CACHE[key] = solver
    return solver


def gmres_solve_sharded_jit(
    matvec,
    diag_h: jax.Array,
    z: complex,
    b: jax.Array,
    data: dict,
    max_iter: int,
    tol: float,
) -> tuple[jax.Array, jax.Array]:
    """JIT GMRES with diagonal right-preconditioner and while-loop stopping."""
    solver = _get_gmres_solver(matvec, data, max_iter, tol, b.dtype)
    return solver(b, diag_h, z)


def _get_feast_runner(
    matvec,
    data: dict,
    n_quad: int,
    n_ritz: int,
    max_iter: int,
    tol: float,
    ry_to_ev: float,
    dtype: jnp.dtype,
    use_conjugate_symmetry: bool = True,
) -> Callable:
    """Return a cached JIT FEAST runner for this (n_quad, n_ritz, max_iter, tol)."""
    key = (n_quad, n_ritz, max_iter, float(tol), float(ry_to_ev), str(dtype), bool(use_conjugate_symmetry))
    if key in _FEAST_RUNNER_CACHE:
        return _FEAST_RUNNER_CACHE[key]

    def _run(X_batch, z_nodes, w_weights, diag_h):
        # X_batch: (n_ritz, 1, nc, nv, nk)
        filtered = jnp.zeros_like(X_batch, dtype=X_batch.dtype)
        iters = jnp.zeros((n_ritz, n_quad), dtype=jnp.int32)
        scale = jnp.asarray(ry_to_ev, dtype=z_nodes.dtype)

        def vec_body(i, carry):
            filtered_local, iters_local = carry
            x = X_batch[i]

            def pole_body(j, pole_carry):
                filt_i, iters_i = pole_carry
                z = z_nodes[j] / scale
                w = w_weights[j] / scale
                y, k_used = gmres_solve_sharded_jit(
                    matvec, diag_h, z, x, data, max_iter=max_iter, tol=tol
                )
                if use_conjugate_symmetry:
                    two = jnp.asarray(2.0, dtype=jnp.real(w).dtype)
                    filt_i = filt_i + two * jnp.real(w * y)
                else:
                    filt_i = filt_i + w * y
                iters_i = iters_i.at[j].set(k_used)
                return filt_i, iters_i

            filt_i = jnp.zeros_like(x, dtype=x.dtype)
            iters_i = jnp.zeros((n_quad,), dtype=jnp.int32)
            filt_i, iters_i = jax.lax.fori_loop(0, n_quad, pole_body, (filt_i, iters_i))
            filtered_local = filtered_local.at[i].set(filt_i)
            iters_local = iters_local.at[i].set(iters_i)
            return filtered_local, iters_local

        filtered, iters = jax.lax.fori_loop(0, n_ritz, vec_body, (filtered, iters))
        return filtered, iters

    runner = jax.jit(_run)
    _FEAST_RUNNER_CACHE[key] = runner
    return runner


def _cast_with_sharding(x: jax.Array, dtype: jnp.dtype) -> jax.Array:
    y = x.astype(dtype)
    sharding = getattr(x, "sharding", None)
    if sharding is not None:
        y = jax.lax.with_sharding_constraint(y, sharding)
    return y


def _build_gmres_data_fp32(data: dict) -> dict:
    return {
        "psi_c_X": _cast_with_sharding(data["psi_c_X"], jnp.complex64),
        "psi_c_Y": _cast_with_sharding(data["psi_c_Y"], jnp.complex64),
        "psi_v_X": _cast_with_sharding(data["psi_v_X"], jnp.complex64),
        "psi_v_Y": _cast_with_sharding(data["psi_v_Y"], jnp.complex64),
        "eps_c": _cast_with_sharding(data["eps_c"], jnp.float32),
        "eps_v": _cast_with_sharding(data["eps_v"], jnp.float32),
        "V_q0": _cast_with_sharding(data["V_q0"], jnp.complex64),
        "W_q": _cast_with_sharding(data["W_q"], jnp.complex64),
    }


@dataclass(frozen=True)
class RitzResult:
    """Result of a Rayleigh-Ritz extraction from a FEAST-filtered subspace."""
    ritz_evals: np.ndarray          # (n,) Ritz values (Ry), sorted
    ritz_coeffs: np.ndarray         # (n, n) coefficient matrix for Ritz vectors (columns)
    s_evals: np.ndarray             # (n,) overlap eigenvalues, sorted ascending
    rayleigh_quotients: np.ndarray  # (n,) per-vector Rayleigh quotients (Ry), sorted
    rel_residuals: np.ndarray       # (n,) ||H psi - lambda psi|| / ||psi|| in full space
    n_total: int                    # number of FEAST vectors in the subspace
    s_floor: float                  # overlap regularization floor used for whitening


def _rayleigh_ritz(
    matvec,
    vectors: list[jax.Array],
    data: dict,
    s_cutoff: float = 1e-6,
    use_tda: bool = True,
) -> RitzResult:
    """Solve the reduced generalized EVP with overlap regularization.

    We regularize the overlap by clipping overlap eigenvalues below
    `s_floor = s_cutoff * max(s_evals)` (plus a tiny absolute floor), rather
    than truncating the subspace dimension. Unphysical eigenpairs are best
    identified after the fact (e.g. outside the target FEAST window and/or
    with large full-space residual).
    """
    n = len(vectors)
    if n == 0:
        return RitzResult(
            ritz_evals=np.array([], dtype=np.float64),
            ritz_coeffs=np.zeros((0, 0), dtype=np.complex128),
            s_evals=np.array([], dtype=np.float64),
            rayleigh_quotients=np.array([], dtype=np.float64),
            rel_residuals=np.array([], dtype=np.float64),
            n_total=0,
            s_floor=0.0,
        )

    args = (data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
            data["eps_c"], data["eps_v"], data["W_R"], data["V_q0"])
    if use_tda:
        # TDA subspace application through the trial-stack matvec: the filtered
        # vectors are (1, nc, nv, nk), so concatenate on the leading axis into a
        # single (n, nc, nv, nk) stack and apply H ONCE — one dispatch, one
        # compiled program, one T-tensor (vs the old per-vector Python loop).
        HV_batch = matvec(jnp.concatenate(vectors, axis=0), *args)
        hv = [HV_batch[i:i + 1] for i in range(len(vectors))]
    else:
        # Non-TDA carries the [X, Y] pair axis (build_bse_ring_matvec_full);
        # not covered by the stack matvec — keep the per-vector application.
        hv = [matvec(v, *args) for v in vectors]

    V = jnp.stack(vectors, axis=0)
    HV = jnp.stack(hv, axis=0)

    V_flat = V.reshape((n, -1))
    HV_flat = HV.reshape((n, -1))

    S_dev = V_flat.conj() @ V_flat.T
    H_dev = V_flat.conj() @ HV_flat.T
    G_dev = HV_flat.conj() @ HV_flat.T

    S_np = np.asarray(jax.device_get(S_dev))
    H_np = np.asarray(jax.device_get(H_dev))
    G_np = np.asarray(jax.device_get(G_dev))
    S_np = 0.5 * (S_np + S_np.conj().T)
    if use_tda:
        H_np = 0.5 * (H_np + H_np.conj().T)
        G_np = 0.5 * (G_np + G_np.conj().T)

    s_evals, s_evecs = np.linalg.eigh(S_np)
    s_evals = s_evals.real
    s_max = float(np.max(s_evals)) if s_evals.size else 0.0
    s_floor = max(float(s_cutoff) * s_max, 1e-30)
    s_clipped = np.maximum(s_evals, s_floor)

    rq = np.real(np.diag(H_np) / np.maximum(np.real(np.diag(S_np)), 1e-30))
    rq = np.sort(rq)

    if s_max <= 0.0:
        return RitzResult(
            ritz_evals=np.array([], dtype=np.float64),
            ritz_coeffs=np.zeros((n, 0), dtype=np.complex128),
            s_evals=s_evals,
            rayleigh_quotients=rq,
            n_total=n,
            rel_residuals=np.full((0,), np.inf, dtype=np.float64),
            s_floor=float(s_floor),
        )

    inv_sqrt = 1.0 / np.sqrt(s_clipped)
    W = s_evecs @ np.diag(inv_sqrt)  # whitening map: coeffs -> S-normed basis
    A = W.conj().T @ H_np @ W
    if use_tda:
        A = 0.5 * (A + A.conj().T)
        evals, evecs = np.linalg.eigh(A)
        order = np.argsort(evals.real)
        evals = evals.real[order]
        evecs = evecs[:, order]
    else:
        evals, evecs = np.linalg.eig(A)
        order = np.argsort(evals.real)
        evals = evals[order]
        evals = evals.real
        evecs = evecs[:, order]
    coeffs = W @ evecs  # (n, n)

    rel_res = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        lam = float(evals[i])
        c = coeffs[:, i]
        psi2 = np.real(np.vdot(c, S_np @ c))
        psi2 = max(psi2, 1e-300)
        r2 = np.real(np.vdot(c, (G_np - 2.0 * lam * H_np + (lam * lam) * S_np) @ c))
        r2 = max(r2, 0.0)
        rel_res[i] = math.sqrt(r2 / psi2)

    return RitzResult(
        ritz_evals=evals,
        ritz_coeffs=coeffs,
        s_evals=s_evals,
        rayleigh_quotients=rq,
        n_total=n,
        rel_residuals=rel_res,
        s_floor=float(s_floor),
    )


def _build_ritz_vectors(
    filtered: list[jax.Array],
    coeffs: np.ndarray,
    sharding,
    use_tda: bool = True,
) -> list[jax.Array]:
    """Reconstruct Ritz vectors from filtered vectors and coefficient matrix.

    Parameters
    ----------
    filtered : list of jax.Array
        The n_total filtered vectors from FEAST contour integration.
    coeffs : np.ndarray, shape (n_total, n_physical)
        Column i gives the linear combination weights for Ritz vector i.
    sharding
        Sharding constraint to apply to output vectors.

    Returns
    -------
    list of jax.Array
        n_physical normalized Ritz vectors.
    """
    n_physical = coeffs.shape[1]
    ritz_vecs = []
    for i in range(n_physical):
        c = coeffs[:, i]
        if use_tda:
            v = sum(float(c[j].real) * filtered[j] for j in range(len(filtered)))
            v = v.real  # BSE-TDA eigenvectors are real
        else:
            v = sum(c[j] * filtered[j] for j in range(len(filtered)))
        norm = jnp.linalg.norm(v)
        v = jnp.where(norm > 0, v / norm, v)
        v = jax.lax.with_sharding_constraint(v, sharding)
        ritz_vecs.append(v)
    return ritz_vecs


def run_feast_ritz(
    data: dict,
    mesh_xy: Mesh,
    windows: list[WindowSpec],
    n_quad: int,
    gamma: float,
    n_ritz: int,
    gmres_max_iter: int,
    gmres_tol: float,
    seed: int,
    ry_to_ev: float = RY_TO_EV_DEFAULT,
    s_cutoff: float = 1e-6,
    feast_iter: int = 1,
    quadrature: str = "zolotarev",
    lambda_min_eV: float | None = None,
    lambda_max_eV: float | None = None,
    zolotarev_rho_scale: float = 1.0,
    n_quad_schedule: list[int] | None = None,
    gmres_fp32: bool = False,
    include_W: bool = True,
    use_tda: bool = True,
) -> dict:
    if use_tda:
        # TDA FEAST (shifted-GMRES contour solves + Rayleigh-Ritz) runs on the
        # trial-stack matvec — a bit-exact, dtype-adaptive drop-in for the ring
        # matvec that also batches the Ritz subspace application in one dispatch.
        matvec = build_bse_stack_matvec(
            mesh_xy,
            data["nkx"],
            data["nky"],
            data["nkz"],
            kernel="bse" if include_W else "rpa",
        )
    else:
        matvec = build_bse_ring_matvec_full(
            mesh_xy,
            data["nkx"],
            data["nky"],
            data["nkz"],
            include_W=include_W,
        )
    sh = make_bse_shardings(mesh_xy)

    ensure_W_R(data, include_W)

    diag_h = build_preconditioner_diagonal_sharded(data, mesh_xy, include_W=include_W, use_tda=use_tda)
    data_gmres = data
    diag_h_gmres = diag_h
    runner_dtype = jnp.complex128
    if gmres_fp32:
        data_gmres = ensure_W_R(_build_gmres_data_fp32(data), include_W)
        diag_h_gmres = _cast_with_sharding(diag_h, jnp.float32)
        runner_dtype = jnp.complex64
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    n_cond_pad = int(data["n_cond_pad"])
    n_val_pad = int(data["n_val_pad"])

    key = jax.random.PRNGKey(seed)
    results = {}

    if n_quad_schedule is None:
        quad_schedule = [n_quad] * feast_iter
    else:
        quad_schedule = list(n_quad_schedule)
        if len(quad_schedule) < feast_iter:
            quad_schedule = quad_schedule + [quad_schedule[-1]] * (feast_iter - len(quad_schedule))
        quad_schedule = quad_schedule[:feast_iter]

    runner_cache: dict[int, Callable] = {}
    for nq in sorted(set(quad_schedule)):
        n_quad_total = nq if use_tda else 2 * nq
        runner_cache[n_quad_total] = _get_feast_runner(
            matvec,
            data_gmres,
            n_quad_total,
            n_ritz,
            gmres_max_iter,
            gmres_tol,
            ry_to_ev,
            runner_dtype,
            use_conjugate_symmetry=use_tda,
        )

    for window in windows:
        # Initial random starting vectors.
        X_list = []
        for _ in range(n_ritz):
            key, subkey = jax.random.split(key)
            x_dtype = jnp.float32 if gmres_fp32 else jnp.float64
            if use_tda:
                x = jax.random.normal(subkey, (1, n_cond_pad, n_val_pad, nk), dtype=x_dtype)
                x = x / jnp.linalg.norm(x)
                x = jax.lax.with_sharding_constraint(x, sh.X)
            else:
                subkey_x, subkey_y = jax.random.split(subkey)
                x0 = jax.random.normal(subkey_x, (1, n_cond_pad, n_val_pad, nk), dtype=x_dtype)
                x1 = jax.random.normal(subkey_y, (1, n_cond_pad, n_val_pad, nk), dtype=x_dtype)
                x = jnp.stack([x0, x1], axis=0)
                x = x / jnp.linalg.norm(x)
                x = jax.lax.with_sharding_constraint(x, sh.X_full)
            X_list.append(x.astype(runner_dtype))
        X_batch = jnp.stack(X_list, axis=0)

        ritz_result = None
        prev_evals = None

        for it in range(feast_iter):
            is_last = (it == feast_iter - 1)
            n_quad_it = quad_schedule[it]
            if use_tda:
                print(f"\n  {window.name} FEAST iteration {it+1}/{feast_iter} (n_quad={n_quad_it})")
            else:
                print(f"\n  {window.name} FEAST iteration {it+1}/{feast_iter} (n_quad={n_quad_it} x2 full contour)")

            if quadrature == "zolotarev":
                if lambda_min_eV is None or lambda_max_eV is None:
                    raise ValueError(
                        "Zolotarev quadrature requires spectral bounds "
                        "(lambda_min_eV, lambda_max_eV)"
                    )
                z_nodes, w_weights = feast_zolotarev_quadrature(
                    window, n_quad_it, lambda_min_eV, lambda_max_eV, rho_scale=zolotarev_rho_scale
                )
            elif quadrature == "ellipse":
                z_nodes, w_weights = feast_ellipse_quadrature(window, n_quad_it, gamma)
            else:
                raise ValueError(f"Unknown quadrature type: {quadrature!r}")
            if not use_tda:
                z_nodes = np.concatenate([z_nodes, np.conj(z_nodes)])
                w_weights = np.concatenate([w_weights, np.conj(w_weights)])
            z_dtype = jnp.complex64 if gmres_fp32 else jnp.complex128
            z_jnp = jnp.asarray(z_nodes, dtype=z_dtype)
            w_jnp = jnp.asarray(w_weights, dtype=z_dtype)

            # --- Filter ---
            n_quad_total = z_nodes.shape[0]
            filtered_batch, iters = runner_cache[n_quad_total](X_batch, z_jnp, w_jnp, diag_h_gmres)

            it_sum = int(jax.device_get(jnp.sum(iters)))
            it_max = int(jax.device_get(jnp.max(iters)))
            n_solves = n_ritz * n_quad_total
            total_matvecs = it_sum + n_solves  # 1 matvec per GMRES iter + 1 initial matvec per solve
            print(
                f"    GMRES: solves={n_solves}, total_iters={it_sum}, max_iters={it_max}, "
                f"total_matvecs={total_matvecs}"
            )

            filtered = [filtered_batch[i] for i in range(n_ritz)]
            if gmres_fp32:
                filtered = [v.astype(jnp.complex128) for v in filtered]

            print(f"  {window.name} Rayleigh-Ritz [{window.a_eV:.2f}, {window.b_eV:.2f}] eV:")
            ritz_result = _rayleigh_ritz(
                matvec, filtered, data,
                s_cutoff=s_cutoff,
                use_tda=use_tda,
            )

            evals_eV = ritz_result.ritz_evals * ry_to_ev
            in_window = (evals_eV >= window.a_eV) & (evals_eV <= window.b_eV)
            n_in = int(np.sum(in_window))

            s_max = float(ritz_result.s_evals.max()) if ritz_result.s_evals.size else 0.0
            eff_rank = int(np.sum(ritz_result.s_evals >= ritz_result.s_floor)) if s_max > 0 else 0
            print(f"    overlap: max(S)={s_max:.3e}, s_floor={ritz_result.s_floor:.3e}, eff_rank={eff_rank}/{n_ritz}")
            print(f"    Ritz values (eV) [{n_in} in-window / {n_ritz} total]:")
            for i, (ev, rr, ok) in enumerate(zip(evals_eV, ritz_result.rel_residuals, in_window), start=1):
                tag = "" if ok else " *"
                print(f"      {i:2d}: {ev:12.6f}   relres={rr:9.2e}{tag}")

            # Convergence check: max change from previous iteration.
            if prev_evals is not None:
                n_cmp = min(len(prev_evals), len(ritz_result.ritz_evals))
                if n_cmp > 0:
                    delta = np.abs(ritz_result.ritz_evals[:n_cmp] - prev_evals[:n_cmp])
                    max_delta_eV = np.max(delta) * ry_to_ev
                    print(f"    Max eigenvalue change: {max_delta_eV:.6e} eV")
            prev_evals = ritz_result.ritz_evals.copy()

            # --- Prepare starting vectors for next iteration ---
            if not is_last:
                # Prefer in-window eigenvectors; fall back to those closest to the window center.
                if n_in > 0:
                    choose = np.where(in_window)[0]
                else:
                    center = 0.5 * (window.a_eV + window.b_eV)
                    choose = np.argsort(np.abs(evals_eV - center))
                choose = choose[: min(len(choose), n_ritz)]
                coeffs_sel = ritz_result.ritz_coeffs[:, choose]
                ritz_vecs = _build_ritz_vectors(filtered, coeffs_sel, sh.X if use_tda else sh.X_full, use_tda=use_tda)

                # Pad back to n_ritz with random vectors if n_physical < n_ritz.
                X_next = []
                for v in ritz_vecs:
                    X_next.append(v.astype(runner_dtype))
                while len(X_next) < n_ritz:
                    key, subkey = jax.random.split(key)
                    x_dtype = jnp.float32 if gmres_fp32 else jnp.float64
                    if use_tda:
                        x = jax.random.normal(subkey, (1, n_cond_pad, n_val_pad, nk), dtype=x_dtype)
                        x = x / jnp.linalg.norm(x)
                        x = jax.lax.with_sharding_constraint(x, sh.X)
                    else:
                        subkey_x, subkey_y = jax.random.split(subkey)
                        x0 = jax.random.normal(subkey_x, (1, n_cond_pad, n_val_pad, nk), dtype=x_dtype)
                        x1 = jax.random.normal(subkey_y, (1, n_cond_pad, n_val_pad, nk), dtype=x_dtype)
                        x = jnp.stack([x0, x1], axis=0)
                        x = x / jnp.linalg.norm(x)
                        x = jax.lax.with_sharding_constraint(x, sh.X_full)
                    X_next.append(x.astype(runner_dtype))
                X_batch = jnp.stack(X_next, axis=0)

        # Final diagnostics for this window (printed after last iteration).
        assert ritz_result is not None
        results[window.name] = ritz_result

    return results

def _create_mesh_xy(px: int, py: int) -> Mesh:
    devices = jax.devices()
    n_devices = len(devices)
    if px * py > n_devices:
        raise ValueError(f"Requested px*py={px*py} devices, but only {n_devices} available")
    mesh_devices = np.array(devices[: px * py]).reshape(px, py)
    return Mesh(mesh_devices, axis_names=("x", "y"))


def estimate_spectral_bounds_sharded(
    data: dict,
    mesh_xy: Mesh,
    n_lanczos: int = 20,
    n_lanczos_max: int | None = None,
    emax_rtol: float = 5e-4,
    emax_atol: float = 0.0,
    emax_patience: int = 2,
    seed: int = 0,
    include_W: bool = True,
    use_tda: bool = True,
) -> dict:
    """Estimate E_min and E_max (Ry) using diagonal gap and short Lanczos.

    Uses an adaptive Lanczos run that stops when the largest Ritz value
    (of the tridiagonal) converges to within the specified tolerances.
    """
    data_fp32 = ensure_W_R(_build_gmres_data_fp32(data), include_W)

    eps_c = data_fp32["eps_c"]
    eps_v = data_fp32["eps_v"]
    n_cond = int(data["n_cond"])
    n_val = int(data["n_val"])

    eps_c_use = eps_c[:, :n_cond]
    eps_v_use = eps_v[:, :n_val]
    eps_c_min = jnp.min(eps_c_use)
    eps_v_max = jnp.max(eps_v_use)
    e_min_ry = (eps_c_min - eps_v_max).real

    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    n_cond_pad = int(data["n_cond_pad"])
    n_val_pad = int(data["n_val_pad"])

    sh = make_bse_shardings(mesh_xy)
    matvec = build_bse_ring_matvec(
        mesh_xy,
        data["nkx"],
        data["nky"],
        data["nkz"],
        include_W=include_W,
    )

    key = jax.random.PRNGKey(seed)

    @jax.jit
    def _make_random_vector(key_in):
        k1, k2 = jax.random.split(key_in)
        q = jax.random.normal(k1, (1, n_cond_pad, n_val_pad, nk), dtype=jnp.float32)
        q = q + 1j * jax.random.normal(k2, (1, n_cond_pad, n_val_pad, nk), dtype=jnp.float32)
        q = q / jnp.linalg.norm(q)
        return jax.lax.with_sharding_constraint(q, sh.X)

    def _tridiag_emax(alpha_seq: list[float], beta_seq: list[float]) -> float:
        t_dim = len(alpha_seq)
        T = np.zeros((t_dim, t_dim), dtype=np.float64)
        for i, alpha in enumerate(alpha_seq):
            T[i, i] = alpha
            if i < t_dim - 1:
                T[i, i + 1] = beta_seq[i]
                T[i + 1, i] = beta_seq[i]
        evals = np.linalg.eigvalsh(T)
        return float(np.max(evals))

    if n_lanczos_max is None:
        n_lanczos_max = n_lanczos

    q = _make_random_vector(key)
    q_prev = jnp.zeros_like(q)
    beta_prev = 0.0

    alphas: list[float] = []
    betas: list[float] = []
    emax_seq: list[float] = []
    converged = False
    stable = 0
    prev_emax: float | None = None

    for _ in range(n_lanczos_max):
        z = matvec(
            q,
            data_fp32["psi_c_X"],
            data_fp32["psi_c_Y"],
            data_fp32["psi_v_X"],
            data_fp32["psi_v_Y"],
            eps_c,
            eps_v,
            data_fp32["W_R"],
            data_fp32["V_q0"],
        )

        alpha = jnp.vdot(q, z).real
        z = z - alpha * q - beta_prev * q_prev
        beta = jnp.linalg.norm(z)

        alpha_f = float(jax.device_get(alpha))
        beta_f = float(jax.device_get(beta))
        alphas.append(alpha_f)
        betas.append(beta_f)

        if beta_f == 0.0:
            converged = True
            break

        q_prev = q
        q = z / beta
        beta_prev = beta

        if len(alphas) >= n_lanczos:
            emax = _tridiag_emax(alphas, betas)
            emax_seq.append(emax)
            if prev_emax is not None:
                delta = abs(emax - prev_emax)
                thresh = float(emax_atol) + float(emax_rtol) * abs(emax)
                if delta <= thresh:
                    stable += 1
                else:
                    stable = 0
            prev_emax = emax
            if stable >= emax_patience:
                converged = True
                break

    if not emax_seq:
        emax_seq.append(_tridiag_emax(alphas, betas))
    e_max_ry = float(emax_seq[-1])
    if not use_tda:
        e_min_ry = -abs(e_max_ry)

    # Count diagonal elements (non-interacting transitions) as a function of energy.
    delta_E = energy_diff_cv_k(eps_c_use, eps_v_use)  # (n_cond, n_val, nk)
    diag_flat = np.asarray(jax.device_get(delta_E.real)).ravel()
    if not use_tda:
        diag_flat = np.concatenate([diag_flat, -diag_flat])

    return {
        "e_min_ry": float(jax.device_get(e_min_ry)),
        "e_max_ry_raw": e_max_ry,
        "n_lanczos_used": len(alphas),
        "n_lanczos_min": int(n_lanczos),
        "n_lanczos_max": int(n_lanczos_max),
        "emax_rtol": float(emax_rtol),
        "emax_atol": float(emax_atol),
        "emax_patience": int(emax_patience),
        "emax_converged": bool(converged),
        "emax_seq": np.array(emax_seq, dtype=np.float64),
        "diag_energies_ry": diag_flat,
    }


def build_default_windows_eV(e_max_eV: float) -> list[WindowSpec]:
    if e_max_eV < 2.0:
        return [WindowSpec("W1", 0.0, float(e_max_eV), "collapsed (Emax < 2 eV)")]
    return [
        WindowSpec("W1", 0.0, 2.0, "low-energy check window"),
        WindowSpec("W2", 2.0, float(e_max_eV), "bulk spectrum"),
    ]


def _zolotarev_step_poles_weights(
    edge: float,
    n_poles: int,
    lambda_min: float,
    lambda_max: float,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Zolotarev poles/weights for a single step function at `edge`.

    Returns (z_nodes, w_weights) such that:
        h_step(lambda) = 1/2 + sum_j 2*Re[w_j / (z_j - lambda)]
    approximates Heaviside(lambda - edge) on [lambda_min, lambda_max].

    rho controls the transition width: the step transitions over
    [edge - rho, edge + rho] in physical units.
    """
    from scipy.special import ellipk, ellipj

    t_min = (lambda_min - edge) / rho  # negative
    t_max = (lambda_max - edge) / rho  # positive
    G = max(abs(t_min), t_max)
    G = max(G, 1.01)  # ensure G > 1

    epsilon = 1.0 / G
    m = epsilon ** 2
    mp = 1.0 - m

    Kp_val = ellipk(mp)
    n = n_poles

    d = np.zeros(n)
    for j in range(1, n + 1):
        u = (2 * j - 1) * Kp_val / (2 * n)
        sn_val, cn_val, _, _ = ellipj(u, mp)
        d[j - 1] = epsilon ** 2 * sn_val ** 2 / cn_val ** 2

    c_zeros = np.zeros(max(n - 1, 0))
    for j in range(1, n):
        u = 2 * j * Kp_val / (2 * n)
        sn_val, cn_val, _, _ = ellipj(u, mp)
        c_zeros[j - 1] = epsilon ** 2 * sn_val ** 2 / cn_val ** 2

    if n == 1:
        A = 1.0 + d[0]
    else:
        A = np.prod(1.0 + d) / np.prod(1.0 + c_zeros)

    beta = np.zeros(n)
    for j in range(n):
        num = A * np.prod(c_zeros - d[j]) if len(c_zeros) > 0 else A
        den = np.prod(np.delete(d, j) - d[j]) if n > 1 else 1.0
        beta[j] = num / den

    z_nodes = edge + 1j * rho * np.sqrt(d)
    w_weights = -beta * rho / 4.0
    return z_nodes.astype(np.complex128), w_weights.astype(np.complex128)


def feast_zolotarev_quadrature(
    window: WindowSpec,
    n_quad: int,
    lambda_min_eV: float,
    lambda_max_eV: float,
    rho_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zolotarev optimal rational filter quadrature for FEAST.

    Builds an indicator function on [a, b] from two Zolotarev step functions:
    h(lambda) = step_up(lambda, a) - step_up(lambda, b). The constant 1/2
    terms cancel, so no offset is needed in the FEAST accumulation loop.

    Parameters
    ----------
    window : WindowSpec
        Target eigenvalue interval [a_eV, b_eV].
    n_quad : int
        Total number of poles in the upper half-plane (split between edges).
    lambda_min_eV, lambda_max_eV : float
        Spectral bounds (eV) from Lanczos.

    Returns
    -------
    z_nodes : ndarray, shape (n_quad,), complex
        Quadrature nodes (shifts) in the upper half-plane (eV).
    w_weights : ndarray, shape (n_quad,), complex
        Quadrature weights (eV).

    References
    ----------
    Guttel, Polizzi, Tang, Viaud, "Zolotarev Quadrature Rules and Load
    Balancing for the FEAST Eigensolver", SIAM J. Sci. Comput. 37 (2015).
    """
    a = window.a_eV
    b = window.b_eV
    rho = rho_scale * 0.5 * (b - a)  # transition half-width = window half-width
    n_left = n_quad // 2
    n_right = n_quad - n_left

    z_L, w_L = _zolotarev_step_poles_weights(a, n_left, lambda_min_eV, lambda_max_eV, rho)
    z_R, w_R = _zolotarev_step_poles_weights(b, n_right, lambda_min_eV, lambda_max_eV, rho)

    # Indicator = step_up(a) - step_up(b): negate right-step weights.
    z_nodes = np.concatenate([z_L, z_R])
    w_weights = np.concatenate([w_L, -w_R])
    return z_nodes, w_weights


def feast_ellipse_quadrature(
    window: WindowSpec,
    n_quad: int,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """BSE-facing wrapper: converts WindowSpec to (center, half_width)."""
    center = 0.5 * (window.a_eV + window.b_eV)
    half_width = 0.5 * (window.b_eV - window.a_eV)
    return _feast_ellipse_quadrature_generic(center, half_width, n_quad, gamma)


def _format_complex_eV(z: complex) -> str:
    return f"{z.real:9.4f} {z.imag:+9.4f}i"


def format_feast_report(
    *,
    px: int,
    py: int,
    ry_to_ev: float,
    bounds: dict,
    buffer_factor: float,
    windows: Iterable[WindowSpec],
    quad_specs: Iterable[QuadratureSpec],
    include_W: bool = True,
    use_tda: bool = True,
) -> str:
    e_min_ry = bounds["e_min_ry"]
    e_max_ry_raw = bounds["e_max_ry_raw"]
    e_max_ry = e_max_ry_raw * buffer_factor

    e_min_eV = e_min_ry * ry_to_ev
    e_max_eV_raw = e_max_ry_raw * ry_to_ev
    e_max_eV = e_max_ry * ry_to_ev

    lines = []
    lines.append("=" * 60)
    lines.append("FEAST SETUP REPORT")
    lines.append("=" * 60)
    kernel_name = "BSE (D+V-W)" if include_W else "RPA (D+V)"
    lines.append(f"Backend     : Sharded BSE matvec (px, py = {px}, {py})")
    mode = "TDA" if use_tda else "full (non-TDA)"
    lines.append(f"Kernel      : {kernel_name}, {mode}")
    lines.append(f"Energies    : eps in Ry; reporting in Ry and eV (Ry->eV = {ry_to_ev})")
    lines.append("")
    lines.append("--- Spectral bounds (Lanczos) ---")
    lines.append(f"Lanczos steps          : {bounds['n_lanczos_used']} (min={bounds['n_lanczos_min']}, max={bounds['n_lanczos_max']})")
    lines.append(f"E_min (diag gap)       : {e_min_ry:10.6f} Ry = {e_min_eV:9.3f} eV")
    lines.append(f"E_max (Lanczos raw)    : {e_max_ry_raw:10.6f} Ry = {e_max_eV_raw:9.3f} eV")
    lines.append(f"Buffer factor          : {buffer_factor:.2f}")
    lines.append(f"E_max (buffered)       : {e_max_ry:10.6f} Ry = {e_max_eV:9.3f} eV")
    if not bounds.get("emax_converged", True):
        lines.append(f"WARNING: Lanczos E_max not converged (rtol={bounds['emax_rtol']}, atol={bounds['emax_atol']}).")
    lines.append("")
    lines.append("--- Windows (eV) ---")
    for w in windows:
        note = f" ({w.note})" if w.note else ""
        lines.append(f"{w.name}: [{w.a_eV:6.2f}, {w.b_eV:6.2f}] eV{note}")
    lines.append("")
    for spec in quad_specs:
        a = spec.window.a_eV
        b = spec.window.b_eV
        c = 0.5 * (a + b)
        r_x = 0.5 * (b - a)
        if spec.quadrature_type == "zolotarev":
            lines.append(f"--- FEAST filter (Zolotarev rational, n_quad={spec.n_quad}) ---")
            lines.append(f"{spec.window.name}: center={c:.3f} eV, half-width={r_x:.3f} eV")
        else:
            r_y = spec.gamma * r_x
            lines.append(f"--- FEAST contour (ellipse, trapezoidal) ---")
            lines.append(f"{spec.window.name}: n_quad = {spec.n_quad}, gamma = {spec.gamma:.3f}")
            lines.append(f"    center={c:.3f} eV, r_x={r_x:.3f} eV, r_y={r_y:.3f} eV")
        if use_tda:
            lines.append("    z_j (eV) and w_j (eV)   [upper half-plane]")
        else:
            lines.append("    z_j (eV) and w_j (eV)   [upper half-plane; lower half-plane added]")
        for j, (z, w) in enumerate(zip(spec.z_nodes, spec.w_weights), start=1):
            lines.append(f"    j={j:2d}: z={_format_complex_eV(z)}, w={_format_complex_eV(w)}")
        width = b - a
        sample = [
            max(0.0, a - 0.1 * width),
            a + 0.1 * width,
            0.5 * (a + b),
            b - 0.1 * width,
            b + 0.1 * width,
        ]
        lines.append("    filter response f(E) = 2 Re Σ w/(z-E):")
        for E in sample:
            denom = spec.z_nodes - E
            fE = 2.0 * np.real(np.sum(spec.w_weights / denom))
            lines.append(f"      E={E:8.3f} eV -> f(E)={fE:8.4f}")
    lines.append("=" * 60)

    return "\n".join(lines)


def _parse_window_arg(values: list[str], default: Tuple[float, float]) -> Tuple[float, float]:
    if not values:
        return default
    if len(values) != 2:
        raise ValueError("Window must have exactly two values: a b")
    a = float(values[0])
    b_str = values[1].lower()
    if b_str == "auto":
        return a, math.nan
    return a, float(b_str)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FEAST setup for sharded BSE")
    parser.add_argument("-i", "--input", required=True, help="COHSEX input file")
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--n-cond", type=int, default=4)
    parser.add_argument("--px", type=int, default=1)
    parser.add_argument("--py", type=int, default=1)
    parser.add_argument("--n-lanczos", type=int, default=10,
                        help="Minimum Lanczos steps for spectral bounds.")
    parser.add_argument("--n-lanczos-max", type=int, default=50,
                        help="Maximum Lanczos steps for spectral bounds.")
    parser.add_argument("--lanczos-rtol", type=float, default=5e-4,
                        help="Relative convergence tolerance for Lanczos E_max.")
    parser.add_argument("--lanczos-atol", type=float, default=0.0,
                        help="Absolute convergence tolerance for Lanczos E_max.")
    parser.add_argument("--lanczos-patience", type=int, default=2,
                        help="Consecutive steps meeting tolerance before stopping.")
    parser.add_argument("--buffer", type=float, default=0.05)
    parser.add_argument("--n-quad1", type=int, default=4,
                        help="Quadrature points for FEAST iteration 1.")
    parser.add_argument("--n-quad2", type=int, default=8,
                        help="Quadrature points for FEAST iteration 2+.")
    parser.add_argument("--feast-ritz", action="store_true", help="Run FEAST GMRES + Ritz extraction.")
    parser.add_argument("--feast-ritz-count", type=int, default=4, help="Ritz values per window.")
    parser.add_argument("--rpa", action="store_true",
                        help="Use RPA kernel (D+V only), skip W0 term entirely.")
    parser.add_argument("--windows-kpm", action="store_true",
                        help="Derive FEAST windows from a KPM DOS partition.")
    parser.add_argument("--windows-kpm-count", type=int, default=4,
                        help="Number of KPM-derived windows (default: 4).")
    parser.add_argument("--kpm-n-moments", type=int, default=100,
                        help="KPM moments for window estimation.")
    parser.add_argument("--kpm-n-random", type=int, default=4,
                        help="KPM random vectors for window estimation.")
    parser.add_argument("--kpm-seed", type=int, default=0,
                        help="KPM RNG seed for window estimation.")
    parser.add_argument("--kpm-n-energy-pts", type=int, default=2000,
                        help="Energy grid size for KPM DOS.")
    parser.add_argument("--kpm-n-lanczos", type=int, default=100,
                        help="Lanczos steps for KPM spectral bounds.")
    parser.add_argument(
        "--s-cutoff",
        type=float,
        default=1e-6,
        help="Overlap regularization floor as a fraction of max(S) (default: 1e-6).",
    )
    parser.add_argument("--feast-iter", type=int, default=2,
                        help="Number of FEAST subspace iterations (re-filter Ritz vectors).")
    parser.add_argument("--quadrature", type=str, default="ellipse",
                        choices=["zolotarev", "ellipse"],
                        help="Quadrature type for FEAST filter (default: ellipse).")
    parser.add_argument("--gmres-max-iter", type=int, default=10, help="GMRES iterations per shift.")
    parser.add_argument("--gmres-tol", type=float, default=1e-2, help="GMRES relative tolerance.")
    parser.add_argument("--gmres-seed", type=int, default=0, help="Random seed for FEAST vectors.")
    parser.add_argument("--gmres-fp32", action="store_true",
                        help="Use FP32 data/GMRES for shifted solves.")
    parser.add_argument("--tda", action="store_true",
                        help="Use Tamm-Dancoff approximation (TDA). Default is full non-TDA.")
    parser.add_argument(
        "--units-ev-per-ry",
        type=float,
        default=RY_TO_EV_DEFAULT,
        help="Conversion factor Ry -> eV",
    )
    parser.add_argument(
        "--window1",
        nargs=2,
        default=None,
        metavar=("A", "B"),
        help="Override window 1 bounds in eV (use 'auto' for B)",
    )
    parser.add_argument(
        "--window2",
        nargs=2,
        default=None,
        metavar=("A", "B"),
        help="Override window 2 bounds in eV (use 'auto' for B)",
    )
    args = parser.parse_args(argv)

    timing.reset()

    # Enable JAX persistent compile cache before any jit compiles.
    try:
        from common.jax_compile_cache import ensure_jax_compile_cache
        ensure_jax_compile_cache()
    except Exception as _e:
        import jax as _jax_mod
        if _jax_mod.process_index() == 0:
            print(f"  [jax compile cache] skipped: {_e}", flush=True)

    mesh_xy = _create_mesh_xy(args.px, args.py)
    restart_file = _find_restart_file(args.input)
    with timing.section("feast.restart_load"):
        data = load_bse_data_from_restart_sharded(
            restart_file,
            n_val=args.n_val,
            n_cond=args.n_cond,
            mesh_xy=mesh_xy,
            input_file=args.input,
        )

    use_tda = args.tda

    with timing.section("feast.lanczos_bounds"):
        bounds = estimate_spectral_bounds_sharded(
            data,
            mesh_xy,
            n_lanczos=args.n_lanczos,
            n_lanczos_max=args.n_lanczos_max,
            emax_rtol=args.lanczos_rtol,
            emax_atol=args.lanczos_atol,
            emax_patience=args.lanczos_patience,
            include_W=not args.rpa,
            use_tda=use_tda,
        )

    e_min_ry = bounds["e_min_ry"]
    e_max_ry = bounds["e_max_ry_raw"] * (1.0 + args.buffer)
    e_min_eV = e_min_ry * args.units_ev_per_ry
    e_max_eV = e_max_ry * args.units_ev_per_ry

    windows = build_default_windows_eV(e_max_eV)

    if args.windows_kpm:
        from . import bse_kpm

        with timing.section("feast.kpm_windows"):
            kpm_result = bse_kpm.run_kpm_dos(
                data,
                mesh_xy,
                n_moments=args.kpm_n_moments,
                n_random=args.kpm_n_random,
                seed=args.kpm_seed,
                n_lanczos=args.kpm_n_lanczos,
                buffer=args.buffer,
                ry_to_ev=args.units_ev_per_ry,
                n_energy_pts=args.kpm_n_energy_pts,
                plot_file="bse_kpm_windows.png",
                n_windows=args.windows_kpm_count,
                emit_outputs=False,
                include_W=not args.rpa,
                use_tda=use_tda,
            )
        windows_ry = kpm_result["windows_ry"]
        windows = [
            WindowSpec(f"W{i+1}", float(a * args.units_ev_per_ry), float(b * args.units_ev_per_ry), "KPM-weighted")
            for i, (a, b) in enumerate(windows_ry)
        ]
    elif args.window1 is not None or args.window2 is not None:
        w1_default = (0.0, 2.0)
        w2_default = (2.0, math.nan)
        w1 = _parse_window_arg(args.window1 or [str(w1_default[0]), str(w1_default[1])], w1_default)
        w2 = _parse_window_arg(args.window2 or [str(w2_default[0]), "auto"], w2_default)

        w1_b = e_max_eV if math.isnan(w1[1]) else w1[1]
        w2_b = e_max_eV if math.isnan(w2[1]) else w2[1]

        windows = [
            WindowSpec("W1", float(w1[0]), float(w1_b), "user window"),
            WindowSpec("W2", float(w2[0]), float(w2_b), "user window"),
        ]

    quad_specs = []
    for w in windows:
        if args.quadrature == "zolotarev":
            z, wts = feast_zolotarev_quadrature(w, args.n_quad2, e_min_eV, e_max_eV)
        else:
            z, wts = feast_ellipse_quadrature(w, args.n_quad2, ELLIPSE_GAMMA_FIXED)
        quad_specs.append(QuadratureSpec(
            window=w, z_nodes=z, w_weights=wts, n_quad=args.n_quad2,
            gamma=ELLIPSE_GAMMA_FIXED, quadrature_type=args.quadrature,
        ))

    report = format_feast_report(
        px=args.px,
        py=args.py,
        ry_to_ev=args.units_ev_per_ry,
        bounds=bounds,
        buffer_factor=1.0 + args.buffer,
        windows=windows,
        quad_specs=quad_specs,
        include_W=not args.rpa,
        use_tda=use_tda,
    )
    print(report)

    # Diagnostic: count non-interacting transitions per window.
    diag_eV = bounds["diag_energies_ry"] * args.units_ev_per_ry
    print(f"--- Diagonal (non-interacting) transition count ---")
    print(f"BSE dimension: {len(diag_eV)}")
    for w in windows:
        count = int(np.sum((diag_eV >= w.a_eV) & (diag_eV <= w.b_eV)))
        print(f"  {w.name} [{w.a_eV:.2f}, {w.b_eV:.2f}] eV: {count} diagonal elements")

    if args.feast_ritz:
        print("\n--- FEAST Ritz extraction ---")
        n_quad_schedule = [args.n_quad1] + [args.n_quad2] * max(args.feast_iter - 1, 0)
        print(
            f"Quadrature: {args.quadrature}, "
            f"n_quad schedule: {' -> '.join(str(nq) for nq in n_quad_schedule)}, "
            f"gamma={ELLIPSE_GAMMA_FIXED:.2f}, "
            f"Ritz per window: {args.feast_ritz_count}, "
            f"GMRES iters: {args.gmres_max_iter}, tol: {args.gmres_tol}, "
            f"FEAST iters: {args.feast_iter}"
        )
        with timing.section("feast.window_solve"):
            results = run_feast_ritz(
                data,
                mesh_xy,
                windows,
                args.n_quad2,
                ELLIPSE_GAMMA_FIXED,
                args.feast_ritz_count,
                args.gmres_max_iter,
                args.gmres_tol,
                args.gmres_seed,
                args.units_ev_per_ry,
                s_cutoff=args.s_cutoff,
                feast_iter=args.feast_iter,
                quadrature=args.quadrature,
                lambda_min_eV=e_min_eV,
                lambda_max_eV=e_max_eV,
                n_quad_schedule=n_quad_schedule,
                gmres_fp32=args.gmres_fp32,
                include_W=not args.rpa,
                use_tda=use_tda,
            )
        wmap = {w.name: w for w in windows}
        for name, rr in results.items():
            w = wmap.get(name)
            if w is None:
                continue
            evals_eV = rr.ritz_evals * args.units_ev_per_ry
            in_window = (evals_eV >= w.a_eV) & (evals_eV <= w.b_eV)
            n_in = int(np.sum(in_window))
            print(f"\n{name} Ritz values (eV) [{n_in} in-window / {rr.n_total} total]:")
            for i, (ev, ok) in enumerate(zip(evals_eV, in_window), start=1):
                tag = "" if ok else " *"
                print(f"  {i:2d}: {ev:12.6f}{tag}")

    timing.report(print_fn=print, title="--- Timing ---")


if __name__ == "__main__":
    main()
