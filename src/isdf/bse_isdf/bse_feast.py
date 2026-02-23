"""FEAST setup utilities for sharded ISDF-BSE.

This module estimates spectral bounds with a short Lanczos run using the
sharded BSE matvec, defines simple windows in eV, and generates FEAST
ellipse-trapezoid quadrature nodes/weights. Output is printed to stdout
in a physics-style report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable

import math
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .bse_ring_comm import build_bse_ring_matvec, make_bse_shardings
from .bse_preconditioner import energy_diff_cv_k
import isdf.common.timing as timing
from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded

jax.config.update("jax_enable_x64", True)


RY_TO_EV_DEFAULT = 13.6056980659
ELLIPSE_GAMMA_FIXED = 0.2

# Cache compiled GMRES kernels by max_iter (per-process).
_GMRES_SOLVER_CACHE: dict[tuple[int, float], Callable] = {}
_FEAST_RUNNER_CACHE: dict[tuple[int, int, int, float], Callable] = {}


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


def build_preconditioner_diagonal_sharded(data: dict, mesh_xy: Mesh) -> jax.Array:
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

    W_q0 = data["W_q"][:, :, 0, 0, 0]
    rho_c = jnp.einsum("kcsm,kcsm->kcm", jnp.conj(psi_c_X), psi_c_X)
    rho_v = jnp.einsum("kvsm,kvsm->kvm", jnp.conj(psi_v_Y), psi_v_Y)
    S_w = jnp.einsum("MN,kvN->kvM", W_q0, rho_v)
    W_diag_kcv = jnp.einsum("kcm,kvm->kcv", rho_c, S_w) / nk

    V_diag = V_diag_kcv.transpose(1, 2, 0)
    W_diag = W_diag_kcv.transpose(1, 2, 0)
    delta_E = energy_diff_cv_k(eps_c, eps_v)

    diag_h = delta_E + V_diag - W_diag
    diag_sharding = NamedSharding(mesh_xy, P("x", "y", None))
    return jax.lax.with_sharding_constraint(diag_h, diag_sharding)


def _get_gmres_solver(
    matvec,
    data: dict,
    max_iter: int,
    tol: float,
) -> Callable:
    """Return a cached JIT-compiled GMRES solver for this max_iter/tol."""
    key = (max_iter, float(tol))
    if key in _GMRES_SOLVER_CACHE:
        return _GMRES_SOLVER_CACHE[key]

    def _solve(b, diag_h, z):
        m_inv = 1.0 / (z - diag_h)
        m_inv = m_inv[None, ...]

        x0 = m_inv * b
        r0 = b - _apply_shifted_matvec(matvec, x0, z, data)
        beta = jnp.linalg.norm(r0)

        v0 = jnp.where(beta == 0.0, r0, r0 / beta)

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
            w = _apply_shifted_matvec(matvec, z_k, z, data)

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
            jitter = 1e-14 * jnp.trace(lhs).real / jnp.maximum(1.0, lhs.shape[0])
            lhs = lhs + jitter * jnp.eye(lhs.shape[0], dtype=lhs.dtype)
            y = jnp.linalg.solve(lhs, rhs)
            resid = jnp.linalg.norm(g - H @ y)
            rel = jnp.where(beta == 0.0, 0.0, resid / beta)

            return k + 1, rel, V, Z, H, g, y

        init = (0, jnp.inf, V, Z, H, g, y)
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
    solver = _get_gmres_solver(matvec, data, max_iter, tol)
    return solver(b, diag_h, z)


def _get_feast_runner(
    matvec,
    data: dict,
    n_quad: int,
    n_ritz: int,
    max_iter: int,
    tol: float,
    ry_to_ev: float,
) -> Callable:
    """Return a cached JIT FEAST runner for this (n_quad, n_ritz, max_iter, tol)."""
    key = (n_quad, n_ritz, max_iter, float(tol), float(ry_to_ev))
    if key in _FEAST_RUNNER_CACHE:
        return _FEAST_RUNNER_CACHE[key]

    def _run(X_batch, z_nodes, w_weights, diag_h):
        # X_batch: (n_ritz, 1, nc, nv, nk)
        filtered = jnp.zeros_like(X_batch, dtype=jnp.complex128)
        iters = jnp.zeros((n_ritz, n_quad), dtype=jnp.int32)

        def vec_body(i, carry):
            filtered_local, iters_local = carry
            x = X_batch[i]

            def pole_body(j, pole_carry):
                filt_i, iters_i = pole_carry
                z = z_nodes[j] / ry_to_ev
                w = w_weights[j] / ry_to_ev
                y, k_used = gmres_solve_sharded_jit(
                    matvec, diag_h, z, x, data, max_iter=max_iter, tol=tol
                )
                filt_i = filt_i + 2.0 * jnp.real(w * y)
                iters_i = iters_i.at[j].set(k_used)
                return filt_i, iters_i

            filt_i = jnp.zeros_like(x, dtype=jnp.complex128)
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


@dataclass(frozen=True)
class RitzResult:
    """Result of a Rayleigh-Ritz extraction with S-eigenvalue filtering."""
    ritz_evals: np.ndarray      # physical Ritz values (Ry), sorted
    ritz_coeffs: np.ndarray     # (n_total, n_physical) coefficient matrix for Ritz vectors
    s_evals: np.ndarray         # all S eigenvalues, sorted ascending
    rayleigh_quotients: np.ndarray  # per-vector Rayleigh quotients (Ry), sorted
    n_physical: int             # number of physical Ritz pairs kept
    n_total: int                # total number of input vectors
    s_threshold: float          # threshold used for filtering


def _solve_reduced_evp(
    S: jax.Array,
    H: jax.Array,
    s_evals: jax.Array,
    s_evecs: jax.Array,
    n_keep: int,
) -> tuple[jax.Array, jax.Array]:
    """Solve the reduced generalized eigenvalue problem keeping n_keep S-eigenvectors.

    Returns (sorted_eigenvalues, coefficients) where coefficients has shape
    (n_total, n_keep) — column i gives the linear combination of the original
    vectors that produces Ritz vector i.
    """
    U = s_evecs[:, -n_keep:]          # largest n_keep S eigenvectors
    s_keep = s_evals[-n_keep:]
    H_red = U.conj().T @ H @ U
    D_inv_sqrt = jnp.diag(1.0 / jnp.sqrt(s_keep))
    A_red = D_inv_sqrt @ H_red @ D_inv_sqrt
    A_red = 0.5 * (A_red + A_red.conj().T)
    evals, evecs = jnp.linalg.eigh(A_red)
    order = jnp.argsort(jnp.real(evals))
    # Coefficients: ritz_vec_i = sum_j coeffs[j, i] * original_vec_j
    coeffs = U @ D_inv_sqrt @ evecs[:, order]
    return jnp.real(evals)[order], coeffs


def _rayleigh_ritz(
    matvec,
    vectors: list[jax.Array],
    data: dict,
    s_cutoff: float = 0.01,
) -> RitzResult:
    """Rayleigh-Ritz with S-eigenvalue truncation of spurious directions.

    Uses a relative cutoff on S eigenvalues to filter spurious directions.
    """
    n = len(vectors)
    if n == 0:
        return RitzResult(
            ritz_evals=np.array([], dtype=np.float64),
            ritz_coeffs=np.zeros((0, 0), dtype=np.float64),
            s_evals=np.array([], dtype=np.float64),
            rayleigh_quotients=np.array([], dtype=np.float64),
            n_physical=0,
            n_total=0,
            s_threshold=0.0,
        )

    hv = []
    for v in vectors:
        hv.append(
            matvec(
                v,
                data["psi_c_X"],
                data["psi_c_Y"],
                data["psi_v_X"],
                data["psi_v_Y"],
                data["eps_c"],
                data["eps_v"],
                data["W_R"],
                data["V_q0"],
            )
        )

    V = jnp.stack(vectors, axis=0)
    HV = jnp.stack(hv, axis=0)

    V_flat = V.reshape((n, -1))
    HV_flat = HV.reshape((n, -1))

    S = V_flat.conj() @ V_flat.T
    H = V_flat.conj() @ HV_flat.T

    S = 0.5 * (S + S.conj().T)
    H = 0.5 * (H + H.conj().T)

    s_evals, s_evecs = jnp.linalg.eigh(S)

    threshold = s_cutoff * jnp.max(s_evals)
    n_cutoff = jnp.sum(s_evals > threshold)
    n_physical = int(jax.device_get(jnp.maximum(n_cutoff, 1)))

    # Rayleigh quotients: H_ii / S_ii for each vector (no extra matvecs needed).
    rq = jnp.real(jnp.diag(H) / jnp.maximum(jnp.real(jnp.diag(S)), 1e-30))
    rq = np.sort(np.asarray(jax.device_get(rq)))

    s_evals_host = np.asarray(jax.device_get(s_evals))
    if n_physical == 0 or s_evals_host.max() <= 0:
        return RitzResult(
            ritz_evals=np.array([], dtype=np.float64),
            ritz_coeffs=np.zeros((n, 0), dtype=np.float64),
            s_evals=s_evals_host,
            rayleigh_quotients=rq,
            n_physical=0,
            n_total=n,
            s_threshold=float(jax.device_get(threshold)),
        )

    evals, coeffs = _solve_reduced_evp(S, H, s_evals, s_evecs, n_physical)
    evals_host = np.asarray(jax.device_get(evals))
    coeffs_host = np.asarray(jax.device_get(coeffs))

    return RitzResult(
        ritz_evals=evals_host,
        ritz_coeffs=coeffs_host,
        s_evals=s_evals_host,
        rayleigh_quotients=rq,
        n_physical=n_physical,
        n_total=n,
        s_threshold=float(jax.device_get(threshold)),
    )


def _diagnostic_subspace_stats(
    matvec,
    vectors: list[jax.Array],
    data: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rayleigh_quotients, S_eigvals, H_eigvals) for diagnostics."""
    n = len(vectors)
    S = np.zeros((n, n), dtype=np.complex128)
    H = np.zeros((n, n), dtype=np.complex128)
    rq = np.zeros((n,), dtype=np.float64)

    hv = []
    for v in vectors:
        hv.append(
            matvec(
                v,
                data["psi_c_X"],
                data["psi_c_Y"],
                data["psi_v_X"],
                data["psi_v_Y"],
                data["eps_c"],
                data["eps_v"],
                data["W_R"],
                data["V_q0"],
            )
        )

    for i in range(n):
        vi = vectors[i]
        hi = hv[i]
        norm = jnp.vdot(vi, vi)
        rq_i = jnp.vdot(vi, hi) / jnp.maximum(norm, 1e-30)
        rq[i] = float(jax.device_get(rq_i).real)
        for j in range(n):
            S_ij = jnp.vdot(vi, vectors[j])
            H_ij = jnp.vdot(vi, hv[j])
            S[i, j] = complex(jax.device_get(S_ij))
            H[i, j] = complex(jax.device_get(H_ij))

    S = 0.5 * (S + S.conj().T)
    H = 0.5 * (H + H.conj().T)
    Se = np.linalg.eigvalsh(S).real
    He = np.linalg.eigvalsh(H).real
    return rq, Se, He


def _build_ritz_vectors(
    filtered: list[jax.Array],
    coeffs: np.ndarray,
    sharding,
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
        c = coeffs[:, i]  # (n_total,) — should be real for TDA-BSE
        v = sum(float(c[j].real) * filtered[j] for j in range(len(filtered)))
        v = v.real  # BSE-TDA eigenvectors are real
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
    s_cutoff: float = 0.01,
    feast_iter: int = 1,
    quadrature: str = "zolotarev",
    lambda_min_eV: float | None = None,
    lambda_max_eV: float | None = None,
    zolotarev_rho_scale: float = 1.0,
    n_quad_schedule: list[int] | None = None,
) -> dict:
    matvec = build_bse_ring_matvec(mesh_xy, data["nkx"], data["nky"], data["nkz"])
    sh = make_bse_shardings(mesh_xy)

    # Convert W_q (reciprocal space) to W_R (real space) for the ring matvec.
    data["W_R"] = jnp.fft.ifftn(data["W_q"], axes=(2, 3, 4), norm="ortho")

    diag_h = build_preconditioner_diagonal_sharded(data, mesh_xy)
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
        runner_cache[nq] = _get_feast_runner(
            matvec,
            data,
            nq,
            n_ritz,
            gmres_max_iter,
            gmres_tol,
            ry_to_ev,
        )

    for window in windows:
        # Initial random starting vectors.
        X_list = []
        for _ in range(n_ritz):
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (1, n_cond_pad, n_val_pad, nk), dtype=jnp.float64)
            x = x / jnp.linalg.norm(x)
            x = jax.lax.with_sharding_constraint(x, sh.X)
            X_list.append(x.astype(jnp.complex128))
        X_batch = jnp.stack(X_list, axis=0)

        ritz_result = None
        prev_evals = None

        for it in range(feast_iter):
            is_last = (it == feast_iter - 1)
            n_quad_it = quad_schedule[it]
            print(f"\n  {window.name} FEAST iteration {it+1}/{feast_iter} (n_quad={n_quad_it})")

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
            z_jnp = jnp.asarray(z_nodes)
            w_jnp = jnp.asarray(w_weights)

            # --- Filter ---
            filtered_batch, iters = runner_cache[n_quad_it](X_batch, z_jnp, w_jnp, diag_h)

            # GMRES iteration summary.
            iters_host = np.asarray(jax.device_get(iters))
            for i in range(n_ritz):
                iters_str = " ".join(f"{iters_host[i, j]}" for j in range(n_quad_it))
                print(f"    {window.name} vec {i+1}: GMRES iters [{iters_str}]")
            avg_gmres = float(iters_host.mean())
            total_gmres = int(iters_host.sum())
            total_matvecs = total_gmres + n_ritz * n_quad_it  # each GMRES iter + 1 initial per solve
            print(f"    GMRES avg iters: {avg_gmres:.1f}, total solves: {n_ritz*n_quad_it}, total matvecs: {total_matvecs}")

            filtered = [filtered_batch[i] for i in range(n_ritz)]

            # --- Rayleigh-Ritz (verbose only on last iteration) ---
            print(f"  {window.name} Rayleigh-Ritz [{window.a_eV:.2f}, {window.b_eV:.2f}] eV:")
            ritz_result = _rayleigh_ritz(
                matvec, filtered, data,
                s_cutoff=s_cutoff,
            )

            # Print Ritz values for this iteration.
            ev_str = ", ".join(f"{v*ry_to_ev:.6f}" for v in ritz_result.ritz_evals)
            print(f"    Ritz evals (eV): [{ev_str}]")

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
                ritz_vecs = _build_ritz_vectors(filtered, ritz_result.ritz_coeffs, sh.X)

                # Pad back to n_ritz with random vectors if n_physical < n_ritz.
                X_next = []
                for v in ritz_vecs:
                    X_next.append(v.astype(jnp.complex128))
                while len(X_next) < n_ritz:
                    key, subkey = jax.random.split(key)
                    x = jax.random.normal(subkey, (1, n_cond_pad, n_val_pad, nk), dtype=jnp.float64)
                    x = x / jnp.linalg.norm(x)
                    x = jax.lax.with_sharding_constraint(x, sh.X)
                    X_next.append(x.astype(jnp.complex128))
                X_batch = jnp.stack(X_next, axis=0)

        # Final diagnostics for this window (printed after last iteration).
        assert ritz_result is not None
        results[window.name] = ritz_result

        # S eigenvalue summary.
        print(f"  S eigenvalues ({ritz_result.n_physical}/{ritz_result.n_total} physical):")
        for i, s in enumerate(ritz_result.s_evals, start=1):
            tag = " *" if s > ritz_result.s_threshold else ""
            print(f"    {i:2d}: {s:.6e}{tag}")

        # Rayleigh quotients.
        print(f"  Rayleigh quotients (eV):")
        for i, val in enumerate(ritz_result.rayleigh_quotients, start=1):
            print(f"    {i:2d}: {val * ry_to_ev:12.6f}")

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
    seed: int = 0,
) -> dict:
    """Estimate E_min and E_max (Ry) using diagonal gap and short Lanczos.

    Returns a dict containing the full per-iteration E_max sequence from the
    Lanczos tridiagonal as a diagnostic for convergence vs. iteration count.
    """
    eps_c = data["eps_c"]
    eps_v = data["eps_v"]
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
    matvec = build_bse_ring_matvec(mesh_xy, data["nkx"], data["nky"], data["nkz"])

    key = jax.random.PRNGKey(seed)

    @jax.jit
    def _make_random_vector(key_in):
        k1, k2 = jax.random.split(key_in)
        q = jax.random.normal(k1, (1, n_cond_pad, n_val_pad, nk), dtype=jnp.float64)
        q = q + 1j * jax.random.normal(k2, (1, n_cond_pad, n_val_pad, nk), dtype=jnp.float64)
        q = q / jnp.linalg.norm(q)
        return jax.lax.with_sharding_constraint(q, sh.X)

    W_R = jnp.fft.ifftn(data["W_q"], axes=(2, 3, 4), norm="ortho")

    q = _make_random_vector(key)
    q_prev = jnp.zeros_like(q)
    beta_prev = 0.0

    alphas: list[float] = []
    betas: list[float] = []

    for _ in range(n_lanczos):
        z = matvec(
            q,
            data["psi_c_X"],
            data["psi_c_Y"],
            data["psi_v_X"],
            data["psi_v_Y"],
            eps_c,
            eps_v,
            W_R,
            data["V_q0"],
        )

        alpha = jnp.vdot(q, z).real
        z = z - alpha * q - beta_prev * q_prev
        beta = jnp.linalg.norm(z)

        alpha_f = float(jax.device_get(alpha))
        beta_f = float(jax.device_get(beta))
        alphas.append(alpha_f)
        betas.append(beta_f)


        if beta_f == 0.0:
            break

        q_prev = q
        q = z / beta
        beta_prev = beta

    t_dim = len(alphas)
    T = np.zeros((t_dim, t_dim), dtype=np.float64)
    for i, alpha in enumerate(alphas):
        T[i, i] = alpha
        if i < t_dim - 1:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
    evals = np.linalg.eigvalsh(T)
    e_max_ry = float(np.max(evals))

    # Count diagonal elements (non-interacting transitions) as a function of energy.
    delta_E = energy_diff_cv_k(eps_c_use, eps_v_use)  # (n_cond, n_val, nk)
    diag_flat = np.asarray(jax.device_get(delta_E.real)).ravel()

    return {
        "e_min_ry": float(jax.device_get(e_min_ry)),
        "e_max_ry_raw": e_max_ry,
        "n_lanczos": t_dim,
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
    a = window.a_eV
    b = window.b_eV
    c = 0.5 * (a + b)
    r_x = 0.5 * (b - a)
    r_y = gamma * r_x

    thetas = np.array([
        math.pi * (2 * j - 1) / (2 * n_quad) for j in range(1, n_quad + 1)
    ])

    z = c + r_x * np.cos(thetas) + 1j * r_y * np.sin(thetas)
    # Upper-half contour weights include the 1/(2πi) factor and Δθ=π/n_quad:
    # w_j = (1/(2 n_quad i)) dz/dθ = (1/(2 n_quad)) (r_y cosθ + i r_x sinθ)
    w = (1.0 / (2.0 * n_quad)) * (r_y * np.cos(thetas) + 1j * r_x * np.sin(thetas))
    return z.astype(np.complex128), w.astype(np.complex128)


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
    lines.append(f"Backend     : Sharded BSE matvec (px, py = {px}, {py})")
    lines.append(f"Energies    : eps in Ry; reporting in Ry and eV (Ry->eV = {ry_to_ev})")
    lines.append("")
    lines.append("--- Spectral bounds (Lanczos) ---")
    lines.append(f"Lanczos steps          : {bounds['n_lanczos']}")
    lines.append(f"E_min (diag gap)       : {e_min_ry:10.6f} Ry = {e_min_eV:9.3f} eV")
    lines.append(f"E_max (Lanczos raw)    : {e_max_ry_raw:10.6f} Ry = {e_max_eV_raw:9.3f} eV")
    lines.append(f"Buffer factor          : {buffer_factor:.2f}")
    lines.append(f"E_max (buffered)       : {e_max_ry:10.6f} Ry = {e_max_eV:9.3f} eV")
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
        lines.append("    z_j (eV) and w_j (eV)   [upper half-plane]")
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
    parser.add_argument("--n-lanczos", type=int, default=10)
    parser.add_argument("--buffer", type=float, default=0.05)
    parser.add_argument("--n-quad1", type=int, default=4,
                        help="Quadrature points for FEAST iteration 1.")
    parser.add_argument("--n-quad2", type=int, default=8,
                        help="Quadrature points for FEAST iteration 2+.")
    parser.add_argument("--feast-ritz", action="store_true", help="Run FEAST GMRES + Ritz extraction.")
    parser.add_argument("--feast-ritz-count", type=int, default=4, help="Ritz values per window.")
    parser.add_argument("--s-cutoff", type=float, default=0.01,
                        help="S eigenvalue cutoff (relative to max) for filtering spurious Ritz pairs.")
    parser.add_argument("--feast-iter", type=int, default=2,
                        help="Number of FEAST subspace iterations (re-filter Ritz vectors).")
    parser.add_argument("--quadrature", type=str, default="ellipse",
                        choices=["zolotarev", "ellipse"],
                        help="Quadrature type for FEAST filter (default: ellipse).")
    parser.add_argument("--gmres-max-iter", type=int, default=4, help="GMRES iterations per shift.")
    parser.add_argument("--gmres-tol", type=float, default=1e-2, help="GMRES relative tolerance.")
    parser.add_argument("--gmres-seed", type=int, default=0, help="Random seed for FEAST vectors.")
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

    mesh_xy = _create_mesh_xy(args.px, args.py)
    restart_file = _find_restart_file(args.input)
    with timing.section("feast.restart_load"):
        data = load_bse_data_from_restart_sharded(
            restart_file,
            n_val=args.n_val,
            n_cond=args.n_cond,
            mesh_xy=mesh_xy,
        )

    with timing.section("feast.lanczos_bounds"):
        bounds = estimate_spectral_bounds_sharded(
            data,
            mesh_xy,
            n_lanczos=args.n_lanczos,
        )

    e_min_ry = bounds["e_min_ry"]
    e_max_ry = bounds["e_max_ry_raw"] * (1.0 + args.buffer)
    e_min_eV = e_min_ry * args.units_ev_per_ry
    e_max_eV = e_max_ry * args.units_ev_per_ry

    windows = build_default_windows_eV(e_max_eV)

    if args.window1 is not None or args.window2 is not None:
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
            )
        for name, rr in results.items():
            print(f"\n{name} Ritz values (eV) [{rr.n_physical} physical / {rr.n_total} total]:")
            for i, val in enumerate(rr.ritz_evals, start=1):
                print(f"  {i:2d}: {val * args.units_ev_per_ry:12.6f}")

    timing.report(print_fn=print, title="--- Timing ---")


if __name__ == "__main__":
    main()
