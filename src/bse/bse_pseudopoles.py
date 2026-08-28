"""FEAST-based pseudopole construction with density-biased random seeds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable

import math
import numpy as np
import h5py

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .bse_feast import (
    RY_TO_EV_DEFAULT,
    feast_ellipse_quadrature,
    feast_zolotarev_quadrature,
    _get_feast_runner,
    _build_gmres_data_fp32,
    ensure_W_R,
    _cast_with_sharding,
    build_preconditioner_diagonal_sharded,
    estimate_spectral_bounds_sharded,
    build_default_windows_eV,
    _parse_window_arg,
)
from .bse_ring_comm import (
    build_bse_ring_matvec,
    build_bse_ring_matvec_full,
    build_density_snapshot_operator,
    build_density_drive_operators,
    build_density_readout_operator_full,
    create_mesh_xy_from_flags,
    make_bse_shardings,
)
from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from common.collectives import gather_to_host
import common.timing as timing

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class WindowSpec:
    name: str
    a_eV: float
    b_eV: float
    note: str


def _orthonormalize(filtered: list[jax.Array], s_cutoff: float) -> tuple[list[jax.Array], np.ndarray, np.ndarray]:
    """Orthonormalize filtered vectors and return basis + coeffs + overlap eigenvalues."""
    n = len(filtered)
    if n == 0:
        return [], np.zeros((0, 0)), np.zeros((0,))

    V = jnp.stack(filtered, axis=0)
    V_flat = V.reshape((n, -1))
    S_dev = V_flat.conj() @ V_flat.T
    S_np = np.asarray(jax.device_get(S_dev))
    S_np = 0.5 * (S_np + S_np.conj().T)

    s_evals, s_evecs = np.linalg.eigh(S_np)
    s_evals = s_evals.real
    s_max = float(np.max(s_evals)) if s_evals.size else 0.0
    s_floor = max(float(s_cutoff) * s_max, 1e-30)
    keep = s_evals >= s_floor
    if not np.any(keep):
        return [], np.zeros((n, 0)), s_evals

    coeffs = s_evecs[:, keep] / np.sqrt(s_evals[keep])
    basis = []
    for i in range(coeffs.shape[1]):
        c = coeffs[:, i]
        v = sum(c[j] * filtered[j] for j in range(n))
        basis.append(v)
    return basis, coeffs, s_evals


def _build_j_metric(basis: list[jax.Array]) -> np.ndarray:
    """Build J-metric matrix J[i,j] = <v_i_X|v_j_X> - <v_i_Y|v_j_Y> for non-TDA.

    For the Casida matrix S = [[A,B],[-B*,-A*]], the correct spectral norm is
    N_p = psi_p^H J psi_p = ||X_p||^2 - ||Y_p||^2, where J = diag(I, -I).
    """
    n = len(basis)
    if n == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    V = jnp.stack(basis, axis=0)
    # V has shape (n, 2, 1, nc, nv, nk) for non-TDA
    dim_half = int(np.prod(V.shape[2:]))
    X = V[:, 0].reshape((n, dim_half))
    Y = V[:, 1].reshape((n, dim_half))
    S_XX = X.conj() @ X.T
    S_YY = Y.conj() @ Y.T
    J = S_XX - S_YY
    return np.asarray(jax.device_get(J))


def _build_reduced_h(matvec, basis: list[jax.Array], data: dict, use_tda: bool) -> np.ndarray:
    """Build reduced H_w = V^H S V in the orthonormal basis."""
    if not basis:
        return np.zeros((0, 0), dtype=np.complex128)
    hv = []
    for v in basis:
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
                data["M_X"],  # hoisted V-term pair-amps (audit P3)
                data["M_Y"],
            )
        )
    V = jnp.stack(basis, axis=0)
    HV = jnp.stack(hv, axis=0)
    V_flat = V.reshape((len(basis), -1))
    HV_flat = HV.reshape((len(basis), -1))
    H_dev = V_flat.conj() @ HV_flat.T
    H_np = np.asarray(jax.device_get(H_dev))
    if use_tda:
        H_np = 0.5 * (H_np + H_np.conj().T)
    return H_np


def _compute_density_snapshots(
    basis: list[jax.Array],
    snapshot_op,
    readout_full,
    sh,
    psi_c_Y: jax.Array,
    psi_v_Y: jax.Array,
    V_q0: jax.Array,
    use_tda: bool,
) -> np.ndarray:
    """Compute C_w columns w(mu)=V(dX+d*Y) for each basis vector.

    This matches the response channel used by `bse_w_exact.py`:
      - TDA:    w = V(d X)
      - non-TDA w = V(d X + d* Y)
    """
    if not basis:
        return np.zeros((0, 0), dtype=np.complex128)
    cols = []
    for v in basis:
        if use_tda:
            s = jax.lax.with_sharding_constraint(v, sh.X)
            w_mu = snapshot_op(s, psi_c_Y, psi_v_Y, V_q0)
        else:
            s_full = jax.lax.with_sharding_constraint(v, sh.X_full)
            w_mu = readout_full(s_full, psi_c_Y, psi_v_Y, V_q0)
        # ``gather_to_host``: the density snapshot comes back on
        # ``sh.d_mu = P(None, 'x')`` (bse_ring_comm.py), i.e. TILED over mesh
        # axis x -- and ``create_mesh_xy`` refuses non-square meshes, so every
        # P>1 run has px>=2 and this column spans processes.  A bare
        # ``device_get`` raised there.  This is the live ``run_pseudopoles`` ->
        # ``_compute_density_snapshots`` path, not a debug one.
        cols.append(gather_to_host(w_mu[0]))
    return np.stack(cols, axis=1)


def _feast_filter(
    X_batch: jax.Array,
    z_nodes: np.ndarray,
    w_weights: np.ndarray,
    matvec,
    data: dict,
    diag_h: jax.Array,
    max_iter: int,
    tol: float,
    ry_to_ev: float,
    gmres_fp32: bool,
    use_tda: bool,
    mesh_xy: Mesh,
    include_W: bool,
) -> tuple[jax.Array, np.ndarray]:
    n_vec = int(X_batch.shape[0])
    data_gmres = data
    diag_h_gmres = diag_h
    runner_dtype = jnp.complex128
    if gmres_fp32:
        # ensure_W_R = the ONE sharded W_q->W_R route (make_w_densifier);
        # with include_W=False it installs the W_q placeholder instead of
        # transforming an operand the matvec never reads.
        data_gmres = ensure_W_R(_build_gmres_data_fp32(data), include_W, mesh_xy)
        diag_h_gmres = _cast_with_sharding(diag_h, jnp.float32)
        runner_dtype = jnp.complex64

    z_dtype = jnp.complex64 if gmres_fp32 else jnp.complex128
    z_jnp = jnp.asarray(z_nodes, dtype=z_dtype)
    w_jnp = jnp.asarray(w_weights, dtype=z_dtype)

    n_quad = int(z_nodes.shape[0])
    runner = _get_feast_runner(
        matvec,
        data_gmres,
        n_quad,
        n_vec,
        max_iter,
        tol,
        ry_to_ev,
        runner_dtype,
        use_conjugate_symmetry=use_tda,
    )
    filtered, iters = runner(X_batch.astype(runner_dtype), z_jnp, w_jnp, diag_h_gmres)
    iters_host = np.asarray(jax.device_get(iters))
    return filtered, iters_host


def run_pseudopoles(
    data: dict,
    mesh_xy: Mesh,
    windows: Iterable[WindowSpec],
    *,
    m0: int,
    n_quad: int,
    p_keep: int,
    n_tail: int,
    gmres_max_iter: int,
    gmres_tol: float,
    seed: int,
    ry_to_ev: float,
    s_cutoff: float,
    quadrature: str,
    use_tda: bool,
    gmres_fp32: bool,
    include_W: bool,
) -> dict:
    if use_tda:
        matvec = build_bse_ring_matvec(
            mesh_xy,
            data["nkx"],
            data["nky"],
            data["nkz"],
            include_W=include_W,
        )
    else:
        matvec = build_bse_ring_matvec_full(
            mesh_xy,
            data["nkx"],
            data["nky"],
            data["nkz"],
            include_W=include_W,
        )

    ensure_W_R(data, include_W, mesh_xy)

    diag_h = build_preconditioner_diagonal_sharded(data, mesh_xy, include_W=include_W, use_tda=use_tda)

    sh = make_bse_shardings(mesh_xy)
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    n_cond_pad = int(data["n_cond_pad"])
    n_val_pad = int(data["n_val_pad"])

    f_op, fbar_op = build_density_drive_operators(
        mesh_xy, data["nkx"], data["nky"], data["nkz"], n_cond_pad, n_val_pad
    )
    snapshot_op = build_density_snapshot_operator(mesh_xy, data["nkx"], data["nky"], data["nkz"])
    readout_full = build_density_readout_operator_full(mesh_xy, data["nkx"], data["nky"], data["nkz"])

    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)

    results = {}

    for window in windows:
        print(f"\n=== Window {window.name} [{window.a_eV:.3f}, {window.b_eV:.3f}] eV ===")
        if quadrature == "zolotarev":
            raise ValueError("Zolotarev quadrature requires spectral bounds; use ellipse here.")
        z_nodes, w_weights = feast_ellipse_quadrature(window, n_quad, gamma=0.2)
        if not use_tda:
            z_nodes = np.concatenate([z_nodes, np.conj(z_nodes)])
            w_weights = np.concatenate([w_weights, np.conj(w_weights)])

        # --- Step A: density-biased seeds (eta -> V eta -> [f, -fbar]) ---
        #
        # For non-TDA we must use the transpose-coupled vertex in the Y sector:
        #   rhs = [f, -fbar],  f = d^\dagger (V eta), fbar = d^T (V eta)
        # (Do not assume fbar == conj(f) for complex Bloch spinors.)
        seeds = []
        for _ in range(m0):
            key, subkey = jax.random.split(key)
            eta = jax.random.normal(
                subkey,
                (1, data["psi_v_Y"].shape[-1], nk),
                dtype=jnp.float32 if gmres_fp32 else jnp.float64,
            )
            eta = jax.lax.with_sharding_constraint(eta, sh.S)
            if use_tda:
                f = f_op(eta, data["psi_c_X"], data["psi_v_X"], data["V_q0"])
                phi = jax.lax.with_sharding_constraint(f, sh.X)
            else:
                f = f_op(eta, data["psi_c_X"], data["psi_v_X"], data["V_q0"])
                f = jax.lax.with_sharding_constraint(f, sh.X)
                fbar = fbar_op(eta, data["psi_c_X"], data["psi_v_X"], data["V_q0"])
                fbar = jax.lax.with_sharding_constraint(fbar, sh.X)
                phi = jnp.stack([f, -fbar], axis=0)
                phi = jax.lax.with_sharding_constraint(phi, sh.X_full)
            seeds.append(phi)
        X_batch = jnp.stack(seeds, axis=0)

        # --- Step B: FEAST filter ---
        filtered_batch, iters = _feast_filter(
            X_batch,
            z_nodes,
            w_weights,
            matvec,
            data,
            diag_h,
            gmres_max_iter,
            gmres_tol,
            ry_to_ev,
            gmres_fp32,
            use_tda,
            mesh_xy,
            include_W,
        )
        filtered = [filtered_batch[i] for i in range(m0)]
        if gmres_fp32:
            filtered = [v.astype(jnp.complex128) for v in filtered]
        it_sum = int(np.sum(iters))
        n_solves = m0 * int(z_nodes.shape[0])
        print(f"  FEAST solves={n_solves}, total_iters={it_sum}")

        # --- Step C: Orthonormalize + reduced operator ---
        basis, coeffs, s_evals = _orthonormalize(filtered, s_cutoff=s_cutoff)
        if not basis:
            print("  WARNING: empty filtered subspace after orthonormalization.")
            results[window.name] = {"omega_bright": np.zeros((0,)), "d_bright": np.zeros((0, 0))}
            continue
        print(f"  Basis size: {len(basis)} (from m0={m0})")

        H_w = _build_reduced_h(matvec, basis, data, use_tda=use_tda)

        # --- Step D: Build C_w (density snapshots) ---
        C_w = _compute_density_snapshots(
            basis,
            snapshot_op,
            readout_full,
            sh,
            data["psi_c_Y"],
            data["psi_v_Y"],
            data["V_q0"],
            use_tda=use_tda,
        )

        # --- Step E: Brightness eigen-decomp ---
        G_w = C_w.conj().T @ C_w
        G_w = 0.5 * (G_w + G_w.conj().T)
        sigma2, W_w = np.linalg.eigh(G_w)
        order = np.argsort(sigma2)[::-1]
        sigma2 = sigma2[order]
        W_w = W_w[:, order]
        sigma = np.sqrt(np.maximum(sigma2, 0.0))

        p_use = min(p_keep, W_w.shape[1])
        Wb = W_w[:, :p_use]
        Wd = W_w[:, p_use:]

        # --- Step F: Bright pseudopoles ---
        H_b = Wb.conj().T @ H_w @ Wb
        evals_b, vecs_b = np.linalg.eig(H_b)
        order_b = np.argsort(evals_b.real)
        evals_b = evals_b[order_b]
        vecs_b = vecs_b[:, order_b]
        # Discard Ritz pairs whose eigenvalue falls outside the window.
        # Imperfect FEAST filtering can leak bright eigenstates from outside
        # the window, producing spurious Ritz values that massively distort
        # the pseudopole reconstruction.
        a_ry = window.a_eV / ry_to_ev
        b_ry = window.b_eV / ry_to_ev
        in_window = (evals_b.real >= a_ry) & (evals_b.real <= b_ry)
        n_discarded = int(np.sum(~in_window))
        if n_discarded > 0:
            print(f"  Discarding {n_discarded} out-of-window Ritz value(s):")
            for idx_d in np.where(~in_window)[0]:
                print(f"    omega={evals_b[idx_d].real * ry_to_ev:.4f} eV"
                      f" (window [{window.a_eV:.1f}, {window.b_eV:.1f}])")
            evals_b = evals_b[in_window]
            vecs_b = vecs_b[:, in_window]

        g_b = Wb @ vecs_b
        d_bright = C_w @ g_b

        # Non-TDA corrections: J-norm + anti-resonant poles.
        #
        # The non-Hermitian Casida resolvent has the Lehmann form (see e.g.
        # Casida 1995, or any linear-response TDDFT reference):
        #
        #   chi(z) = sum_s [ F_s F_s† / (z - Omega_s)
        #                  - F_s* F_s^T / (z + Omega_s) ]
        #
        # where F_s = sum_{cv} [X_s^{cv} rho_{cv} + Y_s^{cv} rho*_{cv}].
        # Note the anti-resonant residue is F_s* F_s^T (NOT F_s F_s†).
        #
        # In our eval formula  Wc[mu,nu] = sum_p w_p d_p[mu] conj(d_p[nu]) / (z - omega_p):
        #   Resonant:      d = F_s,       w = +1, omega = +Omega_s
        #   Anti-resonant: d = conj(F_s), w = -1, omega = -Omega_s
        #
        # This correctly produces:
        #   w * d[mu] * conj(d[nu]) / (z - omega)
        #   = (-1) * conj(F[mu]) * F[nu] / (z + Omega)
        #   = -F* F^T / (z + Omega)    ✓
        weights_b = np.ones(evals_b.shape[0], dtype=np.float64)

        if not use_tda and len(basis) > 0 and evals_b.size > 0:
            J_w = _build_j_metric(basis)
            J_b = Wb.conj().T @ J_w @ Wb
            j_norms = np.array([
                vecs_b[:, p].conj() @ J_b @ vecs_b[:, p]
                for p in range(vecs_b.shape[1])
            ])
            j_norms_real = j_norms.real

            n_bad = int(np.sum(j_norms_real <= 0))
            if n_bad > 0:
                print(f"  WARNING: {n_bad} Ritz vector(s) with non-positive J-norm")
            j_floor = 1e-6
            for p in range(d_bright.shape[1]):
                if j_norms_real[p] > j_floor:
                    d_bright[:, p] /= np.sqrt(j_norms_real[p])
                else:
                    print(f"    Skipping J-norm for pole {p} (N={j_norms_real[p]:.6e})")
            print(f"  J-norms: min={j_norms_real.min():.4f}, max={j_norms_real.max():.4f},"
                  f" mean={j_norms_real.mean():.4f}")

            # Add anti-resonant poles at -Omega_p with weight=-1 and conj(d_p).
            n_res = evals_b.shape[0]
            evals_b = np.concatenate([evals_b, -evals_b])
            d_bright = np.concatenate([d_bright, np.conj(d_bright)], axis=1)
            weights_b = np.concatenate([np.ones(n_res), -np.ones(n_res)])

        # --- Step G: Tail pseudopoles ---
        omega_tail = np.zeros((0,), dtype=np.complex128)
        d_tail = np.zeros((0, C_w.shape[0]), dtype=np.complex128)
        if n_tail > 0 and Wd.size > 0:
            H_d = Wd.conj().T @ H_w @ Wd
            d_list = []
            omega_list = []
            for _ in range(n_tail):
                z = rng.normal(size=(Wd.shape[1],)) + 1j * rng.normal(size=(Wd.shape[1],))
                z = z / np.linalg.norm(z)
                omega = z.conj().T @ H_d @ z
                g = Wd @ z
                d_vec = C_w @ g
                omega_list.append(omega)
                d_list.append(d_vec)
            omega_tail = np.array(omega_list, dtype=np.complex128)
            d_tail = np.stack(d_list, axis=0)

            # Discard tail pseudopoles outside the window
            tail_in = (omega_tail.real >= a_ry) & (omega_tail.real <= b_ry)
            n_tail_disc = int(np.sum(~tail_in))
            if n_tail_disc > 0:
                print(f"  Discarding {n_tail_disc} out-of-window tail pseudopole(s)")
                omega_tail = omega_tail[tail_in]
                d_tail = d_tail[tail_in]

            B_disc = float(np.sum(sigma2[p_use:])) if sigma2.size > p_use else 0.0
            B_tail = float(np.sum(np.abs(d_tail) ** 2))
            if B_disc > 0.0 and B_tail > 0.0:
                alpha = math.sqrt(B_disc / B_tail)
                d_tail = d_tail * alpha

        weights_tail = np.ones(omega_tail.shape[0], dtype=np.float64)
        J_w_save = _build_j_metric(basis) if (not use_tda and len(basis) > 0) else np.zeros((0, 0), dtype=np.complex128)
        results[window.name] = {
            "omega_bright": evals_b,
            "d_bright": d_bright.T,
            "weight_bright": weights_b,
            "omega_tail": omega_tail,
            "d_tail": d_tail,
            "weight_tail": weights_tail,
            "sigma": sigma,
            "s_evals": s_evals,
            "H_w": H_w,
            "C_w": C_w,
            "J_w": J_w_save,
        }

    return results


def write_pseudopoles_h5(
    output_file: str,
    windows: Iterable[WindowSpec],
    results: dict,
    *,
    use_tda: bool,
    include_W: bool,
    ry_to_ev: float,
    m0: int,
    n_quad: int,
    p_keep: int,
    n_tail: int,
    gmres_max_iter: int,
    gmres_tol: float,
    seed: int,
) -> None:
    with h5py.File(output_file, "w") as h5:
        h5.attrs["use_tda"] = int(use_tda)
        h5.attrs["include_W"] = int(include_W)
        h5.attrs["ry_to_ev"] = float(ry_to_ev)
        h5.attrs["m0"] = int(m0)
        h5.attrs["n_quad"] = int(n_quad)
        h5.attrs["p_keep"] = int(p_keep)
        h5.attrs["n_tail"] = int(n_tail)
        h5.attrs["gmres_max_iter"] = int(gmres_max_iter)
        h5.attrs["gmres_tol"] = float(gmres_tol)
        h5.attrs["seed"] = int(seed)

        for w in windows:
            r = results.get(w.name, {})
            g = h5.create_group(w.name)
            g.attrs["a_eV"] = float(w.a_eV)
            g.attrs["b_eV"] = float(w.b_eV)
            omega_b = np.asarray(r.get("omega_bright", np.zeros((0,), dtype=np.complex128)))
            omega_t = np.asarray(r.get("omega_tail", np.zeros((0,), dtype=np.complex128)))
            d_b = np.asarray(r.get("d_bright", np.zeros((0, 0), dtype=np.complex128)))
            d_t = np.asarray(r.get("d_tail", np.zeros((0, 0), dtype=np.complex128)))
            w_b = np.asarray(r.get("weight_bright", np.ones(omega_b.shape[0], dtype=np.float64)))
            w_t = np.asarray(r.get("weight_tail", np.ones(omega_t.shape[0], dtype=np.float64)))
            sigma = np.asarray(r.get("sigma", np.zeros((0,), dtype=np.float64)))
            s_evals = np.asarray(r.get("s_evals", np.zeros((0,), dtype=np.float64)))

            g.create_dataset("omega_bright_ry", data=omega_b)
            g.create_dataset("omega_bright_eV", data=omega_b * ry_to_ev)
            g.create_dataset("d_bright", data=d_b)
            g.create_dataset("weight_bright", data=w_b)
            g.create_dataset("omega_tail_ry", data=omega_t)
            g.create_dataset("omega_tail_eV", data=omega_t * ry_to_ev)
            g.create_dataset("d_tail", data=d_t)
            g.create_dataset("weight_tail", data=w_t)
            g.create_dataset("sigma", data=sigma)
            g.create_dataset("s_evals", data=s_evals)
            H_w = np.asarray(r.get("H_w", np.zeros((0, 0), dtype=np.complex128)))
            C_w = np.asarray(r.get("C_w", np.zeros((0, 0), dtype=np.complex128)))
            J_w = np.asarray(r.get("J_w", np.zeros((0, 0), dtype=np.complex128)))
            g.create_dataset("H_w", data=H_w)
            g.create_dataset("C_w", data=C_w)
            g.create_dataset("J_w", data=J_w)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False, description="FEAST pseudopole construction with biased seeds")
    parser.add_argument("-i", "--input", required=True, help="COHSEX input file")
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--n-cond", type=int, default=4)
    parser.add_argument("--px", type=int, default=None,
                        help="mesh rows; default = the run's square startup "
                             "mesh")
    parser.add_argument("--py", type=int, default=None,
                        help="mesh columns; must equal --px, and px*py must "
                             "be the job's device count")
    parser.add_argument("--m0", type=int, default=6, help="Biased random seeds per window.")
    parser.add_argument("--n-quad", type=int, default=8, help="Quadrature points per half contour.")
    parser.add_argument("--p-keep", type=int, default=4, help="Bright modes retained per window.")
    parser.add_argument("--n-tail", type=int, default=2, help="Stochastic tail pseudomodes per window.")
    parser.add_argument("--gmres-max-iter", type=int, default=10)
    parser.add_argument("--gmres-tol", type=float, default=1e-2)
    parser.add_argument("--gmres-fp32", action="store_true", help="Use FP32 data/GMRES for shifted solves.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ry-to-ev", type=float, default=RY_TO_EV_DEFAULT)
    parser.add_argument("--rpa", action="store_true", help="Use RPA kernel (D+V only), skip W0 term entirely.")
    parser.add_argument("--tda", action="store_true", help="Use TDA (default full non-TDA).")
    parser.add_argument("--nohead", action="store_true",
                        help="Use headless V/W0 arrays if present (V_qmunu_nohead, W0_qmunu_nohead).")
    parser.add_argument("--out", type=str, default="bse_pseudopoles.h5")
    parser.add_argument("--s-cutoff", type=float, default=1e-6,
                        help="Overlap floor for orthonormalization.")
    parser.add_argument("--windows-kpm", action="store_true",
                        help="Derive windows from KPM DOS.")
    parser.add_argument("--windows-kpm-count", type=int, default=4)
    parser.add_argument("--kpm-n-moments", type=int, default=200)
    parser.add_argument("--kpm-n-random", type=int, default=4)
    parser.add_argument("--kpm-seed", type=int, default=0)
    parser.add_argument("--kpm-n-energy-pts", type=int, default=2000)
    parser.add_argument("--kpm-n-lanczos", type=int, default=100)
    parser.add_argument("--buffer", type=float, default=0.05)
    parser.add_argument("--window1", nargs=2, default=None, metavar=("A", "B"))
    parser.add_argument("--window2", nargs=2, default=None, metavar=("A", "B"))
    args = parser.parse_args(argv)

    timing.reset()

    # Omitted --px/--py = the run's canonical square mesh, not 1x1; a given
    # shape must BE that mesh.  Called through the OWNER, not through a local
    # one-line alias: the four ``_create_mesh_xy`` copies this module used to
    # carry one of are why five drivers silently ran un-warmed at P>1.
    mesh_xy = create_mesh_xy_from_flags(args.px, args.py)
    args.px, args.py = tuple(int(n) for n in mesh_xy.devices.shape)
    restart_file = _find_restart_file(args.input)

    with timing.section("pseudopoles.load"):
        data = load_bse_data_from_restart_sharded(
            restart_file,
            n_val=args.n_val,
            n_cond=args.n_cond,
            mesh_xy=mesh_xy,
            use_nohead=args.nohead,
            input_file=args.input,
        )

    with timing.section("pseudopoles.bounds"):
        bounds = estimate_spectral_bounds_sharded(
            data,
            mesh_xy,
            n_lanczos=args.kpm_n_lanczos,
            n_lanczos_max=max(args.kpm_n_lanczos, 50),
            include_W=not args.rpa,
            use_tda=args.tda,
        )

    e_min_ry = bounds["e_min_ry"]
    e_max_ry = bounds["e_max_ry_raw"] * (1.0 + args.buffer)
    e_min_eV = e_min_ry * args.ry_to_ev
    e_max_eV = e_max_ry * args.ry_to_ev

    windows = [WindowSpec(w.name, w.a_eV, w.b_eV, w.note) for w in build_default_windows_eV(e_max_eV)]
    if args.windows_kpm:
        from . import bse_kpm
        kpm_result = bse_kpm.run_kpm_dos(
            data,
            mesh_xy,
            n_moments=args.kpm_n_moments,
            n_random=args.kpm_n_random,
            seed=args.kpm_seed,
            n_lanczos=args.kpm_n_lanczos,
            buffer=args.buffer,
            ry_to_ev=args.ry_to_ev,
            n_energy_pts=args.kpm_n_energy_pts,
            plot_file="bse_kpm_windows.png",
            n_windows=args.windows_kpm_count,
            emit_outputs=False,
            include_W=not args.rpa,
            use_tda=args.tda,
        )
        windows_ry = kpm_result["windows_ry"]
        windows = [
            WindowSpec(f"W{i+1}", float(a * args.ry_to_ev), float(b * args.ry_to_ev), "KPM-weighted")
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

    print("--- Windows ---")
    for w in windows:
        print(f"{w.name}: [{w.a_eV:.3f}, {w.b_eV:.3f}] eV")

    with timing.section("pseudopoles.run"):
        results = run_pseudopoles(
            data,
            mesh_xy,
            windows,
            m0=args.m0,
            n_quad=args.n_quad,
            p_keep=args.p_keep,
            n_tail=args.n_tail,
            gmres_max_iter=args.gmres_max_iter,
            gmres_tol=args.gmres_tol,
            seed=args.seed,
            ry_to_ev=args.ry_to_ev,
            s_cutoff=args.s_cutoff,
            quadrature="ellipse",
            use_tda=args.tda,
            gmres_fp32=args.gmres_fp32,
            include_W=not args.rpa,
        )

    write_pseudopoles_h5(
        args.out,
        windows,
        results,
        use_tda=args.tda,
        include_W=not args.rpa,
        ry_to_ev=args.ry_to_ev,
        m0=args.m0,
        n_quad=args.n_quad,
        p_keep=args.p_keep,
        n_tail=args.n_tail,
        gmres_max_iter=args.gmres_max_iter,
        gmres_tol=args.gmres_tol,
        seed=args.seed,
    )

    print(f"\nPseudopoles written to {args.out}")
    timing.report(print_fn=print, title="--- Timing ---")


if __name__ == "__main__":
    raise SystemExit(main())
