"""Exact W_c(omega) via shifted solves with the (non-TDA) Casida/RPA matrix."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import h5py

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .bse_feast import RY_TO_EV_DEFAULT, gmres_solve_sharded_jit
from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from .bse_ring_comm import (
    build_bse_ring_matvec,
    build_bse_ring_matvec_full,
    build_realspace_random_transition_generator,
    build_density_snapshot_operator,
    make_bse_shardings,
)
from .bse_feast import build_preconditioner_diagonal_sharded
import isdf.common.timing as timing

jax.config.update("jax_enable_x64", True)


def _create_mesh_xy(px: int, py: int) -> Mesh:
    devices = jax.devices()
    n_devices = len(devices)
    if px * py > n_devices:
        raise ValueError(f"Requested px*py={px*py} devices, but only {n_devices} available")
    mesh_devices = np.array(devices[: px * py]).reshape(px, py)
    return Mesh(mesh_devices, axis_names=("x", "y"))


def _parse_cols(col_str: str | None, n_mu: int, n_cols: int | None, seed: int) -> np.ndarray:
    if col_str:
        cols = [int(x) for x in col_str.split(",") if x.strip() != ""]
        cols = [c for c in cols if 0 <= c < n_mu]
        return np.array(cols, dtype=int)
    if n_cols is not None:
        rng = np.random.default_rng(seed)
        if n_cols >= n_mu:
            return np.arange(n_mu, dtype=int)
        return rng.choice(n_mu, size=n_cols, replace=False)
    return np.arange(n_mu, dtype=int)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Exact W_c(omega) via Casida shifted solves")
    parser.add_argument("-i", "--input", required=True, help="COHSEX input file")
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--n-cond", type=int, default=4)
    parser.add_argument("--px", type=int, default=1)
    parser.add_argument("--py", type=int, default=1)
    parser.add_argument("--omega-ev", type=float, default=0.0,
                        help="Real frequency omega in eV (default: 0)")
    parser.add_argument("--eta-ev", type=float, default=0.0,
                        help="Imaginary broadening eta in eV (default: 0)")
    parser.add_argument("--cols", type=str, default=None,
                        help="Comma-separated mu indices to compute (0-based).")
    parser.add_argument("--n-cols", type=int, default=None,
                        help="Randomly sample N columns (if --cols not provided).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gmres-max-iter", type=int, default=10)
    parser.add_argument("--gmres-tol", type=float, default=1e-2)
    parser.add_argument("--gmres-fp32", action="store_true",
                        help="Use FP32 data/GMRES for shifted solves.")
    parser.add_argument("--print-residuals", action="store_true",
                        help="Print relative residuals after each GMRES solve.")
    parser.add_argument("--rpa", action="store_true",
                        help="Use RPA kernel (D+V only), skip W0 term.")
    parser.add_argument("--tda", action="store_true",
                        help="Use TDA (default full non-TDA).")
    parser.add_argument("--nohead", action="store_true",
                        help="Use headless V/W0 arrays if present (V_qmunu_nohead, W0_qmunu_nohead).")
    parser.add_argument("--density-channel", type=str, default="x_plus_y",
                        choices=("x_plus_y", "rx_plus_rstar_y"),
                        help="Non-TDA density channel: 'x_plus_y' uses R(X+Y); "
                             "'rx_plus_rstar_y' uses R X + R* Y.")
    parser.add_argument("--output-scale", type=float, default=1.0,
                        help="Scale factor applied to Wc output (default: 1.0).")
    parser.add_argument("--v-scale", type=float, default=1.0,
                        help="Scale factor applied to V_q0 (affects RHS and output).")
    parser.add_argument("--d-only", action="store_true",
                        help="Disable V/W in the Casida operator (D-only noninteracting check).")
    parser.add_argument("--ry-to-ev", type=float, default=RY_TO_EV_DEFAULT)
    parser.add_argument("--out", type=str, default="Wc_exact.h5")
    args = parser.parse_args(argv)

    timing.reset()

    mesh_xy = _create_mesh_xy(args.px, args.py)
    restart_file = _find_restart_file(args.input)

    with timing.section("w_exact.load"):
        data = load_bse_data_from_restart_sharded(
            restart_file,
            n_val=args.n_val,
            n_cond=args.n_cond,
            mesh_xy=mesh_xy,
            use_nohead=args.nohead,
        )

    nkx = int(data["nkx"])
    nky = int(data["nky"])
    nkz = int(data["nkz"])
    nk = nkx * nky * nkz
    n_rmu = int(data["V_q0"].shape[0])

    if args.v_scale != 1.0:
        data["V_q0"] = data["V_q0"] * jnp.asarray(args.v_scale, dtype=data["V_q0"].dtype)

    use_tda = args.tda
    include_W = not args.rpa

    if use_tda:
        matvec = build_bse_ring_matvec(mesh_xy, nkx, nky, nkz, include_W=include_W)
    else:
        matvec = build_bse_ring_matvec_full(mesh_xy, nkx, nky, nkz, include_W=include_W)

    if "W_R" not in data:
        if include_W:
            data["W_R"] = jnp.fft.ifftn(data["W_q"], axes=(2, 3, 4), norm="ortho")
        else:
            # Placeholder to satisfy matvec signature (not used when include_W=False).
            data["W_R"] = data["W_q"]

    data_op = data
    if args.d_only:
        data_op = dict(data)
        data_op["V_q0"] = jnp.zeros_like(data["V_q0"])
        if "W_q" in data_op:
            data_op["W_q"] = jnp.zeros_like(data["W_q"])
        if "W_R" in data_op:
            data_op["W_R"] = jnp.zeros_like(data["W_R"])
    diag_h = build_preconditioner_diagonal_sharded(data_op, mesh_xy, include_W=include_W, use_tda=use_tda)

    gen = build_realspace_random_transition_generator(mesh_xy, nkx, nky, nkz,
                                                       int(data["n_cond_pad"]), int(data["n_val_pad"]))
    snapshot_op = build_density_snapshot_operator(mesh_xy, nkx, nky, nkz)
    sh = make_bse_shardings(mesh_xy)

    cols = _parse_cols(args.cols, n_rmu, args.n_cols, args.seed)
    print(f"Computing {len(cols)} column(s) out of N_mu={n_rmu}")

    z = (args.omega_ev + 1j * args.eta_ev) / args.ry_to_ev

    dtype_r = jnp.float32 if args.gmres_fp32 else jnp.float64
    dtype_c = jnp.complex64 if args.gmres_fp32 else jnp.complex128

    wc_cols = []

    for idx, nu0 in enumerate(cols, start=1):
        print(f"  Column {idx}/{len(cols)}: nu={nu0}")
        g = jnp.zeros((n_rmu,), dtype=dtype_r).at[nu0].set(1.0)
        r = jnp.broadcast_to(g[None, :, None], (1, n_rmu, nk))
        r = jax.device_put(r, sh.S)

        f = gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"])
        f = jax.lax.with_sharding_constraint(f, sh.X)
        if use_tda:
            rhs = f.astype(dtype_c)
        else:
            # For the non-TDA Liouvillian S = [[A, B], [-B*, -A*]], the usual
            # density-coupling RHS for a real density source is [f, -f].
            rhs = jnp.stack([f, -f], axis=0).astype(dtype_c)
            rhs = jax.lax.with_sharding_constraint(rhs, sh.X_full)

        x, _ = gmres_solve_sharded_jit(
            matvec,
            diag_h,
            z,
            rhs,
            data_op,
            max_iter=args.gmres_max_iter,
            tol=args.gmres_tol,
        )

        if args.print_residuals:
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
            r = rhs - (z * x - hx)
            r_norm = jnp.linalg.norm(r)
            b_norm = jnp.linalg.norm(rhs)
            rel = r_norm / (b_norm + jnp.asarray(1e-30, dtype=b_norm.dtype))
            rel_host, r_host, b_host = map(float, jax.device_get((rel, r_norm, b_norm)))
            print(f"    residual rel={rel_host:.3e} (||r||={r_host:.3e}, ||b||={b_host:.3e})")

        if use_tda:
            s = jax.lax.with_sharding_constraint(x, sh.X)
            w_c = snapshot_op(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
        else:
            X = x[0]
            Y = x[1]
            if args.density_channel == "rx_plus_rstar_y":
                w_x = snapshot_op(X, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
                w_y = snapshot_op(Y, jnp.conj(data["psi_c_Y"]), jnp.conj(data["psi_v_Y"]), data["V_q0"])
                w_c = w_x + w_y
            else:
                s = jax.lax.with_sharding_constraint(X + Y, sh.X)
                w_c = snapshot_op(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
        w_c_host = np.asarray(jax.device_get(w_c[0])) * args.output_scale
        wc_cols.append(w_c_host)

    wc_cols = np.stack(wc_cols, axis=0) if wc_cols else np.zeros((0, n_rmu), dtype=np.complex128)

    with h5py.File(args.out, "w") as h5:
        h5.attrs["omega_ev"] = float(args.omega_ev)
        h5.attrs["eta_ev"] = float(args.eta_ev)
        h5.attrs["ry_to_ev"] = float(args.ry_to_ev)
        h5.attrs["use_tda"] = int(use_tda)
        h5.attrs["include_W"] = int(include_W)
        h5.attrs["gmres_max_iter"] = int(args.gmres_max_iter)
        h5.attrs["gmres_tol"] = float(args.gmres_tol)
        h5.attrs["seed"] = int(args.seed)
        h5.attrs["density_channel"] = str(args.density_channel)
        h5.attrs["output_scale"] = float(args.output_scale)
        h5.create_dataset("columns", data=cols.astype(np.int32))
        h5.create_dataset("Wc", data=wc_cols)

    print(f"Wrote Wc columns to {args.out}")
    timing.report(print_fn=print, title="--- Timing ---")


if __name__ == "__main__":
    main()
