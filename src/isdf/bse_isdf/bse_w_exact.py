"""Exact W_c(omega) via shifted solves with the (non-TDA) Casida/RPA matrix."""
from __future__ import annotations

import numpy as np
import h5py

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .bse_feast import RY_TO_EV_DEFAULT, gmres_solve_sharded_jit
from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from .bse_ring_comm import (
    build_bse_ring_matvec,
    build_bse_ring_matvec_full,
    build_density_drive_operators,
    build_density_readout_operator_full,
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
    parser.add_argument(
        "--print-gmres-iters",
        action="store_true",
        help="Print GMRES iteration count per column and summary statistics.",
    )
    parser.add_argument("--rpa", action="store_true",
                        help="Use RPA kernel (D+V only), skip W0 term.")
    parser.add_argument("--tda", action="store_true",
                        help="Use TDA (default full non-TDA).")
    parser.add_argument("--nohead", action="store_true",
                        help="Use headless V/W0 arrays if present (V_qmunu_nohead, W0_qmunu_nohead).")
    parser.add_argument(
        "--drive-kind",
        type=str,
        default="density",
        choices=("density", "potential"),
        help=(
            "RHS driving convention. 'density' drives with u=V g (default, computes V*chi*V action). "
            "'potential' drives with u=g (skips the leading V), useful to output chi columns directly."
        ),
    )
    parser.add_argument(
        "--write-kind",
        type=str,
        default="Wc",
        choices=("Wc", "Vchi", "chi"),
        help=(
            "Output quantity. 'Wc' writes V(dX+d*Y) scaled by --output-scale (default: 1.0). "
            "'Vchi' writes V(dX+d*Y) without interpreting it as Wc. "
            "'chi' writes density response (dX+d*Y) by inverting V on the host."
        ),
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=None,
        help=(
            "Optional scale factor applied to the written output only (default: 1.0). "
            "This does NOT affect the Casida operator S (D/V/W) or its eigenvalues; "
            "it is purely a post-factor on what is written to disk."
        ),
    )
    parser.add_argument(
        "--v-scale",
        type=float,
        default=1.0,
        help=(
            "Global scale applied to V_q0. If --v-scale-kernel/--v-scale-coupling are unset, "
            "this value is used for both."
        ),
    )
    parser.add_argument(
        "--v-scale-kernel",
        type=float,
        default=None,
        help="Scale factor applied to V_q0 inside the Casida operator (matvec + preconditioner).",
    )
    parser.add_argument(
        "--v-scale-coupling",
        type=float,
        default=None,
        help=(
            "Scale factor applied to V_q0 in the density coupling (RHS drive + readout). "
            "For eigenvalue-only workflows (e.g. FEAST), this coupling scale is irrelevant; "
            "only the kernel scale affects the spectrum."
        ),
    )
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

    v_scale_kernel = float(args.v_scale) if args.v_scale_kernel is None else float(args.v_scale_kernel)
    v_scale_coupling = float(args.v_scale) if args.v_scale_coupling is None else float(args.v_scale_coupling)

    V_q0_base = data["V_q0"]
    V_q0_kernel = V_q0_base * jnp.asarray(v_scale_kernel, dtype=V_q0_base.dtype)
    V_q0_coupling = V_q0_base * jnp.asarray(v_scale_coupling, dtype=V_q0_base.dtype)

    use_tda = args.tda
    include_W = not args.rpa
    v_couples_k = bool(args.rpa)

    if use_tda:
        matvec = build_bse_ring_matvec(
            mesh_xy, nkx, nky, nkz, include_W=include_W, v_couples_k=v_couples_k
        )
    else:
        matvec = build_bse_ring_matvec_full(
            mesh_xy, nkx, nky, nkz, include_W=include_W, v_couples_k=v_couples_k
        )

    if "W_R" not in data:
        if include_W:
            data["W_R"] = jnp.fft.ifftn(data["W_q"], axes=(2, 3, 4), norm="ortho")
        else:
            # Placeholder to satisfy matvec signature (not used when include_W=False).
            data["W_R"] = data["W_q"]

    data_op = data
    if args.d_only:
        data_op = dict(data)
        data_op["V_q0"] = jnp.zeros_like(V_q0_base)
        if "W_q" in data_op:
            data_op["W_q"] = jnp.zeros_like(data["W_q"])
        if "W_R" in data_op:
            data_op["W_R"] = jnp.zeros_like(data["W_R"])
    else:
        # Operator/preconditioner should see the kernel-scaled V.
        data_op = dict(data)
        data_op["V_q0"] = V_q0_kernel
    diag_h = build_preconditioner_diagonal_sharded(data_op, mesh_xy, include_W=include_W, use_tda=use_tda)

    # Density <-> transition couplers (RHS drive and readout).
    f_op, fbar_op = build_density_drive_operators(
        mesh_xy, nkx, nky, nkz, int(data["n_cond_pad"]), int(data["n_val_pad"])
    )
    snapshot_op = build_density_snapshot_operator(mesh_xy, nkx, nky, nkz)
    readout_full = build_density_readout_operator_full(mesh_xy, nkx, nky, nkz)
    sh = make_bse_shardings(mesh_xy)

    cols = _parse_cols(args.cols, n_rmu, args.n_cols, args.seed)
    print(f"Computing {len(cols)} column(s) out of N_mu={n_rmu}")

    z = (args.omega_ev + 1j * args.eta_ev) / args.ry_to_ev

    dtype_r = jnp.float32 if args.gmres_fp32 else jnp.float64
    dtype_c = jnp.complex64 if args.gmres_fp32 else jnp.complex128

    wc_cols = []
    gmres_iters = []
    output_scale = 1.0 if args.output_scale is None else float(args.output_scale)

    # If we need chi (density response), we'll invert V on the host.
    V_coupling_host = None
    if args.write_kind == "chi":
        V_coupling_host = np.asarray(jax.device_get(V_q0_coupling))

    for idx, nu0 in enumerate(cols, start=1):
        print(f"  Column {idx}/{len(cols)}: nu={nu0}")
        g = jnp.zeros((n_rmu,), dtype=dtype_r).at[nu0].set(1.0)
        r = jnp.broadcast_to(g[None, :, None], (1, n_rmu, nk))
        r = jax.device_put(r, sh.S)

        V_drive = V_q0_coupling
        if args.drive_kind == "potential":
            # Drive with u=g directly by setting the leading V to identity.
            V_drive = jnp.eye(n_rmu, dtype=V_q0_base.dtype)
            V_drive = jax.device_put(V_drive, sh.V)

        f = f_op(r, data["psi_c_X"], data["psi_v_X"], V_drive)
        f = jax.lax.with_sharding_constraint(f, sh.X)
        if use_tda:
            rhs = f.astype(dtype_c)
        else:
            # For the non-TDA Liouvillian S = [[A, B], [-B*, -A*]], the usual
            # density-coupling RHS is [f, -fbar] where:
            #   f    = d^\dagger (v g)
            #   fbar = d^T       (v g)
            #
            # For complex Bloch spinors we must not assume fbar == conj(f).
            # Our generator forms M=psi_c^* psi_v and projects with conj(M).
            # Passing conjugated wavefunctions flips that final conjugation,
            # yielding the transpose-coupled vertex fbar.
            fbar = fbar_op(r, data["psi_c_X"], data["psi_v_X"], V_drive)
            fbar = jax.lax.with_sharding_constraint(fbar, sh.X)
            rhs = jnp.stack([f, -fbar], axis=0).astype(dtype_c)
            rhs = jax.lax.with_sharding_constraint(rhs, sh.X_full)

        x, iters = gmres_solve_sharded_jit(
            matvec,
            diag_h,
            z,
            rhs,
            data_op,
            max_iter=args.gmres_max_iter,
            tol=args.gmres_tol,
        )
        gmres_iters.append(int(jax.device_get(iters)))

        if args.print_gmres_iters:
            # Iterations counts do not include the initial residual matvec used to form r0.
            print(f"    gmres_iters={gmres_iters[-1]}")

        if args.print_residuals:
            hx = matvec(
                x,
                data_op["psi_c_X"],
                data_op["psi_c_Y"],
                data_op["psi_v_X"],
                data_op["psi_v_Y"],
                data_op["eps_c"],
                data_op["eps_v"],
                data_op["W_R"],
                data_op["V_q0"],
            )
            r = rhs - (z * x - hx)
            r_norm = jnp.linalg.norm(r)
            b_norm = jnp.linalg.norm(rhs)
            rel = r_norm / (b_norm + jnp.asarray(1e-30, dtype=b_norm.dtype))
            rel_host, r_host, b_host = map(float, jax.device_get((rel, r_norm, b_norm)))
            print(f"    residual rel={rel_host:.3e} (||r||={r_host:.3e}, ||b||={b_host:.3e})")

        if use_tda:
            s = jax.lax.with_sharding_constraint(x, sh.X)
            w_c = snapshot_op(s, data["psi_c_Y"], data["psi_v_Y"], V_q0_coupling)
        else:
            w_c = readout_full(x, data["psi_c_Y"], data["psi_v_Y"], V_q0_coupling)

        out_host = np.asarray(jax.device_get(w_c[0]))
        if args.write_kind == "chi":
            if V_coupling_host is None:
                raise RuntimeError("internal error: V_coupling_host not initialized for write_kind=chi")
            # out_host is V * rho; invert to get rho (density response).
            out_host = np.linalg.solve(V_coupling_host, out_host)
        out_host = out_host * output_scale
        wc_cols.append(out_host)

    wc_cols = np.stack(wc_cols, axis=0) if wc_cols else np.zeros((0, n_rmu), dtype=np.complex128)
    gmres_iters_np = np.asarray(gmres_iters, dtype=np.int32)

    if args.print_gmres_iters and gmres_iters_np.size:
        # Per our GMRES implementation: total shifted-matvec calls ~= 1 + iters.
        avg_iters = float(np.mean(gmres_iters_np))
        avg_matvec = float(np.mean(gmres_iters_np + 1))
        print(
            f"GMRES iters: avg={avg_iters:.2f}, max={int(gmres_iters_np.max())}; "
            f"shifted-matvecs: avg~={avg_matvec:.2f}"
        )

    with h5py.File(args.out, "w") as h5:
        h5.attrs["omega_ev"] = float(args.omega_ev)
        h5.attrs["eta_ev"] = float(args.eta_ev)
        h5.attrs["ry_to_ev"] = float(args.ry_to_ev)
        h5.attrs["use_tda"] = int(use_tda)
        h5.attrs["include_W"] = int(include_W)
        h5.attrs["v_scale"] = float(args.v_scale)
        h5.attrs["v_scale_kernel"] = float(v_scale_kernel)
        h5.attrs["v_scale_coupling"] = float(v_scale_coupling)
        h5.attrs["gmres_max_iter"] = int(args.gmres_max_iter)
        h5.attrs["gmres_tol"] = float(args.gmres_tol)
        h5.attrs["gmres_iters_avg"] = float(np.mean(gmres_iters_np)) if gmres_iters_np.size else float("nan")
        h5.attrs["seed"] = int(args.seed)
        h5.attrs["density_channel"] = "rx_plus_rstar_y"
        h5.attrs["drive_kind"] = str(args.drive_kind)
        h5.attrs["write_kind"] = str(args.write_kind)
        h5.attrs["output_scale"] = float(output_scale)
        h5.create_dataset("columns", data=cols.astype(np.int32))
        h5.create_dataset(str(args.write_kind), data=wc_cols)
        h5.create_dataset("gmres_iters", data=gmres_iters_np)

    print(f"Wrote {args.write_kind} columns to {args.out}")
    timing.report(print_fn=print, title="--- Timing ---")


if __name__ == "__main__":
    main()
