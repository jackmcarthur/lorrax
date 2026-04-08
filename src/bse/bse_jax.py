"""BSE JAX entry points and CLI wrappers."""
from __future__ import annotations

import sys


import jax
import jax.numpy as jnp
from jax import lax

from .bse_ring_comm import (
    apply_bse_hamiltonian_ring,
    build_bse_ring_matvec,
    build_bse_ring_matvec_full,
    create_mesh_2d,
    make_bse_shardings,
    ring_matvec_correctness_check,
    ring_matvec_smoke_test,
    ring_matvec_timing,
)
from .bse_io import _find_restart_file, _load_ring_subset
from .bse_serial import (
    apply_bse_hamiltonian_single_device,
    apply_bse_hamiltonian_single_device_jit,
    symmetrize_W_q,
)
from .bse_lanczos import (
    block_lanczos_eig,
    lanczos_eig_jit,
    simple_lanczos_eig,
    solve_bse,
)
from .bse_io import write_eigenvectors_stream
from .bse_preconditioner import energy_diff_cv_k

jax.config.update("jax_enable_x64", True)

__all__ = [
    "apply_D",
    "apply_bse_hamiltonian",
    "apply_bse_hamiltonian_ring",
    "apply_bse_hamiltonian_single_device",
    "apply_bse_hamiltonian_single_device_jit",
    "apply_V",
    "apply_W",
    "block_lanczos_eig",
    "build_bse_ring_matvec",
    "build_bse_ring_matvec_full",
    "compute_pair_amplitude",
    "create_mesh_2d",
    "lanczos_eig_jit",
    "make_bse_shardings",
    "ring_matvec_correctness_check",
    "ring_matvec_smoke_test",
    "ring_matvec_timing",
    "simple_lanczos_eig",
    "solve_bse",
    "symmetrize_W_q",
]


@jax.jit
def apply_bse_hamiltonian(
    X: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
    psi_c_X: jax.Array,
    psi_c_Y: jax.Array,
    psi_v_X: jax.Array,
    psi_v_Y: jax.Array,
    eps_c: jax.Array,
    eps_v: jax.Array,
    W_q: jax.Array,
    V_q0: jax.Array,
) -> jax.Array:
    nk = nkx * nky * nkz
    D_term = apply_D(X, eps_c, eps_v)
    V_term = apply_V(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, V_q0, nk)
    W_term = apply_W(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_q, nkx, nky, nkz)
    return D_term + V_term - W_term


def apply_D(X: jax.Array, eps_c: jax.Array, eps_v: jax.Array) -> jax.Array:
    delta_E = energy_diff_cv_k(eps_c, eps_v)[None, :, :, :]
    return delta_E * X


def compute_pair_amplitude(psi_c: jax.Array, psi_v: jax.Array) -> jax.Array:
    return jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c), psi_v)


def apply_V(
    X: jax.Array,
    psi_c_X: jax.Array,
    psi_c_Y: jax.Array,
    psi_v_X: jax.Array,
    psi_v_Y: jax.Array,
    V_q0: jax.Array,
    nk: int,
) -> jax.Array:
    M_Y = compute_pair_amplitude(psi_c_Y, psi_v_Y)
    S_partial = jnp.einsum("kcvN,bcvk->bNk", M_Y, X)
    S = lax.psum(S_partial, axis_name="x")

    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))
    S = S / sqrt_nk

    U_partial = jnp.einsum("MN,bNk->bMk", V_q0, S)
    U = lax.psum(U_partial, axis_name="y")

    M_X = compute_pair_amplitude(psi_c_X, psi_v_X)
    VX_partial = jnp.einsum("kcvM,bMk->bcvk", jnp.conj(M_X), U)
    VX = lax.psum_scatter(VX_partial, axis_name="x", scatter_dimension=1, tiled=True)

    return VX / sqrt_nk


def apply_W(
    X: jax.Array,
    psi_c_X: jax.Array,
    psi_c_Y: jax.Array,
    psi_v_X: jax.Array,
    psi_v_Y: jax.Array,
    W_q: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
) -> jax.Array:
    nk = nkx * nky * nkz
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))

    nspinor = psi_c_X.shape[2]
    n_rmu_local_Y = psi_v_Y.shape[-1]
    n_rmu_local_X = psi_c_X.shape[-1]

    R_partial = jnp.einsum("kv sN,bcvk->bcksN", jnp.conj(psi_v_Y), X)
    T_partial = jnp.einsum("kctM,bcksN->bMNtsk", psi_c_X, R_partial)
    T = lax.psum(T_partial, axis_name="x")

    T_k = T.reshape(X.shape[0], n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nkx, nky, nkz)
    T_R = jnp.fft.ifftn(T_k, axes=(5, 6, 7), norm="ortho")

    W_R = jnp.fft.ifftn(W_q, axes=(2, 3, 4), norm="ortho")
    U_R = W_R[None, :, :, None, None, :, :, :] * T_R

    U_q = jnp.fft.fftn(U_R, axes=(5, 6, 7), norm="ortho")
    U = U_q.reshape(X.shape[0], n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nk)

    A_partial = jnp.einsum("kctM,bMNtsk->bcNsk", jnp.conj(psi_c_X), U)
    WX_partial = jnp.einsum("kvsN,bcNsk->bcvk", psi_v_Y, A_partial)
    WX_nu = lax.psum(WX_partial, axis_name="y")
    WX = lax.psum_scatter(WX_nu, axis_name="x", scatter_dimension=1, tiled=True)

    return WX / sqrt_nk


def _main_random_demo() -> None:
    print("Testing BSE matvec with random data...")

    nk, nc, nv, nspinor, n_rmu = 8, 4, 4, 2, 32
    nkx, nky, nkz = 2, 2, 2

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 7)

    psi_c = jax.random.normal(keys[0], (nk, nc, nspinor, n_rmu)) + \
            1j * jax.random.normal(keys[1], (nk, nc, nspinor, n_rmu))
    psi_v = jax.random.normal(keys[2], (nk, nv, nspinor, n_rmu)) + \
            1j * jax.random.normal(keys[3], (nk, nv, nspinor, n_rmu))

    eps_v = jax.random.uniform(keys[4], (nk, nv), minval=-0.5, maxval=-0.1)
    eps_c = jax.random.uniform(keys[5], (nk, nc), minval=0.1, maxval=0.5)

    W_q = jax.random.normal(keys[6], (n_rmu, n_rmu, nkx, nky, nkz)) * 0.01
    V_q0 = jnp.eye(n_rmu) * 0.05

    X = jnp.ones((1, nc, nv, nk), dtype=jnp.complex128)
    X = X / jnp.linalg.norm(X)

    HX = apply_bse_hamiltonian_single_device(
        X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz
    )
    print(f"Input shape: {X.shape}, Output shape: {HX.shape}")
    E_expect = jnp.vdot(X.flatten(), HX.flatten()).real
    ryd2ev = 13.6056980659
    print(f"Expectation value: {E_expect:.6f} Ry = {E_expect * ryd2ev:.4f} eV")

    print("\nRunning Lanczos solver...")
    eigenvalues, _ = solve_bse(
        psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz,
        n_eig=5, max_iter=30,
    )
    print(f"Lowest 5 eigenvalues (Ry): {eigenvalues}")
    print(f"Lowest 5 eigenvalues (eV): {eigenvalues * ryd2ev}")


def _preview_lanczos(
    input_file: str,
    n_val: int,
    n_cond: int,
    n_eig: int = 5,
    write_eigs: int | None = None,
    max_lanczos_iter: int | None = None,
    include_W: bool = True,
    eqp_file: str | None = None,
    n_occ: int | None = None,
) -> None:
    restart_file = _find_restart_file(input_file)
    payload = _load_ring_subset(restart_file, n_val, n_cond, 1, 1, eqp_file=eqp_file, n_occ=n_occ)
    psi_c = payload["psi_c"]
    psi_v = payload["psi_v"]
    eps_c = payload["eps_c"]
    eps_v = payload["eps_v"]
    W_q = payload["W_q"]
    V_q0 = payload["V_q0"]
    nkx = payload["nkx"]
    nky = payload["nky"]
    nkz = payload["nkz"]

    nk = nkx * nky * nkz
    nc_actual = psi_c.shape[1]
    nv_actual = psi_v.shape[1]
    bse_dim = nc_actual * nv_actual * nk
    print(f"BSE problem: {nc_actual} cond x {nv_actual} val x {nk} k = {bse_dim} dimension")

    if max_lanczos_iter is None:
        max_lanczos_iter = max(30, min(200, bse_dim // 2))
    print(f"Lanczos: {max_lanczos_iter} iterations")

    eigenvalues, eigenvectors = solve_bse(
        psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz,
        n_eig=n_eig, max_iter=max_lanczos_iter, include_W=include_W,
    )
    ryd2ev = 13.6056980659
    print(f"Lowest {n_eig} eigenvalues (Ry): {eigenvalues}")
    print(f"Lowest {n_eig} eigenvalues (eV): {eigenvalues * ryd2ev}")

    if write_eigs is not None:
        n_write = n_eig if write_eigs < 0 else min(write_eigs, n_eig)
        write_eigenvectors_stream(
            "eigenvectors.h5",
            eigenvalues,
            eigenvectors,
            n_val,
            n_cond,
            nkx,
            nky,
            nkz,
            n_write,
        )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BSE JAX entry point")
    parser.add_argument("-i", "--input", help="COHSEX input file (for canonical isdf_tensors_*.h5 lookup)")
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--n-cond", type=int, default=4)
    parser.add_argument("--px", type=int, default=1)
    parser.add_argument("--py", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--n-eig", type=int, default=5)
    parser.add_argument("--feast-n-lanczos", type=int, default=10, help="Lanczos steps for FEAST bounds.")
    parser.add_argument("--feast-buffer", type=float, default=0.05, help="Emax buffer fraction for FEAST windows.")
    parser.add_argument("--feast-n-quad1", type=int, default=4, help="Quadrature points for FEAST iteration 1.")
    parser.add_argument("--feast-n-quad2", type=int, default=8, help="Quadrature points for FEAST iteration 2+.")
    parser.add_argument("--feast-quadrature", type=str, default="ellipse",
                        choices=["zolotarev", "ellipse"],
                        help="Quadrature type for FEAST filter (default: ellipse).")
    parser.add_argument(
        "--feast-units-ev-per-ry",
        type=float,
        default=13.6056980659,
        help="Energy conversion Ry -> eV for FEAST report.",
    )
    parser.add_argument("--feast-ritz-count", type=int, default=4, help="Ritz values per window.")
    parser.add_argument("--gmres-max-iter", type=int, default=10, help="GMRES iterations per shift.")
    parser.add_argument("--gmres-tol", type=float, default=1e-2, help="GMRES relative tolerance.")
    parser.add_argument("--gmres-seed", type=int, default=0, help="GMRES random seed.")
    parser.add_argument("--gmres-fp32", action="store_true",
                        help="Use FP32 data/GMRES for shifted solves.")
    parser.add_argument("--tda", action="store_true",
                        help="Use Tamm-Dancoff approximation (TDA). Default is full non-TDA.")
    parser.add_argument("--rpa", action="store_true",
                        help="Use RPA kernel (D+V only), skip W0 term entirely.")
    parser.add_argument("--bse", action="store_true",
                        help="Use full BSE kernel (D+V-W). Overrides default RPA.")
    parser.add_argument("--kpm-window-count", type=int, default=4,
                        help="Number of KPM-derived windows for FEAST.")
    parser.add_argument("--lanczos", action="store_true", help="Run Lanczos preview + timing instead of FEAST.")
    parser.add_argument(
        "--feast-window1",
        nargs=2,
        metavar=("A", "B"),
        help="Override FEAST window 1 bounds in eV (use 'auto' for B).",
    )
    parser.add_argument(
        "--feast-window2",
        nargs=2,
        metavar=("A", "B"),
        help="Override FEAST window 2 bounds in eV (use 'auto' for B).",
    )
    parser.add_argument(
        "--write-eigs",
        nargs="?",
        const=-1,
        type=int,
        help="Write eigenvectors.h5 (optional N, default: n-eig).",
    )
    parser.add_argument(
        "--max-lanczos-iter",
        type=int,
        default=None,
        help="Lanczos iterations for eigensolve (default: auto-scale with problem size).",
    )
    parser.add_argument("--kpm-dos", action="store_true", help="Run KPM Chebyshev DOS and exit.")
    parser.add_argument("--kpm-n-moments", type=int, default=100, help="Chebyshev moments M for KPM.")
    parser.add_argument("--kpm-n-random", type=int, default=4, help="Stochastic trace vectors R for KPM.")
    parser.add_argument("--kpm-n-lanczos", type=int, default=100, help="Lanczos steps for KPM spectral bounds.")
    parser.add_argument("--kpm-emin-ev", type=float, default=None, help="Override KPM E_min (eV).")
    parser.add_argument("--kpm-emax-ev", type=float, default=None, help="Override KPM E_max (eV).")
    parser.add_argument("--kpm-plot-file", type=str, default="bse_dos_kpm.png", help="KPM DOS plot output file.")
    parser.add_argument("--eqp", type=str, default=None,
                        help="Path to BGW eqp1.dat file for QP energy corrections.")
    parser.add_argument("--n-occ", type=int, default=None,
                        help="Number of occupied (valence) bands. Overrides auto-detection from energy sign.")
    parser.add_argument("--ring-test", action="store_true")
    parser.add_argument("--ring-check", action="store_true")
    parser.add_argument("--ring-timing", action="store_true")
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--debug-parallelism", action="store_true")
    args, _ = parser.parse_known_args()

    if args.ring_test:
        ring_matvec_smoke_test()
        raise SystemExit(0)

    if args.ring_check:
        if args.input is None:
            parser.error("--ring-check requires -i/--input")
        ring_matvec_correctness_check(
            args.input,
            args.n_val,
            args.n_cond,
            args.px,
            args.py,
            args.components,
        )
        raise SystemExit(0)

    if args.debug_parallelism:
        _main_random_demo()
        raise SystemExit(0)

    if args.input is None:
        parser.error("Default run requires -i/--input (use --debug-parallelism for random data).")

    use_tda = args.tda

    if args.kpm_dos:
        from . import bse_kpm

        use_rpa = args.rpa or not args.bse
        kpm_argv = [
            "-i", args.input,
            "--n-val", str(args.n_val),
            "--n-cond", str(args.n_cond),
            "--px", str(args.px),
            "--py", str(args.py),
            "--n-moments", str(args.kpm_n_moments),
            "--n-random", str(args.kpm_n_random),
            "--n-lanczos", str(args.kpm_n_lanczos),
            "--n-windows", str(args.kpm_window_count),
            "--plot-file", args.kpm_plot_file,
        ]
        if use_rpa:
            kpm_argv.append("--rpa")
        if use_tda:
            kpm_argv.append("--tda")
        if args.kpm_emin_ev is not None:
            kpm_argv += ["--emin-ev", str(args.kpm_emin_ev)]
        if args.kpm_emax_ev is not None:
            kpm_argv += ["--emax-ev", str(args.kpm_emax_ev)]
        bse_kpm.main(kpm_argv)
        raise SystemExit(0)

    if not args.lanczos:
        from . import bse_feast

        use_rpa = args.rpa or not args.bse
        bse_feast.main(
            [
                "-i",
                args.input,
                "--n-val",
                str(args.n_val),
                "--n-cond",
                str(args.n_cond),
                "--px",
                str(args.px),
                "--py",
                str(args.py),
                "--n-lanczos",
                str(args.feast_n_lanczos),
                "--buffer",
                str(args.feast_buffer),
                "--n-quad1",
                str(args.feast_n_quad1),
                "--n-quad2",
                str(args.feast_n_quad2),
                "--quadrature",
                args.feast_quadrature,
                "--units-ev-per-ry",
                str(args.feast_units_ev_per_ry),
                "--feast-ritz",
                "--feast-ritz-count",
                str(args.feast_ritz_count),
                "--gmres-max-iter",
                str(args.gmres_max_iter),
                "--gmres-tol",
                str(args.gmres_tol),
                "--gmres-seed",
                str(args.gmres_seed),
                *(["--gmres-fp32"] if args.gmres_fp32 else []),
                *(["--rpa"] if use_rpa else []),
                *(["--tda"] if use_tda else []),
                "--windows-kpm",
                "--windows-kpm-count",
                str(args.kpm_window_count),
                "--kpm-n-moments",
                str(args.kpm_n_moments),
                "--kpm-n-random",
                str(args.kpm_n_random),
                "--kpm-seed",
                str(args.gmres_seed),
                "--kpm-n-energy-pts",
                "2000",
                "--kpm-n-lanczos",
                str(args.kpm_n_lanczos),
                *(
                    ["--window1", *args.feast_window1]
                    if args.feast_window1 is not None
                    else []
                ),
                *(
                    ["--window2", *args.feast_window2]
                    if args.feast_window2 is not None
                    else []
                ),
            ]
        )
        raise SystemExit(0)

    if not use_tda:
        raise SystemExit("Lanczos preview currently supports TDA only. Re-run with --tda.")

    _preview_lanczos(
        args.input,
        args.n_val,
        args.n_cond,
        n_eig=args.n_eig,
        write_eigs=args.write_eigs,
        max_lanczos_iter=args.max_lanczos_iter,
        include_W=not (args.rpa or not args.bse),
        eqp_file=args.eqp,
        n_occ=args.n_occ,
    )
    raise SystemExit(0)

    ring_matvec_timing(
        args.input,
        args.n_val,
        args.n_cond,
        args.px,
        args.py,
        args.repeat,
        args.warmup,
        True,
    )
