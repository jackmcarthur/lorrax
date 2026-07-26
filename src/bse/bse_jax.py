"""BSE JAX entry points and CLI wrappers."""
from __future__ import annotations

import os
import sys

# Ensure x64 + jax.distributed bootstrap before any jax-collective code
# (the ring matvec uses lax.psum/ppermute on the 2D mesh, which is silent-
# wrong if processes don't agree on a shared distributed runtime).
# Single-sourced in runtime.bootstrap() (env defaults + jax.distributed
# init + CPU fallback); MUST run before this module's own `import jax`.
from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp

from .bse_ring_comm import (
    build_bse_ring_matvec,
    build_bse_ring_matvec_full,
    create_mesh_2d,
    make_bse_shardings,
    ring_matvec_correctness_check,
    ring_matvec_smoke_test,
)
from .bse_io import _find_restart_file, _load_ring_subset
from .bse_serial import (
    apply_bse_hamiltonian_single_device,
    apply_bse_hamiltonian_single_device_jit,
)
from .bse_lanczos import (
    block_lanczos_eig,
    lanczos_eig_jit,
    simple_lanczos_eig,
    solve_bse,
)
from .bse_io import write_eigenvectors_stream

jax.config.update("jax_enable_x64", True)

__all__ = [
    "apply_bse_hamiltonian_single_device",
    "apply_bse_hamiltonian_single_device_jit",
    "block_lanczos_eig",
    "build_bse_ring_matvec",
    "build_bse_ring_matvec_full",
    "compute_pair_amplitude",
    "create_mesh_2d",
    "lanczos_eig_jit",
    "make_bse_shardings",
    "ring_matvec_correctness_check",
    "ring_matvec_smoke_test",
    "simple_lanczos_eig",
    "solve_bse",
]


def compute_pair_amplitude(psi_c: jax.Array, psi_v: jax.Array) -> jax.Array:
    return jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c), psi_v)


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
    block_size: int = 1,
    rtol: float = 0.0,
    check_every: int = 4,
    matvec_kind: str = "ring",
    n_reorth: int = -1,
    solver_kind: str = "lanczos",
    tda: bool = True,
) -> None:
    restart_file = _find_restart_file(input_file)
    n_devices = jax.device_count()
    # Non-TDA has no 1-device (``solve_bse``) path — it runs through the sharded
    # loader + ``solve_bse_sharded(tda=False)`` on a 1x1 mesh just as well.
    use_sharded = n_devices > 1 or not tda

    if use_sharded:
        # Sharded ring matvec — parallelises (μ,ν) and avoids per-iter
        # 3D-FFT of W_q (precomputes W_R once outside the Lanczos loop).
        from .bse_io import load_bse_data_from_restart_sharded
        from .bse_lanczos import solve_bse_sharded
        mesh_xy = create_mesh_2d()
        # n_occ-aware band split: load_bse_data_from_restart_sharded
        # auto-detects valence by ``mean_enk < fermi_energy``; user-given
        # n_occ replaces that detection identically to _load_ring_subset.
        # We pass fermi_energy=0.0 (default) and rely on enk_full's reference.
        data = load_bse_data_from_restart_sharded(
            restart_file, n_val=n_val, n_cond=n_cond, mesh_xy=mesh_xy,
            input_file=input_file, n_occ=n_occ,
        )
        # T-encoding strategy plumbed via the data dict (see solve_bse_sharded).
        data["matvec_kind"] = matvec_kind
        grid_x, grid_y = mesh_xy.devices.shape
        # EQP override on enk_full (BGW eqp1.dat semantics).
        if eqp_file is not None:
            # Re-slice the band window on the eqp-corrected energies. Uses the
            # loader-CLAMPED band counts (data['n_val']/data['n_cond']), not the
            # raw CLI n_val/n_cond, so an over-request can't slice out of bounds.
            from .bse_io import apply_eqp_and_reslice_bands
            data["eps_v"], data["eps_c"], _ = apply_eqp_and_reslice_bands(
                restart_file, eqp_file, input_file,
                int(data["n_val"]), int(data["n_cond"]), n_occ, grid_x, grid_y)
        nkx = data["nkx"]; nky = data["nky"]; nkz = data["nkz"]
        nk = nkx * nky * nkz
        nc_pad = int(data["n_cond_pad"])
        nv_pad = int(data["n_val_pad"])
        bse_dim = nc_pad * nv_pad * nk
        print(f"BSE problem (sharded {grid_x}x{grid_y}): "
              f"{nc_pad} cond × {nv_pad} val × {nk} k = {bse_dim} dim")
        if max_lanczos_iter is None:
            max_lanczos_iter = max(30, min(200, bse_dim // 2))
        # ``max_lanczos_iter`` is the *total* Krylov dimension upper
        # bound; block path divides by block_size.
        block_max_iter = max(1, max_lanczos_iter // max(1, block_size))
        mode = (f"convergence-driven (rtol={rtol:.1e}, every {check_every} iters)"
                if rtol > 0 else "fixed")
        if block_size > 1:
            print(f"Block Lanczos [{mode}]: ≤ {block_max_iter} block iter × "
                  f"block_size={block_size} = ≤ {block_max_iter * block_size} Krylov dim")
        else:
            print(f"Lanczos [{mode}]: ≤ {block_max_iter} iterations")
        # Resolve full-reorth sentinel (-1) to the actual Krylov depth.
        n_reorth_eff = block_max_iter if n_reorth < 0 else n_reorth
        eigenvalues, eigenvectors, n_iter_done = solve_bse_sharded(
            data, mesh_xy, n_eig=n_eig, max_iter=block_max_iter,
            include_W=include_W, block_size=block_size,
            rtol=rtol, check_every=check_every, n_reorth=n_reorth_eff,
            solver_kind=solver_kind, tda=tda,
        )
        n_done = int(n_iter_done)
        if rtol > 0:
            tag = "Block Lanczos" if block_size > 1 else "Lanczos"
            print(f"{tag} exited at iter {n_done}/{block_max_iter} "
                  f"(Krylov dim = {n_done * block_size})")
    else:
        payload = _load_ring_subset(
            restart_file,
            n_val,
            n_cond,
            1,
            1,
            eqp_file=eqp_file,
            n_occ=n_occ,
            input_file=input_file,
        )
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
            use_tda=tda,
        )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False, description="BSE JAX entry point")
    parser.add_argument("-i", "--input", help="COHSEX input file (for canonical isdf_tensors_*.h5 lookup)")
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--n-cond", type=int, default=4)
    parser.add_argument("--px", type=int, default=1)
    parser.add_argument("--py", type=int, default=1)
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
        help="Lanczos iterations for eigensolve (default: auto-scale with problem size). "
             "When --block-size > 1, this is total Krylov dimension (= block iters × block_size).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="Block-Lanczos block size (default 1 = single-vector Lanczos). "
             "Larger blocks → larger per-call GEMMs and fewer host dispatches.",
    )
    parser.add_argument(
        "--lanczos-rtol",
        type=float,
        default=0.0,
        help="Convergence threshold on Ritz-eigenvalue change between checks. "
             "0 (default) = fixed --max-lanczos-iter iterations, no early exit. "
             ">0 enables ``lax.while_loop`` Lanczos that exits when the lowest "
             "n-eig Ritz values stabilise within rtol (block-Lanczos only).",
    )
    parser.add_argument(
        "--lanczos-check-every",
        type=int,
        default=4,
        help="Convergence check cadence in block iterations (default 4).",
    )
    parser.add_argument(
        "--n-reorth",
        type=int,
        default=-1,
        help="Lanczos partial-reorthogonalisation window. -1 (default) "
             "= full reorth (= max_lanczos_iter); essential for highly "
             "degenerate spectra (e.g. spinor BSE) so Ritz vectors stay "
             "orthogonal across the full Krylov basis. Smaller windows "
             "(e.g. 10) are faster but give ghost eigenvalues that "
             "destroy per-state oscillator strengths.",
    )
    parser.add_argument(
        "--matvec-kind",
        choices=("ring", "gather", "simple"),
        default="ring",
        help="BSE matvec implementation. ``ring`` (default): shard_map + "
             "lax.ppermute (low memory). ``gather``: shard_map + lax.all_gather "
             "(faster on small problems). ``simple``: plain jit + jnp.einsum "
             "+ with_sharding_constraint, no shard_map (XLA auto-partitions).",
    )
    parser.add_argument(
        "--gather-t",
        action="store_true",
        help="(Deprecated alias for --matvec-kind=gather)",
    )
    parser.add_argument(
        "--solver",
        choices=("lanczos", "davidson"),
        default="lanczos",
        help="Eigensolver. ``lanczos`` (default) for fast spectrum-shape "
             "convergence (ε₂(ω)). ``davidson`` for tight per-state "
             "eigenvector convergence using diagonal (E_c−E_v) "
             "preconditioner — better for individual-state oscillator "
             "strengths in densely-packed band-edge spectra.",
    )
    parser.add_argument("--kpm-dos", action="store_true", help="Run KPM Chebyshev DOS and exit.")
    parser.add_argument("--kpm-n-moments", type=int, default=100, help="Chebyshev moments M for KPM.")
    parser.add_argument("--kpm-n-random", type=int, default=4, help="Stochastic trace vectors R for KPM.")
    parser.add_argument("--kpm-n-lanczos", type=int, default=100, help="Lanczos steps for KPM spectral bounds.")
    parser.add_argument("--kpm-emin-ev", type=float, default=None, help="Override KPM E_min (eV).")
    parser.add_argument("--kpm-emax-ev", type=float, default=None, help="Override KPM E_max (eV).")
    parser.add_argument("--kpm-plot-file", type=str, default="bse_dos_kpm.png", help="KPM DOS plot output file.")
    parser.add_argument("--eqp", type=str, default=None, help="Path to BGW eqp1.dat for QP corrections.")
    parser.add_argument("--n-occ", type=int, default=None, help="Number of occupied bands.")
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
        # Propagate the sub-driver's exit code instead of hardcoding 0:
        # delegating used to launder any failure it reported into rc 0.
        raise SystemExit(bse_kpm.main(kpm_argv) or 0)

    if not args.lanczos:
        from . import bse_feast

        use_rpa = args.rpa or not args.bse
        _feast_rc = bse_feast.main(
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
        # Propagate the sub-driver's exit code (see the kpm branch above).
        raise SystemExit(_feast_rc or 0)

    # Non-TDA (full BSE) now flows through the same preview via the
    # ``solve_bse_sharded(tda=False)`` dispatch -> ``bse_nontda`` (structure-
    # preserving definite-pencil / product solve).  TDA stays the default.
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
        block_size=args.block_size,
        rtol=args.lanczos_rtol,
        check_every=args.lanczos_check_every,
        matvec_kind=("gather" if args.gather_t else args.matvec_kind),
        n_reorth=args.n_reorth,
        solver_kind=args.solver,
        tda=use_tda,
    )
    raise SystemExit(0)
