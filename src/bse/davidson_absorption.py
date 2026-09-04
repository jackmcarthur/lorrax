"""BSE absorption via Davidson eigensolver — direct LORRAX analogue
of BGW's ``absorption.x diagonalization`` mode.

Produces the lowest ``n_eig`` *exact* eigenvectors (each converged to a
specified residual tolerance, not Lanczos finite-Krylov approximation
that mixes near-degenerate subspaces) plus their per-state dipole
projections, written as a BGW-format ``eigenvalues_b{1,2,3}.dat``.

Mirrors the matvec/precond/init setup of ``tests/bench/test_davidson_bse.py``
(known-working: converges Si 8×8 lowest 20 eigvals to ~3 meV vs BGW).

Usage:

    export LX_BASE_MODULE=lorrax_A LORRAX_CHECKOUT=$PWD
    lx run -N 1 -G 4 -n 4 -- env PYTHONPATH="$LORRAX_CHECKOUT/src" \\
        python3 -u -m bse.davidson_absorption \\
        -i cohsex.in --n-val 4 --n-cond 4 --n-occ 8 \\
        --eqp <bgw>/eqp.dat --dipole dipole_p_only.h5 \\
        --V-cell 270.107 --n-eig 20 --max-iter 80 \\
        --out-prefix eigenvalues_lorrax_davidson
"""
from __future__ import annotations
import argparse
import time

import h5py

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap()
# (env defaults + jax.distributed init + CPU fallback; all idempotent).
# MUST run before this module's own `import jax`.
from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from solvers.davidson import davidson, warmup_davidson_jit
from .absorption_common import (
    exciton_dipole_projections, load_dipole_h5, slice_dipole_to_bse_window,
    write_eigenvalues_dat,
)
from .bse_davidson_helpers import bse_diagonal_precond, init_bse_subspace
from .bse_io import (
    _find_restart_file,
    load_bse_data_from_restart_sharded, resolve_n_occ,
)
from .bse_ring_comm import create_mesh_2d, make_bse_shardings
from .bse_simple import build_bse_simple_matvec
from common.collectives import gather_to_host
from common.fft_helpers import make_sharded_ifftn_3d


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n\n")[0])
    p.add_argument("-i", "--input", required=True)
    p.add_argument("--n-val", type=int, required=True,
                   help="Kramers pairs (SOC) or SP bands (non-SOC). For Si SOC 8x8: --n-val 4")
    p.add_argument("--n-cond", type=int, required=True)
    p.add_argument("--n-occ", type=int, default=None,
                   help="Default: read from WFN.h5 ifmax via cohsex.in[wfn_file]")
    p.add_argument("--eqp", default=None, help="BGW eqp.dat (recommended)")
    p.add_argument("--dipole", default="dipole_p_only.h5",
                   help="Use --skip-vnl dipole for BGW use_momentum match")
    p.add_argument("--V-cell", type=float, required=True, help="Unit-cell volume bohr³")
    p.add_argument("--n-eig", type=int, default=20)
    p.add_argument("--n-random-init", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--out-prefix", default="eigenvalues_davidson")
    args = p.parse_args(argv)

    restart_file = _find_restart_file(args.input)
    mesh_xy = create_mesh_2d()
    grid_x, grid_y = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)

    rank0 = jax.process_index() == 0
    if rank0:
        print(f"[davidson_absorption] mesh = {grid_x}×{grid_y} = {jax.device_count()} devices",
              flush=True)

    data = load_bse_data_from_restart_sharded(
        restart_file, n_val=args.n_val, n_cond=args.n_cond, mesh_xy=mesh_xy,
        input_file=args.input, n_occ=args.n_occ,
    )

    # Apply EQP via the same flow as bse_jax._preview_lanczos /
    # tests/bench/test_davidson_bse.py: explicit n_occ-based slice, NOT nearest-energy
    # matching (which silently mis-selects bands after QP shifts).
    if args.eqp is not None:
        from .bse_io import apply_eqp_and_reslice_bands
        data["eps_v"], data["eps_c"], n_occ_eff = apply_eqp_and_reslice_bands(
            restart_file, args.eqp, args.input,
            int(data["n_val"]), int(data["n_cond"]), args.n_occ, grid_x, grid_y)
        if rank0:
            print(f"[davidson_absorption] applied EQP {args.eqp} (n_occ={n_occ_eff})", flush=True)

    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc_pad = int(data["n_cond_pad"])
    nv_pad = int(data["n_val_pad"])
    if rank0:
        print(f"[davidson_absorption] BSE: {nc_pad}c × {nv_pad}v × {nk}k = "
              f"{nc_pad*nv_pad*nk}, n_eig={args.n_eig}, max_iter={args.max_iter}", flush=True)

    # ── Build matvec exactly like tests/bench/test_davidson_bse.py ─────────────────────
    matvec_simple = build_bse_simple_matvec(mesh_xy, nkx, nky, nkz, include_W=True)
    _W_ifftn = make_sharded_ifftn_3d(
        mesh_xy, sh.W.spec, sh.W.spec, axes=(2, 3, 4), norm="ortho")
    # W_q is DONATED, and the caller-side reference dropped — copied verbatim
    # from ``bse_lanczos``'s W_R build, which is the same transform at the same
    # kind of top-level boundary.  Same shape in and out, so XLA grants the
    # alias and W_R simply *becomes* W_q's buffer; without it both tiles stay
    # resident for the whole absorption run.  In-jit peak is identical either
    # way (measured, FFT_DONATION_AUDIT.md 2.3: 112.50 MiB/rank donated and
    # not) — the whole win is caller-side, one W tile instead of two:
    # 56.25 MiB/rank on the Si 4x4x4 deck, 404 MB/rank at mu=10015 / P=64,
    # 4.1 GB/rank at mu=32k.  Value-identical: same helper, same axes, same
    # norm, same operand.  Dropping the reference is REQUIRED — donation alone
    # leaves a deleted-buffer array in the dict, which is a worse failure than
    # the leak.
    W_R = jax.jit(_W_ifftn, in_shardings=sh.W, out_shardings=sh.W,
                  donate_argnums=(0,))(data["W_q"])
    data["W_q"] = None          # release the caller-side reference
    jax.block_until_ready(W_R)

    psi_c_X, psi_c_Y = data["psi_c_X"], data["psi_c_Y"]
    psi_v_X, psi_v_Y = data["psi_v_X"], data["psi_v_Y"]
    eps_c, eps_v = data["eps_c"], data["eps_v"]
    V_q0 = data["V_q0"]
    M_X, M_Y = data["M_X"], data["M_Y"]  # hoisted V-term pair-amps (audit P3)

    # The simple matvec's workspace scales worse than linearly with the
    # batch axis m (~m^1.5; 4.5 GB at m=10 → 88 GB at m=50). The state
    # vectors themselves are tiny. Use lax.scan over m=1 calls so XLA
    # reuses the m=1 workspace.
    #
    # Multi-host: jit closures over sharded arrays raise
    # "Closing over jax.Array that spans non-addressable devices".
    # Pass psi_*, eps_*, W_R, V_q0 as arguments to the jit'd function;
    # the outer apply_H is plain python that forwards them at call time.
    @jax.jit
    def matvec_scan(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M_X, M_Y):
        def body(carry, x_one):
            Hx = matvec_simple(x_one[None], psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                               eps_c, eps_v, W_R, V_q0, M_X, M_Y)
            return carry, Hx[0]
        _, HX = jax.lax.scan(body, None, X)
        return HX

    def apply_H(X):
        return matvec_scan(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                           eps_c, eps_v, W_R, V_q0, M_X, M_Y)

    # ── Initial subspace ───────────────────────────────────────────────
    V0 = init_bse_subspace(
        eps_c, eps_v, n_eig=args.n_eig,
        n_cond=int(data["n_cond"]), n_val=int(data["n_val"]),
        n_random=args.n_random_init,
        mesh=mesh_xy, sharding=sh.X, seed=42,
    )
    if rank0:
        print(f"[davidson_absorption] V0.shape = {V0.shape}", flush=True)

    # ── Preconditioner ─────────────────────────────────────────────────
    delta_E_sh = NamedSharding(mesh_xy, P("x", "y", None))
    precond_fn = bse_diagonal_precond(
        eps_c, eps_v, epsilon_shift=1e-3, sharding=delta_E_sh,
    )

    # ── Warmup Ritz JIT at all subspace sizes ─────────────────────────
    warmup_davidson_jit(
        args.n_eig, (nc_pad, nv_pad, nk),
        m_max=4 * args.n_eig, dtype=jnp.complex128, sharding=sh.X,
    )

    # ── Davidson ────────────────────────────────────────────────────
    t0 = time.time()
    eigvals, eigvecs = davidson(
        apply_H, n_eig=args.n_eig, precond_fn=precond_fn, X0=V0,
        m_max=4 * args.n_eig, max_iter=args.max_iter, tol=args.tol,
        verbose=rank0,
    )
    jax.block_until_ready(eigvecs)
    if rank0:
        print(f"[davidson_absorption] Davidson done in {time.time()-t0:.1f}s", flush=True)

    eigvals_np = gather_to_host(eigvals)
    eigvecs_np = gather_to_host(eigvecs)

    if not rank0:
        return

    ryd2ev = 13.6056980659
    eigvals_eV = eigvals_np * ryd2ev
    print(f"[davidson_absorption] lowest 5 (eV): {eigvals_eV[:5]}", flush=True)

    # Dipole projection on each eigvec.
    dipole_cart, deltaE, _ = load_dipole_h5(args.dipole)
    n_occ_for_slice = args.n_occ if args.n_occ is not None else resolve_n_occ(
        np.asarray(gather_to_host(eps_v)), input_file=args.input)
    d_alpha, _ = slice_dipole_to_bse_window(
        dipole_cart, deltaE, n_occ=n_occ_for_slice, n_val=nv_pad, n_cond=nc_pad)
    d_block = np.transpose(d_alpha, (0, 2, 3, 1))   # (3, nc, nv, nk)
    # ⟨0|r̂_α|S⟩ = Σ_t A^S_t conj(d_t), through the one shared site.  This
    # line used to read ``einsum("Scvk,acvk->Sa", eigvecs_np.conj(), d_block)``
    # — the conjugate on A rather than on d.  Its MODULUS was right (the two
    # spellings are complex conjugates, and |·|² is what ε₂ consumes), so this
    # driver's spectra do not move; what moves is the sign of the ``Im``
    # column in the BGW-format eigenvalues_b*.dat, which now matches BGW's own
    # spelling at BSE/diag.f90:711.  The other driver, ``absorption_eigvecs``,
    # conjugated NEITHER factor and was wrong in modulus too.
    proj = exciton_dipole_projections(eigvecs_np, d_block)

    V_super = args.V_cell * nk
    for a, suffix in enumerate(("b1", "b2", "b3")):
        out = f"{args.out_prefix}_{suffix}.dat"
        write_eigenvalues_dat(
            out, eigvals_eV, proj[:, a],
            n_spin=1, n_spinor=2, vol_supercell=V_super,
        )
        print(f"[davidson_absorption] wrote {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
