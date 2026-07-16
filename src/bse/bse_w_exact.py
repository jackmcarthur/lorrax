"""Exact W_c(omega) via shifted solves with the non-TDA RPA density resolvent.

Cross-validates the GW screened Coulomb.  In the ISDF centroid (density) basis
the screened interaction obeys the Casida resolvent identity

    W(omega) - v  =  v (omega - H_RPA)^{-1} v            (omega = 0: static W0)

where H_RPA is the NON-TDA symplectic RPA (test-charge) density-response
Hamiltonian with the bare-exchange RING kernel V (the B1 dense k-summed form):

    H_RPA = [[ D + V ,   V   ],
             [  -V   , -D - V ]]

V sits in BOTH blocks — the RPA ring coupling K^A = (1/Nk)<M_t|v|M_t'>
(``build_bse_ring_matvec_full(..., screening=True)``), NOT the excitonic V_B of
Henneke Eq. 2-20.  This is the test-charge screening whose resolvent resums the
RPA bubble chi = chi0 (1 - v chi0)^{-1}; the exciton V_B kernel is a different
response and does NOT reproduce W (it overshoots the q=0 tile by ~1.8x).  The GW
W is full-RPA, not TDA — dropping the -V/Y block (TDA) fails by construction.

Convention (verified bit-for-bit against the folded static RPA and the on-disk
``W0_qmunu - V_qmunu`` q=0 tile to ~2e-9 — see the "W(0) resolvent cross-check"
section of reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md):

  * Probe column nu: g = e_nu in centroid space; the transition generator applies
    v then the pair-density vertex, f = M^dag (v e_nu)
    (``build_realspace_random_transition_generator``) — the RIGHT vertex of v X v.
  * RHS is that same f with a minus on the anti-resonant (Y) block: rhs = [f; -f]
    (density super-vertex [rho; -rho]; the ring coupling makes the excitation and
    de-excitation vertices coincide, so both blocks carry the SAME f).
  * Shifted solve x = (z - H_RPA)^{-1} rhs at z = (omega + i eta)/Ry via GMRES.
  * Readout s = x[0] + x[1] = X + Y; the density-snapshot vertex applies the pair
    density then v: w_c(mu) = v (M s) = [v chi v]_{mu,nu} = column nu of W - v.

The generator/snapshot k-SUM the pair densities, so the reconstructed tile is the
q=0 block.  H_RPA carries no q=0 head (vhead/whead are a separate rank-1 piece);
``--compare-w0`` loads head-LESS bodies on both sides (``inject_head=False``) and
compares body-to-body.
"""
from __future__ import annotations

import math
import numpy as np
import h5py

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .bse_feast import (
    RY_TO_EV_DEFAULT, gmres_solve_sharded_jit, ensure_W_R,
    build_preconditioner_diagonal_sharded, _apply_shifted_matvec,
)
from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from .bse_ring_comm import (
    build_bse_ring_matvec_full,
    build_realspace_random_transition_generator,
    build_density_snapshot_operator,
    make_bse_shardings,
)
import common.timing as timing

jax.config.update("jax_enable_x64", True)


def _create_mesh_xy(px: int, py: int) -> Mesh:
    devices = jax.devices()
    if px * py > len(devices):
        raise ValueError(f"Requested px*py={px*py} devices, only {len(devices)} available")
    return Mesh(np.array(devices[: px * py]).reshape(px, py), axis_names=("x", "y"))


def _build_rpa_resolvent(mesh_xy: Mesh, data: dict):
    """Assemble the RPA-screening resolvent stack for ``data``.

    Returns ``(matvec, diag_h, gen, snapshot, sh)``.  ``matvec`` is the non-TDA
    symplectic RPA density-response Hamiltonian (screening ring kernel, no W);
    ``ensure_W_R`` populates the placeholder 8th matvec argument.
    """
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    ensure_W_R(data, include_W=False)
    matvec = build_bse_ring_matvec_full(
        mesh_xy, nkx, nky, nkz, include_W=False, screening=True)
    diag_h = build_preconditioner_diagonal_sharded(
        data, mesh_xy, include_W=False, use_tda=False)
    gen = build_realspace_random_transition_generator(
        mesh_xy, nkx, nky, nkz, int(data["n_cond_pad"]), int(data["n_val_pad"]))
    snapshot = build_density_snapshot_operator(mesh_xy, nkx, nky, nkz)
    return matvec, diag_h, gen, snapshot, make_bse_shardings(mesh_xy)


def _resolve_wc_columns(cols, z, data, matvec, diag_h, gen, snapshot, sh,
                        *, max_iter, tol):
    """Column-by-column ``w_c = v (z - H_RPA)^{-1} v e_nu`` (= column nu of
    ``W(omega) - v``, head-less q=0 tile, padded centroid space).

    Returns ``(wc[len(cols), n_rmu] complex128, gmres_resid[len(cols)] float)``.
    """
    n_rmu = int(data["V_q0"].shape[0])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    wc_cols, resids = [], []
    for nu0 in cols:
        g = jnp.zeros((n_rmu,), dtype=jnp.float64).at[int(nu0)].set(1.0)
        r = jax.device_put(jnp.broadcast_to(g[None, :, None], (1, n_rmu, nk)), sh.S)
        f = jax.lax.with_sharding_constraint(
            gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]), sh.X)
        rhs = jax.lax.with_sharding_constraint(
            jnp.stack([f, -f], axis=0).astype(jnp.complex128), sh.X_full)
        x, _ = gmres_solve_sharded_jit(
            matvec, diag_h, z, rhs, data, max_iter=max_iter, tol=tol)
        # Independent relative residual of the shifted system (so quadrature
        # noise vs solver tolerance are distinguishable in the report).
        r_true = rhs - _apply_shifted_matvec(matvec, x, z, data)
        resid = float(jnp.linalg.norm(r_true) / jnp.linalg.norm(rhs))
        s = jax.lax.with_sharding_constraint(x[0] + x[1], sh.X)
        w_c = snapshot(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
        wc_cols.append(np.asarray(jax.device_get(w_c[0])))
        resids.append(resid)
    return np.stack(wc_cols, axis=0), np.asarray(resids)


def _select_compare_cols(T, nlog, n_cols, seed):
    """A mix of the largest-||W0-V|| columns and random columns (logical range)."""
    col_norm = np.linalg.norm(T[:nlog, :nlog], axis=0)
    order = np.argsort(-col_norm)
    n_large = (n_cols + 1) // 2
    large = order[:n_large]
    rng = np.random.default_rng(seed)
    remaining = np.setdiff1d(np.arange(nlog), large)
    n_rand = min(n_cols - n_large, remaining.size)
    rand = (rng.choice(remaining, size=n_rand, replace=False)
            if n_rand > 0 else np.empty(0, dtype=int))
    return np.concatenate([large, rand]).astype(int), col_norm


def _parse_cols(col_str, n_mu, n_cols, seed):
    if col_str:
        cols = [int(x) for x in col_str.split(",") if x.strip() != ""]
        return np.array([c for c in cols if 0 <= c < n_mu], dtype=int)
    if n_cols is not None:
        rng = np.random.default_rng(seed)
        if n_cols >= n_mu:
            return np.arange(n_mu, dtype=int)
        return rng.choice(n_mu, size=n_cols, replace=False)
    return np.arange(n_mu, dtype=int)


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Exact W_c(omega) via the non-TDA RPA density resolvent")
    parser.add_argument("-i", "--input", required=True, help="COHSEX input file")
    parser.add_argument("--compare-w0", action="store_true",
                        help="Cross-check v(0-H_RPA)^-1 v against the restart's "
                             "(W0_qmunu - V_qmunu) q=0 tile.")
    parser.add_argument("--n-val", type=int, default=None,
                        help="Valence bands (default: FULL chi0 window = n_occ).")
    parser.add_argument("--n-cond", type=int, default=None,
                        help="Conduction bands (default: FULL chi0 window).")
    parser.add_argument("--px", type=int, default=1)
    parser.add_argument("--py", type=int, default=1)
    parser.add_argument("--omega-ev", type=float, default=0.0,
                        help="Real frequency omega in eV (default: 0, static W0).")
    parser.add_argument("--eta-ev", type=float, default=0.0,
                        help="Imaginary broadening eta in eV (default: 0).")
    parser.add_argument("--cols", type=str, default=None,
                        help="Comma-separated mu indices to compute (0-based).")
    parser.add_argument("--n-cols", type=int, default=6,
                        help="Number of probe columns (compare-w0: largest-|W0-V| "
                             "+ random mix; else random).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gmres-max-iter", type=int, default=200)
    parser.add_argument("--gmres-tol", type=float, default=1e-10)
    parser.add_argument("--ry-to-ev", type=float, default=RY_TO_EV_DEFAULT)
    parser.add_argument("--out", type=str, default="Wc_exact.h5")
    args = parser.parse_args(argv)

    timing.reset()
    mesh_xy = _create_mesh_xy(args.px, args.py)
    restart_file = _find_restart_file(args.input)

    # BAND-WINDOW PARITY: the Casida pair basis must span the SAME band window the
    # GW chi0 (compute_screening) consumed = ALL occupied x ALL conduction bands
    # stored in the restart (n_val=n_occ, n_cond=nb-n_occ).  Default None -> pass a
    # large count so the loader clamps to the full window.  A smaller hand window
    # would drop transitions chi0 included and fail by construction.
    n_val = args.n_val if args.n_val is not None else 10**9
    n_cond = args.n_cond if args.n_cond is not None else 10**9

    with timing.section("w_exact.load"):
        data = load_bse_data_from_restart_sharded(
            restart_file, n_val=n_val, n_cond=n_cond, mesh_xy=mesh_xy,
            input_file=args.input, inject_head=False)

    n_rmu = int(data["V_q0"].shape[0])
    nlog = int(data["n_rmu"])
    z = (args.omega_ev + 1j * args.eta_ev) / args.ry_to_ev
    print(f"chi0 window: n_val={data['n_val']} n_cond={data['n_cond']} "
          f"(full occ x cond; matches GW compute_screening), "
          f"nk={int(data['nkx']*data['nky']*data['nkz'])}, N_mu={nlog} (padded {n_rmu})")
    print(f"omega={args.omega_ev} eV  eta={args.eta_ev} eV  z={z:.6e} Ry  "
          f"gmres(max_iter={args.gmres_max_iter}, tol={args.gmres_tol:g}); head-less bodies")

    matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh_xy, data)

    if args.compare_w0:
        # Head-less target: (W0_qmunu - V_qmunu) q=0 tile from the loaded bodies.
        W0 = np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
        V0 = np.asarray(jax.device_get(data["V_q0"]))
        T = W0 - V0
        if args.cols:
            cols = _parse_cols(args.cols, nlog, None, args.seed)
            col_norm = np.linalg.norm(T[:nlog, :nlog], axis=0)
        else:
            cols, col_norm = _select_compare_cols(T, nlog, args.n_cols, args.seed)
        print(f"\nW(0) resolvent cross-check: {len(cols)} columns "
              f"(largest-|W0-V| + random)\n")

        with timing.section("w_exact.resolve"):
            wc, resids = _resolve_wc_columns(
                cols, z, data, matvec, diag_h, gen, snapshot, sh,
                max_iter=args.gmres_max_iter, tol=args.gmres_tol)

        hdr = f"{'nu':>5} {'||(W0-V)_col||':>15} {'rel_err':>11} {'max|Delta|':>12} {'gmres_resid':>12}"
        print(hdr)
        print("-" * len(hdr))
        rel_all = []
        for i, nu0 in enumerate(cols):
            tcol = T[:nlog, int(nu0)]
            dcol = wc[i, :nlog] - tcol
            rel = float(np.linalg.norm(dcol) / np.linalg.norm(tcol))
            mx = float(np.max(np.abs(dcol)))
            rel_all.append(rel)
            print(f"{int(nu0):5d} {col_norm[int(nu0)]:15.4e} {rel:11.3e} "
                  f"{mx:12.3e} {resids[i]:12.3e}")
        rel_all = np.asarray(rel_all)
        print("-" * len(hdr))
        print(f"max rel_err = {rel_all.max():.3e}   median = {np.median(rel_all):.3e}   "
              f"max gmres_resid = {resids.max():.3e}")
        print("\nInterpretation: W0_qmunu on disk is the RPA static screened "
              "Coulomb W(0) from chi0 = chi0(iw) minimax-quadratured to w=0; the "
              "resolvent uses the EXACT 1/(e_c-e_v) static denominator, so rel_err "
              "is the GW minimax-integration noise (solver residual is orders "
              "smaller). Closure at this floor confirms W0 = v(0-H_RPA)^-1 v + v.")
    else:
        cols = _parse_cols(args.cols, nlog, args.n_cols, args.seed)
        print(f"\nComputing {len(cols)} W_c(omega) column(s) of N_mu={nlog}")
        with timing.section("w_exact.resolve"):
            wc, resids = _resolve_wc_columns(
                cols, z, data, matvec, diag_h, gen, snapshot, sh,
                max_iter=args.gmres_max_iter, tol=args.gmres_tol)
        with h5py.File(args.out, "w") as h5:
            h5.attrs["omega_ev"] = float(args.omega_ev)
            h5.attrs["eta_ev"] = float(args.eta_ev)
            h5.attrs["ry_to_ev"] = float(args.ry_to_ev)
            h5.attrs["gmres_max_iter"] = int(args.gmres_max_iter)
            h5.attrs["gmres_tol"] = float(args.gmres_tol)
            h5.attrs["kernel"] = "rpa_screening_nonTDA"
            h5.create_dataset("columns", data=cols.astype(np.int32))
            h5.create_dataset("gmres_resid", data=resids)
            h5.create_dataset("Wc", data=wc)
        print(f"Wrote {len(cols)} Wc columns (max gmres_resid={resids.max():.2e}) "
              f"to {args.out}")

    timing.report(print_fn=print, title="--- Timing ---")


if __name__ == "__main__":
    main()
