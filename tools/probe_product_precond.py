#!/usr/bin/env python3
"""A/B probe: product-form preconditioner vs the production diagonal.

One probe column per (arm, q), both driven through the SAME flexible engine
(``w_ladder_precond.fgmres_solve_core``) so the only difference between arms
is the ``precond`` callable.  Reports outer iterations, TOTAL production-
matvec equivalents (the honest cross-arm cost: each inner P/Q application is
one full matvec), true residual ``||b-(z-H)x||/||b||``, and the P/Q Lanczos
definiteness bounds the derivation requires.

Single process, one GPU, prototype scope (stated in the report).
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

from bse.bse_feast import (build_preconditioner_diagonal_sharded,
                           ladder_matvec_operands, matvec_operands)
from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_ring_comm import create_mesh_xy
from bse.bse_w_exact import (_apply_shifted_matvec, build_finite_q_data,
                             build_probe_rhs, enforce_trs_pair_gauge,
                             _symmetry_tables)
from bse.w_ladder import build_ladder_resolvent
from bse.w_ladder_precond import fgmres_solve_core
from bse.w_ladder_product_precond import (lanczos_extremal_bound,
                                          make_half_sum_appliers,
                                          make_product_preconditioner,
                                          make_tda_schur_preconditioner)
from common.collectives import gather_to_host


def host(x):
    return np.asarray(gather_to_host(x))


def run_case(run_dir, deck_name, q_index, col, cap, tol, n_cg_p, n_cg_q,
             out_rows):
    mesh = create_mesh_xy(1, 1)
    deck = os.path.join(run_dir, deck_name)
    restart = _find_restart_file(deck)
    raw = load_bse_data_from_restart_sharded(
        restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=deck, inject_head=False, load_v_full=True)
    data = enforce_trs_pair_gauge(raw, mesh)
    q = tuple(int(v) for v in np.asarray(
        _symmetry_tables(deck).q_irr_kgrid_int, dtype=int)[int(q_index)])
    dq = build_finite_q_data(data, q, mesh)

    matvec, diag_h, gen, snapshot, sh = build_ladder_resolvent(
        mesh, dq, include_w=True)
    operands = ladder_matvec_operands(dq)
    n_rmu = int(dq["V_q0"].shape[0])

    G = np.zeros((1, n_rmu), dtype=np.float64)
    G[0, int(col)] = 1.0
    rhs = build_probe_rhs(G, dq, gen, sh)
    b = rhs.astype(jnp.complex128)
    z0 = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)

    dref = jnp.maximum(jnp.abs(diag_h[0].real), 1e-8)

    # -- definiteness probe (derivation gate) -------------------------------
    apply_P, apply_Q = make_half_sum_appliers(matvec, operands, sh)
    probe_vec = b[0]
    pmin, pmax = lanczos_extremal_bound(apply_P, probe_vec, 24)
    qmin, qmax = lanczos_extremal_bound(apply_Q, probe_vec, 24)
    key = jax.random.PRNGKey(7)
    xa = jax.random.normal(key, probe_vec.shape, dtype=jnp.float64).astype(probe_vec.dtype)
    xb = jax.random.normal(jax.random.PRNGKey(11), probe_vec.shape, dtype=jnp.float64).astype(probe_vec.dtype)
    def asym(op):
        lhs = jnp.vdot(xa, op(xb)); rhs_ = jnp.vdot(op(xa), xb)
        return float(jnp.abs(lhs - rhs_) / jnp.maximum(jnp.abs(lhs), 1e-30))
    p_asym, q_asym = asym(apply_P), asym(apply_Q)
    print(f"[{run_dir}:q{q_index}] P bounds [{pmin:.4e},{pmax:.4e}] asym {p_asym:.3e}  "
          f"Q bounds [{qmin:.4e},{qmax:.4e}] asym {q_asym:.3e}", flush=True)

    def precond_diag(vec, z, args):
        d = args
        return vec / (z - d)

    precond_prod = make_product_preconditioner(
        matvec, sh, n_cg_p=n_cg_p, n_cg_q=n_cg_q, inner="gmres")

    precond_schur = make_tda_schur_preconditioner(
        matvec, sh, n_in_a=12, n_in_at=12, inner="gmres")
    arms = {
        "diag": (precond_diag, diag_h, 1),
        "product": (precond_prod, (operands, dref), 1 + n_cg_p + n_cg_q),
        "tda_schur": (precond_schur, (operands, dref), 1 + 12 + 12 + 1),
    }
    # Candidate (1): inner fixed-trip GMRES on the RPA operator as M^{-1}.
    from bse.w_ladder_product_precond import _fixed_gmres

    def make_rpa_inner_precond(mv_rpa_, m):
        def precond(vec, z, precond_args):
            op_rpa_, diag_rpa_ = precond_args

            def op_full(u2):
                return z * u2 - mv_rpa_(u2, *op_rpa_)

            d2 = jnp.maximum(jnp.abs(diag_rpa_.real), 1e-8)
            # full 2-row solve in one Krylov space (vec has both rows)
            return _fixed_gmres(lambda u: op_full(u), d2, vec, m)
        return precond

    # Diagnostic (0): RPA depth at the SAME q with the SAME engine/diag arm.
    mv_rpa, diag_rpa, gen_r, snap_r, sh_r = build_ladder_resolvent(
        mesh, dq, include_w=False)
    op_rpa = matvec_operands(dq)
    fn_rpa = jax.jit(lambda bb, pa: fgmres_solve_core(
        mv_rpa, bb, z0, op_rpa, precond_diag, pa, int(cap), float(tol)))
    xr, kr = fn_rpa(b, diag_rpa)
    xr.block_until_ready()
    rr = b - _apply_shifted_matvec(mv_rpa, xr, z0, op_rpa).astype(b.dtype)
    rpa_rel = float(jnp.linalg.norm(rr) / jnp.linalg.norm(b))
    print(f"[{run_dir}:q{q_index}:RPA-diag] outer={int(kr)} "
          f"true_rel={rpa_rel:.3e}", flush=True)
    out_rows.append(dict(run=run_dir, q_index=int(q_index), q=list(q),
                         col=int(col), arm="rpa_diag_diagnostic",
                         outer_iters=int(kr), true_rel=rpa_rel))
    # rpa_inner arm: ladder outer, RPA-resolvent preconditioner (m=10).
    # Cost note: one inner RPA matvec is ~4% of a ladder matvec (rung-free),
    # so total ladder-equivalents ~= 1 + k + (k+1)*10*0.04.
    prec_rpa = make_rpa_inner_precond(mv_rpa, 10)
    # NB: _fixed_gmres treats the FULL (2,...) stacked vec as one vector; the
    # RPA operator acts on the same stacked shape, so this is well-formed.
    fn_ri = jax.jit(lambda bb, pa: fgmres_solve_core(
        matvec, bb, z0, operands, prec_rpa, pa, int(cap), float(tol)))
    t0 = time.time(); xi, ki = fn_ri(b, (op_rpa, diag_rpa)); xi.block_until_ready()
    w_cold = time.time() - t0
    t1 = time.time(); xi, ki = fn_ri(b, (op_rpa, diag_rpa)); xi.block_until_ready()
    w_warm = time.time() - t1
    ri = b - _apply_shifted_matvec(matvec, xi, z0, operands).astype(b.dtype)
    ri_rel = float(jnp.linalg.norm(ri) / jnp.linalg.norm(b))
    ki = int(ki)
    ladder_equiv = 1 + ki + (ki + 1) * 10 * 0.04
    print(f"[{run_dir}:q{q_index}:rpa_inner] outer={ki} "
          f"ladder_equiv={ladder_equiv:.1f} true_rel={ri_rel:.3e} "
          f"warm={w_warm:.2f}s", flush=True)
    out_rows.append(dict(run=run_dir, q_index=int(q_index), q=list(q),
                         col=int(col), arm="rpa_inner_m10",
                         outer_iters=ki, ladder_equiv=ladder_equiv,
                         true_rel=ri_rel, wall_cold=w_cold,
                         wall_warm=w_warm))
    for name, (precond, pargs, mv_per_iter) in arms.items():
        fn = jax.jit(lambda bb, pa: fgmres_solve_core(
            matvec, bb, z0, operands, precond, pa,
            int(cap), float(tol)))
        t0 = time.time()
        x, k = fn(b, pargs)
        x.block_until_ready()
        wall = time.time() - t0
        t1 = time.time()
        x, k = fn(b, pargs)
        x.block_until_ready()
        wall_warm = time.time() - t1
        r = b - _apply_shifted_matvec(matvec, x, z0, operands).astype(b.dtype)
        true_rel = float(jnp.linalg.norm(r) / jnp.linalg.norm(b))
        k = int(k)
        # H-matvec-equivalents: r0 apply + k outer applies + (k+1) precond
        # applications (x0 plus one per iteration), each costing mv_per_iter-1
        # inner matvecs for the product arm and 0 for the diagonal.
        inner = (mv_per_iter - 1)
        total_mv = 1 + k + (k + 1) * inner
        row = dict(run=run_dir, q_index=int(q_index), q=list(q), col=int(col),
                   arm=name, outer_iters=k, total_matvec_equiv=int(total_mv),
                   true_rel=true_rel, wall_cold=wall, wall_warm=wall_warm,
                   cap=int(cap), tol=float(tol), n_cg_p=int(n_cg_p),
                   n_cg_q=int(n_cg_q),
                   P_bounds=[pmin, pmax], Q_bounds=[qmin, qmax],
                   P_asym=p_asym, Q_asym=q_asym)
        out_rows.append(row)
        print(f"[{run_dir}:q{q_index}:{name}] outer={k} total_mv={total_mv} "
              f"true_rel={true_rel:.3e} warm={wall_warm:.2f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", required=True,
                    help="semicolon list: run_dir,deck,q_index,col")
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--n-cg-p", type=int, default=6)
    ap.add_argument("--n-cg-q", type=int, default=24)
    args = ap.parse_args()
    rows = []
    for case in args.cases.split(";"):
        run_dir, deck, qi, col = case.split(",")
        run_case(run_dir, deck, int(qi), int(col), args.cap, args.tol,
                 args.n_cg_p, args.n_cg_q, rows)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
