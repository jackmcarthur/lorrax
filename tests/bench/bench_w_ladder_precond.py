#!/usr/bin/env python3
"""Preconditioning bake-off for the ladder-W resolvent (``bse.w_ladder_precond``).

Three routes at the SAME accuracy, on the SAME operator, measured against the
tight-tolerance unpreconditioned oracle:

  baseline  ``w_ladder``'s shifted block-GMRES, diagonal preconditioner
  lift      route A — Hartree ring lifted out by Woodbury, dense N_mu Dyson close
  fgmres    route B — full operator, EXACT ``(z - H_RPA)^{-1}`` preconditioner

Argv-driven (tests/bench convention; pytest does not collect this directory):

    # one-time fixture preparation (runs the GW driver once, ~minutes):
    python3 tests/bench/bench_w_ladder_precond.py --prepare /path/to/rundir

    # correctness gate (exit code is the verdict):
    python3 tests/bench/bench_w_ladder_precond.py --run-dir RUNDIR --check

    # benchmark, workload (i): 5 q x 1 z, FULL basis, probe_chunk=64:
    python3 tests/bench/bench_w_ladder_precond.py --run-dir RUNDIR --bench \
        --nq 5 --nz 1 --chunks all

    # benchmark, workload (ii): 5 q x 8 MPA-shaped z, ONE chunk (x7 to full):
    python3 tests/bench/bench_w_ladder_precond.py --run-dir RUNDIR --bench \
        --nq 5 --nz 8 --chunks 1

``--chunks 1`` times ONE 64-column probe chunk per (q, z); the chunks of a
sweep are identical work on one compiled engine, so the full-basis cost is
``ceil(n_rmu/64)`` times it exactly.  It cannot be used with ``lift``, whose
Dyson close needs every column of ``T`` before it can produce a W tile at all --
that is a property of the method, not of the harness, and the driver says so
rather than quietly timing something else.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)          # tests/harness.py for --prepare

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np


# --- MPA-shaped z bank (Ry): static, imaginary axis, damped complex ----------
Z_BANK = [0.0 + 0.0j, 0.20j, 0.35j, 0.50j,
          0.15 + 0.03j, 0.25 + 0.05j, 0.35 + 0.05j, 0.45 + 0.08j]
Z_CHECK = [0.0 + 0.0j, 0.35j, 0.25 + 0.05j]


def _log(s):
    import jax
    if jax.process_index() == 0:
        print(s, flush=True)


def _payload(run_dir: str, input_name: str):
    from bse import bse_io
    from bse.bse_ring_comm import create_mesh_xy
    input_path = os.path.join(run_dir, input_name)
    restart = bse_io._find_restart_file(input_path)
    mesh = create_mesh_xy(1, 1)
    # FULL chi0 band window on both legs — band-window parity with the W_R
    # kernel, the same call w_ladder.compute_wc_qwedge makes.
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=input_path, inject_head=False, load_v_full=True)
    return data, mesh, input_path, restart


def _rel(got, ref, nlog, n_real):
    """Max per-column relative error over the logical mu rows of a tile."""
    g = np.asarray(got)[:nlog, :n_real]
    r = np.asarray(ref)[:nlog, :n_real]
    den = np.linalg.norm(r, axis=0)
    den = np.where(den == 0.0, 1.0, den)
    return float(np.max(np.linalg.norm(g - r, axis=0) / den))


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--prepare", metavar="DIR", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--input", default="gnppm_test.in")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--matvec-scaling", action="store_true",
                    help="cost and memory of ONE ladder matvec vs block width: "
                         "what a genuinely blocked (matrix-RHS) GMRES could buy, "
                         "and where the W_R-sized rung buffer stops it")
    ap.add_argument("--widths", default="1,2,4,8,16,32,64")
    ap.add_argument("--facade", action="store_true",
                    help="smoke the drop-in wedge facade compute_wc_qwedge_lifted "
                         "against w_ladder.compute_wc_qwedge on one q, one z")
    ap.add_argument("--methods", default="baseline,lift,fgmres")
    ap.add_argument("--nq", type=int, default=5)
    ap.add_argument("--nz", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--chunks", default="all", choices=["all", "1"])
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="production solver tolerance (baseline used 1e-9)")
    ap.add_argument("--oracle-tol", type=float, default=1e-12)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--check-nq", type=int, default=2)
    ap.add_argument("--check-qs", default=None,
                    help="comma list of q_irr INDICES for --check (default 0..check_nq-1)")
    ap.add_argument("--check-zs", default=None,
                    help="comma list of complex z (Ry) for --check; applied to "
                         "EVERY selected q (default: 3 z at the first q, 1 at the rest)")
    ap.add_argument("--skip-pre", action="store_true",
                    help="--check: skip the ring-dyad / seam / RPA-limit cells "
                         "and go straight to the oracle comparison")
    ap.add_argument("--explain-cache", action="store_true")
    args = ap.parse_args()

    if args.prepare:
        import harness
        run_dir = harness.copy_fixture(harness.REG / "gnppm_debug", args.prepare)
        res = harness.run_gw_jax(run_dir, args.input)
        print(f"[prepare] rc={res.returncode} run_dir={run_dir}", flush=True)
        sys.stdout.write(res.stdout[-2000:])
        return 0 if res.returncode == 0 else 2

    if not args.run_dir:
        ap.error("--run-dir is required for --check/--bench")

    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    if args.explain_cache:
        jax.config.update("jax_explain_cache_misses", True)

    from bse.bse_feast import build_preconditioner_diagonal_sharded
    from bse.bse_w_exact import (apply_screening_resolvent_block,
                                 build_finite_q_data, _symmetry_tables)
    from bse.w_ladder import _accumulate_columns
    from bse import w_ladder_precond as wlp
    from common.collectives import gather_to_host

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    data, mesh, input_path, restart = _payload(args.run_dir, args.input)
    sym = _symmetry_tables(input_path)
    q_list = np.asarray(sym.q_irr_kgrid_int, dtype=int)
    nlog = int(data["n_rmu"])
    n_pad = int(data["V_q0"].shape[0])
    blocks_all = wlp._probe_blocks(nlog, n_pad, args.chunk)
    _log(f"[setup] n_rmu={nlog} pad={n_pad} q_irr={len(q_list)} "
         f"nk={int(data['nkx']*data['nky']*data['nkz'])} "
         f"n_val={data['n_val']} n_cond={data['n_cond']} "
         f"chunks={len(blocks_all)} x {args.chunk} methods={methods}")

    # ONE engine for every q and z (F1: W_R depends on k-k', not on q).
    stack = wlp.build_precond_stack(mesh, data, include_w=True,
                                    vertex_flipped=True)

    def _dq(q):
        dq = build_finite_q_data(data, (int(q[0]), int(q[1]), int(q[2])), mesh)
        diag_full = build_preconditioner_diagonal_sharded(
            dq, mesh, include_W=True, use_tda=False)          # baseline / oracle
        diag_lift = wlp.lifted_precond_diagonal(dq, mesh, stack, include_w=True)
        diag_rpa0 = wlp.lifted_precond_diagonal(dq, mesh, stack, include_w=False)
        return dq, diag_full, diag_lift, diag_rpa0

    # --------------------------------------------------------------- baseline
    def run_baseline(dq, diag_full, z, blocks, tol, max_iter):
        acc, its, rr = None, [], []
        for c0, n_real, G in blocks:
            tile, resids, iters = apply_screening_resolvent_block(
                G, complex(z), dq, stack.matvec, diag_full, stack.gen,
                stack.snapshot, stack.sh, max_iter=max_iter, tol=tol,
                return_iters=True)
            acc = _accumulate_columns(acc, tile, c0, n_real, n_pad, mesh)
            its.append(np.asarray(gather_to_host(iters))[:n_real])
            rr.append(np.asarray(gather_to_host(resids))[:n_real])
        return acc, np.concatenate(its), np.concatenate(rr)

    # ------------------------------------------------------------------- lift
    def run_lift(dq, diag_lift, z, blocks, tol, max_iter):
        acc, its, rr = None, [], []
        for c0, n_real, G in blocks:
            tile, resids, iters = wlp.apply_lifted_resolvent_block(
                G, complex(z), dq, stack, diag_lift,
                max_iter=max_iter, tol=tol)
            acc = _accumulate_columns(acc, tile, c0, n_real, n_pad, mesh)
            its.append(np.asarray(gather_to_host(iters))[:n_real])
            rr.append(np.asarray(gather_to_host(resids))[:n_real])
        return (wlp.dyson_close_tile(acc, dq["V_q0"]),
                np.concatenate(its), np.concatenate(rr))

    # ----------------------------------------------------------------- fgmres
    def run_fgmres(dq, diag_rpa0, z, blocks, tol, max_iter, setup_out=None):
        t0 = time.perf_counter()
        chi0v = wlp.build_chi0_v_tile(complex(z), dq, stack, blocks_all,
                                      diag_rpa0)
        B_dense = wlp.build_rpa_dyson_factor(chi0v, dq["V_q0"])
        pargs = wlp.build_rpa_dyson_preconditioner(dq, stack, diag_rpa0, B_dense)
        jax.block_until_ready(B_dense)
        t_setup = time.perf_counter() - t0
        if setup_out is not None:
            setup_out.append(t_setup)
        return _run_seam(dq, z, blocks, tol, max_iter,
                         wlp.rpa_dyson_precond, pargs)

    def _run_seam(dq, z, blocks, tol, max_iter, precond, pargs):
        """The FGMRES seam driving the FULL ladder operator, any (M^-1, args)."""
        acc, its, rr = None, [], []
        for c0, n_real, G in blocks:
            tile, resids, iters = wlp.apply_fgmres_resolvent_block(
                G, complex(z), dq, stack, precond, pargs,
                max_iter=max_iter, tol=tol)
            acc = _accumulate_columns(acc, tile, c0, n_real, n_pad, mesh)
            its.append(np.asarray(gather_to_host(iters))[:n_real])
            rr.append(np.asarray(gather_to_host(resids))[:n_real])
        return acc, np.concatenate(its), np.concatenate(rr)

    # ``seamdiag`` is the SEAM carrying the in-tree diagonal preconditioner: it
    # must reproduce ``baseline`` (same x0, same DGKS, same lstsq, same exit).
    # It is what makes "this is a seam under the shared solver" a measurement
    # instead of a claim, and it prices the seam's own overhead at zero.
    def run_seamdiag(dq, diag_full, z, blocks, tol, max_iter):
        return _run_seam(dq, z, blocks, tol, max_iter,
                         wlp.diagonal_precond, (diag_full,))

    # ---------------------------------------------------- 2x2 pair block (idea 2)
    def run_pair2x2(dq, z, blocks, tol, max_iter):
        ab = wlp.build_pair_2x2_diagonals(dq, mesh, stack, include_w=True,
                                          ring=True)
        return _run_seam(dq, z, blocks, tol, max_iter, wlp.pair2x2_precond, ab)

    def run_lift2x2(dq, z, blocks, tol, max_iter):
        """Route A's operator with the 2x2 pair block instead of the diagonal."""
        ab = wlp.build_pair_2x2_diagonals(dq, mesh, stack, include_w=True,
                                          ring=False)
        acc, its, rr = None, [], []
        d0 = wlp.ringless_payload(dq, stack)
        for c0, n_real, G in blocks:
            tile, resids, iters = wlp.apply_fgmres_resolvent_block(
                G, complex(z), d0, stack, wlp.pair2x2_precond, ab,
                max_iter=max_iter, tol=tol, snapshot_v=stack.eye,
                seed_v=dq["V_q0"])
            acc = _accumulate_columns(acc, tile, c0, n_real, n_pad, mesh)
            its.append(np.asarray(gather_to_host(iters))[:n_real])
            rr.append(np.asarray(gather_to_host(resids))[:n_real])
        return (wlp.dyson_close_tile(acc, dq["V_q0"]),
                np.concatenate(its), np.concatenate(rr))

    RUNNERS = {"baseline": lambda dq, d, z, b, tol, mi: run_baseline(dq, d[1], z, b, tol, mi),
               "seamdiag": lambda dq, d, z, b, tol, mi: run_seamdiag(dq, d[1], z, b, tol, mi),
               "lift": lambda dq, d, z, b, tol, mi: run_lift(dq, d[2], z, b, tol, mi),
               "pair2x2": lambda dq, d, z, b, tol, mi: run_pair2x2(dq, z, b, tol, mi),
               "lift2x2": lambda dq, d, z, b, tol, mi: run_lift2x2(dq, z, b, tol, mi),
               "fgmres": lambda dq, d, z, b, tol, mi: run_fgmres(dq, d[3], z, b, tol, mi)}

    rc = 0

    # ======================================================================
    #  --matvec-scaling : DEEP BLOCKS -- what width buys, and what stops it
    # ======================================================================
    if args.matvec_scaling:
        from bse.bse_feast import matvec_operands
        dqm, _, _, _ = _dq(q_list[0])
        nkm = int(dqm["nkx"] * dqm["nky"] * dqm["nkz"])
        nc, nv = int(dqm["n_cond_pad"]), int(dqm["n_val_pad"])
        # The RPA operator is the SAME matvec minus the direct rung, so the
        # difference between the two curves IS the rung -- the term whose
        # (mu, nu, s, s, k) FFT-chain buffer is the thing that scales with
        # block width.  Built from its own payload copy (ensure_W_R(False)
        # writes a placeholder W_R and must not land on the ladder payload).
        rpa_stack = wlp.build_precond_stack(mesh, dict(data), include_w=False,
                                            vertex_flipped=True)
        dq_rpa = build_finite_q_data(dict(data), tuple(int(x) for x in q_list[0]),
                                     mesh)
        from bse.bse_feast import ensure_W_R
        ensure_W_R(dq_rpa, include_W=False, mesh_xy=mesh)
        wr = np.prod([int(x) for x in dqm["W_R"].shape]) * 16 / 2**20
        _log(f"\n=== matvec cost & memory vs BLOCK WIDTH (q={tuple(int(x) for x in q_list[0])}) ===")
        _log(f"[geom] pair basis (c,v,k) = ({nc},{nv},{nkm}) = {nc*nv*nkm} ; "
             f"N_mu={n_pad} ; W_R tile {tuple(int(x) for x in dqm['W_R'].shape)}"
             f" = {wr:.1f} MiB ; the rung's per-trial-vector (mu,nu,s,s,k) "
             f"buffer = {n_pad*n_pad*2*2*nkm*16/2**20:.1f} MiB")
        hdr = (f"{'nb':>5} {'ladder[ms]':>11} {'per col':>9} {'RPA[ms]':>9} "
               f"{'per col':>9} {'rung share':>11} {'peak[MiB]':>10} "
               f"{'d(peak)/nb':>11}")
        _log(hdr); _log("-" * len(hdr))
        rng = np.random.default_rng(0)
        dev = jax.local_devices()[0]

        def _peak():
            try:
                return dev.memory_stats()["peak_bytes_in_use"] / 2**20
            except Exception:
                return float("nan")

        prev_peak, prev_nb, rows = _peak(), 0, []
        for nb in [int(w) for w in args.widths.split(",")]:
            shp = (2, nb, nc, nv, nkm)
            x = jax.lax.with_sharding_constraint(
                jnp.asarray(rng.standard_normal(shp)
                            + 1j * rng.standard_normal(shp),
                            dtype=jnp.complex128), stack.sh.X_full)
            out = {}
            for tag, mv, ops in (("lad", stack.matvec, matvec_operands(dqm)),
                                 ("rpa", rpa_stack.matvec, matvec_operands(dq_rpa))):
                try:
                    jax.block_until_ready(mv(x, *ops))          # warm/compile
                    reps = 3
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        jax.block_until_ready(mv(x, *ops))
                    out[tag] = 1e3 * (time.perf_counter() - t0) / reps
                except Exception as exc:
                    out[tag] = float("nan")
                    _log(f"[matvec] nb={nb} {tag}: {type(exc).__name__}: "
                         f"{str(exc)[:140]}")
            pk = _peak()
            dpk = (pk - prev_peak) / max(nb - prev_nb, 1)
            _log(f"{nb:5d} {out['lad']:11.2f} {out['lad']/nb:9.3f} "
                 f"{out['rpa']:9.2f} {out['rpa']/nb:9.3f} "
                 f"{1 - out['rpa']/out['lad']:11.1%} {pk:10.1f} {dpk:11.2f}")
            rows.append((nb, out["lad"], out["rpa"], pk))
            prev_peak, prev_nb = pk, nb
            del x
        _log("-" * len(hdr))
        good = [r for r in rows if r[1] == r[1]]
        if len(good) >= 2:
            b1, bn = good[0], good[-1]
            _log(f"[matvec] per-column matvec cost {b1[1]/b1[0]:.3f} ms at nb=1 "
                 f"-> {bn[1]/bn[0]:.3f} ms at nb={bn[0]} "
                 f"({(b1[1]/b1[0])/(bn[1]/bn[0]):.2f}x from GEMM width alone)")
        if not (args.check or args.bench or args.facade):
            return rc

    # ======================================================================
    #  --facade : the drop-in wedge wrapper, against w_ladder's own
    # ======================================================================
    if args.facade:
        # The wrapper against the SAME-PROCESS baseline on the same q, not
        # against w_ladder.compute_wc_qwedge (which has no n_q knob and would
        # solve all 5 q to compare 1).  What is under test here is the wrapper's
        # own plumbing -- loader call, symmetry tables, chunk walk, accumulate,
        # Dyson close, WLadderWedge contract -- since its internals are already
        # covered by --check.
        _log(f"\n=== facade: compute_wc_qwedge_lifted (1 q x 1 z, "
             f"probe_chunk={args.chunk}) vs the in-process baseline ===")
        dqf, dfullf, _, _ = _dq(q_list[0])
        t0 = time.perf_counter()
        ref_t, ref_it, _ = run_baseline(dqf, dfullf, 0.0 + 0.0j, blocks_all,
                                        args.tol, args.max_iter)
        b = np.asarray(gather_to_host(ref_t))
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        got_w = wlp.compute_wc_qwedge_lifted(
            restart, [0.0 + 0.0j], mesh, include_w=True, gmres_tol=args.tol,
            gmres_max_iter=args.max_iter, probe_chunk=args.chunk,
            input_file=input_path, n_q=1)
        t_got = time.perf_counter() - t0
        a = np.asarray(gather_to_host(got_w.wc))[0, 0]
        rel = _rel(a, b, nlog, nlog)
        ok_shape = got_w.wc.shape == (1, 1, n_pad, n_pad)
        ok_nrmu = int(got_w.n_rmu) == nlog
        ok_q = tuple(int(x) for x in got_w.q_irr_kgrid_int[0]) == tuple(
            int(x) for x in q_list[0])
        ok_tables = (got_w.irr_idx_q.size > 0 and got_w.sym_idx_q.size > 0
                     and got_w.q_irr_full_idx.size > 0)
        ok_shard = got_w.wc.sharding.is_equivalent_to(
            jax.sharding.NamedSharding(mesh, P(None, None, "x", "y")),
            got_w.wc.ndim)
        _log(f"[facade] rel_vs_baseline={rel:.3e}  shape={tuple(got_w.wc.shape)}"
             f" n_rmu={got_w.n_rmu} sharding_ok={ok_shard} tables_ok={ok_tables}"
             f" q0_ok={ok_q}")
        _log(f"[facade] iters lifted mean="
             f"{got_w.gmres_iters[0,0,:nlog].mean():.1f} max="
             f"{int(got_w.gmres_iters[0,0,:nlog].max())} resid_max="
             f"{got_w.gmres_resid[0,0,:nlog].max():.2e} | baseline mean="
             f"{ref_it.mean():.1f} max={int(ref_it.max())} | wall "
             f"{t_got:.1f} s vs {t_ref:.1f} s")
        if not (rel <= 1e-8 and ok_shape and ok_nrmu and ok_q and ok_tables
                and ok_shard):
            _log(f"[facade] FAIL")
            rc = 1
        if not (args.check or args.bench):
            return rc

    # ======================================================================
    #  --check : correctness gate
    # ======================================================================
    if args.check:
        check_qs = ([int(x) for x in args.check_qs.split(",")] if args.check_qs
                    else list(range(min(args.check_nq, len(q_list)))))
        check_zs = ([complex(x) for x in args.check_zs.split(",")]
                    if args.check_zs else None)
    # --- pre-cells (structural identities; cheap except the RPA-limit solve)
    if args.check and not args.skip_pre:
        _log("\n=== ring-dyad normalisation (R): matvec|_V - matvec|_0 == s v p ===")
        for qi in range(min(args.check_nq, len(q_list))):
            dq, *_ = _dq(q_list[qi])
            d = wlp.check_ring_dyad_identity(dq, stack, seed=qi)
            ok = d <= 1e-11
            _log(f"[R] q={tuple(int(x) for x in q_list[qi])}: rel={d:.3e} "
                 f"{'OK' if ok else 'FAIL'}")
            if not ok:
                rc = 1

        _log("\n=== seam equivalence: fgmres_solve_core + diagonal_precond "
             "== bse_feast._gmres_solve_core (ONE chunk, q=0, z=0) ===")
        # The seam must be a seam: same preconditioner, same answer, same
        # iteration count.  A drift here means the shared core and the flexible
        # one have forked, which is the one thing this design must not do.
        dq0, dfull0, _, _ = _dq(q_list[0])
        b1 = blocks_all[:1]
        ref1, it1, rr1 = run_baseline(dq0, dfull0, 0.0 + 0.0j, b1,
                                      args.tol, args.max_iter)
        got1, it2, rr2 = run_seamdiag(dq0, dfull0, 0.0 + 0.0j, b1,
                                      args.tol, args.max_iter)
        rel1 = _rel(np.asarray(gather_to_host(got1)),
                    np.asarray(gather_to_host(ref1)), nlog, b1[0][1])
        same_it = bool(np.array_equal(it1, it2))
        ok = (rel1 <= 1e-12) and same_it
        _log(f"[seam] rel={rel1:.3e}  iters identical={same_it} "
             f"(core {it1.mean():.1f}/{int(it1.max())}, seam "
             f"{it2.mean():.1f}/{int(it2.max())}) {'OK' if ok else 'FAIL'}")
        if not ok:
            rc = 1

        _log("\n=== RPA limit of the lift: include_w=False vs the on-disk W0-V ===")
        # H_0 = diag(D,-D) there, so the zeroed-ring diagonal IS (z - H_0): the
        # solve is one iteration and (A) reduces to v chi0 (1 - v chi0)^-1 v.
        rpa_stack = wlp.build_precond_stack(mesh, data, include_w=False,
                                            vertex_flipped=True)
        for qi in range(min(args.check_nq, len(q_list))):
            q = tuple(int(x) for x in q_list[qi])
            dq = build_finite_q_data(data, q, mesh)
            diag0 = wlp.lifted_precond_diagonal(dq, mesh, rpa_stack,
                                                include_w=False)
            acc, its = None, []
            for c0, n_real, G in blocks_all:
                tile, resids, iters = wlp.apply_lifted_resolvent_block(
                    G, 0.0 + 0.0j, dq, rpa_stack, diag0,
                    max_iter=args.max_iter, tol=args.tol)
                acc = _accumulate_columns(acc, tile, c0, n_real, n_pad, mesh)
                its.append(np.asarray(gather_to_host(iters))[:n_real])
            W = np.asarray(gather_to_host(
                wlp.dyson_close_tile(acc, dq["V_q0"])))
            T = (np.asarray(gather_to_host(data["W_q"][:, :, q[0], q[1], q[2]]))
                 - np.asarray(gather_to_host(data["V_q_full"][:, :, q[0], q[1], q[2]])))
            rel = _rel(W, T, nlog, nlog)
            its = np.concatenate(its)
            ok = rel <= 1e-7          # GW minimax-quadrature floor is ~2.5e-9
            _log(f"[RPA-limit] q={q}: rel_vs_disk={rel:.3e} "
                 f"iters mean={its.mean():.1f} max={int(its.max())} "
                 f"{'OK' if ok else 'FAIL'}")
            if not ok:
                rc = 1

    # --- the gate proper: both routes against the tight-tol oracle
    if args.check:
        _log("\n=== ladder: routes vs the 1e-12 unpreconditioned oracle ===")
        hdr = (f"{'q':>10} {'z':>16} {'method':>9} {'rel_vs_oracle':>14} "
               f"{'max_resid':>10} {'it_mean':>8} {'it_max':>7} {'wall[s]':>9}")
        _log(hdr)
        _log("-" * len(hdr))
        for j, qi in enumerate(check_qs):
            q = tuple(int(x) for x in q_list[qi])
            dq, dfull, dlift, drpa = _dq(q_list[qi])
            dd = (dq, dfull, dlift, drpa)
            if check_zs is not None:
                zs = check_zs
            else:
                zs = Z_CHECK if j == 0 else Z_CHECK[:1]
            for z in zs:
                t0 = time.perf_counter()
                ref, oit, orr = run_baseline(dq, dfull, z, blocks_all,
                                             args.oracle_tol, args.max_iter)
                ref_h = np.asarray(gather_to_host(ref))
                t_or = time.perf_counter() - t0
                _log(f"{str(q):>10} {str(z):>16} {'oracle':>9} "
                     f"{'-':>14} {orr.max():10.2e} {oit.mean():8.1f} "
                     f"{int(oit.max()):7d} {t_or:9.1f}")
                for m in methods:
                    if m == "baseline":
                        continue
                    t0 = time.perf_counter()
                    got, it, rr = RUNNERS[m](dq, dd, z, blocks_all,
                                             args.tol, args.max_iter)
                    got_h = np.asarray(gather_to_host(got))
                    t_m = time.perf_counter() - t0
                    rel = _rel(got_h, ref_h, nlog, nlog)
                    _log(f"{str(q):>10} {str(z):>16} {m:>9} {rel:14.3e} "
                         f"{rr.max():10.2e} {it.mean():8.1f} "
                         f"{int(it.max()):7d} {t_m:9.1f}")
                    if not (rel <= 1e-8):
                        _log(f"    FAIL: {m} rel {rel:.3e} > 1e-8")
                        rc = 1
        _log(f"\n[check] verdict rc={rc}")
        if not args.bench:
            return rc

    # ======================================================================
    #  --bench : the shared protocol
    # ======================================================================
    if args.bench:
        z_pts = Z_BANK[:int(args.nz)]
        nq = min(int(args.nq), len(q_list))
        blocks = blocks_all if args.chunks == "all" else blocks_all[:1]
        scale = 1.0 if args.chunks == "all" else float(len(blocks_all))
        if args.chunks == "1" and "lift" in methods:
            _log("[bench] REFUSED: --chunks 1 cannot time 'lift' — its Dyson "
                 "close needs every column of T before a W tile exists. Time "
                 "the lift with --chunks all, or drop it from --methods.")
            return 2
        _log(f"\n=== bench: {nq} q x {len(z_pts)} z, probe_chunk={args.chunk}, "
             f"{len(blocks)} of {len(blocks_all)} chunks"
             + (f" (x{scale:.0f} -> full basis)" if scale != 1.0 else "")
             + f", tol={args.tol:g}, max_iter={args.max_iter} ===")
        # METHODS ARE INTERLEAVED AT THE (q, z) POINT, not run in per-method
        # blocks.  The pool is shared and this workload is host-dispatch-bound,
        # so a slow drift in contention across a 40-minute leg would land
        # entirely on whichever method ran last and show up as a speedup.
        # Interleaving makes every method see the same machine minute by minute;
        # the first point of each method still carries its own compile, which is
        # why `solve` (summed per point) is reported next to `sweep` (wall).
        results = {m: dict(solve=0.0, it=[], rr=[], setup=[], first=None)
                   for m in methods}
        t_sweep0 = time.perf_counter()
        for qi in range(nq):
            dq, dfull, dlift, drpa = _dq(q_list[qi])
            dd = (dq, dfull, dlift, drpa)
            for z in z_pts:
                for m in methods:
                    t0 = time.perf_counter()
                    if m == "fgmres":
                        out, it, rr = run_fgmres(dq, drpa, z, blocks, args.tol,
                                                 args.max_iter,
                                                 setup_out=results[m]["setup"])
                    else:
                        out, it, rr = RUNNERS[m](dq, dd, z, blocks, args.tol,
                                                 args.max_iter)
                    jax.block_until_ready(out)
                    dt = time.perf_counter() - t0
                    results[m]["solve"] += dt
                    if results[m]["first"] is None:
                        results[m]["first"] = dt      # carries the compile
                    results[m]["it"].append(it)
                    results[m]["rr"].append(rr)
                    _log(f"[pt] q={tuple(int(x) for x in q_list[qi])} z={z} "
                         f"{m:>9}: {dt:7.1f} s  it {it.mean():5.1f}/"
                         f"{int(it.max()):3d}  resid {rr.max():.2e}")
        t_sweep = time.perf_counter() - t_sweep0
        for m in methods:
            r = results[m]
            it_all = np.concatenate(r["it"])
            rr_all = np.concatenate(r["rr"])
            r.update(it_mean=float(it_all.mean()), it_max=int(it_all.max()),
                     resid=float(rr_all.max()),
                     warm=r["solve"] - r["first"],
                     setup=float(np.sum(r["setup"])) if r["setup"] else 0.0)
            _log(f"[bench] {m:>9}: solve {r['solve']:8.1f} s (first point "
                 f"{r['first']:6.1f} s carries the compile; warm "
                 f"{r['warm']:8.1f} s)  iters mean {r['it_mean']:5.1f} "
                 f"max {r['it_max']:4d}  max_resid {r['resid']:.2e}"
                 + (f"  precond setup {r['setup']:.1f} s" if r["setup"] else ""))
        _log(f"[bench] whole sweep wall {t_sweep:.1f} s")

        _log("\n" + "=" * 92)
        _log(f"{'method':>9} {'solve[s]':>10} {'warm[s]':>10} "
             f"{'x->full':>10} {'it_mean':>8} {'it_max':>7} {'ms/col-it':>10} "
             f"{'speedup':>8}")
        _log("-" * 92)
        base = results.get("baseline", {}).get("solve")
        n_cols_done = sum(b[1] for b in blocks) * nq * len(z_pts)
        for m in methods:
            r = results[m]
            # ms per column-iteration: the per-iteration price of the method,
            # which is how the preconditioner's own cost shows up.  A method
            # that cuts iterations but raises this has spent the win.
            ms = 1e3 * r["solve"] / max(n_cols_done * r["it_mean"], 1e-9)
            _log(f"{m:>9} {r['solve']:10.1f} {r['warm']:10.1f} "
                 f"{r['solve']*scale:10.1f} {r['it_mean']:8.1f} "
                 f"{r['it_max']:7d} {ms:10.3f} "
                 f"{(base / r['solve']) if base else float('nan'):8.2f}x")
        _log("=" * 92)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
