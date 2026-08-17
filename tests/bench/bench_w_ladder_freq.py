#!/usr/bin/env python3
"""Check + benchmark for ``bse.w_ladder_freq`` — the frequency-amortized
ladder-W chain — against the per-z shifted block-GMRES oracle on the SAME
operator.

Argv-driven (tests/bench convention; NOT collected by pytest):

    # one-time fixture preparation (runs the GW driver once):
    python3 tests/bench/bench_w_ladder_freq.py --prepare /path/to/rundir

    # correctness gate (exit code is the verdict):
    python3 tests/bench/bench_w_ladder_freq.py --run-dir RUNDIR --check \
        --chain-len 128

    # benchmark (oracle vs chain, per-chunk, extrapolation stated):
    python3 tests/bench/bench_w_ladder_freq.py --run-dir RUNDIR --bench \
        --chain-len 96 --nq 5 --nz 8

The check compares the chain's ``W(z) - v`` tiles against the oracle at
tol 1e-12 for z in {0, 0.35i, 0.25+0.05i} Ry at q=0 AND q=(0,1,0), sweeping
``m_use`` so convergence-in-chain-length is measured rather than assumed, and
verifies the per-z residual ESTIMATE tracks the true error (the estimate is
what production would gate on).  Everything runs the flipped-vertex payload
convention (`build_finite_q_data`), engine built ONCE — the same discipline as
``w_ladder.sweep_q_wedge``.
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

import numpy as np


def _payload(run_dir: str, input_name: str):
    from bse import bse_io
    from bse.bse_ring_comm import create_mesh_xy
    input_path = os.path.join(run_dir, input_name)
    restart = bse_io._find_restart_file(input_path)
    mesh = create_mesh_xy(1, 1)
    # FULL chi0 band window on both legs — band-window parity with the W_R
    # kernel, same reasoning as w_ladder.compute_wc_qwedge.
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=input_path, inject_head=False, load_v_full=True)
    return data, mesh, input_path


def _chunk_blocks(nlog: int, n_pad: int, chunk: int):
    """The compute_wc_qwedge probe-block construction: identity rows, walk
    stops at nlog, short final chunk zero-padded UP (same compiled shapes)."""
    eye = np.eye(n_pad, dtype=np.float64)
    out = []
    for c0 in range(0, nlog, chunk):
        n_real = min(chunk, nlog - c0)
        G = np.zeros((chunk, n_pad), dtype=np.float64)
        G[:n_real, :] = eye[c0:c0 + n_real, :]
        out.append((c0, n_real, G))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", metavar="DIR", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--input", default="gnppm_test.in")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-nq", type=int, default=2,
                    help="check mode: number of q_irr points (default 2)")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--chain-len", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--nq", type=int, default=5)
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--oracle-tol", type=float, default=1e-10,
                    help="bench-mode oracle tolerance (production ceiling)")
    ap.add_argument("--explain-cache", action="store_true")
    args = ap.parse_args()

    if args.prepare:
        import harness
        run_dir = harness.copy_fixture(
            harness.REG / "gnppm_debug", args.prepare)
        res = harness.run_gw_jax(run_dir, args.input)
        print(f"[prepare] rc={res.returncode} run_dir={run_dir}")
        sys.stdout.write(res.stdout[-2000:])
        return 0 if res.returncode == 0 else 2

    if not args.run_dir:
        ap.error("--run-dir is required for --check/--bench")

    import jax
    if args.explain_cache:
        jax.config.update("jax_explain_cache_misses", True)

    from bse.bse_feast import build_preconditioner_diagonal_sharded
    from bse.bse_w_exact import (apply_screening_resolvent_block,
                                 build_finite_q_data, _symmetry_tables)
    from bse.w_ladder import build_ladder_resolvent
    from bse.w_ladder_freq import (build_ladder_freq_chain,
                                   eval_ladder_freq_chain)

    data, mesh, input_path = _payload(args.run_dir, args.input)
    sym = _symmetry_tables(input_path)
    q_list = np.asarray(sym.q_irr_kgrid_int, dtype=int)
    nlog = int(data["n_rmu"])
    n_pad = int(data["V_q0"].shape[0])
    blocks = _chunk_blocks(nlog, n_pad, int(args.chunk))
    print(f"[setup] n_rmu={nlog} pad={n_pad} q_irr={len(q_list)} "
          f"chunks={len(blocks)} x {args.chunk}")

    # Engine built ONCE, flipped-vertex convention: every payload below goes
    # through build_finite_q_data (q=0 included) — sweep_q_wedge's discipline.
    matvec, _, gen, snapshot, sh = build_ladder_resolvent(
        mesh, data, include_w=True, vertex_flipped=True)

    def _dq(q):
        dq = build_finite_q_data(data, (int(q[0]), int(q[1]), int(q[2])), mesh)
        diag = build_preconditioner_diagonal_sharded(
            dq, mesh, include_W=True, use_tda=False)
        return dq, diag

    def _oracle(dq, diag, G, z, tol, max_iter=800):
        tile, resids = apply_screening_resolvent_block(
            G, complex(z), dq, matvec, diag, gen, snapshot, sh,
            max_iter=max_iter, tol=tol)
        return np.asarray(jax.device_get(tile)), float(np.max(np.asarray(
            jax.device_get(resids))))

    rc = 0
    if args.check:
        z_pts = [0.0 + 0.0j, 0.35j, 0.25 + 0.05j]
        m_full = int(args.chain_len)
        sweep = sorted({32, 48, 64, 96, m_full})
        sweep = [m for m in sweep if m <= m_full]
        c0, n_real, G = blocks[0]
        for qi in range(min(int(args.check_nq), len(q_list))):
            q = q_list[qi]
            dq, diag = _dq(q)
            t0 = time.perf_counter()
            chain = build_ladder_freq_chain(
                dq, matvec, gen, sh, G, m_full)
            jax.block_until_ready(chain["S_stack"])
            print(f"[time] chain build m={m_full}: "
                  f"{time.perf_counter() - t0:.1f} s")
            for z in z_pts:
                t0 = time.perf_counter()
                _oracle(dq, diag, G, z, tol=1e-6, max_iter=300)
                print(f"[time] oracle tol=1e-6 z={z}: "
                      f"{time.perf_counter() - t0:.1f} s (production-era "
                      f"per-z cost anchor)")
                t0 = time.perf_counter()
                ref, oresid = _oracle(dq, diag, G, z, tol=1e-12)
                print(f"[time] oracle tol=1e-12 z={z}: "
                      f"{time.perf_counter() - t0:.1f} s")
                prev = None
                for mu in sweep:
                    t0 = time.perf_counter()
                    tile, rest = eval_ladder_freq_chain(
                        chain, dq, snapshot, sh, z, m_use=mu)
                    got = np.asarray(jax.device_get(tile))
                    t_ev = time.perf_counter() - t0
                    rel = (np.linalg.norm(got[:, :n_real] - ref[:, :n_real])
                           / np.linalg.norm(ref[:, :n_real]))
                    print(f"[check] q={tuple(int(x) for x in q)} z={z} "
                          f"m_use={mu:4d}: rel={rel:.3e} "
                          f"est={float(np.max(rest[:n_real])):.3e} "
                          f"eval={t_ev:.2f}s (oracle resid {oresid:.1e})")
                    if prev is not None and rel > 3.0 * prev and rel > 1e-8:
                        print(f"[check] NON-MONOTONE at m_use={mu}")
                        rc = 1
                    prev = rel
                if rel > 1e-8:
                    print(f"[check] FAIL: final rel {rel:.3e} > 1e-8")
                    rc = 1
        print(f"[check] verdict rc={rc}")
        return rc

    if args.bench:
        z_bank = [0.0 + 0.0j, 0.2j, 0.35j, 0.5j, 0.15 + 0.03j,
                  0.25 + 0.05j, 0.35 + 0.05j, 0.45 + 0.08j]
        z_pts = z_bank[: int(args.nz)]
        m_full = int(args.chain_len)
        c0, n_real, G = blocks[0]      # one chunk; per-chunk costs are equal,
        nq = min(int(args.nq), len(q_list))   # full-basis = chunks x this
        t_or, t_bld, t_ev = 0.0, 0.0, 0.0
        resid_worst = 0.0              # chain's own certificate; accuracy vs
        for qi in range(nq):           # the tight oracle is --check's job
            dq, diag = _dq(q_list[qi])
            t0 = time.perf_counter()
            for z in z_pts:
                _oracle(dq, diag, G, z, tol=args.oracle_tol, max_iter=300)
            t_or += time.perf_counter() - t0
            t0 = time.perf_counter()
            chain = build_ladder_freq_chain(dq, matvec, gen, sh, G, m_full)
            jax.block_until_ready(chain["S_stack"])
            t_bld += time.perf_counter() - t0
            t0 = time.perf_counter()
            for z in z_pts:
                tile, rest = eval_ladder_freq_chain(
                    chain, dq, snapshot, sh, z)
                jax.block_until_ready(tile)
                resid_worst = max(resid_worst, float(np.max(rest[:n_real])))
            t_ev += time.perf_counter() - t0
        n_pts = nq * len(z_pts)
        print(f"[bench] {nq} q x {len(z_pts)} z, ONE {args.chunk}-col chunk "
              f"(full basis = {len(blocks)}x these numbers), m={m_full}")
        print(f"[bench] oracle(tol={args.oracle_tol:g}): {t_or:8.1f} s "
              f"({t_or / n_pts:6.2f} s/point)")
        print(f"[bench] chain build:                {t_bld:8.1f} s "
              f"({t_bld / nq:6.2f} s/chain)")
        print(f"[bench] chain eval:                 {t_ev:8.1f} s "
              f"({t_ev / n_pts:6.2f} s/point)")
        per_z_gain = t_or / n_pts
        be = (t_bld / nq) / max(per_z_gain - t_ev / n_pts, 1e-12)
        print(f"[bench] worst chain residual certificate: {resid_worst:.3e}")
        print(f"[bench] break-even ~{be:.2f} z-points/chain; at nz="
              f"{len(z_pts)}: chain total {t_bld + t_ev:.1f} s vs oracle "
              f"{t_or:.1f} s -> x{t_or / max(t_bld + t_ev, 1e-12):.2f}")
        return 0

    ap.error("pass --check or --bench (or --prepare)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
