#!/usr/bin/env python3
"""Mixed-precision bake-off for the ladder-W resolvent (``bse.w_ladder_mixedprec``).

The hypothesis: the ladder solve is 96-99 % direct rung (opt_precond RESULTS §8),
the rung is an FFT-convolution + thin-contraction chain whose cost is set by
bytes moved, and an A100 moves complex64 at twice the rate of complex128.  Run
the KRYLOV iteration in c64 and recover c128 accuracy with an outer c128
iterative-refinement loop.

Argv-driven (tests/bench convention; pytest does not collect this directory):

    python3 tests/bench/bench_w_ladder_mixedprec.py --prepare /path/to/rundir

    # does operand casting alone produce a c64 program?  (jaxpr census, seconds)
    ... --run-dir RUNDIR --dtype-audit

    # per-matvec ms, c64 vs c128, at block width 1/2/4 (the whole premise)
    ... --run-dir RUNDIR --matvec --widths 1,2,4

    # where does a c64 GMRES stop converging?  (inner-tol sweep, 1 chunk)
    ... --run-dir RUNDIR --floor

    # THE TABLE: rounds sweep vs the c128 baseline and the 1e-12 c128 oracle
    ... --run-dir RUNDIR --refine --nq 1 --nz 1 --chunk 64

    # gate levels: q=0 hermiticity and W(-q) = conj(W(q)) reciprocity
    ... --run-dir RUNDIR --gate

    # the owner's bar: tile error -> QP-channel energy error, in meV
    ... --run-dir RUNDIR --qp

Every accuracy number in this file is an HONEST ``||b||``-relative TRUE residual
(the c128 residual of the c128 operator) or a per-column relative tile error
against a c128 ``tol=1e-12`` oracle on the SAME operator.  The mixed engine
never reports its inner projected residual.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

RY_TO_MEV = 13605.693122994


# --- MPA-shaped z bank (Ry): static, imaginary axis, damped complex ----------
Z_BANK = [0.0 + 0.0j, 0.35j, 0.25 + 0.05j]


def _log(s):
    import jax
    if jax.process_index() == 0:
        print(s, flush=True)


def _payload(run_dir, input_name, n_val=10**9, n_cond=10**9):
    from bse import bse_io
    from common.collectives import single_device_mesh
    input_path = os.path.join(run_dir, input_name)
    restart = bse_io._find_restart_file(input_path)
    mesh = single_device_mesh()
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=n_val, n_cond=n_cond, mesh_xy=mesh,
        input_file=input_path, inject_head=False, load_v_full=True)
    return data, mesh, input_path


def _probe_blocks(nlog, n_pad, chunk):
    """The identity probe walk ``w_ladder.compute_wc_qwedge`` builds: rows are
    probe columns, the walk stops at ``nlog`` (a pad column has a zero rhs and
    NaNs the whole column)."""
    eye = np.eye(n_pad, dtype=np.float64)
    out = []
    for c0 in range(0, nlog, chunk):
        n_real = min(chunk, nlog - c0)
        G = np.zeros((chunk, n_pad), dtype=np.float64)
        G[:n_real, :] = eye[c0:c0 + n_real, :]
        out.append((c0, n_real, G))
    return out


def _rel_cols(got, ref, nlog, n_real):
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
    ap.add_argument("--dtype-audit", action="store_true")
    ap.add_argument("--matvec", action="store_true")
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--qp", action="store_true")
    ap.add_argument("--tiny", action="store_true",
                    help="2v2c window (the dense-oracle fixture shape): a "
                         "seconds-scale correctness cell before any timing")
    ap.add_argument("--widths", default="1,2,4")
    ap.add_argument("--rounds", default="1,2,3")
    ap.add_argument("--schedules",
                    default="1e-6,1e-6:1e-4,1e-6:1e-6,1e-7:1e-5:1e-5",
                    help="comma-separated refinement SCHEDULES; each is a "
                         "colon-separated per-round inner (c64) tolerance. "
                         "Round r>1 only has to shrink a correction that is "
                         "already ~(round r-1 tol) of ||b||, so a loose "
                         "tolerance there is the whole cost story.")
    ap.add_argument("--tol-low", default="1e-5",
                    help="comma list of inner (c64) GMRES tolerances")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="production c128 baseline tolerance")
    ap.add_argument("--oracle-tol", type=float, default=1e-12)
    ap.add_argument("--max-iter", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--chunks", default="1", choices=["all", "1"])
    ap.add_argument("--precision", default="highest",
                    choices=["default", "high", "highest", "tensorfloat32",
                             "float32", "bfloat16_3x"],
                    help="jax_default_matmul_precision for the f32 dots the "
                         "c64 program emits.  XLA:GPU's DEFAULT is TF32 "
                         "(10-bit mantissa, eps ~ 4.9e-04) -- measured to cost "
                         "the c64 matvec 2.5e-04 forward error against a "
                         "4.7e-08 representation floor.  c128 is unaffected "
                         "(f64 dots have no TF32 path), so this knob moves the "
                         "c64 arm only.")
    ap.add_argument("--nq", type=int, default=1)
    ap.add_argument("--nz", type=int, default=1)
    ap.add_argument("--qs", default=None, help="comma list of q_irr INDICES")
    ap.add_argument("--c128-tols", default="",
                    help="EXTRA c128 baseline tolerances to run as arms. The "
                         "control that separates 'c64 helped' from 'we asked "
                         "for less accuracy': a mixed R=1 arm at tol_low=1e-5 "
                         "is not solving the baseline's problem, so it must be "
                         "compared with a c128 solve at the SAME delivered "
                         "accuracy, not with the 1e-9 production one.")
    ap.add_argument("--reps", type=int, default=3,
                    help="timed passes over the INTERLEAVED arm list; the "
                         "MINIMUM wall per arm is reported")
    ap.add_argument("--no-warm", dest="warm", action="store_false",
                    help="time the COLD call (compile included)")
    args = ap.parse_args()

    if args.prepare:
        import harness
        run_dir = harness.copy_fixture(harness.REG / "gnppm_debug", args.prepare)
        res = harness.run_gw_jax(run_dir, args.input)
        print(f"[prepare] rc={res.returncode} run_dir={run_dir}", flush=True)
        sys.stdout.write(res.stdout[-2000:])
        return 0 if res.returncode == 0 else 2

    if not args.run_dir:
        ap.error("--run-dir is required")

    import jax
    import jax.numpy as jnp

    # BEFORE anything is traced: the f32 dot precision is a compile-time
    # property of every program built below.
    jax.config.update("jax_default_matmul_precision", args.precision)

    from bse.bse_feast import (build_preconditioner_diagonal_sharded,
                               ladder_matvec_operands)
    from bse.bse_w_exact import (apply_screening_resolvent_block,
                                 build_finite_q_data, build_probe_rhs,
                                 enforce_trs_pair_gauge, _symmetry_tables)
    from bse.w_ladder import build_ladder_resolvent, _accumulate_columns
    from bse import w_ladder_mixedprec as mp
    from common.collectives import gather_to_host

    nv = 2 if args.tiny else 10**9
    nc = 2 if args.tiny else 10**9
    data, mesh, input_path = _payload(args.run_dir, args.input, nv, nc)
    sym = _symmetry_tables(input_path)
    q_list = np.asarray(sym.q_irr_kgrid_int, dtype=int)
    nlog = int(data["n_rmu"])
    n_pad = int(data["V_q0"].shape[0])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])

    # The ladder arm's gauge (w_ladder.sweep_q_wedge, include_w=True) — a pure
    # basis choice, applied ONCE before the operator is built.
    data = enforce_trs_pair_gauge(data, mesh)
    matvec, _, gen, snapshot, sh = build_ladder_resolvent(
        mesh, data, include_w=True)

    zs = [complex(z) for z in Z_BANK[:max(1, args.nz)]]
    if args.qs:
        qidx = [int(t) for t in args.qs.split(",")]
    else:
        qidx = list(range(min(args.nq, len(q_list))))
    chunk = min(int(args.chunk), n_pad)
    blocks_all = _probe_blocks(nlog, n_pad, chunk)
    blocks = blocks_all if args.chunks == "all" else blocks_all[:1]
    rounds_list = [int(r) for r in args.rounds.split(",")]
    tols_low = [float(t) for t in args.tol_low.split(",")]
    scheds = [[float(t) for t in s.split(":")]
              for s in args.schedules.split(",") if s.strip()]
    c128_tols = [float(t) for t in args.c128_tols.split(",") if t.strip()]

    _log(f"[setup] n_rmu={nlog} pad={n_pad} nk={nk} n_val={data['n_val']} "
         f"n_cond={data['n_cond']} q_irr={len(q_list)} chunk={chunk} "
         f"blocks={len(blocks)}/{len(blocks_all)} max_iter={args.max_iter} "
         f"tiny={args.tiny}")
    _log(f"[setup] LORRAX_BANDS_GEMM_FFI="
         f"{os.environ.get('LORRAX_BANDS_GEMM_FFI', '<unset>')} "
         f"jax_default_matmul_precision={args.precision!r} "
         f"devices={jax.devices()}")

    def _dq(q):
        dq = build_finite_q_data(data, (int(q[0]), int(q[1]), int(q[2])), mesh)
        diag = build_preconditioner_diagonal_sharded(
            dq, mesh, include_W=True, use_tda=False)
        return dq, diag

    # ---------------- the two solve arms, same seed and same readout --------
    def run_c128(dq, diag, z, tol, blks, max_iter):
        acc, its, rr = None, [], []
        for c0, n_real, G in blks:
            rhs = build_probe_rhs(G, dq, gen, sh)
            tile, resids, iters = apply_screening_resolvent_block(
                G, complex(z), dq, matvec, diag, gen, snapshot, sh,
                max_iter=max_iter, tol=tol, return_iters=True, rhs=rhs,
                operands_fn=ladder_matvec_operands)
            acc = _accumulate_columns(acc, tile, c0, n_real, n_pad, mesh)
            its.append(np.asarray(gather_to_host(iters))[:n_real])
            rr.append(np.asarray(gather_to_host(resids))[:n_real])
        jax.block_until_ready(acc)
        return acc, np.concatenate(its), np.concatenate(rr)

    def run_mixed(dq, diag, z, tol_low, blks, max_iter, low=None):
        """``tol_low`` is the per-round schedule (a list); ``low`` defaults to
        c64 and is c128 only for the CONTROL arm."""
        low = jnp.complex64 if low is None else low
        acc, its, rr = None, [], []
        for c0, n_real, G in blks:
            rhs = build_probe_rhs(G, dq, gen, sh)
            tile, resids, iters = mp.apply_mixed_resolvent_block(
                G, complex(z), dq, matvec, diag, gen, snapshot, sh,
                max_iter=max_iter, tol_low=tol_low,
                operands_fn=ladder_matvec_operands, low_dtype=low, rhs=rhs)
            acc = _accumulate_columns(acc, tile, c0, n_real, n_pad, mesh)
            its.append(np.asarray(gather_to_host(iters))[:n_real])
            rr.append(np.asarray(gather_to_host(resids))[:n_real])
        jax.block_until_ready(acc)
        return acc, np.concatenate(its), np.concatenate(rr)

    def timed(fn, *a, **kw):
        """WARM then time.  Every distinct refinement-round count is its own
        compiled program (``rounds`` is static, it unrolls c128 matvecs), so a
        cold call would price XLA's compiler against the baseline's already-warm
        engine.  ``--no-warm`` prices the cold call instead."""
        if args.warm:
            fn(*a, **kw)
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        return out, time.perf_counter() - t0

    rc = 0

    # ======================================================================
    #  --dtype-audit : did casting the operands actually make a c64 program?
    # ======================================================================
    if args.dtype_audit:
        _log("\n=== DTYPE AUDIT — operand casting alone, or a threaded twin? ===")
        dq, diag = _dq(q_list[0])
        ops_hi = ladder_matvec_operands(dq)
        ops_lo = mp.cast_operands(ops_hi, jnp.complex64)
        _log("[operands] " + ", ".join(
            f"{i}:{str(a.dtype)}->{str(b.dtype)}"
            for i, (a, b) in enumerate(zip(ops_hi, ops_lo))))
        ncp, nvp = int(dq["n_cond_pad"]), int(dq["n_val_pad"])
        rng = np.random.default_rng(0)
        shp = (2, 1, ncp, nvp, nk)
        x128 = jnp.asarray(rng.standard_normal(shp) + 1j * rng.standard_normal(shp),
                           dtype=jnp.complex128)
        x64 = x128.astype(jnp.complex64)
        for tag, x, ops in (("c128", x128, ops_hi), ("c64", x64, ops_lo)):
            cen = mp.jaxpr_dtype_census(lambda xx, *oo: matvec(xx, *oo), x, *ops)
            out = matvec(x, *ops)
            _log(f"[jaxpr {tag:>4}] out dtype={out.dtype}  census="
                 f"{ {k: v for k, v in sorted(cen.items())} }")
            if tag == "c64":
                bad = cen.get("complex128", 0) + cen.get("float64", 0)
                _log(f"[jaxpr c64 ] c128/f64 vars remaining = {bad}"
                     f"  -> {'OPERAND CASTING SUFFICES' if bad == 0 else 'DTYPE LEAK'}")
                if out.dtype != jnp.complex64:
                    _log("[jaxpr c64 ] REFUSED: matvec output is not c64")
                    rc = 2
        # THE FORWARD ERROR OF THE c64 MATVEC IS THE REFINEMENT RATE.  Classical
        # mixed-precision refinement converges at the backward error of the
        # low-precision solve, which is bounded below by this number -- so it,
        # not c64's 1.19e-07 eps, sets how many rounds a target residual costs.
        # Measured on a random vector AND on the physical probe rhs, plus the
        # CANCELLATION ratio that explains any gap to eps: the matvec is a sum
        # of the D, ring and rung terms, and if they cancel, the relative error
        # of the sum is eps x (sum of term norms) / ||result||.
        dqa, _ = _dq(q_list[0])
        rhs_phys = build_probe_rhs(_probe_blocks(nlog, n_pad, 8)[0][2], dqa,
                                   gen, sh)[:, :1]
        for tag, xx in (("random", x128), ("probe rhs", rhs_phys)):
            h = np.asarray(gather_to_host(matvec(xx, *ops_hi)))
            l = np.asarray(gather_to_host(matvec(xx.astype(jnp.complex64),
                                                 *ops_lo)))
            _log(f"[fwd err {tag:>9}] ||A_64 x - A_128 x||/||A_128 x|| = "
                 f"{float(np.linalg.norm(l - h) / np.linalg.norm(h)):.3e}")
        # term-by-term, c128, to price the cancellation
        def _zero(ops, idx):
            o = list(ops)
            for i in idx:
                o[i] = jnp.zeros_like(o[i])
            return tuple(o)
        # operand order: 4=eps_c 5=eps_v 6=W_R 7=V_q0
        full = np.linalg.norm(np.asarray(gather_to_host(matvec(x128, *ops_hi))))
        parts = {}
        for tag, idx in (("D", (6, 7)), ("ring", (4, 5, 6)),
                         ("rung", (4, 5, 7))):
            parts[tag] = np.linalg.norm(np.asarray(gather_to_host(
                matvec(x128, *_zero(ops_hi, idx)))))
        tot = sum(parts.values())
        _log(f"[terms ] ||A x||={full:.4e} ; " + " ".join(
            f"||{k} x||={v:.4e}" for k, v in parts.items())
            + f" ; cancellation (sum/total) = {tot/full:.2f}x"
              f" -> eps-floor ~ {1.19e-7*tot/full:.2e}")

        # WHICH down-cast costs the accuracy?  Cast ONE group at a time (the
        # trial vector counts as a group) and measure the forward error of the
        # otherwise-c128 matvec.  If a cheap operand carries the whole error,
        # keeping IT in c128 costs a few percent of the matvec and buys orders
        # of magnitude of refinement rate -- that is a real design fork, so it
        # is measured rather than assumed.
        GROUPS = {"psi (0-3,10-13)": (0, 1, 2, 3, 10, 11, 12, 13),
                  "eps (4,5)": (4, 5), "W_R (6)": (6,), "V_q0 (7)": (7,),
                  "M (8,9)": (8, 9)}
        ref_out = np.asarray(gather_to_host(matvec(rhs_phys, *ops_hi)))
        rn = np.linalg.norm(ref_out)

        def _r64(a):
            """ROUND to c64/f32 and back: the operand's REPRESENTATION error
            with the arithmetic left in c128, so representation and
            accumulation separate."""
            return mp.cast_low(mp.cast_low(a, jnp.complex64), jnp.complex128)

        def _fwd(ops, xx):
            o = np.asarray(gather_to_host(matvec(xx, *ops)))
            return float(np.linalg.norm(o - ref_out) / rn)

        _log("[fwd scan] REPRESENTATION only (arithmetic stays c128), "
             "one group rounded to c64 at a time:")
        for tag, idx in GROUPS.items():
            o = list(ops_hi)
            for i in idx:
                o[i] = _r64(o[i])
            _log(f"[fwd scan] {tag:>16} rounded -> {_fwd(tuple(o), rhs_phys):.3e}")
        _log(f"[fwd scan] {'x':>16} rounded -> "
             f"{_fwd(ops_hi, _r64(rhs_phys)):.3e}")
        ops_r = tuple(_r64(o) for o in ops_hi)
        e_repr = _fwd(ops_r, _r64(rhs_phys))
        _log(f"[fwd scan] {'EVERYTHING':>16} rounded -> {e_repr:.3e}"
             f"   (c128 arithmetic)")
        _log(f"[fwd scan] {'true c64':>16} program  -> "
             f"{_fwd(ops_lo, rhs_phys.astype(jnp.complex64)):.3e}"
             f"   (representation + c64 ACCUMULATION)")

    # ======================================================================
    #  --matvec : the premise.  per-matvec ms, c64 vs c128, vs block width
    # ======================================================================
    if args.matvec:
        _log("\n=== MATVEC COST: complex64 vs complex128 ===")
        dq, diag = _dq(q_list[0])
        ops_hi = ladder_matvec_operands(dq)
        ops_lo = mp.cast_operands(ops_hi, jnp.complex64)
        ncp, nvp = int(dq["n_cond_pad"]), int(dq["n_val_pad"])
        wr = np.prod([int(s) for s in dq["W_R"].shape])
        _log(f"[geom] pair basis (c,v,k)=({ncp},{nvp},{nk})={ncp*nvp*nk} "
             f"N_mu={n_pad} W_R {tuple(int(s) for s in dq['W_R'].shape)} = "
             f"{wr*16/2**20:.1f} MiB c128 / {wr*8/2**20:.1f} MiB c64")
        hdr = (f"{'nb':>4} {'c128[ms]':>10} {'c64[ms]':>9} {'speedup':>8} "
               f"{'c128/col':>9} {'c64/col':>8}")
        _log(hdr); _log("-" * len(hdr))
        rng = np.random.default_rng(0)
        # THREE passes, INTERLEAVED precisions inside each, MINIMUM reported.
        # The first sweep of a fresh process carries autotuning and allocator
        # growth (measured: nb=4 c128 came out 3.3x its own nb=2 per-column
        # cost on a single-pass sweep); the minimum over passes is the number
        # that survives it, and interleaving keeps pool contention off one arm.
        for nb in [int(w) for w in args.widths.split(",")]:
            shp = (2, nb, ncp, nvp, nk)
            x = jax.lax.with_sharding_constraint(
                jnp.asarray(rng.standard_normal(shp) + 1j * rng.standard_normal(shp),
                            dtype=jnp.complex128), sh.X_full)
            xl = x.astype(jnp.complex64)
            best = {"c128": float("inf"), "c64": float("inf")}
            for _pass in range(3):
                for tag, xx, oo in (("c128", x, ops_hi), ("c64", xl, ops_lo)):
                    try:
                        for _ in range(2):
                            jax.block_until_ready(matvec(xx, *oo))
                        reps = 20 if nb <= 4 else 10
                        t0 = time.perf_counter()
                        for _ in range(reps):
                            jax.block_until_ready(matvec(xx, *oo))
                        best[tag] = min(
                            best[tag],
                            1e3 * (time.perf_counter() - t0) / reps)
                    except Exception as exc:
                        best[tag] = float("nan")
                        _log(f"[matvec] nb={nb} {tag}: {type(exc).__name__}: "
                             f"{str(exc)[:160]}")
            ms = best
            _log(f"{nb:4d} {ms['c128']:10.3f} {ms['c64']:9.3f} "
                 f"{ms['c128']/ms['c64']:8.2f}x {ms['c128']/nb:9.3f} "
                 f"{ms['c64']/nb:8.3f}")
            del x, xl

    # ======================================================================
    #  --floor : how far can a c64 Krylov iteration go on its own?
    # ======================================================================
    if args.floor:
        _log("\n=== c64 GMRES FLOOR (rounds=1, no refinement) ===")
        q = q_list[qidx[0]]
        dq, diag = _dq(q)
        z = zs[0]
        blk = blocks[:1]
        (acc_o, it_o, rr_o), t_o = timed(run_c128, dq, diag, z,
                                         args.oracle_tol, blk, args.max_iter)
        _log(f"[oracle ] c128 tol={args.oracle_tol:.0e}  iters "
             f"{it_o.mean():.1f}/{it_o.max()}  true resid {rr_o.max():.3e}  "
             f"{t_o:.1f} s")
        hdr = (f"{'tol_low':>9} {'iters mean/max':>15} {'true resid':>12} "
               f"{'tile rel err':>13} {'wall[s]':>8}")
        _log(hdr); _log("-" * len(hdr))
        for tl in tols_low:
            (acc, it, rr), t = timed(run_mixed, dq, diag, z, [tl], blk,
                                     args.max_iter)
            _log(f"{tl:9.0e} {it.mean():7.1f}/{it.max():<7d} {rr.max():12.3e} "
                 f"{_rel_cols(gather_to_host(acc), gather_to_host(acc_o), nlog, chunk):13.3e} "
                 f"{t:8.1f}")

    # ======================================================================
    #  --refine : THE TABLE
    # ======================================================================
    if args.refine:
        _log("\n=== REFINEMENT: mixed vs c128 baseline at equal true residual ===")
        _log(f"[protocol] arms INTERLEAVED, {args.reps} passes, MIN wall "
             f"reported per arm (the pool is shared; opt_precond RESULTS §3 "
             f"documents 2.4x contention drift between legs).  Every arm is "
             f"warmed before the first timed pass.")
        for iq in qidx:
            q = q_list[iq]
            dq, diag = _dq(q)
            for z in zs:
                ncols = sum(b[1] for b in blocks)
                _log(f"\n--- q={tuple(int(x) for x in q)}  z={z}  "
                     f"cols={ncols} ---")
                # ARMS: (tag, callable).  The oracle first so `ref` exists.
                arms = [(f"c128 oracle tol={args.oracle_tol:.0e}",
                         lambda: run_c128(dq, diag, z, args.oracle_tol, blocks,
                                          args.max_iter)),
                        (f"c128 baseline tol={args.tol:.0e}",
                         lambda: run_c128(dq, diag, z, args.tol, blocks,
                                          args.max_iter)),
                        # CONTROL: the refinement ENGINE at full precision.
                        # Any gap to the baseline row is the engine (its one
                        # extra c128 verification matvec and the restart), not
                        # the precision -- so the mixed rows can be read as a
                        # precision effect.
                        ("CONTROL c128 engine R=1",
                         lambda: run_mixed(dq, diag, z, [args.tol], blocks,
                                           args.max_iter, jnp.complex128))]
                for tt in c128_tols:
                    arms.append((f"c128 tol={tt:.0e}",
                                 (lambda t_: lambda: run_c128(
                                     dq, diag, z, t_, blocks,
                                     args.max_iter))(tt)))
                for sc in scheds:
                    arms.append(("mixed " + ":".join(f"{x:.0e}" for x in sc),
                                 (lambda s_: lambda: run_mixed(
                                     dq, diag, z, s_, blocks,
                                     args.max_iter))(sc)))
                best = {t: float("inf") for t, _ in arms}
                res = {}
                for _p in range(args.reps + (1 if args.warm else 0)):
                    for tag, fn in arms:
                        t0 = time.perf_counter()
                        out = fn()
                        dt = time.perf_counter() - t0
                        if _p or not args.warm:      # pass 0 is the warm pass
                            best[tag] = min(best[tag], dt)
                        res[tag] = out
                ref = gather_to_host(res[arms[0][0]][0])
                t_base = best[arms[1][0]]
                hdr = (f"{'method':>26} {'iters m/max':>13} {'true resid':>11} "
                       f"{'tile rel err':>12} {'herm':>10} {'wall[s]':>8} "
                       f"{'vs base':>8}")
                _log(hdr); _log("-" * len(hdr))
                n = min(chunk * len(blocks), nlog)
                for tag, _ in arms:
                    acc, it, rr = res[tag]
                    tile = gather_to_host(acc)
                    e = _rel_cols(tile, ref, nlog, n)
                    hm = mp.hermiticity(np.asarray(tile)[:n, :n])
                    t = best[tag]
                    _log(f"{tag:>26} {it.mean():5.1f}/{it.max():<7d} "
                         f"{rr.max():11.3e} {e:12.3e} {hm:10.2e} {t:8.2f} "
                         f"{t_base/t:7.2f}x")

    # ======================================================================
    #  --gate : hermiticity at q=0 and W(-q) = conj(W(q)) reciprocity
    # ======================================================================
    if args.gate:
        _log("\n=== GATE LEVELS: hermiticity (q=0) and reciprocity (q, -q) ===")
        grid = (int(data["nkx"]), int(data["nky"]), int(data["nkz"]))
        z = zs[0]
        n = min(chunk * len(blocks), nlog)

        def tiles_for(q):
            dq, diag = _dq(q)
            out = {}
            (a, i_, r_), t = timed(run_c128, dq, diag, z, args.oracle_tol,
                                   blocks, args.max_iter)
            out["c128 oracle"] = (gather_to_host(a), r_.max(), t)
            (a, i_, r_), t = timed(run_c128, dq, diag, z, args.tol, blocks,
                                   args.max_iter)
            out[f"c128 tol={args.tol:.0e}"] = (gather_to_host(a), r_.max(), t)
            for sc in scheds:
                (a, i_, r_), t = timed(run_mixed, dq, diag, z, sc, blocks,
                                       args.max_iter)
                out["mixed " + ":".join(f"{x:.0e}" for x in sc)] = (
                    gather_to_host(a), r_.max(), t)
            return out

        t0 = tiles_for(q_list[0])
        _log(f"[q=(0,0,0)] hermiticity of the leading {n}x{n} logical block")
        hdr = f"{'method':>26} {'true resid':>11} {'max|W-W^H|/max|W|':>19}"
        _log(hdr); _log("-" * len(hdr))
        for k, (tile, rmax, t) in t0.items():
            _log(f"{k:>26} {rmax:11.3e} {mp.hermiticity(np.asarray(tile)[:n, :n]):19.3e}")

        if len(q_list) > 1:
            q = q_list[1]
            mq = tuple(int((-int(q[i])) % grid[i]) for i in range(3))
            _log(f"\n[reciprocity] q={tuple(int(x) for x in q)} vs "
                 f"-q={mq}: max|W(q) - conj(W(-q))| / max|W(q)| on the "
                 f"{n}x{n} block")
            tq = tiles_for(q)
            tm = tiles_for(np.asarray(mq, dtype=int))
            hdr = f"{'method':>26} {'reciprocity':>13}"
            _log(hdr); _log("-" * len(hdr))
            for k in tq:
                a = np.asarray(tq[k][0])[:n, :n]
                b = np.asarray(tm[k][0])[:n, :n]
                den = np.max(np.abs(a)) or 1.0
                _log(f"{k:>26} {float(np.max(np.abs(a - np.conj(b))))/den:13.3e}")

    # ======================================================================
    #  --qp : the owner's bar.  tile error -> QP-channel energy, in meV
    # ======================================================================
    if args.qp:
        _log("\n=== QP MAPPING (screened-exchange channel proxy), in meV ===")
        _log("Sigma_SEX^q(c,k) = -(1/Nk) sum_v sum_{mu,nu} conj(M[k,c,v,mu]) "
             "Wc[mu,nu] M[k,c,v,nu]  -- the W-bearing term of COHSEX, one q "
             "channel, Ry -> meV.  A PROXY: it is the actual contraction the "
             "Sigma assembly performs on this tile, not a driver QP run.")
        q = q_list[qidx[0]]
        dq, diag = _dq(q)
        z = zs[0]
        M = np.asarray(gather_to_host(dq["M_X"]))       # (k, c, v, mu)

        def sex(tile):
            W = np.asarray(tile)[:nlog, :nlog]
            n = min(chunk * len(blocks), nlog)
            Wn = W[:n, :n]
            Mn = M[:, :, :, :n]
            t = np.einsum("kcvm,mn->kcvn", Mn, Wn)
            return -np.einsum("kcvn,kcvn->kc", np.conj(Mn), t).real / nk

        (acc_o, _, rr_o), _ = timed(run_c128, dq, diag, z, args.oracle_tol,
                                    blocks, args.max_iter)
        e_o = sex(gather_to_host(acc_o))
        _log(f"[scale] |Sigma_SEX| max = {np.abs(e_o).max()*RY_TO_MEV/1e3:.3f} eV "
             f"over {e_o.size} (c,k) states, q={tuple(int(x) for x in q)}")
        hdr = (f"{'method':>26} {'tile rel err':>12} "
               f"{'max|dSigma| [meV]':>18} {'rms [meV]':>10} {'<1 meV?':>8}")
        _log(hdr); _log("-" * len(hdr))
        cases = [(f"c128 tol={args.tol:.0e}", None)]
        for sc in scheds:
            cases.append(("mixed " + ":".join(f"{x:.0e}" for x in sc), sc))
        for tag, sc in cases:
            if sc is None:
                (a, _, _), _ = timed(run_c128, dq, diag, z, args.tol, blocks,
                                     args.max_iter)
            else:
                (a, _, _), _ = timed(run_mixed, dq, diag, z, sc, blocks,
                                     args.max_iter)
            tile = gather_to_host(a)
            d = (sex(tile) - e_o) * RY_TO_MEV
            err = _rel_cols(tile, gather_to_host(acc_o), nlog,
                            min(chunk * len(blocks), nlog))
            _log(f"{tag:>26} {err:12.3e} {np.abs(d).max():18.4f} "
                 f"{np.sqrt((d**2).mean()):10.4f} "
                 f"{'YES' if np.abs(d).max() < 1.0 else 'no':>8}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
