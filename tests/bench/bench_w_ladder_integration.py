#!/usr/bin/env python3
"""Speed-integration legs for the ladder-W solve (``opt_integration`` arm).

Argv-driven (tests/bench convention; pytest does not collect this directory).

    --mode core     A/B the CHANGED ``bse_feast._gmres_solve_core`` against a
                    pristine copy of the same file (``--baseline-src DIR``) on
                    a tiny dense synthetic operator: bit-identity of x and of
                    the iteration count under the legacy stopping norm, and the
                    cost/accuracy effect of the two changes, in one process.
                    Seconds, no fixture.

    --mode solve    ONE probe chunk of the REAL ladder solve at one (q, z) —
                    wall, iteration counts, ms per column-iteration — with the
                    knobs this arm added (``--max-iter``, ``--resid-norm``,
                    ``--block``, ``--x0``).  The minimal-fixture leg: default
                    16 probe columns, 1 q, 1 z.

Scope of every number: single process, on the whole mesh ``lx run -G`` gave it
(``--mesh`` omitted; it defaulted to ``1,1`` until 2026-08-27, so numbers taken
before that date were one device wide however many the job held).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np


def _log(s):
    import jax
    if jax.process_index() == 0:
        print(s, flush=True)


# ---------------------------------------------------------------------------
# --mode core : the solver core, against a pristine copy of itself
# ---------------------------------------------------------------------------


def _call(fn, *a, **kw):
    """Call ``fn`` dropping any keyword it does not accept.

    So the SAME bench body drives the live tree and a PRISTINE copy of it on
    ``PYTHONPATH`` (the baseline arm): the pristine
    ``build_ladder_resolvent`` has no ``fuse_ladder_rung`` and the pristine
    ``apply_screening_resolvent_block`` has neither ``rhs`` nor
    ``resid_relative_to``, and dropping them is exactly "run the old
    behaviour" rather than a second code path in the harness.
    """
    import inspect
    ok = set(inspect.signature(fn).parameters)
    return fn(*a, **{k: v for k, v in kw.items() if k in ok})


def _pristine_core(path: str, live_mod):
    """The PRISTINE ``_gmres_solve_core`` as a callable, in this process.

    The file cannot simply be imported (``bse_feast`` uses package-relative
    imports and there is only one ``bse`` package on the path), and importing a
    second copy of the package would give the A/B two different ``matvec``
    factories.  So: lift the ONE function's source text out of the pristine
    file and exec it against the LIVE module's globals.  Both arms then share
    every helper (``_apply_shifted_matvec``, jnp, jax) and differ in exactly
    the lines under test.
    """
    src = open(path).read()
    i = src.index("def _gmres_solve_core(")
    j = src.index("\ndef ", i + 1)
    ns = dict(live_mod.__dict__)
    exec(compile(src[i:j], path, "exec"), ns)
    return ns["_gmres_solve_core"]


def mode_core(args):
    import jax
    import jax.numpy as jnp
    from bse import bse_feast as live

    rng = np.random.default_rng(0)
    n = int(args.core_n)
    # A stiff-ish non-Hermitian operator with a wide diagonal spread — the
    # feature of the real ladder operator that makes the diagonal
    # preconditioner both useful and a bad initial GUESS.
    diag = (np.linspace(0.5, 40.0, n) + 0.1j * rng.standard_normal(n))
    off = 0.30 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    A = np.diag(diag) + off
    z = complex(args.core_z)
    b_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    A_j = jnp.asarray(A, dtype=jnp.complex128)
    b_j = jnp.asarray(b_np, dtype=jnp.complex128)[None, :]      # batch axis
    diag_j = jnp.asarray(-diag, dtype=jnp.complex128)[None, :]  # diag(H) = -? see below

    # matvec(x, *operands) must be H x.  H = -A here is irrelevant; take H = A
    # and diag_h = diag(A) so (z - H) is the shifted operator the core builds.
    diag_j = jnp.asarray(np.diag(A), dtype=jnp.complex128)[None, :]

    def matvec(x, M):
        return jnp.einsum("ij,bj->bi", M, x)

    ops = (A_j,)
    x_ref = np.linalg.solve(z * np.eye(n) - A, b_np)

    base = None
    if args.baseline_src:
        base = _pristine_core(
            os.path.join(args.baseline_src, "bse", "bse_feast.py"), live)

    def run(core, tol, **kw):
        f = jax.jit(lambda b, d, ops_: core(
            matvec, b, d, jnp.asarray(z, jnp.complex128), ops_,
            int(args.max_iter), jnp.asarray(tol, jnp.float64), **kw))
        x, k = jax.block_until_ready(f(b_j, diag_j, ops))
        t0 = time.perf_counter()
        for _ in range(args.reps):
            x, k = jax.block_until_ready(f(b_j, diag_j, ops))
        dt = (time.perf_counter() - t0) / args.reps
        xn = np.asarray(x)[0]
        rel = float(np.linalg.norm(xn - x_ref) / np.linalg.norm(x_ref))
        true_r = float(np.linalg.norm(b_np - (z * np.eye(n) - A) @ xn)
                       / np.linalg.norm(b_np))
        return xn, int(k), rel, true_r, 1e3 * dt

    tol = float(args.tol)
    _log(f"\n=== --mode core : n={n} z={z} tol={tol:g} cap={args.max_iter} ===")
    hdr = (f"{'arm':<28} {'iters':>6} {'rel vs exact':>13} {'true resid':>12} "
           f"{'ms':>9}")
    _log(hdr)
    _log("-" * len(hdr))
    rows = {}
    if base is not None:
        rows["pristine (r0 norm)"] = run(base, tol)
    rows["live, r0 norm"] = run(live._gmres_solve_core, tol,
                                resid_relative_to="r0")
    rows["live, b norm (default)"] = run(live._gmres_solve_core, tol)
    rows["live, b norm, x0=0"] = run(live._gmres_solve_core, tol,
                                     x0=jnp.zeros_like(b_j))
    for tag, (xn, k, rel, tr, ms) in rows.items():
        _log(f"{tag:<28} {k:6d} {rel:13.3e} {tr:12.3e} {ms:9.3f}")
    _log("-" * len(hdr))

    rc = 0
    if base is not None:
        xb, kb = rows["pristine (r0 norm)"][0], rows["pristine (r0 norm)"][1]
        xl, kl = rows["live, r0 norm"][0], rows["live, r0 norm"][1]
        same_bits = bool(np.array_equal(xb.view(np.float64),
                                        xl.view(np.float64)))
        d = float(np.max(np.abs(xb - xl)) / max(np.max(np.abs(xb)), 1e-300))
        _log(f"[core] legacy-norm A/B: iters {kb} vs {kl}; "
             f"bit-identical x = {same_bits}; max rel diff {d:.3e}")
        # The lstsq hoist alone was bit-identical (measured, 2026-08-16, before
        # the CGS2 change).  Replacing MGS+MGS by CGS2 changes the association
        # order of the orthogonalization, so the bar here is AGREEMENT, not
        # bits: the same exit index and a difference at the solve's own
        # accuracy, which is what "same algorithm, better-conditioned
        # arithmetic" is entitled to.
        if kb != kl or not (d <= 1e-9):
            _log("[core] FAIL: the core rewrite moved the exit index or the "
                 "answer by more than the solve's own accuracy.")
            rc = 1
        speed = rows["pristine (r0 norm)"][4] / max(rows["live, r0 norm"][4], 1e-12)
        _log(f"[core] per-solve cost pristine/live at cap {args.max_iter}: "
             f"{speed:.2f}x")
        tb = rows["pristine (r0 norm)"][3]
        tl = rows["live, b norm (default)"][3]
        _log(f"[core] stopping-norm semantics: delivered TRUE residual "
             f"{tb:.3e} (r0 norm, nominal {tol:g}) vs {tl:.3e} (b norm) "
             f"-- looseness factor {tb / max(tl, 1e-300):.1f}x")
    return rc


# ---------------------------------------------------------------------------
# --mode matvec : the width curve, BOTH accountings, one process
# ---------------------------------------------------------------------------

def _wedge_setup(args):
    """Payload + engine + one q-shifted payload — shared by matvec/solve."""
    import jax
    from bse import bse_io
    from bse.bse_ring_comm import create_mesh_xy_from_flags
    from bse.bse_feast import (build_preconditioner_diagonal_sharded,
                               ladder_matvec_operands, matvec_operands)
    from bse.bse_w_exact import (build_finite_q_data, enforce_trs_pair_gauge,
                                 _symmetry_tables)
    from bse.w_ladder import build_ladder_resolvent

    # ``--mesh`` omitted = the run's mesh, the same rule the six bse drivers'
    # --px/--py follow since 2026-08-27.  It defaulted to "1,1", so every
    # number this bench ever printed under ``lx run -G 4`` was measured on ONE
    # of the four GPUs while the header said otherwise — the same defect the
    # drivers were fixed for, in the instrument used to judge them.  (It is
    # now also a refusal, not a quiet 1x1: create_mesh_xy rejects a shape that
    # does not consume the device list.)
    px, py = ((None, None) if args.mesh is None
              else tuple(int(s) for s in args.mesh.split(",")))
    mesh = create_mesh_xy_from_flags(px, py)
    input_path = os.path.join(args.run_dir, args.input)
    restart = bse_io._find_restart_file(input_path)
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=input_path, inject_head=False, load_v_full=True)
    include_w = bool(args.include_w)
    if include_w:
        data = enforce_trs_pair_gauge(data, mesh)
    matvec, _, gen, snapshot, sh = _call(
        build_ladder_resolvent, mesh, data, include_w=include_w,
        fuse_ladder_rung=bool(int(str(getattr(args, "fuse", 1)).split(",")[0])))
    q_list = np.asarray(_symmetry_tables(input_path).q_irr_kgrid_int, dtype=int)
    ops_fn = ladder_matvec_operands if include_w else matvec_operands
    return dict(mesh=mesh, data=data, matvec=matvec, gen=gen,
                snapshot=snapshot, sh=sh, q_list=q_list, ops_fn=ops_fn,
                include_w=include_w, input_path=input_path,
                build_finite_q_data=build_finite_q_data,
                build_diag=build_preconditioner_diagonal_sharded,
                build_ladder_resolvent=build_ladder_resolvent)


def mode_matvec(args):
    """ONE ladder matvec vs block width, timed TWO ways in the same process.

    ``device``  — ``matvec_reps`` applications inside ONE ``lax.fori_loop``
                  inside one dispatch: pure device time, which is what the
                  production solve (a ``lax.scan`` of GMRES inside one jit)
                  actually pays per column.
    ``dispatch``— one ``block_until_ready`` per application: device time PLUS
                  the host round trip, which the production solve does NOT pay.

    The two bake-off arms disagreed on the b=1 point (16.10 vs 9.57 ms/col on
    this same fixture); this cell settles which accounting each was using and
    which one the production path is entitled to.
    """
    import jax
    import jax.numpy as jnp

    S = _wedge_setup(args)
    q = tuple(int(v) for v in S["q_list"][int(args.q_index.split(",")[0])])
    dq = S["build_finite_q_data"](S["data"], q, S["mesh"])
    ops = S["ops_fn"](dq)
    mv, sh = S["matvec"], S["sh"]
    nk = int(dq["nkx"] * dq["nky"] * dq["nkz"])
    nc, nv = int(dq["n_cond_pad"]), int(dq["n_val_pad"])
    n_pad = int(dq["V_q0"].shape[0])
    ns = int(dq["psi_c_X"].shape[2])
    _log(f"\n=== --mode matvec : q={q} pair=({nc},{nv},{nk}) N_mu={n_pad} "
         f"nspinor={ns} ; rung (mu,nu,s,s,k) buffer/col = "
         f"{n_pad*n_pad*ns*ns*nk*16/2**20:.1f} MiB ===")
    rows_first_width = [int(w) for w in args.batch_widths.split(",")]
    hdr = (f"{'nb':>4} {'device[ms]':>11} {'dev/col':>9} {'dispatch[ms]':>13} "
           f"{'disp/col':>9} {'peak[MiB]':>10}"
           + (f" {'ALT/col':>9}" if (S["include_w"] and args.fuse_ab) else ""))
    _log(hdr)
    _log("-" * len(hdr))
    rng = np.random.default_rng(0)
    dev = jax.local_devices()[0]
    reps = int(args.matvec_reps)
    # The rung-FUSION A/B: same payload, same operands, the other core.
    mv_alt = None
    if S["include_w"] and args.fuse_ab:
        mv_alt, _, _, _, _ = S["build_ladder_resolvent"](
            S["mesh"], S["data"], include_w=True,
            fuse_ladder_rung=not bool(int(str(args.fuse).split(",")[0])))
    rows = []
    for nb in [int(w) for w in args.batch_widths.split(",")]:
        shp = (2, nb, nc, nv, nk)
        x = jax.lax.with_sharding_constraint(
            jnp.asarray(rng.standard_normal(shp) + 1j * rng.standard_normal(shp),
                        dtype=jnp.complex128), sh.X_full)

        @jax.jit
        def _rep(x0, ops_=ops):
            return jax.lax.fori_loop(0, reps, lambda i, y: mv(y, *ops_), x0,
                                     unroll=1)

        jax.block_until_ready(_rep(x))                       # compile
        t0 = time.perf_counter()
        jax.block_until_ready(_rep(x))
        t_dev = 1e3 * (time.perf_counter() - t0) / reps

        jax.block_until_ready(mv(x, *ops))                   # warm
        t0 = time.perf_counter()
        for _ in range(max(reps // 4, 3)):
            jax.block_until_ready(mv(x, *ops))
        t_dis = 1e3 * (time.perf_counter() - t0) / max(reps // 4, 3)

        t_alt = float("nan")
        if mv_alt is not None:
            @jax.jit
            def _rep_alt(x0, ops_=ops):
                return jax.lax.fori_loop(0, reps, lambda i, y: mv_alt(y, *ops_),
                                         x0, unroll=1)
            jax.block_until_ready(_rep_alt(x))
            t0 = time.perf_counter()
            jax.block_until_ready(_rep_alt(x))
            t_alt = 1e3 * (time.perf_counter() - t0) / reps
            if nb == rows_first_width[0]:
                a_out = np.asarray(jax.device_get(mv(x, *ops)))
                b_out = np.asarray(jax.device_get(mv_alt(x, *ops)))
                den = max(float(np.max(np.abs(a_out))), 1e-300)
                rel = float(np.max(np.abs(a_out - b_out)) / den)
                _log(f"[fuse] fused-vs-unfused matvec agreement: rel {rel:.3e} "
                     f"(gate 1e-12) at nb={nb}")
                if not (rel <= 1e-12):
                    _log("[fuse] FAIL: the fused row does not reproduce the "
                         "unfused one -- the conjugation convention moved.")
        try:
            pk = dev.memory_stats()["peak_bytes_in_use"] / 2**20
        except Exception:
            pk = float("nan")
        alt_s = "" if t_alt != t_alt else f" {t_alt/nb:9.3f}"
        _log(f"{nb:4d} {t_dev:11.3f} {t_dev/nb:9.3f} {t_dis:13.3f} "
             f"{t_dis/nb:9.3f} {pk:10.1f}{alt_s}")
        rows.append((nb, t_dev, t_dis, t_alt))
        del x
    _log("-" * len(hdr))
    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        _log(f"[matvec] DEVICE   accounting: {a[1]/a[0]:.3f} ms/col at nb={a[0]}"
             f" -> {b[1]/b[0]:.3f} at nb={b[0]}  ({(a[1]/a[0])/(b[1]/b[0]):.2f}x)")
        _log(f"[matvec] DISPATCH accounting: {a[2]/a[0]:.3f} ms/col at nb={a[0]}"
             f" -> {b[2]/b[0]:.3f} at nb={b[0]}  ({(a[2]/a[0])/(b[2]/b[0]):.2f}x)")
        _log(f"[matvec] host round trip at nb=1 = {a[2]-a[1]:.3f} ms "
             f"({(a[2]-a[1])/max(a[2],1e-9):.0%} of the dispatch number); the "
             f"production solve pays it ZERO times per column (the scan and "
             f"the matvec are inside one compiled program).")
        if len(a) > 3 and a[3] == a[3]:
            tag = ("fused" if int(str(args.fuse).split(",")[0])
                   else "unfused")
            _log(f"[fuse] rung fusion at nb={a[0]}: this build ({tag}) "
                 f"{a[1]/a[0]:.3f} ms/col vs alternate {a[3]/a[0]:.3f} ms/col "
                 f"-- {a[3]/max(a[1], 1e-12):.2f}x")
    return 0


# ---------------------------------------------------------------------------
# --mode solve : one probe chunk of the real ladder solve
# ---------------------------------------------------------------------------

def mode_solve(args):
    """ONE probe chunk of the real ladder solve, arm by arm.

    Arms are the product of ``--fuse`` (rung fusion off/on), ``--resid-norms``
    (stopping-norm semantics), ``--max-iters`` (the cap) and ``--lift``
    (route A, the Woodbury ring removal).  Every arm runs on the SAME payload
    in the SAME process, so the ratios are protected from machine drift.
    """
    import jax
    import jax.numpy as jnp
    from bse import bse_w_exact as bwe
    from bse import w_ladder_precond as wlp
    apply_screening_resolvent_block = bwe.apply_screening_resolvent_block
    build_probe_rhs = getattr(bwe, "build_probe_rhs", None)

    S = _wedge_setup(args)
    q_list, mesh, sh = S["q_list"], S["mesh"], S["sh"]
    data = S["data"]
    n_pad = int(data["V_q0"].shape[0])
    nlog = int(data["n_rmu"])
    ncol = min(int(args.cols), nlog)
    py = mesh.devices.shape[1]
    ncol = max(int(np.ceil(ncol / py)) * py, py)
    G = np.eye(n_pad, dtype=np.float64)[:ncol, :]

    if args.lift and ncol != n_pad:
        _log(f"[solve] --lift skipped: route A's Dyson close needs EVERY column "
             f"of T (it inverts I - T on the padded extent), and this leg probes "
             f"{ncol} of {n_pad}.  Re-run with --cols {n_pad} for the lift arm.")
    fuses = [int(v) for v in str(args.fuse).split(",")]
    engines = {}
    for fv in fuses:
        engines[fv] = _call(S["build_ladder_resolvent"], mesh, data,
                            include_w=S["include_w"],
                            fuse_ladder_rung=bool(fv))

    _log(f"\n=== --mode solve : {ncol} cols, mesh "
         f"{mesh.devices.shape}, include_w={S['include_w']}, "
         f"n_rmu={nlog}/{n_pad} ===")
    hdr = (f"{'q':>9} {'z':>13} {'arm':>16} {'cap':>5} {'norm':>5} "
           f"{'wall[s]':>9} {'it mean':>8} {'it max':>7} {'ms/col-it':>10} "
           f"{'max true resid':>15}")
    _log(hdr)
    _log("-" * len(hdr))
    tiles = {}
    oracles = {}
    for iq in [int(v) for v in args.q_index.split(",")]:
        q = tuple(int(v) for v in q_list[iq])
        dq = S["build_finite_q_data"](data, q, mesh)
        for zs in args.zs.split(";"):
            z = complex(zs)
            if args.oracle_tol > 0:
                # ORACLE: the UNFUSED, plain-route operator at a tight
                # tolerance and a high cap — the reference every arm is
                # measured against.  Built from the fuse=0 engine when one
                # exists so the oracle shares no association order with the
                # fused arms it certifies.
                mv_o, _, gen_o, snap_o, _ = engines.get(
                    0, engines[fuses[0]])
                diag_o = S["build_diag"](dq, mesh, include_W=S["include_w"],
                                         use_tda=False)
                rhs_o = (build_probe_rhs(G, dq, gen_o, sh)
                         if build_probe_rhs is not None else None)
                out_o = _call(apply_screening_resolvent_block,
                              G, z, dq, mv_o, diag_o, gen_o, snap_o, sh,
                              max_iter=int(args.oracle_max_iter),
                              tol=float(args.oracle_tol), return_iters=True,
                              rhs=rhs_o, resid_relative_to="b",
                              operands_fn=S["ops_fn"])
                jax.block_until_ready(out_o)
                oracles[(q, zs)] = np.asarray(jax.device_get(out_o[0]))
                it_o = np.asarray(jax.device_get(out_o[2]))[:ncol]
                _log(f"{str(q):>9} {zs:>13} {'ORACLE':>16} "
                     f"{int(args.oracle_max_iter):5d} {'b':>5} "
                     f"{0.0:9.3f} {it_o.mean():8.2f} {it_o.max():7d} "
                     f"{0.0:10.3f} "
                     f"{np.asarray(jax.device_get(out_o[1]))[:ncol].max():15.3e}")
            for fv in fuses:
                matvec, _, gen, snapshot, _sh = engines[fv]
                diag_hq = S["build_diag"](dq, mesh,
                                          include_W=S["include_w"],
                                          use_tda=False)
                rhs = (build_probe_rhs(G, dq, gen, sh)
                       if build_probe_rhs is not None else None)
                for cap in [int(v) for v in args.max_iters.split(",")]:
                    for nrm in args.resid_norms.split(","):
                        for rep in range(2):        # first pass compiles
                            t0 = time.perf_counter()
                            out = _call(
                                apply_screening_resolvent_block,
                                G, z, dq, matvec, diag_hq, gen, snapshot, sh,
                                max_iter=cap, tol=float(args.tol),
                                return_iters=True, rhs=rhs,
                                resid_relative_to=nrm,
                                operands_fn=S["ops_fn"])
                            jax.block_until_ready(out)
                            dt = time.perf_counter() - t0
                        W_tile, resids, iters = out
                        it = np.asarray(jax.device_get(iters))[:ncol]
                        rs = np.asarray(jax.device_get(resids))[:ncol]
                        tag = ("fused" if fv else "unfused")
                        key = (q, zs, tag, cap, nrm)
                        tiles[key] = np.asarray(jax.device_get(W_tile))
                        _log(f"{str(q):>9} {zs:>13} {tag:>16} {cap:5d} "
                             f"{nrm:>5} {dt:9.3f} {it.mean():8.2f} "
                             f"{it.max():7d} "
                             f"{1e3*dt/max(float(it.sum()),1.0):10.3f} "
                             f"{rs.max():15.3e}")
                if args.lift and fv == fuses[-1] and ncol == n_pad:
                    stack = wlp.build_precond_stack(mesh, data, include_w=True)
                    diag_h0 = wlp.lifted_precond_diagonal(
                        dq, mesh, stack, include_w=True)
                    for rep in range(2):
                        t0 = time.perf_counter()
                        out = wlp.apply_lifted_resolvent_block(
                            G, z, dq, stack, diag_h0,
                            max_iter=int(args.max_iters.split(",")[0]),
                            tol=float(args.tol))
                        jax.block_until_ready(out)
                        dt = time.perf_counter() - t0
                    T_tile, rr, it2 = out
                    # CLOSE IT: the lift's tile is T = Pi v, and the object
                    # the oracle certifies is W - v.  Comparing T against a
                    # W oracle would be a category error, so the Dyson close
                    # runs here, exactly as the facade runs it.
                    W_lift = wlp.dyson_close_tile(
                        T_tile, dq["V_q0"],
                        allow_replicated_solve=bool(args.allow_replicated_dyson))
                    tiles[(q, zs, "lift", int(args.max_iters.split(",")[0]),
                           args.resid_norms.split(",")[0])] = np.asarray(
                        jax.device_get(W_lift))
                    it2 = np.asarray(jax.device_get(it2))[:ncol]
                    rr = np.asarray(jax.device_get(rr))[:ncol]
                    _log(f"{str(q):>9} {zs:>13} {'lift(routeA)':>16} "
                         f"{int(args.max_iters.split(',')[0]):5d} "
                         f"{args.resid_norms.split(',')[0]:>5} {dt:9.3f} "
                         f"{it2.mean():8.2f} {it2.max():7d} "
                         f"{1e3*dt/max(float(it2.sum()),1.0):10.3f} "
                         f"{rr.max():15.3e}")
    _log("-" * len(hdr))
    rc = 0
    if oracles:
        # ACCURACY GATE: every measured arm against the tight-tol oracle
        # solved on the SAME operator at the same (q, z) — not against another
        # arm, so a shared systematic cannot pass the cell.
        _log("[gate] arms vs the 1e-12 same-operator oracle "
             f"(ceiling {args.gate_rel:g}), per-column relative:")
        for (q, zs, tag, cap, nrm), got in sorted(tiles.items()):
            ref = oracles.get((q, zs))
            if ref is None:
                continue
            den = np.linalg.norm(ref[:, :ncol], axis=0)
            den = np.where(den == 0.0, 1.0, den)
            rel = float(np.max(np.linalg.norm(
                got[:, :ncol] - ref[:, :ncol], axis=0) / den))
            ok = rel <= float(args.gate_rel)
            rc |= 0 if ok else 1
            _log(f"  q={q} z={zs:>13} {tag:>12} cap={cap} norm={nrm}: "
                 f"rel {rel:.3e}  {'PASS' if ok else 'FAIL'}")
    keys = sorted(tiles)
    if len(keys) >= 2:
        ref = tiles[keys[0]]
        den = max(float(np.max(np.abs(ref))), 1e-300)
        for k in keys[1:]:
            rel = float(np.max(np.abs(tiles[k] - ref)) / den)
            _log(f"[solve] tile {k} vs {keys[0]}: rel {rel:.3e}")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--mode", default="core",
                    choices=["core", "matvec", "solve"])
    ap.add_argument("--baseline-src", default=None,
                    help="src/ of a PRISTINE tree copy, for the core A/B")
    ap.add_argument("--core-n", type=int, default=256)
    ap.add_argument("--core-z", default="0.0")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--max-iter", type=int, default=200)
    # --mode solve
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--input", default="gnppm_test.in")
    ap.add_argument("--mesh", default=None,
                    help="px,py.  Omitted = the run's canonical square mesh "
                         "(bse_ring_comm.create_mesh_xy_from_flags); a given "
                         "shape must BE that mesh.")
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--q-index", default="0")
    ap.add_argument("--zs", default="0")
    ap.add_argument("--max-iters", default="200")
    ap.add_argument("--resid-norms", default="-")
    ap.add_argument("--lift", type=int, default=0)
    ap.add_argument("--oracle-tol", type=float, default=0.0,
                    help="tight-tol oracle on the same operator; 0 disables")
    ap.add_argument("--oracle-max-iter", type=int, default=300)
    ap.add_argument("--gate-rel", type=float, default=1e-8)
    ap.add_argument("--allow-replicated-dyson", type=int, default=0)
    ap.add_argument("--include-w", type=int, default=1)
    ap.add_argument("--batch-widths", default="1,2,4,8,16")
    ap.add_argument("--matvec-reps", type=int, default=20)
    ap.add_argument("--fuse", default="1",
                    help="rung fusion: 0, 1, or a comma list for the A/B")
    ap.add_argument("--fuse-ab", type=int, default=1)
    args = ap.parse_args(argv)

    if args.mode == "core":
        return mode_core(args)
    if not args.run_dir:
        ap.error(f"--mode {args.mode} needs --run-dir")
    if args.mode == "matvec":
        return mode_matvec(args)
    return mode_solve(args)


if __name__ == "__main__":
    raise SystemExit(main())
