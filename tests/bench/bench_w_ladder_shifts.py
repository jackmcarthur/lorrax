#!/usr/bin/env python3
"""Bake-off driver for shifted-system Krylov sharing on the ladder-W resolvent.

Three arms on ONE operator, ONE payload and ONE probe-block source, so the
only thing that differs between them is stage 2 of
``bse_w_exact.apply_screening_resolvent_block``:

  ``baseline``   the production path — ``apply_screening_resolvent_block``
                 called once per (q, chunk, z), preconditioned GMRES per z.
  ``shifted``    ``w_ladder_shifts.solve_shifted_block`` — ONE unpreconditioned
                 Arnoldi space per column, every z solved in it.
  ``chained``    ``w_ladder_shifts.solve_chained_block`` — production engine and
                 production preconditioner, z ordered by proximity, each shift
                 started from the previous shift's iterate.
  ``hybrid``     ``w_ladder_shifts.solve_hybrid_block`` — a SHORT shared space
                 (which finishes the far-line shifts outright) followed by a
                 preconditioned polish on each remaining shift's correction
                 equation.

Modes
-----
``calibrate``  per-shift residual of the SHARED space vs Arnoldi dimension, on
               a handful of columns.  One run of ``--arnoldi-dim`` matvecs
               gives the whole curve for every z, which is how ``arnoldi_dim``
               gets chosen by measurement rather than by guess.
``gate``       correctness: every arm vs a tight-tol (default 1e-12) per-z
               oracle built with the SAME operator, on ``W(z) - v`` tiles.
``bench``      wall seconds + matvec counts for the requested arms/workloads.

Matvec accounting (the honest cross-track metric), from the source:
  baseline  per (column, z): 1 (r0) + k (iterations) + 1 (true residual) = k+2
  shifted   per column:      m (Arnoldi) + nz (true residuals)
  chained   per (column, z): k + 2 for the first link, k + 3 after (the extra
            one opens the correction equation)
  hybrid    per column:      m + sum_z (k + 2); the correction equation's
            opening residual is free (the shared solve already built it)

Run (Perlmutter, from the sandbox):
  lx run -G 1 -n 1 bash -lc 'source <prelude>; cd $LXA && \
      python3 tests/bench/bench_w_ladder_shifts.py --mode bench --out <dir>'

Not collected by pytest (``tests/bench`` is in ``norecursedirs``); argv-driven
per ``docs/architecture/layers.md`` section 5.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "tests"))

import harness                                                   # noqa: E402

from runtime import bootstrap                                    # noqa: E402
bootstrap()

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P                # noqa: E402

jax.config.update("jax_enable_x64", True)

from bse import bse_io                                           # noqa: E402
from bse.bse_w_exact import (                                    # noqa: E402
    _symmetry_tables, apply_screening_resolvent_block, build_finite_q_data,
)
from bse.bse_feast import build_preconditioner_diagonal_sharded  # noqa: E402
from bse.w_ladder import build_ladder_resolvent                  # noqa: E402
from bse import w_ladder_shifts as wls                           # noqa: E402
from common.collectives import gather_to_host                    # noqa: E402


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
def prepare_run_dir(dest: Path, case="gnppm_debug", deck="gnppm_test.in"):
    """Produce (or reuse) a gnppm_debug run dir carrying the W0 restart.

    The ladder kernel IS the RPA W(0) the ordinary driver writes back into the
    restart (``gw_output.persist_w0_and_head``), so the fixture must be RUN,
    not merely copied.  Reused across invocations when it is already there —
    a driver run is ~2 min and the bench arms are the thing being measured.
    """
    dest = Path(dest)
    input_path = dest / deck
    if input_path.exists() and (dest / "tmp").is_dir():
        return dest, str(input_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    harness.copy_fixture(harness.REG / case, dest)
    t0 = time.perf_counter()
    res = harness.run_gw_jax(dest, deck, timeout=3600)
    if res.returncode != 0:
        raise SystemExit(
            f"fixture driver run failed (rc={res.returncode})\n"
            f"--- stdout tail ---\n{res.stdout[-4000:]}\n"
            f"--- stderr tail ---\n{res.stderr[-4000:]}")
    print(f"[fixture] gnppm driver run: {time.perf_counter()-t0:.1f} s -> {dest}",
          flush=True)
    return dest, str(input_path)


def load_payload(input_path, mesh):
    restart = bse_io._find_restart_file(input_path)
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=input_path, inject_head=False, load_v_full=True)
    return data


def probe_blocks(n_pad, nlog, chunk, n_chunks):
    """``n_chunks`` FULL chunks of the identity — no zero-pad rows.

    Pad rows would be solved (a zero rhs exits at the first check block) and
    would pollute both the wall time and the matvec count with work that is not
    part of either algorithm.  The bench therefore never emits a short chunk;
    the production facade's short final chunk is a separate, already-measured
    concern.
    """
    eye = np.eye(n_pad, dtype=np.float64)
    out = []
    for i in range(n_chunks):
        c0 = i * chunk
        if c0 + chunk > nlog:
            break
        out.append((c0, chunk, eye[c0:c0 + chunk, :]))
    if not out:
        raise SystemExit(f"probe chunk {chunk} does not fit in nlog={nlog}")
    return out


# ---------------------------------------------------------------------------
# z plans
# ---------------------------------------------------------------------------
def z_plans(data, n_mpa_pairs=4):
    """``{name: (z_list_ry, label)}`` — the two protocol workloads.

    The MPA-shaped list is the code's OWN sample plan
    (``gw.mpa.sampling.double_parallel_grid``, insulator, alpha=1), evaluated
    in Rydberg at this payload's largest transition energy, so the shifts are
    the ones MPA would actually ask for: ``2*n_p`` points on two lines parallel
    to the real axis at heights 0.2 and 2 Ry, with the near line's first sample
    exactly at the origin.
    """
    from gw.mpa.sampling import double_parallel_grid
    eps_c = np.asarray(jax.device_get(data["eps_c"]))
    eps_v = np.asarray(jax.device_get(data["eps_v"]))
    omega_m = float(eps_c.max() - eps_v.min())
    grid = double_parallel_grid(n_mpa_pairs, omega_m, material_class="insulator",
                                alpha=1, energy_unit="Ry")
    return {
        "z1": (np.asarray([0.0 + 0.0j]), "z = {0}"),
        "mpa8": (np.asarray(grid, dtype=np.complex128),
                 f"double_parallel_grid(n_p={n_mpa_pairs}, "
                 f"omega_m={omega_m:.4f} Ry)"),
        # The correctness gate's three: the static point, a point on the
        # imaginary axis (the gn_ppm shape) and a damped complex point (the
        # MPA near line).  They differ in KIND, which is what a correctness
        # gate wants; the timing workloads above differ in COUNT.
        "gate3": (np.asarray([0.0 + 0.0j, 0.0 + 0.5j,
                              0.25 * omega_m + 0.2j], dtype=np.complex128),
                  "z = {0, 0.5i, 0.25*omega_m + 0.2i} Ry"),
    }


def order_by_proximity(z_list):
    """Nearest-neighbour walk from the shift closest to the origin.

    The chained arm is only as good as consecutive shifts are close; MPA's own
    ordering is near-line-then-far-line, which jumps 1.8 Ry at the seam.
    """
    z = np.asarray(z_list, dtype=np.complex128)
    remaining = list(range(z.size))
    order = [int(np.argmin(np.abs(z)))]
    remaining.remove(order[0])
    while remaining:
        last = z[order[-1]]
        nxt = min(remaining, key=lambda i: abs(z[i] - last))
        order.append(nxt)
        remaining.remove(nxt)
    return np.asarray(order, dtype=int)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def _tile(x):
    return np.asarray(gather_to_host(x))


def run_baseline(dq, blocks, z_list, stack, *, tol, max_iter, mesh, include_w):
    matvec, _, gen, snapshot, sh = stack
    diag_hq = build_preconditioner_diagonal_sharded(
        dq, mesh, include_W=include_w, use_tda=False)
    out, mv, resid = {}, 0, {}
    for c0, n_real, G in blocks:
        for iz, z in enumerate(z_list):
            W, r, it = apply_screening_resolvent_block(
                G, complex(z), dq, matvec, diag_hq, gen, snapshot, sh,
                max_iter=max_iter, tol=tol, return_iters=True)
            jax.block_until_ready(W)
            it_h = np.asarray(gather_to_host(it))
            mv += int((it_h + 2).sum())
            out[(iz, c0)] = _tile(W)[:, :n_real]
            resid[(iz, c0)] = np.asarray(gather_to_host(r))[:n_real]
    return out, mv, resid, diag_hq


def run_shifted(dq, blocks, z_list, stack, *, tol, arnoldi_dim, check_every):
    matvec, _, gen, snapshot, sh = stack
    nz = len(z_list)
    out, mv, resid, steps_all = {}, 0, {}, []
    for c0, n_real, G in blocks:
        tiles, _proj, true, steps = wls.solve_shifted_block(
            G, z_list, dq, matvec, gen, snapshot, sh,
            arnoldi_dim=arnoldi_dim, tol=tol, check_every=check_every)
        jax.block_until_ready(tiles)
        st = np.asarray(gather_to_host(steps))
        tr = np.asarray(gather_to_host(true))
        steps_all.append(st)
        mv += int((st + nz).sum())
        for iz in range(nz):
            out[(iz, c0)] = _tile(tiles[iz])[:, :n_real]
            resid[(iz, c0)] = tr[iz, :n_real]
    return out, mv, resid, np.concatenate(steps_all)


def run_chained(dq, blocks, z_list, stack, *, tol, max_iter, mesh, include_w):
    matvec, _, gen, snapshot, sh = stack
    diag_hq = build_preconditioner_diagonal_sharded(
        dq, mesh, include_W=include_w, use_tda=False)
    nz = len(z_list)
    out, mv, resid = {}, 0, {}
    for c0, n_real, G in blocks:
        tiles, true, iters = wls.solve_chained_block(
            G, z_list, dq, matvec, diag_hq, gen, snapshot, sh,
            max_iter=max_iter, tol=tol)
        jax.block_until_ready(tiles)
        it = np.asarray(gather_to_host(iters))
        tr = np.asarray(gather_to_host(true))
        # link 0 opens on x=0 (no matvec, static branch); links 1.. pay one.
        mv += int((it + 2).sum() + it[1:, :].size)
        for iz in range(nz):
            out[(iz, c0)] = _tile(tiles[iz])[:, :n_real]
            resid[(iz, c0)] = tr[iz, :n_real]
    return out, mv, resid, it


def run_hybrid(dq, blocks, z_list, stack, *, tol, max_iter, arnoldi_dim,
               check_every, mesh, include_w):
    matvec, _, gen, snapshot, sh = stack
    diag_hq = build_preconditioner_diagonal_sharded(
        dq, mesh, include_W=include_w, use_tda=False)
    nz = len(z_list)
    out, mv, resid, shared = {}, 0, {}, None
    for c0, n_real, G in blocks:
        tiles, true, iters, sh_res = wls.solve_hybrid_block(
            G, z_list, dq, matvec, diag_hq, gen, snapshot, sh,
            arnoldi_dim=arnoldi_dim, max_iter=max_iter, tol=tol,
            check_every=check_every)
        jax.block_until_ready(tiles)
        it = np.asarray(gather_to_host(iters))
        tr = np.asarray(gather_to_host(true))
        shared = np.asarray(gather_to_host(sh_res))
        mv += int(arnoldi_dim * n_real + (it + 2).sum())
        for iz in range(nz):
            out[(iz, c0)] = _tile(tiles[iz])[:, :n_real]
            resid[(iz, c0)] = tr[iz, :n_real]
    return out, mv, resid, (it, shared)


def rel_err(a, b):
    den = np.linalg.norm(b)
    if den == 0.0:
        return float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / den)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def mode_calibrate(args, data, mesh, q_list, stack, plans):
    matvec, _, gen, snapshot, sh = stack
    z_list, label = plans[args.workload]
    print(f"[calibrate] {label}; z = {np.array2string(z_list, precision=4)}",
          flush=True)
    rows = []
    for iq in args.q_index:
        q = tuple(int(v) for v in q_list[iq])
        dq = build_finite_q_data(data, q, mesh)
        n_pad = int(dq["V_q0"].shape[0])
        py = mesh.devices.shape[1]
        cols = list(args.calib_cols)
        n_probe = int(np.ceil(len(cols) / py) * py)
        G = np.zeros((n_probe, n_pad))
        for i, col in enumerate(cols):
            G[i, int(col)] = 1.0
        rhs = wls.seed_probe_block(G, dq, gen, sh)
        ops = tuple(wls.matvec_operands(dq))
        for i, col in enumerate(cols):
            b_col = rhs[:, i][:, None]
            t0 = time.perf_counter()
            dims, hist = wls.shared_space_history(
                matvec, b_col, z_list, ops, m=args.arnoldi_dim,
                stride=args.stride)
            dt = time.perf_counter() - t0
            print(f"\n[calibrate] q={q} column={col}  ({dt:.1f} s for "
                  f"{args.arnoldi_dim} matvecs)", flush=True)
            hdr = "  m   " + "".join(f"  z{j:<10d}" for j in range(len(z_list)))
            print(hdr)
            for d, h in zip(dims, hist):
                print(f"{d:5d}  " + "".join(f"{v:12.3e}" for v in h))
            rows.append(dict(q=list(q), column=int(col),
                             dims=[int(d) for d in dims],
                             resid=[[float(v) for v in h] for h in hist],
                             seconds=dt))
    return dict(mode="calibrate", z=[str(z) for z in z_list], rows=rows)


def mode_gate(args, data, mesh, q_list, stack, plans, include_w):
    z_list, label = plans[args.workload]
    nz = len(z_list)
    print(f"[gate] oracle tol={args.oracle_tol:g}; {label}", flush=True)
    rows = []
    # A NARROWER probe block than the timing workloads use: this cell answers
    # "does the tile agree with the oracle", which every column answers
    # independently, so paying 64 columns of a 600-step Arnoldi to hear the
    # same answer eight times is not more evidence.
    blocks = probe_blocks(int(data["V_q0"].shape[0]), int(data["n_rmu"]),
                          args.gate_chunk, 1)
    for iq in args.q_index:
        q = tuple(int(v) for v in q_list[iq])
        dq = build_finite_q_data(data, q, mesh)
        oracle, _, ores, _ = run_baseline(
            dq, blocks, z_list, stack, tol=args.oracle_tol,
            max_iter=args.oracle_max_iter, mesh=mesh, include_w=include_w)
        arms = {}
        arms["baseline"] = run_baseline(
            dq, blocks, z_list, stack, tol=args.tol, max_iter=args.max_iter,
            mesh=mesh, include_w=include_w)[:3]
        arms["shifted"] = run_shifted(
            dq, blocks, z_list, stack, tol=args.tol,
            arnoldi_dim=args.arnoldi_dim, check_every=args.check_every)[:3]
        if "hybrid" in args.arms:
            arms["hybrid"] = run_hybrid(
                dq, blocks, z_list, stack, tol=args.tol,
                max_iter=args.max_iter, arnoldi_dim=args.hybrid_dim,
                check_every=args.hybrid_dim, mesh=mesh,
                include_w=include_w)[:3]
        if "chained" in args.arms:
            zo = order_by_proximity(z_list)
            t = run_chained(dq, blocks, z_list[zo], stack, tol=args.tol,
                            max_iter=args.max_iter, mesh=mesh,
                            include_w=include_w)
            # t's z index runs over the PROXIMITY order; zo[i] is the original
            # index of the i-th ordered shift, so zo maps ordered -> original.
            arms["chained"] = (
                {(int(zo[iz]), c0): v for (iz, c0), v in t[0].items()},
                t[1],
                {(int(zo[iz]), c0): v for (iz, c0), v in t[2].items()})
        omax = max(float(np.abs(v).max()) for v in ores.values())
        print(f"\n[gate] q={q}  oracle max per-column resid = {omax:.3e}")
        for name, (tiles, _mv, res) in arms.items():
            worst = 0.0
            for key, tile in tiles.items():
                worst = max(worst, rel_err(tile, oracle[key]))
            rmax = max(float(np.abs(v).max()) for v in res.values())
            verdict = ("PASS" if (worst <= args.gate_tol and rmax <= args.tol)
                       else "FAIL")
            print(f"  {name:9s} max rel err vs oracle = {worst:.3e}   "
                  f"max TRUE per-shift resid = {rmax:.3e}   [{verdict}]",
                  flush=True)
            per_z = {}
            for iz in range(nz):
                e = max(rel_err(tiles[k], oracle[k])
                        for k in tiles if k[0] == iz)
                r = max(float(np.abs(res[k]).max()) for k in res if k[0] == iz)
                per_z[str(z_list[iz])] = dict(rel_err=e, true_resid=r)
                print(f"      z={z_list[iz]!s:>28s}  rel={e:.3e}  "
                      f"resid={r:.3e}")
            rows.append(dict(q=list(q), arm=name, max_rel_err=worst,
                             max_true_resid=rmax, verdict=verdict,
                             per_z=per_z))
    return dict(mode="gate", oracle_tol=args.oracle_tol,
                gate_tol=args.gate_tol, rows=rows)


def mode_bench(args, data, mesh, q_list, stack, plans, include_w):
    n_pad = int(data["V_q0"].shape[0])
    blocks = probe_blocks(n_pad, int(data["n_rmu"]), args.probe_chunk,
                          args.n_chunks)
    qs = [tuple(int(v) for v in q_list[i]) for i in args.q_index]
    # Build the per-q payloads ONCE, outside every arm: build_finite_q_data is
    # a host roll of psi/eps and is identical work for all three arms, so
    # leaving it inside the timed region would dilute the ratio with a cost
    # that is not part of any solver.
    dqs = [build_finite_q_data(data, q, mesh) for q in qs]
    rows = []
    for wl in args.workloads:
        z_list, label = plans[wl]
        nz = len(z_list)
        zo = order_by_proximity(z_list)
        print(f"\n=== workload {wl}: nz={nz}  {label}", flush=True)
        for name in args.arms:
            # WARM-UP on the first q: the first (q, chunk, z) point carries the
            # compile, and a compile inside the timed region measures the
            # compiler.  Timed sweep is dispatch-only after this.
            warm = [(blocks[0][0], blocks[0][1], blocks[0][2])]
            _dispatch_arm(name, dqs[0], warm, z_list, zo, stack, args, mesh,
                          include_w)
            t0 = time.perf_counter()
            mv_tot, worst_resid, steps_stat = 0, 0.0, []
            for dq in dqs:
                mv, res, extra = _dispatch_arm(
                    name, dq, blocks, z_list, zo, stack, args, mesh, include_w)
                mv_tot += mv
                worst_resid = max(worst_resid,
                                  max(float(np.abs(v).max())
                                      for v in res.values()))
                if extra is not None:
                    steps_stat.append(extra)
            dt = time.perf_counter() - t0
            ncol = len(qs) * len(blocks) * args.probe_chunk
            extra_s = ""
            if steps_stat:
                s = np.concatenate([np.ravel(x) for x in steps_stat])
                extra_s = (f"  iters/steps min/med/max = {s.min()}/"
                           f"{int(np.median(s))}/{s.max()}")
            print(f"  {name:9s} {dt:9.2f} s   matvecs={mv_tot:9d}   "
                  f"mv/col/z={mv_tot/(ncol*nz):7.2f}   "
                  f"max resid={worst_resid:.2e}{extra_s}", flush=True)
            rows.append(dict(workload=wl, arm=name, seconds=dt,
                             matvecs=mv_tot, columns=ncol, nz=nz,
                             mv_per_col_per_z=mv_tot / (ncol * nz),
                             max_true_resid=worst_resid))
    return dict(mode="bench", rows=rows,
                q=[list(q) for q in qs], probe_chunk=args.probe_chunk,
                n_chunks=args.n_chunks, arnoldi_dim=args.arnoldi_dim)


def gemm_shape_table(data, widths):
    """The matvec's dominant contractions as (M, K, N, batch) vs block width.

    Printed next to the matvec timings because "deep blocks turn this into big
    GEMMs" is a claim about SHAPES, and the shapes say which contractions can
    and cannot benefit.  ``b`` is the matvec's batch axis — the number of
    right-hand vectors pushed through one call, which for the production path
    is 1 (``apply_screening_resolvent_block`` scans probe columns one at a
    time) and for a deep block would be the block width.

    Everything below is read off ``bse_ring_comm``'s einsum subscripts; no
    measurement is involved, which is the point — these are the shapes the
    timings in the table above are explained by.
    """
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    nmu = int(data["V_q0"].shape[0])
    ns = int(data["psi_c_X"].shape[2])
    nc = int(data["psi_c_X"].shape[1])
    nv = int(data["psi_v_X"].shape[1])
    print("\n[gemm] payload: n_mu=%d nc=%d nv=%d nk=%d nspinor=%d"
          % (nmu, nc, nv, nk, ns))
    print("[gemm] dominant contractions, (M x K x N) x batch, vs block width b")
    hdr = f"  {'contraction':46s} {'M':>8s} {'K':>8s} {'N':>8s} {'batch':>7s}"
    for b in widths:
        print(f"\n  --- b = {b} ---")
        print(hdr)
        rows = [
            # ring dyad: the ONLY N_mu x N_mu GEMM in the operator, and at b=1
            # it is a GEMV (N = b).  This is the contraction deep blocks help.
            ("apply_V_ring  V_q0 . S   'MN,bNk->bMk'", nmu, nmu, b, nk),
            ("apply_V_ring  seed       'kcvN,bcvk->bNk'", nmu, nc * nv, b, nk),
            ("apply_V_ring  readout    'kcvM,bMk->bcvk'", nc * nv, nmu, b, nk),
            # direct rung encode/decode: b enters as an OUTPUT axis, so these
            # do grow with depth, but their K stays nc / nv (small).
            ("rung encode-v 'kvsN,bcvk->bcksN'", b * nc, nv, ns * nmu, nk),
            ("rung encode-c 'kctM,bcksN->bMNtsk'", ns * nmu, nc, b * ns * nmu, nk),
            ("rung decode-c 'kctM,bMNtsk->bcNsk'", nc, ns * nmu, b * nmu * ns, nk),
            ("rung decode-v 'kvsN,bcNsk->bcvk'", nv, ns * nmu, b * nc, nk),
        ]
        for name, m, k, n, bat in rows:
            print(f"  {name:46s} {m:8d} {k:8d} {n:8d} {bat:7d}")
        # The staging constraint: T is the W_R-class intermediate and it
        # carries b as a LEADING axis (bse_ring_comm._ring_sum_conduction's
        # `T0 = zeros((R.shape[0], mu_local, nu_local, ns, ns, nk))`).
        t_bytes = b * nmu * nmu * ns * ns * nk * 16
        print(f"  {'>> T (W_R-class rung intermediate)':46s} "
              f"{b} x mu x nu x ns^2 x nk = {t_bytes/2**30:.3f} GiB "
              f"(1x1 mesh; divide by P on a px x py mesh)")


def mode_matvec_tail(args, data):
    gemm_shape_table(data, args.batch_widths)


def mode_matvec(args, data, mesh, q_list, stack):
    """Cost of ONE ring matvec vs its BATCH width — the denominator of every
    matvec count in this bake-off.

    Worth its own mode because it decides how to read the whole table.  The
    production solve scans probe columns one at a time (batch 1); if a batch of
    64 costs about what a batch of 1 costs, the sweep is launch-latency bound
    and the honest cross-track metric (matvec COUNT) and the thing the user
    waits for (wall seconds) come apart — a track that halves the count can
    still lose to one that batches.
    """
    matvec, _, gen, snapshot, sh = stack
    q = tuple(int(v) for v in q_list[args.q_index[0]])
    dq = build_finite_q_data(data, q, mesh)
    ops = tuple(wls.matvec_operands(dq))
    n_pad = int(dq["V_q0"].shape[0])
    rows = []
    for width in args.batch_widths:
        G = np.eye(n_pad)[:width, :]
        rhs = wls.seed_probe_block(G, dq, gen, sh)

        @jax.jit
        def _rep(x):
            def body(_, y):
                return matvec(y, *ops)
            return jax.lax.fori_loop(0, args.matvec_reps, body, x)

        jax.block_until_ready(_rep(rhs))          # compile
        t0 = time.perf_counter()
        jax.block_until_ready(_rep(rhs))
        dt = (time.perf_counter() - t0) / args.matvec_reps
        rows.append(dict(batch=width, seconds_per_matvec=dt,
                         seconds_per_column=dt / width))
        print(f"  batch={width:4d}  {dt*1e3:9.3f} ms/matvec   "
              f"{dt/width*1e3:9.4f} ms/column", flush=True)

    # SEED and PROJECT, timed on their own: the seed is the z-INDEPENDENT
    # stage the production path re-dispatches once per z, so its cost times
    # (nz - 1) is what the hoist returns per (q, chunk) for free.
    width = args.batch_widths[-1]
    G = np.eye(n_pad)[:width, :]
    rhs = wls.seed_probe_block(G, dq, gen, sh)
    jax.block_until_ready(rhs)
    t0 = time.perf_counter()
    for _ in range(args.matvec_reps):
        rhs = wls.seed_probe_block(G, dq, gen, sh)
    jax.block_until_ready(rhs)
    t_seed = (time.perf_counter() - t0) / args.matvec_reps
    s = jnp.zeros(rhs.shape[1:], dtype=rhs.dtype)
    jax.block_until_ready(
        snapshot(s, dq["psi_c_Y"], dq["psi_v_Y"], dq["V_q0"]))
    t0 = time.perf_counter()
    for _ in range(args.matvec_reps):
        out = snapshot(s, dq["psi_c_Y"], dq["psi_v_Y"], dq["V_q0"])
    jax.block_until_ready(out)
    t_proj = (time.perf_counter() - t0) / args.matvec_reps
    print(f"  seed (n_probe={width})    {t_seed*1e3:9.3f} ms  "
          f"= {t_seed/(rows[0]['seconds_per_matvec']):.2f} matvec-equivalents",
          flush=True)
    print(f"  project (n_probe={width}) {t_proj*1e3:9.3f} ms", flush=True)
    gemm_shape_table(dq, args.batch_widths)
    return dict(mode="matvec", q=list(q), reps=args.matvec_reps, rows=rows,
                seed_seconds=t_seed, project_seconds=t_proj,
                seed_probe_width=width, include_w=bool(args.include_w))


def _dispatch_arm(name, dq, blocks, z_list, zo, stack, args, mesh, include_w):
    if name == "baseline":
        _t, mv, res, _ = run_baseline(
            dq, blocks, z_list, stack, tol=args.tol, max_iter=args.max_iter,
            mesh=mesh, include_w=include_w)
        return mv, res, None
    if name == "shifted":
        _t, mv, res, st = run_shifted(
            dq, blocks, z_list, stack, tol=args.tol,
            arnoldi_dim=args.arnoldi_dim, check_every=args.check_every)
        return mv, res, st
    if name == "chained":
        _t, mv, res, it = run_chained(
            dq, blocks, z_list[zo], stack, tol=args.tol,
            max_iter=args.max_iter, mesh=mesh, include_w=include_w)
        return mv, res, it
    if name == "hybrid":
        _t, mv, res, (it, _sh) = run_hybrid(
            dq, blocks, z_list, stack, tol=args.tol, max_iter=args.max_iter,
            arnoldi_dim=args.hybrid_dim, check_every=args.hybrid_dim,
            mesh=mesh, include_w=include_w)
        return mv, res, it
    raise SystemExit(f"unknown arm {name!r}")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="bench",
                    choices=["calibrate", "gate", "bench", "matvec"])
    ap.add_argument("--run-dir", default=None,
                    help="gnppm_debug run dir (created + driven if absent)")
    ap.add_argument("--out", default=None, help="evidence dir for the JSON")
    ap.add_argument("--arms", default="baseline,shifted")
    ap.add_argument("--workloads", default="z1,mpa8")
    ap.add_argument("--workload", default="mpa8",
                    help="single workload for calibrate/gate")
    ap.add_argument("--q-index", default="0,1")
    ap.add_argument("--probe-chunk", type=int, default=64)
    ap.add_argument("--gate-chunk", type=int, default=8)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--arnoldi-dim", type=int, default=200)
    ap.add_argument("--check-every", type=int, default=25)
    ap.add_argument("--hybrid-dim", type=int, default=50,
                    help="shared-space size of the hybrid arm (one block)")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--calib-cols", default="0,1,2,3")
    ap.add_argument("--oracle-tol", type=float, default=1e-12)
    ap.add_argument("--oracle-max-iter", type=int, default=400)
    ap.add_argument("--gate-tol", type=float, default=1e-8)
    ap.add_argument("--batch-widths", default="1,2,4,8,16,32,64")
    ap.add_argument("--matvec-reps", type=int, default=20)
    ap.add_argument("--include-w", type=int, default=1)
    ap.add_argument("--mpa-pairs", type=int, default=4)
    args = ap.parse_args(argv)
    args.arms = [s for s in args.arms.split(",") if s]
    args.workloads = [s for s in args.workloads.split(",") if s]
    args.q_index = [int(s) for s in args.q_index.split(",") if s]
    args.calib_cols = [int(s) for s in args.calib_cols.split(",") if s]
    args.batch_widths = [int(s) for s in args.batch_widths.split(",") if s]

    scratch = os.environ.get("SCRATCH", "/tmp")
    run_dir = Path(args.run_dir or
                   f"{scratch}/bench_w_ladder_shifts/gnppm_run")
    run_dir, input_path = prepare_run_dir(run_dir)

    devs = jax.devices()
    mesh = Mesh(np.array(devs[:1]).reshape(1, 1), axis_names=("x", "y"))
    print(f"[env] jax {jax.__version__}  devices={devs}  mesh=1x1", flush=True)

    data = load_payload(input_path, mesh)
    include_w = bool(args.include_w)
    print(f"[payload] n_rmu={int(data['n_rmu'])} "
          f"mu_pad={int(data['V_q0'].shape[0])} "
          f"nc={int(data['eps_c'].shape[1])} nv={int(data['eps_v'].shape[1])} "
          f"nk={int(data['nkx']*data['nky']*data['nkz'])} "
          f"nspinor={int(data['psi_c_X'].shape[2])} include_w={include_w}",
          flush=True)

    # ONE operator object for every arm — the block-solver caches key on
    # id(matvec), so rebuilding per arm would cost a compile per arm AND stop
    # the arms from being a comparison of stage 2 alone.
    stack = build_ladder_resolvent(mesh, data, include_w=include_w,
                                   vertex_flipped=True)
    sym = _symmetry_tables(input_path)
    q_list = np.asarray(sym.q_irr_kgrid_int, dtype=int)
    print(f"[wedge] {len(q_list)} irreducible q: {q_list.tolist()}", flush=True)

    plans = z_plans(data, n_mpa_pairs=args.mpa_pairs)
    if args.mode == "matvec":
        result = mode_matvec(args, data, mesh, q_list, stack)
    elif args.mode == "calibrate":
        result = mode_calibrate(args, data, mesh, q_list, stack, plans)
    elif args.mode == "gate":
        result = mode_gate(args, data, mesh, q_list, stack, plans, include_w)
    else:
        result = mode_bench(args, data, mesh, q_list, stack, plans, include_w)

    result["argv"] = sys.argv[1:]
    result["run_dir"] = str(run_dir)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        path = out / f"bench_w_ladder_shifts_{args.mode}_{stamp}.json"
        path.write_text(json.dumps(result, indent=1))
        print(f"\n[out] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
