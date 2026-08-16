"""Kernel-level profiling driver for the ladder-W direct rung (track O9).

WHAT THIS IS FOR, and why it is not one of the existing benches.  The three
in-tree ladder benches (``bench_w_ladder_precond``, ``_shifts``,
``_integration``) all measure WALL TIME of an algorithmic arm.  This one exists
one layer down: it produces a process whose GPU work is *only* the ladder
matvec, in a shape and a repeat structure that Nsight Systems (timeline, kernel
mix, launch gaps) and Nsight Compute (per-kernel DRAM traffic and occupancy) can
read cleanly.  So it deliberately does three things the other benches do not:

* **it warms up properly and then repeats**, reporting EVERY rep rather than a
  mean, because the published ``nb=1`` matvec cost (16.10 ms/col against 8.56 at
  ``nb=16``, opt_precond/RESULTS.md §8) is exactly the shape a first-call
  outlier makes, and the same table already flags its own ``nb=1``/``nb=8`` RPA
  entries as first-call outliers.  A mean cannot tell those apart; a per-rep
  list can;
* **it brackets the timed region with ``cudaProfilerStart/Stop``**, so
  ``nsys profile --capture-range=cudaProfilerApi`` and
  ``ncu --profile-from-start off`` see steady-state kernels and no compile;
* **it can run the rung chain stage by stage** (``--mode rung``) with a device
  sync between stages, which is the only way to price ``encode / iFFT / W_R
  multiply / FFT / decode`` separately when XLA has fused them into one module.

The payload is synthetic (no restart, no deck) but carries the PRODUCTION
gnppm_debug geometry: ``n_rmu=399``, ``nk=3x3x1``, ``nc=20``, ``nv=26``,
``nspinor=2`` -- the shapes that make ``W_R`` a 21.9 MiB tile and the rung's
per-trial ``(mu, nu, s, s, k)`` buffer 87.5 MiB.  It is the ``nspinor``
generalisation of ``tests/test_bse_w_ladder_dense._synthetic_payload``; the
numerics are random and no physics claim is made from this file.

Usage (inside the container, one GPU):

    python tests/bench/bench_ladder_rung_kernels.py --mode matvec --widths 1,8
    python tests/bench/bench_ladder_rung_kernels.py --mode rung
    python tests/bench/bench_ladder_rung_kernels.py --mode solve --ncols 4
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2] / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from runtime import bootstrap  # noqa: E402
bootstrap()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

jax.config.update("jax_enable_x64", True)


# --- CUDA profiler / NVTX hooks (ctypes; no new dependency) -----------------

class _Prof:
    """cudaProfilerStart/Stop + NVTX push/pop, best-effort."""

    def __init__(self) -> None:
        self.rt = None
        self.nvtx = None
        for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
            try:
                self.rt = ctypes.CDLL(name)
                break
            except OSError:
                continue
        for name in ("libnvToolsExt.so.1", "libnvToolsExt.so"):
            try:
                self.nvtx = ctypes.CDLL(name)
                break
            except OSError:
                continue

    def start(self) -> None:
        if self.rt is not None:
            self.rt.cudaProfilerStart()

    def stop(self) -> None:
        if self.rt is not None:
            self.rt.cudaProfilerStop()

    def push(self, msg: str) -> None:
        if self.nvtx is not None:
            self.nvtx.nvtxRangePushA(ctypes.c_char_p(msg.encode()))

    def pop(self) -> None:
        if self.nvtx is not None:
            self.nvtx.nvtxRangePop()


PROF = _Prof()


# --- payload ----------------------------------------------------------------

def synthetic_payload(mesh, *, nkx, nky, nkz, nc, nv, nmu, ns, seed=7):
    """gnppm-shaped restart-free BSE payload in the production key contract.

    ``ns`` generalisation of ``tests/test_bse_w_ladder_dense._synthetic_payload``
    (which hard-codes ``nspinor = 1``).  Same physical constraints: ``V_q0`` real
    symmetric, ``W_q`` Hermitian per q with ``W(-q) = conj(W(q))``, well
    separated energies.
    """
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_serial import compute_pair_amplitude

    nk = nkx * nky * nkz
    rng = np.random.default_rng(seed)
    sh = make_bse_shardings(mesh)

    def _c(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / 8.0

    psi_c = _c(nk, nc, ns, nmu)
    psi_v = _c(nk, nv, ns, nmu)
    eps_c = 1.0 + 0.3 * rng.standard_normal((nk, nc))
    eps_v = -1.0 + 0.3 * rng.standard_normal((nk, nv))
    Vq0 = rng.standard_normal((nmu, nmu)) * 0.1
    Vq0 = Vq0 + Vq0.T
    Wq = _c(nmu, nmu, nk) * 0.1
    Wq = 0.5 * (Wq + np.conj(np.transpose(Wq, (1, 0, 2))))
    idx = np.ravel_multi_index(
        tuple((-np.array(np.unravel_index(np.arange(nk), (nkx, nky, nkz))))
              % np.array([[nkx], [nky], [nkz]])), (nkx, nky, nkz))
    Wq = 0.5 * (Wq + np.conj(Wq[:, :, idx]))
    Wq = Wq.reshape(nmu, nmu, nkx, nky, nkz)

    with mesh:
        d = {
            "psi_c_X": jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_x),
            "psi_c_Y": jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_y),
            "psi_v_X": jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_x),
            "psi_v_Y": jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_y),
            "eps_c": jnp.asarray(eps_c), "eps_v": jnp.asarray(eps_v),
            "W_q": jax.lax.with_sharding_constraint(jnp.asarray(Wq), sh.W),
            "V_q0": jax.lax.with_sharding_constraint(jnp.asarray(Vq0), sh.V),
            "nkx": nkx, "nky": nky, "nkz": nkz,
            "n_cond_pad": nc, "n_val_pad": nv, "n_rmu": nmu,
        }
        d["M_X"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(d["psi_c_X"], d["psi_v_X"]), sh.psi_x)
        d["M_Y"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(d["psi_c_Y"], d["psi_v_Y"]), sh.psi_y)
    return d


# --- geometry / roofline bookkeeping ---------------------------------------

def geometry(a) -> dict:
    nk = a.nkx * a.nky * a.nkz
    z = 16  # complex128 bytes
    chain_elems = a.nmu * a.nmu * a.ns * a.ns * nk
    return {
        "nk": nk, "npair": a.nc * a.nv * nk,
        "W_R_bytes": a.nmu * a.nmu * nk * z,
        "chain_bytes_per_trial": chain_elems * z,
        "psi_c_bytes": nk * a.nc * a.ns * a.nmu * z,
        "psi_v_bytes": nk * a.nv * a.ns * a.nmu * z,
        "X_bytes_per_trial": a.nc * a.nv * nk * z,
    }


# --- modes ------------------------------------------------------------------

def build_stack(mesh, data, fuse: bool):
    from bse.w_ladder import build_ladder_resolvent
    return build_ladder_resolvent(mesh, data, include_w=True,
                                  fuse_ladder_rung=fuse)


def _rand_block(mesh, sh, nb, nc, nv, nk, seed=11):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((2, nb, nc, nv, nk))
         + 1j * rng.standard_normal((2, nb, nc, nv, nk))) / 8.0
    with mesh:
        return jax.device_put(jnp.asarray(x), sh.X_full)


def mode_matvec(a, mesh, data) -> dict:
    from bse.bse_feast import ladder_matvec_operands
    matvec, diag_h, gen, snapshot, sh = build_stack(mesh, data, a.fuse)
    ops = ladder_matvec_operands(data)
    nk = a.nkx * a.nky * a.nkz
    out = {}
    for nb in [int(w) for w in a.widths.split(",")]:
        X = _rand_block(mesh, sh, nb, a.nc, a.nv, nk)
        # warm-up: compile + steady state
        for _ in range(a.warmup):
            y = matvec(X, *ops)
            y.block_until_ready()
        reps = []
        PROF.start()
        for i in range(a.reps):
            PROF.push("matvec_nb%d_rep%d" % (nb, i))
            t0 = time.perf_counter()
            y = matvec(X, *ops)
            y.block_until_ready()
            reps.append((time.perf_counter() - t0) * 1e3)
            PROF.pop()
        PROF.stop()
        out[nb] = reps
        print("[matvec] nb=%-3d reps_ms=%s  min=%.3f  per_col_min=%.3f"
              % (nb, " ".join("%.3f" % r for r in reps),
                 min(reps), min(reps) / nb), flush=True)
    return out


def mode_dispatch(a, mesh, data) -> dict:
    """Isolate PYTHON-DISPATCH cost from GPU cost.

    Same executable, two loops: one that syncs every call (what --mode matvec
    times) and one that enqueues `reps` calls and syncs ONCE at the end.  The
    difference is the per-call host round-trip; if the nb=1 penalty is dispatch
    overhead it must show up here.
    """
    from bse.bse_feast import ladder_matvec_operands
    matvec, _, _, _, sh = build_stack(mesh, data, a.fuse)
    ops = ladder_matvec_operands(data)
    nk = a.nkx * a.nky * a.nkz
    out = {}
    for nb in [int(w) for w in a.widths.split(",")]:
        X = _rand_block(mesh, sh, nb, a.nc, a.nv, nk)
        for _ in range(a.warmup):
            matvec(X, *ops).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            matvec(X, *ops).block_until_ready()
        synced = (time.perf_counter() - t0) * 1e3 / a.reps
        t0 = time.perf_counter()
        ys = [matvec(X, *ops) for _ in range(a.reps)]
        ys[-1].block_until_ready()
        pipelined = (time.perf_counter() - t0) * 1e3 / a.reps
        out[nb] = {"synced_ms": synced, "pipelined_ms": pipelined,
                   "gap_ms": synced - pipelined}
        print("[dispatch] nb=%-3d synced=%.3f ms  pipelined=%.3f ms  "
              "per-call host gap=%.3f ms (%.1f%%)"
              % (nb, synced, pipelined, synced - pipelined,
                 100.0 * (synced - pipelined) / synced), flush=True)
    return out


def mode_rung(a, mesh, data) -> dict:
    """Price the rung chain stage by stage, at the production shapes.

    Each stage is its own ``jax.jit`` with a device sync after it, so this is an
    UPPER bound on a fused implementation and a lower bound on the number of
    materialisations: it is the budget, not the production emission.
    """
    from common.fft_helpers import make_sharded_fftn_3d, make_sharded_ifftn_3d
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_feast import ensure_W_R

    sh = make_bse_shardings(mesh)
    ensure_W_R(data, include_W=True, mesh_xy=mesh)
    W_R = data["W_R"]
    nk = a.nkx * a.nky * a.nkz
    spec8 = P(None, "x", "y", None, None, None, None, None)
    ifftn = make_sharded_ifftn_3d(mesh, spec8, spec8, axes=(5, 6, 7), norm="ortho")
    fftn = make_sharded_fftn_3d(mesh, spec8, spec8, axes=(5, 6, 7), norm="ortho")

    rng = np.random.default_rng(5)
    shape6 = (a.nb_rung, a.nmu, a.nmu, a.ns, a.ns, nk)
    T = jnp.asarray((rng.standard_normal(shape6)
                     + 1j * rng.standard_normal(shape6)) / 8.0)
    with mesh:
        T = jax.device_put(T, sh.T)
    shape8 = shape6[:-1] + (a.nkx, a.nky, a.nkz)

    f_reshape = jax.jit(lambda t: t.reshape(shape8))
    f_mul = jax.jit(lambda t, w: w[None, :, :, None, None, :, :, :] * t)
    stages = [
        ("reshape_6d_to_8d", lambda: f_reshape(T)),
    ]
    T8 = f_reshape(T)
    T8.block_until_ready()
    stages += [
        ("ifftn_k", lambda: ifftn(T8)),
    ]
    TR = ifftn(T8)
    TR.block_until_ready()
    stages += [
        ("W_R_multiply", lambda: f_mul(TR, W_R)),
    ]
    UR = f_mul(TR, W_R)
    UR.block_until_ready()
    stages += [
        ("fftn_k", lambda: fftn(UR)),
    ]
    out = {}
    for name, fn in stages:
        for _ in range(a.warmup):
            fn().block_until_ready()
        ts = []
        PROF.start()
        for i in range(a.reps):
            PROF.push("rung_%s_%d" % (name, i))
            t0 = time.perf_counter()
            fn().block_until_ready()
            ts.append((time.perf_counter() - t0) * 1e3)
            PROF.pop()
        PROF.stop()
        out[name] = ts
        print("[rung] %-18s min=%8.3f ms  med=%8.3f ms"
              % (name, min(ts), float(np.median(ts))), flush=True)
    return out


def mode_solve(a, mesh, data) -> dict:
    """One short block-GMRES solve -- the production solver structure."""
    from bse.bse_w_exact import apply_screening_resolvent_block, build_probe_rhs
    from bse.bse_feast import ladder_matvec_operands
    matvec, diag_h, gen, snapshot, sh = build_stack(mesh, data, a.fuse)
    G = np.zeros((a.ncols, a.nmu), dtype=np.float64)
    for i in range(a.ncols):
        G[i, i] = 1.0
    rhs = build_probe_rhs(G, data, gen, sh)
    jax.block_until_ready(rhs)
    out = {}
    for tag in ("compile+run", "steady"):
        t0 = time.perf_counter()
        res = apply_screening_resolvent_block(
            G, complex(0.0), data, matvec, diag_h, gen, snapshot, sh,
            max_iter=a.max_iter, tol=a.tol, return_iters=True, rhs=rhs,
            operands_fn=ladder_matvec_operands)
        jax.block_until_ready(res)
        dt = time.perf_counter() - t0
        out[tag] = dt
        print("[solve] %-12s wall=%.3f s  iters=%s  resid_max=%.3e"
              % (tag, dt, np.asarray(res[2]).tolist(),
                 float(np.max(np.asarray(res[1])))), flush=True)
        if tag == "compile+run":
            PROF.start()
            PROF.push("gmres_solve")
    PROF.pop()
    PROF.stop()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="matvec",
                    choices=["matvec", "rung", "solve", "dispatch", "shapes"])
    ap.add_argument("--nkx", type=int, default=3)
    ap.add_argument("--nky", type=int, default=3)
    ap.add_argument("--nkz", type=int, default=1)
    ap.add_argument("--nc", type=int, default=20)
    ap.add_argument("--nv", type=int, default=26)
    ap.add_argument("--nmu", type=int, default=399)
    ap.add_argument("--ns", type=int, default=2)
    ap.add_argument("--widths", default="1,2,4,8")
    ap.add_argument("--nb-rung", type=int, default=1)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--no-fuse", dest="fuse", action="store_false")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    devs = jax.devices()
    mesh = Mesh(np.asarray(devs[:1]).reshape(1, 1), ("x", "y"))
    g = geometry(a)
    print("[env] jax=%s devices=%s" % (jax.__version__, devs), flush=True)
    print("[geom] n_rmu=%d nk=%dx%dx%d=%d nc=%d nv=%d ns=%d | pair=%d | "
          "W_R=%.1f MiB | chain/trial=%.1f MiB | psi_c=%.2f MiB psi_v=%.2f MiB "
          "| X/trial=%.3f MiB"
          % (a.nmu, a.nkx, a.nky, a.nkz, g["nk"], a.nc, a.nv, a.ns, g["npair"],
             g["W_R_bytes"] / 2**20, g["chain_bytes_per_trial"] / 2**20,
             g["psi_c_bytes"] / 2**20, g["psi_v_bytes"] / 2**20,
             g["X_bytes_per_trial"] / 2**20), flush=True)
    if a.mode == "shapes":
        return 0

    data = synthetic_payload(mesh, nkx=a.nkx, nky=a.nky, nkz=a.nkz, nc=a.nc,
                             nv=a.nv, nmu=a.nmu, ns=a.ns)
    fn = {"matvec": mode_matvec, "rung": mode_rung, "solve": mode_solve,
          "dispatch": mode_dispatch}[a.mode]
    res = fn(a, mesh, data)
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"mode": a.mode, "geom": g, "args": vars(a), "result": res},
            indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
