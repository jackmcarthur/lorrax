"""Correctness gates + strong-scaling driver for ζ-basis W projection.

Exercises ``common.zeta_projection``: build the transfer between a LARGE
ISDF ζ basis (μ_L ~ 10k, wide band window) and a SMALL one (μ_S ~ 500,
a few val/cond bands), then project ``W_L(q; μ_L, ν_L) -> W_S(q; μ_S,
ν_S)`` by congruence.

Cells (``--cell``)
------------------
``dense``      μ_L=192, μ_S=48 — the whole chain against an independent
               dense numpy reference at 1e-12, INCLUDING a check that the
               device-side synthetic W_L equals its host formula (so the
               formula may be trusted as the reference at scale).
``selection``  ψ_left/ψ_right built from a SELECTION matrix (rows of the
               identity picking μ_S of the μ_L indices).  Then
               ``W_S == W_L[idx, idx]`` must hold to round-off — a gate on
               the congruence kernel that needs NO dense reference and so
               runs at FULL production scale.
``roundtrip``  the physically meaningful invariance: build ζ_S as exact
               linear combinations ``ζ_S = A ζ_L`` (so the small basis
               SPANS a subspace of the large one), embed an arbitrary
               small-basis ``W_S0`` into the large basis exactly
               (``W_L = Aᵀ W_S0 conj(A)``), project back, and require
               ``W_S == W_S0``.  This is exactness on representable
               fields — the defining property of the least-squares
               transfer, and it FAILS for ``mode=raw`` (which the cell
               also demonstrates, so the choice is measured, not asserted).
``scale``      production shapes, timed: ζ generation, overlap build,
               Gram, LS solve, and R repetitions of the congruence, plus
               /proc VmHWM per rank and the selection gate at full size.

Everything is deterministic and P-INDEPENDENT by construction (the
synthetic operands are built from small replicated seed vectors inside a
shard_map, from the rank's own axis indices), so ``--cell scale`` prints
a checksum that must agree bit-for-bit-ish across P — a strong-scaling
run doubles as a mesh-invariance gate.

Usage::

    # 4 emulated devices, one process
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      python -u -m common.zeta_projection_test --cell dense --mesh 2x2

    # production scaling point
    srun -N32 --ntasks-per-node=2 -n64 ... python -u -m \
      common.zeta_projection_test --cell scale --mesh 8x8 \
      --mu-l 10008 --mu-s 504 --nq 16 --ng 4032 --reps 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"


def _init():
    if os.environ.get(_DIST):
        return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        # Gloo transport: pin the high-speed fabric BEFORE the backend is
        # created (a backend factory cannot be replaced afterwards).  A
        # no-op when JAX_CPU_COLLECTIVES_IMPLEMENTATION != gloo.  This
        # driver runs on gloo BY MEASUREMENT: job 7879485 shows the mpi
        # collectives cannot create a grouped clique for a standalone
        # driver under any warm-up, while gloo/ib0 passes the same chain.
        try:
            import runtime as _rt
            _rt.pin_gloo_interface()
        except Exception as _exc:                              # noqa: BLE001
            print(f"[zeta_projection_test] pin_gloo_interface skipped: "
                  f"{type(_exc).__name__}: {_exc}", flush=True)
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        one_gpu = cvd and "," not in cvd
        init_kwargs = {"local_device_ids": [0]} if one_gpu else {}
        jax.distributed.initialize(**init_kwargs)
    os.environ[_DIST] = "1"


_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P   # noqa: E402
from jax.experimental import multihost_utils                       # noqa: E402
from jax.experimental.shard_map import shard_map                   # noqa: E402

from common.zeta_projection import (                                # noqa: E402
    ensure_world_clique_ready,
    build_zeta_transfer,
    project_w_between_zeta_bases,
    transfer_operands_from_dense,
    zeta_overlap_block_reshard,
    zeta_overlap_single_axis,
    zeta_gram_single_axis,
    zeta_gram_replicated,
    least_squares_transfer,
)
from common.contract_bands import (                                 # noqa: E402
    ensure_grouped_collectives_ready, bands_gemm_ffi_enabled)
from common.collectives import (                                    # noqa: E402
    barrier, device_put_process_local as _put)


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


# ---------------------------------------------------------------------------
# deterministic, P-independent synthetic operands
# ---------------------------------------------------------------------------
# ζ:  z[q, μ, G] = a(μ,G) · exp(2πi·h(q,μ,G)/M) with h a BIT-MIXING integer
#     hash.  Two properties are load-bearing and were both learned the hard
#     way (gate job 7879471 refused the first attempt):
#       * the phase must be genuinely mixed in μ AND G.  A purely
#         multiplicative hash ``(a·μ + b·G + c·μG) mod 2^20`` makes the
#         phase DIFFERENCE between two ζ's linear in G with a slope ~1/2^20
#         — over n_G ≲ 4k grid points every ζ is then nearly parallel to
#         every other and the Gram is numerically singular.  The ``s·t``
#         cross term below (each factor already xor-shift mixed) is what
#         decorrelates them; measured cond(G_S) ≈ 20 at both gate sizes.
#       * μ-dependent amplitude, so the Gram diagonal is not constant
#         (a constant diagonal is a good smell test that the phases are
#         degenerate — that is exactly what the failing run printed).
#     Everything is integer arithmetic below 2^62, so it is bit-reproducible
#     in numpy and, crucially, evaluated ONLY on each rank's own G block.
_H_M = 2 ** 31


def _make_zeta(mesh, n_q, n_mu, n_g, tag, axes=("x", "y")):
    """(n_q, n_mu, n_G) at ``P(None, None, ('x','y'))``, built rank-locally."""
    ax_x, ax_y = axes
    p_x, p_y = int(mesh.shape[ax_x]), int(mesh.shape[ax_y])
    g_loc = n_g // (p_x * p_y)
    dummy = _put(np.zeros(1), NamedSharding(mesh, P(None)))

    def _body(_d):
        flat = (jax.lax.axis_index(ax_x).astype(jnp.int64) * p_y
                + jax.lax.axis_index(ax_y).astype(jnp.int64))
        g = flat * g_loc + jnp.arange(g_loc, dtype=jnp.int64)
        m = jnp.arange(n_mu, dtype=jnp.int64)
        return _zeta_block(n_q, m, g, n_g, tag)

    return jax.jit(shard_map(
        _body, mesh=mesh, in_specs=(P(None),),
        out_specs=P(None, None, (ax_x, ax_y)), check_rep=False))(dummy)


def _zeta_block(n_q, m_idx, g_idx, n_g, tag):
    """z[q, m_idx, g_idx] — a pure function of the GLOBAL indices, so any
    sharding of the same ζ produces bit-identical values."""
    q = jnp.arange(n_q, dtype=jnp.int64)[:, None, None]
    m = m_idx[None, :, None] + 7 * tag
    gg = g_idx[None, None, :]
    s = (m * 1103515245 + 12345 + 101 * tag) % _H_M
    s = ((s ^ (s >> 7)) * 2654435761) % _H_M
    t = (gg * 1664525 + 1013904223) % _H_M
    t = ((t ^ (t >> 9)) * 40503) % _H_M
    h = (s + t + s * t + q * 374761393) % _H_M
    amp = ((1.0 / (1.0 + 3.0 * (gg.astype(jnp.float64) / float(n_g))))
           * (0.5 + ((m * 2654435761) % 1000).astype(jnp.float64) / 1e3))
    return (amp * jnp.exp(2j * np.pi * h.astype(jnp.float64) / _H_M)
            ).astype(jnp.complex128)


def _make_zeta_2d(mesh, n_q, n_mu, n_g, tag, *, mu_axis, g_axis):
    """The SAME ζ at the two-pass plan's layout.

    ``mu_axis=None``  → ``P(None, None, g_axis)``   (ζ_S: μ replicated)
    ``mu_axis='x'|'y'`` → ``P(None, mu_axis, g_axis)`` (ζ_L: μ and G tiled)

    Values are a pure function of the global (q, μ, G) indices, so this is
    the identical tensor ``_make_zeta`` builds — which is what makes the
    two overlap plans directly comparable at production scale.
    """
    p_g = int(mesh.shape[g_axis])
    g_loc = n_g // p_g
    if mu_axis is None:
        mu_loc, spec = n_mu, P(None, None, g_axis)
    else:
        mu_loc = n_mu // int(mesh.shape[mu_axis])
        spec = P(None, mu_axis, g_axis)
    dummy = _put(np.zeros(1), NamedSharding(mesh, P(None)))

    def _body(_d):
        g = (jax.lax.axis_index(g_axis).astype(jnp.int64) * g_loc
             + jnp.arange(g_loc, dtype=jnp.int64))
        if mu_axis is None:
            m = jnp.arange(mu_loc, dtype=jnp.int64)
        else:
            m = (jax.lax.axis_index(mu_axis).astype(jnp.int64) * mu_loc
                 + jnp.arange(mu_loc, dtype=jnp.int64))
        return _zeta_block(n_q, m, g, n_g, tag)

    return jax.jit(shard_map(
        _body, mesh=mesh, in_specs=(P(None),), out_specs=spec,
        check_rep=False))(dummy)


# W_L:  W[q,i,j] = f[q,i] · conj(f[q,j]) · g[|i-j|]  with g REAL  ⇒ exactly
#       Hermitian, buildable rank-locally from two small replicated seed
#       vectors, identical at every P, and every entry computable on the
#       host in O(1) — which is what lets the selection gate run at
#       production scale with no gather of any (μ_L, μ_L) object.
def _w_seeds(n_q, n_mu, seed=11):
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0.0, 2.0 * np.pi, size=(n_q, n_mu))
    mag = 0.5 + rng.uniform(size=(n_q, n_mu))
    f = (mag * np.exp(1j * ph)).astype(np.complex128)
    d = np.arange(n_mu, dtype=np.float64)
    g = (np.exp(-d / max(n_mu / 8.0, 1.0))
         * (1.0 + 0.3 * np.cos(2.0 * np.pi * 7.0 * d / n_mu)))
    return f, g.astype(np.float64)


def _make_W_L(mesh, f_host, g_host, axes=("x", "y")):
    """(n_q, μ_L, μ_L) Hermitian at ``P(None,'x','y')``, no global transient."""
    ax_x, ax_y = axes
    n_q, n_mu = f_host.shape
    p_x, p_y = int(mesh.shape[ax_x]), int(mesh.shape[ax_y])
    mx, my = n_mu // p_x, n_mu // p_y
    f_dev = _put(f_host, NamedSharding(mesh, P(None, None)))
    g_dev = _put(g_host, NamedSharding(mesh, P(None)))

    def _body(f, g):
        i0 = jax.lax.axis_index(ax_x) * mx
        j0 = jax.lax.axis_index(ax_y) * my
        fi = jax.lax.dynamic_slice_in_dim(f, i0, mx, axis=1)      # (nq, mx)
        fj = jax.lax.dynamic_slice_in_dim(f, j0, my, axis=1)      # (nq, my)
        i = i0 + jnp.arange(mx)
        j = j0 + jnp.arange(my)
        gij = g[jnp.abs(i[:, None] - j[None, :])]                 # (mx, my)
        return (fi[:, :, None] * jnp.conj(fj)[:, None, :]
                * gij[None, :, :].astype(jnp.complex128))

    return jax.jit(shard_map(
        _body, mesh=mesh, in_specs=(P(None, None), P(None)),
        out_specs=P(None, ax_x, ax_y), check_rep=False))(f_dev, g_dev)


def _w_host_block(f, g, rows, cols):
    """Host reference for ``W_L[:, rows, cols]`` — O(len(rows)·len(cols))."""
    fi = f[:, rows]
    fj = f[:, cols]
    d = np.abs(np.asarray(rows)[:, None] - np.asarray(cols)[None, :])
    return fi[:, :, None] * np.conj(fj)[:, None, :] * g[d][None, :, :]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _selection_operands(mesh, n_q, mu_l, mu_s, axes=("x", "y")):
    """ψ_left/ψ_right for T = a selection matrix (rows of the identity).

    The selected large-basis indices are evenly spread over [0, μ_L).
    Built rank-locally from the index arithmetic, so no (μ_S, μ_L) host
    array is ever formed either.
    """
    ax_x, ax_y = axes
    p_x, p_y = int(mesh.shape[ax_x]), int(mesh.shape[ax_y])
    idx = (np.arange(mu_s, dtype=np.int64) * (mu_l // mu_s)) % mu_l
    idx_dev = _put(idx, NamedSharding(mesh, P(None)))

    def _mk(ax, p_ax):
        loc = mu_l // p_ax

        def _body(ix):
            off = jax.lax.axis_index(ax).astype(jnp.int64) * loc
            cols = off + jnp.arange(loc, dtype=jnp.int64)
            # T[q, m, mu] = 1 iff idx[m] == mu  (same T at every q)
            sel = (ix[:, None] == cols[None, :]).astype(jnp.complex128)
            return jnp.broadcast_to(sel[None], (n_q,) + sel.shape)
        return jax.jit(shard_map(
            _body, mesh=mesh, in_specs=(P(None),),
            out_specs=P(None, None, ax), check_rep=False))(idx_dev)

    ops = jax.jit(lambda a, b: transfer_operands_from_dense(a, b, mesh, axes))
    return (*ops(_mk(ax_x, p_x), _mk(ax_y, p_y)), idx)


def _vmhwm_gb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / (1024.0 ** 2)
    except OSError:
        pass
    return float("nan")


def _gather_rank_scalars(x: float) -> np.ndarray:
    return np.asarray(multihost_utils.process_allgather(
        jnp.asarray([float(x)]), tiled=True)).ravel()


def _mesh_from(arg: str) -> Mesh:
    px, py = (int(t) for t in arg.lower().split("x"))
    devs = np.asarray(jax.devices())
    if px * py != devs.size:
        raise SystemExit(
            f"mesh {px}x{py} = {px * py} but jax sees {devs.size} devices")
    return Mesh(devs.reshape(px, py), axis_names=("x", "y"))


def _rel(a, b):
    den = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b))) / den


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------

def cell_dense(mesh, args) -> int:
    n_q, mu_l, mu_s, n_g = args.nq, args.mu_l, args.mu_s, args.ng
    _log(f"--- cell dense: n_q={n_q} μ_L={mu_l} μ_S={mu_s} n_G={n_g} "
         f"mesh={mesh.shape['x']}x{mesh.shape['y']} ---")
    zeta_L = _make_zeta(mesh, n_q, mu_l, n_g, tag=0)
    zeta_S = _make_zeta(mesh, n_q, mu_s, n_g, tag=1)
    f_host, g_host = _w_seeds(n_q, mu_l)
    W_L = _make_W_L(mesh, f_host, g_host)

    # (0) the synthetic W_L on device IS its host formula — this is what
    #     licenses the formula as the reference in the scale cell.
    W_L_full = np.asarray(multihost_utils.process_allgather(W_L, tiled=True))
    W_ref_full = _w_host_block(f_host, g_host, np.arange(mu_l),
                               np.arange(mu_l))
    e_model = _rel(W_L_full, W_ref_full)
    herm = _rel(W_L_full, np.conj(np.swapaxes(W_L_full, -1, -2)))

    psi_l, psi_r, info = build_zeta_transfer(
        zeta_S, zeta_L, mesh, mode=args.mode, print_fn=_log,
        mu_s_chunk=(args.mu_s_chunk or None))
    project = project_w_between_zeta_bases(mesh)
    W_S = jax.jit(project)(W_L, psi_l, psi_r)
    W_S_full = np.asarray(multihost_utils.process_allgather(W_S, tiled=True))

    # independent dense reference, entirely in numpy
    zS = np.asarray(multihost_utils.process_allgather(zeta_S, tiled=True))
    zL = np.asarray(multihost_utils.process_allgather(zeta_L, tiled=True))
    O_ref = np.einsum('qmg,qng->qmn', np.conj(zS), zL)
    if args.mode == "ls":
        G_ref = np.einsum('qmg,qng->qmn', np.conj(zS), zS)
        T_ref = np.linalg.solve(G_ref, O_ref)
    else:
        T_ref = O_ref
    W_ref = np.einsum('qam,qmn,qbn->qab', T_ref, W_L_full, np.conj(T_ref))

    e_proj = _rel(W_S_full, W_ref)
    e_herm_s = _rel(W_S_full, np.conj(np.swapaxes(W_S_full, -1, -2)))
    cond = float(np.max([np.linalg.cond(
        np.einsum('mg,ng->mn', np.conj(zS[q]), zS[q])) for q in range(n_q)]))
    _log(f"  W_L device vs host formula   rel = {e_model:.3e}  (gate 1e-14)")
    _log(f"  W_L hermiticity              rel = {herm:.3e}  (gate 1e-14)")
    _log(f"  W_S vs dense numpy reference rel = {e_proj:.3e}  (gate "
         f"{args.tol:.0e})")
    _log(f"  W_S hermiticity              rel = {e_herm_s:.3e}  (gate 1e-12)")
    _log(f"  cond(G_S) max over q             = {cond:.3e}")
    ok = (e_model < 1e-14 and herm < 1e-14 and e_proj < args.tol
          and e_herm_s < 1e-12)
    _log(f"  {'PASS' if ok else 'FAIL'} [dense]")
    return 0 if ok else 1


def cell_selection(mesh, args) -> int:
    n_q, mu_l, mu_s = args.nq, args.mu_l, args.mu_s
    _log(f"--- cell selection: n_q={n_q} μ_L={mu_l} μ_S={mu_s} ---")
    f_host, g_host = _w_seeds(n_q, mu_l)
    W_L = _make_W_L(mesh, f_host, g_host)
    psi_l, psi_r, idx = _selection_operands(mesh, n_q, mu_l, mu_s)
    project = project_w_between_zeta_bases(mesh)
    W_S = jax.jit(project)(W_L, psi_l, psi_r)
    W_S_full = np.asarray(multihost_utils.process_allgather(W_S, tiled=True))
    ref = _w_host_block(f_host, g_host, idx, idx)
    err = _rel(W_S_full, ref)
    _log(f"  W_S vs W_L[idx, idx]  rel = {err:.3e}  (gate {args.tol:.0e})")
    _log(f"  {'PASS' if err < args.tol else 'FAIL'} [selection]")
    return 0 if err < args.tol else 1


def cell_roundtrip(mesh, args) -> int:
    """Exactness on representable fields (and the raw-mode counterexample)."""
    n_q, mu_l, mu_s, n_g = args.nq, args.mu_l, args.mu_s, args.ng
    _log(f"--- cell roundtrip: n_q={n_q} μ_L={mu_l} μ_S={mu_s} n_G={n_g} ---")
    rng = np.random.default_rng(5)
    A = (rng.standard_normal((mu_s, mu_l))
         + 1j * rng.standard_normal((mu_s, mu_l))) / np.sqrt(2 * mu_l)

    zeta_L = _make_zeta(mesh, n_q, mu_l, n_g, tag=0)
    zL = np.asarray(multihost_utils.process_allgather(zeta_L, tiled=True))
    zS = np.einsum('ml,qlg->qmg', A, zL)          # ζ_S = A ζ_L  EXACTLY
    zeta_S = _put(zS, NamedSharding(mesh, P(None, None, ("x", "y"))))

    W0 = (rng.standard_normal((n_q, mu_s, mu_s))
          + 1j * rng.standard_normal((n_q, mu_s, mu_s)))
    W0 = 0.5 * (W0 + np.conj(np.swapaxes(W0, -1, -2)))
    # exact embedding of the small-basis field into the large basis
    W_L_host = np.einsum('ma,qmn,nb->qab', A, W0, np.conj(A))
    W_L = _put(W_L_host, NamedSharding(mesh, P(None, "x", "y")))

    res = {}
    for mode in ("ls", "raw"):
        psi_l, psi_r, _ = build_zeta_transfer(
            zeta_S, zeta_L, mesh, mode=mode, print_fn=_log, announce=False,
            mu_s_chunk=(args.mu_s_chunk or None))
        W_S = jax.jit(project_w_between_zeta_bases(mesh))(W_L, psi_l, psi_r)
        got = np.asarray(multihost_utils.process_allgather(W_S, tiled=True))
        res[mode] = _rel(got, W0)
        _log(f"  mode={mode:<4s} rel |W_S - W_S0| = {res[mode]:.3e}")
    ok = res["ls"] < args.tol and res["raw"] > 1e-3
    _log(f"  least-squares transfer is EXACT on representable fields; the "
         f"bare-overlap congruence is off by {res['raw']:.2e} (it carries "
         f"the metric scale squared) — the choice is measured, not asserted.")
    _log(f"  {'PASS' if ok else 'FAIL'} [roundtrip]")
    return 0 if ok else 1


def cell_scale(mesh, args) -> int:
    n_q, mu_l, mu_s, n_g = args.nq, args.mu_l, args.mu_s, args.ng
    p_x, p_y = int(mesh.shape["x"]), int(mesh.shape["y"])
    world = p_x * p_y
    _log(f"=== cell scale: P={world} ({p_x}x{p_y})  n_q={n_q} μ_L={mu_l} "
         f"μ_S={mu_s} n_G={n_g} reps={args.reps} "
         f"gemm_ffi={bands_gemm_ffi_enabled()} ===")
    gb = 1e9
    _log(f"  global operands: W_L {n_q * mu_l * mu_l * 16 / gb:.2f} GB, "
         f"ζ_L {n_q * mu_l * n_g * 16 / gb:.2f} GB, "
         f"ζ_S {n_q * mu_s * n_g * 16 / gb:.3f} GB, "
         f"T {n_q * mu_s * mu_l * 16 / gb:.3f} GB, "
         f"W_S {n_q * mu_s * mu_s * 16 / gb:.4f} GB")

    rows = {}

    def _stage(name, fn):
        barrier(f"zp.{name}")
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        barrier(f"zp.{name}.done")
        rows[name] = time.perf_counter() - t0
        _log(f"    [{name:<14s}] {rows[name]:8.3f} s")
        return out

    zeta_L = _stage("zeta_L.gen", lambda: _make_zeta(mesh, n_q, mu_l, n_g, 0))
    zeta_S = _stage("zeta_S.gen", lambda: _make_zeta(mesh, n_q, mu_s, n_g, 1))

    # --- plan A: the G-only-sharded overlap (one pass, psum_scatter chain).
    overlap = jax.jit(zeta_overlap_block_reshard(mesh))
    O_x, O_y = _stage("ovl.1pass", lambda: overlap(zeta_S, zeta_L))
    gram = jax.jit(zeta_gram_replicated(mesh))
    G_S = _stage("gram.1pass", lambda: gram(zeta_S))
    del zeta_L, zeta_S

    # --- plan B: μ_L AND G sharded (two passes, one all-reduce each).  ζ is
    # an analytic function of the GLOBAL indices, so regenerating it at the
    # other layout is cheaper than resharding and gives the identical
    # tensor — which makes the two plans directly comparable.
    ovl_x = jax.jit(zeta_overlap_single_axis(mesh, mu_axis="x", g_axis="y"))
    ovl_y = jax.jit(zeta_overlap_single_axis(mesh, mu_axis="y", g_axis="x"))
    gram_y = jax.jit(zeta_gram_single_axis(mesh, g_axis="y"))

    def _pass_x():
        zS = _make_zeta_2d(mesh, n_q, mu_s, n_g, 1, mu_axis=None, g_axis="y")
        zL = _make_zeta_2d(mesh, n_q, mu_l, n_g, 0, mu_axis="x", g_axis="y")
        return ovl_x(zS, zL), gram_y(zS)

    def _pass_y():
        zS = _make_zeta_2d(mesh, n_q, mu_s, n_g, 1, mu_axis=None, g_axis="x")
        zL = _make_zeta_2d(mesh, n_q, mu_l, n_g, 0, mu_axis="y", g_axis="x")
        return ovl_y(zS, zL)

    Ox2, G_S2 = _stage("ovl.2pass.x", _pass_x)
    Oy2 = _stage("ovl.2pass.y", _pass_y)
    rows["ovl.2pass"] = rows["ovl.2pass.x"] + rows["ovl.2pass.y"]
    _log(f"    [{'ovl.2pass(tot)':<14s}] {rows['ovl.2pass']:8.3f} s   "
         f"vs 1pass {rows['ovl.1pass'] + rows['gram.1pass']:8.3f} s "
         f"(incl. its Gram)")

    # plan parity at PRODUCTION scale — no gather, one jitted reduction
    def _reldiff(a, b):
        f = jax.jit(lambda u, v: jnp.asarray(
            [jnp.max(jnp.abs(u - v)), jnp.max(jnp.abs(u))]),
            out_shardings=NamedSharding(mesh, P()))
        d, n = np.asarray(f(a, b).addressable_data(0))
        return float(d) / max(float(n), 1e-300)

    e_ox, e_oy = _reldiff(O_x, Ox2), _reldiff(O_y, Oy2)
    e_gs = _reldiff(G_S, G_S2)
    ok_plans = max(e_ox, e_oy, e_gs) < args.tol
    _log(f"  overlap plan parity  O_x {e_ox:.3e}  O_y {e_oy:.3e}  "
         f"G_S {e_gs:.3e}  (gate {args.tol:.0e})  "
         f"{'PASS' if ok_plans else 'FAIL'} [plan-parity]")
    del O_x, O_y, G_S

    T_x = _stage("ls_solve.x", lambda: least_squares_transfer(G_S2, Ox2, mesh, "x"))
    T_y = _stage("ls_solve.y", lambda: least_squares_transfer(G_S2, Oy2, mesh, "y"))
    ops = jax.jit(lambda a, b: transfer_operands_from_dense(a, b, mesh))
    psi_l, psi_r = _stage("operands", lambda: ops(T_x, T_y))
    del Ox2, Oy2, T_x, T_y

    f_host, g_host = _w_seeds(n_q, mu_l)
    W_L = _stage("W_L.gen", lambda: _make_W_L(mesh, f_host, g_host))

    project = jax.jit(project_w_between_zeta_bases(mesh))
    W_S = _stage("project.compile+1", lambda: project(W_L, psi_l, psi_r))

    times = []
    for r in range(args.reps):
        barrier(f"zp.rep{r}")
        t0 = time.perf_counter()
        W_S = project(W_L, psi_l, psi_r)
        jax.block_until_ready(W_S)
        barrier(f"zp.rep{r}.done")
        times.append(time.perf_counter() - t0)
    times = np.asarray(times)
    rows["project.min"] = float(times.min())
    rows["project.med"] = float(np.median(times))
    rows["project.mean"] = float(times.mean())
    _log(f"    [project x{args.reps:<7d}] min {times.min():8.3f}  med "
         f"{np.median(times):8.3f}  mean {times.mean():8.3f}  max "
         f"{times.max():8.3f} s")

    # P-invariance checksum of the projected object (no gather of W_L).
    # out_shardings MUST be pinned replicated: without it XLA leaves the
    # 3-vector on a non-fully-addressable sharding and process_allgather
    # refuses it ("Gathering global non-fully-addressable arrays only
    # supports tiled=True") — job 7879488 P=4 died exactly there, after
    # the reps.  Replicated + addressable_data(0) needs no gather at all.
    chk = np.asarray(jax.jit(
        lambda w: jnp.asarray([jnp.sum(w).real, jnp.sum(w).imag,
                               jnp.sum(jnp.abs(w) ** 2).real]),
        out_shardings=NamedSharding(mesh, P()))(W_S).addressable_data(0))
    _log(f"  W_S checksum  Re Σ = {chk[0]:.15e}")
    _log(f"  W_S checksum  Im Σ = {chk[1]:.15e}")
    _log(f"  W_S checksum ‖·‖²  = {chk[2]:.15e}")

    # W_S = T W_L T† with W_L Hermitian is Hermitian for ANY T — so this
    # is a gate on ψ_left and ψ_right being the SAME transfer in the two
    # shardings.  (It caught exactly that: job 7879491 P=4 printed
    # Im Σ W_S = 6.25e+02 where Hermiticity forces 0.)
    e_herm = _reldiff(W_S, jax.jit(
        lambda w: jnp.conj(jnp.swapaxes(w, -1, -2)),
        out_shardings=NamedSharding(mesh, P(None, "x", "y")))(W_S))
    _log(f"  W_S hermiticity      rel = {e_herm:.3e}  (gate {args.tol:.0e})  "
         f"{'PASS' if e_herm < args.tol else 'FAIL'} [hermiticity]")
    ok_plans = ok_plans and e_herm < args.tol

    hwm = _gather_rank_scalars(_vmhwm_gb())
    _log(f"  VmHWM/rank GB: min {hwm.min():.2f}  med {np.median(hwm):.2f}  "
         f"max {hwm.max():.2f}  (sum {hwm.sum():.1f})")
    _log(f"  ROW P={world} project_min={times.min():.4f} "
         f"project_med={np.median(times):.4f} "
         f"ovl1={rows['ovl.1pass']:.4f} gram1={rows['gram.1pass']:.4f} "
         f"ls={rows['ls_solve.x'] + rows['ls_solve.y']:.4f} "
         f"ovl2={rows['ovl.2pass']:.4f} "
         f"hwm_max={hwm.max():.3f} hwm_med={np.median(hwm):.3f} "
         f"chk_re={chk[0]:.12e} chk_n2={chk[2]:.12e}")

    del W_S, psi_l, psi_r, G_S2
    rc = 0 if ok_plans else 1
    if args.with_selection:
        rc = cell_selection(mesh, args) or rc
    return rc


def _rank_checksum(mesh, A, mu_axis, axes=("x", "y")):
    """Per-rank ‖shard‖² of a (n_q, μ_S, μ_L) array sharded on ``mu_axis``.

    Returns a length-P vector, one entry per rank, so a mismatch can be
    localized to the rank (hence the mesh position, hence the replica
    group) whose shard changed — which is what turns "the answer is
    wrong" into "collective X on group Y delivered corrupt data".
    """
    ax_x, ax_y = axes

    def _body(a):
        return jnp.asarray([jnp.sum(jnp.abs(a) ** 2).real])

    v = jax.jit(shard_map(
        _body, mesh=mesh, in_specs=(P(None, None, mu_axis),),
        out_specs=P((ax_x, ax_y)), check_rep=False))(A)
    # The (P,) result is itself sharded one-entry-per-rank, so a bare
    # np.asarray raises "spans non-addressable devices" (it did, job
    # 7879519, and took the localization dump down with it).
    return np.asarray(multihost_utils.process_allgather(v, tiled=True))


def cell_provoke(mesh, args) -> int:
    """Repeat the EXACT P=4 configuration N times and report an incidence rate.

    Job 7879491 returned a non-Hermitian W_S at P=4 (Im Σ = 6.25e+02 where
    the algebra forces 0); job 7879499 re-ran the identical configuration
    and was clean.  A single sighting is not actionable — a RATE is.  This
    cell holds every operand fixed, recomputes both overlap plans N times
    in one process, and compares each rep against its own rep-0 result and
    against the other plan.  Every operand here is deterministic, so ANY
    nonzero difference between reps is corruption, not arithmetic: the
    comparison threshold is EXACT ZERO, not a tolerance.

    On a mismatch it dumps per-rank ‖shard‖² for both plans and both reps,
    which localizes the corruption to a rank / replica group.
    """
    n_q, mu_l, mu_s, n_g = args.nq, args.mu_l, args.mu_s, args.ng
    p_x, p_y = int(mesh.shape["x"]), int(mesh.shape["y"])
    world = p_x * p_y
    _log(f"=== cell provoke: P={world} ({p_x}x{p_y}) n_q={n_q} μ_L={mu_l} "
         f"μ_S={mu_s} n_G={n_g} reps={args.reps} ===")
    _log(f"  comparison threshold is EXACT 0 — the operands are identical "
         f"every rep, so any bit difference is transport corruption.")

    zL = _make_zeta(mesh, n_q, mu_l, n_g, 0)
    zS = _make_zeta(mesh, n_q, mu_s, n_g, 1)
    zL_xy = _make_zeta_2d(mesh, n_q, mu_l, n_g, 0, mu_axis="x", g_axis="y")
    zS_y = _make_zeta_2d(mesh, n_q, mu_s, n_g, 1, mu_axis=None, g_axis="y")
    zL_yx = _make_zeta_2d(mesh, n_q, mu_l, n_g, 0, mu_axis="y", g_axis="x")
    zS_x = _make_zeta_2d(mesh, n_q, mu_s, n_g, 1, mu_axis=None, g_axis="x")
    jax.block_until_ready((zL, zS, zL_xy, zS_y, zL_yx, zS_x))

    ovl1 = jax.jit(zeta_overlap_block_reshard(mesh))
    ovl_x = jax.jit(zeta_overlap_single_axis(mesh, mu_axis="x", g_axis="y"))
    ovl_y = jax.jit(zeta_overlap_single_axis(mesh, mu_axis="y", g_axis="x"))

    def _absmax(a, b):
        f = jax.jit(lambda u, v: jnp.max(jnp.abs(u - v)),
                    out_shardings=NamedSharding(mesh, P()))
        return float(np.asarray(f(a, b).addressable_data(0)))

    ref = {}
    bad, n_checked, xp = [], 0, []
    for rep in range(args.reps):
        barrier(f"zp.provoke{rep}")
        t0 = time.perf_counter()
        A_x, A_y = ovl1(zS, zL)
        B_x = ovl_x(zS_y, zL_xy)
        B_y = ovl_y(zS_x, zL_yx)
        jax.block_until_ready((A_x, A_y, B_x, B_y))
        dt = time.perf_counter() - t0
        cur = {"1p.O_x": A_x, "1p.O_y": A_y, "2p.O_x": B_x, "2p.O_y": B_y}
        # Cross-plan EVERY rep: the two plans compute the same matrix by
        # structurally different collectives, so a large cross-plan gap
        # says WHICH plan is wrong on THIS rep (fp reassociation alone is
        # ~7e-14 here).  Job 7879519 proc0 rep0 showed O_x 7.2e-14 and
        # O_y 1.6e+02 — the one-pass O_y chain, and only that chain.
        cx, cy = _absmax(A_x, B_x), _absmax(A_y, B_y)
        xp.append((rep, cx, cy))
        if cy > 1e-9 or cx > 1e-9:
            _log(f"  *** rep {rep}: CROSS-PLAN gap O_x {cx:.6e} "
                 f"O_y {cy:.6e} (fp noise is ~7e-14) ***")
        if rep == 0:
            ref = dict(cur)
            _log(f"  rep 0: cross-plan |1p-2p| O_x {cx:.3e} O_y {cy:.3e} "
                 f"({dt:.2f} s)")
            continue
        d = {k: _absmax(cur[k], ref[k]) for k in cur}
        n_checked += 1
        worst = max(d.values())
        if worst != 0.0:
            bad.append((rep, dict(d)))
            _log(f"  *** rep {rep}: NONZERO drift vs rep 0 — "
                 + "  ".join(f"{k} {v:.6e}" for k, v in d.items()) + " ***")
            for k, ax in (("1p.O_x", "x"), ("1p.O_y", "y"),
                          ("2p.O_x", "x"), ("2p.O_y", "y")):
                if d[k] == 0.0:
                    continue
                try:
                    c0 = _rank_checksum(mesh, ref[k], ax)
                    c1 = _rank_checksum(mesh, cur[k], ax)
                except Exception as exc:              # noqa: BLE001
                    _log(f"      {k}: rank-checksum dump failed "
                         f"({type(exc).__name__}: {exc}) — incidence "
                         f"measurement continues")
                    continue
                diff = np.nonzero(c1 - c0)[0]
                _log(f"      {k}: ranks with changed ‖shard‖² = "
                     f"{diff.tolist()}  (of {world})")
                for r in diff.tolist():
                    _log(f"        rank {r} (x={r // p_y}, y={r % p_y}): "
                         f"{c0[r]:.15e} -> {c1[r]:.15e}")
        elif rep % 5 == 0:
            _log(f"  rep {rep}: clean ({dt:.2f} s)")

    nx = sum(1 for _, cx, _ in xp if cx > 1e-9)
    ny = sum(1 for _, _, cy in xp if cy > 1e-9)
    _log(f"  CROSS-PLAN P={world}: O_x wrong in {nx}/{len(xp)} reps, "
         f"O_y wrong in {ny}/{len(xp)} reps "
         f"(1-pass vs the structurally independent 2-pass)")
    rate = len(bad) / max(n_checked, 1)
    _log(f"  INCIDENCE P={world}: {len(bad)}/{n_checked} reps drifted "
         f"({100 * rate:.1f}%)")
    _log(f"  {'PASS' if not bad else 'FAIL'} [provoke] "
         f"(PASS = bit-identical across all reps)")
    return 0 if (not bad and nx == 0 and ny == 0) else 1


def _load_real_zeta(loader, n_q, n_mu, n_g_pad, mesh, spec):
    """Read a real ζ slab and deliver it at ``spec`` with G zero-padded.

    ``read_zeta_G_slab`` returns ``(q, μ, ngkmax)`` at ``P(None,('x','y'),
    None)`` — the V_q driver's layout.  The overlap wants G sharded (or μ
    and G sharded), so there is exactly one reshard here; that is the
    interface cost §7.2 of the notes flagged, now paid rather than
    described.  ``sphere_idx=None`` consumes the raw per-q sphere (the
    loader refuses to narrow a per-q sphere with a shared index); the two
    files having the SAME per-q sphere is what
    ``assert_zeta_bases_compatible`` verifies.

    G is zero-padded ngkmax → n_g_pad so the mesh divides it.  A zero G
    column contributes zero to every overlap and every Gram entry, so the
    pad is inert — the same argument the μ pad rests on.
    """
    # The loader shards μ FLAT over the whole mesh (P(None,('x','y'),None)),
    # so the READ needs μ divisible by P = p_x·p_y — a stricter condition
    # than the overlap's (μ divisible by p_x and by p_y separately), and
    # one that 2406 % 4 = 2 and 606 % 4 = 2 both violate (job 7879524).
    # ``valid_mu`` is the loader's own answer: read a P-divisible μ window
    # zero-filled past the logical extent, then slice the pad off before
    # the reshard, so no zero ζ row ever reaches the Gram.
    n_mu, n_q = int(n_mu), int(n_q)
    p_tot = int(mesh.size)
    mu_read = n_mu + (-n_mu) % p_tot
    z = loader.read_zeta_G_slab(
        q_offset=0, q_count=n_q, mu_offset=0, mu_count=mu_read,
        qvec_batch_frac=jnp.zeros((n_q, 3)), sphere_idx=None,
        mesh=mesh, valid_mu=n_mu)
    n_g = int(z.shape[-1])
    pad = int(n_g_pad) - n_g
    if pad < 0:
        raise ValueError(f"n_g_pad={n_g_pad} < ngkmax={n_g}")
    return jax.jit(
        lambda a: jax.lax.with_sharding_constraint(
            jnp.pad(a[:, :n_mu, :], ((0, 0), (0, 0), (0, pad))),
            NamedSharding(mesh, spec)))(z)


def cell_real(mesh, args) -> int:
    """The whole chain on REAL ISDF ζ from two production fits.

    Owner follow-up #1.  ζ_L and ζ_S are read from two real
    ``zeta_q.h5`` files on the SAME deck, grid, q set and per-q G-sphere,
    so the overlap, the Gram, the metric and the conditioning are all
    real physics data.  W_L remains an analytic Hermitian matrix **in the
    real large basis** — no real W exists on disk (V_q/W are in-memory
    in this pipeline) — and that is stated rather than blurred.
    """
    from file_io.zeta_loader import ZetaLoader
    from common.zeta_projection import assert_zeta_bases_compatible

    p_x, p_y = int(mesh.shape["x"]), int(mesh.shape["y"])
    _log(f"--- cell real: ζ_S={args.zeta_small}")
    _log(f"               ζ_L={args.zeta_large}")
    with ZetaLoader(args.zeta_small, mesh=mesh) as ldS, \
            ZetaLoader(args.zeta_large, mesh=mesh) as ldL:
        geom = assert_zeta_bases_compatible(ldS, ldL, print_fn=_log)
        mu_s, mu_l = geom["mu_S"], geom["mu_L"]
        n_q = min(int(args.nq), geom["n_q_on_disk"])
        ngk = geom["ngkmax"]
        n_g = ngk + (-ngk) % (p_x * p_y)
        for name, n in (("mu_S", mu_s), ("mu_L", mu_l)):
            if n % p_x or n % p_y:
                _log(f"  REFUSE: {name}={n} is not divisible by p_x={p_x} "
                     f"and p_y={p_y}.  The READ pad is handled (valid_mu), "
                     f"but padding μ_S into the GRAM would make G_S "
                     f"singular; pick a mesh whose axes divide {mu_s} and "
                     f"{mu_l} (gcd here is 6, so 1/2/3/6 per axis).")
                return 2
        _log(f"  using n_q={n_q} of {geom['n_q_on_disk']}, "
             f"ngkmax {ngk} -> n_G {n_g} (zero pad, inert), "
             f"ratio μ_L/μ_S = {mu_l / mu_s:.2f}")
        _log(f"  global: ζ_L {n_q * mu_l * n_g * 16 / 1e9:.2f} GB, "
             f"ζ_S {n_q * mu_s * n_g * 16 / 1e9:.2f} GB, "
             f"W_L {n_q * mu_l * mu_l * 16 / 1e9:.2f} GB")

        t0 = time.perf_counter()
        zeta_L = _load_real_zeta(ldL, n_q, mu_l, n_g, mesh,
                                 P(None, None, ("x", "y")))
        zeta_S = _load_real_zeta(ldS, n_q, mu_s, n_g, mesh,
                                 P(None, None, ("x", "y")))
        jax.block_until_ready((zeta_L, zeta_S))
        _log(f"  [read+reshard] {time.perf_counter() - t0:.2f} s")

    f_host, g_host = _w_seeds(n_q, mu_l)
    W_L = _make_W_L(mesh, f_host, g_host)

    # --- the independent dense numpy reference FIRST, from the REAL ζ.
    # cond(G_S) on a real ISDF basis is the number the synthetic study
    # cannot supply, and it is also the thing most likely to make the
    # device Cholesky refuse — so it is computed and reported BEFORE any
    # device path can raise.
    zS = np.asarray(multihost_utils.process_allgather(zeta_S, tiled=True))
    zL = np.asarray(multihost_utils.process_allgather(zeta_L, tiled=True))
    W_full = np.asarray(multihost_utils.process_allgather(W_L, tiled=True))
    # BLAS-routed batched matmul, NOT np.einsum: numpy's einsum runs its
    # own C loop for this pattern (~1 GF/s), and O_ref alone is 4e11 flops
    # at these real sizes — job 7879528 was still grinding on it after two
    # minutes.  ``a @ swapaxes(b)`` dispatches zgemm.
    def _gram2(a, b):
        return np.conj(a) @ np.swapaxes(b, -1, -2)

    O_ref = _gram2(zS, zL)
    G_ref = _gram2(zS, zS)
    conds = [float(np.linalg.cond(G_ref[q])) for q in range(n_q)]
    cond = float(np.max(conds))
    ev = np.linalg.eigvalsh(G_ref[0])
    _log(f"  cond(G_S) per q: min {min(conds):.3e} max {cond:.3e}; "
         f"G_S[q=0] eigenvalues span {ev.min():.6e} .. {ev.max():.6e}")
    # The reference MUST apply the same truncation as the device path.
    # Comparing a rank-truncated solve against np.linalg.solve on a matrix
    # with cond 5.7e18 is meaningless — job 7879536 reported 9.996e-01 and
    # that was the REFERENCE being wrong, not the kernel.
    if args.mode == "ls_trunc":
        wv, Vv = np.linalg.eigh(G_ref)
        keepv = wv > args.rcond * wv.max(axis=-1, keepdims=True)
        wiv = np.where(keepv, 1.0 / np.where(keepv, wv, 1.0), 0.0)
        T_ref = ((Vv * wiv[:, None, :])
                 @ np.conj(np.swapaxes(Vv, -1, -2))) @ O_ref
        _log(f"  numpy reference uses the SAME rcond={args.rcond:g} "
             f"truncation (retained {int(keepv[0].sum())}/{mu_s} at q=0)")
    else:
        T_ref = np.linalg.solve(G_ref, O_ref)

    lsmode = args.mode if args.mode != "raw" else "ls"
    rc_kw = {"rcond": args.rcond} if lsmode == "ls_trunc" else {}
    out = {}
    for mode in (lsmode, "raw"):
        kw = rc_kw if mode == lsmode else {}
        try:
            psi_l, psi_r, _ = build_zeta_transfer(
                zeta_S, zeta_L, mesh, mode=mode, print_fn=_log,
                announce=(mode == "raw"), **kw)
        except ValueError as exc:                      # noqa: BLE001
            _log(f"  mode={mode}: build_zeta_transfer REFUSED — {exc}")
            _log(f"  (numpy cond(G_S)={cond:.3e}; a real ISDF small basis "
                 f"whose sphere-metric Gram is near-singular needs the "
                 f"rank-truncated route the FIT already has "
                 f"(charge_zeta_solve=rank_truncate), not a Cholesky.)")
            return 3
        W_S = jax.jit(project_w_between_zeta_bases(mesh))(W_L, psi_l, psi_r)
        out["ls" if mode == lsmode else "raw"] = np.asarray(
            multihost_utils.process_allgather(W_S, tiled=True))
        del psi_l, psi_r, W_S
    W_ref = T_ref @ W_full @ np.conj(np.swapaxes(T_ref, -1, -2))
    e_dense = _rel(out["ls"], W_ref)
    e_herm = _rel(out["ls"], np.conj(np.swapaxes(out["ls"], -1, -2)))

    # --- representability round-trip on the REAL ζ_L.
    # The small basis here is conj(A)·ζ_L for a random A: the point is
    # that ζ_L is REAL, so the transfer, the Gram and the metric are all
    # built from production ζ, while the field is representable in the
    # small span BY CONSTRUCTION and the LS transfer must reproduce W_S0
    # exactly.  Using the true LS image of the real ζ_S would need G_L^-1
    # (μ_L², host-only) — deliberately not done: it is both expensive and
    # exactly the O(μ_L²) object this module exists to avoid.
    rng = np.random.default_rng(3)
    A = ((rng.standard_normal((n_q, mu_s, mu_l))
          + 1j * rng.standard_normal((n_q, mu_s, mu_l)))
         / np.sqrt(2.0 * mu_l))
    W0 = (rng.standard_normal((n_q, mu_s, mu_s))
          + 1j * rng.standard_normal((n_q, mu_s, mu_s)))
    W0 = 0.5 * (W0 + np.conj(np.swapaxes(W0, -1, -2)))
    W_emb = (np.swapaxes(A, -1, -2) @ W0 @ np.conj(A))
    W_emb_d = _put(W_emb, NamedSharding(mesh, P(None, "x", "y")))
    zS_rep = _put(np.conj(A) @ zL,
                  NamedSharding(mesh, P(None, None, ("x", "y"))))
    rt = {}
    for mode in (lsmode, "raw"):
        kw = rc_kw if mode == lsmode else {}
        pl, pr, _ = build_zeta_transfer(zS_rep, zeta_L, mesh, mode=mode,
                                        print_fn=_log, announce=False, **kw)
        got = np.asarray(multihost_utils.process_allgather(
            jax.jit(project_w_between_zeta_bases(mesh))(W_emb_d, pl, pr),
            tiled=True))
        rt["ls" if mode == lsmode else "raw"] = _rel(got, W0)

    ratio = (float(np.max(np.abs(out["raw"])))
             / max(float(np.max(np.abs(out["ls"]))), 1e-300))
    _log(f"  REAL μ_L={mu_l} μ_S={mu_s} n_q={n_q} transfer={lsmode}"
         + (f" rcond={args.rcond:g}" if rc_kw else ""))
    _log(f"    cond(G_S) max over q        = {cond:.4e}")
    _log(f"    W_S vs dense numpy ref      = {e_dense:.3e}  (gate {args.tol:.0e})")
    _log(f"    W_S hermiticity             = {e_herm:.3e}  (gate {args.tol:.0e})")
    _log(f"    round-trip  mode=ls         = {rt['ls']:.3e}  (gate {args.tol:.0e})")
    _log(f"    round-trip  mode=raw        = {rt['raw']:.3e}  (must be >> 1e-3)")
    _log(f"    max|W_S raw| / max|W_S ls|  = {ratio:.4e}")
    # The representability round-trip is a GATE on synthetic ζ and a
    # DIAGNOSTIC on real ζ: its exactness argument needs the LARGE basis
    # well conditioned, and a real μ_L=2406 ISDF Gram is strongly
    # rank-deficient, so round-off in the degenerate directions is
    # amplified without bound.  Reported with the reason, not gated.
    _log("    (round-trip is a DIAGNOSTIC here, not a gate: the real LARGE "
         "basis is itself rank-deficient, so 'exactly representable' is "
         "not numerically meaningful.  On synthetic ζ, cond~20, the same "
         "quantity is 1.5e-15.)")
    ok = (e_dense < args.tol and e_herm < args.tol and rt["raw"] > 1e-3)

    _log(f"    {'PASS' if ok else 'FAIL'} [real]")
    rc = 0 if ok else 1

    del zeta_L, zeta_S, zS_rep
    if args.with_selection:
        sa = argparse.Namespace(**vars(args))
        sa.mu_l, sa.mu_s, sa.nq = mu_l, mu_s, n_q
        rc = cell_selection(mesh, sa) or rc
    return rc


_CELLS = {"dense": cell_dense, "selection": cell_selection,
          "roundtrip": cell_roundtrip, "scale": cell_scale,
          "provoke": cell_provoke,
          "real": cell_real}


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--cell", default="dense", choices=sorted(_CELLS))
    ap.add_argument("--mesh", default=None, help="PXxPY (default: square)")
    ap.add_argument("--nq", type=int, default=3)
    ap.add_argument("--mu-l", type=int, default=192)
    ap.add_argument("--mu-s", type=int, default=48)
    ap.add_argument("--ng", type=int, default=256)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--mode", default="ls",
                    choices=("ls", "ls_trunc", "raw"))
    ap.add_argument("--rcond", type=float, default=0.0,
                    help="relative eigenvalue threshold for mode=ls_trunc")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--with-selection", action="store_true")
    ap.add_argument("--zeta-small", default=None, help="small-basis zeta_q.h5")
    ap.add_argument("--zeta-large", default=None, help="large-basis zeta_q.h5")
    ap.add_argument("--mu-s-chunk", type=int, default=0,
                    help="force the overlap's mu_S chunk size (0 = auto). "
                         "The gate sizes auto-resolve to ONE chunk, so the "
                         "multi-chunk path needs this to be exercised.")
    args = ap.parse_args()

    if args.mesh is None:
        n = len(jax.devices())
        r = int(round(np.sqrt(n)))
        while r > 1 and n % r:
            r -= 1
        args.mesh = f"{n // r}x{r}"
    mesh = _mesh_from(args.mesh)
    _log(f"### zeta_projection_test cell={args.cell} mesh={args.mesh} "
         f"world={jax.process_count()} devices={len(jax.devices())} ###")
    # impl=mpi: the WORLD clique must be created by a real device collective
    # on THIS mesh before any grouped psum_scatter (job 7879482 — the
    # primitive's sync_global_devices-based helper is not sufficient).
    if ensure_world_clique_ready(mesh, print_fn=_log):
        _log("  [mpi] world-clique warm-up collective executed")
    ensure_grouped_collectives_ready(print_fn=_log)
    rc = _CELLS[args.cell](mesh, args)
    _log(f"### cell={args.cell} rc={rc} ###")
    return rc


if __name__ == "__main__":
    sys.exit(main())
