"""PROTOTYPE (track O9): the ladder rung's FFT chain through the EXISTING
fused cuFFT FFI handler, against today's XLA emission.

WHAT IS BEING TESTED.  The direct rung's chain is

    U_q = fftn_k( W_R * ifftn_k(T) ) ,      W_R = ifftn_k(W_q)

on a ``(nb, mu, nu, s, s, k)`` buffer, and the decode that follows wants it as
``(b, k, t, M, s, N)``.  XLA emits that as, measured on an A100 at the
gnppm geometry (n_rmu=399, nk=3x3x1, ns=2, nb=1):

    regular_fft + vector_fft + scal_kernel_val   (ifftn, k-minor)
    loop_multiply_fusion                          (W_R broadcast multiply)
    regular_fft + vector_fft                      (fftn)
    loop_transpose_fusion                         (the decode-layout transpose)

LORRAX already ships a CUDA handler that does the first three of those in one
call with no extra chain-sized buffer: ``lorrax_mklfft_gw_conv``
(``src/ffi/cpp/cufft/fft_flat_k_cuda_ffi.cc:675``), exposed as
``ffi.fft.make_gw_conv_ffi``, computing

    S = FFT[ IFFT[G] . IFFT[W](bcast) . mult ]

with ``G = (nk, a, mx, b, my)`` and ``W = (nk, mx, my)``.  Map
``a -> t, mx -> mu, b -> s, my -> nu`` and that IS the rung's chain, with
``W = W_q`` (the handler does the ``W_R`` inverse transform itself, so
``ensure_W_R`` becomes unnecessary on this path).

The layout is the second half of the intervention and is why this is not a
drop-in: the handler is k-MAJOR (nk is the leading axis) and the rung today is
k-MINOR.  But the decode's canonical operand layout is ``(b, k, t, M, s, N)``
(``contract_bands_block_reshard(extra="leading")``), so a per-trial k-major
chain buffer ``(k, t, mu, s, nu)`` comes out of the handler EXACTLY in the
decode's layout — the transpose is not moved, it is deleted.  Same argument the
GW flat-k FFI made ("k-major layout end to end, never reshaped to the 3-D
k-minor form, which is the whole point", ``ffi/fft.py:302``).

This module builds both spellings of the chain, checks they agree, and times
them.  It changes no production file.

    python tools/rung_chain_proto.py --nb 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[1] / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from runtime import bootstrap  # noqa: E402
bootstrap()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

jax.config.update("jax_enable_x64", True)


def build_xla_chain(mesh, nkx, nky, nkz, sh):
    """Today's emission: 6D->8D reshape, ifftn, W_R multiply, fftn, transpose
    into the decode's ``(b, k, t, M, s, N)`` layout.  Lifted verbatim from
    ``bse_ring_comm._apply_W_from_T`` (:540-577, :881-889) minus the decode."""
    from common.fft_helpers import make_sharded_fftn_3d, make_sharded_ifftn_3d
    spec8 = P(None, "x", "y", None, None, None, None, None)
    ifftn = make_sharded_ifftn_3d(mesh, spec8, spec8, axes=(5, 6, 7), norm="ortho")
    fftn = make_sharded_fftn_3d(mesh, spec8, spec8, axes=(5, 6, 7), norm="ortho")

    def _chain(T, W_R):
        nb, M, N, t, s, nk = T.shape
        T_k = T.reshape(nb, M, N, t, s, nkx, nky, nkz)
        T_R = ifftn(T_k)
        U_R = W_R[None, :, :, None, None, :, :, :] * T_R
        U_q = fftn(U_R)
        U = U_q.reshape(nb, M, N, t, s, nk)
        return jnp.transpose(U, (0, 5, 3, 1, 4, 2))     # (b, k, t, M, s, N)

    return jax.jit(_chain, in_shardings=(sh.T, sh.W), out_shardings=None)


def build_ffi_chain(mesh, nkx, nky, nkz):
    """The fused cuFFT handler on a k-MAJOR per-trial buffer.

    Input  T_km : (nb, nk, t, mu, s, nu)   -- what a k-leading encode emits
    Input  W_q  : (nk, mu, nu)             -- the RECIPROCAL-space tile
    Output      : (nb, nk, t, mu, s, nu)   -- the decode's layout, no transpose
    """
    from ffi.fft import make_gw_conv_ffi
    g_spec = P(None, None, None, None, "x", None, "y")   # 3-D form: k,k,k + trail
    v_spec = P(None, None, None, "x", "y")
    conv = make_gw_conv_ffi(mesh, (nkx, nky, nkz), g_spec, v_spec,
                            norm="ortho", mult=1.0)

    def _chain(T_km, W_q):
        # one handler call per trial vector: the handler's G is rank 5
        # (nk, a, mx, b, my) and the trial axis is a fifth free index.
        if T_km.shape[0] == 1:
            return conv(T_km[0], W_q)[None]
        outs = [conv(T_km[i], W_q) for i in range(T_km.shape[0])]
        return jnp.stack(outs, axis=0)

    return jax.jit(_chain)


def build_ffi_chain_merged(mesh, nkx, nky, nkz):
    """One handler call for the WHOLE block: the trial axis is merged into the
    handler's free ``a`` slot, ``a = nb * t``.

    W broadcasts over ``a`` and ``b``, so any axes that sit before ``mx`` and
    between ``mx`` and ``my`` are free -- the trial index is one of them.  The
    price is that the result is ``(k, nb, t, mu, s, nu)`` and the decode's
    ``extra='leading'`` layout is ``(nb, k, t, mu, s, nu)``, so ONE axis swap
    remains (against three chain-sized passes removed).  ``extra='minor'``
    would not help: it puts the stack axis last, and takes the XLA plan.
    """
    from ffi.fft import make_gw_conv_ffi
    g_spec = P(None, None, None, None, "x", None, "y")
    v_spec = P(None, None, None, "x", "y")
    conv = make_gw_conv_ffi(mesh, (nkx, nky, nkz), g_spec, v_spec,
                            norm="ortho", mult=1.0)

    def _chain(T_kb, W_q):
        nk, nb, t, M, s, N = T_kb.shape
        G = T_kb.reshape(nk, nb * t, M, s, N)
        S = conv(G, W_q).reshape(nk, nb, t, M, s, N)
        return jnp.swapaxes(S, 0, 1)             # (nb, k, t, M, s, N)

    return jax.jit(_chain)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmu", type=int, default=399)
    ap.add_argument("--nkx", type=int, default=3)
    ap.add_argument("--nky", type=int, default=3)
    ap.add_argument("--nkz", type=int, default=1)
    ap.add_argument("--ns", type=int, default=2)
    ap.add_argument("--nb", type=int, default=1)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=4)
    a = ap.parse_args()

    from bse.bse_ring_comm import make_bse_shardings
    nk = a.nkx * a.nky * a.nkz
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    sh = make_bse_shardings(mesh)
    print("[env] jax=%s dev=%s" % (jax.__version__, jax.devices()))
    print("[geom] nb=%d n_rmu=%d nk=%d ns=%d | chain %.1f MiB | W_q %.1f MiB"
          % (a.nb, a.nmu, nk, a.ns,
             a.nb * a.nmu * a.nmu * a.ns * a.ns * nk * 16 / 2**20,
             a.nmu * a.nmu * nk * 16 / 2**20))

    rng = np.random.default_rng(3)
    T6 = (rng.standard_normal((a.nb, a.nmu, a.nmu, a.ns, a.ns, nk))
          + 1j * rng.standard_normal((a.nb, a.nmu, a.nmu, a.ns, a.ns, nk))) / 8.0
    Wq = (rng.standard_normal((a.nmu, a.nmu, nk))
          + 1j * rng.standard_normal((a.nmu, a.nmu, nk))) / 8.0
    Wq3 = Wq.reshape(a.nmu, a.nmu, a.nkx, a.nky, a.nkz)

    with mesh:
        T = jax.device_put(jnp.asarray(T6), sh.T)
        W_R = jax.device_put(
            jnp.asarray(np.fft.ifftn(Wq3, axes=(2, 3, 4), norm="ortho")), sh.W)
        # k-major per-trial operands for the FFI arm
        T_km = jax.device_put(jnp.asarray(
            np.transpose(T6, (0, 5, 3, 1, 4, 2))))       # (b,k,t,M,s,N)
        T_kb = jax.device_put(jnp.asarray(
            np.transpose(T6, (5, 0, 3, 1, 4, 2))))       # (k,b,t,M,s,N)
        W_q_km = jax.device_put(jnp.asarray(
            np.transpose(Wq, (2, 0, 1))))                # (k, mu, nu)

    f_xla = build_xla_chain(mesh, a.nkx, a.nky, a.nkz, sh)
    out_xla = f_xla(T, W_R)
    out_xla.block_until_ready()

    try:
        f_ffi = build_ffi_chain(mesh, a.nkx, a.nky, a.nkz)
        out_ffi = f_ffi(T_km, W_q_km)
        out_ffi.block_until_ready()
    except Exception as exc:
        print("[ffi] UNAVAILABLE: %s: %s" % (type(exc).__name__, str(exc)[:300]))
        return 2

    f_mrg = build_ffi_chain_merged(mesh, a.nkx, a.nky, a.nkz)
    out_mrg = f_mrg(T_kb, W_q_km)
    out_mrg.block_until_ready()

    ox, of, om = np.asarray(out_xla), np.asarray(out_ffi), np.asarray(out_mrg)
    rel = np.max(np.abs(ox - of)) / np.max(np.abs(ox))
    rel_m = np.max(np.abs(ox - om)) / np.max(np.abs(ox))
    print("[check] shapes %s vs %s | max rel diff = %.3e (per-trial) "
          "%.3e (merged)" % (ox.shape, of.shape, rel, rel_m))

    def _time(fn, args, tag):
        for _ in range(a.warmup):
            jax.block_until_ready(fn(*args))
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            ts.append((time.perf_counter() - t0) * 1e3)
        print("[time] %-10s min %8.3f ms  med %8.3f ms  reps %s"
              % (tag, min(ts), float(np.median(ts)),
                 " ".join("%.2f" % t for t in ts)))
        return min(ts)

    t_xla = _time(f_xla, (T, W_R), "xla")
    t_ffi = _time(f_ffi, (T_km, W_q_km), "ffi_trial")
    t_mrg = _time(f_mrg, (T_kb, W_q_km), "ffi_merged")
    print("[verdict] nb=%d chain: xla %.3f ms | ffi_trial %.3f ms (%.2fx) | "
          "ffi_merged %.3f ms (%.2fx)"
          % (a.nb, t_xla, t_ffi, t_xla / t_ffi, t_mrg, t_xla / t_mrg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
