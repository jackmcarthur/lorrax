"""Compare the TWO W Dyson plans (local | distributed) on random V, chi0.

Synthetic inputs: V SPD, chi0 Hermitian negative-semidefinite (like static
response), built at a LOGICAL extent n_log and zero-padded to n_ext so the
μ-pad path is LIVE on both plans (local: solve_at_logical slice/refill;
distributed: identity-embedded block-diagonal system + pad mask).

Gates (all must pass):
  1. rel |W_local - W_dist| < 1e-10
  2. Dyson residual |(1 - V·pref·chi0)·W - V| / |V| < 1e-10 for BOTH plans
     (the strict contract — a block-cyclic LU is not bit-comparable to the
     per-q LU, so the residual is what certifies the solve)
  3. W pad rows/cols are EXACTLY zero on both plans

Usage (CPU mesh, e.g. via the shared holder)::

    alloc_run.sh 2 2 <src> <wd> python -u tests/bench/w_solve_modes_test.py

or on GPU::

    lxalloc
    LORRAX_MPI_TYPE=pmix lxrun python3 -u tests/bench/w_solve_modes_test.py
"""
from __future__ import annotations
import os
import sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _init():
    if os.environ.get(_DIST):
        return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        one_gpu = cvd and "," not in cvd
        init_kwargs = {"local_device_ids": [0]} if one_gpu else {}
        try:
            jax.distributed.initialize(**init_kwargs)
        except Exception:
            pass
    os.environ[_DIST] = "1"
_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from gw.w_isdf import solve_w, _w_solve_pref_scalar
from types import SimpleNamespace


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def main():
    nq = 3            # deliberately non-dividing (remainder q on any mesh)
    n_log = 61        # LOGICAL extent
    n_ext = 64        # PADDED extent — pads LIVE on both plans
    world = jax.process_count()
    if len(sys.argv) > 1:
        Px, Py = (int(t) for t in sys.argv[1].lower().split("x"))
        if Px * Py != world:
            _log(f"mesh {Px}x{Py} != world {world}")
            return 2
    elif world in (1, 4, 16):
        Px = int(np.sqrt(world)); Py = world // Px
    else:
        _log(f"world={world} unsupported (want 1/4/16 or an explicit "
             f"PXxPY argv)")
        return 2

    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py),
                axis_names=("x", "y"))

    if Px != Py and Px > 1 and Py > 1:
        # Rectangular 2-D mesh: ScaLAPACK pXgetrf needs square descriptor
        # blocks, so the DISTRIBUTED plan must REFUSE at resolve time with
        # the resolver's own message — never silently downgrade (quality
        # pattern #6/#8).  This cell gates exactly that behaviour.
        from gw.w_isdf import _get_w_solve_fn_distributed
        try:
            _get_w_solve_fn_distributed(mesh, nq, n_ext, n_log)
        except (ValueError, RuntimeError) as exc:
            _log(f"mesh {Px}x{Py}: distributed plan refused at resolve "
                 f"time (EXPECTED):\n    {exc}")
            _log("PASS")
            return 0
        _log(f"mesh {Px}x{Py}: distributed plan did NOT refuse — "
             f"geometry guard missing")
        _log("FAIL")
        return 1
    sh = NamedSharding(mesh, P(None, "x", "y"))
    dtype = jnp.complex128

    @jax.jit
    def make():
        kv, kc = jax.random.split(jax.random.key(42))
        Vr = jax.random.normal(kv, (nq, n_log, n_log), dtype=jnp.float64)
        Vi = jax.random.normal(jax.random.key(43), (nq, n_log, n_log),
                               dtype=jnp.float64)
        V = (Vr + 1j * Vi).astype(dtype)
        V = V @ jnp.conj(jnp.swapaxes(V, -1, -2)) \
            + (n_log * 2.0) * jnp.eye(n_log, dtype=dtype)[None]
        V = 0.5 * (V + jnp.conj(jnp.swapaxes(V, -1, -2)))
        Cr = jax.random.normal(kc, (nq, n_log, n_log), dtype=jnp.float64)
        Ci = jax.random.normal(jax.random.key(44), (nq, n_log, n_log),
                               dtype=jnp.float64)
        C = (Cr + 1j * Ci).astype(dtype)
        C = C @ jnp.conj(jnp.swapaxes(C, -1, -2))
        C = 0.5 * (C + jnp.conj(jnp.swapaxes(C, -1, -2)))
        chi0 = -0.01 * C
        # Zero-pad to the padded extent — the module's exact-zero pad
        # contract (bilinear-in-zero-padded-ψ).
        pad = ((0, 0), (0, n_ext - n_log), (0, n_ext - n_log))
        return (jax.lax.with_sharding_constraint(jnp.pad(V, pad), sh),
                jax.lax.with_sharding_constraint(jnp.pad(chi0, pad), sh))

    V_q, chi0_q = make()
    jax.block_until_ready(V_q)
    jax.block_until_ready(chi0_q)

    # n_rmu (the LOGICAL extent) is a HARD read in _resolve_w_solve_fn.
    meta = SimpleNamespace(
        nk_tot=nq, nspin=1, nspinor=1, nspinor_wfnfile=1, n_rmu=n_log)
    pref = _w_solve_pref_scalar(meta)

    # solve_w consumes (donates) its χ₀ argument on both plans — pass an
    # independent fresh-buffer copy each time.
    _fresh = jax.jit(lambda x: x + 0)
    W_loc = solve_w(V_q, _fresh(chi0_q), meta, mesh, dyson_solver="local")
    jax.block_until_ready(W_loc)
    W_dst = solve_w(V_q, _fresh(chi0_q), meta, mesh,
                    dyson_solver="distributed")
    jax.block_until_ready(W_dst)

    V_full = np.asarray(multihost_utils.process_allgather(V_q, tiled=True))
    C_full = np.asarray(multihost_utils.process_allgather(chi0_q, tiled=True))
    Wl = np.asarray(multihost_utils.process_allgather(W_loc, tiled=True))
    Wd = np.asarray(multihost_utils.process_allgather(W_dst, tiled=True))

    if jax.process_index() != 0:
        return 0

    def _resid(W):
        A = np.eye(n_ext, dtype=np.complex128)[None] - V_full @ (pref * C_full)
        num = np.linalg.norm((A @ W - V_full).reshape(nq, -1), axis=1)
        den = np.linalg.norm(V_full.reshape(nq, -1), axis=1)
        return (num / den).max()

    rel = np.max(np.abs(Wl - Wd)) / max(np.max(np.abs(Wl)), 1.0)
    res_l, res_d = _resid(Wl), _resid(Wd)
    pad_l = max(np.abs(Wl[:, n_log:, :]).max(), np.abs(Wl[:, :, n_log:]).max())
    pad_d = max(np.abs(Wd[:, n_log:, :]).max(), np.abs(Wd[:, :, n_log:]).max())

    print(f"mesh {Px}x{Py}  nq={nq}  n_log={n_log}  n_ext={n_ext}")
    print(f"rel |W_local - W_dist|          = {rel:.3e}   (gate 1e-10)")
    print(f"Dyson residual local            = {res_l:.3e}   (gate 1e-10)")
    print(f"Dyson residual distributed      = {res_d:.3e}   (gate 1e-10)")
    print(f"max |W pad| local / distributed = {pad_l:.3e} / {pad_d:.3e} "
          f"(gate exact 0)")
    ok = (rel < 1e-10 and res_l < 1e-10 and res_d < 1e-10
          and pad_l == 0.0 and pad_d == 0.0)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
