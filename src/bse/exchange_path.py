"""Exchange tiles V_Q along an exciton momentum path.

For each path point Q the finite-Q TDA kernel contracts the pair density
conj(ψ_c,k+Q) ψ_v,k with the exchange tile at tile momentum q = wrap(−Q)
(``bse.exciton_bands``, LORRAX convention: electron leg shifted by +Q).
:func:`exchange_tiles` builds the stack of those tiles, one row per solve:

* Γ (Q on an integer point): the production q = 0 tile with its rank-one
  head, ``data["V_q0"]``;
* ``ongrid``: the stored tile ``V_qmunu[wrap(−Q)]``, exact;
* ``refit``: a per-Q ζ refit contracted with the producer's Coulomb door
  (``bse.vq_interp.refit_vq``);
* ``interp``: the arbitrary-Q evaluator (``vq_interp.build_vq_evaluator``),
  with the mini-BZ cell-averaged head carried as a rank-three tensor when
  ``head_mbz`` (the LR G* column is dropped from the tile so the head is not
  counted twice).

Every tile is Hermitised, 0.5 (V + V†); the anti-Hermitian residue is fit or
stencil noise.  After the path rows come the certification twins (the stored
tile at each ``cert_idx``) and the ``--vq-mode=both`` refit spot checks, in
that order; ``exciton_bands`` indexes its solve rows off that layout.

Placement.  The stored tiles are committed device arrays from the restart
reader (``V_q0`` from ``read_interaction``; the ``(μ, ν, nkx, nky, nkz)``
tensor at ``P('x','y',None,None,None)``), so ``jax.device_put`` onto
``P('x','y')`` is a reshard with no host copy and no equality all-gather.
The refit tiles are host arrays, identical on every rank, and go through
``device_put_process_local``.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local

__all__ = ["exchange_tiles"]


def _hermitize(V):
    return 0.5 * (V + jnp.conj(V).T)


def _tile_index(q_tile, kgrid):
    """Grid index of a tile momentum on the exchange-tile grid."""
    return tuple(int(i) for i in
                 np.round(q_tile * kgrid).astype(int) % kgrid)


def _host_tile(V_np, n_rmu, n_rmu_pad, grid_xy):
    """A replicated host tile, Hermitised on the logical block and placed."""
    V_pad = np.zeros((n_rmu_pad, n_rmu_pad), dtype=np.complex128)
    V_pad[:n_rmu, :n_rmu] = 0.5 * (V_np[:n_rmu, :n_rmu]
                                   + V_np[:n_rmu, :n_rmu].conj().T)
    return device_put_process_local(V_pad, grid_xy)


def exchange_tiles(Qpath, *, route, mesh_xy, V_q0, V_ongrid, kgrid_vq,
                   n_rmu, n_rmu_pad, refit=None, interp=None, head_mbz=False,
                   alpha=None, cert_idx=(), both_refit=None, both_idx=(),
                   log_fn=print):
    """``(V_stack, M_stack, head_scalars, n_eval_calls)`` for the path.

    ``Qpath`` is ``(nQ, 3)`` fractional.  ``route`` is ``"ongrid"``,
    ``"refit"`` or ``"interp"``.  ``refit`` is ``(zx_fit, rst)`` for the
    refit route; ``interp`` is ``(zx, prep, eval_vq, pinvF, coeffs_packed)``.
    ``both_refit`` is ``(zx, rst)`` for the ``--vq-mode=both`` spot checks
    at ``both_idx``.  ``V_stack`` is ``(n_solve, n_rmu_pad, n_rmu_pad)`` at
    ``P(None,'x','y')``; ``M_stack`` is the per-row head moment
    ``(n_solve, 3, 3)`` when ``head_mbz``, else ``None``.
    """
    from . import vq_interp

    grid_xy = NamedSharding(mesh_xy, P("x", "y"))
    nQ = int(np.shape(Qpath)[0])
    kgrid_vq = np.asarray(kgrid_vq, dtype=np.int64)
    V_rows = []
    # Per-Q cell moment for the tensor head.  Zero for Γ and for every Q off
    # the mini-BZ arm, which makes the head term an exact no-op there.
    M_rows = np.zeros((nQ, 3, 3), dtype=np.float64)
    head_scalars = []
    v_gamma = jax.device_put(V_q0, grid_xy)
    n_eval_calls = 0
    for iQ in range(nQ):
        Qw = Qpath[iQ] - np.round(Qpath[iQ])
        if np.linalg.norm(Qw) < 1e-9:
            V_rows.append(_hermitize(v_gamma))       # production q=0 tile
            continue
        q_tile_np = -Qpath[iQ] - np.round(-Qpath[iQ])   # wrap(−Q)
        n_eval_calls += 1
        if route == "ongrid":
            ix, iy, iz = _tile_index(q_tile_np, kgrid_vq)
            V_rows.append(_hermitize(
                jax.device_put(V_ongrid[:, :, ix, iy, iz], grid_xy)))
        elif route == "refit":
            zx_fit, rst = refit
            V_rows.append(_host_tile(
                vq_interp.refit_vq(zx_fit, rst, q_tile_np, mesh_xy,
                                   log_fn=log_fn), n_rmu, n_rmu_pad, grid_xy))
        else:
            zx, prep, eval_vq, pinvF, coeffs_packed = interp
            q_tile = jnp.asarray(q_tile_np)
            if head_mbz:
                gstar, head_val, M_ab = vq_interp.minibz_head_vlr(
                    zx, prep, q_tile_np, alpha=alpha, moment=True)
                M_rows[iQ] = M_ab
                V_rows.append(_hermitize(eval_vq(
                    q_tile, prep["V_SRc"], pinvF, coeffs_packed,
                    jnp.asarray(0.0, dtype=jnp.float64),
                    jnp.asarray(gstar, dtype=jnp.int32))))
                head_scalars.append(
                    (iQ, float(head_val), float(np.trace(M_ab))))
            else:
                V_rows.append(_hermitize(eval_vq(
                    q_tile, prep["V_SRc"], pinvF, coeffs_packed)))
    # Certification twins: the producer's stored tile at each cert Q, so the
    # eigenvalue difference of the two solve rows is the exchange alone.
    for iQ in cert_idx:
        ix, iy, iz = _tile_index(-Qpath[iQ] - np.round(-Qpath[iQ]), kgrid_vq)
        V_rows.append(_hermitize(
            jax.device_put(V_ongrid[:, :, ix, iy, iz], grid_xy)))
    # --vq-mode=both spot checks: the refit ground truth at both_idx.
    for iQ in both_idx:
        zx, rst = both_refit
        V_np = vq_interp.refit_vq(zx, rst, -Qpath[iQ], mesh_xy)
        V_rows.append(_host_tile(V_np, n_rmu, n_rmu_pad, grid_xy))
    V_stack = jax.device_put(jnp.stack(V_rows),
                             NamedSharding(mesh_xy, P(None, "x", "y")))
    M_stack = None
    if head_mbz:
        # Twin and spot-check rows carry no head tensor (they are point-value
        # comparisons); zero is an exact no-op in the term.
        M_stack = np.concatenate(
            [M_rows, np.zeros((len(cert_idx) + len(both_idx), 3, 3))], axis=0)
    return V_stack, M_stack, head_scalars, n_eval_calls
