"""The ζ back-solve's layout primitives (R4, 2026-09-23).

The back-solve keeps each whole-tile factor on the rank that owns its q
(``distrib_la.batch_layout``, once per channel, through
``isdf.core.zeta_factor_resident``) and moves only the right-hand side.
Route G applies the factor that way on its ``q`` layout
(``isdf.zeta_mubatch.ZetaG``); its value parity against a dense solve is
the route-G P4 gate (tests/multi_device/zeta_mubatch_p4.py).
The r-tile ``solve_zeta`` tiers this file used to compare are retired.

These cells pin the layout contract on a CPU 2x2 mesh.  Geometry is hostile
on purpose (TASTE 11): nq = 5 does not divide the 2x2 mesh (one pad row per
rank block), mu_log = 60 < mu_pad = 64.  Route receipt (TASTE 30): the
factor really sits in the batch layout, pivots with it.  Red twin (TASTE 21):
a face factor declared resident must refuse.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

pytestmark = pytest.mark.mesh(4)

_NQ, _NPAD, _NLOG, _NCOL = 5, 64, 60, 18


def _mesh(px: int, py: int) -> Mesh:
    devs = jax.devices()
    if len(devs) < px * py:
        pytest.skip(f"needs {px * py} devices, have {len(devs)}")
    return Mesh(np.array(devs[:px * py]).reshape(px, py), ("x", "y"))


def _operands(seed: int, *, indefinite: bool):
    """Padded (nq, 64, 64) Gram with a logical 60 block, and (nq, 64, 18) Z."""
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((_NQ, _NLOG, _NLOG + 12))
         + 1j * rng.standard_normal((_NQ, _NLOG, _NLOG + 12)))
    C = A @ np.conj(np.swapaxes(A, 1, 2)) + _NLOG * np.eye(_NLOG)[None]
    if indefinite:
        C = C - 1.3 * _NLOG * np.eye(_NLOG)[None]
    C = 0.5 * (C + np.conj(np.swapaxes(C, 1, 2)))
    Cp = np.zeros((_NQ, _NPAD, _NPAD), np.complex128)
    Cp[:, :_NLOG, :_NLOG] = C
    Z = np.zeros((_NQ, _NPAD, _NCOL), np.complex128)
    Z[:, :_NLOG] = (rng.standard_normal((_NQ, _NLOG, _NCOL))
                    + 1j * rng.standard_normal((_NQ, _NLOG, _NCOL)))
    return Cp, Z


@pytest.mark.parametrize("kind,vertex", [
    ("replicated_rank_truncate", 0), ("replicated_cholesky", 0), ("lu", 1)])
def test_factor_residency_puts_whole_q_tiles_on_their_owners(kind, vertex):
    """``zeta_factor_resident`` under ``local``: the factor (and the LU
    pivots) move to the q-local batch layout."""
    from isdf import core
    from distrib_la import is_batch_layout

    mesh = _mesh(2, 2)
    C, _ = _operands(3 if vertex else 1, indefinite=bool(vertex))
    Cd = jax.device_put(jnp.asarray(C), NamedSharding(mesh, P(None, "x", "y")))
    F = core.factor_c_q(Cd, mesh, vertex_mu_L=vertex, n_rmu_logical=_NLOG,
                        solver_kind=kind, zeta_rcond=1e-10)
    F, piv = F if vertex else (F, None)
    assert (piv is None) == (kind != "lu")
    L_res, piv_res = core.zeta_factor_resident(
        F, piv, mesh, solver_kind=kind)
    assert is_batch_layout(L_res, mesh), "route receipt: factor not q-local"
    assert L_res.shape[0] == -(-_NQ // 4) * 4, L_res.shape
    if piv is not None:
        assert is_batch_layout(piv_res, mesh)


def test_resident_declaration_refuses_a_face_operand():
    from distrib_la import local_batch

    mesh = _mesh(2, 2)
    face = NamedSharding(mesh, P(None, "x", "y"))
    A = jax.device_put(jnp.zeros((_NQ, 8, 8), jnp.complex128), face)
    run = local_batch(lambda a, b: a @ b, mesh, resident=(0,))
    with pytest.raises(ValueError, match="resident"):
        run(A, A)


def test_batch_layout_round_trips_rows_to_their_owners():
    """Rank x*Py+y must own rows [rank*Bp/P, (rank+1)*Bp/P) -- the map the
    q-parallel factor and the local-batch schedule both assume."""
    from distrib_la import batch_layout

    mesh = _mesh(2, 2)
    face = NamedSharding(mesh, P(None, "x", "y"))
    host = np.arange(_NQ * 8 * 6, dtype=np.float64).reshape(_NQ, 8, 6)
    out = batch_layout(jax.device_put(jnp.asarray(host), face), mesh)
    assert out.shape == (8, 8, 6)
    got = np.asarray(jax.device_get(out))
    assert np.array_equal(got[:_NQ], host) and not np.any(got[_NQ:])
    for shard in out.addressable_shards:
        rank = list(mesh.devices.flat).index(shard.device)
        lo = rank * 2
        assert np.array_equal(np.asarray(shard.data),
                              np.concatenate([host, np.zeros((3, 8, 6))])[lo:lo + 2])
    rep = jax.device_put(jnp.arange(_NQ * 3).reshape(_NQ, 3),
                         NamedSharding(mesh, P()))
    rows = batch_layout(rep, mesh)
    assert rows.shape == (8, 3) and np.array_equal(
        np.asarray(jax.device_get(rows))[:_NQ], np.arange(_NQ * 3).reshape(_NQ, 3))
