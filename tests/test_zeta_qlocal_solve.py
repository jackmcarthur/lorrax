"""Parity gate for the ``local`` ζ back-solve tier (R4, 2026-09-23).

The ``local`` tier keeps each whole-tile factor on the rank that owns its q
(``distrib_la.batch_layout``, once per channel) and moves only each r-chunk's
right-hand side face -> batch -> face (``distrib_la.local_batch``).  It
replaced ``per_q``, which all-gathered one (mu, mu) factor per q per r-chunk:
per rank ``nq*mu^2*16*(1+1/Py)`` B every chunk, 29.5 GB at the VI3 12x12 P16
shape.  The per-q arithmetic is unchanged, so the gate is value parity at
the ULP floor against the retained ``replicated`` tier, which runs the SAME
per-q logical-extent kernel the retired ``per_q`` tier ran at batch 1
(its docstring: "identical shapes, identical operand values, therefore
bit-identical arithmetic"), plus an independent NumPy oracle.

Parity class: value-level, 1e-12 relative (TASTE 15).  Observables: zeta and
a V_q-shaped contraction ``V_q = zeta_q W zeta_q^H`` with a fixed Hermitian
positive ``W`` (the Coulomb contraction's shape; the flat-k FFT that forms
the real V_q is a multi-process gate, see tests/conftest.py).  Geometry is
hostile on purpose (TASTE 11): nq = 5 does not divide the 2x2 mesh (one pad
row per rank block), mu_log = 60 < mu_pad = 64, and the RHS width is not a
multiple of P.

Route receipt (TASTE 30): every cell asserts the factor really sits in the
batch layout and that the ``solve_local`` kernel was built.  Red twins
(TASTE 21): a resident factor with its q rows rolled by one must miss the
oracle by far more than the tolerance, and a face factor declared resident
must refuse.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

pytestmark = pytest.mark.mesh(4)

_NQ, _NPAD, _NLOG, _NCOL = 5, 64, 60, 18
_TOL = 1.0e-12


def _mesh(px: int, py: int) -> Mesh:
    devs = jax.devices()
    if len(devs) < px * py:
        pytest.skip(f"needs {px * py} devices, have {len(devs)}")
    return Mesh(np.array(devs[:px * py]).reshape(px, py), ("x", "y"))


def _rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


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


def _vq(zeta):
    """V_q-shaped contraction zeta W zeta^H with a fixed Hermitian W > 0."""
    rng = np.random.default_rng(7)
    B = rng.standard_normal((_NCOL, _NCOL)) + 1j * rng.standard_normal((_NCOL, _NCOL))
    W = B @ np.conj(B.T) + _NCOL * np.eye(_NCOL)
    z = np.asarray(zeta)
    return np.einsum("qmr,rs,qns->qmn", z, W, np.conj(z))


def _solve_all(mesh, L, Z, *, kind, vertex, piv=None):
    from isdf import core

    face = NamedSharding(mesh, P(None, "x", "y"))
    Zd = jax.device_put(jnp.asarray(Z), face)
    ref = core.solve_zeta(L, Zd + 0, mesh, _NQ, vertex_mu_L=vertex,
                          solver_kind=kind, n_rmu_logical=_NLOG,
                          zeta_gather="replicated", lu_piv=piv)
    L_res, piv_res = core.zeta_factor_resident(
        L, piv, mesh, zeta_gather="local", solver_kind=kind)
    from distrib_la import is_batch_layout
    assert is_batch_layout(L_res, mesh), "route receipt: factor not q-local"
    assert L_res.shape[0] == -(-_NQ // 4) * 4, L_res.shape
    if piv is not None:
        assert is_batch_layout(piv_res, mesh)
    core._solve_cache.clear()
    loc = core.solve_zeta(L_res, Zd + 0, mesh, _NQ, vertex_mu_L=vertex,
                          solver_kind=kind, n_rmu_logical=_NLOG,
                          zeta_gather="local", lu_piv=piv_res)
    assert any(k[0] == "solve_local" for k in core._solve_cache
               if isinstance(k, tuple)), "route receipt: local kernel not built"
    # The same tier with a face (non-resident) factor: moved per call.
    loc_face = core.solve_zeta(L, Zd + 0, mesh, _NQ, vertex_mu_L=vertex,
                               solver_kind=kind, n_rmu_logical=_NLOG,
                               zeta_gather="local", lu_piv=piv)
    assert loc.sharding.spec == P(None, ("x", "y"), None), loc.sharding
    r, lf = np.asarray(jax.device_get(ref)), np.asarray(jax.device_get(loc))
    print(f"[qlocal parity] kind={kind} platform={jax.devices()[0].platform} "
          f"zeta rel={_rel(lf, r):.3e} "
          f"face-factor rel={_rel(np.asarray(jax.device_get(loc_face)), r):.3e} "
          f"V_q rel={_rel(_vq(lf), _vq(r)):.3e} bitwise={bool(np.array_equal(lf, r))}")
    return (np.asarray(jax.device_get(ref)), np.asarray(jax.device_get(loc)),
            np.asarray(jax.device_get(loc_face)), L_res, piv_res)


def test_charge_rank_truncate_local_matches_replicated_and_oracle():
    from isdf import core

    mesh = _mesh(2, 2)
    C, Z = _operands(1, indefinite=False)
    Cd = jax.device_put(jnp.asarray(C), NamedSharding(mesh, P(None, "x", "y")))
    kind = "replicated_rank_truncate"
    Bf = core.factor_c_q(Cd, mesh, vertex_mu_L=0, n_rmu_logical=_NLOG,
                         solver_kind=kind, zeta_rcond=1e-10)
    ref, loc, loc_face, L_res, _ = _solve_all(mesh, Bf, Z, kind=kind, vertex=0)
    B = np.asarray(jax.device_get(Bf))[:, :_NLOG, :_NLOG]
    oracle = np.zeros_like(ref)
    oracle[:, :_NLOG] = B @ (np.conj(np.swapaxes(B, 1, 2)) @ Z[:, :_NLOG])
    assert _rel(loc, ref) <= _TOL
    assert _rel(loc_face, ref) <= _TOL
    assert _rel(loc, oracle) <= 1e-11
    assert _rel(_vq(loc), _vq(ref)) <= _TOL
    assert np.all(loc[:, _NLOG:] == 0), "pad rows of zeta must stay exact zeros"

    # Red twin: the q -> owner map is load-bearing.  Rolling the resident
    # factor's rows by one q must be caught by the same comparison.
    rolled = jax.jit(lambda a: jnp.roll(a, 1, axis=0),
                     out_shardings=L_res.sharding)(L_res)
    face = NamedSharding(mesh, P(None, "x", "y"))
    bad = core.solve_zeta(rolled, jax.device_put(jnp.asarray(Z), face), mesh,
                          _NQ, solver_kind=kind, n_rmu_logical=_NLOG,
                          zeta_gather="local")
    assert _rel(np.asarray(jax.device_get(bad)), ref) > 1e-3


def test_charge_cholesky_local_matches_replicated():
    from isdf import core

    mesh = _mesh(2, 2)
    C, Z = _operands(2, indefinite=False)
    Cd = jax.device_put(jnp.asarray(C), NamedSharding(mesh, P(None, "x", "y")))
    kind = "replicated_cholesky"
    L = core.factor_c_q(Cd, mesh, vertex_mu_L=0, n_rmu_logical=_NLOG,
                        solver_kind=kind)
    ref, loc, loc_face, _, _ = _solve_all(mesh, L, Z, kind=kind, vertex=0)
    assert _rel(loc, ref) <= _TOL
    assert _rel(loc_face, ref) <= _TOL
    assert _rel(_vq(loc), _vq(ref)) <= _TOL


@pytest.mark.parametrize("kind", ["transverse_rank_truncate", "lu"])
def test_transverse_local_matches_replicated(kind):
    from isdf import core

    mesh = _mesh(2, 2)
    C, Z = _operands(3, indefinite=True)
    Cd = jax.device_put(jnp.asarray(C), NamedSharding(mesh, P(None, "x", "y")))
    F, piv = core.factor_c_q(Cd, mesh, vertex_mu_L=1, n_rmu_logical=_NLOG,
                             solver_kind=kind, transverse_zeta_rcond=1e-10)
    assert (piv is None) == (kind != "lu")
    ref, loc, loc_face, _, _ = _solve_all(mesh, F, Z, kind=kind, vertex=1,
                                          piv=piv)
    assert _rel(loc, ref) <= _TOL
    assert _rel(loc_face, ref) <= _TOL
    assert _rel(_vq(loc), _vq(ref)) <= _TOL


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
