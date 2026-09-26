"""``solve_bse_sharded(solver_kind="trlan")`` traces against the stack matvec's signature.

The thick-restart BSE route called ``matvec_ring(..., Vq0, MX, MY)`` after the
exchange pair amplitude became ONE operand ``M`` (9879efc5d), so ``--solver
trlan`` raised ``NameError`` at trace time and no cell noticed:
``tests/test_thick_restart_lanczos.py`` drives the generic solver only.

This cell runs the BSE seam end to end on a 1x1 mesh with a stand-in matvec
of the stack matvec's exact positional signature (checked against the source
below), so a wrong operand list fails at trace time here, and the answer is
checked against the stand-in's known spectrum.
"""
from __future__ import annotations

import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

import bse.bse_stack_matvec as SM
from bse.bse_lanczos import solve_bse_sharded

NC, NV, NKX, NKY, NKZ, NMU = 3, 2, 2, 2, 1, 2
NK = NKX * NKY * NKZ


def _required_positional(fn_name):
    """Required positional parameters of ``build_bse_stack_matvec``'s ``fn_name``."""
    tree = ast.parse(inspect.getsource(SM.build_bse_stack_matvec))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            args = node.args.args
            return [a.arg for a in args[:len(args) - len(node.args.defaults)]]
    raise AssertionError(f"{fn_name} not found in build_bse_stack_matvec")


def _stand_in_builder(*_a, **_k):
    # The stack matvec's D term only: H = diag(eps_c - eps_v), same layout.
    def matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M):
        return (eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]) * X
    return matvec


def test_the_stand_in_has_the_stack_matvecs_signature():
    assert _required_positional("_matvec") == list(
        inspect.signature(_stand_in_builder()).parameters)


def test_trlan_route_traces_and_finds_the_lowest_transitions(monkeypatch):
    monkeypatch.setattr(SM, "build_bse_stack_matvec", _stand_in_builder)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    rng = np.random.default_rng(20260926)
    eps_c = jnp.asarray(1.0 + np.sort(rng.uniform(0, 1, (NK, NC)), axis=1))
    eps_v = jnp.asarray(-np.sort(rng.uniform(0, 0.5, (NK, NV)), axis=1))
    z = lambda *s: jnp.zeros(s, jnp.complex128)
    data = {
        "n_cond_pad": NC, "n_val_pad": NV, "nkx": NKX, "nky": NKY, "nkz": NKZ,
        "psi_c_X": z(NK, NC, 1, NMU), "psi_c_Y": z(NK, NC, 1, NMU),
        "psi_v_X": z(NK, NV, 1, NMU), "psi_v_Y": z(NK, NV, 1, NMU),
        "eps_c": eps_c, "eps_v": eps_v, "V_q0": z(NMU, NMU),
        "W_q": z(NMU, NMU, NKX, NKY, NKZ), "M": z(NK, NC, NV, NMU),
    }
    n_eig = 3
    ev, vecs, apps = solve_bse_sharded(
        data, mesh, n_eig=n_eig, max_iter=40, include_W=False,
        solver_kind="trlan", trlan_m_max=12, trlan_n_keep=6)
    want = np.sort((np.asarray(eps_c)[:, :, None]
                    - np.asarray(eps_v)[:, None, :]).ravel())[:n_eig]
    np.testing.assert_allclose(np.asarray(ev).real, want, atol=1e-8)
    assert vecs.shape == (n_eig, 1, NC, NV, NK)
    assert int(apps) >= 12
